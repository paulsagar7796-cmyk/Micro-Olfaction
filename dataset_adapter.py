"""
Adapter from the UCI gas sensor array drift dataset (data/, see fetch_data.py) to
the project's native `SpikeTrain`, so real sensor recordings can drive the
LIF / STDP layers in place of the simulated ingestion pipeline.

Feature modes (16 channels per feature group, one per physical MOx sensor):

    steady    16 ch  DR_norm                      steady-state resistance ratio
    kinetics  32 ch  DR_norm + EMAi0.1            + rising-transient speed
    full      48 ch  DR_norm + EMAi0.1 + EMAd0.1  + decaying-transient speed

Encoding, independently for each feature group g (sensor s, sample n):

    x_ngs = sign_g · feature_g,s                     sign flips EMAd (negative) so stronger = larger
    u_ngs = (x_ngs − lo_g) / (hi_g − lo_g), clipped to [0, 1]
    r_ngs = f_min + (f_max − f_min) · u_ngs          firing rate (Hz)

and every r_ngs becomes its own Poisson or regular spike train of fixed duration.

    normalization="batch"   lo_g, hi_g = global min / max of group g over all samples and
                            sensors of the *training* data (`scaler_bounds`), reused unchanged
                            for test data: keeps absolute magnitude (concentration) information.
    normalization="robust"  lo_g, hi_g = 5th / 95th percentile of group g over the training
                            data; values outside are clipped, so a few outlier samples cannot
                            squash everyone else toward 0.
    normalization="sample"  lo_g, hi_g = min / max over the 16 sensors of each sample:
                            keeps only the relative response pattern.

Either way each group gets its own scale, so large-valued features cannot wash out the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from snn_input import SpikeTrain

DATA_DIR = Path(__file__).resolve().parent / "data"
N_SENSORS = 16
FEATURE_MODES = {
    "steady": ("DR_norm",),
    "kinetics": ("DR_norm", "EMAi0.1"),
    "full": ("DR_norm", "EMAi0.1", "EMAd0.1"),
    "all": ("DR", "DR_norm", "EMAi0.001", "EMAi0.01", "EMAi0.1", "EMAd0.001", "EMAd0.01", "EMAd0.1"),  # 128 ch
}
# EMAd (decaying transients) are negative; flip them so a stronger response gives a larger value
FEATURE_SIGN = {"DR": 1.0, "DR_norm": 1.0, "EMAi0.001": 1.0, "EMAi0.01": 1.0, "EMAi0.1": 1.0,
                "EMAd0.001": -1.0, "EMAd0.01": -1.0, "EMAd0.1": -1.0}
# The dataset's class numbers (gas_id 1–6) have two namings in circulation. UCI's coding sentence gives
# "uci-text"; UCI's own per-batch count table, Vergara et al. (2012)-derived tables and Zhang & Zhang (2015)
# associate the gases with the sample counts that correspond to "count-table". The published results we
# compare against use "count-table", so it is the default. Chemical identity is not provable from public
# material; results are also reported by class number (see fetch_data.GAS_LABEL_NOTE).
GAS_NAMINGS = {
    "count-table": {1: "acetone", 2: "acetaldehyde", 3: "ethanol", 4: "ethylene", 5: "ammonia", 6: "toluene"},
    "uci-text": {1: "ethanol", 2: "ethylene", 3: "ammonia", 4: "acetaldehyde", 5: "acetone", 6: "toluene"},
}
GAS_NAMING = "count-table"
CLASS_NAMES = GAS_NAMINGS[GAS_NAMING]
GAS_ORDER = [CLASS_NAMES[i] for i in range(1, 7)]  # in class-number order


def gas_name(class_id: int) -> str:
    """Gas name of a class number (1–6) under the active naming."""
    return CLASS_NAMES[class_id]
ScalerBounds = dict[str, tuple[float, float]]  # feature group → (lo, hi) of sign-corrected values
ChannelAffine = tuple[np.ndarray, np.ndarray]  # per-channel (scale, shift): x → x·scale + shift, before scaling
NORMALIZATIONS = ("batch", "robust", "sample")
BOUND_PERCENTILES = {"batch": (0.0, 100.0), "robust": (5.0, 95.0)}  # for the fitted normalisations


@dataclass
class BatchData:
    batch_id: int
    features: pd.DataFrame        # (n, 128) all sensor features
    gas: np.ndarray               # (n,) ground-truth gas name
    concentration_ppmv: np.ndarray  # (n,)
    months: str = ""              # recording months covered by the batch, e.g. "3,4,8,9,10"

    def __len__(self) -> int:
        return len(self.gas)

    def feature_group(self, group: str) -> np.ndarray:
        """(n, 16) one feature across all sensors, sign-corrected (see FEATURE_SIGN)."""
        return FEATURE_SIGN[group] * self.features[group_columns(group)].to_numpy(dtype=float)

    def subset(self, indices: np.ndarray) -> BatchData:
        idx = np.asarray(indices)
        return BatchData(self.batch_id, self.features.iloc[idx].reset_index(drop=True),
                         self.gas[idx], self.concentration_ppmv[idx], self.months)

    def channel_matrix(self, feature_mode: str = "steady") -> np.ndarray:
        """(n, n_channels) sign-corrected raw values, in channel_labels(feature_mode) order."""
        return np.hstack([self.feature_group(g) for g in _groups(feature_mode)])

    def channel_moments(self, feature_mode: str = "steady") -> tuple[np.ndarray, np.ndarray]:
        """Per-channel mean and standard deviation over all samples (labels are not used)."""
        x = self.channel_matrix(feature_mode)
        return x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)

    def class_channel_moments(self, feature_mode: str = "steady",
                              classes: list[str] = GAS_ORDER) -> tuple[np.ndarray, np.ndarray]:
        """Per-class, per-channel mean and variance, each (n_classes, n_channels). Uses the labels."""
        x = self.channel_matrix(feature_mode)
        return (np.stack([x[self.gas == c].mean(axis=0) for c in classes]),
                np.stack([x[self.gas == c].var(axis=0) for c in classes]))

    def transformed_channels(self, feature_mode: str = "steady",
                             channel_affine: ChannelAffine | None = None) -> np.ndarray:
        """(n, n_channels) raw channel values after the optional per-channel transform."""
        x = self.channel_matrix(feature_mode)
        if channel_affine is not None:
            scale, shift = channel_affine
            x = x * scale + shift
        return x

    def fit_scaler_bounds(self, feature_mode: str = "steady", percentiles: tuple[float, float] = (0.0, 100.0),
                          channel_affine: ChannelAffine | None = None) -> ScalerBounds:
        """Per feature group (lo, hi) percentiles over all samples and sensors of this batch
        ((0, 100) = global min / max)."""
        groups = _groups(feature_mode)
        parts = np.split(self.transformed_channels(feature_mode, channel_affine), len(groups), axis=1)
        return {g: tuple(float(v) for v in np.percentile(x, percentiles)) for g, x in zip(groups, parts)}

    def encoded_features(self, feature_mode: str = "steady", normalization: str = "batch",
                         scaler_bounds: ScalerBounds | None = None,
                         channel_affine: ChannelAffine | None = None) -> np.ndarray:
        """(n, 16 × groups) features in [0, 1], each group normalised independently.

        With normalization="batch" or "robust", `scaler_bounds` (from the training data) must
        be given; values outside the bounds are clipped. `channel_affine` is applied to the raw
        channel values first (e.g. drift alignment or standardisation).
        """
        return scale_channels(self.transformed_channels(feature_mode, channel_affine), feature_mode,
                              normalization, scaler_bounds)


@dataclass
class EncodedSample:
    spike_train: SpikeTrain       # 16 / 32 / 48 channels depending on feature mode
    gas: str                      # ground-truth label
    concentration_ppmv: float
    rates_hz: np.ndarray          # (n_channels,) rates the spike train was drawn from


def _groups(feature_mode: str) -> tuple[str, ...]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {list(FEATURE_MODES)}")
    return FEATURE_MODES[feature_mode]


def group_columns(group: str) -> list[str]:
    return [f"{group}_{s}" for s in range(1, N_SENSORS + 1)]


def channel_labels(feature_mode: str = "steady") -> list[str]:
    """Channel names in encoding order, e.g. DR_norm_1 … DR_norm_16, EMAi0.1_1 …"""
    return [c for g in _groups(feature_mode) for c in group_columns(g)]


def n_channels(feature_mode: str = "steady") -> int:
    return N_SENSORS * len(_groups(feature_mode))


_cache: dict[Path, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _read(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data_dir not in _cache:
        features, targets = data_dir / "features.csv", data_dir / "targets.csv"
        if not features.exists() or not targets.exists():
            raise FileNotFoundError(f"{data_dir} has no features.csv/targets.csv; run fetch_data.py first")
        _cache[data_dir] = (pd.read_csv(features), pd.read_csv(targets))
    return _cache[data_dir]


def load_batch(batch_id: int = 1, data_dir: Path = DATA_DIR) -> BatchData:
    """All measurements of one batch (1–10), with gas label and concentration."""
    X, y = _read(Path(data_dir))
    mask = (y["batch"] == batch_id).to_numpy()
    if not mask.any():
        raise ValueError(f"batch {batch_id} not found (batches: {sorted(y['batch'].unique())})")
    return BatchData(batch_id, X[mask].reset_index(drop=True),
                     y.loc[mask, "gas_id"].map(CLASS_NAMES).to_numpy(dtype=object),
                     y.loc[mask, "concentration_ppmv"].to_numpy(dtype=float),
                     str(y.loc[mask, "months"].iloc[0]))


def normalize_across_channels(x: np.ndarray) -> np.ndarray:
    """Per-sample min-max scaling across channels to [0, 1]; flat samples map to 0."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    span = hi - lo
    return np.divide(x - lo, span, out=np.zeros_like(x), where=span > 0)


def scale_channels(x: np.ndarray, feature_mode: str, normalization: str,
                   scaler_bounds: ScalerBounds | None) -> np.ndarray:
    """(n, n_channels) channel values (channel_labels order) → [0, 1], each feature group separately."""
    groups = _groups(feature_mode)
    parts = np.split(np.atleast_2d(x), len(groups), axis=1)
    if normalization == "sample":
        return np.hstack([normalize_across_channels(p) for p in parts])
    if normalization not in BOUND_PERCENTILES:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    if scaler_bounds is None or any(g not in scaler_bounds for g in groups):
        raise ValueError(f"scaler_bounds must cover feature groups {list(groups)}")
    return np.hstack([scale_to_bounds(p, *scaler_bounds[g]) for g, p in zip(groups, parts)])


def balanced_moments(mu_c: np.ndarray, var_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced per-channel mean and variance from class-conditional ones (law of total
    variance, every class weighted equally): μ = mean_c μ_c, σ² = mean_c σ²_c + mean_c (μ_c − μ)²."""
    mu = mu_c.mean(axis=0)
    return mu, var_c.mean(axis=0) + ((mu_c - mu) ** 2).mean(axis=0)


def scale_to_bounds(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Min-max scale with fixed bounds, clipping values outside them (e.g. test data) to [0, 1]."""
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def rates_to_spike_train(rates_hz: np.ndarray, duration_s: float = 1.0, dt_s: float = 0.001,
                         mode: str = "poisson", rng: np.random.Generator | None = None,
                         labels: list[str] | None = None) -> SpikeTrain:
    """Spike raster for constant per-channel rates.

    poisson: independent Bernoulli(r·dt) per step (requires r·dt ≤ 1)
    regular: evenly spaced spikes, first at half a period (phase accumulator starting at 0.5)
    """
    rates = np.asarray(rates_hz, dtype=float)
    n_steps = int(round(duration_s / dt_s))
    if np.any(rates * dt_s > 1.0):
        raise ValueError("rate × dt exceeds 1 spike per step; reduce f_max or dt_s")
    if mode == "poisson":
        rng = rng or np.random.default_rng()
        raster = rng.random((n_steps, len(rates))) < rates * dt_s
    elif mode == "regular":
        phase = 0.5 + np.arange(1, n_steps + 1)[:, None] * rates * dt_s
        raster = np.diff(np.floor(np.vstack([np.full(len(rates), 0.5), phase])), axis=0) > 0
    else:
        raise ValueError("mode must be 'poisson' or 'regular'")
    return SpikeTrain(raster, dt_s, labels or [f"ch{i}" for i in range(len(rates))])


def encode_batch(batch: BatchData, duration_s: float = 1.0, f_max: float = 100.0, f_min: float = 0.0,
                 mode: str = "poisson", dt_s: float = 0.001, seed: int | None = 0,
                 feature_mode: str = "steady", normalization: str = "batch",
                 scaler_bounds: ScalerBounds | None = None,
                 channel_affine: ChannelAffine | None = None) -> tuple[list[EncodedSample], ScalerBounds | None]:
    """Rate-encode every sample into a SpikeTrain with n_channels(feature_mode) channels.

    With normalization="batch" (min / max) or "robust" (5th / 95th percentile), pass
    `scaler_bounds=None` for the training set (bounds are fitted on it and returned), then
    pass those bounds when encoding the test set so both share the same absolute scale.
    Returns (samples, scaler_bounds); bounds are None for normalization="sample".
    `channel_affine` (per-channel scale, shift) is applied to the raw channel values before
    any scaling, and before fitting bounds.
    """
    if normalization in BOUND_PERCENTILES and scaler_bounds is None:
        scaler_bounds = batch.fit_scaler_bounds(feature_mode, BOUND_PERCENTILES[normalization], channel_affine)
    rng = np.random.default_rng(seed)
    rates = f_min + (f_max - f_min) * batch.encoded_features(feature_mode, normalization, scaler_bounds,
                                                             channel_affine)
    labels = channel_labels(feature_mode)
    samples = [EncodedSample(rates_to_spike_train(r, duration_s, dt_s, mode, rng, labels), g, c, r)
               for r, g, c in zip(rates, batch.gas, batch.concentration_ppmv)]
    return samples, (scaler_bounds if normalization in BOUND_PERCENTILES else None)


def stratified_split(labels: np.ndarray, test_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Indices (train, test) with each class split test_fraction / (1 − test_fraction)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train, test = [], []
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_test = int(round(test_fraction * len(idx)))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.sort(train), np.sort(test)
