"""
Batch 1 benchmark: UCI gas sensor drift data → rate-coded SpikeTrain → LIFInputLayer →
STDPClassificationLayer → readout, evaluated on a stratified 80/20 split.

    python3 benchmark_batch1.py                            # steady features, defaults below
    python3 benchmark_batch1.py --feature-mode kinetics --readout spiking vote linear --seeds
                                                           # mean ± std over seeds 7 42 100 999 1234
    python3 benchmark_batch1.py --feature-mode kinetics --stdp-duration 1.0 --normalization-interval-ms 200
                                                           # 1 s presentations, weights normalised every 200 ms
    python3 benchmark_batch1.py --feature-mode full --balance --neurons 24 --mode regular

Defaults: LIF input threshold 7.5 mV above rest (lets weaker channels through; best on the drift
benchmark), 64 STDP neurons trained for 1 epoch on 0.2 s of each sample (stdp_exposure_sweep.py
showed 1 s presentations starve neurons however long STDP trains), per-sample normalisation,
random initialisation, spiking readout (when --readout spiking is used) with partial class
weighting (power 0.5).

Readouts (all read the frozen STDP layer; every one listed is evaluated, the first is primary):
    vote      neuron-label vote (hard labels)                                   unsupervised layer +
    soft      each neuron's spikes split across classes by its tuning           label statistics only
    template  cosine match to each class's mean population pattern
    linear    logistic regression on spike counts (diagnostic probe, scikit-learn)
    spiking   trained LIF output layer, one neuron per gas (readout_layer.py)

STDP training exposure: --epochs passes over the training set (fractions allowed), each
presenting the first --stdp-duration seconds of a sample; exposure = presentations ×
stdp-duration. Evaluation always uses whole samples (--duration).

STDP layer health (see layer_health), to diagnose over-training:
    dead         neurons that never fire on the training set
    top-3 share  fraction of all training-set spikes emitted by the 3 most active neurons
    weight PR    mean participation ratio of each neuron's weights, (Σw)² / (n·Σw²):
                 1 = spread evenly over its inputs, → 1/n as they polarise onto a few
    RF corr      mean pairwise correlation between neurons' weight vectors (receptive
                 fields); rises as receptive fields collapse onto the same pattern
    RF rank      effective number of distinct receptive fields (participation ratio of the
                 singular values² of the centred, normalised weight matrix)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from dataset_adapter import (FEATURE_MODES, GAS_ORDER, NORMALIZATIONS, balanced_moments, encode_batch, load_batch,
                             n_channels, stratified_split)
from lif_neurons import LIFInputLayer, LIFParams, one_to_one_weights
from readout_layer import SILENT, ReadoutParams, SpikingReadoutLayer
from snn_input import SpikeTrain
from stdp_layer import COUNT_READOUTS, UNASSIGNED, STDPClassificationLayer, STDPLayerOutput, STDPParams

DEFAULT_SEEDS = [7, 42, 100, 999, 1234]
PRED_COLS = GAS_ORDER + [SILENT]  # SILENT: the readout produced no spikes for the sample
READOUTS = {
    "vote": "neuron-label vote",
    "soft": "soft vote",
    "template": "template matching",
    "linear": "linear probe (logistic regression)",
    "spiking": "spiking readout (trained LIF output layer)",
}
HEALTH = {  # key → label used in reports
    "dead": "dead neurons",
    "top3_share": "top-3 share",
    "weight_pr": "weight PR",
    "rf_corr": "RF corr",
    "rf_rank": "RF rank",
}


@dataclass
class PreparedData:
    """Encoded, LIF-filtered spike trains for one seed's train/test split (full-batch order)."""
    labels: np.ndarray             # gas per sample
    train_idx: np.ndarray
    test_idx: np.ndarray
    filtered: list[SpikeTrain]     # LIF input layer output per sample
    raw_rates: np.ndarray          # (n_samples, n_in) encoder rates
    lif_rates: np.ndarray          # (n_samples, n_in) rates after the LIF input layer
    n_in: int
    init: np.ndarray | None        # normalised training features for --init data
    bounds: dict | None            # scaler bounds fitted on the training split (None for per-sample scaling)
    channel_stats: tuple[np.ndarray, np.ndarray] | None = None  # per-channel (μ, σ) of the training split
    class_channel_stats: tuple[np.ndarray, np.ndarray] | None = None  # per-class, per-channel (μ_c, σ²_c)
    channel_affine: tuple[np.ndarray, np.ndarray] | None = None  # transform applied before scaling, if any


@dataclass
class RunResult:
    seed: int
    readouts: list[str]            # evaluated readouts; the first is primary (confusion matrix)
    accuracy: dict[str, float]     # test accuracy per readout
    train_accuracy: dict[str, float]
    probe_input: float | None      # linear probe on the STDP layer's inputs (LIF rates), if "linear" ran
    ref_raw: float                 # nearest-centroid on encoder rates
    ref_lif: float                 # nearest-centroid after the LIF input layer
    confusion: np.ndarray          # (6, 7) primary readout; rows GAS_ORDER, columns PRED_COLS
    cross: np.ndarray              # (6, 6) share of spikes on each class's labelled neurons
    neurons_per_class: dict[str, int]
    health: dict[str, float]       # layer_health() of the trained STDP layer

    @property
    def primary(self) -> str:
        return self.readouts[0]


def rate_vector(st: SpikeTrain) -> np.ndarray:
    return st.raster.sum(axis=0) / st.duration_s


def nearest_centroid_accuracy(train_x: np.ndarray, train_y: np.ndarray,
                              test_x: np.ndarray, test_y: np.ndarray) -> float:
    """Cosine nearest-centroid on rate vectors: a reference for how separable the encoding is."""
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    classes = np.unique(train_y)
    centroids = unit(np.stack([unit(train_x)[train_y == c].mean(axis=0) for c in classes]))
    return float(np.mean(classes[np.argmax(unit(test_x) @ centroids.T, axis=1)] == test_y))


def balanced_epoch(train_idx: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One shuffled epoch in which every class is presented equally often.

    Each class contributes all of its training samples once, then is up-sampled with
    random repeats (with replacement) until it matches the largest class.
    """
    by_class = [train_idx[labels[train_idx] == c] for c in np.unique(labels[train_idx])]
    target = max(len(idx) for idx in by_class)
    epoch = np.concatenate([np.concatenate([idx, rng.choice(idx, size=target - len(idx), replace=True)])
                            for idx in by_class])
    return rng.permutation(epoch)


def stdp_presentations(train_idx: np.ndarray, labels: np.ndarray, epochs: float, balance: bool,
                       rng: np.random.Generator) -> np.ndarray:
    """Training-set indices in STDP presentation order: `epochs` shuffled passes (class-balanced
    with `balance`); a fractional remainder presents the first part of one more pass (2.5 → 2½)."""
    parts, remaining = [], epochs
    while remaining > 1e-9:
        epoch = balanced_epoch(train_idx, labels, rng) if balance else rng.permutation(train_idx)
        parts.append(epoch if remaining >= 1 else epoch[:int(round(remaining * len(epoch)))])
        remaining -= 1
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def print_matrix(title: str, matrix: np.ndarray, rows: list[str], cols: list[str], fmt: str) -> None:
    width = max(len(c) for c in cols) + 2
    print(f"\n{title}")
    print(" " * 14 + "".join(f"{c[:width - 2]:>{width}}" for c in cols))
    for name, row in zip(rows, matrix):
        print(f"{name:<14}" + "".join(f"{v:>{width}{fmt}}" for v in row))


def prepare_data(args: argparse.Namespace, seed: int, say=print, standardize_channels: bool | str = False) -> PreparedData:
    """Split, encode and LIF-filter Batch 1. Scaler bounds are fitted on the training set only.

    With standardize_channels (True or "pooled"), every raw channel is first standardised with the
    training split's per-channel mean and standard deviation, x → (x − μ₁) / σ₁, for training and
    held-out data alike. With "balanced", the class-balanced statistics are used instead (every gas
    weighted equally; see dataset_adapter.balanced_moments): x → (x − μ_bal) / √(σ²_bal + 1e-8).
    """
    batch = load_batch(1)
    labels = batch.gas
    train_idx, test_idx = stratified_split(labels, test_fraction=0.2, seed=seed)
    train_batch = batch.subset(train_idx)
    mu, sigma = train_batch.channel_moments(args.feature_mode)
    class_stats = train_batch.class_channel_moments(args.feature_mode)
    if standardize_channels == "balanced":
        mu_bal, var_bal = balanced_moments(*class_stats)
        inv = 1.0 / np.sqrt(var_bal + 1e-8)
        affine = (inv, -mu_bal * inv)
    elif standardize_channels in (True, "pooled"):
        affine = (1.0 / sigma, -mu / sigma)
    elif standardize_channels:
        raise ValueError('standardize_channels must be False, True, "pooled" or "balanced"')
    else:
        affine = None
    enc = dict(duration_s=args.duration, f_max=args.f_max, mode=args.mode, feature_mode=args.feature_mode,
               normalization=args.normalization, channel_affine=affine)
    train_samples, bounds = encode_batch(train_batch, seed=seed, **enc)
    test_samples, _ = encode_batch(batch.subset(test_idx), seed=seed + 1, scaler_bounds=bounds, **enc)
    samples = [None] * len(batch)  # back in original order, so indices address the full batch
    for idx, encoded in ((train_idx, train_samples), (test_idx, test_samples)):
        for i, s in zip(idx, encoded):
            samples[i] = s
    n_in = n_channels(args.feature_mode)
    say(f"Batch 1: {len(samples)} samples, {len(train_idx)} train / {len(test_idx)} test "
        f"({args.feature_mode}: {n_in} channels, {args.normalization} normalisation, {args.mode}, "
        f"{args.duration:g} s, f_max {args.f_max:g} Hz, seed {seed})")
    if bounds:
        say("Scaler bounds (train): " + ", ".join(f"{g} [{lo:.2f}, {hi:.2f}]" for g, (lo, hi) in bounds.items()))

    lif = make_input_layer(args, n_in)
    filtered = [lif.run(s.spike_train).spike_train for s in samples]
    raw_rates = np.stack([s.rates_hz for s in samples])
    lif_rates = np.stack([rate_vector(st) for st in filtered])
    say(f"LIF input layer: mean rate {raw_rates.mean():.1f} → {lif_rates.mean():.1f} Hz; "
        f"channels silenced {np.mean((raw_rates > 0) & (lif_rates == 0)):.0%} of active inputs")

    init = (train_batch.encoded_features(args.feature_mode, args.normalization, bounds, affine)
            if args.init == "data" else None)
    return PreparedData(labels, train_idx, test_idx, filtered, raw_rates, lif_rates, n_in, init, bounds,
                        (mu, sigma), class_stats, affine)


def make_stdp_layer(args: argparse.Namespace, data: PreparedData, seed: int) -> STDPClassificationLayer:
    return STDPClassificationLayer(n_inputs=data.n_in, n_neurons=args.neurons, seed=seed, init_samples=data.init,
                                   stdp=STDPParams(normalization_interval_ms=args.normalization_interval_ms))


def make_input_layer(args: argparse.Namespace, n_in: int) -> LIFInputLayer:
    """LIF input layer (one neuron per channel) with its threshold `--lif-threshold` mV above rest."""
    p = LIFParams()
    return LIFInputLayer(one_to_one_weights(n_in), LIFParams(v_th_mv=p.v_rest_mv + args.lif_threshold))


def train_stdp(layer: STDPClassificationLayer, data: PreparedData, order: np.ndarray,
               stdp_duration_s: float | None) -> None:
    """Unsupervised STDP: present each sample in `order`, its first `stdp_duration_s` seconds (None = whole)."""
    for i in order:
        layer.process(data.filtered[i], duration_s=stdp_duration_s, learn=True)


def layer_health(layer: STDPClassificationLayer, train_counts: np.ndarray) -> dict[str, float]:
    """Over-training diagnostics: starvation, winner dominance, weight polarisation, receptive-field collapse."""
    spikes = train_counts.sum(axis=0)                                   # per neuron, whole training set
    w = layer.w                                                         # (n_inputs, n_neurons), w ≥ 0
    pr = w.sum(axis=0) ** 2 / (w.shape[0] * np.maximum((w ** 2).sum(axis=0), 1e-12))
    centred = w - w.mean(axis=0, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=0, keepdims=True), 1e-12)
    corr = unit.T @ unit
    n = corr.shape[0]
    sv2 = np.linalg.svd(unit, compute_uv=False) ** 2
    return {
        "dead": float(np.sum(spikes == 0)),
        "top3_share": float(np.sort(spikes)[-3:].sum() / max(spikes.sum(), 1)),
        "weight_pr": float(pr.mean()),
        "rf_corr": float((corr.sum() - np.trace(corr)) / (n * (n - 1))),
        "rf_rank": float(sv2.sum() ** 2 / max((sv2 ** 2).sum(), 1e-12)),
    }


@dataclass
class Readouts:
    """Readouts fitted on a frozen STDP layer's training-set output. predict() never updates them."""
    layer: STDPClassificationLayer            # frozen; neuron labels / class responses set by fit_readouts
    spiking: SpikingReadoutLayer | None
    probe: object | None                      # scikit-learn LogisticRegression on spike counts

    def predict(self, name: str, outputs: list[STDPLayerOutput]) -> np.ndarray:
        """Predicted gas (or SILENT) per frozen STDP-layer output."""
        if name in COUNT_READOUTS:
            return np.array([self.layer.classify_counts(o.spike_counts, name) or SILENT for o in outputs])
        if name == "linear":
            return self.probe.predict(np.stack([o.spike_counts for o in outputs]))
        if name == "spiking":
            return self.spiking.predict([o.spike_train for o in outputs])
        raise ValueError(f"unknown readout {name!r}")


def fit_readouts(layer: STDPClassificationLayer, train_out: list[STDPLayerOutput], train_truth: np.ndarray,
                 args: argparse.Namespace, seed: int) -> Readouts:
    """Label the neurons and fit every readout in args.readout on the frozen layer's training output."""
    counts = np.stack([o.spike_counts for o in train_out])              # (n_train, n_neurons)
    layer.label_neurons(counts, train_truth)
    probe = spiking = None
    if "linear" in args.readout:
        from sklearn.linear_model import LogisticRegression
        probe = LogisticRegression(max_iter=1000).fit(counts, train_truth)
    if "spiking" in args.readout:
        params = ReadoutParams(class_weight=None if args.readout_class_weight == "none" else "balanced",
                               class_weight_power=args.readout_class_weight_power)
        spiking = SpikingReadoutLayer(layer.n_neurons, GAS_ORDER, params=params, seed=seed)
        spiking.fit([o.spike_train for o in train_out], train_truth)
    return Readouts(layer, spiking, probe)


def evaluate_layer(layer: STDPClassificationLayer, data: PreparedData, args: argparse.Namespace,
                   seed: int) -> RunResult:
    """Freeze the layer, fit each readout in args.readout on the training set, score on the test set."""
    # Frozen: learn=False applies no STDP and no θ updates. One pass per sample yields both the
    # output spike trains (spiking readout) and their counts (other readouts).
    train_out = [layer.process(data.filtered[i], learn=False) for i in data.train_idx]
    test_out = [layer.process(data.filtered[i], learn=False) for i in data.test_idx]
    train_counts = np.stack([o.spike_counts for o in train_out])       # (n_train, n_neurons)
    test_counts = np.stack([o.spike_counts for o in test_out])         # (n_test, n_neurons)
    truth, train_truth = data.labels[data.test_idx], data.labels[data.train_idx]
    readouts = fit_readouts(layer, train_out, train_truth, args, seed)
    per_class = {c: int(np.sum(layer.neuron_labels == c)) for c in GAS_ORDER + [UNASSIGNED]}
    predictions = {n: (readouts.predict(n, train_out), readouts.predict(n, test_out)) for n in args.readout}
    probe_input = None
    if "linear" in args.readout:
        # Diagnostic: the same probe on the layer's inputs shows how much linearly readable
        # information the STDP layer loses.
        from sklearn.linear_model import LogisticRegression
        input_probe = LogisticRegression(max_iter=1000).fit(data.lif_rates[data.train_idx], train_truth)
        probe_input = float(input_probe.score(data.lif_rates[data.test_idx], truth))

    primary_test = predictions[args.readout[0]][1]
    confusion = np.array([[np.sum((truth == t) & (primary_test == p)) for p in PRED_COLS] for t in GAS_ORDER])
    share = test_counts / np.maximum(test_counts.sum(axis=1, keepdims=True), 1)
    cross = np.array([[share[truth == t][:, layer.neuron_labels == c].sum(axis=1).mean()
                       if np.any(layer.neuron_labels == c) else 0.0 for c in GAS_ORDER] for t in GAS_ORDER])
    tr, te = data.train_idx, data.test_idx
    return RunResult(
        seed=seed,
        readouts=list(args.readout),
        accuracy={n: float(np.mean(p[1] == truth)) for n, p in predictions.items()},
        train_accuracy={n: float(np.mean(p[0] == train_truth)) for n, p in predictions.items()},
        probe_input=probe_input,
        ref_raw=nearest_centroid_accuracy(data.raw_rates[tr], train_truth, data.raw_rates[te], truth),
        ref_lif=nearest_centroid_accuracy(data.lif_rates[tr], train_truth, data.lif_rates[te], truth),
        confusion=confusion, cross=cross, neurons_per_class=per_class,
        health=layer_health(layer, train_counts),
    )


def run_benchmark(args: argparse.Namespace, seed: int, verbose: bool = True) -> RunResult:
    """One full run: prepare data, train STDP, evaluate readouts. `seed` drives all randomness."""
    t0 = time.time()
    say = print if verbose else (lambda *_, **__: None)
    data = prepare_data(args, seed, say)
    layer = make_stdp_layer(args, data, seed)
    order = stdp_presentations(data.train_idx, data.labels, args.epochs, args.balance, np.random.default_rng(seed))
    train_stdp(layer, data, order, args.stdp_duration)
    result = evaluate_layer(layer, data, args, seed)
    stdp_s = args.stdp_duration or args.duration
    norm = (f"every {args.normalization_interval_ms:g} ms" if args.normalization_interval_ms
            else "at the end of each presentation")
    say(f"STDP layer: {data.n_in} inputs → {args.neurons} neurons ({args.init} init), {args.epochs:g} "
        f"{'balanced' if args.balance else 'unbalanced'} epochs = {len(order)} presentations × {stdp_s:g} s "
        f"({len(order) * stdp_s:.0f} s of STDP exposure), weights normalised {norm}; "
        f"neurons per class {result.neurons_per_class}")
    if verbose:
        report(result)
        print(f"[{time.time() - t0:.0f} s]")
    return result


def health_line(h: dict[str, float]) -> str:
    return (f"{h['dead']:.1f} dead neurons, top-3 neurons emit {h['top3_share']:.0%} of spikes, "
            f"weight PR {h['weight_pr']:.2f}, RF corr {h['rf_corr']:.2f}, RF rank {h['rf_rank']:.1f}")


def report(r: RunResult, title_suffix: str = "") -> None:
    """Confusion matrix (primary readout), cross-activation, recall, top errors, accuracies, layer health."""
    cols = PRED_COLS if r.confusion[:, -1].any() else GAS_ORDER
    print_matrix(f"Confusion matrix, {READOUTS[r.primary]}{title_suffix} (rows = true gas, columns = predicted)",
                 r.confusion[:, :len(cols)], GAS_ORDER, cols, "d")
    print_matrix(f"Cross-activation{title_suffix} (rows = true gas, columns = share of spikes on neurons labelled as)",
                 r.cross, GAS_ORDER, GAS_ORDER, ".2f")
    recall = r.confusion.diagonal() / np.maximum(r.confusion.sum(axis=1), 1)
    print(f"\nPer-class recall ({r.primary}): " + ", ".join(f"{g} {x:.0%}" for g, x in zip(GAS_ORDER, recall)))
    errors = sorted(((r.confusion[i, j], GAS_ORDER[i], PRED_COLS[j]) for i in range(len(GAS_ORDER))
                     for j in range(len(PRED_COLS)) if j != i and r.confusion[i, j]), reverse=True)
    print("Most frequent misclassifications: " +
          (", ".join(f"{t} → {p} ({n})" for n, t, p in errors[:5]) if errors else "none"))
    print()
    for name in r.readouts:
        print(f"{READOUTS[name]:<44} test {r.accuracy[name]:6.1%}  (train {r.train_accuracy[name]:.1%})")
    if r.probe_input is not None:
        print(f"{'linear probe on the STDP inputs (LIF rates)':<44} test {r.probe_input:6.1%}")
    print(f"Reference, cosine nearest-centroid: encoder rates {r.ref_raw:.1%}, after LIF input layer {r.ref_lif:.1%}")
    print(f"STDP layer health: {health_line(r.health)}")


def summarize(results: list[RunResult]) -> None:
    """Per-seed table, mean ± std per readout, and primary-readout matrices pooled over seeds."""
    names = results[0].readouts
    probe_in = results[0].probe_input is not None
    header = "".join(f" {n:>9}" for n in names) + (f" {'probe in':>9}" if probe_in else "")
    print(f"\nTest accuracy per seed\n{'seed':>6}{header} {'NC raw':>8} {'NC LIF':>8} {'unassigned':>11}")
    for r in results:
        row = "".join(f" {r.accuracy[n]:>9.1%}" for n in names) + (f" {r.probe_input:>9.1%}" if probe_in else "")
        print(f"{r.seed:>6}{row} {r.ref_raw:>8.1%} {r.ref_lif:>8.1%} {r.neurons_per_class[UNASSIGNED]:>11}")

    def stat(values: list[float]) -> str:
        v = np.array(values)
        return f"{v.mean():6.1%} ± {v.std(ddof=1) if len(v) > 1 else 0.0:4.1%}"
    print(f"\nOver {len(results)} seeds (mean ± sample std){'':<17}test                train")
    for n in names:
        print(f"  {READOUTS[n]:<44} {stat([r.accuracy[n] for r in results])}   "
              f"{stat([r.train_accuracy[n] for r in results])}")
    if probe_in:
        print(f"  {'linear probe on the STDP inputs':<44} {stat([r.probe_input for r in results])}")
    print(f"  {'nearest-centroid, encoder rates':<44} {stat([r.ref_raw for r in results])}")
    print(f"  {'nearest-centroid, after LIF input layer':<44} {stat([r.ref_lif for r in results])}")
    print(f"  Unassigned STDP neurons: {np.mean([r.neurons_per_class[UNASSIGNED] for r in results]):.1f} on average")

    mean_of = lambda get: {k: float(np.mean([get(r)[k] for r in results])) for k in get(results[0])}
    pooled = RunResult(seed=-1, readouts=names, accuracy=mean_of(lambda r: r.accuracy),
                       train_accuracy=mean_of(lambda r: r.train_accuracy),
                       probe_input=float(np.mean([r.probe_input for r in results])) if probe_in else None,
                       ref_raw=float(np.mean([r.ref_raw for r in results])),
                       ref_lif=float(np.mean([r.ref_lif for r in results])),
                       confusion=sum(r.confusion for r in results),
                       cross=np.mean([r.cross for r in results], axis=0), neurons_per_class={},
                       health=mean_of(lambda r: r.health))
    report(pooled, title_suffix=f", pooled over {len(results)} seeds")


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by this benchmark and stdp_exposure_sweep.py."""
    parser.add_argument("--neurons", "--n-neurons", type=int, default=64, help="STDP layer size")
    parser.add_argument("--mode", choices=["poisson", "regular"], default="poisson")
    parser.add_argument("--feature-mode", choices=list(FEATURE_MODES), default="steady",
                        help="steady: DR_norm (16 ch); kinetics: + EMAi0.1 (32 ch); full: + EMAd0.1 (48 ch)")
    parser.add_argument("--normalization", choices=list(NORMALIZATIONS), default="sample",
                        help="batch: per-group min/max fitted on the training set (keeps magnitude); "
                             "robust: 5th/95th-percentile bounds with clipping; sample: per-sample relative pattern")
    parser.add_argument("--init", choices=["data", "random"], default="random",
                        help="data: STDP weights start from random training rows; random: uniform noise")
    parser.add_argument("--readout", choices=list(READOUTS), nargs="+", default=["vote"],
                        help="one or more readouts, all evaluated on the same trained layer; the first is "
                             "primary (confusion matrix). vote / soft / template: count-based; linear: "
                             "logistic-regression probe (needs scikit-learn); spiking: trained LIF output layer")
    parser.add_argument("--readout-class-weight", choices=["balanced", "none"], default="balanced",
                        help="spiking readout: weight training errors by inverse class frequency (balanced)")
    parser.add_argument("--readout-class-weight-power", type=float, default=0.5,
                        help="exponent on the inverse class frequency: 1 = fully balanced, 0 = unweighted")
    parser.add_argument("--normalization-interval-ms", type=float, default=None,
                        help="also normalise STDP weights every N ms of learning within a presentation "
                             "(default: only at its end). Not the same as --normalization (feature scaling)")
    parser.add_argument("--balance", action="store_true",
                        help="up-sample classes so each STDP epoch presents every gas equally often")
    parser.add_argument("--lif-threshold", type=float, default=7.5,
                        help="LIF input layer threshold in mV above the resting potential (default 7.5: −57.5 mV; "
                             "15, i.e. −50 mV, was the default before the drift optimisation runs)")
    parser.add_argument("--duration", type=float, default=1.0, help="seconds per sample (encoding and evaluation)")
    parser.add_argument("--f-max", type=float, default=100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_pipeline_args(parser)
    parser.add_argument("--epochs", type=float, default=1,
                        help="STDP training passes over the training set; fractions present the first part "
                             "of one more shuffled pass (e.g. 0.2)")
    parser.add_argument("--stdp-duration", type=float, default=0.2,
                        help="seconds of each training sample STDP learns from; evaluation always uses "
                             "whole samples (--duration)")
    parser.add_argument("--seed", type=int, default=42, help="seed for a single run")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help=f"run each seed and report mean ± std (no values: {' '.join(map(str, DEFAULT_SEEDS))})")
    args = parser.parse_args()
    args.readout = list(dict.fromkeys(args.readout))  # drop duplicates, keep order
    if args.epochs < 0:
        parser.error("--epochs must be ≥ 0")
    if not 0 < args.stdp_duration <= args.duration:
        parser.error("--stdp-duration must be > 0 and ≤ --duration")
    if args.normalization_interval_ms is not None and args.normalization_interval_ms <= 0:
        parser.error("--normalization-interval-ms must be > 0")

    if args.seeds is None:
        run_benchmark(args, args.seed, verbose=True)
        return

    seeds = args.seeds or DEFAULT_SEEDS
    t0 = time.time()
    print(f"Batch 1, {args.feature_mode} features, {args.normalization} normalisation, {args.init} init, "
          f"{'balanced' if args.balance else 'unbalanced'}, {args.neurons} neurons, STDP {args.epochs:g} epochs × "
          f"{args.stdp_duration:g} s, weight normalisation "
          f"{f'every {args.normalization_interval_ms:g} ms' if args.normalization_interval_ms else 'per presentation'}, "
          f"readouts {' '.join(args.readout)} (spiking class weight: {args.readout_class_weight}, power "
          f"{args.readout_class_weight_power:g}) — seeds {seeds}")
    results = []
    for seed in seeds:
        results.append(run_benchmark(args, seed, verbose=False))
        r = results[-1]
        print(f"  seed {seed:>5}: " + ", ".join(f"{n} {r.accuracy[n]:.1%}" for n in r.readouts) +
              f", {r.health['dead']:.0f} dead  [{time.time() - t0:.0f} s]", flush=True)
    summarize(results)


if __name__ == "__main__":
    main()
