"""
Supervised spiking readout: one LIF output neuron per class, fully connected to a
spiking layer (e.g. the STDPClassificationLayer), trained with an error-driven rule.

Why: the STDP layer learns a distributed code (each odor drives a characteristic
*combination* of neurons), which a per-neuron label vote cannot read but a trained
weighted sum can.

Output neurons
──────────────
Output neuron j is the LIF neuron of lif_neurons.py (same equations and exact
integration), driven by the input spike trains s_i(t) through signed weights
(w > 0 excitatory, w < 0 inhibitory):

    τ_m · dV_j/dt   = −(V_j − V_rest) + R_m · I_j
    τ_syn · dI_j/dt = −I_j + τ_syn · Σ_i w_ij · s_i(t)
    V_j ≥ V_th  →  spike, reset, refractory

Decision: ŷ = argmax_j n_j, where n_j is neuron j's spike count over the sample
window; if no output neuron fires at all the sample is "silent".

Learning rule
─────────────
With ν_i the input rate and ν_j the actual output rate over the sample window (Hz),
and a teacher rate ν*_j = f_target for the true class and 0 for all others:

    Δw_ij = η · ν_i · (ν*_j − ν_j)          (Widrow–Hoff / delta rule on spike rates)

i.e. pre-synaptic activity × (desired − actual) post-synaptic rate. ReSuMe
(Ponulak & Kasiński, 2010) is the spike-timing generalisation of this rule. Because
ν_j is the neuron's real spike output, training accounts for the threshold, leak and
refractoriness. Updates are averaged over mini-batches whose samples are simulated
in parallel, weights are clipped to ±w_max, and η decays linearly to 0 over the
epochs so the weights settle instead of oscillating around the solution.

Class weighting (class_weight="balanced"): each sample's error is scaled by its class's
weight

    w_c ∝ (n / (n_classes · n_c)) ** p        n_c = training samples of class c

scaled so the mean weight per training sample is 1 (the effective learning rate is
unchanged). p = class_weight_power: 1 is full inverse-frequency weighting (every class
drives the weights equally, scikit-learn's "balanced"), 0 is no weighting, and the
default 0.5 is in between: rare classes are up-weighted by the square root of their rarity.

Defaults were chosen by *training* accuracy on one Batch 1 split (seed 42).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lif_neurons import LIFInputLayer, LIFParams
from snn_input import SpikeTrain

SILENT = "silent"  # prediction when no output neuron fires


@dataclass
class ReadoutParams:
    target_rate_hz: float = 30.0   # teacher rate of the true class's output neuron (others: 0 Hz)
    eta: float = 2e-4              # initial learning rate, nA per Hz²
    anneal: bool = True            # η decays linearly to 0 over the epochs
    epochs: int = 60
    batch_size: int = 16
    w_init_std_na: float = 0.1     # small random start; the teacher signal breaks symmetry
    w_max_na: float = 20.0         # bound on |w| for excitatory and inhibitory synapses
    class_weight: str | None = None  # "balanced": weight errors by inverse class frequency
    class_weight_power: float = 0.5  # exponent on the inverse frequency: 1 = fully balanced, 0 = unweighted


class SpikingReadoutLayer:
    """One LIF output neuron per class, trained with the delta rule on spike rates."""

    def __init__(self, n_inputs: int, classes: Sequence[str], neuron: LIFParams | None = None,
                 params: ReadoutParams | None = None, seed: int | None = None):
        self.classes = list(classes)
        self.p = params or ReadoutParams()
        self.rng = np.random.default_rng(seed)
        w = self.rng.normal(0.0, self.p.w_init_std_na, size=(n_inputs, len(self.classes)))
        self.lif = LIFInputLayer(w, neuron or LIFParams(), labels=[f"out[{c}]" for c in self.classes])
        self.train_accuracy_history: list[float] = []  # running accuracy per training epoch

    @property
    def w(self) -> np.ndarray:
        return self.lif.w

    # ── simulation ──────────────────────────────────────────────────────────

    def _stack(self, trains: Sequence[SpikeTrain]) -> tuple[np.ndarray, float]:
        """(T, B, n_inputs) raster for B equally long spike trains, plus their dt."""
        dts, lengths = {t.dt_s for t in trains}, {len(t.raster) for t in trains}
        if len(dts) != 1 or len(lengths) != 1:
            raise ValueError("all spike trains must share dt and length")
        rasters = np.stack([t.raster for t in trains], axis=1)
        if rasters.shape[2] != self.lif.n_inputs:
            raise ValueError(f"spike trains have {rasters.shape[2]} channels, readout expects {self.lif.n_inputs}")
        return rasters, dts.pop()

    def _simulate(self, rasters: np.ndarray, dt_s: float) -> np.ndarray:
        """(B, n_classes) output spike counts for a (T, B, n_inputs) batch, all samples in parallel."""
        self.lif.reset(batch_size=rasters.shape[1])
        counts = np.zeros((rasters.shape[1], len(self.classes)))
        for x in rasters:
            counts += self.lif.step(x, dt_s)
        return counts

    def output_counts(self, trains: Sequence[SpikeTrain]) -> np.ndarray:
        return self._simulate(*self._stack(trains))

    # ── training and prediction ─────────────────────────────────────────────

    def fit(self, trains: Sequence[SpikeTrain], targets: Sequence[str], epochs: int | None = None,
            eta: float | None = None) -> SpikingReadoutLayer:
        """Train from the current weights (so repeated calls fine-tune); `epochs` and `eta` override
        params.epochs and params.eta for this call."""
        n_epochs = self.p.epochs if epochs is None else epochs
        base_eta = self.p.eta if eta is None else eta
        rasters, dt = self._stack(trains)
        duration = rasters.shape[0] * dt
        y = np.array([self.classes.index(t) for t in targets])
        teacher = np.zeros((len(y), len(self.classes)))
        teacher[np.arange(len(y)), y] = self.p.target_rate_hz
        in_rates = rasters.sum(axis=0) / duration                           # (B, n_inputs)
        if self.p.class_weight == "balanced":
            per_class = np.bincount(y, minlength=len(self.classes))
            raw = len(y) / (np.count_nonzero(per_class) * np.maximum(per_class, 1))
            class_w = raw ** self.p.class_weight_power
            class_w /= np.sum(per_class * class_w) / len(y)            # mean weight per sample = 1
            sample_w = class_w[y][:, None]
        elif self.p.class_weight is None:
            sample_w = np.ones((len(y), 1))
        else:
            raise ValueError('class_weight must be None or "balanced"')

        for epoch in range(n_epochs):
            eta = base_eta * (1.0 - epoch / n_epochs) if self.p.anneal else base_eta
            correct = 0
            order = self.rng.permutation(len(y))
            for start in range(0, len(y), self.p.batch_size):
                idx = order[start:start + self.p.batch_size]
                counts = self._simulate(rasters[:, idx], dt)
                error = (teacher[idx] - counts / duration) * sample_w[idx]  # (b, n_classes), Hz
                self.lif.w += eta * in_rates[idx].T @ error / len(idx)
                np.clip(self.lif.w, -self.p.w_max_na, self.p.w_max_na, out=self.lif.w)
                correct += int(np.sum(self.predict_counts(counts) == np.array(self.classes)[y[idx]]))
            self.train_accuracy_history.append(correct / len(y))
        return self

    def predict_counts(self, counts: np.ndarray) -> np.ndarray:
        """Class of the most active output neuron per row; SILENT where none fired."""
        labels = np.array(self.classes, dtype=object)[np.argmax(counts, axis=1)]
        labels[counts.sum(axis=1) == 0] = SILENT
        return labels

    def predict(self, trains: Sequence[SpikeTrain]) -> np.ndarray:
        return self.predict_counts(self.output_counts(trains))

    def score(self, trains: Sequence[SpikeTrain], targets: Sequence[str]) -> float:
        return float(np.mean(self.predict(trains) == np.asarray(targets)))
