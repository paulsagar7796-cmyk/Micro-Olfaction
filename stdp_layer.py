"""
SNN classification layer: LIF neurons with STDP learning and winner-takes-all
lateral inhibition, trained without labels on the output of `LIFInputLayer`.

Neurons
───────
Current-based LIF, identical to lif_neurons.py (exact integration, same LIFParams):

    τ_m · dV_i/dt   = −(V_i − V_rest) + R_m · I_i
    τ_syn · dI_i/dt = −I_i + τ_syn · Σ_j w_ji · s_j(t)

with an adaptive threshold V_th,i = V_th + θ_i (homeostasis, below).

Spike-Timing-Dependent Plasticity
─────────────────────────────────
For a pre-synaptic spike at t_pre and a post-synaptic spike at t_post,
Δt = t_post − t_pre:

    Δw = +η₊ · e^(−Δt/τ₊) · f₊(w)    if Δt > 0   (input precedes output → potentiation, LTP)
    Δw = −η₋ · e^(+Δt/τ₋) · f₋(w)    if Δt < 0   (output precedes input → depression, LTD)

Soft bounds keep 0 ≤ w ≤ w_max:  f₊(w) = w_max − w,  f₋(w) = w
(additive mode: f₊ = f₋ = w_max, then clipped).

Summed over all spike pairs this is computed online with exponential traces:

    dx_j/dt = −x_j/τ₊,   x_j ← x_j + 1 on each input spike j      (pre trace)
    dy_i/dt = −y_i/τ₋,   y_i ← y_i + 1 on each output spike i     (post trace)

    on output spike i:  w_ji ← w_ji + η₊ · x_j · f₊(w_ji)    for every input j   (LTP)
    on input spike j:   w_ji ← w_ji − η₋ · y_i · f₋(w_ji)    for every neuron i  (LTD)

Lateral inhibition (winner-takes-all)
─────────────────────────────────────
At most one neuron fires per time step: of the neurons at or above threshold,
the most depolarised one, (V_i − V_th,i), wins. When neuron k fires, every other
neuron is hyperpolarised:

    V_i ← max(V_i − V_inh, V_floor)    for all i ≠ k

so a neuron that already matches the current input keeps suppressing its
competitors, and different neurons specialise on different odors.

Homeostasis and normalisation (stop one neuron winning everything)
──────────────────────────────────────────────────────────────────
    τ_θ · dθ_i/dt = −θ_i,   θ_i ← θ_i + θ₊ on each spike of i   (adaptive threshold)
    θ_i ← θ_i − mean(θ)   (optional) so frequent winners get harder to fire and
                          silent neurons easier, without raising the layer's overall
                          threshold (which would silence weak, low-ppm odors)

    Starvation (dead-neuron recovery). Lateral inhibition can keep a neuron below
    threshold indefinitely even with healthy input drive. A neuron silent for longer
    than T_starve is "starving"; each ms its threshold creeps toward a floor just
    above rest,

        θ_i ← θ_floor + (θ_i − θ_floor) · d        d = threshold_decay per ms,
        θ_floor = V_rest + margin − V_th           (effective threshold → V_rest + margin)

    and it is excluded from the zero-mean centring (which is taken over non-starving
    neurons only), so it can fall far below the mean until it wins again, learns the
    current input, and rejoins the competition. θ ≥ θ_floor always.
    after each sample:  w_·i ← w_·i · W_total / Σ_j w_ji          (synaptic scaling)
                        (with normalization_interval_ms, also every that many ms
                        of learning within a presentation)

Initialisation
──────────────
Default: uniform random weights. With `init_samples` (a normalised training feature
matrix, values in [0, 1]), each neuron's weight vector starts as w_max × one randomly
drawn training row, then is synaptically scaled. Neurons then begin near real odor
patterns instead of equidistant from all of them, so fewer are left with nothing to win.

Classification
──────────────
Learning is unsupervised. Afterwards the mean spike count of every neuron for
every class, R[c, n], is measured on labelled samples, and each neuron is labelled
with its argmax class. A new sample with spike counts x is then classified by one
of three count-based readouts:

    vote      Σ x_n over neurons labelled c                  (hard labels)
    soft      Σ_n x_n · R[c, n] / Σ_c' R[c', n]               (each neuron's spikes split
                                                              across classes by its tuning)
    template  cosine(x, R[c, ·])                              (match the class's mean
                                                              population pattern)

The STDP code is distributed (each odor drives a *combination* of neurons), which
hard labels read poorly; a trained readout (readout_layer.py) reads it best.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from lif_neurons import LIFParams
from snn_input import SpikeTrain


UNASSIGNED = "unassigned"  # label for neurons that never fire during assign_labels
COUNT_READOUTS = ("vote", "soft", "template")  # methods of classify_counts


@dataclass
class STDPParams:
    eta_plus: float = 0.01       # LTP learning rate
    eta_minus: float = 0.012     # LTD learning rate (slightly larger: net depression of uncorrelated inputs)
    tau_plus_ms: float = 20.0    # pre-trace time constant (LTP window)
    tau_minus_ms: float = 20.0   # post-trace time constant (LTD window)
    w_max_na: float = 10.0
    soft_bound: bool = True
    w_total_na: float | None = 20.0  # per-neuron sum of input weights after each sample; None disables
    normalization_interval_ms: float | None = None  # also normalise every N ms of learning within a
                                                    # presentation; None: only at its end


@dataclass
class InhibitionParams:
    v_inh_mv: float = 20.0         # hyperpolarisation of losers when a neuron fires
    v_floor_mv: float = -85.0      # inhibition cannot push V below this
    theta_plus_mv: float = 0.05    # threshold increase per spike (homeostasis); larger values let θ,
                                   # not the learned weights, decide the winner
    tau_theta_s: float = 200.0     # threshold relaxation time constant
    theta_zero_mean: bool = True   # θ only redistributes excitability between neurons (layer mean 0)
    starvation_window_s: float = 5.0         # silence (s, learning time) before a neuron counts as starving
    threshold_decay: float | None = None     # per-ms factor pulling a starving θ toward the floor
                                             # (e.g. 0.999 → τ ≈ 1 s); None = off. Off by default: on Batch 1
                                             # it caused churn and more dead neurons, not fewer
    theta_floor_margin_mv: float = 1.0       # lowest effective threshold, above V_rest


@dataclass
class STDPLayerOutput:
    spike_train: SpikeTrain
    spike_counts: np.ndarray            # (n_neurons,) spikes per neuron over the processed window
    weights_na: np.ndarray              # (n_inputs, n_neurons) snapshot after processing
    v_mv: np.ndarray | None = None      # (T, n_neurons) if recorded
    weight_history: list[np.ndarray] = field(default_factory=list)


class STDPClassificationLayer:
    """Competitive LIF layer that learns odor prototypes with STDP and lateral inhibition."""

    def __init__(self, n_inputs: int, n_neurons: int, neuron: LIFParams | None = None,
                 stdp: STDPParams | None = None, inhibition: InhibitionParams | None = None,
                 seed: int | None = None, init_samples: np.ndarray | None = None):
        self.neuron = neuron or LIFParams()
        self.stdp = stdp or STDPParams()
        self.inh = inhibition or InhibitionParams()
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(0.2, 0.8, size=(n_inputs, n_neurons)) * self.stdp.w_max_na
        if init_samples is not None:
            self._init_from_samples(np.asarray(init_samples, dtype=float), rng)
        self.normalize_weights()
        self.theta = np.zeros(n_neurons)
        self.silent_s = np.zeros(n_neurons)  # learning time since each neuron last fired
        self.neuron_labels: np.ndarray | None = None  # class per neuron, set by assign_labels
        self.reset_state()

    @property
    def n_inputs(self) -> int:
        return self.w.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.w.shape[1]

    # ── dynamics ────────────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset fast state (potentials, currents, traces) between samples; keeps weights and θ."""
        self.v = np.full(self.n_neurons, self.neuron.v_rest_mv)
        self.i_syn = np.zeros(self.n_neurons)
        self.refractory_s = np.zeros(self.n_neurons)
        self.x_pre = np.zeros(self.n_inputs)
        self.y_post = np.zeros(self.n_neurons)

    def _init_from_samples(self, x: np.ndarray, rng: np.random.Generator) -> None:
        """Seed each neuron's weights with w_max × a randomly drawn row of x (n_samples, n_inputs)."""
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"init_samples must have shape (n_samples, {self.n_inputs}), got {x.shape}")
        rows = rng.choice(len(x), size=self.n_neurons, replace=len(x) < self.n_neurons)
        seeded = np.clip(x[rows].T, 0.0, 1.0) * self.stdp.w_max_na      # (n_inputs, n_neurons)
        usable = seeded.sum(axis=0) > 0  # an all-zero row would give a neuron no input at all
        self.w[:, usable] = seeded[:, usable]

    def normalize_weights(self) -> None:
        total = self.stdp.w_total_na
        if total is not None:
            self.w *= total / np.maximum(self.w.sum(axis=0, keepdims=True), 1e-12)
            np.clip(self.w, 0.0, self.stdp.w_max_na, out=self.w)

    def step(self, input_spikes: np.ndarray, dt_s: float, learn: bool = True) -> np.ndarray:
        """Advance one step with a boolean input vector (n_inputs,); return output spikes (n_neurons,)."""
        p, s, h = self.neuron, self.stdp, self.inh
        alpha, beta, kappa = p.propagators(dt_s)
        pre = input_spikes.astype(bool)

        # 0. Traces decay first, so a spike pair Δt apart is weighted by exactly e^(−Δt/τ)
        self.x_pre *= np.exp(-dt_s * 1e3 / s.tau_plus_ms)
        self.y_post *= np.exp(-dt_s * 1e3 / s.tau_minus_ms)

        # 1. Membrane update (exact LIF integration); refractory neurons stay clamped
        v_free = p.v_rest_mv + (self.v - p.v_rest_mv) * alpha + p.r_m_mohm * kappa * self.i_syn
        refractory = self.refractory_s > 0.0
        self.v = np.where(refractory, p.v_reset_mv, v_free)
        self.refractory_s = np.maximum(self.refractory_s - dt_s, 0.0)

        # 2. Threshold + winner-takes-all: one winner, everyone else inhibited
        margin = self.v - (p.v_th_mv + self.theta)
        margin[refractory] = -np.inf
        post = np.zeros(self.n_neurons, dtype=bool)
        winner = int(np.argmax(margin))
        if margin[winner] >= 0.0:
            post[winner] = True
            losers = np.arange(self.n_neurons) != winner
            self.v[losers] = np.maximum(self.v[losers] - h.v_inh_mv, h.v_floor_mv)
            self.v[winner] = p.v_reset_mv
            self.refractory_s[winner] = p.t_ref_ms * 1e-3

        # 3. STDP using traces from earlier steps (current-step pre spikes cannot have caused this post spike)
        if learn:
            if post.any():  # LTP: input before output
                f_plus = (s.w_max_na - self.w[:, post]) if s.soft_bound else s.w_max_na
                self.w[:, post] += s.eta_plus * self.x_pre[:, None] * f_plus
            if pre.any():   # LTD: output before input
                f_minus = self.w[pre] if s.soft_bound else s.w_max_na
                self.w[pre] -= s.eta_minus * self.y_post[None, :] * f_minus
            np.clip(self.w, 0.0, s.w_max_na, out=self.w)

            # Homeostatic threshold, with starving neurons sinking toward the floor
            floor = p.v_rest_mv + h.theta_floor_margin_mv - p.v_th_mv
            self.silent_s = np.where(post, 0.0, self.silent_s + dt_s)
            starving = (self.silent_s > h.starvation_window_s) if h.threshold_decay is not None \
                else np.zeros(self.n_neurons, dtype=bool)
            competing = ~starving
            self.theta[competing] *= np.exp(-dt_s / h.tau_theta_s)
            self.theta[post] += h.theta_plus_mv
            if starving.any():
                decay = h.threshold_decay ** (dt_s * 1e3)
                self.theta[starving] = floor + (self.theta[starving] - floor) * decay
            if h.theta_zero_mean and competing.any():
                self.theta[competing] -= self.theta[competing].mean()
            np.maximum(self.theta, floor, out=self.theta)

        # 4. Trace increments, then deliver this step's input spikes to the synapses
        self.x_pre += pre
        self.y_post += post
        self.i_syn = self.i_syn * beta + pre.astype(float) @ self.w
        return post

    def process(self, spike_train: SpikeTrain, duration_s: float | None = None, learn: bool = True,
                record: bool = False) -> STDPLayerOutput:
        """Present one SpikeTrain for `duration_s` seconds (default: its full length).

        Longer durations append silent input (a rest period); shorter ones truncate.
        With `learn=True` weights and thresholds are updated online, and weights are
        renormalised at the end of the presentation (and every `normalization_interval_ms`
        within it, if set).
        """
        raster = spike_train.raster
        if raster.shape[1] != self.n_inputs:
            raise ValueError(f"spike train has {raster.shape[1]} channels, layer expects {self.n_inputs}")
        dt = spike_train.dt_s
        n_steps = len(raster) if duration_s is None else int(round(duration_s / dt))
        silent = np.zeros(self.n_inputs, dtype=bool)

        interval = self.stdp.normalization_interval_ms
        if interval is not None and interval <= 0:
            raise ValueError("normalization_interval_ms must be > 0")
        every = int(round(interval * 1e-3 / dt)) if learn and interval else 0

        self.reset_state()
        out = np.zeros((n_steps, self.n_neurons), dtype=bool)
        v_rec = np.zeros(out.shape) if record else None
        history = []
        for k in range(n_steps):
            out[k] = self.step(raster[k] if k < len(raster) else silent, dt, learn)
            if every and (k + 1) % every == 0 and k + 1 < n_steps:  # the end-of-presentation one follows below
                self.normalize_weights()
            if record:
                v_rec[k] = self.v
                if k % int(round(0.1 / dt)) == 0:
                    history.append(self.w.copy())
        if learn:
            self.normalize_weights()

        labels = [f"stdp{i}" if self.neuron_labels is None else f"stdp{i}[{self.neuron_labels[i]}]"
                  for i in range(self.n_neurons)]
        return STDPLayerOutput(SpikeTrain(out, dt, labels), out.sum(axis=0), self.w.copy(), v_rec, history)

    # ── classification API ──────────────────────────────────────────────────

    def train(self, samples: Sequence[SpikeTrain], epochs: int = 1, duration_s: float | None = None,
              shuffle_seed: int | None = 0) -> None:
        """Unsupervised STDP training: present every sample `epochs` times in shuffled order."""
        rng = np.random.default_rng(shuffle_seed)
        for _ in range(epochs):
            for idx in rng.permutation(len(samples)):
                self.process(samples[idx], duration_s, learn=True)

    def responses(self, samples: Sequence[SpikeTrain], duration_s: float | None = None) -> np.ndarray:
        """(n_samples, n_neurons) spike counts with learning frozen."""
        return np.stack([self.process(s, duration_s, learn=False).spike_counts for s in samples])

    def assign_labels(self, samples: Sequence[SpikeTrain], targets: Sequence[str],
                      duration_s: float | None = None) -> None:
        """Label each neuron with the class that drives it hardest (mean spikes per sample).

        Neurons that never fire on these samples are labelled UNASSIGNED (otherwise argmax
        over an all-zero row would silently hand them to the first class).
        """
        self.label_neurons(self.responses(samples, duration_s), targets)

    def label_neurons(self, counts: np.ndarray, targets: Sequence[str]) -> None:
        """As assign_labels, from precomputed (n_samples, n_neurons) spike counts."""
        self.classes = sorted(set(targets))
        targets = np.asarray(targets)
        mean = np.stack([counts[targets == c].mean(axis=0) for c in self.classes])  # (C, N)
        labels = np.array(self.classes, dtype=object)[np.argmax(mean, axis=0)]
        labels[mean.max(axis=0) == 0] = UNASSIGNED
        self.neuron_labels = labels
        self.class_response = mean  # R[c, n]: mean spikes of neuron n for class c

    def fit(self, samples: Sequence[SpikeTrain], targets: Sequence[str], epochs: int = 1,
            duration_s: float | None = None) -> STDPClassificationLayer:
        """Train with STDP (labels unused), then label neurons from their responses."""
        self.train(samples, epochs, duration_s)
        self.assign_labels(samples, targets, duration_s)
        return self

    def predict(self, spike_train: SpikeTrain, duration_s: float | None = None) -> str | None:
        """Class whose neurons fire the most spikes in total; None if the layer stays silent."""
        if self.neuron_labels is None:
            raise RuntimeError("call fit() or assign_labels() first")
        return self.classify_counts(self.process(spike_train, duration_s, learn=False).spike_counts)

    def classify_counts(self, counts: np.ndarray, method: str = "vote") -> str | None:
        """Class for per-neuron spike counts; None if the layer stayed silent.

        vote: class whose labelled neurons fire the most spikes in total (no neurons → 0)
        soft: each neuron's spikes split across classes in proportion to its mean response
        template: class whose mean population pattern is most similar (cosine)
        """
        if counts.sum() == 0:
            return None
        if method == "vote":
            votes = [counts[self.neuron_labels == c].sum() for c in self.classes]
        elif method == "soft":
            tuning = self.class_response / np.maximum(self.class_response.sum(axis=0, keepdims=True), 1e-12)
            votes = tuning @ counts
        elif method == "template":
            norms = np.linalg.norm(self.class_response, axis=1) * np.linalg.norm(counts)
            votes = (self.class_response @ counts) / np.maximum(norms, 1e-12)
        else:
            raise ValueError(f"method must be one of {COUNT_READOUTS}")
        return self.classes[int(np.argmax(votes))]

    def score(self, samples: Sequence[SpikeTrain], targets: Sequence[str],
              duration_s: float | None = None) -> float:
        return float(np.mean([self.predict(s, duration_s) == t for s, t in zip(samples, targets)]))


# ─────────────────────────────────────────────────────────────────────────────
# Example: ingestion pipeline → LIF input layer → STDP classification layer
# ─────────────────────────────────────────────────────────────────────────────

def odor_sample(odor: str, ppm: float, seed: int, duration_s: float = 8.0) -> SpikeTrain:
    """One labelled presentation: odor pulse → ingestion → LIF input layer → SpikeTrain."""
    from lif_neurons import LIFInputLayer, one_to_one_weights
    from snn_input import (GasDeliverySchedule, LinearMicroLEDRateEncoder, MOxSensorArray, OdorPulse,
                           SensoryIngestionLayer, default_sensor_array)

    ingestion = SensoryIngestionLayer(MOxSensorArray(default_sensor_array("ZIF-8"), seed=seed),
                                      LinearMicroLEDRateEncoder(), dt_encoder_s=0.001)
    schedule = GasDeliverySchedule([OdorPulse(0.0, duration_s, {odor: ppm})])
    sensor_spikes = ingestion.run(schedule, duration_s).spike_train
    lif = LIFInputLayer(one_to_one_weights(len(sensor_spikes.labels)))
    return lif.run(sensor_spikes).spike_train


def example() -> None:
    odors = ["ethanol", "acetone", "formaldehyde", "ammonia"]
    train_ppm, test_ppm = (8.0, 12.0, 16.0, 20.0, 25.0), (10.0, 18.0, 22.0)

    print("building spike-train datasets (ingestion → LIF input layer)...")
    train = [(odor_sample(o, c, seed=100 + i), o) for i, (o, c) in
             enumerate((o, c) for o in odors for c in train_ppm)]
    test = [(odor_sample(o, c, seed=500 + i), o) for i, (o, c) in
            enumerate((o, c) for o in odors for c in test_ppm)]

    layer = STDPClassificationLayer(n_inputs=12, n_neurons=16, seed=1)
    layer.fit([s for s, _ in train], [t for _, t in train], epochs=3)

    names, counts = np.unique(layer.neuron_labels, return_counts=True)
    print("neurons per class:", {str(n): int(c) for n, c in zip(names, counts)})
    print(f"train accuracy: {layer.score(*zip(*train)):.0%}   test accuracy (unseen ppm): {layer.score(*zip(*test)):.0%}")
    for spikes, target in test:
        print(f"  {target:<13} → {layer.predict(spikes)}")


if __name__ == "__main__":
    example()
