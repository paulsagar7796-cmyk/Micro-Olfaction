"""
Spiking-neural-network input layer built from Leaky Integrate-and-Fire neurons.

It consumes the `SpikeTrain` produced by `snn_input.ingest` (one channel per MOx
sensor) and emits a new `SpikeTrain`, so further SNN layers can be stacked on it.

Neuron model (current-based LIF with exponential synapses)
──────────────────────────────────────────────────────────
Membrane potential of neuron i:

    τ_m · dV_i/dt = −(V_i − V_rest) + R_m · (I_i(t) + I_ext)            (1)

The −(V − V_rest) term is the leak: with no input, V decays exponentially to
V_rest with time constant τ_m = R_m · C_m.

Synaptic current, driven by input spike trains s_j(t) = Σ_k δ(t − t_j^k):

    τ_syn · dI_i/dt = −I_i + τ_syn · Σ_j W_ji · s_j(t)                  (2)

so every input spike on channel j instantly raises I_i by W_ji (nA), and the
current then leaks away with time constant τ_syn.

Threshold, reset and refractoriness:

    if V_i(t) ≥ V_th:  emit a spike,  V_i ← V_reset,  hold V_i for t_ref     (3)

Discretisation (exact integration, Rotter & Diesmann 1999)
──────────────────────────────────────────────────────────
Between input spikes (1)–(2) are linear, so they are solved exactly over a step Δt:

    α = exp(−Δt/τ_m)          membrane decay factor
    β = exp(−Δt/τ_syn)        synaptic decay factor
    κ = τ_syn/(τ_m − τ_syn) · (α − β)     (κ = (Δt/τ_m)·α when τ_m = τ_syn)

    V[t+Δt] = V_rest + (V[t] − V_rest)·α + R_m·κ·I[t] + R_m·I_ext·(1 − α)
    I[t+Δt] = I[t]·β + Σ_j W_ji · s_j[t]

The result does not depend on Δt beyond spike-time quantisation, unlike forward
Euler. For constant drive I the firing rate has the closed form

    f = 1 / ( t_ref + τ_m · ln((V_∞ − V_reset) / (V_∞ − V_th)) ),   V_∞ = V_rest + R_m·I   (4)

(zero when V_∞ ≤ V_th); `lif_rate_constant_current` implements (4).

Units: mV, nA, MΩ (so MΩ·nA = mV), ms for time constants, seconds for dt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snn_input import SpikeTrain


@dataclass
class LIFParams:
    tau_m_ms: float = 20.0      # membrane time constant (leak)
    tau_syn_ms: float = 10.0    # synaptic current decay
    r_m_mohm: float = 10.0      # membrane resistance
    v_rest_mv: float = -65.0
    v_reset_mv: float = -65.0
    v_th_mv: float = -50.0
    t_ref_ms: float = 2.0
    i_ext_na: float = 0.0       # constant bias current

    def propagators(self, dt_s: float) -> tuple[float, float, float]:
        """(α, β, κ) for exact integration over one step of dt_s seconds."""
        dt_ms = dt_s * 1e3
        alpha = np.exp(-dt_ms / self.tau_m_ms)
        beta = np.exp(-dt_ms / self.tau_syn_ms)
        if np.isclose(self.tau_m_ms, self.tau_syn_ms):
            kappa = dt_ms / self.tau_m_ms * alpha
        else:
            kappa = self.tau_syn_ms / (self.tau_m_ms - self.tau_syn_ms) * (alpha - beta)
        return alpha, beta, kappa


def lif_rate_constant_current(i_na: np.ndarray | float, p: LIFParams | None = None) -> np.ndarray:
    """Analytic firing rate (Hz) for a constant input current, equation (4)."""
    p = p or LIFParams()
    v_inf = p.v_rest_mv + p.r_m_mohm * np.asarray(i_na, dtype=float)
    rate = np.zeros_like(v_inf)
    above = v_inf > p.v_th_mv
    t_rise_ms = p.tau_m_ms * np.log((v_inf[above] - p.v_reset_mv) / (v_inf[above] - p.v_th_mv))
    rate[above] = 1e3 / (p.t_ref_ms + t_rise_ms)
    return rate


# ─────────────────────────────────────────────────────────────────────────────
# Input → neuron connectivity (weights W, shape n_inputs × n_neurons, in nA)
# ─────────────────────────────────────────────────────────────────────────────

def one_to_one_weights(n: int, w_na: float = 5.0) -> np.ndarray:
    """Each sensor channel drives its own neuron."""
    return np.eye(n) * w_na


def convergent_weights(n_inputs: int, n_neurons: int, fan_in: int = 4, w_na: float = 2.5,
                       seed: int | None = None) -> np.ndarray:
    """Each neuron sums `fan_in` randomly chosen sensor channels, so it fires for
    co-activation patterns (odor combinations) rather than single sensors."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_inputs, n_neurons))
    for i in range(n_neurons):
        w[rng.choice(n_inputs, size=min(fan_in, n_inputs), replace=False), i] = w_na
    return w


# ─────────────────────────────────────────────────────────────────────────────
# LIF input layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LIFLayerOutput:
    spike_train: SpikeTrain
    v_mv: np.ndarray | None      # (T, N) membrane potential, if recorded
    i_syn_na: np.ndarray | None  # (T, N) synaptic current, if recorded


class LIFInputLayer:
    """Population of LIF neurons driven by the ingestion layer's spike trains."""

    def __init__(self, weights: np.ndarray, params: LIFParams | None = None,
                 labels: list[str] | None = None):
        self.w = np.asarray(weights, dtype=float)
        if self.w.ndim != 2:
            raise ValueError("weights must be a 2-D (n_inputs, n_neurons) matrix")
        self.p = params or LIFParams()
        self.labels = labels or [f"lif{i}" for i in range(self.n_neurons)]
        self.reset()

    @property
    def n_inputs(self) -> int:
        return self.w.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.w.shape[1]

    def reset(self, batch_size: int | None = None) -> None:
        """Return to rest. With `batch_size`, state becomes (batch_size, n_neurons) and step()
        advances that many independent samples in parallel (input (batch_size, n_inputs))."""
        shape = (self.n_neurons,) if batch_size is None else (batch_size, self.n_neurons)
        self.v = np.full(shape, self.p.v_rest_mv)
        self.i_syn = np.zeros(shape)
        self.refractory_s = np.zeros(shape)

    def step(self, input_spikes: np.ndarray, dt_s: float) -> np.ndarray:
        """Advance one step with boolean input (n_inputs,), or (batch, n_inputs) after
        reset(batch_size); return output spikes of matching shape (n_neurons,) / (batch, n_neurons)."""
        p = self.p
        alpha, beta, kappa = p.propagators(dt_s)

        # Membrane: exact leak toward V_rest plus synaptic and bias drive (eq. 1);
        # neurons in their refractory period stay clamped at V_reset.
        v_free = (p.v_rest_mv + (self.v - p.v_rest_mv) * alpha
                  + p.r_m_mohm * kappa * self.i_syn
                  + p.r_m_mohm * p.i_ext_na * (1.0 - alpha))
        refractory = self.refractory_s > 0.0
        self.v = np.where(refractory, p.v_reset_mv, v_free)
        self.refractory_s = np.maximum(self.refractory_s - dt_s, 0.0)

        # Synapses: exponential decay, then add this step's input spikes (eq. 2)
        self.i_syn = self.i_syn * beta + input_spikes.astype(float) @ self.w

        # Threshold, reset, refractory period (eq. 3)
        spikes = ~refractory & (self.v >= p.v_th_mv)
        self.v[spikes] = p.v_reset_mv
        self.refractory_s[spikes] = p.t_ref_ms * 1e-3
        return spikes

    def run(self, spike_train: SpikeTrain, record: bool = False) -> LIFLayerOutput:
        """Process a whole `SpikeTrain` (e.g. `ingest(...).spike_train`) from a reset state."""
        raster = spike_train.raster
        if raster.shape[1] != self.n_inputs:
            raise ValueError(f"spike train has {raster.shape[1]} channels, layer expects {self.n_inputs}")
        self.reset()
        out = np.zeros((len(raster), self.n_neurons), dtype=bool)
        v_rec = np.zeros(out.shape) if record else None
        i_rec = np.zeros(out.shape) if record else None
        for k, input_spikes in enumerate(raster):
            out[k] = self.step(input_spikes, spike_train.dt_s)
            if record:
                v_rec[k], i_rec[k] = self.v, self.i_syn
        return LIFLayerOutput(SpikeTrain(out, spike_train.dt_s, self.labels), v_rec, i_rec)


# ─────────────────────────────────────────────────────────────────────────────
# Example: olfaction ingestion pipeline → LIF input layer
# ─────────────────────────────────────────────────────────────────────────────

def example() -> None:
    from snn_input import DEMO_DURATION_S, DEMO_PULSES, GasDeliverySchedule, build_layer

    # 1. Sensory ingestion: gas → porous surface → MOx array → microLED spike encoder
    ingestion = build_layer(porous="ZIF-8", seed=7, encoder="linear")
    sensor_spikes = ingestion.run(GasDeliverySchedule(DEMO_PULSES), DEMO_DURATION_S).spike_train

    # 2. SNN input layer: one LIF neuron per sensor channel
    lif = LIFInputLayer(one_to_one_weights(len(sensor_spikes.labels)),
                        labels=[f"LIF[{name}]" for name in sensor_spikes.labels])
    snn_spikes = lif.run(sensor_spikes).spike_train

    windows = [("clean air", 0.0, 5.0)] + [("+".join(p.concentrations_ppm), p.t_start_s, p.t_end_s)
                                           for p in DEMO_PULSES]
    print(f"{'channel':<22}" + "".join(f"{name:>24}" for name, *_ in windows))
    for ch, label in enumerate(sensor_spikes.labels):
        cells = []
        for _, t0, t1 in windows:
            f_in = sensor_spikes.firing_rates_hz(t0, t1)[ch]
            f_out = snn_spikes.firing_rates_hz(t0, t1)[ch]
            cells.append(f"{f_in:6.1f} → {f_out:5.1f} Hz")
        print(f"{label:<22}" + "".join(f"{c:>24}" for c in cells))
    print(f"\nsensor spikes: {int(sensor_spikes.raster.sum())}, LIF spikes: {int(snn_spikes.raster.sum())}")
    print("downstream SNN input (time_s, neuron):", np.round(snn_spikes.events()[:3], 4).tolist(), "...")


if __name__ == "__main__":
    example()
