"""
Sensory ingestion layer for a neuromorphic olfaction architecture.

Signal chain (one channel per sensor):

    gas delivery ──► porous surface ──────► MOx chemiresistor ──► microLED spike encoder ──► spike events
    (VOC ppm)        (molecular sieving,       (power-law response,    (LED drive current ──► optical
                      Knudsen pore diffusion,   first-order surface      power ──► photodiode ──► LIF
                      competitive Langmuir      kinetics, drift, noise)  neuron, or linear
                      adsorption → enrichment)                           ppm → frequency)

The array is made heterogeneous ("electronic nose" style) by combining different
metal-oxide chemistries with different heater temperatures, so every VOC produces
a distinct spatio-temporal spike pattern for a downstream spiking network.

Material constants are representative literature-scale values chosen for
qualitative realism; calibrate against measured sensor data before relying on
absolute numbers.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

R_GAS = 8.314            # J / (mol K)
PLANCK_C_EV_NM = 1239.84  # h*c in eV*nm


# ─────────────────────────────────────────────────────────────────────────────
# Chemistry: target VOCs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VOC:
    name: str
    molar_mass_g_mol: float
    kinetic_diameter_nm: float  # sets whether the molecule fits through a pore aperture
    t_opt_c: float              # surface-reaction temperature of peak MOx response


VOC_LIBRARY: dict[str, VOC] = {
    v.name: v
    for v in (
        VOC("ethanol", 46.07, 0.450, 300.0),
        VOC("acetone", 58.08, 0.470, 350.0),
        VOC("toluene", 92.14, 0.585, 375.0),
        VOC("formaldehyde", 30.03, 0.243, 250.0),
        VOC("ammonia", 17.03, 0.260, 300.0),
    )
}
VOC_NAMES: list[str] = list(VOC_LIBRARY)


# ─────────────────────────────────────────────────────────────────────────────
# Porous surface module: adsorption enhancement ahead of the MOx sensors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PorousMaterial:
    """Adsorptive over-layer that pre-concentrates odor molecules at the MOx surface.

    - pore aperture      → which molecules get in (soft molecular-sieving cut-off)
    - pore diameter      → Knudsen transport through the film
    - surface area, porosity → number of adsorption sites (enrichment capacity)
    - affinity, enthalpy → per-VOC Langmuir binding; exothermic, so it weakens
                           as the heater temperature rises
    """
    name: str
    specific_surface_area_m2_g: float
    pore_diameter_nm: float
    porosity: float                     # void fraction, 0..1
    adsorption_enthalpy_kj_mol: float
    k_ref_per_ppm: float                # Langmuir constant at T_REF_K for affinity 1
    capacity_ppm: float                 # site capacity at SSA_REF (ppm-equivalent enrichment)
    diffusion_hindrance: float = 1.0    # <1 for configurational diffusion in micropores
    aperture_nm: float = np.inf         # effective pore window for molecular sieving
    desorption_time_ref_s: float = 1.0  # adsorbate residence time at T_REF_K
    affinity: dict[str, float] = field(default_factory=dict)  # per-VOC binding relative to k_ref

    SSA_REF = 1000.0          # m^2/g
    T_REF_K = 573.15          # 300 °C
    SIEVE_SOFTNESS_NM = 0.02  # width of the sieving cut-off (framework flexibility)

    def affinity_vector(self) -> np.ndarray:
        return np.array([self.affinity.get(v, 0.0) for v in VOC_NAMES])

    def accessibility(self) -> np.ndarray:
        """Per-VOC fraction of the pore network a molecule can enter (0..1)."""
        d_kin = np.array([VOC_LIBRARY[v].kinetic_diameter_nm for v in VOC_NAMES])
        return 1.0 / (1.0 + np.exp((d_kin - self.aperture_nm) / self.SIEVE_SOFTNESS_NM))

    def _van_t_hoff(self, temp_k: float) -> float:
        dh = self.adsorption_enthalpy_kj_mol * 1e3
        return np.exp(dh / R_GAS * (1.0 / temp_k - 1.0 / self.T_REF_K))

    def langmuir_k(self, temp_k: float) -> np.ndarray:
        """Per-VOC equilibrium constant K = k_ads / k_des (1/ppm)."""
        return self.k_ref_per_ppm * self.affinity_vector() * self._van_t_hoff(temp_k)

    def desorption_rate(self, temp_k: float) -> float:
        """k_des (1/s): residence time shortens exponentially with temperature."""
        return 1.0 / (self.desorption_time_ref_s * self._van_t_hoff(temp_k))

    def site_capacity_ppm(self) -> np.ndarray:
        """Per-VOC enrichment capacity Q: sites scale with area and porosity, minus sieved-out pores."""
        return (self.capacity_ppm * self.porosity
                * self.specific_surface_area_m2_g / self.SSA_REF
                * self.accessibility())


POROUS_MATERIALS: dict[str, PorousMaterial] = {
    "none": PorousMaterial(
        "none", 5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ),
    "ZIF-8": PorousMaterial(  # MOF: 11.6 Å cavities; linker swing widens the 3.4 Å window to ~5.5 Å
        "ZIF-8", 1600.0, 1.16, 0.50, 35.0, 0.06, 60.0,
        diffusion_hindrance=1e-4, aperture_nm=0.55, desorption_time_ref_s=1.5,
        affinity={"ethanol": 1.0, "acetone": 0.9, "toluene": 0.8, "formaldehyde": 0.7, "ammonia": 0.5},
    ),
    "SBA-15": PorousMaterial(  # ordered mesoporous silica: open 6 nm channels, no sieving
        "SBA-15", 800.0, 6.0, 0.60, 25.0, 0.03, 60.0,
        desorption_time_ref_s=0.5,
        affinity={"ethanol": 0.8, "acetone": 0.7, "toluene": 0.6, "formaldehyde": 0.9, "ammonia": 0.8},
    ),
    "ZSM-5": PorousMaterial(  # zeolite: ~5.5 Å channels, flexible enough to admit toluene, strong aromatic uptake
        "ZSM-5", 400.0, 0.55, 0.35, 45.0, 0.08, 60.0,
        diffusion_hindrance=1e-2, aperture_nm=0.62, desorption_time_ref_s=3.0,
        affinity={"ethanol": 0.6, "acetone": 0.5, "toluene": 1.4, "formaldehyde": 0.4, "ammonia": 1.0},
    ),
}


@dataclass
class SensingArea:
    """Geometry and porous over-layer of one sensor's active region."""
    porous_material: PorousMaterial = POROUS_MATERIALS["ZIF-8"]
    film_thickness_um: float = 5.0
    active_area_mm2: float = 0.25
    tortuosity: float = 3.0

    def diffusion_time_s(self, temp_k: float) -> np.ndarray:
        """First-mode time constant for VOCs diffusing through the porous film (per VOC)."""
        m = self.porous_material
        if m.pore_diameter_nm <= 0 or m.porosity <= 0:
            return np.zeros(len(VOC_NAMES))
        d_pore = m.pore_diameter_nm * 1e-9
        molar_mass = np.array([VOC_LIBRARY[v].molar_mass_g_mol for v in VOC_NAMES]) * 1e-3
        d_knudsen = d_pore / 3.0 * np.sqrt(8.0 * R_GAS * temp_k / (np.pi * molar_mass))
        d_eff = d_knudsen * m.porosity / self.tortuosity * m.diffusion_hindrance * m.accessibility()
        thickness = self.film_thickness_um * 1e-6
        return 4.0 * thickness**2 / (np.pi**2 * d_eff)


class PorousSurface:
    """Dynamic model of each sensor's porous over-layer, run before the MOx reaction.

    For sensor n and VOC v (all arrays are N × V):

        pore gas    dc_v/dt = (C_v − c_v) / τ_diff,v                    diffusion into the film
        coverage    dθ_v/dt = k_ads,v · c_v · (1 − Σ_u θ_u) − k_des · θ_v  competitive Langmuir kinetics
        delivered   C_eff,v = c_v + Q_v · θ_v                            gas + adsorbed reservoir at the MOx

    Each VOC's steady-state enrichment C_eff/C is 1 + Q·K / (1 + Σ K·C). At
    trace levels it tends to 1 + Q·K, which is the factor the detection limit
    improves by; the MOx response S ∝ C^β then grows by enrichment^β. VOCs in a
    mixture compete for the same sites, so a strongly binding one suppresses
    the enrichment of the others.
    """

    def __init__(self, areas: list[SensingArea], temps_k: list[float]):
        self.capacity = np.stack([a.porous_material.site_capacity_ppm() for a in areas])           # Q, (N, V)
        self.k_des = np.array([a.porous_material.desorption_rate(t) for a, t in zip(areas, temps_k)])[:, None]
        k_eq = np.stack([a.porous_material.langmuir_k(t) for a, t in zip(areas, temps_k)])
        self.k_ads = k_eq * self.k_des                                                              # 1/(ppm s)
        self.tau_diff = np.stack([a.diffusion_time_s(t) for a, t in zip(areas, temps_k)])
        self.reset()

    @classmethod
    def from_sensors(cls, sensors: list[MOxSensor]) -> PorousSurface:
        return cls([s.sensing_area for s in sensors], [s.temp_k for s in sensors])

    @property
    def langmuir_k(self) -> np.ndarray:
        return self.k_ads / self.k_des

    def reset(self) -> None:
        self.c_pore = np.zeros_like(self.capacity)  # gas-phase VOC inside the pores (ppm)
        self.theta = np.zeros_like(self.capacity)   # fractional site coverage per VOC

    def step(self, conc_ppm: np.ndarray, dt: float) -> np.ndarray:
        """Advance by dt for gas-phase ppm (ordered as VOC_NAMES); return C_eff per sensor (N, V)."""
        self.c_pore += (1.0 - np.exp(-dt / np.maximum(self.tau_diff, 1e-9))) * (conc_ppm - self.c_pore)

        # Exponential update per VOC with the other adsorbates' coverage held fixed over the step;
        # unconditionally stable, and the renormalisation keeps total coverage ≤ 1.
        r_ads = self.k_ads * self.c_pore
        rate = r_ads + self.k_des
        sites_for_v = 1.0 - self.theta.sum(axis=1, keepdims=True) + self.theta
        theta_eq = r_ads * sites_for_v / rate
        self.theta = theta_eq + (self.theta - theta_eq) * np.exp(-rate * dt)
        self.theta /= np.maximum(self.theta.sum(axis=1, keepdims=True), 1.0)
        return self.effective_concentration()

    def effective_concentration(self) -> np.ndarray:
        return self.c_pore + self.capacity * self.theta

    def equilibrium_concentration(self, conc_ppm: np.ndarray) -> np.ndarray:
        """Steady-state C_eff (N, V) for a constant gas mixture (competitive Langmuir)."""
        kc = self.langmuir_k * conc_ppm
        return conc_ppm + self.capacity * kc / (1.0 + kc.sum(axis=1, keepdims=True))

    def single_voc_enrichment(self, conc_ppm: float) -> np.ndarray:
        """C_eff / C (N, V) when each VOC is present alone at conc_ppm; conc_ppm=0 gives the trace limit."""
        k = self.langmuir_k
        return 1.0 + self.capacity * k / (1.0 + k * conc_ppm)


def print_surface_report(conc_ppm: float = 20.0, temps_c: tuple[float, ...] = (275.0, 375.0),
                         beta: float = 0.5) -> None:
    """Tabulate how each porous material enhances adsorption and sensitivity per VOC."""
    print(f"\nPorous surface enhancement (single VOC; MOx gain assumes β = {beta})")
    header = (f"{'material':<8}{'T(°C)':>7}  {'VOC':<13}{'access':>7}{'K(1/ppm)':>10}"
              f"{'LOD gain':>10}{f'C_eff/C@{conc_ppm:g}':>13}{'MOx gain':>10}{'τ_diff(s)':>11}{'τ_des(s)':>10}")
    print(header)
    print("─" * len(header))
    for name, material in POROUS_MATERIALS.items():
        if name == "none":
            continue
        for t_c in temps_c:
            area = SensingArea(porous_material=material)
            surface = PorousSurface([area], [t_c + 273.15])
            trace = surface.single_voc_enrichment(0.0)[0]
            at_conc = surface.single_voc_enrichment(conc_ppm)[0]
            access = material.accessibility()
            tau_des = 1.0 / surface.k_des[0, 0]
            for v, voc in enumerate(VOC_NAMES):
                print(f"{name:<8}{t_c:>7.0f}  {voc:<13}{access[v]:>7.2f}{surface.langmuir_k[0, v]:>10.3f}"
                      f"{trace[v]:>10.2f}{at_conc[v]:>13.2f}{at_conc[v] ** beta:>10.2f}"
                      f"{surface.tau_diff[0, v]:>11.2f}{tau_des:>10.2f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# MOx chemiresistor array
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MOxMaterial:
    name: str
    carrier: str            # "n": resistance drops with reducing VOCs, "p": it rises
    r0_ohm: float           # baseline resistance in clean air
    beta: float             # power-law exponent, S = a * C^beta
    tau_rise_s: float
    tau_decay_s: float
    sensitivity: dict[str, float]


MOX_MATERIALS: dict[str, MOxMaterial] = {
    m.name: m
    for m in (
        MOxMaterial("SnO2", "n", 50e3, 0.50, 3.0, 12.0,
                    {"ethanol": 1.2, "acetone": 0.9, "toluene": 0.5, "formaldehyde": 0.7, "ammonia": 0.4}),
        MOxMaterial("WO3", "n", 200e3, 0.55, 4.0, 15.0,
                    {"ethanol": 0.6, "acetone": 1.3, "toluene": 0.3, "formaldehyde": 0.5, "ammonia": 0.9}),
        MOxMaterial("ZnO", "n", 120e3, 0.50, 2.5, 10.0,
                    {"ethanol": 1.0, "acetone": 0.8, "toluene": 0.3, "formaldehyde": 0.9, "ammonia": 0.5}),
        MOxMaterial("In2O3", "n", 30e3, 0.45, 2.0, 9.0,
                    {"ethanol": 0.9, "acetone": 0.6, "toluene": 0.4, "formaldehyde": 1.4, "ammonia": 0.6}),
        MOxMaterial("CuO", "p", 15e3, 0.40, 5.0, 18.0,
                    {"ethanol": 0.4, "acetone": 0.3, "toluene": 0.7, "formaldehyde": 0.5, "ammonia": 0.3}),
        MOxMaterial("NiO", "p", 80e3, 0.40, 5.0, 20.0,
                    {"ethanol": 0.3, "acetone": 0.4, "toluene": 0.6, "formaldehyde": 0.6, "ammonia": 0.5}),
    )
}


@dataclass
class MOxSensor:
    material: MOxMaterial
    operating_temp_c: float
    sensing_area: SensingArea = field(default_factory=SensingArea)

    @property
    def label(self) -> str:
        return f"{self.material.name}@{self.operating_temp_c:.0f}C/{self.sensing_area.porous_material.name}"

    @property
    def temp_k(self) -> float:
        return self.operating_temp_c + 273.15

    def sensitivity_vector(self, temp_width_c: float = 100.0) -> np.ndarray:
        """Per-VOC coefficient a, including the bell-shaped heater-temperature selectivity."""
        return np.array([
            self.material.sensitivity.get(v, 0.0)
            * np.exp(-((self.operating_temp_c - VOC_LIBRARY[v].t_opt_c) / temp_width_c) ** 2)
            for v in VOC_NAMES
        ])


class MOxSensorArray:
    """Vectorised simulation of N heated MOx chemiresistors.

    Each sensor reacts to the concentration its porous surface delivers (see PorousSurface).
    """

    def __init__(self, sensors: list[MOxSensor], noise_rel: float = 0.003,
                 drift_rel_per_sqrt_s: float = 2e-4, seed: int | None = None):
        self.sensors = sensors
        self.noise_rel = noise_rel
        self.drift = drift_rel_per_sqrt_s
        self.rng = np.random.default_rng(seed)

        self.r0 = np.array([s.material.r0_ohm for s in sensors])
        self.sign = np.array([1.0 if s.material.carrier == "n" else -1.0 for s in sensors])
        self.beta = np.array([s.material.beta for s in sensors])[:, None]
        self.tau_rise = np.array([s.material.tau_rise_s for s in sensors])
        self.tau_decay = np.array([s.material.tau_decay_s for s in sensors])
        self.a = np.stack([s.sensitivity_vector() for s in sensors])  # (N, V)
        self.reset()

    @property
    def n(self) -> int:
        return len(self.sensors)

    def reset(self) -> None:
        self.s = np.zeros(self.n)  # surface response state
        self.log_drift = np.zeros(self.n)

    def step(self, effective_ppm: np.ndarray, dt: float) -> np.ndarray:
        """Advance by dt seconds for the (N, V) concentrations delivered to each sensor; returns resistance (Ω)."""
        # 1. MOx surface reaction: power-law response with asymmetric adsorption/desorption kinetics
        s_target = np.sum(self.a * np.maximum(effective_ppm, 0.0) ** self.beta, axis=1)
        tau = np.where(s_target > self.s, self.tau_rise, self.tau_decay)
        self.s += (1.0 - np.exp(-dt / tau)) * (s_target - self.s)

        # 2. Baseline drift (random walk in log R0) and read-out noise
        self.log_drift += self.drift * np.sqrt(dt) * self.rng.standard_normal(self.n)
        r_base = self.r0 * np.exp(self.log_drift)
        r = np.where(self.sign > 0, r_base / (1.0 + self.s), r_base * (1.0 + self.s))
        return r * (1.0 + self.noise_rel * self.rng.standard_normal(self.n))

    def features(self, resistance: np.ndarray, s_saturation: float = 40.0) -> np.ndarray:
        """Normalised log-response in [0, 1] relative to the calibrated clean-air baseline."""
        x = self.sign * np.log(self.r0 / resistance) / np.log1p(s_saturation)
        return np.clip(x, 0.0, 1.0)

    def concentration_estimate(self, resistance: np.ndarray) -> np.ndarray:
        """Per-sensor concentration read-out (ppm) by inverting the calibrated power law.

        Each sensor is calibrated against its most responsive VOC, so the value is a
        ppm-equivalent of that VOC as seen at the MOx grains (i.e. after porous-layer
        pre-concentration).
        """
        s = np.maximum(np.where(self.sign > 0, self.r0 / resistance, resistance / self.r0) - 1.0, 0.0)
        a_cal = self.a.max(axis=1)
        return (s / a_cal) ** (1.0 / self.beta[:, 0])


def default_sensor_array(porous: str = "ZIF-8") -> list[MOxSensor]:
    """12-channel array: 6 MOx chemistries × 2 heater temperatures, shared porous over-layer."""
    area = SensingArea(porous_material=POROUS_MATERIALS[porous])
    return [MOxSensor(MOX_MATERIALS[m], t, area)
            for m in MOX_MATERIALS for t in (275.0, 375.0)]


# ─────────────────────────────────────────────────────────────────────────────
# Spike-encoding interface and microLED implementation
# ─────────────────────────────────────────────────────────────────────────────

class SpikeEncoder(ABC):
    """Converts per-channel sensor signals into spike events.

    `input_kind` tells the ingestion layer what to feed `step`: "features"
    (normalised response, 0..1) or "concentration" (calibrated ppm read-out).
    """
    input_kind: str = "features"

    @abstractmethod
    def reset(self, n_channels: int) -> None: ...

    @abstractmethod
    def step(self, signal: np.ndarray, dt: float) -> np.ndarray:
        """Advance by dt seconds; return a boolean spike vector (one entry per channel)."""


@dataclass
class MicroLEDParams:
    diameter_um: float = 10.0
    wavelength_nm: float = 450.0       # InGaN blue
    eqe_peak_bulk: float = 0.35
    j_peak_a_cm2: float = 15.0         # current density at peak EQE
    sidewall_loss_um: float = 2.0      # size-dependent sidewall non-radiative recombination length
    i_bias_ua: float = 1.0             # drive current at zero response (keeps LED biased, sub-threshold)
    i_max_ua: float = 100.0            # drive current at full-scale response

    @property
    def area_cm2(self) -> float:
        return np.pi * (self.diameter_um * 1e-4 / 2.0) ** 2

    @property
    def eqe_peak(self) -> float:
        return self.eqe_peak_bulk / (1.0 + 4.0 * self.sidewall_loss_um / self.diameter_um)

    def optical_power_w(self, current_a: np.ndarray) -> np.ndarray:
        """L–I characteristic with efficiency droop: EQE(J) = EQE_pk * 2√(J/Jp) / (1 + J/Jp)."""
        j_ratio = np.maximum(current_a, 0.0) / self.area_cm2 / self.j_peak_a_cm2
        eqe = self.eqe_peak * 2.0 * np.sqrt(j_ratio) / (1.0 + j_ratio)
        photon_energy_ev = PLANCK_C_EV_NM / self.wavelength_nm
        return eqe * photon_energy_ev * current_a


@dataclass
class PhotodetectorParams:
    responsivity_a_w: float = 0.2
    coupling_efficiency: float = 0.5
    dark_current_na: float = 1.0


@dataclass
class LIFParams:
    tau_m_ms: float = 50.0
    v_threshold: float = 0.5
    v_reset: float = 0.0
    refractory_ms: float = 1.0
    transimpedance_ohm: float = 1e6
    noise_v: float = 0.01


class MicroLEDSpikeEncoder(SpikeEncoder):
    """Opto-electronic spike encoder.

    Each channel's sensor feature sets the drive current of a dedicated microLED;
    its light is collected by a photodiode whose photocurrent charges a leaky
    integrate-and-fire neuron. Firing rate therefore grows monotonically with
    gas concentration, with a rheobase set by the LED bias point.
    """

    def __init__(self, led: MicroLEDParams | None = None, pd: PhotodetectorParams | None = None,
                 lif: LIFParams | None = None, seed: int | None = None):
        self.led = led or MicroLEDParams()
        self.pd = pd or PhotodetectorParams()
        self.lif = lif or LIFParams()
        self.rng = np.random.default_rng(seed)
        self.last_current_a = self.last_power_w = None

    def reset(self, n_channels: int) -> None:
        self.v = np.zeros(n_channels)
        self.refractory = np.zeros(n_channels)

    def drive_current_a(self, features: np.ndarray) -> np.ndarray:
        i_ua = self.led.i_bias_ua + np.clip(features, 0.0, 1.0) * (self.led.i_max_ua - self.led.i_bias_ua)
        return i_ua * 1e-6

    def photocurrent_a(self, optical_power_w: np.ndarray) -> np.ndarray:
        return (self.pd.responsivity_a_w * self.pd.coupling_efficiency * optical_power_w
                + self.pd.dark_current_na * 1e-9)

    def step(self, features: np.ndarray, dt: float) -> np.ndarray:
        p = self.lif
        current = self.drive_current_a(features)
        power = self.led.optical_power_w(current)
        v_inf = self.photocurrent_a(power) * p.transimpedance_ohm
        self.last_current_a, self.last_power_w = current, power

        decay = np.exp(-dt / (p.tau_m_ms * 1e-3))
        active = self.refractory <= 0.0
        noise = p.noise_v * np.sqrt(1.0 - decay**2) * self.rng.standard_normal(self.v.shape)
        self.v = np.where(active, v_inf + (self.v - v_inf) * decay + noise, self.v)
        self.refractory -= dt

        spikes = active & (self.v >= p.v_threshold)
        self.v[spikes] = p.v_reset
        self.refractory[spikes] = p.refractory_ms * 1e-3
        return spikes


@dataclass
class LinearMicroLEDParams:
    """Linear concentration → microLED → spike-frequency transfer.

    The concentration span [c_min, c_max] maps linearly onto the LED drive current
    [i_turn_on, i_max]; above turn-on the L–I curve is taken as linear (slope
    efficiency = EQE × photon energy, valid below the droop regime), and spike
    frequency is proportional to emitted optical power.
    """
    c_min_ppm: float = 0.0
    c_max_ppm: float = 50.0
    i_turn_on_ua: float = 0.5
    i_max_ua: float = 100.0
    slope_efficiency_w_a: float = 0.5
    f_min_hz: float = 0.0
    f_max_hz: float = 200.0

    def drive_current_a(self, conc_ppm: np.ndarray) -> np.ndarray:
        x = np.clip((np.asarray(conc_ppm, dtype=float) - self.c_min_ppm)
                    / (self.c_max_ppm - self.c_min_ppm), 0.0, 1.0)
        return (self.i_turn_on_ua + x * (self.i_max_ua - self.i_turn_on_ua)) * 1e-6

    def optical_power_w(self, current_a: np.ndarray) -> np.ndarray:
        return self.slope_efficiency_w_a * np.maximum(current_a - self.i_turn_on_ua * 1e-6, 0.0)


def encode_concentration_to_frequency(conc_ppm: np.ndarray,
                                      params: LinearMicroLEDParams | None = None) -> np.ndarray:
    """Map MOx concentration read-outs (ppm, any shape) to spike frequencies (Hz).

    concentration ─(linear)─► LED current ─(linear L–I)─► optical power ─(linear)─► frequency,
    so f = f_min + (f_max − f_min) · clip((C − c_min) / (c_max − c_min), 0, 1).
    """
    p = params or LinearMicroLEDParams()
    power = p.optical_power_w(p.drive_current_a(conc_ppm))
    p_full_scale = p.optical_power_w(np.array(p.i_max_ua * 1e-6))
    return p.f_min_hz + (p.f_max_hz - p.f_min_hz) * power / p_full_scale


class LinearMicroLEDRateEncoder(SpikeEncoder):
    """Rate encoder that fires each channel at `encode_concentration_to_frequency(C)`.

    "regular" emits evenly spaced spikes via a phase accumulator (an LED pulsed
    by a voltage-controlled oscillator); "poisson" emits spikes stochastically at
    the same mean rate.
    """
    input_kind = "concentration"

    def __init__(self, params: LinearMicroLEDParams | None = None, mode: str = "regular",
                 seed: int | None = None):
        if mode not in ("regular", "poisson"):
            raise ValueError("mode must be 'regular' or 'poisson'")
        self.params = params or LinearMicroLEDParams()
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.last_current_a = self.last_power_w = None

    def reset(self, n_channels: int) -> None:
        self.phase = np.zeros(n_channels)

    def step(self, conc_ppm: np.ndarray, dt: float) -> np.ndarray:
        self.last_current_a = self.params.drive_current_a(conc_ppm)
        self.last_power_w = self.params.optical_power_w(self.last_current_a)
        freq = encode_concentration_to_frequency(conc_ppm, self.params)
        if self.mode == "poisson":
            return self.rng.random(freq.shape) < freq * dt
        self.phase += freq * dt
        spikes = self.phase >= 1.0
        self.phase[spikes] -= 1.0
        return spikes


# ─────────────────────────────────────────────────────────────────────────────
# Gas delivery and the ingestion layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OdorPulse:
    t_start_s: float
    t_end_s: float
    concentrations_ppm: dict[str, float]


class GasDeliverySchedule:
    """Sequence of odor pulses in clean-air carrier, with first-order flow-cell mixing."""

    def __init__(self, pulses: list[OdorPulse], tau_mixing_s: float = 0.5):
        for p in pulses:
            unknown = set(p.concentrations_ppm) - set(VOC_NAMES)
            if unknown:
                raise ValueError(f"Unknown VOC(s): {sorted(unknown)}")
        self.pulses = pulses
        self.tau = tau_mixing_s
        self._c = np.zeros(len(VOC_NAMES))

    def reset(self) -> None:
        self._c[:] = 0.0

    def step(self, t: float, dt: float) -> np.ndarray:
        setpoint = np.zeros(len(VOC_NAMES))
        for p in self.pulses:
            if p.t_start_s <= t < p.t_end_s:
                for voc, ppm in p.concentrations_ppm.items():
                    setpoint[VOC_NAMES.index(voc)] += ppm
        self._c += (1.0 - np.exp(-dt / self.tau)) * (setpoint - self._c)
        return self._c.copy()

    def stream(self, duration_s: float, dt: float) -> Iterator[np.ndarray]:
        """Yield the delivered concentration every dt seconds, starting from clean air."""
        self.reset()
        for k in range(int(round(duration_s / dt))):
            yield self.step(k * dt, dt)


@dataclass
class SpikeTrain:
    """Final output of the ingestion layer: one binary spike raster per sensor channel."""
    raster: np.ndarray  # (T, N) bool, sampled on the encoder clock
    dt_s: float
    labels: list[str]

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.raster)) * self.dt_s

    @property
    def duration_s(self) -> float:
        return len(self.raster) * self.dt_s

    def events(self) -> np.ndarray:
        """Address-event representation: rows of (time_s, channel) for a downstream SNN."""
        idx_t, idx_ch = np.nonzero(self.raster)
        return np.column_stack([idx_t * self.dt_s, idx_ch])

    def spike_times(self) -> list[np.ndarray]:
        """Spike times (s) for each channel."""
        return [np.flatnonzero(col) * self.dt_s for col in self.raster.T]

    def firing_rates_hz(self, t0: float, t1: float) -> np.ndarray:
        mask = (self.t >= t0) & (self.t < t1)
        return self.raster[mask].sum(axis=0) / max(t1 - t0, 1e-12)


@dataclass
class IngestionResult:
    spike_train: SpikeTrain
    t_sensor: np.ndarray           # (T_s,)         sensor time base
    concentration_ppm: np.ndarray  # (T_s, V)       gas stream input
    effective_ppm: np.ndarray      # (T_s, N, V)    delivered to the MOx by the porous surface
    surface_coverage: np.ndarray   # (T_s, N, V)    fractional adsorption-site occupancy
    resistance_ohm: np.ndarray     # (T_s, N)
    features: np.ndarray           # (T_s, N)
    sensed_ppm: np.ndarray         # (T_s, N)       calibrated per-sensor concentration read-out
    led_power_w: np.ndarray        # (T_s, N)

    @property
    def t(self) -> np.ndarray:
        return self.spike_train.t

    @property
    def spikes(self) -> np.ndarray:
        return self.spike_train.raster

    @property
    def labels(self) -> list[str]:
        return self.spike_train.labels

    def events(self) -> np.ndarray:
        return self.spike_train.events()

    def firing_rates_hz(self, t0: float, t1: float) -> np.ndarray:
        return self.spike_train.firing_rates_hz(t0, t1)


def _encoder_substeps(dt_sensor_s: float, dt_encoder_s: float) -> int:
    substeps = dt_sensor_s / dt_encoder_s
    if abs(substeps - round(substeps)) > 1e-9 or round(substeps) < 1:
        raise ValueError("dt_sensor_s must be a positive integer multiple of dt_encoder_s")
    return int(round(substeps))


def ingest(gas_stream: Iterable[np.ndarray], surface: PorousSurface, array: MOxSensorArray,
           encoder: SpikeEncoder, dt_sensor_s: float = 0.01, dt_encoder_s: float = 0.0005) -> IngestionResult:
    """Main ingestion loop: turn a stream of gas readings into the final spike train.

    `gas_stream` yields one ppm vector (ordered as VOC_NAMES) every `dt_sensor_s`;
    it can be a scripted schedule (`GasDeliverySchedule.stream`) or a live
    acquisition generator of any length. For each sample, in order:

      1. adsorption – PorousSurface enriches / sieves the gas at every sensor
      2. sensing    – MOxSensorArray turns the delivered concentration into
                      resistance, then into the read-out the encoder expects
      3. encoding   – SpikeEncoder runs at `dt_encoder_s` for the sample's
                      duration, holding the sensor read-out constant

    All three stages are reset first, so each call is an independent recording.
    """
    substeps = _encoder_substeps(dt_sensor_s, dt_encoder_s)
    n = array.n
    surface.reset()
    array.reset()
    encoder.reset(n)

    record: dict[str, list[np.ndarray]] = {k: [] for k in (
        "conc", "effective", "coverage", "res", "feat", "sensed", "power", "spikes")}
    for conc in gas_stream:
        conc = np.asarray(conc, dtype=float)
        if conc.shape != (len(VOC_NAMES),):
            raise ValueError(f"each gas sample must have shape ({len(VOC_NAMES)},) ordered as {VOC_NAMES}")

        # 1. Adsorption: porous surface → concentration delivered to each sensor
        effective = surface.step(conc, dt_sensor_s)

        # 2. Sensing: MOx chemiresistors → resistance → encoder input
        res = array.step(effective, dt_sensor_s)
        feat = array.features(res)
        sensed = array.concentration_estimate(res)
        signal = sensed if encoder.input_kind == "concentration" else feat

        # 3. Encoding: spikes on the faster encoder clock
        block = np.empty((substeps, n), dtype=bool)
        for j in range(substeps):
            block[j] = encoder.step(signal, dt_encoder_s)
        led_power = getattr(encoder, "last_power_w", None)  # only optical encoders expose this

        for key, value in (("conc", conc), ("effective", effective), ("coverage", surface.theta.copy()),
                           ("res", res), ("feat", feat), ("sensed", sensed),
                           ("power", np.zeros(n) if led_power is None else led_power), ("spikes", block)):
            record[key].append(value)

    if not record["conc"]:
        raise ValueError("gas stream produced no samples")

    n_samples = len(record["conc"])
    return IngestionResult(
        spike_train=SpikeTrain(np.concatenate(record["spikes"]), dt_encoder_s,
                               [s.label for s in array.sensors]),
        t_sensor=np.arange(n_samples) * dt_sensor_s,
        concentration_ppm=np.stack(record["conc"]),
        effective_ppm=np.stack(record["effective"]),
        surface_coverage=np.stack(record["coverage"]),
        resistance_ohm=np.stack(record["res"]),
        features=np.stack(record["feat"]),
        sensed_ppm=np.stack(record["sensed"]),
        led_power_w=np.stack(record["power"]),
    )


class SensoryIngestionLayer:
    """Front end of the neuromorphic nose: gas → porous surface → MOx array → spike encoder.

    Bundles the three stages with their clocks; the chemistry is integrated at
    `dt_sensor_s` (slow surface kinetics) and the encoder at `dt_encoder_s`
    (membrane dynamics). See `ingest` for the per-sample sequence.
    """

    def __init__(self, array: MOxSensorArray, encoder: SpikeEncoder,
                 dt_sensor_s: float = 0.01, dt_encoder_s: float = 0.0005,
                 surface: PorousSurface | None = None):
        _encoder_substeps(dt_sensor_s, dt_encoder_s)
        self.array = array
        self.surface = surface or PorousSurface.from_sensors(array.sensors)
        self.encoder = encoder
        self.dt_sensor = dt_sensor_s
        self.dt_encoder = dt_encoder_s

    def ingest(self, gas_stream: Iterable[np.ndarray]) -> IngestionResult:
        return ingest(gas_stream, self.surface, self.array, self.encoder, self.dt_sensor, self.dt_encoder)

    def run(self, schedule: GasDeliverySchedule, duration_s: float) -> IngestionResult:
        return self.ingest(schedule.stream(duration_s, self.dt_sensor))


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PULSES = [
    OdorPulse(5.0, 15.0, {"ethanol": 20.0}),
    OdorPulse(35.0, 45.0, {"acetone": 20.0}),
    OdorPulse(65.0, 75.0, {"toluene": 10.0, "formaldehyde": 5.0}),
]
DEMO_DURATION_S = 95.0


ENCODERS = {
    "lif": lambda seed: MicroLEDSpikeEncoder(seed=seed),
    "linear": lambda seed: LinearMicroLEDRateEncoder(seed=seed),
    "linear-poisson": lambda seed: LinearMicroLEDRateEncoder(mode="poisson", seed=seed),
}


def build_layer(porous: str, seed: int, encoder: str = "lif") -> SensoryIngestionLayer:
    array = MOxSensorArray(default_sensor_array(porous), seed=seed)
    return SensoryIngestionLayer(array, ENCODERS[encoder](seed + 1))


def print_rate_table(result: IngestionResult, title: str) -> None:
    windows = [("clean air", 0.0, 5.0)] + [
        ("+".join(p.concentrations_ppm), p.t_start_s, p.t_end_s) for p in DEMO_PULSES
    ]
    print(f"\n{title}  (mean firing rate, Hz)")
    header = f"{'channel':<22}" + "".join(f"{name:>22}" for name, *_ in windows)
    print(header)
    print("─" * len(header))
    rates = np.stack([result.firing_rates_hz(t0, t1) for _, t0, t1 in windows], axis=1)
    for label, row in zip(result.labels, rates):
        print(f"{label:<22}" + "".join(f"{r:>22.1f}" for r in row))
    print(f"total spikes: {int(result.spikes.sum())}")


def plot_result(result: IngestionResult) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for i, voc in enumerate(VOC_NAMES):
        if result.concentration_ppm[:, i].any():
            axes[0].plot(result.t_sensor, result.concentration_ppm[:, i], label=voc)
    axes[0].set_ylabel("gas (ppm)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(result.t_sensor, result.resistance_ohm / result.resistance_ohm[0], lw=0.8)
    axes[1].set_ylabel("R / R0")
    axes[1].set_yscale("log")

    axes[2].plot(result.t_sensor, result.led_power_w * 1e6, lw=0.8)
    axes[2].set_ylabel("µLED power (µW)")

    t_ev, ch = result.events().T
    axes[3].scatter(t_ev, ch, s=0.3, marker="|", color="black")
    axes[3].set_yticks(range(len(result.labels)), result.labels, fontsize=7)
    axes[3].set_ylabel("spikes")
    axes[3].set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--porous", choices=POROUS_MATERIALS, default="ZIF-8",
                        help="porous over-layer applied to every sensing area")
    parser.add_argument("--encoder", choices=ENCODERS, default="lif",
                        help="lif: photodiode-driven LIF neuron; linear: linear ppm→frequency microLED mapping")
    parser.add_argument("--compare", action="store_true",
                        help="also run without a porous layer to show the adsorption enhancement")
    parser.add_argument("--surface-report", action="store_true",
                        help="print per-material adsorption enhancement and sensitivity gains, then exit")
    parser.add_argument("--plot", action="store_true", help="plot traces and spike raster (needs matplotlib)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.surface_report:
        print_surface_report()
        return

    result = build_layer(args.porous, args.seed, args.encoder).run(GasDeliverySchedule(DEMO_PULSES), DEMO_DURATION_S)
    print_rate_table(result, f"Porous layer: {args.porous}, encoder: {args.encoder}")

    if args.compare and args.porous != "none":
        bare = build_layer("none", args.seed, args.encoder).run(GasDeliverySchedule(DEMO_PULSES), DEMO_DURATION_S)
        print_rate_table(bare, "Porous layer: none (dense film)")

    if args.plot:
        plot_result(result)


if __name__ == "__main__":
    main()
