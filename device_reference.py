"""
NumPy reference of System 2+ as it runs on the microcontroller: the specification for the C port.
check_device_reference.py shows that it reproduces scikit-learn and the paper, and what single precision
changes.

What the device does
  per sample   take the 128 features (the dataset lists them sensor by sensor), reorder them by feature
               type with the EMAd signs flipped, add the 128 log-pattern features, align all 256 with
               running means and variances, and classify with six one-vs-rest logistic units; after a
               confident prediction (pseudo-labelling on), make one SGD update on the predicted label
  per visit    50 calibration passes over the visit's labelled samples, weight K = 3
The initial model (trained on Batch 1, or on the source batches) comes from scikit-learn on the host,
together with the initial running statistics and a few constants.

The readout reimplements scikit-learn 1.9.1's SGDClassifier(loss="log_loss", penalty="l2",
learning_rate="optimal").partial_fit, from linear_model/_sgd_fast.pyx.tp (_plain_sgd) and
_stochastic_gradient.py (fit_binary): one binary unit per class, weights stored as w·s with a lazily
updated scale s, step size η = 1 / (α (t₀ + t − 1)) with t shared by all units, class weights applied to
each unit's positive samples only, and each unit visiting a call's samples in its own order
(Fisher–Yates driven by scikit-learn's xorshift generator, with a per-unit seed that is the same in
every call).

Arithmetic (`Arith`):
  dt       float64 (the paper) or float32 (the Cortex-M4's FPU is single precision; every value and
           constant is then float32, as on the chip)
  style    "paper": as the paper's scripts compute (tune_system2.raw_features, each script's alignment
           formula, numpy's exp/log, BLAS dot products for predictions);
           "chip": as the C code computes (features from sensor-major rows, centred alignment,
           left-to-right sums, and in float32 the Cephes exp/log below)
  orders   calibration passes in numpy's permutation order, as in the paper's runs ("numpy"), or in
           the order of the chip's own xorshift Fisher–Yates shuffle ("device")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import expit

import benchmark_drift as D
import benchmark_literature_comparison as L
import combine_system2 as C
import improve_system2 as I
import tune_system2 as T
from dataset_adapter import FEATURE_MODES, FEATURE_SIGN, N_SENSORS, load_batch, stratified_split
from sweep_system2_budget import TUNED

MAX_INT = int(np.iinfo(np.int32).max)
GROUPS = FEATURE_MODES["all"]
N_RAW = len(GROUPS) * N_SENSORS
EPS = 1e-8
MAX_DLOSS = 1e12
EXPIT64_ZERO = -709.782712893384      # below this, float64's 1 / (1 + e^−x) is exactly 0


@dataclass(frozen=True)
class Arith:
    dt: type = np.float64
    style: str = "paper"
    orders: str = "numpy"

    @property
    def chip(self) -> bool:
        return self.style == "chip"

    def c(self, v: float):
        """A constant as the device holds it."""
        return self.dt(v)


PAPER64 = Arith()
CHIP32 = Arith(np.float32, "chip", "numpy")
CHIP32_DEVICE = Arith(np.float32, "chip", "device")


# ── float32 elementary functions (Cephes expf / logf; the C code performs the same operations) ──
EXP_C1, EXP_C2, LOG2E = 0.693359375, -2.12194440e-4, 1.44269504088896341
EXP_P = (1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1)
LOG_P = (7.0376836292e-2, -1.1514610310e-1, 1.1676998740e-1, -1.2420140846e-1, 1.4249322787e-1,
         -1.6668057665e-1, 2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1)
SQRTHF = 0.707106781186547524
EXP_LO, EXP_HI = -87.3365478515625, 88.0     # inputs clamped so that 2^n stays a normal float32


def expf(x: np.ndarray) -> np.ndarray:
    f = np.float32
    x = np.minimum(np.maximum(np.asarray(x, f), f(EXP_LO)), f(EXP_HI))
    fx = np.floor(x * f(LOG2E) + f(0.5))
    x = x - fx * f(EXP_C1)
    x = x - fx * f(EXP_C2)
    z = x * x
    y = np.full_like(x, f(EXP_P[0]))
    for c in EXP_P[1:]:
        y = y * x + f(c)
    y = y * z + x + f(1.0)
    return np.ldexp(y, fx.astype(np.int32)).astype(f)


def logf(x: np.ndarray) -> np.ndarray:
    """For x > 0 (the device only takes logs of floored, positive features)."""
    f = np.float32
    m, e = np.frexp(np.asarray(x, f))
    small = m < f(SQRTHF)
    e = (e - small).astype(f)
    m = np.where(small, m + m - f(1.0), m - f(1.0)).astype(f)
    z = m * m
    y = np.full_like(m, f(LOG_P[0]))
    for c in LOG_P[1:]:
        y = y * m + f(c)
    y = y * m * z
    y = y + f(EXP_C2) * e
    y = y + f(-0.5) * z
    m = m + y
    return (m + f(EXP_C1) * e).astype(f)


def seqsum(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Left-to-right sum, as a C loop accumulates."""
    return np.take(np.cumsum(a, axis=axis, dtype=a.dtype), -1, axis=axis)


# ── scikit-learn's xorshift generator and shuffle ──
def xorshift(state: int) -> tuple[int, int]:
    """sklearn.utils._random.our_rand_r: (value in [0, 2³¹), new state)."""
    s = state or 1
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s % 0x80000000, s


def fisher_yates(n: int, state: int) -> tuple[np.ndarray, int]:
    """Permutation of range(n) as SequentialDataset.shuffle computes it; returns it and the new state."""
    ind = list(range(n))
    for i in range(n - 1):
        r, state = xorshift(state)
        j = i + r % (n - i)
        ind[i], ind[j] = ind[j], ind[i]
    return np.array(ind, dtype=np.intp), state


@lru_cache(maxsize=None)
def unit_order(n: int, seed: int) -> np.ndarray:
    return fisher_yates(n, seed)[0]


def unit_seeds(random_state: int, n_units: int) -> tuple[int, ...]:
    """The shuffle seed of each one-vs-rest unit, as _fit_multiclass and fit_binary derive them."""
    out = []
    for s in np.random.RandomState(random_state).randint(MAX_INT, size=n_units):
        r = np.random.RandomState(s)
        r.randint(1, MAX_INT)                    # make_dataset's own seed (plain SGD does not use it)
        out.append(int(r.randint(MAX_INT)))
    return tuple(out)


class Readout:
    """Six one-vs-rest logistic units trained by plain SGD, as SGDClassifier.partial_fit."""

    def __init__(self, model, ar: Arith):
        dt, self.ar = ar.dt, ar
        self.classes = np.asarray(model.classes_)
        self.w = np.array(model.coef_, dtype=dt)
        self.b = np.array(model.intercept_, dtype=dt)
        self.t = dt(model.t_)
        pos = np.ones(len(self.classes))
        for c, weight in (model.class_weight or {}).items():
            pos[self.classes == c] = weight
        self.pos_weight = pos.astype(dt)
        self.seeds = unit_seeds(model.random_state, len(self.classes))
        typw = np.sqrt(1.0 / np.sqrt(model.alpha))                # log loss: η₀ = typw / max(1, ∂loss) = typw
        self.alpha, self.opt_init = dt(model.alpha), dt(1.0 / (typw * model.alpha))
        self.reset_below = dt(1e-9 if dt == np.float64 else 1e-6)

    def decision(self, x: np.ndarray) -> np.ndarray:
        if self.ar.chip:
            return seqsum(x[..., None, :] * self.w) + self.b
        return x @ self.w.T + self.b

    def _gradient(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """cgradient_half_binomial (y ∈ {0, 1})."""
        ar, one = self.ar, self.ar.c(1)
        exp = expf if (ar.chip and ar.dt == np.float32) else np.exp
        with np.errstate(over="ignore"):
            big = p > ar.c(-37)
            e = exp(np.where(big, -p, ar.c(0)))
            return np.where(big, ((one - y) - y * e) / (one + e), exp(np.where(big, ar.c(0), p)) - y).astype(ar.dt)

    def partial_fit(self, x: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray) -> None:
        ar, n, u = self.ar, len(x), len(self.classes)
        one, zero = ar.c(1), ar.c(0)
        orders = np.stack([unit_order(n, s) for s in self.seeds])                    # (units, n)
        positive = (np.asarray(labels)[orders] == self.classes[:, None]).astype(ar.dt)
        weight = np.asarray(sample_weight, dtype=ar.dt)[orders]
        rows = np.arange(u)
        w, b, t, scale = self.w, self.b, self.t, one
        for s in range(n):
            xs, y = x[orders[:, s]], positive[:, s]
            p = seqsum(w * xs) * scale + b
            eta = one / (self.alpha * (self.opt_init + t - one))
            dloss = np.clip(self._gradient(y, p), ar.c(-MAX_DLOSS), ar.c(MAX_DLOSS))
            update = -eta * dloss
            update = update * (np.where(y > zero, self.pos_weight, one) * weight[rows, s])
            scale = scale * max(zero, one - eta * self.alpha)
            if scale < self.reset_below:
                w *= scale
                scale = one
            nz = update != zero
            w[nz] += xs[nz] * (update[nz] / scale)[:, None]
            b[nz] += update[nz]
            t = t + one
        w *= scale
        self.t = self.t + ar.c(n)

    def probabilities_ok(self, dec: np.ndarray, k: int, tau: float) -> bool:
        """stream_pseudo's confidence test: normalised one-vs-rest probability of the winner ≥ τ.

        In drifted batches every unit's score can be hugely negative (−100 to −6000): the paper's
        float64 σ(score) values are then tiny but representable, and the winner's share is ≈ 1. In
        float32 they underflow alike, so the chip computes each unit's probability relative to the
        winner's, σ(dᵢ)/σ(m), which cannot underflow for the winner. Where float64's σ(m) itself is 0
        (m < −709.78), the paper's ratio is 0/0 and the test fails; the chip does the same."""
        ar = self.ar
        if ar.chip and ar.dt == np.float32:
            one, m = ar.c(1), dec[k]
            if m < ar.c(EXPIT64_ZERO):
                return False
            if m >= ar.c(0):
                terms = (one + expf(-m)) / (one + expf(-dec))
            else:
                em, a = expf(m), expf(dec - m)
                terms = a * (one + em) / (one + a * em)
            return bool(one / seqsum(terms) >= ar.c(tau))
        p = expit(dec)
        return bool(p[k] / p.sum() >= tau)


class Device:
    """Running alignment plus the readout: everything the chip holds between samples."""

    def __init__(self, model, mu: np.ndarray, var: np.ndarray, ar: Arith, form: str):
        self.ar, self.form = ar, form
        self.readout = Readout(model, ar)
        self.mu, self.var = mu.astype(ar.dt), var.astype(ar.dt)
        self.a = ar.c(TUNED["ema_alpha"])
        self.oma = ar.c(1) - self.a                             # as the scripts' (1 - a), in the device's precision

    def align(self, f: np.ndarray) -> np.ndarray:
        """Features in arrival order → aligned features; updates the running statistics."""
        ar, a, oma, eps = self.ar, self.a, self.oma, self.ar.c(EPS)
        mu, var, out = self.mu, self.var, np.empty_like(f)
        for j, x in enumerate(f):
            mu = a * mu + oma * x
            var = a * var + oma * (x - mu) ** 2
            if self.form == "affine":                           # tune_system2.run / sequential_system2plus.py
                sc = ar.c(1) / np.sqrt(var + eps)
                out[j] = x * sc + (-mu * sc)
            else:                                               # combine_system2.adapt_target, and the chip
                out[j] = (x - mu) / np.sqrt(var + eps)
        self.mu, self.var = mu, var
        return out

    def calibrate(self, y: np.ndarray, labels: np.ndarray, k: float, passes: list[np.ndarray]) -> None:
        weight = np.full(len(y), k, dtype=self.ar.dt)
        for pos in passes:
            self.readout.partial_fit(y[pos], labels[pos], weight)

    def classify(self, y: np.ndarray, truth: np.ndarray, pl: bool) -> tuple[np.ndarray, int, int]:
        r = self.readout
        if not pl:
            return r.classes[np.argmax(r.decision(y), axis=1)], 0, 0
        cfg, used = C.PSEUDO, dict.fromkeys(range(len(r.classes)), 0)
        frac = np.array([cfg["frac"]], dtype=self.ar.dt)
        pred, updates, correct = np.empty(len(y), dtype=object), 0, 0
        for j in range(len(y)):
            dec = r.decision(y[j])
            k = int(np.argmax(dec))
            pred[j] = r.classes[k]
            if r.probabilities_ok(dec, k, cfg["tau"]) and used[k] < cfg["cap"]:
                r.partial_fit(y[j:j + 1], r.classes[k:k + 1], frac)
                used[k] += 1
                updates += 1
                correct += int(r.classes[k] == truth[j])
        return pred, updates, correct


# ── features ──
@lru_cache(maxsize=None)
def sensor_rows(batch_id: int) -> np.ndarray:
    """(n, 128) as the dataset lists them: sensor 1's eight features, then sensor 2's, …"""
    f = load_batch(batch_id).features
    return f[[f"{g}_{s}" for s in range(1, N_SENSORS + 1) for g in GROUPS]].to_numpy(dtype=float)


def chip_features(raw: np.ndarray, ar: Arith) -> np.ndarray:
    dt, n = ar.dt, len(raw)
    sign = np.array([FEATURE_SIGN[g] for g in GROUPS], dtype=dt)
    x = (raw.astype(dt).reshape(n, N_SENSORS, len(GROUPS)) * sign).transpose(0, 2, 1)   # (n, type, sensor)
    floors = np.array([T._floors()[g] for g in GROUPS], dtype=dt)[:, None]
    log = logf if dt == np.float32 else np.log
    v = log(np.maximum(x, floors))
    ratio = v - (seqsum(v) / dt(N_SENSORS))[..., None]
    return np.hstack([x.reshape(n, N_RAW), ratio.reshape(n, N_RAW)]).astype(dt)


def features(batch_id: int, ar: Arith) -> np.ndarray:
    if ar.chip:
        return chip_features(sensor_rows(batch_id), ar)
    return T.raw_features(batch_id, TUNED["features"]).astype(ar.dt)


# ── calibration orders ──
class Passes:
    """Calibration pass orders: numpy's (as the paper's runs) or the chip's xorshift Fisher–Yates."""

    def __init__(self, ar: Arith, key: list[int]):
        self.numpy = ar.orders == "numpy"
        self.rng = np.random.default_rng(key)
        self.state = int(np.random.SeedSequence(key).generate_state(1)[0]) or 1

    def epochs(self, n: int, count: int) -> list[np.ndarray]:
        if self.numpy:
            return [self.rng.permutation(n) for _ in range(count)]
        out = []
        for _ in range(count):
            perm, self.state = fisher_yates(n, self.state)
            out.append(perm)
        return out


def _sgd(seed: int, cw: bool):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=TUNED["sgd_alpha"], random_state=seed,
                         class_weight={I.WEAK: C.CLASS_WEIGHT} if cw else None)


@dataclass
class Scenario:
    """What the host prepares and the device then sees: the initial model and alignment statistics,
    and per maintenance visit the batch, its labelled samples and the later samples, in arrival order."""
    model: object
    mu: np.ndarray
    var: np.ndarray
    paper_form: str                      # alignment formula of the script this scenario reproduces
    k: float
    pl: bool
    passes_key: list[int]
    visits: list[tuple[int, np.ndarray, np.ndarray]]


def sequential_scenario(budget: int, seed: int, k: float, cw: bool, pl: bool) -> Scenario:
    """sequential_system2plus.run: Batch 1's training split on the host, then Batches 2–10."""
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
    b1 = load_batch(1)
    train_idx, _ = stratified_split(b1.gas, test_fraction=0.2, seed=seed)
    x1 = T.raw_features(1, TUNED["features"])[train_idx]
    mu1, sigma1 = x1.mean(axis=0), np.maximum(x1.std(axis=0), 1e-12)
    model = _sgd(seed, cw).fit(x1 * (1.0 / sigma1) + (-mu1 / sigma1), b1.gas[train_idx])
    visits = []
    for b in T.DRIFT:
        batch = load_batch(b)
        visits.append((b, *D.fewshot_split(batch.gas, D.stream_order(seed, b, len(batch)), budget, safe=True)))
    return Scenario(model, mu1, sigma1 ** 2, "affine", k, pl, [seed, 2], visits)


def settings_scenario(setting: int, target: int, seed: int, budget: int, sel: str, k: float, cw: bool,
                      pl: bool) -> Scenario:
    """combine_system2.run_job for one configuration: the source batch on the host, then one visit."""
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
    source = 1 if setting == 1 else target - 1
    xs, ys = T.raw_features(source, TUNED["features"]), load_batch(source).gas
    mu, sigma = xs.mean(axis=0), np.maximum(xs.std(axis=0), 1e-12)
    model = _sgd(seed, cw).fit((xs - mu) / sigma, ys)
    tb = load_batch(target)
    order = D.stream_order(seed, target, len(tb))
    if budget == 0:
        labelled = np.array([], dtype=int)
    elif sel == "ssa":
        picks = set(L.ssa_order(target)[:budget].tolist())
        labelled = np.array([i for i in order if i in picks], dtype=int)
    else:
        labelled = L.balanced(tb.gas, order, budget)
    if len(labelled):
        L.add_classes(model, list(tb.gas[labelled]))
    return Scenario(model, mu, sigma ** 2, "centred", k, pl, [seed, target, budget],
                    [(target, labelled, order[~np.isin(order, labelled)])])


def run(sc: Scenario, ar: Arith, keep: bool = False) -> tuple[list[dict], Device]:
    """The device's side: per visit, align and store the labelled samples, calibrate, then classify."""
    dev = Device(sc.model, sc.mu, sc.var, ar, "centred" if ar.chip else sc.paper_form)
    passes = Passes(ar, sc.passes_key)
    rows = []
    for b, labelled, later in sc.visits:
        gas = load_batch(b).gas
        y = dev.align(features(b, ar)[np.concatenate([labelled, later])])
        dev.calibrate(y[:len(labelled)], gas[labelled], sc.k, passes.epochs(len(labelled), TUNED["epochs"]))
        pred, updates, correct = dev.classify(y[len(labelled):], gas[later], sc.pl)
        rows.append({"batch": b, "accuracy": float(np.mean(pred == gas[later])), "pseudo_updates": updates,
                     "pseudo_correct": correct, **({"pred": pred} if keep else {})})
    return rows, dev


def sequential(budget: int, seed: int, k: float, cw: bool, pl: bool, ar: Arith, keep: bool = False) -> list[dict]:
    """sequential_system2plus.run through the device reference."""
    rows, _ = run(sequential_scenario(budget, seed, k, cw, pl), ar, keep)
    return [dict(r, budget=budget, seed=seed) for r in rows]


def settings(setting: int, target: int, seed: int, budget: int, sel: str, k: float, cw: bool, pl: bool,
             ar: Arith, keep: bool = False) -> dict:
    """combine_system2.run_job for one configuration, through the device reference."""
    (row,), _ = run(settings_scenario(setting, target, seed, budget, sel, k, cw, pl), ar, keep)
    row.pop("batch")
    return dict(row, setting=setting, target=target, budget=budget, selection=sel, seed=seed)
