"""
SOLAR+ ("System 2+" in the code and result files) on arbitrary feature arrays: the functions the
twin-array test (system2_compounds.py) uses, moved unchanged from system2_home.py at tag pre-v4, where
they sit beside the home-recordings test. Here they need none of that test's code. The results they
produce are checked to regenerate byte for byte (validation/reproduce_validation.sh).
"""

from __future__ import annotations

import warnings

import numpy as np

import improve_system2 as I
from combine_system2 import PSEUDO
from sweep_system2_budget import TUNED

K_CAL, EPOCHS = 3.0, TUNED["epochs"]
SEEDS = [7, 42, 100, 999, 1234, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
CLASSES: tuple = ()             # callers pass their classes (system2_compounds.py: the four gases)


def system2plus(xs: np.ndarray, ys: np.ndarray, xt: np.ndarray, yt: np.ndarray, seed: int, labelled_per_class: int,
                pl: bool = True, align: bool = True, classes: tuple = CLASSES,
                calibration: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """combine_system2.adapt_target on these features: returns the scored target indices and predictions.
    The target rows must be in stream (chronological) order, unless `calibration` gives the calibration
    indices directly (system2_compounds.py: a spread-concentration set, not the stream's first arrivals)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import SGDClassifier
    import benchmark_literature_comparison as L
    warnings.simplefilter("ignore", ConvergenceWarning)
    mu, sigma = xs.mean(axis=0), np.maximum(xs.std(axis=0), 1e-12)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=TUNED["sgd_alpha"], random_state=seed).fit((xs - mu) / sigma, ys)
    if calibration is not None:
        labelled = np.asarray(calibration, dtype=int)
    else:
        labelled = []
        for c in classes:
            labelled += list(np.flatnonzero(yt == c)[:labelled_per_class])
        labelled = np.array(sorted(labelled), dtype=int)
    rest = np.array([i for i in range(len(yt)) if i not in set(labelled)], dtype=int)
    if not align:
        return rest, model.predict((xt[rest] - mu) / sigma)
    a, m, v = TUNED["ema_alpha"], mu.copy(), sigma ** 2
    y = np.empty_like(xt)
    for i in np.concatenate([labelled, rest]):
        m = a * m + (1 - a) * xt[i]
        v = a * v + (1 - a) * (xt[i] - m) ** 2
        y[i] = (xt[i] - m) / np.sqrt(v + 1e-8)
    if len(labelled):
        L.add_classes(model, list(yt[labelled]))
        rng = np.random.default_rng([seed, 362])
        weight = np.full(len(labelled), K_CAL)
        for _ in range(EPOCHS):
            perm = rng.permutation(labelled)
            model.partial_fit(y[perm], yt[perm], sample_weight=weight)
    pred = I.stream_pseudo(model, y, rest, yt, PSEUDO)[0] if pl else model.predict(y[rest])
    return rest, pred


def forest(xs: np.ndarray, ys: np.ndarray, xt: np.ndarray, seed: int, n_raw: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=seed).fit(xs[:, :n_raw], ys).predict(xt[:, :n_raw])


def scores(truth: np.ndarray, pred: np.ndarray, classes: tuple = CLASSES) -> dict:
    rec = {c: float(np.mean(pred[truth == c] == c)) for c in classes if np.any(truth == c)}
    return {"balanced_accuracy": float(np.mean(list(rec.values()))), "accuracy": float(np.mean(pred == truth)),
            **{f"recall_{c}": rec.get(c, np.nan) for c in classes}, "n": len(truth)}
