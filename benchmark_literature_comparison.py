"""
Literature comparison on the UCI Gas Sensor Array Drift dataset: the tuned System 2
(tune_system2.py; chosen on Batches 2–5, nothing re-tuned here) under the standard protocols,
next to published results (literature/published_results.csv, transcribed from the cited tables).

    python3 benchmark_literature_comparison.py

Protocols (as defined in Zhang & Zhang 2015, following Vergara et al. 2012):
  Setting 1   train on all of Batch 1; test on each Batch K = 2…10 independently.
  Setting 2   train on all of Batch K−1; test on Batch K.
  In both, per target batch: T labelled target samples (T = 0, 20, 30, 50), then score the rest.
    ssa       the T samples chosen by our reimplementation of the source's representative
              sample selection (farthest-point sampling on the batch's 128 features scaled to
              (−1, 1); gas-agnostic), matching DAELM-S/T's budgets and selection
    balanced  T // (gases present) per gas, first to arrive in the stream (our calibration)
  Our adaptation per target batch: EMA alignment from the source's statistics over the target
  stream (labelled samples first), then `epochs` partial_fit passes over the T labelled samples.
  Labelled samples are excluded from our scores (the source does not say whether it excludes them).
  Native      our own sequential protocol (tune_system2.run): train on 80 % of Batch 1, carry the
              aligned, recalibrated model through Batches 2…10 with N labels per gas per batch.
Our results are mean over 5 seeds (stream order, SGD initialisation); published rows are as reported.
Averages are the mean of the per-batch accuracies (as in the source tables).
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

import benchmark_drift as D
import tune_system2 as T
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import load_batch
from sweep_system2_budget import TUNED

TARGETS = list(range(2, 11))
BUDGETS = [0, 20, 30, 50]
NATIVE_PER_GAS = [5, 10, 15]
PUBLISHED = Path("literature/published_results.csv")


@lru_cache(maxsize=None)
def ssa_order(batch_id: int, k_max: int = 50) -> np.ndarray:
    """Farthest-point sampling (the source's Algorithm 3): the two most distant samples, then repeatedly
    the sample whose nearest selected sample is farthest. Features: the batch's 128 features in (−1, 1)."""
    x = raw = T.raw_features(batch_id, "all")
    lo, hi = raw.min(axis=0), raw.max(axis=0)
    x = 2 * (raw - lo) / np.maximum(hi - lo, 1e-12) - 1
    sq = np.sum(x * x, axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0))
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    chosen = [int(i), int(j)]
    nearest = np.minimum(dist[i], dist[j])
    while len(chosen) < min(k_max, len(x)):
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, dist[nxt])
    return np.array(chosen)


def balanced(gas: np.ndarray, order: np.ndarray, total: int) -> np.ndarray:
    present = list(dict.fromkeys(gas[order]))                      # gases by first arrival
    quota = {g: total // len(present) + (1 if n < total % len(present) else 0) for n, g in enumerate(present)}
    taken, seen = [], {}
    for i in order:
        seen[gas[i]] = seen.get(gas[i], 0) + 1
        if seen[gas[i]] <= quota[gas[i]]:
            taken.append(i)
    return np.array(taken, dtype=int)


def add_classes(model, classes: list[str]) -> None:
    """Give a fitted SGDClassifier untrained rows for classes its training data lacked, so that
    calibration labels can teach them (a new row starts at zero weights, bias below all others)."""
    missing = sorted(set(classes) - set(model.classes_))
    if not missing:
        return
    all_classes = np.array(sorted(set(model.classes_) | set(missing)), dtype=object)
    coef = np.zeros((len(all_classes), model.coef_.shape[1]))
    intercept = np.full(len(all_classes), model.intercept_.min() - 1.0)
    for row, c in enumerate(model.classes_):
        dst = int(np.flatnonzero(all_classes == c)[0])
        coef[dst], intercept[dst] = model.coef_[row], model.intercept_[row]
    model.classes_, model.coef_, model.intercept_ = all_classes, coef, intercept


def adapt(job: tuple[int, int, int, str, int]) -> tuple[tuple, dict]:
    """One target batch under Setting 1 or 2: train on the source, align, calibrate on `budget` labels, score the rest."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import SGDClassifier

    setting, target, budget, selection, seed = job
    cfg = TUNED
    source = 1 if setting == 1 else target - 1
    xs, ys = T.raw_features(source, cfg["features"]), load_batch(source).gas
    mu, sigma = xs.mean(axis=0), np.maximum(xs.std(axis=0), 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=cfg["sgd_alpha"], random_state=seed)
        model.fit((xs - mu) / sigma, ys)
        tb = load_batch(target)
        x = T.raw_features(target, cfg["features"])
        order = D.stream_order(seed, target, len(tb))
        if budget == 0:
            labelled = np.array([], dtype=int)
        elif selection == "ssa":
            picks = set(ssa_order(target)[:budget].tolist())
            labelled = np.array([i for i in order if i in picks], dtype=int)
        else:
            labelled = balanced(tb.gas, order, budget)
        rest = order[~np.isin(order, labelled)]
        a, m, v = cfg["ema_alpha"], mu.copy(), sigma ** 2
        y = np.empty_like(x)
        for i in np.concatenate([labelled, rest]):
            m = a * m + (1 - a) * x[i]
            v = a * v + (1 - a) * (x[i] - m) ** 2
            y[i] = (x[i] - m) / np.sqrt(v + 1e-8)
        if len(labelled):
            add_classes(model, list(tb.gas[labelled]))
            rng = np.random.default_rng([seed, target, budget])
            for _ in range(cfg["epochs"]):
                perm = rng.permutation(labelled)
                model.partial_fit(y[perm], tb.gas[perm])
        pred = model.predict(y[rest])
    return job, {"accuracy": float(np.mean(pred == tb.gas[rest])), "scored": len(rest), "labelled": len(labelled)}


def native(job: tuple[int, int]) -> tuple[tuple, dict]:
    per_gas, seed = job
    r = T.job((TUNED, per_gas, seed))[3]
    return job, {b: {"accuracy": float(np.mean(r[b]["pred"] == r[b]["gas"])),
                     "labelled": len(load_batch(b)) - len(r[b]["index"])} for b in TARGETS}


BATCH_SIZES = {1: 445, 2: 1244, 3: 1586, 4: 161, 5: 197, 6: 2300, 7: 3613, 8: 294, 9: 470, 10: 3600}


def load_published() -> list[dict]:
    """Published rows; checks every row's per-batch values against its reported average under the row's own
    averaging convention (tolerance: half the reported precision + 0.01). Missing batches are NaN."""
    with open(PUBLISHED, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        per_batch = [float(r[f"batch_{k}"]) if r[f"batch_{k}"] else np.nan for k in TARGETS]
        values = np.array(per_batch)
        have = ~np.isnan(values)
        convention = r.get("average_convention", "per-batch mean")
        if convention == "sample-weighted":
            w = np.array([BATCH_SIZES[k] for k in TARGETS])[have]
            expected = float(np.sum(values[have] * w) / w.sum())
        else:
            expected = float(np.mean(values[have]))
        reported = r["reported_average"]
        tolerance = 0.5 * 10 ** -(len(reported.split(".")[1]) if "." in reported else 0) + 0.01
        if convention != "differs from per-batch values" and abs(expected - float(reported)) > tolerance:
            raise ValueError(f"{r['method']} (setting {r['setting']}): per-batch values give {expected:.2f} "
                             f"({convention}), table reports {reported} — check the transcription")
        r["per_batch"] = per_batch
    return rows


def line(name: str, labels: str, per_batch: list[float], spread: str = "") -> str:
    avg, late = np.mean(per_batch), np.mean(per_batch[4:])
    return (f"{name:<34}{labels:>9}" + "".join(f"{v:>7.1f}" for v in per_batch)
            + f"{avg:>8.1f}{spread:>6}{late:>8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/literature_comparison.csv"))
    args = parser.parse_args()
    published = load_published()
    print(f"published rows: {len(published)} from {PUBLISHED}; every row's per-batch values reproduce its "
          f"reported average (±0.02)", flush=True)

    jobs = [(s, k, b, sel, seed) for s in (1, 2) for k in TARGETS for b in BUDGETS
            for sel in (["ssa", "balanced"] if b else ["none"]) for seed in args.seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        ours = dict(pool.map(adapt, jobs, chunksize=4))
        nat = dict(pool.map(native, [(n, s) for n in NATIVE_PER_GAS for s in args.seeds]))
    print(f"our runs done [{time.time() - t0:.0f} s]")

    header = (f"{'method':<34}{'labels':>9}" + "".join(f"{f'B{k}':>7}" for k in TARGETS)
              + f"{'avg 2–10':>9}{'±':>5}{'avg 6–10':>9}")
    rows_out = []
    for setting in (1, 2):
        print(f"\nSetting {setting}: " + ("train on Batch 1, test on each Batch K" if setting == 1
                                         else "train on Batch K−1, test on Batch K")
              + " — accuracy (%); labels = labelled target samples per batch")
        print(header)
        for r in (r for r in published if r["setting"] == str(setting) and r.get("year", "2015") in ("2012", "2015")):
            print(line(r["method"] + " [published]", r["target_labels_per_batch"].replace("not stated", "n/s"),
                       r["per_batch"]))
        for b in BUDGETS:
            for sel in (["ssa", "balanced"] if b else ["none"]):
                per_seed = np.array([[ours[(setting, k, b, sel, s)]["accuracy"] for k in TARGETS] for s in args.seeds])
                per_batch = 100 * per_seed.mean(axis=0)
                spread = f"{100 * per_seed.mean(axis=1).std(ddof=1):.1f}" if len(args.seeds) > 1 else ""
                name = {"none": "Ours: EMA alignment only", "ssa": "Ours, labels chosen as in DAELM",
                        "balanced": "Ours, labels balanced per gas"}[sel]
                print(line(name, str(b), list(per_batch), spread))
                rows_out.append({"setting": setting, "method": name, "labels_per_batch": b,
                                 **{f"batch_{k}": round(v, 2) for k, v in zip(TARGETS, per_batch)},
                                 "average_2_10": round(per_batch.mean(), 2), "average_6_10": round(per_batch[4:].mean(), 2)})
    print(f"\nNative sequential protocol (ours only; model carried and recalibrated batch after batch, "
          f"trained on 80 % of Batch 1)")
    print(header)
    for n in NATIVE_PER_GAS:
        per_seed = np.array([[nat[(n, s)][k]["accuracy"] for k in TARGETS] for s in args.seeds])
        labels = np.mean([nat[(n, s)][k]["labelled"] for s in args.seeds for k in TARGETS])
        per_batch = 100 * per_seed.mean(axis=0)
        print(line(f"Ours, {n} per gas", f"≈{labels:.0f}", list(per_batch),
                   f"{100 * per_seed.mean(axis=1).std(ddof=1):.1f}"))
        rows_out.append({"setting": "native", "method": f"Ours, {n} per gas", "labels_per_batch": round(labels, 1),
                         **{f"batch_{k}": round(v, 2) for k, v in zip(TARGETS, per_batch)},
                         "average_2_10": round(per_batch.mean(), 2), "average_6_10": round(per_batch[4:].mean(), 2)})
    print("\nPublished: Zhang & Zhang, IEEE Trans. Instrum. Meas. 2015 (arXiv:1505.06405), Tables II–III; "
          "'n/s' = label count not stated there. Ours: mean over seeds, ± = std of the 2–10 average.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
