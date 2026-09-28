"""
System 2+ in the sequential maintenance protocol: does pseudo-labelling survive continuous deployment?

    python3 sequential_system2plus.py          # about a minute on 10 cores

Protocol, as for System 2 in tune_system2.run and sweep_system2_budget.py: train on 80 % of Batch 1,
then run Batches 2–10 in order without resetting anything. The alignment statistics, the readout and
every pseudo-label update carry over from batch to batch. At each batch (a maintenance visit) the first
N samples per gas to arrive are labelled and calibrate the readout; the rest are predicted in stream
order and scored.

Conditions (same seeds, same label draws):
  System 2                          K = 1, no class weight, no pseudo-labels (as sweep_system2_budget.py)
  System 2+ without pseudo-labels   K = 3, class 4 (ethylene) × 2
  System 2+                         K = 3, ethylene × 2, streaming pseudo-labels (τ = 0.95, at most 50 per
                                    gas and batch, weight 0.1; each sample is predicted before any update
                                    it triggers)
All settings are the ones chosen on Batches 2–5 in Settings 1–2; nothing is tuned here.

Decision rule, fixed before running. Pseudo-labelling is kept for sequential deployment only if, at every
label budget:
  (a) the paired per-seed difference, System 2+ minus System 2+ without pseudo-labels, on Batches 6–10
      does not have its 95 % CI entirely below zero; and
  (b) the per-seed slope of that difference over Batches 2–10 does not have its 95 % CI entirely below
      zero (harm that builds up over maintenance cycles).
Otherwise the paper recommends disabling pseudo-labelling in sequential deployments.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_drift as D
import combine_system2 as C
import improve_system2 as I
import tune_system2 as T
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import load_batch, stratified_split
from sweep_system2_budget import T_975, TUNED

CONDITIONS = {"System 2": (1.0, False, False), "System 2+ without pseudo-labels": (3.0, True, False),
              "System 2+": (3.0, True, True)}
BUDGETS = [5, 10, 12, 15]
EXTRA_SEEDS = list(range(2001, 2011))


def run(budget: int, seed: int, k: float, cw: bool, pl: bool) -> list[dict]:
    """One sequential deployment; with k = 1 and neither option it is tune_system2.run(TUNED, …)."""
    from sklearn.linear_model import SGDClassifier
    b1 = load_batch(1)
    train_idx, _ = stratified_split(b1.gas, test_fraction=0.2, seed=seed)
    x1_train = T.raw_features(1, TUNED["features"])[train_idx]
    mu1 = x1_train.mean(axis=0)
    sigma1 = np.maximum(x1_train.std(axis=0), 1e-12)
    scale1, shift1 = 1.0 / sigma1, -mu1 / sigma1
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=TUNED["sgd_alpha"], random_state=seed,
                          class_weight={I.WEAK: C.CLASS_WEIGHT} if cw else None)
    model.fit(x1_train * scale1 + shift1, b1.gas[train_idx])

    a, mu, var = TUNED["ema_alpha"], mu1.copy(), sigma1 ** 2
    rng = np.random.default_rng([seed, 2])
    rows = []
    for b in T.DRIFT:
        batch = load_batch(b)
        x = T.raw_features(b, TUNED["features"])
        order = D.stream_order(seed, b, len(batch))
        calibration, test = D.fewshot_split(batch.gas, order, budget, safe=True)
        y = np.empty_like(x)
        for i in np.concatenate([calibration, test]):                  # calibration samples arrive first
            mu = a * mu + (1 - a) * x[i]
            var = a * var + (1 - a) * (x[i] - mu) ** 2
            sc = 1.0 / np.sqrt(var + 1e-8)
            y[i] = x[i] * sc + (-mu * sc)
        weight = np.full(len(calibration), k)
        for _ in range(TUNED["epochs"]):
            perm = rng.permutation(calibration)
            model.partial_fit(y[perm], batch.gas[perm], sample_weight=weight)
        if pl:
            pred, updates, correct = I.stream_pseudo(model, y, test, batch.gas, C.PSEUDO)
        else:
            pred, updates, correct = model.predict(y[test]), 0, 0
        truth = batch.gas[test]
        rows.append({"budget": budget, "seed": seed, "batch": b, "labels": len(calibration), "scored": len(test),
                     "accuracy": float(np.mean(pred == truth)),
                     "weak_recall": float(np.mean(pred[truth == I.WEAK] == I.WEAK)) if np.any(truth == I.WEAK) else np.nan,
                     "pseudo_updates": updates, "pseudo_correct": correct})
    return rows


def job(args: tuple[str, int, int]) -> list[dict]:
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
    name, budget, seed = args
    return [dict(r, condition=name) for r in run(budget, seed, *CONDITIONS[name])]


def self_check() -> None:
    """System 2 here must reproduce tune_system2.run prediction for prediction."""
    ref = T.run(TUNED, 12, 42)
    mine = run(12, 42, *CONDITIONS["System 2"])
    for r in mine:
        acc = float(np.mean(ref[r["batch"]]["pred"] == ref[r["batch"]]["gas"]))
        if acc != r["accuracy"]:
            raise RuntimeError(f"System 2 differs from tune_system2.run on batch {r['batch']}")
    print("self-check: System 2 reproduces tune_system2.run exactly", flush=True)


def ci(v: np.ndarray) -> tuple[float, float, float]:
    half = T_975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/sequential_system2plus.csv"))
    args = parser.parse_args()
    self_check()
    seeds = DEFAULT_SEEDS + EXTRA_SEEDS
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        df = pd.DataFrame([r for rs in pool.map(job, [(c, b, s) for c in CONDITIONS for b in BUDGETS for s in seeds])
                           for r in rs])
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} batch results [{time.time() - t0:.0f} s]; wrote {args.out}")

    sweep = pd.read_csv("results/system2_budget_sweep.csv", float_precision="round_trip")
    mine = df[df.condition == "System 2"].merge(sweep, on=["budget", "seed", "batch"], suffixes=("", "_sweep"))
    same = bool((mine.accuracy == mine.accuracy_sweep).all())
    print(f"System 2 matches results/system2_budget_sweep.csv exactly on {len(mine)} shared batch results: {same}")

    per_seed = lambda d, batches: 100 * d[d.batch.isin(batches)].groupby("seed").accuracy.mean().reindex(seeds).to_numpy()  # noqa: E731
    print(f"\nAccuracy, mean over {len(seeds)} seeds (95 % CI); validation = Batches 2–5, test = Batches 6–10")
    print(f"{'labels/gas':>10}  {'condition':<32}{'validation':>11}{'test':>8}{'95 % CI':>14}{'seeds ≥ 90 %':>14}"
          f"{'ethylene':>10}{'pseudo-labels (precision)':>28}")
    for b in BUDGETS:
        for name in CONDITIONS:
            d = df[(df.condition == name) & (df.budget == b)]
            val, (m, lo, hi) = per_seed(d, T.VALIDATION).mean(), ci(per_seed(d, T.TEST))
            test = d[d.batch.isin(T.TEST)]
            upd = test.pseudo_updates.sum()
            pseudo = f"{upd / len(seeds) / len(T.TEST):.0f}/batch ({100 * test.pseudo_correct.sum() / upd:.1f} %)" if upd else "—"
            print(f"{b:>10}  {name:<32}{val:>10.1f}%{m:>7.1f}%{f'{lo:.1f}–{hi:.1f}':>14}"
                  f"{f'{int((per_seed(d, T.TEST) >= 90).sum())}/{len(seeds)}':>14}{100 * test.weak_recall.mean():>9.1f}%{pseudo:>28}")

    print("\nPseudo-labels vs none (System 2+ minus System 2+ without pseudo-labels), paired over seeds")
    print(f"{'labels/gas':>10}{'test Δ (points)':>18}{'95 % CI':>16}{'slope over Batches 2–10':>26}{'95 % CI':>18}{'verdict':>12}")
    keep = True
    with_pl, without = (df[df.condition == n].set_index(["budget", "seed", "batch"]).accuracy
                        for n in ("System 2+", "System 2+ without pseudo-labels"))
    gap = (100 * (with_pl - without)).rename("gap").reset_index()
    for b in BUDGETS:
        g = gap[gap.budget == b]
        delta = g[g.batch.isin(T.TEST)].groupby("seed").gap.mean().to_numpy()
        slope = np.array([np.polyfit(s.batch, s.gap, 1)[0] for _, s in g.groupby("seed")])
        (dm, dlo, dhi), (sm, slo, shi) = ci(delta), ci(slope)
        harmful = dhi < 0 or shi < 0
        keep &= not harmful
        verdict = "harmful" if harmful else ("helps" if dlo > 0 else "neutral")
        print(f"{b:>10}{dm:>+17.2f}{f'{dlo:+.2f} to {dhi:+.2f}':>16}{f'{sm:+.2f} per batch':>26}"
              f"{f'{slo:+.2f} to {shi:+.2f}':>18}{verdict:>12}")

    print("\nPer batch, 12 labels per gas: accuracy (%) and pseudo-label precision")
    print(f"{'batch':>6}" + "".join(f"{n:>33}" for n in CONDITIONS) + f"{'pseudo-labels (precision)':>28}")
    for bt in T.DRIFT:
        d = df[(df.budget == 12) & (df.batch == bt)]
        cells = "".join(f"{100 * d[d.condition == n].accuracy.mean():>32.1f}%" for n in CONDITIONS)
        p = d[d.condition == "System 2+"]
        print(f"{bt:>6}{cells}{f'{p.pseudo_updates.mean():.0f} ({100 * p.pseudo_correct.sum() / max(p.pseudo_updates.sum(), 1):.1f} %)':>28}")

    print(f"\nDecision (rule fixed before running): pseudo-labelling "
          f"{'can stay on' if keep else 'should be disabled'} in sequential deployments.")


if __name__ == "__main__":
    main()
