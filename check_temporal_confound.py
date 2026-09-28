"""
Temporal-confound check (after Dennler et al. 2021) for the UCI Gas Sensor Array Drift dataset.

    python3 check_temporal_confound.py

Dennler et al. showed, on a related dataset with timestamps, that gases recorded in temporal clusters let a
classifier predict the gas from the sensor state before exposure. This dataset has no timestamps, so:

A. File order. Can a sample's position within its batch file predict the gas (5-NN on the index, 5-fold
   stratified cross-validation, against chance and a label-permutation null)? Is the order chronological,
   or sorted (by gas, then concentration)?
B. Pre-exposure state. Can a baseline resistance be recovered from the provided features? A true baseline
   must not depend on the current gas's concentration; each candidate formula is tested for that.
C. Does System 2+ lean on samples recorded close to its calibration samples? Concentration levels are
   swept in sessions, so for each gas half of its levels (random) are used for calibration and only the
   other levels are scored. Compared with the usual draw (all levels) on exactly the same scored samples,
   under Settings 1 and 2 at 20 and 50 balanced labels. Our pipeline never sees file order: every stream
   is a seeded random permutation.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_drift as D
import benchmark_literature_comparison as L
import combine_system2 as C
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import load_batch

TARGETS, TEST = list(range(2, 11)), [6, 7, 8, 9, 10]
BUDGETS = [20, 50]


def file_order(n_perm: int = 50) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    y = pd.read_csv("data/targets.csv")
    rows, rng = [], np.random.default_rng(0)
    for b in range(1, 11):
        d = y[y.batch == b].reset_index(drop=True)
        runs = (d.gas_id != d.gas_id.shift()).cumsum()
        steps = [np.diff(r.concentration_ppmv.to_numpy()) for _, r in d.groupby(runs) if len(r) > 1]
        sorted_share = np.mean(np.concatenate(steps) >= 0) if steps else np.nan
        idx, gas = np.arange(len(d)).reshape(-1, 1), d.gas.to_numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = lambda labels: cross_val_score(KNeighborsClassifier(5), idx, labels, cv=cv).mean()  # noqa: E731
        null = [acc(rng.permutation(gas)) for _ in range(n_perm)]
        rows.append({"batch": b, "samples": len(d), "same_gas_runs": int(runs.max()),
                     "concentration_sorted_within_runs": sorted_share, "index_5nn_accuracy": acc(gas),
                     "majority_share": d.gas.value_counts(normalize=True).max(), "chance_1_over_k": 1 / d.gas.nunique(),
                     "permutation_null_mean": np.mean(null), "permutation_null_95": np.percentile(null, 95)})
    return pd.DataFrame(rows)


def baseline_candidates() -> pd.DataFrame:
    X, y = pd.read_csv("data/features.csv"), pd.read_csv("data/targets.csv")
    dr = X[[f"DR_{s}" for s in range(1, 17)]].to_numpy()
    dn = X[[f"DR_norm_{s}" for s in range(1, 17)]].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cands = {"DR / DR_norm": dr / dn, "DR·DR_norm / (DR_norm − 1)": dr * dn / (dn - 1),
                 "DR / (DR_norm − 1)": dr / (dn - 1), "DR·DR_norm": dr * dn, "DR / (DR_norm + 1)": dr / (dn + 1)}
    rows = []
    for name, v in cands.items():
        v = np.where(np.isfinite(v) & (v > 0), v, np.nan)
        rho = []
        for b in range(1, 11):
            m = (y.batch == b).to_numpy()
            c = pd.Series(y.concentration_ppmv[m].to_numpy())
            rho.append(np.nanmedian([pd.Series(v[m][:, s]).corr(c, method="spearman") for s in range(16)]))
        rows.append({"candidate": name, "median_rho_with_concentration": np.nanmedian(rho),
                     "min_rho": np.nanmin(rho), "max_rho": np.nanmax(rho)})
    return pd.DataFrame(rows)


def heldout_levels(job: tuple[int, int, int]) -> list[dict]:
    warnings.simplefilter("ignore")
    setting, target, seed = job
    sources = [1] if setting == 1 else [target - 1]
    tb = load_batch(target)
    order = D.stream_order(seed, target, len(tb))
    rng = np.random.default_rng([seed, target, 777])
    calib_level = np.ones(len(tb), dtype=bool)        # sample's (gas, level) may provide calibration labels
    scorable = np.zeros(len(tb), dtype=bool)          # sample is at a held-out level of a multi-level gas
    for g in np.unique(tb.gas):
        members = tb.gas == g
        levels = np.unique(tb.concentration_ppmv[members])
        if len(levels) < 2:
            continue                                  # single-level gas: labels allowed, never scored
        chosen = rng.choice(levels, size=len(levels) // 2, replace=False)
        at_chosen = members & np.isin(tb.concentration_ppmv, chosen)
        calib_level[members & ~at_chosen] = False
        scorable[members & ~at_chosen] = True
    rows = []
    for budget in BUDGETS:
        usual = L.balanced(tb.gas, order, budget)
        held = L.balanced(tb.gas, order[calib_level[order]], budget)
        scored = scorable.copy()
        scored[usual] = False
        for name, lab in (("usual draw (all levels)", usual), ("held-out levels", held),
                          ("no labels", np.array([], dtype=int))):
            r = C.adapt_target(sources, target, seed, lab, budget)
            keep = scored[r["index"]]
            rows.append({"setting": setting, "target": target, "seed": seed, "budget": budget, "condition": name,
                         "scored": int(keep.sum()), "accuracy": float(np.mean(r["pred"][keep] == r["truth"][keep]))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/temporal_confound.csv"))
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    a, b = file_order(), baseline_candidates()
    a.to_csv(args.out.with_name(args.out.stem + "_file_order.csv"), index=False)
    b.to_csv(args.out.with_name(args.out.stem + "_baseline.csv"), index=False)
    print("A. File order within each batch (5-NN on the sample's position; 5-fold stratified CV)")
    print(a.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nB. Candidate baseline resistances from the provided features — a true pre-exposure baseline would show "
          "ρ ≈ 0 with the current concentration")
    print(b.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        c = pd.DataFrame([r for rs in pool.map(heldout_levels, [(s, t, seed) for s in (1, 2) for t in TARGETS
                                                              for seed in args.seeds]) for r in rs])
    c.to_csv(args.out, index=False)
    print("\nC. System 2+ (balanced labels): calibration from all concentration levels vs only from levels never "
          "scored, and no calibration — same scored samples, mean ± std over seeds. 'gain kept' = share of the "
          "usual draw's gain over no labels that survives with held-out levels")
    print(f"{'setting':>8}{'labels':>8}{'batches':>10}{'no labels':>14}{'usual draw':>16}{'held-out levels':>18}"
          f"{'change':>9}{'seeds lower':>13}{'gain kept':>11}{'scored/seed':>13}")
    for setting in (1, 2):
        for budget in BUDGETS:
            for name, targets in (("6–10", TEST), ("2–10", TARGETS)):
                d = c[(c.setting == setting) & (c.budget == budget) & c.target.isin(targets)]
                per = d.groupby(["condition", "seed"]).accuracy.mean().unstack(0) * 100
                u, h, z = per["usual draw (all levels)"], per["held-out levels"], per["no labels"]
                n = d[d.condition == "held-out levels"].groupby("seed").scored.sum().mean()
                kept = (h.mean() - z.mean()) / (u.mean() - z.mean())
                print(f"{setting:>8}{budget:>8}{name:>10}{f'{z.mean():.1f} ± {z.std():.1f}':>14}"
                      f"{f'{u.mean():.1f} ± {u.std():.1f}':>16}{f'{h.mean():.1f} ± {h.std():.1f}':>18}"
                      f"{h.mean() - u.mean():>+9.1f}{f'{int((h < u).sum())}/{len(u)}':>13}{kept:>10.0%}{n:>13,.0f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
