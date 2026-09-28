"""
Label-budget sweep for the tuned System 2 (chosen on Batches 2–5 by tune_system2.py; nothing is
re-tuned here). Reports accuracy on the held-out test Batches 6–10 per budget, and the smallest
budget whose mean test accuracy is ≥ the target.

    python3 sweep_system2_budget.py                                   # budgets 8 10 12 15 20, 5 seeds
    python3 sweep_system2_budget.py --extra-seeds 10                  # + 10 robustness seeds

For each budget: mean ± std over seeds, a 95 % t-interval over seeds, and how many individual
seeds reach the target. Seeds vary the Batch 1 split, the stream order (hence which samples are
labelled) and SGD's initialisation; the batches themselves are fixed, so the interval describes
protocol variation, not sampling of new data.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import tune_system2 as T
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import gas_name

WEAK = gas_name(4)  # the hardest class (class 4)

# Validation winner of tune_system2.py under the count-table gas naming (Batches 2–5 only)
TUNED = dict(T.DEFAULTS, features="all+ratio", sgd_alpha=1e-5, epochs=50, ema_alpha=0.995)
T_975 = {4: 2.776, 9: 2.262, 14: 2.145}  # two-sided 95 % t quantiles by degrees of freedom


def one(job: tuple[int, int]) -> tuple[int, int, dict]:
    budget, seed = job
    return budget, seed, T.job((TUNED, budget, seed))[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 10, 12, 15, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--extra-seeds", type=int, default=0, help="additional robustness seeds (2001, 2002, …)")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/system2_budget_sweep.csv"))
    args = parser.parse_args()
    groups = {"standard seeds": list(args.seeds)}
    if args.extra_seeds:
        groups[f"standard + {args.extra_seeds} extra seeds"] = list(args.seeds) + list(range(2001, 2001 + args.extra_seeds))
    all_seeds = sorted(set(s for g in groups.values() for s in g))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        runs = {(b, s): r for b, s, r in pool.map(one, [(b, s) for b in args.budgets for s in all_seeds])}
    print(f"Tuned System 2: {T.describe(TUNED)}; budgets {args.budgets} [{time.time() - t0:.0f} s]")

    acc = lambda b, s, batches: float(np.mean([np.mean(runs[(b, s)][bb]["pred"] == runs[(b, s)][bb]["gas"])
                                                for bb in batches]))
    rows = [{"budget": b, "seed": s, "batch": bb, "samples": len(runs[(b, s)][bb]["gas"]),
             "accuracy": float(np.mean(runs[(b, s)][bb]["pred"] == runs[(b, s)][bb]["gas"]))}
            for (b, s) in runs for bb in runs[(b, s)]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for name, seeds in groups.items():
        n = len(seeds)
        print(f"\n{name} ({n}): test = Batches 6–10 (held out); target {100 * args.target:g}%")
        print(f"{'labels/gas':>10}{'test mean ± std':>18}{'95% CI':>18}{'seeds ≥ target':>16}"
              f"{'all 2–10':>11}{f'{WEAK} (test)':>21}")
        minimum = None
        for b in args.budgets:
            test = np.array([acc(b, s, T.TEST) for s in seeds])
            half = T_975.get(n - 1, 1.96) * test.std(ddof=1) / np.sqrt(n)
            whole = np.array([acc(b, s, T.DRIFT) for s in seeds])
            ald = np.concatenate([runs[(b, s)][bb]["pred"][runs[(b, s)][bb]["gas"] == WEAK] == WEAK
                                  for s in seeds for bb in T.TEST]).mean()
            print(f"{b:>10}{f'{100 * test.mean():.1f} ± {100 * test.std(ddof=1):.1f}':>18}"
                  f"{f'{100 * (test.mean() - half):.1f}–{100 * (test.mean() + half):.1f}':>18}"
                  f"{f'{int(np.sum(test >= args.target))}/{n}':>16}{100 * whole.mean():>10.1f}%{100 * ald:>20.1f}%")
            if minimum is None and test.mean() >= args.target:
                minimum = (b, test.mean() - half >= args.target, int(np.sum(test >= args.target)), n)
        if minimum:
            b, ci_ok, k, n = minimum
            print(f"→ smallest budget with mean test accuracy ≥ {100 * args.target:g}%: {b} labels per gas "
                  f"({'whole 95% CI above target' if ci_ok else '95% CI still reaches below target'}; {k}/{n} seeds)")
        else:
            print(f"→ no budget reached a mean test accuracy of {100 * args.target:g}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
