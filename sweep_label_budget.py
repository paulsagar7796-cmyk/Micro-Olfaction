"""
Label-budget sweep: how accuracy and the System 2 duty cycle scale with the number of labelled
calibration samples per gas per drift batch.

    python3 sweep_label_budget.py                       # budgets 5 10 20 50 100, 5 seeds
    python3 sweep_label_budget.py --budgets 5 20 --reuse

For every budget N (all with the safe split: a gas with fewer than 2 × N samples in a batch gives
half of them to calibration):
  SNN      benchmark_drift.py --align-ema --few-shot-samples N --safe-split (EMA α 0.99, LIF 7.5,
           frozen STDP), run as a subprocess; its per-sample predictions are saved and reused.
  SGD      benchmark_system2.py's Model A on the same features, splits and test samples.
  Cascade  SNN answer, unless its confidence is below --confidence-threshold, it predicts
           class 4 or class 5 (the confusable pair), or it is silent: then SGD answers. Duty cycle = share of test
           samples for which SGD woke up.

Larger budgets remove more samples from scoring, so each budget has its own test set. The second
table scores every budget on the samples that are test samples at *every* budget (the common test
set), so the budget effect is not mixed with a change of test set.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import benchmark_system2 as S2
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import gas_name

WEAK = gas_name(4)  # the hardest class (class 4)

OUT_DIR = Path("results/label_budget")


def run_snn(budgets: list[int], seeds: list[int], parallel: int, reuse: bool) -> dict[int, Path]:
    """Run benchmark_drift.py once per budget (at most `parallel` at a time); return prediction files."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {b: OUT_DIR / f"snn_predictions_b{b}.csv" for b in budgets}
    todo = [b for b in budgets if not (reuse and files[b].exists())]
    running: list[tuple[int, subprocess.Popen]] = []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < parallel:
            b = todo.pop(0)
            cmd = [sys.executable, "benchmark_drift.py", "--align-ema", "--ema-alpha", "0.99", "--few-shot-samples",
                   str(b), "--safe-split", "--seeds", *map(str, seeds), "--jobs", str(len(seeds)),
                   "--save-predictions", str(files[b]), "--out", str(OUT_DIR / f"snn_drift_b{b}.csv")]
            log = open(OUT_DIR / f"snn_b{b}.log", "w")
            running.append((b, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
            print(f"  SNN, {b} labels per gas: started [{time.time() - t0:.0f} s]", flush=True)
        time.sleep(5)
        for b, p in list(running):
            if p.poll() is not None:
                running.remove((b, p))
                if p.returncode != 0:
                    raise RuntimeError(f"SNN run for budget {b} failed; see {OUT_DIR / f'snn_b{b}.log'}")
                print(f"  SNN, {b} labels per gas: done [{time.time() - t0:.0f} s]", flush=True)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--parallel", type=int, default=2, help="SNN budgets run at the same time (5 processes each)")
    parser.add_argument("--reuse", action="store_true", help="reuse existing SNN prediction files")
    parser.add_argument("--system2-features", choices=["kinetics", "all"], default="kinetics",
                        help="System 2 input: the SNN's 32 channels or all 128 features")
    parser.add_argument("--label-buffer", choices=["none", "all"], default="none",
                        help="SGD fine-tunes on the current visit's labels or on every visit so far")
    parser.add_argument("--buffer-decay", type=float, default=1.0, help="weight of a visit k batches old = decay ** k")
    parser.add_argument("--out", type=Path, default=Path("results/label_budget_sweep.csv"))
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))

    print(f"Label-budget sweep: budgets {budgets} labels per gas per batch, seeds {args.seeds}; System 2: "
          f"{args.system2_features} features, label buffer {args.label_buffer}"
          f"{f' (decay {args.buffer_decay:g})' if args.label_buffer == 'all' else ''}", flush=True)
    files = run_snn(budgets, args.seeds, args.parallel, args.reuse)

    # Per (budget, seed, batch): test indices, truth and the three systems' predictions
    per: dict[tuple[int, int, int], dict] = {}
    for b in budgets:
        snn = S2.load_snn(files[b])
        s2_args = argparse.Namespace(ema_alpha=0.99, few_shot_samples=b, calibration_epochs=20,
                                     batch1="train-split", safe_split=True, features=args.system2_features,
                                     label_buffer=args.label_buffer, buffer_decay=args.buffer_decay)
        with warnings.catch_warnings():
            from sklearn.exceptions import ConvergenceWarning
            warnings.simplefilter("ignore", ConvergenceWarning)
            for seed in args.seeds:
                run = S2.run_seed(s2_args, seed)
                for batch, r in run.items():
                    if batch == 1:
                        continue
                    if set(snn[(seed, batch)]) != set(int(i) for i in r["index"]):
                        raise RuntimeError(f"SNN and SGD test sets differ (budget {b}, seed {seed}, batch {batch})")
                    pred, conf = zip(*(snn[(seed, batch)][int(i)] for i in r["index"]))
                    snn_pred, snn_conf = np.array(pred, dtype=object), np.array(conf)
                    hybrid, awake = S2.cascade(snn_pred, snn_conf, r["A"], args.confidence_threshold)
                    per[(b, seed, batch)] = {"index": np.asarray(r["index"]), "gas": r["gas"], "snn": snn_pred,
                                             "sgd": r["A"], "cascade": hybrid, "awake": awake}
        print(f"  SGD and cascade, {b} labels per gas: done", flush=True)

    systems = ["snn", "sgd", "cascade"]
    rows = []
    for (b, seed, batch), r in sorted(per.items()):
        common = set(r["index"])
        for other in budgets:
            common &= set(per[(other, seed, batch)]["index"])
        keep = np.isin(r["index"], list(common))
        row = {"budget": b, "seed": seed, "batch": batch, "test_samples": len(r["gas"]),
               "common_test_samples": int(keep.sum())}
        for s in systems:
            row[f"{s}_accuracy"] = float(np.mean(r[s] == r["gas"]))
            row[f"{s}_accuracy_common"] = float(np.mean(r[s][keep] == r["gas"][keep])) if keep.any() else np.nan
        row["duty_cycle"] = float(np.mean(r["awake"]))
        row["duty_cycle_common"] = float(np.mean(r["awake"][keep])) if keep.any() else np.nan
        row["class4_total"] = int(np.sum(r["gas"] == WEAK))
        for s in systems:
            row[f"{s}_class4_correct"] = int(np.sum((r["gas"] == WEAK) & (r[s] == WEAK)))
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def stat(budget: int, key: str) -> str:
        per_seed = [np.nanmean([r[key] for r in rows if r["budget"] == budget and r["seed"] == s]) for s in args.seeds]
        v = 100 * np.array(per_seed)
        return f"{v.mean():.1f}% ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}%"

    def class4(budget: int, s: str) -> str:
        sel = [r for r in rows if r["budget"] == budget]
        return f"{100 * sum(r[f'{s}_class4_correct'] for r in sel) / max(1, sum(r['class4_total'] for r in sel)):.0f}%"

    head = "| Labels per Gas | SNN Accuracy | SGD Accuracy | Cascade Accuracy | Cascade Duty Cycle |"
    rule = "|---|---|---|---|---|"
    print(f"\nMean over batches 2–10, mean ± std over {len(args.seeds)} seeds; each budget on its own test set\n")
    print(head + "\n" + rule)
    for b in budgets:
        print(f"| {b} | {stat(b, 'snn_accuracy')} | {stat(b, 'sgd_accuracy')} | {stat(b, 'cascade_accuracy')} "
              f"| {stat(b, 'duty_cycle')} |")
    print(f"\nSame, scored only on the common test set (samples that are test samples at every budget)\n")
    print(head + "\n" + rule)
    for b in budgets:
        print(f"| {b} | {stat(b, 'snn_accuracy_common')} | {stat(b, 'sgd_accuracy_common')} "
              f"| {stat(b, 'cascade_accuracy_common')} | {stat(b, 'duty_cycle_common')} |")
    print(f"\nClass 4 ({WEAK}) recall (own test sets, pooled): "
          + "; ".join(f"{b}: SNN {class4(b, 'snn')}, SGD {class4(b, 'sgd')}, cascade {class4(b, 'cascade')}"
                      for b in budgets))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
