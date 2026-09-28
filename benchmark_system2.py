"""
System 2 benchmark: classical classifiers on the EMA-aligned sensor features, run through the
same drift protocol as the SNN (benchmark_drift.py), plus a simulated System 1 → System 2 cascade.

    python3 benchmark_system2.py
    python3 benchmark_system2.py --simulate-cascade --snn-predictions results/snn_predictions_optC.csv

Harness (identical to `benchmark_drift.py --align-ema --few-shot-samples 5`, seed by seed):
  - Batch 1: the same stratified 80/20 split; models train on the 80% (or all of Batch 1 with
    --batch1 all), standardised with that split's per-channel mean and standard deviation.
  - Batches 2–10: the same stream order, the same calibration split (first N samples per gas),
    and the same running EMA alignment (α = --ema-alpha, calibration samples first), so every
    feature equals what the SNN's encoder receives before per-sample scaling and spikes.
  - Scoring on the same test samples; calibration samples are never scored.

Models (both also scored without calibration, as a reference):
  A  SGDClassifier(log_loss, l2, alpha=1e-4): after Batch 1, --calibration-epochs passes of
     partial_fit over each batch's calibration samples (weights carry over).
  B  RandomForestClassifier(100 trees, max_depth 12): refit each batch on a buffer of the Batch 1
     training data plus every calibration sample seen so far.

Options for System 2 (the SNN is unaffected):
  --features all           all 8 features per sensor (128 channels) instead of the SNN's 32
  --label-buffer all       Model A fine-tunes on every calibration sample collected so far, not only
                           the current visit's; --buffer-decay d weights a visit k batches old by d**k

Cascade (--simulate-cascade): System 1 is the saved SNN run (per-sample prediction and confidence,
the winning output neuron's share of output spikes). System 2 wakes up for a sample when the SNN's
confidence is below --confidence-threshold, when the SNN predicts class 4 or class 5 (the
confusable pair; names from dataset_adapter.gas_name), or when the SNN is silent; otherwise the SNN's answer stands. The duty cycle is
the share of test samples for which System 2 woke up.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
import warnings
from pathlib import Path

import numpy as np

import benchmark_drift as D
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import GAS_ORDER, gas_name, load_batch, stratified_split

FEATURE_MODE = "kinetics"
DRIFT_BATCHES = list(range(2, 11))
AMBIGUOUS = {gas_name(4), gas_name(5)}  # the confusable pair (classes 4 and 5)
MODELS = {"A": "SGD (linear)", "B": "Random forest"}
SWEEP = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]  # 0: ambiguity trigger only; 1.01: System 2 always on


def make_model(kind: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    if kind == "A":
        return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=seed)
    return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=1)


def run_seed(args: argparse.Namespace, seed: int) -> dict:
    """Train on Batch 1, then calibrate and score batch by batch. Returns per-batch test predictions."""
    fm = getattr(args, "features", FEATURE_MODE)
    ema_args = argparse.Namespace(ema_alpha=args.ema_alpha, feature_mode=fm)
    b1 = load_batch(1)
    train_idx, test_idx = stratified_split(b1.gas, test_fraction=0.2, seed=seed)
    mu, sigma = b1.subset(train_idx).channel_moments(fm)                # EMA start and Batch 1 scaling
    standardise = (1.0 / sigma, -mu / sigma)
    fit_idx = np.arange(len(b1)) if args.batch1 == "all" else train_idx
    x_train = b1.subset(fit_idx).transformed_channels(fm, standardise)
    y_train = b1.gas[fit_idx]

    splits, orders = {}, {}
    for b in DRIFT_BATCHES:
        gas = load_batch(b).gas
        splits[b] = D.fewshot_split(gas, D.stream_order(seed, b, len(gas)), args.few_shot_samples,
                                    safe=getattr(args, "safe_split", False))
        orders[b] = np.concatenate(splits[b])                             # calibration first, then test
    ema = D.ema_affines(ema_args, (mu, sigma ** 2), seed, DRIFT_BATCHES, orders)

    rng = np.random.default_rng([seed, 2])
    models = {k: make_model(k, seed).fit(x_train, y_train) for k in MODELS}
    frozen = {k: copy.deepcopy(m) for k, m in models.items()}
    buffer_x, buffer_y = [x_train], [y_train]
    visits_x, visits_y = [], []                                          # labelled calibration visits so far
    label_buffer, decay = getattr(args, "label_buffer", "none"), getattr(args, "buffer_decay", 1.0)

    x_held = b1.subset(test_idx).transformed_channels(fm, standardise)
    out = {1: {"index": test_idx, "gas": b1.gas[test_idx],
               **{k: models[k].predict(x_held) for k in MODELS},
               **{f"{k}_frozen": models[k].predict(x_held) for k in MODELS}}}
    for b in DRIFT_BATCHES:
        batch = load_batch(b)
        x = batch.channel_matrix(fm) * ema[b][0] + ema[b][1]             # per-sample EMA alignment
        calibration, test = splits[b]
        if label_buffer == "all":
            # Model A: partial_fit passes over every labelled visit so far, older visits down-weighted
            visits_x.append(x[calibration])
            visits_y.append(batch.gas[calibration])
            bx, by = np.vstack(visits_x), np.concatenate(visits_y)
            bw = np.concatenate([np.full(len(v), decay ** (len(visits_y) - 1 - k)) for k, v in enumerate(visits_y)])
            for _ in range(args.calibration_epochs):
                order = rng.permutation(len(by))
                models["A"].partial_fit(bx[order], by[order], sample_weight=bw[order])
        else:
            # Model A: partial_fit passes over the calibration samples (shuffled each pass)
            for _ in range(args.calibration_epochs):
                order = rng.permutation(calibration)
                models["A"].partial_fit(x[order], batch.gas[order])
        # Model B: refit on Batch 1 plus every calibration sample so far
        buffer_x.append(x[calibration])
        buffer_y.append(batch.gas[calibration])
        models["B"] = make_model("B", seed).fit(np.vstack(buffer_x), np.concatenate(buffer_y))
        out[b] = {"index": test, "gas": batch.gas[test],
                  **{k: models[k].predict(x[test]) for k in MODELS},
                  **{f"{k}_frozen": frozen[k].predict(x[test]) for k in MODELS}}
    return out


def load_snn(path: Path) -> dict:
    """Per (seed, batch): sample index → (prediction, confidence) from benchmark_drift --save-predictions."""
    snn: dict = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            snn.setdefault((int(r["seed"]), int(r["batch"])), {})[int(r["index"])] = (r["prediction"], float(r["confidence"]))
    return snn


def cascade(snn_pred: np.ndarray, snn_conf: np.ndarray, sys2_pred: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    awake = (snn_conf < threshold) | np.isin(snn_pred, list(AMBIGUOUS)) | (snn_pred == D.SILENT)
    return np.where(awake, sys2_pred, snn_pred), awake


def fmt(values, pct: bool = True) -> str:
    v = np.asarray(values, dtype=float) * (100 if pct else 1)
    return f"{v.mean():5.1f} ± {v.std(ddof=1):4.1f}" if len(v) > 1 else f"{v[0]:5.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ema-alpha", type=float, default=0.99)
    parser.add_argument("--few-shot-samples", type=int, default=5, help="calibration samples per gas per drift batch")
    parser.add_argument("--calibration-epochs", type=int, default=20, help="partial_fit passes for Model A")
    parser.add_argument("--features", choices=["kinetics", "all"], default="kinetics",
                        help="kinetics: the SNN's 32 channels; all: all 8 features per sensor (128 channels)")
    parser.add_argument("--label-buffer", choices=["none", "all"], default="none",
                        help="Model A fine-tunes on the current visit's labels only, or on every visit so far")
    parser.add_argument("--buffer-decay", type=float, default=1.0,
                        help="with --label-buffer all: weight of a visit k batches old = decay ** k")
    parser.add_argument("--safe-split", action="store_true",
                        help="a gas with fewer than 2 × N samples in a batch gives half of them to calibration")
    parser.add_argument("--batch1", choices=["train-split", "all"], default="train-split",
                        help="train on Batch 1's 80%% split (as the SNN) or on all of Batch 1")
    parser.add_argument("--simulate-cascade", action="store_true")
    parser.add_argument("--snn-predictions", type=Path, default=Path("results/snn_predictions_optC.csv"))
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--out", type=Path, default=Path("results/system2_benchmark.csv"))
    args = parser.parse_args()

    t0 = time.time()
    with warnings.catch_warnings():
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", ConvergenceWarning)
        runs = {seed: run_seed(args, seed) for seed in args.seeds}
    print(f"System 2 benchmark: {FEATURE_MODE} EMA-aligned features (α = {args.ema_alpha:g}), "
          f"{args.few_shot_samples}-shot calibration per gas, Batch 1 {args.batch1}; seeds {args.seeds} "
          f"[{time.time() - t0:.0f} s]")

    snn = load_snn(args.snn_predictions) if args.simulate_cascade else None
    rows, seeds = [], sorted(runs)
    variants = [f"{k}{suffix}" for k in MODELS for suffix in ("", "_frozen")]
    for seed in seeds:
        for b, r in runs[seed].items():
            row = {"seed": seed, "batch": b, "samples": len(r["gas"])}
            for v in variants:
                row[f"{v}_accuracy"] = float(np.mean(r[v] == r["gas"]))
                for g in GAS_ORDER:
                    row[f"{v}_{g}_correct"] = int(np.sum((r["gas"] == g) & (r[v] == g)))
            row.update({f"{g}_total": int(np.sum(r["gas"] == g)) for g in GAS_ORDER})
            if snn is not None and b != 1:
                pred, conf = zip(*(snn[(seed, b)][i] for i in r["index"]))  # KeyError = mismatched test sets
                r["snn"], r["snn_conf"] = np.array(pred, dtype=object), np.array(conf)
                row["snn_accuracy"] = float(np.mean(r["snn"] == r["gas"]))
                for k in MODELS:
                    hybrid, awake = cascade(r["snn"], r["snn_conf"], r[k], args.confidence_threshold)
                    row[f"cascade_{k}_accuracy"] = float(np.mean(hybrid == r["gas"]))
                    row[f"cascade_{k}_duty_cycle"] = float(np.mean(awake))
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

    drift_mean = lambda key: [np.mean([r[key] for r in rows if r["seed"] == s and r["batch"] != 1]) for s in seeds]
    recall = lambda v, g: sum(r[f"{v}_{g}_correct"] for r in rows if r["batch"] != 1) / max(
        1, sum(r[f"{g}_total"] for r in rows if r["batch"] != 1))
    held = lambda v: [r[f"{v}_accuracy"] for r in rows if r["batch"] == 1]
    print(f"\nMean accuracy, batches 2–10 (test samples only), mean ± std over {len(seeds)} seeds")
    print(f"{'model':<34}{'batches 2–10':>15}{'Batch 1 held-out':>19}")
    for k, name in MODELS.items():
        print(f"{name + ', calibrated':<34}{fmt(drift_mean(f'{k}_accuracy')):>15}{fmt(held(k)):>19}")
        print(f"{name + ', no calibration':<34}{fmt(drift_mean(f'{k}_frozen_accuracy')):>15}{fmt(held(k)):>19}")
    if snn is not None:
        print(f"{'SNN, Optimization C (System 1)':<34}{fmt(drift_mean('snn_accuracy')):>15}")

    print(f"\nRecall per gas over the drift batches (all seeds pooled)")
    print(f"{'model':<34}" + "".join(f"{g[:12]:>13}" for g in GAS_ORDER))
    for k, name in MODELS.items():
        for suffix, label in (("", "calibrated"), ("_frozen", "no calibration")):
            print(f"{name + ', ' + label:<34}" + "".join(f"{100 * recall(k + suffix, g):>12.1f}%" for g in GAS_ORDER))
    if snn is not None:
        snn_rec = {g: sum(np.sum((r["gas"] == g) & (r["snn"] == g)) for s in seeds for b, r in runs[s].items() if b != 1)
                   / max(1, sum(np.sum(r["gas"] == g) for s in seeds for b, r in runs[s].items() if b != 1)) for g in GAS_ORDER}
        print(f"{'SNN, Optimization C':<34}" + "".join(f"{100 * snn_rec[g]:>12.1f}%" for g in GAS_ORDER))

        print(f"\nCascade: System 2 wakes up when SNN confidence < threshold, the SNN predicts "
              f"{' or '.join(sorted(AMBIGUOUS))} (classes 4 and 5), or the SNN is silent")
        print(f"{'threshold':>14}" + "".join(f"{MODELS[k] + ' accuracy':>26}{'duty cycle':>13}" for k in MODELS))
        for t in sorted(set(SWEEP + [args.confidence_threshold])):
            cells = ""
            for k in MODELS:
                acc, duty = [], []
                for s in seeds:
                    per_batch, awake_all = [], []
                    for b, r in runs[s].items():
                        if b == 1:
                            continue
                        hybrid, awake = cascade(r["snn"], r["snn_conf"], r[k], t)
                        per_batch.append(np.mean(hybrid == r["gas"]))
                        awake_all.append(awake)
                    acc.append(np.mean(per_batch))
                    duty.append(np.mean(np.concatenate(awake_all)))
                cells += f"{fmt(acc):>26}{fmt(duty):>13}"
            label = "always on" if t > 1 else ("ambiguity only" if t == 0 else f"{t:g}")
            marker = "  ← --confidence-threshold" if abs(t - args.confidence_threshold) < 1e-9 else ""
            print(f"{label:>14}{cells}{marker}")
        print("Accuracy: mean over batches 2–10, then mean ± std over seeds. Duty cycle: share of test samples "
              "for which System 2 woke up.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
