"""
System 2 improvements and tuning: CORAL alignment, concentration-independent features, smarter
label selection, and hyperparameters. Everything is chosen on Batches 2–5 (months 3–16) and
reported on Batches 6–10 (months 17–36), which no choice ever looks at.

    python3 tune_system2.py                    # stage 1, stage 2, final report (~10 minutes)

Pieces (all label-free except the calibration labels themselves):
  features   all        the 128 sensor features (current System 2)
             ratio      per feature type: log values minus their mean over the 16 sensors (the
                        pattern across sensors, independent of an overall concentration scale)
             all+ratio  both (256)
  alignment  ema        running per-channel mean and variance (current System 2)
             coral      running mean and covariance: in Batch-1-standardised units u, each sample
                        maps to  R₁^½ · S_t^−½ · (u − m_t),  where R₁ is Batch 1's covariance, m_t and
                        S_t running estimates (retention --ema-alpha / cov_alpha); both covariances
                        shrink toward their diagonal by λ. λ = 1 recovers the EMA alignment.
  selection  first      the first N samples of each gas in stream order (current System 2)
             spread     N samples per gas spread evenly across its concentrations in the batch
  model      SGDClassifier(log_loss, l2, alpha); calibration = `epochs` partial_fit passes.
             The safe split applies throughout (a gas with < 2N samples calibrates on half).

Stage 1: every features × alignment × selection combination at the current settings.
Stage 2: hyperparameters for the stage-1 winner (SGD alpha, epochs, EMA retention, then CORAL's).
Selection criterion: mean validation accuracy (Batches 2–5) at 5 and 20 labels per gas.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

import benchmark_drift as D
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import FEATURE_MODES, GAS_ORDER, gas_name, load_batch, stratified_split

DRIFT = list(range(2, 11))
VALIDATION, TEST = [2, 3, 4, 5], [6, 7, 8, 9, 10]
BUDGETS = [5, 20]
DEFAULTS = {"features": "all", "alignment": "ema", "selection": "first", "sgd_alpha": 1e-4, "epochs": 20,
            "ema_alpha": 0.99, "coral_lambda": 0.3, "cov_alpha": 0.995, "coral_every": 10}


@lru_cache(maxsize=None)
def _floors() -> dict[str, float]:
    """Per feature type: a small positive floor (1e-3 × Batch 1's median magnitude) before taking logs."""
    b1 = load_batch(1)
    return {g: 1e-3 * float(np.median(np.abs(b1.feature_group(g)))) for g in FEATURE_MODES["all"]}


@lru_cache(maxsize=None)
def raw_features(batch_id: int, kind: str) -> np.ndarray:
    batch = load_batch(batch_id)
    if kind == "all":
        return batch.channel_matrix("all")
    parts = []
    for g in FEATURE_MODES["all"]:
        v = np.log(np.maximum(batch.feature_group(g), _floors()[g]))
        parts.append(v - v.mean(axis=1, keepdims=True))
    ratio = np.hstack(parts)
    return ratio if kind == "ratio" else np.hstack([batch.channel_matrix("all"), ratio])


def spread_split(gas: np.ndarray, conc: np.ndarray, order: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Calibration = per gas, `per_class` samples at evenly spaced concentration ranks (safe split:
    half of a gas with < 2N samples); both sets returned in stream order."""
    position = np.empty(len(order), dtype=int)
    position[order] = np.arange(len(order))
    chosen = []
    for g in np.unique(gas):
        members = np.flatnonzero(gas == g)
        q = len(members) // 2 if len(members) < 2 * per_class else per_class
        if q == 0:
            continue
        ranked = members[np.lexsort((position[members], conc[members]))]
        picks = np.round((np.arange(q) + 0.5) * len(ranked) / q - 0.5).astype(int)
        chosen.extend(ranked[picks])
    chosen = np.array(sorted(chosen, key=lambda i: position[i]), dtype=int)
    rest = order[~np.isin(order, chosen)]
    return chosen, rest


def _shrink(c: np.ndarray, lam: float) -> np.ndarray:
    return (1 - lam) * c + lam * np.diag(np.diag(c))


def _sqrt_pair(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, 1e-10)
    return (v * np.sqrt(w)) @ v.T, (v / np.sqrt(w)) @ v.T


def run(cfg: dict, budget: int, seed: int) -> dict:
    """Train on Batch 1, then per drift batch: align (stream order, calibration first), calibrate on the
    labelled samples, predict the rest. Returns {batch: {"index", "gas", "pred"}} plus Batch 1 held-out."""
    from sklearn.linear_model import SGDClassifier

    b1 = load_batch(1)
    train_idx, test_idx = stratified_split(b1.gas, test_fraction=0.2, seed=seed)
    if cfg["features"] == "all":  # computed exactly as benchmark_system2 does (SGD amplifies last-bit differences)
        x1_train, x1_test = b1.subset(train_idx).channel_matrix("all"), b1.subset(test_idx).channel_matrix("all")
    else:
        x1_train, x1_test = raw_features(1, cfg["features"])[train_idx], raw_features(1, cfg["features"])[test_idx]
    mu1 = x1_train.mean(axis=0)
    sigma1 = np.maximum(x1_train.std(axis=0), 1e-12)
    scale1, shift1 = 1.0 / sigma1, -mu1 / sigma1
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=cfg["sgd_alpha"], random_state=seed)
    model.fit(x1_train * scale1 + shift1, b1.gas[train_idx])
    out = {1: {"index": test_idx, "gas": b1.gas[test_idx], "pred": model.predict(x1_test * scale1 + shift1)}}

    a = cfg["ema_alpha"]
    if cfg["alignment"] == "ema":
        mu, var = mu1.copy(), sigma1 ** 2
    else:
        u1 = x1_train * scale1 + shift1
        r1 = np.cov(u1, rowvar=False, bias=True)
        r1_sqrt = _sqrt_pair(_shrink(r1, cfg["coral_lambda"]))[0]
        m, s = np.zeros(len(mu1)), r1.copy()
        transform, seen = np.eye(len(mu1)), 0

    rng = np.random.default_rng([seed, 2])
    for b in DRIFT:
        batch = load_batch(b)
        x = raw_features(b, cfg["features"])
        order = D.stream_order(seed, b, len(batch))
        if cfg["selection"] == "first":
            calibration, test = D.fewshot_split(batch.gas, order, budget, safe=True)
        else:
            calibration, test = spread_split(batch.gas, batch.concentration_ppmv, order, budget)
        y = np.empty_like(x)
        for i in np.concatenate([calibration, test]):                  # calibration samples arrive first
            if cfg["alignment"] == "ema":
                mu = a * mu + (1 - a) * x[i]
                var = a * var + (1 - a) * (x[i] - mu) ** 2
                sc = 1.0 / np.sqrt(var + 1e-8)
                y[i] = x[i] * sc + (-mu * sc)
            else:
                u = x[i] * scale1 + shift1
                m = a * m + (1 - a) * u
                d = u - m
                s = cfg["cov_alpha"] * s + (1 - cfg["cov_alpha"]) * np.outer(d, d)
                if seen % cfg["coral_every"] == 0:
                    transform = r1_sqrt @ _sqrt_pair(_shrink(s, cfg["coral_lambda"]))[1]
                seen += 1
                y[i] = transform @ d
        for _ in range(cfg["epochs"]):
            perm = rng.permutation(calibration)
            model.partial_fit(y[perm], batch.gas[perm])
        out[b] = {"index": test, "gas": batch.gas[test], "pred": model.predict(y[test])}
    return out


def job(args: tuple[dict, int, int]) -> tuple[dict, int, int, dict]:
    cfg, budget, seed = args
    with warnings.catch_warnings():
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cfg, budget, seed, run(cfg, budget, seed)


def evaluate(configs: list[dict], seeds: list[int], pool: ProcessPoolExecutor) -> dict:
    """{(config key, budget, seed): run output} for every config × budget × seed."""
    results = {}
    for cfg, budget, seed, r in pool.map(job, [(c, b, s) for c in configs for b in BUDGETS for s in seeds]):
        results[(key(cfg), budget, seed)] = r
    return results


def key(cfg: dict) -> tuple:
    return tuple(sorted(cfg.items()))


def accuracy(results: dict, cfg: dict, budget: int, seeds: list[int], batches: list[int]) -> np.ndarray:
    """Per seed: mean over `batches` of batch accuracy."""
    return np.array([np.mean([np.mean(results[(key(cfg), budget, s)][b]["pred"] ==
                                      results[(key(cfg), budget, s)][b]["gas"]) for b in batches]) for s in seeds])


def val_score(results: dict, cfg: dict, seeds: list[int]) -> float:
    return float(np.mean([accuracy(results, cfg, b, seeds, VALIDATION).mean() for b in BUDGETS]))


def fmt(v: np.ndarray) -> str:
    return f"{100 * v.mean():5.1f} ± {100 * v.std(ddof=1):3.1f}" if len(v) > 1 else f"{100 * v[0]:5.1f}"


def describe(cfg: dict) -> str:
    diff = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in cfg.items() if DEFAULTS.get(k) != v]
    return ", ".join(diff) if diff else "baseline (current System 2)"


def self_check() -> None:
    """The default configuration must reproduce benchmark_system2's Model A prediction for prediction."""
    import benchmark_system2 as S2
    ref = S2.run_seed(argparse.Namespace(ema_alpha=0.99, few_shot_samples=20, calibration_epochs=20,
                                         batch1="train-split", safe_split=True, features="all",
                                         label_buffer="none", buffer_decay=1.0), 42)
    mine = run(dict(DEFAULTS), 20, 42)
    for b in [1] + DRIFT:
        if not (np.array_equal(ref[b]["index"], mine[b]["index"]) and np.array_equal(ref[b]["A"], mine[b]["pred"])):
            raise RuntimeError(f"default configuration differs from benchmark_system2 (batch {b})")
    lam1 = run(dict(DEFAULTS, alignment="coral", coral_lambda=1.0, cov_alpha=0.99, coral_every=1), 20, 42)
    agree = np.mean(np.concatenate([lam1[b]["pred"] == mine[b]["pred"] for b in DRIFT]))
    print(f"self-check: default configuration matches benchmark_system2 prediction for prediction; "
          f"CORAL with λ = 1 agrees with EMA on {agree:.1%} of predictions (equal up to rounding)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/system2_tuning.csv"))
    args = parser.parse_args()
    seeds = args.seeds
    self_check()
    t0 = time.time()
    results: dict = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        # Stage 1: structure at the current settings
        stage1 = [dict(DEFAULTS, features=f, alignment=a, selection=s)
                  for f, a, s in itertools.product(["all", "ratio", "all+ratio"], ["ema", "coral"], ["first", "spread"])]
        results.update(evaluate(stage1, seeds, pool))
        print(f"\nStage 1 [{time.time() - t0:.0f} s] — validation accuracy (Batches 2–5), mean ± std over "
              f"{len(seeds)} seeds")
        print(f"{'features':<11}{'alignment':<11}{'selection':<11}{'5 labels':>14}{'20 labels':>14}")
        for c in sorted(stage1, key=lambda c: -val_score(results, c, seeds)):
            print(f"{c['features']:<11}{c['alignment']:<11}{c['selection']:<11}"
                  + "".join(f"{fmt(accuracy(results, c, b, seeds, VALIDATION)):>14}" for b in BUDGETS))
        best1 = max(stage1, key=lambda c: val_score(results, c, seeds))

        # Stage 2a: SGD and EMA hyperparameters around the stage-1 winner
        stage2 = [dict(best1, sgd_alpha=al, epochs=ep, ema_alpha=ea) for al, ep, ea in
                  itertools.product([1e-5, 1e-4, 1e-3, 1e-2], [5, 20, 50], [0.98, 0.99, 0.995])]
        results.update(evaluate([c for c in stage2 if (key(c), BUDGETS[0], seeds[0]) not in results], seeds, pool))
        best2 = max(stage2, key=lambda c: val_score(results, c, seeds))
        # Stage 2b: CORAL settings, if CORAL won
        stage2b = []
        if best2["alignment"] == "coral":
            stage2b = [dict(best2, coral_lambda=l, cov_alpha=ca) for l, ca in
                       itertools.product([0.1, 0.3, 0.6, 1.0], [0.99, 0.995, 0.998])]
            results.update(evaluate([c for c in stage2b if (key(c), BUDGETS[0], seeds[0]) not in results], seeds, pool))
        tuned = max(stage2 + stage2b, key=lambda c: val_score(results, c, seeds))
        print(f"\nStage 2 [{time.time() - t0:.0f} s] — top 5 of {len(stage2) + len(stage2b)} settings by validation")
        for c in sorted(stage2 + stage2b, key=lambda c: -val_score(results, c, seeds))[:5]:
            print(f"  {100 * val_score(results, c, seeds):5.1f}%  {describe(c)}")

        baseline = dict(DEFAULTS)
        results.update(evaluate([c for c in (baseline,) if (key(c), BUDGETS[0], seeds[0]) not in results], seeds, pool))

    finalists = {"baseline (current System 2)": baseline, "stage-1 winner": best1, "tuned": tuned}
    print(f"\nFinal — test = Batches 6–10 (never used for any choice); mean ± std over {len(seeds)} seeds")
    print(f"{'':<30}{'labels':>7}{'validation 2–5':>17}{'test 6–10':>14}{'all 2–10':>14}{f'{gas_name(4)} (test)':>21}")
    for name, c in finalists.items():
        for b in BUDGETS:
            ald = [(r["pred"][r["gas"] == gas_name(4)] == gas_name(4)) for s in seeds
                   for bb, r in results[(key(c), b, s)].items() if bb in TEST]
            ald = np.concatenate(ald).mean()
            print(f"{name:<30}{b:>7}{fmt(accuracy(results, c, b, seeds, VALIDATION)):>17}"
                  f"{fmt(accuracy(results, c, b, seeds, TEST)):>14}{fmt(accuracy(results, c, b, seeds, DRIFT)):>14}"
                  f"{100 * ald:>20.1f}%")
    print(f"\nstage-1 winner: {describe(best1)}\ntuned: {describe(tuned)}")
    for b in BUDGETS:
        diff = accuracy(results, tuned, b, seeds, TEST) - accuracy(results, baseline, b, seeds, TEST)
        # same comparison on the samples that are test samples for both (label selection can change the test set)
        common = []
        for s in seeds:
            per_batch = []
            for bb in TEST:
                rt, rb = results[(key(tuned), b, s)][bb], results[(key(baseline), b, s)][bb]
                shared = np.intersect1d(rt["index"], rb["index"])
                at = np.mean(rt["pred"][np.isin(rt["index"], shared)] == rt["gas"][np.isin(rt["index"], shared)])
                ab = np.mean(rb["pred"][np.isin(rb["index"], shared)] == rb["gas"][np.isin(rb["index"], shared)])
                per_batch.append(at - ab)
            common.append(np.mean(per_batch))
        print(f"tuned vs baseline, test, {b} labels: {100 * diff.mean():+.1f} points (better on "
              f"{int(np.sum(diff > 0))}/{len(seeds)} seeds); on the common test samples {100 * np.mean(common):+.1f}")

    rows = []
    for (k, b, s), r in results.items():
        cfg = dict(k)
        for bb, rr in r.items():
            rows.append({**cfg, "budget": b, "seed": s, "batch": bb, "samples": len(rr["gas"]),
                         "accuracy": float(np.mean(rr["pred"] == rr["gas"]))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
