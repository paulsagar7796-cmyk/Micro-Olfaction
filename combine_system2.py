"""
Strong calibration combined with class weights and pseudo-labels, under the literature protocols.

    python3 combine_system2.py          # ~10 minutes on 10 cores

Same harness as improve_system2.py (Settings 1 and 2; 0, 20, 30, 50 labelled target samples per batch;
labels balanced per gas or chosen as in DAELM; validation = target Batches 2–5, test = 6–10).
Configurations: every combination of
  strong calibration  the labelled target samples' loss multiplied by K ∈ {1, 3, 10, 30, 100, 300} during
                      calibration (K = 1: current System 2); DAELM-S itself weights target errors 1000×
                      the source's (C_T = 10, C_S = 0.01)
  class weights       class 4 (ethylene) × 2 in the loss, or off
  pseudo-labels       τ = 0.95, cap 50 per class per batch, step × 0.1 (improve_system2.stream_pseudo), or off
K is chosen on validation accuracy separately for each on/off combination of the other two.
Pre-stated criterion: the combination "closes the gap" if it reaches DAELM-T(50)'s 87.0 % on
Batches 6–10 at 50 labels.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import benchmark_drift as D
import benchmark_literature_comparison as L
import improve_system2 as I
import tune_system2 as T
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import GAS_ORDER, load_batch
from sweep_system2_budget import TUNED

K_VALUES = [1, 3, 10, 30, 100, 300]
PSEUDO = {"tau": 0.95, "cap": 50, "frac": 0.1}
CLASS_WEIGHT = 2.0


def configs() -> list[dict]:
    return [{"k": k, "cw": cw, "pl": pl} for cw, pl, k in itertools.product([False, True], [False, True], K_VALUES)]


def name(cfg: dict) -> str:
    parts = [f"K={cfg['k']}"] + (["class weight"] if cfg["cw"] else []) + (["pseudo-labels"] if cfg["pl"] else [])
    return " + ".join(parts)


def run_job(job: tuple[int, int, int, tuple | None]) -> list[dict]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import SGDClassifier
    warnings.simplefilter("ignore", ConvergenceWarning)

    setting, target, seed, only = job
    source = 1 if setting == 1 else target - 1
    xs, ys = T.raw_features(source, TUNED["features"]), load_batch(source).gas
    mu, sigma = xs.mean(axis=0), np.maximum(xs.std(axis=0), 1e-12)
    us = (xs - mu) / sigma
    tb = load_batch(target)
    x = T.raw_features(target, TUNED["features"])
    order = D.stream_order(seed, target, len(tb))
    streams = {}
    for budget, sel in I.COMBOS:
        if budget == 0:
            labelled = np.array([], dtype=int)
        elif sel == "ssa":
            picks = set(L.ssa_order(target)[:budget].tolist())
            labelled = np.array([i for i in order if i in picks], dtype=int)
        else:
            labelled = L.balanced(tb.gas, order, budget)
        rest = order[~np.isin(order, labelled)]
        a, m, v = TUNED["ema_alpha"], mu.copy(), sigma ** 2
        y = np.empty_like(x)
        for i in np.concatenate([labelled, rest]):
            m = a * m + (1 - a) * x[i]
            v = a * v + (1 - a) * (x[i] - m) ** 2
            y[i] = (x[i] - m) / np.sqrt(v + 1e-8)
        streams[(budget, sel)] = (labelled, rest, y)

    source_models = {}
    for cw in (False, True):
        weights = {I.WEAK: CLASS_WEIGHT} if cw else None
        source_models[cw] = SGDClassifier(loss="log_loss", penalty="l2", alpha=TUNED["sgd_alpha"], random_state=seed,
                                          class_weight=weights).fit(us, ys)
    rows = []
    for cfg in configs():
        if only is not None and name(cfg) not in only:
            continue
        for budget, sel in I.COMBOS:
            if budget == 0 and cfg["k"] != 1:
                continue                                    # K only acts on labelled samples
            labelled, rest, y = streams[(budget, sel)]
            model = copy.deepcopy(source_models[cfg["cw"]])
            diverged = False
            try:
                if len(labelled):
                    L.add_classes(model, list(tb.gas[labelled]))
                    rng = np.random.default_rng([seed, target, budget])
                    weight = np.full(len(labelled), float(cfg["k"]))
                    for _ in range(TUNED["epochs"]):
                        perm = rng.permutation(labelled)
                        model.partial_fit(y[perm], tb.gas[perm], sample_weight=weight)
                updates = correct = 0
                if cfg["pl"]:
                    pred, updates, correct = I.stream_pseudo(model, y, rest, tb.gas, PSEUDO)
                else:
                    pred = model.predict(y[rest])
            except ValueError:                              # SGD overflow at very large K
                diverged, updates, correct = True, 0, 0
                pred = np.full(len(rest), "", dtype=object)
            truth = tb.gas[rest]
            recall = {g: float(np.mean(pred[truth == g] == g)) for g in GAS_ORDER if np.any(truth == g)}
            ks = K_VALUES if budget == 0 else [cfg["k"]]   # budget 0 is the same for every K
            for k in ks:
                rows.append({"config": name(dict(cfg, k=k)), "k": k, "cw": cfg["cw"], "pl": cfg["pl"], "setting": setting,
                             "target": target, "budget": budget, "selection": sel, "seed": seed,
                             "accuracy": np.nan if diverged else float(np.mean(pred == truth)),
                             "macro_recall": np.nan if diverged else float(np.mean(list(recall.values()))),
                             "weak_recall": np.nan if diverged else recall.get(I.WEAK, np.nan),
                             "scored": len(rest), "updates": updates, "pseudo_correct": correct, "diverged": diverged})
    return rows


def adapt_target(sources: list[int], target: int, seed: int, labelled: np.ndarray, rng_key: int,
                 k: float = 3.0, cw: bool = True, pl: bool = True) -> dict:
    """System 2+ on one target batch: train on the source batches (labelled), align the target stream
    (labelled samples first, in the given order, then the rest in stream order), calibrate on `labelled`
    and classify the rest. Returns the scored indices, predictions and truth."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import SGDClassifier
    warnings.simplefilter("ignore", ConvergenceWarning)
    xs = np.vstack([T.raw_features(b, TUNED["features"]) for b in sources])
    ys = np.concatenate([load_batch(b).gas for b in sources])
    mu, sigma = xs.mean(axis=0), np.maximum(xs.std(axis=0), 1e-12)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=TUNED["sgd_alpha"], random_state=seed,
                          class_weight={I.WEAK: CLASS_WEIGHT} if cw else None).fit((xs - mu) / sigma, ys)
    tb = load_batch(target)
    x = T.raw_features(target, TUNED["features"])
    order = D.stream_order(seed, target, len(tb))
    labelled = np.asarray(labelled, dtype=int)
    rest = order[~np.isin(order, labelled)]
    a, m, v = TUNED["ema_alpha"], mu.copy(), sigma ** 2
    y = np.empty_like(x)
    for i in np.concatenate([labelled, rest]):
        m = a * m + (1 - a) * x[i]
        v = a * v + (1 - a) * (x[i] - m) ** 2
        y[i] = (x[i] - m) / np.sqrt(v + 1e-8)
    if len(labelled):
        L.add_classes(model, list(tb.gas[labelled]))
        rng = np.random.default_rng([seed, target, rng_key])
        weight = np.full(len(labelled), float(k))
        for _ in range(TUNED["epochs"]):
            perm = rng.permutation(labelled)
            model.partial_fit(y[perm], tb.gas[perm], sample_weight=weight)
    pred = I.stream_pseudo(model, y, rest, tb.gas, PSEUDO)[0] if pl else model.predict(y[rest])
    return {"index": rest, "pred": pred, "truth": tb.gas[rest]}


def self_check() -> None:
    """K = 1 must reproduce the literature comparison (off, off) and improve_system2 (each on alone) exactly."""
    import pandas as pd
    prev = pd.read_csv("results/system2_improvements.csv")
    for setting, target, seed in ((1, 4, 42), (2, 6, 7)):
        rows = run_job((setting, target, seed, ("K=1", "K=1 + class weight", "K=1 + pseudo-labels")))
        got = {(r["config"], r["budget"], r["selection"]): r["accuracy"] for r in rows}
        for budget, sel in I.COMBOS:
            ref = L.adapt((setting, target, budget, sel, seed))[1]["accuracy"]
            assert got[("K=1", budget, sel)] == ref, ("base", setting, target, budget, sel)
            for mine, theirs in (("K=1 + class weight", "class_weight(w4=2)"),
                                 ("K=1 + pseudo-labels", "pseudo(tau=0.95, cap=50, frac=0.1)")):
                p = prev[(prev.config == theirs) & (prev.setting == setting) & (prev.target == target)
                         & (prev.budget == budget) & (prev.selection == sel) & (prev.seed == seed)].accuracy.iloc[0]
                assert abs(got[(mine, budget, sel)] - p) < 1e-12, (mine, setting, target, budget, sel)
    print("self-check: K = 1 reproduces the base, class-weight and pseudo-label results exactly", flush=True)


def main() -> None:
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/system2_combined.csv"))
    args = parser.parse_args()
    self_check()
    t0 = time.time()
    jobs = [(s, k, seed, None) for s in I.SETTINGS for k in I.TARGETS for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        df = pd.DataFrame([r for rs in pool.map(run_job, jobs) for r in rs])
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} results, {int(df.diverged.sum())} diverged [{time.time() - t0:.0f} s]; wrote {args.out}")
    seeds = sorted(df.seed.unique())

    def score(config: str, targets: list[int], metric: str = "accuracy") -> float:
        d = df[(df.config == config) & df.target.isin(targets)]
        return float(d.groupby("budget")[metric].mean().mean())      # NaN (diverged) → NaN score

    def per_seed(config: str, setting: int, budget: int, sel: str, metric: str = "accuracy") -> np.ndarray:
        d = df[(df.config == config) & (df.setting == setting) & (df.budget == budget)
               & (df.selection == ("none" if budget == 0 else sel)) & df.target.isin(I.TEST)]
        return 100 * d.groupby("seed")[metric].mean().reindex(seeds).to_numpy()

    print("\nValidation accuracy (Batches 2–5) by K — mean over settings, budgets, selections")
    print(f"{'combination':<32}" + "".join(f"{f'K={k}':>9}" for k in K_VALUES))
    chosen = {}
    for cw, pl in itertools.product([False, True], [False, True]):
        base_name = " + ".join(["strong calibration"] + (["class weight"] if cw else []) + (["pseudo-labels"] if pl else []))
        vals = {k: score(name({"k": k, "cw": cw, "pl": pl}), I.VALIDATION) for k in K_VALUES}
        chosen[(cw, pl)] = max((k for k in K_VALUES if not np.isnan(vals[k])), key=lambda k: vals[k])
        print(f"{base_name:<32}" + "".join(f"{100 * v:>8.1f}%" if not np.isnan(v) else f"{'div.':>9}" for v in vals.values())
              + f"   → K = {chosen[(cw, pl)]}")

    published = {r["method"] + str(r["setting"]): r for r in L.load_published()}
    daelm = {0: "SVM-rbf", 20: "DAELM-S(20)", 30: "DAELM-S(30)", 50: "DAELM-T(50)"}
    show = [("Current System 2", "K=1"), ("Class weight + pseudo-labels (K=1)", "K=1 + class weight + pseudo-labels")]
    for cw, pl in itertools.product([False, True], [False, True]):
        k = chosen[(cw, pl)]
        label = " + ".join([f"Strong calib. K={k}"] + (["class weight"] if cw else []) + (["pseudo"] if pl else []))
        show.append((label, name({"k": k, "cw": cw, "pl": pl})))
    for setting in I.SETTINGS:
        for sel, tag in (("balanced", "labels balanced per gas"), ("ssa", "labels chosen as in DAELM")):
            print(f"\nTest accuracy, Batches 6–10, Setting {setting}, {tag} — mean ± std over {len(seeds)} seeds")
            print(f"{'':<52}" + "".join(f"{f'{b} labels':>14}" for b in I.BUDGETS))
            print(f"{'Published: SVM-rbf / DAELM-S(20) / S(30) / T(50)':<52}"
                  + "".join(f"{np.mean(published[daelm[b] + str(setting)]['per_batch'][4:]):>14.1f}" for b in I.BUDGETS))
            for label, cfg in show:
                cells = [per_seed(cfg, setting, b, sel) for b in I.BUDGETS]
                print(f"{label:<52}" + "".join(f"{f'{v.mean():.1f} ± {v.std(ddof=1):.1f}':>14}" for v in cells))

    best = name({"k": chosen[(True, True)], "cw": True, "pl": True})
    print(f"\nPre-stated criterion — full combination ({best}) at 50 labels vs DAELM-T(50) = 87.0 % (Batches 6–10):")
    for setting in I.SETTINGS:
        for sel in ("balanced", "ssa"):
            v = per_seed(best, setting, 50, sel)
            print(f"  Setting {setting}, {sel:<8}: {v.mean():.1f} ± {v.std(ddof=1):.1f}  ({v.mean() - 87.0:+.1f} points; "
                  f"seeds ≥ 87.0: {int(np.sum(v >= 87.0))}/{len(v)})")
    print("\nFull combination vs current System 2, test Batches 6–10 (mean over settings, budgets, selections):")
    diff = [df[(df.config == best) & (df.seed == s) & df.target.isin(I.TEST)].accuracy.mean()
            - df[(df.config == "K=1") & (df.seed == s) & df.target.isin(I.TEST)].accuracy.mean() for s in seeds]
    print(f"  {100 * np.mean(diff):+.1f} points (better on {sum(d > 0 for d in diff)}/{len(seeds)} seeds); "
          f"ethylene recall {100 * score('K=1', I.TEST, 'weak_recall'):.1f}% → {100 * score(best, I.TEST, 'weak_recall'):.1f}%; "
          f"macro recall {100 * score('K=1', I.TEST, 'macro_recall'):.1f}% → {100 * score(best, I.TEST, 'macro_recall'):.1f}%")
    d = df[(df.config == best) & df.target.isin(I.TEST)]
    extra = (d.updates * I.D_IN * I.N_CLASSES / d.scored).mean()
    macs = I.inference_macs({}) + extra
    print(f"  inference ≈ {macs:,.0f} MACs/sample → {macs * I.PJ_PER_MAC * 1e9:.1f} nJ at 3.1 pJ/MAC "
          f"(DAELM {I.DAELM_MACS * I.PJ_PER_MAC * 1e9:.0f} nJ, {I.DAELM_MACS / macs:.0f}× more)")


if __name__ == "__main__":
    main()
