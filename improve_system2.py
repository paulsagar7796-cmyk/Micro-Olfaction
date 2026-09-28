"""
System 2 improvements under the literature protocols: can a cheap change close the gap to DAELM?

    python3 improve_system2.py          # ~15 minutes on 10 cores

Three modifications of the tuned System 2 (EMA alignment + linear SGD readout on the 256 raw + pattern
features), each run on its own, under Setting 1 (train on Batch 1) and Setting 2 (train on Batch K−1)
with 0, 20, 30 and 50 labelled target samples per batch (chosen balanced per gas, or as in DAELM),
exactly as in benchmark_literature_comparison.py:

  projection    a frozen random layer before the linear readout (OS-ELM style): h = act(W·x + b),
                W ~ N(0, 1/d), b ~ N(0, 1), act ∈ {ReLU, tanh}, width ∈ {64, 128, 256, 512, 1024},
                optionally with the input passed straight through as well (direct link); h is
                standardised with the source's statistics; only the readout is trained (its L2
                strength is tuned too).
  class_weight  class 4 (ethylene) weighted w ∈ {1.5, 2, 3, 4} in the logistic loss (source training
                and calibration); tuned on macro recall, since it trades accuracy for balance.
  pseudo        streaming pseudo-labelling on the unlabelled target samples, in arrival order: predict,
                then if the top class probability ≥ τ and that class has had fewer than `cap`
                pseudo-updates in this batch, one SGD step on the predicted label with sample weight
                `frac` (τ ∈ {0.9, 0.95, 0.99}, cap ∈ {10, 25, 50, 100}, frac ∈ {0.03, 0.1, 0.3}).
                Scored prequentially: each prediction is made before any update it triggers.
  linear        control: the unmodified readout with its L2 strength retuned on the same validation
                (1e-5 is the current System 2), so gains are not just retuning.

All choices use target Batches 2–5 (validation; mean over budgets of the mean over settings, label
selections, batches and seeds). Batches 6–10 are the test. Labelled samples are never scored.
Inference energy = multiply-accumulates (MACs) × 3.1 pJ; activations, logs and square roots excluded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import benchmark_drift as D
import benchmark_literature_comparison as L
import tune_system2 as T
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import GAS_ORDER, gas_name, load_batch
from sweep_system2_budget import TUNED

SETTINGS = (1, 2)
TARGETS = list(range(2, 11))
VALIDATION, TEST = [2, 3, 4, 5], [6, 7, 8, 9, 10]
BUDGETS = [0, 20, 30, 50]
SELECTIONS = ["balanced", "ssa"]
COMBOS = [(0, "none")] + [(b, s) for b in BUDGETS[1:] for s in SELECTIONS]
WEAK = gas_name(4)
D_IN, N_CLASSES = 256, 6
PJ_PER_MAC = 3.1e-12
FRONT_END_MACS = 5 * D_IN            # EMA alignment: mean and variance updates, standardisation
DAELM_MACS = 128 + 1000 * 128 + 1000 * N_CLASSES   # scaling, 1000 radial-basis distances, output layer
BASE = ("linear", {"sgd_alpha": 1e-5})


def configs() -> list[tuple[str, dict]]:
    out = [("linear", {"sgd_alpha": a}) for a in (1e-5, 1e-4, 1e-3)]
    out += [("projection", {"hidden": h, "act": act, "direct": dl, "sgd_alpha": a})
            for h, act, dl, a in itertools.product([64, 128, 256, 512, 1024], ["relu", "tanh"], [False, True],
                                                   [1e-5, 1e-4, 1e-3])]
    out += [("class_weight", {"w4": w}) for w in (1.5, 2.0, 3.0, 4.0)]
    out += [("pseudo", {"tau": t, "cap": c, "frac": f})
            for t, c, f in itertools.product([0.9, 0.95, 0.99], [10, 25, 50, 100], [0.03, 0.1, 0.3])]
    return out


def cfg_id(family: str, cfg: dict) -> str:
    return f"{family}(" + ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in cfg.items()) + ")"


def readout_width(cfg: dict) -> int:
    return cfg["hidden"] + (D_IN if cfg["direct"] else 0) if cfg.get("hidden") else D_IN


def inference_macs(cfg: dict) -> int:
    """Per sample: alignment front end + random layer and its standardisation + linear readout."""
    layer = D_IN * cfg["hidden"] + readout_width(cfg) if cfg.get("hidden") else 0
    return FRONT_END_MACS + layer + readout_width(cfg) * N_CLASSES


class FeatureMap:
    """Frozen map on aligned, standardised inputs: identity, or a random layer (+ direct link)
    standardised with the source's statistics."""

    def __init__(self, cfg: dict, seed: int, source_x: np.ndarray):
        self.hidden = cfg.get("hidden")
        if self.hidden:
            rng = np.random.default_rng([seed, 97, self.hidden])
            self.w = rng.standard_normal((source_x.shape[1], self.hidden)) / np.sqrt(source_x.shape[1])
            self.b = rng.standard_normal(self.hidden)
            self.act, self.direct = cfg["act"], cfg["direct"]
            h = self._raw(source_x)
            s = h.std(axis=0)
            self.m, self.scale = h.mean(axis=0), np.where(s > 1e-6, 1.0 / np.maximum(s, 1e-6), 0.0)

    def _raw(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.w + self.b
        h = np.maximum(z, 0.0) if self.act == "relu" else np.tanh(z)
        return np.hstack([x, h]) if self.direct else h

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x if not self.hidden else (self._raw(x) - self.m) * self.scale


def stream_pseudo(model, f: np.ndarray, rest: np.ndarray, gas: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, int]:
    """Prequential pseudo-labelling: predict each sample, then maybe update on its predicted label."""
    from scipy.special import expit
    classes = model.classes_
    coef, icpt = model.coef_, model.intercept_
    used = dict.fromkeys(classes, 0)
    pred = np.empty(len(rest), dtype=object)
    updates = correct = 0
    for j, i in enumerate(rest):
        dec = f[i] @ coef.T + icpt
        k = int(np.argmax(dec))
        pred[j] = classes[k]
        p = expit(dec)                       # one-vs-rest probabilities, normalised as predict_proba does
        if p[k] / p.sum() >= cfg["tau"] and used[classes[k]] < cfg["cap"]:
            model.partial_fit(f[i:i + 1], [classes[k]], sample_weight=[cfg["frac"]])
            coef, icpt = model.coef_, model.intercept_
            used[classes[k]] += 1
            updates += 1
            correct += int(classes[k] == gas[i])
    return pred, updates, correct


def run_job(job: tuple[int, int, int, tuple | None]) -> list[dict]:
    """All configurations for one (setting, target batch, seed); `only` restricts to some configurations."""
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

    streams = {}                                        # (budget, selection) → labelled, rest, aligned features
    for budget, sel in COMBOS:
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

    rows, fitted = [], {}
    for family, cfg in configs():
        if only is not None and cfg_id(family, cfg) not in only:
            continue
        alpha = cfg.get("sgd_alpha", TUNED["sgd_alpha"])
        key = (cfg.get("hidden"), cfg.get("act"), cfg.get("direct"), alpha, cfg.get("w4", 1.0))
        if key not in fitted:                          # source training depends only on these
            fmap = FeatureMap(cfg, seed, us)
            weights = {WEAK: cfg["w4"]} if cfg.get("w4") else None
            model = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, random_state=seed, class_weight=weights)
            model.fit(fmap(us), ys)
            fitted[key] = (fmap, model)
        fmap, source_model = fitted[key]
        for budget, sel in COMBOS:
            labelled, rest, y = streams[(budget, sel)]
            model = copy.deepcopy(source_model)
            f = fmap(y)
            if len(labelled):
                L.add_classes(model, list(tb.gas[labelled]))
                rng = np.random.default_rng([seed, target, budget])
                for _ in range(TUNED["epochs"]):
                    perm = rng.permutation(labelled)
                    model.partial_fit(f[perm], tb.gas[perm])
            updates = pseudo_correct = 0
            if family == "pseudo":
                pred, updates, pseudo_correct = stream_pseudo(model, f, rest, tb.gas, cfg)
            else:
                pred = model.predict(f[rest])
            truth = tb.gas[rest]
            recall = {g: float(np.mean(pred[truth == g] == g)) for g in GAS_ORDER if np.any(truth == g)}
            rows.append({"family": family, "config": cfg_id(family, cfg), "setting": setting, "target": target,
                         "budget": budget, "selection": sel, "seed": seed,
                         "accuracy": float(np.mean(pred == truth)), "macro_recall": float(np.mean(list(recall.values()))),
                         "weak_recall": recall.get(WEAK, np.nan), "scored": len(rest), "updates": updates,
                         "pseudo_correct": pseudo_correct})
    return rows


def self_check() -> None:
    """The base configuration must reproduce benchmark_literature_comparison.adapt exactly; the manual
    probabilities must equal predict_proba; pseudo-labelling that never fires must equal the base."""
    base, never = cfg_id(*BASE), cfg_id("pseudo", {"tau": 2.0, "cap": 10, "frac": 0.1})
    for setting, target, seed in ((1, 4, 42), (2, 6, 7)):
        mine = {(r["budget"], r["selection"]): r["accuracy"] for r in run_job((setting, target, seed, (base,)))}
        for budget, sel in COMBOS:
            ref = L.adapt((setting, target, budget, sel, seed))[1]["accuracy"]
            if ref != mine[(budget, sel)]:
                raise RuntimeError(f"base differs from the literature comparison: setting {setting}, target {target}, "
                                   f"{budget} labels ({sel}): {mine[(budget, sel)]} vs {ref}")
    global configs
    original = configs
    configs = lambda: original() + [("pseudo", {"tau": 2.0, "cap": 10, "frac": 0.1})]  # noqa: E731
    try:
        rows = run_job((2, 6, 7, (base, never)))
    finally:
        configs = original
    acc = {(r["config"], r["budget"], r["selection"]): r["accuracy"] for r in rows}
    agree = all(acc[(never, b, s)] == acc[(base, b, s)] for b, s in COMBOS)

    from scipy.special import expit
    from sklearn.linear_model import SGDClassifier
    rng = np.random.default_rng(0)
    xx, yy = rng.standard_normal((200, 20)), np.array(GAS_ORDER)[rng.integers(0, 6, 200)]
    model = SGDClassifier(loss="log_loss", random_state=0).fit(xx, yy)
    p = expit(xx @ model.coef_.T + model.intercept_)
    probs_ok = np.allclose(p / p.sum(axis=1, keepdims=True), model.predict_proba(xx))
    if not (agree and probs_ok):
        raise RuntimeError(f"self-check failed: pseudo(off) == base: {agree}; probabilities match: {probs_ok}")
    print("self-check: base reproduces the literature comparison exactly (2 jobs × 7 budget/selection cases); "
          "pseudo-labelling that never fires equals the base; manual probabilities equal predict_proba", flush=True)


def main() -> None:
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/system2_improvements.csv"))
    args = parser.parse_args()
    self_check()

    t0 = time.time()
    jobs = [(s, k, seed, None) for s in SETTINGS for k in TARGETS for seed in args.seeds]
    jobs.sort(key=lambda j: -(len(load_batch(1 if j[0] == 1 else j[1] - 1)) + len(load_batch(j[1]))))  # big first
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for rs in pool.map(run_job, jobs) for r in rs]
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} target-batch results [{time.time() - t0:.0f} s]; wrote {args.out}")

    cfgs = dict((cfg_id(f, c), (f, c)) for f, c in configs())
    seeds = sorted(df.seed.unique())

    def score(config: str, metric: str, targets: list[int]) -> float:
        d = df[(df.config == config) & df.target.isin(targets)]
        return float(d.groupby("budget")[metric].mean().mean())

    def cell(config: str, setting: int, budget: int, selection: str, metric: str = "accuracy") -> str:
        sel = "none" if budget == 0 else selection
        d = df[(df.config == config) & (df.setting == setting) & (df.budget == budget) & (df.selection == sel)
               & df.target.isin(TEST)]
        v = 100 * d.groupby("seed")[metric].mean().reindex(seeds).to_numpy()
        return f"{v.mean():5.1f} ± {v.std(ddof=1):3.1f}"

    base_id = cfg_id(*BASE)
    chosen = {"linear": max((k for k, (f, _) in cfgs.items() if f == "linear"), key=lambda k: score(k, "accuracy", VALIDATION))}
    for fam, metric in (("projection", "accuracy"), ("class_weight", "macro_recall"), ("pseudo", "accuracy")):
        chosen[fam] = max((k for k, (f, _) in cfgs.items() if f == fam), key=lambda k: score(k, metric, VALIDATION))
    print("\nChosen on validation (target Batches 2–5):")
    for fam, k in chosen.items():
        metric = "macro_recall" if fam == "class_weight" else "accuracy"
        print(f"  {fam:<13} {k:<60} validation {metric.replace('_', ' ')} {100 * score(k, metric, VALIDATION):.1f}% "
              f"(base {100 * score(base_id, metric, VALIDATION):.1f}%)")

    published = {r["method"] + str(r["setting"]): r for r in L.load_published()}
    daelm = {0: "SVM-rbf", 20: "DAELM-S(20)", 30: "DAELM-S(30)", 50: "DAELM-T(50)"}
    rows_to_show = [("Base (current System 2)", base_id)] + [
        (f"{name}: {chosen[fam].split('(', 1)[1][:-1]}", chosen[fam])
        for fam, name in (("linear", "Linear, retuned"), ("projection", "Projection"),
                          ("class_weight", "Class weight"), ("pseudo", "Pseudo-labels"))]
    for metric, title in (("accuracy", "accuracy"), ("macro_recall", "macro-averaged recall"),
                          ("weak_recall", f"class 4 ({WEAK}) recall")):
        for setting in SETTINGS:
            print(f"\nTest {title}, Batches 6–10, Setting {setting} — mean ± std over {len(seeds)} seeds")
            print(f"{'':<62}" + "".join(f"{f'{b} labels':>14}" for b in BUDGETS))
            if metric == "accuracy":
                vals = [np.mean(published[daelm[b] + str(setting)]["per_batch"][4:]) for b in BUDGETS]
                print(f"{'Published: SVM-rbf (0) / DAELM-S(20) / DAELM-S(30) / DAELM-T(50)':<62}"
                      + "".join(f"{v:>14.1f}" for v in vals))
            for name, k in rows_to_show:
                for selection, tag in (("balanced", "balanced"), ("ssa", "DAELM selection")):
                    label = f"{name[:44]} [{tag}]"
                    print(f"{label:<62}" + "".join(f"{cell(k, setting, b, selection, metric):>14}" for b in BUDGETS))

    print(f"\nProjection trade-off (each width/activation at its validation-best L2; accuracy = mean over settings, "
          f"budgets and selections)")
    print(f"{'layer':<34}{'validation':>12}{'test 6–10':>11}{'MACs':>10}{'energy':>11}")
    print(f"{'none (current System 2)':<34}{100 * score(base_id, 'accuracy', VALIDATION):>11.1f}%"
          f"{100 * score(base_id, 'accuracy', TEST):>10.1f}%{inference_macs(BASE[1]):>10,}"
          f"{inference_macs(BASE[1]) * PJ_PER_MAC * 1e9:>8.1f} nJ")
    groups = {}
    for k, (f, c) in cfgs.items():
        if f == "projection":
            groups.setdefault((c["hidden"], c["act"], c["direct"]), []).append(k)
    for (h, act, dl), ks in sorted(groups.items()):
        k = max(ks, key=lambda k: score(k, "accuracy", VALIDATION))
        macs = inference_macs(cfgs[k][1])
        print(f"{f'{h} {act}' + (' + direct link' if dl else ''):<34}{100 * score(k, 'accuracy', VALIDATION):>11.1f}%"
              f"{100 * score(k, 'accuracy', TEST):>10.1f}%{macs:>10,}{macs * PJ_PER_MAC * 1e9:>8.1f} nJ")
    print(f"{'DAELM (1000 radial-basis neurons, 128 inputs)':<34}{'':>12}{'':>11}{DAELM_MACS:>10,}"
          f"{DAELM_MACS * PJ_PER_MAC * 1e9:>8.1f} nJ")

    print(f"\nClass weighting of class 4 ({WEAK}) — mean over settings, budgets and selections")
    print(f"{'weight':>7}{'val accuracy':>14}{'val macro':>11}{'test accuracy':>15}{'test macro':>12}{f'test {WEAK}':>15}")
    for w, k in [(1.0, base_id)] + [(c["w4"], k) for k, (f, c) in cfgs.items() if f == "class_weight"]:
        print(f"{w:>7g}" + "".join(f"{100 * score(k, m, t):>{wd}.1f}%" for m, t, wd in (
            ("accuracy", VALIDATION, 13), ("macro_recall", VALIDATION, 10), ("accuracy", TEST, 14),
            ("macro_recall", TEST, 11), ("weak_recall", TEST, 14))))

    print("\nPseudo-labelling — top 5 by validation accuracy (test = mean over settings, budgets, selections)")
    print(f"{'configuration':<44}{'validation':>11}{'test':>8}{'updates/batch':>15}{'pseudo-label precision':>24}"
          f"{'extra MACs/sample':>19}")
    ps = sorted((k for k, (f, _) in cfgs.items() if f == "pseudo"), key=lambda k: -score(k, "accuracy", VALIDATION))
    for k in ps[:5]:
        d = df[(df.config == k) & df.target.isin(TEST)]
        precision = d.pseudo_correct.sum() / max(d.updates.sum(), 1)
        extra = (d.updates * readout_width({}) * N_CLASSES / d.scored).mean()
        print(f"{k.split('(', 1)[1][:-1]:<44}{100 * score(k, 'accuracy', VALIDATION):>10.1f}%"
              f"{100 * score(k, 'accuracy', TEST):>7.1f}%{d.updates.mean():>15.1f}{100 * precision:>23.1f}%{extra:>19.0f}")

    print("\nTest accuracy change vs base (Batches 6–10, mean over settings, budgets, selections), per seed:")
    for fam, k in chosen.items():
        per_seed = [(df[(df.config == k) & (df.seed == s) & df.target.isin(TEST)].accuracy.mean()
                     - df[(df.config == base_id) & (df.seed == s) & df.target.isin(TEST)].accuracy.mean()) for s in seeds]
        print(f"  {fam:<13} {100 * np.mean(per_seed):+5.1f} points (better on {sum(x > 0 for x in per_seed)}/{len(seeds)} seeds)")


if __name__ == "__main__":
    main()
