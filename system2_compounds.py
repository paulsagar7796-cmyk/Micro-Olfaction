"""
System 2+ on discrete compounds: leave-one-board-out test on the UCI twin gas sensor arrays (dataset
361), registered in validation/PROTOCOL_S2COMPOUNDS.md. Reuses combine_system2.adapt_target's machinery
(via system2plus_core.system2plus, moved unchanged from system2_home.py at tag pre-v4, which implements the same SGD + alignment + calibration + pseudo-label
pipeline on arbitrary feature arrays) unchanged; only the features (twin_arrays.py) and the fold
structure are new.

    python3 system2_compounds.py selftest   # structural self-check, no accuracy numbers
    python3 system2_compounds.py confirm    # the registered leave-one-board-out test (plus two exploratory checks)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import system2plus_core as S
import twin_arrays as T
from twin_arrays import feature_columns

BOARDS = (1, 2, 3, 4, 5)
GASES = ("carbon monoxide", "ethanol", "ethylene", "methane")
CAL_LEVELS = (10, 30, 50, 70, 90)          # spread ranks 1, 3, 5, 7, 9 of the 10 levels
METHODS = {"System 2+": dict(pl=True, align=True), "System 2+ without pseudo-labels": dict(pl=False, align=True),
           "System 2": dict(pl=False, align=False)}
SEEDS = S.SEEDS


def fold_data(df, target_board: int, kind: str = "all+ratio"):
    """Source (other boards, any order) and target (this board: the calibration trials first within
    each gas, then the rest by ascending level/repetition) feature arrays and labels."""
    cols = feature_columns(kind)
    src = df[df.board != target_board]
    tgt = df[df.board == target_board].copy()
    is_cal = (tgt.level.isin(CAL_LEVELS) & (tgt.rep == 1)).astype(int)
    # lexsort's last key is primary: calibration trials first (adapt_target aligns labelled samples first),
    # then the registered order (gas, concentration level, repetition)
    tgt = tgt.iloc[np.lexsort((tgt.rep.to_numpy(), tgt.level.to_numpy(), tgt.gas.to_numpy(), 1 - is_cal.to_numpy()))]
    xs, ys = src[cols].to_numpy(float), src["gas"].to_numpy(dtype=object)
    xt, yt = tgt[cols].to_numpy(float), tgt["gas"].to_numpy(dtype=object)
    n_cal = int((tgt.level.isin(CAL_LEVELS) & (tgt.rep == 1)).sum())
    return xs, ys, xt, yt, n_cal, tgt


def selftest() -> None:
    """Structural checks only: fold sizes, no source/target overlap, every calibration trial really is
    at a spread rank with rep 1, and the calibration count matches 5 per gas. No accuracy is computed."""
    df = T.load()
    for target in BOARDS:
        xs, ys, xt, yt, n_cal, tgt = fold_data(df, target)
        if n_cal != 5 * len(GASES):
            raise RuntimeError(f"board {target}: {n_cal} calibration trials, expected {5 * len(GASES)}")
        if set(tgt.board.unique()) != {target}:
            raise RuntimeError(f"board {target}: target rows include another board")
        if target in df[df.board != target].board.unique():
            raise RuntimeError(f"board {target}: source/target overlap")
        cal_rows = tgt.iloc[:n_cal]
        if not (cal_rows.level.isin(CAL_LEVELS).all() and (cal_rows.rep == 1).all()):
            raise RuntimeError(f"board {target}: calibration rows are not all spread-rank, rep 1")
        if sorted(cal_rows.gas.value_counts().to_dict().values()) != [5, 5, 5, 5]:
            raise RuntimeError(f"board {target}: calibration is not 5 per gas")
        if len(xs) + len(xt) != len(df):
            raise RuntimeError(f"board {target}: source + target rows do not cover the dataset")
        if not np.array_equal(sorted(np.concatenate([ys, yt])), sorted(df["gas"])):
            raise RuntimeError(f"board {target}: labels do not match the dataset")
        for part in (tgt.iloc[:n_cal], tgt.iloc[n_cal:]):
            keys = list(zip(part.gas, part.level, part.rep))
            if keys != sorted(keys):
                raise RuntimeError(f"board {target}: target trials are not in (gas, level, repetition) order")
    print(f"self-check: {len(BOARDS)} folds, each with {5 * len(GASES)} calibration trials (5 per gas at "
          f"levels {CAL_LEVELS}, repetition 1), no source/target overlap, full coverage")


def evaluate_fold(df, target: int) -> tuple[list[dict], dict]:
    """Per method and seed: scores on the trials after calibration, plus stored predictions (for the
    confusion matrix) and per-fold accuracy (for the paired HD1 test)."""
    xs, ys, xt, yt, n_cal, _ = fold_data(df, target)
    cal = np.arange(n_cal)
    rows, preds = [], {}
    for seed in SEEDS:
        for name, cfg in METHODS.items():
            rest, pred = S.system2plus(xs, ys, xt, yt, seed, 0, cfg["pl"], cfg["align"], classes=GASES, calibration=cal)
            rows.append({"target_board": target, "method": name, "seed": seed, **S.scores(yt[rest], pred, GASES)})
            preds[(name, seed)] = (rest, pred)
        pred_f = S.forest(np.vstack([xs, xt[:n_cal]]), np.concatenate([ys, yt[:n_cal]]), xt, seed,
                          len(feature_columns("all")))[n_cal:]
        rest_f = np.arange(n_cal, len(yt))
        rows.append({"target_board": target, "method": "random forest", "seed": seed, **S.scores(yt[rest_f], pred_f, GASES)})
        preds[("random forest", seed)] = (rest_f, pred_f)
    return rows, {"truth": yt, "n_cal": n_cal, "preds": preds}


def budget_curve(df) -> list[dict]:
    """Secondary, exploratory: accuracy against calibration budget (5, 10, 20 trials per gas), System 2+
    only, board 1 as target, 5 seeds."""
    rows = []
    xs, ys, _, _, _, tgt_full = fold_data(df, 1)
    cols = feature_columns("all+ratio")
    for n in (5, 10, 20):
        levels = CAL_LEVELS if n == 5 else tuple(range(10, 101, 10))
        reps = (1,) if n <= 10 else (1, 2)
        tgt = tgt_full.copy()
        is_cal = (tgt.level.isin(levels) & tgt.rep.isin(reps)).astype(int)
        tgt = tgt.iloc[np.lexsort((tgt.rep.to_numpy(), tgt.level.to_numpy(), tgt.gas.to_numpy(), 1 - is_cal.to_numpy()))]
        n_cal = int((tgt.level.isin(levels) & tgt.rep.isin(reps)).sum())
        xt, yt = tgt[cols].to_numpy(float), tgt["gas"].to_numpy(dtype=object)
        for seed in SEEDS[:5]:
            rest, pred = S.system2plus(xs, ys, xt, yt, seed, 0, True, True, classes=GASES, calibration=np.arange(n_cal))
            rows.append({"calibration_per_gas": n, "seed": seed, **S.scores(yt[rest], pred, GASES)})
    return rows


def exploratory_order(df) -> list[dict]:
    """S2 (exploratory, added in the pre-v4 audit): System 2+ with the target board's scored trials streamed in a
    label-independent random order (per seed and board), against the registered order, which is sorted by gas.
    The calibration trials still come first. The registered results are unchanged."""
    rows = []
    for b in BOARDS:
        xs, ys, xt, yt, n_cal, _ = fold_data(df, b)
        cal = np.arange(n_cal)
        for seed in SEEDS:
            perm = np.random.default_rng([seed, b]).permutation(np.arange(n_cal, len(yt)))
            order = np.r_[cal, perm]
            rest, pred = S.system2plus(xs, ys, xt[order], yt[order], seed, 0, True, True, classes=GASES, calibration=cal)
            rows.append({"target_board": b, "seed": seed, "stream_order": "random",
                         **S.scores(yt[order][rest], pred, GASES)})
    return rows


def confirm(out: Path) -> None:
    from scipy import stats
    import pandas as pd
    t0 = time.time()
    selftest()
    df = T.load()
    print(f"{len(df)} trials, {len(BOARDS)} boards; leave-one-board-out, {5 * len(GASES)} calibration trials per "
          f"fold, {len(SEEDS)} seeds", flush=True)

    all_rows, fold_preds = [], {}
    for target in BOARDS:
        rows, aux = evaluate_fold(df, target)
        all_rows += rows
        fold_preds[target] = aux
        print(f"board {target} done  [{time.time() - t0:.0f} s]", flush=True)

    adf = pd.DataFrame(all_rows)
    method_names = list(METHODS) + ["random forest"]
    summary, per_board = [], {m: {} for m in method_names}
    for name in method_names:
        g = adf[adf.method == name]
        row = {"method": name}
        for m in ("balanced_accuracy", "accuracy", *[f"recall_{gas}" for gas in GASES]):
            row[f"{m}_mean"], row[f"{m}_sd"] = float(g[m].mean()), float(g[m].std(ddof=1))
        by_fold = g.groupby("target_board")["accuracy"].mean()
        per_board[name] = by_fold.to_dict()
        row["per_board_accuracy"] = ", ".join(f"B{b}={a:.3f}" for b, a in sorted(by_fold.items()))
        summary.append(row)

    confusion = {m: {t: {p: 0.0 for p in GASES} for t in GASES} for m in method_names}
    for target in BOARDS:
        truth, n_cal, preds = fold_preds[target]["truth"], fold_preds[target]["n_cal"], fold_preds[target]["preds"]
        for name in method_names:
            for seed in SEEDS:
                rest, pred = preds[(name, seed)]
                for t, p in zip(truth[rest], pred):
                    confusion[name][t][p] += 1.0 / len(SEEDS)

    s2p = pd.Series(per_board["System 2+"]).sort_index()
    s2 = pd.Series(per_board["System 2"]).sort_index()
    wstat = stats.wilcoxon(s2p, s2)
    diff = float((s2p - s2).mean())

    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in summary for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(summary)
    fold_path = out.with_name(out.stem + "_folds.csv")
    fold_rows = [{"target_board": b, "method": m, "accuracy": float(a)} for m in per_board for b, a in per_board[m].items()]
    with open(fold_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target_board", "method", "accuracy"])
        w.writeheader()
        w.writerows(fold_rows)

    print(f"\n{'method':<34}{'balanced accuracy':>22}{'accuracy':>10}" + "".join(f"{g[:6]:>10}" for g in GASES))
    for r in summary:
        ba = f"{r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_sd']:.3f}"
        print(f"{r['method']:<34}{ba:>22}{r['accuracy_mean']:>10.3f}"
              + "".join(f"{r[f'recall_{g}_mean']:>10.3f}" for g in GASES))
    print("\nper-board accuracy (mean over seeds), System 2+ and System 2:")
    for b in BOARDS:
        print(f"  board {b}: System 2+ {s2p[b]:.3f}, System 2 {s2[b]:.3f}, diff {s2p[b] - s2[b]:+.3f}")
    print("\nSystem 2+ confusion (truth -> prediction, mean counts over seeds, calibration trials excluded):")
    for t in GASES:
        print(f"  {t:<16}" + "  ".join(f"{p}: {confusion['System 2+'][t][p]:.1f}" for p in GASES))

    full = next(r for r in summary if r["method"] == "System 2+")
    hd1 = diff >= 0.05 and wstat.pvalue < 0.05
    hd2 = full["accuracy_mean"] >= 0.80
    print(f"\nHD1 (System 2+ beats System 2 by >= 5 points, Wilcoxon over 5 folds p < 0.05): "
          f"{'MET' if hd1 else 'NOT MET'} (+{100 * diff:.1f} points, p = {wstat.pvalue:.3g})")
    print(f"HD2 (System 2+ accuracy >= 0.80, chance = 0.25): {'MET' if hd2 else 'NOT MET'} ({full['accuracy_mean']:.3f})")

    print("\nS1 (exploratory): accuracy against calibration budget, board 1 target, System 2+, 5 seeds")
    bud = budget_curve(df)
    bdf = pd.DataFrame(bud)
    print(bdf.groupby("calibration_per_gas")["accuracy"].agg(["mean", "std"]).round(3).to_string())
    bud_path = out.with_name(out.stem + "_budget.csv")
    with open(bud_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bud[0]))
        w.writeheader()
        w.writerows(bud)
    # per-seed rows (every method, board and seed), so board-level consistency can be checked from committed files
    seeds_path = out.with_name(out.stem + "_seeds.csv")
    with open(seeds_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    print("\nS2 (exploratory, pre-v4 audit): per-board seed consistency, and a label-independent stream order")
    wide = adf[adf.method.isin(["System 2+", "System 2"])].pivot_table(index=["target_board", "seed"], columns="method",
                                                                      values="accuracy")
    order_rows = exploratory_order(df)
    odf = pd.DataFrame(order_rows)
    expl = []
    for b in BOARDS:
        w_b = wide.loc[b]
        wins = int((w_b["System 2+"] > w_b["System 2"]).sum())
        rnd = float(odf[odf.target_board == b]["accuracy"].mean())
        expl.append({"target_board": b, "seeds_system2plus_beats_system2": wins, "seeds": len(w_b),
                     "system2plus_registered_order": float(s2p[b]), "system2plus_random_order": rnd,
                     "system2_unadapted": float(s2[b])})
        print(f"  board {b}: System 2+ beats System 2 in {wins}/{len(w_b)} seeds; System 2+ {s2p[b]:.3f} in the registered "
              f"(gas-sorted) order, {rnd:.3f} in a random order; unadapted {s2[b]:.3f}")
    expl_path = out.with_name(out.stem + "_exploratory.csv")
    with open(expl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(expl[0]))
        w.writeheader()
        w.writerows(expl)
    print(f"\nwrote {out}, {fold_path}, {bud_path}, {seeds_path} and {expl_path}  [{time.time() - t0:.0f} s]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("what", choices=["selftest", "confirm"])
    parser.add_argument("--out", type=Path, default=Path("results/system2_compounds.csv"))
    args = parser.parse_args()
    if args.what == "selftest":
        selftest()
    else:
        confirm(args.out)


if __name__ == "__main__":
    main()
