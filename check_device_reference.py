"""
Steps 1–2 of the C port: evidence that device_reference.py is System 2+ as published, and what running
it in single precision on the chip changes.

    python3 check_device_reference.py          # about 10 minutes on 10 cores

1. Elementary functions: the float32 exp and log (Cephes, as the C code) against float64, in ulps.
2. Features: the chip's path (sensor-major rows, sign flips, left-to-right means) against
   tune_system2.raw_features, in float64.
3. Readout: one maintenance visit (50 calibration passes, then the pseudo-labelled stream) against
   scikit-learn's SGDClassifier, coefficient by coefficient.
4. Paper reproduction: every System 2+ result behind the paper (the Settings 1–2 runs of
   system2_combined.csv and the sequential runs of sequential_system2plus.csv, with and without
   pseudo-labels) rerun through the reference in float64 with the paper's arithmetic; expected identical.
5. Single precision (step 2). The same runs with the chip's arithmetic (centred alignment, left-to-right
   sums, float32 Cephes exp/log, an underflow-safe pseudo-label confidence test), calibration passes
   still in the paper's order so that only the arithmetic changes, compared with the paper's committed
   numbers. Acceptance, fixed before running: float32 is acceptable if every cell reported in the paper
   (System 2+ in Settings 1–2 at each label budget and selection, Batches 6–10; the sequential protocol
   with and without pseudo-labels at 5/10/12/15 labels per gas) moves by less than 0.5 points, and the
   mean change over all cells is within ±0.2 points.
   Also reported: float64 with the chip's arithmetic (the formula changes alone), and float32 with the
   chip's own shuffle for the calibration passes (a different random stream: cells are expected to move
   by about the seed-to-seed noise, with no systematic shift).
6. Supplementary: the Settings cells on 20 seeds, float32 against float64 paired seed by seed.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import combine_system2 as C
import device_reference as R
import improve_system2 as I
import sequential_system2plus as S
import tune_system2 as T
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import load_batch

ARITHS = {"paper64": R.PAPER64, "chip64": R.Arith(np.float64, "chip", "numpy"),
          "chip32": R.CHIP32, "chip32_device": R.CHIP32_DEVICE}
SEQ_CONDITIONS = {"System 2+": True, "System 2+ without pseudo-labels": False}
SEQ_SEEDS = DEFAULT_SEEDS + S.EXTRA_SEEDS
EXTRA_SETTINGS_SEEDS = list(range(2001, 2016))


def ulps(approx: np.ndarray, exact: np.ndarray) -> np.ndarray:
    e32 = exact.astype(np.float32)
    spacing = np.spacing(np.abs(e32)).astype(np.float64)
    return np.abs(approx.astype(np.float64) - exact) / spacing


def check_math() -> None:
    rng = np.random.default_rng(0)
    xe = np.concatenate([rng.uniform(R.EXP_LO, R.EXP_HI, 1_000_000), rng.uniform(-40, 40, 1_000_000)]).astype(np.float32)
    print(f"  exp inputs are clamped to [{R.EXP_LO}, {R.EXP_HI}] (the chip only needs σ and e^−x there); within it:")
    xl = np.exp(rng.uniform(np.log(1e-30), np.log(1e30), 2_000_000)).astype(np.float32)
    for name, mine, numpy32, exact in (("exp", R.expf(xe), np.exp(xe), np.exp(xe.astype(np.float64))),
                                       ("log", R.logf(xl), np.log(xl), np.log(xl.astype(np.float64)))):
        print(f"  {name}: max error {ulps(mine, exact).max():.2f} ulp (numpy's own float32 {name}: "
              f"{ulps(numpy32, exact).max():.2f} ulp) over {len(exact):,} inputs")


def check_features() -> None:
    worst64, worst32 = 0.0, 0.0
    for b in range(1, 11):
        paper = T.raw_features(b, "all+ratio")
        worst64 = max(worst64, float(np.abs(R.chip_features(R.sensor_rows(b), ARITHS["chip64"]) - paper).max()))
        f32 = R.chip_features(R.sensor_rows(b), R.CHIP32).astype(np.float64)
        worst32 = max(worst32, float((np.abs(f32 - paper) / np.maximum(np.abs(paper), 1e-3)).max()))
    print(f"  float64, chip path vs tune_system2.raw_features: max |difference| {worst64:.1e} (all 10 batches)")
    print(f"  float32, chip path: max relative difference {worst32:.1e}")


def check_readout() -> None:
    """Seed 42, Batch 2 of the sequential protocol at 12 labels per gas, System 2+."""
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    warnings.simplefilter("ignore", ConvergenceWarning)
    import benchmark_drift as D
    from dataset_adapter import stratified_split
    seed, ar = 42, R.PAPER64
    b1 = load_batch(1)
    train_idx, _ = stratified_split(b1.gas, test_fraction=0.2, seed=seed)
    x1 = T.raw_features(1, "all+ratio")[train_idx]
    mu1, sigma1 = x1.mean(axis=0), np.maximum(x1.std(axis=0), 1e-12)
    sk = R._sgd(seed, True).fit(x1 * (1.0 / sigma1) + (-mu1 / sigma1), b1.gas[train_idx])
    dev = R.Device(copy.deepcopy(sk), mu1, sigma1 ** 2, ar, "affine")
    batch = load_batch(2)
    order = D.stream_order(seed, 2, len(batch))
    cal, test = D.fewshot_split(batch.gas, order, 12, safe=True)
    y = dev.align(R.features(2, ar)[np.concatenate([cal, test])])
    ycal, ytest = y[:len(cal)], y[len(cal):]
    passes = R.Passes(ar, [seed, 2]).epochs(len(cal), 50)
    dev.calibrate(ycal, batch.gas[cal], 3.0, passes)
    weight = np.full(len(cal), 3.0)
    for pos in passes:
        sk.partial_fit(ycal[pos], batch.gas[cal][pos], sample_weight=weight)
    r = dev.readout
    print(f"  after 50 calibration passes ({len(cal)} samples): max |Δ coef| {np.abs(r.w - sk.coef_).max():.1e}, "
          f"max |Δ intercept| {np.abs(r.b - sk.intercept_).max():.1e}, t equal: {r.t == sk.t_}")
    pred, updates, _ = dev.classify(ytest, batch.gas[test], True)
    sk_pred, sk_updates, _ = I.stream_pseudo(sk, ytest, np.arange(len(ytest)), batch.gas[test], C.PSEUDO)
    print(f"  after the pseudo-labelled stream ({len(ytest)} samples, {updates} updates): predictions identical: "
          f"{bool(np.all(pred == sk_pred))}, max |Δ coef| {np.abs(r.w - sk.coef_).max():.1e}, updates equal: "
          f"{updates == sk_updates}")


def seq_job(args: tuple[str, str, int, int]) -> list[dict]:
    arith, cond, budget, seed = args
    rows = R.sequential(budget, seed, 3.0, True, SEQ_CONDITIONS[cond], ARITHS[arith], keep=True)
    return [dict(r, arith=arith, protocol="sequential", condition=cond) for r in rows]


def set_job(args: tuple[str, int, int, int, int, str]) -> dict:
    arith, setting, target, seed, budget, sel = args
    return dict(R.settings(setting, target, seed, budget, sel, 3.0, True, True, ARITHS[arith], keep=True),
                arith=arith, protocol="settings", condition="System 2+")


KEYS = {"sequential": ["condition", "budget", "seed", "batch"],
        "settings": ["setting", "target", "budget", "selection", "seed"]}


def agreement(df: pd.DataFrame) -> pd.Series:
    """Per run: share of its predictions identical to the float64 paper-arithmetic run."""
    out = pd.Series(np.nan, index=df.index)
    for protocol, keys in KEYS.items():
        d = df[df.protocol == protocol]
        ref = {k: p for k, p in zip(d[d.arith == "paper64"][keys].itertuples(index=False, name=None),
                                    d[d.arith == "paper64"].pred)}
        for i, k, p in zip(d.index, d[keys].itertuples(index=False, name=None), d.pred):
            out[i] = np.mean(p == ref[k])
    return out


def cells(df: pd.DataFrame) -> pd.DataFrame:
    """Paper cells, one row per (arith, cell, seed): mean accuracy over Batches 6–10, in %."""
    out = []
    s = df[(df.protocol == "settings") & df.target.isin(I.TEST)]
    for (arith, setting, budget, sel, seed), d in s.groupby(["arith", "setting", "budget", "selection", "seed"]):
        out.append({"arith": arith, "cell": f"Setting {int(setting)}, {int(budget)} labels"
                    + ("" if budget == 0 else f", {'balanced' if sel == 'balanced' else 'DAELM selection'}"),
                    "seed": seed, "value": 100 * d.accuracy.mean()})
    q = df[(df.protocol == "sequential") & df.batch.isin(I.TEST)]
    for (arith, cond, budget, seed), d in q.groupby(["arith", "condition", "budget", "seed"]):
        out.append({"arith": arith, "cell": f"Sequential, {int(budget)}/gas, {cond}", "seed": seed,
                    "value": 100 * d.accuracy.mean()})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/device_reference.csv"))
    args = parser.parse_args()
    np.seterr(all="ignore")
    print("1. Elementary functions (float32 Cephes vs float64)")
    check_math()
    print("2. Features")
    check_features()
    print("3. Readout vs scikit-learn (float64, paper arithmetic)")
    check_readout()

    t0 = time.time()
    seq = [(a, c, b, s) for a in ARITHS for c in SEQ_CONDITIONS for b in S.BUDGETS for s in SEQ_SEEDS]
    sets = [(a, st, tg, s, b, sel) for a in ARITHS for st in I.SETTINGS for tg in I.TARGETS for s in DEFAULT_SEEDS
            for b, sel in I.COMBOS]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for rs in pool.map(seq_job, seq, chunksize=4) for r in rs]
        rows += list(pool.map(set_job, sets, chunksize=8))
    df = pd.DataFrame(rows)
    df["agreement"] = agreement(df)
    df.drop(columns="pred").to_csv(args.out, index=False)
    print(f"\n{len(df):,} runs through the reference [{time.time() - t0:.0f} s]; wrote {args.out}")

    print("4. Paper reproduction (float64, paper arithmetic) vs the committed results")
    p64 = df[df.arith == "paper64"]
    seq_ref = pd.read_csv("results/sequential_system2plus.csv", float_precision="round_trip")
    comb = pd.read_csv("results/system2_combined.csv", float_precision="round_trip").rename(columns={"updates": "pseudo_updates"})
    comb = comb[comb.config == C.name({"k": 3, "cw": True, "pl": True})]
    for label, d in (("sequential", p64[p64.protocol == "sequential"].merge(seq_ref, on=KEYS["sequential"], suffixes=("", "_ref"))),
                     ("Settings 1–2", p64[p64.protocol == "settings"].merge(comb, on=KEYS["settings"], suffixes=("", "_ref")))):
        same = int(((d.accuracy == d.accuracy_ref) & (d.pseudo_updates == d.pseudo_updates_ref)).sum())
        print(f"  {label}: {same}/{len(d)} batch results identical (accuracy and pseudo-label count); "
              f"max |Δ accuracy| {100 * (d.accuracy - d.accuracy_ref).abs().max():.2g} points")

    committed = pd.concat([comb.assign(protocol="settings", arith="paper", condition="System 2+"),
                           seq_ref[seq_ref.condition.isin(SEQ_CONDITIONS)].assign(protocol="sequential", arith="paper")])
    c = pd.concat([cells(df), cells(committed)]).pivot_table(index=["cell", "seed"], columns="arith", values="value")
    order = list(dict.fromkeys(cells(df[df.arith == "paper64"]).cell))
    agree = df.groupby("arith").agreement.mean()
    print("\n5. Paper cells (Batches 6–10, % accuracy, mean over the paper's seeds) and change from the paper")
    print(f"{'cell':<52}{'paper':>8}{'chip64 Δ':>10}{'chip32 Δ':>10}{'chip32, own shuffle Δ':>23}")
    deltas = {a: [] for a in ARITHS if a != "paper64"}
    for cell in order:
        pc = c["paper"].loc[cell]
        line = f"{cell:<52}{pc.mean():>8.1f}"
        for a in deltas:
            d = (c[a].loc[cell] - pc).to_numpy()
            deltas[a].append(d.mean())
            line += f"{d.mean():>+10.2f}" if a != "chip32_device" else f"{d.mean():>+23.2f}"
        print(line)
    for a, d in deltas.items():
        d = np.array(d)
        print(f"  {a}: mean change {d.mean():+.2f} points, largest |change| {np.abs(d).max():.2f}, "
              f"predictions identical to the float64 reference's {100 * agree[a]:.1f} %")
    d = np.array(deltas["chip32"])
    ok = bool(np.abs(d).max() < 0.5 and abs(d.mean()) <= 0.2)
    print(f"\nStep 2 verdict (rule fixed before running): float32 on the chip is "
          f"{'acceptable' if ok else 'NOT acceptable'} (every cell |Δ| < 0.5: {bool(np.abs(d).max() < 0.5)}; "
          f"mean Δ within ±0.2: {bool(abs(d.mean()) <= 0.2)})")

    print(f"\n6. Supplementary: Settings cells on {len(EXTRA_SETTINGS_SEEDS) + len(DEFAULT_SEEDS)} seeds, float32 vs "
          "float64 paired (5 seeds proved too few to show per-cell effects)")
    extra = [(a, st, tg, s, b, sel) for a in ("paper64", "chip32") for st in I.SETTINGS for tg in I.TEST
             for s in EXTRA_SETTINGS_SEEDS for b, sel in I.COMBOS]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        more = pd.DataFrame(list(pool.map(set_job, extra, chunksize=8))).drop(columns="pred")
    both = pd.concat([df[(df.protocol == "settings") & df.arith.isin(["paper64", "chip32"])].drop(columns="agreement"), more])
    e = cells(both).pivot_table(index=["cell", "seed"], columns="arith", values="value")
    e["d"] = e.chip32 - e.paper64
    per_cell = e.d.groupby(level=0).mean()
    per_seed = e.d.groupby(level=1).mean().to_numpy()
    half = 2.093 * per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    print(f"  per cell: largest |Δ| {per_cell.abs().max():.2f} ({per_cell.abs().idxmax()}), cells with |Δ| ≥ 0.5: "
          f"{int((per_cell.abs() >= 0.5).sum())}/{len(per_cell)}; mean Δ {per_seed.mean():+.3f} "
          f"(95 % CI {per_seed.mean() - half:+.3f} to {per_seed.mean() + half:+.3f})")


if __name__ == "__main__":
    main()
