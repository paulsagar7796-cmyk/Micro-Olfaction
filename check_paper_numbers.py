"""
Checks that the result numbers in paper/comparison_tables.md follow from the files in results/ and
literature/published_results.csv: each number is recomputed, formatted as in the document, and must
appear in the section (or table row) that reports it. Exits with status 1 if any is missing.

    python3 check_paper_numbers.py          # after reproduce.sh, or on the committed results

Numbers the scripts only print (not written to results/) are covered through their logs by
check_reproduction.py.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_literature_comparison as L
import deep_model_cost as M
import improve_system2 as I

DOC = Path("paper/comparison_tables.md")
S2P_FEAT = 256                                         # features per sample (firmware/s2p.h)
FULL = "K=3 + class weight + pseudo-labels"
DAELM_ROWS = {0: "SVM-rbf", 20: "DAELM-S(20)", 30: "DAELM-S(30)", 50: "DAELM-T(50)"}
T_975 = {4: 2.776, 14: 2.145}                          # as sweep_system2_budget.py


def f1(v: float, sign: bool = False) -> str:
    return (f"{v:+.1f}" if sign else f"{v:.1f}").replace("-", "−")


def ms(v: np.ndarray) -> str:
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def gap(a: float, b: float) -> float:
    """Difference of two values as displayed (one decimal), the document's convention."""
    return round(a, 1) - round(b, 1)


class Doc:
    def __init__(self, text: str):
        self.sections = {p.split("\n", 1)[0].strip(): p for p in re.split(r"^## ", text, flags=re.M)[1:]}
        self.n, self.failed = 0, []

    def section(self, prefix: str) -> str:
        return next(v for k, v in self.sections.items() if k.startswith(prefix))

    def expect(self, prefix: str, *strings: str) -> None:
        """Each string must occur in the section; line breaks and quote markers count as single spaces."""
        flat = lambda s: re.sub(r"\s+", " ", re.sub(r"\n> ?", " ", s))  # noqa: E731
        body = flat(self.section(prefix))
        for s in strings:
            self.n += 1
            if flat(s) not in body:
                self.failed.append(f"[{prefix}] missing “{s}”")

    def rows(self, prefix: str) -> list[list[str]]:
        """Table rows of a section as cell lists, with the first cell carried down when left empty."""
        out, first = [], ""
        for line in self.section(prefix).splitlines():
            if line.startswith("| ") and not line.startswith("|---"):
                cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
                first = cells[0] or first
                out.append([first] + cells[1:])
        return out


def combined_checks(doc: Doc, df: pd.DataFrame, pub: dict) -> None:
    def per_seed(config: str, setting: int, budget: int, sel: str, targets: list[int] = I.TEST,
                 metric: str = "accuracy") -> np.ndarray:
        d = df[(df.config == config) & (df.setting == setting) & (df.budget == budget)
               & (df.selection == ("none" if budget == 0 else sel)) & df.target.isin(targets)]
        return 100 * d.groupby("seed")[metric].mean().to_numpy()

    def score(config: str, targets: list[int], metric: str = "accuracy") -> float:
        d = df[(df.config == config) & df.target.isin(targets)]
        return 100 * float(d.groupby("budget")[metric].mean().mean())

    late = lambda method, s: float(np.nanmean(np.array(pub[(method, str(s))]["per_batch"])[4:]))  # noqa: E731
    lit = pd.read_csv("results/literature_comparison.csv")
    for s in I.SETTINGS:
        doc.expect("2.", "| " + " | ".join(f1(late(DAELM_ROWS[b], s)) for b in I.BUDGETS) + " |")
        base = [ms(per_seed("K=1", s, 0, "none"))] + [
            f"{per_seed('K=1', s, b, 'balanced').mean():.1f} / {per_seed('K=1', s, b, 'ssa').mean():.1f}" for b in I.BUDGETS[1:]]
        doc.expect("2.", "| System 2 | " + " | ".join(base) + " |")
        # System 2 = K = 1 of the combination sweep; the literature comparison must agree
        ours = lit[lit.setting.astype(str) == str(s)]
        v = ours[ours.method == "Ours: EMA alignment only"].average_6_10.iloc[0]
        doc.n += 1
        if abs(v - per_seed("K=1", s, 0, "none").mean()) > 0.01:
            doc.failed.append(f"literature_comparison.csv Setting {s} 0 labels {v} ≠ K=1 in system2_combined.csv")
        doc.expect("2.", ms(per_seed(FULL, s, 0, "none")),
                   *[ms(per_seed(FULL, s, b, sel)) for b in I.BUDGETS[1:] for sel in ("balanced", "ssa")])

    lead = {(s, b): gap(per_seed(FULL, s, b, "balanced").mean(), late(DAELM_ROWS[b], s)) for s in I.SETTINGS for b in I.BUDGETS[1:]}
    doc.expect("2.", f"{f1(lead[1, 20], True)} (Setting 1) and {f1(lead[2, 20], True)} (Setting 2) at 20 labels; "
                     f"{f1(lead[1, 30], True)} and {f1(lead[2, 30], True)} at 30; "
                     f"{f1(lead[1, 50], True)} and {f1(lead[2, 50], True)} at 50")
    doc.expect("Headline", f"by {lead[1, 20]:.1f} points in both standard settings")
    doc.n += 1
    if round(lead[1, 20], 1) != round(lead[2, 20], 1):
        doc.failed.append("headline: the 20-label lead is no longer equal in both settings")
    s2 = per_seed(FULL, 2, 50, "balanced")
    doc.expect("2.", f"exceeds 87.0 on {int(np.sum(s2 >= 87.0))} of {len(s2)} seeds")
    gaps = [gap(late(DAELM_ROWS[b], s), per_seed(FULL, s, b, "ssa").mean()) for s in I.SETTINGS for b in I.BUDGETS[1:]]
    doc.n += 1
    if min(gaps) <= 0:
        doc.failed.append("System 2+ with DAELM selection no longer trails DA-ELM at every budget")
    doc.expect("2.", f"by {min(gaps):.1f}–{max(gaps):.1f} points")

    seeds = sorted(df.seed.unique())
    test = df[df.target.isin(I.TEST)]
    diff = [test[(test.config == FULL) & (test.seed == s)].accuracy.mean()
            - test[(test.config == "K=1") & (test.seed == s)].accuracy.mean() for s in seeds]
    doc.expect("2.", f"{100 * np.mean(diff):+.1f} points", f"better on {sum(d > 0 for d in diff)}/{len(seeds)} seeds",
               f"recall {score('K=1', I.TEST, 'weak_recall'):.1f} % → {score(FULL, I.TEST, 'weak_recall'):.1f} %",
               f"macro recall {score('K=1', I.TEST, 'macro_recall'):.1f} % → {score(FULL, I.TEST, 'macro_recall'):.1f} %",
               f"{score('K=3 + class weight + pseudo-labels', I.VALIDATION):.1f} % at K = 3 to "
               f"{score('K=300 + class weight + pseudo-labels', I.VALIDATION):.1f} % at K = 300")
    for label, cfg in (("System 2", "K=1"), ("+ class weight + pseudo-labels", "K=1 + class weight + pseudo-labels"),
                       ("+ strong calibration (K = 3)", "K=3"), ("+ strong calibration + pseudo-labels", "K=3 + pseudo-labels"),
                       ("+ strong calibration + class weight", "K=3 + class weight"), ("System 2+ (all three)", FULL)):
        cells = [f"{per_seed(cfg, s, 20, 'balanced').mean():.1f} / {per_seed(cfg, s, 50, 'balanced').mean():.1f}"
                 for s in I.SETTINGS]
        doc.expect("2.", f"| {label} | {cells[0]} | {cells[1]} |")


def improvements_checks(doc: Doc) -> None:
    df = pd.read_csv("results/system2_improvements.csv", usecols=["config", "target", "budget", "accuracy"])
    score = lambda k, t: 100 * float(df[(df.config == k) & df.target.isin(t)].groupby("budget").accuracy.mean().mean())  # noqa: E731
    cfgs = {I.cfg_id(f, c): (f, c) for f, c in I.configs()}
    groups = {}
    for k, (f, c) in cfgs.items():
        if f == "projection":
            groups.setdefault((c["hidden"], c["act"], c["direct"]), []).append(k)
    best = [max(ks, key=lambda k: score(k, I.VALIDATION)) for ks in groups.values()]
    ratio = [I.inference_macs(cfgs[k][1]) / I.inference_macs(I.BASE[1]) for k in best]
    doc.expect("2.", f"best {max(score(k, I.VALIDATION) for k in best):.1f} % vs "
                     f"{score(I.cfg_id(*I.BASE), I.VALIDATION):.1f} % without, and {min(ratio):.0f}–{max(ratio):.0f}×")


def sequential_checks(doc: Doc, pub: dict) -> None:
    df = pd.read_csv("results/sequential_system2plus.csv", float_precision="round_trip")
    test = df[df.batch.isin(I.TEST)]
    acc = lambda cond, b: 100 * test[(test.condition == cond) & (test.budget == b)].groupby("seed").accuracy.mean().mean()  # noqa: E731
    doc.expect("2.", f"Batches 6–10 ({test.seed.nunique()} seeds): System 2+ reaches "
               + " / ".join(f"{acc('System 2+', b):.1f}" for b in (10, 12, 15)) + " % at 10 / 12 / 15 labels per gas",
               "(System 2: " + " / ".join(f"{acc('System 2', b):.1f}" for b in (10, 12, 15)) + " %)",
               f"{acc('System 2+', 5):.1f} % at 5")
    p = df[df.condition == "System 2+"].groupby(["budget", "batch"])[["pseudo_updates", "pseudo_correct"]].sum()
    precision = 100 * p.pseudo_correct / p.pseudo_updates
    doc.expect("2.", f"{precision.min():.0f}–{precision.max():.0f} % of pseudo-labels are correct in each batch")
    wide = df.pivot_table(index=["budget", "seed", "batch"], columns="condition", values="accuracy")
    gap = (100 * (wide["System 2+"] - wide["System 2+ without pseudo-labels"])).rename("gap").reset_index()
    effects = []
    for b in sorted(gap.budget.unique()):
        g = gap[gap.budget == b]
        effects.append(g[g.batch.isin(I.TEST)].groupby("seed").gap.mean().mean())
        slope = np.array([np.polyfit(s.batch, s.gap, 1)[0] for _, s in g.groupby("seed")])
        half = T_975[len(slope) - 1] * slope.std(ddof=1) / np.sqrt(len(slope))
        doc.n += 1
        if slope.mean() + half < 0:
            doc.failed.append(f"[2.] pseudo-label effect declines across batches at {b} labels per gas")
    doc.expect("2.", f"their effect ({min(effects):+.1f} to {max(effects):+.1f} points) does not decline")
    avg = lambda m, s: float(np.nanmean(pub[(m, s)]["per_batch"]))  # noqa: E731
    doc.expect("2.", "Setting 1 — " + ", ".join(f"{n} {avg(m, '1'):.1f}" for n, m in (
        ("SVM-rbf", "SVM-rbf"), ("CC-PCA", "CC-PCA"), ("SVM-comgfk", "SVM-comgfk"), ("ML-comgfk", "ML-comgfk"))),
        "Setting 2 — " + ", ".join(f"{n} {avg(m, '2'):.1f}" for n, m in (
            ("SVM-rbf", "SVM-rbf"), ("SVM-comgfk", "SVM-comgfk"), ("ML-comgfk", "ML-comgfk"))),
        f"Vergara ensemble {avg('Ensemble', '2'):.1f}")


def literature_checks(doc: Doc, pub: dict) -> None:
    t = pd.read_csv("results/literature_table.csv", dtype={"target_labels": str})
    protocols = {"Setting 1": "Setting 1: Batch 1 → K", "Setting 2": "Setting 2: Batch K−1 → K",
                 "Setting 2, online": "Setting 2, online stream", "Batches 1 + 2 → K": "Multi-source: Batches 1+2 → K"}

    def value(protocol: str, method: str, labels: str) -> str:
        d = t[(t.protocol == protocol) & (t.method == method)]
        if method.startswith("System 2+"):
            d = d[d.target_labels == labels]
        r = d.iloc[0]
        return f"{r.average_6_10:.1f} ± {r.std_6_10:.1f}" if r.year == "ours" else f"{r.average_6_10:.1f}"

    for row in doc.rows("3."):
        if row[0] not in protocols:
            continue
        protocol, method, labels = protocols[row[0]], row[1], row[2].split(",")[0].split(" ")[0]
        method = re.sub(r" \((2015|2018|2022|2024)\)", "", method).replace(", offline", "")
        method = method.replace(", online", " (online)") if method.endswith(", online") else method
        parts = [method] if " / " not in method else [method.split(" / ")[0], "System 2+, " + method.split(" / ")[1]]
        expected = " / ".join(value(protocol, m, labels) for m in parts)
        doc.n += 1
        if row[-1] != expected:
            doc.failed.append(f"[3. table] {row[0]} | {row[1]} | {row[2]}: document {row[-1]}, results {expected}")

    a = lambda p, m, lab="0": float(t[(t.protocol == protocols[p]) & (t.method == m) & (t.target_labels == lab)].average_6_10.iloc[0])  # noqa: E731
    tda1, tda2, amds = a("Setting 1", "TDACNN"), a("Setting 2", "TDACNN"), a("Batches 1 + 2 → K", "AMDS-PFFA")
    ours1, ours2 = a("Setting 1", "System 2+"), a("Setting 2", "System 2+")
    ms0, ms20 = a("Batches 1 + 2 → K", "System 2+"), a("Batches 1 + 2 → K", "System 2+, balanced", "20")
    doc.expect("3.", f"by {gap(tda1, ours1):.1f} (Setting 1) and {gap(tda2, ours2):.1f} (Setting 2) points",
               f"by {gap(amds, ms0):.1f} points at 0 labels", f"by {gap(amds, ms20):.1f} at 20 labels",
               f"({a('Setting 2', 'ODAELM-S', '50 (Kennard-Stone)'):.1f}) trails",
               f"({a('Setting 2', 'System 2+, DAELM selection', '50'):.1f})",
               f"gives {a('Setting 2', 'DAELM-S (reproduced)', '50 (Kennard-Stone)'):.1f}",
               f"Zhang & Zhang's {a('Setting 2', 'DAELM-S(30)', '30'):.1f} at 30",
               f"({a('Setting 2, online', 'ODAELM-S (online)', '<65 (random, variable)'):.1f}, labelled",
               f"gets {a('Setting 2', 'System 2+, balanced', '30'):.1f} at 30 and {a('Setting 2', 'System 2+, balanced', '50'):.1f} at 50")
    rep = lambda m, s: pub[(m, s)]["reported_average"]  # noqa: E731
    doc.expect("3.", f"TDACNN {float(rep('TDACNN', '1')):.1f} (Setting 1) and {rep('TDACNN', '2')} (Setting 2; "
                     f"{np.nanmean(pub[('TDACNN', '2')]['per_batch']):.2f} from its",
               f"ODAELM-S offline {rep('ODAELM-S', '2')}, online {rep('ODAELM-S (online)', '2-online')}",
               f"AMDS-PFFA {rep('AMDS-PFFA', 'multi-source')} over Batches 3–10",
               f"({np.nanmean(pub[('AMDS-PFFA', 'multi-source')]['per_batch']):.1f} as a plain per-batch mean)")
    doc.expect("Where", f"scores {tda1:.1f} % (Setting 1) and {tda2:.1f} %", f"against {ours1:.1f} % and {ours2:.1f} %",
               f"AMDS-PFFA reaches {amds:.1f} %")


def confound_checks(doc: Doc) -> None:
    c = pd.read_csv("results/temporal_confound.csv")
    changes, kept, lower, total = {}, {}, 0, 0
    for batches, targets in (("6–10", I.TEST), ("2–10", list(range(2, 11)))):
        for s in I.SETTINGS:
            for b in (20, 50):
                d = c[(c.setting == s) & (c.budget == b) & c.target.isin(targets)]
                per = d.groupby(["condition", "seed"]).accuracy.mean().unstack(0) * 100
                u, h, z = per["usual draw (all levels)"], per["held-out levels"], per["no labels"]
                changes[batches, s, b] = h.mean() - u.mean()
                kept[batches, s, b] = 100 * (h.mean() - z.mean()) / (u.mean() - z.mean())
                if batches == "6–10":
                    lower, total = lower + int((h < u).sum()), total + len(u)
                    doc.expect("4.", f"| {s} | {b} | {ms(z.to_numpy())} | {ms(u.to_numpy())} | {ms(h.to_numpy())} | "
                                     f"{f1(changes[batches, s, b], True)} | {kept[batches, s, b]:.0f} % |")
    rng = lambda d, key: (min(v for k, v in d.items() if k[0] == key), max(v for k, v in d.items() if k[0] == key))  # noqa: E731
    lo, hi = rng(changes, "2–10")
    klo, khi = rng(kept, "2–10")
    doc.expect("4.", f"the change is {f1(hi)} to {f1(lo)} points and {klo:.0f}–{khi:.0f} %",
               f"on {lower} of {total} seed × setting × budget", "(%.0f–%.0f %%)" % rng(kept, "6–10"))
    lo, hi = rng(changes, "6–10")
    doc.expect("7.", f"optimistic by an estimated {-hi:.0f}–{-lo:.0f} points")

    f = pd.read_csv("results/temporal_confound_file_order.csv").set_index("batch")
    high = f[f.index_5nn_accuracy >= 0.85]
    doc.expect("4.", f"Within {len(high)} of 10 batch files",
               f"with {100 * high.index_5nn_accuracy.min():.0f}–{100 * high.index_5nn_accuracy.max():.1f} % accuracy",
               f"chance {100 * high.chance_1_over_k.min():.0f}–{100 * high.chance_1_over_k.max():.0f} %",
               f"label-permutation null {100 * high.permutation_null_mean.min():.0f}–{100 * high.permutation_null_mean.max():.0f} %",
               f"non-decreasing in {100 * high.concentration_sorted_within_runs.min():.0f}–"
               f"{100 * high.concentration_sorted_within_runs.max():.1f} %",
               *[f"{100 * f.index_5nn_accuracy[b]:.1f} %" for b in f.index if b not in high.index])
    base = pd.read_csv("results/temporal_confound_baseline.csv")
    doc.expect("4.", f"Spearman ρ {base.median_rho_with_concentration.min():+.2f} to {base.median_rho_with_concentration.max():+.2f}")


def fmt_j(joules: float) -> str:
    """An energy as section 5 writes it: mJ / µJ / nJ, two decimals below 10, one below 100, none above."""
    for scale, unit in ((1e-3, "mJ"), (1e-6, "µJ"), (1e-9, "nJ")):
        if joules >= 0.5 * scale or unit == "nJ":
            v = joules / scale
            return f"{v:.2f} {unit}" if v < 10 else f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"


def cost_checks(doc: Doc) -> None:
    """Headline energy, hardware claim and section 5: the Cortex-M4F results (results/cm4_*.csv, datasheet
    constants in cm4_profile.py) and, for the methods not ported, deep_model_cost.py's operation counts."""
    import cm4_profile as P
    en = pd.read_csv("results/cm4_energy.csv", float_precision="round_trip")
    mem = pd.read_csv("results/cm4_memory.csv").iloc[0]
    classes = pd.read_csv("results/cm4_inference_classes.csv")
    L476, A3 = "STM32L476", "Apollo3 Blue"

    def cyc(op: str, v: str = "opt") -> float:
        return float(en[en.operation.str.startswith(op) & (en.variant == v) & (en.chip == L476)].cycles.iloc[0])

    def joules(op: str, chip: str, v: str = "opt") -> float:
        return float(en[en.operation.str.startswith(op) & (en.variant == v) & (en.chip == chip)].energy_joules.iloc[0])

    infer, test, upd, dep, elm = ("inference, per sample", "pseudo-label confidence test", "pseudo-label update",
                                  "deployed average", "DA-ELM inference")
    ratio = {v: cyc(elm, v) / (cyc(infer, v) + cyc(test, v)) for v in P.VARIANTS}
    ratio_dep = {v: cyc(elm, v) / cyc(dep, v) for v in P.VARIANTS}
    doc.expect("Headline", f"at ≈{joules(infer, A3) * 1e6:.0f} µJ per inference (Apollo3 Blue; "
                           f"≈{joules(infer, L476) * 1e6:.0f} µJ on an STM32L476), {ratio['base']:.0f}× less than DA-ELM",
               f"({joules(infer, A3) * 1e6:.2f} and {joules(infer, L476) * 1e6:.1f} µJ)",
               f"({ratio['opt']:.0f}× with both optimised)",
               f"averages about {joules(dep, A3) * 1e6:.0f} µW on the Apollo3 Blue")
    import energy_proxy as E
    doc.expect("Headline", f"draws around {E.CONSTANTS['heater'][0] * 1e3:.0f} mW or more")
    doc.expect("Hardware", "≈60 KB of state", f"≈{joules(dep, A3) * 1e6:.1f} µJ per sample including updates on an Apollo3 "
                                              f"Blue, ≈{joules(dep, L476) * 1e6:.1f} µJ on an STM32L476",
               f"{mem.state_bytes:,} bytes of state with 50 stored labelled samples (the readout is "
               f"{mem.readout_bytes:,} of them), plus at most {mem.stack_bytes_opt} bytes of stack")
    doc.n += 1
    if not 55e3 <= mem.state_bytes + mem.stack_bytes_opt <= 65e3:
        doc.failed.append("hardware claim: state + stack is no longer ≈60 KB")

    s, t, a = M.system2_and_daelm(), M.tdacnn(), M.amds_pffa()
    m1, m7 = t["mmd"][1][0], t["mmd"][7][0]
    lo, hi = (a["macs"] + m for m, _ in a["encoders"].values())
    plo, phi = (a["params"] + p for _, p in a["encoders"].values())
    doc.expect("Hardware", f"TDACNN ≈{t['params'] / 1e3:.0f} KB",
               f"({s['inverse'] / 1e6:.1f} MB in FP32; about {2 * M.ELM_HIDDEN ** 2 / 1e6:.0f} M multiply-accumulates per update)",
               f"would fit in {250 * 250 * 4 / 1e3:.0f} KB")

    pj = P.PJ_PER_CYCLE
    doc.expect("5.", f"{pj[L476] / 3.0:.0f} µA/MHz at 3 V", f"= {pj[L476]:.0f} pJ per cycle",
               f"{pj[A3] / 3.3:.1f} µA/MHz at 3.3 V", f"= {pj[A3]:.1f} pJ per cycle")
    seq = pd.read_csv("results/sequential_system2plus.csv")
    seq = seq[(seq.condition == "System 2+") & (seq.budget == 12)]
    rate = 100 * seq.pseudo_updates.sum() / seq.scored.sum()
    rows = [("System 2+ inference, per sample", infer), ("Pseudo-label confidence test, per sample", test),
            ("One pseudo-label update", upd), (f"Deployed average per sample (updates on {rate:.1f} % of samples)", dep),
            ("Calibration visit, 20 labels (50 passes)", "calibration visit, 20"),
            ("Calibration visit, 50 labels (50 passes)", "calibration visit, 50"), ("DA-ELM inference, per sample", elm)]
    for label, op in rows:
        big = cyc(op) > 1e7
        c = (lambda x: f"{x / 1e6:.1f} M") if big else (lambda x: f"{x:,.0f}")
        doc.expect("5.", f"| {label} | {c(cyc(op, 'base'))} | {c(cyc(op))} | {fmt_j(joules(op, L476))} | "
                         f"{fmt_j(joules(op, A3))} |")
    doc.expect("5.", f"{ratio['base']:.1f}× with the reference implementations", f"{ratio['opt']:.1f}× with both optimised",
               f"for the deployed average, {ratio_dep['base']:.1f}× and {ratio_dep['opt']:.1f}×",
               f"takes {cyc(infer) / 80e6 * 1e3:.2f} ms at 80 MHz (STM32L476) and {cyc(infer) / 48e6 * 1e3:.2f} ms at 48 MHz",
               f"a 20-label calibration visit {cyc('calibration visit, 20') / 80e6:.2f} s and "
               f"{cyc('calibration visit, 20') / 48e6:.2f} s",
               f"{100 * classes[(classes.variant == 'opt') & (classes.instruction_class == 'FP divide / square root')].share_of_inference_cycles.iloc[0]:.0f} % of optimised",
               f"{mem.state_bytes:,} bytes of device state with 50 labelled samples (readout {mem.readout_bytes:,}), at most "
               f"{mem.stack_bytes_base} bytes of library stack; code {mem.code_bytes_base:,} bytes (reference) or "
               f"{mem.code_bytes_opt:,} bytes (optimised)",
               f"at least {s['inverse'] / 1e6:.0f} MB of working memory",
               f"{s['hidden'][1] / 1e6:.1f}–{s['hidden'][7] / 1e6:.0f} MB of hidden-layer outputs",
               # the System 2+ / DAELM table
               f"≈{s['farthest_point'] / 1e9:.2f} G MACs, {s['unlabelled'] / 1e6:.1f} MB buffer for Batch 7",
               f"| {cyc(infer):,.0f} cycles: {fmt_j(joules(infer, A3))} (Apollo3 Blue), {fmt_j(joules(infer, L476))} (STM32L476) | "
               f"{cyc(elm):,.0f} cycles: {fmt_j(joules(elm, A3))}, {fmt_j(joules(elm, L476))} |",
               f"| {(mem.state_bytes - 50 * (4 * S2P_FEAT + 4)) / 1e3:.1f} KB (readout {mem.readout_bytes / 1e3:.1f} KB, running "
               f"statistics {2 * 4 * S2P_FEAT / 1e3:.0f} KB) | {s['daelm_model'] / 1e6:.2f} MB of weights",
               f"{s['buffer'][20] / 1e3:.0f}–{s['buffer'][50] / 1e3:.0f} KB (20–50 labelled samples)",
               f"{s['source'][1] / 1e6:.1f}–{s['source'][7] / 1e6:.1f} MB stored source + "
               f"{s['hidden'][1] / 1e6:.1f}–{s['hidden'][7] / 1e6:.0f} MB hidden outputs + {s['inverse'] / 1e6:.1f} MB solve",
               f"{s['unlabelled'] / 1e6:.1f} MB unlabelled batch + {s['unlabelled_hidden'] / 1e6:.0f} MB hidden outputs",
               f"{cyc('calibration visit, 20') / 1e6:.1f} M cycles at 20 labels ({fmt_j(joules('calibration visit, 20', A3))} "
               f"Apollo3 Blue, {fmt_j(joules('calibration visit, 20', L476))} STM32L476)",
               f"{s['solve'][1] / 1e9:.1f}–{s['solve'][7] / 1e9:.1f} G MACs (closed-form solve)",
               # all compared methods
               f"reference DA-ELM took {cyc(elm, 'base') / I.DAELM_MACS:.0f} cycles per multiply-accumulate, the optimised "
               f"one {cyc(elm) / I.DAELM_MACS:.0f}",
               f"| System 2+ | {cyc(infer):,.0f} cycles ({fmt_j(joules(infer, A3))} on the Apollo3 Blue) | "
               f"{mem.readout_bytes / 1e3:.1f} KB |",
               f"| {mem.state_bytes / 1e3:.1f} KB state with 50 labelled samples |",
               f"| DAELM-S / -T | {cyc(elm):,.0f} cycles ({fmt_j(joules(elm, A3))}) | {s['daelm_model'] / 1e3:.0f} KB / "
               f"{s['daelm_model'] / 4e3:.0f} KB |",
               f"(≈{2 * M.ELM_HIDDEN ** 2 / 1e6:.0f} M MACs) | {s['inverse'] / 1e6:.1f} MB FP32",
               f"| {t['macs']:,} + {m1 / 1e3:.0f}k–{m7 / 1e3:.0f}k MACs (MMD ensemble) | "
               f"{4 * t['params'] / 1e6:.1f} MB / {t['params'] / 1e3:.0f} KB, plus "
               f"{4 * t['mmd'][1][1] / 1e6:.1f}–{4 * t['mmd'][7][1] / 1e6:.1f} MB",
               f"| {lo / 1e3:.0f}k–{hi / 1e3:.0f}k MACs | {4 * plo / 1e6:.1f}–{4 * phi / 1e6:.1f} MB / "
               f"{plo / 1e6:.2f}–{phi / 1e6:.2f} MB",
               f"≥{sum(L.BATCH_SIZES[b] for b in (1, 2, 7)) * 128 * 4 / 1e6:.1f} MB of raw samples")

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        E.presence_part([Path("results/presence_test_tracking.csv")], {k: v[0] for k, v in E.CONSTANTS.items()})
    clean = [float(m.group(1)) for m in re.finditer(r"tracking\s+\w+\s+encoder\s+([\d.]+) nW", out.getvalue())]
    doc.n += 1
    if len(clean) != 3:
        doc.failed.append("presence: expected three clean-air powers (nW) for the tracking configuration")
    else:
        doc.expect("5.", f"≈{min(clean):.0f}–{max(clean):.0f} nW in clean air")


def precision_checks(doc: Doc) -> None:
    """Section 5: what single precision changes (the paper cells of check_device_reference.py, from
    results/device_reference.csv against the committed double-precision results)."""
    import check_device_reference as K
    df = pd.read_csv("results/device_reference.csv", float_precision="round_trip")
    comb = pd.read_csv("results/system2_combined.csv", float_precision="round_trip").rename(columns={"updates": "pseudo_updates"})
    comb = comb[comb.config == "K=3 + class weight + pseudo-labels"]
    seq = pd.read_csv("results/sequential_system2plus.csv", float_precision="round_trip")
    committed = pd.concat([comb.assign(protocol="settings", arith="paper", condition="System 2+"),
                           seq[seq.condition.isin(K.SEQ_CONDITIONS)].assign(protocol="sequential", arith="paper")])
    c = pd.concat([K.cells(df[df.arith == "chip32"]), K.cells(committed)]).pivot_table(
        index=["cell", "seed"], columns="arith", values="value")
    d = (c.chip32 - c.paper).groupby(level=0).mean()
    doc.expect("5.", f"at most {d.abs().max():.2f} points (mean {d.mean():+.2f})")


def pseudo_label_checks(doc: Doc) -> None:
    """Section 2: pseudo-labels taken from samples that every unit rejects (check_pseudo_labels.py)."""
    q = pd.read_csv("results/pseudo_label_scores.csv").set_index("protocol")
    seq, sett = q.loc["sequential"], q.loc["settings"]
    doc.expect("2.", f"{100 * seq.share_all_units_reject:.1f} % of the pseudo-labels in the sequential protocol and "
                     f"{100 * sett.share_all_units_reject:.1f} % in Settings 1–2 (median winning score "
                     f"{f1(seq.median_winning_score_when_all_reject).split('.')[0]} and "
                     f"{f1(sett.median_winning_score_when_all_reject).split('.')[0]})",
               f"right only {100 * seq.precision_all_reject:.1f} % and {100 * sett.precision_all_reject:.1f} % of the time, "
               f"against {100 * seq.precision_winner_accepts:.1f} % and {100 * sett.precision_winner_accepts:.1f} %")


def main() -> None:
    doc = Doc(DOC.read_text())
    pub = {(r["method"], r["setting"]): r for r in L.load_published()}
    combined_checks(doc, pd.read_csv("results/system2_combined.csv"), pub)
    improvements_checks(doc)
    sequential_checks(doc, pub)
    literature_checks(doc, pub)
    confound_checks(doc)
    cost_checks(doc)
    pseudo_label_checks(doc)
    precision_checks(doc)
    for f in doc.failed:
        print("FAIL", f)
    print(f"{doc.n - len(doc.failed)}/{doc.n} checks of {DOC} passed")
    sys.exit(1 if doc.failed else 0)


if __name__ == "__main__":
    main()
