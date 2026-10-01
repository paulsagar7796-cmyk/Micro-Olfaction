"""
Checks the built preprint (preprint/dist/manuscript.html) against its source documents and the code.

    python3 preprint/check_manuscript.py            # after preprint/build.py
    python3 preprint/check_manuscript.py --mutate   # also: every check fails when what it guards is altered
    python3 preprint/check_manuscript.py --final    # also: nothing left to fill in (before the Zenodo upload)

Sources (build.SOURCES): paper-v3's results summary, paper/comparison_tables.md, and the twin-array results
and protocol in validation/, each identical to its version at the tag it comes from.

Checks
  source      each source document is identical to its tagged version (when git is available)
  tables      each table taken from a source equals it cell by cell, bold included, after only the changes
              declared in preprint/edits.json (omitted rows, dropped columns, cuts that delete text, the
              SOLAR rename); the dataset table equals fetch_data.py's published counts
  claims      each claim appears once in the manuscript, each of its passages appears in a source, and every
              number of the claim is a number of those passages (claims are written in the sources' naming,
              System 2 and System 2+, and renamed before they are looked for)
  facts       each statement no source makes (dataset, method settings, software) appears in the manuscript
              and is confirmed against the code
  numbers     every other number in the prose occurs in the source sections its manuscript section draws
              from (its data-src attribute; "*" = every source; a fraction such as 0.872 also counts as 87.2)
  scope       none of the ruled-out material: System 1, energy figures (left to the next paper), the old names
              outside the one sentence that gives the code names
  references  citations and references match one to one; every DOI and arXiv id is one verified by hand
  front       title, authors and statements are metadata.json's; with --final, no TODO and both DOIs set
  pdf         the PDF carries the title and every number of every table

What it cannot see: whether a sentence pairs the right number with the right quantity when both occur in
the same source section. The claims cover the headline statements; the rest needs reading.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILT = HERE / "dist" / "manuscript.html"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import build as B  # noqa: E402

NUM = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")

# Checked by hand on 2026-09-29 and 2026-09-30: Crossref metadata (title, authors, venue, volume, pages) for the
# DOIs, the arXiv API for the arXiv ids, UCI's dataset API and DataCite for the three dataset DOIs.
VERIFIED_DOIS = {
    "10.1016/j.snb.2012.01.074", "10.1016/j.chemolab.2013.10.012", "10.24432/C5RP6W", "10.24432/C5MK6M",
    "10.1109/TIM.2014.2367775", "10.3390/s18030742", "10.1016/j.snb.2022.131739", "10.1109/TIM.2025.3604131",
    "10.1016/j.snb.2022.131668", "10.1016/j.snb.2013.05.027", "10.1080/00401706.1969.10490666",
    "10.1016/j.snb.2016.05.089", "10.24432/C5MW3K",
}
VERIFIED_ARXIV = {"1505.06405", "2110.07509", "2409.13167", "2108.08793"}

# Material that is not part of this preprint: System 1, the old names (except in the sentence marked
# "codenames", which maps the new names to the code's), and energy figures, which are left to the next paper
FORBIDDEN = [r"\bSystem 1\b", r"\bSystem 2\b", r"\bpresence\b", r"\bspik", r"\bneuromorphic", r"\bLIF\b",
             r"\bALIF\b", r"\bsentinel", r"heater-aware", r"heater gating", r"paper-v4", r"pre-v4", r"concept drift",
             r"send-on-delta", r"leaky integrat", r"wake-up", r"false alarm", r"\bNucleo\b", r"\bTGS\d",
             r"system2_home", r"energy_system",
             r"µJ", r"\bmJ\b", r"\bnJ\b", r"\bpJ\b", r"µW", r"\bmW\b", r"µA/MHz", r"\bjoules?\b"]
EXEMPT = {"codenames"}

# (manuscript text in the sources' naming, source passages); None = the text itself is verbatim in a source
CLAIMS: list[tuple[str, list[str] | None]] = [
    ('System 2+ is trained on the other four boards and calibrated on 20 balanced trials of the held-out board', ['With 20 calibration samples per batch balanced across the six gases']),
    ('Batches 6–10 (months 17–36) are held out', ['Batches 6–10 (months 17–36)']),
    ('A concentration-level hold-out reduces few-shot accuracy by 7–11 percentage points, indicating that conventional benchmark results benefit from concentration-level proximity (section )', ['Few-shot accuracies are optimistic by an estimated 7–11 points on Batches 6–10']),
    ('retains 47–67 % of the accuracy gain', ['retains roughly half to two-thirds of the accuracy gain (47–67 %)']),
    ('running alignment of 256 features', ['alignment (EMA, α = 0.995) of 256 features']),
    ('50 passes of SGD', ['(50 passes)']),
    ('3, art', ["strong calibration (labelled target samples' loss × 3)"]),
    ('The published row gives SVM-rbf at 0 labels, DAELM-S(20) at 20, DAELM-S(30) at 30, and DAELM-T(50) at 50', ['| Protocol | Method | 0 labels | 20 labels | 30 labels | 50 labels |']),
    ('System 2 and System 2+: mean ± standard deviation over 5 seeds', ['Our numbers: mean over 5 seeds']),
    ('Alignment moments, weights, and pseudo-label updates persist across batches with periodic calibration, evaluated over 15 seeds with 5 to 15 labels per gas', ['Batches 6–10 (15 seeds)', 'at 10 / 12 / 15 labels per gas', '88.6 % at 5']),
    ('Under gas-agnostic sampling, the method underperforms DA-ELM by 0.9–7.2 percentage points', ['System 2+ trails at every labelled budget (by 0.9–7.2 points), so the advantage depends on balanced calibration']),
    ('Class 4 (ethylene) recall increases from 41.1 % to 52.6 %, lifting macro recall from 74.9 % to 77.9 %', ['class 4 (ethylene) recall 41.1 % → 52.6 %; macro recall 74.9 % → 77.9 %']),
    ('Excessive calibration loss weights degrade generalisation: validation accuracy drops from 87.8 % at K = 3 to 82.9 % at K = 300', ['validation accuracy falls from 87.8 % at K = 3 to 82.9 % at K = 300 (all three components on)']),
    ('Pseudo-label precision ranges between 84 % and 100 % across batches, providing a stable +0.3 to +0.6 point increment without runaway divergence across nine consecutive recalibration cycles', ['Pseudo-labelling shows no confirmation bias over nine maintenance cycles: 84–100 % of pseudo-labels are correct in each batch, and their effect (+0.3 to +0.6 points) does not decline across batches.']),
    ('On Batches 6–10, such instances represent 10.8 % of pseudo-labels in sequential mode and 19.1 % in Settings 1–2 (median winning logit −151 and −176), yielding only 60.4 % and 54.3 % empirical accuracy compared to 95.6 % and 88.9 % for positive-logit instances', ['On Batches 6–10 such samples give 10.8 % of the pseudo-labels in the sequential protocol and 19.1 % in Settings 1–2 (median winning score −151 and −176), and they are right only 60.4 % and 54.3 % of the time, against 95.6 % and 88.9 % for the others.']),
    ('This reflects file sorting rather than physical drift: gas concentrations are monotonically non-decreasing across 86–99.5 % of contiguous sample pairs within these files', ['concentration is non-decreasing in 86–99.5 % of adjacent pairs in those batches']),
    ('(median Spearman ρ +0.27 to +0.70)', None),
    ('However, calibration across distinct concentration levels still retains 47–67 % of the relative accuracy gain achieved under standard draws, indicating that the calibration effect is not an artefact of recording proximity', ['retains roughly half to two-thirds of the accuracy gain (47–67 %)']),
    ('128 sign-corrected + 128 log-pattern = 256 features', ['256 features (the 128 raw features plus log-pattern features per feature type)']),
    ('EMA mean and variance α = 0.995', ['(EMA, α = 0.995)']),
    ('(class 4 weighted × 2)', ['class 4 (ethylene) weighted × 2']),
    ('50 passes, loss × K = 3', ['(50 passes)', 'loss × 3']),
    ('if normalised p ≥ 0.95, ≤ 50 per gas:', ['≥ 0.95, at most 50 per gas per batch']),
    ('one SGD step, weight 0.1', ['an SGD step of weight 0.1']),
    ('Its memory footprint (59,780 bytes) enables full on-chip adaptation within standard microcontroller SRAM, bypassing the multi-megabyte requirements of closed-form or deep adaptive methods', ['59,780 bytes of state']),
    ('On independent physical sensor boards, System 2+ achieves 95.4 % cross-board accuracy and lifts the performance on the most degraded board from 58 % to 90 % (section )', ['with **95.4% accuracy**', 'reached only 58%', 'System 2+ reached **90%**']),
    ('Averaged over the five held-out boards, System 2+ achieves a classification accuracy of 95.4 %', ['with **95.4% accuracy**']),
    ('640 trials in all', None),
    ('64, no', ['64 raw (steady-state', 'plus 64 log-pattern features']),
    ('Accuracy per held-out board, mean over 15 seeds', ['5 boards × 15 seeds']),
    ('For comparison, a random forest given the same 20 calibration trials achieves 98.9 % accuracy across these boards', ['A random forest given the same 20 calibration trials does better: 98.9%']),
    ('and at 50 samples it comes within 1–4 points (and exceeds it in Setting 2)', ['and at 50 samples it comes within 1–4 points (and exceeds it in\n> Setting 2)']),
    ('System 2+ with 20 balanced target-calibration samples exceeds the published DA-ELM result by 4.3 percentage points', ['With 20 calibration samples per batch balanced across the six gases, the method exceeds DA-ELM by\n> 4.3 points']),
    ('(+4.3 in Setting 1 and +4.3 in Setting 2). At 30 labels, differences are −1.5 (Setting 1) and +1.9 (Setting 2); at 50 labels, −3.6 and +1.7', ['+4.3 (Setting 1) and +4.3 (Setting 2) at 20 labels; −1.5 and +1.9 at 30; −3.6 and +1.7 at 50']),
    ('On an emulated Cortex-M4F its inference takes 27,666 cycles', ['| System 2+ inference, per sample | 48,432 | 27,666 |']),
    ('Its adaptation runs entirely on-chip within 59,780 bytes of state, whereas DA-ELM requires at least 4 MB of memory', ['59,780 bytes of state', 'needs at least 4 MB of working memory']),
    ('Adaptation executes entirely on-chip within 59,780 bytes of state, whereas DA-ELM requires at least 4 MB of working memory', ['59,780 bytes of state', 'needs at least 4 MB of working memory']),
    ('runs on an emulated Cortex-M4F in 27,666 cycles per inference', ['| System 2+ inference, per sample | 48,432 | 27,666 |']),
    ('A concentration-level hold-out reduces few-shot accuracy by 7–11 percentage points, indicating that conventional benchmark results can benefit', ['optimistic by an estimated 7–11 points']),
    ('A concentration-level hold-out reduces few-shot accuracy by 7–11 percentage points, indicating that conventional benchmark results benefit', ['optimistic by an estimated 7–11 points']),
    ('reduces few-shot accuracy by 7–11 percentage points across Batches 6–10 (and by 7.2 to 12.9 points across Batches 2–10)', ['optimistic by an estimated 7–11 points on Batches 6–10', 'On Batches 2–10 the change is −7.2 to −12.9 points']),
    ('still retains 47–67 % of the relative accuracy gain', ['(47–67 %)']),
    ("System 2+ achieves 95.4 % cross-board accuracy after calibration on 20 of each board's trials and raises the weakest board from 58 % to 90 %; a random forest given the same calibration trials reaches 98.9 %", ["with **95.4% accuracy** (chance 25%), calibrated on 20 of that board's trials", 'reached only 58%', 'System 2+ reached **90%**', 'does better: 98.9%']),
    ('System 2+ achieves 95.4 % cross-board accuracy and lifts the performance on the most degraded board from 58 % to 90 %', ['with **95.4% accuracy**', 'reached only 58%', 'System 2+ reached **90%**']),
    ('System 2+ weights class 4 (ethylene) by a factor of 2 in the loss', ['class 4 (ethylene) weighted × 2']),
    ('at smoothing parameter α = 0.995', ['(EMA, α = 0.995)']),
    ('performs 50 passes of SGD', ['(50 passes)']),
    ('applies a loss multiplier of K = 3 (strong calibration)', ["strong calibration (labelled target samples' loss × 3)"]),
    ('meets or exceeds 0.95, an SGD update is taken on the predicted label with loss weighted by 0.1, capped at 50 pseudo-labels per gas per batch', ['≥ 0.95, at most 50 per gas per batch', 'an SGD step of weight 0.1']),
    ('(64–1,024 units, ReLU or tanh)', ['(64–1024 units, ReLU or tanh;']),
    ('its top variant achieved 84.9 % versus 85.7 % for the linear readout while increasing inference complexity by 6–97×', ['best 84.9 % vs 85.7 % without, and 6–97× the inference cost']),
    ('Target batches receive 0, 20, 30, or 50 labelled calibration samples', ['| Protocol | Method | 0 labels | 20 labels | 30 labels | 50 labels |']),
    ('mean ± standard deviation across 5 seeds', ['Our numbers: mean over 5 seeds']),
    ('evaluated over 15 seeds with 5 to 15 labels per gas', ['Batches 6–10 (15 seeds)', 'at 10 / 12 / 15 labels per gas', '88.6 % at 5']),
    ('Because seed variation spans ±1–4 points', ['ours varies by ±1–4 points across seeds']),
    ('System 2+ trails published figures by 0.9–7.2 points', ['(by 0.9–7.2 points)']),
    ('System 2+ outperforms System 2 by 2.9 percentage points averaged across all settings, budgets, and selection protocols', ['System 2+ vs System 2: +2.9 points']),
    ('Setting 1 — SVM-rbf 38.9, CC-PCA 45.7, SVM-comgfk 64.0, ML-comgfk 67.3; Setting 2 — SVM-rbf 68.0, SVM-comgfk 69.4, ML-comgfk 79.6, Vergara ensemble 80.0', None),
    ('validation accuracy drops from 87.8 % at K = 3 to 82.9 % at K = 300', ['87.8 % at K = 3 to 82.9 % at K = 300']),
    ('outperforming System 2+ by 8.3 points in Setting 1 and 7.3 points in Setting 2. AMDS-PFFA exceeds our multi-source run by 22.1 points at 0 labels and 6.7 points at 20 labels', ['TDACNN beats System 2+ by 8.3 (Setting 1) and 7.3 (Setting 2) points', 'AMDS-PFFA beats our multi-source run by 22.1 points at 0 labels', 'and by 6.7 at 20 labels']),
    ('ODAELM-S offline with 50 Kennard-Stone labels scores 79.7 % (counting calibration samples as correct predictions), trailing System 2+ with DAELM selection at 86.1 %', ['ODAELM-S at 50 Kennard-Stone labels (79.7) trails System 2+ with DAELM selection (86.1)']),
    ('ODAELM-S online achieves 84.4 % (with calibration samples scored)', ['ODAELM-S online (84.4, labelled samples scored']),
    ('System 2+ reaches 86.8 % at 30 and 88.7 % at 50 balanced labels', ['System 2+ gets 86.8 at 30 and 88.7 at 50 balanced labels']),
    ('scoring 64.7 % in Setting 1 and 72.7 % in Setting 2 versus 56.4 % and 65.4 % for uncalibrated System 2+. AMDS-PFFA achieves 81.3 % at zero target labels', ['it scores 64.7 % (Setting 1) and 72.7 % (Setting 2), against 56.4 % and 65.4 % for System 2+ at 0 labels', 'AMDS-PFFA reaches 81.3 % with 0 target labels']),
    ('System 2+ achieves 91.5 %, 92.7 %, and 93.7 % accuracy at 10, 12, and 15 labels per gas (System 2: 90.0 %, 91.3 %, 92.3 %), and 88.6 % at 5 labels', ['System 2+ reaches 91.5 / 92.7 / 93.7 % at 10 / 12 / 15 labels per gas (System 2: 90.0 / 91.3 / 92.3 %), and 88.6 % at 5']),
    ('Pseudo-label precision ranges between 84 % and 100 % across batches, providing a stable +0.3 to +0.6 point increment', ['84–100 % of pseudo-labels are correct in each batch, and their effect (+0.3 to +0.6 points)']),
    ('such instances represent 10.8 % of pseudo-labels in sequential mode and 19.1 % in Settings 1–2 (median winning logit −151 and −176), yielding only 60.4 % and 54.3 % empirical accuracy compared to 95.6 % and 88.9 %', ['such samples give 10.8 % of the pseudo-labels in the sequential protocol and 19.1 % in Settings 1–2 (median winning score −151 and −176), and they are right only 60.4 % and 54.3 % of the time, against 95.6 % and 88.9 % for the others']),
    ('In 7 of the 10 batch files (Batches 1–6 and 9), row indices predict gas identity with 90–99.6 % accuracy (5-NN on index, 5-fold stratified cross-validation; chance 17–20 %)', ["Within 7 of 10 batch files (Batches 1–6 and 9), a sample's position predicts its gas with 90–99.6 % accuracy (5-NN on the index, 5-fold stratified cross-validation; chance 17–20 %"]),
    ('non-decreasing across 86–99.5 % of contiguous sample pairs', ['non-decreasing in 86–99.5 % of adjacent pairs']),
    ('across all 1,500 evaluation runs, altering final floating-point accuracy by at most 0.31 points (mean drift +0.01)', ['every prediction of the 1,500 runs', 'by at most 0.31 points (mean +0.01)']),
    ('executes an inference in 48,432 cycles with the portable reference code and 27,666 cycles with the optimised kernels (0.35 ms at 80 MHz, 0.58 ms at 48 MHz)', ['| System 2+ inference, per sample | 48,432 | 27,666 |', 'Optimised inference takes 0.35 ms at 80 MHz (STM32L476) and 0.58 ms at 48 MHz']),
    ('A 20-label calibration visit (50 passes) requires 0.30 s at 80 MHz', ['a 20-label calibration visit 0.30 s', '(50 passes)', '0.35 ms at 80 MHz']),
    ('account for 26 % of optimised inference cycles', ['26 % of optimised inference cycles']),
    ('The device state spans 59,780 bytes (including 50 stored calibration vectors and 6,232 bytes for the linear readout), plus 864 bytes of stack and 7,112 bytes of optimised code', ['59,780 bytes of device state with 50 labelled samples (readout 6,232), at most 864 bytes of library stack; code 2,312 bytes (reference) or 7,112 bytes (optimised)']),
    ('(e.g., 128 KB on an STM32L476 or 384 KB on an Apollo3 Blue)', ['the chips have 128 KB (STM32L476) and 384 KB (Apollo3 Blue) of SRAM']),
    ('requires solving a 1,000 × 1,000 linear system requiring at least 4 MB of scratchpad RAM, in addition to 2.0–15 MB of intermediate activations', ['needs at least 4 MB of working memory for the 1,000 × 1,000 system alone, plus 2.0–15 MB of hidden-layer outputs']),
    ('requiring 30k to 241k MACs per inference', ['Its cost is ≈30k MACs with Batch 1 as source and up to ≈241k with Batch 7']),
    ('spans 170k to 758k MACs per inference', ['| AMDS-PFFA | 170k–758k MACs |']),
    ('calibrated on 20 balanced trials of the held-out board', ["calibrated on 20 of that board's trials"]),
    ('transfers poorly (58 % without adaptation), System 2+ restores accuracy to 90 %', ['reached only 58%', 'System 2+ reached **90%**']),
    ('a random forest given the same 20 calibration trials achieves 98.9 % accuracy', ['A random forest given the same 20 calibration trials does better: 98.9%']),
    ('underperforms DA-ELM by 0.9–7.2 percentage points', ['(by 0.9–7.2 points)']),
    ('(74.6 % vs 80.2 % at 20 labels)', ['(74.6 vs 80.2 at 20 labels)']),
    ('optimistic by 7–11 percentage points due to concentration-level, and plausibly same-session, proximity', ['optimistic by an estimated 7–11 points']),
    ('it exceeds published DA-ELM accuracy by 4.3 percentage points on the UCI drift benchmark and achieves 95.4 % accuracy across independent sensor boards', ['exceeds DA-ELM by\n> 4.3 points', '**95.4% accuracy**']),
    ('Its memory footprint (59,780 bytes)', ['59,780 bytes of state']),
]


def code_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _fetch():
    import fetch_data as F
    return F


def _sensors() -> int:
    return len({n.rsplit("_", 1)[1] for n in _fetch().FEATURE_NAMES})


def _months_max() -> int:
    return max(int(m) for v in _fetch().BATCH_MONTHS.values() for m in v.split(","))


def _outputs() -> tuple[int, int]:
    import check_reproduction as R
    sys.path.insert(0, str(ROOT / "validation"))
    import check_validation_outputs as V
    return len(R.OUTPUTS) + len(list((ROOT / "results" / "logs").glob("*.log"))), len(V.OUTPUTS)


# Statements no source makes, confirmed against the archived code: (manuscript text, check, source)
FACTS: list[tuple[str, callable, str]] = [
    ("13,910 measurements", lambda: sum(sum(c) for c in _fetch().EXPECTED_COUNTS.values()) == 13910,
     "fetch_data.py: UCI's per-batch counts"),
    ("recorded over 36 months with an array of 16 MOx sensors",
     lambda: _months_max() == 36 and _sensors() == 16, "fetch_data.py: BATCH_MONTHS, FEATURE_NAMES"),
    ("collected with 16 MOx sensors in 10 batches over 36 months",
     lambda: _sensors() == 16 and len(_fetch().BATCH_MONTHS) == 10 and _months_max() == 36,
     "fetch_data.py: BATCH_MONTHS, FEATURE_NAMES"),
    ("128 features (eight per sensor)", lambda: len(_fetch().FEATURE_NAMES) == 128 and len(_fetch().FEATURES_PER_SENSOR) == 8,
     "fetch_data.py: FEATURE_NAMES"),
    ("floored at 10⁻³ of Batch 1's median magnitude",
     lambda: "1e-3 * float(np.median(np.abs(b1.feature_group(g))))" in code_text("tune_system2.py"),
     "tune_system2.py: _floors"),
    ("λ = 10⁻⁵", lambda: "sgd_alpha=1e-5" in code_text("sweep_system2_budget.py"), "sweep_system2_budget.py: TUNED"),
    ("selected from {1.5, 2, 3, 4}", lambda: "w ∈ {1.5, 2, 3, 4}" in code_text("improve_system2.py"),
     "improve_system2.py: candidate class weights"),
    ("tuning K over {1, 3, 10, 30, 100, 300}", lambda: "K_VALUES = [1, 3, 10, 30, 100, 300]" in code_text("combine_system2.py"),
     "combine_system2.py: K_VALUES"),
    ("trains on 80 % of Batch 1", lambda: "train on 80 % of Batch 1" in code_text("sequential_system2plus.py"),
     "sequential_system2plus.py: protocol"),
    ("(Python 3.13.2, NumPy 2.2.6, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6)",
     lambda: "# Python 3.13.2" in code_text("requirements.txt") and all(x in code_text("requirements.txt").split() for x in
             ("numpy==2.2.6", "scipy==1.18.1", "scikit-learn==1.9.1", "pandas==3.0.6")), "requirements.txt"),
    ("(weight = 1.0 rather than 2.0)",
     lambda: "CLASS_WEIGHT = 2.0" in code_text("combine_system2.py") and "class_weight" not in code_text("system2plus_core.py"),
     "combine_system2.py CLASS_WEIGHT; system2plus_core.py has none"),
    ("reproducing all 45 benchmark results and 8 twin-array validation files identically",
     lambda: _outputs() == (45, 8), "check_reproduction.OUTPUTS with results/logs; validation/check_validation_outputs.OUTPUTS"),
]

# ── HTML ──
class Node:
    def __init__(self, tag: str, attrs: list, parent: "Node | None"):
        self.tag, self.attrs, self.parent, self.children = tag, dict(attrs), parent, []

    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())

    def iter(self):
        yield self
        for c in self.children:
            if isinstance(c, Node):
                yield from c.iter()


VOID = {"meta", "link", "br", "img", "hr", "input", "col", "source", "wbr"}


class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", [], None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


BLOCK = {"p", "li", "h1", "h2", "h3", "figcaption", "div", "ul", "ol", "text", "section", "figure", "header",
         "tr", "td", "th", "table", "thead", "tbody"}
SKIP = {"xref", "cite", "secnum", "capnum", "refnum"}


def text(node: Node, tables: bool = False, skip: set[str] = frozenset()) -> str:
    parts: list[str] = []

    def walk(n):
        if isinstance(n, str):
            parts.append(n)
            return
        if n.tag in ("style", "script", "head") or (n.tag == "table" and not tables):
            return
        if n.tag == "cite" or n.classes() & (SKIP | skip):
            return
        block = n.tag in BLOCK
        parts.append(" " if block else "")
        for c in n.children:
            walk(c)
        parts.append(" " if block else "")
    walk(node)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def cell_md(node: Node) -> str:
    """A table cell back in its source's Markdown: bold as **…**, code as `…`."""
    parts: list[str] = []

    def walk(n):
        if isinstance(n, str):
            parts.append(n)
            return
        mark = {"strong": "**", "code": "`"}.get(n.tag, "")
        parts.append(mark)
        for c in n.children:
            walk(c)
        parts.append(mark)
    for c in node.children:
        walk(c)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def flat(s: str) -> str:
    s = re.sub(r"\n> ?", " ", s)
    return re.sub(r"\s+", " ", s.replace("**", "").replace("`", "")).strip()


def numbers(s: str) -> list[str]:
    return [n.replace(",", "") for n in NUM.findall(s)]


def with_percent(tokens) -> set[str]:
    """The tokens, plus each three-decimal fraction as a percentage (0.872 → 87.2): the same number."""
    out = set(tokens)
    out |= {f"{100 * float(t):.1f}" for t in tokens if re.fullmatch(r"0\.\d{3}", t)}
    return out


def rename(s: str, edits: dict) -> str:
    for old, new in edits["rename"]:
        s = s.replace(old, new)
    return s


# ── model of the built manuscript ──
@dataclass
class Model:
    prose: dict[str, str]                       # section id → prose text (no tables, citations or cross-refs)
    scope_prose: dict[str, str]                 # the same without the sentence that gives the code names
    srcmap: dict[str, str]                      # section id → data-src
    tables: list[tuple[dict, list[list[str]]]]  # (attributes, rows of Markdown cells)
    cites: list[str]
    refs: list[str]
    refs_text: str
    dois: set[str]
    arxiv: set[str]
    front: dict[str, str] = field(default_factory=dict)


def load_model(path: Path) -> Model:
    p = Parser()
    p.feed(path.read_text(encoding="utf-8"))
    body = next(n for n in p.root.iter() if n.tag == "body")
    sections = [n for n in body.children if isinstance(n, Node) and n.tag == "section"]
    prose, scope_prose, srcmap = {}, {}, {}
    for s in sections:
        sid = s.attrs.get("id", "")
        prose[sid] = text(s)
        scope_prose[sid] = text(s, skip=EXEMPT)
        srcmap[sid] = s.attrs.get("data-src", "")
    tables = []
    for t in body.iter():
        if t.tag == "table":
            rows = [[cell_md(c) for c in tr.children if isinstance(c, Node) and c.tag in ("td", "th")]
                    for tr in t.iter() if tr.tag == "tr"]
            tables.append((t.attrs, rows))
    cites = [k for n in body.iter() if n.tag == "cite" for k in n.attrs.get("data-keys", "").split(",")]
    ref_section = next(s for s in sections if s.attrs.get("id") == "sec-references")
    refs = [n.attrs["data-key"] for n in ref_section.iter() if n.tag == "li"]
    raw = path.read_text(encoding="utf-8")
    dois = {d.rstrip(".,;)") for d in re.findall(r"10\.\d{4,9}/[^\s<\"'\]]+", unescape(raw))}
    arxiv = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", raw))
    header = next(n for n in body.iter() if n.tag == "header")
    decl = next(s for s in sections if s.attrs.get("id") == "sec-declarations")
    front = {"title": text(next(n for n in header.iter() if n.tag == "h1")),
             "authors": text(next(n for n in header.iter() if "authors" in n.classes())),
             "header": text(header), "declarations": text(decl),
             "placeholders": str(sum(1 for n in body.iter() if "placeholder" in n.classes())),
             "raw_todo": str(raw.count("TODO"))}
    return Model(prose, scope_prose, srcmap, tables, cites, refs, text(ref_section), dois, arxiv, front)


# ── sources ──
@dataclass
class Sources:
    texts: dict[str, str]
    sections: dict[str, dict[str, str]]
    flat_all: str
    edits: dict


def load_sources(report: "Report") -> Sources:
    texts = {}
    for name, (path, tag, tag_path) in B.SOURCES.items():
        t = (ROOT / path).read_text(encoding="utf-8")
        tagged = B.tagged_text(tag, tag_path)
        report.check("source", tagged is None or tagged == t, f"{path} differs from {tag_path} at tag {tag}")
        texts[name] = t
    edits = json.loads(B.EDITS.read_text(encoding="utf-8"))
    return Sources(texts, {k: B.v3_sections(v) for k, v in texts.items()},
                   " ".join(flat(v) for v in texts.values()), edits)


def section_of(src: Sources, doc: str, prefix: str) -> str:
    return B.v3_section(src.sections[doc], prefix)


def src_numbers(src: Sources, spec: str) -> set[str]:
    tokens: list[str] = []
    for entry in (spec.split("|") if spec != "*" else [f"{d}:*" for d in src.sections]):
        doc, prefix = entry.split(":", 1)
        parts = src.sections[doc].values() if prefix == "*" else [section_of(src, doc, prefix)]
        for t in parts:
            tokens += numbers(t)
    return with_percent(tokens)


# ── checks ──
class Report:
    def __init__(self):
        self.failed: dict[str, list[str]] = {}
        self.counts: dict[str, int] = {}

    def check(self, group: str, ok: bool, message: str) -> None:
        self.counts[group] = self.counts.get(group, 0) + 1
        if not ok:
            self.failed.setdefault(group, []).append(message)


def deletion_only(before: str, after: str) -> bool:
    """`after` keeps whole words of `before`, in order and unchanged, trailing punctuation aside."""
    def words(s: str) -> list[str]:
        return [w.rstrip(",:;") for w in s.split()]
    have, pos = words(before), 0
    for w in words(after):
        if w not in have[pos:]:
            return False
        pos = have.index(w, pos) + 1
    return True


def expected_table(src: Sources, key: str, omit: set[str], report: Report) -> list[list[str]]:
    """The source table with the declared edits applied, computed here independently of build.py."""
    doc, rest = key.split(":", 1)
    section, index = rest.rsplit("#", 1)
    rows = [list(r) for r in B.md_tables(section_of(src, doc, section))[int(index)]]
    report.check("tables", omit <= {r[0] for r in rows}, f"[{key}] declared omissions not in the source: {omit}")
    rows = [rows[0]] + [r for r in rows[1:] if r[0] not in omit]
    spec = src.edits["tables"].get(key, {})
    drop = spec.get("drop_columns", [])
    report.check("tables", set(drop) <= set(rows[0]), f"[{key}] dropped columns not in the source: {drop}")
    keep = [i for i, h in enumerate(rows[0]) if h not in drop]
    rows = [[r[i] for i in keep] for r in rows]
    for cut in spec.get("cells", []):
        row = next((r for r in rows[1:] if r[0] == cut["row"]), None)
        ok = row is not None and cut["column"] in rows[0]
        report.check("tables", ok, f"[{key}] no cell {cut['row']!r} / {cut['column']!r} to cut")
        if ok:
            i = rows[0].index(cut["column"])
            report.check("tables", deletion_only(row[i], cut["text"]),
                         f"[{key}] the cut of {cut['row']!r} / {cut['column']!r} adds text: {row[i]!r} → {cut['text']!r}")
            row[i] = cut["text"]
    return [[rename(c, src.edits) for c in r] for r in rows]


def dataset_rows_expected(src: Sources) -> list[list[str]]:
    F = _fetch()
    naming = B.md_tables(section_of(src, "v3", "1. Gas naming"))[0]
    names = {int(r[0]): r[1] for r in naming[1:]}
    expected = {b: tuple(F.PUBLISHED_COUNTS[b][F.PUBLISHED_COLUMN_FOR_ID[i]] for i in range(6)) for b in F.PUBLISHED_COUNTS}
    head = ["Batch", "Months"] + [f"{c} {names[c]}" for c in range(1, 7)] + ["Total"]
    rows = [head]
    for b in sorted(expected):
        rows.append([str(b), None] + [f"{n:,}" for n in expected[b]] + [f"{sum(expected[b]):,}"])
    totals = [sum(expected[b][i] for b in expected) for i in range(6)]
    rows.append(["All", ""] + [f"{n:,}" for n in totals] + [f"{sum(totals):,}"])
    return rows


def months_of(cell: str) -> list[int]:
    out = []
    for part in cell.split(","):
        a, _, b = part.strip().partition("–")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run_checks(m: Model, src: Sources, meta: dict, final: bool, report: Report) -> None:
    # tables
    seen = set()
    for attrs, rows in m.tables:
        key = attrs.get("data-src") or attrs.get("data-generated") or "?"
        if "data-src" in attrs:
            seen.add(key)
            omit = {o for o in attrs.get("data-omit", "").split("|") if o}
            want = expected_table(src, key, omit, report)
            same = rows == want
            where = "" if same else next((f"row {i}: {a} ≠ {b}" for i, (a, b) in enumerate(zip(rows, want)) if a != b),
                                         f"{len(rows)} rows ≠ {len(want)}")
            report.check("tables", same, f"[{key}] differs from its source after the declared edits: {where}")
        elif attrs.get("data-generated") == "dataset":
            want = dataset_rows_expected(src)
            ok = len(rows) == len(want)
            for got, exp in zip(rows, want):
                if exp[1] is None:                                  # months: compare as month lists
                    b = int(got[0]) if got[0].isdigit() else -1
                    exp = exp[:1] + [got[1] if b in _fetch().BATCH_MONTHS and months_of(got[1]) ==
                                     [int(x) for x in _fetch().BATCH_MONTHS[b].split(",")] else "≠"] + exp[2:]
                ok = ok and got == exp
            report.check("tables", ok, "[dataset] differs from fetch_data.py's published counts or months")
        else:
            report.check("tables", False, f"table without a declared source: {attrs}")
    stale = set(src.edits["tables"]) - seen
    report.check("tables", not stale, f"edits declared for tables the manuscript does not have: {sorted(stale)}")

    # claims
    prose_all = " ".join(m.prose.values())
    for ms, sources in CLAIMS:
        shown = flat(rename(ms, src.edits))
        count = prose_all.count(shown)
        report.check("claims", count == 1, f"{count} times in the manuscript (must be once): “{shown[:90]}…”")
        passages = sources or [ms]
        for s in passages:
            report.check("claims", flat(s) in src.flat_all, f"not in any source: “{flat(s)[:90]}…”")
        pool = with_percent([n for s in passages for n in numbers(flat(s))])
        extra = [n for n in numbers(shown) if n not in pool]
        report.check("claims", not extra, f"numbers {extra} of “{shown[:60]}…” are not in its source passages")

    # facts
    for ms, fn, where in FACTS:
        count = prose_all.count(flat(ms))
        report.check("facts", count == 1, f"{count} times in the manuscript (must be once): “{ms}”")
        try:
            ok = bool(fn())
        except Exception as e:                              # noqa: BLE001
            ok = False
            where = f"{where} ({e})"
        report.check("facts", ok, f"not confirmed by {where}: “{ms}”")

    # numbers: whatever the claims and facts do not cover must occur in the mapped source sections
    covered = sorted([flat(rename(c, src.edits)) for c, _ in CLAIMS] + [flat(f) for f, _, _ in FACTS],
                     key=len, reverse=True)
    for sid, prose in m.prose.items():
        spec = m.srcmap.get(sid, "")
        if spec in ("", "none"):
            continue
        rest = re.sub(r"(?:doi:)?10\.\d{4,9}/\S+", " ", prose)   # DOIs are identifiers, checked under references
        for c in covered:
            rest = rest.replace(c, " ")
        allowed = src_numbers(src, spec)
        for mt in NUM.finditer(rest):
            n = mt.group(0).replace(",", "")
            ctx = rest[max(0, mt.start() - 40):mt.end() + 25]
            report.check("numbers", n in allowed, f"[{sid}] {mt.group(0)} is not in its sources ({spec}): …{ctx}…")

    # scope
    scope_text = " ".join([" ".join(m.scope_prose.values()), m.refs_text, m.front.get("header", "")] +
                          [" ".join(c for r in rows for c in r) for _, rows in m.tables])
    for pattern in FORBIDDEN:
        hit = re.search(pattern, scope_text, flags=re.I)
        report.check("scope", hit is None, f"out-of-scope term /{pattern}/: …{scope_text[max(0, hit.start() - 40):hit.end() + 40] if hit else ''}…")

    # references
    cited = list(dict.fromkeys(m.cites))
    report.check("references", set(cited) == set(m.refs), f"cited but not listed {sorted(set(cited) - set(m.refs))}; "
                                                          f"listed but not cited {sorted(set(m.refs) - set(cited))}")
    report.check("references", cited == m.refs, "the reference list is not in order of first citation")
    allowed_dois = VERIFIED_DOIS | {d for d in (meta.get("preprint_doi"), meta.get("software_doi")) if d}
    report.check("references", m.dois <= allowed_dois, f"DOIs not verified: {sorted(m.dois - allowed_dois)}")
    report.check("references", m.arxiv <= VERIFIED_ARXIV, f"arXiv ids not verified: {sorted(m.arxiv - VERIFIED_ARXIV)}")

    # front matter
    report.check("front", flat(m.front["title"]) == flat(meta["title"]), f"title “{m.front['title']}” ≠ metadata.json")
    for a in meta["authors"]:
        name = f"{a['given_names']} {a['family_names']}"
        report.check("front", name in m.front["authors"], f"author {name} missing from the byline")
    for key, st in meta["statements"].items():
        report.check("front", flat(st) in m.front["declarations"], f"statement {key} differs from metadata.json")
    if final:
        report.check("front", m.front["raw_todo"] == "0" and m.front["placeholders"] == "0",
                     f"{m.front['raw_todo']} TODO(s) and {m.front['placeholders']} placeholder(s) left")
        for f in ("preprint_doi", "software_doi"):
            report.check("front", bool(meta.get(f)), f"{f} missing in metadata.json")


def check_pdf(m: Model, meta: dict, report: Report, final: bool) -> None:
    pdfs = sorted((HERE / "dist").glob("*.pdf"))
    if not pdfs:
        report.check("pdf", not final, "no PDF in preprint/dist (run build.py without --no-pdf)")
        return
    from pypdf import PdfReader
    pdf = pdfs[-1]
    r = PdfReader(pdf)
    report.check("pdf", flat(r.metadata.get("/Title", "")) == flat(meta["title"]), f"{pdf.name}: title differs")
    if final:
        report.check("pdf", pdf.stat().st_mtime >= BUILT.stat().st_mtime - 5, f"{pdf.name} is older than the HTML")
    pdf_text = re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in r.pages))
    pdf_nums = numbers(pdf_text)
    for attrs, rows in m.tables:
        want = [n for row in rows for c in row for n in numbers(flat(c))]
        pool = list(pdf_nums)
        missing = []
        for n in want:
            if n in pool:
                pool.remove(n)
            else:
                missing.append(n)
        report.check("pdf", not missing, f"{pdf.name}: numbers of table {attrs.get('data-src') or attrs.get('data-generated')} "
                                         f"missing from the PDF text: {missing[:8]}")


def summary(report: Report) -> tuple[int, int]:
    total = sum(report.counts.values())
    failed = sum(len(v) for v in report.failed.values())
    return total, failed


def mutate(m: Model, src: Sources, meta: dict) -> bool:
    """Alters one guarded item at a time and confirms that the checks catch it."""
    def fails(mm: Model) -> bool:
        r = Report()
        run_checks(mm, src, meta, False, r)
        return bool(r.failed)

    def bump(num: str) -> str:
        digits = [i for i, ch in enumerate(num) if ch.isdigit()]
        i = digits[-1]
        return num[:i] + str((int(num[i]) + 1) % 10) + num[i + 1:]

    results: dict[str, list[int]] = {}

    def tally(group: str, caught: bool):
        results.setdefault(group, [0, 0])
        results[group][0] += caught
        results[group][1] += 1

    for ti, (attrs, rows) in enumerate(m.tables):
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                mt = NUM.search(cell)
                if not mt:
                    continue
                mm = copy.deepcopy(m)
                mm.tables[ti][1][ri][ci] = cell[:mt.start()] + bump(mt.group(0)) + cell[mt.end():]
                tally("table cells", fails(mm))
    covered = [(flat(rename(c, src.edits)), "claims") for c, _ in CLAIMS] + [(flat(f), "facts") for f, _, _ in FACTS]
    for text_, group in covered:
        mt = NUM.search(text_)
        if not mt:
            continue
        sid = next((s for s, p in m.prose.items() if text_ in p), None)
        if sid is None:
            tally(group, False)
            continue
        mm = copy.deepcopy(m)
        mm.prose[sid] = mm.prose[sid].replace(text_, text_[:mt.start()] + bump(mt.group(0)) + text_[mt.end():], 1)
        tally(group, fails(mm))
    realistic = [0, 0]
    for sid, prose in m.prose.items():
        if m.srcmap.get(sid) in ("", "none"):
            continue
        for mt in NUM.finditer(prose):
            mm = copy.deepcopy(m)
            mm.prose[sid] = prose[:mt.start()] + "987654321" + prose[mt.end():]
            tally("prose numbers (replaced by an absent number)", fails(mm))
            mm.prose[sid] = prose[:mt.start()] + bump(mt.group(0)) + prose[mt.end():]
            realistic[0] += fails(mm)
            realistic[1] += 1
    for pattern in FORBIDDEN:
        term = {r"\bSystem 1\b": "System 1", r"\bSystem 2\b": "System 2", r"\bTGS\d": "TGS2602",
                r"\bjoules?\b": "joules"}.get(pattern, re.sub(r"\\b|\\d", "", pattern))
        mm = copy.deepcopy(m)
        sid = next(iter(mm.scope_prose))
        mm.scope_prose[sid] += f" {term} "
        tally("scope terms", fails(mm))
    mm = copy.deepcopy(m)
    mm.refs = mm.refs[:-1]
    tally("references", fails(mm))
    mm = copy.deepcopy(m)
    mm.dois = mm.dois | {"10.9999/not.verified"}
    tally("references", fails(mm))
    mm = copy.deepcopy(m)
    mm.cites = list(reversed(mm.cites))
    tally("references", fails(mm))

    ok = True
    print("\nmutation test (each item altered alone; the checks must fail):")
    for group, (caught, n) in results.items():
        ok &= caught == n
        print(f"  {group:48s} {caught}/{n} caught")
    print(f"  {'prose numbers (last digit changed; information)':48s} {realistic[0]}/{realistic[1]} caught — the rest "
          "also occur in their source sections, so only reading can tell")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--mutate", action="store_true", help="also show that every check catches an alteration")
    parser.add_argument("--final", action="store_true", help="also require every TODO filled and both DOIs set")
    args = parser.parse_args()
    if not BUILT.exists():
        sys.exit(f"{BUILT.relative_to(ROOT)} not found: run python3 preprint/build.py first")
    meta = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
    report = Report()
    src = load_sources(report)
    model = load_model(BUILT)
    run_checks(model, src, meta, args.final, report)
    check_pdf(model, meta, report, args.final)
    total, failed = summary(report)
    for group, n in report.counts.items():
        bad = report.failed.get(group, [])
        print(f"{group:11s} {n - len(bad)}/{n} passed")
        for msg in bad[:25]:
            print(f"    ✗ {msg}")
        if len(bad) > 25:
            print(f"    … and {len(bad) - 25} more")
    print(f"{total - failed}/{total} checks passed")
    e = src.edits
    print(f"(declared edits, preprint/edits.json: {len(e['rename'])} renamed terms, "
          f"{sum(len(t.get('drop_columns', [])) for t in e['tables'].values())} dropped columns, "
          f"{sum(len(t.get('cells', [])) for t in e['tables'].values())} cut cells)")
    open_items = B.todos(meta) + [f for f in ("preprint_doi", "software_doi") if not meta.get(f)]
    if open_items and not args.final:
        print(f"(draft: {len(open_items)} metadata item(s) to fill in before the upload; --final enforces it)")
    ok = failed == 0
    if args.mutate:
        ok &= mutate(model, src, meta) if ok else False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
