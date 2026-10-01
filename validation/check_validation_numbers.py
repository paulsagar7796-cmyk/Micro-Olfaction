"""
Checks the numbers of validation/RESULTS_S2COMPOUNDS.md (SOLAR+ on the twin gas sensor arrays): each is
recomputed from the result files and must appear verbatim in the section that states it. Ported unchanged
from group C of check_paper_v4_numbers.py at tag pre-v4 (only the document's path differs).

    python3 validation/check_validation_numbers.py            # the checks
    python3 validation/check_validation_numbers.py --mutate   # also: altering any checked number must make its check fail

A number that exists only in the run's log (the Wilcoxon p) is read from that log, which
validation/check_validation_outputs.py compares with the committed version like any result file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)
DOCS = {"compounds": "validation/RESULTS_S2COMPOUNDS.md"}


def flat(s: str) -> str:
    """Bold markers dropped; line breaks and quote markers count as single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"\n> ?", " ", s.replace("**", ""))).strip()


class Docs:
    def __init__(self, texts: dict[str, str]):
        self.sections = {name: {p.split("\n", 1)[0].strip(): flat(p) for p in re.split(r"^## ", t, flags=re.M)[1:]}
                         for name, t in texts.items()}
        self.checks: list[tuple[str, str, str, str]] = []            # group, doc, section, string
        self.failed: list[str] = []

    def section(self, doc: str, prefix: str) -> str:
        return next(v for k, v in self.sections[doc].items() if k.startswith(prefix))

    def expect(self, group: str, doc: str, prefix: str, *strings: str) -> None:
        body = self.section(doc, prefix)
        for s in strings:
            self.checks.append((group, doc, prefix, s))
            if flat(s) not in body:
                self.failed.append(f"[{group}] {DOCS[doc]} § {prefix}: missing “{s}”")


def log_line(path: str, pattern: str) -> re.Match:
    m = re.search(pattern, Path(path).read_text())
    if not m:
        raise RuntimeError(f"{path}: no line matching {pattern!r}")
    return m


def group_c(docs: Docs) -> None:
    """System 2+ on the twin gas sensor arrays."""
    comp = pd.read_csv("results/system2_compounds.csv").set_index("method")
    folds = pd.read_csv("results/system2_compounds_folds.csv").pivot(index="target_board", columns="method",
                                                                     values="accuracy")
    expl = pd.read_csv("results/system2_compounds_exploratory.csv").set_index("target_board")
    bud = pd.read_csv("results/system2_compounds_budget.csv").groupby("calibration_per_gas")["accuracy"].mean()
    gases = ("carbon monoxide", "ethanol", "ethylene", "methane")
    for method, label in (("System 2+", "System 2+"), ("System 2+ without pseudo-labels", "System 2+ without pseudo-labels"),
                          ("System 2", "System 2 (no alignment or calibration)"),
                          ("random forest", "Random forest (source + the 20 calibration trials)")):
        r = comp.loc[method]
        docs.expect("C", "compounds", "Registered results",
                    f"{label} | {r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f} | "
                    + " | ".join(f"{r[f'recall_{g}_mean']:.3f}" for g in gases))
    s2p, s2 = folds["System 2+"], folds["System 2"]
    diff = s2p - s2
    docs.expect("C", "compounds", "Registered results",
                "System 2+ | " + " | ".join(f"{v:.3f}" for v in s2p),
                "System 2 | " + " | ".join(f"{v:.3f}" for v in s2),
                "Difference | " + " | ".join(f"{v:+.3f}" for v in diff))
    wil = log_line("results/logs/system2_compounds.log", r"\+([\d.]+) points, p = ([\d.]+)\)")
    docs.expect("C", "compounds", "Registered results",
                "Adaptation improved accuracy on 5 of 5 held-out boards; with 5 boards, no two-sided test can reach "
                "p < 0.05 (the registered Wilcoxon gives " + wil[2] + ", its minimum)",
                f"a mean gain of at least 5 points, was met (+{100 * diff.mean():.1f})")
    if int((diff > 0).sum()) != 5 or float(wil[1]) != round(100 * diff.mean(), 1):
        docs.failed.append("[C] the five board differences, their mean and the log disagree")
    docs.expect("C", "compounds", "Bottom line",
                f"from a board it was never trained on with {100 * comp.loc['System 2+'].accuracy_mean:.1f}% accuracy",
                f"reached only {100 * s2[5]:.0f}%", f"System 2+ reached {100 * s2p[5]:.0f}%",
                f"adaptation added a small but consistent {100 * diff[[1, 2, 3, 4]].min():.1f}-{100 * diff[[1, 2, 3, 4]].max():.1f} points",
                f"about {100 * diff.median():.1f} points (the median); the mean, {100 * diff.mean():.1f}, mostly reflects board 5",
                "Adaptation improved accuracy on 5 of 5 held-out boards; with 5 boards, no two-sided test can reach p < 0.05.",
                f"does better: {100 * comp.loc['random forest'].accuracy_mean:.1f}%")
    docs.expect("C", "compounds", "Exploratory",
                "Seeds where System 2+ beats System 2 | "
                + " | ".join(f"{int(r.seeds_system2plus_beats_system2)}/{int(r.seeds)}" for _, r in expl.iterrows()),
                "System 2+, registered order (sorted by gas) | " + " | ".join(f"{v:.3f}" for v in expl.system2plus_registered_order),
                "System 2+, random order | " + " | ".join(f"{v:.3f}" for v in expl.system2plus_random_order),
                "System 2, unadapted | " + " | ".join(f"{v:.3f}" for v in expl.system2_unadapted),
                f"board 5 reaches {100 * expl.loc[5, 'system2plus_random_order']:.1f}% against "
                f"{100 * expl.loc[5, 'system2plus_registered_order']:.1f}%",
                # this table copies the log, which rounds with numpy (ties to even): the 20-per-gas mean is exactly
                # 0.9625 (385 of 400 correct), shown as 0.962
                "Accuracy | " + " | ".join(f"{float(np.round(bud[n], 3)):.3f}" for n in (5, 10, 20)))
    if not (expl.system2plus_registered_order.round(12) == s2p.round(12)).all():
        docs.failed.append("[C] the exploratory file's registered-order column differs from the fold results")


def mutate(docs: Docs) -> list[str]:
    """For each check, change one digit of the checked string where it stands in the document and confirm the
    check then fails; a check that still passes would not notice that number being edited."""
    survivors = []
    for group, doc, prefix, s in docs.checks:
        target = flat(s)
        body = docs.section(doc, prefix)
        if target not in body:
            continue
        i = next((k for k, ch in enumerate(target) if ch.isdigit()), None)
        if i is None:
            continue
        mutated = target[:i] + ("1" if target[i] != "1" else "2") + target[i + 1:]
        edited = body.replace(target, mutated, 1)
        if target in edited:
            survivors.append(f"[{group}] {DOCS[doc]} § {prefix}: “{s}” occurs more than once")
    return survivors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--mutate", action="store_true", help="also check that each check notices an edited number")
    args = parser.parse_args()
    docs = Docs({k: Path(v).read_text() for k, v in DOCS.items()})
    group_c(docs)
    n = len(docs.checks)
    for f in docs.failed:
        print(f)
    passed = n - sum(1 for f in docs.failed if "missing" in f)
    print(f"{passed}/{n} checks of validation/RESULTS_S2COMPOUNDS.md passed"
          + ("" if not docs.failed else f"; {len(docs.failed)} problems"))
    status = 1 if docs.failed else 0
    if args.mutate:
        survivors = mutate(docs)
        for s in survivors:
            print("mutation survived:", s)
        print(f"mutation test: {n - len(survivors)}/{n} checks fail when their number is edited")
        status |= 1 if survivors else 0
    sys.exit(status)


if __name__ == "__main__":
    main()
