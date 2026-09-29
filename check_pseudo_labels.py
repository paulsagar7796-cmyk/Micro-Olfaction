"""
Which samples does System 2+ pseudo-label? Its confidence test normalises the six one-vs-rest
probabilities, so a sample can pass even when every unit rejects it (all six scores negative: each
unit says "not my gas"), provided the least-negative unit dominates the others. This counts how many
of the paper's pseudo-labels come from such samples, and how often they are right.

    python3 check_pseudo_labels.py          # about 3 minutes on 10 cores

Runs: the paper's System 2+ runs through device_reference.py in float64 with the paper's arithmetic
(which reproduces scikit-learn; see check_device_reference.py): Settings 1–2, Batches 6–10, every label
budget and selection, 5 seeds; the sequential protocol at 5/10/12/15 labels per gas, 15 seeds.
Output: results/pseudo_label_scores.csv, one row per protocol (and all runs together).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import check_device_reference as K
import combine_system2 as C
import device_reference as R
import improve_system2 as I
import sequential_system2plus as S
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import load_batch


class Recording(R.Device):
    """The device, recording the winning score of every pseudo-labelled sample and whether it was right."""

    def classify(self, y: np.ndarray, truth: np.ndarray, pl: bool):
        r = self.readout
        cfg, used = C.PSEUDO, dict.fromkeys(range(len(r.classes)), 0)
        frac = np.array([cfg["frac"]], dtype=self.ar.dt)
        pred = np.empty(len(y), dtype=object)
        for j in range(len(y)):
            dec = r.decision(y[j])
            k = int(np.argmax(dec))
            pred[j] = r.classes[k]
            if pl and r.probabilities_ok(dec, k, cfg["tau"]) and used[k] < cfg["cap"]:
                r.partial_fit(y[j:j + 1], r.classes[k:k + 1], frac)
                used[k] += 1
                self.log.append((float(dec[k]), bool(r.classes[k] == truth[j])))
        return pred, 0, 0


def run(sc: R.Scenario, visits: set[int]) -> list[tuple[float, bool]]:
    dev = Recording(sc.model, sc.mu, sc.var, R.PAPER64, sc.paper_form)
    passes, log = R.Passes(R.PAPER64, sc.passes_key), []
    for b, labelled, later in sc.visits:
        gas = load_batch(b).gas
        y = dev.align(R.features(b, R.PAPER64)[np.concatenate([labelled, later])])
        dev.calibrate(y[:len(labelled)], gas[labelled], sc.k, passes.epochs(len(labelled), R.TUNED["epochs"]))
        dev.log = []
        dev.classify(y[len(labelled):], gas[later], sc.pl)
        if b in visits:
            log += dev.log
    return log


def job(spec: tuple) -> tuple[str, list]:
    if spec[0] == "sequential":
        return "sequential", run(R.sequential_scenario(*spec[1:]), set(I.TEST))
    return "settings", run(R.settings_scenario(*spec[1:]), set(I.TEST))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/pseudo_label_scores.csv"))
    args = parser.parse_args()
    specs = [("sequential", b, s, 3.0, True, True) for b in S.BUDGETS for s in K.SEQ_SEEDS]
    specs += [("settings", st, tg, s, b, sel, 3.0, True, True) for st in I.SETTINGS for tg in I.TEST
              for s in DEFAULT_SEEDS for b, sel in I.COMBOS]
    logs = {"sequential": [], "settings": []}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for protocol, log in pool.map(job, specs, chunksize=4):
            logs[protocol] += log
    rows = []
    for name, log in (("sequential", logs["sequential"]), ("settings", logs["settings"]),
                      ("all", logs["sequential"] + logs["settings"])):
        m = np.array([s for s, _ in log])
        ok = np.array([c for _, c in log])
        rejected = m < 0
        rows.append({"protocol": name, "pseudo_labels": len(m), "share_all_units_reject": rejected.mean(),
                     "precision_all_reject": ok[rejected].mean(), "precision_winner_accepts": ok[~rejected].mean(),
                     "median_winning_score_when_all_reject": float(np.median(m[rejected]))})
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Pseudo-labels on Batches 6–10 in the paper's System 2+ runs: share taken from samples that every one-vs-rest "
          "unit rejects (all six scores < 0), and precision")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
