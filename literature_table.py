"""
Unified literature table: every published row in literature/published_results.csv next to System 2+ run
under the same protocol, with the protocol differences made explicit.

    python3 literature_table.py

System 2+ rows: Settings 1 and 2 from results/system2_combined.csv (K = 3 + class weight + pseudo-labels);
the multi-source protocol (train on Batches 1 and 2, test Batch K = 3…10) is run here with the same
configuration. All averages are plain means of per-batch accuracies (our convention and Zhang & Zhang's);
rows whose paper used another convention are recomputed from their per-batch values, with the paper's
figure in the notes. "6–10" is the mean over Batches 6–10.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_drift as D
import benchmark_literature_comparison as L
import combine_system2 as C
from benchmark_batch1 import DEFAULT_SEEDS
from dataset_adapter import load_batch

FULL = "K=3 + class weight + pseudo-labels"
OURS_MACS = 3299                      # per sample, 3.1 pJ each (combine_system2.py)
ELM_MACS = 134128                     # 1000 RBF hidden neurons on 128 features (DAELM, ODAELM)


def multi_source(job: tuple[int, int, int]) -> dict:
    warnings.simplefilter("ignore")
    target, seed, budget = job
    tb = load_batch(target)
    lab = L.balanced(tb.gas, D.stream_order(seed, target, len(tb)), budget) if budget else np.array([], dtype=int)
    r = C.adapt_target([1, 2], target, seed, lab, budget)
    return {"target": target, "seed": seed, "budget": budget, "accuracy": float(np.mean(r["pred"] == r["truth"]))}


def ours_row(df: pd.DataFrame, targets: list[int]) -> tuple[list[float], float, float, float]:
    per = df.pivot_table(index="seed", columns="target", values="accuracy") * 100
    per_batch = [per[t].mean() if t in per else np.nan for t in range(2, 11)]
    avail = per[[t for t in targets if t in per]].mean(axis=1)
    late = per[[t for t in range(6, 11) if t in per]].mean(axis=1)
    return per_batch, avail.mean(), late.mean(), late.std(ddof=1)


def main() -> None:
    pub = L.load_published()
    comb = pd.read_csv("results/system2_combined.csv")
    comb = comb[comb.config == FULL]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        ms = pd.DataFrame(pool.map(multi_source, [(t, s, b) for t in range(3, 11) for s in DEFAULT_SEEDS for b in (0, 20)]))

    lines = []   # (group, method, year, labels, uses target batch, labelled scored, per_batch, avg, late, late_sd, macs, note)
    for group, setting in (("Setting 1: Batch 1 → K", "1"), ("Setting 2: Batch K−1 → K", "2"),
                           ("Setting 2, online stream", "2-online"), ("Multi-source: Batches 1+2 → K", "multi-source")):
        for r in (r for r in pub if r["setting"] == setting):
            v = np.array(r["per_batch"])
            macs = ELM_MACS if "ELM" in r["method"] else None
            conv = r["average_convention"]
            note = "" if conv == "per-batch mean" else f"paper reports {r['reported_average']} ({conv})"
            lines.append((group, r["method"], r["year"], r["target_labels_per_batch"], r["uses_unlabelled_target_batch"],
                          r["labelled_samples_scored"], list(v), float(np.nanmean(v)), float(np.nanmean(v[4:])), None, macs, note))
        if setting in ("1", "2"):
            s = int(setting)
            for budget in (0, 20, 30, 50):
                for sel, tag in ((("none", ""),) if budget == 0 else (("balanced", ", balanced"), ("ssa", ", DAELM selection"))):
                    d = comb[(comb.setting == s) & (comb.budget == budget) & (comb.selection == sel)]
                    pb, avg, late, sd = ours_row(d, list(range(2, 11)))
                    lines.append((group, f"System 2+{tag}", "ours", str(budget), "no", "no", pb, avg, late, sd, OURS_MACS, ""))
        if setting == "multi-source":
            for budget in (0, 20):
                d = ms[ms.budget == budget]
                pb, avg, late, sd = ours_row(d, list(range(3, 11)))
                lines.append((group, f"System 2+{', balanced' if budget else ''}", "ours", str(budget), "no", "no",
                              pb, avg, late, sd, OURS_MACS, ""))

    out = []
    group = None
    for g, m, y, lab, unl, sc, pb, avg, late, sd, macs, note in lines:
        if g != group:
            group = g
            print(f"\n{g}")
            print(f"{'method':<34}{'year':>6}{'labels':>24}{'target batch':>14}{'labels scored':>18}"
                  + "".join(f"{f'B{k}':>7}" for k in range(2, 11)) + f"{'avg':>7}{'6–10':>14}{'MACs':>10}")
        late_s = f"{late:.1f}" + (f" ± {sd:.1f}" if sd is not None else "")
        print(f"{m:<34}{y:>6}{lab:>24}{unl[:13]:>14}{sc[:17]:>18}"
              + "".join(f"{v:>7.1f}" if not np.isnan(v) else f"{'—':>7}" for v in pb)
              + f"{avg:>7.1f}{late_s:>14}{(f'{macs:,}' if macs else 'n/s'):>10}" + (f"   {note}" if note else ""))
        out.append({"protocol": g, "method": m, "year": y, "target_labels": lab, "uses_unlabelled_target_batch": unl,
                    "labelled_scored": sc, **{f"batch_{k}": round(v, 2) for k, v in zip(range(2, 11), pb)},
                    "average": round(avg, 6), "average_6_10": round(late, 6),
                    "std_6_10": round(sd, 6) if sd is not None else "", "inference_macs": macs or "", "note": note})
    print("\nAverages: plain means of per-batch accuracies over the batches available (2–10; 3–10 multi-source). "
          "'target batch' = uses the whole unlabelled target batch; 'labels scored' = labelled samples counted in "
          "accuracy. MACs per inference; n/s = not stated in the paper's text. System 2+ = mean over 5 seeds (± std, 6–10).")
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv("results/literature_table.csv", index=False)
    print("wrote results/literature_table.csv")


if __name__ == "__main__":
    main()
