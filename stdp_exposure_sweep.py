"""
STDP over-training sweep: does prolonged unsupervised STDP drive the layer into starvation?

For every (--stdp-duration, seed) one STDP layer is trained once and paused at a series of
cumulative *exposures* — seconds of input spikes the layer has learned from,
exposure = presentations × stdp-duration. At each checkpoint the frozen layer is evaluated
exactly as benchmark_batch1.py does (same data, readouts and layer-health metrics), so a
checkpoint with whole epochs reproduces the benchmark run with that --epochs.

    python3 stdp_exposure_sweep.py --feature-mode kinetics
    python3 stdp_exposure_sweep.py --feature-mode kinetics --stdp-duration 1.0 --exposures 0 71 355 1065

With 355 training samples, 1065 s at 1.0 s per presentation is the benchmark default (3 epochs).
The default exposures line up for 1.0 s and 0.2 s presentations, so presentation length and
total exposure can be told apart. Health metrics are described in benchmark_batch1.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from benchmark_batch1 import (DEFAULT_SEEDS, HEALTH, READOUTS, add_pipeline_args, evaluate_layer, make_stdp_layer,
                              prepare_data, stdp_presentations, train_stdp)
from dataset_adapter import GAS_ORDER
from stdp_layer import UNASSIGNED

DEFAULT_EXPOSURES = [0, 18, 36, 71, 178, 355, 710, 1065, 1775]  # seconds of STDP learning
DEFAULT_STDP_DURATIONS = [1.0, 0.2]                              # seconds per presentation


def run_trajectory(args: argparse.Namespace, stdp_duration: float, seed: int) -> list[dict]:
    """Train one STDP layer through every exposure checkpoint; one result row per checkpoint."""
    if "linear" in args.readout:  # the probe runs on raw spike counts, as in the benchmark
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data = prepare_data(args, seed, say=lambda *_, **__: None)
    layer = make_stdp_layer(args, data, seed)
    train_labels = data.labels[data.train_idx]
    epoch_len = (len(set(train_labels)) * max(np.sum(train_labels == c) for c in set(train_labels))
                 if args.balance else len(data.train_idx))
    targets = sorted({int(round(e / stdp_duration)) for e in args.exposures})
    # Whole passes generated exactly as the benchmark does, so every prefix equals a benchmark run.
    order = stdp_presentations(data.train_idx, data.labels, math.ceil(targets[-1] / len(data.train_idx)),
                               args.balance, np.random.default_rng(seed))[:targets[-1]]

    rows, done = [], 0
    for target in targets:
        train_stdp(layer, data, order[done:target], stdp_duration)
        done = target
        r = evaluate_layer(layer, data, args, seed)
        row = {"stdp_duration_s": stdp_duration, "seed": seed, "presentations": target, "epoch_length": epoch_len,
               "epochs": target / epoch_len, "exposure_s": target * stdp_duration}
        row.update({f"{n}_test": r.accuracy[n] for n in r.readouts})
        row.update({f"{n}_train": r.train_accuracy[n] for n in r.readouts})
        if r.probe_input is not None:
            row["probe_input_test"] = r.probe_input
        row.update({"nc_raw": r.ref_raw, "nc_lif": r.ref_lif}, **r.health)
        row.update({f"neurons_{c}": v for c, v in r.neurons_per_class.items()})
        rows.append(row)
    return rows


def _mean_std(values: Sequence[float], pct: bool) -> str:
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{100 * v.mean():5.1f} ± {100 * sd:4.1f}" if pct else f"{v.mean():5.2f} ± {sd:4.2f}"


def print_tables(rows: list[dict], readouts: list[str], default_duration: float) -> None:
    for d in sorted({r["stdp_duration_s"] for r in rows}, reverse=True):
        sub = [r for r in rows if r["stdp_duration_s"] == d]
        seeds = sorted({r["seed"] for r in sub})
        steps = sorted({r["presentations"] for r in sub})
        at = lambda p: [r for r in sub if r["presentations"] == p]
        print(f"\n═══ STDP learning from {d:g} s per presentation — mean ± std over {len(seeds)} seeds ═══")

        print(f"\n{'exposure':>9} {'epochs':>7} │" + "".join(f" {n:>13}" for n in readouts) + " │"
              + "".join(f" {HEALTH[k]:>13}" for k in HEALTH))
        for p in steps:
            group = at(p)
            acc = "".join(f" {_mean_std([g[f'{n}_test'] for g in group], True):>13}" for n in readouts)
            health = "".join(f" {_mean_std([g[k] for g in group], k == 'top3_share'):>13}" for k in HEALTH)
            print(f"{group[0]['exposure_s']:>8.0f}s {group[0]['epochs']:>7.2f} │{acc} │{health}")

        print(f"\nNeurons per class (mean over seeds)\n{'exposure':>9} │"
              + "".join(f" {c[:12]:>12}" for c in GAS_ORDER + [UNASSIGNED]))
        for p in steps:
            group = at(p)
            print(f"{group[0]['exposure_s']:>8.0f}s │"
                  + "".join(f" {np.mean([g[f'neurons_{c}'] for g in group]):>12.1f}" for c in GAS_ORDER + [UNASSIGNED]))

        # Best exposure per readout vs the benchmark default (3 whole epochs at the full sample length)
        default_p = 3 * sub[0]["epoch_length"] if abs(d - default_duration) < 1e-9 else None
        for n in readouts:
            best = max(steps, key=lambda p: np.mean([g[f"{n}_test"] for g in at(p)]))
            line = (f"best {READOUTS[n]}: {at(best)[0]['exposure_s']:.0f} s "
                    f"({100 * np.mean([g[f'{n}_test'] for g in at(best)]):.1f}%)")
            if default_p in steps and best != default_p:
                by_seed = lambda p: {g["seed"]: g[f"{n}_test"] for g in at(p)}
                diffs = [by_seed(best)[s] - by_seed(default_p)[s] for s in seeds]
                line += (f" vs {at(default_p)[0]['exposure_s']:.0f} s default: {100 * np.mean(diffs):+.1f} points, "
                         f"better on {sum(x > 0 for x in diffs)}/{len(seeds)} seeds")
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_pipeline_args(parser)
    parser.set_defaults(readout=["vote", "spiking", "linear"])
    parser.add_argument("--stdp-duration", type=float, nargs="+", default=DEFAULT_STDP_DURATIONS,
                        help="seconds of each training sample STDP learns from; one trajectory per value")
    parser.add_argument("--exposures", type=float, nargs="+", default=DEFAULT_EXPOSURES,
                        help="checkpoints, in seconds of cumulative STDP learning (0 = untrained random weights)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel trajectories")
    parser.add_argument("--out", type=Path, default=None, help="CSV with one row per (stdp-duration, seed, exposure)")
    args = parser.parse_args()
    args.readout = list(dict.fromkeys(args.readout))
    if any(not 0 < d <= args.duration for d in args.stdp_duration):
        parser.error("every --stdp-duration must be > 0 and ≤ --duration")
    if any(e < 0 for e in args.exposures):
        parser.error("--exposures must be ≥ 0")

    jobs = [(d, s) for d in args.stdp_duration for s in args.seeds]
    print(f"STDP exposure sweep: {args.feature_mode} features, {args.neurons} neurons, readouts "
          f"{' '.join(args.readout)}; STDP durations {args.stdp_duration} s; exposures {args.exposures} s; "
          f"seeds {args.seeds} — {len(jobs)} trajectories on {min(args.jobs, len(jobs))} workers", flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        futures = {pool.submit(run_trajectory, args, d, s): (d, s) for d, s in jobs}
        for future in as_completed(futures):
            d, s = futures[future]
            rows.extend(future.result())
            print(f"  done: {d:g} s presentations, seed {s}  [{time.time() - t0:.0f} s]", flush=True)
    rows.sort(key=lambda r: (-r["stdp_duration_s"], r["seed"], r["presentations"]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.out}")
    print_tables(rows, args.readout, args.duration)


if __name__ == "__main__":
    main()
