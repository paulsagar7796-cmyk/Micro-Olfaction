"""
Loading and feature extraction for the UCI "Twin gas sensor arrays" dataset (fetch_twin_arrays.py).

Each of 640 trials is a 600-s, 100-Hz resistance recording of one board exposed to one gas at one
concentration level, with a clean pulse shape verified on a sample of trials before any feature or
label was computed: baseline until ~57 s, a plateau from ~110-150 s, then recovery. The four windows
below are fixed from that shape alone, identically for every trial:

    baseline   [5, 50] s     before the gas reaches the sensors
    rising     [55, 100] s   the onset transient
    plateau    [110, 150] s  the steady response
    decaying   [160, 400] s  the recovery transient

Features per sensor, in System 2+'s own feature family (Vergara et al., Sens. Actuators B 166-167,
2012; recomputed here from these raw traces, not a byte-for-byte reproduction of their published
values, since their extraction code is not published): steady-state conductance change and its
normalised form, and three-timescale EMA extrema of the rising and decaying conductance transients,
all oriented so a stronger response is a larger positive number. 8 sensors x 8 features = 64 raw
features ("all"), plus 64 "ratio" features (ratio_pattern: log-shape across sensors with the overall
amplitude removed, as tune_system2.raw_features does for the drift dataset) = 128 ("all+ratio"),
matching System 2+'s registered feature count on a smaller array.

    python3 twin_arrays.py --jobs 10   # extract and cache results/twin_arrays_features.csv (~10 min)
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fetch_twin_arrays import ARCHIVE, GASES, trial_names

FS = 100.0                      # Hz
WINDOWS = {"baseline": (5, 50), "rising": (55, 100), "plateau": (110, 150), "decaying": (160, 400)}
ALPHAS = (0.1, 0.01, 0.001)
FEATURE_NAMES = ["DR", "DR_norm"] + [f"EMAi{a}" for a in ALPHAS] + [f"EMAd{a}" for a in ALPHAS]
FLOORS = {"DR": 1e-4, "DR_norm": 1e-4, **{f"EMAi{a}": 1e-6 for a in ALPHAS}, **{f"EMAd{a}": 1e-6 for a in ALPHAS}}
N_SENSORS = 8
NAME_RE = re.compile(r"B(\d)_G([A-Za-z]+)_F(\d+)_R(\d+)\.txt$")
EXPECTED_TRIALS = 640


def slot(t: np.ndarray, lo: float, hi: float) -> slice:
    return slice(int(lo * FS), int(hi * FS))


def ema_extreme(dg: np.ndarray, alpha: float, take_max: bool) -> float:
    """The causal EMA of a difference series, reset to 0 at the window start; its largest value in the
    requested direction (unreached windows return 0)."""
    if len(dg) == 0:
        return 0.0
    y = np.empty(len(dg))
    y[0] = alpha * dg[0]
    for k in range(1, len(dg)):
        y[k] = (1 - alpha) * y[k - 1] + alpha * dg[k]
    return float(y.max() if take_max else -y.min())


def trial_features(r: np.ndarray) -> np.ndarray:
    """(8, 8) feature matrix [sensor, FEATURE_NAMES] for one trial's (60000, 8) resistance trace."""
    g = 1.0 / r                                                     # conductance: n-type sensors, response = rise
    base = g[slot(g, *WINDOWS["baseline"])].mean(axis=0)
    plateau = g[slot(g, *WINDOWS["plateau"])].mean(axis=0)
    dr = plateau - base
    dr_norm = dr / base
    out = np.zeros((N_SENSORS, len(FEATURE_NAMES)))
    out[:, 0], out[:, 1] = dr, dr_norm
    for s in range(N_SENSORS):
        rise = np.diff(g[slot(g, *WINDOWS["rising"]), s])
        decay = np.diff(g[slot(g, *WINDOWS["decaying"]), s])
        for j, a in enumerate(ALPHAS):
            out[s, 2 + j] = ema_extreme(rise, a, take_max=True)
            out[s, 5 + j] = ema_extreme(decay, a, take_max=False)
    return out


def parse_name(name: str) -> dict:
    m = NAME_RE.search(name)
    board, gas, level, rep = m.groups()
    return {"board": int(board), "gas": GASES[gas], "gas_code": gas, "level": int(level), "rep": int(rep)}


def process_one(name: str) -> dict:
    with zipfile.ZipFile(ARCHIVE) as z, z.open(name) as f:
        d = np.loadtxt(io.TextIOWrapper(f))
    feats = trial_features(d[:, 1:])
    row = parse_name(name)
    row["name"] = name
    row["n_samples"] = len(d)
    for s in range(N_SENSORS):
        for j, fname in enumerate(FEATURE_NAMES):
            row[f"{fname}_{s + 1}"] = feats[s, j]
    return row


def log_pattern_row(row: dict) -> dict:
    out = {}
    for fname in FEATURE_NAMES:
        v = np.array([row[f"{fname}_{s + 1}"] for s in range(N_SENSORS)])
        lv = np.log(np.maximum(v, FLOORS[fname]))
        pattern = lv - lv.mean()
        for s in range(N_SENSORS):
            out[f"ratio_{fname}_{s + 1}"] = pattern[s]
    return out


def load(path: Path = Path("results/twin_arrays_features.csv")) -> "pandas.DataFrame":
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip")


def feature_columns(kind: str = "all+ratio") -> list[str]:
    raw = [f"{f}_{s + 1}" for s in range(N_SENSORS) for f in FEATURE_NAMES]
    if kind == "all":
        return raw
    ratio = [f"ratio_{c}" for c in raw]
    return raw + ratio if kind == "all+ratio" else ratio


def self_check(names: list[str]) -> None:
    """The archive holds the registered 640 trials, and the extractor is deterministic: a trial processed
    twice gives identical features."""
    if len(names) != EXPECTED_TRIALS:
        raise RuntimeError(f"{len(names)} trials in the archive, expected {EXPECTED_TRIALS}")
    for name in (names[0], names[len(names) // 2], names[-1]):
        a, b = process_one(name), process_one(name)
        if any(a[k] != b[k] for k in a):
            raise RuntimeError(f"{name}: features differ between two passes")
    print(f"self-check: {len(names)} trials; the extractor repeats exactly on 3 trials")


def extract(out: Path, jobs: int) -> None:
    t0 = time.time()
    with zipfile.ZipFile(ARCHIVE) as z:
        names = trial_names(z)
    self_check(names)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(process_one, names, chunksize=4))
    for row in rows:
        row.update(log_pattern_row(row))
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} trials, {len(FEATURE_NAMES) * N_SENSORS * 2} features each  [{time.time() - t0:.0f} s]")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/twin_arrays_features.csv"))
    args = parser.parse_args()
    extract(args.out, args.jobs)


if __name__ == "__main__":
    main()
