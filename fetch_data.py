"""
Download the UCI "Gas Sensor Array Drift Dataset at Different Concentrations" into data/.

UCI id 224 ("Gas Sensor Array Drift Dataset") is tried first through ucimlrepo, but
UCI does not currently allow it to be imported that way. The fallback downloads the
archive of id 270, the "... at Different Concentrations" extension: the same 13,910
measurements and 128 features (verified identical), plus the concentration of each
measurement.

Outputs (row i of every CSV is the same measurement):
    data/features.csv   128 sensor features, DR_1 … EMAd0.1_16
    data/targets.csv    batch, months, gas_id, gas, concentration_ppmv
    data/metadata.json  source, citations, gas/batch coding, UCI description
    data/raw/           original archive

    python3 fetch_data.py           # skip if already downloaded
    python3 fetch_data.py --force   # re-download
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS = [DATA_DIR / f for f in ("features.csv", "targets.csv", "metadata.json")]

REQUESTED_ID = 224
FALLBACK_ID = 270
# python.org macOS builds ship without system CA certificates; use certifi's bundle (verification stays on)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
ARCHIVE_URL = ("https://archive.ics.uci.edu/static/public/270/"
               "gas+sensor+array+drift+dataset+at+different+concentrations.zip")
METADATA_URL = "https://archive.ics.uci.edu/api/dataset?id=270"

# Gas naming of the class numbers. UCI's coding sentence reads 1=ethanol, 2=ethylene, 3=ammonia,
# 4=acetaldehyde, 5=acetone, 6=toluene, but every source that ties gases to sample counts (UCI's own
# per-batch table, Vergara et al. 2012-derived tables, Zhang & Zhang 2015, later papers) gives the naming
# below; the coding sentence equals that table's column-header order. The published results this project
# compares against use this naming (see GAS_LABEL_NOTE and dataset_adapter.GAS_NAMINGS).
GASES = {1: "acetone", 2: "acetaldehyde", 3: "ethanol", 4: "ethylene", 5: "ammonia", 6: "toluene"}
GAS_LABEL_NOTE = (
    "Two namings of the class numbers circulate. UCI's coding sentence: 1=ethanol, 2=ethylene, 3=ammonia, "
    "4=acetaldehyde, 5=acetone, 6=toluene. UCI's per-batch count table, Vergara et al. (2012)-derived tables and "
    "Zhang & Zhang (IEEE TIM 2015, Table I) associate the gases with sample counts that give 1=acetone, "
    "2=acetaldehyde, 3=ethanol, 4=ethylene, 5=ammonia, 6=toluene; the coding sentence equals that table's "
    "column-header order. The documented concentration ranges fit the coding sentence somewhat better (class 3: "
    "95% of samples at >= 50 ppmv, ammonia's documented lower bound; minimum 2.5 ppmv), and a response-strength "
    "check is inconclusive, so chemical identity is not provable from public material. `gas` uses the "
    "count-table naming, as the published results do; `gas_id` is the raw class number."
)
BATCH_MONTHS = {1: "1,2", 2: "3,4,8,9,10", 3: "11,12,13", 4: "14,15", 5: "16",
                6: "17,18,19,20", 7: "21", 8: "22,23", 9: "24,30", 10: "36"}
# Measurements per batch as published in UCI's table (column order as printed there) …
PUBLISHED_COUNTS = {1: (83, 30, 70, 98, 90, 74), 2: (100, 109, 532, 334, 164, 5),
                    3: (216, 240, 275, 490, 365, 0), 4: (12, 30, 12, 43, 64, 0),
                    5: (20, 46, 63, 40, 28, 0), 6: (110, 29, 606, 574, 514, 467),
                    7: (360, 744, 630, 662, 649, 568), 8: (40, 33, 143, 30, 30, 18),
                    9: (100, 75, 78, 55, 61, 101), 10: (600,) * 6}
# … and the published column that holds each gas_id 1..6 (the only order consistent with all 10 batches)
PUBLISHED_COLUMN_FOR_ID = (4, 3, 0, 1, 2, 5)
EXPECTED_COUNTS = {b: tuple(c[i] for i in PUBLISHED_COLUMN_FOR_ID) for b, c in PUBLISHED_COUNTS.items()}
# 8 features per sensor: steady-state change, its normalised form, and EMA transient
# extrema of the rising (i) / decaying (d) response for α = 0.001, 0.01, 0.1
FEATURES_PER_SENSOR = ["DR", "DR_norm", "EMAi0.001", "EMAi0.01", "EMAi0.1", "EMAd0.001", "EMAd0.01", "EMAd0.1"]
FEATURE_NAMES = [f"{f}_{s}" for s in range(1, 17) for f in FEATURES_PER_SENSOR]


def try_ucimlrepo() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    from ucimlrepo import fetch_ucirepo
    from ucimlrepo.fetch import DatasetNotFoundError

    try:
        dataset = fetch_ucirepo(id=REQUESTED_ID)
    except DatasetNotFoundError as e:
        print(f"ucimlrepo: {e}".split(" Please select")[0])
        return None
    return dataset.data.features, dataset.data.targets


def parse_batch(text: str, batch: int) -> tuple[list[list[float]], list[dict]]:
    """Parse one libsvm-style batch file: 'gas;conc 1:v1 2:v2 … 128:v128' per line."""
    features, targets = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        gas_id, conc = parts[0].split(";")
        row = [0.0] * len(FEATURE_NAMES)
        for item in parts[1:]:
            idx, value = item.split(":")
            row[int(idx) - 1] = float(value)
        features.append(row)
        targets.append({"batch": batch, "months": BATCH_MONTHS[batch], "gas_id": int(gas_id),
                        "gas": GASES[int(gas_id)], "concentration_ppmv": float(conc)})
    return features, targets


def download_archive() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "gas_sensor_array_drift_different_concentrations.zip"
    print(f"downloading UCI dataset {FALLBACK_ID} archive → {archive.relative_to(DATA_DIR.parent)}")
    with urllib.request.urlopen(ARCHIVE_URL, context=SSL_CONTEXT) as resp, open(archive, "wb") as out:
        shutil.copyfileobj(resp, out)
    with urllib.request.urlopen(METADATA_URL, context=SSL_CONTEXT) as resp:
        uci_meta = json.load(resp)["data"]

    features, targets = [], []
    with zipfile.ZipFile(archive) as z:
        for batch in range(1, 11):
            f, t = parse_batch(z.read(f"batch{batch}.dat").decode(), batch)
            features += f
            targets += t
    return pd.DataFrame(features, columns=FEATURE_NAMES), pd.DataFrame(targets), uci_meta


def validate(y: pd.DataFrame) -> None:
    counts = y.groupby(["batch", "gas_id"]).size().unstack(fill_value=0).reindex(columns=range(1, 7), fill_value=0)
    for batch, expected in EXPECTED_COUNTS.items():
        got = tuple(int(c) for c in counts.loc[batch])
        if got != expected:
            raise ValueError(f"batch {batch}: counts per gas_id {got} differ from UCI description {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UCI gas sensor array drift dataset into data/")
    parser.add_argument("--force", action="store_true", help="re-download even if data/ is populated")
    args = parser.parse_args()

    if not args.force and all(p.exists() for p in OUTPUTS):
        print(f"data already present in {DATA_DIR} (use --force to re-download)")
        return
    DATA_DIR.mkdir(exist_ok=True)

    fetched = try_ucimlrepo()
    if fetched is not None:
        X, y = fetched
        meta = {"source": f"ucimlrepo id={REQUESTED_ID}"}
    else:
        X, y, uci_meta = download_archive()
        validate(y)
        meta = {
            "source": ARCHIVE_URL,
            "uci_id": FALLBACK_ID,
            "note": f"Extension of UCI id {REQUESTED_ID} (identical features) that adds concentration labels.",
            "gases": GASES,
            "gas_label_note": GAS_LABEL_NOTE,
            "batch_months": BATCH_MONTHS,
            "features": "8 per sensor × 16 sensors: " + ", ".join(FEATURES_PER_SENSOR),
            "license_note": "Research use only; cite Vergara et al. (2012) and Rodriguez-Lujan et al. (2014).",
            "uci_metadata": uci_meta,
        }

    X.to_csv(DATA_DIR / "features.csv", index=False)
    y.to_csv(DATA_DIR / "targets.csv", index=False)
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"saved to {DATA_DIR}: features {X.shape}, targets {y.shape} {list(y.columns)}")


if __name__ == "__main__":
    main()
