"""
Download the UCI "Twin gas sensor arrays" dataset (id 361) into data/twin_arrays/.

Five physically separate 8-sensor MOx arrays ("boards"), each exposed on its own testing days to the
same 40 conditions (4 gases: ethanol, methane, ethylene, CO; 10 concentration levels each), in random
order, 600 s per trial at 100 Hz, 2-4 repetitions per condition (640 trials, 2.77 GB uncompressed).
Used as an independent real-sensor test of System 2+ on discrete, chemically fixed classes (unlike the
UCI home-activity dataset's wine and banana); see validation/PROTOCOL_S2COMPOUNDS.md.

Kept zipped (data/twin_arrays/raw/uci_361.zip, checksum-verified): extracting all 640 files would use
2.77 GB on disk for no benefit, since they are read directly from the archive.

    python3 fetch_twin_arrays.py           # skip if already downloaded
    python3 fetch_twin_arrays.py --force   # re-download

Source: Fonollosa, Fernandez, Gutierrez-Galvez, Huerta & Marco (2016), Sensors and Actuators B: Chemical
236, "Calibration transfer and drift counteraction in chemical sensor arrays using direct standardization",
doi:10.1016/j.snb.2016.05.089. UCI Machine Learning Repository, doi:10.24432/C5MW3K. Research use only,
commercial use excluded (UCI page).
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

DATA_DIR = Path(__file__).resolve().parent / "data" / "twin_arrays"
ARCHIVE = DATA_DIR / "raw" / "uci_361.zip"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/361/twin+gas+sensor+arrays.zip"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SHA256 = "e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167"

GASES = {"CO": "carbon monoxide", "Ea": "ethanol", "Ey": "ethylene", "Me": "methane"}
BOARDS = (1, 2, 3, 4, 5)
CONCENTRATION_LEVELS = tuple(10 * i for i in range(1, 11))   # F010 .. F100, one gas's own 1..10 scale


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify() -> bool:
    got = sha256(ARCHIVE) if ARCHIVE.exists() else "missing"
    print(f"{'OK      ' if got == SHA256 else 'MISMATCH'} {ARCHIVE.relative_to(DATA_DIR.parent.parent)}")
    return got == SHA256


def trial_names(z: zipfile.ZipFile) -> list[str]:
    return sorted(n for n in z.namelist() if n.split("/")[-1][:1] == "B" and n.endswith(".txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the archive is present")
    args = parser.parse_args()
    if args.force or not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ARCHIVE_URL}")
        with urllib.request.urlopen(ARCHIVE_URL, context=SSL_CONTEXT, timeout=300) as resp, open(ARCHIVE, "wb") as out:
            import shutil
            shutil.copyfileobj(resp, out)
    if not verify():
        raise SystemExit("checksum mismatch: the archive differs from the one the protocol was registered with")
    with zipfile.ZipFile(ARCHIVE) as z:
        names = trial_names(z)
    if len(names) != 640:
        raise SystemExit(f"{len(names)} trials in the archive, expected 640")
    print(f"{len(names)} trials in the archive")


if __name__ == "__main__":
    main()
