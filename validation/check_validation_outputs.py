"""
Compares the regenerated twin-array validation files with their committed versions, the way
check_reproduction.py does for paper-v3's results: CSVs cell by cell (numbers to a relative 1e-9), logs
line by line. Unlike check_reproduction.py, a file with no committed version to compare with is a
failure, not a pass.

    python3 validation/check_validation_outputs.py                  # against git HEAD
    python3 validation/check_validation_outputs.py --against pre-v4 # against another commit or tag
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from check_reproduction import compare_csv  # noqa: E402

OUTPUTS = ["results/twin_arrays_features.csv", "results/system2_compounds.csv", "results/system2_compounds_folds.csv",
           "results/system2_compounds_budget.csv", "results/system2_compounds_seeds.csv",
           "results/system2_compounds_exploratory.csv", "results/logs/twin_arrays.log",
           "results/logs/system2_compounds.log"]
# at tag pre-v4 the same files sit at the same paths
PRE_V4_PATH = {p: p for p in OUTPUTS}


def committed(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--against", default="HEAD", help="commit or tag to compare with (default HEAD)")
    args = parser.parse_args()
    ok = True
    for path in OUTPUTS:
        old = committed(args.against, PRE_V4_PATH[path] if args.against != "HEAD" else path)
        if old is None:
            ok = False
            print(f"new         {path} (no committed version in {args.against}: nothing to compare with)")
            continue
        text = Path(path).read_text()
        if text == old:
            print(f"identical   {path}")
            continue
        problem = compare_csv(path, old) if path.endswith(".csv") else "\n".join(
            list(difflib.unified_diff(old.splitlines(), text.splitlines(), "committed", "now", n=0, lineterm=""))[2:14])
        if problem is None:
            print(f"equal       {path} (numbers within 1e-9, formatting differs)")
        else:
            ok = False
            print(f"DIFFERS     {path}: {problem}")
    print("all regenerated validation results match the committed ones" if ok
          else "some validation results differ from the committed ones, or could not be compared")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
