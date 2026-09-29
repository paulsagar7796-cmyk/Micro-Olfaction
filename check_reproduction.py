"""
Reproduction checks used by reproduce.sh.

    python3 check_reproduction.py env        # Python and package versions match requirements.txt
    python3 check_reproduction.py results    # regenerated results match the versions committed in git

`results` compares each file reproduce.sh writes with its version in git HEAD: CSVs cell by cell
(numbers to a relative 1e-9), logs line by line. On the pinned environment every file is expected to be
identical: SGD makes single results sensitive to library versions and floating-point details, so any
difference is reported, with where it is. Exits with status 1 on a mismatch.
"""

from __future__ import annotations

import difflib
import io
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

OUTPUTS = [f"results/{name}" for name in (
    "system2_tuning.csv", "system2_budget_sweep.csv", "sequential_system2plus.csv", "literature_comparison.csv",
    "system2_improvements.csv",
    "system2_combined.csv", "temporal_confound.csv", "temporal_confound_file_order.csv",
    "temporal_confound_baseline.csv", "literature_table.csv", "pseudo_label_scores.csv", "presence_test.csv",
    "presence_test_nobias.csv", "presence_test_tracking.csv",
    # the C port (regenerated only when the Arm tools are present)
    "device_reference.csv", "firmware_check.csv", "firmware_check_opt.csv", "firmware_check_cm4.csv",
    "firmware_check_cm4_opt.csv", "cm4_cycles.csv", "cm4_energy.csv", "cm4_memory.csv", "cm4_inference_classes.csv")]


def check_env() -> bool:
    ok = True
    lines = Path("requirements.txt").read_text().splitlines()
    want_py = next(re.search(r"Python (\S+)", s).group(1) for s in lines if s.startswith("# Python"))
    if platform.python_version() != want_py:
        print(f"Python {platform.python_version()}, results were produced with {want_py}")
        ok = False
    for name, want in (s.split("==") for s in lines if "==" in s and not s.startswith("#")):
        try:
            have = version(name)
        except PackageNotFoundError:
            have = "not installed"
        if have != want:
            print(f"{name} {have}, results were produced with {want}")
            ok = False
    print(f"environment {'matches' if ok else 'differs from'} requirements.txt "
          f"({platform.system()} {platform.machine()}, Python {platform.python_version()})")
    return ok


def committed(path: str) -> str | None:
    r = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def compare_csv(path: str, old: str) -> str | None:
    import numpy as np
    import pandas as pd
    a, b = pd.read_csv(path), pd.read_csv(io.StringIO(old))
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f"shape/columns {a.shape} vs committed {b.shape}"
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            bad = ~np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True)
            if bad.any():
                i = int(np.argmax(bad))
                return f"column {col}: {int(bad.sum())} values differ (first at row {i}: {x[i]!r} vs committed {y[i]!r})"
        elif not x.fillna("").astype(str).equals(y.fillna("").astype(str)):
            i = int((x.fillna("").astype(str) != y.fillna("").astype(str)).to_numpy().argmax())
            return f"column {col}: row {i} {x.iloc[i]!r} vs committed {y.iloc[i]!r}"
    return None


def check_results() -> bool:
    ok, new = True, []
    for path in OUTPUTS + sorted(str(p) for p in Path("results/logs").glob("*.log")):
        old = committed(path)
        if old is None:
            new.append(path)
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
    for path in new:
        print(f"new         {path} (not committed yet)")
    print("all regenerated results match the committed ones" if ok else "some results differ from the committed ones")
    return ok


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what not in ("env", "results"):
        sys.exit(__doc__)
    sys.exit(0 if (check_env() if what == "env" else check_results()) else 1)


if __name__ == "__main__":
    main()
