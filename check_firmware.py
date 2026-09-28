"""
Step 3 of the C port: the C library (firmware/s2p.c) against its specification, device_reference.py
in single precision, bit for bit.

    python3 check_firmware.py                 # the host build (firmware/build/host_harness), ~3 minutes
    python3 check_firmware.py --target cm4    # the Cortex-M4 build under QEMU (-M netduinoplus2)

Every System 2+ run behind the paper (the Settings 1–2 runs of system2_combined.csv and the sequential
runs, with and without pseudo-labels) is exported as a scenario, run through the compiled C code on
this computer (or on an emulated Cortex-M4F) and through the reference, in two modes: calibration passes in the chip's own xorshift
order (as deployed), and in the paper's numpy order supplied by the host. Compared: every prediction,
the pseudo-label counts, and the final state (weights, intercepts, step counter, alignment statistics)
bit for bit. Expected: all identical.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
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

FIRMWARE = Path("firmware")
HARNESS = FIRMWARE / "build" / "host_harness"
MAGIC = 0x31503253
MODES = {"own shuffle": R.CHIP32_DEVICE, "paper order": R.CHIP32}
SDKS = Path("/Library/Developer/CommandLineTools/SDKs")
CM4_HARNESS = FIRMWARE / "build" / "cm4_harness.elf"


def newest(pattern: str) -> Path:
    found = sorted(Path.home().glob(pattern))
    if not found:
        raise FileNotFoundError(f"~/{pattern} not found (the xPack Arm GCC and QEMU go in ~/opt)")
    return found[-1]


def arm_tools() -> dict:
    """The xPack Arm GCC and QEMU in ~/opt."""
    qemu = newest("opt/xpack-qemu-arm-*")
    return {"arm": newest("opt/xpack-arm-none-eabi-gcc-*/bin"), "qemu": qemu / "bin" / "qemu-system-arm",
            "qemu_include": qemu / "include"}


def qemu_command(qemu: Path, elf: Path, extra: list[str] | None = None) -> list[str]:
    return [str(qemu), "-M", "netduinoplus2", "-nographic", "-monitor", "none", "-serial", "none",
            "-semihosting-config", "enable=on,target=native", *(extra or []), "-kernel", str(elf.resolve())]


def build_cm4(t: dict) -> None:
    r = subprocess.run(["make", "-C", str(FIRMWARE), str(CM4_HARNESS.relative_to(FIRMWARE)), f"ARM_TOOLS={t['arm']}"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    print(f"built {CM4_HARNESS}")


def build() -> None:
    """make; if the default macOS SDK does not link (a linker older than the SDK), try older SDKs."""
    candidates = [None] + (sorted((p for p in SDKS.glob("MacOSX*.*.sdk")), reverse=True) if SDKS.exists() else [])
    for sdk in candidates:
        env = dict(os.environ, **({"SDKROOT": str(sdk)} if sdk else {}))
        subprocess.run(["make", "-C", str(FIRMWARE), "clean"], capture_output=True)
        r = subprocess.run(["make", "-C", str(FIRMWARE)], capture_output=True, text=True, env=env)
        if r.returncode == 0:
            print(f"built {HARNESS}" + (f" (SDK {sdk.name})" if sdk else ""))
            return
    raise RuntimeError(f"cannot build the harness:\n{r.stdout}{r.stderr}")


def write_scenario(sc: R.Scenario, ar: R.Arith, path: Path) -> None:
    f32 = np.float32
    dev = R.Device(sc.model, sc.mu, sc.var, ar, "centred")
    r = dev.readout
    n = len(r.classes)
    index = {c: i for i, c in enumerate(r.classes)}

    def pad(a: np.ndarray, shape: tuple, dtype=f32) -> np.ndarray:
        out = np.zeros(shape, dtype)
        out[:len(a)] = a
        return out

    passes = R.Passes(ar, sc.passes_key)
    parts = [np.array([MAGIC, n, len(sc.visits), int(sc.pl), C.PSEUDO["cap"], int(ar.orders == "numpy"), 0, 0], np.int32),
             np.array([r.alpha, r.opt_init, r.t, C.PSEUDO["tau"], C.PSEUDO["frac"], sc.k, dev.a, dev.oma, R.EPS], f32),
             pad(np.array(r.seeds, np.uint32), (6,), np.uint32), np.array([passes.state], np.uint32),
             pad(r.pos_weight, (6,)), pad(r.w, (6, R.N_RAW * 2)), pad(r.b, (6,)), dev.mu.astype(f32), dev.var.astype(f32),
             np.array([R.T._floors()[g] for g in R.GROUPS], f32)]
    for b, labelled, later in sc.visits:
        raw, gas = R.sensor_rows(b).astype(f32), load_batch(b).gas
        parts += [np.array([len(labelled), len(later)], np.int32), np.array([index[g] for g in gas[labelled]], np.int32)]
        if ar.orders == "numpy":
            perms = passes.epochs(len(labelled), R.TUNED["epochs"])
            parts.append(np.array(perms, np.uint8).reshape(-1))
        parts += [raw[labelled], raw[later]]
    with open(path, "wb") as fh:
        for p in parts:
            fh.write(np.ascontiguousarray(p).tobytes())


def read_output(path: Path, visits: list, n_classes: int) -> tuple[list[np.ndarray], list[int], dict]:
    data, pos = path.read_bytes(), 0

    def take(dtype, count):
        nonlocal pos
        a = np.frombuffer(data, dtype, count, pos)
        pos += a.nbytes
        return a
    preds, updates = [], []
    for _, _, later in visits:
        preds.append(take(np.int32, len(later)))
        updates.append(int(take(np.int32, 1)[0]))
    state = {"w": take(np.float32, 6 * 256).reshape(6, 256)[:n_classes], "b": take(np.float32, 6)[:n_classes],
             "t": take(np.float32, 1), "mu": take(np.float32, 256), "var": take(np.float32, 256)}
    return preds, updates, state


def job(args: tuple) -> dict:
    mode, spec, tmp, qemu = args
    ar = MODES[mode]
    sc = R.sequential_scenario(*spec[1:]) if spec[0] == "sequential" else R.settings_scenario(*spec[1:])
    rows, dev = R.run(sc, ar, keep=True)
    index = {c: i for i, c in enumerate(dev.readout.classes)}
    work = Path(tempfile.mkdtemp(dir=tmp))
    scen, out = work / "scenario.bin", work / "output.bin"
    write_scenario(sc, ar, scen)
    if qemu:                                           # the Cortex-M4 build reads scenario.bin, writes output.bin
        r = subprocess.run(qemu_command(Path(qemu), CM4_HARNESS), cwd=work, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{spec}: {r.stdout}{r.stderr}")
    else:
        subprocess.run([str(HARNESS), str(scen), str(out)], check=True)
    preds, updates, state = read_output(out, sc.visits, len(dev.readout.classes))
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    ref_preds = [np.array([index[g] for g in r["pred"]], np.int32) for r in rows]
    same = sum(int(np.sum(a == b)) for a, b in zip(preds, ref_preds))
    total = sum(len(a) for a in ref_preds)
    ref_state = {"w": dev.readout.w, "b": dev.readout.b, "t": np.array([dev.readout.t], np.float32),
                 "mu": dev.mu, "var": dev.var}
    bitwise = all(np.array_equal(state[k].view(np.uint32), np.asarray(ref_state[k], np.float32).view(np.uint32))
                  for k in state)
    diff = max(float(np.max(np.abs(state[k].astype(float) - np.asarray(ref_state[k], float)))) for k in state)
    return {"mode": mode, "protocol": spec[0], "spec": str(spec[1:]), "predictions": total, "identical": same,
            "updates_equal": updates == [r["pseudo_updates"] for r in rows], "state_bitwise": bitwise,
            "state_max_diff": diff}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--target", choices=["host", "cm4"], default="host")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.out = args.out or Path(f"results/firmware_check{'_cm4' if args.target == 'cm4' else ''}.csv")
    qemu = None
    if args.target == "cm4":
        t = arm_tools()
        build_cm4(t)
        qemu = str(t["qemu"])
    else:
        build()
    specs = [("sequential", b, s, 3.0, True, pl) for pl in (True, False) for b in S.BUDGETS for s in K.SEQ_SEEDS]
    specs += [("settings", st, tg, s, b, sel, 3.0, True, True) for st in I.SETTINGS for tg in I.TARGETS
              for s in DEFAULT_SEEDS for b, sel in I.COMBOS]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(max_workers=args.jobs) as pool:
        df = pd.DataFrame(pool.map(job, [(m, sp, tmp, qemu) for m in MODES for sp in specs], chunksize=4))
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} scenarios [{time.time() - t0:.0f} s]; wrote {args.out}")
    for (mode, protocol), d in df.groupby(["mode", "protocol"], sort=False):
        print(f"  {mode:<12} {protocol:<11} {len(d):>4} scenarios: predictions identical "
              f"{d.identical.sum():,}/{d.predictions.sum():,}; pseudo-label counts equal {int(d.updates_equal.sum())}/{len(d)}; "
              f"final state bit-identical {int(d.state_bitwise.sum())}/{len(d)} (max |Δ| {d.state_max_diff.max():.1e})")
    ok = bool((df.identical == df.predictions).all() and df.updates_equal.all() and df.state_bitwise.all())
    where = "on the emulated Cortex-M4F" if qemu else "on the host"
    print(f"\nC library {where} {'matches' if ok else 'DOES NOT match'} the single-precision reference bit for bit")


if __name__ == "__main__":
    main()
