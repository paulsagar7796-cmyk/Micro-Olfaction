"""
Clean-air / presence-detection test for the SNN front end (simulator in snn_input.py).

Pipeline: gas → porous surface → 12 MOx sensors → linear microLED rate encoder → LIF input
layer (one neuron per channel, lif_neurons.py). Presence detector: an alarm when a spike
population emits at least K spikes (all channels together) within any 1-second window. The
detector taps three populations of the same run, each with its own K:

    encoder      the microLED encoder's spikes, no gate
    LIF 15 mV    the LIF input layer at its default threshold (15 mV above rest)
    LIF 7.5 mV   the LIF input layer at the drift benchmark's threshold

    python3 presence_test.py                              # as simulated before (fixed baseline, LED bias)
    python3 presence_test.py --led-bias 0 --tracking      # LED bias removed, baseline tracking on

1. Calibration: per noise condition and tap, K is the smallest count that stays silent
   through 1 hour of clean air on two calibration seeds (never used for testing):
   max 1-s count + 1.
2. False alarms: 2 hours of clean air on 5 test seeds; alarm episodes per hour, spike rates.
3. Detection: a single 10-s pulse of one gas (5 gases × 0.03–3 ppm) after 5 s of clean air;
   detected if the first alarm comes within 30 s of onset (and none before). Time to alarm
   is measured from pulse onset.
4. Late pulses: after 30 minutes of clean air (drift has built up), 10 pulses of 10 s, one
   every 6 minutes (5 gases × 0.1 and 1 ppm); detected if an alarm starts within 30 s of
   onset (a pulse arriving while the alarm is already on is counted separately, not as
   detected). The 180 s after each onset are left unscored, since gas tails (toluene: up to
   ~110 s) keep the alarm flickering; false alarms are alarms starting in the remaining clean
   time, per hour of that clean time.

Options:
    --led-bias µA   microLED current at zero response (the encoder's i_turn_on). The model
                    emits no light there, so spikes do not depend on it; only the drive power does.
    --tracking      per-channel baseline tracking: the concentration read-out is taken relative
                    to a running baseline resistance instead of the fixed clean-air calibration.
                    Once per second, log R_base ← log R_base + (1 − e^(−1/τ))·(log R − log R_base)
                    (τ = --tracking-tau, 120 s), skipped for a channel that spiked in the last second
                    (so gas is not learned as the new baseline) unless it has been frozen for
                    --max-freeze seconds (300 s), which stops a drift-triggered freeze from sticking.

Noise conditions (sensor read-out noise, baseline drift per √s):
    nominal  0.3 %, 2e-4      the simulator's defaults
    noisy    1 %,   1e-3
    harsh    3 %,   3e-3

Per run the script also records spike counts per tap and the microLED drive current, which
energy_proxy.py turns into an energy estimate.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lif_neurons import LIFInputLayer, LIFParams, one_to_one_weights
from snn_input import (VOC_NAMES, GasDeliverySchedule, LinearMicroLEDParams, LinearMicroLEDRateEncoder,
                       MOxSensorArray, OdorPulse, PorousSurface, SensoryIngestionLayer, default_sensor_array)

DT_SENSOR, DT_ENCODER = 0.01, 0.001
SUBSTEPS = int(round(DT_SENSOR / DT_ENCODER))
BINS_PER_S = int(round(1 / DT_SENSOR))
CONDITIONS = {"nominal": (0.003, 2e-4), "noisy": (0.01, 1e-3), "harsh": (0.03, 3e-3)}
LIF_THRESHOLDS_MV = {"LIF 15 mV": 15.0}
TAPS = ["encoder", *LIF_THRESHOLDS_MV]
CALIBRATION_SEEDS = [1001, 1002]
TEST_SEEDS = [7, 42, 100, 999, 1234]
CONCENTRATIONS = [0.03, 0.1, 0.3, 1.0, 3.0]
PULSE_ON, PULSE_LEN, DETECT_WINDOW, PULSE_RUN = 5.0, 10.0, 30.0, 40.0
LATE_START, LATE_EVERY, LATE_PPM = 1800.0, 360.0, [0.1, 1.0]
LATE_PULSES = [(LATE_START + LATE_EVERY * i, v, p) for i, (p, v) in
               enumerate((p, v) for p in LATE_PPM for v in VOC_NAMES)]
LATE_RUN = LATE_PULSES[-1][0] + LATE_EVERY
LATE_SETTLE = 180.0
SETTINGS = {"led_bias": 0.5, "tracking": False, "tau": 120.0, "max_freeze": 300.0}  # set by main()

def lif_layer(n: int, threshold_mv: float) -> LIFInputLayer:
    return LIFInputLayer(one_to_one_weights(n), LIFParams(v_th_mv=LIFParams().v_rest_mv + threshold_mv))


def build(seed: int, noise: float, drift: float, led_bias: float = 0.5):
    array = MOxSensorArray(default_sensor_array("ZIF-8"), noise_rel=noise, drift_rel_per_sqrt_s=drift, seed=seed)
    lifs = {tap: lif_layer(array.n, th) for tap, th in LIF_THRESHOLDS_MV.items()}
    encoder = LinearMicroLEDRateEncoder(LinearMicroLEDParams(i_turn_on_ua=led_bias), seed=seed + 1)
    return array, PorousSurface.from_sensors(array.sensors), encoder, lifs


def simulate(seed: int, condition: str, pulses: list[OdorPulse], duration_s: float, settings: dict | None = None) -> dict:
    """Same stages as snn_input.ingest, recording only per-10-ms spike counts per tap and LED current.
    With settings["tracking"], the read-out is relative to a per-channel running baseline."""
    cfg = settings or SETTINGS
    array, surface, encoder, lifs = build(seed, *CONDITIONS[condition], cfg["led_bias"])
    schedule = GasDeliverySchedule(pulses)
    n_steps = int(round(duration_s / DT_SENSOR))
    counts = {tap: np.zeros(n_steps, dtype=np.int32) for tap in TAPS}
    led_charge = 0.0                                               # Σ drive current × time, all channels (C)
    surface.reset(); array.reset(); encoder.reset(array.n); schedule.reset()
    for layer in lifs.values():
        layer.reset()
    log_base = np.log(array.r0)                                      # tracked baseline, starts at calibration
    gain = 1.0 - np.exp(-1.0 / cfg["tau"])
    recent = np.zeros(array.n, dtype=int)                            # per-channel spikes in the current second
    frozen_s = np.zeros(array.n)
    for k in range(n_steps):
        effective = surface.step(schedule.step(k * DT_SENSOR, DT_SENSOR), DT_SENSOR)
        res = array.step(effective, DT_SENSOR)
        if cfg["tracking"]:
            sensed = array.concentration_estimate(res * array.r0 / np.exp(log_base))
            if (k + 1) % BINS_PER_S == 0:                            # once per second
                update = (recent == 0) | (frozen_s >= cfg["max_freeze"])
                log_base[update] += gain * (np.log(res[update]) - log_base[update])
                frozen_s = np.where(update, 0.0, frozen_s + 1.0)
                recent[:] = 0
        else:
            sensed = array.concentration_estimate(res)
        step_counts = dict.fromkeys(TAPS, 0)
        for _ in range(SUBSTEPS):
            spikes = encoder.step(sensed, DT_ENCODER)
            recent += spikes
            step_counts["encoder"] += int(spikes.sum())
            for tap, layer in lifs.items():
                step_counts[tap] += int(layer.step(spikes, DT_ENCODER).sum())
        for tap in TAPS:
            counts[tap][k] = step_counts[tap]
        led_charge += float(encoder.last_current_a.sum()) * DT_SENSOR
    return {"counts": counts, "led_charge_c": led_charge}


def window_counts(counts: np.ndarray) -> np.ndarray:
    """Population spikes in the 1-s window ending at each 10-ms bin."""
    c = np.concatenate([[0], np.cumsum(counts)])
    ends = np.arange(1, len(counts) + 1)
    return c[ends] - c[np.maximum(ends - BINS_PER_S, 0)]


def self_check() -> None:
    """The recording loop must reproduce snn_input.ingest + LIFInputLayer.run spike for spike."""
    pulses = [OdorPulse(2.0, 6.0, {"ethanol": 20.0})]
    plain = dict(SETTINGS, tracking=False, led_bias=0.5)
    mine = simulate(5, "noisy", pulses, 10.0, plain)["counts"]
    no_bias = simulate(5, "noisy", pulses, 10.0, dict(plain, led_bias=0.0))["counts"]
    if any(not np.array_equal(mine[t], no_bias[t]) for t in TAPS):
        raise RuntimeError("removing the LED bias changed the spikes")
    array, surface, encoder, lifs = build(5, *CONDITIONS["noisy"])
    res = SensoryIngestionLayer(array, encoder, DT_SENSOR, DT_ENCODER, surface).run(GasDeliverySchedule(pulses), 10.0)
    ref = {"encoder": res.spike_train.raster}
    ref.update({tap: layer.run(res.spike_train).spike_train.raster for tap, layer in lifs.items()})
    for tap in TAPS:
        per_bin = ref[tap].reshape(-1, SUBSTEPS, array.n).sum(axis=(1, 2))
        if not np.array_equal(per_bin, mine[tap]):
            raise RuntimeError(f"recording loop does not reproduce snn_input.ingest ({tap})")
    print("self-check: recording loop matches ingest + LIF layers spike for spike ("
          + ", ".join(f"{tap} {int(mine[tap].sum())}" for tap in TAPS) + " spikes); "
          "spikes are identical with the LED bias removed", flush=True)


def clean_air(job: tuple[str, int, float, dict]) -> dict:
    condition, seed, hours, cfg = job
    r = simulate(seed, condition, [], hours * 3600, cfg)
    return {"condition": condition, "seed": seed, "duration_s": hours * 3600, "led_charge_c": r["led_charge_c"],
            "spikes": {t: int(c.sum()) for t, c in r["counts"].items()},
            "windows": {t: window_counts(c) for t, c in r["counts"].items()}}


def late(job: tuple[str, int, dict]) -> dict:
    condition, seed, cfg = job
    r = simulate(seed, condition, [OdorPulse(t, t + PULSE_LEN, {v: p}) for t, v, p in LATE_PULSES], LATE_RUN, cfg)
    return {"condition": condition, "seed": seed, "windows": {t: window_counts(c) for t, c in r["counts"].items()}}


def pulse(job: tuple[str, int, str, float, dict]) -> dict:
    condition, seed, voc, ppm, cfg = job
    r = simulate(seed, condition, [OdorPulse(PULSE_ON, PULSE_ON + PULSE_LEN, {voc: ppm})], PULSE_RUN, cfg)
    return {"condition": condition, "seed": seed, "voc": voc, "ppm": ppm, "duration_s": PULSE_RUN,
            "led_charge_c": r["led_charge_c"], "spikes": {t: int(c.sum()) for t, c in r["counts"].items()},
            "windows": {t: window_counts(c) for t, c in r["counts"].items()}}


def score_late(lates: list[dict], k: dict) -> dict:
    """Per (condition, tap): pulses detected, pulses met by an alarm already on, false alarms in clean time."""
    stats: dict = {}
    for r in lates:
        for t in TAPS:
            above = r["windows"][t] >= k[(r["condition"], t)]
            starts = np.flatnonzero(np.concatenate([[above[0]], above[1:] & ~above[:-1]]))
            settling = np.zeros(len(above), dtype=bool)
            st = stats.setdefault((r["condition"], t), {"detected": 0, "already_on": 0, "pulses": 0, "false": 0,
                                                        "clean_s": 0.0, "lat": []})
            for onset_s, _, _ in LATE_PULSES:
                a = int(onset_s * BINS_PER_S)
                settling[a:a + int(LATE_SETTLE * BINS_PER_S)] = True
                st["pulses"] += 1
                if above[a - 1]:
                    st["already_on"] += 1
                    continue
                hits = starts[(starts >= a) & (starts < a + DETECT_WINDOW * BINS_PER_S)]
                if len(hits):
                    st["detected"] += 1
                    st["lat"].append((hits[0] + 1 - a) / BINS_PER_S)
            st["false"] += int(np.sum(~settling[starts]))
            st["clean_s"] += float(np.sum(~settling)) / BINS_PER_S
    return stats


def print_late(stats: dict) -> None:
    print(f"\nLate pulses (10 per run after {LATE_START / 60:g} min of drift, {len(TEST_SEEDS)} seeds; "
          f"{LATE_SETTLE:g} s after each onset unscored)")
    print(f"{'condition':<10}{'tap':<12}{'detected':>10}{'alarm already on':>18}{'median time to alarm':>22}"
          f"{'false alarms / clean h':>24}")
    for (c, t), st in stats.items():
        lat = f"{np.median(st['lat']):.1f} s" if st["lat"] else "—"
        print(f"{c:<10}{t:<12}{st['detected']:>6}/{st['pulses']:<3}{st['already_on']:>18}{lat:>22}"
              f"{st['false'] / (st['clean_s'] / 3600):>24.2f}")


def episodes(above: np.ndarray) -> int:
    return int(np.sum(above[1:] & ~above[:-1]) + above[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--test-hours", type=float, default=2.0)
    parser.add_argument("--calibration-hours", type=float, default=1.0)
    parser.add_argument("--led-bias", type=float, default=0.5, help="microLED current at zero response (µA)")
    parser.add_argument("--tracking", action="store_true", help="per-channel baseline tracking")
    parser.add_argument("--tracking-tau", type=float, default=120.0, help="baseline time constant (s)")
    parser.add_argument("--max-freeze", type=float, default=300.0, help="longest baseline freeze (s)")
    parser.add_argument("--late-only", type=Path, default=None, metavar="CSV",
                        help="rerun only the late-pulse test, with the thresholds K of this earlier results file")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/presence_test.csv"))
    args = parser.parse_args()
    self_check()
    cfg = dict(led_bias=args.led_bias, tracking=args.tracking, tau=args.tracking_tau, max_freeze=args.max_freeze)
    print(f"configuration: LED bias {args.led_bias:g} µA, baseline tracking "
          f"{f'on (τ = {args.tracking_tau:g} s, max freeze {args.max_freeze:g} s)' if args.tracking else 'off'}",
          flush=True)

    if args.late_only:
        with open(args.late_only, newline="") as f:
            k = {(r["condition"], r["tap"]): int(r["threshold_k"]) for r in csv.DictReader(f)
                 if r["condition"] in args.conditions and r["tap"] in TAPS}
        print(f"thresholds K from {args.late_only}: {k}", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            lates = list(pool.map(late, [(c, s, cfg) for c in args.conditions for s in TEST_SEEDS]))
        print_late(score_late(lates, k))
        return

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        calib = list(pool.map(clean_air, [(c, s, args.calibration_hours, cfg) for c in args.conditions
                                          for s in CALIBRATION_SEEDS]))
        k = {(c, t): max(1, max(int(r["windows"][t].max()) for r in calib if r["condition"] == c) + 1)
             for c in args.conditions for t in TAPS}
        print(f"calibrated thresholds K (spikes per 1-s window): "
              + "; ".join(f"{c}: " + ", ".join(f"{t} {k[(c, t)]}" for t in TAPS) for c in args.conditions)
              + f"  [{time.time() - t0:.0f} s]", flush=True)
        clean_f = [pool.submit(clean_air, (c, s, args.test_hours, cfg)) for c in args.conditions for s in TEST_SEEDS]
        late_f = [pool.submit(late, (c, s, cfg)) for c in args.conditions for s in TEST_SEEDS]
        pulses = list(pool.map(pulse, [(c, s, v, p, cfg) for c in args.conditions for s in TEST_SEEDS
                                       for v in VOC_NAMES for p in CONCENTRATIONS], chunksize=4))
        clean = [f.result() for f in clean_f]
        lates = [f.result() for f in late_f]
        print(f"clean-air test runs done [{time.time() - t0:.0f} s]", flush=True)
        print(f"pulse runs done [{time.time() - t0:.0f} s]", flush=True)

    rows = []
    onset = int(round(PULSE_ON * BINS_PER_S))
    for r in calib + clean + pulses:
        test = "pulse" if "voc" in r else ("calibration" if r["seed"] in CALIBRATION_SEEDS else "clean-air test")
        for t in TAPS:
            above = r["windows"][t] >= k[(r["condition"], t)]
            row = {"test": test, "led_bias_ua": args.led_bias, "tracking": int(args.tracking),
                   "condition": r["condition"], "tap": t, "seed": r["seed"], "voc": r.get("voc", ""),
                   "ppm": r.get("ppm", 0.0), "duration_s": r["duration_s"], "threshold_k": k[(r["condition"], t)],
                   "spikes": r["spikes"][t], "encoder_spikes": r["spikes"]["encoder"],
                   "max_window": int(r["windows"][t].max()), "led_charge_c": r["led_charge_c"]}
            if test == "pulse":
                first = int(np.argmax(above)) if above.any() else None
                early = first is not None and first < onset
                detected = first is not None and not early and first < onset + DETECT_WINDOW * BINS_PER_S
                row.update(alarms=int(early), detected=int(detected),
                           latency_s=(first + 1 - onset) / BINS_PER_S if detected else "")
            else:
                row.update(alarms=episodes(above), detected="", latency_s="")
            rows.append(row)
    late_stats = score_late(lates, k)
    for (c, t), st in late_stats.items():
        rows.append({"test": "late pulses", "led_bias_ua": args.led_bias, "tracking": int(args.tracking),
                     "condition": c, "tap": t, "seed": "all", "voc": "", "ppm": "", "duration_s": st["clean_s"],
                     "threshold_k": k[(c, t)], "spikes": "", "encoder_spikes": "", "max_window": "", "led_charge_c": "",
                     "alarms": st["false"], "detected": st["detected"],
                     "latency_s": float(np.median(st["lat"])) if st["lat"] else ""})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nClean air ({args.test_hours:g} h × {len(TEST_SEEDS)} test seeds per condition)")
    print(f"{'condition':<10}{'tap':<12}{'K':>5}{'false alarms / h':>18}{'spikes / s':>13}{'busiest 1-s window':>20}")
    for c in args.conditions:
        for t in TAPS:
            sel = [r for r in rows if r["test"] == "clean-air test" and r["condition"] == c and r["tap"] == t]
            secs = sum(r["duration_s"] for r in sel)
            print(f"{c:<10}{t:<12}{k[(c, t)]:>5}{sum(r['alarms'] for r in sel) / (secs / 3600):>18.2f}"
                  f"{sum(r['spikes'] for r in sel) / secs:>13.3f}{max(r['max_window'] for r in sel):>20}")

    print(f"\nDetection of a single 10-s pulse: detected / {len(TEST_SEEDS)} seeds, median time to alarm (s)")
    for c in args.conditions:
        for t in TAPS:
            print(f"\n  {c}, {t} (K = {k[(c, t)]})")
            print(f"  {'gas':<14}" + "".join(f"{f'{p:g} ppm':>15}" for p in CONCENTRATIONS))
            for v in VOC_NAMES:
                cells = ""
                for p in CONCENTRATIONS:
                    sel = [r for r in rows if r["test"] == "pulse" and r["condition"] == c and r["tap"] == t
                           and r["voc"] == v and r["ppm"] == p]
                    lat = [r["latency_s"] for r in sel if r["detected"]]
                    cell = f"{sum(r['detected'] for r in sel)}/{len(sel)}" + (f" {np.median(lat):.1f}s" if lat else "")
                    cells += f"{cell + ('!' if any(r['alarms'] for r in sel) else ''):>15}"
                print(f"  {v:<14}{cells}")
    print("\n'!' marks cells with an alarm before the pulse started (a false alarm).")
    print_late(late_stats)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
