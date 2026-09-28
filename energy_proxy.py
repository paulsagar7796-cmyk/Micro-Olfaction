"""
Energy proxy: operation counts × published energy per operation, for the SNN and the
conventional alternatives. A first-order estimate, not a hardware measurement.

    python3 energy_proxy.py                       # needs results/presence_test.csv
    python3 energy_proxy.py --presence-csv results/presence_test_nobias.csv results/presence_test_tracking.csv

Part A, always-on presence detection (from presence_test.py runs):
    SNN            each encoder spike is one synaptic event into the LIF input layer, each LIF
                   spike one event into the detector; plus the microLED encoder's drive power
                   (the simulator's own drive currents × LED forward voltage); with baseline
                   tracking, one ADC reading and 5 arithmetic operations per channel per second.
    conventional   every sensor channel sampled by an ADC, then baseline-subtract, compare and
                   a slow baseline update (4 arithmetic operations) per sample.

Part B, one classification (a 1-s UCI sample, benchmark_drift.py's network, seed 42):
    SNN            synaptic events counted on real drift samples: encoder → LIF input layer
                   (1 each), LIF → STDP layer (64 each), STDP → readout (6 each) plus lateral
                   inhibition of the other 63 STDP neurons, readout spikes (1 each).
    SGD            System 2 on 128 features: EMA update and standardisation (7 ops per
                   feature), 128 × 6 multiply-accumulates, each reading one weight from SRAM.
    random forest  100 trees × up to 12 comparisons, each reading a threshold and a feature index.

Not modelled, for either side: static/leakage power, clocking, and the MOx heaters, which
are typically far larger than everything here (printed as a reference, not simulated).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

CONSTANTS = {  # name: (value, unit, source)
    "synop": (23.6e-12, "J per synaptic event", "Intel Loihi, Davies et al., IEEE Micro 2018"),
    "mac": (4.6e-12, "J per FP32 multiply-accumulate", "Horowitz, ISSCC 2014 (45 nm): 3.7 pJ mult + 0.9 pJ add"),
    "op": (0.9e-12, "J per FP32 add / compare", "Horowitz, ISSCC 2014 (45 nm)"),
    "sram": (5e-12, "J per 32-bit SRAM read", "Horowitz, ISSCC 2014: ~10 pJ per 64-bit read, 8 KB SRAM"),
    "adc": (0.4e-9, "J per 12-bit ADC conversion", "~100 fJ per conversion step × 4096 (low-power SAR)"),
    "led_vf": (3.0, "V forward voltage, 450 nm InGaN microLED", "typical for blue InGaN LEDs"),
    "heater": (10e-3, "W per MOx sensor heater", "order of magnitude for MEMS micro-hotplates (datasheet-dependent)"),
}
N_SENSORS_SIM = 12


def si(watts: float, unit: str = "W") -> str:
    for scale, prefix in ((1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")):
        if abs(watts) >= scale or scale == 1e-12:
            return f"{watts / scale:.3g} {prefix}{unit}"
    return f"0 {unit}"


def presence_part(paths: list[Path], c: dict) -> None:
    print("\nA. Always-on presence detection (simulator, 12 sensors), average power")
    print(f"{'configuration':<25}{'condition':<9} {'detector':<16}{'clean air':>12}{'1 ppm pulse':>13}"
          f"{'  microLED':>11}{'  spikes':>10}{'  tracking':>11}")
    for path in paths:
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["test"] in ("clean-air test", "pulse")]
        tracking = rows[0].get("tracking", "0") == "1"
        bias = float(rows[0].get("led_bias_ua", 0.5))
        label = f"LED bias {bias:g} µA{', tracking' if tracking else ''}"
        track_power = N_SENSORS_SIM * 1.0 * (c["adc"] + 5 * c["op"]) if tracking else 0.0
        for cond in dict.fromkeys(r["condition"] for r in rows):
            for tap in dict.fromkeys(r["tap"] for r in rows):
                def power(sel: list[dict]) -> tuple[float, float, float]:
                    secs = sum(float(r["duration_s"]) for r in sel)
                    events = sum(int(r["encoder_spikes"]) + (int(r["spikes"]) if tap != "encoder" else 0) for r in sel)
                    led = sum(float(r["led_charge_c"]) for r in sel) / secs * c["led_vf"]
                    spikes = events / secs * c["synop"]
                    return spikes + led + track_power, led, spikes
                clean = [r for r in rows if r["test"] == "clean-air test" and r["condition"] == cond and r["tap"] == tap]
                gas = [r for r in rows if r["test"] == "pulse" and r["condition"] == cond and r["tap"] == tap
                       and float(r["ppm"]) == 1.0]
                (p_clean, led, spikes), (p_gas, _, _) = power(clean), power(gas)
                print(f"{label:<25}{cond:<9} {tap:<16}{si(p_clean):>12}{si(p_gas):>13}{si(led):>11}{si(spikes):>10}"
                      f"{si(track_power):>11}")
    for fs in (10, 100):
        p = N_SENSORS_SIM * fs * (c["adc"] + 4 * c["op"] + c["sram"])
        print(f"{'conventional':<25}{'any':<9} {f'{fs} Hz ADC':<16}{si(p):>12}{si(p):>13}")
    print(f"Reference, not simulated: MOx heaters ≈ {si(N_SENSORS_SIM * c['heater'])} "
          f"({N_SENSORS_SIM} × {si(c['heater'])}), for either detector.")


def classification_part(c: dict, seed: int, per_batch: int) -> None:
    import benchmark_drift as D
    from benchmark_batch1 import make_input_layer
    from dataset_adapter import encode_batch, load_batch

    parser = D.build_parser()
    args = D.resolve_args(parser, parser.parse_args(["--align-ema", "--seeds", str(seed)]))
    net, _ = D.train_on_batch1(args, seed)
    ema = D.ema_affines(args, net.ema_init, seed, args.batches)
    layer, readout = net.readouts.layer, net.readouts.spiking
    n_stdp, n_out = layer.n_neurons, len(readout.classes)
    counts = {"encoder": [], "lif": [], "stdp": [], "readout": []}
    for b in args.batches:
        batch = load_batch(b)
        idx = np.sort(D.stream_order(seed, b, len(batch))[:per_batch])
        samples, _ = encode_batch(batch.subset(idx), duration_s=args.duration, f_max=args.f_max, mode=args.mode,
                                  seed=seed, feature_mode=args.feature_mode, normalization=args.normalization,
                                  channel_affine=(ema[b][0][idx], ema[b][1][idx]))
        lif = make_input_layer(args, samples[0].spike_train.raster.shape[1])
        filtered = [lif.run(s.spike_train).spike_train for s in samples]
        outputs = [layer.process(f, learn=False) for f in filtered]
        counts["encoder"] += [int(s.spike_train.raster.sum()) for s in samples]
        counts["lif"] += [int(f.raster.sum()) for f in filtered]
        counts["stdp"] += [int(o.spike_counts.sum()) for o in outputs]
        counts["readout"] += list(readout.output_counts([o.spike_train for o in outputs]).sum(axis=1))
    m = {k: float(np.mean(v)) for k, v in counts.items()}
    snn_events = m["encoder"] + m["lif"] * n_stdp + m["stdp"] * (n_out + n_stdp - 1) + m["readout"]
    n_feat, n_cls = 128, 6
    sgd = n_feat * 7 * c["op"] + n_feat * n_cls * (c["mac"] + c["sram"]) + n_cls * c["op"]
    forest = 100 * 12 * (c["op"] + 2 * c["sram"])
    print(f"\nB. One classification (1-s UCI sample, seed {seed}, {len(counts['encoder'])} drift samples)")
    print(f"   SNN spikes per sample: encoder {m['encoder']:.0f}, LIF input layer {m['lif']:.0f}, "
          f"STDP layer {m['stdp']:.0f}, readout {m['readout']:.0f} → {snn_events:,.0f} synaptic events")
    print(f"{'classifier':<34}{'energy per sample':>20}{'relative to SGD':>18}")
    for name, e in (("SNN (STDP + spiking readout)", snn_events * c["synop"]), ("SGD, 128 features", sgd),
                    ("Random forest, 100 trees", forest)):
        print(f"{name:<34}{si(e, 'J'):>20}{e / sgd:>17.1f}×")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--presence-csv", type=Path, nargs="+",
                        default=[Path(f"results/presence_test{x}.csv") for x in ("", "_nobias", "_tracking")])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-batch", type=int, default=100)
    parser.add_argument("--skip-classification", action="store_true", help="only the presence-detection part")
    for name, (value, unit, _) in CONSTANTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value, help=unit)
    args = parser.parse_args()
    c = {name: getattr(args, name) for name in CONSTANTS}
    print("Energy per operation (first-order literature values; override with the options above):")
    for name, (_, unit, source) in CONSTANTS.items():
        print(f"  {name:<7} {c[name]:.3g} {unit} — {source}")
    presence_part(args.presence_csv, c)
    if not args.skip_classification:
        classification_part(c, args.seed, args.samples_per_batch)


if __name__ == "__main__":
    main()
