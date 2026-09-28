"""
First-order inference and adaptation cost of System 2+, DA-ELM and the post-2015 methods in
literature/published_results.csv, on the UCI drift dataset (128 features, 6 gases). Every cost in
section 5 of paper/comparison_tables.md comes from here.

    python3 deep_model_cost.py              # needs results/system2_combined.csv (pseudo-label update rate)

Counts multiply-accumulates (MACs) in the layers each paper specifies; activations, normalisation,
pooling, softmax and exponentials are excluded, as in combine_system2.py. Energy = MACs × 3.1 pJ;
memory = parameters × 4 B (FP32) or × 1 B (int8); 1 KB = 1,000 bytes, 1 MB = 10⁶ bytes. A second
accounting for System 2 and DA-ELM inference adds SRAM reads and transcendental functions, with the
per-operation energies of energy_proxy.py. Architecture sources:
  TDACNN (Sens. Actuators B 2022, arXiv:2110.07509): Fig. 6, right column (UCI input 8×1 per sensor);
    §2.1.3 (four classifiers, each fc 100 → fc 100 → softmax); §2.2 (test-time MMD ensemble with an RBF
    kernel against the embeddings of the source samples of the predicted class). Padding is not stated;
    'same' is assumed (three 3×3 'valid' convolutions cannot fit on the 8-high map of block D).
  AMDS-PFFA (arXiv:2409.13167): Fig. 2 and §III-B, UCI numbers. The transformer's token layout, width,
    heads, feed-forward size and depth N are not given; a range is computed for one encoder layer (N = 1)
    under two layouts whose flattened output is 128 wide, as the following 128-neuron layer implies.
    iAFF uses the reduction ratio r = 4 of its reference implementation (Dai et al. 2021).
  ODAELM (Sensors 2018): up to L = 1000 RBF hidden nodes (as DA-ELM); the online update keeps the
    L×L inverse K⁻¹ and applies a Woodbury update per sample (Eqs. 20, 36).
"""

from __future__ import annotations

import improve_system2 as I
from benchmark_literature_comparison import BATCH_SIZES as ALL_BATCH_SIZES

PJ_PER_MAC = I.PJ_PER_MAC
N_FEATURES, N_SENSORS, PER_SENSOR, N_GASES = 128, 16, 8, 6
BATCH_SIZES = {b: ALL_BATCH_SIZES[b] for b in (1, 7)}
ELM_HIDDEN = 1000
FULL = "K=3 + class weight + pseudo-labels"
CALIBRATION_PASSES = 50
LABELS = (20, 50)                                               # labelled target samples per batch


def dense(n_in: int, n_out: int) -> tuple[int, int]:
    return n_in * n_out, n_in * n_out + n_out                   # (MACs, parameters)


def conv(positions: int, kernel: int, c_in: int, c_out: int) -> tuple[int, int]:
    return positions * kernel * c_in * c_out, kernel * c_in * c_out + c_out


def total(parts: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return sum(m for m, _ in parts.values()), sum(p for _, p in parts.values())


def tdacnn() -> dict:
    L = PER_SENSOR
    per_branch = {"A": conv(L, 2, 1, 2), "B": conv(L, 2, 2, 4), "C": conv(L, 2, 4, 6)}
    parts = {f"block {k} ×16": (m * N_SENSORS, p * N_SENSORS) for k, (m, p) in per_branch.items()}
    hw = L * N_SENSORS                                            # width-concatenated 8×16 map
    d = [conv(hw, 9, 4, 6), conv(hw, 9, 6, 6), conv(hw, 9, 6, 6)]
    parts["block D (3× 3×3×6)"] = total(dict(enumerate(d)))
    flat = {"1 (after A)": hw * 2, "2 (after B)": hw * 4, "3 (after D)": hw * 6, "4 (after C)": hw * 6}
    for name, n in flat.items():
        parts[f"classifier {name}"] = total({"fc1": dense(n, 100), "fc2": dense(100, 100), "out": dense(100, N_GASES)})
    macs, params = total(parts)
    # MMD ensemble: RBF kernel between the test embedding (100-d, per classifier) and every source sample of
    # the predicted class; on average n_source / 6 samples per class. All source embeddings must be stored.
    mmd = {b: (4 * n // N_GASES * 100, 4 * n * 100) for b, n in BATCH_SIZES.items()}
    return {"parts": parts, "macs": macs, "params": params, "mmd": mmd}


def encoder(tokens: int, width: int, ff: int = 4) -> tuple[int, int]:
    """One transformer encoder layer (projections, attention, feed-forward) plus the input embedding."""
    embed = dense(N_FEATURES // tokens, width)
    proj = 4 * tokens * width * width
    attn = 2 * tokens * tokens * width
    ffn = 2 * tokens * width * ff * width
    return tokens * embed[0] + proj + attn + ffn, embed[1] + 4 * width * width + 2 * ff * width * width


def amds_pffa() -> dict:
    known = {"S head (128 → 128 → 128)": total({"a": dense(128, 128), "b": dense(128, 128)}),
             "G1, G2 heads (128 → 128 → 64) ×2": tuple(2 * v for v in total({"a": dense(128, 128), "b": dense(128, 64)})),
             "P1, P2 conv 128@1 → 64@1 ×2": tuple(2 * v for v in conv(1, 1, 128, 64)),   # kernel 3 on length 1: centre tap
             "iAFF ×2 (2 MS-CAM, 64 ↔ 16)": (2 * 2 * 2 * (64 * 16 + 16 * 64), 2 * 2 * 2 * (64 * 16 + 16 + 16 * 64 + 64)),
             "classifiers C1, C2 (64 → 25 → 6) ×2": tuple(2 * v for v in total({"a": dense(64, 25), "b": dense(25, 6)}))}
    km, kp = total(known)
    layouts = {"16 sensor tokens × 8 wide": encoder(16, 8), "1 token × 128 wide": encoder(1, 128)}
    return {"parts": known, "macs": km, "params": kp,
            "encoders": {k: (3 * m, 3 * p) for k, (m, p) in layouts.items()}}   # S, G1, G2 at N = 1


def ours_inference_macs() -> int:
    """Base inference plus the mean cost of pseudo-label updates on Batches 6–10, as combine_system2.py."""
    import pandas as pd
    df = pd.read_csv("results/system2_combined.csv", usecols=["config", "target", "updates", "scored"])
    d = df[(df.config == FULL) & df.target.isin(I.TEST)]
    return round(I.inference_macs({}) + (d.updates * I.D_IN * I.N_CLASSES / d.scored).mean())


def system2_and_daelm() -> dict:
    d, c, L, n_big = I.D_IN, N_GASES, ELM_HIDDEN, ALL_BATCH_SIZES[7]
    # System 2+: 50 calibration passes, forward + gradient ≈ 3 × 256 × 6 MACs per labelled sample per pass
    calibration = {n: CALIBRATION_PASSES * n * 3 * d * c for n in LABELS}
    state = (d * c + c + 2 * d + 2 * d) * 4                     # weights, biases, EMA mean/var, source scaling
    buffer = {n: n * d * 4 for n in LABELS}
    # DAELM-S: hidden outputs of stored source + labelled target, HᵀH, one L×L solve (source Batch 1 or 7, 50 labels)
    solve = {b: (ns + LABELS[1]) * L * N_FEATURES + (ns + LABELS[1]) * L * L + L ** 3 / 3 for b, ns in BATCH_SIZES.items()}
    # SRAM-inclusive accounting of System 2 (no pseudo-label updates) and DA-ELM inference, energy_proxy.py constants
    from energy_proxy import CONSTANTS
    mac, sram, op = (CONSTANTS[k][0] for k in ("mac", "sram", "op"))
    fn = 20 * op                                                # log / exp / rsqrt ≈ 20 arithmetic operations
    ours_sram = (N_FEATURES * fn + 3 * N_FEATURES * op          # log-pattern features
                 + d * (10 * op + fn + 2 * 2 * sram)            # EMA alignment: read and write mean and variance
                 + d * c * (mac + sram) + c * op)               # readout with weight reads, argmax
    daelm_sram = (N_FEATURES * 2 * op + L * N_FEATURES * (mac + sram + op) + L * fn + L * c * (mac + sram) + c * op)
    return {"calibration": calibration, "state": state, "buffer": buffer, "solve": solve,
            "source": {b: n * N_FEATURES * 4 for b, n in BATCH_SIZES.items()},
            "hidden": {b: (n + LABELS[1]) * L * 4 for b, n in BATCH_SIZES.items()},
            "inverse": L * L * 4, "daelm_model": (L * N_FEATURES + L + L * c) * 4,
            "unlabelled": n_big * N_FEATURES * 4, "unlabelled_hidden": n_big * L * 4,
            "farthest_point": n_big ** 2 / 2 * N_FEATURES, "ours_sram_j": ours_sram, "daelm_sram_j": daelm_sram}


def fmt_j(j: float) -> str:
    for scale, unit in ((1e-3, "mJ"), (1e-6, "µJ")):
        if j >= scale:
            return f"{j / scale:,.1f} {unit}"
    return f"{j * 1e9:,.1f} nJ" if j < 1e-7 else f"{j * 1e9:,.0f} nJ"


def fmt_e(macs: float) -> str:
    return fmt_j(macs * PJ_PER_MAC)


def fmt_b(n_bytes: float) -> str:
    return f"{n_bytes / 1e3:,.1f} KB" if n_bytes < 1e5 else f"{n_bytes / 1e6:,.2f} MB"


def main() -> None:
    ours = ours_inference_macs()
    s = system2_and_daelm()
    print(f"System 2+: inference {ours:,} MACs ({I.inference_macs({}):,} + pseudo-label updates), {fmt_e(ours)}; "
          f"model + running state {fmt_b(s['state'])} FP32")
    for n in LABELS:
        print(f"  calibration, {n} labels × {CALIBRATION_PASSES} passes: {s['calibration'][n] / 1e6:.1f} M MACs, "
              f"{fmt_e(s['calibration'][n])}; labelled buffer {fmt_b(s['buffer'][n])}; "
              f"state + buffer {fmt_b(s['state'] + s['buffer'][n])}")
    print(f"DA-ELM: inference {I.DAELM_MACS:,} MACs, {fmt_e(I.DAELM_MACS)} ({I.DAELM_MACS / ours:.0f}× System 2+); "
          f"model {fmt_b(s['daelm_model'])} FP32 / {fmt_b(s['daelm_model'] / 4)} int8")
    for b, m in s["solve"].items():
        print(f"  DAELM-S adaptation, source Batch {b} + {LABELS[1]} labels: {m / 1e9:.2f} G MACs, {fmt_e(m)}; stored "
              f"source {fmt_b(s['source'][b])} + hidden outputs {fmt_b(s['hidden'][b])} + solve {fmt_b(s['inverse'])}")
    print(f"  DAELM-T extra, Batch 7 target: unlabelled batch {fmt_b(s['unlabelled'])} + hidden outputs "
          f"{fmt_b(s['unlabelled_hidden'])}")
    print(f"  label selection (farthest point), Batch 7: {s['farthest_point'] / 1e9:.2f} G MACs, "
          f"{fmt_e(s['farthest_point'])}; buffer {fmt_b(s['unlabelled'])}")
    print(f"SRAM-inclusive inference: System 2 (no pseudo-label updates) {fmt_j(s['ours_sram_j'])}, "
          f"DA-ELM {fmt_j(s['daelm_sram_j'])} ({s['daelm_sram_j'] / s['ours_sram_j']:.0f}×)\n")

    t = tdacnn()
    print("TDACNN (UCI configuration)")
    for k, (m, p) in t["parts"].items():
        print(f"  {k:<34}{m:>10,} MACs{p:>10,} params")
    print(f"  {'network':<34}{t['macs']:>10,} MACs{t['params']:>10,} params   "
          f"{fmt_e(t['macs'])}, {fmt_b(4 * t['params'])} FP32 / {fmt_b(t['params'])} int8")
    for b, (m, n) in t["mmd"].items():
        print(f"  + MMD ensemble, source Batch {b}: ≈{m:,} MACs per sample; stored source embeddings "
              f"{n:,} values = {fmt_b(4 * n)} FP32")

    a = amds_pffa()
    print("\nAMDS-PFFA (UCI configuration; inference passes the target through S, G1, G2, P1, P2, both pairs)")
    for k, (m, p) in a["parts"].items():
        print(f"  {k:<40}{m:>10,} MACs{p:>10,} params")
    print(f"  {'specified parts':<40}{a['macs']:>10,} MACs{a['params']:>10,} params")
    for k, (m, p) in a["encoders"].items():
        mm, pp = a["macs"] + m, a["params"] + p
        print(f"  + 3 encoders, N = 1, {k:<24}{m:>10,} MACs{p:>10,} params → total {mm:,} MACs, {fmt_e(mm)}, "
              f"{fmt_b(4 * pp)} FP32 / {fmt_b(pp)} int8")

    L = ELM_HIDDEN
    print(f"\nODAELM: inference as DA-ELM ({I.DAELM_MACS:,} MACs, {fmt_e(I.DAELM_MACS)}); the online update keeps "
          f"K⁻¹ ({L}×{L}) = {fmt_b(s['inverse'])} FP32 and costs ≈{2 * L * L:,} MACs ({fmt_e(2 * L * L)}) per update; "
          f"≤250 nodes would need {fmt_b(250 * 250 * 4)}")
    print(f"Raw samples AMDS-PFFA must hold to adapt to Batch 7 (Batches 1, 2 and 7): "
          f"{fmt_b((ALL_BATCH_SIZES[1] + ALL_BATCH_SIZES[2] + ALL_BATCH_SIZES[7]) * N_FEATURES * 4)} FP32")


if __name__ == "__main__":
    main()
