"""
Where is the hardest class (class 4) lost under drift? Per batch, cross-validated linear probes on three
representations of the same samples (labels are used for this diagnosis only):

    features   aligned sensor features, before spike encoding (what the encoder sees)
    inputs     firing rates of the LIF input layer (after spike encoding)
    STDP       spike counts of the frozen STDP layer (what the readout sees)

Each probe is a standardised logistic regression trained and tested *within* one batch
(stratified k-fold), so it measures how separable the gases are in that representation at
that point in time: an upper bound for any readout recalibrated on that batch. If
class 4 vs class 5 is separable in `features` but not in `STDP`, the SNN loses it; if it
is not separable even in `features`, neither a readout nor an SNN change can recover it.

The network is trained on Batch 1 and frozen exactly as in benchmark_drift.py, with the same
options (alignment, features, seeds); Batch 1 (all samples) is the in-distribution reference.

    python3 probe_drift.py --align-ema
    python3 probe_drift.py --align-ema --seeds 42 --batches 2 8
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import benchmark_drift as D
from benchmark_batch1 import rate_vector
from dataset_adapter import GAS_ORDER, gas_name, load_batch

LEVELS = ["features", "inputs", "STDP"]
TARGET, CONFUSER = gas_name(4), gas_name(5)  # classes 4 and 5


def represent_chunk(net: D.FrozenNetwork, batch_id: int, start: int, stop: int, seed: int,
                    affine: D.ChannelAffine | None) -> dict:
    """Samples [start, stop) of one batch through the frozen pipeline: labels and the three representations."""
    truth, filtered = D.encode_chunk(net.args, net.bounds, batch_id, start, stop, seed, affine)
    features = load_batch(batch_id).subset(np.arange(start, stop)).transformed_channels(net.args.feature_mode, affine)
    counts = np.stack([net.readouts.layer.process(f, learn=False).spike_counts for f in filtered])
    return {"truth": truth, "features": features, "inputs": np.stack([rate_vector(f) for f in filtered]),
            "STDP": counts}


def cv_predictions(x: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions of a standardised logistic regression (stratified k-fold)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = max(2, min(folds, min(np.sum(y == c) for c in np.unique(y))))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return cross_val_predict(model, x, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed))


def probe_batch(chunks: list[dict], folds: int, seed: int) -> dict:
    y = np.concatenate([c["truth"] for c in chunks])
    out = {"n": len(y), "n_target": int(np.sum(y == TARGET))}
    for level in LEVELS:
        pred = cv_predictions(np.concatenate([c[level] for c in chunks]), y, folds, seed)
        target = y == TARGET
        out[level] = {"accuracy": float(np.mean(pred == y)),
                      "target_correct": int(np.sum(pred[target] == TARGET)),
                      "target_as_confuser": int(np.sum(pred[target] == CONFUSER))}
    return out


def main() -> None:
    parser = D.build_parser()
    parser.add_argument("--folds", type=int, default=5, help="cross-validation folds per batch")
    args = D.resolve_args(parser, parser.parse_args())
    if args.adapt_stdp or args.align_ema_balanced or args.few_shot_samples:
        parser.error("probe_drift.py probes the frozen network: use no alignment, --standardize-channels, "
                     "--align-channels or --align-ema")
    batches = [1] + args.batches
    sizes = {b: len(load_batch(b)) for b in batches}
    months = {b: load_batch(b).months for b in batches}
    print(f"Linear probes per batch ({args.folds}-fold, within batch): {args.feature_mode} features, "
          f"{'EMA alignment' if args.align_ema else 'whole-batch alignment' if args.align_channels else 'no alignment'}, "
          f"{args.neurons} STDP neurons; seeds {args.seeds}", flush=True)

    results: dict[int, dict[int, dict]] = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for seed in args.seeds:
            net, _ = D.train_on_batch1(args, seed)
            ema = D.ema_affines(args, net.ema_init, seed, args.batches) if args.align_ema else None
            affines = {1: net.std_affine,
                       **{b: ema[b] if ema else D.drift_affine(args, net, load_batch(b).channel_moments(args.feature_mode))
                          for b in args.batches}}
            futures = {b: [pool.submit(represent_chunk, net, b, s, min(s + args.chunk_size, sizes[b]), seed,
                                       D.affine_rows(affines[b], s, min(s + args.chunk_size, sizes[b])))
                           for s in range(0, sizes[b], args.chunk_size)] for b in batches}
            results[seed] = {b: probe_batch([f.result() for f in fs], args.folds, seed) for b, fs in futures.items()}
            print(f"  seed {seed}: done [{time.time() - t0:.0f} s]", flush=True)

    seeds = sorted(results)
    fmt = lambda v: f"{100 * np.mean(v):5.1f} ± {100 * np.std(v, ddof=1):4.1f}" if len(v) > 1 else f"{100 * v[0]:5.1f}%"
    w = 13 if len(seeds) > 1 else 7
    print(f"\n{'':<27}│ {'probe accuracy (all gases)':^{3 * (w + 1)}} │ {TARGET + ' recall':^{3 * (w + 1)}} │ "
          f"{TARGET + ' → ' + CONFUSER:>{w + 1}}")
    print(f"{'batch':>6}  {'months':<12} {'n':>5} │" + "".join(f" {lv:>{w}}" for lv in LEVELS) + " │"
          + "".join(f" {lv:>{w}}" for lv in LEVELS) + f" │ {'(STDP)':>{w}}")
    for b in batches:
        r = [results[s][b] for s in seeds]
        rec = lambda lv: [x[lv]["target_correct"] / max(x["n_target"], 1) for x in r]
        conf = [x["STDP"]["target_as_confuser"] / max(x["n_target"], 1) for x in r]
        print(f"{str(b) + ('*' if b == 1 else ''):>6}  {months[b]:<12} {r[0]['n']:>5} │"
              + "".join(f" {fmt([x[lv]['accuracy'] for x in r]):>{w}}" for lv in LEVELS) + " │"
              + "".join(f" {fmt(rec(lv)):>{w}}" for lv in LEVELS) + f" │ {fmt(conf):>{w}}")
    drift = [b for b in batches if b != 1]
    pooled = {lv: [sum(results[s][b][lv]["target_correct"] for b in drift) / sum(results[s][b]["n_target"] for b in drift)
                   for s in seeds] for lv in LEVELS}
    mean_acc = {lv: [np.mean([results[s][b][lv]["accuracy"] for b in drift]) for s in seeds] for lv in LEVELS}
    print(f"{'drift batches':<27}│" + "".join(f" {fmt(mean_acc[lv]):>{w}}" for lv in LEVELS) + " │"
          + "".join(f" {fmt(pooled[lv]):>{w}}" for lv in LEVELS) + " │")
    print(f"* Batch 1: all samples, in-distribution. Accuracy: mean over batches; {TARGET} recall pooled over "
          f"drift samples. Labels are used only to fit and score these diagnostic probes.")


if __name__ == "__main__":
    main()
