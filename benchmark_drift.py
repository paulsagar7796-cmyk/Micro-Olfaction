"""
Chronological drift benchmark: train the SNN on Batch 1 (months 1–2) and test it on
Batches 2–10 (up to month 36) to measure resilience to sensor ageing.

    python3 benchmark_drift.py                          # 5 seeds, frozen network
    python3 benchmark_drift.py --adapt-stdp             # 5 seeds, STDP keeps learning, unlabelled
    python3 benchmark_drift.py --seeds 42 --normalization robust

Phase 1  Batch 1, stratified 80/20 split (exactly as benchmark_batch1.py): fit any scaler
         bounds on the training split, train STDP (unsupervised) and the readout(s)
         (supervised) on it.
Phase 2  Freeze the scaler bounds and the readout weights. Without --adapt-stdp the STDP
         layer is frozen too. Checks after every chunk / batch confirm nothing frozen changed.
Phase 3  Batches 2–10 in chronological order, each sample encoded with the Batch 1 scaler
         (per-sample scaling needs none):
           frozen        every sample is classified by the frozen network (parallel chunks).
           --adapt-stdp  continuous unsupervised adaptation, one stream per seed. Each sample
                         is first presented to the STDP layer with learning on (its first
                         --stdp-duration seconds, as in training; STDP weights and adaptive
                         thresholds update), then classified: a frozen pass of the adapted
                         layer over the whole sample, read by the frozen readout. The adapted
                         state carries over to later samples and batches. Labels of Batches
                         2–10 are only used to score predictions, never by a learning rule.
                         Samples carry no timestamps within a batch (file order is grouped by
                         gas), so each batch is streamed in a seeded random order.
         Batch 1's held-out 20% is the in-distribution reference.

Both modes encode identical spike inputs (Poisson seeds depend on seed, batch and chunk
only), so their difference is the adaptation alone.

Per-channel drift correction (unsupervised; labels are never used):
  --align-channels         match each drift batch's per-channel mean and standard deviation,
                           computed over all its samples, to Batch 1's training split:
                           x → (x − μ_t)·σ₁/σ_t + μ₁. Implies --standardize-channels.
  --standardize-channels   standardise every raw channel with Batch 1's training statistics,
                           x → (x − μ₁)/σ₁, for training and all test data, before the
                           per-sample scaler. After alignment this equals (x − μ_t)/σ_t, i.e.
                           each drift batch standardised with its own statistics.
  --align-channels --no-standardize-channels   alignment only, into Batch 1's raw scale
                           (the network is then identical to the uncorrected one).
  --align-ema              online version: running per-channel mean and variance, updated by
                           every sample in stream order (batches chronologically, samples within
                           a batch in the seeded order above), starting from Batch 1's training
                           statistics; each sample is standardised with the statistics current
                           when it arrives. Uses the same Batch-1-standardised network.
                               μ ← α·μ + (1 − α)·x
                               v ← α·v + (1 − α)·(x − μ)²
                               x_std = (x − μ) / √(v + 1e-8)          (--ema-alpha α, default 0.99)
  --align-ema-balanced     class-balanced online version: one running (μ_c, σ²_c) per gas, started
                           from Batch 1's class-conditional training statistics. Each sample is
                           standardised with the class-balanced statistics (law of total variance,
                           every gas weighted equally), classified by the frozen network, and then
                           only the predicted gas's statistics are updated with the raw sample (no
                           update if the readout stays silent). Sequential by construction; true
                           labels are only used to score. The network is trained on Batch 1
                           standardised with the balanced statistics, so training and test match.
  --few-shot-samples N     periodic maintenance: on every drift batch, the first N samples of each
                           gas in stream order (all of them if fewer) form a labelled calibration
                           set. They pass through the alignment first (updating a running EMA),
                           then the frozen STDP layer, and fine-tune the spiking readout for
                           --calibration-epochs (its weights carry over from batch to batch). The
                           rest of the batch is then classified with the calibrated readout and
                           scored; calibration samples are never scored. STDP stays frozen.
                           --calibration-lr-scale s   fine-tune with the readout's learning rate × s
                           --few-shot-override GAS=M  use M calibration samples for that gas instead
                                                      of N (asymmetric boost; repeatable)
                           --rehearsal-samples K      add K Batch 1 training samples per gas (their
                                                      frozen STDP outputs, redrawn every batch) to
                                                      each calibration set, to limit forgetting
                           With --adapt-stdp as well (joint adaptation), every sample, calibration
                           and test alike, first updates the STDP layer without labels (as in
                           --adapt-stdp) and is then read out through the adapted layer.
  --channel-stats {pooled,balanced}   which Batch 1 statistics define the standardisation (and the
                           --align-ema starting point): pooled over all samples (default) or
                           class-balanced (implied by --align-ema-balanced).

Defaults: kinetics features, per-sample scaling, 64 STDP neurons trained for 1 epoch on
0.2 s presentations, spiking readout with partial class weighting, seeds 7 42 100 999 1234.
Reference column: logistic regression on Batch 1's input rates (non-spiking, frozen), to
separate data drift from SNN-specific degradation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark_batch1 import (DEFAULT_SEEDS, READOUTS, Readouts, add_pipeline_args, fit_readouts, make_input_layer,
                              make_stdp_layer, prepare_data, rate_vector, stdp_presentations, train_stdp)
from dataset_adapter import (GAS_ORDER, ChannelAffine, balanced_moments, channel_labels, encode_batch, load_batch,
                             rates_to_spike_train, scale_channels)
from readout_layer import SILENT
from snn_input import SpikeTrain
from stdp_layer import STDPLayerOutput

DRIFT_BATCHES = list(range(2, 11))


@dataclass
class FrozenNetwork:
    """Everything fitted on Batch 1 (Phase 1). Copied to worker processes."""
    args: argparse.Namespace
    bounds: dict | None           # scaler bounds from Batch 1's training split (None: per-sample scaling)
    readouts: Readouts            # includes the STDP layer (adapted in place with --adapt-stdp)
    reference: object             # LogisticRegression on Batch 1 input rates
    snapshot: dict[str, np.ndarray]
    channel_stats: tuple[np.ndarray, np.ndarray] | None = None  # per-channel (μ₁, σ₁), Batch 1 training split
    replay: tuple[list[SpikeTrain], np.ndarray] | None = None   # Batch 1 training STDP outputs + labels (rehearsal)
    class_stats: tuple[np.ndarray, np.ndarray] | None = None    # per-class, per-channel (μ_c, σ²_c), Batch 1 train
    std_affine: ChannelAffine | None = None                     # Phase 1 standardisation, if any
    ema_init: tuple[np.ndarray, np.ndarray] | None = None       # (μ, σ²) the --align-ema trackers start from

    def weights(self) -> dict[str, np.ndarray]:
        w = {"STDP weights": self.readouts.layer.w, "STDP thresholds": self.readouts.layer.theta}
        if self.readouts.spiking is not None:
            w["readout weights"] = self.readouts.spiking.w
        return w

    def unchanged(self, which: list[str] | None = None) -> bool:
        return all(np.array_equal(v, self.snapshot[k]) for k, v in self.weights().items()
                   if which is None or k in which)


def train_on_batch1(args: argparse.Namespace, seed: int) -> tuple[FrozenNetwork, dict]:
    """Phases 1–2: train on Batch 1's training split, freeze, and score Batch 1's held-out 20%."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    data = prepare_data(args, seed, say=lambda *_, **__: None,
                        standardize_channels=args.channel_stats if args.standardize_channels else False)
    layer = make_stdp_layer(args, data, seed)
    order = stdp_presentations(data.train_idx, data.labels, args.epochs, args.balance, np.random.default_rng(seed))
    train_stdp(layer, data, order, args.stdp_duration)
    train_out = [layer.process(data.filtered[i], learn=False) for i in data.train_idx]
    train_truth = data.labels[data.train_idx]
    with warnings.catch_warnings():  # logistic regressions on raw rates / counts, as in benchmark_batch1
        warnings.simplefilter("ignore", ConvergenceWarning)
        readouts = fit_readouts(layer, train_out, train_truth, args, seed)
        reference = LogisticRegression(max_iter=1000).fit(data.lif_rates[data.train_idx], train_truth)
    ema_init = (balanced_moments(*data.class_channel_stats) if args.channel_stats == "balanced"
                else (data.channel_stats[0], data.channel_stats[1] ** 2))
    net = FrozenNetwork(args, data.bounds, readouts, reference, snapshot={}, channel_stats=data.channel_stats,
                        class_stats=data.class_channel_stats, std_affine=data.channel_affine, ema_init=ema_init)
    if getattr(args, "rehearsal_samples", 0):  # kept only when needed: it is large to ship to workers
        net.replay = ([o.spike_train for o in train_out], train_truth)
    net.snapshot = {k: v.copy() for k, v in net.weights().items()}

    test_out = [layer.process(data.filtered[i], learn=False) for i in data.test_idx]
    heldout = {"truth": data.labels[data.test_idx],
               "predictions": {n: readouts.predict(n, test_out) for n in args.readout},
               "reference": reference.predict(data.lif_rates[data.test_idx]),
               "neuron_spikes": np.sum([o.spike_counts for o in test_out], axis=0), "frozen": net.unchanged()}
    return net, heldout


def drift_affine(args: argparse.Namespace, net: FrozenNetwork,
                 stats_t: tuple[np.ndarray, np.ndarray]) -> ChannelAffine | None:
    """Per-channel transform for one drift batch: optional alignment of its moments (stats_t,
    from its own unlabelled samples) to Batch 1's, then Batch 1's standardisation."""
    if not args.align_channels:
        return net.std_affine        # Phase 1 standardisation (or None)
    mu1, sigma1 = net.channel_stats
    scale, shift = np.ones_like(mu1), np.zeros_like(mu1)
    if args.align_channels:        # (x − μ_t)·σ₁/σ_t + μ₁
        mu_t, sigma_t = stats_t
        scale = sigma1 / sigma_t
        shift = mu1 - mu_t * scale
    if args.standardize_channels:  # then (· − μ₁)/σ₁, exactly as Batch 1 was trained
        scale, shift = scale / sigma1, (shift - mu1) / sigma1
    return scale, shift


def stream_order(seed: int, batch_id: int, n: int) -> np.ndarray:
    """Order in which a batch's samples arrive (no timestamps within a batch: seeded shuffle)."""
    return np.random.default_rng([seed, batch_id]).permutation(n)


def ema_affines(args: argparse.Namespace, init: tuple[np.ndarray, np.ndarray], seed: int,
                batches: list[int], orders: dict[int, np.ndarray] | None = None) -> dict[int, ChannelAffine]:
    """Online per-channel standardisation: stream Batches 2–10 in order, one EMA update per sample,
    starting from Batch 1's training (μ, σ²) `init`. Returns per-sample (scale, shift), shape
    (n, channels), in each batch's file order. Unsupervised: only the channel values are used."""
    alpha = args.ema_alpha
    mu, var = init[0].copy(), init[1].copy()
    affines = {}
    for b in sorted(batches):
        x_all = load_batch(b).channel_matrix(args.feature_mode)
        scale, shift = np.empty_like(x_all), np.empty_like(x_all)
        for i in (orders[b] if orders else stream_order(seed, b, len(x_all))):
            x = x_all[i]
            mu = alpha * mu + (1 - alpha) * x
            var = alpha * var + (1 - alpha) * (x - mu) ** 2
            scale[i] = 1.0 / np.sqrt(var + 1e-8)
            shift[i] = -mu * scale[i]
        affines[b] = (scale, shift)
    return affines


def affine_rows(affine: ChannelAffine | None, start: int, stop: int) -> ChannelAffine | None:
    """The part of a batch's transform for samples [start, stop): per-sample transforms are sliced."""
    if affine is None or affine[0].ndim == 1:
        return affine
    return affine[0][start:stop], affine[1][start:stop]


def poisson_uniforms(seed: int, batch_id: int, index: int, chunk_size: int, n_steps: int, n_ch: int) -> np.ndarray:
    """The uniform draws encode_chunk's Poisson encoder uses for sample `index` of a batch, regenerated
    directly: same chunk seed, generator advanced past the draws of the chunk's earlier samples."""
    index = int(index)  # plain int: advance() masks with a 128-bit integer, which NumPy integers overflow
    start = (index // chunk_size) * chunk_size
    bitgen = np.random.PCG64(int(np.random.SeedSequence([seed, batch_id, start]).generate_state(1)[0]))
    bitgen.advance((index - start) * n_steps * n_ch)
    return np.random.Generator(bitgen).random((n_steps, n_ch))


def encode_single(args: argparse.Namespace, bounds: dict | None, batch_id: int, x: np.ndarray, index: int,
                  channel_affine: ChannelAffine, seed: int) -> SpikeTrain:
    """One raw sample → spike train, exactly as encode_chunk encodes it (same rates, same Poisson draws)."""
    encoded = scale_channels(x * channel_affine[0] + channel_affine[1], args.feature_mode, args.normalization,
                             bounds)[0]
    rates, dt = args.f_max * encoded, 0.001
    if args.mode != "poisson":
        return rates_to_spike_train(rates, args.duration, dt, args.mode, labels=channel_labels(args.feature_mode))
    uniforms = poisson_uniforms(seed, batch_id, index, args.chunk_size, int(round(args.duration / dt)), len(rates))
    return SpikeTrain(uniforms < rates * dt, dt, channel_labels(args.feature_mode))


def encode_chunk(args: argparse.Namespace, bounds: dict | None, batch_id: int, start: int, stop: int,
                 seed: int, channel_affine: ChannelAffine | None = None) -> tuple[np.ndarray, list[SpikeTrain]]:
    """Samples [start, stop) of one batch → (gas labels, LIF-filtered spike trains), encoded with
    the frozen Batch 1 scaler (after the optional per-channel transform). Poisson spikes are
    seeded by (seed, batch, start) only."""
    if args.normalization != "sample" and bounds is None:
        raise ValueError("frozen Batch 1 scaler bounds are required")  # never re-fit on drift data
    batch = load_batch(batch_id).subset(np.arange(start, stop))
    chunk_seed = int(np.random.SeedSequence([seed, batch_id, start]).generate_state(1)[0])
    samples, _ = encode_batch(batch, duration_s=args.duration, f_max=args.f_max, mode=args.mode, seed=chunk_seed,
                              feature_mode=args.feature_mode, normalization=args.normalization, scaler_bounds=bounds,
                              channel_affine=channel_affine)
    lif = make_input_layer(args, samples[0].spike_train.raster.shape[1])
    return batch.gas, [lif.run(s.spike_train).spike_train for s in samples]


def score(net: FrozenNetwork, truth: np.ndarray, filtered: list[SpikeTrain],
          outputs: list[STDPLayerOutput]) -> dict:
    return {"truth": truth,
            "predictions": {n: net.readouts.predict(n, outputs) for n in net.args.readout},
            "reference": net.reference.predict(np.stack([rate_vector(f) for f in filtered])),
            "neuron_spikes": np.sum([o.spike_counts for o in outputs], axis=0)}


def classify_chunk(net: FrozenNetwork, batch_id: int, start: int, stop: int, seed: int,
                   channel_affine: ChannelAffine | None = None) -> dict:
    """Frozen Phase 3: classify samples [start, stop) of one batch with the frozen network."""
    truth, filtered = encode_chunk(net.args, net.bounds, batch_id, start, stop, seed, channel_affine)
    outputs = [net.readouts.layer.process(f, learn=False) for f in filtered]
    return {**score(net, truth, filtered, outputs), "frozen": net.unchanged()}


def run_seed_stream(args: argparse.Namespace, seed: int, adapt: bool = True) -> dict[int, dict]:
    """Phases 1–3 for one seed, streaming Batches 2–10 in order through one STDP layer.

    With adapt=True the STDP layer learns from every sample (unsupervised) before classifying
    it; with adapt=False this is the frozen benchmark computed sequentially (used as a check).
    """
    t0 = time.time()
    net, heldout = train_on_batch1(args, seed)
    layer = net.readouts.layer
    w_batch1 = layer.w.copy()
    if adapt and args.adapt_lr_scale != 1:  # gentler (or stronger) STDP during Phase 3 only
        layer.stdp = dataclasses.replace(layer.stdp, eta_plus=layer.stdp.eta_plus * args.adapt_lr_scale,
                                         eta_minus=layer.stdp.eta_minus * args.adapt_lr_scale)
    results = {1: summarize_batch([heldout], args.readout)}
    ema = ema_affines(args, net.ema_init, seed, args.batches) if args.align_ema else None
    for b in sorted(args.batches):
        n = len(load_batch(b))
        affine = ema[b] if ema else drift_affine(args, net, load_batch(b).channel_moments(args.feature_mode))
        encoded = [encode_chunk(args, net.bounds, b, s, min(s + args.chunk_size, n), seed,
                                affine_rows(affine, s, min(s + args.chunk_size, n)))
                   for s in range(0, n, args.chunk_size)]
        truth = np.concatenate([t for t, _ in encoded])      # used for scoring only, below
        filtered = [f for _, fs in encoded for f in fs]
        outputs: list[STDPLayerOutput | None] = [None] * n
        for i in stream_order(seed, b, n):
            if adapt:  # unsupervised: spike input only, no label
                layer.process(filtered[i], duration_s=args.stdp_duration, learn=True)
            outputs[i] = layer.process(filtered[i], learn=False)
        chunk = {**score(net, truth, filtered, outputs),
                 "frozen": net.unchanged(["readout weights"] if adapt else None)}
        results[b] = summarize_batch([chunk], args.readout)
        results[b]["weight_change"] = float(np.linalg.norm(layer.w - w_batch1) / np.linalg.norm(w_batch1))
        print(f"  seed {seed}: batch {b} {'adapted and ' if adapt else ''}classified "
              f"({results[b]['accuracy'][args.readout[0]]:.1%}) [{time.time() - t0:.0f} s]", flush=True)
    return results


def run_seed_balanced(args: argparse.Namespace, seed: int) -> dict[int, dict]:
    """Phases 1–3 with the pseudo-label-conditioned, class-balanced EMA (--align-ema-balanced).

    Sample by sample, in stream order: standardise with the class-balanced statistics of the running
    per-gas trackers, classify with the frozen network, then update the tracker of the *predicted*
    gas with the raw sample. True labels are only used to score, after each batch.
    """
    t0 = time.time()
    net, heldout = train_on_batch1(args, seed)
    mu_c, var_c = net.class_stats[0].copy(), net.class_stats[1].copy()
    lif = make_input_layer(args, len(channel_labels(args.feature_mode)))
    primary, alpha = args.readout[0], args.ema_alpha
    results = {1: summarize_batch([heldout], args.readout)}
    for b in sorted(args.batches):
        batch = load_batch(b)
        x_all, n = batch.channel_matrix(args.feature_mode), len(batch)
        filtered: list[SpikeTrain | None] = [None] * n
        outputs: list[STDPLayerOutput | None] = [None] * n
        predictions = {name: np.empty(n, dtype=object) for name in args.readout}
        updates = np.zeros(len(GAS_ORDER), dtype=int)
        for i in stream_order(seed, b, n):
            macro_mu, macro_var = balanced_moments(mu_c, var_c)
            inv = 1.0 / np.sqrt(macro_var + 1e-8)                     # x_std = (x − μ) / √(σ² + 1e-8)
            filtered[i] = lif.run(encode_single(args, net.bounds, b, x_all[i], i, (inv, -macro_mu * inv),
                                                seed)).spike_train
            outputs[i] = net.readouts.layer.process(filtered[i], learn=False)
            for name in args.readout:
                predictions[name][i] = net.readouts.predict(name, [outputs[i]])[0]
            if predictions[primary][i] in GAS_ORDER:                   # silent readout → no update
                c = GAS_ORDER.index(predictions[primary][i])
                updates[c] += 1
                mu_c[c] = alpha * mu_c[c] + (1 - alpha) * x_all[i]
                var_c[c] = alpha * var_c[c] + (1 - alpha) * (x_all[i] - mu_c[c]) ** 2
        chunk = {"truth": batch.gas, "predictions": predictions,
                 "reference": net.reference.predict(np.stack([rate_vector(f) for f in filtered])),
                 "neuron_spikes": np.sum([o.spike_counts for o in outputs], axis=0), "frozen": net.unchanged()}
        results[b] = summarize_batch([chunk], args.readout)
        results[b]["updates"] = updates
        print(f"  seed {seed}: batch {b} classified ({results[b]['accuracy'][primary]:.1%}) "
              f"[{time.time() - t0:.0f} s]", flush=True)
    return results


def fewshot_split(labels: np.ndarray, order: np.ndarray, per_class: int,
                  overrides: dict[str, int] | None = None, safe: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(calibration, test) indices, both in stream order: the first `per_class` samples of each gas
    (`overrides[gas]` for listed gases) to arrive calibrate, all of them if fewer; the rest are test.
    With `safe`, a gas with fewer than 2 × its quota samples gives half of them (rounded down) to
    calibration, so it always keeps test samples."""
    counts = {g: int(np.sum(labels == g)) for g in set(labels)}

    def quota(gas: str) -> int:
        q = (overrides or {}).get(gas, per_class)
        return counts[gas] // 2 if safe and counts[gas] < 2 * q else q
    seen: dict[str, int] = {}
    calibration, test = [], []
    for i in order:
        seen[labels[i]] = seen.get(labels[i], 0) + 1
        (calibration if seen[labels[i]] <= quota(labels[i]) else test).append(i)
    return np.array(calibration, dtype=int), np.array(test, dtype=int)


def run_seed_fewshot(args: argparse.Namespace, seed: int) -> dict[int, dict]:
    """Phases 1–3 with few-shot readout calibration on every drift batch (--few-shot-samples).

    Per batch: the calibration set (first N per gas in stream order) goes through the alignment,
    the STDP layer and fine-tunes the spiking readout (weights carry over); then the rest of the
    batch is classified with the calibrated readout and scored. With --calibration-epochs 0 the
    readout is not fine-tuned (same split, same alignment order: the control). With --adapt-stdp,
    each sample (calibration and test) first adapts the STDP layer, unsupervised, on its first
    --stdp-duration seconds, then is read out by a frozen pass of the adapted layer.
    """
    t0 = time.time()
    net, heldout = train_on_batch1(args, seed)
    layer, adapt = net.readouts.layer, args.adapt_stdp
    w_batch1 = layer.w.copy()
    if adapt and args.adapt_lr_scale != 1:  # gentler (or stronger) STDP during Phase 3 only
        layer.stdp = dataclasses.replace(layer.stdp, eta_plus=layer.stdp.eta_plus * args.adapt_lr_scale,
                                         eta_minus=layer.stdp.eta_minus * args.adapt_lr_scale)
    results = {1: summarize_batch([heldout], args.readout)}
    splits = {}
    for b in sorted(args.batches):
        gas = load_batch(b).gas
        overrides = args.few_shot_override
        splits[b] = fewshot_split(gas, stream_order(seed, b, len(gas)), args.few_shot_samples, overrides,
                                  safe=args.safe_split)
    orders = {b: np.concatenate(split) for b, split in splits.items()}   # calibration first, then test
    ema = ema_affines(args, net.ema_init, seed, args.batches, orders) if args.align_ema else None
    n_ch = len(channel_labels(args.feature_mode))
    lif = make_input_layer(args, n_ch)

    for b in sorted(args.batches):
        batch = load_batch(b)
        x_all = batch.channel_matrix(args.feature_mode)
        calibration, test = splits[b]
        const = (ema is None and (drift_affine(args, net, batch.channel_moments(args.feature_mode))
                                  or (np.ones(n_ch), np.zeros(n_ch))))
        row = (lambda i: (ema[b][0][i], ema[b][1][i])) if ema else (lambda i: const)

        def stdp_outputs(indices: np.ndarray) -> tuple[list[SpikeTrain], list[STDPLayerOutput]]:
            """Samples in stream order → LIF input spike trains and STDP outputs; with --adapt-stdp each
            sample first updates the layer (unsupervised), then is read out by the adapted layer."""
            filtered, outputs = [], []
            for i in indices:
                f = lif.run(encode_single(args, net.bounds, b, x_all[i], i, row(i), seed)).spike_train
                if adapt:
                    layer.process(f, duration_s=args.stdp_duration, learn=True)
                filtered.append(f)
                outputs.append(layer.process(f, learn=False))
            return filtered, outputs

        # Step A: calibrate the readout on the labelled calibration samples (supervised maintenance)
        if len(calibration) and (args.calibration_epochs > 0 or adapt):
            _, cal_out = stdp_outputs(calibration)
        if len(calibration) and args.calibration_epochs > 0:
            trains, labels = [o.spike_train for o in cal_out], list(batch.gas[calibration])
            if args.rehearsal_samples:  # replay a few Batch 1 training samples per gas (redrawn each batch)
                replay_trains, replay_labels = net.replay
                rng = np.random.default_rng([seed, b, 1])
                pick = np.concatenate([rng.choice(np.flatnonzero(replay_labels == g), replace=False,
                                                  size=min(args.rehearsal_samples, int(np.sum(replay_labels == g))))
                                       for g in GAS_ORDER])
                trains += [replay_trains[i] for i in pick]
                labels += list(replay_labels[pick])
            readout = net.readouts.spiking
            readout.fit(trains, labels, epochs=args.calibration_epochs, eta=readout.p.eta * args.calibration_lr_scale)
        # Step B: classify the remaining samples with the calibrated readout; score only these
        chunks, samples = [], []
        for part in np.array_split(test, max(1, int(np.ceil(len(test) / args.chunk_size)))):
            filtered, outputs = stdp_outputs(part)
            chunks.append({**score(net, batch.gas[part], filtered, outputs),
                           "frozen": True if adapt else net.unchanged(["STDP weights", "STDP thresholds"])})
            if args.save_predictions:  # per-sample prediction and confidence of the spiking readout
                counts = net.readouts.spiking.output_counts([o.spike_train for o in outputs])
                pred = net.readouts.spiking.predict_counts(counts)
                assert np.array_equal(pred, chunks[-1]["predictions"]["spiking"])
                total = counts.sum(axis=1)
                confidence = np.where(total > 0, counts.max(axis=1) / np.maximum(total, 1), 0.0)
                samples += [(int(i), g, p, float(c)) for i, g, p, c in zip(part, batch.gas[part], pred, confidence)]
        results[b] = summarize_batch(chunks, args.readout)
        results[b]["samples"] = samples
        results[b]["calibration"] = len(calibration)
        results[b]["weight_change"] = float(np.linalg.norm(layer.w - w_batch1) / np.linalg.norm(w_batch1))
        print(f"  seed {seed}: batch {b} calibrated on {len(calibration)} and scored on {len(test)} samples "
              f"({results[b]['accuracy'][args.readout[0]]:.1%}) [{time.time() - t0:.0f} s]", flush=True)
    return results


def summarize_batch(chunks: list[dict], readouts: list[str]) -> dict:
    truth = np.concatenate([c["truth"] for c in chunks])
    pred = {n: np.concatenate([c["predictions"][n] for c in chunks]) for n in readouts}
    ref = np.concatenate([c["reference"] for c in chunks])
    primary = pred[readouts[0]]
    return {
        "n": len(truth),
        "accuracy": {n: float(np.mean(p == truth)) for n, p in pred.items()},
        "silent": float(np.mean(primary == SILENT)),
        "reference": float(np.mean(ref == truth)),
        "active": int(np.sum(np.sum([c["neuron_spikes"] for c in chunks], axis=0) > 0)),
        "recall": {g: (int(np.sum((truth == g) & (primary == g))), int(np.sum(truth == g))) for g in GAS_ORDER},
        "frozen": all(c["frozen"] for c in chunks),
    }


def fmt(values: list[float], pct: bool = True) -> str:
    v = np.asarray(values, dtype=float) * (100 if pct else 1)
    unit = "%" if pct else ""
    if len(v) == 1:
        return f"{v[0]:.1f}{unit}"
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def print_report(per_seed: dict[int, dict[int, dict]], months: dict[int, str], args: argparse.Namespace) -> None:
    seeds, names = sorted(per_seed), args.readout
    batches = sorted(per_seed[seeds[0]])
    drift = [b for b in batches if b != 1]
    adapt = args.adapt_stdp
    fewshot = args.few_shot_samples > 0
    w = 10 if len(seeds) == 1 else 12
    extra = f" {'Δw':>{w}}" if adapt else ""
    print(f"\n{'batch':>6}  {'months':<12} {'scored' if fewshot else 'samples':>7}"
          + (f" {'calib':>5}" if fewshot else "") + " │" + "".join(f" {n:>{w}}" for n in names)
          + f" {'silent':>{w}} {'active':>{w}}{extra} │ {'LR on inputs':>{w + 2}}")
    for b in batches:
        r = [per_seed[s][b] for s in seeds]
        dw = f" {fmt([x.get('weight_change', 0.0) for x in r]):>{w}}" if adapt else ""
        cal = f" {r[0].get('calibration', 0):>5}" if fewshot else ""
        print(f"{str(b) + ('*' if b == 1 else ''):>6}  {months[b]:<12} {r[0]['n']:>7}{cal} │"
              + "".join(f" {fmt([x['accuracy'][n] for x in r]):>{w}}" for n in names)
              + f" {fmt([x['silent'] for x in r]):>{w}} {fmt([x['active'] for x in r], pct=False):>{w}}{dw} │"
              + f" {fmt([x['reference'] for x in r]):>{w + 2}}")
    mean_over = lambda get: [np.mean([get(per_seed[s][b]) for b in drift]) for s in seeds]
    pooled = lambda get: [sum(get(per_seed[s][b]) * per_seed[s][b]["n"] for b in drift)
                          / sum(per_seed[s][b]["n"] for b in drift) for s in seeds]
    n_drift = sum(per_seed[seeds[0]][b]["n"] for b in drift)
    pad = f" {'':>{w}} {'':>{w}}" + (f" {'':>{w}}" if adapt else "")
    for label, agg in ((f"mean of batches {drift[0]}–{drift[-1]}", mean_over),
                       (f"all {n_drift:,} {'scored' if fewshot else 'drift'} samples", pooled)):
        print(f"{label:<{34 if fewshot else 28}} │" + "".join(f" {fmt(agg(lambda x, n=n: x['accuracy'][n])):>{w}}" for n in names)
              + pad + f" │ {fmt(agg(lambda x: x['reference'])):>{w + 2}}")
    print(f"* Batch 1 row: its held-out 20% (in-distribution). silent: samples on which the {names[0]} readout "
          f"emitted no spikes. active: STDP neurons (of {args.neurons}) that fired on the batch."
          + (" Δw: ‖W − W_batch1‖ / ‖W_batch1‖ of the adapting STDP weights." if adapt else "")
          + (f" calib: labelled samples used to fine-tune the readout ({args.few_shot_samples} per gas, "
             f"{args.calibration_epochs} epochs), never scored; LR on inputs stays frozen." if fewshot else ""))
    if len(seeds) > 1:
        print(f"Values are mean ± std over {len(seeds)} seeds; summary rows average each seed's batches first.")

    recall = {g: [sum(per_seed[s][b]["recall"][g][i] for s in seeds for b in drift) for i in (0, 1)]
              for g in GAS_ORDER}
    print(f"\nRecall per gas over the drift batches ({READOUTS[names[0]]}): "
          + ", ".join(f"{g} {c / max(t, 1):.0%} ({c}/{t})" for g, (c, t) in recall.items()))
    if args.align_ema_balanced:
        print(f"\nTracker updates per batch — share of the batch's samples assigned to each gas's statistics "
              f"(true share in brackets), mean over seeds:")
        print(f"{'batch':>6} " + "".join(f" {g[:12]:>17}" for g in GAS_ORDER) + f" {'no update':>10}")
        for b in drift:
            n = per_seed[seeds[0]][b]["n"]
            upd = np.mean([per_seed[s][b]["updates"] for s in seeds], axis=0) / n
            true = [per_seed[seeds[0]][b]["recall"][g][1] / n for g in GAS_ORDER]
            print(f"{b:>6} " + "".join(f" {u:>8.0%} ({t:>4.0%})  " for u, t in zip(upd, true))
                  + f" {1 - upd.sum():>9.0%}")
    frozen = all(per_seed[s][b]["frozen"] for s in seeds for b in batches)
    what = ("readout weights" if adapt else "STDP weights and thresholds" if fewshot
            else "STDP weights, STDP thresholds and readout weights")
    if adapt and fewshot:
        print("Frozen check: not applicable — joint adaptation: the STDP layer adapted without labels and the "
              "readout was fine-tuned on the calibration samples only")
        return
    print(f"Frozen check: {what} {'unchanged throughout' if frozen else 'CHANGED — the frozen network was modified!'}"
          + ("; the STDP layer adapted without labels" if adapt else "")
          + ("; the readout was fine-tuned on the calibration samples only" if fewshot else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_pipeline_args(parser)
    parser.set_defaults(feature_mode="kinetics", normalization="sample", readout=["spiking"])
    parser.add_argument("--epochs", type=float, default=1, help="STDP training passes over Batch 1's training split")
    parser.add_argument("--stdp-duration", type=float, default=0.2,
                        help="seconds of each sample STDP learns from (training, and adaptation with --adapt-stdp)")
    parser.add_argument("--adapt-stdp", action="store_true",
                        help="keep STDP (and adaptive thresholds) learning, unsupervised, on Batches 2–10; "
                             "the readout stays frozen")
    parser.add_argument("--adapt-lr-scale", type=float, default=1.0,
                        help="with --adapt-stdp: multiply the STDP learning rates during adaptation (e.g. 0.1)")
    parser.add_argument("--align-channels", action="store_true",
                        help="match each drift batch's per-channel mean and std (unlabelled) to Batch 1's; "
                             "implies --standardize-channels")
    parser.add_argument("--align-ema", action="store_true",
                        help="online alignment: standardise each sample with running (EMA) per-channel "
                             "statistics, updated sample by sample; implies --standardize-channels")
    parser.add_argument("--ema-alpha", type=float, default=0.99,
                        help="EMA retention per sample for --align-ema (0.99 ≈ 100-sample memory, 0.95 ≈ 20)")
    parser.add_argument("--align-ema-balanced", action="store_true",
                        help="class-balanced online alignment: running statistics per predicted gas (pseudo-labels "
                             "from the frozen readout), combined with equal weights; implies --channel-stats balanced")
    parser.add_argument("--channel-stats", choices=["pooled", "balanced"], default=None,
                        help="Batch 1 statistics for the standardisation and the --align-ema start: pooled over all "
                             "samples, or class-balanced (default: balanced with --align-ema-balanced, else pooled)")
    parser.add_argument("--standardize-channels", action=argparse.BooleanOptionalAction, default=None,
                        help="standardise raw channels with Batch 1's training mean and std before the per-sample "
                             "scaler (default: on with --align-channels, off otherwise)")
    parser.add_argument("--few-shot-samples", type=int, default=0,
                        help="labelled calibration samples per gas and drift batch used to fine-tune the readout "
                             "(0: no calibration); they are excluded from scoring")
    parser.add_argument("--calibration-epochs", type=int, default=20,
                        help="readout fine-tuning epochs per drift batch with --few-shot-samples (0: no fine-tuning)")
    parser.add_argument("--few-shot-override", action="append", default=[], metavar="GAS=M",
                        help="with --few-shot-samples: calibration samples for one gas (default: same as others)")
    parser.add_argument("--safe-split", action="store_true",
                        help="with --few-shot-samples: a gas with fewer than 2 × N samples in a batch gives half of "
                             "them to calibration, so it keeps test samples")
    parser.add_argument("--calibration-lr-scale", type=float, default=1.0,
                        help="with --few-shot-samples: fine-tune with the readout learning rate × this")
    parser.add_argument("--rehearsal-samples", type=int, default=0,
                        help="with --few-shot-samples: Batch 1 training samples per gas replayed in each calibration")
    parser.add_argument("--save-predictions", type=Path, default=None,
                        help="with --few-shot-samples: CSV of per-sample spiking-readout predictions and confidence "
                             "(winning output neuron's share of output spikes) on the scored samples")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--batches", type=int, nargs="+", default=DRIFT_BATCHES, help="drift batches to test")
    parser.add_argument("--chunk-size", type=int, default=250, help="samples per encoding / work unit")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="worker processes")
    parser.add_argument("--out", type=Path, default=None, help="CSV with one row per (seed, batch)")
    return parser


def resolve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    """Validate option combinations and fill in dependent defaults (shared with probe_drift.py)."""
    args.readout = list(dict.fromkeys(args.readout))
    args.batches = sorted(set(args.batches))
    if sum([args.align_channels, args.align_ema, args.align_ema_balanced]) > 1:
        parser.error("choose one of --align-channels, --align-ema and --align-ema-balanced")
    if args.align_ema_balanced and args.adapt_stdp:
        parser.error("--align-ema-balanced runs with a frozen network (not with --adapt-stdp)")
    if (args.align_ema or args.align_ema_balanced) and args.standardize_channels is False:
        parser.error("EMA alignment outputs standardised channels, so the network must be trained on them")
    if args.few_shot_samples < 0 or args.calibration_epochs < 0:
        parser.error("--few-shot-samples and --calibration-epochs must be ≥ 0")
    if args.few_shot_samples and args.align_ema_balanced:
        parser.error("--few-shot-samples runs with plain / no alignment (not --align-ema-balanced)")
    if args.few_shot_samples and args.readout[0] != "spiking":
        parser.error("--few-shot-samples fine-tunes the spiking readout: make it the first --readout")
    if args.channel_stats is None:
        args.channel_stats = "balanced" if args.align_ema_balanced else "pooled"
    if args.align_ema_balanced and args.channel_stats != "balanced":
        parser.error("--align-ema-balanced uses class-balanced statistics (--channel-stats balanced)")
    if args.align_channels and args.channel_stats != "pooled":
        parser.error("--align-channels aligns to pooled Batch 1 statistics (--channel-stats pooled)")
    if not 0 < args.ema_alpha < 1:
        parser.error("--ema-alpha must be between 0 and 1")
    if args.standardize_channels is None:
        args.standardize_channels = args.align_channels or args.align_ema or args.align_ema_balanced
    if not 0 < args.stdp_duration <= args.duration:
        parser.error("--stdp-duration must be > 0 and ≤ --duration")
    if any(b not in DRIFT_BATCHES for b in args.batches):
        parser.error(f"--batches must be within {DRIFT_BATCHES}")
    overrides = {}
    for item in args.few_shot_override:
        gas, _, n = item.partition("=")
        if gas not in GAS_ORDER or not n.isdigit():
            parser.error(f"--few-shot-override expects GAS=M with GAS in {GAS_ORDER}, got {item!r}")
        overrides[gas] = int(n)
    args.few_shot_override = overrides
    if overrides and not args.few_shot_samples:
        parser.error("--few-shot-override needs --few-shot-samples")
    if (args.calibration_lr_scale != 1.0 or args.rehearsal_samples) and not args.few_shot_samples:
        parser.error("--calibration-lr-scale and --rehearsal-samples apply to --few-shot-samples calibration")
    if args.safe_split and not args.few_shot_samples:
        parser.error("--safe-split applies to --few-shot-samples calibration")
    if args.save_predictions and not args.few_shot_samples:
        parser.error("--save-predictions is available with --few-shot-samples")
    if args.calibration_lr_scale <= 0 or args.rehearsal_samples < 0:
        parser.error("--calibration-lr-scale must be > 0 and --rehearsal-samples ≥ 0")
    return args


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = resolve_args(parser, parser.parse_args(argv))

    drift_data = {b: load_batch(b) for b in args.batches}
    months = {1: load_batch(1).months, **{b: d.months for b, d in drift_data.items()}}
    scaling = ("per-sample scaling" if args.normalization == "sample"
               else f"{args.normalization} scaling (bounds fitted on Batch 1's training split)")
    channels = {(False, False): "no channel correction",
                (False, True): "channels standardised with Batch 1 statistics",
                (True, False): "channels aligned to Batch 1 moments (raw scale)",
                (True, True): "channels aligned to Batch 1 moments, then standardised"}[
        (args.align_channels, args.standardize_channels)]
    if args.align_ema:
        channels = f"channels standardised online with EMA statistics (α = {args.ema_alpha:g})"
    if args.align_ema_balanced:
        channels = (f"channels standardised online with class-balanced EMA statistics, conditioned on the frozen "
                    f"readout's predictions (α = {args.ema_alpha:g})")
    if args.standardize_channels and args.channel_stats == "balanced":
        channels += ", Batch 1 standardised with class-balanced statistics"
    scaling = f"{channels}, {scaling}, LIF input threshold {args.lif_threshold:g} mV above rest"
    mode = (f"STDP ADAPTING, unsupervised, learning rate × {args.adapt_lr_scale:g} "
            f"({'readout recalibrated with few-shot labels' if args.few_shot_samples else 'readout frozen'})"
            if args.adapt_stdp else "network FROZEN" if not args.few_shot_samples else "STDP FROZEN")
    print(f"Drift benchmark: train on Batch 1 (months 1–2), test on batches {args.batches} — {mode}\n"
          f"{args.feature_mode} features, {scaling}, {args.neurons} STDP neurons, {args.epochs:g} epoch(s) × "
          f"{args.stdp_duration:g} s presentations, readouts {' '.join(args.readout)} (class weight "
          f"{args.readout_class_weight}, power {args.readout_class_weight_power:g}); seeds {args.seeds}", flush=True)

    per_seed: dict[int, dict[int, dict]] = {}
    t0 = time.time()
    if args.few_shot_samples:  # the readout changes from batch to batch: one stream per seed, seeds in parallel
        print(f"Few-shot calibration: {args.few_shot_samples} labelled samples per gas per drift batch"
              + "".join(f" ({n} for {g})" for g, n in args.few_shot_override.items()) + ", "
              f"{args.calibration_epochs} readout fine-tuning epochs at learning rate × {args.calibration_lr_scale:g}"
              f"{f', plus {args.rehearsal_samples} Batch 1 samples per gas replayed' if args.rehearsal_samples else ''}"
              f"; scored on the remaining samples", flush=True)
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.seeds))) as pool:
            futures = {pool.submit(run_seed_fewshot, args, s): s for s in args.seeds}
            for f in as_completed(futures):
                per_seed[futures[f]] = f.result()
                print(f"  seed {futures[f]}: done [{time.time() - t0:.0f} s]", flush=True)
    elif args.align_ema_balanced:  # sequential by construction: one stream per seed, seeds in parallel
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.seeds))) as pool:
            futures = {pool.submit(run_seed_balanced, args, s): s for s in args.seeds}
            for f in as_completed(futures):
                per_seed[futures[f]] = f.result()
                print(f"  seed {futures[f]}: done [{time.time() - t0:.0f} s]", flush=True)
    elif args.adapt_stdp:  # sequential by nature: one stream per seed, seeds in parallel
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.seeds))) as pool:
            futures = {pool.submit(run_seed_stream, args, s, True): s for s in args.seeds}
            for f in as_completed(futures):
                per_seed[futures[f]] = f.result()
                print(f"  seed {futures[f]}: done [{time.time() - t0:.0f} s]", flush=True)
    else:
        batch_stats = {b: d.channel_moments(args.feature_mode) for b, d in drift_data.items()}  # all samples, no labels
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for seed in args.seeds:
                net, heldout = train_on_batch1(args, seed)
                affines = (ema_affines(args, net.ema_init, seed, args.batches) if args.align_ema
                           else {b: drift_affine(args, net, batch_stats[b]) for b in drift_data})
                futures = {b: [pool.submit(classify_chunk, net, b, start, min(start + args.chunk_size, len(d)), seed,
                                           affine_rows(affines[b], start, min(start + args.chunk_size, len(d))))
                               for start in range(0, len(d), args.chunk_size)] for b, d in drift_data.items()}
                per_seed[seed] = {1: summarize_batch([heldout], args.readout)}
                per_seed[seed].update({b: summarize_batch([f.result() for f in fs], args.readout)
                                       for b, fs in futures.items()})
                print(f"  seed {seed}: done [{time.time() - t0:.0f} s]", flush=True)

    print_report(per_seed, months, args)
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_predictions, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "batch", "index", "gas", "prediction", "confidence"])
            for s, batches in sorted(per_seed.items()):
                for b, r in sorted(batches.items()):
                    writer.writerows((s, b, *row) for row in r.get("samples", []))
        print(f"wrote {args.save_predictions}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "channels", "few_shot", "calibration_samples", "seed", "batch", "months", "samples"]
                            + [f"{n}_accuracy" for n in args.readout]
                            + ["silent", "active_neurons", "weight_change", "reference_lr_accuracy"]
                            + [f"{g}_{k}" for g in GAS_ORDER for k in ("correct", "total")])
            for s, batches in sorted(per_seed.items()):
                for b, r in sorted(batches.items()):
                    writer.writerow(["adapt" if args.adapt_stdp else "frozen", channels,
                                     f"{args.few_shot_samples}"
                                     + "".join(f"/{g} {n}" for g, n in args.few_shot_override.items())
                                     + f"x{args.calibration_epochs} lr×{args.calibration_lr_scale:g} "
                                     f"replay {args.rehearsal_samples}", r.get("calibration", 0),
                                     s, b, months[b], r["n"]]
                                    + [r["accuracy"][n] for n in args.readout]
                                    + [r["silent"], r["active"], r.get("weight_change", ""), r["reference"]]
                                    + [v for g in GAS_ORDER for v in r["recall"][g]])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
