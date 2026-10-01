# System 2+ on discrete compounds: registered test

This corrects the confound in the home-recordings test ([RESULTS_S2HOME.md](RESULTS_S2HOME.md)): wine
and banana are not fixed chemical targets, so a test against them cannot separate "System 2+ fails to
adapt" from "the target itself changed." This protocol fixes, before any classification is run, a test
against discrete, chemically stable compounds on a real (not simulated) sensor array independent of the
UCI gas-sensor-array drift dataset the paper's results are on. It was written and committed on
2026-09-29 with the code that runs it (`twin_arrays.py`, `system2_compounds.py`). Any later change is
listed under **Deviations**, and every outcome is reported, whatever it shows.

## Data

UCI Machine Learning Repository, dataset 361, "Twin gas sensor arrays" (Fonollosa, Fernandez,
Gutierrez-Galvez, Huerta and Marco, *Sensors and Actuators B: Chemical* 236, 2016,
doi:10.1016/j.snb.2016.05.089). `fetch_twin_arrays.py` downloads it and checks it against this
checksum:

    e8801de3a98d3db2b48fccf0be4d681879a1441f5732ac6fbee1c48a05b49167  raw/uci_361.zip

Five physically independent 8-sensor Figaro arrays ("boards" 1-5; TGS2611, TGS2612, TGS2610, TGS2602,
each at two heater voltages), each tested on its own days, exposed to the same 40 conditions in random
order: **ethanol, methane, ethylene and carbon monoxide**, at 10 concentration levels each (labelled
F010-F100), 2-4 repetitions per condition, 640 trials in all. Each trial is a 600-s resistance
recording at 100 Hz, with no per-repetition day recorded in the archive.

These four compounds are single, chemically fixed substances, unlike wine and banana. The boards are
independent hardware instances tested on different days, the closest available analogue to the drift
dataset's batches.

## What was looked at before writing this

- The archive's file listing and naming (board, gas, concentration, repetition).
- The response shape of a handful of trials, chosen to span different gases, concentrations and
  boards, plotted as conductance against time. This showed a consistent pulse: flat until about 57 s,
  rising to a plateau by about 100-110 s, held until about 150-160 s, then recovering.
- That features built from this shape (below) increase monotonically with concentration level, sensor
  by sensor, for each of the four gases, in one board's data.
- Nothing else. No classification, alignment or calibration has been run.

## Feature extraction (fixed from the response shape alone)

Four windows, identical for every trial:

| Window | Time | Used for |
|---|---|---|
| baseline | 5-50 s | the pre-exposure reference |
| rising | 55-100 s | the onset transient |
| plateau | 110-150 s | the steady-state response |
| decaying | 160-400 s | the recovery transient |

Per sensor, on conductance G = 1/R (all eight sensors are n-type: conductance rises in every gas
tested):
- **DR** = mean G over the plateau window minus mean G over the baseline window;
- **DR_norm** = DR / baseline G;
- **EMAi(α)**, α ∈ {0.1, 0.01, 0.001}: the largest value of a causal exponential moving average of the
  rising window's conductance differences (y[k] = (1-α)y[k-1] + α·ΔG[k], reset to 0 at the window's
  start);
- **EMAd(α)**: the same on the decaying window's differences, taking the most negative value and
  negating it, so that a stronger recovery is also a larger positive number.

This is the general method of Vergara et al. (2012), the same paper the drift dataset's own published
features follow, applied fresh to these raw traces (their extraction code is not published, so this is
not a claim of bit-exact reproduction). Eight sensors × eight features = 64 raw features ("all"). As in
`tune_system2.raw_features`, a second block of 64 "ratio" features is added: per feature, log(max(v,
floor)) across the eight sensors, minus its mean, the response's shape with the overall amplitude
removed. 128 features in all ("all+ratio"), matching System 2+'s registered feature count on a smaller
array (the drift dataset has 16 sensors; this one has 8).

**Self-check:** the extractor is checked to give identical output when a trial is processed twice, and
`fetch_twin_arrays.py`'s trial count (640) is checked against the archive before any feature is
computed.

## Test structure: leave-one-board-out

For each of the 5 boards in turn as **target**, the other 4 boards are the **source**:
- `combine_system2.adapt_target` is called unchanged: fit on the source boards' trials (standardised),
  align the target board's trials with running mean/variance (EMA, α = 0.995), calibrate on the
  labelled trials, classify the rest, with pseudo-labelling (τ = 0.95, weight 0.1, cap 50 per class).
- **Order within the target board:** trials are sorted by (gas, concentration level, repetition). This
  is a fixed, declared order, not a real time stream (repetition-to-day mapping is not published), so
  it is stated as a limitation, not claimed as genuine temporal drift.
- **Calibration set:** 5 trials per gas (20 total), the ones nearest evenly-spaced ranks 1, 3, 5, 7 and
  9 of that gas's 10 concentration levels in the canonical order (so every calibration set spans weak
  to strong exposures). The remaining trials of the target board are scored.

**Methods, each over the 15 seeds of [PROTOCOL_S2HOME.md](PROTOCOL_S2HOME.md):**

| Method | What it does |
|---|---|
| **System 2+** | The full pipeline above. |
| System 2+ without pseudo-labels | Calibration and alignment, no streaming pseudo-labels. |
| System 2 | The source-trained model alone: no alignment, calibration or pseudo-labels. |
| Random forest | 100 trees on the raw 64 features, calibrated the same way (trained on source plus the target's calibration trials). |

## Decision rules

- **HD1 — does adaptation help?** Met if System 2+'s mean accuracy over the 5 folds and 15 seeds
  exceeds System 2's by at least 5 points, with a paired Wilcoxon test (over the 5 fold means) giving
  p < 0.05.
- **HD2 — is accuracy useful?** Met if System 2+'s mean accuracy is at least **0.80** (chance is 0.25).

**Also reported:** per-gas recall, the confusion matrix, per-board accuracy (to show whether one board
generalises worse than the others), and accuracy against calibration budget (5, 10, 20 trials per gas)
as a secondary, exploratory curve.

## Deviations

1. **2026-09-29: the full test was run once before this protocol and its code were committed.** The
   protocol text above was written before any classification was run, but the complete
   leave-one-board-out test was then executed (to a scratch file, intended as a dry run) before the
   commit. With no separate hold-out in this design, that run was the real test, and its numbers were
   seen: System 2+ accuracy 0.951, System 2 0.872, random forest 0.901; HD1 not met (+7.8 points,
   p = 0.0625), HD2 met. They are recorded here so nothing is hidden. That run used code that differed
   from this protocol in the three ways below.
2. **Code changed to match the registered text (after that run, before the commit).** No protocol text
   was changed.
   - **Order within the target board:** the code ordered trials by (gas, repetition, level); the text
     says (gas, concentration level, repetition). The code now follows the text, with the calibration
     trials first, as `combine_system2.adapt_target` aligns labelled samples first.
   - **Random forest:** the code trained it on the source boards only; the text says source plus the
     target's calibration trials. The code now follows the text.
   - **Self-checks:** the promised checks (extractor repeatability, a 640-trial count) were not
     implemented; they now are. They do not change any number.
3. **HD1 cannot be met as written.** A two-sided Wilcoxon signed-rank test over 5 paired folds has a
   smallest possible p-value of 2/2⁵ = 0.0625, so "p < 0.05" is unattainable whatever the data. The rule
   is kept as registered and reported as not met, with this explanation; it is not replaced after seeing
   results. A usable test would need more folds, or a paired test over seeds within folds, fixed in
   advance.
