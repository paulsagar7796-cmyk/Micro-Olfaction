# Comparison tables — UCI Gas Sensor Array Drift dataset

Generated from `benchmark_literature_comparison.py`, `improve_system2.py`, `combine_system2.py`,
`sweep_system2_budget.py`, `sequential_system2plus.py`, `tune_system2.py`, `literature_table.py`, `deep_model_cost.py`,
`check_temporal_confound.py`, `presence_test.py` and `energy_proxy.py` (results in `results/`).
`./reproduce.sh` reruns all of them and checks every number below against the results
(`check_paper_numbers.py`). Published numbers, transcribed in `literature/published_results.csv`:
- Zhang & Zhang, *Domain Adaptation Extreme Learning Machines for Drift Compensation in E-nose Systems*,
  IEEE Trans. Instrum. Meas. 2015 (arXiv:1505.06405), Tables II–III.
- Ma et al., *Online Sensor Drift Compensation for E-Nose Systems Using Domain Adaptation and Extreme
  Learning Machine*, Sensors 18:742, 2018 (ODAELM), Tables 2–3.
- Zhang et al., *TDACNN: Target-domain-free Domain Adaptation Convolutional Neural Network for Drift
  Compensation in Gas Sensors*, Sens. Actuators B 361:131739, 2022 (arXiv:2110.07509), Tables 4–5.
- Zhang et al., *Unsupervised Attention-Based Multi-Source Domain Adaptation Framework for Drift
  Compensation in Electronic Nose Systems*, arXiv:2409.13167, 2024 (AMDS-PFFA), Table III.

Our numbers: mean over 5 seeds (± = standard deviation across seeds); accuracy is the mean of per-batch
accuracies, as in Zhang & Zhang. All method choices were made on Batches 2–5; Batches 6–10 are held out.
Differences between methods are taken between the values as displayed (one decimal).

## Headline claim

> "With 20 calibration samples per batch balanced across the six gases, the method exceeds DA-ELM by
> 4.3 points in both standard settings, and at 50 samples it comes within 1–4 points (and exceeds it in
> Setting 2), while running online at ≈10 nJ per inference, ~40× less than DA-ELM."

Scope: held-out Batches 6–10 (months 17–36), where DA-ELM's figures are averaged from its per-batch
tables over the same batches; "DA-ELM" = DAELM-S(20) at 20 samples and DAELM-T(50) at 50; energy is a
first-order count of multiply-accumulates at 3.1 pJ each, not a measurement (section 5). The method
("System 2+", section 2) is online and causal within each batch. No newer method in section 3 exceeds
DA-ELM under matched label budgets. DA-ELM's labels are also drawn from the whole target batch, so the
comparison is like-for-like in where labels come from, but absolute few-shot accuracies are optimistic
(section 4).

## Hardware claim

> "System 2+ is the only method compared whose adaptation runs online within microcontroller memory
> (≈60 KB of state, ≈10 nJ per sample including updates)."

Scope: the methods compared are DA-ELM (S and T), ODAELM, TDACNN and AMDS-PFFA. The claim concerns
adaptation, not inference: the deep models fit a microcontroller for inference alone (TDACNN ≈276 KB
with int8 weights). ODAELM also adapts online, but its update keeps a 1,000 × 1,000 inverse matrix
(4.0 MB in FP32, ≈6 µJ per update) at its stated configuration of up to 1,000 hidden nodes. An ODAELM
with ≤250 nodes would fit in 250 KB, but its accuracy at that size is not reported. The ≈60 KB covers the
model, running statistics and up to 50 stored labelled samples. All figures are first-order estimates
(section 5).

## Where the method loses

With zero target labels, **TDACNN holds the state of the art for strict 0-label drift compensation**,
meaning no target labels and no target data. On Batches 6–10 it scores 64.7 % (Setting 1) and 72.7 %
(Setting 2), against 56.4 % and 65.4 % for System 2+ at 0 labels, even though System 2+ also adapts to
the unlabelled target stream. AMDS-PFFA reaches 81.3 % with 0 target labels, but it needs two labelled
source batches and trains on the whole unlabelled target batch before predicting. System 2+'s advantage
starts once a few labelled samples per gas are available (sections 2–3).

## 1. Gas naming

| Class | Name used here (count-table naming) | UCI coding sentence |
|---|---|---|
| 1 | acetone | ethanol |
| 2 | acetaldehyde | ethylene |
| 3 | ethanol | ammonia |
| 4 | ethylene | acetaldehyde |
| 5 | ammonia | acetone |
| 6 | toluene | toluene |

UCI's documentation contradicts itself. Its coding sentence gives the right-hand column. Its own per-batch
count table, tables derived from Vergara et al. (2012), Zhang & Zhang (2015, Table I) and later papers all
tie the gases to sample counts that give the middle column. The coding sentence is identical to that
table's column-header order, which suggests it was read off the headers. The documented concentration
ranges fit the coding sentence somewhat better, and a response-strength check is inconclusive, so the
chemical identity cannot be proven from public material. We use the count-table naming, as the published
results do, and also give class numbers. Only the dataset's authors could settle it definitively.

## 2. Accuracy

**Method.** System 2 = running per-channel mean/variance alignment (EMA, α = 0.995) of 256 features (the
128 raw features plus log-pattern features per feature type), a linear SGD logistic readout trained on the
source batch, and calibration on the labelled target samples (50 passes). **System 2+** adds, all chosen
on Batches 2–5: strong calibration (labelled target samples' loss × 3), class 4 (ethylene) weighted × 2,
and streaming pseudo-labels (an SGD step of weight 0.1 on the predicted label when its probability is
≥ 0.95, at most 50 per gas per batch; each sample is predicted before any update it triggers).

**Protocols.** Setting 1: train on Batch 1, test each Batch K independently. Setting 2: train on Batch
K−1, test on Batch K. Labels = labelled target samples per batch, never scored. "Balanced" = equal labels
per gas (a maintenance visit with a reference cylinder per gas); "DAELM selection" = our
reimplementation of the source's gas-agnostic farthest-point selection.

**Test accuracy, Batches 6–10 (%)** — balanced / DAELM selection

| Protocol | Method | 0 labels | 20 labels | 30 labels | 50 labels |
|---|---|---|---|---|---|
| Setting 1 | Published: SVM-rbf (0) / DAELM-S(20) / DAELM-S(30) / DAELM-T(50) | 29.2 | 75.9 | 83.6 | 87.0 |
| | System 2 | 57.0 ± 1.8 | 75.4 / 69.6 | 78.1 / 73.0 | 80.1 / 82.0 |
| | **System 2+** | 56.4 ± 2.4 | **80.2 ± 2.5** / 71.8 ± 1.2 | 82.1 ± 3.1 / 76.4 ± 2.8 | 83.4 ± 1.4 / 85.2 ± 1.9 |
| Setting 2 | Published: SVM-rbf (0) / DAELM-S(20) / DAELM-S(30) / DAELM-T(50) | 60.7 | 79.0 | 84.9 | 87.0 |
| | System 2 | 65.0 ± 2.1 | 79.8 / 72.6 | 81.8 / 80.3 | 84.4 / 82.0 |
| | **System 2+** | 65.4 ± 4.1 | **83.3 ± 1.8** / 73.8 ± 0.8 | **86.8 ± 1.8** / 82.2 ± 1.8 | **88.7 ± 3.1** / 86.1 ± 1.9 |

System 2+ vs DA-ELM with balanced labels: +4.3 (Setting 1) and +4.3 (Setting 2) at 20 labels; −1.5 and
+1.9 at 30; −3.6 and +1.7 at 50 (the Setting 2 result exceeds 87.0 on 3 of 5 seeds). With DAELM's
gas-agnostic selection, System 2+ trails at every labelled budget (by 0.9–7.2 points), so the advantage
depends on balanced calibration. System 2+ vs System 2: +2.9 points (mean over settings, budgets and selections;
better on 5/5 seeds); class 4 (ethylene) recall 41.1 % → 52.6 %; macro recall 74.9 % → 77.9 %.

**Ablation, balanced labels (%), Batches 6–10** — 20 / 50 labels

| Components | Setting 1 | Setting 2 |
|---|---|---|
| System 2 | 75.4 / 80.1 | 79.8 / 84.4 |
| + class weight + pseudo-labels | 76.9 / 81.1 | 81.8 / 85.3 |
| + strong calibration (K = 3) | 79.1 / 81.9 | 83.8 / 87.7 |
| + strong calibration + pseudo-labels | 80.1 / 82.9 | 84.1 / 88.5 |
| + strong calibration + class weight | 79.2 / 82.2 | 83.1 / 87.9 |
| System 2+ (all three) | 80.2 / 83.4 | 83.3 / 88.7 |

Strong calibration is the main contributor. Larger target weights hurt: validation accuracy falls from
87.8 % at K = 3 to 82.9 % at K = 300 (all three components on). Also tested and rejected on validation: a
frozen random non-linear layer (64–1024 units, ReLU or tanh; best 84.9 % vs 85.7 % without, and 6–97×
the inference cost).

Other published rows (source tables, Batches 2–10 averages): Setting 1 — SVM-rbf 38.9, CC-PCA 45.7,
SVM-comgfk 64.0, ML-comgfk 67.3; Setting 2 — SVM-rbf 68.0, SVM-comgfk 69.4, ML-comgfk 79.6, Vergara
ensemble 80.0 (label use not stated for most). Batches 2–10 averages for System 2 are in
`results/literature_comparison.csv`; they include the tuning batches.

**Sequential maintenance protocol.** The model, its alignment statistics and its pseudo-label updates
carry over from batch to batch, with recalibration at each visit. Batches 6–10 (15 seeds): System 2+
reaches 91.5 / 92.7 / 93.7 % at 10 / 12 / 15 labels per gas (System 2: 90.0 / 91.3 / 92.3 %), and
88.6 % at 5. Pseudo-labelling shows no confirmation bias over nine maintenance cycles: 84–100 % of
pseudo-labels are correct in each batch, and their effect (+0.3 to +0.6 points) does not decline across
batches. Pseudo-labelling was tested between regular recalibrations, not over long periods without
labels.

## 3. Post-2015 methods

**Test accuracy, Batches 6–10 (%)**, mean of per-batch accuracies, recomputed from each paper's per-batch
values; System 2+ rows run under the same protocol (`literature_table.py`).

| Protocol | Method | Target labels per batch | Uses the whole unlabelled target batch | Labelled samples scored | Batches 6–10 |
|---|---|---|---|---|---|
| Setting 1 | TDACNN (2022) | 0 | no (no target data) | — | **64.7** |
| | System 2+ | 0 | no ¹ | — | 56.4 ± 2.4 |
| | DAELM-S(20) (2015) | 20 | no | not stated | 75.9 |
| | System 2+, balanced | 20 | no ¹ | no | **80.2 ± 2.5** |
| | DAELM-T(50) (2015) | 50 | yes | not stated | **87.0** |
| | System 2+, balanced / DAELM selection | 50 | no ¹ | no | 83.4 ± 1.4 / 85.2 ± 1.9 |
| Setting 2 | TDACNN (2022) | 0 | no (no target data) | — | **72.7** |
| | System 2+ | 0 | no ¹ | — | 65.4 ± 4.1 |
| | DAELM-S(20) (2015) | 20 | no | not stated | 79.0 |
| | System 2+, balanced | 20 | no ¹ | no | **83.3 ± 1.8** |
| | ODAELM-S (2018), offline | 50, Kennard-Stone | no | yes | 79.7 |
| | ODAELM-T (2018), offline | 50, Kennard-Stone | yes | yes | 68.5 |
| | DAELM-T(50) (2015) | 50 | yes | not stated | 87.0 |
| | System 2+, balanced / DAELM selection | 50 | no ¹ | no | **88.7 ± 3.1** / 86.1 ± 1.9 |
| Setting 2, online | ODAELM-S (2018), online | random, < 1/20 of samples ² | no | yes | 84.4 |
| | ODAELM-T (2018), online | random, < 1/20 of samples ² | yes | yes | 81.4 |
| Batches 1 + 2 → K | AMDS-PFFA (2024) | 0 | yes (trains on it) | — | **81.3** |
| | System 2+ | 0 | no ¹ | — | 59.2 ± 1.3 |
| | System 2+, balanced | 20 | no ¹ | no | 74.6 ± 1.7 |

¹ System 2+ uses the unlabelled target stream causally (running alignment and pseudo-labels), one sample
at a time, never the whole batch. ² Under 65 per batch except Batches 7 and 9.

- **Zero labels:** TDACNN beats System 2+ by 8.3 (Setting 1) and 7.3 (Setting 2) points and is the
  state of the art without target data. AMDS-PFFA beats our multi-source run by 22.1 points at 0 labels
  and by 6.7 at 20 labels.
- **With labels:** no newer method exceeds DA-ELM or System 2+ at matched budgets. ODAELM's offline
  figures (Setting 2 structure; the paper calls it Setting 1) count its labelled samples as correct
  predictions, yet ODAELM-S at 50 Kennard-Stone labels (79.7) trails System 2+ with DAELM selection
  (86.1). The same paper's reproduction of DAELM-S at 50 labels gives 81.3, below Zhang & Zhang's 84.9
  at 30.
- **Online:** ODAELM-S online (84.4, labelled samples scored, fewer than 1 in 20 samples labelled) is the
  closest comparison to System 2+'s online protocol; System 2+ gets 86.8 at 30 and 88.7 at 50 balanced
  labels, with labelled samples not scored.
- **Batches 2–10 averages** as reported: TDACNN 72.2 (Setting 1) and 81.48 (Setting 2; 81.40 from its
  per-batch values); ODAELM-S offline 83.5, online 85.7; AMDS-PFFA 83.20 over Batches 3–10, weighted by
  sample count (86.6 as a plain per-batch mean).

## 4. Temporal confound and concentration-level hold-out

Dennler et al. (2021) showed, on the 2013 wind-tunnel dataset, that gases recorded in temporal clusters
let classifiers exploit the sensor state before exposure. This dataset has no timestamps, so
`check_temporal_confound.py` tests what can be tested.

**File order.** Within 7 of 10 batch files (Batches 1–6 and 9), a sample's position predicts its gas
with 90–99.6 % accuracy (5-NN on the index, 5-fold stratified cross-validation; chance 17–20 %,
label-permutation null 17–33 %). This reflects sorting, not recording time: within each same-gas run,
concentration is non-decreasing in 86–99.5 % of adjacent pairs in those batches. Batches 7, 8 and 10 are
interleaved (index accuracy 35.9 %, 57.5 % and 15.0 %, against permutation nulls of 17.7 %, 38.7 % and
16.8 %; Batch 10 alternates gas every sample). System 2+ never sees file order: every evaluation streams
each batch in a seeded random permutation, and the index is never a feature. Training and test data are
separated by batch, months apart, not by random splits within a recording session.

**Pre-exposure state.** Dennler's test needs a baseline resistance measured before each exposure. None of
five reconstructions from the provided features is independent of the current concentration (median
Spearman ρ +0.27 to +0.70), so the test cannot be run on this dataset.

**Concentration-level hold-out.** Labels can still sit close to scored samples: concentration levels are
swept in sessions, and a balanced draw can take labels at the same level as the samples it is scored on.
For each gas, half of its concentration levels (chosen at random) may provide labels, and only the other
levels are scored. Three conditions are compared on the same scored samples: labels from all levels (the
usual draw), labels only from the other levels, and no labels.

**System 2+, balanced labels, Batches 6–10 (%)**

| Setting | Labels | No labels | Usual draw | Labels only from other levels | Change | Share of gain kept |
|---|---|---|---|---|---|---|
| 1 | 20 | 63.0 ± 2.8 | 83.3 ± 2.2 | 72.5 ± 10.0 | −10.7 | 47 % |
| 1 | 50 | 63.1 ± 2.7 | 83.3 ± 1.7 | 76.7 ± 6.3 | −6.6 | 67 % |
| 2 | 20 | 62.2 ± 8.5 | 82.2 ± 2.5 | 75.3 ± 5.3 | −7.0 | 65 % |
| 2 | 50 | 62.4 ± 8.8 | 88.9 ± 2.7 | 79.5 ± 4.9 | −9.4 | 64 % |

Only gases recorded at two or more levels are scored, and the usual-draw labels are excluded, so these
figures differ from section 2. On Batches 2–10 the change is −7.2 to −12.9 points and 46–61 % of the
gain is kept. Held-out levels score below the usual draw on 19 of 20 seed × setting × budget
combinations on Batches 6–10.

Calibration from distinct concentration levels retains roughly half to two-thirds of the accuracy gain
(47–67 %), which supports the method's core validity: the calibration effect is not an artefact of
recording proximity. The absolute few-shot numbers in the standard benchmark (sections 2–3), however,
likely benefit from labels recorded at the same concentration levels as the scored samples, and
plausibly in the same sessions. Without timestamps, concentration generalisation and same-session
proximity cannot be separated. DA-ELM's published labels are also drawn from the whole target batch, so
the published comparison is like-for-like in this respect; DA-ELM was not re-run under the hold-out.

## 5. Operational trade-offs

First-order estimates from each method's definition (`deep_model_cost.py`): multiply-accumulates
(MACs) × 3.1 pJ, FP32; memory access, activations, logs and square roots excluded; 1 KB = 1,000 bytes.
Not hardware measurements. (An accounting that adds SRAM reads — 4.6 pJ per MAC plus 5 pJ per weight
read — gives ≈29 nJ for System 2 without pseudo-label updates vs ≈1.4 µJ for DA-ELM, a similar ratio.)

| | System 2+ | DAELM-S | DAELM-T |
|---|---|---|---|
| Learning setting | inductive, online (one sample at a time) | inductive, batch | transductive, batch |
| Causal (each prediction uses only earlier data) | yes | no: labelled samples are chosen from the whole target batch | no: also uses all unlabelled target data |
| Data needed to adapt | the current visit's labelled samples | stored labelled source batch + labelled target samples | base classifier + labelled target + whole unlabelled target batch |
| Label selection | first samples per gas to arrive (no buffer) | farthest-point sampling over the whole target batch (≈0.84 G MACs, ≈2.6 mJ, 1.8 MB buffer for Batch 7) | same |
| Inference per sample | ≈3,300 MACs, ≈10.2 nJ (pseudo-label updates included) | 134,128 MACs, ≈416 nJ (1,000 radial-basis neurons) | ≈416 nJ |
| Model + running state | ≈10 KB | ≈0.5 MB | ≈0.5 MB |
| Adaptation memory | 20–51 KB (20–50 labelled samples) + 6 counters | 0.2–1.8 MB stored source + 2.0–15 MB hidden outputs + 4.0 MB solve | as DAELM-S + 1.8 MB unlabelled batch + 14 MB hidden outputs |
| Adaptation per batch | 4.6–11.5 M MACs, 14–36 µJ (50 passes) | 0.9–4.5 G MACs, ≈2.8–13.8 mJ (closed-form solve) | higher than DAELM-S |
| Always-on trigger | optional spiking presence detector, ≈5–77 nW in clean air (simulator; heaters excluded) | — | — |

**All compared methods** (`deep_model_cost.py`; layer sizes from each paper's text and architecture
figures: TDACNN Fig. 6, AMDS-PFFA Fig. 2)

| Method | MACs per inference | Energy per inference | Weights, FP32 / int8 | How it adapts | Memory to adapt |
|---|---|---|---|---|---|
| System 2+ | 3,299 | ≈10 nJ (updates included) | ≈10 KB | online: calibration at each visit (14–36 µJ) and per-sample pseudo-label steps | ≈31–61 KB (model, running state, 20–50 labelled samples) |
| DAELM-S / -T | 134,128 | ≈416 nJ | 540 KB / 135 KB | closed-form solve per target batch (0.9–4.5 G MACs) | several MB (table above) |
| ODAELM-S / -T | 134,128 | ≈416 nJ | 540 KB / 135 KB | online, a Woodbury update per sample (≈2 M MACs, ≈6.2 µJ) | 4.0 MB FP32 (1,000 × 1,000 inverse) |
| TDACNN | 392,096 + 30k–241k (MMD ensemble) | 1.3–2.0 µJ | 1.1 MB / 276 KB, plus 0.7–5.8 MB of stored source embeddings (FP32) | none in Setting 1; a new source batch means offline retraining (up to 400 epochs) | — (offline) |
| AMDS-PFFA | 170k–758k | 0.5–2.3 µJ | 0.5–3.0 MB / 0.12–0.76 MB | offline training on both labelled source batches and the whole unlabelled target batch, before predicting | ≥2.7 MB of raw samples for Batch 7, plus training state |

TDACNN's test-time ensemble compares each sample with the stored embeddings of the source samples of its
predicted class (RBF kernel, four classifiers). Its cost is ≈30k MACs with Batch 1 as source and up to
≈241k with Batch 7. Its padding is not stated; 'same' is assumed. AMDS-PFFA's transformer layout,
width, heads and depth are not stated. Its range covers one encoder layer, with 16 sensor tokens 8 wide
at the low end and one 128-wide token at the high end. Deeper encoders cost more.

## 6. Earlier per-gas results under the new naming

Results reported before the naming was harmonized used the UCI coding sentence. Translation:

| Earlier name | Class | Name now |
|---|---|---|
| ethanol | 1 | acetone |
| ethylene | 2 | acetaldehyde |
| ammonia | 3 | ethanol |
| acetaldehyde | 4 | ethylene |
| acetone | 5 | ammonia |
| toluene | 6 | toluene |

So the earlier "acetaldehyde problem" is class 4 (ethylene), and the "acetaldehyde vs acetone" confusion
is class 4 vs class 5 (ethylene vs ammonia).

## 7. Caveats

- Published figures are transcribed from four sources and checked against each source's reported
  averages. Two sources' averages differ slightly from their own per-batch values (TDACNN Setting 2:
  81.48 vs 81.40; ODAELM-T online: 85.4 vs 85.1); AMDS-PFFA averages by sample count and tunes a loss
  coefficient per target batch (its Table II). Averages here are recomputed as plain per-batch means.
- Protocols differ between sources: ODAELM counts its labelled samples in accuracy; AMDS-PFFA uses two
  labelled source batches and the unlabelled target batch; Zhang & Zhang do not state whether labelled
  samples are scored (ours are excluded). No source reports a spread; ours varies by ±1–4 points across
  seeds, so the Setting 2 lead at 50 labels (+1.7) is not robust.
- The 20-label advantage holds with balanced calibration only; with the source's gas-agnostic selection
  System 2+ trails. Budgets below 20 were not compared: the sources give no numbers there.
- System 2+ was evaluated in Settings 1 and 2, in the sequential protocol (section 2) and in one
  multi-source run (Batches 1 + 2 → K), which was not tuned: it scores below Setting 1 at the same budget
  (74.6 vs 80.2 at 20 labels). In the sequential protocol, pseudo-labelling was tested with recalibration
  at every batch, not over long periods without labels.
- Few-shot accuracies are optimistic by an estimated 7–11 points on Batches 6–10 when labels cannot come
  from the scored concentration levels (section 4). Same-session proximity cannot be ruled out: the
  dataset has no timestamps, and Dennler et al.'s pre-exposure test cannot be run on it.
- Energy and memory figures are first-order operation counts, not hardware measurements. The deep
  models' costs rest on details their papers do not state (section 5) and are given as ranges.
- Seeds vary the calibration draw and training randomness, not the data. SGD is highly sensitive to
  numerical details: relabelling the gases (which only reorders classes) moved single-seed results and
  changed the validation winner between near-equivalent settings (within 0.5 points). Report curves and
  intervals rather than single thresholds.
