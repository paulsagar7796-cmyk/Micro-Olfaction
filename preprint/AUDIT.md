# Independent audit of the preprint

This guide is for someone other than the author who checks the preprint's results. The check has three
parts:

- **A.** The results regenerate from the public datasets with the archived code.
- **B.** The manuscript's key numbers follow from those results. You recompute them with your own
  spreadsheet or code.
- **C.** The manuscript's claims are supported and stated with their limits.

At the end you write a short report (part D). With your permission, the author archives your report
and reproduction log with the code, and may name you in the manuscript.

Please record what you actually did. A partial audit, reported as partial, is useful. So is a failed
reproduction.

**Names.** The manuscript's SOLAR and SOLAR+ are called **System 2** and **System 2+** in the code and
result files. SOLAR+ is `config = "K=3 + class weight + pseudo-labels"` in
`results/system2_combined.csv`.

## What you need

- **The material:** the review bundle `paper-v3-review-<date>.zip` from the author, unzipped, or later
  the Zenodo software archive. The draft manuscript is in `preprint/dist/` inside the bundle.
- **Python 3.13.2** and the package versions pinned in `requirements.txt`.
  - On macOS with Apple silicon, the archived result files are expected to regenerate byte for byte.
  - On other platforms, stochastic-gradient training can change single results slightly. Record your
    platform. Part B then tells you whether the differences matter.
- **Time:** about 1.5 hours on a 10-core machine for the drift-dataset results, about 20 minutes for the
  microcontroller (C-port) steps, and about 1 minute for the twin-array test.
- **Internet access**, to download the two UCI datasets, which are not redistributed. The twin-array
  archive is 204 MB.
- **Optional:** the xPack Arm GCC and QEMU unpacked in `~/opt` (see `README.md`, "Cortex-M4 builds").
  Without them the C-port steps are skipped, and their committed results are checked as they are.

## Part A: regenerate the results

```bash
cd paper-v3-review-<date>                   # the unzipped folder
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
preprint/audit_reproduction.sh              # everything; "... costs table validation" for a 2-minute trial
```

The script does three things:
1. It snapshots the folder as unpacked in a local git commit.
2. It runs the reproduction in two parts:
   - `./reproduce.sh`, the drift-dataset results: this downloads and checksums the dataset, reruns every
     experiment, compares every result file with the archived one, and checks every number of the results
     summary `paper/comparison_tables.md`;
   - `validation/reproduce_validation.sh`, the twin-array test, which does the same with
     `validation/RESULTS_S2COMPOUNDS.md`.
3. It passes only if **every** result file was compared and none differed. It prints a summary for your
   report and keeps the full log in `preprint/audit/`.

**Why the snapshot matters:** `check_reproduction.py` compares each file with its version in git. In a
folder without git history, it compares nothing, and it still ends with "all regenerated results match
the committed ones". Always use the audit script, or commit the unpacked folder yourself before running
`./reproduce.sh`.

**What a pass looks like:**
- `PASS`;
- paper-v3 results: 45 of 45 files compared, 0 differing and 0 not compared;
- `155/155 checks of paper/comparison_tables.md passed`;
- twin-array results: 8 of 8 compared;
- `22/22 checks of validation/RESULTS_S2COMPOUNDS.md passed`;
- the C-port line says whether those steps ran on your machine.

**Where the twin-array code comes from:** it was copied from a later stage of the project, with its
changes listed in `validation/README.md`. Please check that list as part of the audit.

## Part B: recompute the key numbers yourself

Use your own spreadsheet or code, not the project's scripts. All files are CSV.

Conventions:
- "Mean over seeds" means: for each seed, average over the named batches first, then average those
  per-seed values.
- The ± value is the standard deviation of the per-seed values, with n − 1.
- Accuracies are fractions; multiply by 100.
- Batches 6–10 means `target` 6 to 10.
- In `literature/published_results.csv`, the `setting` column holds text: `"1"`, `"2"` and others.

| # | Quantity (manuscript location) | Inputs and computation | Manuscript value |
|---|---|---|---|
| B1 | SOLAR+, Setting 1, 20 balanced labels (Table 3) | `system2_combined.csv`: SOLAR+, `setting` 1, `budget` 20, `selection` "balanced", `target` 6–10; mean over seeds ± SD | 80.2 ± 2.5 |
| B2 | The same, Setting 2 | as B1 with `setting` 2 | 83.3 ± 1.8 |
| B3 | DA-ELM at 20 labels (Table 3) | `published_results.csv`: `method` "DAELM-S(20)", `setting` "1" and "2"; mean of `batch_6` … `batch_10`. If you can, compare these values with Tables II–III of Zhang & Zhang (2015) | 75.9 and 79.0 |
| B4 | Headline advantage (abstract, 6.1) | B1 − B3 and B2 − B3, from values rounded to one decimal | +4.3 and +4.3 |
| B5 | 50 labels (Table 3, 6.1) | as B1 and B2 with `budget` 50; DA-ELM as B3 with "DAELM-T(50)"; then the differences | 83.4 ± 1.4 and 88.7 ± 3.1 against 87.0; −3.6 and +1.7 |
| B6 | With DA-ELM's own label selection, SOLAR+ trails (6.1) | as B1 and B2 with `selection` "ssa" at `budget` 20, 30 and 50, against "DAELM-S(20)", "DAELM-S(30)" and "DAELM-T(50)" | all six differences negative, 0.9 to 7.2 points |
| B7 | Zero labels (Tables 3 and 5) | as B1 and B2 with `budget` 0, `selection` "none"; TDACNN as B3 with `method` "TDACNN" | 56.4 ± 2.4 and 65.4 ± 4.1; TDACNN 64.7 and 72.7 |
| B8 | Concentration-level hold-out (Table 6) | `results/temporal_confound.csv`: `setting` 1, `budget` 20, `target` 6–10; for each `condition`, mean over seeds. Change = held-out levels − usual draw. Share kept = (held-out levels − no labels) / (usual draw − no labels). Compute both from unrounded means. Repeat for the other three rows | 63.0, 83.3, 72.5; −10.7; 47 % |
| B9 | File order (section 7) | `results/temporal_confound_file_order.csv`: `index_5nn_accuracy` for batches 1–6 and 9, and for 7, 8 and 10 with `permutation_null_mean` | 89.6–99.6 % (shown as "90–99.6 %"); 35.9, 57.5, 15.0 % against 17.7, 38.7, 16.8 % |
| B10 | Sequential protocol (6.5) | `results/sequential_system2plus.csv`: `condition` "System 2+", `batch` 6–10, `budget` 5, 10, 12, 15; mean over the 15 seeds. Then the same for "System 2" | 88.6, 91.5, 92.7, 93.7; SOLAR: 90.0, 91.3, 92.3 at 10, 12, 15 |
| B11 | Pseudo-labels that are correct (6.5) | the same file, `condition` "System 2+": for each (`budget`, `batch`), sum of `pseudo_correct` ÷ sum of `pseudo_updates` over seeds; take the range | 84.0–99.7 % (shown as "84–100 %") |
| B12 | Cycles per inference (Table 9) | `results/cm4_energy.csv`: `cycles` of "inference, per sample", `variant` "opt" (the file also holds energy figures, which this preprint does not use) | 27,666 cycles |
| B13 | DA-ELM's cycles relative to SOLAR+ (section 9) | the same file: `cycles` of "DA-ELM inference, per sample" ÷ (`cycles` of "inference, per sample" + `cycles` of "pseudo-label confidence test, per sample"), for `variant` "base" and "opt" | 30.9× and 32.8× |
| B14 | Memory (section 9) | `results/cm4_memory.csv`: `state_bytes`, `readout_bytes` | 59,780 and 6,232 bytes |
| B15 | Dataset (Table 1) | `data/targets.csv`, downloaded in part A: number of rows, and counts by `batch` and `gas_id` | 13,910; Table 1 |
| B16 | Twin arrays, all boards (Table 7) | `results/system2_compounds.csv`: `accuracy_mean` and `accuracy_sd` for `method` "System 2+", "System 2" and "random forest" | 0.954 ± 0.047; 0.872; 0.989 |
| B17 | Twin arrays, per board (Table 8) | `results/system2_compounds_folds.csv`: `accuracy` by `target_board` for "System 2+" and "System 2"; the differences, their median and their mean | board 5: 0.900 against 0.584; median gain 2.6 points; mean 8.1 |
| B18 | The registered test over five boards (section 8) | a two-sided Wilcoxon signed-rank test on the five board pairs of B17. With all five differences positive, the exact p is 2/2⁵ | p = 0.0625, the smallest possible with 5 boards |

**Judging differences.** On a platform that reproduced byte for byte, every value should match as
displayed.
- **Accuracy numbers:** otherwise, compare any difference with the ± spread across seeds. The
  manuscript's comparisons depend on differences of a few points, so note any value that moves by more
  than about 0.5.
- **Cycle and memory numbers:** these do not depend on the platform.

## Part C: check the claims

Read the manuscript: `preprint/dist/drift-compensation-preprint-v1.pdf` in the bundle. For each point
below, note whether the text supports it and states its limits.

- **Where the advantage holds:** the +4.3-point advantage holds only with balanced labels and at 20
  labels. With DA-ELM's own label selection, SOLAR+ trails.
- **Zero labels:** without target labels, TDACNN is better.
- **Memory:** the hardware claim is about memory. "Adapts on the chip" means 59,780 bytes of state,
  against 128 KB of SRAM on the STM32L476 and 384 KB on the Apollo3 Blue. DA-ELM's adaptation needs
  at least 4 MB.
- **Cycles:** cycle counts come from an emulated chip and the Cortex-M4 timing tables. They are modelled,
  not measured.
- **Energy:** the manuscript makes no energy claims; they are left to the next paper.
- **Few-shot accuracy:** it is optimistic by an estimated 7–11 points (Table 6: changes of −6.6 to
  −10.7).
- **The twin-array test:** its limits are stated:
  - the class weight was not used;
  - the order within a board is declared, not a real time stream;
  - the board is the unit, with N = 5;
  - the registered test cannot be met with five boards;
  - the protocol timeline is disclosed;
  - a random forest does better.
- **Method against code:** the method section matches the code. See `TUNED` in
  `sweep_system2_budget.py`, and `PSEUDO`, `K_VALUES` and `CLASS_WEIGHT` in `combine_system2.py`. For
  the twin arrays, see `system2plus_core.py`.
- **The published comparison figures:** these are transcriptions. If you have access, spot-check a few
  against the cited papers (`literature/published_results.csv` names each table).
- **Anything else:** note anything overstated, unclear or missing.

Also run `python3 preprint/check_manuscript.py`. It ties the manuscript's numbers to the two results
summaries and should end "N/N checks passed". It was written with AI assistance. Treat it as a
convenience; your own recomputation in part B is the independent check.

## Part D: report

1. Copy `preprint/audit/REPORT-TEMPLATE.md` to `preprint/audit/REPORT-<your name>.md` and fill it in.
   State your relationship to the author, and whether you contributed to the work before this audit;
   "independent" is only accurate if you did not.
2. Send the report, with the log from `preprint/audit/`, to the author.
