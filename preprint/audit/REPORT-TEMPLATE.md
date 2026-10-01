# Audit report: preprint

Follow `preprint/AUDIT.md`. Record what you actually did, including anything you skipped or that
failed.

## Auditor

- **Name:**
- **Affiliation or role:**
- **Relationship to the author:** (for example: none, friend, colleague, co-author)
- **Contribution to the work before this audit:** (none, or describe it)
- **Dates of the audit:**
- **Material audited:** (file name of the review bundle or Zenodo archive, and its SHA-256)

## A. Reproduction

- **Machine:** (operating system, CPU)
- **Python version:**
- **Environment matched `requirements.txt`:** (yes / no; if no, what differed)
- **C-port steps:** (ran / skipped because the Arm tools were not installed)
- **Summary printed by `preprint/audit_reproduction.sh`**, both parts (paste it here):

```
```

- **Log file name and SHA-256:**
- **Outcome:** (PASS / FAIL / partial run), and notes

## B. Recomputation with my own spreadsheet or code

**Tool used:**

| # | Quantity | Manuscript value | My value | Match? | Notes |
|---|---|---|---|---|---|
| B1 | SOLAR+, Setting 1, 20 balanced labels | 80.2 ± 2.5 | | | |
| B2 | SOLAR+, Setting 2, 20 balanced labels | 83.3 ± 1.8 | | | |
| B3 | DA-ELM (DAELM-S(20)), Settings 1 and 2 | 75.9; 79.0 | | | |
| B4 | Headline advantage | +4.3; +4.3 | | | |
| B5 | 50 labels, and differences to DAELM-T(50) | 83.4 ± 1.4; 88.7 ± 3.1; 87.0; −3.6; +1.7 | | | |
| B6 | DA-ELM's own selection: SOLAR+ trails | all negative, 0.9–7.2 | | | |
| B7 | Zero labels; TDACNN | 56.4 ± 2.4; 65.4 ± 4.1; 64.7; 72.7 | | | |
| B8 | Hold-out (Table 6), all four rows | as in Table 6 | | | |
| B9 | File order | 89.6–99.6 %; 35.9, 57.5, 15.0 % vs 17.7, 38.7, 16.8 % | | | |
| B10 | Sequential protocol | 88.6, 91.5, 92.7, 93.7; 90.0, 91.3, 92.3 | | | |
| B11 | Pseudo-labels correct | 84.0–99.7 % | | | |
| B12 | Cycles per inference | 27,666 cycles | | | |
| B13 | DA-ELM's cycles relative to SOLAR+ | 30.9×; 32.8× | | | |
| B14 | Memory | 59,780; 6,232 bytes | | | |
| B15 | Dataset | 13,910 rows; Table 1 | | | |
| B16 | Twin arrays, all boards | 0.954 ± 0.047; 0.872; 0.989 | | | |
| B17 | Twin arrays, per board | board 5: 0.900 vs 0.584; median 2.6, mean 8.1 points | | | |
| B18 | Wilcoxon over five boards | p = 0.0625 | | | |

## C. Claims

For each claim, mark it as supported, supported but needing clearer limits, or not supported, and add
notes.

- [ ] The +4.3-point advantage is stated with its conditions: balanced labels, 20 labels.
- [ ] Losses are reported: DA-ELM's selection, zero labels (TDACNN), and the multi-source run
  (AMDS-PFFA).
- [ ] Cycle counts are presented as modelled, not measured; there are no energy claims.
- [ ] The memory claim matches 59,780 bytes against the chips' SRAM, and DA-ELM's at least 4 MB.
- [ ] The twin-array test's limits are stated: no class weight, declared order, N = 5 boards, the unattainable registered test, the disclosed timeline, and a random forest that does better.
- [ ] The changes listed in `validation/README.md` match the copied code.
- [ ] The optimism of few-shot accuracy (7–11 points) is stated.
- [ ] The method section matches the code.
- [ ] Spot-checked published figures against their sources (which ones):
- **`python3 preprint/check_manuscript.py` result:**
- **Other issues:**

## D. Conclusion

- **In my own words, what I verified and what I did not:**
- **May the author name me in the manuscript, with this report and log archived with the code?**
  (yes / no)
- **Signature and date:**
