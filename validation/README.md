# Twin-array validation

This folder holds the independent test reported in the manuscript's section "Validation on independent
sensor boards". It runs SOLAR+ ("System 2+" in the code and result files) on the UCI twin gas sensor
arrays:
- dataset 361, [doi:10.24432/C5MW3K](https://doi.org/10.24432/C5MW3K);
- the dataset's paper: Fonollosa et al., *Sensors and Actuators B: Chemical* 236 (2016) 1044–1053.

## Where the files come from

The test was run at a later stage of this project, at tag `pre-v4`. Its files were copied into this
archive on 2026-09-30.

| File here | At `pre-v4` | Change |
|---|---|---|
| `validation/PROTOCOL_S2COMPOUNDS.md` | `system1/PROTOCOL_S2COMPOUNDS.md` | none |
| `validation/RESULTS_S2COMPOUNDS.md` | `system1/RESULTS_S2COMPOUNDS.md` | none |
| `twin_arrays.py` | same path | none |
| `results/twin_arrays_features.csv`, `results/system2_compounds*.csv` (5 files), `results/logs/twin_arrays.log`, `results/logs/system2_compounds.log` | same paths | none |
| `fetch_twin_arrays.py` | same path | Docstring only. The dataset DOI is corrected from `10.24432/C5W881` (which does not resolve) to UCI's `10.24432/C5MW3K`, and the protocol's path is updated. |
| `system2_compounds.py` | same path | Imports `system2plus_core` instead of `system2_home`; the docstring is updated to match. |
| `system2plus_core.py` | new | `system2plus`, `forest` and `scores`, with the constants they use, moved unchanged from `system2_home.py`. The home-recordings test they sat beside is not part of this preprint. |
| `validation/check_validation_numbers.py` | group C of `check_paper_v4_numbers.py` | the document's path only |
| `validation/check_validation_outputs.py`, `validation/reproduce_validation.sh` | new | |

With the moved code, all eight result files regenerate byte for byte; this was checked on 2026-09-30
against `pre-v4`.

The two documents are unchanged copies, which has three consequences:
- they still refer to the folder `system1/`;
- they link to the home-recordings documents (`PROTOCOL_S2HOME.md`, `RESULTS_S2HOME.md`), which are not
  in this archive;
- they call the method System 2+, where the manuscript calls it SOLAR+.

Some of their sentences also compare with the paper's energy figures. Those figures are not part of this
preprint.

## Not recorded in the protocol

The protocol's test structure says that `combine_system2.adapt_target` is called unchanged. The code does
something slightly different:
- It uses `system2plus`, which runs the same pipeline: standardised source training, running alignment,
  50 calibration passes at weight 3, then pseudo-labels.
- It does **not** apply the class weight. On the drift dataset, SOLAR+ weights class 4 (ethylene) × 2;
  that weight is not used here, although ethylene is one of this dataset's classes too.

The results are what that code produces, and the manuscript describes the test that way.

## Running it

From a git checkout:

```bash
validation/reproduce_validation.sh     # about 1 minute on 10 cores, after a 204 MB download
```

The script downloads the dataset and verifies its checksum, retrying up to three times if the download
breaks off. It then regenerates the features and the test, and does two checks:
- it compares all eight files with their committed versions;
- it checks every number of `validation/RESULTS_S2COMPOUNDS.md`.

A pass ends with "all regenerated validation results match the committed ones" and
"22/22 checks of validation/RESULTS_S2COMPOUNDS.md passed".

From a downloaded copy, which has no git history, use `preprint/audit_reproduction.sh`. It first
records the copy in git, then runs the paper-v3 steps, then this script.
