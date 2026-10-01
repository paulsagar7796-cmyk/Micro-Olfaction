# Preprint

This folder holds the source of the preprint on Zenodo. The manuscript restates results from two
documents and adds no new ones:
- `paper/comparison_tables.md` at tag `paper-v3`;
- the twin-array test in `validation/`, copied from tag `pre-v4` (see `validation/README.md`).

It calls the methods SOLAR and SOLAR+; the code and result files call them System 2 and System 2+.

| File | Purpose |
|---|---|
| `manuscript.html` | the manuscript template: text, figure and references |
| `metadata.json` | authors, title, DOIs and statements, filled into the manuscript and into `../CITATION.cff` and `../LICENSE` |
| `edits.json` | every declared difference between a manuscript table and its source table: the SOLAR rename, and the energy columns and cells left out |
| `build.py` | builds `dist/manuscript.html` and the PDF (headless Chrome); with `--archive <tag>`, the code archive |
| `check_manuscript.py` | checks the built manuscript against its source documents and the code; `--mutate` shows the checks catch changes |
| `ZENODO.md` | step-by-step publication on Zenodo |
| `AUDIT.md` | guide for an independent audit: regenerate the results, recompute the key numbers, check the claims, report |
| `audit_reproduction.sh` | reproduction from a downloaded copy; fails unless every result file was compared and matched |
| `audit/` | the auditor's report form, and the reports and logs of completed audits |

```bash
python3 preprint/build.py && python3 preprint/check_manuscript.py
python3 preprint/build.py --review-bundle     # zip for independent reviewers: paper-v3, the validation, these files, the draft PDF
```

**Reproducing from a downloaded copy:** use `preprint/audit_reproduction.sh`, not `./reproduce.sh`
alone. `reproduce.sh` compares each regenerated file with its version in git. Without git history,
`check_reproduction.py` compares nothing, yet still reports that all results match. The audit script
first snapshots the unpacked folder in git and passes only if every file was compared.

`build.py` generates every results table directly from its source document, with only the changes
declared in `edits.json`, and generates the dataset table from `fetch_data.py`. `check_manuscript.py`
then checks:

- **Source documents:** each is identical to its tagged version.
- **Tables:** they are verified cell by cell, after only the declared edits; a cut may only delete text.
- **Key claims:** each is tied to its source passages, most of them word for word.
- **Other numbers:** each must occur in the source section its manuscript section draws on.
- **Scope:** energy figures, left to the next paper, and System 1 must not appear.
- **Statements no source makes** (dataset facts, method settings, software versions): each is
  confirmed against the code.
- **References:** every DOI is one verified by hand.

One thing is left to reading: whether a sentence pairs the right number with the right quantity, when
both come from the same source section.

## Licences

- The manuscript (`manuscript.html` and the PDF built from it) is licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The code, including `build.py` and `check_manuscript.py`, is under the MIT License in `../LICENSE`.
- The UCI datasets are not part of this repository.
