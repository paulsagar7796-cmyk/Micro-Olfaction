# Publishing the paper-v3 preprint on Zenodo

This covers publishing the preprint as two linked Zenodo records.

The preprint contains paper-v3 plus the twin-array validation. It includes nothing else from the later
work: no System 1, no home recordings, and no energy figures. `build.py` refuses to build an archive
that contains any of that material.

| Record | Zenodo resource type | File | Licence |
|---|---|---|---|
| Preprint | Publication → Preprint | `preprint/dist/drift-compensation-preprint-v1.pdf` | Creative Commons Attribution 4.0 International |
| Software | Software | `preprint/dist/olfactory-simulation-paper-v3-preprint.zip` | MIT License |

Each record links to the other.
- The preprint "is supplemented by" the software.
- The software "is supplement to" the preprint.
- Both "reference" the UCI datasets.

The zip is the repository at tag `paper-v3`, unchanged, plus these additions:
- `LICENSE` and `CITATION.cff`;
- this folder's sources and checker;
- the twin-array validation: `validation/`, its code and its results.

It does not include the datasets; `fetch_data.py` and `fetch_twin_arrays.py` download them from UCI.

Publishing is effectively permanent. After publishing, the metadata can still be edited, but changing
the files needs a new version with a new DOI. Do the sandbox run in step 1 if you want to rehearse
first.

## 0. Fill in what only you can decide

Edit `preprint/metadata.json`. Every `TODO` must go:

- **Authors.** For each author, fill in:
  - given and family names;
  - affiliation, or "Independent researcher";
  - ORCID: optional but recommended, and free at orcid.org;
  - email for the corresponding author. It will be printed in the PDF.
- **`copyright_holder`**: the name in the code's MIT licence, usually yours.
- **`title`**: confirm or change it.
- **`statements.ai_assistance`**: a draft is provided, but it must describe accurately how AI tools
  were used. Many venues now expect this statement.
- **`statements.funding`** and **`statements.competing_interests`**.

Then build and check:

```bash
python3 preprint/build.py                 # the PDF, CITATION.cff and LICENSE
python3 preprint/check_manuscript.py      # every number traced to paper-v3; must end "N/N checks passed"
```

Read the PDF page by page: names, affiliation, email, statements, and the page layout. Filling in the
names shifts the page breaks slightly.

### Independent audit and your redraft (before step 2)

1. Build the bundle for your reviewers with `python3 preprint/build.py --review-bundle`. Send them
   `preprint/dist/paper-v3-review-<date>.zip`; it contains `preprint/AUDIT.md`, the draft PDF and the
   code.
2. Each auditor returns a report (`preprint/audit/REPORT-<name>.md`) and the reproduction log. Put both
   into `preprint/audit/` so they are archived with the code.
3. If you rewrite the manuscript, rerun `build.py` and `check_manuscript.py` after each section:
   - the number checks work on any wording;
   - claims whose sentences you changed are flagged, and they need re-pinning to your new wording.
4. Only then write the AI-assistance statement in `metadata.json`, describing what was actually done.
   `--final` refuses to build while it still starts with TODO.

## 1. Optional: rehearse on the sandbox

1. Create an account at <https://sandbox.zenodo.org>. It is separate from zenodo.org.
2. Do steps 2, 4 and 5 there, with a draft build.

Sandbox DOIs start with `10.5072` and do not resolve, and the sandbox may be wiped at any time. Never
put a sandbox DOI in the final PDF.

## 2. Create both drafts on zenodo.org and reserve their DOIs

1. Log in at <https://zenodo.org>. You can sign up with ORCID.
2. Click **New upload**. Under **Basic information**, answer "Do you already have a DOI?" with **No**,
   then click **Get a DOI now!**. Save the draft and note its DOI, which looks like
   `10.5281/zenodo.1234567`. Set the resource type to **Software** so you can tell the drafts apart.
3. Repeat for the second draft, with resource type **Publication → Preprint**.

Do not delete a draft: its reserved DOI is lost with it. A DOI is registered only when its record is
published.

## 3. Put the DOIs into the files, freeze, and build the archive

1. Put both DOIs into `preprint/metadata.json` as `preprint_doi` and `software_doi`.
2. Build and check in final mode. Both commands must pass:

   ```bash
   python3 preprint/build.py --final
   python3 preprint/check_manuscript.py --final --mutate
   ```

   `--final` refuses any leftover TODO or missing DOI. `--mutate` shows that each check catches an
   altered number.
3. Read the PDF once more.
4. Commit on branch `preprint-v3` and tag the commit (Claude can do this for you):

   The `git add` below includes `preprint/audit/`, so the auditors' reports and logs are archived:

   ```bash
   git add LICENSE CITATION.cff preprint/
   git commit -m "Preprint of paper-v3: manuscript, checker, licence and citation metadata"
   git tag -a paper-v3-preprint -m "paper-v3 as published on Zenodo"
   ```

5. Build the code archive from the tag:

   ```bash
   python3 preprint/build.py --final --archive paper-v3-preprint
   ```

   Before it writes `preprint/dist/olfactory-simulation-paper-v3-preprint.zip`, it checks four things:
   - the archive's files are exactly paper-v3's plus the preprint's and the validation's;
   - no paper-v3 file has changed;
   - nothing else from the later work is included;
   - no dataset files are included.

   It prints the file count and SHA-256.

## 4. Fill in and publish the software record

| Field | Value |
|---|---|
| Files | the zip from step 3 |
| Resource type | Software |
| Title | `Code and results for: <title>`, as in `preprint/dist/zenodo-software.json` |
| Publication date | today |
| Creators | each author as "Family, Given", with ORCID and affiliation |
| Description | the description in `preprint/dist/zenodo-software.json` |
| Licences | MIT License |
| Keywords | the keywords in `metadata.json` |
| Version | `paper-v3-preprint` |
| Language | English |
| Related works | **Is supplement to**: the preprint DOI (resource type Publication / Preprint)<br>**References**: `10.24432/C5MK6M`, `10.24432/C5RP6W` and `10.24432/C5MW3K` (Dataset) |
| Visibility | Public |

Click **Preview**, check everything, then click **Publish**.

## 5. Fill in and publish the preprint record

| Field | Value |
|---|---|
| Files | the PDF |
| Resource type | Publication → Preprint |
| Title | the title in `metadata.json` |
| Publication date | today |
| Creators | as for the software |
| Description | `preprint/dist/abstract.txt` |
| Licences | Creative Commons Attribution 4.0 International |
| Keywords | as for the software |
| Version | `1` |
| Language | English |
| Related works | **Is supplemented by**: the software DOI (resource type Software)<br>**References**: `10.24432/C5MK6M`, `10.24432/C5RP6W` and `10.24432/C5MW3K` (Dataset) |
| References (optional field) | the reference list from the PDF, one entry per line |
| Visibility | Public |

Both DOIs are reserved, so either record can be published first. Publish them on the same day.

`preprint/dist/zenodo-preprint.json` and `zenodo-software.json` hold the same metadata in the format of
Zenodo's REST API, for reference or scripted upload. The web form needs the values entered by hand.

## 6. After publishing

- Open both landing pages and check that the DOIs resolve. DataCite registration can take a few
  minutes.
- Zenodo also mints a *concept DOI* for each record, which always points to its latest version. Cite
  the version DOI for this preprint.
- Metadata can be edited at any time. Changing a file needs **New version**, which gets a new DOI.
- The repository has no remote. If it later goes to GitHub, push branch `preprint-v3` and tag
  `paper-v3-preprint` with it.

## Checklist

- [ ] `metadata.json` complete; `build.py --final` and `check_manuscript.py --final --mutate` pass
- [ ] PDF read page by page: names, affiliation, email, DOIs, statements
- [ ] Committed and tagged `paper-v3-preprint`; archive built from the tag, with its checks passed
- [ ] Optional: sandbox rehearsal done
- [ ] Software record published; preprint record published; each links to the other
- [ ] Landing pages and DOIs checked
