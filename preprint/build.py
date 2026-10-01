"""
Builds the paper-v3 preprint from manuscript.html and metadata.json.

    python3 preprint/build.py                       # draft: unfilled TODO fields are highlighted
    python3 preprint/build.py --final               # refuses to build while a TODO field or a DOI is missing
    python3 preprint/build.py --archive <git ref>   # also the code archive for Zenodo, from that commit or tag
    python3 preprint/build.py --review-bundle       # also a bundle for independent reviewers (preprint/AUDIT.md)
    python3 preprint/build.py --web                 # also dist/manuscript-web.html, the manuscript as a web page

Writes
    preprint/dist/manuscript.html   the filled manuscript: sections, tables, figures and citations numbered;
                                    every results table generated from its source document (paper-v3's
                                    paper/comparison_tables.md, or the twin-array results in validation/),
                                    with only the changes declared in preprint/edits.json, and the dataset
                                    table from fetch_data.py
    preprint/dist/<pdf name>        the same, printed to A4 by headless Chrome, with PDF metadata
    preprint/dist/zenodo-*.json     metadata for the two Zenodo records (preprint and software)
    preprint/dist/abstract.txt      the abstract as plain text, for Zenodo's description field
    CITATION.cff, LICENSE           at the repository root, from metadata.json
    preprint/dist/<name>.zip        with --archive: `git archive` of the ref, checked to be paper-v3, unchanged,
                                    plus the preprint's files and the twin-array validation only

check_manuscript.py checks the built manuscript against its sources; run it after every build.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIST = HERE / "dist"
SOURCE_DOC = ROOT / "paper" / "comparison_tables.md"
SOURCE_TAG = "paper-v3"
# Every number the manuscript states comes from one of these documents; each must be identical to its version
# at the tag (checked whenever git is available): name → (path here, tag, path at the tag)
SOURCES = {
    "v3": ("paper/comparison_tables.md", "paper-v3", "paper/comparison_tables.md"),
    "twin": ("validation/RESULTS_S2COMPOUNDS.md", "pre-v4", "system1/RESULTS_S2COMPOUNDS.md"),
    "twinp": ("validation/PROTOCOL_S2COMPOUNDS.md", "pre-v4", "system1/PROTOCOL_S2COMPOUNDS.md"),
}
EDITS = HERE / "edits.json"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PDF_NAME = "drift-compensation-preprint-v{version}.pdf"
ARCHIVE_NAME = "olfactory-simulation-{tag}"
# Files the preprint adds to the paper-v3 tree; the archive may differ from the tag only by these
VALIDATION_FILES = {"fetch_twin_arrays.py", "twin_arrays.py", "system2plus_core.py", "system2_compounds.py",
                    "results/twin_arrays_features.csv", "results/system2_compounds.csv",
                    "results/system2_compounds_folds.csv", "results/system2_compounds_budget.csv",
                    "results/system2_compounds_seeds.csv", "results/system2_compounds_exploratory.csv",
                    "results/logs/twin_arrays.log", "results/logs/system2_compounds.log"}
ADDED_FILES = {"LICENSE", "CITATION.cff"} | VALIDATION_FILES
ADDED_PREFIXES = ("preprint/", "validation/")
SKIP_IN_ARCHIVE = {"preprint/dist/"}


# ── source documents ──
def tagged_text(tag: str, path: str) -> str | None:
    """The file as committed at the tag, or None outside a git checkout with that tag (the archive)."""
    try:
        return subprocess.run(["git", "show", f"{tag}:{path}"], cwd=ROOT, capture_output=True,
                              check=True).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        return None


def load_sources() -> dict[str, str]:
    texts = {}
    for name, (path, tag, tag_path) in SOURCES.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        tagged = tagged_text(tag, tag_path)
        if tagged is not None and tagged != text:
            sys.exit(f"{path} differs from {tag_path} at tag {tag}: the preprint must use the tagged numbers")
        texts[name] = text
    return texts


def load_source() -> str:
    return load_sources()["v3"]


def load_edits() -> dict:
    return json.loads(EDITS.read_text(encoding="utf-8"))


def rename(text: str, edits: dict) -> str:
    for old, new in edits["rename"]:
        text = text.replace(old, new)
    return text


def v3_sections(text: str) -> dict[str, str]:
    """paper-v3 split at its '## ' headings; the text before the first heading is 'preamble'."""
    parts = re.split(r"^## ", text, flags=re.M)
    out = {"preamble": parts[0]}
    for p in parts[1:]:
        out[p.split("\n", 1)[0].strip()] = p
    return out


def v3_section(sections: dict[str, str], prefix: str) -> str:
    matches = [k for k in sections if k.startswith(prefix)]
    if len(matches) != 1:
        sys.exit(f"paper-v3 section prefix {prefix!r} matches {matches}")
    return sections[matches[0]]


def md_tables(section_text: str) -> list[list[list[str]]]:
    """The Markdown tables of a section, in order: rows of raw cells (header first, separator dropped)."""
    tables, cur = [], None
    for line in section_text.splitlines():
        if line.startswith("|"):
            if cur is None:
                cur = []
                tables.append(cur)
            if re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", line.strip()):
                continue
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        else:
            cur = None
    return tables


def md_inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def edited_rows(src: dict[str, str], key: str, omit: set[str], edits: dict) -> list[list[str]]:
    """A source table ("doc:section#index") with the declared edits applied: omitted rows, dropped columns,
    cut cells (which may only delete text), then the rename."""
    doc, rest = key.split(":", 1)
    section, index = rest.rsplit("#", 1)
    rows = [list(r) for r in md_tables(v3_section(v3_sections(src[doc]), section))[int(index)]]
    absent = omit - {r[0] for r in rows}
    if absent:
        sys.exit(f"{key}: data-omit names rows that the source table does not have: {absent}")
    rows = [rows[0]] + [r for r in rows[1:] if r[0] not in omit]
    spec = edits["tables"].get(key, {})
    header = rows[0]
    for col in spec.get("drop_columns", []):
        if col not in header:
            sys.exit(f"{key}: no column {col!r} to drop")
    keep = [i for i, h in enumerate(header) if h not in spec.get("drop_columns", [])]
    rows = [[r[i] for i in keep] for r in rows]
    header = rows[0]
    for cut in spec.get("cells", []):
        r = next((r for r in rows[1:] if r[0] == cut["row"]), None)
        if r is None or cut["column"] not in header:
            sys.exit(f"{key}: no cell {cut['row']!r} / {cut['column']!r}")
        i = header.index(cut["column"])
        if not is_deletion(r[i], cut["text"]):
            sys.exit(f"{key}: the cut of {cut['row']!r} / {cut['column']!r} adds text: {r[i]!r} → {cut['text']!r}")
        r[i] = cut["text"]
    return [[rename(c, edits) for c in r] for r in rows]


def is_deletion(before: str, after: str) -> bool:
    """True if `after` is `before` with whole words removed: its words, trailing punctuation aside, appear in
    `before` in the same order and unchanged (so no number can be altered, only dropped)."""
    def words(s: str) -> list[str]:
        return [w.rstrip(",:;") for w in s.split()]
    have, pos = words(before), 0
    for w in words(after):
        try:
            pos = have.index(w, pos) + 1
        except ValueError:
            return False
    return True


def table_html(rows: list[list[str]], omit: set[str], cls: str) -> str:
    header, body = rows[0], [r for r in rows[1:] if r[0] not in omit]
    out = [f'<thead><tr>{"".join(f"<th>{md_inline(c)}</th>" for c in header)}</tr></thead>', "<tbody>"]
    for i, r in enumerate(body):
        group = ' class="group-start"' if i > 0 and r[0] else ""
        out.append(f'<tr{group}>{"".join(f"<td>{md_inline(c)}</td>" for c in r)}</tr>')
    out.append("</tbody>")
    return "".join(out)


# ── dataset table (fetch_data.py) ──
def month_ranges(months: str) -> str:
    ms = [int(m) for m in months.split(",")]
    runs, start = [], ms[0]
    for a, b in zip(ms, ms[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}–{a}" if a != start else f"{a}")
            start = b
    return ", ".join(runs)


def dataset_table() -> str:
    sys.path.insert(0, str(ROOT))
    import fetch_data as F
    classes = sorted(F.GASES)
    head = ["Batch", "Months"] + [f"{c} {F.GASES[c]}" for c in classes] + ["Total"]
    rows = []
    for b in sorted(F.EXPECTED_COUNTS):
        counts = F.EXPECTED_COUNTS[b]
        rows.append([str(b), month_ranges(F.BATCH_MONTHS[b])] + [f"{n:,}" for n in counts] + [f"{sum(counts):,}"])
    totals = [sum(F.EXPECTED_COUNTS[b][i] for b in F.EXPECTED_COUNTS) for i in range(len(classes))]
    rows.append(["All", ""] + [f"{n:,}" for n in totals] + [f"{sum(totals):,}"])
    out = [f'<thead><tr>{"".join(f"<th>{html.escape(c)}</th>" for c in head)}</tr></thead>', "<tbody>"]
    for i, r in enumerate(rows):
        cls = ' class="group-start"' if r[0] == "All" else ""
        out.append(f'<tr{cls}>{"".join(f"<td>{html.escape(c)}</td>" for c in r)}</tr>')
    out.append("</tbody>")
    return "".join(out)


# ── metadata ──
def is_todo(v) -> bool:
    return isinstance(v, str) and "TODO" in v


def todos(meta: dict, path: str = "") -> list[str]:
    out = []
    if isinstance(meta, dict):
        for k, v in meta.items():
            if not k.startswith("_"):
                out += todos(v, f"{path}.{k}" if path else k)
    elif isinstance(meta, list):
        for i, v in enumerate(meta):
            out += todos(v, f"{path}[{i}]")
    elif is_todo(meta):
        out.append(path)
    return out


def esc(v: str) -> str:
    """Escaped text; a TODO value is highlighted so that it cannot be missed in a draft."""
    s = html.escape(v, quote=False)
    return f'<span class="placeholder">{s}</span>' if is_todo(v) else s


def full_name(a: dict) -> str:
    return f"{a['given_names']} {a['family_names']}"


def front_matter(meta: dict) -> dict[str, str]:
    authors = meta["authors"]
    affs = list(dict.fromkeys(a["affiliation"] for a in authors if a.get("affiliation")))
    many = len(affs) > 1
    names = []
    for a in authors:
        sup = f"<sup>{affs.index(a['affiliation']) + 1}</sup>" if many and a.get("affiliation") else ""
        star = "<sup>*</sup>" if a.get("corresponding") and len(authors) > 1 else ""
        names.append(f"{esc(a['given_names'])} {esc(a['family_names'])}{sup}{star}")
    authors_html = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    aff_lines = [f'<p class="affils">{f"<sup>{i + 1}</sup> " if many else ""}{esc(x)}</p>' for i, x in enumerate(affs)]
    orcids = [(full_name(a), a["orcid"]) for a in authors if a.get("orcid")]
    if orcids:
        links = "; ".join(f'{esc(n) + ": " if len(authors) > 1 else ""}<a href="https://orcid.org/{html.escape(o)}">'
                          f"orcid.org/{html.escape(o)}</a>" for n, o in orcids)
        aff_lines.append(f'<p class="affils">ORCID: {links}</p>')
    corr = [a for a in authors if a.get("corresponding")] or authors[:1]
    corr_html = "Correspondence: " + "; ".join(
        f"{esc(full_name(a)) + ', ' if len(authors) > 1 else ''}{esc(a['email'])}" for a in corr if a.get("email"))
    date = dt.date.fromisoformat(meta["date"])
    year = date.year
    holders = ", ".join(full_name(a) for a in authors)
    pdoi, sdoi = meta.get("preprint_doi", ""), meta.get("software_doi", "")
    front_doi = f' · <a href="https://doi.org/{html.escape(pdoi)}">doi:{html.escape(pdoi)}</a>' if pdoi else ""
    software_doi_html = (f'<a href="https://doi.org/{html.escape(sdoi)}">doi:{html.escape(sdoi)}</a>' if sdoi
                         else '<span class="placeholder">software DOI to be added</span>')
    st = meta["statements"]
    return {
        "title": esc(meta["title"]),
        "author_names": html.escape(", ".join(full_name(a) for a in authors)),
        "keywords": html.escape("; ".join(meta["keywords"])),
        "version": esc(meta["version"]),
        "date_long": f"{date.day} {date:%B %Y}",
        "front_doi": front_doi,
        "authors_html": authors_html,
        "affiliations_html": "\n  ".join(aff_lines),
        "correspondence_html": corr_html,
        "licence_html": f"© {year} {esc(holders)}. This preprint is licensed under "
                        f'<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.',
        "software_doi_html": software_doi_html,
        "statement_ai": esc(st["ai_assistance"]),
        "statement_funding": esc(st["funding"]),
        "statement_competing": esc(st["competing_interests"]),
    }


# ── numbering and citations ──
def number_everything(doc: str) -> str:
    """Numbers sections (h2, h3), tables and figures in document order, then fills cross-references and
    citations; the reference list is ordered by first citation."""
    ids: dict[str, str] = {}
    counters = {"sec": 0, "sub": 0, "table": 0, "fig": 0}
    in_numbered = False
    out, pos = [], 0
    token = re.compile(r'<section([^>]*)>|<h2>|<h3( id="[^"]+")?>|<figure class="(table|fig)((?: [\w-]+)*)" id="([^"]+)">\s*'
                       r"(?:(<svg.*?</svg>)\s*)?<figcaption>", re.S)
    current_section_id = None
    for m in token.finditer(doc):
        out.append(doc[pos:m.start()])
        pos = m.end()
        s = m.group(0)
        if s.startswith("<section"):
            attrs = m.group(1)
            in_numbered = "unnumbered" not in attrs
            sid = re.search(r'id="([^"]+)"', attrs)
            current_section_id = sid.group(1) if sid else None
            out.append(s)
        elif s == "<h2>":
            if in_numbered:
                counters["sec"] += 1
                counters["sub"] = 0
                if current_section_id:
                    ids[current_section_id] = str(counters["sec"])
                out.append(f'<h2><span class="secnum">{counters["sec"]}</span>')
            else:
                out.append(s)
        elif s.startswith("<h3"):
            counters["sub"] += 1
            num = f'{counters["sec"]}.{counters["sub"]}'
            if m.group(2):
                ids[m.group(2).split('"')[1]] = num
            out.append(f'<h3{m.group(2) or ""}><span class="secnum">{num}</span>')
        else:
            kind, extra, fid, svg = m.group(3), m.group(4), m.group(5), m.group(6)
            counters[kind] += 1
            n = counters[kind]
            ids[fid] = str(n)
            label = "Table" if kind == "table" else "Figure"
            out.append(f'<figure class="{kind}{extra}" id="{fid}">' + (f"\n    {svg}\n    " if svg else "")
                       + f'<figcaption><span class="capnum">{label} {n}.</span> ')
    out.append(doc[pos:])
    doc = "".join(out)

    def xref(m: re.Match) -> str:
        ref = m.group(1)
        if ref not in ids:
            sys.exit(f"unresolved cross-reference {ref!r}")
        return f'<a class="xref" href="#{ref}" data-ref="{ref}">{ids[ref]}</a>'
    doc = re.sub(r'<a class="xref" data-ref="([^"]+)"></a>', xref, doc)

    refs_start = doc.index('<section class="unnumbered" id="sec-references"')
    body, refs = doc[:refs_start], doc[refs_start:]
    order: list[str] = []
    for m in re.finditer(r'<cite data-keys="([^"]+)"></cite>', body):
        for k in m.group(1).split(","):
            if k not in order:
                order.append(k)
    items = dict(re.findall(r'<li data-key="([^"]+)">(.*?)</li>', refs, re.S))
    missing = [k for k in order if k not in items]
    uncited = [k for k in items if k not in order]
    if missing or uncited:
        sys.exit(f"citations without a reference: {missing}; references never cited: {uncited}")
    number = {k: i + 1 for i, k in enumerate(order)}

    def cite(m: re.Match) -> str:
        ns = sorted(number[k] for k in m.group(1).split(","))
        runs, start = [], ns[0]
        for a, b in zip(ns, ns[1:] + [None]):
            if b != a + 1:
                runs.append(f"{start}–{a}" if a - start >= 2 else ", ".join(str(x) for x in range(start, a + 1)))
                start = b
        return f'<cite class="cite" data-keys="{m.group(1)}">[{", ".join(runs)}]</cite>'
    body = re.sub(r'<cite data-keys="([^"]+)"></cite>', cite, body)
    lis = "\n    ".join(f'<li data-key="{k}" id="ref-{k}"><span class="refnum">[{number[k]}]</span><span>{items[k]}</span></li>'
                       for k in order)
    refs = re.sub(r'<ol class="references">.*?</ol>', lambda _: f'<ol class="references">\n    {lis}\n  </ol>', refs,
                  flags=re.S)
    return body + refs


def fill_tables(doc: str, src: dict[str, str]) -> str:
    edits = load_edits()

    def source_table(m: re.Match) -> str:
        attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
        omit = {o for o in attrs.get("data-omit", "").split("|") if o}
        rows = edited_rows(src, attrs["data-src"], omit, edits)
        return f"<table{m.group(1)}>{table_html(rows, set(), attrs.get('class', ''))}</table>"
    doc = re.sub(r"<table((?: [\w-]+=\"[^\"]*\")*? data-src=\"[^\"]+\"(?: [\w-]+=\"[^\"]*\")*)></table>", source_table, doc)
    return re.sub(r'<table class="num" data-generated="dataset"></table>',
                  lambda _: f'<table class="num" data-generated="dataset">{dataset_table()}</table>', doc)


def build_html(meta: dict, src: dict[str, str]) -> str:
    doc = (HERE / "manuscript.html").read_text(encoding="utf-8")
    doc = re.sub(r"<!--.*?-->\n", "", doc, count=1, flags=re.S)       # the template's own comment
    fields = front_matter(meta)
    doc = re.sub(r"\{\{(\w+)\}\}", lambda m: fields[m.group(1)] if m.group(1) in fields else sys.exit(f"no field {m.group(1)}"), doc)
    doc = fill_tables(doc, src)
    head, body = doc.split("</head>", 1)
    return number_everything(head + "</head>" + keep_units(body))


UNIT_PAIR = re.compile(r"(\d(?:[\d.,–]*\d)?) ((?:%|µJ|µW|µA|mJ|nJ|pJ|KB|MB|MHz|mW|ms|°C|points?|cycles)(?!\w))")


def keep_units(body: str) -> str:
    """Keeps a number on the same line as its unit (a no-wrap span around both, with an ordinary space);
    only in text, never inside tags or in the figure's SVG."""
    out, in_svg = [], False
    for part in re.split(r"(<[^>]+>)", body):
        if part.startswith("<"):
            in_svg = part.startswith("<svg") or (in_svg and not part.startswith("</svg"))
            out.append(part)
        else:
            out.append(part if in_svg else UNIT_PAIR.sub(r'<span class="nw">\1 \2</span>', part))
    return "".join(out)


# ── PDF ──
def print_pdf(html_path: Path, pdf_path: Path, meta: dict) -> int:
    if not CHROME.exists():
        sys.exit(f"Chrome not found at {CHROME}")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                        f"--user-data-dir={profile}", "--run-all-compositor-stages-before-draw",
                        "--virtual-time-budget=10000", f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                       check=True, capture_output=True, timeout=300)
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_path.read_bytes())))
    writer.add_metadata({"/Title": meta["title"], "/Author": ", ".join(full_name(a) for a in meta["authors"]),
                         "/Subject": "Preprint, not peer reviewed; licensed CC BY 4.0",
                         "/Keywords": "; ".join(meta["keywords"]), "/Creator": "preprint/build.py (headless Chrome)"})
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return len(writer.pages)


# ── CITATION.cff, LICENSE, Zenodo metadata ──
def yq(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)                              # a valid YAML double-quoted scalar


def citation_cff(meta: dict) -> str:
    def people(indent: str) -> list[str]:
        out = []
        for a in meta["authors"]:
            out.append(f"{indent}- family-names: {yq(a['family_names'])}")
            out.append(f"{indent}  given-names: {yq(a['given_names'])}")
            if a.get("affiliation"):
                out.append(f"{indent}  affiliation: {yq(a['affiliation'])}")
            if a.get("orcid"):
                out.append(f"{indent}  orcid: {yq('https://orcid.org/' + a['orcid'])}")
        return out
    lines = ["# Generated by preprint/build.py from preprint/metadata.json; edit that file, not this one.",
             "cff-version: 1.2.0",
             'message: "If you use this software, please cite the preprint it accompanies (preferred-citation)."',
             "type: software",
             f"title: {yq('Code and results for: ' + meta['title'])}",
             "authors:", *people("  "),
             f"version: {yq(meta['software_version'])}",
             f"date-released: {yq(meta['date'])}",
             "license: MIT"]
    if meta.get("software_doi"):
        lines.append(f"doi: {yq(meta['software_doi'])}")
    lines += ["keywords:", *[f"  - {yq(k)}" for k in meta["keywords"]],
              "preferred-citation:",
              "  type: article",
              "  status: preprint",
              f"  title: {yq(meta['title'])}",
              "  authors:", *people("    "),
              f"  year: {dt.date.fromisoformat(meta['date']).year}",
              '  publisher:',
              '    name: "Zenodo"']
    if meta.get("preprint_doi"):
        lines.append(f"  doi: {yq(meta['preprint_doi'])}")
    lines += ["references:",
              "  - type: data",
              '    title: "Gas Sensor Array Drift at Different Concentrations"',
              "    authors:",
              '      - family-names: "Vergara"',
              '        given-names: "Alexander"',
              "    year: 2012",
              '    publisher:',
              '      name: "UCI Machine Learning Repository"',
              '    doi: "10.24432/C5MK6M"',
              "  - type: data",
              '    title: "Twin gas sensor arrays"',
              "    authors:",
              '      - family-names: "Fonollosa"',
              '        given-names: "Jordi"',
              "    year: 2016",
              '    publisher:',
              '      name: "UCI Machine Learning Repository"',
              '    doi: "10.24432/C5MW3K"']
    return "\n".join(lines) + "\n"


MIT = """MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def abstract_text(doc: str) -> str:
    m = re.search(r'<section class="abstract[^"]*"[^>]*>.*?<p>(.*?)</p>', doc, re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()


def zenodo_records(meta: dict, abstract: str) -> dict[str, dict]:
    creators = [{"name": f"{a['family_names']}, {a['given_names']}",
                 **({"affiliation": a["affiliation"]} if a.get("affiliation") else {}),
                 **({"orcid": a["orcid"]} if a.get("orcid") else {})} for a in meta["authors"]]
    datasets = [{"identifier": d, "relation": "references", "resource_type": "dataset", "scheme": "doi"}
                for d in ("10.24432/C5RP6W", "10.24432/C5MK6M", "10.24432/C5MW3K")]
    pdoi, sdoi = meta.get("preprint_doi") or "<preprint DOI>", meta.get("software_doi") or "<software DOI>"
    preprint = {"upload_type": "publication", "publication_type": "preprint", "title": meta["title"],
                "creators": creators, "description": f"<p>{html.escape(abstract)}</p>", "access_right": "open",
                "license": "cc-by-4.0", "keywords": meta["keywords"], "version": meta["version"],
                "language": "eng", "publication_date": meta["date"], "doi": pdoi,
                "related_identifiers": [{"identifier": sdoi, "relation": "isSupplementedBy",
                                         "resource_type": "software", "scheme": "doi"}] + datasets}
    software_desc = (f"<p>Code, results and manuscript source for the preprint “{html.escape(meta['title'])}” "
                     f"(doi:{html.escape(pdoi)}). This is the project repository at git tag paper-v3, unchanged, with "
                     "a licence, citation metadata, the preprint's source and the twin-array validation "
                     "(<code>validation/</code>) added. To reproduce from this download, run "
                     "<code>preprint/audit_reproduction.sh</code> (see <code>preprint/AUDIT.md</code>): it snapshots "
                     "the unpacked folder in git, runs <code>./reproduce.sh</code>, which downloads the UCI dataset, "
                     "reruns every experiment and checks every number in the results summary, and fails unless every "
                     "regenerated file was compared with the archived one and matched; it then regenerates the "
                     "twin-array validation the same way. "
                     "<code>preprint/check_manuscript.py</code> ties the manuscript's numbers to that summary. The "
                     "datasets themselves are not included.</p>")
    software = {"upload_type": "software", "title": f"Code and results for: {meta['title']}", "creators": creators,
                "description": software_desc, "access_right": "open", "license": "mit",
                "keywords": meta["keywords"], "version": meta["software_version"], "language": "eng",
                "publication_date": meta["date"], "doi": sdoi,
                "related_identifiers": [{"identifier": pdoi, "relation": "isSupplementTo",
                                         "resource_type": "publication-preprint", "scheme": "doi"}] + datasets}
    return {"zenodo-preprint.json": {"metadata": preprint}, "zenodo-software.json": {"metadata": software}}


# ── code archive ──
V4_PATTERNS = re.compile(r"(^|/)(system1/|fetch_home|home_activity|system2_home|presence_home|"
                         r"system1_v2|system1_sentinel|energy_system|check_paper_v4)", re.I)


def git_files(ref: str) -> set[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref], cwd=ROOT, capture_output=True, check=True)
    return set(out.stdout.decode().splitlines())


def build_archive(ref: str) -> Path:
    files, base = git_files(ref), git_files(SOURCE_TAG)
    added, removed = files - base, base - files
    diff = subprocess.run(["git", "diff", "--name-status", "--no-renames", SOURCE_TAG, ref], cwd=ROOT,
                          capture_output=True, check=True).stdout.decode().splitlines()
    changed = sorted(line.split("\t", 1)[1] for line in diff if line.startswith("M"))
    extra = sorted(f for f in added if f not in ADDED_FILES and not f.startswith(ADDED_PREFIXES))
    problems = []
    if removed:
        problems.append(f"files of {SOURCE_TAG} missing from {ref}: {sorted(removed)}")
    if changed:
        problems.append(f"files of {SOURCE_TAG} changed in {ref}: {changed}")
    if extra:
        problems.append(f"files added outside the preprint's and the validation's: {extra}")
    v4 = sorted(f for f in files if V4_PATTERNS.search(f))
    if v4:
        problems.append(f"paper-v4 material in {ref}: {v4}")
    data = sorted(f for f in files if f.startswith("data/") and f != "data/SHA256SUMS")
    if data:
        problems.append(f"dataset files in {ref}: {data}")
    if any(f.startswith(s) for f in files for s in SKIP_IN_ARCHIVE):
        problems.append(f"build outputs committed in {ref}")
    if problems:
        sys.exit("archive refused:\n  " + "\n  ".join(problems))
    name = ARCHIVE_NAME.format(tag=ref.replace("/", "-"))
    out = DIST / f"{name}.zip"
    subprocess.run(["git", "archive", "--format=zip", f"--prefix={name}/", "-o", str(out), ref], cwd=ROOT, check=True)
    with zipfile.ZipFile(out) as z:
        listed = {n[len(name) + 1:] for n in z.namelist() if not n.endswith("/")}
    if listed != files:
        sys.exit(f"archive content differs from {ref}: {sorted(listed ^ files)[:10]}")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"archive  {out.relative_to(ROOT)}: {len(files)} files ({len(base)} from {SOURCE_TAG}, "
          f"{len(added)} added: {', '.join(sorted(added))}), {out.stat().st_size / 1e6:.1f} MB, sha256 {digest}")
    return out


WEB_STYLE = """<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=STIX+Two+Text:ital@0;1&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap">
<style>
  :root {
    --bg: #ffffff; --surface: #f3f5f8; --box: #eef1f4; --ink: #17191e; --ink-soft: #3a3f48; --muted: #5b6371;
    --rule: #d4d9e0; --rule-strong: #17191e; --accent: #1d5a8e; --mark: #ffeaa3; --mark-ink: #3b2f00;
    --serif: "Source Serif 4", "STIX Two Text", Georgia, "Times New Roman", serif;
    --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark; --bg: #121418; --surface: #1a1e24; --box: #20252c; --ink: #e5e8ec; --ink-soft: #c7ccd4;
      --muted: #a0a8b5; --rule: #343a44; --rule-strong: #c7ccd4; --accent: #8fbde9; --mark: #5a4912; --mark-ink: #fff3c4;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark; --bg: #121418; --surface: #1a1e24; --box: #20252c; --ink: #e5e8ec; --ink-soft: #c7ccd4;
    --muted: #a0a8b5; --rule: #343a44; --rule-strong: #c7ccd4; --accent: #8fbde9; --mark: #5a4912; --mark-ink: #fff3c4;
  }
  body { background: var(--bg); color: var(--ink); font-family: var(--serif); font-size: 17px; line-height: 1.62;
         padding-inline: 16px; padding-block: 28px 72px; }
  main { max-width: 44rem; margin: 0 auto; }
  p { margin: 0 0 0.9em; }
  h1 { font-size: 1.95rem; line-height: 1.2; font-weight: 700; margin: 0.3em 0 0.5em; text-wrap: balance; }
  h2 { font-size: 1.3rem; line-height: 1.3; margin: 2.2em 0 0.6em; text-wrap: balance; }
  h3 { font-size: 1.06rem; margin: 1.6em 0 0.4em; }
  .secnum { display: inline-block; min-width: 2.2em; color: var(--muted); font-weight: 600; }
  a { color: var(--accent); text-underline-offset: 2px; }
  a.xref, cite a { text-decoration: none; }
  cite { font-style: normal; color: var(--ink-soft); }
  code { font-family: var(--mono); font-size: 0.84em; background: var(--surface); padding: 0.05em 0.3em; border-radius: 3px; }
  .nw { white-space: nowrap; }
  .placeholder { background: var(--mark); color: var(--mark-ink); padding: 0 0.2em; border-radius: 2px; }
  .review { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 18px 4px;
            margin: 0 0 32px; font-size: 15.5px; line-height: 1.55; }
  .review-label { font-size: 12.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
                  font-weight: 600; margin-bottom: 0.5em; }
  .front .preprint-note { font-size: 13.5px; color: var(--muted); border-bottom: 1px solid var(--rule); padding-bottom: 8px; }
  .front .authors { font-size: 1.1rem; margin: 0 0 0.2em; }
  .front .affils, .front .corr, .front .licence { font-size: 15px; color: var(--ink-soft); margin: 0 0 0.15em; }
  .abstract { border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 28px 0; padding: 4px 0 8px; }
  .abstract h2 { font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 14px 0 8px; }
  .abstract p { font-size: 16px; }
  .keywords { font-size: 15px; color: var(--ink-soft); }
  ul, ol { padding-left: 1.3em; margin: 0 0 1em; }
  li { margin-bottom: 0.45em; }
  figure { margin: 1.6em 0; }
  figcaption { font-size: 14.5px; line-height: 1.5; color: var(--ink-soft); margin: 0.4em 0 0.6em; }
  .capnum { font-weight: 700; color: var(--ink); }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; line-height: 1.4; font-variant-numeric: lining-nums tabular-nums;
          border-top: 1.5px solid var(--rule-strong); border-bottom: 1.5px solid var(--rule-strong); }
  table.num, table.num2 { width: max-content; min-width: 100%; }
  th, td { padding: 6px 10px; text-align: left; vertical-align: top; }
  thead th { border-bottom: 1px solid var(--rule-strong); font-weight: 600; vertical-align: bottom; }
  tbody tr.group-start td { border-top: 1px solid var(--rule); }
  table.num td:not(:first-child), table.num th:not(:first-child),
  table.num2 td:nth-child(n+3), table.num2 th:nth-child(n+3) { text-align: center; white-space: nowrap; }
  table.num2 td:first-child { white-space: nowrap; }
  .tnote { font-size: 13.5px; color: var(--ink-soft); margin-top: 0.5em; }
  .fig svg { display: block; width: 100%; min-width: 640px; height: auto; }
  .fig svg text { fill: var(--ink); font-family: var(--serif); }
  .fig svg g[font-style] text { fill: var(--muted); }
  .fig svg rect { fill: var(--box); stroke: var(--ink-soft); }
  .fig svg rect[fill="#fff"] { fill: var(--bg); }
  .fig svg line { stroke: var(--ink-soft); }
  .fig svg marker path { fill: var(--ink-soft); }
  .declarations p { font-size: 15.5px; }
  ol.references { list-style: none; padding-left: 0; font-size: 14.5px; line-height: 1.5; }
  ol.references li { display: grid; grid-template-columns: 2.6rem 1fr; }
  .refnum { color: var(--muted); }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  @media (max-width: 520px) { body { font-size: 16px; } h1 { font-size: 1.6rem; } }
</style>
"""


def web_page(doc: str, meta: dict) -> str:
    """The built manuscript as a self-contained web page for reviewers: the same body as the PDF, with
    screen typography, light and dark themes, and wide tables and the figure scrolling in their own boxes."""
    body = doc.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    body = re.sub(r"(<table\b.*?</table>)", r'<div class="scroll">\1</div>', body, flags=re.S)
    body = re.sub(r"(<svg\b.*?</svg>)", r'<div class="scroll">\1</div>', body, flags=re.S)
    body = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', body)
    name = meta["title"].split(":")[0].strip() if ":" in meta["title"] else "Preprint draft"
    open_items = todos(meta) + [f for f in ("preprint_doi", "software_doi") if not meta.get(f)]
    pending = ("Items still to be completed are highlighted: the AI-assistance statement, which is written "
               "after the independent audit, and the Zenodo DOIs." if open_items else "")
    banner = (f'<aside class="review" aria-label="About this draft">\n'
              f'  <p class="review-label">Draft for review · version {html.escape(meta["version"])} · not yet published</p>\n'
              f'  <p>This is the draft preprint, built from the same source as its PDF. {pending}</p>\n'
              f'  <p><strong>Comments:</strong> flag anything wrong, unclear or overstated, on this page or directly to the author.</p>\n'
              f'  <p><strong>Reproducing the results:</strong> the author sends you the review bundle '
              f'(<code>paper-v3-review-&lt;date&gt;.zip</code>), which also holds the PDF; follow '
              f'<code>preprint/AUDIT.md</code> inside it.</p>\n'
              f'</aside>')
    return f"<title>{html.escape(name)} preprint</title>\n{WEB_STYLE}<main>\n{banner}\n{body}\n</main>\n"


def build_review_bundle() -> Path:
    """For independent reviewers before the release: paper-v3 exactly as tagged, plus the working tree's
    LICENSE, CITATION.cff and preprint files, with the built manuscript and PDF. Not the Zenodo archive."""
    name = f"paper-v3-review-{dt.date.today():%Y%m%d}"
    out = DIST / f"{name}.zip"
    tagged = subprocess.run(["git", "archive", "--format=zip", SOURCE_TAG], cwd=ROOT, capture_output=True,
                            check=True).stdout
    extra = sorted(ADDED_FILES) + sorted(
        p.relative_to(ROOT).as_posix() for d in ADDED_PREFIXES for p in (ROOT / d).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".zip" and p.name != ".DS_Store"
        and not (p.parent.name == "audit" and p.suffix == ".log"))
    files = []
    with zipfile.ZipFile(io.BytesIO(tagged)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            if info.is_dir():
                continue
            zi = zipfile.ZipInfo(f"{name}/{info.filename}", date_time=info.date_time)
            zi.external_attr, zi.compress_type = info.external_attr, zipfile.ZIP_DEFLATED
            z.writestr(zi, src.read(info))
            files.append(info.filename)
        for rel in extra:
            path = ROOT / rel
            zi = zipfile.ZipInfo(f"{name}/{rel}", date_time=time.localtime(path.stat().st_mtime)[:6])
            zi.external_attr = (0o100755 if os.access(path, os.X_OK) else 0o100644) << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, path.read_bytes())
            files.append(rel)
    bad = [f for f in files if V4_PATTERNS.search(f) or (f.startswith("data/") and f != "data/SHA256SUMS")]
    if bad:
        out.unlink()
        sys.exit(f"review bundle refused (paper-v4 material or dataset files): {bad}")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"bundle   {out.relative_to(ROOT)}: {len(files)} files ({len(files) - len(extra)} from {SOURCE_TAG}), "
          f"{out.stat().st_size / 1e6:.1f} MB, sha256 {digest}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--final", action="store_true", help="refuse to build while a TODO field or a DOI is missing")
    parser.add_argument("--archive", metavar="REF", help="also build the code archive from this git ref (tag or commit)")
    parser.add_argument("--no-pdf", action="store_true", help="skip the PDF (quicker, for checking the HTML)")
    parser.add_argument("--review-bundle", action="store_true", help="also a zip for independent reviewers")
    parser.add_argument("--web", action="store_true", help="also the manuscript as a web page for reviewers")
    args = parser.parse_args()

    meta = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
    open_items = todos(meta) + [f for f in ("preprint_doi", "software_doi") if not meta.get(f)]
    if args.final and open_items:
        sys.exit("still to fill in metadata.json: " + ", ".join(open_items))
    src = load_sources()
    DIST.mkdir(exist_ok=True)
    (HERE / ".gitignore").write_text("dist/\n")

    doc = build_html(meta, src)
    html_path = DIST / "manuscript.html"
    html_path.write_text(doc, encoding="utf-8")
    abstract = abstract_text(doc)
    (DIST / "abstract.txt").write_text(abstract + "\n", encoding="utf-8")
    for name, record in zenodo_records(meta, abstract).items():
        (DIST / name).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "CITATION.cff").write_text(citation_cff(meta), encoding="utf-8")
    year = dt.date.fromisoformat(meta["date"]).year
    (ROOT / "LICENSE").write_text(MIT.format(year=year, holder=meta["copyright_holder"]), encoding="utf-8")
    print(f"html     {html_path.relative_to(ROOT)}")
    if not args.no_pdf:
        pdf = DIST / PDF_NAME.format(version=meta["version"])
        pages = print_pdf(html_path, pdf, meta)
        print(f"pdf      {pdf.relative_to(ROOT)}: {pages} pages, {pdf.stat().st_size / 1e3:.0f} kB")
    print("also     CITATION.cff, LICENSE, preprint/dist/abstract.txt, preprint/dist/zenodo-preprint.json, "
          "preprint/dist/zenodo-software.json")
    if args.web:
        web = DIST / "manuscript-web.html"
        web.write_text(web_page(doc, meta), encoding="utf-8")
        print(f"web      {web.relative_to(ROOT)}")
    if args.review_bundle:
        build_review_bundle()
    if args.archive:
        build_archive(args.archive)
    if open_items:
        print(f"draft: {len(open_items)} item(s) still to fill in metadata.json: {', '.join(open_items)}")


if __name__ == "__main__":
    main()
