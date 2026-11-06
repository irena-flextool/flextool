"""The results catalogue page: every result table with its files, layout, units and results-database name.

Built from ``process_outputs._output_meta`` (OUTPUT_DESCRIPTION, OUTPUT_DIMS, OUTPUT_TRANSFORM), the
results-database spec (``write_spinedb.SPINEDB_OUTPUT_SPEC``), the CSV names of the default output settings
(``schemas/default_plots.yaml`` ``rename``) and the topics of ``results.md``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from flextool.process_outputs._output_meta import (
    OUTPUT_DESCRIPTION,
    OUTPUT_DIMS,
    OUTPUT_TRANSFORM,
    OutputDims,
    Semantics,
    Transform,
)
from flextool.process_outputs.write_spinedb import SPINEDB_OUTPUT_SPEC

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / "schemas" / "default_plots.yaml"
RESULTS_DOC = "results.md"
OTHER_TOPIC = "Other"

_TOPIC_RE = re.compile(r"^## (?P<title>.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*- (?P<text>.+)$")
_ENTITY_RE = re.compile(r"^(?P<classes>.*?)\bentit(?:y|ies)\b(?P<params>.*)$")
_PLAIN_PARAM_RE = re.compile(r"^(?P<classes>(?:`[^`]+`(?:,| and)?\s*)+)`(?P<param>[^`]+)` parameter")
_TABLE_RE = re.compile(r"^table `(?P<stem>[^`]+)`")
_CSV_RE = re.compile(r"^`(?P<csv>[^`]+)\.csv`")


@dataclass(frozen=True)
class Bullet:
    """A ``results.md`` bullet that names a result: by its results-database parameter or by its table."""

    topic: str
    text: str
    pairs: tuple[tuple[str, str], ...]     # (entity class, parameter)
    stem: str | None                       # a bullet of the form "table `<stem>` …"
    csv: str | None = None                 # a bullet of the form "`<csv name>.csv` …"


def stems() -> list[str]:
    """Every result table the catalogue covers: the declared outputs and the results-database tables."""
    return sorted(set(OUTPUT_TRANSFORM) | set(SPINEDB_OUTPUT_SPEC))


def csv_names(settings: Path = DEFAULT_SETTINGS) -> dict[str, str]:
    """CSV file stem of each table under the default output settings (the ``rename`` display names)."""
    data = yaml.safe_load(settings.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for key, entry in (data.get("rename") or {}).items():
        out[str(key)] = str(entry[0]) if isinstance(entry, list) and entry else str(entry)
    return out


def results_db_name(stem: str) -> str | None:
    spec = SPINEDB_OUTPUT_SPEC.get(stem)
    return None if spec is None else f"{spec['ec']}.{spec['param']}"


def parse_results_bullets(markdown: str) -> list[Bullet]:
    """The bullets of ``results.md`` that name a result, with the ``##`` topic they sit under."""
    topic = ""
    out: list[Bullet] = []
    fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if t := _TOPIC_RE.match(line):
            topic = t.group("title")
            continue
        b = _BULLET_RE.match(line)
        if b is None:
            continue
        head = b.group("text").split(" — ", 1)[0]
        if t := _TABLE_RE.match(head):
            out.append(Bullet(topic, head, (), t.group("stem")))
            continue
        if c := _CSV_RE.match(head):
            out.append(Bullet(topic, head, (), None, c.group("csv")))
            continue
        if m := _ENTITY_RE.match(head):
            classes = re.findall(r"`([^`]+)`", m.group("classes"))
            params = re.findall(r"`([^`]+)`", m.group("params"))
        elif m := _PLAIN_PARAM_RE.match(head):
            classes = re.findall(r"`([^`]+)`", m.group("classes"))
            params = [m.group("param")]
        else:
            continue
        if classes and params:
            out.append(Bullet(topic, head, tuple((c, p) for c in classes for p in params), None))
    return out


def topics(markdown: str) -> tuple[dict[str, str], list[Bullet]]:
    """``({stem: topic}, unmatched bullets)``: a table's topic is the first ``results.md`` topic whose bullet names
    its results-database parameter, the table itself or its CSV file; a bullet that names none of them is
    unmatched."""
    by_pair: dict[tuple[str, str], list[str]] = {}
    for stem, spec in SPINEDB_OUTPUT_SPEC.items():
        by_pair.setdefault((spec["ec"], spec["param"]), []).append(stem)
    known = set(stems())
    by_csv = {name: stem for stem, name in csv_names().items() if stem in known}
    found: dict[str, str] = {}
    unmatched: list[Bullet] = []
    for bullet in parse_results_bullets(markdown):
        if bullet.stem is not None:
            hits = [bullet.stem] if bullet.stem in known else []
        elif bullet.csv is not None:
            hits = [by_csv[bullet.csv]] if bullet.csv in by_csv else []
        else:
            hits = [s for pair in bullet.pairs for s in by_pair.get(pair, [])]
        if not hits:
            unmatched.append(bullet)
        for stem in hits:
            found.setdefault(stem, bullet.topic)
    return found, unmatched


def topic_order(markdown: str) -> list[str]:
    return [t.group("title") for line in markdown.splitlines() if (t := _TOPIC_RE.match(line))]


def _measure_lines(stem: str) -> list[tuple[str, str, str]]:
    """``(column, unit, semantics)`` of the measure columns ("*" for every column of a uniform table)."""
    spec = OUTPUT_TRANSFORM.get(stem)
    if spec is None:
        return []
    if isinstance(spec, Transform):
        if spec.semantics is Semantics.DIMENSION:
            return []
        return [("*", spec.unit, spec.semantics.value)]
    return [(col, t.unit, t.semantics.value) for col, t in spec.items() if t.semantics is not Semantics.DIMENSION]


def record(stem: str, topic: str, csv: dict[str, str]) -> dict:
    """The catalogue record of one table."""
    dims: OutputDims = OUTPUT_DIMS[stem]
    return {
        "stem": stem,
        "topic": topic,
        "description": OUTPUT_DESCRIPTION[stem],
        "parquet": f"{stem}.parquet",
        "csv": f"{csv.get(stem, stem)}.csv",
        "dims": {"rows": list(dims.rows), "columns": list(dims.columns), "time": dims.has_time,
                 "period": dims.has_period},
        "columns": {col: {"unit": unit, "semantics": sem} for col, unit, sem in _measure_lines(stem)},
        "membership_set": (isinstance(OUTPUT_TRANSFORM.get(stem), Transform)
                           and OUTPUT_TRANSFORM[stem].semantics is Semantics.DIMENSION),
        "results_db": results_db_name(stem),
        "counterpart": dims.counterpart,
    }


def records(markdown_root: Path) -> list[dict]:
    """Every table's record, in page order (``results.md`` topic order, then "Other"; by name within a
    topic)."""
    markdown = (markdown_root / RESULTS_DOC).read_text(encoding="utf-8")
    found, _ = topics(markdown)
    order = topic_order(markdown) + [OTHER_TOPIC]
    csv = csv_names()
    recs = [record(stem, found.get(stem, OTHER_TOPIC), csv) for stem in stems()]
    return sorted(recs, key=lambda r: (order.index(r["topic"]), r["stem"]))


def _slug(title: str) -> str:
    """The python-markdown ``toc`` anchor of a heading."""
    s = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    return re.sub(r"[-\s]+", "-", s)


def render_catalogue(markdown_root: Path) -> str:
    """The results catalogue page (Markdown)."""
    recs = records(markdown_root)
    out = [
        "# Results catalogue",
        "",
        "Every result table FlexTool writes, generated from the code. A table is written to",
        "`output_parquet/<scenario>/<table>.parquet`, and with CSV output to `output_csv/<scenario>/` under the",
        "CSV name of the default output settings. The parquet's top column level is `scenario`; the levels",
        "below it are listed as column levels. Units are the default convention of FlexTool (MW, MWh, CUR);",
        "see [Model outputs](results.md) for what the results mean and how they are computed.",
        "",
    ]
    topic = None
    for r in recs:
        if r["topic"] != topic:
            topic = r["topic"]
            out += [f"## {topic}", ""]
            if topic != OTHER_TOPIC:
                out += [f"See [{topic}](results.md#{_slug(topic)}) in Model outputs.", ""]
        out += [f"### `{r['stem']}`", "", r["description"], ""]
        out.append(f"- Files: `{r['parquet']}`, `{r['csv']}`")
        rows = ", ".join(f"`{x}`" for x in r["dims"]["rows"]) or "none"
        cols = ", ".join(f"`{x}`" for x in r["dims"]["columns"]) or "none"
        if r["membership_set"]:
            out.append(f"- Membership set; its columns: {rows}")
        else:
            out.append(f"- Rows: {rows}; column levels: {cols}")
            if r["columns"]:
                for col, meta in r["columns"].items():
                    label = "every column" if col == "*" else f"`{col}`"
                    unit = meta["unit"] or "dimensionless"
                    out.append(f"- {label}: {unit}, {meta['semantics']}")
            else:
                out.append("- Units: not declared")
        if r["results_db"]:
            ec, param = r["results_db"].split(".", 1)
            out.append(f"- Results database: `{ec}` entity `{param}` parameter")
        else:
            out.append("- Results database: not written there")
        if r["counterpart"]:
            out.append(f"- Counterpart at the other time resolution: `{r['counterpart']}`")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def unmatched_bullets(markdown_root: Path) -> list[Bullet]:
    """The ``results.md`` bullets that name neither a results-database parameter of the spec nor a table."""
    return topics((markdown_root / RESULTS_DOC).read_text(encoding="utf-8"))[1]

