"""What the assistant knows about each result table, from the output metadata of this FlexTool.

Every table FlexTool writes is declared in ``process_outputs._output_meta``: its one-sentence description
(``OUTPUT_DESCRIPTION``), its row and column levels and its counterpart at the other time resolution
(``OUTPUT_DIMS``) and the transform its values went through, which gives their unit and the kind of quantity
they are (``OUTPUT_TRANSFORM``: a rate at the time step, an annualized total, a level, a price ...).  The topic of
a table is the ``results.md`` section that names it (``docs_build.catalogue.records``, the same records the
results catalogue page shows).

:func:`measures` gives the unit and kind of every data column of a table as it was written: from the declared
transform (for a table of several units, by the column's label), else from the column metadata in the file's own
footer, else unknown.  :func:`search` ranks the tables for a question (BM25 over name, description, topic,
columns and units).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from flextool.assistant.textsearch import BM25, term_counts
from flextool.lean_parquet import LeanLayout
from flextool.process_outputs._output_meta import OUTPUT_DIMS, OUTPUT_TRANSFORM, Semantics, Transform

OTHER_TOPIC = "Other"


@dataclass(frozen=True)
class Measure:
    """The unit (as written, before the tools word it) and the kind of quantity of one data column."""

    unit: str
    semantics: str          # a ``Semantics`` value, or "unknown"


UNKNOWN = Measure("", "unknown")

_lock = threading.Lock()
_records: list[dict] | None = None
_index: BM25 | None = None


def _build_records() -> list[dict]:
    from flextool.assistant import docs_store
    from flextool.docs_build import catalogue

    root = docs_store.markdown_root()
    if root is not None and (root / catalogue.RESULTS_DOC).is_file():
        return catalogue.records(root)
    csv = catalogue.csv_names()
    return [catalogue.record(stem, OTHER_TOPIC, csv) for stem in catalogue.stems()]


def records() -> list[dict]:
    """The catalogue record of every declared table (``docs_build.catalogue.record``), in page order."""
    global _records, _index
    with _lock:
        if _records is None:
            _records = _build_records()
            _index = BM25(term_counts(_search_text(r), r["stem"]) for r in _records)
        return _records


def record(stem: str) -> dict | None:
    return next((r for r in records() if r["stem"] == stem), None)


def topic_of(stem: str) -> str:
    rec = record(stem)
    return rec["topic"] if rec is not None else OTHER_TOPIC


def _search_text(rec: dict) -> str:
    columns = " ".join(f"{col} {c['unit']} {c['semantics']}" for col, c in rec["columns"].items() if col != "*")
    units = " ".join(c["unit"] + " " + c["semantics"] for c in rec["columns"].values())
    dims = " ".join(rec["dims"]["rows"] + rec["dims"]["columns"])
    return " ".join(filter(None, [rec["stem"], rec["topic"], rec["description"], columns, units, dims,
                                  rec.get("results_db") or ""]))


def search(query: str, max_results: int) -> list[dict]:
    """The records that best match *query*, best first."""
    recs = records()
    assert _index is not None
    return [recs[i] for i, _ in _index.search(query, max_results)]


def declared_measures(stem: str) -> dict[str, Measure] | None:
    """``{column label or "*": Measure}`` of a declared table (``"*"`` for every column of a uniform table);
    ``{}`` for a membership set; ``None`` for an undeclared table."""
    spec = OUTPUT_TRANSFORM.get(stem)
    if spec is None:
        return None
    if isinstance(spec, Transform):
        if spec.semantics is Semantics.DIMENSION:
            return {}
        return {"*": Measure(spec.unit, spec.semantics.value)}
    return {label: Measure(t.unit, t.semantics.value) for label, t in spec.items()
            if t.semantics is not Semantics.DIMENSION}


def _footer_key(key: tuple[str, ...], layout: LeanLayout) -> str:
    """The key under which ``write_lean_parquet`` stored a column's metadata: the column label below the
    ``scenario`` level, as text (``derive_column_meta`` keys by ``str`` of the label)."""
    rest = tuple(v for level, v in zip(layout.col, key) if level != "scenario")
    if len(rest) == 1:
        return rest[0]
    return str(rest)


def measures(stem: str, layout: LeanLayout) -> list[Measure]:
    """The :class:`Measure` of every data column of a table (in the order of ``layout.physical``)."""
    declared = declared_measures(stem)
    if declared is not None and "*" in declared:
        return [declared["*"]] * len(layout.physical)
    footer: dict[str, Any] = layout.column_meta or {}
    out: list[Measure] = []
    for key in layout.columns:
        label = key[-1] if key else ""
        if declared is not None and label in declared:
            out.append(declared[label])
            continue
        meta = footer.get(_footer_key(key, layout))
        if isinstance(meta, dict) and meta.get("semantics") and meta.get("semantics") != "dimension":
            out.append(Measure(str(meta.get("unit") or ""), str(meta["semantics"])))
        else:
            out.append(UNKNOWN)
    return out


def membership_set(stem: str, layout: LeanLayout) -> bool:
    """Whether the table is a set of names with no values (a declared membership set or a flat table)."""
    return declared_measures(stem) == {} or not layout.col


def counterpart(stem: str) -> str | None:
    dims = OUTPUT_DIMS.get(stem)
    return dims.counterpart if dims is not None else None


def description(stem: str) -> str | None:
    rec = record(stem)
    return rec["description"] if rec is not None else None


def reset() -> None:
    global _records, _index
    with _lock:
        _records = None
        _index = None
