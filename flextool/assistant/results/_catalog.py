"""What the assistant knows about each result table, from the output metadata of this FlexTool.

Every table FlexTool writes is declared in ``process_outputs._output_meta``: its one-sentence description
(``OUTPUT_DESCRIPTION``), its row and column levels and its counterpart at the other time resolution
(``OUTPUT_DIMS``) and the transform its values went through, which gives their unit and the kind of quantity
they are (``OUTPUT_TRANSFORM``: a rate at the time step, an annualized total, a level, a price ...).  The topic of
a table is the ``results.md`` section that names it (``docs_build.catalogue.records``, the same records the
results catalogue page shows).

:func:`measures` gives the unit and kind of every data column of a table as it was written: from the column metadata
in the file's own footer (the units the writer resolved, also for nodes with a ``unit_of_measure``), else from the
declared transform (for a table of several units, by the column's label), else unknown.  :func:`search` ranks the tables for a question (BM25 over name, description, topic,
columns and units).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from flextool.assistant.textsearch import BM25, term_counts
from flextool.lean_parquet import LeanLayout
from flextool.process_outputs._output_meta import (
    OUTPUT_DIMS,
    OUTPUT_TRANSFORM,
    Semantics,
    Transform,
    resolve_columns,
)

OTHER_TOPIC = "Other"


@dataclass(frozen=True)
class Measure:
    """The unit (as written, before the tools word it) and the kind of quantity of one data column."""

    unit: str
    semantics: str          # a ``Semantics`` value, or "unknown"
    #: A quantity of the carrier of its node (its transform's unit is a template of the node's unit of measure:
    #: flows, energies, storage, prices): such values do not add up over nodes, whose units may differ.
    carrier: bool = False


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


# ---------------------------------------------------------------------------------------------------------
# Search: BM25 with the words users ask in, and the main tables of a few common questions first
#
# Only here, not in ``textsearch`` (shared with the documentation search and its cached index).  Evidence:
# helpfulness analysis 2026-11-12 — "fuel" found no table, "invest" only the dual tables, "unserved energy" put
# start-up energy first and no slack table, "cost" the per-entity cost tables before the system totals.

#: Whole phrases of a question (lower case, at word boundaries) and the terms the tables use for them, added to
#: the query before it is tokenized.
SYNONYMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("fuel", "fuels"), "commodity commodity_cost"),
    (("invest", "investment", "investments", "invested", "built", "build", "expanded"), "capacity invested"),
    (("unserved", "not served", "loss of load", "lost load", "shortage"), "slack_up loss of load"),
    (("excess load",), "slack_down"),
)


@dataclass(frozen=True)
class Preferred:
    """Tables listed first, in this order, when the expanded query has a *trigger* phrase and the question none
    of the *unless* words (a more specific question: "capacity margin", "unit cost").  Among them, a table whose
    name has another word of the question comes first ("connection capacity")."""

    triggers: tuple[str, ...]
    tables: tuple[str, ...]
    unless: tuple[str, ...] = ()


_ENTITY_WORDS = ("unit", "units", "connection", "connections", "node", "nodes", "storage", "entity", "entities",
                 "process", "processes", "group", "groups")
PREFERRED: tuple[Preferred, ...] = (
    Preferred(("unserved", "not served", "loss of load", "lost load", "shortage"),
              ("node_slack_up_d_e", "node_slack_up_dt_e", "node_d_ep")),
    Preferred(("fuel", "fuels"), ("costs_dt_p", "annualized_costs_d_p")),
    Preferred(("capacity",), ("unit_capacity_ed_p", "connection_capacity_ed_p", "node_capacity_ed_p"),
              unless=("margin", "factor", "factors", "cf", "slack", "share", "dual", "marginal", "shadow")),
    Preferred(("cost", "costs", "objective"),
              ("annualized_costs_d_p", "costs_discounted_p_", "costs_discounted_d_p", "costs_dt_p"),
              unless=_ENTITY_WORDS + ("co2", "price", "prices", "fuel", "fuels")),
)
#: Words of a question that name the table a preference puts first ("storage capacity": the node's table).
_NAME_WORDS = {"storage": "node", "storages": "node"}
#: The factor on a dual table's score unless the question asks for one: the marginal values of investment rank
#: above the capacity tables on "invest" otherwise.  "value" is not one of the words ("value of lost load").
DUAL_FACTOR = 0.5
DUAL_WORDS = frozenset({"dual", "duals", "marginal", "shadow"})


def _has_phrase(text: str, phrase: str) -> bool:
    import re

    return re.search(rf"(?<![\w]){re.escape(phrase)}(?![\w])", text) is not None


def expanded_query(query: str) -> str:
    """*query* with the terms of its :data:`SYNONYMS` phrases added."""
    lower = " ".join(query.lower().split())
    extra = [terms for phrases, terms in SYNONYMS if any(_has_phrase(lower, p) for p in phrases)]
    return " ".join([query, *extra])


def _preferred(query: str, expanded: str, known: set[str]) -> list[str]:
    """The tables :data:`PREFERRED` puts first for *query*, in order, without repeats."""
    from flextool.assistant.textsearch import tokens

    lower, wide = " ".join(query.lower().split()), " ".join(expanded.lower().split())
    words = {_NAME_WORDS.get(t, t) for t in tokens(lower)}
    out: list[str] = []
    for pref in PREFERRED:
        if not any(_has_phrase(wide, t) for t in pref.triggers) or any(_has_phrase(lower, u) for u in pref.unless):
            continue
        trigger_words = {w for t in pref.triggers for w in tokens(t)}
        named = words - trigger_words
        ranked = sorted(pref.tables, key=lambda stem: (-len(named & set(stem.lower().split("_"))),
                                                       pref.tables.index(stem)))
        out += [stem for stem in ranked if stem in known and stem not in out]
    return out


def search(query: str, max_results: int) -> list[dict]:
    """The records that best match *query*, best first: the :data:`PREFERRED` tables of the question, then the
    BM25 ranking of the query with its :data:`SYNONYMS`, a dual table's score times :data:`DUAL_FACTOR` unless
    the question asks for duals."""
    from flextool.assistant.textsearch import tokens

    recs = records()
    assert _index is not None
    expanded = expanded_query(query)
    by_stem = {r["stem"]: i for i, r in enumerate(recs)}
    first = _preferred(query, expanded, set(by_stem))
    scores = _index.scores(expanded)
    if not DUAL_WORDS & set(tokens(query)):
        scores = {i: sc * DUAL_FACTOR if recs[i]["stem"].startswith("dual_") else sc for i, sc in scores.items()}
    ranked = [i for i, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
    order = [by_stem[stem] for stem in first] + [i for i in ranked if recs[i]["stem"] not in first]
    return [recs[i] for i in order[:max_results]]


def declared_measures(stem: str) -> dict[str, Measure] | None:
    """``{column label or "*": Measure}`` of a declared table (``"*"`` for every column of a uniform table);
    ``{}`` for a membership set; ``None`` for an undeclared table."""
    spec = OUTPUT_TRANSFORM.get(stem)
    if spec is None:
        return None
    if isinstance(spec, Transform):
        if spec.semantics is Semantics.DIMENSION:
            return {}
        return {"*": Measure(spec.unit, spec.semantics.value, bool(spec.template))}
    return {label: Measure(t.unit, t.semantics.value, bool(t.template)) for label, t in spec.items()
            if t.semantics is not Semantics.DIMENSION}


def _footer_key(key: tuple[str, ...], layout: LeanLayout) -> str:
    """The key under which ``write_lean_parquet`` stored a column's metadata: the column label below the
    ``scenario`` level, as text (``derive_column_meta`` keys by ``str`` of the label)."""
    rest = tuple(v for level, v in zip(layout.col, key) if level != "scenario")
    if len(rest) == 1:
        return rest[0]
    return str(rest)


def _written(footer: dict[str, Any], key: tuple[str, ...], layout: LeanLayout) -> dict | None:
    """The footer metadata of a data column (a scenario's file keys it without the ``scenario`` level, a
    comparison file with it), ``None`` when the footer has none or calls it a dimension."""
    for k in (_footer_key(key, layout), str(tuple(key)) if len(key) > 1 else None):
        meta = footer.get(k) if k is not None else None
        if isinstance(meta, dict) and meta.get("semantics") and meta.get("semantics") != "dimension":
            return meta
    return None


def measures(stem: str, layout: LeanLayout) -> list[Measure]:
    """The :class:`Measure` of every data column of a table (in the order of ``layout.physical``).  A column whose
    unit follows its nodes' ``unit_of_measure`` takes the unit the writer put in the footer (review S8: one source
    of those units); every other column the declared transform's (by the column's label for a table of several
    units), and an undeclared one the footer's."""
    declared = declared_measures(stem)
    footer: dict[str, Any] = layout.column_meta or {}
    uniform = declared["*"] if declared is not None and set(declared) == {"*"} else None
    if uniform is not None and not footer:
        return [uniform] * len(layout.physical)
    # A mixed-unit table: the unit of each column from the same resolver as the result files (its label level).
    resolved = (resolve_columns(stem, list(layout.columns), level_names=list(layout.col))
                if declared and uniform is None and layout.columns else None)
    out: list[Measure] = []
    for i, key in enumerate(layout.columns):
        if resolved is not None and resolved[i].semantics is not Semantics.DIMENSION:
            base = Measure(resolved[i].unit, resolved[i].semantics.value,
                           bool(_column_template(stem, key, layout)))
        elif uniform is not None:
            base = uniform
        else:
            base = None
        # The footer gives the unit of a column that follows its nodes (and of an undeclared one); the others keep
        # this FlexTool's declared unit, so a unit fixed since an older run wrote its footer shows fixed.
        meta = _written(footer, key, layout) if base is None or base.carrier else None
        if meta is not None:
            out.append(Measure(str(meta.get("unit") or ""), base.semantics if base is not None
                               else str(meta["semantics"]), base.carrier if base is not None else False))
        else:
            out.append(base if base is not None else UNKNOWN)
    return out


def _column_template(stem: str, key: tuple[str, ...], layout: LeanLayout) -> str:
    """The unit template of a column of a table of several units (``""`` without one)."""
    spec = OUTPUT_TRANSFORM.get(stem)
    if not isinstance(spec, dict):
        return getattr(spec, "template", "") or ""
    for label in reversed(key):
        tf = spec.get(str(label))
        if tf is not None:
            return tf.template
    tf = spec.get("*")
    return tf.template if tf is not None else ""


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
