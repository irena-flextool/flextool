"""Time-structure fallback rules shared by the CSV path and the DB-side readers.

``TimelineConfig.create_assumptive_parts`` (``_timeline.py``) fills in the parts of the time structure a model
leaves out. The DB-side readers of a solve's period → timeset mapping (``_per_solve_sets``, ``_derived``, the axis
vocabulary of ``spinedb_backend._axis_enums``, the commodity-ladder periods) read the database directly, so they
apply the same rules through :func:`solve_period_timesets`:

* **Rule 1** -- no timeset and exactly one timeline: the timeset :data:`SYNTHETIC_TIMESET` exists.
* **Rule 2** -- no ``timeset_duration`` anywhere and exactly one timeline: every timeset has a block (the
  whole timeline), so every timeset is available to Rule 5b.
* **Rule 4** -- no ``model.solves``: the sole solve entity (``SolveConfig`` auto-wire), else, with one model
  entity, the one solve the solve parameters name.
* **Rule 5a** -- a solve with no realized or invest periods uses its model's ``periods_available``.
* **Rule 5b** -- a solve of the model (Rule 4's list, first model only) without ``period_timeset``, when exactly
  one timeset has a block: that timeset for each of the solve's periods, invest periods first, then realized
  periods (duplicates dropped; the CSV path keeps them, and every consumer reads the mapping as a set or
  first-occurrence list).

Contained (nested) solves and synthetic roll / nested sub-solve names are not covered by Rule 5b, so they get
``None`` unless they author a ``period_timeset``; their callers keep their own fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SYNTHETIC_TIMESET",
    "TimeStructure",
    "rule5_model_solves",
    "solve_period_timesets",
    "time_structure_from_backend",
]

#: Rule 1: the timeset FlexTool creates when the model has no timeset and exactly one timeline. It covers the
#: whole timeline (Rule 2) and every period of the solve (Rule 5b).
SYNTHETIC_TIMESET = "full_timeline"


@dataclass(frozen=True)
class TimeStructure:
    """The parts of a scenario's time structure the fallback rules read (plain names, database order)."""

    #: ``solve.period_timeset`` per solve: ``[(period, timeset), ...]`` in authored order.
    period_timeset: dict[str, list[tuple[str, str]]]
    #: ``model.solves`` per model, in model order.
    model_solves: list[tuple[str, list[str]]]
    #: Entity names of ``model``, ``solve`` and ``timeset``.
    models: list[str]
    solves: list[str]
    timesets: list[str]
    #: Timesets with an authored ``timeset_duration``.
    timesets_with_duration: frozenset[str]
    #: Number of timelines with an authored ``timestep_duration``.
    n_timelines: int
    #: ``solve.invest_periods`` / ``solve.realized_periods`` per solve (1-D arrays only).
    invest_periods: dict[str, list[str]] = field(default_factory=dict)
    realized_periods: dict[str, list[str]] = field(default_factory=dict)
    #: ``model.periods_available`` per model.
    periods_available: dict[str, list[str]] = field(default_factory=dict)
    #: Solves named by some ``solve.contains_solves``.
    contained_solves: frozenset[str] = frozenset()


def rule5_model_solves(ts: TimeStructure) -> tuple[str | None, list[str]]:
    """``(model, solves)`` whose solves Rule 5b covers: the first model's ``model.solves``, or Rule 4's choice
    (``SolveConfig``'s single-solve auto-wire under the model name ``"flextool"``, else the sole model with the
    one solve that ``period_timeset`` of a contained solve or the period parameters name). ``(None, [])`` where
    the CSV path raises instead."""
    if ts.model_solves:
        return ts.model_solves[0]
    if len(ts.solves) == 1:
        return "flextool", [ts.solves[0]]
    if len(ts.models) != 1:
        return None, []
    named = [s for s in ts.solves if s in ts.contained_solves and s in ts.period_timeset]
    named += [s for s in ts.solves if s in ts.realized_periods or s in ts.invest_periods]
    named = list(dict.fromkeys(named))
    return (ts.models[0], named) if len(named) == 1 else (None, [])


def solve_period_timesets(solve: str, ts: TimeStructure) -> list[tuple[str, str]] | None:
    """``[(period, timeset), ...]`` of *solve*: its authored ``period_timeset``, else the Rule 1 / 2 / 4 / 5a /
    5b default (see the module docstring), else ``None``."""
    authored = ts.period_timeset.get(solve)
    if authored:
        return list(authored)
    model, solves = rule5_model_solves(ts)
    if solve not in solves:
        return None
    candidates = list(ts.timesets)
    if not candidates and ts.n_timelines == 1:
        candidates = [SYNTHETIC_TIMESET]
    if ts.timesets_with_duration:
        with_block = [c for c in candidates if c in ts.timesets_with_duration]
    else:
        with_block = candidates if ts.n_timelines == 1 else []
    if len(with_block) != 1:
        return None
    periods = list(dict.fromkeys(ts.invest_periods.get(solve, []) + ts.realized_periods.get(solve, [])))
    if not periods:
        periods = list(dict.fromkeys(ts.periods_available.get(model, [])))
    if not periods:
        return None
    return [(d, with_block[0]) for d in periods]


# ---------------------------------------------------------------------------
# Backend adapter (``spinedb_backend.SpineDBBackend``: ``find_entities`` / ``find_parameter_values``)
# ---------------------------------------------------------------------------

def _byname(row: Any) -> str:
    ent = row.get("entity_byname")
    return ent[0] if isinstance(ent, (tuple, list)) else str(ent)


def _array_rows(backend: Any, entity_class: str, parameter: str) -> dict[str, list[str]]:
    """``{entity: [value, ...]}`` of a 1-D array parameter; other shapes (2-D maps) are left out."""
    out: dict[str, list[str]] = {}
    for row in backend.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter):
        values = getattr(row.get("parsed_value"), "values", None)
        if row.get("type") != "array" or values is None:
            continue
        out[_byname(row)] = [str(v) for v in values]
    return out


def time_structure_from_backend(backend: Any) -> TimeStructure:
    """The :class:`TimeStructure` of the scenario *backend* is bound to."""
    period_timeset: dict[str, list[tuple[str, str]]] = {}
    for row in backend.find_parameter_values(entity_class_name="solve", parameter_definition_name="period_timeset"):
        pv = row.get("parsed_value")
        if row.get("type") != "map" or not hasattr(pv, "indexes"):
            continue
        period_timeset[_byname(row)] = [(str(i), str(v)) for i, v in zip(pv.indexes, pv.values)]
    model_solves = list(_array_rows(backend, "model", "solves").items())
    contained = frozenset(s for values in _array_rows(backend, "solve", "contains_solves").values() for s in values)
    with_duration = frozenset(
        _byname(row) for row in backend.find_parameter_values(
            entity_class_name="timeset", parameter_definition_name="timeset_duration")
        if row.get("type") is not None)
    timelines = {_byname(row) for row in backend.find_parameter_values(
        entity_class_name="timeline", parameter_definition_name="timestep_duration") if row.get("type") is not None}

    def names(entity_class: str) -> list[str]:
        return [row["name"] for row in backend.find_entities(entity_class_name=entity_class)]

    return TimeStructure(
        period_timeset=period_timeset,
        model_solves=model_solves,
        models=names("model"),
        solves=names("solve"),
        timesets=names("timeset"),
        timesets_with_duration=with_duration,
        n_timelines=len(timelines),
        invest_periods=_array_rows(backend, "solve", "invest_periods"),
        realized_periods=_array_rows(backend, "solve", "realized_periods"),
        periods_available=_array_rows(backend, "model", "periods_available"),
        contained_solves=contained,
    )
