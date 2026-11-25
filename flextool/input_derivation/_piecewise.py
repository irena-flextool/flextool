"""``unit.piecewise_linear_efficiency`` read from the scenario-filtered database values (input derivation).

:func:`parse_piecewise_curves` turns each unit's raw value into curve points and collects what is wrong with them,
with the messages of the design (``specs/piecewise_efficiency_final.md`` §5): the value shape (E2, E10), the load
keys (E3-E5), the efficiencies (E6), a falling fuel use (E7) and a falling incremental heat rate (E8, with the
nearest accepted curve).  The valid curves go into the long frame ``(p, d, load, eff)`` that the classifier
(``_process_method``) and the validator (``_validators.validate_piecewise_linear_inputs``) read, through the same
functions the engine uses (:mod:`flextool.engine.input._piecewise_curve`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl
import spinedb_api as api

from flextool.engine.input._piecewise_curve import (
    CONVEX_RTOL,
    CURVE_SCHEMA,
    LOAD_ONE_TOLERANCE,
    Point,
    hull_efficiency,
    lower_hull,
    normalise_load,
    segments,
)

CLASS = "unit"
PARAMETER = "piecewise_linear_efficiency"


@dataclass
class ParsedCurves:
    """The scenario's curves: ``frame`` holds the valid ones (``(p, d, load, eff)``, ``d`` None for one curve for
    every period), ``errors`` the messages of the others by unit, ``units`` every unit with a value."""

    frame: pl.DataFrame
    errors: dict[str, list[str]] = field(default_factory=dict)
    units: set[str] = field(default_factory=set)


def _label_path(value) -> tuple[str | None, ...]:
    """The raw ``index_name`` per level of a parsed value (``()`` for a constant), the first entry of each level
    giving the depth (as the engine reader's ``_index_name_path``)."""
    out: list[str | None] = []
    cur = value
    while True:
        if isinstance(cur, api.Map):
            out.append(cur.index_name or None)
            if not cur.values:
                break
            cur = cur.values[0]
            continue
        if isinstance(cur, api.TimeSeries):
            out.append("time")
        break
    return tuple(out)


def _g(x: float) -> str:
    return f"{x:g}"


def _eff(x: float) -> str:
    """An efficiency for a message: 3 significant digits, at least 2 decimals (0.30, 0.377)."""
    text = f"{x:.3g}"
    if "e" in text:
        return text
    decimals = len(text.split(".")[1]) if "." in text else 0
    return f"{x:.{max(decimals, 2)}f}"


def _curve_text(points: list[Point]) -> str:
    """A curve for a message, every point at full precision (``repr``), so a suggested curve pasted back into
    the database is exactly the one checked here."""
    return "{" + ", ".join(f"{float(load)!r}: {float(eff)!r}" for load, eff in points) + "}"


def curve_problems(points: list[tuple[object, object]]) -> tuple[list[str], list[Point] | None]:
    """The problems of one curve (raw ``(key, efficiency)`` pairs in authored order) and its sorted points when it
    has none.  Check order: E3-E6 (keys and efficiencies), then E7 (fuel falling), then E8 (convexity; the hull
    only for a curve that passes E7).  The texts follow the ``unit 'U' (period P): `` prefix."""
    if not points:
        return ["the curve has no points (it must end at full load, load share 1)"], None
    problems: list[str] = []
    parsed: list[Point] = []
    seen: set[float] = set()
    above_one = False
    for key, eff in points:
        try:
            load = float(key)
        except (TypeError, ValueError):
            load = math.nan
        if not (load > 0.0 and load <= 1.0 + LOAD_ONE_TOLERANCE):
            above_one = above_one or load > 1.0 + LOAD_ONE_TOLERANCE
            problems.append(f"load share '{key}' must be a number above 0 and at most 1 (the efficiency at zero "
                            "output is not defined; the fuel at zero output follows from the first point)")
            continue
        load = normalise_load(load)
        if load in seen:
            problems.append(f"load share {_g(load)} is given twice")
        seen.add(load)
        is_number = isinstance(eff, (int, float)) and not isinstance(eff, bool)
        if not is_number or not math.isfinite(eff) or eff <= 0.0:
            shown = _g(eff) if is_number else repr(eff)
            problems.append(f"efficiency at load {_g(load)} is {shown}; it must be above 0")
            continue
        parsed.append((load, float(eff)))
    # A key above 1 was already refused (E3): the curve's end is that key, so E5 would name a wrong end.
    if seen and max(seen) != 1.0 and not above_one:
        problems.append(f"the curve must end at full load (load share 1); it ends at {_g(max(seen))}")
    if problems:
        return problems, None
    pts = sorted(parsed)
    lines = segments(pts)
    for k, (alpha, _beta) in enumerate(lines):
        if alpha <= 0.0:
            (l0, e0), (l1, e1) = pts[k], pts[k + 1]
            return [f"the fuel use falls from load {_g(l0)} to {_g(l1)} (efficiency {e0!r} → {e1!r}): "
                    "more output for less fuel"], None
    for k in range(1, len(lines)):
        if lines[k][0] < lines[k - 1][0] * (1.0 - CONVEX_RTOL):
            hull = lower_hull(pts)
            kept = set(hull)
            raised = ", ".join(f"at load {_g(load)} from {_eff(eff)} to {_eff(hull_efficiency(hull, load))}"
                               for load, eff in pts if (load, eff) not in kept)
            return [f"the extra fuel per extra output falls from {lines[k - 1][0]:.3g} (load {_g(pts[k - 1][0])}–"
                    f"{_g(pts[k][0])}) to {lines[k][0]:.3g} (load {_g(pts[k][0])}–{_g(pts[k + 1][0])}) for the curve "
                    f"{_curve_text(pts)}; a linear model cannot use such a curve. Nearest accepted curve: "
                    f"{_curve_text(hull)} (it raises the efficiency {raised})"], None
    return [], pts


def _pairs(value) -> list[tuple[object, object]]:
    return list(zip(value.indexes, value.values))


def curves_of_value(value, unit: str = "", periods: set[str] | None = None,
                    ) -> tuple[list[tuple[str | None, list[Point]]], list[str]]:
    """The curves of one parsed ``piecewise_linear_efficiency`` value of ``unit``: the valid ones as ``(period,
    sorted points)`` (``period`` None for one curve for every period) and the messages of the others (the value
    shape, then each curve's :func:`curve_problems`).  ``periods``: the period names a curve per period may use
    (level 1 keys outside it are an E10 error); ``None`` skips that check.  Pure: no database."""
    from flextool.engine.input._param_shapes import entity_label_path_error

    errors: list[str] = []
    path = _label_path(value)
    shape_error = entity_label_path_error(CLASS, PARAMETER, {(unit,): path})
    if shape_error is not None:
        return [], [shape_error.rstrip(".")]
    curves: list[tuple[str | None, list[tuple[object, object]]]] = []
    if len(path) == 1:
        curves.append((None, _pairs(value)))
    else:
        for period, sub in _pairs(value):
            period = str(period)
            if not isinstance(sub, api.Map) or any(isinstance(v, api.Map) for v in sub.values):
                errors.append(f"unit '{unit}', parameter '{PARAMETER}': the value for period {period} is not a curve "
                              "(a map over load shares)")
                continue
            if periods is not None and period not in periods:
                errors.append(f"unit '{unit}', parameter '{PARAMETER}': level 1 of the value is a period, but "
                              f"'{period}' is not a period of the database")
                continue
            curves.append((period, _pairs(sub)))
    valid: list[tuple[str | None, list[Point]]] = []
    for period, pairs in curves:
        problems, points = curve_problems(pairs)
        where = f"unit '{unit}'" + (f" (period {period})" if period is not None else "")
        errors.extend(f"{where}: {p}" for p in problems)
        if points is not None:
            valid.append((period, points))
    return valid, errors


def parse_piecewise_curves(db, periods: set[str] | None = None) -> ParsedCurves:
    """The scenario's ``unit.piecewise_linear_efficiency`` curves of ``db`` (a scenario-filtered DatabaseMapping
    or a ``SpineDBBackend``).  ``periods``: the period names a curve per period may use (level 1 keys outside it are
    an E10 error); ``None`` skips that check."""
    rows: list[tuple[str, str | None, float, float]] = []
    errors: dict[str, list[str]] = {}
    values: dict[str, object] = {}
    for pv in db.find_parameter_values(entity_class_name=CLASS, parameter_definition_name=PARAMETER):
        if pv["type"] is not None:
            values[pv["entity_byname"][0]] = pv["parsed_value"]
    for unit in sorted(values):
        curves, problems = curves_of_value(values[unit], unit, periods)
        if problems:
            errors[unit] = problems
        for period, points in curves:
            rows.extend((unit, period, load, eff) for load, eff in points)
    frame = (pl.DataFrame(rows, schema=CURVE_SCHEMA, orient="row") if rows
             else pl.DataFrame(schema=CURVE_SCHEMA))
    return ParsedCurves(frame=frame, errors=errors, units=set(values))
