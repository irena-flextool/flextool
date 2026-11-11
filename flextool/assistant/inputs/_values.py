"""A stored value as the tools show it: the value itself or a summary, and its shape.

**Summary** (:func:`summarize`): a constant or a string (≤ 500 characters) is shown as it is; a map, a time series
or an array becomes rows ``[index…, value]`` (maps through ``spinedb_api.convert_map_to_table``).  Up to
``max_rows`` rows are shown as a table; a longer value as ``{index_names, n, min, mean, max, first, last,
distinct_first_level}``.

**Shape** (:func:`shape_of`, P0 §4.3), in this order: (1) a parameter the shape registry
(``_param_shapes.PARAM_ALLOWED_SHAPES``) does not list is described by its stored index names and the map ranks
the shipped schema allows; (2) an unnamed index level that the registry cannot settle is decided by FlexTool from
the index values when it runs (``_param_shapes.resolve_param_shape``), so the tool says so instead of guessing;
(3) otherwise ``_param_shapes.check_index_name_path`` names the shape, or its error says what is wrong.
"""
from __future__ import annotations

import datetime
import json
import math
from typing import Any

from flextool.assistant.caps import cap_text
from flextool.assistant.errors import ToolError

STRING_MAX = 500
DISTINCT_MAX = 20
RUN_TIME_NOTE = ("The index levels are not named; FlexTool decides period or time from the index values when it "
                 "runs.")


def _plain(x: Any) -> Any:
    """A JSON-friendly copy of one index or leaf; a datetime (a time series index, a spinedb ``DateTime``) in ISO
    form, ``2020-01-01T00:00:00``."""
    try:
        import numpy as np

        if isinstance(x, np.generic):
            x = x.item()
    except ImportError:
        pass
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (datetime.date, datetime.time)):
        return x.isoformat()
    from spinedb_api import DateTime

    if isinstance(x, DateTime):
        return x.value.isoformat()
    return str(x)


def is_indexed(value: Any) -> bool:
    from spinedb_api.parameter_value import IndexedValue

    return isinstance(value, IndexedValue)


def kind_of(value: Any) -> str:
    from spinedb_api import Array, Map, TimePattern, TimeSeries

    if isinstance(value, Map):
        return "map"
    if isinstance(value, TimeSeries):
        return "time_series"
    if isinstance(value, TimePattern):
        return "time_pattern"
    if isinstance(value, Array):
        return "array"
    if isinstance(value, str):
        return "text"
    if value is None:
        return "none"
    return "constant"


def index_names(value: Any) -> list[str | None]:
    """The ``index_name`` of each level of an indexed value, outermost first (``[]`` for a constant)."""
    from spinedb_api import Map

    if not is_indexed(value):
        return []
    names: list[str | None] = [value.index_name]
    if isinstance(value, Map):
        nested = next((v for v in value.values if isinstance(v, Map)), None)
        if nested is not None:
            names += index_names(nested)
    return names


def rows_of(value: Any) -> list[list[Any]]:
    """``[[index…, leaf]]`` of an indexed value (ragged maps padded with ``None`` at the end of the indexes)."""
    from spinedb_api import Map
    from spinedb_api.parameter_value import convert_map_to_table

    if isinstance(value, Map):
        rows = convert_map_to_table(value, make_square=False)
        depth = max((len(r) for r in rows), default=0)
        return [[_plain(x) for x in r[:-1]] + [None] * (depth - len(r)) + [_plain(r[-1])] for r in rows]
    return [[_plain(i), _plain(v)] for i, v in zip(value.indexes, value.values)]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _time_level(names: list[str | None]) -> int:
    """The index level a ``time_range`` filters: the one named ``time`` / ``t``, else the innermost."""
    for i, n in enumerate(names):
        if isinstance(n, str) and n.strip().lower() in ("time", "t"):
            return i
    return max(len(names) - 1, 0)


def _as_datetime(text: Any) -> datetime.datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


def _names_bound(value: Any, bound: str) -> bool:
    """Whether the index value *value* is the one *bound* names: the same text, or the same datetime
    (``2020-01-01T00:00``, ``2020-01-01 00:00`` and ``2020-01-01`` all name midnight)."""
    if str(value) == bound:
        return True
    when = _as_datetime(bound)
    return when is not None and _as_datetime(value) == when


def select_time_range(rows: list[list[Any]], level: int, lo: str, hi: str) -> list[list[Any]]:
    """The rows whose index at *level* is from *lo* to *hi*, both included, by position in the series (as the
    results tools select): the level's distinct values in order of first appearance, the slice between the two
    bounds (reversed bounds swapped), every row with a value in it.  Rows without a value at *level* (a ragged
    map) are dropped.  A bound the level does not have raises ``bad_input_time_step``."""
    values = list(dict.fromkeys(r[level] for r in rows if r[level] is not None))
    positions = []
    for bound in (lo, hi):
        at = next((i for i, v in enumerate(values) if _names_bound(v, bound)), None)
        if at is None:
            raise ToolError("bad_input_time_step", value=bound, first=values[0] if values else "-",
                            last=values[-1] if values else "-")
        positions.append(at)
    a, b = positions
    kept = set(values[min(a, b):max(a, b) + 1])
    return [r for r in rows if r[level] is not None and r[level] in kept]


def parse_time_range(time_range: Any) -> tuple[str, str] | None:
    if time_range is None:
        return None
    if isinstance(time_range, dict) and isinstance(time_range.get("from"), str) and isinstance(
            time_range.get("to"), str):
        return time_range["from"], time_range["to"]
    raise ToolError("bad_time_range", example='{"from": "t0001", "to": "t0168"}')


def summarize(value: Any, *, max_rows: int, time_range: tuple[str, str] | None = None) -> dict:
    """``{"value": …}`` for a constant or a short string; for an indexed value its rows as a table or a summary."""
    if not is_indexed(value):
        out = _plain(value)
        if isinstance(out, str):
            out = cap_text(out, STRING_MAX)
        return {"value": out}
    names = index_names(value)
    rows = rows_of(value)
    out: dict[str, Any] = {"kind": kind_of(value), "index_names": names}
    if time_range is not None:
        level = _time_level(names)
        lo, hi = time_range
        rows = select_time_range(rows, level, lo, hi)
        out["time_range"] = {"from": lo, "to": hi, "level": level}
    out["n"] = len(rows)
    if len(rows) <= max_rows:
        out["table"] = rows
        return out
    leaves = [r[-1] for r in rows]
    numbers = [float(x) for x in leaves if _is_number(x)]
    if numbers:
        out.update(min=min(numbers), mean=sum(numbers) / len(numbers), max=max(numbers))
        if len(numbers) < len(leaves):
            out["non_numeric"] = len(leaves) - len(numbers)
    out["first"] = rows[0]
    out["last"] = rows[-1]
    first_level = list(dict.fromkeys(r[0] for r in rows))
    out["distinct_first_level"] = first_level[:DISTINCT_MAX]
    if len(first_level) > DISTINCT_MAX:
        out["distinct_first_level_total"] = len(first_level)
    return out


def compact(value: Any) -> Any:
    """A one-line form for listings: the constant itself, else ``{kind, index_names, n, min, mean, max}``."""
    if not is_indexed(value):
        out = _plain(value)
        return cap_text(out, STRING_MAX) if isinstance(out, str) else out
    rows = rows_of(value)
    numbers = [float(r[-1]) for r in rows if _is_number(r[-1])]
    out: dict[str, Any] = {"kind": kind_of(value), "index_names": index_names(value), "n": len(rows)}
    if numbers:
        out.update(min=min(numbers), mean=sum(numbers) / len(numbers), max=max(numbers))
    elif rows:
        out["first"] = rows[0]
    return out


def _canonical(x: Any) -> Any:
    """*x* (a :func:`_plain` leaf or index) in one form for equal values: a number as a float (``1`` and ``1.0``
    are one value, as :func:`same_value` has them)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x) if math.isfinite(float(x)) else str(x)
    return x


def value_key(value: Any) -> str:
    """A stable hash of the whole parsed *value*: its kind, index names and every row of an indexed value, the
    uncut text of a string.  Two values have one key only when they are the same value (:func:`same_value`), so
    a listing may group entities on it (:func:`compact` is a summary: values with the same length, min, mean and
    max share it)."""
    import hashlib

    if is_indexed(value):
        form: Any = [kind_of(value), index_names(value), [[_canonical(x) for x in r] for r in rows_of(value)]]
    else:
        form = ["scalar", type(value).__name__ if isinstance(value, (str, bool)) else "", _canonical(_plain(value))]
    text = json.dumps(form, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def same_value(a: Any, b: Any) -> bool:
    """Whether two parsed values are equal (``1`` equals ``1.0``)."""
    if is_indexed(a) or is_indexed(b):
        if not (is_indexed(a) and is_indexed(b)):
            return False
        return index_names(a) == index_names(b) and rows_of(a) == rows_of(b)
    if _is_number(a) and _is_number(b):
        return float(a) == float(b)
    return _plain(a) == _plain(b)


# ---------------------------------------------------------------------------------------------------------
# Shape


def _axis_names() -> dict[str, str]:
    """``{index name: axis label}`` from ``schemas/flextool_axis_contract.json``: each axis's name, label and
    column synonyms."""
    from flextool._resources import package_data_path

    with open(package_data_path("schemas/flextool_axis_contract.json"), encoding="utf-8") as f:
        contract = json.load(f)
    out: dict[str, str] = {}
    for axis in contract.get("axes", []):
        label = axis.get("label") or axis["name"]
        for name in (axis["name"], label, *(axis.get("column_synonyms") or [])):
            out.setdefault(str(name).lower(), label)
    out.setdefault("time", out.get("t", "timestep"))
    return out


def _levels_text(n: int) -> str:
    return "1 index level" if n == 1 else f"{n} index levels"


def stored_shape(value: Any) -> str:
    """The shape of a stored value in words, with an article as ``allowed_shapes_in_words`` has: ``a constant``,
    ``an array, 1 index level (sequence_index)`` or ``a map, 2 index levels (period, x)``."""
    kind = kind_of(value)
    if kind in ("constant", "text", "none"):
        return "a constant"
    names = index_names(value)
    shown = ", ".join("(unnamed)" if not n else str(n) for n in names)
    article = "an" if kind == "array" else "a"
    return f"{article} {kind.replace('_', ' ')}, {_levels_text(len(names))} ({shown})"


def shape_of(entity_class: str, parameter: str, entity: str, value: Any) -> tuple[str, str | None]:
    """``(shape, shape_note)`` of a stored value (P0 §4.3, in order)."""
    from flextool.assistant.inputs._schema import ranks_in_words, shipped_schema
    from flextool.engine.input._param_shapes import (
        PARAM_ALLOWED_SHAPES,
        _FACET_SHAPES,
        _SHAPE_LABELS,
        _infer_silent_default_labels,
        _label_path_problem,
        PERIOD_TIME_PARAMS,
        allowed_shapes_in_words,
        check_index_name_path,
    )
    from flextool.engine.input._solve_state import FlexToolConfigError

    labels = index_names(value)
    key = (entity_class, parameter)
    allowed = PARAM_ALLOWED_SHAPES.get(key)
    if allowed is None:                                                            # (1)
        types = shipped_schema().value_types.get(key, ())
        note = f"FlexTool reads this parameter as {ranks_in_words(types)}."
        axes = _axis_names()
        known = [f"{n} = {axes[str(n).strip().lower()]}" for n in labels
                 if n and str(n).strip().lower() in axes]
        if known:
            note += f" Index names it knows as axes: {', '.join(known)}."
        return stored_shape(value), note
    depth = len(labels)
    candidates = [s for s in allowed if len(_SHAPE_LABELS[s]) == depth]
    if candidates and all(s in _FACET_SHAPES for s in candidates):               # price-ladder tiers
        problem = _label_path_problem(tuple(labels), allowed, key in PERIOD_TIME_PARAMS)
        if problem is None:
            return allowed_shapes_in_words(set(candidates)), None
        return stored_shape(value), f"{problem}. This parameter takes {allowed_shapes_in_words(allowed)}."
    if candidates and any(n is None for n in _infer_silent_default_labels(list(labels), allowed)):   # (2)
        return stored_shape(value), RUN_TIME_NOTE
    try:                                                                            # (3)
        shape = check_index_name_path(entity_class, parameter, entity, list(labels))
    except FlexToolConfigError as exc:
        return stored_shape(value), str(exc)
    return allowed_shapes_in_words({shape}), None
