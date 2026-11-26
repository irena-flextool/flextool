"""The index names the v2 export writes for maps and arrays.

A name stored in the database stays.  Spine's default (``x`` for a Map, ``i`` for an Array) becomes the
parameter's meaningful name: ``array_index_name`` for an Array; for a Map the ``index_names`` table of
export_settings.yaml, else the parameter's ``_param_shapes`` axis (period, time, tier, load), else the sheet
layout's default (``index_name_defaults``).  Profile maps keep ``x``: the engine reads a profile's index name
(time / period / x = infer), and their meaningful names are a later change.
"""
from __future__ import annotations

from typing import Any

from spinedb_api import Array

# Set by export_to_excel from export_settings.yaml (configure).
_settings: dict[str, Any] = {"index_names": {}, "array_index_name": "sequence_index", "index_name_defaults": {}}

KEEP_DEFAULT_INDEX_NAME: frozenset[tuple[str, str]] = frozenset({("profile", "profile")})


def configure(settings: dict[str, Any]) -> None:
    """Take the index-name settings of export_settings.yaml."""
    _settings["index_names"] = settings.get("index_names", {})
    _settings["array_index_name"] = settings["array_index_name"]
    _settings["index_name_defaults"] = settings.get("index_name_defaults", {})


def array_index_name() -> str:
    """The index name of an Array stored without one of its own."""
    return _settings["array_index_name"]


def layout_index_name(layout: str) -> str:
    """The default index name of a sheet layout (periodic: period, timeseries: time)."""
    return _settings["index_name_defaults"].get(layout, "period")


def axis_label(axis: Any) -> str:
    """The index label of a ``_param_shapes`` :class:`AxisName` (d: period, t: time, i: tier, load: load)."""
    token = getattr(axis, "value", str(axis))
    return {"d": "period", "t": "time", "i": "tier"}.get(token, token)


def shape_level_names(entity_class: str, pname: str, depth: int) -> tuple[str, ...] | None:
    """The level names of the parameter's one ``_param_shapes`` shape with *depth* levels (``period``, ``load``),
    or None when it has none or several."""
    from flextool.engine.input._param_shapes import PARAM_ALLOWED_SHAPES, shape_to_axes
    paths = {shape_to_axes(s).map_levels for s in PARAM_ALLOWED_SHAPES.get((entity_class, pname), ())
             if len(shape_to_axes(s).map_levels) == depth}
    if len(paths) != 1:
        return None
    return tuple(axis_label(axis) for axis in next(iter(paths)))


def index_name_of(value: Any, entity_class: str, pname: str, layout: str) -> str:
    """The index name the export writes for a 1-level Map or an Array *value* (``None``: a Map not set yet)."""
    stored = value.index_name if value is not None else None
    if isinstance(value, Array):
        return stored if stored and stored != "i" else array_index_name()
    if stored and stored != "x":
        return stored
    if (entity_class, pname) in KEEP_DEFAULT_INDEX_NAME:
        return "x"
    named = _settings["index_names"].get(entity_class, {}).get(pname)
    if named:
        return named
    one_level = shape_level_names(entity_class, pname, 1)
    if one_level is not None:
        return one_level[0]
    return layout_index_name(layout)


def sheet_index_names(names: dict[Any, str]) -> tuple[str, dict[Any, str] | None]:
    """The index column label (the most common name) and, when the names differ, every key's name."""
    if not names:
        return "", None
    counts: dict[str, int] = {}
    for name in names.values():
        counts[name] = counts.get(name, 0) + 1
    label = max(counts, key=lambda n: (counts[n], n))
    return label, (dict(names) if len(counts) > 1 else None)
