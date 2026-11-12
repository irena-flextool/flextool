"""Placeholder arguments: the "no value" spellings a model writes for an optional argument it does not mean to set.

Small models fill every optional argument of a schema: ``scenario: ""``, ``member: ""``, ``filters: {}``,
``state: "none"``, ``top_n: 0`` (helpfulness analysis 2026-11-12: ``list_entities(member: "", name_contains: "")``
listed no units, ``state: ""`` was refused).  :func:`unset_placeholders` leaves such an argument out, so the call
means what a call without it means.  It runs in ``call_tool`` after the "arguments must be an object" check and
before the caps are clamped.  The repeated-call guard (:func:`flextool.assistant.session.call_signature`) does not
use it: it compares the model's own arguments.

Only the arguments listed per tool in :data:`EMPTIABLE` are touched, each by its :class:`Kind`.  Never touched: a
required argument, a boolean (``results_usable: false`` is a filter), ``group_by`` (an empty list means "combine
everything into one total"), ``get_slack_summary``'s ``min_value`` (0 is a threshold).  A caps argument
(``max_rows`` …) needs no entry: ``CapArg.resolve`` gives the default for a missing, non-numeric or ≤ 0 value.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class Kind(Enum):
    #: A name or a free-text filter: unset when null, empty, only spaces, or exactly ``*`` ("any").
    TEXT = "text"
    #: A choice from an enum: unset when null, empty, or ``none`` / ``null`` (any case).
    ENUM = "enum"
    #: A list: unset when null, ``""`` or ``[]``.  Its items stay as they are (``*`` patterns are patterns).
    LIST = "list"
    #: A map of level -> values (``filters``): entries whose value is null, ``""`` or ``[]`` are dropped; the map
    #: is unset when it is null, ``""`` or empty after that.
    MAP = "map"
    #: An object with required keys (``time_range``, ``sort``): unset whole when null, ``""``, ``{}``, or when a
    #: required key holds a placeholder (null, empty, only spaces) — half an object is not a request.
    OBJECT = "object"
    #: A count where 0 or less means nothing (``top_n``): unset when null, ``""`` or a number ≤ 0.
    POSITIVE = "positive"
    #: Any other optional value (``offset``, ``min_value``): unset only when null or ``""``.
    VALUE = "value"


T, E, L, M, OBJ, P, V = Kind.TEXT, Kind.ENUM, Kind.LIST, Kind.MAP, Kind.OBJECT, Kind.POSITIVE, Kind.VALUE
_QUERY = {"filters": M, "time_range": OBJ, "agg": E, "sort": OBJ, "top_n": P}

#: Per tool, the optional arguments that may be left out when they hold a placeholder, with how they are read.
#: ``tests/assistant/test_placeholders.py`` checks every entry against the tool's schema (an optional, non-boolean
#: property; never ``group_by``).
EMPTIABLE: dict[str, dict[str, Kind]] = {
    "search_docs": {"kind": E},
    "read_doc": {"section": T},
    "list_sources": {"scenario": T},
    "list_runs": {"source": T, "scenario": T, "state": E, "folders": L, "detail": E, "offset": V},
    "list_entities": {"scenario": T, "source": T, "name_contains": T, "member": T},
    "get_entity": {"scenario": T, "source": T},
    "get_parameter": {"parameters": L, "entities": L, "parameter": T, "entity": L, "scenario": T, "source": T,
                      "time_range": OBJ},
    "describe_parameter": {"parameter": T},
    "compare_inputs": {"source": T, "entity_class": T, "alternative": T},
    "list_result_tables": {"topic": T},
    "query_result": dict(_QUERY),
    "compare_runs": dict(_QUERY),
    "explain_entity": {"entity_class": E, "run": T, "scenario": T, "source": T},
    "get_run_log": {"grep": T, "solve": T},
    "get_slack_summary": {"min_value": V},
}

_NO_VALUE: tuple[Any, ...] = (None, "")


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _empty_list(value: Any) -> bool:
    return isinstance(value, list) and not value


def _placeholder(kind: Kind, value: Any, spec: dict) -> tuple[bool, Any]:
    """``(unset, value to keep)`` of one argument of *kind* (*spec*: its schema property)."""
    if kind is Kind.TEXT:
        return _blank(value) or (isinstance(value, str) and value.strip() == "*"), value
    if kind is Kind.ENUM:
        return _blank(value) or (isinstance(value, str) and value.strip().casefold() in ("none", "null")), value
    if kind is Kind.LIST:
        return value in _NO_VALUE or _empty_list(value), value
    if kind is Kind.MAP:
        if value in _NO_VALUE:
            return True, value
        if not isinstance(value, dict):
            return False, value             # left for the schema check to refuse
        kept = {k: v for k, v in value.items() if not (v in _NO_VALUE or _empty_list(v))}
        return not kept, kept
    if kind is Kind.OBJECT:
        if value in _NO_VALUE or value == {}:
            return True, value
        if not isinstance(value, dict):
            return False, value
        return any(key in value and _blank(value[key]) for key in spec.get("required", ())), value
    if kind is Kind.POSITIVE:
        return value in _NO_VALUE or (_number(value) and value <= 0), value
    return value in _NO_VALUE, value


def unset_placeholders(tool: str, args: dict, schema: dict) -> dict:
    """*args* of the tool *tool* (its input *schema*) without the placeholder arguments of :data:`EMPTIABLE`; a
    map argument keeps only its real entries.  Other arguments are never touched."""
    rules = EMPTIABLE.get(tool)
    if not rules:
        return args
    properties = schema.get("properties", {})
    required = set(schema.get("required", ()))
    out = dict(args)
    for key, kind in rules.items():
        if key not in out or key in required:
            continue
        unset, kept = _placeholder(kind, out[key], properties.get(key, {}))
        if unset:
            del out[key]
        else:
            out[key] = kept
    return out
