"""How the tools word units (O2, O14): the one place a tool result gets a unit text.

FlexTool does not check units.  MW, MWh, CUR (the model currency) and tonnes of CO2 are the documented default
convention, and a model may use others consistently (kg/h and kg for a hydrogen node).  So a unit made of
convention units is shown as ``"MW (default convention)"``; units that are fixed by FlexTool (hours, years,
fractions, counts) are shown plain.

The longer notes (:data:`UNITS_NOTE`, :data:`STORED_VALUES_NOTE`) are sent in full in the first result of a session
that needs them, and after that by their key only (:func:`add_session_notes`).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

CONVENTION_SUFFIX = " (default convention)"

#: A part of a unit (split at "/", "·", spaces and parentheses) that belongs to the default convention.
_CONVENTION_PART = re.compile(r"(?:MW|kW|MVA)\w*|CUR|Mt|t|tCO2\w*")
_SPLIT = re.compile(r"[/·\s()]+")

UNITS_CONVENTION = "units_convention"
STORED_VALUES = "stored_values"

UNITS_NOTE = ("FlexTool does not check units. The units shown are its default convention (MW, MWh, CUR for the "
              "model currency, tonnes of CO2); this model may use others consistently, for example kg/h and kg "
              "for a hydrogen node. Time steps are in hours and lifetimes in years.")
STORED_VALUES_NOTE = ("These are the values stored in the input data. When FlexTool runs it may add defaults, "
                      "derive other values or rescale some (for example inflow with an inflow_method); the "
                      "results show what the model used.")

SESSION_NOTES: dict[str, str] = {UNITS_CONVENTION: UNITS_NOTE, STORED_VALUES: STORED_VALUES_NOTE}

#: Unit words the input tools give for markers that are not a unit of measure.
DIMENSIONLESS = "dimensionless"


def is_convention_unit(unit: str) -> bool:
    """Whether *unit* contains a part of the default convention (MW, kWh, CUR, t, tCO2, MVA ...)."""
    return any(_CONVENTION_PART.fullmatch(part) for part in _SPLIT.split(unit) if part)


def shown_unit(unit: str | None) -> str | None:
    """*unit* as a tool shows it: ``"MW (default convention)"`` for a convention unit, else as it is."""
    if unit is None:
        return None
    unit = unit.strip()
    if not unit:
        return None
    if unit.endswith(CONVENTION_SUFFIX):
        return unit
    return unit + CONVENTION_SUFFIX if is_convention_unit(unit) else unit


def add_session_notes(payload: dict, keys: Iterable[str], sent: set[str]) -> dict:
    """Append the session notes *keys* to ``payload["notes"]``: the full text the first time in a session
    (*sent* records which were sent), the key after that."""
    notes: list[Any] = list(payload.get("notes") or [])
    for key in keys:
        if key in sent:
            entry = key
        else:
            entry = SESSION_NOTES[key]
            sent.add(key)
        if entry not in notes:
            notes.append(entry)
    payload["notes"] = notes
    return payload
