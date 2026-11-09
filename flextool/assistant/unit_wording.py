"""How the tools word units (O2, O14): the one place a tool result gets a unit text.

FlexTool does not check units.  MW, MWh, CUR (the model currency) and tonnes of CO2 are the documented default
convention, and a model may use others consistently (kg/h and kg for a hydrogen node).  So a unit made of
convention units is shown as ``"MW (default convention)"``; units that are fixed by FlexTool (hours, years,
fractions, counts) are shown plain.

The longer notes (:data:`UNITS_NOTE`, :data:`STORED_VALUES_NOTE`) are sent in full in a result until a request
that carried the full note was answered, and after that by their key only (:func:`add_session_notes`).  A note
counts as sent only when it reached the model (review G7, C3-n5): ``fit_json``'s last resort may drop it, a chat
result may be compacted, a question may be stopped before any request carried it.  The front ends settle it with
:func:`delivered_notes`: the chat from each answered request, the connection server from each result it returns.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
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


def add_session_notes(payload: dict, keys: Iterable[str], sent: set[str], in_flight: set[str] | None = None) -> dict:
    """Append the session notes *keys* to ``payload["notes"]``: the key when the note reached the model already
    (*sent*) or an earlier result of the same request carries it in full (*in_flight*, which this records), else
    the full text.  *sent* is only read: :func:`delivered_notes` settles it."""
    notes: list[Any] = list(payload.get("notes") or [])
    for key in keys:
        if key in sent or (in_flight is not None and key in in_flight):
            entry = key
        else:
            entry = SESSION_NOTES[key]
            if in_flight is not None:
                in_flight.add(key)
        if entry not in notes:
            notes.append(entry)
    payload["notes"] = notes
    return payload


def _sent_form(note: str, redact: Callable[[str], str]) -> str:
    """A note as it appears in a result's JSON text: redacted, JSON-escaped, without its quotes."""
    return json.dumps(redact(note), ensure_ascii=False)[1:-1]


def notes_in(text: str, redact: Callable[[str], str]) -> list[str]:
    """The session notes *text* holds in full, as sent (redacted)."""
    return [redact(note) for key, note in SESSION_NOTES.items() if _sent_form(note, redact) in text]


def delivered_notes(texts: Iterable[str], redact: Callable[[str], str]) -> set[str]:
    """The keys of the session notes that one of *texts* (what a request carried) holds in full."""
    found: set[str] = set()
    for text in texts:
        for key, note in SESSION_NOTES.items():
            if key not in found and _sent_form(note, redact) in text:
                found.add(key)
    return found


def restore_notes(texts: list[str], redact: Callable[[str], str]) -> list[str]:
    """*texts* (results carried together without their history) with the full note in place of the first key of a
    note that none of them holds in full, so a key never points to a note the model has not seen."""
    out = list(texts)
    delivered = delivered_notes(out, redact)
    for key, note in SESSION_NOTES.items():
        if key in delivered:
            continue
        token = json.dumps(key)
        at = next((i for i, t in enumerate(out) if token in t), None)
        if at is not None:
            out[at] = out[at].replace(token, json.dumps(redact(note), ensure_ascii=False), 1)
    return out
