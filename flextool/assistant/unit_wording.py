"""How the tools word units (O2, O14): the one place a tool result gets a unit text.

A node's ``unit_of_measure`` (default MWh) sets the units of its amounts, flows (that unit per hour: MW for MWh)
and prices; CUR is the model currency and emissions are in tonnes of CO2.  FlexTool labels units and does not
convert them.  Since unit_of_measure (2026-11-11, owner) a unit is shown as the tool has it, without a caveat: the
label of a node without unit_of_measure is its declared default, not a guess.  A unit text of an older session or
recording that still ends in " (default convention)" is shown without that ending.

The longer notes (:data:`UNITS_NOTE`, :data:`STORED_VALUES_NOTE`) are sent in full in a result until a request
that carried the full note was answered, and after that by their key only (:func:`add_session_notes`).  A note
counts as sent only when it reached the model (review G7, C3-n5): ``fit_json``'s last resort may drop it, a chat
result may be compacted, a question may be stopped before any request carried it.  The front ends settle it with
:func:`delivered_notes`: the chat from each answered request, the connection server from each result it returns.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable
from typing import Any

#: The ending units had before unit_of_measure; :func:`shown_unit` removes it.
OLD_CONVENTION_SUFFIX = " (default convention)"

#: A part of a unit (split at "/", "·", spaces and parentheses) that belongs to the default convention.
_CONVENTION_PART = re.compile(r"(?:MW|kW|MVA)\w*|CUR|Mt|t|tCO2\w*")
_SPLIT = re.compile(r"[/·\s()]+")

#: A part of a unit that is power or energy (MW, MWh, kW, GWh, MVA, MW·s): a quantity of some carrier.
_ENERGY_PART = re.compile(r"[kMGT]?(?:W|VA)h?")

UNITS_CONVENTION = "units_convention"
STORED_VALUES = "stored_values"

UNITS_NOTE = ("Each node's unit_of_measure (default MWh) sets the units of its values: amounts in that unit, flows "
              "in that unit per hour (MW for MWh), prices per that unit. CUR is the model currency; emissions are in "
              "tonnes of CO2. Time steps are in hours and lifetimes in years.")
STORED_VALUES_NOTE = ("These are the values stored in the input data. When FlexTool runs it may add defaults, "
                      "derive other values or rescale some (for example inflow with an inflow_method); the "
                      "results show what the model used.")

SESSION_NOTES: dict[str, str] = {UNITS_CONVENTION: UNITS_NOTE, STORED_VALUES: STORED_VALUES_NOTE}

#: Unit words the input tools give for markers that are not a unit of measure.
DIMENSIONLESS = "dimensionless"


def is_convention_unit(unit: str) -> bool:
    """Whether *unit* contains a part of FlexTool's default units (MW, kWh, CUR, t, tCO2, MVA ...): a result with
    one carries :data:`UNITS_NOTE`."""
    return any(_CONVENTION_PART.fullmatch(part) for part in _SPLIT.split(unit) if part)


def is_energy_unit(unit: str | None) -> bool:
    """Whether *unit* is power or energy (``MW``, ``MWh/a``, ``MW·s``; not ``M CUR``, ``t/a`` or ``units``): such
    values are of the carrier of their node, and a model may hold several (electricity, heat, water with its own
    unit_of_measure, or hydrogen under the same default unit), so they do not add up across nodes."""
    return bool(unit) and any(_ENERGY_PART.fullmatch(part) for part in _SPLIT.split(unit) if part)


def shown_unit(unit: str | None) -> str | None:
    """*unit* as a tool shows it: stripped, without the ending of before unit_of_measure; ``None`` for none."""
    if unit is None:
        return None
    unit = unit.strip().removesuffix(OLD_CONVENTION_SUFFIX).strip()
    return unit or None


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


#: The units a value gets a scaled copy in (:func:`scaled`): each base unit's larger units and their factors.  A
#: copy is given only from the first larger unit up (CUR only from a million: M CUR is the model's cost unit).
LADDERS: dict[str, tuple[tuple[float, str], ...]] = {
    "MWh": ((1e3, "GWh"), (1e6, "TWh")),
    "MW": ((1e3, "GW"),),
    "t": ((1e3, "kt"), (1e6, "Mt")),
    "CUR": ((1e6, "M CUR"),),
}
#: The one ending a scaled unit may have after its base unit (a value per year).
SCALED_SUFFIXES = ("", "/a")


def _three_digits(x: float) -> float:
    return float(f"{x:.3g}")


def scaled(value: Any, unit: str | None, *, carrier: bool = False) -> str | None:
    """A rounded copy of *value* in a larger unit, for reading ("≈1.03 TWh/a" for 1 030 000 MWh/a), or ``None``.

    Only a unit that is a base unit of :data:`LADDERS` exactly, with no ending or "/a" (never a rate such as
    CUR/MWh, nor M CUR, Mt or MW·s), and only when the value is at least the first larger unit after rounding to
    three digits (999 960 MWh is ≈1 TWh, not ≈1e+03 GWh).  *carrier*: the unit is the unit_of_measure of a node,
    which is scaled only when it is power or energy (a node may count in t or another unit of its own)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not unit:
        return None
    unit = shown_unit(unit) or ""
    base, suffix = next(((b, unit[len(b):]) for b in sorted(LADDERS, key=len, reverse=True)
                         if unit.startswith(b) and unit[len(b):] in SCALED_SUFFIXES), (None, ""))
    if base is None or not is_convention_unit(unit) or (carrier and not is_energy_unit(unit)):
        return None
    for factor, name in reversed(LADDERS[base]):
        x = _three_digits(abs(value) / factor)
        if x >= 1:
            text = f"{x:.0f}" if x >= 1000 else f"{x:g}"
            return f"≈{'-' if value < 0 else ''}{text} {name}{suffix}"
    return None


def table_scaling(units: dict[str, str] | None, measures: Iterable[Any],
                  row_units: Any = None) -> tuple[str, bool] | None:
    """The one unit a table's values may be scaled in (:func:`scaled`) and whether it is a carrier's, or ``None``:
    every column shares one unit (*units* ``{"*": unit}``), it does not vary by row, and a node's unit_of_measure
    (a carrier measure) only when it is power or energy."""
    if not units or set(units) != {"*"} or row_units:
        return None
    unit = units["*"]
    carrier = any(getattr(m, "carrier", False) for m in measures)
    return (unit, carrier) if scaled(1e6, unit, carrier=carrier) else None
