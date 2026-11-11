"""Every parameter description starts with its unit bracket, and the bracket fits the
parameter's autoscale quantity (``QUANTITY_UNITS``) or its ``UNIT_OVERRIDES`` entry.

The bracket is the description up to its first ``]``.  A ``DIMENSIONLESS`` parameter never
has a bare ``[-]`` without an override (the autoscale type says "not scaled", not "a ratio");
an ``UNKNOWN`` one has only ``[MW or MWh]``.
"""
from __future__ import annotations

import json

import pytest

from flextool._resources import package_data_path
from flextool.engine.autoscale._quantity_types import (
    PARAMETER_TYPES,
    QUANTITY_UNITS,
    UNIT_OVERRIDES,
    QuantityType,
)


def _definitions() -> dict[tuple[str, str], str]:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    # Keyed like PARAMETER_TYPES: (parameter, class).
    return {(row[1], row[0]): row[4] or "" for row in schema["parameter_definitions"]}


DEFINITIONS = _definitions()


def _bracket(description: str) -> str | None:
    if not description.startswith("[") or "]" not in description:
        return None
    return description[: description.index("]") + 1]


@pytest.mark.parametrize("key", sorted(DEFINITIONS), ids=lambda k: f"{k[1]}.{k[0]}")
def test_description_starts_with_a_valid_unit(key: tuple[str, str]) -> None:
    bracket = _bracket(DEFINITIONS[key])
    assert bracket is not None, f"{key[1]}.{key[0]} has no unit bracket: {DEFINITIONS[key]!r}"
    if key in UNIT_OVERRIDES:
        assert bracket == UNIT_OVERRIDES[key]
        return
    quantity = PARAMETER_TYPES[key]
    assert bracket in QUANTITY_UNITS[quantity], (
        f"{key[1]}.{key[0]}: {bracket} is not a {quantity.name} unit "
        f"({sorted(QUANTITY_UNITS[quantity])}); add it there or to UNIT_OVERRIDES")


def test_dimensionless_never_accepts_a_bare_ratio() -> None:
    assert "[-]" not in QUANTITY_UNITS[QuantityType.DIMENSIONLESS]
    assert QUANTITY_UNITS[QuantityType.UNKNOWN] == frozenset({"[MW or MWh]"})
    assert QUANTITY_UNITS[QuantityType.FRACTION] == frozenset({"[-]"})


def test_every_quantity_has_its_units() -> None:
    assert set(QUANTITY_UNITS) == set(QuantityType)


def test_overrides_name_schema_parameters_and_are_needed() -> None:
    """An override that names no definition, or that its quantity already accepts, is stale."""
    for key, bracket in UNIT_OVERRIDES.items():
        assert key in DEFINITIONS, f"UNIT_OVERRIDES {key} is not a schema parameter"
        assert bracket not in QUANTITY_UNITS[PARAMETER_TYPES[key]], f"UNIT_OVERRIDES {key} is not needed"


# ---------------------------------------------------------------------------------------------------------
# How the assistant tools word units, and the notes on parameters (P0 §4.3, §4.4)


@pytest.mark.parametrize("unit,shown", [
    ("MW·s", "MW·s"),
    ("Mt", "Mt"),
    ("M CUR/a", "M CUR/a"),
    (" CUR/kWh ", "CUR/kWh"),
    ("tCO2/MWh", "tCO2/MWh"),
    ("MW (default convention)", "MW"),                 # a unit text of before unit_of_measure
    ("", None),
    ("units/a", "units/a"),
    ("h", "h"),
    ("fraction", "fraction"),
    ("p.u.", "p.u."),
    ("hours", "hours"),
])
def test_shown_unit_is_the_unit_as_given(unit: str, shown: str | None) -> None:
    """Since unit_of_measure a unit is shown as it is: the label of a node without unit_of_measure is its declared
    default (MWh), so it needs no caveat."""
    from flextool.assistant.unit_wording import shown_unit

    assert shown_unit(unit) == shown


def test_no_unit_wording_outside_unit_wording_py() -> None:
    from pathlib import Path

    import flextool.assistant

    root = Path(flextool.assistant.__file__).parent
    offenders = [p.relative_to(root).as_posix() for p in root.rglob("*.py")
                 if p.name != "unit_wording.py" and "default convention" in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_every_trap_names_a_schema_parameter() -> None:
    from flextool.assistant.inputs._notes import traps

    keys = {(cls, param) for (param, cls) in DEFINITIONS}
    assert set(traps()) <= keys


def test_option_traps_are_the_options_a_database_parameter_sets() -> None:
    from flextool.assistant.inputs._notes import option_trap_text, traps
    from flextool.run_options import OPTIONS

    expected = {tuple(spec.db.split(".", 1)): option_trap_text(spec.flag, spec.env)
                for spec in OPTIONS.values() if spec.db}
    found = {key: t.text for key, ts in traps().items() for t in ts if "overrides this value" in t.text}
    assert found == expected
    assert all("()" not in text and "None" not in text for text in found.values())


def test_per_kw_traps_are_the_per_kw_parameters() -> None:
    from flextool.assistant.inputs._notes import traps

    per_kw = {(cls, param) for (param, cls), desc in DEFINITIONS.items() if desc.startswith("[CUR/kW")}
    found = {key for key, ts in traps().items() for t in ts if "multiplies the value by 1000" in t.text}
    # The ×1000 parameters: invest_cost, fixed_cost and salvage_value of units, connections and nodes, and the
    # group penalty_capacity_margin (P0 §1, verified in the engine).
    assert found == per_kw == {
        (cls, p) for cls in ("unit", "connection", "node") for p in ("invest_cost", "fixed_cost", "salvage_value")
    } | {("group", "penalty_capacity_margin")}
