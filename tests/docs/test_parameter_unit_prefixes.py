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
