"""The schema's value types match the shapes the engine reads.

For every parameter in ``PARAM_ALLOWED_SHAPES`` the master template's type
list must be exactly the types those shapes need: a value type the engine
reads but the schema forbids cannot be entered in the database editor, and a
type the schema allows but the engine rejects fails at run time.  Stochastic
time values (rank 3) are read from the solve's emit fold, not by the shape
resolver, so ``3d_map`` belongs to a parameter iff it is in
``STOCHASTIC_TIME_PARAMS`` or has a 3-D facet shape.

Every definition must also carry a type list.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pytest

from flextool._resources import package_data_path
from flextool.engine.input._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    STOCHASTIC_TIME_PARAMS,
    Shape,
    shape_rank,
)

SCHEMA = package_data_path("schemas/spinedb_schema.json")


def _type_name(shape: Shape) -> str:
    """Schema type name of a resolver shape: ``float`` / ``str`` at rank 0,
    else ``<rank>d_map`` with the shape's rank."""
    if shape == Shape.SCALAR_STR:
        return "str"
    rank = shape_rank(shape)
    return "float" if rank == 0 else f"{rank}d_map"


def _schema_types() -> tuple[dict[tuple[str, str], set[str]], set[tuple[str, str]]]:
    with open(SCHEMA, encoding="utf-8") as f:
        data = json.load(f)
    types: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cls, name, type_name, rank in data["parameter_types"]:
        types[(cls, name)].add(f"{rank}d_map" if type_name == "map" else type_name)
    definitions = {(row[0], row[1]) for row in data["parameter_definitions"]}
    return types, definitions


SCHEMA_TYPES, DEFINITIONS = _schema_types()


@pytest.mark.parametrize("key", sorted(PARAM_ALLOWED_SHAPES), ids=lambda k: f"{k[0]}.{k[1]}")
def test_schema_types_match_engine_shapes(key: tuple[str, str]) -> None:
    assert key in DEFINITIONS, f"{key[0]}.{key[1]} is read by the engine but has no schema definition"
    expected = {_type_name(s) for s in PARAM_ALLOWED_SHAPES[key]}
    if key in STOCHASTIC_TIME_PARAMS:
        expected.add("3d_map")
    assert SCHEMA_TYPES.get(key, set()) == expected


def test_every_definition_has_a_type_list() -> None:
    missing = sorted(k for k in DEFINITIONS if k not in SCHEMA_TYPES)
    assert not missing, f"definitions without a type list: {missing}"
