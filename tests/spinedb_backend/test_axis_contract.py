"""Phase 0 verification tests for the FlexTool axis contract.

The contract (`schemas/flextool_axis_contract.json`) declares which
entity classes / parameter maps each cascade axis sources its vocabulary
from.  These tests gate every contract edit:

1. The contract validates against its own JSON Schema
   (`schemas/flextool_axis_contract.schema.json`).
2. Every `entity_class` / `entity_class_union` axis references a class
   that actually exists in `schemas/spinedb_schema.json`.
3. Every `parameter_keys` / `parameter_value_list` axis references a
   parameter definition (and its parent entity class) that exists in
   the schema.
4. Every synthetic-token-allowlist entry references an axis that exists
   in `axes`.
5. When `mixed_vocab_columns.confirmed` is non-empty, the contract
   includes the entity-union axis ('e') against which those columns are
   cast.

See `schemas/AXIS_CONTRACT.md` for the editor-facing documentation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


from flextool._resources import package_data_path

CONTRACT_PATH = package_data_path("schemas/flextool_axis_contract.json")
CONTRACT_SCHEMA_PATH = package_data_path("schemas/flextool_axis_contract.schema.json")
TEMPLATE_MASTER_PATH = package_data_path("schemas/spinedb_schema.json")


@pytest.fixture(scope="module")
def contract() -> dict:
    """Parsed `flextool_axis_contract.json`."""
    with CONTRACT_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def contract_schema() -> dict:
    """Parsed `flextool_axis_contract.schema.json`."""
    with CONTRACT_SCHEMA_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def template_master() -> dict:
    """Parsed `schemas/spinedb_schema.json`."""
    with TEMPLATE_MASTER_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def entity_class_names(template_master: dict) -> set[str]:
    """Set of entity-class names declared by the schema.

    The schema stores entity_classes as a list of rows whose first
    column is the entity-class name.
    """
    rows = template_master["entity_classes"]
    names: set[str] = set()
    for row in rows:
        if isinstance(row, list) and row:
            names.add(row[0])
        elif isinstance(row, str):
            names.add(row)
    return names


@pytest.fixture(scope="module")
def parameter_definitions(template_master: dict) -> set[tuple[str, str]]:
    """Set of (entity_class, parameter_name) pairs declared by the schema.

    `parameter_definitions` is a list of rows whose first two columns
    are entity_class and parameter_name (per `inventory.csv` conventions
    and direct inspection of `schemas/spinedb_schema.json`).
    """
    rows = template_master.get("parameter_definitions", [])
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        if isinstance(row, list) and len(row) >= 2:
            pairs.add((row[0], row[1]))
    return pairs


def test_contract_loads_and_validates_against_schema(
    contract: dict, contract_schema: dict
) -> None:
    """Both files parse, and the contract validates against the schema."""
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed")
    # Use the draft 2020-12 validator explicitly (the schema declares
    # $schema as draft/2020-12).
    validator_cls = jsonschema.validators.validator_for(contract_schema)
    validator_cls.check_schema(contract_schema)
    validator = validator_cls(contract_schema)
    errors = sorted(validator.iter_errors(contract),
                    key=lambda e: list(e.absolute_path))
    assert errors == [], "Contract failed schema validation:\n" + "\n".join(
        f"  {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in errors
    )


def test_axes_referencing_entity_classes_exist(
    contract: dict, entity_class_names: set[str]
) -> None:
    """Every `entity_class` and `entity_class_union` axis references an
    entity class that exists in `schemas/spinedb_schema.json`."""
    missing: list[str] = []
    for axis in contract["axes"]:
        st = axis["source_type"]
        src = axis.get("source")
        if st == "entity_class":
            if src not in entity_class_names:
                missing.append(f"{axis['name']}: source={src!r}")
        elif st == "entity_class_union":
            for member in src or []:
                if member not in entity_class_names:
                    missing.append(f"{axis['name']}: union member {member!r}")
    assert missing == [], (
        "Axis rows reference entity classes that do not exist in the schema:\n"
        + "\n".join(f"  {m}" for m in missing)
    )


def test_axes_referencing_parameters_exist(
    contract: dict,
    entity_class_names: set[str],
    parameter_definitions: set[tuple[str, str]],
) -> None:
    """Every `parameter_keys` / `parameter_value_list` axis references a
    parameter definition (and its parent entity class) that exists in
    the schema.

    Three address shapes are supported under `source_type: parameter_keys`:

    * `{entity_class, parameter}` — single parameter.
    * `{entity_class, parameters: [...]}` — multiple parameters (union).
    * `{entity_class, parameter_prefix: "..."}` — every parameter whose
      name starts with the prefix; at least one must match.
    * `{carrier, column}` — points at a per-solve scratch frame, not a
      schema parameter — skipped here (carrier-derived; verified at
      Phase 2 against the writer).
    """
    missing: list[str] = []
    for axis in contract["axes"]:
        st = axis["source_type"]
        if st not in ("parameter_keys", "parameter_value_list"):
            continue
        src = axis.get("source") or {}

        # Carrier-derived axes (e.g. d_anchor → period__branch column)
        # are not schema parameters; skip them here.
        if "carrier" in src:
            continue

        ec = src.get("entity_class")
        if ec is None:
            missing.append(f"{axis['name']}: missing entity_class in source")
            continue
        if ec not in entity_class_names:
            missing.append(
                f"{axis['name']}: entity_class {ec!r} not in schema"
            )
            continue

        params: list[str] = []
        if "parameter" in src:
            params.append(src["parameter"])
        if "parameters" in src:
            params.extend(src["parameters"])

        if "parameter_prefix" in src:
            prefix = src["parameter_prefix"]
            matches = [
                pname for (cls, pname) in parameter_definitions
                if cls == ec and pname.startswith(prefix)
            ]
            if not matches:
                missing.append(
                    f"{axis['name']}: no {ec!r} parameter starts "
                    f"with {prefix!r}"
                )
        elif not params:
            missing.append(
                f"{axis['name']}: source has neither 'parameter', "
                f"'parameters', nor 'parameter_prefix'"
            )

        for pname in params:
            if (ec, pname) not in parameter_definitions:
                missing.append(
                    f"{axis['name']}: parameter {pname!r} not defined "
                    f"on {ec!r}"
                )

    assert missing == [], (
        "Axis rows reference parameters that do not exist in the schema:\n"
        + "\n".join(f"  {m}" for m in missing)
    )


def test_synthetic_allowlist_axes_exist(contract: dict) -> None:
    """Every entry in `synthetic_token_allowlist` references an axis
    that exists in `axes`."""
    axis_names = {a["name"] for a in contract["axes"]}
    missing: list[str] = []
    for entry in contract.get("synthetic_token_allowlist", []):
        if entry["axis"] not in axis_names:
            missing.append(
                f"allowlist entry for axis {entry['axis']!r} "
                f"has no matching axis row"
            )
    assert missing == [], "\n".join(missing)


def test_mixed_vocab_columns_have_entity_axis(contract: dict) -> None:
    """When `mixed_vocab_columns.confirmed` is non-empty, the contract
    must include the entity-union axis ('e') against which those
    columns are cast.

    This is a sanity check: the cascade casts source/sink/entity columns
    against `pl.Enum(node ∪ unit ∪ connection)`, so the contract must
    declare that union vocabulary somewhere.
    """
    mvc = contract.get("mixed_vocab_columns", {})
    confirmed = mvc.get("confirmed", [])
    if not confirmed:
        pytest.skip("no mixed-vocab columns declared yet")
    axes_by_name = {a["name"]: a for a in contract["axes"]}
    assert "e" in axes_by_name, (
        "mixed_vocab_columns lists "
        f"{confirmed!r} but no 'e' axis row declares the entity union"
    )
    e_axis = axes_by_name["e"]
    assert e_axis["source_type"] in ("entity_class_union", "union"), (
        f"'e' axis must be a union of entity classes; got "
        f"source_type={e_axis['source_type']!r}"
    )
    src = set(e_axis.get("source") or [])
    expected_minimum = {"node", "unit", "connection"}
    missing = expected_minimum - src
    assert not missing, (
        f"'e' axis union must cover {expected_minimum!r}; "
        f"missing {missing!r}"
    )
