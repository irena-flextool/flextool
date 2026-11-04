"""Schema sections generated from ``spinedb_schema.json`` and the sync rules (``fixture_schema``).

The synthetic fixtures here are the schema's own sections plus a few data
rows, so every rule is exercised without a fixture file.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.fixture_schema import (
    NO_DEFAULT,
    SCHEMA_SECTIONS,
    FixtureSchemaError,
    fixture_version,
    normalise_sections,
    schema_sections,
    section_diffs,
    sync_schema_sections,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pack(value: object, value_type: str | None) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode("ascii"), value_type]


def _set_version(data: dict, version: float) -> None:
    for row in data["parameter_definitions"]:
        if row[:2] == ["model", "version"]:
            row[2] = _pack(version, "float")


def _fixture(**data_sections) -> dict:
    """A packed fixture: data sections first (as exports order them), then the schema sections."""
    out = {"entities": [], "alternatives": [["base", "Base alternative"]], "parameter_values": []}
    out.update(data_sections)
    out.update(schema_sections("packed"))
    return out


@pytest.mark.parametrize("fmt", ["packed", "decoded"])
def test_schema_sections_decode_back_to_the_schema_json(fmt: str) -> None:
    """The DB route keeps every class, list, group, definition and type of the schema JSON."""
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        raw = json.load(f)
    assert normalise_sections(schema_sections(fmt), fmt) == normalise_sections(raw, "raw")


def test_packed_no_default_is_sql_null() -> None:
    """A definition without a default is written as SQL NULL, never as the JSON ``null`` blob."""
    rows = [row for row in schema_sections("packed")["parameter_definitions"]
            if normalise_sections({"parameter_definitions": [row]}, "packed")["parameter_definitions"]
            [(row[0], row[1])]["default"] in (None, NO_DEFAULT)]
    assert rows, "the schema has definitions without a default"
    assert all(row[2] == [None, None] for row in rows), [row[:3] for row in rows if row[2] != [None, None]][:5]


def test_schema_sections_are_fresh_copies() -> None:
    first = schema_sections("packed")
    first["parameter_definitions"].clear()
    assert schema_sections("packed")["parameter_definitions"]


def test_schema_sections_do_not_depend_on_the_hash_seed() -> None:
    code = ("import hashlib, json; from flextool.update_flextool.fixture_schema import schema_sections; "
            "print(hashlib.md5(json.dumps([schema_sections('packed'), schema_sections('decoded')])"
            ".encode()).hexdigest())")
    digests = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True, timeout=120)
        digests.add(out.stdout.strip())
    local = hashlib.md5(json.dumps([schema_sections("packed"), schema_sections("decoded")]).encode()).hexdigest()
    assert digests == {local}


def test_sync_of_a_current_fixture_changes_nothing() -> None:
    data = _fixture()
    assert section_diffs(data, "packed") == []
    assert sync_schema_sections(data, "packed") == data


def test_version_guard_rejects_an_unmigrated_fixture() -> None:
    data = _fixture()
    _set_version(data, FLEXTOOL_DB_VERSION - 1)
    with pytest.raises(FixtureSchemaError, match=f"version {FLEXTOOL_DB_VERSION - 1}, not {FLEXTOOL_DB_VERSION}"):
        sync_schema_sections(data, "packed")


def test_fixture_without_version_is_synced_and_gains_the_version() -> None:
    """The sync-only path: no ``model.version`` definition means written against the current schema."""
    node = next(row for row in schema_sections("packed")["entity_classes"] if row[0] == "node")
    data = {"entity_classes": [node], "entities": [["node", "n1", None]],
            "parameter_definitions": [["node", "inflow", None, None, "old text", None]],
            "parameter_values": [["node", "n1", "inflow", _pack(-5.0, "float"), "base"]]}
    assert fixture_version(data, "packed") is None
    out = sync_schema_sections(data, "packed")
    assert fixture_version(out, "packed") == FLEXTOOL_DB_VERSION
    assert section_diffs(out, "packed") == []
    assert out["entities"] is data["entities"] and out["parameter_values"] is data["parameter_values"]


def test_fixture_only_definition_without_values_is_dropped() -> None:
    data = _fixture()
    data["parameter_definitions"].append(["node", "retired_knob", None, None, "gone", None])
    data["parameter_types"].append(["node", "retired_knob", "float", 0])
    assert "parameter_definitions: node/retired_knob: not in the schema" in section_diffs(data, "packed")
    out = sync_schema_sections(data, "packed")
    assert ("node", "retired_knob") not in {(r[0], r[1]) for r in out["parameter_definitions"]}
    assert section_diffs(out, "packed") == []


def test_value_of_a_fixture_only_definition_raises() -> None:
    data = _fixture(entities=[["node", "n1", None]],
                    parameter_values=[["node", "n1", "retired_knob", _pack(1.0, "float"), "base"]])
    data["parameter_definitions"].append(["node", "retired_knob", None, None, "gone", None])
    with pytest.raises(FixtureSchemaError, match=r"node 'n1' retired_knob \[base\]: parameter not in the schema"):
        sync_schema_sections(data, "packed")


def test_entity_of_a_fixture_only_class_raises() -> None:
    data = _fixture(entities=[["retired_class", "e1", None]])
    data["entity_classes"].append(["retired_class", [], None, None, True])
    with pytest.raises(FixtureSchemaError, match="entity retired_class 'e1': class not in the schema"):
        sync_schema_sections(data, "packed")


def test_value_outside_the_schema_value_list_raises() -> None:
    data = _fixture(entities=[["node", "n1", None]],
                    parameter_values=[["node", "n1", "storage_binding_method", _pack("bind_sideways", "str"),
                                       "base"]])
    with pytest.raises(FixtureSchemaError, match="'bind_sideways': not in the value list"):
        sync_schema_sections(data, "packed")


def test_value_in_the_schema_value_list_passes() -> None:
    data = _fixture(entities=[["node", "n1", None]],
                    parameter_values=[["node", "n1", "storage_binding_method", _pack("bind_forward_only", "str"),
                                       "base"]])
    sync_schema_sections(data, "packed")


def test_data_sections_are_kept_byte_for_byte_and_in_order() -> None:
    """Every non-schema section is the same object, in the same key position; stale sections are replaced."""
    stale = {key: [] for key in SCHEMA_SECTIONS}
    data = {"entity_classes": stale["entity_classes"], "entities": [["node", "n1", "a node"]],
            "parameter_value_lists": stale["parameter_value_lists"],
            "parameter_definitions": [["model", "version", _pack(FLEXTOOL_DB_VERSION, "float"), None, "", None]],
            "parameter_values": [["node", "n1", "inflow", _pack({"index_type": "str", "data": [["t1", 1.0]]}, "map"),
                                  "base"]],
            "alternatives": [["base", ""]], "scenarios": [["s", False, ""]],
            "scenario_alternatives": [["s", "base", None]]}
    before = json.dumps(data)
    out = sync_schema_sections(data, "packed")
    assert json.dumps(data) == before, "the input is not modified"
    assert list(out)[:len(data)] == list(data)
    for key in data:
        if key not in SCHEMA_SECTIONS:
            assert out[key] is data[key]
    assert section_diffs(out, "packed") == []


def test_decoded_format_sync() -> None:
    data = {"entities": [["node", "n1", None]], "alternatives": [["Base", ""]],
            "parameter_values": [["node", "n1", "storage_binding_method", ['"bind_forward_only"', "str"], "Base"]]}
    data.update(schema_sections("decoded"))
    data["parameter_definitions"] = [r for r in data["parameter_definitions"] if r[:2] != ["node", "inflow"]]
    assert section_diffs(data, "decoded") == ["parameter_definitions: node/inflow: missing"]
    assert section_diffs(sync_schema_sections(data, "decoded"), "decoded") == []
