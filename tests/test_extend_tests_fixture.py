"""End-to-end test for the YAML-delta add-script.

Exercises :mod:`flextool.update_flextool.extend_tests_fixture` against
a temporary copy of ``tests/fixtures/tests.json`` — no real fixture is
ever mutated.  Keeps runtime short by skipping any LP solve; the only
DB operations are the json-to-sqlite round-trip plus the delta apply.
"""
from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import pytest

from flextool.update_flextool import extend_tests_fixture as ext

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tests.json"


@pytest.fixture()
def staged_fixture(tmp_path: Path) -> Path:
    """Copy tests.json into tmp_path; return the staged path.

    The module's ``_REPO_ROOT`` resolution is overridden per-call via
    the explicit ``target_path`` argument, so the staged copy can sit
    anywhere on disk.
    """
    dest = tmp_path / "tests.json"
    shutil.copyfile(SOURCE_FIXTURE, dest)
    return dest


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip())


def test_apply_delta_adds_entries(staged_fixture: Path, tmp_path: Path) -> None:
    """Happy path: delta adds entity + parameter_value + scenario."""
    yaml_file = tmp_path / "delta.yaml"
    _write_yaml(
        yaml_file,
        """
        target: ignored-because-target_path-overrides
        new_entities:
          - {class: node, name: stage2c_demo_node, description: "demo node"}
        new_alternatives:
          - {name: stage2c_demo_alt, description: "demo alternative"}
        new_parameter_values:
          - {class: node, entity: [stage2c_demo_node],
             alternative: stage2c_demo_alt,
             parameter: inflow, value: 42.0}
        new_scenarios:
          - {name: stage2c_demo_scen,
             alternatives: [Base, stage2c_demo_alt]}
        """,
    )

    rc = ext.apply_delta(yaml_file, target_path=staged_fixture)
    assert rc == 0

    with open(staged_fixture) as f:
        data = json.load(f)

    # Entity present.
    assert ("node", "stage2c_demo_node", "demo node") in [tuple(r) for r in data["entities"]]
    # Alternative present.
    assert ["stage2c_demo_alt", "demo alternative"] in data["alternatives"]
    # Scenario present with the requested ranked alternatives.
    assert any(s[0] == "stage2c_demo_scen" for s in data["scenarios"])
    sa_for_scen = [s for s in data["scenario_alternatives"] if s[0] == "stage2c_demo_scen"]
    # rank 1 -> Base, rank 2 -> stage2c_demo_alt -> tail None.  The
    # exported form uses the "before" linked-list layout:
    # [scenario, alt, next_alt_or_None].
    assert {tuple(s[1:]) for s in sa_for_scen} == {
        ("Base", "stage2c_demo_alt"),
        ("stage2c_demo_alt", None),
    }
    # Parameter value present.
    pv = [
        r for r in data["parameter_values"]
        if r[0] == "node" and r[1] == "stage2c_demo_node"
        and r[2] == "inflow" and r[4] == "stage2c_demo_alt"
    ]
    assert len(pv) == 1, pv


def test_second_apply_raises_on_collision(staged_fixture: Path, tmp_path: Path) -> None:
    """Re-applying the same YAML must fail (append-only)."""
    yaml_file = tmp_path / "delta.yaml"
    _write_yaml(
        yaml_file,
        """
        target: ignored
        new_entities:
          - {class: node, name: stage2c_dup_node}
        new_alternatives:
          - {name: stage2c_dup_alt}
        new_parameter_values:
          - {class: node, entity: [stage2c_dup_node],
             alternative: stage2c_dup_alt,
             parameter: inflow, value: 3.14}
        new_scenarios:
          - {name: stage2c_dup_scen,
             alternatives: [stage2c_dup_alt]}
        """,
    )
    assert ext.apply_delta(yaml_file, target_path=staged_fixture) == 0
    # Second apply must surface the collisions.
    rc = ext.apply_delta(yaml_file, target_path=staged_fixture)
    assert rc == 1


def test_validate_only_does_not_mutate(staged_fixture: Path, tmp_path: Path) -> None:
    """--validate / validate-only path must leave the fixture untouched."""
    before = staged_fixture.read_text()
    yaml_file = tmp_path / "valid.yaml"
    _write_yaml(
        yaml_file,
        """
        target: ignored
        new_entities:
          - {class: node, name: stage2c_validate_only_node}
        """,
    )
    delta = json.loads(json.dumps(__import__("yaml").safe_load(yaml_file.read_text())))
    schema = ext._load_schema()
    src_idx = ext._index_source(staged_fixture)
    errs = ext.validate_delta(delta, schema, source_index=src_idx)
    assert errs == []
    # File is byte-identical — validation did no I/O on the source.
    assert staged_fixture.read_text() == before


def test_validate_reports_unknown_class(staged_fixture: Path, tmp_path: Path) -> None:
    """Unknown entity_class -> readable error with did-you-mean."""
    schema = ext._load_schema()
    src_idx = ext._index_source(staged_fixture)
    delta = {"new_entities": [{"class": "nodee", "name": "x"}]}
    errs = ext.validate_delta(delta, schema, source_index=src_idx)
    assert len(errs) == 1
    assert "nodee" in errs[0]
    assert "node" in errs[0]  # suggestion fired


def test_validate_reports_unknown_parameter(staged_fixture: Path, tmp_path: Path) -> None:
    """Unknown parameter -> error names the class and suggests a near match."""
    schema = ext._load_schema()
    src_idx = ext._index_source(staged_fixture)
    delta = {
        "new_parameter_values": [
            {
                "class": "node",
                "entity": ["coal_market"],  # exists in fixture
                "alternative": "Base",
                "parameter": "capactiy",  # typo: should be 'capacity'
                "value": 1.0,
            }
        ]
    }
    errs = ext.validate_delta(delta, schema, source_index=src_idx)
    assert len(errs) == 1
    assert "capactiy" in errs[0]
    # The actual schema doesn't have 'capacity' on 'node' specifically;
    # the diagnostic should at least name the class and the unknown
    # parameter, suggestion is best-effort.
    assert "node" in errs[0]


def test_validate_rejects_structured_value(staged_fixture: Path, tmp_path: Path) -> None:
    """Time-series / map values must surface a 'use SpineDB editor' hint."""
    schema = ext._load_schema()
    src_idx = ext._index_source(staged_fixture)
    delta = {
        "new_parameter_values": [
            {
                "class": "node",
                "entity": ["coal_market"],
                "alternative": "Base",
                "parameter": "inflow",
                "value": {"2025-01-01": 1.0, "2025-01-02": 2.0},
            }
        ]
    }
    errs = ext.validate_delta(delta, schema, source_index=src_idx)
    assert len(errs) == 1
    assert "SpineDB editor" in errs[0]


def test_validate_rejects_multidim_dim_mismatch(staged_fixture: Path, tmp_path: Path) -> None:
    """Wrong number of elements for a multi-dim class is reported clearly."""
    schema = ext._load_schema()
    src_idx = ext._index_source(staged_fixture)
    delta = {
        "new_entities": [
            {"class": "connection__node__node", "entities": ["only_two", "elements"]}
        ]
    }
    errs = ext.validate_delta(delta, schema, source_index=src_idx)
    assert len(errs) == 1
    assert "3 dimensions" in errs[0]
    assert "2 element" in errs[0]


def test_apply_fails_when_referenced_entity_missing(
    staged_fixture: Path, tmp_path: Path
) -> None:
    """parameter_value referencing a non-existent entity must fail validation."""
    yaml_file = tmp_path / "bad.yaml"
    _write_yaml(
        yaml_file,
        """
        target: ignored
        new_parameter_values:
          - {class: node, entity: [no_such_node],
             alternative: Base,
             parameter: inflow, value: 1.0}
        """,
    )
    rc = ext.apply_delta(yaml_file, target_path=staged_fixture)
    assert rc == 1
