"""The default input source and scenario of a connection, in a project with two sources (C3-m6).

Both sources are ``tests/fixtures/tests.json`` (``a.sqlite`` numbered 1, ``b.sqlite`` 2); it has 69 scenarios,
among them ``coal`` (alternatives init, west, coal: of the units only coal_plant is active).  The default scenario
belongs to the default source (the FlexTool window's, or ``--input-source``); with no default source (``--scenario``
alone) it applies to any source that has it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool


@pytest.fixture
def two_sources(tmp_path: Path, tests_db: Path) -> Path:
    project = tmp_path / "two sources"
    (project / "input_sources").mkdir(parents=True)
    shutil.copy(tests_db, project / "input_sources" / "a.sqlite")
    shutil.copy(tests_db, project / "input_sources" / "b.sqlite")
    (project / "settings.yaml").write_text("input_source_numbers:\n  a.sqlite: 1\n  b.sqlite: 2\n", encoding="utf-8")
    return project


def _ok(ctx, tool: str, **args) -> dict:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


EXISTING = {"entity_class": "unit", "parameter": "existing", "entity": ["coal_plant"]}


def test_a_default_scenario_applies_to_its_own_source_only(two_sources: Path) -> None:
    ctx = make_context(two_sources, input_source="a.sqlite", scenario="coal")
    listing = _ok(ctx, "list_entities", entity_class="unit", source="b.sqlite")
    assert listing["source"] == "b.sqlite" and listing["scenario"] is None
    assert listing["inactive_in_scenario"] is None and listing["total"] == 10       # every unit, unfiltered
    out = call_tool(ctx, "get_parameter", {**EXISTING, "source": "b.sqlite"})
    assert not out.ok and out.message.startswith("The input source 'b.sqlite' has 69 scenarios")
    on_a = _ok(ctx, "get_parameter", **EXISTING)
    # tests.json: coal_plant existing in alternative coal is 500.
    assert (on_a["source"], on_a["scenario"], on_a["value"]) == ("a.sqlite", "coal", 500.0)
    on_a_by_number = _ok(ctx, "list_entities", entity_class="unit", source="1")
    assert on_a_by_number["scenario"] == "coal"
    assert on_a_by_number["entities"] == [{"entity": ["coal_plant"], "description": None}]


def test_a_default_scenario_without_a_source_applies_where_it_exists(two_sources: Path) -> None:
    unknown = make_context(two_sources, scenario="cli_only")
    listing = _ok(unknown, "list_entities", entity_class="unit", source="b.sqlite")
    assert listing["scenario"] is None and listing["total"] == 10
    known = make_context(two_sources, scenario="coal")
    on_b = _ok(known, "get_parameter", **EXISTING, source="b.sqlite")
    assert (on_b["source"], on_b["scenario"], on_b["value"]) == ("b.sqlite", "coal", 500.0)
