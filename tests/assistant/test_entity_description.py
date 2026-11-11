"""The entity description in ``get_entity`` and ``list_entities`` (P3 §8 test 10, description part; §2.4).

The database is ``tests/fixtures/tests.json`` with three entity descriptions set before it is built: unit
``coal_plant`` names a file under the home folder (redacted to its last part), node ``west`` has 1 500 characters
(cut to 1 000: 999 characters and an ellipsis), node ``heat`` only spaces (no description); every other entity has
none.  Expected values follow from those strings and P0 §6's rules, not from a tool run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flextool.assistant.caps import DESCRIPTION_MAX
from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool
from tests.assistant.conftest import FIXTURES, make_project

HOME_FILE = Path.home() / "plans" / "coal notes.txt"
COAL_TEXT = f"Hard coal plant, see {HOME_FILE.as_posix()} for the retirement plan."
WEST_TEXT = "W" * 1_500


@pytest.fixture(scope="module")
def described_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp("described")
    data = json.loads((FIXTURES / "tests.json").read_text(encoding="utf-8"))
    texts = {("unit", "coal_plant"): COAL_TEXT, ("node", "west"): WEST_TEXT, ("node", "heat"): "   "}
    entities = []
    for entity in data["entities"]:
        key = (entity[0], entity[1]) if isinstance(entity[1], str) else None
        if key in texts:
            entity = [entity[0], entity[1], texts[key], *entity[3:]]       # [class, name, description, …]
        entities.append(entity)
    data["entities"] = entities
    source = work / "described.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    db = work / "tests.sqlite"
    migrate_database(json_to_db(source, db))
    return make_project(work, db)


def _call(project: Path, tool: str, **args) -> dict:
    out = call_tool(make_context(project), tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


def test_get_entity_returns_the_description_redacted(described_project):
    match = _call(described_project, "get_entity", name=["coal_plant"], scenario="coal")["matches"][0]
    # A path under the home folder shrinks to its last part (P0 §6, D3); unquoted, a part ends at a space.
    assert match["description"] == "Hard coal plant, see coal notes.txt for the retirement plan."


def test_a_long_description_is_cut(described_project):
    match = _call(described_project, "get_entity", name=["west"], scenario="coal")["matches"][0]
    assert match["description"] == "W" * (DESCRIPTION_MAX - 1) + "…"
    assert len(match["description"]) == 1_000


def test_no_description_is_null(described_project):
    assert _call(described_project, "get_entity", name=["heat"], scenario="coal_chp")["matches"][0][
        "description"] is None
    assert _call(described_project, "get_entity", name=["coal_chp"], scenario="coal")["matches"][0][
        "description"] is None


def test_list_entities_carries_each_description(described_project):
    listing = _call(described_project, "list_entities", entity_class="node")
    by_name = {tuple(e["entity"]): e["description"] for e in listing["entities"]}
    assert by_name[("west",)] == "W" * (DESCRIPTION_MAX - 1) + "…"
    assert by_name[("heat",)] is None
    assert {name for name, text in by_name.items() if text is not None} == {("west",)}
    units = _call(described_project, "list_entities", entity_class="unit", scenario="coal")
    assert units["entities"] == [{"entity": ["coal_plant"], "description": (
        "Hard coal plant, see coal notes.txt for the retirement plan.")}]
