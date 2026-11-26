"""v82 adds ``life_extension_cost`` and ``life_extension_years`` on unit, connection and node.

The definitions are appended to the unreleased v82 (definitions only, no value move): a database migrated from 81
gets them from the step; a database already at 82 without them (made by an earlier build of v82) is flagged by
``_head_schema_missing_ro`` and completed by the head refresh, and nothing else of it changes.  Both end in the
same state, each definition a constant or a period map, group ``investment``, no default, with the shipped
description; a rerun commits nothing.

Databases: v75b (``_rerun.build_v75b``: the v2 template migrated to 75 plus the v75 seed data), migrated
(never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import v82
from tests.update_flextool._rerun import patch_migration
from flextool.update_flextool.db_migration import migrate_database

from tests.update_flextool._rerun import v75b_url

NAMES = ("life_extension_cost", "life_extension_years")
CLASSES = ("unit", "connection", "node")
#: The six definitions and their capacity unit (the description's first bracket).
SIX = {(cls, name): unit for cls, per in (("unit", "kW"), ("connection", "kW"), ("node", "kWh"))
       for name, unit in ((NAMES[0], f"[CUR/{per}]"), (NAMES[1], "[years]"))}


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (d["description"] or "", tuple(d["parameter_type_list"] or ()),
                                                  d["parameter_group_name"], d["parameter_value_list_name"],
                                                  None if d["default_value"] is None else bytes(d["default_value"]),
                                                  d["default_type"])
            for d in db.get_items("parameter_definition")}


def _state(url: str) -> dict:
    with DatabaseMapping(url) as db:
        return {"defs": _defs(db),
                "values": {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
                            v["alternative_name"]): (bytes(v["value"]), v["type"])
                           for v in db.get_items("parameter_value")},
                "groups": {g["name"]: (g["color"], g["priority"]) for g in db.get_items("parameter_group")}}


def _shipped() -> dict[tuple[str, str], tuple]:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    types: dict[tuple[str, str], list] = {}
    for cls, name, typ, rank in schema["parameter_types"]:
        types.setdefault((cls, name), []).append((typ, rank))
    return {(r[0], r[1]): (r[4], tuple(sorted(types.get((r[0], r[1]), ()))), r[5], r[3], r[2])
            for r in schema["parameter_definitions"]}


def _seed(tmp_path_factory, tag: str, up_to: int) -> str:
    url = v75b_url(tmp_path_factory, tmp_path_factory.mktemp(tag) / "seed.sqlite")
    migrate_database(url, up_to=up_to)
    return url


@pytest.fixture(scope="module")
def migrated(tmp_path_factory) -> dict:
    """A database migrated from 81 to 82, and one at 82 from which the six definitions were removed (the state an
    earlier build of v82 left), then migrated again."""
    url = _seed(tmp_path_factory, "le_from81", 81)
    before = _state(url)
    migrate_database(url)
    head = _seed(tmp_path_factory, "le_at82", FLEXTOOL_DB_VERSION)
    with DatabaseMapping(head) as db:
        for (cls, name) in SIX:
            db.remove_item("parameter_definition",
                           db.get_item("parameter_definition", entity_class_name=cls, name=name)["id"])
        db.commit_session("an earlier build of v82: no life extension")
    without = _state(head)
    path = Path(head.removeprefix("sqlite:///"))
    flagged = m._head_schema_missing_ro(path)
    migrate_database(head)
    return {"url": url, "before": before, "after": _state(url), "head": head, "without": without,
            "flagged": flagged, "refreshed": _state(head), "path": path}


def test_the_step_adds_the_six_definitions(migrated) -> None:
    before, after, shipped = migrated["before"]["defs"], migrated["after"]["defs"], _shipped()
    assert not [k for k in before if k in SIX]
    for key, unit in SIX.items():
        text, types, group, value_list, default, default_type = after[key]
        assert (types, group, value_list, default, default_type) == (("float", "1d_map"), "investment", None, None,
                                                                     None), key
        assert text.startswith(unit) and text == v82._v82_texts()[key] == shipped[key][0], key
        assert shipped[key][1:] == ((("float", 0), ("map", 1)), "investment", None, None), key
    # Both ends of the migration agree: from 81, and at 82 completed by the refresh.
    assert after == migrated["refreshed"]["defs"]


def test_descriptions_say_what_an_extension_does() -> None:
    """The final descriptions: both or neither, a one-off cost at the start, at most what is in service, the
    salvage value given up and received at the extended end of life, financed at discount_rate, retirable, never
    extended again, capacity and not investment in the limits, the period the extension starts in."""
    for cls in CLASSES:
        cost = v82._v82_texts()[(cls, "life_extension_cost")]
        years = v82._v82_texts()[(cls, "life_extension_years")]
        for phrase in ("paid when the extension starts", "at most all of it", "no invest_method is needed",
                       "net_salvage_value the capacity would have received at its end of life (given up",
                       "net_salvage_value received at the extended end of life", "financed at discount_rate",
                       "Not with reinvest_automatic", "never extended again",
                       "can be retired when invest_method allows retirement", "not as investment",
                       "Give both life_extension_cost and life_extension_years, or neither",
                       "the value of the period the extension starts in"):
            assert phrase in cost, (cls, phrase)
        for phrase in ("rounded by end_of_life_method", "Must be above 0", "or neither",
                       "the value of the period the extension starts in"):
            assert phrase in years, (cls, phrase)


def test_a_database_at_82_without_them_is_flagged_and_completed(migrated) -> None:
    """The refresh adds exactly the six, with their shipped fields; every other definition, value and group of the
    database stays as it was."""
    assert migrated["flagged"]
    without, refreshed = migrated["without"], migrated["refreshed"]
    assert set(refreshed["defs"]) - set(without["defs"]) == set(SIX)
    assert {k: v for k, v in refreshed["defs"].items() if k not in SIX} == without["defs"]
    assert refreshed["values"] == without["values"] and refreshed["groups"] == without["groups"]
    assert {k: refreshed["defs"][k] for k in SIX} == {k: migrated["after"]["defs"][k] for k in SIX}
    assert not m._head_schema_missing_ro(migrated["path"])


def _no_commit(db, step, monkeypatch) -> None:
    commits: list[str] = []

    def commit_must_be_empty(db_, message) -> None:
        with pytest.raises(NothingToCommit):
            db_.commit_session(message)
        commits.append(message)

    patch_migration(monkeypatch, "_commit_step", commit_must_be_empty)
    step(db)
    if step is v82._migrate_v82_schema_batch:
        assert len(commits) == 1, commits
    else:
        assert commits == []
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


@pytest.mark.parametrize("part", ["_v82_definitions_part", "_v82_descriptions", "_migrate_v82_schema_batch",
                                  "_refresh_head_schema"])
@pytest.mark.parametrize("which", ["url", "head"])
def test_rerun_commits_nothing(migrated, part, which, monkeypatch) -> None:
    before = _state(migrated[which])
    with DatabaseMapping(migrated[which]) as db:
        _no_commit(db, getattr(m if part == "_refresh_head_schema" else v82, part), monkeypatch)
    assert _state(migrated[which]) == before


def test_a_class_the_database_lacks_gets_no_definition() -> None:
    """The part adds a definition only on a class the database has (a database without ``node``)."""
    with DatabaseMapping("sqlite://", create=True) as db:
        for cls in ("unit", "connection", "model"):
            db.add_item("entity_class", name=cls)
        v82._v82_definitions_part(db)
        names = {(d["entity_class_name"], d["name"]) for d in db.get_items("parameter_definition")}
    assert {k for k in SIX if k[0] != "node"} <= names
    assert not [k for k in names if k[0] == "node"]


@pytest.mark.parametrize("edited", [False, True])
def test_head_refresh_replaces_the_superseded_extension_text(tmp_path_factory, edited) -> None:
    """A database at 82 made by a development build that offered a life extension also under reinvest_automatic has
    that build's life_extension_cost description: it gets the current one (no extension of reinvest_automatic
    capacity); a description the user edited is kept."""
    url = _seed(tmp_path_factory, f"le_dev_text_{edited}", FLEXTOOL_DB_VERSION)
    old = {cls: v82._v82_superseded()[(cls, "life_extension_cost")][1] + (" My note." if edited else "")
           for cls in CLASSES}
    with DatabaseMapping(url) as db:
        for cls, text in old.items():
            pdef = db.get_item("parameter_definition", entity_class_name=cls, name="life_extension_cost")
            db.update_parameter_definition(id=pdef["id"], description=text)
        db.commit_session("a development build's life extension text")
    path = Path(url.removeprefix("sqlite:///"))
    assert m._head_defaults_stale_ro(path) is not edited
    migrate_database(url)
    with DatabaseMapping(url) as db:
        for cls in CLASSES:
            got = db.get_item("parameter_definition", entity_class_name=cls, name="life_extension_cost")["description"]
            assert got == (old[cls] if edited else _shipped()[(cls, "life_extension_cost")][0]), cls
            assert edited or "Not with reinvest_automatic" in got
    assert not m._head_defaults_stale_ro(path)
