"""Database migration to version 80 (all schema changes of one release in one step).

A v79 database (built from a v75 seed JSON and migrated up to 79, never a checked-in ``.sqlite``) is migrated to 80.
The step

* appends to the description of each period-map limit that ``inf`` (``-inf`` for a lower limit) is no limit in that
  period, and to the four time-series limits that a ``new_stepduration`` solve step combining a no-limit time step
  with finite ones has no limit for the whole step (once: a rerun adds nothing);
* renames ``group.capacity_margin`` -> ``capacity_margin_absolute`` and ``capacity_margin_share`` ->
  ``capacity_margin_relative`` (the values go with the definition, in every alternative, unchanged);
* renames the ``capacity_margin_methods`` members ``manual`` -> ``absolute`` and ``share_of_peak`` ->
  ``relative_of_peak_demand`` (list order kept) and rewrites every stored ``capacity_margin_method`` value;
* rewrites the capacity-margin, ``solve.solver_precommand`` and ``solve.solver_arguments`` descriptions.

The new texts and the value list are the shipped schema's; nothing else changes, and a rerun commits nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
RENAMED = {("group", old): ("group", new) for old, new in m.V80_CAPACITY_MARGIN_RENAMES.items()}
#: Stored before the step: (alternative index, parameter, value) on group stoch_g.
STORED = [
    (0, "capacity_margin_method", "manual"),
    (0, "capacity_margin", 120.0),
    (1, "capacity_margin_method", "share_of_peak"),
    (1, "capacity_margin_share", Map(["p2020", "p2025"], [0.15, 0.2], index_name="period")),
    (1, "capacity_margin", Map(["p2020", "p2025"], [50.0, -float("inf")], index_name="period")),
    (2, "capacity_margin_method", "none"),
]


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (d["description"] or "", tuple(d["parameter_type_list"] or ()),
                                                  d["parameter_group_name"], d["parameter_value_list_name"],
                                                  d["default_value"], d["default_type"])
            for d in db.get_items("parameter_definition")}


def _lists(db) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for x in sorted(db.get_items("list_value"), key=lambda x: (x["parameter_value_list_name"], x["index"])):
        out.setdefault(x["parameter_value_list_name"], []).append(x["parsed_value"])
    return out


def _values(db) -> dict:
    return {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
             v["alternative_name"]): (bytes(v["value"]), v["type"]) for v in db.get_items("parameter_value")}


def _shipped() -> tuple[dict, dict[str, list[str]]]:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    lists: dict[str, list[str]] = {}
    for name, value in schema["parameter_value_lists"]:
        lists.setdefault(name, []).append(value)
    return {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}, lists


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp("v80") / "seed.sqlite")
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        alts = sorted(a["name"] for a in db.get_items("alternative"))
        for i, param, value in STORED:
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name="group", entity_byname=("stoch_g",),
                                   parameter_definition_name=param, alternative_name=alts[i], value=val, type=typ)
        db.commit_session("v79 capacity-margin values")
        before = {"defs": _defs(db), "lists": _lists(db), "values": _values(db)}
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        after = {"defs": _defs(db), "lists": _lists(db), "values": _values(db)}
    return {"url": url, "alts": alts, "before": before, "after": after}


def test_appends_the_limit_sentences_once(migrated) -> None:
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    suffixes = m._v80_suffixes()
    assert ("group", "co2_max_period") in suffixes and ("constraint", "constant") in suffixes
    assert len(suffixes[("flowGroup", "max_instant_flow")]) == 2
    for key, sentences in suffixes.items():
        if key not in before:
            continue
        assert after[key][0] == before[key][0].rstrip() + "".join(sentences) == shipped[key]
    for key in (("flowGroup", "max_instant_flow"), ("flowGroup", "min_instant_flow"), ("group", "inertia_limit"),
                ("group", "non_synchronous_limit")):
        assert "new_stepduration, a solve step that combines" in after[key][0]
        assert "for the whole solve step" in after[key][0]


def test_append_completes_a_partly_written_suffix() -> None:
    sentences = (" first.", " second.")
    assert m._v80_append_sentences("Text.", sentences) == "Text. first. second."
    assert m._v80_append_sentences("Text. first.", sentences) == "Text. first. second."
    assert m._v80_append_sentences("Text. first. second.  ", sentences) == "Text. first. second."


def test_renames_the_capacity_margin_definitions(migrated) -> None:
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    for old, new in RENAMED.items():
        assert old in before and new not in before
        assert old not in after and old not in shipped
        assert after[new][0] == shipped[new] == m._v80_texts()[new]
        assert after[new][1:] == before[old][1:]
    assert after[("group", "capacity_margin_absolute")][0].startswith("[MW] ")
    assert "capacity_margin_method relative_of_peak_demand" in after[("group", "capacity_margin_relative")][0]


def test_renames_the_method_values(migrated) -> None:
    before, after = migrated["before"], migrated["after"]
    _, shipped_lists = _shipped()
    assert before["lists"]["capacity_margin_methods"] == ["none", "manual", "share_of_peak"]
    assert after["lists"]["capacity_margin_methods"] == ["none", "absolute", "relative_of_peak_demand"]
    assert after["lists"]["capacity_margin_methods"] == shipped_lists["capacity_margin_methods"]
    key = ("group", "capacity_margin_method")
    assert after["defs"][key][0] == _shipped()[0][key] == m._v80_texts()[key]
    assert after["defs"][key][1:] == before["defs"][key][1:]
    for word in ("absolute: capacity_margin_absolute MW", "relative_of_peak_demand: capacity_margin_relative"):
        assert word in after["defs"][key][0]


def test_rewrites_every_stored_value_and_keeps_the_data(migrated) -> None:
    """Each stored value is under its new name in its alternative: a method value is renamed, a margin value keeps
    its data byte for byte, and no other value changes."""
    old, new = migrated["before"]["values"], migrated["after"]["values"]
    expected = {}
    for (cls, byname, param, alt), (value, typ) in old.items():
        if cls == "group" and param == "capacity_margin_method":
            value, typ = to_database(m.V80_CAPACITY_MARGIN_METHODS[json.loads(value)])
        expected[(cls, byname, m.V80_CAPACITY_MARGIN_RENAMES.get(param, param) if cls == "group" else param,
                  alt)] = (bytes(value), typ)
    assert new == expected
    stored = {(i, param) for i, param, _ in STORED}
    assert len(stored) == len(STORED) and {i for i, _ in stored} == {0, 1, 2}
    alts = migrated["alts"]
    assert json.loads(new[("group", ("stoch_g",), "capacity_margin_method", alts[1])][0]) == "relative_of_peak_demand"
    assert ("group", ("stoch_g",), "capacity_margin_relative", alts[1]) in new


def test_replaces_the_solver_descriptions(migrated) -> None:
    after = migrated["after"]["defs"]
    shipped, _ = _shipped()
    for key in (("solve", "solver_precommand"), ("solve", "solver_arguments")):
        assert after[key][0] == shipped[key] == m._v80_texts()[key] != migrated["before"]["defs"][key][0]
    assert "Set a solver wrapper on your computer instead" in after[("solve", "solver_precommand")][0]
    arguments = after[("solve", "solver_arguments")][0]
    assert "solver's own option names" in arguments and "whichever solver the solve uses" in arguments
    assert "checked against FlexTool's list" in arguments and "solver_config/<solver>.opt" in arguments
    assert "HiGHS knobs" not in arguments


def test_changes_nothing_else(migrated) -> None:
    before, after = migrated["before"], migrated["after"]
    touched = set(m._v80_suffixes()) | set(m._v80_texts()) | set(RENAMED) | {("model", "version")}
    assert ({k: v for k, v in after["defs"].items() if k not in touched}
            == {k: v for k, v in before["defs"].items() if k not in touched})
    assert ({k: v for k, v in after["lists"].items() if k != "capacity_margin_methods"}
            == {k: v for k, v in before["lists"].items() if k != "capacity_margin_methods"})


def test_rerun_commits_nothing(migrated, monkeypatch) -> None:
    commits: list[str] = []

    def commit_must_be_empty(db_, message) -> None:
        with pytest.raises(NothingToCommit):
            db_.commit_session(message)
        commits.append(message)

    monkeypatch.setattr(m, "_commit_step", commit_must_be_empty)
    with DatabaseMapping(migrated["url"]) as db:
        m._migrate_v80_schema_batch(db)
        assert len(commits) == 1
        assert _defs(db) == migrated["after"]["defs"] and _values(db) == migrated["after"]["values"]
