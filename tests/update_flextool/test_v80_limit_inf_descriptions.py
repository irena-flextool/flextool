"""Database migration to version 80 (all schema changes of one release in one step).

A v79 database (v75b, ``_rerun.build_v75b``: the v2 template migrated to 75 plus the v75 seed data,
migrated up to 79; never a checked-in ``.sqlite``) is migrated to 80.
The step

* appends to the description of each period-map limit that ``inf`` (``-inf`` for a lower limit) is no limit in that
  period, and to the four time-series limits that a ``new_stepduration`` solve step combining a no-limit time step
  with finite ones has no limit for the whole step (once: a rerun adds nothing);
* renames ``group.capacity_margin`` -> ``capacity_margin_absolute`` and ``capacity_margin_share`` ->
  ``capacity_margin_relative`` (the values go with the definition, in every alternative, unchanged);
* renames the ``capacity_margin_methods`` members ``manual`` -> ``absolute`` and ``share_of_peak`` ->
  ``relative_of_peak_demand`` (list order kept) and rewrites every stored ``capacity_margin_method`` value (each one
  printed) and a ``manual`` / ``share_of_peak`` default;
* rewrites the capacity-margin, ``solve.solver_precommand``, ``solve.solver_arguments``,
  ``unit.conversion_method``, ``solve.reduce_unit_detail_method`` and ``group.print_dispatch`` descriptions;
* appends ``piecewise_linear`` to ``conversion_methods``, adds ``unit.piecewise_linear_efficiency`` (a curve, or a
  curve per period) and appends the piecewise_linear sentences to ``efficiency``, ``efficiency_at_min_load`` and
  ``min_load``.

The new texts and the value list are the shipped schema's; nothing else changes, and a rerun commits nothing.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, from_database, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from tests.update_flextool._rerun import patch_migration
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db
from tests.update_flextool._rerun import v75b_url

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
    """``({(class, name): description}, lists)`` of the shipped schema; a description v83 replaced is its old text."""
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    lists: dict[str, list[str]] = {}
    for name, value in schema["parameter_value_lists"]:
        lists.setdefault(name, []).append(value)
    v83_back = {key: old for key, (old, _new) in m._v83_superseded_texts().items()}
    return {(r[0], r[1]): v83_back.get((r[0], r[1]), r[4]) for r in schema["parameter_definitions"]}, lists


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    url = v75b_url(tmp_path_factory, tmp_path_factory.mktemp("v80") / "seed.sqlite")
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        alts = sorted(a["name"] for a in db.get_items("alternative"))
        for i, param, value in STORED:
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name="group", entity_byname=("stoch_g",),
                                   parameter_definition_name=param, alternative_name=alts[i], value=val, type=typ)
        db.commit_session("v79 capacity-margin values")
        before = {"defs": _defs(db), "lists": _lists(db), "values": _values(db)}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        after = {"defs": _defs(db), "lists": _lists(db), "values": _values(db)}
    return {"url": url, "alts": alts, "before": before, "after": after, "printed": printed.getvalue()}


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


def test_renamed_values_are_printed(migrated) -> None:
    """Each stored old method value is named in the message, once."""
    alts = migrated["alts"]
    lines = [x for x in migrated["printed"].splitlines() if x.startswith("v80: capacity_margin_method values renamed")]
    assert len(lines) == 1
    assert sorted(lines[0].split("): ", 1)[1].split(", ")) == sorted([
        f"group stoch_g [{alts[0]}]: manual -> absolute",
        f"group stoch_g [{alts[1]}]: share_of_peak -> relative_of_peak_demand"])


@pytest.mark.parametrize(("old", "new"), [("manual", "absolute"), ("share_of_peak", "relative_of_peak_demand")])
def test_an_old_method_default_is_renamed(tmp_path_factory, old, new) -> None:
    """A v79 ``capacity_margin_method`` default ``manual`` / ``share_of_peak`` reads the renamed member at 80 (it
    stopped the migration at 79: the later description update found the old name), and a rerun commits nothing."""
    url = v75b_url(tmp_path_factory, tmp_path_factory.mktemp(f"v80_default_{old}") / "seed.sqlite")
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="group", name="capacity_margin_method")
        val, typ = to_database(old)
        db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
        db.commit_session(f"a {old} default")
        before = _defs(db)
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        version = db.get_item("parameter_definition", entity_class_name="model", name="version")
        assert from_database(version["default_value"], version["default_type"]) == 80
        after = _defs(db)
        key = ("group", "capacity_margin_method")
        assert from_database(*after[key][4:]) == new
        assert after[key][1:4] == before[key][1:4]
        assert after[key][0] == m._v80_texts()[key]
    with DatabaseMapping(url) as db:
        with pytest.raises(NothingToCommit):
            m._v80_capacity_margin_rename(db)
            m._v80_descriptions(db)
            db.commit_session("rerun")


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


def test_replaces_the_reduce_unit_detail_description(migrated) -> None:
    """The full-load rule: min_load_efficiency and piecewise_linear units without online variables, and under the
    min_limits levels a min_load_efficiency unit more efficient at min load, run at their full-load efficiency."""
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    key = ("solve", "reduce_unit_detail_method")
    assert after[key][0] == shipped[key] == m._v80_texts()[key] != before[key][0]
    assert before[key][0] == m._V79_REDUCE_DESCRIPTIONS["reduce_unit_detail_method"]
    assert after[key][1:] == before[key][1:]
    text = after[key][0]
    assert "less fuel than the curve at part load" not in text
    for words in ("min_load_efficiency and piecewise_linear units run at their full-load efficiency",
                  "the same fuel at full output; at part load less or more fuel",
                  "(efficiency_at_min_load above efficiency) then runs at its full-load efficiency",
                  "piecewise_linear curve whose efficiency rises from its first point"):
        assert words in text


def test_replaces_the_print_dispatch_description(migrated) -> None:
    """print_dispatch also gives the group's ramp envelope and its Ramp envelope and Ramp frequency plots."""
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    key = ("group", "print_dispatch")
    assert after[key][0] == shipped[key] == m._v80_texts()[key] != before[key][0]
    assert before[key][0] == "[yes/no] " + m._V58_GROUP_OUTPUT_FLAG_DESCRIPTIONS["print_dispatch"]
    assert after[key][1:] == before[key][1:]
    assert after[key][0].startswith(before[key][0] + " ")
    for words in ("ramp envelope (ramp_envelope_t", "Ramp envelope and Ramp frequency plots"):
        assert words in after[key][0]


PIECEWISE = ("unit", "piecewise_linear_efficiency")


def test_adds_the_piecewise_linear_member(migrated) -> None:
    before, after = migrated["before"]["lists"], migrated["after"]["lists"]
    _, shipped_lists = _shipped()
    assert "piecewise_linear" not in before["conversion_methods"]
    assert after["conversion_methods"] == before["conversion_methods"] + ["piecewise_linear"]
    assert set(after["conversion_methods"]) == set(shipped_lists["conversion_methods"])
    assert shipped_lists["conversion_methods"][-1] == "piecewise_linear"


def test_adds_the_piecewise_linear_efficiency_definition(migrated) -> None:
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    assert PIECEWISE not in before
    description, types, group, value_list, default, default_type = after[PIECEWISE]
    assert types == ("1d_map", "2d_map") and group == "basics" and value_list is None
    assert default is None and default_type is None
    assert description == shipped[PIECEWISE] == m.V80_PIECEWISE_LINEAR_EFFICIENCY_DESCRIPTION
    for words in ("The last point must be at load 1", "the message gives the nearest curve that is accepted",
                  "gets linear start-ups (an info line says so)", "does not use efficiency or efficiency_at_min_load",
                  "a curve per period (map: period → load share → efficiency)"):
        assert words in description


def test_piecewise_linear_descriptions(migrated) -> None:
    before, after = migrated["before"]["defs"], migrated["after"]["defs"]
    shipped, _ = _shipped()
    key = ("unit", "conversion_method")
    assert after[key][0] == shipped[key] == m._v80_texts()[key] != before[key][0]
    assert "piecewise_linear: the fuel use follows the part-load curve piecewise_linear_efficiency" in after[key][0]
    assert after[key][1:] == before[key][1:]
    for key, sentences in m._v80_piecewise_sentences().items():
        assert after[key][0] == before[key][0].rstrip() + "".join(sentences) == shipped[key]
        assert after[key][1:] == before[key][1:]


def test_changes_nothing_else(migrated) -> None:
    before, after = migrated["before"], migrated["after"]
    touched = (set(m._v80_suffixes()) | set(m._v80_texts()) | set(RENAMED) | set(m._v80_piecewise_sentences())
               | {PIECEWISE, ("model", "version")})
    assert ({k: v for k, v in after["defs"].items() if k not in touched}
            == {k: v for k, v in before["defs"].items() if k not in touched})
    changed_lists = {"capacity_margin_methods", "conversion_methods"}
    assert ({k: v for k, v in after["lists"].items() if k not in changed_lists}
            == {k: v for k, v in before["lists"].items() if k not in changed_lists})


def test_rerun_commits_nothing(migrated, monkeypatch) -> None:
    commits: list[str] = []

    def commit_must_be_empty(db_, message) -> None:
        with pytest.raises(NothingToCommit):
            db_.commit_session(message)
        commits.append(message)

    patch_migration(monkeypatch, "_commit_step", commit_must_be_empty)
    with DatabaseMapping(migrated["url"]) as db:
        m._migrate_v80_schema_batch(db)
        assert len(commits) == 1
        assert _defs(db) == migrated["after"]["defs"] and _values(db) == migrated["after"]["values"]


@pytest.mark.parametrize("stamp", [76, 78])
def test_current_template_with_an_older_stamp_gets_no_duplicate_member(tmp_path_factory, stamp) -> None:
    """A database built from the current template but stamped with an older version (as the v77 storage tests
    build theirs): its ``capacity_margin_methods`` list already has the v80 names, and the v79 step adds
    ``share_of_peak`` beside them.  The v80 step drops the old members instead of renaming them into
    duplicates, rewrites a stored old value to its new name and leaves the shipped list."""
    from tests.engine._unit_db import b64

    data = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "tests.json").read_text())
    for pd in data["parameter_definitions"]:
        if pd[0] == "model" and pd[1] == "version":
            pd[2] = b64(stamp, "float")
    assert ["capacity_margin_methods", b64("relative_of_peak_demand", "str")] in data["parameter_value_lists"]
    work = tmp_path_factory.mktemp(f"v80_stamp{stamp}")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        assert {"share_of_peak", "absolute", "relative_of_peak_demand"} <= set(
            _lists(db)["capacity_margin_methods"])
        group = sorted(e["name"] for e in db.get_items("entity", entity_class_name="group"))[0]
        alt = sorted(a["name"] for a in db.get_items("alternative"))[0]
        val, typ = to_database("share_of_peak")
        db.add_update_item("parameter_value", entity_class_name="group", entity_byname=(group,),
                           parameter_definition_name="capacity_margin_method", alternative_name=alt,
                           value=val, type=typ)
        pdef = db.get_item("parameter_definition", entity_class_name="group", name="capacity_margin_method")
        db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
        db.commit_session("an old-name method value and default")
    migrate_database(url, up_to=80)
    _, shipped_lists = _shipped()
    with DatabaseMapping(url) as db:
        members = _lists(db)["capacity_margin_methods"]
        assert sorted(members) == sorted(shipped_lists["capacity_margin_methods"])
        assert len(members) == len(set(members))
        stored = db.get_item("parameter_value", entity_class_name="group", entity_byname=(group,),
                             parameter_definition_name="capacity_margin_method", alternative_name=alt)
        assert stored["parsed_value"] == "relative_of_peak_demand"
        pdef = db.get_item("parameter_definition", entity_class_name="group", name="capacity_margin_method")
        assert from_database(pdef["default_value"], pdef["default_type"]) == "relative_of_peak_demand"
    migrate_database(url, up_to=80)
    with DatabaseMapping(url) as db:
        assert _lists(db)["capacity_margin_methods"] == members
