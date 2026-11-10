"""Database migration to version 79 (all schema changes of one release in one step).

Seed: ``data/v75_seed_stoch_two_period.json`` (a frozen version-75 copy of
``tests/fixtures/stoch_two_period.json``) migrated to 78, then to 79.  The step

* adds ``constraint_online_coeff`` on unit and connection (``1d_map``, group
  ``constraint``, no default, the shipped description);
* rewrites the ``ramp_method`` / ``ramp_cost`` descriptions (``ramp_cost`` is charged);
* adds the value lists and definitions ``solve.reduce_unit_detail_method`` /
  ``reduce_system_detail_method`` (default ``none``, group ``solve_advanced``);
* widens ``group.inertia_limit`` / ``non_synchronous_limit`` to float / 1d_map /
  2d_map and rewrites their descriptions;
* rewrites the ``node.energy_margin_adder`` description, the ``unit.efficiency`` /
  ``efficiency_at_min_load`` descriptions (may be above 1), the 15 user-constraint
  capacity coefficient descriptions (indexed by constraint name) and the
  ``timeset_duration`` typo;
* writes the behaviour-change markers only on a database whose results change (a
  solve with ``new_stepduration``; with also an ``energy_margin_adder`` value, the
  adder marker too).

It changes no other definition and no value, and a rerun commits nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
PREFIX = m.BEHAVIOUR_CHANGE_MARKER_PREFIX

ONLINE_NEW = [("unit", "constraint_online_coeff"), ("connection", "constraint_online_coeff")]
RAMP_CHANGED = [(cls, name) for cls in ("unit__inputNode", "unit__outputNode") for name in ("ramp_method", "ramp_cost")]
REDUCE_NEW = {("solve", "reduce_unit_detail_method"): "reduce_unit_detail_methods",
              ("solve", "reduce_system_detail_method"): "reduce_system_detail_methods"}
UNIT_VALUES = ["none", "integers", "integers_ramps", "integers_min_limits", "integers_ramps_min_limits", "online",
               "online_ramps"]
SYSTEM_VALUES = ["none", "reserves", "inertia", "non_synchronous", "inertia_non_synchronous",
                 "reserves_inertia_non_synchronous", "dc_power_flow",
                 "reserves_inertia_non_synchronous_dc_power_flow"]
GROUP_LIMITS_CHANGED = [("group", "inertia_limit"), ("group", "non_synchronous_limit")]
ADDER = ("node", "energy_margin_adder")
DURATION = ("timeset", "timeset_duration")
EFFICIENCY_CHANGED = {("unit", name) for name in m._V79_UNIT_EFFICIENCY_DESCRIPTIONS}
COEFF_CHANGED = set(m._V79_CONSTRAINT_COEFF_PARAMETERS)
#: Every definition the step adds or changes.
ALL_CHANGED = (set(ONLINE_NEW) | set(RAMP_CHANGED) | set(REDUCE_NEW) | set(GROUP_LIMITS_CHANGED)
               | {ADDER, DURATION} | EFFICIENCY_CHANGED | COEFF_CHANGED)


def _defs(db) -> dict:
    """(class, name) -> (types, group, description, default value, value list, parsed default)."""
    return {(d["entity_class_name"], d["name"]): (tuple(d["parameter_type_list"] or ()), d["parameter_group_name"],
                                                  d["description"], d["default_value"],
                                                  d["parameter_value_list_name"],
                                                  from_database(d["default_value"], d["default_type"]))
            for d in db.get_items("parameter_definition")}


def _lists(db) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for x in sorted(db.get_items("list_value"), key=lambda x: (x["parameter_value_list_name"], x["index"])):
        out.setdefault(x["parameter_value_list_name"], []).append(x["parsed_value"])
    return out


def _markers(db) -> dict:
    return {md["name"][len(PREFIX):]: md["value"] for md in db.get_items("metadata")
            if md["name"].startswith(PREFIX)}


def _shipped() -> dict:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    return {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}


def _set(db, cls: str, name: str, param: str, value: float, alt: str) -> None:
    val, typ = to_database(value)
    db.add_parameter_value(entity_class_name=cls, entity_byname=(name,), parameter_definition_name=param,
                           alternative_name=alt, value=val, type=typ)


def _migrate(tmp_path_factory, tag: str, values: dict) -> dict:
    url = json_to_db(SEED, tmp_path_factory.mktemp(tag) / "seed.sqlite")
    migrate_database(url, up_to=78)
    with DatabaseMapping(url) as db:
        alt = sorted(a["name"] for a in db.get_items("alternative"))[0]
        solve = sorted(e["name"] for e in db.get_items("entity", entity_class_name="solve"))[0]
        node = sorted(e["name"] for e in db.get_items("entity", entity_class_name="node"))[0]
        if "solve" in values:
            _set(db, "solve", solve, "new_stepduration", values["solve"], alt)
        if "node" in values:
            _set(db, "node", node, "energy_margin_adder", values["node"], alt)
        if values:
            db.commit_session("test values")
        before, lists_before, markers_before = _defs(db), _lists(db), _markers(db)
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        after, lists_after, markers = _defs(db), _lists(db), _markers(db)
    return {"url": url, "before": before, "after": after, "lists_before": lists_before, "lists_after": lists_after,
            "markers_before": markers_before, "markers": markers, "solve": solve, "node": node}


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _migrate(tmp_path_factory, "v79_plain", {})


# ─── constraint_online_coeff ─────────────────────────────────────────────


def test_adds_the_online_coeff_definitions(plain) -> None:
    shipped = _shipped()
    for key in ONLINE_NEW:
        assert key not in plain["before"]
        types, group, description, default = plain["after"][key][:4]
        assert types == ("1d_map",)
        assert group == "constraint"
        assert default is None
        assert description == shipped[key]
        assert description.startswith("[unit of the constraint per unit online]")


# ─── ramp_method / ramp_cost descriptions ────────────────────────────────


def test_rewrites_the_ramp_descriptions(plain) -> None:
    shipped = _shipped()
    for key in RAMP_CHANGED:
        description = plain["after"][key][2]
        assert description == shipped[key]
        assert "not functional" not in description
        assert plain["before"][key][2] != description
        assert plain["after"][key][:2] == plain["before"][key][:2]
        assert plain["after"][key][3] == plain["before"][key][3]


# ─── reduce_unit_detail_method / reduce_system_detail_method ────────────


def test_adds_the_reduce_detail_definitions(plain) -> None:
    shipped = _shipped()
    for key, value_list in REDUCE_NEW.items():
        assert key not in plain["before"]
        types, group, description, _default, vl, default = plain["after"][key]
        assert types == ("str",)
        assert vl == value_list
        assert group == "solve_advanced"
        assert default == "none"
        assert description == shipped[key]
        assert description.startswith("[choice]")


def test_adds_the_value_lists_in_order(plain) -> None:
    assert "reduce_unit_detail_methods" not in plain["lists_before"]
    assert plain["lists_after"]["reduce_unit_detail_methods"] == UNIT_VALUES
    assert plain["lists_after"]["reduce_system_detail_methods"] == SYSTEM_VALUES
    assert tuple(UNIT_VALUES) == m.REDUCE_UNIT_DETAIL_METHODS
    assert tuple(SYSTEM_VALUES) == m.REDUCE_SYSTEM_DETAIL_METHODS


def test_completes_a_partial_list(tmp_path_factory) -> None:
    """A list that already holds some values (here only ``none`` and ``online``, at indexes 0 and 5) is completed:
    no value twice, and the added values take indexes after the largest one present."""
    url = json_to_db(SEED, tmp_path_factory.mktemp("v79_partial") / "seed.sqlite")
    migrate_database(url, up_to=78)
    with DatabaseMapping(url) as db:
        db.add_parameter_value_list(name="reduce_unit_detail_methods")
        for index, value in ((0, "none"), (5, "online")):
            val, typ = to_database(value)
            db.add_list_value(parameter_value_list_name="reduce_unit_detail_methods", value=val, type=typ,
                              index=index)
        db.commit_session("partial list")
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        items = sorted(db.get_items("list_value", parameter_value_list_name="reduce_unit_detail_methods"),
                       key=lambda x: x["index"])
    values = [x["parsed_value"] for x in items]
    assert sorted(values) == sorted(UNIT_VALUES) and len(values) == len(UNIT_VALUES)
    assert [x["index"] for x in items] == [0, 5, 6, 7, 8, 9, 10]
    assert values[:2] == ["none", "online"]


# ─── group inertia_limit / non_synchronous_limit per time step ──────────


def test_widens_the_group_limit_types_and_rewrites_the_descriptions(plain) -> None:
    shipped = _shipped()
    for key in GROUP_LIMITS_CHANGED:
        types, group, description, default = plain["after"][key][:4]
        assert plain["before"][key][0] == ("float", "1d_map")
        assert types == ("float", "1d_map", "2d_map")
        assert description == shipped[key]
        assert "Constant, period, time or period+time." in description
        assert plain["before"][key][2] != description
        assert (group, default) == (plain["before"][key][1], plain["before"][key][3])


# ─── time series combined over a solve's new_stepduration step ──────────


def test_rewrites_the_solve_step_series_descriptions(plain) -> None:
    shipped = _shipped()
    assert plain["after"][ADDER][2] == shipped[ADDER] == m._V79_ENERGY_MARGIN_ADDER_DESCRIPTION
    assert "sum of the values of the time steps it combines" in shipped[ADDER]
    for k in EFFICIENCY_CHANGED:
        assert plain["after"][k][2] == shipped[k] == m._V79_UNIT_EFFICIENCY_DESCRIPTIONS[k[1]]
        assert shipped[k].startswith("[-] ") and "above 1" in shipped[k]
    assert len(COEFF_CHANGED) == 15
    for k in COEFF_CHANGED:
        assert plain["after"][k][2] == shipped[k]
        assert shipped[k].startswith("[unit of the constraint per ")
        assert "] A map of coefficients (Index: constraint name, value: coefficient). Puts " in shipped[k]
    assert plain["after"][DURATION][2] == shipped[DURATION] and "the the" not in shipped[DURATION]
    for k in {ADDER, DURATION} | EFFICIENCY_CHANGED | COEFF_CHANGED:
        assert plain["before"][k][2] != shipped[k]
        assert plain["after"][k][:2] + plain["after"][k][3:] == plain["before"][k][:2] + plain["before"][k][3:]


def test_no_marker_without_new_stepduration(tmp_path_factory, plain) -> None:
    assert plain["markers"] == plain["markers_before"]
    adder_only = _migrate(tmp_path_factory, "v79_adder_only", {"node": 10.0})
    assert adder_only["markers"] == adder_only["markers_before"]


def test_markers_name_the_solves_and_nodes(tmp_path_factory) -> None:
    step = _migrate(tmp_path_factory, "v79_step", {"solve": 2.0})
    assert json.loads(step["markers"]["solve_step_series_combined"]) == {"since": 79, "solves": [step["solve"]]}
    assert "energy_margin_adder_per_original_step" not in step["markers"]
    both = _migrate(tmp_path_factory, "v79_both", {"solve": 2.0, "node": 10.0})
    assert json.loads(both["markers"]["energy_margin_adder_per_original_step"]) == {
        "since": 79, "solves": [both["solve"]], "nodes": [both["node"]]}
    with DatabaseMapping(both["url"]) as db:
        m._v79_solve_step_series(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
    with DatabaseMapping(both["url"]) as db:
        m._migrate_v79_schema_batch(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


# ─── the step as a whole ─────────────────────────────────────────────────


def test_changes_nothing_else(plain) -> None:
    skip = ALL_CHANGED | {("model", "version")}
    assert ({k: v for k, v in plain["after"].items() if k not in skip}
            == {k: v for k, v in plain["before"].items() if k not in skip})
    lists_after = {k: v for k, v in plain["lists_after"].items() if k not in REDUCE_NEW.values()}
    assert lists_after == plain["lists_before"]


@pytest.mark.parametrize("part", [
    "_v79_constraint_online_coeff", "_v79_ramp_cost_descriptions", "_v79_reduce_detail_methods",
    "_v79_group_limits_by_time", "_v79_solve_step_series", "_migrate_v79_schema_batch"])
def test_rerun_is_idempotent(plain, part: str) -> None:
    with DatabaseMapping(plain["url"]) as db:
        getattr(m, part)(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
