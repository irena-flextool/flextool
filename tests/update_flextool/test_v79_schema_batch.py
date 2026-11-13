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
  adder marker too);
* adds ``share_of_peak`` to ``capacity_margin_methods`` and ``group.capacity_margin_share``
  (float / 1d_map, group ``investment``, no default), rewrites the ``capacity_margin_method`` /
  ``capacity_margin`` descriptions, and writes the marker ``capacity_margin_counts_by_flow``
  only on a database with a ``manual`` capacity-margin group;
* adds the value list ``constraint_period_methods`` (``per_timestep``, ``period_total``) and
  ``constraint.constraint_period_method`` (``str``, default ``per_timestep``, group ``constraint``);
* rewrites the ``discount_rate`` / ``lifetime`` descriptions of unit / connection / node (0 is the
  straight-line annuity; lifetime required) and writes the marker ``discount_rate_zero_is_zero`` only on a
  database where an investing entity has an explicit ``discount_rate`` ≤ 0;
* writes the marker ``invest_without_cost_offered`` only on a database where an entity that can invest
  (retire) has no ``invest_cost`` (``salvage_value``) or a 0 in it;
* adds the value list ``startup_input_methods`` (``just_input``, ``startup_and_input``, ``startup_only``),
  ``unit__inputNode.startup_input_method`` (``str``, default ``just_input``) and ``unit__inputNode.startup_energy``
  (float / 1d_map, no default), both in group ``tech_advanced``;
* adds ``node.unit_of_measure`` (``str``, default ``MWh``, no value list, group ``basics``) with the description that
  explains the per-kW cost rule.
* rewrites the ``node.storage_binding_method`` / ``timeset.representative_period_weights`` descriptions
  (representative periods work in stochastic solves; blended methods bind forward only in a rolling solve).
* writes the marker ``stochastic_storage_paths_tied`` only on a database with a stochastic solve and a storage
  node on plain ``bind_within_solve``, or on ``bind_forward_only`` (stored or the default) whose
  ``storage_start_end_method`` is not ``fix_start`` / ``fix_start_end`` in every alternative that stores one (none
  stored: the default ``fix_nothing``).
* rewrites the ``solve.solver_precommand`` description (never run, kept so older databases read) and drops it from
  the ``solve.solver_arguments`` description.
* writes the marker ``chained_solve_discount_from_run_start`` only on a database with an ``inflation_rate`` other
  than 0 and a solve that can start after the run start (a ``model.solves`` list of two or more, ``contains_solves``,
  a 2-D period map) or has a fractional ``years_represented`` or one out of the solve's period order.

It changes no other definition and no value, and a rerun commits nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, Map, from_database, to_database
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
SHARE_NEW = ("group", "capacity_margin_share")
MARGIN_CHANGED = {("group", "capacity_margin_method"), ("group", "capacity_margin")}
PERIOD_METHOD_NEW = ("constraint", "constraint_period_method")
DISCOUNT_CHANGED = {(cls, name) for cls in ("unit", "connection", "node") for name in ("discount_rate", "lifetime")}
STARTUP_NEW = {("unit__inputNode", "startup_input_method"), ("unit__inputNode", "startup_energy")}
UOM_NEW = ("node", "unit_of_measure")
RP_CHANGED = {("node", "storage_binding_method"), ("timeset", "representative_period_weights")}
PRECOMMAND_CHANGED = {("solve", "solver_precommand"), ("solve", "solver_arguments")}
#: Every definition the step adds or changes.
ALL_CHANGED = (set(ONLINE_NEW) | set(RAMP_CHANGED) | set(REDUCE_NEW) | set(GROUP_LIMITS_CHANGED)
               | {ADDER, DURATION} | EFFICIENCY_CHANGED | COEFF_CHANGED | {SHARE_NEW} | MARGIN_CHANGED
               | {PERIOD_METHOD_NEW} | DISCOUNT_CHANGED | STARTUP_NEW | {UOM_NEW} | RP_CHANGED
               | PRECOMMAND_CHANGED)


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


def _rerun_changes_nothing(db, step, monkeypatch) -> None:
    """Run ``step`` again on ``db`` and check that it changes nothing.  The batch commits through
    ``m._commit_step``, which swallows ``NothingToCommit``: it is replaced by a check that the commit has nothing to
    commit (and the batch must commit once); a part does not commit, so the session is committed here."""
    commits: list[str] = []

    def commit_must_be_empty(db_, message) -> None:
        with pytest.raises(NothingToCommit):
            db_.commit_session(message)
        commits.append(message)

    monkeypatch.setattr(m, "_commit_step", commit_must_be_empty)
    step(db)
    if step is m._migrate_v79_schema_batch:
        assert len(commits) == 1, commits
    else:
        assert commits == []
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


def _shipped() -> dict:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    return {(r[0], r[1]): r[4] for r in schema["parameter_definitions"]}


def _set(db, cls: str, name: str, param: str, value, alt: str) -> None:
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
        if "group" in values:
            _set(db, "group", "stoch_g", "capacity_margin_method", values["group"], alt)
        unit = sorted(e["name"] for e in db.get_items("entity", entity_class_name="unit"))[0]
        if "unit_disc" in values:
            _set(db, "unit", unit, "invest_method", "invest_no_limit", alt)
            _set(db, "unit", unit, "discount_rate", values["unit_disc"], alt)
        if "storage_method" in values:
            _set(db, "node", node, "node_type", "storage", alt)
            if values["storage_method"] is not None:
                _set(db, "node", node, "storage_binding_method", values["storage_method"], alt)
            for start_alt, start in values.get("storage_start", {}).items():
                _set(db, "node", node, "storage_start_end_method", start, start_alt)
        if "unit_invest_cost" in values:
            _set(db, "unit", unit, "invest_method", "invest_no_limit", alt)
            if values["unit_invest_cost"] is not None:
                _set(db, "unit", unit, "invest_cost", values["unit_invest_cost"], alt)
        if "inflation" in values:
            _set(db, "model", "flextool", "inflation_rate", values["inflation"], alt)
        if "model_solves" in values:
            existing = db.get_parameter_value_item(entity_class_name="model", entity_byname=("flextool",),
                                                   parameter_definition_name="solves", alternative_name=alt)
            val, typ = to_database(Array(values["model_solves"]))
            if existing:
                db.update_parameter_value(id=existing["id"], value=val, type=typ)
            else:
                _set(db, "model", "flextool", "solves", Array(values["model_solves"]), alt)
        if "years" in values:
            _set(db, "solve", solve, "years_represented", values["years"], alt)
        if "contains" in values:
            _set(db, "solve", solve, "contains_solves", Array(values["contains"]), alt)
        if "split" in values:
            _set(db, "solve", solve, "invest_periods", values["split"], alt)
        if values:
            db.commit_session("test values")
        before, lists_before, markers_before = _defs(db), _lists(db), _markers(db)
    migrate_database(url, up_to=79)
    with DatabaseMapping(url) as db:
        after, lists_after, markers = _defs(db), _lists(db), _markers(db)
    return {"url": url, "before": before, "after": after, "lists_before": lists_before, "lists_after": lists_after,
            "markers_before": markers_before, "markers": markers, "solve": solve, "node": node, "unit": unit}


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


def test_markers_name_the_solves_and_nodes(tmp_path_factory, monkeypatch) -> None:
    step = _migrate(tmp_path_factory, "v79_step", {"solve": 2.0})
    assert json.loads(step["markers"]["solve_step_series_combined"]) == {"since": 79, "solves": [step["solve"]]}
    assert "energy_margin_adder_per_original_step" not in step["markers"]
    both = _migrate(tmp_path_factory, "v79_both", {"solve": 2.0, "node": 10.0})
    assert json.loads(both["markers"]["energy_margin_adder_per_original_step"]) == {
        "since": 79, "solves": [both["solve"]], "nodes": [both["node"]]}
    with DatabaseMapping(both["url"]) as db:
        _rerun_changes_nothing(db, m._v79_solve_step_series, monkeypatch)
    with DatabaseMapping(both["url"]) as db:
        _rerun_changes_nothing(db, m._migrate_v79_schema_batch, monkeypatch)


# ─── capacity margin by entity class, share_of_peak ─────────────────────


def test_adds_share_of_peak_and_the_share(plain) -> None:
    shipped = _shipped()
    assert plain["lists_before"]["capacity_margin_methods"] == ["none", "manual"]
    assert plain["lists_after"]["capacity_margin_methods"] == ["none", "manual", "share_of_peak"]
    assert SHARE_NEW not in plain["before"]
    types, group, description, default, value_list = plain["after"][SHARE_NEW][:5]
    assert types == ("float", "1d_map") and group == "investment" and default is None and value_list is None
    assert description == shipped[SHARE_NEW] and description.startswith("[-] ")
    for key in MARGIN_CHANGED:
        assert plain["after"][key][2] == shipped[key] != plain["before"][key][2]
        assert plain["after"][key][:2] + plain["after"][key][3:] == plain["before"][key][:2] + plain["before"][key][3:]
    assert "share_of_peak" in shipped[("group", "capacity_margin_method")]


def test_capacity_margin_marker_only_with_a_manual_group(tmp_path_factory, plain, monkeypatch) -> None:
    assert "capacity_margin_counts_by_flow" not in plain["markers"]
    none = _migrate(tmp_path_factory, "v79_cm_none", {"group": "none"})
    assert "capacity_margin_counts_by_flow" not in none["markers"]
    manual = _migrate(tmp_path_factory, "v79_cm_manual", {"group": "manual"})
    assert json.loads(manual["markers"]["capacity_margin_counts_by_flow"]) == {"since": 79, "groups": ["stoch_g"]}
    with DatabaseMapping(manual["url"]) as db:
        _rerun_changes_nothing(db, m._v79_capacity_margin, monkeypatch)


# ─── constraint_period_method ────────────────────────────────────────────


def test_adds_the_constraint_period_method(plain) -> None:
    shipped = _shipped()
    assert PERIOD_METHOD_NEW not in plain["before"]
    assert "constraint_period_methods" not in plain["lists_before"]
    assert plain["lists_after"]["constraint_period_methods"] == ["per_timestep", "period_total"]
    types, group, description, _default, value_list, default = plain["after"][PERIOD_METHOD_NEW]
    assert types == ("str",) and group == "constraint" and value_list == "constraint_period_methods"
    assert default == "per_timestep"
    assert description == shipped[PERIOD_METHOD_NEW] and description.startswith("[choice]")


# ─── discount_rate 0 is 0 % ──────────────────────────────────────────────


def test_rewrites_the_discount_rate_and_lifetime_descriptions(plain) -> None:
    shipped = _shipped()
    for key in DISCOUNT_CHANGED:
        assert plain["after"][key][2] == shipped[key] != plain["before"][key][2]
        assert plain["after"][key][:2] + plain["after"][key][3:] == plain["before"][key][:2] + plain["before"][key][3:]
    assert "0 gives the straight-line annuity invest_cost / lifetime" in shipped[("unit", "discount_rate")]
    assert "Required when the node invests or retires with a cost" in shipped[("node", "lifetime")]


def test_discount_rate_marker_only_for_an_explicit_zero(tmp_path_factory, plain, monkeypatch) -> None:
    assert "discount_rate_zero_is_zero" not in plain["markers"]
    five = _migrate(tmp_path_factory, "v79_disc_five", {"unit_disc": 0.05})
    assert "discount_rate_zero_is_zero" not in five["markers"]
    zero = _migrate(tmp_path_factory, "v79_disc_zero", {"unit_disc": 0.0})
    assert json.loads(zero["markers"]["discount_rate_zero_is_zero"]) == {"since": 79, "entities": [zero["unit"]]}
    with DatabaseMapping(zero["url"]) as db:
        _rerun_changes_nothing(db, m._v79_discount_rate_zero, monkeypatch)


# ─── investment offered without invest_cost ──────────────────────────────


def test_invest_without_cost_marker_only_without_a_cost(tmp_path_factory, plain, monkeypatch) -> None:
    """The marker names an entity that can invest without an invest_cost or with a 0 in it (a period map
    counts by its elements); a non-zero cost, or no investment, writes none."""
    assert "invest_without_cost_offered" not in plain["markers"]
    costed = _migrate(tmp_path_factory, "v79_inv_cost", {"unit_invest_cost": 1000.0})
    assert "invest_without_cost_offered" not in costed["markers"]
    for tag, cost in (("v79_inv_none", None), ("v79_inv_zero", 0.0)):
        free = _migrate(tmp_path_factory, tag, {"unit_invest_cost": cost})
        assert json.loads(free["markers"]["invest_without_cost_offered"]) == {"since": 79,
                                                                              "entities": [free["unit"]]}
    with DatabaseMapping(free["url"]) as db:
        _rerun_changes_nothing(db, m._v79_invest_without_cost, monkeypatch)


# ─── storage along each stochastic path ──────────────────────────────────


def test_stochastic_storage_marker_only_with_plain_storage(tmp_path_factory, plain, monkeypatch) -> None:
    """The seed's solve is stochastic: the marker comes with a storage node on bind_within_solve (also with
    fix_start) or on the default bind_forward_only with a start that is not fixed in every alternative storing
    one; not with a blended method, a bind_forward_only node whose every stored start method fixes the start, or
    without a storage node."""
    assert "stochastic_storage_paths_tied" not in plain["markers"]
    cases = {
        "v79_ssp_blended": ({"storage_method": "bind_within_period_blended_weights"}, False),
        "v79_ssp_fo_fix_start": ({"storage_method": None, "storage_start": {"base": "fix_start"}}, False),
        "v79_ssp_fo_fix_both": ({"storage_method": "bind_forward_only",
                                 "storage_start": {"base": "fix_start", "stoch": "fix_start_end"}}, False),
        "v79_ssp_default": ({"storage_method": None}, True),
        "v79_ssp_ws": ({"storage_method": "bind_within_solve"}, True),
        "v79_ssp_ws_fix_start": ({"storage_method": "bind_within_solve", "storage_start": {"base": "fix_start"}},
                                 True),
        "v79_ssp_fo_free_somewhere": ({"storage_method": None,
                                       "storage_start": {"base": "fix_start", "stoch": "fix_nothing"}}, True),
        "v79_ssp_fo_fix_end": ({"storage_method": None, "storage_start": {"base": "fix_end"}}, True),
    }
    for tag, (values, marked) in cases.items():
        run = _migrate(tmp_path_factory, tag, values)
        if marked:
            assert json.loads(run["markers"]["stochastic_storage_paths_tied"]) == {"since": 79,
                                                                                   "solves": [run["solve"]]}, tag
            tied = run
        else:
            assert "stochastic_storage_paths_tied" not in run["markers"], tag
    with DatabaseMapping(tied["url"]) as db:
        _rerun_changes_nothing(db, m._v79_stochastic_storage_paths, monkeypatch)


# ─── every solve discounted from the run start ───────────────────────────

DISCOUNT_MARKER = "chained_solve_discount_from_run_start"


def test_discount_marker_only_with_inflation_and_a_later_or_fractional_solve(tmp_path_factory, plain,
                                                                             monkeypatch) -> None:
    """The marker needs an inflation_rate other than 0 and a solve that can start after the run start (two or more
    solves in model.solves, contains_solves, a 2-D period map) or a fractional or out-of-order years_represented; it names those
    solves.  The seed's one solve without inflation, with inflation alone, or two solves without inflation write
    none."""
    assert DISCOUNT_MARKER not in plain["markers"]
    two = ["stoch_2p", "later"]
    split = Map(["p1", "p2"], [Map(["p1"], ["yes"]), Map(["p2"], ["yes"])])
    cases = {
        "v79_dfr_one": ({"inflation": 0.05}, None),
        "v79_dfr_zero": ({"inflation": 0.0, "model_solves": two}, None),
        "v79_dfr_no_inflation": ({"model_solves": two}, None),
        "v79_dfr_whole": ({"inflation": 0.05, "years": Map(["p1"], [5.0])}, None),
        "v79_dfr_two": ({"inflation": 0.05, "model_solves": two}, sorted(two)),
        "v79_dfr_contains": ({"inflation": 0.05, "contains": ["inner"]}, ["stoch_2p"]),
        "v79_dfr_split": ({"inflation": 0.05, "split": split}, ["stoch_2p"]),
        "v79_dfr_fraction": ({"inflation": 0.05, "years": Map(["p1", "p2"], [2.5, 5.0])}, ["stoch_2p"]),
        # The seed solve realizes p2035, p2040: a map in that order (with an unused period after) writes none, one
        # out of that order, or with an unused period before a used one, names the single solve.
        "v79_dfr_in_order": ({"inflation": 0.05, "years": Map(["p2035", "p2040", "p9"], [5.0, 5.0, 5.0])}, None),
        "v79_dfr_out_of_order": ({"inflation": 0.05, "years": Map(["p2040", "p2035"], [5.0, 5.0])}, ["stoch_2p"]),
        "v79_dfr_unused_first": ({"inflation": 0.05, "years": Map(["p9", "p2035", "p2040"], [5.0, 5.0, 5.0])},
                                 ["stoch_2p"]),
    }
    for tag, (values, solves) in cases.items():
        run = _migrate(tmp_path_factory, tag, values)
        if solves is None:
            assert DISCOUNT_MARKER not in run["markers"], tag
        else:
            assert json.loads(run["markers"][DISCOUNT_MARKER]) == {"since": 79, "solves": solves}, tag
            marked = run
    with DatabaseMapping(marked["url"]) as db:
        _rerun_changes_nothing(db, m._v79_discount_from_run_start, monkeypatch)


# ─── startup_input_method / startup_energy ───────────────────────────────


def test_adds_the_startup_input_definitions(plain) -> None:
    shipped = _shipped()
    assert not STARTUP_NEW & set(plain["before"])
    assert "startup_input_methods" not in plain["lists_before"]
    assert plain["lists_after"]["startup_input_methods"] == ["just_input", "startup_and_input", "startup_only"]
    types, group, description, _default, value_list, default = plain["after"][("unit__inputNode",
                                                                              "startup_input_method")]
    assert types == ("str",) and group == "tech_advanced" and value_list == "startup_input_methods"
    assert default == "just_input"
    assert description == shipped[("unit__inputNode", "startup_input_method")] and description.startswith("[choice]")
    assert "heat" in description
    types, group, description, _default, value_list, default = plain["after"][("unit__inputNode", "startup_energy")]
    assert types == ("float", "1d_map") and group == "tech_advanced" and value_list is None and default is None
    assert description == shipped[("unit__inputNode", "startup_energy")]
    assert description.startswith("[MWh per MW started]")


# ─── node.unit_of_measure ────────────────────────────────────────────────


def test_adds_the_node_unit_of_measure(plain) -> None:
    assert UOM_NEW not in plain["before"]
    types, group, description, _default, value_list, default = plain["after"][UOM_NEW]
    assert types == ("str",) and group == "basics" and value_list is None and default == "MWh"
    assert description == _shipped()[UOM_NEW] == m.V79_NODE_UNIT_OF_MEASURE_DESCRIPTION
    assert description.startswith("[text]")
    # The per-kW rule, with its worked number, is explained here (owner decision 2).
    assert "per 0.001 of the rate or amount" in description and "entered as 0.5" in description


# ─── representative periods in stochastic and rolling solves ─────────────


def test_rewrites_the_rp_descriptions(plain) -> None:
    shipped = _shipped()
    for key in RP_CHANGED:
        assert plain["after"][key][2] == shipped[key] == m._v79_rp_texts()[key] != plain["before"][key][2]
        assert plain["after"][key][:2] + plain["after"][key][3:] == plain["before"][key][:2] + plain["before"][key][3:]
    assert "Not allowed in a stochastic solve" in plain["before"][("timeset", "representative_period_weights")][2]
    assert "bind forward only (bind_forward_only), with a warning" in shipped[("node", "storage_binding_method")]
    assert "do not work with stochastic" not in shipped[("node", "storage_binding_method")]


# ─── solver_precommand is never run ──────────────────────────────────────


def test_rewrites_the_solver_precommand_descriptions(plain) -> None:
    shipped = _shipped()
    for key in PRECOMMAND_CHANGED:
        assert plain["after"][key][2] == shipped[key] == m._v79_solver_precommand_texts()[key] != plain["before"][key][2]
        assert plain["after"][key][:2] + plain["after"][key][3:] == plain["before"][key][:2] + plain["before"][key][3:]
    assert "never runs" in shipped[("solve", "solver_precommand")]
    assert "solver_precommand" not in shipped[("solve", "solver_arguments")]


# ─── the step as a whole ─────────────────────────────────────────────────


def test_changes_nothing_else(plain) -> None:
    skip = ALL_CHANGED | {("model", "version")}
    assert ({k: v for k, v in plain["after"].items() if k not in skip}
            == {k: v for k, v in plain["before"].items() if k not in skip})
    lists_after = {k: v for k, v in plain["lists_after"].items()
                   if k not in {*REDUCE_NEW.values(), "constraint_period_methods", "startup_input_methods"}}
    lists_before = dict(plain["lists_before"])
    lists_before["capacity_margin_methods"] = [*lists_before["capacity_margin_methods"], "share_of_peak"]
    assert lists_after == lists_before


@pytest.mark.parametrize("part", [
    "_v79_constraint_online_coeff", "_v79_ramp_cost_descriptions", "_v79_reduce_detail_methods",
    "_v79_group_limits_by_time", "_v79_solve_step_series", "_v79_capacity_margin", "_v79_constraint_period_method",
    "_v79_discount_rate_zero", "_v79_startup_input", "_v79_invest_without_cost", "_v79_node_unit_of_measure",
    "_v79_rp_stochastic_descriptions", "_v79_stochastic_storage_paths", "_v79_solver_precommand_descriptions",
    "_v79_discount_from_run_start", "_migrate_v79_schema_batch"])
def test_rerun_is_idempotent(plain, part: str, monkeypatch) -> None:
    with DatabaseMapping(plain["url"]) as db:
        _rerun_changes_nothing(db, getattr(m, part), monkeypatch)


def test_rerun_check_fails_when_the_batch_changes_something(tmp_path_factory, monkeypatch) -> None:
    """The batch rerun check can fail: a batch rerun that changes a description is caught (on a database of its
    own, since the check commits the change)."""
    own = _migrate(tmp_path_factory, "v79_rerun_check", {})
    with DatabaseMapping(own["url"]) as db:
        def changing_batch(db_) -> None:
            pdef = db_.get_item("parameter_definition", entity_class_name="solve", name="reduce_unit_detail_method")
            db_.update_parameter_definition(id=pdef["id"], description="changed")
            m._commit_step(db_, "v79 rerun")

        monkeypatch.setattr(m, "_migrate_v79_schema_batch", changing_batch)
        with pytest.raises(pytest.fail.Exception):
            _rerun_changes_nothing(db, m._migrate_v79_schema_batch, monkeypatch)
