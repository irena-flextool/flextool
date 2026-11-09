"""Solve-level unit-detail reduction (``solve.reduce_unit_detail_method``, database version 81).

A level only removes or relaxes detail that the cascade produced from the entity data
(``preprocess._reduce_detail.apply_reduce_detail``, run on each sub-solve right after ``load_flextool``).

Hand-built model (on ``tests/fixtures/tests.json``, alternative ``init`` only: one period p2020, timeset ``2day``
cut to two 1-h steps; never a checked-in ``.sqlite``): node ``elec`` with a demand of 100 MW, commodity node
``fuel`` at 20 CUR/MWh, unit ``g1`` fuel -> elec (existing 100 MW, ``min_load_efficiency`` with efficiency 0.5,
efficiency_at_min_load 0.4, min_load 0.5, binary startup, min_uptime 2 h, ramp_method both with ramp_speed 0.005 per minute and
ramp_cost 1 on its output), an ``exact`` connection ``c1`` elec -> elec2 (100 MW) with a 10 MW demand at ``elec2``,
and the user constraint ``c_online`` (units online of g1 ≥ 0.5).

Hand values (design solve_relaxation_design.md, implementation brief WP1 test 3, independent of the code): with
η_full 0.5, η_min 0.4 and min_load 0.5 the curve is section 0.5 · (1/0.4 − 1/0.5) / (1 − 0.5) = 0.5 and slope
1/0.5 − 0.5 = 1.5 (per MW of output and per MW online).  Fuel at 100 MW: 1.5 · 100 + 0.5 · 100 = 200; at 50 MW (one
unit online): 75 + 50 = 125.  Under ``online`` the slope is 1.5 + 0.5 = 2: 200 at 100 MW and 100 at 50 MW.  With the
output conversion_flow_coeff 2 (k_sink 2, k_src 1) the folded slope is 1.5 · 2 = 3 and the section 0.5: fuel 350 /
200 at 100 / 50 MW without reduction, 3.5 · output = 350 / 175 under ``online`` (the full-load input
(slope⁰·k_sink + section⁰)/k_src is kept).  With the input conversion_flow_coeff 2 (k_src 2; units.md: input·2 =
output·slope⁰ + section⁰·online) the fuel flow is (1.5 · output + 0.5 · 100 · online)/2: (150 + 50)/2 = 100 at 100
MW and (75 + 50)/2 = 62.5 at 50 MW; under ``online`` 2 · output / 2 = output: 100 and 50 (folded slope 1).  The cost
is fuel only, so objective ratios are fuel ratios.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from flextool.engine.input._solve_config import REDUCE_SYSTEM_DETAIL_METHODS, REDUCE_UNIT_DETAIL_METHODS
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._reduce_detail import (
    MIN_LOAD_FIELDS, MIN_TIME_FIELDS, ONLINE_FIELDS, RAMP_FIELDS, apply_reduce_detail, system_parts, unit_parts,
)
from tests.engine._unit_db import RunCache, b64, build, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    """A missing autoscale registry entry raises instead of falling back to an unscaled LP."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _base_entities() -> list:
    return [["node", "elec", None], ["node", "fuel", None], ["commodity", "fuelc", None],
            ["group__node", ["electricity", "elec"], None], ["commodity__node", ["fuelc", "fuel"], None],
            ["unit", "g1", None], ["unit__inputNode", ["g1", "fuel"], None],
            ["unit__outputNode", ["g1", "elec"], None]]


def _base_values(demand: float = 100.0) -> list:
    return [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")],
            ["solve", SOLVE, "solver_mip_gap", _f(0.0)],
            ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-demand)],
            ["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)],
            ["unit", "g1", "conversion_method", _s("min_load_efficiency")], ["unit", "g1", "efficiency", _f(0.5)],
            ["unit", "g1", "efficiency_at_min_load", _f(0.4)], ["unit", "g1", "min_load", _f(0.5)],
            ["unit", "g1", "existing", _f(100.0)], ["unit", "g1", "startup_method", _s("binary")]]


def _full_entities() -> list:
    return _base_entities() + [
        ["node", "elec2", None], ["group__node", ["electricity", "elec2"], None],
        ["connection", "c1", None], ["connection__node__node", ["c1", "elec", "elec2"], None],
        ["constraint", "c_online", None]]


def _full_values(unit_level: str | None = None) -> list:
    out = "unit__outputNode"
    vals = _base_values() + [
        ["unit", "g1", "minimum_time_method", _s("min_uptime")], ["unit", "g1", "min_uptime", _f(2.0)],
        [out, ["g1", "elec"], "ramp_method", _s("both")], [out, ["g1", "elec"], "ramp_speed_up", _f(0.005)],
        [out, ["g1", "elec"], "ramp_speed_down", _f(0.005)], [out, ["g1", "elec"], "ramp_cost", _f(1.0)],
        ["node", "elec2", "node_type", _s("balance")], ["node", "elec2", "inflow", _f(-10.0)],
        ["connection", "c1", "existing", _f(100.0)], ["connection", "c1", "transfer_method", _s("exact")],
        ["unit", "g1", "constraint_online_coeff",
         b64({"index_type": "str", "index_name": "constraint", "data": [["c_online", 1.0]]}, "map")],
        ["constraint", "c_online", "sense", _s("greater_than")], ["constraint", "c_online", "constant", _f(0.5)]]
    if unit_level is not None:
        vals.append(["solve", SOLVE, "reduce_unit_detail_method", _s(unit_level)])
    return vals


def _solve(tmp_path_factory, alt: str, values: list, entities: list):
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=["init"], entities=entities)
    assert len(steps) == 1, list(steps)
    step = next(iter(steps.values()))
    assert step.solution is not None, "solve produced no solution"
    return step


def _all_families(pb) -> tuple[set[str], set[str], set[str]]:
    """(row families, column families, families of the integer columns) of a built problem."""
    m = pb.canonicalise()
    rows = {n.split("[", 1)[0] for n in m.row_names}
    cols = {n.split("[", 1)[0] for n in m.col_names}
    ints = {m.col_names[i].split("[", 1)[0] for i in range(len(m.col_names)) if m.col_int[i]}
    return rows, cols, ints


def _families(pb) -> tuple[set[str], set[str]]:
    """(row families, families of the integer columns) of a built problem."""
    m = pb.canonicalise()
    rows = {n.split("[", 1)[0] for n in m.row_names}
    ints = {m.col_names[i].split("[", 1)[0] for i in range(len(m.col_names)) if m.col_int[i]}
    return rows, ints


# ---------------------------------------------------------------------------
# value lists and resolution


def test_engine_values_are_the_schema_value_lists() -> None:
    """The resolver accepts exactly the v81 value lists, in order."""
    from flextool._resources import package_data_path

    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as fh:
        schema = json.load(fh)
    lists: dict[str, list[str]] = {}
    for row in schema["parameter_value_lists"]:
        lists.setdefault(row[0], []).append(row[1])
    assert tuple(lists["reduce_unit_detail_methods"]) == REDUCE_UNIT_DETAIL_METHODS
    assert tuple(lists["reduce_system_detail_methods"]) == REDUCE_SYSTEM_DETAIL_METHODS


def test_level_parts() -> None:
    """online contains every integers_min_limits effect; system values split in the fixed word order."""
    from flextool.engine.preprocess._reduce_detail import UNIT_PARTS

    assert tuple(UNIT_PARTS) == REDUCE_UNIT_DETAIL_METHODS
    assert unit_parts("none") == frozenset()
    assert unit_parts("integers_ramps_min_limits") == {"integers", "ramps", "min_limits"}
    assert unit_parts("online") == {"integers", "min_limits", "online"}
    assert unit_parts("online_ramps") == {"integers", "ramps", "min_limits", "online"}
    assert system_parts("none") == frozenset()
    assert system_parts("inertia_non_synchronous") == {"inertia", "non_synchronous"}
    assert system_parts("dc_power_flow") == {"dc_power_flow"}
    assert system_parts("reserves_inertia_non_synchronous_dc_power_flow") == {
        "reserves", "inertia", "non_synchronous", "dc_power_flow"}
    for bad in ("online_integers", "relax"):
        with pytest.raises(FlexToolConfigError):
            unit_parts(bad)
    with pytest.raises(FlexToolConfigError):
        system_parts("dc_power_flow_reserves")


def test_resolver_roll_realization_and_unknown(tmp_path_factory) -> None:
    """A roll and a realization solve take the base / parent solve's value; absent and blank are none; case is
    ignored; an unknown value raises naming the solve and the valid values."""
    from flextool.engine.input._solve_config import SolveConfig

    sc = object.__new__(SolveConfig)
    sc.reduce_unit_detail_method = {"inv": "Online ", "typo": "onlin", "blank": "  "}
    sc.reduce_system_detail_method = {"inv": "reserves"}
    sc.realized_dispatch_parent = {"inv__realized_dispatch": "inv"}
    assert sc.reduce_unit_detail_for("inv") == "online"
    assert sc.reduce_unit_detail_for("inv_roll_3") == "online"
    assert sc.reduce_unit_detail_for("inv__realized_dispatch") == "online"
    assert sc.reduce_unit_detail_for("inv__realized_dispatch_roll_2") == "online"
    assert sc.reduce_system_detail_for("inv__realized_dispatch") == "reserves"
    assert sc.reduce_unit_detail_for("other") == "none"
    assert sc.reduce_unit_detail_for("blank") == "none"
    assert sc.reduce_system_detail_for("other") == "none"
    with pytest.raises(FlexToolConfigError, match="solve 'typo'.*online_ramps"):
        sc.reduce_unit_detail_for("typo_roll_1")


# ---------------------------------------------------------------------------
# per level: the row families and integer columns


# The row / column families each part removes from the rd_full model (design §2.A, rows U1-U9): the exact direction
# switch (integers), the ramp limits and ramp cost with the start-up / shut-down capacity reduction of the
# ramp-limited online unit (ramps, _online._cap_reduction), min load and min up time (min_limits), every online row
# and column and the user constraint that counts units online (online).
_EXACT = ({"maxFlow_exact_forward", "maxFlow_exact_reverse"}, {"v_exact_forward"})
_RAMPS = ({"ramp_sink_up_constraint", "ramp_sink_down_constraint", "ramp_change_eq", "maxFlow_online_shutdown_integer"},
          {"v_ramp_up", "v_ramp_down"})
_MIN_LIMITS = ({"minFlow_minload_integer", "minimum_uptime_integer"}, set())
_ONLINE = ({"maxOnline_integer", "online_change_integer", "maxStartup_integer", "maxShutdown_integer",
            "maxFlow_online_integer", "maxFlow_online_shutdown_integer", "process_constraint_greater_than"},
           {"v_online_integer", "v_startup_integer", "v_shutdown_integer"})


def _removed(*parts) -> tuple[set[str], set[str]]:
    return set().union(*(p[0] for p in parts)), set().union(*(p[1] for p in parts))


@pytest.mark.parametrize("level, removed, ints", [
    ("none", _removed(), {"v_online_integer", "v_exact_forward"}),
    ("integers", _removed(_EXACT), set()),
    ("integers_ramps", _removed(_EXACT, _RAMPS), set()),
    ("integers_min_limits", _removed(_EXACT, _MIN_LIMITS), set()),
    ("integers_ramps_min_limits", _removed(_EXACT, _RAMPS, _MIN_LIMITS), set()),
    ("online", _removed(_EXACT, _MIN_LIMITS, _ONLINE), set()),
    ("online_ramps", _removed(_EXACT, _RAMPS, _MIN_LIMITS, _ONLINE), set()),
])
def test_row_families_per_level(tmp_path_factory, level, removed, ints) -> None:
    """Each level removes exactly its row and column families, adds none, and leaves no integer column unless
    none."""
    step = _solve(tmp_path_factory, "rd_full", _full_values(), _full_entities())
    none_rows, none_cols, _ = _all_families(build(step.flex_data))
    reduced, report = apply_reduce_detail(step.flex_data, level)
    rows, cols, int_families = _all_families(build(reduced))
    removed_rows, removed_cols = removed
    assert none_rows - rows == removed_rows, level
    assert none_cols - cols == removed_cols, level
    assert rows - none_rows == set() and cols - none_cols == set(), level
    assert int_families == ints
    if level == "none":
        assert reduced is step.flex_data and not report.changed
    else:
        assert report.removed["exact"] == ["c1"]
        assert report.removed["integer_online"] == ["g1"]
        # Integer online columns are relaxed in place (same family name) only while online variables remain.
        assert reduced.relax_integrality is (level.startswith("integers"))
    if level.startswith("online"):
        assert report.removed["user_constraints"] == ["c_online"]
        assert report.removed["min_load_efficiency"] == ["g1"]
        assert len(report.warnings) == 1 and "c_online" in report.warnings[0]
    elif level != "none":
        assert report.notes["fractional_online_constraints"] == ["c_online"]
        assert not report.warnings


def test_field_lists_cover_the_unit_fields(tmp_path_factory) -> None:
    """Every FlexData field of the unit-detail features is cleared by online_ramps or deliberately kept: a new
    field of these features must be added to one of the two lists."""
    import dataclasses
    import re

    from flextool.engine.preprocess.flex_data import FlexData

    kept = {
        "dt_ramp_link", "dt_online_wrap", "dt_ramp_carry",  # step predecessors of online_change and min up/down
        "p_startup_cost",  # unread without the start-up cost sets
        "p_slope", "p_section",  # folded, not cleared (min_load_efficiency)
        "process_min_load_eff", "process_exact", "p_process_constraint_online_coeff",  # cleared by name below
        "relax_integrality",
        "p_process_sink_max_capacity_coef",  # a capacity cap, not a min limit
        "ed_cumulative_min_capacity", "pd_group_cumulative_min_capacity",  # investment limits
    }
    cleared = set(RAMP_FIELDS + MIN_LOAD_FIELDS + MIN_TIME_FIELDS + ONLINE_FIELDS)
    pattern = re.compile(r"online|uptime|downtime|ramp|minload|min_load|exact|startup|min_capacity|mintime|integer")
    fields = {f.name for f in dataclasses.fields(FlexData) if pattern.search(f.name)}
    assert fields - cleared - kept == set()
    step = _solve(tmp_path_factory, "rd_full", _full_values(), _full_entities())
    reduced, _ = apply_reduce_detail(step.flex_data, "online_ramps")
    for name in cleared | {"process_min_load_eff", "process_exact", "p_process_constraint_online_coeff"}:
        assert getattr(reduced, name) is None, name


def test_ramp_carry_kept_without_ramps_and_cleared_with_ramps(tmp_path_factory) -> None:
    """The carried roll-end flow stays under integers / online (ramp limits and ramp cost still need it); every
    *ramps level clears it with the ramp-limit and ramp-cost sets (process_source_sink_ramp_cost, p_ramp_cost_arc,
    the source of the output par.process_ramp_cost)."""
    from polar_high import Param

    step = _solve(tmp_path_factory, "rd_full", _full_values(), _full_entities())
    fd = step.flex_data
    assert fd.process_source_sink_ramp_cost.height > 0 and fd.p_ramp_cost_arc is not None
    carry = Param(("p", "source", "sink", "d", "t"),
                  pl.DataFrame({"p": ["g1"], "source": ["g1"], "sink": ["elec"], "d": ["p2020"], "t": ["t0001"],
                                "value": [0.5]}))
    import copy
    with_carry = copy.copy(fd)
    with_carry.p_roll_continue_flow = carry
    with_carry.p_roll_continue_flow_duration = carry
    for level in ("integers", "integers_min_limits", "online"):
        reduced, _ = apply_reduce_detail(with_carry, level)
        assert reduced.p_roll_continue_flow is carry and reduced.p_roll_continue_flow_duration is carry, level
        assert reduced.process_source_sink_ramp_cost is fd.process_source_sink_ramp_cost, level
        assert reduced.p_ramp_cost_arc is fd.p_ramp_cost_arc, level
    for level in ("integers_ramps", "integers_ramps_min_limits", "online_ramps"):
        reduced, report = apply_reduce_detail(with_carry, level)
        assert reduced.p_roll_continue_flow is None and reduced.p_roll_continue_flow_duration is None, level
        assert reduced.process_source_sink_ramp_cost is None and reduced.p_ramp_cost_arc is None, level
        assert report.removed["ramp_costs"] == ["g1"] and report.removed["ramp_limits"] == ["g1"], level


def test_online_from_the_database_solves(tmp_path_factory) -> None:
    """reduce_unit_detail_method = online authored on the solve: the run is an LP without online variables, the
    user constraint and the exact binary; g1 serves elec and, through c1, elec2."""
    step = _solve(tmp_path_factory, "rd_db_online", _full_values("online"), _full_entities())
    rows, ints = _families(build(step.flex_data))
    assert ints == set()
    assert not any(r.startswith(("maxOnline", "online_change", "maxFlow_exact", "process_constraint")) for r in rows)
    out = utf8(step.solution.value("v_flow")).filter(pl.col("p") == "g1")
    assert out["value"].sum() > 0


# ---------------------------------------------------------------------------
# min_load_efficiency at full-load efficiency (online)


def _fold_values(demand: float, k_sink: float, level: str | None, k_src: float = 1.0) -> list:
    vals = _base_values(demand)
    if k_sink != 1.0:
        vals.append(["unit__outputNode", ["g1", "elec"], "conversion_flow_coeff", _f(k_sink)])
    if k_src != 1.0:
        vals.append(["unit__inputNode", ["g1", "fuel"], "conversion_flow_coeff", _f(k_src)])
    if level is not None:
        vals.append(["solve", SOLVE, "reduce_unit_detail_method", _s(level)])
    return vals


@pytest.mark.parametrize(
    "k_sink, k_src, fuel_none_100, fuel_none_50, fuel_online_100, fuel_online_50, slope_online", [
        (1.0, 1.0, 200.0, 125.0, 200.0, 100.0, 2.0),
        (2.0, 1.0, 350.0, 200.0, 350.0, 175.0, 3.5),
        (1.0, 2.0, 100.0, 62.5, 100.0, 50.0, 1.0),
    ])
def test_min_load_efficiency_folds_to_full_load(tmp_path_factory, k_sink, k_src, fuel_none_100, fuel_none_50,
                                                fuel_online_100, fuel_online_50, slope_online) -> None:
    """Under online the unit's fuel is output × (slope + section): the full-load fuel is kept, part load uses
    less (hand values in the module docstring).  The objective is fuel × 20 CUR/MWh × the same weights in every
    case, so its ratios are the fuel ratios."""
    tag = f"k{int(k_sink)}s{int(k_src)}"

    def run(level, demand, name):
        return _solve(tmp_path_factory, f"rd_fold_{name}_{tag}", _fold_values(demand, k_sink, level, k_src),
                      _base_entities())

    none_100, none_50 = run(None, 100.0, "none_100"), run(None, 50.0, "none_50")
    on_100, on_50 = run("online", 100.0, "on_100"), run("online", 50.0, "on_50")
    slope = utf8(on_50.flex_data.p_slope.frame).filter(pl.col("p") == "g1")["value"]
    section = utf8(on_50.flex_data.p_section.frame).filter(pl.col("p") == "g1")["value"]
    assert slope.to_list() == pytest.approx([slope_online] * slope.len(), rel=1e-12)
    assert section.to_list() == [0.0] * section.len()
    assert on_50.flex_data.process_min_load_eff is None and on_50.flex_data.process_online is None
    obj = {k: float(s.obj) for k, s in (("n100", none_100), ("n50", none_50), ("o100", on_100), ("o50", on_50))}
    assert obj["o100"] / obj["n100"] == pytest.approx(fuel_online_100 / fuel_none_100, rel=1e-7)
    assert obj["n50"] / obj["n100"] == pytest.approx(fuel_none_50 / fuel_none_100, rel=1e-7)
    assert obj["o50"] / obj["n100"] == pytest.approx(fuel_online_50 / fuel_none_100, rel=1e-7)


# ---------------------------------------------------------------------------
# warm start


def test_warm_reuse_blocked_by_relax_integrality(tmp_path_factory) -> None:
    """Two sub-solves whose data differ only by integers on a binary unit (no exact connection): the warm LP is not
    reused, and the blocker names relax_integrality."""
    from flextool.engine.solve._warm import _reuse_blocker

    step = _solve(tmp_path_factory, "rd_fold_none_100_k1", _fold_values(100.0, 1.0, None), _base_entities())
    reduced, report = apply_reduce_detail(step.flex_data, "integers")
    assert report.removed == {"integer_online": ["g1"]}
    assert _reuse_blocker(step.flex_data, reduced) == "relax_integrality"
    assert _reuse_blocker(step.flex_data, apply_reduce_detail(step.flex_data, "none")[0]) is None


# ---------------------------------------------------------------------------
# rolling: the online / ramp carry check sees the reduced data


def test_rolling_online_with_ramp_limit_runs_two_rolls(tmp_path_factory) -> None:
    """Rolling fullYear dispatch cut to 48 h (horizon 36, jump 24: two rolls), coal_plant with linear startup and a
    ramp limit, reduce_unit_detail_method online.  Roll 1 hands over its roll-end flow (ramps are kept) but no
    online carry (no online variables).  The carry check runs on the reduced data (no online process), so roll 2
    builds and ramps from roll 1's flow."""
    from tests.engine._unit_db import PLANT, fullyear_roll_first_hours

    out = "unit__outputNode"
    rel = [PLANT, "west"]
    solve = "dispatch_fullYear_roll"
    vals = fullyear_roll_first_hours(48.0) + [
        ["solve", solve, "rolling_solve_horizon", _f(36.0)], ["solve", solve, "rolling_solve_jump", _f(24.0)],
        ["solve", solve, "reduce_unit_detail_method", _s("online")],
        ["unit", PLANT, "startup_method", _s("linear")],
        [out, rel, "ramp_method", _s("ramp_limit")], [out, rel, "ramp_speed_up", _f(0.005)],
        [out, rel, "ramp_speed_down", _f(0.005)]]
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt="rd_roll_online", values=vals,
                                  base_chain=["init", "west", "coal", "fullYear", solve])
    rolls = list(steps.values())
    assert len(rolls) == 2, list(steps)
    second = rolls[1]
    assert second.solution is not None
    fd = second.flex_data
    assert fd.process_online is None and fd.p_roll_continue_online is None
    assert fd.dt_ramp_carry is not None and fd.dt_ramp_carry.height > 0
    assert fd.p_roll_continue_flow is not None
    assert {str(p) for p in utf8(fd.p_roll_continue_flow.frame)["p"].to_list()} == {PLANT}
    rows, ints = _families(build(fd))
    assert {"ramp_sink_up_constraint", "ramp_sink_down_constraint"} <= rows
    assert not any(r.startswith(("maxOnline", "online_change")) for r in rows) and ints == set()
    # The same check on the authored online set (where load_flextool used to run it) raises: the move is needed.
    import copy

    from flextool.engine.preprocess.params._ramp_links import check_online_ramp_carry

    authored = copy.copy(fd)
    authored.process_online = pl.DataFrame({"p": [PLANT]})
    with pytest.raises(ValueError, match="no online carry"):
        check_online_ramp_carry(authored)
