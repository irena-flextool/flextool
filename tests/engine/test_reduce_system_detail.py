"""Solve-level system-detail reduction (``solve.reduce_system_detail_method``, database version 81).

A level clears the FlexData of the system features it names (``preprocess._reduce_detail``): reserves, inertia,
non-synchronous share limits, DC power flow.  The capacity margin is never reduced.

Hand-built models on ``tests/fixtures/tests.json`` (alternative ``init`` only; never a checked-in ``.sqlite``),
timeset ``2day`` cut to two 1-h steps of period p2020:

* ``sys_full``: node ``elec`` (demand 80 MW) in group ``electricity``; commodity node ``fuel`` at 20 CUR/MWh;
  unit ``g1`` fuel -> elec (100 MW, efficiency 0.5, linear startup, min_load 0.5, inertia_constant 2) holding the
  upward and downward reserve ``rsv`` of ``electricity``; unit ``w1`` -> elec (50 MW, non-synchronous);
  ``electricity`` with an inertia limit and a non-synchronous limit; group ``cm`` (elec) with a manual capacity
  margin in an investment solve.
* ``dc_triangle``: nodes n1, n2, n3 in group ``ac`` (``dc_power_flow_with_angles``), lines c12, c23, c13 of
  100 MW and equal reactance; supply only at n1 (unit g, 300 MW), demand 180 MW at n3 (penalty 10000 CUR/MWh).
* ``reserve_only``: ``g1`` 100 MW at elec, demand 80 MW, upward reserve 30 MW at 1000 CUR/MW/h.

Hand values (design solve_relaxation_design.md, brief WP2 tests 2-3, independent of the code):

* DC: with equal reactances the angle model splits a transfer T from n1 to n3 as 2/3 on the direct line c13 and
  1/3 on the path c12-c23.  c13 binds at 100 MW, so T ≤ 150: 150 MW reach n3, shortfall 30 MW.  Under
  ``dc_power_flow`` the lines are lossless transport limited by capacity: 100 MW direct + 80 MW via n2 = 180 MW,
  shortfall 0.  No losses in either case: generation = delivered = 180 − shortfall.
* Reserves: g1 can hold 100 − 80 = 20 MW of the 30 MW requirement: shortfall 10 MW.  Fuel 80 / 0.5 = 160 MWh/h at
  20 CUR/MWh = 3200 CUR/h; shortfall penalty 10 · 1000 = 10000 CUR/h.  The objective ratio with / without reserves
  is (3200 + 10000) / 3200 = 4.125 (both terms carry the same step and period weights).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.preprocess._reduce_detail import (
    DC_FIELDS, INERTIA_FIELDS, NON_SYNC_FIELDS, RESERVE_FIELDS, SYSTEM_FIELDS, apply_reduce_detail,
)
from tests.engine._unit_db import RunCache, b64, build, unitsize, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
_RUNS = RunCache()
UD_UNIT = "reserve__upDown__unit__node"
UD_GROUP = "reserve__upDown__group"


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


def _timeset() -> list:
    return [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")]]


def _fuel_unit(unit: str, node: str, existing: float) -> tuple[list, list]:
    ents = [["unit", unit, None], ["unit__inputNode", [unit, "fuel"], None], ["unit__outputNode", [unit, node], None]]
    vals = [["unit", unit, "existing", _f(existing)], ["unit", unit, "efficiency", _f(0.5)]]
    return ents, vals


def _fuel() -> tuple[list, list]:
    return ([["node", "fuel", None], ["commodity", "fuelc", None], ["commodity__node", ["fuelc", "fuel"], None]],
            [["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)]])


def _reserve(r: str, ud: str, unit: str, node: str, reservation: float) -> tuple[list, list]:
    grp, key = [r, ud, "electricity"], [r, ud, unit, node]
    ents = [[UD_GROUP, grp, None], [UD_UNIT, key, None]]
    vals = [[UD_GROUP, grp, "reserve_method", _s("timeseries_only")], [UD_GROUP, grp, "reservation", _f(reservation)],
            [UD_GROUP, grp, "penalty_reserve", _f(1000.0)],
            [UD_UNIT, key, "is_enabled", _s("yes")], [UD_UNIT, key, "max_share", _f(1.0)],
            [UD_UNIT, key, "reliability", _f(1.0)]]
    return ents, vals


def _sys_full(system: str | None = None) -> tuple[list, list]:
    fe, fv = _fuel()
    ue, uv = _fuel_unit("g1", "elec", 100.0)
    up_e, up_v = _reserve("rsv", "up", "g1", "elec", 10.0)
    dn_e, dn_v = _reserve("rsv", "down", "g1", "elec", 5.0)
    out = "unit__outputNode"
    ents = fe + ue + [["node", "elec", None], ["group__node", ["electricity", "elec"], None],
                      ["unit", "w1", None], [out, ["w1", "elec"], None],
                      ["group", "cm", None], ["group__node", ["cm", "elec"], None],
                      ["reserve", "rsv", None]] + up_e + [dn_e[0]] + [dn_e[1]]
    vals = _timeset() + fv + uv + up_v + dn_v + [
        ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-80.0)],
        ["unit", "g1", "startup_method", _s("linear")], ["unit", "g1", "min_load", _f(0.5)],
        [out, ["g1", "elec"], "inertia_constant", _f(2.0)],
        ["unit", "w1", "existing", _f(50.0)], ["unit", "w1", "conversion_method", _s("none")],
        [out, ["w1", "elec"], "is_non_synchronous", _s("yes")],
        ["group", "electricity", "has_inertia", _s("yes")], ["group", "electricity", "inertia_limit", _f(100.0)],
        ["group", "electricity", "penalty_inertia", _f(10.0)],
        ["group", "electricity", "has_non_synchronous", _s("yes")],
        ["group", "electricity", "non_synchronous_limit", _f(0.5)],
        ["group", "electricity", "penalty_non_synchronous", _f(10.0)],
        ["group", "cm", "capacity_margin_method", _s("manual")], ["group", "cm", "capacity_margin", _f(10.0)],
        ["group", "cm", "penalty_capacity_margin", _f(1000.0)],
        ["solve", SOLVE, "invest_periods", b64({"value_type": "str", "data": ["p2020"], "index_name": "period"},
                                               "array")]]
    if system is not None:
        vals.append(["solve", SOLVE, "reduce_system_detail_method", _s(system)])
    return ents, vals


def _solve(tmp_path_factory, alt: str, entities: list, values: list):
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=["init"], entities=entities)
    assert len(steps) == 1, list(steps)
    step = next(iter(steps.values()))
    assert step.solution is not None, "solve produced no solution"
    return step


def _families(pb) -> tuple[set[str], set[str]]:
    m = pb.canonicalise()
    return {n.split("[", 1)[0] for n in m.row_names}, {n.split("[", 1)[0] for n in m.col_names}


def _dc_triangle(system: str | None = None) -> tuple[list, list]:
    fe, fv = _fuel()
    ue, uv = _fuel_unit("g", "n1", 300.0)
    ents = fe + ue + [["node", n, None] for n in ("n1", "n2", "n3")] + [["group", "ac", None]]
    ents += [["group__node", ["ac", n], None] for n in ("n1", "n2", "n3")]
    vals = _timeset() + fv + uv + [["group", "ac", "transfer_method", _s("dc_power_flow_with_angles")],
                                   ["node", "n3", "inflow", _f(-180.0)], ["node", "n3", "penalty_up", _f(10000.0)]]
    vals += [["node", n, "node_type", _s("balance")] for n in ("n1", "n2", "n3")]
    for c, a, z in (("c12", "n1", "n2"), ("c23", "n2", "n3"), ("c13", "n1", "n3")):
        ents += [["connection", c, None], ["connection__node__node", [c, a, z], None]]
        vals += [["connection", c, "existing", _f(100.0)], ["connection", c, "reactance", _f(0.1)]]
    if system is not None:
        vals.append(["solve", SOLVE, "reduce_system_detail_method", _s(system)])
    return ents, vals


# ---------------------------------------------------------------------------
# per level: the row and column families


# The families each system part removes from sys_full (design §2.B rows S1-S3).  The units' reserve coupling
# (maxFlow, the min-load floor) keeps its rows and loses only the v_reserve terms.
_RESERVES = ({"reserveBalance_timeseries_eq", "reserve_process_upward", "reserve_process_downward"},
             {"v_reserve", "vq_reserve"})
_INERTIA = ({"inertia_constraint"}, {"vq_inertia"})
_NON_SYNC = ({"non_sync_constraint"}, {"vq_non_synchronous"})


def _removed(*parts) -> tuple[set[str], set[str]]:
    return set().union(*(p[0] for p in parts)), set().union(*(p[1] for p in parts))


@pytest.mark.parametrize("level, removed, keys", [
    ("none", _removed(), set()),
    ("reserves", _removed(_RESERVES), {"reserves"}),
    ("inertia", _removed(_INERTIA), {"inertia"}),
    ("non_synchronous", _removed(_NON_SYNC), {"non_synchronous"}),
    ("inertia_non_synchronous", _removed(_INERTIA, _NON_SYNC), {"inertia", "non_synchronous"}),
    ("reserves_inertia_non_synchronous", _removed(_RESERVES, _INERTIA, _NON_SYNC),
     {"reserves", "inertia", "non_synchronous"}),
    # sys_full has no DC power flow group: dc_power_flow finds nothing to reduce there
    ("dc_power_flow", _removed(), set()),
    ("reserves_inertia_non_synchronous_dc_power_flow", _removed(_RESERVES, _INERTIA, _NON_SYNC),
     {"reserves", "inertia", "non_synchronous"}),
])
def test_row_families_per_system_level(tmp_path_factory, level, removed, keys) -> None:
    """Each level removes exactly its families, adds none, and keeps the capacity margin."""
    ents, vals = _sys_full()
    step = _solve(tmp_path_factory, "rs_full", ents, vals)
    none_rows, none_cols = _families(build(step.flex_data))
    assert {"capacityMargin", "reserveBalance_timeseries_eq", "inertia_constraint", "non_sync_constraint"} <= none_rows
    reduced, report = apply_reduce_detail(step.flex_data, "none", level)
    rows, cols = _families(build(reduced))
    assert none_rows - rows == removed[0], level
    assert none_cols - cols == removed[1], level
    assert rows - none_rows == set() and cols - none_cols == set(), level
    assert "capacityMargin" in rows and "vq_capacity_margin" in cols
    assert set(report.removed) == keys
    if "reserves" in keys:
        assert report.removed["reserves"] == ["rsv__down__electricity", "rsv__up__electricity"]
    for key in keys - {"reserves"}:
        assert report.removed[key] == ["electricity"]
    if level == "dc_power_flow":
        assert reduced is step.flex_data


def test_system_level_from_the_database(tmp_path_factory) -> None:
    """reduce_system_detail_method authored on the solve reaches the solved data."""
    ents, vals = _sys_full("reserves_inertia_non_synchronous")
    step = _solve(tmp_path_factory, "rs_full_db", ents, vals)
    fd = step.flex_data
    for name in RESERVE_FIELDS + INERTIA_FIELDS + NON_SYNC_FIELDS:
        assert getattr(fd, name) is None, name
    rows, cols = _families(build(fd))
    assert not any(r.startswith(("reserve", "inertia", "non_sync")) for r in rows)
    assert "capacityMargin" in rows


def test_field_lists_cover_the_system_fields(tmp_path_factory) -> None:
    """Every FlexData field of the system features is cleared by its part (none is kept): a new field of these
    features must be added to its list."""
    import dataclasses
    import re

    from flextool.engine.preprocess.flex_data import FlexData

    patterns = {"reserves": r"reserve|prundt", "inertia": r"inertia", "non_synchronous": r"nonsync|non_sync",
                "dc_power_flow": r"dc_|_dc|angle|suscept|reactance"}
    names = {f.name for f in dataclasses.fields(FlexData)}
    for part, pattern in patterns.items():
        found = {n for n in names if re.search(pattern, n, re.IGNORECASE)}
        assert found == set(SYSTEM_FIELDS[part]), (part, found ^ set(SYSTEM_FIELDS[part]))
    ents, vals = _sys_full()
    step = _solve(tmp_path_factory, "rs_full", ents, vals)
    reduced, _ = apply_reduce_detail(step.flex_data, "none", "reserves_inertia_non_synchronous_dc_power_flow")
    for name in RESERVE_FIELDS + INERTIA_FIELDS + NON_SYNC_FIELDS + DC_FIELDS:
        assert getattr(reduced, name) is None, name


def test_interaction_notes(tmp_path_factory) -> None:
    """online with reserves and inertia kept: g1's reserve comes from built capacity, the inertia of electricity
    is counted from flows, and g1's downward reserve reaches down to zero flow (its min load is left out).  Each
    note goes when the system detail goes; integers_min_limits keeps the online variables (only the min-load
    note)."""
    ents, vals = _sys_full()
    fd = _solve(tmp_path_factory, "rs_full", ents, vals).flex_data
    _, rep = apply_reduce_detail(fd, "online", "none")
    assert rep.notes == {"reserves_from_capacity": ["g1"], "inertia_from_flows": ["electricity"],
                         "downward_reserve_to_zero": ["g1"]}
    _, rep = apply_reduce_detail(fd, "online", "reserves")
    assert rep.notes == {"inertia_from_flows": ["electricity"]}
    _, rep = apply_reduce_detail(fd, "online", "reserves_inertia_non_synchronous")
    assert rep.notes == {}
    _, rep = apply_reduce_detail(fd, "integers_min_limits", "none")
    assert rep.notes == {"downward_reserve_to_zero": ["g1"]}
    _, rep = apply_reduce_detail(fd, "integers", "none")
    assert rep.notes == {}


# ---------------------------------------------------------------------------
# DC power flow -> lossless transport


def _net_flow(step, conn: str) -> list[float]:
    """Signed MW of a two-way connection (v_flow − v_flow_back) × unitsize, by step."""
    size = unitsize(step, conn)
    fwd = utf8(step.solution.value("v_flow")).filter(pl.col("p") == conn).sort("d", "t")["value"]
    back = utf8(step.solution.value("v_flow_back")).filter(pl.col("p") == conn).sort("d", "t")
    back = back["value"] if back.height else pl.Series([0.0] * fwd.len())
    return [(f - b) * size for f, b in zip(fwd.to_list(), back.to_list())]


def _gen(step, unit: str, sink: str) -> list[float]:
    rows = utf8(step.solution.value("v_flow")).filter((pl.col("p") == unit) & (pl.col("sink") == sink))
    return (rows.sort("d", "t")["value"] * unitsize(step, unit)).to_list()


def _shortfall(step, node: str) -> list[float]:
    rows = utf8(step.solution.value("vq_state_up")).filter(pl.col("n") == node).sort("d", "t")
    return rows["value"].to_list() if rows.height else [0.0, 0.0]


@pytest.mark.parametrize("level, delivered, short", [(None, 150.0, 30.0), ("dc_power_flow", 180.0, 0.0)])
def test_dc_lines_become_lossless_transport(tmp_path_factory, level, delivered, short) -> None:
    """Angles: the direct line binds at 100 MW and carries 2/3 of the transfer, so 150 MW arrive (shortfall 30).
    dc_power_flow: no angle rows, the lines keep their 100 MW capacity: 180 MW arrive (shortfall 0).  Lossless
    in both: generation = delivered."""
    ents, vals = _dc_triangle(level)
    step = _solve(tmp_path_factory, f"rs_dc_{level}", ents, vals)
    rows, cols = _families(build(step.flex_data))
    if level is None:
        assert {"dc_flow_eq", "dc_reference_angle_eq"} <= rows and "v_angle" in cols
    else:
        assert not ({"dc_flow_eq", "dc_reference_angle_eq"} & rows) and "v_angle" not in cols
        assert {"maxFlow", "maxFlow_back", "nodeBalance_eq"} <= rows
        for name in DC_FIELDS:
            assert getattr(step.flex_data, name) is None, name
    c12, c23, c13 = (_net_flow(step, c) for c in ("c12", "c23", "c13"))
    gen = _gen(step, "g", "n1")
    for k in range(2):
        assert c13[k] + c23[k] == pytest.approx(delivered, abs=1e-6)
        assert gen[k] == pytest.approx(delivered, abs=1e-6)
        assert c12[k] == pytest.approx(c23[k], abs=1e-6)
        assert max(abs(c12[k]), abs(c23[k]), abs(c13[k])) <= 100.0 + 1e-6
        assert _shortfall(step, "n3")[k] == pytest.approx(short, abs=1e-6)
    if level is None:
        assert c13 == pytest.approx([100.0, 100.0], abs=1e-6)


# ---------------------------------------------------------------------------
# reserves


def _reserve_only(system: str | None) -> tuple[list, list]:
    fe, fv = _fuel()
    ue, uv = _fuel_unit("g1", "elec", 100.0)
    re_, rv = _reserve("rsv", "up", "g1", "elec", 30.0)
    ents = fe + ue + [["node", "elec", None], ["group__node", ["electricity", "elec"], None],
                      ["reserve", "rsv", None]] + re_
    vals = _timeset() + fv + uv + rv + [["node", "elec", "node_type", _s("balance")],
                                        ["node", "elec", "inflow", _f(-80.0)]]
    if system is not None:
        vals.append(["solve", SOLVE, "reduce_system_detail_method", _s(system)])
    return ents, vals


def test_reserves_left_out_drops_the_shortfall_penalty(tmp_path_factory) -> None:
    """With reserves g1 holds 20 of the 30 MW (shortfall 10 MW); under reserves no reserve rows or columns and the
    objective is lower by exactly the shortfall penalty: ratio (3200 + 10000) / 3200 (module docstring)."""
    full = _solve(tmp_path_factory, "rs_res_none", *_reserve_only(None))
    red = _solve(tmp_path_factory, "rs_res_red", *_reserve_only("reserves"))
    held = utf8(full.solution.value("v_reserve")).filter(pl.col("p") == "g1").sort("d", "t")
    assert (held["value"] * unitsize(full, "g1")).to_list() == pytest.approx([20.0, 20.0], abs=1e-6)
    rows, cols = _families(build(red.flex_data))
    assert not any(r.startswith("reserve") for r in rows) and not ({"v_reserve", "vq_reserve"} & cols)
    assert _gen(red, "g1", "elec") == pytest.approx([80.0, 80.0], abs=1e-6)
    assert float(full.solution.obj) / float(red.solution.obj) == pytest.approx(13200.0 / 3200.0, rel=1e-7)
