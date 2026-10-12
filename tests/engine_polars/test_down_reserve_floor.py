"""A unit cannot offer downward reserve that takes it below its floor.

Reserve that REDUCES a unit arc — ``down`` at the node the unit delivers
into, ``up`` at the node it consumes from (a consumption cut) — needs
headroom below the current flow:

* ``minFlow_minload_<kind>``: ``Σ outputs − Rdn ≥ v_online · min_load``;
* ``minFlow_output_floor_<kind>``: ``out_k − Rdn_k ≥ v_online · min_load ·
  capacity_min_coeff_k``;
* ``minFlow_reserve``: ``Σ outputs − Rdn ≥ 0`` for other units;
* ``minFlow_reserve_arc``: ``v_flow − Rdn ≥ 0`` per arc of a multi-output
  unit or outside the output measure (an indirect unit's input).

A source-side reserve on a direct input→output unit is converted to output
flow by ``1 / slope`` (the ``reserveBalance`` V1 convention counts reserve
as MW at the reserve node).  Connections are excluded.

Fixtures: ``tests.json`` + override alternatives (built from JSON) adding
a reserve ``rx`` on group ``electricity`` (nodes west / east / north) or
``fuel`` (coal_market).  The requirement is set far above what the unit
can offer so the LP offers as much as the rules allow; the rest is
shortfall (``vq_reserve``).  The shortfall penalty is low (1 €/MWh) so the
LP never generates extra energy just to be able to offer reserve.
"""
from __future__ import annotations

import pytest

from tests.engine_polars._unit_db import (
    CHP, CHP_CHAIN, COAL, PLANT, PLANT_CHAIN, b64, build, cstr_over, demand,
    flow, no_ratio_constraint, online, reserve, solve, time_map, uc, utf8,
)

TOL = 1e-5


def _reserve(unit: str, node: str, ud: str, *, group: str = "electricity",
             reservation: float = 5000.0, cls: str = "unit",
             penalty: float = 1.0) -> tuple:
    """Entities + values of a timeseries reserve ``rx`` (``ud``) on
    ``group`` offered by ``unit`` at ``node``."""
    rel = f"reserve__upDown__{cls}__node"
    entities = [
        ["reserve", "rx", None],
        ["reserve__upDown__group", ["rx", ud, group], None],
        [rel, ["rx", ud, unit, node], None],
    ]
    values = [
        ["reserve__upDown__group", ["rx", ud, group], "reserve_method",
         b64("timeseries_only", "str")],
        ["reserve__upDown__group", ["rx", ud, group], "reservation",
         b64(reservation, "float")],
        ["reserve__upDown__group", ["rx", ud, group], "penalty_reserve",
         b64(penalty, "float")],
        [rel, ["rx", ud, unit, node], "is_enabled", b64("yes", "str")],
        [rel, ["rx", ud, unit, node], "max_share", b64(1.0, "float")],
        [rel, ["rx", ud, unit, node], "reliability", b64(1.0, "float")],
    ]
    return entities, values


def _plant(tpf, alt: str, values: list, ud: str = "down",
           node: str = "west", group: str = "electricity"):
    ents, vals = _reserve(PLANT, node, ud, group=group)
    return solve(tpf, alt=alt, base_chain=PLANT_CHAIN, entities=ents,
                 values=vals + values)


_WEST_DIP = [["node", "west", "inflow",
              time_map({"t0005": 0.0, "t0006": 0.0}, -400.0)]]


def test_direct_unit_down_reserve_respects_min_load(tmp_path_factory) -> None:
    """Online direct unit (linear UC, min_load 0.5): out − down ≥ 0.5 ·
    online at every step and binds where the unit runs; the unmet
    requirement is shortfall, not reserve below min load.  When the unit
    is off (zero-demand steps) it offers no downward reserve."""
    step = _plant(tmp_path_factory, "dr_minload", uc(PLANT, 0.5) + _WEST_DIP)
    out = flow(step, PLANT, COAL, "west")
    down = reserve(step, PLANT, "rx", "west")
    on = online(step, PLANT)
    slack = out - down - 0.5 * on
    assert (slack >= -TOL).all()
    running = out > 1.0
    assert slack.filter(running).abs().max() == pytest.approx(0.0, abs=TOL)
    assert down.filter(~running).abs().max() == pytest.approx(0.0, abs=TOL)
    assert step.solution.value("vq_reserve")["value"].max() > 0.0
    pb = build(step.flex_data)
    assert cstr_over(pb, "minFlow_minload_linear") is not None
    assert cstr_over(pb, "minFlow_reserve") is None


def test_direct_unit_without_online_down_reserve_le_flow(
        tmp_path_factory) -> None:
    """No startup method: ``down ≤ out`` (``minFlow_reserve``) — a unit at
    0 output offers no downward reserve."""
    step = _plant(tmp_path_factory, "dr_noonline", _WEST_DIP)
    out = flow(step, PLANT, COAL, "west")
    down = reserve(step, PLANT, "rx", "west")
    assert ((out - down) >= -TOL).all()
    assert (out - down).abs().max() == pytest.approx(0.0, abs=TOL)
    assert down[4] == pytest.approx(0.0, abs=TOL)
    over = cstr_over(build(step.flex_data), "minFlow_reserve")
    assert over is not None
    assert set(over["p"].unique().to_list()) == {PLANT}


def test_direct_unit_input_side_up_reserve(tmp_path_factory) -> None:
    """Tripwire for the ``1 / slope`` conversion: ``up`` reserve at the
    direct unit's INPUT node (coal_market, group ``fuel``) is a fuel-draw
    cut in node MW; the output may only drop to min load, so
    ``up ≤ (out − 0.5 · online) · slope`` (slope = 1 / 0.4).  If the
    ``reserveBalance`` source-side slope factor (V1 note in _reserve.py) is
    ever added, this conversion must change with it."""
    step = _plant(tmp_path_factory, "dr_srcup", uc(PLANT, 0.5) + _WEST_DIP,
                  ud="up", node=COAL, group="fuel")
    out = flow(step, PLANT, COAL, "west")
    up = reserve(step, PLANT, "rx", COAL)
    on = online(step, PLANT)
    head = (out - 0.5 * on) * 2.5 - up
    assert (head >= -1e-4).all()
    running = out > 1.0
    assert head.filter(running).abs().max() == pytest.approx(0.0, abs=1e-4)
    assert up.filter(running).max() > 1.0


def _chp_reserve(tpf, alt: str, values: list):
    ents, vals = _reserve(CHP, "west", "down")
    return solve(tpf, alt=alt, base_chain=CHP_CHAIN, entities=ents,
                 values=no_ratio_constraint() + vals + values)


def test_multi_output_unit_down_reserve_per_arc(tmp_path_factory) -> None:
    """Two-output CHP with ``down`` on west only (heat high, west low): the
    per-arc row keeps down ≤ west although west + heat is large."""
    step = _chp_reserve(tmp_path_factory, "dr_chp_arc",
                        [demand("west", 100.0), demand("heat", 800.0)])
    west = flow(step, CHP, CHP, "west")
    down = reserve(step, CHP, "rx", "west")
    assert ((west - down) >= -TOL).all()
    assert down.max() == pytest.approx(west.max(), rel=1e-6)
    assert down.max() < 150.0
    over = cstr_over(build(step.flex_data), "minFlow_reserve_arc")
    assert over is not None
    assert set(over.select("p", "source", "sink").unique().iter_rows()) == {
        (CHP, CHP, "west")}


def test_multi_output_unit_reserve_with_floors(tmp_path_factory) -> None:
    """Online CHP (min_load 0.4) with a per-output floor on west
    (``capacity_min_coeff`` 0.5): west − down ≥ 0.2 · online and
    west + heat − down ≥ 0.4 · online."""
    step = _chp_reserve(tmp_path_factory, "dr_chp_floor",
                        [demand("west", 300.0), demand("heat", 500.0),
                         ["unit__outputNode", [CHP, "west"],
                          "capacity_min_coeff", b64(0.5, "float")]]
                        + uc(CHP, 0.4))
    west, heat = flow(step, CHP, CHP, "west"), flow(step, CHP, CHP, "heat")
    down = reserve(step, CHP, "rx", "west")
    on = online(step, CHP)
    assert ((west - down - 0.2 * on) >= -TOL).all()
    assert ((west + heat - down - 0.4 * on) >= -TOL).all()
    assert down.max() > 1.0
    pb = build(step.flex_data)
    assert cstr_over(pb, "minFlow_output_floor_linear") is not None
    # the per-output floor replaces the zero per-arc row on west
    assert cstr_over(pb, "minFlow_reserve_arc") is None


def test_consumer_up_reserve_le_consumption(tmp_path_factory) -> None:
    """Sink-less consumer (``dr_increase_demand``, input west) offering
    ``up`` reserve at west (a consumption cut): up ≤ consumption, and
    ≤ consumption − min_load · online with UC.  A high shortfall penalty
    makes consuming (to be able to cut it) worthwhile."""
    ents, vals = _reserve("dr_increase_demand", "west", "up",
                          penalty=20000.0)
    for alt, extra in (("dr_cons", []),
                       ("dr_cons_uc", uc("dr_increase_demand", 0.5))):
        step = solve(tmp_path_factory, alt=alt,
                     base_chain=PLANT_CHAIN + ["dr_increase_demand"],
                     entities=ents, values=vals + extra)
        cons = flow(step, "dr_increase_demand", "west", "dr_increase_demand")
        up = reserve(step, "dr_increase_demand", "rx", "west")
        floor = (0.5 * online(step, "dr_increase_demand") if extra
                 else 0.0 * cons)
        assert ((cons - up - floor) >= -TOL).all()
        assert up.max() > 1.0


def test_connection_down_reserve_adds_no_rows(tmp_path_factory) -> None:
    """Connections are excluded from the reserve floors (a connection
    offers downward reserve both by cutting import and raising export)."""
    ents, vals = _reserve("west_east", "east", "down", cls="connection")
    step = solve(tmp_path_factory, alt="dr_conn",
                 base_chain=PLANT_CHAIN + ["network"], entities=ents,
                 values=vals)
    pb = build(step.flex_data)
    for name in ("minFlow_reserve", "minFlow_reserve_arc"):
        over = cstr_over(pb, name)
        assert over is None or "west_east" not in over["p"].to_list()
    assert utf8(step.solution.value("v_reserve")).height > 0
