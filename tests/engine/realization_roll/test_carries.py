"""What a realization roll carries to the next roll (design rev 7.1 §8.3 T-c1 / T-c2 / T-c3).

Model (``_spec``): one period ``p1`` on t01-t03 (1 h steps, ``years_represented`` 1: factor 8760 / 3 = 2920), a
rolling ``within_solve`` solve with jump 1 and horizon 2: R_0 = ``p1`` t01 + tails ``p1_A`` / ``p1_B`` t02 (forecasts made
at t02), R_1 = ``p1`` t02 + tails t03 (made at t03), R_2 = ``p1`` t03 (deterministic on ``R``).  Node ``elec`` (in the
stochastic group): demand 1 MW on ``R`` (made at t01), 1 on ``A`` and 0 on ``B`` (weights 0.5 / 0.5, made at t02 and
t03).  ``coal_u`` (10 MW, efficiency 1, 10 EUR/MWh) and ``gas`` (10 MW, 30 EUR/MWh).

The period budget B below is 5840 per year = 5840 x 3 / 8760 = 2 over the period.  A path row of roll k reaches
f = (realized h + jump h + tail h) / 3 of the period: R_0 2/3 (cap 4/3), R_1 3/3 (cap 2 - the carry).  With x_k the
jump's coal and a_k tail A's coal, the cost before the factor is 10 x_k + 30 (1 - x_k) + 0.5 (10 a_k + 30 (1 - a_k)):
coal is worth 20 in the jump and 10 (0.5 x 20) in tail A, so the jump takes coal first:
* R_0: x_0 + a_0 <= 4/3 -> x_0 = 1, a_0 = 1/3: 45 - 20 - 10/3 = 21.667 -> 63 266.67; carried 1.
* R_1: x_1 + a_1 <= 2 - 1 -> x_1 = 1, a_1 = 0: 25 -> 73 000; carried 1 + 1 = 2.
* R_2: 2 - 2 = 0 left, gas: 30 -> 87 600.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as sr
from tests.engine.realization_roll import test_trunk_sites as ts

pack, pmap = scm.pack, scm.pmap
pytestmark = pytest.mark.solver
STEPS = ("t01", "t02", "t03")
FACTOR = 8760 / 3
BUDGET_OBJECTIVES = [21.0 + 2.0 / 3.0, 25.0, 30.0]


def _spec(*, entities=(), active=(), values=(), coal_cost: float = 10.0) -> dict:
    demand = sr._series({"R": {"t01": {t: -1.0 for t in STEPS}},
                         "A": {"t02": {"t02": -1.0, "t03": -1.0}, "t03": {"t03": -1.0}},
                         "B": {"t02": {"t02": 0.0, "t03": 0.0}, "t03": {"t03": 0.0}}})
    fan = sr._fan([("A", {"t02": 0.5, "t03": 0.5}, "no"), ("B", {"t02": 0.5, "t03": 0.5}, "no"),
                   ("R", {"t01": 0.0}, "yes")])
    ents = [["node", "elec", None], ["unit", "gas", None], ["unit", "coal_u", None],
            ["unit__outputNode", ["gas", "elec"], None], ["unit__outputNode", ["coal_u", "elec"], None],
            ["group__node", ["sg", "elec"], None], *entities]
    act = [("node", "elec"), ("unit", "gas"), ("unit", "coal_u"), *active]
    vals = [*sr._node("elec", "balance", demand), *sr._unit("gas", "elec", 10.0, 30.0),
            *sr._unit("coal_u", "elec", 10.0, coal_cost), *values]
    return sr._model(STEPS, jump=1, horizon=2, fan=fan, entities=ents, active=act, values=vals)


_COAL_FUEL = ([["node", "coal_n", None], ["commodity", "coal", None], ["commodity__node", ["coal", "coal_n"], None],
               ["unit__inputNode", ["coal_u", "coal_n"], None]],
              [("node", "coal_n")],
              [["node", "coal_n", "node_type", pack("commodity", "str"), "init"]])


def _jump_coal(step) -> float:
    """MWh of coal burnt in a roll's jump: coal_u's input flow on its ``p1`` cells x unit size 10 MW x 1 h."""
    frame = step.solution.value("v_flow").with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "d"))
    rows = frame.filter((pl.col("p") == "coal_u") & (pl.col("source") == "coal_n") & (pl.col("d") == "p1"))
    assert rows.height == 1
    return float(rows["value"].sum()) * 10.0


def test_co2_carry_is_the_jumps_emissions(tmp_path, monkeypatch):
    """T-c1: ``co2_max_period`` 5840 t/a on group ``co2g`` (coal 1 t/MWh, price 0): the module docstring's budget.
    The carry read by roll k+1 (``p_co2_cum_realized_tonnes``) = the tonnes of the jumps so far, from v_flow x
    content x step duration over the ``p1`` cells: 1 t after R_0, 2 t after R_1."""
    entities, active, values = _COAL_FUEL
    spec = _spec(entities=[*entities, ["group", "co2g", None], ["group__node", ["co2g", "coal_n"], None]],
                 active=active,
                 values=[*values, ["commodity", "coal", "co2_content", pack(1.0, "float"), "init"],
                         ["commodity", "coal", "price", pack(0.0, "float"), "init"],
                         ["group", "co2g", "co2_method", pack("period", "str"), "init"],
                         ["group", "co2g", "co2_max_period",
                          pack(pmap([("p1", 5840.0)], index_name="period"), "map"), "init"]])
    built, steps, datas = sr._run(spec, tmp_path, monkeypatch)
    assert len(built) == 3
    rolls = list(steps.values())
    assert [_jump_coal(s) for s in rolls] == pytest.approx([1.0, 1.0, 0.0], abs=1e-9)
    carried = [ts._values(d, "p_co2_cum_realized_tonnes", "g", "d") for d in datas[1:]]
    assert carried == [pytest.approx({("co2g", "p1"): 1.0}), pytest.approx({("co2g", "p1"): 2.0})]
    assert carried[0][("co2g", "p1")] == pytest.approx(_jump_coal(rolls[0]), rel=1e-9)
    assert carried[1][("co2g", "p1")] == pytest.approx(_jump_coal(rolls[0]) + _jump_coal(rolls[1]), rel=1e-9)
    assert sr._objectives(steps) == pytest.approx([o * FACTOR for o in BUDGET_OBJECTIVES], rel=1e-9)


def test_ladder_carry_is_the_jumps_purchases(tmp_path, monkeypatch):
    """T-c2: coal (no operating cost) on ``price_ladder_annual``: tier 1 at 10 EUR/MWh up to 5840 MWh/a, tier 2 at 30
    (= gas): the module docstring's budget with tier-1 purchases.  The carry read by roll k+1
    (``p_ladder_cum_realized_mwh`` of (coal, tier 1, p1)) = the jumps' tier-1 purchases: 1 MWh after R_0, 2 after
    R_1.  Objectives as T-c1 (tier 2 and gas both cost 30)."""
    entities, active, values = _COAL_FUEL
    tier = {"type": "map", "index_type": "str", "rank": 1}
    tiers = {"index_type": "str", "rank": 2, "index_name": "tier",
             "data": [["1", {**tier, "data": [["price", 10.0], ["quantity", 5840.0]]}],
                      ["2", {**tier, "data": [["price", 30.0], ["quantity", float("inf")]]}]]}
    spec = _spec(entities=entities, active=active, coal_cost=0.0,
                 values=[*values, ["commodity", "coal", "price_method", pack("price_ladder_annual", "str"), "init"],
                         ["commodity", "coal", "price_ladder_annual", pack(tiers, "map"), "init"]])
    built, steps, datas = sr._run(spec, tmp_path, monkeypatch)
    assert len(built) == 3
    carried = [ts._values(d, "p_ladder_cum_realized_mwh", "c", "i", "d") for d in datas[1:]]
    assert carried == [pytest.approx({("coal", "1", "p1"): 1.0}), pytest.approx({("coal", "1", "p1"): 2.0})]
    assert sr._objectives(steps) == pytest.approx([o * FACTOR for o in BUDGET_OBJECTIVES], rel=1e-9)


def test_flow_group_carry_is_the_jumps_flow(tmp_path, monkeypatch):
    """T-c3: flow group ``fg`` = coal_u's output, ``max_cumulative_flow`` 0.5 MW (a constant): 0.5 x 3 = 1.5 MWh over
    the period, carried across the rolls (reference/flow_groups.md).  A path row of roll k allows 0.5 x (realized h
    + jump h + tail h) - the jumps' coal so far:
    R_0: 0.5 x 2 = 1: x_0 + a_0 <= 1 -> x_0 = 1, a_0 = 0: 10 + 0.5 x 30 = 25 -> 73 000; carried 1.
    R_1: 0.5 x 3 - 1 = 0.5: x_1 = 0.5, a_1 = 0: 5 + 15 + 15 = 35 -> 102 200; carried 1.5.
    R_2 (t03): 0.5 x 3 - 1.5 = 0 left, gas: 30 -> 87 600."""
    spec = _spec(entities=[["flowGroup", "fg", None], ["flowGroup__unit__node", ["fg", "coal_u", "elec"], None]],
                 values=[["flowGroup", "fg", "max_cumulative_flow", pack(0.5, "float"), "init"]])
    built, steps, datas = sr._run(spec, tmp_path, monkeypatch)
    assert len(built) == 3
    assert sr._objectives(steps) == pytest.approx([25.0 * FACTOR, 35.0 * FACTOR, 30.0 * FACTOR], rel=1e-9)
    carried = [{str(g): v for g, v in d.flowgroup_realized.select(pl.col("g").cast(pl.Utf8), "flow").rows()}
               for d in datas[1:]]
    assert carried == [pytest.approx({"fg": 1.0}), pytest.approx({"fg": 1.5})]
