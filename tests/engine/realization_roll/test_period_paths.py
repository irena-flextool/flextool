"""Period-level rows of a split realization roll sum the path: the base period's jump and one tail (design rev 7.1
§2.3, §8.1 T-pc / T-cw weighted; MR-b3).

Each test forces the live plan through the seam and suppresses the spawned realization solve
(``test_trunk_sites._live_plan``, D6), as in ``test_split_roll_lp``.  The model (``_base``): one period ``p1`` on
t01-t04 (1 h steps, ``years_represented`` 1, factor 8760 / 4 = 2190), a rolling ``within_solve`` solve with jump 2,
horizon 4: R_0 = ``p1`` t01-t02 (the jump, on the realized branch ``R``) + ``p1_A`` / ``p1_B`` t03-t04 (forecasts made
at t03, weight 0.5 each); R_1 = ``p1`` t03-t04 (the last roll, deterministic on ``R``).  Node ``elec`` (in the
stochastic group) has demand 1 MW per step on ``R``, 1 on ``A`` and 0 on ``B``; ``coal_u`` 10 EUR/MWh and ``gas`` 30.

Every period-level row on a path (``co2_max_period``, ``maxCumulative_flow_period``, ``nodeBalancePeriod_eq``,
``process_constraint_less_than_period_total``) is on a tail only and sums the jump cells too.  Every expected
value is derived by hand in the test's docstring.
"""
from __future__ import annotations

import pytest

from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as sr

pack, pmap = scm.pack, scm.pmap
STEPS = ("t01", "t02", "t03", "t04")


def _base(*, entities=(), active=(), values=(), demand_b: float = 0.0, gas_cost=30.0, coal: bool = True,
          coal_cost: float = 10.0) -> dict:
    demand = sr._series({"R": {"t01": {t: -1.0 for t in STEPS}},
                         "A": {"t03": {"t03": -1.0, "t04": -1.0}},
                         "B": {"t03": {"t03": -demand_b, "t04": -demand_b}}})
    fan = sr._fan([("A", {"t03": 0.5}, "no"), ("B", {"t03": 0.5}, "no"), ("R", {"t01": 0.0}, "yes")])
    ents = [["node", "elec", None], ["unit", "gas", None], ["unit__outputNode", ["gas", "elec"], None],
            ["group__node", ["sg", "elec"], None], *entities]
    act = [("node", "elec"), ("unit", "gas"), *active]
    vals = [*sr._node("elec", "balance", demand), *sr._unit("gas", "elec", 10.0, gas_cost), *values]
    if coal:
        ents += [["unit", "coal_u", None], ["unit__outputNode", ["coal_u", "elec"], None]]
        act.append(("unit", "coal_u"))
        vals += sr._unit("coal_u", "elec", 10.0, coal_cost)
    return sr._model(STEPS, jump=2, horizon=4, fan=fan, entities=ents, active=act, values=vals)


def _co2() -> dict:
    """``coal_u`` burns commodity ``coal`` (1 t/MWh, price 0) from node ``coal_n`` in group ``co2g``:
    ``co2_max_period`` 4380 t/a (4380 x 4 / 8760 = 2 t over the period)."""
    entities = [["node", "coal_n", None], ["commodity", "coal", None], ["commodity__node", ["coal", "coal_n"], None],
                ["unit__inputNode", ["coal_u", "coal_n"], None], ["group", "co2g", None],
                ["group__node", ["co2g", "coal_n"], None]]
    values = [["node", "coal_n", "node_type", pack("commodity", "str"), "init"],
              ["commodity", "coal", "co2_content", pack(1.0, "float"), "init"],
              ["commodity", "coal", "price", pack(0.0, "float"), "init"],
              ["group", "co2g", "co2_method", pack("period", "str"), "init"],
              ["group", "co2g", "co2_max_period", pack(pmap([("p1", 4380.0)], index_name="period"), "map"), "init"]]
    return _base(entities=entities, active=[("node", "coal_n")], values=values)


def _sum(frame, **keys) -> float:
    import polars as pl

    rows = frame.filter(*(pl.col(k).cast(pl.Utf8) == v for k, v in keys.items()))
    return float(rows["value"].sum())


def _rows_named(mps, family: str) -> dict:
    """``{row: {column: coefficient}}`` of the rows of ``family`` in the MPS file ``mps``."""
    lp = scm.lp_snapshot().parse_mps(mps)
    out: dict = {r: {} for r in lp.rows if r.startswith(family + "[")}
    for (row, col), coef in lp.coefs.items():
        if row in out:
            out[row][col] = coef
    return out


def test_co2_cap_on_the_path(tmp_path, monkeypatch):
    """T-pc.  f_d_k[p1_A] = (0 realized + 2 jump h + 2 tail h) / 4 = 1, so each tail row caps the path at
    4380 x 1 t/a, i.e. (jump coal x + the tail's coal) x 8760 / 4 <= 4380: x + coal_A <= 2 (coal_B too).  Cost of R_0
    (before the factor): jump 10x + 30(2 - x) = 60 - 20x; tail A (weight 0.5, 2 MWh) coal 2 - x, gas x:
    0.5 x (20 - 10x + 30x) = 10 + 10x; B 0.  Total 70 - 10x -> x = 2: 50 -> 2190 x 50 = 109 500.  The realized jump
    burns 2 t, the carry; R_1 (f_d_k (2 + 2) / 4 = 1): 4380 - 2 x 8760 / 4 = 0 t/a left, gas only: 2 x 30 x 2190 =
    131 400.  (Path rows with f_d_k[p1_A] = 0.5, the tail's hours only: x + coal_A <= 1 -> x = 1, 2190 x 70 = 153 300;
    rows per name: x <= 1, coal_A <= 1 -> 2190 x 60 = 131 400.)"""
    built, steps, datas = sr._run(_co2(), tmp_path, monkeypatch)
    assert len(built) == 2
    first = list(steps.values())[0]
    assert sr.ts._values(datas[0], "p_co2_f_d_k", "d") == {("p1",): 0.5, ("p1_A",): 1.0, ("p1_B",): 1.0}
    rows = _rows_named(built[0][1], "co2_max_period")
    assert sorted(rows) == ["co2_max_period[co2g,p1_A]", "co2_max_period[co2g,p1_B]"]
    for row, f in (("co2_max_period[co2g,p1_A]", "p1_A"), ("co2_max_period[co2g,p1_B]", "p1_B")):
        assert sorted(c for c in rows[row] if c.startswith("v_flow")) == [
            f"v_flow[coal_u,coal_n,elec,{d},{t}]" for d, t in (("p1", "t01"), ("p1", "t02"), (f, "t03"), (f, "t04"))]
    # v_flow is per unit size (the existing 10 MW): 0.2 x 10 = 2 MWh of coal in the jump.
    assert _sum(first.solution.value("v_flow"), p="coal_u", d="p1") == pytest.approx(0.2, abs=1e-9)
    assert sr.ts._values(datas[1], "p_co2_cum_realized_tonnes", "g", "d") == pytest.approx({("co2g", "p1"): 2.0})
    assert sr._objectives(steps) == pytest.approx([109_500.0, 131_400.0], rel=1e-9)


def test_unconverted_co2_rows_are_refused(tmp_path, monkeypatch):
    """With the path helper bypassed, co2_max_period builds one row per name; G1 refuses the label's row."""
    from flextool.engine.model import core

    monkeypatch.setattr(core, "d_period_path", lambda b, *families: None)
    with pytest.raises(RuntimeError, match=r"Internal error: co2_max_period on period 'p1' of roll 'st' is not "
                                           r"path-aware"):
        sr._run(_co2(), tmp_path, monkeypatch)


def test_ladder_tier_on_the_path(tmp_path, monkeypatch):
    """``coal_u`` (no operating cost) burns commodity ``coal`` on the annual ladder: tier 1 at 10 EUR/MWh up to 4380
    MWh/a (x 4 / 8760 x f_d_k[p1_A] = 1: 2 MWh on the path), tier 2 at 30.  The tail row caps tier-1 purchases of
    the path, jump + tail <= 2 MWh: the costs of T-pc, R_0 = 109 500; R_1 has 4380 x 4 / 8760 - 2 realized = 0 MWh
    left in tier 1: 2 x 30 -> 131 400.  (Rows per name: jump <= 1, tail <= 1 -> 131 400.)"""
    tier = {"type": "map", "index_type": "str", "rank": 1}
    tiers = {"index_type": "str", "rank": 2, "index_name": "tier",
             "data": [["1", {**tier, "data": [["price", 10.0], ["quantity", 4380.0]]}],
                      ["2", {**tier, "data": [["price", 30.0], ["quantity", float("inf")]]}]]}
    spec = _base(entities=[["node", "coal_n", None], ["commodity", "coal", None],
                           ["commodity__node", ["coal", "coal_n"], None],
                           ["unit__inputNode", ["coal_u", "coal_n"], None]],
                 active=[("node", "coal_n")],
                 values=[["node", "coal_n", "node_type", pack("commodity", "str"), "init"],
                         ["commodity", "coal", "price_method", pack("price_ladder_annual", "str"), "init"],
                         ["commodity", "coal", "price_ladder_annual", pack(tiers, "map"), "init"]],
                 coal_cost=0.0)
    built, steps, _datas = sr._run(spec, tmp_path, monkeypatch)
    assert sorted(_rows_named(built[0][1], "ladder_tier_cap_annual_roll")) == [
        "ladder_tier_cap_annual_roll[coal,1,p1_A]", "ladder_tier_cap_annual_roll[coal,1,p1_B]"]
    assert sr._objectives(steps) == pytest.approx([109_500.0, 131_400.0], rel=1e-9)


def test_cumulative_flow_on_the_path(tmp_path, monkeypatch):
    """Flow group ``fg`` = coal_u's output, ``max_cumulative_flow`` 0.5 MW in p1: the tail row caps the path's coal
    at 0.5 x 4 h = 2 MWh (x + coal_A <= 2): R_0 as T-pc, 109 500.  R_1's own row (p1 t03-t04): coal <= 0.5 x 2 = 1
    (no carry across rolls): 10 + 30 = 40 -> 87 600.  (Rows per name: x <= 1, coal_A <= 1 -> 2190 x 60 = 131 400.)"""
    spec = _base(entities=[["flowGroup", "fg", None], ["flowGroup__unit__node", ["fg", "coal_u", "elec"], None]],
                 values=[["flowGroup", "fg", "max_cumulative_flow",
                          pack(pmap([("p1", 0.5)], index_name="period"), "map"), "init"]])
    built, steps, _datas = sr._run(spec, tmp_path, monkeypatch)
    assert sorted(_rows_named(built[0][1], "maxCumulative_flow_period")) == [
        "maxCumulative_flow_period[fg,p1_A]", "maxCumulative_flow_period[fg,p1_B]"]
    assert sr._objectives(steps) == pytest.approx([109_500.0, 87_600.0], rel=1e-9)


def test_period_balance_on_the_path(tmp_path, monkeypatch):
    """Node ``water`` (balance_within_period, inflow 0.5 MW) feeds ``hyd`` (1 MW, no cost) into elec; no coal, gas
    10 EUR/MWh in t01-t02 and 30 in t03-t04, demand 1 on every branch.  The tail row balances the path: water used
    in the jump + in the tail = 0.5 x 4 = 2 MWh, all in the tail (it saves 30 there, 10 in the jump): R_0 = jump gas
    2 x 10 = 20 -> 2190 x 20 = 43 800.  R_1 balances its own window: 1 MWh of water, 1 of gas at 30 -> 65 700.
    (Rows per name: 1 MWh in the jump and 1 in each tail -> 10 + 30 = 40 -> 87 600.)"""
    spec = _base(coal=False, demand_b=1.0, gas_cost={"t01": 10.0, "t02": 10.0, "t03": 30.0, "t04": 30.0},
                 entities=[["node", "water", None], ["unit", "hyd", None],
                           ["unit__inputNode", ["hyd", "water"], None], ["unit__outputNode", ["hyd", "elec"], None]],
                 active=[("node", "water"), ("unit", "hyd")],
                 values=[*sr._node("water", "balance_within_period", pack(0.5, "float")),
                         *sr._unit("hyd", "elec", 1.0, 0.0)])
    built, steps, _datas = sr._run(spec, tmp_path, monkeypatch)
    assert sorted(_rows_named(built[0][1], "nodeBalancePeriod_eq")) == [
        "nodeBalancePeriod_eq[water,p1_A]", "nodeBalancePeriod_eq[water,p1_B]"]
    assert sr._objectives(steps) == pytest.approx([43_800.0, 65_700.0], rel=1e-9)


def test_period_total_constraint_on_the_path(tmp_path, monkeypatch):
    """User constraint ``c`` (period_total, coal_u's output <= 4380 per year: w = 8760 / 4 per MWh, so <= 2 MWh over
    the period).  The rolls realize p1 in halves, so the run leaves period_total rows out; with that check forced to
    enforce, the window fraction of a path is (2 + 2) / 4 = 1 and the tail rows hold the path: R_0 as T-pc, 109 500.
    R_1 (p1 t03-t04, window fraction 0.5) has no row: coal only, 2 x 10 -> 43 800.  (Per name every fraction is
    0.5: no row, R_0 = 10 x 2 + 0.5 x 20 = 30 -> 65 700.)"""
    from flextool.engine.solve import _recursive_solve

    def enforcing(self, solve, *args):
        self.state.solve.period_total_enforcing_solves.add(solve)

    monkeypatch.setattr(_recursive_solve.RecursiveSolveBuilder, "_check_period_total_rolls", enforcing)
    spec = _base(entities=[["constraint", "c", None]],
                 values=[["constraint", "c", "sense", pack("less_than", "str"), "init"],
                         ["constraint", "c", "constant", pack(4380.0, "float"), "init"],
                         ["constraint", "c", "constraint_period_method", pack("period_total", "str"), "init"],
                         ["unit__outputNode", ["coal_u", "elec"], "constraint_flow_coeff",
                          pack(pmap([("c", 1.0)], index_name="constraint"), "map"), "init"]])
    built, steps, _datas = sr._run(spec, tmp_path, monkeypatch)
    assert sorted(_rows_named(built[0][1], "process_constraint_less_than_period_total")) == [
        "process_constraint_less_than_period_total[c,p1_A]", "process_constraint_less_than_period_total[c,p1_B]"]
    assert _rows_named(built[1][1], "process_constraint_less_than_period_total") == {}
    assert sr._objectives(steps) == pytest.approx([109_500.0, 43_800.0], rel=1e-9)


def test_step_weights_one_scale_over_the_path(tmp_path, monkeypatch):
    """T-cw weighted: ``test_split_roll_lp._tcw`` with timeset weights 2, 1, 1.  One scale over the path p1 t01 +
    p1_f t02-t03 (= the roll without forecasts): 3 / 4 -> 1.5, 0.75, 0.75.  R_0 = 2920 x (1.5 x 10 + 0.25 x 0.75 x
    (10 + 10) + 0.75 x 0.75 x (20 + 20)) = 2920 x 41.25 = 120 450 (per name: 1 on the label, 1, 1 on a tail ->
    131 400).  R_1 = p1 t02 + tails t03 (weights 1, 1 -> scale 1): 2920 x (10 + 0.5 x 10 + 0.5 x 10) = 58 400; R_2 =
    p1 t03 (weight 1 -> 1): 29 200."""
    spec = sr._tcw()
    spec["parameter_values"].append(["timeset", "ts", "timeset_weights",
                                     pack(pmap([("t01", 2.0), ("t02", 1.0), ("t03", 1.0)], index_name="time"),
                                          "map"), "init"])
    built, steps, datas = sr._run(spec, tmp_path, monkeypatch)
    assert sr.ts._values(datas[0], "p_timestep_weight", "d", "t") == {
        ("p1", "t01"): 1.5, ("p1_A", "t02"): 0.75, ("p1_A", "t03"): 0.75,
        ("p1_B", "t02"): 0.75, ("p1_B", "t03"): 0.75}
    assert sr._objectives(steps) == pytest.approx([120_450.0, 58_400.0, 29_200.0], rel=1e-9)
