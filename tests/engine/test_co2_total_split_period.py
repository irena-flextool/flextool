"""``co2_max_total`` in a chain whose solves each hold part of one period (item 1042, D49): the budget is split by the
hour weight of each solve's window, so the chain's realized emissions equal a single solve's limit.

Model: ``test_flowgroup_cumulative_carry._spec`` with ``chain`` and ``split`` 3: solve ``s1`` holds p1 on t01-t03
(period share 3 / 8760, objective factor 2920), ``s2`` p1 on t04 (share 1 / 8760, factor 8760).  ``coal_u`` draws
coal (1 t CO2 per MWh, price 0) from ``coal_n`` in group ``co2g``; demand 1 MW; coal 10 EUR/MWh.
``co2_max_total`` 2190 t: a single solve of p1 on t01-t04 caps E·8760 / 4 ≤ 2190, i.e. 1 t over the 4 h.

s1 holds its 3 h as the whole period: E_1·2920 ≤ 2190, so E_1 ≤ 0.75 t.  s2 knows the period as the 3 h s1
realized plus its own 1 h: share* = 4 / 8760, so its hour weighs 8760 / 4 = 2190 and its share reached is 4 / 4 = 1.
s1's realized tonnes are weighed the same: (E_1 + E_2)·2190 ≤ 2190, E_2 ≤ 1 − E_1.  Before, s2 weighed its hour
8760 and s1's tonnes 2920: E_2·8760 ≤ 2190 − E_1·2920.

The 12-step chains (``_chain12``) run the same unit and cap on p1 over t01-t12 (12 h: 3 t in a single solve, 0.25 t
per hour reached), gas 40, so coal always runs up to the cap.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import test_flowgroup_cumulative_carry as fc

pack = fc.pack
pytestmark = pytest.mark.solver


def _co2_chain(gas_s1: float) -> dict:
    spec = fc._spec(gas={"t01": gas_s1, "t02": gas_s1, "t03": gas_s1, "t04": 40.0}, coal=10.0, limits=[],
                    chain=True, fuel_arc=True, split=3.0)
    spec["entities"] = [e for e in spec["entities"] if not e[0].startswith("flowGroup")]
    spec["entities"] += [["group", "co2g", None], ["group__node", ["co2g", "coal_n"], None]]
    for v in spec["parameter_values"]:
        if v[:3] == ["unit", "coal_u", "efficiency"]:
            v[3] = pack(1.0, "float")
    spec["parameter_values"] += [["commodity", "coal", "co2_content", pack(1.0, "float"), "init"],
                                 ["group", "co2g", "co2_method", pack("total", "str"), "init"],
                                 ["group", "co2g", "co2_max_total", pack(2190.0, "float"), "init"]]
    return spec


def _coal(step) -> float:
    """MWh of coal_u's output in *step* (unit size 10 MW, 1 h steps)."""
    frame = step.solution.value("v_flow").with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source"))
    rows = frame.filter((pl.col("p") == "coal_u") & (pl.col("source") == "coal_n"))
    return float(rows["value"].sum()) * 10.0


def test_split_period_budget_is_shared_by_hour_weight(tmp_path):
    """Gas 40 everywhere: coal is cheaper, so s1 burns its 0.75 t and s2 the remaining 0.25 t (1 t in total).
    s1: 0.75 x 10 + 2.25 x 40 = 97.5 -> x 2920 = 284 700.  s2: 0.25 x 10 + 0.75 x 40 = 32.5 -> x 8760 = 284 700
    (before: s2's RHS 2190 - 0.75 x 2920 = 0, gas only, 40 -> 350 400; 0.75 t in total).  The carrier s2 hands on
    weighs the period's 1 t by s2's share: represented 1 x 1 / (4 / 8760) = 2190 (before: 0.75 x 2920 + 0 = 2190
    from s1 alone, which a later solve would add to 0.25 x 8760 of a burning s2)."""
    steps = fc._run(_co2_chain(40.0), tmp_path)
    assert len(steps) == 2
    assert [_coal(s) for s in steps] == pytest.approx([0.75, 0.25], abs=1e-9)
    assert [s.solution.obj for s in steps] == pytest.approx([97.5 * 2920.0, 32.5 * 8760.0], rel=1e-9)
    carried = steps[1].handoff.cumulative_co2.select(pl.col("period").cast(pl.Utf8), "value", "represented").rows()
    assert carried == [("p1", pytest.approx(1.0, abs=1e-9), pytest.approx(2190.0, rel=1e-9))]


def test_split_period_unused_budget_carries_to_the_later_part(tmp_path):
    """Gas 5 in s1: s1 burns no coal, 3 x 5 = 15 -> 43 800.  s2 may burn 1 - 0 = 1 t: coal 1, 10 -> 87 600 (before:
    E_2 x 8760 <= 2190, 0.25 t: 2.5 + 30 = 32.5 -> 284 700)."""
    steps = fc._run(_co2_chain(5.0), tmp_path)
    assert [_coal(s) for s in steps] == pytest.approx([0.0, 1.0], abs=1e-9)
    assert [s.solution.obj for s in steps] == pytest.approx([15.0 * 2920.0, 10.0 * 8760.0], rel=1e-9)


STEPS12 = tuple(f"t{i:02d}" for i in range(1, 13))


def _chain12(solves: list, timesets: dict, *, years: "dict | None" = None, gas=40.0, cap: float = 2190.0,
             group_node: str = "coal_n") -> dict:
    """The 4-step model's node, units and CO2 cap on a 12-step timeline.  *solves* ``(name, periods, (jump, horizon)
    | None[, realized periods])``: *periods* a timeset name (p1 on it) or ``{period: timeset}``, every period realized
    unless listed; *timesets* ``{name: (first step, hours)}``; *years* ``{period: years_represented}`` (1 when not
    given); *gas* a constant or ``{step: cost}`` over the 12 steps (the 4-step time map has no t05-t12); *cap* the
    ``co2_max_total``; *group_node* the node of group ``co2g``."""
    spec = _co2_chain(40.0)
    spec["entities"] = [e for e in spec["entities"] if e[0] not in ("solve", "timeset")
                        and e[:2] != ["group__node", ["co2g", "coal_n"]]]
    spec["entities"] += [*(["timeset", t, None] for t in timesets), *(["solve", s[0], None] for s in solves),
                         ["group__node", ["co2g", group_node], None]]
    values = [v for v in spec["parameter_values"]
              if v[0] not in ("solve", "timeset") and v[:3] != ["model", "flextool", "solves"]
              and v[:3] != ["timeline", "tl", "timestep_duration"]]
    gas_cost = ["unit__outputNode", ["gas", "elec"], "other_operational_cost"]
    gas_value = pack(gas, "float") if isinstance(gas, float) else pack(
        fc.pmap(list(gas.items()), index_name="time"), "map")
    values = [[*gas_cost, gas_value, "init"] if v[:3] == gas_cost
              else [*v[:3], pack(cap, "float"), v[4]] if v[:3] == ["group", "co2g", "co2_max_total"] else v
              for v in values]
    values += [["timeline", "tl", "timestep_duration",
                pack(fc.pmap([(t, 1.0) for t in STEPS12], index_name="time"), "map"), "init"],
               ["model", "flextool", "solves", fc.array([s[0] for s in solves]), "init"]]
    for name, (start, hours) in timesets.items():
        values += [["timeset", name, "timeline", pack("tl", "str"), "init"],
                   ["timeset", name, "timeset_duration", pack(fc.pmap([(start, hours)], index_name="time"), "map"),
                    "init"]]
    for name, periods, roll, *realized in solves:
        periods = {"p1": periods} if isinstance(periods, str) else periods
        rows = [("solve_mode", pack("rolling_window" if roll else "single_solve", "str")),
                ("period_timeset", pack(fc.pmap(list(periods.items()), index_name="period"), "map")),
                ("realized_periods", fc.array(realized[0] if realized else list(periods))),
                ("years_represented", pack(fc.pmap([(p, (years or {}).get(p, 1.0)) for p in periods],
                                                   index_name="period"), "map"))]
        if roll:
            rows += [("rolling_solve_jump", pack(float(roll[0]), "float")),
                     ("rolling_solve_horizon", pack(float(roll[1]), "float"))]
        values += [["solve", name, par, val, "init"] for par, val in rows]
    spec["parameter_values"] = values
    return spec


def _cap_rhs(step) -> float:
    from tests.engine._lp_rows import row_bounds
    from tests.engine._unit_db import build

    return row_bounds(build(step.flex_data), "co2_max_total", "co2g")["co2_max_total[co2g]"][1]


def test_split_period_then_rolling_shares_the_budget_by_hour_weight(tmp_path):
    """s1 single on t01-t03, then s2 rolling on t04-t12 (jump 3, horizon 3).  s1: share 3 / 8760, E_1·2920 <= 2190,
    0.75 t.  s2 knows p1 as s1's 3 h plus its own 9 h: share* = 12 / 8760, every hour 730 per t, share reached
    (realized h + the roll's 3 h) / 12.  Roll 0: 2190·6/12 - 0.75·730 = 547.5; roll 1: 2190·9/12 - 1.5·730 = 547.5;
    roll 2: 2190 - 2.25·730 = 547.5: each roll 547.5 / 730 = 0.75 t, 3 t in all, a single solve's limit.  (Before: the
    rolls weighed s1's 0.75 t at 2920 against their 973.3 per t and roll 0's RHS was 2190·6/9 - 2190 < 0:
    infeasible.)"""
    steps = fc._run(_chain12([("s1", "a", None), ("s2", "b", (3, 3))], {"a": ("t01", 3.0), "b": ("t04", 9.0)}),
                    tmp_path)
    assert len(steps) == 4
    assert [_coal(s) for s in steps] == pytest.approx([0.75] * 4, abs=1e-9)
    assert [_cap_rhs(s) for s in steps[1:]] == pytest.approx([547.5] * 3, rel=1e-9)


def test_rolling_then_single_part_of_the_period(tmp_path):
    """s1 rolling on t01-t09 (jump 3, horizon 3, share 9 / 8760, 973.3 per t: each roll 2190·k/3 - realized, 0.75 t),
    then s2 single on t10-t12: share* = (9 + 3) / 8760, 730 per t, 2190 - 2.25·730 = 547.5, 0.75 t; 3 t in all."""
    steps = fc._run(_chain12([("s1", "a", (3, 3)), ("s2", "b", None)], {"a": ("t01", 9.0), "b": ("t10", 3.0)}),
                    tmp_path)
    assert [_coal(s) for s in steps] == pytest.approx([0.75] * 4, abs=1e-9)
    assert _cap_rhs(steps[-1]) == pytest.approx(547.5, rel=1e-9)


def test_three_single_solves_each_hold_a_third(tmp_path):
    """s1, s2, s3 single on 4 h each.  s1: share 4 / 8760, 2190 per t, 1 t.  s2: share* 8 / 8760, 1095 per t,
    2190 - 1·1095 = 1095, 1 t.  s3: share* 12 / 8760, 730 per t, 2190 - 2·730 = 730, 1 t; 3 t in all."""
    steps = fc._run(_chain12([("s1", "a", None), ("s2", "b", None), ("s3", "c", None)],
                             {"a": ("t01", 4.0), "b": ("t05", 4.0), "c": ("t09", 4.0)}), tmp_path)
    assert [_coal(s) for s in steps] == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    assert [_cap_rhs(s) for s in steps] == pytest.approx([2190.0, 1095.0, 730.0], rel=1e-9)


def _coal_in(step, period: str) -> float:
    """MWh of coal_u's output in *period* of *step*."""
    frame = step.solution.value("v_flow").with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "d"))
    rows = frame.filter((pl.col("p") == "coal_u") & (pl.col("source") == "coal_n") & (pl.col("d") == period))
    return float(rows["value"].sum()) * 10.0


OWN_PERIODS = ([("s1", {"p1": "a"}, None), ("s2", {"p2": "b"}, None)], {"a": ("t01", 4.0), "b": ("t05", 4.0)})


def test_solves_with_their_own_periods_refuse_a_total_cap(tmp_path):
    """D54: s1 holds p1, s2 holds p2, and co2g (on coal_n, which carries coal) has a co2_max_total: neither solve
    sees the other's period, so the run stops before anything solves, naming both (one more solve)."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError) as err:
        fc._run(_chain12(*OWN_PERIODS), tmp_path)
    assert str(err.value).startswith("co2_max_total of group(s) 'co2g' limits the emissions over all the periods of "
                                     "the model, but solve 's1' (and 1 more solve) does not have period(s) 'p2': ")


def test_a_total_cap_without_an_emitter_has_no_row_and_no_check(tmp_path):
    """The same solves with co2g on elec, a node without a CO2-bearing commodity: no process emits in co2g, so the
    loader builds no co2_max_total row and the run is not stopped; coal is not limited (4 MWh in each solve)."""
    steps = fc._run(_chain12(*OWN_PERIODS, group_node="elec"), tmp_path)
    assert [_coal(s) for s in steps] == pytest.approx([4.0, 4.0], abs=1e-9)


def test_a_period_without_years_is_not_a_model_period(tmp_path):
    """s1 holds p0 (t01, years_represented 0) and p1 (t02-t04), s2 p1 on t05-t12.  p0 represents no years, so s2
    sees every period that counts (p1) and the run goes on.  p0 carries no weight in the row: its 1 MWh of coal is
    free.  s1: p1 share 3/8760, E·2920 <= 2190, 0.75 t.  s2: share* (3 + 8)/8760, 796.36 per t, 2190 - 0.75·796.36
    = 1592.7, so 2 t over its 8 h; p1 in all 2.75 t = 11 h x 0.25."""
    steps = fc._run(_chain12([("s1", {"p0": "z", "p1": "a"}, None), ("s2", "b", None)],
                             {"z": ("t01", 1.0), "a": ("t02", 3.0), "b": ("t05", 8.0)}, years={"p0": 0.0}),
                    tmp_path)
    assert [_coal_in(steps[0], "p0"), _coal_in(steps[0], "p1"), _coal_in(steps[1], "p1")] == pytest.approx(
        [1.0, 0.75, 2.0], abs=1e-9)


def test_two_solves_that_see_both_periods_carry_the_realized_one(tmp_path):
    """s1 and s2 both hold p1 (t01-t02) and p2 (t03-t04), share 2/8760 each, years 1; s1 realizes p1, s2 p2.
    co2_max_total 4380 = 1 t in a single solve (a tonne weighs 4380).  Gas 40 in p1 and 30 in p2, coal 10.
    s1 (both periods reached): (E_p1 + E_p2)·4380 <= 4380; coal saves more in p1: 1 t there.  Objective
    4380 x (1 x 10 + 1 x 40 + 2 x 30) = 481 800.  s2: p1 has 2 h realized by s1, so its share* is (2 + 2)/8760
    (2190 per t) and s1's 1 t counts 2190: E_p1·2190 + E_p2·4380 <= 4380 - 2190 = 2190.  A tonne of budget
    saves 30/2190 in p1 against 20/4380 in p2, so s2 burns its 1 t in p1 again and none in its realized p2:
    481 800.  Realized in all: 1 t, the limit."""
    gas = {t: 40.0 if t in ("t01", "t02") else 30.0 for t in STEPS12}
    both = {"p1": "a", "p2": "b"}
    steps = fc._run(_chain12([("s1", both, None, ["p1"]), ("s2", both, None, ["p2"])],
                             {"a": ("t01", 2.0), "b": ("t03", 2.0)}, gas=gas, cap=4380.0), tmp_path)
    assert [(_coal_in(s, "p1"), _coal_in(s, "p2")) for s in steps] == [
        (pytest.approx(1.0, abs=1e-9), pytest.approx(0.0, abs=1e-9))] * 2
    assert _cap_rhs(steps[1]) == pytest.approx(2190.0, rel=1e-9)
    assert [s.solution.obj for s in steps] == pytest.approx([481_800.0] * 2, rel=1e-9)
