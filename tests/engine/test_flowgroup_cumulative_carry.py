"""Flow-group cumulative limits carry across rolls and solves (D19, D44): each roll may use the limit times the hours
realized so far plus its own hours, minus the flow earlier rolls and solves realized.

Model (``_spec``): 1 h steps t01-t04 on one timeline, ``years_represented`` 1.  Node ``elec`` has demand 1 MW;
``coal_u`` and ``gas`` are 10 MW each, with the costs each case gives (EUR/MWh).  Flow group ``fg`` = (coal_u, elec)
unless a case says otherwise.  The rolling solve ``st`` has jump 1 and horizon 2: R0 {t01, t02}, R1 {t02, t03},
R2 {t03, t04}, R3 {t04}.  The objective of a roll is its operating cost times 8760 / the period's hours (one period
of 4 h: 2190; two periods of 2 h: 4380).  Every expected value is derived by hand in the case's docstring; "before"
gives the values without the carry (each roll limited to its own window).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm

pack, pmap, array = scm.pack, scm.pmap, scm.array
pytestmark = pytest.mark.solver
STEPS = ("t01", "t02", "t03", "t04")
DECREASING = {"t01": 40.0, "t02": 35.0, "t03": 30.0, "t04": 25.0}


def _cost(cost) -> list:
    if isinstance(cost, float):
        return pack(cost, "float")
    return pack(pmap(list(cost.items()), index_name="time"), "map")


def _unit(name: str, cost, efficiency: float = 1.0) -> list:
    return [["unit", name, "efficiency", pack(efficiency, "float"), "init"],
            ["unit", name, "existing", pack(10.0, "float"), "init"],
            ["unit__outputNode", [name, "elec"], "other_operational_cost", _cost(cost), "init"]]


def _spec(*, gas, coal, limits: list, two_periods: bool = False, chain: bool = False,
          fuel_arc: bool = False, split: float = 2.0) -> dict:
    """The model above.  *limits*: ``(parameter, packed value)`` on ``fg``.  *two_periods*: p1 on t01-t02 and p2 on
    t03-t04 (else p1 on t01-t04).  *chain*: two single solves ``s1`` (p1 on t01-t02) and ``s2`` (p1 on t03-t04)
    instead of the rolling solve, ``s1`` on the first *split* hours (ts_a) and ``s2`` on the rest (ts_b).  *fuel_arc*: coal_u (efficiency 0.5) draws from commodity node ``coal_n`` (price 0)
    and ``fg`` = (coal_u, coal_n)."""
    from flextool.update_flextool.fixture_schema import schema_sections

    spec = schema_sections("packed")
    spec["alternatives"] = [["init", ""]]
    spec["scenarios"] = [["base", False, ""]]
    spec["scenario_alternatives"] = [["base", "init", None]]
    solves = ["s1", "s2"] if chain else ["st"]
    member = ["fg", "coal_u", "coal_n" if fuel_arc else "elec"]
    ents = [["model", "flextool", None], ["timeline", "tl", None], ["timeset", "ts", None],
            ["timeset", "ts_a", None], ["timeset", "ts_b", None], ["node", "elec", None],
            ["unit", "gas", None], ["unit", "coal_u", None],
            ["unit__outputNode", ["gas", "elec"], None], ["unit__outputNode", ["coal_u", "elec"], None],
            ["flowGroup", "fg", None], ["flowGroup__unit__node", member, None],
            *(["solve", s, None] for s in solves)]
    active = [("node", "elec"), ("unit", "gas"), ("unit", "coal_u")]
    values = [
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in STEPS], index_name="time"), "map"),
         "init"],
        *(["timeset", name, par, val, "init"] for name, start, hours in
          (("ts", "t01", 4.0), ("ts_a", "t01", split), ("ts_b", STEPS[int(split)], 4.0 - split))
          for par, val in (("timeline", pack("tl", "str")),
                           ("timeset_duration", pack(pmap([(start, hours)], index_name="time"), "map")))),
        ["node", "elec", "node_type", pack("balance", "str"), "init"],
        ["node", "elec", "inflow", pack(-1.0, "float"), "init"],
        ["node", "elec", "penalty_up", pack(10000.0, "float"), "init"],
        ["node", "elec", "penalty_down", pack(10000.0, "float"), "init"],
        *_unit("gas", gas), *_unit("coal_u", coal, 0.5 if fuel_arc else 1.0),
        *(["flowGroup", "fg", par, val, "init"] for par, val in limits),
        ["model", "flextool", "solves", array(solves), "init"],
    ]
    if fuel_arc:
        ents += [["node", "coal_n", None], ["commodity", "coal", None], ["commodity__node", ["coal", "coal_n"], None],
                 ["unit__inputNode", ["coal_u", "coal_n"], None]]
        active.append(("node", "coal_n"))
        values += [["node", "coal_n", "node_type", pack("commodity", "str"), "init"],
                   ["commodity", "coal", "price", pack(0.0, "float"), "init"]]
    if chain:
        for solve, timeset in (("s1", "ts_a"), ("s2", "ts_b")):
            values += [["solve", solve, par, val, "init"] for par, val in (
                ("solve_mode", pack("single_solve", "str")),
                ("period_timeset", pack(pmap([("p1", timeset)], index_name="period"), "map")),
                ("realized_periods", array(["p1"])),
                ("years_represented", pack(pmap([("p1", 1.0)], index_name="period"), "map")))]
    else:
        periods = {"p1": "ts_a", "p2": "ts_b"} if two_periods else {"p1": "ts"}
        values += [["solve", "st", par, val, "init"] for par, val in (
            ("solve_mode", pack("rolling_window", "str")),
            ("period_timeset", pack(pmap(list(periods.items()), index_name="period"), "map")),
            ("realized_periods", array(list(periods))),
            ("years_represented", pack(pmap([(p, 1.0) for p in periods], index_name="period"), "map")),
            ("rolling_solve_jump", pack(1.0, "float")),
            ("rolling_solve_horizon", pack(2.0, "float")))]
    spec["entities"] = ents
    spec["entity_alternatives"] = [[c, [n], "init", True] for c, n in active]
    spec["parameter_values"] = values
    return spec


def _run(spec: dict, folder) -> list:
    _built, steps = scm.run_spec(spec, folder)
    return list(steps.values())


def _objectives(steps, factor: float) -> list[float]:
    return [s.solution.obj / factor for s in steps]


def _carried(step, column: str) -> dict:
    """``{period: value}`` of the column *column* of ``fg``'s realized history that *step* read
    (``flex_data.flowgroup_realized``; empty without history)."""
    frame = step.flex_data.flowgroup_realized
    if frame is None:
        return {}
    frame = frame.filter(pl.col("g").cast(pl.Utf8) == "fg")
    return {str(d): v for d, v in frame.select(pl.col("d").cast(pl.Utf8), column).rows()}


def _jump_flow(step, source: str, sink: str, t: str, scale: float = 10.0) -> float:
    """MW of coal_u's flow (source -> sink) at step *t* of *step* x unit size 10 MW."""
    frame = step.solution.value("v_flow").with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink", "t"))
    rows = frame.filter((pl.col("p") == "coal_u") & (pl.col("source") == source) & (pl.col("sink") == sink)
                        & (pl.col("t") == t))
    return float(rows["value"].sum()) * scale


def test_max_constant_carries_across_rolls(tmp_path):
    """A: ``max_cumulative_flow`` 0.5 MW (budget 0.5 x 4 = 2 MWh), coal 10, gas 40/35/30/25 (coal to the earliest step).
    R0 coal <= 0.5 x 2 = 1 -> t01 1: 10 + 35 = 45.  R1 <= 0.5 x (1 + 2) - 1 = 0.5 -> t02 0.5: 5 + 17.5 + 30 = 52.5.
    R2 <= 0.5 x 4 - 1.5 = 0.5 -> t03 0.5: 5 + 15 + 25 = 45.  R3 <= 0.5 x 4 - 2 = 0: 25.  Realized 2 MWh (before:
    1 + 1 + 1 + 0.5 = 3.5, objectives 45, 40, 35, 17.5).  The history read by R1-R3: flow 1, 1.5, 2 over 1, 2, 3 h;
    the weight years / period share = 2190."""
    steps = _run(_spec(gas=DECREASING, coal=10.0, limits=[("max_cumulative_flow", pack(0.5, "float"))]), tmp_path)
    assert len(steps) == 4
    assert _objectives(steps, 2190.0) == pytest.approx([45.0, 52.5, 45.0, 25.0], rel=1e-9)
    realized = [_jump_flow(s, "coal_u", "elec", t) for s, t in zip(steps, STEPS)]
    assert realized == pytest.approx([1.0, 0.5, 0.5, 0.0], abs=1e-9)
    assert sum(realized) <= 2.0 + 1e-9
    assert _carried(steps[0], "flow") == {}
    assert [_carried(s, "flow") for s in steps[1:]] == [
        pytest.approx({"p1": v}) for v in (1.0, 1.5, 2.0)]
    assert [_carried(s, "hours") for s in steps[1:]] == [
        pytest.approx({"p1": v}) for v in (1.0, 2.0, 3.0)]
    assert [_carried(s, "weight") for s in steps[1:]] == [pytest.approx({"p1": 2190.0})] * 3


def test_min_constant_carries_and_the_last_roll_is_feasible(tmp_path):
    """B: ``min_cumulative_flow`` 0.5 MW (2 MWh), gas 10, coal 40/35/30/25 (coal to the latest step).
    R0 >= 1 -> t02 1, t01 0: 10 + 35 = 45.  R1 >= 1.5 - 0 -> t03 1, t02 0.5: 17.5 + 5 + 30 = 52.5.  R2 >= 2 - 0.5 = 1.5
    -> t04 1, t03 0.5: 15 + 5 + 25 = 45.  R3 >= 2 - 1 = 1 -> t04 1: 25.  Realized 0 + 0.5 + 0.5 + 1 = 2 MWh (before:
    0.5, objectives 45, 40, 35, 17.5)."""
    steps = _run(_spec(gas=10.0, coal=DECREASING, limits=[("min_cumulative_flow", pack(0.5, "float"))]), tmp_path)
    assert all(s.solution.optimal for s in steps)
    assert _objectives(steps, 2190.0) == pytest.approx([45.0, 52.5, 45.0, 25.0], rel=1e-9)
    realized = [_jump_flow(s, "coal_u", "elec", t) for s, t in zip(steps, STEPS)]
    assert realized == pytest.approx([0.0, 0.5, 0.5, 1.0], abs=1e-9)


def test_period_map_carries_per_period(tmp_path):
    """C: two periods, p1 (t01-t02) and p2 (t03-t04), ``max_cumulative_flow`` {p1: 0.5, p2: 0.25}; coal 10, gas as A.
    R0 p1 <= 0.5 x 2 = 1 -> t01 1: 10 + 35 = 45.  R1 p1 <= 0.5 x (1 + 1) - 1 = 0 -> t02 0; p2 <= 0.25 x 1 -> t03 0.25:
    35 + 2.5 + 22.5 = 60.  R2 p2 <= 0.25 x 2 = 0.5 -> t03 0.5: 5 + 15 + 25 = 45.  R3 p2 <= 0.25 x (1 + 1) - 0.5 = 0: 25.
    Realized p1 1, p2 0.5 (before: 1.5 and 0.75, objectives 45, 47.5, 45, 21.25).  The history keeps the periods
    apart: R1 reads p1 1, R2 p1 1 (its t02 realized 0), R3 p1 1 and p2 0.5."""
    limit = pack(pmap([("p1", 0.5), ("p2", 0.25)], index_name="period"), "map")
    steps = _run(_spec(gas=DECREASING, coal=10.0, limits=[("max_cumulative_flow", limit)], two_periods=True),
                 tmp_path)
    assert len(steps) == 4
    assert _objectives(steps, 4380.0) == pytest.approx([45.0, 60.0, 45.0, 25.0], rel=1e-9)
    assert [_carried(s, "flow") for s in steps[1:]] == [
        pytest.approx({"p1": 1.0}), pytest.approx({"p1": 1.0}), pytest.approx({"p1": 1.0, "p2": 0.5})]


def test_carry_counts_the_fuel_draw_of_a_member_source_arc(tmp_path):
    """D: as A, with ``fg`` = coal_u's fuel arc from ``coal_n`` (efficiency 0.5: 2 MWh fuel per MWh output) and
    ``min_cumulative_flow`` -1 MW: the net flow into coal_n is minus the fuel, Σ >= -1 x 4 = -4, i.e. output <= 2 MWh.
    The dispatch and objectives are those of A; the history read by R1-R3 is -2, -3, -4, minus the jumps' fuel
    computed from v_flow."""
    steps = _run(_spec(gas=DECREASING, coal=10.0, limits=[("min_cumulative_flow", pack(-1.0, "float"))],
                       fuel_arc=True), tmp_path)
    assert _objectives(steps, 2190.0) == pytest.approx([45.0, 52.5, 45.0, 25.0], rel=1e-9)
    fuel = [2.0 * _jump_flow(s, "coal_n", "elec", t) for s, t in zip(steps, STEPS)]
    assert fuel == pytest.approx([2.0, 1.0, 1.0, 0.0], abs=1e-9)
    carried = [_carried(s, "flow") for s in steps[1:]]
    assert carried == [pytest.approx({"p1": v}) for v in (-2.0, -3.0, -4.0)]
    assert carried[1]["p1"] == pytest.approx(-(fuel[0] + fuel[1]), rel=1e-9)


def test_constant_carries_across_the_solves_of_a_chain(tmp_path):
    """E: two solves on p1, ``s1`` (t01-t02) and ``s2`` (t03-t04), period share 2 / 8760 each (factor 4380);
    ``max_cumulative_flow`` 0.5 MW; coal 10, gas 5/5/40/35.  s1: gas is cheaper, coal 0: 5 + 5 = 10.  s2: coal <=
    0.5 x (2 + 2) - 0 = 2 -> coal at t03 and t04: 20 (before: <= 1, 10 + 35 = 45)."""
    steps = _run(_spec(gas={"t01": 5.0, "t02": 5.0, "t03": 40.0, "t04": 35.0}, coal=10.0,
                       limits=[("max_cumulative_flow", pack(0.5, "float"))], chain=True), tmp_path)
    assert len(steps) == 2
    assert _objectives(steps, 4380.0) == pytest.approx([10.0, 20.0], rel=1e-9)
    assert _carried(steps[1], "hours") == pytest.approx({"p1": 2.0})


def test_constant_carries_across_unequal_parts_of_a_period(tmp_path):
    """F: as E with p1 split 3 h + 1 h (``s1`` t01-t03, share 3 / 8760, factor 2920; ``s2`` t04, share 1 / 8760,
    factor 8760); ``max_cumulative_flow`` 0.2 MW; coal 10, gas 5 in s1 and 40 in s2.  s1: gas is cheaper, coal 0:
    3 x 5 = 15.  s2: the period row allows 0.2 x (3 + 1) - 0 = 0.8, and so does the solve row: its history weighs
    s1's 3 h by s2's own years / share (8760), carry 8760 x (0.2 x 3 - 0) / r_max 8760 = 0.6, RHS 0.2 x 1 + 0.6.
    Coal 0.8: 8 + 0.2 x 40 = 16.  (Weighing s1's hours by s1's share, 2920, the solve row allowed 0.2 + 0.2 = 0.4:
    4 + 24 = 28.)"""
    from tests.engine._lp_rows import row_bounds
    from tests.engine._unit_db import build

    gas = {"t01": 5.0, "t02": 5.0, "t03": 5.0, "t04": 40.0}
    steps = _run(_spec(gas=gas, coal=10.0, limits=[("max_cumulative_flow", pack(0.2, "float"))], chain=True,
                       split=3.0), tmp_path)
    assert len(steps) == 2
    assert [s.solution.obj for s in steps] == pytest.approx([15.0 * 2920.0, 16.0 * 8760.0], rel=1e-9)
    lp = build(steps[1].flex_data)
    assert row_bounds(lp, "maxCumulative_flow_solve", "fg")["maxCumulative_flow_solve[fg]"][1] == pytest.approx(
        0.8, rel=1e-12)
    assert row_bounds(lp, "maxCumulative_flow_period", "fg")["maxCumulative_flow_period[fg,p1]"][1] == pytest.approx(
        0.8, rel=1e-12)
