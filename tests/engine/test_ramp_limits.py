"""Ramp limits in MW per step and the step each step ramps from (#34).

The allowed change between two consecutive step averages is ramp_speed × 60 × the mean of the two step durations
× capacity.  A ramp row exists only where a step has a predecessor that is adjacent on the timeline in the
un-rolled solve: no cyclic wrap at the first step of a solve or of a period that does not continue the previous
one, no row across a gap in the timeline or at the start of a representative block; a continuation roll's first
step ramps from the flow the previous roll realized.  Rows inside one period that cannot bind (room ≥ capacity)
are not built; cross-period and carried rows always are.

Fixtures: ``tests.json`` (coal_plant 500 MW, 1 unit, ramp on its output at west) plus override alternatives,
built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_bounds, row_terms, step_of
from tests.engine._unit_db import PLANT, b64, build, flow, make_db, run, utf8

pytestmark = pytest.mark.solver

ARC = f"{PLANT},coal_market,west,"
TOL = 1e-6


def _ramp(speed: float, cls: str = "unit__outputNode", rel: list | None = None, dirs=("up", "down")) -> list:
    rel = rel or [PLANT, "west"]
    out = [[cls, rel, "ramp_method", b64("ramp_limit", "str")]]
    out += [[cls, rel, f"ramp_speed_{d}", b64(speed, "float")] for d in dirs]
    return out


def _tmap(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, v] for t, v in values.items()]}, "map")


def _pmap(values: dict, index_name: str = "period") -> dict:
    return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}


def _solve(tmp_path_factory, alt, values, chain=("init", "west", "coal"), entities=None) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities)
    return run(tmp_path_factory, url, scen)


def _last(steps: dict):
    return list(steps.values())[-1]


TWO_PERIODS_ENTS = [["timeset", "ta", None], ["timeset", "tb", None]]
TWO_PERIODS = [
    ["timeset", "ta", "timeline", b64("y2020", "str")],
    ["timeset", "tb", "timeline", b64("y2020", "str")],
    ["timeset", "ta", "timeset_duration", b64(_pmap({"t0001": 24.0}, "time"), "map")],
    ["timeset", "tb", "timeset_duration", b64(_pmap({"t0025": 24.0}, "time"), "map")],
    ["solve", "y2020_2day_dispatch", "period_timeset", b64(_pmap({"p2020": "ta", "p2025": "tb"}), "map")],
    ["solve", "y2020_2day_dispatch", "realized_periods",
     b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")],
]


# ---------------------------------------------------------------------------
# step length


def test_room_scales_with_step_duration(tmp_path_factory) -> None:
    """2-hour steps t0001–t0024, 1-hour steps after; ramp 0.002/min = 0.12 of capacity per hour.  Rows have
    coefficients ±1 (MW form) and room 0.24 between two 2-h steps, 0.18 from the last 2-h step to the first
    1-h step, 0.12 between 1-h steps.  Solved, a change between two 2-h steps exceeds one hour's ramp (60 MW).
    Before: coefficients ±2 with RHS 0.24, i.e. 60 MW per step whatever its length."""
    durs = {f"t{i:04d}": (2.0 if i <= 24 else 1.0) for i in range(1, 49)}
    step = _last(_solve(tmp_path_factory, "ramp_2h",
                        _ramp(0.002) + [["timeline", "y2020", "timestep_duration", _tmap(durs)]]))
    pb = build(step.flex_data)
    terms = row_terms(pb, "ramp_sink_up_constraint", ARC)
    bounds = row_bounds(pb, "ramp_sink_up_constraint", ARC)
    for t, room in (("t0002", 0.24), ("t0024", 0.24), ("t0025", 0.18), ("t0026", 0.12), ("t0048", 0.12)):
        row = f"ramp_sink_up_constraint[{ARC}p2020,{t}]"
        assert coef(terms[row], f"v_flow[{ARC}p2020,{t}]") == pytest.approx(1.0)
        prev = f"t{int(t[1:]) - 1:04d}"
        assert coef(terms[row], f"v_flow[{ARC}p2020,{prev}]") == pytest.approx(-1.0)
        assert bounds[row][1] == pytest.approx(room)
    f = flow(step, PLANT, "coal_market", "west").to_list()
    assert max(abs(f[i] - f[i - 1]) for i in range(1, 24)) > 60.0 + 1e-3


# ---------------------------------------------------------------------------
# which steps have a predecessor


def test_no_row_without_predecessor(tmp_path_factory) -> None:
    """5weeks timeset (blocks t0001–4, t0009–12, t0013–16, t0025–28, t0037–40 … in two periods): ramp rows
    only at steps with an adjacent predecessor (dt_ramp_link), none at a period start or after a gap.  t0013
    follows t0012 directly, so it keeps its row.  Before: a row at every step (wrap and gap links)."""
    step = _last(_solve(tmp_path_factory, "ramp_5w", _ramp(0.002), chain=("init", "west", "coal", "coal_ramp",
                                                                          "5weeks")))
    fd = step.flex_data
    rows = row_terms(build(fd), "ramp_sink_up_constraint", ARC)
    have = {step_of(r) for r in rows}
    link = {(d, t) for d, t in utf8(fd.dt_ramp_link).select("d", "t").iter_rows()}
    assert have == link
    for d in ("p2020", "p2025"):
        for t in ("t0001", "t0009", "t0025", "t0037"):
            assert (d, t) not in have
        assert (d, "t0013") in have and (d, "t0002") in have


def test_non_binding_room_builds_no_row(tmp_path_factory) -> None:
    """Ramp 0.2/min on 1-h steps (12 × capacity per step) cannot bind inside a period: no ramp rows and the
    objective of the same model without the ramp."""
    with_ramp = _last(_solve(tmp_path_factory, "ramp_free", _ramp(0.2)))
    without = _last(_solve(tmp_path_factory, "ramp_none", []))
    assert not row_terms(build(with_ramp.flex_data), "ramp_sink_up_constraint")
    assert not row_terms(build(with_ramp.flex_data), "ramp_sink_down_constraint")
    assert with_ramp.solution.obj == pytest.approx(without.solution.obj, rel=1e-9)


def test_representative_blocks_do_not_link(tmp_path_factory) -> None:
    """A representative-period timeset with two 24-step blocks t0001 and t0025 that are adjacent on the
    timeline: no ramp row at either block's first step (they represent different base periods), rows inside
    the blocks.  Before: t0025 ramped from t0024 (and t0001 wrapped)."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 0.5], ["t0025", 0.5]]}]]}
    step = _last(_solve(tmp_path_factory, "ramp_rp", _ramp(0.002) + [
        ["timeset", "rp2", "timeline", b64("y2020", "str")],
        ["timeset", "rp2", "timeset_duration", b64(_pmap({"t0001": 24.0, "t0025": 24.0}, "time"), "map")],
        ["timeset", "rp2", "representative_period_weights", b64(rpw, "map")],
        ["solve", "y2020_2day_dispatch", "period_timeset", b64(_pmap({"p2020": "rp2"}), "map")]],
        entities=[["timeset", "rp2", None]]))
    have = {step_of(r) for r in row_terms(build(step.flex_data), "ramp_sink_up_constraint", ARC)}
    assert ("p2020", "t0001") not in have and ("p2020", "t0025") not in have
    assert ("p2020", "t0002") in have and ("p2020", "t0026") in have and ("p2020", "t0048") in have


def test_aggregated_steps_keep_ramp_rows(tmp_path_factory) -> None:
    """Solve-level new_stepduration 6 h (fullYear, 72 h → 12 steps): consecutive aggregated steps are adjacent
    in the solve's own step space, so each step after the first has a ramp row with room 0.002·60·6 = 0.72.
    Review F1: the derived adjacency compared raw timeline ranks (6 apart) and dropped every row."""
    step = _last(_solve(tmp_path_factory, "ramp_6h", _ramp(0.002) + [
        ["solve", "y2020_fullYear_dispatch", "new_stepduration", b64(6.0, "float")]],
        chain=("init", "west", "coal", "fullYear")))
    pb = build(step.flex_data)
    terms = row_terms(pb, "ramp_sink_up_constraint", ARC)
    bounds = row_bounds(pb, "ramp_sink_up_constraint", ARC)
    assert len(terms) == step.flex_data.dt.height - 1 == 11
    row = f"ramp_sink_up_constraint[{ARC}p2020,t0007]"
    assert coef(terms[row], f"v_flow[{ARC}p2020,t0001]") == pytest.approx(-1.0)
    assert bounds[row][1] == pytest.approx(0.72)


def test_aggregated_blocks_keep_their_gaps(tmp_path_factory) -> None:
    """5weeks aggregated to 2-h steps: rows inside a block and across touching blocks (t0013 after t0011),
    none after a gap (t0009, t0025, t0037) or at a period start."""
    step = _last(_solve(tmp_path_factory, "ramp_5w2h", _ramp(0.002) + [
        ["solve", "y2020_5week", "new_stepduration", b64(2.0, "float")]],
        chain=("init", "west", "coal", "coal_ramp", "5weeks")))
    have = {step_of(r) for r in row_terms(build(step.flex_data), "ramp_sink_up_constraint", ARC)}
    for t in ("t0001", "t0009", "t0025", "t0037"):
        assert ("p2020", t) not in have
    assert ("p2020", "t0003") in have and ("p2020", "t0013") in have


# ---------------------------------------------------------------------------
# periods that continue the timeline


def test_cross_period_link(tmp_path_factory) -> None:
    """p2020 = t0001–t0024 and p2025 = t0025–t0048 of one timeline: p2025's first row ramps from p2020's last
    step; no row at p2020 t0001.  Before: both first-step rows were absolute caps v_flow[t1] ≤ 0.12 (the lag
    pointed at a step outside the period), capping the first-step output at 60 MW."""
    step = _last(_solve(tmp_path_factory, "ramp_2p", _ramp(0.002) + TWO_PERIODS, entities=TWO_PERIODS_ENTS))
    pb = build(step.flex_data)
    terms = row_terms(pb, "ramp_sink_up_constraint", ARC)
    assert f"ramp_sink_up_constraint[{ARC}p2020,t0001]" not in terms
    row = terms[f"ramp_sink_up_constraint[{ARC}p2025,t0025]"]
    assert coef(row, f"v_flow[{ARC}p2020,t0024]") == pytest.approx(-1.0)
    assert coef(row, f"v_flow[{ARC}p2025,t0025]") == pytest.approx(1.0)
    f = flow(step, PLANT, "coal_market", "west").to_list()
    assert f[0] > 60.0 + 1e-3


def test_cross_period_row_kept_when_capacity_drops(tmp_path_factory) -> None:
    """Room ρ = 1 (ramp 1/60 per minute, 1-h steps): rows inside a period cannot bind and are dropped, but the
    cross-period down row is built, because the capacity of p2025 (100 MW) is below p2020's (500 MW):
    v_flow[p2025,t0025] − v_flow[p2020,t0024] ≥ −1 × 0.2 units, so coal at t0024 is held to 200 MW and the row
    binds.  Dropping every ρ ≥ 1 row would let it run at 450 MW."""
    vals = (_ramp(1.0 / 60.0, dirs=("down",)) + TWO_PERIODS
            + [["unit", PLANT, "existing", b64(_pmap({"p2020": 500.0, "p2025": 100.0}), "map")],
               ["node", "west", "inflow", b64(-450.0, "float")]])
    step = _last(_solve(tmp_path_factory, "ramp_2p_drop", vals, entities=TWO_PERIODS_ENTS))
    pb = build(step.flex_data)
    rows = row_terms(pb, "ramp_sink_down_constraint", ARC)
    assert set(rows) == {f"ramp_sink_down_constraint[{ARC}p2025,t0025]"}
    v = (utf8(step.solution.value("v_flow"))
         .filter((pl.col("p") == PLANT) & (pl.col("sink") == "west")))
    at = {(d, t): x for d, t, x in v.select("d", "t", "value").iter_rows()}
    assert at[("p2020", "t0024")] - at[("p2025", "t0025")] == pytest.approx(0.2, abs=TOL)


def test_online_unit_at_cross_period_link(tmp_path_factory) -> None:
    """The cross-period up row of a unit with linear online variables carries its startups of that step."""
    vals = _ramp(0.002) + TWO_PERIODS + [["unit", PLANT, "startup_method", b64("linear", "str")],
                                         ["unit", PLANT, "min_load", b64(0.2, "float")]]
    step = _last(_solve(tmp_path_factory, "ramp_2p_uc", vals, entities=TWO_PERIODS_ENTS))
    row = row_terms(build(step.flex_data), "ramp_sink_up_constraint", ARC)[f"ramp_sink_up_constraint[{ARC}p2025,t0025]"]
    assert coef(row, f"v_flow[{ARC}p2020,t0024]") == pytest.approx(-1.0)
    assert coef(row, f"v_startup_linear[{PLANT},p2025,t0025]") == pytest.approx(-1.0)


def test_midhorizon_branches_ramp_from_trunk(tmp_path_factory) -> None:
    """Mid-horizon stochastic fixture (p2035 t0001–3, p2040 branches from t0004): every p2040 branch's first
    row ramps from p2035's last step; no row at p2035 t0001.  Before: absolute caps at both period starts."""
    from tests.engine._unit_db import FLEXTOOL_ROOT

    url, scen = make_db(tmp_path_factory, alt="ramp_mid",
                        values=_ramp(0.002, rel=["base", "elec"]), base_chain=["init", "base", "stoch"],
                        fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_midhorizon_nreal0.json")
    step = _last(run(tmp_path_factory, url, scen))
    terms = row_terms(build(step.flex_data), "ramp_sink_up_constraint", "base,base,elec,")
    assert "ramp_sink_up_constraint[base,base,elec,p2035,t0001]" not in terms
    firsts = [r for r in terms if step_of(r)[1] == "t0004"]
    assert len(firsts) >= 2
    for r in firsts:
        assert coef(terms[r], "v_flow[base,base,elec,p2035,t0003]") == pytest.approx(-1.0)


def test_derived_link_equals_step_previous(tmp_path_factory) -> None:
    """dt_ramp_link built from the source timeline equals the jump == 1 rows of solve_data/step_previous
    (make_step_jump) on gaps, continuing periods, solve-level new_stepduration (review F1), aggregated blocks
    with gaps (the solve's timeline keeps them, #101) and the representative-block rule."""
    from flextool.engine.preprocess.flex_data import _read_ramp_link

    for alt, vals, chain, ents in (
        ("lk_5w", _ramp(0.002), ("init", "west", "coal", "coal_ramp", "5weeks"), None),
        ("lk_2p", _ramp(0.002) + TWO_PERIODS, ("init", "west", "coal"), TWO_PERIODS_ENTS),
        ("lk_6h", _ramp(0.002) + [["solve", "y2020_fullYear_dispatch", "new_stepduration", b64(6.0, "float")]],
         ("init", "west", "coal", "fullYear"), None),
        ("lk_5w2h", _ramp(0.002) + [["solve", "y2020_5week", "new_stepduration", b64(2.0, "float")]],
         ("init", "west", "coal", "coal_ramp", "5weeks"), None),
    ):
        url, scen = make_db(tmp_path_factory, alt=alt, values=vals, base_chain=list(chain), entities=ents)
        step = _last(run(tmp_path_factory, url, scen))
        from pathlib import Path
        csv = _read_ramp_link(Path("solve_data"), provider=step.flex_data_provider)
        derived = step.flex_data.dt_ramp_link
        assert csv is not None and derived is not None
        assert set(utf8(csv).rows()) == set(utf8(derived).rows())


# ---------------------------------------------------------------------------
# rolling


ROLL_CHAIN = ("init", "west", "battery", "wind", "coal", "fullYear", "dispatch_fullYear_roll")


def _realized_flow(steps: dict, arc=(PLANT, "coal_market", "west")) -> list[tuple[str, str, float]]:
    """Realized (d, t, v_flow) of ``arc`` across all rolls, in roll order."""
    out = []
    for s in steps.values():
        real = utf8(s.flex_data.realized_dispatch.rename({"period": "d", "step": "t"})
                    if "period" in s.flex_data.realized_dispatch.columns else s.flex_data.realized_dispatch)
        v = (utf8(s.solution.value("v_flow"))
             .filter((pl.col("p") == arc[0]) & (pl.col("source") == arc[1]) & (pl.col("sink") == arc[2]))
             .join(real.select("d", "t"), on=["d", "t"], how="semi"))
        order = utf8(s.flex_data.dt).with_row_index("i")
        out += v.join(order, on=["d", "t"]).sort("i").select("d", "t", "value").rows()
    return out


@pytest.fixture(scope="module")
def rolling(tmp_path_factory):
    """Rolling fullYear dispatch (4 periods × 72 h, horizon 36, jump 24) with the coal ramp."""
    return _solve(tmp_path_factory, "ramp_roll", _ramp(0.002) + [
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]], chain=ROLL_CHAIN)


def test_roll_continues_from_previous_flow(rolling) -> None:
    """At every roll boundary inside a period the realized change stays within one step's ramp (0.12 units);
    the continuation roll's first-step row has no predecessor column (the previous roll's flow is on its
    RHS).  Across a period boundary (p2020 t0072 → p2025 t0001, a new year) there is no link.  Before: the
    first-step row tied the roll to its own look-ahead end, and the realized trajectory jumped by up to 0.222
    units at a boundary."""
    traj = _realized_flow(rolling)
    for (d0, t0, f0), (d1, t1, f1) in zip(traj, traj[1:]):
        if d0 == d1:
            assert abs(f1 - f0) <= 0.12 + TOL, (d0, t0, t1, f0, f1)
    steps = list(rolling.values())
    second = steps[1]
    assert second.flex_data.dt_ramp_carry is not None
    row = row_terms(build(second.flex_data), "ramp_sink_up_constraint", ARC)[
        f"ramp_sink_up_constraint[{ARC}p2020,t0025]"]
    assert set(row) == {f"v_flow[{ARC}p2020,t0025]"}
    carried = [f for d, t, f in traj if (d, t) == ("p2020", "t0024")][0]
    ub = row_bounds(build(second.flex_data), "ramp_sink_up_constraint", ARC)[
        f"ramp_sink_up_constraint[{ARC}p2020,t0025]"][1]
    assert ub == pytest.approx(0.12 + carried, abs=1e-9)
    fourth = steps[3]   # first roll of p2025: its predecessor p2020 t0072 is a different year
    assert fourth.flex_data.dt_ramp_carry is None
    assert f"ramp_sink_up_constraint[{ARC}p2025,t0001]" not in row_terms(
        build(fourth.flex_data), "ramp_sink_up_constraint", ARC)


def test_roll_without_look_ahead(tmp_path_factory) -> None:
    """Horizon = jump (no look-ahead): every roll still ramps from the previous roll's last step."""
    steps = _solve(tmp_path_factory, "ramp_roll_nolook", _ramp(0.002) + [
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(24.0, "float")],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]], chain=ROLL_CHAIN)
    traj = _realized_flow(steps)
    for (d0, t0, f0), (d1, t1, f1) in zip(traj, traj[1:]):
        if d0 == d1:
            assert abs(f1 - f0) <= 0.12 + TOL, (d0, t0, t1)
    carried = [s for s in steps.values() if s.flex_data.dt_ramp_carry is not None]
    assert len(carried) == 8   # 3 rolls per period, the first of each period has no adjacent predecessor


def test_roll_across_continuing_periods(tmp_path_factory) -> None:
    """Two continuing periods (p2020 = t0001–24, p2025 = t0025–48) solved in rolls of 18 h, jump 12: the roll
    that starts p2025 ramps from p2020 t0024 (the last realized step of the roll before, which also looked
    ahead into p2025), so the realized trajectory is ramp-feasible across the period boundary."""
    vals = _ramp(0.002) + TWO_PERIODS + [
        ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(18.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")]]
    steps = _solve(tmp_path_factory, "ramp_roll_2p", vals, entities=TWO_PERIODS_ENTS)
    traj = _realized_flow(steps)
    assert [(d, t) for d, t, _ in traj][23:25] == [("p2020", "t0024"), ("p2025", "t0025")]
    for (_, _, f0), (_, _, f1) in zip(traj, traj[1:]):
        assert abs(f1 - f0) <= 0.12 + TOL
    third = list(steps.values())[2]
    carry = utf8(third.flex_data.dt_ramp_carry)
    assert carry.select("d", "t", "d_previous", "t_previous").rows() == [("p2025", "t0025", "p2020", "t0024")]


def test_roll_carriers_take_the_last_realized_step_in_time(tmp_path_factory) -> None:
    """Review F3 / item 100: continuing periods p2020 (t0001–24) and a2025 (t0025–48; the name sorts before
    p2020), coal linear online with a ramp, the battery storage, rolls of 16 h with jump 10: roll 2 realizes
    p2020 t0011–t0020 … and roll 3 realizes p2020 t0021–t0024 + a2025 t0025–t0030.  The next roll's carriers are
    read at a2025 t0030 (the last realized step in time), not at the (d, t)-sorted last pair p2020 t0024: the ramp
    carry links a2025 t0031 to a2025 t0030 with that step's flow, the online carry is that step's v_online, the
    storage carry that step's state.  Before: (p2020, t0024) for all three, so roll 4 had no ramp row at its first
    step and started online and storage from the wrong step."""
    tp = [v for v in TWO_PERIODS if v[2] not in ("period_timeset", "realized_periods")] + [
        ["solve", "y2020_2day_dispatch", "period_timeset", b64(_pmap({"p2020": "ta", "a2025": "tb"}), "map")],
        ["solve", "y2020_2day_dispatch", "realized_periods",
         b64({"value_type": "str", "data": ["p2020", "a2025"]}, "array")]]
    vals = _ramp(0.002) + tp + [
        ["unit", PLANT, "startup_method", b64("linear", "str")],
        ["unit", PLANT, "min_load", b64(0.2, "float")],
        ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(16.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(10.0, "float")]]
    steps = list(_solve(tmp_path_factory, "ramp_span", vals, chain=("init", "west", "battery", "coal"),
                        entities=TWO_PERIODS_ENTS).values())
    spanning = [i for i, s in enumerate(steps)
                if set(utf8(s.flex_data.realized_dispatch).select(pl.first()).to_series().to_list())
                >= {"p2020", "a2025"}]
    assert spanning, "no roll realizes steps of both periods"
    i = spanning[0]
    prev, nxt = steps[i], steps[i + 1]
    carry = utf8(nxt.flex_data.dt_ramp_carry)
    assert carry.select("d", "t", "d_previous", "t_previous").rows() == [("a2025", "t0031", "a2025", "t0030")]
    last = ("a2025", "t0030")
    flow_last = [x for d, t, x in utf8(prev.solution.value("v_flow"))
                 .filter((pl.col("p") == PLANT) & (pl.col("sink") == "west")).select("d", "t", "value").rows()
                 if (d, t) == last][0]
    f_hat = utf8(nxt.flex_data.p_roll_continue_flow.frame)["value"].to_list()
    assert f_hat == [pytest.approx(flow_last)]
    on_last = [x for p, d, t, x in utf8(prev.solution.value("v_online_linear")).rows() if (p, d, t) == (PLANT, *last)]
    assert utf8(nxt.flex_data.p_roll_continue_online.frame).filter(pl.col("p") == PLANT)["value"].to_list() == [
        pytest.approx(on_last[0])]
    us = float(utf8(prev.flex_data.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    st_last = [x for n, d, t, x in utf8(prev.solution.value("v_state")).rows() if (n, d, t) == ("battery", *last)]
    assert utf8(nxt.flex_data.p_roll_continue_state.frame).filter(pl.col("n") == "battery")["value"].to_list() == [
        pytest.approx(st_last[0] * us)]
    assert f"ramp_sink_up_constraint[{ARC}a2025,t0031]" in row_terms(build(nxt.flex_data), "ramp_sink_up_constraint",
                                                                      ARC)


def test_roll_carry_needs_adjacency() -> None:
    """ramp_roll_carry: the carry applies where the complete solve's predecessor of a period's first step is
    the carried step and adjacent; not before a gap; forecast branches starting at the anchor's first step
    take it too, other periods do not."""
    from flextool.engine.preprocess.params._ramp_links import ramp_roll_carry

    dt = pl.DataFrame({"d": ["p1", "p1", "b1", "b1", "p2"], "t": ["t0005", "t0006", "t0005", "t0006", "t0001"]})
    carry = pl.DataFrame({"process": ["u"], "source": ["s"], "sink": ["k"], "period": ["p1"], "step": ["t0004"],
                          "duration": [1.0], "value": [0.3]})

    def link(prev_t, jump):
        return pl.DataFrame({"period": ["p1", "p2"], "time": ["t0005", "t0001"],
                             "previous_period": ["p1", "p1"], "previous_within_solve": [prev_t, "t0006"],
                             "jump": [str(jump), "-5"]})

    step = carry.select("period", "step", "duration")
    pb = pl.DataFrame({"period": ["p1", "p1"], "branch": ["p1", "b1"]})
    rows, flows = ramp_roll_carry(dt, step, link("t0004", 1), pb, flows=carry)
    assert sorted(rows.select("d", "t").rows()) == [("b1", "t0005"), ("p1", "t0005")]
    assert flows["value"].to_list() == [0.3, 0.3]
    rows, flows = ramp_roll_carry(dt, step, link("t0004", 1), pb)                    # no ramp limits: link only
    assert sorted(rows.select("d", "t").rows()) == [("b1", "t0005"), ("p1", "t0005")] and flows is None
    assert ramp_roll_carry(dt, step, link("t0004", 5), None) == (None, None)        # gap before t0005
    assert ramp_roll_carry(dt, step, link("t0003", 1), None) == (None, None)        # not the carried step
    assert ramp_roll_carry(dt, None, link("t0004", 1), None) == (None, None)        # first roll
    # a representative-block start never takes the carry (review F5)
    rp = pl.DataFrame({"d": ["p1", "b1"], "t": ["t0005", "t0005"]})
    assert ramp_roll_carry(dt, step, link("t0004", 1), pb, flows=carry, rp_block_first=rp) == (None, None)


# ---------------------------------------------------------------------------
# ramp output


def _write_csv(tmp_path_factory, alt, values, chain, entities=None) -> tuple[dict, "pl.DataFrame", object]:
    """Solve, write the CSV outputs, return (steps, unit_ramp__outputNode__dt, unit_ramp__inputNode__dt)."""
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities)
    work = tmp_path_factory.mktemp(f"{alt}_out")
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
    last = list(steps.values())[-1]
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  solution=last.solution, solve_name=last.solve_name,
                  solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                  flex_data_provider=last.flex_data_provider)
    out_dir = work / "output_csv" / scen
    frames = []
    for name in ("unit_ramp__outputNode__dt.csv", "unit_ramp__inputNode__dt.csv"):
        path = out_dir / name
        frames.append(_read_dt_csv(path) if path.exists() else None)
    return steps, frames[0], frames[1]


def _read_dt_csv(path) -> dict:
    """``{(unit, node): {(period, time): value or None}}`` of a two-header-row ``*__dt.csv`` output."""
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    cols = list(zip(rows[0][3:], rows[1][3:]))
    out: dict = {c: {} for c in cols}
    for r in rows[2:]:
        for c, cell in zip(cols, r[3:]):
            out[c][(r[1], r[2])] = float(cell) if cell != "" else None
    return out


def _flow_mw(steps, arc) -> dict[tuple[str, str], float]:
    out = {}
    for s in steps.values():
        real = utf8(s.flex_data.realized_dispatch.rename({"period": "d", "step": "t"})
                    if "period" in s.flex_data.realized_dispatch.columns else s.flex_data.realized_dispatch)
        us = float(utf8(s.flex_data.p_unitsize.frame).filter(pl.col("p") == arc[0])["value"][0])
        v = (utf8(s.solution.value("v_flow"))
             .filter((pl.col("p") == arc[0]) & (pl.col("source") == arc[1]) & (pl.col("sink") == arc[2]))
             .join(real.select("d", "t"), on=["d", "t"], how="semi"))
        out.update({(d, t): x * us for d, t, x in v.select("d", "t", "value").iter_rows()})
    return out


def test_ramp_output_empty_without_predecessor(tmp_path_factory) -> None:
    """Single solve: the reported ramp is empty at t0001 (no predecessor; it was 0 from the wrap) and the flow
    change elsewhere."""
    steps, out, _ = _write_csv(tmp_path_factory, "ro_single", _ramp(0.001, dirs=("up",)),
                               ("init", "west", "coal"))
    col = out[(PLANT, "west")]
    assert col[("p2020", "t0001")] is None
    f = _flow_mw(steps, (PLANT, "coal_market", "west"))
    for t in ("t0002", "t0004", "t0048"):
        prev = f"t{int(t[1:]) - 1:04d}"
        assert col[("p2020", t)] == pytest.approx(f[("p2020", t)] - f[("p2020", prev)], abs=1e-4)


def test_ramp_output_at_roll_boundary_on_direct_input(tmp_path_factory) -> None:
    """Rolling (horizon 36, jump 24) with an e_boiler-style direct unit (west → heat, efficiency 0.5) ramp on
    its input: at a roll's first step the reported input ramp is the change of the reported input flow from
    the previous roll's last realized step (the input column is v_flow × unitsize / efficiency, so the
    carried v_flow alone would be wrong by the slope)."""
    eb_ents = [["unit", "e_boiler", None], ["unit__inputNode", ["e_boiler", "west"], None],
               ["unit__outputNode", ["e_boiler", "heat"], None]]
    vals = [["unit", "e_boiler", "efficiency", b64(0.5, "float")],
            ["unit", "e_boiler", "existing", b64(100.0, "float")],
            ["unit", "e_boiler", "conversion_method", b64("constant_efficiency", "str")],
            ["node", "heat", "penalty_up", b64(1000.0, "float")],
            ["node", "heat", "inflow", _tmap({f"t{i:04d}": -(20.0 + 60.0 * (i % 3 == 0)) for i in range(1, 73)})],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]]
    vals += _ramp(0.002, cls="unit__inputNode", rel=["e_boiler", "west"])
    steps, _, inp = _write_csv(tmp_path_factory, "ro_roll_in", vals, ROLL_CHAIN[:-2] + ("heat", "fullYear",
                                                                                      "dispatch_fullYear_roll"),
                               entities=eb_ents)
    col = inp[("e_boiler", "west")]
    f = _flow_mw(steps, ("e_boiler", "west", "heat"))   # output-measured MW; reported input = f / 0.5
    for t, prev in (("t0025", "t0024"), ("t0049", "t0048"), ("t0026", "t0025")):
        assert col[("p2020", t)] == pytest.approx((f[("p2020", t)] - f[("p2020", prev)]) / 0.5, abs=1e-4)
    assert abs(f[("p2020", "t0025")] - f[("p2020", "t0024")]) > 1.0   # the boundary carries a real change
    assert col[("p2020", "t0001")] is None


def test_stochastic_roll_branches_take_the_carry(tmp_path_factory) -> None:
    """Rolling stochastic 1-week fixture (horizon 96, jump 24, forecast branches over the whole window): on
    every continuation roll the realized period and each forecast branch ramp from the previous roll's realized
    flow, at their common first step, without a predecessor column, and the carried flow is on the RHS:
    ub = 0.12 + f̂.  A lower-limit profile keeps coal at ≥ 0.1 of capacity everywhere, so f̂ > 0 (the realized
    trunk of this fixture otherwise dispatches nothing; review F7)."""
    from tests.engine._unit_db import FLEXTOOL_ROOT

    url, scen = make_db(tmp_path_factory, alt="ramp_stroll",
                        values=_ramp(0.002, rel=[PLANT, "demand_node"]) + [
                            ["profile", "coal_floor", "profile", b64(0.1, "float")],
                            ["unit__node__profile", [PLANT, "demand_node", "coal_floor"], "profile_method",
                             b64("lower_limit", "str")]],
                        entities=[["profile", "coal_floor", None],
                                  ["unit__node__profile", [PLANT, "demand_node", "coal_floor"], None]],
                        base_chain=["init", "base", "system", "1week_rolling"],
                        fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json")
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) > 2
    assert steps[0].flex_data.dt_ramp_carry is None
    arc = f"{PLANT},coal_market,demand_node,"
    for s in steps[1:]:
        fd = s.flex_data
        firsts = (utf8(fd.dt).with_row_index("i").group_by("d", maintain_order=True)
                  .agg(pl.col("t").sort_by("i").first()))
        carry = utf8(fd.dt_ramp_carry)
        assert sorted(carry.select("d", "t").rows()) == sorted(firsts.rows())
        assert carry["d_previous"].unique().to_list() == ["period1"]
        f_hat = utf8(fd.p_roll_continue_flow.frame)["value"].unique().to_list()
        assert len(f_hat) == 1 and f_hat[0] >= 0.1 - 1e-9
        pb = build(fd)
        rows = row_terms(pb, "ramp_sink_up_constraint", arc)
        bounds = row_bounds(pb, "ramp_sink_up_constraint", arc)
        for d, t in firsts.rows():
            row = f"ramp_sink_up_constraint[{arc}{d},{t}]"
            assert set(rows[row]) <= {f"v_flow[{arc}{d},{t}]"}
            assert bounds[row][1] == pytest.approx(0.12 + f_hat[0], abs=1e-9)


def test_ramp_output_at_roll_boundary_without_ramp_limits(tmp_path_factory) -> None:
    """Rolling with horizon == jump (24 h) and no ramp limit anywhere: the reported ramp at a roll's first step
    is the realized change from the previous roll's last step (review F4: the boundary link came only with a ramp
    carry, so the cell was empty)."""
    vals = [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(24.0, "float")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]]
    steps, out, _ = _write_csv(tmp_path_factory, "ro_noramp", vals, ROLL_CHAIN)
    col = out[(PLANT, "west")]
    f = _flow_mw(steps, (PLANT, "coal_market", "west"))
    for t, prev in (("t0025", "t0024"), ("t0049", "t0048")):
        assert col[("p2020", t)] == pytest.approx(f[("p2020", t)] - f[("p2020", prev)], abs=1e-4)
    assert col[("p2020", "t0001")] is None and col[("p2025", "t0001")] is None
