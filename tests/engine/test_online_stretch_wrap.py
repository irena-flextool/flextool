"""The online commitment cycles within its stretch of the timeline (``params._ramp_links.online_stretch_wrap``).

A step without a ramp predecessor (``dt_ramp_link``) — the first step of a solve or of a period that does not
continue the previous one, the step after a gap in the timeline, a representative-block start — takes as its online
predecessor the last step of its own stretch of the period (a run of steps along ``dt_ramp_link``).  The commitment
never links across a gap, a representative-block start or into another period, the same as ramp limits and the
minimum up / down time look-back.

Unit cases on hand-built frames, and one relational solve: a timeset of two 24-h blocks with a gap between them
gives the same commitment as solving each block alone.  Fixtures are built from ``tests.json`` under the pytest tmp
dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.preprocess.params._ramp_links import online_stretch_wrap, ramp_link_fields
from tests.engine._unit_db import PLANT, PLANT_CHAIN, TIMESTEPS, b64, make_db, run, time_map, uc, utf8


def _link(rows):
    return pl.DataFrame(rows, schema=["d", "t", "d_previous", "t_previous"], orient="row")


def _dt(rows):
    return pl.DataFrame(rows, schema=["d", "t"], orient="row")


def _chain(d, ts):
    return [(d, b, d, a) for a, b in zip(ts, ts[1:])]


def test_each_block_after_a_gap_wraps_to_its_own_end():
    """Two blocks t1-t3 and t5-t7 of one period with a gap: each block start wraps to its own block end."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3", "t5", "t6", "t7")])
    link = _link(_chain("p", ["t1", "t2", "t3"]) + _chain("p", ["t5", "t6", "t7"]))
    assert online_stretch_wrap(dt, link).rows() == [("p", "t1", "p", "t3"), ("p", "t5", "p", "t7")]


def test_a_continuing_period_wraps_within_its_own_period():
    """p2 continues p1 (a cross-period link to p2's first step): p1's first step wraps to p1's last step, not to
    p2's end; p2's first step has a ramp predecessor and no wrap."""
    dt = _dt([("p1", "t1"), ("p1", "t2"), ("p2", "t3"), ("p2", "t4")])
    link = _link(_chain("p1", ["t1", "t2"]) + [("p2", "t3", "p1", "t2")] + _chain("p2", ["t3", "t4"]))
    assert online_stretch_wrap(dt, link).rows() == [("p1", "t1", "p1", "t2")]


def test_adjacent_representative_blocks_cycle_each_on_their_own():
    """Two adjacent representative blocks t1-t2 and t3-t4: the second block's start has no ramp link (it represents
    another base period), so each block cycles on its own."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3", "t4")])
    link = _link(_chain("p", ["t1", "t2"]) + _chain("p", ["t3", "t4"]))
    assert online_stretch_wrap(dt, link).rows() == [("p", "t1", "p", "t2"), ("p", "t3", "p", "t4")]


def test_pre_reveal_steps_wrap_to_their_own_last_step():
    """A mid-horizon fan: the pre-reveal steps t1-t3 of p, then branches b1 / b2 linked from (p, t3).  The first
    step of p wraps to t3 whatever the number of branches; the branch starts have ramp predecessors."""
    dt = _dt([("p", "t1"), ("p", "t2"), ("p", "t3"), ("b1", "t4"), ("b1", "t5"), ("b2", "t4"), ("b2", "t5")])
    link = _link(_chain("p", ["t1", "t2", "t3"]) + [("b1", "t4", "p", "t3"), ("b2", "t4", "p", "t3")]
                 + _chain("b1", ["t4", "t5"]) + _chain("b2", ["t4", "t5"]))
    assert online_stretch_wrap(dt, link).rows() == [("p", "t1", "p", "t3")]


def test_first_period_fan_branches_wrap_to_their_own_end():
    """A fan from the first step: each branch period wraps to its own last step; a one-step stretch to itself."""
    dt = _dt([("b1", "t1"), ("b1", "t2"), ("b2", "t1"), ("b2", "t2"), ("b3", "t1")])
    link = _link(_chain("b1", ["t1", "t2"]) + _chain("b2", ["t1", "t2"]))
    assert online_stretch_wrap(dt, link).rows() == [("b1", "t1", "b1", "t2"), ("b2", "t1", "b2", "t2"),
                                                    ("b3", "t1", "b3", "t1")]


def test_a_same_period_link_between_steps_that_are_not_consecutive_raises():
    dt = _dt([("p", t) for t in ("t1", "t2", "t3")])
    with pytest.raises(ValueError, match="not consecutive"):
        online_stretch_wrap(dt, _link([("p", "t3", "p", "t1")]))


def test_no_links_no_wrap():
    assert online_stretch_wrap(_dt([("p", "t1")]), None) is None
    assert ramp_link_fields(None, None) == {"dt_ramp_link": None, "dt_online_wrap": None}


# ---------------------------------------------------------------------------
# A gap model == the blocks solved alone


HIGH = 400.0          # MW, within the 500 MW coal_plant and above its 200 MW minimum load
BLOCK_1 = TIMESTEPS[:24]          # t0001-t0024
BLOCK_2 = TIMESTEPS[48:72]        # t0049-t0072


def _timeset(blocks: list[tuple[str, float]]) -> list:
    return ["timeset", "2day", "timeset_duration",
            b64({"index_type": "str", "index_name": "time", "data": [[s, d] for s, d in blocks]}, "map")]


def _values(blocks):
    # Demand: block 1 high then 0, block 2 0 then high.
    demand = {t: -HIGH for t in BLOCK_1[:12] + BLOCK_2[12:]}
    demand.update({t: 0.0 for t in BLOCK_1[12:] + BLOCK_2[:12]})
    return uc(PLANT, 0.4, "binary") + [
        ["unit", PLANT, "startup_cost", b64(50.0, "float")],
        ["node", "west", "inflow", time_map(demand, 0.0)],
        _timeset(blocks)]


SOLVE = "y2020_2day_dispatch"
ROLLING = [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
           ["solve", SOLVE, "rolling_solve_horizon", b64(24.0, "float")],
           ["solve", SOLVE, "rolling_solve_jump", b64(24.0, "float")]]
TWO_PERIODS_ENTS = [["timeset", "ta", None], ["timeset", "tb", None]]


def _two_periods() -> list:
    """p2020 on block 1 (t0001-t0024) and p2025 on block 2 (t0049-t0072) of one timeline: p2025 does not continue
    p2020 (there is a gap between them)."""
    def pmap(data, index_name="period"):
        return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in data.items()]}
    return [["timeset", "ta", "timeline", b64("y2020", "str")],
            ["timeset", "tb", "timeline", b64("y2020", "str")],
            ["timeset", "ta", "timeset_duration", b64(pmap({"t0001": 24.0}, "time"), "map")],
            ["timeset", "tb", "timeset_duration", b64(pmap({"t0049": 24.0}, "time"), "map")],
            ["solve", SOLVE, "period_timeset", b64(pmap({"p2020": "ta", "p2025": "tb"}), "map")],
            ["solve", SOLVE, "realized_periods", b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")]]


def _run(tmp_path_factory, name, values, entities=None):
    url, scen = make_db(tmp_path_factory, alt=name, values=values, base_chain=PLANT_CHAIN, entities=entities)
    return list(run(tmp_path_factory, url, scen).values())


@pytest.fixture(scope="module")
def gap_runs(tmp_path_factory):
    out = {}
    for name, blocks in (("gap_two_blocks", [("t0001", 24.0), ("t0049", 24.0)]),
                         ("gap_block_one", [("t0001", 24.0)]),
                         ("gap_block_two", [("t0049", 24.0)])):
        steps = _run(tmp_path_factory, name, _values(blocks))
        assert len(steps) == 1, steps
        out[name] = steps[0]
    out["gap_rolling"] = _run(tmp_path_factory, "gap_rolling",
                              _values([("t0001", 24.0), ("t0049", 24.0)]) + ROLLING)
    return out


def _uc(step, var, realized=False):
    v = utf8(step.solution.value(f"{var}_integer")).filter(pl.col("p") == PLANT)
    if realized:
        real = utf8(step.flex_data.realized_dispatch.rename({"period": "d", "step": "t"}))
        v = v.join(real.select("d", "t"), on=["d", "t"], how="semi")
    value = pl.col("value").round(6)
    return v.select("d", "t", pl.when(value == 0.0).then(0.0).otherwise(value).alias("value")).sort("d", "t")


def _rolled(steps, var):
    return pl.concat([_uc(s, var, realized=True) for s in steps]).sort("d", "t")


@pytest.mark.solver
def test_a_gap_model_commits_as_its_blocks_solved_alone(gap_runs):
    """The blocks of a gap model each cycle on their own, so the commitment equals solving each block alone.

    Hand derivation (one integer unit, min load 200 MW, start-up cost 50, demand 400 MW where 'high'): block 1
    is high in t0001-t0012 and 0 after, so the unit is online in t0001-t0012 and offline at its end t0024; its first
    step wraps to t0024 (offline), so it starts up at t0001.  Block 2 is 0 in t0049-t0060 and high after: offline
    at t0049 (it wraps to t0072, online: a shut-down, no cost) and a start-up at t0061.  Two start-ups, one per
    block — the same as each block alone.  (Before, t0001 wrapped to t0072, online, so block 1 had no start-up, and
    t0049 followed t0024 across the gap.)"""
    gap, one, two = gap_runs["gap_two_blocks"], gap_runs["gap_block_one"], gap_runs["gap_block_two"]
    for alone in (one, two):
        fd = alone.flex_data
        assert fd.p_roll_continue_online is None and fd.dt_online_carry is None     # a block alone takes no carry
    for var in ("v_online", "v_startup", "v_shutdown"):
        assert _uc(gap, var).equals(pl.concat([_uc(one, var), _uc(two, var)]).sort("d", "t")), var
    su = _uc(gap, "v_startup").filter(pl.col("value") > 0.5)
    assert su.select("t").to_series().to_list() == ["t0001", "t0061"]
    assert su["value"].to_list() == [1.0, 1.0]
    # The wrap is the one derived from the final ramp links (every setter of dt_ramp_link sets it with them).
    fd = gap.flex_data
    assert fd.dt_online_wrap.equals(online_stretch_wrap(fd.dt, fd.dt_ramp_link))
    assert fd.dt_online_wrap.rows() == [("p2020", "t0001", "p2020", "t0024"), ("p2020", "t0049", "p2020", "t0072")]


@pytest.mark.solver
def test_a_roll_after_a_gap_takes_no_carry(gap_runs):
    """The gap model rolled one block per roll (horizon = jump = 24 h): the second roll starts after the gap, so
    it does not continue the first roll's commitment (no ``dt_online_carry``; before, the carry fell back onto its
    first step), and the rolled commitment equals the blocks solved alone.  Hand derivation as above: roll 1 ends
    offline at t0024, but roll 2's first step t0049 wraps to t0072 (online), a shut-down there; the carried
    offline state would have given no shut-down at t0049."""
    first, second = gap_runs["gap_rolling"]
    assert first.flex_data.p_roll_continue_online is None
    assert second.flex_data.p_roll_continue_online is not None      # the first roll handed its commitment on
    assert second.flex_data.dt_online_carry is None                 # ... and the gap stops it
    assert second.flex_data.dt_online_wrap.rows() == [("p2020", "t0049", "p2020", "t0072")]
    one, two = gap_runs["gap_block_one"], gap_runs["gap_block_two"]
    for var in ("v_online", "v_startup", "v_shutdown"):
        assert _rolled([first, second], var).equals(pl.concat([_uc(one, var), _uc(two, var)]).sort("d", "t")), var
    sd = _rolled([first, second], "v_shutdown").filter(pl.col("value") > 0.5)
    assert sd.select("t").to_series().to_list() == ["t0013", "t0049"]


@pytest.fixture(scope="module")
def period_runs(tmp_path_factory):
    # Demand: p2020 (t0001-t0024) high throughout; p2025 (t0049-t0072) high in t0049-t0060, 0 after.
    demand = {t: -HIGH for t in BLOCK_1 + BLOCK_2[:12]}
    demand.update({t: 0.0 for t in BLOCK_2[12:]})
    values = uc(PLANT, 0.4, "binary") + [["unit", PLANT, "startup_cost", b64(50.0, "float")],
                                         ["node", "west", "inflow", time_map(demand, 0.0)]] + _two_periods()
    single = _run(tmp_path_factory, "period_single", values, TWO_PERIODS_ENTS)
    rolled = _run(tmp_path_factory, "period_rolling", values + ROLLING, TWO_PERIODS_ENTS)
    return single, rolled


@pytest.mark.solver
def test_a_roll_into_a_period_that_does_not_continue_takes_no_carry(period_runs):
    """Two periods with a gap between them, rolled one period per roll: the roll into p2025 takes no carry, so the
    rolled commitment equals the single solve.  Hand derivation: p2020 is high throughout (online all along, its
    first step wraps to its own online end: no start-up); p2025 is high first and 0 at its end t0072, so its first
    step t0049 wraps to an offline state and starts up — in the single solve and in the roll.  Before, the roll
    carried p2020's online end into t0049 and charged no start-up there."""
    (single,), (first, second) = period_runs
    assert second.flex_data.dt_online_carry is None
    for var in ("v_online", "v_startup", "v_shutdown"):
        assert _rolled([first, second], var).equals(_uc(single, var)), var
    su = _uc(single, "v_startup").filter(pl.col("value") > 0.5)
    assert su.select("d", "t").rows() == [("p2025", "t0049")]
