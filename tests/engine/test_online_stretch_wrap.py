"""The online commitment cycles within its stretch of the timeline (``params._ramp_links.online_stretch_wrap``), or
starts free (``online_free_start_cells``).

A step without a ramp predecessor (``dt_ramp_link``) is a stretch start.  At the first step of a solve, of a
forecast branch starting there and of a period that does not continue the previous one (a period with one stretch),
the units online start free: no predecessor, no start-up or shut-down there.  The other stretch starts (the step
after a gap in the timeline, a representative-block start) take as their online predecessor the last step of their
own stretch of the period: the samples cycle.  The commitment never links across a gap, a representative-block
start or into another period, the same as ramp limits and the minimum up / down time look-back.

Unit cases on hand-built frames, and relational solves: a timeset of two 24-h blocks with a gap between them (the
blocks cycle), two periods with a gap rolled or solved at once (each period starts free), and a period start inside
a rolling window.  Fixtures are built from ``tests.json`` under the pytest tmp dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.preprocess.params._ramp_links import (
    online_free_start_cells, online_stretch_wrap, ramp_link_fields,
)
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


def _free(dt, link, rp_block_first=None):
    return online_free_start_cells(dt, online_stretch_wrap(dt, link), rp_block_first).rows()


def test_free_start_at_the_solve_start_and_at_a_period_that_does_not_continue():
    """Two continuous periods on the same timeset, p2 not continuing p1: the first step of each period (by solve
    order, not by step name) starts free."""
    dt = _dt([("p1", "t1"), ("p1", "t2"), ("p2", "t1"), ("p2", "t2")])
    link = _link(_chain("p1", ["t1", "t2"]) + _chain("p2", ["t1", "t2"]))
    assert _free(dt, link) == [("p1", "t1"), ("p2", "t1")]


def test_no_free_start_in_a_continuing_period():
    """p2 continues p1: only the solve's first step starts free."""
    dt = _dt([("p1", "t1"), ("p1", "t2"), ("p2", "t3"), ("p2", "t4")])
    link = _link(_chain("p1", ["t1", "t2"]) + [("p2", "t3", "p1", "t2")] + _chain("p2", ["t3", "t4"]))
    assert _free(dt, link) == [("p1", "t1")]


def test_a_sampled_period_cycles_at_every_block_also_the_first():
    """A period of two timeblocks with a gap is a sample: no free start, the first block included."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3", "t5", "t6", "t7")])
    link = _link(_chain("p", ["t1", "t2", "t3"]) + _chain("p", ["t5", "t6", "t7"]))
    assert _free(dt, link) == []


def test_a_representative_block_at_the_solve_start_cycles():
    """A period of one representative block: its start is in rp_block_first and cycles."""
    dt = _dt([("p", "t1"), ("p", "t2")])
    link = _link(_chain("p", ["t1", "t2"]))
    assert _free(dt, link, _dt([("p", "t1")])) == []


def test_free_start_in_every_branch_of_a_fan_at_the_solve_start():
    """A fan from the first step: every branch starts free; a mid-horizon fan's branches have a ramp predecessor."""
    dt = _dt([("b1", "t1"), ("b1", "t2"), ("b2", "t1"), ("b2", "t2")])
    link = _link(_chain("b1", ["t1", "t2"]) + _chain("b2", ["t1", "t2"]))
    assert _free(dt, link) == [("b1", "t1"), ("b2", "t1")]
    dt = _dt([("p", "t1"), ("p", "t2"), ("b1", "t3"), ("b2", "t3")])
    link = _link(_chain("p", ["t1", "t2"]) + [("b1", "t3", "p", "t2"), ("b2", "t3", "p", "t2")])
    assert _free(dt, link) == [("p", "t1")]



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
def test_a_gap_model_cycles_within_each_block(gap_runs):
    """The blocks of a gap model each cycle on their own; a block solved alone starts free.

    Hand derivation (one integer unit, min load 200 MW, start-up cost 50, demand 400 MW where 'high'): block 1
    is high in t0001-t0012 and 0 after, so the unit is online in t0001-t0012 and offline at its end t0024; its first
    step wraps to t0024 (offline), so it starts up at t0001.  Block 2 is 0 in t0049-t0060 and high after: offline
    at t0049 (it wraps to t0072, online: a shut-down, no cost) and a start-up at t0061.  Two start-ups, one per
    block, each on the block's own cycle (never across the gap).
    Each block solved alone is a single continuous period: its first step starts free (no start-up or shut-down
    there).  So the units online are the same as in the gap model; block 1 alone has no start-up at t0001 and block
    2 alone no shut-down at t0049, and block 2 alone still starts up at t0061."""
    gap, one, two = gap_runs["gap_two_blocks"], gap_runs["gap_block_one"], gap_runs["gap_block_two"]
    for alone in (one, two):
        fd = alone.flex_data
        assert fd.p_roll_continue_online is None and fd.dt_online_carry is None     # a block alone takes no carry
    assert _uc(gap, "v_online").equals(pl.concat([_uc(one, "v_online"), _uc(two, "v_online")]).sort("d", "t"))
    su = _uc(gap, "v_startup").filter(pl.col("value") > 0.5)
    assert su.select("t").to_series().to_list() == ["t0001", "t0061"]
    assert su["value"].to_list() == [1.0, 1.0]
    sd = _uc(gap, "v_shutdown").filter(pl.col("value") > 0.5)
    assert sd.select("t").to_series().to_list() == ["t0013", "t0049"]
    alone_su = pl.concat([_uc(one, "v_startup"), _uc(two, "v_startup")]).filter(pl.col("value") > 0.5)
    assert alone_su.select("t").to_series().to_list() == ["t0061"]
    alone_sd = pl.concat([_uc(one, "v_shutdown"), _uc(two, "v_shutdown")]).filter(pl.col("value") > 0.5)
    assert alone_sd.select("t").to_series().to_list() == ["t0013"]
    # The wrap is the one derived from the final ramp links (every setter of dt_ramp_link sets it with them).
    fd = gap.flex_data
    assert fd.dt_online_wrap.equals(online_stretch_wrap(fd.dt, fd.dt_ramp_link))
    assert fd.dt_online_wrap.rows() == [("p2020", "t0001", "p2020", "t0024"), ("p2020", "t0049", "p2020", "t0072")]
    assert online_free_start_cells(fd.dt, fd.dt_online_wrap, fd.rp_block_first).height == 0


def test_rolling_over_a_gap_is_an_error(tmp_path_factory):
    """The gap model rolled one block per roll: a rolling solve over a period with two timeblocks is a configuration
    error (owner decision, Phase 5: rolling solves need a continuous timeline).  Before, the second roll started after
    the gap and took no commitment carry; that case can no longer be set up."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="runs over a timeline with gaps"):
        _run(tmp_path_factory, "gap_rolling", _values([("t0001", 24.0), ("t0049", 24.0)]) + ROLLING)


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
    rolled commitment equals the single solve.  Hand derivation: p2020 is high throughout (online all along); p2025
    is high first and 0 at its end t0072.  The first step of each period starts free (each is a continuous period
    that does not continue the previous one), in the single solve and in the roll: online at t0001 and t0049 with no
    start-up, a shut-down at t0061.  (Before, t0049 wrapped to its offline end t0072 and started up there.)"""
    (single,), (first, second) = period_runs
    assert second.flex_data.dt_online_carry is None
    for var in ("v_online", "v_startup", "v_shutdown"):
        assert _rolled([first, second], var).equals(_uc(single, var)), var
    assert _uc(single, "v_startup").filter(pl.col("value") > 0.5).height == 0
    sd = _uc(single, "v_shutdown").filter(pl.col("value") > 0.5)
    assert sd.select("d", "t").rows() == [("p2025", "t0061")]
    on = _uc(single, "v_online").filter(pl.col("t").is_in(["t0001", "t0049"]))
    assert on["value"].to_list() == [1.0, 1.0]


ROLLING_36_24 = [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
                 ["solve", SOLVE, "rolling_solve_horizon", b64(36.0, "float")],
                 ["solve", SOLVE, "rolling_solve_jump", b64(24.0, "float")]]


@pytest.mark.solver
def test_a_period_start_inside_a_rolling_window_starts_free(tmp_path_factory):
    """A roll whose window holds the end of p2020 and the start of p2025 (which does not continue p2020): p2025's
    first step starts free, as it would at the start of a roll or in a single solve.

    Hand derivation (the period_runs demand; horizon 36 h, jump 24 h): roll 1 holds p2020 t0001-t0024 and the
    look-ahead p2025 t0049-t0060, all high (400 MW, only the coal plant serves it), so the unit is online at every
    step: v_online[p2025,t0049] = 1, and a free start has v_startup = 0 there.  The step keeps its stretch wrap row
    (to t0060) in dt_online_wrap, but it is a free start, so the wrap is not used: where the jump ends inside p2025,
    the predecessor of t0049 then does not depend on the window's end.
    """
    demand = {t: -HIGH for t in BLOCK_1 + BLOCK_2[:12]}
    demand.update({t: 0.0 for t in BLOCK_2[12:]})
    values = uc(PLANT, 0.4, "binary") + [["unit", PLANT, "startup_cost", b64(50.0, "float")],
                                         ["node", "west", "inflow", time_map(demand, 0.0)]] + _two_periods()
    first = _run(tmp_path_factory, "period_start_in_window", values + ROLLING_36_24, TWO_PERIODS_ENTS)[0]
    at = (pl.col("d") == "p2025") & (pl.col("t") == "t0049")
    assert _uc(first, "v_online").filter(at)["value"].to_list() == [1.0]
    assert _uc(first, "v_startup").filter(at)["value"].to_list() == [0.0]
    fd = first.flex_data
    assert ("p2025", "t0049", "p2025", "t0060") in utf8(fd.dt_online_wrap).rows()
    assert online_free_start_cells(fd.dt, fd.dt_online_wrap, fd.rp_block_first).rows() == [
        ("p2020", "t0001"), ("p2025", "t0049")]
