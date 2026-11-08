"""Solve-level ``new_stepduration`` keeps the timeline's gaps and every timeset's steps (#101, #103).

The solve's aggregated timeline partitions the WHOLE original timeline: it is cut at every block start and end of
the solve's timesets and each piece (block or gap) is aggregated from its first step.  So two steps follow each other
in the solve exactly when they are adjacent on the original timeline (storage state chains, intraperiod blocks, ramp
and roll links, representative blocks), and two timesets on one timeline both keep their steps (two periods, a
stochastic trunk).

Fixtures: ``tests.json`` (5weeks = t0001×4, t0009×4, t0013×4, t0025×4, t0037×4 of the 72-step y2020 timeline;
t0009 and t0013 touch) and ``stoch_two_period_midhorizon_nreal0.json`` (p2035 = ts1 t0001–3, p2040 branches = ts2
t0004–6), plus override alternatives built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._timeline import TimelineConfig, get_active_time, make_step_jump
from tests.engine._lp_rows import coef, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, run, utf8

pytestmark = pytest.mark.solver

BATTERY = ["init", "west", "wind", "battery"]
STEP_2H_5W = [["solve", "y2020_5week", "new_stepduration", b64(2.0, "float")]]
MIDHORIZON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_midhorizon_nreal0.json"


def _pmap(values: dict, index_name: str = "period") -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}, "map")


TWO_PERIODS_ENTS = [["timeset", "ta", None], ["timeset", "tb", None]]
TWO_PERIODS_2H = [
    ["timeset", "ta", "timeline", b64("y2020", "str")],
    ["timeset", "tb", "timeline", b64("y2020", "str")],
    ["timeset", "ta", "timeset_duration", _pmap({"t0001": 24.0}, "time")],
    ["timeset", "tb", "timeset_duration", _pmap({"t0025": 24.0}, "time")],
    ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "ta", "p2025": "tb"})],
    ["solve", "y2020_2day_dispatch", "realized_periods",
     b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")],
    ["solve", "y2020_2day_dispatch", "new_stepduration", b64(2.0, "float")],
]


def _run(tmp_path_factory, alt, values, chain, entities=None, fixture_json=None) -> dict:
    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities, **kw)
    return run(tmp_path_factory, url, scen)


def _last(steps: dict):
    return list(steps.values())[-1]


def _dt(fd) -> list[tuple[str, str]]:
    return [tuple(r) for r in utf8(fd.dt).select("d", "t").iter_rows()]


def _csv_step_previous(step) -> pl.DataFrame:
    from flextool.engine.preprocess.flex_data import _read_step_previous

    return utf8(_read_step_previous(Path("solve_data") / "step_previous.csv", provider=step.flex_data_provider))


_LAG_COLS = ["d", "t", "t_previous", "t_previous_within_timeset", "d_previous", "t_previous_within_solve"]


# ---------------------------------------------------------------------------
# the aggregated timeline itself


def _config(blocks: dict[str, list], solves: dict[str, list], steps: dict[str, float | None],
            n_steps: int = 72, durations: list[float] | None = None) -> tuple[TimelineConfig, SimpleNamespace]:
    """Timeline ``T`` of ``n_steps`` 1 h steps, or of the steps in ``durations``."""
    base = [(f"t{i:04d}", "1") for i in range(1, n_steps + 1)] if durations is None else [
        (f"t{i:04d}", str(d)) for i, d in enumerate(durations, start=1)]
    tl = TimelineConfig(
        timelines=defaultdict(list, {"T": base}),
        timesets=list(blocks), timesets__timeline=defaultdict(str, {ts: "T" for ts in blocks}),
        timeset_durations=defaultdict(list, blocks), new_step_durations=steps)
    sc = SimpleNamespace(timesets_used_by_solves=solves, model_solve={"m": list(solves)}, contains_solves={})
    return tl, sc


FIVE_WEEKS = [("t0001", 4), ("t0009", 4), ("t0013", 4), ("t0025", 4), ("t0037", 4)]


def test_gaps_stay_gaps() -> None:
    """5weeks + Δ 2: the solve's timeline lists every 2-h chunk of the 72-h timeline (gaps included), each block
    has two chunks, and make_step_jump's jump is 1 only between steps adjacent on the timeline.  Before: only the
    10 block chunks, numbered consecutively, so the gaps vanished (jump 1 from t0003 to t0009)."""
    tl, sc = _config({"w5": list(FIVE_WEEKS)}, {"s": [("p2020", "w5")]}, {"s": 2.0})
    tl.create_timeline_from_timestep_duration(sc)
    new_tl = tl.timesets__timeline["w5"]
    assert tl.original_timeline[new_tl] == "T"
    assert [s for s, _d in tl.timelines[new_tl]] == [f"t{i:04d}" for i in range(1, 73, 2)]
    assert tl.timeset_durations["w5"] == [(start, 2) for start, _n in FIVE_WEEKS]
    active = get_active_time("s", sc.timesets_used_by_solves, tl.timeset_durations, tl.timelines,
                             tl.timesets__timeline)
    jumps = {r[1]: r[6] for r in make_step_jump(active, [("p2020", "p2020")], [])}
    assert jumps == {"t0001": -19, "t0003": 1, "t0009": 3, "t0011": 1, "t0013": 1, "t0015": 1,
                     "t0025": 5, "t0027": 1, "t0037": 5, "t0039": 1}


def test_two_timesets_share_one_timeline() -> None:
    """Two timesets on one timeline (ta t0001-24, tb t0025-48) of one solve with Δ 2 both keep their 12 steps.
    Before: the second timeset's rewrite overwrote the first one's timeline (#103)."""
    tl, sc = _config({"ta": [("t0001", 24)], "tb": [("t0025", 24)]},
                     {"s": [("p2020", "ta"), ("p2025", "tb")]}, {"s": 2.0})
    tl.create_timeline_from_timestep_duration(sc)
    assert tl.timesets__timeline["ta"] == tl.timesets__timeline["tb"]
    active = get_active_time("s", sc.timesets_used_by_solves, tl.timeset_durations, tl.timelines,
                             tl.timesets__timeline)
    assert [e.timestep for e in active["p2020"]] == [f"t{i:04d}" for i in range(1, 25, 2)]
    assert [e.timestep for e in active["p2025"]] == [f"t{i:04d}" for i in range(25, 49, 2)]


def test_shared_timeset_same_step() -> None:
    """Two solves with the same Δ sharing a gapped timeset aggregate it from the original timeline once: one
    timeline, the same as for either solve alone.  Before: the second solve re-aggregated the first one's
    rewrite (``T_s1_s2``, without the gaps)."""
    alone, sc_alone = _config({"w5": list(FIVE_WEEKS)}, {"s1": [("p2020", "w5")]}, {"s1": 2.0})
    alone.create_timeline_from_timestep_duration(sc_alone)
    shared, sc = _config({"w5": list(FIVE_WEEKS)}, {"s1": [("p2020", "w5")], "s2": [("p2025", "w5")]},
                         {"s1": 2.0, "s2": 2.0})
    shared.create_timeline_from_timestep_duration(sc)
    assert len(shared.timelines[shared.timesets__timeline["w5"]]) == 36
    assert shared.timelines[shared.timesets__timeline["w5"]] == alone.timelines[alone.timesets__timeline["w5"]]
    assert shared.timeset_durations["w5"] == alone.timeset_durations["w5"]
    # Idempotent: a second call aggregates from the original timeline again.
    shared.create_timeline_from_timestep_duration(sc)
    assert shared.timelines[shared.timesets__timeline["w5"]] == alone.timelines[alone.timesets__timeline["w5"]]


def test_shared_timeset_with_and_without_step_raises() -> None:
    """A timeset shared by a solve with new_stepduration and a solve of the model without one raises (one
    timeline per timeset).  Before: the solve without new_stepduration silently ran on the aggregated
    timeline with unaggregated data."""
    tl, sc = _config({"w5": list(FIVE_WEEKS)}, {"s1": [("p2020", "w5")], "s2": [("p2020", "w5")]}, {"s1": 2.0})
    with pytest.raises(FlexToolConfigError, match="no new_stepduration"):
        tl.create_timeline_from_timestep_duration(sc)
    # A solve that is not part of the model does not count.
    tl, sc = _config({"w5": list(FIVE_WEEKS)}, {"s1": [("p2020", "w5")], "s2": [("p2020", "w5")]}, {"s1": 2.0})
    sc.model_solve = {"m": ["s1"]}
    tl.create_timeline_from_timestep_duration(sc)


def test_partial_overlap_warns(caplog) -> None:
    """A block of one timeset that starts inside a 2-h step of another timeset's block cuts that step short:
    warned, naming both timesets."""
    tl, sc = _config({"ta": [("t0001", 8)], "tb": [("t0004", 4)]},
                     {"s": [("p2020", "ta"), ("p2025", "tb")]}, {"s": 2.0}, n_steps=12)
    with caplog.at_level(logging.WARNING):
        tl.create_timeline_from_timestep_duration(sc)
    assert [s for s, _d in tl.timelines[tl.timesets__timeline["ta"]]] == [
        "t0001", "t0003", "t0004", "t0006", "t0008", "t0009", "t0011"]
    assert tl.timeset_durations["ta"] == [("t0001", 5)]
    assert tl.timeset_durations["tb"] == [("t0004", 2)]
    assert any("'ta'" in r.message and "'tb'" in r.message for r in caplog.records)


@pytest.mark.parametrize("step", [2.0, 4.0, 5.0])
def test_rp_base_starts_on_the_timeline(step) -> None:
    """Rep blocks t0001 / t0073 (24 h) on a 96-h timeline, base periods t0001, t0025, t0049, t0073: every base
    start is a step of the aggregated timeline and the base chain runs in time order for any Δ.  Before (Δ 5):
    t0049 fell inside a gap chunk, sorted last and the chain became t0025←t0001, t0073←t0025, t0049←t0073."""
    from flextool.engine.preprocess.emit._solve_writers import _compute_rp_period_frames

    bases = ("t0001", "t0025", "t0049", "t0073")
    tl, sc = _config({"rp": [("t0001", 24), ("t0073", 24)]}, {"s": [("p", "rp")]}, {"s": step}, n_steps=96)
    tl.rp_weights = {"rp": {b: {"t0001": 0.5, "t0073": 0.5} for b in bases}}
    tl.create_timeline_from_timestep_duration(sc)
    steps = [s for s, _d in tl.timelines[tl.timesets__timeline["rp"]]]
    assert set(bases) <= set(steps)
    frames = _compute_rp_period_frames(tl.rp_weights["rp"], tl.timeset_durations["rp"], "p", steps)
    assert sorted(frames["rp_base_chain.csv"].rows()) == [
        ("t0025", "t0001"), ("t0049", "t0025"), ("t0073", "t0049")]


def test_rp_base_start_off_the_timeline_raises() -> None:
    """Item 112: a base period that does not start at a step of the timeline (t0097 on a 96-step timeline) is a
    configuration error.  Before: it was sorted after every other base period and the inter-period storage chain
    linked it out of time order, silently."""
    from flextool.engine.preprocess.emit._solve_writers import _compute_rp_period_frames

    steps = [f"t{i:04d}" for i in range(1, 97)]
    weights = {b: {"t0001": 0.5, "t0025": 0.5} for b in ("t0001", "t0025", "t0097")}
    with pytest.raises(FlexToolConfigError, match="t0097"):
        _compute_rp_period_frames(weights, [("t0001", 24), ("t0025", 24)], "p", steps)
    frames = _compute_rp_period_frames({b: weights[b] for b in ("t0025", "t0001")}, [("t0001", 24), ("t0025", 24)],
                                "p", steps)
    assert frames["rp_base_chain.csv"].rows() == [("t0025", "t0001")]


def test_rp_base_start_guard(monkeypatch) -> None:
    """If a base start on the original timeline were still not a step of the aggregated timeline, the build raises
    instead of letting the rp chain sort it last (guard; forced here by dropping the base-start cuts)."""
    import flextool.engine.input._timeline as T

    orig = T._aggregate_timeline_pieces
    monkeypatch.setattr(T, "_aggregate_timeline_pieces",
                        lambda steps, blocks, sd, log, extra_cuts=None, **kw: orig(steps, blocks, sd, log, **kw))
    tl, sc = _config({"rp": [("t0001", 24), ("t0073", 24)]}, {"s": [("p", "rp")]}, {"s": 5.0}, n_steps=96)
    tl.rp_weights = {"rp": {b: {"t0001": 0.5, "t0073": 0.5} for b in ("t0001", "t0025", "t0049", "t0073")}}
    with pytest.raises(FlexToolConfigError, match="t0049"):
        tl.create_timeline_from_timestep_duration(sc)


# ---------------------------------------------------------------------------
# solves


def test_blocked_storage_keeps_state_change(tmp_path_factory) -> None:
    """battery (bind_within_timeblock) on 5weeks + Δ 2: each 2-h step's balance carries the state change from its
    predecessor (t0003 from t0001), t0001 wraps to its block's last step t0003, t0009 to t0015 (the t0009 and
    t0013 blocks touch).  Before: every 2-h step looked like a block start, so no step had a state change term
    and the battery could not shift energy."""
    step = _last(_run(tmp_path_factory, "agg_bat", STEP_2H_5W, BATTERY + ["5weeks"]))
    fd = step.flex_data
    rows = row_terms(build(fd), "nodeBalance_eq", "battery,p2020,")
    row = rows["nodeBalance_eq[battery,p2020,t0003]"]
    s = coef(row, "v_state[battery,p2020,t0001]")
    assert s > 0
    assert coef(row, "v_state[battery,p2020,t0003]") == pytest.approx(-s, rel=1e-3)
    lag = {(d, t): tp for d, t, tp in utf8(fd.dtttdt).select("d", "t", "t_previous_within_timeset").iter_rows()}
    assert lag[("p2020", "t0001")] == "t0003"
    assert lag[("p2020", "t0009")] == "t0015"
    assert lag[("p2020", "t0013")] == "t0011"
    assert lag[("p2020", "t0025")] == "t0027"


def test_blocked_intraperiod_blocks(tmp_path_factory) -> None:
    """bind_intraperiod_blocks on 5weeks + Δ 2: four blocks per period (t0001, t0009, t0025, t0037) as in the
    hourly solve.  Before: every 2-h step was a block (10 per period)."""
    step = _last(_run(tmp_path_factory, "agg_ipb", STEP_2H_5W,
                      BATTERY + ["5weeks", "bind_battery_intraperiod_blocks"]))
    pb = utf8(step.flex_data.period_block)
    for d in ("p2020", "p2025"):
        assert sorted(pb.filter(pl.col("d") == d)["b_first"].to_list()) == ["t0001", "t0009", "t0025", "t0037"]


def test_two_timesets_one_timeline(tmp_path_factory) -> None:
    """Two periods on two timesets of one timeline with Δ 2: dt has both periods, 12 steps each.  Before: only
    p2025 (the last timeset's rewrite won) (#103)."""
    step = _last(_run(tmp_path_factory, "agg_2p", TWO_PERIODS_2H, ["init", "west", "coal"],
                      entities=TWO_PERIODS_ENTS))
    dt = _dt(step.flex_data)
    assert len(dt) == 24
    assert sum(1 for d, _t in dt if d == "p2020") == 12 and sum(1 for d, _t in dt if d == "p2025") == 12


def test_stochastic_trunk_kept(tmp_path_factory) -> None:
    """Mid-horizon stochastic fixture + Δ 2: the trunk p2035 keeps t0001 (2 h) and t0003 (1 h); every p2040
    branch's first step follows p2035's t0003.  Before: p2035 was dropped (#103)."""
    vals = [["solve", "stoch_2p_mid", "new_stepduration", b64(2.0, "float")]]
    step = _last(_run(tmp_path_factory, "agg_mid", vals, ["init", "base", "stoch"], fixture_json=MIDHORIZON))
    fd = step.flex_data
    durs = {(d, t): v for d, t, v in utf8(fd.p_step_duration.frame).select("d", "t", "value").iter_rows()}
    assert durs[("p2035", "t0001")] == pytest.approx(2.0)
    assert durs[("p2035", "t0003")] == pytest.approx(1.0)
    lag = utf8(fd.dtttdt).filter(pl.col("d").str.starts_with("p2040") & (pl.col("t") == "t0004"))
    assert lag.height >= 2
    assert set(lag.select("d_previous", "t_previous_within_solve").iter_rows()) == {("p2035", "t0003")}


def test_rolling_blocked_carry(tmp_path_factory) -> None:
    """Rolling dispatch (horizon 8 h, jump 4 h) over 5weeks + Δ 2 with the coal ramp: only the roll that starts at
    t0013 carries the previous roll's flow (from t0011); the rolls starting after a gap (t0009, t0025, t0037) do
    not, and no ramp row links t0003 and t0009 inside a roll.  Before: all four continuation rolls carried."""
    ramp = [["unit__outputNode", ["coal_plant", "west"], "ramp_method", b64("ramp_limit", "str")],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_up", b64(0.002, "float")],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_down", b64(0.002, "float")]]
    vals = ramp + [
        ["solve", "dispatch_fullYear_roll", "period_timeset", _pmap({"p2020": "5weeks"})],
        ["solve", "dispatch_fullYear_roll", "realized_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(8.0, "float")],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(4.0, "float")],
        ["solve", "dispatch_fullYear_roll", "new_stepduration", b64(2.0, "float")]]
    steps = _run(tmp_path_factory, "agg_roll", vals, ["init", "west", "coal", "fullYear", "dispatch_fullYear_roll"])
    carry = {}
    for s in steps.values():
        fd = s.flex_data
        first = _dt(fd)[0][1]
        c = fd.dt_ramp_carry
        carry[first] = None if c is None or c.height == 0 else sorted(utf8(c).drop("d").rows())
        link = {tuple(r) for r in utf8(fd.dt_ramp_link).select("t", "t_previous").iter_rows()} \
            if fd.dt_ramp_link is not None else set()
        assert ("t0009", "t0003") not in link
    assert set(carry) == {"t0001", "t0009", "t0013", "t0025", "t0037"}
    assert carry["t0009"] is None and carry["t0025"] is None and carry["t0037"] is None
    assert carry["t0013"] is not None and all("t0011" in r for r in carry["t0013"])


@pytest.mark.parametrize("case", ["5w", "5w2h", "2p2h", "mid2h"])
def test_derived_lags_equal_step_previous(tmp_path_factory, case) -> None:
    """The derived dtttdt (ranks in the solve's own timeline) equals make_step_jump's step_previous on the lag
    columns, unaggregated and aggregated.  Before: on 5weeks + Δ 2 every aggregated step was a block start.
    (``t_previous`` of a period's first step is left out: make_step_jump takes the period's own last step there,
    the derived frame the predecessor period's.  It is read only by the block-interior filter
    (``dtttdt_block_interior``, ``t_previous_within_timeset == t_previous``), where the difference adds or drops a
    redundant row tying a block's first state to its last; pre-existing and independent of new_stepduration,
    item 110.)"""
    if case == "5w":
        steps = _run(tmp_path_factory, "lag_5w", [], BATTERY + ["5weeks"])
    elif case == "5w2h":
        steps = _run(tmp_path_factory, "lag_5w2h", STEP_2H_5W, BATTERY + ["5weeks"])
    elif case == "2p2h":
        steps = _run(tmp_path_factory, "lag_2p2h", TWO_PERIODS_2H, ["init", "west", "coal"],
                     entities=TWO_PERIODS_ENTS)
    else:
        steps = _run(tmp_path_factory, "lag_mid2h", [["solve", "stoch_2p_mid", "new_stepduration", b64(2.0, "float")]],
                     ["init", "base", "stoch"], fixture_json=MIDHORIZON)
    step = _last(steps)
    csv = _csv_step_previous(step).select(_LAG_COLS).sort("d", "t")
    derived = utf8(step.flex_data.dtttdt).select(_LAG_COLS).sort("d", "t")
    assert csv.drop("t_previous").rows() == derived.drop("t_previous").rows()
    inner = pl.col("d_previous") == pl.col("d")
    assert (csv.filter(inner).select("d", "t", "t_previous").rows()
            == derived.filter(inner).select("d", "t", "t_previous").rows())


def test_step_space_ranks(tmp_path_factory) -> None:
    """5weeks + Δ 2: the active steps' ranks are their positions in the solve's timeline (36 2-h chunks), so
    adjacent steps differ by 1 and a gap shows as a larger difference.  Before: raw timeline ranks."""
    from flextool.engine.preprocess.params._derived import _to_solve_step_space

    step = _last(_run(tmp_path_factory, "rank_5w2h", STEP_2H_5W, BATTERY + ["5weeks"]))
    labels = ["t0001", "t0003", "t0009", "t0011", "t0013", "t0015", "t0025", "t0027", "t0037", "t0039"]
    per_period = {"p2020": [(t, int(t[1:])) for t in labels]}
    assert _to_solve_step_space(per_period, {"p2020": "y2020"}, Path("."), step.flex_data_provider)
    assert [rk for _t, rk in per_period["p2020"]] == [1, 2, 5, 6, 7, 8, 13, 14, 19, 20]


def test_scale_in_proportion_equals_unaggregated(tmp_path_factory) -> None:
    """west scaled with scale_in_proportion on 5weeks: the inflow multiplier with Δ 2 equals the hourly solve's
    (both normalise over the whole timeline).  Before: the aggregated solve summed the inflow over the block
    hours only, against the aggregated timeline's (block-only) duration."""
    vals = [["node", "west", "inflow_method", b64("scale_in_proportion", "str")],
            ["node", "west", "annual_flow", _pmap({"p2020": 5.0e6, "p2025": 5.0e6})]]
    mult = {}
    for alt, extra in (("sip_1h", []), ("sip_2h", STEP_2H_5W)):
        step = _last(_run(tmp_path_factory, alt, vals + extra, BATTERY + ["5weeks"]))
        df = step.flex_data_provider.get("solve_data/period_flow_proportional_multiplier")
        assert df is not None and df.height > 0
        mult[alt] = sorted(utf8(df).rows())
    assert [r[:2] for r in mult["sip_1h"]] == [r[:2] for r in mult["sip_2h"]]
    for a, b in zip(mult["sip_1h"], mult["sip_2h"]):
        assert float(a[-1]) == pytest.approx(float(b[-1]), rel=1e-9)


def test_rp_blocks_with_step(tmp_path_factory) -> None:
    """Representative timeset rpg (rep blocks t0001, t0049 of 24 h; base periods t0001, t0025, t0049) with a
    bind_within_period_blended_weights battery and Δ 2: rp_block_first = {t0001, t0049}, the blocks' last steps
    t0023 / t0071, the base periods chain in time order (t0025 after t0001) and every 2-h step inside a rep block
    carries its state change.  Before: no state change inside the blocks and t0025 sorted after t0049."""
    def rep(w: float) -> dict:
        return {"type": "map", "index_type": "str", "index_name": "representative_period", "rank": 1,
                "data": [["t0001", w], ["t0049", 1.0 - w]]}
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", rep(1.0)], ["t0025", rep(0.5)], ["t0049", rep(0.0)]]}
    vals = [["timeset", "rpg", "timeline", b64("y2020", "str")],
            ["timeset", "rpg", "timeset_duration", _pmap({"t0001": 24.0, "t0049": 24.0}, "time")],
            ["timeset", "rpg", "representative_period_weights", b64(rpw, "map")],
            ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "rpg"})],
            ["node", "battery", "storage_binding_method", b64("bind_within_period_blended_weights", "str")],
            ["solve", "y2020_2day_dispatch", "new_stepduration", b64(2.0, "float")]]
    step = _last(_run(tmp_path_factory, "agg_rp", vals, BATTERY + ["coal"], entities=[["timeset", "rpg", None]]))
    fd = step.flex_data
    assert sorted(utf8(fd.rp_block_first).select("t").to_series().to_list()) == ["t0001", "t0049"]
    assert sorted(utf8(fd.p_rp_last_step).select(pl.last()).to_series().to_list()) == ["t0023", "t0071"]
    pb = build(fd)
    bal = row_terms(pb, "nodeBalance_eq", "battery,p2020,")
    for t, tp in (("t0003", "t0001"), ("t0023", "t0021"), ("t0051", "t0049")):
        assert coef(bal[f"nodeBalance_eq[battery,p2020,{t}]"], f"v_state[battery,p2020,{tp}]") > 0
    assert "v_state_rp_start[battery,p2020,t0049]" in " ".join(bal["nodeBalance_eq[battery,p2020,t0049]"])
    chain = set(row_terms(pb, "rp_inter_period_balance"))
    assert chain == {"rp_inter_period_balance[battery,p2020,t0025,p2020,t0001]",
                     "rp_inter_period_balance[battery,p2020,t0049,p2020,t0025]"}


def test_realized_dispatch_sibling_inherits_step(tmp_path_factory) -> None:
    """separate_dispatch parent with a solve-level new_stepduration 2 on a 3-h timeline: the auto-spawned
    realized-dispatch sibling runs at the parent's step (t0001 2 h, t0003 1 h, inflow summed per step) and costs
    the same as the hourly sibling (constant demand).  Before: the sibling got the aggregated step labels with
    unaggregated data — steps t0001–t0003 of 1 h but inflow only at t0001 / t0003 (−100 each), a third of the
    demand dropped."""
    fixture = FLEXTOOL_ROOT / "tests" / "fixtures" / "w4b_auto_dispatch.json"
    chain = ["init", "base", "stoch"]
    hourly = _run(tmp_path_factory, "sib_1h", [], chain, fixture_json=fixture)["stoch_rd__realized_dispatch"]
    steps = _run(tmp_path_factory, "sib_2h", [["solve", "stoch_rd", "new_stepduration", b64(2.0, "float")]], chain,
                 fixture_json=fixture)
    sib = steps["stoch_rd__realized_dispatch"]
    fd = sib.flex_data
    assert sorted(utf8(fd.p_step_duration.frame).select("d", "t", "value").rows()) == [
        ("p2035", "t0001", 2.0), ("p2035", "t0003", 1.0)]
    assert sorted(utf8(fd.p_inflow.frame).select("t", "value").rows()) == [("t0001", -200.0), ("t0003", -100.0)]
    assert sib.solution.obj == pytest.approx(hourly.solution.obj, rel=1e-9)


# ---------------------------------------------------------------------------
# base step length: new_stepduration aggregates by the timeline's time steps, whatever their length (#200)


def _steps_of(tl: TimelineConfig, ts: str) -> list[tuple[str, str]]:
    return [(s, d) for s, d in tl.timelines[tl.timesets__timeline[ts]]]


@pytest.mark.parametrize(("durations", "step", "expected"), [
    ([2.0] * 72, 4.0, [(f"t{i:04d}", "4.0") for i in range(1, 73, 2)]),
    ([0.5] * 72, 1.0, [(f"t{i:04d}", "1.0") for i in range(1, 73, 2)]),
    ([1.0] * 24 + [2.0] * 24, 2.0,
     [(f"t{i:04d}", "2.0") for i in range(1, 25, 2)] + [(f"t{i:04d}", "2.0") for i in range(25, 49)]),
    ([1.0, 2.0] * 24, 3.0, [(f"t{i:04d}", "3.0") for i in range(1, 49, 2)]),
], ids=["2h-by-4", "0.5h-by-1", "1h-then-2h-by-2", "1h2h-by-3"])
def test_chunks_of_any_base_step(durations, step, expected) -> None:
    """The new time steps hold exactly new_stepduration hours of the timeline's steps, for 2 h, 0.5 h and mixed base
    steps (a chunk closes when it reaches the new step, it never adds one more base step)."""
    n = len(durations)
    tl, sc = _config({"all": [("t0001", n)]}, {"s": [("p2020", "all")]}, {"s": step}, durations=durations)
    tl.create_timeline_from_timestep_duration(sc)
    assert _steps_of(tl, "all") == expected
    assert tl.timeset_durations["all"] == [("t0001", len(expected))]


def test_rounded_fractional_steps_fit(caplog) -> None:
    """20-minute steps entered as 0.333333 h with new_stepduration 2: two new steps of six base steps each (their
    sums fall 2e-6 h short of 2 h), no error and no short-step warning."""
    tl, sc = _config({"all": [("t0001", 12)]}, {"s": [("p2020", "all")]}, {"s": 2.0}, durations=[0.333333] * 12)
    with caplog.at_level(logging.WARNING):
        tl.create_timeline_from_timestep_duration(sc)
    steps = _steps_of(tl, "all")
    assert [s for s, _d in steps] == ["t0001", "t0007"]
    assert [float(d) for _s, d in steps] == [pytest.approx(6 * 0.333333, rel=1e-12)] * 2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_long_gap_step_is_its_own_chunk() -> None:
    """A 24 h filler step in the gap between two blocks of 1 h steps, new_stepduration 2: the gap step is one new
    step of 24 h (never active, so not an error); the blocks get 2 h steps."""
    durations = [1.0] * 4 + [24.0] + [1.0] * 4
    tl, sc = _config({"w": [("t0001", 4), ("t0006", 4)]}, {"s": [("p2020", "w")]}, {"s": 2.0}, durations=durations)
    tl.create_timeline_from_timestep_duration(sc)
    assert _steps_of(tl, "w") == [("t0001", "2.0"), ("t0003", "2.0"), ("t0005", "24.0"), ("t0006", "2.0"),
                                  ("t0008", "2.0")]
    assert tl.timeset_durations["w"] == [("t0001", 2), ("t0006", 2)]


@pytest.mark.parametrize(("durations", "step", "match"), [
    ([2.0] * 12, 3.0, r"new_stepduration 3 h is not a whole multiple of the 2 h time steps of timeline 'T'"),
    ([2.0] * 12, 1.0, r"new_stepduration 1 h is not a whole multiple of the 2 h time steps"),
    ([1.0] * 12, 1.5, r"new_stepduration 1.5 h is not a whole multiple of the 1 h time steps"),
    ([1.0, 2.0] * 6, 2.0, r"new_stepduration 2 h does not fit the time steps of timeline 'T': the new time step starting at "
                            r"'t0001' would end inside base time step 't0002' \(3 h in total\).*must end where"),
], ids=["2h-by-3", "2h-by-1", "1h-by-1.5", "1h2h-by-2"])
def test_step_not_fitting_the_base_steps_raises(durations, step, match) -> None:
    """A new_stepduration that is not a whole multiple of the timeline's steps, or (mixed step lengths) that would
    end inside a base step, stops with a configuration error naming the solve.  Before: longer new steps were made
    silently (2 h base by 3 → 4 h steps)."""
    tl, sc = _config({"all": [("t0001", len(durations))]}, {"s": [("p2020", "all")]}, {"s": step},
                     durations=durations)
    with pytest.raises(FlexToolConfigError, match=r"Solve 's': " + match):
        tl.create_timeline_from_timestep_duration(sc)


# ---------------------------------------------------------------------------
# base step length: the time series follow the base steps


WEST6 = [-589.0, -537.0, -506.0, -482.0, -472.0, -454.0]
WIND6 = [0.38, 0.40, 0.43, 0.46, 0.45, 0.49]
CONST = -500.0
LABELS = ["t0001", "t0003", "t0005"]


def _frame(header: list[str], rows: list[list]) -> pl.DataFrame:
    return pl.DataFrame([[str(v) for v in r] for r in rows], schema=header, orient="row")


def _inputs() -> dict[str, pl.DataFrame]:
    t = [f"t{i:04d}" for i in range(1, 7)]
    br = [b for b in ("b1", "b2") for _ in t]
    return {
        "pt_node_inflow": _frame(["node", "time", "pt_node_inflow"], [["west", s, v] for s, v in zip(t, WEST6)]),
        "pbt_node_inflow": _frame(["node", "branch", "time_start", "time", "pbt_node_inflow"],
                                  [["west", b, "t0001", s, v * (1.0 if b == "b1" else 1.5)]
                                   for b, s, v in zip(br, t * 2, WEST6 * 2)]),
        "pt_profile": _frame(["profile", "time", "pt_profile"], [["wind", s, v] for s, v in zip(t, WIND6)]),
        "pdt_process": _frame(["process", "processParam", "period", "time", "pdt_process"],
                              [["coal", "availability", d, s, round(0.5 + 0.07 * i + k, 6)]
                               for k, d in enumerate(("p2020", "p2025")) for i, s in enumerate(t)]),
        "pt_node": _frame(["node", "nodeParam", "time", "pt_node"],
                          [["west", "availability", s, v + 1.0] for s, v in zip(t, WIND6)]
                          + [["battery", "storage_state_reference_value", "t0003", 0.5],
                             ["battery2", "storage_state_reference_value", "t0004", 0.7]]),
        "p_node": _frame(["node", "nodeParam", "p_node"], [["east", "inflow", CONST], ["east", "penalty_up", 900]]),
    }


def _averaged(durations: list[float], step: float, tmp_path: Path) -> tuple[dict[str, pl.DataFrame], dict]:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    tl, sc = _config({"all": [("t0001", 6)]}, {"s": [("p2020", "all")]}, {"s": step}, durations=durations)
    tl.create_timeline_from_timestep_duration(sc)
    provider = FlexDataProvider()
    inputs = _inputs()
    for name, frame in inputs.items():
        provider.put(f"input/{name}", frame)
    tl.create_averaged_timeseries("s", sc, logging.getLogger(__name__), provider=provider, work_folder=tmp_path)
    names = list(inputs) + ["node_full_timeline_flow"]
    return {n: provider.get(f"solve_data/{n}") for n in names if provider.has(f"solve_data/{n}")}, inputs


def _groups(frame: pl.DataFrame) -> dict[tuple, list[tuple[str, float]]]:
    """``{key columns before time: [(time, value), ...]}`` in row order."""
    ti = frame.columns.index("time")
    out: dict[tuple, list[tuple[str, float]]] = defaultdict(list)
    for r in frame.iter_rows():
        out[tuple(r[:ti])].append((r[ti], float(r[ti + 1])))
    return out


BASES = [([1.0] * 6, 2.0), ([2.0] * 6, 4.0), ([0.5] * 6, 1.0), ([1.0, 2.0] * 3, 3.0)]
BASE_IDS = ["1h-by-2", "2h-by-4", "0.5h-by-1", "1h2h-by-3"]


@pytest.mark.parametrize(("durations", "step"), BASES, ids=BASE_IDS)
def test_series_follow_the_base_steps(durations, step, tmp_path) -> None:
    """Each new step (t0001, t0003, t0005) combines two base steps: inflow is their sum, the other series their
    duration-weighted mean, a constant inflow counts once per base step, a storage reference value at a new step
    passes through and one at another base step moves to the new step that contains it.  Before (other than 1 h
    bases): the new step's hours were taken as the number of rows to read."""
    out, inputs = _averaged(durations, step, tmp_path)
    ds = [durations[0:2], durations[2:4], durations[4:6]]
    for name in ("pt_node_inflow", "pbt_node_inflow"):
        got, src = _groups(out[name]), _groups(inputs[name])
        assert set(got) == set(src) | ({("east",)} if name == "pt_node_inflow" else set())
        for key, rows in src.items():
            values = [v for _t, v in rows]
            assert [t for t, _v in got[key]] == LABELS
            assert [v for _t, v in got[key]] == [values[0] + values[1], values[2] + values[3], values[4] + values[5]]
            assert sum(v for _t, v in got[key]) == pytest.approx(sum(values), rel=1e-12)
    east = _groups(out["pt_node_inflow"])[("east",)]
    assert east == [(t, CONST * 2) for t in LABELS]
    assert sum(v for _t, v in east) == CONST * 6
    for name in ("pt_profile", "pdt_process", "pt_node"):
        got, src = _groups(out[name]), _groups(inputs[name])
        for key, rows in src.items():
            if "storage_state_reference_value" in key:
                continue
            values = [v for _t, v in rows]
            expected = [sum(v * d for v, d in zip(values[2 * k:2 * k + 2], ds[k])) / sum(ds[k]) for k in range(3)]
            assert [t for t, _v in got[key]] == LABELS
            assert [v for _t, v in got[key]] == pytest.approx(expected, abs=1e-6)
    ref = _groups(out["pt_node"])
    assert ref[("battery", "storage_state_reference_value")] == [("t0003", 0.5)]
    assert ref[("battery2", "storage_state_reference_value")] == [("t0003", 0.7)]
    flow = {n: float(v) for n, v in out["node_full_timeline_flow"].iter_rows()}
    years = sum(durations) / 8760.0
    assert flow["east"] == pytest.approx(CONST * 6 / years, rel=1e-12)
    assert flow["west"] == pytest.approx(sum(WEST6) / years, rel=1e-12)


def test_mixed_steps_weight_the_average(tmp_path) -> None:
    """Base [1 h, 2 h] by 3: the profile of new step t0001 is (0.38·1 + 0.40·2)/3 = 0.393333, not the plain mean
    0.39."""
    out, _inputs_ = _averaged([1.0, 2.0] * 3, 3.0, tmp_path)
    assert _groups(out["pt_profile"])[("wind",)][0] == ("t0001", 0.393333)


# ---------------------------------------------------------------------------
# base step length: solves


TIMESTEPS72 = [f"t{i:04d}" for i in range(1, 73)]


def _tmap(pairs: list, index_name: str = "time") -> list:
    return b64({"index_type": "str", "index_name": index_name, "rank": 1, "data": [list(p) for p in pairs]}, "map")


def _base_series() -> dict[str, list[float]]:
    """The fixture's west inflow and wind profile over the 72 steps (from tests.json)."""
    import base64 as _b64
    import json as _json

    data = _json.loads((FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json").read_text())
    out = {}
    for cls, ent, param, val, alt in data["parameter_values"]:
        if (cls, ent, param, alt) in (("node", "west", "inflow", "west"), ("profile", "wind_profile", "profile", "wind")):
            m = _json.loads(_b64.b64decode(val[0]))
            assert [t for t, _v in m["data"]] == TIMESTEPS72
            out[param] = [float(v) for _t, v in m["data"]]
    return out


def _timeline(base: float) -> list:
    return ["timeline", "y2020", "timestep_duration", _tmap([(t, base) for t in TIMESTEPS72], "timestep")]


def _assert_same_lp(steps_a: list, steps_b: list) -> None:
    """Every sub-solve's LP is the same: rows, columns, coefficients, bounds and costs (the same floats)."""
    assert len(steps_a) == len(steps_b) > 0
    for a, b in zip(steps_a, steps_b):
        ma, mb = build(a.flex_data).canonicalise(), build(b.flex_data).canonicalise()
        assert list(ma.row_names) == list(mb.row_names), a.solve_name
        assert list(ma.col_names) == list(mb.col_names), a.solve_name
        assert list(ma.col_ptr) == list(mb.col_ptr) and list(ma.row_idx) == list(mb.row_idx), a.solve_name
        assert list(ma.val) == list(mb.val), a.solve_name
        assert list(ma.row_lb) == list(mb.row_lb) and list(ma.row_ub) == list(mb.row_ub), a.solve_name
        assert list(ma.col_lb) == list(mb.col_lb) and list(ma.col_ub) == list(mb.col_ub), a.solve_name
        assert list(ma.col_obj) == list(mb.col_obj), a.solve_name


def _inflow_dt(step) -> list[tuple[str, float]]:
    fd = step.flex_data
    dt = set(_dt(fd))
    rows = utf8(fd.p_inflow.frame).filter(pl.col("n") == "west").select("d", "t", "value").rows()
    return sorted((t, v) for d, t, v in rows if (d, t) in dt)


@pytest.mark.parametrize("base", [2.0, 0.5], ids=["2h-by-4", "0.5h-by-1"])
def test_coarse_solve_equals_native_twin(tmp_path_factory, base) -> None:
    """2day on a timeline of 2 h (0.5 h) steps with new_stepduration 4 (1) builds the same LP as the twin timeline
    that has the 4 h (1 h) steps natively (t0001, t0003, ...; inflow summed and profile averaged pairwise), and the
    west inflow sums to the 48 base steps' −29 103 (t0001: −589 − 537 = −1126).  Before: 2 h by 4 read four steps
    per new step (Σ doubled), 0.5 h by 1 read one (Σ −14 607)."""
    series = _base_series()
    infl, prof = series["inflow"], series["profile"]
    twin_steps = TIMESTEPS72[0::2]
    twin = [
        ["timeline", "y2020", "timestep_duration", _tmap([(t, 2 * base) for t in twin_steps], "timestep")],
        ["timeset", "2day", "timeset_duration", _tmap([("t0001", 24.0)], "time")],
        ["node", "west", "inflow", _tmap([(t, infl[2 * k] + infl[2 * k + 1]) for k, t in enumerate(twin_steps)])],
        ["profile", "wind_profile", "profile",
         _tmap([(t, round((prof[2 * k] + prof[2 * k + 1]) / 2, 6)) for k, t in enumerate(twin_steps)])],
    ]
    coarse = [_timeline(base), ["solve", "y2020_2day_dispatch", "new_stepduration", b64(2 * base, "float")]]
    tag = str(base).replace(".", "p")
    a = list(_run(tmp_path_factory, f"bsl_coarse_{tag}", coarse, BATTERY).values())
    b = list(_run(tmp_path_factory, f"bsl_twin_{tag}", twin, BATTERY).values())
    _assert_same_lp(a, b)
    rows = _inflow_dt(a[-1])
    assert len(rows) == 24
    assert rows[0] == ("t0001", -1126.0)
    assert sum(v for _t, v in rows) == pytest.approx(-29103.0, rel=1e-12)
    assert sum(infl[:48]) == -29103.0


def test_constant_inflow_counts_per_base_step(tmp_path_factory) -> None:
    """Constant west inflow −500 (MWh per step) on 2 h steps with new_stepduration 4: every new step gets −1000 and
    the 2day total equals the total without new_stepduration (−500 × 48).  Before: −2000 per step (×4 h)."""
    const = [_timeline(2.0), ["node", "west", "inflow", b64(-500.0, "float")]]
    coarse = _inflow_dt(_last(_run(tmp_path_factory, "bsl_const_4", const + [
        ["solve", "y2020_2day_dispatch", "new_stepduration", b64(4.0, "float")]], BATTERY)))
    fine = _inflow_dt(_last(_run(tmp_path_factory, "bsl_const_2", const, BATTERY)))
    assert len(coarse) == 24 and {v for _t, v in coarse} == {-1000.0}
    assert len(fine) == 48
    assert sum(v for _t, v in coarse) == sum(v for _t, v in fine) == -500.0 * 48


def test_scale_to_annual_flow_constant_on_coarse_steps(tmp_path_factory) -> None:
    """fullYear (72 steps of 2 h = 144 h) with a constant west inflow −500 scaled to annual_flow 1e6: the scaled
    inflow sums to −1e6 × 144/8760 with and without new_stepduration 4 (the full-timeline denominator counts the
    constant once per base step, like the coarse inflow)."""
    vals = [_timeline(2.0), ["node", "west", "inflow", b64(-500.0, "float")],
            ["node", "west", "inflow_method", b64("scale_to_annual_flow", "str")],
            ["node", "west", "annual_flow", b64(1.0e6, "float")]]
    chain = ["init", "west", "wind", "fullYear"]
    coarse = _inflow_dt(_last(_run(tmp_path_factory, "bsl_ftf_4", vals + [
        ["solve", "y2020_fullYear_dispatch", "new_stepduration", b64(4.0, "float")]], chain)))
    fine = _inflow_dt(_last(_run(tmp_path_factory, "bsl_ftf_2", vals, chain)))
    assert len(coarse) == 36 and len(fine) == 72
    expected = -1.0e6 * 144.0 / 8760.0
    assert sum(v for _t, v in coarse) == pytest.approx(expected, rel=1e-9)
    assert sum(v for _t, v in fine) == pytest.approx(expected, rel=1e-9)
