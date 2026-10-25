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
            n_steps: int = 72) -> tuple[TimelineConfig, SimpleNamespace]:
    tl = TimelineConfig(
        timelines=defaultdict(list, {"T": [(f"t{i:04d}", "1") for i in range(1, n_steps + 1)]}),
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
    the derived frame the predecessor period's — a difference independent of new_stepduration.)"""
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
    assert chain == {"rp_inter_period_balance[battery,t0025,t0001]", "rp_inter_period_balance[battery,t0049,t0025]"}
