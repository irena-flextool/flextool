"""Coarse ``new_stepduration`` groups (Q1: #11, #12, #33, #36, #8 and the c15b found defects).

Owner decision Q1: every flow stays on the solve's step; a node of a group whose ``new_stepduration`` is longer
than the solve's step is balanced once per coarse step by summing its fine balance terms (``nodeBalanceCoarse_eq``),
and a coarse storage node keeps one state per coarse step (``stateConstantCoarse_eq``).  Online status, reserves
and angles stay on the solve's step too; listing a unit or connection in a coarse group has no effect on time
resolution (Q-c15-A default, warned where it would have mattered).

Fixtures: ``tests/fixtures/lh2_three_region.json`` (h2_* / lh2_* in the 24-h ``daily_group``, elec_* hourly),
``tests.json`` (battery in an added 24-h group), ``case14_dc_power_flow.json`` (bus_14 in an added 2-h group),
``stochastics.json``, ``lh2_three_region_rp_invest.json``, each with an override alternative built from JSON under
the pytest tmp dir.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model import _self_discharge
from flextool.engine.model._pdt_join import compute_nodeState_dt
from tests.engine._lp_rows import coef, row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, last_stochastic, make_db, run, utf8

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"
LH2 = FIX / "lh2_three_region.json"
LH2_RP = FIX / "lh2_three_region_rp_invest.json"
DC = FIX / "case14_dc_power_flow.json"
STOCH = FIX / "stochastics.json"
BATTERY = ["init", "west", "wind", "coal", "battery"]
DAILY = [["group", "daily", None], ["group__node", ["daily", "battery"], None]]


def _chain(fixture: Path, scenario: str) -> list[str]:
    data = json.loads(fixture.read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


def _run(tmp_path_factory, alt, values, chain, entities=None, fixture_json=None) -> dict:
    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities, **kw)
    return run(tmp_path_factory, url, scen)


def _run_wf(tmp_path_factory, alt, values, chain, entities=None, fixture_json=None) -> tuple[dict, Path]:
    """As ``_run``, also returning the work folder (outputs under ``output_raw/``)."""
    from flextool.engine import run_chain_from_db

    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities, **kw)
    wf = tmp_path_factory.mktemp(f"{alt}_w")
    return run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True), wf


def _last(steps: dict):
    """The last solve, not an engine-added realization solve after a stochastic one."""
    return last_stochastic(steps)


def _f(x) -> list:
    return b64(float(x), "float")


def _s(x) -> list:
    return b64(x, "str")


def _pmap(values: dict, index_name: str = "period") -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}, "map")


def _arr(*xs) -> list:
    return b64({"value_type": "str", "data": list(xs)}, "array")


def _lh2(tmp_path_factory, alt, values=(), entities=None) -> dict:
    return _run(tmp_path_factory, alt, list(values), _chain(LH2, "lh2_three_region"), entities, LH2)


def _steps_of(fd, node: str, period: str | None = None) -> dict[str, list[str]]:
    """coarse step b_first -> its fine steps, in order, of ``node`` (in ``period`` when given)."""
    ncs = utf8(fd.node_coarse_step).filter(pl.col("n") == node)
    if period is not None:
        ncs = ncs.filter(pl.col("d") == period)
    out: dict[str, list[str]] = {}
    for _n, _d, b, t in ncs.iter_rows():
        out.setdefault(b, []).append(t)
    return out


# ---------------------------------------------------------------------------
# lh2: arcs between hourly and daily nodes conserve energy (#11), own arcs only (#8), explicit units (#36)


def test_mixed_arc_conserves_energy(tmp_path_factory) -> None:
    """electrolyser_A (elec_A hourly -> h2_A daily, efficiency 0.7): every hour's forward flow is in h2_A's daily
    row with +unitsize·Δt and in elec_A's hourly row with −unitsize/0.7·Δt.  Before (#11 b): the daily row held
    only t0001 with the 24-h weight (−6000) and the other 23 hours created hydrogen from free electricity."""
    step = _last(_lh2(tmp_path_factory, "c_mixed"))
    pb = build(step.flex_data)
    coarse = row_terms(pb, "nodeBalanceCoarse_eq", "h2_A,y2030,")
    fine = row_terms(pb, "nodeBalance_eq", "elec_A,y2030,")
    assert "nodeBalanceCoarse_eq[h2_A,y2030,t0001]" in coarse and not row_terms(pb, "nodeBalance_eq", "h2_A,")
    day = coarse["nodeBalanceCoarse_eq[h2_A,y2030,t0001]"]
    for h in range(1, 25):
        col = f"v_flow[electrolyser_A,elec_A,h2_A,y2030,t{h:04d}]"
        assert day[col] == pytest.approx(250.0)
        assert fine[f"nodeBalance_eq[elec_A,y2030,t{h:04d}]"][col] == pytest.approx(-250.0 / 0.7)
    assert not any("t0025" in c for c in day if c.startswith("v_flow"))


def test_both_coarse_arc_every_step(tmp_path_factory) -> None:
    """liquefier_A (h2_A daily -> lh2_A daily) and pipe_AB (lh2_A <-> lh2_B daily): every fine flow column is in
    both nodes' daily rows.  Before (#11 a): only the coarse step's column entered, ×24; the other 23 hours were
    free columns that costs and outputs still read."""
    pb = build(_last(_lh2(tmp_path_factory, "c_both")).flex_data)
    h2 = row_terms(pb, "nodeBalanceCoarse_eq", "h2_A,y2030,t0025")["nodeBalanceCoarse_eq[h2_A,y2030,t0025]"]
    lh2 = row_terms(pb, "nodeBalanceCoarse_eq", "lh2_A,y2030,t0025")["nodeBalanceCoarse_eq[lh2_A,y2030,t0025]"]
    for h in range(25, 49):
        liq = f"v_flow[liquefier_A,h2_A,lh2_A,y2030,t{h:04d}]"
        assert h2[liq] < 0 and lh2[liq] > 0
        assert f"v_flow[pipe_AB,lh2_A,lh2_B,y2030,t{h:04d}]" in lh2


def test_block_row_own_arcs(tmp_path_factory) -> None:
    """liquefier_A with a second output lh2_B (#8): lh2_A's daily row holds only arcs into or out of lh2_A, lh2_B's
    only those of lh2_B.  Before: each row held the output arcs of both nodes."""
    pb = build(_last(_lh2(tmp_path_factory, "c_twosink", [
        ["unit__outputNode", ["liquefier_A", "lh2_B"], "capacity_max_coeff", _f(0.5)],
        ["unit__outputNode", ["liquefier_A", "lh2_A"], "capacity_max_coeff", _f(0.5)]],
        [["unit__outputNode", ["liquefier_A", "lh2_B"], None]])).flex_data)
    for node, other in (("lh2_A", "lh2_B"), ("lh2_B", "lh2_A")):
        row = row_terms(pb, "nodeBalanceCoarse_eq", f"{node},y2030,t0001")[f"nodeBalanceCoarse_eq[{node},y2030,t0001]"]
        liq = [c for c in row if c.startswith("v_flow[liquefier_A,")]
        assert liq and all(f",{node},y2030," in c for c in liq)
        assert not any(f"liquefier_A,{other}," in c or f",{other},y2030" in c for c in liq)


def test_explicit_coarse_unit_fine_side(tmp_path_factory, caplog, monkeypatch) -> None:
    """elz_u (elec_A hourly -> h2_A daily) listed in daily_group (#36): its flow enters elec_A's hourly balance at
    every hour and h2_A's daily balance; a warning says the listing has no effect on time resolution.  Before:
    no elec_A term at all (block-compatibility filter) — hydrogen from free electricity."""
    from flextool.engine.preprocess.params import _coarse

    monkeypatch.setattr(_coarse, "_WARNED", set())      # the warning is given once per process
    caplog.set_level(logging.WARNING)
    step = _last(_lh2(tmp_path_factory, "c_elz", [
        ["unit", "elz_u", "efficiency", _f(0.7)], ["unit", "elz_u", "existing", _f(300.0)],
        ["unit", "elz_u", "conversion_method", _s("constant_efficiency")]],
        [["unit", "elz_u", None], ["unit__inputNode", ["elz_u", "elec_A"], None],
         ["unit__outputNode", ["elz_u", "h2_A"], None], ["group__unit", ["daily_group", "elz_u"], None]]))
    pb = build(step.flex_data)
    fine = row_terms(pb, "nodeBalance_eq", "elec_A,y2030,")
    for h in range(1, 169):
        row = fine[f"nodeBalance_eq[elec_A,y2030,t{h:04d}]"]
        assert coef(row, f"v_flow[elz_u,elec_A,h2_A,y2030,t{h:04d}]") < 0
    day = row_terms(pb, "nodeBalanceCoarse_eq", "h2_A,y2030,t0001")["nodeBalanceCoarse_eq[h2_A,y2030,t0001]"]
    assert sum(1 for c in day if c.startswith("v_flow[elz_u,")) == 24
    assert any("'elz_u' is listed in group 'daily_group'" in r.message and "elec_A" in r.message
               for r in caplog.records)


def test_coarse_unit_online_every_step(tmp_path_factory) -> None:
    """liquefier_A (daily nodes) with linear online (#12): one v_online column per hour, every maxFlow_online row
    bounds the flow by its own hour's online capacity, and the start-up at t0025 lags t0024.  Before: v_online only
    at the 7 coarse steps, maxFlow_online read 0 in 161 hours (v_flow ≤ 0) and every day counted as a start-up."""
    step = _last(_lh2(tmp_path_factory, "c_uc", [
        ["unit", "liquefier_A", "startup_method", _s("linear")], ["unit", "liquefier_A", "min_load", _f(0.3)],
        ["unit", "liquefier_A", "virtual_unitsize", _f(50.0)]]))
    pb = build(step.flex_data)
    m = pb.canonicalise()
    assert sum(1 for c in m.col_names if c.startswith("v_online_linear[liquefier_A,")) == 168
    rows = row_terms(pb, "maxFlow_online_linear", "liquefier_A,")
    assert len(rows) == 168
    for name, terms in rows.items():
        t = name[:-1].split(",")[-1]
        assert coef(terms, f"v_online_linear[liquefier_A,y2030,{t}]") < 0
    start = row_terms(pb, "online_change_linear", "liquefier_A,y2030,t0025")
    terms = start["online_change_linear[liquefier_A,y2030,t0025]"]
    assert "v_online_linear[liquefier_A,y2030,t0024]" in terms


def test_coarse_outputs_not_expanded(tmp_path_factory) -> None:
    """The written v_flow of liquefier_A and pipe_AB (daily nodes) equals the LP's hourly values; the coarse storage
    state is constant within each day.  Before: the coarse step's value was copied to every hour (output
    expansion)."""
    steps, wf = _run_wf(tmp_path_factory, "c_out", [], _chain(LH2, "lh2_three_region"), None, LH2)
    step = _last(steps)
    out = pl.read_parquet(next((wf / "output_raw").glob("v_flow__*.parquet")))
    flows = utf8(step.solution.value("v_flow"))
    for p, src, snk in (("liquefier_A", "h2_A", "lh2_A"), ("pipe_AB", "lh2_A", "lh2_B")):
        lp = flows.filter((pl.col("p") == p) & (pl.col("source") == src) & (pl.col("sink") == snk)).sort("t")
        col = next(c for c in out.columns if p in c and src in c and snk in c)
        assert out.height == lp.height
        assert out[col].to_list() == pytest.approx(lp["value"].to_list(), abs=1e-9)
    state = utf8(step.solution.value("v_state")).filter(pl.col("n") == "lh2_A").sort("t")["value"].to_list()
    for day in range(7):
        assert max(state[24 * day:24 * day + 24]) - min(state[24 * day:24 * day + 24]) < 1e-7


def test_rp_coarse_rows_start_from_rp_state(tmp_path_factory) -> None:
    """lh2 representative periods (bind_within_period_blended_weights, daily lh2 nodes): each representative
    day's first coarse row carries the free start state v_state_rp_start, and the coarse steps restart at each
    representative block."""
    step = _last(_run(tmp_path_factory, "c_rp", [], _chain(LH2_RP, "lh2_three_region_rp_invest"),
                      fixture_json=LH2_RP))
    fd = step.flex_data
    assert sorted(_steps_of(fd, "lh2_A", "y2030")) == ["t0001", "t0025"]
    assert sorted(_steps_of(fd, "lh2_A", "y2040")) == ["t0049", "t0073"]
    pb = build(fd)
    rows = row_terms(pb, "nodeBalanceCoarse_eq", "lh2_A,y2030,")
    for b in ("t0001", "t0025"):
        assert f"v_state_rp_start[lh2_A,y2030,{b}]" in rows[f"nodeBalanceCoarse_eq[lh2_A,y2030,{b}]"]


# ---------------------------------------------------------------------------
# DC power flow bus on a coarse group (#33); a coarse group without storage


def _dc(tmp_path_factory, alt, storage: bool):
    four = _pmap({f"t000{i}": 1.0 for i in range(1, 5)}, "time")
    ents = [["timeline", "tl_dispatch", None], ["group", "blk", None], ["group__node", ["blk", "bus_14"], None]]
    vals = [["timeline", "tl_dispatch", "timestep_duration", four],
            ["timeset", "ts_dispatch", "timeset_duration", _pmap({"t0001": 4.0}, "time")],
            ["node", "bus_14", "inflow", _pmap({"t0001": -14.9, "t0002": -40.0, "t0003": -14.9, "t0004": -40.0},
                                               "time")],
            ["group", "blk", "new_stepduration", _f(2.0)]]
    if storage:
        vals += [["node", "bus_14", "node_type", _s("storage")], ["node", "bus_14", "existing", _f(100.0)],
                 ["node", "bus_14", "storage_state_start", _f(0.5)],
                 ["node", "bus_14", "storage_start_end_method", _s("fix_start")]]
    return _last(_run(tmp_path_factory, alt, vals, _chain(DC, "dc_opf_test"), ents, DC))


def test_dc_coarse_bus_back_flow(tmp_path_factory) -> None:
    """case14 bus_14 (storage) in a 2-h group: its first coarse row holds v_flow and v_flow_back of both its
    lines at t0001 and t0002 (±unitsize·Δt).  Before (#33): only the two forward t0001 terms, no v_flow_back, so
    bus_14 exported at capacity without its balance seeing it."""
    step = _dc(tmp_path_factory, "c_dcs", True)
    row = row_terms(build(step.flex_data), "nodeBalanceCoarse_eq", "bus_14,")
    terms = row[next(k for k in row if k.endswith(",t0001]"))]
    for line in ("line_13_14", "line_9_14"):
        for t in ("t0001", "t0002"):
            fwd = terms[next(c for c in terms if c.startswith(f"v_flow[{line},") and c.endswith(f",{t}]"))]
            back = terms[next(c for c in terms if c.startswith(f"v_flow_back[{line},") and c.endswith(f",{t}]"))]
            assert fwd == pytest.approx(-back) and fwd != 0


def test_coarse_group_without_storage(tmp_path_factory) -> None:
    """bus_14 a balance node in a 2-h group: two coarse rows (t0001, t0003), no hourly rows.  Before: a coarse
    group did nothing without a storage node in the model (byte-identical to the hourly LP)."""
    step = _dc(tmp_path_factory, "c_dcn", False)
    pb = build(step.flex_data)
    assert sorted(row_terms(pb, "nodeBalanceCoarse_eq", "bus_14,")) == [
        "nodeBalanceCoarse_eq[bus_14,p1,t0001]", "nodeBalanceCoarse_eq[bus_14,p1,t0003]"]
    assert not row_terms(pb, "nodeBalance_eq", "bus_14,")
    lo, hi = row_bounds(pb, "nodeBalanceCoarse_eq", "bus_14,p1,t0001")["nodeBalanceCoarse_eq[bus_14,p1,t0001]"]
    assert lo == pytest.approx(54.9) and hi == pytest.approx(54.9)


# ---------------------------------------------------------------------------
# tests.json battery: coarse relative to the solve's step, rolling, binding methods, two groups


def test_coarse_relative_to_solve_step(tmp_path_factory) -> None:
    """battery in a 24-h group with the solve's new_stepduration = 2: only the battery is coarse, with coarse
    steps t0001 and t0025 of 12 two-hour steps each.  Before: every node of the 2-h solve counted as coarse
    (absolute 1-h threshold), so west and coal_market were balanced daily."""
    step = _last(_run(tmp_path_factory, "c_2h", [["group", "daily", "new_stepduration", _f(24.0)],
                                                ["solve", "y2020_2day_dispatch", "new_stepduration", _f(2.0)]],
                      BATTERY, DAILY))
    fd = step.flex_data
    assert utf8(fd.nodeCoarse)["n"].to_list() == ["battery"]
    steps = _steps_of(fd, "battery")
    assert sorted(steps) == ["t0001", "t0025"] and all(len(v) == 12 for v in steps.values())
    assert row_terms(build(fd), "nodeBalance_eq", "west,")


def test_not_coarse_equals_no_group(tmp_path_factory) -> None:
    """A group whose step equals the solve's step is not coarse: the LP equals the one without the group."""
    a = _last(_run(tmp_path_factory, "c_eq1", [["group", "daily", "new_stepduration", _f(1.0)]], BATTERY, DAILY))
    b = _last(_run(tmp_path_factory, "c_eq0", [], BATTERY))
    assert a.flex_data.nodeCoarse is None
    ma, mb = build(a.flex_data).canonicalise(), build(b.flex_data).canonicalise()
    assert list(ma.row_names) == list(mb.row_names) and list(ma.col_names) == list(mb.col_names)
    assert list(ma.val) == list(mb.val)


def test_group_step_not_multiple_raises(tmp_path_factory) -> None:
    """A coarse group step that is not a multiple of the solve's step (5 h on a 2-h solve) is a configuration
    error.  Before: NotImplementedError inside the block emitter, swallowed by run_model — the coarse handling
    silently disappeared."""
    with pytest.raises(FlexToolConfigError, match="not a multiple"):
        _run(tmp_path_factory, "c_5h", [["group", "daily", "new_stepduration", _f(5.0)],
                                        ["solve", "y2020_2day_dispatch", "new_stepduration", _f(2.0)]],
             BATTERY, DAILY)


def _roll(jump: float) -> list:
    return [["group", "daily", "new_stepduration", _f(24.0)],
            ["solve", "dispatch_fullYear_roll", "period_timeset", _pmap({"p2020": "fullYear"})],
            ["solve", "dispatch_fullYear_roll", "realized_periods", _arr("p2020")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(48.0)],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(jump)]]


def _alpha_beta_of_step(fd, node: str, d: str, b_first: str) -> tuple[float, float]:
    """(α_b, β_b) of the self-discharge of a coarse step: α summed over its fine steps, β at its first step."""
    frame = _self_discharge.loss_coefficients(fd, compute_nodeState_dt(fd)).frame.with_columns(
        pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
    steps = utf8(fd.node_coarse_step).filter((pl.col("n") == node) & (pl.col("d") == d) & (pl.col("b_first") == b_first))
    rows = frame.join(steps.select("n", "d", "t"), on=["n", "d", "t"], how="inner")
    return float(rows["alpha"].sum()), float(rows.filter(pl.col("t") == b_first)["beta"][0])


def test_coarse_rolling(tmp_path_factory) -> None:
    """battery (bind_forward_only) in a 24-h group, rolling 48 h / 24 h over 72 h: every roll has coarse rows, and
    the first coarse row of rolls 2 and 3 hands over the previous roll's state (its bound carries
    roll_continue_state).  Before: rolls never built the coarse frames (the battery was balanced hourly).  A cyclic
    battery takes no carry (#119, tests/engine/test_rolling_cyclic_storage.py)."""
    steps = list(_run(tmp_path_factory, "c_roll", _roll(24.0),
                      BATTERY + ["bind_battery_forward", "fullYear", "dispatch_fullYear_roll"], DAILY).values())
    assert len(steps) == 3
    for i, step in enumerate(steps):
        fd = step.flex_data
        pb = build(fd)
        rows = row_bounds(pb, "nodeBalanceCoarse_eq", "battery,")
        assert rows, f"roll {i + 1}: no coarse rows"
        if i == 0:
            continue
        first = utf8(fd.dt)["t"][0]
        su = float(fd.p_state_unitsize.frame["value"][0])
        carried = float(utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == "battery")["value"][0])
        inflow = (utf8(fd.p_inflow.frame).filter(pl.col("n") == "battery")
                  .join(utf8(fd.node_coarse_step).filter(pl.col("b_first") == first), on=["n", "d", "t"]))
        lo, _hi = rows[f"nodeBalanceCoarse_eq[battery,p2020,{first}]"]
        # The carry is the coarse step's previous state: weighted by (1 − β_b) of its self-discharge.
        _alpha, beta = _alpha_beta_of_step(fd, "battery", "p2020", first)
        assert lo == pytest.approx(-float(inflow["value"].sum()) - (1 - beta) * carried)
        terms = row_terms(pb, "nodeBalanceCoarse_eq", f"battery,p2020,{first}")[
            f"nodeBalanceCoarse_eq[battery,p2020,{first}]"]
        # The state terms telescope to −state_unitsize (end of the coarse step; the start is the carry in the
        # bound) plus the self-discharge, and there is no wrap: only states of the coarse step itself.
        states = {c: v for c, v in terms.items() if c.startswith("v_state[")}
        assert sum(states.values()) < -su * 0.99
        in_step = set(utf8(fd.node_coarse_step).filter((pl.col("n") == "battery") & (pl.col("b_first") == first))["t"])
        assert {c.rstrip("]").split(",")[-1] for c in states} <= in_step
        prev = steps[i - 1]
        last_realized = utf8(prev.flex_data.realized_dispatch)["step"][-1]
        state = utf8(prev.solution.value("v_state")).filter((pl.col("n") == "battery")
                                                             & (pl.col("t") == last_realized))["value"][0]
        assert carried == pytest.approx(state * su, rel=1e-9, abs=1e-9)


def _roll_cut_checks(steps) -> None:
    first = _steps_of(steps[0].flex_data, "battery")
    assert {b: len(ts) for b, ts in first.items()} == {"t0001": 12, "t0013": 24, "t0037": 12}
    second = _steps_of(steps[1].flex_data, "battery")
    assert {b: len(ts) for b, ts in second.items()} == {"t0013": 12, "t0025": 24, "t0049": 12}
    # The state handed over is the state after the realized hours only: roll 1's state at t0012 (end of its first,
    # 12-h coarse step) equals the start state plus the net energy of t0001-t0012.
    fd, sol = steps[0].flex_data, steps[0].solution
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    state = utf8(sol.value("v_state")).filter(pl.col("n") == "battery").sort("t")
    carried = float(utf8(steps[1].flex_data.p_roll_continue_state.frame).filter(pl.col("n") == "battery")["value"][0])
    assert carried == pytest.approx(float(state.filter(pl.col("t") == "t0012")["value"][0]) * su, rel=1e-9)
    assert state.filter(pl.col("t") <= "t0012")["value"].n_unique() == 1


def test_coarse_rolling_cut_at_realized_end(tmp_path_factory) -> None:
    """Rolling 48 h / 12 h with a 24-h group: each roll's coarse steps end at its last realized step (t0012 in
    roll 1: t0001-12, t0013-36, t0037-48), so the state a roll hands over is the state after the realized hours
    only, never one that already includes look-ahead flows (critique 6)."""
    steps = list(_run(tmp_path_factory, "c_roll12", _roll(12.0), BATTERY + ["fullYear", "dispatch_fullYear_roll"],
                      DAILY).values())
    _roll_cut_checks(steps)


def test_coarse_rolling_cut_ignores_output_horizon(tmp_path_factory) -> None:
    """``model.output_horizon`` (write the whole horizon) changes only the output rows: the coarse steps still
    end at each roll's realized end and every roll's LP is the one without the flag (review F1).  Before the
    fix the cut read the output row set, so roll 1 got coarse steps t0001-24 / t0025-48 and handed over a state
    that already included look-ahead hours t0013-t0024."""
    chain = BATTERY + ["fullYear", "dispatch_fullYear_roll"]
    with_flag = list(_run(tmp_path_factory, "c_roll12oh", _roll(12.0) + [
        ["model", "flexTool", "output_horizon", _s("yes")]], chain, DAILY).values())
    _roll_cut_checks(with_flag)
    without = list(_run(tmp_path_factory, "c_roll12no", _roll(12.0), chain, DAILY).values())
    assert len(with_flag) == len(without)
    for a, b in zip(with_flag, without):
        ma, mb = build(a.flex_data).canonicalise(), build(b.flex_data).canonicalise()
        assert list(ma.row_names) == list(mb.row_names) and list(ma.col_names) == list(mb.col_names)
        assert list(ma.val) == list(mb.val)


def test_coarse_storage_binding(tmp_path_factory) -> None:
    """battery bind_forward_only with fix_start in a 24-h group over two periods (ta t0001-24, tb t0025-48): the
    first coarse row carries the start state (no wrap from the last step), and p2025's first coarse row starts
    from p2020's last state.  Before: the block chain was cyclic within each period whatever the binding method,
    and fix_start never reached a coarse node."""
    vals = [["timeset", "ta", "timeline", _s("y2020")], ["timeset", "tb", "timeline", _s("y2020")],
            ["timeset", "ta", "timeset_duration", _pmap({"t0001": 24.0}, "time")],
            ["timeset", "tb", "timeset_duration", _pmap({"t0025": 24.0}, "time")],
            ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "ta", "p2025": "tb"})],
            ["solve", "y2020_2day_dispatch", "realized_periods", _arr("p2020", "p2025")],
            ["group", "daily", "new_stepduration", _f(24.0)]]
    step = _last(_run(tmp_path_factory, "c_fwd", vals, BATTERY + ["bind_battery_forward"],
                      DAILY + [["timeset", "ta", None], ["timeset", "tb", None]]))
    pb = build(step.flex_data)
    rows = row_terms(pb, "nodeBalanceCoarse_eq", "battery,")
    fd = step.flex_data
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    first = rows["nodeBalanceCoarse_eq[battery,p2020,t0001]"]
    states = {c: v for c, v in first.items() if c.startswith("v_state[")}
    # No wrap: only p2020 states; telescoped they leave −state_unitsize on the day (start − end of the day), plus
    # the self-discharge α_b of the day on its state (the start is the fix_start energy in the bound).
    assert all(c.startswith("v_state[battery,p2020,") for c in states)
    alpha, beta = _alpha_beta_of_step(fd, "battery", "p2020", "t0001")
    # Closed form (oracle batch B, #152): the battery loses 1e-4 per hour, so the 24-h coarse step has
    # h_b = 24·ln(1.0001) = 0.00239988, α_b = h_b/(1 − e^{−h_b}) − 1 = 0.00120042, β_b = 1 − h_b/(e^{h_b} − 1)
    # = 0.00119946 (α_b + β_b = h_b).
    h_b = 24 * math.log1p(1e-4)
    assert beta == pytest.approx(1 - h_b / math.expm1(h_b), rel=1e-9)
    assert beta == pytest.approx(0.00119946, rel=1e-6)
    assert alpha == pytest.approx(h_b / -math.expm1(-h_b) - 1, rel=1e-9)
    assert alpha == pytest.approx(0.00120042, rel=1e-6)
    assert sum(states.values()) == pytest.approx(-su * (1 + alpha), rel=1e-9)
    # fix_start: the start energy state_start × existing is in the bound (the summed −inflow minus it).
    lo, hi = row_bounds(pb, "nodeBalanceCoarse_eq", "battery,p2020,t0001")["nodeBalanceCoarse_eq[battery,p2020,t0001]"]
    inflow = (float(utf8(fd.p_inflow.frame).filter((pl.col("n") == "battery") & (pl.col("d") == "p2020"))
                    ["value"].sum()) if fd.p_inflow is not None else 0.0)
    start = float(utf8((fd.p_state_start * fd.p_state_existing_capacity).frame)
                  .filter((pl.col("n") == "battery") & (pl.col("d") == "p2020"))["value"][0])
    assert start > 0
    assert lo == pytest.approx(-inflow - (1 - beta) * start, rel=1e-9) and hi == pytest.approx(lo)
    # p2025 starts from p2020's last state (forward chain), +state_unitsize·(1 − β_b).
    second = rows["nodeBalanceCoarse_eq[battery,p2025,t0025]"]
    _a2, beta2 = _alpha_beta_of_step(fd, "battery", "p2025", "t0025")
    assert second["v_state[battery,p2020,t0024]"] == pytest.approx(su * (1 - beta2))


def test_two_coarse_groups(tmp_path_factory) -> None:
    """battery in a 24-h group and ev_battery in a 48-h group: each node has its own coarse steps and no row mixes
    them.  Before: both groups shared one period_block_time, so each node got the other's rows."""
    step = _last(_run(tmp_path_factory, "c_two", [["group", "daily", "new_stepduration", _f(24.0)],
                                                 ["group", "twoday", "new_stepduration", _f(48.0)]],
                      BATTERY + ["ev"], DAILY + [["group", "twoday", None],
                                                 ["group__node", ["twoday", "ev_battery"], None]]))
    fd = step.flex_data
    assert sorted(_steps_of(fd, "battery")) == ["t0001", "t0025"]
    assert sorted(_steps_of(fd, "ev_battery")) == ["t0001"]
    rows = row_terms(build(fd), "nodeBalanceCoarse_eq")
    assert sorted(rows) == ["nodeBalanceCoarse_eq[battery,p2020,t0001]", "nodeBalanceCoarse_eq[battery,p2020,t0025]",
                            "nodeBalanceCoarse_eq[ev_battery,p2020,t0001]"]
    for name, terms in rows.items():
        node = name.split("[")[1].split(",")[0]
        assert all(f"[{node}," in c for c in terms if c.startswith(("v_state[", "vq_state_")))


def test_coarse_stochastic(tmp_path_factory) -> None:
    """stochastics.json 2-day dispatch with hydro_reservoir in a 24-h group: coarse rows in every period of dt
    (realized and forecast branches); each branch's first coarse row starts from the trunk's state; the
    non-anticipativity rows are those of the hourly model."""
    chain = _chain(STOCH, "2_day_stochastic_dispatch")
    grp = [["group", "daily", None], ["group__node", ["daily", "hydro_reservoir"], None]]
    coarse = _last(_run(tmp_path_factory, "c_sto", [["group", "daily", "new_stepduration", _f(24.0)]], chain, grp,
                        STOCH))
    fine = _last(_run(tmp_path_factory, "c_sto0", [], chain, None, STOCH))
    fd = coarse.flex_data
    periods = set(utf8(fd.dt)["d"].to_list())
    assert set(utf8(fd.node_coarse_step)["d"].to_list()) == periods and len(periods) > 1
    pb, pf = build(fd), build(fine.flex_data)
    rows = row_terms(pb, "nodeBalanceCoarse_eq", "hydro_reservoir,")
    lag = {(d, t): (dp, tp) for d, t, dp, tp in
           utf8(fd.dtttdt).select("d", "t", "d_previous", "t_previous_within_solve").iter_rows()}
    for d in periods - {utf8(fd.dt)["d"][0]}:
        b = min(t for dd, t in utf8(fd.dt).iter_rows() if dd == d)
        dp, tp = lag[(d, b)]
        if dp != d:
            assert f"v_state[hydro_reservoir,{dp},{tp}]" in rows[f"nodeBalanceCoarse_eq[hydro_reservoir,{d},{b}]"]
    na = lambda p: sorted(n for n in p.canonicalise().row_names if n.startswith("non_anticipativity"))  # noqa: E731
    assert na(pb) == na(pf)


# ---------------------------------------------------------------------------
# prices: coarse node prices have the sign of nodeBalance_eq prices; intraperiod blocks too


def test_coarse_price_sign(tmp_path_factory) -> None:
    """west hourly with coal at a constant marginal cost (capacity never binding); node h2 in a 24-h group served
    only by an electrolyser (efficiency 0.7): the h2 price is the west price / 0.7 at every step — the same sign
    as the west price.  Before: the coarse row's dual was extracted with the sign convention of nodeBalance_eq
    although the row was written negated (and hydrogen was free)."""
    ents = [["node", "h2", None], ["unit", "elz", None], ["unit__inputNode", ["elz", "west"], None],
            ["unit__outputNode", ["elz", "h2"], None], ["group", "daily", None], ["group__node", ["daily", "h2"], None]]
    vals = [["node", "h2", "node_type", _s("balance")], ["node", "h2", "inflow", _f(-10.0)],
            ["node", "h2", "penalty_up", _f(10000.0)], ["node", "h2", "penalty_down", _f(10000.0)],
            ["unit", "elz", "efficiency", _f(0.7)], ["unit", "elz", "existing", _f(100.0)],
            ["unit", "elz", "conversion_method", _s("constant_efficiency")],
            ["unit", "coal_plant", "existing", _f(5000.0)],
            ["group", "daily", "new_stepduration", _f(24.0)]]
    # The battery alternative only puts a storage node in the model (before the fix a coarse group needed one).
    _steps, wf = _run_wf(tmp_path_factory, "c_price", vals, ["init", "west", "coal", "battery"], ents)
    dual = pl.read_parquet(next((wf / "output_raw").glob("v_dual_node_balance__*.parquet")))
    west, h2 = dual["west"].to_list(), dual["h2"].to_list()
    assert len(h2) == 48 and len(set(round(w, 6) for w in west)) == 1 and west[0] != 0
    for w, h in zip(west, h2):
        assert h == pytest.approx(w / 0.7, rel=1e-6)


def test_intraperiod_price_sign(tmp_path_factory) -> None:
    """battery bind_intraperiod_blocks on 5weeks: its block price has the sign of the price of the electricity
    node it trades with (critique 11).  The block row has the sign of nodeBalance_eq (σ = +1).  Before:
    nodeBalanceBlock_eq was written as the negation of nodeBalance_eq but extracted with the nodeBalance_eq sign,
    so the block prices had the opposite sign."""
    _steps, wf = _run_wf(tmp_path_factory, "c_ipb", [], ["init", "west", "wind", "coal", "battery", "5weeks",
                                                         "bind_battery_intraperiod_blocks"])
    dual = pl.read_parquet(next((wf / "output_raw").glob("v_dual_node_balance__*.parquet")))
    bat, west = dual["battery"].to_list(), dual["west"].to_list()
    assert any(abs(b) > 1e-6 for b in bat)
    for b, w in zip(bat, west):
        assert b * w >= -1e-9


# ---------------------------------------------------------------------------
# nested fix_price from a coarse parent (critique 7)


def test_nested_fix_price_from_coarse_parent(tmp_path_factory) -> None:
    """Nested invest (24 h) -> storage (6 h) -> rolling dispatch with battery fix_price and the battery in a 12-h
    group (not coarse in the 24-h solve, coarse in the 6-h one): the storage solve hands the dispatch the battery
    price of each fix step = its coarse row's dual / s / (tw[d,t] · inflation[d] / period share[d] · bw[d,t])
    (critique 7; tw and bw are per fine step, read from the solve, 1 where the solve has none).  Before: the
    storage solve had no nodeBalance_eq row for the battery, so no price reached the child."""
    from flextool.engine.solve._orchestration import _LEGACY_DEFAULT_OBJECTIVE_SCALE as s_obj
    vals = [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
            ["node", "battery", "storage_nested_fix_method", _s("fix_price")],
            ["group", "half_day", "new_stepduration", _f(12.0)]]
    steps = _run(tmp_path_factory, "c_nfp", vals, _chain(FIX / "tests.json",
                                                          "multi_fullYear_battery_nested_24h_invest_one_solve"),
                 [["group", "half_day", None], ["group__node", ["half_day", "battery"], None]])
    assert steps["invest_24h"].flex_data.nodeCoarse is None          # 12 h < 24 h: not coarse
    parent = steps["storage_fullYear_6h"]
    fd = parent.flex_data
    price = parent.handoff.fix_storage_price
    assert price is not None and price.filter(pl.col("node") == "battery").height > 0
    dual = {tuple(k.split(",")): v for k, v in parent.solution.constraint_dual("nodeBalanceCoarse_eq").iter_rows()}
    step_of = {(n, d, t): b for n, d, b, t in utf8(fd.node_coarse_step).iter_rows()}
    infl = dict(utf8(fd.p_inflation_op.frame).select("d", "value").iter_rows())
    share = dict(utf8(fd.p_period_share.frame).select("d", "value").iter_rows())

    def per_step(param) -> dict[tuple[str, str], float]:
        if param is None:
            return {}
        frame = utf8(getattr(param, "frame", param))
        return {(d, t): w for d, t, w in frame.select("d", "t", "value").iter_rows()}

    tw = per_step(getattr(fd, "p_timestep_weight", None))
    bw = per_step(getattr(fd, "pdt_branch_weight", None))
    nonzero = 0
    for n, d, t, v in price.filter(pl.col("node") == "battery").select(
            "node", "period", "step", "p_fix_storage_price").iter_rows():
        b = step_of[(n, d, t)]
        w = tw.get((d, t), 1.0) * infl[d] / share[d] * bw.get((d, t), 1.0)
        assert v == pytest.approx(dual[(n, d, b)] / s_obj / w, rel=1e-9, abs=1e-9)
        nonzero += abs(v) > 0
    assert nonzero > 0


def test_nested_fix_price_reaches_dispatch_rolls(tmp_path_factory) -> None:
    """Nested invest (24 h) -> storage (6 h) -> rolling dispatch (horizon 36 h, jump 24 h) with battery fix_price: a
    roll's reference price of a period is the storage solve's price at the parent step matched to the period's last
    step in the roll (roll 2: p2020, roll 11: p2035; 47.5 here).  The objective values the end state only in the
    roll's terminal period (period_last): roll 11 has the term on v_state[battery,p2035,t0072], roll 2 has none on
    p2020's last step (item 134).  Before: the matching map reached only the disk, the reference-price derive (which
    reads the Provider) saw none, and every roll got 0."""
    vals = [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
            ["node", "battery", "storage_nested_fix_method", _s("fix_price")]]
    steps = _run(tmp_path_factory, "c_nfp_roll", vals, _chain(FIX / "tests.json",
                                                               "multi_fullYear_battery_nested_24h_invest_one_solve"))
    price = {(n, d, t): v for n, d, t, v in utf8(steps["storage_fullYear_6h"].handoff.fix_storage_price)
             .select("node", "period", "step", "p_fix_storage_price").iter_rows()}

    def check(roll: int, period: str) -> str:
        fd = steps[f"dispatch_fullYear_roll_roll_{roll}"].flex_data
        ref = dict(utf8(fd.p_storage_state_reference_price.frame).filter(pl.col("n") == "battery")
                   .select("d", "value").iter_rows())
        last = utf8(fd.nodeState_last_dt).filter((pl.col("n") == "battery") & (pl.col("d") == period))["t"][0]
        upper = utf8(fd.dtt_timeline_matching).filter((pl.col("d") == period) & (pl.col("t") == last))["t_upper"][0]
        want = float(price[("battery", period, upper)])
        assert want != 0.0
        assert ref[period] == pytest.approx(want, rel=1e-12), (roll, period)
        return last

    last11 = check(11, "p2035")
    last2 = check(2, "p2020")
    m11 = build(steps["dispatch_fullYear_roll_roll_11"].flex_data).canonicalise()
    m2 = build(steps["dispatch_fullYear_roll_roll_2"].flex_data).canonicalise()
    obj11 = {nm: float(m11.col_obj[i]) for i, nm in enumerate(m11.col_names)}
    obj2 = {nm: float(m2.col_obj[i]) for i, nm in enumerate(m2.col_names)}
    assert obj11[f"v_state[battery,p2035,{last11}]"] < 0.0
    assert obj2[f"v_state[battery,p2020,{last2}]"] == 0.0


def test_group_step_relative_to_solve_step(caplog, monkeypatch) -> None:
    """Coarse means longer than the solve's step: a group step equal to the solve's step is not coarse (silently),
    a shorter one is not coarse with a warning, a longer one that is not a multiple raises."""
    from flextool.engine.preprocess.params import _coarse

    monkeypatch.setattr(_coarse, "_WARNED", set())
    caplog.set_level(logging.WARNING)
    groups = {"n1": "eq", "n2": "short", "n3": "long"}
    steps = {"default": 2.0, "eq": 2.0, "short": 1.0, "long": 6.0}
    assert _coarse.coarse_groups(groups, steps, 2.0) == {"long": 6.0}
    assert [r.message for r in caplog.records if "'short'" in r.message]
    assert not [r.message for r in caplog.records if "'eq'" in r.message]
    with pytest.raises(FlexToolConfigError, match="not a multiple"):
        _coarse.coarse_groups({"n": "odd"}, {"odd": 5.0}, 2.0)
