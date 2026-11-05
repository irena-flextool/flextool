"""Coarse processes with online variables: units online, start-ups and shut-downs once per coarse step.

Owner decision 2026-11-04 ("Coarse flows for units with online variables / reserves"): a unit (or connection)
whose every balance node is coarse keeps one flow per coarse step also when it has online variables; its units
online, start-ups and shut-downs are decided once per coarse step (the start / stop at the step's start), a
minimum up / down time counts whole coarse steps, the online capacity and minimum-load rows are averaged over the
step, and the outputs show units online on every solve step and each start-up / shut-down at its step's first
solve step.  An online process with ramp limits (start-up / shut-down capacity reduction, online ramps) or with
reserves keeps the solve's step, and so does one whose forecast tie or realization pin covers only part of a
coarse step.

Fixtures: ``tests/fixtures/lh2_three_region.json`` (liquefier_A h2_A -> lh2_A between daily nodes, one week of
hourly steps), ``tests/fixtures/stochastics.json`` and ``tests.json`` (hand-built model), with override
alternatives built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, last_stochastic, make_db, run, utf8, without_realization
from tests.engine.test_coarse_flows import (
    LH2_COARSE, LH2_RP, STOCH, _chain, _coarse_procs, _f, _forecasts, _keys, _lh2, _s, _stoch_values,
    _STOCH_ENTITIES, _tmap_lh2, _total_objective,
)

pytestmark = pytest.mark.solver

_DAYS = [f"t{24 * k + 1:04d}" for k in range(7)]
# liquefier_A as the lp_snapshot fixture coarse_uc: linear online, min_load 0.3, unitsize 50 (4 units of 200 MW).
_UC = [["unit", "liquefier_A", "startup_method", _s("linear")], ["unit", "liquefier_A", "min_load", _f(0.3)],
       ["unit", "liquefier_A", "virtual_unitsize", _f(50.0)]]
# ... with a start-up cost, a minimum up / down time longer than a day, and a liquid hydrogen demand at lh2_A that
# alternates between 10 MW and 150 MW from day to day (10 MW on the first) with almost no storage there, so the
# units start and stop.
_UC_FULL = _UC + [["unit", "liquefier_A", "startup_cost", _f(50.0)],
                  ["node", "lh2_A", "existing", _f(1.0)],
                  ["node", "lh2_A", "inflow", _tmap_lh2(lambda h: -10.0 if (h // 24) % 2 == 0 else -150.0)],
                  ["unit", "liquefier_A", "minimum_time_method", _s("both")],
                  ["unit", "liquefier_A", "min_uptime", _f(30.0)], ["unit", "liquefier_A", "min_downtime", _f(30.0)]]


def _cols(pb, prefix: str) -> list[str]:
    return [c for c in pb.canonicalise().col_names if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# Structure


def test_online_unit_is_coarse_with_one_column_per_day(tmp_path_factory) -> None:
    """coarse_uc: liquefier_A (linear online, min_load 0.3) is coarse.  Its units online, start-ups and shut-downs
    have 7 columns (one per day, at the day's first hour), online_change / maxOnline / maxStartup 7 rows, the online
    capacity and minimum load one averaged row per day (maxFlow_online_linear_coarse, minFlow_minload_linear_coarse)
    and no hourly row; its v_flow has 7 columns."""
    fd = last_stochastic(_lh2(tmp_path_factory, "co_struct", _UC)).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    pb = build(fd)
    for var in ("v_online_linear", "v_startup_linear", "v_shutdown_linear"):
        cols = _cols(pb, f"{var}[liquefier_A,")
        assert [_keys(c)[-1] for c in cols] == _DAYS, var
    assert [_keys(c)[-1] for c in _cols(pb, "v_flow[liquefier_A,h2_A,lh2_A,")] == _DAYS
    for fam in ("online_change_linear", "maxOnline_linear", "maxStartup_linear", "minFlow_minload_linear_coarse"):
        assert sorted(_keys(r)[-1] for r in row_terms(pb, fam, "liquefier_A,")) == _DAYS, fam
    assert len(row_terms(pb, "maxFlow_online_linear_coarse", "liquefier_A,h2_A,lh2_A,")) == 7
    assert not row_terms(pb, "maxFlow_online_linear", "liquefier_A,")
    assert not row_terms(pb, "minFlow_minload_linear", "liquefier_A,")
    # The day-2 row: online − (online of day 1) − startup + shutdown = 0; day 1 wraps to day 7 (stretch start).
    rows = row_terms(pb, "online_change_linear", "liquefier_A,y2030,t0025")
    assert rows["online_change_linear[liquefier_A,y2030,t0025]"] == pytest.approx({
        "v_online_linear[liquefier_A,y2030,t0025]": 1.0, "v_online_linear[liquefier_A,y2030,t0001]": -1.0,
        "v_startup_linear[liquefier_A,y2030,t0025]": -1.0, "v_shutdown_linear[liquefier_A,y2030,t0025]": 1.0})
    first = row_terms(pb, "online_change_linear", "liquefier_A,y2030,t0001")["online_change_linear[liquefier_A,y2030,t0001]"]
    assert first["v_online_linear[liquefier_A,y2030,t0145]"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Exact: the coarse model == the fine model with every coarse-process value tied over each coarse step

_LEVELS = ("v_flow", "v_flow_back", "v_online_linear", "v_online_integer")
_EVENTS = ("v_startup_linear", "v_startup_integer", "v_shutdown_linear", "v_shutdown_integer")


def _add_ties(pb, fd) -> None:
    """In the fine LP ``pb`` of ``fd`` (every process at the solve's step): for each coarse process of
    ``fd.process_coarse_step`` and each interior fine step t of a coarse step b, ``x[…,t] = x[…,b]`` for a level x
    (flows, units online) and ``x[…,t] = 0`` for an event x (start-ups, shut-downs)."""
    from polar_high.engine import Var

    steps = getattr(fd, "process_coarse_step", None)
    if steps is None or steps.height == 0:
        return
    st = steps.select(*(pl.col(c).cast(pl.Utf8).alias(f"_{c}") for c in ("p", "d", "b_first", "t")))
    for name in (*_LEVELS, *_EVENTS):
        var = pb._vars.get(name)
        if var is None:
            continue
        dims = var.dims
        fr = var.frame.with_columns(*(pl.col(c).cast(pl.Utf8).alias(f"_{c}") for c in dims))
        inner = (fr.join(st, on=["_p", "_d", "_t"], how="inner")
                 .filter(pl.col("_t") != pl.col("_b_first")))
        if inner.height == 0:
            continue
        over = inner.select(dims)
        lhs = {"a": Var(name, dims, inner.select(*dims, "col_id"))}
        if name in _LEVELS:
            first = fr.select(*[f"_{c}" for c in dims], pl.col("col_id").alias("__first"))
            keys = [f"_{c}" for c in dims[:-1]]
            paired = inner.join(first.rename({"_t": "_b_first"}), on=[*keys, "_b_first"], how="inner")
            assert paired.height == inner.height, name
            lhs["b"] = -Var(name, dims, paired.select(*dims, pl.col("__first").alias("col_id")))
        pb.add_cstr(f"tie_{name}", over=over, sense="==", lhs_terms=lhs, rhs_terms={})


def _tied_fine(monkeypatch) -> None:
    """Build every solve fine (the model ignores processCoarse) with the coarse steps tied (:func:`_add_ties`).
    The tie rows are a test device outside the autoscale registry: not strict there."""
    import flextool.engine.model.core as core
    from flextool.engine.model import _coarse_flows

    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    monkeypatch.setattr(_coarse_flows, "prepare", lambda b: None)
    original = core.build_flextool

    def build_tied(pb, data, *args, **kwargs):
        out = original(pb, data, *args, **kwargs)
        _add_ties(pb, data)
        return out

    monkeypatch.setattr(core, "build_flextool", build_tied)


_ROLLING = [["solve", "lh2_week", "solve_mode", _s("rolling_window")],
            ["solve", "lh2_week", "rolling_solve_horizon", _f(72.0)],
            ["solve", "lh2_week", "rolling_solve_jump", _f(48.0)]]
# Integer units online (startup_method binary in the fixture's schema version; migrated to integer).
_INTEGER = [["unit", "liquefier_A", "startup_method", _s("binary")],
            ["solve", "lh2_week", "solver_mip_gap", _f(1e-9)]]
# min_load_efficiency: the section input (units online × section) in the conversion and the balance.
_SECTION = [["unit", "liquefier_A", "conversion_method", _s("min_load_efficiency")],
            ["unit", "liquefier_A", "efficiency_at_min_load", _f(0.6)]]


def _uc_case(tmp_path_factory, case: str, alt: str) -> dict:
    if case == "rp":
        url, scen = make_db(tmp_path_factory, alt=alt, values=_UC_FULL, fixture_json=LH2_RP,
                            base_chain=_chain(LH2_RP, "lh2_three_region_rp_invest"))
        return run(tmp_path_factory, url, scen)
    extra = {"single": [], "rolling": _ROLLING, "integer": _INTEGER, "section": _SECTION}[case]
    base = [v for v in _UC_FULL if case != "integer" or v[2] != "startup_method"]
    return _lh2(tmp_path_factory, alt, base + extra)


def _startups(steps: dict, kind: str = "linear") -> float:
    total = 0.0
    for s in without_realization(steps).values():
        v = s.solution.value(f"v_startup_{kind}") if f"v_startup_{kind}" in s.solution._vars else None
        if v is not None:
            total += float(utf8(v).filter(pl.col("p") == "liquefier_A")["value"].sum())
    return total


@pytest.mark.parametrize("case", ["single", "rolling", "rp", "integer", "section"])
def test_coarse_online_equals_tied_fine_model(tmp_path_factory, monkeypatch, case) -> None:
    """liquefier_A with linear (or integer) online variables, min_load 0.3, a start-up cost and 30-h minimum up /
    down times, every parameter constant within a day: the coarse model has the objective of the hourly model whose
    flows and units online are tied over each day and whose start-ups / shut-downs are 0 after a day's first hour —
    in a single solve, a rolling solve (72 h / 48 h: the units online and the 30-h minimum times carried across the
    roll), a representative-period invest solve, with integer units online and with a min_load_efficiency section.
    The start-ups are counted once per day (the coarse model starts units; a representative day is one coarse
    step that wraps to itself, so it has none)."""
    kind = "integer" if case == "integer" else "linear"
    coarse = _uc_case(tmp_path_factory, case, f"co_tie_{case}")
    assert all("liquefier_A" in _coarse_procs(s.flex_data) for s in without_realization(coarse).values())
    _tied_fine(monkeypatch)
    tied = _uc_case(tmp_path_factory, case, f"co_tie_{case}_fine")
    assert len(coarse) == len(tied)
    assert _total_objective(coarse) == pytest.approx(_total_objective(tied), rel=1e-9)
    assert (_startups(coarse, kind) > 1e-6) == (case != "rp")
    assert _startups(coarse, kind) == pytest.approx(_startups(tied, kind), rel=1e-6, abs=1e-9)
    if case == "integer":
        on = utf8(last_stochastic(coarse).solution.value("v_online_integer")).filter(pl.col("p") == "liquefier_A")
        assert on.height == 7
        assert all(abs(v - round(v)) < 1e-6 for v in on["value"].to_list())


# ---------------------------------------------------------------------------
# Roll carry: units online and the minimum up / down time history across a roll


def _value_at(step, var: str, t: str) -> float:
    v = utf8(step.solution.value(var)).filter((pl.col("p") == "liquefier_A") & (pl.col("t") == t))
    assert v.height == 1, (var, t, v)
    return float(v["value"][0])


def test_roll_carry_of_coarse_units_online(tmp_path_factory) -> None:
    """Rolling 72 h / 48 h: roll 0 realizes t0001–t0048, whose last step t0048 lies inside liquefier_A's day-2
    coarse step (first hour t0025).  Roll 1 continues from it: units online carried = roll 0's day-2 value; the
    on-line history has every realized hour that ends less than 30 h before the end, ends at t0048 (its reference row:
    hours_to_end = its own 1-h duration) and holds the day-2 start-up at t0025 (hours_to_end 24) with 0 at the
    other hours of the day; the 30-h minimum up time carries that start-up to roll 1's first hour t0049
    (the hours 24 + 0 … 24 + 5 < 30 of the carry), whose row is day 3's minimum-uptime row; the rows of days 4
    and 5 look back one day each (24 < 30)."""
    from flextool.engine.preprocess.params._mintime import history_reference_step

    rolls = list(without_realization(_uc_case(tmp_path_factory, "rolling", "co_roll_carry")).values())
    r0, r1 = rolls[0], rolls[1]
    fd = r1.flex_data
    on = utf8(fd.p_roll_continue_online.frame).filter(pl.col("p") == "liquefier_A")["value"].to_list()
    assert on == pytest.approx([_value_at(r0, "v_online_linear", "t0025")])
    hist = (fd.online_startup_history.with_columns(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t")))
            .filter(pl.col("p") == "liquefier_A").sort("hours_to_end"))
    assert history_reference_step(hist) == ("y2030", "t0048")
    assert hist["t"].to_list() == [f"t{h:04d}" for h in range(48, 19, -1)]
    assert hist["hours_to_end"].to_list() == pytest.approx([float(k) for k in range(1, 30)])
    su = _value_at(r0, "v_startup_linear", "t0025")
    assert su > 1e-6
    by_t = dict(zip(hist["t"].to_list(), hist["startup"].to_list()))
    assert by_t.pop("t0025") == pytest.approx(su)
    assert all(v == 0.0 for t, v in by_t.items() if t >= "t0025")
    carry = utf8(fd.uptime_carry).filter(pl.col("p") == "liquefier_A").sort("t")
    assert carry["t"].to_list() == [f"t{h:04d}" for h in range(49, 55)]
    assert carry["value"].to_list() == pytest.approx([su] * 6)
    # The coarse model reads it once, on day 3's row: units online − its own start-up ≥ the carried start-up.
    pb = build(fd)
    rows = row_terms(pb, "minimum_uptime_linear", "liquefier_A,")
    assert sorted(_keys(r)[-1] for r in rows) == ["t0049", "t0073", "t0097"]
    lo = row_bounds(pb, "minimum_uptime_linear", "liquefier_A,y2030,t0049")[
        "minimum_uptime_linear[liquefier_A,y2030,t0049]"][0]
    assert lo == pytest.approx(su)


# ---------------------------------------------------------------------------
# Hand-built model (tests.json, 2 days of 1-h steps, period p2020): coal_market -> gen_d -> hd
#
# hd: a balance node in a 24-h group, demand 25 MW for the 24 hours of day 1 (600 MWh), 0 on day 2; penalty_up
# 10000 and penalty_down 1000 €/MWh (P = 1000: a surplus costs P).  gen_d: coal_market (commodity, no balance: neutral;
# coal price 20 €/MWh) -> hd, efficiency 1, existing 100 MW = one unit of 100 MW (virtual_unitsize 100), linear
# online, min_load 0.5, startup_cost 1000 €/MW (1e5 € per unit started).  gen_d is coarse: on_k, su_k, sd_k once
# per day k (k = 1, 2), flow F_k.
# Rows (EQ. headers of online_change, maxFlow_online_coarse, minFlow_minload_coarse, minimum_uptime):
#   on_1 − on_2 = su_1 − sd_1 (day 1 is the stretch start: its predecessor wraps to the stretch's last step, day 2),
#   on_2 − on_1 = su_2 − sd_2;  50·on_k ≤ F_k ≤ 100·on_k (MW);  day 1 balance F_1 = 25 (a shortfall costs 10000);
#   day 2 surplus 24·F_2 MWh at P + 20 €/MWh (penalty and fuel).
# (a) min_uptime 24 h: the day-2 row looks back to day 1 only if elapsed(day 1 -> day 2) = 24 < 24: no; the day-1 row
#     has no ancestor (no wrap in the look-back).  Both hold only their own step: no minimum_uptime row.
#     Cost 1e5·(su_1 + su_2) + 1200·(P + 20)·on_2 (F_2 = 50·on_2 at least) + fuel of day 1.  0.25 ≤ on_1 ≤ 0.5;
#     su_1 ≥ on_1 − on_2.  on_2 = x > 0 saves at most 1e5·x of start-up and costs 1200·1020·x: on_2 = 0, on_1 = 0.25,
#     su_1 = 0.25, sd_1 = 0, su_2 = 0, sd_2 = 0.25; F_1 = 25 MW, F_2 = 0.
# (b) min_uptime 30 h: the day-2 row looks back to day 1 (24 < 30): on_2 ≥ su_1 + su_2.  With on_2 = y ≤ 0.25:
#     su_1 ≥ 0.25 − y and y ≥ su_1 ⇒ y ≥ 0.125; cost 1e5·(0.25 − y) + 1200·1020·y rises with y ⇒ y = 0.125 (for any
#     P > 1e5/1200 − 20 = 63.3).  on_1 = 0.25 (a higher on_1 needs a larger su_1 and y), su_1 = 0.125, sd_1 = 0,
#     on_2 = 0.125, su_2 = 0, sd_2 = 0.125; F_1 = 25 MW, F_2 = 50·0.125 = 6.25 MW (150 MWh of surplus on day 2).

_HAND_CHAIN = ["init", "west", "coal"]
_HAND_ENTITIES = [["node", "hd", None], ["unit", "gen_d", None], ["group", "daily", None],
                  ["unit__inputNode", ["gen_d", "coal_market"], None], ["unit__outputNode", ["gen_d", "hd"], None],
                  ["group__node", ["daily", "hd"], None]]
_HOURS = [f"t{h:04d}" for h in range(1, 49)]


def _hand_uc_values(min_uptime: float) -> list:
    demand = b64({"index_type": "str", "index_name": "time",
                  "data": [[t, -25.0 if h < 24 else 0.0] for h, t in enumerate(_HOURS)]}, "map")
    vals = [["group", "daily", "new_stepduration", _f(24.0)],
            ["node", "hd", "node_type", _s("balance")], ["node", "hd", "inflow", demand],
            ["node", "hd", "penalty_up", _f(10000.0)], ["node", "hd", "penalty_down", _f(1000.0)],
            ["unit", "gen_d", "conversion_method", _s("constant_efficiency")], ["unit", "gen_d", "efficiency", _f(1.0)],
            ["unit", "gen_d", "existing", _f(100.0)], ["unit", "gen_d", "virtual_unitsize", _f(100.0)],
            ["unit", "gen_d", "startup_method", _s("linear")], ["unit", "gen_d", "min_load", _f(0.5)],
            ["unit", "gen_d", "startup_cost", _f(1000.0)],
            ["unit", "gen_d", "minimum_time_method", _s("min_uptime")],
            ["unit", "gen_d", "min_uptime", _f(min_uptime)]]
    return vals


def _hand_uc(tmp_path_factory, alt: str, min_uptime: float):
    url, scen = make_db(tmp_path_factory, alt=alt, values=_hand_uc_values(min_uptime), base_chain=_HAND_CHAIN,
                        entities=_HAND_ENTITIES)
    return last_stochastic(run(tmp_path_factory, url, scen))


def _hourly(step, var: str) -> list[float]:
    from flextool.engine.output._coarse_flows import var_values

    v = var_values(step.solution, step.flex_data, var)
    rows = utf8(v.drop("value")).with_columns(value=v["value"]).filter(pl.col("p") == "gen_d").sort("t")
    assert rows["t"].to_list() == _HOURS
    return rows["value"].to_list()


@pytest.mark.parametrize("min_uptime, on, su, sd", [
    (24.0, (0.25, 0.0), (0.25, 0.0), (0.0, 0.25)),
    (30.0, (0.25, 0.125), (0.125, 0.0), (0.0, 0.125)),
])
def test_hand_built_coarse_unit_commitment(tmp_path_factory, min_uptime, on, su, sd) -> None:
    """The hand-built model (derivation above): units online, start-ups and shut-downs once per day, on every hour of
    the day for units online and at the day's first hour for start-ups / shut-downs (0 at its other hours); the
    flow 25 MW on day 1 and 50 · on_2 MW on day 2; minimum_uptime rows only where the look-back reaches the previous
    day (min_uptime 30 h: the day-2 row; 24 h: none)."""
    step = _hand_uc(tmp_path_factory, f"co_hand_{int(min_uptime)}", min_uptime)
    assert "gen_d" in _coarse_procs(step.flex_data)
    assert _hourly(step, "v_online_linear") == pytest.approx([on[0]] * 24 + [on[1]] * 24, abs=1e-9)
    for var, (first, second) in (("v_startup_linear", su), ("v_shutdown_linear", sd)):
        assert _hourly(step, var) == pytest.approx([first] + [0.0] * 23 + [second] + [0.0] * 23, abs=1e-9), var
    assert [f * 100.0 for f in _hourly(step, "v_flow")] == pytest.approx([25.0] * 24 + [50.0 * on[1]] * 24, abs=1e-7)
    rows = row_terms(build(step.flex_data), "minimum_uptime_linear", "gen_d,")
    if min_uptime == 24.0:
        assert not rows
    else:
        assert rows == {"minimum_uptime_linear[gen_d,p2020,t0025]": pytest.approx(
            {"v_online_linear[gen_d,p2020,t0025]": 1.0, "v_startup_linear[gen_d,p2020,t0001]": -1.0,
             "v_startup_linear[gen_d,p2020,t0025]": -1.0})}


# ---------------------------------------------------------------------------
# Stochastic: units online tied once per wholly tied coarse step; a partly tied step keeps the process fine

# stochastics.json with the coarse 'upper' reservoir of test_coarse_flows and a second daily reservoir 'top'
# feeding 'upper' through 'top_turbine' (linear online).  Neither node is in the stochastic group, so only the
# units-online tie (non_anticipativity_online_linear, every online process) covers top_turbine.
_TOP_ENTITIES = _STOCH_ENTITIES + [["node", "top", None], ["group__node", ["daily", "top"], None],
                                   ["unit", "top_turbine", None], ["unit__inputNode", ["top_turbine", "top"], None],
                                   ["unit__outputNode", ["top_turbine", "upper"], None]]


def _top_values(step: float) -> list:
    return _stoch_values(step) + [
        ["node", "top", "node_type", _s("storage")], ["node", "top", "existing", _f(3000.0)],
        ["node", "top", "inflow", _f(100.0)], ["node", "top", "storage_start_end_method", _s("fix_start")],
        ["node", "top", "storage_state_start", _f(0.5)],
        ["node", "top", "penalty_up", _f(5000.0)], ["node", "top", "penalty_down", _f(5000.0)],
        ["unit", "top_turbine", "efficiency", _f(1.0)], ["unit", "top_turbine", "existing", _f(300.0)],
        ["unit", "top_turbine", "virtual_unitsize", _f(100.0)], ["unit", "top_turbine", "startup_method", _s("linear")],
        ["unit", "top_turbine", "min_load", _f(0.2)], ["unit", "top_turbine", "startup_cost", _f(10.0)]]


def _top(tmp_path_factory, alt: str, scenario: str, step: float) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_top_values(step), base_chain=_chain(STOCH, scenario),
                        entities=_TOP_ENTITIES, fixture_json=STOCH)
    return run(tmp_path_factory, url, scen)


def _online_tie_reach(fd) -> list[tuple[str, str, str]]:
    """``(column, d, t)``: each units-online column of a non_anticipativity_online_linear row with a fine step it
    holds that is not tied (a coarse column holds every fine step of its coarse step)."""
    tie = utf8(fd.dt_non_anticipativity)
    tied = {(f, t) for a, fs in _forecasts(fd).items() for f in fs
            for t in tie.filter(pl.col("d") == a)["t"].to_list()}
    held: dict[tuple[str, str, str], list[str]] = {}
    if fd.process_coarse_step is not None:
        for p, d, b, t in utf8(fd.process_coarse_step).select("p", "d", "b_first", "t").iter_rows():
            held.setdefault((p, d, b), []).append(t)
    out = []
    for terms in row_terms(build(fd), "non_anticipativity_online_linear").values():
        for col in terms:
            p, d, t = _keys(col)
            out += [(col, d, s) for s in held.get((p, d, t), [t]) if (d, s) not in tied]
    return out


@pytest.mark.parametrize("step", [24.0, 48.0])
def test_stochastic_online_tie_stays_within_the_committed_steps(tmp_path_factory, step) -> None:
    """Rolling stochastic dispatch (96-h horizon, 24-h jump): each roll ties the units online of every online
    process between the forecasts over its 24 committed hours.  With 24-h coarse steps a committed day is one
    coarse step: top_turbine is coarse in every roll, tied by one row per forecast pair at the day's first hour,
    and every forecast cuts the tied hours into the same coarse steps as its reference forecast.  With 48-h steps a
    roll's first coarse step holds 24 committed and 24 uncommitted hours: top_turbine keeps the solve's step there
    (no tie row holds an uncommitted hour)."""
    steps = without_realization(_top(tmp_path_factory, f"co_sto_roll{int(step)}", "1_week_rolling_wind", step))
    rolls = [s.flex_data for s in steps.values() if _forecasts(s.flex_data)]
    assert len(rolls) >= 5
    coarse_in = []
    for fd in rolls:
        assert _online_tie_reach(fd) == []
        coarse_in.append("top_turbine" in _coarse_procs(fd))
    if step == 24.0:
        assert all(coarse_in)
        for fd in rolls:
            rows = [r for r in row_terms(build(fd), "non_anticipativity_online_linear") if "top_turbine" in r]
            pairs = sum(len(fs) - 1 for fs in _forecasts(fd).values())
            assert len(rows) == pairs, rows
            st = utf8(fd.process_coarse_step).filter(pl.col("p") == "top_turbine")
            tie_t = set(utf8(fd.dt_non_anticipativity)["t"].to_list())
            for fs in _forecasts(fd).values():
                cut = [st.filter((pl.col("d") == f) & pl.col("t").is_in(sorted(tie_t))).select("t", "b_first")
                       .sort("t").rows() for f in fs]
                assert all(c == cut[0] for c in cut)
    else:
        assert not coarse_in[0] and not all(coarse_in)


# ---------------------------------------------------------------------------
# Realization solve: units online pinned to the committed values


def test_pin_rule_keeps_a_partly_or_varying_pinned_process_fine() -> None:
    """flex_data.coarse_pin_varies: a coarse process whose units-online pin covers only part of one of its coarse
    steps, or changes within one, keeps the solve's step; one pinned to the same value on every hour of each coarse
    step (or not pinned) does not."""
    from types import SimpleNamespace

    from polar_high import Param

    from flextool.engine.preprocess.flex_data import coarse_pin_varies

    hours = [f"t{h}" for h in range(1, 7)]
    steps = pl.DataFrame({"p": [p for p in ("const", "part", "vary", "free") for _ in hours],
                          "d": ["d1"] * 24, "b_first": (["t1"] * 3 + ["t4"] * 3) * 4, "t": hours * 4})
    pins = pl.DataFrame({"p": ["const"] * 6 + ["part"] * 4 + ["vary"] * 6, "d": ["d1"] * 16,
                         "t": hours + hours[:4] + hours,
                         "value": [1.0] * 3 + [2.0] * 3 + [1.0] * 4 + [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]})
    fd = SimpleNamespace(p_fix_online=Param(("p", "d", "t"), pins))
    assert coarse_pin_varies(fd, steps) == {"part", "vary"}
    assert coarse_pin_varies(SimpleNamespace(p_fix_online=None), steps) == set()


def _realized_branches() -> list:
    """2day_dispatch's ``stochastic_branches`` with branch ``realized`` flagged ``realized: yes``: the engine adds a
    realization solve that dispatches ``period1`` with the committed (tied) units online pinned."""
    branches = ("realized", "upper", "lower", "mid")
    return b64({"index_type": "str", "data": [
        ["period1", {"type": "map", "index_type": "str", "data": [
            [b, {"type": "map", "index_type": "str", "data": [
                ["t0001", {"type": "map", "index_type": "str",
                           "data": [["yes" if b == "realized" else "no", 1.0]]}]]}]
            for b in branches]}]]}, "map")


def test_realization_solve_pins_coarse_units_online_once_per_day(tmp_path_factory) -> None:
    """2-day stochastic dispatch with a realized designation (every hour tied): top_turbine is coarse in the
    stochastic solve, its committed units online are handed to the realization solve on every hour (the value of
    its day), so the pin is the same on every hour of each day and top_turbine is coarse there too, pinned by one
    online_fix_eq row per day; its realized units online equal the committed ones."""
    url, scen = make_db(tmp_path_factory, alt="co_pin",
                        values=_top_values(24.0) + [["solve", "2day_dispatch", "stochastic_branches",
                                                     _realized_branches()]],
                        base_chain=_chain(STOCH, "2_day_stochastic_dispatch"), entities=_TOP_ENTITIES,
                        fixture_json=STOCH)
    steps = run(tmp_path_factory, url, scen)
    real = [s for k, s in steps.items() if "__realized_dispatch" in k]
    assert len(real) == 1
    fd = real[0].flex_data
    assert "top_turbine" in _coarse_procs(fd)
    pin = utf8(fd.p_fix_online.frame).filter(pl.col("p") == "top_turbine").sort("t")
    assert pin["t"].to_list() == [f"t{h:04d}" for h in range(1, 49)]
    days = [pin["value"][:24].unique().to_list(), pin["value"][24:].unique().to_list()]
    assert all(len(v) == 1 for v in days)
    pb = build(fd)
    rows = row_terms(pb, "online_fix_eq_linear", "top_turbine,")
    assert sorted(_keys(r)[-1] for r in rows) == ["t0001", "t0025"]
    on = utf8(real[0].solution.value("v_online_linear")).filter(pl.col("p") == "top_turbine").sort("t")
    assert on["value"].to_list() == pytest.approx([days[0][0], days[1][0]], abs=1e-9)


# ---------------------------------------------------------------------------
# Outputs


def test_outputs_equal_tied_fine_model(tmp_path_factory, monkeypatch) -> None:
    """The hand-built model with min_uptime 30 h (unique optimum): every written primal output (output_raw v_* /
    vq_*: flows, units online, start-ups, shut-downs, slacks) of the coarse model equals that of the hourly model
    tied over each day — units online on every hour of the day, each start-up / shut-down at the day's first hour
    and 0 at its other hours (a copied start-up would count it 24 times in the start-up totals and costs)."""
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import assert_outputs_equal, output_frames

    folders = {}
    for kind in ("coarse", "tied"):
        if kind == "tied":
            _tied_fine(monkeypatch)
        alt = f"co_out_{kind}"
        url, scen = make_db(tmp_path_factory, alt=alt, values=_hand_uc_values(30.0), base_chain=_HAND_CHAIN,
                            entities=_HAND_ENTITIES)
        folders[kind] = tmp_path_factory.mktemp(f"{alt}_w")
        steps = run_chain_from_db(url, scen, work_folder=folders[kind], keep_solutions=True)
        assert "gen_d" in _coarse_procs(last_stochastic(steps).flex_data)
    assert_outputs_equal(folders["coarse"], folders["tied"], "y2020_2day_dispatch")
    outs = output_frames(folders["coarse"], "y2020_2day_dispatch")
    assert {"v_online_linear", "v_startup_linear", "v_shutdown_linear"} <= set(outs)


def test_registry_has_explicit_online_coarse_families() -> None:
    """Every online <family>{sfx}_coarse resolves to its own registry entry with its fine family's type."""
    from flextool.engine.autoscale._layer2_types import CONSTRAINT_FAMILIES, lookup_cstr

    for fam in ("maxFlow_online", "maxFlow_back_online", "maxOutputSum_online", "minFlow_minload",
                "minFlow_output_floor"):
        for kind in ("linear", "integer"):
            name = f"{fam}_{kind}_coarse"
            assert lookup_cstr(name) is CONSTRAINT_FAMILIES[name]
            assert CONSTRAINT_FAMILIES[name] == lookup_cstr(f"{fam}_{kind}")


def test_notice_reasons_of_a_coarse_online_unit(tmp_path_factory) -> None:
    """The behaviour notice names a coarse online unit with the reasons its results can differ from the solve's
    step: units online, start-ups and shut-downs once per coarse step, and (with a minimum up / down time) the
    minimum times counted in whole coarse steps; a min load that varies within a day is named too.  Units without
    online variables keep their reasons (none here)."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    fd = last_stochastic(_lh2(tmp_path_factory, "co_reasons", [v for v in _UC_FULL if v[2] != "min_load"] + [
        ["unit", "liquefier_A", "min_load", _tmap_lh2(lambda h: 0.2 + 0.2 * (h % 2))]])).flex_data
    why = coarse_process_reasons(fd, fd.process_coarse_step)
    assert why["liquefier_A"] == ["its units online, start-ups and shut-downs are decided once per coarse step",
                                  "its minimum up / down time counts whole coarse steps",
                                  "its minimum load varies within a coarse step"]
    assert "pipe_AB" not in why
