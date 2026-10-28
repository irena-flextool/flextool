"""Startup / shutdown capacity reduction of units with online variables and ramp limits (#49).

A unit starting in step t reaches at most ``SU = min(1, min_load + ½ · ramp_speed_up · 60 · Δt)`` of its capacity on
average over the step (synchronised at min load, then ramping; owner question Q-c14a-A), a unit shutting down after
step t at most ``SD`` (down speed) in t.  The online capacity rows lose ``(availability − SU) · v_startup[t]`` and
``(availability − SD) · v_shutdown[t⁺]`` as a pair of rows (``maxFlow_online_*`` + ``maxFlow_online_shutdown_*``;
``maxInputFlow_online_*`` + ``…_shutdown_*`` on an indirect unit's input arc), so capacity online for one step gets
the smaller limit.  Only where the arc has a ramp limit (``ramp_method`` set).

Fixtures: ``tests.json`` (coal_plant 500 MW at west, coal_chp indirect coal_market → west + heat) plus override
alternatives, built from JSON.  Coefficients are computed in the tests from the inputs.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_terms, step_of
from tests.engine._unit_db import CHP, COAL, PLANT, b64, build, make_db, run, unitsize, utf8

pytestmark = pytest.mark.solver

ARC = f"{PLANT},coal_market,west,"
TOL = 1e-6
ML, SPEED = 0.3, 0.005          # min_load; ramp 0.005/min = 0.3 of capacity per 1-h step
SU = min(1.0, ML + 0.5 * SPEED * 60.0 * 1.0)


def _ramp(speed: float = SPEED, cls: str = "unit__outputNode", rel=None, method: str | None = "ramp_limit") -> list:
    rel = rel or [PLANT, "west"]
    out = [[cls, rel, f"ramp_speed_{d}", b64(speed, "float")] for d in ("up", "down")]
    if method is not None:
        out.append([cls, rel, "ramp_method", b64(method, "str")])
    return out


def _uc(unit: str = PLANT, ml: float = ML, method: str = "linear") -> list:
    return [["unit", unit, "startup_method", b64(method, "str")], ["unit", unit, "min_load", b64(ml, "float")]]


def _tmap(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, v] for t, v in values.items()]}, "map")


def _demand(mw: dict[int, float], default: float = 0.0) -> list:
    return [["node", "west", "inflow", _tmap({f"t{i:04d}": -mw.get(i, default) for i in range(1, 49)})]]


def _solve(tmp_path_factory, alt, values, chain=("init", "west", "coal"), entities=None, fixture_json=None):
    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=list(chain), entities=entities, **kw)
    return run(tmp_path_factory, url, scen)


def _last(steps):
    return list(steps.values())[-1]


def _vals(step, var: str, unit: str, extra=None) -> dict[tuple[str, str], float]:
    v = utf8(step.solution.value(var)).filter(pl.col("p") == unit)
    if extra:
        for k, x in extra.items():
            v = v.filter(pl.col(k) == x)
    return {(d, t): x for d, t, x in v.select("d", "t", "value").iter_rows()}


@pytest.fixture(scope="module")
def start_case(tmp_path_factory):
    """No demand in t0001–t0010, 400 MW afterwards: coal_plant must start."""
    return _last(_solve(tmp_path_factory, "su_start", _uc() + _ramp()
                        + _demand({i: 0.0 for i in range(1, 11)}, 400.0)))


def test_startup_step_limited_to_su(start_case) -> None:
    """maxFlow_online_linear has −(1 − SU) on v_startup at every step, and the solution keeps the started units
    at SU at most: flow ≤ online − (1 − SU)·startup.  Before: no term (a started unit could run at full output)."""
    step = start_case
    rows = row_terms(build(step.flex_data), "maxFlow_online_linear", ARC)
    assert rows
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_startup_linear[{PLANT},{d},{t}]") == pytest.approx(1.0 - SU), row
    f = _vals(step, "v_flow", PLANT, {"sink": "west"})
    on = _vals(step, "v_online_linear", PLANT)
    su = _vals(step, "v_startup_linear", PLANT)
    assert max(su.values()) > 0.1
    for k in f:
        assert f[k] <= on[k] - (1.0 - SU) * su[k] + TOL, k


def test_startup_step_integer(tmp_path_factory) -> None:
    """The same with binary online variables: the _integer rows carry the reduction."""
    step = _last(_solve(tmp_path_factory, "su_int", _uc(method="binary") + _ramp()
                        + _demand({i: 0.0 for i in range(1, 11)}, 400.0)))
    rows = row_terms(build(step.flex_data), "maxFlow_online_integer", ARC)
    assert rows
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_startup_integer[{PLANT},{d},{t}]") == pytest.approx(1.0 - SU), row
    assert row_terms(build(step.flex_data), "maxFlow_online_shutdown_integer", ARC)


def test_shutdown_step_limited_to_sd(start_case) -> None:
    """maxFlow_online_shutdown_linear: −(1 − SD) on v_shutdown of the next step (same period), none at the last
    step; flow ≤ online − (1 − SD)·shutdown[t⁺] in the solution."""
    step = start_case
    rows = row_terms(build(step.flex_data), "maxFlow_online_shutdown_linear", ARC)
    assert len(rows) == step.flex_data.dt.height - 1
    for row, terms in rows.items():
        d, t = step_of(row)
        nxt = f"t{int(t[1:]) + 1:04d}"
        assert coef(terms, f"v_shutdown_linear[{PLANT},{d},{nxt}]") == pytest.approx(1.0 - SU), row
        assert not [c for c in terms if c.startswith("v_startup")]   # SU == SD: no residual startup term
    f = _vals(step, "v_flow", PLANT, {"sink": "west"})
    on = _vals(step, "v_online_linear", PLANT)
    sd = _vals(step, "v_shutdown_linear", PLANT)
    for (d, t), x in f.items():
        nxt = (d, f"t{int(t[1:]) + 1:04d}")
        if nxt in sd:
            assert x <= on[(d, t)] - (1.0 - SU) * sd[nxt] + TOL


def test_one_step_run_feasible(tmp_path_factory) -> None:
    """Demand only in t0010 (120 MW = 0.24 units): units started at t0010 and stopped at t0011 run at
    min_load ≤ flow / started ≤ SU, so the demand is met without slack.  GMPL's single row with both
    reductions would allow only (1 − 2·(1 − SU)) < min_load of the started capacity."""
    step = _last(_solve(tmp_path_factory, "su_one", _uc() + _ramp() + _demand({10: 120.0}, 0.0)
                        + [["node", "west", "penalty_down", b64(5000.0, "float")]]))
    up = utf8(step.solution.value("vq_state_up")).filter(pl.col("n") == "west")["value"]
    assert float(up.abs().max()) < 1e-6
    f = _vals(step, "v_flow", PLANT, {"sink": "west"})
    on = _vals(step, "v_online_linear", PLANT)
    assert f[("p2020", "t0010")] == pytest.approx(0.24, abs=1e-6)
    assert ML * on[("p2020", "t0010")] - TOL <= f[("p2020", "t0010")] <= SU * on[("p2020", "t0010")] + TOL


def test_availability_caps_reduction(tmp_path_factory) -> None:
    """availability 0.4 < SU: no startup term (started units are limited by availability anyway); availability
    0.9: coefficient 0.9 − SU (GMPL's 1 − SU would limit started units to availability − (1 − SU))."""
    for avail, alt in ((0.4, "su_av4"), (0.9, "su_av9")):
        step = _last(_solve(tmp_path_factory, alt, _uc() + _ramp() + [["unit", PLANT, "availability",
                                                                         b64(avail, "float")]]))
        rows = row_terms(build(step.flex_data), "maxFlow_online_linear", ARC)
        for row, terms in rows.items():
            hit = [c for c in terms if c.startswith("v_startup_linear")]
            if avail - SU <= 0:
                assert not hit, row
            else:
                assert coef(terms, "v_startup_linear") == pytest.approx(avail - SU), row


def test_no_reduction_without_ramp_limit(tmp_path_factory) -> None:
    """ramp_speed authored without ramp_method: no ramp row and no reduction (how_to: "if the unit also has ramp
    limits")."""
    step = _last(_solve(tmp_path_factory, "su_noramp", _uc() + _ramp(method=None)))
    pb = build(step.flex_data)
    for terms in row_terms(pb, "maxFlow_online_linear", ARC).values():
        assert not [c for c in terms if c.startswith("v_startup")]
    assert not row_terms(pb, "maxFlow_online_shutdown_linear")


def test_input_side_reduction_lets_a_unit_start_at_min_load(tmp_path_factory) -> None:
    """coal_chp (indirect, min_load_efficiency: efficiency 0.9 at full, 0.6 at min load 0.3) with a ramp on its
    fuel input: maxInputFlow_online_linear rows on the fuel arc (no reserve needed) with startup coefficient
    input_cap · (1 − SU_in), SU_in = x + ½·0.12, x = the fuel drawn at min load per unit of input limit (> min_load
    because of the min-load section).  A forced start (west 300 MW from t0011, coal_chp the only unit) is
    feasible and the unit can run at min load in its startup step."""
    vals = (_uc(CHP, 0.3) + _ramp(0.002, cls="unit__inputNode", rel=[CHP, COAL])
            + [["unit", CHP, "conversion_method", b64("min_load_efficiency", "str")],
               ["unit", CHP, "efficiency_at_min_load", b64(0.6, "float")],
               ["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
               ["node", "heat", "inflow", b64(0.0, "float")]]
            + _demand({i: 0.0 for i in range(1, 11)}, 300.0))
    step = _last(_solve(tmp_path_factory, "su_chp", vals, chain=("init", "west", "coal_chp", "heat")))
    fd = step.flex_data
    slope = float(utf8(fd.p_slope.frame).filter(pl.col("p") == CHP)["value"][0])
    section = float(utf8(fd.p_section.frame).filter(pl.col("p") == CHP)["value"][0])
    assert section > 0
    x = (slope * 0.3 + section) / (slope * 1.0 + section)
    su_in = min(1.0, x + 0.5 * 0.002 * 60.0)
    cap = float(utf8(fd.p_indirect_input_cap.frame).filter((pl.col("p") == CHP) & (pl.col("source") == COAL))
                ["value"][0])
    rows = row_terms(build(fd), "maxInputFlow_online_linear", f"{CHP},{COAL},{CHP},")
    assert rows
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_startup_linear[{CHP},{d},{t}]") == pytest.approx(cap * (1.0 - su_in)), row
    assert row_terms(build(fd), "maxInputFlow_online_shutdown_linear", f"{CHP},{COAL},{CHP},")
    up = utf8(step.solution.value("vq_state_up")).filter(pl.col("n") == "west")["value"]
    assert float(up.abs().max()) < 1e-6
    assert max(_vals(step, "v_startup_linear", CHP).values()) > 0.1
    # min load in a startup step: the fuel for 0.3 units of output fits under the reduced input cap
    assert x * cap < su_in * cap + 1e-12 and x > 0.3


def test_reduction_not_across_representative_blocks(tmp_path_factory) -> None:
    """With representative blocks t0001–24 and t0025–48 (adjacent on the timeline) the step before a block start
    has no successor: no shutdown row at t0024 (or t0048)."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 0.5], ["t0025", 0.5]]}]]}
    pm = {"index_type": "str", "index_name": "time", "data": [["t0001", 24.0], ["t0025", 24.0]]}
    step = _last(_solve(tmp_path_factory, "su_rp", _uc() + _ramp() + [
        ["timeset", "rp2", "timeline", b64("y2020", "str")],
        ["timeset", "rp2", "timeset_duration", b64(pm, "map")],
        ["timeset", "rp2", "representative_period_weights", b64(rpw, "map")],
        ["solve", "y2020_2day_dispatch", "period_timeset",
         b64({"index_type": "str", "index_name": "period", "data": [["p2020", "rp2"]]}, "map")]],
        entities=[["timeset", "rp2", None]]))
    have = {step_of(r) for r in row_terms(build(step.flex_data), "maxFlow_online_shutdown_linear", ARC)}
    assert ("p2020", "t0024") not in have and ("p2020", "t0048") not in have
    assert ("p2020", "t0023") in have and ("p2020", "t0025") in have


# ---------------------------------------------------------------------------
# min_load / availability shapes: constant vs equivalent maps give the same rows (CLAUDE.md shape rule)


def _shape_values(kind: str, periods: list[str], value: float, param: str) -> list:
    if kind == "constant":
        return [["unit", PLANT, param, b64(value, "float")]]
    if kind == "time":
        return [["unit", PLANT, param, _tmap({f"t{i:04d}": value for i in range(1, 49)})]]
    return [["unit", PLANT, param, b64({"index_type": "str", "index_name": "period",
                                        "data": [[p, value] for p in periods]}, "map")]]


def _online_rows(steps) -> list:
    out = []
    for s in steps.values():
        pb = build(s.flex_data)
        out.append({fam: row_terms(pb, fam) for fam in ("maxFlow_online_linear", "maxFlow_online_shutdown_linear")})
    return out


@pytest.mark.parametrize("solve_kind", ["single", "rolling", "stochastic"])
def test_min_load_and_availability_shapes_same_lp(tmp_path_factory, solve_kind) -> None:
    from tests.engine._unit_db import FLEXTOOL_ROOT

    if solve_kind == "stochastic":
        fixture = FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json"
        chain, periods, node = ("base", "init", "system", "2day"), ["period1"], "demand_node"
        extra: list = []
    else:
        fixture, chain, periods, node = None, ("init", "west", "coal"), ["p2020"], "west"
        extra = ([["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
                  ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
                  ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(24.0, "float")]]
                 if solve_kind == "rolling" else [])
    base = _uc(ml=0.0)[:1] + _ramp(rel=[PLANT, node]) + extra
    results = {}
    for ml_kind in ("constant", "time", "period"):
        for av_kind in ("constant", "time"):
            if (ml_kind, av_kind) not in (("constant", "constant"), ("time", "constant"), ("period", "constant"),
                                          ("constant", "time")):
                continue
            vals = (base + _shape_values(ml_kind, periods, ML, "min_load")
                    + _shape_values(av_kind, periods, 0.9, "availability"))
            steps = _solve(tmp_path_factory, f"shp_{solve_kind}_{ml_kind}_{av_kind}", vals, chain=chain,
                           fixture_json=fixture)
            results[(ml_kind, av_kind)] = _online_rows(steps)
    ref = results[("constant", "constant")]
    assert any(any("v_startup_linear" in c for terms in rows["maxFlow_online_linear"].values() for c in terms)
               for rows in ref)
    for key, got in results.items():
        assert got == ref, key


# ---------------------------------------------------------------------------
# input-arc online limit at derated full output


def _chp_mle(avail: float, *, ramp: bool, extra=None) -> list:
    vals = (_uc(CHP, 0.3)
            + [["unit", CHP, "conversion_method", b64("min_load_efficiency", "str")],
               ["unit", CHP, "efficiency_at_min_load", b64(0.6, "float")],
               ["unit", CHP, "availability", b64(avail, "float")],
               ["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
               ["node", "heat", "inflow", b64(0.0, "float")],
               ["node", "west", "inflow", _tmap({f"t{i:04d}": -3000.0 for i in range(1, 49)})]])
    if ramp:
        vals += _ramp(0.005, cls="unit__inputNode", rel=[CHP, COAL])
    return vals + (extra or [])


def _x(fd, y: float) -> float:
    """Fuel at output level y per unit of full-load fuel, x(y) = (slope·y + section)/(slope + section)."""
    slope = float(utf8(fd.p_slope.frame).filter(pl.col("p") == CHP)["value"][0])
    section = float(utf8(fd.p_section.frame).filter(pl.col("p") == CHP)["value"][0])
    return (slope * y + section) / (slope + section)


def test_input_limit_keeps_derated_full_output(tmp_path_factory) -> None:
    """coal_chp (min_load_efficiency, availability 0.8, demand far above capacity, online all the time): the
    fuel arc's online coefficient (maxInputFlow_online, ramp-limited case) and its v_flow upper bound
    (constant-capacity case, no ramp limit) are both input_cap · x(0.8) — the section is drawn per unit
    online, not per available unit — so the output reaches the true derated full output, 0.8 × capacity,
    with or without the ramp limit (#102).  Before: the v_flow bound alone used input_cap · 0.8, capping
    the no-ramp output below 0.8 × capacity even though the ramp-limited case (already fixed by the online input rows)
    did not."""
    no_ramp = _last(_solve(tmp_path_factory, "av_noramp", _chp_mle(0.8, ramp=False),
                           chain=("init", "west", "coal_chp", "heat")))
    with_ramp = _last(_solve(tmp_path_factory, "av_ramp", _chp_mle(0.8, ramp=True),
                             chain=("init", "west", "coal_chp", "heat")))
    f0 = _vals(no_ramp, "v_flow", CHP, {"sink": "west"})
    f1 = _vals(with_ramp, "v_flow", CHP, {"sink": "west"})
    cap_units = 1000.0 / unitsize(no_ramp, CHP)  # coal_chp existing 1000 MW (module docstring)
    for k in f0:
        assert f0[k] == pytest.approx(0.8 * cap_units, rel=1e-6), k
        assert f1[k] == pytest.approx(f0[k], abs=1e-6), k
    fd = with_ramp.flex_data
    cap = float(utf8(fd.p_indirect_input_cap.frame).filter((pl.col("p") == CHP) & (pl.col("source") == COAL))
                ["value"][0])
    rows = row_terms(build(fd), "maxInputFlow_online_linear", f"{CHP},{COAL},{CHP},")
    assert rows
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_online_linear[{CHP},{d},{t}]") == pytest.approx(-cap * _x(fd, 0.8)), row
        su_in = min(1.0, _x(fd, 0.3) + 0.5 * 0.005 * 60.0)
        hit = [c for c in terms if c.startswith("v_startup_linear")]
        if _x(fd, 0.8) - su_in > 0:
            assert coef(terms, "v_startup_linear") == pytest.approx(cap * (_x(fd, 0.8) - su_in)), row
        else:
            assert not hit, row


def test_consume_more_row_at_derated_full_output(tmp_path_factory) -> None:
    """c13's "consume more" row (maxInputFlow_online on a fuel arc with downward reserve at coal_market) uses the
    same limit: fuel + reserve ≤ input_cap · x(availability) · online, so holding the reserve row does not lower
    the output below the unit's output without the reserve.  Before: input_cap · availability · online
    (0.7455 units)."""
    grp = ["rf", "down", "fuel"]
    ents = [["reserve", "rf", None], ["reserve__upDown__group", grp, None],
            ["reserve__upDown__unit__node", ["rf", "down", CHP, COAL], None]]
    res = [["reserve__upDown__group", grp, "reserve_method", b64("timeseries_only", "str")],
           ["reserve__upDown__group", grp, "reservation", b64(1.0, "float")],
           ["reserve__upDown__group", grp, "penalty_reserve", b64(1.0, "float")],
           ["reserve__upDown__unit__node", ["rf", "down", CHP, COAL], "is_enabled", b64("yes", "str")],
           ["reserve__upDown__unit__node", ["rf", "down", CHP, COAL], "max_share", b64(1.0, "float")],
           ["reserve__upDown__unit__node", ["rf", "down", CHP, COAL], "reliability", b64(1.0, "float")]]
    step = _last(_solve(tmp_path_factory, "av_res", _chp_mle(0.8, ramp=False, extra=res),
                        chain=("init", "west", "coal_chp", "heat"), entities=ents))
    fd = step.flex_data
    cap = float(utf8(fd.p_indirect_input_cap.frame).filter((pl.col("p") == CHP) & (pl.col("source") == COAL))
                ["value"][0])
    rows = row_terms(build(fd), "maxInputFlow_online_linear", f"{CHP},{COAL},{CHP},")
    assert rows
    for row, terms in rows.items():
        d, t = step_of(row)
        assert coef(terms, f"v_online_linear[{CHP},{d},{t}]") == pytest.approx(-cap * _x(fd, 0.8)), row
    base = _last(_solve(tmp_path_factory, "av_nores", _chp_mle(0.8, ramp=False),
                        chain=("init", "west", "coal_chp", "heat")))
    f = _vals(step, "v_flow", CHP, {"sink": "west"})
    f0 = _vals(base, "v_flow", CHP, {"sink": "west"})
    for k in f0:
        assert f[k] == pytest.approx(f0[k], abs=1e-6), k
