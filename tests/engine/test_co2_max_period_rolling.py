"""``co2_max_period`` in rolling solves applies to each roll's own window (#54).

A roll of a rolling solve is a synthetic sub-solve: its ``p_period_share`` is the complete (parent) solve's
share, so each roll compared its window's emissions, annualised as if they covered the whole solve, with the
annual cap: the cap was loose by complete hours / window hours and did not bind.  The cap is now scaled by
``p_window_fraction[d] = (Σ_t step_duration[d,t] / 8760) / p_period_share[d]`` (the roll's window, realized
part and look-ahead, per period; exactly 1.0 outside rolls), so a roll is the single solve of its window and
the dual keeps the objective's units.

Fixtures: ``tests.json`` ``init/west/coal/co2_limit`` (coal_plant 500 MW, slope 2.5, co2_content 0.34) plus
an override alternative, ``stoch_two_period.json`` and ``stochastics.json`` (built from JSON, never a
checked-in DB).  ``y2020_2day_dispatch`` is 48 one-hour steps per period.  Magnitudes stay below the autoscale
Layer-2 trigger (warm ``p_inflow`` pushes ignore the Layer-2 row scale, #88).
"""
from __future__ import annotations

import dataclasses
import math
import sys

import polars as pl
import pytest

from tests.engine._unit_db import BASE_FIXTURE_JSON, FLEXTOOL_ROOT, RunCache, b64, build, make_db, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
CHAIN = ["init", "west", "coal", "co2_limit"]
FIX = FLEXTOOL_ROOT / "tests" / "fixtures"


def _pmap(values: dict) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _cap(values: dict[str, float]) -> list:
    return ["group", "co2_limit", "co2_max_period", _pmap(values)]


def _rolling(horizon: float, jump: float) -> list:
    return [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
            ["solve", SOLVE, "rolling_solve_horizon", b64(horizon, "float")],
            ["solve", SOLVE, "rolling_solve_jump", b64(jump, "float")]]


TWO_PERIODS = [
    ["solve", SOLVE, "period_timeset", _pmap({"p2020": "2day", "p2025": "2day"})],
    ["solve", SOLVE, "realized_periods",
     b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")],
]


# Tests that solve the same model share one run (``_run`` is never called under a monkeypatch).
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _run(tmp_path_factory, alt: str, values: list, *, warm: bool = False, base_chain=CHAIN,
         fixture_json=None) -> dict:
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=base_chain,
                                  fixture_json=fixture_json or BASE_FIXTURE_JSON, warm=warm)
    assert steps and all(st.solution is not None for st in steps.values())
    return steps


def _fraction(fd) -> dict[str, float]:
    p = fd.p_window_fraction
    assert p is not None
    assert p.dims == ("d",)
    frame = utf8(p.frame)
    assert frame["d"].is_unique().all()
    return dict(frame.select("d", "value").iter_rows())


def _hours(fd) -> dict[str, float]:
    sd = utf8(fd.p_step_duration.frame)
    return dict(sd.group_by("d").agg(pl.col("value").sum()).sort("d").iter_rows())


def _tonnes(step) -> dict[str, float]:
    """CO2 of coal_plant per period over the solve's own steps (cold solves: Solution labels are current)."""
    fd = step.flex_data
    assert utf8(fd.p_slope.frame)["value"].unique().to_list() == [2.5]
    assert utf8(fd.p_co2_content.frame)["value"].unique().to_list() == [0.34]
    size = float(utf8(fd.p_unitsize.frame).filter(pl.col("p") == "coal_plant")["value"][0])
    vf = utf8(step.solution.value("v_flow")).filter(pl.col("p") == "coal_plant")
    sd = utf8(fd.p_step_duration.frame).rename({"value": "sd"})
    rows = vf.join(sd, on=["d", "t"], how="inner")
    assert rows.height == vf.height
    out = rows.group_by("d").agg((pl.col("value") * pl.col("sd") * size * 2.5 * 0.34).sum().alias("e"))
    return dict(out.sort("d").iter_rows())


def _dual(step) -> list[float]:
    return step.solution.constraint_dual("co2_max_period")["dual"].to_list()


def _carry(fd, d: str) -> tuple[float, float, float]:
    """``(f_d_k[d], co2_cum_realized_tonnes[d], period_share[d])`` as carried into this roll's build
    (c18b, #38, Q22) -- the inputs ``_co2_period_cap_with_carry`` (``model/core.py``) turns into the
    roll's RHS: ``cap[d] = co2_max_period[d]*f_d_k[d] - co2_cum_realized_tonnes[d]/period_share[d]``, so
    in raw (non-annualised) tonnes the roll's own-window emissions must satisfy
    ``emissions[d] <= co2_max_period[d]*f_d_k[d]*period_share[d] - co2_cum_realized_tonnes[d]``.
    Defaults (1.0, 0.0) match the helper's own fallback: a period outside ``f_d_k`` is fully reached
    (pre-chain / pre-rolling behaviour); the first roll of a chain carries no realized total yet."""
    fdk_p = getattr(fd, "p_co2_f_d_k", None)
    fdk = 1.0
    if fdk_p is not None:
        row = utf8(fdk_p.frame).filter(pl.col("d") == d)
        if row.height > 0:
            fdk = float(row["value"][0])
    realized_p = getattr(fd, "p_co2_cum_realized_tonnes", None)
    realized = 0.0
    if realized_p is not None:
        row = utf8(realized_p.frame).filter(pl.col("d") == d)
        if row.height > 0:
            realized = float(row["value"][0])
    share = float(utf8(fd.p_period_share.frame).filter(pl.col("d") == d)["value"][0])
    return fdk, realized, share


def _roll(i: int) -> str:
    return f"{SOLVE}_roll_{i}"


# ---------------------------------------------------------------------------
# One period


def test_rolling_without_lookahead_equals_single_solve(tmp_path_factory) -> None:
    """Rolling 12 h / 12 h (no look-ahead): every hour is realized once, each roll is the single solve of
    its 12 h, so the realized emissions and the CO2 price equal the 48 h single solve's.  Before the fix the
    rolls did not bind (19 924 t instead of 16 438 t, dual 0)."""
    cap = 3.0e6
    single = list(_run(tmp_path_factory, "co2roll_single", [_cap({"p2020": cap})]).values())
    assert len(single) == 1
    e_single = _tonnes(single[0])["p2020"]
    assert e_single * 8760 / 48 == pytest.approx(cap, rel=1e-6)  # binding in the single solve
    (d_single,) = _dual(single[0])
    assert d_single < 0

    rolls = list(_run(tmp_path_factory, "co2roll_roll12", [_cap({"p2020": cap})] + _rolling(12.0, 12.0)).values())
    assert len(rolls) == 4
    assert sum(_tonnes(st)["p2020"] for st in rolls) == pytest.approx(e_single, rel=1e-6)
    for st in rolls:
        assert _fraction(st.flex_data) == {"p2020": 0.25}
        (d_roll,) = _dual(st)
        assert d_roll == pytest.approx(d_single, rel=1e-6), st.solve_name


def test_cumulative_budget_caps_each_rolls_window(tmp_path_factory) -> None:
    """Rolling 24 h / 12 h (window > jump, so windows overlap): the owner's carried budget (c18b, #38,
    Q22) -- allocation through this roll's horizon end minus what earlier rolls already realized -- caps
    each roll's own-window emissions, not the flat per-window rate a pre-c18b check would impose:
    ``emissions[p2020] <= co2_max_period[p2020]*f_d_k[p2020]*period_share[p2020] -
    co2_cum_realized_tonnes[p2020]`` (raw tonnes; see ``_carry``'s docstring for the derivation from
    ``_co2_period_cap_with_carry``).

    Roll 0's window [0,24) is allowed the full cap through hour 24 (f_d_k=0.5): 3.0e6*0.5*(48/8760) =
    8219.18 t, all of it unrealized slack since nothing was realized before roll 0 -- the window emits
    exactly that (binding).  Only its first 12 h get realized (jump=12), locking in 3154.88 t of that
    8219.18 t allowance; the other 5064.30 t was look-ahead that never became realized, so the realized
    carrier into roll 1 is only 3154.88 t.  Roll 1's window [12,36) is then allowed up to hour 36
    (f_d_k=0.75): 3.0e6*0.75*(48/8760) - 3154.88 = 9173.89 t -- MORE than the 8219.18 t a flat per-window
    rate would allow over its own 24 h, because roll 0's unrealized look-ahead slack was never charged
    against the realized carrier and so remains available to roll 1.  This is the intended behaviour
    (never violates the cumulative budget, which is what this test checks), not a regression of the old
    (pre-c18b) ``test_each_roll_caps_its_window``, which asserted the flat per-window rate directly and
    no longer holds."""
    cap = 3.0e6
    rolls = list(_run(tmp_path_factory, "co2roll_roll24", [_cap({"p2020": cap})] + _rolling(24.0, 12.0)).values())
    assert [_fraction(st.flex_data) for st in rolls] == [{"p2020": f} for f in (0.5, 0.5, 0.5, 0.25)]
    for st in rolls:
        p = st.flex_data.p_window_fraction
        assert p.frame.equals(p.frame.sort("d"))
        fdk, realized, share = _carry(st.flex_data, "p2020")
        lhs = _tonnes(st)["p2020"]
        rhs = cap * fdk * share - realized
        assert lhs <= rhs * (1 + 1e-9) + 1e-6, st.solve_name
        assert lhs == pytest.approx(rhs, rel=1e-6), st.solve_name


def test_output_horizon_leaves_the_realized_carrier_unchanged(tmp_path_factory) -> None:
    """``model.output_horizon`` is an output-only debug flag: with rolling 24 h / 12 h (look-ahead 12 h) and the
    binding cap above, every roll carries the same realized CO2 into the next roll and builds the same LP as
    without the flag.  Before, the flag widened the realized row set to each roll's whole window, so the carried
    total and the reached fraction ``f_d_k`` included the look-ahead hours (roll 1 carried roll 0's whole-window
    8219.18 t at f_d_k 1.0 instead of the realized 3154.88 t at 0.75)."""
    cap = 3.0e6
    values = [_cap({"p2020": cap})] + _rolling(24.0, 12.0)
    flag = [["model", "flexTool", "output_horizon", b64("yes", "str")]]
    with_flag = list(_run(tmp_path_factory, "co2roll_oh_on", values + flag).values())
    without = list(_run(tmp_path_factory, "co2roll_oh_off", values).values())
    assert len(with_flag) == len(without) == 4
    for a, b in zip(with_flag, without):
        assert a.solve_name == b.solve_name
        assert _carry(a.flex_data, "p2020") == _carry(b.flex_data, "p2020"), a.solve_name
        ma, mb = build(a.flex_data).canonicalise(), build(b.flex_data).canonicalise()
        assert list(ma.row_names) == list(mb.row_names) and list(ma.col_names) == list(mb.col_names)
        assert list(ma.val) == list(mb.val), a.solve_name


# ---------------------------------------------------------------------------
# Two periods, rolls across the boundary


CAPS = {"p2020": 1.0e6, "p2025": 0.5e6}


@pytest.fixture(scope="module")
def boundary_runs(tmp_path_factory) -> dict:
    """p2020 + p2025 (2 days each, both realized), rolling 30 h / 6 h: 16 rolls, 4-7 straddle the boundary."""
    from flextool.engine import run_chain_from_db

    values = [_cap(CAPS)] + TWO_PERIODS + _rolling(30.0, 6.0)
    url, scen = make_db(tmp_path_factory, alt="co2roll_boundary", values=values, base_chain=CHAIN)
    out = {}
    for warm in (True, False):
        wf = tmp_path_factory.mktemp(f"co2roll_boundary_w{int(warm)}")
        # The cold run also writes the CSV dump (no effect on the LP) and a snapshot (``csv_workdir``), so a
        # roll can be reloaded from disk (tests/csv_reload/test_co2_max_period_rolling.py, c24 S1-8).
        out[warm] = run_chain_from_db(url, scen, warm=warm, keep_solutions=True, work_folder=wf,
                                      csv_dump=not warm)
        if not warm:
            out["csv_workdir"] = wf
            next(reversed(list(out[warm].values()))).flex_data_provider.snapshot_processed_inputs(wf)
    assert list(out[True]) == list(out[False]) == [_roll(i) for i in range(16)]
    return out


def test_rolls_across_a_period_boundary(boundary_runs) -> None:
    hot, cold = boundary_runs[True], boundary_runs[False]
    # (a) per-period fractions of the straddling rolls
    expected = {0: {"p2020": 30 / 48}, 4: {"p2020": 24 / 48, "p2025": 6 / 48},
                5: {"p2020": 18 / 48, "p2025": 12 / 48}, 6: {"p2020": 12 / 48, "p2025": 18 / 48},
                7: {"p2020": 6 / 48, "p2025": 24 / 48}, 15: {"p2025": 6 / 48}}
    for i, want in expected.items():
        assert _fraction(cold[_roll(i)].flex_data) == pytest.approx(want, rel=1e-12), i
    # (b) each roll and period: the owner's carried cumulative budget (c18b, #38, Q22), not the old flat
    # per-window rate -- an overlapping window can legitimately absorb an earlier roll's unrealized
    # look-ahead slack (see test_cumulative_budget_caps_each_rolls_window's derivation), so this checks
    # the cumulative invariant (``_carry``) directly instead.
    for i in range(16):
        st = cold[_roll(i)]
        for d, e in _tonnes(st).items():
            fdk, realized, share = _carry(st.flex_data, d)
            rhs = CAPS[d] * fdk * share - realized
            assert e <= rhs * (1 + 1e-9) + 1e-6, (i, d)
            assert e == pytest.approx(rhs, rel=1e-6), (i, d)
    # (c) warm == cold; no roll reuses the previous roll's LP (its ``dt`` labels shift)
    for i in range(16):
        assert not hot[_roll(i)].warm_used, f"roll {i} reused an LP built for another window"
    for i in range(16):
        assert hot[_roll(i)].solution.obj == pytest.approx(cold[_roll(i)].solution.obj, rel=1e-9), i


def test_co2_cap_without_window_fraction_fails_clearly(boundary_runs) -> None:
    """A co2-capped build without the fraction, with a period missing, or with a fraction <= 0 or not finite
    raises instead of dropping or zeroing that period's cap."""
    fd = boundary_runs[False][_roll(5)].flex_data
    assert "co2_max_period" in set(build(fd).cstr_names())
    with pytest.raises(ValueError, match=r"'co2_max_period'.*p_window_fraction"):
        build(dataclasses.replace(fd, p_window_fraction=None))
    frame = fd.p_window_fraction.frame
    one = type(fd.p_window_fraction)(("d",), frame.filter(pl.col("d").cast(pl.Utf8) == "p2020"))
    with pytest.raises(ValueError, match=r"p_window_fraction lacks periods \['p2025'\]"):
        build(dataclasses.replace(fd, p_window_fraction=one))
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        p = type(fd.p_window_fraction)(("d",), frame.with_columns(
            value=pl.when(pl.col("d").cast(pl.Utf8) == "p2025").then(pl.lit(bad)).otherwise(pl.col("value"))))
        with pytest.raises(ValueError, match=r"p_window_fraction must be finite and above 0"):
            build(dataclasses.replace(fd, p_window_fraction=p))


# ---------------------------------------------------------------------------
# Outside rolls the fraction is exactly 1.0


def test_window_fraction_is_one_outside_rolls(tmp_path_factory) -> None:
    """Single solve, multi-period single solve with investments, stochastic single solve (branch periods
    included): exactly 1.0 for every dt period."""
    cases = [
        ("co2roll_one_single", [], CHAIN, None),
        ("co2roll_one_multi", [], ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "5weeks",
                               "invest_solveSequence_5weeks", "invest_allYears_5weeks", "co2_limit"], None),
        ("co2roll_one_stoch", [], ["init", "base", "stoch"], FIX / "stoch_two_period.json"),
    ]
    for alt, values, chain, fixture in cases:
        steps = _run(tmp_path_factory, alt, values, base_chain=chain, fixture_json=fixture)
        for st in steps.values():
            fd = st.flex_data
            periods = set(utf8(fd.dt.select("d"))["d"].to_list())
            frac = _fraction(fd)
            assert set(frac) == periods, (alt, st.solve_name)
            assert all(v == 1.0 for v in frac.values()), (alt, st.solve_name, frac)
        if alt == "co2roll_one_stoch":
            assert any("_" in d for d in frac), "no stochastic branch periods in the fixture"


def _solve_name_from_stack() -> str:
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_locals.get("complete_solve_name")
        if isinstance(name, str) and name:
            return name
        frame = frame.f_back
    return "?"


class _Stop(Exception):
    pass


def _record_builds(monkeypatch, n: int) -> list:
    """Record (solve name, window hours, share hours, fraction) of the first ``n`` LP builds, then stop."""
    import flextool.engine.model.core as core

    seen: list = []
    orig = core.build_flextool

    def wrapped(pb, fd, *a, **k):
        share = {d: v * 8760 for d, v in utf8(fd.p_period_share.frame).select("d", "value").iter_rows()}
        seen.append((_solve_name_from_stack(), _hours(fd), share, _fraction(fd)))
        if len(seen) >= n:
            raise _Stop()
        return orig(pb, fd, *a, **k)

    monkeypatch.setattr(core, "build_flextool", wrapped)
    return seen


def _run_until_stop(tmp_path_factory, alt: str, values: list, base_chain, fixture_json=None) -> None:
    from flextool.engine import run_chain_from_db

    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=base_chain, **kw)
    with pytest.raises(_Stop):
        run_chain_from_db(url, scen, warm=False, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))


def test_nested_sub_solves_fraction(tmp_path_factory, monkeypatch) -> None:
    """Nested invest -> storage -> rolling dispatch (36 h / 24 h): the per-period invest sub-solves and the
    storage solve cover their periods whole, so the fraction is exactly 1.0 (snapped); the dispatch rolls get
    window / complete hours."""
    seen = _record_builds(monkeypatch, 8)
    chain = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest", "5weeks",
             "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
             "storage_fullYear_6h", "invest_oneYear4Times_5weeks", "bind_battery_forward",
             "invest_twoYears4Times_5weeks", "unlimited_investment"]
    _run_until_stop(tmp_path_factory, "co2roll_nested", [
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]], chain)
    kinds = set()
    for name, hours, share, frac in seen:
        assert set(frac) == set(hours) == set(share), name
        for d in frac:
            if abs(hours[d] - share[d]) <= 1e-9 * hours[d]:
                assert frac[d] == 1.0, (name, d, frac[d])
            else:
                assert frac[d] == pytest.approx(hours[d] / share[d], rel=1e-12), (name, d)
        kinds.add(name.split("_", 1)[0])
    assert kinds == {"invest", "storage", "dispatch"}, [s[0] for s in seen]
    assert any(f[d] < 1.0 for n, _, _, f in seen if n.startswith("dispatch") for d in f)
    assert all(v == 1.0 for n, _, _, f in seen if not n.startswith("dispatch") for v in f.values())


def test_stochastic_rolls_fraction(tmp_path_factory, monkeypatch) -> None:
    """Stochastic rolling (``1_week_rolling_wind``): every roll's fraction is its window hours over the
    complete 168 h, the same for the anchor period and each forecast branch period.  The fixture's realized: yes
    designation adds a realization roll after each stochastic roll (only the anchor period, its jump window):
    its fraction follows the same rule."""
    seen = _record_builds(monkeypatch, 12)
    _run_until_stop(tmp_path_factory, "co2roll_stoch_roll", [], _stoch_chain(), FIX / "stochastics.json")
    assert len(seen) == 12
    stochastic = [s for s in seen if "__realized_dispatch" not in s[0]]
    assert len(stochastic) == 6, [s[0] for s in seen]
    for name, _hours, _share, frac in stochastic:
        assert len(frac) > 1, (name, frac)  # anchor + branch periods
    for name, hours, share, frac in seen:
        assert all(share[d] == pytest.approx(168.0, rel=1e-12) for d in share), (name, share)
        for d in frac:
            assert frac[d] == pytest.approx(hours[d] / 168.0, rel=1e-12), (name, d)
        assert len(set(frac.values())) == 1, (name, frac)
    assert any(next(iter(f.values())) < 1.0 for _, _, _, f in seen)


def _stoch_chain() -> list[str]:
    import json

    data = json.loads((FIX / "stochastics.json").read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == "1_week_rolling_wind"]
    nxt = {a: b for _, a, b in rows}
    first = [a for a in nxt if a not in set(nxt.values())]
    assert len(first) == 1
    chain, alt = [], first[0]
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


# ---------------------------------------------------------------------------
# Rolling stochastic fan: every forecast continues from its period's realized history (#59)
#
# ``1_week_rolling_wind`` (stochastics.json): one period ``period1`` of 168 one-hour steps (timeset ``1week``,
# timeset_duration 168), rolling horizon 96 h, jump 24 h -> 7 stochastic rolls k = 0..6 starting at hour 24k.
# Every roll fans into the forecasts ``period1_lower`` / ``_mid`` / ``_upper`` at its first step; the
# ``realized: yes`` branch makes the engine add a realization roll after each stochastic roll, which realizes
# that roll's 24 h jump under the period label ``period1``.  So stochastic roll k starts with 24k realized
# hours and its forecasts cover min(96, 168 - 24k) more:
#     f_d_k[forecast] = (24k + min(96, 168 - 24k)) / 168 = min(24k + 96, 168) / 168   (roll 1: 120/168 = 0.714)
#     f_d_k[period1]  = 24k / 168                                                    (no steps of its own)
# The group ``co2_limit`` holds gas_market (gas co2_content 0.2, cap 2e6 t/a in period1): gas runs in every
# realization roll, so the realized emissions grow roll by roll (coal does not run there).

STOCH_ENTITIES = [["group", "co2_limit", None], ["group__node", ["co2_limit", "gas_market"], None]]
STOCH_CAP = 2.0e6
STOCH_VALUES = [["commodity", "gas", "co2_content", b64(0.2, "float")],
                ["group", "co2_limit", "co2_method", b64("period", "str")],
                ["group", "co2_limit", "co2_max_period", _pmap({"period1": STOCH_CAP})]]
STOCH_SOLVE = "1week_rolling"
FORECASTS = ("period1_lower", "period1_mid", "period1_upper")


@pytest.fixture(scope="module")
def stochastic_fan_rolls(tmp_path_factory) -> dict:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="co2roll_stoch_fan", values=STOCH_VALUES, base_chain=_stoch_chain(),
                        entities=STOCH_ENTITIES, fixture_json=FIX / "stochastics.json")
    steps = run_chain_from_db(url, scen, keep_solutions=True, work_folder=tmp_path_factory.mktemp("co2roll_fan_w"))
    assert steps and all(st.solution is not None for st in steps.values())
    names = list(steps)
    assert [n for n in names if "__realized_dispatch" not in n] == [f"{STOCH_SOLVE}_roll_{k}" for k in range(7)]
    assert [n for n in names if "__realized_dispatch" in n] == [
        f"{STOCH_SOLVE}__realized_dispatch_roll_{k}" for k in range(7)]
    return steps


def _period_values(p) -> dict[str, float]:
    assert p is not None
    return dict(utf8(p.frame).select("d", "value").iter_rows())


def _forecast_rhs(fd) -> dict[str, float]:
    from tests.engine._lp_rows import row_bounds

    rows = row_bounds(build(fd), "co2_max_period", "co2_limit,")
    return {nm[nm.index(",") + 1:-1]: ub for nm, (_lb, ub) in rows.items()}


def test_stochastic_forecast_f_d_k_counts_realized_hours(stochastic_fan_rolls) -> None:
    """Each forecast's ``f_d_k`` counts the realized hours of its period (the label's carry) plus its own window;
    before, it counted the window alone (roll 1: 96/168 instead of 120/168), so a forecast's period cap allocated
    nothing for the hours already realized."""
    for k in range(7):
        fd = stochastic_fan_rolls[f"{STOCH_SOLVE}_roll_{k}"].flex_data
        fdk = _period_values(fd.p_co2_f_d_k)
        assert set(fdk) == {"period1", *FORECASTS}, k
        assert fdk["period1"] == pytest.approx(24 * k / 168, rel=1e-12, abs=1e-15), k
        for b in FORECASTS:
            assert fdk[b] == pytest.approx(min(24 * k + 96, 168) / 168, rel=1e-12), (k, b)
        rhs = _forecast_rhs(fd)
        assert set(rhs) == set(FORECASTS), k
        assert all(math.isfinite(v) for v in rhs.values()), (k, rhs)


STOCH_TOTAL_CAP = 1.0e7


@pytest.fixture(scope="module")
def stochastic_total_rolls(tmp_path_factory) -> dict:
    """The rolling week with ``co2_max_total`` = 1e7 t on gas_market (co2_method total)."""
    from flextool.engine import run_chain_from_db

    values = [["commodity", "gas", "co2_content", b64(0.2, "float")],
              ["group", "co2_limit", "co2_method", b64("total", "str")],
              ["group", "co2_limit", "co2_max_total", b64(STOCH_TOTAL_CAP, "float")]]
    url, scen = make_db(tmp_path_factory, alt="co2roll_stoch_total", values=values, base_chain=_stoch_chain(),
                        entities=STOCH_ENTITIES, fixture_json=FIX / "stochastics.json")
    steps = run_chain_from_db(url, scen, keep_solutions=True, work_folder=tmp_path_factory.mktemp("co2tot_fan_w"))
    assert steps and all(st.solution is not None for st in steps.values())
    return steps


def test_stochastic_total_cap_reached_share_counts_each_hour_once(stochastic_total_rolls) -> None:
    """``co2_max_total``'s reached share counts the period once, through its label: the f_d_k of a forecast, which
    holds the realized hours.  Stochastic roll k reaches min(24k + 96, 168) / 168 of the week, realization roll k
    (24k + 24) / 168 (years_represented 1, one period); adding the label's own 24k / 168 on top of a forecast's
    f_d_k would count the realized hours twice."""
    steps = stochastic_total_rolls
    for k in range(7):
        stochastic = steps[f"{STOCH_SOLVE}_roll_{k}"].flex_data
        realization = steps[f"{STOCH_SOLVE}__realized_dispatch_roll_{k}"].flex_data
        assert stochastic.co2_horizon_fraction_reached == pytest.approx(min(24 * k + 96, 168) / 168, rel=1e-12), k
        assert realization.co2_horizon_fraction_reached == pytest.approx((24 * k + 24) / 168, rel=1e-12), k


def _gas_tonnes(step) -> float:
    """CO2 of the gas burnt over a solve's steps (gas co2_content 0.2): v_flow · unitsize · slope · step_duration."""
    fd = step.flex_data
    size = float(utf8(fd.p_unitsize.frame).filter(pl.col("p") == "gas_plant")["value"][0])
    vf = utf8(step.solution.value("v_flow")).filter((pl.col("p") == "gas_plant") & (pl.col("source") == "gas_market"))
    slope = utf8(fd.p_slope.frame).filter(pl.col("p") == "gas_plant").select("d", "t", slope="value")
    sd = utf8(fd.p_step_duration.frame).select("d", "t", sd="value")
    rows = vf.join(slope, on=["d", "t"], how="inner").join(sd, on=["d", "t"], how="inner")
    assert rows.height == vf.height > 0
    return float((rows["value"] * size * rows["slope"] * rows["sd"] * 0.2).sum())


def test_stochastic_forecast_period_cap_carries_realized_history(stochastic_fan_rolls) -> None:
    """Each forecast's ``co2_max_period`` RHS continues from its period's realized emissions:
    ``2e6 · f_d_k − R_k / period_share``, R_k = the tonnes emitted by realization rolls 0..k−1 (read from their
    solutions), period_share = 168/8760.  Before, a forecast joined the realized carrier on its own name, found
    nothing and took R = 0, so every forecast could emit the period's realized tonnes again."""
    share = 168 / 8760
    realized = 0.0
    for k in range(7):
        fd = stochastic_fan_rolls[f"{STOCH_SOLVE}_roll_{k}"].flex_data
        if k > 0:
            assert realized > 0.0, k
        rhs = _forecast_rhs(fd)
        assert set(rhs) == set(FORECASTS), k
        for b in FORECASTS:
            want = STOCH_CAP * min(24 * k + 96, 168) / 168 - realized / share
            assert rhs[b] == pytest.approx(want, rel=1e-9), (k, b)
            assert rhs[b] >= 0.0, (k, b)
        realized += _gas_tonnes(stochastic_fan_rolls[f"{STOCH_SOLVE}__realized_dispatch_roll_{k}"])


def test_stochastic_rolls_total_cap_per_forecast_path(stochastic_total_rolls) -> None:
    """Every stochastic roll keeps one ``co2_max_total`` row per forecast path, each with the whole remaining budget
    and only its own forecast's emissions; before, a roll had no scenario paths and summed the three forecasts'
    emissions into one row against one budget (about 3× too tight).

    Hand values: roll k's reached share is min(24k + 96, 168)/168 (one period, years_represented 1), and the
    realized carry R_k is the tonnes the realization rolls 0..k−1 emitted (read from their solutions).  The row's
    LHS weighs a tonne by years_represented / period_share = 1 / (168/8760) = 8760/168 (timeset_duration 168 h,
    one-hour steps), so the carry is subtracted with that weight too:
    RHS = 1e7·min(24k + 96, 168)/168 − R_k·8760/168, the same on every path.
    Roll 0: 1e7·96/168 = 5 714 285.71.  Roll 1: realization roll 0 dispatches hours 1-24; its solution runs
    gas_plant (unitsize 1000, efficiency 0.9) at 1000 MW in 19 h and 0, 160.6, 969.3, 748.7, 896.5 MW in the
    other 5, 21 775.1 MWh out, so R_1 = 21 775.1 / 0.9 · 0.2 = 4 838.9 t (checked here through the input-side
    fuel flow), and RHS = 1e7·120/168 − 4 838.9·8760/168 = 7 142 857.14 − 252 314.6 = 6 890 542.5.  The
    pre-fix RHS subtracted raw tonnes: 7 138 018.2."""
    from tests.engine._lp_rows import row_bounds, row_terms

    realized = 0.0
    for k in range(7):
        fd = stochastic_total_rolls[f"{STOCH_SOLVE}_roll_{k}"].flex_data
        pb = build(fd)
        assert row_bounds(pb, "co2_max_total", "") == {}, k
        rows = row_bounds(pb, "co2_max_total_path", "co2_limit,")
        assert set(rows) == {f"co2_max_total_path[co2_limit,{f.removeprefix('period1_')}]" for f in FORECASTS}, k
        want = STOCH_TOTAL_CAP * min(24 * k + 96, 168) / 168 - realized * 8760 / 168
        if k == 1:
            assert realized == pytest.approx(4838.9, abs=0.05)
            assert want == pytest.approx(6_890_542.5, abs=1.0)
        for nm, (_lb, ub) in rows.items():
            assert ub == pytest.approx(want, rel=1e-9), (k, nm)
        for nm, terms in row_terms(pb, "co2_max_total_path", "co2_limit,").items():
            periods = {col.rsplit(",", 2)[-2] for col in terms if col.startswith("v_flow[")}
            assert periods == {f"period1_{nm[nm.rindex(',') + 1:-1]}"}, (k, nm, periods)
        realized += _gas_tonnes(stochastic_total_rolls[f"{STOCH_SOLVE}__realized_dispatch_roll_{k}"])
    for k in range(7):
        pb = build(stochastic_total_rolls[f"{STOCH_SOLVE}__realized_dispatch_roll_{k}"].flex_data)
        assert row_bounds(pb, "co2_max_total_path", "") == {}, k
        assert set(row_bounds(pb, "co2_max_total", "")) == {"co2_max_total[co2_limit]"}, k


# ---------------------------------------------------------------------------
# co2_max_total binding across rolls: the realized carry is weighted like the LHS

YEARS_5 = [["solve", SOLVE, "years_represented", _pmap({"p2020": 5.0, "p2025": 5.0})]]
BINDING_TOTAL_CAP = 2.0e7


def _total_cap(cap: float) -> list:
    return [["group", "co2_limit", "co2_method", b64("total", "str")],
            ["group", "co2_limit", "co2_max_total", b64(cap, "float")]]


def test_rolling_total_cap_binds_at_the_budget(tmp_path_factory) -> None:
    """A binding ``co2_max_total`` lets a rolling solve emit the budget, not more.

    p2020 + p2025, 2 days (48 one-hour steps) each, years_represented 5 each: co2_max_total's LHS weighs a tonne
    of either period by years_represented / period_share = 5 / (48/8760) = 912.5.  Rolling 48 h / 48 h: roll 0 is
    p2020, roll 1 is p2025, every hour is realized once.  Roll 0 reaches 5·1 / (5 + 5) = 0.5 of the represented
    years, so it may emit 0.5·C represented tonnes; roll 1 reaches 1.0 and may emit C − E_0·912.5.  Unconstrained,
    coal_plant emits about 2e4 t in 48 h (1.8e7 represented), so C = 2e7 binds in both rolls and the two rolls
    together emit exactly C represented tonnes, as the single solve of both periods does.  Before, roll 1 got
    C − E_0 (raw tonnes, about 1.1e4) and the week emitted about 2.8e7."""
    values = _total_cap(BINDING_TOTAL_CAP) + TWO_PERIODS + YEARS_5
    weight = 5 * 8760 / 48

    single = list(_run(tmp_path_factory, "co2tot_single", values).values())
    assert len(single) == 1
    assert sum(_tonnes(single[0]).values()) * weight == pytest.approx(BINDING_TOTAL_CAP, rel=1e-6)
    assert single[0].solution.constraint_dual("co2_max_total")["dual"].to_list()[0] != 0.0

    rolls = list(_run(tmp_path_factory, "co2tot_roll48", values + _rolling(48.0, 48.0)).values())
    assert [st.solve_name for st in rolls] == [_roll(0), _roll(1)]
    represented = [sum(_tonnes(st).values()) * weight for st in rolls]
    assert represented[0] == pytest.approx(0.5 * BINDING_TOTAL_CAP, rel=1e-6)
    assert sum(represented) == pytest.approx(BINDING_TOTAL_CAP, rel=1e-6)
    for st in rolls:
        (dual,) = st.solution.constraint_dual("co2_max_total")["dual"].to_list()
        assert dual != 0.0, st.solve_name


def test_stochastic_rolling_week_total_cap_binds_at_the_budget(tmp_path_factory) -> None:
    """The rolling stochastic week (7 stochastic rolls, each followed by a realization roll that realizes its 24 h)
    with a binding ``co2_max_total`` = 1e6 t: the realized week emits exactly the budget.

    One period of 168 h, years_represented 1: the LHS weighs a tonne by 1 / (168/8760) = 8760/168.  Realization
    roll k reaches (24k + 24)/168 of the week, so it may emit C·(k + 1)/7 minus what rolls 0..k−1 realized (in
    the same weight): the realized total through roll k never exceeds C·(k + 1)/7, and a day that needs less gas
    (windier) leaves its allowance to the later days.  Gas burns about 4.8e3 t on the first day (2.5e5 weighted,
    see the per-path test above), far over 1e6/7 = 1.4e5, so the last realization roll binds and the week's
    realized emissions are C.  Before, the carry was subtracted as raw tonnes, so each realization roll got
    about C·(k + 1)/7 again and the week emitted several times the budget."""
    from flextool.engine import run_chain_from_db

    cap = 1.0e6
    values = [["commodity", "gas", "co2_content", b64(0.2, "float")],
              ["group", "co2_limit", "co2_method", b64("total", "str")],
              ["group", "co2_limit", "co2_max_total", b64(cap, "float")]]
    url, scen = make_db(tmp_path_factory, alt="co2roll_stoch_total_binding", values=values,
                        base_chain=_stoch_chain(), entities=STOCH_ENTITIES, fixture_json=FIX / "stochastics.json")
    steps = run_chain_from_db(url, scen, keep_solutions=True,
                              work_folder=tmp_path_factory.mktemp("co2tot_fan_binding_w"))
    realization = [steps[f"{STOCH_SOLVE}__realized_dispatch_roll_{k}"] for k in range(7)]
    weighted, duals = 0.0, []
    for k, st in enumerate(realization):
        weighted += _gas_tonnes(st) * 8760 / 168
        assert weighted <= cap * (k + 1) / 7 * (1 + 1e-9), k
        (dual,) = st.solution.constraint_dual("co2_max_total")["dual"].to_list()
        duals.append(dual)
    assert duals[-1] != 0.0
    assert weighted == pytest.approx(cap, rel=1e-6)


def test_chained_solves_total_cap_binds_at_the_budget(tmp_path_factory) -> None:
    """Two chained solves (``y2020_2day_dispatch`` realizes p2020, a second single solve realizes p2025), both 2 days
    of one-hour steps with years_represented 5: a tonne weighs 5 / (48/8760) = 912.5 in either solve's
    ``co2_max_total`` row.  The first solve's periods are p2020 alone, so it reaches 1.0 of its represented years
    and the cap C = 2e7 does not bind: coal_plant emits its unconstrained 19 924 t (1.818e7 weighted).  The second
    solve may emit C − 19 924·912.5 = 1.8e6 weighted, so it binds and the chain emits exactly C.  Before, it got
    C − 19 924 (raw tonnes) and emitted another 1.8e7."""
    from flextool.engine import run_chain_from_db

    second = "y2025_2day_dispatch"
    cap = BINDING_TOTAL_CAP
    values = _total_cap(cap) + [
        ["model", "flexTool", "solves", b64({"value_type": "str", "data": [SOLVE, second]}, "array")],
        ["solve", SOLVE, "years_represented", _pmap({"p2020": 5.0})],
        ["solve", second, "solve_mode", b64("single_solve", "str")],
        ["solve", second, "solver", b64("highs", "str")],
        ["solve", second, "period_timeset", _pmap({"p2025": "2day"})],
        ["solve", second, "realized_periods",
         b64({"value_type": "str", "index_name": "period", "data": ["p2025"]}, "array")],
        ["solve", second, "years_represented", _pmap({"p2025": 5.0})],
    ]
    url, scen = make_db(tmp_path_factory, alt="co2tot_chain", values=values, base_chain=CHAIN,
                        entities=[["solve", second, None]])
    steps = run_chain_from_db(url, scen, keep_solutions=True, work_folder=tmp_path_factory.mktemp("co2tot_chain_w"))
    assert list(steps) == [SOLVE, second]
    weight = 5 * 8760 / 48
    first_t = _tonnes(steps[SOLVE])["p2020"]
    assert first_t == pytest.approx(19924.0, rel=1e-6)
    assert (first_t + _tonnes(steps[second])["p2025"]) * weight == pytest.approx(cap, rel=1e-6)
    (dual,) = steps[second].solution.constraint_dual("co2_max_total")["dual"].to_list()
    assert dual != 0.0
    carry = utf8(steps[second].flex_data.p_co2_cum_realized_represented.frame)
    assert carry.select("g", "d").rows() == [("co2_limit", "p2020")]
    assert carry["value"][0] == pytest.approx(first_t * weight, rel=1e-12)
    # a carried row without its weighted value cannot give a budget: refused, not counted as 0
    fd = steps[second].flex_data
    unknown = type(fd.p_co2_cum_realized_represented)(
        ("g", "d"), fd.p_co2_cum_realized_represented.frame.with_columns(value=pl.lit(None, dtype=pl.Float64)))
    with pytest.raises(ValueError, match=r"co2_max_total: the realized CO2 carry .*co2_limit,p2020"):
        build(dataclasses.replace(fd, p_co2_cum_realized_represented=unknown))
