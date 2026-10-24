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
import sys

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, utf8

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


def _run(tmp_path_factory, alt: str, values: list, *, warm: bool = False, base_chain=CHAIN,
         fixture_json=None) -> dict:
    from flextool.engine import run_chain_from_db

    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=base_chain, **kw)
    steps = run_chain_from_db(url, scen, warm=warm, keep_solutions=True,
                              work_folder=tmp_path_factory.mktemp(f"{alt}_w{int(warm)}"))
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


def test_each_roll_caps_its_window(tmp_path_factory) -> None:
    """Rolling 24 h / 12 h: each roll's window emissions, annualised over the window's hours, equal the cap
    (binding); the window fraction is 24/48 for the first three rolls and 12/48 for the truncated last one."""
    cap = 3.0e6
    rolls = list(_run(tmp_path_factory, "co2roll_roll24", [_cap({"p2020": cap})] + _rolling(24.0, 12.0)).values())
    assert [_fraction(st.flex_data) for st in rolls] == [{"p2020": f} for f in (0.5, 0.5, 0.5, 0.25)]
    for st in rolls:
        p = st.flex_data.p_window_fraction
        assert p.frame.equals(p.frame.sort("d"))
        hours = _hours(st.flex_data)["p2020"]
        rate = _tonnes(st)["p2020"] * 8760 / hours
        assert rate <= cap * (1 + 1e-9), st.solve_name
        assert rate == pytest.approx(cap, rel=1e-6), st.solve_name


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
        # The cold run also writes the CSV dump (no effect on the LP), so a roll can be reloaded from disk.
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
    # (b) each roll and period: annualised window emissions within that period's cap
    for i in range(16):
        st = cold[_roll(i)]
        hours = _hours(st.flex_data)
        for d, e in _tonnes(st).items():
            assert e * 8760 / hours[d] <= CAPS[d] * (1 + 1e-9), (i, d)
    # (c) warm == cold; no roll reuses the previous roll's LP (its ``dt`` labels shift)
    for i in range(16):
        assert not hot[_roll(i)].warm_used, f"roll {i} reused an LP built for another window"
    for i in range(16):
        assert hot[_roll(i)].solution.obj == pytest.approx(cold[_roll(i)].solution.obj, rel=1e-9), i


def test_window_fraction_survives_dump_reload(boundary_runs, tmp_path) -> None:
    """A dumped roll (CSV workdir, no DB reader) rebuilds its own fraction at the load boundary."""
    from flextool.engine import load_flextool

    for i, want in ((5, {"p2020": 18 / 48, "p2025": 12 / 48}), (15, {"p2025": 6 / 48})):
        fd = boundary_runs[False][_roll(i)].flex_data
        out = tmp_path / f"dumped_roll_{i}"
        fd.dump_csvs(out, copy_meta_from=boundary_runs["csv_workdir"])
        redo = load_flextool(out)
        assert _fraction(redo) == _fraction(fd), i
        assert _fraction(fd) == pytest.approx(want, rel=1e-12), i


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
    complete 168 h, the same for the anchor period and each forecast branch period."""
    seen = _record_builds(monkeypatch, 6)
    _run_until_stop(tmp_path_factory, "co2roll_stoch_roll", [], _stoch_chain(), FIX / "stochastics.json")
    assert len(seen) == 6
    for name, hours, share, frac in seen:
        assert len(frac) > 1, (name, frac)  # anchor + branch periods
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
