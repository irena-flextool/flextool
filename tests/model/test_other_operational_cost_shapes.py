"""A period × time ``other_operational_cost`` is charged in the objective on every solve type (#74).

The variable-cost terms of the objective are inner-join products of an index set and a cost:

    Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_unit_source} v_flow·unitsize·slope · other_operational_cost_source[p,s,d,t]
    (and the same for the unit output side, connections, and processes without an efficiency)

so a cost cell reaches the LP only if its index row exists.  Before the fix the index sets were the per-solve emit
seeds, whose cascade has no period × time tier: a cost authored as a map keyed by period and time had no index
rows and vanished from the objective (single solve, rolls, nested sub-solves, stochastic solves), while the cost
outputs, which multiply the flows by the per-side cost Params, still reported it.

The criterion (CLAUDE.md "constant-vs-equivalent-map"): a period × time map equal everywhere to ``v`` gives the
scalar-``v`` LP, and a time-varying curve as period × time equals the same curve as a time map, in single,
rolling, nested and stochastic solves; warm = cold in rolls (every roll builds its own LP).

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``1_week_rolling_wind`` of ``stochastics.json`` without its stochastic data, plus the connection ``link``
  (``tests/model/test_rolling_period_maps._deterministic_rolling_json``) and gas_plant capped at 200 MW (so that
  coal_plant runs and its input-side cost reaches the objective): one period ``period1`` (168 h), 7 rolls of 96 h,
  or one single solve.  Closed form of its objective: see :func:`_closed_form`.
- ``2_day_stochastic_dispatch`` of ``stochastics.json``: ``period1`` and four branch periods.
- ``invest_5weeks`` of ``tests.json`` with nested investment: the sub-solves ``invest_5weeks_<p>``.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, utf8
from tests.model.test_rolling_period_maps import (
    NESTED_CHAIN,
    NESTED_PERIODS,
    ROLL_CHAIN,
    T168,
    TESTS_JSON,
    _deterministic_rolling_json,
    pmap,
    ptmap,
    scalar,
    tmap,
)

REPO = Path(__file__).resolve().parents[2]
STOCHASTICS_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
OUTPUT_CONFIG = str(REPO / "flextool" / "schemas" / "default_plots.yaml")
PERIOD = "period1"
T48 = [f"t{i:04d}" for i in range(1, 49)]
T8760 = [f"t{i:04d}" for i in range(1, 8761)]
STOCH_PERIODS = (PERIOD,) + tuple(f"period1_{b}" for b in ("realized", "upper", "lower", "mid"))
COSTS = {"in": 3.0, "out": 2.0, "conn": 1.5}
INDEX_SETS = ("pssdt_varCost_noEff", "pssdt_varCost_eff_unit_source", "pssdt_varCost_eff_unit_sink",
              "pssdt_varCost_eff_connection")
GAS_CAP = 200.0
LINK = [["connection", "link", "existing", scalar(100.0)],
        ["connection", "link", "transfer_method", b64("regular", "str")],
        ["unit", "gas_plant", "existing", scalar(GAS_CAP)]]


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _shaped(v: float, shape: str, periods, timesteps) -> list:
    if shape == "scalar":
        return scalar(v)
    if shape == "pmap":
        return pmap({p: v for p in periods})
    if shape == "tmap":
        return tmap({t: v for t in timesteps})
    return ptmap({p: {t: v for t in timesteps} for p in periods})


def _curve(base: float) -> dict:
    return {t: round(base + 0.37 * (i % 24), 6) for i, t in enumerate(T168)}


def _arc_costs(value) -> list:
    """``value(kind)`` on the unit input arc, the unit output arc and the connection."""
    return [["unit__inputNode", ["coal_plant", "coal_market"], "other_operational_cost", value("in")],
            ["unit__outputNode", ["gas_plant", "demand_node"], "other_operational_cost", value("out")],
            ["connection", "link", "other_operational_cost", value("conn")]]


_RUNS: dict = {}
_WORK: dict = {}      # id(steps) -> the run's work folder (holds output_raw/ for write_outputs)


def _run(tmp_path_factory, key: str, *, fixture_json, base_chain, values, warm: bool = True) -> dict:
    from flextool.engine import run_chain_from_db

    full = json.dumps([key, str(fixture_json), base_chain, values, warm])
    if full not in _RUNS:
        alt = f"c10a_{len(_RUNS)}"
        fj = fixture_json(tmp_path_factory, alt) if callable(fixture_json) else fixture_json
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=base_chain, fixture_json=fj)
        work = tmp_path_factory.mktemp(f"{alt}_w")
        steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True, warm=warm)
        assert steps and all(s.solution is not None for s in steps.values())
        _RUNS[full] = steps
        _WORK[id(steps)] = work
    return _RUNS[full]


def _det(tmp_path_factory, values: list, *, single: bool, warm: bool = True) -> dict:
    values = LINK + list(values)
    if single:
        values.append(["solve", "1week_rolling", "solve_mode", b64("single_solve", "str")])
    return _run(tmp_path_factory, "det", fixture_json=_deterministic_rolling_json, base_chain=ROLL_CHAIN,
                values=values, warm=warm)


def _det_const(tmp_path_factory, shape: str, *, single: bool) -> dict:
    return _det(tmp_path_factory, _arc_costs(lambda k: _shaped(COSTS[k], shape, [PERIOD], T168)), single=single)


def _canon(x) -> pl.DataFrame | None:
    if x is None:
        return None
    f = getattr(x, "frame", x)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    f = utf8(f)
    return f.sort(f.columns)


def _wind_profile_sum(hours: int) -> float:
    """Σ of the ``system`` wind profile of ``stochastics.json`` over its first ``hours`` steps (85.879296713 over
    168 h, 47.220043351 over 96 h)."""
    import base64

    data = json.loads(STOCHASTICS_JSON.read_text())
    pv = next(v for v in data["parameter_values"] if v[0] == "profile" and v[1] == "wind" and v[4] == "system")
    series = json.loads(base64.b64decode(pv[3][0]))["data"]
    return sum(float(v) for _t, v in series[:hours])


def _closed_form(hours: int) -> tuple[float, float]:
    """``(objective, other operational cost per window)`` of the deterministic week with the scalar costs, for a
    window of ``hours`` steps from t0001 that starts and ends with the reservoir at 2 500 MWh.

    Data (stochastics.json ``system`` + this file): demand 3 000 MW; wind ≤ 4 000 · profile (free); hydro: the
    reservoir's inflow 400 MW reaches demand one-to-one (efficiency 2, half to downriver through hydro_divide) and
    start = end, so hydro delivers 400 · hours MWh in total; gas_plant ≤ 200 MW at 6/0.9 + 2 (out cost) = 8.667;
    coal_plant ≤ 2 000 MW at 16/0.9 + 3/0.9 (in cost on the fuel) = 21.111; link (gas_market → coal_market, 100 MW,
    O&M 1.5) sells into coal_market at 16 what it buys at 6: −8.5 · 100 per hour at its cap whatever else runs.

    Gas is cheaper than coal, so it runs at its cap every hour, and coal is the marginal unit in every hour: the
    residual 3 000 − wind − hydro can stay ≥ 200 everywhere (the windiest hour, profile 0.6527, wind 2 611 MW,
    leaves 389 MW, so hydro delivers at most 189 MW there, above downriver's 100, and the reservoir moves the rest
    to coal hours), there is no curtailment and no slack, and how hydro is spread over the hours does not change
    the cost.  Thermal energy T = 3 000·h − 4 000·Σprofile −
    400·h, gas G = 200·h, coal C = T − G.  Per window: 8.667·G + 21.111·C − 850·h; the objective weighs every hour
    by 8760/168 (period share of the full 168-h period, also in a 96-h roll).  O&M: 2·G + 3·C/0.9 + 1.5·100·h.

    168 h: T = 93 282.813, C = 59 682.813, objective **73 436 461.77**, O&M **291 342.71** per week.
    96 h (roll 0): T = 60 719.827, C = 41 519.827, objective **50 126 475.78**."""
    t = 3000.0 * hours - 4000.0 * _wind_profile_sum(hours) - 400.0 * hours
    g = GAS_CAP * hours
    c = t - g
    per_window = (6.0 / 0.9 + COSTS["out"]) * g + (16.0 + COSTS["in"]) / 0.9 * c - (16.0 - 6.0 - COSTS["conn"]) * 100.0 * hours
    om = COSTS["out"] * g + COSTS["in"] * c / 0.9 + COSTS["conn"] * 100.0 * hours
    return per_window * 8760.0 / 168.0, om


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def _assert_same_cost_sets(a, b, name: str) -> None:
    for field in INDEX_SETS + ("p_pssdt_varCost",):
        fa, fb = _canon(getattr(a, field)), _canon(getattr(b, field))
        assert (fa is None) == (fb is None), (name, field, fa, fb)
        if fa is not None:
            assert fa.equals(fb), (name, field)


def _rows(x) -> int:
    return 0 if x is None else _canon(x).height


# ---------------------------------------------------------------------------
# Single solve


def test_period_time_cost_equals_scalar_single_solve(tmp_path_factory) -> None:
    """The scalar run's objective is the closed form of :func:`_closed_form` (73 436 461.77); the period × time run
    equals it.  Before the fix the period × time run had no index rows and its objective lacked the O&M cost (on the
    fixture without the gas cap: 23 666 882.67 vs 34 708 947.47)."""
    pt = next(iter(_det_const(tmp_path_factory, "ptmap", single=True).values()))
    sc = next(iter(_det_const(tmp_path_factory, "scalar", single=True).values()))
    d = pt.flex_data
    assert (_rows(d.pssdt_varCost_eff_unit_source), _rows(d.pssdt_varCost_eff_unit_sink),
            _rows(d.pssdt_varCost_eff_connection)) == (168, 168, 336)
    _assert_same_cost_sets(d, sc.flex_data, "single")
    cost = _canon(d.p_pssdt_varCost)
    assert set(cost["value"].to_list()) == {1.5, 2.0, 3.0}
    want, _om = _closed_form(168)
    assert want == pytest.approx(73_436_461.77, abs=0.01)
    assert float(sc.solution.obj) == pytest.approx(want, rel=1e-8)
    assert _rel(float(pt.solution.obj), float(sc.solution.obj)) < 1e-9


@pytest.mark.parametrize("shape", ["pmap", "tmap"])
def test_other_shapes_equal_scalar_single_solve(tmp_path_factory, shape: str) -> None:
    """Guard: the shapes that worked before the fix keep the scalar LP."""
    m = next(iter(_det_const(tmp_path_factory, shape, single=True).values()))
    sc = next(iter(_det_const(tmp_path_factory, "scalar", single=True).values()))
    _assert_same_cost_sets(m.flex_data, sc.flex_data, shape)
    assert _rel(float(m.solution.obj), float(sc.solution.obj)) < 1e-9


def _write_outputs(steps: dict, monkeypatch) -> Path:
    """The csv outputs of a run, as ``tests/test_cost_aggregation_semantics._run_scenario`` writes them."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    workdir = _WORK[id(steps)]
    monkeypatch.chdir(workdir)
    last = next(reversed(steps.values()))
    provider = getattr(last, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(workdir)
    write_outputs(
        scenario_name="om_cost", output_location=str(workdir), subdir="om_cost", output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"], fallback_output_location=str(workdir), raw_output_dir=str(workdir / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider())
    return workdir / "output_csv" / "om_cost"


def _summary(csv_dir: Path) -> tuple[float, float]:
    """``(solver objective, calculated total)`` of ``summary_solve.csv`` (M currency)."""
    import csv

    objective = total = None
    with open(csv_dir / "summary_solve.csv") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            if "Total cost (calculated) full horizon" in row[0]:
                total = float(row[1])
            elif objective is None and row[0] and not row[0].startswith(('"', "Created", "CapMargin", "Period")):
                try:
                    objective = float(row[1])
                except ValueError:
                    continue
    assert objective is not None and total is not None
    return objective, total


def _other_operational(csv_dir: Path) -> float:
    import pandas as pd

    costs = pd.read_csv(csv_dir / "costs__dt.csv", index_col=[0, 1, 2])
    return float(costs["other operational"].sum())


def test_outputs_equal_lp_cost(tmp_path_factory, monkeypatch) -> None:
    """The outputs and the LP agree on the O&M cost (the bug: the outputs showed a cost the LP never charged).

    For the period × time run and the scalar run the calculated total equals the solver objective, and the O&M
    output (``costs__dt`` *other operational*, CUR per step, summed over the week) is the closed form
    2·G + 3·C/0.9 + 1.5·100·168 = 291 342.71 (:func:`_closed_form`).  Before the fix the period × time run's
    objective lacked the O&M cost that its outputs still contained."""
    out = {}
    for shape in ("ptmap", "scalar"):
        steps = _det_const(tmp_path_factory, shape, single=True)
        csv_dir = _write_outputs(steps, monkeypatch)
        objective, total = _summary(csv_dir)
        assert total == pytest.approx(objective, rel=1e-6), (shape, objective, total)
        out[shape] = _other_operational(csv_dir)
    _obj, want = _closed_form(168)
    assert want == pytest.approx(291_342.71, abs=0.01)
    assert out["scalar"] == pytest.approx(want, rel=1e-6)
    assert out["ptmap"] == pytest.approx(out["scalar"], rel=1e-9)


# ---------------------------------------------------------------------------
# Rolling (synthetic sub-solves)


def _rolls(steps: dict) -> list[str]:
    rolls = [k for k in steps if "_roll_" in k]
    assert len(rolls) == 7 and rolls == list(steps), list(steps)
    return rolls


def test_period_time_cost_equals_scalar_in_rolls(tmp_path_factory) -> None:
    """Every roll: same index sets and objective; roll 0 (t0001–t0096, reservoir back at its 2 500 MWh reference
    value at the window's end) is the 96-h closed form of :func:`_closed_form`, 50 126 475.78.  Before the fix the
    period × time rolls lacked the O&M cost (on the fixture without the gas cap roll 0 had 16 101 651.90 vs
    23 184 719.53)."""
    pt = _det_const(tmp_path_factory, "ptmap", single=False)
    sc = _det_const(tmp_path_factory, "scalar", single=False)
    for name in _rolls(pt):
        _assert_same_cost_sets(pt[name].flex_data, sc[name].flex_data, name)
        assert _rows(pt[name].flex_data.pssdt_varCost_eff_connection) > 0, name
        assert _rel(float(pt[name].solution.obj), float(sc[name].solution.obj)) < 1e-9, name
    want, _om = _closed_form(96)
    assert want == pytest.approx(50_126_475.78, abs=0.01)
    assert float(sc[_rolls(sc)[0]].solution.obj) == pytest.approx(want, rel=1e-6)


def _det_curve(tmp_path_factory, shape: str, *, single: bool, warm: bool = True) -> dict:
    base = {"in": 3.0, "out": 2.0, "conn": 1.5}
    wrap = (lambda c: ptmap({PERIOD: c})) if shape == "ptmap" else tmap
    return _det(tmp_path_factory, _arc_costs(lambda k: wrap(_curve(base[k]))), single=single, warm=warm)


@pytest.mark.parametrize("single", [True, False], ids=["single", "rolls"])
def test_period_time_curve_equals_time_curve(tmp_path_factory, single: bool) -> None:
    """A time-varying cost as a period × time map and as a time map: the same index sets and objective."""
    pt = _det_curve(tmp_path_factory, "ptmap", single=single)
    tm = _det_curve(tmp_path_factory, "tmap", single=single)
    assert list(pt) == list(tm)
    for name in pt:
        _assert_same_cost_sets(pt[name].flex_data, tm[name].flex_data, name)
        assert _rel(float(pt[name].solution.obj), float(tm[name].solution.obj)) < 1e-9, name


def test_warm_rolling_period_time_cost_matches_cold(tmp_path_factory) -> None:
    """Warm = cold in every roll with a time-varying period × time cost on the unit arcs and the connection, and
    both equal the warm run of the same curve as a time map (before the fix both lost the cost)."""
    cold = _det_curve(tmp_path_factory, "ptmap", single=False, warm=False)
    hot = _det_curve(tmp_path_factory, "ptmap", single=False, warm=True)
    ref = _det_curve(tmp_path_factory, "tmap", single=False, warm=True)
    for name in _rolls(cold):
        assert _rel(float(hot[name].solution.obj), float(cold[name].solution.obj)) < 1e-9, name
        assert _rel(float(hot[name].solution.obj), float(ref[name].solution.obj)) < 1e-9, name


# ``y2020_2day_dispatch`` of tests.json rolled 24 h / 12 h (the warm vehicle of the balance-penalty tests):
# every roll builds its own LP (its dt labels shift).  coal_plant (an efficiency, the eff_unit_* terms) and
# coal_chp (indirect, noEff).
WARM_CHAIN = ["init", "west", "coal", "coal_chp", "heat"]
WARM_VALUES = [
    ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")],
    ["node", "west", "inflow", b64(-1000.0, "float")],
    ["node", "heat", "inflow", b64(0.0, "float")],
]


def _warm_costs(value) -> list:
    return [["unit__inputNode", [u, "coal_market"], "other_operational_cost", value(3.0)] for u in ("coal_plant",
                                                                                                    "coal_chp")] + \
           [["unit__outputNode", [u, "west"], "other_operational_cost", value(2.0)] for u in ("coal_plant",
                                                                                              "coal_chp")]


def _warm_run(tmp_path_factory, value, warm: bool) -> dict:
    return _run(tmp_path_factory, "warm", fixture_json=TESTS_JSON, base_chain=WARM_CHAIN,
                values=WARM_VALUES + _warm_costs(value), warm=warm)


def _rising(base: float) -> dict:
    from tests.engine._unit_db import TIMESTEPS

    return {t: round(base + 0.01 * i, 6) for i, t in enumerate(TIMESTEPS)}


def test_rolling_cost_shapes_rebuild_every_roll(tmp_path_factory) -> None:
    """The DB-direct index sets on the warm path:

    - constant costs: no roll reuses the previous roll's LP (``dt`` labels shift), and every set is non-empty
      (eff_unit_source / _sink 24 rows, noEff 48);
    - a rising cost as a period × time map: warm = cold per roll, and equal to the same curve as a time map."""
    const = _warm_run(tmp_path_factory, scalar, warm=True)
    assert [s.warm_used for s in const.values()] == [False] * len(const)
    d = next(iter(const.values())).flex_data
    assert (_rows(d.pssdt_varCost_eff_unit_source), _rows(d.pssdt_varCost_eff_unit_sink),
            _rows(d.pssdt_varCost_noEff)) == (24, 24, 48)
    hot = _warm_run(tmp_path_factory, lambda v: ptmap({"p2020": _rising(v)}), warm=True)
    hot_tmap = _warm_run(tmp_path_factory, lambda v: tmap(_rising(v)), warm=True)
    cold = _warm_run(tmp_path_factory, lambda v: ptmap({"p2020": _rising(v)}), warm=False)
    assert list(hot) == list(cold) == list(hot_tmap)
    for name in hot:
        assert _rel(float(hot[name].solution.obj), float(cold[name].solution.obj)) < 1e-9, name
        assert _rel(float(hot[name].solution.obj), float(hot_tmap[name].solution.obj)) < 1e-9, name
    assert not any(s.warm_used for run in (hot, hot_tmap) for s in run.values())


# ---------------------------------------------------------------------------
# Stochastic and nested sub-solves


def _stoch_chain() -> list[str]:
    rows = [r for r in json.loads(STOCHASTICS_JSON.read_text())["scenario_alternatives"]
            if r[0] == "2_day_stochastic_dispatch"]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def test_period_time_cost_stochastic_branches(tmp_path_factory) -> None:
    """``2_day_stochastic_dispatch``, unit arcs: ``{period1: {t: v}}`` vs scalar.  The branch periods inherit the
    anchor's cells (index rows on the 4 forecast branch periods; the period label period1 has no dispatch); same
    objective (before the fix the map lost the branch cells and the objectives differed)."""
    def values(shape):
        return [["unit__inputNode", ["coal_plant", "coal_market"], "other_operational_cost",
                 _shaped(COSTS["in"], shape, [PERIOD], T48)],
                ["unit__outputNode", ["gas_plant", "demand_node"], "other_operational_cost",
                 _shaped(COSTS["out"], shape, [PERIOD], T48)]]
    pt = next(iter(_run(tmp_path_factory, "stoch", fixture_json=STOCHASTICS_JSON, base_chain=_stoch_chain(),
                        values=values("ptmap")).values()))
    sc = next(iter(_run(tmp_path_factory, "stoch", fixture_json=STOCHASTICS_JSON, base_chain=_stoch_chain(),
                        values=values("scalar")).values()))
    _assert_same_cost_sets(pt.flex_data, sc.flex_data, "stoch")
    for field in ("pssdt_varCost_eff_unit_source", "pssdt_varCost_eff_unit_sink"):
        idx = _canon(getattr(pt.flex_data, field))
        assert set(idx["d"].to_list()) == set(STOCH_PERIODS) - {PERIOD}, field
        assert idx.height == 4 * 48, field
    assert _rel(float(pt.solution.obj), float(sc.solution.obj)) < 1e-9


def test_period_time_cost_nested_sub_solves(tmp_path_factory) -> None:
    """Nested ``invest_5weeks``: ``coal_plant ← coal_market`` as a period × time map over the four periods vs the
    scalar: same index rows and objective per sub-solve (before the fix p2020 had 873 968 539.38 vs
    1 000 967 201.21)."""
    def values(shape):
        return [["unit__inputNode", ["coal_plant", "coal_market"], "other_operational_cost",
                 _shaped(COSTS["in"], shape, NESTED_PERIODS, T8760)],
                ["solve", "invest_5weeks", "contains_solves",
                 b64({"type": "array", "value_type": "str", "data": []}, "array")]]
    pt = _run(tmp_path_factory, "nested", fixture_json=TESTS_JSON, base_chain=NESTED_CHAIN, values=values("ptmap"))
    sc = _run(tmp_path_factory, "nested", fixture_json=TESTS_JSON, base_chain=NESTED_CHAIN,
              values=values("scalar"))
    assert list(pt) == [f"invest_5weeks_{p}" for p in NESTED_PERIODS] == list(sc)
    for name in pt:
        _assert_same_cost_sets(pt[name].flex_data, sc[name].flex_data, name)
        assert _rows(pt[name].flex_data.pssdt_varCost_eff_unit_source) > 0, name
        assert _rel(float(pt[name].solution.obj), float(sc[name].solution.obj)) < 1e-9, name
