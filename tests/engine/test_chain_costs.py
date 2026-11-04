"""Reported costs of chained solves: discount factors, lumps and windows from the start of the run.

Fixture ``tests.json`` (built from JSON).  ``model.inflation_rate`` r = 0.04 (alternative
``invest_solveSequence_5weeks``), ``inflation_offset_operations`` 0.5 and ``inflation_offset_investment`` 0 (the
defaults), four periods p2020 … p2035 of 5 years each, so the run years are Y = 0, 5, 10, 15 and the horizon ends at
20.  Hand derivation of the factors (``_npv`` "Run-level fields of the reported costs")::

    F_ops(d) = Σ_{y=0..4} 1.04^-(Y_d + y + 0.5)        F_inv(d) = Σ_{y=0..4} 1.04^-(Y_d + y)

coal_plant: invest_cost 400 CUR/kW, discount_rate 0.05, lifetime 30 (longer than the run), no lifetime_method
(reinvest_automatic: the operating window of a decision is every run period from its start of operation on),
fixed_cost 100 CUR/kW/yr.  Annuity A = 400 · 1000 · 0.05 / (1 − 1.05^−30) CUR/MW/yr.  So, for a decision in d with
start of operation c(d)::

    lump_invest[d] = A · Σ_{d' ≥ c(d)} F_inv(d')        lump_fixed[d] = 100 · 1000 · Σ_{d' ≥ c(d)} F_ops(d')

The chain ``invest_solveSequence_5weeks`` (y2020 … y2035: each solve holds its own period and the next one) gets
these values in every solve, while each solve's own factors restart at its first period.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, utf8
from tests.engine.test_construction_lead_time_calendar import ROLLING, ROLLING_VALUES
from tests.engine.test_construction_lead_time_limits import ENTITIES, _values
from tests.engine.test_user_constraint_capacity_terms import CHAIN, PLANT, SINGLE

PERIODS = ["p2020", "p2025", "p2030", "p2035"]
YEARS = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}


def _f(d: str, offset: float) -> float:
    return sum(1.04 ** -(YEARS[d] + y + offset) for y in range(5))


F_OPS = {d: _f(d, 0.5) for d in PERIODS}
F_INV = {d: _f(d, 0.0) for d in PERIODS}
ANNUITY = 400.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -30)
FIXED = 100.0 * 1000.0


def _later(d: str) -> list[str]:
    return PERIODS[PERIODS.index(d):]


def _d(param) -> dict[str, float]:
    frame = utf8(param.frame)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _window(frame: pl.DataFrame, anchor: str, entity: str = PLANT) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a, d in utf8(frame).filter(pl.col("e") == entity).sort(anchor, "d").select(anchor, "d").iter_rows():
        out.setdefault(a, []).append(d)
    return out


def _steps(tmp_path_factory, alt: str, values: list, chain: list[str]) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
    return run(tmp_path_factory, url, scen)


def _assert_run_factors(fd) -> None:
    op, inv = _d(fd.p_inflation_op_run), _d(fd.p_inflation_inv_run)
    assert sorted(op) == PERIODS and sorted(inv) == PERIODS
    for d in PERIODS:
        assert op[d] == pytest.approx(F_OPS[d], rel=1e-12), d
        assert inv[d] == pytest.approx(F_INV[d], rel=1e-12), d


def test_chain_reports_from_the_start_of_the_run(tmp_path_factory) -> None:
    """Every solve of the chain gets the factors of all four run periods from the start of the run, and the
    lumps of its decisions over the operating window up to the end of the run (p2035), not only over its own two
    periods; the solve's own factors (the objective's) are unchanged and restart at its first period."""
    steps = _steps(tmp_path_factory, "chain_run_fields", [], CHAIN)
    assert list(steps) == ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]
    for k, (name, step) in enumerate(steps.items()):
        fd = step.flex_data
        own = PERIODS[k:k + 2]
        _assert_run_factors(fd)
        # The objective's factors: discounted from the solve's own first period.
        assert _d(fd.p_inflation_op) == pytest.approx({d: F_OPS[PERIODS[i]] for i, d in enumerate(own)},
                                                       rel=1e-12), name
        assert _window(fd.edd_operating_window, "d_invest") == {d: _later(d) for d in own}, name
        assert _ed(fd.ed_annuity) == pytest.approx({d: ANNUITY for d in own}, rel=1e-12), name
        assert _ed(fd.ed_entity_annual_discounted_run) == pytest.approx(
            {d: ANNUITY * sum(F_INV[x] for x in _later(d)) for d in own}, rel=1e-12), name
        fixed = _ed(fd.ed_lifetime_fixed_cost_run)
        for d in own:
            assert fixed[d] == pytest.approx(FIXED * sum(F_OPS[x] for x in _later(d)), rel=1e-12), (name, d)
    # 100 MW decided in p2025: fixed cost 100 · 1e5 · (3.7315 + 3.0671 + 2.5209) = 93.19 M (the single solve's).
    assert 100.0 * _ed(steps["y2025_5week"].flex_data.ed_lifetime_fixed_cost_run)["p2025"] == pytest.approx(
        93.1948e6, rel=1e-5)


def test_lead_time_window_starts_at_start_of_operation(tmp_path_factory) -> None:
    """coal_plant with a 5-year construction lead time: the y2020 solve offers only the p2020 decision, which
    starts to operate in p2025.  Its window is p2025 … p2035 and its lumps equal those of a p2025 decision
    without a lead time (same operating years, same invest and fixed cost); the p2025 decision (which would start
    after the solve) is not a vintage of the window."""
    steps = _steps(tmp_path_factory, "chain_run_lead",
                   [["unit", PLANT, "construction_lead_time", b64(5.0, "float")]], CHAIN)
    fd = steps["y2020_5week"].flex_data
    _assert_run_factors(fd)
    assert _window(fd.edd_operating_window, "d_invest") == {"p2020": _later("p2025")}
    assert _ed(fd.ed_entity_annual_discounted_run)["p2020"] == pytest.approx(
        ANNUITY * sum(F_INV[x] for x in _later("p2025")), rel=1e-12)
    assert _ed(fd.ed_lifetime_fixed_cost_run)["p2020"] == pytest.approx(
        FIXED * sum(F_OPS[x] for x in _later("p2025")), rel=1e-12)


_RUN_FIELDS = ("ed_entity_annual_discounted", "ed_entity_annual_divest_discounted", "ed_lifetime_fixed_cost",
               "ed_lifetime_fixed_cost_divest")


def _assert_run_is_solve(fd) -> None:
    assert fd.p_inflation_op_run is fd.p_inflation_op
    for name in _RUN_FIELDS:
        assert getattr(fd, f"{name}_run") is getattr(fd, name), name


def test_single_solve_run_fields_are_its_own(tmp_path_factory) -> None:
    """One solve over the whole run: the run fields ARE the solve's fields (the reported costs do not change),
    and the investment factor and windows are those of the solve."""
    steps = _steps(tmp_path_factory, "single_run_fields", [], SINGLE)
    (fd,) = [s.flex_data for s in steps.values()]
    _assert_run_is_solve(fd)
    _assert_run_factors(fd)
    assert _window(fd.edd_operating_window, "d_invest") == {d: _later(d) for d in PERIODS}
    lump = _ed(fd.ed_entity_annual_discounted_run)
    assert lump == pytest.approx({d: ANNUITY * sum(F_INV[x] for x in _later(d)) for d in PERIODS}, rel=1e-12)


def _rolling(periods: dict[str, str]) -> list:
    def arr(items: list[str]) -> list:
        return b64({"type": "array", "value_type": "str", "data": items, "index_name": "period"}, "array")

    def pmap(data: dict) -> list:
        return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in data.items()]}, "map")
    return ROLLING_VALUES + [
        ["model", "flexTool", "inflation_rate", b64(0.04, "float")],
        ["solve", "y2020_2day_dispatch", "period_timeset", pmap(periods)],
        ["solve", "y2020_2day_dispatch", "realized_periods", arr(list(periods))],
        ["solve", "y2020_2day_dispatch", "years_represented", pmap({d: 5.0 for d in periods})]]


def test_rolls_of_one_period_use_their_own_factors(tmp_path_factory) -> None:
    """A rolling solve over p2020 alone: every roll holds the whole run, so its run factors are its own."""
    steps = _steps(tmp_path_factory, "roll_one_period", _rolling({"p2020": "2day"}), ROLLING)
    assert len(steps) > 1
    for name, step in steps.items():
        fd = step.flex_data
        _assert_run_is_solve(fd)
        assert _d(fd.p_inflation_op_run) == pytest.approx({"p2020": F_OPS["p2020"]}, rel=1e-12), name
        assert _d(fd.p_inflation_inv_run) == pytest.approx({"p2020": F_INV["p2020"]}, rel=1e-12), name


def test_rolls_across_a_period_boundary_discount_from_the_run_start(tmp_path_factory) -> None:
    """A rolling solve over p2020 and p2025: every roll, also one that holds only p2025, gets the factors of both
    run periods from the start of the run; a roll's own factor (the rolling solve's, which starts the run) is the
    same for the periods it holds."""
    steps = _steps(tmp_path_factory, "roll_two_periods", _rolling({"p2020": "2day", "p2025": "2day"}), ROLLING)
    late = 0
    for name, step in steps.items():
        fd = step.flex_data
        run_op = _d(fd.p_inflation_op_run)
        assert run_op == pytest.approx({d: F_OPS[d] for d in ("p2020", "p2025")}, rel=1e-12), name
        assert _d(fd.p_inflation_inv_run) == pytest.approx({d: F_INV[d] for d in ("p2020", "p2025")},
                                                            rel=1e-12), name
        own = _d(fd.p_inflation_op)
        assert own == pytest.approx({d: run_op[d] for d in own}, rel=1e-12), name
        late += list(own) == ["p2025"]
    assert late > 0, "no roll holds p2025 alone"


def test_run_window_keeps_the_fan_lineage() -> None:
    """The run walk of a stochastic fan counts each calendar period once along a lineage: a decision in the
    branch period p2035_high operates in p2035_high and p2040_high, not in the other branch's p2040_low; a period
    of the run outside the solve (p2045) is in every window; the bounded window stops at the end of the
    lifetime (yr 0 + 10)."""
    from flextool.engine.preprocess.params._npv import _run_window

    yr = {"p2035_high": 0.0, "p2035_low": 0.0, "p2040_high": 5.0, "p2040_low": 5.0, "p2045": 10.0}
    in_use = ["p2035_high", "p2035_low", "p2040_high", "p2040_low"]
    lineage = pl.DataFrame({"d": ["p2035_high", "p2035_high", "p2035_low", "p2035_low"],
                            "d_other": ["p2035_high", "p2040_high", "p2035_low", "p2040_low"]})
    anchors = pl.DataFrame({"e": ["u", "u"], "d": ["p2035_high", "p2035_high"], "kind": ["unbounded", "bounded"],
                            "yr_avail": [0.0, 0.0], "life": [0.0, 10.0]})
    walk = _run_window(anchors, yr, lineage, in_use)
    got = {k: sorted(walk.filter(pl.col("kind") == k)["d_all"].to_list()) for k in ("unbounded", "bounded")}
    assert got == {"unbounded": ["p2035_high", "p2040_high", "p2045"], "bounded": ["p2035_high", "p2040_high"]}


# ---------------------------------------------------------------------------
# Reported costs of chained solves (the written results)
# ---------------------------------------------------------------------------
#
# The probe of the design: coal_plant pinned to 100 MW coming online in p2025 and 50 MW in p2030 (``_values`` of
# ``test_construction_lead_time_limits``; L = construction lead time), the battery without investment, so that
# the chain and the single solve decide and dispatch alike.  The chain must then report what the single solve
# reports: the same costs per period, discounted from the start of the run, each counted once.

COST_STEMS = ("cost_unit_discounted__d", "cost_unit_annualized__d", "cost_connection_discounted__d",
              "cost_connection_annualized__d", "cost_node_discounted__d", "cost_node_annualized__d",
              "annualized_costs__d", "costs_discounted", "group_CO2_price_period__d")
ONLINE = {"p2025": 100.0, "p2030": 50.0}
MEGA = 1.0e6


def _no_battery_invest(chain: list[str]) -> list[str]:
    return [a for a in chain if a != "battery_invest"]


def _results(tmp_path_factory, alt: str, values: list, chain: list[str], *, entities: list | None = None,
             fixture_json=None) -> dict[str, pd.DataFrame]:
    """Run the scenario and write its CSV results: ``{csv stem: frame}`` of the cost results present."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities, **kw)
    work = tmp_path_factory.mktemp(f"{alt}_w")
    steps = run_chain_from_db(url, scenario_name=scen, work_folder=work, csv_dump=True)
    assert steps and all(s.optimal for s in steps.values())
    last = next(reversed(list(steps.values())))
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
                  flex_data_provider=getattr(last, "flex_data_provider", None),
                  raw_output_dir=str(work / "output_raw"))
    out = work / "output_csv" / scen
    return {stem: pd.read_csv(out / f"{stem}.csv") for stem in COST_STEMS if (out / f"{stem}.csv").exists()}


def _without_solve(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` without its solve column and without rows of zeros (an entity without costs in a period)."""
    keys = [c for c in ("period", "unit", "connection", "node", "category") if c in frame.columns]
    frame = frame.drop(columns=[c for c in ("solve",) if c in frame.columns])
    numeric = [c for c in frame.columns if c not in keys]
    frame = frame[frame[numeric].fillna(0.0).ne(0.0).any(axis=1)]
    return frame.sort_values(keys).reset_index(drop=True)


def _assert_same_results(case: dict[str, pd.DataFrame], twin: dict[str, pd.DataFrame]) -> None:
    """Every cost result of ``case`` equals its ``twin`` (per period and entity; the solve column aside)."""
    assert sorted(case) == sorted(twin)
    for stem in case:
        got, want = _without_solve(case[stem]), _without_solve(twin[stem])
        assert list(got.columns) == list(want.columns), stem
        keys = [c for c in got.columns if not pd.api.types.is_numeric_dtype(got[c])]
        assert got[keys].equals(want[keys]), stem
        for col in got.columns:
            if col not in keys:
                assert got[col].to_numpy() == pytest.approx(want[col].to_numpy(), rel=1e-9, abs=1e-9,
                                                            nan_ok=True), (stem, col)


def _row(frame: pd.DataFrame, period: str, unit: str = PLANT) -> pd.Series:
    rows = frame[(frame["period"] == period) & (frame["unit"] == unit)]
    assert len(rows) == 1, (period, unit)
    return rows.iloc[0]


def _probe_values(lead: float) -> list:
    return _values(lead, ONLINE)


def _assert_annualized_investment(results: dict[str, pd.DataFrame]) -> None:
    """The annualized investment and fixed cost invested of coal_plant: in every period, the yearly amounts of
    every vintage operating in it (reinvest_automatic, lifetime 30: from its start of operation to the end of the
    run).  100 MW operate from p2025, 50 MW more from p2030::

        investment[d]          = MW_operating[d] · A,  A = 400 · 1000 · 0.05 / (1 − 1.05^−30) = 26 020.57 CUR/MW/yr
        fixed cost invested[d] = MW_operating[d] · 100 · 1000
        MW_operating = {p2020: 0, p2025: 100, p2030: 150, p2035: 150}

    so 0, 2.60206, 3.90309, 3.90309 M/yr and 0, 10, 15, 15 M/yr; the system columns (only coal_plant invests)
    are the same.  Before: the lump in the period it is reported, divided by that period's factor (p2025 only)."""
    operating = {"p2020": 0.0, "p2025": 100.0, "p2030": 150.0, "p2035": 150.0}
    unit = results["cost_unit_annualized__d"]
    system = results["annualized_costs__d"].set_index("period")
    for d, mw in operating.items():
        rows = unit[(unit["period"] == d) & (unit["unit"] == PLANT)]
        invest = rows["investment"].sum() if len(rows) else 0.0
        fixed = rows["fixed cost invested"].sum() if len(rows) else 0.0
        assert invest * MEGA == pytest.approx(mw * ANNUITY, rel=1e-7, abs=1e-6), d
        assert fixed * MEGA == pytest.approx(mw * FIXED, rel=1e-7, abs=1e-6), d
        assert system.loc[d, "unit investment & retirement"] * MEGA == pytest.approx(mw * ANNUITY, rel=1e-7,
                                                                                     abs=1e-6), d
        assert system.loc[d, "fixed cost invested"] * MEGA == pytest.approx(mw * FIXED, rel=1e-7, abs=1e-6), d


@pytest.mark.solver
def test_chain_reports_the_costs_of_the_single_solve(tmp_path_factory) -> None:
    """No lead time, with a CO2 cap per period: every cost result of the chain equals the single solve's, and
    the CO2 price of a period (its dual ÷ the realizing solve's own factor) is unchanged.

    Hand anchors (the module docstring's factors): fixed cost of the 100 MW decided in p2025, over p2025 …
    p2035, 100 · 1e5 · (F_ops(p2025) + F_ops(p2030) + F_ops(p2035)) = 93.1948 M; the fixed cost of the existing
    500 MW in p2030, 500 · 1e5 · F_ops(p2030) = 153.35 M, which is 50 M a year (before: the chain charged
    100 · 1e5 · (F(p2025) + F(p2030)) from the solve's own start = 82.72 M, and 600 MW in p2030)."""
    values = _probe_values(0.0)
    single = _results(tmp_path_factory, "costs_single", values, _no_battery_invest(SINGLE) + ["co2_limit"],
                      entities=ENTITIES)
    chain = _results(tmp_path_factory, "costs_chain", values, _no_battery_invest(CHAIN) + ["co2_limit"],
                     entities=ENTITIES)
    _assert_same_results(chain, single)
    discounted = chain["cost_unit_discounted__d"]
    assert _row(discounted, "p2025")["fixed cost invested"] * MEGA == pytest.approx(
        100.0 * FIXED * sum(F_OPS[d] for d in _later("p2025")), rel=1e-7)
    assert _row(discounted, "p2030")["fixed cost pre-existing"] * MEGA == pytest.approx(
        500.0 * FIXED * F_OPS["p2030"], rel=1e-7)
    assert _row(chain["cost_unit_annualized__d"], "p2030")["fixed cost pre-existing"] * MEGA == pytest.approx(
        500.0 * FIXED, rel=1e-7)
    _assert_annualized_investment(chain)
    _assert_annualized_investment(single)
    price = chain["group_CO2_price_period__d"].set_index("period")["co2_limit"]
    assert (price > 0).any()


@pytest.mark.solver
def test_lead_time_costs_in_the_period_capacity_starts_operating(tmp_path_factory) -> None:
    """Construction lead time 5 years: the decisions of p2020 (100 MW) and p2025 (50 MW) start operating in p2025
    and p2030.  The chain reports what the single solve reports, and both report the investment and fixed cost
    of new capacity where the capacity starts operating, as for the same capacity built without a lead time:
    nothing in p2020 (the y2020 solve, which decides the p2020 investment, reports no investment cost), the
    p2020 decision in p2025, reported by the solve that realizes p2025."""
    values = _probe_values(5.0)
    single = _results(tmp_path_factory, "lead_single", values, _no_battery_invest(SINGLE), entities=ENTITIES)
    chain = _results(tmp_path_factory, "lead_chain", values, _no_battery_invest(CHAIN), entities=ENTITIES)
    no_lead = _results(tmp_path_factory, "lead_none", _probe_values(0.0), _no_battery_invest(SINGLE),
                       entities=ENTITIES)
    _assert_same_results(chain, single)
    _assert_same_results(single, no_lead)
    discounted = chain["cost_unit_discounted__d"]
    first = _row(discounted, "p2020")
    assert first["investment"] == 0.0 and first["fixed cost invested"] == 0.0
    start = _row(discounted, "p2025")
    assert start["solve"] == "y2025_5week"
    assert start["investment"] * MEGA == pytest.approx(100.0 * ANNUITY * sum(F_INV[d] for d in _later("p2025")),
                                                       rel=1e-7)
    # The annualized columns follow the capacity's operation: from p2025 (the p2020 decision) and p2030.
    _assert_annualized_investment(chain)


def _array(values: list[str]) -> list:
    return b64({"value_type": "str", "data": values, "index_name": "period"}, "array")


def _period_map(values: dict) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _solves(names: list[str]) -> list:
    return [["model", "flexTool", "solves",
             b64({"value_type": "str", "data": names, "index_name": "sequence_index"}, "array")]]


def _solve(name: str, periods: list[str], **lists: list[str]) -> list:
    rows = [["solve", name, "solve_mode", b64("single_solve", "str")],
            ["solve", name, "period_timeset", _period_map(dict.fromkeys(periods, "5weeks"))],
            ["solve", name, "years_represented", _period_map(dict.fromkeys(periods, 5.0))]]
    return rows + [["solve", name, param, _array(items)] for param, items in lists.items()]


@pytest.mark.solver
def test_nested_invest_and_dispatch_solves_report_the_single_solve(tmp_path_factory) -> None:
    """An investment solve realizes the decisions of p2020 … p2035 (no dispatch) and a dispatch solve runs on the
    capacity it hands over: the investment cost comes once, from the investment solve, the operating costs from
    the dispatch solve, and the results equal the single solve's."""
    values = (_probe_values(0.0) + _solves(["cc_invest", "cc_dispatch"])
              + _solve("cc_invest", PERIODS, invest_periods=PERIODS, realized_invest_periods=PERIODS)
              + _solve("cc_dispatch", PERIODS, realized_periods=PERIODS))
    entities = ENTITIES + [["solve", "cc_invest", None], ["solve", "cc_dispatch", None]]
    nested = _results(tmp_path_factory, "nested_costs", values, _no_battery_invest(CHAIN), entities=entities)
    single = _results(tmp_path_factory, "nested_single", _probe_values(0.0), _no_battery_invest(SINGLE),
                      entities=ENTITIES)
    _assert_same_results(nested, single)


@pytest.mark.solver
def test_unrealized_look_ahead_of_the_last_solve(tmp_path_factory) -> None:
    """A chain y2020, y2025, y2030 whose last solve holds p2035 as an unrealized look-ahead: the fixed-cost lumps
    of new capacity cover the run up to the end of p2035, and so does the fixed cost of the existing capacity
    (p2035: 500 · 1e5 · F_ops(p2035)).  The results equal those of one solve over p2020 … p2035 that realizes
    p2020 … p2030."""
    realized = PERIODS[:3]
    chain = _results(tmp_path_factory, "lookahead_chain",
                     _probe_values(0.0) + _solves(["y2020_5week", "y2025_5week", "y2030_5week"]),
                     _no_battery_invest(CHAIN), entities=ENTITIES)
    single = _results(tmp_path_factory, "lookahead_single",
                      _probe_values(0.0) + [["solve", "y2020_2035_5week", "realized_periods", _array(realized)],
                                            ["solve", "y2020_2035_5week", "realized_invest_periods",
                                             _array(realized)]],
                      _no_battery_invest(SINGLE), entities=ENTITIES)
    _assert_same_results(chain, single)
    discounted = chain["cost_unit_discounted__d"]
    assert _row(discounted, "p2035")["fixed cost pre-existing"] * MEGA == pytest.approx(
        500.0 * FIXED * F_OPS["p2035"], rel=1e-7)
    assert _row(discounted, "p2030")["fixed cost invested"] * MEGA == pytest.approx(
        50.0 * FIXED * (F_OPS["p2030"] + F_OPS["p2035"]), rel=1e-7)


@pytest.mark.solver
def test_rolls_across_a_period_boundary_report_the_single_solve(tmp_path_factory) -> None:
    """A rolling dispatch over p2020 and p2025 (24 h windows, no storage, so each hour is dispatched alike)
    reports what the same solve without rolling reports: operating costs and the fixed cost of the existing
    500 MW, 500 · 1e5 · F_ops(d) in each period, discounted from the start of the run also in the rolls that
    hold p2025 only."""
    periods = {"p2020": "2day", "p2025": "2day"}
    fixed = [["unit", PLANT, "fixed_cost", b64(100.0, "float")]]
    rolling = _results(tmp_path_factory, "roll_costs", _rolling(periods) + fixed, ROLLING)
    whole = _results(tmp_path_factory, "roll_costs_whole",
                     [v for v in _rolling(periods) if v not in ROLLING_VALUES] + fixed, ROLLING)
    _assert_same_results(rolling, whole)
    discounted = rolling["cost_unit_discounted__d"]
    for d in periods:
        assert _row(discounted, d)["fixed cost pre-existing"] * MEGA == pytest.approx(
            500.0 * FIXED * F_OPS[d], rel=1e-7), d


SALVAGE = 200.0


def _retire_values(lifetime: float) -> list:
    """coal_plant (existing 500 MW, fixed_cost 100, salvage_value 200, discount_rate 0.05) retires exactly 100
    MW in p2025 and invests nothing."""
    pins = _period_map({d: (100.0 if d == "p2025" else 0.0) for d in PERIODS})
    values = [["unit", PLANT, "invest_method", b64("retire_period", "str")],
              ["unit", PLANT, "retire_min_period", pins],
              ["unit", PLANT, "retire_max_period", pins],
              ["unit", PLANT, "salvage_value", b64(SALVAGE, "float")],
              ["unit", PLANT, "lifetime", b64(lifetime, "float")]]
    return values


@pytest.mark.solver
@pytest.mark.parametrize("lifetime", [30.0, 10.0], ids=["long", "short"])
def test_retirement_costs(tmp_path_factory, lifetime) -> None:
    """100 MW of the existing 500 MW (reinvest_automatic: the existing capacity does not end, so the retirement
    counts until the end of the run) retire in p2025.  The chain reports what the single solve reports.  The
    retirement's window is p2025 … p2035 for both lifetimes (the lifetime no longer cuts it): the saving of fixed
    cost is the fixed cost the 100 MW no longer pay there, 100 · 1e5 · Σ F_ops(d) (the operations factor, as the
    fixed cost of existing capacity is charged); the salvage value is spread over the R = 15 years of the window,
    A = 200 · 1000 · 0.05 / (1 − 1.05^−15), retirement = −100 · A · Σ F_inv(d).  The fixed cost of the existing
    capacity stays that of the user's 500 MW (the retirement is in its own lump, not charged again)."""
    values = _retire_values(lifetime)
    tag = f"retire_{int(lifetime)}"
    chain = _results(tmp_path_factory, f"{tag}_chain", values, _no_battery_invest(CHAIN))
    single = _results(tmp_path_factory, f"{tag}_single", values, _no_battery_invest(SINGLE))
    _assert_same_results(chain, single)
    window = _later("p2025")
    discounted = chain["cost_unit_discounted__d"]
    retired = _row(discounted, "p2025")
    assert retired["fixed cost reduction of divestments"] * MEGA == pytest.approx(
        -100.0 * FIXED * sum(F_OPS[d] for d in window), rel=1e-7)
    salvage_annuity = SALVAGE * 1000.0 * 0.05 / (1.0 - 1.05 ** -15.0)
    assert retired["retirement"] * MEGA == pytest.approx(
        -100.0 * salvage_annuity * sum(F_INV[d] for d in window), rel=1e-7)
    for d in PERIODS:
        assert _row(discounted, d)["fixed cost pre-existing"] * MEGA == pytest.approx(
            500.0 * FIXED * F_OPS[d], rel=1e-7), d
    # Annualized: in every period of the window, the yearly salvage annuity and the yearly fixed cost of the
    # retired 100 MW, A(200, 0.05, 15) = 19 268.46 CUR/MW/yr: −1.926846 and −10 M/yr in p2025 … p2035; nothing in
    # p2020.
    for annualized in (chain["cost_unit_annualized__d"], single["cost_unit_annualized__d"]):
        for d in PERIODS:
            rows = annualized[(annualized["period"] == d) & (annualized["unit"] == PLANT)]
            mw = 100.0 if d in window else 0.0
            assert rows["retirement"].sum() * MEGA == pytest.approx(-mw * salvage_annuity, rel=1e-7, abs=1e-6), d
            assert rows["fixed cost reduction of divestments"].sum() * MEGA == pytest.approx(
                -mw * FIXED, rel=1e-7, abs=1e-6), d


# ---------------------------------------------------------------------------
# Stochastic: the realization model
# ---------------------------------------------------------------------------
#
# ``stoch_realized_dispatch.json``: the stochastic solve ``stoch_rd`` (one period p2035, a fan of identical
# forecasts high / low revealed at t0001, a realized designation rlz) invests in the peaker (shared first-stage
# investment); the realization solve ``stoch_rd_realized_dispatch`` dispatches p2035 on that capacity.  Here
# with fixed costs (wind: existing 100 MW, peaker: new) and inflation, p2035 representing 5 years.

STOCH_RD_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_realized_dispatch.json"
STOCH_RD_CHAIN = ["init", "base", "stoch"]
STOCH_RD_VALUES = [["model", "flextool", "inflation_rate", b64(0.04, "float")],
                   ["unit", "wind", "fixed_cost", b64(10.0, "float")],
                   ["unit", "peaker", "fixed_cost", b64(20.0, "float")],
                   ["solve", "stoch_rd", "years_represented", _period_map({"p2035": 5.0})],
                   ["solve", "stoch_rd_realized_dispatch", "years_represented", _period_map({"p2035": 5.0})]]


def _stoch_variant(tmp_path_factory, name: str, mutate) -> Path:
    data = json.loads(STOCH_RD_JSON.read_text())
    mutate(data)
    out = tmp_path_factory.mktemp(f"{name}_json") / f"{name}.json"
    out.write_text(json.dumps(data))
    return out


def _deterministic(data: dict) -> None:
    """The fixture without its fan: one deterministic solve that invests and dispatches p2035."""
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if pv[2] not in ("stochastic_branches", "include_stochastics")
                                and not (pv[0] == "model" and pv[2] == "solves")]
    data["parameter_values"].append(["model", "flextool", "solves",
                                     b64({"value_type": "str", "data": ["stoch_rd"]}, "array"), "init"])


@pytest.mark.solver
def test_stochastic_realization_reports_the_deterministic_twin(tmp_path_factory) -> None:
    """The stochastic solve with identical forecasts and its realization solve report what the deterministic twin
    reports: the investment and its fixed cost once (from the stochastic solve, at p2035), the operations and
    the fixed cost of the existing wind from the realization solve, whose fixed investment is not charged
    again."""
    stoch = _results(tmp_path_factory, "stoch_rd_costs", STOCH_RD_VALUES, STOCH_RD_CHAIN, fixture_json=STOCH_RD_JSON)
    twin = _results(tmp_path_factory, "stoch_rd_twin", STOCH_RD_VALUES, STOCH_RD_CHAIN,
                    fixture_json=_stoch_variant(tmp_path_factory, "stoch_rd_det", _deterministic))
    _assert_same_results(stoch, twin)


FAN_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"


FAN_VALUES = [["model", "flextool", "inflation_rate", b64(0.04, "float")],
              ["unit", "wind", "fixed_cost", b64(10.0, "float")],
              ["unit", "peaker", "fixed_cost", b64(20.0, "float")]]
FAN_F_INV = {"p2035": 1.0, "p2040": 1.04 ** -1}
FAN_F_OPS = {"p2035": 1.04 ** -0.5, "p2040": 1.04 ** -1.5}
FAN_ANNUITY = 1.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -1)  # 1050 CUR/MW/yr
FAN_FIXED = 20.0 * 1000.0


def _assert_fan_reports_no_operations(fan: dict[str, pd.DataFrame]) -> None:
    discounted = fan["cost_unit_discounted__d"]
    for d in ("p2035", "p2040"):
        assert _row(discounted, d, "wind")["fixed cost pre-existing"] * MEGA == pytest.approx(
            100.0 * 10.0e3 * FAN_F_OPS[d], rel=1e-7), d
    operations = [c for c in discounted.columns
                  if c not in ("solve", "period", "unit", "investment", "retirement", "fixed cost pre-existing",
                               "fixed cost invested")]
    assert operations and (discounted[operations].fillna(0.0) == 0.0).all().all()
    total = fan["costs_discounted"].set_index("category").iloc[:, 0]
    assert total.filter(like="penalty").abs().sum() == 0.0
    assert total[["commodity_cost", "other operational", "starts", "co2"]].abs().sum() == 0.0


@pytest.mark.solver
def test_fan_without_realization_reports_investment_and_existing_fixed_cost(tmp_path_factory) -> None:
    """``stoch_two_period_invest.json`` with a shared investment (no recourse): a fan of forecasts high / low
    revealed at p2035 (one year each, p2035 and p2040) and no realization solve.  No operating or penalty cost
    is reported (there are no realized operations); the investment and the fixed cost of new peaker capacity are
    reported in the period it starts operating, over the run's periods it operates in; the fixed cost of the
    existing 100 MW wind is reported in both periods.  With r = 0.04, offsets ops 0.5 / inv 0:
    F_inv = {p2035: 1, p2040: 1.04^-1}, F_ops = {p2035: 1.04^-0.5, p2040: 1.04^-1.5}; peaker annuity
    1 · 1000 · 0.05 / (1 − 1.05^-1) = 1050 CUR/MW/yr, fixed cost 20 · 1000, wind fixed cost 10 · 1000.

    The builds by hand: the deficit (demand − wind) is 60 / 80 MW (high / low) in p2035 and 90 / 120 MW in
    p2040, penalty 1e4.  A decision of p2040 operates in the p2040 branches, so p2035 builds 80 MW (its larger
    deficit) and p2040 adds the 40 MW that are cheaper there (no p2035 annuity or fixed cost):
    p2035 investment 80 · 1050 · (F_inv(p2035) + F_inv(p2040)) (reinvest_automatic: it keeps operating in p2040),
    p2040 investment 40 · 1050 · F_inv(p2040)."""
    fan = _results(tmp_path_factory, "stoch_fan_costs", FAN_VALUES, ["init", "base", "stoch"],
                   fixture_json=FAN_JSON)
    discounted = fan["cost_unit_discounted__d"]
    first = _row(discounted, "p2035", "peaker")
    assert first["investment"] * MEGA == pytest.approx(80.0 * FAN_ANNUITY * sum(FAN_F_INV.values()), rel=1e-7)
    assert first["fixed cost invested"] * MEGA == pytest.approx(80.0 * FAN_FIXED * sum(FAN_F_OPS.values()),
                                                                rel=1e-7)
    second = _row(discounted, "p2040", "peaker")
    assert second["investment"] * MEGA == pytest.approx(40.0 * FAN_ANNUITY * FAN_F_INV["p2040"], rel=1e-7)
    assert second["fixed cost invested"] * MEGA == pytest.approx(40.0 * FAN_FIXED * FAN_F_OPS["p2040"], rel=1e-7)
    _assert_fan_reports_no_operations(fan)


# ---------------------------------------------------------------------------
# Behaviour-change notice
# ---------------------------------------------------------------------------

NOTICE_ID = "chained_solve_costs_from_run_start"
NOTICE_MARKER = f"flextool_behaviour_change:{NOTICE_ID}"


def _printed(tmp_path_factory, capsys, alt: str, values: list, chain: list[str], marker: bool,
             entities: list | None = None) -> list[str]:
    """The notice lines a run prints."""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities)
    if marker:
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=NOTICE_MARKER, value="77")
            db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]


@pytest.mark.solver
def test_notice_lists_every_solve_of_the_run_once(tmp_path_factory, capsys) -> None:
    lines = _printed(tmp_path_factory, capsys, "cc_notice", [], _no_battery_invest(CHAIN), marker=True)
    assert len(lines) == 1
    assert "Affected solves: y2020_5week, y2025_5week, y2030_5week, y2035_5week." in lines[0]
    assert NOTICE_MARKER in lines[0]


@pytest.mark.solver
def test_notice_names_nested_solves(tmp_path_factory, capsys) -> None:
    values = (_solves(["cc_invest", "cc_dispatch"])
              + _solve("cc_invest", PERIODS, invest_periods=PERIODS, realized_invest_periods=PERIODS)
              + _solve("cc_dispatch", PERIODS, realized_periods=PERIODS))
    entities = [["solve", "cc_invest", None], ["solve", "cc_dispatch", None]]
    lines = _printed(tmp_path_factory, capsys, "cc_notice_nested", values, _no_battery_invest(CHAIN), marker=True,
                     entities=entities)
    assert len(lines) == 1 and "Affected solves: cc_invest, cc_dispatch." in lines[0]


@pytest.mark.solver
@pytest.mark.parametrize("chain, marker", [(CHAIN, False), (SINGLE, True)], ids=["no_marker", "single_solve"])
def test_no_notice(tmp_path_factory, capsys, chain, marker) -> None:
    alt = f"cc_no_notice_{marker}"
    assert _printed(tmp_path_factory, capsys, alt, [], _no_battery_invest(chain), marker=marker) == []


def test_notice_is_registered_and_written_by_the_upgrade() -> None:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES
    from flextool.update_flextool import db_migration

    assert NOTICE_ID in BEHAVIOUR_CHANGES
    assert NOTICE_ID in db_migration._V77_UNCONDITIONAL_MARKERS
