"""Reported costs of chained solves: discount factors and investment lumps from the start of the run.

Fixture ``tests.json`` (built from JSON).  ``model.social_discount_rate`` r = 0.04 (alternative
``invest_solveSequence_5weeks``), ``discount_offset_operations`` 0.5 and ``discount_offset_investment`` 0 (the
defaults), four periods p2020 … p2035 of 5 years each, so the run years are Y = 0, 5, 10, 15 and the horizon ends at
20.  Hand derivation of the factors (``_run_costs`` "Run-level fields of the reported costs")::

    F_ops(d) = Σ_{y=0..4} 1.04^-(Y_d + y + 0.5)        F_inv(d) = Σ_{y=0..4} 1.04^-(Y_d + y)

coal_plant: invest_cost 400 CUR/kW, discount_rate 0.05, lifetime 30 (longer than the run), no lifetime_method
(reinvest_automatic: the operating window of a decision is every run period from its start of operation on),
fixed_cost 100 CUR/kW/yr.  Annuity A = 400 · 1000 · 0.05 / (1 − 1.05^−30) / 1.05 CUR/MW/yr (paid at the start of
each year, offset 0: the year-end annuity divided by 1 + discount_rate).  So, for a decision in d with
start of operation c(d)::

    lump_invest[d] = A · Σ_{d' ≥ c(d)} F_inv(d')

The chain ``invest_solveSequence_5weeks`` (y2020 … y2035: each solve holds its own period and the next one) gets
these values in every solve, and each solve's own factors (the objective's) are those of its periods: every solve
discounts from the start of the run (item 242).
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
from flextool.engine.preprocess.params._cost_lines import TABLE_LINES

PERIODS = ["p2020", "p2025", "p2030", "p2035"]
YEARS = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}


def _f(d: str, offset: float) -> float:
    return sum(1.04 ** -(YEARS[d] + y + offset) for y in range(5))


F_OPS = {d: _f(d, 0.5) for d in PERIODS}
F_INV = {d: _f(d, 0.0) for d in PERIODS}
#: The annuity, paid at the start of each year (``discount_offset_investment`` 0): the year-end annuity / 1.05.
ANNUITY = 400.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -30) / 1.05
FIXED = 100.0 * 1000.0


def _later(d: str) -> list[str]:
    return PERIODS[PERIODS.index(d):]


def _d(param) -> dict[str, float]:
    frame = utf8(param.frame)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


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
    periods; the solve's own factors (the objective's) are the run factors of its periods: every solve discounts
    from the start of the run (before, they restarted at the solve's first period)."""
    steps = _steps(tmp_path_factory, "chain_run_fields", [], CHAIN)
    assert list(steps) == ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]
    for k, (name, step) in enumerate(steps.items()):
        fd = step.flex_data
        own = PERIODS[k:k + 2]
        _assert_run_factors(fd)
        # The objective's factors: discounted from the start of the run, as the reported costs.
        assert _d(fd.p_inflation_op) == pytest.approx({d: F_OPS[d] for d in own}, rel=1e-12), name
        lines = fd.edl_invest_lines_run.filter(pl.col("e") == PLANT)
        assert {(a, li, b): v for a, li, b, v in lines.select("d", "line", "d_report", "value").iter_rows()} == \
            pytest.approx({(d, "investment", d): ANNUITY * sum(F_INV[x] for x in _later(d)) for d in own},
                          rel=1e-12), name
        assert _ed(fd.ed_entity_annual_discounted_run) == pytest.approx(
            {d: ANNUITY * sum(F_INV[x] for x in _later(d)) for d in own}, rel=1e-12), name


def test_lead_time_window_starts_at_start_of_operation(tmp_path_factory) -> None:
    """coal_plant with a 5-year construction lead time: the y2020 solve offers only the p2020 decision, which
    starts to operate in p2025.  It operates in p2025 … p2035 and its lump equals that of a p2025 decision
    without a lead time (same operating years, same invest cost)."""
    steps = _steps(tmp_path_factory, "chain_run_lead",
                   [["unit", PLANT, "construction_lead_time", b64(5.0, "float")]], CHAIN)
    fd = steps["y2020_5week"].flex_data
    _assert_run_factors(fd)
    assert _ed(fd.ed_entity_annual_discounted_run)["p2020"] == pytest.approx(
        ANNUITY * sum(F_INV[x] for x in _later("p2025")), rel=1e-12)


def _assert_run_is_solve(fd) -> None:
    assert fd.p_inflation_op_run is fd.p_inflation_op
    assert fd.ed_entity_annual_discounted_run is fd.ed_entity_annual_discounted


def test_single_solve_run_fields_are_its_own(tmp_path_factory) -> None:
    """One solve over the whole run: the run fields ARE the solve's fields (the reported costs do not change),
    and the investment factor and lumps are those of the solve."""
    steps = _steps(tmp_path_factory, "single_run_fields", [], SINGLE)
    (fd,) = [s.flex_data for s in steps.values()]
    _assert_run_is_solve(fd)
    _assert_run_factors(fd)
    lump = _ed(fd.ed_entity_annual_discounted_run)
    assert lump == pytest.approx({d: ANNUITY * sum(F_INV[x] for x in _later(d)) for d in PERIODS}, rel=1e-12)


def _rolling(periods: dict[str, str]) -> list:
    def arr(items: list[str]) -> list:
        return b64({"type": "array", "value_type": "str", "data": items, "index_name": "period"}, "array")

    def pmap(data: dict) -> list:
        return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in data.items()]}, "map")
    return ROLLING_VALUES + [
        ["model", "flexTool", "social_discount_rate", b64(0.04, "float")],
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
    from flextool.engine.preprocess.params._run_costs import _run_window

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
    out = _write_csv(tmp_path_factory, alt, values, chain, entities=entities, fixture_json=fixture_json)
    return {stem: pd.read_csv(out / f"{stem}.csv") for stem in COST_STEMS if (out / f"{stem}.csv").exists()}


def _write_csv(tmp_path_factory, alt: str, values: list, chain: list[str], *, entities: list | None = None,
               fixture_json=None) -> Path:
    """Run the scenario and write its CSV results; return the CSV folder."""
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
    return work / "output_csv" / scen


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
    """The annualized investment and fixed cost invested of coal_plant (reinvest_automatic, lifetime 30: from its
    start of operation to the end of the run).  100 MW start to operate in p2025, 50 MW more in p2030.  The
    investment is an event at the start of operation, the decision's capital over its years in service inside the
    horizon, and its annualized value is that event divided by the period's investment factor (spread evenly over
    the period's years)::

        investment[d]          = MW_new[d] · A · Σ_{d' ≥ d} F_inv(d') / F_inv(d),
                                 A = 400 · 1000 · 0.05 / (1 − 1.05^−30) / 1.05 = 24 781.50 CUR/MW/yr
        fixed cost invested[d] = MW_operating[d] · 100 · 1000
        MW_new = {p2025: 100, p2030: 50},  MW_operating = {p2020: 0, p2025: 100, p2030: 150, p2035: 150}

    The system columns are the same (only coal_plant invests).  Before: the yearly annuity of every vintage
    operating in the period."""
    new = {"p2025": 100.0, "p2030": 50.0}
    operating = {"p2020": 0.0, "p2025": 100.0, "p2030": 150.0, "p2035": 150.0}
    unit = results["cost_unit_annualized__d"]
    system = results["annualized_costs__d"].set_index("period")
    for d, mw in operating.items():
        event = new.get(d, 0.0) * ANNUITY * sum(F_INV[x] for x in _later(d)) / F_INV[d]
        rows = unit[(unit["period"] == d) & (unit["unit"] == PLANT)]
        invest = rows["investment"].sum() if len(rows) else 0.0
        fixed = rows["fixed cost invested"].sum() if len(rows) else 0.0
        assert invest * MEGA == pytest.approx(event, rel=1e-7, abs=1e-6), d
        assert fixed * MEGA == pytest.approx(mw * FIXED, rel=1e-7, abs=1e-6), d
        assert system.loc[d, "investment"] * MEGA == pytest.approx(event, rel=1e-7, abs=1e-6), d
        assert system.loc[d, "fixed cost invested"] * MEGA == pytest.approx(mw * FIXED, rel=1e-7, abs=1e-6), d


@pytest.mark.solver
def test_chain_reports_the_costs_of_the_single_solve(tmp_path_factory) -> None:
    """No lead time, with a CO2 cap per period: every cost result of the chain equals the single solve's, and
    the CO2 price of a period (its dual ÷ the realizing solve's own factor) is unchanged.

    Hand anchors (the module docstring's factors): the fixed cost of new capacity in each period it is paid, the
    100 MW decided in p2025 in p2025, 100 · 1e5 · F_ops(p2025) = 37.3154 M, and with the 50 MW of p2030 the 150 MW
    in service in p2030, 150 · 1e5 · F_ops(p2030); the fixed cost of the existing 500 MW in p2030, 500 · 1e5 ·
    F_ops(p2030) = 153.35 M, which is 50 M a year (before: the chain charged 100 · 1e5 · (F(p2025) + F(p2030))
    from the solve's own start = 82.72 M, and 600 MW in p2030)."""
    values = _probe_values(0.0)
    single = _results(tmp_path_factory, "costs_single", values, _no_battery_invest(SINGLE) + ["co2_limit"],
                      entities=ENTITIES)
    chain = _results(tmp_path_factory, "costs_chain", values, _no_battery_invest(CHAIN) + ["co2_limit"],
                     entities=ENTITIES)
    _assert_same_results(chain, single)
    discounted = chain["cost_unit_discounted__d"]
    assert _row(discounted, "p2025")["fixed cost invested"] * MEGA == pytest.approx(
        100.0 * FIXED * F_OPS["p2025"], rel=1e-7)
    assert 100.0 * FIXED * F_OPS["p2025"] == pytest.approx(37.3154e6, rel=1e-5)
    assert _row(discounted, "p2030")["fixed cost invested"] * MEGA == pytest.approx(
        150.0 * FIXED * F_OPS["p2030"], rel=1e-7)
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
    # The annualized investment is each start of operation's event: p2025 (the p2020 decision) and p2030.
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
    """A chain y2020, y2025, y2030 whose last solve holds p2035 as an unrealized look-ahead: the fixed costs of
    new capacity are reported in each period up to p2035 (the 150 MW in service in p2030 and in p2035: 150 · 1e5 ·
    F_ops(d)), and so is the fixed cost of the existing capacity (p2035: 500 · 1e5 · F_ops(p2035)).  The results
    equal those of one solve over p2020 … p2035 that realizes p2020 … p2030."""
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
    for d in ("p2030", "p2035"):
        assert _row(discounted, d)["fixed cost invested"] * MEGA == pytest.approx(
            150.0 * FIXED * F_OPS[d], rel=1e-7), d


def _summary_totals(csv_dir: Path) -> dict[str, float]:
    """``{"full": …, "realized": …}``: the calculated totals (M CUR) of ``summary_solve.csv``."""
    out: dict[str, float] = {}
    for line in (csv_dir / "summary_solve.csv").read_text(encoding="utf-8").splitlines():
        if line.startswith('"Total cost (calculated) full horizon'):
            out["full"] = float(line.split(",")[1])
        elif line.startswith('"Total cost (calculated) realized periods'):
            out["realized"] = float(line.split(",")[1])
    return out


@pytest.mark.solver
def test_summary_realized_total_leaves_out_the_unrealized_look_ahead(tmp_path_factory) -> None:
    """The chain of the previous test: the summary's realized-periods total is the full-horizon total without the
    rows of p2035, which no solve realizes.  The only costs there are the fixed costs of coal_plant, the only
    entity with a fixed cost: the existing 500 MW and the 150 MW built in p2025 and p2030, in service in p2035 (no
    capacity starts to operate in p2035), so full − realized = (500 + 150) · 1e5 · F_ops(p2035) / 1e6 M CUR.
    Before, the summary printed the full-horizon total on both lines."""
    out = _write_csv(tmp_path_factory, "summary_lookahead_chain",
                     _probe_values(0.0) + _solves(["y2020_5week", "y2025_5week", "y2030_5week"]),
                     _no_battery_invest(CHAIN), entities=ENTITIES)
    totals = _summary_totals(out)
    assert totals["full"] - totals["realized"] == pytest.approx(650.0 * FIXED * F_OPS["p2035"] / MEGA, rel=1e-7)
    per_period = pd.read_csv(out / "costs_discounted__d.csv")
    realized = per_period[per_period["period"].isin(PERIODS[:3])].drop(columns=["solve", "period"])
    assert totals["realized"] == pytest.approx(realized.to_numpy().sum(), rel=1e-7)


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
    """coal_plant (existing 500 MW, fixed_cost 100, net_salvage_value 200, discount_rate 0.05) retires exactly 100
    MW in p2025 and invests nothing."""
    pins = _period_map({d: (100.0 if d == "p2025" else 0.0) for d in PERIODS})
    values = [["unit", PLANT, "invest_method", b64("retire_period", "str")],
              ["unit", PLANT, "retire_min_period", pins],
              ["unit", PLANT, "retire_max_period", pins],
              ["unit", PLANT, "net_salvage_value", b64(SALVAGE, "float")],
              ["unit", PLANT, "lifetime_technical", b64(lifetime, "float")]]
    return values


@pytest.mark.solver
@pytest.mark.parametrize("lifetime", [30.0, 10.0], ids=["long", "short"])
def test_retirement_costs(tmp_path_factory, lifetime) -> None:
    """100 MW of the existing 500 MW (reinvest_automatic: the capacity is replaced at its end of life, so the
    retirement counts until the end of the run) retire in p2025.  The retired 100 MW count in p2025 … p2035 for both
    lifetimes: the saving of fixed cost is the fixed cost the 100 MW no longer pay, reported in each of those
    periods, −100 · 1e5 · F_ops(d) (the operations factor, as the fixed cost of existing capacity is charged; p2025:
    −100 · 1e5 · 3.73154 = −37.3154 M), the same in the chain and the single solve.  The fixed cost of the existing
    capacity stays that of the user's 500 MW.

    The salvage settlement is what each solve charges (D(5) = 1.04^−5, F_inv the investment factors, discount_rate
    5 %, offset 0, af = 0.05 / (1 − 1.05^−L) / 1.05): the retired MW receives S = 200 000 at the start of p2025,
    gives back the existing plant's level salvage credits a_x = S·1.05^−L·af over its periods the solve sees, and,
    with lifetime 10, also the level price of its replacement (online in p2030 at invest_cost 400, its own salvage
    credit netted in: A − a_x, A = 400 000·af) over the replacement's periods the solve sees.  The single solve sees
    p2020 … p2035; the chain's y2025 solve its history p2020 and its own p2025, p2030 (its horizon):

        lifetime 30 (one plant past the horizon): settlement = −100·(S·D(5) − a_x·Σ_seen F_inv)
        lifetime 10: settlement = −100·(S·D(5) − a_x·(F_inv(p2020) + F_inv(p2025)) + (A − a_x)·Σ_{seen, ≥ p2030}
                     F_inv)

    The settlement is reported by event (each cost line in the period it happens): *salvage economic* −100·S·D(5)
    at p2025 (received); the existing plant's credits given back, +100·a_x·Σ_seen,plant F_inv, as *salvage
    replaced* where the plant's end of life is reported (lifetime 30: past the horizon, so the last period p2035;
    lifetime 10: p2030); with lifetime 10 the replacement's capital given back, *replacement* −100·A·Σ_{seen, ≥
    p2030} F_inv at p2030 (where the replacement would come online; *retirement* holds only the existing plant's
    settlement, which has no contract: 0), and its salvage credits, *salvage replaced* +100·a_x·Σ_{seen, ≥ p2030}
    F_inv at its end of life (year 20, the horizon end: the last period p2035).  The lines add up to the settlement; the cost tables show
    the salvage causes as one *salvage* line, their sum per period.  Annualized: each line divided by the
    investment factor of the period it is reported in."""
    values = _retire_values(lifetime)
    tag = f"retire_{int(lifetime)}"
    chain = _results(tmp_path_factory, f"{tag}_chain", values, _no_battery_invest(CHAIN))
    single = _results(tmp_path_factory, f"{tag}_single", values, _no_battery_invest(SINGLE))
    fixed_columns = ["fixed cost pre-existing", "fixed cost reduction of divestments"]
    stems = ("cost_unit_discounted__d", "cost_unit_annualized__d")
    _assert_same_results({k: chain[k][["period", "unit", *fixed_columns]] for k in stems},
                         {k: single[k][["period", "unit", *fixed_columns]] for k in stems})
    window = _later("p2025")
    af = 0.05 / (1.0 - 1.05 ** -lifetime) / 1.05
    a_x = SALVAGE * 1000.0 * 1.05 ** -lifetime * af
    capital = 400.0 * 1000.0 * af
    seen = {"single": PERIODS, "chain": ["p2020", "p2025", "p2030"]}
    for kind, results in (("chain", chain), ("single", single)):
        periods = seen[kind]
        want = {("p2025", "salvage economic"): -100.0 * SALVAGE * 1000.0 * 1.04 ** -5}
        if lifetime > 20.0:
            want[("p2035", "salvage replaced")] = 100.0 * a_x * sum(F_INV[d] for d in periods)
        else:
            later = sum(F_INV[d] for d in periods if d >= "p2030")
            want[("p2030", "salvage replaced")] = 100.0 * a_x * (F_INV["p2020"] + F_INV["p2025"])
            want[("p2030", "replacement")] = -100.0 * capital * later
            want[("p2035", "salvage replaced")] = 100.0 * a_x * later
        discounted = results["cost_unit_discounted__d"]
        assert -100.0 * FIXED * F_OPS["p2025"] == pytest.approx(-37.3154e6, rel=1e-5)
        annualized = results["cost_unit_annualized__d"]
        for d in PERIODS:
            row, yearly = _row(discounted, d), _row(annualized, d)
            for line in ("retirement", "replacement", "salvage"):
                # The cost tables show the salvage causes as one salvage line.
                amount = sum(v for (p_, cause), v in want.items() if p_ == d and cause.startswith(line))
                assert row[line] * MEGA == pytest.approx(amount, rel=1e-7, abs=1e-3), (kind, d, line)
                assert yearly[line] * MEGA == pytest.approx(amount / F_INV[d], rel=1e-7, abs=1e-3), (kind, d, line)
            assert row["fixed cost pre-existing"] * MEGA == pytest.approx(500.0 * FIXED * F_OPS[d], rel=1e-7), d
            mw = 100.0 if d in window else 0.0
            assert row["fixed cost reduction of divestments"] * MEGA == pytest.approx(
                -mw * FIXED * F_OPS[d], rel=1e-7, abs=1e-6), d
            assert yearly["fixed cost reduction of divestments"] * MEGA == pytest.approx(
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
STOCH_RD_VALUES = [["model", "flextool", "social_discount_rate", b64(0.04, "float")],
                   ["unit", "wind", "fixed_cost", b64(10.0, "float")],
                   ["unit", "peaker", "fixed_cost", b64(20.0, "float")],
                   ["solve", "stoch_rd", "years_represented", _period_map({"p2035": 5.0})],
                   ["solve", "stoch_rd_realized_dispatch", "years_represented", _period_map({"p2035": 5.0})]]


def _stoch_variant(tmp_path_factory, name: str, mutate) -> Path:
    data = json.loads(STOCH_RD_JSON.read_text(encoding="utf-8"))
    mutate(data)
    out = tmp_path_factory.mktemp(f"{name}_json") / f"{name}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
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


def _realized_fan_chain_db(tmp_path_factory) -> tuple[str, str]:
    """The fan → realize → next solve chain of :func:`test_chain_of_realized_fans_reports_from_the_run_start`,
    built from the schema JSON."""
    import spinedb_api as api

    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    def m(d: dict, name: str):
        return api.Map(list(d), list(d.values()), index_name=name)

    steps = [f"t{i:04d}" for i in range(1, 9)]
    # Wind CF per branch: (p2030 steps, p2040 steps); the t0001 forecast covers both periods, the t0005 one p2040.
    cf = {"high": (0.9, 0.9), "low": (0.3, 0.25), "rlz": (0.5, 0.6)}
    profile = m({b: m({"t0001": m(dict(zip(steps, [c30] * 4 + [c40] * 4)), "time"),
                       "t0005": m(dict(zip(steps[4:], [c40] * 4)), "time")}, "analysis_time")
                 for b, (c30, c40) in cf.items()}, "branch")

    def fan(period: str, at: str):
        return m({period: m({b: m({at: m({r: w}, "realized")}, "analysis_time")
                             for b, (r, w) in {"high": ("no", 0.5), "low": ("no", 0.5),
                                               "rlz": ("yes", 0.0)}.items()}, "branch")}, "period")

    def solve(name: str, period: str, timeset: str, at: str) -> list:
        own = api.Array([period])
        return [("solve", name, "solve_mode", "single_solve"),
                ("solve", name, "period_timeset", m({period: timeset}, "period")),
                ("solve", name, "years_represented", m({period: 2.0}, "period")),
                ("solve", name, "invest_periods", own), ("solve", name, "realized_invest_periods", own),
                ("solve", name, "realized_periods", own), ("solve", name, "stochastic_branches", fan(period, at))]

    entities = [("model", "flextool"), ("timeline", "tl"), ("timeset", "ts30"), ("timeset", "ts40"),
                ("node", "elec"), ("unit", "wind"), ("unit__outputNode", ("wind", "elec")), ("profile", "wprof"),
                ("unit__node__profile", ("wind", "elec", "wprof")), ("group", "g"), ("group__unit", ("g", "wind")),
                ("solve", "s30"), ("solve", "s40")]
    values = [
        ("model", "flextool", "solves", api.Array(["s30", "s40"])),
        ("model", "flextool", "social_discount_rate", 0.1),
        ("timeline", "tl", "timestep_duration", m({t: 1.0 for t in steps}, "time")),
        ("timeset", "ts30", "timeline", "tl"), ("timeset", "ts30", "timeset_duration", m({"t0001": 4.0}, "time")),
        ("timeset", "ts40", "timeline", "tl"), ("timeset", "ts40", "timeset_duration", m({"t0005": 4.0}, "time")),
        ("node", "elec", "node_type", "balance"), ("node", "elec", "inflow", -100.0),
        ("node", "elec", "penalty_up", 10000.0), ("node", "elec", "penalty_down", 10000.0),
        ("unit", "wind", "existing", 0.0), ("unit", "wind", "efficiency", 1.0),
        ("unit", "wind", "virtual_unitsize", 1.0), ("unit", "wind", "invest_method", "invest_total"),
        ("unit", "wind", "invest_cost", 1000.0), ("unit", "wind", "lifetime_technical", 30.0),
        ("unit", "wind", "discount_rate", 0.05), ("unit", "wind", "invest_max_total", 1000.0),
        ("unit", "wind", "fixed_cost", 10.0),
        ("unit__outputNode", ("wind", "elec"), "other_operational_cost", 1.0),
        ("unit__node__profile", ("wind", "elec", "wprof"), "profile_method", "upper_limit"),
        ("profile", "wprof", "profile", profile),
        ("group", "g", "include_stochastics", "yes"),
        *solve("s30", "p2030", "ts30", "t0001"), *solve("s40", "p2040", "ts40", "t0005")]
    db = tmp_path_factory.mktemp("realized_fan_chain") / "chain.sqlite"
    initialize_database(str(FLEXTOOL_ROOT / "flextool" / "schemas" / "spinedb_schema.json"), str(db))
    url = f"sqlite:///{db.resolve().as_posix()}"
    with open_database_mapping(url) as mp:
        _, errors = api.import_data(
            mp, alternatives=[("a", "")], scenarios=[("sc", False, "")], scenario_alternatives=[("sc", "a", None)],
            entities=entities, entity_alternatives=[("node", ("elec",), "a", True), ("unit", ("wind",), "a", True)],
            parameter_values=[(c, e, p, v, "a") for c, e, p, v in values])
        assert not errors, errors
        mp.commit_session("realized fan chain")
    return url, "sc"


@pytest.mark.solver
def test_chain_of_realized_fans_reports_from_the_run_start(tmp_path_factory) -> None:
    """Two stochastic solves in a chain, each with a fan of forecasts high / low (weights 0.5) and a realized
    designation rlz at its own period's start: s30 (p2030) → s30__realized_dispatch → s40 (p2040) →
    s40__realized_dispatch.  Before: the run stopped ("periods ['p2030_rlz'] have no year from the start of the
    run"); the realized branch copy p2030_rlz is no period of the run, its operations are reported by the
    realization solve at the year of its real period p2030.

    By hand: p2030 and p2040 each represent 2 years, so the run years are Y = 0 (p2030) and 2 (p2040); inflation
    r = 0.1, offsets ops 0.5 / inv 0 (the defaults)::

        F_inv(d) = Σ_{y=0,1} 1.1^-(Y_d + y)         F_inv = {p2030: 1.909091, p2040: 1.577761}
        F_ops(d) = Σ_{y=0,1} 1.1^-(Y_d + y + 0.5)   F_ops = {p2030: 1.820247, p2040: 1.504336}

    Both investments and operations of s40 are discounted from the start of the run (Y = 2), not from the
    start of s40, in the reported costs and in s40's own objective (item 242): its p_inflation_op of p2040 is
    1.1^-2.5 + 1.1^-3.5 = 1.504336.  Wind: annuity A = 1000 · 1000 · 0.05 / (1 − 1.05^−30) / 1.05 (paid at year
    starts) = 61 953.75 CUR/MW/yr, fixed cost
    10 000 CUR/MW/yr, reinvest_automatic (operates to the end of the run).  The builds: the low forecast must
    cover the demand of 100 MW (penalty 1e4): 100 / 0.3 = 333.33 MW in p2030, 100 / 0.25 = 400 MW in p2040, so
    66.67 MW more.  Discounted: p2030 333.33 · A · (F_inv(p2030) + F_inv(p2040)), p2040 66.67 · A · F_inv(p2040);
    fixed cost invested the capacity in service in d (333.33 MW in p2030, 400 MW in p2040) · 10 000 · F_ops(d);
    the annualized investment is that event ÷ F_inv(d).
    Operations: the realized wind (CF 0.5 / 0.6, enough for the demand) produces 100 MWh/h at 1 CUR/MWh: 0.876 M/yr
    annualized, · F_ops(d) discounted."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = _realized_fan_chain_db(tmp_path_factory)
    work = tmp_path_factory.mktemp("realized_fan_chain_w")
    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
    assert list(steps) == ["s30", "s30__realized_dispatch", "s40", "s40__realized_dispatch"]
    s40_ops = _d(steps["s40"].flex_data.p_inflation_op)
    assert s40_ops["p2040"] == pytest.approx(1.1 ** -2.5 + 1.1 ** -3.5, rel=1e-12)
    assert s40_ops["p2040"] == pytest.approx(1.504336, rel=1e-6)
    last = next(reversed(list(steps.values())))
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
                  flex_data_provider=getattr(last, "flex_data_provider", None),
                  raw_output_dir=str(work / "output_raw"))
    out = work / "output_csv" / scen
    f_inv = {"p2030": 1.0 + 1.1 ** -1, "p2040": 1.1 ** -2 + 1.1 ** -3}
    f_ops = {"p2030": 1.1 ** -0.5 + 1.1 ** -1.5, "p2040": 1.1 ** -2.5 + 1.1 ** -3.5}
    annuity = 1000.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -30) / 1.05
    built = {"p2030": 100.0 / 0.3, "p2040": 100.0 / 0.25 - 100.0 / 0.3}
    window = {"p2030": ["p2030", "p2040"], "p2040": ["p2040"]}
    operations = 100.0 * 1.0 * 8760.0
    discounted = pd.read_csv(out / "cost_unit_discounted__d.csv")
    annualized = pd.read_csv(out / "cost_unit_annualized__d.csv")
    for d in ("p2030", "p2040"):
        row = _row(discounted, d, "wind")
        event = built[d] * annuity * sum(f_inv[p] for p in window[d])
        operating = sum(built[p] for p in built if p <= d)
        assert row["investment"] * MEGA == pytest.approx(event, rel=1e-6), d
        assert row["fixed cost invested"] * MEGA == pytest.approx(operating * 10_000.0 * f_ops[d], rel=1e-6), d
        assert row["other operational"] * MEGA == pytest.approx(operations * f_ops[d], rel=1e-6), d
        row = _row(annualized, d, "wind")
        assert row["investment"] * MEGA == pytest.approx(event / f_inv[d], rel=1e-6), d
        assert row["fixed cost invested"] * MEGA == pytest.approx(operating * 10_000.0, rel=1e-6), d
        assert row["other operational"] * MEGA == pytest.approx(operations, rel=1e-6), d


FAN_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"


FAN_VALUES = [["model", "flextool", "social_discount_rate", b64(0.04, "float")],
              ["unit", "wind", "fixed_cost", b64(10.0, "float")],
              ["unit", "peaker", "fixed_cost", b64(20.0, "float")]]
FAN_F_INV = {"p2035": 1.0, "p2040": 1.04 ** -1}
FAN_F_OPS = {"p2035": 1.04 ** -0.5, "p2040": 1.04 ** -1.5}
#: One payment at the start of the one year: the year-end annuity 1050 / 1.05 = 1000 CUR/MW/yr (= the cost).
FAN_ANNUITY = 1.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -1) / 1.05
FAN_FIXED = 20.0 * 1000.0


def _assert_fan_reports_no_operations(fan: dict[str, pd.DataFrame]) -> None:
    discounted = fan["cost_unit_discounted__d"]
    for d in ("p2035", "p2040"):
        assert _row(discounted, d, "wind")["fixed cost pre-existing"] * MEGA == pytest.approx(
            100.0 * 10.0e3 * FAN_F_OPS[d], rel=1e-7), d
    operations = [c for c in discounted.columns
                  if c not in ("solve", "period", "unit", *TABLE_LINES, "fixed cost pre-existing",
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
    1 · 1000 · 0.05 / (1 − 1.05^-1) / 1.05 = 1000 CUR/MW/yr (paid at the start of its one year), fixed cost 20 · 1000, wind fixed cost 10 · 1000.

    The builds by hand: the deficit (demand − wind) is 60 / 80 MW (high / low) in p2035 and 90 / 120 MW in
    p2040, penalty 1e4.  A decision of p2040 operates in the p2040 branches, so p2035 builds 80 MW (its larger
    deficit) and p2040 adds the 40 MW that are cheaper there (no p2035 annuity or fixed cost):
    p2035 investment 80 · 1000 · F_inv(p2035); the peaker's lifetime_technical is 1 year, so the p2035 plant is
    replaced in p2040 (reinvest_automatic, at the same invest_cost) and that replacement is reported when it comes
    online: p2040 replacement 80 · 1000 · F_inv(p2040) and investment 40 · 1000 · F_inv(p2040).  The fixed cost of
    new capacity is that of the capacity in service in each period: p2035 80 · 20 000 · F_ops(p2035), p2040 (the
    replacement and the new 40 MW) 120 · 20 000 · F_ops(p2040)."""
    fan = _results(tmp_path_factory, "stoch_fan_costs", FAN_VALUES, ["init", "base", "stoch"],
                   fixture_json=FAN_JSON)
    discounted = fan["cost_unit_discounted__d"]
    first = _row(discounted, "p2035", "peaker")
    assert first["investment"] * MEGA == pytest.approx(80.0 * FAN_ANNUITY * FAN_F_INV["p2035"], rel=1e-7)
    assert first["fixed cost invested"] * MEGA == pytest.approx(80.0 * FAN_FIXED * FAN_F_OPS["p2035"], rel=1e-7)
    second = _row(discounted, "p2040", "peaker")
    assert second["investment"] * MEGA == pytest.approx(40.0 * FAN_ANNUITY * FAN_F_INV["p2040"], rel=1e-7)
    assert second["replacement"] * MEGA == pytest.approx(80.0 * FAN_ANNUITY * FAN_F_INV["p2040"], rel=1e-7)
    assert second["fixed cost invested"] * MEGA == pytest.approx(120.0 * FAN_FIXED * FAN_F_OPS["p2040"], rel=1e-7)
    _assert_fan_reports_no_operations(fan)


FAN_LEAD_TIME = [["unit", "peaker", "construction_lead_time", b64(1.0, "float")]]


def _assert_fan_investment(fan: dict[str, pd.DataFrame], built: dict[str, float], operating: dict[str, float]) -> None:
    """The peaker's reported investment costs: ``built[d]`` MW decided in d (branch-weighted), operating in the
    run periods after their start ``operating[d']`` MW.  Discounted: lifetime_technical 1 year, so every MW
    operating in d is a plant of d: a new one (investment built[d] · 1000 · F_inv(d)) or the replacement of an
    earlier build ((operating[d] − built[d]) · 1000 · F_inv(d)); the fixed cost invested in d = operating[d] ·
    20 000 · F_ops(d); annualized: each event divided by F_inv(d) and operating[d] · 20 000 per year."""
    discounted, annualized = fan["cost_unit_discounted__d"], fan["cost_unit_annualized__d"]
    for d in ("p2035", "p2040"):
        new = built.get(d, 0.0)
        want_fix = operating[d] * FAN_FIXED * FAN_F_OPS[d]
        row = _row(discounted, d, "peaker")
        assert row["investment"] * MEGA == pytest.approx(new * FAN_ANNUITY * FAN_F_INV[d], rel=1e-7, abs=1e-6), d
        assert row["replacement"] * MEGA == pytest.approx((operating[d] - new) * FAN_ANNUITY * FAN_F_INV[d],
                                                          rel=1e-7, abs=1e-6), d
        assert row["fixed cost invested"] * MEGA == pytest.approx(want_fix, rel=1e-7, abs=1e-6), d
        row = _row(annualized, d, "peaker")
        assert row["investment"] * MEGA == pytest.approx(new * FAN_ANNUITY, rel=1e-7, abs=1e-6), d
        assert row["replacement"] * MEGA == pytest.approx((operating[d] - new) * FAN_ANNUITY, rel=1e-7, abs=1e-6), d
        assert row["fixed cost invested"] * MEGA == pytest.approx(operating[d] * FAN_FIXED, rel=1e-7, abs=1e-6), d
    _assert_fan_reports_no_operations(fan)


@pytest.mark.solver
def test_fan_without_realization_reports_lead_time_investment(tmp_path_factory) -> None:
    """The fan of :func:`test_fan_without_realization_reports_investment_and_existing_fixed_cost` with a
    construction lead time of 1 year for the peaker: a decision of p2035 starts to operate in p2040 (a decision of
    p2040 would start after the horizon and is not offered).  Nothing operates in the p2035 branches, so the one
    shared p2035 decision covers the larger p2040 deficit, 120 MW (low).  It operates in the p2040 branches,
    whose weights 0.25 + 0.75 sum to 1, so the reported costs are the unweighted amounts, at p2040:
    investment 120 · 1000 · F_inv(p2040) = 115 384.62, fixed cost invested 120 · 20 000 · F_ops(p2040); annualized
    120 · 1000 and 120 · 20 000 per year in p2040 only.  Before: the start p2040 is not a realized investment
    period of the fan, so nothing was reported."""
    fan = _results(tmp_path_factory, "stoch_fan_lead", FAN_VALUES + FAN_LEAD_TIME, ["init", "base", "stoch"],
                   fixture_json=FAN_JSON)
    discounted = fan["cost_unit_discounted__d"]
    assert _row(discounted, "p2040", "peaker")["investment"] * MEGA == pytest.approx(120.0 * 1000.0 / 1.04,
                                                                                    rel=1e-7)
    # Moved from the decision period p2035 to its start p2040: F_inv / F_ops of p2040 only.
    second = _row(discounted, "p2040", "peaker")
    assert second["fixed cost invested"] * MEGA == pytest.approx(120.0 * FAN_FIXED * FAN_F_OPS["p2040"], rel=1e-7)
    annualized = fan["cost_unit_annualized__d"]
    assert _row(annualized, "p2040", "peaker")["investment"] * MEGA == pytest.approx(120.0 * FAN_ANNUITY,
                                                                                    rel=1e-7)
    assert _row(annualized, "p2035", "peaker")["investment"] == pytest.approx(0.0, abs=1e-12)
    _assert_fan_reports_no_operations(fan)


@pytest.mark.solver
@pytest.mark.parametrize("lead_time", [False, True], ids=["no_lead_time", "lead_time"])
def test_recourse_fan_reports_the_branch_weighted_investment(tmp_path_factory, lead_time) -> None:
    """The fan under ``stochastic_invest_method = recourse``: each forecast decides its own investment, and the
    reported investment costs are each branch's investment weighted by its branch weight (high 0.25, low 0.75),
    at the label of the period the capacity starts to operate in.

    No lead time: each branch builds its p2035 deficit in p2035 and adds the rest of its p2040 deficit in p2040
    (cheaper there: no p2035 annuity or fixed cost): high 60 + 30, low 80 + 40 MW.  Weighted: p2035
    0.25 · 60 + 0.75 · 80 = 75 MW (operates in p2035 and p2040), p2040 0.25 · 30 + 0.75 · 40 = 37.5 MW;
    operating 75 MW in p2035 and 112.5 MW in p2040.

    Lead time of 1 year: only p2035 decisions are offered, and they start in p2040; nothing operates in p2035,
    so each branch builds its p2040 deficit: high 90, low 120 MW.  Weighted: 0.25 · 90 + 0.75 · 120 = 112.5 MW
    decided in p2035, reported at p2040 with F_inv / F_ops of p2040 only; operating 112.5 MW in p2040.

    Before: no investment cost was reported under recourse (the branch investment was not in the outputs)."""
    values = FAN_VALUES + (FAN_LEAD_TIME if lead_time else [])
    fan = _results(tmp_path_factory, f"stoch_recourse_{lead_time}", values, ["init", "base", "stoch", "recourse"],
                   fixture_json=FAN_JSON)
    if lead_time:
        discounted = fan["cost_unit_discounted__d"]
        assert _row(discounted, "p2035", "peaker")["investment"] == pytest.approx(0.0, abs=1e-12)
        second = _row(discounted, "p2040", "peaker")
        assert second["investment"] * MEGA == pytest.approx(112.5 * FAN_ANNUITY * FAN_F_INV["p2040"], rel=1e-7)
        assert second["fixed cost invested"] * MEGA == pytest.approx(112.5 * FAN_FIXED * FAN_F_OPS["p2040"],
                                                                     rel=1e-7)
        annualized = fan["cost_unit_annualized__d"]
        for d, mw in {"p2035": 0.0, "p2040": 112.5}.items():
            row = _row(annualized, d, "peaker")
            assert row["investment"] * MEGA == pytest.approx(mw * FAN_ANNUITY, rel=1e-7, abs=1e-6), d
            assert row["fixed cost invested"] * MEGA == pytest.approx(mw * FAN_FIXED, rel=1e-7, abs=1e-6), d
        _assert_fan_reports_no_operations(fan)
    else:
        _assert_fan_investment(fan, built={"p2035": 75.0, "p2040": 37.5}, operating={"p2035": 75.0,
                                                                                      "p2040": 112.5})


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


# ---------------------------------------------------------------------------
# Reported costs without the run-level fields (a CSV-dump reload)
# ---------------------------------------------------------------------------

def test_reload_without_run_fields_warns_once(capsys) -> None:
    """A FlexData with the solve's own factors but no run-level fields (what a CSV-dump reload rebuilds) makes
    the output write print one warning, however many solves; one with the run fields prints none."""
    from types import SimpleNamespace

    from flextool.process_outputs.read_parameters import RUN_FIELDS_MISSING_WARNING, warn_if_run_fields_missing

    reload = SimpleNamespace(p_inflation_op=object(), p_inflation_op_run=None)
    run = SimpleNamespace(p_inflation_op=object(), p_inflation_op_run=object())
    capsys.readouterr()
    assert warn_if_run_fields_missing([reload, reload, run]) is True
    assert capsys.readouterr().out.count(RUN_FIELDS_MISSING_WARNING) == 1
    assert warn_if_run_fields_missing([run, None]) is False
    assert RUN_FIELDS_MISSING_WARNING not in capsys.readouterr().out


@pytest.mark.solver
def test_solved_chain_has_the_run_fields(tmp_path_factory) -> None:
    """Every solve of a solved chain carries the run-level fields, so its output write does not warn."""
    from flextool.process_outputs.read_parameters import warn_if_run_fields_missing

    steps = _steps(tmp_path_factory, "cc_run_fields", [], _no_battery_invest(CHAIN))
    assert len(steps) == 4 and all(s.flex_data is not None for s in steps.values())
    assert warn_if_run_fields_missing([s.flex_data for s in steps.values()]) is False
