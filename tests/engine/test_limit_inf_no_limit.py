"""``inf`` in a limit is "no limit" for that period: the row is not built.

A period-map (or constant) limit whose value is ``inf`` (``-inf`` for a lower limit) in a period has no row for that
(entity, period); before, ``co2_max_period = inf`` was a free row (−inf, inf), a margin of ``-inf`` a free row, and
a huge finite stand-in such as 1e18 reached the LP as a right-hand side.  The criterion, per family, in single,
rolling and stochastic solves:

    LP(limit = inf at X)  =  LP(limit = V at X)  −  the rows of the family at X  (− their slack columns)

with V a finite value that does not bind, so both runs have the same solution and the later rolls of a rolling solve
are built from the same history.  For a lower limit whose 0 already sets no floor (``inertia_limit``,
``invest_min_period``, ``cumulative_min_capacity``), ``-inf`` is the same as 0: the LPs are equal.

The other side: the opposite infinity (``-inf`` on an upper limit, ``inf`` on a lower one) is refused, and a finite
value of at least 1e15 is kept but warned about once per parameter and entity.

Expected outcomes are written by hand (the rows that must be gone, the variable bound of an unlimited investment
period: (existing + max_flow_for_unconstrained_variables) / unitsize = (500 + 1e6) / 500 for coal_plant), never
taken from the new code's output.  Fixtures: ``tests.json`` and ``stochastics.json`` plus an override alternative,
built from JSON.
"""
from __future__ import annotations

import math

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, TIMESTEPS, RunCache, b64, build, unitsize, utf8

pytestmark = pytest.mark.solver

INF = float("inf")
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json"

SOLVE = "y2020_2day_dispatch"
PLANT_CHAIN = ["init", "west", "coal"]
CO2_CHAIN = PLANT_CHAIN + ["co2_limit"]
# Four 5-year periods p2020..p2035 in one solve, all investment periods (y2020_2035_5week).
MULTI_YEAR_CHAIN = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "5weeks",
                    "invest_solveSequence_5weeks", "invest_allYears_5weeks", "co2_limit"]
INVEST_SOLVE = "y2020_2035_5week"
YEARS = ["p2020", "p2025", "p2030", "p2035"]
FLOW_GROUP = "coal_fuel"
FLOW_ENTITIES = [["flowGroup", FLOW_GROUP, None],
                 ["flowGroup__unit__node", [FLOW_GROUP, PLANT, "coal_market"], None]]
STOCH_CHAIN = ["base", "init", "system", "2day"]


def pmap(values: dict, index_name: str = "period") -> list:
    return b64({"type": "map", "index_type": "str", "index_name": index_name,
                "data": [[k, v] for k, v in values.items()]}, "map")


def num(v: float) -> list:
    return b64(float(v), "float")


def text(v: str) -> list:
    return b64(v, "str")


TWO_PERIODS = [
    ["solve", SOLVE, "period_timeset", pmap({"p2020": "2day", "p2025": "2day"})],
    ["solve", SOLVE, "realized_periods",
     b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")],
]
ROLLING = TWO_PERIODS + [["solve", SOLVE, "solve_mode", text("rolling_window")],
                         ["solve", SOLVE, "rolling_solve_horizon", num(24.0)],
                         ["solve", SOLVE, "rolling_solve_jump", num(12.0)]]


_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _steps(tmp_path_factory, alt: str, values: list, chain: list[str], *, entities=None, fixture_json=None):
    kwargs = {} if fixture_json is None else {"fixture_json": fixture_json}
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities,
                                  **kwargs)
    assert steps and all(st.solution is not None for st in steps.values())
    return steps


def _lp(step) -> tuple[dict, dict]:
    """``({row: [lb, ub, {col: coef}]}, {col: (lb, ub, obj)})`` of the step's LP as built (before scaling)."""
    m = build(step.flex_data).canonicalise()
    rows = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][m.col_names[c]] = float(m.val[k])
    cols = {m.col_names[c]: (float(m.col_lb[c]), float(m.col_ub[c]), float(m.col_obj[c])) for c in range(m.n_cols)}
    return rows, cols


def _assert_lp_minus(a_steps: dict, b_steps: dict, rows_gone: tuple[str, ...], cols_gone: tuple[str, ...] = (),
                     *, expect_gone: bool = True) -> None:
    """Every step of ``a_steps`` has the LP of the same step of ``b_steps`` without the rows whose name starts with
    one of ``rows_gone`` and the columns whose name starts with one of ``cols_gone``; with ``expect_gone`` some such
    row exists in ``b``."""
    assert list(a_steps) == list(b_steps)
    gone_total = 0
    for key in a_steps:
        rows_a, cols_a = _lp(a_steps[key])
        rows_b, cols_b = _lp(b_steps[key])
        drop_r = {r for r in rows_b if r.startswith(rows_gone)} if rows_gone else set()
        drop_c = {c for c in cols_b if c.startswith(cols_gone)} if cols_gone else set()
        gone_total += len(drop_r)
        want_rows = {r: [lb, ub, {c: v for c, v in coefs.items() if c not in drop_c}]
                     for r, (lb, ub, coefs) in rows_b.items() if r not in drop_r}
        want_cols = {c: v for c, v in cols_b.items() if c not in drop_c}
        assert set(rows_a) == set(want_rows), (key, sorted(set(rows_a) ^ set(want_rows))[:10])
        assert set(cols_a) == set(want_cols), (key, sorted(set(cols_a) ^ set(want_cols))[:10])
        for r, v in want_rows.items():
            assert rows_a[r] == v, (key, r, rows_a[r], v)
        for c, v in want_cols.items():
            assert cols_a[c] == v, (key, c, cols_a[c], v)
    if expect_gone:
        assert gone_total > 0, f"no row {rows_gone} in the reference LP: the test does not test anything"


def _unitsize(step) -> float:
    """coal_plant's unitsize (an input of the step)."""
    return unitsize(step, PLANT)


def _existing(step, period: str) -> float:
    """coal_plant's existing capacity in ``period`` [MW] (an input of the step)."""
    import polars as pl

    frame = utf8(step.flex_data.p_entity_all_existing.frame.lazy().collect())
    return float(frame.filter((pl.col("e") == PLANT) & (pl.col("d") == period))["value"][0])


def _assert_lp_equal(a_steps: dict, b_steps: dict) -> None:
    _assert_lp_minus(a_steps, b_steps, (), (), expect_gone=False)


# ---------------------------------------------------------------------------
# co2_max_period (upper): single, rolling, stochastic


def test_co2_inf_period_single(tmp_path_factory) -> None:
    """Four periods: inf in p2025 drops co2_max_period[co2_limit,p2025]; the other periods keep their 1e9 t cap."""
    cap = {d: 1.0e9 for d in YEARS}
    a = _steps(tmp_path_factory, "inf_co2_a", [["group", "co2_limit", "co2_max_period", pmap({**cap, "p2025": INF})]],
               MULTI_YEAR_CHAIN)
    b = _steps(tmp_path_factory, "inf_co2_b", [["group", "co2_limit", "co2_max_period", pmap(cap)]],
               MULTI_YEAR_CHAIN)
    _assert_lp_minus(a, b, ("co2_max_period[co2_limit,p2025]",))


def test_co2_inf_constant_is_no_cap(tmp_path_factory) -> None:
    """A constant inf caps no period: the LP has no co2_max_period row at all (before: a free row)."""
    a = _steps(tmp_path_factory, "inf_co2c_a", [["group", "co2_limit", "co2_max_period", num(INF)]], CO2_CHAIN)
    b = _steps(tmp_path_factory, "inf_co2c_b", [["group", "co2_limit", "co2_max_period", num(1.0e9)]], CO2_CHAIN)
    _assert_lp_minus(a, b, ("co2_max_period[",))


def test_co2_inf_period_rolling(tmp_path_factory) -> None:
    """Rolling over p2020 + p2025 (24 h windows, 12 h jumps): every roll of p2020 loses its p2020 cap row, the
    p2025 rows and the rolling carry are unchanged."""
    a = _steps(tmp_path_factory, "inf_co2r_a",
               ROLLING + [["group", "co2_limit", "co2_max_period", pmap({"p2020": INF, "p2025": 1.0e9})]], CO2_CHAIN)
    b = _steps(tmp_path_factory, "inf_co2r_b",
               ROLLING + [["group", "co2_limit", "co2_max_period", pmap({"p2020": 1.0e9, "p2025": 1.0e9})]],
               CO2_CHAIN)
    assert len(a) > 2
    _assert_lp_minus(a, b, ("co2_max_period[co2_limit,p2020]",))


STOCH_CO2 = [["commodity", "gas", "co2_content", num(0.2)], ["group", "co2_limit", "co2_method", text("period")]]
STOCH_CO2_ENTITIES = [["group", "co2_limit", None], ["group__node", ["co2_limit", "gas_market"], None]]


def test_co2_inf_period_stochastic(tmp_path_factory) -> None:
    """The stochastic 2-day dispatch: the branch periods inherit period1's inf, so no forecast has a cap row."""
    a = _steps(tmp_path_factory, "inf_co2s_a",
               STOCH_CO2 + [["group", "co2_limit", "co2_max_period", pmap({"period1": INF})]], STOCH_CHAIN,
               entities=STOCH_CO2_ENTITIES, fixture_json=STOCH_JSON)
    b = _steps(tmp_path_factory, "inf_co2s_b",
               STOCH_CO2 + [["group", "co2_limit", "co2_max_period", pmap({"period1": 1.0e9})]], STOCH_CHAIN,
               entities=STOCH_CO2_ENTITIES, fixture_json=STOCH_JSON)
    _assert_lp_minus(a, b, ("co2_max_period[",))


# ---------------------------------------------------------------------------
# flowGroup flow limits (upper and signed lower): single, rolling, stochastic


@pytest.mark.parametrize(("param", "family", "no_limit", "finite"), [
    ("max_cumulative_flow", "maxCumulative_flow_period", INF, 1.0e6),
    ("min_cumulative_flow", "minCumulative_flow_period", -INF, -1.0e6),
])
def test_cumulative_flow_inf_period(tmp_path_factory, param, family, no_limit, finite) -> None:
    """Two periods: the no-limit value in p2020 drops the p2020 row, p2025 keeps its row."""
    a = _steps(tmp_path_factory, f"inf_{param}_a",
               TWO_PERIODS + [["flowGroup", FLOW_GROUP, param, pmap({"p2020": no_limit, "p2025": finite})]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    b = _steps(tmp_path_factory, f"inf_{param}_b",
               TWO_PERIODS + [["flowGroup", FLOW_GROUP, param, pmap({"p2020": finite, "p2025": finite})]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    _assert_lp_minus(a, b, (f"{family}[{FLOW_GROUP},p2020]",))


def test_cumulative_flow_inf_constant_is_no_solve_limit(tmp_path_factory) -> None:
    """A constant max_cumulative_flow limits the solve total and every period; inf leaves both rows out."""
    a = _steps(tmp_path_factory, "inf_mcf_s_a", [["flowGroup", FLOW_GROUP, "max_cumulative_flow", num(INF)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    b = _steps(tmp_path_factory, "inf_mcf_s_b", [["flowGroup", FLOW_GROUP, "max_cumulative_flow", num(1.0e6)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    _assert_lp_minus(a, b, ("maxCumulative_flow_solve[", "maxCumulative_flow_period["))


def _instant_time_map(step_value: float, other: float) -> list:
    return b64({"index_type": "str", "index_name": "time",
                "data": [[t, step_value if t == "t0005" else other] for t in TIMESTEPS]}, "map")


@pytest.mark.parametrize("rolling", [False, True])
def test_instant_flow_inf_step(tmp_path_factory, rolling) -> None:
    """A time map of max_instant_flow 1e5 MW with inf at t0005: only the t0005 rows are gone (single and every
    roll that holds t0005)."""
    extra = ROLLING if rolling else TWO_PERIODS
    a = _steps(tmp_path_factory, f"inf_mif_{rolling}_a",
               extra + [["flowGroup", FLOW_GROUP, "max_instant_flow", _instant_time_map(INF, 1.0e5)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    b = _steps(tmp_path_factory, f"inf_mif_{rolling}_b",
               extra + [["flowGroup", FLOW_GROUP, "max_instant_flow", _instant_time_map(1.0e5, 1.0e5)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    _assert_lp_minus(a, b, (f"maxInstant_flow[{FLOW_GROUP},p2020,t0005]", f"maxInstant_flow[{FLOW_GROUP},p2025,t0005]"))


def test_min_instant_flow_minus_inf_period_stochastic(tmp_path_factory) -> None:
    """Stochastic 2-day dispatch: min_instant_flow = -inf in period1 (inherited by every branch period) builds no
    minInstant_flow row."""
    ents = [["flowGroup", "fg", None], ["flowGroup__unit__node", ["fg", "gas_plant", "demand_node"], None]]
    a = _steps(tmp_path_factory, "inf_minif_s_a", [["flowGroup", "fg", "min_instant_flow", pmap({"period1": -INF})]],
               STOCH_CHAIN, entities=ents, fixture_json=STOCH_JSON)
    b = _steps(tmp_path_factory, "inf_minif_s_b", [["flowGroup", "fg", "min_instant_flow", pmap({"period1": -1.0e5})]],
               STOCH_CHAIN, entities=ents, fixture_json=STOCH_JSON)
    _assert_lp_minus(a, b, ("minInstant_flow[",))


# ---------------------------------------------------------------------------
# User constraint constant (less_than: inf, greater_than: -inf)


FLOOR = "coal_plant_floor"          # greater_than, Σ flow(coal_plant → west) ≥ 150, in must_run_constraint_one
FLOOR_CHAIN = PLANT_CHAIN + ["must_run_constraint_one"]


@pytest.mark.parametrize("rolling", [False, True])
def test_constraint_minus_inf_period(tmp_path_factory, rolling) -> None:
    """greater_than with constant {p2020: -inf, p2025: -1e6}: no p2020 row; p2025 keeps its rows."""
    extra = ROLLING if rolling else TWO_PERIODS
    a = _steps(tmp_path_factory, f"inf_cn_{rolling}_a",
               extra + [["constraint", FLOOR, "constant", pmap({"p2020": -INF, "p2025": -1.0e6})]], FLOOR_CHAIN)
    b = _steps(tmp_path_factory, f"inf_cn_{rolling}_b",
               extra + [["constraint", FLOOR, "constant", pmap({"p2020": -1.0e6, "p2025": -1.0e6})]], FLOOR_CHAIN)
    _assert_lp_minus(a, b, (f"process_constraint_greater_than[{FLOOR},p2020,",))


def test_constraint_less_than_inf_constant(tmp_path_factory) -> None:
    """less_than with the constant inf: the constraint has no row."""
    sense = [["constraint", FLOOR, "sense", text("less_than")]]
    a = _steps(tmp_path_factory, "inf_cnle_a", sense + [["constraint", FLOOR, "constant", num(INF)]], FLOOR_CHAIN)
    b = _steps(tmp_path_factory, "inf_cnle_b", sense + [["constraint", FLOOR, "constant", num(1.0e6)]], FLOOR_CHAIN)
    _assert_lp_minus(a, b, (f"process_constraint_less_than[{FLOOR},",))


@pytest.mark.parametrize(("sense", "value"), [("equal", INF), ("greater_than", INF), ("less_than", -INF)])
def test_constraint_infinite_constant_that_cannot_hold_is_an_error(tmp_path_factory, sense, value) -> None:
    with pytest.raises(FlexToolConfigError, match=rf"constraint '{FLOOR}', parameter 'constant'"):
        _steps(tmp_path_factory, f"inf_cnbad_{sense}_{value > 0}",
               [["constraint", FLOOR, "sense", text(sense)], ["constraint", FLOOR, "constant", num(value)]],
               FLOOR_CHAIN)


# ---------------------------------------------------------------------------
# Group limits: non-synchronous share (upper), inertia (lower, -inf ≡ 0), capacity margin (lower)


NON_SYNC = [["group", "electricity", "has_non_synchronous", text("yes")],
            ["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", text("yes")]]
NON_SYNC_CHAIN = ["init", "west", "coal", "wind"]


@pytest.mark.parametrize("rolling", [False, True])
def test_non_sync_inf_period(tmp_path_factory, rolling) -> None:
    """non_synchronous_limit {p2020: inf, p2025: 1}: no p2020 row and no p2020 slack."""
    extra = ROLLING if rolling else TWO_PERIODS
    a = _steps(tmp_path_factory, f"inf_ns_{rolling}_a",
               extra + NON_SYNC + [["group", "electricity", "non_synchronous_limit", pmap({"p2020": INF, "p2025": 1.0})]],
               NON_SYNC_CHAIN)
    b = _steps(tmp_path_factory, f"inf_ns_{rolling}_b",
               extra + NON_SYNC + [["group", "electricity", "non_synchronous_limit", pmap({"p2020": 1.0, "p2025": 1.0})]],
               NON_SYNC_CHAIN)
    _assert_lp_minus(a, b, ("non_sync_constraint[electricity,p2020,",), ("vq_non_synchronous[electricity,p2020,",))


@pytest.mark.parametrize("rolling", [False, True])
def test_inertia_minus_inf_is_no_requirement(tmp_path_factory, rolling) -> None:
    """inertia_limit = -inf in p2020 is the same LP as 0 there (no requirement)."""
    extra = ROLLING if rolling else TWO_PERIODS
    chain = ["init", "west", "coal", "wind", "inertia"]
    a = _steps(tmp_path_factory, f"inf_in_{rolling}_a",
               extra + [["group", "electricity", "inertia_limit", pmap({"p2020": -INF, "p2025": 200.0})]], chain)
    b = _steps(tmp_path_factory, f"inf_in_{rolling}_b",
               extra + [["group", "electricity", "inertia_limit", pmap({"p2020": 0.0, "p2025": 200.0})]], chain)
    _assert_lp_equal(a, b)


def _two_invest_periods() -> list:
    return [["solve", INVEST_SOLVE, "invest_periods",
             b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")]]


CAPACITY_MARGIN_CHAIN = MULTI_YEAR_CHAIN + ["network", "capacity_margin"]


def test_capacity_margin_minus_inf_period(tmp_path_factory) -> None:
    """capacity_margin {p2020: -inf, p2025: 100} on the solve investing in p2020 and p2025: no p2020 capacityMargin
    rows and no p2020 slack; p2025 keeps both."""
    a = _steps(tmp_path_factory, "inf_cm_a", _two_invest_periods() + [
        ["group", "capacity_margin", "capacity_margin_absolute", pmap({"p2020": -INF, "p2025": 100.0})]], CAPACITY_MARGIN_CHAIN)
    b = _steps(tmp_path_factory, "inf_cm_b", _two_invest_periods() + [
        ["group", "capacity_margin", "capacity_margin_absolute", pmap({"p2020": -1.0e3, "p2025": 100.0})]],
        CAPACITY_MARGIN_CHAIN)
    _assert_lp_minus(a, b, ("capacityMargin[capacity_margin,p2020,",), ("vq_capacity_margin[capacity_margin,p2020]",))


# ---------------------------------------------------------------------------
# Investment limits


def test_entity_invest_max_period_inf(tmp_path_factory) -> None:
    """coal_plant with invest_period and invest_max_period {p2020: inf, p2025: 300}: no maxInvest_entity_period_p
    row in p2020, and the p2020 invest bound reads no limit: (existing + 1e6) / unitsize units, the bound of an
    invest_no_limit method (max_flow_for_unconstrained_variables 1e6), against (existing + 300) / unitsize with the
    limit 300: the two bounds differ by (1e6 − 300) / unitsize."""
    base = _two_invest_periods() + [["unit", PLANT, "invest_method", text("invest_period")]]
    a = _steps(tmp_path_factory, "inf_imp_a",
               base + [["unit", PLANT, "invest_max_period", pmap({"p2020": INF, "p2025": 300.0})]], MULTI_YEAR_CHAIN)
    b = _steps(tmp_path_factory, "inf_imp_b",
               base + [["unit", PLANT, "invest_max_period", pmap({"p2020": 300.0, "p2025": 300.0})]], MULTI_YEAR_CHAIN)
    rows_a, _ = _lp(list(a.values())[-1])
    bound = f"maxInvest_var_bound[{PLANT},p2020]"
    us = _unitsize(list(a.values())[-1])
    rows_b, _ = _lp(list(b.values())[-1])
    assert math.isclose((rows_a[bound][1] - rows_b[bound][1]) * us, 1.0e6 - 300.0, rel_tol=1e-9)
    assert f"maxInvest_entity_period_p[{PLANT},p2020]" not in rows_a
    assert f"maxInvest_entity_period_p[{PLANT},p2020]" in rows_b
    assert rows_a[f"maxInvest_entity_period_p[{PLANT},p2025]"][1] == 300.0


@pytest.mark.parametrize(("method", "param", "family"), [
    ("invest_period", "invest_min_period", "minInvest_entity_period_p"),
    ("cumulative_limits", "cumulative_min_capacity", "minCumulative_capacity_p"),
])
def test_entity_min_limit_minus_inf_is_zero(tmp_path_factory, method, param, family) -> None:
    """A minimum of -inf sets no floor, as 0 does: the same LP."""
    base = _two_invest_periods() + [["unit", PLANT, "invest_method", text(method)]]
    if method == "invest_period":
        base.append(["unit", PLANT, "invest_max_period", num(1.0e5)])
    a = _steps(tmp_path_factory, f"inf_{param}_a", base + [["unit", PLANT, param, pmap(
        {"p2020": -INF, "p2025": 100.0, "p2030": 100.0, "p2035": 100.0})]], MULTI_YEAR_CHAIN)
    b = _steps(tmp_path_factory, f"inf_{param}_b", base + [["unit", PLANT, param, pmap(
        {"p2020": 0.0, "p2025": 100.0, "p2030": 100.0, "p2035": 100.0})]], MULTI_YEAR_CHAIN)
    _assert_lp_equal(a, b)


def test_group_invest_max_period_inf(tmp_path_factory) -> None:
    """Group wind_coal (coal_plant, wind_plant) with invest_period and invest_max_period {p2020: inf, p2025: 1e5}:
    no p2020 group row; the rest is the LP of {p2020: 1e5, p2025: 1e5}."""
    base = _two_invest_periods() + [["group", "wind_coal", "invest_method", text("invest_period")]]
    a = _steps(tmp_path_factory, "inf_gimp_a", base + [
        ["group", "wind_coal", "invest_max_period", pmap({"p2020": INF, "p2025": 1.0e5})]], MULTI_YEAR_CHAIN)
    b = _steps(tmp_path_factory, "inf_gimp_b", base + [
        ["group", "wind_coal", "invest_max_period", pmap({"p2020": 1.0e5, "p2025": 1.0e5})]], MULTI_YEAR_CHAIN)
    _assert_lp_minus(a, b, ("maxInvestGroup_entity_period_p[wind_coal,p2020]",))


def test_group_cumulative_max_capacity_inf(tmp_path_factory) -> None:
    """Group wind_coal with cumulative_limits and cumulative_max_capacity 1e5 except inf in p2030."""
    base = [["group", "wind_coal", "invest_method", text("cumulative_limits")]]
    cap = {d: 1.0e5 for d in YEARS}
    a = _steps(tmp_path_factory, "inf_gcmc_a", base + [
        ["group", "wind_coal", "cumulative_max_capacity", pmap({**cap, "p2030": INF})]], MULTI_YEAR_CHAIN)
    b = _steps(tmp_path_factory, "inf_gcmc_b", base + [
        ["group", "wind_coal", "cumulative_max_capacity", pmap(cap)]], MULTI_YEAR_CHAIN)
    _assert_lp_minus(a, b, ("maxInvestGroup_entity_cumulative[wind_coal,p2030]",))


def test_entity_cumulative_max_capacity_inf(tmp_path_factory) -> None:
    """coal_plant with cumulative_limits and cumulative_max_capacity {p2020..p2035: 1e5, p2030: inf}: no p2030
    maxCumulative_capacity_p row; the p2030 invest bound is (existing + 1e6) / unitsize, against 1e5 / unitsize
    with the finite cap (the bound of a cumulative cap is the cap itself)."""
    base = [["unit", PLANT, "invest_method", text("cumulative_limits")]]
    cap = {d: 1.0e5 for d in YEARS}
    a = _steps(tmp_path_factory, "inf_cmc_a", base + [["unit", PLANT, "cumulative_max_capacity",
                                                       pmap({**cap, "p2030": INF})]], MULTI_YEAR_CHAIN)
    b = _steps(tmp_path_factory, "inf_cmc_b", base + [["unit", PLANT, "cumulative_max_capacity", pmap(cap)]],
               MULTI_YEAR_CHAIN)
    step_a, step_b = list(a.values())[-1], list(b.values())[-1]
    rows_a, _ = _lp(step_a)
    rows_b, _ = _lp(step_b)
    assert f"maxCumulative_capacity_p[{PLANT},p2030]" not in rows_a
    assert f"maxCumulative_capacity_p[{PLANT},p2030]" in rows_b
    assert rows_a[f"maxCumulative_capacity_p[{PLANT},p2025]"] == rows_b[f"maxCumulative_capacity_p[{PLANT},p2025]"]
    us = _unitsize(step_a)
    existing = _existing(step_a, "p2030")
    bound = f"maxInvest_var_bound[{PLANT},p2030]"
    assert math.isclose(rows_a[bound][1] * us, existing + 1.0e6, rel_tol=1e-9)
    assert math.isclose(rows_b[bound][1] * us, 1.0e5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# The wrong infinity is refused; a huge finite value is warned about (kept as given)


@pytest.mark.parametrize(("cls", "ent", "param", "value", "chain"), [
    ("group", "co2_limit", "co2_max_period", -INF, CO2_CHAIN),
    ("group", "electricity", "inertia_limit", INF, ["init", "west", "coal", "wind", "inertia"]),
])
def test_opposite_infinity_is_an_error(tmp_path_factory, cls, ent, param, value, chain) -> None:
    with pytest.raises(FlexToolConfigError, match=rf"{cls} '{ent}', parameter '{param}': p2020 \({value:g}\)"):
        _steps(tmp_path_factory, f"inf_bad_{param}", [[cls, ent, param, pmap({"p2020": value})]], chain)


def test_huge_finite_limit_warns_once_and_is_kept(tmp_path_factory, coded_log) -> None:
    """co2_max_period = 1e15 in p2020 of a rolling solve: one warning (not one per roll) naming the group, the
    parameter, the period and the value, suggesting inf; the p2020 cap rows keep a finite upper bound of that size
    (the value is used as given, not dropped)."""
    steps = _steps(tmp_path_factory, "inf_huge",
                   ROLLING + [["group", "co2_limit", "co2_max_period", pmap({"p2020": 1.0e15, "p2025": 1.0e9})]],
                   CO2_CHAIN)
    out = coded_log()
    msg = "Warning: limit has a very large value (numerically unstable) ('group co2_limit, co2_max_period') [W-0514]"
    assert out.count(msg) == 1, out[-2000:]
    assert "W-0514 group co2_limit, co2_max_period: p2020 (1e+15)" in out
    rows, _ = _lp(list(steps.values())[0])
    caps = [ub for r, (_lb, ub, _c) in rows.items() if r.startswith("co2_max_period[co2_limit,p2020]")]
    assert caps and all(math.isfinite(ub) and ub > 1.0e14 for ub in caps)


# ---------------------------------------------------------------------------
# A solve step of new_stepduration that mixes inf and finite time steps: the step is inf, warned about once


COARSE = [["solve", SOLVE, "new_stepduration", num(2.0)]]


def _instant_time_map_pair(first: float, second: float, other: float) -> list:
    """A time map: ``first`` at t0005, ``second`` at t0006 (one solve step of new_stepduration 2: t0005), ``other``
    elsewhere."""
    at = {"t0005": first, "t0006": second}
    return b64({"index_type": "str", "index_name": "time",
                "data": [[t, at.get(t, other)] for t in TIMESTEPS]}, "map")


def test_no_limit_keys_are_means() -> None:
    """The mixed-step warning covers MEAN (and SUM) steps, where one infinite time step makes the step infinite;
    every time-series limit is a MEAN."""
    from flextool.engine.input._solve_step_aggregation import NO_LIMIT_SIDE, TIME_AGGREGATION, Agg

    assert {k: TIME_AGGREGATION[k] for k in NO_LIMIT_SIDE} == {k: Agg.MEAN for k in NO_LIMIT_SIDE}


@pytest.mark.parametrize("rolling", [False, True])
def test_instant_flow_inf_mixed_in_solve_step(tmp_path_factory, coded_log, rolling) -> None:
    """new_stepduration 2: max_instant_flow 1e5 MW with inf at t0005 and 1e5 at t0006, so the solve step t0005
    (t0005 + t0006) has the mean (inf + 1e5) / 2 = inf: no maxInstant_flow row at t0005, one warning per run
    naming the flowGroup, the parameter and p2020/t0005.  The reference has 1e5 at t0005 and 2e5 at t0006: the step
    is (1e5 + 2e5) / 2 = 1.5e5 (finite, no warning); every other step is 1e5 in both."""
    extra = COARSE + (ROLLING if rolling else TWO_PERIODS)
    a = _steps(tmp_path_factory, f"inf_mif_mix_{rolling}_a",
               extra + [["flowGroup", FLOW_GROUP, "max_instant_flow", _instant_time_map_pair(INF, 1.0e5, 1.0e5)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    out = coded_log()
    msg = (f"Warning: limit is inf in only part of a new_stepduration step; dropped "
           f"('flowGroup {FLOW_GROUP}, max_instant_flow') [W-0513]")
    assert out.count(msg) == 1, out[-3000:]
    assert f"W-0513 flowGroup {FLOW_GROUP}, max_instant_flow: solve steps p2020/t0005" in out, out[-3000:]
    b = _steps(tmp_path_factory, f"inf_mif_mix_{rolling}_b",
               extra + [["flowGroup", FLOW_GROUP, "max_instant_flow", _instant_time_map_pair(1.0e5, 2.0e5, 1.0e5)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
    assert "W-0513" not in coded_log()
    assert len(a) > (2 if rolling else 0)
    gone = (f"maxInstant_flow[{FLOW_GROUP},p2020,t0005]", f"maxInstant_flow[{FLOW_GROUP},p2025,t0005]")
    _assert_lp_minus(a, b, gone)
    step = next(s for s in b.values() if any(r.startswith(gone) for r in _lp(s)[0]))
    rows, _ = _lp(step)
    assert [ub for r, (_lb, ub, _c) in rows.items() if r.startswith(gone)][0] == 1.5e5
    assert not any(r.startswith(f"maxInstant_flow[{FLOW_GROUP},p2020,t0006]") for r in rows)


def test_instant_flow_inf_and_minus_inf_in_solve_step_is_an_error(tmp_path_factory) -> None:
    """inf at t0005 and -inf at t0006 (one solve step of new_stepduration 2) have no mean: refused."""
    with pytest.raises(FlexToolConfigError, match=rf"flowGroup '{FLOW_GROUP}', parameter 'max_instant_flow': "
                                                  r"p2020/t0005.*combine time steps of inf and -inf"):
        _steps(tmp_path_factory, "inf_mif_mix_both", COARSE + TWO_PERIODS + [
            ["flowGroup", FLOW_GROUP, "max_instant_flow", _instant_time_map_pair(INF, -INF, 1.0e5)]],
               PLANT_CHAIN, entities=FLOW_ENTITIES)
