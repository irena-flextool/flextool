"""A period map that leaves out a period of the solve is an error; an explicit 0 is a value.

A limit given as a period map must give a value for every period the solve uses (including look-ahead
periods; for the investment limits and the capacity margin, every investment period of the solve).  Before, a
missing period was silent: ``co2_max_period`` became a cap of 0, ``inertia_limit`` no requirement, a missing
``cumulative_min_capacity`` / cumulative / instant flow limit no row; the investment maxima failed with an error
about another entity.  Now the reader raises ``FlexToolConfigError`` naming the class, the entity, the parameter
and the missing periods.  A constant applies to every period.  The check only covers limits that are switched on
(the method or the flag), and a stochastic branch period inherits its anchor's value.

Two related fixes:

- (a) an explicit 0 in ``max_cumulative_flow`` / ``min_cumulative_flow`` / ``max_instant_flow`` /
  ``min_instant_flow`` is a limit of 0 (the net flow into a flowGroup is signed); before it was dropped as
  "no limit".
- (b) a ``co2_method`` that caps emissions per period (or in total) without a ``co2_max_period``
  (``co2_max_total``) value is an error; before it was a cap of 0 in every period, or a developer error.

Expected outcomes are specified by hand (error / no error, a 0 right-hand side), not taken from the code.
Fixtures: ``tests.json`` and ``stochastics.json`` plus an override alternative, built from JSON.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_bounds
from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, TIMESTEPS, b64, build, make_db, run, utf8

# Four 5-year periods p2020..p2035 in one solve (y2020_2035_5week), all of them investment periods.
MULTI_YEAR_CHAIN = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "5weeks",
                    "invest_solveSequence_5weeks", "invest_allYears_5weeks", "co2_limit"]
INVEST_SOLVE = "y2020_2035_5week"
# y2020_5week: periods p2020 (realized) and p2025 (look-ahead).
LOOKAHEAD_CHAIN = ["init", "west", "coal", "5weeks", "co2_limit"]
# y2020_2day_dispatch: one period p2020.
PLANT_CHAIN = ["init", "west", "coal"]
INERTIA_CHAIN = ["init", "west", "coal", "wind", "inertia"]
FLOW_GROUP = "coal_fuel"
FLOW_ENTITIES = [["flowGroup", FLOW_GROUP, None],
                 ["flowGroup__unit__node", [FLOW_GROUP, PLANT, "coal_market"], None]]
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json"


def pmap(values: dict, index_name: str | None = "period") -> list:
    """A period map (``index_name=None``: Spine's silent ``x`` label)."""
    m = {"type": "map", "index_type": "str", "rank": 1, "data": [[k, v] for k, v in values.items()]}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


def num(v: float) -> list:
    return b64(float(v), "float")


def text(v: str) -> list:
    return b64(v, "str")


def _run(tmp_path_factory, alt: str, values: list, chain: list[str], *, entities=None, fixture_json=None):
    kwargs = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities, **kwargs)
    return run(tmp_path_factory, url, scen)


def _bounds(tmp_path_factory, alt: str, values: list, family: str, chain=PLANT_CHAIN, entities=FLOW_ENTITIES):
    step = list(_run(tmp_path_factory, alt, values, chain, entities=entities).values())[-1]
    return row_bounds(build(step.flex_data), family)


# ---------------------------------------------------------------------------
# A missing period is an error


def test_co2_max_period_gap_is_an_error(tmp_path_factory) -> None:
    """``co2_max_period = {p2020: 3e6, p2025: 2.4e6}`` on the four-period solve: p2030 and p2035 are missing
    (before: a cap of 0 t in both)."""
    with pytest.raises(FlexToolConfigError,
                       match=r"group 'co2_limit', parameter 'co2_max_period'.*p2030, p2035"):
        _run(tmp_path_factory, "c1_co2_gap",
             [["group", "co2_limit", "co2_max_period", pmap({"p2020": 3.0e6, "p2025": 2.4e6})]],
             MULTI_YEAR_CHAIN)


def test_lookahead_period_is_checked(tmp_path_factory) -> None:
    """y2020_5week realizes p2020 and looks ahead into p2025: a map with p2020 only misses p2025."""
    with pytest.raises(FlexToolConfigError, match=r"group 'co2_limit', parameter 'co2_max_period'.*p2025"):
        _run(tmp_path_factory, "c1_co2_lookahead",
             [["group", "co2_limit", "co2_max_period", pmap({"p2020": 3.0e6})]], LOOKAHEAD_CHAIN)


@pytest.mark.parametrize(("flag", "raises"), [("yes", True), ("no", False)])
def test_inertia_limit_gap_follows_the_flag(tmp_path_factory, flag, raises) -> None:
    """``inertia_limit = {p2025: 200}`` on the one-period (p2020) solve: an error with ``has_inertia = yes``,
    none with ``has_inertia = no`` (the limit is not used)."""
    values = [["group", "electricity", "inertia_limit", pmap({"p2025": 200.0})],
              ["group", "electricity", "has_inertia", text(flag)]]
    if raises:
        with pytest.raises(FlexToolConfigError,
                           match=r"group 'electricity', parameter 'inertia_limit'.*p2020"):
            _run(tmp_path_factory, f"c1_inertia_{flag}", values, INERTIA_CHAIN)
    else:
        assert _run(tmp_path_factory, f"c1_inertia_{flag}", values, INERTIA_CHAIN)


def _two_invest_periods() -> list:
    """y2020_2035_5week invests in p2020 and p2025 only; p2030 and p2035 are dispatch-only."""
    return [["solve", INVEST_SOLVE, "invest_periods",
             b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")]]


def test_invest_max_period_missing_dispatch_only_period_is_fine(tmp_path_factory) -> None:
    """``invest_max_period = {p2020: 250, p2025: 350}`` covers both investment periods; the dispatch-only
    p2030 / p2035 need no value."""
    values = _two_invest_periods() + [
        ["unit", PLANT, "invest_max_period", pmap({"p2020": 250.0, "p2025": 350.0})]]
    assert _run(tmp_path_factory, "c1_imp_dispatch_only", values, MULTI_YEAR_CHAIN)


def test_invest_max_period_missing_invest_period_is_an_error(tmp_path_factory) -> None:
    """``invest_max_period = {p2020: 250}`` misses the investment period p2025 (before: the FA-12 wording)."""
    values = _two_invest_periods() + [["unit", PLANT, "invest_max_period", pmap({"p2020": 250.0})]]
    with pytest.raises(FlexToolConfigError, match=r"unit 'coal_plant', parameter 'invest_max_period'.*p2025"):
        _run(tmp_path_factory, "c1_imp_invest_gap", values, MULTI_YEAR_CHAIN)


def test_silent_label_partial_cumulative_max_capacity_is_an_error(tmp_path_factory) -> None:
    """A ``cumulative_max_capacity`` map with Spine's silent ``x`` label and only p2020 / p2025 misses the
    investment periods p2030 and p2035."""
    values = [["unit", PLANT, "invest_method", text("cumulative_limits")],
              ["unit", PLANT, "cumulative_max_capacity", pmap({"p2020": 500.0, "p2025": 500.0}, index_name=None)]]
    with pytest.raises(FlexToolConfigError,
                       match=r"unit 'coal_plant', parameter 'cumulative_max_capacity'.*p2030, p2035"):
        _run(tmp_path_factory, "c1_cmc_silent", values, MULTI_YEAR_CHAIN)


def test_error_names_only_the_map_entity(tmp_path_factory) -> None:
    """coal_plant has a constant (every period), wind_plant a map with p2020 only: the error names wind_plant
    and its missing p2025, p2030, p2035, and not coal_plant."""
    values = [["unit", PLANT, "invest_method", text("cumulative_limits")],
              ["unit", "wind_plant", "invest_method", text("cumulative_limits")],
              ["unit", PLANT, "cumulative_max_capacity", num(500.0)],
              ["unit", "wind_plant", "cumulative_max_capacity", pmap({"p2020": 500.0})]]
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, "c1_cmc_two", values, MULTI_YEAR_CHAIN)
    msg = str(err.value)
    assert "unit 'wind_plant', parameter 'cumulative_max_capacity'" in msg
    assert "p2025, p2030, p2035" in msg
    assert "coal_plant" not in msg


# ---------------------------------------------------------------------------
# No false errors


def test_explicit_zero_is_a_value(tmp_path_factory) -> None:
    """``cumulative_min_capacity = {p2020: 0, p2025..p2035: 100}`` covers every period: no error (the zero is
    checked before the zero filter).  A min of 0 is no floor, so p2020 has no row."""
    values = [["unit", PLANT, "invest_method", text("cumulative_limits")],
              ["unit", PLANT, "cumulative_min_capacity",
               pmap({"p2020": 0.0, "p2025": 100.0, "p2030": 100.0, "p2035": 100.0})]]
    step = list(_run(tmp_path_factory, "c1_cmin_zero", values, MULTI_YEAR_CHAIN).values())[-1]
    rows = utf8(step.flex_data.ed_cumulative_min_capacity.frame.lazy().collect())
    assert sorted(rows.filter(rows["e"] == PLANT)["d"].to_list()) == ["p2025", "p2030", "p2035"]


def test_branch_period_inherits_the_anchor_value(tmp_path_factory) -> None:
    """stochastics.json 2day_dispatch fans period1 into forecast branches; ``co2_max_period = {period1: 2e6}``
    gives no branch period a value of its own, yet every branch period inherits 2e6: no error."""
    values = [["commodity", "gas", "co2_content", num(0.2)],
              ["group", "co2_limit", "co2_method", text("period")],
              ["group", "co2_limit", "co2_max_period", pmap({"period1": 2.0e6})]]
    entities = [["group", "co2_limit", None], ["group__node", ["co2_limit", "gas_market"], None]]
    steps = _run(tmp_path_factory, "c1_stoch", values, ["base", "init", "system", "2day"], entities=entities,
                 fixture_json=STOCH_JSON)
    for step in steps.values():
        cap = utf8(step.flex_data.p_co2_max_period.frame.lazy().collect())
        periods = set(utf8(step.flex_data.dt)["d"].to_list())
        assert set(cap["d"].to_list()) == periods
        assert set(cap["value"].to_list()) == {2.0e6}


# ---------------------------------------------------------------------------
# (a) an explicit 0 in a flowGroup flow limit is a limit


def test_max_cumulative_flow_zero_in_a_map_is_a_row(tmp_path_factory) -> None:
    rows = _bounds(tmp_path_factory, "c1_mcf_map0",
                   [["flowGroup", FLOW_GROUP, "max_cumulative_flow", pmap({"p2020": 0.0})]],
                   "maxCumulative_flow_period")
    assert rows[f"maxCumulative_flow_period[{FLOW_GROUP},p2020]"][1] == 0.0


def test_max_cumulative_flow_zero_constant_is_a_row(tmp_path_factory) -> None:
    rows = _bounds(tmp_path_factory, "c1_mcf_s0",
                   [["flowGroup", FLOW_GROUP, "max_cumulative_flow", num(0.0)]], "maxCumulative_flow_solve")
    assert rows[f"maxCumulative_flow_solve[{FLOW_GROUP}]"][1] == 0.0


def test_min_cumulative_flow_zero_is_a_row(tmp_path_factory) -> None:
    rows = _bounds(tmp_path_factory, "c1_mincf_0",
                   [["flowGroup", FLOW_GROUP, "min_cumulative_flow", pmap({"p2020": 0.0})]],
                   "minCumulative_flow_period")
    assert rows[f"minCumulative_flow_period[{FLOW_GROUP},p2020]"][0] == 0.0


def test_max_instant_flow_zero_step_is_a_row(tmp_path_factory) -> None:
    """A time map of 1000 with 0 at t0005: t0005 has a row with the upper bound 0."""
    tmap = b64({"index_type": "str", "index_name": "time",
                "data": [[t, 0.0 if t == "t0005" else 1000.0] for t in TIMESTEPS]}, "map")
    rows = _bounds(tmp_path_factory, "c1_mif_0",
                   [["flowGroup", FLOW_GROUP, "max_instant_flow", tmap]], "maxInstant_flow")
    assert rows[f"maxInstant_flow[{FLOW_GROUP},p2020,t0005]"][1] == 0.0


# ---------------------------------------------------------------------------
# (b) a CO2 cap method needs its value


@pytest.mark.parametrize("sibling", [True, False])
def test_co2_period_method_without_value_is_an_error(tmp_path_factory, sibling) -> None:
    """Group co2_price with ``co2_method = price_period`` and no ``co2_max_period``, next to co2_limit that has
    one (before: a cap of 0) and alone (before: a developer error)."""
    chain = PLANT_CHAIN + (["co2_limit"] if sibling else [])
    with pytest.raises(FlexToolConfigError,
                       match=r"group 'co2_price', co2_method 'price_period' needs parameter 'co2_max_period'"):
        _run(tmp_path_factory, f"c1_co2_noval_{sibling}",
             [["group", "co2_price", "co2_method", text("price_period")]], chain)


def test_co2_total_method_without_value_is_an_error(tmp_path_factory) -> None:
    with pytest.raises(FlexToolConfigError,
                       match=r"group 'co2_limit', co2_method 'total' needs parameter 'co2_max_total'"):
        _run(tmp_path_factory, "c1_co2_total_noval",
             [["group", "co2_limit", "co2_method", text("total")]], PLANT_CHAIN + ["co2_limit"])
