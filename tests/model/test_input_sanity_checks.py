"""Values that make no sense or are almost surely typing mistakes (#170).

``validate_basic_sanity`` (``input_derivation.run``, before the parameter read) stops the run on values the model
cannot use sensibly and warns about values that are likely entered in the wrong unit.  Before the check:
``efficiency 0`` turned coal_plant off, ``efficiency -0.4`` paid it to run, ``capacity_max_coeff -1`` and a minimum
above its maximum ended "infeasible on presolve" without a hint, and ``availability 95`` made more capacity than
there is, all without a word; a negative ``virtual_unitsize`` (reported as ignored) still multiplied the
capacity.  A negative ``existing`` is supported (anti-energy) but warned about.  (``existing`` as a time series or keyed by time steps or numbers is
``validate_period_keys``': ``test_period_key_checks.py``.)

Databases: ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal`` (coal_plant: input
coal_market, output west, efficiency 0.4, existing 500; model period p2020, timeline t0001..t0072) or on
``init, west, coal, wind, network`` (connection west_east).  Messages are the expected text, written by hand.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests._support.coded_capture import capture_coded
from tests.engine._unit_db import PLANT_CHAIN, b64, make_db, run

NETWORK_CHAIN = ["init", "west", "coal", "wind", "network"]
PREFIX = "Input values out of range:\n  "


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _f(x: float) -> list:
    return b64(x, "float")


def _map(data: list, index_type: str = "str") -> list:
    return b64({"index_type": index_type, "data": data}, "map")


def _validate(url: str, scenario: str) -> None:
    from flextool.input_derivation._validators import validate_basic_sanity
    from flextool.spinedb_backend import SpineDBBackend

    with SpineDBBackend(url, scenario) as backend:
        validate_basic_sanity(backend)


def _error(tmp_path_factory, alt: str, values: list, chain: list[str] = PLANT_CHAIN) -> str:
    with pytest.raises(FlexToolConfigError) as err:
        _validate(*make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain))
    return str(err.value)


def test_zero_efficiency_stops_the_run(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b170_eff0", base_chain=PLANT_CHAIN,
                        values=[["unit", "coal_plant", "efficiency", _f(0.0)]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert str(err.value) == PREFIX + (
        "unit 'coal_plant' efficiency = 0 (alternative 'b170_eff0'): efficiency must be above 0 (0 gives no output, "
        "a negative value pays the unit to run). Use a fraction such as 0.4.")


def test_negative_connection_efficiency_names_the_period(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_conn_eff", [
        ["connection", "west_east", "efficiency", _map([["p2020", 0.9], ["p2030", -0.9]])]], NETWORK_CHAIN)
    assert "connection 'west_east' efficiency = -0.9 (alternative 'b170_conn_eff', at p2030): efficiency must " \
           "be above 0" in msg, msg


@pytest.mark.parametrize("alt, values, expected", [
    ("b170_minload", [["unit", "coal_plant", "min_load", _f(1.5)]],
     "unit 'coal_plant' min_load = 1.5 (alternative 'b170_minload'): min_load is a share of the online capacity "
     "and must lie in [0, 1] (e.g. 0.3 for 30 %)."),
    ("b170_avail_neg", [["unit", "coal_plant", "availability", _f(-0.1)]],
     "unit 'coal_plant' availability = -0.1 (alternative 'b170_avail_neg'): availability must be 0 or above."),
    ("b170_cmax", [["unit__outputNode", ["coal_plant", "west"], "capacity_max_coeff", _f(-1.0)]],
     "unit__outputNode 'coal_plant, west' capacity_max_coeff = -1 (alternative 'b170_cmax'): capacity_max_coeff "
     "must be 0 or above (it is the share of the capacity this output may use)."),
])
def test_out_of_range_values_are_errors(tmp_path_factory, alt, values, expected) -> None:
    assert expected in _error(tmp_path_factory, alt, values)


def test_invest_min_total_above_max_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_total", [
        ["unit", "coal_plant", "invest_method", b64("invest_total", "str")],
        ["unit", "coal_plant", "invest_min_total", _f(900.0)],
        ["unit", "coal_plant", "invest_max_total", _f(800.0)]])
    assert ("unit 'coal_plant' invest_min_total = 900 is above invest_max_total = 800 (alternatives 'b170_total', "
            "'b170_total'; invest_method invest_total): no capacity can meet both, so the problem is infeasible. "
            "Lower invest_min_total or raise invest_max_total.") in msg, msg


def test_invest_min_period_above_a_constant_max_is_an_error_per_period(tmp_path_factory) -> None:
    """Before the check this run ended 'infeasible on presolve' (FlexToolSolveError) without a hint."""
    url, scen = make_db(tmp_path_factory, alt="b170_period", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "invest_method", b64("invest_period", "str")],
        ["unit", "coal_plant", "invest_cost", _f(400.0)],
        ["unit", "coal_plant", "lifetime_technical", _f(30.0)],
        ["unit", "coal_plant", "invest_min_period", _map([["p2020", 200.0]])],
        ["unit", "coal_plant", "invest_max_period", _f(100.0)]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert "unit 'coal_plant' invest_min_period = 200 is above invest_max_period = 100 in period p2020" \
        in str(err.value), str(err.value)


def test_min_above_max_is_silent_when_the_method_does_not_use_the_pair(tmp_path_factory) -> None:
    _validate(*make_db(tmp_path_factory, alt="b170_unused", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "invest_method", b64("invest_total", "str")],
        ["unit", "coal_plant", "invest_min_period", _f(200.0)],
        ["unit", "coal_plant", "invest_max_period", _f(100.0)]]))


def test_cumulative_period_in_only_one_map_is_not_compared(tmp_path_factory) -> None:
    _validate(*make_db(tmp_path_factory, alt="b170_cum_apart", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "invest_method", b64("cumulative_limits", "str")],
        ["unit", "coal_plant", "cumulative_min_capacity", _map([["p2020", 900.0]])],
        ["unit", "coal_plant", "cumulative_max_capacity", _map([["p2030", 100.0]])]]))


def test_cumulative_min_above_max_in_a_shared_period_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_cum", [
        ["unit", "coal_plant", "invest_method", b64("cumulative_limits", "str")],
        ["unit", "coal_plant", "cumulative_min_capacity", _map([["p2020", 900.0]])],
        ["unit", "coal_plant", "cumulative_max_capacity", _map([["p2020", 100.0], ["p2030", 1000.0]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant' cumulative_min_capacity = 900 is above cumulative_max_capacity = 100 in period p2020 "
        "(alternatives 'b170_cum', 'b170_cum'; invest_method cumulative_limits): no capacity can meet both, so the "
        "problem is infeasible. Lower cumulative_min_capacity or raise cumulative_max_capacity."), msg


@pytest.mark.parametrize("name", ["lifetime_technical", "payback_time"])
def test_zero_lifetime_of_an_investing_unit_is_an_error(tmp_path_factory, name) -> None:
    msg = _error(tmp_path_factory, f"b170_life_{name}", [
        ["unit", "coal_plant", "invest_method", b64("invest_no_limit", "str")],
        ["unit", "coal_plant", name, _f(0.0)]])
    assert (f"unit 'coal_plant' {name} = 0 (alternative 'b170_life_{name}'): {name} must be above 0 when the unit "
            f"can invest or retire; with 0 the investment is charged for no years. Give the {name} in years.") in msg


@pytest.mark.parametrize("name", ["lifetime_technical", "payback_time"])
def test_zero_lifetime_without_investment_is_silent(tmp_path_factory, name) -> None:
    _validate(*make_db(tmp_path_factory, alt=f"b170_life_off_{name}", base_chain=PLANT_CHAIN,
                       values=[["unit", "coal_plant", name, _f(0.0)]]))


def test_negative_discount_rate_is_an_error(tmp_path_factory) -> None:
    """Before v79 a discount_rate ≤ 0 was read as 5 %; 0 is now 0 % and a negative rate is refused."""
    msg = _error(tmp_path_factory, "le_disc_neg", [["unit", "coal_plant", "discount_rate", _f(-0.01)]])
    assert ("unit 'coal_plant' discount_rate = -0.01 (alternative 'le_disc_neg'): discount_rate must be 0 or above "
            "(0 gives the straight-line annuity invest_cost / payback_time).") in msg, msg


def test_zero_discount_rate_is_silent(tmp_path_factory) -> None:
    _validate(*make_db(tmp_path_factory, alt="le_disc_zero", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "invest_method", b64("invest_no_limit", "str")],
        ["unit", "coal_plant", "invest_cost", _f(400.0)],
        ["unit", "coal_plant", "lifetime_technical", _f(30.0)],
        ["unit", "coal_plant", "discount_rate", _f(0.0)]]))


def test_investing_unit_with_a_cost_needs_a_lifetime(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "le_life_missing", [
        ["unit", "coal_plant", "invest_method", b64("invest_no_limit", "str")],
        ["unit", "coal_plant", "invest_cost", _f(400.0)]])
    assert ("unit 'coal_plant' has no lifetime_technical, but it can invest (invest_method invest_no_limit) and has "
            "an invest_cost. Give the lifetime_technical in years.") in msg, msg


def test_bounded_lifetime_method_with_existing_needs_a_lifetime(tmp_path_factory) -> None:
    """No investment, but reinvest_choice ends the existing 500 MW after start + lifetime: without a lifetime
    the capacity would be gone in every period."""
    msg = _error(tmp_path_factory, "le_life_bounded", [
        ["unit", "coal_plant", "lifetime_method", b64("reinvest_choice", "str")]])
    assert ("unit 'coal_plant' has no lifetime_technical, but its lifetime_method is reinvest_choice and it has "
            "existing capacity, so its capacity ends after the lifetime. Give the lifetime_technical in "
            "years.") in msg, msg


@pytest.mark.parametrize("alt, values", [
    ("le_life_default", []),
    ("le_life_retire_stray_cost", [["unit", "coal_plant", "invest_method", b64("retire_no_limit", "str")],
                                   ["unit", "coal_plant", "invest_cost", _f(400.0)]]),
    ("le_life_invest_period_total_salvage", [
        ["unit", "coal_plant", "invest_method", b64("invest_period_total", "str")],
        ["unit", "coal_plant", "net_salvage_value", _f(100.0)]]),
])
def test_lifetime_not_needed_is_silent(tmp_path_factory, alt, values) -> None:
    """The default reinvest_automatic without investment, a retiring unit with a stray invest_cost (its
    annuity uses net_salvage_value), and an invest_period_total unit with existing capacity and a stray
    net_salvage_value (it cannot retire) need no lifetime."""
    _validate(*make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN,
                       values=values or [["unit", "coal_plant", "efficiency", _f(0.4)]]))


@pytest.mark.parametrize("alt, values, chain, code, name, cls, detail", [
    ("b170_avail_pct", [["unit", "coal_plant", "availability", _f(95.0)]], PLANT_CHAIN,
     "W-0226", "coal_plant", "unit", "availability = 95 (alternative 'b170_avail_pct')"),
    ("b170_conn_gain", [["connection", "west_east", "efficiency", _f(1.1)]], NETWORK_CHAIN,
     "W-0222", "west_east", "connection", "efficiency = 1.1 (alternative 'b170_conn_gain')"),
    ("b170_vus", [["unit", "coal_plant", "virtual_unitsize", _f(0.0)]], PLANT_CHAIN,
     "W-0224", "coal_plant", "unit", "virtual_unitsize = 0 (alternative 'b170_vus')"),
])
def test_likely_unit_mistakes_warn(tmp_path_factory, alt, values, chain, code, name, cls, detail) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain))
    assert got.names(code, cls=cls) == [name]
    assert got.detail(code, name) == detail


def test_negative_existing_of_a_unit_warns_and_the_run_continues(tmp_path_factory) -> None:
    """Anti-energy units are supported (the anti_energy_plant example), so a warning, not an error."""
    with capture_coded() as got:
        run(tmp_path_factory, *make_db(tmp_path_factory, alt="b170_anti", base_chain=PLANT_CHAIN,
                                       values=[["unit", "coal_plant", "existing", _f(-50.0)]]))
    assert got.names("W-0223", cls="unit") == ["coal_plant"]
    assert got.detail("W-0223", "coal_plant") == "existing = -50 (alternative 'b170_anti')"


def test_negative_existing_in_a_period_map_names_the_period(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b170_anti_map", base_chain=PLANT_CHAIN,
                           values=[["unit", "coal_plant", "existing", _map([["p2020", -50.0]])]]))
    assert got.detail("W-0223", "coal_plant") == "existing = -50 (alternative 'b170_anti_map', at p2020)"


def test_negative_existing_of_a_node_warns(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b170_node_neg", base_chain=PLANT_CHAIN,
                           values=[["node", "west", "existing", _f(-10.0)]]))
    assert got.names("W-0223", cls="node") == ["west"]
    assert got.detail("W-0223", "west") == "existing = -10 (alternative 'b170_node_neg')"


def test_negative_existing_of_a_connection_warns(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b170_conn_neg", base_chain=NETWORK_CHAIN,
                           values=[["connection", "west_east", "existing", _f(-100.0)]]))
    assert got.names("W-0223", cls="connection") == ["west_east"]
    assert got.detail("W-0223", "west_east") == "existing = -100 (alternative 'b170_conn_neg')"


def test_zero_existing_is_silent(tmp_path_factory) -> None:
    with capture_coded() as got:
        _validate(*make_db(tmp_path_factory, alt="b170_ex0", base_chain=PLANT_CHAIN, values=[
            ["unit", "coal_plant", "existing", _f(0.0)], ["node", "west", "existing", _f(0.0)]]))
    assert "W-0223" not in got.codes()


def test_negative_virtual_unitsize_is_ignored(tmp_path_factory) -> None:
    """The warning says "ignored": the unit is one unit of its existing capacity (unitsize 200, capacity 200), not
    200 units of -5 MW (an anti-energy unit of -1000 MW, which the cascade made of any non-zero value)."""
    import polars as pl

    from tests.engine._unit_db import utf8

    with capture_coded() as got:
        steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt="b170_vu_neg", base_chain=PLANT_CHAIN, values=[
            ["unit", "coal_plant", "existing", _f(200.0)], ["unit", "coal_plant", "virtual_unitsize", _f(-5.0)]]))
    assert got.detail("W-0224", "coal_plant") == "virtual_unitsize = -5 (alternative 'b170_vu_neg')"
    fd = next(iter(steps.values())).flex_data
    unitsize = utf8(fd.p_unitsize.frame).filter(pl.col("p") == "coal_plant")["value"].to_list()
    capacity = utf8(fd.p_entity_all_existing.frame).filter(pl.col("e") == "coal_plant")["value"].to_list()
    assert unitsize == [200.0]
    assert capacity == [200.0]


def test_profile_entered_as_percent_warns(tmp_path_factory) -> None:
    """``warn_profiles_as_percent`` runs after the parameter read (on the reader's profile frames)."""
    with capture_coded() as got:
        run(tmp_path_factory, *make_db(tmp_path_factory, alt="b170_wind_pct", base_chain=["init", "west", "wind"],
                                       values=[["profile", "wind_profile", "profile",
                                                _map([["t0001", 0.3], ["t0002", 35.0]])]]))
    assert got.names("W-0307") == ["wind_profile"]
    assert got.detail("W-0307", "wind_profile") == "35 (alternative 'b170_wind_pct', at t0002)"


def test_shares_slightly_above_one_are_silent(tmp_path_factory) -> None:
    """Measured or reanalysis capacity factors reach a little above 1 (rounding, over-performance)."""
    with capture_coded() as got:
        run(tmp_path_factory, *make_db(tmp_path_factory, alt="b170_cf", base_chain=["init", "west", "wind"], values=[
            ["profile", "wind_profile", "profile", _map([["t0001", 0.3], ["t0002", 1.0235]])],
            ["unit", "coal_plant", "availability", _f(1.2)]]))
    assert "W-0307" not in got.codes() and "W-0226" not in got.codes(), got.codes()


def test_errors_of_the_input_checks_come_in_one_message(tmp_path_factory) -> None:
    """A profile link without data, an out-of-range value and a mistyped period key: one error, one rerun."""
    url, scen = make_db(tmp_path_factory, alt="b170_all", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "efficiency", _f(0.0)],
        ["unit", "coal_plant", "startup_cost", _map([["y2020", 10.0]])]],
        entities=[["profile", "ghost", None], ["unit__node__profile", ["coal_plant", "west", "ghost"], None]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert str(err.value) == (
        "Map keys that are not periods of any solve in the database:\n  unit 'coal_plant', parameter "
        "'startup_cost' (alternative 'b170_all'): map key 'y2020' is not a period. Did you mean 'p2020'?\nIf a key "
        "is a period of data you keep for other models, add it to model.periods_available."
        "\n\nProfile links without profile data:\n  unit__node__profile (coal_plant, west, ghost): profile 'ghost' "
        "has no value in any alternative, so it holds the flow at 0 (profile_method is upper_limit (the default)). "
        "Give the profile a value in an alternative of the scenario, set profile_method lower_limit or none, or "
        "remove the link."
        "\n\n" + PREFIX + "unit 'coal_plant' efficiency = 0 (alternative 'b170_all'): efficiency must be above 0 "
        "(0 gives no output, a negative value pays the unit to run). Use a fraction such as 0.4.")


# ---------------------------------------------------------------------------
# The age of existing capacity: age_at_model_start, or model.start_year − build_year


@pytest.mark.parametrize("alt, values, expected", [
    ("age_both", [["unit", "coal_plant", "age_at_model_start", _f(3.0)],
                  ["unit", "coal_plant", "build_year", _f(2017.0)],
                  ["model", "flexTool", "start_year", _f(2020.0)]],
     "unit 'coal_plant' has both age_at_model_start 3 and build_year 2017: give one of them (alternative "
     "'age_both')."),
    ("age_no_start", [["unit", "coal_plant", "build_year", _f(2017.0)]],
     "unit 'coal_plant' has build_year 2017 but the model has no start_year: give model.start_year, or use "
     "age_at_model_start (alternative 'age_no_start')."),
    ("age_negative", [["unit", "coal_plant", "age_at_model_start", _f(-2.0)]],
     "unit 'coal_plant' age_at_model_start = -2: the age must be 0 or more (alternative 'age_negative')."),
    ("age_built_late", [["unit", "coal_plant", "build_year", _f(2025.0)],
                        ["model", "flexTool", "start_year", _f(2020.0)]],
     "unit 'coal_plant' build_year 2025 is after model.start_year 2020: capacity built after the model start is "
     "an investment (alternative 'age_built_late')."),
])
def test_existing_age_errors(tmp_path_factory, alt, values, expected) -> None:
    msg = _error(tmp_path_factory, alt, values)
    assert msg == PREFIX + expected, msg


def test_model_start_years_must_agree(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="age_two_starts", base_chain=PLANT_CHAIN, values=[
        ["model", "flexTool", "start_year", _f(2020.0)], ["model", "other", "start_year", _f(2025.0)]],
        entities=[["model", "other", None]])
    with pytest.raises(FlexToolConfigError) as err:
        _validate(url, scen)
    assert str(err.value) == PREFIX + (
        "model.start_year differs between the models of the scenario: flexTool 2020 (alternative 'age_two_starts'), "
        "other 2025 (alternative 'age_two_starts'); give the models one start year.")


def test_age_and_build_year_with_a_start_year_are_silent(tmp_path_factory) -> None:
    _validate(*make_db(tmp_path_factory, alt="age_ok", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "build_year", _f(2017.0)], ["unit", "coal_chp", "age_at_model_start", _f(0.0)],
        ["model", "flexTool", "start_year", _f(2020.0)]]))
