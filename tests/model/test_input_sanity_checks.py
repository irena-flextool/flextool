"""Values that make no sense or are almost surely typing mistakes (#170).

``validate_basic_sanity`` (``input_derivation.run``, before the parameter read) stops the run on values the model
cannot use sensibly and warns about values that are likely entered in the wrong unit.  Before the check:
``efficiency 0`` turned coal_plant off, ``efficiency -0.4`` paid it to run, ``capacity_max_coeff -1`` and a minimum
above its maximum ended "infeasible on presolve" without a hint, ``existing`` as a time map or keyed by 2020 gave no
capacity or crashed, and ``availability 1.5`` made more capacity than there is, all without a word.

Databases: ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal`` (coal_plant: input
coal_market, output west, efficiency 0.4, existing 500; model period p2020, timeline t0001..t0072) or on
``init, west, coal, wind, network`` (connection west_east).  Messages are the expected text, written by hand.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import PLANT_CHAIN, TIMESTEPS, b64, make_db, run

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
        validate_basic_sanity(backend._db, set(TIMESTEPS))


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
     "unit 'coal_plant' availability = -0.1 (alternative 'b170_avail_neg'): availability must lie in [0, 1]."),
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
        ["unit", "coal_plant", "lifetime", _f(30.0)],
        ["unit", "coal_plant", "invest_min_period", _map([["p2020", 200.0]])],
        ["unit", "coal_plant", "invest_max_period", _f(100.0)]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert "unit 'coal_plant' invest_min_period = 200 is above invest_max_period = 100 in period p2020" \
        in str(err.value), str(err.value)


def test_min_above_max_is_silent_when_not_compared(tmp_path_factory) -> None:
    """The method does not use the period pair, and the total maps share no period."""
    _validate(*make_db(tmp_path_factory, alt="b170_unused", base_chain=PLANT_CHAIN, values=[
        ["unit", "coal_plant", "invest_method", b64("invest_total", "str")],
        ["unit", "coal_plant", "invest_min_period", _f(200.0)],
        ["unit", "coal_plant", "invest_max_period", _f(100.0)],
        ["unit", "coal_plant", "cumulative_min_capacity", _map([["p2020", 900.0]])],
        ["unit", "coal_plant", "cumulative_max_capacity", _map([["p2030", 100.0]])]]))


def test_zero_lifetime_of_an_investing_unit_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_life", [
        ["unit", "coal_plant", "invest_method", b64("invest_no_limit", "str")],
        ["unit", "coal_plant", "lifetime", _f(0.0)]])
    assert ("unit 'coal_plant' lifetime = 0 (alternative 'b170_life'): lifetime must be above 0 when the unit can "
            "invest or retire; with 0 the investment is charged for no years. Give the lifetime in years.") in msg


def test_zero_lifetime_without_investment_is_silent(tmp_path_factory) -> None:
    _validate(*make_db(tmp_path_factory, alt="b170_life_off", base_chain=PLANT_CHAIN,
                       values=[["unit", "coal_plant", "lifetime", _f(0.0)]]))


def test_existing_keyed_by_time_steps_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_exist_t", [
        ["unit", "coal_plant", "existing", _map([["t0001", 500.0], ["t0002", 500.0]])]])
    assert ("unit 'coal_plant' existing (alternative 'b170_exist_t') is keyed by time step 't0001': existing cannot "
            "vary over time. Use a constant or a map keyed by period (p2020: 500, ...).") in msg, msg


def test_existing_as_a_time_series_is_an_error(tmp_path_factory) -> None:
    series = b64({"index": {"start": "2020-01-01T00:00:00", "resolution": "1h"}, "data": [500.0, 500.0]},
                 "time_series")
    msg = _error(tmp_path_factory, "b170_exist_ts", [["unit", "coal_plant", "existing", series]])
    assert "unit 'coal_plant' existing (alternative 'b170_exist_ts') is a time series: existing cannot vary over " \
           "time." in msg, msg


def test_existing_keyed_by_numbers_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_exist_num", [
        ["unit", "coal_plant", "existing", _map([[2020.0, 500.0]], "float")]])
    assert ("unit 'coal_plant' existing (alternative 'b170_exist_num') is a map keyed by numbers (2020): periods "
            "are names such as 'p2020'. Use the period names of the model as keys.") in msg, msg


@pytest.mark.parametrize("alt, values, chain, expected", [
    ("b170_avail_pct", [["unit", "coal_plant", "availability", _f(1.5)]], PLANT_CHAIN,
     "Warning: unit 'coal_plant' availability = 1.5 (alternative 'b170_avail_pct'): availability above 1 makes "
     "more capacity available than there is; it is a share (0.95, not 95)."),
    ("b170_conn_gain", [["connection", "west_east", "efficiency", _f(1.1)]], NETWORK_CHAIN,
     "Warning: connection 'west_east' efficiency = 1.1 (alternative 'b170_conn_gain'): a connection efficiency "
     "above 1 creates energy on the way."),
    ("b170_vus", [["unit", "coal_plant", "virtual_unitsize", _f(0.0)]], PLANT_CHAIN,
     "Warning: unit 'coal_plant' virtual_unitsize = 0 (alternative 'b170_vus'): virtual_unitsize must be above 0; "
     "the value is ignored (the entity is one unit of its capacity)."),
    ("b170_wind_pct", [["profile", "wind_profile", "profile", _map([["t0001", 0.3], ["t0002", 35.0]])]],
     ["init", "west", "wind"],
     "Warning: profile 'wind_profile' = 35 (alternative 'b170_wind_pct', at t0002) limits a unit flow (upper_limit "
     "or fixed) and is above 1: the profile is a share of the capacity (0.35, not 35)."),
])
def test_likely_unit_mistakes_warn(tmp_path_factory, capsys, alt, values, chain, expected) -> None:
    _validate(*make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain))
    assert expected in capsys.readouterr().out


def test_negative_existing_is_silent(tmp_path_factory, capsys) -> None:
    """Anti-energy units are supported (the anti_energy_plant example)."""
    _validate(*make_db(tmp_path_factory, alt="b170_anti", base_chain=PLANT_CHAIN,
                       values=[["unit", "coal_plant", "existing", _f(-50.0)]]))
    assert "Warning" not in capsys.readouterr().out
