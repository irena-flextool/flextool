"""Values that make no sense or are almost surely typing mistakes (#170).

``validate_basic_sanity`` (``input_derivation.run``, before the parameter read) stops the run on values the model
cannot use sensibly and warns about values that are likely entered in the wrong unit.  Before the check:
``efficiency 0`` turned coal_plant off, ``efficiency -0.4`` paid it to run, ``capacity_max_coeff -1`` and a minimum
above its maximum ended "infeasible on presolve" without a hint, and ``availability 95`` made more capacity than
there is, all without a word.  (``existing`` as a time series or keyed by time steps or numbers is
``validate_period_keys``': ``test_period_key_checks.py``.)

Databases: ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal`` (coal_plant: input
coal_market, output west, efficiency 0.4, existing 500; model period p2020, timeline t0001..t0072) or on
``init, west, coal, wind, network`` (connection west_east).  Messages are the expected text, written by hand.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
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
        ["unit", "coal_plant", "lifetime", _f(30.0)],
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


def test_zero_lifetime_of_an_investing_unit_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b170_life", [
        ["unit", "coal_plant", "invest_method", b64("invest_no_limit", "str")],
        ["unit", "coal_plant", "lifetime", _f(0.0)]])
    assert ("unit 'coal_plant' lifetime = 0 (alternative 'b170_life'): lifetime must be above 0 when the unit can "
            "invest or retire; with 0 the investment is charged for no years. Give the lifetime in years.") in msg


def test_zero_lifetime_without_investment_is_silent(tmp_path_factory) -> None:
    _validate(*make_db(tmp_path_factory, alt="b170_life_off", base_chain=PLANT_CHAIN,
                       values=[["unit", "coal_plant", "lifetime", _f(0.0)]]))


@pytest.mark.parametrize("alt, values, chain, expected", [
    ("b170_avail_pct", [["unit", "coal_plant", "availability", _f(95.0)]], PLANT_CHAIN,
     "Warning: unit 'coal_plant' availability = 95 (alternative 'b170_avail_pct'): values above 1.5 look like "
     "percentages; availability is a share (0.95, not 95)."),
    ("b170_conn_gain", [["connection", "west_east", "efficiency", _f(1.1)]], NETWORK_CHAIN,
     "Warning: connection 'west_east' efficiency = 1.1 (alternative 'b170_conn_gain'): a connection efficiency "
     "above 1 creates energy on the way."),
    ("b170_vus", [["unit", "coal_plant", "virtual_unitsize", _f(0.0)]], PLANT_CHAIN,
     "Warning: unit 'coal_plant' virtual_unitsize = 0 (alternative 'b170_vus'): virtual_unitsize must be above 0; "
     "the value is ignored (the entity is one unit of its capacity)."),
])
def test_likely_unit_mistakes_warn(tmp_path_factory, capsys, alt, values, chain, expected) -> None:
    _validate(*make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain))
    assert expected in capsys.readouterr().out


def test_negative_existing_is_silent(tmp_path_factory, capsys) -> None:
    """Anti-energy units are supported (the anti_energy_plant example)."""
    _validate(*make_db(tmp_path_factory, alt="b170_anti", base_chain=PLANT_CHAIN,
                       values=[["unit", "coal_plant", "existing", _f(-50.0)]]))
    assert "Warning" not in capsys.readouterr().out


def test_profile_entered_as_percent_warns(tmp_path_factory, capsys) -> None:
    """``warn_profiles_as_percent`` runs after the parameter read (on the reader's profile frames)."""
    run(tmp_path_factory, *make_db(tmp_path_factory, alt="b170_wind_pct", base_chain=["init", "west", "wind"],
                                   values=[["profile", "wind_profile", "profile",
                                            _map([["t0001", 0.3], ["t0002", 35.0]])]]))
    assert ("Warning: profile 'wind_profile' = 35 (alternative 'b170_wind_pct', at t0002) limits a unit flow "
            "(upper_limit or fixed): values above 1.5 look like percentages; the profile is a share of the capacity "
            "(0.35, not 35).") in capsys.readouterr().out


def test_shares_slightly_above_one_are_silent(tmp_path_factory, capsys) -> None:
    """Measured or reanalysis capacity factors reach a little above 1 (rounding, over-performance)."""
    run(tmp_path_factory, *make_db(tmp_path_factory, alt="b170_cf", base_chain=["init", "west", "wind"], values=[
        ["profile", "wind_profile", "profile", _map([["t0001", 0.3], ["t0002", 1.0235]])],
        ["unit", "coal_plant", "availability", _f(1.2)]]))
    out = capsys.readouterr().out
    assert "look like percentages" not in out and "wind_profile" not in out, out


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
