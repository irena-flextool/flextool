"""Map keys that are not periods (items 46 / 65) and timelines without durations (audit bug 8).

``validate_period_keys`` (``input_derivation.run``, before the parameter read) stops the run on a period-axis map
key that is no period of the database.  Before the check ``existing {y2020: 500}`` gave coal_plant no capacity,
``startup_cost {y2020: ...}`` lost the cost and an extra key ``p2030x`` was dropped, all without a word.
``validate_timeline_timestep_duration`` checks only the timelines a solve uses.

Databases: ``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal`` (coal_plant: existing
500; model period p2020, solve y2020_2day_dispatch, timeset 2day on timeline y2020, t0001..t0072).  The periods of
the whole database include p2025, p2030 and p2035 (the ``dispatch_fullYear_roll`` solve of another scenario).
Messages are the expected text, written by hand.
"""
from __future__ import annotations

import contextlib
import io

import pytest
import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import PLANT_CHAIN, b64, make_db, run

PREFIX = "Map keys that are not periods of the model:\n  "
ESCAPE = "\nIf a key is a period of data you keep for other models, add it to model.periods_available."


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _map(data: list, index_name: str | None = None, index_type: str = "str") -> list:
    value = {"index_type": index_type, "data": data}
    if index_name is not None:
        value["index_name"] = index_name
    return b64(value, "map")


def _check(url: str, scenario: str) -> str:
    """Run the two checks as ``input_derivation.run`` does (whole-database facts, then the scenario filter);
    return what they print."""
    from flextool.input_derivation import _validators as v
    from flextool.spinedb_backend import SpineDBBackend

    with api.DatabaseMapping(url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        whole = v.database_time_facts(db)
        api.filters.scenario_filter.scenario_filter_from_dict(
            db, api.filters.scenario_filter.scenario_filter_config(scenario))
        backend = SpineDBBackend.__new__(SpineDBBackend)
        backend._db = db
        backend._api = api
        backend._scenario_name = scenario
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v.validate_timeline_timestep_duration(db)
            v.validate_period_keys(backend, whole)
    return out.getvalue()


def _error(tmp_path_factory, alt: str, values: list, entities: list | None = None) -> str:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN, entities=entities)
    with pytest.raises(FlexToolConfigError) as err:
        _check(url, scen)
    return str(err.value)


def _silent(tmp_path_factory, alt: str, values: list, entities: list | None = None) -> None:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN, entities=entities)
    assert _check(url, scen) == ""


def test_mistyped_existing_period_stops_the_run(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b46_lab", base_chain=PLANT_CHAIN,
                        values=[["unit", "coal_plant", "existing", _map([["y2020", 500.0]], "period")]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert str(err.value) == PREFIX + (
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_lab'): map key 'y2020' is not a period of the "
        "model. Did you mean 'p2020'?") + ESCAPE


def test_mistyped_existing_period_silent_index(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_x", [["unit", "coal_plant", "existing", _map([["y2020", 500.0]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_x'): map key 'y2020' is not a period of the "
        "model. Did you mean 'p2020'?") + ESCAPE


def test_only_the_mistyped_key_is_named(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_extra", [
        ["unit", "coal_plant", "startup_cost", _map([["p2020", 10.0], ["p2030x", 5.0]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'startup_cost' (alternative 'b46_extra'): map key 'p2030x' is not a period "
        "of the model. Did you mean 'p2030'?") + ESCAPE


def test_time_step_key_on_a_period_only_parameter(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_ts", [
        ["unit", "coal_plant", "startup_cost", _map([["t0001", 10.0], ["t0002", 10.0]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'startup_cost' (alternative 'b46_ts'): map key 't0001' is a time step, but "
        "startup_cost cannot vary over time. Use a constant or a map keyed by period (p2020: ..., p2030: ...).") \
        + ESCAPE


def test_numeric_keys(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_num", [
        ["unit", "coal_plant", "startup_cost", _map([[2020.0, 10.0]], index_type="float")]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'startup_cost' (alternative 'b46_num'): map keys are numbers ('2020.0'); "
        "periods are names such as 'p2020'. Did you mean 'p2020'?") + ESCAPE


def test_numeric_existing_keys_stop_the_run_without_a_polars_error(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b46_num_ex", base_chain=PLANT_CHAIN,
                        values=[["unit", "coal_plant", "existing", _map([[2020.0, 500.0]], index_type="float")]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert "is a map keyed by numbers (2020): periods are names such as 'p2020'" in str(err.value)


def test_existing_time_map_stops_the_run(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b46_ex_t", base_chain=PLANT_CHAIN,
                        values=[["unit", "coal_plant", "existing", _map([["t0001", 500.0], ["t0002", 500.0]])]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert "existing (alternative 'b46_ex_t') is keyed by time step 't0001': existing cannot vary over time" \
        in str(err.value)


def test_period_of_another_scenario_is_fine(tmp_path_factory) -> None:
    # p2030 is a period of the dispatch_fullYear_roll solve only (another scenario).
    _silent(tmp_path_factory, "b46_other", [
        ["unit", "coal_plant", "existing", _map([["p2020", 500.0], ["p2030", 100.0]], "period")]])


def test_period_in_periods_available_is_fine(tmp_path_factory) -> None:
    _silent(tmp_path_factory, "b46_avail", [
        ["model", "flexTool", "periods_available", b64({"value_type": "str", "data": ["p2099"]}, "array")],
        ["unit", "coal_plant", "existing", _map([["p2020", 500.0], ["p2099", 100.0]])]])


def test_legit_period_map_and_time_series_are_silent(tmp_path_factory) -> None:
    _silent(tmp_path_factory, "b46_ok", [
        ["unit", "coal_plant", "startup_cost", _map([["p2020", 10.0]])],
        ["unit", "coal_plant", "availability", _map([["t0001", 0.9], ["t0002", 0.8]])],
        # keyed by constraint name, not a period map
        ["unit__outputNode", ["coal_plant", "west"], "constraint_flow_coeff", _map([["hydro_divide", 1.0]])]])


def test_spare_timeline_without_durations_runs(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b8_spare", base_chain=PLANT_CHAIN, values=[],
                        entities=[["timeline", "spare_tl", None]])
    steps = run(tmp_path_factory, url, scen)
    assert list(steps.values())[-1].solution.obj == pytest.approx(1_144_037_750.0, rel=1e-9)


def test_used_timeline_without_durations_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b8_used", [
        ["timeset", "spare_ts", "timeline", b64("spare_tl", "str")],
        ["solve", "y2020_2day_dispatch", "period_timeset", _map([["p2020", "spare_ts"]], "period")]],
        entities=[["timeline", "spare_tl", None], ["timeset", "spare_ts", None]])
    assert msg == (
        "timeline 'timestep_duration' is not set for: spare_tl (timeset 'spare_ts' of solve "
        "'y2020_2day_dispatch').  Every timeline a solve uses needs a Map(timestep -> duration_in_hours); "
        "without it all time-weighted quantities collapse to zero.")


def test_zero_hour_step_is_an_error(tmp_path_factory) -> None:
    steps = [[f"t{i:04d}", 0.0 if i == 3 else 1.0] for i in range(1, 73)]
    msg = _error(tmp_path_factory, "b8_zero", [["timeline", "y2020", "timestep_duration", _map(steps)]])
    assert msg == (
        "timeline 'y2020' (timeset '2day' of solve 'y2020_2day_dispatch') timestep_duration = 0 (alternative "
        "'b8_zero', at t0003): a time step must last more than 0 hours (a step of 0 h or less weighs its flows, "
        "costs and balances by 0 or less).")
