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
from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import PLANT_CHAIN, b64, last_stochastic, make_db, run, utf8

PREFIX = "Map keys that are not periods of any solve in the database:\n  "
ESCAPE = "\nIf a key is a period of data you keep for other models, add it to model.periods_available."
SHAPE = "Period maps that the model cannot read:\n  "


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
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_lab'): map key 'y2020' is not a period. "
        "Did you mean 'p2020'?") + ESCAPE


def test_mistyped_existing_period_silent_index(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_x", [["unit", "coal_plant", "existing", _map([["y2020", 500.0]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_x'): map key 'y2020' is not a period. "
        "Did you mean 'p2020'?") + ESCAPE


def test_only_the_mistyped_key_is_named(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_extra", [
        ["unit", "coal_plant", "startup_cost", _map([["p2020", 10.0], ["p2030x", 5.0]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'startup_cost' (alternative 'b46_extra'): map key 'p2030x' is not a period. "
        "Did you mean 'p2030'?") + ESCAPE


def test_silent_time_capable_map_keyed_by_no_period(tmp_path_factory) -> None:
    # availability takes a period or a time map; p2050 is no period of the database and does not look like a
    # time step (t0001..): a period map with a mistyped key.  Suggestion: the scenario's p2020 (by spelling).
    msg = _error(tmp_path_factory, "b46_p2050", [["unit", "coal_plant", "availability", _map([["p2050", 0.5]])]])
    assert msg == PREFIX + (
        "unit 'coal_plant', parameter 'availability' (alternative 'b46_p2050'): map key 'p2050' is not a period. "
        "Did you mean 'p2020'?") + ESCAPE


def test_time_step_key_on_a_period_only_parameter(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_ts", [
        ["unit", "coal_plant", "startup_cost", _map([["t0001", 10.0], ["t0002", 10.0]])]])
    assert msg == SHAPE + (
        "unit 'coal_plant', parameter 'startup_cost' (alternative 'b46_ts'): map key 't0001' is a time step, but "
        "startup_cost cannot vary over time. Use a constant or a map keyed by period (p2020: ..., p2030: ...).")


def test_numeric_keys(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "b46_num", [
        ["unit", "coal_plant", "startup_cost", _map([[2020.0, 10.0]], index_type="float")]])
    assert msg == SHAPE + (
        "unit 'coal_plant', parameter 'startup_cost' (alternative 'b46_num'): map keys are numbers ('2020.0'); "
        "periods are names such as 'p2020'. Did you mean 'p2020'?")


def test_numeric_existing_keys_stop_the_run_without_a_polars_error(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b46_num_ex", base_chain=PLANT_CHAIN,
                        values=[["unit", "coal_plant", "existing", _map([[2020.0, 500.0]], index_type="float")]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert str(err.value) == SHAPE + (
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_num_ex'): map keys are numbers ('2020.0'); "
        "periods are names such as 'p2020'. Did you mean 'p2020'?")


def test_existing_time_map_stops_the_run(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b46_ex_t", base_chain=PLANT_CHAIN,
                        values=[["unit", "coal_plant", "existing", _map([["t0001", 500.0], ["t0002", 500.0]])]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    assert str(err.value) == SHAPE + (
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_ex_t'): map key 't0001' is a time step, but "
        "existing cannot vary over time. Use a constant or a map keyed by period (p2020: ..., p2030: ...).")


def test_existing_as_a_time_series_is_an_error(tmp_path_factory) -> None:
    series = b64({"index": {"start": "2020-01-01T00:00:00", "resolution": "1h"}, "data": [500.0, 500.0]},
                 "time_series")
    msg = _error(tmp_path_factory, "b46_ex_ts", [["unit", "coal_plant", "existing", series]])
    assert msg == SHAPE + (
        "unit 'coal_plant', parameter 'existing' (alternative 'b46_ex_ts') is a time series, but existing cannot "
        "vary over time. Use a constant or a map keyed by period (p2020: ..., p2030: ...).")


@pytest.mark.parametrize("index_name", ["period", None])
def test_period_named_like_a_time_step_is_a_period(tmp_path_factory, index_name) -> None:
    """#76: a period whose name is also a time step name ('t0001', in model.periods_available) keys a period map."""
    _silent(tmp_path_factory, f"b46_p76_{index_name}", [
        ["model", "flexTool", "periods_available", b64({"value_type": "str", "data": ["t0001"]}, "array")],
        ["unit", "coal_plant", "existing", _map([["p2020", 500.0], ["t0001", 400.0]], index_name)]])


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


# ---------------------------------------------------------------------------
# Review fix pass (b2 WP-2): stochastic branch periods, periods of other scenarios in the period-only reader,
# Rule 5b timelines, NaN steps, solve period lists and rep-period databases.

STOCH_JSON = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "stoch_two_period.json"
STOCH_CHAIN = ["init", "base", "stoch"]  # stoch_2p: p2035 (branches high / low) and p2040 (continuation)


def _startup(steps: dict, unit: str) -> dict[str, float]:
    frame = utf8(last_stochastic(steps).flex_data.p_startup_cost.frame)
    return {d: v for p, d, v in frame.select("p", "d", "value").rows() if p == unit}


def test_branch_period_keys_are_periods(tmp_path_factory) -> None:
    """A branch period's own value wins over its anchor's (``_param_shapes._with_branch_periods``), so a period
    map keyed by ``p2035_high`` (a branch of p2035) or ``p2040_low`` (the continuation into p2040) passes the
    check and is read: p2035_high's authored 0 is dropped by the startup-cost zero filter, p2035_low takes the
    anchor's 7, p2040_low its own 4, p2040_high the anchor's 9."""
    url, scen = make_db(tmp_path_factory, alt="b2fp_branch", base_chain=STOCH_CHAIN, fixture_json=STOCH_JSON, values=[
        ["unit", "thermal", "startup_method", b64("linear", "str")],
        ["unit", "thermal", "min_load", b64(0.2, "float")],
        ["unit", "thermal", "startup_cost",
         _map([["p2035", 7.0], ["p2035_high", 0.0], ["p2040", 9.0], ["p2040_low", 4.0]], "period")]])
    assert _check(url, scen) == ""
    got = _startup(run(tmp_path_factory, url, scen), "thermal")
    assert {d: got[d] for d in ("p2035_low", "p2040_high", "p2040_low")} == {
        "p2035_low": 7.0, "p2040_high": 9.0, "p2040_low": 4.0}
    assert "p2035_high" not in got


def test_mistyped_branch_period_key(tmp_path_factory) -> None:
    url, scen = make_db(tmp_path_factory, alt="b2fp_branch_typo", base_chain=STOCH_CHAIN, fixture_json=STOCH_JSON,
                        values=[["unit", "thermal", "startup_cost", _map([["p2035", 7.0], ["p2035_hgih", 1.0]])]])
    with pytest.raises(FlexToolConfigError) as err:
        _check(url, scen)
    assert str(err.value) == PREFIX + (
        "unit 'thermal', parameter 'startup_cost' (alternative 'b2fp_branch_typo'): map key 'p2035_hgih' is not a "
        "period. Did you mean one of 'p2035_high', 'p2035_low'?") + ESCAPE


@pytest.mark.parametrize("extra", [
    pytest.param([], id="other_scenario"),
    pytest.param([["model", "flexTool", "periods_available", b64({"value_type": "str", "data": ["p2099"]}, "array")]],
                 id="periods_available")])
def test_valid_inactive_period_is_not_spread_over_every_period(tmp_path_factory, extra) -> None:
    """A valid key that is no period of this scenario (p2030: a period of another scenario's solve; p2099: only in
    ``model.periods_available``) passes the input check and is not spread over the scenario's periods: coal_plant
    has one startup cost in p2020, its own 10.  (A production load has no axis Enums, so the period-only reader's
    inner join with the solve's periods drops the row; under axis Enums the failed-cast guard drops it, see
    ``tests/engine/test_broadcast_null_cast_fix.py``.)"""
    other = "p2099" if extra else "p2030"
    url, scen = make_db(tmp_path_factory, alt=f"b2fp_inactive_{other}", base_chain=PLANT_CHAIN, values=[
        *extra, ["unit", "coal_plant", "startup_cost", _map([["p2020", 10.0], [other, 5.0]], "period")]])
    assert _check(url, scen) == ""
    frame = utf8(list(run(tmp_path_factory, url, scen).values())[-1].flex_data.p_startup_cost.frame)
    assert frame.filter(pl.col("p") == "coal_plant").select("d", "value").rows() == [("p2020", 10.0)]


@pytest.mark.parametrize("parameter", ["invest_periods", "realized_invest_periods"])
def test_period_named_only_by_a_solve_period_list_is_fine(tmp_path_factory, parameter) -> None:
    _silent(tmp_path_factory, f"b2fp_{parameter}", [
        ["solve", "y2020_2day_dispatch", parameter, b64({"value_type": "str", "data": ["p2020", "p2045"]}, "array")],
        ["unit", "coal_plant", "existing", _map([["p2020", 500.0], ["p2045", 300.0]], "period")]])


def test_nan_step_duration_is_an_error(tmp_path_factory) -> None:
    steps = [[f"t{i:04d}", float("nan") if i == 5 else 1.0] for i in range(1, 73)]
    msg = _error(tmp_path_factory, "b8_nan", [["timeline", "y2020", "timestep_duration", _map(steps)]])
    assert msg == (
        "timeline 'y2020' (timeset '2day' of solve 'y2020_2day_dispatch') timestep_duration = nan (alternative "
        "'b8_nan', at t0005): a time step must last more than 0 hours (a step of 0 h or less weighs its flows, "
        "costs and balances by 0 or less).")


def test_rule_5b_solve_timeline_is_checked(tmp_path) -> None:
    """Solve s1 authors ``period_timeset`` (ts1 on tl1); s2 has none and takes ts2, the only timeset with a block
    (Rule 5b), on tl2.  tl2 has no durations: an error naming ts2 and s2 (before, only tl1 was checked)."""
    from flextool.input_derivation import _validators as v

    url = f"sqlite:///{tmp_path / 'rule5b.sqlite'}"
    with api.DatabaseMapping(url, create=True) as db:
        api.import_data(
            db,
            entity_classes=[(c,) for c in ("model", "solve", "timeset", "timeline")],
            parameter_definitions=[("model", "solves"), ("model", "periods_available"), ("solve", "period_timeset"),
                                   ("solve", "realized_periods"), ("solve", "invest_periods"),
                                   ("solve", "contains_solves"), ("timeset", "timeline"),
                                   ("timeset", "timeset_duration"), ("timeline", "timestep_duration")],
            entities=[("model", "m"), ("solve", "s1"), ("solve", "s2"), ("timeset", "ts1"), ("timeset", "ts2"),
                      ("timeline", "tl1"), ("timeline", "tl2")],
            parameter_values=[
                ("model", "m", "solves", api.Array(["s1", "s2"])),
                ("solve", "s1", "period_timeset", api.Map(["p1"], ["ts1"])),
                ("solve", "s1", "realized_periods", api.Array(["p1"])),
                ("solve", "s2", "realized_periods", api.Array(["p2"])),
                ("timeset", "ts1", "timeline", "tl1"),
                ("timeset", "ts2", "timeline", "tl2"),
                ("timeset", "ts2", "timeset_duration", api.Map(["t1"], [2.0])),
                ("timeline", "tl1", "timestep_duration", api.Map(["t1", "t2"], [1.0, 1.0]))],
            alternatives=[("Base",)])
        db.commit_session("rule 5b")
    with api.DatabaseMapping(url) as db:
        with pytest.raises(FlexToolConfigError) as err:
            v.validate_timeline_timestep_duration(db)
    assert str(err.value) == (
        "timeline 'timestep_duration' is not set for: tl2 (timeset 'ts2' of solve 's2').  Every timeline a solve "
        "uses needs a Map(timestep -> duration_in_hours); without it all time-weighted quantities collapse to zero.")
