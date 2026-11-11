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


# ---------------------------------------------------------------------------
# Item 73: profile.profile.  An unnamed 1-D map keyed by periods is a period map; keys that are periods and time
# steps of the model are an error; a period map that misses periods of the scenario's solves is warned about once
# (the profile is 0 there).  The scenario's only period is p2020; p2025 is a period of another scenario.

PROFILE_BOTH = ("profile 'wind_profile', parameter 'profile' (alternative 'c73_both'): the map keys match both "
                "periods and time steps of the model; set the map's index name to 'period' or 'time' "
                "(keys 'p2020', 't0001').")


def test_profile_map_with_period_and_time_step_keys_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "c73_both",
                 [["profile", "wind_profile", "profile", _map([["t0001", 0.5], ["p2020", 0.3]])]])
    assert msg == SHAPE + PROFILE_BOTH


def test_labelled_time_profile_map_is_silent(tmp_path_factory) -> None:
    """A map labelled ``time`` is a time series: neither the period-map warning nor the mixed-keys error."""
    _silent(tmp_path_factory, "c73_lab", [
        ["profile", "wind_profile", "profile", _map([["t0001", 0.5], ["t0002", 0.3]], "time")]])


@pytest.mark.parametrize("index_name", [None, "period"])
def test_profile_period_map_without_the_scenario_period_warns(tmp_path_factory, index_name) -> None:
    alt = f"c73_miss_{index_name}"
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN, values=[
        ["profile", "wind_profile", "profile", _map([["p2025", 0.3]], index_name)]])
    assert _check(url, scen) == (
        f"Warning: profile 'wind_profile', parameter 'profile' (alternative '{alt}') has no value for period(s) "
        "p2020: the profile is 0 in those periods. Add the periods to the map, or give the profile a constant "
        "value.\n")


@pytest.mark.parametrize("index_name", [None, "period"])
def test_profile_period_map_covering_the_scenario_periods_is_silent(tmp_path_factory, index_name) -> None:
    _silent(tmp_path_factory, f"c73_ok_{index_name}", [
        ["profile", "wind_profile", "profile", _map([["p2020", 0.3], ["p2025", 0.4]], index_name)]])


# O11: only the solves that run count.  s_extra (period p2025, timeset 2day) is defined in the override
# alternative; it runs only when model.solves lists it.
EXTRA_SOLVE = [["solve", "s_extra", None]]
EXTRA_SOLVE_VALUES = [
    ["solve", "s_extra", "period_timeset", _map([["p2025", "2day"]], "period")],
    ["solve", "s_extra", "realized_periods", b64({"value_type": "str", "data": ["p2025"]}, "array")],
    ["profile", "wind_profile", "profile", _map([["p2020", 0.3]])]]


def test_profile_period_map_ignores_a_solve_that_does_not_run(tmp_path_factory) -> None:
    """s_extra is in no model.solves: its period p2025 is not a period the profile needs."""
    _silent(tmp_path_factory, "c73_unused", EXTRA_SOLVE_VALUES, entities=EXTRA_SOLVE)


def test_profile_period_map_warns_for_a_solve_that_runs(tmp_path_factory) -> None:
    """model.solves lists s_extra after y2020_2day_dispatch: the profile has no value for its period p2025."""
    alt = "c73_used"
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN, entities=EXTRA_SOLVE, values=[
        *EXTRA_SOLVE_VALUES,
        ["model", "flexTool", "solves",
         b64({"value_type": "str", "data": ["y2020_2day_dispatch", "s_extra"]}, "array")]])
    assert _check(url, scen) == (
        f"Warning: profile 'wind_profile', parameter 'profile' (alternative '{alt}') has no value for period(s) "
        "p2025: the profile is 0 in those periods. Add the periods to the map, or give the profile a constant "
        "value.\n")


def test_profile_time_map_is_silent(tmp_path_factory) -> None:
    _silent(tmp_path_factory, "c73_time", [
        ["profile", "wind_profile", "profile", _map([["t0001", 0.5], ["t0002", 0.3]])]])


# ---------------------------------------------------------------------------
# Item 223: a 2-D profile map whose outer level has no index name is a period x time map when its outer keys are
# periods (item 73's rule).  Outer keys that are time steps, both periods and time steps, or neither stop the run;
# a period x time map that misses a period of the solves that run gets the item-73 warning.


def _map2(outer: dict[str, dict[str, float]], index_name: str | None = None, inner_name: str | None = None) -> list:
    """A 2-D map ``{outer: {inner: v}}``; ``None`` names: Spine's silent ``x``."""
    def inner(values: dict[str, float]) -> dict:
        m = {"type": "map", "index_type": "str", "data": [[k, v] for k, v in values.items()]}
        if inner_name is not None:
            m["index_name"] = inner_name
        return m
    value = {"index_type": "str", "data": [[k, inner(v)] for k, v in outer.items()]}
    if index_name is not None:
        value["index_name"] = index_name
    return b64(value, "map")


CURVE_2D = {"t0001": 0.2, "t0002": 0.4, "t0003": 0.6}


def test_2d_profile_with_time_step_outer_keys_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "c223_time",
                 [["profile", "wind_profile", "profile", _map2({"t0001": CURVE_2D})]])
    assert msg == SHAPE + (
        "profile 'wind_profile', parameter 'profile' (alternative 'c223_time'): the outer keys of the two-level map "
        "are time steps ('t0001'). A two-level profile map is period x time: key the outer level by period (index "
        "name 'period') and the inner level by time step, or use a one-level time map.")


def test_2d_profile_with_period_and_time_step_outer_keys_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "c223_both",
                 [["profile", "wind_profile", "profile", _map2({"p2020": CURVE_2D, "t0001": CURVE_2D})]])
    assert msg == SHAPE + (
        "profile 'wind_profile', parameter 'profile' (alternative 'c223_both'): the map keys match both periods and "
        "time steps of the model; set the map's index name to 'period' or 'time' (keys 'p2020', 't0001').")


def test_2d_profile_with_outer_keys_that_are_no_period_is_an_error(tmp_path_factory) -> None:
    msg = _error(tmp_path_factory, "c223_foo",
                 [["profile", "wind_profile", "profile", _map2({"foo": CURVE_2D})]])
    assert msg == PREFIX + (
        "profile 'wind_profile', parameter 'profile' (alternative 'c223_foo'): outer map key 'foo' is not a period "
        "(a two-level profile map is period x time).") + ESCAPE


def test_2d_profile_covering_the_scenario_period_is_silent(tmp_path_factory) -> None:
    _silent(tmp_path_factory, "c223_ok", [["profile", "wind_profile", "profile", _map2({"p2020": CURVE_2D})]])


def test_2d_profile_missing_a_period_of_a_solve_that_runs_warns(tmp_path_factory) -> None:
    """model.solves runs y2020_2day_dispatch (p2020) and s_extra (p2025); the map has p2020 only."""
    alt = "c223_miss"
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN, entities=EXTRA_SOLVE, values=[
        *EXTRA_SOLVE_VALUES[:2],
        ["profile", "wind_profile", "profile", _map2({"p2020": CURVE_2D})],
        ["model", "flexTool", "solves",
         b64({"value_type": "str", "data": ["y2020_2day_dispatch", "s_extra"]}, "array")]])
    assert _check(url, scen) == (
        f"Warning: profile 'wind_profile', parameter 'profile' (alternative '{alt}') has no value for period(s) "
        "p2025: the profile is 0 in those periods. Add the periods to the map, or give the profile a constant "
        "value.\n")


# The solve: wind_plant (chain init, west, coal, wind) with existing 100 MW and a *fixed* profile.  The profile
# {p2020: {t0001: 0.2, t0002: 0.4, t0003: 0.6}} gives the flows 20 / 40 / 60 MW (100 x the profile) in t0001..t0003
# and 0 at the steps the map does not name; the unnamed map and the map labelled period / time build the same LP.

WIND_CHAIN = [*PLANT_CHAIN, "wind"]


def _wind_run(tmp_path_factory, alt: str, profile: list):
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=WIND_CHAIN, values=[
        ["unit", "wind_plant", "existing", b64(100.0, "float")],
        ["unit__node__profile", ["wind_plant", "west", "wind_profile"], "profile_method", b64("fixed", "str")],
        ["profile", "wind_profile", "profile", profile]])
    return list(run(tmp_path_factory, url, scen).values())[-1]


# The three spellings of one period x time map: outer level unnamed (Spine's ``x``) and inner unnamed, outer
# labelled ``period`` and inner unnamed (the reader names that inner level ``x``), both labelled (the reference).
SHAPES_2D = {
    "unnamed": lambda outer: _map2(outer),
    "period_outer": lambda outer: _map2(outer, "period"),
    "labelled": lambda outer: _map2(outer, "period", "time"),
}


@pytest.fixture(scope="module")
def wind_2d(tmp_path_factory):
    return {name: _wind_run(tmp_path_factory, f"c223_run_{name}", shape({"p2020": CURVE_2D}))
            for name, shape in SHAPES_2D.items()}


@pytest.mark.parametrize("shape", ["unnamed", "period_outer"])
def test_unnamed_2d_profile_fixes_the_flows(wind_2d, shape) -> None:
    from tests.engine._unit_db import flow

    flows = flow(wind_2d[shape], "wind_plant", "wind_plant", "west").to_list()
    assert flows[:3] == pytest.approx([20.0, 40.0, 60.0], abs=1e-6)
    assert max(abs(f) for f in flows[3:]) == pytest.approx(0.0, abs=1e-6)


def _assert_same_lp(step, ref) -> None:
    for field in ("p_profile_value", "p_flow_upper", "p_flow_lower"):
        a, b = getattr(step.flex_data, field, None), getattr(ref.flex_data, field, None)
        if a is None and b is None:
            continue
        assert utf8(a.frame).sort(a.frame.columns[:-1]).equals(utf8(b.frame).sort(b.frame.columns[:-1])), field
    assert float(step.solution.obj) == pytest.approx(float(ref.solution.obj), rel=1e-12)
    a, b = utf8(step.solution.value("v_flow")), utf8(ref.solution.value("v_flow"))
    assert a.sort(a.columns[:-1]).equals(b.sort(b.columns[:-1]))


@pytest.mark.parametrize("shape", ["unnamed", "period_outer"])
def test_unnamed_2d_profile_equals_the_labelled_map(wind_2d, shape) -> None:
    _assert_same_lp(wind_2d[shape], wind_2d["labelled"])


def test_2d_profile_with_period_inner_keys_is_an_error(tmp_path_factory) -> None:
    """An unnamed inner level keyed by periods would be read as time steps (and give 0)."""
    msg = _error(tmp_path_factory, "c223_inner",
                 [["profile", "wind_profile", "profile", _map2({"p2020": {"p2020": 0.5}}, "period")]])
    assert msg == SHAPE + (
        "profile 'wind_profile', parameter 'profile' (alternative 'c223_inner'): the inner keys of the two-level "
        "map are periods ('p2020'), not time steps. A two-level profile map is period x time: key the outer level "
        "by period and the inner level by time step.")


def _hand_flow(t: str) -> float:
    """100 MW x the profile: 20 / 40 / 60 MW at t0001..t0003, 0 at the steps the map does not name."""
    return 100.0 * CURVE_2D.get(t, 0.0)


def _assert_hand_flows(step, unit: str, sink: str) -> set[tuple[str, str]]:
    from tests.engine._unit_db import unitsize

    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == unit) & (pl.col("source") == unit) & (pl.col("sink") == sink)))
    assert rows.height > 0
    size = unitsize(step, unit)
    for d, t, v in zip(rows["d"], rows["t"], rows["value"]):
        assert v * size == pytest.approx(_hand_flow(t), abs=1e-6), (d, t)
    return {(d, t) for d, t in zip(rows["d"], rows["t"])}


# Rolling: y2020_2day_dispatch rolled 24 h / 12 h (rolls start at t0001, t0013, ...); each roll's flows follow
# the map at its own steps, so the first roll has 20 / 40 / 60 MW at t0001..t0003 and every other step is 0.
ROLLING = [
    ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")],
]


def _wind_rolling(tmp_path_factory, alt: str, profile: list) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=WIND_CHAIN, values=[
        *ROLLING,
        ["unit", "wind_plant", "existing", b64(100.0, "float")],
        ["unit__node__profile", ["wind_plant", "west", "wind_profile"], "profile_method", b64("fixed", "str")],
        ["profile", "wind_profile", "profile", profile]])
    return run(tmp_path_factory, url, scen)


@pytest.fixture(scope="module")
def wind_2d_rolling(tmp_path_factory):
    return {name: _wind_rolling(tmp_path_factory, f"c223_roll_{name}", shape({"p2020": CURVE_2D}))
            for name, shape in SHAPES_2D.items()}


@pytest.mark.parametrize("shape", ["unnamed", "period_outer"])
def test_unnamed_2d_profile_rolling(wind_2d_rolling, shape) -> None:
    steps, ref = wind_2d_rolling[shape], wind_2d_rolling["labelled"]
    assert len(steps) == len(ref) > 1
    seen: set[tuple[str, str]] = set()
    for step, ref_step in zip(steps.values(), ref.values()):
        seen |= _assert_hand_flows(step, "wind_plant", "west")
        _assert_same_lp(step, ref_step)
    assert {("p2020", t) for t in CURVE_2D} <= seen


# Stochastic: 2_day_stochastic_dispatch of stochastics.json (anchor period1, branch periods period1_<branch>).
# The map is keyed by the anchor; every branch period inherits its curve, so wind_plant (100 MW, fixed) gives
# 20 / 40 / 60 MW at t0001..t0003 in each branch and 0 at the other steps.
STOCHASTICS_JSON = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "stochastics.json"
STOCH_BRANCH_PERIODS = {f"period1_{b}" for b in ("realized", "upper", "lower", "mid")}


def _stoch_chain() -> list[str]:
    import json

    rows = [r for r in json.loads(STOCHASTICS_JSON.read_text())["scenario_alternatives"]
            if r[0] == "2_day_stochastic_dispatch"]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


@pytest.fixture(scope="module")
def wind_2d_stoch(tmp_path_factory):
    out = {}
    for name, shape in SHAPES_2D.items():
        url, scen = make_db(tmp_path_factory, alt=f"c223_stoch_{name}", base_chain=_stoch_chain(),
                            fixture_json=STOCHASTICS_JSON, values=[
                                ["unit", "wind_plant", "existing", b64(100.0, "float")],
                                ["unit__node__profile", ["wind_plant", "demand_node", "wind"], "profile_method",
                                 b64("fixed", "str")],
                                ["profile", "wind", "profile", shape({"period1": CURVE_2D})]])
        out[name] = last_stochastic(run(tmp_path_factory, url, scen))
    return out


@pytest.mark.parametrize("shape", ["unnamed", "period_outer"])
def test_unnamed_2d_profile_stochastic(wind_2d_stoch, shape) -> None:
    seen = _assert_hand_flows(wind_2d_stoch[shape], "wind_plant", "demand_node")
    assert {d for d, _t in seen} == STOCH_BRANCH_PERIODS
    for d in STOCH_BRANCH_PERIODS:
        assert {(d, t) for t in CURVE_2D} <= seen, d
    _assert_same_lp(wind_2d_stoch[shape], wind_2d_stoch["labelled"])
