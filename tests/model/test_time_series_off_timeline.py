"""Time-series keys that are not time steps of any timeline (#205).

The readers join a time series with the solve's time steps, so a key that is not a time step of a timeline was
dropped without a word.  ``validate_time_series_keys`` (``input_derivation.run``, once per run) now warns about such
keys (one line per class and parameter) and stops the run when none of a value's time keys is on a timeline.  The
model is unchanged: the objectives are those measured before the check (the values are dropped as before).

Databases are ``tests/fixtures/tests.json`` + one override alternative on the chain ``init, west, coal`` (timeline
``y2020`` = t0001..t0072, 1 h steps; ``west.inflow`` hourly over t0001..t0072).  Key counts worked out by hand:
- 3 h timeline t0001, t0004, ..., t0070 (24 steps) with the hourly inflow: 72 - 24 = 48 keys are off, the first
  t0002, t0003, t0005, t0006, t0008;
- the hourly inflow plus t0073..t0080: 8 of 80 keys are off, the first t0073..t0077.
"""
from __future__ import annotations

import base64
import json

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import BASE_FIXTURE_JSON, PLANT_CHAIN, b64, make_db, run


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _west_hourly() -> list:
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    west = next(pv for pv in data["parameter_values"]
                if pv[0] == "node" and pv[1] == "west" and pv[2] == "inflow" and pv[4] == "west")
    return json.loads(base64.b64decode(west[3][0]))["data"]


HOURLY = _west_hourly()
EXTRA = HOURLY + [[f"t{i:04d}", -1e5] for i in range(73, 81)]


def _map(data: list, index_name: str | None = "time") -> list:
    m = {"index_type": "str", "data": data}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


TIMELINE_3H = [["timeline", "y2020", "timestep_duration", _map([[f"t{i:04d}", 3.0] for i in range(1, 73, 3)])],
               ["timeset", "2day", "timeset_duration", _map([["t0001", 24.0]])]]
NSD3 = ["solve", "y2020_2day_dispatch", "new_stepduration", b64(3.0, "float")]

WARN_A = ("Warning: node 'west' parameter 'inflow': 48 of 72 time keys are not time steps of any timeline (t0002, "
          "t0003, t0005, t0006, t0008 and 43 more); their values are not used. Timeline 'y2020': 24 steps, "
          "t0001 .. t0070.")
WARN_C = ("Warning: node 'west' parameter 'inflow': 8 of 80 time keys are not time steps of any timeline (t0073, "
          "t0074, t0075, t0076, t0077 and 3 more); their values are not used. Timeline 'y2020': 72 steps, "
          "t0001 .. t0072.")
INFLOW_WARNING = "Warning: node 'west' parameter 'inflow'"


def _db(tmp_path_factory, alt: str, values: list) -> tuple[str, str]:
    return make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN)


def _validate(url: str, scenario: str) -> None:
    from flextool.input_derivation._validators import validate_time_series_keys
    from flextool.spinedb_backend import SpineDBBackend

    with SpineDBBackend(url, scenario) as backend:
        validate_time_series_keys(backend)


@pytest.mark.parametrize("alt, values, warning, objective", [
    ("b205_a_hourly_on_3h", TIMELINE_3H + [["node", "west", "inflow", _map(HOURLY)]], WARN_A, 92_235_500.0),
    ("b205_c_extra_keys", [["node", "west", "inflow", _map(EXTRA)]], WARN_C, 1_144_037_750.0),
    ("b205_b_nsd_extra_keys", [NSD3, ["node", "west", "inflow", _map(EXTRA)]], WARN_C, 1_136_436_625.0),
])
def test_off_timeline_keys_warn_once(tmp_path_factory, capsys, alt, values, warning, objective) -> None:
    steps = run(tmp_path_factory, *_db(tmp_path_factory, alt, values))
    out = capsys.readouterr().out
    assert out.count(INFLOW_WARNING) == 1 and warning in out, out
    assert list(steps.values())[-1].solution.obj == pytest.approx(objective, rel=1e-9)


def test_rolling_solve_warns_once(tmp_path_factory, capsys) -> None:
    rolling = [["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
               ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
               ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")]]
    steps = run(tmp_path_factory, *_db(tmp_path_factory, "b205_rolling",
                                       rolling + [["node", "west", "inflow", _map(EXTRA)]]))
    assert len(steps) > 1
    out = capsys.readouterr().out
    assert out.count(INFLOW_WARNING) == 1 and WARN_C in out, out


def test_no_key_on_a_timeline_is_an_error(tmp_path_factory) -> None:
    wrong_case = [[k.upper(), v] for k, v in HOURLY]
    url, scen = _db(tmp_path_factory, "b205_wrong_case", [["node", "west", "inflow", _map(wrong_case)]])
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    msg = str(err.value)
    assert ("node 'west' parameter 'inflow' (alternative 'b205_wrong_case'): none of its 72 time keys (T0001, "
            "T0002, T0003 ...)") in msg and "case sensitive" in msg and "y2020: t0001 .. t0072" in msg, msg


def test_series_longer_than_the_timeset_passes(tmp_path_factory, capsys) -> None:
    # All 72 keys are steps of the timeline; the timeset uses 24 of them.
    _validate(*_db(tmp_path_factory, "b205_subset", [
        ["timeset", "2day", "timeset_duration", _map([["t0001", 24.0]])],
        ["node", "west", "inflow", _map(HOURLY)]]))
    assert "Warning:" not in capsys.readouterr().out


def test_unnamed_stochastic_map_branch_names_are_not_time_keys(tmp_path_factory, capsys) -> None:
    def level(data):
        return {"type": "map", "index_type": "str", "data": data}
    times = level([[k, v] for k, v in HOURLY])
    stoch = b64({"index_type": "str", "data": [
        [branch, level([["t0001", times], ["t0025", times]])] for branch in ("upper", "lower")]}, "map")
    _validate(*_db(tmp_path_factory, "b205_stochastic", [["node", "west", "inflow", stoch]]))
    assert "Warning:" not in capsys.readouterr().out


def test_unnamed_stochastic_map_off_time_key_warns(tmp_path_factory, capsys) -> None:
    def level(data):
        return {"type": "map", "index_type": "str", "data": data}
    times = level([[k, v] for k, v in HOURLY] + [["t0099", 1.0]])
    stoch = b64({"index_type": "str", "data": [["upper", level([["t0001", times]])]]}, "map")
    _validate(*_db(tmp_path_factory, "b205_stochastic_off", [["node", "west", "inflow", stoch]]))
    out = capsys.readouterr().out
    assert "parameter 'inflow': 1 of 73 time keys are not time steps of any timeline (t0099)" in out, out


def _datetime_steps(first_hour: int, n: int) -> list[str]:
    return [f"2050-01-{1 + h // 24:02d}T{h % 24:02d}:00:00" for h in range(first_hour, first_hour + n)]


@pytest.mark.parametrize("start, warning", [
    ("2050-01-01T00:00:00", None),
    # one hour later: the last stamp 2050-01-04T00:00:00 is past the 72-step timeline
    ("2050-01-01T01:00:00", "1 of 72 time keys are not time steps of any timeline (2050-01-04T00:00:00)"),
])
def test_time_series_value_on_a_datetime_timeline(tmp_path_factory, capsys, start, warning) -> None:
    # a second timeline (the other series of the chain stay on y2020's t0001..t0072)
    timeline = [["timeline", "y2050", "timestep_duration", _map([[s, 1.0] for s in _datetime_steps(0, 72)])]]
    series = b64({"index": {"start": start, "resolution": "1h", "ignore_year": False, "repeat": False},
                  "data": [v for _k, v in HOURLY]}, "time_series")
    _validate(*make_db(tmp_path_factory, alt=f"b205_ts_{start[11:13]}", base_chain=PLANT_CHAIN,
                       values=timeline + [["node", "west", "inflow", series]],
                       entities=[["timeline", "y2050", None]]))
    out = capsys.readouterr().out
    if warning is None:
        assert "Warning:" not in out, out
    else:
        assert f"Warning: node 'west' parameter 'inflow': {warning}" in out, out


def test_timeset_weight_on_a_start_that_is_not_a_step_warns(tmp_path_factory, capsys) -> None:
    _validate(*_db(tmp_path_factory, "b205_weights", [
        ["timeset", "2day", "timeset_weights", _map([["t0001", 0.5], ["x0001", 0.5]])]]))
    out = capsys.readouterr().out
    assert ("Warning: timeset '2day' parameter 'timeset_weights': 1 of 2 time keys are not time steps of any "
            "timeline (x0001)") in out, out


@pytest.mark.parametrize("extra, warning", [
    ([], None),
    # one key past the end: the period key p2020 is not a time key
    ([["t0073", 1.0]], "1 of 73 time keys are not time steps of any timeline (t0073)"),
])
def test_period_time_map(tmp_path_factory, capsys, extra, warning) -> None:
    hourly = [[k, 0.9] for k, _v in HOURLY]
    value = b64({"index_type": "str", "index_name": "period", "data": [
        ["p2020", {"type": "map", "index_type": "str", "index_name": "time", "data": hourly + extra}]]}, "map")
    _validate(*_db(tmp_path_factory, f"b205_pt_{len(extra)}", [["unit", "coal_plant", "availability", value]]))
    out = capsys.readouterr().out
    if warning is None:
        assert "Warning:" not in out, out
    else:
        assert f"Warning: unit 'coal_plant' parameter 'availability': {warning}" in out, out
