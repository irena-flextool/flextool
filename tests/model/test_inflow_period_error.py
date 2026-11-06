"""A period-keyed ``inflow`` is a configuration error (#68).

``inflow`` is a constant, a time series or a stochastic 3-D map; it cannot be period-based (docs/reference.md,
"Timesteps and periods").  No reader picks up a period-keyed value: before the check it silently became inflow 0 on
every solve type, so the demand disappeared.  ``validate_node_inflow_not_period`` now stops the run in
``input_derivation.run``, before any solve, for:
- a 1-D map with ``index_name`` ``period``;
- any 2-D map (period x time is the usual intent);
- a 1-D map with Spine's silent ``x`` index whose keys are all period names and none a timestep name.

Databases are built from ``tests/fixtures/tests.json`` (never a checked-in ``.sqlite``); ``west`` is the demand
node, ``p2020`` the period, ``t0001..t0072`` the timesteps.
"""
from __future__ import annotations

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import PLANT_CHAIN, b64, make_db, run

T72 = [f"t{i:04d}" for i in range(1, 73)]


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _map(data: list, index_name: str | None) -> list:
    m = {"type": "map", "index_type": "str", "data": data}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


PERIOD_MAP = _map([["p2020", -500.0]], "period")
PERIOD_TIME_MAP = _map([["p2020", {"type": "map", "index_type": "str", "index_name": "time",
                                   "data": [[t, -500.0] for t in T72]}]], "period")
SILENT_PERIOD_MAP = _map([["p2020", -500.0]], None)
SILENT_TIME_MAP = _map([[t, -500.0] for t in T72], None)


def _db(tmp_path_factory, alt: str, inflow: list) -> tuple[str, str]:
    return make_db(tmp_path_factory, alt=alt, values=[["node", "west", "inflow", inflow]], base_chain=PLANT_CHAIN)


@pytest.mark.parametrize("alt, inflow, reason", [
    ("c8_inflow_period", PERIOD_MAP, "index_name 'period'"),
    ("c8_inflow_period_time", PERIOD_TIME_MAP, "a 2-D map"),
    ("c8_inflow_silent_period", SILENT_PERIOD_MAP, "all map keys are period names"),
])
def test_period_keyed_inflow_is_rejected(tmp_path_factory, alt, inflow, reason) -> None:
    url, scenario = _db(tmp_path_factory, alt, inflow)
    with pytest.raises(FlexToolConfigError, match="west.*keyed by period") as err:
        run(tmp_path_factory, url, scenario)
    msg = str(err.value)
    assert reason in msg and alt in msg and "p2020" in msg and "annual_flow" in msg, msg


def _validate(url: str, scenario: str) -> None:
    from flextool.input_derivation._validators import validate_node_inflow_not_period
    from flextool.spinedb_backend import SpineDBBackend

    with SpineDBBackend(url, scenario) as backend:
        validate_node_inflow_not_period(backend)
        # the validator evicts every parse it makes (Track A memory contract)
        for pv in backend.parameter_value_items("node", "inflow"):
            assert pv.mapped_item._parsed_value is None


@pytest.mark.parametrize("alt, inflow", [
    ("c8_inflow_silent_time", SILENT_TIME_MAP),
    ("c8_inflow_time", _map([[t, -500.0] for t in T72], "time")),
    ("c8_inflow_scalar", b64(-500.0, "float")),
])
def test_time_and_constant_inflow_pass(tmp_path_factory, alt, inflow) -> None:
    _validate(*_db(tmp_path_factory, alt, inflow))


def test_validator_rejects_silent_period_map_directly(tmp_path_factory) -> None:
    with pytest.raises(FlexToolConfigError, match="west.*keyed by period"):
        _validate(*_db(tmp_path_factory, "c8_inflow_direct", SILENT_PERIOD_MAP))
