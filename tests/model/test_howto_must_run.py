"""The must-run how-to examples (``docs/how_to/thermal_chp.md``, "How to make a unit run all the time").

Scenarios ``must_run_*`` of ``tests/fixtures/tests.json`` (built from JSON, never a checked-in ``.sqlite``).
Base chain ``init, west, coal, wind, must_run_system``: a constant 300 MW demand at *west*; *wind_plant*
1000 MW at 1 CUR/MWh whose profile is at least 0.33 in every hour (so wind always covers the remainder);
*coal_plant* 500 MW at 20 CUR/MWh fuel / 0.4 = 50 CUR/MWh; *baseload_plant* (``must_run_profile`` only) the same;
*steam_1..3* 100 MW each, efficiency 0.4, with
``other_operational_cost`` 0, 1 and 2 CUR/MWh, so the optimum is unique per unit.  Hand values per hour, the
same in all 48 hours (the objectives are checked in ``tests/scenarios.yaml``):

    must_run_none                coal 0, wind 300
    must_run_profile             baseload_plant 0.4 x 500 = 200, coal 0, wind 100   (profile lower_limit)
    must_run_constraint_one      coal 150, wind 150                        (flow >= 150)
    must_run_fleet_output_floor  steam_1 100, steam_2/3 0, wind 200        (sum of flows >= 100)
    must_run_flowgroup           steam_1 100, steam_2/3 0, wind 200        (min_instant_flow 100)
    must_run_fleet_two_online    online 1, 1, 0; steam_1 50, steam_2 50, wind 200   (sum online >= 2, min_load 0.5)
    must_run_min_load_only       online 0, 0, 0; steam 0, wind 300         (min_load alone is not a must-run)
    must_run_heat_demand         coal_chp electricity 150, heat 300 (heat = 2 x electricity), wind 150
"""
from __future__ import annotations

import pytest

from tests.engine._unit_db import flow, online, run

pytestmark = pytest.mark.solver

STEAM = ("steam_1", "steam_2", "steam_3")
HOURS = 48
_STEPS: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module", autouse=True)
def _free_steps():
    yield
    _STEPS.clear()


def _step(tmp_path_factory, test_db_url: str, scenario: str):
    """The one solve of *scenario* (solved once per module)."""
    if scenario not in _STEPS:
        steps = run(tmp_path_factory, test_db_url, scenario)
        assert len(steps) == 1, list(steps)
        _STEPS[scenario] = next(iter(steps.values()))
    return _STEPS[scenario]


def _assert_every_hour(series, expected: float, what: str) -> None:
    values = series.to_list()
    assert len(values) == HOURS, f"{what}: {len(values)} steps"
    assert values == pytest.approx([expected] * HOURS, abs=1e-6), f"{what}: {values[:4]} ..."


@pytest.mark.parametrize(
    ("scenario", "coal", "wind"),
    [
        ("must_run_none", 0.0, 300.0),
        ("must_run_constraint_one", 150.0, 150.0),
    ],
)
def test_coal_plant_floor(tmp_path_factory, test_db_url, scenario, coal, wind):
    """The control case and the user-constraint floor: coal_plant runs at exactly its floor."""
    step = _step(tmp_path_factory, test_db_url, scenario)
    _assert_every_hour(flow(step, "coal_plant", "coal_market", "west"), coal, "coal_plant")
    _assert_every_hour(flow(step, "wind_plant", "wind_plant", "west"), wind, "wind_plant")


def test_profile_floor(tmp_path_factory, test_db_url):
    """The profile lower_limit holds baseload_plant at 0.4 x 500 MW; the equally priced coal_plant stays off."""
    step = _step(tmp_path_factory, test_db_url, "must_run_profile")
    _assert_every_hour(flow(step, "baseload_plant", "coal_market", "west"), 200.0, "baseload_plant")
    _assert_every_hour(flow(step, "coal_plant", "coal_market", "west"), 0.0, "coal_plant")
    _assert_every_hour(flow(step, "wind_plant", "wind_plant", "west"), 100.0, "wind_plant")


@pytest.mark.parametrize("scenario", ["must_run_fleet_output_floor", "must_run_flowgroup"])
def test_fleet_output_floor(tmp_path_factory, test_db_url, scenario):
    """An output floor on the fleet runs the cheapest unit at the floor: steam_1 100 MW, the others 0."""
    step = _step(tmp_path_factory, test_db_url, scenario)
    for unit, mw in zip(STEAM, (100.0, 0.0, 0.0)):
        _assert_every_hour(flow(step, unit, "coal_market", "west"), mw, unit)
    _assert_every_hour(flow(step, "coal_plant", "coal_market", "west"), 0.0, "coal_plant")
    _assert_every_hour(flow(step, "wind_plant", "wind_plant", "west"), 200.0, "wind_plant")


@pytest.mark.parametrize(
    ("scenario", "units_online", "output"),
    [
        ("must_run_fleet_two_online", (1.0, 1.0, 0.0), (50.0, 50.0, 0.0)),
        ("must_run_min_load_only", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ],
)
def test_online_count(tmp_path_factory, test_db_url, scenario, units_online, output):
    """A count on the online units keeps two whole units online at min load; min_load alone keeps none."""
    step = _step(tmp_path_factory, test_db_url, scenario)
    for unit, count, mw in zip(STEAM, units_online, output):
        # binary startup: whole units online, unitsize = existing 100 MW
        _assert_every_hour(online(step, unit, "integer") / 100.0, count, f"{unit} online")
        _assert_every_hour(flow(step, unit, "coal_market", "west"), mw, unit)
    _assert_every_hour(flow(step, "wind_plant", "wind_plant", "west"), 300.0 - sum(output), "wind_plant")


def test_output_floor_cheaper_than_online_count(tmp_path_factory, test_db_url):
    """The count is stricter than the output floor: two units at min load (5 250 CUR/h) cost more than one unit
    at the floor (5 200 CUR/h), for the same 100 MW of steam output."""
    def cost_per_hour(scenario: str) -> float:
        step = _step(tmp_path_factory, test_db_url, scenario)
        steam = [flow(step, u, "coal_market", "west").sum() / HOURS for u in STEAM]
        wind = flow(step, "wind_plant", "wind_plant", "west").sum() / HOURS
        return sum(mw * (50.0 + k) for k, mw in enumerate(steam)) + wind * 1.0

    floor = cost_per_hour("must_run_fleet_output_floor")
    count = cost_per_hour("must_run_fleet_two_online")
    assert floor == pytest.approx(5200.0, abs=1e-4)
    assert count == pytest.approx(5250.0, abs=1e-4)


def test_heat_demand_runs_chp(tmp_path_factory, test_db_url):
    """The heat demand makes coal_chp run: heat 300, electricity half of it, wind covers the rest."""
    step = _step(tmp_path_factory, test_db_url, "must_run_heat_demand")
    # coal_chp has two outputs (indirect): its output arcs run from the unit itself
    _assert_every_hour(flow(step, "coal_chp", "coal_chp", "heat"), 300.0, "coal_chp heat")
    _assert_every_hour(flow(step, "coal_chp", "coal_chp", "west"), 150.0, "coal_chp electricity")
    _assert_every_hour(flow(step, "coal_plant", "coal_market", "west"), 0.0, "coal_plant")
    _assert_every_hour(flow(step, "wind_plant", "wind_plant", "west"), 150.0, "wind_plant")


def test_profile_unit_reported_fuel_input(tmp_path_factory, test_db_url):
    """baseload_plant makes 200 MW at efficiency 0.4, so its written fuel input is 500 MW in every hour."""
    import warnings

    import pandas as pd

    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    scenario = "must_run_profile"
    work = tmp_path_factory.mktemp("must_run_profile_out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = run_chain_from_db(test_db_url, scenario, work_folder=work, keep_solutions=True)
        last = next(reversed(steps.values()))
        write_outputs(
            scenario_name=scenario, output_location=str(work), subdir=scenario,
            write_methods=["parquet"], fallback_output_location=str(work),
            raw_output_dir=str(work / "output_raw"),
            solution=last.solution, solve_name=last.solve_name,
            solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
            flex_data_provider=last.flex_data_provider,
        )
    frame = pd.read_parquet(work / "output_parquet" / scenario / "unit_inputNode_dt_ee.parquet")
    fuel = frame[str((scenario, "baseload_plant", "coal_market"))].astype(float).abs()
    assert len(fuel) == HOURS
    assert fuel.to_list() == pytest.approx([200.0 / 0.4] * HOURS, abs=1e-6)
