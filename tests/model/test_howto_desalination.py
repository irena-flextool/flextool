"""The desalination how-to examples (``docs/how_to/thermal_chp.md``, "How to model cogeneration with desalination").

Scenarios ``desal_*`` of ``tests/fixtures/tests.json`` (built from JSON, never a checked-in ``.sqlite``).  Chain
``init, west, coal, desal_system`` (+ one alternative): a constant 300 MW demand at *west*; a water demand of
3000 m³/h at *water* (penalty 50 CUR/m³); coal 20 CUR/MWh of fuel, *coal_plant* 500 MW at efficiency 0.4.
*med_desal* makes 12 m³ per MWh of *desal_heat* (3000 m³/h); *bp_chp* (efficiency 0.85) makes heat = 2 × power
(user constraint *bp_chp_ratio*).  Hand values per hour, the same in all 48 hours:

    desal_backpressure   MED 3000 <- heat 250; bp_chp power 125, fuel 375 / 0.85; coal_plant 175
    desal_extraction     bp_chp off; ext_chp (0.45 x fuel = power + 0.15 x heat) heat 250, power 300, fuel 750;
                         coal_plant 0 (20 / 0.45 = 44.4 < 50 CUR/MWh_e); power >= 0.5 x heat
    desal_ro_invest      ro_desal (250 m3/MWh_e) invested to its 2000 m3/h limit: RO 2000 (8 MW), MED 1000
                         (heat 83.33, power 41.67), coal_plant 266.33
    desal_water_storage  water is a 15 000 m3 storage, demand 12 h at 2000 then 12 h at 4000 m3/h: MED 3000 flat,
                         state 0 -> 12 000 (t0012) -> 0 (t0024), twice

The objectives are checked in ``tests/scenarios.yaml``.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import flow, run, unitsize, utf8

pytestmark = pytest.mark.solver

HOURS = 48
SCENARIOS = ("desal_backpressure", "desal_extraction", "desal_ro_invest", "desal_water_storage")
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


def _every_hour(series: pl.Series, expected: float, what: str) -> None:
    values = series.to_list()
    assert len(values) == HOURS, f"{what}: {len(values)} steps"
    assert values == pytest.approx([expected] * HOURS, abs=1e-6), f"{what}: {values[:4]} ..."


def _slack(step, var: str, node: str) -> float:
    try:
        frame = utf8(step.solution.value(var))
    except (KeyError, ValueError):
        return 0.0
    return float(frame.filter(pl.col("n") == node)["value"].abs().sum())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_water_balance_without_slack(tmp_path_factory, test_db_url, scenario):
    """The water demand is met by the desalination units, with no water or heat slack."""
    step = _step(tmp_path_factory, test_db_url, scenario)
    for node in ("water", "desal_heat", "west"):
        for var in ("vq_state_up", "vq_state_down"):
            assert _slack(step, var, node) == pytest.approx(0.0, abs=1e-6), (node, var)
    water = flow(step, "med_desal", "desal_heat", "water").sum()
    if scenario == "desal_ro_invest":
        water += flow(step, "ro_desal", "west", "water").sum()
    assert water == pytest.approx(3000.0 * HOURS, rel=1e-9)


def test_backpressure_ratio(tmp_path_factory, test_db_url):
    """bp_chp makes heat = 2 x power: 250 MW of heat for MED's 3000 m3/h, 125 MW of power."""
    step = _step(tmp_path_factory, test_db_url, "desal_backpressure")
    _every_hour(flow(step, "med_desal", "desal_heat", "water"), 3000.0, "med_desal")
    _every_hour(flow(step, "bp_chp", "bp_chp", "desal_heat"), 250.0, "bp_chp heat")
    _every_hour(flow(step, "bp_chp", "bp_chp", "west"), 125.0, "bp_chp power")
    _every_hour(flow(step, "bp_chp", "coal_market", "bp_chp"), 375.0 / 0.85, "bp_chp fuel")
    _every_hour(flow(step, "coal_plant", "coal_market", "west"), 175.0, "coal_plant")


def test_extraction_operating_region(tmp_path_factory, test_db_url):
    """ext_chp makes all of the power (cheaper than coal_plant) and the heat, inside its region: power >= 0.5 x heat
    at every step.  bp_chp is off, so its ratio constraint has no terms and is left out of the model."""
    step = _step(tmp_path_factory, test_db_url, "desal_extraction")
    heat = flow(step, "ext_chp", "ext_chp", "desal_heat")
    power = flow(step, "ext_chp", "ext_chp", "west")
    _every_hour(heat, 250.0, "ext_chp heat")
    _every_hour(power, 300.0, "ext_chp power")
    _every_hour(flow(step, "ext_chp", "coal_market", "ext_chp"), 750.0, "ext_chp fuel")
    _every_hour(flow(step, "coal_plant", "coal_market", "west"), 0.0, "coal_plant")
    assert all(p >= 0.5 * q - 1e-6 for p, q in zip(power.to_list(), heat.to_list()))
    assert "bp_chp" not in set(utf8(step.solution.value("v_flow"))["p"].to_list())


def test_reverse_osmosis_invested_to_its_limit(tmp_path_factory, test_db_url):
    """RO water (0.2 CUR/m3) is cheaper than MED water (0.857843 CUR/m3) by more than its annuity, so ro_desal is
    invested to its 2000 m3/h limit and MED makes the remaining 1000 m3/h."""
    step = _step(tmp_path_factory, test_db_url, "desal_ro_invest")
    invest = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == "ro_desal")
    assert invest.height == 1
    assert invest["value"][0] * unitsize(step, "ro_desal") == pytest.approx(2000.0, rel=1e-9)
    _every_hour(flow(step, "ro_desal", "west", "water"), 2000.0, "ro_desal")
    _every_hour(flow(step, "med_desal", "desal_heat", "water"), 1000.0, "med_desal")
    _every_hour(flow(step, "coal_plant", "coal_market", "west"), 300.0 + 8.0 - 125.0 / 3.0, "coal_plant")


def test_water_storage_lets_med_run_flat(tmp_path_factory, test_db_url):
    """With the storage, MED runs at its 3000 m3/h every hour; the state fills to 12 000 m3 in the 12 low-demand
    hours and empties in the 12 high-demand hours, twice."""
    step = _step(tmp_path_factory, test_db_url, "desal_water_storage")
    _every_hour(flow(step, "med_desal", "desal_heat", "water"), 3000.0, "med_desal")
    state = (utf8(step.solution.value("v_state")).filter(pl.col("n") == "water").sort("d", "t"))
    state_unitsize = float(utf8(step.flex_data.p_state_unitsize.frame)
                           .filter(pl.col("n") == "water")["value"][0])
    day = [1000.0 * h for h in range(1, 13)] + [12000.0 - 1000.0 * h for h in range(1, 13)]
    assert (state["value"] * state_unitsize).to_list() == pytest.approx(day * 2, abs=1e-6)
