"""The emission-trajectory how-to examples (``docs/how_to/emissions.md``, "How to compare emission trajectories").

Scenarios ``co2_without_measures``, ``co2_existing_measures`` and ``co2_additional_measures`` of
``tests/fixtures/tests.json`` (built from JSON, never a checked-in ``.sqlite``).  Chain ``init, west, coal,
co2_trajectory_system, <trajectory>``: four periods p2020..p2035 on the 2-day timeset (48 h each),
``years_represented`` 5 each, a constant 300 MW demand, *coal_plant* at 50 CUR/MWh emitting 0.34 / 0.4 =
0.85 t/MWh, *clean_plant* at 80 CUR/MWh without emissions.  Hand values:

    emissions [t/a] = coal MWh in the 48 h x 0.85 x 8760 / 48
    unconstrained: coal 300 MW -> 2 233 800 t/a, so every cap below that binds (emissions = cap)
    annual cost [CUR] = 8760 / 48 x sum over the 48 h of (50 x coal + 80 x clean)
                      = 210.24 M - 35.294118 x cap  (binding),  131.4 M  (not binding)
    objective [M CUR] = sum over the periods of 5 x annual cost / 1e6
                      = 2628.0 (without) < 3004.8 (existing) < 3322.447059 (additional)

Within a period the coal hours are not unique, so only period totals are checked.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import run, unitsize, utf8

pytestmark = pytest.mark.solver

PERIODS = ("p2020", "p2025", "p2030", "p2035")
ANNUAL = 8760.0 / 48.0
T_PER_MWH = 0.34 / 0.4
CAPS = {
    "co2_without_measures": (3.0e6, 3.0e6, 3.0e6, 3.0e6),
    "co2_existing_measures": (2.0e6, 1.8e6, 1.6e6, 1.4e6),
    "co2_additional_measures": (2.0e6, 1.5e6, 1.0e6, 0.5e6),
}
UNCONSTRAINED_T = 300.0 * T_PER_MWH * 8760.0  # 2 233 800 t/a
OBJECTIVE = {
    "co2_without_measures": 2628.0,
    "co2_existing_measures": 3004.8,
    "co2_additional_measures": 3322.447059,
}
_STEPS: dict[str, object] = {}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module", autouse=True)
def _free_steps():
    yield
    _STEPS.clear()


def _step(tmp_path_factory, test_db_url: str, scenario: str):
    if scenario not in _STEPS:
        steps = run(tmp_path_factory, test_db_url, scenario)
        assert len(steps) == 1, list(steps)
        _STEPS[scenario] = next(iter(steps.values()))
    return _STEPS[scenario]


def _mwh_per_period(step, unit: str, source: str) -> dict[str, float]:
    """MWh of *unit*'s output to west in the 48 h of each period."""
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == unit) & (pl.col("source") == source) & (pl.col("sink") == "west"))
            .group_by("d").agg(pl.col("value").sum()))
    out = {d: v * unitsize(step, unit) for d, v in zip(rows["d"], rows["value"])}
    assert sorted(out) == list(PERIODS), out
    return out


@pytest.mark.parametrize("scenario", list(CAPS))
def test_emissions_follow_the_trajectory(tmp_path_factory, test_db_url, scenario):
    """Each period emits its cap where the cap is below the unconstrained 2 233 800 t/a, else the unconstrained
    amount."""
    step = _step(tmp_path_factory, test_db_url, scenario)
    coal = _mwh_per_period(step, "coal_plant", "coal_market")
    for d, cap in zip(PERIODS, CAPS[scenario]):
        emitted = coal[d] * T_PER_MWH * ANNUAL
        assert emitted == pytest.approx(min(cap, UNCONSTRAINED_T), rel=1e-7), (d, emitted, cap)


def test_tighter_trajectory_costs_more(tmp_path_factory, test_db_url):
    """The operating cost of each trajectory, from its flows, is the hand value, and a tighter trajectory costs
    more."""
    total = {}
    for scenario in CAPS:
        step = _step(tmp_path_factory, test_db_url, scenario)
        coal = _mwh_per_period(step, "coal_plant", "coal_market")
        clean = _mwh_per_period(step, "clean_plant", "clean_plant")
        total[scenario] = sum(5.0 * ANNUAL * (50.0 * coal[d] + 80.0 * clean[d]) for d in PERIODS) / 1e6
        assert total[scenario] == pytest.approx(OBJECTIVE[scenario], rel=1e-7), scenario
    assert total["co2_without_measures"] < total["co2_existing_measures"] < total["co2_additional_measures"]
