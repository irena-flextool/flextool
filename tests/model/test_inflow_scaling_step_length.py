"""``scale_to_annual_and_peak_flow`` on time steps of any length and on solves with ``new_stepduration`` (#201).

The method maps the profile affinely, ``new = (peak/peak_old)(1 + c) old − peak c``, with ``c`` chosen so the
annualised sum meets ``annual_flow``; the annual sum of a constant peak is ``peak × (time steps in the period) /
(period share of the year)`` (docs/how_to.md "How to set the demand in a node").  It used to be ``peak × 8760``,
right only on 1 h steps.  ``peak_inflow`` and ``inflow`` are energy per time step of the timeline; a solve with
``new_stepduration`` scales each original time step as the solve without it does and sums them into its steps.

Databases are ``tests/fixtures/tests.json`` + one override alternative on ``init, west, wind, battery``: the 2day
solve (period p2020) uses the first 48 steps of the 72-step timeline; west's profile there sums to −29 103 with its
extreme −872 at t0043 (also the extreme of all 72 steps).  For 2 h steps the period has 96 h, so it is 96/8760 of a
year and the scaled inflow must sum to ``annual_flow × 96/8760`` with the extreme step at ``peak_inflow``.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, run, utf8

pytestmark = pytest.mark.solver

BATTERY = ["init", "west", "wind", "battery"]
TIMESTEPS72 = [f"t{i:04d}" for i in range(1, 73)]
PEAK = -1100.0
ANNUAL = 4.0e6
PEAK_VALUES = [["node", "west", "inflow_method", b64("scale_to_annual_and_peak_flow", "str")],
               ["node", "west", "annual_flow", b64(ANNUAL, "float")],
               ["node", "west", "peak_inflow", b64(PEAK, "float")]]


def _tmap(pairs: list, index_name: str = "time") -> list:
    return b64({"index_type": "str", "index_name": index_name, "rank": 1, "data": [list(p) for p in pairs]}, "map")


def _pmap(values: dict, index_name: str = "period") -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}, "map")


def _timeline(durations: list[float]) -> list:
    return ["timeline", "y2020", "timestep_duration", _tmap(list(zip(TIMESTEPS72, durations)), "timestep")]


def _new_step(hours: float) -> list:
    return ["solve", "y2020_2day_dispatch", "new_stepduration", b64(hours, "float")]


def _inflow(tmp_path_factory, alt: str, values: list) -> dict[tuple[str, str], float]:
    """West's ``p_inflow`` on the solve's (period, time) steps."""
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=BATTERY)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    dt = {tuple(r) for r in utf8(fd.dt).select("d", "t").iter_rows()}
    rows = utf8(fd.p_inflow.frame).filter(pl.col("n") == "west").select("d", "t", "value").rows()
    return {(d, t): v for d, t, v in rows if (d, t) in dt}


@pytest.mark.parametrize("durations, hours", [
    ([1.0] * 72, 48.0), ([2.0] * 72, 96.0), ([0.5] * 72, 24.0), ([1.0, 2.0] * 36, 72.0),
], ids=["1h", "2h", "0.5h", "1h2h"])
def test_peak_and_annual_flow_on_any_step_length(tmp_path_factory, durations, hours) -> None:
    """The 48 steps of the 2day solve sum to annual_flow × (period hours)/8760 and the profile's extreme step t0043
    is at peak_inflow; no step goes beyond it.  Before (2 h): the sum missed (−4e6·96/8760 = −43 836 expected)."""
    tag = str(hours).replace(".", "p")
    inflow = _inflow(tmp_path_factory, f"pkl_{tag}", PEAK_VALUES + [_timeline(durations)])
    assert len(inflow) == 48
    assert sum(inflow.values()) == pytest.approx(-ANNUAL * hours / 8760.0, rel=1e-9)
    assert inflow[("p2020", "t0043")] == pytest.approx(PEAK, rel=1e-12)
    assert min(inflow.values()) >= PEAK * (1.0 + 1e-12)


@pytest.mark.parametrize("base, step", [(1.0, 2.0), (2.0, 4.0)], ids=["1h-by-2", "2h-by-4"])
def test_coarse_step_is_the_sum_of_its_original_steps(tmp_path_factory, base, step) -> None:
    """With new_stepduration (two original steps per new step) every new step t0001, t0003, ... carries the sum
    of its two original steps as the solve without new_stepduration scales them, and the total meets annual_flow
    × 48·base/8760.  Before: the peak was matched by the combined step, so it was halved per original step."""
    tag = str(base).replace(".", "p")
    tl = [_timeline([base] * 72)]
    fine = _inflow(tmp_path_factory, f"pkc_fine_{tag}", PEAK_VALUES + tl)
    coarse = _inflow(tmp_path_factory, f"pkc_coarse_{tag}", PEAK_VALUES + tl + [_new_step(step)])
    labels = TIMESTEPS72[0:48:2]
    assert sorted(t for _d, t in coarse) == labels
    for k, t in enumerate(labels):
        pair = fine[("p2020", TIMESTEPS72[2 * k])] + fine[("p2020", TIMESTEPS72[2 * k + 1])]
        assert coarse[("p2020", t)] == pytest.approx(pair, rel=1e-12), t
    assert sum(coarse.values()) == pytest.approx(-ANNUAL * 48 * base / 8760.0, rel=1e-9)


def test_two_periods_on_their_own_steps(tmp_path_factory) -> None:
    """Two periods on two timesets of one timeline (p2020 = t0001-t0024, p2025 = t0025-t0048, 1 h): each period's
    inflow sums to its own annual_flow × 24/8760 (3e6 → −8219.18, 5e6 → −13 698.63).  Before: the sum the scaling
    divides by ran over the steps of both periods, so neither period met its annual_flow."""
    values = [
        ["timeset", "ta", "timeline", b64("y2020", "str")],
        ["timeset", "tb", "timeline", b64("y2020", "str")],
        ["timeset", "ta", "timeset_duration", _pmap({"t0001": 24.0}, "time")],
        ["timeset", "tb", "timeset_duration", _pmap({"t0025": 24.0}, "time")],
        ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "ta", "p2025": "tb"})],
        ["solve", "y2020_2day_dispatch", "realized_periods",
         b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")],
        ["node", "west", "inflow_method", b64("scale_to_annual_and_peak_flow", "str")],
        ["node", "west", "annual_flow", _pmap({"p2020": 3.0e6, "p2025": 5.0e6})],
        ["node", "west", "peak_inflow", b64(PEAK, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="pk_two_periods", values=values, base_chain=BATTERY,
                        entities=[["timeset", "ta", None], ["timeset", "tb", None]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    dt = {tuple(r) for r in utf8(fd.dt).select("d", "t").iter_rows()}
    rows = utf8(fd.p_inflow.frame).filter(pl.col("n") == "west").select("d", "t", "value").rows()
    per_period: dict[str, list[float]] = {}
    for d, t, v in rows:
        if (d, t) in dt:
            per_period.setdefault(d, []).append(v)
    assert {d: len(v) for d, v in per_period.items()} == {"p2020": 24, "p2025": 24}
    assert sum(per_period["p2020"]) == pytest.approx(-3.0e6 * 24 / 8760.0, rel=1e-9)
    assert sum(per_period["p2025"]) == pytest.approx(-5.0e6 * 24 / 8760.0, rel=1e-9)
