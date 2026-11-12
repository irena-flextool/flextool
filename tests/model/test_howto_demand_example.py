"""The howto_demand example (docs/how_to/basics.md "How to set the demand in a node") serves its demand (item 207).

``flextool/schemas/canonical_databases/howto_demand.json``: node ``nodeA`` with a 3-week hourly demand profile
(504 one-hour steps, the same 3 weeks in each of the periods p1 / p2 / p3), ``coal_plant`` (``existing`` 3500 MW, active
in the ``coal`` alternative) supplying it from ``coal_market``.  Scenario ``scale_to_peak_and_annual`` scales the
profile to ``annual_flow`` 1e7 / 1.5e7 / 2e7 MWh with ``peak_inflow`` -1500 / -2200 / -3000 MW.

Before the fix ``coal_plant`` had no entity_alternative (inactive everywhere: nodeA was served by the penalty slack
only) and ``peak_inflow`` -500 / -600 / -700 MW could not reach the annual flow (warning; scaled inflow above the peak
at off-peak steps).

Hand derivation (docs formula ``inflow' = r(1+c)·inflow - peak·c``, ``r = peak / peak_old``): the profile sums to
-387182 MWh with peak -1051 MW; the annualised scaled profile is S = r · 387182 · 8760/504 and the flat profile at the
peak N = |peak| · 8760, so c = (S - A) / (N - S): p1 r = 1500/1051, S = 9604566, N = 13140000, A = 1e7 ->
c = -0.111851; p2 c = -0.176135; p3 (twice p1) c = -0.111851.  Period sums A · 504/8760 = -575342.47 / -863013.70 /
-1150684.93 MWh; every step stays a demand, the smallest demand is -703.96 / -1116.98 / -1407.92 MW.
"""
from __future__ import annotations

import contextlib
import io
import json

import polars as pl
import pytest

from flextool._resources import package_data_path
from flextool.update_flextool.initialize_database import initialize_database
from tests.engine._unit_db import flow, run

pytestmark = pytest.mark.solver

PEAK = {"p1": -1500.0, "p2": -2200.0, "p3": -3000.0}
C = {"p1": -0.111851, "p2": -0.176135, "p3": -0.111851}
PERIOD_SUM = {"p1": -575342.47, "p2": -863013.70, "p3": -1150684.93}
SMALLEST_DEMAND = {"p1": -703.96, "p2": -1116.98, "p3": -1407.92}


def _profile() -> dict[str, float]:
    data = json.loads(package_data_path("schemas/canonical_databases/howto_demand.json").read_text())
    pv = next(p for p in data["parameter_values"] if p[:3] == ["node", "nodeA", "inflow"] and p[4] == "timeseries")
    # The timeset 3weeks: 168 steps from each of t0001, t1681, t3361.
    used = {f"t{start + i:04d}" for start in (1, 1681, 3361) for i in range(168)}
    return {t: float(v) for t, v in json.loads(pv[3][0])["data"] if t in used}


@pytest.fixture(scope="module")
def solved(tmp_path_factory) -> tuple:
    """``(last step, printed text)`` of the scenario scale_to_peak_and_annual."""
    sqlite = tmp_path_factory.mktemp("howto_demand") / "howto_demand.sqlite"
    initialize_database(str(package_data_path("schemas/canonical_databases/howto_demand.json")), str(sqlite))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        steps = run(tmp_path_factory, f"sqlite:///{sqlite}", "scale_to_peak_and_annual")
    return list(steps.values())[-1], out.getvalue()


@pytest.fixture(scope="module")
def step(solved):
    return solved[0]


def _inflow(step) -> pl.DataFrame:
    f = step.flex_data.p_inflow.frame
    return (f.filter(pl.col("n").cast(pl.Utf8) == "nodeA")
            .with_columns(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).sort("d", "t"))


def test_profile_facts() -> None:
    profile = _profile()
    assert len(profile) == 504
    assert sum(profile.values()) == pytest.approx(-387182.0)
    assert min(profile.values()) == -1051.0


@pytest.mark.parametrize("d", ["p1", "p2", "p3"])
def test_scaled_inflow_hits_peak_and_annual_flow(step, d: str) -> None:
    profile = _profile()
    rows = _inflow(step).filter(pl.col("d") == d)
    assert rows.height == 504
    values = dict(zip(rows["t"], rows["value"]))
    r = PEAK[d] / -1051.0
    for t, x in profile.items():
        assert values[t] == pytest.approx(r * (1 + C[d]) * x - PEAK[d] * C[d], rel=1e-5, abs=0.02), t
    assert sum(values.values()) == pytest.approx(PERIOD_SUM[d], rel=1e-8)
    assert min(values.values()) == pytest.approx(PEAK[d], rel=1e-9)
    assert max(values.values()) == pytest.approx(SMALLEST_DEMAND[d], abs=0.01)
    assert max(values.values()) < 0.0


def test_coal_plant_serves_the_demand(step) -> None:
    """3500 MW covers the largest peak (3000 MW): coal_plant's output equals the demand in every step."""
    demand = -_inflow(step)["value"]
    supply = flow(step, "coal_plant", "coal_market", "nodeA")
    assert supply.len() == demand.len() == 3 * 504
    assert (supply - demand).abs().max() == pytest.approx(0.0, abs=1e-6)


def test_no_scaling_warning(solved) -> None:
    """Neither the peak-reach warning nor the #276 empty-node warning for nodeA is printed."""
    out = solved[1]
    assert "peak_inflow" not in out and "node 'nodeA'" not in out, out
