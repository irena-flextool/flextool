"""Inflow scaling in stochastic solves: a branch period scales over its parent's complete time steps.

``steps_complete_solve`` lists the complete steps of the trunk periods only; a stochastic branch period (``p2040_high``,
``p2040_low``, ...) takes its parent's, the same mapping (``period__branch``) its ``complete_hours_in_period`` uses.
The scalings used to see no steps for a branch period and stopped with "the inflow sums to 0" (``scale_to_annual_flow``)
or "no inflow profile" (``scale_to_annual_and_peak_flow``).

Database: ``tests/fixtures/stoch_two_period_hedge.json`` (p2035 = t0001-t0003, p2040 = t0004-t0006, 1 h steps, each
period 3/8760 of a year), chains ``hedge`` (p2040 branches) and ``hedge_ws`` (both periods branch), plus an override
giving node ``elec`` the profile −100, −140, −90 | −130, −80, −160 and ``annual_flow`` 1e6.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, utf8

pytestmark = pytest.mark.solver

FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_hedge.json"
CHAINS = {"hedge": ["init", "base", "stoch", "rp"], "hedge_ws": ["init", "base", "stoch", "ws"]}
PROFILE = {"t0001": -100.0, "t0002": -140.0, "t0003": -90.0, "t0004": -130.0, "t0005": -80.0, "t0006": -160.0}
ANNUAL = 1.0e6
PEAK = -150.0


def _inflow(tmp_path_factory, alt: str, chain: str, method: str) -> dict[str, dict[str, float]]:
    """``{period: {time: p_inflow}}`` of node elec in the stochastic solve."""
    values = [["node", "elec", "inflow", b64({"index_type": "str", "index_name": "time",
                                               "data": [[t, v] for t, v in PROFILE.items()]}, "map")],
              ["node", "elec", "inflow_method", b64(method, "str")],
              ["node", "elec", "annual_flow", b64(ANNUAL, "float")],
              ["node", "elec", "peak_inflow", b64(PEAK, "float")]]
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=CHAINS[chain], fixture_json=Path(FIXTURE))
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    dt = {tuple(r) for r in utf8(fd.dt).select("d", "t").iter_rows()}
    out: dict[str, dict[str, float]] = {}
    for d, t, v in utf8(fd.p_inflow.frame).filter(pl.col("n") == "elec").select("d", "t", "value").rows():
        if (d, t) in dt:
            out.setdefault(d, {})[t] = v
    return out


def _period_steps(d: str) -> list[str]:
    return ["t0001", "t0002", "t0003"] if d.startswith("p2035") else ["t0004", "t0005", "t0006"]


@pytest.mark.parametrize("chain", list(CHAINS))
def test_annual_flow_in_branch_periods(tmp_path_factory, chain) -> None:
    """scale_to_annual_flow: every period (trunk or branch) is its profile × annual_flow·(3/8760)/|Σ profile|:
    p2035 × 1e6·3/(8760·330), p2040 × 1e6·3/(8760·370); each sums to −1e6·3/8760 = −342.466.  Before: a branch
    period stopped the run ("the inflow sums to 0 over the period's timesteps")."""
    inflow = _inflow(tmp_path_factory, f"stoch_af_{chain}", chain, "scale_to_annual_flow")
    branches = [d for d in inflow if d.endswith(("_high", "_low"))]
    assert branches
    for d, by_t in inflow.items():
        steps = _period_steps(d)
        assert sorted(by_t) == steps, d
        k = ANNUAL * 3 / (8760.0 * abs(sum(PROFILE[t] for t in steps)))
        for t in steps:
            assert by_t[t] == pytest.approx(PROFILE[t] * k, rel=1e-12), (d, t)
        assert sum(by_t.values()) == pytest.approx(-ANNUAL * 3 / 8760.0, rel=1e-12), d


@pytest.mark.parametrize("chain", list(CHAINS))
def test_peak_and_annual_flow_in_branch_periods(tmp_path_factory, chain) -> None:
    """scale_to_annual_and_peak_flow: every period sums to −1e6·3/8760 = −342.466, its own extreme step (t0002 in
    p2035, t0006 in p2040) is at peak_inflow −150 and every step stays a demand; a branch period equals its parent's
    trunk scaling (high == low).  Before: a branch period stopped the run ("no inflow profile")."""
    inflow = _inflow(tmp_path_factory, f"stoch_pk_{chain}", chain, "scale_to_annual_and_peak_flow")
    for d, by_t in inflow.items():
        assert sorted(by_t) == _period_steps(d), d
        assert sum(by_t.values()) == pytest.approx(-ANNUAL * 3 / 8760.0, rel=1e-12), d
        assert by_t["t0002" if d.startswith("p2035") else "t0006"] == pytest.approx(PEAK, rel=1e-12), d
        assert max(by_t.values()) < 0.0, d
    assert inflow["p2040_high"] == inflow["p2040_low"]
