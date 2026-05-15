"""Surface B.17 — Investment & Fixed Cost (objective contributions).

Closed-form perturbation tests for two currently-untested obj terms:

* B17-3 — process divest lifetime FC credit
          ``- v_divest_p * p_unitsize * ed_lifetime_fixed_cost_divest``
          model.py:2583-2591
* B17-4 — process divest annuity credit
          ``- v_divest_p * p_unitsize * ed_entity_annual_divest_discounted``
          model.py:2587-2593

Both share a fixture: extend ``toy_invest_3d`` with a divest-eligible
process holding ``existing=5`` units (so divestment is feasible).  Zero
inflow / zero demand pins ``v_flow=0`` so the maxToSink constraint
``v_flow + Σ v_divest ≤ existing/unitsize`` permits ``v_divest`` up to
the upper bound ``p_entity_max_units=5``.  With a *positive* divest
credit the LP maximises ``v_divest`` — yielding a closed-form Δobj per
period under the perturbation.
"""
from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


def _solve(data: FlexData):
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    assert sol.optimal, "LP did not solve to optimality"
    return pb, sol


def _enable_divest(data: FlexData, *, lfd: float, annd: float) -> FlexData:
    """Mutate ``toy_invest_3d`` to add divest-eligible process ``u`` over
    all 3 periods with ``existing=5`` units.  Sets the two divest-cost
    params to (lfd, annd) on every (e, d).  Keeps invest active (per
    the base fixture) but neutralises invest-cost so v_invest=0 stays
    optimal: ``ed_entity_annual_discounted=0`` is set by the caller via
    a separate replace if wanted; here we leave it as the fixture's 10.0
    (positive cost ⇒ no investment).
    """
    periods = ["d1", "d2", "d3"]
    pd_div = pl.DataFrame({"p": ["u"] * 3, "d": periods})
    ed_div = pd_div.rename({"p": "e"})
    # edd_divest_active: one "alive" pair per period — divest at d_div
    # is alive at every d ≥ d_div.  Simplest matching the invest fixture:
    # diagonal d_div == d (divest takes effect in same period).
    edd_div = pl.DataFrame(
        {"p": ["u"] * 3, "d_divest": periods, "d": periods})

    # Existing capacity = 5 on every period (was 0 in the fixture).
    pss_d = (data.process_source_sink
             .join(pl.DataFrame({"d": periods}), how="cross"))
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss_d.with_columns(value=pl.lit(5.0))
             .select("p", "source", "sink", "d", "value"))
    # Also widen p_flow_upper so the structural upper isn't tighter.
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        data.pss_dt.with_columns(value=pl.lit(5.0))
            .select("p", "source", "sink", "d", "t", "value"))

    lfd_param = Param(("e", "d"),
        pl.DataFrame({"e": ["u"] * 3, "d": periods, "value": [lfd] * 3}))
    annd_param = Param(("e", "d"),
        pl.DataFrame({"e": ["u"] * 3, "d": periods, "value": [annd] * 3}))

    return replace(
        data,
        pd_divest_set=pd_div,
        ed_divest_set=ed_div,
        edd_divest_active=edd_div,
        ed_lifetime_fixed_cost_divest=lfd_param,
        ed_entity_annual_divest_discounted=annd_param,
        p_flow_upper=p_flow_upper,
        p_flow_upper_existing=p_flow_upper_existing,
    )


def test_b17_3_and_4_process_divest_credits_isolated(toy_invest_3d):
    """Covers B17-3 (lifetime FC divest credit) + B17-4 (annuity divest credit).

    Two independent perturbations on the SAME divest-enabled fixture:
      • Δlfd 10→30 with annd=0 ⇒ Δobj_3 = -(30-10)*Σv_divest*unitsize
      • Δannd 10→40 with lfd=0 ⇒ Δobj_4 = -(40-10)*Σv_divest*unitsize
    With existing=5, max_units=5, zero demand ⇒ v_divest=5 each of 3 periods
    ⇒ Σ v_divest = 15, unitsize=1.
    """
    # --- B17-3: lifetime-FC perturbation (annd held at 0) -----------------
    base_data = _enable_divest(toy_invest_3d, lfd=10.0, annd=0.0)
    pert_data = _enable_divest(toy_invest_3d, lfd=30.0, annd=0.0)
    _, sol_b = _solve(base_data)
    _, sol_p = _solve(pert_data)
    vd_sum_b = float(sol_b.value("v_divest_p")["value"].sum())
    vd_sum_p = float(sol_p.value("v_divest_p")["value"].sum())
    # Sanity: v_divest pinned at upper bound under positive credit (5*3=15).
    assert vd_sum_b == pytest.approx(15.0, rel=1e-9)
    assert vd_sum_p == pytest.approx(15.0, rel=1e-9)
    # Hand-calc: Δobj = -(30-10) * 15 * 1 = -300.
    assert float(sol_p.obj) - float(sol_b.obj) == pytest.approx(-300.0, rel=1e-7)

    # --- B17-4: annuity perturbation (lfd held at 0) ----------------------
    base2 = _enable_divest(toy_invest_3d, lfd=0.0, annd=10.0)
    pert2 = _enable_divest(toy_invest_3d, lfd=0.0, annd=40.0)
    _, sol_b2 = _solve(base2)
    _, sol_p2 = _solve(pert2)
    assert float(sol_b2.value("v_divest_p")["value"].sum()) == pytest.approx(15.0, rel=1e-9)
    assert float(sol_p2.value("v_divest_p")["value"].sum()) == pytest.approx(15.0, rel=1e-9)
    # Hand-calc: Δobj = -(40-10) * 15 * 1 = -450.
    assert float(sol_p2.obj) - float(sol_b2.obj) == pytest.approx(-450.0, rel=1e-7)
