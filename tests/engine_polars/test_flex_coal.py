"""Stage-3: flexpy ``coal`` model — single-process commodity-buy
dispatch.  Parity vs flextool + four perturbations."""

from dataclasses import replace
from pathlib import Path

import pytest
import polars as pl

from polar_high import Problem, Param
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_coal"


def _scale(p: Param, factor: float) -> Param:
    return Param(p.dims, p.frame.with_columns(value=pl.col("value") * factor))


def _solve(data) -> "Solution":
    pb = Problem()
    build_flextool(pb, data)
    return pb.solve()


@pytest.fixture(scope="module")
def coal_data():
    return load_flextool(WORK)


def test_coal_parity(coal_data):
    """flexpy obj matches the parity oracle for the coal scenario.

    Γ.4: prefer ``golden_obj.json`` if present (DB-direct captured),
    fall back to flextool's recorded ``v_obj_*.parquet`` otherwise.
    """
    sol = _solve(coal_data)
    assert sol.optimal
    from _golden import assert_obj_within
    assert_obj_within(sol.obj, WORK,
                       parquet_glob="v_obj__y2020_2day_dispatch.parquet")


def test_coal_huge_capacity_closed_form(coal_data):
    """With 10× capacity, no slack is needed and dispatch is 1:1 with
    demand.  Obj reduces to a closed-form sum of commodity costs.

    Both ``p_flow_upper`` and ``p_flow_upper_existing`` need scaling
    since the maxToSink RHS now prefers ``p_flow_upper_existing`` for
    direct processes (so that ``max_invest_cum`` baked in by
    preprocessing doesn't relax the bound when ``ed_invest`` is empty).
    """
    data = replace(coal_data,
                   p_flow_upper=_scale(coal_data.p_flow_upper, 10.0),
                   p_flow_upper_existing=_scale(
                       coal_data.p_flow_upper_existing, 10.0))
    sol = _solve(data)
    inflow_sum = (data.p_inflow.frame
        .filter(pl.col("n") == "west")
        .with_columns(neg=-pl.col("value"))
        .join(data.p_step_duration.frame.rename({"value": "dur"}),  on=["d","t"])
        .join(data.p_rp_cost_weight.frame.rename({"value": "rpcw"}), on=["d","t"])
        .join(data.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(data.p_period_share.frame.rename({"value": "psh"}), on="d")
        .with_columns(weighted = pl.col("neg") * pl.col("dur")
                                   * pl.col("rpcw") * pl.col("infl")
                                   / pl.col("psh"))
        ["weighted"].sum())
    expected = 20.0 * 2.5 * inflow_sum   # price × slope × Σ demand·op_factor
    assert sol.optimal
    assert abs(sol.obj - expected) / max(1.0, expected) < 1e-6


def test_coal_zero_price_dispatch_at_max(coal_data):
    """Free fuel → no incentive to keep slack positive when capacity
    is sufficient.  v_flow should hit upper bound at every demand-binding
    timestep."""
    data = replace(coal_data, p_commodity_price=_scale(coal_data.p_commodity_price, 0.0))
    sol = _solve(data)
    assert sol.optimal
    # at peak demand timesteps (t1-t3, where demand=589/537/506 > 500*v),
    # v_flow must be at its upper bound (=1.0)
    flow = sol.value("v_flow").sort("d","t")
    assert flow["value"][0] == pytest.approx(1.0, abs=1e-9)


def test_coal_doubled_price_keeps_dispatch_unchanged(coal_data):
    """At the default scale, slack penalty (900 €/MWh-elec) >> commodity
    cost per MWh-elec (price × slope = 50, 100, ...) so the LP still
    prefers fuel.  Doubling the price should not change v_flow."""
    sol_a = _solve(coal_data)
    data_b = replace(coal_data,
                     p_commodity_price=_scale(coal_data.p_commodity_price, 2.0))
    sol_b = _solve(data_b)
    flow_a = sol_a.value("v_flow")["value"].to_numpy()
    flow_b = sol_b.value("v_flow")["value"].to_numpy()
    assert (abs(flow_a - flow_b) < 1e-9).all()
