"""Tier-6 perturbation: commodity-buy-eff obj term — see
``audit/objective_audit.md`` §2.2.

The §2.2 term is::

    + Σ_{(p, source, sink) ∈ process_source_sink_eff, (d,t) ∈ dt}
          v_flow * p_unitsize * pdtProcess_slope
              × pdtCommodity_price × p_step_duration × p_timestep_weight
              × p_inflation_op / p_period_share
              × pdt_branch_weight
              × (sink_coef / source_coef if process_unit)

Direct ``p_unitsize`` perturbation does not yield a clean closed-form
prediction because ``p_unitsize`` simultaneously multiplies the cost
coefficient and the nodeBalance coefficient on the same v_flow — so
the LP rebalances and the obj is approximately invariant.  Per the
proposal's fall-back guidance (Phase-1 instructions, test #1), we use
the equivalent perturbation that catches the same wired-partial
multiplicative chain: scale ``p_commodity_price`` × 2.0.

Doubling commodity price in work_coal does not move v_flow (slack
penalty 900 €/MWh still dominates), so:

    expected_delta = (factor - 1.0) * commodity_term_baseline

where ``commodity_term_baseline`` is the §2.2 sum evaluated on the
baseline v_flow.  A failure here narrows the bug to: a missing
``p_unitsize`` / ``p_slope`` / ``p_commodity_price`` / ``op_factor``
factor or the ladder filter (``c not in commodity_with_ladder``) on
the §2.2 term.
"""

import polars as pl
import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars._param_shapes import promote_param_to_dt

from tests.engine_polars.perturbation._harness import (
    scale_param,
    solve_full,
    assert_obj_changed_by,
)


SCENARIO = "coal"


@pytest.fixture(scope="module")
def work(scenario_workdir):
    return scenario_workdir(SCENARIO)

@pytest.fixture(scope="module")
def coal_data(work):
    return load_flextool(work)


def _commodity_buy_eff_term(d, sol) -> float:
    """Closed-form value of obj §2.2 (commodity buy, eff) on the given
    baseline solution.  Uses the .mod's factor list."""
    flow = sol.value("v_flow").rename({"value": "v_flow"})
    # join to flow_from_commodity_eff to filter to eff buys
    pss_eff = d.flow_from_commodity_eff   # (p, source, sink, c)
    df = (
        pss_eff
        .join(flow, left_on=["p", "source", "sink"],
                    right_on=["p", "source", "sink"], how="inner")
        .join(d.p_unitsize.frame.rename({"value": "us"}), on="p")
        .join(d.p_slope.frame.rename({"value": "slope"}), on=["p", "d", "t"])
        .join(promote_param_to_dt(d.p_commodity_price, d.dt)
                  .rename({"value": "cprice"}).collect(),
              on=["c", "d", "t"])
        .join(d.p_step_duration.frame.rename({"value": "dur"}), on=["d", "t"])
        .join(d.p_timestep_weight.frame.rename({"value": "rpcw"}), on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
    )
    df = df.with_columns(
        contrib=pl.col("v_flow") * pl.col("us") * pl.col("slope")
                * pl.col("cprice") * pl.col("dur") * pl.col("rpcw")
                * pl.col("infl") / pl.col("psh")
    )
    return float(df["contrib"].sum())


@pytest.mark.perturbation
def test_perturb_p_unitsize_scales_commodity_buy_eff(coal_data):
    factor = 2.0  # applied to p_commodity_price (see module docstring)

    pb_base, sol_base = solve_full(coal_data)
    base_obj = float(sol_base.obj)

    # Closed-form value of the §2.2 term at the baseline LP solution.
    term_base = _commodity_buy_eff_term(coal_data, sol_base)

    perturbed = scale_param(coal_data, "p_commodity_price", factor)
    pb_pert, sol_pert = solve_full(perturbed)
    perturbed_obj = float(sol_pert.obj)

    # Sanity: doubling commodity price (which is dwarfed by the 900
    # €/MWh slack penalty) must not move v_flow at LP optimum.
    fb = sol_base.value("v_flow").sort("p", "source", "sink", "d", "t")
    fp = sol_pert.value("v_flow").sort("p", "source", "sink", "d", "t")
    assert (fb["value"].to_numpy()
            == pytest.approx(fp["value"].to_numpy(), abs=1e-7))

    expected_delta = (factor - 1.0) * term_base
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
