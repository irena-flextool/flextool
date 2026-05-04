"""Tier-6 perturbation: ``p_co2_price`` is the per-(group, period, time)
multiplier on the CO2 obj term — see ``audit/objective_audit.md`` §4.2
(eff buy with slope).

work_coal_co2_price has a single source-side eff process (coal_plant)
that pulls flow from a CO2-priced commodity node.  Doubling
``p_co2_price`` doubles the CO2 obj coefficient on every (p, source,
sink, d, t) tuple in ``flow_from_co2_priced``; because demand is
forced (single generator, no alternative supply but slack at
penalty 900 €/MWh which is dominated by the coal cost mix) the LP
optimum's v_flow doesn't move and the obj delta equals the baseline
CO2 term.

A failure here narrows the bug to: a missing ``p_co2_price`` factor
on the §4.2 obj term, or a missed factor in the joined chain
(unitsize, slope, co2_content, op_factor).

flextool counterpart:
``flextool/tests/perturbation/test_perturb_co2_price_scales_co2_term.py``.
"""

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.perturbation._harness import (
    scale_param,
    solve_full,
    assert_obj_changed_by,
)


WORK = Path(__file__).resolve().parents[1] / "data" / "work_coal_co2_price"


@pytest.fixture(scope="module")
def co2_data():
    return load_flextool(WORK)


def _co2_term(d, sol) -> float:
    """Closed-form value of obj §4.2 (CO2 eff buy) on the baseline LP."""
    flow = sol.value("v_flow").rename({"value": "v_flow"})
    df = (
        d.flow_from_co2_priced
        .join(flow, on=["p", "source", "sink"], how="inner")
        .join(d.p_unitsize.frame.rename({"value": "us"}), on="p")
        .join(d.p_slope.frame.rename({"value": "slope"}), on=["p", "d", "t"])
        .join(d.p_co2_content.frame.rename({"value": "cc"}), on="c")
        .join(d.p_co2_price.frame.rename({"value": "cprice"}),
              on=["g", "d", "t"])
        .join(d.p_step_duration.frame.rename({"value": "dur"}), on=["d", "t"])
        .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}),
              on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
    )
    df = df.with_columns(
        contrib=pl.col("v_flow") * pl.col("us") * pl.col("slope")
                * pl.col("cc") * pl.col("cprice") * pl.col("dur")
                * pl.col("rpcw") * pl.col("infl") / pl.col("psh")
    )
    return float(df["contrib"].sum())


@pytest.mark.perturbation
def test_perturb_co2_price_scales_co2_term(co2_data):
    if co2_data.p_co2_price is None:
        pytest.skip("p_co2_price not present in fixture")
    if co2_data.flow_from_co2_priced is None or co2_data.flow_from_co2_priced.height == 0:
        pytest.skip("flow_from_co2_priced empty in fixture")

    factor = 2.0

    pb_base, sol_base = solve_full(co2_data)
    base_obj = float(sol_base.obj)
    term_base = _co2_term(co2_data, sol_base)
    assert term_base > 0, (
        f"baseline co2 component is {term_base!r}; perturbation test needs "
        f"a non-zero co2 obj term — fixture choice issue")

    perturbed = scale_param(co2_data, "p_co2_price", factor)
    pb_pert, sol_pert = solve_full(perturbed)
    perturbed_obj = float(sol_pert.obj)

    # Sanity: doubling co2_price must not move v_flow at LP optimum
    # (work_coal_co2_price has only coal_plant, with slack as the only
    # alternative; the existing LP solution remains optimal).
    fb = sol_base.value("v_flow").sort("p", "source", "sink", "d", "t")
    fp = sol_pert.value("v_flow").sort("p", "source", "sink", "d", "t")
    assert (fb["value"].to_numpy()
            == pytest.approx(fp["value"].to_numpy(), abs=1e-7))

    expected_delta = (factor - 1.0) * term_base
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
