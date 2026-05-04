"""Tier-6 perturbation: ``p_penalty_up`` is the per-(node, period,
time) coefficient on the upward state-slack obj term — see
``audit/objective_audit.md`` §1.1.

The §1.1 term is::

    + Σ_{(n, d, t) ∈ nodeBalance_dt}
          vq_state_up[n, d, t]
              * p_penalty_up[n, d, t]
              * p_node_capacity_for_scaling[n, d]   (when present)
              * p_step_duration * p_rp_cost_weight
              * p_inflation_op / p_period_share

work_coal has dispatch generators *and* slack (the slack penalty
900 €/MWh is dwarfed by the coal cost mix, so coal serves load and
slack is only used where the LP cannot route flow).  Doubling
``p_penalty_up`` doubles the slack obj coefficient on every (n, d,
t) tuple; because the slack value is pinned by ``nodeBalance_eq``
(no alternative), the LP optimum's vq_state_up doesn't move and the
obj delta equals the baseline §1.1 term.

A failure here narrows the bug to: a missing ``p_penalty_up``
factor on §1.1, or a misapplied factor in the joined chain
(node_capacity_for_scaling, op_factor).

flextool counterpart:
``flextool/tests/perturbation/test_perturb_penalty_up_scales_slack.py``
(uses pure-slack ``base`` scenario where the entire obj is slack;
flexpy's mapping table assigns ``work_coal``, so we compute the
§1.1 term closed-form from the baseline solution rather than
asserting ``expected_delta == base_obj``).
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


WORK = Path(__file__).resolve().parents[1] / "data" / "work_coal"


@pytest.fixture(scope="module")
def coal_data():
    return load_flextool(WORK)


def _slack_up_term(d, sol) -> float:
    """Closed-form value of obj §1.1 (state-slack up) on the baseline LP."""
    slack = sol.value("vq_state_up").rename({"value": "vq_up"})
    df = (
        slack
        .join(d.p_penalty_up.frame.rename({"value": "pen"}),
              on=["n", "d", "t"])
        .join(d.p_step_duration.frame.rename({"value": "dur"}),
              on=["d", "t"])
        .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}),
              on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
    )
    if d.p_node_capacity_for_scaling is not None:
        df = df.join(
            d.p_node_capacity_for_scaling.frame.rename({"value": "ncs"}),
            on=["n", "d"],
        )
        df = df.with_columns(
            contrib=pl.col("vq_up") * pl.col("pen") * pl.col("ncs")
                    * pl.col("dur") * pl.col("rpcw")
                    * pl.col("infl") / pl.col("psh")
        )
    else:
        df = df.with_columns(
            contrib=pl.col("vq_up") * pl.col("pen")
                    * pl.col("dur") * pl.col("rpcw")
                    * pl.col("infl") / pl.col("psh")
        )
    return float(df["contrib"].sum())


@pytest.mark.perturbation
def test_perturb_penalty_up_scales_slack(coal_data):
    factor = 2.0

    pb_base, sol_base = solve_full(coal_data)
    base_obj = float(sol_base.obj)
    term_base = _slack_up_term(coal_data, sol_base)
    assert term_base > 0, (
        f"baseline slack-up component is {term_base!r}; perturbation "
        f"test needs a non-zero slack term — fixture choice issue")

    perturbed = scale_param(coal_data, "p_penalty_up", factor)
    pb_pert, sol_pert = solve_full(perturbed)
    perturbed_obj = float(sol_pert.obj)

    # Sanity: doubling penalty_up must not move vq_state_up at LP
    # optimum (slack is pinned by nodeBalance_eq).
    sb = sol_base.value("vq_state_up").sort("n", "d", "t")
    sp = sol_pert.value("vq_state_up").sort("n", "d", "t")
    assert (sb["value"].to_numpy()
            == pytest.approx(sp["value"].to_numpy(), abs=1e-7))

    expected_delta = (factor - 1.0) * term_base
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
