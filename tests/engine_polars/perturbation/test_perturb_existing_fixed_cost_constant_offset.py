"""Tier-6 perturbation: existing-entity fixed-cost constant — see
``audit/objective_audit.md`` §8.1 (HIGH IMPACT, currently MISSING).

The .mod has::

    + Σ_{e ∈ entity, d ∈ period_in_use}
          p_entity_all_existing[e, d] * ed_fixed_cost[e, d]
              * p_inflation_factor_operations_yearly[d] * pd_branch_weight[d]

This is a *constant* (no decision variables), so the solver drops it
during optimisation but flextool reports it in the published
``v_obj``.  flexpy currently does **not** add this constant to the
reported obj — see ``flextool/model.py:1149`` ("DEFERRED").

Strategy: scale ``p_ed_fixed_cost`` × 2.0 on the work_coal_retire
fixture (which has ``coal_plant`` existing with a non-zero fixed
cost).  The expected obj delta is::

    delta = Σ_{(e,d): existing & fixed_cost ≠ 0}
              p_entity_all_existing[e, d] * (factor - 1.0)
                  * ed_fixed_cost[e, d]
                  * p_inflation_op[d]
                  * pd_branch_weight[d]      # ≡ 1.0 in deterministic

Because the perturbed term is a constant in the LP, the optimal
v_flow does not change → the rest of the obj is unchanged.

Marked ``xfail`` until §8.1 is wired into ``flextool/model.py``.  When
wired, this test flips to PASS and is the canonical regression test
for the term.
"""

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import load_flextool

from tests.perturbation._harness import (
    scale_param,
    solve_obj,
    assert_obj_changed_by,
)


WORK = Path(__file__).resolve().parents[1] / "data" / "work_coal_retire"


@pytest.fixture(scope="module")
def retire_data():
    return load_flextool(WORK)


def _existing_fixed_cost_constant(d) -> float:
    """Closed-form value of obj §8.1 from the (e, d) parameters."""
    if d.p_ed_fixed_cost is None or d.p_entity_all_existing is None:
        return 0.0
    df = (
        d.p_entity_all_existing.frame.rename({"value": "exist"})
        .join(d.p_ed_fixed_cost.frame.rename({"value": "fc"}), on=["e", "d"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
    )
    df = df.with_columns(contrib=pl.col("exist") * pl.col("fc") * pl.col("infl"))
    return float(df["contrib"].sum())


@pytest.mark.perturbation
def test_perturb_p_entity_existing_fixed_cost_constant_obj_offset(retire_data):
    if retire_data.p_ed_fixed_cost is None:
        pytest.skip("p_ed_fixed_cost not present in fixture")
    if retire_data.p_entity_all_existing is None:
        pytest.skip("p_entity_all_existing not present in fixture")

    factor = 2.0

    base_obj = solve_obj(retire_data, include_existing_fixed_cost=True)
    term_base = _existing_fixed_cost_constant(retire_data)
    assert term_base != 0.0, (
        f"work_coal_retire should exercise §8.1 (got constant=0); "
        f"check ed_fixed_cost / p_entity_all_existing fixture data")

    perturbed = scale_param(retire_data, "p_ed_fixed_cost", factor)
    perturbed_obj = solve_obj(perturbed, include_existing_fixed_cost=True)

    expected_delta = (factor - 1.0) * term_base
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
