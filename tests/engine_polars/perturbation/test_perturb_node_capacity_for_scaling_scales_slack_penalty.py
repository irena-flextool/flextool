"""Tier-6 perturbation: ``node_capacity_for_scaling`` is the per-node
multiplier that aligns flexpy's penalty (€/MWh of slack) with the
.mod's penalty (€/MWh of node capacity).  See
``audit/objective_audit.md`` §1.1 and §1.2 (HIGH IMPACT).

The work_base fixture is slack-only (no processes) so the entire
objective is the sum of vq_state_up/down × penalty_up/down terms.
Scaling ``node_capacity_for_scaling`` × 2 should double the penalty
coefficient on every slack term — and because slack values are pinned
by ``nodeBalance_eq`` (vq_up - vq_down = -inflow with both ≥ 0), the
LP optimum doesn't shift.  Therefore the new obj is exactly 2 × the
old.

A failure here narrows the bug to: the
``node_capacity_for_scaling[n,d]`` factor missing or mis-applied on
the §1.1 / §1.2 slack obj terms in ``flextool/model.py``.
"""

from pathlib import Path

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.perturbation._harness import (
    scale_param,
    solve_obj,
    assert_obj_changed_by,
)


WORK = Path(__file__).resolve().parents[1] / "data" / "work_base"


@pytest.fixture(scope="module")
def base_data():
    return load_flextool(WORK)


@pytest.mark.perturbation
def test_perturb_node_capacity_for_scaling_scales_slack_penalty(base_data):
    if base_data.p_node_capacity_for_scaling is None:
        pytest.skip("p_node_capacity_for_scaling not loaded for this fixture")

    factor = 2.0

    base_obj = solve_obj(base_data)
    perturbed = scale_param(base_data, "p_node_capacity_for_scaling", factor)
    perturbed_obj = solve_obj(perturbed)

    # Slack-only fixture — entire obj is slack-penalty terms.  Each carries
    # ``node_capacity_for_scaling`` linearly; LP optimum is preserved
    # because slack values are pinned by nodeBalance_eq.
    expected_delta = (factor - 1.0) * base_obj
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
