"""Tier-6 perturbation: ``p_period_share`` (=
``complete_period_share_of_year``) is a *denominator* on every
operational (dispatch) objective term — slack, commodity, CO2,
varCost, startup, online-section.  See ``audit/objective_audit.md``
§1, §2, §4, §5, §6 (universal denominator).

work_coal has only dispatch terms.  Scaling ``p_period_share`` × 2.0
divides every obj coefficient by 2 → obj halves.  The LP optimum is
preserved because every cost coefficient scales by the same factor.

A failure here narrows the bug to: the ``/p_period_share`` factor is
missing on one of the dispatch obj terms (e.g. computed once for some
terms but not all).
"""

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.perturbation._harness import (
    scale_param,
    solve_obj,
    assert_obj_changed_by,
)


SCENARIO = "coal"


@pytest.fixture(scope="module")
def work(scenario_workdir):
    return scenario_workdir(SCENARIO)

@pytest.fixture(scope="module")
def coal_data(work):
    return load_flextool(work)


@pytest.mark.perturbation
def test_perturb_p_period_share_inversely_scales_dispatch_costs(coal_data):
    factor = 2.0

    base_obj = solve_obj(coal_data)
    perturbed = scale_param(coal_data, "p_period_share", factor)
    perturbed_obj = solve_obj(perturbed)

    # ``op_factor = step_duration * p_timestep_weight * p_inflation_op
    #               / p_period_share``; doubling p_period_share halves
    # every dispatch obj term.
    expected_delta = (1.0 / factor - 1.0) * base_obj
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
