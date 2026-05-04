"""Tier-6 perturbation: ``p_inflation_op`` is a universal multiplier on
**every** operational (dispatch) objective term — slack penalty,
commodity buy/sell, CO2, varCost, startup, online-section terms.  See
``audit/objective_audit.md`` §1, §2, §4, §5, §6 (and § universal column).

The work_coal fixture has only dispatch costs (no investments, no
fixed-cost annuities, no constants) so doubling ``p_inflation_op``
must double the entire objective — and because the scaling is uniform
the LP optimum is preserved (every cost coefficient scales by the same
factor).

A failure here narrows the bug to: the ``p_inflation_op`` factor is
not multiplied through one of the dispatch obj terms (or is multiplied
twice, or has the wrong sign).
"""

from pathlib import Path

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.perturbation._harness import (
    scale_param,
    solve_obj,
    assert_obj_changed_by,
)


WORK = Path(__file__).resolve().parents[1] / "data" / "work_coal"


@pytest.fixture(scope="module")
def coal_data():
    return load_flextool(WORK)


@pytest.mark.perturbation
def test_perturb_p_inflation_op_scales_all_dispatch_costs(coal_data):
    factor = 2.0

    base_obj = solve_obj(coal_data)
    perturbed = scale_param(coal_data, "p_inflation_op", factor)
    perturbed_obj = solve_obj(perturbed)

    # work_coal has only dispatch terms: every obj term carries the
    # ``p_inflation_op`` factor, so the entire obj scales by ``factor``.
    expected_delta = (factor - 1.0) * base_obj
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
