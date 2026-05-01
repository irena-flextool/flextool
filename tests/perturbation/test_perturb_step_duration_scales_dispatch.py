"""Tier 6 perturbation test #5 — step duration multiplier.

Mutation
--------
``solve_data/steps_in_use.csv`` ``step_duration`` column is
doubled. ``steps_in_use.csv`` is the file the mod reads to populate
``step_duration[d, t]`` (see ``flextool.mod`` line ~781:
``table data IN 'CSV' 'solve_data/steps_in_use.csv' : dt <- ...``).

Spec prediction (and why it's wrong)
------------------------------------
The spec predicts ``obj × 2`` on the assumption that ``step_duration``
is a "universal time multiplier" on every obj term. In flextool's
actual mod, every operational obj term carries the factor::

    step_duration[d, t] / complete_period_share_of_year[d]

and ``complete_period_share_of_year[d] = sum_t step_duration[d, t] / 8760``
— a sum of the very same step_duration values. Scaling
step_duration uniformly by 2 therefore doubles BOTH numerator and
denominator, and the ratio is unchanged. The objective is
invariant under uniform step_duration scaling.

This is a real model behaviour, not a bug, so we mark the test
``xfail`` with the closed-form prediction documented above. A
``XPASS`` here would itself be a regression signal — the most
likely cause being someone removing ``complete_period_share_of_year``
from the obj and breaking annualization.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.perturbation._harness import (
    assert_obj_changed_by,
    rerun_and_get_obj,
    run_baseline,
    scale_input_csv_column,
)


@pytest.mark.perturbation
@pytest.mark.xfail(
    reason=(
        "step_duration appears in numerator (per-step costs) and denominator "
        "(complete_period_share_of_year, which is a sum of step_duration "
        "values). Uniformly scaling step_duration by 2 cancels out — the "
        "objective is invariant, so observed_delta = 0 ≠ baseline_obj."
    ),
    strict=True,
)
def test_perturb_step_duration_scales_dispatch(
    test_db_url: str, test_bin_dir: Path, workdir: Path
) -> None:
    scenario = "coal"
    runner, base_obj = run_baseline(workdir, scenario, test_db_url, test_bin_dir)

    factor = 2.0
    unpatch = scale_input_csv_column(
        workdir,
        "solve_data/steps_in_use.csv",
        column="step_duration",
        factor=factor,
    )
    try:
        perturbed_obj = rerun_and_get_obj(runner, workdir, scenario)
    finally:
        unpatch()

    # Spec prediction (will fail per the xfail reason above).
    expected_delta = (factor - 1.0) * base_obj
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
