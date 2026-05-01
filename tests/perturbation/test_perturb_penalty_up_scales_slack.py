"""Tier 6 perturbation test #4 — slack penalty multiplier.

Mutation
--------
``solve_data/pdtNode.csv`` rows with ``param == 'penalty_up'`` have
their ``value`` column doubled. This is the file the mod reads
(``table data IN 'CSV' 'solve_data/pdtNode.csv'``) for the
``pdtNode[n, 'penalty_up', d, t]`` parameter that multiplies the
upward state-slack obj term.

Predicted delta
---------------
``base`` is a pure-slack scenario (no generators), so 100 % of the
baseline objective is the upward slack penalty. Doubling the
penalty doubles the obj exactly:

    expected_delta = baseline_obj.

(LP-optimum invariance: scaling one obj term by ``k`` only scales
the optimum's value when nothing else competes; here the only obj
contribution is that one term, so the optimum scales 1:1.)
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
def test_perturb_penalty_up_scales_slack(
    test_db_url: str, test_bin_dir: Path, workdir: Path
) -> None:
    scenario = "base"
    runner, base_obj = run_baseline(workdir, scenario, test_db_url, test_bin_dir)

    # Double penalty_up on every (period, time) row in pdtNode.
    unpatch = scale_input_csv_column(
        workdir,
        "solve_data/pdtNode.csv",
        column="value",
        factor=2.0,
        param="penalty_up",
    )
    try:
        perturbed_obj = rerun_and_get_obj(runner, workdir, scenario)
    finally:
        unpatch()

    expected_delta = base_obj  # full obj is slack penalty
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
