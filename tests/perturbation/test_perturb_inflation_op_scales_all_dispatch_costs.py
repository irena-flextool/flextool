"""Tier 6 perturbation test #1 — operational inflation factor.

Mutation
--------
``solve_data/p_inflation_factor_operations_yearly.csv`` rows with
``period == 'p2020'`` have their ``value`` column doubled. The mod
reads this via ``table data IN`` and uses
``p_inflation_factor_operations_yearly[d]`` as a universal
multiplier on every operational obj term in period ``d`` —
commodity_cost, co2, starts, slack penalties, etc.

Predicted delta
---------------
``coal`` has only operational costs (commodity_cost + slack
penalty); no investment, no fixed costs, no capacity-margin term
— see ``costs_discounted.csv``. Doubling the operational inflation
factor therefore doubles every objective contribution::

    expected_delta = (factor - 1) * baseline_obj
                   = baseline_obj  (with factor=2)

Scenarios with non-operational obj terms (fixed costs of pre-existing
entities, capacity-margin penalty in absolute money) would not scale
1:1 — the inflation factor only multiplies operational/dispatch terms.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.perturbation._harness import (
    _parse_costs_discounted_by_category,
    assert_obj_changed_by,
    rerun_and_get_obj,
    run_baseline,
    scale_input_csv_column,
)


@pytest.mark.perturbation
def test_perturb_inflation_op_scales_all_dispatch_costs(
    test_db_url: str, test_bin_dir: Path, workdir: Path
) -> None:
    scenario = "coal"
    runner, base_obj = run_baseline(workdir, scenario, test_db_url, test_bin_dir)

    # Sanity check: confirm the scenario is pure-operational. If a
    # future change to the test fixture adds a fixed-cost term, this
    # check fails fast with a clear message rather than a confusing
    # tolerance miss.
    base_categories = _parse_costs_discounted_by_category(
        workdir / "output_csv" / scenario / "costs_discounted.csv"
    )
    non_scaling_categories = (
        "fixed cost pre-existing",
        "fixed cost invested",
        "fixed cost reduction of divestments",
        "unit investment & retirement",
        "connection investment & retirement",
        "storage investment & retirement",
        "capacity margin penalty",
    )
    for cat in non_scaling_categories:
        amount = base_categories.get(cat, 0.0)
        assert abs(amount) < 1e-9, (
            f"scenario {scenario!r} has a non-zero {cat!r} component "
            f"({amount}); the inflation_operations factor does not scale "
            f"this term, so the closed-form prediction would be wrong. "
            f"Either pick a pure-dispatch scenario or compute "
            f"expected_delta = (factor - 1) * (base_obj - non_scaling)"
        )

    factor = 2.0
    unpatch = scale_input_csv_column(
        workdir,
        "solve_data/p_inflation_factor_operations_yearly.csv",
        column="value",
        factor=factor,
        period="p2020",
    )
    try:
        perturbed_obj = rerun_and_get_obj(runner, workdir, scenario)
    finally:
        unpatch()

    expected_delta = (factor - 1.0) * base_obj
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
