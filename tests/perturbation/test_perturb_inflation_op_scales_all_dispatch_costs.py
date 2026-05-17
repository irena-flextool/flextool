"""Tier 6 perturbation test #1 — operational inflation factor.

Δ.22 — ported to the native-cascade harness.  Mutation targets
``FlexData.p_inflation_op`` (dims ``(d,)``) directly.

Predicted delta
---------------
``coal`` has only operational costs (commodity_cost + slack penalty);
no investment, no fixed costs, no capacity-margin term — see
``costs_discounted.csv``.  Doubling the operational inflation factor
therefore doubles every objective contribution::

    expected_delta = (factor - 1) * baseline_obj
                   = baseline_obj  (with factor=2)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.perturbation._harness import (
    _parse_costs_discounted_by_category,
    assert_obj_changed_by,
    cascade_baseline,
    perturbed_obj,
)


@pytest.mark.perturbation
def test_perturb_inflation_op_scales_all_dispatch_costs(
    test_db_url: str, workdir: Path,
) -> None:
    scenario = "coal"
    flex_data, base_obj = cascade_baseline(workdir, scenario, test_db_url)

    # Sanity check: confirm the scenario is pure-operational.  If a
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
        assert abs(amount) < 1.0, (
            f"scenario {scenario!r} has a non-zero {cat!r} component "
            f"({amount}); the inflation_operations factor does not scale "
            f"this term, so the closed-form prediction would be wrong. "
            f"Either pick a pure-dispatch scenario or compute "
            f"expected_delta = (factor - 1) * (base_obj - non_scaling)"
        )

    factor = 2.0
    # ``p_inflation_op`` is keyed by ``d``; the legacy harness filtered
    # to ``period='p2020'`` but this scenario only has p2020 anyway.
    p_obj = perturbed_obj(flex_data, "p_inflation_op", factor,
                           filters={"d": "p2020"})

    expected_delta = (factor - 1.0) * base_obj
    assert_obj_changed_by(base_obj, p_obj, expected_delta)
