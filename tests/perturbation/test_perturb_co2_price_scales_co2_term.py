"""Tier 6 perturbation test #2 — CO2 price multiplier.

Δ.22 — ported to the native-cascade harness.  The mutation now targets
``FlexData.p_co2_price`` directly (a polar_high ``Param`` of dims
``(g, d, t)``) instead of the workdir ``solve_data/pdtGroup.csv`` row
``param == 'co2_price'`` slice.

Predicted delta
---------------
For a non-degenerate LP, scaling one obj term's coefficient by ``k``
scales that term's contribution at the optimum by ``k`` IF the optimal
dispatch doesn't change.  In ``coal_co2_price`` demand is fixed at
500 MW and only one generator (coal) can serve it, so dispatch is
forced and the bound is tight::

    expected_delta = (factor - 1) * baseline_co2_obj_component
                   = baseline_co2_obj_component  (with factor=2)
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
def test_perturb_co2_price_scales_co2_term(
    test_db_url: str, workdir: Path,
) -> None:
    scenario = "coal_co2_price"
    flex_data, base_obj = cascade_baseline(workdir, scenario, test_db_url)

    base_categories = _parse_costs_discounted_by_category(
        workdir / "output_csv" / scenario / "costs_discounted.csv"
    )
    base_co2 = base_categories.get("co2", 0.0)
    assert base_co2 > 0, (
        f"baseline co2 component is {base_co2!r}; perturbation test needs "
        f"a non-zero co2 obj term — pick a different scenario"
    )

    factor = 2.0
    p_obj = perturbed_obj(flex_data, "p_co2_price", factor)

    expected_delta = (factor - 1.0) * base_co2
    assert_obj_changed_by(base_obj, p_obj, expected_delta)
