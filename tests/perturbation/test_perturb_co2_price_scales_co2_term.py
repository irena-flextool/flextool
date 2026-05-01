"""Tier 6 perturbation test #2 — CO2 price multiplier.

Mutation
--------
``solve_data/pdtGroup.csv`` rows with ``param == 'co2_price'`` have
their ``value`` column doubled. The mod reads pdtGroup via
``table data IN 'CSV' 'solve_data/pdtGroup.csv'`` and uses
``pdtGroup[g, 'co2_price', d, t]`` in the obj's CO2 term.

Predicted delta
---------------
For a non-degenerate LP, scaling one obj term's coefficient by ``k``
scales that term's contribution at the optimum by ``k`` IF the
optimal dispatch doesn't change. If the change is small enough that
the optimal dispatch is fixed, then::

    expected_delta = (factor - 1) * baseline_co2_obj_component
                   = baseline_co2_obj_component  (with factor=2)

Read the baseline ``co2`` row from
``output_csv/coal_co2_price/costs_discounted.csv``. Note that
doubling the co2 price MIGHT shift dispatch (less coal use); if so,
the closed-form delta is an upper bound. In ``coal_co2_price`` the
demand is fixed at 500 MW and only one generator (coal) can serve
it, so dispatch is forced and the bound is tight.
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
def test_perturb_co2_price_scales_co2_term(
    test_db_url: str, test_bin_dir: Path, workdir: Path
) -> None:
    scenario = "coal_co2_price"
    runner, base_obj = run_baseline(workdir, scenario, test_db_url, test_bin_dir)

    # Capture the baseline co2 obj component from costs_discounted.csv.
    base_categories = _parse_costs_discounted_by_category(
        workdir / "output_csv" / scenario / "costs_discounted.csv"
    )
    base_co2 = base_categories.get("co2", 0.0)
    assert base_co2 > 0, (
        f"baseline co2 component is {base_co2!r}; perturbation test needs "
        f"a non-zero co2 obj term — pick a different scenario"
    )

    # Double co2_price on every (group, period, time) row.
    factor = 2.0
    unpatch = scale_input_csv_column(
        workdir,
        "solve_data/pdtGroup.csv",
        column="value",
        factor=factor,
        param="co2_price",
    )
    try:
        perturbed_obj = rerun_and_get_obj(runner, workdir, scenario)
    finally:
        unpatch()

    expected_delta = (factor - 1.0) * base_co2
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
