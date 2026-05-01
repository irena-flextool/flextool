"""Tier 6 perturbation test #3 — startup cost multiplier.

Mutation
--------
``solve_data/pdProcess.csv`` rows with ``param == 'startup_cost'``
have their ``value`` column doubled. The mod reads pdProcess via
``table data IN 'CSV' 'solve_data/pdProcess.csv'`` and uses
``pdProcess[p, 'startup_cost', d]`` in the obj's startup term.

Scenario choice deviation
-------------------------
The spec recommends scenario ``coal_min_load``. In that scenario the
LP keeps coal_plant online for every timestep (no cycling), so
``starts == 0`` and the prediction
``expected_delta = baseline_starts_component`` collapses to 0 — the
test would pass tautologically with no signal. ``coal_min_load_wind``
has wind variability that forces coal cycling (LP-relaxed
``v_startup_linear``), giving a non-zero baseline starts component
to scale. Same constraint family, same mutation target — better
diagnostic.

Predicted delta
---------------
``expected_delta = (factor - 1) * baseline_starts_component`` —
read from the ``starts`` row of ``costs_discounted.csv``. As with
co2 price (test #2), this assumes the optimal startup pattern
doesn't change under the doubled cost; with a fixed wind profile
and a single thermal unit, the LP relaxation tends to keep the
same fractional online schedule.
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
def test_perturb_startup_cost_scales_starts_term(
    test_db_url: str, test_bin_dir: Path, workdir: Path
) -> None:
    scenario = "coal_min_load_wind"
    runner, base_obj = run_baseline(workdir, scenario, test_db_url, test_bin_dir)

    # Capture the baseline starts obj component from costs_discounted.csv.
    base_categories = _parse_costs_discounted_by_category(
        workdir / "output_csv" / scenario / "costs_discounted.csv"
    )
    base_starts = base_categories.get("starts", 0.0)
    assert base_starts > 0, (
        f"baseline starts component is {base_starts!r}; perturbation test needs "
        f"a non-zero starts obj term — pick a different scenario"
    )

    # Double startup_cost on every (process, period) row.
    factor = 2.0
    unpatch = scale_input_csv_column(
        workdir,
        "solve_data/pdProcess.csv",
        column="value",
        factor=factor,
        param="startup_cost",
    )
    try:
        perturbed_obj = rerun_and_get_obj(runner, workdir, scenario)
    finally:
        unpatch()

    expected_delta = (factor - 1.0) * base_starts
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
