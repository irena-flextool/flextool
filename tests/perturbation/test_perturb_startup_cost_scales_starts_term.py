"""Tier 6 perturbation test #3 — startup cost multiplier.

Δ.22 — ported to the native-cascade harness.  Mutation targets
``FlexData.p_startup_cost`` (dims ``(p, d)``) directly.

Scenario choice deviation
-------------------------
The spec recommends scenario ``coal_min_load``.  In that scenario the
LP keeps coal_plant online for every timestep (no cycling), so
``starts == 0`` and the prediction
``expected_delta = baseline_starts_component`` collapses to 0 — the
test would pass tautologically with no signal.  ``coal_min_load_wind``
has wind variability that forces coal cycling (LP-relaxed
``v_startup_linear``), giving a non-zero baseline starts component to
scale.  Same constraint family, same mutation target — better
diagnostic.

Predicted delta
---------------
``expected_delta = (factor - 1) * baseline_starts_component`` — read
from the ``starts`` row of ``costs_discounted.csv``.  As with co2
price (test #2), this assumes the optimal startup pattern doesn't
change under the doubled cost.
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
@pytest.mark.xfail(
    reason=(
        "Blocked by NEW BUG A3/A4 (specs/model_bugs.md): every UC "
        "scenario in the main DB — coal_min_load_wind, coal_min_load, "
        "coal_unit_size_MIP_wind, coal_wind_min_uptime — trips a "
        "KeyError for 'coal_plant' in process_outputs.out_flows when "
        "write_outputs builds the unit-online dataframe.  The "
        "startup-cost perturbation needs costs_discounted.csv to read "
        "the baseline 'starts' component, and that file cannot be "
        "written until A3 is fixed.  The port itself is complete and "
        "the in-memory mutation path works (verified via the other 4 "
        "perturbation tests); only the cost-decomposition read is "
        "blocked."
    ),
    strict=True,
)
def test_perturb_startup_cost_scales_starts_term(
    test_db_url: str, workdir: Path,
) -> None:
    scenario = "coal_min_load_wind"
    flex_data, base_obj = cascade_baseline(workdir, scenario, test_db_url)

    base_categories = _parse_costs_discounted_by_category(
        workdir / "output_csv" / scenario / "costs_discounted.csv"
    )
    base_starts = base_categories.get("starts", 0.0)
    assert base_starts > 0, (
        f"baseline starts component is {base_starts!r}; perturbation test needs "
        f"a non-zero starts obj term — pick a different scenario"
    )

    factor = 2.0
    p_obj = perturbed_obj(flex_data, "p_startup_cost", factor)

    expected_delta = (factor - 1.0) * base_starts
    assert_obj_changed_by(base_obj, p_obj, expected_delta)
