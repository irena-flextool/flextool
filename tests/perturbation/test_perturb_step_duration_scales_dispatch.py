"""Tier 6 perturbation test #5 — step_duration on storage dynamics.

The spec's original prediction (``obj × 2`` from doubling
``step_duration``) was wrong: in flextool's mod every operational obj
term carries ``step_duration[d, t] / complete_period_share_of_year[d]``
and ``complete_period_share_of_year`` is itself a sum of step_duration
values, so the ratio is invariant under uniform scaling.  That
cancellation is **by design** — operational weight of one year needs
to equal one year so it matches the investment-annuity weighting.  A
dispatch-only test of that invariant would also be hard to make robust
on top of the harness's solve_data-level mutation: step_duration
cascades into several preprocessing-derived files
(``complete_period_share_of_year.csv`` and friends) that the patch
doesn't repaint, so a naive "double step_duration in solve_data" run
ends up with an inconsistent state, not a true uniform scaling.

What ``step_duration`` *does* affect — which the user flagged
explicitly — is **storage dynamics**.  Storage state evolution carries
an explicit ``step_duration`` factor that is not cancelled by the
annualisation ratio (``v_state[t] = v_state[t-1] +
(charge - discharge) * step_duration``), so doubling the step makes
the model see twice the energy charged/discharged per timestep.  The
optimal dispatch then changes and the obj moves.  That is the
regression class this test pins: a bug that drops ``step_duration``
from the storage state-balance constraint would leave the obj
unchanged when it should not.

We don't have a clean closed-form for the magnitude — the new optimum
depends on the wind/load profile and how soon the doubled charge rate
binds the storage capacity — so this is a "delta is non-trivially
non-zero" assertion rather than a numerical match.  That is weaker
than the other four perturbation tests but is the right granularity
for this multiplier on this scenario.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.perturbation._harness import (
    rerun_and_get_obj,
    run_baseline,
    scale_input_csv_column,
)


@pytest.mark.perturbation
def test_perturb_step_duration_changes_storage_obj(
    test_db_url: str, test_bin_dir: Path, workdir: Path
) -> None:
    scenario = "wind_battery"
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

    delta = perturbed_obj - base_obj
    # Floor: scenario-relative (1‰ of base_obj) but at least 1 M CUR so
    # solver float noise can't masquerade as "step_duration influences
    # storage".  A regression that silently drops step_duration from the
    # storage state-balance leaves the obj unchanged → assertion trips.
    min_delta = max(abs(base_obj) * 1e-3, 1.0)
    assert abs(delta) >= min_delta, (
        f"step_duration ×2 left wind_battery obj effectively unchanged "
        f"(delta={delta!r}, min_delta={min_delta!r}, base_obj={base_obj!r}). "
        f"Storage state-balance should carry step_duration without "
        f"cancellation; if this asserts, step_duration may be silently "
        f"dropped from the storage dynamics."
    )
