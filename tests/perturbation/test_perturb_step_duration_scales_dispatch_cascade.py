"""Tier 6 perturbation test #5 — step_duration on storage dynamics.

Δ.22 — ported to the native-cascade harness.  Mutation targets
``FlexData.p_step_duration`` (dims ``(d, t)``) directly.

The spec's original prediction (``obj × 2`` from doubling
``step_duration``) was wrong: in flextool every operational obj term
carries ``step_duration[d, t] / complete_period_share_of_year[d]`` and
``complete_period_share_of_year`` is itself a sum of step_duration
values, so the ratio is invariant under uniform scaling.  That
cancellation is **by design** — operational weight of one year needs
to equal one year so it matches the investment-annuity weighting.

What ``step_duration`` *does* affect is storage dynamics.  Storage
state evolution carries an explicit ``step_duration`` factor that is
NOT cancelled by the annualisation ratio
(``v_state[t] = v_state[t-1] + (charge - discharge) * step_duration``),
so doubling the step makes the model see twice the energy
charged/discharged per timestep.  The optimal dispatch then changes
and the obj moves.  That is the regression class this test pins: a
bug that drops ``step_duration`` from the storage state-balance
constraint would leave the obj unchanged when it should not.

We don't have a clean closed-form for the magnitude — the new optimum
depends on the wind/load profile and how soon the doubled charge rate
binds the storage capacity — so this is a "delta is non-trivially
non-zero" assertion rather than a numerical match.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.perturbation._harness import (
    cascade_baseline,
    perturbed_obj,
)


@pytest.mark.perturbation
def test_perturb_step_duration_changes_storage_obj(
    test_db_url: str, workdir: Path,
) -> None:
    scenario = "wind_battery"
    flex_data, base_obj = cascade_baseline(workdir, scenario, test_db_url)

    factor = 2.0
    p_obj = perturbed_obj(flex_data, "p_step_duration", factor)

    delta = p_obj - base_obj
    # Floor: scenario-relative (1‰ of base_obj) but at least 1 CUR so
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
