"""Tier-6 perturbation: ``p_step_duration`` is the per-(period, time)
duration multiplier that links MW-scale flows to MWh in
``nodeBalance_eq`` and into storage state dynamics.  A regression
that silently drops ``p_step_duration`` from the storage / nodeBalance
chain would leave the LP obj effectively unchanged when this Param
is scaled.  See ``audit/objective_audit.md`` (universal numerator
column) and the ``nodeBalance_eq`` block in ``flextool/model.py``
where every flow term is multiplied by ``p_step_duration`` (and the
storage state-change term carries it implicitly via the duration of
the flow on the source/sink side).

The flextool counterpart pins this same regression class on a
storage-bearing scenario (``wind_battery``); the spec's original
prediction (``obj × 2``) would only hold if ``p_step_duration`` were
*just* an op-factor numerator, but in flexpy as in flextool it also
appears in nodeBalance and (via the flow contributions to
state_change) in storage dynamics.  Those two appearances do not
cancel cleanly under a uniform scaling — so we assert
"delta is non-trivially non-zero" rather than a numerical match.

A regression that drops ``p_step_duration`` from
``nb_terms["state_change"]``-coupled flows or from the slack
weighting in nodeBalance leaves the perturbed obj equal (or nearly
equal) to the baseline obj, tripping this assertion.

flextool counterpart:
``flextool/tests/perturbation/test_perturb_step_duration_scales_dispatch.py``.
"""

from pathlib import Path

import pytest

from flextool.engine_polars import load_flextool

from tests.engine_polars.perturbation._harness import (
    scale_param,
    solve_obj,
)


WORK = Path(__file__).resolve().parents[1] / "data" / "work_wind_battery"


@pytest.fixture(scope="module")
def battery_data():
    return load_flextool(WORK)


@pytest.mark.perturbation
def test_perturb_step_duration_scales_dispatch(battery_data):
    factor = 2.0

    base_obj = solve_obj(battery_data)
    perturbed = scale_param(battery_data, "p_step_duration", factor)
    perturbed_obj = solve_obj(perturbed)

    delta = perturbed_obj - base_obj
    # Floor: 1‰ of |base_obj| but at least 1.0 so solver float noise
    # cannot masquerade as a genuine ``p_step_duration`` coupling.  A
    # regression that silently drops ``p_step_duration`` from the
    # nodeBalance flow weighting (or from storage state-balance flow
    # contributions) leaves the obj unchanged → assertion trips.
    min_delta = max(abs(base_obj) * 1e-3, 1.0)
    assert abs(delta) >= min_delta, (
        f"step_duration ×{factor} left wind_battery obj effectively "
        f"unchanged (delta={delta!r}, min_delta={min_delta!r}, "
        f"base_obj={base_obj!r}). "
        f"nodeBalance and storage dynamics should carry "
        f"p_step_duration; if this asserts, p_step_duration may be "
        f"silently dropped from the flow / state-balance chain."
    )
