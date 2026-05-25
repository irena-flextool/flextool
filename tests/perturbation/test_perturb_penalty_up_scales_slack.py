"""Tier 6 perturbation test #4 — slack penalty multiplier.

Δ.22 — ported to the native-cascade harness.  Mutation targets
``FlexData.p_penalty_up`` (dims ``(n, d, t)``) directly.

Predicted delta
---------------
``base`` is a pure-slack scenario (no generators), so 100 % of the
baseline objective is the upward slack penalty.  Doubling the penalty
doubles the obj exactly::

    expected_delta = baseline_obj.

(LP-optimum invariance: scaling one obj term by ``k`` only scales the
optimum's value when nothing else competes; here the only obj
contribution is that one term, so the optimum scales 1:1.)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.perturbation._harness import (
    assert_obj_changed_by,
    cascade_baseline,
    perturbed_obj,
)


@pytest.mark.perturbation
def test_perturb_penalty_up_scales_slack(
    test_db_url: str, workdir: Path,
) -> None:
    scenario = "base"
    flex_data, base_obj = cascade_baseline(workdir, scenario, test_db_url)

    p_obj = perturbed_obj(flex_data, "p_penalty_up", 2.0)

    expected_delta = base_obj  # full obj is slack penalty
    assert_obj_changed_by(base_obj, p_obj, expected_delta)
