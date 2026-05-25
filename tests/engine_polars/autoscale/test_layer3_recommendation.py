"""Layer 3 (HiGHS-native top-up) recommendation unit tests.

Constructs synthetic :class:`RangeReport` inputs covering the
documented branches of :func:`recommend_layer3`:

* **In-zone** — every magnitude already inside HiGHS' comfort zone.
  Layer 3 must pick ``N_obj = N_bnd = 0`` (no-op).
* **Cost large-end only** — only the objective overshoots HiGHS'
  ``|c| <= 1e+4`` ceiling; bounds stay clean.  ``N_obj`` is the
  power-of-two exponent that pulls the worst cost into the comfort
  zone, ``N_bnd == 0``.
* **Bound large-end only** — symmetric: bounds overshoot, cost stays
  clean.  ``N_bnd != 0``, ``N_obj == 0``.
* **Severe overshoot (D's escape)** — ``max(|b|) >= 1e+9`` and the
  naive clamp would crush the small end.  Geometric-centering branch
  fires and the chosen exponent lies between the two clamping
  alternatives.
* **Refuse-to-scale (Rivendell-shaped)** — moderate overshoot where
  the naive recommendation would crush the small end below ``1e-4``.
  Layer 3 must refuse and emit ``N_bnd == 0``.
* **Manual override** — ``config.user_bound_scale`` set; Layer 3 must
  use that integer verbatim and emit a "manual override" reasoning
  string.  Objective auto-recommendation still runs.
* **Clamp** — an input that would yield ``N=-50`` is clamped to
  ``-30`` (HiGHS' option-range floor).

The tests do not invoke HiGHS — :func:`recommend_layer3` is a pure
function over the post-Layer-2 :class:`RangeReport`, so the synthetic
inputs are sufficient.
"""
from __future__ import annotations

import math

import pytest

from flextool.engine_polars.autoscale import (
    AutoScaleConfig,
    Layer3Plan,
    RangeReport,
    recommend_layer3,
)


def _cfg(user_bound_scale: int | None = None) -> AutoScaleConfig:
    return AutoScaleConfig(
        enabled=True,
        threshold_decades=9.0,
        user_bound_scale=user_bound_scale,
        report_yaml_path=None,
    )


def _ranges(
    *,
    cost=(math.nan, math.nan),
    bound=(math.nan, math.nan),
    rhs=(math.nan, math.nan),
    matrix=(1e-2, 1e2),
) -> RangeReport:
    """Build a :class:`RangeReport` from per-group magnitude pairs."""
    return RangeReport(
        matrix=matrix,
        cost=cost,
        bound=bound,
        rhs=rhs,
        cross_group_max_ratio=math.nan,
        trigger=True,
    )


def test_in_zone_no_scaling() -> None:
    """All magnitudes inside HiGHS' comfort zone → both N values 0."""
    r = _ranges(cost=(1.0, 1e3), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_layer3(r, _cfg())
    assert plan.user_objective_scale == 0
    assert plan.user_bound_scale == 0
    assert plan.simplex_scale_strategy == 2
    assert "in-zone" in plan.reasoning


def test_cost_overshoot_only() -> None:
    """``max(|c|) = 1e+7`` → N_obj = floor(log2(1e+4 / 1e+7)) = -10."""
    r = _ranges(cost=(1.0, 1e7), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_layer3(r, _cfg())
    # 1e+4 / 1e+7 = 1e-3.  log2(1e-3) ≈ -9.97 → floor = -10.
    assert plan.user_objective_scale == -10
    assert plan.user_bound_scale == 0


def test_bound_overshoot_only_moderate_clamp_large() -> None:
    """Bound overshoot of ~3 decades, min comfortably above 1e-4 → clamp.

    ``max(|b|) = 1e+8``; the naive clamp does NOT crush the small
    end (min 1e-2 stays at 1e-10 → wait, that's below 1e-4).  Use a
    higher floor to keep the test on the clamp-large branch.
    """
    # max_b = 1e+8, min_b = 1e+1.  Clamp-large brings max to ~1e6.
    # dl = floor(log2(1e+6 / 1e+8)) = floor(-6.64) = -7.
    # scaled_min = 1e+1 * 2**-7 ≈ 0.078 → above 1e-4 → no refuse.
    r = _ranges(cost=(1.0, 1e2), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_layer3(r, _cfg())
    assert plan.user_objective_scale == 0
    assert plan.user_bound_scale == -7
    assert "auto" in plan.reasoning


def test_severe_overshoot_geometric_escape() -> None:
    """``max(|b|) >= 1e+9`` AND naive clamp crushes min → geo-centering.

    ``min_b = 1e-3``, ``max_b = 1e+12``: naive dl = floor(log2(1e-6)) = -20,
    scaled_min = 1e-3 * 2**-20 ≈ 9.5e-10 (well below 1e-4) → refuse-or-escape.
    Severe trigger: 1e+12 >= 1e+9 ✓ → escape via geometric centering.

    Geo-centering exponent: log2(sqrt(1e-4 * 1e+6) / sqrt(1e-3 * 1e+12))
    = log2(sqrt(1e+2) / sqrt(1e+9)) = log2(10 / ~31623) ≈ log2(3.16e-4)
    ≈ -11.6 → round = -12.
    """
    r = _ranges(cost=(1.0, 1e2), bound=(1e-3, 1e12), rhs=(1e-3, 1e12))
    plan = recommend_layer3(r, _cfg())
    assert plan.user_bound_scale == -12
    assert "escape" in plan.reasoning


def test_refuse_to_scale_rivendell_shape() -> None:
    """Moderate overshoot where the naive clamp would crush min → refuse.

    Mirrors the Rivendell B0/S17 shape: tight col bounds at 1.0, RHS up
    to ~2e+8.  Naive dl ≈ floor(log2(1e+6 / 2e+8)) = floor(-7.64) = -8.
    With min ~1.84e-3, scaled_min = 1.84e-3 * 2**-8 ≈ 7.2e-6 → below
    1e-4.  Severe trigger: 2e+8 < 1e+9 → refuse.  N_bnd = 0.
    """
    r = _ranges(cost=(1.0, 1e2), bound=(1.0, 1.0), rhs=(1.84e-3, 2.02e8))
    plan = recommend_layer3(r, _cfg())
    assert plan.user_bound_scale == 0
    assert "refuse" in plan.reasoning


def test_manual_override_disables_auto_for_bounds_only() -> None:
    """``config.user_bound_scale`` set → that integer wins for bounds.

    Objective auto-recommendation still runs.  Reasoning string surfaces
    "manual override" for auditability.
    """
    r = _ranges(cost=(1.0, 1e7), bound=(1e-2, 1e3), rhs=(1e-2, 1e3))
    plan = recommend_layer3(r, _cfg(user_bound_scale=-5))
    assert plan.user_bound_scale == -5
    # Cost auto still picks N_obj = -10.
    assert plan.user_objective_scale == -10
    assert "manual override" in plan.reasoning
    assert "user_bound_scale=-5" in plan.reasoning


def test_manual_override_zero_is_respected() -> None:
    """Manual override of 0 must short-circuit auto for bounds.

    ``user_bound_scale=0`` is a valid explicit "leave alone" — Layer 3
    must NOT auto-pick a non-zero value when the operator forced 0.
    """
    r = _ranges(cost=(1.0, 1e2), bound=(1e1, 1e8), rhs=(1e1, 1e8))
    plan = recommend_layer3(r, _cfg(user_bound_scale=0))
    assert plan.user_bound_scale == 0
    assert "manual override" in plan.reasoning


def test_clamp_to_30() -> None:
    """An input that would yield N=-50 is clamped to N=-30.

    Cost magnitude of 1e+20 implies ``floor(log2(1e+4 / 1e+20)) = -54``;
    the clamp brings this to -30 (HiGHS' option-range floor).
    """
    r = _ranges(cost=(1.0, 1e20), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_layer3(r, _cfg())
    assert plan.user_objective_scale == -30


def test_empty_cost_yields_zero() -> None:
    """An LP with no finite-non-zero costs (``cost=(nan, nan)``) → N_obj=0."""
    r = _ranges(cost=(math.nan, math.nan), bound=(1.0, 1.0), rhs=(1.0, 1.0))
    plan = recommend_layer3(r, _cfg())
    assert plan.user_objective_scale == 0
    assert plan.user_bound_scale == 0
