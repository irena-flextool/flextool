"""Unit tests for ``recommend_user_bound_scale_from_lp``.

Regression coverage for the Rivendell S17_horizon2035_slice failure where
the polars port's anchored-to-max heuristic picked ``N=-19`` on a
Rivendell-shaped LP (``abs_min=1.0``, ``abs_max≈6e5``), crushing all
bounds to ~2e-6 and causing HiGHS' presolve to declare the user-scaled
model infeasible.

The fix ports the geometric-midpoint policy + 6-decade spread threshold
from ``flextool.flextoolrunner.scaling.decide_user_bound_scale``
(Agent 18e).
"""
from __future__ import annotations

from flextool.engine_polars.scaling import (
    USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN,
    recommend_user_bound_scale_from_lp,
)


def test_empty_lp_ranges_returns_zero():
    assert recommend_user_bound_scale_from_lp({}) == 0


def test_no_bounds_returns_zero():
    assert recommend_user_bound_scale_from_lp(
        {"row_bound": None, "col_bound": None}
    ) == 0


def test_rivendell_shape_below_spread_threshold_returns_zero():
    """Rivendell S17: abs_min=1.0, abs_max≈5.87e5 → spread ≈ 5.77 decades.

    Below the 6-decade trigger, so no ``user_bound_scale`` is recommended
    and HiGHS handles internal scaling on its own.  This is the case
    that was broken pre-fix (anchored-to-max picked N=-19).
    """
    lp_ranges = {
        "row_bound": (1.0, 5.87e5),
        "col_bound": (1.0, 1.0),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    assert n == 0


def test_wide_spread_returns_negative_n_clamped():
    """Spread > 6 decades (small abs_min, large abs_max): the recommender
    picks N centered on the geometric midpoint, clamped to [-10, 0]."""
    lp_ranges = {
        "row_bound": (1e-3, 1e7),
        "col_bound": (1e-2, 1e3),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    # geo_mid = sqrt(1e7 * 1e-3) = sqrt(1e4) = 100, N = -round(log2(100)) = -7
    # Both within the [-10, 0] clamp.
    assert -10 <= n <= 0
    assert n < 0  # non-trivial
    # Verify the small bound survives: 1e-3 * 2^n >= a reasonable floor.
    assert 1e-3 * (2 ** n) >= 1e-6


def test_pathological_huge_max_clamped_at_min():
    """Very large abs_max with effectively-zero abs_min should not return
    N below USER_BOUND_SCALE_MIN (= -10).
    """
    lp_ranges = {
        "row_bound": (1e-300, 1e15),
        "col_bound": (1.0, 1.0),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    assert n >= USER_BOUND_SCALE_MIN
    assert n <= USER_BOUND_SCALE_MAX


def test_effectively_zero_min_falls_back_to_floor():
    """abs_min below ``BOUND_ABS_MIN_EFFECTIVE_ZERO`` (1e-30) should
    trigger the floor-ratio fallback rather than collapse log2(geo_mid)
    to -inf."""
    lp_ranges = {
        "row_bound": (1e-300, 1e6),
        "col_bound": None,
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    # With abs_min=1e-300 < BOUND_ABS_MIN_EFFECTIVE_ZERO, falls back to
    # effective_min = 1e6 * 1e-6 = 1.0; geo_mid = sqrt(1e6) = 1000;
    # N = -round(log2(1000)) = -10 (clamped from -10 exactly).
    assert n == USER_BOUND_SCALE_MIN  # -10


def test_narrow_spread_no_scaling():
    """Within-6-decade bound spread: no scaling needed."""
    lp_ranges = {
        "row_bound": (0.1, 1e3),
        "col_bound": (1.0, 1e2),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    assert n == 0


def test_returned_n_is_always_in_clamp():
    """For a sweep of bound ranges, the result is always in clamp."""
    for max_b in [1e2, 1e6, 1e10, 1e15]:
        for min_b in [1e-15, 1e-6, 1e-3, 1.0]:
            n = recommend_user_bound_scale_from_lp(
                {"row_bound": (min_b, max_b), "col_bound": None}
            )
            assert USER_BOUND_SCALE_MIN <= n <= USER_BOUND_SCALE_MAX
