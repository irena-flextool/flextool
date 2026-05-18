"""Unit tests for ``recommend_user_bound_scale_from_lp``.

Regression coverage for the Rivendell S17_horizon2035_slice failure where
the polars port's anchored-to-max heuristic picked ``N=-19`` on a
Rivendell-shaped LP (``abs_min=1.0``, ``abs_max≈6e5``), crushing all
bounds to ~2e-6 and causing HiGHS' presolve to declare the user-scaled
model infeasible.

The fix ports the geometric-midpoint policy + 6-decade spread threshold
from ``flextool.flextoolrunner.scaling.decide_user_bound_scale``
(Agent 18e).

Rivendell bug 1+5+6 follow-up: ``recommend_user_bound_scale_from_lp``
now considers *column bounds only* (matching ``decide_user_bound_scale``
in flextoolrunner/scaling.py, which is fed col_lower/col_upper via
``compute_bound_stats``).  Pooling row bounds with col bounds inflated
the spread on every realistic energy model — cumulative-resource and
annual-flow RHS values routinely reach 1e+6 while column bounds are
typically O(1) — and the resulting ``N=-10`` recommendation broke HiGHS
presolve on Rivendell B0/S17/S22.  HiGHS scales col + row bounds by
``2^N`` but leaves the constraint matrix alone; shrinking ``[0, 1]``
column bounds to ``[0, ~1e-3]`` while matrix coefficients up to ``1e+3``
remain unchanged makes every flow constraint structurally infeasible
because the variable can no longer represent the original flow
magnitudes its constraint RHS expects.
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
    """Rivendell B0/S17/S22: col_bound=(1.0, 1.0), row_bound up to 3.3e6.

    Even with a huge row-bound spread, the column-bound spread is 0
    decades, well below the 6-decade trigger, so no ``user_bound_scale``
    is recommended.  This is the Rivendell-bug-1+5+6 regression case:
    pre-fix, the recommender pooled row + col bounds and picked N=-10,
    which dragged column bounds down to ~1e-3 while leaving matrix
    coefficients at ~1e+3, breaking HiGHS presolve.
    """
    lp_ranges = {
        "row_bound": (0.805, 3.31e6),
        "col_bound": (1.0, 1.0),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    assert n == 0


def test_wide_col_spread_returns_negative_n_clamped():
    """Column-bound spread > 6 decades: the recommender picks N centered
    on the geometric midpoint of the col-bound range, clamped to [-10, 0].
    """
    lp_ranges = {
        "row_bound": (1.0, 1e3),       # irrelevant — col_bound drives the decision
        "col_bound": (1e-3, 1e7),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    # col-only: geo_mid = sqrt(1e7 * 1e-3) = sqrt(1e4) = 100,
    # N = -round(log2(100)) ≈ -7, well within [-10, 0].
    assert -10 <= n <= 0
    assert n < 0  # non-trivial
    # Verify the small bound survives: 1e-3 * 2^n >= a reasonable floor.
    assert 1e-3 * (2 ** n) >= 1e-6


def test_pathological_huge_col_max_clamped_at_min():
    """Very large abs_max on the *column* side with effectively-zero
    abs_min should not return N below USER_BOUND_SCALE_MIN (= -10).
    """
    lp_ranges = {
        "row_bound": (1.0, 1e3),
        "col_bound": (1e-300, 1e15),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    assert n >= USER_BOUND_SCALE_MIN
    assert n <= USER_BOUND_SCALE_MAX


def test_effectively_zero_col_min_falls_back_to_floor():
    """col_bound abs_min below ``BOUND_ABS_MIN_EFFECTIVE_ZERO`` (1e-30)
    should trigger the floor-ratio fallback rather than collapse
    log2(geo_mid) to -inf."""
    lp_ranges = {
        "row_bound": None,
        "col_bound": (1e-300, 1e6),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    # With abs_min=1e-300 < BOUND_ABS_MIN_EFFECTIVE_ZERO, falls back to
    # effective_min = 1e6 * 1e-6 = 1.0; geo_mid = sqrt(1e6) = 1000;
    # N = -round(log2(1000)) ≈ -10.
    assert n == USER_BOUND_SCALE_MIN  # -10


def test_narrow_spread_no_scaling():
    """Within-6-decade column-bound spread: no scaling needed."""
    lp_ranges = {
        "row_bound": (0.1, 1e3),
        "col_bound": (1.0, 1e2),
    }
    n = recommend_user_bound_scale_from_lp(lp_ranges)
    assert n == 0


def test_row_only_ranges_ignored():
    """``user_bound_scale`` shrinks col + row bounds but does NOT scale
    the matrix.  A huge row_bound with no col_bound info therefore must
    NOT trigger a recommendation — that's the Rivendell B0/S17/S22 case
    where row_bound up to 3.3e+6 spuriously drove N=-10.
    """
    n = recommend_user_bound_scale_from_lp(
        {"row_bound": (1e-3, 1e15), "col_bound": None}
    )
    assert n == 0


def test_returned_n_is_always_in_clamp():
    """For a sweep of col-bound ranges, the result is always in clamp."""
    for max_b in [1e2, 1e6, 1e10, 1e15]:
        for min_b in [1e-15, 1e-6, 1e-3, 1.0]:
            n = recommend_user_bound_scale_from_lp(
                {"row_bound": None, "col_bound": (min_b, max_b)}
            )
            assert USER_BOUND_SCALE_MIN <= n <= USER_BOUND_SCALE_MAX
