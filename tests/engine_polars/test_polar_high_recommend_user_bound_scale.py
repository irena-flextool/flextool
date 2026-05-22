"""Unit tests for ``polar_high.engine._recommend_user_bound_scale``.

Regression coverage for the Rivendell S17_horizon2035_slice failure where
the polars port's anchored-to-max heuristic picked ``N=-19`` on a
Rivendell-shaped LP (``abs_min=1.0``, ``abs_max≈6e5``), crushing all
bounds to ~2e-6 and causing HiGHS' presolve to declare the user-scaled
model infeasible.

The fix ports the geometric-midpoint policy + 6-decade spread threshold
from ``flextool.flextoolrunner.scaling.decide_user_bound_scale``
(Agent 18e), and Rivendell bug 1+5+6 restricted the heuristic to
*column bounds only* (matching the flextoolrunner contract which is fed
col_lower/col_upper via ``compute_bound_stats``).  HiGHS scales col + row
bounds by ``2^N`` but leaves the constraint matrix alone; including row
bounds in the spread inflates the recommendation on every realistic
energy model — cumulative-resource and annual-flow RHS values routinely
reach 1e+6 while column bounds are typically O(1) — and the resulting
``N=-10`` recommendation broke HiGHS presolve on Rivendell B0/S17/S22.

These tests pin the col_bound-only contract on the polar-high helper
(``flextool.engine_polars.scaling.recommend_user_bound_scale_from_lp``
is being deleted in the same change; its behaviour has been ported
into the polar-high helper since polar-high d8cb34d).
"""
from __future__ import annotations

import math

from polar_high.engine import _recommend_user_bound_scale


# polar-high's clamp range — baked into the helper, mirrored here for
# readability.  See ``polar_high.engine._recommend_user_bound_scale``.
USER_BOUND_SCALE_MIN = -10
USER_BOUND_SCALE_MAX = 0


def test_rivendell_shape_below_spread_threshold_returns_zero():
    """Rivendell B0/S17/S22: col_bound=(1.0, 1.0).

    Pre-fix, recommend_user_bound_scale_from_lp pooled row + col bounds
    and picked N=-10 on this LP because row_bound reached ~3.3e6;
    column bounds were (1.0, 1.0) — 0 decades — so the col-only helper
    must return 0 regardless of how wide the row bounds are.
    """
    assert _recommend_user_bound_scale(1.0, 1.0) == 0


def test_wide_col_spread_returns_negative_n_clamped():
    """Column-bound spread > 6 decades: the recommender picks N centered
    on the geometric midpoint of the col-bound range, clamped to [-10, 0].
    """
    n = _recommend_user_bound_scale(1e-3, 1e7)
    # geo_mid = sqrt(1e7 * 1e-3) = sqrt(1e4) = 100,
    # N = -round(log2(100)) ≈ -7, well within [-10, 0].
    assert USER_BOUND_SCALE_MIN <= n <= USER_BOUND_SCALE_MAX
    assert n < 0  # non-trivial
    # Verify the small bound survives: 1e-3 * 2^n >= a reasonable floor.
    assert 1e-3 * (2 ** n) >= 1e-6


def test_inf_sentinels_return_zero():
    """The stream-time accumulator initialises ``col_bound`` to
    ``(inf, -inf)``; if a Param batch sets it but nothing later updates
    it (e.g. an LP with no finite column bounds at all), the helper may
    see ``(math.inf, math.inf)``.  Returning 0 there means "leave HiGHS'
    own scaling alone", which is the correct default when there's no
    information to act on.
    """
    assert _recommend_user_bound_scale(math.inf, math.inf) == 0
    assert _recommend_user_bound_scale(-math.inf, math.inf) == 0


def test_returned_n_is_always_in_clamp():
    """For a sweep of col-bound ranges, the result is always in clamp.

    Catches pathological values like ``(1e-300, 1e15)`` where naive
    ``log2(sqrt(abs_max * abs_min))`` could collapse to inf / nan.
    """
    for max_b in (1e2, 1e6, 1e10, 1e15):
        for min_b in (1e-15, 1e-6, 1e-3, 1.0):
            if min_b > max_b:
                continue
            n = _recommend_user_bound_scale(min_b, max_b)
            assert USER_BOUND_SCALE_MIN <= n <= USER_BOUND_SCALE_MAX, (
                f"col_bound=({min_b!r}, {max_b!r}) returned N={n}, "
                f"outside the clamp [{USER_BOUND_SCALE_MIN}, "
                f"{USER_BOUND_SCALE_MAX}]"
            )
