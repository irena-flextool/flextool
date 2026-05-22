"""Unit tests for ``polar_high.engine._recommend_user_bound_scale``.

The helper is now a direct port of HiGHS' own ``suggestScaling`` lambda
at ``HighsSolve.cpp:570-607`` — it pulls ``max(bound_max, rhs_max)`` into
HiGHS' ``[kExcessivelySmallBoundValue, kExcessivelyLargeBoundValue]`` =
``[1e-4, 1e+6]`` comfort zone using outer-rounded log2, and reproduces
the integer that HiGHS prints in its ``"Consider setting the
user_bound_scale option to <N>"`` recommendation byte-for-byte.

These tests pin the two key empirical cases (DES and Rivendell pre-fix)
plus the surrounding behaviour: comfort-zone short-circuit, empty
ranges, ``min`` is deliberately ignored, scale-up direction is
supported, and the result stays inside the clamp.
"""
from __future__ import annotations

from polar_high.engine import _recommend_user_bound_scale


# polar-high's defensive clamp range — see
# ``polar_high.engine._USER_BOUND_SCALE_CLAMP_LO/HI``.
USER_BOUND_SCALE_CLAMP_LO = -30
USER_BOUND_SCALE_CLAMP_HI = 30


def test_des_shape_reproduces_highs_printed_recommendation():
    """DES: ``bound=(1, 1)``, ``rhs=(2e-5, 4e+7)``.

    ``max=4e+7`` > ``1e+6`` -> ratio=1e+6/4e+7=0.025, log2=-5.32,
    floor=-6.  HiGHS prints exactly this recommendation on the DES
    scenario.
    """
    assert _recommend_user_bound_scale((1.0, 1.0), (2e-5, 4e+7)) == -6


def test_rivendell_prefix_shape_returns_minus_two():
    """Rivendell pre-fix LP: ``bound=(1, 1)``, ``rhs=(0.82, 3.07e+6)``.

    ``max=3.07e+6`` > ``1e+6`` -> ratio=1e+6/3.07e+6=0.326,
    log2=-1.617, floor=-2.  The historical ``N=-10`` came from a
    different (broken) heuristic — HiGHS' own formula was always
    conservative enough to keep Rivendell solving on this LP.
    """
    assert _recommend_user_bound_scale((1.0, 1.0), (0.82, 3.07e+6)) == -2


def test_in_comfort_zone_returns_zero():
    """``max(bound, rhs)`` already in ``[1e-4, 1e+6]`` -> 0."""
    assert _recommend_user_bound_scale((1.0, 1.0), (1.0, 100.0)) == 0


def test_both_ranges_none_returns_zero():
    """No finite entries on either side -> 0 (HiGHS' default)."""
    assert _recommend_user_bound_scale(None, None) == 0


def test_only_min_below_threshold_does_not_trigger_scaling():
    """HiGHS' formula deliberately ignores ``min``.

    With ``rhs=(1e-10, 1.0)`` the max is 1.0 — well inside the comfort
    zone — so the recommendation is 0 even though the min is far below
    ``1e-4``.  This documents the known behaviour: a small min on an
    otherwise in-zone model never triggers ``user_bound_scale``
    (matrix scaling, not bound scaling, is the right tool for that).
    """
    assert _recommend_user_bound_scale((1.0, 1.0), (1e-10, 1.0)) == 0


def test_max_below_small_threshold_scales_up():
    """``max=1e-8`` < ``1e-4`` -> ratio=1e+4, log2=13.29, ceil=14."""
    assert _recommend_user_bound_scale(None, (1e-8, 1e-8)) == 14


def test_bound_range_alone_drives_recommendation():
    """If only ``bound`` is supplied and it's outside the zone, scale."""
    # bound_max=1e+8 > 1e+6 -> ratio=0.01, log2=-6.64, floor=-7.
    assert _recommend_user_bound_scale((1.0, 1e+8), None) == -7


def test_current_user_bound_scale_is_added():
    """The recommendation is delta-on-top of an existing scale."""
    # Same DES inputs, but caller already has user_bound_scale=2.
    # Delta is -6, so result is -4.
    assert _recommend_user_bound_scale(
        (1.0, 1.0), (2e-5, 4e+7), current_user_bound_scale=2
    ) == -4


def test_returned_n_is_always_in_clamp():
    """Sweep of bound/rhs ranges: result stays inside the clamp."""
    for max_v in (1e-30, 1e-10, 1e-4, 1.0, 1e+6, 1e+15, 1e+30):
        for min_v in (1e-30, 1e-15, 1.0):
            if min_v > max_v:
                continue
            n = _recommend_user_bound_scale(None, (min_v, max_v))
            assert USER_BOUND_SCALE_CLAMP_LO <= n <= USER_BOUND_SCALE_CLAMP_HI, (
                f"rhs=({min_v!r}, {max_v!r}) returned N={n}, "
                f"outside the clamp "
                f"[{USER_BOUND_SCALE_CLAMP_LO}, {USER_BOUND_SCALE_CLAMP_HI}]"
            )
