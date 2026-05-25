"""Unit tests for ``flextool.engine_polars.scaling``.

Covers the bound-scaling helpers, the row-scaling decision logic that
is still callable in isolation (``maybe_auto_apply_row_scaling``,
``resolve_auto_scale``), and the underlying numerical primitives
(``_family_stats``, ``_pooled_spread_log10``,
``_recommend_scale_the_objective``).

The disk-API entry-point (``analyze_solve(name, input_dir)``) was
retired in Tier 4 Commit 4 along with the runner-side
``flextool.flextoolrunner.scaling`` module.  The replacement
in-memory entry-point (``analyze_solve(solve_name, flex_data, ...)``)
is exercised by the engine-driven tests under
``tests/engine_polars/scaling/``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make flextool importable when running the tests in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flextool.engine_polars.scaling import (
    BOUND_SPREAD_THRESHOLD,
    COST_FAMILIES,
    DEFAULT_OBJECTIVE_SCALE,
    FORCE_USER_BOUND_SCALE_ENV_VAR,
    FamilyStats,
    RHS_FAMILIES,
    ScaleTable,
    USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN,
    _family_stats,
    _pooled_spread_log10,
    _recommend_scale_the_objective,
    apply_bound_scale_decision,
    clear_cache,
    compute_bound_stats,
    decide_user_bound_scale,
    maybe_auto_apply_row_scaling,
    resolve_auto_scale,
    resolve_force_user_bound_scale,
    update_bound_scale_in_cache,
)


# ---------------------------------------------------------------------------
# _family_stats
# ---------------------------------------------------------------------------


def test_family_stats_empty() -> None:
    stats = _family_stats([])
    assert stats.n_values == 0
    assert stats.log10_max is None


def test_family_stats_all_zero() -> None:
    stats = _family_stats([0.0, 0.0, 0.0])
    assert stats.n_values == 3
    assert stats.n_zero == 3
    assert stats.n_nonzero == 0
    assert stats.log10_max is None


def test_family_stats_mixed() -> None:
    stats = _family_stats([1.0, 10.0, 100.0, 1000.0, 0.0])
    assert stats.n_values == 5
    assert stats.n_zero == 1
    assert stats.n_nonzero == 4
    assert stats.log10_min == 0.0
    assert stats.log10_max == 3.0
    # Median of {0, 1, 2, 3} = 1.5 (linear interpolation).
    assert stats.log10_median == 1.5


def test_family_stats_negative_values_kept_by_magnitude() -> None:
    stats = _family_stats([-10.0, -100.0, 10.0])
    # Absolute magnitudes: 10, 10, 100 → log10: 1, 1, 2.
    assert stats.log10_min == 1.0
    assert stats.log10_max == 2.0


# ---------------------------------------------------------------------------
# _recommend_scale_the_objective
# ---------------------------------------------------------------------------


def test_recommend_scale_powers_of_10() -> None:
    # rough_obj ≈ 1e9 → scale = 1e-9
    assert _recommend_scale_the_objective(1e9) == 1e-9
    # rough_obj ≈ 3e7 → log10=7.48 → round=7 → scale = 1e-7
    assert _recommend_scale_the_objective(3e7) == 1e-7
    # rough_obj ≈ 1e6 → scale = 1e-6
    assert _recommend_scale_the_objective(1e6) == 1e-6


def test_recommend_scale_clamped_to_range() -> None:
    # Absurdly large → clamped at 1e-12.
    assert _recommend_scale_the_objective(1e20) == 1e-12
    # Absurdly small positive → clamped at 1e0 on the high side.
    assert _recommend_scale_the_objective(1e-3) == 1.0


def test_recommend_scale_zero_and_negative_fallback() -> None:
    assert _recommend_scale_the_objective(0.0) == DEFAULT_OBJECTIVE_SCALE
    assert _recommend_scale_the_objective(-1.0) == DEFAULT_OBJECTIVE_SCALE


# ---------------------------------------------------------------------------
# _pooled_spread_log10
# ---------------------------------------------------------------------------


def _family_stats_from_bounds(abs_min: float, abs_max: float) -> FamilyStats:
    """Minimal :class:`FamilyStats` with just the fields the pooled-spread
    logic consults (``abs_min`` / ``abs_max``)."""
    return FamilyStats(
        n_values=2,
        n_zero=0,
        n_nonzero=2,
        log10_min=None,  # not used by _pooled_spread_log10
        log10_max=None,
        log10_median=None,
        log10_p10=None,
        log10_p90=None,
        abs_min=abs_min,
        abs_max=abs_max,
        abs_median=None,
    )


def _empty_family_stats() -> FamilyStats:
    return FamilyStats(n_values=0, n_zero=0, n_nonzero=0)


def _synthetic_family_ranges(
    *,
    rhs_abs: dict[str, tuple[float, float]] | None = None,
    cost_abs: dict[str, tuple[float, float]] | None = None,
) -> dict[str, FamilyStats]:
    """Build a ``family_ranges`` dict with just the entries
    :func:`_pooled_spread_log10` consults.  Entries not supplied become
    empty stats.
    """
    ranges: dict[str, FamilyStats] = {}
    rhs_abs = rhs_abs or {}
    for name in RHS_FAMILIES:
        if name in rhs_abs:
            mn, mx = rhs_abs[name]
            ranges[name] = _family_stats_from_bounds(mn, mx)
        else:
            ranges[name] = _empty_family_stats()
    cost_abs = cost_abs or {}
    for name in COST_FAMILIES:
        if name in cost_abs:
            mn, mx = cost_abs[name]
            ranges[name] = _family_stats_from_bounds(mn, mx)
        else:
            ranges[name] = _empty_family_stats()
    return ranges


def test_pooled_spread_handles_empty_and_partial() -> None:
    # Fully empty → 0.0
    ranges = _synthetic_family_ranges()
    assert _pooled_spread_log10(ranges, RHS_FAMILIES) == 0.0
    assert _pooled_spread_log10(ranges, COST_FAMILIES) == 0.0
    # Only one family populated → spread derived from that family alone.
    ranges = _synthetic_family_ranges(
        rhs_abs={"node_inflow": (1e-1, 1e5)}
    )
    assert (
        abs(_pooled_spread_log10(ranges, RHS_FAMILIES) - 6.0) < 1e-9
    )
    # Two families pooled → the widest overall (min of mins, max of maxs).
    ranges = _synthetic_family_ranges(
        rhs_abs={
            "node_inflow": (1.0, 1e4),         # 4 decades
            "node_annual_flow": (1e-3, 1e2),   # 5 decades
        }
    )
    # pooled: overall_min = 1e-3, overall_max = 1e4 → 7 decades.
    assert (
        abs(_pooled_spread_log10(ranges, RHS_FAMILIES) - 7.0) < 1e-9
    )


# ---------------------------------------------------------------------------
# resolve_auto_scale
# ---------------------------------------------------------------------------


def test_resolve_auto_scale_cli_true() -> None:
    assert resolve_auto_scale(True) is True


def test_resolve_auto_scale_env_truthy(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "1")
    assert resolve_auto_scale(False) is True
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "yes")
    assert resolve_auto_scale(False) is True
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "on")
    assert resolve_auto_scale(False) is True


def test_resolve_auto_scale_env_falsy(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "0")
    assert resolve_auto_scale(False) is False
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "")
    assert resolve_auto_scale(False) is False
    monkeypatch.delenv("FLEXTOOL_AUTO_SCALE", raising=False)
    assert resolve_auto_scale(False) is False


# ---------------------------------------------------------------------------
# maybe_auto_apply_row_scaling
# ---------------------------------------------------------------------------


def _fake_table(rec: str) -> ScaleTable:
    return ScaleTable(
        solve_name="s",
        use_row_scaling=rec,  # type: ignore[arg-type]
        scale_the_objective=1e-6,
        family_ranges={
            "entity_unitsize": FamilyStats(n_values=0, n_zero=0, n_nonzero=0)
        },
        unitsize_spread_log10=4.0,
        rough_obj_estimate=1e9,
        timestamp="2026-04-22T00:00:00",
        source_dir="/tmp/fake",
    )


def test_maybe_auto_apply_off_returns_none() -> None:
    assert (
        maybe_auto_apply_row_scaling(
            "s", _fake_table("yes"), user_setting=None, auto_scale=False
        )
        is None
    )


def test_maybe_auto_apply_on_empty_user_applies() -> None:
    # No user setting → apply recommendation.
    assert (
        maybe_auto_apply_row_scaling(
            "s", _fake_table("yes"), user_setting=None, auto_scale=True
        )
        == "yes"
    )
    assert (
        maybe_auto_apply_row_scaling(
            "s", _fake_table("no"), user_setting="", auto_scale=True
        )
        == "no"
    )


def test_maybe_auto_apply_on_respects_user_yes() -> None:
    # User explicitly said yes → analyzer stays out of the way.
    assert (
        maybe_auto_apply_row_scaling(
            "s",
            _fake_table("no"),
            user_setting="yes",
            auto_scale=True,
        )
        is None
    )


def test_maybe_auto_apply_on_respects_user_no() -> None:
    # User explicitly said no → analyzer stays out of the way.
    assert (
        maybe_auto_apply_row_scaling(
            "s",
            _fake_table("yes"),
            user_setting="no",
            auto_scale=True,
        )
        is None
    )


def test_maybe_auto_apply_unrecognised_user_value_applies() -> None:
    # "auto" is not a valid DB value; treat as unset and apply.
    assert (
        maybe_auto_apply_row_scaling(
            "s",
            _fake_table("yes"),
            user_setting="auto",
            auto_scale=True,
        )
        == "yes"
    )


# ---------------------------------------------------------------------------
# Agent 18c — variable bound scaling
# ---------------------------------------------------------------------------


INF = float("inf")
NAN = float("nan")


def test_compute_bound_stats_skips_inf_nan_zero() -> None:
    """Non-finite, NaN, and zero bounds contribute nothing."""
    abs_min, abs_max, spread = compute_bound_stats(
        col_lower=[0.0, -INF, NAN, 10.0],
        col_upper=[INF, 0.0, NAN, 100.0],
    )
    # Only 10 and 100 remain.
    assert abs_min == 10.0
    assert abs_max == 100.0
    assert abs(spread - 1.0) < 1e-9


def test_compute_bound_stats_empty_returns_none() -> None:
    """With nothing finite and non-zero, abs_min/abs_max are None and spread is 0."""
    abs_min, abs_max, spread = compute_bound_stats(
        col_lower=[-INF, 0.0, NAN],
        col_upper=[INF, 0.0, NAN],
    )
    assert abs_min is None
    assert abs_max is None
    assert spread == 0.0


def test_compute_bound_stats_negative_values_via_abs() -> None:
    """Negative bounds use their absolute value for spread."""
    abs_min, abs_max, spread = compute_bound_stats(
        col_lower=[-1000.0, -10.0],
        col_upper=[0.0, -1.0],
    )
    # abs set {1000, 10, 1} → min=1, max=1000, spread=3.
    assert abs_min == 1.0
    assert abs_max == 1000.0
    assert abs(spread - 3.0) < 1e-9


def test_decide_user_bound_scale_below_threshold_returns_zero() -> None:
    """Spread at or below threshold → no scaling."""
    # Exactly 6 decades → still zero (strict >).
    assert decide_user_bound_scale(1e6, 6.0, bound_abs_min=1.0) == 0
    # Under the threshold → zero.
    assert decide_user_bound_scale(1e5, 4.0, bound_abs_min=1.0) == 0
    # No bound info → zero.
    assert decide_user_bound_scale(None, 10.0) == 0
    assert decide_user_bound_scale(0.0, 10.0) == 0


def test_decide_user_bound_scale_rivendell_shape() -> None:
    """Agent 18e: geometric-midpoint on rivendell S19's range.

    Rivendell S19 has ``[2e-3, 1e+6]`` (spread ~8.7 decades).
    geo_mid = sqrt(1e6 * 2e-3) = sqrt(2e3) ≈ 45.
    N = -round(log2(45)) = -round(5.49) = -5.

    HiGHS' own hint is ``-8``; being close-but-softer is the goal.
    """
    n = decide_user_bound_scale(1e6, 9.0, bound_abs_min=2e-3)
    assert n < 0
    assert USER_BOUND_SCALE_MIN <= n <= USER_BOUND_SCALE_MAX
    # geo_mid = sqrt(1e6 * 2e-3) = sqrt(2000) ≈ 44.7 → log2 ≈ 5.48 → N=-5.
    assert n == -5


def test_decide_user_bound_scale_no_abs_min_uses_floor() -> None:
    """Without abs_min, the formula falls back to the 6-decade floor.

    floor_min = abs_max * BOUND_ABS_MIN_FLOOR_RATIO = 1e6 * 1e-6 = 1.
    geo_mid = sqrt(1e6 * 1) = 1000 → N = -round(log2(1000)) = -10 (clamped).
    """
    n = decide_user_bound_scale(1e6, 9.0, bound_abs_min=None)
    # sqrt(1e6 * 1.0) = 1000 → log2 = 9.97 → N=-10 (clamped to MIN).
    assert n == USER_BOUND_SCALE_MIN  # -10


def test_decide_user_bound_scale_clamp_low_on_pathological() -> None:
    """Very wide range ``[1e-30, 1]`` — clamp floors at -10.

    geo_mid = sqrt(1 * 1e-30) = 1e-15 → log2 ≈ -49.8 → N=+50 → clamped to 0
    (positive N not allowed).  But with abs_min respected up to the floor
    (1 * 1e-6 = 1e-6), geo_mid = sqrt(1 * 1e-6) = 1e-3 → log2 = -9.97 →
    N=+10 → clamped to 0.

    This variant: ``[1e-30, 1e20]`` — abs_max large, abs_min tiny:
    floor_min = 1e20 * 1e-6 = 1e14.  geo_mid = sqrt(1e20 * 1e14) = 1e17 →
    N = -56 → clamped to -10.
    """
    n = decide_user_bound_scale(1e20, 50.0, bound_abs_min=1e-30)
    assert n == USER_BOUND_SCALE_MIN  # -10


def test_decide_user_bound_scale_clamp_high() -> None:
    """Small bound_abs_max (below 1) would imply positive N; clamp to 0.

    abs_max=0.1 → floor_min = 1e-7.  geo_mid = sqrt(0.1 * 1e-7) ≈ 1e-4 →
    log2 ≈ -13.3 → N = +13 → clamped to 0.
    """
    n = decide_user_bound_scale(0.1, 8.0, bound_abs_min=1e-4)
    assert n == USER_BOUND_SCALE_MAX  # i.e. 0


def test_decide_user_bound_scale_narrow_bounds_no_scaling() -> None:
    """Narrow bounds ``(1, 1000)`` — spread=3 is below threshold → N=0."""
    # Spread under threshold → zero regardless of the abs_max value.
    assert decide_user_bound_scale(1000.0, 3.0, bound_abs_min=1.0) == 0


def test_decide_user_bound_scale_single_value_bounds() -> None:
    """Single bound value (abs_min == abs_max) doesn't div-by-zero.

    Even though spread==0 triggers the early-return, force a synthetic
    spread > threshold to exercise the formula on identical values.
    geo_mid = sqrt(100 * 100) = 100 → log2 = 6.64 → N = -7.
    """
    n = decide_user_bound_scale(100.0, 10.0, bound_abs_min=100.0)
    # log2(100) = 6.64 → round 7 → N = -7
    assert n == -7


def test_decide_user_bound_scale_all_small_bounds_returns_zero() -> None:
    """All bounds ~1e-3 (spread forced > 6 for test) — N>0 clamped to 0.

    geo_mid = sqrt(1e-3 * 1e-3) = 1e-3 → log2 = -9.97 → N = +10 →
    clamped to USER_BOUND_SCALE_MAX = 0.
    """
    n = decide_user_bound_scale(1e-3, 10.0, bound_abs_min=1e-3)
    assert n == USER_BOUND_SCALE_MAX  # 0


def test_resolve_force_user_bound_scale_unset(monkeypatch) -> None:
    monkeypatch.delenv(FORCE_USER_BOUND_SCALE_ENV_VAR, raising=False)
    assert resolve_force_user_bound_scale() is None


def test_resolve_force_user_bound_scale_valid(monkeypatch) -> None:
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "-8")
    assert resolve_force_user_bound_scale() == -8
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "0")
    assert resolve_force_user_bound_scale() == 0


def test_resolve_force_user_bound_scale_garbage(monkeypatch) -> None:
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "not-a-number")
    assert resolve_force_user_bound_scale() is None
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "")
    assert resolve_force_user_bound_scale() is None


def test_resolve_force_user_bound_scale_clamps(monkeypatch) -> None:
    """Out-of-range values are clamped into [MIN, MAX]."""
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "-9999")
    assert resolve_force_user_bound_scale() == USER_BOUND_SCALE_MIN
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "9999")
    assert resolve_force_user_bound_scale() == USER_BOUND_SCALE_MAX


def test_apply_bound_scale_decision_respects_user_opt_set(monkeypatch) -> None:
    """When the user has already set user_bound_scale via the user's highs.opt,
    the runtime must not override — return 0 regardless of spread."""
    monkeypatch.delenv(FORCE_USER_BOUND_SCALE_ENV_VAR, raising=False)
    n, abs_min, abs_max, spread, source = apply_bound_scale_decision(
        solve_name="s",
        col_lower=[0.0] * 3,
        col_upper=[1e9, 1e6, 1.0],  # wide spread
        auto_scale=True,
        user_opt_set=True,
    )
    assert n == 0
    assert source == "user-opt"
    assert spread > BOUND_SPREAD_THRESHOLD


def test_apply_bound_scale_decision_force_env_wins(monkeypatch) -> None:
    """Force env var overrides both auto_scale and user-opt? No — it
    overrides auto_scale gating, but a user-opt setting still wins.
    Design: user-opt always takes precedence (check the other test)."""
    monkeypatch.setenv(FORCE_USER_BOUND_SCALE_ENV_VAR, "-8")
    n, _, _, _, source = apply_bound_scale_decision(
        solve_name="s",
        col_lower=[0.0] * 2,
        col_upper=[1.0, 10.0],  # narrow spread
        auto_scale=False,       # even when auto_scale is off
        user_opt_set=False,
    )
    assert n == -8
    assert source == "force-env"


def test_apply_bound_scale_decision_auto_off_leaves_zero(monkeypatch) -> None:
    """When auto_scale is off AND no force env → return 0 regardless of spread."""
    monkeypatch.delenv(FORCE_USER_BOUND_SCALE_ENV_VAR, raising=False)
    n, _, _, spread, source = apply_bound_scale_decision(
        solve_name="s",
        col_lower=[0.0, 0.0],
        col_upper=[1e9, 1.0],  # 9-decade spread
        auto_scale=False,
        user_opt_set=False,
    )
    assert n == 0
    assert source == "auto-scale-off"
    assert spread > BOUND_SPREAD_THRESHOLD


def test_apply_bound_scale_decision_auto_on_triggers(monkeypatch) -> None:
    """auto_scale=True + wide spread + no opt-out → analyser picks N."""
    monkeypatch.delenv(FORCE_USER_BOUND_SCALE_ENV_VAR, raising=False)
    n, _, abs_max, spread, source = apply_bound_scale_decision(
        solve_name="s",
        col_lower=[0.0, 0.0],
        col_upper=[1e6, 1.0],  # 6-decade — exactly at threshold → still 0.
        auto_scale=True,
        user_opt_set=False,
    )
    # Threshold is strict >: spread == 6 → no scaling.
    assert n == 0
    assert source == "below-threshold"

    n2, _, _, spread2, source2 = apply_bound_scale_decision(
        solve_name="s2",
        col_lower=[0.0, 0.0],
        col_upper=[1e8, 1.0],  # 8-decade spread
        auto_scale=True,
        user_opt_set=False,
    )
    assert n2 < 0
    assert USER_BOUND_SCALE_MIN <= n2 <= USER_BOUND_SCALE_MAX
    assert source2 == "auto-scale"
    assert spread2 > BOUND_SPREAD_THRESHOLD


def test_update_bound_scale_in_cache_populates_fields() -> None:
    """The post-load cache update mutates the right fields without
    losing the rest of the ScaleTable."""
    clear_cache()
    from flextool.engine_polars.scaling import _scale_cache
    t = ScaleTable(
        solve_name="s",
        use_row_scaling="no",
        scale_the_objective=1e-6,
        family_ranges={},
        unitsize_spread_log10=1.0,
        rough_obj_estimate=1e6,
        timestamp="2026-04-22T00:00:00",
        source_dir="/tmp",
    )
    _scale_cache["s"] = t
    update_bound_scale_in_cache(
        "s", n=-8, abs_min=2e-3, abs_max=1e6, spread_log10=8.70,
    )
    assert t.user_bound_scale == -8
    assert t.bound_abs_min == 2e-3
    assert t.bound_abs_max == 1e6
    assert abs(t.bound_spread_log10 - 8.70) < 1e-9
    # Untouched fields survived.
    assert t.use_row_scaling == "no"
    assert t.scale_the_objective == 1e-6


def test_update_bound_scale_in_cache_no_entry_is_noop() -> None:
    """Calling update for a solve that was never analysed is a no-op,
    not an exception."""
    clear_cache()
    update_bound_scale_in_cache(
        "never-analysed", n=-5, abs_min=1.0, abs_max=1e5, spread_log10=5.0,
    )  # must not raise


def test_scale_table_defaults_bound_fields() -> None:
    """Bound-scale fields default cleanly for freshly-constructed tables."""
    t = ScaleTable(
        solve_name="s",
        use_row_scaling="no",
        scale_the_objective=1e-6,
        family_ranges={},
        unitsize_spread_log10=0.0,
        rough_obj_estimate=0.0,
        timestamp="t",
        source_dir="d",
    )
    assert t.user_bound_scale == 0
    assert t.bound_spread_log10 == 0.0
    assert t.bound_abs_min is None
    assert t.bound_abs_max is None
