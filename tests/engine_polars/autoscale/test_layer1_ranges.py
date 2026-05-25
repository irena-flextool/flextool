"""Layer 1 (detect) self-test.

Constructs a small LP with hand-picked coefficient magnitudes spanning
twelve decades and verifies that :func:`compute_ranges` recovers the
four ranges plus the cross-group ratio, and that ``trigger`` flips at
the documented 9-decade threshold.

The LP is built three ways to exercise every Layer 1 entry point:

* :func:`ranges_from_arrays` — the low-level kernel.
* :func:`ranges_from_streamed` — the polar-high ``Solution`` adapter,
  fed a synthetic dict mirroring ``Solution.streamed_lp_ranges``.
* :func:`compute_ranges` end-to-end on a small polar-high ``Problem``
  (pre-solve, exercising the ``_build_lp_arrays`` fallback path).

The three reports must agree bit-for-bit — same magnitude reduction,
same trigger.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from flextool.engine_polars.autoscale import (
    AutoScaleConfig,
    RangeReport,
    compute_ranges,
    ranges_from_arrays,
    ranges_from_streamed,
)


def _config(threshold: float = 9.0) -> AutoScaleConfig:
    return AutoScaleConfig(
        enabled=True,
        threshold_decades=threshold,
        user_bound_scale=None,
        report_yaml_path=None,
    )


def test_ranges_from_arrays_twelve_decade_span() -> None:
    """A hand-built LP with magnitudes 1e-6 … 1e6 must report the full
    twelve-decade spread and trigger at the default 9-decade threshold."""
    # Matrix: |a| ∈ {1e-6, 1.0, 1e3, 1e6}.
    matrix = np.array([1e-6, 1.0, 1e3, 1e6, 0.0, np.inf, np.nan], dtype=np.float64)
    # Cost: |c| ∈ {1e-3, 1e2}; cost ratio = 1e5.
    cost = np.array([1e-3, 1e2, 0.0], dtype=np.float64)
    # Column bounds: ±inf and 0 must be filtered; finite non-zeros are
    # {2.0, 5e4}; bound ratio = 2.5e4.
    col_lower = np.array([0.0, -np.inf, -2.0], dtype=np.float64)
    col_upper = np.array([np.inf, 5e4, np.inf], dtype=np.float64)
    # Row bounds: finite non-zeros are {10.0, 1e4}; rhs ratio = 1e3.
    row_lower = np.array([-np.inf, 10.0], dtype=np.float64)
    row_upper = np.array([1e4, np.inf], dtype=np.float64)

    report = ranges_from_arrays(
        matrix_values=matrix,
        cost=cost,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        config=_config(),
    )

    assert report.matrix == pytest.approx((1e-6, 1e6))
    assert report.cost == pytest.approx((1e-3, 1e2))
    assert report.bound == pytest.approx((2.0, 5e4))
    assert report.rhs == pytest.approx((10.0, 1e4))
    # Cross-group ratio is max(his) / min(los) = 1e6 / 1e-6 = 1e12.
    assert report.cross_group_max_ratio == pytest.approx(1e12)
    # Matrix alone (1e12) > 1e9, so trigger fires.
    assert report.trigger is True


def test_ranges_from_arrays_below_threshold_does_not_trigger() -> None:
    """A 3-decade-spread LP must NOT trigger at 9 decades."""
    report = ranges_from_arrays(
        matrix_values=np.array([1.0, 10.0, 100.0]),
        cost=np.array([1.0, 5.0]),
        col_lower=np.array([1.0]),
        col_upper=np.array([100.0]),
        row_lower=np.array([1.0]),
        row_upper=np.array([1000.0]),
        config=_config(),
    )
    assert report.trigger is False
    # Cross-group ratio = 1000 / 1.0 = 1e3.
    assert report.cross_group_max_ratio == pytest.approx(1e3)


def test_ranges_from_arrays_empty_group_returns_nan_pair() -> None:
    """A group with no finite non-zero entries must report ``(nan, nan)``
    and be excluded from the cross-group ratio."""
    report = ranges_from_arrays(
        matrix_values=np.array([1.0]),
        cost=np.array([0.0, np.inf]),  # all-zero / all-inf → empty
        col_lower=np.array([1.0]),
        col_upper=np.array([10.0]),
        row_lower=np.array([2.0]),
        row_upper=np.array([20.0]),
        config=_config(),
    )
    assert math.isnan(report.cost[0]) and math.isnan(report.cost[1])
    # Cross-group ignores cost; max=20, min=1 → 20.
    assert report.cross_group_max_ratio == pytest.approx(20.0)
    assert report.trigger is False


def test_ranges_from_streamed_matches_arrays() -> None:
    """The streamed-ranges adapter must produce the same report as the
    low-level kernel when given equivalent inputs."""
    streamed = {
        "matrix": (1e-6, 1e6),
        "cost": (1e-3, 1e2),
        "col_bound": (2.0, 5e4),
        "row_bound": (10.0, 1e4),
    }
    via_adapter = ranges_from_streamed(streamed, _config())
    via_arrays = ranges_from_arrays(
        matrix_values=np.array([1e-6, 1e6]),
        cost=np.array([1e-3, 1e2]),
        col_lower=np.array([2.0]),
        col_upper=np.array([5e4]),
        row_lower=np.array([10.0]),
        row_upper=np.array([1e4]),
        config=_config(),
    )
    assert via_adapter == via_arrays


def test_ranges_from_streamed_handles_none_categories() -> None:
    """``None`` entries (polar-high's "no finite non-zero" sentinel)
    must map to ``(nan, nan)`` and be excluded from the cross-group
    ratio."""
    streamed = {
        "matrix": (1.0, 10.0),
        "cost": None,
        "col_bound": None,
        "row_bound": (1.0, 1000.0),
    }
    report = ranges_from_streamed(streamed, _config())
    assert math.isnan(report.cost[0])
    assert math.isnan(report.bound[0])
    assert report.cross_group_max_ratio == pytest.approx(1000.0)
    assert report.trigger is False


def test_threshold_decades_controls_trigger() -> None:
    """Trigger must fire at threshold = N when cross-group spread > 10**N."""
    arrays = dict(
        matrix_values=np.array([1.0]),
        cost=np.array([1.0]),
        col_lower=np.array([1.0]),
        col_upper=np.array([1e4]),
        row_lower=np.array([1.0]),
        row_upper=np.array([1.0]),
    )
    # spread = 1e4 → trigger at threshold=3, not at threshold=5.
    fires = ranges_from_arrays(**arrays, config=_config(3.0))
    doesnt = ranges_from_arrays(**arrays, config=_config(5.0))
    assert fires.trigger is True
    assert doesnt.trigger is False


def test_compute_ranges_on_polar_high_problem() -> None:
    """End-to-end: build a tiny ``polar_high.Problem``, run Layer 1's
    pre-solve fallback path, and verify the four ranges + trigger.

    The LP:
        minimize  1e-3 x  +  1e2 y  +  1.0 z
        subject to
            1e-6 x  +   1.0 y  +  1e3 z  <=  1e4
            1e6 x   +   1.0 y               >=  10.0
            x ∈ [0, 5e4],  y ∈ [-2.0, inf),  z ∈ [0, inf)

    Expected ranges:
        Matrix |a| ∈ {1e-6, 1.0, 1e3, 1e6}      → (1e-6, 1e6)
        Cost   |c| ∈ {1e-3, 1e2, 1.0}           → (1e-3, 1e2)
        Bound  finite non-zero ∈ {2.0, 5e4}     → (2.0, 5e4)
        RHS    finite non-zero ∈ {10.0, 1e4}    → (10.0, 1e4)
        Cross-group: 1e6 / 1e-6 = 1e12 > 1e9 → trigger.
    """
    from polar_high.engine import Problem

    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=5e4)
    y = pb.add_var("y", "i", idx, lower=-2.0, upper=float("inf"))
    z = pb.add_var("z", "i", idx, lower=0.0, upper=float("inf"))

    pb.set_objective(1e-3 * x + 1e2 * y + 1.0 * z, sense="min")
    pb.add_cstr(
        "c1",
        over=idx,
        sense="<=",
        lhs_terms={"x": 1e-6 * x, "y": 1.0 * y, "z": 1e3 * z},
        rhs_terms={"k": 1e4},
    )
    pb.add_cstr(
        "c2",
        over=idx,
        sense=">=",
        lhs_terms={"x": 1e6 * x, "y": 1.0 * y},
        rhs_terms={"k": 10.0},
    )

    report = compute_ranges(pb, _config())
    assert report.matrix == pytest.approx((1e-6, 1e6))
    assert report.cost == pytest.approx((1e-3, 1e2))
    assert report.bound == pytest.approx((2.0, 5e4))
    assert report.rhs == pytest.approx((10.0, 1e4))
    assert report.cross_group_max_ratio == pytest.approx(1e12)
    assert report.trigger is True


def test_compute_ranges_rejects_unrecognised_input() -> None:
    """Passing something that's neither a Problem nor a Solution must
    raise — silently degrading would hide wiring bugs."""
    with pytest.raises(TypeError):
        compute_ranges(object(), _config())
