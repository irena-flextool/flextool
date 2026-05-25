"""Unit tests for ``flextool.engine_polars.scaling_report``.

Focused on the symbol-level diagnostics that remain callable in
isolation after the disk-API retirement in Tier 4 Commit 4:

1. Bimodal detector on synthetic ``log10`` distributions — tight
   clusters and uniform distributions must NOT be flagged; clean
   two-cluster inputs MUST be flagged.
2. The locked composite-scale-mismatch recommendation template and
   its renderer (using a ``MismatchPair`` constructed directly,
   bypassing the disk-driven ``find_composite_mismatches``).

The disk-API tests that drove ``find_composite_mismatches(input_dir)``,
``parse_highs_log(...)``, ``write_scaling_report(... input_dir=...,
highs_log_path=...)`` and ``_collect_family_log10_values(...)`` were
removed when the runner-side module was deleted; the in-memory
report is exercised by ``tests/engine_polars/scaling/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flextool.engine_polars.scaling_report import (
    BIMODAL_GAP_DECADES,
    MISMATCH_RECOMMENDATION,
    BimodalSplit,
    MismatchPair,
    _format_mismatch_recommendation,
    detect_bimodal,
)


# ---------------------------------------------------------------------------
# Bimodal detector
# ---------------------------------------------------------------------------


def test_bimodal_flat_distribution_not_bimodal() -> None:
    """A tight unimodal cluster must not be flagged."""
    log10_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert detect_bimodal(log10_vals) is None


def test_bimodal_uniform_wide_not_bimodal() -> None:
    """A uniformly-spread distribution with no clear gap is not bimodal.

    10 values evenly spaced over 5 decades — largest adjacent gap is
    0.55, well under the 2-decade threshold.
    """
    log10_vals = [i * 5 / 9 for i in range(10)]
    assert detect_bimodal(log10_vals) is None


def test_bimodal_clean_two_clusters() -> None:
    """Two tight clusters separated by 4 decades must be flagged."""
    lower = [-2.0, -2.05, -1.95, -2.1, -1.9]
    upper = [3.0, 3.1, 2.9, 3.2, 2.8]
    split = detect_bimodal(lower + upper)
    assert isinstance(split, BimodalSplit)
    assert split.gap_decades > BIMODAL_GAP_DECADES
    assert split.n_lower == 5
    assert split.n_upper == 5
    assert split.lower_share == 0.5
    assert split.upper_share == 0.5
    assert split.lower_center_log10 < 0
    assert split.upper_center_log10 > 0


def test_bimodal_too_small_minority_not_flagged() -> None:
    """A lone outlier (<10% of values) must not trigger bimodal flagging."""
    main = [0.0] * 20
    outlier = [5.0]  # 1/21 ~ 4.8% < 10%
    assert detect_bimodal(main + outlier) is None


def test_bimodal_minimum_size() -> None:
    """Detector needs at least 4 values to bother computing a share."""
    assert detect_bimodal([0.0, 5.0]) is None
    assert detect_bimodal([0.0, 0.1, 5.0]) is None


# ---------------------------------------------------------------------------
# Mismatch recommendation template
# ---------------------------------------------------------------------------


def test_recommendation_text_has_both_options() -> None:
    """The locked recommendation template must include aggregate+sequential."""
    assert "Aggregate the small-side units" in MISMATCH_RECOMMENDATION
    assert "sequential models" in MISMATCH_RECOMMENDATION
    assert "{node}" in MISMATCH_RECOMMENDATION
    assert "{small_entity}" in MISMATCH_RECOMMENDATION
    assert "{large_entity}" in MISMATCH_RECOMMENDATION


def test_recommendation_rendered_ascii_only() -> None:
    """Rendered recommendation must be ASCII (no unicode)."""
    m = MismatchPair(
        process="tiny_unit",
        node="west",
        role="source",
        process_unitsize=0.01,
        node_unitsize=1000.0,
        ratio=100000.0,
        small_entity="tiny_unit",
        small_size=0.01,
        large_entity="mega_unit",
        large_size=1000.0,
    )
    text = _format_mismatch_recommendation(m)
    text.encode("ascii")  # must not raise
    assert "west" in text
    assert "tiny_unit" in text
    assert "mega_unit" in text
