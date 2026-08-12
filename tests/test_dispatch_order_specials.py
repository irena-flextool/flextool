"""Tests for dispatch column ordering with plot-settings config_order.

Stage 4.1 routes dispatch stacking order through
``resolve_dispatch_colors_and_order`` (entity columns only).  Special tokens
(POSITIVE_SPECIAL / NEGATIVE_SPECIAL) are deliberately left OUT of
``config_order`` so they keep their pipeline-fixed positions.  These tests
pin two invariants of ``_order_dispatch_columns``:

1. A ``config_order`` that matches no column produces byte-identical output
   to the legacy ``config_order=None`` (else) branch — so a project with no
   matching dispatch entities renders exactly as before.
2. Special tokens stay pinned (negatives bottom, positives top) even when a
   non-empty ``config_order`` (entity names) is supplied.
"""

from __future__ import annotations

import pandas as pd

from flextool.scenario_comparison.dispatch_data import _order_dispatch_columns


def _make_df() -> pd.DataFrame:
    idx = pd.Index([1, 2, 3], name="time")
    return pd.DataFrame(
        {
            "wind": [10.0, 5.0, 8.0],
            "coal": [3.0, 4.0, 2.0],
            "battery": [2.0, -1.0, 3.0],   # mixed → split
            "LossOfLoad": [0.0, 1.0, 0.0],
            "Charge": [-1.0, -2.0, -1.0],
            "internal_losses": [-0.5, -0.4, -0.3],
            "Curtailed": [0.1, 0.2, 0.0],
        },
        index=idx,
    )


def test_nonmatching_config_order_matches_else_branch():
    legacy = _order_dispatch_columns(_make_df(), config_order=None)
    nomatch = _order_dispatch_columns(
        _make_df(), config_order=["nonexistent_a", "nonexistent_b"]
    )
    assert list(legacy.columns) == list(nomatch.columns)


def test_specials_stay_pinned_with_entity_config_order():
    out = _order_dispatch_columns(_make_df(), config_order=["coal", "wind"])
    cols = list(out.columns)
    # Negative specials sit at the very bottom (after any split neg part),
    # positive specials at the very top (before the Curtailed line overlay).
    assert cols.index("internal_losses") < cols.index("coal")
    assert cols.index("Charge") < cols.index("coal")
    assert cols.index("LossOfLoad") == len(cols) - 2  # just before Curtailed
    assert cols[-1] == "Curtailed"
    # Entity order honored: coal before wind (config order), between specials.
    assert cols.index("coal") < cols.index("wind")


def _make_tie_df(col_order: list[str]) -> pd.DataFrame:
    """DataFrame with two positive and two negative unlisted columns whose
    within-sign std devs are exactly equal, built in *col_order*.

    ``pos_a`` / ``pos_b`` share the same values (equal std dev), as do
    ``neg_a`` / ``neg_b``; none appear in any config so they land in the
    std-dev "remaining" buckets where the tie-break must decide their order.
    """
    idx = pd.Index([1, 2, 3], name="time")
    data = {
        "pos_a": [1.0, 3.0, 2.0],
        "pos_b": [1.0, 3.0, 2.0],   # identical → equal std dev to pos_a
        "neg_a": [-1.0, -3.0, -2.0],
        "neg_b": [-1.0, -3.0, -2.0],  # identical → equal std dev to neg_a
    }
    return pd.DataFrame({c: data[c] for c in col_order}, index=idx)


def test_equal_std_ties_resolve_by_name_config_branch():
    """Unlisted columns with equal std dev order by name, independent of the
    input column order (config_order branch)."""
    cfg = ["something_else"]  # matches nothing → all fall to "remaining"
    forward = _order_dispatch_columns(
        _make_tie_df(["pos_a", "pos_b", "neg_a", "neg_b"]), config_order=cfg
    )
    reverse = _order_dispatch_columns(
        _make_tie_df(["neg_b", "neg_a", "pos_b", "pos_a"]), config_order=cfg
    )
    assert list(forward.columns) == list(reverse.columns)
    cols = list(forward.columns)
    # Name-sorted within each sign bucket.
    assert cols.index("pos_a") < cols.index("pos_b")
    assert cols.index("neg_a") < cols.index("neg_b")


def test_equal_std_ties_resolve_by_name_else_branch():
    """Same deterministic tie-break in the config-less (else) branch."""
    forward = _order_dispatch_columns(
        _make_tie_df(["pos_a", "pos_b", "neg_a", "neg_b"]), config_order=None
    )
    reverse = _order_dispatch_columns(
        _make_tie_df(["neg_b", "neg_a", "pos_b", "pos_a"]), config_order=None
    )
    assert list(forward.columns) == list(reverse.columns)
    cols = list(forward.columns)
    assert cols.index("pos_a") < cols.index("pos_b")
    assert cols.index("neg_a") < cols.index("neg_b")
