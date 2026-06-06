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
