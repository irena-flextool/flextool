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


# ---------------------------------------------------------------------------
# Mixed-sign entities hug the zero axis
#
# A bidirectional flow is split into ``<base>_pos`` / ``<base>_neg`` halves.
# We require both halves to sit against the zero axis so the band straddles
# zero and its two excursions read clearly.  In pandas' stacked area the
# FIRST column of each sign (in the returned column order) sits nearest the
# axis: positives stack upward from 0 in list order, negatives downward from
# 0 in list order.  So "nearest zero" for a sign == first occurrence of that
# sign in the returned column list.  (Empirically verified: with column order
# [A+, B+, A-, B-] the polygons span A+:0..1, B+:1..3, A-:-1..0, B-:-3..-1.)
# ---------------------------------------------------------------------------


def _make_mixed_df() -> pd.DataFrame:
    """One mixed entity (battery) plus a pure-positive (wind), a pure-negative
    regular (load), a negative special (Charge) and a positive special
    (LossOfLoad)."""
    idx = pd.Index([1, 2, 3], name="time")
    return pd.DataFrame(
        {
            "wind": [10.0, 5.0, 8.0],       # pure positive
            "load": [-2.0, -3.0, -1.0],     # pure negative, regular
            "battery": [2.0, -1.0, 3.0],    # mixed → split
            "Charge": [-1.0, -2.0, -1.0],   # negative special
            "LossOfLoad": [0.0, 1.0, 0.0],  # positive special
        },
        index=idx,
    )


def _neg_pos_split(out: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (negatives, positives) in returned column order (excluding the
    ``Curtailed`` line overlay).  After splitting, every stacked column is
    single-signed."""
    cols = [c for c in out.columns if c != "Curtailed"]
    negs = [c for c in cols if (out[c] <= 0).all()]
    poss = [c for c in cols if (out[c] >= 0).all()]
    return negs, poss


def _assert_battery_hugs_axis(out: pd.DataFrame) -> None:
    negs, poss = _neg_pos_split(out)
    # Nearest zero == first in that sign's list order.  Battery's negative half
    # is the last negative stacked before the zero axis (topmost of the neg
    # stack); its positive half is the first positive after it (bottom of the
    # pos stack).  Both adjacent to zero.
    assert negs[0] == "battery_neg", negs
    assert poss[0] == "battery_pos", poss
    # Specials stay furthest from the axis (last in their sign's list).
    assert negs[-1] == "Charge", negs
    assert poss[-1] == "LossOfLoad", poss
    # And in the flat returned order the positive half is the first positive
    # right after the negative-to-positive boundary.
    cols = [c for c in out.columns if c != "Curtailed"]
    assert cols.index("battery_pos") == cols.index("Charge") + 1


def test_mixed_entity_hugs_axis_config_branch():
    out = _order_dispatch_columns(_make_mixed_df(), config_order=["wind"])
    _assert_battery_hugs_axis(out)


def test_mixed_entity_hugs_axis_else_branch():
    out = _order_dispatch_columns(_make_mixed_df(), config_order=None)
    _assert_battery_hugs_axis(out)


def _make_two_mixed_df() -> pd.DataFrame:
    """Two mixed entities plus a pure-positive and a pure-negative regular."""
    idx = pd.Index([1, 2, 3], name="time")
    return pd.DataFrame(
        {
            "wind": [10.0, 5.0, 8.0],
            "load": [-2.0, -3.0, -1.0],
            "storA": [2.0, -1.0, 3.0],   # mixed
            "storB": [5.0, -4.0, 1.0],   # mixed
        },
        index=idx,
    )


def test_two_mixed_entities_innermost_matches_both_sides_else_branch():
    out = _order_dispatch_columns(_make_two_mixed_df(), config_order=None)
    negs, poss = _neg_pos_split(out)
    # The two nearest-zero columns on each side are the two mixed halves.
    assert set(negs[:2]) == {"storA_neg", "storB_neg"}, negs
    assert set(poss[:2]) == {"storA_pos", "storB_pos"}, poss
    # Innermost (nearest-zero) entity is the SAME on both sides.
    assert negs[0].removesuffix("_neg") == poss[0].removesuffix("_pos")
    # And the second-nearest entity also matches across signs.
    assert negs[1].removesuffix("_neg") == poss[1].removesuffix("_pos")


def test_two_mixed_entities_innermost_matches_both_sides_config_branch():
    # Config lists storB first → storB is the config-preferred mixed entity and
    # sits nearest zero on both sides (config position wins among mixed).
    out = _order_dispatch_columns(
        _make_two_mixed_df(), config_order=["storB", "storA", "wind"]
    )
    negs, poss = _neg_pos_split(out)
    assert set(negs[:2]) == {"storA_neg", "storB_neg"}, negs
    assert set(poss[:2]) == {"storA_pos", "storB_pos"}, poss
    assert negs[0] == "storB_neg", negs
    assert poss[0] == "storB_pos", poss
    assert negs[1].removesuffix("_neg") == poss[1].removesuffix("_pos")
