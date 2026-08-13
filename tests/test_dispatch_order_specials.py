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


# ---------------------------------------------------------------------------
# special_order: reorder the special slots WITHIN each sign group
#
# ``special_order`` is the dispatch specials' file order (the picker's
# top-to-bottom order of ``categories.dispatch`` keys) = the visual
# top-to-bottom stack order.  Empirically (probe reading area-polygon
# y-extents): positives stack UP from 0 in LIST order, so the visual TOP of the
# positive block is the LAST list column → list order is the REVERSE of visual
# top-to-bottom.  Negatives stack DOWN from 0 in list order, so the visual TOP
# of the negative block (nearest the axis) is the FIRST list column → list
# order EQUALS visual top-to-bottom.  ``_order_dispatch_columns`` must map
# *special_order* accordingly, and default to the historical hardcoded slots
# when it is ``None``.
# ---------------------------------------------------------------------------

# The b285bd63 dialog order (positives POSITIVE_SPECIAL, negatives
# reversed(NEGATIVE_SPECIAL)); this is the default that must reproduce the
# historical hardcoded output byte-for-byte.
_DEFAULT_SPECIAL_ORDER = [
    "LossOfLoad", "Discharge", "Import",
    "internal_losses", "Export", "Charge",
]


def _make_specials_df() -> pd.DataFrame:
    """A frame carrying all six special columns (distinct magnitudes so each
    area polygon is identifiable) and nothing else."""
    idx = pd.Index(range(4), name="time")
    return pd.DataFrame(
        {
            "LossOfLoad": [3.0] * 4,
            "Discharge": [2.0] * 4,
            "Import": [1.0] * 4,
            "internal_losses": [-1.0] * 4,
            "Export": [-2.0] * 4,
            "Charge": [-3.0] * 4,
        },
        index=idx,
    )


def _visual_top_to_bottom(out: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Render *out* and read each area polygon's y-extent, returning
    ``(pos_top_to_bottom, neg_top_to_bottom)`` — the special columns ordered by
    their visual position (highest upper edge first) within each sign block."""
    import numpy as np

    from flextool.scenario_comparison.dispatch_plots import (
        _build_dispatch_figure,
    )

    cols = [c for c in out.columns if c != "Curtailed"]
    fig = _build_dispatch_figure(
        out[cols], None, title="probe", timeline=(0, len(out)),
    )
    ax = fig.axes[0]
    extents = []
    for name, coll in zip(cols, ax.collections):
        ys = coll.get_paths()[0].vertices[:, 1]
        extents.append((name, float(np.min(ys)), float(np.max(ys))))
    pos = [e for e in extents if e[2] > 1e-9]
    neg = [e for e in extents if e[1] < -1e-9]
    pos_ttb = [e[0] for e in sorted(pos, key=lambda e: -e[2])]
    neg_ttb = [e[0] for e in sorted(neg, key=lambda e: -e[2])]
    return pos_ttb, neg_ttb


def test_default_special_order_matches_hardcoded_none_else_branch():
    """PARITY (else branch): special_order == the b285bd63 dialog default
    reproduces the historical config_order=None output byte-for-byte."""
    legacy = _order_dispatch_columns(_make_specials_df(), config_order=None)
    defaulted = _order_dispatch_columns(
        _make_specials_df(),
        config_order=None,
        special_order=list(_DEFAULT_SPECIAL_ORDER),
    )
    assert list(defaulted.columns) == list(legacy.columns)
    # And that historical column order is exactly the documented slots.
    assert list(legacy.columns) == [
        "internal_losses", "Export", "Charge",   # neg, visual top→bottom
        "Import", "Discharge", "LossOfLoad",      # pos, list order (reversed)
    ]


def test_default_special_order_matches_hardcoded_none_config_branch():
    """PARITY (config branch): same byte-for-byte match when a (non-matching)
    entity config_order is also supplied — special slots are independent."""
    cfg = ["some_entity"]
    legacy = _order_dispatch_columns(_make_specials_df(), config_order=cfg)
    defaulted = _order_dispatch_columns(
        _make_specials_df(),
        config_order=cfg,
        special_order=list(_DEFAULT_SPECIAL_ORDER),
    )
    assert list(defaulted.columns) == list(legacy.columns)


def test_default_special_order_renders_historical_visual_stack():
    """The default special_order renders positives top→bottom
    LossOfLoad,Discharge,Import and negatives top→bottom
    internal_losses,Export,Charge (the historical stack)."""
    out = _order_dispatch_columns(
        _make_specials_df(),
        config_order=None,
        special_order=list(_DEFAULT_SPECIAL_ORDER),
    )
    pos_ttb, neg_ttb = _visual_top_to_bottom(out)
    assert pos_ttb == ["LossOfLoad", "Discharge", "Import"]
    assert neg_ttb == ["internal_losses", "Export", "Charge"]


def test_permuting_positive_special_order_permutes_the_stack():
    """Permuting the positives within special_order permutes the positive stack
    (visual top→bottom follows special_order) and never moves a positive below
    the axis."""
    # Put Import at the visual top, then LossOfLoad, then Discharge.
    special = ["Import", "LossOfLoad", "Discharge",
               "internal_losses", "Export", "Charge"]
    out = _order_dispatch_columns(
        _make_specials_df(), config_order=None, special_order=special,
    )
    pos_ttb, neg_ttb = _visual_top_to_bottom(out)
    # Visual top→bottom of the positive block == the special_order positives.
    assert pos_ttb == ["Import", "LossOfLoad", "Discharge"]
    # Negatives untouched.
    assert neg_ttb == ["internal_losses", "Export", "Charge"]
    # Every positive special stays strictly above the axis (present in the
    # positive polygons, never in the negative ones).
    assert set(pos_ttb) == {"Import", "LossOfLoad", "Discharge"}
    assert not (set(pos_ttb) & set(neg_ttb))


def test_permuting_negative_special_order_permutes_the_stack():
    """Permuting the negatives within special_order permutes the negative stack
    (visual top→bottom follows special_order) and never moves a negative above
    the axis."""
    # Put Charge at the visual top (nearest the axis), then Export, then losses.
    special = ["LossOfLoad", "Discharge", "Import",
               "Charge", "Export", "internal_losses"]
    out = _order_dispatch_columns(
        _make_specials_df(), config_order=None, special_order=special,
    )
    pos_ttb, neg_ttb = _visual_top_to_bottom(out)
    assert neg_ttb == ["Charge", "Export", "internal_losses"]
    assert pos_ttb == ["LossOfLoad", "Discharge", "Import"]
    assert not (set(pos_ttb) & set(neg_ttb))


def test_special_order_applies_in_config_branch_column_order():
    """In the config branch the permuted special slots appear in the returned
    column list order (negatives first in list = visual top; positives reversed
    in list = visual top last)."""
    special = ["Import", "Discharge", "LossOfLoad",
               "Charge", "Export", "internal_losses"]
    out = _order_dispatch_columns(
        _make_specials_df(), config_order=["some_entity"], special_order=special,
    )
    cols = list(out.columns)
    # Negatives: list order == visual top→bottom == special_order negatives.
    assert cols[:3] == ["Charge", "Export", "internal_losses"]
    # Positives: list order == reverse of special_order positives.
    assert cols[3:] == ["LossOfLoad", "Discharge", "Import"]


def test_negative_config_bands_follow_picker_list_order():
    """Negative bands must read the same way as the picker list: an entity
    lower in the list sits lower in the plot.  Regression — negatives came out
    upside-down because config_order (one top-to-bottom sequence) was applied
    with the same sort direction as positives, which stack the opposite way."""
    from flextool.plot_outputs.color_template import (
        resolve_dispatch_colors_and_order,
    )
    # Picker/file order top->bottom = [Load_A, Load_B].
    tmpl = {"entities": {"flowGroup": {"Load_A": "#ff0000", "Load_B": "#0000ff"}}}
    _, cfg = resolve_dispatch_colors_and_order(tmpl, ["Load_A", "Load_B"])
    df = pd.DataFrame(
        {"Load_A": [-1.0, -1, -1], "Load_B": [-2.0, -2, -2]},
        index=pd.RangeIndex(3),
    )
    cols = [str(c) for c in _order_dispatch_columns(df, config_order=cfg).columns]
    # Negatives stack down with first-in-list nearest the axis (top), so
    # Load_A (top of the picker list) must precede Load_B.
    assert cols.index("Load_A") < cols.index("Load_B")


def test_positive_config_bands_follow_picker_list_order():
    """Positives already read correctly — guard that the negative fix didn't
    flip them."""
    from flextool.plot_outputs.color_template import (
        resolve_dispatch_colors_and_order,
    )
    tmpl = {"entities": {"flowGroup": {"Gen_A": "#ff0000", "Gen_B": "#0000ff"}}}
    _, cfg = resolve_dispatch_colors_and_order(tmpl, ["Gen_A", "Gen_B"])
    df = pd.DataFrame(
        {"Gen_A": [1.0, 1, 1], "Gen_B": [2.0, 2, 2]}, index=pd.RangeIndex(3),
    )
    cols = [str(c) for c in _order_dispatch_columns(df, config_order=cfg).columns]
    # Positives stack up with last-in-list at the top, so Gen_A (top of list)
    # must come AFTER Gen_B in list order (nearest-axis = bottom = last drawn).
    assert cols.index("Gen_B") < cols.index("Gen_A")
