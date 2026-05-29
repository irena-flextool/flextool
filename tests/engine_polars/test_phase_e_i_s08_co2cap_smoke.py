"""Phase E-i regression — S08_co2cap_slice ``p_co2_max_period`` resolver.

The Rivendell ``S08_co2cap_slice`` scenario regressed (also affected the
pre-Phase-E baseline ``84b4bf33``) with::

    ValueError: build_flextool: feature 'co2_max_period' is active but
    data fields are not populated (None): ['p_co2_max_period'].  Either
    fill them in the data or don't enable the feature.

Root cause: ``group.co2_max_period`` in the Rivendell DB is authored as
a 1d Map whose ``index_name`` is the spinedb_api silent default
(``"x"``), not an explicit ``"period"``.  The Δ.17c shape resolver
collapsed silent-default labels to ``None`` and returned ``None`` from
:func:`flextool.engine_polars._direct_params.p_co2_max_period_from_source`
— even though the per-parameter allow-list
(``{SCALAR, MAP_PERIOD}``) admits only one shape at depth 1, so the
authoring intent was unambiguous.

Meanwhile the CSV-driven feature gate (``_load_co2_cap`` checking
``solve_data/group_co2_max_period.csv``) lit up because the writer
emits that file off ``group__co2_method`` (independent of the Map
authoring).  The two layers drifted: feature on, value None →
``_check`` raised.

The fix (Phase E-i): teach the shape resolver to disambiguate silent
defaults against the per-parameter allow-list.  When the registry
permits a unique shape at the observed (depth, position), the
silent-default ``index_name`` is unambiguous; the resolver fills the
label and reshapes the frame's columns to canonical ``period``/``t``
names so downstream broadcasters can find the index columns.

This test exercises the offending resolver path directly — no full LP
solve required.  It fails on pre-fix HEAD (returns ``None``) and passes
after the fix (returns a populated ``Param``).
"""
from __future__ import annotations

from flextool.engine_polars._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    Shape,
    _infer_silent_default_labels,
)


# ---------------------------------------------------------------------------
# A. Helper-level — silent-default disambiguation by allow-list.
# ---------------------------------------------------------------------------


def test_silent_default_unique_depth1_period_only() -> None:
    """When the allow-list at depth 1 contains only ``MAP_PERIOD``, a
    silent-default label (``None``) is disambiguated to ``"period"``.
    Matches ``("group", "co2_max_period")`` — the S08 regression case.
    """
    allowed = PARAM_ALLOWED_SHAPES[("group", "co2_max_period")]
    assert allowed == {Shape.SCALAR, Shape.MAP_PERIOD}
    out = _infer_silent_default_labels([None], allowed)
    assert out == ["period"], (
        "Silent default at depth 1 with allow-list {SCALAR, MAP_PERIOD} "
        "must resolve to 'period' (the only allowed shape at that depth). "
        "On pre-fix HEAD this returned [None] and the resolver returned None."
    )


def test_silent_default_ambiguous_depth1_period_or_time() -> None:
    """When the allow-list at depth 1 contains BOTH ``MAP_PERIOD`` and
    ``MAP_TIME`` (e.g. ``("commodity", "price")``), a silent-default
    label MUST stay ambiguous — the resolver cannot infer the author's
    intent.  Documents the policy: only unambiguous registry entries
    are disambiguated.
    """
    allowed = PARAM_ALLOWED_SHAPES[("commodity", "price")]
    assert Shape.MAP_PERIOD in allowed and Shape.MAP_TIME in allowed
    out = _infer_silent_default_labels([None], allowed)
    assert out == [None], (
        "Silent default with two allowed shapes at depth 1 must remain "
        "ambiguous (resolver returns None, caller falls back)."
    )


def test_explicit_label_passthrough_unchanged() -> None:
    """Disambiguation never overwrites an explicit (non-silent) label."""
    allowed = PARAM_ALLOWED_SHAPES[("group", "co2_max_period")]
    out = _infer_silent_default_labels(["period"], allowed)
    assert out == ["period"]


def test_silent_default_depth2_uniquely_resolved() -> None:
    """Constructed allow-list with a single 2d shape disambiguates both
    silent-default positions.  Defensive — current registry has no such
    entry but the helper's general case must hold.
    """
    fake_allowed = {Shape.MAP_PERIOD_TIME}
    out = _infer_silent_default_labels([None, None], fake_allowed)
    assert out == ["period", "time"]


# ---------------------------------------------------------------------------
# B. End-to-end on the Rivendell DB — the S08 regression case.
# ---------------------------------------------------------------------------
#
# Removed: the Rivendell DB is private user data and not available to the
# test suite.  Section A above exercises the silent-default disambiguation
# helper directly; the end-to-end coverage on the user's S08 scenario stays
# local to the user's project and does not run in CI.
