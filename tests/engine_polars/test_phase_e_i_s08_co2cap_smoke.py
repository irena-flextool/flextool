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

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._direct_params import (
    p_co2_max_period_from_source,
)
from flextool.engine_polars._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    Shape,
    _infer_silent_default_labels,
    resolve_param_shape,
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


_RIVENDELL_DB = Path(
    "/home/jkiviluo/sources/flextool-engine/projects/Rivendell/input_sources/rivendell.sqlite"
)


@pytest.fixture(scope="module")
def rivendell_source_s08():
    """SpineDbReader against Rivendell scenario ``S08_co2cap_slice``.

    Skips when the DB isn't checked out (e.g. CI doesn't ship it).
    """
    if not _RIVENDELL_DB.exists():
        pytest.skip(f"Rivendell DB not present at {_RIVENDELL_DB}")
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    return SpineDbReader(f"sqlite:///{_RIVENDELL_DB}", "S08_co2cap_slice")


def test_s08_co2_max_period_has_silent_default_index_name(
    rivendell_source_s08,
) -> None:
    """Sanity-check: the DB authoring really uses the silent default
    ``"x"`` for ``group.co2_max_period``.  If a future DB regen sets
    ``index_name="period"`` explicitly, this test becomes a no-op
    (the resolver path no longer needs the disambiguation step) and
    can be removed alongside the helper.
    """
    info = rivendell_source_s08.parameter_shape_info(
        "group", "co2_max_period")
    assert info == ["x"], (
        f"Rivendell S08 DB Map index_name is {info!r}; "
        "the regression test assumes the silent default 'x'. "
        "If the DB has been re-authored with an explicit 'period' index_name, "
        "remove this assertion (the disambiguation helper is dormant for "
        "explicit labels and the rest of the suite still covers it)."
    )


def test_s08_resolve_param_shape_returns_map_period(
    rivendell_source_s08,
) -> None:
    """The Δ.17c resolver MUST return a non-None ResolvedShape for
    ``("group", "co2_max_period")`` on the S08 scenario, with
    ``shape=MAP_PERIOD`` and a frame whose period column is canonical
    ``"period"`` (not the raw ``"x"``).
    """
    resolved = resolve_param_shape(
        rivendell_source_s08, "group", "co2_max_period")
    assert resolved is not None, (
        "Pre-fix HEAD returned None here — the resolver collapsed the "
        "silent-default index_name to None and bailed out.  Post-fix, "
        "disambiguation against PARAM_ALLOWED_SHAPES[(group, "
        "co2_max_period)] = {SCALAR, MAP_PERIOD} infers MAP_PERIOD."
    )
    assert resolved.shape == Shape.MAP_PERIOD
    assert resolved.period_index_column == "period", (
        f"Frame columns: {resolved.frame.columns}; "
        "expected the silent-default 'x' column to be renamed to 'period'."
    )
    # Frame carries the actual period rows.
    assert "period" in resolved.frame.columns
    assert resolved.frame.height > 0


def test_s08_p_co2_max_period_from_source_is_populated(
    rivendell_source_s08,
) -> None:
    """End-to-end resolver assertion: the consumer site in
    ``apply_direct_params`` must receive a non-None ``Param`` for
    ``p_co2_max_period`` so that ``build_flextool`` doesn't raise on
    the ``co2_max_period`` feature gate.
    """
    # Replicate the period_filter shape the consumer uses (dt frame's
    # unique periods).  For this test we synthesise it from the DB's own
    # period entities — the resolver only looks at columns ``d, t``.
    periods = rivendell_source_s08.entities("period")
    dt = (periods.rename({"name": "d"})
                .with_columns(pl.lit("t000001").alias("t"))
                .select("d", "t"))
    result = p_co2_max_period_from_source(
        rivendell_source_s08, period_filter=dt)
    assert result is not None, (
        "Pre-fix HEAD: returned None → build_flextool raised "
        "'co2_max_period feature active but p_co2_max_period is None'."
    )
    # Param's columns should be (g, d, value).
    frame = result.frame.collect() if hasattr(result.frame, "collect") else result.frame
    assert {"g", "d", "value"}.issubset(set(frame.columns)), (
        f"Param frame columns: {frame.columns}"
    )
    # At least the co2_group is present for at least one in-window period.
    assert frame.height > 0
    assert (frame.select(pl.col("g") == "co2_group").to_series().any())
