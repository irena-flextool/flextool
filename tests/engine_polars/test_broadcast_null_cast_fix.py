"""Failed-cast guard for :func:`broadcast_to_period_time` (spec
``fix_map_reading.md`` §3.4).

``broadcast_to_period_time`` casts the period / time axis columns against
the model's Enum vocabulary with ``strict=False``, so a token that is NOT
in the vocabulary (an unknown or inactive period / timestep) silently
becomes ``null``.  Before the fix, the downstream null-pattern split read
that ``null`` as a "missing axis" default and broadcast the row across the
WHOLE axis — producing duplicate / wrong ``(e, d, t)`` keys (the reported
repro: a ``MAP_TIME`` row for ``t01`` ended up carrying both 50 and 60).

The fix captures the per-row null state BEFORE the cast, drops the rows
whose token failed the cast (non-null before, null after), and drives the
split off the pre-cast masks.  For data whose tokens are all valid/active
(the normal case) nothing is dropped and the result is byte-for-byte
unchanged.

These tests construct ``ResolvedShape`` frames directly and call the
broadcaster under an active global axis-enum vocabulary (the production
path — that is when the strict=False cast runs).
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars._axis_enums import (
    reset_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine_polars._param_shapes import (
    ResolvedShape,
    Shape,
    broadcast_to_period_time,
)

# Two active periods × two active timesteps — the model's full vocabulary.
_PERIODS = ["p2025", "p2030"]
_TIMES = ["t01", "t02"]
_D_ENUM = pl.Enum(_PERIODS)
_T_ENUM = pl.Enum(_TIMES)


@pytest.fixture()
def axis_enums():
    """Activate the cascade-wide axis-enum vocabulary (production path)."""
    token = set_global_axis_enums({"d": _D_ENUM, "t": _T_ENUM})
    try:
        yield
    finally:
        reset_global_axis_enums(token)


def _dt() -> pl.DataFrame:
    """The active-solve (d, t) grid, carrying Enum dtypes as the cascade
    does after ``apply_derived_a`` runs."""
    return pl.DataFrame(
        {
            "d": pl.Series(
                [p for p in _PERIODS for _ in _TIMES], dtype=_D_ENUM),
            "t": pl.Series(_TIMES * len(_PERIODS), dtype=_T_ENUM),
        }
    )


def _collect(param) -> pl.DataFrame:
    frame = param.frame
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    # Normalise dim dtypes to Utf8 so assertions compare on labels.
    casts = [pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value"]
    return frame.with_columns(casts)


def _assert_no_dupes(df: pl.DataFrame, keys: list[str]) -> None:
    assert df.height == df.select(keys).unique().height, (
        f"duplicate {keys} keys:\n{df}")


# ---------------------------------------------------------------------------
# Bug: an unknown / inactive token must NOT be broadcast across the axis.
# ---------------------------------------------------------------------------


def test_map_time_unknown_token_not_broadcast(axis_enums):
    """A MAP_TIME row with an unknown timestep token must be dropped, not
    cross-joined across every active timestep (the reported repro where
    (b, t01) carried both 50 and 60)."""
    frame = pl.DataFrame(
        {
            "name": ["fg", "fg"],
            "t": ["t01", "UNKNOWN_T"],
            "value": [50.0, 60.0],
        }
    )
    resolved = ResolvedShape(
        shape=Shape.MAP_TIME,
        frame=frame,
        entity_dim_columns=("name",),
        period_index_column=None,
        time_index_column="t",
    )
    out = _collect(broadcast_to_period_time(resolved, "group", _dt()))
    _assert_no_dupes(out, ["group", "t"])
    # The unknown token's value never appears, and the one valid row is
    # kept exactly once (not spread across t02).
    assert 60.0 not in out["value"].to_list()
    assert sorted(out["t"].to_list()) == ["t01"]
    assert out.sort("t").to_dicts() == [
        {"group": "fg", "t": "t01", "value": 50.0}]


def test_map_period_unknown_token_not_broadcast(axis_enums):
    """Same guard for the MAP_PERIOD null-index branch."""
    frame = pl.DataFrame(
        {
            "name": ["fg", "fg"],
            "period": ["p2025", "UNKNOWN_P"],
            "value": [50.0, 60.0],
        }
    )
    resolved = ResolvedShape(
        shape=Shape.MAP_PERIOD,
        frame=frame,
        entity_dim_columns=("name",),
        period_index_column="period",
        time_index_column=None,
    )
    out = _collect(broadcast_to_period_time(resolved, "group", _dt()))
    _assert_no_dupes(out, ["group", "d"])
    assert 60.0 not in out["value"].to_list()
    assert sorted(out["d"].to_list()) == ["p2025"]
    assert out.sort("d").to_dicts() == [
        {"group": "fg", "d": "p2025", "value": 50.0}]


def test_map_period_time_unknown_token_not_broadcast(axis_enums):
    """MAP_PERIOD_TIME four-way split: a row whose period token is unknown
    must not fall into the time-only default branch and be broadcast
    across every active period (which created the duplicate (d, t) key)."""
    frame = pl.DataFrame(
        {
            "name": ["fg", "fg"],
            "period": ["p2025", "UNKNOWN_P"],
            "t": ["t01", "t01"],
            "value": [50.0, 60.0],
        }
    )
    resolved = ResolvedShape(
        shape=Shape.MAP_PERIOD_TIME,
        frame=frame,
        entity_dim_columns=("name",),
        period_index_column="period",
        time_index_column="t",
    )
    out = _collect(broadcast_to_period_time(resolved, "group", _dt()))
    _assert_no_dupes(out, ["group", "d", "t"])
    assert 60.0 not in out["value"].to_list()
    assert out.sort(["d", "t"]).to_dicts() == [
        {"group": "fg", "d": "p2025", "t": "t01", "value": 50.0}]


def test_map_period_time_unknown_time_token_not_broadcast(axis_enums):
    """Mirror of the above, but the unknown token is on the time axis — it
    must not fall into the period-only default branch."""
    frame = pl.DataFrame(
        {
            "name": ["fg", "fg"],
            "period": ["p2025", "p2025"],
            "t": ["t01", "UNKNOWN_T"],
            "value": [50.0, 60.0],
        }
    )
    resolved = ResolvedShape(
        shape=Shape.MAP_PERIOD_TIME,
        frame=frame,
        entity_dim_columns=("name",),
        period_index_column="period",
        time_index_column="t",
    )
    out = _collect(broadcast_to_period_time(resolved, "group", _dt()))
    _assert_no_dupes(out, ["group", "d", "t"])
    assert 60.0 not in out["value"].to_list()
    assert out.sort(["d", "t"]).to_dicts() == [
        {"group": "fg", "d": "p2025", "t": "t01", "value": 50.0}]


# ---------------------------------------------------------------------------
# No-op for all-valid data: the fix must not change the normal broadcast.
# ---------------------------------------------------------------------------


def test_all_valid_full_broadcast_unchanged(axis_enums):
    """Mixed authoring with ALL-valid tokens exercises every sub-branch of
    the MAP_PERIOD_TIME split and must produce the full, unchanged
    broadcast with no dropped or duplicated rows."""
    frame = pl.DataFrame(
        {
            "name": ["fgA", "fgB", "fgC", "fgD"],
            # fgA: fully keyed 2-D cell.
            # fgB: period-only default (null t) → broadcast over t01, t02.
            # fgC: time-only default (null d) → broadcast over p2025, p2030.
            # fgD: scalar default (both null) → broadcast over whole grid.
            "period": ["p2025", "p2025", None, None],
            "t": ["t01", None, "t01", None],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )
    resolved = ResolvedShape(
        shape=Shape.MAP_PERIOD_TIME,
        frame=frame,
        entity_dim_columns=("name",),
        period_index_column="period",
        time_index_column="t",
    )
    out = _collect(broadcast_to_period_time(resolved, "group", _dt()))
    _assert_no_dupes(out, ["group", "d", "t"])
    got = {
        (r["group"], r["d"], r["t"]): r["value"] for r in out.to_dicts()
    }
    expected = {
        ("fgA", "p2025", "t01"): 10.0,
        ("fgB", "p2025", "t01"): 20.0,
        ("fgB", "p2025", "t02"): 20.0,
        ("fgC", "p2025", "t01"): 30.0,
        ("fgC", "p2030", "t01"): 30.0,
        ("fgD", "p2025", "t01"): 40.0,
        ("fgD", "p2025", "t02"): 40.0,
        ("fgD", "p2030", "t01"): 40.0,
        ("fgD", "p2030", "t02"): 40.0,
    }
    assert got == expected


def test_all_valid_map_time_and_map_period_unchanged(axis_enums):
    """All-valid 1-D MAP_TIME / MAP_PERIOD rows broadcast to their native
    axis exactly, with no rows dropped."""
    time_frame = pl.DataFrame(
        {"name": ["fg", "fg"], "t": ["t01", "t02"], "value": [10.0, 20.0]}
    )
    out_t = _collect(broadcast_to_period_time(
        ResolvedShape(
            shape=Shape.MAP_TIME, frame=time_frame,
            entity_dim_columns=("name",), period_index_column=None,
            time_index_column="t"),
        "group", _dt()))
    assert out_t.sort("t").to_dicts() == [
        {"group": "fg", "t": "t01", "value": 10.0},
        {"group": "fg", "t": "t02", "value": 20.0},
    ]

    period_frame = pl.DataFrame(
        {"name": ["fg", "fg"], "period": ["p2025", "p2030"],
         "value": [10.0, 20.0]}
    )
    out_d = _collect(broadcast_to_period_time(
        ResolvedShape(
            shape=Shape.MAP_PERIOD, frame=period_frame,
            entity_dim_columns=("name",), period_index_column="period",
            time_index_column=None),
        "group", _dt()))
    assert out_d.sort("d").to_dicts() == [
        {"group": "fg", "d": "p2025", "value": 10.0},
        {"group": "fg", "d": "p2030", "value": 20.0},
    ]
