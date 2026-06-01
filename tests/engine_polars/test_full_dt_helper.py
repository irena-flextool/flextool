"""Unit tests for :func:`flextool.engine_polars._full_dt.slice_full_dt`.

Covers the four behaviours of the full_dt slice helper (specs §10.1):
identity for strict-static frames (no ``(d,t)`` axis), coordinate slicing
to a roll window, the empty-roll edge case, and row-order preservation.
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._full_dt import slice_full_dt


def _roll(period_step_pairs):
    return pl.DataFrame(
        {
            "period": [p for p, _ in period_step_pairs],
            "step": [s for _, s in period_step_pairs],
        },
        schema={"period": pl.Utf8, "step": pl.Utf8},
    )


def test_identity_when_no_dt_cols():
    """A frame with neither ``period`` nor ``time`` is returned unchanged."""
    full = pl.DataFrame(
        {"commodity": ["coal", "gas"], "value": ["1.0", "2.0"]},
        schema={"commodity": pl.Utf8, "value": pl.Utf8},
    )
    roll = _roll([("d1", "t1")])
    out = slice_full_dt(full, roll)
    assert out.equals(full)


def test_slice_to_subset_of_dt():
    """Only rows whose (period, time) is in the roll window survive."""
    full = pl.DataFrame(
        {
            "commodity": ["coal", "coal", "coal", "gas"],
            "period": ["d1", "d1", "d2", "d1"],
            "time": ["t1", "t2", "t1", "t1"],
            "value": ["10", "11", "12", "20"],
        },
        schema={c: pl.Utf8 for c in ("commodity", "period", "time", "value")},
    )
    # Roll window covers (d1,t1) and (d2,t1) only.
    roll = _roll([("d1", "t1"), ("d2", "t1")])
    out = slice_full_dt(full, roll)
    expected = full.filter(
        ((pl.col("period") == "d1") & (pl.col("time") == "t1"))
        | ((pl.col("period") == "d2") & (pl.col("time") == "t1"))
    )
    assert out.equals(expected)
    # The (d1,t2) coordinate (coal=11) and any (d,t) not in the roll are gone.
    assert "11" not in out["value"].to_list()


def test_empty_roll_yields_empty_with_schema():
    """An empty roll window yields zero rows but the full frame's schema."""
    full = pl.DataFrame(
        {
            "commodity": ["coal"],
            "period": ["d1"],
            "time": ["t1"],
            "value": ["10"],
        },
        schema={c: pl.Utf8 for c in ("commodity", "period", "time", "value")},
    )
    roll = _roll([])
    out = slice_full_dt(full, roll)
    assert out.height == 0
    assert out.schema == full.schema


def test_row_order_preserved():
    """The semi-join keeps the left (full) frame's row order."""
    # Deliberately interleave commodities and (d,t) so a reordering would
    # be visible.
    full = pl.DataFrame(
        {
            "commodity": ["z", "a", "z", "a", "m"],
            "period": ["d1", "d1", "d2", "d2", "d1"],
            "time": ["t1", "t1", "t1", "t1", "t2"],
            "value": ["1", "2", "3", "4", "5"],
        },
        schema={c: pl.Utf8 for c in ("commodity", "period", "time", "value")},
    )
    # Keep every (d,t) so the only thing under test is order, not filtering.
    roll = _roll([("d1", "t1"), ("d2", "t1"), ("d1", "t2")])
    out = slice_full_dt(full, roll)
    assert out["value"].to_list() == ["1", "2", "3", "4", "5"]
    assert out.equals(full)


def test_accepts_lazy_inputs():
    """Both arguments may be passed as LazyFrames."""
    full = pl.DataFrame(
        {
            "commodity": ["coal", "gas"],
            "period": ["d1", "d2"],
            "time": ["t1", "t1"],
            "value": ["10", "20"],
        },
        schema={c: pl.Utf8 for c in ("commodity", "period", "time", "value")},
    )
    roll = _roll([("d1", "t1")])
    out = slice_full_dt(full.lazy(), roll.lazy())
    assert out["value"].to_list() == ["10"]
    assert isinstance(out, pl.DataFrame)
