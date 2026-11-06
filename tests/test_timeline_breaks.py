"""Timeline breaks for plots: the break rule on hand-written step adjacency, the keyed split / gap insertion, and
the one break-file format.

The break rule (``flextool.process_outputs._timeline_breaks.timeline_breaks``): a realized step other than the
first is a break when it has no ramp link or its link predecessor is not a realized step.  ``insert_timeline_breaks``
puts ``gap_rows`` NaN rows before each break row; rows keyed by ``(period, time)`` match the pairs, rows keyed by
time only match on time when the frame's period is known or the breaks hold one period.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from flextool.plot_outputs.format_helpers import (
    insert_timeline_breaks,
    load_timeline_breaks,
    split_at_timeline_breaks,
    write_timeline_breaks,
)
from flextool.process_outputs._timeline_breaks import timeline_breaks

LINK = ["period", "time", "period_previous", "time_previous"]


def _dt(rows) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(rows, names=["period", "time"])


def _link(rows) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(rows, names=LINK)


def _chain(d, ts):
    return [(d, b, d, a) for a, b in zip(ts, ts[1:])]


def _pairs(idx: pd.MultiIndex) -> list[tuple[str, str]]:
    return list(idx)


# ---------------------------------------------------------------------------
# the break rule


def test_step_after_a_gap_breaks():
    """t1-t3 and t5-t6 of one period (no link into t5): break at t5 only."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3", "t5", "t6")])
    link = _link(_chain("p", ["t1", "t2", "t3"]) + _chain("p", ["t5", "t6"]))
    assert _pairs(timeline_breaks(dt, link)) == [("p", "t5")]


def test_adjacent_representative_blocks_break():
    """Two adjacent representative blocks t1-t2 and t3-t4: the second start has no link, so it breaks."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3", "t4")])
    link = _link(_chain("p", ["t1", "t2"]) + _chain("p", ["t3", "t4"]))
    assert _pairs(timeline_breaks(dt, link)) == [("p", "t3")]


def test_continuing_period_is_joined():
    """p2 continues p1 (a cross-period link into p2's first step): no break."""
    dt = _dt([("p1", "t1"), ("p1", "t2"), ("p2", "t3"), ("p2", "t4")])
    link = _link(_chain("p1", ["t1", "t2"]) + [("p2", "t3", "p1", "t2")] + _chain("p2", ["t3", "t4"]))
    assert _pairs(timeline_breaks(dt, link)) == []


def test_period_that_does_not_continue_breaks():
    """p2 reuses p1's hours (no link into its first step): break at (p2, t1)."""
    dt = _dt([("p1", "t1"), ("p1", "t2"), ("p2", "t1"), ("p2", "t2")])
    link = _link(_chain("p1", ["t1", "t2"]) + _chain("p2", ["t1", "t2"]))
    assert _pairs(timeline_breaks(dt, link)) == [("p2", "t1")]


def test_roll_seam_with_a_carry_is_joined():
    """The second roll's first step t3 is carried from the previous roll's realized t2: no break."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3", "t4")])
    link = _link([("p", "t2", "p", "t1"), ("p", "t3", "p", "t2"), ("p", "t4", "p", "t3")])
    assert _pairs(timeline_breaks(dt, link)) == []


def test_link_from_a_step_that_is_not_realized_breaks():
    """A link whose predecessor is not a realized step (a look-ahead step of another roll) is a break."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t5", "t6")])
    link = _link([("p", "t2", "p", "t1"), ("p", "t5", "p", "t4"), ("p", "t6", "p", "t5")])
    assert _pairs(timeline_breaks(dt, link)) == [("p", "t5")]


def test_first_row_never_breaks_and_rule_is_order_free():
    """The first output row is never a break, and reordering the periods does not invent breaks."""
    rows = [("p1", "t1"), ("p1", "t2"), ("p2", "t3"), ("p2", "t4")]
    link = _link(_chain("p1", ["t1", "t2"]) + [("p2", "t3", "p1", "t2")] + _chain("p2", ["t3", "t4"]))
    swapped = _dt(rows[2:] + rows[:2])
    assert _pairs(timeline_breaks(swapped, link)) == [("p1", "t1")]
    assert _pairs(timeline_breaks(_dt(rows[:1]), _link([]))) == []


def test_empty_link_gives_no_breaks_and_logs_info(caplog):
    """No adjacency known: no breaks (not every step), an INFO note; none when every step starts a block."""
    dt = _dt([("p", t) for t in ("t1", "t2", "t3")])
    with caplog.at_level(logging.INFO, logger="flextool.process_outputs._timeline_breaks"):
        assert len(timeline_breaks(dt, _link([]), scenario="s1")) == 0
    notes = [r for r in caplog.records if "step adjacency unavailable" in r.getMessage()]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="flextool.process_outputs._timeline_breaks"):
        assert len(timeline_breaks(dt, _link([]), block_first=dt)) == 0
    assert not [r for r in caplog.records if "step adjacency unavailable" in r.getMessage()]


# ---------------------------------------------------------------------------
# split / insert


def _frame(index: pd.Index, columns=("a", "b")) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame({c: np.arange(n, dtype=float) + 100 * k for k, c in enumerate(columns)}, index=index)


TWO_PERIODS = _dt([("p1", f"t{i:04d}") for i in range(47, 52)] + [("p2", f"t{i:04d}") for i in range(47, 52)])


def test_split_is_keyed_on_period_and_time():
    """A break at (p2, t0049) splits p2 only, not p1's t0049; the period start (p2, t0047) is not in the set."""
    df = _frame(TWO_PERIODS)
    segments = split_at_timeline_breaks(df, _dt([("p2", "t0049")]))
    assert [len(s) for s in segments] == [7, 3]
    assert segments[1].index[0] == ("p2", "t0049")


def test_insert_puts_gap_rows_at_the_breaks_and_keeps_values():
    """n + 3 x breaks rows; NaN rows exactly before each break row, labelled with the row before them; the other
    rows equal the input row for row."""
    df = _frame(TWO_PERIODS)
    breaks = _dt([("p2", "t0047"), ("p2", "t0049")])
    out = insert_timeline_breaks(df, breaks)
    assert len(out) == len(df) + 3 * 2
    gap = out.isna().all(axis=1).to_numpy()
    assert np.flatnonzero(gap).tolist() == [5, 6, 7, 10, 11, 12]
    assert list(out.index[5:8]) == [("p1", "t0051")] * 3
    assert list(out.index[10:13]) == [("p2", "t0048")] * 3
    pd.testing.assert_frame_equal(out[~gap], df)


def test_insert_without_matching_breaks_returns_the_frame():
    df = _frame(TWO_PERIODS)
    assert insert_timeline_breaks(df, _dt([("p3", "t0049")])) is df
    assert insert_timeline_breaks(df, None) is df


def _time_only(times) -> pd.DataFrame:
    return _frame(pd.Index(times, name="time"))


TIMES = [f"t{i:04d}" for i in range(47, 52)]


def test_time_only_rows_match_when_the_breaks_hold_one_period():
    out = insert_timeline_breaks(_time_only(TIMES), _dt([("p2", "t0049")]))
    assert np.flatnonzero(out.isna().all(axis=1).to_numpy()).tolist() == [2, 3, 4]


def test_time_only_rows_without_a_known_period_and_several_break_periods_get_no_breaks():
    df = _time_only(TIMES)
    assert insert_timeline_breaks(df, _dt([("p1", "t0048"), ("p2", "t0049")])) is df


def test_time_only_rows_use_the_file_member_period():
    """The frame was cut to file member p1: only p1's breaks apply."""
    out = insert_timeline_breaks(_time_only(TIMES), _dt([("p1", "t0048"), ("p2", "t0049")]), period="p1")
    assert np.flatnonzero(out.isna().all(axis=1).to_numpy()).tolist() == [1, 2, 3]


def test_time_only_rows_use_the_period_column_level():
    """Columns per period (a subplot per period): breaks apply when those periods share one set of break times."""
    cols = pd.MultiIndex.from_tuples([("p1", "x"), ("p2", "x")], names=["period", "unit"])
    df = pd.DataFrame(np.arange(10.0).reshape(5, 2), index=pd.Index(TIMES, name="time"), columns=cols)
    same = insert_timeline_breaks(df, _dt([("p1", "t0049"), ("p2", "t0049"), ("p3", "t0050")]))
    assert np.flatnonzero(same.isna().all(axis=1).to_numpy()).tolist() == [2, 3, 4]
    assert insert_timeline_breaks(df, _dt([("p1", "t0048"), ("p2", "t0049")])) is df


# ---------------------------------------------------------------------------
# one file format


def test_write_and_load_round_trip_with_union(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_timeline_breaks(a, _dt([("p1", "t0049")]))
    write_timeline_breaks(b, _dt([("p1", "t0049"), ("p2", "t0001")]))
    assert sorted(load_timeline_breaks(a, b)) == [("p1", "t0049"), ("p2", "t0001")]
    assert list(load_timeline_breaks(a, b).names) == ["period", "time"]


@pytest.mark.parametrize("old", [pd.DataFrame({"period": ["p2020"], "t": ["t0049"]}),
                                 pd.DataFrame({"break_time": ["t0049"]}),
                                 pd.DataFrame({"time": ["t0049"]})])
def test_earlier_formats_are_ignored(tmp_path, old):
    """GMPL-era files (period in the first column), comparison files (``break_time``) and time-only files give no
    breaks: they are not read from their first column."""
    from flextool.lean_parquet import write_lean_parquet

    write_lean_parquet(old, tmp_path / "timeline_breaks.parquet", index=False)
    assert len(load_timeline_breaks(tmp_path)) == 0
    assert len(load_timeline_breaks(tmp_path / "missing")) == 0
