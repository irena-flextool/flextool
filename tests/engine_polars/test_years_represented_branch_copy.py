"""Year-row branch copies must use period-name EQUALITY, not substring
containment (design ``specs/step05_continuation_fanout_design.md``
§1.3 / §3.4 / §5.6).

DEFECT PIN: pre-fix, ``derive_years_represented`` /
``derive_period_years`` matched branch rows with
``pd_pair[0] in period__years[0]`` — substring containment — so with
prefix-overlapping period names (``p1`` / ``p10``), ``p10``'s year
rows were ALSO copied to ``p1``'s branches.

The committed integration fixture deliberately uses non-overlapping
names (``p2035`` / ``p2040``), so this pure unit test is the sole pin
for the substring bug.
"""
from __future__ import annotations

from flextool.engine_polars._emit_solve_writers import (
    derive_period_years,
    derive_years_represented,
)

# p1 branches into p1_low; p10 is a later, prefix-overlapping period.
PERIOD_BRANCH = [
    ("p1", "p1"),
    ("p1", "p1_low"),
    ("p10", "p10"),
]
YEARS_REPRESENTED = [("p1", "1"), ("p10", "1")]


def test_years_represented_no_substring_copy() -> None:
    df = derive_years_represented(PERIOD_BRANCH, YEARS_REPRESENTED)
    rows = [tuple(r) for r in df.rows()]
    # p1's year row is copied to its branch p1_low...
    assert ("p1_low", "0", "0", "1.0") in rows
    # ...but p10's year row must NOT be (substring bug: "p1" in "p10").
    assert ("p1_low", "1.0", "1.0", "1.0") not in rows, (
        "p10's year rows leaked to p1_low via substring containment"
    )
    # And the real periods carry exactly their own rows.
    assert ("p1", "0", "0", "1.0") in rows
    assert ("p10", "1.0", "1.0", "1.0") in rows
    assert len(rows) == 3


def test_period_years_no_substring_copy() -> None:
    df = derive_period_years(PERIOD_BRANCH, YEARS_REPRESENTED)
    rows = [tuple(r) for r in df.rows()]
    assert rows == [
        ("p1", "0"),
        ("p1_low", "0"),
        ("p10", "1.0"),
    ], "p1_low must inherit only p1's cumulative-year row"
