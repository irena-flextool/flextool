"""The cohorts a solve hands to the later solves of a run (``_cohorts.realized_cohorts``).

A two-solve chain by hand: solve 1 realizes an investment of 2 units (unitsize 50 MW) in p2020 and retires 1 unit
of existing capacity there; solve 2 (handed solve 1's carriers) realizes 3 units in p2025 and 0.5 units again of the
p2020 decision.  Expected, from these numbers only: solve 2 carries the p2020 cohort with its solve-1 row unchanged
and size 100 + 25 = 125 MW, the p2025 cohort with size 150 MW and the years of its own cohort-table row, and the
p2020 retirement of 50 MW under the existing cohort.  Each cohort keeps its price per year in service (``a_cap``).
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from flextool.engine.preprocess.params._cohorts import (
    PRICE_COLUMNS,
    REALIZED_RETIRE_SCHEMA,
    _cohort_frame,
    realized_cohorts,
)

INF = math.inf
UNITSIZE = {"plant": 50.0}


def _table(rows: list[tuple]) -> pl.DataFrame:
    """The cohort table of ``rows`` given up to ``a_cap``; the other price columns are null."""
    return _cohort_frame([row + (None,) * (len(PRICE_COLUMNS) - 1) for row in rows])


EXISTING = ("existing:p2020", "plant", "existing", "reinvest_choice", "next_seam", None, "p2020", 0.0, 0.0, 12.0,
            12.0, 12.0, 15.0, "p2035", None)
SOLVE1 = _table([EXISTING,
                 ("invest:p2020", "plant", "invest", "reinvest_choice", "next_seam", "p2020", "p2020", 0.0, 0.0, 12.0,
                  12.0, None, 15.0, "p2035", 45_130.0)])
SOLVE2 = _table([EXISTING,
                 ("invest:p2020", "plant", "invest", "reinvest_choice", "next_seam", "p2020", "p2020", 0.0, 0.0, 12.0,
                  12.0, None, 15.0, "p2035", 45_130.0),
                 ("invest:p2025", "plant", "invest", "reinvest_choice", "next_seam", "p2025", "p2025", 5.0, 0.0, 30.0,
                  35.0, None, INF, None, 26_020.0)])


def test_two_solve_chain_carries_each_cohort_once_with_its_own_years():
    c1, r1 = realized_cohorts(None, None, SOLVE1, {("plant", "p2020"): 2.0}, {("plant", "p2020"): 1.0}, UNITSIZE)
    assert c1.select("cohort", "period_decided", "period_online", "y_on", "y_eol", "period_eol", "size").rows() == [
        ("invest:p2020", "p2020", "p2020", 0.0, 15.0, "p2035", 100.0)]
    assert r1.rows() == [("plant", "existing:p2020", "p2020", 50.0)]

    c2, r2 = realized_cohorts(c1, r1, SOLVE2, {("plant", "p2025"): 3.0, ("plant", "p2020"): 0.5}, {}, UNITSIZE)
    assert c2.select("cohort", "y_on", "y_life", "y_eol", "period_eol", "a_cap", "size").rows() == [
        ("invest:p2020", 0.0, 12.0, 15.0, "p2035", 45_130.0, 125.0),
        ("invest:p2025", 5.0, 30.0, INF, None, 26_020.0, 150.0)]
    assert r2.rows() == r1.rows()
    assert dict(r2.schema) == REALIZED_RETIRE_SCHEMA


def test_nothing_realized_hands_nothing():
    assert realized_cohorts(None, None, SOLVE1, {}, {}, UNITSIZE) == (None, None)


def test_an_investment_without_a_cohort_row_raises():
    with pytest.raises(ValueError, match="no start of operation"):
        realized_cohorts(None, None, SOLVE1, {("plant", "p2030"): 1.0}, {}, UNITSIZE)
