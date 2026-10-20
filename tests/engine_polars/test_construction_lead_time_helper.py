"""Slice H — construction lead time: commissioning-year helper (solver-free).

Drives :func:`._derived_existing.commissioning_year_lf` directly against a
hand-built stub source on the design §2.3 uneven calendar (seams
``S = {0, 5, 8, 20}``), pinning ``c = snap(yr(o) + L, method)`` for every
method (§7d) and the two-level default (§7e).

The stub feeds the seam calendar through ``solve.years_from_start``
(``_p_years_d_lf`` arm 3), exactly like ``test_period_walk_lineage``.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from flextool.engine_polars._derived_existing import commissioning_year_lf

# §2.3 calendar: p1 [0,5), p2 [5,8), p3 [8,20), p4 [20, tail).
PERIODS = ["p1", "p2", "p3", "p4"]
YEARS = [0.0, 5.0, 8.0, 20.0]


class _StubSource:
    """Answers ``parameter(entity_class, name)`` from a dict; ``KeyError``
    otherwise (mirrors ``test_period_walk_lineage._StubSource``)."""

    def __init__(self, params: dict[tuple[str, str], pl.DataFrame]):
        self._params = params

    def parameter(self, entity_class: str, name: str) -> pl.DataFrame:
        try:
            return self._params[(entity_class, name)]
        except KeyError:
            raise KeyError((entity_class, name)) from None

    def parameter_default(self, entity_class: str, name: str):
        raise KeyError((entity_class, name))


def _years_df() -> pl.DataFrame:
    return pl.DataFrame({
        "name": ["s"] * len(PERIODS),
        "period": PERIODS,
        "value": YEARS,
    })


def _source(lead: float | None, method: str | None,
            entity: str = "nuke", ec: str = "unit") -> _StubSource:
    params: dict[tuple[str, str], pl.DataFrame] = {
        ("solve", "years_from_start"): _years_df(),
    }
    if lead is not None:
        params[(ec, "construction_lead_time")] = pl.DataFrame(
            {"name": [entity], "value": [float(lead)]})
    if method is not None:
        params[(ec, "construction_lead_time_method")] = pl.DataFrame(
            {"name": [entity], "value": [method]})
    return _StubSource(params)


def _anchor(order_period: str, entity: str = "nuke") -> pl.LazyFrame:
    return pl.LazyFrame({"e": [entity], "d": [order_period]})


def _yr_c(lead, method, order_period):
    """Return the single resolved yr_c, or ``None`` when the frame is
    empty (helper returned None)."""
    src = _source(lead, method)
    out = commissioning_year_lf(src, "s", _anchor(order_period), PERIODS)
    if out is None:
        return None
    df = out.collect()
    assert df.height == 1, f"expected one row, got {df.height}"
    return df["yr_c"][0]


# --- §7d seam-snap, per method, on the uneven calendar -----------------

@pytest.mark.parametrize("lead,method,order,expected", [
    # order p1 (yr 0), L=6 -> t*=6 inside p2 [5,8): lo=5, hi=8.
    (6.0, "previous_seam", "p1", 5.0),
    (6.0, "next_seam", "p1", 8.0),
    (6.0, "closest_seam", "p1", 5.0),   # |6-5|=1 < |6-8|=2
    # order p1, L=4 -> t*=4 inside p1 [0,5): lo=0, hi=5.
    (4.0, "next_seam", "p1", 5.0),
    (4.0, "closest_seam", "p1", 5.0),   # |4-0|=4 > |4-5|=1 -> hi
    # order p1, L=6.5 -> t*=6.5, exact midpoint 5<->8: tie -> hi.
    (6.5, "closest_seam", "p1", 8.0),
    (6.5, "previous_seam", "p1", 5.0),
    (6.5, "next_seam", "p1", 8.0),
    # order p2 (yr 5), L=10 -> t*=15 inside p3 [8,20): lo=8, hi=20.
    (10.0, "previous_seam", "p2", 8.0),
    (10.0, "next_seam", "p2", 20.0),
    (10.0, "closest_seam", "p2", 20.0),  # |15-8|=7 > |15-20|=5 -> hi
    # order p3 (yr 8), L=15 -> t*=23 > max seam 20: hi absent.
    (15.0, "previous_seam", "p3", 20.0),  # snap to last seam
    (15.0, "closest_seam", "p3", 20.0),   # snap to last seam
])
def test_seam_snap(lead, method, order, expected) -> None:
    assert _yr_c(lead, method, order) == expected


def test_next_seam_out_of_horizon_is_inf() -> None:
    """order p3 (yr 8), L=15 -> t*=23 > 20, next_seam -> +inf sentinel
    (no in-horizon commissioning seam -> capacity never alive)."""
    got = _yr_c(15.0, "next_seam", "p3")
    assert got is not None and math.isinf(got)


def test_previous_seam_snapping_back_to_order_is_dropped() -> None:
    """order p1 (yr 0), L=4, previous_seam -> lo=0 == yr(o) -> no effective
    lag -> the row is dropped -> helper returns None (byte-parity)."""
    assert _yr_c(4.0, "previous_seam", "p1") is None


# --- immediate / L=0 -> None (byte-parity path) ------------------------

def test_immediate_method_returns_none() -> None:
    """Explicit immediate suppresses the lag even with L>0 -> None."""
    assert _yr_c(6.0, "immediate", "p1") is None


def test_zero_lead_returns_none() -> None:
    assert _yr_c(0.0, "closest_seam", "p1") is None


def test_no_lead_param_returns_none() -> None:
    """No construction_lead_time param at all -> L=0 everywhere -> None."""
    src = _StubSource({("solve", "years_from_start"): _years_df()})
    assert commissioning_year_lf(src, "s", _anchor("p1"), PERIODS) is None


# --- §7e two-level default ---------------------------------------------

def test_default_lag_is_closest_seam() -> None:
    """L>0, no explicit method -> closest_seam (order p1 L=6 -> 5)."""
    assert _yr_c(6.0, None, "p1") == 5.0


def test_default_no_lead_is_immediate() -> None:
    """L=0, no explicit method -> immediate -> None."""
    assert _yr_c(0.0, None, "p1") is None


def test_explicit_method_wins_over_default() -> None:
    """L>0 with explicit previous_seam -> honoured (order p1 L=6 -> 5)."""
    assert _yr_c(6.0, "previous_seam", "p1") == 5.0
