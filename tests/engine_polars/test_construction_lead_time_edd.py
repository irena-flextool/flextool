"""Slice H — construction lead time: availability-walk integration.

Exercises the P2 chain end to end (``commissioning_year_lf`` ->
``period_walk_iterator`` commission injection -> availability filters ->
``edd_invest_set``) against a hand-built source on the design §2.3 uneven
calendar (seams ``S = {0, 5, 8, 20}``), without invoking the LP.  Pins:

* §7a deterministic lag — the alive (order -> available) set shifts by the
  snapped commissioning year, per method;
* §7c byte-parity — ``immediate`` / ``L=0`` == no lead time at all;
* §7f lifetime-from-commissioning — a bounded cohort's window is
  ``[yr_c, yr_c + life)``, not ``[yr_c, yr_order + life)``.

The §2.5 [F3] silent-no-lag guard is pinned in
``test_construction_lead_time_guard``.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars._derived_existing import (
    assert_no_forced_out_of_horizon_commission,
    commissioning_year_lf,
    edd_invest_set_lf,
)
from flextool.engine_polars._solve_state import FlexToolConfigError

PERIODS = ["p1", "p2", "p3", "p4"]
YEARS = [0.0, 5.0, 8.0, 20.0]
YR = dict(zip(PERIODS, YEARS))


class _StubSource:
    """Minimal source: ``entities`` + ``parameter`` from dicts."""

    def __init__(self, entities: dict[str, pl.DataFrame],
                 params: dict[tuple[str, str], pl.DataFrame]):
        self._entities = entities
        self._params = params

    def entities(self, entity_class: str) -> pl.DataFrame:
        try:
            return self._entities[entity_class]
        except KeyError:
            raise KeyError(entity_class) from None

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


def _source(*, entity="nuke", lifetime_method="reinvest_automatic",
            life=None, lead=None, method=None,
            invest_forced=None) -> _StubSource:
    entities = {"unit": pl.DataFrame({"name": [entity]})}
    params: dict[tuple[str, str], pl.DataFrame] = {
        ("solve", "years_from_start"): _years_df(),
        ("unit", "lifetime_method"): pl.DataFrame(
            {"name": [entity], "value": [lifetime_method]}),
    }
    if life is not None:
        params[("unit", "lifetime")] = pl.DataFrame(
            {"name": [entity], "value": [float(life)]})
    if lead is not None:
        params[("unit", "construction_lead_time")] = pl.DataFrame(
            {"name": [entity], "value": [float(lead)]})
    if method is not None:
        params[("unit", "construction_lead_time_method")] = pl.DataFrame(
            {"name": [entity], "value": [method]})
    if invest_forced is not None:
        period, val = invest_forced
        params[("unit", "invest_forced")] = pl.DataFrame(
            {"name": [entity], "period": [period], "value": [float(val)]})
    return _StubSource(entities, params)


def _alive(source, order="p1", entity="nuke"):
    """Alive periods of edd_invest_set for (entity, order), with the lag
    threaded through commissioning_year_lf exactly as the derive site does."""
    ed_invest = pl.LazyFrame({"e": [entity], "d": [order]})
    commission_lf = commissioning_year_lf(
        source, "s", ed_invest, PERIODS)
    edd = edd_invest_set_lf(
        source, "s", ed_invest, PERIODS, PERIODS,
        commission_lf=commission_lf).collect()
    return sorted(edd.filter(pl.col("d_invest") == order)["d"].to_list())


# --- §7a deterministic lag (reinvest_automatic — lower bound only) ------

def test_no_lead_time_alive_from_order() -> None:
    """No lead time -> capacity alive from the order period (p1..p4)."""
    assert _alive(_source(lead=None, method=None)) == PERIODS


@pytest.mark.parametrize("method,expected", [
    # order p1 (yr 0), L=6 -> t*=6.  previous/closest -> c=5 (p2,p3,p4);
    # next -> c=8 (p3,p4).
    ("previous_seam", ["p2", "p3", "p4"]),
    ("closest_seam", ["p2", "p3", "p4"]),
    ("next_seam", ["p3", "p4"]),
])
def test_lag_shifts_alive_set_per_method(method, expected) -> None:
    assert _alive(_source(lead=6.0, method=method)) == expected


def test_immediate_is_byte_parity_with_no_lead() -> None:
    """§7c — explicit immediate (even with L>0) == no lead time at all."""
    lagged = _alive(_source(lead=6.0, method="immediate"))
    base = _alive(_source(lead=None, method=None))
    assert lagged == base == PERIODS


def test_zero_lead_is_byte_parity() -> None:
    """§7c — L=0 with a method still resolves to immediate -> no shift."""
    assert _alive(_source(lead=0.0, method="closest_seam")) == PERIODS


def test_default_method_is_closest_seam() -> None:
    """L>0, no explicit method -> closest_seam -> c=5 (p2,p3,p4)."""
    assert _alive(_source(lead=6.0, method=None)) == ["p2", "p3", "p4"]


# --- §7f lifetime-from-commissioning (bounded cohort) -------------------

def test_lifetime_from_commissioning_bounded_cohort() -> None:
    """reinvest_choice, L=6 next_seam (c=8), life=15: alive [8, 23) ->
    p3 (8), p4 (20) — NOT [8, 15) (which would truncate to p3 only) and
    NOT [0, 15) (lifetime-from-order, no lag)."""
    src = _source(lifetime_method="reinvest_choice", life=15.0,
                  lead=6.0, method="next_seam")
    assert _alive(src) == ["p3", "p4"]


def test_bounded_cohort_no_lag_control() -> None:
    """Same bounded cohort with no lead time: alive [0, 15) -> p1, p2, p3
    (yr 0/5/8 < 15; p4 yr 20 expired).  Proves the lag is what moves the
    window in the test above."""
    src = _source(lifetime_method="reinvest_choice", life=15.0)
    assert _alive(src) == ["p1", "p2", "p3"]


# --- §7k [F5] forced-order out-of-horizon guard -------------------------

def _commission(src, order="p3"):
    return commissioning_year_lf(
        src, "s", pl.LazyFrame({"e": ["nuke"], "d": [order]}), PERIODS)


def test_forced_out_of_horizon_raises() -> None:
    """order p3 (yr 8), L=15 next_seam -> t*=23 > 20 -> yr_c=+inf; with
    invest_forced pinned at (nuke, p3) the guard raises FlexToolConfigError
    rather than pay-for-never-delivered capacity."""
    src = _source(lead=15.0, method="next_seam", invest_forced=("p3", 1.0))
    commission_lf = _commission(src, "p3")
    with pytest.raises(FlexToolConfigError, match="beyond the model horizon"):
        assert_no_forced_out_of_horizon_commission(src, commission_lf)


def test_voluntary_out_of_horizon_does_not_raise() -> None:
    """Same out-of-horizon order but NOT forced -> allowed (the sentinel
    makes it inert; the model simply never picks it)."""
    src = _source(lead=15.0, method="next_seam")
    commission_lf = _commission(src, "p3")
    # No raise.
    assert_no_forced_out_of_horizon_commission(src, commission_lf)


def test_forced_in_horizon_does_not_raise() -> None:
    """A forced order whose commissioning IS in-horizon (closest_seam snaps
    to the last seam) is fine — only +inf (next_seam past the horizon)
    trips the guard."""
    src = _source(lead=15.0, method="closest_seam",
                  invest_forced=("p3", 1.0))
    commission_lf = _commission(src, "p3")
    assert_no_forced_out_of_horizon_commission(src, commission_lf)
