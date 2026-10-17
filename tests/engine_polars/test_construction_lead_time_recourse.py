"""Slice H — §7i seam-calendar identity under branch-copied periods.

The [F2] risk is that a branch order period's ``yr(o)`` (from the
commissioning helper) desyncs from the ``yr_dall`` the availability walk
compares it against.  Both read years through the identical
``_p_years_d_lf`` provider/workdir block, so a fan member period (a
byte-copy of its anchor, carrying the anchor's year) must receive the
EXACT same lag as its anchor.

This is exercised with a branch-copied calendar: ``p2035`` and its fan
member ``p2035_low`` share year 15; ``p2040`` / ``p2040_low`` share year
20.  An order at ``p2035_low`` must produce the same commissioning year
and the same alive set as an order at ``p2035`` — proving both landed on
one calendar (a pure variable-value check would not).
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._derived_existing import (
    commissioning_year_lf,
    edd_invest_set_lf,
)

# h2020 is HISTORY (not in period_in_use); the two p2035 siblings share
# year 15, the two p2040 siblings share year 20 (branch byte-copies).
VOCAB = ["h2020", "p2035", "p2035_low", "p2040", "p2040_low"]
YEARS = [0.0, 15.0, 15.0, 20.0, 20.0]
PIU = ["p2035", "p2035_low", "p2040", "p2040_low"]


class _StubSource:
    def __init__(self, entities, params):
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


def _source(method="next_seam", lead=3.0, entity="nuke"):
    return _StubSource(
        {"unit": pl.DataFrame({"name": [entity]})},
        {
            ("solve", "years_from_start"): pl.DataFrame({
                "name": ["s"] * len(VOCAB),
                "period": VOCAB, "value": YEARS,
            }),
            ("unit", "lifetime_method"): pl.DataFrame(
                {"name": [entity], "value": ["reinvest_automatic"]}),
            ("unit", "construction_lead_time"): pl.DataFrame(
                {"name": [entity], "value": [float(lead)]}),
            ("unit", "construction_lead_time_method"): pl.DataFrame(
                {"name": [entity], "value": [method]}),
        })


def _yr_c(src, order):
    out = commissioning_year_lf(
        src, "s", pl.LazyFrame({"e": ["nuke"], "d": [order]}), PIU)
    df = out.collect()
    return df.filter(pl.col("d") == order)["yr_c"][0]


def _alive(src, order):
    ed = pl.LazyFrame({"e": ["nuke"], "d": [order]})
    commission_lf = commissioning_year_lf(src, "s", ed, PIU)
    edd = edd_invest_set_lf(
        src, "s", ed, PIU, PIU, commission_lf=commission_lf).collect()
    return sorted(edd.filter(pl.col("d_invest") == order)["d"].to_list())


def test_branch_copy_order_gets_same_commissioning_year() -> None:
    """yr(p2035_low) == yr(p2035) == 15 -> identical snapped yr_c."""
    src = _source()
    assert _yr_c(src, "p2035_low") == _yr_c(src, "p2035")


def test_branch_copy_order_gets_same_alive_set() -> None:
    """The alive (order -> available) set for the branch order equals its
    byte-copied anchor's — the calendar-identity guarantee (§7i)."""
    src = _source()
    anchor_alive = _alive(src, "p2035")
    branch_alive = _alive(src, "p2035_low")
    # order yr 15, L=3 -> t*=18, next_seam -> yr_c=20 -> alive p2040 siblings.
    assert branch_alive == anchor_alive == ["p2040", "p2040_low"]
