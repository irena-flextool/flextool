"""Slice H — §7b stochastic staging composition (solver-free).

Encoding A keeps ``v_invest`` order-indexed; the invest-NA window
(``non_anticipativity_invest_periods``, v72) ties ``v_invest[e, o]`` across
branches for order periods ``o`` in the window.  Staging is then a
CONSEQUENCE of the deterministic lag, needing no new constraint (design
§4): to be online at the reveal period a long-lead asset must be *ordered*
early enough that its commissioning reaches the reveal — and if that order
period falls in the (pre-reveal) window it is already tied.

This pins the provable backbone of §4.2 without the LP: **which order
periods can make capacity available at the reveal period** under each lead
time.  On the §4.2 calendar (p1=0, p2=5, p3=8, p4=20), reveal at p3, window
= {p1, p2}:

* solar (L=1) — a post-reveal order (p3) already serves p3 -> REACTS
  (diverges per branch);
* gas (L=6) — no post-reveal order serves p3; the serving orders are
  {p1, p2} ⊆ window -> the p3-serving decision is TIED;
* nuclear (L=12) — only order p1 serves p3 -> FULLY TIED.

If those serving order periods lie in the window, the *existing* invest-NA
families tie them across branches — no Slice H constraint.
"""
from __future__ import annotations

import polars as pl

from flextool.engine_polars._derived_existing import (
    commissioning_year_lf,
    edd_invest_set_lf,
)

PERIODS = ["p1", "p2", "p3", "p4"]
YEARS = [0.0, 5.0, 8.0, 20.0]
REVEAL = "p3"
WINDOW = {"p1", "p2"}  # pre-reveal (non_anticipativity_invest_periods)


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


def _source(entity, lead, method="closest_seam"):
    return _StubSource(
        {"unit": pl.DataFrame({"name": [entity]})},
        {
            ("solve", "years_from_start"): pl.DataFrame({
                "name": ["s"] * len(PERIODS),
                "period": PERIODS, "value": YEARS,
            }),
            ("unit", "lifetime_method"): pl.DataFrame(
                {"name": [entity], "value": ["reinvest_automatic"]}),
            ("unit", "construction_lead_time"): pl.DataFrame(
                {"name": [entity], "value": [float(lead)]}),
            ("unit", "construction_lead_time_method"): pl.DataFrame(
                {"name": [entity], "value": [method]}),
        })


def _alive(src, entity, order):
    ed = pl.LazyFrame({"e": [entity], "d": [order]})
    commission_lf = commissioning_year_lf(src, "s", ed, PERIODS)
    edd = edd_invest_set_lf(
        src, "s", ed, PERIODS, PERIODS, commission_lf=commission_lf).collect()
    return set(edd.filter(pl.col("d_invest") == order)["d"].to_list())


def _orders_serving(entity, lead):
    """Order periods whose commissioning makes capacity available at the
    reveal period."""
    src = _source(entity, lead)
    return {o for o in PERIODS if REVEAL in _alive(src, entity, o)}


def test_solar_short_lead_reacts_at_reveal() -> None:
    """L=1: a post-reveal order (p3 itself) serves p3 -> solar can REACT to
    the reveal (diverge per branch); staging does NOT force it early."""
    serving = _orders_serving("solar", 1.0)
    assert REVEAL in serving, serving
    assert not serving <= WINDOW, "short-lead must have a post-reveal server"


def test_gas_medium_lead_partly_tied() -> None:
    """L=6: no post-reveal order reaches p3; the p3-serving orders are the
    pre-reveal window {p1, p2} -> the p3-serving decision is TIED by the
    existing invest-NA window."""
    serving = _orders_serving("gas", 6.0)
    assert REVEAL not in serving, "a post-reveal order must NOT reach p3"
    assert serving == {"p1", "p2"}, serving
    assert serving <= WINDOW


def test_nuclear_long_lead_fully_tied() -> None:
    """L=12: only order p1 reaches p3 -> fully tied to the earliest
    pre-reveal order."""
    serving = _orders_serving("nuclear", 12.0)
    assert serving == {"p1"}, serving
    assert serving <= WINDOW


def test_staging_orders_lie_in_window_for_long_lead() -> None:
    """The composition claim: for both long-lead techs every p3-serving
    order lies in the invest-NA window, so the ALREADY-BUILT
    non_anticipativity_invest families tie them across branches — no new
    Slice H constraint (design §4)."""
    for entity, lead in (("gas", 6.0), ("nuclear", 12.0)):
        assert _orders_serving(entity, lead) <= WINDOW
