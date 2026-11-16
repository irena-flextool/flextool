"""The lifetime and salvage roles and the names they are read under (``flextool.engine.input._legacy_names``).

A unit's technical lifetime, payback time and net salvage value are read, per entity, from the first name of the
role's chain the entity has a value of: ``payback_time`` → ``lifetime_technical`` → ``lifetime``,
``lifetime_technical`` → ``lifetime``, ``net_salvage_value`` → ``salvage_value``.  A class with values under both a
new name and the old name it replaces is an error.  The new names are defined by the database schema from the next
data version on; here the sources are in memory.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from flextool.engine.input._inmemory_reader import InMemoryReader
from flextool.engine.input._legacy_names import (
    LIFETIME_TECHNICAL,
    NET_SALVAGE_VALUE,
    NO_REINVESTMENT,
    PAYBACK_TIME,
    canonical_lifetime_method,
    source_physical_name,
    source_role_sources,
)
from flextool.engine.input._param_shapes import PARAM_ALLOWED_SHAPES, Shape
from flextool.engine.input._solve_state import FlexToolConfigError


def _const(values: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame({"name": list(values), "value": list(values.values())})


def _source(params: dict[str, dict[str, object]], units: tuple[str, ...] = ("a", "b")) -> InMemoryReader:
    """A source with units ``units`` and the constant unit parameters ``params`` (``{parameter: {unit: value}}``);
    a parameter not in ``params`` is not defined."""
    return InMemoryReader({"unit": pl.DataFrame({"name": list(units)})},
                          {("unit", name): _const(values) for name, values in params.items()},
                          defaults={("unit", "discount_rate"): 0.05})


def test_physical_name_prefers_a_defined_new_name() -> None:
    src = _source({"lifetime_technical": {"a": 20.0}, "payback_time": {"a": 10.0}, "net_salvage_value": {"a": 1.0}})
    assert source_physical_name(src, LIFETIME_TECHNICAL) == "lifetime_technical"
    assert source_physical_name(src, PAYBACK_TIME) == "payback_time"
    assert source_physical_name(src, NET_SALVAGE_VALUE) == "net_salvage_value"
    assert source_role_sources(src, "unit", LIFETIME_TECHNICAL) == {"a": "lifetime_technical"}
    assert source_role_sources(src, "unit", PAYBACK_TIME) == {"a": "payback_time"}
    assert source_role_sources(src, "unit", NET_SALVAGE_VALUE) == {"a": "net_salvage_value"}


def test_physical_name_falls_back_to_the_legacy_name() -> None:
    src = _source({"lifetime": {"a": 20.0, "b": 30.0}, "salvage_value": {"b": 5.0}})
    assert source_physical_name(src, LIFETIME_TECHNICAL) == "lifetime"
    assert source_physical_name(src, PAYBACK_TIME) == "lifetime"
    assert source_physical_name(src, NET_SALVAGE_VALUE) == "salvage_value"
    for role in (LIFETIME_TECHNICAL, PAYBACK_TIME):
        assert source_role_sources(src, "unit", role) == {"a": "lifetime", "b": "lifetime"}
    assert source_role_sources(src, "unit", NET_SALVAGE_VALUE) == {"b": "salvage_value"}


def test_payback_time_falls_back_per_entity_to_the_technical_lifetime() -> None:
    src = _source({"lifetime_technical": {"a": 20.0, "b": 30.0}, "payback_time": {"a": 10.0}})
    assert source_role_sources(src, "unit", PAYBACK_TIME) == {"a": "payback_time", "b": "lifetime_technical"}
    assert source_role_sources(src, "unit", LIFETIME_TECHNICAL) == {"a": "lifetime_technical",
                                                                    "b": "lifetime_technical"}


@pytest.mark.parametrize(("new", "old", "role"), [
    ("lifetime_technical", "lifetime", LIFETIME_TECHNICAL),
    ("payback_time", "lifetime", PAYBACK_TIME),
    ("net_salvage_value", "salvage_value", NET_SALVAGE_VALUE),
])
def test_physical_name_refuses_both_with_values(new, old, role) -> None:
    src = _source({new: {"a": 10.0}, old: {"b": 20.0}})
    with pytest.raises(FlexToolConfigError, match=rf"unit: '{old}' is given \(b\) next to '{new}' \(a\)"):
        source_role_sources(src, "unit", role)


def test_lifetime_method_no_investment_reads_as_no_reinvestment() -> None:
    assert canonical_lifetime_method("no_investment") == NO_REINVESTMENT == "no_reinvestment"
    for kept in ("no_reinvestment", "reinvest_choice", "reinvest_automatic"):
        assert canonical_lifetime_method(kept) == kept


# ---------------------------------------------------------------------------
# The annuity on the payback time, the window on the technical lifetime
#
# invest_cost c = 1000 CUR/kW, discount_rate r = 0.05.  Hand values:
#   n = 10: 1.05^-10 = 0.6139133, factor 0.05 / (1 - 0.6139133) = 0.1295046, annuity 1e6 · 0.1295046
#           = 129 504.57 CUR/MW/a;
#   n = 20: 1.05^-20 = 0.3768895, factor 0.05 / (1 - 0.3768895) = 0.0802426, annuity 80 242.59 CUR/MW/a.

ANN_N10 = 129_504.57
ANN_N20 = 80_242.59


@pytest.fixture
def new_names_registered(monkeypatch) -> None:
    """Register the new names' shapes, as the next data version's schema definitions do (a constant or a period
    map, like ``lifetime``)."""
    for cls in ("unit", "node", "connection"):
        for name in ("lifetime_technical", "payback_time", "net_salvage_value"):
            monkeypatch.setitem(PARAM_ALLOWED_SHAPES, (cls, name), {Shape.SCALAR, Shape.MAP_PERIOD})


def _invest_source(lifetimes: dict[str, float]) -> InMemoryReader:
    return _source({"invest_cost": {"u": 1000.0}, "discount_rate": {"u": 0.05},
                    "invest_method": {"u": "invest_no_limit"}, **{k: {"u": v} for k, v in lifetimes.items()}},
                   units=("u",))


def _annuity_and_window(src: InMemoryReader) -> tuple[float, float]:
    from flextool.engine.preprocess.params import _npv as npv

    ed = pl.LazyFrame({"e": ["u"], "d": ["p2020"]})
    ann = npv._per_method_annuity_lf(src, ed, "invest_cost", npv._INVEST_NOT_ALLOWED).collect()
    window = npv._ed_lifetime_on_lf(src, ed).collect()
    return float(ann["ann"][0]), float(window["life"][0])


def test_payback_time_split(new_names_registered) -> None:
    """lifetime_technical 20, payback_time 10: the annuity is over 10 years (129 504.57 per MW, not the 20-year
    80 242.59), the window over 20."""
    ann, life = _annuity_and_window(_invest_source({"lifetime_technical": 20.0, "payback_time": 10.0}))
    assert math.isclose(ann, ANN_N10, rel_tol=1e-7)
    assert life == 20.0


def test_payback_time_defaults_to_the_technical_lifetime(new_names_registered) -> None:
    """lifetime_technical 20 and no payback_time: the annuity is over the technical lifetime, 20 years."""
    ann, life = _annuity_and_window(_invest_source({"lifetime_technical": 20.0}))
    assert math.isclose(ann, ANN_N20, rel_tol=1e-7)
    assert life == 20.0


def test_payback_time_is_read_per_entity(new_names_registered) -> None:
    """Two units of one class, a with payback_time 10, b with only its technical lifetime 20 (both lifetime_technical
    20): a's annuity is over 10 years, b's over 20, and both windows over 20."""
    from flextool.engine.preprocess.params import _npv as npv

    src = _source({"invest_cost": {"a": 1000.0, "b": 1000.0}, "discount_rate": {"a": 0.05, "b": 0.05},
                   "invest_method": {"a": "invest_no_limit", "b": "invest_no_limit"},
                   "lifetime_technical": {"a": 20.0, "b": 20.0}, "payback_time": {"a": 10.0}})
    ed = pl.LazyFrame({"e": ["a", "b"], "d": ["p2020", "p2020"]})
    ann = dict(npv._per_method_annuity_lf(src, ed, "invest_cost", npv._INVEST_NOT_ALLOWED)
               .select("e", "ann").collect().iter_rows())
    assert math.isclose(ann["a"], ANN_N10, rel_tol=1e-7)
    assert math.isclose(ann["b"], ANN_N20, rel_tol=1e-7)
    window = dict(npv._ed_lifetime_on_lf(src, ed).select("e", "life").collect().iter_rows())
    assert window == {"a": 20.0, "b": 20.0}
