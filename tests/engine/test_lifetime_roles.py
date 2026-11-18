"""The payback-time rule of the lifetime parameters (``flextool.engine.input._lifetime_roles``).

A unit's payback time is read, per entity, from ``payback_time``, else from its ``lifetime_technical``; the windows
and the end of life always use ``lifetime_technical``.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from flextool.engine.input._inmemory_reader import InMemoryReader
from flextool.engine.input._lifetime_roles import PAYBACK_TIME, source_role_sources


def _const(values: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame({"name": list(values), "value": list(values.values())})


def _source(params: dict[str, dict[str, object]], units: tuple[str, ...] = ("a", "b")) -> InMemoryReader:
    """A source with units ``units`` and the constant unit parameters ``params`` (``{parameter: {unit: value}}``);
    a parameter not in ``params`` is not defined."""
    return InMemoryReader({"unit": pl.DataFrame({"name": list(units)})},
                          {("unit", name): _const(values) for name, values in params.items()},
                          defaults={("unit", "discount_rate"): 0.05})


def test_payback_time_falls_back_per_entity_to_the_technical_lifetime() -> None:
    src = _source({"lifetime_technical": {"a": 20.0, "b": 30.0}, "payback_time": {"a": 10.0}})
    assert source_role_sources(src, "unit", PAYBACK_TIME) == {"a": "payback_time", "b": "lifetime_technical"}


# ---------------------------------------------------------------------------
# The annuity on the payback time, the window on the technical lifetime
#
# invest_cost c = 1000 CUR/kW, discount_rate r = 0.05, the default discount_offset_investment 0 (payments at the
# start of each year: the year-end annuity divided by 1 + r).  Hand values:
#   n = 10: 1.05^-10 = 0.6139133, factor 0.05 / (1 - 0.6139133) = 0.1295046, annuity 1e6 · 0.1295046 / 1.05
#           = 123 337.69 CUR/MW/a;
#   n = 20: 1.05^-20 = 0.3768895, factor 0.05 / (1 - 0.3768895) = 0.0802426, annuity 1e6 · 0.0802426 / 1.05
#           = 76 421.51 CUR/MW/a.

ANN_N10 = 123_337.69
ANN_N20 = 76_421.51


def _invest_source(lifetimes: dict[str, float]) -> InMemoryReader:
    return _source({"invest_cost": {"u": 1000.0}, "discount_rate": {"u": 0.05},
                    "invest_method": {"u": "invest_no_limit"}, **{k: {"u": v} for k, v in lifetimes.items()}},
                   units=("u",))


def _db_path(lifetimes: dict[str, float]) -> tuple[float, float]:
    """``(annuity, window)`` of unit u read from the database (``_npv``)."""
    from flextool.engine.preprocess.params import _npv as npv

    src = _invest_source(lifetimes)
    ed = pl.LazyFrame({"e": ["u"], "d": ["p2020"]})
    ann = npv._per_method_annuity_lf(src, ed, "invest_cost", npv._INVEST_NOT_ALLOWED).collect()
    window = npv._ed_lifetime_on_lf(src, ed).collect()
    return float(ann["ann"][0]), float(window["life"][0])


def _emit_path(lifetimes: dict[str, float]) -> tuple[float, float]:
    """``(annuity, window)`` of unit u read from the CSV-generation lookup (``_entity_annual``): the annuity and the
    technical lifetime the windows of the emitted sets use."""
    from flextool.engine.input._lifetime_roles import LIFETIME_TECHNICAL, lookup_roles
    from flextool.engine.preprocess.emit._entity_annual import _annuities
    from tests.engine.test_npv_cluster import _Lookup

    lookup = _Lookup({("u", "invest_cost"): 1000.0, ("u", "discount_rate"): 0.05,
                      **{("u", k): v for k, v in lifetimes.items()}})
    inp = {"pp": lookup, "roles": {"pp": lookup_roles(lookup)}, "offset_inv": 0.0}
    ann = _annuities(inp, "invest_cost", [("u", "p2020", "pp", 1)])[0]
    return ann, lookup.get("u", LIFETIME_TECHNICAL, "p2020")


@pytest.mark.parametrize("path", [_db_path, _emit_path], ids=["db", "emit"])
def test_payback_time_split(path) -> None:
    """lifetime_technical 20, payback_time 10: the annuity is over 10 years (123 337.69 per MW, not the 20-year
    76 421.51), the window over 20."""
    ann, life = path({"lifetime_technical": 20.0, "payback_time": 10.0})
    assert math.isclose(ann, ANN_N10, rel_tol=1e-7)
    assert life == 20.0


@pytest.mark.parametrize("path", [_db_path, _emit_path], ids=["db", "emit"])
def test_payback_time_defaults_to_the_technical_lifetime(path) -> None:
    """lifetime_technical 20 and no payback_time: the annuity is over the technical lifetime, 20 years."""
    ann, life = path({"lifetime_technical": 20.0})
    assert math.isclose(ann, ANN_N20, rel_tol=1e-7)
    assert life == 20.0


class _DefaultBroadcastingSource(InMemoryReader):
    """A source whose ``parameter`` adds the definition's default for every entity without an explicit value (as
    the database reader does), while ``parameter_explicit`` gives the explicit values only."""

    def parameter(self, entity_class: str, parameter_name: str) -> pl.DataFrame:
        explicit = self.parameter_explicit(entity_class, parameter_name)
        default = self.parameter_default(entity_class, parameter_name)
        if default is None:
            return explicit
        missing = [e for e in self.entities(entity_class)["name"].to_list() if e not in explicit["name"].to_list()]
        return pl.concat([explicit, pl.DataFrame({"name": missing, "value": [default] * len(missing)})])

    def parameter_explicit(self, entity_class: str, parameter_name: str) -> pl.DataFrame:
        return InMemoryReader.parameter(self, entity_class, parameter_name)


def test_a_schema_default_is_not_a_payback_time() -> None:
    """The fallback reads explicit values only: with a ``payback_time`` default on the definition (none is shipped)
    and no unit with its own ``payback_time``, every unit pays back over its technical lifetime, not the default."""
    src = _DefaultBroadcastingSource(
        {"unit": pl.DataFrame({"name": ["a", "b"]})},
        {("unit", "lifetime_technical"): _const({"a": 20.0, "b": 30.0}),
         ("unit", "payback_time"): pl.DataFrame(schema={"name": pl.Utf8, "value": pl.Float64})},
        defaults={("unit", "payback_time"): 5.0})
    assert src.parameter("unit", "payback_time").height == 2
    assert source_role_sources(src, "unit", PAYBACK_TIME) == {"a": "lifetime_technical", "b": "lifetime_technical"}
