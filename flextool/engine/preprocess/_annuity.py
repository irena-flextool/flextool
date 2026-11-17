"""The capital-recovery (annuity) factor and the level yearly price shared by every investment-cost path.

Formula::

    ann(r, n) = r / (1 − (1 + r)^−n)      r ≠ 0
              = 1 / n                     r = 0 (|r| < 1e-9)
    annuity_factor(r, n, off) = ann(r, n) · (1 + r)^(off − 1)

    annualised cost A [CUR/MW/a] = invest_cost [CUR/kW] · 1000 · annuity_factor(discount_rate, payback_time,
                                                                                 inflation_offset_investment)

    G(x) = Σ_{k<⌈x⌉} min(1, x − k)·v^k = (1 − v^⌊x⌋)/(1 − v) + (x − ⌊x⌋)·v^⌊x⌋,   v = 1/(1 + i)
         = x                                                                   i = 0 (|i| < 1e-9)
    level_factor(n, L, i) = G(n) / G(L)

    level yearly price a_cap [CUR/MW/a] = A · level_factor(payback_time, lifetime_technical, inflation_rate)

What it does: ``annuity_factor`` spreads an investment (or a retirement's net salvage value) over ``payback_time``
years at the interest ``discount_rate``.  ``r = 0`` is the straight-line limit of the formula: the annuity is
``cost / payback_time``.  An entity without ``payback_time`` pays back over its ``lifetime_technical``.  The
payment of service year k falls at ``off`` years into that year (``inflation_offset_investment``: 0 its start,
1 its end), so ``(1 + r)^(off − 1)`` sizes the payments for that moment: the loan taken when the capacity comes
online is then worth exactly ``cost`` at ``discount_rate`` for every ``off`` (``ann`` alone is the year-end loan
of ``off`` 1; at ``off`` 0 it was worth ``cost · (1 + r)``).
``level_factor`` turns that loan into the level yearly cost of the plant over its real lifetime: the present value
(at the model's ``inflation_rate`` i) of ``n`` yearly payments A, A·G(n), spread evenly over ``L`` years, gives
A·G(n)/G(L) a year.  It does not depend on when the plant is built, so one price per year in service holds for
every build period and period length.  With ``discount_rate = inflation_rate`` the price is
``invest_cost · 1000 · annuity_factor(i, L)`` whatever ``n`` (a whole number of years); with a ``discount_rate`` below the inflation rate a
shorter payback time costs more.  A fractional last year counts its fraction (the rule of a fractional
``years_represented``).

Notes:

* A missing ``discount_rate`` takes the schema default (0.05, ``DEFAULT_DISCOUNT_RATE`` when the source has no
  default); an explicit 0 stays 0.  Before schema v79 any value ≤ 0 became 5 % (the ``discount_rate_zero_is_zero``
  behaviour change); a negative value is now an input error (``validate_basic_sanity``).
* The payback time has no default: a missing or non-positive one where the cost is non-zero is an error
  (:func:`require_lifetime`), never a silent 20 years.
* ``|r| < 1e-9`` takes ``1/n`` and ``|i| < 1e-9`` takes ``G(x) = x``: the closed forms lose precision there (at
  ``r = 1e-12`` it is 9e-5 relative off).
* ``annuity_factor`` is a polars expression only: every path forms the annuity with it, so they agree bit for bit.
* ``level_factor`` is float only: the price of a (payback time, lifetime) pair is computed once in Python, so equal
  inputs give a bit-equal price whatever the layout of the frame that holds them.

Used, as the one form of the annuity, by ``preprocess/params/_npv.py`` (the values the model reads and the reported
costs) and ``preprocess/emit/_entity_annual.py`` (the per-solve ``ed_entity_annual*.csv``).
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

DEFAULT_DISCOUNT_RATE = 0.05

# Below this |r| (|i|) the factor is taken as the straight-line 1/n (G(x) = x).
_SMALL_RATE = 1e-9


def annuity_factor(r: pl.Expr, n: pl.Expr, offset: float) -> pl.Expr:
    """``r / (1 − (1+r)^−n) · (1+r)^(offset − 1)``; ``1/n`` for ``|r| < 1e-9`` (the r → 0 limit), ``offset`` the
    model's ``inflation_offset_investment`` (module docstring).  At ``offset`` 1 the timing factor is exactly 1.0."""
    ann = (pl.when(r.abs() < _SMALL_RATE)
             .then(1.0 / n)
             .otherwise(r / (1.0 - (1.0 + r).pow(-n))))
    return ann * (1.0 + r).pow(offset - 1.0)


def _level_sum(x: float, i: float) -> float:
    """G(x) of the module docstring (float)."""
    if abs(i) < _SMALL_RATE:
        return x
    v = 1.0 / (1.0 + i)
    whole = math.floor(x)
    return (1.0 - v ** whole) / (1.0 - v) + (x - whole) * v ** whole


def level_factor(n: float, life: float, i: float) -> float:
    """G(n) / G(life): the level yearly price of an annuity paid for ``n`` years, spread over ``life`` years at the
    discount rate ``i`` (module docstring)."""
    return _level_sum(n, i) / _level_sum(life, i)


def default_discount_rate(source=None) -> float:
    """The schema default of ``discount_rate`` (unit, connection and node share it), else 0.05."""
    if source is not None:
        for cls in ("unit", "connection", "node"):
            try:
                value = source.parameter_default(cls, "discount_rate")
            except (KeyError, AttributeError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return DEFAULT_DISCOUNT_RATE


def require_lifetime(missing: Iterable[tuple[str, str]], cost_param: str, lifetime_param: str) -> None:
    """Raise when an entity has a non-zero ``cost_param`` in a period but no positive ``lifetime_param`` (the
    payback time of the annuity) there.

    ``missing``: ``(entity, period)`` pairs (the first few are named).  ``validate_basic_sanity`` stops an entity
    without any lifetime earlier; this catches a period map that lacks the period."""
    pairs = list(missing)
    if not pairs:
        return
    shown = ", ".join(f"'{e}' in period {d}" for e, d in pairs[:5])
    raise FlexToolConfigError(
        f"{lifetime_param} is missing or not above 0 where {cost_param} is not 0: {shown}. The annuity needs the "
        f"{lifetime_param} in years for every period the entity can invest or retire in; give {lifetime_param} a "
        "value for these periods (or a constant).")
