"""The capital-recovery (annuity) factor shared by every investment-cost path.

Formula::

    annuity_factor(r, n) = r / (1 − (1 + r)^−n)      r ≠ 0
                         = 1 / n                     r = 0 (|r| < 1e-9)

    annualised cost [CUR/MW/a] = invest_cost [CUR/kW] · 1000 · annuity_factor(discount_rate, lifetime)

What it does: spreads an investment (or a retirement's salvage value) over ``lifetime`` years at the interest
``discount_rate``.  ``r = 0`` is the straight-line limit of the formula: the annuity is ``cost / lifetime``.

Notes:

* A missing ``discount_rate`` takes the schema default (0.05, ``DEFAULT_DISCOUNT_RATE`` when the source has no
  default); an explicit 0 stays 0.  Before schema v79 any value ≤ 0 became 5 % (the ``discount_rate_zero_is_zero``
  behaviour change); a negative value is now an input error (``validate_basic_sanity``).
* ``lifetime`` has no default: a missing or non-positive lifetime where the cost is non-zero is an error
  (:func:`require_lifetime`), never a silent 20 years.
* ``|r| < 1e-9`` takes ``1/n``: the closed form loses precision there (at ``r = 1e-12`` it is 9e-5 relative
  off), so both helpers agree to rounding.

Used by ``preprocess/params/_npv.py`` (lazy, the values the model reads) and ``preprocess/emit/_entity_annual.py``
(the per-solve ``ed_entity_annual*.csv`` that build the invest / divest sets).
"""
from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from flextool.engine.input._solve_state import FlexToolConfigError

DEFAULT_DISCOUNT_RATE = 0.05

# Below this |r| the factor is taken as the straight-line 1/n.
_SMALL_RATE = 1e-9


def annuity_factor(r: float, n: float) -> float:
    """``r / (1 − (1+r)^−n)``; ``1/n`` for ``|r| < 1e-9`` (the r → 0 limit)."""
    if abs(r) < _SMALL_RATE:
        return 1.0 / n
    return r / (1.0 - (1.0 + r) ** -n)


def annuity_factor_expr(r: pl.Expr, n: pl.Expr) -> pl.Expr:
    """Polars form of :func:`annuity_factor` (same formula, same small-rate branch)."""
    return (pl.when(r.abs() < _SMALL_RATE)
              .then(1.0 / n)
              .otherwise(r / (1.0 - (1.0 + r).pow(-n))))


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


def require_lifetime(missing: Iterable[tuple[str, str]], cost_param: str) -> None:
    """Raise when an entity has a non-zero ``cost_param`` in a period but no positive ``lifetime`` there.

    ``missing``: ``(entity, period)`` pairs (the first few are named).  ``validate_basic_sanity`` stops an entity
    without any lifetime earlier; this catches a period map that lacks the period."""
    pairs = list(missing)
    if not pairs:
        return
    shown = ", ".join(f"'{e}' in period {d}" for e, d in pairs[:5])
    raise FlexToolConfigError(
        f"lifetime is missing or not above 0 where {cost_param} is not 0: {shown}. The annuity needs the "
        "lifetime in years for every period the entity can invest or retire in; give lifetime a value for "
        "these periods (or a constant).")
