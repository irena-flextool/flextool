"""``group.co2_method`` values per CO2 term: the price, the period cap and the total cap.

A group gets every term its method names (``price_period`` → price and period cap, ``price_period_total`` →
all three, ``none`` → none).  Shared by the solve_data CSV producer (``emit/_leaf_sets.derive_group_co2``),
the DB projection (``params/_projection.group_co2_max_period``) and the input check
(``input_derivation/_validators.validate_group_limit_flags``) so all three read a method the same way.
"""
from __future__ import annotations

CO2_PRICE_METHODS: frozenset[str] = frozenset((
    "price", "price_period", "price_total", "price_period_total",
))
CO2_MAX_PERIOD_METHODS: frozenset[str] = frozenset((
    "period", "price_period", "period_total", "price_period_total",
))
CO2_MAX_TOTAL_METHODS: frozenset[str] = frozenset((
    "total", "price_total", "period_total", "price_period_total",
))
