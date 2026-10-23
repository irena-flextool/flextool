"""``reserve__upDown__group.reserve_method`` values per requirement partition.

A group belongs to every requirement its method names: ``timeseries_only`` → timeseries,
``timeseries_and_dynamic`` → timeseries and dynamic, ``all`` → all three, ``no_reserve`` → none.  Shared by
the solve_data CSV producer (``emit/_mid_sets.derive_reserve_method_partition``) and the DB projection
(``params/_projection.reserve_upDown_group_method``) so both give the same partitions.
"""
from __future__ import annotations

RESERVE_TIMESERIES_METHODS: frozenset[str] = frozenset((
    "timeseries_only", "timeseries_and_dynamic",
    "timeseries_and_large_failure", "all",
))
RESERVE_DYNAMIC_METHODS: frozenset[str] = frozenset((
    "dynamic_only", "timeseries_and_dynamic",
    "dynamic_and_large_failure", "all",
))
RESERVE_N_1_METHODS: frozenset[str] = frozenset((
    "large_failure_only", "timeseries_and_large_failure",
    "dynamic_and_large_failure", "all",
))

# Partition name (the FlexData field suffix ``reserve_upDown_group_method_<name>``) → its method values.
RESERVE_METHODS_BY_PARTITION: dict[str, frozenset[str]] = {
    "timeseries": RESERVE_TIMESERIES_METHODS,
    "dynamic": RESERVE_DYNAMIC_METHODS,
    "n_1": RESERVE_N_1_METHODS,
}
