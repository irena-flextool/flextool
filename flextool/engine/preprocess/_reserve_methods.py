"""``reserve__upDown__group.reserve_method`` values per requirement partition.

A group belongs to every requirement its method names: ``timeseries_only`` → timeseries,
``timeseries_and_dynamic`` → timeseries and dynamic, ``all`` → all three, ``no_reserve`` → none.  Shared by
the solve_data CSV producer (``emit/_mid_sets.derive_reserve_method_partition``) and the DB projection
(``params/_projection.reserve_upDown_group_method``) so both give the same partitions.
"""
from __future__ import annotations

#: The ``reserve_method`` without a requirement (the schema default): its groups get no reserve variables or rows.
RESERVE_NO_RESERVE: str = "no_reserve"
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


def demanded_reserves(
    *,
    reservation_pairs: "set[tuple[str, str]]",
    group_methods: "list[tuple[str, str, str, str]]",
    group_nodes: "list[tuple[str, str]]",
    relationship_ratios: "dict[tuple[str, str, str, str], dict[str, float]]",
    group_ratios: "dict[tuple[str, str, str], float]",
) -> "set[tuple[str, str]]":
    """The ``(r, ud)`` pairs with a reserve requirement, whose providers get ``v_reserve``.

    ``(r, ud)`` is demanded when

    1. some group of it has a non-zero ``reservation`` (``reservation_pairs``); or
    2. some ``(r, ud, g)`` has a dynamic method and an ``increase_reserve_ratio > 0`` drives it: on a
       relationship ``(p, r, ud, n)`` at a node n of g, or on the group itself (``group_ratios``); or
    3. some ``(r, ud, g)`` has a large-failure (n-1) method and a relationship ``(p, r, ud, n)`` at a node n
       of g has a ``large_failure_ratio > 0``.

    Dynamic and n-1 requirements are reserve demand by themselves: they need no ``reservation``.  A dynamic
    or n-1 group without a driving ratio stays inactive (no all-zero ``v_reserve`` columns).  Rule 3 does
    not check that the failing process has a flow into (up) or out of (down) the group's node: an n-1 ratio
    on a process without a flow in the failing direction activates the reserve, but its n-1 rows are empty
    (the providers get ``v_reserve`` columns that no requirement row uses).

    Inputs are plain label tuples: ``group_methods`` ``(r, ud, g, method)``, ``group_nodes`` ``(g, n)``,
    ``relationship_ratios`` ``{(p, r, ud, n): {"increase_reserve_ratio": v, "large_failure_ratio": v}}``,
    ``group_ratios`` ``{(r, ud, g): increase_reserve_ratio}``.  Shared by the solve_data CSV producer
    (``emit/_reserve_sets._compute_process_reserve_active``) and the DB projection
    (``params/_derived.prundt_from_source``) so both activate the same reserves.
    """
    out = set(reservation_pairs)
    nodes_of: dict[str, set[str]] = {}
    for g, n in group_nodes:
        nodes_of.setdefault(g, set()).add(n)
    driving: dict[tuple[str, str, str], set[str]] = {}
    for (_p, r, ud, n), ratios in relationship_ratios.items():
        for name in ("increase_reserve_ratio", "large_failure_ratio"):
            if (ratios.get(name) or 0.0) > 0.0:
                driving.setdefault((r, ud, name), set()).add(n)
    for r, ud, g, method in group_methods:
        if (r, ud) in out:
            continue
        nodes = nodes_of.get(g, set())
        if method in RESERVE_DYNAMIC_METHODS and (
                (group_ratios.get((r, ud, g)) or 0.0) > 0.0
                or driving.get((r, ud, "increase_reserve_ratio"), set()) & nodes):
            out.add((r, ud))
        elif method in RESERVE_N_1_METHODS and driving.get((r, ud, "large_failure_ratio"), set()) & nodes:
            out.add((r, ud))
    return out
