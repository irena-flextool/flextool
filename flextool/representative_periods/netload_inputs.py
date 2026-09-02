"""Database reader for the net-load representative-period signal.

This is the *input* half of the net-load clustering feature (the pure math
lives in :mod:`flextool.representative_periods.netload`). It reads everything
the builder needs out of an already-open, scenario-filtered
:class:`spinedb_api.DatabaseMapping` — mirroring how
:mod:`flextool.representative_periods.preprocess` opens the database (see
``preprocess.preprocess_representative_periods`` around the
``with DatabaseMapping(db_url) as db:`` block) — and returns a plain
:class:`NetloadInputs` dataclass. No solving, no schema writes.

The net-load signal is a real-MW aggregate per *aggregation unit* rather than
the per-series normalized feature stack of ``preprocess._build_clustering_matrix``.
An aggregation unit is either a **region group** (when groups carry the boolean
``use_for_representative_periods`` flag) or, in the fallback, a single **node**
(each node is its own aggregation unit). On a pre-v68 (un-migrated) database the
flag parameter is not yet defined; that genuinely-absent case is detected and
handled gracefully by simply activating the per-node fallback, while a read
failure on a *defined* flag is allowed to surface rather than silently degrade.

FlexTool sign convention (the gotcha, shared with ``force_include``): **demand
is NEGATIVE inflow**, supply is positive inflow. A demand node's ``inflow``
value is negative; its demand magnitude is ``|value|``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import spinedb_api as api
from spinedb_api import DatabaseMapping

from flextool.engine_polars._db_reader import DictMode, params_to_dict
from flextool.engine_polars._derived_existing import _INVEST_NOT_ALLOWED

# The schema default of ``unit__node__profile.profile_method``. A unit whose
# arc never sets ``profile_method`` explicitly returns NO row from the param
# reader and MUST be treated as this value (VRE).
_PROFILE_METHOD_DEFAULT = "upper_limit"

# The value of ``group.use_for_representative_periods`` (a ``yes_no`` param)
# that opts a group in as an aggregation unit.
_FLAG_YES = "yes"


@dataclass(frozen=True)
class VreUnit:
    """A variable-renewable (``profile_method == upper_limit``) generation unit.

    Attributes:
        node: The node the unit's profiled output arc connects to.
        profile: The availability profile name (0-1 series) governing the arc.
        existing_cap: Existing capacity (``unit.existing``; 0.0 if unset).
        unitsize: Unit size (``unit.virtual_unitsize``; 1.0 placeholder when
            unset — carried for later phases, unused by the Phase 1-2 math).
        investable: Whether the unit's ``invest_method`` permits *building* new
            capacity — i.e. a method NOT in
            :data:`flextool.engine_polars._derived_existing._INVEST_NOT_ALLOWED`
            (``not_allowed`` and the retire-only methods). This is exactly the
            engine's ``entityInvest`` membership test, kept canonical by
            importing that set.
    """

    node: str
    profile: str
    existing_cap: float
    unitsize: float
    investable: bool


@dataclass
class NetloadInputs:
    """Everything :mod:`netload` needs to build the net-load clustering matrix.

    Every collection is emitted in sorted (name) order for determinism.

    Attributes:
        units_by_group: Aggregation-unit name → member node names. Under
            ``granularity == "group"`` this is a flagged region group → its
            ``group__node`` members; under ``"node"`` it is ``{node: [node]}``
            for every node.
        demand_ts: Node → time-varying inflow series ``[(key, value), ...]``
            (negative = demand, FlexTool convention).
        demand_scalar: Node → scalar (constant) inflow level as a float.
        vre: Unit name → :class:`VreUnit`.
        profiles: Profile name → availability series ``[(key, value), ...]``.
        step_durations: Timestep key → duration (from the first timeline).
        granularity: ``"group"`` when at least one group is flagged, else
            ``"node"`` (the per-node fallback).
    """

    units_by_group: dict[str, list[str]] = field(default_factory=dict)
    demand_ts: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    demand_scalar: dict[str, float] = field(default_factory=dict)
    vre: dict[str, VreUnit] = field(default_factory=dict)
    profiles: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    step_durations: dict[str, float] = field(default_factory=dict)
    granularity: str = "node"


def _coerce_float(value: object) -> float | None:
    """Coerce a ``params_to_dict`` scalar (a *string*) to float, or ``None``."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _read_inflows(
    db: DatabaseMapping,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    """Split node ``inflow`` into time-varying series and scalar demand levels.

    Mirrors ``preprocess._read_time_series``: ``params_to_dict`` returns a
    scalar float as a *string*, so scalars are the non-``list`` entries; they
    are coerced with ``float(...)`` (skipping anything non-numeric) and collected
    as ``demand_scalar``. Both maps are returned sorted by node name.
    """
    raw_inflows = params_to_dict(db=db, cl="node", par="inflow", mode=DictMode.DICT)
    demand_ts: dict[str, list[tuple[str, float]]] = {}
    demand_scalar: dict[str, float] = {}
    for name in sorted(raw_inflows):
        value = raw_inflows[name]
        if isinstance(value, list):
            # A Map comes back as ``[(key, value), ...]`` pairs. An ``Array``
            # comes back as a bare value list (no keys), which the downstream
            # ``for key, value in ts`` unpack cannot consume. Skip it with a
            # warning rather than crashing on the unpack.
            if all(
                isinstance(item, (tuple, list)) and len(item) == 2
                for item in value
            ):
                demand_ts[name] = value
            else:
                print(
                    f"  Net-load: skipping inflow of node '{name}': "
                    f"Array-typed (keyless) series is not supported; "
                    f"use a Map (timestep → value)."
                )
        else:
            coerced = _coerce_float(value)
            if coerced is not None:
                demand_scalar[name] = coerced
    return demand_ts, demand_scalar


def _read_profiles(
    db: DatabaseMapping,
) -> dict[str, list[tuple[str, float]]]:
    """Read time-varying profile series (name → ``[(key, value), ...]``), sorted."""
    raw_profiles = params_to_dict(
        db=db, cl="profile", par="profile", mode=DictMode.DICT
    )
    return {
        name: raw_profiles[name]
        for name in sorted(raw_profiles)
        if isinstance(raw_profiles[name], list)
    }


def _read_vre(db: DatabaseMapping) -> dict[str, VreUnit]:
    """Identify VRE units from ``unit__node__profile`` upper-limit arcs.

    A ``(unit, node, profile)`` arc is VRE when its ``profile_method`` is
    ``upper_limit`` — the SCHEMA DEFAULT, so an arc with no explicit
    ``profile_method`` row (absent from the param reader) counts as VRE via
    default-fill. Each qualifying unit's ``existing`` / ``virtual_unitsize`` /
    ``invest_method`` scalars are read to build its :class:`VreUnit`.

    A unit with several upper-limit arcs keeps the FIRST in sorted
    ``(unit, node, profile)`` order (deterministic); such multi-arc VRE units
    are rare in FlexTool models (a VRE unit is one output node + one profile).
    Returns a dict keyed by unit name, in sorted order.
    """
    # profile_method values keyed by the (unit, node, profile) byname tuple.
    methods: dict[tuple[str, str, str], str] = {}
    for pv in db.find_parameter_values(
        entity_class_name="unit__node__profile",
        parameter_definition_name="profile_method",
    ):
        byname = tuple(pv["entity_byname"])
        methods[byname] = api.from_database(pv["value"], pv["type"])

    # Unit-level scalars (strings from params_to_dict; coerced below).
    existing_raw = params_to_dict(db=db, cl="unit", par="existing", mode=DictMode.DICT)
    unitsize_raw = params_to_dict(
        db=db, cl="unit", par="virtual_unitsize", mode=DictMode.DICT
    )
    invest_raw = params_to_dict(
        db=db, cl="unit", par="invest_method", mode=DictMode.DICT
    )

    # Sorted (unit, node, profile) arcs → first upper-limit arc per unit.
    arcs = sorted(
        tuple(item["element_name_list"])
        for item in db.get_entity_items(entity_class_name="unit__node__profile")
    )
    vre: dict[str, VreUnit] = {}
    for arc in arcs:
        unit, node, profile = arc
        method = methods.get(arc, _PROFILE_METHOD_DEFAULT)
        if method != _PROFILE_METHOD_DEFAULT:
            continue
        if unit in vre:
            # Keep the first (sorted) upper-limit arc for a multi-arc unit.
            continue
        existing_cap = _coerce_float(existing_raw.get(unit))
        unitsize = _coerce_float(unitsize_raw.get(unit))
        invest_method = invest_raw.get(unit, "not_allowed")
        vre[unit] = VreUnit(
            node=node,
            profile=profile,
            existing_cap=existing_cap if existing_cap is not None else 0.0,
            # virtual_unitsize is a later-phase concern; default 1.0 keeps it a
            # usable divisor. Unused by the Phase 1-2 net-load math.
            unitsize=unitsize if unitsize is not None else 1.0,
            investable=invest_method not in _INVEST_NOT_ALLOWED,
        )
    # Re-key in sorted unit order for deterministic iteration downstream.
    return {u: vre[u] for u in sorted(vre)}


def _read_step_durations(db: DatabaseMapping) -> dict[str, float]:
    """Timestep key → duration from the first timeline (sorted-key dict)."""
    timelines = params_to_dict(
        db=db, cl="timeline", par="timestep_duration", mode=DictMode.DICT
    )
    if not timelines:
        return {}
    first = next(iter(timelines))
    data = timelines[first]
    if not isinstance(data, list):
        return {}
    durations: dict[str, float] = {}
    for key, dur in data:
        coerced = _coerce_float(dur)
        if coerced is not None:
            durations[str(key)] = coerced
    return durations


def _read_all_nodes(db: DatabaseMapping) -> list[str]:
    """Sorted names of every ``node`` entity."""
    return sorted(
        item["entity_byname"][0]
        for item in db.get_entity_items(entity_class_name="node")
    )


def _read_units_by_group(
    db: DatabaseMapping,
    all_nodes: list[str],
) -> tuple[dict[str, list[str]], str]:
    """Resolve aggregation units and the granularity.

    When one or more groups carry ``use_for_representative_periods == yes``, the
    aggregation units are those flagged groups, each resolved to its
    ``group__node`` members (``granularity = "group"``). A node that belongs to
    two or more flagged groups is reported on stdout (a plain ``print`` warning,
    no raise) and counted in EACH group.

    When no group is flagged — including on a pre-v68 database where the flag
    parameter is not yet defined (detected via the definition probe below) —
    the fallback is per-node: each node is its own aggregation unit
    (``granularity = "node"``).
    """
    # The flag parameter may be absent on a pre-v68 (un-migrated) database.
    # Probe the *definition* rather than swallowing every exception: a
    # genuinely-undefined parameter is the documented per-node fallback,
    # whereas a read failure on a parameter that *is* defined is a real
    # error that must surface instead of silently degrading to per-node.
    if not db.get_parameter_definition_item(
        entity_class_name="group",
        name="use_for_representative_periods",
    ):
        return {node: [node] for node in all_nodes}, "node"

    flags = params_to_dict(
        db=db,
        cl="group",
        par="use_for_representative_periods",
        mode=DictMode.DICT,
    )

    flagged = sorted(
        name for name, value in flags.items() if str(value) == _FLAG_YES
    )
    if not flagged:
        return {node: [node] for node in all_nodes}, "node"

    flagged_set = set(flagged)
    group_nodes: dict[str, set[str]] = {name: set() for name in flagged}
    for item in db.get_entity_items(entity_class_name="group__node"):
        group_name, node_name = item["element_name_list"]
        if group_name in flagged_set:
            group_nodes[group_name].add(node_name)

    # Report (do not raise) nodes shared by >= 2 flagged groups; count in each.
    node_to_groups: dict[str, list[str]] = {}
    for group_name in flagged:
        for node_name in group_nodes[group_name]:
            node_to_groups.setdefault(node_name, []).append(group_name)
    for node_name in sorted(node_to_groups):
        groups = sorted(node_to_groups[node_name])
        if len(groups) >= 2:
            print(
                f"  Net-load: node '{node_name}' belongs to "
                f"{len(groups)} flagged representative-period groups "
                f"{groups} — counted in each."
            )

    units_by_group = {name: sorted(group_nodes[name]) for name in flagged}
    return units_by_group, "group"


def read_netload_inputs(db: DatabaseMapping) -> NetloadInputs:
    """Read all net-load inputs from an open, scenario-filtered database mapping.

    Args:
        db: An open :class:`spinedb_api.DatabaseMapping`, already scenario-
            filtered by the caller (as ``preprocess`` does before reading time
            series). This function calls ``db.fetch_all("parameter_value")``
            defensively so it also works from a freshly-opened mapping in tests.

    Returns:
        A fully-populated :class:`NetloadInputs` with every collection sorted.
    """
    db.fetch_all("parameter_value")

    demand_ts, demand_scalar = _read_inflows(db)
    profiles = _read_profiles(db)
    vre = _read_vre(db)
    step_durations = _read_step_durations(db)
    all_nodes = _read_all_nodes(db)
    units_by_group, granularity = _read_units_by_group(db, all_nodes)

    return NetloadInputs(
        units_by_group=units_by_group,
        demand_ts=demand_ts,
        demand_scalar=demand_scalar,
        vre=vre,
        profiles=profiles,
        step_durations=step_durations,
        granularity=granularity,
    )
