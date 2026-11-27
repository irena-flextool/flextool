"""The master / region boundary of constraints and groups: the side of each user constraint, the feature groups
and the investment-limit fields a side keeps, the groups whose limit binds, and the CO2 partition."""
from __future__ import annotations

import polars as pl

from polar_high import Param

from flextool.engine.preprocess.flex_data import FlexData


def _user_constraint_sides(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
    master_local_procs: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Aggregate, per user-constraint id (``cn``), the decomposition
    sides ("master" / "region") of every referenced entity across every
    user-constraint frame.

    Shared by the mixed-constraint hard error
    (:func:`_validate_user_constraints`) and the master/region row
    partition (:func:`_master_side_constraint_ids`).  Entities on
    neither side (shared nodes, non-node tokens) are neutral.

    Returns ``(sides_by_cn, refs_by_cn)``; ``refs_by_cn`` carries
    human-readable ``"entity (side)"`` strings for error messages.
    """
    sides_by_cn: dict[str, set[str]] = {}
    refs_by_cn: dict[str, set[str]] = {}

    def _node_side(n: str) -> str | None:
        if n in master_nodes:
            return "master"
        if n in all_region_nodes:
            return "region"
        return None

    def _add(cn: str, entity: str, side: str | None) -> None:
        if side is None:
            return
        sides_by_cn.setdefault(cn, set()).add(side)
        refs_by_cn.setdefault(cn, set()).add(f"{entity} ({side})")

    def _frame_of(obj) -> pl.DataFrame | None:
        if obj is None:
            return None
        f = obj.frame if isinstance(obj, Param) else obj
        return f if f.height > 0 else None

    # Arc-keyed references: (p, source, sink, cn).  Each row contributes
    # the sides of BOTH terminal nodes (a row on a region↔master
    # coupling arc is itself mixed) plus the master-local proc side.
    for fld in ("flow_constraint_idx", "p_flow_constraint_coef"):
        f = _frame_of(getattr(data, fld, None))
        if f is None or "cn" not in f.columns:
            continue
        for r in f.iter_rows(named=True):
            cn = r["cn"]
            if r["p"] in master_local_procs:
                # The flow of a process the master holds whole is a master column, whichever its nodes.
                _add(cn, r["p"], "master")
                continue
            _add(cn, r["source"], _node_side(r["source"]))
            _add(cn, r["sink"], _node_side(r["sink"]))
    # Node-keyed references: (n, cn).
    for fld in ("p_node_constraint_state_coeff",
                "p_node_constraint_invested_capacity_coeff",
                "p_node_constraint_prebuilt_capacity_coeff",
                "p_node_constraint_capacity_coeff",
                "p_node_constraint_retired_capacity_coeff",
                "p_node_constraint_in_construction_capacity_coeff"):
        f = _frame_of(getattr(data, fld, None))
        if f is None or "cn" not in f.columns:
            continue
        for r in f.iter_rows(named=True):
            _add(r["cn"], r["n"], _node_side(r["n"]))
    # Process-keyed references: (p, cn).  A process's side is the union
    # of its arc-endpoint node sides (a master-local proc is master; a
    # coupling connection contributes both sides and therefore raises).
    proc_sides: dict[str, set[str]] = {}
    if data.process_source_sink is not None:
        for r in data.process_source_sink.iter_rows(named=True):
            if r["p"] in master_local_procs:
                continue
            s = proc_sides.setdefault(r["p"], set())
            for n in (r["source"], r["sink"]):
                side = _node_side(n)
                if side is not None:
                    s.add(side)
    for p in master_local_procs:
        proc_sides.setdefault(p, set()).add("master")
    for fld in ("p_process_constraint_invested_capacity_coeff",
                "p_process_constraint_prebuilt_capacity_coeff",
                "p_process_constraint_capacity_coeff",
                "p_process_constraint_retired_capacity_coeff",
                "p_process_constraint_in_construction_capacity_coeff",
                "p_process_constraint_online_coeff"):
        f = _frame_of(getattr(data, fld, None))
        if f is None or "cn" not in f.columns:
            continue
        for r in f.iter_rows(named=True):
            for side in proc_sides.get(r["p"], set()):
                _add(r["cn"], r["p"], side)

    return sides_by_cn, refs_by_cn


def _master_side_constraint_ids(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
    master_local_procs: set[str],
) -> set[str]:
    """User-constraint ids (``cn``) whose EVERY sided entity reference
    is master-side.  Those constraints live whole in the Benders master:
    :func:`master_network_data` keeps exactly their rows and
    :func:`split` drops them from every region (a region copy would
    degenerate to ``0 sense constant`` once its master-keyed coefficient
    rows are scrubbed).  Assumes :func:`_validate_user_constraints` ran
    (no mixed constraints)."""
    sides_by_cn, _ = _user_constraint_sides(
        data, all_region_nodes, master_nodes, master_local_procs)
    return {cn for cn, sides in sides_by_cn.items() if sides == {"master"}}


#: Group-feature SET fields (``(g,)``): a group present in one of these
#: activates the corresponding group-level constraint family
#: (capacity margin / inertia / non-sync).  Membership is ``group_node``.
_GROUP_FEATURE_SET_FIELDS: tuple[str, ...] = (
    "groupCapacityMargin", "groupInertia", "groupNonSync",
)

#: Group-feature PARAM fields (``(g,)`` / ``(g, d)``) filtered alongside
#: the set fields when partitioning feature groups to the master.
_GROUP_FEATURE_PARAM_FIELDS: tuple[str, ...] = (
    "p_inv_group_cap", "p_group_capacity_for_scaling",
    "pdGroup_capacity_margin",
)


def _master_side_feature_groups(
    data: FlexData,
    all_region_nodes: set[str],
    master_nodes: "frozenset[str] | set[str]",
) -> set[str]:
    """Feature-carrying groups whose member nodes are ALL master-hosted.

    A group in one of the :data:`_GROUP_FEATURE_SET_FIELDS` sets with
    at least one master-hosted member must have EVERY member
    master-hosted — the group constraint sums over its members and
    cannot be enforced whole on either side otherwise (hard error,
    never a silent degrade).  Groups with no master members are left to
    the regions (today's semantics); bare ``group_node`` membership
    rows of non-feature groups are filtered silently.
    """
    feature_gs: set[str] = set()
    for fld in _GROUP_FEATURE_SET_FIELDS:
        f = getattr(data, fld, None)
        if f is not None and f.height > 0:
            feature_gs |= set(f["g"].cast(pl.Utf8).to_list())
    if not feature_gs:
        return set()
    members_by_g: dict[str, set[str]] = {}
    gn = data.group_node
    if gn is not None:
        for r in gn.iter_rows(named=True):
            members_by_g.setdefault(r["g"], set()).add(r["n"])
    master = set(master_nodes)
    kept: set[str] = set()
    for g in sorted(feature_gs):
        members = members_by_g.get(g, set())
        master_hit = members & master
        if not master_hit:
            continue
        if members <= master:
            kept.add(g)
            continue
        raise RuntimeError(
            f"split: feature-carrying group {g!r} straddles the "
            f"region/master boundary — it has master-hosted member "
            f"node(s) {sorted(master_hit)} AND non-master member(s) "
            f"{sorted(members - master)}.  A group-level constraint "
            f"(capacity margin / inertia / non-sync) sums over its "
            f"members and cannot be split between a region subproblem "
            f"and the Benders master — regroup the nodes so every "
            f"member is on one side."
        )
    return kept


#: Entity-keyed (``e``) investment / retirement limit sets and Params (the rows of ``_cumulative_invest`` and the
#: per-entity maxima of ``model/core/``).  A region keeps the rows of its own entities only (item 286).
_ENTITY_LIMIT_FIELDS: tuple[str, ...] = (
    "ed_invest_period_set", "ed_divest_period_set",
    "ed_invest_min_period", "ed_divest_min_period", "ed_invest_max_period", "ed_divest_max_period",
    "e_invest_total", "e_divest_total",
    "e_invest_min_total", "e_divest_min_total", "e_invest_max_total", "e_divest_max_total",
    "ed_invest_cumulative", "ed_cumulative_max_capacity", "ed_cumulative_min_capacity",
)

#: Group-keyed (``g``) investment / retirement / cumulative-capacity limit sets (membership ``group_entity``).
_GROUP_LIMIT_SET_FIELDS: tuple[str, ...] = (
    "gd_invest_period", "gd_divest_period", "g_invest_total", "g_divest_total", "g_invest_cumulative",
)

#: The group limit Params filtered alongside :data:`_GROUP_LIMIT_SET_FIELDS`.
_GROUP_LIMIT_PARAM_FIELDS: tuple[str, ...] = (
    "p_group_invest_max_period", "p_group_invest_min_period",
    "p_group_retire_max_period", "p_group_retire_min_period",
    "p_group_invest_max_total", "p_group_invest_min_total",
    "p_group_retire_max_total", "p_group_retire_min_total",
    "pd_group_cumulative_max_capacity", "pd_group_cumulative_min_capacity",
)


#: The group limit Params of :data:`_GROUP_LIMIT_PARAM_FIELDS` that are minima (the others are maxima).
_GROUP_LIMIT_MIN_FIELDS: frozenset[str] = frozenset(f for f in _GROUP_LIMIT_PARAM_FIELDS if "_min_" in f)


def _limit_groups(data: FlexData) -> dict[str, str]:
    """``{group: "min" | "max"}``: the groups whose investment, retirement or cumulative-capacity limit has a value
    that binds a sum over the group's members ("min" when one of them is a minimum).

    A group counts only with an ``invest_method`` that enables its limit (it is in a set of
    :data:`_GROUP_LIMIT_SET_FIELDS`) **and** a value of a limit parameter (:data:`_GROUP_LIMIT_PARAM_FIELDS`) in
    the solve: a method alone (or a value in an alternative outside the scenario) makes no row.  A minimum is a
    value > 0 (the readers drop 0 and ``-inf``); a maximum a finite value > 0: ``inf`` is no limit, and a maximum
    of 0 splits exactly between regions (each region's members then build or retire nothing, as the whole group).
    """
    enabled: set[str] = set()
    for fld in _GROUP_LIMIT_SET_FIELDS:
        f = getattr(data, fld, None)
        if f is not None and f.height > 0 and "g" in f.columns:
            enabled |= {str(g) for g in f.get_column("g").to_list()}
    out: dict[str, str] = {}
    if not enabled:
        return out
    for fld in _GROUP_LIMIT_PARAM_FIELDS:
        p = getattr(data, fld, None)
        frame = getattr(p, "frame", None)
        if frame is None or frame.height == 0 or "g" not in frame.columns or "value" not in frame.columns:
            continue
        value = pl.col("value").cast(pl.Float64)
        binding = (value > 0) if fld in _GROUP_LIMIT_MIN_FIELDS else (value.is_finite() & (value > 0))
        kind = "min" if fld in _GROUP_LIMIT_MIN_FIELDS else "max"
        for g in frame.filter(binding).get_column("g").cast(pl.Utf8).unique().to_list():
            if g in enabled and out.get(g) != "min":
                out[g] = kind
    return out


def _co2_master_partition(
    data: FlexData,
    master_local_triples: set[tuple[str, str, str]],
    region_master_triples: set[tuple[str, str, str]],
    master_local_procs: "set[str] | frozenset[str]" = frozenset(),
) -> set[str]:
    """Validate the CO2 frames against the master boundary and return
    the CO2-capped groups whose EVERY capped flow is a master-local arc
    (those cap constraints live whole in the master).

    A start-up draw (``startup_from_co2_capped`` / ``_capped_total``,
    unit__inputNode.startup_input_method) is a term on the unit's
    ``v_startup``: it is on the master side when the unit is a
    master-local process (``master_local_procs``), else on the region
    side (the unit's own region subproblem), and counts towards the
    group's side like a capped flow.

    Hard errors (never a silent degrade):

    * a CO2-priced flow row on a region↔master coupling arc — the arc's
      flow lives natively in the master while the region holds the
      half-flow, so neither side can carry the CO2 cost whole;
    * a capped group mixing master-local and region-side flows — the
      shared cap cannot be enforced on either side alone.
    """
    for fld in ("flow_from_co2_priced", "flow_from_co2_priced_noEff"):
        df = getattr(data, fld, None)
        if df is None or df.height == 0:
            continue
        for r in df.iter_rows(named=True):
            t = (r["p"], r["source"], r["sink"])
            if t in region_master_triples:
                raise RuntimeError(
                    f"split: CO2-priced flow {t} sits on a "
                    f"region↔master coupling arc — the CO2 cost "
                    f"cannot be split between a region subproblem and "
                    f"the Benders master.  Keep CO2-priced flows off "
                    f"the handover connections."
                )
    master_flags_by_g: dict[str, list[bool]] = {}
    for fld in ("flow_from_co2_capped", "flow_from_co2_capped_noEff",
                "flow_from_co2_capped_total",
                "flow_from_co2_capped_total_noEff"):
        df = getattr(data, fld, None)
        if df is None or df.height == 0:
            continue
        for r in df.iter_rows(named=True):
            t = (r["p"], r["source"], r["sink"])
            master_flags_by_g.setdefault(r["g"], []).append(
                t in master_local_triples)
    for fld in ("startup_from_co2_capped", "startup_from_co2_capped_total"):
        df = getattr(data, fld, None)
        if df is None or df.height == 0:
            continue
        for p, g in df.select(pl.col("p").cast(pl.Utf8), pl.col("g").cast(pl.Utf8)).iter_rows():
            master_flags_by_g.setdefault(g, []).append(p in master_local_procs)
    kept: set[str] = set()
    for g in sorted(master_flags_by_g):
        flags = master_flags_by_g[g]
        if all(flags):
            kept.add(g)
        elif any(flags):
            raise RuntimeError(
                f"split: CO2-capped group {g!r} straddles the "
                f"region/master boundary — it caps both master-local "
                f"and region-side flows (a start-up draw is on its "
                f"unit's side).  A shared CO2 cap cannot be "
                f"split between a region subproblem and the Benders "
                f"master — regroup so every capped flow is on one side."
            )
    return kept
