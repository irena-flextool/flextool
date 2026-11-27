"""The Benders master's reduced :class:`FlexData` (:func:`master_network_data`, the inverse of the split) and the
node / constraint / CO2 / group field lists it empties or keeps."""
from __future__ import annotations

import dataclasses

import polars as pl

from polar_high import Param

from flextool.engine.preprocess.flex_data import FlexData
from flextool.engine.preprocess._reduce_detail import STARTUP_FIELDS
from flextool.engine.input._axis_enums import get_global_axis_enums, schema_dtype

from flextool.engine.solve._region_filter._regions import load_region_membership
from flextool.engine.solve._region_filter._frames import _filter_frame, _filter_param, _is_in_keep
from flextool.engine.solve._region_filter._arcs import _classify_arcs, _master_local_procs, master_units
from flextool.engine.solve._region_filter._master_side import (
    _ENTITY_LIMIT_FIELDS,
    _GROUP_FEATURE_PARAM_FIELDS,
    _GROUP_FEATURE_SET_FIELDS,
    _co2_master_partition,
    _master_side_constraint_ids,
    _master_side_feature_groups,
)
from flextool.engine.solve._region_filter._checks import _validate_user_constraints


# ---------------------------------------------------------------------------
# Benders network-only master producer (the INVERSE of ``split``)
# ---------------------------------------------------------------------------


#: FlexData fields keyed on a NODE axis ``n`` (node entity sets, storage,
#: inflow, penalties, profiles, …).  The master OMITS every terminal node
#: from balance, so each of these is emptied (set to ``None``) — a node
#: absent from ``nodeBalance`` / ``nodeStateBlock`` gets no balance row
#: (Phase-3 design §1.3; ``model/core/`` builds balance only over the
#: populated node sets).
_MASTER_NODE_FIELDS: tuple[str, ...] = (
    "nodeBalance_dt", "nodeBalancePeriod",
    "nodeState", "nodeState_dt", "nodeState_first_dt", "nodeState_last_dt",
    "nodeState_rp", "nodeStateBlock",
    "nodeCoarse", "node_coarse_step", "node_coarse_interior",
    "storage_bind_within_timeblock", "storage_bind_forward_only",
    "storage_bind_within_solve", "storage_bind_within_period", "storage_bind_within_solve_blended_weights",
    "storage_bind_within_period_blended_weights",
    "storage_bind_forward_only_blended_weights", "storage_fix_start", "storage_fix_end", "storage_roll_end",
    "p_state_upper", "p_state_unitsize", "p_state_self_discharge",
    "p_state_start", "p_state_end", "p_state_existing_capacity",
    "storage_use_reference_value", "p_storage_state_reference_value",
    "p_storage_state_reference_price",
    "node_profile_upper", "node_profile_lower", "node_profile_fixed",
    "p_node_availability", "p_roll_continue_state",
    "n_fix_storage_quantity", "ndt_fix_storage_quantity",
    "p_fix_storage_quantity", "n_fix_storage_usage",
    "ndt_fix_storage_usage", "p_fix_storage_usage", "ndt_fix_storage_price",
    "p_node_capacity_for_scaling",
    # node-keyed invest/divest sets (the trade nodes are not the master's
    # invest variables — only the cross connections are).
    "nd_invest_set", "nd_divest_set",
    # CO2 / reserve / user-constraint / group features — all in-region
    # recourse, not part of the network-only master.  (The commodity
    # frames + ``p_commodity_price`` are REQUIRED-present by the PROCESSES
    # feature, so they are EMPTIED rather than nulled below.)
    "flow_from_co2_priced", "flow_from_co2_priced_noEff",
    "p_co2_content", "p_co2_price",
    "group_co2_max_period", "flow_from_co2_capped",
    "flow_from_co2_capped_noEff", "p_co2_max_period", "group_d_co2_capped",
    "group_co2_max_total", "flow_from_co2_capped_total",
    "flow_from_co2_capped_total_noEff", "p_co2_max_total",
    "flow_constraint_idx", "p_flow_constraint_coef", "p_constraint_constant",
    "cdt_eq", "cdt_le", "cdt_ge",
    "p_node_constraint_invested_capacity_coeff",
    "p_process_constraint_invested_capacity_coeff",
    "p_node_constraint_state_coeff",
    "p_node_constraint_prebuilt_capacity_coeff",
    "p_process_constraint_prebuilt_capacity_coeff",
    "p_node_constraint_capacity_coeff",
    "p_process_constraint_capacity_coeff",
    "p_node_constraint_retired_capacity_coeff",
    "p_process_constraint_retired_capacity_coeff",
    "p_node_constraint_in_construction_capacity_coeff",
    "p_process_constraint_in_construction_capacity_coeff",
    "p_process_constraint_online_coeff",
    "groupCapacityMargin", "groupInertia", "groupNonSync", "group_node",
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
    "p_inv_group_cap", "p_group_capacity_for_scaling",
    "pdGroup_capacity_margin",
)


# ---------------------------------------------------------------------------
# Master-hosted-mode partition of ``_MASTER_NODE_FIELDS``.  With a
# non-empty ``master_hosted_nodes`` set the master KEEPS master-side
# content instead of nulling everything: each tuple below routes its
# fields through the matching keep-filter in ``master_network_data``.
# Every ``_MASTER_NODE_FIELDS`` member appears in exactly one of these
# groups (pinned by a test) so a schema addition to the legacy tuple
# cannot silently skip the master-hosted path.
# ---------------------------------------------------------------------------

#: ``n``-keyed frames/Params → filter rows to the master-hosted nodes
#: (balance/state/storage/profile/availability/fix/invest-set frames;
#: ``group_node`` bare membership rows are filtered silently the same
#: way, mirroring ``split``).
_MASTER_N_KEYED_FIELDS: tuple[str, ...] = (
    "nodeBalance_dt", "nodeBalancePeriod",
    "nodeState", "nodeState_dt", "nodeState_first_dt", "nodeState_last_dt",
    "nodeState_rp", "nodeStateBlock",
    "nodeCoarse", "node_coarse_step", "node_coarse_interior",
    "storage_bind_within_timeblock", "storage_bind_forward_only",
    "storage_bind_within_solve", "storage_bind_within_period", "storage_bind_within_solve_blended_weights",
    "storage_bind_within_period_blended_weights",
    "storage_bind_forward_only_blended_weights", "storage_fix_start", "storage_fix_end", "storage_roll_end",
    "p_state_upper", "p_state_unitsize", "p_state_self_discharge",
    "p_state_start", "p_state_end", "p_state_existing_capacity",
    "storage_use_reference_value", "p_storage_state_reference_value",
    "p_storage_state_reference_price",
    "node_profile_upper", "node_profile_lower", "node_profile_fixed",
    "p_node_availability", "p_roll_continue_state",
    "n_fix_storage_quantity", "ndt_fix_storage_quantity",
    "p_fix_storage_quantity", "n_fix_storage_usage",
    "ndt_fix_storage_usage", "p_fix_storage_usage", "ndt_fix_storage_price",
    "p_node_capacity_for_scaling",
    "nd_invest_set", "nd_divest_set",
    "group_node",
)

#: ``cn``-keyed user-constraint frames/Params → keep rows of the
#: all-master constraint ids (:func:`_master_side_constraint_ids`).
_MASTER_CN_KEYED_FIELDS: tuple[str, ...] = (
    "flow_constraint_idx", "p_flow_constraint_coef",
    "p_constraint_constant", "cdt_eq", "cdt_le", "cdt_ge",
    "p_node_constraint_invested_capacity_coeff",
    "p_process_constraint_invested_capacity_coeff",
    "p_node_constraint_state_coeff",
    "p_node_constraint_prebuilt_capacity_coeff",
    "p_process_constraint_prebuilt_capacity_coeff",
    "p_node_constraint_capacity_coeff",
    "p_process_constraint_capacity_coeff",
    "p_node_constraint_retired_capacity_coeff",
    "p_process_constraint_retired_capacity_coeff",
    "p_node_constraint_in_construction_capacity_coeff",
    "p_process_constraint_in_construction_capacity_coeff",
    "p_process_constraint_online_coeff",
)

#: Arc-keyed CO2-priced flow frames → keep master-local arc rows
#: (coupling-arc rows hard-error in :func:`_co2_master_partition`).
_MASTER_CO2_ARC_FIELDS: tuple[str, ...] = (
    "flow_from_co2_priced", "flow_from_co2_priced_noEff",
)

#: ``g``-keyed CO2-cap frames → keep rows of the all-master capped
#: groups (:func:`_co2_master_partition`).
_MASTER_CO2_GROUP_FIELDS: tuple[str, ...] = (
    "group_co2_max_period", "flow_from_co2_capped",
    "flow_from_co2_capped_noEff", "p_co2_max_period",
    "group_d_co2_capped",
    "group_co2_max_total", "flow_from_co2_capped_total",
    "flow_from_co2_capped_total_noEff", "p_co2_max_total",
)

#: Pure lookup tables consumed by the kept CO2 rows via joins — carried
#: whole (rows for un-kept groups/commodities are inert).
_MASTER_CO2_LOOKUP_FIELDS: tuple[str, ...] = (
    "p_co2_content", "p_co2_price",
)

#: ``p``-keyed group-feature side frames → filter to the master's
#: kept processes (rows for absent flows are inert Where-filters).
_MASTER_GROUP_PROC_FIELDS: tuple[str, ...] = (
    "process_sink_inertia", "process_source_inertia",
    "process_sink_nonSync", "process_group_inside_nonSync",
)


def master_network_data(
    data: FlexData,
    regions: list[str],
    *,
    region_membership: dict[str, dict[str, set[str]]] | None = None,
    master_hosted_nodes: frozenset[str] = frozenset(),
) -> FlexData:
    """Build the Benders MASTER's reduced :class:`FlexData` — the INVERSE
    of :func:`split`.

    Returns a reduced :class:`FlexData` containing ONLY the cross-region
    ``(p, source, sink)`` arcs plus their invest / cost / timeline params,
    with no balance node (every node is in a region; the block / state /
    inflow frames are emptied too).  ``build_flextool`` over the result
    generates the master skeleton natively:

    * ``v_flow[conn, source, sink, d, t]`` for every cross arc (built from
      ``process_source_sink × dt``, independent of ``nodeBalance``);
    * ``v_invest_p[conn, d]`` over ``pd_invest_set``;
    * the capacity-tied ``maxFlow`` row (greenfield ⇒ ``flow_upper_rhs=0``
      with ``-v_invest_p`` on the LHS ⇒ ``v_flow ≤ v_invest_p``);
    * the invest annuity cost + (when authored) the connection flow cost.

    Reuses the cross-arc CLASSIFICATION (:func:`_classify_arcs`) — the same
    detection :func:`split` uses — rather than re-deriving it.

    Parameters
    ----------
    data
        Whole-system :class:`FlexData`.
    regions
        The region group names the splitter partitions on.
    region_membership
        Pre-computed membership (see :func:`load_region_membership`); when
        omitted, re-derived from *data*.
    master_hosted_nodes
        Master-hosted node mode (see :func:`compute_master_hosted_nodes`
        / :func:`split`).  With the default EMPTY set the function takes
        today's exact path (explicit early branches — byte-identical).
        Non-empty: the master KEEPS the master-side model instead of
        emptying it — see Notes.

    Notes
    -----
    With the default empty ``master_hosted_nodes``, the reduced FlexData
    KEEPS only cross-arc rows in the process / arc / arc-cost /
    arc-block frames and the cross connections in the invest / unitsize
    / max-units params; EVERY node-keyed frame (balance, state, inflow,
    penalties, storage, profiles, CO2, groups, user constraints) is
    emptied (``None``).

    With a NON-EMPTY ``master_hosted_nodes``:

    * the arc keep-set widens to cross-region ∪ region↔master coupling
      ∪ master-local arcs (and the proc/cost keep-sets follow), so
      master-local connections AND units build natively in the master;
    * ``n``-keyed frames are FILTERED to the master-hosted nodes instead
      of nulled (balance, inflow, penalties, storage/state, profiles,
      availability, fix-storage, node invest sets);
    * the entity-keyed invest frames keep
      ``procs-of-kept-arcs ∪ master_hosted_nodes`` (master storage
      invest needs the ``e``-keyed annuity/lifetime/max-units rows);
    * the master-hosted nodes are the master's balance nodes (on the
      plain path the master has none);
    * user constraints / CO2 caps / group features whose referenced
      entities are ALL master-side are kept; straddling ones hard-error
      (same validation family as :func:`split`).

    In both modes the solve-data-keyed timeline frames (``dt``,
    ``p_step_duration``, ``p_timestep_weight``, ``p_inflation_op``,
    ``p_period_share``, the RP / block frames) carry through the
    ``dataclasses.replace`` shallow copy unchanged, so the master's
    ``v_flow`` lives on the SAME ``(d, t)`` grid as the region pinned
    half-flows (Phase-3 §3.5 guard (a)).
    """
    if region_membership is None:
        region_membership = load_region_membership(data, regions)
    region_nodes = {r: m["nodes"] for r, m in region_membership.items()}

    new = dataclasses.replace(data)

    if data.process_source_sink is None:
        raise RuntimeError(
            "master_network_data: no process_source_sink — nothing to "
            "decompose"
        )

    # Reuse the cross-region classification (same detection as ``split``).
    # With the default empty master set the extra classification frames
    # are empty and this path is byte-identical to the historical 2-way
    # behaviour.
    # The units the master holds whole (:func:`master_units`) are master-side endpoints of their arcs, as in
    # ``split``: their arcs at region nodes are couplings, their other arcs master-local.
    in_master = master_units(data, region_nodes, master_hosted_nodes)
    master_mode = bool(master_hosted_nodes) or bool(in_master)
    _pss_tagged, cross, region_master, master_local = _classify_arcs(
        data.process_source_sink, region_nodes,
        master_nodes=frozenset(master_hosted_nodes) | in_master,
    )
    cross_keys: set[tuple[str, str, str]] = {
        (r["p"], r["source"], r["sink"]) for r in cross.iter_rows(named=True)
    }
    master_cns: set[str] = set()
    master_groups_feature: set[str] = set()
    master_groups_co2: set[str] = set()
    master_local_keys: set[tuple[str, str, str]] = set()
    if not master_mode:
        # Today's exact path: region↔region cross arcs are the only
        # master content.
        if cross.height == 0:
            raise RuntimeError(
                "master_network_data: no cross-region arcs found"
            )
    else:
        all_region_nodes: set[str] = set()
        for ns in region_nodes.values():
            all_region_nodes |= ns
        overlap = set(master_hosted_nodes) & all_region_nodes
        if overlap:
            raise RuntimeError(
                f"master_network_data: master_hosted_nodes overlap "
                f"region membership: {sorted(overlap)} — a node is "
                f"either master-hosted (in no region group) or in "
                f"exactly one region, never both."
            )
        region_master_keys = {
            (r["p"], r["source"], r["sink"])
            for r in region_master.iter_rows(named=True)
        }
        master_local_keys = {
            (r["p"], r["source"], r["sink"])
            for r in master_local.iter_rows(named=True)
        }
        master_balance_nodes = set()
        for fld in ("nodeBalance", "nodeState"):
            f = getattr(data, fld, None)
            if f is not None and f.height > 0:
                master_balance_nodes |= (
                    set(f["n"].cast(pl.Utf8).to_list())
                    & set(master_hosted_nodes))
        if (not cross_keys and not region_master_keys
                and not master_local_keys and not master_balance_nodes):
            raise RuntimeError(
                "master_network_data: no coupling arcs and no master "
                "content — nothing to host in the Benders master"
            )
        # Same hard-validation family as ``split`` (D-a / R5): never
        # silently mis-partition authored data.  The messages are
        # identical to ``split``'s, so a driver run that already raised
        # there cannot raise differently here.
        _ml_procs = _master_local_procs(
            data.process_source_sink, master_local) | set(in_master)
        _validate_user_constraints(
            data, all_region_nodes, master_hosted_nodes, _ml_procs)
        master_cns = _master_side_constraint_ids(
            data, all_region_nodes, master_hosted_nodes, _ml_procs)
        master_groups_feature = _master_side_feature_groups(
            data, all_region_nodes, master_hosted_nodes)
        master_groups_co2 = _co2_master_partition(
            data, master_local_keys, region_master_keys, _ml_procs)
        # Arc keep-set = cross ∪ region↔master coupling ∪ master-local
        # (audit 0.D: the coupling-arc-only case is load-bearing).
        cross_keys = cross_keys | region_master_keys | master_local_keys
    cross_procs: set[str] = {k[0] for k in cross_keys}
    # Entity keep-set for the ``e``-keyed invest frames: master-hosted
    # STORAGE invest needs the node entities alongside the connections/
    # units of the kept arcs.
    entity_keep: set[str] = (
        cross_procs | set(master_hosted_nodes) if master_mode
        else cross_procs
    )

    _enums = getattr(data, "_axis_enums", None) or get_global_axis_enums()

    def _keep_cross_triple(df: pl.DataFrame | None) -> pl.DataFrame | None:
        """Keep ONLY the rows whose (p, source, sink) is a cross arc."""
        if df is None:
            return None
        if not all(c in df.columns for c in ("p", "source", "sink")):
            return df
        key_df = pl.DataFrame(
            {
                "p":      [t[0] for t in cross_keys],
                "source": [t[1] for t in cross_keys],
                "sink":   [t[2] for t in cross_keys],
            },
            schema={"p": schema_dtype(_enums, "p"),
                    "source": schema_dtype(_enums, "source"),
                    "sink": schema_dtype(_enums, "sink")},
        )
        return df.join(key_df, on=("p", "source", "sink"), how="semi")

    def _keep_cross_triple_param(p: Param | None) -> Param | None:
        if p is None:
            return None
        if not all(c in p.dims for c in ("p", "source", "sink")):
            # p-keyed but not arc-keyed (e.g. p_unitsize): keep cross procs.
            return _keep_proc_param(p)
        f = _keep_cross_triple(p.frame)
        return Param(p.dims, f, name=p.name)

    def _keep_proc(df: pl.DataFrame | None) -> pl.DataFrame | None:
        if df is None or "p" not in df.columns:
            return df
        return df.filter(_is_in_keep("p", cross_procs))

    def _keep_proc_param(p: Param | None) -> Param | None:
        if p is None:
            return None
        if "p" not in p.dims:
            return p
        return Param(p.dims, p.frame.filter(_is_in_keep("p", cross_procs)),
                     name=p.name)

    def _keep_entity_param(p: Param | None) -> Param | None:
        """Keep only the cross connections (+ master-hosted nodes in
        master mode) on an entity-axis (``e``) param."""
        if p is None:
            return None
        if "e" not in p.dims:
            return p
        return Param(p.dims, p.frame.filter(_is_in_keep("e", entity_keep)),
                     name=p.name)

    def _keep_entity_frame(df):
        """Keep cross connections (+ master-hosted nodes in master mode)
        on an ``e``-axis frame OR Param (some invest sets ship as plain
        DataFrames, others as Params)."""
        if df is None:
            return None
        if isinstance(df, Param):
            return _keep_entity_param(df)
        if "e" not in df.columns:
            return df
        return df.filter(_is_in_keep("e", entity_keep))

    # ---- Process / arc topology: KEEP only cross arcs ----
    new.process_source_sink = _keep_cross_triple(data.process_source_sink)
    new.process_source_sink_eff = _keep_cross_triple(data.process_source_sink_eff)
    new.process_source_sink_noEff = _keep_cross_triple(data.process_source_sink_noEff)
    # One-variable connections: the master's v_flow_back (and maxFlow_back) only on the kept arcs (item 301).
    new.process_source_sink_2way_1var = _keep_cross_triple(data.process_source_sink_2way_1var)
    new.pss_dt = None  # rebuilt on demand from process_source_sink × dt.
    new.process_source_canonical = _keep_proc(data.process_source_canonical)
    new.process_sink_canonical = _keep_proc(data.process_sink_canonical)
    new.flow_to_n = _keep_cross_triple(data.flow_to_n)
    new.flow_from_n = _keep_cross_triple(data.flow_from_n)
    new.flow_from_nodeBalance_eff = _keep_cross_triple(data.flow_from_nodeBalance_eff)
    new.flow_from_nodeBalance_noEff = _keep_cross_triple(data.flow_from_nodeBalance_noEff)
    new.process_unit = _keep_proc(data.process_unit)
    new.process_indirect = _keep_proc(data.process_indirect)
    # Commodity frames + price are REQUIRED-present by the PROCESSES feature.
    # Cross arcs (trade pipes) are not commodity-fed, so keeping only their
    # rows EMPTIES these frames while preserving the schema (not None).
    new.flow_from_commodity_eff = _keep_cross_triple(data.flow_from_commodity_eff)
    new.flow_from_commodity_noEff = _keep_cross_triple(data.flow_from_commodity_noEff)
    new.flow_to_commodity = _keep_cross_triple(data.flow_to_commodity)
    new.p_commodity_price = _keep_proc_param(data.p_commodity_price)

    # ---- Per-arc operating / capacity params ----
    new.p_unitsize = _keep_proc_param(data.p_unitsize)
    new.p_all_entity_unitsize = _keep_entity_param(data.p_all_entity_unitsize)
    new.p_flow_upper = _keep_cross_triple_param(data.p_flow_upper)
    new.p_flow_upper_existing = _keep_cross_triple_param(data.p_flow_upper_existing)
    new.p_arc_max_cap_coef = _keep_cross_triple_param(data.p_arc_max_cap_coef)
    new.p_indirect_input_cap = _keep_cross_triple_param(
        data.p_indirect_input_cap)
    new.process_source_sink_uncapped = _keep_cross_triple(
        data.process_source_sink_uncapped)
    new.p_slope = _keep_proc_param(data.p_slope)
    # piecewise_linear units with two or more lines: only the kept processes (none are cross connections).
    for _f in ("process_piecewise", "process_piecewise_needs_online", "process_piecewise_remap",
               "p_piecewise_points"):
        setattr(new, _f, _keep_proc(getattr(data, _f)))
    new.p_piecewise_slope = _keep_proc_param(data.p_piecewise_slope)
    new.p_piecewise_intercept = _keep_proc_param(data.p_piecewise_intercept)
    new.p_process_existing_count = _keep_proc_param(data.p_process_existing_count)
    new.p_process_availability = _keep_proc_param(data.p_process_availability)
    # Ramp limits and ramp costs: only the kept (cross) arcs.
    for _f in ("process_source_sink_ramp_limit_sink_up", "process_source_sink_ramp_limit_sink_down",
               "process_source_sink_ramp_limit_source_up", "process_source_sink_ramp_limit_source_down",
               "process_source_sink_ramp_cost"):
        setattr(new, _f, _keep_cross_triple(getattr(data, _f)))
    new.p_ramp_cost_arc = _keep_cross_triple_param(getattr(data, "p_ramp_cost_arc", None))
    new.p_roll_continue_flow = _keep_cross_triple_param(data.p_roll_continue_flow)
    new.p_roll_continue_flow_duration = _keep_cross_triple_param(getattr(data, "p_roll_continue_flow_duration", None))
    new.pd_neg_cap = _keep_proc_param(data.pd_neg_cap)

    # ---- Profiles (process-keyed) ----
    new.process_profile_upper = _keep_proc(data.process_profile_upper)
    new.process_profile_lower = _keep_proc(data.process_profile_lower)
    new.process_profile_fixed = _keep_proc(data.process_profile_fixed)

    # ---- Per-arc / per-process variable-cost frames (flow cost) ----
    new.pssdt_varCost_eff_connection = _keep_cross_triple(
        getattr(data, "pssdt_varCost_eff_connection", None))
    new.pssdt_varCost_eff_unit_source = _keep_cross_triple(
        getattr(data, "pssdt_varCost_eff_unit_source", None))
    new.pssdt_varCost_eff_unit_sink = _keep_cross_triple(
        getattr(data, "pssdt_varCost_eff_unit_sink", None))
    new.pssdt_varCost_noEff = _keep_cross_triple(
        getattr(data, "pssdt_varCost_noEff", None))
    new.p_pssdt_varCost = _keep_cross_triple_param(getattr(data, "p_pssdt_varCost", None))
    new.p_pdt_varCost_source = _keep_proc_param(getattr(data, "p_pdt_varCost_source", None))
    new.p_pdt_varCost_sink = _keep_proc_param(getattr(data, "p_pdt_varCost_sink", None))
    new.p_pdt_varCost_process = _keep_proc_param(getattr(data, "p_pdt_varCost_process", None))

    # ---- Invest params for the cross connections ----
    new.pd_invest_set = _keep_proc(data.pd_invest_set)
    new.pd_divest_set = _keep_proc(data.pd_divest_set)
    new.ed_invest_set = _keep_entity_frame(data.ed_invest_set)
    new.ed_divest_set = _keep_entity_frame(data.ed_divest_set)
    new.cohort = _keep_entity_frame(getattr(data, "cohort", None))
    new.cohort_alive = _keep_entity_frame(getattr(data, "cohort_alive", None))
    new.ed_invest_commission = _keep_entity_frame(getattr(data, "ed_invest_commission", None))
    new.edd_invest_commission = _keep_entity_frame(getattr(data, "edd_invest_commission", None))
    new.edd_invest_in_construction = _keep_entity_frame(getattr(data, "edd_invest_in_construction", None))
    new.e_construction_lead_time = _keep_entity_frame(getattr(data, "e_construction_lead_time", None))
    new.p_entity_carried_commission = _keep_entity_param(getattr(data, "p_entity_carried_commission", None))
    new.p_entity_carried_in_construction = _keep_entity_param(
        getattr(data, "p_entity_carried_in_construction", None))
    new.p_entity_retired_carried = _keep_entity_param(getattr(data, "p_entity_retired_carried", None))
    new.p_entity_max_units = _keep_entity_param(data.p_entity_max_units)
    new.ed_lifetime_fixed_cost = _keep_entity_param(data.ed_lifetime_fixed_cost)
    new.ed_lifetime_fixed_cost_divest = _keep_entity_param(data.ed_lifetime_fixed_cost_divest)
    new.ed_entity_annual_discounted = _keep_entity_param(data.ed_entity_annual_discounted)
    for name in ("ed_divest_salvage", "ed_existing_salvage_credit", "ed_existing_replacement",
                 "ed_entity_annual_discounted_run",
                 "p_entity_pre_existing", "cd_divest_built_cost",
                 "cd_divest_built_fixed", "cd_extend_cost", "cd_extend_fixed"):
        setattr(new, name, _keep_entity_param(getattr(data, name, None)))
    for name in ("cd_divest_built", "cdd_divest_built", "cohort_divest_built", "cd_extend", "cdd_extend",
                 "cohort_extend", "edl_invest_lines_run", "edl_divest_lines", "edl_existing_credit_lines",
                 "edl_existing_replacement_lines", "cdl_divest_built_lines", "cdl_extend_lines"):
        setattr(new, name, _keep_entity_frame(getattr(data, name, None)))
    # Investment / retirement limits of the master's own entities only (the master counterpart of item 286: an
    # entity floor of a region entity would be a master row with an empty left-hand side).
    for name in _ENTITY_LIMIT_FIELDS:
        setattr(new, name, _keep_entity_frame(getattr(data, name, None)))
    new.p_entity_previously_invested_capacity = _keep_entity_param(
        data.p_entity_previously_invested_capacity)
    new.p_entity_invested = _keep_entity_param(data.p_entity_invested)
    new.p_entity_divested = _keep_entity_param(data.p_entity_divested)
    new.p_entity_all_existing = _keep_entity_param(data.p_entity_all_existing)
    new.p_ed_fixed_cost = _keep_entity_param(data.p_ed_fixed_cost)

    # ---- Node-keyed / recourse frames ----
    if not master_mode:
        # Today's exact path: drop in-region features that reference the
        # OMITTED terminal nodes (or in-region recourse not part of the
        # network-only master).
        for fld in _MASTER_NODE_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld, None)
    else:
        # Master-hosted mode: KEEP the master-side content.  Every
        # ``_MASTER_NODE_FIELDS`` member is routed through exactly one
        # of the keep-filters below (partition pinned by test).
        def _filter_field(obj, col: str, keep: set[str]):
            """Filter a frame OR Param to ``col ∈ keep`` (no-op on
            ``None`` / missing column, mirroring the region-side
            helpers)."""
            if obj is None:
                return None
            if isinstance(obj, Param):
                return _filter_param(obj, col, keep)
            return _filter_frame(obj, col, keep)

        def _semi_triples(obj, keys: set[tuple[str, str, str]]):
            """Keep only the rows whose (p, source, sink) is in *keys*
            (frame or Param)."""
            if obj is None:
                return None
            frame = obj.frame if isinstance(obj, Param) else obj
            if not all(c in frame.columns for c in ("p", "source", "sink")):
                return obj
            key_df = pl.DataFrame(
                {
                    "p":      [t[0] for t in keys],
                    "source": [t[1] for t in keys],
                    "sink":   [t[2] for t in keys],
                },
                schema={"p": schema_dtype(_enums, "p"),
                        "source": schema_dtype(_enums, "source"),
                        "sink": schema_dtype(_enums, "sink")},
            )
            out = frame.join(key_df, on=("p", "source", "sink"), how="semi")
            if isinstance(obj, Param):
                return Param(obj.dims, out, name=obj.name)
            return out

        master_set = set(master_hosted_nodes)
        for fld in _MASTER_N_KEYED_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "n",
                                      master_set))
        for fld in _MASTER_CN_KEYED_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "cn",
                                      master_cns))
        for fld in _MASTER_CO2_ARC_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _semi_triples(getattr(data, fld, None),
                                      master_local_keys))
        for fld in _MASTER_CO2_GROUP_FIELDS:
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "g",
                                      master_groups_co2))
        # ``_MASTER_CO2_LOOKUP_FIELDS`` (p_co2_content / p_co2_price):
        # pure lookup tables joined by the kept CO2 rows — carried whole
        # through the shallow copy (rows for un-kept groups are inert,
        # and the PROCESSES feature requires presence).
        for fld in (_GROUP_FEATURE_SET_FIELDS
                    + _GROUP_FEATURE_PARAM_FIELDS):
            if hasattr(new, fld):
                setattr(new, fld,
                        _filter_field(getattr(data, fld, None), "g",
                                      master_groups_feature))
        for fld in _MASTER_GROUP_PROC_FIELDS:
            if hasattr(new, fld):
                obj = getattr(data, fld, None)
                setattr(new, fld,
                        _keep_proc_param(obj) if isinstance(obj, Param)
                        else _keep_proc(obj))

    # Start-up input draws (startup_input_method): none in the network-only master; in master-hosted mode the
    # units of the kept arcs (the master-local units: a region unit's draw is in its region subproblem) at a
    # master-hosted node or a node of no region (a commodity node, priced like the unit's fuel arc).  A master
    # unit drawing from a region node is refused by split (_validate_master_units).
    _startup_pn = getattr(data, "process_startup_node", None)
    _region_node_union = {n for ns in region_nodes.values() for n in ns}
    startup_nodes_kept = (set() if _startup_pn is None else
                          {str(n) for n in _startup_pn.get_column("n").to_list()} - _region_node_union)
    for fld in STARTUP_FIELDS:
        obj = getattr(data, fld, None)
        if master_mode and obj is not None:
            if isinstance(obj, Param):
                obj = _filter_param(_filter_param(obj, "p", cross_procs), "n", startup_nodes_kept)
            else:
                obj = _filter_frame(_filter_frame(obj, "p", cross_procs), "n", startup_nodes_kept)
            if (obj.frame if isinstance(obj, Param) else obj).height == 0:
                obj = None
        else:
            obj = None
        setattr(new, fld, obj)

    # Balance: only the master-hosted nodes (none on the plain path).  Every region node, the ends of the
    # coupling arcs included, has its balance in its region, so the master's coupling ``v_flow`` is free except
    # for ``maxFlow``, its bound and the master's own rows.  ``build_flextool`` builds a master without a balance
    # node with no balance slacks or rows.  The frames stay present (filtered, not None): ALWAYS needs them.
    master_set = set(master_hosted_nodes)
    new.nodeBalance = _filter_frame(data.nodeBalance, "n", master_set)
    new.p_inflow = _filter_param(data.p_inflow, "n", master_set)
    new.p_penalty_up = _filter_param(data.p_penalty_up, "n", master_set)
    new.p_penalty_down = _filter_param(data.p_penalty_down, "n", master_set)
    # Every master flow keeps the solve's step: the regions pin the master's half-flows per fine step.
    new.processCoarse = None
    new.process_coarse_step = None

    return new
