"""Layer 2 (semantic per-type scaling) family registries.

This module declares, for every variable and constraint family FlexTool
emits, the :class:`QuantityType` that determines its Layer-2 scaling
group.

The registry is populated by walking every ``add_var(...)`` and
``add_cstr(...)`` call site in ``flextool/engine/`` (see comments
on each entry).  An unregistered name raises :class:`KeyError` at
lookup time — Layer 2 refuses to silently default, on principle (see
``feedback_no_shortcuts`` user memory).

Per :class:`CstrFamily` semantics:

* ``rhs_type=None`` is a sentinel for *user-supplied composite-LHS
  constraints* (the ``process_constraint_*`` / ``node_balance_fix_*``
  family) and the small number of structural rows whose RHS is
  identically zero (``dc_flow_eq``, ``fix_v_invest_no_investment_eq``)
  — Layer 2 has no per-row scaling to apply there.  The column scaler
  on each LHS variable still propagates through these rows.
* ``member_class_resolver="group_capacity"`` is the trigger for
  :func:`resolve_group_capacity_type` — the constraint name itself is
  ``[MW or MWh]`` ambiguous (group invest / divest / cumulative
  capacity); apply-time inspects the constraint suffix to pick POWER
  vs ENERGY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._quantity_types import QuantityType


@dataclass(frozen=True)
class VarFamily:
    """Per-variable Layer-2 metadata.

    Attributes
    ----------
    column_type:
        :class:`QuantityType` of one unit of the variable as it appears
        in the LP **column** (before any per-row coefficient
        multiplication).  Layer 2 uses this to pick the column scaler.
    multiplier_param:
        Name of the parameter (e.g. ``"p_unitsize"``) whose magnitude
        the column scaler is *expected* to absorb so the matrix entry
        ``A_{ij} = multiplier * coef`` lands near 1.  ``None`` if no
        such single dominant multiplier applies.  Informational; the
        Layer-2 bucketing in :func:`_layer2.bucket_coefficients`
        derives factors from observed magnitudes, not from this hint.
    """

    column_type: QuantityType
    multiplier_param: Optional[str] = None


@dataclass(frozen=True)
class CstrFamily:
    """Per-constraint Layer-2 metadata.

    Attributes
    ----------
    rhs_type:
        :class:`QuantityType` of the RHS / row-bound vector for this
        family.  Layer 2 uses this to pick the row scaler.  ``None``
        explicitly opts out of per-row scaling (user-defined
        composite-LHS constraints; zero-RHS structural rows).
    member_class_resolver:
        When set to ``"group_capacity"``, Layer 2 ignores ``rhs_type``
        and calls :func:`resolve_group_capacity_type` with the
        constraint's suffix (``_p`` / ``_n``) to choose POWER vs ENERGY
        at apply time.  Used by the
        ``maxInvestGroup_entity_*`` / ``maxDivestGroup_entity_*`` /
        ``maxInvest_entity_total*`` / ``maxDivest_entity_total*`` /
        ``maxCumulative_capacity`` / ``minCumulative_capacity`` /
        ``minInvest_entity_total*`` / ``minDivest_entity_total*``
        families that carry ``[MW or MWh]`` group-capacity RHS params.
    """

    rhs_type: Optional[QuantityType]
    member_class_resolver: Optional[str] = None


# ---------------------------------------------------------------------------
# Variable families.
#
# The Layer-2 ``column_type`` is the type of one unit of the LP column
# variable itself.  Matrix-entry physical units are
# ``column_type × multiplier_param`` (e.g. v_flow's column is
# DIMENSIONLESS; its row coefficient is ``p_unitsize`` → POWER on the
# row).  Bucketing in Layer 2 uses the *effective* type per (row, col)
# pair, not the column type alone.

VARIABLE_FAMILIES: dict[str, VarFamily] = {
    # ── Process / connection flow (dispatched per timestep).
    # flextool/engine/model/core.py _declare_variables
    "v_flow": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),

    # ── Storage / nodeBalance slacks.  Carry ENERGY directly (no
    # unitsize multiplier on the column).
    # flextool/engine/model/core.py _declare_variables
    "vq_state_up": VarFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _declare_variables
    "vq_state_down": VarFamily(QuantityType.ENERGY),

    # ── Storage state (per-step, per-block).  ENERGY column.
    # flextool/engine/model/core.py _declare_variables
    "v_state": VarFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _declare_variables
    "v_state_inter": VarFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _declare_variables
    "v_state_rp_start": VarFamily(QuantityType.ENERGY),

    # ── Unit-commitment families.  Column counts a number of units
    # online / starting / shutting; DIMENSIONLESS.  The matrix coef
    # ``p_unitsize`` (or ``v_unitsize`` from process_existing_count)
    # turns the row into POWER as needed.  Integer variants are
    # mathematically the same column type.
    # flextool/engine/model/_online.py add_variables
    "v_online_linear": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/_online.py add_variables
    "v_startup_linear": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/_online.py add_variables
    "v_shutdown_linear": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/_online.py add_variables
    "v_online_integer": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/_exact.py add_variables: the direction switch of an exact connection (binary, a
    # direction, not a capacity: no unitsize multiplier).
    "v_exact_forward": VarFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py add_variables
    "v_startup_integer": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/_online.py add_variables
    "v_shutdown_integer": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),

    # ── Investment / divestment counts.  DIMENSIONLESS column whose
    # matrix coefficient ``p_unitsize`` (process / connection) or
    # ``p_state_unitsize`` (node) sets the row's effective POWER /
    # ENERGY type.
    # flextool/engine/model/core.py _declare_variables
    "v_invest_p": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/core.py _declare_variables
    "v_divest_p": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/core.py _declare_variables
    "v_invest_n": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_state_unitsize"),
    # flextool/engine/model/core.py _declare_variables
    "v_divest_n": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_state_unitsize"),

    # ── Reserve.  DIMENSIONLESS column; ``p_unitsize`` multiplier
    # carries the per-row POWER coefficient.
    # flextool/engine/model/_reserve.py add_variables
    "v_reserve": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine/model/_reserve.py add_variables
    "vq_reserve": VarFamily(QuantityType.POWER),
    # flextool/engine/model/_reserve.py add_variables (total reserve of an n-1 group, MW)
    "v_reserve_total": VarFamily(QuantityType.POWER),

    # ── Commodity ladder.  v_trade is DIMENSIONLESS; row coef
    # ``unitsize`` turns it into ENERGY for the ladder caps and the
    # balance row.
    # flextool/engine/model/_commodity_ladder.py add_variables
    "v_trade": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="unitsize"),

    # ── Group-level slacks.  Direct physical units.
    # flextool/engine/model/_group_slack.py add_variables
    "vq_capacity_margin": VarFamily(QuantityType.POWER),
    # flextool/engine/model/_group_slack.py add_variables
    "vq_inertia": VarFamily(QuantityType.INERTIA),
    # flextool/engine/model/_group_slack.py add_variables
    "vq_non_synchronous": VarFamily(QuantityType.POWER),

    # ── DC power flow.
    # ``v_angle`` (radians).  Dimensionless in our taxonomy.
    # flextool/engine/model/_dc_power_flow.py add_variables
    "v_angle": VarFamily(QuantityType.DIMENSIONLESS),
    # ``v_flow_back`` mirrors v_flow on the reverse direction.
    # flextool/engine/model/core.py _declare_variables
    "v_flow_back": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),

    # ── Benders (Option C) hand-built master families.  The hand master
    # (``master="hand"``) is built AUTOSCALE-OFF, so Layer 2 never scales
    # these columns; registering them keeps the registry-coverage invariant
    # complete (CLAUDE.md #1).
    # ``f`` = cross-region trade flow, in the SAME unitsize-normalised units
    # as the region half-flow ``v_flow`` ⇒ same family as v_flow.
    # flextool/engine/solve/_benders.py:483
    "f": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # ``C`` = invested trade-connection capacity (normalised) ⇒ same family
    # as v_invest_p.
    # flextool/engine/solve/_benders.py:473
    "C": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # ``eta`` = per-region recourse cost (enters the objective at coef 1.0
    # carrying the region's currency cost) ⇒ CURRENCY column.
    # flextool/engine/solve/_benders.py:489
    "eta": VarFamily(QuantityType.CURRENCY),
}


# ---------------------------------------------------------------------------
# Constraint families.
#
# The key is either the full constraint name (exact match) or a prefix
# (the constraint name's ``startswith(key + "_")``).  Suffixed UC
# families (``maxOnline_linear`` / ``maxOnline_integer``) are matched
# by prefix; the per-family Layer-2 type is the same for both because
# they carry the same RHS structure.
#
# ``rhs_type=None`` ⇒ Layer 2 does NOT apply a per-row factor to that
# family (the rows participate in column scaling only).

# Suffix-resolved (`_p` ⇒ POWER, `_n` ⇒ ENERGY) group capacity rows
# are flagged via member_class_resolver="group_capacity".

CONSTRAINT_FAMILIES: dict[str, CstrFamily] = {
    # ── Node balance (the central energy equation) — ENERGY.
    # flextool/engine/model/core.py _eq_node_balance
    "nodeBalance_eq": CstrFamily(QuantityType.ENERGY),
    # Per-period energy balance for balance_within_period nodes — ENERGY.
    # flextool/engine/model/core.py _eq_node_balance_period
    "nodeBalancePeriod_eq": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_block_coupling
    "nodeBalanceBlock_eq": CstrFamily(QuantityType.ENERGY),
    # Per-coarse-step energy balance of coarse new_stepduration group nodes — ENERGY.
    # flextool/engine/model/core.py _eq_node_balance_coarse
    "nodeBalanceCoarse_eq": CstrFamily(QuantityType.ENERGY),

    # ── Storage state max / state ladder.  ENERGY.
    # flextool/engine/model/core.py _eq_max_state
    "maxState": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_rp_storage_chain
    "maxState_rp_start": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_block_coupling
    "stateConstantWithinBlock_eq": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_node_balance_coarse
    "stateConstantCoarse_eq": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_rp_storage_chain
    "rp_inter_period_balance": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_rp_storage_chain
    "rp_inter_period_cyclic": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_rp_storage_chain
    "rp_inter_period_max_state": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_storage_start_binding
    "storage_state_start_binding": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_storage_start_binding_cyclic
    "storage_state_start_binding_cyclic_period": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_storage_end_binding
    "storage_state_end_binding": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_storage_reference_value
    "storage_state_solve_horizon_reference_value": CstrFamily(QuantityType.ENERGY),

    # ── Flow caps from process side.  DIMENSIONLESS (capacity coefficient).
    # The cap is a fraction × existing_count; the row's effective type
    # lives in the coefficient, not the RHS.
    # flextool/engine/model/core.py _eq_max_flow
    "maxFlow": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/core.py _eq_max_flow
    "maxFlow_negCap": CstrFamily(QuantityType.DIMENSIONLESS),
    # Reverse-flow capacity cap on v_flow_back for method_2way_1var_off
    # arcs (DC + non-DC).  flextool/engine/model/core.py (maxFlow_back).
    "maxFlow_back": CstrFamily(QuantityType.DIMENSIONLESS),
    # Coarse processes: the arc-local rows averaged over each coarse step (Σ_t w·row_t, Σ_t w = 1) keep their
    # fine family's type.  flextool/engine/model/_coarse_flows.py add_cstr (f"{name}_coarse").  Explicit: the
    # fine-family prefixes would otherwise resolve them silently.
    "maxFlow_coarse": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxFlow_negCap_coarse": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxFlow_back_coarse": CstrFamily(QuantityType.DIMENSIONLESS),
    # Direction switch of exact connections, flextool/engine/model/_exact.py.  Explicit: the "maxFlow" prefix
    # would otherwise resolve them silently.
    "maxFlow_exact_forward": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxFlow_exact_reverse": CstrFamily(QuantityType.DIMENSIONLESS),
    # Indirect-unit capacity on the SUM of output flows (+ its neg-cap
    # ``≥`` twin).  flextool/engine/model/core.py (_eq_max_output_sum).
    "maxOutputSum": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxOutputSum_negCap": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxOutputSum_coarse": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxOutputSum_negCap_coarse": CstrFamily(QuantityType.DIMENSIONLESS),
    # Legacy name for the DC-only back-flow cap; the cap is now emitted as
    # ``maxFlow_back`` over the full arc set.  Retained for back-compat.
    "maxToSink_back": CstrFamily(QuantityType.DIMENSIONLESS),
    # Ramp constraints, prefix-matched: ramp_<side>_<dir>_constraint.
    # flextool/engine/model/core.py _eq_ramp
    "ramp": CstrFamily(QuantityType.DIMENSIONLESS),

    # ── Investment caps — entity scope.
    # flextool/engine/model/core.py _eq_invest_max_units
    "maxInvest_var_bound": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_invest_max_units
    "maxDivest_var_bound": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_invest_max_units
    "maxInvest_var_bound_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_invest_max_units
    "maxDivest_var_bound_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_invest_period_caps
    "maxInvest_entity_period_p": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_invest_period_caps
    "maxInvest_entity_period_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_invest_period_caps
    "maxDivest_entity_period_p": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_invest_period_caps
    "maxDivest_entity_period_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_invest_total_caps
    "maxInvest_entity_total": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_invest_total_caps
    "maxInvest_entity_total_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_invest_total_caps
    "maxDivest_entity_total": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_invest_total_caps
    "maxDivest_entity_total_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/core.py _eq_retired_existing
    "maxRetired_existing_p": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/core.py _eq_retired_existing
    "maxRetired_existing_n": CstrFamily(QuantityType.ENERGY),

    # Slice D per-path total caps (design §7.2/§14) — one cap row per
    # (entity, scenario-leaf).  Emitted by
    # ``_eq_invest_total_caps`` (model/core.py) when >1 scenario leaf exists
    # (recourse); the leaf-suffixed sums replace the cross-scenario
    # ``over=("d",)`` total.  Registered EXACTLY (``_path`` is not a
    # known ``lookup_cstr`` suffix, so a missing row silently reverts to
    # an un-scaled LP under STRICT).
    "maxInvest_entity_total_path": CstrFamily(QuantityType.POWER),
    "maxInvest_entity_total_path_n": CstrFamily(QuantityType.ENERGY),
    "maxDivest_entity_total_path": CstrFamily(QuantityType.POWER),
    "maxDivest_entity_total_path_n": CstrFamily(QuantityType.ENERGY),

    # min-invest / min-divest entity totals — group_capacity resolved
    # via _p / _n suffix in _cumulative_invest.py.  Always named
    # _path (one row per (entity, leaf), required even with a single
    # scenario leaf — see _eq_min_invest_total's EQ. header); the bare
    # (no-_path) names are no longer emitted.
    # _cumulative_invest._eq_min_invest_total (minInvest_entity_total_path_{p,n})
    "minInvest_entity_total_path": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    # _cumulative_invest._eq_min_divest_total (minDivest_entity_total_path_{p,n})
    "minDivest_entity_total_path": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # min-invest / min-divest entity period (prefix-matched _p/_n) —
    # POWER for _p, ENERGY for _n.
    # _cumulative_invest._eq_min_entity_period (min{Invest,Divest}_entity_period_{p,n})
    "minInvest_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minDivest_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # Group invest / divest variants — `[MW or MWh]` resolved at
    # apply time.
    # _cumulative_invest._eq_group_entity_period ({max,min}{Invest,Divest}Group_entity_period_{p,n})
    "maxInvestGroup_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minInvestGroup_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "maxDivestGroup_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minDivestGroup_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    # _cumulative_invest._eq_group_invest_total / _eq_group_divest_total.
    # The min side (both invest and divest) is always named _path
    # (one row per (group, leaf), required even with a single scenario leaf
    # -- same reason as the entity min-total families); the bare
    # (no-_path) min names are no longer emitted.  The max side keeps the
    # bare name for the deterministic (single-leaf) case and only uses
    # _path under recourse, like the entity max-total families.
    "maxInvestGroup_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "maxDivestGroup_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    # Slice D per-path total caps (extended
    # to the invest side and to both min families):
    # ``{max,min}{Invest,Divest}Group_entity_total_path_{p,n}`` emitted by
    # ``_cumulative_invest._eq_group_invest_total`` / ``_eq_group_divest_total``.  The
    # ``_p`` / ``_n`` member-class suffix resolves POWER vs ENERGY off
    # this key (``lookup_cstr`` strips it since the resolver is set).
    "maxInvestGroup_entity_total_path": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minInvestGroup_entity_total_path": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "maxDivestGroup_entity_total_path": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minDivestGroup_entity_total_path": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    # _cumulative_invest._eq_group_invest_cumulative ({max,min}InvestGroup_entity_cumulative).
    # Mixed MW (process members) + MWh (storage-node members) row with no ``_p``/``_n`` suffix to
    # resolve a quantity type from — ``resolve_cstr_rhs_type`` raises on an unsuffixed
    # "group_capacity" name, so Layer 2 must skip this family (plain row scaling only).
    "maxInvestGroup_entity_cumulative": CstrFamily(None),
    "minInvestGroup_entity_cumulative": CstrFamily(None),

    # Cumulative capacity per-entity — `_p` / `_n` suffix.
    # _cumulative_invest._eq_cumulative_capacity ({max,min}Cumulative_capacity_{p,n})
    "maxCumulative_capacity": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minCumulative_capacity": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # Group-level cumulative / period / instant flow — POWER.
    # _cumulative_invest._eq_cumulative_flow_solve / _eq_cumulative_flow_period.  The solve family
    # gets a per-leaf _path row under recourse (leaf_rows), same RHS quantity as the legacy row.
    "maxCumulative_flow_solve": CstrFamily(QuantityType.POWER),
    "minCumulative_flow_solve": CstrFamily(QuantityType.POWER),
    "maxCumulative_flow_solve_path": CstrFamily(QuantityType.POWER),
    "minCumulative_flow_solve_path": CstrFamily(QuantityType.POWER),
    "maxCumulative_flow_period": CstrFamily(QuantityType.POWER),
    "minCumulative_flow_period": CstrFamily(QuantityType.POWER),
    # _cumulative_invest._eq_instant_flow ({max,min}Instant_flow)
    "maxInstant_flow": CstrFamily(QuantityType.POWER),
    "minInstant_flow": CstrFamily(QuantityType.POWER),

    # ── User-defined constraints: composite LHS, RHS in caller's
    # coordinates.  Layer 2 skips per-row scaling.
    # flextool/engine/model/core.py _eq_user_constraints (process_constraint_equal / _less_than / _greater_than;
    # the _period variants are the one-row-per-period rows of constraints without flow or state terms)
    "process_constraint_equal": CstrFamily(None),
    "process_constraint_less_than": CstrFamily(None),
    "process_constraint_greater_than": CstrFamily(None),
    "process_constraint_equal_period": CstrFamily(None),
    "process_constraint_less_than_period": CstrFamily(None),
    "process_constraint_greater_than_period": CstrFamily(None),

    # ── Node-balance fix (used by manual demand patching).  Zero RHS
    # in the most common case; treat as DIMENSIONLESS so the row
    # coefficient is the rescaled quantity.
    # flextool/engine/model/core.py _eq_node_balance_fix_quantity
    "node_balance_fix_quantity_eq_lower": CstrFamily(None),
    # flextool/engine/model/core.py _eq_storage_usage_fix
    "node_storage_usage_fix_le": CstrFamily(None),

    # ── Conversion (efficiency-balanced flow row).
    # flextool/engine/model/core.py _eq_conversion
    "conversion_indirect": CstrFamily(QuantityType.ENERGY),
    # Coarse processes, averaged over each coarse step — flextool/engine/model/_coarse_flows.py add_cstr.
    "conversion_indirect_coarse": CstrFamily(QuantityType.ENERGY),

    # ── CO2 caps — EMISSION_MASS.
    # flextool/engine/model/core.py _eq_co2_max_period
    "co2_max_period": CstrFamily(QuantityType.EMISSION_MASS),
    # flextool/engine/model/core.py _eq_co2_max_total
    "co2_max_total": CstrFamily(QuantityType.EMISSION_MASS),
    # Q16 (#52) per-path row: _eq_co2_max_total passes the constraint name as a variable (legacy vs
    # "_path"), so the literal-grep test cannot see it.  Registered EXACTLY, same reason as the
    # entity/group total-cap _path families above.
    "co2_max_total_path": CstrFamily(QuantityType.EMISSION_MASS),

    # ── Unit-commitment (online / startup / shutdown / minimum times).
    # All DIMENSIONLESS — they bound counts.
    # flextool/engine/model/_online.py (maxOnline_<sfx>)
    "maxOnline": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py (maxStartup_<sfx>)
    "maxStartup": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py (maxShutdown_<sfx>)
    "maxShutdown": CstrFamily(QuantityType.DIMENSIONLESS),
    # c14c part B — flextool/engine/model/_online.py (online_change_<sfx>): the equality that replaced
    # online__startup_<sfx> / online__shutdown_<sfx> (two >= rows).
    "online_change": CstrFamily(QuantityType.DIMENSIONLESS),
    # OC1 #1 sub-slice (b) — separate_dispatch sibling v_online PIN
    # (_online.py online_fix_eq_<linear|integer>; the resolver strips the UC
    # suffix to this base).  Pins the dimensionless unit-count v_online to
    # the parent's committed schedule.
    "online_fix_eq": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_reserve.py _add_reserve_fix — realization-solve
    # v_reserve PIN (unit-count, like v_reserve).
    "reserve_fix_eq": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py (maxFlow_online_<sfx>)
    "maxFlow_online": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py
    # (maxOutputSum_online_<sfx>)
    "maxOutputSum_online": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py (maxInputFlow_online_<sfx>: an indirect
    # unit's input arc carrying downward reserve at its source, unit-count flow)
    "maxInputFlow_online": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py (maxFlow_online_shutdown_<sfx> /
    # maxInputFlow_online_shutdown_<sfx>: the pre-shutdown twin of the online
    # capacity rows with the startup / shutdown capacity reduction).  Registered
    # explicitly: the "maxFlow_online" prefix would otherwise resolve them silently.
    "maxFlow_online_shutdown": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxInputFlow_online_shutdown": CstrFamily(QuantityType.DIMENSIONLESS),
    # c14c critique item 1 — flextool/engine/model/_online.py (maxFlow_online_shutdown_branch_<sfx> /
    # maxInputFlow_online_shutdown_branch_<sfx>): the pre-shutdown twin's per-successor rows at a branch point
    # (several mid-horizon branches continuing from the same trunk step).  Registered explicitly: the
    # "maxFlow_online_shutdown" / "maxInputFlow_online_shutdown" prefixes would otherwise resolve them silently.
    "maxFlow_online_shutdown_branch": CstrFamily(QuantityType.DIMENSIONLESS),
    "maxInputFlow_online_shutdown_branch": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py (maxFlow_back_online_<sfx>: the reverse flow of an online one-variable
    # connection ≤ online capacity).  Explicit: the "maxFlow_back" prefix would otherwise resolve it silently.
    "maxFlow_back_online": CstrFamily(QuantityType.DIMENSIONLESS),
    # _unit_floors.py (minFlow_minload_<sfx>: per-unit floor)
    "minFlow_minload": CstrFamily(QuantityType.DIMENSIONLESS),
    # _unit_floors.py (minFlow_output_floor_<sfx>: per-output
    # floor of a multi-output unit; unit-count flow units)
    "minFlow_output_floor": CstrFamily(QuantityType.DIMENSIONLESS),
    # _unit_floors.py: downward-reserve headroom rows (zero
    # floor) per unit / per arc; unit-count flow units
    "minFlow_reserve": CstrFamily(QuantityType.DIMENSIONLESS),
    "minFlow_reserve_arc": CstrFamily(QuantityType.DIMENSIONLESS),
    # model/core.py _eq_min_input_share: homogeneous input-mixing row
    # (input_share_min) in unit-count flow units
    "minInputShare": CstrFamily(QuantityType.DIMENSIONLESS),
    # Coarse processes, averaged over each coarse step — flextool/engine/model/_coarse_flows.py add_cstr.
    "minInputShare_coarse": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py add_constraints (minimum_uptime_<sfx>)
    "minimum_uptime": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine/model/_online.py add_constraints (minimum_downtime_<sfx>)
    "minimum_downtime": CstrFamily(QuantityType.DIMENSIONLESS),

    # ── DC PF.
    # flextool/engine/model/_dc_power_flow.py _eq_dc_flow — RHS=0, skip.
    "dc_flow_eq": CstrFamily(None),
    # flextool/engine/model/_dc_power_flow.py _eq_reference_angle — fixed-zero angle
    # at the reference node; RHS=0, skip per-row scaling.
    "dc_reference_angle_eq": CstrFamily(None),

    # ── Reserve.
    # flextool/engine/model/_reserve.py add_constraints
    "reserveBalance_timeseries_eq": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/_reserve.py add_constraints
    "reserveBalance_dynamic_eq": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/_reserve.py add_constraints (reserveBalance_up_n_1_eq / _down_n_1_eq)
    "reserveBalance_up_n_1_eq": CstrFamily(QuantityType.POWER),
    "reserveBalance_down_n_1_eq": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/_reserve.py add_constraints.  RHS 0, but typed POWER so its
    # v_reserve · unitsize entries bucket as POWER like in the n-1 rows.
    "reserve_total_upper": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/_reserve.py add_constraints (reserve_process_upward / _downward)
    "reserve_process_upward": CstrFamily(QuantityType.POWER),
    "reserve_process_downward": CstrFamily(QuantityType.POWER),
    # Connection reserve by its net position, in units of unitsize like maxFlow
    # (flextool/engine/model/_reserve.py _add_connection_reserve_constraints).  Registered explicitly (the
    # suffixed online / exact names too): no prefix of another family may resolve them silently.
    "connection_reserve_up": CstrFamily(QuantityType.DIMENSIONLESS),
    "connection_reserve_down": CstrFamily(QuantityType.DIMENSIONLESS),
    "connection_reserve_up_online": CstrFamily(QuantityType.DIMENSIONLESS),
    "connection_reserve_down_online": CstrFamily(QuantityType.DIMENSIONLESS),
    "connection_reserve_up_exact": CstrFamily(QuantityType.DIMENSIONLESS),
    "connection_reserve_down_exact": CstrFamily(QuantityType.DIMENSIONLESS),
    # Storage energy-adequacy coupling for reserve_duration (#322).  Both
    # framed in MWh: v_state·unitsize (± E) vs a stored-energy bound.
    # flextool/engine/model/_reserve.py _add_storage_energy_constraints
    "reserve_storage_up_floor": CstrFamily(QuantityType.ENERGY),
    "reserve_storage_down_headroom": CstrFamily(QuantityType.ENERGY),

    # ── Group slack constraints.
    # flextool/engine/model/_group_slack.py _eq_capacity_margin
    "capacityMargin": CstrFamily(QuantityType.POWER),
    # flextool/engine/model/_group_slack.py _eq_inertia
    "inertia_constraint": CstrFamily(QuantityType.INERTIA),
    # flextool/engine/model/_group_slack.py _eq_non_sync
    "non_sync_constraint": CstrFamily(QuantityType.POWER),

    # ── Commodity ladder.  ENERGY (MWh).
    # flextool/engine/model/_commodity_ladder.py _eq_ladder_balance
    "commodity_ladder_balance": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/_commodity_ladder.py _eq_tier_cap_annual
    "ladder_tier_cap_annual_roll": CstrFamily(QuantityType.ENERGY),
    # flextool/engine/model/_commodity_ladder.py _eq_tier_cap_cumulative
    "ladder_tier_cap_cumulative_roll": CstrFamily(QuantityType.ENERGY),
    # Q16 (#58) per-path row: _eq_tier_cap_cumulative passes the constraint name as a variable
    # (legacy vs "_path"), so the literal-grep test cannot see it.  Registered EXACTLY, same reason
    # as the entity/group total-cap _path families.
    "ladder_tier_cap_cumulative_roll_path": CstrFamily(QuantityType.ENERGY),

    # ── Forbid-invest rows.  RHS = 0 by construction; skip per-row.
    # _cumulative_invest._eq_fix_no_investment
    "fix_v_invest_no_investment_eq_p": CstrFamily(None),
    "fix_v_invest_no_investment_eq_n": CstrFamily(None),

    # ── Non-anticipativity (cross-branch equality).  RHS = 0; skip.
    # flextool/engine/model/_non_anticipativity.py:add_dispatch_constraints
    "non_anticipativity_storage_use": CstrFamily(None),
    "non_anticipativity_storage_state_first": CstrFamily(None),
    "non_anticipativity_online_integer": CstrFamily(None),
    "non_anticipativity_online_linear": CstrFamily(None),
    "non_anticipativity_reserve": CstrFamily(None),

    # ── Investment non-anticipativity (cross-branch invest/divest
    # equality). RHS = 0; skip per-row.
    # flextool/engine/model/_non_anticipativity.py:add_invest_constraints
    "non_anticipativity_invest_p": CstrFamily(None),
    "non_anticipativity_invest_n": CstrFamily(None),
    "non_anticipativity_divest_p": CstrFamily(None),
    "non_anticipativity_divest_n": CstrFamily(None),

    # ── Profile constraints (per timestep).
    # Node profiles emit in model._eq_state_profiles with names
    # ``profile_state_upper_limit`` / ``_lower_limit`` / ``_fixed``;
    # process profiles in model._eq_profiles with ``profile_flow_*``.  Per
    # the autoscaler handoff, profile state is POWER for unit /
    # connection, ENERGY for node.  Node-side rows go through one
    # call site (``_eq_state_profiles``) so we register the names
    # here without a member_class_resolver — they are exclusively
    # ENERGY-typed.  Process-side ``profile_flow_*`` rows are POWER.
    "profile_state_upper_limit": CstrFamily(QuantityType.ENERGY),
    "profile_state_lower_limit": CstrFamily(QuantityType.ENERGY),
    "profile_state_fixed": CstrFamily(QuantityType.ENERGY),
    "profile_flow_upper_limit": CstrFamily(QuantityType.POWER),
    "profile_flow_lower_limit": CstrFamily(QuantityType.POWER),
    "profile_flow_fixed": CstrFamily(QuantityType.POWER),
    # Coarse processes, averaged over each coarse step — flextool/engine/model/_coarse_flows.py add_cstr.
    "profile_flow_upper_limit_coarse": CstrFamily(QuantityType.POWER),
    "profile_flow_lower_limit_coarse": CstrFamily(QuantityType.POWER),
    "profile_flow_fixed_coarse": CstrFamily(QuantityType.POWER),

    # ── Benders (Option C) hand-built master constraints.  The hand master
    # is built AUTOSCALE-OFF, so Layer 2 never scales these rows; registering
    # them keeps the registry-coverage invariant complete (CLAUDE.md #1).
    # ``maxInvest``: C[conn] <= max_units — capacity bound, mirrors the
    # entity-scope ``maxInvest_var_bound`` (POWER RHS).
    # flextool/engine/solve/_benders.py:520
    "maxInvest": CstrFamily(QuantityType.POWER),
    # ``trade_capacity``: C[conn] - f[arc,d,t] >= 0 — the RHS is identically
    # zero (no per-row scaling), so CstrFamily(None) like the other zero-RHS
    # structural rows; the column scalers on C / f still propagate.
    # flextool/engine/solve/_benders.py:504
    "trade_capacity": CstrFamily(None),
}


def lookup_var(name: str) -> VarFamily:
    """Return the :class:`VarFamily` registered for ``name``.

    Raises :class:`KeyError` for any unregistered variable; Layer 2
    must refuse to silently treat unknown variables as DIMENSIONLESS,
    which would hide drift between this registry and the LP build.
    """
    return VARIABLE_FAMILIES[name]


def lookup_cstr(name: str) -> CstrFamily:
    """Return the :class:`CstrFamily` registered for ``name``.

    Exact match wins.  Otherwise we strip a trailing ``_linear``,
    ``_integer``, ``_p``, ``_n`` suffix and retry (covers
    ``maxOnline_linear`` → ``maxOnline``, ``maxInvest_entity_total_n``
    is already exact, ``minInvest_entity_total_p`` →
    ``minInvest_entity_total``).  Suffix stripping is conservative:
    only the known UC and member-class suffixes are removed.

    Raises :class:`KeyError` if no registration matches.
    """
    if name in CONSTRAINT_FAMILIES:
        return CONSTRAINT_FAMILIES[name]
    # Strip UC suffixes first.
    for sfx in ("_linear", "_integer"):
        if name.endswith(sfx):
            base = name[: -len(sfx)]
            if base in CONSTRAINT_FAMILIES:
                return CONSTRAINT_FAMILIES[base]
    # Then member-class suffix (_p / _n).  This is the prefix that
    # member_class_resolver entries cover.  We only strip when the
    # *unsuffixed* name matches an entry whose resolver is set, so a
    # row that happens to end with ``_p`` but is registered with an
    # explicit suffix (e.g. ``maxInvest_entity_period_p``) stays on
    # the exact-match path.
    for sfx in ("_p", "_n"):
        if name.endswith(sfx):
            base = name[: -len(sfx)]
            if (base in CONSTRAINT_FAMILIES
                    and CONSTRAINT_FAMILIES[base].member_class_resolver):
                return CONSTRAINT_FAMILIES[base]
    # Prefix dispatch (the contract documented in this module's header and
    # in test_registry_coverage): a registry key ``K`` matches a dynamic
    # constraint name when ``name.startswith(K + "_")``.  This is what the
    # ``"ramp"`` entry relies on — the ramp family
    # (``ramp_{side}_{dir}_constraint``, model._eq_ramp) carries middle and
    # suffix tokens (``_sink_up_constraint``) that neither the exact-match
    # nor the UC / member-class suffix-strip paths above can resolve.
    # Longest matching key wins so a more specific registration is never
    # shadowed by a shorter prefix.  Runs last, after exact + suffix-strip,
    # so it can only resolve names those paths leave unresolved — never
    # change an existing resolution.
    prefix_match: Optional[str] = None
    for key in CONSTRAINT_FAMILIES:
        if name.startswith(key + "_") and (
                prefix_match is None or len(key) > len(prefix_match)):
            prefix_match = key
    if prefix_match is not None:
        return CONSTRAINT_FAMILIES[prefix_match]
    raise KeyError(name)


def resolve_cstr_rhs_type(name: str) -> Optional[QuantityType]:
    """Final per-name rhs_type after suffix resolution.

    For ``member_class_resolver="group_capacity"`` families, the
    constraint name's ``_p`` / ``_n`` suffix selects POWER vs ENERGY
    (via :func:`resolve_group_capacity_type`).  Returns ``None`` for
    families that opt out of per-row scaling.
    """
    fam = lookup_cstr(name)
    if fam.member_class_resolver == "group_capacity":
        if name.endswith("_p"):
            return QuantityType.POWER
        if name.endswith("_n"):
            return QuantityType.ENERGY
        # Unsuffixed (shouldn't happen for group-capacity registered
        # entries) — fall through.  Surface as KeyError so we don't
        # silently misclassify.
        raise KeyError(
            f"{name!r}: group_capacity resolver requires _p / _n suffix"
        )
    return fam.rhs_type


__all__ = [
    "VarFamily",
    "CstrFamily",
    "VARIABLE_FAMILIES",
    "CONSTRAINT_FAMILIES",
    "lookup_var",
    "lookup_cstr",
    "resolve_cstr_rhs_type",
]
