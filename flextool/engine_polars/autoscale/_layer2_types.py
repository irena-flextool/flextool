"""Layer 2 (semantic per-type scaling) family registries.

This module declares, for every variable and constraint family FlexTool
emits, the :class:`QuantityType` that determines its Layer-2 scaling
group.

The registry is populated by walking every ``add_var(...)`` and
``add_cstr(...)`` call site in ``flextool/engine_polars/`` (see comments
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
    # flextool/engine_polars/model.py:482
    "v_flow": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),

    # ── Storage / nodeBalance slacks.  Carry ENERGY directly (no
    # unitsize multiplier on the column).
    # flextool/engine_polars/model.py:484
    "vq_state_up": VarFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:485
    "vq_state_down": VarFamily(QuantityType.ENERGY),

    # ── Storage state (per-step, per-block).  ENERGY column.
    # flextool/engine_polars/model.py:490
    "v_state": VarFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:502
    "v_state_inter": VarFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:505
    "v_state_rp_start": VarFamily(QuantityType.ENERGY),

    # ── Unit-commitment families.  Column counts a number of units
    # online / starting / shutting; DIMENSIONLESS.  The matrix coef
    # ``p_unitsize`` (or ``v_unitsize`` from process_existing_count)
    # turns the row into POWER as needed.  Integer variants are
    # mathematically the same column type.
    # flextool/engine_polars/model.py:511
    "v_online_linear": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:512
    "v_startup_linear": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:513
    "v_shutdown_linear": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:518
    "v_online_integer": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:520
    "v_startup_integer": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:521
    "v_shutdown_integer": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),

    # ── Investment / divestment counts.  DIMENSIONLESS column whose
    # matrix coefficient ``p_unitsize`` (process / connection) or
    # ``p_state_unitsize`` (node) sets the row's effective POWER /
    # ENERGY type.
    # flextool/engine_polars/model.py:523
    "v_invest_p": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:525
    "v_divest_p": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/model.py:527
    "v_invest_n": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_state_unitsize"),
    # flextool/engine_polars/model.py:529
    "v_divest_n": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_state_unitsize"),

    # ── Reserve.  DIMENSIONLESS column; ``p_unitsize`` multiplier
    # carries the per-row POWER coefficient.
    # flextool/engine_polars/_reserve.py:275
    "v_reserve": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
    # flextool/engine_polars/_reserve.py:282
    "vq_reserve": VarFamily(QuantityType.POWER),

    # ── Commodity ladder.  v_trade is DIMENSIONLESS; row coef
    # ``unitsize`` turns it into ENERGY for the ladder caps and the
    # balance row.
    # flextool/engine_polars/_commodity_ladder.py:399
    "v_trade": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="unitsize"),

    # ── Group-level slacks.  Direct physical units.
    # flextool/engine_polars/_group_slack.py:522
    "vq_capacity_margin": VarFamily(QuantityType.POWER),
    # flextool/engine_polars/_group_slack.py:902
    "vq_inertia": VarFamily(QuantityType.INERTIA),
    # flextool/engine_polars/_group_slack.py:1045
    "vq_non_synchronous": VarFamily(QuantityType.POWER),

    # ── DC power flow.
    # ``v_angle`` (radians).  Dimensionless in our taxonomy.
    # flextool/engine_polars/_dc_power_flow.py:250
    "v_angle": VarFamily(QuantityType.DIMENSIONLESS),
    # ``v_flow_back`` mirrors v_flow on the reverse direction.
    # flextool/engine_polars/_dc_power_flow.py:296
    "v_flow_back": VarFamily(QuantityType.DIMENSIONLESS, multiplier_param="p_unitsize"),
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
    # flextool/engine_polars/model.py:1336
    "nodeBalance_eq": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:1792
    "nodeBalanceBlock_eq": CstrFamily(QuantityType.ENERGY),

    # ── Storage state max / state ladder.  ENERGY.
    # flextool/engine_polars/model.py:2788
    "maxState": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:1159
    "maxState_rp_start": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:1570
    "stateConstantWithinBlock_eq": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:905
    "rp_inter_period_balance": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:1039
    "rp_inter_period_cyclic": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:1144
    "rp_inter_period_max_state": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:2985
    "storage_state_start_binding": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:3072
    "storage_state_start_binding_cyclic_period": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:3124
    "storage_state_solve_horizon_reference_value": CstrFamily(QuantityType.ENERGY),

    # ── Flow caps from process side.  DIMENSIONLESS (capacity coefficient).
    # The cap is a fraction × existing_count; the row's effective type
    # lives in the coefficient, not the RHS.
    # flextool/engine_polars/model.py:2045
    "maxToSink": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:2075
    "maxToSink_negCap": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/_dc_power_flow.py:414
    "maxToSink_back": CstrFamily(QuantityType.DIMENSIONLESS),
    # Ramp constraints, prefix-matched: ramp_<side>_<dir>_constraint.
    # flextool/engine_polars/model.py:2164
    "ramp": CstrFamily(QuantityType.DIMENSIONLESS),

    # ── Investment caps — entity scope.
    # flextool/engine_polars/model.py:2189
    "maxInvest_var_bound": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/model.py:2197
    "maxDivest_var_bound": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/model.py:2217
    "maxInvest_var_bound_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:2225
    "maxDivest_var_bound_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:2257
    "maxInvest_entity_period_p": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/model.py:2274
    "maxInvest_entity_period_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:2292
    "maxDivest_entity_period_p": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/model.py:2309
    "maxDivest_entity_period_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:2380
    "maxInvest_entity_total": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/model.py:2432
    "maxInvest_entity_total_n": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/model.py:2405
    "maxDivest_entity_total": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/model.py:2457
    "maxDivest_entity_total_n": CstrFamily(QuantityType.ENERGY),

    # min-invest / min-divest entity totals — group_capacity resolved
    # via _p / _n suffix in _cumulative_invest.py.
    # flextool/engine_polars/_cumulative_invest.py:456 (minInvest_entity_total_p)
    # flextool/engine_polars/_cumulative_invest.py:496 (minInvest_entity_total_n)
    "minInvest_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    # flextool/engine_polars/_cumulative_invest.py:464 (minDivest_entity_total_p)
    # flextool/engine_polars/_cumulative_invest.py:504 (minDivest_entity_total_n)
    "minDivest_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # min-invest / min-divest entity period (prefix-matched _p/_n) —
    # POWER for _p, ENERGY for _n.
    # flextool/engine_polars/_cumulative_invest.py:381 (minInvest_entity_period_p)
    # flextool/engine_polars/_cumulative_invest.py:395 (minInvest_entity_period_n)
    "minInvest_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minDivest_entity_period": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # Group invest / divest variants — `[MW or MWh]` resolved at
    # apply time.
    # flextool/engine_polars/_cumulative_invest.py:681 (maxInvestGroup_entity_period_p)
    # flextool/engine_polars/_cumulative_invest.py:701 (maxInvestGroup_entity_period_n)
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
    # flextool/engine_polars/_cumulative_invest.py:786 (maxInvestGroup_entity_total_p)
    # flextool/engine_polars/_cumulative_invest.py:819 (maxInvestGroup_entity_total_n)
    "maxInvestGroup_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minInvestGroup_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "maxDivestGroup_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minDivestGroup_entity_total": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    # flextool/engine_polars/_cumulative_invest.py:845 (maxInvestGroup_entity_cumulative_p)
    # flextool/engine_polars/_cumulative_invest.py:869 (maxInvestGroup_entity_cumulative_n)
    "maxInvestGroup_entity_cumulative": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minInvestGroup_entity_cumulative": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # Cumulative capacity per-entity — `_p` / `_n` suffix.
    # flextool/engine_polars/_cumulative_invest.py:588 (maxCumulative_capacity_p)
    # flextool/engine_polars/_cumulative_invest.py:632 (maxCumulative_capacity_n)
    "maxCumulative_capacity": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),
    "minCumulative_capacity": CstrFamily(
        None, member_class_resolver="group_capacity"
    ),

    # Group-level cumulative / period / instant flow — POWER.
    # flextool/engine_polars/_cumulative_invest.py:1050 (maxCumulative_flow_solve)
    # flextool/engine_polars/_cumulative_invest.py:1085 (maxCumulative_flow_period)
    "maxCumulative_flow_solve": CstrFamily(QuantityType.POWER),
    "minCumulative_flow_solve": CstrFamily(QuantityType.POWER),
    "maxCumulative_flow_period": CstrFamily(QuantityType.POWER),
    "minCumulative_flow_period": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/_cumulative_invest.py:1111 (maxInstant_flow / minInstant_flow)
    "maxInstant_flow": CstrFamily(QuantityType.POWER),
    "minInstant_flow": CstrFamily(QuantityType.POWER),

    # ── User-defined constraints: composite LHS, RHS in caller's
    # coordinates.  Layer 2 skips per-row scaling.
    # flextool/engine_polars/model.py:2740 (process_constraint_equal / _less_than / _greater_than)
    "process_constraint_equal": CstrFamily(None),
    "process_constraint_less_than": CstrFamily(None),
    "process_constraint_greater_than": CstrFamily(None),

    # ── Node-balance fix (used by manual demand patching).  Zero RHS
    # in the most common case; treat as DIMENSIONLESS so the row
    # coefficient is the rescaled quantity.
    # flextool/engine_polars/model.py:1392
    "node_balance_fix_quantity_eq_lower": CstrFamily(None),
    # flextool/engine_polars/model.py:1539
    "node_storage_usage_fix_le": CstrFamily(None),

    # ── Conversion (efficiency-balanced flow row).
    # flextool/engine_polars/model.py:2509
    "conversion_indirect": CstrFamily(QuantityType.ENERGY),

    # ── CO2 caps — EMISSION_MASS.
    # flextool/engine_polars/model.py:2542
    "co2_max_period": CstrFamily(QuantityType.EMISSION_MASS),
    # flextool/engine_polars/model.py:2577
    "co2_max_total": CstrFamily(QuantityType.EMISSION_MASS),

    # ── Unit-commitment (online / startup / shutdown / minimum times).
    # All DIMENSIONLESS — they bound counts.
    # flextool/engine_polars/model.py:3732 (maxOnline_<sfx>)
    "maxOnline": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3735 (maxStartup_<sfx>)
    "maxStartup": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3738 (maxShutdown_<sfx>)
    "maxShutdown": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3744 (online__startup_<sfx>)
    "online__startup": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3748 (online__shutdown_<sfx>)
    "online__shutdown": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3770 (maxToSink_online_<sfx>)
    "maxToSink_online": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3785 (minToSink_minload_<sfx>)
    "minToSink_minload": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3814 (minimum_uptime_<sfx>)
    "minimum_uptime": CstrFamily(QuantityType.DIMENSIONLESS),
    # flextool/engine_polars/model.py:3853 (minimum_downtime_<sfx>)
    "minimum_downtime": CstrFamily(QuantityType.DIMENSIONLESS),

    # ── DC PF.
    # flextool/engine_polars/_dc_power_flow.py:397 — RHS=0, skip.
    "dc_flow_eq": CstrFamily(None),
    # flextool/engine_polars/_dc_power_flow.py:340 — fixed-zero angle
    # at the reference node; RHS=0, skip per-row scaling.
    "dc_reference_angle_eq": CstrFamily(None),

    # ── Reserve.
    # flextool/engine_polars/_reserve.py:373
    "reserveBalance_timeseries_eq": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/_reserve.py:449
    "reserveBalance_dynamic_eq": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/_reserve.py:508 (reserveBalance_up_n_1_eq / _down_n_1_eq)
    "reserveBalance_up_n_1_eq": CstrFamily(QuantityType.POWER),
    "reserveBalance_down_n_1_eq": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/_reserve.py:576 (reserve_process_upward / _downward)
    "reserve_process_upward": CstrFamily(QuantityType.POWER),
    "reserve_process_downward": CstrFamily(QuantityType.POWER),

    # ── Group slack constraints.
    # flextool/engine_polars/_group_slack.py:859
    "capacityMargin": CstrFamily(QuantityType.POWER),
    # flextool/engine_polars/_group_slack.py:984
    "inertia_constraint": CstrFamily(QuantityType.INERTIA),
    # flextool/engine_polars/_group_slack.py:1157
    "non_sync_constraint": CstrFamily(QuantityType.POWER),

    # ── Commodity ladder.  ENERGY (MWh).
    # flextool/engine_polars/_commodity_ladder.py:521
    "commodity_ladder_balance": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/_commodity_ladder.py:561  ← the H2_trade trigger
    "ladder_tier_cap_annual_roll": CstrFamily(QuantityType.ENERGY),
    # flextool/engine_polars/_commodity_ladder.py:607
    "ladder_tier_cap_cumulative_roll": CstrFamily(QuantityType.ENERGY),

    # ── Forbid-invest rows.  RHS = 0 by construction; skip per-row.
    # flextool/engine_polars/_cumulative_invest.py:284, 298
    "fix_v_invest_no_investment_eq_p": CstrFamily(None),
    "fix_v_invest_no_investment_eq_n": CstrFamily(None),

    # ── Non-anticipativity (cross-branch equality).  RHS = 0; skip.
    # flextool/engine_polars/model.py:273, 306, 336, 364
    "non_anticipativity_storage_use": CstrFamily(None),
    "non_anticipativity_online_integer": CstrFamily(None),
    "non_anticipativity_online_linear": CstrFamily(None),
    "non_anticipativity_reserve": CstrFamily(None),

    # ── Profile constraints (per timestep).
    # Node profiles emit at model.py:2879 with names
    # ``profile_state_upper_limit`` / ``_lower_limit`` / ``_fixed``;
    # process profiles at model.py:3914 with ``profile_flow_*``.  Per
    # the autoscaler handoff, profile state is POWER for unit /
    # connection, ENERGY for node.  Node-side rows go through one
    # call site (``_add_node_profile_cstr``) so we register the names
    # here without a member_class_resolver — they are exclusively
    # ENERGY-typed.  Process-side ``profile_flow_*`` rows are POWER.
    "profile_state_upper_limit": CstrFamily(QuantityType.ENERGY),
    "profile_state_lower_limit": CstrFamily(QuantityType.ENERGY),
    "profile_state_fixed": CstrFamily(QuantityType.ENERGY),
    "profile_flow_upper_limit": CstrFamily(QuantityType.POWER),
    "profile_flow_lower_limit": CstrFamily(QuantityType.POWER),
    "profile_flow_fixed": CstrFamily(QuantityType.POWER),
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
    # (``ramp_{side}_{dir}_constraint``, model.py:2328) carries middle and
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
