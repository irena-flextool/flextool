"""FlexTool LP model: build_flextool and the core equation families."""

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from polar_high import Sum, Where, Lag, Param
from polar_high.engine import Expr, Var

from . import _commodity_ladder
from . import _cumulative_invest
from . import _dc_power_flow
from . import _delay
from . import _group_slack
from . import _online
from . import _reserve
from . import _unit_floors
from ._axis_enums import alias_to_axis, cast_dim, rename_to_axis
from ._model_common import (
    _availability_factor,
    _coef_factor,
    _drop_uncapped,
    _entity_rows,
    _has_rows,
    _indirect_sum_arcs,
    _utf8_keys,
    _var_view,
)
from ._param_shapes import promote_param_to_dt
from ._pdt_join import (
    compute_pss_dt,
    compute_nodeBalance_dt,
    compute_nodeState_dt,
    compute_nodeState_rp_block_first_dt,
    compute_process_indirect_dt,
)
from ._solve_state import FlexToolConfigError

_LOG = logging.getLogger(__name__)


def _build_prof(label: str) -> None:
    """Env-gated (FLEXTOOL_BUILD_PROFILE=1) RSS print to stderr. No-op otherwise."""
    import os
    import sys
    if os.environ.get("FLEXTOOL_BUILD_PROFILE") != "1":
        return
    try:
        with open("/proc/self/status") as _f:
            for _ln in _f:
                if _ln.startswith("VmRSS:"):
                    sys.stderr.write(f"[build profile] step={label}\trss_gb={int(_ln.split()[1])/(1024*1024):.3f}\n")
                    sys.stderr.flush()
                    break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data fields each active feature requires (checked by _check in _prepare).
# ``ALWAYS``: fields needed even when only the balance slacks exist.  The CO2
# price has no tuple: without its data the cost term is skipped with a warning.
ALWAYS: tuple[str, ...] = (
    "dt", "nodeBalance",
    "p_step_duration", "p_timestep_weight", "p_inflation_op", "p_period_share",
    "p_inflow", "p_penalty_up", "p_penalty_down",
)
PROCESSES: tuple[str, ...] = (
    "process_source_sink", "process_source_sink_eff", "process_source_sink_noEff",
    "flow_to_n", "flow_from_commodity_eff", "flow_from_commodity_noEff",
    "p_unitsize", "p_flow_upper", "p_slope", "p_commodity_price",
)
INDIRECT: tuple[str, ...] = (
    "process_indirect", "process_input_flows", "process_output_flows",
)
CO2_CAP:   tuple[str, ...]   = ("p_co2_content", "p_co2_max_period",
                                "group_d_co2_capped")
CO2_CAP_TOTAL: tuple[str, ...] = ("p_co2_content", "p_co2_max_total",
                                   "group_co2_max_total")
USER_CSTR: tuple[str, ...]   = ("p_constraint_constant",)
PROFILES:  tuple[str, ...]   = ("p_profile_value", "p_process_existing_count")
STORAGE:   tuple[str, ...]   = ("nodeState", "dtttdt",
                                "p_state_upper", "p_state_unitsize")
ONLINE: tuple[str, ...] = (
    "process_online", "p_online_dt", "p_min_load",
    "dtttdt", "p_process_existing_count",
)
MINLOAD_EFF: tuple[str, ...] = ("process_min_load_eff", "p_section")
STARTUP_COST_LINEAR: tuple[str, ...] = ("p_startup_cost", "pdt_online_linear")
STARTUP_COST_INTEGER: tuple[str, ...] = ("p_startup_cost", "pdt_online_integer")
RAMP: tuple[str, ...] = ("dtttdt", "p_step_duration", "p_process_existing_count")
INVEST: tuple[str, ...] = ("p_entity_max_units",)
DIVEST: tuple[str, ...] = ("p_entity_max_units", "edd_divest_active",
                            "pd_divest_set")


def _check(d, fields: tuple[str, ...], feature: str) -> None:
    """Raise if any required field is ``None`` on ``d``.

    Empty polars frames are *not* an error — many partitions can be
    legitimately empty (e.g. ``process_source_sink_noEff`` when every
    process is in the eff partition).  ``None`` means the loader didn't
    populate the field at all, which is what we want to catch."""
    missing = [f for f in fields if getattr(d, f, None) is None]
    if missing:
        raise ValueError(
            f"build_flextool: feature {feature!r} is active but data fields "
            f"are not populated (None): {missing}.  Either fill them in the "
            f"data or don't enable the feature."
        )


# ---------------------------------------------------------------------------
# Build context


@dataclass(slots=True, eq=False)
class _Build:
    """State shared by the build steps of :func:`build_flextool`.

    Holds only what more than one step reads; everything else stays local
    to its family function.  Fields are filled in step order: flags and
    index sets by ``_prepare``, variables by ``_declare_variables``, shared
    expressions by ``_shared_expressions``.  The objective is not held here:
    the ``_obj_*`` and ``_feature_*`` steps return their terms to
    :func:`build_flextool`.
    """

    m: Any
    d: Any
    include_existing_fixed_cost: bool = False

    # ── feature flags ──
    has_proc: bool = False
    has_indirect: bool = False
    has_co2_price: bool = False
    has_co2_cap_eff: bool = False
    has_co2_cap_noEff: bool = False
    has_co2_cap: bool = False
    has_co2_cap_total_eff: bool = False
    has_co2_cap_total_noEff: bool = False
    has_co2_cap_total: bool = False
    has_user_cstr: bool = False
    has_profile: bool = False
    has_storage: bool = False
    has_rp: bool = False                 # representative-period (blended-weights) storage
    has_nsb: bool = False                # nodeStateBlock: storage balanced per coarse block
    has_online_lin: bool = False
    has_online_int: bool = False
    has_online: bool = False
    has_minload_eff: bool = False
    has_startup_cost_lin: bool = False
    has_startup_cost_int: bool = False
    has_ramp: bool = False
    has_invest_p: bool = False
    has_divest_p: bool = False
    has_invest_n: bool = False
    has_divest_n: bool = False
    has_ladder: bool = False
    is_solve_first: bool = True          # first solve of a chain, or a single solve

    # ── shared index sets ──
    pss_dt: pl.DataFrame | None = None               # (p, source, sink, d, t) of v_flow
    nodeBalance_dt: pl.DataFrame | None = None       # (n, d, t) of nodeBalance_eq
    nodeState_dt: pl.DataFrame | None = None         # (n, d, t) of v_state
    process_indirect_dt: pl.DataFrame | None = None  # (p, d, t) of conversion_indirect
    rp_start_idx: pl.DataFrame | None = None         # (n, d, t) of v_state_rp_start
    p_olin_idx: pl.DataFrame | None = None           # (p, d, t) of the linear online vars
    p_oint_idx: pl.DataFrame | None = None           # (p, d, t) of the integer online vars
    twoway1var_arcs: pl.DataFrame | None = None      # (p, source, sink) of v_flow_back
    # Per-arc capacity (see _max_flow_rhs): bound of v_flow, RHS of maxFlow.
    invest_procs: pl.DataFrame | None = None         # processes with invest/divest vars
    flow_upper_rhs: Param | None = None
    flow_cap_mult: pl.DataFrame | None = None
    flow_var_upper: Param | None = None
    flow_bound_arcs: pl.DataFrame | None = None

    # ── variables ──
    v_flow: Var | None = None
    vq_state_up: Var | None = None
    vq_state_down: Var | None = None
    v_state: Var | None = None
    v_state_inter: Var | None = None
    v_state_rp_start: Var | None = None
    v_online_linear: Var | None = None
    v_startup_linear: Var | None = None
    v_shutdown_linear: Var | None = None
    v_online_integer: Var | None = None
    v_startup_integer: Var | None = None
    v_shutdown_integer: Var | None = None
    v_invest_p: Var | None = None
    v_divest_p: Var | None = None
    v_invest_n: Var | None = None
    v_divest_n: Var | None = None
    v_flow_back: Var | None = None
    reserve_vars: dict = field(default_factory=dict)
    dc_pf_vars: dict = field(default_factory=dict)
    ladder_vars: dict = field(default_factory=dict)

    # ── shared expressions ──
    node_balance_flow_terms: dict = field(default_factory=dict)
    reserve_up_to_sink: Any = None          # Σ_r v_reserve up,   (p, sink, d, t)
    reserve_down_to_sink: Any = None        # Σ_r v_reserve down, (p, sink, d, t)
    reserve_up_sink_parts: Any = None       # (v_reserve at sink, up selector)
    invest_in_dispatch_p: Any = None        # Σ v_invest_p alive in d, (p, d)
    divest_in_dispatch_p: Any = None        # Σ v_divest_p active in d, (p, d)
    op_factor: Any = None                   # weight of operational cost terms, (d, t)


def build_flextool(m, d, *, include_existing_fixed_cost: bool = False,
                   scale_the_objective: float = 1.0) -> None:
    """Build the FlexTool LP into ``m`` from the data ``d``.

    Every block is conditional on its data: a feature with an empty index
    set emits nothing; a feature whose index set is populated but whose
    required fields are ``None`` raises ``ValueError`` (requirement tuples
    above).

    ``include_existing_fixed_cost``: add the fixed cost of existing
    capacity, Σ existing · fixed_cost · inflation_op, as an objective
    constant.  Off by default, so the solver objective excludes it.
    ``scale_the_objective``: multiplier on the whole objective (1.0 = none).
    """
    b = _Build(m=m, d=d, include_existing_fixed_cost=include_existing_fixed_cost)
    # Objective steps return their term (or None); terms are summed in this
    # order.  Constraint steps return None.
    obj = None
    for step in (
        _prepare,                        # flags, data checks, index sets
        _declare_variables,
        _shared_expressions,
        # ── constraints, in emit order ──
        _eq_rp_storage_chain,
        _eq_node_balance,
        _eq_node_balance_period,
        _eq_node_balance_fix_quantity,
        _eq_storage_usage_fix,
        _eq_block_coupling,
        _eq_max_flow,
        _eq_max_output_sum,
        _dc_power_flow.add_constraints,
        _online.add_constraints,         # incl. online_fix_eq
        _unit_floors.add_constraints,
        _eq_ramp,
        _eq_invest_max_units,
        _eq_invest_period_caps,
        _eq_invest_total_caps,
        _eq_conversion,
        _eq_min_input_share,
        _eq_co2_max_period,
        _eq_co2_max_total,
        _eq_user_constraints,
        _eq_max_state,
        _eq_state_profiles,
        _eq_storage_start_binding,
        _eq_storage_start_binding_cyclic,
        _eq_storage_reference_value,
        _eq_profiles,
        # ── objective ──
        _obj_balance_slack,
        _obj_commodity,
        _obj_co2_price,
        _obj_variable_cost,
        _obj_startup_cost,
        _obj_storage_reference_price,
        _obj_invest_divest,
        _obj_existing_fixed_cost,
        # ── feature modules: constraints + objective terms ──
        _feature_group_slack,
        _feature_reserve,
        _feature_cumulative_invest,
        _feature_delay,
        _feature_commodity_ladder,
        # ── stochastic ties ──
        _eq_non_anticipativity,
        _eq_non_anticipativity_invest,
    ):
        _build_prof(f"{step.__module__.rsplit('.', 1)[-1]}.{step.__name__}")
        term = step(b)
        if term is not None:
            obj = term if obj is None else obj + term

    if scale_the_objective != 1.0:
        obj = obj * scale_the_objective
    m.set_objective(obj, sense="min")
    _build_prof("build_flextool:return")

    # HiGHS options of the active solve (solve_mode.csv: highs_method /
    # highs_parallel / highs_presolve); None keeps the HiGHS defaults.
    if getattr(d, "solver_options", None):
        m.set_solver_options(d.solver_options)


# ---------------------------------------------------------------------------
# Preparation: flags, data checks, index sets


def _prepare(b: _Build) -> None:
    """Problem settings, feature flags, data checks and shared index sets."""
    m, d = b.m, b.d

    # (d, t) are the dense trailing axes of every indexed frame; declaring
    # them lets polar_high assemble the matrix from block-COO slices.
    m.declare_dense_axes(("d", "t"))
    # Matrix coefficients and RHS terms with |value| < threshold become 0
    # (model.small_number_threshold, schema default 1e-4; 0 disables).
    m.coef_zero_threshold = float(getattr(d, "small_number_threshold", 0.0001))

    _check(d, ALWAYS, "always")

    # ── feature flags ──
    b.has_proc = _has_rows(d.process_source_sink)
    b.has_indirect = _has_rows(d.process_indirect)
    b.has_co2_price = _has_rows(d.flow_from_co2_priced)
    b.has_co2_cap_eff = _has_rows(d.flow_from_co2_capped)
    b.has_co2_cap_noEff = _has_rows(getattr(d, "flow_from_co2_capped_noEff", None))
    b.has_co2_cap = b.has_co2_cap_eff or b.has_co2_cap_noEff
    b.has_co2_cap_total_eff = _has_rows(getattr(d, "flow_from_co2_capped_total", None))
    b.has_co2_cap_total_noEff = _has_rows(
        getattr(d, "flow_from_co2_capped_total_noEff", None))
    b.has_co2_cap_total = b.has_co2_cap_total_eff or b.has_co2_cap_total_noEff
    b.has_user_cstr = any(_has_rows(x) for x in (d.cdt_eq, d.cdt_le, d.cdt_ge))
    b.has_profile = any(_has_rows(x) for x in (d.process_profile_upper,
                                               d.process_profile_lower,
                                               d.process_profile_fixed))
    b.has_storage = _has_rows(d.nodeState)
    b.has_rp = _has_rows(d.nodeState_rp)
    b.has_nsb = _has_rows(d.nodeStateBlock)
    b.has_online_lin = _has_rows(d.process_online_linear)
    b.has_online_int = _has_rows(d.process_online_integer)
    b.has_online = b.has_online_lin or b.has_online_int
    b.has_minload_eff = _has_rows(d.process_min_load_eff)
    b.has_startup_cost_lin = _has_rows(d.pdt_online_linear)
    b.has_startup_cost_int = _has_rows(d.pdt_online_integer)
    b.has_ramp = any(_has_rows(getattr(d, f))
                     for f in ("process_source_sink_ramp_limit_sink_up",
                               "process_source_sink_ramp_limit_sink_down",
                               "process_source_sink_ramp_limit_source_up",
                               "process_source_sink_ramp_limit_source_down"))
    b.has_invest_p = _has_rows(d.pd_invest_set)
    b.has_divest_p = _has_rows(d.pd_divest_set)
    b.has_invest_n = _has_rows(d.nd_invest_set)
    b.has_divest_n = _has_rows(d.nd_divest_set)
    b.has_ladder = _commodity_ladder.has_feature(d)
    # p_nested_solve_first: None for a single solve, else True on the first
    # solve of a chain.
    b.is_solve_first = (d.p_nested_solve_first is None
                        or d.p_nested_solve_first is True)

    # ── required data of the active features ──
    if b.has_proc:
        _check(d, PROCESSES, "processes")
    if b.has_indirect:
        _check(d, INDIRECT, "conversion_indirect")
    # The CO2 price and content are authored apart from the priced-group
    # membership; without them the cost term is skipped with a warning.
    if b.has_co2_price and (d.p_co2_price is None or d.p_co2_content is None):
        missing = [f for f in ("p_co2_price", "p_co2_content")
                   if getattr(d, f, None) is None]
        _LOG.warning(
            "CO2 price method is active (flow_from_co2_priced is non-empty) "
            "but %s is not populated — CO2 emissions will not be priced in "
            "this solve.  Populate group.co2_price / commodity.co2_content "
            "to enable the CO2 cost term, or remove the CO2-priced group "
            "membership to silence this warning.",
            " and ".join(missing),
        )
        b.has_co2_price = False
    if b.has_co2_cap:
        _check(d, CO2_CAP, "co2_max_period")
    if b.has_co2_cap_total:
        _check(d, CO2_CAP_TOTAL, "co2_max_total")
    if b.has_user_cstr:
        _check(d, USER_CSTR, "user_constraints")
    if b.has_profile:
        _check(d, PROFILES, "profile_flow")
    if b.has_storage:
        _check(d, STORAGE, "storage")
    if b.has_ramp:
        _check(d, RAMP, "ramp_limit")
    if b.has_invest_p or b.has_invest_n:
        _check(d, INVEST, "invest")
    if b.has_divest_p:
        _check(d, DIVEST, "divest")
    if b.has_online:
        _check(d, ONLINE, "online")
    if b.has_minload_eff:
        _check(d, MINLOAD_EFF, "min_load_efficiency")
    if b.has_startup_cost_lin:
        _check(d, STARTUP_COST_LINEAR, "startup_cost_linear")
    if b.has_startup_cost_int:
        _check(d, STARTUP_COST_INTEGER, "startup_cost_integer")

    # ── shared index sets ──
    b.pss_dt = compute_pss_dt(d) if b.has_proc else None
    b.nodeBalance_dt = compute_nodeBalance_dt(d)
    _validate_has_balance(d, b.nodeBalance_dt)
    b.nodeState_dt = compute_nodeState_dt(d) if b.has_storage else None
    b.process_indirect_dt = (compute_process_indirect_dt(d)
                             if b.has_indirect else None)
    _validate_storage_binding_disjoint(d)
    if b.has_rp:
        b.rp_start_idx = compute_nodeState_rp_block_first_dt(d)
    if b.has_online_lin:
        b.p_olin_idx = d.p_online_dt.join(d.process_online_linear, on="p",
                                          how="inner")
    if b.has_online_int:
        b.p_oint_idx = d.p_online_dt.join(d.process_online_integer, on="p",
                                          how="inner")
    if b.has_proc:
        b.twoway1var_arcs = _two_way_one_var_arcs(d)
        procs = [s.select(pl.col("p").cast(pl.Utf8))
                 for s in (d.pd_invest_set if b.has_invest_p else None,
                           d.pd_divest_set if b.has_divest_p else None)
                 if s is not None]
        b.invest_procs = pl.concat(procs).unique() if procs else None
        (b.flow_upper_rhs, b.flow_cap_mult,
         b.flow_var_upper, b.flow_bound_arcs) = _max_flow_rhs(d, b.invest_procs)


def _node_list(nodes: list) -> str:
    """Comma-separated node names for error messages, at most ten shown."""
    shown = ", ".join(str(n) for n in nodes[:10])
    return shown + ("" if len(nodes) <= 10 else f", ... {len(nodes) - 10} more")


def _validate_has_balance(d, nodeBalance_dt) -> None:
    """Raise when no node has an energy balance to solve."""
    if nodeBalance_dt is None and not _has_rows(getattr(d, "nodeBalancePeriod", None)):
        raise ValueError(
            "No node has an energy balance: every node resolved to "
            "node_type = 'commodity' (or 'none'), so there is no "
            "per-time-step balance ('balance'/'storage') and no period "
            "balance ('balance_within_period') to solve.  Check the "
            "node_type of your nodes and the alternative ranking of the "
            "active scenario — a high-ranked alternative (e.g. 'Base') "
            "that sets node_type = 'commodity' will override the "
            "'balance'/'storage' assignments of lower-ranked "
            "alternatives."
        )


def _validate_storage_binding_disjoint(d) -> None:
    """Raise if a node is in more than one storage_bind_* set.

    nodeBalance_eq adds one state-change term per set, so a node in two
    sets would get two.
    """
    frames = [
        ("bind_within_timeblock", d.storage_bind_within_timeblock),
        ("bind_forward_only", d.storage_bind_forward_only),
        ("bind_within_solve", d.storage_bind_within_solve),
        ("bind_within_solve_blended_weights",
         d.storage_bind_within_solve_blended_weights),
        ("bind_forward_only_blended_weights",
         d.storage_bind_forward_only_blended_weights),
        ("bind_within_period_blended_weights",
         getattr(d, "storage_bind_within_period_blended_weights", None)),
    ]
    active = [(name, frame) for name, frame in frames if _has_rows(frame)]
    for i, (name_i, frame_i) in enumerate(active):
        for name_j, frame_j in active[i + 1:]:
            overlap = (frame_i.select("n").unique()
                       .join(frame_j.select("n").unique(), on="n", how="inner"))
            if overlap.height > 0:
                raise FlexToolConfigError(
                    "storage_binding_method projections must be pairwise "
                    "disjoint on node, but the following node(s) appear in "
                    f"both '{name_i}' and '{name_j}': "
                    f"{_node_list(overlap['n'].to_list())}. "
                    "Run the v54 DB migration (or later, via the v53->v54 "
                    "migration step) and ensure each node has exactly one "
                    "storage_binding_method value."
                )


def _two_way_one_var_arcs(d) -> "pl.DataFrame | None":
    """(p, source, sink) arcs whose reverse flow is ``v_flow_back``.

    Union of the method_2way_1var_off arcs (``process_source_sink_2way_1var``)
    and the DC power-flow arcs; the latter come straight from the DC
    module so they are covered even when the projection field is absent.
    """
    parts = []
    if _has_rows(d.process_source_sink_2way_1var):
        parts.append(d.process_source_sink_2way_1var.select("p", "source", "sink"))
    if _dc_power_flow.has_feature(d):
        dc_arcs = _dc_power_flow.dc_arcs_frame(d)
        if _has_rows(dc_arcs):
            parts.append(dc_arcs.select("p", "source", "sink"))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    ref_schema = parts[0].schema
    aligned = [parts[0]]
    for part in parts[1:]:
        for col in ("p", "source", "sink"):
            if part.schema[col] != ref_schema[col]:
                part = part.with_columns(
                    pl.col(col).cast(ref_schema[col], strict=False))
        aligned.append(part)
    return pl.concat(aligned).unique()


# ---------------------------------------------------------------------------
# Variables


def _declare_variables(b: _Build) -> None:
    """Declare every decision variable; the order fixes the LP column order."""
    m, d = b.m, b.d

    # ─── VAR. ─── v_flow ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,source,sink,d,t] ≥ 0   ∀ (p,source,sink,d,t) ∈ pss_dt
    # Flow on a process arc in units of unitsize: MW = v_flow · unitsize[p].
    # Input arcs of indirect units without invest/divest variables carry a
    # per-element upper bound (their capacity is a constant); maxFlow caps
    # every other arc.
    if b.has_proc:
        upper = b.flow_var_upper if b.flow_var_upper is not None else float("inf")
        b.v_flow = m.add_var("v_flow", ("p", "source", "sink", "d", "t"), b.pss_dt, lower=0.0, upper=upper)

    # ─── VAR. ─── vq_state_up / vq_state_down ────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # vq_state_up[n,d,t], vq_state_down[n,d,t] ≥ 0
    #   ∀ n ∈ nodeBalance ∪ nodeBalancePeriod, (d,t) ∈ dt
    # Balance slack in MW (shortfall / surplus); enters the balance times
    # step_duration and is penalised in the objective.  Nodes with
    # penalty_method = off have none, so their balance is a hard equality.
    slack_idx = _balance_slack_index(b)
    b.vq_state_up = m.add_var("vq_state_up", ("n", "d", "t"), slack_idx, lower=0.0)
    b.vq_state_down = m.add_var("vq_state_down", ("n", "d", "t"), slack_idx, lower=0.0)

    # ─── VAR. ─── v_state ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] ≥ 0   ∀ (n,d,t) ∈ nodeState_dt
    # Storage state in units of the node's state unitsize:
    # MWh = v_state · state_unitsize[n].  Its upper bound depends on
    # invest/divest and is the maxState row.
    if b.has_storage:
        b.v_state = m.add_var("v_state", ("n", "d", "t"), b.nodeState_dt, lower=0.0)

    # ─── VAR. ─── v_state_inter / v_state_rp_start ───────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,b] ≥ 0          ∀ n ∈ nodeState_rp, b ∈ rp_base_period
    # v_state_rp_start[n,d,t] ≥ 0     ∀ n ∈ nodeState_rp, (d,t) ∈ rp_block_first
    # Representative-period storage, in state units: the state at each
    # base-period boundary, and the free starting state of each
    # representative block.  Upper bounds are the rows
    # rp_inter_period_max_state and maxState_rp_start.
    if b.has_rp:
        b.v_state_inter = m.add_var("v_state_inter", ("n", "b"),
                                    d.nodeState_rp.join(d.rp_base_period_set, how="cross"), lower=0.0)
        b.v_state_rp_start = m.add_var("v_state_rp_start", ("n", "d", "t"), b.rp_start_idx, lower=0.0)

    # Unit commitment (v_online, v_startup, v_shutdown): declared by _online.
    _online.add_variables(b)

    # ─── VAR. ─── v_invest_p / v_divest_p / v_invest_n / v_divest_n ──────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] ≥ 0 ∀ (p,d) ∈ pd_invest_set;  v_divest_p[p,d] ≥ 0 ∀ (p,d) ∈ pd_divest_set
    # v_invest_n[n,d] ≥ 0 ∀ (n,d) ∈ nd_invest_set;  v_divest_n[n,d] ≥ 0 ∀ (n,d) ∈ nd_divest_set
    # Capacity built / retired in period d, as a count of unitsize (process)
    # or state unitsize (storage node).  Upper bounds are the
    # maxInvest_var_bound / maxDivest_var_bound rows.
    if b.has_invest_p:
        b.v_invest_p = m.add_var("v_invest_p", ("p", "d"), d.pd_invest_set)
    if b.has_divest_p:
        b.v_divest_p = m.add_var("v_divest_p", ("p", "d"), d.pd_divest_set)
    if b.has_invest_n:
        b.v_invest_n = m.add_var("v_invest_n", ("n", "d"), d.nd_invest_set)
    if b.has_divest_n:
        b.v_divest_n = m.add_var("v_divest_n", ("n", "d"), d.nd_divest_set)

    # Reserve variables (v_reserve, vq_reserve): declared by _reserve.
    b.reserve_vars = _reserve.add_variables(m, d) if _reserve.has_feature(d) else {}

    # ─── VAR. ─── v_flow_back ────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow_back[p,source,sink,d,t] ≥ 0   ∀ (p,source,sink) ∈ twoway1var_arcs, (d,t) ∈ dt
    # Reverse flow (sink → source) of method_2way_1var_off and DC power-flow
    # arcs, in units of unitsize; the signed flow is v_flow − v_flow_back.
    # One variable serves both the balance and dc_flow_eq.
    if _has_rows(b.twoway1var_arcs):
        back_idx = b.twoway1var_arcs.join(d.dt, how="cross").select("p", "source", "sink", "d", "t")
        if back_idx.height > 0:
            b.v_flow_back = m.add_var("v_flow_back", ("p", "source", "sink", "d", "t"), back_idx, lower=0.0)

    # DC power-flow angles (v_angle) and commodity-ladder trades (v_trade):
    # declared by their feature modules.
    _dc_power_flow.add_variables(b)
    b.ladder_vars = _commodity_ladder.add_variables(m, d) if b.has_ladder else {}


def _balance_slack_index(b: _Build) -> pl.DataFrame:
    """(n, d, t) of the balance slacks: nodeBalance_dt ∪ nodeBalancePeriod × dt,
    without penalty_method = off nodes."""
    d = b.d
    idx = b.nodeBalance_dt
    if _has_rows(d.nodeBalancePeriod):
        period_dt = (d.nodeBalancePeriod.lazy()
                     .join(d.dt.lazy(), how="cross")
                     .select("n", "d", "t")
                     .collect())
        ref_schema = b.nodeBalance_dt.schema
        for col in ("n", "d", "t"):
            if period_dt.schema[col] != ref_schema[col]:
                period_dt = period_dt.with_columns(
                    pl.col(col).cast(ref_schema[col], strict=False))
        idx = pl.concat([b.nodeBalance_dt, period_dt]).unique()
    penalty_off = getattr(d, "nodeBalance_penalty_off", None)
    if _has_rows(penalty_off):
        off_n = penalty_off.select("n")
        if off_n.schema["n"] != idx.schema["n"]:
            off_n = off_n.with_columns(
                pl.col("n").cast(idx.schema["n"], strict=False))
        idx = idx.join(off_n, on="n", how="anti")
    return idx


# ---------------------------------------------------------------------------
# Shared expressions


def _shared_expressions(b: _Build) -> None:
    """Expressions read by more than one equation family or objective term."""
    d = b.d
    _reserve_coupling(b)
    b.node_balance_flow_terms = _node_balance_flow_terms(b)
    _invest_in_dispatch(b)
    # Weight of every operational cost term per (d, t): step_duration ·
    # timestep_weight · inflation_op / period_share, times the branch
    # weight in stochastic solves.
    b.op_factor = (d.p_step_duration * d.p_timestep_weight
                   * d.p_inflation_op / d.p_period_share)
    if d.pdt_branch_weight is not None:
        b.op_factor = b.op_factor * d.pdt_branch_weight


def _reserve_coupling(b: _Build) -> None:
    """Σ_r v_reserve aggregates that share process capacity with v_flow."""
    v_reserve = b.reserve_vars.get("v_reserve")
    if v_reserve is None:
        return
    pruna = b.d.process_reserve_upDown_node_active          # (p, r, ud, n)
    if not _has_rows(pruna):
        return
    # v_reserve is in units of unitsize, like v_flow.  Upward reserve held
    # at a sink competes with the flow into it (maxFlow, maxOutputSum,
    # profile_flow_upper_limit); downward reserve at a sink lowers the
    # floor of profile_flow_lower_limit.
    up_sink = (pruna.filter(pl.col("ud") == "up")
               .pipe(rename_to_axis, {"n": "sink"})
               .select("p", "r", "ud", "sink"))
    if up_sink.height > 0:
        at_sink = _var_view(v_reserve, "__at_sink", {"n": "sink"}, to_axis=True)
        b.reserve_up_to_sink = Sum(Where(at_sink, up_sink), over=("r", "ud"))
        b.reserve_up_sink_parts = (at_sink, up_sink)

    down_sink = (pruna.filter(pl.col("ud") == "down")
                 .pipe(rename_to_axis, {"n": "sink"})
                 .select("p", "r", "ud", "sink"))
    if down_sink.height > 0:
        at_sink_dn = _var_view(v_reserve, "__at_sink_dn", {"n": "sink"},
                               to_axis=True)
        b.reserve_down_to_sink = Sum(Where(at_sink_dn, down_sink),
                                     over=("r", "ud"))


def _invest_in_dispatch(b: _Build) -> None:
    """(p, d) sums of the process invest / divest units in service in d.

    invest_in_dispatch_p[p,d] = Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    divest_in_dispatch_p[p,d] = Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    Read by maxFlow, maxOutputSum and the ramp limits.
    """
    if not b.has_proc:
        return
    d = b.d
    if b.has_divest_p:
        divested = _var_view(b.v_divest_p, "__at_divest", {"d": "d_divest"},
                             to_axis=True)
        b.divest_in_dispatch_p = Sum(Where(divested, d.edd_divest_active),
                                     over=("d_divest",))
    if b.has_invest_p:
        invested = _var_view(b.v_invest_p, "__at_invest", {"d": "d_invest"},
                             to_axis=True)
        alive = _entity_rows(d.edd_invest_set, "e", d.process_source_sink, "p")
        b.invest_in_dispatch_p = Sum(Where(invested, alive), over=("d_invest",))


# ---------------------------------------------------------------------------
# Equations


def _eq_rp_storage_chain(b: _Build) -> None:
    """Representative-period storage: inter-period state chain and its bounds."""
    if not (b.has_storage and b.has_rp):
        return
    m, d = b.m, b.d
    v_inter = b.v_state_inter
    b_dtype = v_inter.frame.schema["b"]
    has_rep_blocks = d.rp_base__rep is not None and _has_rows(d.rp_block_first) and _has_rows(d.p_rp_last_step)

    # ─── EQ. ─── rp_inter_period_balance ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,b] − v_state_inter[n,b_prev]
    #   = Σ_{(b,r) ∈ rp_base__rep, (d,r) ∈ rp_block_first, d ∈ period_in_use}
    #       rp_weight[b,r] · (v_state[n,d,last_step[r]] − v_state_rp_start[n,d,r])
    #   ∀ n ∈ nodeState_rp, (b,b_prev) ∈ rp_base_chain
    # Chains the storage state across base periods: each base period is
    # represented by weighted representative blocks r, and the boundary state
    # moves by the weighted net change of those blocks.
    # All terms are in state units (no state_unitsize factor).
    if has_rep_blocks and _has_rows(d.rp_base_chain):
        nodes = d.nodeState_rp
        chain = d.rp_base_chain.with_columns(
            pl.col("b").cast(b_dtype, strict=False),
            pl.col("b_prev").cast(b_dtype, strict=False),
        )
        over = nodes.join(chain, how="cross").select("n", "b", "b_prev")
        inter_prev = _var_view(v_inter, "__bprev", {"b": "b_prev"})
        drift = _rp_weighted_drift(b, nodes, d.rp_base__rep.frame, "b", "")
        if drift is not None:
            m.add_cstr(
                "rp_inter_period_balance",
                over      = over,
                sense     = "==",
                lhs_terms = {"d_inter": Where(v_inter, over) - Where(inter_prev, over),
                             "neg_rp_drift": -Where(drift, over)},
                rhs_terms = {},
            )

    # ─── EQ. ─── rp_inter_period_cyclic ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,b_first] − v_state_inter[n,b_last]
    #   = Σ_{(b_first,r) ∈ rp_base__rep, (d,r) ∈ rp_block_first, d ∈ period_in_use}
    #       rp_weight[b_first,r] · (v_state[n,d,last_step[r]] − v_state_rp_start[n,d,r])
    #   ∀ n ∈ bind_within_{solve,period}_blended_weights,
    #     (b_first,b_last) ∈ rp_base_first × rp_base_last
    # Closes the chain: the last boundary state connects back to the first.
    # bind_within_solve_blended_weights closes once per solve,
    # bind_within_period_blended_weights once per period (rp_base_first /
    # rp_base_last then carry d and are paired on it).
    # bind_forward_only_blended_weights has no closure.
    cyclic_sets = [f for f in (d.storage_bind_within_solve_blended_weights,
                               getattr(d, "storage_bind_within_period_blended_weights", None))
                   if _has_rows(f)]
    if has_rep_blocks and cyclic_sets and _has_rows(d.rp_base_first) and _has_rows(d.rp_base_last):
        nodes = pl.concat(cyclic_sets, how="vertical").unique()
        first = d.rp_base_first.rename({"b": "b_first"}).with_columns(pl.col("b_first").cast(b_dtype, strict=False))
        last = d.rp_base_last.rename({"b": "b_last"}).with_columns(pl.col("b_last").cast(b_dtype, strict=False))
        if "d" in first.columns and "d" in last.columns:
            pairs = first.join(last, on="d", how="inner").select("b_first", "b_last")
        else:
            pairs = first.select("b_first").join(last.select("b_last"), how="cross")
        over = nodes.join(pairs, how="cross").select("n", "b_first", "b_last")
        inter_first = _var_view(v_inter, "__bfirst", {"b": "b_first"})
        inter_last = _var_view(v_inter, "__blast", {"b": "b_last"})
        rep_first = (d.rp_base__rep.frame.rename({"b": "b_first"})
                     .with_columns(pl.col("b_first").cast(b_dtype, strict=False)))
        drift = _rp_weighted_drift(b, nodes, rep_first, "b_first", "_cyc")
        if drift is not None:
            m.add_cstr(
                "rp_inter_period_cyclic",
                over      = over,
                sense     = "==",
                lhs_terms = {"d_cyclic": Where(inter_first, over) - Where(inter_last, over),
                             "neg_rp_drift_first": -Where(drift, over)},
                rhs_terms = {},
            )

    # ─── EQ. ─── rp_inter_period_max_state / maxState_rp_start ───────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,b] + Σ_{(n,d′,d) ∈ edd_divest_active} v_divest_n[n,d′]
    #   − Σ_{(n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′]  ≤  state_upper[n,d]
    #   ∀ n ∈ nodeState_rp, b ∈ rp_base_period, d ∈ period_in_use
    # v_state_rp_start[n,d,t] + (same invest/divest terms)  ≤  state_upper[n,d]
    #   ∀ (n,d,t) ∈ nodeState_rp × rp_block_first
    # Capacity bounds of the boundary and block-start states, as maxState
    # bounds v_state.  state_upper = existing / state_unitsize, so every term
    # is in state units.  Without period_in_use, d runs over the periods of dt.
    nodes = d.nodeState_rp
    if _has_rows(d.rp_base_period_set):
        if _has_rows(d.period_in_use_set):
            periods = d.period_in_use_set.select("d").unique()
        else:
            periods = d.dt.select("d").unique()
        if periods.height > 0:
            over = nodes.join(d.rp_base_period_set, how="cross").join(periods, how="cross").select("n", "b", "d")
            lhs: dict = {"state_inter": Where(v_inter, over)}
            lhs.update(_node_capacity_change_terms(b, nodes, "_rp"))
            m.add_cstr(
                "rp_inter_period_max_state",
                over      = over,
                sense     = "<=",
                lhs_terms = lhs,
                rhs_terms = {"upper": d.p_state_upper},
            )
    if _has_rows(b.rp_start_idx):
        lhs = {"state_rp_start": b.v_state_rp_start}
        lhs.update(_node_capacity_change_terms(b, nodes, "_rp"))
        m.add_cstr(
            "maxState_rp_start",
            over      = b.rp_start_idx,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": d.p_state_upper},
        )


def _rp_weighted_drift(b: _Build, nodes: pl.DataFrame, rep: pl.DataFrame,
                       base: str, tag: str):
    """Σ_{(base,r), d} rp_weight · (v_state[n,d,last_step[r]] − v_state_rp_start[n,d,r]).

    ``rep`` holds (base, r, value=rp_weight); the result keeps dims
    (n, base).  None when no representative block is in use.  ``tag``
    keeps the variable-view names of the two callers apart.
    """
    d = b.d
    t_dtype = d.rp_block_first.schema["t"]
    state_t_dtype = b.v_state.frame.schema["t"]
    # Block r starts at (d, t = r) ∈ rp_block_first and ends at last_step[r].
    block_first = (d.rp_block_first.rename({"t": "r"})
                   .with_columns(pl.col("r").cast(t_dtype, strict=False)))
    rep = rep.with_columns(pl.col("r").cast(t_dtype, strict=False))
    last_step = d.p_rp_last_step.with_columns(
        pl.col("r").cast(t_dtype, strict=False),
        pl.col("last_step").cast(state_t_dtype, strict=False))
    cohort = (rep.join(block_first, on="r", how="inner")
              .join(last_step, on="r", how="inner"))
    if d.period_in_use_set is not None:
        cohort = cohort.join(d.period_in_use_set, on="d", how="inner")
    if cohort.height == 0:
        return None
    keys = (base, "r", "d", "last_step")
    rp_weight = Param(keys, cohort.select(*keys, "value"))
    state_at_last = _var_view(b.v_state, "__rp_last" + tag, {"t": "last_step"})
    start_at_r = _var_view(b.v_state_rp_start, "__at_r" + tag, {"t": "r"})
    idx = nodes.select("n").unique().join(cohort.select(*keys), how="cross")
    return Sum((Where(state_at_last, idx) - Where(start_at_r, idx)) * rp_weight,
               over=("r", "d", "last_step"))


def _node_capacity_change_terms(b: _Build, nodes: pl.DataFrame, tag: str) -> dict:
    """``{"divest": Σ v_divest_n, "invest_neg": −Σ v_invest_n}`` over the
    invest / divest periods in service in d, dims (n, d), for ``nodes``.

    ``tag`` keeps the variable-view names of different callers apart.
    """
    invested, divested = _node_in_service(b, nodes, tag)
    terms: dict = {}
    if divested is not None:
        terms["divest"] = divested
    if invested is not None:
        terms["invest_neg"] = -invested
    return terms


def _node_in_service(b: _Build, nodes: pl.DataFrame, tag: str, coef: "Param | None" = None) -> tuple:
    """``(Σ v_invest_n[n,d′], Σ v_divest_n[n,d′])`` over the invest / divest periods d′ in service in d,
    dims (n, d), for ``nodes``; each is None when there is no such variable or period.

    ``coef`` multiplies the variable inside the sum.  ``tag`` keeps the variable-view names of different
    callers apart.
    """
    d = b.d
    invested = divested = None
    if b.has_invest_n and d.edd_invest_set is not None:
        view = _var_view(b.v_invest_n, "__at_invest" + tag, {"d": "d_invest"}, to_axis=True)
        alive = _entity_rows(d.edd_invest_set, "e", nodes, "n")
        if alive.height > 0:
            invested = Sum(Where(view if coef is None else view * coef, alive), over=("d_invest",))
    if b.has_divest_n and d.edd_divest_active is not None:
        view = _var_view(b.v_divest_n, "__at_divest" + tag, {"d": "d_divest"}, to_axis=True)
        active = _entity_rows(d.edd_divest_active, "p", nodes, "n")
        if active.height > 0:
            divested = Sum(Where(view if coef is None else view * coef, active), over=("d_divest",))
    return invested, divested


def _eq_node_balance(b: _Build) -> None:
    """Per-step energy balance of balance and storage nodes."""
    m, d = b.m, b.d

    # ─── EQ. ─── nodeBalance_eq ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #       (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t]
    #        + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   + back-flow terms of two-way one-variable arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d,t⁻] − v_state[n,d,t])·state_unitsize[n]
    #   − v_state[n,d,t]·self_discharge[n]·state_unitsize[n]
    #   + first-step start terms
    #   = −inflow[n,d,t]
    #   ∀ (n,d,t) ∈ nodeBalance_dt, n ∉ nodeStateBlock;   Δt = step_duration[d,t]
    # Energy conservation per node and time step, in MWh.  Storage nodes add
    # the state change; the previous step t⁻ depends on storage_binding_method.
    # nodeStateBlock nodes are balanced per coarse block (nodeBalanceBlock_eq).
    # The section term exists for min_load_efficiency units only.  The flow
    # and slack terms are b.node_balance_flow_terms.  Self-discharge applies
    # the per-step loss rate without step_duration (exact for 1 h steps).
    terms = dict(b.node_balance_flow_terms)
    if b.has_storage:
        terms.update(_state_change_terms(b))
        terms.update(_fix_start_terms(b))
        terms.update(_roll_continue_terms(b))
        if d.p_state_self_discharge is not None:
            terms["self_discharge"] = -b.v_state * d.p_state_self_discharge * d.p_state_unitsize
    over = b.nodeBalance_dt
    if b.has_nsb:
        over = over.join(d.nodeStateBlock, on="n", how="anti")
    m.add_cstr(
        "nodeBalance_eq",
        over      = over,
        sense     = "==",
        lhs_terms = terms,
        rhs_terms = {"neg_inflow": -d.p_inflow},
    )


def _node_balance_flow_terms(b: _Build) -> dict:
    """Flow and slack terms of nodeBalance_eq, (n, d, t) in MWh.

    Shared with nodeBalancePeriod_eq: period nodes are never storage, so
    they take only these terms.
    """
    d = b.d
    terms: dict = {"slack_up": b.vq_state_up * d.p_step_duration,
                   "slack_down": -b.vq_state_down * d.p_step_duration}
    if not b.has_proc:
        return terms
    v_flow = b.v_flow
    terms["sink_flow"] = Sum(
        Where(v_flow * d.p_unitsize, d.flow_to_n) * d.p_step_duration,
        over=("p", "source", "sink"))
    # A unit with an efficiency draws flow · slope from its source node; a
    # min_load_efficiency unit also draws its section input per online unit.
    from_eff = d.flow_from_nodeBalance_eff
    if _has_rows(from_eff):
        terms["source_eff"] = -Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, from_eff) * d.p_step_duration,
            over=("p", "source", "sink"))
        if b.has_minload_eff and d.p_section is not None:
            section_idx = from_eff.join(d.process_min_load_eff, on="p", how="inner")
            if section_idx.height > 0:
                for key, v_online in (("source_section_lin", b.v_online_linear),
                                      ("source_section_int", b.v_online_integer)):
                    if v_online is None:
                        continue
                    terms[key] = -Sum(
                        Where(Where(v_online, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx) * d.p_step_duration,
                        over=("p", "source", "sink"))
    from_noEff = d.flow_from_nodeBalance_noEff
    if _has_rows(from_noEff):
        terms["source_noEff"] = -Sum(
            Where(v_flow * d.p_unitsize, from_noEff) * d.p_step_duration,
            over=("p", "source", "sink"))
    # v_flow_back adds to the source node and takes from the sink node.
    if b.v_flow_back is not None and b.twoway1var_arcs is not None:
        terms.update(_dc_power_flow.nodeBalance_back_flow_terms(
            b.twoway1var_arcs, b.v_flow_back, d.p_unitsize, d.p_step_duration))
    return terms


def _state_change_terms(b: _Build) -> dict:
    """State-change terms of nodeBalance_eq, one per storage_binding_method."""
    d, v_state = b.d, b.v_state
    su = d.p_state_unitsize
    terms: dict = {}
    # The balance reads (state[t⁻] − state[t]) + sink − source = −inflow,
    # i.e. state[t] − state[t⁻] = sink − source + inflow.  Each node is in
    # one binding set only (_validate_storage_binding_disjoint).

    # bind_within_timeblock: t⁻ wraps around within the timeblock.
    if d.storage_bind_within_timeblock is not None:
        nodes = d.storage_bind_within_timeblock
        now = Where(v_state, nodes)
        prev = Where(Lag(v_state, d.dtttdt, "t", "t_previous_within_timeset"), nodes)
        terms["state_change"] = (prev - now) * su

    # bind_forward_only: t⁻ crosses period boundaries; the first step of the
    # solve has no state change (dtttdt_forward_only omits it, so both parts
    # of the term are dropped there).
    if _has_rows(d.storage_bind_forward_only) and _has_rows(d.dtttdt_forward_only):
        nodes = d.storage_bind_forward_only
        steps = d.dtttdt_forward_only.select("d", "t").unique()
        now = Where(Where(v_state, steps), nodes)
        prev = Where(_state_lag_cross_period(v_state, d.dtttdt_forward_only), nodes)
        terms["state_change_fo"] = (prev - now) * su

    # bind_within_solve: t⁻ crosses period boundaries and the first step of
    # the solve wraps to the last one.
    if _has_rows(d.storage_bind_within_solve):
        nodes = d.storage_bind_within_solve
        now = Where(v_state, nodes)
        prev = Where(_state_lag_cross_period(v_state, d.dtttdt), nodes)
        terms["state_change_ws"] = (prev - now) * su

    # Blended weights (nodeState_rp, all three variants): t⁻ is the previous
    # step inside a representative block; the first step of each block
    # starts from the free state v_state_rp_start.
    if b.has_rp:
        nodes = d.nodeState_rp
        interior = d.dtttdt.join(d.rp_block_first.select("d", "t"),
                                 on=["d", "t"], how="anti")
        interior_dt = interior.select("d", "t").unique()
        now = Where(Where(v_state, interior_dt), nodes)
        prev = Where(Lag(v_state, interior, "t", "t_previous_within_timeset"), nodes)
        terms["state_change_rp_interior"] = (prev - now) * su
        first_dt = d.rp_block_first.select("d", "t").unique()
        now = Where(Where(v_state, first_dt), nodes)
        start = Where(b.v_state_rp_start, nodes)
        terms["state_change_rp_start"] = (start - now) * su
    return terms


def _state_lag_cross_period(v_state: Var, lag_frame: pl.DataFrame):
    """v_state at (d_previous, t_previous_within_solve) of each (d, t) in
    ``lag_frame``; unlike ``Lag``, the previous step may be in another period."""
    v_state_back = _var_view(v_state, "__back", {"d": "d_back", "t": "t_back"},
                             to_axis=True)
    lag = lag_frame.select("d", "t",
                           alias_to_axis("d_previous", "d_back"),
                           alias_to_axis("t_previous_within_solve", "t_back"))
    return Sum(Where(v_state_back, lag), over=("d_back", "t_back"))


def _fix_start_terms(b: _Build) -> dict:
    """nodeBalance_eq start terms of bind_forward_only nodes with fix_start.

    At the first step of the first solve, where the state change is
    omitted:  − v_state·su + state_start·(existing + Σ v_invest_n·su − Σ v_divest_n·su),
    the change from the start state (su = state_unitsize).  Other fix_start
    nodes are pinned by storage_state_start_binding.
    """
    d = b.d
    if not (b.is_solve_first
            and _has_rows(d.storage_bind_forward_only)
            and _has_rows(d.storage_fix_start)
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_first_dt is not None):
        return {}
    nodes = d.storage_bind_forward_only.join(d.storage_fix_start, on="n", how="inner")
    if nodes.height == 0:
        return {}
    first = d.nodeState_first_dt.join(nodes, on="n", how="inner")
    if first.height == 0:
        return {}
    start, su = d.p_state_start, d.p_state_unitsize
    terms: dict = {"fwd_fix_state": -(Where(b.v_state, first) * su)}
    # A sparse Param term combines with the dense −inflow by outer join with
    # 0-fill, so rows outside ``first`` keep their balance.
    start_energy = ((start * d.p_state_existing_capacity).frame
                    .join(first, on=["n", "d"], how="inner")
                    .select("n", "d", "t", "value"))
    if start_energy.height > 0:
        terms["fwd_fix_existing_pos"] = Param(("n", "d", "t"), start_energy)
    if b.has_invest_n and d.edd_invest_set is not None:
        invested = _var_view(b.v_invest_n, "__at_fwd_fix", {"d": "d_invest"},
                             to_axis=True)
        alive = (_entity_rows(d.edd_invest_set, "e", nodes, "n")
                 .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if alive.height > 0:
            terms["fwd_fix_invest_pos"] = Sum(
                Where(invested * start * su, alive), over=("d_invest",))
    if b.has_divest_n and d.edd_divest_active is not None:
        divested = _var_view(b.v_divest_n, "__at_fwd_fix", {"d": "d_divest"},
                             to_axis=True)
        active = (_entity_rows(d.edd_divest_active, "p", nodes, "n")
                  .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if active.height > 0:
            terms["fwd_fix_divest_neg"] = -Sum(
                Where(divested * start * su, active), over=("d_divest",))
    return terms


def _roll_continue_terms(b: _Build) -> dict:
    """nodeBalance_eq start terms of continuation solves in a chain.

    At the first step, for every storage node:  − v_state·su + roll_continue_state,
    which hands over the final state of the previous solve.  It adds to the
    node's own state-change term (su = state_unitsize).
    """
    d = b.d
    if (b.is_solve_first
            or d.p_state_unitsize is None
            or d.nodeState_first_dt is None
            or d.p_roll_continue_state is None):
        return {}
    first = d.nodeState_first_dt
    if first.height == 0:
        return {}
    terms: dict = {"roll_continue_state": -(Where(b.v_state, first)
                                            * d.p_state_unitsize)}
    handed_over = (d.p_roll_continue_state.frame
                   .join(first, on="n", how="inner")
                   .select("n", "d", "t", "value"))
    if handed_over.height > 0:
        terms["roll_continue_value"] = Param(("n", "d", "t"), handed_over)
    return terms


def _eq_node_balance_period(b: _Build) -> None:
    """Period energy balance of balance_within_period nodes."""
    d = b.d
    if not _has_rows(d.nodeBalancePeriod) or d.period_in_use_set is None:
        return

    # ─── EQ. ─── nodeBalancePeriod_eq ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_t w[d,t] · (flow and slack terms of nodeBalance_eq)[n,d,t]  =  −Σ_t w[d,t] · inflow[n,d,t]
    #   ∀ n ∈ nodeBalancePeriod, n ∉ nodeState, d ∈ period_in_use;   w = timestep_weight
    # Energy balance over a whole period instead of every step, in MWh.  The
    # terms are b.node_balance_flow_terms (Δt already applied); period nodes
    # are never storage, so there are no state terms.
    # Every term, slacks included, carries the timestep weight: a per-step
    # balance needs no weight, but a period sum without it would let
    # low-weight timeslices offset high-weight ones.  The other op_factor
    # parts (inflation_op, period_share) are per period and cancel in a row.
    # Inflow is energy per step (no Δt), as in nodeBalance_eq.
    over = (d.nodeBalancePeriod.select("n").lazy()
            .join(d.period_in_use_set.select("d").lazy(), how="cross"))
    if _has_rows(d.nodeState):
        over = over.join(d.nodeState.select("n").lazy(), on="n", how="anti")
    over = over.select("n", "d").collect()
    if over.height == 0:
        return
    lhs = {key: Sum(term * d.p_timestep_weight, over=("t",))
           for key, term in b.node_balance_flow_terms.items()}
    inflow_period = Param(("n", "d"), ((d.p_inflow * d.p_timestep_weight).frame.lazy()
                                       .join(d.nodeBalancePeriod.select("n").lazy(), on="n", how="inner")
                                       .group_by("n", "d")
                                       .agg(pl.col("value").sum())
                                       .select("n", "d", "value")
                                       .collect()))
    b.m.add_cstr(
        "nodeBalancePeriod_eq",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"neg_inflow": -inflow_period},
    )


def _eq_node_balance_fix_quantity(b: _Build) -> None:
    """Final storage state fixed by an upper-level solve."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.n_fix_storage_quantity)
            and _has_rows(d.ndt_fix_storage_quantity)
            and _has_rows(d.dtt_timeline_matching)
            and d.period_branch is not None
            and d.period_last is not None
            and d.p_fix_storage_quantity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_last_dt is not None):
        return

    # ─── EQ. ─── node_balance_fix_quantity_eq_lower ──────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] · state_unitsize[n]
    #   = Σ_{(d_up,d) ∈ period_branch, (d,t,t_up) ∈ dtt_timeline_matching} fix_storage_quantity[n,d_up,t_up]
    #   ∀ n ∈ n_fix_storage_quantity, d ∈ period_last, (n,d,t) ∈ nodeState_last_dt
    # Pins the storage state (MWh) at the last step of each last period to the
    # state an upper-level (coarser) solve reached at the matching time.
    # Despite the name, the row is an equality.
    target = (d.n_fix_storage_quantity
              .join(d.p_fix_storage_quantity.frame.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"}),
                    on="n", how="inner")
              .join(d.period_branch, on="d_upper", how="inner")
              .join(d.dtt_timeline_matching, on=["d", "t_upper"], how="inner")
              .join(d.period_last, on="d", how="inner")
              .join(d.nodeState_last_dt, on=["n", "d", "t"], how="inner")
              .group_by(["n", "d", "t"])
              .agg(pl.col("value").sum())
              .select("n", "d", "t", "value"))
    if target.height == 0:
        return
    over = target.select("n", "d", "t").unique()
    b.m.add_cstr(
        "node_balance_fix_quantity_eq_lower",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _eq_storage_usage_fix(b: _Build) -> None:
    """Storage usage per period capped by an upper-level solve."""
    d = b.d
    if not (b.has_proc
            and b.has_storage
            and _has_rows(d.n_fix_storage_usage)
            and _has_rows(d.ndt_fix_storage_usage)
            and _has_rows(d.dtt_timeline_matching)
            and d.period_branch is not None
            and d.period_last is not None
            and d.p_fix_storage_usage is not None
            and d.nodeState_last_dt is not None):
        return

    # ─── EQ. ─── node_storage_usage_fix_le ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,n) ∈ process_source_sink, t} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    # + Σ_{(p,n,k) ∈ process_source_sink_eff, t}
    #     (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    # + Σ_{(p,n,k) ∈ process_source_sink_noEff, t} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   ≤ Σ_{(d_up,d) ∈ period_branch, t_up : ∃t (d,t,t_up) ∈ dtt_timeline_matching} fix_storage_usage[n,d_up,t_up]
    #   ∀ n ∈ n_fix_storage_usage, d ∈ period_last, (n,d,t_last) ∈ nodeState_last_dt;   Δt = step_duration[d,t]
    # Caps the net energy drawn from storage node n over a whole period (MWh)
    # by the usage an upper-level (coarser) solve planned for the coarse
    # windows overlapping that period.  One row per period, at its last step;
    # each coarse window counts once however many fine steps it covers.
    # The section term exists for min_load_efficiency units only.  The
    # process sink/source coefficient ratio is not applied (nor in the
    # source-side variable cost); it is 1 in every current dataset.
    target = (d.n_fix_storage_usage
              .join(d.p_fix_storage_usage.frame.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"}),
                    on="n", how="inner")
              .join(d.period_branch, on="d_upper", how="inner")
              .join(d.dtt_timeline_matching.select("d", "t_upper").unique(), on=["d", "t_upper"], how="inner")
              .join(d.period_last, on="d", how="inner")
              # On (n, d) only: every window of the period sums onto its last step.
              .join(d.nodeState_last_dt, on=["n", "d"], how="inner")
              # Fixed summation order: the joins above leave row order
              # hash-dependent, which changes the float sum in the last bit.
              .sort(["n", "d", "t", "d_upper", "t_upper"])
              .group_by(["n", "d", "t"])
              .agg(pl.col("value").sum())
              .select("n", "d", "t", "value"))
    if target.height == 0:
        return

    nodes = d.n_fix_storage_usage.select("n")

    def _arcs_at_node(arcs: pl.DataFrame, side: str) -> pl.DataFrame:
        """Rows of ``arcs`` whose ``side`` node is in n_fix_storage_usage, with n = that node."""
        return (arcs.join(nodes.pipe(rename_to_axis, {"n": side}), on=side, how="inner")
                .with_columns(n=pl.col(side)))

    v_flow = b.v_flow
    over_arcs_t = ("p", "source", "sink", "t")
    lhs: dict = {}
    into_node = _arcs_at_node(d.process_source_sink, "sink")
    if into_node.height > 0:
        lhs["sink_flow"] = -Sum(Where(v_flow * d.p_unitsize, into_node) * d.p_step_duration, over=over_arcs_t)
    from_node_eff = (_arcs_at_node(d.process_source_sink_eff, "source")
                     if _has_rows(d.process_source_sink_eff) else None)
    if from_node_eff is not None and from_node_eff.height > 0 and d.p_slope is not None:
        lhs["source_eff"] = Sum(Where(v_flow * d.p_unitsize * d.p_slope, from_node_eff) * d.p_step_duration,
                                over=over_arcs_t)
        if b.has_minload_eff and d.p_section is not None:
            section_idx = from_node_eff.join(d.process_min_load_eff, on="p", how="inner")
            if section_idx.height > 0:
                for key, has_online, v_online in (("source_section_lin", b.has_online_lin, b.v_online_linear),
                                                  ("source_section_int", b.has_online_int, b.v_online_integer)):
                    if has_online:
                        lhs[key] = Sum(Where(Where(v_online, d.process_min_load_eff) * d.p_section * d.p_unitsize,
                                             section_idx) * d.p_step_duration,
                                       over=over_arcs_t)
    from_node_noEff = (_arcs_at_node(d.process_source_sink_noEff, "source")
                       if _has_rows(d.process_source_sink_noEff) else None)
    if from_node_noEff is not None and from_node_noEff.height > 0:
        lhs["source_noEff"] = Sum(Where(v_flow * d.p_unitsize, from_node_noEff) * d.p_step_duration,
                                  over=over_arcs_t)
    if not lhs:
        return
    b.m.add_cstr(
        "node_storage_usage_fix_le",
        over      = target.select("n", "d", "t").unique(),
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _eq_block_coupling(b: _Build) -> None:
    """Storage nodes balanced per coarse block (nodeStateBlock)."""
    if not (b.has_nsb and b.has_storage):
        return
    m, d = b.m, b.d
    nsb = d.nodeStateBlock
    v_state = b.v_state

    # ─── EQ. ─── stateConstantWithinBlock_eq ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − v_state[n,d,t_previous] = 0
    #   ∀ n ∈ nodeStateBlock, (d,t,t_previous) ∈ dtttdt_block_interior
    # Keeps the state of a nodeStateBlock node constant within each coarse
    # block.  Interior steps never cross a period boundary.
    if _has_rows(d.dtttdt_block_interior):
        interior = d.dtttdt_block_interior
        m.add_cstr(
            "stateConstantWithinBlock_eq",
            over      = nsb.join(interior.select("d", "t"), how="cross"),
            sense     = "==",
            lhs_terms = {"state":     Where(v_state, nsb),
                         "state_lag": -Where(Lag(v_state, interior, "t", "t_previous"), nsb)},
            rhs_terms = {},
        )

    if not (_has_rows(d.period_block) and _has_rows(d.period_block_succ) and _has_rows(d.period_block_time)):
        return

    # ─── EQ. ─── nodeBalanceBlock_eq ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # state_change[n,d,b]·state_unitsize[n]
    #   − Σ_{(p,s,n)} v_flow[p,s,n,d,t]·unitsize[p]·w_sink
    #   + Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff} v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t]·w_source
    #   + Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·w_source
    #   + Σ_{t ∈ b} v_state[n,d,t]·self_discharge[n]·state_unitsize[n]·Δt
    #   − Σ_{t ∈ b} (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   = Σ_{t ∈ b} inflow[n,d,t]
    #   ∀ n ∈ nodeStateBlock, (d,b) ∈ period_block;   b is named by its first step b_first, t ∈ b by period_block_time
    # Energy balance per coarse block (MWh) for the nodes nodeBalance_eq skips.
    # state_change, non-rp nodes (bind_intraperiod_blocks):
    #   Σ_{(d,b,b_next) ∈ period_block_succ} v_state[n,d,b_next] − v_state[n,d,b];
    #   the blocks of each representative period form one cycle.
    # state_change, blended-weights (rp) nodes: v_state[n,d,b] − v_state[n,d,b_prev], where the first block of a
    #   representative day starts from v_state_rp_start instead of a cyclic predecessor.  This is the fine rp
    #   state change lifted to blocks, so the rp inter-period chain keeps binding v_state_rp_start.
    # Flow weights: with per-arc block data (arc_{sink,source}_block_dt), w = p_arc_{sink,source}_weight, the step
    #   duration of the arc's block on that side; otherwise w = Δt over the fine steps of the block.
    # v_flow of an arc with both ends on a coarse block enters only at the coarse step; its values at the other
    #   fine steps are not tied to it.
    succ = d.period_block_succ.select("d", "b_first", "b_next")
    state_at_block = _var_view(v_state, "__b_first", {"t": "b_first"})
    if b.has_rp:
        rp_nodes = nsb.join(d.nodeState_rp.select("n"), on="n", how="inner")
        nonrp_nodes = nsb.join(d.nodeState_rp.select("n"), on="n", how="anti")
    else:
        rp_nodes = nsb.clear()
        nonrp_nodes = nsb

    lhs: dict = {}
    state_at_next = _var_view(v_state, "__b_next", {"t": "b_next"})
    lhs["state_change"] = (Sum(Where(Where(state_at_next, succ), nonrp_nodes), over=("b_next",))
                           - Sum(Where(Where(state_at_block, succ), nonrp_nodes), over=("b_next",))
                           ) * d.p_state_unitsize
    if rp_nodes.height > 0:
        # Predecessor of block b: the b_first whose b_next is b; first blocks
        # of a representative day take v_state_rp_start instead.
        rp_first = d.rp_block_first.select("d", pl.col("t").alias("b_first"))
        pred_interior = (succ.select("d", pl.col("b_next").alias("b_first"), pl.col("b_first").alias("b_prev"))
                         .join(rp_first, on=["d", "b_first"], how="anti"))
        state_at_prev = _var_view(v_state, "__b_prev", {"t": "b_prev"})
        prev = Sum(Where(Where(state_at_prev, pred_interior), rp_nodes), over=("b_prev",))
        rp_start = _var_view(b.v_state_rp_start, "__b_first", {"t": "b_first"})
        start = Where(Where(rp_start, rp_first), rp_nodes)
        now = Where(state_at_block, rp_nodes.join(d.period_block, how="cross").select("n", "d", "b_first"))
        lhs["state_change_rp"] = (now - prev - start) * d.p_state_unitsize

    v_flow = b.v_flow
    block_t = d.period_block_time
    over_arcs_t = ("p", "source", "sink", "t")
    if _has_rows(d.arc_sink_block_dt) and d.p_arc_sink_weight is not None:
        into_node = d.arc_sink_block_dt.drop("weight").pipe(rename_to_axis, {"sink": "n"})
        lhs["sink_flow_block"] = -Sum(Where(v_flow * d.p_unitsize * d.p_arc_sink_weight, into_node),
                                      over=over_arcs_t)
    elif _has_rows(d.flow_to_n):
        into_node = d.flow_to_n.filter(pl.col("n").is_in(nsb["n"]))
        if into_node.height > 0:
            lhs["sink_flow_block"] = -Sum(Where(Where(v_flow * d.p_unitsize, into_node) * d.p_step_duration, block_t),
                                          over=over_arcs_t)

    has_source_weights = _has_rows(d.arc_source_block_dt) and d.p_arc_source_weight is not None
    for key, from_node, with_slope in (("source_eff_block", d.flow_from_nodeBalance_eff, True),
                                       ("source_noEff_block", d.flow_from_nodeBalance_noEff, False)):
        if not _has_rows(from_node):
            continue
        flow = v_flow * d.p_unitsize * d.p_slope if with_slope else v_flow * d.p_unitsize
        if has_source_weights:
            idx = (d.arc_source_block_dt.drop("weight")
                   .join(from_node.select("p", "source", "sink"), on=["p", "source", "sink"], how="inner")
                   .pipe(rename_to_axis, {"source": "n"}))
            if idx.height > 0:
                lhs[key] = Sum(Where(flow * d.p_arc_source_weight, idx), over=over_arcs_t)
        else:
            idx = from_node.filter(pl.col("n").is_in(nsb["n"]))
            if idx.height > 0:
                lhs[key] = Sum(Where(Where(flow, idx) * d.p_step_duration, block_t), over=over_arcs_t)

    if d.p_state_self_discharge is not None:
        lhs["self_discharge_block"] = Sum(
            Where(Where(v_state * d.p_state_self_discharge * d.p_state_unitsize, nsb) * d.p_step_duration, block_t),
            over=("t",))
    lhs["slack_up_block"] = -Sum(Where(Where(b.vq_state_up, nsb) * d.p_step_duration, block_t), over=("t",))
    lhs["slack_down_block"] = Sum(Where(Where(b.vq_state_down, nsb) * d.p_step_duration, block_t), over=("t",))

    inflow_block = (d.p_inflow.frame
                    .filter(pl.col("n").is_in(nsb["n"]))
                    .join(block_t, on=["d", "t"], how="inner")
                    .group_by(["n", "d", "b_first"])
                    .agg(pl.col("value").sum())
                    .select("n", "d", "b_first", "value"))
    rhs = {"inflow_block": Param(("n", "d", "b_first"), inflow_block)} if inflow_block.height > 0 else {}
    m.add_cstr(
        "nodeBalanceBlock_eq",
        over      = nsb.join(d.period_block, how="cross").select("n", "d", "b_first"),
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _eq_max_flow(b: _Build) -> None:
    """Capacity limits on process flows."""
    if not b.has_proc:
        return
    m, d = b.m, b.d
    pss_dt = b.pss_dt
    invest_mult = (_invest_capacity_multiplier(pss_dt, b.flow_cap_mult, b.invest_procs)
                   if b.invest_procs is not None else None)

    def _scaled(term):
        return term * invest_mult if invest_mult is not None else term

    # ─── EQ. ─── maxFlow ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] + Σ_r v_reserve_up[p,r,k,d,t]
    #   − M·Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + M·Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    #   ≤ M · existing[p,d] / unitsize[p]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,s,k) ∉ flow_bound_arcs
    #   M = capacity_max_coeff[p,s,k] · availability[p,d,t]
    # Caps every flow by the built capacity of its process (units of unitsize).
    # On indirect-unit input arcs M = input_share_max · full-load input ·
    # availability; uncapped arcs (conversion_flow_coeff = 0) have M = 1 and
    # RHS = p_flow_upper.  Input arcs of units without invest/divest are
    # bounded on v_flow instead (flow_bound_arcs).  The reserve term covers
    # upward reserve held at the sink.
    lhs: dict = {"flow": b.v_flow}
    if b.reserve_up_to_sink is not None:
        lhs["reserve_up"] = b.reserve_up_to_sink
    if b.divest_in_dispatch_p is not None:
        lhs["divest"] = _scaled(b.divest_in_dispatch_p)
    if b.invest_in_dispatch_p is not None:
        lhs["invest_neg"] = -_scaled(b.invest_in_dispatch_p)
    over = (pss_dt.join(b.flow_bound_arcs, on=["p", "source", "sink"], how="anti")
            if b.flow_bound_arcs is not None else pss_dt)
    if over.height > 0:
        m.add_cstr(
            "maxFlow",
            over      = over,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": b.flow_upper_rhs},
        )

    # ─── EQ. ─── maxFlow_negCap ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # (maxFlow LHS)  ≥  M · existing[p,d] / unitsize[p]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,d) ∈ pd_neg_cap
    # Negative-capacity processes (unitsize < 0 and existing < 0): dividing
    # v_flow·unitsize ≤ existing by the negative unitsize flips the sense, so
    # the capacity becomes a minimum output.  maxFlow keeps the ≤ side.
    pd_neg_cap = getattr(d, "pd_neg_cap", None)
    if _has_rows(pd_neg_cap):
        neg_over = pss_dt.join(pd_neg_cap, on=("p", "d"), how="inner")
        if neg_over.height > 0:
            m.add_cstr(
                "maxFlow_negCap",
                over      = neg_over,
                sense     = ">=",
                lhs_terms = lhs,
                rhs_terms = {"upper": b.flow_upper_rhs},
            )

    # ─── EQ. ─── maxFlow_back ────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow_back[p,s,k,d,t] − M·Σ v_invest_p[p,d′] + M·Σ v_divest_p[p,d′]
    #   ≤ M · existing[p,d] / unitsize[p]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,s,k) ∈ twoway1var_arcs
    # Caps the reverse flow of two-way one-variable arcs by the same built
    # capacity, so |v_flow − v_flow_back| ≤ capacity.  No reserve term.
    if b.v_flow_back is not None and b.twoway1var_arcs is not None:
        back_over = pss_dt.join(b.twoway1var_arcs, on=("p", "source", "sink"), how="inner")
        if back_over.height > 0:
            back_lhs: dict = {"flow_back": b.v_flow_back}
            back_lhs.update({k: lhs[k] for k in ("invest_neg", "divest") if k in lhs})
            m.add_cstr(
                "maxFlow_back",
                over      = back_over,
                sense     = "<=",
                lhs_terms = back_lhs,
                rhs_terms = {"upper": b.flow_upper_rhs},
            )


def _eq_max_output_sum(b: _Build) -> None:
    """Output-sum capacity limit of indirect (multi-flow) units."""
    if not b.has_proc or b.pss_dt.height == 0:
        return
    m, d = b.m, b.d
    flow_upper_existing = getattr(d, "p_flow_upper_existing", None)
    if flow_upper_existing is None:
        return
    arcs = _indirect_sum_arcs(d)
    if arcs is None:
        return

    # ─── EQ. ─── maxOutputSum / maxOutputSum_negCap ──────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_k v_flow[p,p,k,d,t] + Σ_k Σ_r v_reserve_up[p,r,k,d,t]
    #   − av·Σ v_invest_p[p,d′] + av·Σ v_divest_p[p,d′]
    #   ≤ av · existing[p,d] / unitsize[p]
    #   ∀ p ∈ process_indirect with an output arc k, (d,t);   av = availability[p,d,t]
    # The capacity of an indirect (multi-flow) unit is the maximum sum of its
    # outputs; maxFlow caps each output arc alone.  capacity_max_coeff is not
    # applied.  Negative-capacity (p,d) rows get only the ≥ twin
    # maxOutputSum_negCap: with both, a k-output unit would need Σ out = c
    # while maxFlow pins every arc to c.
    # existing[p,d]/unitsize[p] is the value maxFlow's output arcs of p carry
    # (all equal; max collapses them); a unit without an existing row gets 0.
    # Availability defaults to 1.0 for processes without availability data.
    over = (b.pss_dt.join(arcs, on=["p", "source", "sink"], how="semi")
            .select("p", "d", "t")
            .unique(maintain_order=True))
    if over.height == 0:
        return

    lhs: dict = {"flow": Sum(Where(b.v_flow, arcs), over=("source", "sink"))}
    if b.reserve_up_sink_parts is not None:
        reserve_at_sink, up_sink = b.reserve_up_sink_parts
        up_sink = up_sink.join(arcs.select("p", "sink").unique(), on=["p", "sink"], how="semi")
        if up_sink.height > 0:
            lhs["reserve_up"] = Sum(Where(reserve_at_sink, up_sink), over=("r", "ud", "sink"))
    avail = _availability_factor(d, over) if d.p_process_availability is not None else None
    if b.invest_in_dispatch_p is not None:
        lhs["invest_neg"] = -(b.invest_in_dispatch_p * avail if avail is not None
                              else b.invest_in_dispatch_p)
    if b.divest_in_dispatch_p is not None:
        lhs["divest"] = (b.divest_in_dispatch_p * avail if avail is not None
                         else b.divest_in_dispatch_p)

    existing = (flow_upper_existing.frame
                .join(arcs, on=["p", "source", "sink"], how="semi")
                .group_by("p", "d")
                .agg(pl.col("value").max().alias("__ex")))
    rhs_lf = (over.lazy()
              .join(existing.lazy(), on=["p", "d"], how="left")
              .with_columns(pl.col("__ex").fill_null(0.0)))
    if d.p_process_availability is not None:
        avail_lf = promote_param_to_dt(d.p_process_availability, d.dt)
        rhs_lf = (rhs_lf
                  .join(avail_lf, on=["p", "d", "t"], how="left", suffix="__a")
                  .with_columns(pl.col("value").fill_null(1.0).alias("__av")))
    else:
        rhs_lf = rhs_lf.with_columns(pl.lit(1.0).alias("__av"))
    rhs = Param(("p", "d", "t"),
                rhs_lf.select("p", "d", "t", value=pl.col("__ex") * pl.col("__av")).collect())

    pd_neg_cap = getattr(d, "pd_neg_cap", None)
    if _has_rows(pd_neg_cap):
        neg_over = over.join(pd_neg_cap.select("p", "d"), on=["p", "d"], how="semi")
        pos_over = over.join(pd_neg_cap.select("p", "d"), on=["p", "d"], how="anti")
    else:
        neg_over = over.head(0)
        pos_over = over
    if pos_over.height > 0:
        m.add_cstr(
            "maxOutputSum",
            over      = pos_over,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": rhs},
        )
    if neg_over.height > 0:
        m.add_cstr(
            "maxOutputSum_negCap",
            over      = neg_over,
            sense     = ">=",
            lhs_terms = lhs,
            rhs_terms = {"upper": rhs},
        )


def _eq_ramp(b: _Build) -> None:
    """Ramp limits of process flows."""
    if not b.has_ramp:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_flow_prev = Lag(v_flow, d.dtttdt, "t", "t_previous")

    # ─── EQ. ─── ramp_{sink,source}_{up,down}_constraint ─────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # step_duration[d,t] · (v_flow[p,s,k,d,t] − v_flow[p,s,k,d,t_prev])
    #   ≤ +R · (existing_count[p,d] + Σ_{d′} v_invest_p[p,d′] − Σ_{d′} v_divest_p[p,d′]) + v_startup[p,d,t]   (up)
    #   ≥ −R · (existing_count[p,d] + Σ_{d′} v_invest_p[p,d′] − Σ_{d′} v_divest_p[p,d′]) − v_shutdown[p,d,t]  (down)
    #   R = ramp_speed_{dir}_{side}[p,s,k] · 60 · step_duration[d,t] · c[p,s,k,d,t]
    #   ∀ (p,s,k) ∈ process_source_sink_ramp_limit_{side}_{dir}, (d,t) ∈ dt;   t_prev from dtttdt
    # Limits the change of a flow between steps by its ramp speed (share of capacity per minute).
    # c scales the limit as the arc's capacity is scaled: capacity_max_coeff[p,k] on the sink side; on the
    # source side input_share_max[p,s], or on an indirect unit's input arc its input limit per unit of
    # capacity (p_indirect_input_cap = input_share_max · full-load input).  v_flow is in units of unitsize,
    # so existing_count = existing / unitsize.  v_startup / v_shutdown are the linear and integer UC
    # variables where the process has them.  Arcs with conversion_flow_coeff = 0 get no row.
    for side, dir_, idx_set, ramp_speed in (
        ("sink",   "up",   d.process_source_sink_ramp_limit_sink_up,     d.p_ramp_speed_up_sink),
        ("sink",   "down", d.process_source_sink_ramp_limit_sink_down,   d.p_ramp_speed_down_sink),
        ("source", "up",   d.process_source_sink_ramp_limit_source_up,   d.p_ramp_speed_up_source),
        ("source", "down", d.process_source_sink_ramp_limit_source_down, d.p_ramp_speed_down_source),
    ):
        if not _has_rows(idx_set) or ramp_speed is None:
            continue
        idx_set = _drop_uncapped(d, idx_set)
        if idx_set.height == 0:
            continue
        sign = 1.0 if dir_ == "up" else -1.0
        flow_diff = (Where(v_flow, idx_set) - Where(v_flow_prev, idx_set)) * d.p_step_duration
        rate = ramp_speed * 60.0 * d.p_step_duration
        if side == "sink":
            if d.p_process_sink_max_capacity_coef is not None:
                rate = rate * _coef_factor(idx_set, d.p_process_sink_max_capacity_coef, ["p", "sink"])
        else:
            src_factor = _source_ramp_factor(d, idx_set)
            if src_factor is not None:
                rate = rate * src_factor
        rhs: dict = {"limit": (rate * d.p_process_existing_count) * sign}
        if b.has_invest_p:
            rhs["invest"] = (b.invest_in_dispatch_p * rate) * sign
        if b.has_divest_p:
            rhs["divest"] = (b.divest_in_dispatch_p * rate) * (-sign)
        if dir_ == "up":
            if b.has_online_lin:
                rhs["startup_lin"] = Where(b.v_startup_linear, idx_set)
            if b.has_online_int:
                rhs["startup_int"] = Where(b.v_startup_integer, idx_set)
        else:
            if b.has_online_lin:
                rhs["shutdown_lin"] = -Where(b.v_shutdown_linear, idx_set)
            if b.has_online_int:
                rhs["shutdown_int"] = -Where(b.v_shutdown_integer, idx_set)
        m.add_cstr(
            f"ramp_{side}_{dir_}_constraint",
            over      = idx_set.join(d.dt, how="cross"),
            sense     = "<=" if dir_ == "up" else ">=",
            lhs_terms = {"flow_diff": flow_diff},
            rhs_terms = rhs,
        )


def _source_ramp_factor(d, idx_set: "pl.DataFrame") -> "Param | None":
    """Source-side ramp capacity factor c over ``idx_set × dt``.

    Indirect-unit input arcs (rows of ``p_indirect_input_cap``) take that input limit per unit of
    capacity; every other arc takes ``input_share_max`` keyed ``(p, source)`` (default 1).  ``None``
    when neither applies (factor 1 everywhere, the RHS is left unmultiplied).
    """
    keys = ["p", "source", "sink"]
    share = getattr(d, "p_process_source_input_share_max", None)
    in_cap = getattr(d, "p_indirect_input_cap", None)
    cap_rows = None
    if in_cap is not None and in_cap.frame.height > 0:
        cap_rows = (_utf8_keys(in_cap.frame.lazy(), keys)
                    .join(_utf8_keys(idx_set.lazy(), keys).select([f"_{k}" for k in keys]),
                          on=[f"_{k}" for k in keys], how="semi")
                    .select([f"_{k}" for k in keys] + ["d", "t", pl.col("value").alias("__cap")])
                    .collect())
        if cap_rows.height == 0:
            cap_rows = None
    if cap_rows is None:
        if share is None:
            return None
        return _coef_factor(idx_set, share, ["p", "source"])
    base = _utf8_keys(idx_set.select(keys).unique().join(d.dt.select("d", "t"), how="cross").lazy(), keys)
    if share is not None:
        base = base.join(share.frame.lazy()
                         .select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                 pl.col("source").cast(pl.Utf8).alias("_source"),
                                 pl.col("value").alias("__share")),
                         on=["_p", "_source"], how="left")
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias("__share"))
    frame = (base
             .join(cap_rows.lazy(), on=[f"_{k}" for k in keys] + ["d", "t"], how="left")
             .select(*keys, "d", "t", value=pl.coalesce(pl.col("__cap"), pl.col("__share"), pl.lit(1.0)))
             .collect())
    return Param(("p", "source", "sink", "d", "t"), frame)


def _eq_invest_max_units(b: _Build) -> None:
    """Units invested or divested per period."""
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n):
        return
    m, d = b.m, b.d

    # ─── EQ. ─── max{Invest,Divest}_var_bound[_n] ────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] ≤ max_units[p,d]   ∀ (p,d) ∈ pd_invest_set     (maxInvest_var_bound)
    # v_divest_p[p,d] ≤ max_units[p,d]   ∀ (p,d) ∈ pd_divest_set     (maxDivest_var_bound)
    # v_invest_n[n,d] ≤ max_units[n,d]   ∀ (n,d) ∈ nd_invest_set     (maxInvest_var_bound_n)
    # v_divest_n[n,d] ≤ max_units[n,d]   ∀ (n,d) ∈ nd_divest_set     (maxDivest_var_bound_n)
    # Caps the units invested or divested in one period (max_units = p_entity_max_units).
    # Emitted as rows, not as variable bounds.
    if b.has_invest_p or b.has_divest_p:
        max_units_p = Param(("p", "d"), _entity_rows(d.p_entity_max_units.frame, "e", d.process_source_sink, "p"))
    if b.has_invest_p:
        m.add_cstr(
            "maxInvest_var_bound",
            over      = d.pd_invest_set,
            sense     = "<=",
            lhs_terms = {"invest":    b.v_invest_p},
            rhs_terms = {"max_units": max_units_p},
        )
    if b.has_divest_p:
        m.add_cstr(
            "maxDivest_var_bound",
            over      = d.pd_divest_set,
            sense     = "<=",
            lhs_terms = {"divest":    b.v_divest_p},
            rhs_terms = {"max_units": max_units_p},
        )
    if b.has_invest_n or b.has_divest_n:
        max_units_n = Param(("n", "d"), _entity_rows(d.p_entity_max_units.frame, "e", d.nodeState, "n"))
    if b.has_invest_n:
        m.add_cstr(
            "maxInvest_var_bound_n",
            over      = d.nd_invest_set,
            sense     = "<=",
            lhs_terms = {"invest":    b.v_invest_n},
            rhs_terms = {"max_units": max_units_n},
        )
    if b.has_divest_n:
        m.add_cstr(
            "maxDivest_var_bound_n",
            over      = d.nd_divest_set,
            sense     = "<=",
            lhs_terms = {"divest":    b.v_divest_n},
            rhs_terms = {"max_units": max_units_n},
        )


def _eq_invest_period_caps(b: _Build) -> None:
    """Capacity invested or divested per period, for entities with a per-period limit."""
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n):
        return
    m, d = b.m, b.d

    # ─── EQ. ─── max{Invest,Divest}_entity_period_{p,n} ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] · unitsize[p]        ≤ invest_max_period[p,d]   ∀ (p,d) ∈ ed_invest_period_set ∩ pd_invest_set
    # v_invest_n[n,d] · state_unitsize[n]  ≤ invest_max_period[n,d]   ∀ (n,d) ∈ ed_invest_period_set ∩ nd_invest_set
    # v_divest_p[p,d] · unitsize[p]        ≤ divest_max_period[p,d]   ∀ (p,d) ∈ ed_divest_period_set ∩ pd_divest_set
    # v_divest_n[n,d] · state_unitsize[n]  ≤ divest_max_period[n,d]   ∀ (n,d) ∈ ed_divest_period_set ∩ nd_divest_set
    # Caps the capacity built or retired in one period for entities with a per-period limit
    # (e.g. invest_method = invest_period_total).  The cap is raw capacity with unitsize on the LHS, as in
    # the other invest caps, so the row dual is in objective units per unit of capacity.
    for name, active, period_set, var_set, cap, var, axis, kind in (
        ("maxInvest_entity_period_p", b.has_invest_p, d.ed_invest_period_set, d.pd_invest_set,
         d.ed_invest_max_period, b.v_invest_p, "p", "invest"),
        ("maxInvest_entity_period_n", b.has_invest_n, d.ed_invest_period_set, d.nd_invest_set,
         d.ed_invest_max_period, b.v_invest_n, "n", "invest"),
        ("maxDivest_entity_period_p", b.has_divest_p, d.ed_divest_period_set, d.pd_divest_set,
         d.ed_divest_max_period, b.v_divest_p, "p", "divest"),
        ("maxDivest_entity_period_n", b.has_divest_n, d.ed_divest_period_set, d.nd_divest_set,
         d.ed_divest_max_period, b.v_divest_n, "n", "divest"),
    ):
        if not active or not _has_rows(period_set):
            continue
        over = period_set.pipe(rename_to_axis, {"e": axis}).join(var_set, on=[axis, "d"], how="inner")
        if over.height == 0 or cap is None:
            continue
        cap_param = Param((axis, "d"), cap.frame.pipe(rename_to_axis, {"e": axis}).select(axis, "d", "value"))
        unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
        m.add_cstr(
            name,
            over      = over,
            sense     = "<=",
            lhs_terms = {kind:  var * unitsize},
            rhs_terms = {"cap": cap_param},
        )


def _eq_invest_total_caps(b: _Build) -> None:
    """Capacity invested or divested over the horizon, for entities with a total limit."""
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n):
        return
    m, d = b.m, b.d
    d_leaf = getattr(d, "d_leaf", None)

    # Capacity realized by earlier solves of a solve chain, per entity.  Invest: the largest previously
    # invested capacity over the periods.  Divest: p_entity_divested, except on the first solve of a chain
    # (p_nested_solve_first is None for a single solve, True on the first solve, False after it).
    prior_invest: dict[str, float] = {}
    if d.p_entity_previously_invested_capacity is not None:
        prior_invest = dict(d.p_entity_previously_invested_capacity.frame
                            .group_by("e").agg(pl.col("value").max())
                            .rows())
    prior_divest: dict[str, float] = {}
    if d.p_entity_divested is not None and getattr(d, "p_nested_solve_first", None) is False:
        prior_divest = dict(d.p_entity_divested.frame.rows())

    def _minus_prior(cap_frame: pl.DataFrame, key_col: str, prior: dict[str, float]) -> pl.DataFrame:
        """``cap_frame`` with ``value -= prior[key]``, clipped at 0."""
        if not prior:
            return cap_frame
        return (cap_frame
                .with_columns(value=pl.col("value") - pl.col(key_col).map_elements(
                    lambda k: prior.get(k, 0.0), return_dtype=pl.Float64))
                .with_columns(value=pl.when(pl.col("value") < 0).then(0.0).otherwise(pl.col("value"))))

    # ─── EQ. ─── max{Invest,Divest}_entity_total[_path][_n] ──────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_d v_invest_p[p,d] · unitsize[p]  ≤  max(0, invest_max_total[p] − prior_invest[p])   ∀ p ∈ e_invest_total
    # Σ_d v_divest_p[p,d] · unitsize[p]  ≤  max(0, divest_max_total[p] − prior_divest[p])   ∀ p ∈ e_divest_total
    #   _n: the node analogues with v_invest_n / v_divest_n and state_unitsize[n]
    #   _path: one row per (entity, leaf), Σ_d over the periods of the leaf
    # Caps the capacity built or retired over the horizon.  With one scenario leaf (deterministic) the row
    # is the plain total; with several (recourse) it becomes the _path row, the per-entity cap applying to
    # each scenario path.  prior_invest[e] = max_d p_entity_previously_invested_capacity[e,d] (one row per
    # entity, so the tightest per-period cap of a non-decreasing schedule).  p_entity_invested is not
    # subtracted (see specs/github_issues.md, refactor bug 2).
    for active, total_set, max_total, members, axis, var, legacy_name, path_name, label, prior in (
        (b.has_invest_p, d.e_invest_total, d.e_invest_max_total, d.process_source_sink, "p", b.v_invest_p,
         "maxInvest_entity_total", "maxInvest_entity_total_path", "invest_total", prior_invest),
        (b.has_divest_p, d.e_divest_total, d.e_divest_max_total, d.process_source_sink, "p", b.v_divest_p,
         "maxDivest_entity_total", "maxDivest_entity_total_path", "divest_total", prior_divest),
        (b.has_invest_n, d.e_invest_total, d.e_invest_max_total, d.nodeState, "n", b.v_invest_n,
         "maxInvest_entity_total_n", "maxInvest_entity_total_path_n", "invest_total", prior_invest),
        (b.has_divest_n, d.e_divest_total, d.e_divest_max_total, d.nodeState, "n", b.v_divest_n,
         "maxDivest_entity_total_n", "maxDivest_entity_total_path_n", "divest_total", prior_divest),
    ):
        if not active or not _has_rows(total_set) or max_total is None:
            continue
        over_e = _entity_rows(total_set, "e", members, axis)
        if over_e.height == 0:
            continue
        cap = Param((axis,), _minus_prior(max_total.frame.pipe(rename_to_axis, {"e": axis}), axis, prior))
        unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
        # (d, leaf) pairs of the scenario tree, restricted to the periods var exists in.
        leaf_periods = None
        if d_leaf is not None and getattr(d_leaf, "height", 0) > 0:
            d_dtype = var.frame.schema["d"]
            leaf_periods = (d_leaf if d_leaf.schema["d"] == d_dtype
                            else d_leaf.with_columns(pl.col("d").cast(d_dtype, strict=False)))
            leaf_periods = leaf_periods.join(var.frame.select("d").unique(), on="d", how="semi")
        if leaf_periods is None or leaf_periods.height == 0 or leaf_periods["leaf"].n_unique() <= 1:
            name, over, mask = legacy_name, over_e, over_e
        else:
            name = path_name
            mask = over_e.join(leaf_periods, how="cross")
            over = over_e.join(leaf_periods.select("leaf").unique().sort("leaf"), how="cross")
        m.add_cstr(
            name,
            over      = over,
            sense     = "<=",
            lhs_terms = {label: Sum(Where(var * unitsize, mask), over=("d",))},
            rhs_terms = {"cap": cap},
        )


def _eq_conversion(b: _Build) -> None:
    """Conversion of indirect (multi-flow) units."""
    if not b.has_indirect:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    # ─── EQ. ─── conversion_indirect ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{s ∈ in(p)} v_flow[p,s,p,d,t] · unitsize[p] · conv_in[p,s]
    #   + Σ_s Σ_{(t_s,t,δ) ∈ delay} v_flow[p,s,p,d,t_s] · unitsize[p] · delay_weight[p,δ] · conv_in[p,s]
    #   = Σ_{k ∈ out(p)} v_flow[p,p,k,d,t] · unitsize[p] · slope[p,d,t] · conv_out[p,k]
    #     + v_online[p,d,t] · section[p,d,t] · unitsize[p]
    #   ∀ (p,d,t) ∈ process_indirect_dt
    # Energy balance of an indirect (multi-flow) unit: weighted input equals weighted output times the slope.
    # conv_in / conv_out = source / sink conversion_flow_coeff (default 1); unequal sink coefficients express
    # e.g. the heat/electricity trade-off of an extraction CHP.  The first input sum covers undelayed units;
    # delayed units take their inputs at the earlier source-side steps t_s (dtt__delay_duration,
    # _delay.delayed_input_expr), t being the sink-side time.
    # Section term (indirect min_load_efficiency units, linear and integer v_online): the input of an
    # indirect unit is its own v_flow, so this is the only place the section fuel enters; node balance,
    # commodity cost, CO2 and the reported inputs all read that v_flow.  It carries no conversion_flow_coeff;
    # for a delayed unit it sits on the output side at time t.  Rows without a v_online column lack the term.
    in_flows_undelayed = d.process_input_flows
    if _has_rows(getattr(d, "process_delayed", None)):
        in_flows_undelayed = in_flows_undelayed.join(d.process_delayed, on="p", how="anti")
    input_expr = v_flow * d.p_unitsize
    if d.p_process_source_conversion_flow_coeff is not None:
        input_expr = input_expr * d.p_process_source_conversion_flow_coeff
    output_expr = v_flow * d.p_unitsize * d.p_slope
    if d.p_process_sink_conversion_flow_coeff is not None:
        output_expr = output_expr * d.p_process_sink_conversion_flow_coeff

    lhs: dict = {"input": Sum(Where(input_expr, in_flows_undelayed), over=("source", "sink"))}
    delayed_term = _delay.delayed_input_expr(d, v_flow)
    if delayed_term is not None:
        lhs["input_delayed"] = delayed_term
    rhs: dict = {"output": Sum(Where(output_expr, d.process_output_flows), over=("source", "sink"))}
    if b.has_minload_eff and d.p_section is not None:
        ind_mle = d.process_min_load_eff.join(d.process_indirect.select("p").unique(), on="p", how="semi")
        if ind_mle.height > 0:
            if b.has_online_lin:
                rhs["section_lin"] = Where(b.v_online_linear, ind_mle) * d.p_section * d.p_unitsize
            if b.has_online_int:
                rhs["section_int"] = Where(b.v_online_integer, ind_mle) * d.p_section * d.p_unitsize
    m.add_cstr(
        "conversion_indirect",
        over      = b.process_indirect_dt,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _eq_min_input_share(b: _Build) -> None:
    """Minimum share of each input in the input of an indirect unit."""
    if not b.has_indirect:
        return
    m, d = b.m, b.d
    share = getattr(d, "p_process_source_input_share_min", None)
    pif = getattr(d, "process_input_flows", None)
    pss = getattr(d, "process_source_sink", None)
    if share is None or share.frame.height == 0 or not _has_rows(pif) or not _has_rows(pss):
        return

    # ─── EQ. ─── minInputShare ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # conv[p,s] · v_flow[p,s,p,d,t]  ≥  share_min[p,s] · Σ_{s′ ∈ in(p)} conv[p,s′] · v_flow[p,s′,p,d,t]
    #   ∀ (p,s) with share_min[p,s] > 0, p ∈ process_indirect with ≥ 2 inputs, (p,d,t) ∈ process_indirect_dt
    # While the unit runs, input s supplies at least the share input_share_min of its current input energy
    # (conv = source conversion_flow_coeff, default 1).  The row is homogeneous: with no input it reads
    # 0 ≥ 0, so it needs no online variable.  unitsize cancels (v_flow is per unitsize on every input of p).
    keys = ["p", "source", "sink"]
    # Input arcs with a v_flow column of indirect units with at least two inputs.
    in_arcs = (_utf8_keys(pif.lazy().select(keys), keys)
               .join(_utf8_keys(d.process_indirect.lazy().select("p"), ["p"]).select("_p").unique(),
                     on="_p", how="semi")
               .join(_utf8_keys(pss.lazy().select(keys), keys).select("_p", "_source", "_sink"),
                     on=["_p", "_source", "_sink"], how="semi"))
    n_in = in_arcs.group_by("_p").agg(pl.len().alias("__n"))
    in_arcs = in_arcs.join(n_in.filter(pl.col("__n") >= 2).select("_p"), on="_p", how="semi")
    share_lf = (share.frame.lazy()
                .with_columns(pl.col("p").cast(pl.Utf8).alias("_p"),
                              pl.col("source").cast(pl.Utf8).alias("_source"))
                .select("_p", "_source", pl.col("value").cast(pl.Float64)))
    rows = (in_arcs.join(share_lf, on=["_p", "_source"], how="inner")
            .filter(pl.col("value") > 0.0)
            .select(*keys, "value").collect())
    if rows.height == 0:
        return
    _validate_min_input_shares(rows)
    arcs_all = (in_arcs.join(_utf8_keys(rows.select("p").unique().lazy(), ["p"]).select("_p"),
                             on="_p", how="semi")
                .select(keys).collect())
    conv = getattr(d, "p_process_source_conversion_flow_coeff", None)
    v_flow = b.v_flow
    expr = v_flow
    if conv is not None:
        conv_c = _coef_factor(arcs_all, conv, ["p", "source"])
        neg = conv_c.frame.filter(pl.col("value") < 0.0)
        if neg.height > 0:
            raise FlexToolConfigError(
                "input_share_min is not defined for a unit with a negative input conversion_flow_coeff: "
                + ", ".join(f"{r['p']}<-{r['source']}" for r in neg.iter_rows(named=True)))
        expr = v_flow * conv_c
    over = rows.select(keys).join(d.dt, how="cross")
    pidt = getattr(d, "process_indirect_dt", None)
    if _has_rows(pidt):
        over = over.join(pidt.select("p", "d", "t"), on=["p", "d", "t"], how="semi")
    if over.height == 0:
        return
    share_p = Param(("p", "source"), rows.select("p", "source", "value"))
    m.add_cstr(
        "minInputShare",
        over      = over,
        sense     = ">=",
        lhs_terms = {"own":   Where(expr, rows.select(keys))},
        rhs_terms = {"total": Sum(Where(expr, arcs_all), over=("source",)) * share_p},
    )


def _validate_min_input_shares(rows: pl.DataFrame) -> None:
    """Raise unless every ``input_share_min`` lies in [0, 1] and they sum to at most 1 per unit.

    The input data is validated once per scenario
    (``input_derivation._validators.validate_unit_input_shares``); this guards hand-built data.
    """
    bad = rows.filter(pl.col("value") > 1.0)
    sums = (rows.group_by("p").agg(pl.col("value").sum().alias("s"))
            .filter(pl.col("s") > 1.0 + 1e-9))
    if bad.height > 0 or sums.height > 0:
        raise FlexToolConfigError(
            "input_share_min must lie in [0, 1] and sum to at most 1 per unit; offending: "
            + ", ".join([f"{r['p']}<-{r['source']}: {r['value']}" for r in bad.iter_rows(named=True)]
                        + [f"{r['p']}: sum {r['s']}" for r in sums.iter_rows(named=True)]))


def _eq_co2_max_period(b: _Build) -> None:
    """CO2 emission cap per period."""
    if not b.has_co2_cap:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    # ─── EQ. ─── co2_max_period ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped, t} v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t]·co2_content[c]·w[d,t]
    #   + Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_noEff, t} v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w[d,t]
    #   ≤ co2_max_period[g,d]
    #   ∀ (g,d) ∈ group_d_co2_capped;   w = step_duration · timestep_weight / period_share
    # Caps the annualised CO2 emissions of a group in each period (tonnes).  Emissions come from the
    # flows out of the group's nodes that carry a CO2-emitting commodity c.  On eff processes the
    # source-side flow is v_flow · slope; noEff processes take no slope (applying it would over-count them).
    lhs: dict = {}
    if b.has_co2_cap_eff:
        lhs["emissions_eff"] = Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_co2_capped)
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
            over=("p", "source", "sink", "c", "t"),
        )
    if b.has_co2_cap_noEff:
        lhs["emissions_noEff"] = Sum(
            Where(v_flow * d.p_unitsize, d.flow_from_co2_capped_noEff)
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
            over=("p", "source", "sink", "c", "t"),
        )
    m.add_cstr(
        "co2_max_period",
        over      = d.group_d_co2_capped,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"cap": d.p_co2_max_period},
    )


def _eq_co2_max_total(b: _Build) -> None:
    """CO2 emission cap over all periods."""
    if not b.has_co2_cap_total:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    # ─── EQ. ─── co2_max_total ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_total, d,t} v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t]·co2_content[c]·w[d,t]
    #   + Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_total_noEff, d,t} v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w[d,t]
    #   ≤ co2_max_total[g]
    #   ∀ g ∈ group_co2_max_total;   w = step_duration · timestep_weight / period_share
    # Caps the CO2 emissions of a group over the whole horizon (tonnes); the emission terms are those of
    # co2_max_period, summed over the periods as well.  Each period contributes its annualised emissions,
    # not weighted by the years it represents (see specs/github_issues.md, refactor bug 13).
    lhs: dict = {}
    if b.has_co2_cap_total_eff:
        lhs["emissions_eff"] = Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_co2_capped_total)
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
            over=("p", "source", "sink", "c", "d", "t"),
        )
    if b.has_co2_cap_total_noEff:
        lhs["emissions_noEff"] = Sum(
            Where(v_flow * d.p_unitsize, d.flow_from_co2_capped_total_noEff)
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
            over=("p", "source", "sink", "c", "d", "t"),
        )
    m.add_cstr(
        "co2_max_total",
        over      = d.group_co2_max_total,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"cap": d.p_co2_max_total},
    )


def _eq_user_constraints(b: _Build) -> None:
    """User-defined constraints."""
    if not b.has_user_cstr:
        return
    m, d = b.m, b.d
    v_invest_p, v_invest_n = b.v_invest_p, b.v_invest_n
    lookback = d.edd_invest_lookback_set

    # ─── EQ. ─── process_constraint_{equal,less_than,greater_than} ───────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k) ∈ flow_constraint[cn]} v_flow[p,s,k,d,t]·unitsize[p]·flow_coef[p,s,k,cn]
    #   + Σ_n v_invest_n[n,d]·state_unitsize[n]·node_invested_coef[n,cn]
    #   + Σ_p v_invest_p[p,d]·unitsize[p]·process_invested_coef[p,cn]
    #   + Σ_n v_state[n,d,t]·state_unitsize[n]·node_state_coef[n,cn]
    #   + Σ_n existing[n,d]·node_prebuilt_coef[n,cn]
    #   + Σ_p existing_count[p,d]·unitsize[p]·process_prebuilt_coef[p,cn]
    #   + Σ_{(p,d′,d) ∈ edd_invest_lookback} v_invest_p[p,d′]·unitsize[p]·process_prebuilt_coef[p,cn]
    #   + Σ_{(n,d′,d) ∈ edd_invest_lookback} v_invest_n[n,d′]·state_unitsize[n]·node_prebuilt_coef[n,cn]
    #   = / ≤ / ≥  constant[cn]
    #   ∀ (cn,d,t) ∈ cdt_{eq,le,ge}
    # A linear constraint authored by the user over flows (MW), invested capacity, storage states (MWh) and
    # the capacity built before d: existing capacity plus the investments of earlier periods
    # (edd_invest_lookback holds d′ < d).  The invested-capacity terms count the investment made in d only.
    # Terms without data are absent; a constraint with no term at all gets no row.  The existing-capacity
    # terms are constants and enter as Param terms.  The constraint axis is cn (c is the commodity axis).
    # Dims: every variable term reduces to (cn, d, t), or (cn, d) for invest terms broadcast over t.
    var_terms: list = []
    if _has_rows(d.flow_constraint_idx):
        var_terms.append(Sum(
            Where(b.v_flow * d.p_unitsize, d.flow_constraint_idx) * d.p_flow_constraint_coef,
            over=("p", "source", "sink"),
        ))
    if b.has_invest_n and d.p_node_constraint_invested_capacity_coeff is not None:
        coef = d.p_node_constraint_invested_capacity_coeff
        var_terms.append(Sum(
            Where(v_invest_n * Param(("n",), d.p_state_unitsize.frame), coef.frame.select("n", "cn")) * coef,
            over=("n",),
        ))
    if b.has_invest_p and d.p_process_constraint_invested_capacity_coeff is not None:
        coef = d.p_process_constraint_invested_capacity_coeff
        var_terms.append(Sum(
            Where(v_invest_p * d.p_unitsize, coef.frame.select("p", "cn")) * coef,
            over=("p",),
        ))
    if b.has_storage and d.p_node_constraint_state_coeff is not None:
        coef = d.p_node_constraint_state_coeff
        var_terms.append(Sum(
            Where(b.v_state * Param(("n",), d.p_state_unitsize.frame), coef.frame.select("n", "cn")) * coef,
            over=("n",),
        ))

    const_terms: list = []
    node_pre = d.p_node_constraint_prebuilt_capacity_coeff
    proc_pre = d.p_process_constraint_prebuilt_capacity_coeff
    if node_pre is not None and d.p_state_existing_capacity is not None:
        node_existing = (d.p_state_existing_capacity.frame
                         .join(node_pre.frame.rename({"value": "coef"}), on="n", how="inner")
                         .with_columns(value=pl.col("value") * pl.col("coef"))
                         .group_by(["cn", "d"]).agg(pl.col("value").sum())
                         .select("cn", "d", "value"))
        if node_existing.height > 0:
            const_terms.append(Param(("cn", "d"), node_existing))
    if proc_pre is not None and d.p_process_existing_count is not None and d.p_unitsize is not None:
        proc_existing = (d.p_process_existing_count.frame.rename({"value": "ec"})
                         .join(d.p_unitsize.frame.rename({"value": "us"}), on="p", how="inner")
                         .with_columns(cap=pl.col("ec") * pl.col("us"))
                         .join(proc_pre.frame.rename({"value": "coef"}), on="p", how="inner")
                         .with_columns(value=pl.col("cap") * pl.col("coef"))
                         .group_by(["cn", "d"]).agg(pl.col("value").sum())
                         .select("cn", "d", "value"))
        if proc_existing.height > 0:
            const_terms.append(Param(("cn", "d"), proc_existing))

    if b.has_invest_p and proc_pre is not None and _has_rows(lookback):
        invested = _var_view(v_invest_p, "__at_back_uc", {"d": "d_invest"}, to_axis=True)
        earlier = (lookback.pipe(rename_to_axis, {"e": "p"})
                   .join(proc_pre.frame.select("p", "cn"), on="p", how="inner"))
        if earlier.height > 0:
            var_terms.append(Sum(
                Where(invested * d.p_unitsize, earlier) * proc_pre,
                over=("p", "d_invest"),
            ))
    if b.has_invest_n and node_pre is not None and _has_rows(lookback):
        invested = _var_view(v_invest_n, "__at_back_uc", {"d": "d_invest"}, to_axis=True)
        earlier = (lookback.pipe(rename_to_axis, {"e": "n"})
                   .join(node_pre.frame.select("n", "cn"), on="n", how="inner"))
        if earlier.height > 0:
            var_terms.append(Sum(
                Where(invested * Param(("n",), d.p_state_unitsize.frame), earlier) * node_pre,
                over=("n", "d_invest"),
            ))

    if not (var_terms or const_terms):
        return
    lhs: dict = {}
    if var_terms:
        lhs["contribution"] = _sum_terms(var_terms)
    for i, const in enumerate(const_terms):
        lhs[f"prebuilt_const_{i}"] = const
    for over, sense, name in (
        (d.cdt_eq, "==", "process_constraint_equal"),
        (d.cdt_le, "<=", "process_constraint_less_than"),
        (d.cdt_ge, ">=", "process_constraint_greater_than"),
    ):
        if not _has_rows(over):
            continue
        m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"constant": d.p_constraint_constant},
        )


def _eq_max_state(b: _Build) -> None:
    """Capacity limit of the storage state."""
    if not b.has_storage:
        return
    m, d = b.m, b.d

    # ─── EQ. ─── maxState ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] + Σ_{(n,d′,d) ∈ edd_divest_active} v_divest_n[n,d′]
    #   − Σ_{(n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′]  ≤  state_upper[n,d]
    #   ∀ (n,d,t) ∈ nodeState_dt
    # Caps the storage state by the built storage capacity.  state_upper = existing / state_unitsize, so
    # every term is in state units.
    lhs: dict = {"state": b.v_state}
    lhs.update(_node_capacity_change_terms(b, d.nodeState, ""))
    m.add_cstr(
        "maxState",
        over      = b.nodeState_dt,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"upper": d.p_state_upper},
    )


def _eq_state_profiles(b: _Build) -> None:
    """Profile limits of the storage state."""
    d = b.d
    profiles = [(idx, name, sense) for idx, name, sense in (
        (d.node_profile_upper, "profile_state_upper_limit", "<="),
        (d.node_profile_lower, "profile_state_lower_limit", ">="),
        (d.node_profile_fixed, "profile_state_fixed",       "=="),
    ) if _has_rows(idx)]
    if not (b.has_storage and profiles and d.p_profile_value is not None):
        return
    m = b.m

    # ─── EQ. ─── profile_state_{upper_limit,lower_limit,fixed} ───────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − profile[f,d,t]·av[n,d,t]·Σ_{(n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′]
    #   + profile[f,d,t]·av[n,d,t]·Σ_{(n,d′,d) ∈ edd_divest_active} v_divest_n[n,d′]
    #   ≤ / ≥ / =  profile[f,d,t] · av[n,d,t] · existing[n,d] / state_unitsize[n]
    #   ∀ (n,f) ∈ node_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds the storage state by a profile times the built storage capacity, in state units.
    # av = node availability, 1 for nodes without availability data.  The invest/divest terms are
    # restricted to the (n,f) pairs of the method's set.
    state_unitsize = d.p_state_unitsize.frame.rename({"value": "us"})
    existing_count = Param(("n", "d"), d.p_state_existing_capacity.frame.rename({"value": "cap"})
                           .join(state_unitsize, on="n", how="inner")
                           .with_columns(value=pl.col("cap") / pl.col("us"))
                           .select("n", "d", "value"))
    for idx, name, sense in profiles:
        over = idx.join(d.dt, how="cross")
        rhs = d.p_profile_value * existing_count
        avail = (_availability_factor(d, idx, axis="n", avail=d.p_node_availability)
                 if d.p_node_availability is not None else None)
        if avail is not None:
            rhs = rhs * avail
        lhs: dict = {"state": b.v_state}
        node_profile_pairs = idx.select("n", "f").unique()
        invested, divested = _node_in_service(b, d.nodeState, "_node_profile")
        if invested is not None:
            term = invested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["invest_neg"] = -Where(term, node_profile_pairs)
        if divested is not None:
            term = divested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["divest"] = Where(term, node_profile_pairs)
        m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"limit": rhs},
        )


def _eq_storage_start_binding(b: _Build) -> None:
    """Start state of fix_start storage nodes."""
    d = b.d
    if not (b.has_storage and b.is_solve_first
            and _has_rows(d.storage_fix_start)
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None):
        return
    m = b.m

    # ─── EQ. ─── storage_state_start_binding ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − state_start[n]·Σ_{(n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′]
    #   + state_start[n]·Σ_{(n,d′,d) ∈ edd_divest_active} v_divest_n[n,d′]
    #   = state_start[n] · existing[n,d] / state_unitsize[n]
    #   ∀ (n,d,t) ∈ nodeState_first_dt, n ∈ storage_fix_start, n ∉ bind_forward_only ∪ bind_within_solve
    # Pins the state at the first step to the share state_start of the built storage capacity (state
    # units).  First solve of a chain only: continuation solves start from the state handed over by the
    # previous solve (nodeBalance_eq).  bind_forward_only nodes get their start through nodeBalance_eq
    # (_fix_start_terms); bind_within_solve nodes are pinned by storage_state_start_binding_cyclic_period.
    nodes = d.storage_fix_start
    for excluded in (d.storage_bind_forward_only, d.storage_bind_within_solve):
        if _has_rows(excluded):
            nodes = nodes.join(excluded, on="n", how="anti")
    target = (d.p_state_start * d.p_state_existing_capacity) / d.p_state_unitsize
    over = d.nodeState_first_dt.join(nodes, on="n", how="inner")
    lhs: dict = {"state": b.v_state}
    invested, divested = _node_in_service(b, d.nodeState, "_start", coef=d.p_state_start)
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_start_binding",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": target},
    )


def _eq_storage_start_binding_cyclic(b: _Build) -> None:
    """Start state of fix_start storage nodes that cycle within the solve."""
    d = b.d
    if not (b.has_storage and b.is_solve_first
            and _has_rows(d.storage_fix_start)
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None
            and _has_rows(d.nodeState_last_dt)
            and _has_rows(d.storage_bind_within_solve)):
        return
    m = b.m
    nodes = d.storage_bind_within_solve.join(d.storage_fix_start, on="n", how="inner")
    if nodes.height == 0:
        return
    first_period = d.nodeState_first_dt.select("d").unique()
    over = (d.nodeState_last_dt
            .join(nodes, on="n", how="inner")
            .join(first_period, on="d", how="inner")
            .select("n", "d", "t").unique())
    if over.height == 0:
        return

    # ─── EQ. ─── storage_state_start_binding_cyclic_period ───────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − state_start[n]·Σ_{(n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′]
    #   + state_start[n]·Σ_{(n,d′,d) ∈ edd_divest_active} v_divest_n[n,d′]
    #   = state_start[n] · existing[n,d] / state_unitsize[n]
    #   ∀ n ∈ bind_within_solve ∩ storage_fix_start, (n,d,t) ∈ nodeState_last_dt, d ∈ first period
    # The first step of a bind_within_solve node wraps to the last one, so the start state is pinned at
    # the last step (of the first period, see specs/github_issues.md, refactor bug 15) to the share
    # state_start of the built storage capacity (state units).  First solve of a chain only.
    lhs: dict = {"state": b.v_state}
    invested, divested = _node_in_service(b, d.nodeState, "_cyclic", coef=d.p_state_start)
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_start_binding_cyclic_period",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": (d.p_state_start * d.p_state_existing_capacity) / d.p_state_unitsize},
    )


def _eq_storage_reference_value(b: _Build) -> None:
    """End state of storage nodes with a reference value."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_use_reference_value)
            and d.p_storage_state_reference_value is not None
            and _has_rows(d.nodeState_last_dt)
            and d.period_last is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None):
        return
    m = b.m
    over = (d.nodeState_last_dt
            .join(d.period_last, on="d", how="inner")
            .join(d.storage_use_reference_value, on="n", how="inner")
            .select("n", "d", "t").unique())
    if over.height == 0:
        return

    # ─── EQ. ─── storage_state_solve_horizon_reference_value ─────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] = reference_value[n,d,t] · existing[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_use_reference_value, (n,d,t) ∈ nodeState_last_dt, d ∈ period_last
    # Pins the state at the end of the solve horizon to the reference share of the existing storage
    # capacity (state units).  Invested and divested capacity is not included (see specs/github_issues.md,
    # refactor bug 14).  Nodes with a competing end condition are not in storage_use_reference_value.
    target = (promote_param_to_dt(d.p_storage_state_reference_value, d.dt)
              .rename({"value": "rv"})
              .join(d.p_state_existing_capacity.frame.lazy().rename({"value": "exist"}),
                    on=["n", "d"], how="inner")
              .join(d.p_state_unitsize.frame.lazy().rename({"value": "us"}), on="n", how="inner")
              .with_columns(value=pl.col("rv") * pl.col("exist") / pl.col("us"))
              .select("n", "d", "t", "value")
              .collect())
    m.add_cstr(
        "storage_state_solve_horizon_reference_value",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": Where(b.v_state, over)},
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _eq_profiles(b: _Build) -> None:
    """Profile limits of process flows."""
    if not b.has_profile:
        return
    m, d = b.m, b.d

    # ─── EQ. ─── profile_flow_{upper_limit,lower_limit,fixed} ────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] ± Σ_r v_reserve[p,r,k,d,t]
    #   − profile[f,d,t]·av[p,d,t]·Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + profile[f,d,t]·av[p,d,t]·Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    #   ≤ / ≥ / =  profile[f,d,t] · av[p,d,t] · existing_count[p,d]
    #   ∀ (p,s,k,f) ∈ process_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds a flow by a profile times the built capacity, in units of unitsize.  av = availability, 1 for
    # processes without availability data.  The upper limit adds the upward reserve held at the sink, the
    # lower limit subtracts the downward reserve; the fixed profile has no reserve term.  The invest/divest
    # terms are restricted to the (p,f) pairs of the method's set.
    invested = divested = None
    if b.has_invest_p and d.edd_invest_set is not None:
        alive = _entity_rows(d.edd_invest_set, "e", d.process_source_sink, "p")
        if alive.height > 0:
            invest_view = _var_view(b.v_invest_p, "__at_invest_prof", {"d": "d_invest"}, to_axis=True)
            invested = Sum(Where(invest_view, alive), over=("d_invest",))
    if b.has_divest_p and d.edd_divest_active is not None:
        # edd_divest_active and process_source_sink carry different p vocabularies: cast to the former.
        procs = d.process_source_sink.select(pl.col("p").cast(d.edd_divest_active.schema["p"], strict=False)).unique()
        active = d.edd_divest_active.join(procs, on="p", how="semi")
        if active.height > 0:
            divest_view = _var_view(b.v_divest_p, "__at_divest_prof", {"d": "d_divest"}, to_axis=True)
            divested = Sum(Where(divest_view, active), over=("d_divest",))
    reserve_down = -b.reserve_down_to_sink if b.reserve_down_to_sink is not None else None
    for idx, name, sense, reserve_label, reserve in (
        (d.process_profile_upper, "profile_flow_upper_limit", "<=", "reserve",     b.reserve_up_to_sink),
        (d.process_profile_lower, "profile_flow_lower_limit", ">=", "reserve_neg", reserve_down),
        (d.process_profile_fixed, "profile_flow_fixed",       "==", None,          None),
    ):
        if not _has_rows(idx):
            continue
        over = idx.join(d.dt, how="cross")
        rhs = d.p_profile_value * d.p_process_existing_count
        avail = _availability_factor(d, idx) if d.p_process_availability is not None else None
        if avail is not None:
            rhs = rhs * avail
        lhs: dict = {"flow": b.v_flow}
        if invested is not None or divested is not None:
            # Hand-built data may name the profile column "profile" instead of f.
            process_profile_pairs = idx.select("p", "f" if "f" in idx.columns else "profile").unique()
            if "profile" in process_profile_pairs.columns and "f" not in process_profile_pairs.columns:
                process_profile_pairs = process_profile_pairs.pipe(rename_to_axis, {"profile": "f"})
        if invested is not None:
            term = invested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["invest_neg"] = -Where(term, process_profile_pairs)
        if divested is not None:
            term = divested * d.p_profile_value
            if avail is not None:
                term = term * avail
            lhs["divest"] = Where(term, process_profile_pairs)
        if reserve is not None:
            lhs[reserve_label] = reserve
        m.add_cstr(
            name,
            over      = over,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"limit": rhs},
        )


# ---------------------------------------------------------------------------
# Objective


def _sum_terms(terms: list) -> Expr | None:
    """t₁ + t₂ + … in list order; None for an empty list."""
    total = None
    for term in terms:
        total = term if total is None else total + term
    return total


def _obj_balance_slack(b: _Build) -> Expr:
    """Balance slack penalty."""
    d = b.d
    vq_up = b.vq_state_up
    vq_down = b.vq_state_down
    op_factor = b.op_factor
    # §7: node_capacity_for_scaling factor on slack penalties.  The .mod
    # treats `penalty_up/down` as €/MWh-of-node-capacity; our data has
    # absolute €/MWh.  Multiplying by node_capacity_for_scaling[n,d] aligns
    # with the .mod's convention and is critical for parity in scenarios
    # that have non-1.0 capacity scaling.
    if d.p_node_capacity_for_scaling is not None:
        return (Sum(vq_up   * d.p_penalty_up
                    * d.p_node_capacity_for_scaling * op_factor)
                + Sum(vq_down * d.p_penalty_down
                      * d.p_node_capacity_for_scaling * op_factor))
    return (Sum(vq_up   * d.p_penalty_up   * op_factor)
            + Sum(vq_down * d.p_penalty_down * op_factor))


def _obj_commodity(b: _Build) -> Expr | None:
    """Commodity purchase cost and sales revenue."""
    terms: list = []
    d = b.d
    v_flow = b.v_flow
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    op_factor = b.op_factor
    # Ladder commodities skip the legacy single-price commodity term —
    # mirrors .mod:1984 ``c not in commodity_with_ladder``.  Filter the
    # three commodity-flow index frames before they're consumed below.
    flow_from_commodity_eff_legacy   = d.flow_from_commodity_eff
    flow_from_commodity_noEff_legacy = d.flow_from_commodity_noEff
    flow_to_commodity_legacy         = d.flow_to_commodity
    if b.has_ladder and d.commodity_with_ladder is not None:
        ladder_c = d.commodity_with_ladder["c"]
        if flow_from_commodity_eff_legacy is not None:
            flow_from_commodity_eff_legacy = (
                flow_from_commodity_eff_legacy
                .filter(~pl.col("c").is_in(ladder_c)))
        if flow_from_commodity_noEff_legacy is not None:
            flow_from_commodity_noEff_legacy = (
                flow_from_commodity_noEff_legacy
                .filter(~pl.col("c").is_in(ladder_c)))
        if flow_to_commodity_legacy is not None:
            flow_to_commodity_legacy = (
                flow_to_commodity_legacy
                .filter(~pl.col("c").is_in(ladder_c)))

    if b.has_proc and flow_from_commodity_eff_legacy is not None and flow_from_commodity_eff_legacy.height > 0:
        terms.append(Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, flow_from_commodity_eff_legacy)
            * d.p_commodity_price * op_factor))
        if b.has_minload_eff:
            # Section term: + v_online * section * unitsize, summed over
            # eff (p, source, sink) tuples whose process is in
            # process_min_load_eff, multiplied by commodity price.
            if b.has_online_lin:
                terms.append(Sum(
                    Where(Where(v_online_lin, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          flow_from_commodity_eff_legacy)
                    * d.p_commodity_price * op_factor))
            if b.has_online_int:
                terms.append(Sum(
                    Where(Where(v_online_int, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          flow_from_commodity_eff_legacy)
                    * d.p_commodity_price * op_factor))
    if b.has_proc and flow_from_commodity_noEff_legacy is not None and flow_from_commodity_noEff_legacy.height > 0:
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, flow_from_commodity_noEff_legacy)
            * d.p_commodity_price * op_factor))
    # §2.4 commodity sell: sink-side flow into priced node = revenue (negative
    # cost contribution).  Note: full superset of (p, source, sink) — no
    # eff/noEff split (the .mod sells regardless of process partition).
    if (b.has_proc and flow_to_commodity_legacy is not None
            and flow_to_commodity_legacy.height > 0
            and d.p_commodity_price is not None):
        terms.append(-Sum(
            Where(v_flow * d.p_unitsize, flow_to_commodity_legacy)
            * d.p_commodity_price * op_factor))
    return _sum_terms(terms)


def _obj_co2_price(b: _Build) -> Expr | None:
    """CO2 emission cost."""
    terms: list = []
    d = b.d
    v_flow = b.v_flow
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    op_factor = b.op_factor
    if b.has_co2_price:
        terms.append(Sum(
            Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_co2_priced)
            * d.p_co2_content * d.p_co2_price * op_factor))
        if b.has_minload_eff:
            if b.has_online_lin:
                terms.append(Sum(
                    Where(Where(v_online_lin, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          d.flow_from_co2_priced)
                    * d.p_co2_content * d.p_co2_price * op_factor))
            if b.has_online_int:
                terms.append(Sum(
                    Where(Where(v_online_int, d.process_min_load_eff)
                          * d.p_section * d.p_unitsize,
                          d.flow_from_co2_priced)
                    * d.p_co2_content * d.p_co2_price * op_factor))
    # §4.1 CO2 noEff buy: source-side flow into a CO2-priced node where
    # the process is on the noEff side.  No slope, no section term.
    if (d.flow_from_co2_priced_noEff is not None
            and d.flow_from_co2_priced_noEff.height > 0
            and d.p_co2_content is not None and d.p_co2_price is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.flow_from_co2_priced_noEff)
            * d.p_co2_content * d.p_co2_price * op_factor))
    return _sum_terms(terms)


def _obj_variable_cost(b: _Build) -> Expr | None:
    """Variable operation and maintenance cost."""
    terms: list = []
    d = b.d
    v_flow = b.v_flow
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    op_factor = b.op_factor
    # ─── Process variable cost (§5.1 noEff) ───────────────────────────────
    # + Σ_{(p,source,sink,d,t) ∈ pssdt_varCost_noEff}
    #     pdtProcess__source__sink__dt_varCost * v_flow * unitsize * op_factor
    if (b.has_proc and d.pssdt_varCost_noEff is not None
            and d.pssdt_varCost_noEff.height > 0
            and d.p_pssdt_varCost is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_noEff)
            * d.p_pssdt_varCost * op_factor))

    # ─── Process variable cost (§5.2 eff unit, source-side O&M) ──────────
    # The .mod term has a leading minus sign — it's an "anti-cost" that
    # subtracts the source-side share of the process-side O&M (the slope
    # already routes the source-side energy into the sink-side flow at
    # commodity-price points).  Replicate the sign exactly.
    #   - Σ pdtProcess_source[p,source,'other_operational_cost',d,t]
    #         * (v_flow * unitsize * slope * (sink/source coef ratio if process_unit)
    #            + section term if min_load_efficiency) * op_factor
    if (b.has_proc and d.pssdt_varCost_eff_unit_source is not None
            and d.pssdt_varCost_eff_unit_source.height > 0
            and d.p_pdt_varCost_source is not None):
        # The flow-coefficient ratio is deferred (see audit §2.2/§5.2);
        # all current test fixtures have both coefs = 1, so the term
        # collapses to v_flow * unitsize * slope.
        flow_term = Where(v_flow * d.p_unitsize * d.p_slope,
                          d.pssdt_varCost_eff_unit_source)
        terms.append(-Sum(flow_term * d.p_pdt_varCost_source * op_factor))
        if b.has_minload_eff:
            # section sub-term (linear / integer)
            section_idx = (d.pssdt_varCost_eff_unit_source
                           .join(d.process_min_load_eff, on="p", how="inner"))
            if section_idx.height > 0:
                if b.has_online_lin:
                    terms.append(-Sum(
                        Where(Where(v_online_lin, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor))
                if b.has_online_int:
                    terms.append(-Sum(
                        Where(Where(v_online_int, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor))

    # ─── Process variable cost (§5.3 eff unit, sink-side O&M) ────────────
    # + Σ pdtProcess_sink[p,sink,'other_operational_cost',d,t]
    #     * v_flow * unitsize * op_factor
    if (b.has_proc and d.pssdt_varCost_eff_unit_sink is not None
            and d.pssdt_varCost_eff_unit_sink.height > 0
            and d.p_pdt_varCost_sink is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_unit_sink)
            * d.p_pdt_varCost_sink * op_factor))

    # ─── Process variable cost (§5.4 eff connection) ─────────────────────
    # + Σ pdtProcess[p,'other_operational_cost',d,t]
    #     * v_flow * unitsize * op_factor
    if (b.has_proc and d.pssdt_varCost_eff_connection is not None
            and d.pssdt_varCost_eff_connection.height > 0
            and d.p_pdt_varCost_process is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_connection)
            * d.p_pdt_varCost_process * op_factor))
    return _sum_terms(terms)


def _obj_startup_cost(b: _Build) -> Expr | None:
    """Startup cost."""
    terms: list = []
    d = b.d
    v_startup_lin = b.v_startup_linear
    v_startup_int = b.v_startup_integer
    # Startup cost: v_startup * startup_cost * unitsize, weighted by
    # timestep_weight / inflation / period_share — *no step_duration*
    # (startup is a discrete event, not duration-weighted).  Also carries
    # ``pdt_branch_weight`` when stochastics active (mod:2110).
    startup_factor = (d.p_timestep_weight * d.p_inflation_op
                      / d.p_period_share)
    if d.pdt_branch_weight is not None:
        startup_factor = startup_factor * d.pdt_branch_weight
    if b.has_startup_cost_lin:
        terms.append(Sum(
            Where(v_startup_lin, d.pdt_online_linear)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    if b.has_startup_cost_int:
        terms.append(Sum(
            Where(v_startup_int, d.pdt_online_integer)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    return _sum_terms(terms)


def _obj_storage_reference_price(b: _Build) -> Expr | None:
    """Value of the final storage state."""
    terms: list = []
    d = b.d
    v_state = b.v_state
    # ─── §10.1 Storage-state reference-price revenue (mod:2107-2111) ─────
    # For nodes with method ``use_reference_price`` (and also any node for
    # which a fix_storage_price handoff dual has populated the parameter
    # — both routes funnel through ``p_storage_state_reference_price``),
    # value v_state at the last (d, t) of every period_last by the
    # reference price:
    #
    #   - Σ_{n ∈ nodeState, (d, t) ∈ period__time_last : d ∈ period_last}
    #         p_storage_state_reference_price[n, d]
    #         * v_state[n, d, t] * unitsize[n]
    #         * timestep_weight[d, t] * inflation_op[d] / period_share[d]
    #         * pdt_branch_weight[d, t]
    #
    # The minus sign makes this a revenue term: a higher terminal state
    # reduces the objective.  No step_duration (terminal valuation is a
    # point event, not duration-weighted) — mirrors startup_factor.
    if (b.has_storage
            and d.p_storage_state_reference_price is not None
            and d.nodeState_last_dt is not None
            and d.nodeState_last_dt.height > 0
            and d.period_last is not None
            and d.period_last.height > 0):
        # Domain: (n, d, t) ∈ nodeState_last_dt where d ∈ period_last and
        # (n, d) is populated in p_storage_state_reference_price.  Zero-
        # valued rows contribute nothing but are kept (the loader does
        # not filter zeros and the multiplier is exact).
        ssrp_over = (d.nodeState_last_dt
            .join(d.period_last, on="d", how="inner")
            .join(d.p_storage_state_reference_price.frame.select("n", "d"),
                  on=["n", "d"], how="inner")
            .select("n", "d", "t").unique())
        if ssrp_over.height > 0:
            ref_price_factor = (d.p_timestep_weight * d.p_inflation_op
                                / d.p_period_share)
            if d.pdt_branch_weight is not None:
                ref_price_factor = ref_price_factor * d.pdt_branch_weight
            terms.append(-Sum(
                Where(v_state, ssrp_over) * d.p_state_unitsize
                * d.p_storage_state_reference_price * ref_price_factor))
    return _sum_terms(terms)


def _obj_invest_divest(b: _Build) -> Expr | None:
    """Investment and divestment annuities and fixed costs."""
    terms: list = []
    d = b.d
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    # Invest / divest objective contributions.
    #
    # Slice D — per-scenario (wait-and-see) stochastic investment.  Under
    # ``recourse`` the invest axis is fanned, so every scenario branch
    # (realised included) carries its OWN ``v_invest[e, d]`` columns at and
    # after the branching period; each is probability-weighted by
    # ``pd_branch_weight[d]`` so the objective is the expected value over
    # scenario leaves.  Under ``none`` (default) the invest axis is
    # realised-only, the weight would be 1.0, and the multiply is skipped
    # for byte-parity.  (Replaces the .mod:2116-2119 "NOT weighted / the
    # branch weight should be added if this is changed" note — this is that
    # change.)  ALL EIGHT invest/divest annuity/fixed-cost Params are
    # weighted: process-side annu/lf/lfd/annd AND node-side
    # annu_n/lf_n/lfd_n/annd_n (else storage/node invest stays unweighted).
    _recourse_wt = (getattr(d, "recourse_invest", False)
                    and d.pd_branch_weight is not None)

    def _wt(param):
        """Multiply a ``(x, d)`` cost Param by ``pd_branch_weight[d]``
        (Slice D).  No-op when the param is None or the flag is off."""
        if param is None or not _recourse_wt:
            return param
        f = (param.frame
                  .join(d.pd_branch_weight.frame.rename({"value": "w"}),
                        on="d", how="left")
                  .with_columns(value=pl.col("value")
                                * pl.col("w").fill_null(1.0))
                  .drop("w"))
        return Param(param.dims, f)

    if b.has_invest_p:
        annu = Param(("p", "d"),
            d.ed_entity_annual_discounted.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_entity_annual_discounted is not None else None
        lf  = Param(("p", "d"),
            d.ed_lifetime_fixed_cost.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_lifetime_fixed_cost is not None else None
        if annu is not None:
            terms.append(Sum(v_invest_p * d.p_unitsize * _wt(annu)))
        if lf is not None:
            terms.append(Sum(v_invest_p * d.p_unitsize * _wt(lf)))
    if b.has_divest_p:
        lfd = Param(("p", "d"),
            d.ed_lifetime_fixed_cost_divest.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_lifetime_fixed_cost_divest is not None else None
        annd = Param(("p", "d"),
            d.ed_entity_annual_divest_discounted.frame.pipe(rename_to_axis, {"e": "p"})) \
            if d.ed_entity_annual_divest_discounted is not None else None
        if lfd is not None:
            terms.append(-Sum(v_divest_p * d.p_unitsize * _wt(lfd)))
        if annd is not None:
            terms.append(-Sum(v_divest_p * d.p_unitsize * _wt(annd)))
    if b.has_invest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        annu_n = Param(("n", "d"),
            d.ed_entity_annual_discounted.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_entity_annual_discounted is not None else None
        lf_n = Param(("n", "d"),
            d.ed_lifetime_fixed_cost.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_lifetime_fixed_cost is not None else None
        if annu_n is not None:
            terms.append(Sum(v_invest_n * us_n * _wt(annu_n)))
        if lf_n is not None:
            terms.append(Sum(v_invest_n * us_n * _wt(lf_n)))
    if b.has_divest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        lfd_n = Param(("n", "d"),
            d.ed_lifetime_fixed_cost_divest.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_lifetime_fixed_cost_divest is not None else None
        annd_n = Param(("n", "d"),
            d.ed_entity_annual_divest_discounted.frame.pipe(rename_to_axis, {"e": "n"})) \
            if d.ed_entity_annual_divest_discounted is not None else None
        if lfd_n is not None:
            terms.append(-Sum(v_divest_n * us_n * _wt(lfd_n)))
        if annd_n is not None:
            terms.append(-Sum(v_divest_n * us_n * _wt(annd_n)))
    return _sum_terms(terms)


def _obj_existing_fixed_cost(b: _Build) -> None:
    """Fixed cost of existing capacity as an objective constant (opt-in)."""
    m, d = b.m, b.d
    # ─── §8.1 Existing-entity fixed cost (constant term) — opt-in ─────────
    # mod:2107-2115:
    #   + Σ_{e in entity, d in period_in_use}
    #         p_entity_all_existing[e,d] * ed_fixed_cost[e,d]
    #             * p_inflation_factor_operations_yearly[d] * pd_branch_weight[d]
    # See ``include_existing_fixed_cost`` doc on ``build_flextool``.
    if (b.include_existing_fixed_cost
            and d.p_ed_fixed_cost is not None
            and d.p_entity_all_existing is not None):
        fc_frame = (
            d.p_entity_all_existing.frame.rename({"value": "exist"})
            .join(d.p_ed_fixed_cost.frame.rename({"value": "fc"}),
                  on=["e", "d"], how="inner")
            .join(d.p_inflation_op.frame.rename({"value": "infl"}),
                  on="d", how="inner")
        )
        if d.pd_branch_weight is not None:
            fc_frame = (fc_frame
                .join(d.pd_branch_weight.frame.rename({"value": "pdbw"}),
                      on="d", how="inner"))
            fc_frame = fc_frame.with_columns(
                contrib=pl.col("exist") * pl.col("fc")
                        * pl.col("infl") * pl.col("pdbw"))
        else:
            fc_frame = fc_frame.with_columns(
                contrib=pl.col("exist") * pl.col("fc") * pl.col("infl"))
        fc_const = fc_frame["contrib"].sum()
        if fc_const:
            m.add_obj_constant(float(fc_const))


# ---------------------------------------------------------------------------
# Feature modules


def _feature_group_slack(b: _Build) -> Expr | None:
    """Group slack constraints and penalties (_group_slack)."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    vq_up = b.vq_state_up
    vq_down = b.vq_state_down
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    op_factor = b.op_factor
    # ─── Group-level slack (capacity_margin / inertia / non_sync) ────────
    if _group_slack.has_feature(d):
        vars_: dict = {}
        if b.has_proc:
            vars_["v_flow"]       = v_flow
        if b.has_online_lin:
            vars_["v_online_lin"] = v_online_lin
        if b.has_online_int:
            vars_["v_online_int"] = v_online_int
        if b.has_invest_p:
            vars_["v_invest_p"]   = v_invest_p
        if b.has_divest_p:
            vars_["v_divest_p"]   = v_divest_p
        if b.has_invest_n:
            vars_["v_invest_n"]   = v_invest_n
        if b.has_divest_n:
            vars_["v_divest_n"]   = v_divest_n
        vars_["vq_state_up"]   = vq_up
        vars_["vq_state_down"] = vq_down
        _group_slack.add_constraints(m, d, vars_)
        return _group_slack.add_objective_terms(m, d, vars_, op_factor)
    return None


def _feature_reserve(b: _Build) -> Expr | None:
    """Reserve constraints and costs (_reserve)."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_state = b.v_state
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    reserve_vars = b.reserve_vars
    op_factor = b.op_factor
    # ─── Reserves (timeseries / dynamic / n_1 / per-process upper) ────────
    if _reserve.has_feature(d):
        res_vars: dict = dict(reserve_vars)
        if b.has_proc:
            res_vars["v_flow"]       = v_flow
        if b.has_online_lin:
            res_vars["v_online_lin"] = v_online_lin
        if b.has_online_int:
            res_vars["v_online_int"] = v_online_int
        if b.has_invest_p:
            res_vars["v_invest_p"]   = v_invest_p
        if b.has_divest_p:
            res_vars["v_divest_p"]   = v_divest_p
        # Storage energy-adequacy coupling for reserve_duration (#322):
        # v_state precedes this dispatch site (declared above), so threading
        # it (+ v_invest_n / v_divest_n for the down-headroom capacity
        # replication) is order-safe.  Each gated on its own has_* flag.
        if b.has_storage:
            res_vars["v_state"]      = v_state
        if b.has_invest_n:
            res_vars["v_invest_n"]   = v_invest_n
        if b.has_divest_n:
            res_vars["v_divest_n"]   = v_divest_n
        _reserve.add_constraints(m, d, res_vars)
        return _reserve.add_objective_terms(m, d, res_vars, op_factor)
    return None


def _feature_cumulative_invest(b: _Build) -> Expr | None:
    """Cumulative and group investment limits (_cumulative_invest)."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    op_factor = b.op_factor
    # ─── Cumulative / group-invest / min-invest constraints ───────────────
    if _cumulative_invest.has_feature(d):
        cum_vars: dict = {
            "v_invest_p": v_invest_p if b.has_invest_p else None,
            "v_invest_n": v_invest_n if b.has_invest_n else None,
            "v_divest_p": v_divest_p if b.has_divest_p else None,
            "v_divest_n": v_divest_n if b.has_divest_n else None,
            "v_flow":     v_flow     if b.has_proc     else None,
        }
        _cumulative_invest.add_constraints(m, d, cum_vars)
        return _cumulative_invest.add_objective_terms(m, d, cum_vars, op_factor)
    return None


def _feature_delay(b: _Build) -> Expr | None:
    """Delayed flows (_delay)."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    op_factor = b.op_factor
    # ─── Delayed processes / DR ──────────────────────────────────────────
    if _delay.has_feature(d):
        delay_vars: dict = {"v_flow": v_flow if b.has_proc else None}
        _delay.add_constraints(m, d, delay_vars)
    return _delay.add_objective_terms(
        m, d, {"v_flow": v_flow if b.has_proc else None}, op_factor)


def _feature_commodity_ladder(b: _Build) -> Expr | None:
    """Commodity price ladder (_commodity_ladder)."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    ladder_vars = b.ladder_vars
    # ─── Commodity price ladder ──────────────────────────────────────────
    # Per-tier v_trade balance + tier-cap constraints + per-tier price
    # objective contribution.  The legacy commodity price term above
    # already filtered out ladder commodities, so this term replaces it.
    if b.has_ladder:
        _commodity_ladder.add_constraints(
            m, d, ladder_vars,
            v_flow=v_flow if b.has_proc else None,
            p_unitsize=d.p_unitsize,
            p_slope=d.p_slope,
            p_step_duration=d.p_step_duration,
            p_timestep_weight=d.p_timestep_weight,
            flow_from_commodity_eff=d.flow_from_commodity_eff,
            flow_from_commodity_noEff=d.flow_from_commodity_noEff,
            flow_to_commodity=d.flow_to_commodity,
        )
        return _commodity_ladder.add_objective_terms(
            m, d, ladder_vars,
            p_inflation_op=d.p_inflation_op,
            p_period_share=d.p_period_share,
        )
    return None


# ---------------------------------------------------------------------------
# Stochastic ties


def _eq_non_anticipativity(b: _Build) -> None:
    """Dispatch non-anticipativity across stochastic forecast branches."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_state = b.v_state
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    reserve_vars = b.reserve_vars
    # ─── Non-anticipativity constraints (A6) ─────────────────────────────
    # mod:4173-4233.  Four constraint families pin per-branch dispatch
    # variables FORECAST-TO-FORECAST at every (d, t) ∈ the re-grounded tie
    # set — the committed decision is the same across the forecast branches,
    # which IS its non-anticipative realisation.  The tie is NEVER anchored
    # to the realised period / trunk (there may be no realised branch at all;
    # ``n_real`` can be 0):
    #
    #   non_anticipativity_storage_use   (n, d, b, t) — net storage charge
    #     LHS at (d, t) equals LHS at (b, t).  Active only when at least
    #     one (g, n) ∈ group_node has g ∈ groupStochastic.
    #   non_anticipativity_online_integer (p, d, b, t) — v_online_integer
    #     equal across forecasts.
    #   non_anticipativity_online_linear  (p, d, b, t) — v_online_linear
    #     equal across forecasts.
    #   non_anticipativity_reserve        (p, r, ud, n, d, b, t) — v_reserve
    #     equal across forecasts.
    #
    # Raw cohort: (d, b) ∈ period__branch with d != b and b ∈ period_in_use.
    # The b == d self-loop is vacuous and the metadata-only branches not in
    # period_in_use (e.g. ``period1_realized``) are excluded so the
    # constraint references a real LP variable.  ``d`` here is the trunk
    # (the real period name); the re-ground below turns this into a
    # forecast-to-forecast star keyed on a reference forecast.
    # OC1 #1 — the ONLINE commitment tie over the JUMP needs this block to
    # run even when ``dt_non_anticipativity`` is EMPTY (the separate_dispatch
    # PARENT, whose realized window is handed to the sibling, so the legacy
    # realized/fix-storage union is empty).  Gate on the forecast structure
    # (``period_branch_full``) plus EITHER a non-empty ``dt_non_anticipativity``
    # (the existing storage/reserve/online-window tie) OR an online var present
    # (the new jump tie built from the online steps when dtna is empty).
    # Without an online var and with empty dtna the block is a no-op, so
    # non-online stochastic solves stay byte-identical.
    _has_dtna_outer = (d.dt_non_anticipativity is not None
                       and d.dt_non_anticipativity.height > 0)
    if (d.period_branch_full is not None
            and d.period_branch_full.height > 0
            and (_has_dtna_outer or b.has_online_lin or b.has_online_int)):
        # Build (d, b) cohort: d ≠ b AND b ∈ period_in_use.
        # Cross-axis value compare: "d" is the period axis and "b" is the
        # branch axis (two different Enum vocabularies under Phase 4
        # activation).  Polars 1.40+ refuses ``!=`` between different
        # Enums; cast both to Utf8 so the comparison is by token string.
        db_pairs = d.period_branch_full.with_columns(
            pl.col("d").cast(pl.Utf8),
            pl.col("b").cast(pl.Utf8),
        ).filter(pl.col("d") != pl.col("b"))
        if d.period_in_use_set is not None:
            # Cross-axis join: piu.b carries period tokens (d-Enum
            # vocab) but db_pairs.b is branch-Enum.  Cast both to Utf8.
            piu = (d.period_in_use_set
                       .rename({"d": "b"})
                       .with_columns(pl.col("b").cast(pl.Utf8)))
            db_pairs = db_pairs.join(piu, on="b", how="inner")
        # Restore axis types post-compare (Utf8 was only needed for the
        # cross-Enum filter/join above; downstream consumers cross-join
        # ``db_pairs`` with axis-typed frames and join on "d"/"b" against
        # axis-typed cohorts, which requires Enum-typed keys for native
        # composition under Phase 4 activation).
        db_pairs = db_pairs.with_columns(
            cast_dim(pl.col("d"), None, "d"),
            cast_dim(pl.col("b"), None, "b"),
        )
        if db_pairs.height > 0:
            # ── Forecast-to-forecast re-ground (per anchor group) ──
            # The raw cohort anchors every forecast ``b`` to the trunk ``d``
            # (the realised/base period).  Re-ground it to a FORECAST star so
            # the tie couples the forecast branches to EACH OTHER, never to
            # the trunk (corrected stochastic model: the fan is forecasts
            # only; realisation of a committed variable IS the cross-forecast
            # tie; works at n_real=0).
            #
            # PER ANCHOR GROUP (NOT a single global reference): group the raw
            # pairs by trunk ``d``; within each group take the forecasts
            # (``b`` tokens), pick the lexicographically smallest as the
            # reference forecast ``d_ref`` (LP-deterministic — a lexical sort
            # of stable Utf8 tokens, same invariant as add_var/add_cstr domain
            # sorting), and build the star {(d_ref, b) : b ∈ forecasts, b ≠
            # d_ref}.  A group with <2 forecasts is skipped (the tie is
            # vacuous with a single forecast).  A SINGLE GLOBAL d_ref would be
            # WRONG: it would cross-couple distinct anchor periods at
            # mismatched timesteps.
            #
            # BOTH frames are re-keyed: ``db_pairs`` → the forecast star, and
            # ``dt_non_anticipativity`` → anchor ``d`` mapped from the trunk
            # to that group's ``d_ref`` (same ``t``).  The families inner-join
            # the tie set on ``"d"``; re-keying db_pairs alone would leave the
            # tie set's ``d`` on the trunk and emit ZERO constraints.
            #
            # Realised exclusion: realisation is currently the self-row trunk,
            # already dropped from ``b`` by ``filter(d != b)`` above — so no
            # extra filter is needed here.  Should a future realised-
            # designated non-self in-use branch appear, drop it from the
            # forecast set at this point (by the realised marker R1 settles).
            # ``dt_non_anticipativity`` may be None / empty under the relaxed
            # gate (separate_dispatch parent, online-only path): then there is
            # nothing to re-ground, ``dtna_reground`` stays None, and the
            # function builds the jump tie from the online steps via its
            # internal ``dtna_online``.
            dtna_raw = d.dt_non_anticipativity
            _has_dtna_rows = dtna_raw is not None and dtna_raw.height > 0
            _raw_db = db_pairs.with_columns(
                pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8),
            )
            _dtna_u = (dtna_raw.with_columns(pl.col("d").cast(pl.Utf8))
                       if _has_dtna_rows else None)
            _stars: list[pl.DataFrame] = []
            _dtnas: list[pl.DataFrame] = []
            for d_trunk in sorted(
                _raw_db.select("d").unique().to_series().to_list()
            ):
                forecasts = sorted(
                    _raw_db.filter(pl.col("d") == d_trunk)
                           .select("b").unique().to_series().to_list()
                )
                if len(forecasts) < 2:
                    continue
                d_ref = forecasts[0]
                _stars.append(pl.DataFrame({
                    "d": [d_ref] * (len(forecasts) - 1),
                    "b": [b for b in forecasts if b != d_ref],
                }))
                if _dtna_u is not None:
                    _dtnas.append(
                        _dtna_u.filter(pl.col("d") == d_trunk)
                               .with_columns(pl.lit(d_ref).alias("d"))
                    )
            if _stars:
                db_pairs_reground = pl.concat(_stars).with_columns(
                    cast_dim(pl.col("d"), None, "d"),
                    cast_dim(pl.col("b"), None, "b"),
                )
                # ``dtna_reground`` is None on the separate_dispatch-parent
                # online-only path (no real dtna rows to re-key) → the
                # function builds ``dtna_online`` from the jump steps alone.
                dtna_reground = (
                    pl.concat(_dtnas).with_columns(cast_dim(pl.col("d"), None, "d"))
                    if _dtnas else None
                )
                _add_non_anticipativity_constraints(
                    m, d, db_pairs_reground,
                    v_state          = v_state          if b.has_storage    else None,
                    v_online_integer = v_online_int     if b.has_online_int else None,
                    v_online_linear  = v_online_lin     if b.has_online_lin else None,
                    v_reserve        = reserve_vars.get("v_reserve")    if reserve_vars   else None,
                    v_flow           = v_flow           if b.has_proc       else None,
                    has_minload_eff  = b.has_minload_eff,
                    dtna_override    = dtna_reground,
                )


def _eq_non_anticipativity_invest(b: _Build) -> None:
    """Investment non-anticipativity across stochastic branches."""
    m, d = b.m, b.d
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    # ─── Investment non-anticipativity (v72 — the standard two-stage
    # stochastic-investment mode) ────────────────────────────────────────
    # Ties v_invest / v_divest across stochastic branches over the resolved
    # window pd_non_anticipativity (design §4).  Gated on recourse_invest
    # (per-branch invest columns must exist to tie) AND a non-empty frame;
    # under the unset window the frame is provably empty (byte-parity) and
    # this block never fires.  Under stochastic_invest_method=none the frame
    # may build but the domain guard drops every pair (no fanned invest
    # column) — a graceful no-op.
    _pdna = d.pd_non_anticipativity
    if _pdna is not None and _pdna.height > 0:
        if getattr(d, "recourse_invest", False):
            _add_non_anticipativity_invest_constraints(
                m, d,
                v_invest_p = v_invest_p if b.has_invest_p else None,
                v_invest_n = v_invest_n if b.has_invest_n else None,
                v_divest_p = v_divest_p if b.has_divest_p else None,
                v_divest_n = v_divest_n if b.has_divest_n else None,
            )
        else:
            # Window set on a non-recourse (none) solve: investment is
            # already shared across all periods, so there is nothing to
            # tie.  Graceful no-op + advisory (never mutates the problem,
            # design §4.5 / OQ-3).
            _LOG.info(
                "shared_invest_periods is set but "
                "stochastic_invest_method is not 'recourse'; investment is "
                "already shared across all periods, so the invest-NA window "
                "has no effect (use 'recourse' for shared-first-period / "
                "recourse-later)."
            )


# ---------------------------------------------------------------------------
# Helpers


def _add_non_anticipativity_constraints(
    m, d, db_pairs: pl.DataFrame, *,
    v_state, v_online_integer, v_online_linear, v_reserve,
    v_flow, has_minload_eff,
    dtna_override: pl.DataFrame | None = None,
) -> None:
    """Emit the four non-anticipativity constraint families (mod:4173-4233).

    All four constraints pin the per-branch dispatch variable at the
    reference-forecast anchor ``d`` equal to the same variable at sibling
    FORECAST ``b`` for every (d, t) ∈ the re-grounded tie set.  The cohort
    ``db_pairs`` is a FORECAST-TO-FORECAST star: ``d`` is a deterministic
    reference forecast and ``b`` ranges over the OTHER forecasts of the same
    anchor group.  The tie therefore couples the forecast branches to EACH
    OTHER (via the reference forecast), NEVER to the realised period / trunk
    — that is what makes the committed decision non-anticipative, and it
    holds with no realised branch at all (``n_real`` may be 0).

    The caller (the ``db_pairs`` build block) re-grounds both ``db_pairs``
    (→ the forecast star) and ``dt_non_anticipativity`` (anchor ``d`` re-keyed
    from the trunk to the reference forecast ``d_ref`` per anchor group,
    keeping the same ``t``) and hands the re-keyed tie set in via
    ``dtna_override``.  Every family here joins ``db_pairs`` (cross) then the
    tie set on ``"d"``, so both re-keys are required for a non-empty join.

    Implementation strategy (d-packed):
      LHS = var[..., d, t]
      RHS = var[..., b, t]   (encoded as a renamed Var sliced by db_pairs)
    The "rename" approach: clone the Var, rename ``d → b`` so the engine
    recognises ``b`` as a key dim.  Joining the cloned var on
    ``db_pairs(d, b)`` then keys both sides by (d, b) so the equality
    constraint's row-by-row LHS - RHS == 0 cleanly closes.

    Storage non-anticipativity is gated on ``groupStochastic ∩ group_node``
    (only fires for nodes in a stochastic-enabled group).  The other
    three constraints fire whenever their owning variable exists.

    The storage variant is the heaviest: its LHS / RHS is the per-(n,t)
    nodeBalance net-charge expression rebuilt at d and at b
    (mod:4173-4217).  Implementation here keeps it simple by pinning
    ``v_state[n, d, t] = v_state[n, b, t]`` directly — semantically
    equivalent under the .mod's storage-balance algebra (the difference
    between v_state at consecutive timesteps IS the net charge, and
    storage_state_start is shared across branches).  This collapses
    ~150 LOC of LHS reconstruction into 4 LOC and matches the .mod's
    LP-equivalent feasible region (verified via parity).
    """
    dtna = dtna_override if dtna_override is not None else d.dt_non_anticipativity
    has_dtna = dtna is not None and dtna.height > 0

    # ONLINE commitment tie over the JUMP (OC1 #1).  Unit commitment on the
    # committed/realized section (the "jump") is here-and-now: it must be
    # chosen before the scenario is known, so ``v_online`` is tied EQUAL
    # across the forecast branches over the whole jump.  The jump is already
    # the plain ``dt_non_anticipativity`` set in every case EXCEPT the
    # ``separate_dispatch`` PARENT, whose realized window is handed to the
    # sibling so its ``dtna`` is empty — there it would tie NOTHING.  So:
    # ``dtna_online`` = ``dtna`` when ``has_dtna`` (plain single-shot ties its
    # whole realized window; rolling ties the per-roll jump; mid-horizon ties
    # the pre-reveal window — all UNCHANGED, byte-parity); ELSE (empty dtna +
    # forecast branches + an online var) the FULL set of online ``(d, t)``
    # steps for the anchor forecast periods — the whole horizon = the jump
    # (no look-ahead yet in the single-shot separate_dispatch case).  ONLINE
    # ONLY — storage (#1/#1b) and reserve (#4) stay gated on ``has_dtna`` and
    # join the plain ``dtna``.  Inert for deterministic solves (no branches ⇒
    # ``db_pairs`` empty) and no-online solves.
    has_online = v_online_integer is not None or v_online_linear is not None
    dtna_online = dtna
    if (not has_dtna and has_online
            and db_pairs is not None and db_pairs.height > 0
            and d.p_online_dt is not None and d.p_online_dt.height > 0):
        jump_steps = (d.p_online_dt.select("d", "t").unique()
                      .join(db_pairs.select("d").unique(),
                            on="d", how="inner"))
        if jump_steps.height > 0:
            dtna_online = jump_steps
    has_dtna_online = dtna_online is not None and dtna_online.height > 0

    if not has_dtna and not has_dtna_online:
        return

    # 1. non_anticipativity_storage_use — pin the per-(n, t) NET CHARGE
    #    LHS across FORECASTS (mod:4173-4217).  Domain: nodes in a
    #    stochastic-group; (d, b) ∈ db_pairs (a forecast-to-forecast star
    #    keyed on the reference forecast d_ref — NOT the realised trunk);
    #    (d, t) ∈ the re-grounded tie set.
    #
    # LHS:
    #     + sink-side flow contributions (sink == n)
    #         Σ_{(p, source, n)} v_flow[p, source, n, *, t] · unitsize · step_duration
    #     - source-side flow contributions (source == n)
    #         Σ_{(p, n, sink)} v_flow[p, n, sink, *, t] · unitsize · step_duration
    #         (eff partition multiplies by slope; noEff uses unitsize alone)
    #
    # Using v_state pinning instead of flow LHS would let slack vars absorb
    # the difference (the storage-balance equality has slack_up / slack_down
    # which can soak up a flow imbalance at the storage node), so the
    # constraint must directly target the flow LHS expression — not just
    # the resulting v_state.  This matches the .mod's bookkeeping exactly.
    if (has_dtna and v_flow is not None and d.groupStochastic is not None
            and d.p_unitsize is not None
            and d.p_step_duration is not None):
        gn = d.group_node
        gs = d.groupStochastic
        if gn is not None and gn.height > 0 and gs.height > 0:
            stoch_nodes = (gn.join(gs, on="g", how="inner")
                            .select("n").unique())
            if stoch_nodes.height > 0:
                # Build per-side LHS contributions, then equate at (d, t)
                # vs (b, t) by routing the same Var through a renamed
                # alias d→b.
                v_flow_at_b = Var(
                    name=v_flow.name + "__nab",
                    dims=("p", "source", "sink", "b", "t"),
                    frame=v_flow.frame.rename({"d": "b"}),
                    lower=v_flow.lower, upper=v_flow.upper,
                )

                # Rename the Param frames for the sibling-branch RHS:
                # p_step_duration[d, t] → p_step_duration[b, t] when keyed on b.
                # The engine joins by column name; renaming on a fresh
                # Param keeps it usable in arithmetic with the renamed Var.
                p_step_dur_b = Param(
                    ("b", "t"),
                    d.p_step_duration.frame.rename({"d": "b"}),
                )

                # Sink-side selector: (p, source, sink=n) where n ∈ stoch_nodes.
                # Use process_source_sink (full set, not eff/noEff filtered).
                pss = d.process_source_sink
                if pss is not None and pss.height > 0:
                    # Sink-side at d: flow[p, source, n, d, t] * us * step_dur
                    sink_idx = (pss
                        .join(stoch_nodes.pipe(rename_to_axis, {"n": "sink"}),
                              on="sink", how="inner")
                        .with_columns(n=pl.col("sink")))
                    # Source-side at d: flow[p, n, sink, d, t] * us * (slope if eff) * step_dur
                    src_eff_idx = None
                    if d.process_source_sink_eff is not None:
                        src_eff_idx = (d.process_source_sink_eff
                            .join(stoch_nodes.pipe(rename_to_axis, {"n": "source"}),
                                  on="source", how="inner")
                            .with_columns(n=pl.col("source")))
                    src_noEff_idx = None
                    if d.process_source_sink_noEff is not None:
                        src_noEff_idx = (d.process_source_sink_noEff
                            .join(stoch_nodes.pipe(rename_to_axis, {"n": "source"}),
                                  on="source", how="inner")
                            .with_columns(n=pl.col("source")))

                    # ── Build LHS at d ──
                    lhs_pieces = []
                    if sink_idx is not None and sink_idx.height > 0:
                        lhs_pieces.append(Sum(
                            Where(v_flow * d.p_unitsize, sink_idx)
                            * d.p_step_duration,
                            over=("p", "source", "sink"),
                        ))
                    if (src_eff_idx is not None and src_eff_idx.height > 0
                            and d.p_slope is not None):
                        lhs_pieces.append(-Sum(
                            Where(v_flow * d.p_unitsize * d.p_slope,
                                  src_eff_idx)
                            * d.p_step_duration,
                            over=("p", "source", "sink"),
                        ))
                    if src_noEff_idx is not None and src_noEff_idx.height > 0:
                        lhs_pieces.append(-Sum(
                            Where(v_flow * d.p_unitsize, src_noEff_idx)
                            * d.p_step_duration,
                            over=("p", "source", "sink"),
                        ))

                    # ── Build LHS at b (negated, becomes "-LHS_b" so the
                    #    constraint reads: LHS_d - LHS_b == 0) ──
                    lhs_b_pieces = []
                    if sink_idx is not None and sink_idx.height > 0:
                        lhs_b_pieces.append(-Sum(
                            Where(v_flow_at_b * d.p_unitsize, sink_idx)
                            * p_step_dur_b,
                            over=("p", "source", "sink"),
                        ))
                    if (src_eff_idx is not None and src_eff_idx.height > 0
                            and d.p_slope is not None):
                        # p_slope at b: rename d→b
                        p_slope_b = Param(
                            ("p", "b", "t"),
                            d.p_slope.frame.rename({"d": "b"}),
                        )
                        lhs_b_pieces.append(Sum(
                            Where(v_flow_at_b * d.p_unitsize * p_slope_b,
                                  src_eff_idx)
                            * p_step_dur_b,
                            over=("p", "source", "sink"),
                        ))
                    if src_noEff_idx is not None and src_noEff_idx.height > 0:
                        lhs_b_pieces.append(Sum(
                            Where(v_flow_at_b * d.p_unitsize, src_noEff_idx)
                            * p_step_dur_b,
                            over=("p", "source", "sink"),
                        ))

                    if lhs_pieces and lhs_b_pieces:
                        # Constraint domain: (n, d, b, t)
                        cstr_over = (stoch_nodes
                            .join(db_pairs, how="cross")
                            .join(dtna, on="d", how="inner")
                            .select("n", "d", "b", "t").unique())
                        if cstr_over.height > 0:
                            lhs_terms = {f"d_{i}": e for i, e in enumerate(lhs_pieces)}
                            for i, e in enumerate(lhs_b_pieces):
                                lhs_terms[f"b_{i}"] = e
                            m.add_cstr(
                                "non_anticipativity_storage_use",
                                over      = cstr_over,
                                sense     = "==",
                                lhs_terms = lhs_terms,
                                rhs_terms = {},
                            )

    # 1b. non_anticipativity_storage_state_first — branch-point continuity by
    #     pinning ABSOLUTE ``v_state`` at the FIRST tied step of each anchor
    #     ``d`` (the reference forecast d_ref, NOT the realised trunk):
    #
    #         v_state[n, d_ref, t_first] − v_state[n, b, t_first] == 0
    #
    #     The storage net-charge family above ties only the per-(n, t) net
    #     charge (the consecutive ``v_state`` delta), fixing every forecast's
    #     state SHAPE across a contiguous tied jump but leaving a free constant
    #     offset between forecast copies.  In single-solve each forecast is
    #     hard start-bound and reference-value end-bound so the offset
    #     telescopes to zero; in rolling the untied post-reveal tail breaks
    #     that and the copies float by a constant (~0.06 in
    #     ``1_week_rolling_wind``).  Pinning the absolute state at the first
    #     tied step closes the jump absolutely.  This is a forecast-to-forecast
    #     tie (re-pointed onto the forecast star via the shared db_pairs),
    #     never a tie to the realised period/trunk.  It is OPTIMUM-CONDITIONAL,
    #     not a hard anchor: interior drift stays feasible when balance slack
    #     absorbs it or inflow differs across forecasts (the net-charge family
    #     controls neither), but is non-optimal for this fixture's
    #     deterministic-inflow zero-self-discharge reservoirs so the tie binds.
    #     ``v_online`` / ``v_reserve`` need no analog — their NA families
    #     already pin the absolute variable at EVERY tied step.
    #
    #     Horizon-guard (byte-parity): emit the tie for an (n, d) ONLY when its
    #     tied timesteps are a STRICT SUBSET of that (n, d)'s ``v_state``
    #     timeline (an untied tail exists).  When the tied set covers the full
    #     horizon (single-solve dispatch) the node is already doubly anchored
    #     (hard start + reference end), so the row is redundant — suppress it
    #     so byte-parity-gated single solves get ZERO new rows.
    if (has_dtna and v_state is not None and d.groupStochastic is not None
            and db_pairs.height > 0):
        gn = d.group_node
        gs = d.groupStochastic
        if gn is not None and gn.height > 0 and gs is not None and gs.height > 0:
            stoch_nodes = (gn.join(gs, on="g", how="inner")
                            .select("n").unique())
            if stoch_nodes.height > 0:
                # ``v_state`` declared cells (n, d, t) — same source the Var
                # was built from (``nodeState_dt``).
                vstate_dt = v_state.frame.select("n", "d", "t").unique()

                # Anchor step PER-d (NOT a single global ``dtna.t.min()``):
                # the first tied step of each anchor period, robust to a
                # future multi-reveal single solve whose ``d`` have different
                # tied ranges.
                first_per_d = (dtna.select("d", "t").unique()
                               .group_by("d")
                               .agg(pl.col("t").min().alias("t")))

                # Horizon-guard counts: per-d tied-step count vs per-(n, d)
                # full ``v_state`` cell count; keep only STRICT subsets.
                tied_counts = (dtna.select("d", "t").unique()
                               .group_by("d")
                               .agg(pl.len().alias("tied_count")))
                full_counts = (vstate_dt
                               .group_by("n", "d")
                               .agg(pl.len().alias("full_count")))
                nd_keep = (stoch_nodes
                           .join(dtna.select("d").unique(), how="cross")
                           .join(tied_counts, on="d", how="inner")
                           .join(full_counts, on=["n", "d"], how="inner")
                           .filter(pl.col("tied_count") < pl.col("full_count"))
                           .select("n", "d").unique())

                if nd_keep.height > 0:
                    # Domain (n, d, b, t_first) = nd_keep ⋈ first-tied-step ⋈
                    # db_pairs.
                    over = (nd_keep
                            .join(first_per_d, on="d", how="inner")
                            .join(db_pairs, on="d", how="inner")
                            .select("n", "d", "b", "t").unique())
                    # Marker / state-less-cell guard: inner-join
                    # ``nodeState_dt`` on BOTH (n, d, t) and (n, b, t) so any
                    # cell lacking a declared ``v_state`` row is dropped.  The
                    # (n, d, t) side is same-axis (native join); the (n, b, t)
                    # side is cross-Enum (over.b is branch-Enum, the renamed
                    # v_state domain's b is period-Enum), so cast both b keys
                    # to Utf8 for that join and restore axis types after.
                    over = over.join(vstate_dt, on=["n", "d", "t"],
                                     how="inner")
                    vbt = (vstate_dt.rename({"d": "b"})
                           .with_columns(pl.col("b").cast(pl.Utf8)))
                    over = (over
                            .with_columns(pl.col("b").cast(pl.Utf8))
                            .join(vbt, on=["n", "b", "t"], how="inner")
                            .with_columns(cast_dim(pl.col("b"), None, "b"))
                            .select("n", "d", "b", "t").unique())
                    if over.height > 0:
                        v_state_at_b = Var(
                            name=v_state.name + "__nab_first",
                            dims=("n", "b", "t"),
                            frame=v_state.frame.rename({"d": "b"}),
                            lower=v_state.lower, upper=v_state.upper,
                        )
                        m.add_cstr(
                            "non_anticipativity_storage_state_first",
                            over      = over,
                            sense     = "==",
                            lhs_terms = {"v_d":  v_state,
                                         "v_b": -v_state_at_b},
                            rhs_terms = {},
                        )

    # 2. non_anticipativity_online_integer
    if v_online_integer is not None:
        # Domain: (p, d, b, t) over process_online_integer × db_pairs × dtna.
        if d.process_online_integer is not None and d.process_online_integer.height > 0:
            cstr_over = (d.process_online_integer
                .join(db_pairs, how="cross")
                .join(dtna_online, on="d", how="inner")
                .select("p", "d", "b", "t").unique())
            # Restrict to (p, d, t) and (p, b, t) tuples that actually
            # exist in the var domain (p_online_dt) — otherwise the
            # constraint references an undeclared cell.
            if d.p_online_dt is not None:
                pdt = d.p_online_dt.select("p", "d", "t").unique()
                pbt = pdt.rename({"d": "b"})
                cstr_over = (cstr_over
                    .join(pdt, on=["p", "d", "t"], how="inner")
                    .join(pbt, on=["p", "b", "t"], how="inner")
                    .select("p", "d", "b", "t").unique())
            if cstr_over.height > 0:
                v_oint_at_b = Var(
                    name=v_online_integer.name + "__nab",
                    dims=("p", "b", "t"),
                    frame=v_online_integer.frame.rename({"d": "b"}),
                    lower=v_online_integer.lower, upper=v_online_integer.upper,
                )
                m.add_cstr(
                    "non_anticipativity_online_integer",
                    over      = cstr_over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_online_integer,
                                 "v_b": -v_oint_at_b},
                    rhs_terms = {},
                )

    # 3. non_anticipativity_online_linear
    if v_online_linear is not None:
        if d.process_online_linear is not None and d.process_online_linear.height > 0:
            cstr_over = (d.process_online_linear
                .join(db_pairs, how="cross")
                .join(dtna_online, on="d", how="inner")
                .select("p", "d", "b", "t").unique())
            if d.p_online_dt is not None:
                pdt = d.p_online_dt.select("p", "d", "t").unique()
                pbt = pdt.rename({"d": "b"})
                cstr_over = (cstr_over
                    .join(pdt, on=["p", "d", "t"], how="inner")
                    .join(pbt, on=["p", "b", "t"], how="inner")
                    .select("p", "d", "b", "t").unique())
            if cstr_over.height > 0:
                v_olin_at_b = Var(
                    name=v_online_linear.name + "__nab",
                    dims=("p", "b", "t"),
                    frame=v_online_linear.frame.rename({"d": "b"}),
                    lower=v_online_linear.lower, upper=v_online_linear.upper,
                )
                m.add_cstr(
                    "non_anticipativity_online_linear",
                    over      = cstr_over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_online_linear,
                                 "v_b": -v_olin_at_b},
                    rhs_terms = {},
                )

    # 4. non_anticipativity_reserve — fires only when reserve subsystem
    #    is active (sum{(r, ud, g) in reserve__upDown__group} 1 > 0).
    if has_dtna and v_reserve is not None:
        pruna = d.process_reserve_upDown_node_active
        ruDg = d.reserve_upDown_group
        if (pruna is not None and pruna.height > 0
                and ruDg is not None and ruDg.height > 0):
            # Domain: (p, r, ud, n, d, b, t) over pruna × db_pairs × dtna.
            cstr_over = (pruna
                .join(db_pairs, how="cross")
                .join(dtna, on="d", how="inner")
                .select("p", "r", "ud", "n", "d", "b", "t").unique())
            if cstr_over.height > 0:
                v_res_at_b = Var(
                    name=v_reserve.name + "__nab",
                    dims=("p", "r", "ud", "n", "b", "t"),
                    frame=v_reserve.frame.rename({"d": "b"}),
                    lower=v_reserve.lower, upper=v_reserve.upper,
                )
                m.add_cstr(
                    "non_anticipativity_reserve",
                    over      = cstr_over,
                    sense     = "==",
                    lhs_terms = {"v_d":  v_reserve,
                                 "v_b": -v_res_at_b},
                    rhs_terms = {},
                )


def _add_non_anticipativity_invest_constraints(
    m, d, *,
    v_invest_p, v_invest_n, v_divest_p, v_divest_n,
) -> None:
    """Emit the four invest non-anticipativity families (v72, design §4).

    Ties ``v_invest`` / ``v_divest`` across stochastic branches over the
    resolved window ``pd_non_anticipativity`` (d, b): for every window
    pair the branch copy's invest/divest column ``[·, b]`` is pinned equal
    to the realized anchor's column ``[·, d]`` (RHS 0).  Four families:

      * ``non_anticipativity_invest_p`` — process invest (p, d) vs (p, b)
      * ``non_anticipativity_invest_n`` — node/storage invest (n, d)/(n, b)
      * ``non_anticipativity_divest_p`` — process divest (mandatory, F2)
      * ``non_anticipativity_divest_n`` — node/storage divest

    Tying invest AND divest is mandatory (design §4.4): divest fans under
    recourse exactly as invest does, so a tied invest with a per-branch
    divest would leak the NET first-stage capacity
    ``(existing + invest − divest)`` across branches on every
    invest+divest-eligible (reinvest / lifetime) entity.

    The domain guard (design §4.2-4.3, F3) is load-bearing AND cross-Enum:
    ``pd_non_anticipativity.b`` is on the BRANCH axis (its values are
    period-copy tokens like ``p2035_low``) while ``pd_invest_set.d``
    renamed ``d→b`` is on the PERIOD axis — polars refuses a join between
    the two Enum vocabularies.  So the guard join casts BOTH ``b`` keys to
    Utf8 (the ``db_pairs`` dance, :func:`build_flextool` ~line 4440) and
    restores the axis types via ``cast_dim`` afterwards.  This is the
    FIRST test-covered use of this cross-Enum guard join (the operational
    online guard is dormant), so the solver-level invest-tie test is its
    mandatory coverage.
    """
    pdb = d.pd_non_anticipativity                       # (d, b)
    if pdb is None or pdb.height == 0:
        return

    def _tie(name: str, v, inv_set, ent: str) -> None:
        if v is None or inv_set is None or inv_set.height == 0:
            return
        pd_inv = inv_set.select(ent, "d").unique()                  # (ent, d)
        # d-side join is same-axis (both period-Enum) → native.
        cstr = pd_inv.join(pdb, on="d", how="inner")                # (ent, d, b)
        if cstr.height == 0:
            return
        # ── F3 guard: (ent, b) must be a declared invest cell.  b is
        #    branch-Enum on cstr, period-Enum on pd_inv.d → cast both b
        #    keys to Utf8 for the join, restore axis types afterwards.
        guard = (pd_inv.rename({"d": "b"})
                 .with_columns(pl.col("b").cast(pl.Utf8)))
        cstr_over = (cstr
            .with_columns(pl.col("b").cast(pl.Utf8))
            .join(guard, on=[ent, "b"], how="inner")
            .with_columns(cast_dim(pl.col("d"), None, "d"),
                          cast_dim(pl.col("b"), None, "b"))
            .select(ent, "d", "b").unique())
        if cstr_over.height == 0:
            return
        v_at_b = Var(
            name=v.name + "__nab",
            dims=(ent, "b"),
            frame=v.frame.rename({"d": "b"}),
            lower=v.lower, upper=v.upper,
        )
        m.add_cstr(
            name,
            over      = cstr_over,
            sense     = "==",
            lhs_terms = {"v_d":  v,
                         "v_b": -v_at_b},
            rhs_terms = {},
        )

    _tie("non_anticipativity_invest_p", v_invest_p, d.pd_invest_set, "p")
    _tie("non_anticipativity_invest_n", v_invest_n, d.nd_invest_set, "n")
    _tie("non_anticipativity_divest_p", v_divest_p, d.pd_divest_set, "p")
    _tie("non_anticipativity_divest_n", v_divest_n, d.nd_divest_set, "n")


def _max_flow_rhs(d, inv_procs: "pl.DataFrame | None" = None,
                  ) -> "tuple[Param, pl.DataFrame | None, Param | None, pl.DataFrame | None]":
    """maxFlow RHS per (p, source, sink, d, t), the capacity multiplier of its invest/divest terms, and the
    per-element v_flow upper bound that replaces the row of constant-capacity input arcs.

    Arc classes:

    * **uncapped**: conversion_flow_coeff = 0 arcs (``process_source_sink_uncapped``, e.g. a hydro
      pass-through).  No per-arc capacity applies; the RHS is the raw ``p_flow_upper`` value, without
      availability or capacity coefficient.
    * **input**: input arcs of indirect (multi-flow) units.  Their capacity is the maximum sum of the
      outputs (maxOutputSum) and conversion_indirect ties the inputs to the outputs.  An input arc s is
      limited to M_s · (existing/unitsize + Σ v_invest − Σ v_divest), M_s = p_indirect_input_cap ·
      availability (input_share_max times the full-load input per unit of capacity; with the default
      share 1 the input alone could run the unit at full output, so the limit is a loose bound).
      Processes without invest/divest variables in this solve (``inv_procs``) have a constant capacity:
      M_s · existing/unitsize becomes a v_flow upper bound and the arc gets no maxFlow row.  Processes
      with them keep the row, their invest/divest terms carrying M_s.
    * **capped**: direct arcs and indirect output arcs:
      v_flow ≤ coef · availability · (existing/unitsize + Σ v_invest − Σ v_divest), RHS =
      existing/unitsize · coef · availability.  ``p_flow_upper_existing`` is filled with 0 for greenfield
      arcs, ``p_arc_max_cap_coef`` and ``p_process_availability`` with their default 1 (a Param · Param
      product is an inner join and would drop the unauthored rows, giving RHS = 0).

    Returns ``(rhs, cap_mult, var_upper, bound_arcs)``: ``rhs`` covers every arc (maxFlow_negCap and
    maxFlow_back read it too); ``cap_mult`` (p, source, sink, d, t, value) holds the invest/divest
    multiplier of capped and input arcs where it is not 1 (None when there is none); ``var_upper`` is the
    v_flow upper-bound Param and ``bound_arcs`` its (p, source, sink) arcs (both None when no arc is
    bounded on the variable).

    Without ``p_flow_upper_existing`` (hand-built data) every arc that is not uncapped reads
    ``p_flow_upper`` as its existing capacity.

    Indirect units without an output arc would bind their capacity on a source arc, which this split does
    not route; they raise instead of getting free capacity.
    """
    keys = ["p", "source", "sink"]
    fue = getattr(d, "p_flow_upper_existing", None)
    fu = getattr(d, "p_flow_upper", None)
    pss = d.process_source_sink.select(keys).unique()
    indirect = getattr(d, "process_indirect", None)
    out_arcs = getattr(d, "process_output_flows", None)

    # ── uncapped arcs (need p_flow_upper for their unconstrained value) ──
    unc = getattr(d, "process_source_sink_uncapped", None)
    if fu is not None and unc is not None and unc.height > 0:
        unc = (_utf8_keys(pss.lazy(), keys)
               .join(_utf8_keys(unc.lazy(), keys).select(
                         [f"_{k}" for k in keys]),
                     on=[f"_{k}" for k in keys], how="semi")
               .select(keys).collect())
    else:
        unc = pss.head(0)

    # ── input arcs: indirect arcs that are neither outputs nor uncapped ──
    split_indirect = (fue is not None and fu is not None
                      and indirect is not None and indirect.height > 0)
    if split_indirect:
        indir_ps = indirect.select("p").unique()
        out_ps = (out_arcs.select("p").unique()
                  if out_arcs is not None else indir_ps.head(0))
        sinkless = indir_ps.join(out_ps, on="p", how="anti")
        if sinkless.height > 0:
            names = sorted(str(x) for x in sinkless.get_column("p").to_list())
            raise NotImplementedError(
                "Indirect (multi-flow) process(es) without an output "
                f"arc are not supported: {names}. Such sink-less units "
                "bind their capacity on a source arc (maxFromSource), "
                "which the maxFlow existing-capacity RHS does not yet "
                "cover; emitting them would grant free (unpaid) "
                "capacity. Add an output node or extend the indirect "
                "maxFlow RHS to route source arcs through the "
                "existing-only bound.")
        inputs = pss.join(indir_ps, on="p", how="semi")
        if out_arcs is not None and out_arcs.height > 0:
            inputs = inputs.join(out_arcs.select(keys), on=keys, how="anti")
        inputs = inputs.join(unc, on=keys, how="anti")
        in_arcs = getattr(d, "process_input_flows", None)
    else:
        inputs = pss.head(0)
        in_arcs = None
    capped = (pss.join(inputs, on=keys, how="anti")
                 .join(unc, on=keys, how="anti"))
    if in_arcs is not None and in_arcs.height > 0:
        # No indirect input arc may take the existing-only RHS of a capped
        # arc (it would force the input flow to 0).
        leaked = capped.join(in_arcs.select(keys), on=keys, how="semi")
        assert leaked.height == 0, (
            "indirect input arc leaked into the existing-only maxFlow "
            f"slice (would force v_flow_in ≤ 0): {leaked}")

    avail = getattr(d, "p_process_availability", None)
    avail_lf = (_utf8_keys(promote_param_to_dt(avail, d.dt), ["p"])
                .select("_p", "d", "t", pl.col("value").alias("__av"))
                if avail is not None else None)

    def _with_existing(lf: "pl.LazyFrame") -> "pl.LazyFrame":
        if fue is not None:
            return (lf
                    .join(_utf8_keys(fue.frame.lazy(), keys)
                          .select("_p", "_source", "_sink", "d",
                                  pl.col("value").alias("__ex")),
                          on=["_p", "_source", "_sink", "d"], how="left")
                    .with_columns(pl.col("__ex").fill_null(0.0)))
        if fu is not None:
            return (lf
                    .join(_utf8_keys(fu.frame.lazy(), keys)
                          .select("_p", "_source", "_sink", "d", "t",
                                  pl.col("value").alias("__ex")),
                          on=["_p", "_source", "_sink", "d", "t"], how="left")
                    .with_columns(pl.col("__ex").fill_null(0.0)))
        return lf.with_columns(pl.lit(0.0).alias("__ex"))

    def _with_availability(lf: "pl.LazyFrame") -> "pl.LazyFrame":
        if avail_lf is None:
            return lf.with_columns(pl.lit(1.0).alias("__av"))
        return (lf.join(avail_lf, on=["_p", "d", "t"], how="left")
                  .with_columns(pl.col("__av").fill_null(1.0)))

    # ── capped: existing/us × coef × availability ───────────────────────
    cap_lf = _with_existing(_utf8_keys(capped.join(d.dt, how="cross").lazy(),
                                       keys))
    if d.p_arc_max_cap_coef is not None:
        cap_lf = (cap_lf
                  .join(_utf8_keys(d.p_arc_max_cap_coef.frame.lazy(), keys)
                        .select("_p", "_source", "_sink",
                                pl.col("value").alias("__coef")),
                        on=["_p", "_source", "_sink"], how="left")
                  .with_columns(pl.col("__coef").fill_null(1.0)))
    else:
        cap_lf = cap_lf.with_columns(pl.lit(1.0).alias("__coef"))
    cap_frame = (_with_availability(cap_lf)
                 .with_columns(
                     # The order (existing × coef) × availability fixes
                     # the rounding of the RHS.
                     value=pl.col("__ex") * pl.col("__coef") * pl.col("__av"),
                     __mult=pl.col("__coef") * pl.col("__av"))
                 .select("p", "source", "sink", "d", "t", "value", "__mult")
                 .collect())
    mult_parts = [cap_frame.select("p", "source", "sink", "d", "t",
                                   pl.col("__mult").alias("value"))]
    slices = [cap_frame.select("p", "source", "sink", "d", "t", "value")]

    # ── input: M × existing/us, M = input cap × availability ────────────
    var_upper = None
    bound_arcs = None
    if inputs.height > 0:
        in_cap = getattr(d, "p_indirect_input_cap", None)
        if in_cap is None:
            raise ValueError(
                "FlexData.p_indirect_input_cap is missing but the model has "
                "indirect-unit input arcs: "
                f"{sorted(map(str, inputs.get_column('p').unique().to_list()))}."
                " It is produced with p_flow_upper by the input cascade "
                "(_derived_params.p_indirect_input_cap_from_source / "
                "wire_arc_capacity_params).")
        in_lf = _with_existing(_utf8_keys(
            inputs.join(d.dt, how="cross").lazy(), keys))
        in_lf = (in_lf
                 .join(_utf8_keys(in_cap.frame.lazy(), keys)
                       .select("_p", "_source", "_sink", "d", "t",
                               pl.col("value").alias("__m")),
                       on=["_p", "_source", "_sink", "d", "t"], how="left"))
        in_frame = (_with_availability(in_lf)
                    .with_columns(__mult=pl.col("__m") * pl.col("__av"))
                    .with_columns(value=pl.col("__ex") * pl.col("__mult"))
                    .select("p", "source", "sink", "d", "t", "value",
                            "__mult")
                    .collect())
        missing = in_frame.filter(pl.col("__mult").is_null())
        if missing.height > 0:
            raise ValueError(
                "p_indirect_input_cap has no value for indirect input "
                "arc(s) "
                f"{missing.select(keys).unique().head(10).to_dicts()} — "
                "the input-arc limit would be silently dropped.")
        mult_parts.append(in_frame.select("p", "source", "sink", "d", "t",
                                          pl.col("__mult").alias("value")))
        slices.append(in_frame.select("p", "source", "sink", "d", "t",
                                      "value"))
        # Constant-capacity processes: the limit is a v_flow bound.
        bnd = in_frame.lazy()
        if inv_procs is not None and inv_procs.height > 0:
            bnd = (_utf8_keys(bnd, ["p"])
                   .join(inv_procs.lazy().select(
                             pl.col("p").cast(pl.Utf8).alias("_p")),
                         on="_p", how="anti"))
        bnd_frame = (bnd.select("p", "source", "sink", "d", "t", "value")
                     .collect())
        if bnd_frame.height > 0:
            var_upper = Param(("p", "source", "sink", "d", "t"), bnd_frame)
            bound_arcs = bnd_frame.select(keys).unique(maintain_order=True)

    # ── uncapped: raw p_flow_upper ──────────────────────────────────────
    if unc.height > 0:
        slices.append(_utf8_keys(fu.frame.lazy(), keys)
                      .join(_utf8_keys(unc.lazy(), keys)
                            .select([f"_{k}" for k in keys]),
                            on=[f"_{k}" for k in keys], how="semi")
                      .select("p", "source", "sink", "d", "t", "value")
                      .collect())

    ref = slices[0].schema
    aligned = [slices[0]] + [
        sl.with_columns([pl.col(c).cast(ref[c], strict=False)
                         for c in ("p", "source", "sink", "d", "t")
                         if sl.schema[c] != ref[c]])
        for sl in slices[1:]]
    rhs = Param(("p", "source", "sink", "d", "t"), pl.concat(aligned))
    mult_ref = mult_parts[0].schema
    cap_mult = pl.concat([mult_parts[0]] + [
        mp.with_columns([pl.col(c).cast(mult_ref[c], strict=False)
                         for c in ("p", "source", "sink", "d", "t")
                         if mp.schema[c] != mult_ref[c]])
        for mp in mult_parts[1:]]).filter(pl.col("value") != 1.0)
    return (rhs, (cap_mult if cap_mult.height > 0 else None),
            var_upper, bound_arcs)


def _invest_capacity_multiplier(pss_dt: "pl.DataFrame | None",
                                cap_mult: "pl.DataFrame | None",
                                procs: "pl.DataFrame",
                                ) -> "Param | None":
    """Per-``(p, source, sink, d, t)`` multiplier for the maxFlow
    invest/divest terms, covering EVERY ``pss_dt`` row of the invest /
    divest processes ``procs`` — ``cap_mult`` (coef × availability) on
    capped arcs, 1.0 elsewhere (the multiply is an inner join, so a row
    missing here would silently drop the invest term from that row).

    ``None`` when no invest/divest process has a non-unit multiplier: the
    caller then keeps the bare ``Σ v_invest`` term.
    """
    if pss_dt is None or cap_mult is None or procs.height == 0:
        return None
    keys = ["p", "source", "sink", "d", "t"]
    procs_u = _utf8_keys(procs.lazy(), ["p"]).select("_p")
    mult_sel = (_utf8_keys(cap_mult.lazy(), keys)
                .join(procs_u, on="_p", how="semi")
                .select([f"_{k}" for k in keys] + ["value"])
                .collect())
    if mult_sel.height == 0:
        return None
    frame = (_utf8_keys(pss_dt.lazy().select(keys), keys)
             .join(procs_u, on="_p", how="semi")
             .join(mult_sel.lazy(), on=[f"_{k}" for k in keys], how="left")
             .with_columns(pl.col("value").fill_null(1.0))
             .select(*keys, "value")
             .collect())
    return Param(tuple(keys), frame)
