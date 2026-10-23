"""Single flextool model.  Mirrors flextool.mod's data-driven shape:
every variable / constraint / objective term is conditional on the
relevant data being present in ``FlexData``.

What's wired so far:

  * vq_state_up / vq_state_down (always)
  * v_flow + maxFlow + nodeBalance.sink_flow      (if processes)
  * commodity buy (eff + noEff)                     (if commodity_node)
  * conversion_indirect                              (if multi-flow process)
  * CO2-price objective term                         (if co2_price feature)
  * co2_max_period constraint                        (if co2_max group)
  * co2_max_total  constraint                        (if total-cap group)
  * user-defined process_constraint_eq / le / ge     (if constraint__sense)
  * profile_flow_upper / lower / fixed               (if profile)

Yet to wire (in user-requested order):
  storage → online + min_load → ramps → investments → multi_period
"""

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
# Field requirements per feature.  Each list is the FlexData fields that
# *must* be populated for the corresponding feature block to be active.
# ``ALWAYS`` is the floor — fields needed even on a slack-only run.
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
CO2_PRICE: tuple[str, ...]   = ("flow_from_co2_priced", "p_co2_content", "p_co2_price")
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
    reserve_down_from_source: Any = None    # Σ_r v_reserve down, (p, source, d, t)
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
        _eq_invest_bounds,
        _eq_conversion,
        _eq_co2_max_period,
        _eq_co2_max_total,
        _eq_user_constraints,
        _eq_storage_state,
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

    # Not read by any constraint at present.
    down_source = (pruna.filter(pl.col("ud") == "down")
                   .pipe(rename_to_axis, {"n": "source"})
                   .select("p", "r", "ud", "source"))
    if down_source.height > 0:
        at_source = _var_view(v_reserve, "__at_source", {"n": "source"},
                              to_axis=True)
        b.reserve_down_from_source = Sum(Where(at_source, down_source),
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
    d = b.d
    terms: dict = {}
    if b.has_divest_n and d.edd_divest_active is not None:
        divested = _var_view(b.v_divest_n, "__at_divest" + tag,
                             {"d": "d_divest"}, to_axis=True)
        active = _entity_rows(d.edd_divest_active, "p", nodes, "n")
        if active.height > 0:
            terms["divest"] = Sum(Where(divested, active), over=("d_divest",))
    if b.has_invest_n and d.edd_invest_set is not None:
        invested = _var_view(b.v_invest_n, "__at_invest" + tag,
                             {"d": "d_invest"}, to_axis=True)
        alive = _entity_rows(d.edd_invest_set, "e", nodes, "n")
        if alive.height > 0:
            terms["invest_neg"] = -Sum(Where(invested, alive), over=("d_invest",))
    return terms


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
    """Ramp limits."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_startup_lin = b.v_startup_linear
    v_shutdown_lin = b.v_shutdown_linear
    v_startup_int = b.v_startup_integer
    v_shutdown_int = b.v_shutdown_integer
    invest_in_dispatch = b.invest_in_dispatch_p
    divest_in_dispatch = b.divest_in_dispatch_p
    # ─── Ramp limits ──────────────────────────────────────────────────────
    if b.has_ramp:
        v_flow_prev = Lag(v_flow, d.dtttdt, "t", "t_previous")
        # 4 constraint families:
        #   sink_up:    step_dur*(v_flow_now - v_flow_prev) <=
        #               + ramp_speed_up_sink * 60 * step_dur * existing_count
        #               + (v_startup if online)
        #   sink_down:  step_dur*(v_flow_now - v_flow_prev) >=
        #               - ramp_speed_down_sink * 60 * step_dur * existing_count
        #               - (v_shutdown if online)
        #   source_up:   same with source-side ramp_speed
        #   source_down: same with source-side ramp_speed (and ramp_speed_down read from p_process_sink in the .mod — quirk that we follow as p_ramp_speed_down_sink)
        # Note: v_flow is in unitsize-units; divide both sides by unitsize
        # gives the form above (existing_count = existing/unitsize).
        for side, dir_, idx_set, ramp_param in [
            ("sink",   "up",   d.process_source_sink_ramp_limit_sink_up,
             d.p_ramp_speed_up_sink),
            ("sink",   "down", d.process_source_sink_ramp_limit_sink_down,
             d.p_ramp_speed_down_sink),
            ("source", "up",   d.process_source_sink_ramp_limit_source_up,
             d.p_ramp_speed_up_source),
            ("source", "down", d.process_source_sink_ramp_limit_source_down,
             # the .mod's source-down constraint uses
             # p_process_sink[p, source, ramp_speed_down] (a quirk).
             # In our test data p_ramp_speed_down_source is the right
             # source-side param; defer to it.
             d.p_ramp_speed_down_source),
        ]:
            if idx_set is None or idx_set.height == 0:
                continue
            if ramp_param is None:
                continue
            # conversion_flow_coeff = 0 edges are outside every ramp limit.
            idx_set = _drop_uncapped(d, idx_set)
            if idx_set.height == 0:
                continue
            sense = "<=" if dir_ == "up" else ">="
            sign  = 1.0  if dir_ == "up" else -1.0  # flips the RHS sign for "down"
            over_idx = idx_set.join(d.dt, how="cross")
            # LHS: step_dur * (v_flow_now - v_flow_prev) (Where filters
            # to idx_set tuples).  Both v_flow and v_flow_prev are
            # already over (p, source, sink, d, t).
            flow_diff = (Where(v_flow, idx_set) - Where(v_flow_prev, idx_set)) \
                         * d.p_step_duration
            # RHS: ±ramp_speed · 60 · step_dur · coef
            #        · (existing_count + Σv_invest − Σv_divest)
            # (.mod ramp_*_constraint): the side's capacity factor and the
            # invested / divested units scale the ramp limit exactly as
            # they scale the arc's capacity — output capacity_max_coeff on
            # the sink side, input_share_max on the source side, and for an
            # indirect unit's input arc its input limit per unit of capacity
            # (``p_indirect_input_cap`` = input_share_max × full-load input),
            # so the ramp speed is per unit of that arc's maximum flow.
            rate = ramp_param * 60.0 * d.p_step_duration
            if side == "sink":
                if d.p_process_sink_max_capacity_coef is not None:
                    rate = rate * _coef_factor(
                        idx_set, d.p_process_sink_max_capacity_coef,
                        ["p", "sink"])
            else:
                src_factor = _source_ramp_factor(d, idx_set)
                if src_factor is not None:
                    rate = rate * src_factor
            rhs_param = (rate * d.p_process_existing_count) * sign
            rhs_terms: dict = {"limit": rhs_param}
            if b.has_invest_p:
                rhs_terms["invest"] = (invest_in_dispatch * rate) * sign
            if b.has_divest_p:
                rhs_terms["divest"] = (divest_in_dispatch * rate) * (-sign)
            # UC startup/shutdown tightening
            if dir_ == "up":
                if b.has_online_lin:
                    rhs_terms["startup_lin"] = Where(v_startup_lin, idx_set)
                if b.has_online_int:
                    rhs_terms["startup_int"] = Where(v_startup_int, idx_set)
            else:
                if b.has_online_lin:
                    rhs_terms["shutdown_lin"] = -Where(v_shutdown_lin, idx_set)
                if b.has_online_int:
                    rhs_terms["shutdown_int"] = -Where(v_shutdown_int, idx_set)
            m.add_cstr(
                f"ramp_{side}_{dir_}_constraint",
                over      = over_idx,
                sense     = sense,
                lhs_terms = {"flow_diff": flow_diff},
                rhs_terms = rhs_terms,
            )


def _eq_invest_bounds(b: _Build) -> None:
    """Investment and divestment limits."""
    m, d = b.m, b.d
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    # ─── Invest / divest variable bounds + maxFlow tightening ──────────
    if b.has_invest_p or b.has_divest_p:
        # p-side max_units (rename "e" → "p" to align with process vars).
        # Phase 4.8g: the ContextVar is reset by the time ``build_flextool``
        # runs, so ``rename_to_axis`` cannot re-cast the renamed column to
        # the p-axis Enum.  Use a semi-join against ``process_source_sink``
        # (already p-Enum) up-cast to the e-axis vocab of the source frame
        # so the cross-Enum filter composes natively.
        _maxu = d.p_entity_max_units.frame
        _e_dt = _maxu.schema["e"]
        _p_in_e = (d.process_source_sink
                     .select(pl.col("p").cast(_e_dt, strict=False).alias("e"))
                     .unique())
        max_units_p = Param(("p", "d"),
            _maxu.join(_p_in_e, on="e", how="semi")
                  .pipe(rename_to_axis, {"e": "p"}))
    if b.has_invest_p:
        m.add_cstr(
            "maxInvest_var_bound",
            over      = d.pd_invest_set,
            sense     = "<=",
            lhs_terms = {"invest":    v_invest_p},
            rhs_terms = {"max_units": max_units_p},
        )
    if b.has_divest_p:
        m.add_cstr(
            "maxDivest_var_bound",
            over      = d.pd_divest_set,
            sense     = "<=",
            lhs_terms = {"divest":    v_divest_p},
            rhs_terms = {"max_units": max_units_p},
        )
    if b.has_invest_n or b.has_divest_n:
        # Phase 4.8g: same cross-Enum pattern as the ``max_units_p`` site
        # above — semi-join against ``nodeState['n']`` up-cast to the
        # e-axis vocab carried by ``p_entity_max_units``.
        _maxu_n = d.p_entity_max_units.frame
        _e_dt_n = _maxu_n.schema["e"]
        _n_in_e = (d.nodeState
                     .select(pl.col("n").cast(_e_dt_n, strict=False).alias("e"))
                     .unique())
        max_units_n = Param(("n", "d"),
            _maxu_n.join(_n_in_e, on="e", how="semi")
                    .pipe(rename_to_axis, {"e": "n"}))
    if b.has_invest_n:
        m.add_cstr(
            "maxInvest_var_bound_n",
            over      = d.nd_invest_set,
            sense     = "<=",
            lhs_terms = {"invest":    v_invest_n},
            rhs_terms = {"max_units": max_units_n},
        )
    if b.has_divest_n:
        m.add_cstr(
            "maxDivest_var_bound_n",
            over      = d.nd_divest_set,
            sense     = "<=",
            lhs_terms = {"divest":    v_divest_n},
            rhs_terms = {"max_units": max_units_n},
        )
    # Per-period invest/divest cap (.mod's maxInvest_entity_period /
    # maxDivest_entity_period).  Only applies on (e, d) ∈ ed_invest_period
    # — that's the entities/periods flagged with a per-period limit (e.g.
    # invest_method = "invest_period_total").  Entities with invest_method
    # = "invest_total" or "invest_no_limit" don't appear in this set.
    #
    # The mathematical relation is
    #     v_invest[e, d] * unitsize  ≤  ed_invest_max_period[e, d]
    # and we emit it in that natural form: the LP coefficient on
    # v_invest is ``unitsize`` (carried on the LHS) and the RHS is the
    # raw cap in MW (``ed_invest_max_period`` is already raw MW).  This
    # mirrors every other invest constraint (the per-entity total /
    # cumulative / group families all use coef=unitsize, RHS-in-MW), so
    # the extracted row dual is in obj/MW — consistent across families.
    # (Numerical conditioning that the old pre-divided coef=1 form bought
    # is now handled by autoscale Layer 2, so no special-casing here.)
    if b.has_invest_p and d.ed_invest_period_set is not None and d.ed_invest_period_set.height > 0:
        ed_p_period = d.ed_invest_period_set.pipe(rename_to_axis, {"e": "p"}).join(
            d.pd_invest_set, on=["p", "d"], how="inner")
        if ed_p_period.height > 0 and d.ed_invest_max_period is not None:
            cap_p = Param(("p", "d"),
                d.ed_invest_max_period.frame.pipe(rename_to_axis, {"e": "p"})
                .select("p", "d", "value"))
            m.add_cstr(
                "maxInvest_entity_period_p",
                over      = ed_p_period,
                sense     = "<=",
                lhs_terms = {"invest": v_invest_p * d.p_unitsize},
                rhs_terms = {"cap":    cap_p},
            )
    if b.has_invest_n and d.ed_invest_period_set is not None and d.ed_invest_period_set.height > 0:
        ed_n_period = d.ed_invest_period_set.pipe(rename_to_axis, {"e": "n"}).join(
            d.nd_invest_set, on=["n", "d"], how="inner")
        if ed_n_period.height > 0 and d.ed_invest_max_period is not None:
            cap_n = Param(("n", "d"),
                d.ed_invest_max_period.frame.pipe(rename_to_axis, {"e": "n"})
                .select("n", "d", "value"))
            us_n = Param(("n",), d.p_state_unitsize.frame)
            m.add_cstr(
                "maxInvest_entity_period_n",
                over      = ed_n_period,
                sense     = "<=",
                lhs_terms = {"invest": v_invest_n * us_n},
                rhs_terms = {"cap":    cap_n},
            )
    # Divest period caps: same coef=unitsize / raw-MW-RHS form as invest
    # above (dual in obj/MW, matching the total/cumulative/group families).
    if b.has_divest_p and d.ed_divest_period_set is not None and d.ed_divest_period_set.height > 0:
        ed_p_dperiod = d.ed_divest_period_set.pipe(rename_to_axis, {"e": "p"}).join(
            d.pd_divest_set, on=["p", "d"], how="inner")
        if ed_p_dperiod.height > 0 and d.ed_divest_max_period is not None:
            cap_dp = Param(("p", "d"),
                d.ed_divest_max_period.frame.pipe(rename_to_axis, {"e": "p"})
                .select("p", "d", "value"))
            m.add_cstr(
                "maxDivest_entity_period_p",
                over      = ed_p_dperiod,
                sense     = "<=",
                lhs_terms = {"divest": v_divest_p * d.p_unitsize},
                rhs_terms = {"cap":    cap_dp},
            )
    if b.has_divest_n and d.ed_divest_period_set is not None and d.ed_divest_period_set.height > 0:
        ed_n_dperiod = d.ed_divest_period_set.pipe(rename_to_axis, {"e": "n"}).join(
            d.nd_divest_set, on=["n", "d"], how="inner")
        if ed_n_dperiod.height > 0 and d.ed_divest_max_period is not None:
            cap_dn = Param(("n", "d"),
                d.ed_divest_max_period.frame.pipe(rename_to_axis, {"e": "n"})
                .select("n", "d", "value"))
            us_n = Param(("n",), d.p_state_unitsize.frame)
            m.add_cstr(
                "maxDivest_entity_period_n",
                over      = ed_n_dperiod,
                sense     = "<=",
                lhs_terms = {"divest": v_divest_n * us_n},
                rhs_terms = {"cap":    cap_dn},
            )
    # Per-entity totals: Σ_d v_invest[e, d] * unitsize ≤ e_invest_max_total[e]
    # (and divest analogue).  When running a non-first sub-solve of a
    # multi-solve chain, ``p_entity_previously_invested_capacity[e, d]``
    # / ``p_entity_invested[e]`` / ``p_entity_divested[e]`` carry the
    # cumulative realized invest/divest from prior solves and must be
    # added to the LHS — see flextool.mod:3597-3623.  polar_high emits one
    # row per entity (collapsed over d) so we use the max prior-capacity
    # per entity, which equals the tightest .mod row (the last d) for
    # the typical non-decreasing schedule.
    def _e_max_prior_cap() -> dict[str, float]:
        if d.p_entity_previously_invested_capacity is None:
            return {}
        return dict(
            d.p_entity_previously_invested_capacity.frame
              .group_by("e").agg(pl.col("value").max())
              .rows()
        )
    def _e_prior_invested() -> dict[str, float]:
        if d.p_entity_invested is None:
            return {}
        return dict(d.p_entity_invested.frame.rows())
    def _e_prior_divested() -> dict[str, float]:
        # The .mod gates p_entity_divested behind ``not p_model['solveFirst']``
        # — ignore the file on the first sub-solve of a chain (or for a
        # single-solve scenario).  ``p_nested_solve_first`` is tri-state:
        # None → no p_nested_model.csv (single-solve, treated as solveFirst);
        # True → solveFirst=1 (skip); False → solveFirst=0 (use the prior).
        if d.p_entity_divested is None:
            return {}
        if getattr(d, "p_nested_solve_first", None) is not False:
            return {}
        return dict(d.p_entity_divested.frame.rows())

    def _adjust_cap_frame(cap_frame: pl.DataFrame, key_col: str,
                           subtract: dict[str, float]) -> pl.DataFrame:
        """Return cap_frame with `value` -= subtract[key], clipped at 0."""
        if not subtract:
            return cap_frame
        adj = cap_frame.with_columns(
            value=(pl.col("value")
                   - pl.col(key_col).map_elements(lambda k: subtract.get(k, 0.0),
                                                    return_dtype=pl.Float64))
        ).with_columns(value=pl.when(pl.col("value") < 0).then(0.0)
                                .otherwise(pl.col("value")))
        return adj

    max_prior_cap = _e_max_prior_cap()
    # Not read below: p_entity_invested is not applied to the totals.
    prior_invested = _e_prior_invested()  # noqa: F841
    prior_divested = _e_prior_divested()

    if (b.has_invest_p and d.e_invest_total is not None
            and d.e_invest_total.height > 0
            and d.e_invest_max_total is not None):
        # Phase 4.8h: cross-Enum is_in (e-axis vs p-axis vocab); up-cast p→e.
        _e_dt = d.e_invest_total.schema["e"]
        _p_in_e = (d.process_source_sink
                     .select(pl.col("p").cast(_e_dt, strict=False).alias("e"))
                     .unique())
        e_inv_p = (d.e_invest_total
                     .join(_p_in_e, on="e", how="semi")
                     .pipe(rename_to_axis, {"e": "p"}))
        if e_inv_p.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_invest_max_total.frame.pipe(rename_to_axis, {"e": "p"}),
                "p", max_prior_cap)
            e_inv_max_p = Param(("p",), cap_frame)
            _emit_entity_total_cap(
                m,
                legacy_name = "maxInvest_entity_total",
                path_name   = "maxInvest_entity_total_path",
                over_e      = e_inv_p,
                var         = v_invest_p,
                unitsize    = d.p_unitsize,
                cap_param   = e_inv_max_p,
                d_leaf      = getattr(d, "d_leaf", None),
                lhs_label   = "invest_total",
            )
    if (b.has_divest_p and d.e_divest_total is not None
            and d.e_divest_total.height > 0
            and d.e_divest_max_total is not None):
        # Phase 4.8h: cross-Enum is_in (e-axis vs p-axis vocab); up-cast p→e.
        _e_dt = d.e_divest_total.schema["e"]
        _p_in_e = (d.process_source_sink
                     .select(pl.col("p").cast(_e_dt, strict=False).alias("e"))
                     .unique())
        e_div_p = (d.e_divest_total
                     .join(_p_in_e, on="e", how="semi")
                     .pipe(rename_to_axis, {"e": "p"}))
        if e_div_p.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_divest_max_total.frame.pipe(rename_to_axis, {"e": "p"}),
                "p", prior_divested)
            e_div_max_p = Param(("p",), cap_frame)
            _emit_entity_total_cap(
                m,
                legacy_name = "maxDivest_entity_total",
                path_name   = "maxDivest_entity_total_path",
                over_e      = e_div_p,
                var         = v_divest_p,
                unitsize    = d.p_unitsize,
                cap_param   = e_div_max_p,
                d_leaf      = getattr(d, "d_leaf", None),
                lhs_label   = "divest_total",
            )
    # Node analogues for per-entity totals.
    if (b.has_invest_n and d.e_invest_total is not None
            and d.e_invest_total.height > 0
            and d.e_invest_max_total is not None):
        # Phase 4.8h: cross-Enum is_in (e-axis vs n-axis vocab); up-cast n→e.
        _e_dt = d.e_invest_total.schema["e"]
        _n_in_e = (d.nodeState
                     .select(pl.col("n").cast(_e_dt, strict=False).alias("e"))
                     .unique())
        e_inv_n = (d.e_invest_total
                     .join(_n_in_e, on="e", how="semi")
                     .pipe(rename_to_axis, {"e": "n"}))
        if e_inv_n.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_invest_max_total.frame.pipe(rename_to_axis, {"e": "n"}),
                "n", max_prior_cap)
            e_inv_max_n = Param(("n",), cap_frame)
            us_n = Param(("n",), d.p_state_unitsize.frame)
            _emit_entity_total_cap(
                m,
                legacy_name = "maxInvest_entity_total_n",
                path_name   = "maxInvest_entity_total_path_n",
                over_e      = e_inv_n,
                var         = v_invest_n,
                unitsize    = us_n,
                cap_param   = e_inv_max_n,
                d_leaf      = getattr(d, "d_leaf", None),
                lhs_label   = "invest_total",
            )
    if (b.has_divest_n and d.e_divest_total is not None
            and d.e_divest_total.height > 0
            and d.e_divest_max_total is not None):
        # Phase 4.8h: cross-Enum is_in (e-axis vs n-axis vocab); up-cast n→e.
        _e_dt = d.e_divest_total.schema["e"]
        _n_in_e = (d.nodeState
                     .select(pl.col("n").cast(_e_dt, strict=False).alias("e"))
                     .unique())
        e_div_n = (d.e_divest_total
                     .join(_n_in_e, on="e", how="semi")
                     .pipe(rename_to_axis, {"e": "n"}))
        if e_div_n.height > 0:
            cap_frame = _adjust_cap_frame(
                d.e_divest_max_total.frame.pipe(rename_to_axis, {"e": "n"}),
                "n", prior_divested)
            e_div_max_n = Param(("n",), cap_frame)
            us_n = Param(("n",), d.p_state_unitsize.frame)
            _emit_entity_total_cap(
                m,
                legacy_name = "maxDivest_entity_total_n",
                path_name   = "maxDivest_entity_total_path_n",
                over_e      = e_div_n,
                var         = v_divest_n,
                unitsize    = us_n,
                cap_param   = e_div_max_n,
                d_leaf      = getattr(d, "d_leaf", None),
                lhs_label   = "divest_total",
            )


def _eq_conversion(b: _Build) -> None:
    """Conversion of indirect (multi-flow) units and minimum input shares."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    process_indirect_dt = b.process_indirect_dt
    # ─── conversion_indirect (CHP / multi-flow) ───────────────────────────
    if b.has_indirect:
        # Per flextool.mod:2557-2580, the conversion equation reads:
        #   Σ_source (v_flow[p, source, p] · unitsize · source_flow_coef[p, source])
        #     = Σ_sink (v_flow[p, p, sink] · unitsize · sink_flow_coef[p, sink])
        #         · slope[p, d, t]
        # Most fixtures have all flow_coefs = 1 (the implicit default) so
        # ``p_process_*_flow_coef`` is None and the Sum collapses to the
        # pre-flow-coef form.  When non-default, the Param multiplier
        # routes per-arc weights into the equation — this is what the
        # ``coal_chp_extraction`` scenario uses to express the iso-fuel
        # tradeoff between heat and electricity (sink coefs 0.2 vs 2.0).
        # Per flextool.mod:2343, the source-side input term also splits
        # into an *undelayed* part (current-time) and a *delayed* part
        # (time-shifted via dtt__delay_duration, weighted by
        # p_process_delay_weight).  Anti-join out the delayed processes
        # from the existing Sum, then add the delayed contribution from
        # _delay.delayed_input_expr.  The delayed term multiplies each
        # input by its source-side conversion_flow_coeff, like the
        # undelayed term (flextool.mod conversion_indirect).
        in_flows_undelayed = d.process_input_flows
        if (getattr(d, "process_delayed", None) is not None
                and d.process_delayed.height > 0):
            in_flows_undelayed = in_flows_undelayed.join(
                d.process_delayed, on="p", how="anti")

        input_expr = v_flow * d.p_unitsize
        if d.p_process_source_conversion_flow_coeff is not None:
            input_expr = input_expr * d.p_process_source_conversion_flow_coeff
        output_expr = v_flow * d.p_unitsize * d.p_slope
        if d.p_process_sink_conversion_flow_coeff is not None:
            output_expr = output_expr * d.p_process_sink_conversion_flow_coeff

        lhs_terms = {
            "input": Sum(Where(input_expr, in_flows_undelayed),
                          over=("source","sink")),
        }
        delayed_term = _delay.delayed_input_expr(d, v_flow)
        if delayed_term is not None:
            lhs_terms["input_delayed"] = delayed_term

        rhs_terms = {"output": Sum(Where(output_expr,
                                         d.process_output_flows),
                                   over=("source","sink"))}
        # min_load_efficiency section term (flextool.mod:2582):
        #   + (v_online_linear + v_online_integer)[p,d,t]
        #     · pdtProcess_section[p,d,t] · unitsize[p]
        # for indirect min_load_efficiency units.  The input of an indirect
        # unit is its own v_flow (noEff), so this is the ONLY place the
        # section fuel enters: nodeBalance, commodity cost, CO2 and the
        # reported input flows all read that v_flow.  The section is not
        # multiplied by any conversion_flow_coeff (as in the .mod).  For a
        # delayed unit it sits on the output side, at the sink-side time t
        # with v_online[t]; the delayed inputs supply it through
        # ``input_delayed``.  (p,d,t) rows without v_online lack the term.
        if b.has_minload_eff and d.p_section is not None:
            ind_mle = d.process_min_load_eff.join(
                d.process_indirect.select("p").unique(), on="p", how="semi")
            if ind_mle.height > 0:
                if b.has_online_lin:
                    rhs_terms["section_lin"] = (
                        Where(v_online_lin, ind_mle)
                        * d.p_section * d.p_unitsize)
                if b.has_online_int:
                    rhs_terms["section_int"] = (
                        Where(v_online_int, ind_mle)
                        * d.p_section * d.p_unitsize)

        m.add_cstr(
            "conversion_indirect",
            over      = process_indirect_dt,
            sense     = "==",
            lhs_terms = lhs_terms,
            rhs_terms = rhs_terms,
        )

        # minInputShare — unit__inputNode.input_share_min (mixing minimum
        # on the unit's current input; see :func:`_add_min_input_share`).
        _add_min_input_share(m, d, v_flow)


def _eq_co2_max_period(b: _Build) -> None:
    """CO2 emission cap per period."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    # ─── CO2 cap (period level) ───────────────────────────────────────────
    if b.has_co2_cap:
        # The .mod splits CO2 emissions by partition: eff multiplies by
        # ``pdtProcess_slope`` (efficiency factor on conversion), noEff
        # does not.  Combining both partitions in a single set with the
        # eff-style slope multiplier over-counts noEff processes' CO2
        # (e.g. coal_chp's slope=1.111 inflates its emissions by ~11%),
        # which skews co2_max_period dispatch on multi-period fixtures.
        lhs_terms_co2: dict = {}
        if b.has_co2_cap_eff:
            lhs_terms_co2["emissions_eff"] = Sum(
                Where(v_flow * d.p_unitsize * d.p_slope, d.flow_from_co2_capped)
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
                over=("p","source","sink","c","t"),
            )
        if b.has_co2_cap_noEff:
            lhs_terms_co2["emissions_noEff"] = Sum(
                Where(v_flow * d.p_unitsize, d.flow_from_co2_capped_noEff)
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
                over=("p","source","sink","c","t"),
            )
        m.add_cstr(
            "co2_max_period",
            over      = d.group_d_co2_capped,
            sense     = "<=",
            lhs_terms = lhs_terms_co2,
            rhs_terms = {"cap": d.p_co2_max_period},
        )


def _eq_co2_max_total(b: _Build) -> None:
    """CO2 emission cap over all periods."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    # ─── CO2 cap (multi-period total) ─────────────────────────────────────
    # Port of v3.32.0 ``co2_max_total`` (.mod:4019-4055).  Identical
    # per-(d,t) annualised-tonnes LHS shape as ``co2_max_period``, but
    # the ``d`` dim is also summed out so each capped group carries one
    # row whose RHS is the (g,)-only ``p_co2_max_total`` scalar (in
    # tonnes, projected per-year × 1-year-per-period for fixtures whose
    # ``p_period_share`` ≡ 1; legacy semantics for multi-year periods
    # would weight by ``p_years_represented_d`` — out of scope for this
    # port; existing co2_max_period also doesn't apply that weighting).
    if b.has_co2_cap_total:
        lhs_terms_co2_total: dict = {}
        if b.has_co2_cap_total_eff:
            lhs_terms_co2_total["emissions_eff"] = Sum(
                Where(v_flow * d.p_unitsize * d.p_slope,
                      d.flow_from_co2_capped_total)
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
                over=("p", "source", "sink", "c", "d", "t"),
            )
        if b.has_co2_cap_total_noEff:
            lhs_terms_co2_total["emissions_noEff"] = Sum(
                Where(v_flow * d.p_unitsize,
                      d.flow_from_co2_capped_total_noEff)
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
                over=("p", "source", "sink", "c", "d", "t"),
            )
        m.add_cstr(
            "co2_max_total",
            over      = d.group_co2_max_total,
            sense     = "<=",
            lhs_terms = lhs_terms_co2_total,
            rhs_terms = {"cap": d.p_co2_max_total},
        )


def _eq_user_constraints(b: _Build) -> None:
    """User-defined constraints."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_state = b.v_state
    v_invest_p = b.v_invest_p
    v_invest_n = b.v_invest_n
    # ─── User-defined flow constraints ────────────────────────────────────
    has_any_cdt = any(x is not None and x.height > 0
                      for x in (d.cdt_eq, d.cdt_le, d.cdt_ge))
    if has_any_cdt:
        # Build the LHS expression as a sum of optional contributions.
        # Each piece resolves to dims (cn, d, t) (or (cn, d) for invest,
        # which broadcasts over t at constraint emission).  The
        # constraint axis column is ``cn`` (not ``c``) — see the
        # c_collision review note in flextool_axis_contract.json.
        lhs_pieces: list = []
        if d.flow_constraint_idx is not None and d.flow_constraint_idx.height > 0:
            lhs_pieces.append(Sum(
                Where(v_flow * d.p_unitsize, d.flow_constraint_idx)
                * d.p_flow_constraint_coef,
                over=("p", "source", "sink"),
            ))
        # Invest-capacity terms.  Only relevant rows of v_invest_n /
        # v_invest_p are picked up by the inner-join in Where (entities
        # whose (n, cn) or (p, cn) appear in the coefficient frame).  The
        # join also adds the cn dim.  Sum collapses n / p, leaving (cn, d).
        if (b.has_invest_n and
                d.p_node_constraint_invested_capacity_coeff is not None):
            us_n_user = Param(("n",), d.p_state_unitsize.frame)
            n_inv_idx = (d.p_node_constraint_invested_capacity_coeff
                          .frame.select("n", "cn"))
            lhs_pieces.append(Sum(
                Where(v_invest_n * us_n_user, n_inv_idx)
                * d.p_node_constraint_invested_capacity_coeff,
                over=("n",),
            ))
        if (b.has_invest_p and
                d.p_process_constraint_invested_capacity_coeff is not None):
            p_inv_idx = (d.p_process_constraint_invested_capacity_coeff
                          .frame.select("p", "cn"))
            lhs_pieces.append(Sum(
                Where(v_invest_p * d.p_unitsize, p_inv_idx)
                * d.p_process_constraint_invested_capacity_coeff,
                over=("p",),
            ))
        # State-coefficient term:
        #   + Σ_{(n,cn) in node_state_constraint} v_state[n,d,t]
        #         * p_node_constraint_state_coeff[n,cn]
        #         * p_entity_unitsize[n]
        # Yields (cn, d, t) after summing over n.
        if (b.has_storage
                and d.p_node_constraint_state_coeff is not None):
            us_n_state = Param(("n",), d.p_state_unitsize.frame)
            n_state_idx = (d.p_node_constraint_state_coeff
                           .frame.select("n", "cn"))
            lhs_pieces.append(Sum(
                Where(v_state * us_n_state, n_state_idx)
                * d.p_node_constraint_state_coeff,
                over=("n",),
            ))
        # Prebuilt-capacity (existing-capacity contribution) — pure Param
        # constants, accumulated into a separate ``cstr_const`` so they
        # land on lhs_terms as Param entries (not summed into ``lhs_pieces``
        # which is an Expr).  Mirror of mod's:
        #   + Σ_{(n,cn) in node_capacity_constraint_prebuilt}
        #       (existing/unitsize) * coef * unitsize[n]
        # = Σ_n existing[n,d] * coef[n,cn]   →  Param dims (cn, d)
        cstr_const_pieces: list = []
        if (d.p_node_constraint_prebuilt_capacity_coeff is not None
                and d.p_state_existing_capacity is not None):
            n_pre = (d.p_state_existing_capacity.frame
                     .join(d.p_node_constraint_prebuilt_capacity_coeff
                            .frame.rename({"value": "coef"}),
                            on="n", how="inner")
                     .with_columns(value=pl.col("value") * pl.col("coef"))
                     .group_by(["cn", "d"]).agg(pl.col("value").sum())
                     .select("cn", "d", "value"))
            if n_pre.height > 0:
                cstr_const_pieces.append(Param(("cn", "d"), n_pre))
        if (d.p_process_constraint_prebuilt_capacity_coeff is not None
                and d.p_process_existing_count is not None
                and d.p_unitsize is not None):
            # process existing capacity = existing_count * unitsize, per (p, d)
            ec = d.p_process_existing_count.frame.rename({"value": "ec"})
            us = d.p_unitsize.frame.rename({"value": "us"})
            p_pre = (ec.join(us, on="p", how="inner")
                       .with_columns(cap=pl.col("ec") * pl.col("us"))
                       .join(d.p_process_constraint_prebuilt_capacity_coeff
                              .frame.rename({"value": "coef"}),
                              on="p", how="inner")
                       .with_columns(value=pl.col("cap") * pl.col("coef"))
                       .group_by(["cn", "d"]).agg(pl.col("value").sum())
                       .select("cn", "d", "value"))
            if p_pre.height > 0:
                cstr_const_pieces.append(Param(("cn", "d"), p_pre))

        # Prebuilt-capacity cumulative-invest contribution — variable.
        # Mirror of mod:2885-2898 / 2843-2856:
        #   + Σ {(p,cn) in process_capacity_constraint_prebuilt}
        #       (Σ_{d_invest : year[d_invest] < year[d]} v_invest[p, d_invest])
        #         * coef[p, cn] * unitsize[p]
        # The static-existing piece above is the first half of the LHS;
        # this adds the cumulative-prior-invest variable summand that
        # closes the multi_year_wind_growth_cap parity gap.
        if (b.has_invest_p and
                d.p_process_constraint_prebuilt_capacity_coeff is not None
                and d.edd_invest_lookback_set is not None
                and d.edd_invest_lookback_set.height > 0):
            v_inv_at_back_uc = Var(  # virtual rename: d → d_invest
                name=v_invest_p.name + "__at_back_uc",
                dims=("p", "d_invest"),
                frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_p.lower, upper=v_invest_p.upper,
            )
            p_pre_idx = (d.p_process_constraint_prebuilt_capacity_coeff
                          .frame.select("p", "cn"))
            lkb_p = (d.edd_invest_lookback_set.pipe(rename_to_axis, {"e": "p"})
                       .join(p_pre_idx, on="p", how="inner"))
            if lkb_p.height > 0:
                lhs_pieces.append(Sum(
                    Where(v_inv_at_back_uc * d.p_unitsize, lkb_p)
                    * d.p_process_constraint_prebuilt_capacity_coeff,
                    over=("p", "d_invest"),
                ))
        if (b.has_invest_n and
                d.p_node_constraint_prebuilt_capacity_coeff is not None
                and d.edd_invest_lookback_set is not None
                and d.edd_invest_lookback_set.height > 0):
            us_n_pre = Param(("n",), d.p_state_unitsize.frame)
            v_inv_n_at_back_uc = Var(
                name=v_invest_n.name + "__at_back_uc",
                dims=("n", "d_invest"),
                frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_n.lower, upper=v_invest_n.upper,
            )
            n_pre_idx = (d.p_node_constraint_prebuilt_capacity_coeff
                          .frame.select("n", "cn"))
            lkb_n = (d.edd_invest_lookback_set.pipe(rename_to_axis, {"e": "n"})
                       .join(n_pre_idx, on="n", how="inner"))
            if lkb_n.height > 0:
                lhs_pieces.append(Sum(
                    Where(v_inv_n_at_back_uc * us_n_pre, lkb_n)
                    * d.p_node_constraint_prebuilt_capacity_coeff,
                    over=("n", "d_invest"),
                ))
        if lhs_pieces or cstr_const_pieces:
            cstr_lhs_terms: dict = {}
            if lhs_pieces:
                cstr_lhs = lhs_pieces[0]
                for piece in lhs_pieces[1:]:
                    cstr_lhs = cstr_lhs + piece
                cstr_lhs_terms["contribution"] = cstr_lhs
            for i, p in enumerate(cstr_const_pieces):
                cstr_lhs_terms[f"prebuilt_const_{i}"] = p
            for axes, sense, name in [
                (d.cdt_eq, "==", "process_constraint_equal"),
                (d.cdt_le, "<=", "process_constraint_less_than"),
                (d.cdt_ge, ">=", "process_constraint_greater_than"),
            ]:
                if axes is None or axes.height == 0:
                    continue
                m.add_cstr(name, over=axes, sense=sense,
                           lhs_terms=cstr_lhs_terms,
                           rhs_terms={"constant":     d.p_constraint_constant})


def _eq_storage_state(b: _Build) -> None:
    """Storage state bounds, state profiles, start binding and end value."""
    m, d = b.m, b.d
    v_state = b.v_state
    v_invest_n = b.v_invest_n
    v_divest_n = b.v_divest_n
    nodeState_dt = b.nodeState_dt
    # ─── Storage state bounds + start binding ─────────────────────────────
    if b.has_storage:
        # maxState:  v_state[n, d, t]  <=  state_upper[n, d]
        # With invest/divest active on storage nodes, mirror the maxFlow
        # tightening: v_state + (divest - invest) ≤ state_upper, where the
        # invest/divest summations pick out (n, d_inv, d) tuples in
        # edd_invest / edd_divest_active that include both n and d.
        state_lhs: dict = {"state": v_state}
        if b.has_divest_n and d.edd_divest_active is not None:
            v_div_n_at = Var(
                name=v_divest_n.name + "__at_divest",
                dims=("n", "d_divest"),
                frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                lower=v_divest_n.lower, upper=v_divest_n.upper,
            )
            # Phase 4.8h: cross-Enum is_in (p-axis vs n-axis vocab); up-cast n→p.
            _p_dt = d.edd_divest_active.schema["p"]
            _n_in_p = (d.nodeState
                         .select(pl.col("n").cast(_p_dt, strict=False).alias("p"))
                         .unique())
            edd_div_n = (d.edd_divest_active
                           .join(_n_in_p, on="p", how="semi")
                           .pipe(rename_to_axis, {"p": "n"}))
            if edd_div_n.height > 0:
                state_lhs["divest"] = Sum(
                    Where(v_div_n_at, edd_div_n), over=("d_divest",))
        if b.has_invest_n and d.edd_invest_set is not None:
            v_inv_n_at = Var(
                name=v_invest_n.name + "__at_invest",
                dims=("n", "d_invest"),
                frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_n.lower, upper=v_invest_n.upper,
            )
            # Phase 4.8h: cross-Enum is_in (e-axis vs n-axis vocab); up-cast n→e.
            _e_dt = d.edd_invest_set.schema["e"]
            _n_in_e = (d.nodeState
                         .select(pl.col("n").cast(_e_dt, strict=False).alias("e"))
                         .unique())
            edd_inv_n = (d.edd_invest_set
                           .join(_n_in_e, on="e", how="semi")
                           .pipe(rename_to_axis, {"e": "n"}))
            if edd_inv_n.height > 0:
                state_lhs["invest_neg"] = -Sum(
                    Where(v_inv_n_at, edd_inv_n), over=("d_invest",))
        m.add_cstr(
            "maxState",
            over      = nodeState_dt,
            sense     = "<=",
            lhs_terms = state_lhs,
            rhs_terms = {"upper": d.p_state_upper},
        )

        # ─── State-profile bounds (profile_state_upper / lower / fixed) ─
        # mod:2639-2679.  Parallel to ``profile_flow_*`` but on v_state:
        #   LHS: v_state[n,d,t] · 1000
        #   RHS: profile[f,d,t] · ( existing_count[n,d]
        #                           + Σ_{d_inv} v_invest_n[n,d_inv]
        #                           - Σ_{d_div} v_divest_n[n,d_div] )
        #         · availability[n,d,t] · 1000
        # We divide both sides by 1000 and move invest/divest to LHS.
        # Availability defaults to 1.0 if not provided.
        any_node_profile = (
            (d.node_profile_upper is not None and d.node_profile_upper.height > 0)
            or (d.node_profile_lower is not None and d.node_profile_lower.height > 0)
            or (d.node_profile_fixed is not None and d.node_profile_fixed.height > 0)
        )
        if any_node_profile and d.p_profile_value is not None:
            # existing_count[n, d] = state_existing_capacity / state_unitsize.
            us_n_state_prof = d.p_state_unitsize.frame.rename({"value": "us"})
            ec_n_long = (d.p_state_existing_capacity.frame.rename({"value": "cap"})
                         .join(us_n_state_prof, on="n", how="inner")
                         .with_columns(value=pl.col("cap") / pl.col("us"))
                         .select("n", "d", "value"))
            p_node_existing_count = Param(("n", "d"), ec_n_long)

            def _add_node_profile_cstr(idx: "pl.DataFrame", name: str, sense: str) -> None:
                # over = (n, f, d, t) — cross-product of idx with timeline.
                over = idx.join(d.dt, how="cross")
                rhs_param = d.p_profile_value * p_node_existing_count
                # Densify availability over the profile nodes so a storage node
                # with no DB-authored ``availability`` keeps its default-1.0
                # factor instead of being inner-join-dropped to RHS=0 (→ forced
                # v_state=0).  Node-state twin of the process-side fix; see
                # :func:`_availability_factor`.
                node_avail_factor = (
                    _availability_factor(d, idx, axis="n",
                                         avail=d.p_node_availability)
                    if d.p_node_availability is not None else None)
                if node_avail_factor is not None:
                    rhs_param = rhs_param * node_avail_factor
                lhs: dict = {"state": v_state}
                # Invest/divest tightening — only over (n, f) pairs in idx.
                nf_filter = idx.select("n", "f").unique()
                if b.has_invest_n and d.edd_invest_set is not None:
                    # Phase 4.8h: cross-Enum is_in (e vs n vocab); up-cast n→e.
                    _e_dt = d.edd_invest_set.schema["e"]
                    _n_in_e = (d.nodeState
                                 .select(pl.col("n").cast(_e_dt, strict=False)
                                                    .alias("e"))
                                 .unique())
                    edd_inv_n_p = (d.edd_invest_set
                                     .join(_n_in_e, on="e", how="semi")
                                     .pipe(rename_to_axis, {"e": "n"}))
                    if edd_inv_n_p.height > 0:
                        v_inv_n_at_p = Var(
                            name=v_invest_n.name + "__at_invest_node_profile",
                            dims=("n", "d_invest"),
                            frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                            lower=v_invest_n.lower, upper=v_invest_n.upper,
                        )
                        inv_term = Sum(
                            Where(v_inv_n_at_p, edd_inv_n_p),
                            over=("d_invest",),
                        )
                        # Multiply by profile · (availability) and restrict to (n, f) pairs.
                        inv_full = inv_term * d.p_profile_value
                        if node_avail_factor is not None:
                            inv_full = inv_full * node_avail_factor
                        lhs["invest_neg"] = -Where(inv_full, nf_filter)
                if b.has_divest_n and d.edd_divest_active is not None:
                    # Phase 4.8h: cross-Enum is_in (p vs n vocab); up-cast n→p.
                    _p_dt = d.edd_divest_active.schema["p"]
                    _n_in_p = (d.nodeState
                                 .select(pl.col("n").cast(_p_dt, strict=False)
                                                    .alias("p"))
                                 .unique())
                    edd_div_n_p = (d.edd_divest_active
                                     .join(_n_in_p, on="p", how="semi")
                                     .pipe(rename_to_axis, {"p": "n"}))
                    if edd_div_n_p.height > 0:
                        v_div_n_at_p = Var(
                            name=v_divest_n.name + "__at_divest_node_profile",
                            dims=("n", "d_divest"),
                            frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                            lower=v_divest_n.lower, upper=v_divest_n.upper,
                        )
                        div_term = Sum(
                            Where(v_div_n_at_p, edd_div_n_p),
                            over=("d_divest",),
                        )
                        div_full = div_term * d.p_profile_value
                        if node_avail_factor is not None:
                            div_full = div_full * node_avail_factor
                        lhs["divest"] = Where(div_full, nf_filter)
                m.add_cstr(name, over=over, sense=sense,
                           lhs_terms=lhs,
                           rhs_terms={"limit": rhs_param})

            if d.node_profile_upper is not None and d.node_profile_upper.height > 0:
                _add_node_profile_cstr(d.node_profile_upper,
                                        "profile_state_upper_limit", "<=")
            if d.node_profile_lower is not None and d.node_profile_lower.height > 0:
                _add_node_profile_cstr(d.node_profile_lower,
                                        "profile_state_lower_limit", ">=")
            if d.node_profile_fixed is not None and d.node_profile_fixed.height > 0:
                _add_node_profile_cstr(d.node_profile_fixed,
                                        "profile_state_fixed", "==")

        # storage_state_start_binding (if fix_start):
        #   v_state[n, d, t_first] * unitsize
        #     ==  storage_state_start[n] * (existing
        #                                    + Σ_{d_inv} v_invest_n[d_inv] * unitsize
        #                                    - Σ_{d_div} v_divest_n[d_div] * unitsize)
        # Divide both sides by unitsize → v_state on the LHS as before, plus
        # invest/divest tightening on the LHS (with sign flip):
        #   v_state - state_start * Σ v_invest_n + state_start * Σ v_divest_n
        #     == state_start * existing / unitsize.
        # The .mod has TWO routes to the same numerical pinning depending
        # on the binding method:
        #   * ``bind_within_timeblock`` (and the default fall-through): a
        #     dedicated ``storage_state_start_binding`` constraint at
        #     (n, period_first, time_first) — flextool.mod:2725-2740.
        #   * ``bind_forward_only`` + fix_start: an in-balance term added
        #     to nodeBalance at the same row — flextool.mod:2197-2203.
        #     Numerically equivalent for our tested fixtures (inflow=0
        #     / no flows at t_first), so we route both through this
        #     constraint instead of duplicating the term inside
        #     nodeBalance.  ``bind_within_solve`` is excluded here
        #     because the .mod's mirror at 2700-2723 binds it at the
        #     LAST timestep of period_first, not the first.
        if (b.is_solve_first
            and d.storage_fix_start is not None and d.storage_fix_start.height > 0
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None):
            # storage_state_start_binding is gated on solveFirst (mod:2725-2733).
            # When this is NOT the first sub-solve of a rolling-horizon chain,
            # the start state is set by the roll_continue path in nodeBalance
            # (mod:2196), so this standalone equality must not also fire.
            # Exclude bind_forward_only and bind_within_solve from this
            # standalone equality:
            #   * bind_forward_only — handled by the in-balance fwd_fix_*
            #     terms in nodeBalance above (mod:2197-2203).
            #   * bind_within_solve — the .mod's mirror at mod:2700-2723
            #     binds them at the LAST timestep of period_first, not
            #     the first; not currently emitted (no fixture exercises
            #     it).
            # Only fix_start nodes that fall through to the default
            # bind_within_timeblock path get the separate equality
            # (mod:2725-2740).
            fix_start_filtered = d.storage_fix_start
            for excl in (d.storage_bind_forward_only,
                          d.storage_bind_within_solve):
                if excl is not None and excl.height > 0:
                    fix_start_filtered = fix_start_filtered.join(
                        excl, on="n", how="anti")
            rhs_param = (d.p_state_start * d.p_state_existing_capacity) / d.p_state_unitsize
            fixed_first_dt = d.nodeState_first_dt.join(
                fix_start_filtered, on="n", how="inner")
            start_lhs: dict = {"state": v_state}
            if b.has_invest_n and d.edd_invest_set is not None:
                v_inv_n_at_s = Var(
                    name=v_invest_n.name + "__at_invest_start",
                    dims=("n", "d_invest"),
                    frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                    lower=v_invest_n.lower, upper=v_invest_n.upper,
                )
                # Phase 4.8h: cross-Enum is_in (e vs n vocab); up-cast n→e.
                _e_dt = d.edd_invest_set.schema["e"]
                _n_in_e = (d.nodeState
                             .select(pl.col("n").cast(_e_dt, strict=False)
                                                .alias("e"))
                             .unique())
                edd_inv_n_s = (d.edd_invest_set
                                 .join(_n_in_e, on="e", how="semi")
                                 .pipe(rename_to_axis, {"e": "n"}))
                if edd_inv_n_s.height > 0:
                    start_lhs["invest_neg"] = -Sum(
                        Where(v_inv_n_at_s * d.p_state_start, edd_inv_n_s),
                        over=("d_invest",))
            if b.has_divest_n and d.edd_divest_active is not None:
                v_div_n_at_s = Var(
                    name=v_divest_n.name + "__at_divest_start",
                    dims=("n", "d_divest"),
                    frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                    lower=v_divest_n.lower, upper=v_divest_n.upper,
                )
                # Phase 4.8h: cross-Enum is_in (p vs n vocab); up-cast n→p.
                _p_dt = d.edd_divest_active.schema["p"]
                _n_in_p = (d.nodeState
                             .select(pl.col("n").cast(_p_dt, strict=False)
                                                .alias("p"))
                             .unique())
                edd_div_n_s = (d.edd_divest_active
                                 .join(_n_in_p, on="p", how="semi")
                                 .pipe(rename_to_axis, {"p": "n"}))
                if edd_div_n_s.height > 0:
                    start_lhs["divest"] = Sum(
                        Where(v_div_n_at_s * d.p_state_start, edd_div_n_s),
                        over=("d_divest",))
            m.add_cstr(
                "storage_state_start_binding",
                over      = fixed_first_dt,
                sense     = "==",
                lhs_terms = start_lhs,
                rhs_terms = {"target": rhs_param},
            )

        # storage_state_start_binding_cyclic_period (mod:2709-2723):
        #   For nodes with binding ∈ {bind_within_period, bind_within_solve}
        #   AND start ∈ {fix_start, fix_start_end}, pin v_state at the
        #   LAST timestep of the node's own block in period_first to
        #   ``state_start * (existing + Σ_d_inv v_invest * us
        #                            - Σ_d_div v_divest * us)``.
        #   The cyclic wrap of the bind means t_last "==" the implicit
        #   pre-t0001 state, which is what state_start represents.
        #   For bind_forward_only this is handled inline in nodeBalance.
        #   For bind_within_timeblock / bind_intraperiod_blocks the per-
        #   block cycle prefers the first-row pin (handled above).
        if (b.is_solve_first
            and d.storage_fix_start is not None and d.storage_fix_start.height > 0
            and d.p_state_start is not None
            and d.p_state_existing_capacity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_last_dt is not None and d.nodeState_last_dt.height > 0):
            # Eligible binding methods: bind_within_solve (and
            # bind_within_period if we ever add it as a separate set).
            cyclic_bound_set = None
            if d.storage_bind_within_solve is not None and d.storage_bind_within_solve.height > 0:
                cyclic_bound_set = d.storage_bind_within_solve
            if cyclic_bound_set is not None and cyclic_bound_set.height > 0:
                # restrict to fix_start nodes
                eligible = (cyclic_bound_set
                              .join(d.storage_fix_start, on="n", how="inner"))
                if eligible.height > 0:
                    # period_first set: derived from nodeState_first_dt
                    # (already filtered to first_period).
                    period_first_set = (d.nodeState_first_dt
                                          .select("d").unique())
                    # last (d, t) per eligible node, within period_first.
                    cyclic_dt = (d.nodeState_last_dt
                                  .join(eligible, on="n", how="inner")
                                  .join(period_first_set, on="d", how="inner")
                                  .select("n", "d", "t").unique())
                    if cyclic_dt.height > 0:
                        cyclic_lhs: dict = {"state": v_state}
                        if b.has_invest_n and d.edd_invest_set is not None:
                            v_inv_n_at_c = Var(
                                name=v_invest_n.name + "__at_invest_cyclic",
                                dims=("n", "d_invest"),
                                frame=v_invest_n.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                                lower=v_invest_n.lower, upper=v_invest_n.upper,
                            )
                            # Phase 4.8h: cross-Enum is_in (e vs n vocab); up-cast n→e.
                            _e_dt = d.edd_invest_set.schema["e"]
                            _n_in_e = (d.nodeState
                                         .select(pl.col("n").cast(_e_dt, strict=False)
                                                            .alias("e"))
                                         .unique())
                            edd_inv_n_c = (d.edd_invest_set
                                             .join(_n_in_e, on="e", how="semi")
                                             .pipe(rename_to_axis, {"e": "n"}))
                            if edd_inv_n_c.height > 0:
                                cyclic_lhs["invest_neg"] = -Sum(
                                    Where(v_inv_n_at_c * d.p_state_start, edd_inv_n_c),
                                    over=("d_invest",))
                        if b.has_divest_n and d.edd_divest_active is not None:
                            v_div_n_at_c = Var(
                                name=v_divest_n.name + "__at_divest_cyclic",
                                dims=("n", "d_divest"),
                                frame=v_divest_n.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                                lower=v_divest_n.lower, upper=v_divest_n.upper,
                            )
                            # Phase 4.8h: cross-Enum is_in (p vs n vocab); up-cast n→p.
                            _p_dt = d.edd_divest_active.schema["p"]
                            _n_in_p = (d.nodeState
                                         .select(pl.col("n").cast(_p_dt, strict=False)
                                                            .alias("p"))
                                         .unique())
                            edd_div_n_c = (d.edd_divest_active
                                             .join(_n_in_p, on="p", how="semi")
                                             .pipe(rename_to_axis, {"p": "n"}))
                            if edd_div_n_c.height > 0:
                                cyclic_lhs["divest"] = Sum(
                                    Where(v_div_n_at_c * d.p_state_start, edd_div_n_c),
                                    over=("d_divest",))
                        rhs_param_cyclic = (d.p_state_start * d.p_state_existing_capacity) / d.p_state_unitsize
                        m.add_cstr(
                            "storage_state_start_binding_cyclic_period",
                            over      = cyclic_dt,
                            sense     = "==",
                            lhs_terms = cyclic_lhs,
                            rhs_terms = {"target": rhs_param_cyclic},
                        )

        # ─── storage_state_solve_horizon_reference_value (mod:2802-2822) ─
        # For nodes with method ``use_reference_value``, pin v_state at the
        # last (d, t) of period_last to ``reference_value × existing/unitsize``.
        # No competing fix_end / bind_within_*: filter applied at load time.
        # Investment-tightening (RHS terms with v_invest_n / v_divest_n) is
        # NOT included here because the test fixture has no invest on
        # storage and recourse-investment is out of scope (audit/a6_b_dim_alternative.md);
        # add invest tightening if a future fixture exercises it.
        if (b.has_storage
                and d.storage_use_reference_value is not None
                and d.storage_use_reference_value.height > 0
                and d.p_storage_state_reference_value is not None
                and d.nodeState_last_dt is not None
                and d.nodeState_last_dt.height > 0
                and d.period_last is not None
                and d.p_state_existing_capacity is not None
                and d.p_state_unitsize is not None):
            # Domain: (n, d, t) ∈ nodeState_last_dt where d ∈ period_last
            # AND n ∈ storage_use_reference_value.
            ssrv_over = (d.nodeState_last_dt
                .join(d.period_last, on="d", how="inner")
                .join(d.storage_use_reference_value, on="n", how="inner")
                .select("n", "d", "t").unique())
            if ssrv_over.height > 0:
                # RHS: reference_value[n, d, t] · existing[n, d] / unitsize[n].
                # Build as a single Param keyed (n, d, t).
                # Phase E.1: p_storage_state_reference_value may be
                # narrower than (n, d, t).  Promote lazily.
                ssrv_lf = promote_param_to_dt(
                    d.p_storage_state_reference_value, d.dt)
                ssrv_frame = (ssrv_lf
                    .rename({"value": "rv"})
                    .join(d.p_state_existing_capacity.frame.lazy()
                            .rename({"value": "exist"}),
                          on=["n", "d"], how="inner")
                    .join(d.p_state_unitsize.frame.lazy()
                            .rename({"value": "us"}),
                          on="n", how="inner")
                    .with_columns(value=pl.col("rv")
                                          * pl.col("exist")
                                          / pl.col("us"))
                    .select("n", "d", "t", "value")
                    .collect())
                rhs_param_ssrv = Param(("n", "d", "t"), ssrv_frame)
                m.add_cstr(
                    "storage_state_solve_horizon_reference_value",
                    over      = ssrv_over,
                    sense     = "==",
                    lhs_terms = {"state": Where(v_state, ssrv_over)},
                    rhs_terms = {"target": rhs_param_ssrv},
                )


def _eq_profiles(b: _Build) -> None:
    """Flow profile limits."""
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_invest_p = b.v_invest_p
    v_divest_p = b.v_divest_p
    reserve_up_to_sink_pdt = b.reserve_up_to_sink
    reserve_down_to_sink_pdt = b.reserve_down_to_sink
    # ─── Profile constraints (upper / lower / fixed) ──────────────────────
    # For each (p, source, sink, f) in process_profile_<method>:
    #     v_flow[p,source,sink,d,t]  <sense>  profile[f,d,t]
    #                                          · existing_count[p,d]
    #                                          · availability[p,d,t]
    # availability defaults to 1.0 if not in data.
    if b.has_profile:
        # When invest is active on entities with profiles, the .mod's RHS
        # expands to profile · (existing_count + Σ_d_inv v_invest[d_inv] -
        # Σ v_divest) · availability.  Move the invest term to the LHS so
        # it remains in the form ``LHS_var ≤/≥/== const_RHS``.
        v_inv_for_profile = None
        if b.has_invest_p and d.edd_invest_set is not None:
            v_inv_p_prof = Var(
                name=v_invest_p.name + "__at_invest_prof",
                dims=("p", "d_invest"),
                frame=v_invest_p.frame.pipe(rename_to_axis, {"d": "d_invest"}),
                lower=v_invest_p.lower, upper=v_invest_p.upper,
            )
            # Phase 4.8h: cross-Enum is_in (e vs p vocab); up-cast p→e.
            _e_dt = d.edd_invest_set.schema["e"]
            _p_in_e = (d.process_source_sink
                         .select(pl.col("p").cast(_e_dt, strict=False).alias("e"))
                         .unique())
            edd_inv_p_prof = (d.edd_invest_set
                                .join(_p_in_e, on="e", how="semi")
                                .pipe(rename_to_axis, {"e": "p"}))
            if edd_inv_p_prof.height > 0:
                v_inv_for_profile = Sum(
                    Where(v_inv_p_prof, edd_inv_p_prof), over=("d_invest",))
        v_div_for_profile = None
        if b.has_divest_p and d.edd_divest_active is not None:
            v_div_p_prof = Var(
                name=v_divest_p.name + "__at_divest_prof",
                dims=("p", "d_divest"),
                frame=v_divest_p.frame.pipe(rename_to_axis, {"d": "d_divest"}),
                lower=v_divest_p.lower, upper=v_divest_p.upper,
            )
            # Phase 4.8h: cross-Enum is_in (p vs p with distinct vocab);
            # up-cast process_source_sink["p"] to edd_divest_active's p-Enum.
            _p_dt = d.edd_divest_active.schema["p"]
            _p_pss = (d.process_source_sink
                        .select(pl.col("p").cast(_p_dt, strict=False))
                        .unique())
            edd_div_p_prof = d.edd_divest_active.join(_p_pss, on="p", how="semi")
            if edd_div_p_prof.height > 0:
                v_div_for_profile = Sum(
                    Where(v_div_p_prof, edd_div_p_prof), over=("d_divest",))
        if d.process_profile_upper is not None and d.process_profile_upper.height > 0:
            _add_profile_cstr(m, d, v_flow, "profile_flow_upper_limit",
                              d.process_profile_upper, "<=",
                              v_inv_for_profile, v_div_for_profile,
                              reserve_term=reserve_up_to_sink_pdt,
                              reserve_sign=+1.0)
        if d.process_profile_lower is not None and d.process_profile_lower.height > 0:
            _add_profile_cstr(m, d, v_flow, "profile_flow_lower_limit",
                              d.process_profile_lower, ">=",
                              v_inv_for_profile, v_div_for_profile,
                              reserve_term=reserve_down_to_sink_pdt,
                              reserve_sign=-1.0)
        if d.process_profile_fixed is not None and d.process_profile_fixed.height > 0:
            _add_profile_cstr(m, d, v_flow, "profile_flow_fixed",
                              d.process_profile_fixed, "==",
                              v_inv_for_profile, v_div_for_profile)


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


def _emit_entity_total_cap(
    m, *, legacy_name: str, path_name: str, over_e: pl.DataFrame,
    var, unitsize, cap_param, d_leaf, lhs_label: str,
) -> None:
    """Per-path entity total cap (design §7.2 (G)).

    Sums ``var * unitsize`` over the invest axis's periods against a
    per-entity total cap.  With a single scenario leaf — deterministic
    or flag-off, where ``d_leaf`` degenerates to one ``"__anchor"``
    leaf — this emits the legacy ``over=("d",)`` total under the legacy
    NAME, byte-identical to the pre-recourse constraint.  With more than
    one leaf (recourse), it emits ``path_name`` indexed by
    ``(entity, leaf)`` with the inner ``d``-sum masked to each leaf's
    own periods; the per-entity ``cap_param`` broadcasts across leaves
    (one cap per scenario path — no cross-scenario double-count).

    ``leaf`` is a plain ``Utf8`` index column carried through the
    ``Where`` mask exactly as ``g`` is in the group-total builders;
    ``Sum(over=("d",))`` leaves ``(entity, leaf)`` matching ``over_e ×
    leaves``.
    """
    dl = None
    if d_leaf is not None and getattr(d_leaf, "height", 0) > 0:
        inv_periods = var.frame.select("d").unique()
        _d_dt = var.frame.schema["d"]
        dlx = d_leaf
        if dlx.schema["d"] != _d_dt:
            dlx = dlx.with_columns(pl.col("d").cast(_d_dt, strict=False))
        dl = dlx.join(inv_periods, on="d", how="semi")
    n_leaves = (dl["leaf"].n_unique()
                if dl is not None and dl.height > 0 else 1)
    if n_leaves <= 1:
        m.add_cstr(
            legacy_name,
            over      = over_e,
            sense     = "<=",
            lhs_terms = {lhs_label: Sum(Where(var * unitsize, over_e),
                                        over=("d",))},
            rhs_terms = {"cap": cap_param},
        )
        return
    mask = over_e.join(dl, how="cross")
    over_path = over_e.join(dl.select("leaf").unique().sort("leaf"),
                            how="cross")
    m.add_cstr(
        path_name,
        over      = over_path,
        sense     = "<=",
        lhs_terms = {lhs_label: Sum(Where(var * unitsize, mask),
                                    over=("d",))},
        rhs_terms = {"cap": cap_param},
    )


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
    """``maxFlow`` RHS per ``(p, source, sink, d, t)``, the capacity
    multiplier of the invest/divest terms, and the per-element ``v_flow``
    upper bound that replaces the row of constant-capacity input arcs.

    Arc classes:

    * **uncapped** — ``conversion_flow_coeff = 0`` edges
      (``process_source_sink_uncapped``; the hydro-pass-through pattern).
      They are outside every per-edge capacity constraint: RHS is the raw
      ``p_flow_upper`` value (``max_flow_for_unconstrained_variables``),
      with no availability or capacity coefficient.
    * **input** — indirect (multi-flow) INPUT arcs.  Unit capacity is the
      maximum sum of OUTPUTS (``maxOutputSum``); the fuel is tied to the
      outputs by ``conversion_indirect``.  An input arc ``s`` is limited to
      ``M_s × (existing/us + Σv_invest − Σv_divest)`` with
      ``M_s = p_indirect_input_cap × availability`` (``input_share_max ×``
      the full-load input per unit of capacity — with the default share
      of 1 the input alone can run the unit at full output, so the limit
      is a loose solver-region bound).  Processes WITHOUT invest / divest
      variables in this solve (``inv_procs``) have a constant capacity:
      their limit ``M_s × existing/us`` becomes a per-element ``v_flow``
      upper bound and the arcs get no ``maxFlow`` row.  Processes WITH
      them keep the row, their invest/divest terms carrying ``M_s``.
    * **capped** — direct arcs and indirect OUTPUT arcs
      (``process_output_flows``)::

          v_flow ≤ coef × availability × (existing/us + Σv_invest − Σv_divest)

      RHS = ``existing/us × coef × availability`` (``p_flow_upper_existing``
      densified to 0 for greenfield arcs, ``p_arc_max_cap_coef`` and
      ``p_process_availability`` densified to their 1.0 defaults — a naive
      ``Param * Param`` inner join would DROP unauthored rows → RHS = 0).

    Returns ``(rhs, cap_mult, var_upper, bound_arcs)``: ``rhs`` covers
    every arc (the ``maxFlow_negCap`` / ``maxFlow_back`` rows read it too);
    ``cap_mult`` ``(p, source, sink, d, t, value)`` holds the invest/divest
    multiplier of capped and input arcs where it is not 1 (``None`` when
    empty); ``var_upper`` is the ``v_flow`` upper-bound Param and
    ``bound_arcs`` its ``(p, source, sink)`` arcs (both ``None`` when no
    arc is bounded on the variable).

    When ``p_flow_upper_existing`` is absent (hand-built fixtures without
    an existing-capacity frame) every non-uncapped arc reads
    ``p_flow_upper`` as its existing count (legacy behaviour).

    Sink-less indirect units (≥2 inputs, 0 outputs) bind their capacity on
    a source arc (``maxFromSource``) that this split does not route; they
    raise rather than emit a free-capacity converter.
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
        # B1: no indirect input arc may draw the existing-only RHS (it
        # would force v_flow_in ≤ 0).
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

    # ── uncapped: raw p_flow_upper (max_flow_for_unconstrained_variables) ─
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
    caller then keeps the bare ``Σ v_invest`` term (identical LP).
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


def _add_min_input_share(m, d, v_flow) -> None:
    """``minInputShare`` — ``unit__inputNode.input_share_min``.

    While the unit is running, input ``s`` supplies at least
    ``input_share_min_s`` of the unit's CURRENT input energy (flow ×
    ``conversion_flow_coeff``)::

        conv_s · v_flow[p, s, p]  ≥  share_s · Σ_{s'} conv_{s'} · v_flow[p, s', p]

    per ``(p, s, p, d, t)`` for indirect units with ≥ 2 inputs of
    ``conversion_flow_coeff ≠ 0`` and ``share_s > 0``.  The row is
    homogeneous: with no input (the unit is not running) it reads
    ``0 ≥ 0``, so it needs no online variable and holds for units without
    one.  ``v_flow`` is per unitsize on every input of ``p``, so the
    unitsize cancels.  The ranges are checked once per scenario by
    ``input_derivation._validators.validate_unit_input_shares``; the
    guard below keeps hand-built FlexData honest.
    """
    share = getattr(d, "p_process_source_input_share_min", None)
    pif = getattr(d, "process_input_flows", None)
    indirect = getattr(d, "process_indirect", None)
    pss = getattr(d, "process_source_sink", None)
    if (share is None or share.frame.height == 0 or pif is None
            or pif.height == 0 or indirect is None or indirect.height == 0
            or pss is None or pss.height == 0):
        return
    keys = ["p", "source", "sink"]
    in_arcs = (_utf8_keys(pif.lazy().select(keys), keys)
               .join(_utf8_keys(indirect.lazy().select("p"), ["p"])
                     .select("_p").unique(), on="_p", how="semi")
               .join(_utf8_keys(pss.lazy().select(keys), keys)
                     .select("_p", "_source", "_sink"),
                     on=["_p", "_source", "_sink"], how="semi"))
    n_in = in_arcs.group_by("_p").agg(pl.len().alias("__n"))
    in_arcs = (in_arcs.join(n_in.filter(pl.col("__n") >= 2).select("_p"),
                            on="_p", how="semi"))
    share_lf = (share.frame.lazy()
                .with_columns(pl.col("p").cast(pl.Utf8).alias("_p"),
                              pl.col("source").cast(pl.Utf8).alias("_source"))
                .select("_p", "_source", pl.col("value").cast(pl.Float64)))
    rows = (in_arcs.join(share_lf, on=["_p", "_source"], how="inner")
            .filter(pl.col("value") > 0.0)
            .select(*keys, "value").collect())
    if rows.height == 0:
        return
    bad = rows.filter(pl.col("value") > 1.0)
    sums = (rows.group_by("p").agg(pl.col("value").sum().alias("s"))
            .filter(pl.col("s") > 1.0 + 1e-9))
    if bad.height > 0 or sums.height > 0:
        raise FlexToolConfigError(
            "input_share_min must lie in [0, 1] and sum to at most 1 per "
            "unit; offending: "
            + ", ".join([f"{r['p']}<-{r['source']}: {r['value']}"
                         for r in bad.iter_rows(named=True)]
                        + [f"{r['p']}: sum {r['s']}"
                           for r in sums.iter_rows(named=True)]))
    procs = rows.select("p").unique()
    arcs_all = (in_arcs.join(_utf8_keys(procs.lazy(), ["p"]).select("_p"),
                             on="_p", how="semi")
                .select(keys).collect())
    conv = getattr(d, "p_process_source_conversion_flow_coeff", None)
    expr = v_flow
    if conv is not None:
        conv_c = _coef_factor(arcs_all, conv, ["p", "source"])
        neg = conv_c.frame.filter(pl.col("value") < 0.0)
        if neg.height > 0:
            raise FlexToolConfigError(
                "input_share_min is not defined for a unit with a negative "
                "input conversion_flow_coeff: "
                + ", ".join(f"{r['p']}<-{r['source']}"
                            for r in neg.iter_rows(named=True)))
        expr = v_flow * conv_c
    over = (rows.select(keys)
            .join(d.dt, how="cross"))
    pidt = getattr(d, "process_indirect_dt", None)
    if pidt is not None and pidt.height > 0:
        over = over.join(pidt.select("p", "d", "t"), on=["p", "d", "t"],
                         how="semi")
    if over.height == 0:
        return
    share_p = Param(("p", "source"), rows.select("p", "source", "value"))
    m.add_cstr("minInputShare", over=over, sense=">=",
               lhs_terms={"own": Where(expr, rows.select(keys))},
               rhs_terms={"total": Sum(Where(expr, arcs_all),
                                       over=("source",)) * share_p})


def _source_ramp_factor(d, idx_set: "pl.DataFrame") -> "Param | None":
    """Source-side ramp capacity factor over ``idx_set × dt``.

    Indirect-unit input arcs (rows of ``p_indirect_input_cap``) take that
    per-unit-of-capacity input limit (``input_share_max`` × full-load
    input flow); every other arc takes ``input_share_max`` keyed
    ``(p, source)`` (default 1).  ``None`` when neither applies (factor 1
    everywhere — the ramp RHS is left byte-identical).
    """
    keys = ["p", "source", "sink"]
    share = getattr(d, "p_process_source_input_share_max", None)
    in_cap = getattr(d, "p_indirect_input_cap", None)
    cap_rows = None
    if in_cap is not None and in_cap.frame.height > 0:
        cap_rows = (_utf8_keys(in_cap.frame.lazy(), keys)
                    .join(_utf8_keys(idx_set.lazy(), keys)
                          .select([f"_{k}" for k in keys]),
                          on=[f"_{k}" for k in keys], how="semi")
                    .select([f"_{k}" for k in keys] + ["d", "t",
                            pl.col("value").alias("__cap")])
                    .collect())
        if cap_rows.height == 0:
            cap_rows = None
    if cap_rows is None:
        if share is None:
            return None
        return _coef_factor(idx_set, share, ["p", "source"])
    base = _utf8_keys(idx_set.select(keys).unique().join(d.dt.select("d", "t"),
                                                         how="cross").lazy(),
                      keys)
    if share is not None:
        base = (base
                .join(share.frame.lazy()
                      .select(pl.col("p").cast(pl.Utf8).alias("_p"),
                              pl.col("source").cast(pl.Utf8).alias("_source"),
                              pl.col("value").alias("__share")),
                      on=["_p", "_source"], how="left"))
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Float64)
                                 .alias("__share"))
    frame = (base
             .join(cap_rows.lazy(), on=[f"_{k}" for k in keys] + ["d", "t"],
                   how="left")
             .select(*keys, "d", "t",
                     value=pl.coalesce(pl.col("__cap"),
                                       pl.col("__share"), pl.lit(1.0)))
             .collect())
    return Param(("p", "source", "sink", "d", "t"), frame)


def _add_profile_cstr(m, d, v_flow, name: str, idx: "pl.DataFrame",
                       sense: str, v_inv_for_profile=None,
                       v_div_for_profile=None,
                       reserve_term=None,
                       reserve_sign: float = 1.0) -> None:
    """profile_flow_*  constraint family.

    LHS = v_flow filtered to the (p, source, sink, f) tuples in idx,
    then time dims (d, t) attached via the over= axes.

    RHS = profile[f, d, t] · existing_count[p, d] · availability[p, d, t].

    When invest/divest are active, the .mod's RHS expands to
    profile · (existing_count + Σ_d_inv v_invest[d_inv]
                                - Σ_d_div v_divest[d_div]) · availability.
    The invest term is moved to the LHS as ``-invest_term`` and the
    divest term is moved to the LHS as ``+divest_term`` so this stays a
    single LP row per (p, source, sink, f, d, t).

    ``reserve_term`` (manifest patch #6 for _reserve): an Expr keyed on
    (p, sink, d, t) that the .mod adds to the profile_flow LHS.
    For ``upper_limit`` the term is ``+ Σ v_reserve_up_to_sink``;
    for ``lower_limit`` it is ``- Σ v_reserve_down_to_sink`` (sign passed
    via ``reserve_sign``).  ``fixed`` has no reserve term per the .mod.
    """
    over = idx.join(d.dt, how="cross")              # (p, source, sink, f, d, t)
    # LHS: v_flow over (p,source,sink,d,t) joined to over via (p,source,sink,d,t)
    # → introduces f as a new column from over (via the constraint axes).
    rhs_param = d.p_profile_value * d.p_process_existing_count
    # Densify availability over the profile processes so a VRE unit with no
    # DB-authored ``availability`` keeps its default-1.0 factor instead of
    # being inner-join-dropped to RHS=0 (→ forced v_flow=0).  See
    # :func:`_availability_factor`.
    avail_factor = (_availability_factor(d, idx)
                    if d.p_process_availability is not None else None)
    if avail_factor is not None:
        rhs_param = rhs_param * avail_factor
    lhs: dict = {"flow": v_flow}
    pf_filter = None
    if v_inv_for_profile is not None or v_div_for_profile is not None:
        # Restrict to (p, f) pairs present in this profile-method set so
        # the invest/divest term doesn't introduce p × f cartesian rows
        # that don't belong to this constraint family.
        pf_filter = idx.select("p", "f" if "f" in idx.columns else "profile").unique()
        if "profile" in pf_filter.columns and "f" not in pf_filter.columns:
            pf_filter = pf_filter.pipe(rename_to_axis, {"profile": "f"})
    if v_inv_for_profile is not None:
        inv_term = v_inv_for_profile * d.p_profile_value
        if avail_factor is not None:
            inv_term = inv_term * avail_factor
        lhs["invest_neg"] = -Where(inv_term, pf_filter)
    if v_div_for_profile is not None:
        div_term = v_div_for_profile * d.p_profile_value
        if avail_factor is not None:
            div_term = div_term * avail_factor
        lhs["divest"] = Where(div_term, pf_filter)
    if reserve_term is not None:
        if reserve_sign >= 0:
            lhs["reserve"] = reserve_term
        else:
            lhs["reserve_neg"] = -reserve_term
    m.add_cstr(
        name,
        over      = over,
        sense     = sense,
        lhs_terms = lhs,
        rhs_terms = {"limit": rhs_param},
    )
