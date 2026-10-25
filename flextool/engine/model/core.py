"""FlexTool LP model: builds the FlexTool LP from FlexData ``d`` into a polar_high Problem ``m``.

:func:`build_flextool` is the table of contents: ``_prepare`` (flags, data checks, index sets),
``_declare_variables``, ``_shared_expressions``, then one step per equation family in emit order; the
objective is the sum of the terms the ``_obj_*`` / ``objective_terms`` steps return.  Every step takes the
``_Build`` context ``b``.  The core model that every run uses lives here as ``_eq_<family>`` /
``_obj_<family>``.  An optional feature that brings its own variables lives in its own module with
``add_variables(b)`` / ``add_constraints(b)`` / ``objective_terms(b)``.  Shared helpers: ``_common``.

Each row, objective term and variable has an ``EQ.`` / ``OBJ.`` / ``VAR.`` header with its formula;
grep ``EQ. ─── <name>`` to find an equation.

Variables, in declaration (column) order.  p process, n node, d period, t step, b base period, g group,
r reserve, ud up/down, c commodity, i price tier.  "× unitsize" converts to MW / MWh.
  v_flow[p,source,sink,d,t]            flow on a process arc; × unitsize = MW
  vq_state_up / _down[n,d,t]           balance slack (shortfall / surplus), MW
  v_state[n,d,t]                       storage state; × state_unitsize = MWh
  v_state_inter[n,b]                   representative-period storage state at base-period boundaries
  v_state_rp_start[n,d,t]              free start state of each representative block
  v_{online,startup,shutdown}_{linear,integer}[p,d,t]
                                       units online / starting / stopping, count of unitsize  [_online]
  v_{invest,divest}_{p,n}[p|n,d]       capacity built / retired in d, count of unitsize
  v_reserve[p,r,ud,n,d,t]              reserve held by p for n; × unitsize = MW               [_reserve]
  vq_reserve[r,ud,g,d,t]               reserve shortfall, fraction of the requirement scale   [_reserve]
  v_reserve_total[r,ud,g,d,t]          total reserve of a group with n-1 rows, MW             [_reserve]
  v_flow_back[p,source,sink,d,t]       reverse flow of 2-way 1-var and DC arcs; × unitsize = MW
  v_angle[n,d,t]                       voltage angle, rad in [−π, π]                          [_dc_power_flow]
  v_trade[c,n,d,i]                     commodity bought in tier i; × commodity unitsize = MWh [_commodity_ladder]
  vq_capacity_margin[g,d], vq_inertia[g,d,t], vq_non_synchronous[g,d,t]
                                       group floor slacks, fraction of the group's scale      [_group_slack]

Equation families, in emit order (module in brackets when not model/core.py):
  Balance & storage   rp_inter_period_{balance,cyclic,max_state}, maxState_rp_start; nodeBalance_eq;
                      nodeBalancePeriod_eq; node_balance_fix_quantity_eq_lower; node_storage_usage_fix_le;
                      stateConstantWithinBlock_eq, nodeBalanceBlock_eq
  Flows & capacity    maxFlow, maxFlow_negCap, maxFlow_back; maxOutputSum(_negCap);
                      dc_reference_angle_eq, dc_flow_eq [_dc_power_flow]
  Units               max{Online,Startup,Shutdown}, online__{startup,shutdown}, maxFlow_online,
                      maxOutputSum_online, maxInputFlow_online, minimum_{uptime,downtime}, online_fix_eq
                      [_online];
                      minFlow_{minload,output_floor,reserve,reserve_arc} [_unit_floors];
                      ramp_{sink,source}_{up,down}_constraint
  Investment          max{Invest,Divest}_var_bound[_n], _entity_period_{p,n}, _entity_total[_path][_n]
  Conversion          conversion_indirect (delayed inputs from [_delay]); minInputShare
  Emissions & user    co2_max_period, co2_max_total; process_constraint_{equal,less_than,greater_than}
  Storage state       maxState; profile_state_{upper_limit,lower_limit,fixed};
                      storage_state_start_binding[_cyclic_period]; storage_state_solve_horizon_reference_value
  Profiles            profile_flow_{upper_limit,lower_limit,fixed}
  Objective           balance slack penalty, commodity cost and sales, CO2 price, variable cost, startup cost,
                      storage reference price, invest/divest annuities, existing fixed cost (optional constant)
  Feature modules     capacityMargin, inertia_constraint, non_sync_constraint + slack penalties [_group_slack];
                      reserveBalance_*, reserve_total_upper, reserve_process_*, reserve_storage_*
                      + shortfall penalty [_reserve];
                      investment floors, cumulative, group and flow-group limits [_cumulative_invest];
                      commodity_ladder_balance, ladder_tier_cap_* + tier costs [_commodity_ladder]
  Stochastic ties     non_anticipativity_* (dispatch, then investment) [_non_anticipativity]
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from polar_high import Sum, Where, Lag, Param
from polar_high.engine import Expr, Var

from flextool.engine.model import _commodity_ladder
from flextool.engine.model import _cumulative_invest
from flextool.engine.model import _dc_power_flow
from flextool.engine.model import _delay
from flextool.engine.model import _group_slack
from flextool.engine.model import _non_anticipativity
from flextool.engine.model import _online
from flextool.engine.model import _reserve
from flextool.engine.model import _unit_floors
from flextool.engine.input._axis_enums import alias_to_axis, rename_to_axis
from flextool.engine.model._common import (
    _availability_factor,
    _coef_factor,
    _drop_uncapped,
    _entity_rows,
    _has_rows,
    _indirect_sum_arcs,
    _utf8_keys,
    _var_view,
)
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.model._pdt_join import (
    compute_pss_dt,
    compute_nodeBalance_dt,
    compute_nodeState_dt,
    compute_nodeState_rp_block_first_dt,
    compute_process_indirect_dt,
)
from flextool.engine.input._solve_state import FlexToolConfigError

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
                                "group_d_co2_capped", "p_window_fraction")
CO2_CAP_TOTAL: tuple[str, ...] = ("p_co2_content", "p_co2_max_total",
                                   "group_co2_max_total", "p_years_represented_d")
USER_CSTR: tuple[str, ...]   = ("p_constraint_constant",)
PROFILES:  tuple[str, ...]   = ("p_profile_value", "p_process_existing_count")
STORAGE:   tuple[str, ...]   = ("nodeState", "dtttdt",
                                "p_state_upper", "p_state_unitsize")
ONLINE: tuple[str, ...] = (
    "process_online", "p_online_dt",
    "dtttdt", "p_process_existing_count",
)
# The minFlow_minload floor's field: checked when some unit has min_load > 0 (process_minload), so a loader gap
# does not drop the floor silently.  An online unit without min_load has none (min_load 0).
MIN_LOAD: tuple[str, ...] = ("p_min_load",)
MINLOAD_EFF: tuple[str, ...] = ("process_min_load_eff", "p_section")
STARTUP_COST_LINEAR: tuple[str, ...] = ("p_startup_cost", "pdt_online_linear")
STARTUP_COST_INTEGER: tuple[str, ...] = ("p_startup_cost", "pdt_online_integer")
RAMP: tuple[str, ...] = ("dt_ramp_link", "p_step_duration", "p_process_existing_count")
# Checked whenever a group has capacity_margin_method = manual: without period_invest or the group scaling the
# capacity margin would be skipped without a word.
CAPACITY_MARGIN: tuple[str, ...] = ("period_invest", "p_inv_group_cap")
INVEST: tuple[str, ...] = ("p_entity_max_units",)
DIVEST: tuple[str, ...] = ("p_entity_max_units", "edd_divest_active",
                            "pd_divest_set")


def _check_covers_dt(d, field: str, feature: str, *, positive: bool = False) -> None:
    """Raise if the (d,)-keyed Param ``field`` lacks a period of ``dt`` (or, with ``positive``, has a
    value that is not finite and above 0).

    co2_max_total multiplies each period's emissions by its years (``p_years_represented_d``) and
    co2_max_period scales each period's cap by its window fraction (``p_window_fraction``); a period
    without a row would silently drop out of the cap, and a zero fraction would set the cap to 0.  The
    loader builds both Params over the dt periods, so a gap is a loader bug."""
    frame = getattr(d, field).frame
    have = set(frame["d"].cast(pl.Utf8).to_list())
    missing = [p for p in d.dt["d"].cast(pl.Utf8).unique(maintain_order=True).to_list() if p not in have]
    if missing:
        raise ValueError(
            f"build_flextool: {field} lacks periods {missing} of dt, so {feature} would drop them; this is "
            f"a loader bug (the Param must cover every dt period).")
    if positive:
        v = frame["value"].cast(pl.Float64)
        bad = frame.filter(~(v.is_finite() & (v > 0)).fill_null(True))
        if bad.height:
            raise ValueError(
                f"build_flextool: {field} must be finite and above 0 for every period, got "
                f"{dict(zip(bad['d'].cast(pl.Utf8).to_list(), bad['value'].to_list()))}; {feature} would "
                f"get a wrong cap.  This is a loader bug.")


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
    the ``_obj_*`` and ``objective_terms`` steps return their terms to
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
    has_reserve: bool = False
    has_capacity_margin: bool = False
    has_inertia: bool = False
    has_non_sync: bool = False
    has_cumulative_invest: bool = False
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
    v_reserve: Var | None = None
    vq_reserve: Var | None = None
    v_reserve_total: Var | None = None
    vq_capacity_margin: Var | None = None
    vq_inertia: Var | None = None
    vq_non_synchronous: Var | None = None
    v_angle: Var | None = None
    v_trade: Var | None = None

    # ── shared expressions ──
    node_balance_flow_terms: dict = field(default_factory=dict)
    reserve_up_to_sink: Any = None          # Σ_r v_reserve up,   (p, sink, d, t)
    reserve_down_to_sink: Any = None        # Σ_r v_reserve down, (p, sink, d, t)
    reserve_up_sink_parts: Any = None       # (v_reserve at sink, up selector)
    # Source-side reserves of units, per arc (p, source, sink, d, t), converted to the arc's flow:
    reserve_down_at_source: dict = field(default_factory=dict)  # down at the source: raises the flow
    reserve_up_at_source: dict = field(default_factory=dict)    # up at the source: lowers the flow
    reserve_down_input_arcs: pl.DataFrame | None = None  # (p, source, sink) carrying reserve_down_at_source
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
    capacity, Σ existing · fixed_cost · inflation_op · pd_branch_weight, to
    the objective as a constant (OBJ. existing_fixed_cost).  It changes the
    objective value, not the solution.  Off by default.
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
        # ── core objective terms ──
        _obj_balance_slack,
        _obj_commodity,
        _obj_co2_price,
        _obj_co2_price_noEff,
        _obj_variable_cost,
        _obj_startup_cost,
        _obj_storage_reference_price,
        _obj_invest_divest,
        _obj_existing_fixed_cost,
        # ── further constraint families and their objective terms ──
        _group_slack.add_constraints,
        _group_slack.objective_terms,
        _reserve.add_constraints,
        _reserve.objective_terms,
        _cumulative_invest.add_constraints,
        _commodity_ladder.add_constraints,
        _commodity_ladder.objective_terms,
        # ── stochastic ties ──
        _non_anticipativity.add_dispatch_constraints,
        _non_anticipativity.add_invest_constraints,
    ):
        _build_prof(f"{step.__module__.rsplit('.', 1)[-1]}.{step.__name__}")
        term = step(b)
        if term is not None:
            obj = term if obj is None else obj + term

    if scale_the_objective != 1.0:
        obj = obj * scale_the_objective
    m.set_objective(obj, sense="min")
    _build_prof("build_flextool:return")


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
    b.has_reserve = _has_rows(getattr(d, "reserve_upDown_group", None))
    b.has_capacity_margin = _group_slack.has_capacity_margin(d)
    b.has_inertia = _group_slack.has_inertia(d)
    b.has_non_sync = _group_slack.has_non_sync(d)
    b.has_cumulative_invest = _cumulative_invest.has_feature(d)
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
        _check_covers_dt(d, "p_window_fraction", "co2_max_period", positive=True)
    if b.has_co2_cap_total:
        _check(d, CO2_CAP_TOTAL, "co2_max_total")
        _check_covers_dt(d, "p_years_represented_d", "co2_max_total")
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
        if _has_rows(getattr(d, "process_minload", None)):
            _check(d, MIN_LOAD, "min_load")
    if b.has_minload_eff:
        _check(d, MINLOAD_EFF, "min_load_efficiency")
    if b.has_startup_cost_lin:
        _check(d, STARTUP_COST_LINEAR, "startup_cost_linear")
    if b.has_startup_cost_int:
        _check(d, STARTUP_COST_INTEGER, "startup_cost_integer")
    if _has_rows(getattr(d, "groupCapacityMargin", None)):
        _check(d, CAPACITY_MARGIN, "capacity_margin")

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

    # Reserve variables (v_reserve, vq_reserve, v_reserve_total): declared by _reserve.
    _reserve.add_variables(b)

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
    _commodity_ladder.add_variables(b)

    # Group soft-floor slacks (vq_capacity_margin, vq_inertia, vq_non_synchronous): declared by _group_slack.
    _group_slack.add_variables(b)


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
    v_reserve = b.v_reserve
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

    # Source-side reserve of units (consume more / consume less), per arc and converted to the arc's
    # v_flow: factor 1 where v_flow is the input (an indirect unit's input arc, a sink-less unit), 1/slope
    # on a direct input→output arc.  Down at the source raises the flow (maxFlow, maxFlow_online,
    # maxInputFlow_online, profile_flow_upper_limit); up at the source lowers it
    # (profile_flow_lower_limit; the floors in _unit_floors).  Connections are not coupled.
    unit_arcs = _unit_floors.all_unit_arcs(b.d)
    if unit_arcs is None:
        return
    at_source = _var_view(v_reserve, "__at_source", {"n": "source"}, to_axis=True)
    down_flow, down_eff = _unit_floors.increasing_source_selectors(b.d, unit_arcs)
    _, up_flow, up_eff = _unit_floors._reducing_reserve_selectors(b.d, unit_arcs)
    down_arcs = []
    for target, sel, direct, key in ((b.reserve_down_at_source, down_flow, False, "reserve_down_source"),
                                     (b.reserve_down_at_source, down_eff, True, "reserve_down_source_eff"),
                                     (b.reserve_up_at_source, up_flow, False, "reserve_up_source"),
                                     (b.reserve_up_at_source, up_eff, True, "reserve_up_source_eff")):
        if sel is None:
            continue
        arcs = _source_reserve_arcs(unit_arcs, sel, direct=direct)
        if arcs.height == 0:
            continue
        term = Where(at_source, sel)
        if direct:
            term = term * _unit_floors._inverse_slope(b.d, sel)
        target[key] = Where(Sum(term, over=("r", "ud")), arcs)
        if target is b.reserve_down_at_source:
            down_arcs.append(arcs)
    if down_arcs:
        b.reserve_down_input_arcs = pl.concat(down_arcs).unique(maintain_order=True)


def _arcs_semi(frame: pl.DataFrame, arcs: pl.DataFrame) -> pl.DataFrame:
    """Rows of ``frame`` whose ``(p, source, sink)`` is in ``arcs`` (Utf8 keys; ``frame``'s columns kept)."""
    keys = ["p", "source", "sink"]
    return (_utf8_keys(frame.lazy(), keys)
            .join(_utf8_keys(arcs.lazy().select(keys), keys).select([f"_{k}" for k in keys]).unique(),
                  on=[f"_{k}" for k in keys], how="semi")
            .select(frame.columns).collect())


def _source_reserve_arcs(unit_arcs: pl.DataFrame, sel: pl.DataFrame, *, direct: bool) -> pl.DataFrame:
    """The unit arcs ``(p, source, sink)`` of the class of ``sel`` (direct input→output arcs, or arcs whose
    v_flow is the input) whose ``(p, source)`` carries a reserve of ``sel``."""
    keys = ["p", "source", "sink"]
    a = _utf8_keys(unit_arcs.lazy(), keys)
    a = a.filter(pl.col("_source") != pl.col("_p"))
    a = a.filter((pl.col("_sink") != pl.col("_p")) if direct else (pl.col("_sink") == pl.col("_p")))
    hit = _utf8_keys(sel.lazy().select("p", "source"), ["p", "source"]).select("_p", "_source").unique()
    return a.join(hit, on=["_p", "_source"], how="semi").select(keys).collect()


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
    #   slope[p,d,t] = (1/eta − min_load/eta_min) / (1 − min_load)
    #   section[p,d,t] ≈ min_load·(1/eta_min − 1/eta) / (1 − min_load)
    #     for min_load_efficiency units (eta = efficiency, default 1; min_load = 0 where not authored;
    #     eta_min = efficiency_at_min_load, else efficiency); slope = 1/eta and no section for the other units
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
    # v_flow[p,s,k,d,t] + Σ_r v_reserve_up[p,r,k,d,t] + Σ_r v_reserve_down[p,r,s,d,t] · φ[p,s,k,d,t]
    #   − M·Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + M·Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    #   ≤ M · existing[p,d] / unitsize[p]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,s,k) ∉ flow_bound_arcs or (p,s,k) ∈ reserve_down_input_arcs
    #   M = capacity_max_coeff[p,s,k] · availability[p,d,t]
    #   φ = 1 where v_flow is the input (k = p), 1/slope[p,d,t] on a direct unit's input→output arc
    #       (s ≠ p ≠ k); the down-reserve term only for units
    # Caps every flow by the built capacity of its process (units of unitsize).
    # On indirect-unit input arcs M = input_share_max · full-load input ·
    # availability; uncapped arcs (conversion_flow_coeff = 0) have M = 1 and
    # RHS = p_flow_upper.  Input arcs of units without invest/divest are
    # bounded on v_flow instead (flow_bound_arcs).  The reserve terms are the
    # reserve that raises the flow: upward reserve held at the sink, and
    # downward reserve held at the source of a unit (it consumes more),
    # converted to the arc's flow.  A bound input arc gets this row as well
    # when it carries such a reserve (its v_flow bound stays, redundant).
    lhs: dict = {"flow": b.v_flow}
    if b.reserve_up_to_sink is not None:
        lhs["reserve_up"] = b.reserve_up_to_sink
    lhs.update(b.reserve_down_at_source)
    if b.divest_in_dispatch_p is not None:
        lhs["divest"] = _scaled(b.divest_in_dispatch_p)
    if b.invest_in_dispatch_p is not None:
        lhs["invest_neg"] = -_scaled(b.invest_in_dispatch_p)
    over = (pss_dt.join(b.flow_bound_arcs, on=["p", "source", "sink"], how="anti")
            if b.flow_bound_arcs is not None else pss_dt)
    if b.flow_bound_arcs is not None and b.reserve_down_input_arcs is not None:
        bound_with_reserve = _arcs_semi(b.flow_bound_arcs, b.reserve_down_input_arcs)
        if bound_with_reserve.height > 0:
            over = pl.concat([over, _arcs_semi(pss_dt, bound_with_reserve)])
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
    link, carry = ramp_step_links(d)

    # ─── EQ. ─── ramp_{sink,source}_{up,down}_constraint ─────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] + R↑[p,s,k,d,t]
    #   ≤ +ρ · c[p,s,k,d,t] · (existing_count[p,d] + Σ_{d″} v_invest_p[p,d″] − Σ_{d″} v_divest_p[p,d″])
    #     + v_startup[p,d,t]                                                                           (up)
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − R↓[p,s,k,d,t]
    #   ≥ −ρ · c[p,s,k,d,t] · (existing_count[p,d] + Σ v_invest_p − Σ v_divest_p) − v_shutdown[p,d,t]    (down)
    #   ρ = ramp_speed_{dir}_{side}[p,node] · 60 · (step_duration[d′,t′] + step_duration[d,t]) / 2
    #   ∀ (p,s,k) ∈ process_source_sink_ramp_limit_{side}_{dir}, (d,t,d′,t′) ∈ dt_ramp_link;
    #     on a link inside one period (d′ = d) only where ρ < 1
    # On a continuation roll the first step of a period ramps from the previous roll's realized flow
    #   (dt_ramp_carry): v_flow[p,s,k,d′,t′] → f̂[p,s,k,d,t] (p_roll_continue_flow, a constant on the RHS) and
    #   step_duration[d′,t′] → its carried duration.
    # Steps without a predecessor, on arcs that hold reserve in that direction there (prundt):
    #   R↑ ≤ ρ·c·cap (up),  −R↓ ≥ −ρ·c·cap (down),  ρ = ramp_speed · 60 · step_duration[d,t] < 1;  no startup term.
    # R↑ = Σ_r v_reserve[p,r,up,k,d,t] + φ·Σ_r v_reserve[p,r,down,s,d,t]   (raises the arc: up at its sink, down at its source)
    # R↓ = Σ_r v_reserve[p,r,down,k,d,t] + φ·Σ_r v_reserve[p,r,up,s,d,t]   (lowers it: down at its sink, up at its source)
    #   φ = 1 where v_flow is the input (indirect unit's input arc, sink-less unit), 1/slope on a direct arc; units only.
    # Limits the change of a flow between two consecutive steps, plus the reserve it holds in that direction, to its
    # ramp room: ramp_speed is a share of capacity per minute and flows are step averages, so the room between two
    # step averages is the speed × the distance between the two step centres (a valid bound for every trajectory
    # that obeys the speed; it is the tight one for a linear ramp).  A step has a predecessor (dt_ramp_link) only
    # where it is adjacent on the timeline in the un-rolled solve: no cyclic wrap at the first step of a solve or of
    # a period that does not continue the previous one, no row across a gap or at the start of a representative
    # block.  Rows inside one period with ρ ≥ 1 cannot bind and are not built: the arc's capacity row (maxFlow*,
    # with the same raising reserve and coefficient M ≤ c, availability ≤ 1) gives v_flow[t] + R↑ ≤ c·cap and
    # v_flow[t′] ≥ 0, the floors (minFlow_*) give v_flow[t] − R↓ ≥ 0 and v_flow[t′] ≤ c·cap, with the same built
    # capacity at both steps.  Cross-period and carried rows are always built (the capacity can differ).
    # c scales the room as the arc's capacity is scaled: capacity_max_coeff[p,k] on the sink side; on the source
    # side input_share_max[p,s], or on an indirect unit's input arc p_indirect_input_cap.  v_flow and v_reserve
    # are in units of unitsize, existing_count = existing / unitsize.  v_startup / v_shutdown are the linear and
    # integer UC variables where the process has them: units starting (stopping) in a step add to its upward
    # (downward) ramp room.  Arcs with conversion_flow_coeff = 0 get no row.
    keys = ["p", "source", "sink"]
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
        up = dir_ == "up"
        sign = 1.0 if up else -1.0
        rows = _ramp_rows(d, idx_set, ramp_speed, side, dir_, link, carry)
        if rows.height == 0:
            continue
        over = rows.select(*keys, "d", "t")
        flow_rows = rows.filter(pl.col("__kind") != "reserve").select(*keys, "d", "t")
        linked = rows.filter(pl.col("__kind") == "link")
        carried = rows.filter(pl.col("__kind") == "carry")

        lhs: dict = {}
        if flow_rows.height > 0:
            lhs["flow"] = Where(v_flow, flow_rows)
        if linked.height > 0:
            prev = _var_view(v_flow, "__ramp_prev", {"d": "d_previous", "t": "t_previous"}, to_axis=True)
            pairs = linked.select(*keys, "d", "t",
                                  pl.col("__d_prev").cast(prev.frame.schema["d_previous"]).alias("d_previous"),
                                  pl.col("__t_prev").cast(prev.frame.schema["t_previous"]).alias("t_previous"))
            lhs["flow_prev"] = -Sum(Where(prev, pairs), over=("d_previous", "t_previous"))
        if up:
            if b.reserve_up_to_sink is not None:
                lhs["reserve_up"] = b.reserve_up_to_sink
            lhs.update(b.reserve_down_at_source)
        else:
            if b.reserve_down_to_sink is not None:
                lhs["reserve_down"] = -b.reserve_down_to_sink
            lhs.update({k: -v for k, v in b.reserve_up_at_source.items()})

        rate = Param(("p", "source", "sink", "d", "t"), rows.select(*keys, "d", "t", value=pl.col("__room")))
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
        if flow_rows.height > 0:
            for kind, has in (("linear", b.has_online_lin), ("integer", b.has_online_int)):
                if not has:
                    continue
                if up:
                    rhs[f"startup_{kind}"] = Where(getattr(b, f"v_startup_{kind}"), flow_rows)
                else:
                    rhs[f"shutdown_{kind}"] = -Where(getattr(b, f"v_shutdown_{kind}"), flow_rows)
        if carried.height > 0:
            rhs["roll_continue_flow"] = Param(("p", "source", "sink", "d", "t"),
                                              carried.select(*keys, "d", "t", value=pl.col("__f_hat")))
        m.add_cstr(
            f"ramp_{side}_{dir_}_constraint",
            over      = over,
            sense     = "<=" if up else ">=",
            lhs_terms = lhs,
            rhs_terms = rhs,
        )


def ramp_step_links(d) -> "tuple[pl.DataFrame, pl.DataFrame]":
    """The ramp predecessors of the steps of ``dt``, Utf8 keys ``_d, _t``.

    ``link``: ``(_d, _t, __d_prev, __t_prev, __dur, __dur_prev)`` from ``dt_ramp_link`` (both steps in dt).
    ``carry``: ``(_d, _t, __d_prev, __t_prev, __dur, __dur_prev)`` from ``dt_ramp_carry`` (the predecessor is
    the previous roll's step; its duration is carried), without steps that also have a link.
    """
    dur = d.p_step_duration.frame.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                                         pl.col("value").cast(pl.Float64).alias("__dur"))
    schema = {"_d": pl.Utf8, "_t": pl.Utf8, "__d_prev": pl.Utf8, "__t_prev": pl.Utf8,
              "__dur": pl.Float64, "__dur_prev": pl.Float64}
    link = pl.DataFrame(schema=schema)
    raw = getattr(d, "dt_ramp_link", None)
    if raw is not None and raw.height > 0:
        link = (raw.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                           pl.col("d_previous").cast(pl.Utf8).alias("__d_prev"),
                           pl.col("t_previous").cast(pl.Utf8).alias("__t_prev"))
                .join(dur, on=["_d", "_t"], how="inner")
                .join(dur.rename({"_d": "__d_prev", "_t": "__t_prev", "__dur": "__dur_prev"}),
                      on=["__d_prev", "__t_prev"], how="inner")
                .select(list(schema)))
    carry = pl.DataFrame(schema=schema)
    raw_c = getattr(d, "dt_ramp_carry", None)
    if raw_c is not None and raw_c.height > 0 and getattr(d, "p_roll_continue_flow", None) is not None:
        carry = (raw_c.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                              pl.col("d_previous").cast(pl.Utf8).alias("__d_prev"),
                              pl.col("t_previous").cast(pl.Utf8).alias("__t_prev"),
                              pl.col("duration_previous").cast(pl.Float64).alias("__dur_prev"))
                 .join(dur, on=["_d", "_t"], how="inner")
                 .join(link.select("_d", "_t"), on=["_d", "_t"], how="anti")
                 .select(list(schema)))
    return link, carry


def _ramp_rows(d, idx_set: pl.DataFrame, ramp_speed: Param, side: str, dir_: str,
               link: pl.DataFrame, carry: pl.DataFrame) -> pl.DataFrame:
    """The rows of one ramp family: ``(p, source, sink, d, t)`` in the dtypes of ``idx_set`` / ``dt`` with
    ``__kind`` (``link`` / ``carry`` / ``reserve``), ``__room`` (ρ), ``__d_prev`` / ``__t_prev`` (Utf8) and
    ``__f_hat`` (the carried flow of ``carry`` rows).  See the EQ. header of :func:`_eq_ramp`."""
    keys = ["p", "source", "sink"]
    speed = ramp_speed.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
                                    pl.col("value").cast(pl.Float64).alias("__speed"))
    base = (_utf8_keys(idx_set.select(keys).unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
                       .lazy(), keys + ["d", "t"])
            .with_columns(pl.col(f"_{side}").alias("_n"))
            .join(speed.lazy(), on=["_p", "_n"], how="inner"))
    with_link = (base.join(link.lazy(), on=["_d", "_t"], how="inner")
                 .with_columns(__kind=pl.lit("link"), __f_hat=pl.lit(None, pl.Float64)))
    parts = [with_link.filter((pl.col("__d_prev") != pl.col("_d"))
                              | (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0 < 1.0))]
    if carry.height > 0:
        f_hat = d.p_roll_continue_flow.frame.select(
            pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("source").cast(pl.Utf8).alias("_source"),
            pl.col("sink").cast(pl.Utf8).alias("_sink"), pl.col("d").cast(pl.Utf8).alias("_d"),
            pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__f_hat"))
        parts.append(base.join(carry.lazy(), on=["_d", "_t"], how="inner")
                     .join(f_hat.lazy(), on=["_p", "_source", "_sink", "_d", "_t"], how="inner")
                     .with_columns(__kind=pl.lit("carry")))
    prundt = getattr(d, "prundt", None)
    if prundt is not None and prundt.height > 0:
        # Steps without a predecessor where the arc holds reserve in this direction: at its sink in the same
        # direction, at its source in the opposite one.
        held = prundt.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("ud").cast(pl.Utf8).alias("_ud"),
                                    pl.col("n").cast(pl.Utf8).alias("_rn"), pl.col("d").cast(pl.Utf8).alias("_d"),
                                    pl.col("t").cast(pl.Utf8).alias("_t"))
        at_sink = held.filter(pl.col("_ud") == dir_).select("_p", pl.col("_rn").alias("_sink"), "_d", "_t").unique()
        at_source = (held.filter(pl.col("_ud") != dir_)
                     .select("_p", pl.col("_rn").alias("_source"), "_d", "_t").unique())
        lone = base.join(link.lazy().select("_d", "_t"), on=["_d", "_t"], how="anti")
        if carry.height > 0:
            lone = lone.join(carry.lazy().select("_d", "_t"), on=["_d", "_t"], how="anti")
        lone = lone.join(d.p_step_duration.frame.lazy().select(
            pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
            pl.col("value").cast(pl.Float64).alias("__dur")), on=["_d", "_t"], how="inner")
        held_rows = pl.concat([
            lone.join(at_sink, on=["_p", "_sink", "_d", "_t"], how="semi"),
            lone.join(at_source, on=["_p", "_source", "_d", "_t"], how="semi"),
        ]).unique(subset=["_p", "_source", "_sink", "_d", "_t"], keep="first", maintain_order=True)
        parts.append(held_rows
                     .with_columns(__d_prev=pl.lit(None, pl.Utf8), __t_prev=pl.lit(None, pl.Utf8),
                                   __dur_prev=pl.col("__dur"), __kind=pl.lit("reserve"),
                                   __f_hat=pl.lit(None, pl.Float64))
                     .filter(pl.col("__speed") * 60.0 * pl.col("__dur") < 1.0))
    cols = [*keys, "d", "t", "__kind", "__d_prev", "__t_prev", "__f_hat",
            (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0).alias("__room")]
    return pl.concat([p.select(cols) for p in parts]).collect()


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
    # subtracted (known limitation).
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
    #   slope[p,d,t] = (1/eta − min_load/eta_min) / (1 − min_load)
    #   section[p,d,t] ≈ min_load·(1/eta_min − 1/eta) / (1 − min_load)
    #     for min_load_efficiency units (eta = efficiency, default 1; min_load = 0 where not authored;
    #     eta_min = efficiency_at_min_load, else efficiency); slope = 1/eta and no section for the other units
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


def _eff_fuel_input(b: _Build, arcs: pl.DataFrame) -> list[tuple[str, Expr]]:
    """Fuel input of processes with an efficiency on the arcs ``arcs`` (p, source, sink, ...).

    ``[("eff", v_flow·unitsize·slope)]`` plus, for min_load_efficiency units, ``("section_lin" |
    "section_int", v_online·section·unitsize)``, each restricted to ``arcs``; MWh/h per (arc, d, t), unweighted.
    Shared by the commodity cost (commodity_cost_eff), the CO2 price (co2_price_eff) and both CO2 caps
    (co2_max_period, co2_max_total), so their fuel measure cannot drift apart.
    """
    d = b.d
    terms: list[tuple[str, Expr]] = [("eff", Where(b.v_flow * d.p_unitsize * d.p_slope, arcs))]
    if b.has_minload_eff:
        if b.has_online_lin:
            terms.append(("section_lin", Where(Where(b.v_online_linear, d.process_min_load_eff)
                                               * d.p_section * d.p_unitsize, arcs)))
        if b.has_online_int:
            terms.append(("section_int", Where(Where(b.v_online_integer, d.process_min_load_eff)
                                               * d.p_section * d.p_unitsize, arcs)))
    return terms


def _eq_co2_max_period(b: _Build) -> None:
    """CO2 emission cap per period."""
    if not b.has_co2_cap:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    # ─── EQ. ─── co2_max_period ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped, t}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c]·w[d,t]
    #   + Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_noEff, t} v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w[d,t]
    #   ≤ co2_max_period[g,d] · window_fraction[d]
    #   ∀ (g,d) ∈ group_d_co2_capped;   w = step_duration · timestep_weight / period_share
    #   window_fraction[d] = (Σ_t step_duration[d,t] / 8760) / period_share[d]  (1 outside rolling solves)
    # Caps the annualised CO2 emissions of a group in each period (tonnes per year): dividing by
    # window_fraction, the LHS is the emissions of the solve's own steps of d annualised over their hours.
    # In a rolling solve period_share is the complete solve's share, so the cap is scaled to the roll's
    # window (realized part and look-ahead, per period); the dual keeps the objective's weighting.  Each
    # roll caps the rate of its window, not the realized total of the period.  Emissions come from
    # the fuel that processes draw from the group's nodes when it carries a CO2-emitting commodity c,
    # measured as in co2_price_eff (_eff_fuel_input): on eff processes the output flow times slope plus,
    # for min_load_efficiency units (linear and integer v_online), the section fuel of the units online.
    # noEff processes take neither slope nor section (applying them would over-count).
    lhs: dict = {}
    if b.has_co2_cap_eff:
        for key, term in _eff_fuel_input(b, d.flow_from_co2_capped):
            lhs[f"emissions_{key}"] = Sum(
                term
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
        rhs_terms = {"cap": d.p_co2_max_period * d.p_window_fraction},
    )


def _eq_co2_max_total(b: _Build) -> None:
    """CO2 emission cap over all periods."""
    if not b.has_co2_cap_total:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    # ─── EQ. ─── co2_max_total ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_total, d,t}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c]·w_y[d,t]
    #   + Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_total_noEff, d,t}
    #     v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w_y[d,t]
    #   ≤ co2_max_total[g]
    #   ∀ g ∈ group_co2_max_total;   w_y = step_duration · timestep_weight / period_share · years_represented[d]
    # Caps the CO2 emissions of a group over the horizon the solve represents (tonnes): each period's
    # annualised emissions (as in co2_max_period, section fuel included) times the years the period
    # represents, summed over the periods — the measure of the model CO2 output.  years_represented is not
    # discounted.  Each solve of a chain and each roll of a rolling solve gets the full cap for only its
    # own periods / window, so there the cap binds more loosely than intended (known limitation).
    lhs: dict = {}
    if b.has_co2_cap_total_eff:
        for key, term in _eff_fuel_input(b, d.flow_from_co2_capped_total):
            lhs[f"emissions_{key}"] = Sum(
                term
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / d.p_period_share
                * d.p_years_represented_d,
                over=("p", "source", "sink", "c", "d", "t"),
            )
    if b.has_co2_cap_total_noEff:
        lhs["emissions_noEff"] = Sum(
            Where(v_flow * d.p_unitsize, d.flow_from_co2_capped_total_noEff)
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share
            * d.p_years_represented_d,
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
    # the last step of the first period (known limitation: not the last step of the horizon) to the share
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
    # capacity (state units).  Invested and divested capacity is not included (known limitation).  Nodes
    # with a competing end condition are not in storage_use_reference_value.
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
    # v_flow[p,s,k,d,t] ± (Σ_r v_reserve[p,r,k,d,t] + Σ_r v_reserve[p,r,s,d,t] · φ[p,s,k,d,t])
    #   − profile[f,d,t]·av[p,d,t]·Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + profile[f,d,t]·av[p,d,t]·Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    #   ≤ / ≥ / =  profile[f,d,t] · av[p,d,t] · existing_count[p,d]
    #   ∀ (p,s,k,f) ∈ process_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds a flow by a profile times the built capacity, in units of unitsize.  av = availability, 1 for
    # processes without availability data.  The upper limit adds the reserve that raises the arc's flow
    # (up at the sink; down at a unit's source), the lower limit subtracts the reserve that lowers it (down
    # at the sink; up at a unit's source); a source-side reserve is converted to the arc's flow by φ (1
    # where v_flow is the input, 1/slope on a direct unit's input→output arc).  The fixed profile has no
    # reserve term.  The invest/divest terms are restricted to the (p,f) pairs of the method's set.
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
    source_terms = {
        "profile_flow_upper_limit": dict(b.reserve_down_at_source),
        "profile_flow_lower_limit": {f"{k}_neg": -v for k, v in b.reserve_up_at_source.items()},
    }
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
        lhs.update(source_terms.get(name, {}))
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


# Every operational cost term is weighted by b.op_factor:
#   w_op[d,t] = step_duration[d,t] · timestep_weight[d,t] · inflation_op[d] / period_share[d] · branch_weight[d,t]
# step_duration turns a rate (currency/h) into currency per step, timestep_weight scales a representative step
# to the time it stands for, / period_share scales the modelled part of the period to a full year, and
# inflation_op discounts that year's operating cost; branch_weight is 1 outside stochastic solves.  Terms for point events
# (startups, end-of-horizon valuation) use w_ev[d,t] = w_op[d,t] / step_duration[d,t].
# The commodity ladder (_commodity_ladder) uses the per-period form
# inflation_op[d] / period_share[d] · branch_weight[d], because its purchases are per period.


def _obj_balance_slack(b: _Build) -> Expr:
    """Penalty on the node balance slacks."""
    d = b.d
    vq_up = b.vq_state_up
    vq_down = b.vq_state_down
    op_factor = b.op_factor

    # ─── OBJ. ─── balance_slack_penalty ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(n,d,t)} vq_state_up[n,d,t] · penalty_up[n,d,t] · ncs[n,d] · w_op[d,t]
    #   + Σ_{(n,d,t)} vq_state_down[n,d,t] · penalty_down[n,d,t] · ncs[n,d] · w_op[d,t]
    #   ∀ (n,d,t) with balance slacks (nodeBalance ∪ nodeBalancePeriod, penalty_method ≠ off)
    #   penalty is defined on every slack cell: the node's authored value, else the schema default
    #   ncs = node_capacity_for_scaling, factor 1 when the parameter is absent;   currency
    # Prices the energy a node balance cannot supply (up) or absorb (down); penalty in currency/MWh.
    # The slacks enter the balance without ncs, so with node_capacity_for_scaling ≠ 1 the effective
    # penalty per MWh is penalty · ncs (unreachable today; known inconsistency).
    if d.p_node_capacity_for_scaling is not None:
        return (Sum(vq_up   * d.p_penalty_up
                    * d.p_node_capacity_for_scaling * op_factor)
                + Sum(vq_down * d.p_penalty_down
                      * d.p_node_capacity_for_scaling * op_factor))
    return (Sum(vq_up   * d.p_penalty_up   * op_factor)
            + Sum(vq_down * d.p_penalty_down * op_factor))


def _obj_commodity(b: _Build) -> Expr | None:
    """Commodity purchase cost and sales revenue."""
    if not b.has_proc:
        return None
    d = b.d
    v_flow = b.v_flow
    op_factor = b.op_factor
    # Commodities with a price ladder are priced by _commodity_ladder, not by the single commodity price.
    ladder_c = (d.commodity_with_ladder["c"]
                if b.has_ladder and d.commodity_with_ladder is not None else None)

    def _single_price(flows: "pl.DataFrame | None") -> "pl.DataFrame | None":
        if flows is None or ladder_c is None:
            return flows
        return flows.filter(~pl.col("c").is_in(ladder_c))

    buy_eff = _single_price(d.flow_from_commodity_eff)
    buy_noEff = _single_price(d.flow_from_commodity_noEff)
    sell = _single_price(d.flow_to_commodity)
    terms: list = []

    # ─── OBJ. ─── commodity_cost_eff ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c) ∈ flow_from_commodity_eff, (d,t)}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · commodity_price[c,d,t] · w_op[d,t]
    # Cost of the commodity c bought from node s by processes with an efficiency: the input is the
    # output flow times slope, plus the section input per online unit (min_load_efficiency units only,
    # linear and integer online alike).  Price in currency/MWh.
    if _has_rows(buy_eff):
        terms += [Sum(term * d.p_commodity_price * op_factor) for _key, term in _eff_fuel_input(b, buy_eff)]

    # ─── OBJ. ─── commodity_cost_noEff ───────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c) ∈ flow_from_commodity_noEff, (d,t)} v_flow[p,s,k,d,t]·unitsize[p] · commodity_price[c,d,t] · w_op[d,t]
    # Cost of the commodity bought by processes without an efficiency: the input equals the flow.
    if _has_rows(buy_noEff):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, buy_noEff)
            * d.p_commodity_price * op_factor))

    # ─── OBJ. ─── commodity_sales ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,k,c) ∈ flow_to_commodity, (d,t)} v_flow[p,s,k,d,t]·unitsize[p] · commodity_price[c,d,t] · w_op[d,t]
    # Revenue from flows into a commodity node k, sold at the commodity price (negative cost).  Every
    # process arc counts, with or without efficiency: the sink-side flow is the delivered energy.
    if _has_rows(sell) and d.p_commodity_price is not None:
        terms.append(-Sum(
            Where(v_flow * d.p_unitsize, sell)
            * d.p_commodity_price * op_factor))
    return _sum_terms(terms)


def _obj_co2_price(b: _Build) -> Expr | None:
    """CO2 emission cost of processes with an efficiency."""
    if not b.has_co2_price:
        return None
    d = b.d

    # ─── OBJ. ─── co2_price_eff ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_priced, (d,t)}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # Cost of the CO2 emitted when processes with an efficiency draw a CO2-bearing commodity c from a
    # node of a CO2-priced group g; content in t/MWh, price in currency/t.  The section term exists
    # for min_load_efficiency units only.  Flows into priced nodes earn no removal credit (known
    # limitation).
    return _sum_terms([Sum(term * d.p_co2_content * d.p_co2_price * b.op_factor)
                       for _key, term in _eff_fuel_input(b, d.flow_from_co2_priced)])


def _obj_co2_price_noEff(b: _Build) -> Expr | None:
    """CO2 emission cost of processes without an efficiency."""
    d = b.d
    if not (_has_rows(d.flow_from_co2_priced_noEff)
            and d.p_co2_content is not None and d.p_co2_price is not None):
        return None

    # ─── OBJ. ─── co2_price_noEff ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_priced_noEff, (d,t)}
    #     v_flow[p,s,k,d,t]·unitsize[p] · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # As co2_price_eff for processes without an efficiency: the input equals the flow, no slope and
    # no section term.
    return Sum(
        Where(b.v_flow * d.p_unitsize, d.flow_from_co2_priced_noEff)
        * d.p_co2_content * d.p_co2_price * b.op_factor)


def _obj_variable_cost(b: _Build) -> Expr | None:
    """Variable operation and maintenance cost of process flows."""
    if not b.has_proc:
        return None
    d = b.d
    v_flow = b.v_flow
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    op_factor = b.op_factor
    terms: list = []

    # ─── OBJ. ─── variable_cost_noEff ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_noEff} v_flow[p,s,k,d,t]·unitsize[p] · varCost[p,s,k,d,t] · w_op[d,t]
    # pssdt_varCost_noEff = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_noEff, varCost[p,s,k,d,t] ≠ 0}
    # O&M cost of the flows of processes without an efficiency; varCost (currency/MWh) is the arc's
    # combined process, source-side and sink-side other_operational_cost (every authored shape: constant,
    # period, time, period × time, stochastic time).
    if (_has_rows(d.pssdt_varCost_noEff)
            and d.p_pssdt_varCost is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_noEff)
            * d.p_pssdt_varCost * op_factor))

    # ─── OBJ. ─── variable_cost_eff_source ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_unit_source}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · other_operational_cost_source[p,s,d,t] · w_op[d,t]
    # pssdt_varCost_eff_unit_source = {(p,s,k) ∈ process_source_sink_eff} × dt where
    #     other_operational_cost_source[p,s,d,t] ≠ 0 (every authored shape: constant, period, time,
    #     period × time, stochastic time).
    # O&M cost on the input side of units with an efficiency, priced on the input energy (flow times
    # slope, plus the section input of min_load_efficiency units).  Added like the other O&M terms
    # (GMPL subtracted it).  The sink/source coefficient ratio of the input is not applied (refactor
    # bug 10).
    if (_has_rows(d.pssdt_varCost_eff_unit_source)
            and d.p_pdt_varCost_source is not None):
        flow_term = Where(v_flow * d.p_unitsize * d.p_slope,
                          d.pssdt_varCost_eff_unit_source)
        terms.append(Sum(flow_term * d.p_pdt_varCost_source * op_factor))
        if b.has_minload_eff:
            section_idx = (d.pssdt_varCost_eff_unit_source
                           .join(d.process_min_load_eff, on="p", how="inner"))
            if section_idx.height > 0:
                if b.has_online_lin:
                    terms.append(Sum(
                        Where(Where(v_online_lin, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor))
                if b.has_online_int:
                    terms.append(Sum(
                        Where(Where(v_online_int, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor))

    # ─── OBJ. ─── variable_cost_eff_sink ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_unit_sink} v_flow[p,s,k,d,t]·unitsize[p] · other_operational_cost_sink[p,k,d,t] · w_op[d,t]
    # pssdt_varCost_eff_unit_sink = {(p,s,k) ∈ process_source_sink_eff} × dt where
    #     other_operational_cost_sink[p,k,d,t] ≠ 0 (every authored shape, as above).
    # O&M cost on the output side of units with an efficiency, priced on the output flow.
    if (_has_rows(d.pssdt_varCost_eff_unit_sink)
            and d.p_pdt_varCost_sink is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_unit_sink)
            * d.p_pdt_varCost_sink * op_factor))

    # ─── OBJ. ─── variable_cost_eff_connection ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_connection} v_flow[p,s,k,d,t]·unitsize[p] · other_operational_cost[p,d,t] · w_op[d,t]
    # pssdt_varCost_eff_connection = {(p,s,k) ∈ process_source_sink_eff} × dt where
    #     other_operational_cost[p,d,t] ≠ 0 (constant, period, time or period × time).
    # O&M cost of connections with an efficiency, priced on the flow of each direction.
    if (_has_rows(d.pssdt_varCost_eff_connection)
            and d.p_pdt_varCost_process is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_connection)
            * d.p_pdt_varCost_process * op_factor))
    return _sum_terms(terms)


def _obj_startup_cost(b: _Build) -> Expr | None:
    """Startup cost of unit-commitment processes."""
    if not (b.has_startup_cost_lin or b.has_startup_cost_int):
        return None
    d = b.d
    terms: list = []

    # ─── OBJ. ─── startup_cost ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,d,t) ∈ pdt_online_{linear,integer}} v_startup_{linear,integer}[p,d,t] · startup_cost[p,d] · unitsize[p] · w_ev[d,t]
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight
    # Cost of starting up units, startup_cost in currency/MW started.  A startup is an event, so the
    # weight has no step_duration.
    startup_factor = (d.p_timestep_weight * d.p_inflation_op
                      / d.p_period_share)
    if d.pdt_branch_weight is not None:
        startup_factor = startup_factor * d.pdt_branch_weight
    if b.has_startup_cost_lin:
        terms.append(Sum(
            Where(b.v_startup_linear, d.pdt_online_linear)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    if b.has_startup_cost_int:
        terms.append(Sum(
            Where(b.v_startup_integer, d.pdt_online_integer)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    return _sum_terms(terms)


def _obj_storage_reference_price(b: _Build) -> Expr | None:
    """Value of the storage content left at the end of the solve horizon."""
    d = b.d
    if not (b.has_storage
            and d.p_storage_state_reference_price is not None
            and _has_rows(d.nodeState_last_dt)
            and _has_rows(d.period_last)):
        return None
    over = (d.nodeState_last_dt
            .join(d.period_last, on="d", how="inner")
            .join(d.p_storage_state_reference_price.frame.select("n", "d"),
                  on=["n", "d"], how="inner")
            .select("n", "d", "t").unique())
    if over.height == 0:
        return None

    # ─── OBJ. ─── storage_reference_price ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(n,d,t) ∈ nodeState_last_dt, d ∈ period_last, (n,d) ∈ reference_price}
    #     v_state[n,d,t]·state_unitsize[n] · reference_price[n,d] · w_ev[d,t]
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight
    # Values the energy left in storage at the end of the solve horizon (revenue, hence the minus),
    # reference_price in currency/MWh.  The price comes from use_reference_price nodes or from the
    # storage-price handoff of a previous solve.  An end-of-horizon valuation is a point event, so the
    # weight has no step_duration.  Rows with a zero price are kept.
    ref_price_factor = (d.p_timestep_weight * d.p_inflation_op
                        / d.p_period_share)
    if d.pdt_branch_weight is not None:
        ref_price_factor = ref_price_factor * d.pdt_branch_weight
    return -Sum(
        Where(b.v_state, over) * d.p_state_unitsize
        * d.p_storage_state_reference_price * ref_price_factor)


def _obj_invest_divest(b: _Build) -> Expr | None:
    """Investment and divestment costs of processes and storage nodes."""
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n):
        return None
    d = b.d
    terms: list = []
    # Under recourse investment every scenario branch carries its own invest/divest variables from the
    # branching period on, so each cost is weighted by the branch probability pd_branch_weight[d] and the
    # term is the expected cost.  Otherwise bw = 1 and no weight is applied.
    recourse = getattr(d, "recourse_invest", False) and d.pd_branch_weight is not None

    def _cost(param: "Param | None", ent: str) -> "Param | None":
        """Entity cost param (e, d) on axis ``ent``, times pd_branch_weight under recourse; None stays None."""
        if param is None:
            return None
        cost = Param((ent, "d"), param.frame.pipe(rename_to_axis, {"e": ent}))
        if not recourse:
            return cost
        f = (cost.frame
             .join(d.pd_branch_weight.frame.rename({"value": "w"}), on="d", how="left")
             .with_columns(value=pl.col("value") * pl.col("w").fill_null(1.0))
             .drop("w"))
        return Param(cost.dims, f)

    # ─── OBJ. ─── invest_cost_p ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,d) ∈ pd_invest_set} v_invest_p[p,d]·unitsize[p] · annual_discounted[p,d] · bw[d]
    #   + Σ_{(p,d) ∈ pd_invest_set} v_invest_p[p,d]·unitsize[p] · lifetime_fixed_cost[p,d] · bw[d]
    # Cost of process capacity built in period d, currency/MW: the investment annuity summed over the
    # discounted years the capacity serves within the horizon, and the inflation-adjusted fixed cost
    # over its remaining lifetime.  No w_op: both factors are already per period.
    if b.has_invest_p:
        annu = _cost(d.ed_entity_annual_discounted, "p")
        lf = _cost(d.ed_lifetime_fixed_cost, "p")
        if annu is not None:
            terms.append(Sum(b.v_invest_p * d.p_unitsize * annu))
        if lf is not None:
            terms.append(Sum(b.v_invest_p * d.p_unitsize * lf))

    # ─── OBJ. ─── divest_cost_p ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,d) ∈ pd_divest_set} v_divest_p[p,d]·unitsize[p] · lifetime_fixed_cost_divest[p,d] · bw[d]
    #   − Σ_{(p,d) ∈ pd_divest_set} v_divest_p[p,d]·unitsize[p] · annual_divest_discounted[p,d] · bw[d]
    # Saving from process capacity retired in period d, currency/MW: the fixed cost no longer paid and
    # the salvage-value annuity over the discounted years of the remaining lifetime.
    if b.has_divest_p:
        lfd = _cost(d.ed_lifetime_fixed_cost_divest, "p")
        annd = _cost(d.ed_entity_annual_divest_discounted, "p")
        if lfd is not None:
            terms.append(-Sum(b.v_divest_p * d.p_unitsize * lfd))
        if annd is not None:
            terms.append(-Sum(b.v_divest_p * d.p_unitsize * annd))

    # ─── OBJ. ─── invest_cost_n ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(n,d) ∈ nd_invest_set} v_invest_n[n,d]·state_unitsize[n] · annual_discounted[n,d] · bw[d]
    #   + Σ_{(n,d) ∈ nd_invest_set} v_invest_n[n,d]·state_unitsize[n] · lifetime_fixed_cost[n,d] · bw[d]
    # As invest_cost_p for storage capacity, currency/MWh of storage.
    if b.has_invest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        annu_n = _cost(d.ed_entity_annual_discounted, "n")
        lf_n = _cost(d.ed_lifetime_fixed_cost, "n")
        if annu_n is not None:
            terms.append(Sum(b.v_invest_n * us_n * annu_n))
        if lf_n is not None:
            terms.append(Sum(b.v_invest_n * us_n * lf_n))

    # ─── OBJ. ─── divest_cost_n ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(n,d) ∈ nd_divest_set} v_divest_n[n,d]·state_unitsize[n] · lifetime_fixed_cost_divest[n,d] · bw[d]
    #   − Σ_{(n,d) ∈ nd_divest_set} v_divest_n[n,d]·state_unitsize[n] · annual_divest_discounted[n,d] · bw[d]
    # As divest_cost_p for storage capacity.
    if b.has_divest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        lfd_n = _cost(d.ed_lifetime_fixed_cost_divest, "n")
        annd_n = _cost(d.ed_entity_annual_divest_discounted, "n")
        if lfd_n is not None:
            terms.append(-Sum(b.v_divest_n * us_n * lfd_n))
        if annd_n is not None:
            terms.append(-Sum(b.v_divest_n * us_n * annd_n))
    return _sum_terms(terms)


def _obj_existing_fixed_cost(b: _Build) -> None:
    """Fixed cost of existing capacity as an objective constant (opt-in)."""
    d = b.d
    if not (b.include_existing_fixed_cost
            and d.p_ed_fixed_cost is not None
            and d.p_entity_all_existing is not None):
        return

    # ─── OBJ. ─── existing_fixed_cost ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(e,d)} existing[e,d] · fixed_cost[e,d] · inflation_op[d] · bw[d]
    #   bw = pd_branch_weight, factor 1 when absent
    # Fixed cost of the capacity that exists in each period, currency (fixed_cost in currency/MW/year).
    # A constant: it moves the objective value but not the solution.  Added only with
    # include_existing_fixed_cost; when pd_branch_weight is present, periods without a weight row drop out.
    fc_frame = (
        d.p_entity_all_existing.frame.rename({"value": "exist"})
        .join(d.p_ed_fixed_cost.frame.rename({"value": "fc"}),
              on=["e", "d"], how="inner")
        .join(d.p_inflation_op.frame.rename({"value": "infl"}),
              on="d", how="inner")
    )
    if d.pd_branch_weight is not None:
        fc_frame = (fc_frame
                    .join(d.pd_branch_weight.frame.rename({"value": "pdbw"}), on="d", how="inner")
                    .with_columns(contrib=pl.col("exist") * pl.col("fc") * pl.col("infl") * pl.col("pdbw")))
    else:
        fc_frame = fc_frame.with_columns(
            contrib=pl.col("exist") * pl.col("fc") * pl.col("infl"))
    fc_const = fc_frame["contrib"].sum()
    if fc_const:
        b.m.add_obj_constant(float(fc_const))


# ---------------------------------------------------------------------------
# Helpers


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
                "(_derived.p_indirect_input_cap_from_source / "
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
