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
  v_flow[p,source,sink,d,t]            flow on a process arc; × unitsize = MW (a coarse process: one column per
                                       coarse step, read on every step through _coarse_flows.alias)
  vq_state_up / _down[n,d,t]           balance slack (shortfall / surplus), MW
  v_state[n,d,t]                       storage state; × state_unitsize = MWh
  v_state_inter[n,d,b]                 representative-period storage state at base-period boundaries of d
  v_state_rp_start[n,d,t]              free start state of each representative block
  v_{online,startup,shutdown}_{linear,integer}[p,d,t]
                                       units (connections) online / starting / stopping, count of unitsize
                                                                                              [_online]
  v_{invest,divest}_{p,n}[p|n,d]       capacity built / retired in d, count of unitsize
  v_divest_built_{p,n}[p|n,c,d], v_divested_built_{p,n}[p|n,d]
                                       model-built cohort c retired in d, retired so far     [_retire_built]
  v_extend_{p,n}[p|n,c,d], v_extended_{p,n}[p|n,d]
                                       cohort c kept beyond its end of life from d, extended in service [_extend]
  v_reserve[p,r,ud,n,d,t]              reserve held by p for n; × unitsize = MW               [_reserve]
  vq_reserve[r,ud,g,d,t]               reserve shortfall, fraction of the requirement scale   [_reserve]
  v_reserve_total[r,ud,g,d,t]          total reserve of a group with n-1 rows, MW             [_reserve]
  v_flow_back[p,source,sink,d,t]       reverse flow of 2-way 1-var and DC arcs; × unitsize = MW
  v_exact_forward[p,d,t]               exact connection set to its forward direction, binary   [_exact]
  v_angle[n,d,t]                       voltage angle, rad in [−π, π]                          [_dc_power_flow]
                                       (units online, start-ups, reserves, directions, angles and the inertia /
                                       non-synchronous slacks of coarse entities: one column per coarse step)
  v_trade[c,n,d,i]                     commodity bought in tier i; × commodity unitsize = MWh [_commodity_ladder]
  vq_capacity_margin[g,d], vq_inertia[g,d,t], vq_non_synchronous[g,d,t]
                                       group floor slacks, fraction of the group's scale      [_group_slack]

Equation families, in emit order (module in brackets when not model/core.py):
  Balance & storage   rp_inter_period_{balance,cyclic,fan_start,max_state}, maxState_rp_start;
                      rp_inter_period_{start_fix,start_carry,end_fix,start_max_state,start_min_state,end_match,start_match,
                      reference_value} (the ends of the base-period chain); nodeBalance_eq;
                      nodeBalanceCoarse_eq, stateConstantCoarse_eq (coarse new_stepduration groups);
                      nodeBalancePeriod_eq; node_balance_fix_quantity_eq_lower; node_storage_usage_fix_le;
                      stateConstantWithinBlock_eq, nodeBalanceBlock_eq (bind_intraperiod_blocks)
  Flows & capacity    maxFlow, maxFlow_negCap, maxFlow_back; maxOutputSum(_negCap);
                      dc_reference_angle_eq, dc_flow_eq [_dc_power_flow]
                      (the arc-local families of coarse processes — maxFlow, maxOutputSum, conversion_indirect,
                      conversion_piecewise, minInputShare and profile_flow_* — are written once per coarse step as
                      <family>_coarse [_coarse_flows])
  Units               max{Online,Startup,Shutdown}, online_change, maxFlow_online[_shutdown[_branch]],
                      maxFlow_back_online, maxOutputSum_online, maxInputFlow_online[_shutdown[_branch]],
                      minimum_{uptime,downtime}, online_fix_eq [_online];
                      maxFlow_exact_{forward,reverse} (exact connections) [_exact];
                      minFlow_{minload,output_floor,reserve,reserve_arc} [_unit_floors];
                      ramp_{sink,source}_{up,down}_constraint; ramp_change_eq (ramp cost)
  Investment          max{Invest,Divest}_var_bound[_n], _entity_period_{p,n}, _entity_total[_path][_n];
                      maxRetired_existing_{p,n}; divestedBuilt_{p,n}, maxDivest_cohort_{p,n} [_retire_built];
                      extendedCapacity_{p,n}, maxExtend_cohort_{p,n} [_extend]
  Conversion          conversion_indirect (delayed inputs from [_delay]); conversion_piecewise (part-load curves of
                      piecewise_linear units); minInputShare
  Emissions & user    co2_max_period, co2_max_total; process_constraint_{equal,less_than,greater_than}[_period]
  Storage state       maxState; profile_state_{upper_limit,lower_limit,fixed};
                      storage_state_start_binding_cyclic_period; storage_within_solve_path_end,
                      storage_state_fan_start (stochastic fans); storage_state_end_binding;
                      storage_state_roll_{anchor,close}[_share] (rolling cyclic storage, per instance or
                      period); storage_state_roll_match (rolling forward chains);
                      storage_state_end_match[_share] (forward chains whose start the solve decides);
                      storage_state_solve_horizon_reference_value
  Profiles            profile_flow_{upper_limit,lower_limit,fixed}
  Objective           balance slack penalty, commodity cost and sales, CO2 price, variable cost, startup cost,
                      ramp cost, storage reference price, invest/divest annuities, existing fixed cost (optional
                      constant),
                      divest_built_cost [_retire_built], extend_cost [_extend]
  Feature modules     capacityMargin, inertia_constraint, non_sync_constraint + slack penalties [_group_slack];
                      reserveBalance_*, reserve_total_upper, reserve_process_*, connection_reserve_*,
                      reserve_storage_*
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

from flextool.engine.model import _balance_terms
from flextool.engine.model import _capacity
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _commodity_ladder
from flextool.engine.model import _cumulative_invest
from flextool.engine.model import _dc_power_flow
from flextool.engine.model import _delay
from flextool.engine.model import _exact
from flextool.engine.model import _group_slack
from flextool.engine.model import _lead_time
from flextool.engine.model import _non_anticipativity
from flextool.engine.model import _online
from flextool.engine.model import _reserve
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.model import _self_discharge
from flextool.engine.model import _unit_floors
from flextool.engine.input._axis_enums import alias_to_axis, align_join_dtypes, join_aligned, rename_to_axis
from flextool.engine.model._common import (
    _availability_factor,
    _connection_procs,
    _coef_factor,
    _drop_uncapped,
    _entity_rows,
    _has_rows,
    _indirect_sum_arcs,
    dispatch_periods,
    _utf8_keys,
    _var_view,
    chain_prior,
    branch_periods_inherit,
    check_max_cap,
    leaf_rows,
    minus_prior,
)
from flextool.engine.input._param_shapes import drop_no_limit_rows, promote_param_to_dt, without_no_limit_rows
from flextool.engine.model._pdt_join import (
    compute_pss_dt,
    compute_nodeBalance_dt,
    compute_nodeState_dt,
    compute_nodeState_rp_block_first_dt,
    compute_process_indirect_dt,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve import _objective_options

_LOG = logging.getLogger(__name__)

# Objective cutoff relative to the largest |cost| (polar-high
# ``Problem.obj_coef_zero_rel_threshold``, owner decision Q-gate-A).  About 100
# machine epsilons: it removes float residue (a 1e-17 cost next to O(1) costs)
# while the widest legitimate cost spread in the LP-snapshot fixtures is
# min/max = 1.4e-12 (11.8 decades).  Relative, not ``small_number_threshold``:
# real costs go down to 4e-5 (slack penalties) and the objective may be scaled.
OBJ_COEF_ZERO_REL_THRESHOLD: float = 1e-14


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
# piecewise_linear units with two or more lines (conversion_piecewise; set at the load boundary, params/_piecewise).
PIECEWISE: tuple[str, ...] = ("process_piecewise", "p_piecewise_slope", "p_piecewise_intercept")
CO2_CAP:   tuple[str, ...]   = ("p_co2_content", "p_co2_max_period",
                                "group_d_co2_capped", "p_window_fraction")
CO2_CAP_TOTAL: tuple[str, ...] = ("p_co2_content", "p_co2_max_total",
                                   "group_co2_max_total", "p_years_represented_d")
USER_CSTR: tuple[str, ...]   = ("p_constraint_constant",)
PROFILES:  tuple[str, ...]   = ("p_profile_value", "p_process_existing_count")
STORAGE:   tuple[str, ...]   = ("nodeState", "dtttdt",
                                "p_state_upper", "p_state_unitsize")
# Online reads its predecessors only from the ramp links (dt_ramp_link) and the stretch wrap (dt_online_wrap, with
# the free starts of _ramp_links.online_free_start_cells): without them every step would start from zero units
# online (a start-up charged at every step).
ONLINE: tuple[str, ...] = (
    "process_online", "p_online_dt",
    "dt_ramp_link", "dt_online_wrap", "p_process_existing_count",
)
# The minFlow_minload floor's field: checked when some unit has min_load > 0 (process_minload), so a loader gap
# does not drop the floor silently.  An online unit without min_load has none (min_load 0).
MIN_LOAD: tuple[str, ...] = ("p_min_load",)
MINLOAD_EFF: tuple[str, ...] = ("process_min_load_eff", "p_section")
STARTUP_COST_LINEAR: tuple[str, ...] = ("p_startup_cost", "pdt_online_linear")
STARTUP_COST_INTEGER: tuple[str, ...] = ("p_startup_cost", "pdt_online_integer")
RAMP: tuple[str, ...] = ("dt_ramp_link", "p_step_duration", "p_process_existing_count")
# The ramp cost needs no dt_ramp_link: a solve with only carried rows, or none, builds what it has.
RAMP_COST: tuple[str, ...] = ("p_unitsize", "p_ramp_cost_arc")
# Checked whenever a group has capacity_margin_method = absolute: without period_invest or the group scaling the
# capacity margin would be skipped without a word.
CAPACITY_MARGIN: tuple[str, ...] = ("period_invest", "p_inv_group_cap")
# Investment adds the units of the earlier investment periods still in service (the cohorts' cohort_alive,
# processes and nodes, _capacity); without it an investment would add no capacity.  A retirement removes
# pre-existing capacity only, up to p_entity_pre_existing (maxRetired_existing_{p,n}), in the periods its reach
# dd_forward and the existing cohort's cohort_alive give.
INVEST: tuple[str, ...] = ("p_entity_max_units", "cohort_alive")
DIVEST: tuple[str, ...] = ("p_entity_max_units", "cohort_alive", "dd_forward", "p_entity_pre_existing",
                            "pd_divest_set")
# The node analogue of DIVEST: a node retirement removes its capacity in the same periods and costs per state unit
# (p_state_unitsize); without them a node divest would cost or earn but never lose capacity.
DIVEST_N: tuple[str, ...] = ("p_entity_max_units", "cohort_alive", "dd_forward", "nd_divest_set",
                              "p_entity_pre_existing", "p_state_unitsize")
# A retirement of model-built capacity counts in the periods of cdd_divest_built (into the capacity rows through
# v_divested_built), is bounded by its cohort (cohort_divest_built) and costs or earns cd_divest_built_cost /
# cd_divest_built_fixed; without them it would remove capacity for nothing or remove none.
DIVEST_BUILT: tuple[str, ...] = ("cdd_divest_built", "cohort_divest_built", "cd_divest_built_cost",
                                 "cd_divest_built_fixed")
# A life extension counts in the periods of cdd_extend (into the capacity rows through v_extended), is bounded by
# its parent cohort (cohort_extend, dd_forward) and costs cd_extend_cost / cd_extend_fixed; without them it would
# add capacity for nothing or add none.
EXTEND: tuple[str, ...] = ("cdd_extend", "cohort_extend", "dd_forward", "cd_extend_cost", "cd_extend_fixed")


def _check_covers_dt(d, field: str, feature: str, *, positive: bool = False, message: str | None = None) -> None:
    """Raise if the (d,)-keyed Param ``field`` lacks a period of ``dt`` (or, with ``positive``, has a
    value that is not finite and above 0).

    co2_max_total multiplies each period's emissions by its years (``p_years_represented_d``); a period
    without a row would silently drop out of the cap.  ``p_window_fraction`` is still required /
    validated here for co2_max_period (CO2_CAP) even though its cap row no longer multiplies by it
    directly (#38, Q22 replaced that with the cumulative ``p_co2_f_d_k`` carry, see
    ``_co2_period_cap_with_carry``) — the check is cheap dt-completeness insurance kept for whatever
    else may still read it. A zero fraction would set a window-fraction-driven cap to 0.  The loader
    builds both Params over the dt periods, so a gap is a loader bug.

    ``message``: for a gap the user can cause (an edited or partial ``solve_data`` CSV, a hand-built
    FlexData), the plain-English text of a ``FlexToolConfigError`` instead of the loader-bug
    ``ValueError``; ``{periods}`` in it is replaced by the missing periods ('p2025', 'p2030')."""
    frame = getattr(d, field).frame
    have = set(frame["d"].cast(pl.Utf8).to_list())
    missing = [p for p in d.dt["d"].cast(pl.Utf8).unique(maintain_order=True).to_list() if p not in have]
    if missing:
        if message is not None:
            raise FlexToolConfigError(message.format(periods=", ".join(f"'{p}'" for p in missing)))
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


def _check_covers_dt_steps(d, field: str, message: str) -> None:
    """Raise ``FlexToolConfigError(message)`` if the (d, t)-keyed Param ``field`` lacks a (period, time step)
    of ``dt``: every product with it (the cost weights ``op_factor`` / ``ev_factor``, the balance terms) is an
    inner join, so the missing steps would drop out of the LP without a word.  ``{steps}`` in ``message`` is
    replaced by the first missing pairs ((p2025, t0001), ...)."""
    frame = getattr(d, field).frame
    keys = d.dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    missing = keys.join(frame.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique(),
                        on=["d", "t"], how="anti", maintain_order="left")
    if missing.height:
        shown = ", ".join(f"({p}, {t})" for p, t in missing.head(5).iter_rows())
        if missing.height > 5:
            shown += f" and {missing.height - 5} more"
        raise FlexToolConfigError(message.format(steps=shown))


# Plain-English errors of the always-on coverage checks of ``_prepare`` (a partial ``solve_data`` CSV of a
# reloaded run or a hand-built FlexData; the database path builds every Param over the solve's periods and steps).
_INFLATION_OP_GAP = (
    "The discount factor of operating costs (p_inflation_factor_operations_yearly) has no value for period(s) "
    "{periods}; the operating costs of those periods would be left out. Add the period(s) to "
    "solve_data/p_inflation_factor_operations_yearly.csv, or re-run from the database.")
_STEP_DURATION_GAP = (
    "The time step duration (p_step_duration) has no value for (period, time step) {steps}; those time steps "
    "would be left out of the costs and the balances. Add them to solve_data/steps_in_use.csv, or re-run from "
    "the database.")
_TIMESTEP_WEIGHT_GAP = (
    "The time step weight (p_timestep_weight) has no value for (period, time step) {steps}; the operating costs "
    "of those time steps would be left out. Add them to solve_data/timestep_weight.csv, or re-run from the "
    "database.")
_PERIOD_SHARE_GAP = (
    "The share of the year of a period (complete_period_share_of_year) has no value for period(s) {periods}; the "
    "operating costs of those periods would be left out. Add the period(s) to "
    "solve_data/complete_period_share_of_year_calc.csv, or re-run from the database.")


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
    scale_the_objective: float = 1.0
    solve_name: str | None = None

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
    has_nsb: bool = False                # nodeStateBlock: bind_intraperiod_blocks storage, balanced per timeset block
    has_coarse: bool = False             # nodeCoarse: nodes of coarse new_stepduration groups
    has_coarse_proc: bool = False        # processCoarse: processes with one flow per coarse step (_coarse_flows)
    has_online_lin: bool = False
    has_online_int: bool = False
    has_online: bool = False
    has_minload_eff: bool = False
    has_piecewise: bool = False          # process_piecewise: conversion_piecewise rows (multi-line curves)
    has_startup_cost_lin: bool = False
    has_startup_cost_int: bool = False
    has_startup_draw: bool = False       # process_startup_node: start-up energy drawn from input nodes
    has_ramp: bool = False
    has_ramp_cost: bool = False          # process_source_sink_ramp_cost: arcs whose flow change is priced
    has_invest_p: bool = False
    has_divest_p: bool = False
    has_invest_n: bool = False
    has_divest_n: bool = False
    has_divest_built_p: bool = False     # cd_divest_built of processes: model-built capacity can be retired
    has_divest_built_n: bool = False     # ... of storage nodes (_retire_built)
    has_extend_p: bool = False           # cd_extend of processes: capacity can be extended (_extend)
    has_extend_n: bool = False           # ... of storage nodes
    has_ladder: bool = False
    has_reserve: bool = False
    has_capacity_margin: bool = False
    has_inertia: bool = False
    has_non_sync: bool = False
    has_cumulative_invest: bool = False

    # ── shared index sets ──
    pss_dt: pl.DataFrame | None = None               # (p, source, sink, d, t) of v_flow
    nodeBalance_dt: pl.DataFrame | None = None       # (n, d, t) of nodeBalance_eq
    nodeState_dt: pl.DataFrame | None = None         # (n, d, t) of v_state
    process_indirect_dt: pl.DataFrame | None = None  # (p, d, t) of conversion_indirect
    rp_start_idx: pl.DataFrame | None = None         # (n, d, t) of v_state_rp_start
    p_olin_idx: pl.DataFrame | None = None           # (p, d, t) of the linear online vars
    p_oint_idx: pl.DataFrame | None = None           # (p, d, t) of the integer online vars
    # (p, d, t) of the online rows and of the registered online / startup / shutdown columns: p_olin_idx /
    # p_oint_idx with a coarse process only at its coarse steps' first fine steps (_coarse_flows.domain).
    p_olin_rows: pl.DataFrame | None = None
    p_oint_rows: pl.DataFrame | None = None
    # (p, r, ud, n, d, t) of the registered v_reserve columns and of non_anticipativity_reserve:
    # prundt with a coarse process only at its coarse steps' first fine steps (_coarse_flows.domain).
    prundt_rows: pl.DataFrame | None = None
    twoway1var_arcs: pl.DataFrame | None = None      # (p, source, sink) of v_flow_back
    # Per-arc capacity (see _max_flow_rhs): bound of v_flow, RHS of maxFlow.
    invest_procs: pl.DataFrame | None = None         # processes with invest/divest vars
    flow_upper_rhs: Param | None = None
    flow_cap_mult: pl.DataFrame | None = None
    flow_var_upper: Param | None = None
    flow_bound_arcs: pl.DataFrame | None = None
    # Coarse processes (_coarse_flows.prepare): their coarse steps (p, d, b_first, t, w) and w as a Param.
    coarse_steps: pl.DataFrame | None = None
    coarse_w: Param | None = None

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
    v_divest_built_p: Var | None = None
    v_divest_built_n: Var | None = None
    v_divested_built_p: Var | None = None
    v_divested_built_n: Var | None = None
    v_extend_p: Var | None = None
    v_extend_n: Var | None = None
    v_extended_p: Var | None = None
    v_extended_n: Var | None = None
    v_flow_back: Var | None = None
    v_ramp_up: Var | None = None
    v_ramp_down: Var | None = None
    # The rows R of ramp_change_eq (_ramp_cost_rows; _ramp_rows columns), built once at declare time.
    ramp_cost_rows: pl.DataFrame | None = None
    v_exact_forward: Var | None = None
    exact_arcs: pl.DataFrame | None = None       # (p, source, sink, forward) of the exact connections
    v_reserve: Var | None = None
    vq_reserve: Var | None = None
    v_reserve_total: Var | None = None
    vq_capacity_margin: Var | None = None
    v_capacity_margin_slack_energy: Var | None = None
    vq_inertia: Var | None = None
    vq_non_synchronous: Var | None = None
    v_angle: Var | None = None
    angle_steps: pl.DataFrame | None = None      # (n, d, b_first, t) of the coarse-angle nodes (_dc_power_flow)
    group_row_steps: dict = field(default_factory=dict)  # row-rule family -> its coarse steps (_group_slack)
    v_trade: Var | None = None

    # ── shared expressions ──
    node_balance_flow_terms: dict = field(default_factory=dict)
    reserve_up_to_sink: Any = None          # Σ_r v_reserve up,   (p, sink, d, t)
    reserve_up_to_sink_units: Any = None    # the same for units only (connections: connection_reserve_* rows)
    reserve_down_to_sink: Any = None        # Σ_r v_reserve down, (p, sink, d, t)
    reserve_up_sink_parts: Any = None       # (v_reserve at sink, up selector)
    # Source-side reserves of units, per arc (p, source, sink, d, t), converted to the arc's flow:
    reserve_down_at_source: dict = field(default_factory=dict)  # down at the source: raises the flow
    reserve_up_at_source: dict = field(default_factory=dict)    # up at the source: lowers the flow
    reserve_down_input_arcs: pl.DataFrame | None = None  # (p, source, sink) carrying reserve_down_at_source
    cap: Any = None                         # _capacity.CapacityInService: the capacity in service
    op_factor: Any = None                   # weight of operational cost terms, (d, t)
    ev_factor: Any = None                   # weight of event cost terms (start-ups, ramp changes), (d, t)
    self_discharge: Any = None              # _self_discharge.LossCoefficients, None without a loss


def build_flextool(m, d, *, include_existing_fixed_cost: bool = False,
                   scale_the_objective: float = 1.0, solve_name: str | None = None) -> None:
    """Build the FlexTool LP into ``m`` from the data ``d``.

    Every block is conditional on its data: a feature with an empty index
    set emits nothing; a feature whose index set is populated but whose
    required fields are ``None`` raises ``ValueError`` (requirement tuples
    above).

    ``include_existing_fixed_cost``: add the fixed cost of existing
    capacity, Σ existing · fixed_cost · inflation_op · pd_branch_weight, to
    the objective as a constant (OBJ. existing_fixed_cost).  It changes the
    objective value, not the solution.  Off by default.
    ``scale_the_objective``: multiplier on the whole objective, constant
    included (1.0 = none).  Declared to polar-high
    (``Problem.set_objective_scale``), which then reads objective-valued
    solver options (targets, bounds, cutoffs, absolute gaps) in unscaled
    currency.
    ``solve_name``: the solve this LP is for; named in the configuration errors and warnings of the build.
    """
    b = _Build(m=m, d=d, include_existing_fixed_cost=include_existing_fixed_cost,
               scale_the_objective=scale_the_objective, solve_name=solve_name)
    # Objective steps return their term (or None); terms are summed in this
    # order.  Constraint steps return None.
    obj = None
    for step in (
        _prepare,                        # flags, data checks, index sets
        _declare_variables,
        _shared_expressions,
        # ── constraints, in emit order ──
        _eq_rp_storage_chain,
        _eq_rp_chain_ends,
        _eq_node_balance,
        _eq_node_balance_coarse,
        _eq_node_balance_period,
        _eq_node_balance_fix_quantity,
        _eq_storage_usage_fix,
        _eq_block_coupling,
        _eq_max_flow,
        _eq_max_output_sum,
        _dc_power_flow.add_constraints,
        _online.add_constraints,         # incl. online_fix_eq
        _exact.add_constraints,
        _unit_floors.add_constraints,
        _eq_ramp,
        _eq_ramp_cost,
        _eq_invest_max_units,
        _eq_invest_period_caps,
        _eq_invest_total_caps,
        _eq_retired_existing,
        _retire_built.add_constraints,
        _extend.add_constraints,
        _eq_conversion,
        _eq_conversion_piecewise,
        _eq_min_input_share,
        _eq_co2_max_period,
        _eq_co2_max_total,
        _eq_user_constraints,
        _eq_max_state,
        _eq_state_profiles,
        _eq_storage_start_binding_cyclic,
        _eq_storage_within_solve_path_end,
        _eq_storage_state_fan_start,
        _eq_storage_end_binding,
        _eq_storage_roll_end,
        _eq_storage_end_match,
        _eq_storage_reference_value,
        _eq_profiles,
        # ── core objective terms ──
        _obj_balance_slack,
        _obj_commodity,
        _obj_co2_price,
        _obj_co2_price_noEff,
        _obj_variable_cost,
        _obj_startup_cost,
        _obj_ramp_cost,
        _obj_storage_reference_price,
        _obj_invest_divest,
        _obj_existing_fixed_cost,
        _obj_existing_salvage,
        _obj_existing_replacement,
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
    _objective_options.set_objective_scale(m, scale_the_objective)
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
    # Objective coefficients are not covered by that floor.  A summed cost below
    # 2**floor(log2(OBJ_COEF_ZERO_REL_THRESHOLD · max|cost|)) becomes 0 (judged before
    # autoscale), so a float residue cannot stretch the cost range that autoscale
    # Layer 3 centres the objective on (docs/dev/scaling.md "Tiny cost coefficients").
    m.obj_coef_zero_rel_threshold = OBJ_COEF_ZERO_REL_THRESHOLD

    _check(d, ALWAYS, "always")
    # The operating-cost weight (op_factor, _shared_expressions) is an inner join of these four: a period or
    # time step one of them lacks would drop out of the objective without a word.
    _check_covers_dt(d, "p_inflation_op", "operating costs", message=_INFLATION_OP_GAP)
    _check_covers_dt(d, "p_period_share", "operating costs", message=_PERIOD_SHARE_GAP)
    _check_covers_dt_steps(d, "p_step_duration", _STEP_DURATION_GAP)
    _check_covers_dt_steps(d, "p_timestep_weight", _TIMESTEP_WEIGHT_GAP)

    # ── feature flags ──
    b.has_proc = _has_rows(d.process_source_sink)
    b.has_indirect = _has_rows(d.process_indirect)
    # (#19) A group can be priced purely for its removal credit (no emitting
    # flow at all) — widen the flag so that case still activates the price
    # term; _obj_co2_price guards each side independently.
    # A start-up draw (process_startup_node) is a fuel draw that is not a flow: a group whose only emitter is a
    # start-up draw still has its CO2 price term and caps.
    b.has_online_lin = _has_rows(d.process_online_linear)
    b.has_online_int = _has_rows(d.process_online_integer)
    b.has_online = b.has_online_lin or b.has_online_int
    b.has_startup_draw = (b.has_online and _has_rows(getattr(d, "process_startup_node", None))
                          and getattr(d, "p_startup_energy", None) is not None)
    b.has_co2_price = (_has_rows(d.flow_from_co2_priced)
                        or _has_rows(getattr(d, "flow_to_co2_priced", None))
                        or (b.has_startup_draw and _has_rows(getattr(d, "startup_from_co2_priced", None))))
    b.has_co2_cap_eff = _has_rows(d.flow_from_co2_capped)
    b.has_co2_cap_noEff = _has_rows(getattr(d, "flow_from_co2_capped_noEff", None))
    b.has_co2_cap = (b.has_co2_cap_eff or b.has_co2_cap_noEff
                     or (b.has_startup_draw and _has_rows(getattr(d, "startup_from_co2_capped", None))))
    b.has_co2_cap_total_eff = _has_rows(getattr(d, "flow_from_co2_capped_total", None))
    b.has_co2_cap_total_noEff = _has_rows(
        getattr(d, "flow_from_co2_capped_total_noEff", None))
    b.has_co2_cap_total = (b.has_co2_cap_total_eff or b.has_co2_cap_total_noEff
                           or (b.has_startup_draw
                               and _has_rows(getattr(d, "startup_from_co2_capped_total", None))))
    b.has_user_cstr = any(_has_rows(x) for x in (d.cdt_eq, d.cdt_le, d.cdt_ge))
    b.has_profile = any(_has_rows(x) for x in (d.process_profile_upper,
                                               d.process_profile_lower,
                                               d.process_profile_fixed))
    b.has_storage = _has_rows(d.nodeState)
    b.has_rp = _has_rows(d.nodeState_rp)
    b.has_nsb = _has_rows(d.nodeStateBlock)
    b.has_coarse = _has_rows(getattr(d, "nodeCoarse", None)) and _has_rows(getattr(d, "node_coarse_step", None))
    b.has_minload_eff = _has_rows(d.process_min_load_eff)
    b.has_piecewise = _has_rows(getattr(d, "process_piecewise", None))
    b.has_startup_cost_lin = _has_rows(d.pdt_online_linear)
    b.has_startup_cost_int = _has_rows(d.pdt_online_integer)
    b.has_ramp = any(_has_rows(getattr(d, f))
                     for f in ("process_source_sink_ramp_limit_sink_up",
                               "process_source_sink_ramp_limit_sink_down",
                               "process_source_sink_ramp_limit_source_up",
                               "process_source_sink_ramp_limit_source_down"))
    b.has_ramp_cost = _has_rows(getattr(d, "process_source_sink_ramp_cost", None))
    b.has_invest_p = _has_rows(d.pd_invest_set)
    b.has_divest_p = _has_rows(d.pd_divest_set)
    b.has_invest_n = _has_rows(d.nd_invest_set)
    b.has_divest_n = _has_rows(d.nd_divest_set)
    _retire_built.prepare(b)
    _extend.prepare(b)
    b.has_ladder = _commodity_ladder.has_feature(d)
    b.has_reserve = _has_rows(getattr(d, "reserve_upDown_group", None))
    b.has_capacity_margin = _group_slack.has_capacity_margin(d)
    b.has_inertia = _group_slack.has_inertia(d)
    b.has_non_sync = _group_slack.has_non_sync(d)
    b.has_cumulative_invest = _cumulative_invest.has_feature(d)

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
        _validate_online_constraint_coeff(d)
    if b.has_profile:
        _check(d, PROFILES, "profile_flow")
    if b.has_storage:
        _check(d, STORAGE, "storage")
    if b.has_ramp:
        _check(d, RAMP, "ramp_limit")
    if b.has_ramp_cost:
        _check(d, RAMP_COST, "ramp_cost")
    if b.has_invest_p or b.has_invest_n:
        _check(d, INVEST, "invest")
    if b.has_divest_p:
        _check(d, DIVEST, "divest")
    if b.has_divest_n:
        _check(d, DIVEST_N, "divest")
    if b.has_divest_built_p or b.has_divest_built_n:
        _check(d, DIVEST_BUILT, "retirement of model-built capacity")
    if b.has_extend_p or b.has_extend_n:
        _check(d, EXTEND, "life extension")
    if b.has_online:
        _validate_no_online_dc_lines(d)
        _check(d, ONLINE, "online")
        if _has_rows(getattr(d, "process_minload", None)):
            _check(d, MIN_LOAD, "min_load")
    if b.has_minload_eff:
        _check(d, MINLOAD_EFF, "min_load_efficiency")
    if b.has_piecewise:
        _check(d, PIECEWISE, "piecewise_linear")
    if b.has_startup_cost_lin:
        _check(d, STARTUP_COST_LINEAR, "startup_cost_linear")
    if b.has_startup_cost_int:
        _check(d, STARTUP_COST_INTEGER, "startup_cost_integer")
    if _has_rows(getattr(d, "process_startup_node", None)):
        _validate_startup_inputs(d)
    if _has_rows(getattr(d, "groupCapacityMargin", None)):
        _check(d, CAPACITY_MARGIN, "capacity_margin")

    # ── shared index sets ──
    b.pss_dt = compute_pss_dt(d) if b.has_proc else None
    b.nodeBalance_dt = compute_nodeBalance_dt(d)
    _validate_has_balance(d, b.nodeBalance_dt)
    b.nodeState_dt = compute_nodeState_dt(d) if b.has_storage else None
    b.self_discharge = _self_discharge.loss_coefficients(d, b.nodeState_dt) if b.has_storage else None
    b.process_indirect_dt = (compute_process_indirect_dt(d)
                             if b.has_indirect else None)
    _validate_storage_binding_disjoint(d)
    if b.has_storage:
        _validate_storage_start_end_values(d)
        _check_storage_accounting(b)
        _warn_storage_slack_bypass(b)
    if b.has_rp:
        b.rp_start_idx = compute_nodeState_rp_block_first_dt(d)
    if b.has_online_lin:
        b.p_olin_idx = d.p_online_dt.join(d.process_online_linear, on="p",
                                          how="inner")
        b.p_olin_rows = b.p_olin_idx
    if b.has_online_int:
        b.p_oint_idx = d.p_online_dt.join(d.process_online_integer, on="p",
                                          how="inner")
        b.p_oint_rows = b.p_oint_idx
    if b.has_proc:
        b.twoway1var_arcs = _two_way_one_var_arcs(d)
        _warn_one_variable_efficiency(b)
        procs = [s.select(pl.col("p").cast(pl.Utf8))
                 for s in (d.pd_invest_set if b.has_invest_p else None,
                           d.pd_divest_set if b.has_divest_p else None,
                           _retire_built.options(d, "p") if b.has_divest_built_p else None,
                           _extend.options(d, "p") if b.has_extend_p else None)
                 if s is not None]
        b.invest_procs = pl.concat(procs).unique() if procs else None
        (b.flow_upper_rhs, b.flow_cap_mult,
         b.flow_var_upper, b.flow_bound_arcs) = _max_flow_rhs(d, b.invest_procs)
        _coarse_flows.prepare(b)
        if b.has_online_lin:
            b.p_olin_rows = _coarse_flows.domain(b, b.p_olin_idx)
        if b.has_online_int:
            b.p_oint_rows = _coarse_flows.domain(b, b.p_oint_idx)
        if b.has_coarse_proc and b.flow_bound_arcs is not None:
            # A coarse arc is capped per coarse step by maxFlow_coarse, never by a per-step variable bound.
            fine = ~pl.col("p").cast(pl.Utf8).is_in(_coarse_flows.coarse_procs(b)["p"].cast(pl.Utf8).implode())
            bound = b.flow_bound_arcs.filter(fine)
            if bound.height < b.flow_bound_arcs.height:
                upper = b.flow_var_upper.frame.filter(fine)
                b.flow_bound_arcs = bound if bound.height > 0 else None
                b.flow_var_upper = (Param(b.flow_var_upper.dims, upper, name=b.flow_var_upper.name)
                                    if upper.height > 0 else None)


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


DC_ONLINE_MESSAGE = (
    "Connection(s) {names} have a startup_method but are in a dc_power_flow_with_angles group. A DC power-flow "
    "line's flow is set by the voltage angles; switching it off would need line-switching constraints that "
    "FlexTool does not have. Remove startup_method from these connections, or take them out of the DC "
    "power-flow group.")


def _validate_no_online_dc_lines(d) -> None:
    """Raise when a DC power-flow line has online variables (``startup_method`` ≠ ``no_startup``)."""
    dc = getattr(d, "connection_dc_power_flow", None)
    if not _has_rows(dc):
        return
    online = [f for f in (d.process_online_linear, d.process_online_integer) if _has_rows(f)]
    on_p = {str(x) for f in online for x in f.get_column("p").to_list()}
    hit = sorted({str(x) for x in dc.get_column("p").to_list()} & on_p)
    if hit:
        raise FlexToolConfigError(DC_ONLINE_MESSAGE.format(names=", ".join(hit)))


def _validate_online_constraint_coeff(d) -> None:
    """Raise when a unit / connection of the solve has a ``constraint_online_coeff`` but no online variables.

    Dropping the term instead could tighten the row (``flow − 100·online ≤ 0`` would become ``flow ≤ 0``), so
    it is a configuration error.  The online set holds the ``minimum_time_method`` promotion (units only).
    Coefficients on processes not in the solve, or pointing to constraints the solve does not write (for
    example ``is_enabled`` no), are ignored, as for the other coefficients.
    """
    coef = d.p_process_constraint_online_coeff
    if coef is None or not _has_rows(d.process_source_sink):
        return
    rows = _collected(coef.frame).select(pl.col("p").cast(pl.Utf8), pl.col("cn").cast(pl.Utf8))
    in_solve = {str(x) for x in d.process_source_sink.get_column("p").to_list()}
    online = {str(x) for f in (d.process_online_linear, d.process_online_integer) if _has_rows(f)
              for x in f.get_column("p").to_list()}
    units = ({str(x) for x in d.process_unit.get_column("p").to_list()} if _has_rows(d.process_unit) else set())
    written = {str(x) for f in (d.cdt_eq, d.cdt_le, d.cdt_ge) if _has_rows(f)
               for x in f.get_column("cn").unique().to_list()}
    missing = sorted({(p, cn) for p, cn in rows.iter_rows()
                      if p in in_solve and p not in online and cn in written})
    if missing:
        shown = ", ".join(f"{'unit' if p in units else 'connection'} '{p}' in constraint '{cn}'"
                          for p, cn in missing)
        how = ("linear or binary (or, on a unit, a minimum_time_method)" if any(p in units for p, _ in missing)
               else "linear or binary")
        raise FlexToolConfigError(
            f"constraint_online_coeff needs online variables, but these have none: {shown}. Set startup_method "
            f"{how}, or remove the coefficient.")


def _validate_storage_binding_disjoint(d) -> None:
    """Raise if a node is in more than one storage_bind_* set.

    nodeBalance_eq adds one state-change term per set, so a node in two
    sets would get two.
    """
    frames = [
        ("bind_within_timeblock", d.storage_bind_within_timeblock),
        ("bind_forward_only", d.storage_bind_forward_only),
        ("bind_within_solve", d.storage_bind_within_solve),
        ("bind_within_period", d.storage_bind_within_period),
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


def _start_periods(d, names: set[str]) -> dict[str, list[str]]:
    """``{node: [periods]}``: the periods of the nodes' first steps in this solve (nodeState_first_dt)."""
    out: dict[str, list[str]] = {}
    if not _has_rows(d.nodeState_first_dt):
        return out
    for n, p in d.nodeState_first_dt.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).unique().iter_rows():
        if n in names:
            out.setdefault(n, []).append(p)
    return out


def _check_storage_accounting(b: _Build) -> None:
    """Storage start and end conditions that let a solve use energy it never produced (storage_semantics, Q7).

    * use_reference_value below the start on storage the solve can invest in: warned where the reference row is
      built (_warn_reference_below_start).  (fix_start_end with storage_state_end below storage_state_start is not
      warned about: a drawdown the user stated; the matching end still caps the investment.)
    * fix_start on a forward chain (bind_forward_only, bind_forward_only_blended_weights) whose capacity at the start
      is not a decision of the solve, with an open end (no fix_end, no use_reference_value, no upper-level fix): the
      start is an endowment the run may use up; a warning with its size (owner decision).
    * A forward chain without representative periods whose explicit start (fix_start, or the reference start of
      use_reference_value) is on capacity the solve invests in or extends during the start period, in a solve that
      does not hold the last step of the model: the matching end (storage_state_end_match_share) needs both ends in
      one solve, so it is not applied; a warning.  A free start is not warned about: the later solve that reaches
      the end of the model matches it (_run_model._sequence_match_rows, _anchor_cyclic_in_rolling).  (A
      representative-period chain always gets rp_inter_period_end_match.)
    "Investable" counts life extensions in the start period too (:func:`_start_decision_periods`).
    Each warning prints one line per solve, naming how many nodes and the first (owner rule); the values per node
    go to the log at debug level.
    """
    from flextool.engine import _behaviour_changes
    from flextool import run_warnings
    d = b.d
    carried = set()
    if _has_rows(_self_discharge.carried_nodes(d)):
        carried = set(str(n) for n in _self_discharge.carried_nodes(d)["n"].to_list())
    fix_start = set(str(n) for n in d.storage_fix_start["n"].to_list()) if _has_rows(d.storage_fix_start) else set()
    fix_end = (set(str(n) for n in d.storage_fix_end["n"].to_list())
               if _has_rows(getattr(d, "storage_fix_end", None)) else set())
    share = ({str(n): float(v) for n, v in d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), "value").iter_rows()}
             if d.p_state_start is not None else {})
    starting = (fix_start | set(str(n) for n in d.nodeState["n"].to_list())) - carried
    periods = _start_periods(d, starting)
    invest = _start_decision_periods(b)

    def investable(n: str) -> bool:
        return bool(invest.get(n, set()) & set(periods.get(n, [])))

    where = (f"solve '{_behaviour_changes.complete_solve_of(b.solve_name)}'" if b.solve_name else "this solve")
    _warn_reference_start_mismatch(b, share)

    forward = set()
    for frame in (d.storage_bind_forward_only, d.storage_bind_forward_only_blended_weights):
        if _has_rows(frame):
            forward |= set(str(n) for n in frame["n"].to_list())
    forward -= carried
    if not forward:
        return
    nested = set()
    steps = _nested_fixed_end(b, with_price=True)
    if steps is not None:
        nested = set(steps["n"].to_list())
    reference = (set(str(n) for n in d.storage_use_reference_value["n"].to_list())
                 if _has_rows(d.storage_use_reference_value) else set())
    open_end = forward - nested - reference - fix_end
    endowed = sorted(n for n in open_end & fix_start if n in share and not investable(n))
    if endowed and d.p_state_existing_capacity is not None:
        exist = {(str(n), str(p)): float(v) for n, p, v in d.p_state_existing_capacity.frame.select(
            pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), "value").iter_rows()}
        sizes, mwh = {}, {}
        for n in endowed:
            first = sorted(periods.get(n, []))
            mwh[n] = share[n] * exist.get((n, first[0]), 0.0) if first else 0.0
            if mwh[n] > 0:
                sizes[n] = f"{n} {mwh[n]:.6g} MWh"
        if sizes:
            # One line per solve (the first solve of each level: a carried start is not an endowment).
            total = sum(mwh[n] for n in sizes)
            run_warnings.warn_grouped(
                "W-0534", sorted(sizes), solve=b.solve_name,
                details=f"Endowment in {where}: {', '.join(sizes[n] for n in sorted(sizes))} "
                        f"({total:.6g} MWh in total, on the existing capacity before any retirement).")
    if not d.storage_solve_at_model_end:
        # A free start (fix_nothing, not use_reference_value) is matched by the later solve that reaches the end of
        # the model and continues the chain (_run_model._sequence_match_rows, _anchor_cyclic_in_rolling: the level
        # the model started the node from at its nesting level), so only an explicit start on capacity the solve
        # decides goes unmatched.
        rp_nodes = set(str(n) for n in d.nodeState_rp["n"].to_list()) if _has_rows(d.nodeState_rp) else set()
        decided = sorted(n for n in forward - nested - rp_nodes if investable(n) and (n in fix_start or n in reference))
        if decided:
            run_warnings.warn_grouped("W-0535", decided, solve=b.solve_name)


def _warn_reference_start_mismatch(b: _Build, share: dict) -> None:
    """Warn about use_reference_value nodes whose fix_start share differs from the reference value at their first
    step (owner decision, Phase 5): fix_start wins.  A node without fix_start starts from the reference
    (``preprocess.flex_data._reference_start_as_fix_start``), so the two agree there."""
    from flextool.engine.preprocess.flex_data import reference_start_shares
    ref = reference_start_shares(b.d)
    off = sorted(n for n, v in ref.items() if n in share and abs(share[n] - v) > 1e-9)
    if not off:
        return
    from flextool import run_warnings
    run_warnings.warn_grouped(
        "W-0536", off, solve=b.solve_name,
        details={n: f"storage_state_start {share[n]:.6g}, reference {ref[n]:.6g}" for n in off})


def _warn_storage_slack_bypass(b: _Build) -> None:
    """Warn about storage nodes whose balance slack can act as a bypass of a hard balance.

    A storage node with balance slack (vq_state_up / vq_state_down: penalty_method not off) that a unit or a
    connection links to a node with penalty_method off: the storage slack supplies or absorbs energy that flows on
    to (or comes from) the node meant to balance exactly, so the hard balance is bypassed at the storage node's
    penalty, which can be far below what an unmet demand elsewhere should cost.  Only pure storage nodes (no own
    inflow or demand, e.g. a battery): a carrier node's slack is the intended outlet of its unmet demand.  One
    printed line per solve (the rolls of a rolling solve count as one solve), the node pairs in the log.
    """
    from flextool import run_warnings
    d = b.d
    off = getattr(d, "nodeBalance_penalty_off", None)
    if not (_has_rows(off) and _has_rows(d.nodeState) and _has_rows(d.process_source_sink)):
        return
    off_nodes = set(str(n) for n in off["n"].to_list())
    balanced = set(str(n) for n in d.nodeBalance["n"].to_list()) if _has_rows(d.nodeBalance) else set()
    if _has_rows(d.nodeBalancePeriod):
        balanced |= set(str(n) for n in d.nodeBalancePeriod["n"].to_list())
    slack_storage = (set(str(n) for n in d.nodeState["n"].to_list()) & balanced) - off_nodes
    # Only pure storage (e.g. a battery): a node with its own inflow or demand is a carrier node, whose slack is the
    # intended outlet of its unmet demand (owner decision, Phase 5).
    inflow = getattr(d, "p_inflow", None)
    if slack_storage and inflow is not None and _has_rows(inflow.frame):
        slack_storage -= set(str(n) for n in inflow.frame.filter(pl.col("value").abs() > 0)
                             .select(pl.col("n").cast(pl.Utf8)).unique()["n"].to_list())
    if not slack_storage:
        return
    nodes_of: dict[str, set[str]] = {}
    for p, src, snk in d.process_source_sink.select(
            pl.col("p").cast(pl.Utf8), pl.col("source").cast(pl.Utf8), pl.col("sink").cast(pl.Utf8)).iter_rows():
        nodes_of.setdefault(p, set()).update((src, snk))
    linked: dict[str, set[str]] = {}
    for nodes in nodes_of.values():
        for n in nodes & slack_storage:
            hard = (nodes & off_nodes) - {n}
            if hard:
                linked.setdefault(n, set()).update(hard)
    if not linked:
        return
    run_warnings.warn_grouped(
        "W-0537", sorted(linked), solve=b.solve_name,
        details={n: f"linked to penalty_method off node(s) {_node_list(sorted(linked[n]))}" for n in sorted(linked)})


STORAGE_START_END_MESSAGE = (
    "Storage node(s) fix the storage state without the value to fix it at: {items}. Set storage_state_start "
    "(0-1) on these nodes, or set storage_start_end_method to fix_nothing (or fix_end) if the start should not be "
    "fixed; likewise storage_state_end (0-1), or fix_nothing (or fix_start) if the end should not be fixed. "
    "Databases exported to Excel by earlier FlexTool versions may carry fix_start copied from the old default.")


def _param_nodes(p) -> set[str]:
    """The nodes a node Param ``(n,)`` has a row for."""
    if p is None:
        return set()
    return {str(n) for n in p.frame.get_column("n").to_list()}


def _validate_storage_start_end_values(d) -> None:
    """Raise when a storage node fixes its start (fix_start, fix_start_end) without its own storage_state_start,
    or its end (fix_end, fix_start_end) without its own storage_state_end; every such node is named at once.

    The database upgrade to version 77 rewrote the nodes where it could tell what the method did; the nodes it
    could not decide for (scenarios used the stored value differently) are in its marker and named here."""
    storage = {str(n) for n in d.nodeState.get_column("n").to_list()} if _has_rows(d.nodeState) else set()
    if not storage:
        return
    fix_start = ({str(n) for n in d.storage_fix_start.get_column("n").to_list()}
                 if _has_rows(d.storage_fix_start) else set())
    fix_end = ({str(n) for n in d.storage_fix_end.get_column("n").to_list()}
               if _has_rows(getattr(d, "storage_fix_end", None)) else set())
    missing_start = sorted((fix_start & storage) - _param_nodes(d.p_state_start))
    missing_end = sorted((fix_end & storage) - _param_nodes(getattr(d, "p_state_end", None)))
    if not missing_start and not missing_end:
        return
    methods: dict[str, str] = {}
    for n in fix_start:
        methods[n] = "fix_start_end" if n in fix_end else "fix_start"
    for n in fix_end - fix_start:
        methods[n] = "fix_end"
    items = []
    for n in sorted(set(missing_start) | set(missing_end)):
        params = [p for p, miss in (("storage_state_start", missing_start), ("storage_state_end", missing_end))
                  if n in miss]
        items.append(f"{n} ({methods[n]}, missing {' and '.join(params)})")
    message = STORAGE_START_END_MESSAGE.format(items="; ".join(items))
    from flextool.engine import _behaviour_changes
    undecided = _behaviour_changes.marker_json("storage_fix_method_rewritten").get("undecided") or []
    named = set(missing_start) | set(missing_end)
    for entry in undecided:
        node, alternative, _method, scenarios = entry[0], entry[1], entry[2], entry[3]
        if node in named:
            message += (f" The database upgrade could not choose for {node} (alternative {alternative}) because "
                        f"scenarios {', '.join(scenarios)} used it differently.")
    raise FlexToolConfigError(message)


def _warn_one_variable_efficiency(b: _Build) -> None:
    """W-0589 (once per solve): the one-variable connections (twoway1var_arcs) with an efficiency ≠ 1 in the solve.

    Their one signed flow takes f · slope at the source and gives f at the sink in both directions
    (nodeBalance_back_flow_terms), so the efficiency is a conversion (e.g. MWh to t), not a loss.  Only the
    connections between nodes in different units keep it: the load boundary sets the slope of the others to 1
    (``_derived.apply_one_variable_efficiency``, W-0591).  Named: the connection, with its efficiency range over
    the solve's (d, t) in the details.
    """
    d = b.d
    if not _has_rows(b.twoway1var_arcs) or d.p_slope is None or not _has_rows(d.process_source_sink_eff):
        return
    arcs = join_aligned(b.twoway1var_arcs.select("p", "source", "sink"),
                        d.process_source_sink_eff.select("p", "source", "sink"),
                        on=["p", "source", "sink"], how="semi")
    if arcs.height == 0:
        return
    slope = d.p_slope.frame
    slope = slope.collect() if isinstance(slope, pl.LazyFrame) else slope
    if _has_rows(getattr(d, "dt", None)):
        slope = join_aligned(slope, d.dt.select("d", "t").unique(), on=["d", "t"], how="semi")
    names = sorted({str(p) for p in arcs.get_column("p").to_list()})
    hit = (slope.filter(pl.col("p").cast(pl.Utf8).is_in(names) & ((pl.col("value") - 1.0).abs() > 1e-9))
           .group_by(pl.col("p").cast(pl.Utf8))
           .agg((1.0 / pl.col("value")).min().alias("lo"), (1.0 / pl.col("value")).max().alias("hi"))
           .sort("p"))
    if hit.height == 0:
        return
    details = {p: (f"efficiency {lo:g}" if lo == hi else f"efficiency {lo:g} to {hi:g}")
               for p, lo, hi in hit.iter_rows()}
    from flextool import run_warnings
    run_warnings.warn_grouped("W-0589", list(details), solve=b.solve_name, details=details)


def _two_way_one_var_arcs(d) -> "pl.DataFrame | None":
    """(p, source, sink) arcs whose reverse flow is ``v_flow_back``.

    Union of the method_2way_1var_off arcs (``process_source_sink_2way_1var``)
    and the DC power-flow arcs; the latter come straight from the DC
    module so they are covered even when the projection field is absent.
    """
    parts = []
    two_way = getattr(d, "process_source_sink_2way_1var", None)
    if _has_rows(two_way):
        parts.append(two_way.select("p", "source", "sink"))
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
    # ─── VAR. ─── v_flow (coarse processes) ──────────────────────────────────
    # v_flow[p,s,k,d,b] ≥ 0   ∀ p ∈ processCoarse, (p,s,k) arcs, (d,b) the coarse steps of p (b = first fine step)
    # The flow of a coarse process: one rate per coarse step, the same on its fine steps.  b.v_flow is indexed
    # on every fine step (_coarse_flows.alias): a fine row of a coarse process reads its step's column.
    if b.has_proc:
        upper = b.flow_var_upper if b.flow_var_upper is not None else float("inf")
        b.v_flow = _coarse_flows.alias(
            b, m.add_var("v_flow", ("p", "source", "sink", "d", "t"), _coarse_flows.domain(b, b.pss_dt),
                         lower=0.0, upper=upper),
            b.pss_dt)

    # ─── VAR. ─── v_ramp_up / v_ramp_down ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_ramp_up[p,s,k,d,t] ≥ 0, v_ramp_down[p,s,k,d,t] ≥ 0   ∀ (p,s,k,d,t) ∈ R
    #   R = the rows of ramp_change_eq (_ramp_cost_rows): (p,s,k) ∈ process_source_sink_ramp_cost with a ramp
    #   predecessor at (d,t) (dt_ramp_link or dt_ramp_carry; for a coarse process t = b, its coarse step)
    # The increase and the decrease of v_flow from the ramp predecessor, in units of unitsize (as v_flow).  Priced
    # equally by the ramp cost, so at most one of them is > 0 at an optimum (up + down = |Δv_flow|).  Declared
    # only when some arc has a ramp cost.
    if b.has_proc and b.has_ramp_cost:
        b.ramp_cost_rows = _ramp_cost_rows(b)
        if b.ramp_cost_rows.height > 0:
            r_idx = b.ramp_cost_rows.select("p", "source", "sink", "d", "t")
            b.v_ramp_up = m.add_var("v_ramp_up", ("p", "source", "sink", "d", "t"), r_idx, lower=0.0)
            b.v_ramp_down = m.add_var("v_ramp_down", ("p", "source", "sink", "d", "t"), r_idx, lower=0.0)

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
    # v_state_inter[n,d,b] ≥ 0        ∀ n ∈ nodeState_rp, (d,b) ∈ rp_base_period
    # v_state_rp_start[n,d,t] ≥ 0     ∀ n ∈ nodeState_rp, (d,t) ∈ rp_block_first
    # Representative-period storage, in state units: the state at the end of
    # base period b of period d, and the free starting state of each
    # representative block.  Every period has its own base periods (two
    # periods may share base-period names).  Upper bounds are the rows
    # rp_inter_period_max_state and maxState_rp_start.
    if b.has_rp:
        b.v_state_inter = m.add_var("v_state_inter", ("n", "d", "b"),
                                    d.nodeState_rp.join(d.rp_base_period_set, how="cross").select("n", "d", "b"),
                                    lower=0.0)
        b.v_state_rp_start = m.add_var("v_state_rp_start", ("n", "d", "t"), b.rp_start_idx, lower=0.0)

    # Unit commitment (v_online, v_startup, v_shutdown): declared by _online.
    _online.add_variables(b)

    # ─── VAR. ─── v_invest_p / v_divest_p / v_invest_n / v_divest_n ──────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] ≥ 0 ∀ (p,d) ∈ pd_invest_set;  v_divest_p[p,d] ≥ 0 ∀ (p,d) ∈ pd_divest_set
    # v_invest_n[n,d] ≥ 0 ∀ (n,d) ∈ nd_invest_set;  v_divest_n[n,d] ≥ 0 ∀ (n,d) ∈ nd_divest_set
    # Capacity built / retired in period d, as a count of unitsize (process)
    # or state unitsize (storage node).  Only pre-existing capacity can be retired:
    # pd/nd_divest_set hold the periods where the entity has some (before its end of life).  Upper bounds are the
    # maxInvest_var_bound / maxDivest_var_bound rows.
    if b.has_invest_p:
        b.v_invest_p = m.add_var("v_invest_p", ("p", "d"), d.pd_invest_set)
    if b.has_divest_p:
        b.v_divest_p = m.add_var("v_divest_p", ("p", "d"), d.pd_divest_set)
    if b.has_invest_n:
        b.v_invest_n = m.add_var("v_invest_n", ("n", "d"), d.nd_invest_set)
    if b.has_divest_n:
        b.v_divest_n = m.add_var("v_divest_n", ("n", "d"), d.nd_divest_set)
    # Retirement of model-built capacity (v_divest_built, v_divested_built): declared by _retire_built.
    _retire_built.add_variables(b)
    # Life extension (v_extend, v_extended): declared by _extend.
    _extend.add_variables(b)

    # Reserve variables (v_reserve, vq_reserve, v_reserve_total): declared by _reserve.
    _reserve.add_variables(b)

    # ─── VAR. ─── v_flow_back ────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow_back[p,source,sink,d,t] ≥ 0   ∀ (p,source,sink) ∈ twoway1var_arcs, (d,t) ∈ dt
    # Reverse flow (sink → source) of method_2way_1var_off and DC power-flow
    # arcs, in units of unitsize; the signed flow is v_flow − v_flow_back.
    # One variable serves both the balance and dc_flow_eq.  A coarse process has one per coarse step, as v_flow.
    if _has_rows(b.twoway1var_arcs):
        back_idx = b.twoway1var_arcs.join(d.dt, how="cross").select("p", "source", "sink", "d", "t")
        if back_idx.height > 0:
            b.v_flow_back = _coarse_flows.alias(
                b, m.add_var("v_flow_back", ("p", "source", "sink", "d", "t"), _coarse_flows.domain(b, back_idx),
                             lower=0.0),
                back_idx)

    # Direction switch of exact connections (v_exact_forward), DC power-flow angles (v_angle) and
    # commodity-ladder trades (v_trade): declared by their feature modules.
    _exact.add_variables(b)
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
    b.cap = _capacity.build_capacity(b)
    # Weight of every operational cost term per (d, t): step_duration ·
    # timestep_weight · inflation_op / period_share, times the branch
    # weight in stochastic solves.
    b.op_factor = (d.p_step_duration * d.p_timestep_weight
                   * d.p_inflation_op / d.p_period_share)
    if d.pdt_branch_weight is not None:
        b.op_factor = b.op_factor * d.pdt_branch_weight
    # Weight of event cost terms (start-ups, ramp changes, start-up energy): timestep_weight · inflation_op /
    # period_share, times the branch weight; no step_duration (an event is not a rate).
    b.ev_factor = d.p_timestep_weight * d.p_inflation_op / d.p_period_share
    if d.pdt_branch_weight is not None:
        b.ev_factor = b.ev_factor * d.pdt_branch_weight


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
        # maxFlow and maxFlow_online couple the upward reserve of units only: a connection's reserve is limited
        # by its net position at the node (connection_reserve_* rows, _reserve).  Hand-built data without
        # process_unit keeps the process-generic coupling (_common.connection_procs is then None).
        if _connection_procs(b.d) is None:
            b.reserve_up_to_sink_units = b.reserve_up_to_sink
        else:
            units = b.d.process_unit.select(pl.col("p").cast(pl.Utf8).alias("__p")).unique()
            unit_sink = (up_sink.with_columns(pl.col("p").cast(pl.Utf8).alias("__p"))
                         .join(units, on="__p", how="semi").drop("__p"))
            if unit_sink.height > 0:
                b.reserve_up_to_sink_units = Sum(Where(at_sink, unit_sink), over=("r", "ud"))

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


def _without_no_limit_rows(over: pl.DataFrame, rhs: "Param | None") -> pl.DataFrame:
    """``over`` without the rows whose ``rhs`` is +inf: a capacity of inf is no limit, so such a row is left
    out (the Benders splitter gives a cross-region half-flow an inf capacity; the master holds the real one).
    Keys are matched as Utf8 on the dims ``rhs`` and ``over`` share; ``over`` comes back unchanged (same object)
    when ``rhs`` has no inf value."""
    if rhs is None or over.height == 0:
        return over
    inf = rhs.frame.filter(pl.col("value") == float("inf"))
    if inf.height == 0:
        return over
    keys = [k for k in rhs.dims if k in over.columns]
    return (_utf8_keys(over.lazy(), keys)
            .join(_utf8_keys(inf.lazy().select(keys), keys).select([f"_{k}" for k in keys]).unique(),
                  on=[f"_{k}" for k in keys], how="anti")
            .select(over.columns).collect())


def _source_reserve_arcs(unit_arcs: pl.DataFrame, sel: pl.DataFrame, *, direct: bool) -> pl.DataFrame:
    """The unit arcs ``(p, source, sink)`` of the class of ``sel`` (direct input→output arcs, or arcs whose
    v_flow is the input) whose ``(p, source)`` carries a reserve of ``sel``."""
    keys = ["p", "source", "sink"]
    a = _utf8_keys(unit_arcs.lazy(), keys)
    a = a.filter(pl.col("_source") != pl.col("_p"))
    a = a.filter((pl.col("_sink") != pl.col("_p")) if direct else (pl.col("_sink") == pl.col("_p")))
    hit = _utf8_keys(sel.lazy().select("p", "source"), ["p", "source"]).select("_p", "_source").unique()
    return a.join(hit, on=["_p", "_source"], how="semi").select(keys).collect()


# ---------------------------------------------------------------------------
# Equations


def _eq_rp_storage_chain(b: _Build) -> None:
    """Representative-period storage: the base-period state chain of every period and its bounds."""
    if not (b.has_storage and b.has_rp):
        return
    m, d = b.m, b.d
    v_inter = b.v_state_inter
    b_dtype = v_inter.frame.schema["b"]
    d_dtype = v_inter.frame.schema["d"]
    has_rep_blocks = d.rp_base__rep is not None and _has_rows(d.rp_block_first) and _has_rows(d.p_rp_last_step)
    nodes = d.nodeState_rp

    def typed(frame: pl.DataFrame) -> pl.DataFrame:
        """Base-period labels (b, b_prev) onto v_state_inter's b dtype, periods (d, d_prev) onto its d dtype."""
        return frame.with_columns(
            [pl.col(c).cast(b_dtype, strict=False) for c in ("b", "b_prev") if c in frame.columns]
            + [pl.col(c).cast(pl.Utf8).cast(d_dtype, strict=False) for c in ("d", "d_prev") if c in frame.columns])

    first = typed(d.rp_base_first) if _has_rows(d.rp_base_first) else None    # (d, b)
    last = typed(d.rp_base_last) if _has_rows(d.rp_base_last) else None       # (d, b)
    period_previous = typed(d.rp_period_previous) if _has_rows(d.rp_period_previous) else None
    inter_prev = _var_view(v_inter, "__prev", {"d": "d_prev", "b": "b_prev"})
    drift = _rp_weighted_drift(b, nodes) if has_rep_blocks else None          # (n, d, b)
    keys = ("n", "d", "b", "d_prev", "b_prev")

    # ─── EQ. ─── rp_inter_period_balance ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − v_state_inter[n,d′,b′] = drift[n,d,b]
    #   drift[n,d,b] = Σ_{(d,b,r) ∈ rp_base__rep, (d,r) ∈ rp_block_first, d ∈ periods of dt}
    #                    rp_weight[d,b,r] · (v_state[n,d,last_step[d,r]] − v_state_rp_start[n,d,r])
    #   ∀ n ∈ nodeState_rp, (d,b,b′) ∈ rp_base_chain, d′ = d                         (within a period)
    #   ∀ n ∈ bind_within_solve_blended_weights ∪ bind_forward_only_blended_weights,
    #     (d,d′) ∈ rp_period_previous, b = first base period of d, b′ = last of d′    (period to period)
    # Chains the storage state across the base periods of each period: a base
    # period is represented by weighted representative blocks r, and the
    # boundary state moves by the weighted net change of those blocks.  The
    # across-solve variants continue from the last base period of the previous
    # period in solve order, as bind_within_solve / bind_forward_only continue
    # v_state across periods.  All terms are in state units.
    edges = []
    if _has_rows(d.rp_base_chain):
        chain = typed(d.rp_base_chain).with_columns(d_prev=pl.col("d"))
        edges.append(nodes.join(chain.select("d", "b", "d_prev", "b_prev"), how="cross"))
    across = [f for f in (d.storage_bind_within_solve_blended_weights, d.storage_bind_forward_only_blended_weights)
              if _has_rows(f)]
    if across and period_previous is not None and first is not None and last is not None:
        cross = (period_previous
                 .join(first, on="d", how="inner")
                 .join(last.rename({"d": "d_prev", "b": "b_prev"}), on="d_prev", how="inner")
                 .select("d", "b", "d_prev", "b_prev"))
        edges.append(pl.concat(across, how="vertical").unique().join(cross, how="cross"))
    if drift is not None and edges:
        over = pl.concat([e.select(*keys) for e in edges], how="vertical").unique(maintain_order=True)
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
    # v_state_inter[n,d,b_first(d)] − v_state_inter[n,d′,b_last(d′)] = drift[n,d,b_first(d)]
    #   n ∈ bind_within_period_blended_weights: every rp period d, d′ = d
    #   n ∈ bind_within_solve_blended_weights: (d,d′) ∈ d_first × d_last, d_first = the rp periods without a
    #     predecessor in rp_period_previous, d_last = the rp periods that are no period's predecessor
    # Closes the chain: the last boundary state connects back to the first.
    # bind_within_period_blended_weights closes every period on itself,
    # bind_within_solve_blended_weights closes once per path of the solve:
    # deterministic, one pair (first period, last period).  In a stochastic
    # solve the rp periods form a tree (rp_period_previous): a fan after the
    # trunk period (the shared period before the split) has one d_first and
    # one d_last per forecast branch, so every branch's last period ends at
    # the level the trunk period starts from; a fan in the first period has
    # N roots and N leaves, and the N × N pairs are the loop of each path plus
    # an equal chain start of all the roots.
    # bind_forward_only_blended_weights has no closure.
    closures = []
    if first is not None and last is not None:
        within_period = d.storage_bind_within_period_blended_weights
        if _has_rows(within_period):
            pairs = first.join(last.rename({"b": "b_prev"}), on="d", how="inner").with_columns(d_prev=pl.col("d"))
            closures.append(within_period.join(pairs.select("d", "b", "d_prev", "b_prev"), how="cross"))
        within_solve = d.storage_bind_within_solve_blended_weights
        if _has_rows(within_solve):
            periods = first.select("d").unique()
            if period_previous is not None:
                d_first = periods.join(period_previous.select("d"), on="d", how="anti")
                d_last = periods.join(period_previous.select(pl.col("d_prev").alias("d")), on="d", how="anti")
            else:
                d_first = d_last = periods
            pair = (first.join(d_first, on="d", how="inner")
                    .join(last.join(d_last, on="d", how="inner").rename({"d": "d_prev", "b": "b_prev"}),
                          how="cross"))
            closures.append(within_solve.join(pair.select("d", "b", "d_prev", "b_prev"), how="cross"))
    if drift is not None and closures:
        over = pl.concat([c.select(*keys) for c in closures], how="vertical").unique(maintain_order=True)
        m.add_cstr(
            "rp_inter_period_cyclic",
            over      = over,
            sense     = "==",
            lhs_terms = {"d_cyclic": Where(v_inter, over) - Where(inter_prev, over),
                         "neg_rp_drift_first": -Where(drift, over)},
            rhs_terms = {},
        )

    # ─── EQ. ─── rp_inter_period_fan_start ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b_first(d)] − drift[n,d,b_first(d)]
    #   − v_state_inter[n,d″,b_first(d″)] + drift[n,d″,b_first(d″)] = 0
    #   ∀ n ∈ bind_forward_only_blended_weights, n ∉ the nodes of rp_inter_period_start_fix,
    #     d = the first root (sorted), d″ ∈ the other roots
    #   roots = the rp periods of period_first_of_solve (the forecast branch periods of a fan in the solve's
    #     first period; one root without a fan: no rows), read as the periods of nodeState_first_dt, which is
    #     built from period_first_of_solve
    # Every forecast of a fan in the solve's first period starts its base-period
    # chain from the same level: the chain start v_state_inter − drift of the
    # first base period, which no other row binds in a forward-only chain.  A
    # fix_start node is not included: rp_inter_period_start_fix pins each
    # root's chain start at the share of that forecast's own capacity.  The
    # bind_within_solve_blended_weights roots are
    # already tied by rp_inter_period_cyclic; the base-period levels inside a
    # tie window need no tie of their own (they follow from the tied net flows,
    # the tied block starts and this shared chain start).
    if (drift is not None and first is not None and _has_rows(d.storage_bind_forward_only_blended_weights)
            and _has_rows(d.nodeState_first_dt)):
        root_names = sorted(set(str(x) for x in d.nodeState_first_dt["d"].to_list())
                            & set(str(x) for x in first["d"].to_list()))
        if len(root_names) >= 2:
            roots = first.filter(pl.col("d").cast(pl.Utf8).is_in(root_names))
            ref = roots.filter(pl.col("d").cast(pl.Utf8) == root_names[0])
            other = (roots.filter(pl.col("d").cast(pl.Utf8) != root_names[0])
                     .rename({"d": "d_prev", "b": "b_prev"}))
            fan_nodes = d.storage_bind_forward_only_blended_weights.select("n").unique()
            pinned = _rp_start_pin_nodes(b)
            if pinned is not None:
                fan_nodes = fan_nodes.filter(~pl.col("n").cast(pl.Utf8).is_in(pinned["n"].cast(pl.Utf8).implode()))
            over = (fan_nodes
                    .join(ref.select("d", "b"), how="cross")
                    .join(other.select("d_prev", "b_prev"), how="cross")
                    .select(*keys).sort(pl.col("d_prev").cast(pl.Utf8)))
            drift_prev = _rp_weighted_drift(b, nodes, at_prev=True)
            if over.height > 0 and drift_prev is not None:
                m.add_cstr(
                    "rp_inter_period_fan_start",
                    over      = over,
                    sense     = "==",
                    lhs_terms = {"d_fan": Where(v_inter, over) - Where(inter_prev, over),
                                 "neg_rp_drift": -Where(drift, over),
                                 "rp_drift_prev": Where(drift_prev, over)},
                    rhs_terms = {},
                )

    # ─── EQ. ─── rp_inter_period_max_state / maxState_rp_start ───────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] + retired[n,d]
    #   − invested[n,d]  ≤  state_upper[n,d]
    #   ∀ n ∈ nodeState_rp, (d,b) ∈ rp_base_period, d ∈ periods of dt
    # v_state_rp_start[n,d,t] + (same invest/divest terms)  ≤  state_upper[n,d]
    #   ∀ (n,d,t) ∈ nodeState_rp × rp_block_first
    # Capacity bounds of the boundary and block-start states, as maxState
    # bounds v_state: a base period of d is bounded by d's capacity.
    # state_upper = capacity / state_unitsize, so every term is in state units.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if _has_rows(d.rp_base_period_set):
        base_periods = typed(d.rp_base_period_set)
        if _has_rows(d.period_in_use_set):
            base_periods = base_periods.join(
                dispatch_periods(d).select(pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False)).unique(),
                on="d", how="inner")
        over = nodes.join(base_periods.select("d", "b"), how="cross").select("n", "d", "b")
        if over.height > 0:
            lhs: dict = {"state_inter": Where(v_inter, over)}
            lhs.update(_capacity.node_capacity_change_terms(b, nodes, "_rp"))
            m.add_cstr(
                "rp_inter_period_max_state",
                over      = over,
                sense     = "<=",
                lhs_terms = lhs,
                rhs_terms = {"upper": d.p_state_upper},
            )
    if _has_rows(b.rp_start_idx):
        lhs = {"state_rp_start": b.v_state_rp_start}
        lhs.update(_capacity.node_capacity_change_terms(b, nodes, "_rp"))
        m.add_cstr(
            "maxState_rp_start",
            over      = b.rp_start_idx,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": d.p_state_upper},
        )


def _rp_typed(b: _Build, frame: pl.DataFrame) -> pl.DataFrame:
    """Base-period labels (b, b_prev) onto v_state_inter's b dtype, periods (d, d_prev) onto its d dtype."""
    schema = b.v_state_inter.frame.schema
    return frame.with_columns(
        [pl.col(c).cast(schema["b"], strict=False) for c in ("b", "b_prev") if c in frame.columns]
        + [pl.col(c).cast(pl.Utf8).cast(schema["d"], strict=False) for c in ("d", "d_prev") if c in frame.columns])


def _rp_chain_paths(b: _Build) -> "tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None":
    """``(roots, leaves, paths)`` of the base-period chain of a representative-period solve, typed onto
    v_state_inter.

    roots (d, b): the first base period of each rp period without a predecessor in rp_period_previous (the solve's
    first period; each forecast branch of a fan in the first period); leaves (d, b): the last base period of each rp
    period that is no period's predecessor (the solve's last period; each terminal branch); paths (d, b, d_prev,
    b_prev): each leaf (d_prev, b_prev) with the root (d, b) its periods descend from.  None without a chain.
    """
    d = b.d
    if not (_has_rows(d.rp_base_first) and _has_rows(d.rp_base_last)):
        return None
    first = _rp_typed(b, d.rp_base_first).select("d", "b")
    last = _rp_typed(b, d.rp_base_last).select("d", "b")
    if _has_rows(d.period_in_use_set):
        in_use = dispatch_periods(d).select(pl.col("d").cast(pl.Utf8))
        first = first.filter(pl.col("d").cast(pl.Utf8).is_in(in_use["d"].implode()))
        last = last.filter(pl.col("d").cast(pl.Utf8).is_in(in_use["d"].implode()))
    prev_of: dict[str, str] = {}
    if _has_rows(d.rp_period_previous):
        prev_of = {str(a): str(p) for a, p in d.rp_period_previous.select(
            pl.col("d").cast(pl.Utf8), pl.col("d_prev").cast(pl.Utf8)).iter_rows()}
    periods = [str(x) for x in first["d"].cast(pl.Utf8).to_list()]
    is_prev = set(prev_of.values())
    root_names = [p for p in periods if p not in prev_of]
    leaf_names = [str(x) for x in last["d"].cast(pl.Utf8).to_list() if str(x) not in is_prev]
    if not root_names or not leaf_names:
        return None

    def root_of(period: str) -> str:
        seen = set()
        while period in prev_of and period not in seen:
            seen.add(period)
            period = prev_of[period]
        return period

    roots = first.filter(pl.col("d").cast(pl.Utf8).is_in(root_names))
    leaves = last.filter(pl.col("d").cast(pl.Utf8).is_in(leaf_names))
    pairs = pl.DataFrame({"_root": [root_of(p) for p in leaf_names], "_leaf": leaf_names},
                         schema={"_root": pl.Utf8, "_leaf": pl.Utf8})
    paths = (pairs
             .join(roots.with_columns(_root=pl.col("d").cast(pl.Utf8)), on="_root", how="inner")
             .join(leaves.rename({"d": "d_prev", "b": "b_prev"}).with_columns(_leaf=pl.col("d_prev").cast(pl.Utf8)),
                   on="_leaf", how="inner")
             .select("d", "b", "d_prev", "b_prev"))
    return roots, leaves, paths


def _rp_node_set(b: _Build, frame) -> "pl.DataFrame | None":
    """``frame`` (n) ∩ nodeState_rp, typed onto v_state_inter's n; None when empty."""
    d = b.d
    if not (_has_rows(frame) and _has_rows(d.nodeState_rp)):
        return None
    n_dtype = b.v_state_inter.frame.schema["n"]
    out = (frame.select(pl.col("n").cast(pl.Utf8)).unique()
           .join(d.nodeState_rp.select(pl.col("n").cast(pl.Utf8)).unique(), on="n", how="inner")
           .select(pl.col("n").cast(n_dtype, strict=False)).sort("n"))
    return out if out.height > 0 else None


def _rp_minus(frame: "pl.DataFrame | None", other) -> "pl.DataFrame | None":
    """``frame`` (n) without the nodes of ``other`` (any frame with n); None when empty."""
    if frame is None:
        return None
    if _has_rows(other):
        frame = frame.filter(~pl.col("n").cast(pl.Utf8).is_in(other.select(pl.col("n").cast(pl.Utf8))["n"].implode()))
    return frame if frame.height > 0 else None


def _nested_nodes(b: _Build, *, with_price: bool) -> "pl.DataFrame | None":
    """``(n,)`` Utf8: the nodes an upper-level solve fixes at some step of this solve (_nested_fixed_end)."""
    steps = _nested_fixed_end(b, with_price=with_price)
    return None if steps is None else steps.select("n").unique()


def _rp_start_pin_nodes(b: _Build) -> "pl.DataFrame | None":
    """``(n,)``: the representative-period nodes whose chain start fix_start pins (storage_fix_start ∩
    nodeState_rp, not carried); None when none or the data to pin is missing."""
    d = b.d
    if not (b.has_storage and b.has_rp and d.p_state_start is not None and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None):
        return None
    nodes = _rp_node_set(b, d.storage_fix_start)
    return _rp_minus(nodes, _self_discharge.carried_nodes(d))


def _rp_end_pin_nodes(b: _Build) -> "pl.DataFrame | None":
    """``(n,)``: the representative-period nodes whose chain end fix_end pins (storage_fix_end ∩ nodeState_rp, in a
    solve whose chain ends at the end of the model: it holds the model's last period, storage_rp_chain_at_model_end,
    or its last step; no upper-level fix_quantity / fix_usage on the node)."""
    d = b.d
    at_end = d.storage_solve_at_model_end or getattr(d, "storage_rp_chain_at_model_end", False)
    if not (b.has_storage and b.has_rp and at_end and d.p_state_end is not None
            and b.cap.existing_state_n is not None and d.p_state_unitsize is not None):
        return None
    return _rp_minus(_rp_node_set(b, d.storage_fix_end), _nested_nodes(b, with_price=False))


def _start_decision_periods(b: _Build) -> dict[str, set[str]]:
    """``{node: {periods}}``: the periods in which this solve decides storage capacity of the node: an investment
    (nd_invest_set) or a life extension (``_extend.options``: the extended cohort starts to operate there).  A start
    in such a period is on capacity the solve decides, so it is not an endowment (fix_start_energy and
    node_in_service count the extended capacity too)."""
    d = b.d
    out: dict[str, set[str]] = {}
    frames = [d.nd_invest_set] if b.has_invest_n and _has_rows(d.nd_invest_set) else []
    extend = _extend.options(d, "n")
    if extend is not None:
        frames.append(extend)
    for frame in frames:
        for n, p in frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).unique().iter_rows():
            out.setdefault(n, set()).add(p)
    return out


def _invest_at_start_nodes(b: _Build, nodes: "pl.DataFrame | None", periods: list[str]) -> set[str]:
    """The ``nodes`` (n) whose capacity this solve decides (invests in or extends, :func:`_start_decision_periods`)
    during one of ``periods`` (the periods the chain or the solve starts in): their capacity at the start, so also
    a fix_start start, is a decision of the solve."""
    if nodes is None:
        return set()
    names = set(str(n) for n in nodes["n"].to_list())
    wanted = set(periods)
    return {n for n, ps in _start_decision_periods(b).items() if n in names and ps & wanted}


def _eq_rp_chain_ends(b: _Build) -> None:
    """Representative-period storage: the absolute level at the start and the end of the base-period chain
    (fix_start, fix_end, use_reference_value, the start's capacity bounds and the matching end)."""
    if not (b.has_storage and b.has_rp and b.v_state_inter is not None):
        return
    m, d = b.m, b.d
    has_rep_blocks = d.rp_base__rep is not None and _has_rows(d.rp_block_first) and _has_rows(d.p_rp_last_step)
    shape = _rp_chain_paths(b)
    if not has_rep_blocks or shape is None:
        return
    roots, leaves, paths = shape
    drift = _rp_weighted_drift(b, d.nodeState_rp)
    if drift is None:
        return
    v_inter = b.v_state_inter
    inter_prev = _var_view(v_inter, "__chain_end", {"d": "d_prev", "b": "b_prev"})
    forward = _rp_node_set(b, d.storage_bind_forward_only_blended_weights)
    start_pin = _rp_start_pin_nodes(b)
    end_pin = _rp_end_pin_nodes(b)
    su = d.p_state_unitsize

    # ─── EQ. ─── rp_inter_period_start_fix ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − drift[n,d,b] − state_start[n]·invested[n,d] + state_start[n]·retired[n,d]
    #   = state_start[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_fix_start ∩ nodeState_rp, n ∉ carried, (d,b) ∈ chain roots (the first base period of the
    #     solve's first period; of each forecast branch of a fan in the first period)
    # The level at the start of the chain (the level the first base period starts from: its end level less the
    # weighted net change of its blocks) is the share state_start of the storage capacity in service, in state
    # units.  The block-internal states start from free offsets (v_state_rp_start), so the start is pinned on the
    # chain, the absolute level, not on a state inside the first block.  With bind_within_solve_blended_weights /
    # bind_within_period_blended_weights the closure (rp_inter_period_cyclic) then holds the cycle at that level, as
    # storage_state_start_binding_cyclic_period does without representative periods.  Only where no state of the
    # node was realised before this solve (the first solve of the model and of each nested level).
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if start_pin is not None:
        over = start_pin.join(roots, how="cross").select("n", "d", "b")
        lhs: dict = {"state_inter": Where(v_inter, over), "neg_rp_drift": -Where(drift, over)}
        invested, divested = _capacity.node_in_service(b, start_pin, "_rp_start", coef=d.p_state_start)
        if invested is not None:
            lhs["invest_neg"] = -invested
        if divested is not None:
            lhs["divest"] = divested
        _notice_nodes("rp_storage_chain_pinned", start_pin, b)
        m.add_cstr(
            "rp_inter_period_start_fix",
            over      = over,
            sense     = "==",
            lhs_terms = lhs,
            rhs_terms = {"target": (d.p_state_start * b.cap.existing_state_n) / su},
        )

    # ─── EQ. ─── rp_inter_period_end_fix ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − state_end[n]·invested[n,d] + state_end[n]·retired[n,d]
    #   = state_end[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_fix_end ∩ nodeState_rp, (d,b) ∈ chain leaves (the last base period of the solve's last
    #     period; of each terminal branch), the chain ends at the end of the model (storage_rp_chain_at_model_end: the
    #     solve holds the model's last period, whose last base period ends with it), no upper-level fix_quantity /
    #     fix_usage on n
    # The level at the end of the chain (the end of the last base period) is the share state_end of the storage
    # capacity in service, in every solve whose horizon holds the last step of the model, as
    # storage_state_end_binding pins v_state without representative periods.
    if end_pin is not None:
        over = end_pin.join(leaves, how="cross").select("n", "d", "b")
        lhs = {"state_inter": Where(v_inter, over)}
        invested, divested = _capacity.node_in_service(b, end_pin, "_rp_end", coef=d.p_state_end)
        if invested is not None:
            lhs["invest_neg"] = -invested
        if divested is not None:
            lhs["divest"] = divested
        _notice_nodes("rp_storage_chain_pinned", end_pin, b)
        m.add_cstr(
            "rp_inter_period_end_fix",
            over      = over,
            sense     = "==",
            lhs_terms = lhs,
            rhs_terms = {"target": (d.p_state_end * b.cap.existing_state_n) / su},
        )

    if forward is None:
        return
    # ─── EQ. ─── rp_inter_period_start_carry ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # (v_state_inter[n,d,b] − drift[n,d,b]) · state_unitsize[n] = roll_continue_state[n]
    #   ∀ n ∈ bind_forward_only_blended_weights ∩ nodeState_rp ∩ carried, (d,b) ∈ chain roots
    # A representative-period solve that follows another solve (a sequence, or a roll) starts the base-period chain
    # of a forward node from the level realised before it (the carry, MWh, routed by the cascade), as a
    # bind_forward_only node without representative periods does at its first step (owner decision).  Before, the
    # chain started free.  The block-internal states keep their free offsets: only the chain is absolute.
    carried = _rp_node_set(b, _self_discharge.carried_nodes(d))
    carry_start = None if carried is None else _rp_minus(
        forward.join(carried, on="n", how="semi"), start_pin)
    if carry_start is not None and d.p_roll_continue_state is not None:
        over = carry_start.join(roots, how="cross").select("n", "d", "b")
        level = (d.p_roll_continue_state.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
                 .join(over.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                   pl.col("b").cast(pl.Utf8)), on="n", how="inner")
                 .select("n", "d", "b", "value"))
        _notice_nodes("rp_storage_chain_pinned", carry_start, b)
        m.add_cstr(
            "rp_inter_period_start_carry",
            over      = over,
            sense     = "==",
            lhs_terms = {"state_inter": Where(v_inter, over) * su, "neg_rp_drift": -Where(drift, over) * su},
            rhs_terms = {"carry": Param(("n", "d", "b"), _rp_typed(b, level))},
        )
    # The forward chains with a free start: neither the start pin nor the carry above applies.
    free_start = _rp_minus(_rp_minus(forward, start_pin), carry_start)

    # ─── EQ. ─── rp_inter_period_start_max_state / rp_inter_period_start_min_state ─
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − drift[n,d,b] + retired[n,d] − invested[n,d]  ≤  state_upper[n,d]
    # v_state_inter[n,d,b] − drift[n,d,b]  ≥  0
    #   ∀ n ∈ bind_forward_only_blended_weights ∩ nodeState_rp, n ∉ the nodes of rp_inter_period_start_fix,
    #     (d,b) ∈ chain roots
    # The level a forward chain starts from is a physical level: within the storage capacity in service, as every
    # base-period level (rp_inter_period_max_state).  Without the bounds the start of a forward chain had no bound.
    if free_start is not None:
        over = free_start.join(roots, how="cross").select("n", "d", "b")
        lhs = {"state_inter": Where(v_inter, over), "neg_rp_drift": -Where(drift, over)}
        upper = dict(lhs)
        upper.update(_capacity.node_capacity_change_terms(b, free_start, "_rp_start_max"))
        _notice_nodes("rp_storage_chain_pinned", free_start, b)
        m.add_cstr(
            "rp_inter_period_start_max_state",
            over      = over,
            sense     = "<=",
            lhs_terms = upper,
            rhs_terms = {"upper": d.p_state_upper},
        )
        m.add_cstr(
            "rp_inter_period_start_min_state",
            over      = over,
            sense     = ">=",
            lhs_terms = lhs,
            rhs_terms = {},
        )

    # The forward chains whose start is a decision of this solve (storage_semantics: the matching end applies):
    # a free start, or a fix_start share of capacity the solve can invest in during the start period; not where an
    # upper-level solve fixes the node, nor where fix_start_end pins the end at least at the start share.
    root_periods = [str(x) for x in roots["d"].cast(pl.Utf8).to_list()]
    investable = _invest_at_start_nodes(b, _rp_minus(forward, free_start), root_periods)
    closed_by_end = set()
    if end_pin is not None and d.p_state_start is not None and d.p_state_end is not None:
        shares = (d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("s"))
                  .join(d.p_state_end.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("e")), on="n"))
        pinned = set(str(n) for n in end_pin["n"].to_list())
        closed_by_end = {n for n, s, e in shares.iter_rows() if n in pinned and e >= s - 1e-12}
    match_names = (set() if free_start is None else set(str(n) for n in free_start["n"].to_list())) | \
        (investable - closed_by_end)
    nested = _nested_nodes(b, with_price=True)
    if nested is not None:
        match_names -= set(nested["n"].to_list())
    ref_nodes = _rp_minus(_rp_node_set(b, d.storage_use_reference_value), nested)
    ref_nodes = _rp_minus(ref_nodes, end_pin)
    if ref_nodes is not None:
        ref_nodes = ref_nodes.filter(pl.col("n").cast(pl.Utf8).is_in(forward["n"].cast(pl.Utf8).implode()))
        ref_nodes = ref_nodes if ref_nodes.height > 0 else None
    match = forward.filter(pl.col("n").cast(pl.Utf8).is_in(sorted(match_names)))
    endowment = None
    if match.height > 0:
        # Endowment (state units): the start a fix_start node has on the capacity that exists before the solve
        # decides; 0 for a free start.  Only the start on the capacity the solve adds must come back.
        endowment = (match.select(pl.col("n").cast(pl.Utf8)).join(
            pl.DataFrame({"d": root_periods}, schema={"d": pl.Utf8}), how="cross"))
        if investable and d.p_state_start is not None and b.cap.existing_state_n is not None:
            exist = (d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("s"))
                     .filter(pl.col("n").is_in(sorted(investable)))
                     .join(b.cap.existing_state_n.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                              pl.col("value").alias("cap")), on="n")
                     .join(su.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("us")), on="n")
                     .select("n", "d", value=-pl.col("s") * pl.col("cap") / pl.col("us")))
            endowment = endowment.join(exist, on=["n", "d"], how="left")
        else:
            endowment = endowment.with_columns(value=pl.lit(None, dtype=pl.Float64))
        endowment = endowment.with_columns(pl.col("value").fill_null(0.0))

    # ─── EQ. ─── rp_inter_period_end_match ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d′,b′] − (v_state_inter[n,d,b] − drift[n,d,b])  ≥  − state_start[n]·existing[n,d] / su[n]
    #   ∀ n ∈ bind_forward_only_blended_weights ∩ nodeState_rp whose start is a decision of the solve: a free start
    #     (no fix_start, or carried), or fix_start while the solve can invest in n in the start period d (not with
    #     fix_start_end at state_end ≥ state_start); n without an upper-level fix;
    #     (d,b,d′,b′) ∈ chain paths (root (d,b), the leaf (d′,b′) of each path); the right-hand side is 0 for a
    #     free start
    # The matching end: a forward chain whose start the solve chooses ends at least at that start, so the solve
    # cannot take energy it never produced (a free start full, the end empty; or storage built as a fuel tank whose
    # fix_start share is spent once).  With fix_start only the start on the capacity the solve adds must come back:
    # the start on existing capacity is the user's explicit endowment (state units).  An inequality: a start of 0,
    # or no investment, always meets it, so it never forces slack.  fix_end and use_reference_value still set the
    # end; with a free start this bounds the start by that end.
    if match.height > 0:
        over = match.join(paths, how="cross").select("n", "d", "b", "d_prev", "b_prev")
        rhs = Param(("n", "d"), _rp_typed(b, endowment.with_columns(
            pl.col("n").cast(v_inter.frame.schema["n"], strict=False))).select("n", "d", "value"))
        _notice_nodes("storage_end_matches_start", match, b)
        m.add_cstr(
            "rp_inter_period_end_match",
            over      = over,
            sense     = ">=",
            lhs_terms = {"state_end": Where(inter_prev, over), "neg_state_start": -Where(v_inter, over),
                         "rp_drift_start": Where(drift, over)},
            rhs_terms = {"endowment": rhs},
        )

    # ─── EQ. ─── rp_inter_period_start_match ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] · state_unitsize[n]  ≥  min(start[n], capacity[n,d])
    #   ∀ n ∈ storage_roll_end(match) ∩ bind_forward_only_blended_weights ∩ nodeState_rp, (d,b) ∈ chain leaves,
    #     not pinned by fix_end, no upper-level fix
    #   start = the level (MWh) the node started the model from (a free start, recorded by the cascade)
    # The matching end of a carried chain: a later representative-period solve that reaches the end of the model
    # ends the chain at least at the level the model started the node from, so the solves together return it.
    starts = _roll_end_rows(d, ("match",))
    if starts is not None and b.cap.existing_state_n is not None:
        names = set(starts["n"].cast(pl.Utf8).to_list()) & set(forward["n"].cast(pl.Utf8).to_list())
        if end_pin is not None:
            names -= set(end_pin["n"].cast(pl.Utf8).to_list())
        if nested is not None:
            names -= set(nested["n"].to_list())
        nodes = forward.filter(pl.col("n").cast(pl.Utf8).is_in(sorted(names)))
        if nodes.height > 0:
            over = nodes.join(leaves, how="cross").select("n", "d", "b")
            level = (over.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
                     .join(starts.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("_s")), on="n")
                     .join(b.cap.existing_state_n.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                              pl.col("value").alias("_c")), on=["n", "d"], how="left")
                     .select("n", "d", "b", value=pl.min_horizontal("_s", pl.col("_c").fill_null(pl.col("_s")))))
            _notice_nodes("storage_end_matches_start", nodes, b)
            m.add_cstr(
                "rp_inter_period_start_match",
                over      = over,
                sense     = ">=",
                lhs_terms = {"state_inter": Where(v_inter, over) * su},
                rhs_terms = {"start": Param(("n", "d", "b"), _rp_typed(b, level))},
            )

    # ─── EQ. ─── rp_inter_period_reference_value ─────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] − reference_value[n,d,t]·invested[n,d] + reference_value[n,d,t]·retired[n,d]
    #   = reference_value[n,d,t] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_use_reference_value ∩ bind_forward_only_blended_weights ∩ nodeState_rp, (d,b) ∈ chain leaves,
    #     t = the node's last step in d (nodeState_last_dt), no upper-level fix on n
    # use_reference_value sets the level at the end of the chain (the absolute level at the end of the horizon), not
    # a state inside the last block.  A reference value below the start share of a node the solve can invest in:
    # the matching end above stays (the solve cannot build storage to spend its start share) and caps the
    # investment; a warning (_warn_reference_below_start).
    if ref_nodes is not None and d.p_storage_state_reference_value is not None and _has_rows(d.nodeState_last_dt):
        last_step = (d.nodeState_last_dt.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
                     .join(leaves.select(pl.col("d").cast(pl.Utf8)), on="d", how="inner")
                     .filter(pl.col("n").is_in(ref_nodes["n"].cast(pl.Utf8).implode())))
        rv = (promote_param_to_dt(d.p_storage_state_reference_value, d.dt)
              .select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8), "value")
              .join(last_step.lazy(), on=["n", "d", "t"], how="inner")
              .select("n", "d", "value").unique(subset=["n", "d"], keep="first").collect())
        if investable and d.p_state_start is not None:
            share = {str(n): v for n, v in d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), "value").iter_rows()}
            _warn_reference_below_start(b, {n for n, _d, v in rv.iter_rows()
                                            if n in investable - closed_by_end and v < share.get(n, 0.0) - 1e-12})
        target = (rv.rename({"value": "rv"})
                  .join(b.cap.existing_state_n.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                                            pl.col("value").alias("exist")), on=["n", "d"], how="left")
                  .join(su.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("us")), on="n", how="inner")
                  .with_columns(value=pl.col("rv") * pl.col("exist").fill_null(0.0) / pl.col("us")))
        part = target
        nodes = ref_nodes.filter(pl.col("n").cast(pl.Utf8).is_in(part["n"].implode()))
        over = _rp_typed(b, nodes.join(leaves, how="cross").select("n", "d", "b"))
        over = over.join(_rp_typed(b, part.select("n", "d").with_columns(
            pl.col("n").cast(v_inter.frame.schema["n"], strict=False))), on=["n", "d"], how="semi")
        if over.height > 0:
            coef = Param(("n", "d"), _rp_typed(b, part.select("n", "d", pl.col("rv").alias("value")).with_columns(
                pl.col("n").cast(v_inter.frame.schema["n"], strict=False))))
            lhs = {"state_inter": Where(v_inter, over)}
            invested, divested = _capacity.node_in_service(b, nodes, "_rp_ref", coef=coef)
            if invested is not None:
                lhs["invest_neg"] = -invested
            if divested is not None:
                lhs["divest"] = divested
            _notice_nodes("rp_storage_chain_pinned", nodes, b)
            m.add_cstr(
                "rp_inter_period_reference_value",
                over      = over,
                sense     = "==",
                lhs_terms = lhs,
                rhs_terms = {"target": Param(("n", "d"), _rp_typed(b, part.select("n", "d", "value").with_columns(
                    pl.col("n").cast(v_inter.frame.schema["n"], strict=False))))},
            )


def _rp_weighted_drift(b: _Build, nodes: pl.DataFrame, *, at_prev: bool = False):
    """drift[n,d,b] = Σ_r rp_weight[d,b,r] · (v_state[n,d,last_step[d,r]] − v_state_rp_start[n,d,r]).

    Block r of period d starts at (d, t = r) ∈ rp_block_first and ends at last_step[d,r].  Dims (n, d, b);
    with ``at_prev`` dims (n, d_prev, b_prev), the drift of the other side of a period-to-period row.
    None when no representative block is in use.
    """
    d = b.d
    t_dtype = d.rp_block_first.schema["t"]
    d_dtype = d.rp_block_first.schema["d"]
    b_dtype = b.v_state_inter.frame.schema["b"]
    state_t_dtype = b.v_state.frame.schema["t"]
    block_first = (d.rp_block_first.rename({"t": "r"})
                   .with_columns(pl.col("r").cast(t_dtype, strict=False)))
    rep = d.rp_base__rep.frame.with_columns(
        pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False),
        pl.col("b").cast(b_dtype, strict=False),
        pl.col("r").cast(t_dtype, strict=False))
    last_step = d.p_rp_last_step.with_columns(
        pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False),
        pl.col("r").cast(t_dtype, strict=False),
        pl.col("last_step").cast(state_t_dtype, strict=False))
    cohort = (rep.join(block_first, on=["d", "r"], how="inner")
              .join(last_step, on=["d", "r"], how="inner"))
    if d.period_in_use_set is not None:
        cohort = cohort.join(dispatch_periods(d).with_columns(pl.col("d").cast(pl.Utf8).cast(d_dtype, strict=False)),
                             on="d", how="inner")
    if cohort.height == 0:
        return None
    keys = ("d", "b", "r", "last_step")
    if at_prev:
        cohort = cohort.rename({"d": "d_prev", "b": "b_prev"})
        keys = ("d_prev", "b_prev", "r", "last_step")
        state_at_last = _var_view(b.v_state, "__rp_last_prev", {"d": "d_prev", "t": "last_step"})
        start_at_r = _var_view(b.v_state_rp_start, "__at_r_prev", {"d": "d_prev", "t": "r"})
    else:
        state_at_last = _var_view(b.v_state, "__rp_last", {"t": "last_step"})
        start_at_r = _var_view(b.v_state_rp_start, "__at_r", {"t": "r"})
    rp_weight = Param(keys, cohort.select(*keys, "value"))
    idx = nodes.select("n").unique().join(cohort.select(*keys), how="cross")
    return Sum((Where(state_at_last, idx) - Where(start_at_r, idx)) * rp_weight,
               over=("r", "last_step"))


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
    #   − Σ_{(p,n) ∈ startup_to_balance} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]   (no Δt: an event)
    #   + back-flow terms of two-way one-variable arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d,t⁻] − v_state[n,d,t])·state_unitsize[n]
    #   − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·state_unitsize[n]
    #   h = Δt·ln(1 + self_discharge_loss[n,d,t]),  α = h/(1 − e^{−h}) − 1,  β = 1 − h/(e^{h} − 1)
    #   S⁻ = the state the step starts from: v_state[n,d′,t⁻] by the binding method, v_state_rp_start at a
    #        representative block's first step, the fix_start start state / the roll carry at an open chain's
    #        first step, v_state[n,d,t] where the chain is open with neither
    #   + start terms of open-start nodes (fix_start in the first solve, the roll carry in continuation solves)
    #   = −inflow[n,d,t]
    #   ∀ (n,d,t) ∈ nodeBalance_dt, n ∉ nodeStateBlock ∪ nodeCoarse;   Δt = step_duration[d,t]
    #   section[p,d,t] = min_load·(1/eta_min − 1/eta) / (1 − min_load)
    #   slope[p,d,t] = 1/eta − section[p,d,t]   (= (1/eta − min_load/eta_min) / (1 − min_load))
    #     for min_load_efficiency units (eta = efficiency, default 1; min_load = 0 where not authored;
    #     eta_min = efficiency_at_min_load, else efficiency); slope = 1/eta and no section for the other units
    #   on a direct (one-input, one-output) unit, slope and section above are further scaled by
    #     k_sink/k_src (slope) and 1/k_src (section) — the unit's output/input conversion_flow_coeff
    #     (default 1 each; a zero coefficient keeps its uncapped-edge meaning instead, unscaled)
    # Energy conservation per node and time step, in MWh.  Storage nodes add
    # the state change; the previous step t⁻ depends on storage_binding_method.
    # nodeStateBlock nodes are balanced per timeset block (nodeBalanceBlock_eq),
    # nodeCoarse nodes per coarse step (nodeBalanceCoarse_eq).
    # The section term exists for min_load_efficiency units only.  The start-up
    # term (core._startup_draw) is the start-up energy of the units with a
    # start-up input node n (startup_input_method startup_and_input /
    # startup_only), MWh per start-up event; a coarse process starts at the
    # first fine step of its coarse step.  The flow, start-up and slack terms
    # are b.node_balance_flow_terms.  Self-discharge: the
    # content decays continuously at ln(1 + loss) per hour; flows within a
    # step are spread evenly over the step.  Exact for any step length while
    # the loss is constant within the step; a coarse step / block with a
    # time-varying loss uses h_b = Σ Δt·λ_t, exact for the idle decay only
    # (#152).  The decay is applied to a weighted average of the start and
    # the end state (model/_self_discharge.py); the model folds it into the
    # state change as
    # (S⁻·(1 − β) − S·(1 + α))·state_unitsize.
    over = b.nodeBalance_dt
    if b.has_nsb:
        over = over.join(d.nodeStateBlock, on="n", how="anti")
    if b.has_coarse:
        over = over.join(d.nodeCoarse, on="n", how="anti")
    m.add_cstr(
        "nodeBalance_eq",
        over      = over,
        sense     = "==",
        lhs_terms = _node_balance_terms(b),
        rhs_terms = {"neg_inflow": -d.p_inflow},
    )


def _node_balance_terms(b: _Build) -> dict:
    """Every LHS term of nodeBalance_eq, (n, d, t) in MWh: flows, slacks and, for storage nodes, the state
    change with its self-discharge, and the start / roll-continuation terms.  nodeBalanceCoarse_eq and
    nodeBalanceBlock_eq sum them over each coarse step / block."""
    terms = dict(b.node_balance_flow_terms)
    if b.has_storage:
        terms.update(_state_change_terms(b))
        terms.update(_fix_start_terms(b))
        terms.update(_roll_continue_terms(b))
        terms.update(_open_start_loss_terms(b))
    return terms


def _sum_over_coarse_step(term, step_of: pl.DataFrame):
    """``term`` (n, d, t) summed over the fine steps t of each coarse step: (n, d, b_first).  A constant
    (``Param``) term is summed in its frame (sorted before the group-by, so the sum order is fixed)."""
    if isinstance(term, Param):
        frame = (term.frame
                 .join(step_of, on=["n", "d", "t"], how="inner")
                 .sort("n", "d", "b_first", "t")
                 .group_by(["n", "d", "b_first"], maintain_order=True)
                 .agg(pl.col("value").sum())
                 .select("n", "d", "b_first", "value"))
        return Param(("n", "d", "b_first"), frame)
    return Sum(Where(term, step_of), over=("t",))


def _eq_node_balance_coarse(b: _Build) -> None:
    """Energy balance per coarse step of the nodes of coarse new_stepduration groups."""
    if not b.has_coarse:
        return
    m, d = b.m, b.d
    step_of = d.node_coarse_step

    # ─── EQ. ─── nodeBalanceCoarse_eq ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{t ∈ b} [ Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #       (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n) ∈ startup_to_balance} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]
    #   + back-flow terms of two-way one-variable and DC arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d,t⁻] − v_state[n,d,t])·state_unitsize[n] − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·su[n]
    #   + start terms of open-start nodes ]
    #   = − Σ_{t ∈ b} inflow[n,d,t]
    #   ∀ n ∈ nodeCoarse, (n,d,b) ∈ node_coarse_step;   t ∈ b: (n,d,b,t) ∈ node_coarse_step;   Δt = step_duration[d,t]
    #   t⁻, start and roll terms per storage_binding_method exactly as in nodeBalance_eq
    #   a coarse process (processCoarse) has one v_flow per coarse step of its nodes: its fine terms read the same
    #   column, so its leg is v_flow[p,·,·,d,b]·unitsize·Σ_{t∈b} Δt (·slope_t on an efficiency source)
    #   α, β, S⁻ as in nodeBalance_eq with h_b = Σ_{t∈b} Δt·ln(1 + loss[n,d,t]), Δ_b = Σ_{t∈b} Δt: α spread over
    #   the fine steps by Δt/Δ_b, β at the first fine step only (the decay of the coarse step's own state over the
    #   coarse step; the flow weighting assumes the loss constant within the coarse step)
    # nodeBalance_eq summed over the fine steps of each coarse step of the node's group: every flow stays on the
    # solve's steps and each coarse leg sums its fine flows, so an arc between a fine and a coarse node, or
    # between two coarse nodes, conserves energy at both ends.  The node may shift energy freely within a coarse
    # step.  With the state constant within the coarse step (stateConstantCoarse_eq) the state terms telescope to
    # v_state[n,d,b⁻] − v_state[n,d,b] (b⁻ = the coarse step before b by the node's binding method), so storage
    # binding methods, fix_start and the roll carry apply to coarse nodes as to fine ones.  Same orientation and
    # units as nodeBalance_eq: its dual is the node's price (currency/MWh) over the coarse step.
    lhs = {key: _sum_over_coarse_step(term, step_of) for key, term in _node_balance_terms(b).items()}
    inflow = _sum_over_coarse_step(-d.p_inflow, step_of)
    rhs = {"neg_inflow": inflow} if inflow.frame.height > 0 else {}
    m.add_cstr(
        "nodeBalanceCoarse_eq",
        over      = step_of.select("n", "d", "b_first").unique(maintain_order=True),
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )

    # ─── EQ. ─── stateConstantCoarse_eq ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − v_state[n,d,t_previous] = 0
    #   ∀ (n,d,t,t_previous) ∈ node_coarse_interior   (n ∈ nodeCoarse ∩ nodeState; t_previous: the step before t
    #   within its coarse step)
    # A coarse storage node keeps one state per coarse step.
    if b.has_storage and _has_rows(d.node_coarse_interior):
        interior = d.node_coarse_interior
        state_prev = _var_view(b.v_state, "__prev", {"t": "t_previous"})
        steps = interior.select("n", "d", "t")
        m.add_cstr(
            "stateConstantCoarse_eq",
            over      = steps,
            sense     = "==",
            lhs_terms = {"state":      Where(b.v_state, steps),
                         "state_prev": -Sum(Where(state_prev, interior), over=("t_previous",))},
            rhs_terms = {},
        )


def _node_balance_flow_terms(b: _Build) -> dict:
    """Flow and slack terms of nodeBalance_eq, (n, d, t) in MWh.

    The flow terms come from ``_balance_terms.balance_flow_terms``, which the non-anticipativity storage tie
    shares.  Also used by nodeBalancePeriod_eq: period nodes are never storage, so they take only these terms.
    """
    d = b.d
    terms: dict = {"slack_up": b.vq_state_up * d.p_step_duration,
                   "slack_down": -b.vq_state_down * d.p_step_duration}
    terms.update(_balance_terms.balance_flow_terms(b))
    return terms


def _state_change_terms(b: _Build) -> dict:
    """State-change terms of nodeBalance_eq, one per storage_binding_method, with the self-discharge folded in
    (_weighted_change).

    The balance reads (state[t⁻] − state[t]) + sink − source = −inflow, i.e. state[t] − state[t⁻] = sink − source +
    inflow.  The branches and their lags are _self_discharge.state_links (each node is in one binding set only,
    _validate_storage_binding_disjoint): bind_within_timeblock wraps within the timeblock, bind_forward_only crosses
    period boundaries and has no state change at the solve's first step, bind_within_solve wraps the solve's first
    step to its last (with a stochastic fan in the first period, each root forecast's first step to the last step of
    its own path: dtttdt_within_solve), bind_within_period wraps each period's first step to the period's own last step,
    bind_intraperiod_blocks (nodeStateBlock) follows dtttdt_block_lag (these terms reach nodeBalanceBlock_eq only,
    which sums them over each block), and the blended-weights nodes chain inside each representative block whose
    first step starts from the free state v_state_rp_start.
    """
    d, v_state = b.d, b.v_state
    terms: dict = {}
    for link in _self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb):
        if link.cross_period:
            prev = _state_lag_cross_period(v_state, link.lag)
        else:
            prev = Lag(v_state, link.lag, "t", link.col)
        now = v_state if link.all_steps else Where(v_state, link.lag.select("d", "t").unique())
        terms[link.key] = _weighted_change(b, Where(prev, link.nodes), Where(now, link.nodes))
    if b.has_rp:
        nodes = d.nodeState_rp
        first_dt = d.rp_block_first.select("d", "t").unique()
        now = Where(Where(v_state, first_dt), nodes)
        start = Where(b.v_state_rp_start, nodes)
        terms["state_change_rp_start"] = _weighted_change(b, start, now)
    return terms


def _weighted_change(b: _Build, prev, now):
    """(prev·(1 − β) − now·(1 + α))·state_unitsize: the state change of a step with its self-discharge
    (_self_discharge); (prev − now)·state_unitsize when no storage node has a loss."""
    su = b.d.p_state_unitsize
    loss = b.self_discharge
    if loss is None:
        return (prev - now) * su
    return (prev * loss.keep_prev - now * loss.keep_now) * su


def _state_lag_cross_period(v_state: Var, lag_frame: pl.DataFrame):
    """v_state at (d_previous, t_previous_within_solve) of each (d, t) in
    ``lag_frame``; unlike ``Lag``, the previous step may be in another period."""
    v_state_back = _var_view(v_state, "__back", {"d": "d_back", "t": "t_back"},
                             to_axis=True)
    lag = lag_frame.select("d", "t",
                           alias_to_axis("d_previous", "d_back"),
                           alias_to_axis("t_previous_within_solve", "t_back"))
    return Sum(Where(v_state_back, lag), over=("d_back", "t_back"))


def _cyclic_nodes(b: _Build) -> "pl.DataFrame | None":
    """Storage nodes whose chain closes on itself within the solve (_self_discharge.cyclic_nodes)."""
    return _self_discharge.cyclic_nodes(b.d)


def _open_start_nodes(b: _Build) -> "pl.DataFrame | None":
    """Storage nodes whose chain starts open at the solve's first step (_self_discharge.open_start_nodes)."""
    return _self_discharge.open_start_nodes(b.d)


def _keep_prev_values(b: _Build, frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` (n, d, t, value) × (1 − β) of its step: a start energy weighted as the step's previous state."""
    loss = b.self_discharge
    if loss is None:
        return frame
    return (frame.join(loss.keep_prev.frame.rename({"value": "_keep"}), on=["n", "d", "t"], how="left")
            .with_columns(value=pl.col("value") * pl.col("_keep").fill_null(1.0))
            .select("n", "d", "t", "value"))


def _fix_start_terms(b: _Build) -> dict:
    """nodeBalance_eq start terms of open-start nodes (_open_start_nodes) with fix_start."""
    # ─── fix_start terms (in nodeBalance_eq) ─────────────────────────────────
    #   − v_state[n,d,t]·(1 + α[n,d,t])·state_unitsize[n]
    #   + (1 − β[n,d,t])·state_start[n]·(existing[n,d] + invested[n,d]·su[n]
    #                                    − retired[n,d]·su[n])
    #   ∀ (n,d,t) ∈ nodeState_first_dt, n ∈ open-start nodes ∩ storage_fix_start, n ∉ carried
    # The first step of a node with no realised state from before the solve (carried: the nodes of
    # roll_continue_state; so the first solve of the model and of each nested level), where the state change is
    # omitted, starts from the start state (MWh); α, β: the self-discharge weights (_self_discharge, 0 without a
    # loss).  Other fix_start nodes are pinned by storage_state_start_binding_cyclic_period (cyclic) or
    # rp_inter_period_start_fix (representative-period nodes: the start of their base-period chain).
    d = b.d
    first = _self_discharge.fix_start_first(d)
    if first is None:
        return {}
    nodes = first.select("n").unique()
    start, su = d.p_state_start, d.p_state_unitsize
    loss = b.self_discharge
    state = Where(b.v_state, first)
    terms: dict = {"fwd_fix_state": -(state * su) if loss is None else -(state * loss.keep_now * su)}
    # A sparse Param term combines with the dense −inflow by outer join with
    # 0-fill, so rows outside ``first`` keep their balance.
    start_energy = ((start * b.cap.existing_state_n).frame
                    .join(first, on=["n", "d"], how="inner")
                    .select("n", "d", "t", "value"))
    if start_energy.height > 0:
        terms["fwd_fix_existing_pos"] = Param(("n", "d", "t"), _keep_prev_values(b, start_energy))
    if b.has_invest_n:
        invested = _var_view(b.v_invest_n, "__at_fwd_fix", {"d": "d_invest"},
                             to_axis=True)
        alive = (_entity_rows(b.cap.alive, "e", nodes, "n")
                 .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if alive.height > 0:
            term = Sum(Where(invested * start * su, alive), over=("d_invest",))
            terms["fwd_fix_invest_pos"] = term if loss is None else term * loss.keep_prev
    if b.has_divest_n:
        divested = _var_view(b.v_divest_n, "__at_fwd_fix", {"d": "d_divest"},
                             to_axis=True)
        active = (_entity_rows(b.cap.retired_frame("n"), "n", nodes, "n")
                  .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if active.height > 0:
            term = Sum(Where(divested * start * su, active), over=("d_divest",))
            terms["fwd_fix_divest_neg"] = -(term if loss is None else term * loss.keep_prev)
    built = _retire_built.retired(b, "n")
    if built is not None:
        rows = (_entity_rows(_retire_built.retired_frame(b, "n"), "n", nodes, "n")
                .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if rows.height > 0:
            term = Where(built * start * su, rows)
            terms["fwd_fix_divest_built_neg"] = -(term if loss is None else term * loss.keep_prev)
    ext = _extend.extended(b, "n")
    if ext is not None:
        # The extended storage capacity in service (_extend): a capacity term, as an investment's.
        rows = (_entity_rows(_extend.extended_frame(b, "n"), "n", nodes, "n")
                .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if rows.height > 0:
            term = Where(ext * start * su, rows)
            terms["fwd_fix_extended_pos"] = term if loss is None else term * loss.keep_prev
    return terms


def _roll_continue_terms(b: _Build) -> dict:
    """nodeBalance_eq start terms of a solve that continues from a realised state (the carry)."""
    # ─── roll-continuation terms (in nodeBalance_eq) ─────────────────────────
    #   − v_state[n,d,t]·(1 + α[n,d,t])·state_unitsize[n] + (1 − β[n,d,t])·roll_continue_state[n]
    #   ∀ (n,d,t) ∈ nodeState_first_dt, n ∈ carried, n ∉ cyclic nodes ∪ nodeState_rp
    # A node continues from the latest state realised before this solve at its own or a lower nesting level (the
    # previous roll, the previous solve of a sequence, the dispatch below a storage solve; roll_continue_state, MWh,
    # routed by the cascade: solve._carry_routing).  α, β: the self-discharge weights of an open-start node
    # (_self_discharge).  A cyclic node (_cyclic_nodes) cycles within the solve and takes no carry (how_to/solve_sequences.md
    # rolling notes); a bind_within_solve node of a rolling roll that continues from a carried state is bind_forward_only
    # in that roll (_run_model._anchor_cyclic_in_rolling) and takes the carry here, its horizon end in
    # storage_state_roll_anchor / _close; a representative-period node starts each block from v_state_rp_start (no carry).  Nodes of
    # neither kind outside the open-start set take the unweighted form.
    d = b.d
    first = _self_discharge.roll_first(d)
    if first is None:
        return {}
    su = d.p_state_unitsize
    loss = b.self_discharge
    handed_over = (d.p_roll_continue_state.frame
                   .join(first, on="n", how="inner")
                   .select("n", "d", "t", "value"))
    open_start = _open_start_nodes(b)
    if loss is None or open_start is None:
        terms: dict = {"roll_continue_state": -(Where(b.v_state, first) * su)}
        if handed_over.height > 0:
            terms["roll_continue_value"] = Param(("n", "d", "t"), handed_over)
        return terms
    weighted = first.join(open_start, on="n", how="semi")
    plain = first.join(open_start, on="n", how="anti")
    terms = {}
    if weighted.height > 0:
        terms["roll_continue_state"] = -(Where(b.v_state, weighted) * loss.keep_now * su)
    if plain.height > 0:
        terms["roll_continue_state_rp"] = -(Where(b.v_state, plain) * su)
    if handed_over.height > 0:
        value = pl.concat([_keep_prev_values(b, handed_over.join(weighted.select("n"), on="n", how="semi")),
                           handed_over.join(weighted.select("n"), on="n", how="anti")], how="vertical")
        terms["roll_continue_value"] = Param(("n", "d", "t"), value)
    return terms


def _open_start_loss_terms(b: _Build) -> dict:
    """nodeBalance_eq self-discharge of an open chain's first step that starts from neither fix_start nor the
    carry (_self_discharge.open_first): the step has no state change, its state is constant over the step, so the
    loss is h·v_state·su (h = α + β)."""
    loss = b.self_discharge
    if loss is None:
        return {}
    first = _self_discharge.open_first(b.d)
    if first is None:
        return {}
    return {"self_discharge_open": -(Where(b.v_state, first) * loss.h * b.d.p_state_unitsize)}


def _eq_node_balance_period(b: _Build) -> None:
    """Period energy balance of balance_within_period nodes."""
    d = b.d
    if not _has_rows(d.nodeBalancePeriod) or d.period_in_use_set is None:
        return

    # ─── EQ. ─── nodeBalancePeriod_eq ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_t w[d,t] · (flow, start-up and slack terms of nodeBalance_eq)[n,d,t]  =  −Σ_t w[d,t] · inflow[n,d,t]
    #   ∀ n ∈ nodeBalancePeriod, n ∉ nodeState, d ∈ periods of dt;   w = timestep_weight
    # Energy balance over a whole period instead of every step, in MWh.  The
    # terms are b.node_balance_flow_terms (Δt already applied); period nodes
    # are never storage, so there are no state terms.
    # Every term, slacks included, carries the timestep weight: a per-step
    # balance needs no weight, but a period sum without it would let
    # low-weight timeslices offset high-weight ones.  The other op_factor
    # parts (inflation_op, period_share) are per period and cancel in a row.
    # Inflow is energy per step (no Δt), as in nodeBalance_eq.
    over = (d.nodeBalancePeriod.select("n").lazy()
            .join(dispatch_periods(d).lazy(), how="cross"))
    if _has_rows(d.nodeState):
        over = over.join(d.nodeState.select("n").lazy(), on="n", how="anti")
    over = over.select("n", "d").collect()
    if over.height == 0:
        return
    lhs = {key: Sum(term * d.p_timestep_weight, over=("t",))
           for key, term in b.node_balance_flow_terms.items()}
    inflow_period = Param(("n", "d"), ((d.p_inflow * d.p_timestep_weight).frame.lazy()
                                       .join(d.nodeBalancePeriod.select("n").lazy(), on="n", how="inner")
                                       # Fixed summation order (the join leaves row order unspecified).
                                       .sort("n", "d", "t")
                                       .group_by("n", "d", maintain_order=True)
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


def _fix_quantity_target(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t, value)``: the storage state (MWh) an upper-level solve fixes at the last step of each last
    period (node_balance_fix_quantity_eq_lower); None when the solve has no such fix."""
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
        return None
    target = (d.n_fix_storage_quantity
              .join(d.p_fix_storage_quantity.frame.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"}),
                    on="n", how="inner")
              .join(d.period_branch, on="d_upper", how="inner")
              .join(d.dtt_timeline_matching, on=["d", "t_upper"], how="inner")
              .join(d.period_last, on="d", how="inner")
              .join(d.nodeState_last_dt, on=["n", "d", "t"], how="inner")
              # Fixed summation order: the joins above leave row order unspecified.
              .sort(["n", "d", "t", "d_upper", "t_upper"])
              .group_by(["n", "d", "t"], maintain_order=True)
              .agg(pl.col("value").sum())
              .select("n", "d", "t", "value"))
    target = _restated(d, target, "fix_quantity")
    return target if target.height > 0 else None


def _restated(d, target: pl.DataFrame, kind: str) -> pl.DataFrame:
    """``target`` (n, d, t, value) with the value of each node that ``storage_roll_end`` restates by ``kind``
    (_run_model._nested_end_rows: the upper level interpolated inside an upper step, the cumulative fix_usage cap)."""
    rows = _roll_end_rows(d, (kind,))
    if rows is None or target.height == 0:
        return target
    over = rows.select(pl.col("n").cast(pl.Utf8).alias("_n"), pl.col("value").alias("_v"))
    return (target.with_columns(_n=pl.col("n").cast(pl.Utf8))
            .join(over, on="_n", how="left")
            .with_columns(value=pl.coalesce("_v", "value"))
            .drop("_n", "_v"))


def _eq_node_balance_fix_quantity(b: _Build) -> None:
    """Final storage state fixed by an upper-level solve."""
    d = b.d

    # ─── EQ. ─── node_balance_fix_quantity_eq_lower ──────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] · state_unitsize[n]
    #   = Σ_{(d_up,d) ∈ period_branch, (d,t,t_up) ∈ dtt_timeline_matching} fix_storage_quantity[n,d_up,t_up]
    #     (storage_roll_end(fix_quantity)[n] instead, where it is given)
    #   ∀ n ∈ n_fix_storage_quantity, d ∈ period_last, (n,d,t) ∈ nodeState_last_dt
    # Pins the storage state (MWh) at the last step of each last period to the
    # state an upper-level (coarser) solve reached at the matching time.  Where
    # the horizon ends inside an upper step, the cascade restates the target as
    # the upper level interpolated at that instant (the upper solve's level is
    # linear within its step; _run_model._nested_end_rows), not the step's end.
    # Despite the name, the row is an equality.
    target = _fix_quantity_target(b)
    if target is None:
        return
    over = target.select("n", "d", "t").unique()
    b.m.add_cstr(
        "node_balance_fix_quantity_eq_lower",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _fix_usage_target(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t, value)``: the storage usage (MWh) an upper-level solve caps over each last period, at the period's
    last step (node_storage_usage_fix_le); None when the solve has no such cap."""
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
        return None
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
              .group_by(["n", "d", "t"], maintain_order=True)
              .agg(pl.col("value").sum())
              .select("n", "d", "t", "value"))
    target = _restated(d, target, "usage_cap")
    target = _with_period_caps(d, target)
    return target if target.height > 0 else None


def _with_period_caps(d, target: pl.DataFrame) -> pl.DataFrame:
    """``target`` (n, d, t, value) with the ``usage_cap_period`` rows of ``storage_roll_end``: the cumulative
    fix_usage cap of a period of a roll before its horizon's last period, at the period's last step in the roll
    (_run_model._nested_end_rows; a period already in ``target`` keeps its row)."""
    sre = getattr(d, "storage_roll_end", None)
    if not _has_rows(sre) or "period" not in sre.columns:
        return target
    extra = (sre.filter((pl.col("kind") == "usage_cap_period") & pl.col("period").is_not_null())
             .select(pl.col("n").cast(pl.Utf8), pl.col("period").cast(pl.Utf8).alias("d"), "value"))
    if extra.height == 0:
        return target
    fixed = d.n_fix_storage_usage.select(pl.col("n").cast(pl.Utf8)).unique()
    last = d.nodeState_last_dt.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                      pl.col("t").cast(pl.Utf8))
    extra = (extra.join(fixed, on="n", how="semi").join(last, on=["n", "d"], how="inner")
             .join(target.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)), on=["n", "d"], how="anti")
             .select("n", "d", "t", "value").sort("n", "d", "t"))
    if extra.height == 0:
        return target
    schema = target.schema
    extra = extra.with_columns([pl.col(c).cast(schema[c], strict=False) for c in ("n", "d", "t")]
                               + [pl.col("value").cast(schema["value"])])
    return pl.concat([target, extra], how="vertical")


def _fix_price_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the last step of each last period that an upper-level solve prices for a nested fix_price (the
    storage reference price term values the state there); None when the solve has no such step."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.ndt_fix_storage_price)
            and _has_rows(d.dtt_timeline_matching)
            and d.period_branch is not None
            and d.period_last is not None
            and d.nodeState_last_dt is not None):
        return None
    steps = (d.ndt_fix_storage_price.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"})
             .join(d.period_branch, on="d_upper", how="inner")
             .join(d.dtt_timeline_matching, on=["d", "t_upper"], how="inner")
             .join(d.period_last, on="d", how="inner")
             .join(d.nodeState_last_dt, on=["n", "d", "t"], how="inner")
             .select("n", "d", "t").unique())
    return steps if steps.height > 0 else None


def _nested_fixed_end(b: _Build, *, with_price: bool) -> "pl.DataFrame | None":
    """``(n, d, t)``, all Utf8: the steps at which an upper-level solve fixes a nested storage node, i.e. the end of a
    last period: node_balance_fix_quantity_eq_lower (fix_quantity), node_storage_usage_fix_le (fix_usage) and, with
    ``with_price``, the nested fix_price steps.  None when there are none."""
    parts = [_fix_quantity_target(b), _fix_usage_target(b)]
    if with_price:
        parts.append(_fix_price_steps(b))
    frames = [f.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
              for f in parts if f is not None]
    if not frames:
        return None
    return pl.concat(frames).unique().sort("n", "d", "t")


def _without_steps(over: pl.DataFrame, steps: "pl.DataFrame | None") -> pl.DataFrame:
    """``over`` (n, d, t) without the rows in ``steps`` (all-Utf8 (n, d, t)), keeping ``over``'s dtypes and order."""
    if steps is None:
        return over
    utf = steps.rename({"n": "_n", "d": "_d", "t": "_t"})
    return (over.with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(utf, on=["_n", "_d", "_t"], how="anti")
            .drop("_n", "_d", "_t"))


def _eq_storage_usage_fix(b: _Build) -> None:
    """Storage usage per period capped by an upper-level solve."""
    d = b.d
    target = _fix_usage_target(b)
    if target is None:
        return

    # ─── EQ. ─── node_storage_usage_fix_le ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,n) ∈ process_source_sink, t} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    # + Σ_{(p,n,k) ∈ process_source_sink_eff, t}
    #     (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    # + Σ_{(p,n,k) ∈ process_source_sink_noEff, t} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   ≤ Σ_{(d_up,d) ∈ period_branch, t_up : ∃t (d,t,t_up) ∈ dtt_timeline_matching} fix_storage_usage[n,d_up,t_up]
    #     (storage_roll_end(usage_cap)[n] instead, where it is given)
    #   ∀ n ∈ n_fix_storage_usage, d ∈ period_last, (n,d,t_last) ∈ nodeState_last_dt;   Δt = step_duration[d,t]
    #   and, in a roll, ∀ (n,d) ∈ storage_roll_end(usage_cap_period): d a period of the roll before its horizon's
    #     last one, RHS that cumulative cap
    # Caps the net energy drawn from storage node n over a whole period (MWh)
    # by the usage an upper-level (coarser) solve planned for the coarse
    # windows overlapping that period.  One row per period, at its last step;
    # each coarse window counts once however many fine steps it covers.  The
    # cascade restates the cap where that differs from the cumulative rule
    # (_run_model._nested_end_rows): the upper usage from the period's start in
    # the solve instance to the horizon end, each window pro rata to its hours
    # in that span, less the usage the instance's earlier rolls realised in the
    # period — so a rolling lower level uses at most what the upper planned.
    # The section term exists for min_load_efficiency units only.  The
    # process sink/source coefficient ratio is not applied (nor in the
    # source-side variable cost); it is 1 in every current dataset.
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
    """bind_intraperiod_blocks storage nodes (nodeStateBlock), balanced per timeset block."""
    if not (b.has_nsb and b.has_storage):
        return
    m, d = b.m, b.d
    nsb = d.nodeStateBlock
    v_state = b.v_state
    # nodeBalance_eq skips these nodes: without blocks they would have no energy balance at all.
    if not (_has_rows(d.period_block) and _has_rows(d.period_block_time) and _has_rows(d.dtttdt_block_lag)):
        names = ", ".join(sorted(str(n) for n in nsb["n"].to_list()))
        raise FlexToolConfigError(f"bind_intraperiod_blocks node(s) {names} have no timeset blocks in this solve")

    # ─── EQ. ─── stateConstantWithinBlock_eq ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − v_state[n,d,t_previous] = 0
    #   ∀ n ∈ nodeStateBlock, (d,t,t_previous) ∈ dtttdt_block_interior
    # Keeps the state of a nodeStateBlock node constant within each timeset
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

    # ─── EQ. ─── nodeBalanceBlock_eq ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{t ∈ b} [ Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #       (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n) ∈ startup_to_balance} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]
    #   + back-flow terms of two-way one-variable and DC arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d′,t⁻] − v_state[n,d,t])·state_unitsize[n]   ((d′,t⁻) of (d,t) in dtttdt_block_lag; for a
    #     fix_start_end node in block_forward_lag)
    #   − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·state_unitsize[n]
    #   + start terms of open-start nodes (fix_start in the first solve, the roll carry in a continuation solve) ]
    #   = − Σ_{t ∈ b} inflow[n,d,t]
    #   ∀ n ∈ nodeStateBlock, (d,b) ∈ period_block;   t ∈ b by period_block_time;   Δt = step_duration[d,t]
    #   α, β, S⁻ as in nodeBalanceCoarse_eq with the block b (h_b = Σ_{t∈b} Δt·ln(1 + loss[n,d,t])); S⁻ at the
    #   block's first step is the previous block's state
    # nodeBalance_eq summed over each timeset block of a bind_intraperiod_blocks node.  The state is constant within
    # a block (stateConstantWithinBlock_eq), so the state terms telescope to v_state[b⁻] − v_state[b]: v_state of a
    # block is its state at the end of the block.  b⁻ is the previous block of the period, the first block of a
    # period continuing from the period's last block (the blocks of a period form a loop).  In a rolling solve the
    # blocks also end at the roll's realized end; a period the roll holds whole loops as above, while in a period
    # the roll holds only in part (period_block_chain) the blocks chain forward: the period's first block continues
    # from the previous step of the solve, and at the solve's first step the chain starts from fix_start (first
    # roll), the carried level (later rolls) or, with neither, with no state change.  A fix_start_end node ignores
    # the loop (its start and end are pinned separately): its blocks chain forward in every period like that
    # (_self_discharge.block_forward_nodes / block_forward_lag).  Same orientation and units as nodeBalance_eq: its
    # dual is the node's price (currency/MWh) over the block.
    terms = _node_balance_terms(b)
    step_of = (nsb.select("n").join(d.period_block_time.select("d", "b_first", "t"), how="cross")
               .select("n", "d", "b_first", "t"))
    lhs = {key: _sum_over_coarse_step(term, step_of) for key, term in terms.items()}
    inflow = _sum_over_coarse_step(-d.p_inflow, step_of)
    rhs = {"neg_inflow": inflow} if inflow.frame.height > 0 else {}
    m.add_cstr(
        "nodeBalanceBlock_eq",
        over      = nsb.select("n").join(d.period_block.select("d", "b_first"), how="cross").select("n", "d", "b_first"),
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _eq_max_flow(b: _Build) -> None:
    """Capacity limits on process flows."""
    if not b.has_proc:
        return
    d = b.d
    pss_dt = b.pss_dt
    invest_mult = (_invest_capacity_multiplier(pss_dt, b.flow_cap_mult, b.invest_procs)
                   if b.invest_procs is not None else None)

    def _scaled(term):
        return term * invest_mult if invest_mult is not None else term

    # ─── EQ. ─── maxFlow ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] + Σ_r v_reserve_up[p,r,k,d,t] + Σ_r v_reserve_down[p,r,s,d,t] · φ[p,s,k,d,t]
    #   − M·invested[p,d]
    #   + M·retired[p,d]
    #   ≤ M · existing_count[p,d]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,s,k) ∉ flow_bound_arcs or (p,s,k) ∈ reserve_down_input_arcs
    #   M = capacity_max_coeff[p,s,k] · availability[p,d,t] on capped / uncapped arcs
    #   φ = 1 where v_flow is the input (k = p), 1/slope[p,d,t] on a direct unit's input→output arc
    #       (s ≠ p ≠ k); both reserve terms only for units (connections: connection_reserve_*, _reserve)
    # Caps every flow by the built capacity of its process (units of unitsize).
    # existing_count[p,d] = existing_mw[p,d] / unitsize[p]: existing_mw from the cohorts (engine.md "Capacity
    # symbols": the existing cohort with the user's existing of d, the investments earlier solves realized, minus
    # their carried retirements).
    # On indirect-unit input arcs M = input_share_max · full-load input · a, a = x(availability) =
    # (slope·W(a₀) + section)/(slope·W(1) + section) (a₀ = availability, W(y) the output-side fuel width
    # filled to y, section only for a unit with an online variable); a ≥ availability (never tighter than
    # the physical derated input), = availability when every output shares one conversion_flow_coeff (#102).
    # Uncapped arcs (conversion_flow_coeff = 0) have M = 1 and RHS = p_flow_upper.  Input arcs of units
    # without invest/divest are bounded on v_flow instead (flow_bound_arcs).  The reserve terms are the
    # reserve that raises the flow: upward reserve held at the sink, and
    # downward reserve held at the source of a unit (it consumes more),
    # converted to the arc's flow.  A bound input arc gets this row as well
    # when it carries such a reserve (its v_flow bound stays, redundant).
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    # A row whose RHS is inf (existing capacity inf: no limit; the Benders splitter's cross-region half-flows)
    #   is left out, here and in maxFlow_back.
    lhs: dict = {"flow": b.v_flow}
    if b.reserve_up_to_sink_units is not None:
        lhs["reserve_up"] = b.reserve_up_to_sink_units
    lhs.update(b.reserve_down_at_source)
    if b.cap.retired_p is not None:
        lhs["divest"] = _scaled(b.cap.retired_p)
    if b.cap.invested_p is not None:
        lhs["invest_neg"] = -_scaled(b.cap.invested_p)
    over = (pss_dt.join(b.flow_bound_arcs, on=["p", "source", "sink"], how="anti")
            if b.flow_bound_arcs is not None else pss_dt)
    if b.flow_bound_arcs is not None and b.reserve_down_input_arcs is not None:
        bound_with_reserve = _arcs_semi(b.flow_bound_arcs, b.reserve_down_input_arcs)
        if bound_with_reserve.height > 0:
            over = pl.concat([over, _arcs_semi(pss_dt, bound_with_reserve)])
    over = _without_no_limit_rows(over, b.flow_upper_rhs)
    if over.height > 0:
        _coarse_flows.add_cstr(
            b, "maxFlow",
            over      = over,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": b.flow_upper_rhs},
        )

    # ─── EQ. ─── maxFlow_negCap ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # (maxFlow LHS)  ≥  M · existing_count[p,d]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,d) ∈ pd_neg_cap
    # Negative-capacity processes (unitsize < 0 and existing < 0): dividing
    # v_flow·unitsize ≤ capacity by the negative unitsize flips the sense, so
    # the capacity becomes a minimum output.  maxFlow keeps the ≤ side.
    pd_neg_cap = getattr(d, "pd_neg_cap", None)
    if _has_rows(pd_neg_cap):
        neg_over = pss_dt.join(pd_neg_cap, on=("p", "d"), how="inner")
        if neg_over.height > 0:
            _coarse_flows.add_cstr(
                b, "maxFlow_negCap",
                over      = neg_over,
                sense     = ">=",
                lhs_terms = lhs,
                rhs_terms = {"upper": b.flow_upper_rhs},
            )

    # ─── EQ. ─── maxFlow_back ────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow_back[p,s,k,d,t] − M·Σ v_invest_p[p,d′] + M·Σ v_divest_p[p,d′]
    #   ≤ M · existing_count[p,d]
    #   ∀ (p,s,k,d,t) ∈ pss_dt, (p,s,k) ∈ twoway1var_arcs
    # Caps the reverse flow of two-way one-variable arcs by the same built
    # capacity, so |v_flow − v_flow_back| ≤ capacity.  No reserve term.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if b.v_flow_back is not None and b.twoway1var_arcs is not None:
        back_over = _without_no_limit_rows(
            pss_dt.join(b.twoway1var_arcs, on=("p", "source", "sink"), how="inner"), b.flow_upper_rhs)
        if back_over.height > 0:
            back_lhs: dict = {"flow_back": b.v_flow_back}
            back_lhs.update({k: lhs[k] for k in ("invest_neg", "divest") if k in lhs})
            _coarse_flows.add_cstr(
                b, "maxFlow_back",
                over      = back_over,
                sense     = "<=",
                lhs_terms = back_lhs,
                rhs_terms = {"upper": b.flow_upper_rhs},
            )


def _eq_max_output_sum(b: _Build) -> None:
    """Output-sum capacity limit of indirect (multi-flow) units."""
    if not b.has_proc or b.pss_dt.height == 0:
        return
    d = b.d
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
    #   ≤ av · existing_count[p,d]
    #   ∀ p ∈ process_indirect with an output arc k, (d,t);   av = availability[p,d,t]
    # The capacity of an indirect (multi-flow) unit is the maximum sum of its
    # outputs; maxFlow caps each output arc alone.  capacity_max_coeff is not
    # applied.  Negative-capacity (p,d) rows get only the ≥ twin
    # maxOutputSum_negCap: with both, a k-output unit would need Σ out = c
    # while maxFlow pins every arc to c.
    # existing_count[p,d] is the value maxFlow's output arcs of p carry
    # (all equal; max collapses them); a unit without an existing row gets 0.
    # Availability defaults to 1.0 for processes without availability data.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
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
    if b.cap.invested_p is not None:
        lhs["invest_neg"] = -(b.cap.invested_p * avail if avail is not None
                              else b.cap.invested_p)
    if b.cap.retired_p is not None:
        lhs["divest"] = (b.cap.retired_p * avail if avail is not None
                         else b.cap.retired_p)

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
        _coarse_flows.add_cstr(
            b, "maxOutputSum",
            over      = pos_over,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"upper": rhs},
        )
    if neg_over.height > 0:
        _coarse_flows.add_cstr(
            b, "maxOutputSum_negCap",
            over      = neg_over,
            sense     = ">=",
            lhs_terms = lhs,
            rhs_terms = {"upper": rhs},
        )


def _eq_ramp(b: _Build) -> None:
    """Ramp limits of process flows."""
    if not b.has_ramp:
        return
    d = b.d
    link, carry = ramp_step_links(d)

    # ─── EQ. ─── ramp_{sink,source}_{up,down}_constraint ─────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Units without online variables (as c14a):
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] + R↑[p,s,k,d,t]
    #   ≤ +ρ̄ · c[p,s,k,d,t] · (existing_count[p,d] + Σ_{d″} v_invest_p[p,d″] − Σ_{d″} v_divest_p[p,d″])        (up)
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − R↓[p,s,k,d,t]
    #   ≥ −ρ̄ · c[p,s,k,d,t] · (existing_count[p,d] + Σ v_invest_p − Σ v_divest_p)                              (down)
    # Units with online variables (linear or integer class), room on the units online, split by unit state:
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] + R↑[p,s,k,d,t]
    #   ≤ c · ( ρ̄ · (v_online[p,d,t] − v_startup[p,d,t]) + SU[p,s,k,d,t] · v_startup[p,d,t] )
    #     − F[p,s,k,d′,t′] · v_shutdown[p,d,t]                                                                  (up)
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − R↓[p,s,k,d,t]
    #   ≥ −c · ( ρ̄ · (v_online[p,d,t] − v_startup[p,d,t]) + SD′[p,s,k,d′,t′] · v_shutdown[p,d,t] )
    #     + F[p,s,k,d,t] · v_startup[p,d,t]                                                                     (down)
    #   ρ̄ = ramp_speed_{dir}_{side}[p,node] · 60 · (step_duration[d′,t′] + step_duration[d,t]) / 2
    #   SU = min(1, ml[p,s,k,d,t] + ½ · ramp_speed_up_{side} · 60 · step_duration[d,t])
    #   SD′ = min(1, ml[p,s,k,d′,t′] + ½ · ramp_speed_down_{side} · 60 · step_duration[d′,t′])
    #   ml = min_load (output-measured arc) or x(min_load) / input_share_max (indirect unit's input arc), 0
    #     without min_load;  F = the arc's floor per unit online (arc_floor_share): min_load on the only arc of
    #     a unit's output measure, min_load · capacity_min_coeff on an output of a multi-output unit with
    #     capacity_min_coeff > 0, else 0
    #   ∀ (p,s,k) ∈ process_source_sink_ramp_limit_{side}_{dir}, (d,t,d′,t′) ∈ dt_ramp_link; on a link inside one
    #     period (d′ = d) only where ρ̄ < 1
    # On a continuation roll the first step of a period ramps from the previous roll's realized flow
    #   (dt_ramp_carry): v_flow[p,s,k,d′,t′] → f̂[p,s,k,d,t] (p_roll_continue_flow, a constant), step_duration
    #   [d′,t′] → its carried duration; for units with online variables F[d′,t′] → 0 (no shutdown term on the
    #   up row) and SD′ → max(1, f̂[p,s,k,d,t] / (c[p,s,k,d,t] · ĉ[p])) (1 where the carried online count
    #   ĉ = p_roll_continue_online is 0 — no stoppers then; exact and always feasible, unlike a fixed κ/c).
    #   On the carried down row of a unit without online variables f̂ → min(f̂, U[p,s,k,d,t] + ρ̄ · c ·
    #   existing_count[p,d]) (fine steps), U the arc's maxFlow RHS (its flow ceiling at the existing capacity): a
    #   carried ramp-down yields to an outage the previous solve did not see — the row then asks at most U (W-0592).
    # A coarse process (processCoarse, _coarse_flows.ramp_rows) ramps between its coarse steps: t = b (named by its
    #   first fine step), (d′,t′) = b⁻, the coarse step that holds the fine predecessor of b's first step, and
    #   step_duration → Δ_b = Σ_{t∈b} Δt, so ρ̄ = speed · 60 · (Δ_b⁻ + Δ_b)/2 (no row inside a coarse step; with
    #   24-h steps an in-period row exists only for a speed below 1/1440 per minute); a time-varying source factor
    #   c is its coarse-step mean; the carried step is the arc's last realized coarse step (its length carried).
    #   With online variables v_online is the step's value, v_startup / v_shutdown the events at b (start-ups and
    #   shut-downs relative to b⁻, online_change); c and F are their means over b (F[b⁻] over b⁻) and
    #   SU[b] = mean share over b of a unit started at b's start: ml + ½·ρ_b while ρ_b ≤ 1 − ml, else
    #   1 − (1 − ml)²/(2·ρ_b), ρ_b = ramp_speed_up · 60 · Δ_b, ml the mean over b (SD′[b⁻] likewise with the down
    #   speed over b⁻: the exact mean of a linear ramp that reaches full load within the step).
    # Steps without a predecessor, on arcs that hold reserve in that direction there (prundt):
    #   R↑ ≤ ρ·c·cap (up),  −R↓ ≥ −ρ·c·cap (down),  ρ = ramp_speed · 60 · step_duration[d,t] < 1;
    #   cap = v_online[p,d,t] for units with online variables, built units otherwise.
    # ─── EQ. ─── ramp_{sink,source}_{up,down}_reserve_coarse ─────────────────
    # R↑[p,s,k,d,b] ≤ room_b · cap (up),  −R↓[p,s,k,d,b] ≥ −room_b · cap (down)
    #   ∀ arc of a coarse process with the ramp limit, holding reserve in that direction on its coarse step (d,b)
    #   (b = its first fine step) with some ρ_t < 1;  room_b = min_{t∈b, ρ_t<1} ρ_t · c[p,s,k,d,t], ρ_t the room of
    #   the fine row at t: ramp_speed · 60 · (step_duration[d′,t′] + step_duration[d,t]) / 2 with t's ramp
    #   predecessor (d′,t′) (a carried one: the arc's carried duration p_roll_continue_flow_duration, as the
    #   carried ramp row), ramp_speed · 60 · step_duration[d,t] without one;  c the arc's capacity factor at t;
    #   cap = v_online (units online of the step) or the built units
    # A coarse process holds one reserve and one flow per coarse step: the fine rows inside the step (no flow change)
    # would bound the reserve by each step's own room, so the smallest binds; the ramp between coarse steps above
    # still holds the reserve together with the flow change.
    # R↑ = Σ_r v_reserve[p,r,up,k,d,t] + φ·Σ_r v_reserve[p,r,down,s,d,t]
    #   (raises the arc: up at its sink, down at its source)
    # R↓ = Σ_r v_reserve[p,r,down,k,d,t] + φ·Σ_r v_reserve[p,r,up,s,d,t]
    #   (lowers it: down at its sink, up at its source)
    #   φ = 1 where v_flow is the input (indirect unit's input arc, sink-less unit), 1/slope on a direct arc;
    #   units only.
    # Limits the change of a flow between two consecutive steps, plus the reserve it holds in that direction, to its
    # ramp room: ramp_speed is a share of capacity per minute and flows are step averages, so the room between two
    # step averages is the speed × the distance between the two step centres (a valid bound for every trajectory
    # that obeys the speed; it is the tight one for a linear ramp).  For a unit with online variables the units
    # online in both steps change by at most ρ̄ each, a unit starting in step t reaches at most SU (half a step's
    # ramp above min load, the startup-step limit of maxFlow_online) and a stopping unit drops from at least its
    # floor F; downwards a stopping unit drops at most from SD′ (the pre-shutdown limit) and a starting unit rises
    # to at least F.  v_startup / v_shutdown are relative to the same predecessor (online_change), so a start/stop
    # pair with online unchanged gains c·(ml − ½·ρ·Δ) − F ≤ 0 of room on the only output of a unit
    # (research_backbone_startup_ramp.md); on an input, or on one output of a multi-output unit, a pair can still
    # gain up to the min-load share, paid by the startup cost.  A step has a predecessor (dt_ramp_link) only where
    # it is adjacent on the timeline in the un-rolled solve: no cyclic wrap at the first step of a solve or of a
    # period that does not continue the previous one, no row across a gap or at the start of a representative
    # block.  Rows inside one period with ρ̄ ≥ 1 cannot bind and are not built (unchanged from c14a: every term of
    # the online-capacity split is still implied there, design §1.4).  Cross-period and carried rows are always
    # built (the capacity, or the online predecessor, can differ).
    # c scales the room as the arc's capacity is scaled: capacity_max_coeff[p,k] on the sink side; on the source
    # side input_share_max[p,s], or on an indirect unit's input arc p_indirect_input_cap.  v_flow and v_reserve
    # are in units of unitsize, existing_count = capacity / unitsize.  Arcs with conversion_flow_coeff = 0 get no
    # row.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
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
        room_rows = None
        if b.has_coarse_proc:
            # A coarse process ramps between its coarse steps (_coarse_flows.ramp_rows); the reserve it holds in this
            # direction has its own room at the solve's steps of each coarse step (_coarse_flows.reserve_room_rows).
            fine_arcs, coarse_arcs = _coarse_flows.split_arcs(b, idx_set)
            parts = []
            if fine_arcs.height > 0:
                parts.append(_ramp_rows(d, fine_arcs, ramp_speed, side, dir_, link, carry))
            if coarse_arcs.height > 0:
                parts.append(_coarse_flows.ramp_rows(b, coarse_arcs, ramp_speed, side, link, carry))
                room_rows = _coarse_flows.reserve_room_rows(b, coarse_arcs, ramp_speed, side, dir_, link, carry)
            rows = pl.concat(parts) if parts else None
        else:
            rows = _ramp_rows(d, idx_set, ramp_speed, side, dir_, link, carry)
        for name, family_rows in ((f"ramp_{side}_{dir_}_constraint", rows),
                                  (f"ramp_{side}_{dir_}_reserve_coarse", room_rows)):
            if family_rows is not None and family_rows.height > 0:
                _ramp_family(b, name, family_rows, idx_set, ramp_speed, side, up, keys)


def _ramp_cost_rows(b: _Build) -> pl.DataFrame:
    """The rows R of ``ramp_change_eq`` (:func:`_ramp_rows` columns, ``__kind`` ``link`` / ``carry``): every arc of
    ``process_source_sink_ramp_cost`` at every step with a ramp predecessor, the same predecessors as the ramp
    limits (:func:`ramp_step_links`) but without their room filter; a coarse process at its coarse steps
    (``_coarse_flows.ramp_rows``)."""
    d = b.d
    link, carry = ramp_step_links(d)
    idx_set = d.process_source_sink_ramp_cost.select("p", "source", "sink")
    if b.has_coarse_proc:
        fine_arcs, coarse_arcs = _coarse_flows.split_arcs(b, idx_set)
        parts = []
        if fine_arcs.height > 0:
            parts.append(_ramp_rows(d, fine_arcs, None, "sink", "up", link, carry))
        if coarse_arcs.height > 0:
            parts.append(_coarse_flows.ramp_rows(b, coarse_arcs, None, "sink", link, carry))
        return pl.concat(parts)
    return _ramp_rows(d, idx_set, None, "sink", "up", link, carry)


def _eq_ramp_cost(b: _Build) -> None:
    """The change of a priced flow from its ramp predecessor, split into v_ramp_up and v_ramp_down."""
    if b.v_ramp_up is None:
        return
    rows = b.ramp_cost_rows
    keys = ["p", "source", "sink"]
    over = rows.select(*keys, "d", "t")
    carried = rows.filter(pl.col("__kind") == "carry")

    # ─── EQ. ─── ramp_change_eq ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − v_ramp_up[p,s,k,d,t] + v_ramp_down[p,s,k,d,t] = 0
    #   ∀ (p,s,k) ∈ process_source_sink_ramp_cost, (d,t,d′,t′) ∈ dt_ramp_link   (every link, no room filter)
    # carried row (dt_ramp_carry): v_flow[p,s,k,d,t] − v_ramp_up + v_ramp_down = f̂[p,s,k,d,t] (p_roll_continue_flow)
    # coarse process: t = b (its coarse step, named by the first fine step), (d′,t′) = b⁻, as _coarse_flows.ramp_rows
    # Splits the change of a flow between a step and its ramp predecessor into an increase and a decrease, both
    # priced by the ramp cost (OBJ. ramp_cost), in units of unitsize.  The predecessors are those of the ramp limits:
    # no row at the first step of a solve, after a gap in the timeline, at a period that does not continue the
    # previous one or at a representative-block start; cross-period links are kept; a stochastic branch's first step
    # links to the shared predecessor; a continuation roll's first step ramps from the flow the previous roll
    # realized, so a change at a roll boundary is charged once, in the roll that realizes it.  Arcs with
    # conversion_flow_coeff = 0 have rows too (that parameter removes the ramp limits, not the cost).  Reserves are
    # not part of the change.  An indirect unit's input and output arcs are priced separately, each by its own side.
    lhs = _ramp_flow_change(b, rows, keys)
    lhs["ramp_up"] = -Where(b.v_ramp_up, over)
    lhs["ramp_down"] = Where(b.v_ramp_down, over)
    rhs: dict = {}
    if carried.height > 0:
        rhs["roll_continue_flow"] = Param(("p", "source", "sink", "d", "t"),
                                          carried.select(*keys, "d", "t", value=pl.col("__f_hat")))
    b.m.add_cstr(
        "ramp_change_eq",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _ramp_family(b: _Build, name: str, rows: pl.DataFrame, idx_set: pl.DataFrame, ramp_speed: Param, side: str,
                 up: bool, keys: list) -> None:
    """The rows ``rows`` (:func:`_ramp_rows` columns) of the ramp family ``name`` (EQ. header in :func:`_eq_ramp`)."""
    m, d = b.m, b.d
    sign = 1.0 if up else -1.0
    over = rows.select(*keys, "d", "t")
    carried = rows.filter(pl.col("__kind") == "carry")

    lhs = _ramp_flow_change(b, rows, keys)
    if up:
        if b.reserve_up_to_sink is not None:
            lhs["reserve_up"] = b.reserve_up_to_sink
        lhs.update(b.reserve_down_at_source)
    else:
        if b.reserve_down_to_sink is not None:
            lhs["reserve_down"] = -b.reserve_down_to_sink
        lhs.update({k: -v for k, v in b.reserve_up_at_source.items()})

    # Split rows by UC class membership (design §3.2): a process has one startup method, so it belongs to
    # at most one online class; the rest are built rows (c14a's existing_count / invest / divest terms).
    online_rows: dict = {}
    online_p_frames = []
    for kind, _p_idx_k, online_set_k, v_online_k, v_startup_k, v_shutdown_k in _online.uc_classes(b):
        sub = (_utf8_keys(rows.lazy(), ["p"])
               .join(_utf8_keys(online_set_k.lazy().select("p"), ["p"]).select("_p").unique(),
                     on="_p", how="semi")
               .select(rows.columns).collect())
        if sub.height > 0:
            online_rows[kind] = (sub, v_online_k, v_startup_k, v_shutdown_k)
            online_p_frames.append(sub.select("p").unique())
    if online_p_frames:
        excl_p = pl.concat(online_p_frames).unique()
        built_rows = (_utf8_keys(rows.lazy(), ["p"])
                      .join(_utf8_keys(excl_p.lazy(), ["p"]).select("_p"), on="_p", how="anti")
                      .select(rows.columns).collect())
    else:
        built_rows = rows

    rhs: dict = {}
    if built_rows.height > 0:
        rate_built = _ramp_rate(d, idx_set, side, built_rows, keys, b=b)
        rhs["limit"] = (rate_built * b.cap.existing_count_p) * sign
        if b.has_invest_p:
            rhs["invest"] = (b.cap.invested_p * rate_built) * sign
        if b.cap.retired_p is not None:
            rhs["divest"] = (b.cap.retired_p * rate_built) * (-sign)
    for kind, (sub, v_online_k, v_startup_k, v_shutdown_k) in online_rows.items():
        for term_name, term in _online_ramp_rhs(b, idx_set, ramp_speed, side, up, sub,
                                                v_online_k, v_startup_k, v_shutdown_k, keys).items():
            rhs[f"{term_name}_{kind}"] = term
    if carried.height > 0:
        f_hat = carried.select(*keys, "d", "t", value=pl.col("__f_hat"))
        if not up and built_rows.height > 0:
            f_hat = _carried_ramp_down_yields(b, idx_set, side, built_rows, f_hat, keys)
        rhs["roll_continue_flow"] = Param(("p", "source", "sink", "d", "t"), f_hat)
    m.add_cstr(
        name,
        over      = over,
        sense     = "<=" if up else ">=",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _carried_ramp_down_yields(b: _Build, idx_set: pl.DataFrame, side: str, built_rows: pl.DataFrame,
                              f_hat: pl.DataFrame, keys: list) -> pl.DataFrame:
    """*f_hat* ``(p, source, sink, d, t, value)`` of the carried down-ramp rows with f̂ ≤ U + ρ̄ · c · ex on the fine
    carried rows of units without online variables (*built_rows*), U = the arc's maxFlow RHS (``b.flow_upper_rhs``,
    also the v_flow bound of a bound arc), ex = existing_count[p,d]: the row then asks flow ≥ U at most, the arc's
    own flow ceiling.  A higher f̂ (an outage the previous solve did not see) makes the row infeasible without
    investment; the carried ramp-down ends there instead (W-0592).  Every other row keeps its f̂."""
    d = b.d
    rows = built_rows.filter(pl.col("__kind") == "carry")
    if b.has_coarse_proc and rows.height > 0:
        rows = _coarse_flows.split_arcs(b, rows)[0]
    ex = b.cap.existing_count_p
    if rows.height == 0 or ex is None or b.flow_upper_rhs is None or not _has_rows(d.process_unit):
        return f_hat
    rows = _utf8_keys(rows.lazy(), ["p"]).join(
        _utf8_keys(d.process_unit.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi"
    ).select(rows.columns).collect()
    if rows.height == 0:
        return f_hat
    rows = _c_values(rows, _c_param_plain(d, idx_set, side), keys)
    all_keys = [*keys, "d", "t"]
    upper = b.flow_upper_rhs.frame
    cap = (_utf8_keys(rows.lazy(), all_keys)
           .join(_utf8_keys(upper.lazy(), all_keys).select(*(f"_{k}" for k in all_keys),
                                                           pl.col("value").alias("__upper")),
                 on=[f"_{k}" for k in all_keys], how="inner")
           .join(_utf8_keys(ex.frame.lazy(), ["p", "d"]).select("_p", "_d", pl.col("value").alias("__ex")),
                 on=["_p", "_d"], how="inner")
           .select(*(f"_{k}" for k in all_keys),
                   (pl.col("__upper") + pl.col("__c") * pl.col("__ex") * pl.col("__room")).alias("__cap"),
                   pl.col("__f_hat"))
           .filter(pl.col("__f_hat") > pl.col("__cap"))
           .collect())
    if cap.height == 0:
        return f_hat
    from flextool.engine.preprocess.params._mintime import note_carried_obligation_cut
    note_carried_obligation_cut({r["_p"]: f"solve {b.solve_name}: the carried ramp-down ends at step "
                                          f"({r['_d']}, {r['_t']})" for r in cap.iter_rows(named=True)})
    return (_utf8_keys(f_hat.lazy(), all_keys)
            .join(cap.lazy().select(*(f"_{k}" for k in all_keys), "__cap"), on=[f"_{k}" for k in all_keys],
                  how="left", maintain_order="left")
            .select(*all_keys, pl.min_horizontal("value", pl.col("__cap").fill_null(pl.col("value"))).alias("value"))
            .collect())


def _ramp_flow_change(b: _Build, rows: pl.DataFrame, keys: list) -> dict:
    """The flow change of the ramp rows ``rows`` (:func:`_ramp_rows` columns): ``flow`` = v_flow at (d,t) on the
    ``link`` and ``carry`` rows, ``flow_prev`` = −v_flow at the predecessor (d′,t′) on the ``link`` rows (a carried
    predecessor is the constant f̂ on the right side).  Shared by the ramp limits and the ramp cost."""
    v_flow = b.v_flow
    flow_rows = rows.filter(pl.col("__kind") != "reserve").select(*keys, "d", "t")
    linked = rows.filter(pl.col("__kind") == "link")
    lhs: dict = {}
    if flow_rows.height > 0:
        lhs["flow"] = Where(v_flow, flow_rows)
    if linked.height > 0:
        prev = _var_view(v_flow, "__ramp_prev", {"d": "d_previous", "t": "t_previous"}, to_axis=True)
        pairs = linked.select(*keys, "d", "t",
                              pl.col("__d_prev").cast(prev.frame.schema["d_previous"]).alias("d_previous"),
                              pl.col("__t_prev").cast(prev.frame.schema["t_previous"]).alias("t_previous"))
        lhs["flow_prev"] = -Sum(Where(prev, pairs), over=("d_previous", "t_previous"))
    return lhs


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


def _ramp_rows(d, idx_set: pl.DataFrame, ramp_speed: "Param | None", side: str, dir_: str,
               link: pl.DataFrame, carry: pl.DataFrame) -> pl.DataFrame:
    """The rows of one ramp family: ``(p, source, sink, d, t)`` in the dtypes of ``idx_set`` / ``dt`` with
    ``__kind`` (``link`` / ``carry`` / ``reserve``), ``__room`` (ρ), ``__d_prev`` / ``__t_prev`` (Utf8) and
    ``__f_hat`` (the carried flow of ``carry`` rows).  See the EQ. header of :func:`_eq_ramp`.  With
    ``ramp_speed=None`` (the ramp cost, :func:`_eq_ramp_cost`) only the ``link`` and ``carry`` rows, every one of
    them (no room filter), and ``__room`` null."""
    keys = ["p", "source", "sink"]
    base = _utf8_keys(idx_set.select(keys).unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
                      .lazy(), keys + ["d", "t"])
    if ramp_speed is not None:
        speed = ramp_speed.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                        pl.col(side).cast(pl.Utf8).alias("_n"),
                                        pl.col("value").cast(pl.Float64).alias("__speed"))
        base = (base.with_columns(pl.col(f"_{side}").alias("_n"))
                .join(speed.lazy(), on=["_p", "_n"], how="inner"))
    with_link = (base.join(link.lazy(), on=["_d", "_t"], how="inner")
                 .with_columns(__kind=pl.lit("link"), __f_hat=pl.lit(None, pl.Float64)))
    if ramp_speed is None:
        parts = [with_link]
    else:
        parts = [with_link.filter((pl.col("__d_prev") != pl.col("_d"))
                                  | (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0
                                     < 1.0))]
    if carry.height > 0:
        f_hat = d.p_roll_continue_flow.frame.select(
            pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("source").cast(pl.Utf8).alias("_source"),
            pl.col("sink").cast(pl.Utf8).alias("_sink"), pl.col("d").cast(pl.Utf8).alias("_d"),
            pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__f_hat"))
        parts.append(base.join(carry.lazy(), on=["_d", "_t"], how="inner")
                     .join(f_hat.lazy(), on=["_p", "_source", "_sink", "_d", "_t"], how="inner")
                     .with_columns(__kind=pl.lit("carry")))
    prundt = getattr(d, "prundt", None)
    if ramp_speed is None:
        cols = [*keys, "d", "t", "__kind", "__d_prev", "__t_prev", "__f_hat", pl.lit(None, pl.Float64).alias("__room")]
        return pl.concat([p.select(cols) for p in parts]).collect()
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


def _ramp_rate(d, idx_set: "pl.DataFrame", side: str, rows_subset: "pl.DataFrame", keys: list,
               b: "_Build | None" = None) -> Param:
    """``c · ρ̄`` (or ``c · ρ`` on reserve-only rows) over ``rows_subset`` (:func:`_ramp_rows` output, any
    ``__kind``): the per-row ramp room already scaled by the arc's capacity factor ``c`` (c14c design §3,
    EQ. header of :func:`_eq_ramp`).  A coarse process's time-varying source factor is its coarse step's mean
    (``b`` given: :func:`_coarse_flows.at_first_steps`).  The reserve-room rows of a coarse process
    (:func:`_coarse_flows.reserve_room_rows`) already hold c in their room, min_t ρ_t · c_t, and are not scaled
    again."""
    if b is not None and b.has_coarse_proc and "__kind" in rows_subset.columns:
        fine_rows, coarse_rows = _coarse_flows.split_arcs(b, rows_subset)
        is_room = pl.col("__kind") == "reserve"
        room_rows = coarse_rows.filter(is_room)
        if room_rows.height > 0:
            rest = pl.concat([fine_rows, coarse_rows.filter(~is_room)])
            room = room_rows.select(*keys, "d", "t", value=pl.col("__room").cast(pl.Float64))
            if rest.height > 0:
                other = _ramp_rate(d, idx_set, side, rest, keys, b=b).frame.select(*keys, "d", "t", "value")
                room = pl.concat([other, room.select(*(pl.col(c).cast(other.schema[c]) for c in other.columns))])
            return Param(("p", "source", "sink", "d", "t"), room)
    rate = Param(("p", "source", "sink", "d", "t"), rows_subset.select(*keys, "d", "t", value=pl.col("__room")))
    if side == "sink":
        if d.p_process_sink_max_capacity_coef is not None:
            rate = rate * _coef_factor(idx_set, d.p_process_sink_max_capacity_coef, ["p", "sink"])
    else:
        src_factor = _source_ramp_factor(d, idx_set)
        if b is not None and b.has_coarse_proc:
            src_factor = _coarse_flows.at_first_steps(b, src_factor)
        if src_factor is not None:
            rate = rate * src_factor
    return rate


def _c_param_plain(d, idx_set: "pl.DataFrame", side: str) -> "Param | None":
    """The ramp row's capacity factor ``c`` alone (without ``ρ̄``): :func:`_coef_factor` on the sink side,
    :func:`_source_ramp_factor` on the source side.  ``None`` means 1.0 everywhere."""
    if side == "sink":
        if d.p_process_sink_max_capacity_coef is None:
            return None
        return _coef_factor(idx_set, d.p_process_sink_max_capacity_coef, ["p", "sink"])
    return _source_ramp_factor(d, idx_set)


def _c_values(rows: "pl.DataFrame", c_param: "Param | None", keys: list) -> "pl.DataFrame":
    """``rows`` + ``__c`` (float, default 1.0): the plain capacity factor of each row, from ``c_param``
    (:func:`_c_param_plain`'s output, possibly ``None`` meaning 1.0 everywhere)."""
    if c_param is None:
        return rows.with_columns(__c=pl.lit(1.0))
    cf = c_param.frame
    on = [k for k in ("p", "source", "sink", "d", "t") if k in cf.columns]
    joined = (_utf8_keys(rows.lazy(), on)
              .join(_utf8_keys(cf.lazy(), on).select([f"_{k}" for k in on] + [pl.col("value").alias("__c")]),
                    on=[f"_{k}" for k in on], how="left")
              .with_columns(pl.col("__c").fill_null(1.0))
              .collect())
    return joined.select(rows.columns + ["__c"])


def _ramp_limit_share(b: "_Build", ramp_speed: Param, side: str, points: "pl.DataFrame") -> "pl.DataFrame":
    """``(p, source, sink, d, t, value)`` over ``points`` (``p, source, sink, d, t``, dtypes of ``idx_set`` /
    ``dt``): ``min(1, ml + ½ · ramp_speed · 60 · step_duration[d,t])`` — the startup-step (SU, called with the
    family's own up-speed) or pre-shutdown (SD′, down-speed) output limit of a unit starting / stopping at
    ``points`` (c14c design §1.2/§3.1).  ``ml`` via :func:`flextool.engine.model._online._min_load_share`,
    ``input_side`` per arc (an indirect unit's input arc: ``source != p``, ``sink == p``, ``p`` in
    ``process_indirect``); this row's own ``ramp_speed`` only — no min over sides (unlike ``maxFlow_online``,
    which keeps the tighter of the two).  A point of a coarse process (``t`` = the first fine step of its coarse
    step b) takes the mean share over the whole step, ``_coarse_flows.step_limit_share`` of the step-mean ml and
    ρ_b = ramp_speed · 60 · Δ_b."""
    if b.has_coarse_proc:
        fine_pts, coarse_pts = _coarse_flows.split_arcs(b, points)
        if coarse_pts.height > 0:
            parts = [_coarse_ramp_limit_share(b, ramp_speed, side, coarse_pts)]
            if fine_pts.height > 0:
                parts.insert(0, _ramp_limit_share(b, ramp_speed, side, fine_pts))
            return pl.concat(parts)
    d = b.d
    keys = ["p", "source", "sink"]
    pts = _utf8_keys(points.lazy(), keys + ["d", "t"]).collect()
    if pts.height == 0:
        return pts.select(*keys, "d", "t").with_columns(value=pl.lit(0.0))
    indirect = getattr(d, "process_indirect", None)
    if indirect is not None and indirect.height > 0:
        ind_p = _utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique().collect()
        is_in = ((pts["_source"] != pts["_p"]) & (pts["_sink"] == pts["_p"])
                 & pts["_p"].is_in(ind_p["_p"].to_list()))
    else:
        is_in = pl.Series([False] * pts.height)
    in_rows = pts.filter(is_in)
    out_rows = pts.filter(~is_in)
    ml_parts = []
    if in_rows.height > 0:
        ml_parts.append(_online._min_load_share(b, in_rows.lazy(), input_side=True).collect())
    if out_rows.height > 0:
        ml_parts.append(_online._min_load_share(b, out_rows.lazy(), input_side=False).collect())
    ml = (pl.concat(ml_parts) if ml_parts
          else pl.DataFrame(schema={"_p": pl.Utf8, "_source": pl.Utf8, "_sink": pl.Utf8, "_d": pl.Utf8,
                                    "_t": pl.Utf8, "__ml": pl.Float64}))
    dur = d.p_step_duration.frame.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
        pl.col("value").cast(pl.Float64).alias("__dur"))
    speed = ramp_speed.frame.select(
        pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
        pl.col("value").cast(pl.Float64).alias("__speed"))
    out = (pts.join(ml, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
           .with_columns(pl.col("__ml").fill_null(0.0))
           .join(dur, on=["_d", "_t"], how="inner")
           .with_columns(pl.col(f"_{side}").alias("_n"))
           .join(speed, on=["_p", "_n"], how="inner")
           .with_columns(value=pl.min_horizontal(
               pl.lit(1.0), pl.col("__ml") + 0.5 * pl.col("__speed") * 60.0 * pl.col("__dur")))
           .select(*keys, "d", "t", "value"))
    return out


def _coarse_ramp_limit_share(b: "_Build", ramp_speed: Param, side: str, points: "pl.DataFrame") -> "pl.DataFrame":
    """:func:`_ramp_limit_share` at ``points`` of coarse processes (``t`` = b_first): over the fine steps t of each
    coarse step b, Δ_b = Σ Δt and the Δt-weighted mean ml, then the mean share of a unit starting at the step's
    start (stopping at its end) with ρ_b = ramp_speed · 60 · Δ_b (``_coarse_flows.step_limit_share``)."""
    d = b.d
    keys = ["p", "source", "sink"]
    st = b.coarse_steps.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                               pl.col("b_first").alias("_b"), pl.col("t").cast(pl.Utf8).alias("__tf"),
                               pl.col("w").alias("__w"))
    pts = (_utf8_keys(points.select(*keys, "d", "t").unique().lazy(), keys + ["d", "t"]).collect()
           .rename({"_t": "_b"})
           .join(st, on=["_p", "_d", "_b"], how="inner")
           .rename({"__tf": "_t"}))
    fine = pts.select(*keys, pl.col("_d").cast(d.dt.schema["d"]).alias("d"),
                      pl.col("_t").cast(d.dt.schema["t"]).alias("t")).unique()
    share = _ramp_limit_share_terms(b, side, fine)
    dur = d.p_step_duration.frame.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
        pl.col("value").cast(pl.Float64).alias("__dur"))
    speed = ramp_speed.frame.select(
        pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
        pl.col("value").cast(pl.Float64).alias("__speed"))
    step_rows = (pts.join(share, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
                 .with_columns(pl.col("__ml").fill_null(0.0))
                 .join(dur, on=["_d", "_t"], how="inner")
                 .with_columns((pl.col("__w") * pl.col("__ml")).alias("__ml")))
    return (_coarse_flows.ordered_sum(step_rows, [*keys, "d", "t", "_p", "_source", "_sink"], "_t",
                                      values=("__dur", "__ml"))
            .with_columns(pl.col(f"_{side}").alias("_n"))
            .join(speed, on=["_p", "_n"], how="inner")
            .with_columns(value=_coarse_flows.step_limit_share(pl.col("__ml"),
                                                               pl.col("__speed") * 60.0 * pl.col("__dur")))
            .select(*keys, "d", "t", "value"))


def _ramp_limit_share_terms(b: "_Build", side: str, points: "pl.DataFrame") -> "pl.DataFrame":
    """``(_p, _source, _sink, _d, _t, __ml)``: the min-load share ml of :func:`_ramp_limit_share` at ``points``
    (``p, source, sink, d, t``), input-side per arc."""
    d = b.d
    keys = ["p", "source", "sink"]
    pts = _utf8_keys(points.lazy(), keys + ["d", "t"]).collect()
    indirect = getattr(d, "process_indirect", None)
    if indirect is not None and indirect.height > 0:
        ind_p = _utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique().collect()
        is_in = ((pts["_source"] != pts["_p"]) & (pts["_sink"] == pts["_p"])
                 & pts["_p"].is_in(ind_p["_p"].to_list()))
    else:
        is_in = pl.Series([False] * pts.height)
    parts = []
    if pts.filter(is_in).height > 0:
        parts.append(_online._min_load_share(b, pts.filter(is_in).lazy(), input_side=True).collect())
    if pts.filter(~is_in).height > 0:
        parts.append(_online._min_load_share(b, pts.filter(~is_in).lazy(), input_side=False).collect())
    return (pl.concat(parts) if parts
            else pl.DataFrame(schema={"_p": pl.Utf8, "_source": pl.Utf8, "_sink": pl.Utf8, "_d": pl.Utf8,
                                      "_t": pl.Utf8, "__ml": pl.Float64}))


def _predecessor_points(rows_subset: "pl.DataFrame", d) -> "pl.DataFrame":
    """``(p, source, sink, d, t)`` at the predecessor identity of the rows of ``rows_subset`` (:func:`_ramp_rows`
    link rows: ``__d_prev`` / ``__t_prev``), in the dtypes of ``d.dt``, each point once: at a branch point several
    rows (one per branch) share the predecessor, and a repeated point would repeat its value in every join on it."""
    return rows_subset.select(
        "p", "source", "sink",
        pl.col("__d_prev").cast(d.dt.schema["d"]).alias("d"),
        pl.col("__t_prev").cast(d.dt.schema["t"]).alias("t")).unique(maintain_order=True)


def _branch_point_notice(link_rows: "pl.DataFrame") -> None:
    """Behaviour-change notice naming the units whose downward ramp rows meet a branch point: several rows (one per
    branch) share the predecessor step, where the shut-down share used to be multiplied (:func:`_predecessor_points`
    counts it once)."""
    shared = (link_rows.select(pl.col("p").cast(pl.Utf8), pl.col("source").cast(pl.Utf8),
                               pl.col("sink").cast(pl.Utf8), "__d_prev", "__t_prev")
              .filter(pl.struct("p", "source", "sink", "__d_prev", "__t_prev").is_duplicated()))
    if shared.height > 0:
        from flextool.engine import _behaviour_changes

        _behaviour_changes.notice("ramp_branch_point_shutdown_once", "units", sorted(set(shared["p"].to_list())))


def _attach_at_predecessor(rows_subset: "pl.DataFrame", values: "pl.DataFrame", keys: list) -> "pl.DataFrame":
    """``values`` (``p, source, sink, d, t, value``), evaluated at the predecessor identity, reattached to
    ``rows_subset``'s OWN ``(d, t)`` via its ``__d_prev`` / ``__t_prev`` bridge (Utf8).  0 where ``rows_subset``
    has no match (no predecessor value — the carried-predecessor callers restrict to link rows only)."""
    left = _utf8_keys(rows_subset.lazy(), keys)
    right = (_utf8_keys(values.lazy(), keys + ["d", "t"])
             .select([f"_{k}" for k in keys] + [pl.col("_d").alias("__d_prev"), pl.col("_t").alias("__t_prev"),
                                                "value"]))
    return (left.join(right, on=[f"_{k}" for k in keys] + ["__d_prev", "__t_prev"], how="left")
            .select(*keys, "d", "t", value=pl.col("value").fill_null(0.0)).collect())


def _attach_at_self(rows_subset: "pl.DataFrame", values: "pl.DataFrame", keys: list) -> "pl.DataFrame":
    """``values`` (``p, source, sink, d, t, value``) left-joined onto ``rows_subset``'s own ``(d, t)``, 0
    where absent."""
    left = _utf8_keys(rows_subset.lazy(), keys + ["d", "t"])
    right = _utf8_keys(values.lazy(), keys + ["d", "t"]).select(
        [f"_{k}" for k in keys] + ["_d", "_t", "value"])
    return (left.join(right, on=[f"_{k}" for k in keys] + ["_d", "_t"], how="left")
            .select(*keys, "d", "t", value=pl.col("value").fill_null(0.0)).collect())


def _online_ramp_rhs(b: "_Build", idx_set: "pl.DataFrame", ramp_speed: Param, side: str, up: bool,
                     sub: "pl.DataFrame", v_online_k, v_startup_k, v_shutdown_k, keys: list) -> dict:
    """RHS terms of one UC class's rows of a ramp family, room on the units online split by unit state (c14c
    design §3): ``{"online": ..., "startup": ..., "shutdown": ...}`` (without the ``_{kind}`` suffix the
    caller adds; present only where non-empty).  ``sub`` is this class's rows of :func:`_ramp_rows` (any
    ``__kind``: link / carry / reserve) — one process belongs to at most one UC class.
    """
    d = b.d
    sign = 1.0 if up else -1.0
    terms: dict = {"online": (v_online_k * _ramp_rate(d, idx_set, side, sub, keys, b=b)) * sign}

    flow_kind = sub.filter(pl.col("__kind") != "reserve")
    if flow_kind.height == 0:
        return terms
    link_kind = flow_kind.filter(pl.col("__kind") == "link")
    carry_kind = flow_kind.filter(pl.col("__kind") == "carry")
    c_param = _c_param_plain(d, idx_set, side)
    floor_tbl = _unit_floors.arc_floor_share(d, idx_set)
    if b.has_coarse_proc:
        # A coarse process: c and F are their coarse-step means, keyed at each step's first fine step (b, b⁻).
        c_param = _coarse_flows.at_first_steps(b, c_param)
        if floor_tbl is not None:
            floor_tbl = _coarse_flows.at_first_steps(
                b, Param(("p", "source", "sink", "d", "t"), floor_tbl.select(*keys, "d", "t", "value"))).frame
    c_vals = _c_values(flow_kind, c_param, keys)

    if up:
        # startup: c·(SU_t − ρ̄) — can be 0 or negative (keep the sign, drop exact zeros).
        su = _ramp_limit_share(b, ramp_speed, side, flow_kind.select(*keys, "d", "t"))
        su_coef = (c_vals.join(su.rename({"value": "__su"}), on=[*keys, "d", "t"], how="left")
                   .with_columns(pl.col("__su").fill_null(0.0))
                   .with_columns(value=pl.col("__c") * (pl.col("__su") - pl.col("__room")))
                   .filter(pl.col("value") != 0.0)
                   .select(*keys, "d", "t", "value"))
        if su_coef.height > 0:
            terms["startup"] = v_startup_k * Param(tuple(keys) + ("d", "t"), su_coef)
        # shutdown: −F[d′,t′], link rows only (carry rows: F = 0, no term, design §3.3).
        if link_kind.height > 0 and floor_tbl is not None:
            f_prev = (_attach_at_predecessor(link_kind, floor_tbl, keys)
                      .filter(pl.col("value") > 0.0)
                      .with_columns(value=-pl.col("value")))
            if f_prev.height > 0:
                terms["shutdown"] = v_shutdown_k * Param(tuple(keys) + ("d", "t"), f_prev)
    else:
        # startup: c·ρ̄ + F[d,t] — the row's own step, link and carry rows alike.
        f_self = (_attach_at_self(flow_kind, floor_tbl, keys) if floor_tbl is not None
                 else flow_kind.select(*keys, "d", "t").with_columns(value=pl.lit(0.0)))
        su_coef = (c_vals.join(f_self.rename({"value": "__f"}), on=[*keys, "d", "t"], how="left")
                   .with_columns(pl.col("__f").fill_null(0.0))
                   .with_columns(value=pl.col("__c") * pl.col("__room") + pl.col("__f"))
                   .filter(pl.col("value") != 0.0)
                   .select(*keys, "d", "t", "value"))
        if su_coef.height > 0:
            terms["startup"] = v_startup_k * Param(tuple(keys) + ("d", "t"), su_coef)
        # shutdown: −c·SD′[d′,t′] on link rows (ml-based); −c·SD′_carry on carry rows:
        #   SD′_carry = max(1, f̂ / (c · ĉ)), ĉ = p_roll_continue_online, 1.0 where ĉ = 0 — no stoppers then.
        shutdown_parts = []
        if link_kind.height > 0:
            _branch_point_notice(link_kind)
            sd = _ramp_limit_share(b, ramp_speed, side, _predecessor_points(link_kind, d))
            sd_at_row = _attach_at_predecessor(link_kind, sd, keys)
            c_link = _c_values(link_kind, c_param, keys)
            shutdown_parts.append(
                c_link.join(sd_at_row.rename({"value": "__sd"}), on=[*keys, "d", "t"], how="left")
                .with_columns(pl.col("__sd").fill_null(0.0))
                .with_columns(value=-pl.col("__c") * pl.col("__sd"))
                .select(*keys, "d", "t", "value"))
        if carry_kind.height > 0:
            rco = d.p_roll_continue_online
            rco_df = (rco.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("value").alias("__c_hat"))
                      if rco is not None else pl.DataFrame(schema={"_p": pl.Utf8, "__c_hat": pl.Float64}))
            c_carry = _c_values(carry_kind, c_param, keys)
            shutdown_parts.append(
                _utf8_keys(c_carry.lazy(), ["p"]).join(rco_df.lazy(), on="_p", how="left")
                .with_columns(pl.col("__c_hat").fill_null(0.0))
                .with_columns(__sd=pl.when(pl.col("__c_hat") > 0.0)
                              .then(pl.max_horizontal(
                                  pl.lit(1.0), pl.col("__f_hat") / (pl.col("__c") * pl.col("__c_hat"))))
                              .otherwise(1.0))
                .with_columns(value=-pl.col("__c") * pl.col("__sd"))
                .select(*keys, "d", "t", "value").collect())
        if shutdown_parts:
            sd_all = pl.concat(shutdown_parts).filter(pl.col("value") != 0.0)
            if sd_all.height > 0:
                terms["shutdown"] = v_shutdown_k * Param(tuple(keys) + ("d", "t"), sd_all)
    return terms


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
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n
            or b.has_divest_built_p or b.has_divest_built_n):
        return
    m, d = b.m, b.d

    # ─── EQ. ─── max{Invest,Divest}_entity_period_{p,n} ──────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,o,d) ∈ edd_invest_commission} v_invest_p[p,o] · unitsize[p] + carried_commission[p,d]
    #                                      ≤ invest_max_period[p,d]   ∀ (p,d) with a decision coming online in d,
    #                                                                   p ∈ ed_invest_period_set
    # (n: v_invest_n with state_unitsize[n], n ∈ ed_invest_period_set ∩ nd_invest_set)
    # R_p[p,d] · unitsize[p]        ≤ divest_max_period[p,d]   ∀ (p,d) ∈ ed_divest_period_set ∩ (pd_divest_set ∪ B_p)
    # R_n[n,d] · state_unitsize[n]  ≤ divest_max_period[n,d]   ∀ (n,d) ∈ ed_divest_period_set ∩ (nd_divest_set ∪ B_n)
    #   R[e,d] = v_divest[e,d] + Σ_c v_divest_built[e,c,d] (existing and model-built capacity retired in d, each
    #     term where it has a column); B = the (e,d) of cd_divest_built (_retire_built)
    #   (p,d) and (n,d) whose invest_max_period / retire_max_period is inf have no row (no limit in that period)
    # Caps the capacity coming online or retired in one period for entities with a per-period limit
    # (e.g. invest_method = invest_period_total).  An investment decided in o comes online in d = c(o)
    # (construction lead time); without a lead time c(o) = o and the row is v_invest_p[p,d] · unitsize[p] ≤
    # invest_max_period[p,d] over ed_invest_period_set ∩ pd_invest_set, the form built when no decision of the
    # solve has a lead time.  carried_commission is the capacity earlier solves of a chain decided that comes
    # online in d.  The cap is raw capacity with unitsize on the LHS, as in the other invest caps, so the row
    # dual is in objective units per unit of capacity.
    for name, active, period_set, var_set, cap, var, axis, kind, param_name in (
        ("maxInvest_entity_period_p", b.has_invest_p, d.ed_invest_period_set, d.pd_invest_set,
         d.ed_invest_max_period, b.v_invest_p, "p", "invest", "invest_max_period"),
        ("maxInvest_entity_period_n", b.has_invest_n, d.ed_invest_period_set, d.nd_invest_set,
         d.ed_invest_max_period, b.v_invest_n, "n", "invest", "invest_max_period"),
        ("maxDivest_entity_period_p", b.has_divest_p, d.ed_divest_period_set, d.pd_divest_set,
         d.ed_divest_max_period, b.v_divest_p, "p", "divest", "retire_max_period"),
        ("maxDivest_entity_period_n", b.has_divest_n, d.ed_divest_period_set, d.nd_divest_set,
         d.ed_divest_max_period, b.v_divest_n, "n", "divest", "retire_max_period"),
    ):
        # A per-period retirement limit also sums the retirements of model-built capacity (_retire_built).
        built = _retire_built.decided(b, axis) if kind == "divest" else None
        if not (active or built is not None) or not _has_rows(period_set):
            continue
        unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
        if kind == "invest" and _lead_time.active(d):
            if cap is None:
                continue
            entities = period_set.pipe(rename_to_axis, {"e": axis}).join(var_set, on=[axis, "d"], how="semi")
            rows = _lead_time.commission_rows(d, var, axis).select(axis, "d").unique()
            rows, ents = align_join_dtypes(rows, entities.select(axis).unique(), [axis])
            check_max_cap(rows.join(ents, on=axis, how="semi"), cap.frame.pipe(rename_to_axis, {"e": axis}),
                          axis, True, param_name=param_name)  # FA-12
            _lead_time.period_limit(b, name, axis, var, unitsize, entities, cap, "<=", param_name)
            continue
        cells = var_set if built is None else _retire_built.retirement_cells(b, axis, var if active else None)
        over = period_set.pipe(rename_to_axis, {"e": axis}).join(cells, on=[axis, "d"], how="inner")
        if over.height == 0 or cap is None:
            continue
        cap_frame = cap.frame.pipe(rename_to_axis, {"e": axis})
        check_max_cap(over, cap_frame, axis, True, param_name=param_name)  # FA-12
        cap_param = Param((axis, "d"), cap_frame.select(axis, "d", "value"))
        # A period with the limit inf has no row (no limit there).
        over = without_no_limit_rows(over, cap_param, "upper")
        if over.height == 0:
            continue
        lhs: dict = {kind: var * unitsize} if active else {}
        if built is not None:
            lhs["divest_built"] = built * unitsize
        m.add_cstr(
            name,
            over      = over,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"cap": drop_no_limit_rows(cap_param, "upper")},
        )


def _eq_invest_total_caps(b: _Build) -> None:
    """Capacity invested or divested over the horizon, for entities with a total limit."""
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n
            or b.has_divest_built_p or b.has_divest_built_n):
        return
    m, d = b.m, b.d
    d_leaf = getattr(d, "d_leaf", None)

    # Capacity realized (invested) or retired (divested) by all earlier solves of a solve chain,
    # per entity (Q10) — every investment of the chain counts, retired or not; retirements count
    # whether or not a later solve re-invests in the same entity.
    prior_invest = chain_prior(d, "invest")
    prior_divest = chain_prior(d, "divest")

    # ─── EQ. ─── max{Invest,Divest}_entity_total[_path][_n] ──────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{d ∈ D(leaf)} v_invest_p[p,d] · unitsize[p]  ≤  max(0, invest_max_total[p] − prior_inv[p])
    #   ∀ p ∈ e_invest_total ∩ process_source_sink ∩ P(v_invest_p) (, leaf)
    # Σ_{d ∈ D(leaf)} R_p[p,d] · unitsize[p]  ≤  max(0, divest_max_total[p] − prior_div[p])
    #   ∀ p ∈ e_divest_total ∩ process_source_sink ∩ P(v_divest_p ∪ v_divest_built_p) (, leaf)
    #   R[e,d] = v_divest[e,d] + Σ_c v_divest_built[e,c,d] (existing and model-built capacity retired in d)
    #   prior_inv[e] = p_entity_invested[e]  (capacity realized by all earlier solves of the
    #     chain, retired or not); prior_div[e] = p_entity_divested[e]  (capacity retired by all
    #     earlier solves of the chain, existing and model-built, at realized non-branch periods only)
    #   _n: v_invest_n / R_n with state_unitsize[n], n ∈ nodeState
    #   _path: one row per (entity, leaf), D(leaf) = periods of the leaf, only for (entity, leaf)
    #     pairs that have the variable
    # Caps the amount built (retired) over the chain; reinvestments count, retirements do not
    # reduce it.  Capacity in service is cumulative_max_capacity.  Rows only for entities that have
    # the variable in this solve (#43).  If a user realizes a period in one solve and invests in it
    # again in a later one, it is counted twice in prior_inv — same as the old (now-removed)
    # max_d(p_entity_previously_invested_capacity) form; no fixture does this.
    for active, total_set, max_total, axis, var, legacy_name, path_name, label, prior, param_name in (
        (b.has_invest_p, d.e_invest_total, d.e_invest_max_total, "p", b.v_invest_p,
         "maxInvest_entity_total", "maxInvest_entity_total_path", "invest_total", prior_invest,
         "invest_max_total"),
        (b.has_divest_p, d.e_divest_total, d.e_divest_max_total, "p", b.v_divest_p,
         "maxDivest_entity_total", "maxDivest_entity_total_path", "divest_total", prior_divest,
         "retire_max_total"),
        (b.has_invest_n, d.e_invest_total, d.e_invest_max_total, "n", b.v_invest_n,
         "maxInvest_entity_total_n", "maxInvest_entity_total_path_n", "invest_total", prior_invest,
         "invest_max_total"),
        (b.has_divest_n, d.e_divest_total, d.e_divest_max_total, "n", b.v_divest_n,
         "maxDivest_entity_total_n", "maxDivest_entity_total_path_n", "divest_total", prior_divest,
         "retire_max_total"),
    ):
        # A retirement total also sums the retirements of model-built capacity (_retire_built).
        built = _retire_built.decided(b, axis) if label == "divest_total" else None
        if not (active or built is not None) or not _has_rows(total_set) or max_total is None:
            continue
        # Entities with flows (process_source_sink) or a state (nodeState) AND the variable: a
        # Benders region subproblem carries dangling v_invest_p columns of the coupling pipes the
        # master owns; their cap row belongs to the master only.
        members = d.process_source_sink if axis == "p" else d.nodeState
        if not _has_rows(members):
            continue
        cells = _retire_built.retirement_cells(b, axis, var if active else None) if built is not None else var.frame
        over_e = _entity_rows(total_set, "e", members, axis).join(
            cells.select(axis).unique(), on=axis, how="semi")
        if over_e.height == 0:
            continue
        max_total_frame = max_total.frame.pipe(rename_to_axis, {"e": axis})
        check_max_cap(over_e, max_total_frame, axis, False, param_name=param_name)  # FA-12
        cap = Param((axis,), minus_prior(max_total_frame, axis, prior))
        unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
        terms = over_e.join(cells.select(axis, "d").unique(), on=axis, how="inner")
        is_path, over, mask = leaf_rows(terms, axis, var if built is None else cells, d_leaf)
        name = path_name if is_path else legacy_name
        lhs: dict = {label: Sum(Where(var * unitsize, mask), over=("d",))} if active else {}
        if built is not None:
            lhs[f"{label}_built"] = Sum(Where(built * unitsize, mask), over=("d",))
        m.add_cstr(
            name,
            over      = over,
            sense     = "<=",
            lhs_terms = lhs,
            rhs_terms = {"cap": cap},
        )



def _eq_retired_existing(b: _Build) -> None:
    """Retired capacity within the capacity that existed before the run."""
    if not (b.has_divest_p or b.has_divest_n):
        return
    m, d = b.m, b.d

    # ─── EQ. ─── maxRetired_existing_{p,n} ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # retired[p,d] · unitsize[p]  ≤  pre_existing[p,d] − retired_carried[p,d]
    #   ∀ (p,d) with some (p,d′,d) ∈ retired, p ∈ process_source_sink
    # retired[n,d] · state_unitsize[n]  ≤  pre_existing[n,d] − retired_carried[n,d]
    #   ∀ (n,d) with some (n,d′,d) ∈ retired, n ∈ nodeState
    #   retired[e,d] = Σ_{d′ : (e,d′,d) ∈ retired} v_divest[e,d′] (_capacity), retired = {(e,d′,d) : (e,d′) ∈
    #     ed_divest_set, yr[d′] ≤ yr[d] < EOL[e]}, EOL[e] the end of life of the entity's existing cohort:
    #     yr[d0] − age[e] + lifetime_technical[e,d0] rounded to a period start by end_of_life_method, d0 the
    #     run's first period, age from age_at_model_start or model.start_year − build_year (none for
    #     reinvest_automatic)
    #   pre_existing = p_entity_pre_existing (the existing cohort's size: the user's existing × unitsize,
    #     lifetime-gated, MW, MWh)
    #   retired_carried = p_entity_retired_carried (the existing cohort's retirements that earlier solves of a
    #     chain realized and that still count in d; handoff realized_retire, engine.md "Capacity symbols")
    # Only capacity that existed before the run can be retired: the capacity retired so far and still counting in
    # d stays within the pre-existing capacity of d (a declining existing map, or an entity that also invests,
    # cannot retire more than was there).  A retirement option exists only where pre_existing > 0.
    pre_frame = _capacity.existing_not_retired(b)
    for axis, active_set, var, members in (
            ("p", b.cap.retired_frame("p"), b.v_divest_p if b.has_divest_p else None,
             d.process_source_sink),
            ("n", b.cap.retired_frame("n"), b.v_divest_n if b.has_divest_n else None, d.nodeState)):
        if var is None or not _has_rows(active_set) or not _has_rows(members):
            continue
        active = _entity_rows(active_set, axis, members, axis)
        if active.height == 0:
            continue
        unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
        view = _var_view(var, "__retired_existing", {"d": "d_divest"}, to_axis=True)
        retired = Sum(Where(view * unitsize, active), over=("d_divest",))
        over = active.select(axis, "d").unique().sort(axis, "d")
        if pre_frame is not None:
            cap_frame = _entity_rows(pre_frame, "e", over, axis)
            over, cap_frame = align_join_dtypes(over, cap_frame, [axis, "d"])
            cap_frame = cap_frame.join(over, on=[axis, "d"], how="semi").select(axis, "d", "value")
        else:
            cap_frame = over.with_columns(value=pl.lit(0.0))
        m.add_cstr(
            f"maxRetired_existing_{axis}",
            over      = over,
            sense     = "<=",
            lhs_terms = {"retired":  retired},
            rhs_terms = {"existing": Param((axis, "d"), cap_frame)},
        )


def _eq_conversion(b: _Build) -> None:
    """Conversion of indirect (multi-flow) units."""
    if not b.has_indirect:
        return
    d = b.d
    v_flow = b.v_flow

    # ─── EQ. ─── conversion_indirect ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{s ∈ in(p)} v_flow[p,s,p,d,t] · unitsize[p] · conv_in[p,s]
    #   + Σ_s Σ_{(t_s,t,δ) ∈ delay} v_flow[p,s,p,d,t_s] · unitsize[p] · delay_weight[p,δ] · conv_in[p,s]
    #   = Σ_{k ∈ out(p)} v_flow[p,p,k,d,t] · unitsize[p] · slope[p,d,t] · conv_out[p,k]
    #     + v_online[p,d,t] · section[p,d,t] · unitsize[p]
    #   ∀ (p,d,t) ∈ process_indirect_dt
    #   section[p,d,t] = min_load·(1/eta_min − 1/eta) / (1 − min_load)
    #   slope[p,d,t] = 1/eta − section[p,d,t]   (= (1/eta − min_load/eta_min) / (1 − min_load))
    #     for min_load_efficiency units (eta = efficiency, default 1; min_load = 0 where not authored;
    #     eta_min = efficiency_at_min_load, else efficiency); slope = 1/eta and no section for the other units
    #     (indirect units never fold conv_in/conv_out into slope/section themselves — see conv_in/conv_out below)
    #   reduce_unit_detail_method online*: constant efficiency at full load, slope' = slope + section / W,
    #     section' = 0 (W: weighted output of one unit online at full output; W = 1: 1/eta); a multi-line
    #     piecewise_linear unit joins this row with slope' = 1/η(1) (W = 1), else max_g(pw_slope·W + pw_intercept) / W
    #     (_reduce_detail module docstring, "Full-load efficiency")
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
    over = b.process_indirect_dt
    if b.has_piecewise:
        # Multi-line piecewise_linear units convert by conversion_piecewise instead.
        over = (_utf8_keys(over.lazy(), ["p"])
                .join(_utf8_keys(d.process_piecewise.lazy().select("p"), ["p"]).select("_p").unique(),
                      on="_p", how="anti")
                .select(over.columns).collect())
    _coarse_flows.add_cstr(
        b, "conversion_indirect",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _eq_conversion_piecewise(b: _Build) -> None:
    """Part-load fuel curve of the piecewise_linear units with two or more lines."""
    if not b.has_piecewise:
        return
    d = b.d
    v_flow = b.v_flow
    pw = d.process_piecewise.select("p").unique()
    indirect = {str(x) for x in d.process_indirect.get_column("p").to_list()} if b.has_indirect else set()
    outside = sorted({str(x) for x in pw.get_column("p").to_list()} - indirect)
    if outside:
        raise ValueError(f"conversion_piecewise: piecewise_linear unit(s) {outside} are not indirect (process_indirect); "
                         "the load boundary forces their fork.")

    # ─── EQ. ─── conversion_piecewise ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{s ∈ in(p)} v_flow[p,s,p,d,t] · unitsize[p] · conv_in[p,s]
    #   + Σ_s Σ_{(t_s,t,δ) ∈ delay} v_flow[p,s,p,d,t_s] · unitsize[p] · delay_weight[p,δ] · conv_in[p,s]
    #   ≥ pw_slope[p,g,d] · Σ_{o ∈ out(p)} v_flow[p,p,o,d,t] · unitsize[p] · conv_out[p,o]
    #     + pw_intercept[p,g,d] · unitsize[p] · U[p,d,t]
    #   ∀ (p,g,d,t): p ∈ process_piecewise, (p,d,t) ∈ process_indirect_dt, g ∈ seg(p,d)
    #   U[p,d,t] = v_online_linear[p,d,t] | v_online_integer[p,d,t]            p with online variables
    #   U[p,d,t] = existing_count[p,d] + invested[p,d] − retired[p,d]          p without (existing_count → RHS)
    #   pw_intercept[p,g,d] = ℓ_{g−1}·ℓ_g·(1/η_{g−1} − 1/η_g) / (ℓ_g − ℓ_{g−1}),  pw_slope[p,g,d] = 1/η_g − pw_intercept/ℓ_g
    #     (ℓ_i, η_i: piecewise_linear_efficiency points, merged when collinear; origin line g = 0: 1/η₀, 0 when the
    #     efficiency falls from the first point; without online variables: 1/max η, 0 and the falling segments;
    #     reduce_unit_detail_method online*: no row, the unit runs at its full-load efficiency in
    #     conversion_indirect, _reduce_detail._full_load_piecewise)
    # Part-load fuel curve of a piecewise_linear unit: its weighted input is at least each line of the curve,
    # so the input follows the curve's upper envelope; the curve is convex (checked before the solve), so the
    # cheapest feasible input lies on the curve.  Scaled by the units online (the perspective of the curve:
    # a fleet at equal load), or by the units built for a unit without online variables.  conv_in / conv_out =
    # conversion_flow_coeff (default 1); the load share is the weighted output per unit of U·unitsize, as the
    # min_load_efficiency section.  An input without cost (or with a negative value) can be burnt above the
    # curve; the post-solve check warns.
    # A coarse process (processCoarse) has one row per line and coarse step (conversion_piecewise_coarse,
    # _coarse_flows.add_cstr): the flows and v_online are one value per coarse step and the lines have no time
    # axis, so the averaged row is the fine row at the coarse step.  A delayed unit's inputs are those at the
    # earlier source-side steps t_s, as conversion_indirect.
    pidt = b.process_indirect_dt
    keys = ["p", "d"]
    lines = d.p_piecewise_slope.frame
    lines = lines.collect() if isinstance(lines, pl.LazyFrame) else lines
    over = (_utf8_keys(pidt.lazy(), keys)
            .join(_utf8_keys(lines.lazy().select("p", "seg", "d"), keys).select("_p", "_d", "seg"),
                  on=["_p", "_d"], how="inner")
            .select("p", "seg", "d", "t")
            .collect())
    if over.height == 0:
        return

    in_flows_undelayed = d.process_input_flows
    if _has_rows(getattr(d, "process_delayed", None)):
        in_flows_undelayed = in_flows_undelayed.join(d.process_delayed, on="p", how="anti")
    input_expr = v_flow * d.p_unitsize
    if d.p_process_source_conversion_flow_coeff is not None:
        input_expr = input_expr * d.p_process_source_conversion_flow_coeff
    output_expr = v_flow * d.p_unitsize
    if d.p_process_sink_conversion_flow_coeff is not None:
        output_expr = output_expr * d.p_process_sink_conversion_flow_coeff
    output_expr = output_expr * d.p_piecewise_slope
    pw_in = d.process_input_flows.join(pw, on="p", how="semi")
    in_flows_undelayed = in_flows_undelayed.join(pw, on="p", how="semi")
    pw_out = d.process_output_flows.join(pw, on="p", how="semi")

    lhs: dict = {"input": Sum(Where(input_expr, in_flows_undelayed), over=("source", "sink"))}
    delayed_term = _delay.delayed_input_expr(d, v_flow)
    if delayed_term is not None and _has_rows(getattr(d, "process_delayed", None)):
        if d.process_delayed.join(pw, on="p", how="semi").height > 0:
            lhs["input_delayed"] = delayed_term
    if pw_in.height == 0 or pw_out.height == 0:
        raise ValueError("conversion_piecewise: piecewise_linear units need an input and an output arc.")
    rhs: dict = {"output": Sum(Where(output_expr, pw_out), over=("source", "sink"))}

    # U: the units online (units with online variables), else the units built.
    intercept = d.p_piecewise_intercept
    online_p: list[pl.DataFrame] = []
    for kind, flag, online_set, var in (("lin", b.has_online_lin, d.process_online_linear, b.v_online_linear),
                                        ("int", b.has_online_int, d.process_online_integer, b.v_online_integer)):
        if not flag or not _has_rows(online_set):
            continue
        on_pw = pw.join(online_set.select(pl.col("p").cast(pw.schema["p"], strict=False)).unique(),
                        on="p", how="semi")
        if on_pw.height == 0:
            continue
        online_p.append(on_pw)
        rhs[f"section_{kind}"] = Where(var, on_pw) * intercept * d.p_unitsize
    built = pw.join(pl.concat(online_p).unique(), on="p", how="anti") if online_p else pw
    if built.height > 0:
        ic = intercept.frame
        ic = ic.collect() if isinstance(ic, pl.LazyFrame) else ic
        ic_built = (_utf8_keys(ic.lazy(), ["p"])
                    .join(_utf8_keys(built.lazy(), ["p"]).select("_p"), on="_p", how="semi")
                    .select("p", "seg", "d", "value").collect())
        # intercept · unitsize per unit built, (p, seg, d).
        us = d.p_unitsize.frame
        us = us.collect() if isinstance(us, pl.LazyFrame) else us
        per_unit = (_utf8_keys(ic_built.lazy(), ["p"])
                    .join(_utf8_keys(us.lazy().select("p", pl.col("value").alias("__us")), ["p"])
                          .select("_p", "__us"), on="_p", how="left")
                    .select("p", "seg", "d", value=pl.col("value") * pl.col("__us").fill_null(1.0))
                    .collect())
        per_unit_p = Param(("p", "seg", "d"), per_unit)
        ec = b.cap.existing_count_p
        if ec is not None:
            ecf = ec.frame
            ecf = ecf.collect() if isinstance(ecf, pl.LazyFrame) else ecf
            existing = (_utf8_keys(per_unit.lazy(), keys)
                        .join(_utf8_keys(ecf.lazy().select("p", "d", pl.col("value").alias("__n")), keys)
                              .select("_p", "_d", "__n"), on=["_p", "_d"], how="inner")
                        .select("p", "seg", "d", value=pl.col("value") * pl.col("__n"))
                        .filter(pl.col("value") != 0.0)
                        .collect())
            if existing.height > 0:
                rhs["cap_existing"] = Param(("p", "seg", "d"), existing)
        if b.cap.invested_p is not None:
            rhs["cap_invest"] = b.cap.invested_p * per_unit_p
        if b.cap.retired_p is not None:
            rhs["cap_divest"] = b.cap.retired_p * per_unit_p * -1.0
    _coarse_flows.add_cstr(
        b, "conversion_piecewise",
        over      = over,
        sense     = ">=",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _eq_min_input_share(b: _Build) -> None:
    """Minimum share of each input in the input of an indirect unit."""
    if not b.has_indirect:
        return
    d = b.d
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
    _coarse_flows.add_cstr(
        b, "minInputShare",
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
    sums = (rows.sort("p", "source").group_by("p", maintain_order=True).agg(pl.col("value").sum().alias("s"))
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
    Shared by the commodity cost (commodity_cost_eff), the CO2 price (co2_price_eff), both CO2 caps
    (co2_max_period, co2_max_total) and the commodity ladder balance (commodity_ladder_balance,
    _commodity_ladder.py), so their fuel measure cannot drift apart.
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


def _startup_draw(b: _Build, pn: "pl.DataFrame | None", *, at_branch: bool = False) -> list[tuple[str, Expr]]:
    """The start-up energy that units draw from their start-up input nodes, restricted to ``pn`` (p, n, ...).

    ``[("su_lin" | "su_int", v_startup · startup_energy · unitsize)]`` in MWh per (p, n, d, t) and the other
    columns of ``pn``, unweighted: an event, so no Δt.  ``at_branch`` indexes the variable and the period of
    startup_energy at the forecast branch ``b`` instead of ``d`` (the b side of non_anticipativity_storage_use).
    Empty without start-up variables, without a start-up node in ``pn`` or without ``p_startup_energy``.  Shared by
    the node balances, the commodity cost, the CO2 price, both CO2 caps and the commodity ladder balance, so their
    start-up measure cannot drift apart.
    """
    # A shared term, not a row: its formula is in the headers of the rows and objective terms that use it
    # (nodeBalance_eq and its coarse / block / period variants, non_anticipativity_storage_use, co2_max_period,
    # co2_max_total, commodity_ladder_balance; OBJ. commodity_cost_startup, co2_price_startup):
    #   draw[p,n,d,t] = v_startup_{linear,integer}[p,d,t] · startup_energy[p,n,d] · unitsize[p]   [MWh, an event]
    #     ∀ (p,d,t) ∈ online_rows(kind) (a coarse process: the first fine step of each coarse step), (p,n) ∈ pn
    # A start-up takes startup_energy (MWh per MW started) from the start-up input node n (startup_input_method
    # startup_and_input or startup_only).  Not a flow: it is not in v_flow, the conversion, capacities, input
    # shares, ramps, reserves, constraint_flow_coeff, flow groups or the capacity margin.  Linear and integer
    # start-ups alike (v_startup_integer is continuous): a linear unit stays an LP.
    d = b.d
    se = getattr(d, "p_startup_energy", None)
    if se is None or not _has_rows(pn):
        return []
    if at_branch:
        se = Param(tuple("b" if x == "d" else x for x in se.dims), se.frame.rename({"d": "b"}))
    terms: list[tuple[str, Expr]] = []
    for key, v in (("su_lin", b.v_startup_linear), ("su_int", b.v_startup_integer)):
        if v is None:
            continue
        if at_branch:
            v = _var_view(v, "__nab", {"d": "b"})
        terms.append((key, Where(v * se * d.p_unitsize, pn)))
    return terms


def _validate_startup_inputs(d) -> None:
    """Raise when a unit of the solve has a start-up input node (process_startup_node) but no online variables.

    The start-up draw is a term on the units started (v_startup), so without online variables it would be left
    out without a word.  The input derivation checks the database (validate_startup_inputs); this check covers
    hand-built data."""
    pn = d.process_startup_node
    online = {str(x) for f in (d.process_online_linear, d.process_online_integer) if _has_rows(f)
              for x in f.get_column("p").to_list()}
    missing = sorted({str(p) for p in pn.get_column("p").to_list()} - online)
    if missing:
        raise FlexToolConfigError(
            f"startup_input_method needs online variables, but these units have a start-up input and none: "
            f"{', '.join(missing)}. Set startup_method linear or binary (or a minimum_time_method), or set "
            "startup_input_method just_input.")


def _co2_period_cap_with_carry(d) -> Param:
    """co2_max_period's remaining budget (Q22): allocated up to the end of this roll's horizon minus
    what earlier rolls already realized -- the commodity ladder's realized-based carried-budget pattern
    (#89/#143), reused here because, unmodified, ``co2_max_period`` has the exact bug #38 names for
    ``co2_max_total``: a period that straddles several overlapping rolling windows (e.g. a 30h window on
    a 6h jump) gets the FULL per-window cap again on every one of those rolls, since nothing carries
    what was already realized -- over-allowing roughly by the number of rolls touching the period.

    cap[g,d] = co2_max_period[g,d]·f_d_k[d] − co2_cum_realized_tonnes[g,α(d)] / period_share[d]

    α(d) is the period label of a forecast branch period d (d itself otherwise): the realization solve carries
    the realized tonnes under the label, and share, f_d_k and the realized tonnes of a forecast come from α(d)
    unless it has its own row (``branch_periods_inherit``), so every forecast continues from its period's
    realized emissions.

    f_d_k[d] -- ``p_co2_f_d_k``, the raw ``solve_data/f_d_k.csv`` Param, 1.0 when it has no row for
    ``d`` or is absent entirely (the old, full-cap-every-roll behaviour) -- is the per-period "share of
    d's hours reached by the end of this roll's horizon": CUMULATIVE across rolls (folds a finished
    period in at 1.0, #143), unlike ``p_window_fraction`` (still validated for dt-completeness, no
    longer used here), which only measures this roll's OWN window and is not safe to subtract a
    cross-roll realized carrier against (it can shrink roll-to-roll independent of what was realized,
    driving the remainder negative). Dividing the realized carrier by period_share clears the same
    /period_share the EQ's own LHS carries, so the row is, after that cancels, "this roll's own raw
    window emissions ≤ rate[g,d]·(hours reached so far) − realized tonnes" — feasible by construction:
    an earlier roll could only realize part of its own, smaller, earlier allocation.
    """
    keys = d.group_d_co2_capped.select("g", "d").unique()
    cap = keys.join(d.p_co2_max_period.frame.rename({"value": "cap"}), on=["g", "d"], how="left")
    fdk = getattr(d, "p_co2_f_d_k", None)
    if fdk is not None:
        cap = (cap.join(branch_periods_inherit(d, fdk.frame, ()).rename({"value": "fdk"}), on="d", how="left")
               .with_columns(pl.col("fdk").fill_null(1.0)))
    else:
        cap = cap.with_columns(fdk=pl.lit(1.0))
    cap = cap.join(branch_periods_inherit(d, d.p_period_share.frame, ()).rename({"value": "share"}),
                   on="d", how="left")
    realized = getattr(d, "p_co2_cum_realized_tonnes", None)
    if realized is not None:
        cap = (cap.join(branch_periods_inherit(d, realized.frame, ("g",)).rename({"value": "realized"}),
                        on=["g", "d"], how="left")
               .with_columns(pl.col("realized").fill_null(0.0)))
    else:
        cap = cap.with_columns(realized=pl.lit(0.0))
    cap = cap.with_columns(value=pl.col("cap") * pl.col("fdk") - pl.col("realized") / pl.col("share"))
    return Param(("g", "d"), cap.select("g", "d", "value"))


def _eq_co2_max_period(b: _Build) -> None:
    """CO2 emission cap per period."""
    # A group whose co2_max_period is inf in every period of the solve has no (g, d) left (_direct: an inf period
    # leaves group_d_co2_capped).
    if not b.has_co2_cap or not _has_rows(b.d.group_d_co2_capped):
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    # ─── EQ. ─── co2_max_period ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped, t}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c]·w[d,t]
    #   + Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_noEff, t} v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w[d,t]
    #   − Σ_{(p,s,k,c,g) ∈ flow_to_co2_capped, t} v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w[d,t]
    #   + Σ_{(p,n,c,g) ∈ startup_from_co2_capped, t} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]
    #       · co2_content[c]·w[d,t]/step_duration[d,t]   (start-up energy: an event, no step_duration)
    #   ≤ co2_max_period[g,d] · f_d_k[d]  −  co2_cum_realized_tonnes[g,α(d)] / period_share[d]
    #   ∀ (g,d) ∈ group_d_co2_capped;   w = step_duration · timestep_weight / period_share
    #   group_d_co2_capped = the capped groups × the periods of dt, without the (g,d) where co2_max_period = inf
    #   (no cap in that period: no row and no carry)
    #   f_d_k[d] = (realized hours of α(d) from earlier solves + this solve's hours of d) / complete hours of α(d)
    #   α(d) = the period label of a forecast branch period d, d itself otherwise; period_share, f_d_k and the
    #   realized carry of a forecast come from α(d) unless it has its own row: every forecast continues from
    #   its period's realized history (the realization solve's carry)
    # Caps the annualised CO2 emissions of a group in each period (tonnes per year): dividing by
    # period_share, the LHS is the emissions of the solve's own steps of d annualised over their hours.
    # The RHS (_co2_period_cap_with_carry) is this period's remaining budget as of this roll: f_d_k[d]
    # (1 outside rolling / chained solves) is the cumulative share of d's hours reached by the end of
    # this roll's horizon, and the realized carrier (0 on the first solve of a chain) is what earlier
    # rolls / solves already emitted for this exact period, net of removal credits (#19) — never
    # negative by construction (#38, Q22; see the helper's own docstring).  Emissions come from
    # the fuel that processes draw from the group's nodes when it carries a CO2-emitting commodity c,
    # measured as in co2_price_eff (_eff_fuel_input): on eff processes the output flow times slope plus,
    # for min_load_efficiency units (linear and integer v_online), the section fuel of the units online.
    # noEff processes take neither slope nor section (applying them would over-count).  The removal term
    # (#19, Q14) subtracts flows into the same group's node(s) — e.g. DAC/BECCS — unsplit, no slope, same
    # measure as flow_to_co2_priced; it only loosens the cap, never tightens it.
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
    if _has_rows(getattr(d, "flow_to_co2_capped", None)):
        lhs["removal"] = -Sum(
            Where(v_flow * d.p_unitsize, d.flow_to_co2_capped)
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share,
            over=("p", "source", "sink", "c", "t"),
        )
    if b.has_startup_draw:
        for key, term in _startup_draw(b, getattr(d, "startup_from_co2_capped", None)):
            lhs[f"emissions_startup_{key}"] = Sum(
                term * d.p_co2_content * d.p_timestep_weight / d.p_period_share,
                over=("p", "n", "c", "t"),
            )
    m.add_cstr(
        "co2_max_period",
        over      = d.group_d_co2_capped,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"cap": _co2_period_cap_with_carry(d)},
    )


def _co2_total_cap_with_carry(d) -> Param:
    """co2_max_total's remaining budget (#38): allocated up to the end of this roll's horizon minus
    what earlier rolls / chained solves already realized -- same carried-budget pattern as
    :func:`_co2_period_cap_with_carry`.  ``co2_max_total`` is an absolute whole-horizon total in the
    LHS's units (each step's tonnes × years_represented / period_share), so the allocated-so-far share
    needs only the horizon-fraction scaling, and the realized carry must be in the same units.

    cap[g] = co2_max_total[g] · horizon_fraction_reached  −  Σ_d R_rep[g,d]
    R_rep[g,d] = Σ_{realized steps of d} tonnes · years_represented[d] / period_share[d]   (of the realizing solve)

    The constraint the chain enforces is one whole-horizon Σ_d E_d · y_d / s_d ≤ co2_max_total, so the
    periods an earlier roll / solve realized enter with the weight THAT solve's LHS gave them.  R_rep is the
    ``represented`` column of the realized carrier, built at extraction (``compute_co2_rolling_accumulator``):
    a period finished earlier need not be in this solve's ``p_period_share``.  Subtracting physical tonnes
    instead (the carry co2_max_period divides by its own share) under-counted the history by
    years_represented / period_share (8760/168 for a week-long rolling period) and let a rolling or chained
    solve emit several times the budget.

    horizon_fraction_reached (``flex_data.co2_horizon_fraction_reached``, 1.0 outside rolling / chained
    solves) is Σ_d years_represented[d]·f_d_k[d] / Σ_d years_represented[d] over every period of the
    whole chain (not just this roll's own dt) — the years_represented-weighted share of the chain's
    total represented years reached by the end of this roll.  Realized is summed over every period
    (unlike the per-period cap, this is a single whole-horizon budget) and is net of removal credits
    (#19); 0 on the first solve of a chain.  A carried row without a represented value (a carrier from
    outside the cascade without the column, or a period its realizing solve had no share for) raises.
    """
    cap = (d.group_co2_max_total.select("g").unique()
           .join(d.p_co2_max_total.frame.rename({"value": "cap"}), on="g", how="left")
           .with_columns(cap=pl.col("cap") * d.co2_horizon_fraction_reached))
    realized = getattr(d, "p_co2_cum_realized_represented", None)
    if realized is not None:
        rows = realized.frame.join(cap.select("g"), on="g", how="semi")
        missing = rows.filter(pl.col("value").is_null())
        if missing.height:
            raise ValueError(
                "co2_max_total: the realized CO2 carry has no years_represented / period_share weighted value "
                f"for {sorted(missing.select(pl.format('{},{}', 'g', 'd')).to_series().cast(pl.Utf8).to_list())}"
                "; the carried budget cannot be computed.  The carrier comes from the realizing solve's "
                "extraction (cumulative_co2 'represented' column).")
        # Fixed summation order over the periods (the semi join leaves row order unspecified).
        realized_g = rows.sort("g", "d").group_by("g", maintain_order=True).agg(pl.col("value").sum().alias("realized"))
        cap = cap.join(realized_g, on="g", how="left").with_columns(pl.col("realized").fill_null(0.0))
    else:
        cap = cap.with_columns(realized=pl.lit(0.0))
    return Param(("g",), cap.with_columns(value=pl.col("cap") - pl.col("realized")).select("g", "value"))


def _eq_co2_max_total(b: _Build) -> None:
    """CO2 emission cap over all periods, one row per scenario path under recourse."""
    if not b.has_co2_cap_total:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow
    d_leaf = getattr(d, "d_leaf", None)

    # Q16 (#52): with several scenario leaves, restrict the Σ to each leaf's own periods instead
    # of summing every forecast branch into one shared row (over-tightens the budget by about the
    # number of forecast branches).  ``terms`` spans every (g, d) of the group × the solve's whole
    # in-use domain (d_leaf's own d column, not just v_flow's: the min-load section-fuel term reads
    # v_online, whose d-domain can differ from v_flow's, #38/c18b) so no in-use period is dropped.
    # ``leaf_rows`` gates on whether v_flow itself spans >1 leaf (deterministic / single-leaf solves
    # degenerate to the legacy shape, byte-identical).
    if d_leaf is not None and _has_rows(d_leaf):
        terms = d.group_co2_max_total.select("g").unique().join(
            d_leaf.select("d").unique(), how="cross")
        is_path, over, mask = leaf_rows(terms, "g", v_flow, d_leaf)
    else:
        is_path, over, mask = False, d.group_co2_max_total, None

    def _leafify(arcs: "pl.DataFrame | None") -> "pl.DataFrame | None":
        """``arcs`` (p, source, sink, c, g) fanned to (…, d, leaf) via ``mask`` — a no-op (and no
        byte change) outside recourse."""
        if arcs is None or not is_path:
            return arcs
        return arcs.join(mask.select("g", "d", "leaf"), on="g", how="inner")

    # ─── EQ. ─── co2_max_total[_path] ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_total, d∈D(leaf),t}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c]·w_y[d,t]
    #   + Σ_{(p,s,k,c,g) ∈ flow_from_co2_capped_total_noEff, d∈D(leaf),t}
    #     v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w_y[d,t]
    #   − Σ_{(p,s,k,c,g) ∈ flow_to_co2_capped_total, d∈D(leaf),t} v_flow[p,s,k,d,t]·unitsize[p]·co2_content[c]·w_y[d,t]
    #   + Σ_{(p,n,c,g) ∈ startup_from_co2_capped_total, d∈D(leaf),t}
    #     v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]·co2_content[c]·w_y[d,t]/step_duration[d,t]
    #   ≤ co2_max_total[g] · horizon_fraction_reached  −  Σ_d R_rep[g,d]
    #   ∀ g ∈ group_co2_max_total (, leaf);   w_y = step_duration · timestep_weight / period_share · years_represented[d]
    #   R_rep[g,d] = realized tonnes of d from earlier rolls / solves · years_represented[d] / period_share[d],
    #   with the share and years of the solve that realized them (the w_y its own row gave them)
    # Caps the CO2 emissions of a group over the horizon the solve represents (tonnes): each period's
    # annualised emissions (as in co2_max_period, section fuel included) times the years the period
    # represents, summed over the periods — the measure of the model CO2 output.  years_represented is not
    # discounted.  The RHS (_co2_total_cap_with_carry) is the remaining budget as of this roll: without it,
    # each solve of a chain and each roll of a rolling solve got the FULL cap again for only its own
    # periods / window, over-allowing roughly by the number of rolls / chained solves (#38) — fixed the
    # same way the commodity ladder's cumulative tier cap was (#89/#143): horizon_fraction_reached scales
    # the fixed whole-horizon budget down to what's allocated so far, and the realized carrier (0 on the
    # first solve of a chain) — net of removal credits (#19), weighted like the LHS (R_rep, not physical
    # tonnes: a realized step counts years_represented / period_share times its tonnes, as it did in the
    # row of the solve that realized it) — is subtracted; never negative by construction.  The removal term (#19, Q14) mirrors co2_max_period's: flows into the same group's
    # node(s), unsplit, no slope; it only loosens the cap.
    # _path (Q16, #52): under recourse every forecast branch got its own copy of the whole-horizon
    # budget summed into ONE row, letting N branches spend N× the cap between them (the stochastic
    # fan is parallel futures, not N consecutive years).  One row per (g, leaf) fixes it: the LHS
    # sums only the leaf's own periods (shared pre-reveal trunk periods belong to every leaf, same
    # as the invest/divest _path families); the RHS is the SAME remaining whole-horizon budget
    # repeated per leaf (horizon_fraction_reached and the realized carry are chain-wide, not
    # per-branch quantities — the carry is kept exactly as _co2_total_cap_with_carry computes it,
    # same as the entity/group total-cap _path families reuse their own bare cap per leaf).
    lhs: dict = {}
    if b.has_co2_cap_total_eff:
        for key, term in _eff_fuel_input(b, _leafify(d.flow_from_co2_capped_total)):
            lhs[f"emissions_{key}"] = Sum(
                term
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / d.p_period_share
                * d.p_years_represented_d,
                over=("p", "source", "sink", "c", "d", "t"),
            )
    if b.has_co2_cap_total_noEff:
        lhs["emissions_noEff"] = Sum(
            Where(v_flow * d.p_unitsize, _leafify(d.flow_from_co2_capped_total_noEff))
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share
            * d.p_years_represented_d,
            over=("p", "source", "sink", "c", "d", "t"),
        )
    if _has_rows(getattr(d, "flow_to_co2_capped_total", None)):
        lhs["removal"] = -Sum(
            Where(v_flow * d.p_unitsize, _leafify(d.flow_to_co2_capped_total))
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / d.p_period_share
            * d.p_years_represented_d,
            over=("p", "source", "sink", "c", "d", "t"),
        )
    if b.has_startup_draw:
        for key, term in _startup_draw(b, _leafify(getattr(d, "startup_from_co2_capped_total", None))):
            lhs[f"emissions_startup_{key}"] = Sum(
                term * d.p_co2_content * d.p_timestep_weight / d.p_period_share * d.p_years_represented_d,
                over=("p", "n", "c", "d", "t"),
            )
    cap_scalar = _co2_total_cap_with_carry(d)
    if is_path:
        name = "co2_max_total_path"
        cap_param = Param(("g", "leaf"), over.join(cap_scalar.frame, on="g", how="left"))
    else:
        name = "co2_max_total"
        cap_param = cap_scalar
    m.add_cstr(
        name,
        over      = over,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"cap": cap_param},
    )


def _user_constraint_flow_terms(b: _Build, idx: pl.DataFrame, weight: Param | None = None) -> list:
    """The flow terms of the user constraints on the arcs ``idx`` (p, source, sink, cn), open dims (cn, d, t).

    coef[p,s,k,cn]·unitsize[p]·v_flow[p,s,k,d,t], plus on a one-variable two-way arc (``twoway1var_arcs``:
    ``no_losses_no_variable_cost`` without fork, DC power-flow lines) the same coefficient on its reverse flow
    v_flow_back[p,s,k,d,t]: a two-variable connection carries its reverse flow on its own reverse arc, which takes
    the coefficient of (c, n) too (both arcs touch n), so a connection counts by coef·(forward + reverse) whatever
    its transfer method.  ``weight`` (d, t) multiplies both (the period_total row).
    """
    terms = []
    pairs = [(b.v_flow, idx)]
    if b.v_flow_back is not None and _has_rows(b.twoway1var_arcs):
        back = join_aligned(idx, b.twoway1var_arcs.select("p", "source", "sink"),
                            on=["p", "source", "sink"], how="semi")
        if back.height > 0:
            pairs.append((b.v_flow_back, back))
    for var, rows in pairs:
        term = Where(var * b.d.p_unitsize, rows) * b.d.p_flow_constraint_coef
        terms.append(term if weight is None else term * weight)
    return terms


def _eq_user_constraints(b: _Build) -> None:
    """User-defined constraints."""
    if not b.has_user_cstr:
        return
    d = b.d
    v_invest_p, v_invest_n = b.v_invest_p, b.v_invest_n
    lookback = b.cap.prebuilt

    # ─── EQ. ─── process_constraint_{equal,less_than,greater_than}[_period] ──
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k) ∈ flow_constraint[cn]} v_flow[p,s,k,d,t]·unitsize[p]·flow_coef[p,s,k,cn]
    #   + Σ_{(p,s,k) ∈ flow_constraint[cn] ∩ twoway1var_arcs} v_flow_back[p,s,k,d,t]·unitsize[p]·flow_coef[p,s,k,cn]
    #   + Σ_n v_state[n,d,t]·state_unitsize[n]·node_state_coef[n,cn]
    #   + Σ_e inv[e,d]·invested_coef[e,cn]
    #   + Σ_e con[e,d]·in_construction_coef[e,cn]
    #   + Σ_e (v_divest[e,d] + Σ_c v_divest_built[e,c,d])·us[e]·retired_coef[e,cn]
    #   + Σ_e pre[e,d]·prebuilt_coef[e,cn]
    #   + Σ_e cap[e,d]·capacity_coef[e,cn]
    #   + Σ_{p ∈ process_online} v_online[p,d,t]·online_coef[p,cn]
    #   = / ≤ / ≥  constant[cn] (or constant[cn,d] for a period map)
    #   ∀ (cn,d,t) ∈ cdt_{eq,le,ge} with a flow, state or online coefficient   (process_constraint_*)
    #   ∀ (cn,d) of the other constraints of cdt_{eq,le,ge}             (process_constraint_*_period)
    #   cdt_le / cdt_ge leave out the (cn,d) whose constant is inf (less_than) / −inf (greater_than): no
    #   constraint in that period (_direct.apply_constraint_no_limit)
    # where, for e a process p (us = unitsize, existing = capacity = existing_count·unitsize) or a storage node n
    # (us = state_unitsize, existing = capacity, the storage capacity), MW (MWh):
    #   inv[e,d] = Σ_{(e,o,d) ∈ edd_invest_commission} v_invest[e,o]·us[e] + carried_commission[e,d]
    #   con[e,d] = Σ_{(e,o,d) ∈ edd_invest_in_construction} v_invest[e,o]·us[e] + carried_in_construction[e,d]
    #   pre[e,d] = existing[e,d] − carried_commission[e,d] + prebuilt[e,d]·us[e]
    #                            − Σ_{(e,d′,d) ∈ retired, d′ ≠ d} v_divest[e,d′]·us[e]
    #                            − (v_divested_built[e,d] − Σ_c v_divest_built[e,c,d])·us[e]
    #                            + (v_extended[e,d] − Σ_c v_extend[e,c,d])·us[e]
    #   cap[e,d] = existing[e,d] + invested[e,d]·us[e] − retired[e,d]·us[e]
    #   invested, prebuilt, retired: the capacity in service (_capacity; retired includes v_divested_built, the
    #     model-built capacity retired so far, _retire_built, less v_extended, the capacity extended beyond its
    #     end of life, _extend: capacity, never in inv or con)
    # A linear constraint authored by the user over flows (MW), storage states (MWh) and capacities.  The
    # invested term is the capacity that comes online in d (an investment decided in o with a construction
    # lead time comes online in c(o); without one c(o) = o and inv[e,d] = v_invest[e,d]·us[e]); the
    # in-construction term is the capacity decided but not yet online (zero without a lead time); the retired
    # term counts the retirements of d; pre is the capacity in service at the start of d (existing plus earlier
    # investments still alive, minus earlier retirements, also those of the same solve: a single solve and a
    # chain of solves give the same row); cap is the capacity in service in d, the measure of maxFlow and the
    # cumulative-capacity rows.  cap = pre + inv − retired, with or without a lead time.  carried_* are the
    # investments earlier solves of a chain decided with a lead time (constants; existing holds them from their
    # start of operation, so pre subtracts the part coming online in d).
    # flow_coef[p,s,k,cn] is the sum of the constraint_flow_coeff of (p, s) and (p, k): a connection's coefficient
    # at either end lands on both its arcs, so a two-variable connection counts by coef·(forward + reverse).  A
    # one-variable two-way arc (no_losses_no_variable_cost, DC power flow; signed flow = v_flow − v_flow_back) gives
    # its reverse flow v_flow_back the same coefficient: the same row for every transfer method
    # (_user_constraint_flow_terms).
    # The online term counts the units online (of unitsize) in each step; integer with binary startup, continuous
    # with linear.  v_online is the variable of the process's UC class (v_online_linear / v_online_integer).
    # Terms without data are absent.  The existing-capacity terms are constants and enter as Param terms.
    # A constraint without flow, state and online terms is the same in every step of a period, so it gets one row
    # per period (the _period family) instead of one per step.  The constraint axis is cn (c is the commodity
    # axis).  Dims: flow, state and online terms (cn, d, t); every capacity term (cn, d), broadcast over t on the
    # per-step rows.  A row whose terms all vanish in the solve (its entities have no flow, state, investment
    # or retirement there) is left out: a configuration error when the remaining existing-capacity constant
    # sense constant cannot hold, else a warning naming the constraint.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    # Row rule (coarse steps): a per-step constraint whose flow and online terms are all on coarse processes and
    # whose state terms are on coarse nodes gets one row per coarse step, process_constraint_*_coarse = the row's Δt-weighted
    # mean over the step (_coarse_flows.add_cstr); its coefficients are constant, so it is the per-step row itself
    # on one flow per step (the units online of a coarse process are one level per coarse step).  A constraint
    # with a fine flow, online or state term keeps its rows per solve step and reads the coarse flows and units
    # online at their constant value.
    flow_terms: list = []
    if _has_rows(d.flow_constraint_idx):
        flow_terms.extend(Sum(term, over=("p", "source", "sink"))
                          for term in _user_constraint_flow_terms(b, d.flow_constraint_idx))
    invest_terms: list = []
    lead_time = _lead_time.active(d)
    for axis, has_invest, var, coef in (
        ("n", b.has_invest_n, v_invest_n, d.p_node_constraint_invested_capacity_coeff),
        ("p", b.has_invest_p, v_invest_p, d.p_process_constraint_invested_capacity_coeff),
    ):
        if not has_invest or coef is None:
            continue
        unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
        if lead_time:
            # The investments coming online in d: v_invest[e,o] with c(o) = d.
            online = (_lead_time.commission_rows(d, var, axis)
                      .join(coef.frame.select(axis, "cn"), on=axis, how="inner"))
            if online.height > 0:
                invest_terms.append(Sum(
                    Where(_var_view(var, "__at_online_uc", {"d": "d_invest"}, to_axis=True) * unitsize, online)
                    * coef,
                    over=(axis, "d_invest"),
                ))
            continue
        invest_terms.append(Sum(
            Where(var * unitsize, coef.frame.select(axis, "cn")) * coef,
            over=(axis,),
        ))
    # In construction in d: v_invest[e,o] with o ≤ d < c(o).
    for axis, has_invest, var, coef in (
        ("n", b.has_invest_n, v_invest_n, d.p_node_constraint_in_construction_capacity_coeff),
        ("p", b.has_invest_p, v_invest_p, d.p_process_constraint_in_construction_capacity_coeff),
    ):
        if not has_invest or coef is None or not lead_time:
            continue
        rows = _lead_time.in_construction_rows(d, var, axis)
        if rows is None:
            continue
        building = rows.join(coef.frame.select(axis, "cn"), on=axis, how="inner")
        if building.height > 0:
            unitsize = d.p_unitsize if axis == "p" else Param(("n",), d.p_state_unitsize.frame)
            invest_terms.append(Sum(
                Where(_var_view(var, "__at_constr_uc", {"d": "d_invest"}, to_axis=True) * unitsize, building) * coef,
                over=(axis, "d_invest"),
            ))
    state_terms: list = []
    state_cns: list = []
    if b.has_storage and d.p_node_constraint_state_coeff is not None:
        coef = d.p_node_constraint_state_coeff
        state_terms.append(Sum(
            Where(b.v_state * Param(("n",), d.p_state_unitsize.frame), coef.frame.select("n", "cn")) * coef,
            over=("n",),
        ))
        state_cns = _collected(coef.frame).get_column("cn").cast(pl.Utf8).unique().to_list()

    online_terms: list = []
    online_cns: list = []
    online_by_class: list = []
    online_coef = d.p_process_constraint_online_coeff
    if b.has_online and online_coef is not None:
        coef_frame = _collected(online_coef.frame)
        for kind, _p_idx, online_set, v_online, _v_su, _v_sd in _online.uc_classes(b):
            # The coefficient covers both UC classes (Utf8 p); each v_online_* carries its own p vocabulary.
            p_dtype = _online.online_rows(b, kind).schema["p"]
            members = online_set.select(pl.col("p").cast(pl.Utf8)).unique()
            coef_k = (coef_frame.with_columns(pl.col("p").cast(pl.Utf8))
                      .join(members, on="p", how="semi")
                      .with_columns(pl.col("p").cast(p_dtype)))
            if coef_k.height == 0:
                continue
            coef_k_param = Param(("p", "cn"), coef_k)
            online_terms.append(Sum(Where(v_online, coef_k.select("p", "cn")) * coef_k_param, over=("p",)))
            online_cns.extend(coef_k.get_column("cn").cast(pl.Utf8).unique().to_list())
            online_by_class.append((v_online, coef_k))

    const_terms: list = []
    node_pre = d.p_node_constraint_prebuilt_capacity_coeff
    proc_pre = d.p_process_constraint_prebuilt_capacity_coeff
    node_cap = d.p_node_constraint_capacity_coeff
    proc_cap = d.p_process_constraint_capacity_coeff
    for coef in (node_pre, proc_pre, node_cap, proc_cap):
        existing = _user_constraint_existing(b, coef, node=coef is node_pre or coef is node_cap)
        if existing is not None:
            const_terms.append(existing)
    # Capacity earlier solves of a chain decided with a construction lead time: coming online in d it is in the
    # invested term, not in pre-built (existing carries it from d on); in construction in d it is in the
    # in-construction term.
    for coef, kind, node, sign in (
        (d.p_node_constraint_invested_capacity_coeff, "commission", True, 1.0),
        (d.p_process_constraint_invested_capacity_coeff, "commission", False, 1.0),
        (node_pre, "commission", True, -1.0),
        (proc_pre, "commission", False, -1.0),
        (d.p_node_constraint_in_construction_capacity_coeff, "in_construction", True, 1.0),
        (d.p_process_constraint_in_construction_capacity_coeff, "in_construction", False, 1.0),
    ):
        carried = _user_constraint_carried(d, coef, kind, node=node, sign=sign)
        if carried is not None:
            const_terms.append(carried)

    capacity_terms: list = []
    if b.has_invest_p and proc_pre is not None and _has_rows(lookback):
        invested = _var_view(v_invest_p, "__at_back_uc", {"d": "d_invest"}, to_axis=True)
        earlier = (lookback.pipe(rename_to_axis, {"e": "p"})
                   .join(proc_pre.frame.select("p", "cn"), on="p", how="inner"))
        if earlier.height > 0:
            capacity_terms.append(Sum(
                Where(invested * d.p_unitsize, earlier) * proc_pre,
                over=("p", "d_invest"),
            ))
    if b.has_invest_n and node_pre is not None and _has_rows(lookback):
        invested = _var_view(v_invest_n, "__at_back_uc", {"d": "d_invest"}, to_axis=True)
        earlier = (lookback.pipe(rename_to_axis, {"e": "n"})
                   .join(node_pre.frame.select("n", "cn"), on="n", how="inner"))
        if earlier.height > 0:
            capacity_terms.append(Sum(
                Where(invested * Param(("n",), d.p_state_unitsize.frame), earlier) * node_pre,
                over=("n", "d_invest"),
            ))
    netted: set[str] = set()
    for axis in ("p", "n"):
        raw_unitsize = d.p_unitsize if axis == "p" else d.p_state_unitsize
        if raw_unitsize is None:
            continue
        unitsize = raw_unitsize if axis == "p" else Param(("n",), raw_unitsize.frame)
        has_invest, has_divest = ((b.has_invest_p, b.has_divest_p) if axis == "p"
                                  else (b.has_invest_n, b.has_divest_n))
        invest, divest = (v_invest_p, b.v_divest_p) if axis == "p" else (v_invest_n, b.v_divest_n)
        active = b.cap.retired_frame(axis)
        pre, cap = (proc_pre, proc_cap) if axis == "p" else (node_pre, node_cap)
        retired = (d.p_process_constraint_retired_capacity_coeff if axis == "p"
                   else d.p_node_constraint_retired_capacity_coeff)
        # Pre-built: the retirements of the earlier periods.
        if has_divest and pre is not None and _has_rows(active):
            earlier = (active.filter(pl.col("d_divest").cast(pl.Utf8) != pl.col("d").cast(pl.Utf8))
                       .join(pre.frame.select(axis, "cn"), on=axis, how="inner"))
            if earlier.height > 0:
                view = _var_view(divest, "__at_div_pre_uc", {"d": "d_divest"}, to_axis=True)
                capacity_terms.append(-Sum(Where(view * unitsize, earlier) * pre, over=(axis, "d_divest")))
                netted.update(earlier.get_column("cn").cast(pl.Utf8).to_list())
        # Capacity in service: the investments alive in d and every retirement up to d.
        if has_invest and cap is not None and _has_rows(b.cap.alive):
            alive = (_entity_rows(b.cap.alive, "e", _collected(cap.frame), axis)
                     .join(cap.frame.select(axis, "cn"), on=axis, how="inner"))
            if alive.height > 0:
                view = _var_view(invest, "__at_inv_cap_uc", {"d": "d_invest"}, to_axis=True)
                capacity_terms.append(Sum(Where(view * unitsize, alive) * cap, over=(axis, "d_invest")))
        if has_divest and cap is not None and _has_rows(active):
            gone = active.join(cap.frame.select(axis, "cn"), on=axis, how="inner")
            if gone.height > 0:
                view = _var_view(divest, "__at_div_cap_uc", {"d": "d_divest"}, to_axis=True)
                capacity_terms.append(-Sum(Where(view * unitsize, gone) * cap, over=(axis, "d_divest")))
        # Retired in d.
        if has_divest and retired is not None:
            capacity_terms.append(Sum(
                Where(divest * unitsize, retired.frame.select(axis, "cn")) * retired,
                over=(axis,),
            ))
        capacity_terms.extend(_user_constraint_built_terms(b, axis, unitsize, pre, cap, retired))
        capacity_terms.extend(_user_constraint_extended_terms(b, axis, unitsize, pre, cap))
    if netted:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("user_constraint_prebuilt_nets_retirements", "user constraints", sorted(netted),
                                  detail="Results change only if the model retires capacity.")

    # Term order of the per-step rows: flow, invested, state, pre-built, the newer capacity terms, then online.
    step_terms = flow_terms + invest_terms + state_terms + capacity_terms + online_terms
    period_terms = invest_terms + capacity_terms
    op_cns = set(state_cns) | set(online_cns)
    if flow_terms:
        op_cns.update(d.flow_constraint_idx.get_column("cn").cast(pl.Utf8).unique().to_list())
    # The (cn, d) that keep a variable in this solve; a row elsewhere would read constant sense constant.
    live_period = _user_constraint_live_keys(period_terms)
    live_step = live_period | _user_constraint_live_keys(flow_terms + state_terms + online_terms)
    empty_rows: list[tuple[pl.DataFrame, str]] = []
    # The constant is moved to the right-hand side as constant − Σ const_terms.  A (cn,) constant summed with a
    # (cn, d) existing-capacity term would keep d empty for every constraint without such a term, and those rows
    # would read 0 instead of their constant: give the constant the (cn, d) of the rows first.
    rhs_constant = d.p_constraint_constant
    row_periods = [_collected(o).select("d") for o in (d.cdt_eq, d.cdt_le, d.cdt_ge) if _has_rows(o)]
    if const_terms and row_periods and rhs_constant is not None and "d" not in rhs_constant.dims:
        periods = pl.concat(row_periods).unique(maintain_order=True)
        rhs_constant = Param(("cn", "d"), _collected(rhs_constant.frame).join(periods, how="cross")
                             .select("cn", "d", "value"))
    uc_members = None
    if b.has_coarse_proc:
        from flextool.engine.preprocess.flex_data import coarse_row_members

        uc_members = coarse_row_members(d, "user_constraint")
    # constraint_period_method period_total: its constraints get the one annualised row per period below instead.
    period_total = d.constraint_period_total
    pt_cns = (sorted(period_total.get_column("cn").cast(pl.Utf8).unique().to_list())
              if period_total is not None and period_total.height > 0 else [])
    for over, sense, name in (
        (d.cdt_eq, "==", "process_constraint_equal"),
        (d.cdt_le, "<=", "process_constraint_less_than"),
        (d.cdt_ge, ">=", "process_constraint_greater_than"),
    ):
        if not _has_rows(over):
            continue
        if pt_cns:
            over = over.filter(~pl.col("cn").cast(pl.Utf8).is_in(pt_cns))
            if over.height == 0:
                continue
        is_op = pl.col("cn").cast(pl.Utf8).is_in(sorted(op_cns))
        for rows, terms, live, family in ((over.filter(is_op), step_terms, live_step, name),
                                          (over.filter(~is_op).select("cn", "d").unique(maintain_order=True),
                                           period_terms, live_period, name + "_period")):
            if rows.height == 0:
                continue
            has_var = (pl.repeat(True, rows.height, eager=True) if "*" in live
                       else _cn_d_key(rows).is_in(sorted(live)))
            empty_rows.append((rows.filter(~has_var).select("cn", "d"), sense))
            rows = rows.filter(has_var)
            if rows.height == 0:
                continue
            lhs: dict = {"contribution": _sum_terms(terms)}
            for i, const in enumerate(const_terms):
                lhs[f"prebuilt_const_{i}"] = const
            _coarse_flows.add_row_rule_cstr(
                b, family,
                over      = rows,
                sense     = sense,
                lhs_terms = lhs,
                rhs_terms = {"constant": rhs_constant},
                partition = (None if uc_members is None or "t" not in rows.columns
                             else _coarse_flows.row_partition(b, uc_members, rows)),
            )
    _check_user_constraints_without_terms(b, empty_rows, const_terms)
    if pt_cns:
        _eq_user_constraints_period_total(b, pt_cns, invest_terms + capacity_terms, online_by_class, const_terms,
                                          rhs_constant)


def _user_constraint_built_terms(b: _Build, axis: str, unitsize: Param, pre: "Param | None", cap: "Param | None",
                                 retired: "Param | None") -> list:
    """The model-built retirement terms of the user constraints on ``axis`` (_retire_built), dims (cn, d):

        pre:     − (v_divested_built[e,d] − Σ_c v_divest_built[e,c,d])·us[e]·prebuilt_coef[e,cn]
        cap:     − v_divested_built[e,d]·us[e]·capacity_coef[e,cn]
        retired: + Σ_c v_divest_built[e,c,d]·us[e]·retired_coef[e,cn]

    the retirements decided before d that still count there, every retirement counting in d, and those decided
    in d (the model-built part of retired[e,d] and of the retirements of d)."""
    built = _retire_built.retired(b, axis)
    if built is None:
        return []
    decided = _retire_built.decided(b, axis)
    counted, options = _retire_built.retired_frame(b, axis), _retire_built.decided_frame(b, axis)
    terms: list = []
    for coef, sign, with_decided in ((pre, -1.0, True), (cap, -1.0, False)):
        if coef is None:
            continue
        rows = counted.join(coef.frame.select(axis, "cn"), on=axis, how="inner")
        if rows.height == 0:
            continue
        terms.append(Sum(Where(built * unitsize, rows) * coef, over=(axis,)) * sign)
        if with_decided:
            now = options.join(coef.frame.select(axis, "cn"), on=axis, how="inner")
            if now.height > 0:
                terms.append(Sum(Where(decided * unitsize, now) * coef, over=(axis,)))
    if retired is not None:
        now = options.join(retired.frame.select(axis, "cn"), on=axis, how="inner")
        if now.height > 0:
            terms.append(Sum(Where(decided * unitsize, now) * retired, over=(axis,)))
    return terms


def _user_constraint_extended_terms(b: _Build, axis: str, unitsize: Param, pre: "Param | None",
                                    cap: "Param | None") -> list:
    """The life-extension terms of the user constraints on ``axis`` (_extend), dims (cn, d):

        pre: + (v_extended[e,d] − Σ_c v_extend[e,c,d])·us[e]·prebuilt_coef[e,cn]
        cap: + v_extended[e,d]·us[e]·capacity_coef[e,cn]

    the extended capacity in service from the start of d (an extension decided in d starts to operate there) and
    every extension in service in d.  An extension is capacity, never an investment: it has no term with
    invested_coef or in_construction_coef."""
    ext = _extend.extended(b, axis)
    if ext is None:
        return []
    decided = _extend.decided(b, axis)
    counted, options = _extend.extended_frame(b, axis), _extend.decided_frame(b, axis)
    terms: list = []
    for coef, with_decided in ((pre, True), (cap, False)):
        if coef is None:
            continue
        rows = counted.join(coef.frame.select(axis, "cn"), on=axis, how="inner")
        if rows.height == 0:
            continue
        terms.append(Sum(Where(ext * unitsize, rows) * coef, over=(axis,)))
        if with_decided:
            now = options.join(coef.frame.select(axis, "cn"), on=axis, how="inner")
            if now.height > 0:
                terms.append(-Sum(Where(decided * unitsize, now) * coef, over=(axis,)))
    return terms


def _eq_user_constraints_period_total(b: _Build, pt_cns: list[str], capacity_terms: list, online_by_class: list,
                                      const_terms: list, rhs_constant: "Param | None") -> None:
    """The rows of the user constraints with ``constraint_period_method`` ``period_total``: one per (cn, d)."""
    d = b.d

    # ─── EQ. ─── process_constraint_{equal,less_than,greater_than}_period_total ──
    # ─────────────────────────────────────────────────────────────────────────────
    # Σ_t w[d,t]·( Σ_{(p,s,k) ∈ flow_constraint[cn]} v_flow[p,s,k,d,t]·unitsize[p]·flow_coef[p,s,k,cn]
    #              + Σ_{(p,s,k) ∈ flow_constraint[cn] ∩ twoway1var_arcs} v_flow_back[p,s,k,d,t]·unitsize[p]
    #                                                                   ·flow_coef[p,s,k,cn]
    #              + Σ_{p ∈ process_online} v_online[p,d,t]·online_coef[p,cn] )
    #   + H[d]·( Σ_e inv[e,d]·invested_coef[e,cn] + Σ_e con[e,d]·in_construction_coef[e,cn]
    #            + Σ_e v_divest[e,d]·us[e]·retired_coef[e,cn] + Σ_e pre[e,d]·prebuilt_coef[e,cn]
    #            + Σ_e cap[e,d]·capacity_coef[e,cn] )
    #   = / ≤ / ≥  constant[cn] (or constant[cn,d] for a period map)
    #   ∀ (cn,d) ∈ cdt_{eq,le,ge}, cn ∈ constraint_period_total, window_fraction[d] = 1
    # w[d,t] = step_duration[d,t]·timestep_weight[d,t] / period_share[d];   H[d] = Σ_t w[d,t]
    # inv, con, pre, cap, us, the v_flow_back term and the existing / carried constants as in process_constraint_*
    # (_eq_user_constraints).
    # One row per period over the period's time-integrated, annualised terms (per year: flows in MWh, units online
    # in unit-hours, the constant in the constraint's unit per year), the weighting of co2_max_period.  H is the
    # number of hours the period's modelled steps stand for in a year (8760 for a full year, also with
    # representative periods, whose weights are normalised); computed, not 8760, so a roll window or a varying step
    # duration gets its own.  Every capacity term, and the existing / carried capacity constants, are multiplied by
    # H: each term is its per-step quantity integrated over the period, so a coefficient keeps its per-step meaning
    # (flow − 0.3·capacity ≤ 0 caps the annual capacity factor at 0.3).  State terms are rejected by the reader
    # (a state summed over steps is not an energy).  A row needs the whole period in the solve: a period that the
    # solve holds in part (window_fraction < 1, the look-ahead tail of a roll) has no row; a rolling solve whose
    # rolls do not each realize whole periods has none at all (user_constraint_period_total_off, the run's
    # pre-solve check names the constraints and the solve).  Stochastic branch periods are their own d (no
    # probability weight: each forecast meets the limit).  Coarse flows and units online enter through their alias
    # on the fine steps (exact: Σ_t v·Δt).  Dims: flow and online terms (cn, d, t) summed over t; capacity terms
    # and constants (cn, d).
    if d.user_constraint_period_total_off:
        return
    rows_by_sense = [(rows.filter(pl.col("cn").cast(pl.Utf8).is_in(pt_cns)).select("cn", "d")
                      .unique(maintain_order=True), sense, name)
                     for rows, sense, name in ((d.cdt_eq, "==", "process_constraint_equal_period_total"),
                                               (d.cdt_le, "<=", "process_constraint_less_than_period_total"),
                                               (d.cdt_ge, ">=", "process_constraint_greater_than_period_total"))
                     if _has_rows(rows)]
    rows_by_sense = [(rows, sense, name) for rows, sense, name in rows_by_sense if rows.height > 0]
    if not rows_by_sense:
        return
    feature = "constraint_period_method period_total"
    _check_covers_dt(d, "p_period_share", feature, positive=True)
    _check_covers_dt(d, "p_timestep_weight", feature)
    _check_covers_dt(d, "p_step_duration", feature)
    w = d.p_step_duration * d.p_timestep_weight / d.p_period_share
    w_frame = _collected(w.frame).sort("d", "t")
    hours = Param(("d",), w_frame.group_by("d", maintain_order=True).agg(pl.col("value").sum()))
    # The periods the solve holds whole (window_fraction 1); the others have no row.
    whole = None
    if d.p_window_fraction is not None:
        frac = _collected(d.p_window_fraction.frame)
        over = frac.filter(pl.col("value").cast(pl.Float64) > 1.0 + 1e-9)
        if over.height > 0:
            raise ValueError(f"{feature}: window_fraction above 1 for period(s) "
                             f"{sorted(over.get_column('d').cast(pl.Utf8).unique().to_list())}: the solve's "
                             "period_share and its step durations disagree (the row would be over-annualised)")
        whole = (frac.filter(pl.col("value").cast(pl.Float64) >= 1.0 - 1e-9)
                 .get_column("d").cast(pl.Utf8).to_list())

    terms: list = []
    if _has_rows(d.flow_constraint_idx):
        idx = d.flow_constraint_idx.filter(pl.col("cn").cast(pl.Utf8).is_in(pt_cns))
        if idx.height > 0:
            terms.extend(Sum(term, over=("p", "source", "sink", "t"))
                         for term in _user_constraint_flow_terms(b, idx, w))
    if capacity_terms:
        terms.append(_sum_terms(capacity_terms) * hours)
    for v_online, coef_k in online_by_class:
        coef_pt = coef_k.filter(pl.col("cn").cast(pl.Utf8).is_in(pt_cns))
        if coef_pt.height == 0:
            continue
        terms.append(Sum(Where(v_online, coef_pt.select("p", "cn")) * Param(("p", "cn"), coef_pt) * w,
                         over=("p", "t")))
    consts = [const * hours for const in const_terms]
    live = _user_constraint_live_keys(terms)
    empty_rows: list[tuple[pl.DataFrame, str]] = []
    for rows, sense, name in rows_by_sense:
        if whole is not None:
            rows = rows.filter(pl.col("d").cast(pl.Utf8).is_in(whole))
        has_var = (pl.repeat(True, rows.height, eager=True) if "*" in live
                   else _cn_d_key(rows).is_in(sorted(live)))
        empty_rows.append((rows.filter(~has_var), sense))
        rows = rows.filter(has_var)
        if rows.height == 0:
            continue
        lhs: dict = {"contribution": _sum_terms(terms)}
        for i, const in enumerate(consts):
            lhs[f"prebuilt_const_{i}"] = const
        b.m.add_cstr(
            name,
            over      = rows,
            sense     = sense,
            lhs_terms = lhs,
            rhs_terms = {"constant": rhs_constant},
        )
    _check_user_constraints_without_terms(b, empty_rows, consts)


def _cn_d_key(frame: pl.DataFrame) -> pl.Series:
    """``cn`` and ``d`` of each row joined into one Utf8 key (for membership tests across Enum dtypes)."""
    return frame.select(pl.concat_str([pl.col("cn").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)],
                                      separator="\x1f")).to_series()


def _user_constraint_live_keys(terms: list) -> set[str]:
    """The :func:`_cn_d_key` keys of the (cn, d) where a term of *terms* has a non-zero variable coefficient.

    ``{"*"}`` (every row has a variable) when a term cannot be reduced to (cn, d) here.
    """
    keys: set[str] = set()
    for expr in terms:
        for term in expr.terms:
            if "cn" not in term.dims or "d" not in term.dims or term.where_frames or term.where_map_frames:
                # Not reducible to (cn, d) here: count every row of the family as having a variable.
                return {"*"}
            frame = (term.lazy.filter(pl.col("coef") != 0.0).select("cn", "d").unique().collect())
            keys.update(_cn_d_key(frame).to_list())
    return keys


def _check_user_constraints_without_terms(b: _Build, empty_rows: list[tuple[pl.DataFrame, str]],
                                          const_terms: list) -> None:
    """User-constraint rows left without a variable in this solve: an error where ``constant sense constant``
    cannot hold, else warning W-0004 (one line per solve, each constraint once per run) — the row is left out
    either way."""
    d = b.d
    empty = [(rows, sense) for rows, sense in empty_rows if rows.height > 0]
    if not empty:
        return
    keys = ["cn", "d"]

    def by_cn_d(param: "Param | None", alias: str) -> pl.DataFrame | None:
        if param is None:
            return None
        frame = _collected(param.frame).with_columns(pl.col(c).cast(pl.Utf8) for c in param.dims)
        return frame.rename({"value": alias}).select(*param.dims, alias)

    rhs = by_cn_d(d.p_constraint_constant, "rhs")
    consts = [f for f in (by_cn_d(c, f"lhs_{i}") for i, c in enumerate(const_terms)) if f is not None]
    failing: list[str] = []
    holding: dict[str, list[str]] = {}
    shown = {"==": "=", "<=": "<=", ">=": ">="}
    for rows, sense in empty:
        frame = rows.with_columns(pl.col(c).cast(pl.Utf8) for c in keys).unique(maintain_order=True)
        if rhs is not None:
            frame = frame.join(rhs, on=[c for c in rhs.columns if c in keys], how="left")
        else:
            frame = frame.with_columns(rhs=pl.lit(0.0))
        lhs_cols = []
        for const in consts:
            frame = frame.join(const, on=[c for c in const.columns if c in keys], how="left")
            lhs_cols.append(const.columns[-1])
        frame = frame.with_columns(
            rhs=pl.col("rhs").fill_null(0.0),
            lhs=pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in lhs_cols]) if lhs_cols else pl.lit(0.0))
        for row in frame.select("cn", "d", "lhs", "rhs").iter_rows(named=True):
            lhs, rhs_value = float(row["lhs"]), float(row["rhs"])
            tol = 1e-9 * max(1.0, abs(lhs), abs(rhs_value))
            holds = {"==": abs(lhs - rhs_value) <= tol, "<=": lhs <= rhs_value + tol,
                     ">=": lhs >= rhs_value - tol}[sense]
            text = f"{lhs:g} {shown[sense]} {rhs_value:g}"
            if holds:
                holding.setdefault(row["cn"], []).append(f"period {row['d']}: {text}")
            else:
                failing.append(f"{row['cn']} (period {row['d']}: {text})")
    where = f"solve '{b.solve_name}'" if b.solve_name else "this solve"
    if failing:
        raise FlexToolConfigError(
            f"User constraints without any variable in {where} cannot hold: {', '.join(failing)}. Their "
            "coefficients only refer to flows, storage states, units online, investments or retirements that this "
            "solve does not have. Remove the constraint from the solve, or give it a term that the solve has.")
    if holding:
        from flextool import run_warnings
        run_warnings.warn_grouped("W-0004", sorted(holding), solve=b.solve_name or "this solve", per_run=True,
                                  details={cn: "; ".join(cases) for cn, cases in sorted(holding.items())})


def _collected(frame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _user_constraint_carried(d, coef: "Param | None", kind: str, *, node: bool, sign: float) -> "Param | None":
    """``sign · Σ_e carried[e,d]·coef[e,cn]`` as a ``(cn, d)`` Param: the capacity earlier solves of a chain decided
    with a construction lead time that comes online (``kind="commission"``) or is in construction
    (``kind="in_construction"``) in d; None without rows."""
    if coef is None:
        return None
    axis = "n" if node else "p"
    carried = _lead_time.carried(d, kind, axis)
    if carried is None:
        return None
    carried, coef_frame = align_join_dtypes(carried, coef.frame.rename({"value": "coef"}), [axis])
    frame = (carried.join(coef_frame, on=axis, how="inner")
             .with_columns(value=pl.col("value") * pl.col("coef") * sign))
    frame = _collected(frame.sort("cn", "d", axis).group_by(["cn", "d"], maintain_order=True)
                       .agg(pl.col("value").sum()).select("cn", "d", "value"))
    return Param(("cn", "d"), frame) if frame.height > 0 else None


def _user_constraint_existing(b: _Build, coef: "Param | None", *, node: bool) -> "Param | None":
    """``Σ_e existing[e,d]·coef[e,cn]`` as a ``(cn, d)`` Param (MW, MWh for nodes); None without rows."""
    if coef is None:
        return None
    d, cap = b.d, b.cap
    if node:
        if cap.existing_state_n is None:
            return None
        frame = (cap.existing_state_n.frame
                 .join(coef.frame.rename({"value": "coef"}), on="n", how="inner")
                 .with_columns(value=pl.col("value") * pl.col("coef")))
    else:
        if cap.existing_count_p is None or d.p_unitsize is None:
            return None
        frame = (cap.existing_count_p.frame.rename({"value": "ec"})
                 .join(d.p_unitsize.frame.rename({"value": "us"}), on="p", how="inner")
                 .with_columns(cap=pl.col("ec") * pl.col("us"))
                 .join(coef.frame.rename({"value": "coef"}), on="p", how="inner")
                 .with_columns(value=pl.col("cap") * pl.col("coef")))
    # Fixed summation order over the members (the joins leave row order unspecified).
    frame = _collected(frame.sort("cn", "d", "n" if node else "p").group_by(["cn", "d"], maintain_order=True)
                       .agg(pl.col("value").sum()).select("cn", "d", "value"))
    return Param(("cn", "d"), frame) if frame.height > 0 else None


def _eq_max_state(b: _Build) -> None:
    """Capacity limit of the storage state."""
    if not b.has_storage:
        return
    m, d = b.m, b.d

    # ─── EQ. ─── maxState ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] + retired[n,d]
    #   − invested[n,d]  ≤  state_upper[n,d]
    #   ∀ (n,d,t) ∈ nodeState_dt
    # Caps the storage state by the built storage capacity.  state_upper = capacity / state_unitsize, so
    # every term is in state units.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    lhs: dict = {"state": b.v_state}
    lhs.update(_capacity.node_capacity_change_terms(b, d.nodeState, ""))
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
    # v_state[n,d,t] − profile[f,d,t]·av[n,d,t]·invested[n,d]
    #   + profile[f,d,t]·av[n,d,t]·retired[n,d]
    #   ≤ / ≥ / =  profile[f,d,t] · av[n,d,t] · capacity[n,d] / state_unitsize[n]
    #   ∀ (n,f) ∈ node_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds the storage state by a profile times the built storage capacity, in state units.
    # av = node availability, 1 for nodes without availability data.  The invest/divest terms are
    # restricted to the (n,f) pairs of the method's set.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    state_unitsize = d.p_state_unitsize.frame.rename({"value": "us"})
    existing_count = Param(("n", "d"), b.cap.existing_state_n.frame.rename({"value": "cap"})
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
        invested, divested = _capacity.node_in_service(b, d.nodeState, "_node_profile")
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


def _roll_end_rows(d, kinds) -> "pl.DataFrame | None":
    """``(n, value)`` of the storage_roll_end rows of the ``kinds`` (_run_model._anchor_cyclic_in_rolling); None
    when there are none."""
    sre = getattr(d, "storage_roll_end", None)
    if not _has_rows(sre):
        return None
    out = sre.filter(pl.col("kind").is_in(list(kinds))).select("n", "value")
    return out if out.height > 0 else None


def _cyclic_pin_coefficient(b: _Build) -> "Param | None":
    """``(n)``: the share of the capacity a cyclic node's first-roll wrap is pinned at: storage_state_start of the
    fix_start nodes, storage_state_end of the ``wrap_end`` nodes of storage_roll_end (a fix_end node of a rolling
    instance that reaches the model end, without fix_start).  None when neither."""
    d = b.d
    parts = []
    if _has_rows(d.storage_fix_start) and d.p_state_start is not None:
        parts.append(d.p_state_start.frame.join(d.storage_fix_start.select("n"), on="n", how="semi")
                     .select("n", "value"))
    wrap_end = _roll_end_rows(d, ("wrap_end",))
    if wrap_end is not None and d.p_state_end is not None:
        end = d.p_state_end.frame.join(wrap_end.select("n"), on="n", how="semi").select("n", "value")
        if parts:
            end = end.join(parts[0].select("n"), on="n", how="anti")
        parts.append(end)
    parts = [f for f in parts if f.height > 0]
    if not parts:
        return None
    return Param(("n",), pl.concat(parts, how="vertical"))


def _cyclic_start_pin_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the steps storage_state_start_binding_cyclic_period pins (the wrap targets of the cyclic
    fix_start nodes' first steps, and of the ``wrap_end`` nodes of storage_roll_end, less the steps an upper-level
    solve fixes, which are logged); None when there are none."""
    d = b.d
    coef = _cyclic_pin_coefficient(b)
    if not (b.has_storage
            and coef is not None
            and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None
            and _has_rows(d.nodeState_first_dt)):
        return None
    cyclic = _self_discharge.node_union((d.storage_bind_within_timeblock, d.storage_bind_within_solve,
                                          d.storage_bind_within_period, _self_discharge.block_cyclic_nodes(d)))
    if cyclic is None:
        return None
    nodes = cyclic.join(coef.frame.select(pl.col("n").cast(cyclic.schema["n"], strict=False)), on="n", how="inner")
    nodes = _self_discharge.without_nodes(nodes, _self_discharge.carried_nodes(d))
    if nodes is None:
        return None
    # The step each node's first step starts from: its wrap under the node's own lag (the lags of the state
    # change, _self_discharge.state_links).
    first = d.nodeState_first_dt.join(nodes, on="n", how="inner").select(
        pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    links = _self_discharge.previous_steps(_self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb))
    wrap = (first.join(links, on=["n", "d", "t"], how="inner")
            .select("n", pl.col("d_prev").alias("d"), pl.col("t_prev").alias("t")).unique())
    fixed = _fix_quantity_target(b)
    if fixed is not None:
        # The wrap target is the step an upper-level solve fixes (a nested roll whose horizon is one timeblock, or
        # the solve for bind_within_solve): the two equalities would sit on one variable, and the upper value wins.
        fixed = fixed.select(pl.col("n").cast(pl.Utf8).alias("_n"), pl.col("d").cast(pl.Utf8).alias("_d"),
                             pl.col("t").cast(pl.Utf8).alias("_t"))
        clash = wrap.rename({"n": "_n", "d": "_d", "t": "_t"}).join(fixed, on=["_n", "_d", "_t"], how="semi")
        if clash.height > 0:
            _LOG.info("storage fix_start not pinned where the upper-level solve fixes the state at the same step: %s",
                      ", ".join(f"{n} {dd} {tt}" for n, dd, tt in clash.sort("_n", "_d", "_t").iter_rows()))
            wrap = wrap.join(clash.rename({"_n": "n", "_d": "d", "_t": "t"}), on=["n", "d", "t"], how="anti")
    over = (b.nodeState_dt
            .with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(wrap.rename({"n": "_n", "d": "_d", "t": "_t"}), on=["_n", "_d", "_t"], how="semi")
            .select("n", "d", "t").unique(maintain_order=True))
    return over if over.height > 0 else None


def _eq_storage_start_binding_cyclic(b: _Build) -> None:
    """Start state of fix_start storage nodes whose first step wraps to a later step of the solve."""
    d = b.d
    over = _cyclic_start_pin_steps(b)
    if over is None:
        return
    m = b.m

    # ─── EQ. ─── storage_state_start_binding_cyclic_period ───────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d′,t′] − state_start[n]·invested[n,d′]
    #   + state_start[n]·retired[n,d′]
    #   = state_start[n] · capacity[n,d′] / state_unitsize[n]
    #   ∀ n ∈ (storage_fix_start ∪ storage_roll_end(wrap_end)) ∩ (bind_within_timeblock ∪ bind_within_solve
    #     ∪ bind_within_period ∪ looping nodeStateBlock), n ∉ carried, (n,d,t) ∈ nodeState_first_dt,
    #     (d′,t′) = the step (d,t) starts from (the node's wrap), (n,d′,t′) ∉ the steps
    #     node_balance_fix_quantity_eq_lower fixes
    # The first step of a cyclic node starts from its wrap target, so the start state is pinned there, to the
    # share state_start of the built storage capacity (state units): the last step of the first timeblock
    # (bind_within_timeblock), of the solve (bind_within_solve) or of the first period (bind_within_period, also
    # bind_within_timeblock in a rolling solve, and bind_intraperiod_blocks; in a roll, of its part in the roll).  Only where no state of the node was
    # realised before this solve (the first solve of the model and of each nested level); a later solve neither
    # pins nor carries a cyclic node (it cycles within the solve; a later roll of a rolling bind_within_solve node
    # chains forward from the carry instead, _run_model._anchor_cyclic_in_rolling).  Where an
    # upper-level solve fixes the state at the wrap target the upper value wins (the two equalities would sit on
    # one variable).  A bind_intraperiod_blocks node that chains forward (fix_start_end, or a roll that holds its
    # first period only in part) has its start in nodeBalanceBlock_eq (_fix_start_terms); fix_start_end nodes of the
    # other cyclic methods are forward-only in the solve (_run_model._fix_start_end_chains_forward).
    # In a rolling instance that reaches the model end, a fix_end node without fix_start is pinned the same way at
    # storage_state_end (state_start below stands for that share; storage_roll_end kind wrap_end): its cycle level
    # is the end value, as in a solve that is not rolling, where the end pin sits on the wrap target.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    coef = _cyclic_pin_coefficient(b)
    lhs: dict = {"state": b.v_state}
    invested, divested = _capacity.node_in_service(b, d.nodeState, "_cyclic", coef=coef)
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_start_binding_cyclic_period",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": (coef * b.cap.existing_state_n) / d.p_state_unitsize},
    )


def _utf8_cells(frame: "pl.DataFrame | None", cols=("n", "d", "t")) -> pl.DataFrame:
    """``frame``'s ``cols`` as Utf8 (an empty frame for None)."""
    if frame is None:
        return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
    return frame.select([pl.col(c).cast(pl.Utf8) for c in cols])


def _state_cells_typed(b: _Build, frame: pl.DataFrame, rename: dict) -> pl.DataFrame:
    """Utf8 ``frame`` cast onto v_state's dtypes; the columns of ``rename`` ({column: v_state dim}) take that dim's
    dtype."""
    schema = b.v_state.frame.schema
    return frame.with_columns([pl.col(c).cast(schema[rename.get(c, c)], strict=False) for c in frame.columns])


def _state_keys_typed(b: _Build, frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` (n, d, t, value) with its keys cast onto v_state's dtypes."""
    schema = b.v_state.frame.schema
    return frame.with_columns([pl.col(c).cast(schema[c], strict=False) for c in ("n", "d", "t")])


def _notice_nodes(change_id: str, nodes, b: "_Build | None" = None) -> None:
    """The behaviour-change notice of ``change_id`` for the nodes of ``nodes`` (a frame with n, or names): one
    printed line per change and complete solve of ``b`` (how many nodes and the first; owner rule), the names in the
    log at debug level, however many rows of the build name the change."""
    if nodes is None:
        return
    names = nodes["n"].to_list() if isinstance(nodes, pl.DataFrame) else list(nodes)
    if names:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice(change_id, "nodes", sorted(set(str(n) for n in names)),
                                  solve=(b.solve_name if b is not None else None) or "")


def _notice_stochastic_storage_paths(names) -> None:
    from flextool.engine import _behaviour_changes
    _behaviour_changes.notice("stochastic_storage_paths_tied", "nodes", sorted(set(str(n) for n in names)))


def _eq_storage_within_solve_path_end(b: _Build) -> None:
    """bind_within_solve in a stochastic solve: the last step of every path ends at the same level."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_bind_within_solve)
            and _has_rows(d.period_last)
            and _has_rows(d.nodeState_first_dt)
            and _has_rows(d.nodeState_last_dt)):
        return
    last_periods = sorted(set(str(x) for x in d.period_last["d"].to_list()))
    if len(last_periods) < 2:
        return
    m = b.m

    # ─── EQ. ─── storage_within_solve_path_end ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t_last(d)] − v_state[n,d*,t_last(d*)] = 0
    #   ∀ n ∈ bind_within_solve, d ∈ period_last, d ≠ d*, (n,d,t_last(d)) and (n,d*,t_last(d*)) ∈ nodeState_last_dt,
    #     unless both ends are already set (each is a step storage_state_start_binding_cyclic_period pins or
    #     node_balance_fix_quantity_eq_lower fixes)
    #   d* = the last period the solve's first steps wrap to (the state-change lag of bind_within_solve,
    #     _self_discharge.state_links): the last period key of the solve, the trunk's wrap target
    # bind_within_solve closes one loop per path of a stochastic solve.  The first step of the solve continues from
    # the last step of d* only, so this row ends every other path (period_last: the solve's last period and its
    # forecast branch siblings) at the same level: every branch's end is the level the trunk starts from.  With a
    # fan in the solve's first period each root forecast's first step continues from the last step of its own path
    # (dtttdt_within_solve), so equal ends also give all the forecasts the same start.  A deterministic solve has
    # one last period and no rows.  A fix_start node keeps the rows where only one end is pinned (a fan after the
    # trunk period: the pin sits at d*'s end); with both ends pinned or fixed the row would only repeat (or
    # contradict) those values.
    ws_links = [link for link in _self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb)
                if link.key == "state_change_ws"]
    if not ws_links:
        return
    wraps = (_utf8_cells(d.nodeState_first_dt)
             .join(_self_discharge.previous_steps(ws_links), on=["n", "d", "t"], how="inner"))
    targets = sorted(set(wraps["d_prev"].to_list()) & set(last_periods))
    if not targets:
        return
    # Convention: d* is the last wrap target in sorted name order (the trunk wraps to the last period key; a root
    # of a first-period fan to the leaf its dtttdt_within_solve picks, also by sorted name, _derived.leaf_of).
    d_star = targets[-1]
    ends = (_utf8_cells(d.nodeState_last_dt)
            .join(_utf8_cells(d.storage_bind_within_solve, ("n",)), on="n", how="semi")
            .filter(pl.col("d").is_in(last_periods)))
    star_ends = ends.filter(pl.col("d") == d_star).select("n", d_end="d", t_end="t")
    over = (ends.filter(pl.col("d") != d_star)
            .join(star_ends, on="n", how="inner")
            .select("n", "d", "t", "d_end", "t_end"))
    set_parts = [_utf8_cells(f) for f in (_cyclic_start_pin_steps(b), _fix_quantity_target(b)) if f is not None]
    if set_parts and over.height > 0:
        already = pl.concat(set_parts, how="vertical").unique()
        both = (over.join(already, on=["n", "d", "t"], how="semi")
                .join(already.rename({"d": "d_end", "t": "t_end"}), on=["n", "d_end", "t_end"], how="semi"))
        over = over.join(both, on=list(over.columns), how="anti")
    if over.height == 0:
        return
    over = _state_cells_typed(b, over.sort("n", "d"), {"d_end": "d", "t_end": "t"})
    m.add_cstr(
        "storage_within_solve_path_end",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": b.v_state,
                     "state_end": -_var_view(b.v_state, "__path_end", {"d": "d_end", "t": "t_end"})},
        rhs_terms = {},
    )
    _notice_stochastic_storage_paths(over["n"].to_list())


def _eq_storage_state_fan_start(b: _Build) -> None:
    """bind_forward_only with a free start: every forecast of a fan in the solve's first period starts at the
    same level."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_bind_forward_only)
            and _has_rows(d.nodeState_first_dt)):
        return
    roots = sorted(set(str(x) for x in d.nodeState_first_dt["d"].to_list()))
    if len(roots) < 2:
        return
    m = b.m

    # ─── EQ. ─── storage_state_fan_start ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t_first(d)] − v_state[n,d″,t_first(d″)] = 0
    #   ∀ n ∈ bind_forward_only, (n,d,t_first(d)) and (n,d″,t_first(d″)) ∈ nodeState_first_dt with no start term
    #     (n ∉ storage_fix_start ∪ carried), d = the first root (sorted), d″ ∈ the other roots, one row
    #     per group of roots that non_anticipativity_storage_state_first does not already tie to d's group
    #   roots = the periods of nodeState_first_dt (built from period_first_of_solve): the forecast branch periods
    #     of a fan in the solve's first period; one root without a fan: no rows
    # The first step of a bind_forward_only node has no state change (dtttdt_forward_only): without fix_start or a
    # carried state it is free, so each forecast of a fan in the first period could pick its own start level from
    # what it later sees.  The fan has only forecasts (no realized member), so this row is what makes their start
    # the same.  A fix_start or carried node starts every root from the same value already (_fix_start_terms,
    # the roll carry).  For a node of an include_stochastics group, non_anticipativity_storage_state_first may tie
    # the first step of a root to its reference forecast already (a tie window shorter than the root period):
    # those pairs are left out, read from that row's domain.
    first = (_utf8_cells(d.nodeState_first_dt)
             .join(_utf8_cells(d.storage_bind_forward_only, ("n",)), on="n", how="semi"))
    for started in (_self_discharge.fix_start_first(d), _self_discharge.carried_nodes(d)):
        if started is not None:
            first = first.join(_utf8_cells(started, ("n",)).unique(), on="n", how="anti")
    if first.height == 0:
        return
    ref = first.filter(pl.col("d") == roots[0])
    other = first.filter(pl.col("d") != roots[0]).select("n", b="d", t_b="t")
    over = ref.join(other, on="n", how="inner").select("n", "d", "t", "b", "t_b")
    tied = _non_anticipativity.storage_state_first_cells(b)
    if tied is not None and over.height > 0:
        edges = (_utf8_cells(tied, ("n", "d", "b", "t"))
                 .join(first, on=["n", "d", "t"], how="semi")
                 .join(first.rename({"d": "b"}), on=["n", "b", "t"], how="semi"))
        if edges.height > 0:
            keep = []
            for node, group in over.group_by("n", maintain_order=True):
                parent: dict[str, str] = {}

                def find(x: str) -> str:
                    while parent.get(x, x) != x:
                        x = parent[x]
                    return x

                for x, y in edges.filter(pl.col("n") == node[0]).select("d", "b").iter_rows():
                    parent[find(x)] = find(y)
                for row in group.sort("b").iter_rows(named=True):
                    if find(row["b"]) != find(roots[0]):
                        parent[find(row["b"])] = find(roots[0])
                        keep.append(row)
            over = pl.DataFrame(keep, schema=over.schema, orient="row")
    if over.height == 0:
        return
    over = _state_cells_typed(b, over.sort("n", "b"), {"b": "d", "t_b": "t"})
    m.add_cstr(
        "storage_state_fan_start",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": b.v_state,
                     "state_root": -_var_view(b.v_state, "__fan_start", {"d": "b", "t": "t_b"})},
        rhs_terms = {},
    )
    _notice_stochastic_storage_paths(over["n"].to_list())


def _end_binding_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the steps storage_state_end_binding pins (the last step of the entire model, of the fix_end /
    fix_start_end nodes, less the steps an upper-level fix_quantity / fix_usage fixes); None when there are none."""
    d = b.d
    if not (b.has_storage
            and d.storage_solve_at_model_end
            and _has_rows(d.storage_fix_end)
            and d.p_state_end is not None
            and _has_rows(d.nodeState_last_dt)
            and d.period_last is not None
            and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None):
        return None
    over = (d.nodeState_last_dt
            .join(d.period_last, on="d", how="inner")
            .join(d.storage_fix_end, on="n", how="inner")
            .select("n", "d", "t").unique())
    if _has_rows(d.nodeState_rp):
        # A representative-period node's end is the end of its base-period chain (rp_inter_period_end_fix), not a
        # state inside the last representative block.
        over = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="anti")
    # An upper-level storage fix at the step wins (fix_quantity: a second equality on the same state; fix_usage: with
    # the end pinned the usage cap can be infeasible).  A nested fix_price does not conflict and keeps the pin.
    over = _without_steps(over, _nested_fixed_end(b, with_price=False))
    over = over.sort("n", "d", "t")
    return over if over.height > 0 else None


def _eq_storage_end_binding(b: _Build) -> None:
    """End state of fix_end / fix_start_end storage nodes at the last step of the entire model, in every sub-solve
    whose horizon holds that step."""
    d = b.d
    over = _end_binding_steps(b)
    if over is None:
        return

    # ─── EQ. ─── storage_state_end_binding ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − state_end[n]·invested[n,d]
    #   + state_end[n]·retired[n,d]
    #   = state_end[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_fix_end, (n,d,t) ∈ nodeState_last_dt, d ∈ period_last, storage_solve_at_model_end,
    #     (n,d,t) ∉ the steps node_balance_fix_quantity_eq_lower / node_storage_usage_fix_le fix
    # Pins the state at the end of the last step of the entire model to the share state_end of the built storage
    # capacity (state units), in every sub-solve whose horizon holds that step (storage_solve_at_model_end,
    # _carry_routing.reaches_model_end: e.g. both an investment solve and the last dispatch roll; each nested level's
    # solve that holds it; a roll whose look-ahead runs to it); each last period (each terminal branch of a
    # stochastic solve) ends there.  storage_fix_end = fix_end ∪ fix_start_end.  Where an
    # upper-level solve fixes the quantity or caps the usage at that step the upper value wins; a nested fix_price
    # steers the end through the objective and keeps the pin.  A fix_start_end node chains forward whatever its
    # storage_binding_method (_run_model._fix_start_end_chains_forward, _self_discharge.block_forward_nodes), so its
    # start is in the first step's balance and never on this state; a cyclic node is pinned here by fix_end only.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    m = b.m
    lhs: dict = {"state": Where(b.v_state, over)}
    invested, divested = _capacity.node_in_service(b, d.storage_fix_end, "_end", coef=d.p_state_end)
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_end_binding",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": (d.p_state_end * b.cap.existing_state_n) / d.p_state_unitsize},
    )


def _end_match_nodes(b: _Build) -> "tuple[pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None]":
    """``(free_first, share_first, ends)`` of the matching end of forward chains without representative periods.

    free_first (n, d, t): the first steps of the bind_forward_only nodes whose chain starts free in this solve
    (no fix_start, no carry); share_first (n, d, t): those of the fix_start nodes the solve can invest in during the
    first period (not with fix_start_end at state_end ≥ state_start); ends (n, d, t): the last step of each last
    period of those nodes.  Only in a solve that holds both the start of the chain (not carried: the first solve of
    the model and of each nested level) and the last step of the model (storage_solve_at_model_end), and not for a
    node an upper-level solve fixes.  Each None when empty.
    """
    d = b.d
    none = (None, None, None)
    # The first roll of a rolling solve that does not reach the end of the model: its free-start nodes get the row
    # at its horizon end (storage_roll_end kind match_first, _run_model._anchor_cyclic_in_rolling).
    first_roll = _roll_end_rows(d, ("match_first",))
    if not (b.has_storage and (d.storage_solve_at_model_end or first_roll is not None)
            and _has_rows(d.storage_bind_forward_only)
            and _has_rows(d.nodeState_last_dt) and _has_rows(d.period_last) and _has_rows(d.nodeState_first_dt)):
        return none
    forward = d.storage_bind_forward_only.select(pl.col("n").cast(pl.Utf8)).unique()
    if not d.storage_solve_at_model_end:
        forward = forward.join(first_roll.select(pl.col("n").cast(pl.Utf8)), on="n", how="semi")
    nested = _nested_nodes(b, with_price=True)
    if nested is not None:
        forward = forward.join(nested, on="n", how="anti")
    if _has_rows(d.nodeState_rp):
        forward = forward.join(d.nodeState_rp.select(pl.col("n").cast(pl.Utf8)), on="n", how="anti")
    if forward.height == 0:
        return none

    def restrict(first) -> "pl.DataFrame | None":
        if first is None:
            return None
        out = first.filter(pl.col("n").cast(pl.Utf8).is_in(forward["n"].implode()))
        return out if out.height > 0 else None

    free_first = restrict(_self_discharge.open_first(d))
    share_first = restrict(_self_discharge.fix_start_first(d)) if d.storage_solve_at_model_end else None
    if share_first is not None:
        periods = sorted(set(str(x) for x in share_first["d"].to_list()))
        investable = _invest_at_start_nodes(b, share_first.select("n").unique(), periods)
        if _has_rows(d.storage_fix_end) and d.p_state_end is not None and d.p_state_start is not None:
            shares = (d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("s"))
                      .join(d.p_state_end.frame.select(pl.col("n").cast(pl.Utf8), pl.col("value").alias("e")),
                            on="n"))
            fix_end = set(str(n) for n in d.storage_fix_end["n"].to_list())
            investable -= {n for n, s, e in shares.iter_rows() if n in fix_end and e >= s - 1e-12}
        share_first = share_first.filter(pl.col("n").cast(pl.Utf8).is_in(sorted(investable)))
        share_first = share_first if share_first.height > 0 else None
    names = set()
    for f in (free_first, share_first):
        if f is not None:
            names |= set(str(n) for n in f["n"].to_list())
    if not names:
        return none
    ends = (d.nodeState_last_dt.join(d.period_last, on="d", how="inner")
            .filter(pl.col("n").cast(pl.Utf8).is_in(sorted(names)))
            .select("n", "d", "t").unique().sort("n", "d", "t"))
    return free_first, share_first, (ends if ends.height > 0 else None)


def _eq_storage_end_match(b: _Build) -> None:
    """The matching end of forward chains (no representative periods) whose start is a decision of the solve."""
    d = b.d
    free_first, share_first, ends = _end_match_nodes(b)
    if ends is None:
        return
    m = b.m
    end_view = _var_view(b.v_state, "__end_match", {"d": "d_end", "t": "t_end"})
    ends = ends.rename({"d": "d_end", "t": "t_end"})

    # ─── EQ. ─── storage_state_end_match ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d_end,t_end] − v_state[n,d,t]  ≥  0
    #   ∀ n ∈ bind_forward_only, (n,d,t) ∈ nodeState_first_dt with a free start (no fix_start, n ∉ carried),
    #     (n,d_end,t_end) ∈ nodeState_last_dt, d_end ∈ period_last, storage_solve_at_model_end
    #     (or n ∈ storage_roll_end(match_first): the first roll of a rolling solve), no upper-level fix on n
    # The matching end: a forward chain that starts free (the solve chooses the level of the first step, which has
    # no state change) ends the model at least at that level, so the solve cannot start full and end empty with
    # energy it never produced.  An inequality: a start of 0 always meets it, so it does not force slack by
    # itself.  fix_end and use_reference_value still set the end; with a free start this bounds the start by that
    # end.  With a stochastic fan in the first period every root start is tied (storage_state_fan_start), so each
    # last step is held to it.  In a rolling solve the first roll gets the row at its horizon end, so it does not
    # choose its start for free while the rolls reaching the end of the model return it (storage_state_roll_match).
    if free_first is not None:
        over = (free_first.select("n", "d", "t")
                .join(ends.with_columns(pl.col("n").cast(free_first.schema["n"], strict=False)), on="n", how="inner")
                .sort("n", "d", "t", "d_end", "t_end"))
        if over.height > 0:
            _notice_nodes("storage_end_matches_start", over.select("n").unique(), b)
            m.add_cstr(
                "storage_state_end_match",
                over      = over,
                sense     = ">=",
                lhs_terms = {"state_end": Where(end_view, over), "neg_state_start": -Where(b.v_state, over)},
                rhs_terms = {},
            )

    # ─── EQ. ─── storage_state_end_match_share ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d_end,t_end] − state_start[n]·invested[n,d] + state_start[n]·retired[n,d]  ≥  0
    #   ∀ n ∈ bind_forward_only ∩ storage_fix_start, n ∉ carried, (n,d,t) ∈ nodeState_first_dt, the solve can
    #     invest in n in d (nd_invest_set), not fix_start_end with state_end ≥ state_start; (n,d_end,t_end) as above
    # The matching end of a fix_start start on capacity the solve can invest in: the start on the capacity the solve
    # adds in the first period (state units) comes back by the end of the model, so storage is not built as a fuel
    # tank whose fix_start share is spent once.  The start on existing capacity is the user's endowment and may be
    # used.  No investment always meets it.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    if share_first is not None and d.p_state_start is not None:
        over = (share_first.select("n", "d", "t")
                .join(ends.with_columns(pl.col("n").cast(share_first.schema["n"], strict=False)), on="n", how="inner")
                .sort("n", "d", "t", "d_end", "t_end"))
        if over.height > 0:
            nodes = over.select("n").unique()
            lhs: dict = {"state_end": Where(end_view, over)}
            invested, divested = _capacity.node_in_service(b, nodes, "_end_match", coef=d.p_state_start)
            if invested is not None:
                lhs["invest_neg"] = -invested
            if divested is not None:
                lhs["divest"] = divested
            _notice_nodes("storage_end_matches_start", nodes, b)
            m.add_cstr(
                "storage_state_end_match_share",
                over      = over,
                sense     = ">=",
                lhs_terms = lhs,
                rhs_terms = {},
            )


def _roll_horizon_ends(b: _Build, nodes: pl.DataFrame) -> "pl.DataFrame | None":
    """``(n, d, t)`` of ``nodes`` (n): the last step of each last period (the roll's horizon end; each terminal
    branch of a stochastic roll); None when there are none."""
    d = b.d
    if not (_has_rows(d.nodeState_last_dt) and _has_rows(d.period_last)):
        return None
    over = (d.nodeState_last_dt.join(d.period_last, on="d", how="inner")
            .join(nodes.select(pl.col("n").cast(d.nodeState_last_dt.schema["n"], strict=False)), on="n", how="semi")
            .select("n", "d", "t").unique().sort("n", "d", "t"))
    return over if over.height > 0 else None


def _first_scope_ends(b: _Build, nodes: pl.DataFrame) -> "pl.DataFrame | None":
    """``(n, d, t)`` of ``nodes`` (n): where the first scope of a rolling roll ends within the roll — the step the
    roll's first step would wrap to: the last step of the first period in the roll (bind_within_period, also
    bind_within_timeblock, which a rolling solve uses as bind_within_period, and bind_intraperiod_blocks).  None
    when there are none."""
    d = b.d
    if not (_has_rows(d.nodeState_first_dt) and _has_rows(d.nodeState_last_dt)):
        return None
    names = nodes.select(pl.col("n").cast(pl.Utf8)).unique()
    first = (d.nodeState_first_dt.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
             .join(names, on="n", how="semi"))
    if first.height == 0:
        return None
    last = d.nodeState_last_dt.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
    utf = first.select("n", "d").join(last, on=["n", "d"], how="inner").unique()
    if utf.height == 0:
        return None
    over = (b.nodeState_dt
            .with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(utf.rename({"n": "_n", "d": "_d", "t": "_t"}), on=["_n", "_d", "_t"], how="semi")
            .select("n", "d", "t").unique().sort("n", "d", "t"))
    return over if over.height > 0 else None


def _do_nothing_level(b: _Build, start: pl.DataFrame, ends: pl.DataFrame) -> pl.DataFrame:
    """``(n, d, t, value)`` (Utf8 keys, MWh): the anchor level of each horizon end of ``ends`` — the lower of the
    carried level ``start`` (n, value) and the level the storage would reach from it by doing nothing.

    Doing nothing: no flow in or out, only the node's own ``inflow`` (a negative one is a forced outflow) and its
    self-discharge, step by step along the forward chain from the roll's first step (dtttdt_forward_only):
    S_t = (S⁻·(1 − β_t) + inflow_t) / (1 + α_t), clipped to [0, existing capacity] (the balance rows'
    self-discharge weights, _self_discharge).  That level is reachable by construction, so the anchor never forces
    slack; a lossless node without inflow keeps the carried level.
    """
    d = b.d
    start = start.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
    ends = ends.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
    out = ends.join(start, on="n", how="inner")
    names = set(start["n"].to_list())
    inflow = None
    if d.p_inflow is not None and _has_rows(d.p_inflow.frame) and "t" in d.p_inflow.frame.columns:
        inflow = (d.p_inflow.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                          pl.col("t").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
                  .filter(pl.col("n").is_in(names) & (pl.col("value") != 0.0)))
    loss = None
    if b.self_discharge is not None:
        loss = (b.self_discharge.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                              pl.col("t").cast(pl.Utf8), "alpha", "beta")
                .filter(pl.col("n").is_in(names)))
    moving = set()
    for frame in (inflow, loss.filter((pl.col("alpha") > 0) | (pl.col("beta") > 0)) if loss is not None else None):
        if frame is not None:
            moving |= set(frame["n"].to_list())
    lag = d.dtttdt_forward_only
    if not moving or not _has_rows(lag) or not _has_rows(d.nodeState_first_dt):
        return out
    nxt: dict = {}
    for dd, tt, dp, tp in lag.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                                     pl.col("d_previous").cast(pl.Utf8),
                                     pl.col("t_previous_within_solve").cast(pl.Utf8)).iter_rows():
        nxt.setdefault((dp, tp), []).append((dd, tt))
    e_of = {} if inflow is None else {(n, dd, tt): v for n, dd, tt, v in inflow.iter_rows()}
    ab_of = {} if loss is None else {(n, dd, tt): (a, bb) for n, dd, tt, a, bb in loss.iter_rows()}
    cap_of: dict = {}
    if b.cap.existing_state_n is not None:
        cap_of = {(n, dd): v for n, dd, v in b.cap.existing_state_n.frame.select(
            pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("value").cast(pl.Float64)).iter_rows()}
    first = d.nodeState_first_dt.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
    carry = dict(start.iter_rows())
    level: dict = {}
    for node in sorted(moving & names):
        roots = [(dd, tt) for dd, tt in first.filter(pl.col("n") == node).select("d", "t").iter_rows()]
        queue = [(step, carry[node]) for step in roots]
        while queue:
            (dd, tt), prev = queue.pop()
            a, bb = ab_of.get((node, dd, tt), (0.0, 0.0))
            now = (prev * (1.0 - bb) + e_of.get((node, dd, tt), 0.0)) / (1.0 + a)
            now = min(max(now, 0.0), cap_of.get((node, dd), float("inf")))
            level[(node, dd, tt)] = now
            queue.extend((step, now) for step in nxt.get((dd, tt), ()))
    if not level:
        return out
    held = pl.DataFrame([(n, dd, tt, v) for (n, dd, tt), v in level.items()],
                        schema={"n": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "_hold": pl.Float64}, orient="row")
    return (out.join(held, on=["n", "d", "t"], how="left")
            .with_columns(value=pl.min_horizontal("value", pl.col("_hold").fill_null(pl.col("value"))))
            .drop("_hold"))


def _capped_at_capacity(b: _Build, level: pl.DataFrame) -> pl.DataFrame:
    """``level`` (n, d, t, value MWh, Utf8 keys) lowered to the storage capacity in service in the row's period
    (existing, MWh; owner decision: a closure keeps the start energy in MWh but never asks for more than the period
    can hold)."""
    cap = b.cap.existing_state_n
    if cap is None:
        return level
    return (level.join(cap.frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                        pl.col("value").cast(pl.Float64).alias("_cap")), on=["n", "d"], how="left")
            .with_columns(value=pl.min_horizontal("value", pl.col("_cap").fill_null(pl.col("value"))))
            .drop("_cap"))


def _eq_storage_roll_end(b: _Build) -> None:
    """The horizon end of the bind_within_solve nodes of a rolling roll that continues from a carried state."""
    d = b.d
    if not (b.has_storage and _has_rows(getattr(d, "storage_roll_end", None)) and d.p_state_unitsize is not None):
        return
    m = b.m
    pinned_end = _end_binding_steps(b)

    def rows(kinds, *, with_price: bool):
        nodes = _roll_end_rows(d, kinds)
        if nodes is None:
            return None, None
        over = (_first_scope_ends(b, nodes) if all(k.startswith("scope_") for k in kinds)
                else _roll_horizon_ends(b, nodes))
        if over is None:
            return None, None
        over = _without_steps(_without_steps(over, _nested_fixed_end(b, with_price=with_price)),
                              None if pinned_end is None else _utf8_cells(pinned_end))
        return (over, nodes) if over.height > 0 else (None, None)

    # ─── EQ. ─── storage_state_roll_anchor ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t]·state_unitsize[n] ≥ min(carry[n], hold[n,d,t])
    #   ∀ n ∈ storage_roll_end(anchor), (n,d,t) ∈ nodeState_last_dt, d ∈ period_last;
    #     n ∈ storage_roll_end(scope_anchor), (n,d,t) = the end of the roll's first scope (_first_scope_ends: the
    #     last step of the first period in the roll);
    #     (n,d,t) ∉ the steps an upper-level solve fixes (fix_quantity, fix_usage, fix_price),
    #     (n,d,t) ∉ the steps storage_state_end_binding pins
    #   carry = the level the roll starts from (MWh, roll_continue_state); hold = the level reached from it by doing
    #   nothing (_do_nothing_level: inflow and self-discharge only, clipped to [0, existing capacity])
    # A bind_within_solve node of a rolling solve continues each roll from the level realised by the previous roll
    # (the cascade rewrites it to bind_forward_only for the roll, _run_model._anchor_cyclic_in_rolling), and a roll
    # whose horizon does not reach the end of its solve ends at least at that level: each roll's plan is cyclic on
    # its own start, as the first roll's wrap is.  An inequality, so that it can always be reached: storing more has
    # a cost, so a lossless node ends at the carried level.  Where an upper-level solve fixes, caps or prices the
    # state at the step, that replaces the anchor; fix_end (at the last step of the model) wins.  The same per
    # period (bind_within_period, bind_intraperiod_blocks; bind_within_timeblock in a rolling solve is
    # bind_within_period, _run_model._timeblock_binds_within_period_in_roll): a roll that
    # starts inside a period an earlier roll began continues it from the carry (scope_carry: its first step has no
    # wrap) and anchors the period's part in the roll; for bind_intraperiod_blocks also the roll that begins a period
    # at its first step and holds it in part (its blocks chain forward from the carry).
    over, nodes = rows(("anchor",), with_price=True)
    scope_over, scope_nodes = rows(("scope_anchor",), with_price=True)
    if scope_over is not None:
        # A bind_within_period / bind_intraperiod_blocks node whose first period continues from the carry: the
        # anchor sits where that period's part in the roll ends.
        over = scope_over if over is None else pl.concat([over, scope_over.cast(over.schema)]).unique()
        nodes = scope_nodes if nodes is None else pl.concat([nodes, scope_nodes.cast(nodes.schema)])
    if over is not None:
        level = _do_nothing_level(b, nodes, over)
        m.add_cstr(
            "storage_state_roll_anchor",
            over      = over,
            sense     = ">=",
            lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
            rhs_terms = {"level": Param(("n", "d", "t"), _state_keys_typed(b, level))},
        )

    # ─── EQ. ─── storage_state_roll_close ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t]·state_unitsize[n] = start[n]
    #   ∀ n ∈ storage_roll_end(close), (n,d,t) ∈ nodeState_last_dt, d ∈ period_last;
    #     n ∈ storage_roll_end(scope_close), (n,d,t) = the end of the roll's first scope (_first_scope_ends);
    #     (n,d,t) ∉ the steps an upper-level fix_quantity / fix_usage fixes,
    #     (n,d,t) ∉ the steps storage_state_end_binding pins
    #   start = the level (MWh) the first roll of the instance (scope_close: the roll that began the period)
    #   started from (the cascade records it), at most the storage capacity in service in d (owner
    #   decision: the start energy is kept in MWh, so capacity built later adds none, but a closure never asks
    #   for more than the closing period can hold)
    # A roll whose horizon holds the last step of its solve instance closes the cycle of bind_within_solve: it ends at
    # the level the instance started from, so the realised rolls together return what they took (every roll whose
    # horizon reaches that step, e.g. two when the horizon is not a multiple of the jump).  An upper-level
    # fix_quantity / fix_usage at the step replaces it (that target carries the upper solve's closure); a nested
    # fix_price only values the state and keeps it; fix_end wins at the last step of the model.  Per period
    # (bind_within_timeblock in a rolling solve is bind_within_period): the roll that holds the end of a period it
    # continues (or reaches the end of the instance) ends
    # the scope at its start level, so a realised roll boundary inside the scope creates no energy (logic doc
    # Lemma 4).
    over, nodes = rows(("close",), with_price=False)
    scope_over, scope_nodes = rows(("scope_close",), with_price=False)
    if scope_over is not None:
        # The end of a continued period that the roll holds: it ends at the level the period started from in the
        # roll that began it.
        over = scope_over if over is None else pl.concat([over, scope_over.cast(over.schema)]).unique()
        nodes = scope_nodes if nodes is None else pl.concat([nodes, scope_nodes.cast(nodes.schema)])
    if over is not None:
        level = _capped_at_capacity(b, over.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
                                    .join(nodes.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64)),
                                          on="n"))
        m.add_cstr(
            "storage_state_roll_close",
            over      = over,
            sense     = "==",
            lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
            rhs_terms = {"start": Param(("n", "d", "t"), _state_keys_typed(b, level))},
        )

    # ─── EQ. ─── storage_state_roll_match ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t]·state_unitsize[n] ≥ min(start[n], capacity[n,d])
    #   ∀ n ∈ storage_roll_end(match) ∖ nodeState_rp, (n,d,t) ∈ nodeState_last_dt, d ∈ period_last,
    #     (n,d,t) ∉ the steps an upper-level solve fixes, (n,d,t) ∉ the steps storage_state_end_binding pins
    #   start = the level (MWh) the first roll started the node from (free: fix_nothing; the cascade records it)
    # The matching end of a forward chain in a rolling solve or a sequence: a roll (or a later solve) that reaches
    # the end of the model ends a bind_forward_only node whose start the first roll (the first solve) chose at least
    # at that start (storage_state_end_match in a solve that holds the whole model).  An inequality: the first roll's own horizon end is held to its start
    # (storage_state_end_match), so it starts no higher than it can refill.
    over, nodes = rows(("match",), with_price=True)
    if over is not None and _has_rows(d.nodeState_rp):
        over = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="anti")
        over = over if over.height > 0 else None
    if over is not None:
        level = _capped_at_capacity(b, over.select(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
                                    .join(nodes.select(pl.col("n").cast(pl.Utf8), pl.col("value").cast(pl.Float64)),
                                          on="n"))
        m.add_cstr(
            "storage_state_roll_match",
            over      = over,
            sense     = ">=",
            lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
            rhs_terms = {"start": Param(("n", "d", "t"), _state_keys_typed(b, level))},
        )

    # ─── EQ. ─── storage_state_roll_anchor_share / storage_state_roll_close_share ──
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − state_start[n]·invested[n,d] + state_start[n]·retired[n,d]
    #   (≥ | =) state_start[n] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_roll_end(anchor_share | close_share), (n,d,t) as above (the anchor's or the closure's
    #     exclusions)
    # The first roll of a nested lower level under an upper-level fix_quantity / fix_usage starts from fix_start
    # (forward, _fix_start_terms, not from its wrap: the wrap target would be the upper target), so where no upper
    # target lies at its horizon end the anchor (≥, a roll that does not reach the end of its instance) or the
    # closure (=) is the fix_start share of the capacity in service.
    if b.cap.existing_state_n is not None and d.p_state_start is not None:
        for kind, name, sense, with_price in (("anchor_share", "storage_state_roll_anchor_share", ">=", True),
                                              ("close_share", "storage_state_roll_close_share", "==", False)):
            over, nodes = rows((kind,), with_price=with_price)
            if over is None:
                continue
            lhs: dict = {"state": Where(b.v_state, over)}
            invested, divested = _capacity.node_in_service(b, nodes.select("n"), f"_{kind}", coef=d.p_state_start)
            if invested is not None:
                lhs["invest_neg"] = -invested
            if divested is not None:
                lhs["divest"] = divested
            m.add_cstr(
                name,
                over      = over,
                sense     = sense,
                lhs_terms = lhs,
                rhs_terms = {"target": (d.p_state_start * b.cap.existing_state_n) / d.p_state_unitsize},
            )


def _horizon_end_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the storage steps where the solve horizon ends for the storage_solve_horizon_method — the last
    step of each last period (the solve's last period and its stochastic branch siblings, period_last) and, in a
    rolling solve, the end of a chained bind_intraperiod_blocks tail that a looping period follows
    (_self_discharge.block_tail_ends).  None when there are none."""
    d = b.d
    if not _has_rows(d.nodeState_last_dt):
        return None
    parts = []
    if _has_rows(d.period_last):
        parts.append(d.nodeState_last_dt.join(d.period_last, on="d", how="inner").select("n", "d", "t"))
    tails = _self_discharge.block_tail_ends(d)
    if tails is not None:
        parts.append(tails.select("n", "d", "t"))
    if not parts:
        return None
    out = pl.concat(parts, how="vertical").unique()
    return out if out.height > 0 else None


def _eq_storage_reference_value(b: _Build) -> None:
    """End state of storage nodes with a reference value."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.storage_use_reference_value)
            and d.p_storage_state_reference_value is not None
            and b.cap.existing_state_n is not None
            and d.p_state_unitsize is not None):
        return
    ends = _horizon_end_steps(b)
    if ends is None:
        return
    m = b.m
    over = (ends
            .join(d.storage_use_reference_value, on="n", how="inner")
            .select("n", "d", "t").unique())
    if _has_rows(d.nodeState_rp):
        # A representative-period node takes its reference value on the end of its base-period chain
        # (rp_inter_period_reference_value), not on a state inside the last representative block; a cyclic one
        # none (its closure wins).
        rp_ref = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="semi")
        _notice_nodes("rp_storage_chain_pinned", rp_ref.select("n").unique(), b)
        over = over.join(d.nodeState_rp.select(pl.col("n").cast(over.schema["n"], strict=False)), on="n", how="anti")
    if _has_rows(d.nodeStateBlock):
        # A bind_intraperiod_blocks node (in storage_use_reference_value in rolling solves only) is pinned where the
        # roll holds the period in part (period_block_chain): at the roll end and at the end of a chained tail that a
        # looping period follows; where its blocks loop the end is tied to the start.
        chain = [str(x) for x in d.period_block_chain["d"].to_list()] if _has_rows(d.period_block_chain) else []
        is_block = pl.col("n").cast(pl.Utf8).is_in([str(x) for x in d.nodeStateBlock["n"].to_list()])
        over = over.filter(~is_block | pl.col("d").cast(pl.Utf8).is_in(chain))
    # An upper-level storage fix at the step (fix_quantity, fix_usage or fix_price) overrides the horizon-end
    # reference value (how_to/solve_sequences.md, nested solves); the reference stays at every other roll end.
    over = _without_steps(over, _nested_fixed_end(b, with_price=True))
    # A rolling roll that closes a bind_intraperiod_blocks period it continues (storage_roll_end kind scope_close)
    # ends the period at its start level there; the reference stays at the roll's other ends.
    closes = _roll_end_rows(d, ("scope_close",))
    if closes is not None:
        over = _without_steps(over, _utf8_cells(_first_scope_ends(b, closes)))
    if over.height == 0:
        return

    # ─── EQ. ─── storage_state_solve_horizon_reference_value ─────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − reference_value[n,d,t]·invested[n,d]
    #   + reference_value[n,d,t]·retired[n,d]
    #   = reference_value[n,d,t] · capacity[n,d] / state_unitsize[n]
    #   ∀ n ∈ storage_use_reference_value, (n,d,t) ∈ horizon_end,
    #     (n ∈ nodeStateBlock ⇒ d ∈ period_block_chain), (n,d,t) ∉ the steps an upper-level solve fixes
    #   horizon_end = nodeState_last_dt ∩ period_last  ∪  block_tail_end
    # Pins the state at the end of the solve horizon to the reference share of the built storage capacity
    # (state units), as maxState and the start pins count it.  period_last is the solve's last period and its
    # stochastic branch siblings only: an earlier period of a multi-period solve is not a horizon end.  Nodes with a
    # competing end condition are not in storage_use_reference_value; a bind_intraperiod_blocks node is, in a
    # rolling solve, and is pinned where the roll holds the period only in part (there its blocks chain forward
    # instead of looping): at the roll end and at block_tail_end, the last step of a chained part-period tail that
    # a period held whole follows (nothing continues from it, _self_discharge.block_tail_ends).  Where an
    # upper-level solve fixes the node at the step (nested fix_quantity, fix_usage, fix_price) the upper value
    # overrides the reference value: two equalities on one state (fix_quantity), a usage cap the pin could make
    # infeasible (fix_usage), or a price meant to steer that end (fix_price).
    # reference_value on the pinned rows only: as a coefficient of the invest / divest sums it would otherwise span
    # every (n, d_invest, d, t) of the solve.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    rv = (promote_param_to_dt(d.p_storage_state_reference_value, d.dt)
          .join(over.lazy(), on=["n", "d", "t"], how="inner")
          .select("n", "d", "t", "value")
          .collect())
    target = (rv.lazy().rename({"value": "rv"})
              .join(b.cap.existing_state_n.frame.lazy().rename({"value": "exist"}),
                    on=["n", "d"], how="inner")
              .join(d.p_state_unitsize.frame.lazy().rename({"value": "us"}), on="n", how="inner")
              .with_columns(value=pl.col("rv") * pl.col("exist") / pl.col("us"))
              .select("n", "d", "t", "value")
              .collect())
    # A reference value below the start share (fix_start, or the reference at the first step) of a node whose start
    # the solve decides by investing: the matching end (storage_state_end_match_share) stays, as for fix_start_end,
    # and caps the investment; a warning (_warn_reference_below_start).
    _free, share_first, _ends = _end_match_nodes(b)
    if share_first is not None and d.p_state_start is not None:
        share = {str(n): v for n, v in d.p_state_start.frame.select(pl.col("n").cast(pl.Utf8), "value").iter_rows()}
        matched = set(str(n) for n in share_first["n"].to_list())
        _warn_reference_below_start(b, {str(n) for n, v in rv.select("n", "value").iter_rows()
                                        if str(n) in matched and v < share[str(n)] - 1e-12})
    lhs: dict = {"state": Where(b.v_state, over)}
    invested, divested = _capacity.node_in_service(b, d.storage_use_reference_value, "_ref_value",
                                                   coef=Param(("n", "d", "t"), rv))
    if invested is not None:
        lhs["invest_neg"] = -invested
    if divested is not None:
        lhs["divest"] = divested
    m.add_cstr(
        "storage_state_solve_horizon_reference_value",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _warn_reference_below_start(b: _Build, nodes: set) -> None:
    """Warn about the nodes whose use_reference_value end lies below their start share while the solve can invest
    in their storage in the period they start in (owner decision, Phase 5; before a configuration error): as for
    fix_start_end, the matching end stays, so storage built must return its start energy, which caps the
    investment.  One printed line per solve."""
    if not nodes:
        return
    from flextool import run_warnings
    run_warnings.warn_grouped("W-0538", sorted(nodes), solve=b.solve_name)


def _eq_profiles(b: _Build) -> None:
    """Profile limits of process flows."""
    if not b.has_profile:
        return
    d = b.d

    # ─── EQ. ─── profile_flow_{upper_limit,lower_limit,fixed} ────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] ± (Σ_r v_reserve[p,r,k,d,t] + Σ_r v_reserve[p,r,s,d,t] · φ[p,s,k,d,t])
    #   − profile[f,d,t]·av[p,d,t]·invested[p,d]
    #   + profile[f,d,t]·av[p,d,t]·retired[p,d]
    #   ≤ / ≥ / =  profile[f,d,t] · av[p,d,t] · existing_count[p,d]
    #   ∀ (p,s,k,f) ∈ process_profile_{upper,lower,fixed}, (d,t) ∈ dt
    # Bounds a flow by a profile times the built capacity, in units of unitsize.  av = availability, 1 for
    # processes without availability data.  The upper limit adds the reserve that raises the arc's flow
    # (up at the sink; down at a unit's source), the lower limit subtracts the reserve that lowers it (down
    # at the sink; up at a unit's source); a source-side reserve is converted to the arc's flow by φ (1
    # where v_flow is the input, 1/slope on a direct unit's input→output arc).  The fixed profile has no
    # reserve term.  The invest/divest terms are restricted to the (p,f) pairs of the method's set.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    # The cached sums of _capacity, present only where a period is in service (no term over an empty domain).
    invested = divested = None
    if b.has_invest_p and _entity_rows(b.cap.alive, "e", d.process_source_sink, "p").height > 0:
        invested = b.cap.invested_p
    if b.has_divest_p or b.has_divest_built_p or b.has_extend_p:
        # The retirements and process_source_sink carry different p vocabularies: cast to the former.
        retired_p = b.cap.retired_frame("p")
        procs = d.process_source_sink.select(pl.col("p").cast(retired_p.schema["p"], strict=False)).unique()
        if (retired_p.join(procs, on="p", how="semi").height > 0 or b.has_divest_built_p
                or b.has_extend_p):
            divested = b.cap.retired_p
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
        rhs = d.p_profile_value * b.cap.existing_count_p
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
        _coarse_flows.add_cstr(
            b, name,
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
#   inflation_op[d] (the social discount factor of operating costs) = Σ_y pyr[d,y]·(1 + social_discount_rate)^−(
#     base[d,y] + pyr[d,y]·discount_offset_operations), y the years d represents, pyr their length, base[d,y] the
#     years from the run start to the start of y
# step_duration turns a rate (currency/h) into currency per step, timestep_weight scales a representative step
# to the time it stands for, / period_share scales the modelled part of the period to a full year, and
# inflation_op discounts that year's operating cost; branch_weight is 1 outside stochastic solves.  Terms for point events
# (startups, ramp changes, end-of-horizon valuation) use w_ev[d,t] = w_op[d,t] / step_duration[d,t].
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

    # ─── OBJ. ─── commodity_cost_startup ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,n,c) ∈ startup_from_commodity, (d,t)}
    #     v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p] · commodity_price[c,d,t] · w_ev[d,t]
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight   (as startup_cost)
    # Cost of the start-up energy that units draw from the commodity node n of commodity c (startup_input_method
    # startup_and_input / startup_only; core._startup_draw).  The draw is MWh per start-up event, so the weight
    # has no step_duration.  Commodities with a price ladder are priced by _commodity_ladder.
    if b.has_startup_draw:
        buy_su = _single_price(getattr(d, "startup_from_commodity", None))
        terms += [Sum(term * d.p_commodity_price * b.ev_factor) for _key, term in _startup_draw(b, buy_su)]

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
    """CO2 emission cost of processes with an efficiency, net of removal credit."""
    if not b.has_co2_price:
        return None
    d = b.d
    terms: list[Expr] = []

    # ─── OBJ. ─── co2_price_eff ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_priced, (d,t)}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # Cost of the CO2 emitted when processes with an efficiency draw a CO2-bearing commodity c from a
    # node of a CO2-priced group g; content in t/MWh, price in currency/t.  The section term exists
    # for min_load_efficiency units only.
    if _has_rows(d.flow_from_co2_priced):
        terms.extend(Sum(term * d.p_co2_content * d.p_co2_price * b.op_factor)
                     for _key, term in _eff_fuel_input(b, d.flow_from_co2_priced))

    # ─── OBJ. ─── co2_removal_credit ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,k,c,g) ∈ flow_to_co2_priced, (d,t)}
    #     v_flow[p,s,k,d,t]·unitsize[p] · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # Credit for flows into the group's priced commodity node(s) — removal (e.g. DAC/BECCS modelled
    # as a flow into a CO2-content commodity node of the priced group g).  No slope or section term:
    # the delivered flow is the physical quantity removed, not a fuel input (mirrors flow_to_commodity,
    # the commodity-sale term).  Ported from GMPL's "Receiving credits for removing CO2" (#19, Q14).
    if _has_rows(getattr(d, "flow_to_co2_priced", None)):
        terms.append(-Sum(
            Where(b.v_flow * d.p_unitsize, d.flow_to_co2_priced)
            * d.p_co2_content * d.p_co2_price * b.op_factor))

    # ─── OBJ. ─── co2_price_startup ──────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,n,c,g) ∈ startup_from_co2_priced, (d,t)}
    #     v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p] · co2_content[c] · co2_price[g,d,t] · w_ev[d,t]
    # Cost of the CO2 of the start-up energy drawn from a node of a CO2-priced group g that carries the
    # CO2-bearing commodity c (core._startup_draw); an event, so the weight w_ev has no step_duration.
    if b.has_startup_draw and _has_rows(getattr(d, "startup_from_co2_priced", None)):
        terms.extend(Sum(term * d.p_co2_content * d.p_co2_price * b.ev_factor)
                     for _key, term in _startup_draw(b, d.startup_from_co2_priced))
    return _sum_terms(terms)


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
    startup_factor = b.ev_factor
    if b.has_startup_cost_lin:
        terms.append(Sum(
            Where(b.v_startup_linear, d.pdt_online_linear)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    if b.has_startup_cost_int:
        terms.append(Sum(
            Where(b.v_startup_integer, d.pdt_online_integer)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    return _sum_terms(terms)


def _obj_ramp_cost(b: _Build) -> Expr | None:
    """Cost of changing the flow of the arcs with a ramp cost."""
    if b.v_ramp_up is None:
        return None
    d = b.d

    # ─── OBJ. ─── ramp_cost ──────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ R} (v_ramp_up + v_ramp_down)[p,s,k,d,t] · unitsize[p] · rc[p,s,k] · w_ev[d,t]
    #   rc[p,s,k] = p_ramp_cost_arc = Σ_{side ∈ {source s, sink k}, ramp_method[p,side] ∈ {ramp_cost, both}}
    #               ramp_cost[p,side]   (> 0; a direct unit's one arc takes both sides)
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight   (as startup_cost)
    # Cost of changing a flow, ramp_cost in currency per MW of change of the step-average flow, up and down
    # alike.  A change is an event, as a start-up, so the weight has no step_duration: ramping 0 → X → 0 costs
    # 2·rc·X whatever the step lengths.  A start-up from 0 pays both its startup cost and this cost.
    factor = b.ev_factor
    return _sum_terms([Sum(v * d.p_ramp_cost_arc * d.p_unitsize * factor)
                       for v in (b.v_ramp_up, b.v_ramp_down)])


def _obj_storage_reference_price(b: _Build) -> Expr | None:
    """Value of the storage content left at the end of the solve horizon."""
    d = b.d
    if not (b.has_storage
            and d.p_storage_state_reference_price is not None):
        return None
    ends = _horizon_end_steps(b)
    if ends is None:
        return None
    over = (ends
            .join(d.p_storage_state_reference_price.frame.select("n", "d"),
                  on=["n", "d"], how="inner")
            .select("n", "d", "t").unique())
    if over.height == 0:
        return None

    # ─── OBJ. ─── storage_reference_price ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(n,d,t) ∈ horizon_end, (n,d) ∈ reference_price}
    #     v_state[n,d,t]·state_unitsize[n] · reference_price[n,d] · w_ev[d,t]
    #   horizon_end = nodeState_last_dt ∩ period_last  ∪  block_tail_end   (as storage_state_solve_horizon_
    #                 reference_value: the solve's last period and its branch siblings, and in a rolling solve the
    #                 end of a chained bind_intraperiod_blocks tail that a looping period follows)
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
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n
            or b.has_divest_built_p or b.has_divest_built_n or b.has_extend_p or b.has_extend_n):
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

    def _costs(var: str, *fields: str) -> list["Param | None"]:
        """The cost frames ``fields`` of a solve that has ``var`` investment variables.  The preprocessing gives
        each of them (zero rows where no cost is authored); none at all means the cost derivation did not run for
        the solve, which would leave the investments free in the objective, so it is an error.  (A retirement
        without its frames only loses its saving: no check.)"""
        params = [getattr(d, f, None) for f in fields]
        if all(p is None for p in params):
            raise FlexToolConfigError(
                f"The solve has {var} variables but none of the cost coefficients {', '.join(fields)}: the "
                "decisions would be free in the objective. The cost derivation did not run for this solve; please "
                "report it as a FlexTool bug.")
        return params

    # ─── OBJ. ─── invest_cost_p ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,d) ∈ pd_invest_set} v_invest_p[p,d]·unitsize[p] · annual_discounted[p,d] · bw[d]
    #   + Σ_{(p,d) ∈ pd_invest_set} v_invest_p[p,d]·unitsize[p] · lifetime_fixed_cost[p,d] · bw[d]
    #   annual_discounted[p,d] = (a_cap[p,c] − a_salv[p,c]) · Σ_{d′ ∈ L(p,d)} inflation_inv[d′]
    #                                                                       (reinvest_choice / no_reinvestment)
    #   inflation_inv[d] (the social discount factor of investment payments) = Σ_y pyr[d,y]·(1 +
    #     social_discount_rate)^−(base[d,y] + pyr[d,y]·discount_offset_investment) (y, pyr, base as for inflation_op)
    #                          = Σ_{g ∈ gen(p,d)} a_b[g] · Σ_{d′ ∈ L_g(p,d)} inflation_inv[d′]   (reinvest_automatic)
    #   gen(p,d): the chain of the capacity decided in d: g1 online at c(d); g_{k+1} online at y_eol[g_k] =
    #     round_end_of_life(y_on[g_k] + lifetime_technical[p, d_on(g_k)], end_of_life_method[p]) (at least one
    #     period: the next seam when that is at or before y_on[g_k]), until the end of the horizon
    #   L_g(p,d) = the solve's periods d′ on the walk lineage with yr[c] ≤ yr[d′], y_on[g] ≤ yr[d′] < y_eol[g]
    #   a_b[g] = a_cap[g] − a_salv[g] as below, every value read at d_on(g) (the period g comes online in, on d′'s
    #     lineage inside a stochastic fan), S at d_eol(g)
    #   c = c(d) the period the capacity comes online; L(p,d) = the solve's periods d′ with yr[c] ≤ yr[d′] < y_eol[p,d]
    #     (no upper end for reinvest_automatic): the periods the cohort of the decision is in service in, y_eol =
    #     round_end_of_life(yr[c] + lifetime_technical[p,c], end_of_life_method[p]) (engine.md "Capacity symbols")
    #   a_cap[p,c] = A[p,c]·G(n)/G(L) (A[p,c] when n = L); A[p,c] = ann(invest_cost[p,c], discount_rate[p,c], n);
    #     n = payback_time[p,c] (lifetime_technical[p,c] where the entity has none); L = lifetime_technical[p,c];
    #     G(x) = Σ_{k<⌈x⌉} min(1, x−k)·(1+social_discount_rate)^−k
    #   a_salv[p,c] = S·(1+r)^−L·af(r, L), S = 1000·net_salvage_value[p,d_eol] (d_eol the period y_eol starts; the
    #     last period after the last period start), r = discount_rate[p,c]
    #   lifetime_fixed_cost[p,d] = 1000 · Σ_{d′ ∈ L(p,d)} fixed_cost[p,d′]·inflation_op[d′]
    #   ann(c, r, n) = c·1000·af(r, n);  af(r, x) = r / (1 − (1+r)^−x) · (1+r)^(off_inv − 1) (1 / x for r = 0);
    #     off_inv = discount_offset_investment: each payment falls off_inv years into its service year
    # Cost of process capacity decided in period d, currency/MW. Capital: the level yearly cost of a plant of its real
    # lifetime (the present value of the loan, paid over payback_time at discount_rate, spread evenly over
    # lifetime_technical years; the payments are sized for when they fall, so the loan taken when the capacity comes
    # online is worth invest_cost at discount_rate for every off_inv), charged for each year the cohort is in service
    # inside the horizon (discounted per period). end_of_life_method sets the years in service, not the price;
    # payback_time and discount_rate set only the size (with discount_rate = social_discount_rate payback_time changes
    # nothing for a whole number of years). The salvage value at the end of the real lifetime lowers the price: its
    # present value at discount_rate, levelled over the lifetime_technical years (the textbook capital recovery with
    # salvage when payback_time = lifetime_technical). A reinvest_automatic plant is replaced at the end of its life
    # by a plant at the invest cost (and discount_rate, payback_time, lifetime_technical, salvage value) of the period
    # the replacement comes online, and so on to the end of the horizon; each pays its own level yearly cost for its
    # years in service (with constant values the chain charges the first plant's annuity for every period).
    # The inflation-adjusted fixed cost of each period the capacity
    # operates in is charged over the same periods (it stops at the rounded end of life), as for existing capacity.
    # With a construction lead time the capacity operates from its start of operation c(d): both windows start there,
    # and the invest cost, interest rate, payback time and lifetime are those of c(d): their period maps describe the
    # plant as constructed (invest_cost is an overnight cost); without one c(d) = d. No w_op: both factors are already
    # per period. A solve with invest variables and neither cost frame is an error, not a free investment.
    if b.has_invest_p:
        annu, lf = (_cost(p, "p") for p in _costs("v_invest_p", "ed_entity_annual_discounted",
                                                    "ed_lifetime_fixed_cost"))
        if annu is not None:
            terms.append(Sum(b.v_invest_p * d.p_unitsize * annu))
        if lf is not None:
            terms.append(Sum(b.v_invest_p * d.p_unitsize * lf))

    # ─── OBJ. ─── divest_cost_p ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,d) ∈ pd_divest_set} v_divest_p[p,d]·unitsize[p] · lifetime_fixed_cost_divest[p,d] · bw[d]
    #   − Σ_{(p,d) ∈ pd_divest_set} v_divest_p[p,d]·unitsize[p] · divest_salvage[p,d] · bw[d]
    #   lifetime_fixed_cost_divest[p,d] = Σ_{d′ ∈ W(p,d)} fixed_cost[p,d′]·1000·inflation_op[d′]
    #   divest_salvage[p,d] = [g1 in service in d]·(D(d)·S_d − a_x[p]·S̄_x[p,d])
    #                         − [g a replacement, m > 0]·set[g,d] + Σ_{replacements g′ ≥ g} a_b[g′]·S̄[g′,d]
    #     (reinvest_automatic: g the generation of p's existing chain in service in d, g1 the existing plant;
    #      m = yr[d] − y_on[g]; set, S̄[g′,d] and a_b as in divest_built_cost (_retire_built), a_b of one plant
    #      built in d_on(g′); other entities: the first term, g1 the existing cohort)
    #   W(p,d) = the solve's periods d′ with yr[d] ≤ yr[d′] < EOL[p] (the window of retired[p,·]: until the
    #     end of life of p's pre-existing capacity, the end of the horizon for reinvest_automatic)
    #   S_d = 1000·net_salvage_value[p,d]; D(d) = (1+social_discount_rate)^−(years from the run start to the start of d)
    #   a_x[p] = S_x·(1+r_x)^−R_x·af(r_x, R_x) (af: invest_cost_p), S_x = 1000·net_salvage_value[p,d_eol] (the
    #     last period's value after the last period start), r_x = discount_rate[p,d0], R_x = y_eol[p] − y0 (y_end[p]
    #     − y0 for a life past the horizon; 0 for a plant that serves no period), y_eol / y_end of g1 for
    #     reinvest_automatic (a_x = 0 without a lifetime); y0 / d0 the start of the run
    #   S̄_x[p,d] = Σ_{d′ ∈ history ∪ solve, y0 ≤ yr[d′] < EOL[p], on d's lineage} inflation_inv[d′]
    # Saving from process capacity retired in period d, currency/MW: the fixed cost of the existing capacity it no
    # longer pays (each period's, with the factor the existing fixed cost has), and its salvage settlement.  The
    # investment in existing capacity is sunk; it holds its salvage value as a level credit a_x per year in service
    # (existing_salvage); a retired MW receives its net salvage value at the start of d and gives back every credit,
    # so it earns exactly S_d·D(d).  A kept plant's credits are counted inside the horizon only.  Existing
    # reinvest_automatic capacity is replaced at its end of life (existing_replacement): retiring it also avoids every
    # replacement still to come, and retiring a replacement settles that plant's contract as it ran (a replacement
    # retired at its own start was never bought).
    if b.has_divest_p:
        lfd = _cost(d.ed_lifetime_fixed_cost_divest, "p")
        salvage = _cost(d.ed_divest_salvage, "p")
        if lfd is not None:
            terms.append(-Sum(b.v_divest_p * d.p_unitsize * lfd))
        if salvage is not None:
            terms.append(-Sum(b.v_divest_p * d.p_unitsize * salvage))

    # ─── OBJ. ─── invest_cost_n ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(n,d) ∈ nd_invest_set} v_invest_n[n,d]·state_unitsize[n] · annual_discounted[n,d] · bw[d]
    #   + Σ_{(n,d) ∈ nd_invest_set} v_invest_n[n,d]·state_unitsize[n] · lifetime_fixed_cost[n,d] · bw[d]
    # As invest_cost_p for storage capacity, currency/MWh of storage.
    if b.has_invest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        annu_n, lf_n = (_cost(p, "n") for p in _costs("v_invest_n", "ed_entity_annual_discounted",
                                                        "ed_lifetime_fixed_cost"))
        if annu_n is not None:
            terms.append(Sum(b.v_invest_n * us_n * annu_n))
        if lf_n is not None:
            terms.append(Sum(b.v_invest_n * us_n * lf_n))

    # ─── OBJ. ─── divest_cost_n ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(n,d) ∈ nd_divest_set} v_divest_n[n,d]·state_unitsize[n] · lifetime_fixed_cost_divest[n,d] · bw[d]
    #   − Σ_{(n,d) ∈ nd_divest_set} v_divest_n[n,d]·state_unitsize[n] · divest_salvage[n,d] · bw[d]
    # As divest_cost_p for storage capacity (W over the window of retired[n,·]).
    if b.has_divest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        lfd_n = _cost(d.ed_lifetime_fixed_cost_divest, "n")
        salvage_n = _cost(d.ed_divest_salvage, "n")
        if lfd_n is not None:
            terms.append(-Sum(b.v_divest_n * us_n * lfd_n))
        if salvage_n is not None:
            terms.append(-Sum(b.v_divest_n * us_n * salvage_n))

    # The retirement of model-built capacity (divest_built_cost_{p,n}): _retire_built.
    terms.extend(_retire_built.objective_terms(b))
    # The life extensions (extend_cost_{p,n}): _extend.
    terms.extend(_extend.objective_terms(b))
    return _sum_terms(terms)


def _obj_existing_fixed_cost(b: _Build) -> None:
    """Fixed cost of existing capacity as an objective constant (opt-in)."""
    d = b.d
    if not (b.include_existing_fixed_cost
            and d.p_ed_fixed_cost is not None
            and b.cap.existing_mw_e is not None):
        return

    # ─── OBJ. ─── existing_fixed_cost ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(e,d)} existing[e,d] · fixed_cost[e,d] · inflation_op[d] · bw[d]
    #   bw = pd_branch_weight, factor 1 when absent
    # Fixed cost of the capacity that exists in each period, currency (fixed_cost in currency/MW/year).
    # A constant: it moves the objective value but not the solution.  Added only with
    # include_existing_fixed_cost; when pd_branch_weight is present, periods without a weight row drop out.
    # Multiplied by scale_the_objective like every other objective term.
    fc_frame = (
        b.cap.existing_mw_e.frame.rename({"value": "exist"})
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
    fc_const = fc_frame.sort("e", "d")["contrib"].sum()  # fixed summation order (the joins leave it unspecified)
    if fc_const:
        b.m.add_obj_constant(float(fc_const) * b.scale_the_objective)


def _obj_existing_salvage(b: _Build) -> None:
    """The salvage value of existing capacity as a level yearly credit, an objective constant (opt-in)."""
    d = b.d
    credit = getattr(d, "ed_existing_salvage_credit", None)
    if not (b.include_existing_fixed_cost and credit is not None and b.cap.existing_cohort_size_e is not None):
        return

    # ─── OBJ. ─── existing_salvage ───────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(e,d) ∈ credit} (pre_existing[e,d] − retired_carried[e,d]) · a_x[e]·inflation_inv[d] · bw[d]
    #   pre_existing = p_entity_pre_existing (the existing cohort's size, lifetime-gated, MW, MWh),
    #   retired_carried = p_entity_retired_carried (its retirements realized by earlier solves of a chain)
    #   credit = ed_existing_salvage_credit[e,d] = a_x[e]·inflation_inv[d] over the periods y0 ≤ yr[d] < EOL[e] of
    #     an entity that may retire (a_x: divest_cost_p)
    #   bw = pd_branch_weight, factor 1 when absent
    # The salvage value of existing capacity, received at its end of life, as a level yearly income over its
    # service years from the run start, the same credit as a built plant's (net of the retirements of earlier
    # solves; a retirement in this solve gives its credits back through divest_cost_p).  A constant, added only with
    # include_existing_fixed_cost, multiplied by scale_the_objective like every other objective term.
    exist = _capacity.existing_not_retired(b).rename({"value": "exist"})
    frame, credit_frame = align_join_dtypes(exist, credit.frame.rename({"value": "credit"}), ["e", "d"])
    frame = frame.join(credit_frame, on=["e", "d"], how="inner")
    if d.pd_branch_weight is not None:
        frame, weights = align_join_dtypes(frame, d.pd_branch_weight.frame.rename({"value": "pdbw"}), ["d"])
        frame = (frame.join(weights, on="d", how="inner")
                      .with_columns(contrib=pl.col("exist") * pl.col("credit") * pl.col("pdbw")))
    else:
        frame = frame.with_columns(contrib=pl.col("exist") * pl.col("credit"))
    constant = frame.sort("e", "d")["contrib"].sum()  # fixed summation order (the joins leave it unspecified)
    if constant:
        b.m.add_obj_constant(-float(constant) * b.scale_the_objective)


def _obj_existing_replacement(b: _Build) -> None:
    """The replacement of existing reinvest_automatic capacity at its end of life, an objective constant
    (opt-in)."""
    d = b.d
    price = getattr(d, "ed_existing_replacement", None)
    if not (b.include_existing_fixed_cost and price is not None and b.cap.existing_cohort_size_e is not None):
        return

    # ─── OBJ. ─── existing_replacement ───────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(e,d) ∈ price} (pre_existing[e,d] − retired_carried[e,d]) · a_b[g(e,d)]·inflation_inv[d] · bw[d]
    #   price = ed_existing_replacement[e,d] = a_b[g(e,d)]·inflation_inv[d], g(e,d) the replacement of e's existing
    #     capacity in service in d (a generation g ≥ 2 of the existing reinvest_automatic chain: online at the end of
    #     life of the one before, round_end_of_life(y_on + lifetime_technical[e, d_on(g)]), until the horizon end)
    #   a_b[g] = a_cap[g] − a_salv[g] (invest_cost_p) of one plant built in d_on(g) (on d's lineage in a stochastic
    #     fan), its salvage at its end of life; 0 without an invest_cost (and no salvage value)
    #   pre_existing, retired_carried as in existing_salvage; bw = pd_branch_weight, factor 1 when absent
    # The replacement of existing capacity at its end of life, at the invest cost (and the other values) of the period
    # the replacement comes online: no decision changes it except retiring the capacity first, whose retirement
    # coefficient (divest_cost_p) gives the replacement back whether or not this constant is added.  A constant,
    # added only with include_existing_fixed_cost, multiplied by scale_the_objective like every other objective term.
    exist = _capacity.existing_not_retired(b).rename({"value": "exist"})
    frame, price_frame = align_join_dtypes(exist, price.frame.rename({"value": "price"}), ["e", "d"])
    frame = frame.join(price_frame, on=["e", "d"], how="inner")
    if d.pd_branch_weight is not None:
        frame, weights = align_join_dtypes(frame, d.pd_branch_weight.frame.rename({"value": "pdbw"}), ["d"])
        frame = (frame.join(weights, on="d", how="inner")
                      .with_columns(contrib=pl.col("exist") * pl.col("price") * pl.col("pdbw")))
    else:
        frame = frame.with_columns(contrib=pl.col("exist") * pl.col("price"))
    constant = frame.sort("e", "d")["contrib"].sum()  # fixed summation order (the joins leave it unspecified)
    if constant:
        b.m.add_obj_constant(float(constant) * b.scale_the_objective)


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
      limited to M_s · (existing_count + Σ v_invest − Σ v_divest), M_s = p_indirect_input_cap · a
      (input_share_max times the full-load input per unit of capacity; with the default share 1 the
      input alone could run the unit at full output, so the limit is a loose bound).  ``a`` is
      ``x(availability)`` (:func:`flextool.engine.model._online._fuel_share`), the input drawn at derated
      full output per unit of full-load input — the min-load section is drawn per unit online, so plain
      availability would cap the derated full output of a min_load_efficiency unit (fed the ``__av``
      column computed below, the section credited only where the unit has an online variable; #102).
      Processes without invest/divest variables in this solve (``inv_procs``) have a constant capacity:
      M_s · existing_count becomes a v_flow upper bound and the arc gets no maxFlow row.  Processes
      with them keep the row, their invest/divest terms carrying M_s.
    * **capped**: direct arcs and indirect output arcs:
      v_flow ≤ coef · availability · (existing_count + Σ v_invest − Σ v_divest), RHS =
      existing_count · coef · availability.  ``p_flow_upper_existing`` is filled with 0 for greenfield
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

    # ── input: M × existing/us, M = input cap × a, a = x(availability) ──
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
        in_lf = _utf8_keys(_with_availability(in_lf), ["d", "t"])
        if avail is not None:
            # a = x(availability) on every indirect unit's input arc (not only online units, #102); feed
            # _fuel_share the __av already joined above as its level, so the bound and the online rows
            # (_online._input_availability_share) read the same availability.  The section is credited
            # only to units with an online variable (_online.add_variables): a non-online
            # min_load_efficiency unit's balance equation applies no section.
            online_frames = [f for f in (getattr(d, "process_online_linear", None),
                                         getattr(d, "process_online_integer", None))
                             if _has_rows(f)]
            online_p = (pl.concat([_utf8_keys(f.lazy().select("p"), ["p"]).select("_p")
                                   for f in online_frames]).unique()
                        if online_frames else pl.LazyFrame(schema={"_p": pl.Utf8}))
            levels = in_lf.select("_p", "_d", "_t", pl.col("__av").alias("__y")).unique()
            x = _online._fuel_share(d, levels, online_p)
            in_lf = (in_lf
                     .join(x, on=["_p", "_d", "_t"], how="left")
                     .with_columns(__av=pl.coalesce(pl.col("__x"), pl.col("__av"))))
        in_frame = (in_lf
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
