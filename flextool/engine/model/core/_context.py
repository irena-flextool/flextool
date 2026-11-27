"""The build context of the FlexTool LP: :class:`_Build`, the data fields each feature requires and the data checks."""

import logging
from dataclasses import dataclass, field
from typing import Any
import polars as pl
from polar_high import Param
from polar_high.engine import Var
from flextool.engine.input._solve_state import FlexToolConfigError


_LOG = logging.getLogger(__name__)


# Objective cutoff relative to the largest |cost| (polar-high
# ``Problem.obj_coef_zero_rel_threshold``, owner decision Q-gate-A).  About 100
# machine epsilons: it removes float residue (a 1e-17 cost next to O(1) costs)
# while the widest legitimate cost spread in the LP-snapshot fixtures is
# min/max = 1.4e-12 (11.8 decades).  Relative, not ``small_number_threshold``:
# real costs go down to 4e-5 (slack penalties) and the objective may be scaled.
OBJ_COEF_ZERO_REL_THRESHOLD: float = 1e-14


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
    # The path / path_at_step families (_period_rows.PERIOD_ROW_KINDS) whose rows are built per path (base period
    # + branch period); G1 accepts their rows on branch periods.
    path_converted: set = field(default_factory=set)
