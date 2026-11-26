"""Preparation of the build: problem settings, feature flags, data checks, warnings and shared index sets."""

import polars as pl
from polar_high import Param
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _commodity_ladder
from flextool.engine.model import _cumulative_invest
from flextool.engine.model import _dc_power_flow
from flextool.engine.model import _group_slack
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.model import _self_discharge
from flextool.engine.input._axis_enums import join_aligned, schema_dtype
from flextool.engine.model._common import _has_rows
from flextool.engine.model._pdt_join import (
    compute_pss_dt,
    compute_nodeBalance_dt,
    compute_nodeState_dt,
    compute_nodeState_rp_block_first_dt,
    compute_process_indirect_dt,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model.core._context import (
    ALWAYS,
    CAPACITY_MARGIN,
    CO2_CAP,
    CO2_CAP_TOTAL,
    DIVEST,
    DIVEST_BUILT,
    DIVEST_N,
    EXTEND,
    INDIRECT,
    INVEST,
    MINLOAD_EFF,
    MIN_LOAD,
    OBJ_COEF_ZERO_REL_THRESHOLD,
    ONLINE,
    PIECEWISE,
    PROCESSES,
    PROFILES,
    RAMP,
    RAMP_COST,
    STARTUP_COST_INTEGER,
    STARTUP_COST_LINEAR,
    STORAGE,
    USER_CSTR,
    _Build,
    _INFLATION_OP_GAP,
    _LOG,
    _PERIOD_SHARE_GAP,
    _STEP_DURATION_GAP,
    _TIMESTEP_WEIGHT_GAP,
    _check,
    _check_covers_dt,
    _check_covers_dt_steps,
)
from flextool.engine.model.core._storage_links import _nested_fixed_end, _start_decision_periods
from flextool.engine.model.core._flows import _max_flow_rhs
from flextool.engine.model.core._conversion import _validate_startup_inputs
from flextool.engine.model.core._user_constraints import _collected


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
    if b.nodeBalance_dt is None:
        b.nodeBalance_dt = _no_balance_dt(d)
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


def _no_balance_dt(d) -> pl.DataFrame:
    """The empty ``(n, d, t)`` frame of a build without per-step balance nodes (a Benders master holding only
    flows between regions, or a model with only ``balance_within_period`` nodes): the balance slacks and
    ``nodeBalance_eq`` are then built with no columns and rows.  Keeps the axis dtypes of ``nodeBalance`` and
    ``dt``.  A user's model without any balance node is refused when it is loaded
    (``flex_data._load._validate_has_balance``)."""
    nb = d.nodeBalance
    n = (nb.select("n").head(0) if nb is not None
         else pl.DataFrame(schema={"n": schema_dtype(getattr(d, "_axis_enums", None), "n")}))
    return n.join(d.dt.select("d", "t").head(0), how="cross")


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
        how = (" (or, on a unit, a minimum_time_method)" if any(p in units for p, _ in missing) else "")
        raise FlexToolConfigError(
            f"constraint_online_coeff needs online variables, but these have none: {shown}. Set a startup_method "
            f"other than no_startup{how}, or remove the coefficient.")


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
    (``preprocess.flex_data._load._reference_start_as_fix_start``), so the two agree there."""
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
