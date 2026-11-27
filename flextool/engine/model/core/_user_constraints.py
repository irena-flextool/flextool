"""User-defined constraints (process_constraint_*)."""

import polars as pl
from polar_high import Sum, Where, Param
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _lead_time
from flextool.engine.model import _online
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.input._axis_enums import align_join_dtypes, join_aligned, rename_to_axis
from flextool.engine.model._common import (
    _entity_rows,
    _has_rows,
    _var_view,
    PATH_D,
    d_period_path,
    on_paths,
    path_rows,
    path_values,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model.core._context import _Build, _check_covers_dt
from flextool.engine.model.core._objective import _sum_terms


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
    # The online term counts the units online (of unitsize) in each step; whole units with startup_method integer
    # or integer_fix_in_uc, continuous with linear or linear_fix_in_uc.  v_online is the variable of the process's
    # UC class (v_online_linear / v_online_integer).
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
    #   split realization roll (d_period_path): the row on a branch period p_f sums the cells d ∈ {p, p_f} (p its
    #   base period: the jump, then the tail), each at its own weight; no row on p; H[p_f]
    #   and window_fraction[p_f] are the path's (Σ over p, p_f), the capacity terms stay on p_f
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
    # Path rows are unreachable while _recursive_solve._check_period_total_rolls holds: a split roll never realizes
    # a period whole, so the run builds no period_total row in it.  Converted so that G1 never meets an unconverted one.
    path = d_period_path(b, *(name for _rows, _sense, name in rows_by_sense))
    if path is not None:
        hours = Param(("d",), path_values(hours.frame, (), path))
    # The periods the solve holds whole (window_fraction 1, of the path under a split roll); the others have no row.
    whole = None
    if d.p_window_fraction is not None:
        frac = _collected(d.p_window_fraction.frame)
        if path is not None:
            frac = path_values(frac.select("d", pl.col("value").cast(pl.Float64)), (), path)
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
    capacity_at = len(terms) if capacity_terms else None
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
    if path is not None:
        # The time sums of each path's cells on its row period (capacity terms are per row period already); a row
        # is live when a cell of its path is.
        terms = [term if i == capacity_at else on_paths(term, path) for i, term in enumerate(terms)]
        rows_of: dict = {}
        for cell, row in path.select(pl.col("d", PATH_D).cast(pl.Utf8)).rows():
            rows_of.setdefault(cell, []).append(row)
        if "*" not in live:
            live = {f"{cn}\x1f{row}" for cn, cell in (k.split("\x1f") for k in live) for row in rows_of.get(cell, ())}
    empty_rows: list[tuple[pl.DataFrame, str]] = []
    for rows, sense, name in rows_by_sense:
        if path is not None:
            rows = path_rows(rows, path)
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
