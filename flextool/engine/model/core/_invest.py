"""Investment and retirement limits of processes and storage nodes."""

import polars as pl
from polar_high import Sum, Where, Param
from flextool.engine.model import _capacity
from flextool.engine.model import _lead_time
from flextool.engine.model import _retire_built
from flextool.engine.input._axis_enums import align_join_dtypes, rename_to_axis
from flextool.engine.model._common import (
    _entity_rows,
    _has_rows,
    _var_view,
    chain_prior,
    check_max_cap,
    leaf_rows,
    minus_prior,
)
from flextool.engine.input._param_shapes import drop_no_limit_rows, without_no_limit_rows
from flextool.engine.model.core._context import _Build


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
