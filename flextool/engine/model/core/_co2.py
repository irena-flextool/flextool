"""CO2 emission caps per period and over all periods."""

import polars as pl
from polar_high import Sum, Where, Param
from flextool.engine.input._axis_enums import join_aligned
from flextool.engine.model._common import (
    _has_rows,
    branch_periods_inherit,
    d_period_path,
    leaf_rows,
    on_paths,
    path_rows,
)
from flextool.engine.model.core._context import _Build
from flextool.engine.model.core._conversion import _eff_fuel_input, _startup_draw


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
    #   split realization roll (d_period_path): the row on a branch period p_f sums the cells d ∈ {p, p_f} (p its
    #   base period: the jump, then the tail), each at its own weight; no row on p
    #   f_d_k[d] = (realized hours of α(d) from earlier solves + this solve's hours of d) / complete hours of α(d)
    #     (+ the jump hours of p for a branch period d = p_f of a split realization roll: the path's hours)
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
    # Path rows (split realization roll, d_period_path): the row of a branch period p_f sums the cells of its base
    # period p (the jump) and of p_f (the tail), each at its own w; p has no row.  f_d_k[p_f] counts the jump
    # hours, so the RHS is the path's budget.
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
    over = d.group_d_co2_capped
    if (path := d_period_path(b, "co2_max_period")) is not None:
        lhs = {key: on_paths(term, path) for key, term in lhs.items()}
        over = path_rows(over, path)
    m.add_cstr(
        "co2_max_period",
        over      = over,
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"cap": _co2_period_cap_with_carry(d)},
    )


def _co2_total_share(d) -> "pl.DataFrame | None":
    """``[d, share]`` for the periods that the solves of a chain split (item 1042), None when there is none.

    Each solve's ``period_share`` is its own complete part of a period, so a chain whose solves each hold part of
    one period (s1 on 3 h, s2 on the next 9 h, single or rolling) annualised each part as if it were the whole
    period, and co2_max_total weighed the parts' tonnes differently.  For a period with hours that earlier solves
    realized (``p_co2_hours_before_solve``) the row uses share* = period_share + those hours / 8760: the period as
    known to the solve, the earlier parts and its own.  Every hour of the period has one weight in the row, the
    realized tonnes included (:func:`_co2_total_cap_with_carry`), and its share reached is counted against share*
    (``_co2_carry.split_f_d_k``), so the budget is split by hour weight.  A single solve, and a rolling solve that
    holds its periods alone, have no such period: nothing changes there."""
    from flextool.engine.preprocess._co2_carry import split_share

    before = getattr(d, "p_co2_hours_before_solve", None)
    if before is None or before.frame.height == 0 or d.p_period_share is None:
        return None
    return split_share(d.p_period_share.frame, before.frame).rename({"value": "share"})


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
        if (split := _co2_total_share(d)) is not None:
            rows = _co2_split_represented(rows, d.p_co2_cum_realized_tonnes.frame, split, d)
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


def _co2_split_represented(rows: pl.DataFrame, tonnes: pl.DataFrame, split: pl.DataFrame, d) -> pl.DataFrame:
    """``rows`` ``[g, d, value]`` (the represented realized carry) with each split period's value the period's
    realized tonnes × years_represented / share* (:func:`_co2_total_share`): one weight for every hour of the
    period.  Shared by the row (this solve's carry) and the extraction (the carrier it hands on)."""
    years = getattr(d, "p_years_represented_d", None)
    factor = split.with_columns(years=pl.lit(1.0)) if years is None else join_aligned(
        split, years.frame.select("d", years="value"), on="d", how="left").with_columns(pl.col("years").fill_null(1.0))
    redone = join_aligned(tonnes.select("g", "d", tonnes="value"), factor, on="d", how="inner").select(
        "g", "d", rep=pl.col("tonnes") * pl.col("years") / pl.col("share"))
    return (join_aligned(rows, redone, on=["g", "d"], how="left")
            .with_columns(value=pl.when(pl.col("rep").is_null()).then(pl.col("value")).otherwise(pl.col("rep")))
            .drop("rep"))


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
    #   ∀ g ∈ group_co2_max_total (, leaf);   w_y = step_duration · timestep_weight / share_eff · years_represented[d]
    #   R_rep[g,d] = realized tonnes of d from earlier rolls / solves · years_represented[d] / period_share[d],
    #   with the share and years of the solve that realized them (the w_y its own row gave them)
    #   share_eff[d] = period_share[d], except for a period with hours that earlier solves of the chain realized
    #   (H_before[d] > 0, item 1042): share*[d] = period_share[d] + H_before[d] / 8760, its R_rep the period's realized
    #   tonnes · years_represented[d] / share*[d] and its share reached (realized h + this roll's h) / (share*·8760)
    #   in horizon_fraction_reached
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
    split = _co2_total_share(d)
    share = d.p_period_share if split is None else Param(("d",), join_aligned(
        d.p_period_share.frame, split, on="d", how="left").select(
        "d", value=pl.when(pl.col("share").is_null()).then(pl.col("value")).otherwise(pl.col("share"))))
    lhs: dict = {}
    if b.has_co2_cap_total_eff:
        for key, term in _eff_fuel_input(b, _leafify(d.flow_from_co2_capped_total)):
            lhs[f"emissions_{key}"] = Sum(
                term
                * d.p_co2_content
                * d.p_step_duration * d.p_timestep_weight / share
                * d.p_years_represented_d,
                over=("p", "source", "sink", "c", "d", "t"),
            )
    if b.has_co2_cap_total_noEff:
        lhs["emissions_noEff"] = Sum(
            Where(v_flow * d.p_unitsize, _leafify(d.flow_from_co2_capped_total_noEff))
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / share
            * d.p_years_represented_d,
            over=("p", "source", "sink", "c", "d", "t"),
        )
    if _has_rows(getattr(d, "flow_to_co2_capped_total", None)):
        lhs["removal"] = -Sum(
            Where(v_flow * d.p_unitsize, _leafify(d.flow_to_co2_capped_total))
            * d.p_co2_content
            * d.p_step_duration * d.p_timestep_weight / share
            * d.p_years_represented_d,
            over=("p", "source", "sink", "c", "d", "t"),
        )
    if b.has_startup_draw:
        for key, term in _startup_draw(b, _leafify(getattr(d, "startup_from_co2_capped_total", None))):
            lhs[f"emissions_startup_{key}"] = Sum(
                term * d.p_co2_content * d.p_timestep_weight / share * d.p_years_represented_d,
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
