"""Conversion of indirect units (conversion_indirect, conversion_piecewise, minInputShare) and the fuel input
and start-up draw that other families read."""

import polars as pl
from polar_high import Sum, Where, Param
from polar_high.engine import Expr
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _delay
from flextool.engine.model._common import _coef_factor, _has_rows, _utf8_keys, _var_view
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model.core._context import _Build


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
            f"{', '.join(missing)}. Set a startup_method other than no_startup (or a minimum_time_method), or set "
            "startup_input_method just_input.")
