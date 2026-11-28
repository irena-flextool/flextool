"""Capacity limits on process flows (maxFlow, maxOutputSum) and the maxFlow capacity terms."""

import polars as pl
from polar_high import Sum, Where, Param
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _online
from flextool.engine.model._common import _availability_factor, _has_rows, _indirect_sum_arcs, _utf8_keys
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.model.core._context import _Build


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
    # On indirect-unit input arcs M = input_share_max · full-load input · a: a = availability for a unit with an
    # online variable (the share of its units available, each drawing up to its full-load input; maxOnline),
    # else a = x(availability) = slope·W(a₀) / (slope·W(1)) (_online._fuel_share without a section: the balance
    # of a unit without an online variable has none; a₀ = availability, W(y) the output-side fuel width filled to
    # y; a piecewise_linear unit's curve in place of slope·W); a ≥ availability (never
    # tighter than the physical derated input), = availability when every output shares one
    # conversion_flow_coeff (#102).
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
      availability for a unit with an online variable (availability is the share of its units available,
      each drawing up to its full-load input) and otherwise ``x(availability)``
      (:func:`flextool.engine.model._online._fuel_share` without a section), the input drawn at derated
      full output per unit of full-load input (fed the ``__av`` column computed below; #102).
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

    # ── input: M × existing/us, M = input cap × a, a = availability or x(availability) ──
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
            # a = x(availability) on the input arcs of indirect units without an online variable (#102), fed
            # the __av already joined above as its level; their balance equation applies no section, so none
            # is credited.  A unit with an online variable keeps a = availability: availability is the share
            # of its units available (maxOnline), each drawing up to its full-load input.
            online_frames = [f for f in (getattr(d, "process_online_linear", None),
                                         getattr(d, "process_online_integer", None))
                             if _has_rows(f)]
            online_p = (pl.concat([_utf8_keys(f.lazy().select("p"), ["p"]).select("_p")
                                   for f in online_frames]).unique()
                        if online_frames else pl.LazyFrame(schema={"_p": pl.Utf8}))
            levels = (in_lf.select("_p", "_d", "_t", pl.col("__av").alias("__y")).unique()
                      .join(online_p, on="_p", how="anti"))
            x = _online._fuel_share(d, levels, online_p.head(0))
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
