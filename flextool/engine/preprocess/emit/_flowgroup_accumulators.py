"""The realized flow of each flow group with a cumulative flow limit, carried across rolls and solves (D19).

``{max,min}Cumulative_flow_{period,solve}`` (``model/_cumulative_invest.py``) allow a roll the limit times the hours
realized so far plus its own hours, minus the flow realized so far.  This module builds that history after a solve:
per (group, period) the net flow into the group over the solve's realized steps (``flex_data.realized_dispatch``),
added to the prior carrier (``K.HANDOFF_CUMULATIVE_FLOWGROUP``, populated by
``_provider_translators.translate_handoff_to_provider``).

The net flow F[g,d,t] is the one of ``_cumulative_invest._group_flow``, evaluated on the solution:

    F = Σ v_flow·us                                  on an arc into a member node (flow_to_n)
      − Σ (v_flow·us·slope + v_online·section·us)    on an eff arc out of a member node (section: min_load_efficiency)
      − Σ v_flow·us                                  on a noEff arc out of a member node
      − Σ v_flow_back·us                             at the sink of a one-variable two-way arc (twoway1var_arcs)
      + Σ v_flow_back·us·σ                           at its source (σ = slope on an eff arc, else 1)

each at the (g, p, n) of ``group_process_node``.  The carrier columns are ``flow`` = Σ_t F·sd·tw and ``hours`` =
Σ_t sd·tw over the realized steps of each period (the period rows' measure), and ``weight`` =
years_represented[d] / period_share[d] of the latest solve that realized steps of d (the solve row's raw weight,
before its normalisation r_max).  One weight per period: the solve row weighs every hour of a period alike, also when
the solves of a chain hold unequal parts of it (each solve's period_share is its own part), so it stays the
weighted sum of the period rows.  Every period the solve realizes gets a row for every group, also with
no flow, so the hours count.

``realized_dispatch`` holds no cell of a stochastic fan: only the jump is carried, and the jump of a split
realization roll under the period label.  A ``separate_dispatch`` parent passes the prior carrier on
(``is_dispatch_sibling_parent``): its realized-dispatch sibling re-solves the same window and carries it.
"""
from __future__ import annotations

import polars as pl

from flextool.engine.input._axis_enums import alias_to_axis, join_aligned, rename_to_axis, schema_dtype
from flextool.engine.preprocess.emit._co2_accumulators import _with_represented

#: The carrier's columns (``SolveHandoff.cumulative_flowgroup``).
CARRIER_COLUMNS = ("group", "period", "flow", "hours", "weight")
_VALUE_COLUMNS = CARRIER_COLUMNS[2:]
_SUMMED = ("flow", "hours")
_ARC = ["p", "source", "sink"]


def _schema() -> dict:
    return {"group": schema_dtype(None, "group"), "period": schema_dtype(None, "period"),
            **{c: pl.Float64 for c in _VALUE_COLUMNS}}


def flowgroup_limited_groups(flex_data) -> pl.DataFrame:
    """``[g]``: the flow groups with a finite cumulative flow limit in the solve (constant or period map, max or
    min), sorted."""
    frames = [param.frame.select("g") for name in ("p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
                                                    "pd_max_cumulative_flow", "pd_min_cumulative_flow")
              if (param := getattr(flex_data, name, None)) is not None and param.frame.height > 0]
    if not frames:
        return pl.DataFrame(schema={"g": schema_dtype(None, "g")})
    return pl.concat(frames, how="vertical").unique().sort("g")


def _param(flex_data, name: str, alias: str) -> pl.DataFrame | None:
    param = getattr(flex_data, name, None)
    if param is None or param.frame.height == 0:
        return None
    return param.frame.rename({"value": alias}).with_columns(pl.col(alias).cast(pl.Float64))


def _with(frame: pl.DataFrame, factor: pl.DataFrame | None, on: list[str], col: str) -> pl.DataFrame:
    """*frame* with the factor column *col* of *factor* joined on *on* (1.0 where it has no row or no factor)."""
    if factor is None:
        return frame.with_columns(pl.lit(1.0).alias(col))
    return join_aligned(frame, factor.select(*on, col), on=on, how="left").with_columns(pl.col(col).fill_null(1.0))


def _var(flex_data, sol, name: str, realized_dt: pl.DataFrame) -> pl.DataFrame | None:
    """The values of variable *name* on the fine realized steps (``var_values``), Float64; None when absent."""
    from flextool.engine.output._coarse_flows import var_values

    frame = var_values(sol, flex_data, name)
    if frame is None or frame.height == 0:
        return None
    return (join_aligned(frame, realized_dt, on=["d", "t"], how="semi")
            .with_columns(pl.col("value").cast(pl.Float64)))


def _realized_net_flow(flex_data, sol, members: pl.DataFrame, realized_dt: pl.DataFrame) -> list[pl.DataFrame]:
    """The terms of F (MW) at the realized steps, ``[g, d, t, value]`` frames, one row per arc term (not summed)."""
    from flextool.engine.model.core._prepare import _two_way_one_var_arcs

    us = _param(flex_data, "p_unitsize", "us")
    slope = _param(flex_data, "p_slope", "slope")
    pss_eff = getattr(flex_data, "process_source_sink_eff", None)
    pss_noeff = getattr(flex_data, "process_source_sink_noEff", None)
    parts: list[pl.DataFrame] = []
    flow = _var(flex_data, sol, "v_flow", realized_dt)
    if flow is not None:
        flow = _with(flow, us, ["p"], "us")
        to_n = getattr(flex_data, "flow_to_n", None)
        if to_n is not None and to_n.height > 0:
            arcs = join_aligned(to_n.select(*_ARC, "n"), members, on=["p", "n"], how="inner")
            parts.append(join_aligned(flow, arcs, on=_ARC, how="inner")
                         .select("g", "d", "t", value=pl.col("value") * pl.col("us")))
        from_n = getattr(flex_data, "flow_from_n", None)
        if from_n is not None and from_n.height > 0:
            out_arcs = join_aligned(from_n.select(*_ARC, "n"), members, on=["p", "n"], how="inner")
            if pss_eff is not None and pss_eff.height > 0:
                eff = join_aligned(out_arcs, pss_eff.select(_ARC), on=_ARC, how="semi")
                parts.append(_with(join_aligned(flow, eff, on=_ARC, how="inner"), slope, ["p", "d", "t"], "slope")
                             .select("g", "d", "t", value=-(pl.col("value") * pl.col("us") * pl.col("slope"))))
                parts.extend(_section_terms(flex_data, sol, eff, realized_dt, us))
            if pss_noeff is not None and pss_noeff.height > 0:
                noeff = join_aligned(out_arcs, pss_noeff.select(_ARC), on=_ARC, how="semi")
                parts.append(join_aligned(flow, noeff, on=_ARC, how="inner")
                             .select("g", "d", "t", value=-(pl.col("value") * pl.col("us"))))
    back = _var(flex_data, sol, "v_flow_back", realized_dt)
    two_way = _two_way_one_var_arcs(flex_data)
    if back is not None and two_way is not None and two_way.height > 0:
        back = _with(join_aligned(back, two_way.select(_ARC), on=_ARC, how="semi"), us, ["p"], "us")
        at_sink = join_aligned(back, members, left_on=["p", "sink"], right_on=["p", "n"], how="inner")
        parts.append(at_sink.select("g", "d", "t", value=-(pl.col("value") * pl.col("us"))))
        at_source = join_aligned(back, members, left_on=["p", "source"], right_on=["p", "n"], how="inner")
        if pss_eff is not None and pss_eff.height > 0:
            sloped = join_aligned(at_source, pss_eff.select(_ARC), on=_ARC, how="semi")
            plain = join_aligned(at_source, pss_eff.select(_ARC), on=_ARC, how="anti")
            parts.append(_with(sloped, slope, ["p", "d", "t"], "slope")
                         .select("g", "d", "t", value=pl.col("value") * pl.col("us") * pl.col("slope")))
        else:
            plain = at_source
        parts.append(plain.select("g", "d", "t", value=pl.col("value") * pl.col("us")))
    return [p for p in parts if p.height > 0]


def _section_terms(flex_data, sol, eff_arcs: pl.DataFrame, realized_dt: pl.DataFrame,
                   us: pl.DataFrame | None) -> list[pl.DataFrame]:
    """``− v_online·section·us`` of the min_load_efficiency units on the member eff arcs (``eff_arcs`` with g): one
    term per arc, as ``_eff_fuel_input``'s section terms are restricted to each arc."""
    pmle = getattr(flex_data, "process_min_load_eff", None)
    section = _param(flex_data, "p_section", "section")
    if pmle is None or pmle.height == 0 or section is None:
        return []
    arcs = join_aligned(eff_arcs, pmle.select("p").unique(), on="p", how="semi").select("g", *_ARC)
    if arcs.height == 0:
        return []
    out = []
    for name in ("v_online_linear", "v_online_integer"):
        online = _var(flex_data, sol, name, realized_dt)
        if online is None:
            continue
        online = join_aligned(online.select("p", "d", "t", "value"), arcs, on="p", how="inner")
        online = _with(join_aligned(online, section, on=["p", "d", "t"], how="inner"), us, ["p"], "us")
        out.append(online.select("g", "d", "t",
                                 value=-(pl.col("value") * pl.col("section") * pl.col("us"))))
    return out


def _prior_frame(prior: pl.DataFrame | None, groups: pl.DataFrame) -> pl.DataFrame:
    if prior is None or prior.height == 0:
        return pl.DataFrame(schema=_schema())
    prior = prior.select(alias_to_axis("group", "group"), alias_to_axis("period", "period"),
                         *(pl.col(c).cast(pl.Float64) for c in _VALUE_COLUMNS))
    return join_aligned(prior, groups.select(alias_to_axis("g", "group")), on="group", how="semi")


def compute_flowgroup_rolling_accumulator(flex_data, sol, *, prior: pl.DataFrame | None = None,
                                          is_dispatch_sibling_parent: bool = False) -> pl.DataFrame:
    """The ``cumulative_flowgroup`` carrier after this solve: the prior carrier plus this solve's realized flow
    and hours per (group, period), columns :data:`CARRIER_COLUMNS`, sorted.  Empty without a limited flow group."""
    groups = flowgroup_limited_groups(flex_data)
    if groups.height == 0:
        return pl.DataFrame(schema=_schema())
    prior_f = _prior_frame(prior, groups)
    realized = getattr(flex_data, "realized_dispatch", None)
    sd = _param(flex_data, "p_step_duration", "sd")
    if is_dispatch_sibling_parent or sol is None or realized is None or realized.height == 0 or sd is None:
        return prior_f.sort("group", "period")
    realized_dt = realized.pipe(rename_to_axis, {"period": "d", "step": "t"}).select("d", "t").unique()
    weight = _with(join_aligned(sd, realized_dt, on=["d", "t"], how="semi"),
                   _param(flex_data, "p_timestep_weight", "tw"), ["d", "t"], "tw")
    weight = weight.select("d", "t", w=pl.col("sd") * pl.col("tw"))
    gpn = getattr(flex_data, "group_process_node", None)
    members = (join_aligned(gpn.select("g", "p", "n"), groups, on="g", how="semi").unique()
               if gpn is not None and gpn.height > 0 else None)
    terms = _realized_net_flow(flex_data, sol, members, realized_dt) if members is not None else []
    hours = weight.sort("d", "t").group_by("d", maintain_order=True).agg(pl.col("w").sum().alias("hours"))
    this = groups.join(hours, how="cross")
    if terms:
        flow = (join_aligned(pl.concat(terms, how="vertical"), weight, on=["d", "t"], how="inner")
                .sort("g", "d", "t")  # fixed summation order
                .group_by(["g", "d"], maintain_order=True)
                .agg((pl.col("value") * pl.col("w")).sum().alias("flow")))
        this = join_aligned(this, flow, on=["g", "d"], how="left")
    else:
        this = this.with_columns(flow=pl.lit(None, dtype=pl.Float64))
    this = this.select(alias_to_axis("g", "group"), alias_to_axis("d", "period"),
                       value=pl.col("flow").fill_null(0.0), hours="hours")
    weight = _with_represented(this.select("group", "period", value=pl.lit(1.0)), flex_data)["represented"]
    if weight.is_null().any():
        raise ValueError("cumulative_flowgroup: a realized period of the solve has no period_share; the carried "
                         "flow-group history cannot be weighted for max/min_cumulative_flow.")
    this = this.select("group", "period", flow="value", hours="hours", weight=weight)
    # A period this solve realized takes its weight; the prior weight stays for the other periods.
    combined = (pl.concat([prior_f, this], how="vertical")
                .group_by(["group", "period"], maintain_order=True)
                .agg(*(pl.col(c).sum() for c in _SUMMED), pl.col("weight").last()))
    return combined.sort("group", "period").select(*CARRIER_COLUMNS)


def flowgroup_handoff(flex_data, sol, prior_handoff, is_dispatch_sibling_parent: bool) -> pl.DataFrame | None:
    """``SolveHandoff.cumulative_flowgroup`` of the solve (``build_handoff_from_solution``): the prior carrier plus
    this solve's realized flow (:func:`compute_flowgroup_rolling_accumulator`); the prior carrier as it is without
    ``flex_data``; None when there is nothing to carry."""
    prior = prior_handoff.cumulative_flowgroup if prior_handoff is not None else None
    if flex_data is None:
        return prior
    carrier = compute_flowgroup_rolling_accumulator(flex_data, sol, prior=prior,
                                                    is_dispatch_sibling_parent=is_dispatch_sibling_parent)
    return carrier if carrier.height > 0 else None


def load_flowgroup_rolling_carry(provider) -> pl.DataFrame | None:
    """``FlexData.flowgroup_realized``: the ``cumulative_flowgroup`` handoff carrier the solve consumes, as
    ``[g, d, flow, hours, weight]`` (Float64 values); None without a prior carrier (the
    first solve of a chain, or no flow group with a cumulative limit).  The consumers are the carry terms of
    ``{max,min}Cumulative_flow_{period,solve}`` (``model/_cumulative_invest.py``)."""
    if provider is None:
        return None
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame

    raw = read_handoff_frame(provider, K.HANDOFF_CUMULATIVE_FLOWGROUP)
    if raw is None or raw.height == 0:
        return None
    return (raw.pipe(rename_to_axis, {"group": "g", "period": "d"})
            .select("g", "d", *(pl.col(c).cast(pl.Float64, strict=False) for c in _VALUE_COLUMNS)))
