"""Input burnt above the part-load curve of the piecewise_linear units (the post-solve check, W5).

A ``piecewise_linear`` unit with two or more lines converts by ``conversion_piecewise``: its weighted input is *at
least* each line of its curve (``model/core._eq_conversion_piecewise``).  The cheapest input lies on the curve, so
the input is above it only where that costs nothing or pays: an input without value (a surplus, a free or
curtailed input), a negative value (a CO2 credit, a negative price), or a reserve or a minimum on the input.  The
check measures, per realized step,

    q[p,d,t]      = Σ_{s ∈ in(p)} v_flow[p,s,p,d,t] · unitsize[p] · conv_in[p,s]  (+ the delayed inputs)
    need[p,d,t]   = max_g ( pw_slope[p,g,d] · x[p,d,t] + pw_intercept[p,g,d] · Y[p,d,t] )
    x[p,d,t]      = Σ_{o ∈ out(p)} v_flow[p,p,o,d,t] · unitsize[p] · conv_out[p,o]
    Y[p,d,t]      = v_online[p,d,t] · unitsize[p]                  (units with online variables)
                  = (existing_count[p,d] + Σ v_invest_p − Σ v_divest_p) · unitsize[p]   (the other units)
    excess[p,d,t] = q − need                                        [MW of weighted input]

with the data and the solution of each solve (:func:`piecewise_input_and_excess`, called by ``read_parameters`` per
solve, so a rolling run's rolls are each measured with their own lines and investments).  The warning
(:func:`warn_piecewise_excess`) counts the realized steps with ``excess > 1e-5 · max(1, q)`` (an interior-point
tolerance does not warn) and gives the largest and the total ``excess · step_duration``.  Single-line units keep
the conversion equality and are not measured.  No result table (design v1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine.preprocess.flex_data import FlexData

_logger = logging.getLogger(__name__)

#: Relative threshold of the excess: ``excess > EXCESS_RTOL · max(1, q)`` warns.
EXCESS_RTOL = 1e-5


def _utf8(frame) -> "pl.DataFrame | None":
    if frame is None:
        return None
    frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return frame.with_columns(*(pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value"))


def _param(param, cols: tuple[str, ...], alias: str) -> "pl.DataFrame | None":
    frame = _utf8(None if param is None else param.frame)
    if frame is None or frame.height == 0 or any(c not in frame.columns for c in cols):
        return None
    return frame.select(*cols, pl.col("value").cast(pl.Float64).alias(alias))


def _value(solution, name: str) -> "pl.DataFrame | None":
    if solution is None or name not in getattr(solution, "_vars", {}):
        return None
    return _utf8(solution.value(name)).with_columns(pl.col("value").cast(pl.Float64))


def _with_coeff(flows: pl.DataFrame, coeff, side: str) -> pl.DataFrame:
    """``flows`` with the ``conversion_flow_coeff`` of the ``side`` (source / sink) arc as ``k`` (default 1)."""
    k = _param(coeff, ("p", side), "k")
    if k is None:
        return flows.with_columns(k=pl.lit(1.0))
    return flows.join(k, on=["p", side], how="left").with_columns(pl.col("k").fill_null(1.0))


def piecewise_input_and_excess(flex_data: "FlexData", solution: "Solution | None") -> "pl.DataFrame | None":
    """``(p, d, t, input, excess)`` [MW of weighted input] of the multi-line piecewise_linear units of one solve
    (the module docstring); None without such a unit or without a solution."""
    pw = _utf8(getattr(flex_data, "process_piecewise", None))
    if pw is None or pw.height == 0:
        return None
    pw = pw.select("p").unique()
    flow = _value(solution, "v_flow")
    slope = _param(getattr(flex_data, "p_piecewise_slope", None), ("p", "seg", "d"), "slope")
    intercept = _param(getattr(flex_data, "p_piecewise_intercept", None), ("p", "seg", "d"), "intercept")
    us = _param(flex_data.p_unitsize, ("p",), "us")
    if flow is None or slope is None or intercept is None or us is None:
        return None
    flow = flow.join(pw, on="p", how="semi").join(us, on="p", how="inner").with_columns(
        mw=pl.col("value") * pl.col("us"))
    keys = ["p", "source", "sink"]

    # x: the weighted output.
    out_arcs = _utf8(flex_data.process_output_flows).select(keys).join(pw, on="p", how="semi")
    out = _with_coeff(flow.join(out_arcs, on=keys, how="semi"), flex_data.p_process_sink_conversion_flow_coeff,
                      "sink")
    x = out.group_by("p", "d", "t").agg((pl.col("mw") * pl.col("k")).sum().alias("x"))

    # q: the weighted input; a delayed unit's inputs at the earlier source-side steps.
    in_arcs = _utf8(flex_data.process_input_flows).select(keys).join(pw, on="p", how="semi")
    delayed = _utf8(getattr(flex_data, "process_delayed", None))
    if delayed is not None and delayed.height > 0:
        in_arcs = in_arcs.join(delayed.select("p"), on="p", how="anti")
    inp = _with_coeff(flow.join(in_arcs, on=keys, how="semi"), flex_data.p_process_source_conversion_flow_coeff,
                      "source")
    pieces = [inp.group_by("p", "d", "t").agg((pl.col("mw") * pl.col("k")).sum().alias("q"))]
    if delayed is not None and delayed.height > 0:
        pieces.append(_delayed_input(flex_data, flow, delayed.select("p").join(pw, on="p", how="semi")))
    pieces = [f for f in pieces if f is not None and f.height > 0]
    if not pieces:
        return None
    q = pl.concat(pieces).group_by("p", "d", "t").agg(pl.col("q").sum())

    # Y: the units the curve is scaled by, in MW.
    y_parts: list[pl.DataFrame] = []
    online_p: list[pl.DataFrame] = []
    for set_name, var in (("process_online_linear", "v_online_linear"), ("process_online_integer", "v_online_integer")):
        members = _utf8(getattr(flex_data, set_name, None))
        if members is None or members.height == 0:
            continue
        members = members.select("p").unique().join(pw, on="p", how="semi")
        if members.height == 0:
            continue
        online_p.append(members)
        val = _value(solution, var)
        if val is None:
            continue
        y_parts.append(val.join(members, on="p", how="semi").join(us, on="p", how="inner")
                       .select("p", "d", "t", y=pl.col("value") * pl.col("us")))
    built = pw.join(pl.concat(online_p).unique(), on="p", how="anti") if online_p else pw
    if built.height > 0:
        cap = x.select("p", "d", "t").join(built, on="p", how="semi")
        exist = _param(getattr(flex_data, "p_process_existing_count", None), ("p", "d"), "exist")
        cap = (cap.join(exist, on=["p", "d"], how="left") if exist is not None
               else cap.with_columns(exist=pl.lit(None, dtype=pl.Float64)))
        for var, edd, d_col, alias in (("v_invest_p", "edd_invest_set", "d_invest", "inv"),
                                       ("v_divest_p", "edd_divest_active", "d_divest", "div")):
            units = _invest_units(solution, var, getattr(flex_data, edd, None), d_col)
            cap = (cap.join(units.rename({"value": alias}), on=["p", "d"], how="left") if units is not None
                   else cap.with_columns(pl.lit(None, dtype=pl.Float64).alias(alias)))
        y_parts.append(cap.join(us, on="p", how="inner").select(
            "p", "d", "t", y=(pl.col("exist").fill_null(0.0) + pl.col("inv").fill_null(0.0)
                              - pl.col("div").fill_null(0.0)) * pl.col("us")))
    if not y_parts:
        return None
    y = pl.concat(y_parts)

    lines = slope.join(intercept, on=["p", "seg", "d"], how="inner")
    need = (x.join(y, on=["p", "d", "t"], how="inner")
            .join(lines, on=["p", "d"], how="inner")
            .group_by("p", "d", "t")
            .agg((pl.col("slope") * pl.col("x") + pl.col("intercept") * pl.col("y")).max().alias("need")))
    return (q.join(need, on=["p", "d", "t"], how="inner")
            .select("p", "d", "t", input=pl.col("q"), excess=pl.col("q") - pl.col("need"))
            .sort("p", "d", "t"))


def _invest_units(solution, var: str, edd, d_col: str) -> "pl.DataFrame | None":
    """``(p, d, value)``: Σ_{d′} ``var``[p,d′] over ``edd`` (p | e, d′ = ``d_col``, d) — the units invested /
    divested in service in d, as ``U`` of ``conversion_piecewise``."""
    val = _value(solution, var)
    edd = _utf8(edd)
    if val is None or edd is None or edd.height == 0:
        return None
    edd = edd.rename({"e": "p"}) if "e" in edd.columns else edd
    return (edd.join(val.rename({"d": d_col}), on=["p", d_col], how="inner")
            .group_by("p", "d").agg(pl.col("value").sum()))


def _delayed_input(flex_data: "FlexData", flow: pl.DataFrame, units: pl.DataFrame) -> "pl.DataFrame | None":
    """``(p, d, t, q)``: the delayed inputs of the delayed units at their sink-side step t (``_delay``):
    Σ_s Σ_δ v_flow[p,s,p,d,t_s] · unitsize · delay_weight[p,δ] · conv_in[p,s]."""
    arcs = _utf8(getattr(flex_data, "process_source_sink_delayed", None))
    durations = _utf8(getattr(flex_data, "process_delayed__duration", None))
    shifts = _utf8(getattr(flex_data, "dtt__delay_duration", None))
    weight = _param(getattr(flex_data, "p_process_delay_weight", None), ("p", "td"), "w")
    if units.height == 0 or any(f is None or f.height == 0 for f in (arcs, durations, shifts)) or weight is None:
        return None
    arcs = arcs.filter(pl.col("sink") == pl.col("p")).select("p", "source", "sink").join(units, on="p", how="semi")
    at = (arcs.join(durations.select("p", "td"), on="p", how="inner")
          .join(shifts.select("d", "t_source", "t_sink", "td"), on="td", how="inner")
          .join(weight, on=["p", "td"], how="inner"))
    f = _with_coeff(flow.rename({"t": "t_source"}), flex_data.p_process_source_conversion_flow_coeff, "source")
    return (at.join(f, on=["p", "source", "sink", "d", "t_source"], how="inner")
            .group_by("p", "d", pl.col("t_sink").alias("t"))
            .agg((pl.col("mw") * pl.col("w") * pl.col("k")).sum().alias("q")))


@dataclass(frozen=True)
class Excess:
    """One unit's input above its curve over the realized steps: ``steps`` with ``excess > 1e-5 · max(1, q)``,
    ``largest`` [input units] at ``at`` (period, step; the first of a tie) and ``total`` [input units]."""

    unit: str
    steps: int
    largest: float
    at: tuple[str, str]
    total: float


def piecewise_excess(par, s) -> list[Excess]:
    """The :class:`Excess` of every multi-line piecewise_linear unit with input above its curve in some realized
    step (``par.process_piecewise_input_dt`` / ``par.process_piecewise_excess_dt`` [MW], after ``drop_levels``),
    in unit order."""
    exc = getattr(par, "process_piecewise_excess_dt", None)
    inp = getattr(par, "process_piecewise_input_dt", None)
    if exc is None or inp is None or exc.empty or len(exc.columns) == 0:
        return []
    realized = s.dt_realize_dispatch
    exc = exc.reindex(index=realized)
    inp = inp.reindex(index=realized, columns=exc.columns)
    hours = par.step_duration.reindex(realized).to_numpy(dtype=float)
    out: list[Excess] = []
    for unit in sorted(str(c) for c in exc.columns):
        e = exc[unit].to_numpy(dtype=float)
        q = inp[unit].to_numpy(dtype=float)
        hit = ~np.isnan(e) & (np.nan_to_num(e) > EXCESS_RTOL * np.maximum(np.abs(np.nan_to_num(q)), 1.0))
        if not hit.any():
            continue
        energy = pd.Series(e * hours, index=realized)[hit]
        at = energy.idxmax()
        out.append(Excess(unit=unit, steps=int(hit.sum()), largest=float(energy.max()),
                          at=(str(at[0]), str(at[1])), total=float(energy.sum())))
    return out


def warn_piecewise_excess(par, s, node_units: "Mapping[str, str] | None" = None,
                          logger: logging.Logger | None = None) -> list[Excess]:
    """Log W5 for each unit of :func:`piecewise_excess`; the amounts are in the unit of measure of its input nodes
    (MWh unless all its inputs share another unit; ``input units`` when they differ).  Returns the findings."""
    found = piecewise_excess(par, s)
    if not found:
        return found
    log = logger or _logger
    node_units = node_units or {}
    sources = getattr(s, "process_source", None)
    pairs = [(str(p), str(n)) for p, n in sources] if sources is not None and len(sources) else []
    for f in found:
        units = {node_units.get(n, "MWh") for p, n in pairs if p == f.unit}
        unit = units.pop() if len(units) == 1 else "input units"
        log.warning(
            "Unit '%s' used more input than its curve needs in %d time step%s (largest %.4g %s in %s/%s, total "
            "%.4g %s): its input had no value or a negative value there (a surplus, a free or curtailed input, a "
            "CO2 credit) or a reserve or a minimum on its input made it pay, so the solution is not unique or "
            "burns input for nothing.", f.unit, f.steps, "" if f.steps == 1 else "s", f.largest, unit, f.at[0], f.at[1], f.total, unit)
    return found
