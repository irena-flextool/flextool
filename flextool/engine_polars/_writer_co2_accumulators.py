"""Native port of ``write_co2_rolling_accumulators`` — Gap F close-out.

Mirrors ``_co2_tonnes_this_roll``: for each realized ``(d, t)``
(``flex_data.realized_dispatch``), per ``v_flow[p, source, sink, d, t]``
whose ``source`` is a CO2 commodity node — add
``content/1000 * value * us * dur * rpw`` (noEff) or that × ``slope``
× ``fc_sink/fc_source`` (eff, ``p`` ∈ ``process_unit``); per row whose
``sink`` is a CO2 node — subtract the same flow piece (no slope).
Attribute to groups via ``group_node``; combine with
``prior_handoff.cumulative_co2``.

``min_load_efficiency``'s section term is deferred (same MVP as legacy
— warn and under-report).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine_polars._axis_enums import (
    alias_to_axis,
    lit_axis,
    rename_to_axis,
    schema_dtype,
)

if TYPE_CHECKING:
    from ._solve_handoff import SolveHandoff

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical writer-port emitter — mirrors the ``_write(df, path)`` idiom
# in :mod:`._writer_arc_unions` and the four other patched modules.  Every
# CSV emission in this module is funnelled through here so the per-sub-solve
# :mod:`._flex_data_accumulator` monkey-patch can stash the frame.
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


# ---------------------------------------------------------------------------
# Structural inputs not yet on FlexData (read once per call).
# ---------------------------------------------------------------------------


def _read_co2_max_total_groups(
    work_folder: Path,
    *, provider: "object | None" = None,
) -> set[str]:
    """Active ``co2_max_total`` cap groups from ``input/group__co2_method.csv``.

    Provider-only after Step 2.5 Phase C — returns an empty set if
    the Provider doesn't carry the frame.  Cascade callers always
    populate the Provider with the SpineDBBackend spec output.
    """
    if provider is None or not provider.has("input/group__co2_method"):
        return set()
    df = provider.get("input/group__co2_method")
    if df.height == 0 or {"group", "co2_method"} - set(df.columns):
        return set()
    return set(
        df.filter(pl.col("co2_method").is_in(
            ["total", "price_total", "period_total", "price_period_total"]
        ))["group"].cast(pl.Utf8).to_list()
    )


def _read_commodity_node_co2(
    work_folder: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """``(c, n)`` pairs from ``solve_data/commodity_node_co2.csv``.

    Provider-only after Step 2.5 Phase C.
    """
    if provider is None or not provider.has("solve_data/commodity_node_co2"):
        return pl.DataFrame(schema={
            "c": schema_dtype(None, "c"),
            "n": schema_dtype(None, "n"),
        })
    df = provider.get("solve_data/commodity_node_co2")
    if df.height == 0:
        return pl.DataFrame(schema={
            "c": schema_dtype(None, "c"),
            "n": schema_dtype(None, "n"),
        })
    return (df.pipe(rename_to_axis, {"commodity": "c", "node": "n"})
            .select("c", "n").unique())


def _param_frame(param, *cols: str) -> pl.DataFrame:
    """Return ``param.frame`` or an empty frame with the given dim columns."""
    if param is None or not hasattr(param, "frame"):
        schema = {c: schema_dtype(None, c) for c in cols}
        schema["value"] = pl.Float64
        return pl.DataFrame(schema=schema)
    return param.frame


def _set_frame(df, *cols: str) -> pl.DataFrame:
    if df is None or df.height == 0:
        return pl.DataFrame(schema={c: schema_dtype(None, c) for c in cols})
    return df


# ---------------------------------------------------------------------------
# Compute.
# ---------------------------------------------------------------------------


_SCHEMA_OUT = {
    "group": pl.Utf8, "period": pl.Utf8,
    "p_co2_cum_realized_tonnes": pl.Float64,
}


def compute_co2_rolling_accumulator(
    flex_data,
    sol,
    *,
    work_folder: Path,
    prior_cumulative_co2: "pl.DataFrame | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Native compute of ``co2_cum_realized_tonnes`` for the current solve.

    Returns ``[group, period, p_co2_cum_realized_tonnes]``.  Empty frame
    when no CO2-cap group is active.
    """
    co2_groups = _read_co2_max_total_groups(work_folder, provider=provider)
    if not co2_groups:
        return pl.DataFrame(schema=_SCHEMA_OUT)

    cn_co2 = _read_commodity_node_co2(work_folder, provider=provider)
    if cn_co2.height == 0:
        return pl.DataFrame(schema=_SCHEMA_OUT)

    if sol is None or "v_flow" not in getattr(sol, "_vars", {}):
        return _passthrough_prior(prior_cumulative_co2, co2_groups)
    v_flow = sol.value("v_flow")
    if v_flow is None or v_flow.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    p_step_duration = _param_frame(
        getattr(flex_data, "p_step_duration", None), "d", "t",
    ).select("d", "t", pl.col("value").alias("dur"))
    if p_step_duration.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    p_rp_weight = _param_frame(
        getattr(flex_data, "p_rp_cost_weight", None), "d", "t",
    ).select("d", "t", pl.col("value").alias("rpw"))

    realized = _set_frame(
        getattr(flex_data, "realized_dispatch", None), "period", "step",
    )
    if realized.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)
    realized_dt = (realized
        .select(alias_to_axis("period", "d"), alias_to_axis("step", "t"))
        .unique())

    pss_eff = _set_frame(
        getattr(flex_data, "process_source_sink_eff", None),
        "p", "source", "sink",
    ).with_columns(branch=lit_axis("eff", "branch"))
    pss_noeff = _set_frame(
        getattr(flex_data, "process_source_sink_noEff", None),
        "p", "source", "sink",
    ).with_columns(branch=lit_axis("noEff", "branch"))
    pss = pl.concat([pss_eff, pss_noeff], how="diagonal")

    process_unit = _set_frame(
        getattr(flex_data, "process_unit", None), "p",
    ).with_columns(is_unit=pl.lit(True))

    co2_content = _param_frame(
        getattr(flex_data, "p_co2_content", None), "c",
    ).select("c", pl.col("value").alias("content"))
    if co2_content.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    us_param = (getattr(flex_data, "p_all_entity_unitsize", None)
                or getattr(flex_data, "p_unitsize", None))
    unitsize = _param_frame(us_param, "p").select(
        pl.col("p").alias("p_us"), pl.col("value").alias("us"))

    slope = _param_frame(
        getattr(flex_data, "p_slope", None), "p", "d", "t",
    ).select("p", "d", "t", pl.col("value").alias("slope"))

    fc_source = _param_frame(
        getattr(flex_data, "p_process_source_flow_coef", None),
        "p", "source",
    ).select("p", "source", pl.col("value").alias("fc_source"))
    fc_sink = _param_frame(
        getattr(flex_data, "p_process_sink_flow_coef", None),
        "p", "sink",
    ).select("p", "sink", pl.col("value").alias("fc_sink"))

    gn_full = _set_frame(
        getattr(flex_data, "group_node", None), "g", "n",
    )
    gn = (gn_full.filter(pl.col("g").is_in(list(co2_groups)))
          if {"g", "n"}.issubset(gn_full.columns) and gn_full.height > 0
          else pl.DataFrame(schema={
              "g": schema_dtype(None, "g"),
              "n": schema_dtype(None, "n"),
          }))

    pmle = _set_frame(getattr(flex_data, "process_min_load_eff", None), "p")
    if pmle.height > 0:
        _logger.warning(
            "compute_co2_rolling_accumulator: model uses min_load_efficiency "
            "on %d process(es); native compute under-reports their section "
            "term (LP cap binds slightly tighter). Affected: %s",
            pmle.height,
            sorted(pmle["p"].cast(pl.Utf8).to_list())[:10],
        )

    # Filter v_flow: realized timesteps × incident to a CO2 node × non-zero.
    nodes_with_co2 = list(cn_co2["n"].cast(pl.Utf8).unique().to_list())
    vf = (v_flow
        .filter((pl.col("source").is_in(nodes_with_co2))
                | (pl.col("sink").is_in(nodes_with_co2)))
        .filter(pl.col("value").abs() > 0.0)
        .join(realized_dt, on=["d", "t"], how="inner"))
    if vf.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    # Annotate every row (left joins with default fills).
    def _ljoin(frame, payload, fill, **kw):
        if payload.height == 0:
            return frame.with_columns(
                **{k: pl.lit(v) for k, v in fill.items()})
        return (frame.join(payload, how="left", **kw)
                     .with_columns(*[pl.col(k).fill_null(v)
                                     for k, v in fill.items()]))

    vf = vf.join(pss, on=["p", "source", "sink"], how="left") \
           .join(p_step_duration, on=["d", "t"], how="left") \
           .with_columns(pl.col("dur").fill_null(0.0))
    vf = _ljoin(vf, p_rp_weight, {"rpw": 1.0}, on=["d", "t"])
    vf = _ljoin(vf, unitsize, {"us": 1.0}, left_on="p", right_on="p_us")
    vf = _ljoin(vf, slope, {"slope": 1.0}, on=["p", "d", "t"])
    vf = _ljoin(vf, fc_sink, {"fc_sink": 1.0}, on=["p", "sink"])
    vf = _ljoin(vf, fc_source, {"fc_source": 1.0}, on=["p", "source"])
    vf = _ljoin(vf, process_unit, {"is_unit": False}, on="p")

    # Emission branch (source ∈ CO2 nodes).  Skip rows whose pss isn't
    # categorised (branch null) — mirrors the legacy "skip uncategorised".
    cn_emis = cn_co2.pipe(rename_to_axis, {"n": "source", "c": "c_emis"})
    co2_emis = co2_content.rename({"c": "c_emis", "content": "content_emis"})
    emis = (vf
        .join(cn_emis, on="source", how="inner")
        .join(co2_emis, on="c_emis", how="left")
        .with_columns(pl.col("content_emis").fill_null(0.0))
        .filter(pl.col("content_emis") != 0.0)
        .filter(pl.col("branch").is_not_null())
        .with_columns(
            coeff=pl.when(pl.col("branch") == "eff")
                    .then(pl.when(pl.col("is_unit"))
                            .then(pl.col("fc_sink")
                                  / pl.when(pl.col("fc_source") != 0.0)
                                      .then(pl.col("fc_source"))
                                      .otherwise(1.0))
                            .otherwise(1.0))
                    .otherwise(1.0),
            slope_used=pl.when(pl.col("branch") == "eff")
                         .then(pl.col("slope")).otherwise(1.0),
        )
        .with_columns(
            contribution=(
                pl.col("content_emis") / 1000.0
                * pl.col("value") * pl.col("us") * pl.col("dur")
                * pl.col("rpw") * pl.col("slope_used") * pl.col("coeff")),
            attr_node=pl.col("source"),
        )
        .select("attr_node", alias_to_axis("d", "period"), "contribution"))

    # Removal branch (sink ∈ CO2 nodes).
    cn_rem = cn_co2.pipe(rename_to_axis, {"n": "sink", "c": "c_rem"})
    co2_rem = co2_content.rename({"c": "c_rem", "content": "content_rem"})
    rem = (vf
        .join(cn_rem, on="sink", how="inner")
        .join(co2_rem, on="c_rem", how="left")
        .with_columns(pl.col("content_rem").fill_null(0.0))
        .filter(pl.col("content_rem") != 0.0)
        .with_columns(
            contribution=(
                -pl.col("content_rem") / 1000.0
                * pl.col("value") * pl.col("us") * pl.col("dur")
                * pl.col("rpw")),
            attr_node=pl.col("sink"),
        )
        .select("attr_node", alias_to_axis("d", "period"), "contribution"))

    contrib = pl.concat([emis, rem], how="vertical")
    if contrib.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    # Attribute to groups.
    if gn.height > 0:
        attributed = (contrib
            .join(gn, left_on="attr_node", right_on="n", how="inner")
            .group_by(["g", "period"])
            .agg(pl.col("contribution").sum().alias("value"))
            .pipe(rename_to_axis, {"g": "group"}))
    else:
        # No group_node mapping → fan out to every active group.
        per_period = (contrib
            .group_by("period")
            .agg(pl.col("contribution").sum().alias("value")))
        attributed = pl.concat(
            [per_period.with_columns(group=lit_axis(g, "group"))
             for g in sorted(co2_groups)],
            how="vertical",
        ).select("group", "period", "value")

    return _combine_with_prior(attributed, prior_cumulative_co2)


def _combine_with_prior(
    this_roll: pl.DataFrame,
    prior_cumulative_co2: "pl.DataFrame | None",
) -> pl.DataFrame:
    """Outer-join this-roll + prior, sum, return ``[group, period,
    p_co2_cum_realized_tonnes]`` sorted."""
    if prior_cumulative_co2 is None or prior_cumulative_co2.height == 0:
        prior = pl.DataFrame(schema={
            "group": schema_dtype(None, "group"),
            "period": schema_dtype(None, "period"),
            "prior": pl.Float64})
    else:
        prior = (prior_cumulative_co2
            .with_columns(prior=pl.col("value").cast(pl.Float64))
            .select("group", "period", "prior"))
    combined = (this_roll
        .join(prior, on=["group", "period"], how="full", coalesce=True)
        .with_columns(
            p_co2_cum_realized_tonnes=(
                pl.col("value").fill_null(0.0)
                + pl.col("prior").fill_null(0.0)))
        .select("group", "period", "p_co2_cum_realized_tonnes")
        .sort(["group", "period"]))
    return combined


def _passthrough_prior(prior_cumulative_co2, co2_groups) -> pl.DataFrame:
    """Return the prior carrier (restricted to active groups) when this
    roll contributes zero rows."""
    if prior_cumulative_co2 is None or prior_cumulative_co2.height == 0:
        return pl.DataFrame(schema=_SCHEMA_OUT)
    return (prior_cumulative_co2
        .filter(pl.col("group").is_in(list(co2_groups)))
        .with_columns(value=pl.col("value").cast(pl.Float64))
        .select("group", "period",
                pl.col("value").alias("p_co2_cum_realized_tonnes"))
        .sort(["group", "period"]))


# ---------------------------------------------------------------------------
# CSV writer (legacy-compat wrapper).
# ---------------------------------------------------------------------------


def _format_co2_cum_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply legacy ``%.8g`` formatting + return a 3-col Utf8 frame whose
    write_csv output matches the legacy emitter byte-for-byte.

    Schema after: ``[group: Utf8, period: Utf8,
    p_co2_cum_realized_tonnes: Utf8]``.  An empty frame still carries the
    schema, so :func:`_write` emits the header-only CSV the legacy code
    produced via ``write_text("group,period,p_co2_cum_realized_tonnes\\n")``.
    """
    if frame.height == 0:
        return pl.DataFrame(
            schema={"group": pl.Utf8, "period": pl.Utf8,
                    "p_co2_cum_realized_tonnes": pl.Utf8},
        )
    return frame.with_columns(
        pl.col("p_co2_cum_realized_tonnes")
          .map_elements(lambda v: format(float(v), ".8g"),
                        return_dtype=pl.Utf8)
    )


def derive_co2_cum_realized_tonnes(
    flex_data,
    sol,
    *,
    work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Return the canonical formatted ``co2_cum_realized_tonnes`` CSV frame.

    Thin wrapper around :func:`compute_co2_rolling_accumulator` +
    :func:`_format_co2_cum_frame`.  Used by both
    :func:`write_co2_rolling_accumulator` (which funnels through
    :func:`_write` so the accumulator captures the frame) and tests.
    """
    prior_df = (prior_handoff.cumulative_co2
                if prior_handoff is not None else None)
    frame = compute_co2_rolling_accumulator(
        flex_data, sol,
        work_folder=work_folder,
        prior_cumulative_co2=prior_df,
        provider=provider,
    )
    return _format_co2_cum_frame(frame)


def write_co2_rolling_accumulator(
    flex_data,
    sol,
    *,
    solve_name: str,
    work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Compute + write ``solve_data/co2_cum_realized_tonnes.csv`` natively.

    Returns the wide ``[group, period, p_co2_cum_realized_tonnes]`` (raw
    float-valued) frame — used by callers to populate
    ``SolveHandoff.cumulative_co2``.  The CSV emission funnels its formatted
    version through :func:`_write` so the per-sub-solve
    :mod:`._flex_data_accumulator` captures the byte-canonical frame.
    """
    out_path = work_folder / "solve_data" / "co2_cum_realized_tonnes.csv"
    prior_df = (prior_handoff.cumulative_co2
                if prior_handoff is not None else None)
    frame = compute_co2_rolling_accumulator(
        flex_data, sol,
        work_folder=work_folder,
        prior_cumulative_co2=prior_df,
        provider=provider,
    )
    _write(_format_co2_cum_frame(frame), out_path)
    _logger.info("wrote %s (%d rows)", out_path, frame.height)
    return frame


# ---------------------------------------------------------------------------
# Legacy-signature shim (monkey-patch target).
# ---------------------------------------------------------------------------


def write_co2_rolling_accumulators_native(
    h, *, solve_name: str, work_folder: Path,
    prior_handoff=None, flex_data=None, sol=None,
):
    """Legacy-signature shim (monkey-patch target).  Falls back to the
    legacy disk-reading impl when ``flex_data`` / ``sol`` aren't supplied.
    """
    work_folder = Path(work_folder)
    out_path = work_folder / "solve_data" / "co2_cum_realized_tonnes.csv"
    if flex_data is not None and sol is not None:
        write_co2_rolling_accumulator(
            flex_data, sol, solve_name=solve_name,
            work_folder=work_folder, prior_handoff=prior_handoff)
        return [out_path]
    from flextool.process_outputs import cumulative_handoffs as _legacy
    fn = getattr(_legacy, "_legacy_write_co2_rolling_accumulators",
                 _legacy.write_co2_rolling_accumulators)
    return fn(h, solve_name=solve_name, work_folder=work_folder,
              prior_handoff=prior_handoff)


__all__ = [
    "compute_co2_rolling_accumulator",
    "derive_co2_cum_realized_tonnes",
    "write_co2_rolling_accumulator",
    "write_co2_rolling_accumulators_native",
]
