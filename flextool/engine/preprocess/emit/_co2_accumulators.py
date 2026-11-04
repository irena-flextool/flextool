"""Per-roll CO2 rolling accumulator emission (Gap F close-out).

For each realized ``(d, t)`` (``flex_data.realized_dispatch``), per ``v_flow[p, source, sink, d, t]``
whose ``source`` is a CO2 commodity node — add ``content * value * us * dur * rpw`` (noEff) or that ×
``slope`` (eff — ``slope`` already carries a direct unit's conversion_flow_coeff ratio, #10 / Q12); per
row whose ``sink`` is a CO2 node — subtract the same flow piece (no slope).  No further unit conversion:
``co2_content`` is already CO2 tons per MWh (``docs/reference.md``), the same measure core.py's
``co2_max_period`` / ``co2_max_total`` / ``co2_price_eff`` rows use directly — an extra ``/1000`` here
would silently undercount the carried realized total a thousandfold against what the LP itself counts.
Attribute to groups via ``group_node``; combine with the prior roll's
accumulator carried in the Provider under ``K.HANDOFF_CUMULATIVE_CO2``
(populated by ``_provider_translators.translate_handoff_to_provider``).

``min_load_efficiency``'s section term (v_online × section × unitsize, mirroring core.py's
``_eff_fuel_input``) is included on the source-side branch when ``flex_data.p_section`` is available;
otherwise it is omitted and a warning logged (under-reports, LP cap binds slightly tighter than the
carried figure says).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    lit_axis,
    rename_to_axis,
    schema_dtype,
)

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural inputs not yet on FlexData (read once per call).
# ---------------------------------------------------------------------------


def _read_co2_capped_groups(
    work_folder: Path,
    *, provider: "object | None" = None,
) -> set[str]:
    """Active CO2-*capped* groups (``co2_max_period`` or ``co2_max_total``, #38, Q22) from
    ``input/group__co2_method.csv``.

    Both caps' carried-budget RHS (``_co2_period_cap_with_carry``, ``_co2_total_cap_with_carry`` in
    ``model/core.py``) need this roll's realized tonnes, so the gate is the union of
    ``_leaf_sets._CO2_MAX_PERIOD_METHOD`` and ``_leaf_sets._CO2_MAX_TOTAL_METHOD`` — every method that
    carries an actual cap (plain ``price``/``price_period``/``price_total`` excluded: pricing alone has
    no RHS to carry).  Originally gated on ``_CO2_MAX_TOTAL_METHOD`` alone, which silently dropped every
    period-only-capped group (``co2_method="period"``, no ``"total"``): the realized carrier came back
    empty for them every roll, so ``co2_max_period``'s cap re-granted its own fraction-scaled allocation
    with nothing subtracted (#38 regression caught by
    ``test_co2_max_period_rolling.py::test_rolling_without_lookahead_equals_single_solve``).

    Provider-only after Step 2.5 Phase C — returns an empty set if
    the Provider doesn't carry the frame.  Cascade callers always
    populate the Provider with the SpineDBBackend spec output.
    """
    if provider is None or not provider.has("input/group__co2_method"):
        return set()
    df = provider.get("input/group__co2_method")
    if df.height == 0 or {"group", "co2_method"} - set(df.columns):
        return set()
    from flextool.engine.preprocess.emit._leaf_sets import (
        _CO2_MAX_PERIOD_METHOD,
        _CO2_MAX_TOTAL_METHOD,
    )
    capped_methods = _CO2_MAX_PERIOD_METHOD | _CO2_MAX_TOTAL_METHOD
    return set(
        df.filter(pl.col("co2_method").is_in(list(capped_methods)))
        ["group"].cast(pl.Utf8).to_list()
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
    "p_co2_cum_realized_represented": pl.Float64,
}


def compute_co2_rolling_accumulator(
    flex_data,
    sol,
    *,
    work_folder: Path,
    prior_cumulative_co2: "pl.DataFrame | None" = None,
    provider: "object | None" = None,
    is_dispatch_sibling_parent: bool = False,
) -> pl.DataFrame:
    """Native compute of ``co2_cum_realized_tonnes`` for the current solve.

    Returns ``[group, period, p_co2_cum_realized_tonnes, p_co2_cum_realized_represented]``.  Empty
    frame when no CO2-cap group is active.

    ``p_co2_cum_realized_tonnes`` is the physical tonnes (co2_max_period's carry, divided by the period
    share there).  ``p_co2_cum_realized_represented`` is the same tonnes in co2_max_total's units: each
    realized step's tonnes × ``years_represented[d] / period_share[d]`` of THIS (realizing) solve, the
    weight co2_max_total's LHS gives the step's emissions.  It is computed here because a period finished
    by an earlier solve of a chain need not be in a later solve's ``p_period_share``, and a chained solve
    can dispatch a period at a different share.  A realized period that THIS solve has no
    ``p_period_share`` row for gets a null, which co2_max_total refuses (``_co2_total_cap_with_carry``).
    ``p_years_represented_d`` absent means 1 year per period (its loader's default).

    ``realized_dispatch`` (below) holds no cell of a stochastic fan: the
    period label (the original period name) has no dispatch and the
    forecast branches are never realized, so a fan's tonnes come only from
    the realization solve's own extraction (#59).

    ``is_dispatch_sibling_parent`` (#59 fix 3, applied the same way as the
    commodity-ladder carry): True when THIS solve is the parent of an
    auto-spawned realized-dispatch sibling (OC1 separate_dispatch).  The
    sibling re-solves the SAME realized window with its own real
    objective right after this solve and carries ITS OWN prior roll
    forward for ``cumulative_co2`` (``_run_model``'s ``ROLL_FORWARD_FIELDS``
    routes it from the sibling's own previous roll, not from this
    parent's handoff) — so this parent must add nothing of its own here,
    or the realized tonnes of those hours would be counted twice.
    """
    co2_groups = _read_co2_capped_groups(work_folder, provider=provider)
    if not co2_groups:
        return pl.DataFrame(schema=_SCHEMA_OUT)

    if is_dispatch_sibling_parent:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    cn_co2 = _read_commodity_node_co2(work_folder, provider=provider)
    if cn_co2.height == 0:
        return pl.DataFrame(schema=_SCHEMA_OUT)

    from flextool.engine.output._coarse_flows import flow_values

    # Every fine step, also of coarse processes (one LP column per coarse step).
    v_flow = flow_values(sol, flex_data)
    if v_flow is None or v_flow.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    p_step_duration = _param_frame(
        getattr(flex_data, "p_step_duration", None), "d", "t",
    ).select("d", "t", pl.col("value").alias("dur"))
    if p_step_duration.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    p_rp_weight = _param_frame(
        getattr(flex_data, "p_timestep_weight", None), "d", "t",
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

    co2_content = _param_frame(
        getattr(flex_data, "p_co2_content", None), "c",
    ).select("c", pl.col("value").alias("content"))
    if co2_content.height == 0:
        return _passthrough_prior(prior_cumulative_co2, co2_groups)

    us_param = (getattr(flex_data, "p_all_entity_unitsize", None)
                or getattr(flex_data, "p_unitsize", None))
    # ``p_all_entity_unitsize`` is keyed on the entity-union axis ``e``
    # (unit ∪ node ∪ connection); ``p_unitsize`` is keyed on the process
    # axis ``p``.  Either carries the per-process unitsize we join against
    # ``v_flow.p`` below, but under a different column name — read whichever
    # entity column is present.  (``_param_frame``'s empty fallback uses the
    # requested name, so default to ``p``.)
    us_frame = _param_frame(us_param, "p")
    us_entity_col = "e" if "e" in us_frame.columns else "p"
    # Cast the unitsize entity column to ``v_flow.p``'s dtype so the
    # ``left_on="p", right_on="p_us"`` join below never mixes Enum + Utf8
    # (the entity-union ``e`` enum and the process ``p`` enum are distinct
    # dtypes even when their values overlap).
    _p_dtype = v_flow.schema.get("p", pl.Utf8)
    unitsize = us_frame.select(
        pl.col(us_entity_col).cast(_p_dtype, strict=False).alias("p_us"),
        pl.col("value").alias("us"))

    slope = _param_frame(
        getattr(flex_data, "p_slope", None), "p", "d", "t",
    ).select("p", "d", "t", pl.col("value").alias("slope"))

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
    if pmle.height > 0 and getattr(flex_data, "p_section", None) is None:
        _logger.warning(
            "compute_co2_rolling_accumulator: model uses min_load_efficiency "
            "on %d process(es) but p_section is unavailable; native compute "
            "under-reports their section term (LP cap binds slightly "
            "tighter). Affected: %s",
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
            # ``slope`` already carries a direct unit's conversion_flow_coeff ratio (#10 / Q12),
            # so no separate fc_sink/fc_source coefficient is applied on top of it.
            slope_used=pl.when(pl.col("branch") == "eff")
                         .then(pl.col("slope")).otherwise(1.0),
        )
        .with_columns(
            contribution=(
                pl.col("content_emis")
                * pl.col("value") * pl.col("us") * pl.col("dur")
                * pl.col("rpw") * pl.col("slope_used")),
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
                -pl.col("content_rem")
                * pl.col("value") * pl.col("us") * pl.col("dur")
                * pl.col("rpw")),
            attr_node=pl.col("sink"),
        )
        .select("attr_node", alias_to_axis("d", "period"), "contribution"))

    # Min-load efficiency section fuel (v_online_linear / v_online_integer × section × unitsize),
    # mirroring core.py's _eff_fuel_input section_lin / section_int terms: the LP's CO2 cap rows
    # (co2_price_eff, co2_max_period, co2_max_total) count this on top of the eff slope term above, so
    # the carried realized-so-far figure (#38, Q22) must too, or it would understate what the LP
    # actually counted and loosen the next roll's carried budget.  Only the source-side (emission)
    # branch has a section term -- the removal branch has none in the LP either.  Restricted to eff
    # arcs via ``pss``'s own branch tag, same as the slope term.
    minload = pl.DataFrame(schema={
        "attr_node": schema_dtype(None, "n"), "period": schema_dtype(None, "d"),
        "contribution": pl.Float64,
    })
    if pmle.height > 0 and getattr(flex_data, "p_section", None) is not None:
        section_arcs = (pss.filter(pl.col("branch") == "eff").select("p", "source", "sink")
                        .join(pmle, on="p", how="inner")
                        .join(cn_emis, on="source", how="inner"))
        if section_arcs.height > 0:
            online_long = None
            sol_vars = getattr(sol, "_vars", {}) or {}
            for var_name in ("v_online_linear", "v_online_integer"):
                if var_name in sol_vars:
                    vd = sol.value(var_name)
                    if vd is not None and vd.height > 0:
                        online_long = (vd.select("p", "d", "t", "value") if online_long is None
                                       else pl.concat([online_long, vd.select("p", "d", "t", "value")],
                                                       how="vertical"))
            if online_long is not None and online_long.height > 0:
                online = (online_long
                    .join(realized_dt, on=["d", "t"], how="inner")
                    .join(section_arcs, on="p", how="inner"))
                if online.height > 0:
                    p_section = _param_frame(flex_data.p_section, "p", "d", "t").select(
                        "p", "d", "t", pl.col("value").alias("section"))
                    online = (online.join(p_section, on=["p", "d", "t"], how="left")
                              .with_columns(pl.col("section").fill_null(0.0))
                              .join(p_step_duration, on=["d", "t"], how="left")
                              .with_columns(pl.col("dur").fill_null(0.0)))
                    online = _ljoin(online, p_rp_weight, {"rpw": 1.0}, on=["d", "t"])
                    online = _ljoin(online, unitsize, {"us": 1.0}, left_on="p", right_on="p_us")
                    online = (online.join(co2_emis, on="c_emis", how="left")
                        .with_columns(pl.col("content_emis").fill_null(0.0))
                        .filter(pl.col("content_emis") != 0.0)
                        .with_columns(
                            contribution=(
                                pl.col("content_emis")
                                * pl.col("value") * pl.col("section")
                                * pl.col("us") * pl.col("dur") * pl.col("rpw")),
                            attr_node=pl.col("source"))
                        .select("attr_node", alias_to_axis("d", "period"), "contribution"))
                    if online.height > 0:
                        minload = online

    contrib = pl.concat([emis, rem, minload], how="vertical")
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

    attributed = _with_represented(attributed, flex_data)
    return _combine_with_prior(attributed, prior_cumulative_co2)


def _with_represented(attributed: pl.DataFrame, flex_data) -> pl.DataFrame:
    """``attributed`` ``[group, period, value]`` plus ``represented`` = value · years_represented[d] /
    period_share[d] of the extracting solve (null where the solve has no period_share row for d)."""
    share = _param_frame(getattr(flex_data, "p_period_share", None), "d").select(
        pl.col("d").cast(pl.Utf8).alias("_pk"), pl.col("value").cast(pl.Float64).alias("share"))
    years_p = getattr(flex_data, "p_years_represented_d", None)
    out = (attributed.with_columns(_pk=pl.col("period").cast(pl.Utf8))
           .join(share.unique("_pk"), on="_pk", how="left"))
    if years_p is None:
        out = out.with_columns(years=pl.lit(1.0))
    else:
        years = years_p.frame.select(pl.col("d").cast(pl.Utf8).alias("_pk"),
                                     pl.col("value").cast(pl.Float64).alias("years"))
        out = out.join(years.unique("_pk"), on="_pk", how="left")
    return out.with_columns(
        represented=pl.col("value") * pl.col("years") / pl.col("share"),
    ).select("group", "period", "value", "represented")


def _combine_with_prior(
    this_roll: pl.DataFrame,
    prior_cumulative_co2: "pl.DataFrame | None",
) -> pl.DataFrame:
    """Outer-join this-roll + prior, sum, return ``[group, period,
    p_co2_cum_realized_tonnes, p_co2_cum_realized_represented]`` sorted.

    A null ``represented`` on either side (a period the extracting solve had no period share for, or a
    prior carrier without the column) stays null in the sum, so co2_max_total refuses it."""
    if prior_cumulative_co2 is None or prior_cumulative_co2.height == 0:
        prior = pl.DataFrame(schema={
            "group": schema_dtype(None, "group"),
            "period": schema_dtype(None, "period"),
            "prior": pl.Float64, "prior_rep": pl.Float64})
    else:
        prior = (_with_represented_column(prior_cumulative_co2)
            .select("group", "period",
                    prior=pl.col("value").cast(pl.Float64),
                    prior_rep=pl.col("represented")))
    combined = (this_roll
        .join(prior, on=["group", "period"], how="full", coalesce=True)
        .with_columns(
            p_co2_cum_realized_tonnes=(
                pl.col("value").fill_null(0.0)
                + pl.col("prior").fill_null(0.0)),
            # a side without rows contributes 0; a side WITH a row but no represented value stays null
            p_co2_cum_realized_represented=(
                pl.when(pl.col("value").is_null()).then(0.0).otherwise(pl.col("represented"))
                + pl.when(pl.col("prior").is_null()).then(0.0).otherwise(pl.col("prior_rep"))))
        .select("group", "period", "p_co2_cum_realized_tonnes", "p_co2_cum_realized_represented")
        .sort(["group", "period"]))
    return combined


def _with_represented_column(prior_cumulative_co2: pl.DataFrame) -> pl.DataFrame:
    """The handoff carrier with a Float64 ``represented`` column (null when the carrier has none)."""
    if "represented" in prior_cumulative_co2.columns:
        return prior_cumulative_co2.with_columns(pl.col("represented").cast(pl.Float64, strict=False))
    return prior_cumulative_co2.with_columns(represented=pl.lit(None, dtype=pl.Float64))


def _passthrough_prior(prior_cumulative_co2, co2_groups) -> pl.DataFrame:
    """Return the prior carrier (restricted to active groups) when this
    roll contributes zero rows."""
    if prior_cumulative_co2 is None or prior_cumulative_co2.height == 0:
        return pl.DataFrame(schema=_SCHEMA_OUT)
    return (_with_represented_column(prior_cumulative_co2)
        .filter(pl.col("group").is_in(list(co2_groups)))
        .with_columns(value=pl.col("value").cast(pl.Float64))
        .select("group", "period",
                pl.col("value").alias("p_co2_cum_realized_tonnes"),
                pl.col("represented").alias("p_co2_cum_realized_represented"))
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


__all__ = [
    "compute_co2_rolling_accumulator",
]
