"""Loaders of the time structure, the nodes and the process topology (always loaded)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import polars as pl
from polar_high import Param

if TYPE_CHECKING:
    from collections.abc import Callable
    from flextool.engine.input._input_source import InputSource

from flextool.engine.input._axis_enums import cast_dim, get_global_axis_enums, rename_to_axis
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.flex_data._readers import (
    _provider_has,
    _provider_read,
    _read_long,
    _read_p_flow_max,
    _read_unitsize,
    _read_wide_per_entity,
)


# ---------------------------------------------------------------------------
# Time + node helpers (always loaded)

def _load_time(sd: Path,
                *,
                provider: "object | None" = None):
    # ``steps_in_use.csv`` is the canonical source for both the dt set
    # and step_duration (.mod reads them together at flextool.mod:781).
    # ``dt.csv`` and ``p_step_duration.csv`` are .mod printf debug-exports
    # that only cover dispatch periods — using them silently drops the
    # invest-period (d, t) rows in multi-period scenarios.
    siu = _provider_read(provider, "solve_data/steps_in_use",
                          sd / "steps_in_use.csv").pipe(
        rename_to_axis, {"period": "d", "step": "t", "step_duration": "value"})
    # Phase E-d — the in-memory accumulator returns Utf8-typed frames
    # (writers funnel through ``_to_utf8_frame``); cast ``value`` to
    # Float64 so downstream Param arithmetic doesn't hit a
    # ``arithmetic on string and numeric not allowed`` error.
    if "value" in siu.columns and siu.schema["value"] != pl.Float64:
        siu = siu.with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
    # Sort dt canonically by (d, t) so it is pinned chronologically (d, t
    # are pl.Enum with chronological category order).  dt is tiny, so this
    # sort is negligible — and because the big (entity, d, t) grids are
    # built as ``entity_frame.join(dt, how="cross")`` (left-major /
    # right-minor), a pre-sorted dt makes those grids ALREADY globally
    # lex-sorted by the declared dim order, letting the add_var/add_cstr
    # determinism wrapper skip its O(n log n) boundary sort.
    dt = siu.select("d", "t").sort(["d", "t"])
    step_dur = Param(("d","t"), siu.select("d", "t", "value"))
    # timestep_weight: canonical ``timestep_weight.csv``
    # (.mod's ``p_rp_cost_weight.csv`` is a printf debug-export).
    # Defaults to 1.0 per (d, t) when the canonical file is empty
    # (matches .mod's ``param p_rp_cost_weight ... default 1`` clause).
    tsw_default = dt.with_columns(value=pl.lit(1.0))
    tsw_path = sd / "timestep_weight.csv"
    if _provider_has(provider, "solve_data/timestep_weight", tsw_path):
        tsw_df = _provider_read(provider, "solve_data/timestep_weight", tsw_path)
        if tsw_df.height > 0:
            # canonical column is named ``weight`` per .mod's ``table data IN``.
            value_col = "weight" if "weight" in tsw_df.columns else "value"
            tsw_df = (tsw_df.pipe(rename_to_axis, {"period": "d", "time": "t",
                                    value_col: "value"})
                          .with_columns(value=pl.col("value")
                                                 .cast(pl.Float64, strict=False))
                          .select("d", "t", "value"))
            # Left-join the default with explicit overrides.
            tsw_default = (tsw_default.join(tsw_df, on=["d","t"], how="left",
                                            suffix="__r")
                                     .with_columns(value=pl.coalesce(
                                          pl.col("value__r"), pl.col("value")))
                                     .select("d","t","value"))
    tsw = Param(("d","t"), tsw_default)
    infl = Param(("d",),
        _read_long(sd / "p_inflation_factor_operations_yearly.csv",
                    rename={"period": "d"}, provider=provider))
    # complete_period_share_of_year: canonical
    # ``complete_period_share_of_year_calc.csv``.
    psh = Param(("d",),
        _read_long(sd / "complete_period_share_of_year_calc.csv",
                    rename={"period": "d"}, provider=provider))
    return dt, step_dur, tsw, infl, psh


def _load_node(sd: Path, dt: pl.DataFrame,
                *,
                provider: "object | None" = None):
    nb = _provider_read(provider, "solve_data/nodeBalance",
                         sd / "nodeBalance.csv").pipe(rename_to_axis, {"node": "n"})
    # ``nodeBalancePeriod`` — nodes whose balance is enforced once per
    # period (legacy ``balance_within_period``) rather than per (d, t).
    # Same single-column (``node`` → ``n``) shape, dtype and axis handling
    # as ``nb``.  The CSV may be absent (older fixtures predating the
    # feature) or header-only (no period nodes); in both cases keep ``nbp``
    # an empty DataFrame with ``nb``'s exact schema so every downstream
    # union collapses cleanly to ``nb`` (byte-identical legacy behavior).
    if _provider_has(provider, "solve_data/nodeBalancePeriod",
                     sd / "nodeBalancePeriod.csv"):
        nbp = _provider_read(provider, "solve_data/nodeBalancePeriod",
                             sd / "nodeBalancePeriod.csv").pipe(
                                 rename_to_axis, {"node": "n"})
    else:
        nbp = nb.clear()
    # pdtNodeInflow.csv is canonical (.mod reads it via `table data IN`).
    # TODO(Δ.18+): retire pdtNodeInflow.csv read when ``apply_derived_a``
    # extends ``p_inflow_from_source`` to cover ``inflow_method ∈ {scale_to_*}``
    # and stochastic 3d_map shapes.
    inflow_long = _read_wide_per_entity(sd / "pdtNodeInflow.csv",
                                          rename={"entity":"n"},
                                          provider=provider)

    # Δ.18 — CSV-fallback seed for ``p_penalty_up`` / ``p_penalty_down``
    # from the wide-by-param ``pdtNode.csv`` slice.  Override chain
    # (``apply_derived_a`` via ``p_penalty_up_from_source`` /
    # ``p_penalty_down_from_source``) overlays these when active; for
    # synthetic per-sub-solve fixtures the snapshot CSV is the only source.
    empty_n_d_t = pl.DataFrame(schema={"n": pl.Utf8, "d": pl.Utf8,
                                         "t": pl.Utf8, "value": pl.Float64})
    pen_up_df = empty_n_d_t
    pen_dn_df = empty_n_d_t
    pdtnode_path = sd / "pdtNode.csv"
    if _provider_has(provider, "solve_data/pdtNode", pdtnode_path):
        df_pn = _provider_read(provider, "solve_data/pdtNode", pdtnode_path)
        if df_pn.height > 0 and {"node", "param", "period", "time", "value"}.issubset(df_pn.columns):
            df_pn = (df_pn.pipe(rename_to_axis, {"node": "n", "period": "d", "time": "t"})
                          .with_columns(value=pl.col("value")
                                                  .cast(pl.Float64, strict=False)
                                                  .fill_null(0.0)))
            up = df_pn.filter(pl.col("param") == "penalty_up").select("n", "d", "t", "value")
            if up.height > 0:
                pen_up_df = up
            dn = df_pn.filter(pl.col("param") == "penalty_down").select("n", "d", "t", "value")
            if dn.height > 0:
                pen_dn_df = dn

    # Nodes with penalty_method == 'off' get NO balance-slack (penalty)
    # variables -> hard node balance.  Absent CSV / no 'off' rows -> None
    # (byte-identical legacy behaviour: every node keeps its slack).
    penalty_off = None
    if _provider_has(provider, "solve_data/node__penalty_method",
                     sd / "node__penalty_method.csv"):
        pm = _provider_read(provider, "solve_data/node__penalty_method",
                            sd / "node__penalty_method.csv")
        # Emitted header is `node,penalty_method`; tolerate a `method`
        # value-column too (mirrors storage_binding_method's dual schema).
        if "penalty_method" in pm.columns:
            pm = pm.pipe(rename_to_axis, {"node": "n", "penalty_method": "method"})
        elif "method" in pm.columns:
            pm = pm.pipe(rename_to_axis, {"node": "n"})
        off = pm.filter(pl.col("method") == "off").select("n").unique()
        if off.height > 0:
            # Align n dtype to nb's axis so the model-side anti-join matches.
            if off.schema["n"] != nb.schema["n"]:
                off = off.with_columns(pl.col("n").cast(nb.schema["n"], strict=False))
            penalty_off = off

    # Phase E.3: ``nodeBalance_dt`` no longer materialised; consumers
    # call ``_pdt_join.compute_nodeBalance_dt`` on demand.
    return (nb, nbp, None,
            Param(("n","d","t"), inflow_long.select("n","d","t","value")),
            Param(("n","d","t"), pen_up_df),
            Param(("n","d","t"), pen_dn_df),
            penalty_off)


def _load_inflow_unmargined(sd: Path, *, provider=None) -> "Param | None":
    """``solve_data/pdtNodeInflow_unmargined`` as ``Param((n, d, t))``, or None.

    The energy-margin emitters write it (the inflow before the margin) only when a margin is applied
    in some solve (``emit/_energy_margin._record_unmargined_inflow``); None otherwise.
    """
    path = sd / "pdtNodeInflow_unmargined.csv"
    if not _provider_has(provider, _provider_key(path), path):
        return None
    long = _read_wide_per_entity(path, rename={"entity": "n"}, provider=provider)
    return Param(("n", "d", "t"), long.select("n", "d", "t", "value"))


# ---------------------------------------------------------------------------
# Process-topology helpers (skipped if no processes)

def _load_process_topology(inp: Path, sd: Path, dt: pl.DataFrame,
                            *, source: "InputSource | None" = None,
                            provider: "object | None" = None,
                            dump_topology: "Callable[[Path, object], dict | None] | None" = None):
    # Δ.17b Gap B: ``process_source_sink_canonical`` produces flextool's
    # preprocessing-side collapsed shape directly from Spine (DIRECT methods
    # cross-joined; INDIRECT methods kept as 2-arc form; 2way reverse arcs
    # added; noConversion fallbacks handled).
    empty_return = {k: None for k in ("pss","pss_eff","pss_noEff","pss_dt",
                                       "flow_to_n","flow_from_n",
                                       "flow_from_commodity_eff",
                                       "flow_from_commodity_noEff",
                                       "unitsize","flow_upper","slope","commodity_price",
                                       "pss_source_canonical","pss_sink_canonical")}

    # Without a DB reader the topology comes only from ``dump_topology``, which only
    # ``flextool.engine._csv_reload`` passes (built from its seeded provider): ``None`` means "no processes",
    # otherwise a dict with the five frames below.
    if source is None:
        if dump_topology is None:
            from flextool.engine.input._solve_state import FlexToolConfigError
            raise FlexToolConfigError("process topology needs the DB reader")
        parts = dump_topology(sd, provider)
        if parts is None:
            return empty_return
        pss, pss_eff, pss_noEff = parts["pss"], parts["pss_eff"], parts["pss_noEff"]
        pss_source_canonical = parts["pss_source_canonical"]
        pss_sink_canonical = parts["pss_sink_canonical"]
    else:
        from flextool.engine.preprocess.params._projection import process_source_sink_canonical, _try_entities
        canonical = process_source_sink_canonical(source)
        if canonical.height == 0:
            return empty_return
        pss = (canonical.select("p", "source", "sink")
                          .unique()
                          .sort("p", "source", "sink"))
        pss_eff = (canonical
            .filter(pl.col("method") == "eff")
            .select("p", "source", "sink").unique()
            .sort("p", "source", "sink"))
        pss_noEff = (canonical
            .filter(pl.col("method") == "noEff")
            .select("p", "source", "sink").unique()
            .sort("p", "source", "sink"))

        # Canonical source/sink per process from the original entity tables:
        # unit__inputNode → (p, source), unit__outputNode → (p, sink),
        # connection__node__node → (p, source=node_1) and (p, sink=node_2).
        src_parts: list[pl.DataFrame] = []
        snk_parts: list[pl.DataFrame] = []
        _uin = _try_entities(source, "unit__inputNode")
        if _uin is not None and _uin.height > 0:
            src_parts.append(
                _uin.pipe(rename_to_axis, {"unit": "p", "node": "source"})
                    .select("p", "source"))
        _uout = _try_entities(source, "unit__outputNode")
        if _uout is not None and _uout.height > 0:
            snk_parts.append(
                _uout.pipe(rename_to_axis, {"unit": "p", "node": "sink"})
                     .select("p", "sink"))
        _cnn = _try_entities(source, "connection__node__node")
        if _cnn is not None and _cnn.height > 0:
            src_parts.append(
                _cnn.pipe(rename_to_axis,
                          {"connection": "p", "node_1": "source"})
                    .select("p", "source"))
            snk_parts.append(
                _cnn.pipe(rename_to_axis,
                          {"connection": "p", "node_2": "sink"})
                    .select("p", "sink"))
        pss_source_canonical = (
            pl.concat(src_parts).unique().sort("p", "source") if src_parts else None
        )
        pss_sink_canonical = (
            pl.concat(snk_parts).unique().sort("p", "sink") if snk_parts else None
        )

    # ``n`` derives from sink/source which are entity-union (``e``)
    # axis values — they may carry node OR process tokens (indirect
    # units' arcs).  Cast against the ``e`` axis (union) rather than
    # the narrower ``n`` (node-only) axis so process-vocab tokens are
    # preserved instead of silently nulled.  Downstream eb_local.n is
    # also up-cast to ``e`` below so the join composes natively in
    # Enum.  Mirrors the source-driven cascade pattern in
    # ``_block.flow_to_n_block_filtered`` /
    # ``filter_flow_n_by_block``.
    _enums = get_global_axis_enums()
    flow_to_n   = pss.with_columns(
        n=cast_dim(pl.col("sink"), _enums, "e"))
    flow_from_n = pss.with_columns(
        n=cast_dim(pl.col("source"), _enums, "e"))

    # Every arc enters the balance of both its nodes; a node of a coarse new_stepduration group sums the
    # flows over its coarse step (nodeBalanceCoarse_eq).

    cn = _provider_read(provider, "input/commodity__node",
                          inp / "commodity__node.csv")
    # CSV read produces Utf8 columns; project to the canonical axis
    # column names so the joins against pss_eff/pss_noEff line up on
    # Enum-typed source/sink columns (the union ``e`` axis).  The
    # ``node → source`` / ``node → sink`` renames cast against ``e``
    # via rename_to_axis.
    cn_as_source = cn.pipe(rename_to_axis,
                            {"node": "source", "commodity": "c"})
    cn_as_sink = cn.pipe(rename_to_axis,
                          {"node": "sink", "commodity": "c"})
    flow_from_commodity_eff = (pss_eff
        .join(cn_as_source, on="source", how="inner")
        .select("p","source","sink","c"))
    flow_from_commodity_noEff = (pss_noEff
        .join(cn_as_source, on="source", how="inner")
        .select("p","source","sink","c"))
    # §2.4 commodity sell: sink-side flow into a commodity-priced node.
    # No slope correction — straight v_flow * unitsize * commodity_price.
    flow_to_commodity = (pss
        .join(cn_as_sink, on="sink", how="inner")
        .select("p","source","sink","c"))

    # ``p_unitsize`` is overwritten by ``apply_derived_b.p_unitsize_from_source``
    # but the returned ``unitsize`` Param is also consumed inline by
    # ``_load_profiles`` (None → blank profile dict) and ``_load_storage``
    # (used in cap_pd / state_unitsize cascades).  Keep the seed.
    # TODO(Δ.12c+): when ``_load_profiles`` / ``_load_storage`` consume the
    # source-driven p_unitsize via flex_data, drop this seed read.
    unitsize_long = _read_unitsize(sd / "p_entity_unitsize.csv", provider=provider)
    unitsize_p = (unitsize_long.pipe(rename_to_axis, {"e": "p"})
                       .filter(pl.col("p").is_in(pss["p"].unique())))

    # ``p_slope`` is produced by ``apply_derived_b.p_slope_from_source``; this provider seed is kept for
    # synthetic per-sub-solve names. The DB overlay normally wins.
    slope_long = _read_wide_per_entity(sd / "pdtProcess_slope.csv",
                                         rename={"entity":"p"},
                                         provider=provider)
    # Δ.17c Gap C: ``p_commodity_price`` produced authoritatively by
    # ``apply_direct_params`` via ``p_commodity_price_from_source``
    # (uses the ``_param_shapes`` resolver — scalar / 1d_map[period] /
    # 1d_map[time] cascade with explicit allow-list).  Local
    # pdtCommodity.csv slice dropped.
    cp_long = None

    # flow_upper is the canonical ``p_flow_max.csv`` long-format file
    # the .mod reads via ``table data IN`` (`[process, source, sink,
    # period, time], p_flow_max~value`).
    # TODO(Δ.12c+): no override-chain helper covers ``p_flow_upper`` yet —
    # the preprocessed p_flow_max.csv bakes in invest_max_cum etc. that
    # the source-driven path would have to recompute.
    flow_upper_psskdt = _read_p_flow_max(sd / "p_flow_max.csv", provider=provider)

    return dict(
        pss = pss,
        pss_eff = pss_eff,
        pss_noEff = pss_noEff,
        # Phase E.3: ``pss_dt`` is no longer materialised here; consumers
        # call ``_pdt_join.compute_pss_dt`` on demand.
        pss_dt = None,
        flow_to_n = flow_to_n,
        flow_from_n = flow_from_n,
        flow_from_commodity_eff = flow_from_commodity_eff,
        flow_from_commodity_noEff = flow_from_commodity_noEff,
        flow_to_commodity = flow_to_commodity,
        pss_source_canonical = pss_source_canonical,
        pss_sink_canonical   = pss_sink_canonical,
        unitsize = Param(("p",), unitsize_p.select("p","value")),
        flow_upper = Param(("p","source","sink","d","t"), flow_upper_psskdt),
        slope = Param(("p","d","t"), slope_long.select("p","d","t","value")),
        # Δ.17c Gap C: ``p_commodity_price`` populated authoritatively by
        # ``apply_direct_params``.  The seed used to materialise a fully-
        # zero-filled (c, d, t) Param for fixtures with no explicit price
        # (preprocessed pdtCommodity.csv emits 0.0 rows for every cell);
        # the model/core/ ``PROCESSES`` invariant requires the field non-None.
        # We satisfy it with an empty Param as a placeholder; ``Sum(Where(
        # ..., ...) * d.p_commodity_price * ...)`` joins yield no rows
        # which is the same behaviour as zero-filled rows in the LP.
        commodity_price = (Param(("c","d","t"), cp_long)
                            if cp_long is not None
                            else Param(("c","d","t"),
                                        pl.DataFrame(schema={
                                            "c": pl.Utf8, "d": pl.Utf8,
                                            "t": pl.Utf8,
                                            "value": pl.Float64}))),
    )
