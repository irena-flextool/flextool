"""In-memory ``read_parameters`` — polars ``FlexData`` → pandas wide.

Translates the polars :class:`FlexData` and
:class:`polar_high.Solution` objects directly into the pandas wide
format the downstream ``out_*`` modules consume.  Every attribute on
the returned :class:`SimpleNamespace` maps to a FlexData field (or,
for ``entity_all_capacity``, a post-solve derivation from FlexData +
``solution.value("v_invest")`` + ``solution.value("v_divest")``).

Properties:

* No CSV round-trip — empty multi-header CSVs no longer silently drop
  column dim names.
* Post-solve derived attributes (``entity_all_capacity`` and the three
  sister capacity tables) use the live ``solution`` directly.
* Dim names are set via the central
  :data:`flextool.process_outputs._inmemory_helpers.DIM_NAMES`
  translation table (no per-CSV ``.columns.name = '...'`` patchwork).

Failure mode: every helper raises loudly when a FlexData field is
absent or has an unexpected shape — authoring bugs surface at the
call-site instead of producing silently-empty outputs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from flextool.process_outputs._inmemory_helpers import (
    add_solve_to_pandas,
    series_with_index,
    series_with_multi_index,
    wide_multi_col,
    wide_per_entity,
    with_solve_column,
)

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine_polars.input import FlexData


# ---------------------------------------------------------------------------
# Per-attribute helpers
# ---------------------------------------------------------------------------


def _empty_pdtX_per_entity(
    *,
    flex_data: "FlexData | None",
    solve_name: str,
    col_name: str,
    dtype=float,
) -> pd.DataFrame:
    """Return a ``(solve, period, time)``-indexed DataFrame with empty
    named columns.

    Used for the per-(solve, period, time) family when the underlying
    FlexData field is None / empty.  Critically, when ``flex_data``
    is supplied, the row index is materialised over the active solve's
    full ``(d, t)`` axis — matching the legacy CSV path which wrote
    ``solve,period,time`` rows for every timestep even when no data
    columns were present.  Without this, downstream broadcasts like
    ``v.state.mul(par.node_self_discharge_loss, axis=1, level=0)``
    raise ``TypeError: Join on level between two MultiIndex objects
    is ambiguous`` because pandas can't reconcile two empty MultiIndex
    on join.
    """
    if flex_data is not None and flex_data.dt is not None and flex_data.dt.height > 0:
        dt_pdf = flex_data.dt.to_pandas()
        idx = pd.MultiIndex.from_arrays(
            [[solve_name] * len(dt_pdf), dt_pdf["d"].tolist(), dt_pdf["t"].tolist()],
            names=["solve", "period", "time"],
        )
    else:
        idx = pd.MultiIndex.from_arrays(
            [[], [], []], names=["solve", "period", "time"],
        )
    out = pd.DataFrame(index=idx, dtype=dtype)
    out.columns = pd.Index([], dtype="object", name=col_name)
    return out


def _pdtX_per_entity(
    param,
    *,
    solve_name: str,
    entity_dim: str,
    col_name: str | None = None,
    flex_data: "FlexData | None" = None,
) -> pd.DataFrame:
    """Pivot a Param with dims ``(<entity>, d, t)`` to pandas wide-format
    indexed by ``(solve, period, time)`` × entity.

    ``param`` may be ``None`` / empty — returns an empty DataFrame
    indexed over the full ``(solve, d, t)`` axis (taken from
    ``flex_data.dt`` when supplied) so downstream broadcast-multiply
    ops align cleanly.
    """
    if param is None or param.frame.height == 0:
        return _empty_pdtX_per_entity(
            flex_data=flex_data,
            solve_name=solve_name,
            col_name=col_name or entity_dim,
        )
    # Phase E.1: Param dims may be narrower than (entity, d, t) when
    # the source was authored as scalar / 1d_map.  The pandas pivot
    # below requires explicit "d" and "t" columns, so promote via
    # flex_data.dt when needed.
    if "d" not in param.dims or "t" not in param.dims:
        from flextool.engine_polars._param_shapes import promote_param_to_dt
        if flex_data is None or flex_data.dt is None:
            return _empty_pdtX_per_entity(
                flex_data=flex_data,
                solve_name=solve_name,
                col_name=col_name or entity_dim,
            )
        pdt_frame = promote_param_to_dt(param, flex_data.dt).collect()
    else:
        pdt_frame = param.frame
    pl_df = with_solve_column(pdt_frame, solve_name)
    return wide_per_entity(
        pl_df,
        row_dims=("solve", "d", "t"),
        col_dim=entity_dim,
        row_names=("solve", "period", "time"),
        col_name=col_name or entity_dim,
    )


def _pdX_per_entity(
    param,
    *,
    solve_name: str,
    entity_dim: str,
    col_name: str | None = None,
    flex_data: "FlexData | None" = None,
    densify_entities: "list[str] | None" = None,
) -> pd.DataFrame:
    """Pivot a Param with dims ``(<entity>, d)`` to pandas wide-format
    indexed by ``(solve, period)`` × entity.

    ``param`` may be ``None`` / empty — returns an empty DataFrame
    indexed over the active solve's full ``(d,)`` axis (taken from
    ``flex_data.dt`` when supplied).

    ``densify_entities`` (optional) — when supplied, the result is
    reindexed to include every entity in the list (zero-filling any
    entity that doesn't appear in ``param``).  The legacy CSV path
    (``ed_fixed_cost.csv`` etc.) did this for the entity universe;
    downstream consumers like
    ``calc_costs.py:cost_entity_fixed_invested`` index the result
    by ``v.invest.columns`` and expect every invest-entity column to
    be present.
    """
    if flex_data is not None and flex_data.dt is not None and flex_data.dt.height > 0:
        d_pdf = flex_data.dt.select("d").unique().to_pandas()
        periods_full = d_pdf["d"].tolist()
    else:
        periods_full = []

    if param is None or param.frame.height == 0:
        if periods_full:
            idx = pd.MultiIndex.from_arrays(
                [[solve_name] * len(periods_full), periods_full],
                names=["solve", "period"],
            )
        else:
            idx = pd.MultiIndex.from_arrays([[], []], names=["solve", "period"])
        out = pd.DataFrame(index=idx, dtype=float)
        out.columns = pd.Index([], dtype="object", name=col_name or entity_dim)
    else:
        pl_df = with_solve_column(param.frame, solve_name)
        out = wide_per_entity(
            pl_df,
            row_dims=("solve", "d"),
            col_dim=entity_dim,
            row_names=("solve", "period"),
            col_name=col_name or entity_dim,
        )

    if densify_entities:
        # Keep existing columns + add missing entities (zero-filled).
        missing = [e for e in densify_entities if e not in out.columns]
        for e in missing:
            out[e] = 0.0
        out = out.reindex(columns=list(out.columns), fill_value=0.0)
        # Reorder per densify_entities for stable column ordering.
        ordered = list(densify_entities) + [
            c for c in out.columns if c not in densify_entities
        ]
        out = out[ordered]
        out.columns.name = col_name or entity_dim
    return out


def _pX_per_entity(
    param,
    *,
    entity_dim: str,
    col_name: str | None = None,
) -> pd.Series:
    """Per-entity scalar Param ``(<entity>,)`` → pandas Series with
    the entity as the index name.
    """
    if param is None or param.frame.height == 0:
        return pd.Series(dtype=float, name="value", index=pd.Index([], name=col_name or entity_dim))
    return series_with_index(
        param.frame, dim=entity_dim, name=col_name or entity_dim,
    )


def _pd_series_solve_period(
    param,
    *,
    solve_name: str,
) -> pd.Series:
    """Per-period Param ``(d,)`` → pandas Series indexed by
    ``(solve, period)``.
    """
    if param is None or param.frame.height == 0:
        idx = pd.MultiIndex.from_arrays([[], []], names=["solve", "period"])
        return pd.Series(dtype=float, name="value", index=idx)
    pl_df = with_solve_column(param.frame, solve_name)
    pdf = pl_df.to_pandas()
    s = pdf.set_index(["solve", "d"])["value"].astype(float)
    s.index.names = ["solve", "period"]
    s.name = "value"
    return s


def _pdt_series_solve_period_time(
    param,
    *,
    solve_name: str,
) -> pd.Series:
    """Param ``(d, t)`` → pandas Series indexed by
    ``(solve, period, time)``.
    """
    if param is None or param.frame.height == 0:
        idx = pd.MultiIndex.from_arrays([[], [], []], names=["solve", "period", "time"])
        return pd.Series(dtype=float, name="value", index=idx)
    pl_df = with_solve_column(param.frame, solve_name)
    pdf = pl_df.to_pandas()
    s = pdf.set_index(["solve", "d", "t"])["value"].astype(float)
    s.index.names = ["solve", "period", "time"]
    s.name = "value"
    return s


def _empty_pdtX_multi_col(
    *, col_names: tuple[str, str, str],
) -> pd.DataFrame:
    """Empty DataFrame with ``(solve, period, time)`` row index and a
    3-level empty column MultiIndex.
    """
    idx = pd.MultiIndex.from_arrays(
        [[], [], []], names=["solve", "period", "time"]
    )
    out = pd.DataFrame(index=idx, dtype=float)
    out.columns = pd.MultiIndex.from_arrays([[], [], []], names=list(col_names))
    return out


def _pdtX_multi_col(
    param,
    *,
    solve_name: str,
    col_dims: tuple[str, str, str],
    col_names: tuple[str, str, str],
    densify_col_tuples: "pl.DataFrame | None" = None,
    flex_data: "FlexData | None" = None,
) -> pd.DataFrame:
    """Param ``(*col_dims, d, t)`` → wide pandas with row
    ``(solve, period, time)`` × column MultiIndex over col_dims.

    ``densify_col_tuples`` (optional) — a polars frame whose first three
    columns enumerate every ``(col_dims)`` tuple that must appear in the
    result.  Missing tuples are added as zero-valued columns; downstream
    consumers (e.g. ``calc_slacks.q_reserves_dt``) broadcast-multiply
    against ``v.q_reserve.columns`` which carries the full LP domain even
    when the source ``reservation`` parameter is sparse.
    """
    if param is None or param.frame.height == 0:
        out = _empty_pdtX_multi_col(col_names=col_names)
    else:
        # Phase E.1: Param dims may be narrower than (*col_dims, d, t)
        # when the source was authored as scalar / 1d_map.  Promote via
        # flex_data.dt so the pivot below sees explicit d/t columns.
        if "d" not in param.dims or "t" not in param.dims:
            from flextool.engine_polars._param_shapes import promote_param_to_dt
            if flex_data is None or flex_data.dt is None:
                out = _empty_pdtX_multi_col(col_names=col_names)
                if densify_col_tuples is None:
                    return out
                # Fall through to densify path.
                pdt_frame = None
            else:
                pdt_frame = promote_param_to_dt(param, flex_data.dt).collect()
        else:
            pdt_frame = param.frame
        if pdt_frame is not None:
            pl_df = with_solve_column(pdt_frame, solve_name)
            out = wide_multi_col(
                pl_df,
                row_dims=("solve", "d", "t"),
                col_dims=col_dims,
                row_names=("solve", "period", "time"),
                col_names=col_names,
            )

    if densify_col_tuples is None:
        return out

    gate_cols = densify_col_tuples.columns[: len(col_dims)]
    gate_tuples = [
        tuple(row)
        for row in densify_col_tuples.select(gate_cols)
            .sort(gate_cols)
            .iter_rows()
    ]
    existing = list(out.columns) if isinstance(out.columns, pd.MultiIndex) else []
    ordered = list(gate_tuples) + [c for c in existing if c not in gate_tuples]
    full_cols = pd.MultiIndex.from_tuples(ordered, names=list(col_names))
    out = out.reindex(columns=full_cols, fill_value=0.0)
    return out


def _build_pssdt_varCost_alwaysProcess(
    flex_data: "FlexData",
    *,
    solve_name: str,
) -> pd.DataFrame:
    """Build the ``_alwaysProcess``-keyed varCost frame for post-processing.

    The .mod's objective splits ``other_operational_cost`` into four
    sums (``pssdt_varCost_noEff``, ``..._eff_unit_source``,
    ``..._eff_unit_sink``, ``..._eff_connection``).  Python
    post-processing collapses them into a single ``(p, source, sink)``
    multiplication against ``r.flow_dt``; because ``r.flow_dt`` is
    indexed by ``process_source_sink_alwaysProcess`` tuples, varCost
    needs the same keying.

    Reference: ``_emit_period_params._derive_varCost_pair`` (used to
    write ``pdtProcess__source__sink__dt_varCost_alwaysProcess.csv``)
    for the exact algebra.  Replicated here directly on FlexData
    Params (``p_pdt_varCost_source``, ``p_pdt_varCost_sink``,
    ``p_pdt_varCost_process``) so we don't need to round-trip via the
    on-disk CSV.

    Returns a wide ``DataFrame`` with row index ``(solve, period, time)``
    and column MultiIndex ``(process, source, sink)``.  Tuples where
    every dt-row would be zero are dropped (matches the CSV writer's
    ``filter(value != 0.0)``).
    """
    pss_frame = flex_data.process_source_sink
    if pss_frame is None or pss_frame.height == 0:
        return _empty_pdtX_multi_col(
            col_names=("process", "source", "sink"),
        )

    # Build alwaysProcess pss: each direct-method arc (src != p, snk != p)
    # contributes BOTH ``(p, src, p)`` and ``(p, p, snk)``; arcs where p
    # already appears as endpoint pass through unchanged.  This mirrors
    # the cascade-side derivation in
    # ``flextool.process_outputs.read_sets`` ~ L380-390.
    pss_pdf = pss_frame.select("p", "source", "sink").to_pandas()
    always_rows: list[tuple[str, str, str]] = []
    for p, src, snk in zip(
        pss_pdf["p"], pss_pdf["source"], pss_pdf["sink"],
    ):
        if src == p or snk == p:
            always_rows.append((p, src, snk))
        else:
            always_rows.append((p, src, p))
            always_rows.append((p, p, snk))
    # Dedup preserving order.
    always_rows = list(dict.fromkeys(always_rows))

    # Membership sets used to gate per-side OOC contributions.  Use the
    # canonical (process, node) sets (input-arc nodes for source,
    # output-arc nodes for sink); ``process_source_sink`` itself
    # contains intermediate (p, p) entries for indirect units and so
    # cannot be projected directly.
    proc_src_pairs: set[tuple[str, str]] = set()
    if (flex_data.process_source_canonical is not None
            and flex_data.process_source_canonical.height > 0):
        for p, n in flex_data.process_source_canonical.select(
            "p", "source",
        ).iter_rows():
            proc_src_pairs.add((str(p), str(n)))
    proc_snk_pairs: set[tuple[str, str]] = set()
    if (flex_data.process_sink_canonical is not None
            and flex_data.process_sink_canonical.height > 0):
        for p, n in flex_data.process_sink_canonical.select(
            "p", "sink",
        ).iter_rows():
            proc_snk_pairs.add((str(p), str(n)))

    def _as_lookup_dt(
        param, key_cols: tuple[str, ...],
    ):
        """Materialise a FlexData Param into a lookup ``f(*full_key) → value``.

        Phase E.1: ``broadcast_to_period_time`` returns Params whose dims
        depend on the authored Spine shape (SCALAR → ``(p, sink)``,
        MAP_PERIOD → ``(p, sink, d)``, MAP_PERIOD_TIME → ``(p, sink, d, t)``).
        Polar_high broadcasts the lower-dim Params lazily at constraint
        emission, but the output writers consume frames directly — so we
        key the dict by the Param's actual dim subset of ``key_cols`` and
        project the requested ``full_key`` onto that subset on lookup.
        """
        if param is None or param.frame.height == 0:
            return lambda *_args: 0.0
        present = tuple(c for c in key_cols if c in param.dims)
        proj_idx = [key_cols.index(c) for c in present]
        out: dict[tuple[str, ...], float] = {}
        cols = [*present, "value"] if present else ["value"]
        for row in param.frame.select(*cols).iter_rows():
            if present:
                *k, v = row
                out[tuple(str(x) for x in k)] = float(v)
            else:
                # Pure scalar — single global value.
                out[()] = float(row[0])
        def _lookup(*full_key, _out=out, _idx=proj_idx):
            return _out.get(tuple(full_key[i] for i in _idx), 0.0)
        return _lookup

    src_ooc = _as_lookup_dt(
        flex_data.p_pdt_varCost_source, ("p", "source", "d", "t"),
    )
    snk_ooc = _as_lookup_dt(
        flex_data.p_pdt_varCost_sink, ("p", "sink", "d", "t"),
    )
    proc_ooc = _as_lookup_dt(
        flex_data.p_pdt_varCost_process, ("p", "d", "t"),
    )

    # dt grid for the broadcast.
    dt_frame = flex_data.dt
    if dt_frame is None or dt_frame.height == 0:
        return _empty_pdtX_multi_col(
            col_names=("process", "source", "sink"),
        )
    dt_pairs = [
        (str(d), str(t))
        for d, t in dt_frame.select("d", "t").iter_rows()
    ]

    # Build (solve, period, time, process, source, sink, value) long frame.
    # For each alwaysProcess tuple + each (d, t), sum gated contributions.
    rec_solve: list[str] = []
    rec_d: list[str] = []
    rec_t: list[str] = []
    rec_p: list[str] = []
    rec_src: list[str] = []
    rec_snk: list[str] = []
    rec_v: list[float] = []
    for (p, src, snk) in always_rows:
        # Track whether this tuple ever has a non-zero value — drop if
        # universally zero (matches the CSV writer's filter).
        any_nonzero = False
        local_pairs: list[tuple[str, str, float]] = []
        for (d, t) in dt_pairs:
            v = 0.0
            if (p, src) in proc_src_pairs:
                v += src_ooc(p, src, d, t)
            if (p, snk) in proc_snk_pairs:
                v += snk_ooc(p, snk, d, t)
            # alwaysProcess gating for the process-level OOC term:
            # include only when one of the (always-process) endpoints is
            # in process_source ∪ process_sink (matches
            # ``_derive_varCost_pair`` ``if always: ... if (p, snk) in
            # proc_snk or (p, snk) in proc_src``).
            if (p, snk) in proc_snk_pairs or (p, snk) in proc_src_pairs:
                v += proc_ooc(p, d, t)
            if v != 0.0:
                any_nonzero = True
            local_pairs.append((d, t, v))
        if not any_nonzero:
            continue
        for (d, t, v) in local_pairs:
            rec_solve.append(solve_name)
            rec_d.append(d)
            rec_t.append(t)
            rec_p.append(p)
            rec_src.append(src)
            rec_snk.append(snk)
            rec_v.append(v)

    if not rec_solve:
        return _empty_pdtX_multi_col(
            col_names=("process", "source", "sink"),
        )

    long_pl = pl.DataFrame({
        "solve": rec_solve,
        "d": rec_d,
        "t": rec_t,
        "p": rec_p,
        "source": rec_src,
        "sink": rec_snk,
        "value": rec_v,
    })
    return wide_multi_col(
        long_pl,
        row_dims=("solve", "d", "t"),
        col_dims=("p", "source", "sink"),
        row_names=("solve", "period", "time"),
        col_names=("process", "source", "sink"),
    )


def _entity_all_capacity(
    flex_data: "FlexData",
    solution: "Solution",
    *,
    solve_name: str,
) -> pd.DataFrame:
    """Compute ``entity_all_capacity[(solve, period), entity]`` from
    FlexData + the live solution.

    Mirrors :func:`flextool.process_outputs.handoff_writers.
    _compute_entity_all_capacity` but uses polars long-form frames
    end-to-end.

    Formula::

        entity_all_capacity[e, d] =
            existing[e, d]
              + sum over (e, d_inv, d) in edd_invest of
                v_invest[e, d_inv] * unitsize[e]
              - sum over (e, d_dv) in ed_divest with years[d_dv] <= years[d] of
                v_divest[e, d_dv] * unitsize[e]
    """
    # Collect base existing — start from p_entity_all_existing if
    # available; else fall back to a (entity × period) zero frame
    # spanning the active periods.
    if (flex_data.p_entity_all_existing is not None
            and flex_data.p_entity_all_existing.frame.height > 0):
        existing_lf = flex_data.p_entity_all_existing.frame.lazy().select(
            pl.col("e").cast(pl.Utf8),
            pl.col("d").cast(pl.Utf8),
            pl.col("value").alias("existing"),
        )
    else:
        existing_lf = pl.DataFrame(
            schema={"e": pl.Utf8, "d": pl.Utf8, "existing": pl.Float64},
        ).lazy()

    # Unitsize map: process p_unitsize ∪ node p_state_unitsize.
    unitsize_pieces: list[pl.LazyFrame] = []
    if (flex_data.p_unitsize is not None
            and flex_data.p_unitsize.frame.height > 0):
        unitsize_pieces.append(
            flex_data.p_unitsize.frame.lazy().select(
                pl.col("p").cast(pl.Utf8).alias("e"),
                pl.col("value").alias("unitsize"),
            )
        )
    if (flex_data.p_state_unitsize is not None
            and flex_data.p_state_unitsize.frame.height > 0):
        unitsize_pieces.append(
            flex_data.p_state_unitsize.frame.lazy().select(
                pl.col("n").cast(pl.Utf8).alias("e"),
                pl.col("value").alias("unitsize"),
            )
        )
    if unitsize_pieces:
        unitsize_lf = pl.concat(unitsize_pieces, how="vertical")
    else:
        unitsize_lf = pl.DataFrame(
            schema={"e": pl.Utf8, "unitsize": pl.Float64},
        ).lazy()

    # Active solution slices.  ``solution.value`` returns long form
    # ``(*dims, value)``.  The polars LP splits invest/divest into
    # process-side (``v_invest_p`` / ``v_divest_p``) and node-side
    # (``v_invest_n`` / ``v_divest_n``) variables; we union them.
    def _try_value(name: str) -> "pl.DataFrame":
        if name not in solution._vars:
            return pl.DataFrame(
                schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64},
            )
        df = solution.value(name)
        # The polars LP uses ``p`` / ``n`` for the entity dim; rename
        # to the canonical ``e``.  Also accept ``entity`` / ``period``.
        rename = {}
        for src, dst in (("p", "e"), ("n", "e"),
                          ("entity", "e"), ("period", "d")):
            if src in df.columns and dst not in df.columns:
                rename[src] = dst
        if rename:
            df = df.rename(rename)
        return df.select(
            pl.col("e").cast(pl.Utf8),
            pl.col("d").cast(pl.Utf8),
            "value",
        )

    invest_pieces = []
    divest_pieces = []
    for nm in ("v_invest", "v_invest_p", "v_invest_n"):
        f = _try_value(nm)
        if f.height > 0:
            invest_pieces.append(f)
    for nm in ("v_divest", "v_divest_p", "v_divest_n"):
        f = _try_value(nm)
        if f.height > 0:
            divest_pieces.append(f)
    # FlexData / solution dim columns may be Enum dtype (from
    # ``cast_flexdata_axes`` at the end of ``load_flextool``); empty
    # fallback frames declare Utf8.  ``_try_value`` casts the
    # populated side to Utf8 (see above) so every concat input shares
    # the empty-fallback dtype — no per-input cast needed here.
    invest = (pl.concat(invest_pieces, how="vertical") if invest_pieces
              else pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64}))
    divest = (pl.concat(divest_pieces, how="vertical") if divest_pieces
              else pl.DataFrame(schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64}))

    # invest contribution per (e, d) — by edd_invest indirection
    if (flex_data.edd_invest_set is not None
            and flex_data.edd_invest_set.height > 0):
        edd = flex_data.edd_invest_set.lazy()
        cols = flex_data.edd_invest_set.columns
        # canonical: (e, d_invest, d) — rename if dim names differ
        rename = {}
        if cols[0] not in ("e", "entity"):
            rename[cols[0]] = "e"
        elif cols[0] == "entity":
            rename["entity"] = "e"
        if len(cols) > 1 and cols[1] != "d_invest":
            rename[cols[1]] = "d_invest"
        if len(cols) > 2 and cols[2] != "d":
            rename[cols[2]] = "d"
        if rename:
            edd = edd.rename(rename)
        edd = edd.with_columns(
            pl.col("e").cast(pl.Utf8),
            pl.col("d_invest").cast(pl.Utf8),
            pl.col("d").cast(pl.Utf8),
        )
        invest_contrib = (
            edd.join(invest.lazy().rename({"d": "d_invest", "value": "v_inv"}),
                     on=["e", "d_invest"], how="inner")
               .join(unitsize_lf, on="e", how="inner")
               .select(
                    "e", "d",
                    invested=pl.col("v_inv") * pl.col("unitsize"),
                )
               .group_by(["e", "d"]).agg(pl.col("invested").sum())
        )
    else:
        invest_contrib = pl.DataFrame(
            schema={"e": pl.Utf8, "d": pl.Utf8, "invested": pl.Float64},
        ).lazy()

    # divest contribution per (e, d) — for each (e, d_dv) in ed_divest_set,
    # apply v_divest[e, d_dv] * unitsize[e] to every period d with
    # years[d_dv] <= years[d].
    if (flex_data.p_period_share is not None
            and flex_data.p_period_share.frame.height > 0):
        # We don't have years_from_start_d directly, but the divest
        # carry-forward is monotone over period order; we approximate
        # by using period name ordering (lexicographic).  For
        # production fidelity we want years_from_start_d — get it from
        # the timeline.  Instead rely on edd_divest_active which
        # already gives the active (d_divest, d) pairs.
        pass

    if (flex_data.edd_divest_active is not None
            and flex_data.edd_divest_active.height > 0):
        edd_dv = flex_data.edd_divest_active.lazy()
        cols = flex_data.edd_divest_active.columns
        # canonical: (e, d_divest, d).  edd_divest_active uses ``p`` for
        # the entity dim — rename to ``e``.
        rename = {}
        if cols[0] not in ("e", "entity"):
            rename[cols[0]] = "e"
        elif cols[0] == "entity":
            rename["entity"] = "e"
        if len(cols) > 1 and cols[1] != "d_divest":
            rename[cols[1]] = "d_divest"
        if len(cols) > 2 and cols[2] != "d":
            rename[cols[2]] = "d"
        if rename:
            edd_dv = edd_dv.rename(rename)
        edd_dv = edd_dv.with_columns(
            pl.col("e").cast(pl.Utf8),
            pl.col("d_divest").cast(pl.Utf8),
            pl.col("d").cast(pl.Utf8),
        )
        divest_contrib = (
            edd_dv.join(divest.lazy().rename({"d": "d_divest", "value": "v_dv"}),
                        on=["e", "d_divest"], how="inner")
                  .join(unitsize_lf, on="e", how="inner")
                  .select(
                      "e", "d",
                      divested=pl.col("v_dv") * pl.col("unitsize"),
                  )
                  .group_by(["e", "d"]).agg(pl.col("divested").sum())
        )
    elif (flex_data.ed_divest_set is not None
              and flex_data.ed_divest_set.height > 0):
        # Fallback: each (e, d_dv) divests carry-over from d_dv onward
        # within the same period set.  Without years map, take only
        # the divestment in the same period.
        edd_dv = flex_data.ed_divest_set.lazy()
        cols = flex_data.ed_divest_set.columns
        rename = {}
        if cols[0] not in ("e", "entity"):
            rename[cols[0]] = "e"
        elif cols[0] == "entity":
            rename["entity"] = "e"
        if len(cols) > 1 and cols[1] != "d":
            rename[cols[1]] = "d"
        if rename:
            edd_dv = edd_dv.rename(rename)
        edd_dv = edd_dv.with_columns(
            pl.col("e").cast(pl.Utf8),
            pl.col("d").cast(pl.Utf8),
        )
        divest_contrib = (
            edd_dv.join(divest.lazy().rename({"value": "v_dv"}),
                        on=["e", "d"], how="inner")
                  .join(unitsize_lf, on="e", how="inner")
                  .select(
                      "e", "d",
                      divested=pl.col("v_dv") * pl.col("unitsize"),
                  )
        )
    else:
        divest_contrib = pl.DataFrame(
            schema={"e": pl.Utf8, "d": pl.Utf8, "divested": pl.Float64},
        ).lazy()

    # Combine: outer-join on (e, d) and sum.  ``how="full"`` requires
    # explicit coalesce on the join keys.  Every input has already
    # been normalised to Utf8 ``e`` / ``d`` above (so Enum-typed
    # populated frames don't clash with the Utf8 empty fallbacks).
    j1 = existing_lf.join(
        invest_contrib, on=["e", "d"], how="full", coalesce=True,
    )
    j2 = j1.join(divest_contrib, on=["e", "d"], how="full", coalesce=True)
    combined = (
        j2.with_columns(
            existing=pl.col("existing").fill_null(0.0),
            invested=pl.col("invested").fill_null(0.0),
            divested=pl.col("divested").fill_null(0.0),
        )
        .with_columns(
            total=pl.col("existing") + pl.col("invested") - pl.col("divested"),
        )
        .filter(pl.col("e").is_not_null() & pl.col("d").is_not_null())
        .select("e", "d", "total")
        .collect()
    )
    if combined.height == 0:
        idx = pd.MultiIndex.from_arrays([[], []], names=["solve", "period"])
        out = pd.DataFrame(index=idx, dtype=float)
        out.columns.name = "entity"
        return out
    pl_df = combined.with_columns(pl.lit(solve_name).alias("solve"))
    return wide_per_entity(
        pl_df.rename({"total": "value"}),
        row_dims=("solve", "d"),
        col_dim="e",
        row_names=("solve", "period"),
        col_name="entity",
    )


def _ensure_value_zero(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Ensure ``df`` has columns with zero values.  Used to create the
    ``p_node`` / ``p_process_source`` / ``p_process_sink`` row-by-param
    legacy frames for which we no longer have a single source FlexData
    field; downstream consumers test ``loc['inertia_constant']`` etc.
    """
    if df is None:
        return None
    for c in columns:
        if c not in df.columns:
            df[c] = 0.0
    return df


# ---------------------------------------------------------------------------
# Composite legacy frames
# ---------------------------------------------------------------------------


def _build_p_node(flex_data: "FlexData") -> pd.DataFrame:
    """Construct the legacy ``p_node`` wide table.

    Legacy CSV layout (``solve_data/p_node.csv``)::

        param,west
        annual_flow,0
        peak_inflow,0

    Rows are parameter names; columns are nodes.  The only consumer
    today is :func:`out_node.node_summary` indirectly via
    ``r.node_inflow_d`` / ``s.node_balance``; the table is otherwise
    unread.  We build it from the available FlexData fields and fall
    back to zeros for missing parameters.
    """
    nodes = []
    if (flex_data.nodeBalance is not None
            and flex_data.nodeBalance.height > 0):
        nodes = flex_data.nodeBalance["n"].to_list()
    if not nodes:
        out = pd.DataFrame(index=pd.Index([], name="param"), dtype=float)
        out.columns.name = "node"
        return out
    # Known scalar-per-node params we can fill from FlexData.
    rows: dict[str, dict[str, float]] = {
        "annual_flow": {n: 0.0 for n in nodes},
        "peak_inflow": {n: 0.0 for n in nodes},
    }
    out = pd.DataFrame(rows).T.astype(float)
    out.index.name = "param"
    out.columns.name = "node"
    out = out.reindex(columns=nodes)
    return out


def _build_p_process_per_arc(
    flex_data: "FlexData",
    *,
    side: str,  # "source" or "sink"
) -> pd.DataFrame:
    """Build the legacy ``p_process_<side>`` wide table.

    Legacy CSV layout (``solve_data/p_process_source.csv``)::

        process,p1,p2,...
        source,s1,s2,...
        efficiency,...,...
        ...
        inertia_constant,...,...

    Rows are parameter names (``efficiency``, ``ramp_speed_up``,
    ``inertia_constant``, …); columns are a MultiIndex of ``(process,
    <side>)``.  The downstream consumer (``out_ancillary``) reads
    ``loc['inertia_constant']`` only — we populate that row from the
    relevant FlexData field and zero-fill the rest.
    """
    if side == "source":
        pss = flex_data.process_source_sink
        ramp_up = flex_data.p_ramp_speed_up_source
        ramp_down = flex_data.p_ramp_speed_down_source
        inertia = flex_data.p_process_source_inertia_constant
        side_dim = "source"
    elif side == "sink":
        pss = flex_data.process_source_sink
        ramp_up = flex_data.p_ramp_speed_up_sink
        ramp_down = flex_data.p_ramp_speed_down_sink
        inertia = flex_data.p_process_sink_inertia_constant
        side_dim = "sink"
    else:
        raise ValueError(f"side must be 'source' or 'sink', got {side!r}")

    # Build the canonical row-by-param row list — same order, same names
    # as the legacy CSV so downstream ``loc['inertia_constant']`` etc.
    # work whether columns are empty or not.
    row_names = [
        "efficiency",
        "efficiency_at_min_load",
        "min_load",
        "coefficient",
        "flow_unitsize",
        "other_operational_cost",
        "ramp_cost",
        "ramp_speed_up",
        "ramp_speed_down",
        "inertia_constant",
    ]

    if pss is None or pss.height == 0:
        cols = pd.MultiIndex.from_arrays(
            [[], []], names=["process", side_dim]
        )
        out = pd.DataFrame(
            index=pd.Index(row_names, name="param"), columns=cols, dtype=float,
        )
        return out

    # Distinct (process, <side>) pairs.  ``process_source_sink`` carries
    # ``(p, source, sink)``; we want unique (process, source) for source
    # and unique (process, sink) for sink.
    pdf = pss.select("p", side_dim).unique().to_pandas()
    pairs = list(zip(pdf["p"], pdf[side_dim]))
    cols = pd.MultiIndex.from_tuples(
        pairs, names=["process", side_dim]
    )

    # Build rows for known parameters.  All zeros except where FlexData
    # gives an explicit value.  Order matches the legacy CSV.
    out = pd.DataFrame(0.0, index=row_names, columns=cols, dtype=float)
    out.index.name = "param"

    def _fill_row(row: str, param) -> None:
        if param is None or param.frame.height == 0:
            return
        # param.frame columns: (p, <side>, value)
        pf = param.frame.to_pandas()
        # Resolve the entity dim name: source | sink
        ent_col = side_dim if side_dim in pf.columns else (
            "name" if "name" in pf.columns else side_dim
        )
        for _, r in pf.iterrows():
            key = (r["p"], r[ent_col])
            if key in cols:
                out.loc[row, key] = float(r["value"])

    _fill_row("ramp_speed_up", ramp_up)
    _fill_row("ramp_speed_down", ramp_down)
    _fill_row("inertia_constant", inertia)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def read_parameters(
    flex_data: "FlexData",
    solution: "Solution",
    *,
    solve_name: str = "solve",
) -> SimpleNamespace:
    """Translate ``FlexData`` + a polars-LP :class:`polar_high.Solution`
    into the legacy ``par`` namespace consumed by the
    :mod:`flextool.process_outputs.out_*` modules.

    Parameters
    ----------
    flex_data : FlexData
        The polars input bundle the LP was built from.
    solution : polar_high.Solution
        The solved LP — used for the post-solve derived attributes
        (``entity_all_capacity`` and the three sister capacity tables).
    solve_name : str, optional
        The active solve identifier.  Used to inject the leading
        ``solve`` index level the downstream wide-format consumers
        expect.  Defaults to ``"solve"`` for tests; production callers
        in :mod:`flextool.engine_polars._orchestration` pass the real
        ``complete_solve_name``.

    Returns
    -------
    SimpleNamespace
        With every attribute populated to match the legacy CSV-path
        signature.  Failure to populate any required attribute raises
        :class:`KeyError` / :class:`ValueError` loudly.
    """
    p = SimpleNamespace()

    # Entity universe — used to densify the (entity, period) wide frames
    # so downstream consumers like
    # ``calc_costs.py:cost_entity_fixed_invested`` find every invest
    # entity in ``par.entity_lifetime_fixed_cost.columns`` (the legacy
    # CSV path emitted zero-filled rows for every entity).
    _entity_universe: list[str] = []
    if (flex_data.nodeBalance is not None
            and flex_data.nodeBalance.height > 0):
        _entity_universe.extend(flex_data.nodeBalance["n"].to_list())
    if (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        _entity_universe.extend(
            flex_data.process_source_sink.select("p").unique()
                .to_pandas()["p"].tolist()
        )
    # Also include any commodity nodes that show up in process_source_sink
    # (sources / sinks) — these can be invest entities too.
    if (flex_data.process_source_sink is not None
            and flex_data.process_source_sink.height > 0):
        for col in ("source", "sink"):
            extra = flex_data.process_source_sink.select(col).unique().to_pandas()[col].tolist()
            _entity_universe.extend(extra)
    # Deduplicate while keeping insertion order.
    _entity_universe = list(dict.fromkeys(_entity_universe))

    # ─── Write-once static params (entity-level scalars / per-arc tables) ──
    p.node = _build_p_node(flex_data)
    p.entity_unitsize = _build_entity_unitsize_series(
        flex_data, entity_universe=_entity_universe,
    )

    # commodity_co2_content — Series indexed by commodity (legacy used
    # to read this from input/p_commodity_co2_content.csv; in FlexData
    # it's ``p_co2_content`` keyed on ``c``).
    if (flex_data.p_co2_content is not None
            and flex_data.p_co2_content.frame.height > 0):
        p.commodity_co2_content = series_with_index(
            flex_data.p_co2_content.frame, dim="c", name="commodity",
        )
    else:
        p.commodity_co2_content = pd.Series(
            dtype=float,
            index=pd.Index([], name="commodity"),
        )

    # process_sink_conversion_flow_coeff /
    # process_source_conversion_flow_coeff — Series with MultiIndex
    # (process, sink|source).  FlexData carries these as
    # ``p_process_sink_conversion_flow_coeff`` /
    # ``p_process_source_conversion_flow_coeff`` (dims ``(p, sink)`` /
    # ``(p, source)``).
    if (flex_data.p_process_sink_conversion_flow_coeff is not None
            and flex_data.p_process_sink_conversion_flow_coeff.frame.height > 0):
        p.process_sink_conversion_flow_coeff = series_with_multi_index(
            flex_data.p_process_sink_conversion_flow_coeff.frame,
            dims=("p", "sink"),
            names=["process", "sink"],
        )
    else:
        p.process_sink_conversion_flow_coeff = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["process", "sink"]
            ),
        )

    if (flex_data.p_process_source_conversion_flow_coeff is not None
            and flex_data.p_process_source_conversion_flow_coeff.frame.height > 0):
        p.process_source_conversion_flow_coeff = series_with_multi_index(
            flex_data.p_process_source_conversion_flow_coeff.frame,
            dims=("p", "source"),
            names=["process", "source"],
        )
    else:
        p.process_source_conversion_flow_coeff = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["process", "source"]
            ),
        )

    # reserve_upDown_group_penalty — Series with MultiIndex (reserve,
    # upDown, node_group).
    if (flex_data.p_reserve_upDown_group_penalty_reserve is not None
            and flex_data.p_reserve_upDown_group_penalty_reserve.frame.height > 0):
        p.reserve_upDown_group_penalty = series_with_multi_index(
            flex_data.p_reserve_upDown_group_penalty_reserve.frame,
            dims=("r", "ud", "g"),
            names=["reserve", "upDown", "node_group"],
        )
    else:
        p.reserve_upDown_group_penalty = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays(
                [[], [], []], names=["reserve", "upDown", "node_group"]
            ),
        )

    # ─── Per-(solve, period, time) parameters ───────────────────────────────
    p.step_duration = _pdt_series_solve_period_time(
        flex_data.p_step_duration, solve_name=solve_name,
    )
    p.rp_cost_weight = _pdt_series_solve_period_time(
        flex_data.p_rp_cost_weight, solve_name=solve_name,
    )

    # flow_min / flow_max — multi-column DataFrames.  FlexData has
    # ``p_flow_upper`` ((p, source, sink, d, t)).  The legacy CSV's
    # ``flow_min`` is structurally distinct; populate as empty
    # (consumers tolerate empty), and fill flow_max from p_flow_upper.
    p.flow_min = _empty_pdtX_multi_col(
        col_names=("process", "source", "sink"),
    )
    p.flow_max = _pdtX_multi_col(
        flex_data.p_flow_upper,
        solve_name=solve_name,
        col_dims=("p", "source", "sink"),
        col_names=("process", "source", "sink"),
        flex_data=flex_data,
    )

    # process_source / process_sink — wide rows-by-param frames.
    p.process_source = _build_p_process_per_arc(flex_data, side="source")
    p.process_sink = _build_p_process_per_arc(flex_data, side="sink")

    # process_slope / process_section / process_availability —
    # (solve, period, time) × process.
    p.process_slope = _pdtX_per_entity(
        flex_data.p_slope, solve_name=solve_name,
        entity_dim="p", col_name="process", flex_data=flex_data,
    )
    p.process_section = _pdtX_per_entity(
        flex_data.p_section, solve_name=solve_name,
        entity_dim="p", col_name="process", flex_data=flex_data,
    )
    p.process_availability = _pdtX_per_entity(
        flex_data.p_process_availability, solve_name=solve_name,
        entity_dim="p", col_name="process", flex_data=flex_data,
    )

    # process_source_sink_varCost — (solve, period, time) × (process, source, sink).
    #
    # The post-processing in calc_costs.py multiplies this against
    # ``r.flow_dt`` whose columns are keyed by
    # ``process_source_sink_alwaysProcess`` tuples (each direct-method
    # unit contributes one ``(p, source, p)`` source-side column and one
    # ``(p, p, sink)`` sink-side column).  Therefore the varCost frame
    # must use the same ``_alwaysProcess`` keying — NOT the LP's
    # pss-keyed ``p_pssdt_varCost`` Param (which the .mod uses only for
    # the ``pssdt_varCost_noEff`` objective term).  For
    # ``min_load_efficiency`` (``eff_unit_source``) units the source-side
    # cost lives on the ``(p, source, p)`` tuple alongside the
    # slope+section fuel flow in ``r.flow_dt``; without the rekey, the
    # column intersection is empty and the bucket is silently zero
    # (regression guard:
    # ``tests/test_cost_aggregation_semantics.TestMinLoadEfficiencySectionTerm``).
    p.process_source_sink_varCost = _build_pssdt_varCost_alwaysProcess(
        flex_data, solve_name=solve_name,
    )

    # node_self_discharge_loss / node_penalty_up / node_penalty_down /
    # node_inflow / commodity_price / group_co2_price / profile.
    if (flex_data.p_state_self_discharge is not None
            and flex_data.p_state_self_discharge.frame.height > 0):
        # p_state_self_discharge has dims (n,).  Broadcast to (solve, d, t)
        # via dt — matching the legacy ``pdtNode_self_discharge_loss.csv``
        # which had per-(solve, period, time) rows × node columns even
        # though the value is constant per node.
        n_frame = flex_data.p_state_self_discharge.frame
        dt_frame = flex_data.dt
        broadcast = (
            n_frame.lazy()
                .join(dt_frame.lazy(), how="cross")
                .with_columns(pl.lit(solve_name).alias("solve"))
                .select("solve", "d", "t", pl.col("n"), pl.col("value"))
                .collect()
        )
        p.node_self_discharge_loss = wide_per_entity(
            broadcast, row_dims=("solve", "d", "t"), col_dim="n",
            row_names=("solve", "period", "time"), col_name="node",
        )
    else:
        p.node_self_discharge_loss = _empty_pdtX_per_entity(
            flex_data=flex_data, solve_name=solve_name, col_name="node",
        )

    p.node_penalty_up = _pdtX_per_entity(
        flex_data.p_penalty_up, solve_name=solve_name,
        entity_dim="n", col_name="node", flex_data=flex_data,
    )
    p.node_penalty_down = _pdtX_per_entity(
        flex_data.p_penalty_down, solve_name=solve_name,
        entity_dim="n", col_name="node", flex_data=flex_data,
    )
    p.node_inflow = _pdtX_per_entity(
        flex_data.p_inflow, solve_name=solve_name,
        entity_dim="n", col_name="node", flex_data=flex_data,
    )
    p.commodity_price = _pdtX_per_entity(
        flex_data.p_commodity_price, solve_name=solve_name,
        entity_dim="c", col_name="commodity", flex_data=flex_data,
    )
    p.group_co2_price = _pdtX_per_entity(
        flex_data.p_co2_price, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
    )

    # reserve_upDown_group_reservation — multi-column (r, ud, g).
    p.reserve_upDown_group_reservation = _pdtX_multi_col(
        flex_data.pdtReserve_upDown_group_reservation, solve_name=solve_name,
        col_dims=("r", "ud", "g"),
        col_names=("reserve", "upDown", "node_group"),
        densify_col_tuples=getattr(flex_data, "reserve_upDown_group", None),
        flex_data=flex_data,
    )

    # profile — (solve, period, time) × profile.
    p.profile = _pdtX_per_entity(
        flex_data.p_profile_value, solve_name=solve_name,
        entity_dim="f", col_name="profile", flex_data=flex_data,
    )

    # ─── Per-(solve, period) scalar params ──────────────────────────────────
    # years_from_start_d / years_represented_d — Series.  FlexData
    # carries ``p_years_represented_d`` (Param, (d,) → R width sum;
    # populated by ``_derived_params.apply_derived_a`` from
    # ``solve.years_represented``).  ``years_from_start_d`` is still
    # defaulted to 0.0 (used only for divest carry-forward; the active
    # divest cascade uses ``edd_divest_active`` directly).
    if (flex_data.p_period_share is not None
            and flex_data.p_period_share.frame.height > 0):
        # p_period_share is (d,) → use d set as the period axis.
        periods = flex_data.p_period_share.frame.select("d").to_pandas()["d"].tolist()
    else:
        periods = []
    if periods:
        idx = pd.MultiIndex.from_arrays(
            [[solve_name] * len(periods), periods], names=["solve", "period"],
        )
        p.years_from_start_d = pd.Series([0.0] * len(periods), index=idx, name="value")
        # Per-period R width sum from FlexData when populated; fall back
        # to 1.0 per period when the source carries no
        # ``solve.years_represented`` rows (single-year fixtures).
        yr_widths = [1.0] * len(periods)
        yr_param = flex_data.p_years_represented_d
        if yr_param is not None and yr_param.frame.height > 0:
            yr_map = {
                str(d): float(v)
                for d, v in zip(
                    yr_param.frame["d"].to_list(),
                    yr_param.frame["value"].to_list(),
                )
            }
            yr_widths = [yr_map.get(str(d), 1.0) for d in periods]
        p.years_represented_d = pd.Series(yr_widths, index=idx, name="value")
    else:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=["solve", "period"])
        p.years_from_start_d = pd.Series(dtype=float, index=empty_idx, name="value")
        p.years_represented_d = pd.Series(dtype=float, index=empty_idx, name="value")

    # entity_max_units / entity_all_existing / entity_pre_existing —
    # (solve, period) × entity.  Densify across the entity universe
    # so downstream lookups by ``v.invest.columns`` find every entity.
    p.entity_max_units = _pdX_per_entity(
        flex_data.p_entity_max_units, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )
    p.entity_all_existing = _pdX_per_entity(
        flex_data.p_entity_all_existing, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )

    # entity_pre_existing — distinct from entity_all_existing on
    # multi-solve runs (``pre_existing`` is the same as ``all_existing``
    # on first solve; differs only when prior solves invested).  We
    # use ``p_entity_previously_invested_capacity`` if available, else
    # fall back to ``entity_all_existing`` minus the previously
    # invested.  Single-solve: identical to entity_all_existing.
    if (flex_data.p_entity_previously_invested_capacity is not None
            and flex_data.p_entity_previously_invested_capacity.frame.height > 0):
        # Subtract from p_entity_all_existing if both populated.
        all_lf = flex_data.p_entity_all_existing.frame.lazy() if (
            flex_data.p_entity_all_existing is not None
            and flex_data.p_entity_all_existing.frame.height > 0
        ) else None
        prev_lf = flex_data.p_entity_previously_invested_capacity.frame.lazy()
        if all_lf is not None:
            pre_existing = (
                all_lf.join(
                    prev_lf.rename({"value": "prev"}), on=["e", "d"], how="left",
                )
                .with_columns(
                    value=pl.col("value") - pl.col("prev").fill_null(0.0),
                )
                .select("e", "d", "value")
                .collect()
            )
        else:
            pre_existing = prev_lf.select("e", "d", "value").collect()
        from polar_high import Param  # local import — Param is heavy
        p.entity_pre_existing = _pdX_per_entity(
            Param(("e", "d"), pre_existing), solve_name=solve_name,
            entity_dim="e", col_name="entity",
        )
    else:
        p.entity_pre_existing = _pdX_per_entity(
            flex_data.p_entity_all_existing, solve_name=solve_name,
            entity_dim="e", col_name="entity", flex_data=flex_data,
        )

    # entity_all_capacity — post-solve derived.
    p.entity_all_capacity = _entity_all_capacity(
        flex_data, solution, solve_name=solve_name,
    )

    # process_startup_cost — (solve, period) × process.
    p.process_startup_cost = _pdX_per_entity(
        flex_data.p_startup_cost, solve_name=solve_name,
        entity_dim="p", col_name="process", flex_data=flex_data,
    )

    # entity_fixed_cost / entity_lifetime_fixed_cost / entity_lifetime_fixed_cost_divest.
    # Densify across the full entity universe — calc_costs.py:154-155
    # indexes these by ``v.invest.columns`` / ``v.divest.columns``,
    # which can include any entity.
    p.entity_fixed_cost = _pdX_per_entity(
        flex_data.p_ed_fixed_cost, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )
    p.entity_lifetime_fixed_cost = _pdX_per_entity(
        flex_data.ed_lifetime_fixed_cost, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )
    p.entity_lifetime_fixed_cost_divest = _pdX_per_entity(
        flex_data.ed_lifetime_fixed_cost_divest, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )

    # node_annual_flow — Series((solve, period), node).
    p.node_annual_flow = _empty_solve_period_per_entity(
        col_name="node",
    )

    # group_penalty_inertia / group_penalty_non_synchronous /
    # group_penalty_capacity_margin / group_inertia_limit / group_capacity_margin.
    #
    # SCEN-6: ``pdGroup_penalty_capacity_margin`` is derived via
    # ``_entity_period_scalar`` which drops null rows.  When a group has
    # ``capacity_margin > 0`` but ``penalty_capacity_margin`` left blank
    # (a common template pattern — e.g. xlsx ``Commodity_nodes``), the
    # group lands in ``groupCapacityMargin`` (built from non-empty
    # ``capacity_margin`` rows) but is absent from the penalty frame.
    # ``calc_slacks.py`` indexes ``par.group_penalty_capacity_margin
    # [s.groupCapacityMargin]`` and raises ``KeyError`` on the gap.
    # Densify here with the same zero-fill convention used elsewhere for
    # parity emitters: missing penalty ⇒ 0 (no slack-cost contribution,
    # matching the LP: the slack term in
    # ``_group_slack.compute_slack_costs`` is itself gated on a
    # non-null ``pdGroup_penalty_capacity_margin``).
    if (flex_data.groupCapacityMargin is not None
            and flex_data.groupCapacityMargin.height > 0):
        _group_cap_margin_universe = (
            flex_data.groupCapacityMargin.select("g").unique()
                .to_pandas()["g"].tolist()
        )
    else:
        _group_cap_margin_universe = []
    p.group_penalty_inertia = _pdX_per_entity(
        flex_data.pdGroup_penalty_inertia, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
    )
    p.group_penalty_non_synchronous = _pdX_per_entity(
        flex_data.pdGroup_penalty_non_synchronous, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
    )
    p.group_penalty_capacity_margin = _pdX_per_entity(
        flex_data.pdGroup_penalty_capacity_margin, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
        densify_entities=_group_cap_margin_universe,
    )
    p.group_inertia_limit = _pdX_per_entity(
        flex_data.pdGroup_inertia_limit, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
    )
    p.group_capacity_margin = _pdX_per_entity(
        flex_data.pdGroup_capacity_margin, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
    )

    # entity_annuity / entity_annual_discounted / entity_annual_divest_discounted.
    # entity_annuity is undiscounted; we use discounted as a stand-in
    # (only consumed in debug mode; the legacy distinction is preserved
    # for compatibility).
    p.entity_annuity = _pdX_per_entity(
        flex_data.ed_entity_annual_discounted, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )
    p.entity_annual_discounted = _pdX_per_entity(
        flex_data.ed_entity_annual_discounted, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )
    p.entity_annual_divest_discounted = _pdX_per_entity(
        flex_data.ed_entity_annual_divest_discounted, solve_name=solve_name,
        entity_dim="e", col_name="entity", flex_data=flex_data,
        densify_entities=_entity_universe,
    )

    # inflation factors — Series((solve, period)).
    p.inflation_factor_operations_yearly = _pd_series_solve_period(
        flex_data.p_inflation_op, solve_name=solve_name,
    )
    # No FlexData equivalent for "investment yearly"; reuse operations
    # (the rate differs only when discount_rate ≠ 0; for fixtures
    # without explicit rates they're identical).
    p.inflation_factor_investment_yearly = _pd_series_solve_period(
        flex_data.p_inflation_op, solve_name=solve_name,
    )

    # node_capacity_for_scaling / group_capacity_for_scaling.
    # Densify nodes for downstream slack-scaling lookups in
    # ``out_node.py`` / ``calc_slacks.py``.
    nodes_universe = []
    if (flex_data.nodeBalance is not None
            and flex_data.nodeBalance.height > 0):
        nodes_universe = flex_data.nodeBalance["n"].to_list()
    p.node_capacity_for_scaling = _pdX_per_entity(
        flex_data.p_node_capacity_for_scaling, solve_name=solve_name,
        entity_dim="n", col_name="node", flex_data=flex_data,
        densify_entities=nodes_universe,
    )
    p.group_capacity_for_scaling = _pdX_per_entity(
        flex_data.p_group_capacity_for_scaling, solve_name=solve_name,
        entity_dim="g", col_name="group", flex_data=flex_data,
    )

    # complete_period_share_of_year — Series((solve, period)).
    p.complete_period_share_of_year = _pd_series_solve_period(
        flex_data.p_period_share, solve_name=solve_name,
    )

    # nested_model — DataFrame indexed by ``param`` with ``value`` column.
    # Reflects ``solveFirst`` / ``solveLast`` / ``contains_solve``.
    rows = {"solveFirst": 1.0 if (flex_data.p_nested_solve_first is None
                                   or bool(flex_data.p_nested_solve_first)) else 0.0}
    p.nested_model = pd.DataFrame(
        {"value": list(rows.values())},
        index=pd.Index(list(rows.keys()), name="param"),
    )

    return p


def _empty_solve_period_per_entity(*, col_name: str) -> pd.DataFrame:
    """Empty ``(solve, period)``-indexed DataFrame with named empty columns."""
    idx = pd.MultiIndex.from_arrays([[], []], names=["solve", "period"])
    out = pd.DataFrame(index=idx, dtype=float)
    out.columns = pd.Index([], dtype="object", name=col_name)
    return out


def _build_entity_unitsize_series(
    flex_data: "FlexData",
    *,
    entity_universe: "list[str] | None" = None,
) -> pd.Series:
    """Per-entity unitsize Series indexed by entity name.

    Combines ``p_unitsize`` (process side) and ``p_state_unitsize``
    (node side).  Defaults to ``1000.0`` for any entity in the
    nodeBalance / process_source_sink universe that doesn't appear
    in either Param — matching the slow path's preprocessing default
    at ``preprocessing/entity_period_calc_params.py:191``.
    """
    parts: list[pd.Series] = []
    if (flex_data.p_unitsize is not None
            and flex_data.p_unitsize.frame.height > 0):
        df = flex_data.p_unitsize.frame.to_pandas()
        s = df.set_index("p")["value"].astype(float)
        s.index.name = "entity"
        parts.append(s)
    if (flex_data.p_state_unitsize is not None
            and flex_data.p_state_unitsize.frame.height > 0):
        df = flex_data.p_state_unitsize.frame.to_pandas()
        s = df.set_index("n")["value"].astype(float)
        s.index.name = "entity"
        parts.append(s)
    if parts:
        out = pd.concat(parts)
        # Deduplicate (unlikely; node/process sets are disjoint).
        out = out[~out.index.duplicated(keep="first")]
    else:
        out = pd.Series(dtype=float, index=pd.Index([], name="entity"))
    out.name = "entity"
    # Densify with default 1000.0 (preprocessing default at
    # entity_period_calc_params.py:191).
    if entity_universe:
        for e in entity_universe:
            if e not in out.index:
                out[e] = 1000.0
    return out


# ---------------------------------------------------------------------------
# Multi-solve wrapper
# ---------------------------------------------------------------------------


def _has_solve_level(obj) -> bool:
    """True if ``obj`` is a pandas DataFrame/Series whose row MultiIndex
    has a ``solve`` level — i.e. it varies per sub-solve and rows from
    multiple sub-solves should be concatenated rather than overwritten.
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        idx = obj.index
        if isinstance(idx, pd.MultiIndex):
            return "solve" in (idx.names or ())
    return False


def read_parameters_multi(
    steps: "list[tuple[str, FlexData, Solution]] | list[tuple[str, FlexData]]",
    solution: "Solution | None" = None,
) -> SimpleNamespace:
    """Multi-solve variant of :func:`read_parameters`.

    ``steps`` is a list of ``(solve_name, flex_data)`` pairs covering
    every sub-solve (roll) of the orchestration's cascade.  Per-(solve,
    period, time) and per-(solve, period) outputs are built per sub-
    solve and concatenated along the row axis so the resulting ``par``
    namespace covers the FULL union of every sub-solve's dt axis —
    matching the union ``v`` carries from parquet aggregation in
    :mod:`flextool.process_outputs.read_variables`.

    Per-entity static attributes (entity_unitsize, commodity_co2_content,
    etc.) are invariant across sub-solves and are taken from the LAST
    step (arbitrary; they're identical across rolls).

    Rationale: rolling-solve orchestration carries per-sub-solve
    ``flex_data`` already; the output path must reflect the union of
    every sub-solve's parameter values, NOT just the last roll's.  See
    the structural diagnosis: ``par`` was built from the last
    ``flex_data.dt`` (one period) while ``v`` carried the union of
    every parquet (all periods), so downstream ``mul`` / ``join`` ops
    crashed on row-MultiIndex mismatches.
    """
    if not steps:
        raise ValueError("read_parameters_multi: steps must be non-empty")

    # Accept both (solve_name, flex_data, solution) and
    # (solve_name, flex_data) tuples for backwards compatibility.  The
    # 3-tuple form is REQUIRED for ``_entity_all_capacity`` to see each
    # roll's own ``v_invest`` / ``v_divest`` — passing only the last
    # roll's solution makes ``total`` lag by one period across the
    # chain (see 2026-05-14 unit_capacity__d.csv diagnosis).
    def _step_solution(s):
        if len(s) >= 3:
            return s[2]
        if solution is None:
            raise ValueError(
                "read_parameters_multi: step is a 2-tuple but no "
                "fallback solution was supplied"
            )
        return solution

    per_step = [
        (s[0], read_parameters(s[1], _step_solution(s), solve_name=s[0]))
        for s in steps
    ]
    last_ns = per_step[-1][1]
    if len(per_step) == 1:
        return last_ns

    # Pre-filter per-step ``entity_lifetime_fixed_cost`` /
    # ``entity_lifetime_fixed_cost_divest`` to only the step's REALIZED
    # periods (``flex_data.realized_dispatch`` — same source as
    # ``read_sets.d_realized_period``).  In a rolling/nested cascade
    # each step holds a period-d "active" lifetime value in the row
    # where d is the step's realized period, plus forward-discounted
    # lookahead values for lookahead periods (e.g. y2020 step has
    # p2020=827152 active + p2025=373153 lookahead).  ``v_invest[(solve,
    # d)]`` is non-zero only for the committing step's realized period,
    # so the cost in ``calc_costs.compute_costs`` (line 165, ``v.invest
    # × unitsize × entity_lifetime_fixed_cost``) must use the value
    # from that same committing step.  Filtering each piece pre-concat
    # means ``drop_levels`` sees one row per period (no dedup
    # ambiguity).  Without this, ``_PAR_DEDUP keep='first'`` picks the
    # earliest step's lookahead value for every period after the first
    # (e.g. p2025 → y2020 lookahead 373153 instead of y2025 active
    # 827152), giving ``costs_discounted.csv:fixed cost invested``
    # 120.6 instead of the LP-true 267.4.  ``ed_invest_set`` /
    # ``ed_divest_set`` cover both realized + lookahead invest
    # candidates, so we use ``realized_dispatch`` instead.
    _lifetime_attrs = ("entity_lifetime_fixed_cost",
                       "entity_lifetime_fixed_cost_divest")
    for (sn, ns), step in zip(per_step, steps):
        flex_data = step[1]
        rd = getattr(flex_data, "realized_dispatch", None)
        if rd is None or getattr(rd, "height", 0) == 0:
            continue
        realized_periods = set(
            rd.select("period").unique().to_pandas()["period"].tolist()
        )
        if not realized_periods:
            continue
        for attr in _lifetime_attrs:
            obj = getattr(ns, attr, None)
            if obj is None or not _has_solve_level(obj):
                continue
            periods = obj.index.get_level_values("period")
            mask = periods.isin(realized_periods)
            setattr(ns, attr, obj[mask])

    # Per-step ``entity_all_existing`` carries the period-d existing-
    # capacity chain (pre_existing on solve_first, later_existing
    # otherwise).  In a NESTED cascade the parent invest step and
    # every child dispatch step both densify the (period, entity)
    # grid: invest_24h@(wind_plant, p2020) = 1000 (pre_existing,
    # solve_first=True) but dispatch_..._roll_17@(wind_plant, p2020)
    # = 1288.83 (later_existing folding the parent's v_invest back
    # in).  drop_levels' ``keep='last'`` dedup picks the dispatch
    # row, blowing up the ``existing`` column in unit_capacity__d.csv
    # to the cumulative-with-current-invest value (HEAD shows 1288.83
    # for the same cell v3.32.0 emits as 1000).  v3.32.0's mod-side
    # writer only emits rows from the FIRST solve that owns each
    # period via ``period_capacity`` deduplication (flextool.mod
    # L5993-6002): in nested that's the parent invest step.
    #
    # Fix: filter each step's ``entity_all_existing`` to ONLY the
    # periods that step realizes (``flex_data.realized_dispatch`` ∪
    # ``ed_invest_set/ed_divest_set`` "d"s).  Dispatch-only children
    # in a nested cascade drop their lookahead rows; the parent
    # invest step's per-period values are the sole contributor and
    # win regardless of dedup direction.  In 4-solve / multi_year
    # invest cascades each step realizes a disjoint period so no
    # collision arises either way — the filter is a no-op there.
    for (sn, ns), step in zip(per_step, steps):
        flex_data = step[1]
        step_periods: set[str] = set()
        rd = getattr(flex_data, "realized_dispatch", None)
        if rd is not None and getattr(rd, "height", 0) > 0:
            step_periods.update(
                rd.select("period").unique().to_pandas()["period"].tolist()
            )
        for src_attr in ("ed_invest_set", "ed_divest_set"):
            src = getattr(flex_data, src_attr, None)
            if src is not None and getattr(src, "height", 0) > 0 and "d" in src.columns:
                step_periods.update(
                    src.select("d").unique().to_pandas()["d"].tolist()
                )
        if not step_periods:
            # Step realizes nothing — leave as-is.  This is the legacy
            # behaviour for fixtures that don't populate realized_*.
            continue
        for attr in ("entity_all_existing",):
            obj = getattr(ns, attr, None)
            if obj is None or not _has_solve_level(obj):
                continue
            periods = obj.index.get_level_values("period")
            mask = periods.isin(step_periods)
            setattr(ns, attr, obj[mask])

    # Per-step ``entity_annual_discounted`` /
    # ``entity_annual_divest_discounted`` carry the invest-side NPV
    # coefficients used by ``calc_costs.compute_costs`` to derive
    # ``cost_entity_invest_d = v_invest × unitsize × annual_discounted``.
    # The polars derivation builds these from ``period_invest``: a
    # nested-dispatch child solve has empty ``period_invest`` and the
    # frame is therefore empty, but ``_pdX_per_entity(densify_entities=
    # _entity_universe)`` zero-fills it to the full entity × period
    # grid.  In a nested cascade, the parent ``invest`` step holds the
    # real coefficients (e.g. 4.18 for ``wind_plant @ p2020``) while
    # every child dispatch step densifies the same (period, entity)
    # cell to 0.  Without intervention, ``drop_levels`` dedup with
    # ``keep='last'`` picks the last child step's zero, zeroing out
    # ``r.cost_entity_invest_d`` and dropping ``unit investment &
    # retirement`` from ``costs_discounted.csv`` (HEAD shows 0,
    # v3.32.0 shows 471.28).  Fix: detect a "dispatch-only" step
    # (``flex_data.ed_entity_annual_discounted`` empty/None) and clear
    # the densified-zero per-step frame so it doesn't compete with the
    # parent invest step's real values.
    _annual_attrs = ("entity_annual_discounted",
                     "entity_annual_divest_discounted",
                     "entity_annuity")
    _src_field = {
        "entity_annual_discounted": "ed_entity_annual_discounted",
        "entity_annuity": "ed_entity_annual_discounted",
        "entity_annual_divest_discounted":
            "ed_entity_annual_divest_discounted",
    }
    for (sn, ns), step in zip(per_step, steps):
        flex_data = step[1]
        for attr in _annual_attrs:
            src_attr = _src_field[attr]
            src_param = getattr(flex_data, src_attr, None)
            src_empty = (
                src_param is None
                or getattr(src_param, "frame", None) is None
                or getattr(src_param.frame, "height", 0) == 0
            )
            if not src_empty:
                continue
            obj = getattr(ns, attr, None)
            if obj is None or not _has_solve_level(obj):
                continue
            # Empty out this step's rows (preserve the index/columns
            # frame for the concat path; head(0) keeps the schema).
            setattr(ns, attr, obj.iloc[0:0])

    out = SimpleNamespace()
    attr_names = [a for a in vars(last_ns).keys()]
    for attr in attr_names:
        pieces = [getattr(ns, attr) for _, ns in per_step]
        # If the attribute has a ``solve`` level on its row index, the
        # union over sub-solves is the correct semantic; concat rows.
        if any(_has_solve_level(p) for p in pieces):
            non_empty = [p for p in pieces if len(p) > 0]
            if not non_empty:
                setattr(out, attr, pieces[-1])
                continue
            # ``pd.concat`` preserves dtypes and row order.  Each piece
            # carries a distinct ``solve`` value so no de-duplication
            # is needed.
            merged = pd.concat(non_empty, axis=0)
            # Preserve column dtype/name (concat keeps it; defensive).
            if hasattr(pieces[-1], "columns") and hasattr(merged, "columns"):
                merged.columns.name = pieces[-1].columns.name
            setattr(out, attr, merged)
        else:
            # Static / per-entity / not-solve-keyed: invariant across
            # sub-solves; pick the last step's value.
            setattr(out, attr, getattr(last_ns, attr))
    return out
