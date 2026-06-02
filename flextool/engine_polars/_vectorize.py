"""Shared vectorize-per-roll helpers.

The heavy preprocessing families (``pdtProcess``, ``pdtNode``,
``pdtProcess_{source,sink}``, ``pdtCommodity``, ``pdtGroup``, …) each
derive a dense ``(domain × dt)`` frame by looping over every
``(entity, period, time)`` cell and calling a scalar cascade
(``PdtLookup.get`` and friends).  That inner loop is recomputed on every
roll and is the per-roll preprocessing hotspot.

This module provides the reusable polars pieces that let a family
replace ONLY that inner compute with vectorized left-joins +
``coalesce`` (cascade-priority order) + group-by-sum (the folds), still
**per roll, over the roll's own window** — no cache, no full-domain
frame, no slice.  See ``specs/vectorize_per_roll.md`` for the full
design + the adversarial-critique corrections folded into the fold here.

Engine / rendering decoupling (design §3)
-----------------------------------------
The join/coalesce/group-by graph collects to a **Float64** value column
under either engine (eager ``df.collect()`` default, or
``df.collect(engine="streaming")`` as a later per-family speed upgrade).
Rendering to the legacy ``repr(float)`` string form is ALWAYS the
post-collect Python loop :func:`_render_value_column` — engine
independent, so streaming is a one-line flag flip with no rendering
parity risk.  **Never** render with ``.cast(Utf8)`` (it diverges from
``repr`` on sci-notation exponent padding and ``NaN``/``nan``).
"""
from __future__ import annotations

import polars as pl

__all__ = [
    "build_entity_dt_grid",
    "build_entity_period_grid",
    "lift_dict_to_lookup",
    "build_fold_frame",
    "coalesce_value",
    "collect_value_frame",
    "_render_value_column",
]


# ---------------------------------------------------------------------------
# Rendering — repr loop, engine-independent (NEVER .cast(Utf8)).
# ---------------------------------------------------------------------------

def _render_value_column(s: pl.Series) -> pl.Series:
    """Render a Float64 value series to the legacy ``repr(v)`` strings.

    Mirrors the legacy emitters' ``f",{repr(v)}\\n"`` exactly: a bare
    ``repr(x)`` per cell (NOT ``repr(float(x))`` — Tier-B int-vs-float
    cells must survive where a family emits them).  This is the ONLY
    supported render path; ``.cast(Utf8)`` diverges from ``repr`` on
    sci-notation exponent padding and ``NaN``/``nan``.
    """
    return pl.Series("value", [repr(x) for x in s], dtype=pl.Utf8)


# ---------------------------------------------------------------------------
# Base grid — entity-major ``domain × dt`` with integer order keys.
# ---------------------------------------------------------------------------

def build_entity_dt_grid(
    domain: list[tuple],
    dt: list[tuple[str, str]],
    *,
    key_cols: list[str],
) -> pl.DataFrame:
    """Build the entity-major ``domain × dt`` grid.

    *domain* is the legacy ``_read_pairs`` / ``_read_triples`` **list**
    (order AND duplicates preserved — never ``.unique()`` it: S-M2).  Its
    tuples carry the entity-key columns named by *key_cols* (e.g.
    ``["process", "param"]`` or ``["process", "side", "param"]``).  *dt*
    is the ``(period, time)`` list from ``steps_in_use``.

    The result carries two integer order keys:

    * ``__eo`` — the domain row position (entity-major ordering key),
    * ``__to`` — the ``dt`` row position,

    so the final ``.sort([__eo, __to])`` reproduces the legacy nested
    ``for entity: for (d, t):`` emission order even under streaming
    (which may reorder).  An empty domain or dt yields a zero-row grid
    with the correct schema (all key cols Utf8 + the two order keys).
    """
    n_keys = len(key_cols)
    base_data: dict[str, list] = {
        key_cols[i]: [row[i] for row in domain] for i in range(n_keys)
    }
    base_data["__eo"] = list(range(len(domain)))
    base = pl.DataFrame(
        base_data,
        schema={
            **{c: pl.Utf8 for c in key_cols},
            "__eo": pl.Int64,
        },
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("__one"))

    dt_df = pl.DataFrame(
        {
            "period": [d for (d, _t) in dt],
            "time": [t for (_d, t) in dt],
            "__to": list(range(len(dt))),
        },
        schema={"period": pl.Utf8, "time": pl.Utf8, "__to": pl.Int64},
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("__one"))

    return base.join(dt_df, on="__one", how="inner").drop("__one")


# ---------------------------------------------------------------------------
# Base grid — entity-major ``domain × period`` (period-only, no time).
# ---------------------------------------------------------------------------

def build_entity_period_grid(
    domain: list[tuple],
    periods: list[str],
    *,
    key_cols: list[str],
) -> pl.DataFrame:
    """Build the entity-major ``domain × period`` grid (no time axis).

    Mirror of :func:`build_entity_dt_grid` but with a single ``period``
    axis instead of ``(period, time)`` — for the period-only families
    (e.g. ``pdGroup``).  *domain* is the legacy entity list (order AND
    duplicates preserved — never ``.unique()`` it: S-M2), its tuples
    carrying the entity-key columns named by *key_cols*.  *periods* is the
    ``period_in_use`` list (order AND duplicates preserved — never
    ``.unique()`` it either).

    The result carries two integer order keys:

    * ``__eo`` — the domain row position (entity-major ordering key),
    * ``__po`` — the ``periods`` row position,

    so the final ``.sort([__eo, __po])`` reproduces the legacy nested
    ``for entity: for d:`` emission order even under streaming (which may
    reorder).  An empty domain or periods list yields a zero-row grid with
    the correct schema (all key cols Utf8 + period Utf8 + the two order
    keys).
    """
    n_keys = len(key_cols)
    base_data: dict[str, list] = {
        key_cols[i]: [row[i] for row in domain] for i in range(n_keys)
    }
    base_data["__eo"] = list(range(len(domain)))
    base = pl.DataFrame(
        base_data,
        schema={
            **{c: pl.Utf8 for c in key_cols},
            "__eo": pl.Int64,
        },
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("__one"))

    period_df = pl.DataFrame(
        {
            "period": list(periods),
            "__po": list(range(len(periods))),
        },
        schema={"period": pl.Utf8, "__po": pl.Int64},
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("__one"))

    return base.join(period_df, on="__one", how="inner").drop("__one")


# ---------------------------------------------------------------------------
# Lookup frames — lift a cascade dict to a Float64-valued lookup frame.
# ---------------------------------------------------------------------------

def lift_dict_to_lookup(
    d: dict,
    key_cols: list[str],
    value_col: str,
) -> pl.DataFrame:
    """Lift a cascade ``dict[tuple] -> float`` to a polars lookup frame.

    Lift from the already-built (last-wins-deduped) dict, NOT the raw
    CSV — lifting from CSV would re-introduce duplicate join keys and
    explode the left-join (S-claim4).  *key_cols* names the tuple
    positions (all Utf8); *value_col* names the Float64 value column.

    An empty dict yields an explicit empty-schema frame so the
    downstream join-key dtypes still line up (Utf8 keys + Float64 value).
    """
    if not d:
        return pl.DataFrame(
            {c: [] for c in [*key_cols, value_col]},
            schema={
                **{c: pl.Utf8 for c in key_cols},
                value_col: pl.Float64,
            },
        )
    keys = list(d.keys())
    n = len(key_cols)
    if n == 1:
        data = {key_cols[0]: [k if not isinstance(k, tuple) else k[0]
                              for k in keys]}
    else:
        data = {key_cols[i]: [k[i] for k in keys] for i in range(n)}
    data[value_col] = list(d.values())
    return pl.DataFrame(
        data,
        schema={
            **{c: pl.Utf8 for c in key_cols},
            value_col: pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# The fold — stochastic (branch 1) + parent-period (branch 2).
# Critique-corrected: multi-parent multiplicity + stoch fall-through.
# ---------------------------------------------------------------------------

def build_fold_frame(
    *,
    pbt: dict,
    pbt_key_cols: list[str],
    out_key_cols: list[str],
    ts_for_d: dict[str, list[str]],
    tb_for_d: dict[str, list[str]],
    pe_for_d: dict[str, list[str]],
    stoch_entities: set[str],
    stoch_filter_cols: list[str],
    periods: list[str],
) -> pl.DataFrame | None:
    """Vectorize the stochastic + parent-period fold (branches 1 & 2).

    Reproduces ``PdtLookup.get`` branches 1-2 exactly (the scalar
    ``_pdt_lookup.py:407-463`` cascade):

    * **Branch 1 (stochastic):** for an entity whose ``stoch_filter_cols``
      key is in *stoch_entities*, sum ``pbt[(e, …, tb, ts, t)]`` over
      ``tb ∈ tb_for_d[d] × ts ∈ ts_for_d[d]``.
    * **Branch 2 (parent-period):** sum ``pbt`` over
      ``pe ∈ pe_for_d[d], tb ∈ tb_for_d[pe], ts ∈ ts_for_d[d]``.

    Critique corrections vs the template (design §4):

    * **Multi-parent multiplicity (S2):** the parent expansion KEEPS
      ``pe`` in its key ``(period, pe, tb, ts)`` and is NOT
      ``.unique()``-d, so a pbt value shared by two parents is added
      once per parent — then the join output is group-by-summed on the
      OUTPUT key.
    * **Stoch fall-through (S2/§12.6):** the parent fold is NOT filtered
      to non-stoch entities.  ``v_stoch`` and ``v_parent`` are computed
      as separate frames and coalesced ``coalesce(v_stoch, v_parent)``
      (stoch-first preserves branch priority), so a stoch entity that
      misses branch 1 can still hit branch 2.
    * **Duplicate-row invariant (S3):** the expansion is NOT
      ``.unique()``-d — production (``_read_pairs_to_dict``) does not
      dedup, so it sums once per duplicate ``(tb)``/``(ts)``/``(pe)``
      occurrence; matching that requires preserving duplicates here.

    *pbt_key_cols* names the pbt entity-key columns (``["process",
    "param"]`` for pdtProcess/pdtNode, ``["process", "side", "param"]``
    for PerSide).  *out_key_cols* names the OUTPUT group key (entity key
    cols + ``["period", "time"]``).  *stoch_filter_cols* names the
    column(s) the stoch membership test keys on — note the PerSide M1
    quirk: it filters on the ``process`` column ALONE, not the 3-col key.

    Returns a ``(*out_key_cols, "v_fold")`` frame, or ``None`` when no
    fold rows exist (no pbt / no expansion / empty join).
    """
    if not pbt:
        return None

    # --- pbt frame: entity-key cols + (tb, ts, time) + v_pbt ---------------
    keys = list(pbt.keys())
    n_ent = len(pbt_key_cols)
    pbt_data: dict[str, list] = {
        pbt_key_cols[i]: [k[i] for k in keys] for i in range(n_ent)
    }
    pbt_data["tb"] = [k[n_ent] for k in keys]
    pbt_data["ts"] = [k[n_ent + 1] for k in keys]
    pbt_data["time"] = [k[n_ent + 2] for k in keys]
    pbt_data["v_pbt"] = list(pbt.values())
    pbt_df = pl.DataFrame(
        pbt_data,
        schema={
            **{c: pl.Utf8 for c in pbt_key_cols},
            "tb": pl.Utf8,
            "ts": pl.Utf8,
            "time": pl.Utf8,
            "v_pbt": pl.Float64,
        },
    )

    # --- expansion rows (preserve duplicates — S3) -------------------------
    # Stoch: (period, tb, ts) for ts in ts_for_d[d], tb in tb_for_d[d].
    # Parent: (period, pe, tb, ts) for ts in ts_for_d[d], pe in pe_for_d[d],
    #         tb in tb_for_d[pe].  Keep pe (S2 multiplicity).
    stoch_period: list[str] = []
    stoch_tb: list[str] = []
    stoch_ts: list[str] = []
    par_period: list[str] = []
    par_pe: list[str] = []
    par_tb: list[str] = []
    par_ts: list[str] = []
    for d in periods:
        ts_list = ts_for_d.get(d, ())
        tb_list = tb_for_d.get(d, ())
        pe_list = pe_for_d.get(d, ())
        for ts in ts_list:
            for tb in tb_list:
                stoch_period.append(d)
                stoch_tb.append(tb)
                stoch_ts.append(ts)
            for pe in pe_list:
                for tb in tb_for_d.get(pe, ()):
                    par_period.append(d)
                    par_pe.append(pe)
                    par_tb.append(tb)
                    par_ts.append(ts)

    stoch_list = list(stoch_entities)

    # --- branch 1: stochastic fold ----------------------------------------
    v_stoch = None
    if stoch_period and stoch_list:
        stoch_exp = pl.DataFrame(
            {"period": stoch_period, "tb": stoch_tb, "ts": stoch_ts},
            schema={"period": pl.Utf8, "tb": pl.Utf8, "ts": pl.Utf8},
        )
        j = pbt_df.join(stoch_exp, on=["tb", "ts"], how="inner")
        # Stoch membership keys on stoch_filter_cols (process alone for
        # PerSide; the full entity key for pdtProcess/pdtNode).
        if len(stoch_filter_cols) == 1:
            mask = pl.col(stoch_filter_cols[0]).is_in(stoch_list)
        else:
            # multi-col membership — stoch_entities holds tuples
            stoch_struct = pl.struct(stoch_filter_cols)
            mask = stoch_struct.is_in(stoch_list)
        j = j.filter(mask)
        if j.height > 0:
            v_stoch = (
                j.group_by(out_key_cols)
                 .agg(pl.col("v_pbt").sum().alias("v_stoch"))
            )

    # --- branch 2: parent-period fold (NO stoch filter — fall-through) -----
    v_parent = None
    if par_period:
        par_exp = pl.DataFrame(
            {
                "period": par_period,
                "pe": par_pe,
                "tb": par_tb,
                "ts": par_ts,
            },
            schema={
                "period": pl.Utf8,
                "pe": pl.Utf8,
                "tb": pl.Utf8,
                "ts": pl.Utf8,
            },
        )
        j = pbt_df.join(par_exp, on=["tb", "ts"], how="inner")
        if j.height > 0:
            v_parent = (
                j.group_by(out_key_cols)
                 .agg(pl.col("v_pbt").sum().alias("v_parent"))
            )

    # --- coalesce stoch-first ---------------------------------------------
    if v_stoch is not None and v_parent is not None:
        fold = (
            v_stoch.join(
                v_parent, on=out_key_cols, how="full", coalesce=True,
            )
            .with_columns(
                pl.coalesce(pl.col("v_stoch"), pl.col("v_parent"))
                .alias("v_fold")
            )
            .select([*out_key_cols, "v_fold"])
        )
    elif v_stoch is not None:
        fold = v_stoch.rename({"v_stoch": "v_fold"})
    elif v_parent is not None:
        fold = v_parent.rename({"v_parent": "v_fold"})
    else:
        return None
    return fold


# ---------------------------------------------------------------------------
# Coalesce assembler + collect.
# ---------------------------------------------------------------------------

def coalesce_value(exprs: list[pl.Expr], alias: str = "value_f") -> pl.Expr:
    """``pl.coalesce`` over *exprs* in cascade-priority order."""
    return pl.coalesce(exprs).alias(alias)


def collect_value_frame(
    lf: "pl.LazyFrame | pl.DataFrame",
    *,
    key_cols: list[str],
    value_f_col: str = "value_f",
    sort_cols: list[str] | None = None,
    engine: str = "eager",
) -> pl.DataFrame:
    """Collect the join/coalesce graph and render the value column.

    Collects under *engine* (``"eager"`` → ``lf.collect()``; anything
    else → ``lf.collect(engine="streaming")``), sorts by *sort_cols*
    (default ``["__eo", "__to"]`` for entity-major order), renders the
    Float64 *value_f_col* via :func:`_render_value_column` AFTER collect
    (engine-independent), and selects ``[*key_cols, "value"]``.
    """
    if sort_cols is None:
        sort_cols = ["__eo", "__to"]
    if isinstance(lf, pl.LazyFrame):
        df = lf.collect() if engine == "eager" else lf.collect(
            engine="streaming")
    else:
        df = lf
    df = df.sort(sort_cols)
    value = _render_value_column(df[value_f_col])
    return df.select(key_cols).with_columns(value)
