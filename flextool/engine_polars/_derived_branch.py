"""Cluster D — multi-branch / stochastic propagation (Δ.8).

Lazy-polars port of flextool's stochastic-branch derived helpers.
Cluster D is the fourth of six derived-helper port phases per
``audit/native_data_path_design_derived_clusters.md``.

Cluster D fields (per
``audit/native_data_path_design_derived_clusters.md``):

* ``pd_branch_weight`` — per-period branch probability weight.  In
  deterministic / non-stochastic fixtures, defaults to 1.0 per
  realised period.  In multi-branch fixtures: normalises
  ``solve_branch_weight[d]`` against the sum across sibling branches
  that share the same ``(d2, b)`` parent and the same first-step.
* ``pdt_branch_weight`` — per-(d, t) variant.  Same normalisation but
  iterating over ``dt`` rows instead of period × first-step pairs.
  Output is dense over ``dt`` (mirroring the .mod's ``param
  pdt_branch_weight {(d, t) in dt}`` declaration).
* ``dt_non_anticipativity`` — (d, t) where the four
  ``non_anticipativity_*`` constraints fire.  Built as
  ``realized_dispatch ∪ fix_storage_timesteps``.  Empty when
  stochastics are inactive.
* ``period_branch_full`` — full ``period__branch.csv`` (anchor →
  sibling).  Used by the model layer's storage / online / reserve
  non-anticipativity coupling.
* ``period_in_use_set`` — periods active in the active solve.
  ``realized_periods ∪ stochastic_branches ∪ invest_periods ∪
  fix_storage_periods``.
* ``dd_same_scenario`` — (d, d_other) scenario-lineage pairs: some
  scenario leaf-path contains both periods (recourse plan §6b
  Slice A; ``specs/sliceA_lineage_frames_design.md`` §2.2).
  All-pairs over ``period_in_use`` for deterministic solves.
  Consumed by nothing until Slice B.
* ``pd_non_anticipativity`` — (d, b) period pairs whose INVEST
  decisions must be tied when the ``non_anticipativity_invest_p/n``
  families land (Slice E).  Provably empty for every currently
  constructible solve (design §2.3).

All helpers are lazy ``pl.LazyFrame`` chains; the public
``apply_branch_cluster`` entry collects once per emitted Param.

Algorithm reference: flextool's
``preprocessing/period_calculated_params.py:write_branch_weights:364-451``
and ``preprocessing/per_solve_sets.py:96-101, 267-276``.

R-O6 invariant (per ``audit/a6_b_dim_alternative.md``): branches stay
realised-only for invest.  This module does NOT introduce per-branch
``v_invest`` variables; it only emits the *operational* probability
weights and the non-anticipativity gate that pin storage / online /
reserve dispatch across siblings.  The Slice A scenario-lineage
frames (``dd_same_scenario`` / ``pd_non_anticipativity``) are
descriptive bookkeeping only — the invariant still holds; branches do
not enter ``invest_periods`` in this slice.

Workdir CSV reads (deferred to in-memory ``SolveContext`` per Δ.9+):

* ``solve_data/period__branch.csv`` — anchor → sibling pairs.
* ``solve_data/solve_branch_weight.csv`` — branch → input weight.
* ``solve_data/first_timesteps.csv`` — period → first step.
* ``solve_data/period_in_use_set.csv`` — output domain (active set).
* ``solve_data/realized_dispatch.csv`` — realised dispatch (d, t).
* ``solve_data/fix_storage_timesteps.csv`` — fix-storage (d, t).
* ``input/groupIncludeStochastics.csv`` — stochastic-coupling groups.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from ._axis_enums import (
    alias_to_axis,
    cast_dim,
    rename_to_axis,
    schema_dtype,
)
from ._emit_provider_io import _provider_key


def _provider_get(provider, path: "Path") -> "pl.DataFrame | None":
    """Provider-only fetch.  Returns ``None`` when the Provider is
    missing or doesn't carry *path*'s canonical key.
    """
    if provider is None:
        return None
    key = _provider_key(path)
    if not provider.has(key):
        return None
    return provider.get(key)

# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# Workdir CSV readers — lazy frames
# ---------------------------------------------------------------------------


def _empty_lf(schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Return an empty :class:`pl.LazyFrame` with the given schema."""
    return pl.DataFrame(schema=schema).lazy()


def _maybe_provider_lf(provider, path: Path,
                       rename: dict[str, str] | None = None,
                       ) -> pl.LazyFrame | None:
    """Return a lazy frame from the Provider with optional rename.

    Returns ``None`` when the Provider is missing or doesn't carry the
    canonical key for *path*.
    """
    df = _provider_get(provider, path)
    if df is None or df.height == 0:
        return None
    lf = df.lazy()
    if rename:
        # Only rename columns that actually exist (defensive against
        # column-name drift across fixture vintages).
        cols = df.columns
        applied = {k: v for k, v in rename.items() if k in cols}
        if applied:
            lf = lf.pipe(rename_to_axis, applied)
    return lf


def period_branch_pairs_lf(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """Read ``solve_data/period__branch.csv`` as a lazy ``(d, b)`` frame.

    Returns an empty frame (with the correct schema) when the file is
    absent / empty — non-stochastic fixtures' "deterministic" path.

    The ``b`` column is the *sibling branch* name (in non-stochastic
    fixtures, ``d == b`` for every row).

    Path B Cat B (WriterSnapshot top-7): when ``ctx`` is supplied, the
    cached ``ctx.period_branch`` (canonical ``[d_anchor, b]``) is
    rewritten to ``[d, b]`` lazily — avoids re-reading the CSV on every
    cascade pass.
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "b": schema_dtype(_enums, "b")}
    if ctx is not None:
        pb_df = getattr(ctx, "period_branch", None)
        if pb_df is not None and pb_df.height > 0:
            return (pb_df.lazy()
                          .select(alias_to_axis("d_anchor", "d"),
                                  cast_dim(pl.col("b"), _enums, "b"))
                          .unique())
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    lf = _maybe_provider_lf(provider, p, rename={"period": "d", "branch": "b"})
    if lf is None:
        return _empty_lf(schema)
    return lf.select(
        cast_dim(pl.col("d"), _enums, "d"),
        cast_dim(pl.col("b"), _enums, "b"),
    ).unique()


def solve_branch_weights_lf(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """Read ``solve_data/solve_branch_weight.csv`` as ``(b, w)`` lazy.

    The flextool CSV header is either ``branch,p_branch_weight_input``
    (stochastic fixtures) or ``branch,value`` (deterministic).  We
    normalise to ``(b, w)``.

    Defaults to the empty frame (schema only) when the file is absent.

    Path B Cat B (WriterSnapshot top-7): when ``ctx`` is supplied, the
    cached ``ctx.solve_branch_weight`` (canonical
    ``[b, p_branch_weight_input]``) is used directly.
    """
    schema = {"b": schema_dtype(_enums, "b"), "w": pl.Float64}
    if ctx is not None:
        sbw = getattr(ctx, "solve_branch_weight", None)
        if sbw is not None and sbw.height > 0:
            return (sbw.lazy()
                          .select(cast_dim(pl.col("b"), _enums, "b"),
                                  pl.col("p_branch_weight_input").alias("w")))
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "solve_branch_weight.csv"
    df = _provider_get(provider, p)
    if df is None or df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    b_col = "branch" if "branch" in cols else cols[0]
    if "p_branch_weight_input" in cols:
        v_col = "p_branch_weight_input"
    elif "value" in cols:
        v_col = "value"
    elif len(cols) >= 2:
        v_col = cols[1]
    else:
        return _empty_lf(schema)
    return (df.lazy()
              .select(alias_to_axis(b_col, "b"),
                      pl.col(v_col).cast(pl.Float64, strict=False).alias("w")))


def first_timesteps_lf(workdir: Path | None,
                       *, provider: "object | None" = None) -> pl.LazyFrame:
    """Read ``solve_data/first_timesteps.csv`` as ``(d, ts)`` lazy.

    Maps each period (or branch period) to its first timestep — the
    discriminator the ``pd_branch_weight`` algorithm uses for grouping
    sibling branches.
    """
    schema = {"d": schema_dtype(_enums, "d"), "ts": pl.Utf8}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "first_timesteps.csv"
    df = _provider_get(provider, p)
    if df is None or df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    s_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(alias_to_axis(d_col, "d"),
                      pl.col(s_col).cast(pl.Utf8, strict=False).alias("ts")))


def period_in_use_set_lf(workdir: Path | None,
                          source: "InputSource | None" = None,
                          active_solve: str | None = None,
                          *,
                          ctx: "object | None" = None,
                          provider: "object | None" = None,
                          ) -> pl.LazyFrame:
    """Read ``solve_data/period_in_use_set.csv`` as ``(d,)`` lazy.

    When the workdir CSV is absent (single-solve / no chain runner),
    fall back to ``realized_periods ∪ invest_periods`` from the
    ``solve`` parameters.  This is sufficient for non-stochastic
    fixtures.

    Path B Cat B (WriterSnapshot top-7): when ``ctx`` is supplied, the
    cached ``ctx.period_in_use`` (canonical ``[d]``) is used directly.
    """
    schema = {"d": schema_dtype(_enums, "d")}
    if ctx is not None:
        piu_df = getattr(ctx, "period_in_use", None)
        if piu_df is not None and piu_df.height > 0:
            return piu_df.lazy().select("d").unique()
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        df = _provider_get(provider, p)
        if df is not None and df.height > 0 and df.columns:
            col = df.columns[0]
            return (df.lazy()
                      .select(alias_to_axis(col, "d"))
                      .unique())
    # Fall back to source-derived realized + invest.
    if source is None or active_solve is None:
        return _empty_lf(schema)
    parts: list[pl.LazyFrame] = []
    for ec, par in (("solve", "realized_periods"),
                     ("solve", "invest_periods")):
        try:
            df = source.parameter(ec, par)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy()
                       .filter(pl.col("name") == active_solve)
                       .select(alias_to_axis("value", "d")))
    if not parts:
        return _empty_lf(schema)
    return pl.concat(parts).unique()


def realized_dispatch_lf(workdir: Path | None,
                          *, provider: "object | None" = None) -> pl.LazyFrame:
    """Read ``solve_data/realized_dispatch.csv`` as ``(d, t)`` lazy."""
    schema = {"d": schema_dtype(_enums, "d"),
              "t": schema_dtype(_enums, "t")}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "realized_dispatch.csv"
    df = _provider_get(provider, p)
    if df is None or df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    t_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(alias_to_axis(d_col, "d"),
                      alias_to_axis(t_col, "t"))
              .unique())


def fix_storage_timesteps_lf(workdir: Path | None,
                              *, provider: "object | None" = None) -> pl.LazyFrame:
    """Read ``solve_data/fix_storage_timesteps.csv`` as ``(d, t)`` lazy."""
    schema = {"d": schema_dtype(_enums, "d"),
              "t": schema_dtype(_enums, "t")}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "fix_storage_timesteps.csv"
    df = _provider_get(provider, p)
    if df is None or df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    t_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(alias_to_axis(d_col, "d"),
                      alias_to_axis(t_col, "t"))
              .unique())


def steps_in_use_lf(workdir: Path | None,
                     *, provider: "object | None" = None) -> pl.LazyFrame:
    """Read ``solve_data/steps_in_use.csv`` as ``(d, t)`` lazy."""
    schema = {"d": schema_dtype(_enums, "d"),
              "t": schema_dtype(_enums, "t")}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "steps_in_use.csv"
    df = _provider_get(provider, p)
    if df is None or df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    t_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(alias_to_axis(d_col, "d"),
                      alias_to_axis(t_col, "t"))
              .unique())


# ---------------------------------------------------------------------------
# pd_branch_weight / pdt_branch_weight — lazy normalisation
# ---------------------------------------------------------------------------


def pd_branch_weight_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """Per-period branch weight (lazy) — full multi-branch cascade.

    Mirrors flextool's
    ``preprocessing/period_calculated_params.py:write_branch_weights:364-451``
    pd loop::

        pd_branch_weight[d] = w[d] / sum w[b] over branches b such that
            (d2, b) ∈ period__branch
            AND (b, ts) ∈ period__time_first
            AND (d, ts) ∈ period__time_first    (same first-step as d)
            AND (d2, d) ∈ period__branch        (same parent)

    The (d2, b) pair iteration deliberately counts a branch ``b`` once
    per parent ``d2`` it shares with ``d`` (so multi-parent siblings
    contribute to the denominator multiple times — matching the .mod's
    ``sum {d2, b in period__branch}`` semantics).

    Defaults to ``1.0`` per realised period when ``period__branch`` is
    silent (deterministic fixtures).

    Returns a lazy ``(d, value)`` frame.  Empty frame when no
    ``period_in_use`` rows exist.
    """
    pb = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    piu = period_in_use_set_lf(workdir, source, active_solve, ctx=ctx,
                               provider=provider).collect()
    if piu.height == 0:
        return _empty_lf({"d": pl.Utf8, "value": pl.Float64})
    if pb.height == 0:
        # Deterministic fallback: 1.0 per realised period.
        return (piu.lazy()
                  .select(pl.col("d"))
                  .with_columns(value=pl.lit(1.0))
                  .sort("d"))
    bw = solve_branch_weights_lf(workdir, ctx=ctx, provider=provider).collect()
    ft = first_timesteps_lf(workdir, provider=provider).collect()
    # weights mapping with fallback 1.0 (mirrors flextool's
    # ``branch_weight.get(b, 1.0)``).
    w_lookup = {row["b"]: float(row["w"]) for row in bw.iter_rows(named=True)
                if row["w"] is not None}

    # Build the parent set: pb_set = frozenset((d2, b)).
    pb_set = {(row["d"], row["b"]) for row in pb.iter_rows(named=True)}
    # times_with_first[ts] = {b : (b, ts) ∈ first_ts}
    first_ts_map = {row["d"]: row["ts"] for row in ft.iter_rows(named=True)}
    times_with_first: dict[str, set[str]] = {}
    for d, ts in first_ts_map.items():
        times_with_first.setdefault(ts, set()).add(d)

    def w(b: str) -> float:
        return w_lookup.get(b, 1.0)

    rows: list[tuple[str, float]] = []
    period_list = piu["d"].to_list()
    for d in period_list:
        ts = first_ts_map.get(d)
        if ts is None:
            continue
        branches_at_ts = times_with_first.get(ts, set())
        denom = 0.0
        # Iterate (d2, b) pairs (mirrors mod's ``sum {d2, b in pb}``)
        for row in pb.iter_rows(named=True):
            d2, b = row["d"], row["b"]
            if b not in branches_at_ts:
                continue
            if (d2, d) not in pb_set:
                continue
            denom += w(b)
        if denom == 0.0:
            continue
        rows.append((d, w(d) / denom))
    if not rows:
        return _empty_lf({"d": pl.Utf8, "value": pl.Float64})
    out = pl.DataFrame(rows, schema={"d": schema_dtype(_enums, "d"),
                                       "value": pl.Float64},
                       orient="row").sort("d")
    return out.lazy()


def pdt_branch_weight_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    dt: pl.DataFrame | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """Per-(d, t) branch weight (lazy) — full multi-branch cascade.

    Mirrors flextool's
    ``preprocessing/period_calculated_params.py:write_branch_weights:364-451``
    pdt loop::

        pdt_branch_weight[d, t] = w[d] / sum w[b] over branches b such that
            (d2, b) ∈ period__branch
            AND (b, t) ∈ dt
            AND (d2, d) ∈ period__branch

    The output is dense over ``dt`` (the ``input.py`` cascade
    overrides any helper output with a dt × value left-join, so we
    must supply at least every (d, t) in dt).  When no
    ``period__branch`` rows are present, defaults to 1.0 per (d, t).

    The denominator-zero rows fall through to 1.0 so dense semantics
    are preserved (matching ``input.py``'s coalesce default).

    Returns a lazy ``(d, t, value)`` frame.
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "t": schema_dtype(_enums, "t"),
              "value": pl.Float64}
    if dt is None or dt.height == 0:
        # Fall back to steps_in_use as the dense domain.
        dt_lf = steps_in_use_lf(workdir, provider=provider).collect()
        if dt_lf.height == 0:
            return _empty_lf(schema)
        dt_pairs_df = dt_lf
    else:
        dt_pairs_df = dt.select("d", "t").unique()
    pb = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    if pb.height == 0:
        # Default 1.0 per (d, t).
        return (dt_pairs_df.lazy()
                  .select(cast_dim(pl.col("d"), _enums, "d"),
                          cast_dim(pl.col("t"), _enums, "t"))
                  .with_columns(value=pl.lit(1.0))
                  .sort("d", "t"))
    bw = solve_branch_weights_lf(workdir, ctx=ctx, provider=provider).collect()
    w_lookup = {row["b"]: float(row["w"]) for row in bw.iter_rows(named=True)
                if row["w"] is not None}
    pb_set = {(row["d"], row["b"]) for row in pb.iter_rows(named=True)}
    # branches_for_t[t] = {b : (b, t) ∈ dt} — derived from the dt set.
    # Mirrors flextool's ``branches_for_t`` from steps_in_use rows.
    branches_for_t: dict[str, set[str]] = {}
    dt_pairs: list[tuple[str, str]] = []
    for r in dt_pairs_df.iter_rows(named=True):
        d, t = str(r["d"]), str(r["t"])
        dt_pairs.append((d, t))
        branches_for_t.setdefault(t, set()).add(d)

    def w(b: str) -> float:
        return w_lookup.get(b, 1.0)

    rows: list[tuple[str, str, float]] = []
    pb_iter = list(pb.iter_rows(named=True))
    for d, t in dt_pairs:
        branches_with_t = branches_for_t.get(t, set())
        denom = 0.0
        for row in pb_iter:
            d2, b = row["d"], row["b"]
            if b not in branches_with_t:
                continue
            if (d2, d) not in pb_set:
                continue
            denom += w(b)
        if denom == 0.0:
            # Dense semantics — fall through to 1.0.
            rows.append((d, t, 1.0))
        else:
            rows.append((d, t, w(d) / denom))
    if not rows:
        return _empty_lf(schema)
    out = pl.DataFrame(rows, schema=schema, orient="row").sort("d", "t")
    return out.lazy()


# ---------------------------------------------------------------------------
# dt_non_anticipativity — (d, t) realised-dispatch ∪ fix-storage-timesteps
# ---------------------------------------------------------------------------


def dt_non_anticipativity_lf(workdir: Path | None,
                              *, provider: "object | None" = None) -> pl.LazyFrame:
    """Compute ``dt_non_anticipativity`` lazily as ``(d, t)``.

    Mirrors flextool's
    ``preprocessing/per_solve_sets.py:267-276``::

        dt_non_anticipativity = realized_dispatch ∪ fix_storage_timesteps

    The four ``non_anticipativity_*`` constraints fire on this set
    (storage_use, online_int, online_lin, reserve).

    Returns the empty frame (schema only) when no stochastic / chain
    activity is present — which keeps the model layer's
    non-anticipativity constraints disabled by default.
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "t": schema_dtype(_enums, "t")}
    rd = realized_dispatch_lf(workdir, provider=provider).collect()
    fs = fix_storage_timesteps_lf(workdir, provider=provider).collect()
    if rd.height == 0 and fs.height == 0:
        return _empty_lf(schema)
    parts: list[pl.LazyFrame] = []
    if rd.height > 0:
        parts.append(rd.lazy())
    if fs.height > 0:
        parts.append(fs.lazy())
    return pl.concat(parts).unique().sort("d", "t")


# ---------------------------------------------------------------------------
# period_branch_full / period_in_use_set — exposed as Param-side frames
# ---------------------------------------------------------------------------


def period_branch_full_lf(workdir: Path | None,
                           *, provider: "object | None" = None) -> pl.LazyFrame:
    """The unfiltered ``period__branch.csv`` as ``(d, b)`` lazy.

    Distinct from the existing ``period_branch`` rolling-handoff field
    (which renames columns to ``d_upper`` / ``d``).  This is the raw
    anchor → sibling map consumed by the model layer's
    non-anticipativity constraints.
    """
    return period_branch_pairs_lf(workdir, provider=provider).select("d", "b")


# ---------------------------------------------------------------------------
# Public Param helpers (collect-at-boundary)
# ---------------------------------------------------------------------------


def pd_branch_weight_param(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> Param | None:
    """Public ``pd_branch_weight`` :class:`Param` builder.

    Returns ``None`` when no realised periods are available (matches
    the ``apply_derived_g`` skip-on-None contract).
    """
    df = pd_branch_weight_lf(workdir, source, active_solve,
                              ctx=ctx, provider=provider).collect()
    if df.height == 0:
        return None
    return Param(("d",), df)


def pdt_branch_weight_param(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    dt: pl.DataFrame | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> Param | None:
    """Public ``pdt_branch_weight`` :class:`Param` builder."""
    df = pdt_branch_weight_lf(workdir, source, active_solve, dt,
                                ctx=ctx, provider=provider).collect()
    if df.height == 0:
        return None
    return Param(("d", "t"), df)


def dt_non_anticipativity_df(workdir: Path | None,
                              *, provider: "object | None" = None,
                              ) -> pl.DataFrame | None:
    """Public ``dt_non_anticipativity`` plain-DataFrame builder."""
    df = dt_non_anticipativity_lf(workdir, provider=provider).collect()
    if df.height == 0:
        return None
    return df


def period_branch_full_df(workdir: Path | None,
                           *, provider: "object | None" = None,
                           ) -> pl.DataFrame | None:
    """Public ``period_branch_full`` plain-DataFrame builder."""
    df = period_branch_full_lf(workdir, provider=provider).collect()
    if df.height == 0:
        return None
    return df


def period_in_use_set_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame | None:
    """Public ``period_in_use_set`` plain-DataFrame builder."""
    df = period_in_use_set_lf(workdir, source, active_solve,
                                ctx=ctx, provider=provider).collect()
    if df.height == 0:
        return None
    return df


# ---------------------------------------------------------------------------
# Scenario-lineage frames — Slice A of the recourse-investment plan
# (``specs/recourse_investment_plan.md`` §6b; design
# ``specs/sliceA_lineage_frames_design.md``).  Consumed by NOTHING at
# HEAD — pure bookkeeping registered on FlexData for Slices B/D/E.
# ---------------------------------------------------------------------------


def _lineage_inputs(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> tuple[list[tuple[str, str]], list[str], dict[str, str]]:
    """Collect ``(pb_rows, piu_list, tb_of)`` as plain-str structures.

    Shared input acquisition for :func:`dd_same_scenario_lf` and
    :func:`pd_non_anticipativity_lf` (design §3.1.1):

    * ``pb_rows`` — ``period__branch`` (anchor d, member b) pairs in
      CSV row order (period order).  Guarded-ctx acquisition per the
      ``dtttdt_from_source`` precedent (``_derived_params.py:
      6578-6605``): ``ctx.period_branch`` is trusted only when BOTH
      columns are null-free — the ``d_anchor`` axis vocabulary is
      empty by contract, so under enum activation the ctx frame's
      anchor column is null-poisoned.  Fallback reads the raw
      provider/workdir CSV directly (NOT via
      :func:`period_branch_pairs_lf`, whose ``.unique()`` does not
      maintain row order — ``pos`` below is derived from the (d, d)
      self-row order, which must stay the period order).
    * ``piu_list`` — ``period_in_use`` members, de-duplicated in row
      order.  Order is content-irrelevant for the lineage frames
      (``pos`` comes from the anchors); it only steers iteration.
    * ``tb_of`` — SBTB member-period → time-branch map (ctx-first via
      ``SolveContext.solve_branch_time_branch``, provider/workdir
      fallback).

    All values are extracted to Python strings — comparisons between
    ``b``-column (branch enum) and ``d``-column (period enum) values
    must never happen Enum-vs-Utf8 or cross-Enum in polars.
    """
    # --- period__branch --------------------------------------------------
    pb_rows: list[tuple[str, str]] = []
    got_pb = False
    if ctx is not None:
        pb_ctx = getattr(ctx, "period_branch", None)
        if (pb_ctx is not None and pb_ctx.height > 0
                and pb_ctx["d_anchor"].null_count() == 0
                and pb_ctx["b"].null_count() == 0):
            pb_rows = [(str(a), str(b)) for a, b in
                       zip(pb_ctx["d_anchor"].to_list(),
                           pb_ctx["b"].to_list())]
            got_pb = True
    if not got_pb and workdir is not None:
        pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
        df = _provider_get(provider, pb_path)
        if df is not None and df.height > 0 and len(df.columns) >= 2:
            cols = df.columns
            d_col = "period" if "period" in cols else cols[0]
            b_col = "branch" if "branch" in cols else cols[1]
            pb_rows = [(str(a), str(b)) for a, b in
                       zip(df[d_col].to_list(), df[b_col].to_list())]
    pb_rows = list(dict.fromkeys(pb_rows))

    # --- period_in_use ---------------------------------------------------
    piu_df = period_in_use_set_lf(workdir, source, active_solve,
                                  ctx=ctx, provider=provider).collect()
    piu = list(dict.fromkeys(
        str(v) for v in piu_df["d"].to_list() if v is not None))

    # --- solve_branch__time_branch (SBTB) --------------------------------
    tb_of: dict[str, str] = {}
    if ctx is not None:
        sbtb = getattr(ctx, "solve_branch_time_branch", None)
        if sbtb is not None and sbtb.height > 0:
            tb_of = {str(d): str(tb) for d, tb in
                     zip(sbtb["d"].to_list(),
                         sbtb["time_branch"].to_list())}
    if not tb_of and workdir is not None:
        sbtb_path = (Path(workdir) / "solve_data"
                     / "solve_branch__time_branch.csv")
        df = _provider_get(provider, sbtb_path)
        if (df is not None and df.height > 0
                and "period" in df.columns and "branch" in df.columns):
            tb_of = {str(d): str(tb) for d, tb in
                     zip(df["period"].to_list(), df["branch"].to_list())}
    return pb_rows, piu, tb_of


def _classify_lineage(
    pb_rows: list[tuple[str, str]],
    piu: list[str],
    tb_of: dict[str, str],
) -> tuple[list[str], dict[str, int], list[str], dict[str, str]]:
    """Classify ``period_in_use`` into anchors / synthetic members.

    Returns ``(anchors, pos, synthetic, anchor_of)``:

    * ``anchors`` — real-named periods: ordered ``d`` values of the
      ``(d, d)`` self-rows of ``period__branch`` restricted to
      ``period_in_use`` (period order).  When ``period__branch`` is
      silent entirely (fixture-only loads without per-solve
      scaffolding), every ``period_in_use`` member is an anchor — the
      deterministic single-trunk fallback, mirroring
      :func:`pd_branch_weight_lf`'s ``pb.height == 0`` branch.
    * ``pos`` — anchor → period-order index.
    * ``synthetic`` — ordered ``period_in_use`` members that are not
      anchors (branch-period fan members with LP variables).
    * ``anchor_of`` — synthetic member → its anchor (from the fan rows
      ``d != b``).

    Failure semantics (design §3.1.1, mirroring ``dtttdt_from_source``
    ``_derived_params.py:6634-6650``): a synthetic ``period_in_use``
    member without a ``period__branch`` fan row, with an anchor that
    has no in-use self-row, or without an SBTB time-branch entry means
    the per-solve artefacts are inconsistent — RAISE instead of
    silently mis-classifying branch periods onto the trunk (the same
    silent-lineage failure class the 4.0.3 fan-out fix eliminated).
    """
    if not pb_rows:
        anchors = list(piu)
        return anchors, {a: i for i, a in enumerate(anchors)}, [], {}
    piu_set = dict.fromkeys(piu)
    anchors = list(dict.fromkeys(
        d for d, b in pb_rows if d == b and d in piu_set))
    pos = {a: i for i, a in enumerate(anchors)}
    anchor_of: dict[str, str] = {}
    for d, b in pb_rows:
        if b != d and b not in anchor_of:
            anchor_of[b] = d
    synthetic = [m for m in piu if m not in pos]
    for m in synthetic:
        if m not in anchor_of:
            raise ValueError(
                "scenario lineage: period_in_use member "
                f"{m!r} has no period__branch row (neither a (d, d) "
                "self-row nor a fan row) — inconsistent per-solve "
                "artefacts; cannot classify the period onto a "
                "scenario leaf."
            )
        if anchor_of[m] not in pos:
            raise ValueError(
                "scenario lineage: synthetic period_in_use member "
                f"{m!r} is anchored to {anchor_of[m]!r}, which has no "
                "in-use (d, d) self-row in period__branch — "
                "inconsistent per-solve artefacts."
            )
        if m not in tb_of:
            raise ValueError(
                "scenario lineage: period__branch declares branch "
                f"period {m!r} (in period_in_use) but "
                "solve_branch__time_branch has no entry for it — "
                "cannot resolve the period's scenario (time-branch).  "
                "The per-solve emitter (emit_branch_weights_and_map) "
                "writes this frame for every solve; its absence "
                "indicates an inconsistent workdir/Provider state."
            )
    return anchors, pos, synthetic, anchor_of


def _lineage_leaves(
    anchors: list[str],
    pos: dict[str, int],
    synthetic: list[str],
    anchor_of: dict[str, str],
    tb_of: dict[str, str],
) -> list[list[str]]:
    """Build the solve's scenario leaves (design §2.2).

    Trunk leaf = all anchors (period order).  One leaf per time-branch
    possessing at least one synthetic ``period_in_use`` member (F3 —
    never iterate SBTB's own time-branches: SBTB also carries the
    realized branch, whose synthetic-member set is empty).  Per branch
    leaf, the per-period representative follows the F6 authority rule:
    the branch-local member is authoritative; a real anchor joins the
    leaf only when it is strictly pre-reveal AND the branch has no
    member for that period.
    """
    leaves: list[list[str]] = [list(anchors)]
    branch_tbs = dict.fromkeys(tb_of[m] for m in synthetic)
    for br in branch_tbs:
        members = [m for m in synthetic if tb_of[m] == br]
        covered = dict.fromkeys(anchor_of[m] for m in members)
        k = min(pos[anchor_of[m]] for m in members)
        pre = [a for a in anchors if pos[a] < k and a not in covered]
        leaf = sorted(pre + members,
                      key=lambda x: pos[x] if x in pos
                      else pos[anchor_of[x]])
        leaves.append(leaf)
    return leaves


def _finalize_lineage_frame(rows: list[tuple[str, str]],
                            col_a: str, axis_a: str,
                            col_b: str, axis_b: str,
                            what: str) -> pl.LazyFrame:
    """Utf8-construct → non-strict ``cast_dim`` → live null assertion
    → deterministic Utf8-key sort (design §3.1.2 steps 4-5, F7).

    Constructing directly with a strict Enum schema would raise inside
    polars at construction and make the null check dead code; the
    Utf8-construct → non-strict-cast → assert sequence keeps the check
    live.  A nulled row after the cast means a period/branch token is
    missing from the axis vocabulary — a wiring bug, not a data state.
    When activation is off, ``cast_dim`` is a no-op and the assertion
    is trivially true.
    """
    raw = pl.DataFrame({col_a: [r[0] for r in rows],
                        col_b: [r[1] for r in rows]},
                       schema={col_a: pl.Utf8, col_b: pl.Utf8})
    out = raw.select(cast_dim(pl.col(col_a), _enums, axis_a),
                     cast_dim(pl.col(col_b), _enums, axis_b))
    if out[col_a].null_count() or out[col_b].null_count():
        bad = sorted({v for col in (col_a, col_b)
                      for v, c in zip(raw[col].to_list(),
                                      out[col].to_list())
                      if c is None})
        raise ValueError(
            f"{what}: axis-enum cast nulled token(s) {bad} — the "
            "token is missing from the live axis vocabulary (wiring "
            "bug; the 4.0.4 vocabulary splice includes every "
            "continuation branch-period token)."
        )
    out = out.sort(pl.col(col_a).cast(pl.Utf8),
                   pl.col(col_b).cast(pl.Utf8))
    return out.lazy()


def dd_same_scenario_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """``dd_same_scenario`` — (d, d_other) scenario-lineage pairs.

    "``d_other`` is on ``d``'s information path", formalized as: some
    scenario leaf-path contains both periods (design §2.2).  Symmetric
    and reflexive; domain = ``period_in_use`` only (metadata-only fan
    members and history periods excluded).  For deterministic /
    realized-only solves the frame degenerates to the all-pairs cross
    product over ``period_in_use`` ("everything shares one path").

    Binding consumer rule (F4, restated from the design): a lineage
    semi-join may only REMOVE pairs whose BOTH ends are in
    ``period_in_use``; any pair with either end outside the domain
    passes through unconditionally (history-anchored walk rows).

    Returns a lazy ``(d, d_other)`` frame, deterministically sorted by
    Utf8-cast keys.  Typed empty frame when ``period_in_use`` is
    empty.
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "d_other": schema_dtype(_enums, "d")}
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if not piu:
        return _empty_lf(schema)
    anchors, pos, synthetic, anchor_of = _classify_lineage(
        pb_rows, piu, tb_of)
    leaves = _lineage_leaves(anchors, pos, synthetic, anchor_of, tb_of)
    seen: dict[tuple[str, str], None] = {}
    for leaf in leaves:
        for x in leaf:
            for y in leaf:
                seen[(x, y)] = None
    if not seen:
        return _empty_lf(schema)
    return _finalize_lineage_frame(list(seen), "d", "d", "d_other",
                                   "d_other", "dd_same_scenario")


def dd_same_scenario_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Collect wrapper for :func:`dd_same_scenario_lf` — always a
    typed frame, never ``None`` (design §2.5)."""
    return dd_same_scenario_lf(workdir, source, active_solve,
                               ctx=ctx, provider=provider).collect()


def d_leaf_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """``d_leaf`` — (d, leaf) scenario-leaf PARTITION of ``period_in_use``.

    Each in-use period maps to exactly ONE scenario leaf (design §7.2):
    real-named anchors → the ``"__realized"`` leaf; synthetic fan
    members → their time-branch id (``tb_of[m]``).  Because the
    continuation fan starts at the solve's first step (§K), leaves
    PARTITION the invest axis — no period is on two leaves — so the map
    is well-defined and the per-leaf invest sums are disjoint.

    Consumed by the per-path total caps
    (``maxInvest/maxDivest_entity_total_path`` — ``model.py`` — and the
    divest ``maxDivestGroup_entity_total_path`` — ``_cumulative_invest.py``):
    the leaf column joins into each cap row's key so a total cap applies
    once per scenario path, never cross-scenario double-counting.

    Deterministic / flag-off solves degenerate to a single
    ``"__realized"`` leaf (``period__branch`` silent → every in-use
    member is an anchor, mirroring :func:`_classify_lineage`'s
    ``pb_rows == []`` branch), so the total caps stay on the legacy
    single-row shape (byte-parity).  ``leaf`` is a plain ``Utf8``
    column — it is NOT an axis vocabulary; it only keys constraint rows.
    """
    schema = {"d": schema_dtype(_enums, "d"), "leaf": pl.Utf8}
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if not piu:
        return _empty_lf(schema)
    anchors, _pos, synthetic, _anchor_of = _classify_lineage(
        pb_rows, piu, tb_of)
    rows: list[tuple[str, str]] = [(a, "__realized") for a in anchors]
    rows += [(m, tb_of[m]) for m in synthetic]
    raw = pl.DataFrame({"d": [r[0] for r in rows],
                        "leaf": [r[1] for r in rows]},
                       schema={"d": pl.Utf8, "leaf": pl.Utf8})
    out = raw.select(cast_dim(pl.col("d"), _enums, "d"), pl.col("leaf"))
    if out["d"].null_count():
        bad = sorted({v for v, c in zip(raw["d"].to_list(),
                                        out["d"].to_list()) if c is None})
        raise ValueError(
            f"d_leaf: axis-enum cast nulled period token(s) {bad} — the "
            "token is missing from the live d-axis vocabulary (wiring "
            "bug; the 4.0.4 vocabulary splice includes every continuation "
            "branch-period token)."
        )
    out = out.sort(pl.col("d").cast(pl.Utf8), "leaf")
    return out.lazy()


def d_leaf_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Collect wrapper for :func:`d_leaf_lf` — always a typed frame."""
    return d_leaf_lf(workdir, source, active_solve,
                     ctx=ctx, provider=provider).collect()


def pd_non_anticipativity_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
    branch_start: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    """``pd_non_anticipativity`` — (d, b) invest-NA period pairs.

    The period-level analog of dispatch ``db_pairs``: pairs whose
    INVEST decisions must be tied when Slice E's
    ``non_anticipativity_invest_p/n`` families land.  ``(d, b)`` is
    emitted iff ``b`` is a synthetic in-use fan member, ``d`` is its
    anchor, and ``d`` lies STRICTLY before the information reveal of
    ``b``'s branch (the branching period itself is NOT tied — recourse
    invest at the branching period is per-scenario by owner decision,
    plan §7).

    ``branch_start`` (optional, ``[time_branch, d_start]``) supplies
    the authoritative reveal period per time-branch — the Slice E
    hook.  When ``None`` (the current-universe normal; the runner-side
    ``branch_start_time_lists`` is not persisted), the fallback
    ``k(br) = min over synthetic members m of br of pos(anchor(m))``
    is exact for every topology constructible at HEAD, under which the
    emit condition is unsatisfiable — the frame is provably EMPTY for
    every currently-constructible solve (design §2.3).  On future
    fanned-pre-reveal shapes the fallback deliberately under-detects;
    Slice E must supply ``branch_start``.

    Returns a lazy ``(d, b)`` frame (``d`` = anchor / pin target on
    the d axis, ``b`` = pinned sibling member on the branch axis,
    mirroring dispatch ``db_pairs`` column roles).
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "b": schema_dtype(_enums, "b")}
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if not piu:
        return _empty_lf(schema)
    anchors, pos, synthetic, anchor_of = _classify_lineage(
        pb_rows, piu, tb_of)
    if not synthetic:
        return _empty_lf(schema)
    bs_of: dict[str, str] = {}
    if branch_start is not None and branch_start.height > 0:
        bs_of = {str(tb): str(d) for tb, d in
                 zip(branch_start["time_branch"].to_list(),
                     branch_start["d_start"].to_list())}
    k_fallback: dict[str, int] = {}
    for m in synthetic:
        br = tb_of[m]
        p = pos[anchor_of[m]]
        if br not in k_fallback or p < k_fallback[br]:
            k_fallback[br] = p
    rows: list[tuple[str, str]] = []
    for m in synthetic:
        br = tb_of[m]
        if br in bs_of:
            d_start = bs_of[br]
            if d_start not in pos:
                raise ValueError(
                    "pd_non_anticipativity: branch_start maps "
                    f"time-branch {br!r} to period {d_start!r}, which "
                    "is not an in-use anchor period."
                )
            k = pos[d_start]
        else:
            k = k_fallback[br]
        if pos[anchor_of[m]] < k:
            rows.append((anchor_of[m], m))
    if not rows:
        return _empty_lf(schema)
    return _finalize_lineage_frame(rows, "d", "d", "b", "b",
                                   "pd_non_anticipativity")


def pd_non_anticipativity_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
    branch_start: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Collect wrapper for :func:`pd_non_anticipativity_lf` — always a
    typed frame, never ``None`` (design §2.5)."""
    return pd_non_anticipativity_lf(
        workdir, source, active_solve, ctx=ctx, provider=provider,
        branch_start=branch_start).collect()


# ---------------------------------------------------------------------------
# check_recourse_npv_preconditions — the three Slice A §2.2 NPV proof
# obligations, shipped as a pure checker (Slice B design §6).  Exercised
# by tests only until the Slice D activation sites wire it at runtime.
# ---------------------------------------------------------------------------


def _read_solve_data_csv(workdir: Path | None,
                         provider: "object | None",
                         name: str) -> pl.DataFrame | None:
    """Provider-only fetch of ``solve_data/<name>`` — the canonical
    cascade pathway (matches every other reader in this module).
    Returns ``None`` when the Provider is missing or does not carry the
    frame.

    The obligation checker verifies the *emitted* solve_data year rows
    (design §6.2 obligation (iii)); post-Step-2 those bytes reach the
    cascade through the Provider, so this reader routes through
    ``provider.get`` and never touches disk.  The caller owns ensuring
    the Provider carries the true emitted CSVs — in tests the fixtures
    seed the Provider from the workdir (``seed_provider_from_dir`` /
    ``_mk_provider``); at the Slice D activation site the same
    solve_data frames the rest of the cascade already consumes are in
    the Provider by construction.  ``workdir`` is retained in the
    signature only to key the canonical Provider path.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / name
    df = _provider_get(provider, p)
    if df is not None and df.height > 0:
        return df
    return None


def _p_years_d_rows(workdir: Path | None,
                    provider: "object | None",
                    ) -> dict[str, float] | None:
    """``{period: cumulative year}`` from ``solve_data/p_years_d.csv``
    (fallback ``period_with_history.csv``, ``param`` column) — the
    canonical emitted CSVs that carry the fan-member byte-copies
    (``_emit_solve_writers.py`` years-represented / period-years
    copy loops)."""
    df = _read_solve_data_csv(workdir, provider, "p_years_d.csv")
    val_col = None
    if df is not None and "period" in df.columns:
        val_col = "value" if "value" in df.columns else df.columns[-1]
    else:
        df = _read_solve_data_csv(workdir, provider,
                                  "period_with_history.csv")
        if df is not None and "period" in df.columns:
            val_col = "param" if "param" in df.columns else df.columns[-1]
    if df is None or val_col is None:
        return None
    out: dict[str, float] = {}
    for d, v in zip(df["period"].to_list(), df[val_col].to_list()):
        if d is None or v is None:
            continue
        try:
            out[str(d)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _p_years_represented_rows(workdir: Path | None,
                              provider: "object | None",
                              ) -> dict[str, list[tuple[str, float]]] | None:
    """Ordered ``{period: [(year_label, width), ...]}`` from
    ``solve_data/p_years_represented.csv`` (canonical columns
    ``period, years_from_solve, p_years_from_solve,
    p_years_represented`` — label = second column, width = last)."""
    df = _read_solve_data_csv(workdir, provider,
                              "p_years_represented.csv")
    if df is None or "period" not in df.columns or len(df.columns) < 2:
        return None
    label_col = df.columns[1]
    width_col = df.columns[-1]
    out: dict[str, list[tuple[str, float]]] = {}
    for d, y, w in zip(df["period"].to_list(), df[label_col].to_list(),
                       df[width_col].to_list()):
        if d is None or y is None or w is None:
            continue
        try:
            out.setdefault(str(d), []).append((str(y), float(w)))
        except (TypeError, ValueError):
            continue
    return out


def check_recourse_npv_preconditions(
        workdir: Path | None,
        source: "InputSource | None" = None,
        active_solve: str | None = None,
        *,
        ctx: "object | None" = None,
        provider: "object | None" = None,
        weight_tol: float = 1e-9,
        ) -> list[str]:
    """Return violation messages for the three Slice A §2.2 NPV
    obligations ((i) leaf-weight constancy, (ii) cohort sum-to-1,
    (iii) byte-equal fan-member year rows).  Empty list == all hold.
    Deterministic message order.  Raising wrapper:
    :func:`assert_recourse_npv_preconditions`.

    These obligations pin the arithmetic that makes a SET-shaped
    (unweighted) lineage filter numerically correct for
    trunk-crossing windows and realized-anchor windows (recourse plan
    §9.4) — on the invest AND divest side (variant-agnostic: they
    constrain inputs, not any one walker).  Runtime wiring is the
    Slice D activation site (once per solve, before the first
    lineage-passing call); until then this checker is exercised by
    tests only.

    SOUNDNESS SCOPE (Slice B design F4/§6.2): obligation (iii) is
    checked against the emitted solve_data year frames (``p_years_d`` /
    ``period_with_history`` / ``p_years_represented``).  This checker
    performs NO disk I/O — it reads those frames through the Provider
    (``provider.get``), so the CALLER owns making the true emitted CSVs
    available on the Provider before invoking it (tests seed the
    Provider from the workdir; the Slice D activation site inherits the
    same solve_data frames the rest of the cascade already consumes).
    The check underwrites the lineage-filter arithmetic IF AND ONLY IF the
    walker's year/factor source is those same CSVs (the Slice D
    year-threading prerequisite, Slice B design §6.4).  If Slice D
    instead anchor-maps branch-period years/factors inside the
    walker, a FOURTH obligation is required: per fan member, the
    anchor-map's resolved (year, factor) equals the CSV rows this
    checker verified — without it the checker re-decouples from the
    walker, recreating the §6.4 defect.

    Obligation (i)'s chains are defined against SBTB (which carries
    the realized branch's real-named members too, so the trunk chain
    is covered).  A Slice E mid-horizon reveal will need to re-scope
    constancy to post-reveal segments — out of scope here.
    """
    violations: list[str] = []
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if not piu:
        return []
    piu_set = dict.fromkeys(piu)

    wdf = pd_branch_weight_lf(workdir, source, active_solve,
                              ctx=ctx, provider=provider).collect()
    weights: dict[str, float] = {}
    for d, v in zip(wdf["d"].to_list(), wdf["value"].to_list()):
        if d is not None and v is not None:
            weights[str(d)] = float(v)

    # --- (i) per-leaf weight constancy ------------------------------------
    # Member chains = PIU representatives grouped by SBTB time-branch.
    chains: dict[str, list[str]] = {}
    for m in piu:
        tb = tb_of.get(m)
        if tb is not None:
            chains.setdefault(tb, []).append(m)
    for tb, members in chains.items():
        present = [(m, weights[m]) for m in members if m in weights]
        missing = [m for m in members if m not in weights]
        if missing and present:
            violations.append(
                "obligation (i) leaf-weight constancy: chain "
                f"{tb!r} member(s) {missing} carry no pd_branch_weight "
                "row while sibling members do."
            )
        if len({v for _, v in present}) > 1:
            detail = ", ".join(f"{m}={v!r}" for m, v in present)
            violations.append(
                "obligation (i) leaf-weight constancy: chain "
                f"{tb!r} weights are not constant along the leaf: "
                f"{detail}."
            )

    # --- (ii) cohort weights sum to 1 -------------------------------------
    # Fan rows (anchor -> members) in period__branch row order.
    fan_of: dict[str, list[str]] = {}
    for d, b in pb_rows:
        if b != d:
            fan_of.setdefault(d, []).append(b)
    for a, members in fan_of.items():
        if a not in piu_set:
            continue
        cohort = [a] + [m for m in members if m in piu_set]
        s = sum(weights.get(m, 0.0) for m in cohort)
        if abs(s - 1.0) > weight_tol:
            violations.append(
                "obligation (ii) cohort weights sum to 1: anchor "
                f"{a!r} cohort {cohort} weights sum to {s!r}."
            )

    # --- (iii) byte-equal fan-member year rows ----------------------------
    fan_pairs = [(a, m) for a, members in fan_of.items()
                 for m in members if a in piu_set and m in piu_set]
    if fan_pairs:
        pyd = _p_years_d_rows(workdir, provider)
        pyr = _p_years_represented_rows(workdir, provider)
        if pyd is None:
            violations.append(
                "obligation (iii) byte-equal year rows: cannot verify "
                "— neither solve_data/p_years_d.csv nor "
                "period_with_history.csv is readable while fan pairs "
                "exist."
            )
        if pyr is None:
            violations.append(
                "obligation (iii) byte-equal year rows: cannot verify "
                "— solve_data/p_years_represented.csv is not readable "
                "while fan pairs exist."
            )
        for a, m in fan_pairs:
            if pyd is not None:
                ya, ym = pyd.get(a), pyd.get(m)
                if ym != ya:
                    violations.append(
                        "obligation (iii) byte-equal year rows: "
                        f"p_years_d[{m!r}] = {ym!r} differs from its "
                        f"anchor p_years_d[{a!r}] = {ya!r}."
                    )
            if pyr is not None:
                ra = pyr.get(a, [])
                rm = pyr.get(m, [])
                if rm != ra:
                    violations.append(
                        "obligation (iii) byte-equal year rows: "
                        f"p_years_represented rows of {m!r} ({rm!r}) "
                        f"differ from its anchor {a!r} ({ra!r})."
                    )
    return violations


def assert_recourse_npv_preconditions(
        workdir: Path | None,
        source: "InputSource | None" = None,
        active_solve: str | None = None,
        *,
        ctx: "object | None" = None,
        provider: "object | None" = None,
        weight_tol: float = 1e-9,
        ) -> None:
    """Raising wrapper for :func:`check_recourse_npv_preconditions`."""
    violations = check_recourse_npv_preconditions(
        workdir, source, active_solve, ctx=ctx, provider=provider,
        weight_tol=weight_tol)
    if violations:
        from flextool.engine_polars._solve_state import LineageFilterError
        raise LineageFilterError(
            "recourse NPV preconditions violated:\n  "
            + "\n  ".join(violations)
        )


# ---------------------------------------------------------------------------
# apply_branch_cluster — single-pass entry for the apply_derived_g
# integration.  Mutates ``flex_data`` in place.
# ---------------------------------------------------------------------------


def apply_branch_cluster(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Cluster D Params to ``flex_data``.

    Order:
      1. ``period_branch_full`` (no dependencies; trivial CSV unroll).
      2. ``period_in_use_set`` (depends on workdir or source).
      3. ``dt_non_anticipativity`` (depends on realized + fix-storage CSVs).
      2b. ``dd_same_scenario`` / ``pd_non_anticipativity`` — Slice A
         scenario-lineage frames (always-on; degenerate all-pairs /
         empty content for deterministic solves; consumed by nothing
         until Slice B/E).
      4. ``pd_branch_weight`` (depends on period_branch_full + first_ts +
         period_in_use_set + branch weights).
      5. ``pdt_branch_weight`` (depends on dt + period_branch_full +
         branch weights).

    Δ.12b — assignment is now unconditional.  Each helper returns
    ``None`` when the corresponding CSV is missing/empty (e.g.
    single-solve fixtures with no branches); ``None`` is the explicit
    "feature inactive" signal — same outcome the seed produces.  No
    silent fall-through to a CSV-loaded value.

    R-O6 invariant: this helper does NOT touch ``invest_periods`` or
    ``v_invest`` — only the operational dispatch-side weights and
    the non-anticipativity gate.  The Slice A lineage frames are
    descriptive bookkeeping only; branches do not enter
    ``invest_periods`` in this slice.
    """
    dt = getattr(flex_data, "dt", None)

    # 1-2. Set frames — None == "no branches / no realized periods" is
    # the legitimate inactive-feature signal.  Seed-preserving overlay:
    # when this lazy override yields nothing (no provider + no workdir
    # CSVs reachable, e.g. fixture-only ``load_flextool`` paths), keep
    # the seed value set by ``_load_branch_artefacts`` — the override is
    # the cascade-side producer, not a forced reset.
    _new_pbf = period_branch_full_df(workdir, provider=provider)
    if _new_pbf is not None or getattr(flex_data, "period_branch_full", None) is None:
        flex_data.period_branch_full = _new_pbf
    _new_piu = period_in_use_set_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if _new_piu is not None or getattr(flex_data, "period_in_use_set", None) is None:
        flex_data.period_in_use_set = _new_piu
    _new_dtn = dt_non_anticipativity_df(workdir, provider=provider)
    if _new_dtn is not None or getattr(flex_data, "dt_non_anticipativity", None) is None:
        flex_data.dt_non_anticipativity = _new_dtn

    # 2b. Scenario-lineage frames (Slice A — recourse plan §6b).
    #     Always-on; degenerate all-pairs content for deterministic
    #     solves; consumed by nothing until Slice B/E.  Unconditional
    #     assignment — these fields have no CSV seed to preserve
    #     (builders always return a typed frame, never None).
    flex_data.dd_same_scenario = dd_same_scenario_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    flex_data.pd_non_anticipativity = pd_non_anticipativity_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    # Slice D per-path total caps (design §7.2): the (d, leaf) partition
    # that keeps entity/group total caps per-scenario-path.  Single
    # "__realized" leaf for deterministic solves (byte-parity).
    flex_data.d_leaf = d_leaf_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)

    # 3-4. Branch-weight Params (lazy ports of the previous eager
    # helpers in ``_derived_params.py``).
    _new_pd_bw = pd_branch_weight_param(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if _new_pd_bw is not None or getattr(flex_data, "pd_branch_weight", None) is None:
        flex_data.pd_branch_weight = _new_pd_bw

    pdt_bw = pdt_branch_weight_param(workdir, source, active_solve, dt,
                                          ctx=ctx, provider=provider)
    if pdt_bw is not None and dt is not None and dt.height > 0:
        # Match input.py's dense-dt semantics: when dt is supplied,
        # build a (d, t)-dense Param via left-join + coalesce default
        # of 1.0.  This mirrors the previous CSV-cascade behaviour at
        # input.py:2845-2870 (preserves exact frame shape).
        # Defensive re-cast: cast d/t to canonical Enum so the left-join
        # against ``pdt_bw.lazy`` (Enum d/t) composes cleanly even when
        # ``dt`` arrives with Utf8 columns.
        base = (dt.lazy()
                  .with_columns(value=pl.lit(1.0))
                  .select(alias_to_axis("d", "d"),
                          alias_to_axis("t", "t"),
                          "value"))
        joined = (base
                    .join(pdt_bw.lazy,
                          on=["d", "t"], how="left", suffix="__r")
                    .with_columns(value=pl.coalesce(
                        pl.col("value__r"), pl.col("value")))
                    .select("d", "t", "value")
                    .collect())
        pdt_bw = Param(("d", "t"), joined)
    # Seed-preserving overlay (same rationale as the set frames above):
    # only overwrite the seed when this cascade-side path produces a
    # value.  Fixture-only loads (no provider, no workdir scaffolding)
    # rely on ``_load_branch_artefacts``' disk-fallback seed.
    if pdt_bw is not None or getattr(flex_data, "pdt_branch_weight", None) is None:
        flex_data.pdt_branch_weight = pdt_bw


__all__ = [
    "period_branch_pairs_lf",
    "solve_branch_weights_lf",
    "first_timesteps_lf",
    "period_in_use_set_lf",
    "realized_dispatch_lf",
    "fix_storage_timesteps_lf",
    "steps_in_use_lf",
    "pd_branch_weight_lf",
    "pdt_branch_weight_lf",
    "dt_non_anticipativity_lf",
    "period_branch_full_lf",
    "dd_same_scenario_lf",
    "dd_same_scenario_df",
    "d_leaf_lf",
    "d_leaf_df",
    "pd_non_anticipativity_lf",
    "pd_non_anticipativity_df",
    "pd_branch_weight_param",
    "pdt_branch_weight_param",
    "dt_non_anticipativity_df",
    "period_branch_full_df",
    "period_in_use_set_df",
    "apply_branch_cluster",
    "check_recourse_npv_preconditions",
    "assert_recourse_npv_preconditions",
]
