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

import contextvars
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    cast_dim,
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine.preprocess._provider_io import _provider_key


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
    from flextool.engine.input._input_source import InputSource


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
        # ``ctx.period_branch`` is null-free: under axis Enums its loader
        # raises on a label outside the ``d_anchor`` / ``branch`` vocabulary.
        if (pb_df is not None and pb_df.height > 0
                and "d_anchor" in pb_df.columns):
            return (pb_df.lazy()
                          .select(alias_to_axis("d_anchor", "d"),
                                  cast_dim(pl.col("b"), _enums, "b"))
                          .unique(maintain_order=True))
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    lf = _maybe_provider_lf(provider, p, rename={"period": "d", "branch": "b"})
    if lf is None:
        return _empty_lf(schema)
    return lf.select(
        cast_dim(pl.col("d"), _enums, "d"),
        cast_dim(pl.col("b"), _enums, "b"),
    ).unique(maintain_order=True)


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
            return piu_df.lazy().select("d").unique(maintain_order=True)
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        df = _provider_get(provider, p)
        if df is not None and df.height > 0 and df.columns:
            col = df.columns[0]
            return (df.lazy()
                      .select(alias_to_axis(col, "d"))
                      .unique(maintain_order=True))
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
    return pl.concat(parts).unique(maintain_order=True)


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
              .unique(maintain_order=True))


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
              .unique(maintain_order=True))


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
              .unique(maintain_order=True))


def fan_period_labels(pb_set: "set[tuple[str, str]]") -> set[str]:
    """The period labels of a solve: the only definition of a period label.

    A *period label* is a period ``d`` that is both a self-row (``(d, d) ∈ period__branch``) AND the parent of ≥1
    forecast member (``∃ b ≠ d : (d, b) ∈ period__branch``): the original name of a period that opens a stochastic
    fan.  It is bookkeeping only — period-level investment, years / NPV rows, the lineage anchor — and never has
    dispatch ((d, t) rows): the fan is forecasts-only (Σ = 1), so the label carries no probability mass and the
    branch-weight producers emit weight 0 for it and skip it from every forecast denominator.

    A self-row with NO fan member is a pre-reveal period (or a deterministic one) and is NOT a label, so it keeps
    its full 1.0 weight.  Returns the empty set for a deterministic solve (``pb_set`` empty).
    """
    trunk = {d for d, b in pb_set if d == b}
    has_fan = {d for d, b in pb_set if b != d}
    return trunk & has_fan


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
    # Forecasts-only cohort rule (R2/W7, unconditional) — a PHANTOM period is a
    # trunk self-row ``(d, d) ∈ pb`` that ALSO fans out to ≥1 forecast member
    # ``(d, b), b ≠ d``.  The stochastic fan is forecasts-only (Σ=1), so the
    # phantom trunk carries NO probability mass: it is skipped from every
    # denominator and emits weight 0; the forecast members renormalise among
    # themselves.  A trunk with NO fan member is a genuine pre-reveal cohort-of-
    # one (or a deterministic period) and is NOT phantom → it stays 1.0 (the
    # ``has_fan`` guard).  This is the W4d-A exclusion made unconditional (no
    # ``dispatch_realization_method`` marker) and extended to the invest weight.
    phantom = fan_period_labels(pb_set)
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
        if d in phantom:
            # A period label has no steps (no first timestep) but keeps its weight-0 row.
            rows.append((d, 0.0))
            continue
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
            if b in phantom:
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

    The output is dense over ``dt`` (the ``flex_data.py`` cascade
    overrides any helper output with a dt × value left-join, so we
    must supply at least every (d, t) in dt).  When no
    ``period__branch`` rows are present, defaults to 1.0 per (d, t).

    The denominator-zero rows fall through to 1.0 so dense semantics
    are preserved (matching ``flex_data.py``'s coalesce default).

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
    # Forecasts-only cohort rule (R2, unconditional) — see
    # :func:`pd_branch_weight_lf`.  A PHANTOM period (``(d, d) ∈ pb`` that also
    # fans out to a forecast member) carries no probability mass: emit 0 and
    # skip it from every forecast denominator.  Applied unconditionally (no
    # ``dispatch_realization_method`` marker); a lone pre-reveal / deterministic
    # trunk has no fan member and is NOT phantom → stays 1.0.
    phantom = fan_period_labels(pb_set)
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
        if d in phantom:
            # Phantom trunk operational weight → 0 (forecasts carry the mass).
            rows.append((d, t, 0.0))
            continue
        branches_with_t = branches_for_t.get(t, set())
        denom = 0.0
        for row in pb_iter:
            d2, b = row["d"], row["b"]
            if b not in branches_with_t:
                continue
            if (d2, d) not in pb_set:
                continue
            if b in phantom:
                # Exclude the phantom trunk from the forecast denominator.
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
    """Read ``dt_non_anticipativity`` lazily as ``(d, t)``.

    Slice G (F1) — the operational non-anticipativity set is now the
    single source of truth materialised by
    ``_per_solve.emit_per_solve_sets`` as
    ``solve_data/dt_non_anticipativity_set.csv``, driven by the resolved
    ``non_anticipativity_periods`` window:

    * unset window  -> the legacy ``realized_dispatch ∪ fix_storage``
      union (byte-identical to prior behaviour);
    * empty window  -> a header-only (present-but-empty) CSV -> the
      model's ``if dtna.height == 0: return`` (``model/core.py``) skips ALL
      four families -> operations free from t0;
    * curated window -> those periods' timesteps ∩ ``steps_in_use``.

    The four ``non_anticipativity_*`` constraints fire on this set
    (storage_use, online_int, online_lin, reserve).  This reader sorts,
    so the CSV's row-order is irrelevant to byte-parity.

    Defensive fallback: when the set CSV is absent from the Provider
    (older snapshots / direct callers that never ran per-solve emit),
    recompute the legacy ``realized_dispatch ∪ fix_storage`` union so
    behaviour is unchanged.
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "t": schema_dtype(_enums, "t")}
    if workdir is None:
        return _empty_lf(schema)
    # Authoritative set CSV (emitted every solve; may be header-only).
    p = Path(workdir) / "solve_data" / "dt_non_anticipativity_set.csv"
    df = _provider_get(provider, p)
    if df is not None:
        # Present -> the window has already been resolved into this file.
        # Empty-but-present -> empty window -> operations free from t0.
        if df.height == 0:
            return _empty_lf(schema)
        cols = df.columns
        d_col = "period" if "period" in cols else cols[0]
        t_col = ("step" if "step" in cols else
                 ("time" if "time" in cols else cols[1]))
        return (df.lazy()
                  .select(alias_to_axis(d_col, "d"),
                          alias_to_axis(t_col, "t"))
                  .unique()
                  .sort("d", "t"))
    # Defensive fallback — recompute the legacy union.
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
# Branch periods inherit the period-keyed values of their anchor period
# ---------------------------------------------------------------------------
#
#   value[k, br] = authored[k, br]            if k has an authored row in br
#                = authored[k, anchor(br)]    otherwise, (anchor, br) ∈ period__branch, anchor ≠ br
#
# Branch periods (``period1_upper``, ``p2035_high``, …) are minted per solve, so no input value can be
# keyed by them; they are possible futures of their anchor period, so the anchor's per-period value
# applies.  A period map ``{anchor: v}`` therefore gives the anchor's branch periods the same rows as
# the scalar ``v``.


def _validated_pairs(pairs: "pl.DataFrame | None", caller: str) -> "pl.DataFrame | None":
    """``(anchor, br)`` Utf8 pairs without self rows, unique and sorted; None when there are none.

    Missing ``anchor`` / ``br`` columns, a null cell, or a branch period with more than one anchor
    raise ``ValueError``: one anchor value per branch period is what inheritance needs.
    """
    if pairs is None or pairs.height == 0:
        return None
    missing = sorted({"anchor", "br"} - set(pairs.columns))
    if missing:
        raise ValueError(f"{caller}: pairs frame lacks column(s) {missing}; expected (anchor, br).")
    out = pairs.select(pl.col("anchor").cast(pl.Utf8), pl.col("br").cast(pl.Utf8))
    if out["anchor"].null_count() or out["br"].null_count():
        raise ValueError(f"{caller}: pairs frame has null anchor or branch periods.")
    out = out.filter(pl.col("anchor") != pl.col("br")).unique().sort("br", "anchor")
    if out.height == 0:
        return None
    multi = (out.group_by("br").agg(pl.col("anchor").n_unique().alias("n"))
                .filter(pl.col("n") > 1).sort("br"))
    if multi.height > 0:
        raise ValueError(
            f"{caller}: branch period(s) {multi['br'].to_list()} have more than one anchor period in "
            "period__branch; a branch period inherits the values of exactly one anchor.")
    return out


def branch_anchor_pairs(provider: "object | None",
                        periods: "pl.Series | list[str] | None",
                        ) -> pl.DataFrame | None:
    """``(anchor, br)`` pairs of the stochastic branch periods, Utf8; None when there are none.

    Read from ``solve_data/period__branch`` (columns ``period`` = anchor, ``branch``).  Self rows
    (trunks, deterministic periods) are dropped.  ``periods`` is required: it keeps only the branch
    periods in that set (the consumer's in-use periods), so realized-designated and zero-weight
    branch periods without active time are filtered out.  Pass ``None`` explicitly only when the
    caller restricts to its own period set downstream.  A branch period with more than one anchor
    raises ``ValueError``.
    """
    pb = _provider_get(provider, Path("solve_data") / "period__branch.csv")
    if pb is None or pb.height == 0:
        return None
    pairs = pb.select(pl.col(pb.columns[0]).cast(pl.Utf8).alias("anchor"),
                      pl.col(pb.columns[1]).cast(pl.Utf8).alias("br"))
    if periods is not None:
        keep = pl.Series("br", list(periods), dtype=pl.Utf8)
        pairs = pairs.filter(pl.col("br").is_in(keep.implode()))
    return _validated_pairs(pairs, "branch_anchor_pairs")


def inherit_anchor_period(frame: pl.DataFrame, pairs: pl.DataFrame | None, *,
                          group_keys: "tuple[str, ...] | list[str]",
                          period_col: str = "d") -> pl.DataFrame:
    """``frame`` plus, for every branch period of ``pairs``, the rows of its anchor period.

    ``frame`` has a period column ``period_col`` and any other columns (keys and values).
    ``pairs`` holds ``(anchor, br)``; it is validated here (columns, nulls, one anchor per branch
    period; self rows dropped; Utf8 / Enum / Categorical all accepted).  For each ``group_keys``
    value, a branch period that already has rows of its own keeps them and gets none of the
    anchor's ("branch-native rows win per ``group_keys`` + period", null keys compared as equal):
    an authored value set for a period (e.g. a whole tier curve keyed by tier, or a time curve) is
    never mixed with the anchor's.  The rule is the one ``PdLookup`` (own row, else anchor),
    ``derive_years_represented`` and ``_anchor_expand_explicit`` use; because of it, applying the
    helper twice, or after an upstream expansion to branch periods, adds nothing.  Rows with a null
    period are never joined to an anchor.

    Authored rows come first, inherited rows follow in ``(anchor rows, br)`` order.  The period
    column is joined as Utf8 and cast back to its input dtype strictly (a null from the cast is a
    wiring bug and raises).  None / empty ``pairs`` returns ``frame`` unchanged.

    Eager by design: every caller passes a source frame or a de-duplicated lookup dict, before any
    lazy broadcast or vectorized-emit join.  Do not add a collect on a lazy emit path to call it.
    """
    pairs = _validated_pairs(pairs, "inherit_anchor_period")
    if pairs is None or frame.height == 0:
        return frame
    group_keys = list(group_keys)
    cols = frame.columns
    d_dtype = frame.schema[period_col]
    base = frame.with_columns(pl.col(period_col).cast(pl.Utf8))
    inherited = (base.join(pairs, left_on=period_col, right_on="anchor",
                           how="inner", maintain_order="left_right")
                     .drop(period_col)
                     .rename({"br": period_col})
                     .join(base.select([*group_keys, period_col]).unique(),
                           on=[*group_keys, period_col], how="anti", nulls_equal=True,
                           maintain_order="left")
                     .select(cols))
    if inherited.height == 0:
        return frame
    out = pl.concat([base, inherited], how="vertical")
    if d_dtype != pl.Utf8:
        casted = out.with_columns(pl.col(period_col).cast(d_dtype, strict=False))
        if casted[period_col].null_count() > out[period_col].null_count():
            bad = sorted(set(out.filter(casted[period_col].is_null()
                                        & out[period_col].is_not_null())[period_col].to_list()))
            raise ValueError(
                f"inherit_anchor_period: casting period token(s) {bad} back to {d_dtype} gave "
                "nulls (the token is missing from the period vocabulary).")
        out = casted
    return out


def branch_periods_inherit_carry(frame: pl.DataFrame,
                                 pairs: "pl.DataFrame | Iterable[tuple[str, str]] | None", *,
                                 group_keys: "tuple[str, ...] | list[str]" = (),
                                 period_col: str = "d") -> pl.DataFrame:
    """A cross-solve carry ``frame`` (realized hours, purchases or emissions keyed by period) plus a copy of
    each period label's rows for its forecast branch periods.

    The realization solve writes every carry under the period label (the realized branch has the real
    period name); a forecast branch period of the same period has no row of its own, yet every forecast
    continues from its period's realized history (the realization solve's carry).  ``pairs`` holds
    ``(anchor, br)`` either as a frame or as tuples (``period__branch`` rows ``(period, branch)``); self rows
    are dropped.  The rule is
    :func:`inherit_anchor_period`'s: a branch period that has a row of its own keeps it.

    Apply it in each consumer that reads the carry per period, never once where the handoff is read:
    several consumers sum a carry over the periods (the cumulative ladder's realized purchases, the
    ``co2_max_total`` realized emissions, the fold of finished periods into ``f_d_k``), and a copy per
    forecast would be counted again in every such sum.
    """
    if pairs is not None and not isinstance(pairs, pl.DataFrame):
        rows = [(str(a), str(b)) for a, b in pairs]
        pairs = pl.DataFrame(rows, schema={"anchor": pl.Utf8, "br": pl.Utf8}, orient="row")
    return inherit_anchor_period(frame, pairs, group_keys=group_keys, period_col=period_col)


# The branch-anchor pairs of the load being built.  ``load_flextool`` activates them (from its own
# provider) for the duration of the load when it reads parameters from the DB, and resets them in
# its ``finally``; the DB-direct readers of period-keyed input apply them through
# :func:`inherit_active_branch_periods`.  Outside a load nothing is active and the readers return
# the authored periods only.  Mirrors ``_axis_enums.set_global_axis_enums``.
_ACTIVE_BRANCH_ANCHOR_PAIRS: "contextvars.ContextVar[pl.DataFrame | None]" = contextvars.ContextVar(
    "flextool_branch_anchor_pairs", default=None)


def solve_branch_anchor_pairs(provider: "object | None") -> pl.DataFrame | None:
    """The active solve's ``(anchor, br)`` pairs, restricted to its in-use periods; None when none.

    ``period_in_use`` is the first column of ``solve_data/period_in_use_set`` (provider-only, the
    form :func:`branch_anchor_pairs` uses); it excludes realized-designated and zero-weight branch
    periods, which have no LP variables.  ``period__branch`` with non-self rows but no
    ``period_in_use_set`` raises ``ValueError`` (a wiring bug: skipping would leave the branch
    periods without their anchor's values).  With the axis enums active, a branch period outside
    the live ``d`` vocabulary raises ``ValueError`` (it would otherwise become a null period and be
    broadcast as a default over every period).
    """
    pb = _provider_get(provider, Path("solve_data") / "period__branch.csv")
    if pb is None or pb.height == 0:
        return None
    if pb.filter(pl.col(pb.columns[0]).cast(pl.Utf8) != pl.col(pb.columns[1]).cast(pl.Utf8)).height == 0:
        return None
    piu = _provider_get(provider, Path("solve_data") / "period_in_use_set.csv")
    if piu is None or piu.height == 0:
        raise ValueError(
            "solve_branch_anchor_pairs: period__branch has stochastic branch periods but "
            "solve_data/period_in_use_set is missing or empty.")
    pairs = branch_anchor_pairs(provider, piu[piu.columns[0]].cast(pl.Utf8).drop_nulls().to_list())
    enums = get_global_axis_enums()
    if pairs is not None and enums is not None and enums.get("d") is not None:
        vocab = set(enums["d"].categories.to_list())
        bad = sorted(set(pairs["br"].to_list()) - vocab)
        if bad:
            raise ValueError(
                f"solve_branch_anchor_pairs: branch period(s) {bad} are not in the period (d) "
                "vocabulary of the axis enums.")
    return pairs


def activate_branch_anchor_pairs(pairs: "pl.DataFrame | None") -> "contextvars.Token":
    """Make ``pairs`` the active branch-anchor pairs; returns the token for the reset."""
    return _ACTIVE_BRANCH_ANCHOR_PAIRS.set(_validated_pairs(pairs, "activate_branch_anchor_pairs"))


def deactivate_branch_anchor_pairs(token: "contextvars.Token") -> None:
    """Restore the active pairs to what they were before the matching activation."""
    _ACTIVE_BRANCH_ANCHOR_PAIRS.reset(token)


def active_branch_anchor_pairs() -> pl.DataFrame | None:
    """The active branch-anchor pairs, or None outside a stochastic ``load_flextool``."""
    return _ACTIVE_BRANCH_ANCHOR_PAIRS.get()


def inherit_active_branch_periods(frame: pl.DataFrame, *,
                                  group_keys: "tuple[str, ...] | list[str]",
                                  period_col: str = "d") -> pl.DataFrame:
    """:func:`inherit_anchor_period` with the active pairs; ``frame`` itself when none are active.

    The one entry point for readers of period-keyed input during a load.
    """
    pairs = _ACTIVE_BRANCH_ANCHOR_PAIRS.get()
    if pairs is None:
        return frame
    return inherit_anchor_period(frame, pairs, group_keys=group_keys, period_col=period_col)


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
      CSV row order (period order).  ``ctx.period_branch`` first (its
      loader raises under axis Enums on a label outside the vocabulary,
      so it is null-free); without a ctx frame the raw
      provider/workdir CSV is read directly (NOT via
      :func:`period_branch_pairs_lf`, which de-duplicates the rows —
      ``pos`` below is derived from the (d, d) self-row order, which
      must stay the period order).
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
        if pb_ctx is not None and pb_ctx.height > 0:
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
    ``_derived.py:6634-6650``): a synthetic ``period_in_use``
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
) -> list[tuple[str, list[str]]]:
    """Build the solve's scenario leaves (design §2.2).

    Returns ``(label, members)`` pairs (F4, Slice E design §6.2): the
    real-named calendar-anchor leaf labeled ``"__anchor"`` (its members
    are exactly the ``anchors`` list — the real-named, calendar-canonical
    periods, owned by the realized branch when one is designated and by no
    branch otherwise; this is NOT a "realized" scenario, just the shared
    real-named chain) and each
    branch leaf labeled its time-branch id — so a caller that keys
    constraint rows off the leaf (``d_leaf``) carries the authoritative
    label rather than re-deriving and zipping it (a brittle
    two-independent-iteration alignment).

    Anchor leaf = all anchors (period order).  One leaf per time-branch
    possessing at least one synthetic ``period_in_use`` member (F3 —
    never iterate SBTB's own time-branches; a forecast branch with an
    empty synthetic-member set contributes no leaf).  Per branch
    leaf, the per-period representative follows the F6 authority rule:
    the branch-local member is authoritative; a real anchor joins the
    leaf only when it is strictly pre-reveal AND the branch has no
    member for that period (Slice E Option B: a shared pre-reveal trunk
    period thus belongs to EVERY branch leaf's path).
    """
    leaves: list[tuple[str, list[str]]] = [("__anchor", list(anchors))]
    branch_tbs = dict.fromkeys(tb_of[m] for m in synthetic)
    for br in branch_tbs:
        members = [m for m in synthetic if tb_of[m] == br]
        covered = dict.fromkeys(anchor_of[m] for m in members)
        k = min(pos[anchor_of[m]] for m in members)
        pre = [a for a in anchors if pos[a] < k and a not in covered]
        leaf = sorted(pre + members,
                      key=lambda x: pos[x] if x in pos
                      else pos[anchor_of[x]])
        leaves.append((br, leaf))
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
    for _label, leaf in leaves:
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


def dd_same_scenario_annuity_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """``dd_same_scenario`` for the NPV ANNUITY / fixed-cost window walks
    — the capacity frame :func:`dd_same_scenario_lf` with branch copies
    of the SAME calendar slot collapsed to one representative on the
    ``d_other`` side, per anchor ``d`` (Slice E design §4.1 vs §8.1
    reconciliation).

    The capacity lineage (``dd_same_scenario``, consumed by the ``edd``
    walks) must put a shared pre-reveal trunk period on EVERY branch
    leaf's path — e.g. ``(p2035, p2040_low)`` present — so the shared
    first-stage capacity is alive in every scenario's dispatch.  But the
    ANNUITY window sums an inflation factor over the leaf's periods: if
    the shared trunk's window kept BOTH ``p2040`` and ``p2040_low`` (the
    same calendar year 2040 under two mutually-exclusive scenarios) it
    would charge the first-stage capacity THREE period-annuities instead
    of two — the latent NPV over-count.  The capacity is physically alive
    for ONE second-stage calendar period; the scenario only affects
    dispatch, not the certain capacity payment.  So for the annuity we
    keep one member per CALENDAR anchor (preferring the real-named
    anchor, whose inflation factor is authoritative), giving the shared
    trunk the window ``{p2035, p2040}`` → Σ = 2.

    Byte-parity: deterministic / fan-at-first / single-period solves have
    no shared trunk — no anchor's leaf-union carries two members with the
    same calendar anchor — so the dedup is a no-op and the frame equals
    :func:`dd_same_scenario_lf`.  Deterministic periods that happen to
    share a calendar year are DISTINCT anchors, so they are never
    collapsed (the dedup keys on the anchor, not the year).
    """
    schema = {"d": schema_dtype(_enums, "d"),
              "d_other": schema_dtype(_enums, "d")}
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if not piu:
        return _empty_lf(schema)
    seen = _annuity_lineage_pairs(*_classify_lineage(pb_rows, piu, tb_of), tb_of)
    if not seen:
        return _empty_lf(schema)
    return _finalize_lineage_frame(seen, "d", "d", "d_other",
                                   "d_other", "dd_same_scenario_annuity")


def _annuity_lineage_pairs(
    anchors: list[str],
    pos: dict[str, int],
    synthetic: list[str],
    anchor_of: dict[str, str],
    tb_of: dict[str, str],
) -> list[tuple[str, str]]:
    """The ``(d, d_other)`` pairs of :func:`dd_same_scenario_annuity_lf`, from a :func:`_classify_lineage`
    result: every ``d_other`` on a lineage leaf through ``d``, one member per calendar anchor (the real-named
    anchor when it is reachable, otherwise the first-seen copy)."""
    leaves = _lineage_leaves(anchors, pos, synthetic, anchor_of, tb_of)
    # Calendar anchor of every in-use period (anchors map to themselves).
    calendar_anchor = {a: a for a in anchors}
    for m in synthetic:
        calendar_anchor[m] = anchor_of[m]
    # Gather every d_other reachable from d across ALL leaves (a shared
    # trunk sits on more than one leaf, so its reachable set unions the
    # branches — exactly the over-count source).
    reachable: dict[str, dict[str, None]] = {}
    for _label, leaf in leaves:
        for x in leaf:
            bucket = reachable.setdefault(x, {})
            for y in leaf:
                bucket[y] = None
    seen: dict[tuple[str, str], None] = {}
    for x, ys in reachable.items():
        best: dict[str, str] = {}
        for y in ys:
            a = calendar_anchor[y]
            # Prefer the real-named anchor member as the representative;
            # otherwise the first-seen member for that calendar slot.
            if a not in best or y == a:
                best[a] = y
        for y in best.values():
            seen[(x, y)] = None
    return list(seen)


def annuity_window_pairs(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    provider: "object | None" = None,
) -> list[tuple[str, str]] | None:
    """The ``(d, d_other)`` lineage of the investment-annuity, salvage and fixed-cost window sums, or None
    when the solve has no stochastic fan.

    Every one of those sums adds a per-period factor (inflation × the period's years) over the periods the
    capacity built in ``d`` serves.  In a fan, ``period_in_use`` holds each calendar period several times:
    the period label and one copy per forecast.  The sum must count each calendar period once along a
    lineage, whatever the invest method: under recourse the branch's own path; under ``none`` the period
    label's path, which is the bookkeeping walk over the calendar periods (the label stands for its
    forecasts, and every forecast copy shares the label's years).  These are exactly the pairs of
    :func:`dd_same_scenario_annuity_lf`.

    None (no filtering) when no ``period_in_use`` member is a forecast copy: deterministic and rolling
    solves, where every pair is on the one path and the filter would be a no-op.
    """
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, provider=provider)
    if not piu:
        return None
    classified = _classify_lineage(pb_rows, piu, tb_of)
    if not classified[2]:
        return None
    return _annuity_lineage_pairs(*classified, tb_of)


def annuity_window_lineage_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame | None:
    """:func:`annuity_window_pairs` as a typed ``(d, d_other)`` frame for the NPV walks; None without a fan."""
    pairs = annuity_window_pairs(workdir, source, active_solve, provider=provider)
    if not pairs:
        return None
    return _finalize_lineage_frame(pairs, "d", "d", "d_other", "d_other",
                                   "annuity_window_lineage").collect()


def dd_same_scenario_annuity_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Collect wrapper for :func:`dd_same_scenario_annuity_lf`."""
    return dd_same_scenario_annuity_lf(workdir, source, active_solve,
                                       ctx=ctx, provider=provider).collect()


def d_leaf_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> pl.LazyFrame:
    """``d_leaf`` — (d, leaf) scenario-path RELATION over ``period_in_use``.

    Without a fan every in-use period is on the one ``"__anchor"`` leaf.
    With a fan there is one leaf per forecast time-branch: its own branch
    periods, the shared pre-reveal periods and the period labels of its fan
    periods (period-level ancestors: no dispatch, a here-and-now investment
    under ``stochastic_invest_method = none``); no ``"__anchor"`` leaf.  Because the
    continuation fan starts at the solve's first step (§K), leaves
    PARTITION the invest axis — no period is on two leaves — so the map
    is well-defined and the per-leaf invest sums are disjoint.

    Consumed by the per-path total caps
    (``maxInvest/maxDivest_entity_total_path`` — ``model/core.py`` — and the
    divest ``maxDivestGroup_entity_total_path`` — ``_cumulative_invest.py``):
    the leaf column joins into each cap row's key so a total cap applies
    once per scenario path, never cross-scenario double-counting.

    Deterministic / flag-off solves degenerate to a single
    ``"__anchor"`` leaf (``period__branch`` silent → every in-use
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
    anchors, pos, synthetic, anchor_of = _classify_lineage(
        pb_rows, piu, tb_of)
    # Partition → RELATION (Slice E design §6.2).  Under Option B a shared
    # pre-reveal trunk anchor belongs to EVERY scenario leaf's path, so it
    # must count into every leaf's per-path total cap — the strict
    # partition (anchor → __anchor only) would under-count the branch
    # leaves' totals.  Rebuild from ``_lineage_leaves`` (the same relation
    # ``dd_same_scenario`` uses), carrying the authoritative leaf label
    # (F4).  Byte-parity: deterministic / fan-at-first solves have no
    # pre-reveal anchors, so each branch leaf gains nothing and the map is
    # identical to the legacy partition.
    labeled = _lineage_leaves(anchors, pos, synthetic, anchor_of, tb_of)
    if synthetic:
        # A fan's period labels have no dispatch, so the ``__anchor`` chain (pre-reveal periods + labels) is not
        # an operational path.  Each branch leaf holds the labels of its own fan periods as their period-level
        # ancestors instead: a here-and-now investment at a label (stochastic_invest_method = none) counts on
        # every branch path; under recourse the label has no investment column, so nothing is double counted.
        labels = fan_period_labels(set(pb_rows))
        joined: list[tuple[str, list[str]]] = []
        for label, members in labeled:
            if label == "__anchor":
                continue
            ancestors = [anchor_of[m] for m in members if m in anchor_of and anchor_of[m] in labels]
            joined.append((label, list(dict.fromkeys(
                sorted(members + [a for a in ancestors if a not in members],
                       key=lambda x: pos[x] if x in pos else pos[anchor_of[x]])))))
        labeled = joined
    rows: list[tuple[str, str]] = []
    for label, members in labeled:
        rows += [(d, label) for d in members]
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


def _read_invest_tie_periods(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> "frozenset[str] | None":
    """Resolve the v72 invest-NA window (the periods to tie) for this solve.

    Returns ``None`` when the window is unset (no
    ``pd_non_anticipativity_invest_periods.csv`` was emitted) — the
    byte-parity legacy path.  Returns a ``frozenset`` of period tokens
    otherwise (possibly empty, though the emitter only writes the CSV
    for a non-empty window).

    The per-solve CSV materialised by
    ``_per_solve.emit_per_solve_sets`` is authoritative and requires
    a ``provider``.  The ``ctx`` argument is accepted for signature parity
    with sibling readers but is deliberately NOT a source for this window:
    ``SolveConfig.non_anticipativity_invest_periods`` is a
    ``{solve: [...]}`` mapping, so reading it off ctx would silently
    iterate solve-name keys — the CSV avoids that footgun.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "pd_non_anticipativity_invest_periods.csv"
    df = _provider_get(provider, p)
    if df is None:
        return None
    if df.height == 0:
        return frozenset()
    col = "period" if "period" in df.columns else df.columns[0]
    return frozenset(str(v) for v in df[col].to_list() if v is not None)


def forecast_star_pairs(forecasts_by_anchor: "dict[str, list[str]]") -> "list[tuple[str, str]]":
    """The forecast star of each anchor period: ``(d_ref, b)`` for every other forecast ``b`` of the anchor.

    ``d_ref`` is the anchor's first sorted forecast (its reference forecast).  Ties join forecasts to each other,
    never to the period label (which has no dispatch and, under recourse, no investment), so they hold without a
    realized branch.  An anchor with fewer than two forecasts has nothing to tie.  Anchors in sorted order.
    """
    pairs: list[tuple[str, str]] = []
    for anchor in sorted(forecasts_by_anchor):
        forecasts = sorted(set(forecasts_by_anchor[anchor]))
        pairs.extend((forecasts[0], b) for b in forecasts[1:])
    return pairs


def pd_non_anticipativity_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
    branch_start: pl.DataFrame | None = None,
    tie_periods: "frozenset[str] | None" = None,
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

    The v72 invest-NA window drives the emit by MEMBERSHIP: ``tie_periods``
    is the resolved window (a ``frozenset`` of anchor periods).  When
    supplied, the synthetic members whose anchor lies in the window are
    tied in the forecast star of that anchor (:func:`forecast_star_pairs`:
    ``(d_ref, m)`` for every other member ``m``; members − 1 pairs) — the
    standard two-stage mode.  The anchor itself is a period label with no
    investment column under recourse, so nothing is tied to it.  When ``tie_periods`` is ``None`` it is auto-resolved
    from ctx / the per-solve CSV via :func:`_read_invest_tie_periods`;
    still ``None`` (unset) falls back to the legacy ``branch_start``
    reveal-boundary path below, which is provably EMPTY for every
    currently-constructible solve (byte-parity, design §2.3).

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
    if tie_periods is None:
        tie_periods = _read_invest_tie_periods(
            workdir, ctx=ctx, provider=provider)
    pb_rows, piu, tb_of = _lineage_inputs(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if not piu:
        return _empty_lf(schema)
    anchors, pos, synthetic, anchor_of = _classify_lineage(
        pb_rows, piu, tb_of)
    if not synthetic:
        return _empty_lf(schema)
    if tie_periods is not None:
        # v72 window-membership emit (design §3.2): tie the synthetic
        # members whose real-named anchor is in the window to each other,
        # in the forecast star of that anchor.  Supersedes the
        # reveal-boundary gate below (which cannot express an at-branching-
        # period tie).  An empty window -> no rows -> empty frame.
        by_anchor: dict[str, list[str]] = {}
        for m in synthetic:
            if anchor_of[m] in tie_periods:
                by_anchor.setdefault(anchor_of[m], []).append(m)
        rows = forecast_star_pairs(by_anchor)
        if not rows:
            return _empty_lf(schema)
        return _finalize_lineage_frame(rows, "d", "d", "b", "b",
                                       "pd_non_anticipativity")
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
    tie_periods: "frozenset[str] | None" = None,
) -> pl.DataFrame:
    """Collect wrapper for :func:`pd_non_anticipativity_lf` — always a
    typed frame, never ``None`` (design §2.5)."""
    return pd_non_anticipativity_lf(
        workdir, source, active_solve, ctx=ctx, provider=provider,
        branch_start=branch_start, tie_periods=tie_periods).collect()


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
    seed the Provider from the workdir (``provider_from_csv_dirs`` /
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
    (``_solve_writers.py`` years-represented / period-years
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
    """Return violation messages for the three recourse-NPV obligations;
    empty list == all hold, deterministic order.  Raising wrapper:
    :func:`assert_recourse_npv_preconditions`.  These obligations pin the
    arithmetic that makes a SET-shaped (unweighted) lineage filter
    numerically correct for trunk-crossing and realized-anchor windows, on
    the invest AND divest side.  Runtime wiring is the Slice D activation
    site; until then this checker is exercised by tests only.

    The fan is forecasts-only (Σ = 1): a trunk period that is both a
    self-row and the parent of ≥1 forecast is a PHANTOM carrying weight 0,
    and the forecast members renormalise among themselves.  Under that
    model:

      (i)  leaf-weight constancy — the probability-weight is constant along
           each post-reveal forecast chain (a weight-0 phantom / pre-reveal
           anchor is excluded, not a violation);
      (ii) cohort sum-to-1 — the forecast members of a revealing cohort sum
           to 1.  The producer (``pd_branch_weight_lf``) renormalises each
           first-step group to 1 and zeroes the phantom, so a well-formed
           cohort satisfies (ii) BY CONSTRUCTION; (ii) retains diagnostic
           power only against a malformed cohort straddling different
           first-step groups (see ``test_p3b_cohort_sum_violation``);
      (iii) byte-equal fan-member year rows.

    Obligation (iii) is checked against the emitted solve_data year frames
    (``p_years_d`` / ``period_with_history`` / ``p_years_represented``).
    The checker does NO disk I/O — it reads frames through the Provider, so
    the CALLER must seed the true emitted CSVs onto the Provider first.  The
    check underwrites the filter arithmetic only if the walker's year/factor
    source is those same CSVs (the Slice D year-threading prerequisite).

    Obligation (i)'s chains are defined against SBTB (which carries the
    realized branch's real-named members, so the trunk chain is covered).
    A strictly pre-reveal shared-trunk anchor is a certain first-stage
    decision carrying weight 1.0, so the obligation-(i) loop below excludes
    it from the constancy chains — mixing its 1.0 with a branch's scenario
    weight would be a false violation.  For fan-at-first-step / det solves
    there are no pre-reveal anchors, so the exclusion is a no-op and the
    check is byte-unchanged.  That excluded anchor's weight-1.0 is a
    cohort-of-one (verified by neither obligation) — an acceptable
    structural blind spot, not an unchecked assumption.
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
    # Member chains = PIU representatives grouped by SBTB time-branch.  A
    # shared pre-reveal trunk anchor is a cohort-of-one first-stage decision
    # (weight 1.0); the post-reveal members carry the scenario weight — the
    # correct two-stage weighting, not a violation.  Constancy is required
    # only over the POST-REVEAL segment, so pre-reveal anchors are excluded
    # (a no-op, byte-unchanged, for fan-at-first-step / deterministic solves).
    anchors_i, pos_i, synthetic_i, anchor_of_i = _classify_lineage(
        pb_rows, piu, tb_of)
    pre_reveal_anchors: set[str] = set()
    if synthetic_i:
        for br in dict.fromkeys(tb_of[m] for m in synthetic_i):
            br_members = [m for m in synthetic_i if tb_of[m] == br]
            k_reveal = min(pos_i[anchor_of_i[m]] for m in br_members)
            for a in anchors_i:
                if pos_i[a] < k_reveal:
                    pre_reveal_anchors.add(a)
    chains: dict[str, list[str]] = {}
    for m in piu:
        tb = tb_of.get(m)
        if tb is not None and m not in pre_reveal_anchors:
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
    # Forecasts-only (R2/R4): the weight-0 phantom anchor contributes 0 and
    # the forecast members renormalise to 1, so a well-formed cohort sums to
    # 1 BY CONSTRUCTION (a renormalisation tautology on the producer path).
    # This loop still catches a malformed cohort whose forecast members sit
    # in different first-step normalization groups (each → 1.0 → sum > 1).
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
        from flextool.engine.input._solve_state import LineageFilterError
        raise LineageFilterError(
            "recourse NPV preconditions violated:\n  "
            + "\n  ".join(violations)
        )


# ---------------------------------------------------------------------------
# apply_branch_cluster — single-pass entry for the apply_derived_g
# integration.  Mutates ``flex_data`` in place.
# ---------------------------------------------------------------------------


def apply_lineage_frames(
    flex_data: object,
    workdir: Path,
    source: "InputSource | None",
    active_solve: str | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> None:
    """Assign the scenario-lineage frames ``dd_same_scenario``,
    ``pd_non_anticipativity`` and ``d_leaf`` to ``flex_data``.

    Shared by single solves (:func:`apply_branch_cluster`) and by the
    synthetic sub-solves (rolls, ``<base>_<anchor>`` sub-solves) that skip
    the per-solve override passes, so every solve of a fan carries its
    scenario paths.  The builders read only ``ctx`` and the solve's own
    workdir CSVs (``period__branch``, ``period_in_use_set``,
    ``solve_branch__time_branch``), so they are solve-agnostic.
    Unconditional assignment: the fields have no CSV seed to preserve
    (the builders always return a typed frame, never None); a solve
    without a fan gets degenerate content (``d_leaf`` = one ``__anchor``
    leaf, so the per-path families keep their legacy rows).
    """
    flex_data.dd_same_scenario = dd_same_scenario_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    flex_data.pd_non_anticipativity = pd_non_anticipativity_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    # The (d, leaf) partition that keeps the total caps per scenario path.
    flex_data.d_leaf = d_leaf_df(
        workdir, source, active_solve, ctx=ctx, provider=provider)


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
      2b. ``dd_same_scenario`` / ``pd_non_anticipativity`` / ``d_leaf`` —
         the scenario-lineage frames (:func:`apply_lineage_frames`).
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

    # 2b. Scenario-lineage frames.
    apply_lineage_frames(flex_data, workdir, source, active_solve,
                         ctx=ctx, provider=provider)

    # 3-4. Branch-weight Params (lazy ports of the previous eager
    # helpers in ``_derived.py``).
    _new_pd_bw = pd_branch_weight_param(
        workdir, source, active_solve, ctx=ctx, provider=provider)
    if _new_pd_bw is not None or getattr(flex_data, "pd_branch_weight", None) is None:
        flex_data.pd_branch_weight = _new_pd_bw

    pdt_bw = pdt_branch_weight_param(workdir, source, active_solve, dt,
                                          ctx=ctx, provider=provider)
    if pdt_bw is not None and dt is not None and dt.height > 0:
        # Match flex_data.py's dense-dt semantics: when dt is supplied,
        # build a (d, t)-dense Param via left-join + coalesce default
        # of 1.0.  This mirrors the previous CSV-cascade behaviour at
        # flex_data.py:2845-2870 (preserves exact frame shape).
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
    "dd_same_scenario_annuity_lf",
    "dd_same_scenario_annuity_df",
    "annuity_window_pairs",
    "annuity_window_lineage_df",
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
