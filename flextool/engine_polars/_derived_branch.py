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

All helpers are lazy ``pl.LazyFrame`` chains; the public
``apply_branch_cluster`` entry collects once per emitted Param.

Algorithm reference: flextool's
``preprocessing/period_calculated_params.py:write_branch_weights:364-451``
and ``preprocessing/per_solve_sets.py:96-101, 267-276``.

R-O6 invariant (per ``audit/a6_b_dim_alternative.md``): branches stay
realised-only for invest.  This module does NOT introduce per-branch
``v_invest`` variables; it only emits the *operational* probability
weights and the non-anticipativity gate that pin storage / online /
reserve dispatch across siblings.

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

from polar_high_opt import Param

from ._input_source import _read_csv_file

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# Workdir CSV readers — lazy frames
# ---------------------------------------------------------------------------


def _empty_lf(schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Return an empty :class:`pl.LazyFrame` with the given schema."""
    return pl.DataFrame(schema=schema).lazy()


def _maybe_csv_lf(path: Path,
                  rename: dict[str, str] | None = None,
                  schema: dict[str, pl.DataType] | None = None,
                  ) -> pl.LazyFrame | None:
    """Return ``_read_csv_file(path).lazy()`` rename-and-cast helper.

    Returns ``None`` when the file is absent or empty.
    """
    if not path.exists():
        return None
    df = _read_csv_file(path)
    if df.height == 0:
        return None
    lf = df.lazy()
    if rename:
        # Only rename columns that actually exist (defensive against
        # column-name drift across fixture vintages).
        cols = df.columns
        applied = {k: v for k, v in rename.items() if k in cols}
        if applied:
            lf = lf.rename(applied)
    return lf


def period_branch_pairs_lf(workdir: Path | None) -> pl.LazyFrame:
    """Read ``solve_data/period__branch.csv`` as a lazy ``(d, b)`` frame.

    Returns an empty frame (with the correct schema) when the file is
    absent / empty — non-stochastic fixtures' "deterministic" path.

    The ``b`` column is the *sibling branch* name (in non-stochastic
    fixtures, ``d == b`` for every row).
    """
    schema = {"d": pl.Utf8, "b": pl.Utf8}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "period__branch.csv"
    lf = _maybe_csv_lf(p, rename={"period": "d", "branch": "b"})
    if lf is None:
        return _empty_lf(schema)
    return lf.select(
        pl.col("d").cast(pl.Utf8, strict=False),
        pl.col("b").cast(pl.Utf8, strict=False),
    ).unique()


def solve_branch_weights_lf(workdir: Path | None) -> pl.LazyFrame:
    """Read ``solve_data/solve_branch_weight.csv`` as ``(b, w)`` lazy.

    The flextool CSV header is either ``branch,p_branch_weight_input``
    (stochastic fixtures) or ``branch,value`` (deterministic).  We
    normalise to ``(b, w)``.

    Defaults to the empty frame (schema only) when the file is absent.
    """
    schema = {"b": pl.Utf8, "w": pl.Float64}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "solve_branch_weight.csv"
    if not p.exists():
        return _empty_lf(schema)
    df = _read_csv_file(p)
    if df.height == 0:
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
              .select(pl.col(b_col).cast(pl.Utf8, strict=False).alias("b"),
                      pl.col(v_col).cast(pl.Float64, strict=False).alias("w")))


def first_timesteps_lf(workdir: Path | None) -> pl.LazyFrame:
    """Read ``solve_data/first_timesteps.csv`` as ``(d, ts)`` lazy.

    Maps each period (or branch period) to its first timestep — the
    discriminator the ``pd_branch_weight`` algorithm uses for grouping
    sibling branches.
    """
    schema = {"d": pl.Utf8, "ts": pl.Utf8}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "first_timesteps.csv"
    if not p.exists():
        return _empty_lf(schema)
    df = _read_csv_file(p)
    if df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    s_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(pl.col(d_col).cast(pl.Utf8, strict=False).alias("d"),
                      pl.col(s_col).cast(pl.Utf8, strict=False).alias("ts")))


def period_in_use_set_lf(workdir: Path | None,
                          source: "InputSource | None" = None,
                          active_solve: str | None = None,
                          ) -> pl.LazyFrame:
    """Read ``solve_data/period_in_use_set.csv`` as ``(d,)`` lazy.

    When the workdir CSV is absent (single-solve / no chain runner),
    fall back to ``realized_periods ∪ invest_periods`` from the
    ``solve`` parameters.  This is sufficient for non-stochastic
    fixtures.

    The chain-runner-resident scaffolding will surface this set
    in-memory at the ``SolveContext`` boundary in Δ.9+.
    """
    schema = {"d": pl.Utf8}
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if p.exists():
            df = _read_csv_file(p)
            if df.height > 0 and df.columns:
                col = df.columns[0]
                return (df.lazy()
                          .select(pl.col(col).cast(pl.Utf8, strict=False)
                                                .alias("d"))
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
                       .select(pl.col("value").cast(pl.Utf8, strict=False)
                                                  .alias("d")))
    if not parts:
        return _empty_lf(schema)
    return pl.concat(parts).unique()


def realized_dispatch_lf(workdir: Path | None) -> pl.LazyFrame:
    """Read ``solve_data/realized_dispatch.csv`` as ``(d, t)`` lazy."""
    schema = {"d": pl.Utf8, "t": pl.Utf8}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "realized_dispatch.csv"
    if not p.exists():
        return _empty_lf(schema)
    df = _read_csv_file(p)
    if df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    t_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(pl.col(d_col).cast(pl.Utf8, strict=False).alias("d"),
                      pl.col(t_col).cast(pl.Utf8, strict=False).alias("t"))
              .unique())


def fix_storage_timesteps_lf(workdir: Path | None) -> pl.LazyFrame:
    """Read ``solve_data/fix_storage_timesteps.csv`` as ``(d, t)`` lazy."""
    schema = {"d": pl.Utf8, "t": pl.Utf8}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "fix_storage_timesteps.csv"
    if not p.exists():
        return _empty_lf(schema)
    df = _read_csv_file(p)
    if df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    t_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(pl.col(d_col).cast(pl.Utf8, strict=False).alias("d"),
                      pl.col(t_col).cast(pl.Utf8, strict=False).alias("t"))
              .unique())


def steps_in_use_lf(workdir: Path | None) -> pl.LazyFrame:
    """Read ``solve_data/steps_in_use.csv`` as ``(d, t)`` lazy."""
    schema = {"d": pl.Utf8, "t": pl.Utf8}
    if workdir is None:
        return _empty_lf(schema)
    p = Path(workdir) / "solve_data" / "steps_in_use.csv"
    if not p.exists():
        return _empty_lf(schema)
    df = _read_csv_file(p)
    if df.height == 0:
        return _empty_lf(schema)
    cols = df.columns
    d_col = "period" if "period" in cols else cols[0]
    t_col = ("step" if "step" in cols else
             ("time" if "time" in cols else cols[1]))
    return (df.lazy()
              .select(pl.col(d_col).cast(pl.Utf8, strict=False).alias("d"),
                      pl.col(t_col).cast(pl.Utf8, strict=False).alias("t"))
              .unique())


# ---------------------------------------------------------------------------
# pd_branch_weight / pdt_branch_weight — lazy normalisation
# ---------------------------------------------------------------------------


def pd_branch_weight_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
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
    pb = period_branch_pairs_lf(workdir).collect()
    piu = period_in_use_set_lf(workdir, source, active_solve).collect()
    if piu.height == 0:
        return _empty_lf({"d": pl.Utf8, "value": pl.Float64})
    if pb.height == 0:
        # Deterministic fallback: 1.0 per realised period.
        return (piu.lazy()
                  .select(pl.col("d"))
                  .with_columns(value=pl.lit(1.0))
                  .sort("d"))
    bw = solve_branch_weights_lf(workdir).collect()
    ft = first_timesteps_lf(workdir).collect()
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
    out = pl.DataFrame(rows, schema={"d": pl.Utf8, "value": pl.Float64},
                       orient="row").sort("d")
    return out.lazy()


def pdt_branch_weight_lf(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    dt: pl.DataFrame | None = None,
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
    schema = {"d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64}
    if dt is None or dt.height == 0:
        # Fall back to steps_in_use as the dense domain.
        dt_lf = steps_in_use_lf(workdir).collect()
        if dt_lf.height == 0:
            return _empty_lf(schema)
        dt_pairs_df = dt_lf
    else:
        dt_pairs_df = dt.select("d", "t").unique()
    pb = period_branch_pairs_lf(workdir).collect()
    if pb.height == 0:
        # Default 1.0 per (d, t).
        return (dt_pairs_df.lazy()
                  .select(pl.col("d").cast(pl.Utf8, strict=False),
                          pl.col("t").cast(pl.Utf8, strict=False))
                  .with_columns(value=pl.lit(1.0))
                  .sort("d", "t"))
    bw = solve_branch_weights_lf(workdir).collect()
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


def dt_non_anticipativity_lf(workdir: Path | None) -> pl.LazyFrame:
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
    schema = {"d": pl.Utf8, "t": pl.Utf8}
    rd = realized_dispatch_lf(workdir).collect()
    fs = fix_storage_timesteps_lf(workdir).collect()
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


def period_branch_full_lf(workdir: Path | None) -> pl.LazyFrame:
    """The unfiltered ``period__branch.csv`` as ``(d, b)`` lazy.

    Distinct from the existing ``period_branch`` rolling-handoff field
    (which renames columns to ``d_upper`` / ``d``).  This is the raw
    anchor → sibling map consumed by the model layer's
    non-anticipativity constraints.
    """
    return period_branch_pairs_lf(workdir).select("d", "b")


# ---------------------------------------------------------------------------
# Public Param helpers (collect-at-boundary)
# ---------------------------------------------------------------------------


def pd_branch_weight_param(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
) -> Param | None:
    """Public ``pd_branch_weight`` :class:`Param` builder.

    Returns ``None`` when no realised periods are available (matches
    the ``apply_derived_g`` skip-on-None contract).
    """
    df = pd_branch_weight_lf(workdir, source, active_solve).collect()
    if df.height == 0:
        return None
    return Param(("d",), df)


def pdt_branch_weight_param(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
    dt: pl.DataFrame | None = None,
) -> Param | None:
    """Public ``pdt_branch_weight`` :class:`Param` builder."""
    df = pdt_branch_weight_lf(workdir, source, active_solve, dt).collect()
    if df.height == 0:
        return None
    return Param(("d", "t"), df)


def dt_non_anticipativity_df(workdir: Path | None) -> pl.DataFrame | None:
    """Public ``dt_non_anticipativity`` plain-DataFrame builder."""
    df = dt_non_anticipativity_lf(workdir).collect()
    if df.height == 0:
        return None
    return df


def period_branch_full_df(workdir: Path | None) -> pl.DataFrame | None:
    """Public ``period_branch_full`` plain-DataFrame builder."""
    df = period_branch_full_lf(workdir).collect()
    if df.height == 0:
        return None
    return df


def period_in_use_set_df(
    workdir: Path | None,
    source: "InputSource | None" = None,
    active_solve: str | None = None,
) -> pl.DataFrame | None:
    """Public ``period_in_use_set`` plain-DataFrame builder."""
    df = period_in_use_set_lf(workdir, source, active_solve).collect()
    if df.height == 0:
        return None
    return df


# ---------------------------------------------------------------------------
# apply_branch_cluster — single-pass entry for the apply_derived_g
# integration.  Mutates ``flex_data`` in place.
# ---------------------------------------------------------------------------


def apply_branch_cluster(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    active_solve: str | None = None,
) -> None:
    """Apply Cluster D Params to ``flex_data``.

    Order:
      1. ``period_branch_full`` (no dependencies; trivial CSV unroll).
      2. ``period_in_use_set`` (depends on workdir or source).
      3. ``dt_non_anticipativity`` (depends on realized + fix-storage CSVs).
      4. ``pd_branch_weight`` (depends on period_branch_full + first_ts +
         period_in_use_set + branch weights).
      5. ``pdt_branch_weight`` (depends on dt + period_branch_full +
         branch weights).

    Honors the "skip on None" contract (no field is overwritten with
    ``None``).  Per the architectural pattern: the lazy port co-exists
    with the eager ``input.py`` CSV reader for now; ``input.py`` runs
    first and sets the fields from CSV, then this helper overrides
    when the lazy path produces a value.

    R-O6 invariant: this helper does NOT touch ``invest_periods`` or
    ``v_invest`` — only the operational dispatch-side weights and
    the non-anticipativity gate.
    """
    dt = getattr(flex_data, "dt", None)

    # 1-2. Set frames (may already be loaded by input.py; only override
    # when we have a non-empty result).
    pbf = period_branch_full_df(workdir)
    if pbf is not None:
        flex_data.period_branch_full = pbf

    piu = period_in_use_set_df(workdir, source, active_solve)
    if piu is not None:
        flex_data.period_in_use_set = piu

    dtna = dt_non_anticipativity_df(workdir)
    if dtna is not None:
        flex_data.dt_non_anticipativity = dtna

    # 3-4. Branch-weight Params (lazy ports of the previous eager
    # helpers in ``_derived_params.py``).
    pd_bw = pd_branch_weight_param(workdir, source, active_solve)
    if pd_bw is not None:
        flex_data.pd_branch_weight = pd_bw

    pdt_bw = pdt_branch_weight_param(workdir, source, active_solve, dt)
    if pdt_bw is not None:
        # Match input.py's dense-dt semantics: when dt is supplied,
        # build a (d, t)-dense Param via left-join + coalesce default
        # of 1.0.  This mirrors the previous CSV-cascade behaviour at
        # input.py:2845-2870 (preserves exact frame shape).
        if dt is not None and dt.height > 0:
            base = (dt.lazy()
                      .with_columns(value=pl.lit(1.0))
                      .select("d", "t", "value"))
            joined = (base
                        .join(pdt_bw.lazy,
                              on=["d", "t"], how="left", suffix="__r")
                        .with_columns(value=pl.coalesce(
                            pl.col("value__r"), pl.col("value")))
                        .select("d", "t", "value")
                        .collect())
            pdt_bw = Param(("d", "t"), joined)
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
    "pd_branch_weight_param",
    "pdt_branch_weight_param",
    "dt_non_anticipativity_df",
    "period_branch_full_df",
    "period_in_use_set_df",
    "apply_branch_cluster",
]
