"""Cluster C — profile cascade (Δ.7).

Lazy-polars port of flextool's
``preprocessing/entity_period_calc_params.py::write_pdtProfile``
algorithm — the 5-branch fallback that resolves
``profile.profile`` parameter values to per-(profile, d, t) rows.

Cluster C field (per
``audit/native_data_path_design_derived_clusters.md``):

* ``p_profile_value`` — per-(f, d, t) profile value, resolved across a
  multi-tier cascade.

Algorithm (mirror of ``write_pdtProfile`` at
``entity_period_calc_params.py:767-947``)
-----------------------------------------------------------------

For each (profile p, dispatch period d, timestep t) ∈ profile × dt the
output value is picked by the first matching branch:

    1. **Stochastic UNION fold** — when ``p`` is referenced by a
       stochastic relationship class (``unit__node__profile``,
       ``connection__profile``, ``node__profile`` whose entity is in a
       ``include_stochastics`` group), sum
       ``Σ_{tb,ts} pbt_profile[p, tb, ts, t]`` over
       ``(tb, ts) ∈ tb_for_d[d] × ts_for_d[d]``.  Skip if no hit.

    2. **Parent-period fold** — for each parent ``pe`` of ``d``
       (``period__branch[d_anchor=pe, branch=d]``), sum
       ``Σ_{tb,ts} pbt_profile[p, tb, ts, t]`` over
       ``(tb, ts) ∈ tb_for_d[pe] × ts_for_d[d]``.  Skip if no hit.

    3. **Time series** — ``pt_profile[(p, t)]`` (``profile.profile`` of
       runtime type ``Map(time → float)`` or ``time_series``).

    4. **Scalar** — ``p_profile[p]`` (``profile.profile`` of runtime type
       ``float`` / ``str`` / ``bool``).

    5. **Zero** — fallback.

Branches 1 and 2 require the per-solve scaffolding ``period__branch``,
``solve_branch__time_branch``, ``first_timesteps``, and
``groupIncludeStochastics`` / membership tables.  Δ.7 keeps these in the
workdir-CSV layer for the stochastic path; the chain-runner inputs are
preserved.  The deterministic path (branches 3-5) is fully native.

Architectural notes
-------------------

* **Drop the ``_check_canonical_keys`` predicate.**  The Δ.6 close
  stanza identified ``_derived_params.py:p_profile_value_from_source``
  as guarding against non-canonical Map keys (``x``/``i`` from
  zero-``index_name`` Spine values) by returning ``None`` and falling
  back to the CSV path.  Cluster C drops the predicate: the stochastic
  3d_map case is now handled explicitly via the per-solve CSV, the
  deterministic cases are handled via per-row dispatch on the raw
  parameter value's runtime type rather than column-name sniffing.
* **Lazy polars throughout.**  Every helper returns a
  :class:`pl.LazyFrame`; the public ``apply_profile_cascade`` boundary
  collects once.
* **Per-solve workdir CSVs for tiers 1 and 2.**  Mirroring Cluster B's
  ``edd_history`` / ``ppec_handoff`` boundary (Δ.6), the stochastic
  scaffolding lives in ``solve_data/*.csv`` until the chain runner
  surfaces it in-memory (Δ.8+).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from ._writer_provider_io import _provider_key
from ._axis_enums import alias_to_axis


def _provider_has_key(provider, path: "Path") -> bool:
    """Provider-only existence check used by the profile cascade.
    Returns ``False`` when *provider* is ``None`` or the key is
    missing.  The disk-fallback arm was removed in Step 2.5.
    """
    if provider is None:
        return False
    return provider.has(_provider_key(path))


def _provider_read(provider, path: "Path") -> "pl.DataFrame":
    """Provider-only read used by the profile cascade.  Guard with
    :func:`_provider_has_key` first; calling without a provider
    raises :class:`ValueError`.
    """
    if provider is None:
        raise ValueError(
            "_provider_read requires a FlexDataProvider; guard with "
            "_provider_has_key first."
        )
    return provider.get(_provider_key(path))

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# Per-entity profile classification: scalar / period / time / branch+t
# ---------------------------------------------------------------------------


def _classify_profile_rows(source: "InputSource") -> dict[str, str]:
    """Return ``{profile_name: tier}`` where tier ∈ {``scalar``,
    ``period``, ``time``, ``stochastic``}.

    Reads ``profile.profile`` raw values from the source.  Mirrors
    flextool's input-writer dispatch
    (``flextoolrunner/input_writer.py:399-424``):

    * ``float`` / ``str`` / ``bool`` → tier ``scalar`` (``p_profile.csv``).
    * Map with ``index_name=='time'`` → tier ``time`` (``pt_profile.csv``).
    * Map with ``index_name=='period'`` → tier ``period``
      (``pd_profile.csv`` — note: not consumed by ``write_pdtProfile``).
    * 3-deep Map → tier ``stochastic`` (``pbt_profile.csv``).
    * Other Map shapes (deterministic 1d_map, 2d_map(period, t)) →
      tier ``period`` or ``time`` per outermost ``index_name``.

    Profiles absent from the parameter table are not in the returned
    dict.  The caller treats them as tier 5 (zero fallback).

    Implementation strategy: peek at the SpineDbReader's per-row raw
    value via ``_param_rows`` when available; fall back to column-name
    heuristics on the CSV-shaped frame for sources that don't expose
    raw parameter values (``InMemoryReader``).
    """
    # Try to access the typed raw values via SpineDbReader's internals.
    rows_by_name: dict[str, object] = {}
    if hasattr(source, "_class_name_to_id") and hasattr(source, "_param_rows"):
        cls_id = source._class_name_to_id.get("profile")  # type: ignore[attr-defined]
        if cls_id is not None:
            pdef = source._pdef_by_class_name.get((cls_id, "profile"))  # type: ignore[attr-defined]
            if pdef is not None:
                raw_rows = source._param_rows.get((cls_id, pdef["id"]), [])  # type: ignore[attr-defined]
                for eid, v in raw_rows:
                    name = source._entity_by_id[eid][1]  # type: ignore[attr-defined]
                    rows_by_name[name] = v
    if rows_by_name:
        return {n: _tier_for_value(v) for n, v in rows_by_name.items()}
    # Column-shape fallback for InMemoryReader (used in unit tests).
    try:
        df = source.parameter("profile", "profile")
    except KeyError:
        return {}
    if df.height == 0:
        return {}
    cols = set(df.columns)
    out: dict[str, str] = {}
    for name in df["name"].unique().to_list():
        if cols == {"name", "value"}:
            out[name] = "scalar"
        elif cols == {"name", "period", "value"}:
            out[name] = "period"
        elif cols == {"name", "t", "value"}:
            out[name] = "time"
        elif cols == {"name", "period", "t", "value"}:
            # 2d_map(period, t) — fold as time-axis per period.  Treat
            # as a multi-row variant of "time" here; the resolver knows.
            out[name] = "period_time"
        else:
            # Generic / stochastic 3d_map — defer to workdir-CSV path.
            out[name] = "stochastic"
    return out


def _tier_for_value(v: object) -> str:
    """Classify a single raw Spine parameter value.

    Mirror of the input-writer dispatch logic.
    """
    from spinedb_api.parameter_value import (
        Map, TimeSeries, Array,
    )
    if not isinstance(v, (Map, TimeSeries, Array)):
        return "scalar"
    if isinstance(v, TimeSeries):
        return "time"
    if isinstance(v, Map):
        # Walk to find depth + outer index name.
        depth = 0
        cur: object = v
        while isinstance(cur, Map):
            depth += 1
            if not cur.values:
                break
            cur = cur.values[0]
        outer_idx = (v.index_name or "").lower()
        if depth == 1:
            if outer_idx == "period":
                return "period"
            if outer_idx == "time":
                return "time"
            # Empty or generic index — assume time-axis (the typical
            # flextool 1d_map default).
            return "time"
        if depth == 2:
            if outer_idx == "period":
                return "period_time"
            # 2d_map without "period" as the outer key — treat as
            # stochastic (defer to CSV) for safety.
            return "stochastic"
        # Deeper than 2 → 3d_map (branch × time_start × t).
        return "stochastic"
    # Array — uncommon for profiles; treat as time-axis.
    return "time"


# ---------------------------------------------------------------------------
# Tier-3 / tier-4 / tier-5 builders (deterministic)
# ---------------------------------------------------------------------------


def _profile_time_lf(source: "InputSource",
                          dt_lf: pl.LazyFrame,
                          time_profiles: list[str],
                          workdir: Path | None = None,
                          *,
                          provider: "object | None" = None,
                          ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for time-axis profiles.

    Tier 3 of the cascade: profile values keyed by ``t`` only,
    broadcast over the dispatch periods of dt.

    Prefers ``workdir/solve_data/pt_profile.csv`` (per-solve averaged
    values written by ``TimelineConfig.create_averaged_timeseries``)
    when present — those values are aggregated to the solve's
    ``new_stepduration``, matching the dispatch timeline.  Falls back
    to the raw Spine ``profile.profile`` parameter when the CSV is
    absent (e.g. tests that bypass the runner).

    The Spine fallback path: the source column may be ``t`` (when
    ``index_name='time'``) or ``i`` (default index name).  Either way
    the column maps 1:1 to t.  Mirrors how ``pdtNodeInflow`` already
    sources averaged values from ``solve_data/pt_node_inflow.csv``.
    """
    empty_schema = {
        "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
    }
    if not time_profiles:
        return pl.LazyFrame(schema=empty_schema)

    # Prefer per-solve averaged CSV when the runner has produced it.
    if workdir is not None:
        csv_path = Path(workdir) / "solve_data" / "pt_profile.csv"
        if _provider_has_key(provider, csv_path):
            try:
                csv_df = _provider_read(provider, csv_path)
                csv_lf = (csv_df.lazy()
                          .filter(pl.col("profile").is_in(time_profiles))
                          .select(
                              alias_to_axis(
                                  pl.col("profile").cast(pl.Utf8), "f"),
                              alias_to_axis(pl.col("time").cast(pl.Utf8), "t"),
                              pl.col("pt_profile")
                                  .cast(pl.Float64, strict=False)
                                  .alias("value"),
                          ))
                return (csv_lf
                        .join(dt_lf, on="t", how="inner")
                        .select("f", "d", "t", "value"))
            except Exception:
                # Malformed CSV — fall through to Spine source.
                pass

    try:
        raw = source.parameter("profile", "profile")
    except KeyError:
        return pl.LazyFrame(schema=empty_schema)
    if raw.height == 0:
        return pl.LazyFrame(schema=empty_schema)
    cols = raw.columns
    # Find the time-axis column.  Prefer 't' (canonical) over 'i'
    # (generic Map fallback) when both are present.
    if "t" in cols:
        t_col = "t"
    elif "i" in cols:
        t_col = "i"
    elif "time" in cols:
        t_col = "time"
    else:
        return pl.LazyFrame(schema=empty_schema)
    raw_lf = (raw.lazy()
                  .filter(pl.col("name").is_in(time_profiles))
                  .select(alias_to_axis("name", "f"),
                          alias_to_axis(pl.col(t_col).cast(pl.Utf8), "t"),
                          pl.col("value").cast(pl.Float64, strict=False)))
    return (raw_lf
              .join(dt_lf, on="t", how="inner")
              .select("f", "d", "t", "value"))


def _profile_period_time_lf(source: "InputSource",
                                  dt_lf: pl.LazyFrame,
                                  pt_profiles: list[str],
                                  ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for 2d_map(period, t) profiles.

    The (d, t) keys are direct; only inner-join with dt_lf for
    membership.
    """
    if not pt_profiles:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    try:
        raw = source.parameter("profile", "profile")
    except KeyError:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    if raw.height == 0:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    cols = raw.columns
    if "period" not in cols:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    if "t" in cols:
        t_col = "t"
    elif "i" in cols:
        t_col = "i"
    else:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    raw_lf = (raw.lazy()
                  .filter(pl.col("name").is_in(pt_profiles))
                  .select(alias_to_axis("name", "f"),
                          alias_to_axis(pl.col("period").cast(pl.Utf8), "d"),
                          alias_to_axis(pl.col(t_col).cast(pl.Utf8), "t"),
                          pl.col("value").cast(pl.Float64, strict=False)))
    return raw_lf.join(dt_lf, on=["d", "t"], how="inner")


def _profile_period_only_lf(source: "InputSource",
                                  dt_lf: pl.LazyFrame,
                                  period_profiles: list[str],
                                  ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for 1d_map(period) profiles.

    Note: flextool's ``write_pdtProfile`` does **not** consult
    ``pd_profile.csv``; the only deterministic period-keyed branch is
    via the parent-period fold of ``pbt_profile`` (tier 2).  This
    helper exists for completeness and is wired in only when the
    fixture explicitly populates a 1d_map_period profile (rare /
    historical).  Mirror behaviour: broadcast across t of the matching
    period.
    """
    if not period_profiles:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    try:
        raw = source.parameter("profile", "profile")
    except KeyError:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    if raw.height == 0:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    cols = raw.columns
    if "period" not in cols:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    raw_lf = (raw.lazy()
                  .filter(pl.col("name").is_in(period_profiles))
                  .select(alias_to_axis("name", "f"),
                          alias_to_axis(pl.col("period").cast(pl.Utf8), "d"),
                          pl.col("value").cast(pl.Float64, strict=False)))
    return (raw_lf
              .join(dt_lf, on="d", how="inner")
              .select("f", "d", "t", "value"))


def _profile_scalar_lf(source: "InputSource",
                            dt_lf: pl.LazyFrame,
                            scalar_profiles: list[str],
                            ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for scalar profiles.

    Tier 4 of the cascade: scalar value broadcast over the full dt grid.
    """
    if not scalar_profiles:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    try:
        raw = source.parameter("profile", "profile")
    except KeyError:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    if raw.height == 0:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    raw_lf = (raw.lazy()
                  .filter(pl.col("name").is_in(scalar_profiles))
                  .select(alias_to_axis("name", "f"),
                          pl.col("value").cast(pl.Float64, strict=False))
                  .unique(subset=["f"]))
    return (raw_lf
              .join(dt_lf, how="cross")
              .select("f", "d", "t", "value"))


# ---------------------------------------------------------------------------
# Tier-1 / tier-2 builders (stochastic — workdir-CSV-fed)
# ---------------------------------------------------------------------------


def _profile_stochastic_lf(workdir: Path | None,
                                stoch_profiles: list[str],
                                *,
                                provider: "object | None" = None,
                                ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for stochastic 3d_map profiles.

    Branches 1 and 2 of flextool's ``write_pdtProfile`` cascade.

    Per-solve scaffolding consumed (workdir/solve_data/):

    * ``period__branch.csv`` — (anchor, branch) tuples; defines parents.
    * ``solve_branch__time_branch.csv`` — (anchor, time_branch); the
      stochastic-fold's outer-time-branch axis.
    * ``first_timesteps.csv`` — (anchor, time_start); the
      stochastic-fold's outer-time-start axis.
    * ``input/pbt_profile.csv`` — (profile, branch, time_start, time,
      value) — the 3d_map values themselves.
    * ``input/groupIncludeStochastics.csv`` + relationship CSVs — gate
      Branch 1 (per-profile is_stoch flag).

    Output: lazy frame ``[f, d, t, value]`` covering only the
    (profile, d, t) tuples that actually hit Branch 1 or Branch 2.
    The caller layers tiers 3-5 over the gaps.
    """
    if workdir is None or not stoch_profiles:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    workdir = Path(workdir)
    inp = workdir / "input"
    sd = workdir / "solve_data"

    # Read pbt_profile.csv: (profile, branch, time_start, time, pbt_profile).
    pbt_path = inp / "pbt_profile.csv"
    if not _provider_has_key(provider, pbt_path):
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    try:
        pbt = _provider_read(provider, pbt_path)
    except Exception:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    if pbt.height == 0:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    pbt = pbt.rename({"pbt_profile": "value"})
    pbt = pbt.filter(pl.col("profile").is_in(stoch_profiles))
    pbt_lf = pbt.lazy().select(
        alias_to_axis(pl.col("profile").cast(pl.Utf8), "f"),
        pl.col("branch").cast(pl.Utf8).alias("tb"),
        pl.col("time_start").cast(pl.Utf8).alias("ts"),
        alias_to_axis(pl.col("time").cast(pl.Utf8), "t"),
        pl.col("value").cast(pl.Float64, strict=False),
    )

    # Read per-solve scaffolding.
    pe_for_d = _read_pairs(sd / "period__branch.csv", "branch", "period",
                             provider=provider)
    tb_for_d = _read_pairs(sd / "solve_branch__time_branch.csv",
                                    "period", "branch", provider=provider)
    ts_for_d = _read_pairs(sd / "first_timesteps.csv", "period", "step",
                             provider=provider)

    # Stochastic UNION gate per profile.
    stoch_set = _read_stoch_profiles(inp, provider=provider)
    stoch_active = [p for p in stoch_profiles if p in stoch_set]

    # Branch 1: stochastic UNION fold.
    # For each (p, d, t): sum pbt[p, tb, ts, t] over (tb, ts) ∈
    #     tb_for_d[d] × ts_for_d[d], iff p ∈ stoch_active.
    parts: list[pl.LazyFrame] = []
    if stoch_active and tb_for_d and ts_for_d:
        # Build a join frame: (d, tb, ts) tuples.
        d_tb_ts = []
        for d, tbs in tb_for_d.items():
            tss = ts_for_d.get(d, [])
            for tb in tbs:
                for ts in tss:
                    d_tb_ts.append((d, tb, ts))
        if d_tb_ts:
            d_tb_ts_lf = pl.LazyFrame(
                {"d": [r[0] for r in d_tb_ts],
                 "tb": [r[1] for r in d_tb_ts],
                 "ts": [r[2] for r in d_tb_ts]})
            stoch_lf = (pbt_lf
                          .filter(pl.col("f").is_in(stoch_active))
                          .join(d_tb_ts_lf, on=["tb", "ts"], how="inner")
                          .group_by(["f", "d", "t"])
                          .agg(pl.col("value").sum())
                          .with_columns(branch_priority=pl.lit(1, dtype=pl.Int8))
                       )
            parts.append(stoch_lf)

    # Branch 2: parent-period fold.
    # For each (p, d, t): sum pbt[p, tb, ts, t] over (tb, ts) ∈
    #     tb_for_d[pe] × ts_for_d[d] for each pe ∈ pe_for_d[d].
    if pe_for_d and tb_for_d and ts_for_d:
        # Build (d, tb, ts) where tb is from each parent's tb_for_d.
        d_tb_ts_pp = []
        for d, parents in pe_for_d.items():
            tss = ts_for_d.get(d, [])
            for pe in parents:
                tbs = tb_for_d.get(pe, [])
                for tb in tbs:
                    for ts in tss:
                        d_tb_ts_pp.append((d, tb, ts))
        if d_tb_ts_pp:
            d_tb_ts_pp_lf = pl.LazyFrame(
                {"d": [r[0] for r in d_tb_ts_pp],
                 "tb": [r[1] for r in d_tb_ts_pp],
                 "ts": [r[2] for r in d_tb_ts_pp]})
            pp_lf = (pbt_lf
                       .join(d_tb_ts_pp_lf, on=["tb", "ts"], how="inner")
                       .group_by(["f", "d", "t"])
                       .agg(pl.col("value").sum())
                       .with_columns(branch_priority=pl.lit(2, dtype=pl.Int8))
                     )
            parts.append(pp_lf)

    if not parts:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    # Combine and resolve priority: lower priority wins.
    union = pl.concat(parts, how="vertical")
    return (union
              .sort(["f", "d", "t", "branch_priority"])
              .group_by(["f", "d", "t"])
              .agg(pl.col("value").first())
              .select("f", "d", "t", "value"))


def _read_pairs(path: Path, key_col: str, val_col: str,
                  *, provider: "object | None" = None,
                  ) -> dict[str, list[str]]:
    """Read a 2-col CSV into ``{key: [val, …]}``.  Empty / missing →
    ``{}``.
    """
    if not _provider_has_key(provider, path):
        return {}
    out: dict[str, list[str]] = {}
    try:
        from ._writer_provider_io import _provider_open
        fh_ctx = _provider_open(provider, _provider_key(path), path)
        if fh_ctx is None:
            return {}
        with fh_ctx as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                return out
            # Find column indices.
            try:
                ki = header.index(key_col)
                vi = header.index(val_col)
            except ValueError:
                return out
            for row in reader:
                if len(row) > max(ki, vi):
                    k = row[ki]
                    v = row[vi]
                    if k and v:
                        out.setdefault(k, []).append(v)
    except Exception:
        return {}
    return out


def _read_stoch_profiles(inp: Path,
                            *,
                            provider: "object | None" = None) -> set[str]:
    """Return the set of profile names referenced by stochastic
    relationships.  Mirrors
    ``entity_period_calc_params.py:852-899``.
    """
    groups_stoch: set[str] = set()
    p = inp / "groupIncludeStochastics.csv"
    if _provider_has_key(provider, p):
        try:
            df = _provider_read(provider, p)
            col = df.columns[0] if df.columns else None
            if col is not None and df.height > 0:
                groups_stoch = set(df[col].cast(pl.Utf8).to_list())
        except Exception:
            pass
    if not groups_stoch:
        return set()

    stoch_processes: set[str] = set()
    p = inp / "group__process.csv"
    if _provider_has_key(provider, p):
        try:
            df = _provider_read(provider, p)
            for row in df.iter_rows(named=True):
                if row.get(df.columns[0]) in groups_stoch and row.get(df.columns[1]):
                    stoch_processes.add(row[df.columns[1]])
        except Exception:
            pass
    stoch_nodes: set[str] = set()
    p = inp / "group__node.csv"
    if _provider_has_key(provider, p):
        try:
            df = _provider_read(provider, p)
            for row in df.iter_rows(named=True):
                if row.get(df.columns[0]) in groups_stoch and row.get(df.columns[1]):
                    stoch_nodes.add(row[df.columns[1]])
        except Exception:
            pass

    stoch_profile: set[str] = set()
    p = inp / "process__profile__profile_method.csv"
    if _provider_has_key(provider, p):
        try:
            df = _provider_read(provider, p)
            for row in df.iter_rows(named=True):
                process = row.get("process")
                profile = row.get("profile")
                if process in stoch_processes and profile:
                    stoch_profile.add(profile)
        except Exception:
            pass
    p = inp / "node__profile__profile_method.csv"
    if _provider_has_key(provider, p):
        try:
            df = _provider_read(provider, p)
            for row in df.iter_rows(named=True):
                node = row.get("node")
                profile = row.get("profile")
                if node in stoch_nodes and profile:
                    stoch_profile.add(profile)
        except Exception:
            pass
    p = inp / "process__node__profile__profile_method.csv"
    if _provider_has_key(provider, p):
        try:
            df = _provider_read(provider, p)
            for row in df.iter_rows(named=True):
                process = row.get("process")
                profile = row.get("profile")
                if process in stoch_processes and profile:
                    stoch_profile.add(profile)
        except Exception:
            pass
    return stoch_profile


# ---------------------------------------------------------------------------
# Public entry: p_profile_value_lf + apply_profile_cascade
# ---------------------------------------------------------------------------


def p_profile_value_lf(source: "InputSource",
                            dt: pl.DataFrame,
                            workdir: Path | None = None,
                            *,
                            provider: "object | None" = None,
                            ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for the canonical ``p_profile_value``.

    Entry point for the cluster C cascade.  Composes the 5-branch
    fallback from per-tier lazy frames:

    1. Stochastic / parent-period folds (tiers 1, 2) — when ``workdir``
       is provided and stochastic profiles are present.
    2. Time-axis (tier 3) — for profiles with ``index_name='time'`` /
       generic 1d_map values.
    3. Period-time (2d_map) — for profiles with ``index_name='period'``
       and a child Map / TimeSeries.
    4. Period-only (1d_map(period)) — for profiles with
       ``index_name='period'`` and scalar children.  Note: flextool's
       ``write_pdtProfile`` doesn't actually consult this tier — kept
       for completeness when fixtures populate it.
    5. Scalar (tier 4) — broadcast across (d, t).
    6. Tier 5 (zero) — implicit; the LP-build site treats absence as 0.

    The output schema is ``[f, d, t, value]``.  Profiles absent from the
    parameter table contribute no rows — the LP-build site's own gating
    suppresses them where the schema requires.

    Parameters
    ----------
    source
        :class:`InputSource` for the spine data.
    dt
        Eager ``[d, t, ...]`` dispatch frame from
        ``flex_data.dt`` / ``dt_and_step_duration_from_source``.
    workdir
        Optional ``Path`` for the per-solve workdir.  Required for
        stochastic profile resolution (tiers 1-2) which read
        per-solve scaffolding from ``solve_data/``.  When omitted, the
        stochastic tiers are skipped (tiers 3-5 still emit).
    """
    if dt is None:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })
    dt_lf = dt.lazy().select(
        pl.col("d").cast(pl.Utf8),
        pl.col("t").cast(pl.Utf8),
    ).unique()

    tiers = _classify_profile_rows(source)
    scalar_profiles = sorted(p for p, t in tiers.items() if t == "scalar")
    period_profiles = sorted(p for p, t in tiers.items() if t == "period")
    time_profiles = sorted(p for p, t in tiers.items() if t == "time")
    pt_profiles = sorted(p for p, t in tiers.items() if t == "period_time")
    stoch_profiles = sorted(p for p, t in tiers.items() if t == "stochastic")

    # Build per-tier frames with priority annotation; lower wins.
    parts: list[pl.LazyFrame] = []

    if stoch_profiles and workdir is not None:
        st = _profile_stochastic_lf(workdir, stoch_profiles, provider=provider)
        parts.append(st.with_columns(prio=pl.lit(1, dtype=pl.Int8)))

    if pt_profiles:
        pt = _profile_period_time_lf(source, dt_lf, pt_profiles)
        parts.append(pt.with_columns(prio=pl.lit(2, dtype=pl.Int8)))

    if time_profiles:
        tt = _profile_time_lf(source, dt_lf, time_profiles, workdir=workdir,
                               provider=provider)
        parts.append(tt.with_columns(prio=pl.lit(3, dtype=pl.Int8)))

    if period_profiles:
        pp = _profile_period_only_lf(source, dt_lf, period_profiles)
        parts.append(pp.with_columns(prio=pl.lit(4, dtype=pl.Int8)))

    if scalar_profiles:
        sc = _profile_scalar_lf(source, dt_lf, scalar_profiles)
        parts.append(sc.with_columns(prio=pl.lit(5, dtype=pl.Int8)))

    # Tier 5 — zero fallback for every profile entity × dt cell.
    # Mirrors flextool's ``write_pdtProfile`` final ``else: 0.0`` branch
    # (entity_period_calc_params.py:946-947).  Every profile listed in
    # ``profile.csv`` (the entity table) gets a row for each (d, t) of
    # the dispatch frame; tiers 1-4 then overlay the actual values.
    try:
        ents = source.entities("profile")
        if ents.height > 0:
            ent_lf = (ents.lazy()
                          .select(alias_to_axis("name", "f"))
                          .unique())
            zero_lf = (ent_lf
                          .join(dt_lf, how="cross")
                          .with_columns(value=pl.lit(0.0, dtype=pl.Float64))
                          .with_columns(prio=pl.lit(99, dtype=pl.Int8)))
            parts.append(zero_lf)
    except KeyError:
        pass

    if not parts:
        return pl.LazyFrame(schema={
            "f": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8, "value": pl.Float64,
        })

    union = pl.concat(parts, how="vertical")
    return (union
              .sort(["f", "d", "t", "prio"])
              .group_by(["f", "d", "t"])
              .agg(pl.col("value").first())
              .select("f", "d", "t", "value")
              .sort(["f", "d", "t"]))


def p_profile_value_from_source_v2(source: "InputSource",
                                          dt: pl.DataFrame,
                                          workdir: Path | None = None,
                                          *,
                                          provider: "object | None" = None,
                                          ) -> Param | None:
    """Δ.7 lazy port replacement for
    ``_derived_params.p_profile_value_from_source``.

    Returns ``Param(("f", "d", "t"), frame)`` or ``None`` when the
    cascade produces no rows (no profiles in the source, or all
    profiles unresolved).
    """
    if dt is None:
        return None
    lf = p_profile_value_lf(source, dt, workdir=workdir, provider=provider)
    out = lf.collect()
    if out.height == 0:
        return None
    return Param(("f", "d", "t"), out)


def apply_profile_cascade(flex_data: object,
                              source: "InputSource",
                              workdir: Path | None,
                              *,
                              provider: "object | None" = None) -> None:
    """Public entry: write ``flex_data.p_profile_value`` from cluster C.

    Invoked after :func:`apply_derived_a` so ``flex_data.dt`` is
    populated.  Matches flextool's per-solve cascade: emits a Param
    when at least one profile resolves; the helper returns ``None``
    when no profile is declared (tier-5 / zero is implicit at the
    LP-build site).

    Δ.12b — assignment is unconditional; ``None`` is the explicit
    "no profile data" signal (no silent fall-through to a CSV-loaded
    seed value).
    """
    dt = getattr(flex_data, "dt", None)
    if dt is None:
        return
    flex_data.p_profile_value = p_profile_value_from_source_v2(
        source, dt, workdir=workdir, provider=provider)


__all__ = [
    "p_profile_value_lf",
    "p_profile_value_from_source_v2",
    "apply_profile_cascade",
]
