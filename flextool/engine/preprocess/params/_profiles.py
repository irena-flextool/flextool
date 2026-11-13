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
``groupIncludeStochastics`` / membership tables, read from the sub-solve's
provider (``solve_data/*`` / ``input/*`` keys; the work folder on disk is never
read).  The deterministic path (branches 3-5) is fully native.

Architectural notes
-------------------

* **Drop the ``_check_canonical_keys`` predicate.**  The Δ.6 close
  stanza identified ``_derived.py:p_profile_value_from_source``
  as guarding against non-canonical Map keys (``x``/``i`` from
  zero-``index_name`` Spine values) by returning ``None`` and falling
  back to the CSV path.  Cluster C drops the predicate: the stochastic
  3d_map case is now handled explicitly via the per-solve provider frames, the
  deterministic cases are handled via per-row dispatch on the raw
  parameter value's runtime type rather than column-name sniffing.
* **Lazy polars throughout.**  Every helper returns a
  :class:`pl.LazyFrame`; the public ``apply_profile_cascade`` boundary
  collects once.
* **Per-solve provider frames for tiers 1 and 2.**  The stochastic
  scaffolding comes from the provider's ``solve_data/*`` keys (the
  per-solve emit), not from typed FlexData fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.params._branch import inherit_active_branch_periods
from flextool.engine.preprocess.emit._pdt_lookup import (
    _anchor_fan_members,
    _check_pbt_branch_coverage,
    _check_pre_reveal_reads_have_forecast,
    _check_roll_forecast_covers_lookahead,
    _check_ungrouped_reads_have_realized_branch,
    read_fan_for_d,
    ts_tiers,
)
from flextool.engine.input._axis_enums import (
    alias_to_axis,
    schema_dtype,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.input._solve_step_aggregation import aggregate_if_active


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None


def _cell_str(value: "object | None") -> str:
    """Reproduce a ``csv.reader`` cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; ``csv.reader`` then reads those
    strings back.  Mirror that here so dict keys/values stay byte-identical
    to the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


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
    from flextool.engine.input._input_source import InputSource


# ---------------------------------------------------------------------------
# Per-entity profile classification: scalar / period / time / branch+t
# ---------------------------------------------------------------------------


def _classify_profile_rows(source: "InputSource",
                           dt: "pl.DataFrame | None" = None) -> dict[str, str]:
    """Return ``{profile_name: tier}`` where tier ∈ {``scalar``,
    ``period``, ``time``, ``period_time``, ``stochastic``}.

    Reads ``profile.profile`` raw values from the source and
    dispatches on Python type:

    * ``float`` / ``str`` / ``bool`` → tier ``scalar`` (``p_profile.csv``).
    * Map with ``index_name=='time'`` → tier ``time`` (``pt_profile.csv``).
    * Map with ``index_name=='period'`` → tier ``period``.
    * 1-D Map without an index name (Spine's silent ``x``) → classified by
      its keys (:func:`_silent_map_tier`, item 73): ``period`` when they hit
      the solve's periods and none of its time steps, else ``time``.
    * 3-deep Map → tier ``stochastic`` (``pbt_profile.csv``).
    * 2-D Map → ``period_time`` when the outer level is ``period``, or has no
      index name and its keys are periods by the silent-map rule (item 223),
      else ``stochastic``.

    *dt* is the solve's ``(d, t)`` frame: the period / time-step vocabulary of
    the silent-map rule (:func:`~flextool.engine.input._param_shapes._period_time_vocab`,
    so the solve's own periods plus the active stochastic branch anchors).
    Without *dt* a silent map stays ``time`` (the pre-item-73 reading).

    Profiles absent from the parameter table are not in the returned
    dict.  The caller treats them as tier 5 (zero fallback).

    Implementation strategy: peek at the SpineDbReader's per-row raw
    value via ``_param_rows`` when available; fall back to the frame's
    columns for sources that don't expose raw parameter values
    (``InMemoryReader``).  The column fallback reads a frame with ONE index
    column that is neither ``period`` nor ``t`` / ``time`` (a silent ``x``)
    by the same key rule, never as ``stochastic``; a frame with the columns
    ``x`` and ``x_2`` (or ``x`` and ``t``) is a 2-D map with an unnamed outer
    level, ``period_time`` when the ``x`` keys are periods.
    """
    from flextool.engine.input._param_shapes import _period_time_vocab
    periods, timesteps = _period_time_vocab(dt)
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
        return {n: _tier_for_value(v, periods, timesteps) for n, v in rows_by_name.items()}
    # Column-shape fallback for InMemoryReader (used in unit tests).
    try:
        df = source.parameter("profile", "profile")
    except KeyError:
        return {}
    if df.height == 0:
        return {}
    cols = set(df.columns)
    index_cols = cols - {"name", "value"}
    silent_col = next(iter(index_cols)) if (
        len(index_cols) == 1 and not index_cols & {"period", "t", "time"}) else None
    # A 2-D map with an unnamed outer level: the reader's ``x`` and ``x_2`` (or ``t``) columns (item 223).
    silent_outer = "x" if (index_cols in ({"x", "x_2"}, {"x", "t"})) else None
    out: dict[str, str] = {}
    for name in df["name"].unique().to_list():
        if cols == {"name", "value"}:
            out[name] = "scalar"
        elif cols == {"name", "period", "value"}:
            out[name] = "period"
        elif cols == {"name", "t", "value"}:
            out[name] = "time"
        elif cols in ({"name", "period", "t", "value"}, {"name", "period", "x", "value"}):
            # 2d_map(period, t) — fold as time-axis per period.  Treat
            # as a multi-row variant of "time" here; the resolver knows.
            out[name] = "period_time"
        elif silent_col is not None:
            # A 1-D map under a silent index column: the same key rule as a raw value.
            keys = df.filter(pl.col("name") == name).get_column(silent_col).cast(pl.Utf8).to_list()
            out[name] = _silent_map_tier(keys, periods, timesteps)
        elif silent_outer is not None:
            # The outer level of an unnamed 2-D map: period x time when its keys are periods (item 223).
            keys = df.filter(pl.col("name") == name).get_column(silent_outer).cast(pl.Utf8).to_list()
            out[name] = "period_time" if _silent_map_tier(keys, periods, timesteps) == "period" else "stochastic"
        else:
            # Generic / stochastic 3d_map — defer to the per-solve pbt_profile frame.
            out[name] = "stochastic"
    return out


def _silent_map_tier(keys: "object", periods: "set[str] | None",
                     timesteps: "set[str] | None") -> str:
    """Tier of a 1-D profile map without an index name (item 73): ``period`` when its keys hit the solve's
    periods and none of its time steps, else ``time``.

    A key that is not a period of this solve (a period of another roll / sub-solve, or of no solve) does not
    stop the period reading: the map has no value in this solve's other periods and the profile is 0 there, as
    for a map labelled ``period``.  Unlike the parameter resolver, which drops such an entity without a word
    (``_param_shapes._silent_entity_axes``), the run is told: ``validate_period_keys`` warns once per run about
    a profile period map that misses model periods, and stops the run on a map whose keys match both periods
    and time steps of the model (so the "both" case never reaches a solve from the database; here it keeps the
    time reading).  Keys matching neither stay ``time`` (0, as before; ``validate_period_keys`` reports keys
    that are no period).
    """
    key_set = {str(k) for k in keys if k is not None}
    hits_periods = bool(periods) and not key_set.isdisjoint(periods)
    hits_timesteps = bool(timesteps) and not key_set.isdisjoint(timesteps)
    return "period" if hits_periods and not hits_timesteps else "time"


def _tier_for_value(v: object, periods: "set[str] | None" = None,
                    timesteps: "set[str] | None" = None) -> str:
    """Classify a single raw Spine parameter value.

    Mirror of the input-writer dispatch logic.  *periods* / *timesteps*: the
    solve's vocabulary for a 1-D map without an index name
    (:func:`_silent_map_tier`); ``None`` keeps such a map ``time``.
    """
    from spinedb_api.parameter_value import (
        Map, TimeSeries, Array,
    )
    from flextool.engine.input._param_shapes import _normalise_label
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
            if _normalise_label(v.index_name) is None:
                # Spine's silent ``x`` (or no name): read by the keys (item 73).
                return _silent_map_tier(v.indexes, periods, timesteps)
            # Any other explicit name — assume time-axis (the typical
            # flextool 1d_map default).
            return "time"
        if depth == 2:
            if outer_idx == "period":
                return "period_time"
            if (_normalise_label(v.index_name) is None
                    and _silent_map_tier(v.indexes, periods, timesteps) == "period"):
                # Unnamed outer level keyed by periods (item 223, item 73's rule): period x time.
                return "period_time"
            # Any other 2d_map: stochastic (the per-solve pbt_profile frame, which only a
            # 3-level map fills; validate_period_keys stops the run on such a 2-D profile).
            return "stochastic"
        # Deeper than 2 → 3d_map (branch × time_start × t).
        return "stochastic"
    # Array — uncommon for profiles; treat as time-axis.
    return "time"


def _profile_rows(source: "InputSource", names: list[str], *,
                  cache: bool) -> pl.DataFrame:
    """``profile.profile`` rows of the profiles in *names*.

    A :class:`SpineDbReader` unrolls only those profiles'
    values (``parameter_for_entities``), so a scalar / period /
    period x time tier never unrolls the time-series profiles that the
    time tier takes from the per-solve ``pt_profile.csv``.  ``cache=True``
    memoises the frame on the reader: the synthetic path runs this cascade
    in every roll / nested sub-solve of a run with the same reader, and
    the non-time tiers do not depend on the roll (the per-roll ``dt``
    join happens afterwards).  Other sources (``InMemoryReader``) return
    the full frame filtered to *names*.
    """
    by_entities = getattr(source, "parameter_for_entities", None)
    if by_entities is not None:
        return by_entities("profile", "profile", names, cache=cache)
    raw = source.parameter("profile", "profile")
    if raw.height == 0:
        return raw
    return raw.filter(pl.col("name").is_in(names))


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

    Prefers the provider's ``solve_data/pt_profile`` frame (per-solve
    averaged values from ``TimelineConfig.create_averaged_timeseries``)
    when present — those values are aggregated to the solve's
    ``new_stepduration``, matching the dispatch timeline.  Falls back
    to the raw Spine ``profile.profile`` parameter when the provider
    lacks the frame (e.g. tests that bypass the runner).

    The Spine fallback path: the source column may be ``t`` (when
    ``index_name='time'``) or ``i`` / ``x`` (no index name).  Either way
    the column maps 1:1 to t.  Mirrors how ``pdtNodeInflow`` already
    sources averaged values from ``solve_data/pt_node_inflow.csv``.
    """
    empty_schema = {
        "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
    }
    if not time_profiles:
        return pl.LazyFrame(schema=empty_schema)

    # Prefer the per-solve averaged frame when the runner has produced it.
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
        # Only reached without a per-solve pt_profile.csv: not memoised (a
        # time-series frame is large and this is not the per-roll path).
        raw = _profile_rows(source, time_profiles, cache=False)
    except KeyError:
        return pl.LazyFrame(schema=empty_schema)
    if raw.height == 0:
        return pl.LazyFrame(schema=empty_schema)
    cols = raw.columns
    # Find the time-axis column.  Prefer 't' (canonical) over 'i' / 'x'
    # (a map without an index name) when both are present.
    t_col = next((c for c in ("t", "i", "time", "x") if c in cols), None)
    if t_col is None:
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
    membership.  In a solve with ``new_stepduration`` each solve step takes
    the duration-weighted mean of its time steps first
    (:func:`~flextool.engine.input._solve_step_aggregation.aggregate_if_active`).
    """
    if not pt_profiles:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    try:
        raw = _profile_rows(source, pt_profiles, cache=True)
    except KeyError:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    if raw.height == 0:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    cols = raw.columns
    # The outer (period) key: ``period``, or ``x`` of a map whose outer level has no index name (item 223).
    # The inner (time) key: ``t`` (or ``i``) of a labelled inner level; ``x_2`` of an unnamed inner level under
    # an unnamed outer one; ``x`` of an unnamed inner level under an outer level labelled ``period`` (the reader
    # names the first unnamed level ``x``, whatever its depth).  A frame holding several shapes has the other
    # columns null on each row, so ``x`` is the time key exactly on the rows whose ``period`` is set.
    period_cols = [c for c in ("period", "x") if c in cols]
    t_exprs = [pl.col(c).cast(pl.Utf8) for c in ("t", "i", "x_2") if c in cols]
    if "period" in cols and "x" in cols:
        t_exprs.append(pl.when(pl.col("period").is_not_null()).then(pl.col("x").cast(pl.Utf8)))
    if not period_cols or not t_exprs:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    t_col = "t"
    raw = raw.select("name",
                     pl.coalesce([pl.col(c).cast(pl.Utf8) for c in period_cols]).alias("period"),
                     pl.coalesce(t_exprs).alias(t_col),
                     "value")
    # Branch periods get their anchor period's curve (this reader bypasses _param_shapes).
    raw = inherit_active_branch_periods(raw.filter(pl.col("name").is_in(pt_profiles)),
                                        group_keys=("name",), period_col="period")
    # A solve with new_stepduration: one value per solve step, the step mean (_solve_step_aggregation).
    raw = aggregate_if_active(
        raw.select("name", "period", t_col, pl.col("value").cast(pl.Float64, strict=False)),
        ("profile", "profile"), time_col=t_col, steps=dt_lf.select("t").collect().get_column("t"))
    raw_lf = (raw.lazy()
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

    A map labelled ``period`` and a map without an index name whose keys
    are periods (:func:`_silent_map_tier`, item 73) both land here: the
    silent map's key column (Spine's ``x``) is read as the period.
    Broadcast across t of the matching period; a period the map does not
    name has no row (the profile is 0 there, the zero tier).
    """
    if not period_profiles:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    try:
        raw = _profile_rows(source, period_profiles, cache=True)
    except KeyError:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    if raw.height == 0:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    # The period key: the ``period`` column of a labelled map, else the one key column of a map without an
    # index name (``x``); a frame holding both kinds has the other column null on each row.
    key_cols = [c for c in raw.columns if c not in ("name", "value", "t", "time")]
    if "period" in key_cols:
        key_cols = ["period"] + [c for c in key_cols if c != "period"]
    if not key_cols:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    raw = raw.select("name", pl.coalesce([pl.col(c).cast(pl.Utf8) for c in key_cols]).alias("period"), "value")
    # Branch periods get their anchor period's value (this reader bypasses _param_shapes).
    raw = inherit_active_branch_periods(raw.filter(pl.col("name").is_in(period_profiles)),
                                        group_keys=("name",), period_col="period")
    raw_lf = (raw.lazy()
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
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    try:
        raw = _profile_rows(source, scalar_profiles, cache=True)
    except KeyError:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    if raw.height == 0:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
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
# Tier-1 / tier-2 builders (stochastic — fed from the provider's per-solve frames)
# ---------------------------------------------------------------------------


def _profile_stochastic_lf(workdir: Path | None,
                                stoch_profiles: list[str],
                                *,
                                provider: "object | None" = None,
                                ) -> pl.LazyFrame:
    """Lazy ``[f, d, t, value]`` for stochastic 3d_map profiles.

    Branches 1 and 2 of flextool's ``write_pdtProfile`` cascade.

    Per-solve scaffolding consumed (provider keys; *workdir* only forms them):

    * ``solve_data/period__branch`` — (anchor, branch) tuples; defines parents.
    * ``solve_data/solve_branch__time_branch`` — (anchor, time_branch); the
      stochastic-fold's outer-time-branch axis.
    * ``solve_data/first_timesteps`` — (anchor, time_start); the
      stochastic-fold's outer-time-start axis.
    * ``solve_data/pbt_profile`` — (profile, branch, time_start, time,
      value) — the 3d_map values themselves.
    * ``input/groupIncludeStochastics`` + relationship frames — gate
      Branch 1 (per-profile is_stoch flag).

    Output: lazy frame ``[f, d, t, value]`` covering only the
    (profile, d, t) tuples that actually hit Branch 1 or Branch 2.
    The caller layers tiers 3-5 over the gaps.
    """
    if workdir is None or not stoch_profiles:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    workdir = Path(workdir)
    inp = workdir / "input"
    sd = workdir / "solve_data"

    # Read pbt_profile.csv: (profile, branch, time_start, time, pbt_profile).
    pbt_path = sd / "pbt_profile.csv"
    if not _provider_has_key(provider, pbt_path):
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    try:
        pbt = _provider_read(provider, pbt_path)
    except Exception:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    if pbt.height == 0:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
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
    # The fan start of each branch period (``branch_time_start``; empty without
    # a fan), and the roll look-ahead periods that read it first (``ts_tiers``).
    fan_for_d = read_fan_for_d(sd, provider=provider)

    # Stochastic UNION gate per profile.
    stoch_set = _read_stoch_profiles(inp, provider=provider)
    stoch_active = [p for p in stoch_profiles if p in stoch_set]

    # The same input checks as the non-profile pbt_* params (one definition of
    # "realized branch"): a solve without stochastic branches needs a realized
    # branch for a stochastic profile; a profile outside every include_stochastics
    # group reads the realized branch on a fan's branch periods, so a
    # forecasts-only fan cannot serve it; and a profile must author every active
    # branch of a fan or none of them.
    if stoch_active and not tb_for_d:
        # This solve has no branch assignment of its own and none projected onto it
        # from a parent (``_StochasticTimeSource.has_realized_branch_rows`` -- same
        # csv, same rule): a solve without stochastic branches still needs a
        # realized branch to take a stochastic profile value from (owner decision #82).
        names = ", ".join(f"'{p}'" for p in stoch_active)
        raise FlexToolConfigError(
            f"profile {names}, parameter 'profile': the value is a stochastic forecast (a map over "
            "branch, analysis time and time), but this solve has no realized branch to take the value "
            "from: no row of its stochastic_branches says realized: yes, and no other solve hands its "
            "realized branch to this one. Either mark one branch realized: yes in this solve's "
            "stochastic_branches, or give the profile a plain value instead of a forecast map. Check "
            "also which alternative of the scenario gives the value."
        )
    anchor_members = _anchor_fan_members(sd / "period__branch.csv", provider=provider)
    pbt_dict = {
        (row["profile"], "profile", row["branch"], row["time_start"], row["time"]): row["value"]
        for row in pbt.iter_rows(named=True)
    }
    _check_ungrouped_reads_have_realized_branch(pbt_dict, 1, frozenset(stoch_set), anchor_members, tb_for_d,
                                                ts_for_d, fan_for_d=fan_for_d, kind="profile")
    if stoch_active:
        _check_pbt_branch_coverage({k: v for k, v in pbt_dict.items() if k[0] in stoch_active}, 1, anchor_members,
                                   tb_for_d, ts_for_d, fan_for_d, kind="profile")
    # Before the reveal, a period reads the forecast made when the solve or roll starts, and it must cover it.
    _check_pre_reveal_reads_have_forecast(pbt_dict, 1, frozenset(stoch_active), tb_for_d, ts_for_d, pe_for_d,
                                          fan_for_d, sd, provider=provider, kind="profile")
    # In a roll of a rolling solve the roll's own forecast must cover the roll's look-ahead periods.
    _check_roll_forecast_covers_lookahead(pbt_dict, 1, frozenset(stoch_active), tb_for_d, ts_for_d, pe_for_d,
                                          fan_for_d, sd, provider=provider, kind="profile")

    # Each fold reads, per value cell, the first analysis-time tier with a
    # value: d's own first step and the fan start, in ``ts_tiers`` order.  The tier
    # rides along as ``tier``; a part is summed per (f, d, t, tier) and the
    # priority ``branch * 2 + tier`` keeps branch order first, tier second.
    def _fold_frame(rows: list[tuple[str, str, str, int]]) -> pl.LazyFrame:
        return (pl.LazyFrame(
            {"d": [r[0] for r in rows],
             "tb": [r[1] for r in rows],
             "ts": [r[2] for r in rows],
             "tier": [r[3] for r in rows]},
            schema={"d": pl.Utf8, "tb": pl.Utf8, "ts": pl.Utf8, "tier": pl.Int8})
            .with_columns(alias_to_axis("d", "d")))

    def _sum_by_tier(lf: pl.LazyFrame, branch: int) -> pl.LazyFrame:
        return (lf.group_by(["f", "d", "t", "tier"])
                  .agg(pl.col("value").sum())
                  .with_columns(branch_priority=(pl.lit(2 * branch, dtype=pl.Int8) + pl.col("tier"))
                                .cast(pl.Int8))
                  .drop("tier"))

    # Branch 1: stochastic UNION fold.
    # For each (p, d, t): sum pbt[p, tb, ts, t] over (tb, ts) ∈
    #     tb_for_d[d] × tier(d), iff p ∈ stoch_active.
    parts: list[pl.LazyFrame] = []
    if stoch_active and tb_for_d and ts_for_d:
        # Build a join frame: (d, tb, ts, tier) tuples.
        d_tb_ts = []
        for d, tbs in tb_for_d.items():
            for tier, tss in enumerate(ts_tiers(d, ts_for_d, fan_for_d)):
                for tb in tbs:
                    for ts in tss:
                        d_tb_ts.append((d, tb, ts, tier))
        if d_tb_ts:
            stoch_lf = _sum_by_tier(pbt_lf
                                    .filter(pl.col("f").is_in(stoch_active))
                                    .join(_fold_frame(d_tb_ts), on=["tb", "ts"], how="inner"), 1)
            parts.append(stoch_lf)

    # Branch 2: parent-period fold.
    # For each (p, d, t): sum pbt[p, tb, ts, t] over (tb, ts) ∈
    #     tb_for_d[pe] × tier(d) for each pe ∈ pe_for_d[d].
    if pe_for_d and tb_for_d and ts_for_d:
        # Build (d, tb, ts, tier) where tb is from each parent's tb_for_d.
        d_tb_ts_pp = []
        for d, parents in pe_for_d.items():
            for tier, tss in enumerate(ts_tiers(d, ts_for_d, fan_for_d)):
                for pe in parents:
                    tbs = tb_for_d.get(pe, [])
                    for tb in tbs:
                        for ts in tss:
                            d_tb_ts_pp.append((d, tb, ts, tier))
        if d_tb_ts_pp:
            pp_lf = _sum_by_tier(pbt_lf.join(_fold_frame(d_tb_ts_pp), on=["tb", "ts"], how="inner"), 2)
            parts.append(pp_lf)

    if not parts:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
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
    """Read a 2-col provider frame (key formed from *path*) into
    ``{key: [val, …]}``.  Empty / missing → ``{}``.
    """
    if not _provider_has_key(provider, path):
        return {}
    out: dict[str, list[str]] = {}
    try:
        df = _provider_read(provider, path)
        # The frame carries the header names as ``df.columns``; map by
        # name exactly as the legacy ``header.index(...)`` lookup did.
        if key_col not in df.columns or val_col not in df.columns:
            return out
        for k_cell, v_cell in df.select(key_col, val_col).iter_rows():
            k = _cell_str(k_cell)
            v = _cell_str(v_cell)
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
        stochastic profile resolution (tiers 1-2), whose per-solve
        scaffolding (``solve_data/`` / ``input/`` keys) is read from
        *provider*.  When omitted, the stochastic tiers are skipped
        (tiers 3-5 still emit).
    provider
        :class:`FlexDataProvider` carrying that scaffolding.  Required
        when *workdir* is given (``ValueError`` otherwise); optional
        for the DB-only form.
    """
    if dt is None:
        return pl.LazyFrame(schema={
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
        })
    # Defensive re-cast: callers occasionally pass dt with Utf8 d/t (e.g. stochastic
    # fixtures where a CSV path bypasses upstream casting). Re-cast here so every
    # tier helper composes on a consistent dtype.
    dt_lf = (dt.lazy()
                .select(alias_to_axis("d", "d"), alias_to_axis("t", "t"))
                .unique())

    # The stochastic / averaged-time tiers read the per-solve scaffolding
    # from *provider* only (never from ``workdir`` on disk), so a workdir
    # without a provider is a caller error.  The DB-only form
    # (``workdir=None``, no provider) stays valid: tiers 3-5 need neither.
    if workdir is not None and provider is None:
        raise ValueError(
            "p_profile_value_lf: a workdir needs a FlexDataProvider for the "
            "per-solve scaffolding; the work folder on disk is never read."
        )

    tiers = _classify_profile_rows(source, dt)
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
            "f": schema_dtype(_enums, "f"), "d": schema_dtype(_enums, "d"), "t": schema_dtype(_enums, "t"), "value": pl.Float64,
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
    ``_derived.p_profile_value_from_source``.

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
