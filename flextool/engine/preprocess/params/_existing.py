"""Cluster B — existing chain & invest sets (Δ.6).

Lazy-polars port of flextool's existing-capacity cascade and the
invest/divest set family.  Replaces the eager-Python helpers spread
across :mod:`._derived` (``p_entity_all_existing_from_source``,
``ed_invest_set_from_source`` / ``ed_divest_set_from_source``,
``p_entity_max_units_from_source``, etc.) and the workdir-CSV reads
they currently depend on.

Cluster B fields (per
``audit/native_data_path_design_derived_clusters.md``):

* ``entityInvest`` / ``entityDivest`` — the (e,)-shaped entity
  projection of ``entity__invest_method`` filtered by the
  not-allowed-method constants
  (``preprocessing/invest_method_sets.py``).  These were the trap that
  Δ.5 caught: the eager helper derived them from
  ``ed_invest_set`` / ``ed_divest_set`` instead of from
  ``entity__invest_method``.  This module derives them correctly.
* ``e_invest_total`` / ``e_divest_total`` — projections of the above
  filtered by INVEST_TOTAL / RETIRE_TOTAL methods
  (``preprocessing/invest_total_sets.py``).
* ``ed_invest_set`` / ``ed_divest_set`` — (e, d) cross-product
  entityInvest (entityDivest) × period_invest, minus the no_reinvestment
  lifetime-expiry forbidden set (retirement: only with pre-existing
  capacity); the cost does not gate it (#218).
* ``pd_invest_set`` / ``nd_invest_set`` / ``pd_divest_set`` /
  ``nd_divest_set`` — partitions of ``ed_invest_set`` / ``ed_divest_set``
  by entity-class (``unit ∪ connection`` → process-side; ``node`` →
  node-side).
* The periods each investment and each retirement counts in are views of the cohort table
  (:mod:`._cohorts`: ``invest_alive``, ``invest_prebuilt``, ``retire_active``), built for the model by
  ``model/_capacity.py``.
* ``p_entity_all_existing`` / ``p_entity_retired_carried`` — the capacity a solve starts from and the carried
  retirements of the existing cohort are emitted per solve from the cohorts
  (``emit/_chain_params.emit_existing_capacity``); preprocess reads them (:func:`apply_existing_chain`).
* ``p_entity_max_units`` — per-(e, d) max unit count via the
  unitsize cascade × the max_capacity branch fork.

The shared :func:`._walks.period_walk_iterator` lifted out of
Cluster A's ``_npv`` is consumed throughout for the lifetime-
window walks.

All helpers are lazy until the public ``apply_existing_chain`` boundary
calls ``.collect()`` once per emitted Param.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    align_join_dtypes,
    rename_to_axis,
    schema_dtype,
)
from flextool.engine.preprocess.params._branch import inherit_active_branch_periods
from flextool.engine.preprocess.params._walks import period_walk_iterator, WindowMethod


# Substrate handle for the cascade-wide axis enum vocabulary.
# Bare ``None`` here; ``cast_dim`` / ``schema_dtype`` in
# ``_axis_enums`` fall back to ``_LIVE_AXIS_ENUMS_CTX`` (the live
# ContextVar) when this is ``None``, so substrate sites pick up
# activation set by ``load_flextool`` automatically.
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext


# -- Method-enum constants ---------------------------------------------------
# The one definition of which ``invest_method`` values forbid investing /
# retiring.  Every other module imports these (pinned against the schema
# value lists ``invest_methods`` / ``group_invest_methods`` by
# tests/engine/test_invest_method_lists.py).

# Every value of the schema list ``invest_methods``.
INVEST_METHODS_ALL: frozenset[str] = frozenset((
    "not_allowed", "invest_no_limit", "invest_period", "invest_total",
    "invest_period_total",
    "retire_no_limit", "retire_period", "retire_total", "retire_period_total",
    "invest_retire_no_limit", "invest_retire_period",
    "invest_retire_total", "invest_retire_period_total",
    "cumulative_limits",
))
INVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
    "retire_period_total",
))
DIVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
    "invest_period_total",
))
# Groups have their own value list (``group_invest_methods``, database
# version 77): ``not_active`` is the group's "no limits" value.
GROUP_INVEST_NOT_ACTIVE: frozenset[str] = frozenset(
    {"not_active"} | (INVEST_NOT_ALLOWED - {"not_allowed"}))
GROUP_DIVEST_NOT_ACTIVE: frozenset[str] = frozenset(
    {"not_active"} | (DIVEST_NOT_ALLOWED - {"not_allowed"}))
_INVEST_NOT_ALLOWED = INVEST_NOT_ALLOWED
_DIVEST_NOT_ALLOWED = DIVEST_NOT_ALLOWED
# preprocessing/invest_total_sets.py:25-32
_INVEST_TOTAL_METHODS: frozenset[str] = frozenset((
    "invest_total", "invest_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
_DIVEST_TOTAL_METHODS: frozenset[str] = frozenset((
    "retire_total", "retire_period_total",
    "invest_retire_total", "invest_retire_period_total",
))
# preprocessing/invest_divest_sets.py:33-40
_INVEST_PERIOD_METHODS: frozenset[str] = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))
_DIVEST_PERIOD_METHODS: frozenset[str] = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
))

# preprocessing/method_with_fallback_sets.py::_LIFETIME_METHOD_DEFAULT
_LIFETIME_METHOD_DEFAULT = "reinvest_automatic"


# ---------------------------------------------------------------------------
# Building blocks (lazy frames)
# ---------------------------------------------------------------------------


def _entity_class_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e, ec]``: every (entity, entity_class) pairing.

    Mirror of :func:`._lifetime_inputs._entity_class_lf` (kept local to
    avoid cross-module coupling).
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        try:
            df = source.entities(ec)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.lit(ec).alias("ec"),
        ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "ec": pl.Utf8,
        })
    return pl.concat(parts, how="vertical")


def _all_entities_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` — union of unit + node + connection."""
    return _entity_class_lf(source).select("e").unique()


def _entity_method_lf(source: "InputSource",
                          parameter_name: str,
                          ) -> pl.LazyFrame:
    """Lazy ``[e, method]`` from ``unit/node/connection.<param>``.

    ``parameter_name`` ∈ {``invest_method``, ``lifetime_method``}.
    Mirror of :func:`._lifetime_inputs._entity_method_lf`.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        try:
            df = source.parameter(ec, parameter_name)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.col("value").cast(pl.Utf8, strict=False).alias("method"),
        ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "method": pl.Utf8,
        })
    return pl.concat(parts, how="vertical")


def _lifetime_method_with_default_lf(
        source: "InputSource",
        all_entities_lf: pl.LazyFrame,
        ) -> pl.LazyFrame:
    """Lazy ``[e, method]`` for ``lifetime_method`` with default fill.

    Entities without an explicit ``lifetime_method`` row get
    ``reinvest_automatic``.  Mirror of
    :func:`._lifetime_inputs._lifetime_method_with_default_lf`.
    """
    explicit = _entity_method_lf(source, "lifetime_method")
    explicit_entities = explicit.select("e").unique()
    default_e = (all_entities_lf
                   .join(explicit_entities, on="e", how="anti")
                   .with_columns(method=pl.lit(_LIFETIME_METHOD_DEFAULT)))
    return pl.concat([explicit, default_e], how="vertical")


def _per_entity_param_lf(source: "InputSource",
                            parameter_name: str,
                            ) -> pl.LazyFrame:
    """Lazy ``[e, d, value, is_scalar]``.

    Mirror of :func:`._lifetime_inputs._per_entity_param_lf`.

    Period-dim detection: a per-period parameter exposes exactly one
    index column alongside ``name`` / ``value``.  The canonical name is
    ``period``, but the source may surface the column under the user's
    ``Map.index_name`` (e.g. ``x``) when the spinedb default is
    overridden.  Treat any single non-``name``/``value`` column as the
    period dim — checking only ``"period" in cols`` mis-classifies
    user-renamed indices as scalars, which then explode through
    :func:`_resolve_per_period_lf`'s ``scalar.join(on="e")``.

    A stochastic branch period inherits its anchor's explicit row here
    (unconditionally, not only under recourse — #63): a branch period is
    never itself authored, so without this every per-period ``existing``
    / ``construction_lead_time`` / ``virtual_unitsize`` row is invisible in every
    branch and falls back to ``fill=0.0`` in :func:`_resolve_per_period_lf`.
    """
    parts: list[pl.LazyFrame] = []
    for ec in ("unit", "node", "connection"):
        try:
            df = source.parameter(ec, parameter_name)
        except KeyError:
            continue
        if df.height == 0:
            continue
        cols = df.columns
        # Any extra column beyond name / value is the per-period index.
        # ``existing`` / ``lifetime`` / ``virtual_unitsize`` have at
        # most one such extra column; fall back to ``period`` when
        # present (typical), otherwise pick the first non-name/value
        # column (covers ``x`` / other user-renamed Map indices).
        extra = [c for c in cols if c not in ("name", "value")]
        period_col = "period" if "period" in extra else (extra[0] if extra else None)
        if period_col is not None:
            # Branch periods never carry their own row; inherit the
            # anchor's (no-op outside a stochastic load, and a no-op on
            # the null-index / scalar rows — ``inherit_anchor_period``
            # never joins a null period).
            df = inherit_active_branch_periods(
                df, group_keys=("name",), period_col=period_col)
            # Per-ROW scalar detection — see the matching helper in
            # ``_lifetime_inputs._per_entity_param_lf`` for the full rationale.
            # A constant ``existing`` (etc.) stored in an entity class
            # that ALSO carries period-Map values for sibling entities
            # surfaces with a NULL index (``x = null``).  That row must
            # broadcast across the period universe (``is_scalar=True``),
            # not be emitted as an explicit ``(e, d=null)`` row that
            # never joins the period grid and silently zeroes out.
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                alias_to_axis(period_col, "d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.col(period_col).is_null().alias("is_scalar"),
            ))
        else:
            parts.append(df.lazy().select(
                alias_to_axis("name", "e"),
                pl.lit(None, dtype=schema_dtype(_enums, "d")).alias("d"),
                pl.col("value").cast(pl.Float64, strict=False),
                pl.lit(True).alias("is_scalar"),
            ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64, "is_scalar": pl.Boolean,
        })
    return pl.concat(parts, how="vertical")


def _resolve_per_period_lf(per_param: pl.LazyFrame,
                              ed_lf: pl.LazyFrame,
                              fill: float = 0.0,
                              ) -> pl.LazyFrame:
    """Resolve ``pdX[e, param, d]`` cascade in lazy form.

    Mirror of :func:`._lifetime_inputs._resolve_per_period_lf`.
    """
    # Align ``per_param``'s ``e`` / ``d`` dtypes to ``ed_lf``'s before
    # joining — ``per_param`` was built from a String-typed CSV read
    # while ``ed_lf`` may now carry Enum-cast columns.
    ed_schema = ed_lf.collect_schema()
    e_dt = ed_schema.get("e", pl.Utf8)
    d_dt = ed_schema.get("d", pl.Utf8)
    if e_dt != pl.Utf8 or d_dt != pl.Utf8:
        per_param = per_param.with_columns(
            pl.col("e").cast(e_dt, strict=False),
            pl.col("d").cast(d_dt, strict=False),
        )
    explicit = (per_param
                  .filter(~pl.col("is_scalar"))
                  .select("e", "d", pl.col("value").alias("v_explicit")))
    scalar = (per_param
                .filter(pl.col("is_scalar"))
                .select("e", pl.col("value").alias("v_scalar")))
    return (ed_lf
              .join(explicit, on=["e", "d"], how="left")
              .join(scalar, on="e", how="left")
              .with_columns(
                  value=pl.coalesce(
                      pl.col("v_explicit"),
                      pl.col("v_scalar"),
                      pl.lit(fill, dtype=pl.Float64),
                  ),
              )
              .drop("v_explicit", "v_scalar"))


# ---------------------------------------------------------------------------
# Slice H — construction lead time: commissioning-year helper
# ---------------------------------------------------------------------------

# Two-level default: an entity with a lead time but no explicit method
# commissions at the closest seam; with no lead time it is immediate.
_CONSTRUCTION_METHOD_DEFAULT_LAG = "closest_seam"
_CONSTRUCTION_METHOD_IMMEDIATE = "immediate"


def _construction_method_with_default_lf(source: "InputSource",
                                             lead_lf: pl.LazyFrame,
                                             ) -> pl.LazyFrame:
    """Lazy ``[e, d, L, method]`` — resolve ``construction_lead_time_method``
    with the two-level default (Slice H design §2.4 step 2 / §6.2).

    ``lead_lf`` is the per-(e, d) resolved lead time ``[e, d, L]``.  An
    explicit method (per entity) always wins.  Where unset, the method
    resolves to ``immediate`` when ``L == 0`` and ``closest_seam`` when
    ``L > 0`` — so existing (L=0) models stay byte-identical while a set
    lead time enables the lag without a mandatory method entry.  The
    resolution is per-row because ``L`` may be period-varying (a Map).
    """
    explicit = (_entity_method_lf(source, "construction_lead_time_method")
                    .filter(pl.col("method").is_not_null())
                    .unique(subset=["e"]))
    # Align the explicit-method ``e`` (String from CSV) to lead_lf.e's
    # dtype (may be Enum on the post-load path) before the join.
    e_dt = lead_lf.collect_schema().get("e", pl.Utf8)
    if e_dt != pl.Utf8:
        explicit = explicit.with_columns(pl.col("e").cast(e_dt, strict=False))
    return (lead_lf
              .join(explicit, on="e", how="left")
              .with_columns(
                  method=pl.when(pl.col("method").is_not_null())
                           .then(pl.col("method"))
                           .when(pl.col("L") > 0.0)
                           .then(pl.lit(_CONSTRUCTION_METHOD_DEFAULT_LAG))
                           .otherwise(pl.lit(_CONSTRUCTION_METHOD_IMMEDIATE)),
              ))


def _solve_years_represented(source: "InputSource", active_solve: str | None) -> dict[str, float] | None:
    """``{period: years_represented}`` of ``solve.years_represented`` for the active solve; ``None`` without it."""
    if active_solve is None:
        return None
    from flextool.engine.input._param_shapes import solve_years_represented
    try:
        df = solve_years_represented(source)
    except (KeyError, AttributeError):
        return None
    if df is None or df.height == 0:
        return None
    sub = df.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        return None
    out: dict[str, float] = {}
    for d, v in sub.select(pl.col("period").cast(pl.Utf8), "value").iter_rows():
        try:
            out[str(d)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def lead_time_decisions(source: "InputSource", anchor_ed_lf: pl.LazyFrame) -> pl.DataFrame:
    """``[e, d, L, method]`` (Utf8 keys): the decisions of ``anchor_ed_lf`` with a construction lead time > 0 and a
    method other than ``immediate``."""
    lead_lf = (_resolve_per_period_lf(
                    _per_entity_param_lf(source, "construction_lead_time"),
                    anchor_ed_lf, fill=0.0)
                 .rename({"value": "L"})
                 .select("e", "d", "L"))
    return (_construction_method_with_default_lf(source, lead_lf)
              .filter((pl.col("L") > 0.0) & (pl.col("method") != _CONSTRUCTION_METHOD_IMMEDIATE))
              .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), "L", "method")
              .collect())


def _solve_walk_years(source: "InputSource",
                         active_solve: str | None,
                         workdir: Path | None = None,
                         provider: "object | None" = None,
                         ) -> dict[str, float] | None:
    """``{period: year}`` of the solve's own calendar, resolved as :func:`._walks.period_walk_iterator` resolves
    it (the same provider / workdir fallback), so these years are the ones the availability walk compares;
    ``None`` when the solve has no period years."""
    if provider is None:
        from flextool.engine.preprocess.params._derived import (
            _fan_walk_provider,
            _fan_walk_workdir,
        )
        provider = _fan_walk_provider()
        if provider is not None and workdir is None:
            workdir = _fan_walk_workdir()
    from flextool.engine.preprocess.params._derived import _p_years_d_lf
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        return None
    walk_yr: dict[str, float] = {}
    for d, y in pyd_lf.select(pl.col("d").cast(pl.Utf8), "yr").collect().iter_rows():
        if d is not None and y is not None:
            walk_yr.setdefault(str(d), float(y))
    return walk_yr


def commissioning_year_lf(source: "InputSource",
                              active_solve: str | None,
                              anchor_ed_lf: pl.LazyFrame,
                              period_in_use: list[str],
                              *,
                              workdir: Path | None = None,
                              provider: "object | None" = None,
                              run_calendar: "object | None" = None,
                              ) -> pl.LazyFrame | None:
    """Lazy ``[e, d, yr_c, d_c, L, yr_c_run]`` — the start of operation of each investment decision ``(e, d)``
    with an effective construction lead time, or ``None`` when no decision has one.

    ``d`` is the decision period ``o``.  The start of operation is ``t* = yr(o) + L`` rounded by
    ``construction_lead_time_method`` to a *seam* of the run calendar (every period start of the whole run and
    the end of the model horizon):

    * ``previous_seam`` — the last seam ``<= t*``;
    * ``next_seam``     — the first seam ``>= t*``;
    * ``closest_seam``  — the nearer of the two, a tie goes to the later seam.

    ``yr_c_run`` is the rounded start on the run calendar (``+inf`` when it is at or after the end of the model
    horizon: the capacity would never operate), ``d_c`` the run period that starts there (null for ``+inf``).
    ``yr_c`` is the same start on the solve's own year calendar (the one the availability walk compares
    against), ``+inf`` when ``d_c`` is not a period of the solve (the investment would start to operate after
    the solve).  Decisions that resolve to ``immediate`` or whose start rounds back onto ``o`` are left out, so a
    model without lead times gets ``None`` (the walks keep ``yr_d``, byte-identical).

    ``run_calendar`` is the :class:`._run_calendar.RunCalendar` of the run; without one (a solve built on its
    own) the solve's in-use periods and its ``years_represented`` stand in for it.

    The solve's year of ``o`` comes from :func:`._derived._p_years_d_lf` through the same provider / workdir
    resolution :func:`._walks.period_walk_iterator` uses, so a branch period's year lands on the calendar the
    walk compares against.  The solve calendar and the run calendar differ by a constant offset (a solve that is
    not handed the run history counts years from its own first period); a varying offset, or two solves giving
    a period different years, raises :class:`CommissioningLagError` on a model with a lead time.
    """
    from flextool.engine.input._solve_state import CommissioningLagError
    from flextool.engine.preprocess.params._run_calendar import (
        YEAR_TOL,
        snap_to_seam,
        solve_calendar,
        solve_year_offset,
    )

    if not period_in_use:
        return None
    # Whether the model configures a lag (some construction_lead_time > 0): a calendar problem is then an error
    # instead of a silent unlagged solve.
    lag_cfg = _construction_lag_configured(source)

    walk_yr = _solve_walk_years(source, active_solve, workdir, provider)
    if walk_yr is None:
        if lag_cfg:
            raise CommissioningLagError(
                "The start of operation of investments with a construction_lead_time cannot be found: the solve "
                f"'{active_solve}' has no period years.")
        return None
    in_use = [str(d) for d in period_in_use]
    in_use_yr = {d: walk_yr[d] for d in in_use if d in walk_yr}
    if lag_cfg and not in_use_yr:
        raise CommissioningLagError(
            "The start of operation of investments with a construction_lead_time cannot be found: no period of "
            f"solve '{active_solve}' has a year.")

    # Per-(e, o) lead time over the anchor grid, then the method; only decisions with a lead time can move.
    lead_df = lead_time_decisions(source, anchor_ed_lf)
    if lead_df.height == 0 or not in_use_yr:
        return None

    rc = run_calendar
    if rc is None:
        rc = solve_calendar(in_use_yr, _solve_years_represented(source, active_solve))
    if rc.conflicts:
        period, s_a, y_a, s_b, y_b = rc.conflicts[0]
        raise CommissioningLagError(
            f"Construction lead times need one year for each period, but solve '{s_a}' puts period '{period}' "
            f"{y_a:g} years and solve '{s_b}' {y_b:g} years from the start of the model.  Give the solves the "
            f"same years_represented for the periods they share.")
    offset, mismatch = solve_year_offset(rc, in_use_yr)
    if mismatch is not None:
        raise CommissioningLagError(
            f"Solve '{active_solve}' puts period '{mismatch}' {in_use_yr[mismatch]:g} years after its start, which "
            "does not match the run calendar.  Construction lead times need one year for each period.")
    if offset is None:
        raise CommissioningLagError(
            f"The periods of solve '{active_solve}' are not on the run calendar; the start of operation of "
            "investments with a construction_lead_time cannot be found.")
    seams = rc.seams()
    horizon_end = rc.horizon_end
    inf = float("inf")

    rows: list[tuple[str, str, float, str | None, float, float]] = []
    for e, o, lead, method in lead_df.iter_rows():
        yr_o_run = walk_yr.get(o, 0.0) + offset
        yr_c_run = snap_to_seam(yr_o_run + float(lead), str(method), seams)
        if abs(yr_c_run - yr_o_run) <= YEAR_TOL:
            continue  # rounds back onto the decision period: no lag
        if horizon_end is not None and yr_c_run >= horizon_end - YEAR_TOL:
            yr_c_run = inf
        d_c = None if yr_c_run == inf else rc.period_at(yr_c_run)
        yr_c = inf
        if yr_c_run != inf:
            for y in in_use_yr.values():
                if abs(y + offset - yr_c_run) <= YEAR_TOL:
                    yr_c = y
                    break
        rows.append((e, o, yr_c, d_c, float(lead), yr_c_run))
    if not rows:
        return None
    out = pl.DataFrame(rows, schema={"e": pl.Utf8, "d": pl.Utf8, "yr_c": pl.Float64, "d_c": pl.Utf8,
                                     "L": pl.Float64, "yr_c_run": pl.Float64}, orient="row")
    return out.lazy()


def _construction_lag_configured(source: "InputSource") -> bool:
    """True when some entity carries a ``construction_lead_time > 0``: the model configures a lag.

    Decides whether a :func:`commissioning_year_lf` failure must be loud (a lag-configured model) or is
    harmless (no lag anywhere: the helper returns ``None`` and the walk keeps ``yr_d``).  Any
    ``construction_lead_time > 0`` counts, even with an explicit ``immediate`` method, so a misconfiguration is
    never masked.
    """
    try:
        lead = _per_entity_param_lf(source, "construction_lead_time")
        return bool(
            lead.select((pl.col("value") > 0.0).any().alias("has"))
                .collect()
                .item())
    except Exception:  # pragma: no cover — defensive: absence == no lag
        return False


def offered_invest_set(ed_invest: pl.DataFrame,
                         commission_lf: pl.LazyFrame | None,
                         ) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """``(offered, not_offered)``: ``ed_invest`` without the decisions whose capacity would start to operate
    after the solve (``yr_c = +inf``), and those decisions (``[e, d, d_c, L, yr_c_run]``; ``None`` when every
    decision is offered).  Without a lead time ``ed_invest`` is returned unchanged."""
    if commission_lf is None or ed_invest is None or ed_invest.height == 0:
        return ed_invest, None
    late = (commission_lf.filter(pl.col("yr_c").is_infinite())
                         .select("e", "d", "d_c", "L", "yr_c_run").collect())
    if late.height == 0:
        return ed_invest, None
    key = pl.concat_str([pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)], separator="\x1f")
    late_keys = late.select(key.alias("k"))["k"]
    offered = ed_invest.filter(~key.is_in(late_keys.implode()))
    return offered, late


def warn_investments_not_offered(source: "InputSource",
                                     active_solve: str | None,
                                     not_offered: pl.DataFrame | None,
                                     period_in_use: list[str],
                                     run_calendar: "object | None" = None,
                                     ) -> None:
    """Warn (W-0524 / W-0525, one line per solve) about the entities whose investment decisions are not offered
    because the capacity would start to operate after the end of the solve or of the model horizon.  The debug log
    names each entity's lead time, decision periods and the horizon."""
    if not_offered is None or not_offered.height == 0:
        return
    from flextool import run_warnings

    classes = {str(e): str(ec) for e, ec in _entity_class_lf(source)
               .select(pl.col("e").cast(pl.Utf8), "ec").collect().iter_rows()}
    model_end = getattr(run_calendar, "last_period", None)
    solve_end = str(period_in_use[-1]) if period_in_use else "?"
    found: dict[str, dict[str, str]] = {"W-0524": {}, "W-0525": {}}
    for (e,), grp in not_offered.sort("e", "d").group_by(["e"], maintain_order=True):
        e = str(e)
        cls = classes.get(e, "entity")
        for after_model, part in ((True, grp.filter(pl.col("yr_c_run").is_infinite())),
                                  (False, grp.filter(~pl.col("yr_c_run").is_infinite()))):
            if part.height == 0:
                continue
            leads = sorted({float(v) for v in part["L"].to_list()})
            lead_txt = " / ".join(f"{v:g}" for v in leads)
            periods = ", ".join(str(d) for d in part["d"].to_list())
            if after_model:
                horizon = f"the end of period {model_end}" if model_end else "the end of the model horizon"
                found["W-0524"][e] = (f"{cls}: construction_lead_time {lead_txt} years; investments decided in "
                                      f"{periods} would start to operate after {horizon}")
            else:
                found["W-0525"][e] = (f"{cls}: construction_lead_time {lead_txt} years; investments decided in "
                                      f"{periods} would start to operate after {solve_end}")
    for code, details in found.items():
        if details:
            run_warnings.warn_grouped(code, list(details), solve=active_solve, details=details)


def lead_time_sets(source: "InputSource",
                     active_solve: str | None,
                     ed_invest: pl.DataFrame,
                     edd_invest: pl.DataFrame | None,
                     commission_lf: pl.LazyFrame,
                     period_in_use: list[str],
                     workdir: Path | None = None,
                     *,
                     lineage: pl.DataFrame | None = None,
                     ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(edd_invest_commission, edd_invest_in_construction)``, both ``[e, d_invest, d]`` (Utf8), of a solve with
    construction lead times.

    * commission: the period ``d`` in which the capacity of decision ``(e, d_invest)`` starts to operate — the
      periods of ``edd_invest`` at the start year ``yr_c`` (the branch periods of that year along the decision's
      lineage under recourse).  A decision without a lead time starts in its own period: ``(e, o, o)``.
    * in construction: the periods ``d`` with ``yr(d_invest) <= yr(d) < yr_c`` of the decisions with a lead time,
      lineage-filtered like the availability walk.

    ``commission_lf`` is the frame of :func:`commissioning_year_lf` (offered decisions keep a finite ``yr_c``).
    The period years are those of the in-construction walk (the boundary's walker Provider), so a branch period has
    the year of its period label under every ``stochastic_invest_method``.
    """
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL

    schema = {"e": pl.Utf8, "d_invest": pl.Utf8, "d": pl.Utf8}
    lag = (commission_lf.filter(pl.col("yr_c").is_finite())
                        .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8).alias("d_invest"), "yr_c")
                        .collect())
    walk_yr = _solve_walk_years(source, active_solve, workdir) or {}
    years = pl.DataFrame({"d": list(walk_yr), "yr_d": list(walk_yr.values())},
                         schema={"d": pl.Utf8, "yr_d": pl.Float64})
    ed = ed_invest.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8).alias("d_invest")).unique()
    plain = (ed.join(lag.select("e", "d_invest"), on=["e", "d_invest"], how="anti")
               .with_columns(d=pl.col("d_invest")))
    if edd_invest is not None and edd_invest.height > 0:
        moved = (edd_invest.select(pl.col(c).cast(pl.Utf8) for c in ("e", "d_invest", "d"))
                 .join(lag, on=["e", "d_invest"], how="inner")
                 .join(years, on="d", how="left")
                 .filter((pl.col("yr_d") - pl.col("yr_c")).abs() <= YEAR_TOL)
                 .select("e", "d_invest", "d"))
    else:
        moved = pl.DataFrame(schema=schema)
    commission = (pl.concat([plain.select("e", "d_invest", "d"), moved], how="vertical")
                    .unique().sort("e", "d_invest", "d"))

    in_construction = pl.DataFrame(schema=schema)
    if lag.height > 0:
        anchor = lag.select("e", pl.col("d_invest").alias("d"))
        anchor = anchor.join(years.rename({"yr_d": "yr_o"}), on="d", how="left")
        life_lf = (lag.rename({"d_invest": "d"})
                      .join(years.rename({"yr_d": "yr_o"}), on="d", how="left")
                      # The window ends just before the start of operation (YEAR_TOL keeps that period out).
                      .select("e", "d", (pl.col("yr_c") - pl.col("yr_o").fill_null(0.0) - YEAR_TOL).alias("life"))
                      .lazy())
        walk = period_walk_iterator(
            source, active_solve, anchor.select("e", "d").lazy(),
            [str(d) for d in period_in_use], [str(d) for d in period_in_use],
            window_method=WindowMethod.BOUNDED, life_lf=life_lf, factor_side=None,
            workdir=workdir, lineage=lineage)
        in_construction = (walk.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8).alias("d_invest"),
                                       pl.col("d_all").cast(pl.Utf8).alias("d"))
                               .unique().sort("e", "d_invest", "d").collect())
    return commission, in_construction


def carried_lead_time_capacity(source: "InputSource",
                                  active_solve: str | None,
                                  realized_commission: pl.DataFrame | None,
                                  period_in_use: list[str],
                                  workdir: Path | None = None,
                                  run_calendar: "object | None" = None,
                                  provider: "object | None" = None,
                                  ) -> tuple["Param | None", "Param | None"]:
    """``(carried_commission, carried_in_construction)`` Params ``[e, d]`` (MW, MWh for storage) of a solve of a
    chain: the investments that earlier solves realized with a construction lead time, by the period of this solve
    in which they start to operate, and by the periods in which they are still in construction
    (``yr(decision) <= yr(d) < yr(start)``).

    ``realized_commission`` is the handoff carrier ``[entity, period, commission_period, value]``; the years are
    those of the run calendar (the solve's own years shifted onto it; a branch period has the year of its period
    label).  ``(None, None)`` without carried investments in this solve's periods.
    """
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL

    if realized_commission is None or realized_commission.height == 0 or not period_in_use:
        return None, None
    from flextool.engine.preprocess.params._derived import _fan_walk_scope_provider
    walk_yr = _solve_walk_years(source, active_solve, workdir,
                                _fan_walk_scope_provider(workdir, provider=provider)) or {}
    rc_years = dict(getattr(run_calendar, "years", None) or {})
    offset = 0.0
    if rc_years:
        shared = [d for d in walk_yr if d in rc_years]
        if shared:
            offset = rc_years[shared[0]] - walk_yr[shared[0]]

    def run_year(d: str) -> float | None:
        if d in rc_years:
            return rc_years[d]
        if d in walk_yr:
            return walk_yr[d] + offset
        return None

    in_use = [(str(d), run_year(str(d))) for d in period_in_use]
    comm: dict[tuple[str, str], float] = {}
    cons: dict[tuple[str, str], float] = {}
    for e, o, c, v in realized_commission.select(
            pl.col("entity").cast(pl.Utf8), pl.col("period").cast(pl.Utf8),
            pl.col("commission_period").cast(pl.Utf8),
            pl.col("value").cast(pl.Float64, strict=False)).iter_rows():
        if v is None or v == 0.0:
            continue
        yr_o, yr_c = run_year(o), run_year(c)
        if yr_o is None or yr_c is None:
            continue
        for d, yr_d in in_use:
            if yr_d is None:
                continue
            if abs(yr_d - yr_c) <= YEAR_TOL:
                comm[(e, d)] = comm.get((e, d), 0.0) + v
            elif yr_o - YEAR_TOL <= yr_d < yr_c - YEAR_TOL:
                cons[(e, d)] = cons.get((e, d), 0.0) + v

    def as_param(rows: dict[tuple[str, str], float]) -> "Param | None":
        if not rows:
            return None
        df = pl.DataFrame([(e, d, v) for (e, d), v in sorted(rows.items())],
                          schema={"e": pl.Utf8, "d": pl.Utf8, "value": pl.Float64}, orient="row")
        return Param(("e", "d"), df)

    return as_param(comm), as_param(cons)


# ---------------------------------------------------------------------------
# §3.7.0 — entityInvest / entityDivest / e_invest_total / e_divest_total
# ---------------------------------------------------------------------------


def entity_invest_set_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``entityInvest``.

    Mirrors flextool's
    ``preprocessing/invest_method_sets.py::write_invest_method_sets``::

        entityInvest = {e : ∃m s.t. (e, m) ∈ entity__invest_method
                                AND m ∉ invest_method_not_allowed}

    The Δ.5 ``ed_*``-derivation trap: an entity may have an
    ``invest_method`` row but no ``ed_invest_set`` row (e.g. its
    no_reinvestment window has ended, or the solve has no investment
    period).  Such an entity is in ``entityInvest`` but not in any
    ``ed_invest`` (e, d) pair.  Cluster A's
    ``ed_lifetime_fixed_cost_divest`` produces rows for these entities;
    deriving from ``ed_*`` would miss them.
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    return (methods_lf
              .filter(~pl.col("method").is_in(list(_INVEST_NOT_ALLOWED)))
              .select("e")
              .unique())


def entity_divest_set_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``entityDivest``.  Symmetric of
    :func:`entity_invest_set_lf` with the divest-not-allowed filter."""
    methods_lf = _entity_method_lf(source, "invest_method")
    return (methods_lf
              .filter(~pl.col("method").is_in(list(_DIVEST_NOT_ALLOWED)))
              .select("e")
              .unique())


def e_invest_total_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``e_invest_total``.

    Mirrors ``preprocessing/invest_total_sets.py:74-90``:

        e_invest_total = entityInvest filtered by invest_method ∈
                              {invest_total, invest_period_total,
                               invest_retire_total,
                               invest_retire_period_total}
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    invest_universe = entity_invest_set_lf(source)
    has_total = (methods_lf
                    .filter(pl.col("method").is_in(list(_INVEST_TOTAL_METHODS)))
                    .select("e").unique())
    return invest_universe.join(has_total, on="e", how="inner")


def e_divest_total_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e]`` for ``e_divest_total``.  Symmetric of
    :func:`e_invest_total_lf` with the retire-total method filter."""
    methods_lf = _entity_method_lf(source, "invest_method")
    divest_universe = entity_divest_set_lf(source)
    has_total = (methods_lf
                    .filter(pl.col("method").is_in(list(_DIVEST_TOTAL_METHODS)))
                    .select("e").unique())
    return divest_universe.join(has_total, on="e", how="inner")



# ---------------------------------------------------------------------------
# §3.7.1 — pd_invest_set / nd_invest_set partitions
# ---------------------------------------------------------------------------


def _entity_class_partition_lf(source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[e, kind]`` where ``kind ∈ {process, node}``.

    process = unit ∪ connection (processes); node = node.  Disjoint
    classes — every entity falls in exactly one.
    """
    parts: list[pl.LazyFrame] = []
    for ec, kind in (("unit", "process"),
                       ("connection", "process"),
                       ("node", "node")):
        try:
            df = source.entities(ec)
        except KeyError:
            continue
        if df.height == 0:
            continue
        parts.append(df.lazy().select(
            alias_to_axis("name", "e"),
            pl.lit(kind).alias("kind"),
        ))
    if not parts:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "kind": pl.Utf8,
        })
    return pl.concat(parts, how="vertical").unique()


def pd_invest_set_lf(source: "InputSource",
                            ed_invest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[p, d]`` — ``ed_invest_set`` partitioned to processes
    (``unit ∪ connection``).  Mirror of
    ``invest_divest_sets.py:215-217``.
    """
    cls_lf = _entity_class_partition_lf(source)
    return (ed_invest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "process")
              .pipe(rename_to_axis, {"e": "p"})
              .select("p", "d")
              .unique()
              .sort("p", "d"))


def _storage_nodes_only(nd_lf: pl.LazyFrame, source: "InputSource") -> pl.LazyFrame:
    """``nd_lf`` restricted to the nodes with ``node_type`` ``storage`` (``nodeState``)."""
    from flextool.engine.preprocess.params._projection import nodeState

    state = nodeState(source)
    n_dtype = nd_lf.collect_schema().get("n")
    return nd_lf.join(state.lazy().select(pl.col("n").cast(n_dtype)), on="n", how="semi")


def nd_invest_set_lf(source: "InputSource",
                            ed_invest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[n, d]`` — ``ed_invest_set`` partitioned to storage nodes (``nodeState``).

    Node capacity is storage capacity (``state_unitsize``): a non-storage node with an
    ``invest_method`` gets no variable (``validate_node_invest_without_storage`` warns).
    """
    cls_lf = _entity_class_partition_lf(source)
    return (_storage_nodes_only(ed_invest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "node")
              .pipe(rename_to_axis, {"e": "n"})
              .select("n", "d"), source)
              .unique()
              .sort("n", "d"))


def pd_divest_set_lf(source: "InputSource",
                            ed_divest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[p, d]`` — ``ed_divest_set`` partitioned to processes."""
    cls_lf = _entity_class_partition_lf(source)
    return (ed_divest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "process")
              .pipe(rename_to_axis, {"e": "p"})
              .select("p", "d")
              .unique()
              .sort("p", "d"))


def nd_divest_set_lf(source: "InputSource",
                            ed_divest_lf: pl.LazyFrame,
                            ) -> pl.LazyFrame:
    """Lazy ``[n, d]`` — ``ed_divest_set`` partitioned to storage nodes (``nodeState``).

    Node capacity is storage capacity (``state_unitsize``): a non-storage node with an
    ``invest_method`` gets no variable (``validate_node_invest_without_storage`` warns).
    """
    cls_lf = _entity_class_partition_lf(source)
    return (_storage_nodes_only(ed_divest_lf
              .join(cls_lf, on="e", how="inner")
              .filter(pl.col("kind") == "node")
              .pipe(rename_to_axis, {"e": "n"})
              .select("n", "d"), source)
              .unique()
              .sort("n", "d"))


def ed_invest_period_set_lf(source: "InputSource",
                                  ed_invest_lf: pl.LazyFrame,
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d]`` — ``ed_invest`` filtered to entities whose
    ``invest_method`` is in :data:`_INVEST_PERIOD_METHODS`.

    Mirror of ``preprocessing/invest_divest_sets.py`` 's
    ``ed_invest_period`` set: keeps only the (e, d) pairs that carry
    a per-period invest cap (``maxInvest_entity_period`` /
    ``minInvest_entity_period`` constraint indices).  Used in
    ``model._eq_invest_period_caps`` to index the per-period invest cap rows.
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    period_e = (methods_lf
                   .filter(pl.col("method").is_in(list(_INVEST_PERIOD_METHODS)))
                   .select("e")
                   .unique())
    return (ed_invest_lf
              .join(period_e, on="e", how="inner")
              .select("e", "d")
              .unique()
              .sort("e", "d"))


def ed_divest_period_set_lf(source: "InputSource",
                                  ed_divest_lf: pl.LazyFrame,
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d]`` — ``ed_divest`` filtered to entities whose
    ``invest_method`` is in :data:`_DIVEST_PERIOD_METHODS`.  Mirror of
    :func:`ed_invest_period_set_lf` for divest.
    """
    methods_lf = _entity_method_lf(source, "invest_method")
    period_e = (methods_lf
                   .filter(pl.col("method").is_in(list(_DIVEST_PERIOD_METHODS)))
                   .select("e")
                   .unique())
    return (ed_divest_lf
              .join(period_e, on="e", how="inner")
              .select("e", "d")
              .unique()
              .sort("e", "d"))


# ---------------------------------------------------------------------------
# §3.11 — p_entity_all_existing (existing-chain walk)
# ---------------------------------------------------------------------------


def p_entity_retired_carried_from_provider(workdir: Path | None,
                                           *,
                                           provider: "object | None" = None,
                                           ) -> "Param | None":
    """``p_entity_retired_carried`` Param ``[e, d]`` (MW, MWh for storage): the retirements of the existing cohort
    that earlier solves of a chain realized and that still count in the solve's periods, as the solve emitted them
    (``solve_data/p_entity_retired_carried``, ``_chain_params.emit_existing_capacity``), sorted.  ``None`` without
    carried retirements (or without the frame: a solve built on its own)."""
    from flextool.engine.preprocess.params._derived import _read_handoff_e_d_from_workdir
    if workdir is None:
        return None
    param = _read_handoff_e_d_from_workdir(workdir, "p_entity_retired_carried", provider=provider)
    if param is None:
        return None
    return Param(("e", "d"), param.frame.sort("e", "d"))


def p_entity_pre_existing_lf(source: "InputSource",
                                  active_solve: str | None,
                                  period_in_use: list[str],
                                  *,
                                  workdir: Path | None = None,
                                  provider: "object | None" = None,
                                  ) -> pl.LazyFrame:
    """Lazy ``[e, d, value]`` for ``p_entity_pre_existing`` —
    lifetime-gated ``entity.existing × virtual_unitsize`` cascade.

    Algorithm (mirror of
    ``preprocessing/entity_period_calc_params.py:write_p_entity_pre_existing``):

      For each entity e:
        v_existing = pdProcess[e, 'existing', d] (if e ∈ process)
                   else pdNode[e, 'existing', d] (if e ∈ node)
                   else 0
        v_unit = p_process[e, 'virtual_unitsize'] (if e ∈ process)
                 else p_node[e, 'virtual_unitsize'] (if e ∈ node)
                 else 0
        v = v_existing * v_unit  if v_unit > 0 else v_existing
            in the periods its existing cohort is in service (y[d] < y_eol, :mod:`._cohorts`:
            always for reinvest_automatic; until the end of life counted from the start of the run for
            reinvest_choice / no_reinvestment); 0 thereafter, and for an entity without a lifetime_method.

    The ``v_existing × v_unit`` term is the canonical capacity (a
    counted entity's ``existing`` is the count, not the capacity).
    Only applied when ``v_unit > 0``: a ``virtual_unitsize`` of 0 or below
    is ignored (``validate_basic_sanity`` warns about it), so ``existing``
    is then the capacity itself.
    """
    if not period_in_use:
        return pl.LazyFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
            "value": pl.Float64,
        })
    all_e_lf = _all_entities_lf(source)
    piu_lf = (pl.LazyFrame({"d": period_in_use})
                .with_columns(pl.col("d").cast(schema_dtype(_enums, "d"),
                                                  strict=False)))
    grid_lf = all_e_lf.join(piu_lf, how="cross")

    # Resolve existing per (e, d): explicit row, else the entity's constant, else 0.
    ex_per = _per_entity_param_lf(source, "existing")
    grid_with_ex = (_resolve_per_period_lf(ex_per, grid_lf, fill=0.0)
                       .rename({"value": "ex"}))

    # virtual_unitsize per entity (scalar across periods).
    vu_per = _per_entity_param_lf(source, "virtual_unitsize")
    # Aggregate to (e, vu) — virtual_unitsize is non-period for
    # process / node (per write_p_entity_pre_existing's read of
    # p_process / p_node directly, NOT pd_process / pd_node).
    vu_scalar = (vu_per
                    .filter(pl.col("is_scalar"))
                    .select("e", pl.col("value").alias("vu"))
                    .unique(subset=["e"], keep="last"))
    grid_with_vu = (grid_with_ex
                       .join(vu_scalar, on="e", how="left")
                       .with_columns(vu=pl.col("vu").fill_null(0.0)))

    # The existing cohort of each entity: in service while y[d] < y_eol.
    from flextool.engine.preprocess.params._cohorts import cohort_years, existing_cohorts, within_life
    years = cohort_years(source, active_solve, workdir, provider=provider)
    ends = existing_cohorts(source, active_solve, workdir, provider=provider, years=years).lazy().select("e", "y_eol")
    periods = list(dict.fromkeys(str(d) for d in period_in_use))
    period_years = (pl.LazyFrame({"d": periods, "y": [float(years.get(d, 0.0)) for d in periods]},
                                 schema={"d": pl.Utf8, "y": pl.Float64})
                      .with_columns(pl.col("d").cast(schema_dtype(_enums, "d"), strict=False)))
    grid_with_vu, ends = align_join_dtypes(grid_with_vu, ends, ["e"])

    # Compose: value = ex × vu (or ex if vu ≤ 0: ignored) while in service; 0 else.
    capacity_expr = (pl.when(pl.col("vu") > 0.0)
                          .then(pl.col("ex") * pl.col("vu"))
                          .otherwise(pl.col("ex")))
    return (grid_with_vu
              .join(period_years, on="d", how="left", maintain_order="left")
              .join(ends, on="e", how="left", maintain_order="left")
              .with_columns(
                  value=pl.when(within_life(pl.col("y"), None, pl.col("y_eol")))
                              .then(capacity_expr)
                              .otherwise(0.0),
              )
              .select("e", "d", "value"))


# ---------------------------------------------------------------------------
# Public entry: apply boundary
# ---------------------------------------------------------------------------


def apply_existing_chain(flex_data: object,
                              source: "InputSource",
                              workdir: Path,
                              *,
                              handoff: object | None = None,
                              ctx: "SolveContext | None" = None,
                              provider: "object | None" = None) -> None:
    """Set the chain-carried constants of ``flex_data`` (mutates in place):

    * ``p_entity_retired_carried`` — the carried retirements of the existing cohort, as the solve emitted them
      (:func:`p_entity_retired_carried_from_provider`).  ``p_entity_all_existing`` itself is read from the
      emitted ``solve_data/p_entity_all_existing`` (``_derived.p_entity_all_existing_from_source`` and the
      synthetic sub-solves' seed), the one value every consumer reads.
    * ``p_entity_carried_commission`` / ``p_entity_carried_in_construction`` — the investments earlier solves
      realized with a construction lead time, coming online / in construction in the solve's periods (handoff
      ``realized_commission``; ``handoff`` is the fallback when the Provider has none).
    """
    from flextool.engine.preprocess.params._derived import _read_active_solve, _period_in_use_set

    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    period_in_use = _period_in_use_set(source, active_solve, workdir, ctx=ctx, provider=provider)
    flex_data.p_entity_retired_carried = p_entity_retired_carried_from_provider(workdir, provider=provider)

    realized_commission = None
    if provider is not None:
        from flextool.engine.preprocess import _provider_keys as K
        from flextool.engine.preprocess._provider_translators import read_handoff_frame
        realized_commission = read_handoff_frame(provider, K.HANDOFF_REALIZED_COMMISSION)
    if (realized_commission is None or realized_commission.height == 0) and handoff is not None:
        realized_commission = getattr(handoff, "realized_commission", None)
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    carried_commission, carried_in_construction = carried_lead_time_capacity(
        source, active_solve, realized_commission, period_in_use, workdir,
        run_calendar=read_run_calendar(provider), provider=provider)
    flex_data.p_entity_carried_commission = carried_commission
    flex_data.p_entity_carried_in_construction = carried_in_construction


__all__ = [
    # Method-enum constants (re-exported for direct-test use).
    "INVEST_METHODS_ALL", "INVEST_NOT_ALLOWED", "DIVEST_NOT_ALLOWED",
    "GROUP_INVEST_NOT_ACTIVE", "GROUP_DIVEST_NOT_ACTIVE",
    "_INVEST_NOT_ALLOWED", "_DIVEST_NOT_ALLOWED",
    "_INVEST_TOTAL_METHODS", "_DIVEST_TOTAL_METHODS",
    "_INVEST_PERIOD_METHODS", "_DIVEST_PERIOD_METHODS",
    # Public lazy helpers — entity sets.
    "entity_invest_set_lf", "entity_divest_set_lf",
    "e_invest_total_lf", "e_divest_total_lf",
    # Public lazy helpers — partitions.
    "pd_invest_set_lf", "nd_invest_set_lf",
    "pd_divest_set_lf", "nd_divest_set_lf",
    "ed_invest_period_set_lf", "ed_divest_period_set_lf",
    # Public lazy helpers — existing chain.
    "p_entity_pre_existing_lf", "p_entity_retired_carried_from_provider",
    "apply_existing_chain",
]
