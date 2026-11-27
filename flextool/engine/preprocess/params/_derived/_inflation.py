"""Inflation and the year cascade (§3.1.3, §3.7.5/6): ``p_inflation_op``, its multi-year cascade and the
per-period years, ``periodAll`` and the scalar inflation inputs."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.input._param_shapes import solve_years_represented

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import _provider_has_key, _provider_read, _try_param
from flextool.engine.preprocess.params._derived._periods import _solve_periods


# ---------------------------------------------------------------------------
# §3.1.3 — p_inflation_op
# ---------------------------------------------------------------------------


def p_inflation_op_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
) -> Param | None:
    """Period-level inflation factor for operations.

    Audit §3.1.3 says the formula compounds ``(1 + social_discount_rate)^...``;
    the actual flextool implementation
    (``period_calculated_params.py:280-322``) computes
    ``sum_y p_years_represented[d,y] * (1+rate)^until_op[d,y]``
    where ``until_op[d, y] = base[d,y] + pyr[d,y] * offset_op``,
    ``base[d, y] = sum_{y'<y} pyr[d, y']``.

    Γ.8.E coverage:
      * ``model.social_discount_rate ∈ {None, 0}`` (the default) — formula
        collapses to ``ops_factor[d] = Σ_y pyr[d, y] * 1 =
        years_represented[solve, d]`` since each period has a single
        flat scalar in the 1D-Map shape exposed by InputSource (each
        scalar IS the total summed over the period's year-set).
      * Multi-year-per-period (``years_represented`` set per solve,
        rate=0) — handled here, mirrors the regression that
        ``work_fullYear_roll`` exposes when CSV retirement Step 3 lands.

    Non-trivial-rate (``rate != 0``) and ``discount_offset`` cascade
    are handled by :func:`p_inflation_op_multi_year_from_source`
    (Γ.3.C, applied in :func:`apply_derived_c`).

    Returns ``None`` only when no solve / no dt is available; the
    helper otherwise produces the canonical frame so CSV retirement
    won't reintroduce the simple-1.0 regression.
    """
    if dt is None:
        return None
    rate = _try_param(source, "model", "social_discount_rate")
    rate_v = 0.0
    if rate is not None and rate.height > 0:
        rate_v = float(rate["value"][0])
    if rate_v != 0.0:
        # Non-trivial rate handled in the multi-year cascade helper
        # (which runs later in apply_derived_c).  Returning None
        # here lets the simple-default 1.0 stay in flex_data; the
        # later helper overlays the correct factor.
        return None

    # Rate = 0.  ``ops_factor[d] = Σ_y pyr[d, y]``.  The 1D Map shape
    # InputSource exposes for ``solve.years_represented`` already
    # collapses the inner year axis into a per-period scalar — so
    # that scalar IS the sum we need.
    #
    # Rolling solves: ``active_solve`` in the workdir is the per-roll
    # name (e.g. ``dispatch_fullYear_roll_roll_71``) but the DB has
    # ``years_represented`` keyed by the parent solve
    # (``dispatch_fullYear_roll``).  flextool's preprocessing copies
    # the parent's value down to every roll; we mirror that here
    # by trying the roll name first, then the parent name (strip
    # ``_roll_<N>`` suffix).
    yr = None
    if active_solve is not None:
        yr_raw = solve_years_represented(source)
        if yr_raw is not None:
            candidate_names = [active_solve]
            import re
            parent = re.sub(r"_roll_\d+$", "", active_solve)
            if parent != active_solve:
                candidate_names.append(parent)
            yr_lf = None
            for cand in candidate_names:
                lf = (yr_raw.lazy()
                            .filter(pl.col("name") == cand)
                            .select(alias_to_axis("period", "d"),
                                    pl.col("value").cast(pl.Float64).alias("yr_total")))
                if lf.collect().height > 0:
                    yr_lf = lf
                    break
            if yr_lf is not None:
                d_unique_lf = dt.lazy().select("d").unique()
                from flextool.engine.input._axis_enums import align_join_dtypes
                d_unique_lf, yr_lf = align_join_dtypes(
                    d_unique_lf, yr_lf, ["d"])
                yr_joined = (d_unique_lf.join(yr_lf, on="d", how="left")
                                         .with_columns(
                                             value=pl.col("yr_total").fill_null(1.0)
                                         )
                                         .select("d", "value")
                                         .sort("d"))
                yr_collected = yr_joined.collect()
                if yr_collected.height > 0:
                    yr = yr_collected

    if yr is None:
        # Fallback: trivial 1.0 per realized period (no
        # ``years_represented`` row → defaults to 1 year per period).
        # Defensive re-cast: ensure d is canonical Enum on the Param.
        yr = (dt.lazy()
                .select(alias_to_axis("d", "d")).unique()
                .with_columns(value=pl.lit(1.0).cast(pl.Float64))
                .sort("d")
                .collect())
    if yr.height == 0:
        return None
    return Param(("d",), yr)


# ---------------------------------------------------------------------------
# §3.7.5/6 — Multi-year inflation cascade  (default-rate path only)
# ---------------------------------------------------------------------------
#
# Replaces the simple Γ.3.A ``p_inflation_op`` when ``model.social_discount_rate
# != 0`` AND the active solve carries non-trivial ``years_represented`` /
# ``period_timeset``.  The full cascade is documented at
# ``period_calculated_params.py:280-322``: for each period d, sum across
# (d, y) ∈ years_for_period, weighted by ``years_represented[d, y]`` of
# ``(1+r)^{-(base[d, y] + pyr * offset_op)}`` where base counts cumulative
# years across the GLOBAL year universe.
#
# Γ.3.C scope: produce the cascade for the simplest non-trivial case —
# every period has exactly one represented year and ``years_from_start``
# is supplied as an absolute year offset.  This collapses to
# ``inflation_op[d] = (1+r)^{-(yr[d] + offset_op)}``.  More general shapes
# (multi-year-per-period, missing offset) gate-fail and the CSV value
# survives.


def p_inflation_op_multi_year_from_source(source: "InputSource",
                                             active_solve: str,
                                             dt: pl.DataFrame,
                                             ) -> "Param | None":
    """Multi-year inflation factor for the default 1-year-per-period case.

    Returns None when:
      * ``model.social_discount_rate == 0`` (Γ.3.A's simple path covers it).
      * The active solve has no ``years_from_start`` parameter.
      * Any period has > 1 represented year (defer to CSV).
    """
    rate_df = _try_param(source, "model", "social_discount_rate")
    rate_v = 0.0
    if rate_df is not None and rate_df.height > 0:
        rate_v = float(rate_df["value"][0])
    if rate_v == 0.0:
        return None  # Γ.3.A's trivial path applies.
    offset_df = _try_param(source, "model", "discount_offset_operations")
    offset_v = 0.5
    if offset_df is not None and offset_df.height > 0:
        offset_v = float(offset_df["value"][0])

    yfs = _try_param(source, "solve", "years_from_start")
    if yfs is None or "period" not in yfs.columns:
        return None
    yfs_sub = yfs.filter(pl.col("name") == active_solve)
    if yfs_sub.height == 0:
        return None
    # Multi-year-per-period guard: if ``years_represented`` Map has
    # multiple (period, year) rows for any period, defer.
    yr = solve_years_represented(source)
    if yr is not None:
        yr_sub = yr.filter(pl.col("name") == active_solve)
        if yr_sub.height > 0:
            counts = (yr_sub.group_by("period").agg(pl.len().alias("n")))
            if counts["n"].max() and counts["n"].max() > 1:
                return None

    factor = 1.0 + rate_v
    if dt is None or dt.height == 0:
        return None
    yfs_lf = yfs_sub.lazy().select(
        alias_to_axis("period", "d"),
        pl.col("value").cast(pl.Float64).alias("yr"),
    )
    # Defensive re-cast: align d to canonical Enum so the left-join
    # against yfs_lf (Enum d) composes cleanly.
    out = (dt.lazy().select(alias_to_axis("d", "d")).unique()
              .join(yfs_lf, on="d", how="left")
              .with_columns(yr=pl.col("yr").fill_null(0.0))
              .with_columns(
                  value=pl.lit(factor).pow(-(pl.col("yr") + offset_v))
              )
              .select("d", "value")
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


# ===========================================================================
# Γ.3.F — Lifetime cascade + handoff state + multi-year inflation cascade
# ===========================================================================
#
# This batch lands the §3.1.3 multi-year inflation cascade in full
# (replacing Γ.3.A's simple-1-year-per-period subcase), the §3.7.5/6
# discounted lifetime-fixed-cost / annual-discounted-invest annuity
# families, and the §3.7.7/8 rolling-handoff Param read side.  It is
# the dependency-final piece of the §3.7 + §3.11 cohort.
#
# Per the architectural shift (no defensive feature gating), helpers
# return ``None`` only when the underlying source is missing or the
# fixture's invest/lifetime configuration is degenerate (no
# entityInvest, no lifetime_method, etc.).  When data is present, the
# helper produces the canonical frame; the parity test catches any
# discrepancy.
#
# Authoritative algorithm references:
#   * §3.1.3 (multi-year cascade): preprocessing/period_calculated_params.py:200-322.
#   * §3.7.5/6 (lifetime + annual_discounted): preprocessing/entity_annual_calc_params.py:105-348.
#   * §3.7.7/8 (handoff state): preprocessing/entity_period_calc_params.py:1525-1600.


# ---------------------------------------------------------------------------
# §3.1.3 — multi-year inflation cascade (full version)
# ---------------------------------------------------------------------------


def _solve_inflation_inputs(source: "InputSource") -> tuple[float, float, float]:
    """Read the three scalar inflation inputs from ``model``.

    Returns ``(rate, offset_invest, offset_ops)``.  Mirrors flextool's
    ``_scalar_max`` over ``p_inflation_*.csv``:
        * rate              → 0.0 when no explicit Spine value
        * offset_invest     → 0.0 when no explicit Spine value
        * offset_operations → 0.5 when no explicit Spine value

    flextool's input_writer skips Spine *defaults* (only writes
    explicit parameter rows to ``input/p_inflation_*.csv``), so the
    downstream cascade reads ``_scalar_max(... default)`` and falls back
    to the constant defaults documented above.  We mirror that: when
    the SpineDbReader returns only the broadcast-default row (no
    explicit per-entity row), we fall back to flextool's defaults.

    We can't tell from a returned frame whether a value came from
    explicit storage or default broadcast; we use ``parameter_default``
    plus row count to detect the all-default case.  When the default
    matches every row's value AND no explicit per-class row exists,
    treat as "no explicit" (use flextool's CSV default).
    """
    def _explicit_max(par: str, default: float) -> float:
        df = _try_param(source, "model", par)
        if df is None or df.height == 0:
            return default
        # Detect "all rows = Spine default broadcast" case.  When the
        # Spine reader's default-fill kicked in, every row's value
        # equals the parameter_default and the CSV writer would have
        # produced an empty file.  Fall back to flextool's constant.
        try:
            spine_default = source.parameter_default("model", par)
        except KeyError:
            spine_default = None
        if spine_default is not None:
            try:
                spine_default_f = float(spine_default)
            except (ValueError, TypeError):
                spine_default_f = None
            if spine_default_f is not None:
                # If every row equals the default, treat as no explicit data.
                vals = df["value"].cast(pl.Float64, strict=False).to_list()
                if vals and all(
                        (v is not None and abs(v - spine_default_f) < 1e-12)
                        for v in vals):
                    return default
        try:
            return float(df["value"].cast(pl.Float64).max())
        except Exception:
            return default

    rate = _explicit_max("social_discount_rate", 0.0)
    off_inv = _explicit_max("discount_offset_investment", 0.0)
    off_ops = _explicit_max("discount_offset_operations", 0.5)
    return rate, off_inv, off_ops


def _years_for_period_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    period_set: list[str],
                                    *,
                                    provider: "object | None" = None,
                                    ) -> dict[str, list[tuple[str, float]]]:
    """Derive the per-period ``(year_label, width)`` list emitted into
    ``solve_data/p_years_represented.csv`` (matches the
    ``emit_years_represented`` output line-for-line).

    Algorithm:
      For each (period, R) in solve.years_represented[active_solve],
      with R > 0:
        rows = ceil(R), remaining = R, year_count starts at solve-global 0.
        For i in range(rows):
          width = min(1.0, remaining)
          emit (period, str(year_count), width)
          year_count += width; remaining -= width

    Two arms.  With ``provider`` (canonical): read
    ``solve_data/p_years_represented.csv``, which also copies the
    anchor period's rows to its stochastic branch periods
    (``period__branch``).  Without it (or when the CSV is absent): the
    source arm below, which does NOT expand to branch periods (a branch
    period gets no rows).  ``p_years_represented_d`` and the recourse
    walker pass ``provider``; the inflation and NPV callers still use the
    source arm, so a branch period gets no years there.

    Returns a dict ``{period: [(year_label, width), ...]}``.  When the
    parameter is absent or active_solve has no rows, default to one row
    per period in the order supplied (width=1, year_label = "0", "1", …).
    """
    # The emitted ``p_years_represented.csv`` already byte-copies
    # fan-member year rows (``derive_years_represented`` branch loop), so
    # read it directly to give branch periods their anchor's
    # (year_label, width) rows.  Callers that pass ``provider=None`` get
    # the source path below.
    if provider is not None:
        p = Path("solve_data") / "p_years_represented.csv"
        if _provider_has_key(provider, p):
            df = _provider_read(provider, p)
            need = {"period", "years_from_solve", "p_years_represented"}
            if df.height > 0 and need.issubset(df.columns):
                canon: dict[str, list[tuple[str, float]]] = {}
                for r in df.iter_rows(named=True):
                    canon.setdefault(str(r["period"]), []).append(
                        (str(r["years_from_solve"]),
                         float(r["p_years_represented"])))
                if canon:
                    return canon

    yr_p = solve_years_represented(source)
    out: dict[str, list[tuple[str, float]]] = {}
    if active_solve is None or yr_p is None:
        # Default fallback — one year per period, width 1, integer-indexed.
        for i, d in enumerate(period_set):
            out[d] = [(str(int(i)), 1.0)]
        return out

    # Rolling solves: ``active_solve`` is the per-roll name
    # (``<parent>_roll_<N>``) but the DB has the parameter on the parent
    # solve.  flextool's preprocessing copies the parent value down to
    # each roll.  Try the active_solve first, then the parent name.
    sub = yr_p.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        import re
        parent = re.sub(r"_roll_\d+$", "", active_solve)
        if parent != active_solve:
            sub = yr_p.filter(pl.col("name") == parent)
    if sub.height == 0:
        for i, d in enumerate(period_set):
            out[d] = [(str(int(i)), 1.0)]
        return out

    # Iterate in CSV order — flextool walks the Spine Map order, which
    # the Spine reader presents row-by-row.  We sort by period to match
    # the canonical ordering when periods aren't already sorted in the
    # source — flextool's solve_writers iterates the dict in insertion
    # order, which mirrors the periodAll set ordering.  For our purposes
    # the period_set arg gives the canonical order.
    rows_by_d: dict[str, float] = {}
    for r in sub.iter_rows(named=True):
        rows_by_d[str(r["period"])] = float(r["value"])

    import math
    year_count = 0.0
    for d in period_set:
        R = rows_by_d.get(d, 0.0)
        if R <= 0:
            out[d] = []
            continue
        rows = int(math.ceil(R))
        remaining = R
        emitted: list[tuple[str, float]] = []
        for _i in range(rows):
            width = min(1.0, remaining)
            # Mirror solve_writers' int-cast path: when year_count is an
            # integer (R = whole numbers) we emit ``str(int(yc))``
            # literals; with fractional widths flextool writes the
            # float-stringified value.  Emulate the same:
            if year_count == float(int(year_count)):
                lbl = str(int(year_count))
            else:
                lbl = str(year_count)
            emitted.append((lbl, width))
            year_count += width
            remaining -= width
        out[d] = emitted
    return out


def _periodAll_from_source(source: "InputSource",
                              active_solve: str | None,
                              *,
                              provider: "object | None" = None,
                              workdir: Path | None = None) -> list[str]:
    """Compute the periodAll set for the active solve.

    Mirrors flextool's ``periodAll_set.csv`` shape.  When ``workdir``
    is provided AND the per-solve preprocessing has already written
    the file, prefer that (it's the authoritative cumulative-history
    union for chained solves).  Falls back to the simple
    realized ∪ invest union from the source when the file is absent.
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "periodAll_set.csv"
        df = None
        if provider is not None and provider.has(_provider_key(p)):
            df = provider.get(_provider_key(p))
        if df is not None and df.height > 0 and "period" in df.columns:
            return df["period"].cast(pl.Utf8, strict=False).to_list()
    seen: dict[str, None] = {}
    if active_solve is None:
        return []
    for par in ("realized_periods", "invest_periods"):
        ps = _solve_periods(source, active_solve, par) or []
        for d in ps:
            seen.setdefault(d, None)
    return list(seen.keys())


def _inflation_yearly_from_source(source: "InputSource",
                                     active_solve: str | None,
                                     workdir: Path | None = None,
                                     *,
                                     provider: "object | None" = None,
                                     ) -> tuple[dict[str, float],
                                                 dict[str, float]] | None:
    """Compute ``p_inflation_factor_investment_yearly[d]`` and
    ``p_inflation_factor_operations_yearly[d]`` per
    ``period_calculated_params.py:200-322``.

    Returns ``(invest_factor, ops_factor)`` dicts keyed on period.
    Returns ``None`` only when the periodAll set is empty.

    Algorithm:
      1. ``social_discount_rate``, ``discount_offset_investment`` / ``discount_offset_operations`` from model.
      2. For each period ``d`` ∈ periodAll, build ``years_for_period[d]``
         via :func:`_years_for_period_from_source`.
      3. Build the GLOBAL year set = union of all year labels across
         all periods, sort numerically (or lexically when non-numeric).
      4. For each period ``d``:
         a. Walk the global year set computing
            ``global_pos[y2] = sum_{y' < y2 in global order} pyr[d, y']``
            (inner accumulator restricted to the period's row).
         b. For each y in years_for_period[d]:
              base = global_pos[y]
              until_inv = base + pyr[d, y] * offset_invest
              until_ops = base + pyr[d, y] * offset_ops
         c. ``inv_factor[d] = sum_y pyr[d, y] * (1 + rate)^(-until_inv)``
            ``ops_factor[d] = sum_y pyr[d, y] * (1 + rate)^(-until_ops)``
            When ``sum_y pyr[d, y] == 0`` → factor = 1.0.
    """
    period_universe = _periodAll_from_source(source, active_solve, workdir=workdir, provider=provider)
    if not period_universe:
        return None
    rate, off_inv, off_ops = _solve_inflation_inputs(source)
    one_plus_inflation_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    yfp = _years_for_period_from_source(source, active_solve,
                                          period_universe)

    # Global year set — union of all year labels.
    global_year_set: dict[str, None] = {}
    for years_list in yfp.values():
        for y, _w in years_list:
            global_year_set.setdefault(y, None)
    try:
        sorted_global_years = sorted(global_year_set.keys(),
                                       key=lambda y: float(y))
    except ValueError:
        sorted_global_years = sorted(global_year_set.keys())

    inflation_invest: dict[str, float] = {}
    inflation_ops: dict[str, float] = {}

    # Build a fast lookup: pyr_lookup[(d, y)] = width.
    pyr_lookup: dict[tuple[str, str], float] = {}
    for d, years in yfp.items():
        for y, w in years:
            pyr_lookup[(d, y)] = w

    for d in period_universe:
        years_for_d = yfp.get(d, [])
        try:
            sorted_d_years = sorted(years_for_d, key=lambda yw: float(yw[0]))
        except ValueError:
            sorted_d_years = sorted(years_for_d, key=lambda yw: yw[0])

        # global_pos[y2] = sum_{y' in global, y' < y2} pyr[d, y'] (default 1)
        cumulative = 0.0
        global_pos: dict[str, float] = {}
        for y2 in sorted_global_years:
            global_pos[y2] = cumulative
            cumulative += pyr_lookup.get((d, y2), 1.0)

        sum_p_years_for_d = sum(w for _y, w in sorted_d_years)
        if sum_p_years_for_d > 0:
            inv_factor = 0.0
            ops_factor = 0.0
            for y, w in sorted_d_years:
                base = global_pos.get(y, 0.0)
                until_inv = base + w * off_inv
                until_ops = base + w * off_ops
                inv_factor += w * (one_plus_inflation_inv ** until_inv)
                ops_factor += w * (one_plus_inflation_inv ** until_ops)
            inflation_invest[d] = inv_factor
            inflation_ops[d] = ops_factor
        else:
            inflation_invest[d] = 1.0
            inflation_ops[d] = 1.0

    return inflation_invest, inflation_ops
