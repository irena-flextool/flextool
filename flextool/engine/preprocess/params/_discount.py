"""Discount factors: the inflation scalars, the year rows, the per-period factors, the solve's discount base from the
run start and the run years.

The model's ``social_discount_rate`` / ``discount_offset_investment`` / ``discount_offset_operations``
(:func:`_solve_inflation_scalars`), the year rows of each period (:func:`_inflation_year_rows_lf`, from the solve's
history :func:`year_rows_from_history` / :func:`solve_year_rows`) and the discount factors of investments and of
operations per period (:func:`_inflation_factors_lf`, ``p_inflation_op``: :func:`p_inflation_op_from_source`); the
years of the run's periods from the run calendar (:func:`run_years`) and the run-level investment factors
(:func:`_run_inv_factors`).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, schema_dtype

# Substrate handle for the cascade-wide axis enum vocabulary (``None``: ``cast_dim`` / ``schema_dtype`` fall
# back to the live ContextVar ``_LIVE_AXIS_ENUMS_CTX``; see ``_npv``).
_enums: "dict | None" = None

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


# Inflation defaults — mirrors period_calculated_params.py:221-227 and
# _solve_inflation_inputs in _derived.py.
_INFL_RATE_DEFAULT = 0.0
_INFL_OFFSET_INV_DEFAULT = 0.0
_INFL_OFFSET_OPS_DEFAULT = 0.5


# ---------------------------------------------------------------------------
# Inflation factors (lazy)
# ---------------------------------------------------------------------------


def _solve_inflation_scalars(source: "InputSource"
                                  ) -> tuple[float, float, float]:
    """Read ``model.social_discount_rate`` / ``model.discount_offset_investment`` /
    ``model.discount_offset_operations`` scalars, falling back to
    flextool's CSV-side defaults when the Spine reader returned
    only-default rows.

    Mirrors :func:`._derived._solve_inflation_inputs` semantics.
    """
    def _explicit_max(par: str, default: float) -> float:
        try:
            df = source.parameter("model", par)
        except KeyError:
            return default
        if df.height == 0:
            return default
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
                vals = df["value"].cast(pl.Float64, strict=False).to_list()
                if vals and all(
                        v is not None and abs(v - spine_default_f) < 1e-12
                        for v in vals):
                    return default
        try:
            return float(df["value"].cast(pl.Float64).max())
        except Exception:
            return default

    return (
        _explicit_max("social_discount_rate", _INFL_RATE_DEFAULT),
        _explicit_max("discount_offset_investment", _INFL_OFFSET_INV_DEFAULT),
        _explicit_max("discount_offset_operations", _INFL_OFFSET_OPS_DEFAULT),
    )


def _years_for_period_lf(source: "InputSource",
                              active_solve: str | None,
                              period_universe: list[str],
                              *,
                              provider: "object | None" = None,
                              ) -> pl.LazyFrame:
    """Build the per-(d, year_label, width) frame from
    ``solve.years_represented`` mirroring
    :func:`._derived._years_for_period_from_source` in lazy form.

    Returns columns ``[d, y, width]`` where ``y`` is a string label.
    Empty if no rows.  ``provider`` (Slice D α-1) revives the canonical
    ``p_years_represented.csv`` arm carrying fan-member year rows.
    """
    # Materialise scalars eagerly — they're solve-level, tiny.
    from flextool.engine.preprocess.params._derived import _years_for_period_from_source
    yfp = _years_for_period_from_source(
        source, active_solve, period_universe, provider=provider)
    rows: list[tuple[str, str, float]] = []
    for d, years in yfp.items():
        for y, w in years:
            rows.append((d, y, w))
    if not rows:
        return pl.LazyFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "y": pl.Utf8, "width": pl.Float64,
        })
    return pl.LazyFrame(rows, schema=["d", "y", "width"], orient="row").with_columns(
        alias_to_axis("d", "d"))


def _inflation_year_rows_lf(source: "InputSource",
                           active_solve: str | None,
                           period_universe: list[str],
                           *,
                           provider: "object | None" = None,
                           year_rows: pl.DataFrame | None = None,
                           ) -> pl.LazyFrame | None:
    """Lazy ``[d, off, width, base, inv_contrib, ops_contrib]``: one row per represented year of a period of
    ``period_universe`` (``d`` on the period axis), ``off`` the years of the period before it, ``width`` its width,
    ``base`` the years from the discount base to its start and ``inv_contrib`` / ``ops_contrib`` its discounted width
    (the terms of :func:`_inflation_factors_lf`).

    Algorithm — port of ``preprocessing/period_calculated_params.py:230-322``:

      1. Scalar inputs: ``rate``, ``offset_invest``, ``offset_operations``.
      2. For each (d, y) ∈ ``years_for_period``:
           ``base[d, y] = Σ_{y' ∈ global, y' < y} pyr[d, y']``  (default 1)
           ``until_inv[d, y] = base + pyr × offset_invest``
           ``until_ops[d, y] = base + pyr × offset_operations``
           ``inv_contrib[d, y] = pyr[d, y] × (1+rate)^(-until_inv[d, y])`` (``ops_contrib`` likewise).

    The cumulative-base computation (step 2) is per-period over the
    GLOBAL year set, so it's shaped naturally as a polars window:

      base[d, y] = Σ_{y'<y in global order} pyr[d, y']
                 = (cumulative_sum of pyr[d, y'] over y'<y, partitioned by d)

    where unbound (d, y) pairs contribute 1 (default ``years_represented``).  ``off`` counts only the period's own
    rows.

    ``year_rows`` (``[d, width, base]``, one row per represented year of a period): the years given explicitly
    instead of from ``years_for_period``; ``base`` is the number of years from the discount base to the start of
    that year, so ``until = base + width × offset``.

    ``None`` when there are no year rows (every period has the factor 1).  The rows keep the order the factor sums
    take them in (the order of ``year_rows`` / of ``years_for_period``).
    """
    rate, off_inv, off_ops = _solve_inflation_scalars(source)
    one_plus_inv = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    if year_rows is not None:
        return (year_rows.lazy()
                  .select(alias_to_axis(pl.col("d").cast(pl.Utf8), "d"), pl.col("width").cast(pl.Float64),
                          pl.col("base").cast(pl.Float64))
                  .with_columns(
                      off=(pl.col("width").cum_sum() - pl.col("width")).over("d", order_by="base"),
                      inv_contrib=pl.col("width") * (one_plus_inv ** (pl.col("base")
                                                                      + pl.col("width") * off_inv)),
                      ops_contrib=pl.col("width") * (one_plus_inv ** (pl.col("base")
                                                                      + pl.col("width") * off_ops)))
                  .select("d", "off", "width", "base", "inv_contrib", "ops_contrib"))
    yfp_lf = _years_for_period_lf(
        source, active_solve, period_universe, provider=provider)

    # Materialise per-period × per-y.  At LP scale this is small (typically
    # ≤ 100 (d, y) rows even for 50-year horizon × 20 representative years),
    # so collecting once is cheap; the rest stays lazy.
    yfp_eager = yfp_lf.collect()
    if yfp_eager.height == 0:
        # No years_represented data — every period gets the trivial 1.0
        # factor (mirrors period_calculated_params.py:300).
        return None
    period_lf = pl.LazyFrame({"d": period_universe}).with_columns(
        alias_to_axis("d", "d"))

    # Build the global year set with numerical sort when possible.
    try:
        global_years = sorted(
            yfp_eager["y"].unique().to_list(), key=lambda y: float(y)
        )
    except ValueError:
        global_years = sorted(yfp_eager["y"].unique().to_list())

    # Cross-join (d, y_global) so missing rows default to width=1.0
    # (mirrors flextool's default of p_years_represented).
    global_y_lf = pl.LazyFrame({"y": global_years}).with_columns(
        y_idx=pl.int_range(0, pl.len(), dtype=pl.Int64)
    )
    full = (period_lf
              .join(global_y_lf, how="cross")
              .join(yfp_eager.lazy(), on=["d", "y"], how="left")
              .with_columns(
                  pyr=pl.col("width").fill_null(1.0),
              ))

    # Cumulative base[d, y] = Σ pyr[d, y'] over y' < y (within d).
    full = (full
              .sort(["d", "y_idx"])
              .with_columns(
                  base=pl.col("pyr").cum_sum().over("d") - pl.col("pyr"),
              ))

    # Restrict to (d, y) actually bound to this period.
    bound = (yfp_eager.lazy()
                .select("d", "y")
                .join(full, on=["d", "y"], how="inner"))

    # Per-(d, y) factor contributions.
    return (bound
              .with_columns(
                  until_inv=pl.col("base") + pl.col("pyr") * off_inv,
                  until_ops=pl.col("base") + pl.col("pyr") * off_ops,
                  off=(pl.col("pyr").cum_sum() - pl.col("pyr")).over("d", order_by="y_idx"),
              ).with_columns(
                  inv_contrib=pl.col("pyr") * (one_plus_inv ** pl.col("until_inv")),
                  ops_contrib=pl.col("pyr") * (one_plus_inv ** pl.col("until_ops")),
              )
              .select("d", "off", pl.col("pyr").alias("width"), "base", "inv_contrib", "ops_contrib"))


def _inflation_factors_lf(source: "InputSource",
                                active_solve: str | None,
                                period_universe: list[str],
                                *,
                                provider: "object | None" = None,
                                year_rows: pl.DataFrame | None = None,
                                ) -> pl.LazyFrame:
    """Lazy ``(d, inv_factor, ops_factor)`` frame: per period the sum of its year rows
    (:func:`_inflation_year_rows_lf`), ``inv_factor[d] = Σ_y inv_contrib[d, y]`` and ``ops_factor[d] = Σ_y
    ops_contrib[d, y]``.  A period without rows (or with zero total width) gets 1.0.
    """
    if not period_universe:
        return pl.LazyFrame(schema={
            "d": schema_dtype(_enums, "d"),
            "inv_factor": pl.Float64, "ops_factor": pl.Float64,
        })
    period_lf = pl.LazyFrame({"d": [str(d) for d in period_universe]}).with_columns(alias_to_axis("d", "d"))
    rows = _inflation_year_rows_lf(source, active_solve, period_universe, provider=provider, year_rows=year_rows)
    if rows is None:
        return period_lf.with_columns(inv_factor=pl.lit(1.0, dtype=pl.Float64),
                                      ops_factor=pl.lit(1.0, dtype=pl.Float64))
    factor_per_d = (rows
                      .group_by("d")
                      .agg(pl.col("inv_contrib").sum().alias("inv_factor"),
                           pl.col("ops_contrib").sum().alias("ops_factor"),
                           pl.col("width").sum().alias("sum_pyr"))
                      # Sum-pyr == 0 → factor = 1.0 (period_calculated_params.py:299-301).
                      .filter(pl.col("sum_pyr") > 0)
                      .select("d", "inv_factor", "ops_factor"))
    # Periods absent from years_represented (or with R=0) get 1.0
    # (mirrors flextool's writer default of 1.0).
    return (period_lf
              .join(factor_per_d, on="d", how="left")
              .with_columns(inv_factor=pl.col("inv_factor").fill_null(1.0),
                            ops_factor=pl.col("ops_factor").fill_null(1.0)))


def _factor_frame(source: "InputSource",
                  active_solve: str | None,
                  period_in_use: list[str],
                  period_universe: list[str],
                  provider: "object | None",
                  factor: str,
                  year_rows: pl.DataFrame | None = None,
                  ) -> pl.LazyFrame:
    """``[d_all, factor]`` (Utf8 key): the solve's per-period discount factor (``ops_factor``: ``p_inflation_op``,
    the factor of the existing fixed cost in the objective; ``inv_factor``: its investment twin)."""
    param = p_inflation_op_from_source(source, active_solve, period_in_use, period_universe, provider=provider,
                                       factor=factor, year_rows=year_rows)
    if param is None:
        return pl.LazyFrame(schema={"d_all": pl.Utf8, "factor": pl.Float64})
    return param.frame.lazy().select(pl.col("d").cast(pl.Utf8).alias("d_all"), pl.col("value").alias("factor"))


def _retirement_discount_lf(source: "InputSource",
                            active_solve: str | None,
                            period_universe: list[str],
                            *,
                            provider: "object | None",
                            year_rows: pl.DataFrame | None) -> pl.LazyFrame:
    """Lazy ``[d, disc]`` (``d`` Utf8): D(d) = (1+i)^−base₀(d), the discount of a one-off cash flow at the start of
    period ``d`` (its first year's ``base``, :func:`_inflation_year_rows_lf`; no offset: a retirement settles when
    the capacity leaves, at the start of the period).  1.0 for a period without year rows."""
    rate = _solve_inflation_scalars(source)[0]
    periods = pl.LazyFrame({"d": [str(d) for d in period_universe]}, schema={"d": pl.Utf8})
    rows = _inflation_year_rows_lf(source, active_solve, period_universe, provider=provider, year_rows=year_rows)
    if rows is None:
        return periods.with_columns(disc=pl.lit(1.0))
    start = rows.group_by("d").agg(pl.col("base").min()).select(pl.col("d").cast(pl.Utf8), "base")
    v = (1.0 / (1.0 + rate)) if rate != -1.0 else 1.0
    return (periods.join(start, on="d", how="left")
                   .with_columns(disc=pl.when(pl.col("base").is_null()).then(1.0).otherwise(v ** pl.col("base")))
                   .select("d", "disc"))


def _run_inv_factors(source: "InputSource",
                     active_solve: str | None,
                     period_in_use: list[str],
                     period_universe: list[str],
                     period_with_history: list[str],
                     years: dict[str, float],
                     *,
                     provider: "object | None",
                     year_rows: pl.DataFrame | None,
                     run_factor: "Param | None",
                     ) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """``(factors, walk_periods, universe)``: the investment factor ``[d_all, factor]`` (Utf8) of every period of
    the solve's history and periods in use, from the start of the run (``run_factor``, ``p_inflation_inv_run``, where
    it has the period; the solve's own factor otherwise, equal on its periods), the years ``[d_all, y]`` of those
    periods (``years``: :func:`._cohorts.cohort_years`) and the period universe the factors are computed over.  The
    take-back of a salvage credit or a level price sums these factors over the periods a cohort is in service.  A
    period of the history that is neither in use nor a period of the run calendar is no period of the run (the
    branch copy of a realized designation, ``<period>_<branch>``, :func:`run_years`): it holds no years of its own."""
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    calendar = read_run_calendar(provider)
    in_use = {str(d) for d in period_in_use}
    history = [str(d) for d in period_with_history
               if calendar is None or str(d) in in_use or str(d) in calendar.years]
    periods = list(dict.fromkeys([*history, *(str(d) for d in period_in_use)]))
    universe = list(dict.fromkeys([*(str(d) for d in period_universe), *periods]))
    factors = (_inflation_factors_lf(source, active_solve, universe, provider=provider, year_rows=year_rows)
               .select(pl.col("d").cast(pl.Utf8).alias("d_all"), pl.col("inv_factor").alias("factor")).collect())
    if run_factor is not None:
        run = run_factor.frame.select(pl.col("d").cast(pl.Utf8).alias("d_all"), pl.col("value").alias("_run"))
        factors = (factors.join(run, on="d_all", how="full", coalesce=True)
                          .select("d_all", factor=pl.coalesce("_run", "factor")))
    walk_periods = pl.DataFrame({"d_all": [d for d in periods if d in years],
                                 "y": [float(years[d]) for d in periods if d in years]},
                                schema={"d_all": pl.Utf8, "y": pl.Float64})
    return factors, walk_periods, universe


def _payment_rows(source: "InputSource",
                  active_solve: str | None,
                  period_in_use: list[str],
                  universe: list[str],
                  periods: set[str],
                  workdir: Path | None,
                  *,
                  provider: "object | None",
                  year_rows: pl.DataFrame | None) -> pl.DataFrame:
    """``[base, width, contrib]``: one row per represented calendar year of the run (the run calendar's periods and
    widths, :func:`run_years`; the solve's own years ``periods`` when the solve is the run), ``base`` its years from
    the start of the run and ``contrib = width·(1+i)^−(base + width·off_inv)`` the discounted width a yearly payment
    of it has (:func:`_inflation_year_rows_lf`).  The forecast copies of a period share its years: each year once."""
    run = run_years(source, active_solve, list(period_in_use), workdir, provider=provider)
    if run is not None:
        rows = _inflation_year_rows_lf(source, active_solve, sorted(run[0]), year_rows=_run_year_rows(*run))
    else:
        rows = _inflation_year_rows_lf(source, active_solve, universe, provider=provider, year_rows=year_rows)
        if rows is not None:
            rows = rows.filter(pl.col("d").is_in(sorted(periods)))
    if rows is None:
        return pl.DataFrame(schema={"base": pl.Float64, "width": pl.Float64, "contrib": pl.Float64})
    return (rows.select("base", "width", contrib=pl.col("inv_contrib"))
                .sort("base").unique(subset=["base"], keep="first", maintain_order=True).collect())


def _walk_ops_factor(source: "InputSource", active_solve: str | None, period_universe: list[str],
                     year_rows: pl.DataFrame | None) -> pl.LazyFrame:
    """``[d_all, factor]``: the operations factor the walks of :func:`._lifetime_inputs.period_walk_iterator` sum (the
    same factors, with the boundary's walker Provider of a solve with branch periods)."""
    from flextool.engine.preprocess.params._derived import _fan_walk_provider
    return (_inflation_factors_lf(source, active_solve, period_universe, provider=_fan_walk_provider(),
                                  year_rows=year_rows)
              .select(pl.col("d").alias("d_all"), pl.col("ops_factor").alias("factor")))


def p_inflation_op_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    period_in_use: list[str],
                                    period_universe: list[str] | None = None,
                                    *,
                                    provider: "object | None" = None,
                                    factor: str = "ops_factor",
                                    year_rows: pl.DataFrame | None = None,
                                    ) -> "Param | None":
    """Public entry: ``p_inflation_op[d]`` over ``period_in_use``.

    ``factor="inv_factor"`` gives the investment-side factor over the same periods instead.  ``year_rows``
    (:func:`solve_year_rows`): the years from the start of the run, when the solve's own count does not give them.

    Lazy port of
    :func:`._derived.p_inflation_op_full_cascade_from_source`.
    Replaces the workdir read of ``solve_data/p_inflation_factor_operations_yearly.csv``
    when wired through :func:`._lifetime_costs.apply_npv`.

    ``provider`` — forwarded to :func:`_inflation_factors_lf` /
    :func:`_years_for_period_lf` so a stochastic branch period reads its
    anchor's ``years_represented`` rows from the canonical
    ``solve_data/p_years_represented.csv`` (the same arm the recourse
    annuity walk in ``._walks.period_walk_iterator`` already uses via
    this same ``_inflation_factors_lf`` helper).  Without it every
    branch period falls back to the source arm's un-copied rows and is
    under-weighted relative to its anchor (#51).  This is a direct
    per-``d`` lookup (no window sum over ``d_all``), so there is no
    double-counting to guard against here — unlike the annuity walk,
    which already applies its own scenario-lineage dedup before
    aggregating the same factor over a lifetime window.
    """
    if not period_in_use:
        return None
    if period_universe is None:
        period_universe = period_in_use
    factors_lf = _inflation_factors_lf(source, active_solve, period_universe,
                                        provider=provider, year_rows=year_rows)
    out = (pl.LazyFrame({"d": period_in_use})
              .with_columns(alias_to_axis("d", "d"))
              .join(factors_lf, on="d", how="left")
              .with_columns(
                  # Unreachable on the solve path: ``_inflation_factors_lf`` gives every period of
                  # ``period_universe`` a factor (1.0 where years_represented has none), and the periods in use
                  # are periods of that universe.  The fill only keeps a caller's universe that misses a period
                  # in use from leaving it a null factor (build_flextool checks the factor covers dt).
                  value=pl.col(factor).fill_null(1.0),
              )
              .select("d", "value")
              .sort("d")
              .collect())
    if out.height == 0:
        return None
    return Param(("d",), out)


def _period_history_years(workdir: Path | None, provider: "object | None") -> dict[str, float] | None:
    """``{period: years from the start of the run}`` of the solve's ``period_with_history`` (history first, then the
    solve's own periods; a branch period carries its anchor's year); ``None`` when the solve has none."""
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "period_with_history.csv"
    if not _provider_has_key(provider, path):
        return None
    df = _provider_read(provider, path)
    if df.height == 0 or "period" not in df.columns:
        return None
    col = "param" if "param" in df.columns else df.columns[-1]
    out: dict[str, float] = {}
    for d, y in df.select(pl.col("period").cast(pl.Utf8),
                          pl.col(col).cast(pl.Float64, strict=False)).iter_rows():
        if d is not None and y is not None:
            out.setdefault(str(d), float(y))
    return out or None


def _split_years(represented: float) -> list[float]:
    """Year widths of a period that represents ``represented`` years: ⌈R⌉ years of min(1, remaining), as the
    solve's own ``years_represented`` is split (:func:`._derived._years_for_period_from_source`)."""
    import math
    out: list[float] = []
    remaining = float(represented)
    for _ in range(int(math.ceil(represented)) if represented > 0 else 0):
        width = min(1.0, remaining)
        out.append(width)
        remaining -= width
    return out


def run_years(source: "InputSource",
              active_solve: str | None,
              period_in_use: list[str],
              workdir: Path | None,
              *,
              provider: "object | None" = None,
              ) -> tuple[dict[str, float], dict[str, list[float]]] | None:
    """``(yr, widths)`` of the periods the reported costs of the solve cover, counted from the start of the run:
    ``yr[d]`` the start of ``d``, ``widths[d]`` its represented years in order.  The solve's own periods (its
    ``period_with_history``) take their year from it and their widths from the solve's ``years_represented``;
    the other periods of the run calendar take both from the calendar.

    ``None`` when the solve's own fields are the run fields: no run calendar or no period history, or every run
    period is in use by the solve and the solve starts the run."""
    from flextool.engine.preprocess.params._derived import _years_for_period_from_source
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL, read_run_calendar
    calendar = read_run_calendar(provider)
    history = _period_history_years(workdir, provider)
    if calendar is None or history is None:
        return None
    in_use = [str(d) for d in period_in_use]
    # A period of the history that is neither in use nor a period of the run calendar is no period of the run:
    # the branch copy of a realized designation (``<period>_<branch>``), whose operations the realization solve
    # owns, holds no years of its own.
    history = {d: y for d, y in history.items() if d in in_use or d in calendar.years}
    outside = [d for d in calendar.years if d not in history and d not in in_use]
    starts = [history[d] for d in in_use if d in history]
    if (not outside and set(history) <= set(in_use) and starts
            and abs(min(starts)) <= YEAR_TOL):
        return None
    yr: dict[str, float] = {}
    for d in [*in_use, *history, *outside]:
        if d in yr:
            continue
        if d in history:
            yr[d] = history[d]
        elif d in calendar.years:
            yr[d] = calendar.years[d]
        else:
            raise ValueError(
                f"Period '{d}' of solve '{active_solve}' has no year from the start of the run (neither in the "
                "solve's period history nor in the run calendar); its costs cannot be discounted.")
    own = _years_for_period_from_source(source, active_solve, in_use, provider=provider)
    widths: dict[str, list[float]] = {}
    for d in yr:
        if d in in_use:
            widths[d] = [float(w) for _, w in own.get(d, [])]
        else:
            widths[d] = _split_years(calendar.years_represented.get(d, 0.0))
    return yr, widths


def _run_year_rows(yr: dict[str, float], widths: dict[str, list[float]]) -> pl.DataFrame:
    """``[d, width, base]``: one row per represented year, ``base`` its years from the start of the run."""
    rows: list[tuple[str, float, float]] = []
    for d, start in yr.items():
        cum = 0.0
        for w in widths.get(d, []):
            rows.append((d, w, start + cum))
            cum += w
    return pl.DataFrame(rows, schema={"d": pl.Utf8, "width": pl.Float64, "base": pl.Float64}, orient="row")


# ---------------------------------------------------------------------------
# The discount base of a solve's own factors (the objective's)
# ---------------------------------------------------------------------------
#
# EQ. base[d,y] = pwh[d] + Σ_{y' of d before y} pyr[d,y']
#     F_inv(d) = Σ_y pyr[d,y]·(1+r)^-(base[d,y] + pyr[d,y]·offset_inv)
#     F_ops(d) = Σ_y pyr[d,y]·(1+r)^-(base[d,y] + pyr[d,y]·offset_ops)
#   Every solve (a real solve, a roll, a nested ``<solve>_<period>`` sub-solve) discounts its investments and its
#   operations from the start of the run, as one solve over the same periods does.  pwh[d] is the start of
#   period d from the start of the run (``period_with_history``: the earlier solves' periods first, a branch period
#   at its anchor's year); pyr the widths of the (complete) solve's own ``years_represented`` rows
#   (``p_years_represented``).  Where an earlier solve gave a period a different width, the history gives each
#   period's start and the solve's own map its widths (the run calendar reports the conflict).
# Notes: the solve's own count (``years_from_solve``, from 0 at the solve's first period, each global year label
#   one year wide where the period has no row) gives the same base for a solve that starts the run with whole-year
#   widths; such a solve keeps that count (byte-identical LP).  It differs for a solve that starts after the run
#   start (its own count restarts at 0), and for fractional years (item 259: a period of 2.5 years put the next
#   period at year 3, not 2.5).  The investment walk of a deterministic solve counts from the database map over
#   periodAll (``_years_for_period_from_source``), so a whole map counted investments from the run start and
#   operations from the solve start; both now take ``base``.


def _years_by_label_bases(years: dict[str, list[tuple[str, float]]]) -> dict[str, list[float]]:
    """``{d: [base of each year]}`` of the solve's own count: over the global set of year labels, the sum of the
    period's widths before the year, 1 for a label the period has no row of (:func:`_inflation_factors_lf` without
    ``year_rows``, ``emit/_period_calc``)."""
    labels: dict[str, None] = {}
    for rows in years.values():
        for y, _ in rows:
            labels.setdefault(y, None)
    try:
        ordered = sorted(labels, key=float)
    except ValueError:
        ordered = sorted(labels)
    out: dict[str, list[float]] = {}
    for d, rows in years.items():
        width = dict(rows)
        pos: dict[str, float] = {}
        cum = 0.0
        for y in ordered:
            pos[y] = cum
            cum += width.get(y, 1.0)
        out[d] = [pos[y] for y, _ in rows]
    return out


def _sorted_years(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    try:
        return sorted(rows, key=lambda r: float(r[0]))
    except ValueError:
        return sorted(rows, key=lambda r: r[0])


def year_rows_from_history(history: dict[str, float] | None,
                           years: dict[str, list[tuple[str, float]]],
                           *,
                           solve: str | None = None,
                           also_counted: dict[str, list[tuple[str, float]]] | None = None,
                           ) -> pl.DataFrame | None:
    """``[d, y, width, base]``: the represented years of the solve (``years``: ``{d: [(year label, width)]}``,
    its ``p_years_represented`` rows) with ``base`` their years from the start of the run (the ``EQ.`` block
    above), for :func:`_inflation_factors_lf` (``year_rows``) and ``emit/_period_calc``.

    ``None`` when the solve's own count already gives every year that base (a solve that starts the run with
    whole-year widths): the solve keeps that count.  ``also_counted`` is another count of the same solve (the
    database map over periodAll, which the deterministic investment walk reads); a base it gives differently
    also takes the rows.  ``None`` without a period history.  A period with years but no history is an error
    only when the rows are needed (the gate over the periods with history fails): otherwise the solve keeps its
    own count, as before."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    if not history or not years:
        return None
    own = {d: _sorted_years(rows) for d, rows in years.items() if rows}
    missing = [d for d in own if d not in history]
    rows_out: list[tuple[str, str, float, float]] = []
    new_base: dict[str, list[float]] = {}
    for d, rows in own.items():
        if d not in history:
            continue
        cum = 0.0
        bases: list[float] = []
        for y, w in rows:
            bases.append(history[d] + cum)
            rows_out.append((d, y, float(w), history[d] + cum))
            cum += w
        new_base[d] = bases
    counts = [_years_by_label_bases(own)]
    if also_counted:
        other = {d: _sorted_years(rows) for d, rows in also_counted.items() if rows}
        other_bases = _years_by_label_bases(other)
        # Only the periods the map has years of (a branch period has none there: a fan's walk reads the
        # solve's own rows).
        counts.append({d: other_bases[d] for d in own if d in other_bases})
    differs = any(len(count[d]) != len(new_base[d])
                  or any(abs(a - b) > YEAR_TOL for a, b in zip(count[d], new_base[d]))
                  for count in counts for d in count if d in new_base)
    if not differs:
        return None
    if missing:
        raise ValueError(
            f"Period '{missing[0]}' of solve '{solve}' has represented years but no year from the start of the run "
            "(it is not in the solve's period history); its costs cannot be discounted.")
    return pl.DataFrame(rows_out, schema={"d": pl.Utf8, "y": pl.Utf8, "width": pl.Float64, "base": pl.Float64},
                        orient="row")


def notice_discount_from_run_start(solve: str | None, rate: float) -> None:
    """The behaviour-change notice of a solve whose own count of years does not give its years from the start of
    the run (``year_rows`` not None), when inflation discounts its costs: its objective moved to the run's base.
    A roll is named by its rolling solve."""
    import re

    from flextool.engine import _behaviour_changes
    if solve is None or rate == 0.0:
        return
    _behaviour_changes.notice("chained_solve_discount_from_run_start", "solves",
                              [re.sub(r"_roll_\d+$", "", str(solve))])


def _represented_years(provider: "object | None") -> dict[str, list[tuple[str, float]]]:
    """``{d: [(year label, width)]}`` of the solve's ``solve_data/p_years_represented.csv`` (``{}`` without)."""
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read
    path = Path("solve_data") / "p_years_represented.csv"
    if provider is None or not _provider_has_key(provider, path):
        return {}
    df = _provider_read(provider, path)
    if df.height == 0 or not {"period", "years_from_solve", "p_years_represented"} <= set(df.columns):
        return {}
    out: dict[str, list[tuple[str, float]]] = {}
    for d, y, w in df.select(pl.col("period").cast(pl.Utf8), pl.col("years_from_solve").cast(pl.Utf8),
                             pl.col("p_years_represented").cast(pl.Float64, strict=False)).iter_rows():
        if d is not None and y is not None and w is not None:
            out.setdefault(str(d), []).append((str(y), float(w)))
    return out


def solve_year_rows(workdir: Path | None,
                    provider: "object | None",
                    source: "InputSource | None" = None,
                    active_solve: str | None = None,
                    period_universe: list[str] | None = None,
                    ) -> pl.DataFrame | None:
    """:func:`year_rows_from_history` of the solve of ``workdir``: its period history and its represented years;
    with ``source`` the database map over ``period_universe`` (the deterministic investment walk's count) is
    compared too.  ``None``: the solve keeps its own count."""
    history = _period_history_years(workdir, provider)
    years = _represented_years(provider)
    if not history or not years:
        return None
    also = None
    if source is not None and period_universe:
        from flextool.engine.preprocess.params._derived import _years_for_period_from_source
        also = _years_for_period_from_source(source, active_solve, list(period_universe))
    return year_rows_from_history(history, years, solve=active_solve, also_counted=also)


__all__ = [
    "p_inflation_op_from_source",
    "run_years",
    "notice_discount_from_run_start",
    "solve_year_rows",
    "year_rows_from_history",
]
