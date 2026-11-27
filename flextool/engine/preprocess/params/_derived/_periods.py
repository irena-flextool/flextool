"""The active solve's periods: its period lists, ``period_first``, the run start period, ``p_years_d``,
``period_in_use`` and the branch periods of a stochastic fan."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import alias_to_axis, schema_dtype
from flextool.engine.preprocess.params._branch import period_branch_pairs_lf
from flextool.engine.input._param_shapes import solve_years_represented

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _enums,
    _labels_of_pairs,
    _provider_has_key,
    _provider_read,
    _recourse_invest_active,
    _try_param,
)


# ===========================================================================
# Γ.3.C — invest/divest + online/UC + group slack + existing fixed cost
# ===========================================================================
#
# Per the audit doc §3.7 / §3.8 / §3.11 / §3.12.  Each helper composes a
# lazy chain over ``InputSource`` reads and ``.collect()``s once at the
# boundary.  Each overlay is gated on a frame-equal pre-check against the
# CSV-loaded value (``apply_derived_c``) so multi-year invest cascades
# / scaling-active fixtures fall back to the CSV value instead of being
# corrupted by a simple-path approximation.
#
# Integration: ``apply_derived_c`` runs after Γ.3.A + Γ.3.B; see
# ``flex_data/_load.py::load_flextool``.

# ---------------------------------------------------------------------------
# §3.7.0 — Helpers (active-solve scope, period_invest, p_years_d, etc.)
# ---------------------------------------------------------------------------


def _resolve_synthetic_solve(source: "InputSource",
                                  active_solve: str | None,
                                  ) -> tuple[str, str] | None:
    """Recognise a synthetic per-sub-solve name of the form
    ``<base>_<anchor>`` and return ``(base, anchor)``.

    Δ.19 — flextool's orchestrator synthesises per-period sub-solve
    names at runtime by joining a Spine ``solve`` name with one of its
    2-D period map's outer keys (``SolveConfig._split_2d_period_maps``
    makes ``<base>_<anchor>`` with ``derive_solve(<base>, ...)``).  These synthetic
    names don't exist as rows in Spine; the per-solve override chain
    would otherwise return None for every ``solve``-keyed parameter.

    Algorithm: try every suffix split (rightmost underscore first) and
    return the first ``(base, anchor)`` where ``base`` is a Spine
    ``solve`` entity.  ``base`` may itself contain underscores
    (``invest_5weeks_p2020`` splits as ``base='invest_5weeks'``,
    ``anchor='p2020'``).

    Returns ``None`` when no split matches a Spine solve.
    """
    if active_solve is None or "_" not in active_solve:
        return None
    try:
        ents = source.entities("solve")
    except KeyError:
        return None
    if ents is None or ents.height == 0:
        return None
    name_col = "name" if "name" in ents.columns else ents.columns[0]
    spine_solves = set(ents[name_col].to_list())
    parts = active_solve.split("_")
    for i in range(len(parts) - 1, 0, -1):
        base = "_".join(parts[:i])
        anchor = "_".join(parts[i:])
        if base in spine_solves:
            return base, anchor
    return None


def _solve_periods(source: "InputSource", active_solve: str,
                    parameter_name: str) -> list[str] | None:
    """Read the period-list-valued ``solve.<parameter>`` for the active solve.

    ``solve.realized_periods`` is stored as a Spine Array
    (``[name, i (int), value=period]``); ``solve.invest_periods`` is
    stored either as an Array (single-anchor solves) or as a 2D Map
    (nested multi-invest solves: ``[name, x=anchor, i=period,
    value="yes"]``).  This helper returns the period list regardless of
    shape.

    The inner Map level's column name is taken from the Spine
    ``index_name`` and is not guaranteed to be ``"i"`` — some
    hand-authored fixtures (e.g. ``examples.sqlite::invest_5weeks``)
    leave both Map levels named ``"x"``, producing the post-unroll
    shape ``[name, x, value]``.  We therefore compute the period
    column as ``df.columns[-2]`` (the last non-value column) instead
    of hardcoding ``"i"``.

    Δ.19 — when ``active_solve`` doesn't appear in the param table,
    fall back to the synthetic ``<base>_<anchor>`` recognition: for
    Map-shaped params filter ``name==base AND x==anchor`` and return
    the inner-period values; for Array-shaped params return
    ``[anchor]`` (the single realised period of the synthetic
    sub-solve).
    """
    if active_solve is None:
        return None
    df = _try_param(source, "solve", parameter_name)
    if df is None:
        return None
    is_map = "x" in df.columns
    # Inner-period column = deepest non-entity, non-value column.
    # For Map-shaped params this is ``df.columns[-2]`` (covers both
    # the canonical ``"i"`` shape and the hand-authored ``"x"``-named
    # inner-index shape from ``examples.sqlite``).  For Array shapes
    # the value column itself carries the period.
    period_col = df.columns[-2] if is_map else "value"
    sub = df.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        # Synthetic fallback — split active_solve as <base>_<anchor>.
        resolved = _resolve_synthetic_solve(source, active_solve)
        if resolved is None:
            return None
        base, anchor = resolved
        if is_map:
            sub = df.filter((pl.col("name") == base)
                              & (pl.col("x") == anchor))
            if sub.height == 0:
                return None
            return sub[period_col].cast(pl.Utf8, strict=False).to_list()
        # Array param: synthetic sub-solve realises only its anchor.
        # Confirm the anchor is in base's array; if not, return None.
        base_sub = df.filter(pl.col("name") == base)
        if base_sub.height == 0:
            return None
        base_periods = set(base_sub["value"].cast(pl.Utf8, strict=False).to_list())
        return [anchor] if anchor in base_periods else None
    # Non-synthetic path.  For Map params filter to the (most-common)
    # anchor's row set; in practice the non-synthetic Map case doesn't
    # surface (orchestrator always synthesises sub-solves), but if it
    # ever did we'd return the union of inner periods deterministically.
    if "i" in sub.columns and not is_map:
        sub = sub.sort("i")
    if is_map:
        return sub[period_col].cast(pl.Utf8, strict=False).unique(maintain_order=True).to_list()
    return sub["value"].to_list()


# ---------------------------------------------------------------------------
# §3.7.1 — ed_invest_set / ed_divest_set
# ---------------------------------------------------------------------------


def _read_period_first(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None,
                          *,
                          provider: "object | None" = None,
                          ) -> list[str]:
    """Return ``period_first`` for the active solve (Γ.6.D).

    The first period of the first timeset on the active solve, plus any
    stochastic-branch siblings (matching the legacy ``write_periods``
    output).

    Resolution order:
      1. ``solve_data/period_first.csv`` if present (canonical).
      2. First entry of ``solve.realized_periods`` (non-stochastic
         fallback for tests that don't have the CSV).

    Returns ``[]`` when no usable source.
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_first.csv"
        if _provider_has_key(provider, p):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, p)
                if df.height > 0 and "period" in df.columns:
                    return df["period"].cast(pl.Utf8, strict=False).to_list()
            except Exception:
                pass
    if active_solve is None:
        return []
    realized = _solve_periods(source, active_solve, "realized_periods") or []
    invest = _solve_periods(source, active_solve, "invest_periods") or []
    seq = realized or invest
    return [seq[0]] if seq else []


def run_start_period(years: "dict[str, float]") -> str | None:
    """The period the model run starts in, among a solve's period years ``years`` (``p_years_d``: the period
    history first, each period's years from the start of the run, a branch period with its label's year): the
    first period with the smallest year.  ``None`` without periods.

    The end of life of pre-existing capacity counts from this period in every solve of the run (a later solve of
    a chain holds the earlier periods as its history), so a chain and a single solve age it alike, and a fan of
    the first period gives it one start, not one per branch."""
    from flextool.engine.preprocess.params._run_calendar import YEAR_TOL
    start: str | None = None
    for d, y in years.items():
        if start is None or y < years[start] - YEAR_TOL:
            start = d
    return start


# ---------------------------------------------------------------------------
# §3.7.2 / §3.7.3 — the solve's period years and the invest / divest options
# ---------------------------------------------------------------------------


def _p_years_d_lf(source: "InputSource",
                    active_solve: str,
                    workdir: Path | None = None,
                    *,
                    provider: "object | None" = None,
                    ) -> pl.LazyFrame | None:
    """Build per-period ``p_years_d`` for the active solve.

    Resolution order (Γ.6.D):
      1. ``solve_data/p_years_d.csv`` from workdir, if present — the
         canonical post-preprocessing CSV that already encodes the
         cumulative year offset.
      2. ``solve_data/period_with_history.csv`` (column ``param`` holds
         the cumulative year value).
      3. ``solve.years_from_start`` 1d_map(period) on the active solve
         (rare; not in current SpineDB schema but kept as a safety net).
      4. ``solve.years_represented`` — cumulative-sum the per-period
         year-count to derive the offset (matches
         ``_p_discount_years_from_source``).
      5. Integer-indexed fallback ``[0, 1, 2, ...]`` over period_in_use.

    Returns lazy frame ``[d, yr]`` or None when nothing usable exists.
    """
    # 1 — workdir CSV (authoritative).
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "p_years_d.csv"
        if _provider_has_key(provider, p):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, p)
                if df.height > 0 and "period" in df.columns:
                    val_col = ("value" if "value" in df.columns
                                else df.columns[-1])
                    return df.lazy().select(
                        alias_to_axis("period", "d"),
                        pl.col(val_col).cast(pl.Float64, strict=False)
                                          .alias("yr"),
                    )
            except Exception:
                pass
        p2 = Path(workdir) / "solve_data" / "period_with_history.csv"
        if _provider_has_key(provider, p2):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, p2)
                if df.height > 0 and "period" in df.columns:
                    val_col = ("param" if "param" in df.columns
                                else df.columns[-1])
                    return df.lazy().select(
                        alias_to_axis("period", "d"),
                        pl.col(val_col).cast(pl.Float64, strict=False)
                                          .alias("yr"),
                    )
            except Exception:
                pass
    # 2 — explicit years_from_start (rare; not present in current schema).
    yfs = _try_param(source, "solve", "years_from_start")
    if yfs is not None and "period" in yfs.columns:
        sub = yfs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            return sub.lazy().select(
                alias_to_axis("period", "d"),
                pl.col("value").cast(pl.Float64).alias("yr"),
            )
    # 3 — cumulative-sum from years_represented.
    yrp = solve_years_represented(source)
    if yrp is not None and active_solve is not None:
        sub = yrp.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            # Map iteration order — the SpineDbReader returns rows in DB
            # storage order which mirrors the canonical Map index order.
            cum = 0.0
            rows: list[tuple[str, float]] = []
            for r in sub.iter_rows(named=True):
                d = str(r["period"])
                rows.append((d, cum))
                try:
                    cum += float(r["value"])
                except Exception:
                    pass
            if rows:
                return pl.LazyFrame({
                    "d": [r[0] for r in rows],
                    "yr": [r[1] for r in rows],
                }).with_columns(
                    pl.col("d").cast(schema_dtype(_enums, "d"),
                                       strict=False),
                ).lazy()
    # 4 — fallback to integer-indexed period_in_use.
    periods = _period_in_use_set(source, active_solve, provider=provider)
    if periods:
        return pl.LazyFrame({
            "d": periods,
            "yr": [float(i) for i in range(len(periods))],
        }).with_columns(
            pl.col("d").cast(schema_dtype(_enums, "d"), strict=False),
        ).lazy()
    return None


def _period_in_use_set(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None = None,
                          *,
                          provider: "object | None" = None,
                          ctx: "SolveContext | None" = None) -> list[str]:
    """Compute the canonical ``period_in_use`` set for the active solve.

    Mirrors flextool's ``preprocessing/per_solve_sets.py:95-101``:
    distinct periods in ``solve_data/steps_in_use.csv`` (a.k.a. the
    realised ``dt`` set).  When ``ctx`` is supplied, the typed
    :pyattr:`SolveContext.period_in_use` frame is consulted first — same
    semantics as the workdir CSV path but served from cache.  When
    ``workdir`` is provided AND the file is present (no ctx), the
    authoritative CSV is read directly.

    Falls back to ``solve.realized_periods ∪ solve.invest_periods`` when
    no in-memory / workdir source is available — sufficient for non-
    stochastic fixtures.
    """
    if ctx is not None:
        piu = ctx.period_in_use
        if piu.height > 0:
            return piu["d"].cast(pl.Utf8, strict=False).to_list()
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if _provider_has_key(provider, p):  # Phase E-j — seed-aware
            df = _provider_read(provider, p)
            if df.height > 0 and "period" in df.columns:
                return df["period"].cast(pl.Utf8, strict=False).to_list()
    if active_solve is None:
        return []
    realized = _solve_periods(source, active_solve, "realized_periods") or []
    invest = _solve_periods(source, active_solve, "invest_periods") or []
    seen: dict[str, None] = {}
    for d in realized + invest:
        seen.setdefault(d, None)
    return list(seen.keys())


def _expand_branch_periods(period_order: list[str],
                              workdir: Path | None,
                              *,
                              ctx: "SolveContext | None" = None,
                              provider: "object | None" = None,
                              ) -> list[str]:
    """Given a period_order coming from solve.realized_periods +
    period_timeset, add stochastic-branch periods read from
    ``solve_data/period__branch.csv`` (anchor → branch map), filtered
    to those that actually appear in the canonical ``period_in_use_set``
    or ``steps_in_use.csv`` (excludes realized-output-only branches).

    For each anchor in period_order, append all unique in-use branches
    whose anchor is the anchor — preserving anchor's relative position.

    Path B Cat B: when ``ctx`` is provided, the in-memory
    :pyattr:`SolveContext.period_branch` and
    :pyattr:`SolveContext.period_in_use` frames are preferred over the
    workdir CSVs.  Falls back to reading the CSVs when ``ctx`` is None
    or the typed accessors are empty (and the workdir path exists).
    """
    # Try ctx-side first: canonical in-memory carriers.
    pb_df: pl.DataFrame | None = None
    in_use: set[str] = set()
    if ctx is not None:
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            # Canonical schema is [d_anchor, b]; remap to legacy ('period', 'branch')
            # only for the local iteration below.
            pb_df = pb_ctx.rename({"d_anchor": "anchor", "b": "br"})
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            in_use = set(piu_ctx["d"].to_list())
    if pb_df is None:
        if workdir is None:
            return list(period_order)
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
            return list(period_order)
        try:
            df = _provider_read(provider, p)
        except Exception:
            return list(period_order)
        if df.height == 0:
            return list(period_order)
        if not in_use:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
                try:
                    piu = _provider_read(provider, piu_path)
                    if piu.height > 0:
                        in_use = set(piu["period"].to_list())
                except Exception:
                    in_use = set()
        pb_df = df.rename({"period": "anchor", "branch": "br"})
    anchor_to_branches: dict[str, list[str]] = {}
    for row in pb_df.iter_rows(named=True):
        anchor_to_branches.setdefault(row["anchor"], []).append(row["br"])
    out: list[str] = []
    seen: set[str] = set()
    for d in period_order:
        if d not in seen:
            out.append(d)
            seen.add(d)
        for br in anchor_to_branches.get(d, []):
            if br in seen:
                continue
            if in_use and br not in in_use:
                continue
            out.append(br)
            seen.add(br)
    return out


def _expand_invest_branch_periods(period_invest: "list[str] | None",
                                     workdir: Path | None,
                                     *,
                                     ctx: "SolveContext | None" = None,
                                     provider: "object | None" = None,
                                     ) -> "list[str] | None":
    """Slice D (A, §3.1) — invest-side mirror of
    :func:`_expand_branch_periods`.

    When the recourse flag is active, append each real invest period's
    synthetic branch members (filtered to ``period_in_use``) after the
    anchor, so ``period_invest`` carries ``{p_k, p_k_b, …}`` — the axis
    on which ``pd/nd_invest_set`` (and thus ``v_invest``) get their
    columns.  Flag-off (or empty axis) → returns ``period_invest``
    unchanged (byte-parity).  Reuses — does not fork — the dispatch-side
    helper so invest and dispatch axes stay consistent.

    Under recourse a period label (the original name of a period that
    opens a fan, :func:`._branch.fan_period_labels`) has no investment
    column: every forecast branch invests on its own after the reveal,
    so nothing is invested here-and-now at the label.
    """
    if not period_invest:
        return period_invest
    if not _recourse_invest_active(workdir, ctx=ctx, provider=provider):
        return period_invest
    expanded = _expand_branch_periods(list(period_invest), workdir,
                                      ctx=ctx, provider=provider)
    pb = period_branch_pairs_lf(workdir, ctx=ctx, provider=provider).collect()
    labels = _labels_of_pairs(pb["d"].to_list(), pb["b"].to_list())
    return [d for d in expanded if str(d) not in labels]
