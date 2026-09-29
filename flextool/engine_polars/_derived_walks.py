"""Shared period-walk utilities for derived-helper clusters (Δ.6).

This module hosts the lazy-polars period-walk + lifetime-window join
utility consumed by both Cluster A (annual integration / NPV; Δ.5) and
Cluster B (existing chain & invest sets; Δ.6) of the derived-helper
port.

The Δ.5 close stanza in ``progress.md`` flagged that the
``period_walk_iterator`` helper in ``_derived_npv.py`` was a candidate
for extraction so Cluster B can re-use it without circular import.
This file is the realised refactor.

User-locked decisions (per the Δ.6 dispatch):

* **Lazy polars throughout.** Helpers return ``pl.LazyFrame`` chains;
  the ``Param`` constructor calls ``.collect()`` at the boundary.
* **``window_method`` enum dispatches bounded vs. unbounded** lifetime
  windows in a single helper signature (replacing the Δ.5 ``bounded:
  bool`` + conditional kwarg).
* **No defensive gating** — invalid inputs raise.

Algorithm
---------

The walk implements the per-(e, d) "integrate over lifetime window"
pattern shared by Cluster A (NPV / lifetime fixed cost) and Cluster B
(``edd_history`` / ``edd_invest_set``) derivations.

For each (e, d) anchor pair, walk every ``d_all`` ∈ ``period_in_use``
and collect those that match the window predicate:

* :data:`WindowMethod.BOUNDED` —
  ``pdy[d_all] ∈ [pdy[d], pdy[d] + life[e, d])``.
* :data:`WindowMethod.UNBOUNDED_FORWARD` —
  ``pdy[d_all] ≥ pdy[d]``.
* :data:`WindowMethod.STRICT_LOOKBACK_BOUNDED` —
  ``pdy[d_all] > pdy[d]`` AND ``pdy[d_all] < pdy[d] + life[e, d]``
  (used by Cluster B's ``edd_invest_lookback_set`` for bounded
  ``reinvest_choice`` / ``no_investment`` entities).
* :data:`WindowMethod.STRICT_LOOKBACK_UNBOUNDED` —
  ``pdy[d_all] > pdy[d]`` (no lifetime cap; for unbounded
  ``reinvest_automatic`` entities in Cluster B's lookback walk).

The aggregation is configurable: callers can either pass a per-d_all
weight column to sum (``factor_side``: ``"inv"`` / ``"ops"`` →
inflation factor), or skip aggregation entirely and return the
``(e, d, d_all)`` triples (``factor_side=None``) for set-shape outputs.

Scenario-lineage filter (recourse plan §6b Slice B)
---------------------------------------------------

``lineage`` optionally restricts the ``(d, d_all)`` pair domain to the
Slice A ``dd_same_scenario`` contract
(``specs/sliceA_lineage_frames_design.md`` §2.2/§2.5.3) BEFORE the
``factor_side`` aggregation — remove a walk row ``(e, d, d_all)`` iff
``d ∈ period_in_use`` AND ``d_all ∈ period_in_use`` AND
``(d, d_all) ∉ lineage``; any row with either end outside
``period_in_use`` (history-anchored rows) passes unconditionally.
Every production caller passes ``lineage=None`` (no filtering) until
the Slice C/D recourse activation lands.
"""
from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine_polars._axis_enums import rename_to_axis

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


class WindowMethod(enum.Enum):
    """Lifetime-window selector for :func:`period_walk_iterator`.

    Members:

    * ``BOUNDED`` — ``pdy[d_all] ∈ [pdy[d], pdy[d] + life[e, d])``.
      Used by Cluster A's ``reinvest_choice`` / ``no_investment`` cohort
      and by Cluster A's divest-side NPV (always bounded, raw lifetime).
    * ``UNBOUNDED_FORWARD`` — ``pdy[d_all] ≥ pdy[d]``.  Used by
      Cluster A's ``reinvest_automatic`` cohort.
    * ``STRICT_LOOKBACK_BOUNDED`` — ``pdy[d] < pdy[d_all] < pdy[d] +
      life[e, d]``.  Used by Cluster B's
      ``edd_invest_lookback_set`` walk where the invest is treated as
      having taken effect strictly *before* the dispatch period.
    * ``BOUNDED_INCLUSIVE_LOOKBACK`` — ``pdy[d_all] >= pdy[d]`` AND
      ``pdy[d_all] < pdy[d] + life[e, d]``.  Used by Cluster B's
      ``edd_history`` triple set: the (e, d_history, d) triple is
      kept when ``d_history`` ≤ ``d`` and within the lifetime window
      (the "history → current" cascade in
      ``invest_divest_sets.py:241-248``).  Note: this is the same as
      :data:`BOUNDED` semantically; we expose it as a separate name so
      that the call site is self-documenting.
    """

    BOUNDED = "bounded"
    UNBOUNDED_FORWARD = "unbounded_forward"
    STRICT_LOOKBACK_BOUNDED = "strict_lookback_bounded"
    STRICT_LOOKBACK_UNBOUNDED = "strict_lookback_unbounded"
    BOUNDED_INCLUSIVE_LOOKBACK = "bounded_inclusive_lookback"


def _assert_lineage_castable(lineage: pl.DataFrame,
                             d_dtype: pl.DataType) -> None:
    """Raise ``ValueError`` when any lineage token nulls under a
    non-strict cast to the walk's ``d`` dtype.

    A silently-nulled token would make the marker left-join miss and
    DELETE a pair that IS in the lineage frame — exactly the
    silent-lineage failure class the recourse program keeps
    eliminating.  Mirrors the Slice A ``_finalize_lineage_frame``
    live-assert idiom (``_derived_branch.py``).

    Empirical polars 1.40.1 behavior this guard exists for (Slice B
    design W5(c)): a NON-STRICT cast between two different Enum
    vocabularies (and Utf8 → Enum) NULLS every token missing from the
    target vocabulary instead of raising — so it is THIS guard that
    raises, never polars itself.
    """
    bad: list[str] = []
    for col in ("d", "d_other"):
        raw = lineage[col]
        cast = raw.cast(d_dtype, strict=False)
        bad.extend(str(v) for v, c in zip(raw.to_list(), cast.to_list())
                   if c is None)
    if bad:
        from flextool.engine_polars._solve_state import LineageFilterError
        raise LineageFilterError(
            "period_walk_iterator lineage filter: token(s) "
            f"{sorted(set(bad))} in the lineage frame are not castable "
            f"to the walk's d dtype {d_dtype!r} — a silently-nulled "
            "token would delete lineage pairs from the walk (Slice B "
            "design §4.2 null guard)."
        )


def _apply_lineage_filter(
        walk: pl.LazyFrame,
        lineage: pl.DataFrame,
        period_in_use: list[str],
        d_dtype: pl.DataType,
        *,
        anchor_col: str = "d",
        dall_col: str = "d_all",
        ) -> pl.LazyFrame:
    """Slice A semi-join contract (sliceA design §2.2/§2.5.3):
    remove ``(anchor, other)`` iff BOTH ends ∈ ``period_in_use`` AND
    the pair is not in ``lineage``; rows with either end outside
    ``period_in_use`` (history-anchored walk rows) pass through
    unconditionally.

    Dtype alignment is fixed-direction: the LINEAGE side is cast
    (non-strict) to the walk's ``d`` dtype; the walk's column dtypes
    are invariant under the filter.  ``align_join_dtypes`` was
    evaluated for this seam and REJECTED (Slice B design §4.2 F6): its
    preference rule casts toward whichever side is Enum, so on the
    raw-CSV cascade path (walk ``d`` = Utf8, lineage = Enum) it would
    mutate the walk's output dtype mid-pipeline and null-poison walk
    tokens outside the lineage vocabulary (history anchors are exactly
    such candidates).

    ``anchor_col`` / ``dall_col`` parametrise the walk-side column
    names so the one helper also serves the non-walker pair-former
    ``edd_divest_active_from_source`` (``(d_divest, d)`` frame —
    Slice B design §7.0); both resolve to the d-axis vocabulary.
    """
    lin = (lineage.lazy()
             .select(
                 pl.col("d").cast(d_dtype, strict=False)
                     .alias(anchor_col),
                 pl.col("d_other").cast(d_dtype, strict=False)
                     .alias(dall_col))
             .unique()                      # mandatory: a duplicated
                                            # pair must not fan the join
             .with_columns(_lin=pl.lit(True)))
    in_piu = (pl.col(anchor_col).cast(pl.Utf8).is_in(period_in_use)
              & pl.col(dall_col).cast(pl.Utf8).is_in(period_in_use))
    return (walk
              .join(lin, on=[anchor_col, dall_col], how="left")
              .filter(pl.col("_lin").fill_null(False) | ~in_piu)
              .drop("_lin"))


def period_walk_iterator(
        source: "InputSource",
        active_solve: str | None,
        ed_lf: pl.LazyFrame,
        period_in_use: list[str],
        period_universe: list[str],
        *,
        window_method: WindowMethod,
        life_lf: pl.LazyFrame | None,
        factor_side: str | None,
        workdir = None,
        lineage: pl.DataFrame | None = None,
        provider: "object | None" = None,
        ) -> pl.LazyFrame:
    """Lazy per-(e, d) walk over ``period_in_use``, gated by lifetime.

    Parameters
    ----------
    source
        ``InputSource`` for the cluster's data (used to read
        ``p_years_d`` / yearly inflation factors via lazy helpers
        in ``_derived_npv``).
    active_solve
        Active solve name for ``p_years_d`` resolution.
    ed_lf
        Lazy frame ``[e, d, ...]`` enumerating entity-period anchor
        pairs.  Extra columns are dropped — only ``e``, ``d`` survive.
    period_in_use
        Active-solve dispatch periods over which ``d_all`` ranges.
    period_universe
        Period universe for the inflation factor computation
        (typically ``periodAll``).  Only consulted when
        ``factor_side`` is set.
    window_method
        See :class:`WindowMethod`.
    life_lf
        Lazy ``[e, d, life]`` providing the lifetime per (e, d).
        Required for any bounded variant; ignored for
        :data:`WindowMethod.UNBOUNDED_FORWARD`.
    factor_side
        ``"inv"`` / ``"ops"`` selects the per-d_all inflation factor
        and aggregates ``Σ_{d_all matching} factor[d_all]`` per (e, d)
        (returns ``[e, d, factor]``).  ``None`` returns the unaggregated
        ``[e, d, d_all]`` triple frame (set-shape, used by Cluster B).
    workdir
        Optional ``Path`` for resolving ``p_years_d`` from
        ``solve_data/p_years_d.csv`` (preferred when present — it's
        the canonical post-preprocessing CSV that already encodes the
        cumulative year offset for the active solve).  Cluster A's
        callers don't pass this (they're called from the apply_npv
        boundary which has consumed the workdir already); Cluster B's
        invest-history callers do.
    lineage
        Optional ``(d, d_other)`` scenario-lineage frame
        (``flex_data.dd_same_scenario``, Slice A — see
        ``specs/sliceA_lineage_frames_design.md`` §2.2/§2.5.3).  When
        provided, the ``(d, d_all)`` pairs are filtered BEFORE the
        ``factor_side`` aggregation / set-shape dedup under the
        Slice A semi-join contract: remove a walk row
        ``(e, d, d_all)`` iff ``d ∈ period_in_use`` AND
        ``d_all ∈ period_in_use`` AND ``(d, d_all) ∉ lineage``; any
        row with either end outside ``period_in_use`` passes through
        unconditionally.  ``None`` (the default, and every production
        call at HEAD) applies no filtering — the hot path gains one
        pointer comparison.  The two empty-input early returns below
        deliberately do not consult ``lineage`` (with no pairs there
        is nothing to filter).

    Returns
    -------
    Lazy frame.  Schema depends on ``factor_side``:

    * ``factor_side="inv" | "ops"`` → ``[e, d, factor]``.
    * ``factor_side=None`` → ``[e, d, d_all]`` (one row per matching
      triple).
    """
    from flextool.engine_polars._axis_enums import empty_like
    # Slice D α-1: fall back to the boundary-scoped recourse Provider (and
    # its workdir — the canonical year readers key the Provider path off
    # it) so the year/factor arms revive under the flag without threading
    # ``provider``/``workdir`` through every NPV/edd walker caller.
    # Off-flag both are ``None`` → arms dead → byte-parity.
    if provider is None:
        from flextool.engine_polars._derived_params import (
            _recourse_walk_provider,
            _recourse_walk_workdir,
        )
        provider = _recourse_walk_provider()
        if provider is not None and workdir is None:
            workdir = _recourse_walk_workdir()
    if not period_in_use:
        if factor_side is None:
            return empty_like(ed_lf, ["e", "d"],
                                extra={"d_all": ed_lf.collect_schema().get(
                                    "d", pl.Utf8)},
                                lazy=True)
        return ed_lf.select("e", "d").with_columns(
            factor=pl.lit(0.0, dtype=pl.Float64))
    # Lazy import to avoid circular dependency at module-load time.
    from ._derived_params import _p_years_d_lf
    # Slice D α-1 (§5.1): under recourse the caller forwards ``provider`` so
    # the canonical ``p_years_d.csv`` arm — which byte-copies fan-member year
    # rows — goes live, giving branch periods their anchor's year-from-start
    # (correct annuity windows).  ``provider is None`` → arm dead → today's
    # fill_null(0.0) behaviour → flag-off byte-parity (the W6 today-bug pin).
    pyd_lf = _p_years_d_lf(source, active_solve, workdir, provider=provider)
    if pyd_lf is None:
        # Without years offsets, the integral collapses to 0 / no rows.
        if factor_side is None:
            return empty_like(ed_lf, ["e", "d"],
                                extra={"d_all": ed_lf.collect_schema().get(
                                    "d", pl.Utf8)},
                                lazy=True)
        return ed_lf.select("e", "d").with_columns(
            factor=pl.lit(0.0, dtype=pl.Float64))

    bounded_methods = {
        WindowMethod.BOUNDED,
        WindowMethod.STRICT_LOOKBACK_BOUNDED,
        WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
    }
    if window_method in bounded_methods and life_lf is None:
        raise ValueError(
            f"life_lf is required for window_method={window_method.value}"
        )
    # STRICT_LOOKBACK_UNBOUNDED is permitted with life_lf=None
    # (the strict-lookback predicate doesn't reference lifetime).

    # Anchor years.
    pyd_anchor = pyd_lf.pipe(rename_to_axis, {"d": "d", "yr": "yr_d"})
    # d_all years.
    pyd_all = pyd_lf.rename({"d": "d_all", "yr": "yr_dall"})
    # ``period_in_use`` is a plain ``list[str]``; the cross join with
    # ``ed_lf`` (which may carry Enum-typed ``d``) and the subsequent
    # joins against ``pyd_*`` (CSV-read String) need a single dtype.
    # Cast ``d_all`` to match ``ed_lf.d``'s dtype, then push the same
    # dtype onto ``pyd_anchor.d`` / ``pyd_all.d_all`` so every join key
    # is consistent.
    ed_d_dtype = ed_lf.collect_schema().get("d", pl.Utf8)
    piu_lf = pl.LazyFrame({"d_all": period_in_use}).with_columns(
        pl.col("d_all").cast(ed_d_dtype, strict=False))
    if ed_d_dtype != pl.Utf8:
        pyd_anchor = pyd_anchor.with_columns(
            pl.col("d").cast(ed_d_dtype, strict=False))
        pyd_all = pyd_all.with_columns(
            pl.col("d_all").cast(ed_d_dtype, strict=False))

    walk = (ed_lf
              .select("e", "d")
              .join(piu_lf, how="cross")
              .join(pyd_anchor, on="d", how="left")
              .join(pyd_all, on="d_all", how="left")
              .with_columns(
                  yr_d=pl.col("yr_d").fill_null(0.0),
                  yr_dall=pl.col("yr_dall").fill_null(0.0),
              )
            )

    if window_method == WindowMethod.UNBOUNDED_FORWARD:
        walk = walk.filter(pl.col("yr_dall") >= pl.col("yr_d"))
    elif window_method == WindowMethod.STRICT_LOOKBACK_UNBOUNDED:
        walk = walk.filter(pl.col("yr_dall") > pl.col("yr_d"))
    elif window_method == WindowMethod.STRICT_LOOKBACK_BOUNDED:
        # Align life_lf's dim-column dtypes to walk's before joining.
        walk_schema = walk.collect_schema()
        life_lf = life_lf.with_columns(
            pl.col("e").cast(walk_schema.get("e", pl.Utf8), strict=False),
            pl.col("d").cast(walk_schema.get("d", pl.Utf8), strict=False))
        walk = (walk
                  .join(life_lf, on=["e", "d"], how="left")
                  .with_columns(life=pl.col("life").fill_null(0.0))
                  .filter(pl.col("yr_dall") > pl.col("yr_d"))
                  .filter(pl.col("yr_dall") < pl.col("yr_d") + pl.col("life"))
                )
    else:  # BOUNDED / BOUNDED_INCLUSIVE_LOOKBACK
        walk_schema = walk.collect_schema()
        life_lf = life_lf.with_columns(
            pl.col("e").cast(walk_schema.get("e", pl.Utf8), strict=False),
            pl.col("d").cast(walk_schema.get("d", pl.Utf8), strict=False))
        walk = (walk
                  .join(life_lf, on=["e", "d"], how="left")
                  .with_columns(life=pl.col("life").fill_null(0.0))
                  .filter(pl.col("yr_dall") >= pl.col("yr_d"))
                  .filter(pl.col("yr_dall") < pl.col("yr_d") + pl.col("life"))
                )

    # Scenario-lineage filter (Slice B) — PRE-aggregation, so the
    # factor-side Σ over d_all only ever sees same-scenario pairs.
    if lineage is not None:
        _assert_lineage_castable(lineage, ed_d_dtype)
        walk = _apply_lineage_filter(walk, lineage,
                                     period_in_use, ed_d_dtype)

    if factor_side is None:
        return (walk
                  .select("e", "d", "d_all")
                  .unique())

    # Inflation factor sum.
    from ._derived_npv import _inflation_factors_lf
    factors_lf = _inflation_factors_lf(
        source, active_solve, period_universe, provider=provider)
    if factor_side == "inv":
        factors_lf = factors_lf.select(
            "d", pl.col("inv_factor").alias("factor"))
    elif factor_side == "ops":
        factors_lf = factors_lf.select(
            "d", pl.col("ops_factor").alias("factor"))
    else:
        raise ValueError(
            f"factor_side must be 'inv', 'ops', or None; got {factor_side!r}")
    factor_dall = factors_lf.rename({"d": "d_all"})
    if ed_d_dtype != pl.Utf8:
        factor_dall = factor_dall.with_columns(
            pl.col("d_all").cast(ed_d_dtype, strict=False))

    return (walk
              .join(factor_dall, on="d_all", how="left")
              .with_columns(factor=pl.col("factor").fill_null(1.0))
              .group_by(["e", "d"])
              .agg(pl.col("factor").sum().alias("factor")))


__all__ = [
    "WindowMethod",
    "period_walk_iterator",
]
