"""Existing capacity: the existing unit count of processes, ``p_flow_upper_existing``,
``p_entity_all_existing`` and :func:`apply_derived_d` (Γ.3.D)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import cast_frame_axes, get_global_axis_enums, rename_to_axis

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _entity_classes_lookup,
    _provider_has_key,
    _provider_read,
    _read_active_solve,
    _try_param,
    _try_param_explicit,
)
from flextool.engine.preprocess.params._derived._unitsize import UNITSIZE_DEFAULT, _entity_unitsize_lf
from flextool.engine.preprocess.params._derived._periods import _period_in_use_set
from flextool.engine.preprocess.params._derived._arc_capacity import (
    p_flow_upper_from_source,
    p_indirect_input_cap_from_source,
)
from flextool.engine.preprocess.params._derived._reserves import process_reserve_upDown_node_active_from_source


# ---------------------------------------------------------------------------
# §3.6.2 — p_process_existing_count
# ---------------------------------------------------------------------------


def _read_p_entity_all_existing_csv(workdir: Path | None,
                                       *,
                                       provider: "object | None" = None,
                                       ) -> pl.DataFrame | None:
    """Γ.6.D — read ``solve_data/p_entity_all_existing`` from *provider*.

    Returns ``[e, d, value]``, or None when *workdir* is None or the
    provider does not carry the frame.  Used by helpers that need the
    chained existing capacity (incl. multi-solve handoff) without
    recomputing it from raw ``entity.existing``.

    ``emit_existing_capacity`` (``emit/_chain_params.py``) puts the
    frame into the provider; *workdir* only forms the key, nothing is
    read from disk.
    """
    if workdir is None:
        return None
    p = Path(workdir) / "solve_data" / "p_entity_all_existing.csv"
    if not _provider_has_key(provider, p):
        return None
    try:
        df = _provider_read(provider, p)
    except Exception:
        return None
    if (df.height == 0 or "entity" not in df.columns
            or "period" not in df.columns or "value" not in df.columns):
        return None
    return (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .select("e", "d", "value"))


def _process_capacity_per_unit_lf(cap_lf: pl.LazyFrame, source: "InputSource") -> pl.LazyFrame:
    """Lazy ``[p, d, value]``: the capacity ``[e, d, value]`` (MW / MWh) of each entity divided by its unitsize
    (:func:`_entity_unitsize_lf`; ``UNITSIZE_DEFAULT`` where absent).

    capacity = existing × virtual_unitsize when virtual_unitsize is above 0, else existing, so the result is the
    existing unit count: ``existing`` itself with virtual_unitsize, 1 without it (unitsize = existing; a
    virtual_unitsize of 0 or below is ignored).
    """
    us_lf = _entity_unitsize_lf(source)
    return (cap_lf.pipe(rename_to_axis, {"e": "p"})
                  .join(us_lf.pipe(rename_to_axis, {"e": "p"}), on="p", how="left")
                  .with_columns(us=pl.col("us").fill_null(UNITSIZE_DEFAULT))
                  .with_columns(value=pl.col("value") / pl.col("us"))
                  .select("p", "d", "value"))


def _explicit_process_existing(source: "InputSource") -> list[str]:
    """Units and connections with an explicit ``existing`` row (Spine's default-broadcast rows excluded)."""
    names: list[str] = []
    for cls in ("unit", "connection"):
        ex = _try_param_explicit(source, cls, "existing")
        if ex is not None and ex.height > 0:
            names.extend(str(x) for x in ex.get_column("name").unique().to_list())
    return sorted(set(names))


def _pre_existing_process_capacity_lf(source: "InputSource",
                                       active_solve: str | None,
                                       periods: list[str],
                                       workdir: Path | None,
                                       *,
                                       provider: "object | None" = None,
                                       ) -> pl.LazyFrame | None:
    """Lazy ``[e, d, value]``: the lifetime-gated existing capacity (``p_entity_pre_existing``: existing ×
    virtual_unitsize when it is set, else existing) of each unit / connection with an explicit ``existing``, over
    ``periods``.  ``None`` when no process has an explicit ``existing`` or ``periods`` is empty."""
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    names = _explicit_process_existing(source)
    if not names or not periods:
        return None
    return (p_entity_pre_existing_lf(source, active_solve, periods, workdir=workdir, provider=provider)
              .filter(pl.col("e").cast(pl.Utf8).is_in(names)))


def p_process_existing_count_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    active_solve: str | None = None,
    workdir: Path | None = None,
    *,
    provider: "object | None" = None,
) -> Param | None:
    """Per-(p, d) existing unit count = capacity / unitsize (= ``existing`` when virtual_unitsize is above 0).

    capacity[p, d] = existing[p, d] × virtual_unitsize[p] when virtual_unitsize is above 0, else existing[p, d];
    unitsize per the canonical cascade (:func:`_entity_unitsize_lf`)::

        unitsize = virtual_unitsize (if explicitly set and above 0)
                  OR existing       (if non-zero)
                  OR 1000.0

    The capacity comes from the canonical chained ``p_entity_all_existing`` (multi-solve handoff state and
    lifetime gate integrated) when the run has emitted it; otherwise from ``p_entity_pre_existing`` (the
    lifetime-gated existing × virtual_unitsize cascade over the periods of ``dt``) for the units / connections
    with an explicit ``existing``.

    Returns ``None`` when no process has an explicit existing capacity.
    """
    if dt is None:
        return None

    pae_csv = _read_p_entity_all_existing_csv(workdir, provider=provider)
    if pae_csv is not None and pae_csv.height > 0:
        # Restrict to process-side entities (unit ∪ connection).
        proc_set, _, _ = _entity_classes_lookup(source)
        per_proc = pae_csv.filter(pl.col("e").is_in(list(proc_set)))
        if per_proc.height > 0:
            df = (_process_capacity_per_unit_lf(per_proc.lazy(), source)
                     .sort("p", "d")
                     .collect())
            if df.height > 0:
                return Param(("p", "d"), df)

    periods = [str(d) for d in dt.get_column("d").unique(maintain_order=True).to_list()]
    cap_lf = _pre_existing_process_capacity_lf(source, active_solve, periods, workdir, provider=provider)
    if cap_lf is None:
        return None
    out = (_process_capacity_per_unit_lf(cap_lf, source)
              .sort("p", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "d"), out)


# ---------------------------------------------------------------------------
# §3.3.3 — p_flow_upper_existing (existing_count = capacity / unitsize per arc)
# ---------------------------------------------------------------------------


def _flow_upper_existing_from_chained_csv(source: "InputSource",
                                              pss: pl.DataFrame,
                                              workdir: Path | None,
                                              *,
                                              provider: "object | None" = None,
                                              ) -> "Param | None":
    """Γ.6.D — build ``p_flow_upper_existing`` from the canonical
    ``solve_data/p_entity_all_existing.csv`` when it exists, divided by
    the per-entity unitsize cascade and cross-joined with ``pss``.

    Returns None when the CSV is absent (caller falls back to the
    raw-existing path).
    """
    pae_csv = _read_p_entity_all_existing_csv(workdir, provider=provider)
    if pae_csv is None or pae_csv.height == 0:
        return None
    proc_set, _, _ = _entity_classes_lookup(source)
    per_proc = pae_csv.filter(pl.col("e").is_in(list(proc_set)))
    if per_proc.height == 0:
        return None
    base = _process_capacity_per_unit_lf(per_proc.lazy(), source)
    if pss is None or pss.height == 0:
        return None
    out = (pss.lazy()
              .join(base, on="p", how="inner")
              .select("p", "source", "sink", "d", "value")
              .sort("p", "source", "sink", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d"), out)


def p_flow_upper_existing_from_source(source: "InputSource",
                                         pss: pl.DataFrame,
                                         active_solve: str | None = None,
                                         workdir: Path | None = None,
                                         *,
                                         ctx: "SolveContext | None" = None,
                                         provider: "object | None" = None,
                                         ) -> "Param | None":
    """Per-arc structural existing bound: the existing unit count (capacity / unitsize per (p, d), =
    ``existing`` when virtual_unitsize is above 0) cross-joined with ``pss``.

    capacity[p, d] = existing[p, d] × virtual_unitsize[p] when virtual_unitsize is above 0, else existing[p, d];
    unitsize = virtual_unitsize above 0, else existing, else 1000 (:func:`_entity_unitsize_lf`).  The capacity comes from
    the canonical chained ``p_entity_all_existing`` (multi-solve handoff and lifetime gate integrated) when the run
    has emitted it; otherwise from ``p_entity_pre_existing`` (the lifetime-gated existing × virtual_unitsize
    cascade) over ``period_in_use`` (:func:`_fallback_period_in_use`) for the units / connections with an explicit
    ``existing``.

    Schema: ``Param[(p, source, sink, d), value]``.

    Returns ``None`` when the source has no usable ``existing`` data —
    caller falls back to CSV value.
    """
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None and pss is not None:
        pss = cast_frame_axes(pss, _enums)
    chained = _flow_upper_existing_from_chained_csv(
        source, pss, workdir, provider=provider)
    if chained is not None:
        return chained

    if pss is None or pss.height == 0:
        return None
    periods = _fallback_period_in_use(source, workdir, ctx=ctx, provider=provider)
    cap_lf = _pre_existing_process_capacity_lf(source, active_solve, periods, workdir, provider=provider)
    if cap_lf is None:
        return None
    base = _process_capacity_per_unit_lf(cap_lf, source)
    out = (pss.lazy()
              .join(base, on="p", how="inner")
              .select("p", "source", "sink", "d", "value")
              .sort("p", "source", "sink", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d"), out)


def _fallback_period_in_use(source: "InputSource",
                            workdir: Path | None = None,
                            *,
                            ctx: "SolveContext | None" = None,
                            provider: "object | None" = None,
                            ) -> list[str]:
    """``period_in_use`` for the provider-less existing fallback.

    flextool's per_solve_sets.py:95-101 defines ``period_in_use`` as the union of realised + invest periods plus
    stochastic-branch siblings.  The in-memory :pyattr:`SolveContext.period_in_use` frame is consulted first, then
    the canonical ``period_in_use_set.csv`` (includes stochastic branches) when ``workdir`` is provided; otherwise
    the realised + invest union of the source's solves.
    """
    if ctx is not None:
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            return [str(d) for d in piu_ctx.get_column("d").unique(maintain_order=True).to_list()]
    if workdir is not None:
        piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
        if _provider_has_key(provider, piu_path):  # Step 1-g-5 — Provider-first
            piu_df = _provider_read(provider, piu_path)
            if piu_df.height > 0 and "period" in piu_df.columns:
                return [str(d) for d in piu_df.get_column("period").unique(maintain_order=True).to_list()]
    periods: list[str] = []
    for par in ("realized_periods", "invest_periods"):
        rp = _try_param(source, "solve", par)
        if rp is None:
            continue
        periods.extend(str(d) for d in rp.get_column("value").to_list())
    return list(dict.fromkeys(periods))


# =============================================================================
# Γ.3.D — final derived-Param batch.  Narrow, conservative scope:
#
#   * §3.11 existing-entity fixed cost — ``p_entity_all_existing``
#     (sum of pre_existing + previously_invested per (e, d)).
#   * §3.13 reserve cleanup — narrow gate-on-equality overlay for the
#     `_increase_reserve_ratio` / `_large_failure_ratio` Projection sets
#     when the underlying SpineDB reserve relationships are present.
#
# Storage / lifetime cascade / handoff state / ladder / delay / multi-
# branch Params are deferred to Γ.3.E; their algorithms either depend
# on per-solve state not in Spine, or require a fully-faithful port of
# flextool preprocessing's coarse-block synthesis (the flex_data loaders)
# whose risk profile exceeds the time budget for this batch.  See
# ``progress.md`` for the deferred-list bookkeeping.
# =============================================================================


def p_entity_all_existing_from_source(source: "InputSource",
                                        active_solve: str | None,
                                        workdir: Path | None = None,
                                        *,
                                        ctx: "SolveContext | None" = None,
                                        provider: "object | None" = None,
                                        ) -> "Param | None":
    """§3.11 — sum of pre-existing capacity per (entity, period).

    First-solve (or single-solve) collapses to:

        p_entity_all_existing[e, d] = pre_existing[e, d]

    where ``pre_existing[e, d]`` is the lifetime-gated existing
    capacity (:func:`._existing.p_entity_pre_existing_lf`), with
    capacity[e, d] = existing[e, d] × virtual_unitsize[e] when
    virtual_unitsize is above 0, else existing[e, d]:

      * ``reinvest_automatic`` → capacity[e, d].
      * ``reinvest_choice`` / ``no_reinvestment`` → capacity[e, d]
        only while ``p_years_d[d] < life_sum(e)``; 0 thereafter.
      * Any other ``lifetime_method`` → 0.

    In a run the solve emits ``solve_data/p_entity_all_existing`` (``_chain_params.emit_existing_capacity``: the
    existing cohort with the user's ``existing`` of each period, plus the investments earlier solves realized,
    minus the carried retirements); that frame is read when present, so every consumer reads one value.
    """
    piu = _period_in_use_set(source, active_solve, workdir, ctx=ctx, provider=provider)
    if not piu:
        return None

    # Γ.6.D — chain-run handoff: prefer the canonical per-solve
    # ``solve_data/p_entity_all_existing.csv`` when present.  This CSV
    # carries the lifetime-cumulative + previously-invested chain
    # already integrated by the chain runner / flextool's preprocessing
    # for sub-solves N>1 (see
    # ``entity_period_calc_params.write_p_entity_existing_chain``).
    # When the CSV is absent (single-solve fixtures, or DB-direct sweep
    # tooling without preprocessing), fall back to deriving from
    # ``entity.existing`` + lifetime gate.
    #
    # Path B Cat B: when ``ctx`` is provided, the in-memory
    # :pyattr:`SolveContext.p_entity_all_existing` frame is consulted
    # first.  Falls back to the CSV-read path when ``ctx`` is None /
    # the typed accessor is empty.
    if ctx is not None:
        pae_ctx = ctx.p_entity_all_existing
        if pae_ctx.height > 0 \
                and "entity" in pae_ctx.columns \
                and "period" in pae_ctx.columns \
                and "value" in pae_ctx.columns:
            df2 = (pae_ctx.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                          .with_columns(value=pl.col("value")
                                                  .cast(pl.Float64,
                                                          strict=False)
                                                  .fill_null(0.0))
                          .select("e", "d", "value")
                          .sort("e", "d"))
            if df2.height > 0:
                return Param(("e", "d"), df2)
    elif workdir is not None:
        pae_path = Path(workdir) / "solve_data" / "p_entity_all_existing.csv"
        if _provider_has_key(provider, pae_path):  # Phase E-j — seed-aware
            try:
                df = _provider_read(provider, pae_path)
                if df.height > 0 and "entity" in df.columns \
                        and "period" in df.columns and "value" in df.columns:
                    df2 = (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
                              .with_columns(value=pl.col("value")
                                                       .cast(pl.Float64,
                                                                strict=False)
                                                       .fill_null(0.0))
                              .select("e", "d", "value")
                              .sort("e", "d"))
                    if df2.height > 0:
                        return Param(("e", "d"), df2)
            except Exception:
                pass
    # No chained frame: the lifetime-gated existing capacity of every unit / node / connection
    # (p_entity_pre_existing: existing × virtual_unitsize when it is set, else existing; 0 without existing).
    from flextool.engine.preprocess.params._existing import p_entity_pre_existing_lf
    out = (p_entity_pre_existing_lf(source, active_solve, piu, workdir=workdir, provider=provider)
             .sort("e", "d")
             .collect())
    if out.height == 0:
        return None
    return Param(("e", "d"), out)


# ---------------------------------------------------------------------------
# Public — Γ.3.D field list (for selective scope-checking).
# ---------------------------------------------------------------------------


D_PUBLIC_FIELDS: tuple[str, ...] = (
    "p_entity_all_existing",
    "process_reserve_upDown_node_active",
)


def apply_derived_d(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.D Derived Params, mutating ``flex_data`` in place.

    Scope (Tier-1 selected items):
      * §3.11 ``p_entity_all_existing`` (simple existing-only path).
      * §3.13 ``process_reserve_upDown_node_active`` (reliability>0 set).

    Δ.12b — assignment is unconditional; helpers are authoritative
    producers.  ``None`` is the explicit "feature inactive" signal
    (no reserve relationships /
    no chained existing capacity).  Hard exceptions propagate.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)

    # ─── §3.11 p_entity_all_existing — Δ.12b: unconditional ──────────
    flex_data.p_entity_all_existing = p_entity_all_existing_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)

    # Δ.12c — ``apply_existing_chain`` was previously invoked here, but
    # it consumes the handoff carriers (``p_entity_previously_invested_capacity``
    # / ``p_entity_divested``) which are populated by ``apply_derived_f``.
    # Calling it here meant the chain-summation read stale (or seed-only)
    # carriers on the first iteration.  Moved to ``_apply_db_overrides``
    # so it runs strictly after ``apply_derived_f``.

    # ─── §3.13 process_reserve_upDown_node_active — Δ.12b: unconditional
    flex_data.process_reserve_upDown_node_active = (
        process_reserve_upDown_node_active_from_source(source))

    # ─── §3.3.5 p_flow_upper (Δ.26) ───────────────────────────────────
    # Per-(p, source, sink, d, t) structural max flow.  Native port of
    # ``preprocessing/process_arc_unions.py:write_p_flow_max`` (file:line
    # `process_arc_unions.py:1469-1624`).  Wired in apply_derived_d so
    # ``p_entity_all_existing`` (set immediately above) is available;
    # ``p_entity_max_units`` from apply_derived_c is also picked up
    # when the fixture has invest activity.
    pss_for_upper = getattr(flex_data, "process_source_sink", None)
    dt_for_upper = getattr(flex_data, "dt", None)
    if (pss_for_upper is not None and pss_for_upper.height > 0
            and dt_for_upper is not None and dt_for_upper.height > 0):
        slope_for_upper = getattr(flex_data, "p_slope", None)
        section_for_upper = getattr(flex_data, "p_section", None)
        unitsize_for_upper = getattr(flex_data, "p_unitsize", None)
        max_units_for_upper = getattr(flex_data, "p_entity_max_units", None)
        existing_for_upper = getattr(flex_data, "p_entity_all_existing", None)
        pfu_db = p_flow_upper_from_source(
            source,
            pss_for_upper,
            dt_for_upper,
            p_slope=slope_for_upper,
            p_section=section_for_upper,
            p_unitsize=unitsize_for_upper,
            p_entity_max_units=max_units_for_upper,
            p_entity_all_existing=existing_for_upper,
        )
        if pfu_db is not None:
            flex_data.p_flow_upper = pfu_db
        # Input-arc limit of indirect units per unit of capacity
        # (``input_share_max`` × full-load fuel); model/core/ requires it
        # whenever indirect input arcs exist.
        flex_data.p_indirect_input_cap = p_indirect_input_cap_from_source(
            source, pss_for_upper, dt_for_upper,
            p_slope=slope_for_upper, p_section=section_for_upper)
