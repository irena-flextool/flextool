"""Γ.3.C — the invest boundaries: the recourse anchor scope they arm (Slice D), the investment options,
construction lead times, retirement and zero-cost notices, :func:`apply_derived_c` and
:func:`apply_synthetic_invest_sets`."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import align_join_dtypes
from flextool.engine.input._lifetime_roles import (
    NET_SALVAGE_VALUE,
)
from flextool.engine.preprocess.params._period_values import ed_values_online
from flextool.engine.input._solve_state import CommissioningLagError, FlexToolConfigError
from flextool.engine.preprocess.params._cohorts import (
    BOUNDED_METHODS,
    _entity_methods,
    build_cohort_frames,
    cohort_years,
    existing_cohorts,
    existing_ends,
    existing_eol,
    invest_alive,
    retire_active,
)
from flextool.engine.preprocess.params._existing import INVEST_NOT_ALLOWED as _INVEST_NOT_ALLOWED

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _provider_has_key,
    _provider_read,
    _read_active_solve,
    _recourse_invest_active,
    _try_param,
)
from flextool.engine.preprocess.params._derived._periods import (
    _expand_invest_branch_periods,
    _period_in_use_set,
    _read_period_first,
    _solve_periods,
    run_start_period,
)
from flextool.engine.preprocess.params._derived._inflation import p_inflation_op_multi_year_from_source
from flextool.engine.preprocess.params._derived._process_method import _classify_process_method
from flextool.engine.preprocess.params._derived._process_flows import p_section_from_source
from flextool.engine.preprocess.params._derived._invest_sets import (
    ed_divest_set_from_source,
    ed_invest_set_from_source,
    p_entity_max_units_from_source,
    retirable_existing_pairs,
)
from flextool.engine.preprocess.params._derived._scaling_inflow import (
    p_group_capacity_for_scaling_from_source,
    p_inv_group_cap_from_source,
    p_negative_inflow_from_inflow,
    p_node_capacity_for_scaling_from_source,
    p_positive_inflow_from_inflow,
    pdtNodeInflow_per_step_from_inflow,
    process_group_inside_nonSync_from_flex,
    real_inflow,
)
from flextool.engine.preprocess.params._derived._handoff_state import p_entity_divested_from_handoff


# ─── Slice D (B, §4) — per-period parameter anchor-mapping ─────────────
# Module-level holder for the (anchor, br) synthetic-branch pairs of the
# ACTIVE solve.  Set fresh at the top of every Layer-4 apply boundary
# (``apply_derived_c`` / ``apply_npv`` / ``apply_synthetic_invest_sets``)
# to the recourse pairs, or ``None`` when the flag is off / deterministic.
# Read by the explicit-per-period producers (_resolve_per_period_lf and
# _period_values._ed_period_param_lf, which serves the lifetime, cost and
# interest-rate readers, the zero-cost investment notice and
# p_entity_max_units_from_source) to inherit each
# anchor's per-(e,d) value onto its branch periods (§4.1).  ``None`` ==
# no-op == flag-off byte-parity.  The set-at-boundary-top invariant means
# a stale value can never reach a producer: every boundary overwrites it
# before any producer runs, and producers run nowhere else.
_RECOURSE_ANCHOR_PAIRS: "pl.DataFrame | None" = None


# Boundary-scoped Provider (+ its workdir, which the canonical readers need
# to key the Provider path) for the period walks' year / factor arms.  Set at
# each Layer-4 boundary alongside the anchor pairs whenever the solve has a
# stochastic fan (branch periods), under every ``invest_solve_branching_method``,
# so ``period_walk_iterator`` and ``_existing._solve_walk_years`` read the
# canonical ``p_years_d.csv`` / ``p_years_represented.csv`` arms, which give
# each branch period the years of its period label, WITHOUT threading
# ``provider``/``workdir`` through the NPV/edd walker-caller signatures.
# ``None`` (a solve without branch periods) → the walks read the solve's own
# ``years_represented`` (its periods are all labels, so the years agree).
_FAN_WALK_PROVIDER: "object | None" = None


_FAN_WALK_WORKDIR: "Path | None" = None


def _fan_walk_provider() -> "object | None":
    """The active boundary's walker Provider of a solve with branch periods, or ``None``."""
    return _FAN_WALK_PROVIDER


def _fan_walk_workdir() -> "Path | None":
    """The active boundary's walker workdir of a solve with branch periods, or ``None``."""
    return _FAN_WALK_WORKDIR


def _recourse_branch_axis_present() -> bool:
    """True iff the active boundary is recourse-active with a genuine branch
    invest axis — i.e. the anchor-pairs holder was armed non-empty.  The
    capability conjunct (§2), computed from the provider (branch-cluster
    frames on ``flex_data`` are not populated until derived_g)."""
    return _RECOURSE_ANCHOR_PAIRS is not None


def _build_recourse_anchor_pairs(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """(anchor, br) pairs for in-use synthetic branches under recourse.

    Returns ``None`` when the recourse flag is off (byte-parity default);
    the pairs themselves are :func:`_fan_branch_pairs`.
    """
    if not _recourse_invest_active(workdir, ctx=ctx, provider=provider):
        return None
    return _fan_branch_pairs(workdir, ctx=ctx, provider=provider)


def _fan_branch_pairs(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """(anchor, br) pairs of the solve's in-use synthetic branch periods,
    under every ``invest_solve_branching_method``; ``None`` without any.

    Mirrors :func:`_expand_branch_periods`' acquisition (ctx-first, then
    provider/workdir ``period__branch`` + ``period_in_use``); keeps only
    synthetic (``anchor != br``) branches that carry LP variables (``br``
    in ``period_in_use``).
    """
    pb_df: "pl.DataFrame | None" = None
    in_use: set[str] = set()
    if ctx is not None:
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            pb_df = pb_ctx.rename({"d_anchor": "anchor", "b": "br"})
        piu_ctx = ctx.period_in_use
        if piu_ctx.height > 0:
            in_use = set(piu_ctx["d"].to_list())
    if pb_df is None:
        if workdir is None:
            return None
        p = Path(workdir) / "solve_data" / "period__branch.csv"
        if not _provider_has_key(provider, p):
            return None
        pb_df = _provider_read(provider, p)
        if pb_df.height == 0:
            return None
        if not in_use:
            piu_path = Path(workdir) / "solve_data" / "period_in_use_set.csv"
            if _provider_has_key(provider, piu_path):
                piu = _provider_read(provider, piu_path)
                if piu.height > 0:
                    in_use = set(piu["period"].to_list())
        pb_df = pb_df.rename({"period": "anchor", "branch": "br"})
    out = (pb_df.select(
                pl.col("anchor").cast(pl.Utf8),
                pl.col("br").cast(pl.Utf8))
               .filter(pl.col("anchor") != pl.col("br")))
    if in_use:
        out = out.filter(pl.col("br").is_in(list(in_use)))
    out = out.unique()
    return out if out.height > 0 else None


def _fan_walk_scope_provider(workdir: Path | None, *, provider: "object | None" = None) -> "object | None":
    """``provider`` when the solve has branch periods or is under recourse (the walker Provider
    :func:`_enter_recourse_anchor_scope` arms), else ``None``: for a reader outside a Layer-4 boundary."""
    if _recourse_invest_active(workdir, provider=provider) or _fan_branch_pairs(workdir, provider=provider) is not None:
        return provider
    return None


def _reset_recourse_scope() -> None:
    """Clear all three recourse process-globals to ``None``.

    Defensive per-solve reset: called once at the top of the cascade
    entry (:func:`flex_data._db_overrides._apply_db_overrides`) BEFORE any pass runs, so a
    future per-period producer wired into an EARLIER cascade step cannot
    inherit a stale anchor-pairs holder / walker Provider from a prior
    flag-on solve in a chained flag-on→flag-off multi-solve.  The three
    flag-aware Layer-4 boundaries still overwrite these unconditionally
    (to ``None`` when flag-off), so this reset is belt-and-suspenders;
    the invariant it guards is "no producer sees a global from a
    different solve".
    """
    global _RECOURSE_ANCHOR_PAIRS, _FAN_WALK_PROVIDER, _FAN_WALK_WORKDIR
    _RECOURSE_ANCHOR_PAIRS = None
    _FAN_WALK_PROVIDER = None
    _FAN_WALK_WORKDIR = None


def _enter_recourse_anchor_scope(
    workdir: Path | None,
    *,
    ctx: "object | None" = None,
    provider: "object | None" = None,
) -> None:
    """Set :data:`_RECOURSE_ANCHOR_PAIRS` for the active solve.

    Called at the top of each Layer-4 apply boundary BEFORE any explicit-
    per-period producer OR walker runs.  Overwrites unconditionally (to
    ``None`` when flag-off) so no stale value from a prior solve can leak.
    Also arms the boundary-scoped walker Provider whenever the solve has
    branch periods (any ``invest_solve_branching_method``) or its investments branch,
    so every period walk gives a branch period its label's years; the
    anchor pairs stay recourse-only.
    """
    global _RECOURSE_ANCHOR_PAIRS, _FAN_WALK_PROVIDER, _FAN_WALK_WORKDIR
    active = _recourse_invest_active(workdir, ctx=ctx, provider=provider)
    pairs = _fan_branch_pairs(workdir, ctx=ctx, provider=provider)
    walk = active or pairs is not None
    _FAN_WALK_PROVIDER = provider if walk else None
    _FAN_WALK_WORKDIR = (
        Path(workdir) if walk and workdir is not None else None)
    _RECOURSE_ANCHOR_PAIRS = pairs if active else None


def _anchor_expand_explicit(explicit_lf: pl.LazyFrame) -> pl.LazyFrame:
    """§4.1 — add branch rows inheriting each anchor's per-(e, d) value.

    ``explicit_lf`` is an ``(e, d, <value…>)`` lazy frame keyed by REAL
    anchor periods (any value-column name(s); all are preserved).  When
    the recourse anchor-pairs holder is set, each
    ``(e, anchor, v)`` row spawns ``(e, br, v)`` for every in-use branch
    ``br`` of ``anchor``; a branch that already carries its OWN explicit
    row keeps it (branch-native wins, via order-preserving dedupe).
    Holder ``None`` (flag-off / deterministic) → returns the frame
    unchanged (byte-parity).  Operates in Utf8 on ``d`` then restores the
    original dtype.  The Utf8→Enum round-trip is lossless for anchor rows
    by construction (their ``d`` arrived already typed as ``d_dtype``, so
    each token is in-vocab), and every inherited branch (``br``) token is
    covered by the 4.0.4 vocabulary splice; a nulled token therefore
    signals a wiring bug and is hard-raised (mirroring
    :func:`._branch.d_leaf_lf`) rather than silently dropped.
    """
    pairs = _RECOURSE_ANCHOR_PAIRS
    if pairs is None or pairs.height == 0:
        return explicit_lf
    d_dtype = explicit_lf.collect_schema().get("d")
    base = explicit_lf.with_columns(pl.col("d").cast(pl.Utf8))
    inherited = (base.join(pairs.lazy(),
                              left_on="d", right_on="anchor", how="inner")
                     .drop("d")
                     .rename({"br": "d"})
                     .select(base.collect_schema().names()))
    combined = pl.concat([base, inherited], how="vertical_relaxed")
    out = combined.unique(subset=["e", "d"], keep="first", maintain_order=True)
    if d_dtype is not None and d_dtype != pl.Utf8:
        # Restore the original (non-Utf8, i.e. Enum) ``d`` dtype.  A
        # ``strict=False`` cast would SILENTLY null any token absent from
        # the live ``d``-axis vocabulary; hard-raise instead so a splice
        # gap surfaces loudly (never fires in valid runs — see docstring).
        out_df = out.collect()
        casted = out_df.with_columns(pl.col("d").cast(d_dtype, strict=False))
        if casted["d"].null_count():
            bad = sorted({v for v, c in zip(out_df["d"].to_list(),
                                            casted["d"].to_list())
                          if c is None})
            raise ValueError(
                f"_anchor_expand_explicit: axis-enum cast nulled period "
                f"token(s) {bad} — the token is missing from the live "
                "d-axis vocabulary (wiring bug; the 4.0.4 vocabulary splice "
                "includes every continuation branch-period token)."
            )
        return casted.lazy()
    return out


def _has_branch_invest_axis(flex_data: "object | None") -> bool:
    """Capability conjunct (Slice B §8, design §2): True iff the invest
    axis actually carries synthetic branch members.

    Reuses :func:`_stochastic_detect.is_genuinely_stochastic` — a branch
    period enters ``period_invest`` iff it is a synthetic ``period__branch``
    token that is also in ``period_in_use_set``.  Keeps the lineage filter
    provably inert on any solve whose invest axis carries no branch member
    even if the flag was authored (defense-in-depth behind Guard 1).
    """
    if flex_data is None:
        return False
    from flextool.engine.preprocess._stochastic_detect import (
        is_genuinely_stochastic as _igs,
    )
    return _igs(flex_data)


# ===========================================================================
# Γ.3.C integration entrypoint
# ===========================================================================


DERIVED_C_FIELDS = (
    # §3.7
    "ed_invest_set", "ed_divest_set",
    "cohort", "cohort_alive", "dd_forward",
    "p_entity_max_units",
    # §3.8
    "p_section",
    "uptime_lookback", "downtime_lookback",
    "pdt_uptime_set", "pdt_downtime_set",
    # §3.12
    "process_group_inside_nonSync",
    "p_group_capacity_for_scaling", "p_inv_group_cap",
    "p_node_capacity_for_scaling",
    "p_positive_inflow", "p_negative_inflow",
    "pdtNodeInflow_per_step",
    # §3.1.3 multi-year (extends Γ.3.A)
    "p_inflation_op",
)


def _set_eq_sorted(a: pl.DataFrame | None,
                     b: pl.DataFrame | None,
                     keys: tuple[str, ...]) -> bool:
    """Frame equality on a *set-shaped* DataFrame (no value column).

    Sort by ``keys`` then call ``.equals``.  Either being None or empty
    → equal iff the other is None / empty.
    """
    a_empty = a is None or a.height == 0
    b_empty = b is None or b.height == 0
    if a_empty and b_empty:
        return True
    if a_empty or b_empty:
        return False
    cols_a = set(a.columns)
    cols_b = set(b.columns)
    if cols_a != cols_b:
        return False
    sk = [k for k in keys if k in cols_a]
    return a.sort(sk).equals(b.sort(sk))


def _build_commission_lf(source: "InputSource",
                             active_solve: str | None,
                             ed_lf: "pl.LazyFrame",
                             period_in_use,
                             workdir: "Path | None",
                             *,
                             provider: "object | None" = None,
                             ) -> "pl.LazyFrame | None":
    """The start-of-operation frame of the investment decisions (construction lead time), failures surfaced.

    Shared by :func:`apply_derived_c`, :func:`apply_synthetic_invest_sets` and the per-solve emit's investment
    window (``emit/_per_solve._invest_window``).  Returns the frame of
    :func:`._existing.commissioning_year_lf` on the run calendar the sub-solve ``provider`` carries, or ``None``
    when there are no in-use periods or no decision has a lead time.  ``provider`` gives only the run calendar;
    the solve's year calendar is resolved as the availability walks resolve it.

    A :class:`CommissioningLagError` from the helper propagates unchanged.  Any other helper failure is
    swallowed to ``None`` only when the model configures no construction lead time (byte-parity for models
    without one); on a lag-configured model it is re-raised as :class:`CommissioningLagError`, so a
    misconfiguration never turns into a silently unlagged solve.
    """
    if not period_in_use:
        return None
    from flextool.engine.preprocess.params._existing import (
        _construction_lag_configured as _lag_configured,
        commissioning_year_lf as _commissioning_year_lf,
    )
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    try:
        return _commissioning_year_lf(
            source, active_solve, ed_lf, list(period_in_use),
            workdir=workdir, run_calendar=read_run_calendar(provider))
    except CommissioningLagError:
        raise
    except Exception as exc:
        if _lag_configured(source):
            raise CommissioningLagError(
                "Finding the start of operation of investments with a construction_lead_time failed; refusing "
                "to solve the model without the lead time."
            ) from exc
        return None


def _offer_invest_set(flex_data: object,
                         source: "InputSource",
                         active_solve: str | None,
                         ed_inv: "pl.DataFrame | None",
                         period_in_use,
                         workdir: "Path | None",
                         *,
                         provider: "object | None" = None,
                         ) -> tuple["pl.DataFrame | None", "pl.LazyFrame | None"]:
    """``(ed_invest offered, commission frame)``: drop the investment decisions whose capacity would start to
    operate after the solve or the model horizon, and those that would never operate (an end of life rounded to
    their own start of operation, :func:`._cohorts.never_operating_investments`), each with a warning; keep the
    start of operation of the rest on ``flex_data.ed_invest_commission`` (``[e, d, d_c, yr_c]``, the offered
    decisions with a lead time: the start period and its year on the solve calendar; ``None`` without).  Every set
    of investment variables, rows and cost terms is built from the offered set.

    When decisions are dropped, the invest-set fields of ``flex_data`` are cleared, so no earlier value (seed)
    brings them back where the offered set leaves one empty."""
    flex_data.ed_invest_commission = None
    if ed_inv is None or ed_inv.height == 0 or not period_in_use:
        return ed_inv, None
    from flextool.engine.preprocess.params._cohorts import never_operating_investments
    from flextool.engine.preprocess.params._existing import (
        offered_invest_set as _offered,
        warn_investments_not_offered as _warn,
    )
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar
    commission_lf = _build_commission_lf(
        source, active_solve, ed_inv.lazy().select("e", "d"), period_in_use, workdir, provider=provider)
    offered, late = ed_inv, None
    if commission_lf is not None:
        from flextool.engine import _behaviour_changes
        lagged = commission_lf.select(pl.col("e").cast(pl.Utf8)).unique().sort("e").collect()["e"].to_list()
        _behaviour_changes.notice("construction_lead_time_commissioning", "entities", lagged)
        offered, late = _offered(ed_inv, commission_lf)
        _warn(source, active_solve, late, list(period_in_use), read_run_calendar(provider))
    never = never_operating_investments(source, active_solve, offered, workdir, provider=provider,
                                        commission_lf=commission_lf)
    if never.height > 0:
        offered, never = align_join_dtypes(offered, never, ["e", "d"])
        offered = offered.join(never, on=["e", "d"], how="anti")
    if commission_lf is not None:
        kept = commission_lf.filter(pl.col("yr_c").is_finite()).select("e", "d", "d_c", "yr_c").collect()
        if never.height > 0:
            kept, never = align_join_dtypes(kept, never, ["e", "d"])
            kept = kept.join(never, on=["e", "d"], how="anti")
        flex_data.ed_invest_commission = kept if kept.height > 0 else None
    if late is not None or never.height > 0:
        # The sets built from ed_invest_set are set below only where they have rows; clear the earlier values
        # (seeds, also the seeded investment cohorts) so a set left empty by the dropped decisions does not keep
        # them.
        for name in ("ed_invest_set", "pd_invest_set", "nd_invest_set", "ed_invest_period_set",
                     "cohort", "cohort_alive"):
            setattr(flex_data, name, None)
    return offered, commission_lf


def _apply_lead_time_sets(flex_data: object,
                              source: "InputSource",
                              active_solve: str | None,
                              ed_inv: "pl.DataFrame | None",
                              commission_lf: "pl.LazyFrame | None",
                              period_in_use,
                              workdir: "Path | None",
                              *,
                              lineage: "pl.DataFrame | None" = None,
                              ) -> None:
    """``flex_data.edd_invest_commission`` / ``edd_invest_in_construction`` / ``e_construction_lead_time`` (``[e, L]``,
    the longest lead time of the entity's decisions) of a solve where some investment decision has a construction
    lead time (offered or not); all ``None`` otherwise, so the model keeps the formulation without lead times.

    Then checks the period maps of the per-period investment limits against the periods in which the solve's
    investments start to operate (:func:`._direct.require_invest_period_coverage`)."""
    _set_lead_time_sets(flex_data, source, active_solve, ed_inv, commission_lf, period_in_use, workdir,
                        lineage=lineage)
    from flextool.engine.preprocess.params._direct import require_invest_period_coverage
    require_invest_period_coverage(source, flex_data, active_solve)


def _set_lead_time_sets(flex_data: object,
                        source: "InputSource",
                        active_solve: str | None,
                        ed_inv: "pl.DataFrame | None",
                        commission_lf: "pl.LazyFrame | None",
                        period_in_use,
                        workdir: "Path | None",
                        *,
                        lineage: "pl.DataFrame | None" = None,
                        ) -> None:
    """The lead-time sets of :func:`_apply_lead_time_sets`."""
    flex_data.edd_invest_commission = None
    flex_data.edd_invest_in_construction = None
    flex_data.e_construction_lead_time = None
    if commission_lf is None or ed_inv is None or ed_inv.height == 0:
        return
    from flextool.engine.preprocess.params._existing import lead_time_sets as _lead_time_sets
    commission, in_construction = _lead_time_sets(
        source, active_solve, ed_inv, invest_alive(getattr(flex_data, "cohort_alive", None)), commission_lf,
        list(period_in_use), workdir, lineage=lineage)
    flex_data.edd_invest_commission = commission if commission.height > 0 else None
    flex_data.edd_invest_in_construction = in_construction if in_construction.height > 0 else None
    lagged = (commission_lf.select(pl.col("e").cast(pl.Utf8), "L").group_by("e").agg(pl.col("L").max())
                           .sort("e").collect())
    flex_data.e_construction_lead_time = lagged if lagged.height > 0 else None


def _fan_branch_years_notice(source: "InputSource",
                             flex_data: object,
                             ed_inv: "pl.DataFrame | None",
                             ed_div: "pl.DataFrame | None") -> None:
    """Behaviour-change notice of a solve with a stochastic fan without recourse: its branch periods now have the
    years of their period labels in the investment, retirement and lifetime windows.  Names the entities with
    investment or retirement options and those with a ``reinvest_choice`` / ``no_reinvestment`` lifetime."""
    if _fan_walk_provider() is None or getattr(flex_data, "recourse_invest", False):
        return
    names: dict[str, None] = {}
    for frame in (ed_inv, ed_div):
        if frame is not None and frame.height > 0:
            for e in frame["e"].cast(pl.Utf8).unique().sort().to_list():
                names.setdefault(str(e), None)
    for e, method in _entity_methods(source).items():
        if method in BOUNDED_METHODS:
            names.setdefault(e, None)
    if names:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("stochastic_branch_period_years", "entities", sorted(names))


def _retirement_without_existing_warning(source: "InputSource", offered: pl.DataFrame) -> None:
    """Warn about the entities that have a retirement option but no ``existing`` capacity in any period (none
    given, or 0 everywhere) and no investment option (no model-built capacity to retire either): the option has no
    effect — no retirement variable, and their retired-capacity terms (user constraints, group retirement limits)
    are 0.  Each entity once per run.  An entity with existing capacity whose retirement periods are past its end of
    life is a valid model and is not warned about (the end of life is in the retirement notice)."""
    from flextool import run_warnings
    from flextool.engine.preprocess.params._cohorts import LIFETIME_METHODS
    from flextool.engine.preprocess.params._lifetime_inputs import (
        _entity_invest_set_lf,
        _lifetime_method_with_default_lf,
        _per_entity_param_lf,
    )
    with_existing = (_per_entity_param_lf(source, "existing")
                       .filter(pl.col("value").cast(pl.Float64, strict=False) > 0.0)
                       .select("e").unique().collect())
    entities = offered.select("e").unique()
    entities, with_existing = align_join_dtypes(entities, with_existing, ["e"])
    lost = entities.join(with_existing, on="e", how="anti")
    if lost.height == 0:
        return
    bounded = (_lifetime_method_with_default_lf(source, lost.lazy())
               .filter(pl.col("method").is_in(list(LIFETIME_METHODS))).select("e"))
    bounded, investing = align_join_dtypes(bounded, _entity_invest_set_lf(source, _INVEST_NOT_ALLOWED), ["e"])
    built = bounded.join(investing, on="e", how="semi").collect()
    lost, built = align_join_dtypes(lost, built, ["e"])
    names = sorted(str(e) for e in lost.join(built, on="e", how="anti")["e"].to_list())
    run_warnings.warn_grouped("W-0584", names, per_run=True)


def _check_retirement_minimums(flex_data: object,
                               source: "InputSource",
                               active_solve: str | None,
                               workdir: Path | None,
                               offered: "pl.DataFrame | None",
                               kept: "pl.DataFrame | None",
                               active_frames: "list[pl.DataFrame | None]",
                               period_in_use: list[str],
                               *,
                               provider: "object | None" = None) -> None:
    """:func:`._retire_existing.check_retirement_minimums` on the solve's retirement options."""
    if offered is None or offered.height == 0:
        return
    from flextool.engine.preprocess.params._existing import (
        e_divest_total_lf,
        ed_divest_period_set_lf,
        p_entity_pre_existing_lf,
        p_entity_retired_carried_from_provider,
    )
    from flextool.engine.preprocess.params._retire_existing import (
        active_windows,
        check_retirement_minimums,
        ed_values,
    )
    pre_df = (p_entity_pre_existing_lf(source, active_solve, [str(d) for d in period_in_use],
                                       workdir=workdir, provider=provider)
                .select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), "value").collect())
    pre = {(e, d): float(v) for e, d, v in pre_df.iter_rows()}
    from flextool.engine.preprocess.params._retire_built import check_minimums_inputs
    built_options, built_total, built_alive = check_minimums_inputs(getattr(flex_data, "cohort_divest_built", None),
                                                                    getattr(flex_data, "cd_divest_built", None))
    check_retirement_minimums(
        offered=offered, kept=kept,
        period_method_pairs=ed_divest_period_set_lf(source, offered.lazy()).collect(),
        total_method_entities=e_divest_total_lf(source).collect()["e"].cast(pl.Utf8).to_list(),
        windows=active_windows(active_frames), pre=pre,
        min_period=getattr(flex_data, "ed_divest_min_period", None),
        min_total=getattr(flex_data, "e_divest_min_total", None),
        prior=p_entity_divested_from_handoff(provider=provider),
        carried=ed_values(p_entity_retired_carried_from_provider(workdir, provider=provider)),
        eol=existing_ends(existing_cohorts(source, active_solve, workdir, provider=provider)),
        built_options=built_options, built_total=built_total, built_alive=built_alive)


def _end_of_life_notice(source: "InputSource",
                        active_solve: str | None,
                        workdir: Path | None,
                        *,
                        provider: "object | None" = None) -> None:
    """Behaviour-change notice ``retirement_of_existing_capacity_only`` for the end of life of existing capacity:
    names the reinvest_choice / no_reinvestment entities of a solve whose end of life changed, a later solve of a
    chain (its period history starts before its own first period) or a solve whose first period branches."""
    from flextool.engine import _behaviour_changes
    if not _behaviour_changes.has_marker("retirement_of_existing_capacity_only"):
        return
    first = _read_period_first(source, active_solve, workdir, provider=provider)
    years = cohort_years(source, active_solve, workdir, provider=provider)
    start = run_start_period(years)
    later_in_chain = bool(first) and start is not None and years.get(first[0], 0.0) > years.get(start, 0.0)
    if not (later_in_chain or len(first) > 1):
        return
    names = sorted(e for e, y_end in existing_eol(existing_cohorts(source, active_solve, workdir, provider=provider,
                                                                   years=years)).items() if y_end > 0.0)
    if names:
        _behaviour_changes.notice("retirement_of_existing_capacity_only", "entities", names)


def _retirement_notice(offered: pl.DataFrame, kept: pl.DataFrame) -> None:
    """Behaviour-change notice ``retirement_of_existing_capacity_only``: names the entities with a retirement
    option (and, in the detail, those left without one for lack of existing capacity)."""
    from flextool.engine import _behaviour_changes
    names = sorted(set(offered["e"].cast(pl.Utf8).to_list()))
    lost = sorted(set(names) - set(kept["e"].cast(pl.Utf8).to_list()))
    detail = ("Without existing capacity in the retirement periods (no retirement variable): "
              + ", ".join(lost) + ".") if lost else ""
    _behaviour_changes.notice("retirement_of_existing_capacity_only", "entities", names, detail)


#: User-constraint coefficients that offered an entity's investment (retirement) variable before
#: ``invest_without_cost_offered`` whatever its cost: such entities are not named by the notice.
_OLD_RULE_INVEST_COEFFS: tuple[str, ...] = (
    "constraint_invested_capacity_coeff",
    "constraint_cumulative_pre_built_capacity_coeff",
    "constraint_capacity_coeff",
    "constraint_in_construction_capacity_coeff",
)


_OLD_RULE_DIVEST_COEFFS: tuple[str, ...] = (*_OLD_RULE_INVEST_COEFFS, "constraint_retired_capacity_coeff")


def _zero_cost_pairs(source: "InputSource", pairs: "pl.DataFrame | None", cost_param: str,
                     coeffs: tuple[str, ...], commission: "pl.DataFrame | pl.LazyFrame | None" = None,
                     ) -> dict[str, list[str]]:
    """``{entity: [periods]}`` of the ``pairs`` (``[e, d]``) whose ``cost_param`` is missing or 0, leaving out the
    entities a user-constraint capacity coefficient names (their variable existed before).  The cost is that of the
    period the capacity comes online (``commission`` ``[e, d, d_c, ...]``, the decisions with a construction lead
    time, :func:`._period_values.ed_values_online`; the period itself without one)."""
    if pairs is None or pairs.height == 0:
        return {}
    coeff_entities: set[str] = set()
    for cls in ("unit", "node", "connection"):
        for pname in coeffs:
            df = _try_param(source, cls, pname)
            if df is not None and "name" in df.columns:
                coeff_entities.update(df.get_column("name").cast(pl.Utf8).to_list())
    cost = ed_values_online(source, cost_param, pairs.lazy().select("e", "d"), 0.0, commission).collect()
    out: dict[str, list[str]] = {}
    for e, d, v in cost.select("e", "d", "value").iter_rows():
        if str(e) not in coeff_entities and v == 0.0:
            out.setdefault(str(e), []).append(str(d))
    return out


def zero_cost_invest_notice(source: "InputSource", ed_invest: "pl.DataFrame | None",
                            ed_divest: "pl.DataFrame | None",
                            commission: "pl.DataFrame | pl.LazyFrame | None" = None) -> None:
    """Behaviour-change notice ``invest_without_cost_offered``: names the entities that now get an investment
    variable where their ``invest_cost`` is missing or 0 (at the period the capacity comes online: ``commission``,
    the start of operation of a construction lead time), or a retirement variable where their
    ``net_salvage_value`` is (before, such a period had no variable unless a user-constraint capacity coefficient
    named the entity)."""
    from flextool.engine import _behaviour_changes
    if not _behaviour_changes.has_marker("invest_without_cost_offered"):
        return
    invest = _zero_cost_pairs(source, ed_invest, "invest_cost", _OLD_RULE_INVEST_COEFFS, commission)
    divest = _zero_cost_pairs(source, ed_divest, NET_SALVAGE_VALUE, _OLD_RULE_DIVEST_COEFFS)
    # A later solve of a multi-solve / rolling run lists only the entities not yet announced.
    fresh = set(_behaviour_changes.unannounced("invest_without_cost_offered", sorted(set(invest) | set(divest))))
    invest = {e: ds for e, ds in invest.items() if e in fresh}
    divest = {e: ds for e, ds in divest.items() if e in fresh}

    def _listed(found: dict[str, list[str]]) -> str:
        return ", ".join(f"{e} ({', '.join(ds)})" for e, ds in sorted(found.items()))

    detail = []
    if invest:
        detail.append("Investment without invest_cost (entity, periods): " + _listed(invest) + ".")
    if divest:
        detail.append(f"Retirement without {NET_SALVAGE_VALUE} (entity, periods): " + _listed(divest) + ".")
    _behaviour_changes.notice("invest_without_cost_offered", "entities", sorted(set(invest) | set(divest)),
                              " ".join(detail))


def _class_partitions(source: "InputSource", ed_set: "pl.DataFrame | None",
                      partition_fns: tuple) -> tuple["pl.DataFrame | None", ...]:
    """The class partitions of the (e, d) set *ed_set*, one frame (None when empty) per function of
    *partition_fns* (each ``fn(source, ed_lf) -> LazyFrame``); all None when *ed_set* is None or empty."""
    if ed_set is None or ed_set.height == 0:
        return tuple(None for _ in partition_fns)
    ed_lf = ed_set.lazy()
    frames = (fn(source, ed_lf).collect() for fn in partition_fns)
    return tuple(frame if frame.height > 0 else None for frame in frames)


def _seeded_invest(flex_data: object) -> "pl.DataFrame | None":
    """``[e, d_invest, d]`` of the investment cohorts the loader seeded from ``edd_invest`` (``None`` without):
    the cohorts of a solve whose investment walk finds no period in service."""
    seed = invest_alive(getattr(flex_data, "cohort_alive", None))
    return seed if seed.height > 0 else None


def apply_derived_c(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.C Derived Params, mutating ``flex_data`` in place.

    Order:
      1. Multi-year inflation (extends Γ.3.A's default-rate path).
      2. §3.12 group slack (smallest, foundational for cap_margin).
      3. §3.8 online/UC.
      4. §3.7 invest/divest cascade.
      5. §3.11 (capped by §3.7.5/6 — no separate Params).

    Δ.3 replaced the previous ``derived_overrides_c`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    dt_csv = getattr(flex_data, "dt", None)
    sd_csv = getattr(flex_data, "p_step_duration", None)

    # Slice D — resolve the per-scenario stochastic-invest flag once and
    # stamp it on FlexData (survives the region-filter dataclasses.replace)
    # so the model layer reads ``d.recourse_invest`` without a workdir touch.
    flex_data.recourse_invest = _recourse_invest_active(
        workdir, ctx=ctx, provider=provider)
    # Slice D §4 — arm the per-period anchor-map scope for the invest/
    # divest set predicates + cap readers built below (no-op flag-off).
    _enter_recourse_anchor_scope(workdir, ctx=ctx, provider=provider)
    # Slice D §8 (E) — the scenario-lineage frame for the edd builders.
    # Active only under recourse AND a genuine branch invest axis — the
    # anchor-pairs holder just armed above is exactly that conjunct
    # (non-None ⟺ recourse-active AND synthetic branches in PIU).  Build
    # the lineage frame ON-DEMAND from the provider (``dd_same_scenario_df``
    # is a standalone builder): ``flex_data.dd_same_scenario`` is not
    # populated until ``apply_branch_cluster`` in derived_g, which runs
    # AFTER this boundary.  Hoist the precondition check ABOVE the edd
    # try/except blocks (§6) so a LineageFilterError is loud, not swallowed.
    # ``None`` flag-off → edd builders keep today's unfiltered behaviour.
    if _RECOURSE_ANCHOR_PAIRS is not None:
        from flextool.engine.preprocess.params._branch import (
            assert_recourse_npv_preconditions as _assert_recourse,
            dd_same_scenario_df as _dd_same_scenario_df,
        )
        _c_lineage = _dd_same_scenario_df(
            workdir, source, active_solve, ctx=ctx, provider=provider)
        _assert_recourse(workdir, source, active_solve,
                         ctx=ctx, provider=provider)
    else:
        _c_lineage = None

    # Δ.12b — assignment is unconditional except for fields with
    # documented helper-coverage gaps (multi-year cascade extends
    # apply_derived_a's path; fall-throughs noted inline).

    # ─── Multi-year inflation cascade ─────────────────────────────────
    # TODO(Δ.12b helper-fix): the multi-year helper EXTENDS rather than
    # replaces apply_derived_a's p_inflation_op (single-year path) — when
    # the multi-year cascade isn't applicable (single-year fixture) the
    # helper returns None and we want to retain apply_derived_a's value.
    # Keep the conditional assignment.
    if active_solve is not None and dt_csv is not None:
        infl_my = p_inflation_op_multi_year_from_source(
            source, active_solve, dt_csv)
        if infl_my is not None:
            flex_data.p_inflation_op = infl_my

    # ─── §3.12 group slack ────────────────────────────────────────────
    # TODO(Δ.12b helper-fix): the helpers cover the scaling-inactive
    # default path; scaling-active fixtures (``solve.use_row_scaling``)
    # require the pow10 cascade in ``lp_scaling_params.py:91-244`` which
    # the seed-side ``_group_slack`` reads directly from
    # ``solve_data/group_capacity_for_scaling.csv`` /
    # ``solve_data/inv_group_cap.csv``.  Keep the conditional assignment.
    gcs_db = p_group_capacity_for_scaling_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if gcs_db is not None:
        flex_data.p_group_capacity_for_scaling = gcs_db
    igc_db = p_inv_group_cap_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if igc_db is not None:
        flex_data.p_inv_group_cap = igc_db
    # Δ.29 — tactical default-1.0 path for p_node_capacity_for_scaling.
    # The proper fix is the lp_scaling_params pow10 cascade
    # (``preprocessing/lp_scaling_params.py:91-244``); until that lands,
    # default to 1.0 broadcast across (nodeBalance, period_in_use).
    # Without this default, the fast path leaves the field None for
    # fixtures with no preprocessed CSV → the slack penalty becomes
    # the only sink for unmet demand without the row-scaling factor,
    # but more critically when the pow10 cascade is active CSV-side and
    # the fast path emits None, the LP misses the scaling on penalty
    # terms (penalty becomes a free source of energy).  Default 1.0
    # produces the same numeric coefficient as the None-branch in
    # ``model._obj_balance_slack`` but exposes the field uniformly so future
    # consumers (e.g. region filter, warm cascade) see a populated
    # frame.
    ncs_db = p_node_capacity_for_scaling_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if ncs_db is not None:
        flex_data.p_node_capacity_for_scaling = ncs_db

    # Group-accounting inflow frames — from the real inflow (before the invest-solve energy margin; the
    # CSV-loaded or override-set p_inflow when no solve has a margin).  Helpers return None when it is None.
    inflow_real = real_inflow(getattr(flex_data, "p_inflow", None), getattr(flex_data, "p_inflow_unmargined", None))
    flex_data.p_positive_inflow = p_positive_inflow_from_inflow(inflow_real)
    flex_data.p_negative_inflow = p_negative_inflow_from_inflow(inflow_real)
    flex_data.pdtNodeInflow_per_step = pdtNodeInflow_per_step_from_inflow(inflow_real, sd_csv)

    # process_group_inside_nonSync — internal transfers of the non-synchronous groups.
    flex_data.process_group_inside_nonSync = process_group_inside_nonSync_from_flex(flex_data)

    # ─── §3.8 online / UC ─────────────────────────────────────────────
    classified = _classify_process_method(source)
    if dt_csv is not None and classified is not None and classified.height > 0:
        flex_data.p_section = p_section_from_source(source, dt_csv, classified)

    # The uptime / downtime look-backs follow the ramp predecessor links, which apply_derived_e builds from the
    # source timeline; they are set there.

    # ─── §3.7 invest / divest ─────────────────────────────────────────
    # Δ.12b — set frames (None or non-empty); keep height>0 as a
    # structural filter to preserve the SET-frame contract.
    ed_inv_db = ed_invest_set_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    # Construction lead time: the start of operation of each investment decision, on the run calendar.  The
    # decisions whose capacity would start to operate after the solve or the model horizon are not offered.
    # The frame is threaded into the availability walk of the investment cohorts below;
    # ``None`` when no decision has a lead time (byte-parity).
    _c_period_in_use = _period_in_use_set(source, active_solve, workdir,
                                             ctx=ctx, provider=provider)
    ed_inv_db, commission_lf = _offer_invest_set(
        flex_data, source, active_solve,
        ed_inv_db if ed_inv_db is not None else getattr(flex_data, "ed_invest_set", None),
        _c_period_in_use, workdir, provider=provider)
    if ed_inv_db is not None and ed_inv_db.height > 0:
        flex_data.ed_invest_set = ed_inv_db

    offered_div = ed_divest_set_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider, existing_only=False)
    ed_div_db = offered_div
    if offered_div is not None and offered_div.height > 0:
        # Only pre-existing capacity can be retired: the retirement options of periods without it are dropped.
        ed_div_db = retirable_existing_pairs(offered_div, source, active_solve, workdir, provider=provider)
        _retirement_without_existing_warning(source, offered_div)
        _retirement_notice(offered_div, ed_div_db)
        flex_data.ed_divest_set = ed_div_db if ed_div_db.height > 0 else None
        if ed_div_db.height == 0:
            # The seeds of the retirement partitions describe the options before the filter.
            flex_data.pd_divest_set = flex_data.nd_divest_set = flex_data.ed_divest_period_set = None
    zero_cost_invest_notice(source, ed_inv_db, ed_div_db, commission_lf)

    # pd/nd_invest_set, pd/nd_divest_set, ed_invest/divest_period_set: the
    # class partitions of the effective ed_*_set (the one just produced,
    # or the seed-loaded one when this run contributes nothing, the
    # synthetic-name case).  A partition is always rebuilt from that set
    # and never kept from its seed: the seeds are partitions of the
    # unfiltered ``entityInvest × period_invest`` (``emit/_per_solve``),
    # so a kept seed would hold the (entity, period) pairs the
    # ``no_reinvestment`` filter removed from ``ed_invest_set`` (an
    # investment variable for a closed window, or ``p_entity_max_units``
    # missing for an investment the model still builds).  With no
    # ed_*_set row there is no partition.
    from flextool.engine.preprocess.params._existing import (
        pd_invest_set_lf as _pd_invest_lf,
        nd_invest_set_lf as _nd_invest_lf,
        pd_divest_set_lf as _pd_divest_lf,
        nd_divest_set_lf as _nd_divest_lf,
        ed_invest_period_set_lf as _ed_invest_period_lf,
        ed_divest_period_set_lf as _ed_divest_period_lf,
    )
    ed_inv_for_partition = (
        ed_inv_db if (ed_inv_db is not None and ed_inv_db.height > 0)
        else getattr(flex_data, "ed_invest_set", None))
    ed_div_for_partition = (
        ed_div_db if (ed_div_db is not None and ed_div_db.height > 0)
        else getattr(flex_data, "ed_divest_set", None))
    flex_data.pd_invest_set, flex_data.nd_invest_set, flex_data.ed_invest_period_set = _class_partitions(
        source, ed_inv_for_partition, (_pd_invest_lf, _nd_invest_lf, _ed_invest_period_lf))
    flex_data.pd_divest_set, flex_data.nd_divest_set, flex_data.ed_divest_period_set = _class_partitions(
        source, ed_div_for_partition, (_pd_divest_lf, _nd_divest_lf, _ed_divest_period_lf))

    _fan_branch_years_notice(source, flex_data, ed_inv_for_partition, ed_div_for_partition)
    _end_of_life_notice(source, active_solve, workdir, provider=provider)
    carried_cohort, carried_retire = _carried_cohorts(provider)

    # Dispatch-only gate: when neither ed_invest nor ed_divest carries
    # any (e, d) row, mirror ``flex_data/_load_features.py::_load_invest`` blank-out by
    # nulling every invest-cascade Param.
    #
    # Δ.18 — gate must consult the *effective* invest/divest sets (the
    # seed-loaded value persists when the override returned empty, e.g.
    # for synthetic per-sub-solve names that don't exist in Spine).  Using
    # ed_inv_db / ed_div_db alone fired the blank-out spuriously on
    # per-sub-solve snapshot fixtures, nulling p_entity_max_units etc.
    # even though the snapshot CSV had genuine invest activity.
    inv_empty = ed_inv_for_partition is None or ed_inv_for_partition.height == 0
    div_empty = ed_div_for_partition is None or ed_div_for_partition.height == 0
    # A cohort an earlier solve built can be retired here even without an investment or existing-capacity option.
    if inv_empty and div_empty and not _carried_retirable(offered_div, carried_cohort) \
            and not _extending(source):
        _check_retirement_minimums(flex_data, source, active_solve, workdir, offered_div, ed_div_db, [],
                                   _c_period_in_use, provider=provider)
        for blank_field in (
            "e_invest_total", "e_divest_total",
            "e_invest_max_total", "e_divest_max_total",
            "e_invest_min_total", "e_divest_min_total",
            "p_entity_max_units",
            "ed_invest_period_set", "ed_divest_period_set",
            "ed_invest_max_period", "ed_divest_max_period",
            # Read only where an investment or retirement variable exists;
            # nulled so no seed outlives the empty investment set.
            "cohort", "cohort_alive", "dd_forward",
            "cd_divest_built", "cdd_divest_built", "cohort_divest_built",
            "cd_extend", "cdd_extend", "cohort_extend",
        ):
            setattr(flex_data, blank_field, None)
        return

    # The cohort table over the (possibly-overlaid) ed_invest_set: the existing capacity and each investment
    # decision with the periods it is in service, and the reach of a retirement.  A failure here propagates: a
    # missing set would drop investments or retirements from the capacity.
    ed_inv_used = ed_inv_db if (ed_inv_db is not None and ed_inv_db.height > 0) \
                   else getattr(flex_data, "ed_invest_set", None)
    frames = build_cohort_frames(source, active_solve, ed_inv_used, list(_c_period_in_use), workdir,
                                 provider=provider, lineage=_c_lineage, commission_lf=commission_lf,
                                 seed=_seeded_invest(flex_data))
    # The life extensions of the cohorts that end their life in an investment period: cohorts of their own.
    frames = _apply_extensions(flex_data, source, active_solve, workdir, frames, carried_cohort, carried_retire,
                               provider=provider)
    flex_data.cohort, flex_data.cohort_alive, flex_data.dd_forward = (
        frames.cohort, frames.cohort_alive, frames.dd_forward)
    # Construction lead time: the period each decision starts to operate in and the periods it is in construction.
    _apply_lead_time_sets(flex_data, source, active_solve, ed_inv_used, commission_lf, _c_period_in_use, workdir,
                          lineage=_c_lineage)

    # The periods a retirement of existing capacity counts in (the model builds the same, model/_capacity.py).
    edda_db = retire_active(getattr(flex_data, "pd_divest_set", None), frames.cohort_alive, frames.dd_forward, "p")
    ndda_db = retire_active(getattr(flex_data, "nd_divest_set", None), frames.cohort_alive, frames.dd_forward, "n")
    # The retirement options of model-built capacity: the per-period retirement limits apply to them too.
    built = _apply_built_retirement(flex_data, source, active_solve, workdir, frames, offered_div, carried_cohort,
                                    carried_retire, _c_period_in_use, provider=provider)
    if built is not None:
        flex_data.ed_divest_period_set = _class_partitions(
            source, _with_built_options(ed_div_for_partition, built.options), (_ed_divest_period_lf,))[0]

    # p_entity_max_units uses ed_invest plus the per-period-in-use grid
    # and the unitsize cascade.
    try:
        pae_for_max_units = getattr(flex_data, "p_entity_all_existing", None)
        pemu_db = p_entity_max_units_from_source(
            source, ed_inv_used, active_solve, workdir=workdir,
            p_entity_all_existing=pae_for_max_units, ctx=ctx,
            commission=getattr(flex_data, "ed_invest_commission", None))
    except FlexToolConfigError:
        # A shape error from the resolver: keeping the legacy CSV seed would hide it
        # behind a wrong p_entity_max_units (the seed misses recourse branch periods).
        raise
    except Exception:
        pemu_db = None
    if pemu_db is not None:
        flex_data.p_entity_max_units = pemu_db
    if built is not None:
        from flextool.engine.preprocess.params._retire_built import with_bounds
        flex_data.cohort_divest_built = with_bounds(flex_data.cohort_divest_built, flex_data,
                                                    _entity_unitsizes(flex_data))
    _check_retirement_minimums(flex_data, source, active_solve, workdir, offered_div, ed_div_db,
                               [edda_db, ndda_db], _c_period_in_use, provider=provider)


def _carried_cohorts(provider: "object | None") -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """``(realized_cohort, realized_retire)``: the cohorts earlier solves of the run realized and their retirements
    (the handoff, an override first); ``None`` each without rows."""
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    return (read_handoff_frame(provider, K.HANDOFF_REALIZED_COHORT),
            read_handoff_frame(provider, K.HANDOFF_REALIZED_RETIRE))


def _extending(source: "InputSource") -> bool:
    """Whether a unit, node or connection gives a life extension (``_extend.extending_entities``): its cohorts can
    be extended without an investment or retirement option."""
    from flextool.engine.preprocess.params._extend import extending_entities
    return bool(extending_entities(source))


def _apply_extensions(flex_data: object,
                      source: "InputSource",
                      active_solve: str | None,
                      workdir: Path | None,
                      frames: "object",
                      carried: "pl.DataFrame | None",
                      carried_retire: "pl.DataFrame | None",
                      *,
                      provider: "object | None" = None):
    """Set ``cd_extend`` / ``cdd_extend`` / ``cohort_extend`` (:mod:`._extend`) on ``flex_data`` from the solve's
    cohort frames ``frames`` and the cohorts earlier solves realized (``carried`` / ``carried_retire``, the handoff)
    and return the frames with the extension cohorts and their periods in service (``frames`` unchanged, and the
    fields ``None``, without an extension option)."""
    from dataclasses import replace

    from flextool.engine.preprocess.params._cohorts import run_cohort_calendar
    from flextool.engine.preprocess.params._extend import (
        append_extensions,
        existing_in_service_before,
        extension_frames,
    )
    years = cohort_years(source, active_solve, workdir, provider=provider)
    period_invest = _expand_invest_branch_periods(_solve_periods(source, active_solve, "invest_periods"), workdir,
                                                  provider=provider) or []
    cal = run_cohort_calendar(source, active_solve, years, provider) if years else None
    before = (existing_in_service_before(source, active_solve, cal, carried_retire, workdir=workdir, provider=provider)
              if cal is not None else None)
    ext = extension_frames(source, frames.cohort, frames.cohort_alive, frames.dd_forward,
                           [str(d) for d in period_invest], years, cal, carried=carried,
                           carried_retire=carried_retire, existing_before=before)
    flex_data.cd_extend = None if ext is None else ext.options
    flex_data.cdd_extend = None if ext is None else ext.counting
    flex_data.cohort_extend = None if ext is None else ext.parents
    if ext is None:
        return frames
    cohort, alive = append_extensions(frames.cohort, frames.cohort_alive, ext)
    return replace(frames, cohort=cohort, cohort_alive=alive)


def _carried_retirable(offered: "pl.DataFrame | None", carried: "pl.DataFrame | None") -> bool:
    """Whether a carried investment cohort of an entity with a lifetime_method has a retirement option in the solve's
    investment periods (``offered`` ``[e, d]``)."""
    if offered is None or offered.height == 0 or carried is None or carried.height == 0:
        return False
    from flextool.engine.preprocess.params._cohorts import KIND_INVEST, LIFETIME_METHODS
    entities = sorted({str(e) for e in offered["e"].unique().to_list()})
    return carried.filter((pl.col("kind") == KIND_INVEST) & pl.col("method").is_in(list(LIFETIME_METHODS))
                          & pl.col("entity").is_in(entities)).height > 0


def _apply_built_retirement(flex_data: object,
                            source: "InputSource",
                            active_solve: str | None,
                            workdir: Path | None,
                            frames: "object",
                            offered: "pl.DataFrame | None",
                            carried: "pl.DataFrame | None",
                            carried_retire: "pl.DataFrame | None",
                            period_in_use: "list[str]",
                            *,
                            provider: "object | None" = None):
    """Set ``cd_divest_built`` / ``cdd_divest_built`` / ``cohort_divest_built`` (:mod:`._retire_built`) on
    ``flex_data`` from the solve's cohort frames ``frames``; returns the :class:`._retire_built.BuiltRetirement`
    (``None``, and the fields ``None``, without a retirement option of model-built capacity)."""
    from flextool.engine.preprocess.params._retire_built import built_retirable_notice, built_retirement_options
    built = built_retirement_options(frames.cohort, frames.cohort_alive, frames.dd_forward, offered, carried,
                                     carried_retire, cohort_years(source, active_solve, workdir, provider=provider),
                                     [str(d) for d in period_in_use])
    flex_data.cd_divest_built = None if built is None else built.options
    flex_data.cdd_divest_built = None if built is None else built.counting
    flex_data.cohort_divest_built = None if built is None else built.cohorts
    built_retirable_notice(flex_data.cohort_divest_built)
    return built


def _with_built_options(ed_div: "pl.DataFrame | None", options: pl.DataFrame) -> pl.DataFrame:
    """``[e, d]``: the retirement options of existing capacity ``ed_div`` and those of model-built capacity."""
    built = options.select("e", "d").unique()
    if ed_div is None or ed_div.height == 0:
        return built.sort("e", "d")
    ed_div, built = align_join_dtypes(ed_div.select("e", "d"), built, ["e", "d"])
    return pl.concat([ed_div, built]).unique().sort("e", "d")


def _entity_unitsizes(flex_data: object) -> "dict[str, float]":
    """``{entity: unitsize}`` of the processes (``p_unitsize``) and storage nodes (``p_state_unitsize``)."""
    out: dict[str, float] = {}
    for name, col in (("p_unitsize", "p"), ("p_state_unitsize", "n")):
        param = getattr(flex_data, name, None)
        if param is None:
            continue
        for e, v in param.frame.select(col, pl.col("value").cast(pl.Float64)).iter_rows():
            if v is not None:
                out.setdefault(str(e), float(v))
    return out


def apply_synthetic_invest_sets(flex_data: object,
                                     source: "InputSource",
                                     active_solve: str,
                                     synthetic: tuple[str, str],
                                     workdir: "Path | None" = None,
                                     *,
                                     provider: "object | None" = None) -> None:
    """Δ.19 — populate the invest-set FlexData fields for a synthetic
    ``<base>_<anchor>`` solve, replacing the matching disk reads in
    ``_invest_seeds.py``.

    The synthetic-aware :func:`_solve_periods` returns the right period
    list for the anchor (Map-shaped ``invest_periods`` filtered to the
    synthetic anchor); the eager helpers in this module and the lazy
    partition helpers in :mod:`._existing` are pure derivations
    on top of that period list so they work as-is.  Fields populated
    (mirrors the seed names in ``_invest_seeds.py``):

      * ``ed_invest_set`` / ``ed_divest_set``
      * ``pd_invest_set`` / ``nd_invest_set`` (process / node partition
        of ``ed_invest_set``)
      * ``pd_divest_set`` / ``nd_divest_set`` (process / node partition
        of ``ed_divest_set``)
      * ``cohort`` / ``cohort_alive`` (the existing and investment cohorts and
        the periods they are in service; requires workdir for the period
        years + ``period_in_use_set.csv``; no ``dd_forward``).

    The per-period cap subsets (``ed_invest_period_set`` /
    ``ed_divest_period_set``) stay on the ``_invest_seeds.py`` CSV-seed
    path.  The NPV cost lumps are not set here: the investment annuity and
    lifetime fixed cost come from :func:`._npv.apply_invest_lumps` (called
    by :func:`._lifetime_costs.apply_run_fields_of_synthetic_solve`), the divest lumps
    from their emitted ``solve_data`` frames.
    """
    base, anchor = synthetic
    # Slice D §4 — arm the per-period anchor-map scope (no-op flag-off).
    _enter_recourse_anchor_scope(workdir, provider=provider)
    # ed_invest_set / ed_divest_set — eager helpers use _solve_periods
    # which is synthetic-aware (Δ.19); pass the synthetic name through.
    # Γ.6.D forbidden filter is applied internally by the helper.
    ed_inv = ed_invest_set_from_source(
        source, active_solve, workdir=workdir, provider=provider)
    # Construction lead time: drop the decisions whose capacity would start to operate after the solve or the
    # model horizon (with a warning) before every set below is built from ed_invest_set.
    synth_commission_lf = None
    # A synthetic sub-solve (a roll, a nested solve renamed for its period; the solves of a 2-D period map are
    # real solves of the reader and never come here) counts one year per period on its own calendar, so the start
    # of operation and the cost window of a construction lead time cannot be placed in it.
    from flextool.engine.preprocess.params._existing import (
        _construction_lag_configured as _lag_configured,
        lead_time_decisions as _lead_time_decisions,
    )
    if ed_inv is not None and ed_inv.height > 0 and _lag_configured(source):
        lagged = _lead_time_decisions(source, ed_inv.lazy().select("e", "d"))
        if lagged.height > 0:
            longest = lagged.group_by("e").agg(pl.col("L").max()).sort("e")
            entities = ", ".join(f"{e} ({lead:g} years)" for e, lead in longest.iter_rows())
            raise FlexToolConfigError(
                f"Solve '{active_solve}' is a sub-solve of '{base}' (a roll, or a nested solve renamed for its "
                "period) that invests, and FlexTool cannot place the start of operation of a construction lead "
                f"time in such a sub-solve. Entities with a lead time: {entities}. Make the investments in a "
                "single_solve (a solve of its own in model.solves), or remove the lead time.")
    if workdir is not None:
        ed_inv, synth_commission_lf = _offer_invest_set(
            flex_data, source, active_solve,
            ed_inv if ed_inv is not None else getattr(flex_data, "ed_invest_set", None),
            _period_in_use_set(source, active_solve, workdir, provider=provider), workdir, provider=provider)
    if ed_inv is not None and ed_inv.height > 0:
        flex_data.ed_invest_set = ed_inv
    ed_div = ed_divest_set_from_source(
        source, active_solve, workdir, provider=provider)
    if ed_div is not None and ed_div.height > 0:
        flex_data.ed_divest_set = ed_div
    zero_cost_invest_notice(source, ed_inv, ed_div, synth_commission_lf)

    # pd/nd_invest_set, pd/nd_divest_set partitions — pure entity-class
    # projections, no solve dependency beyond the input ed_*_set frame.
    from flextool.engine.preprocess.params._existing import (
        pd_invest_set_lf as _pd_invest_lf,
        nd_invest_set_lf as _nd_invest_lf,
        pd_divest_set_lf as _pd_divest_lf,
        nd_divest_set_lf as _nd_divest_lf,
    )
    if ed_inv is not None and ed_inv.height > 0:
        flex_data.pd_invest_set, flex_data.nd_invest_set = _class_partitions(
            source, ed_inv, (_pd_invest_lf, _nd_invest_lf))
    if ed_div is not None and ed_div.height > 0:
        flex_data.pd_divest_set, flex_data.nd_divest_set = _class_partitions(
            source, ed_div, (_pd_divest_lf, _nd_divest_lf))

    # The cohort table (the periods each investment is in service) over the sub-solve's period_in_use_set.  The
    # sub-solve places no retirement reach (no dd_forward), so the model gives it no retirement and no pre-built
    # capacity (_cohorts.in_service).
    if (ed_inv is not None and ed_inv.height > 0
            and workdir is not None):
        period_in_use = _period_in_use_set(source, active_solve, workdir, provider=provider)
        # The start of operation of the decisions with a construction lead time was found (and the decisions
        # starting after the solve dropped) where ed_invest_set was built above.
        commission_lf = synth_commission_lf
        # Slice D §8 — synthetic edd lineage filter (gated + hoisted check).
        # Build on-demand from the provider (branch-cluster runs later); the
        # anchor-pairs holder armed at the top of this fn is the conjunct.
        if _RECOURSE_ANCHOR_PAIRS is not None:
            from flextool.engine.preprocess.params._branch import (
                assert_recourse_npv_preconditions as _assert_recourse,
                dd_same_scenario_df as _dd_same_scenario_df,
            )
            _s_lineage = _dd_same_scenario_df(
                workdir, source, active_solve, provider=provider)
            _assert_recourse(workdir, source, active_solve, provider=provider)
        else:
            _s_lineage = None
        # A failure here propagates: missing cohorts would drop every investment from the capacity.
        frames = build_cohort_frames(source, active_solve, ed_inv, list(period_in_use), workdir,
                                     provider=provider, lineage=_s_lineage, commission_lf=commission_lf,
                                     seed=_seeded_invest(flex_data), with_forward=False)
        flex_data.cohort, flex_data.cohort_alive, flex_data.dd_forward = (
            frames.cohort, frames.cohort_alive, frames.dd_forward)
        _apply_lead_time_sets(flex_data, source, active_solve, ed_inv, commission_lf, period_in_use, workdir,
                              lineage=_s_lineage)
