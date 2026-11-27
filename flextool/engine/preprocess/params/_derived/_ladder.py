"""Γ.3.G — the commodity price ladder (§3.17: ``p_f_d_k``, ``p_ladder_cum_realized_mwh``) and
:func:`apply_derived_g`."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import cast_frame_axes, get_global_axis_enums, rename_to_axis
from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.preprocess.params._branch import branch_periods_inherit_carry

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import (
    _provider_has_key,
    _provider_read,
    _read_active_solve,
    _try_param,
)
from flextool.engine.preprocess.params._derived._delay import apply_delay_params
from flextool.engine.preprocess.params._derived._reserves import prundt_from_source
from flextool.engine.preprocess.params._derived._time import _read_period_branch_pairs


# ===========================================================================
# Γ.3.G — Residual Derived Params (audit §3.13/3.15/3.17/3.18 + §3.6.1 cascade)
#
# Final batch.  Closes the Derived class:
#
#   * §3.17 commodity ladder (`p_f_d_k`, `p_ladder_cum_realized_mwh`)
#   * §3.13 reserves remainder (`prundt`)
#   * §3.15 delay (`dtt__delay_duration`, `p_process_delay_weight`)
#   * §3.18 multi-branch normalisation (full cascade for both
#     ``pdt_branch_weight`` and ``pd_branch_weight``)
#   * §3.6.1 stochastic 3d_map profile cascade (branch-keyed alternative
#     resolution for ``p_profile_value``)
#
# No defensive gating — helpers either produce the canonical frame or
# return None when the structural source data is genuinely absent (e.g.
# a fixture without commodity-ladder rows).  Architectural shift per the
# task spec: the CSV path is the debug oracle, not a runtime fallback;
# parity test failures are the right signal.
# ===========================================================================


# ---------------------------------------------------------------------------
# §3.17 — Commodity ladder
# ---------------------------------------------------------------------------


def p_f_d_k_from_source(
    source: "InputSource",
    active_solve: str | None,
    workdir: Path | None,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> "Param | None":
    """Per-period "fraction realised this solve" for the price-ladder
    feature (audit §3.17.1).

    Algorithm — mirror of
    ``preprocessing/period_calculated_params.py:324-344``::

        f_d_k[d] = (p_ladder_cum_sim_hours[α(d)] + sum_t step_duration[d, t])
                 / (complete_period_share_of_year[d] * 8760)

    α(d) is the period label of a forecast branch period d (d itself otherwise, or when d has a
    carry row of its own): the realization solve carries the realized hours under the label, and every
    forecast continues from its period's realized history (``branch_periods_inherit_carry``).

    Domain: ``period_in_use``, plus one row per period that the chain has already finished and
    dropped from ``period_in_use`` (value 1.0 — a period only leaves ``period_in_use`` once it is
    fully realised, so its f_d_k is exactly 1.0 from then on).  Carrying those rows forward instead
    of letting the period vanish from the domain is what lets ``_eq_tier_cap_cumulative``'s
    ``Σ_d f_d_k`` accumulate monotonically across period boundaries, rather than resetting to the
    new period's own local progress each time the chain advances (#143).  Inputs:

    * ``solve_data/steps_in_use.csv`` — per-(d, t) step duration sum.
    * ``solve_data/ladder_cum_sim_hours.csv`` — handoff carrier; default
      0.0 when row absent.
    * ``solve_data/complete_period_share_of_year_calc.csv`` — per-period
      hour share of the year (already-computed §3.1.4 frame).

    Returns ``None`` unless the price-ladder feature is active (i.e. at
    least one commodity has ``price_method ∈ {price_ladder_annual,
    price_ladder_cumulative}`` — flextool emits ``f_d_k.csv`` always
    but ``engine/model/_commodity_ladder.py:load`` only consumes it when
    ladder commodities exist; the parity test mirrors that semantic).
    """
    if workdir is None:
        return None
    # Gate on ladder activity — mirrors the CSV-side behaviour
    # where the loader only emits p_f_d_k when there are ladder
    # commodities.
    sd = Path(workdir) / "solve_data"
    cwl_path = sd / "commodity_with_ladder.csv"
    if _provider_has_key(provider, cwl_path):  # Phase E-j — seed-aware (no-op: writer not in expected_basenames)
        cwl = _provider_read(provider, cwl_path)
        if cwl.height == 0:
            return None
    else:
        # Fall back to source data: any commodity with a non-default
        # price_method counts as ladder-active.
        method = _try_param(source, "commodity", "price_method")
        if method is None or method.height == 0:
            return None
        ladder = method.filter(pl.col("value") != "price")
        if ladder.height == 0:
            return None
    # Path B Cat B (WriterSnapshot top-7): prefer SolveContext's typed
    # ``steps_in_use`` and ``period_share_of_year`` accessors when
    # available — already in canonical ``d`` / ``t`` / ``step_duration``
    # form, so we skip the per-call rename + select allocation.  Falls
    # back to the CSV-read path for legacy callers without ``ctx``.
    if ctx is not None and ctx.steps_in_use.height > 0:
        siu_df = ctx.steps_in_use
        if "d" not in siu_df.columns or "step_duration" not in siu_df.columns:
            return None
        sum_step = (siu_df.lazy()
                      .group_by("d")
                      .agg(pl.col("step_duration").sum().alias("sum_step")))
    else:
        siu = sd / "steps_in_use.csv"
        if not _provider_has_key(provider, siu):  # Phase E-j — seed-aware
            return None
        df = _provider_read(provider, siu)
        if df.height == 0 or "period" not in df.columns:
            return None
        sum_step = (df.lazy()
                      .with_columns(pl.col("step_duration")
                                      .cast(pl.Float64, strict=False))
                      .group_by("period")
                      .agg(pl.col("step_duration").sum().alias("sum_step"))
                      .pipe(rename_to_axis, {"period": "d"}))
    # complete_period_share_of_year[d]
    if ctx is not None and ctx.period_share_of_year.height > 0:
        csy = (ctx.period_share_of_year.lazy()
                  .with_columns(pl.col("value").alias("share")))
    else:
        csy_path = sd / "complete_period_share_of_year_calc.csv"
        if not _provider_has_key(provider, csy_path):  # Phase E-j — seed-aware
            # Try the non-_calc variant (some fixtures only emit the latter).
            csy_path = sd / "complete_period_share_of_year.csv"
        if not _provider_has_key(provider, csy_path):  # Phase E-j — seed-aware (no-op for non-_calc variant)
            return None
        csy = (_provider_read(provider, csy_path).lazy()
                  .pipe(rename_to_axis, {"period": "d"})
                  .with_columns(pl.col("value").cast(pl.Float64, strict=False)
                                      .alias("share")))
    # ladder_cum_sim_hours — default 0.0 when absent.
    # Phase 4.1a / 4.2-1g — the cross-roll cumulative-sim-hours accumulator
    # rides on ``handoff/cum_sim_hours`` (canonical schema
    # ``[period, p_ladder_cum_sim_hours]``).  The legacy
    # ``solve_data/ladder_cum_sim_hours`` CSV fallback was dropped in
    # 4.2-1g — the translator pipeline always populates the handoff key.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    cum_raw = read_handoff_frame(provider, K.HANDOFF_CUM_SIM_HOURS)
    if cum_raw is not None and cum_raw.height > 0 and \
            "p_ladder_cum_sim_hours" in cum_raw.columns:
        cum_lf = (cum_raw.lazy()
                   .pipe(rename_to_axis, {"period": "d",
                            "p_ladder_cum_sim_hours": "cum"})
                   .with_columns(pl.col("cum")
                                   .cast(pl.Float64, strict=False)))
        # A forecast branch period continues from its period's realized hours, written under the period
        # label (emit_period_calculated_params, the emitted twin): the numerator joins the inherited
        # carry; the fold of finished periods below keeps the raw keys.
        cum_join_lf = branch_periods_inherit_carry(
            cum_lf.collect(), _read_period_branch_pairs(workdir, provider=provider)).lazy()
    else:
        cum_lf = None
        cum_join_lf = None
    # period_in_use as the output domain — Path B Cat B: prefer
    # SolveContext's typed accessor (already canonical ``[d]``).
    if ctx is not None and ctx.period_in_use.height > 0:
        piu_lf = ctx.period_in_use.lazy().select("d")
    else:
        piu_path = sd / "period_in_use_set.csv"
        if not _provider_has_key(provider, piu_path):  # Phase E-j — seed-aware
            return None
        piu_lf = (_provider_read(provider, piu_path).lazy()
                    .pipe(rename_to_axis, {"period": "d"})
                    .select("d"))
    # Phase 4.8f: ctx-side (period_in_use, period_share_of_year,
    # steps_in_use) and CSV-side ``d`` columns may arrive in different
    # dtype regimes (Enum from SolveContext, Utf8 from raw CSV).  Defend
    # the cascade join keys by casting every contributing LazyFrame to
    # the canonical axis Enum vocabulary when active.
    _enums = get_global_axis_enums()
    if _enums is not None:
        piu_lf = cast_frame_axes(piu_lf, _enums)
        sum_step = cast_frame_axes(sum_step, _enums)
        csy = cast_frame_axes(csy, _enums)
        if cum_lf is not None:
            cum_lf = cast_frame_axes(cum_lf, _enums)
            cum_join_lf = cast_frame_axes(cum_join_lf, _enums)
    out_lf = (piu_lf
                .join(sum_step.select("d", "sum_step"), on="d", how="left")
                .join(csy.select("d", "share"), on="d", how="left"))
    if cum_lf is not None:
        out_lf = out_lf.join(cum_join_lf.select("d", "cum"), on="d", how="left")
        out_lf = out_lf.with_columns(pl.col("cum").fill_null(0.0))
    else:
        out_lf = out_lf.with_columns(cum=pl.lit(0.0))
    # A split realization roll (a period label with steps): a branch period's path also holds its label's jump
    # (emit_period_calculated_params, the emitted twin).  Empty without a label with steps.
    pairs = [(a, b) for a, b in _read_period_branch_pairs(workdir, provider=provider) if a != b]
    jump = (pl.LazyFrame(pairs, schema={"_a": pl.Utf8, "_d": pl.Utf8}, orient="row")
            .join(sum_step.select(pl.col("d").cast(pl.Utf8).alias("_a"), pl.col("sum_step").alias("jump")),
                  on="_a", how="inner")
            .group_by("_d").agg(pl.col("jump").sum()).collect())
    if jump.height > 0:
        out_lf = (out_lf.with_columns(_d=pl.col("d").cast(pl.Utf8)).join(jump.lazy(), on="_d", how="left")
                  .with_columns(sum_step=pl.col("sum_step").fill_null(0.0) + pl.col("jump").fill_null(0.0))
                  .drop("_d", "jump"))
    out_lf = (out_lf
                .with_columns(pl.col("sum_step").fill_null(0.0))
                .with_columns(value=(pl.col("cum") + pl.col("sum_step"))
                                       / (pl.col("share") * HOURS_PER_YEAR))
                .select("d", "value"))
    if cum_lf is not None:
        # A period leaves `period_in_use` only once the rolling chain has moved past it, which
        # (periods are visited in order, never revisited) only happens once it is fully realised —
        # at that point its frozen `cum_sim_hours` already equals its own complete hours, i.e. its
        # true f_d_k is exactly 1.0.  Folding those periods back in (instead of letting them drop
        # out of the domain) is what keeps Σ_d f_d_k monotone across period boundaries for
        # ladder_tier_cap_cumulative_roll, whose Σ_d f_d_k must accumulate over every period the
        # chain has ever touched, not reset to the current period's own local progress each time
        # the chain advances to a new one (#143's root cause).
        completed_lf = (cum_lf.select("d").unique()
                         .join(piu_lf.select("d"), on="d", how="anti")
                         .with_columns(value=pl.lit(1.0, dtype=pl.Float64)))
        out_lf = pl.concat([out_lf, completed_lf], how="vertical_relaxed")
    out = out_lf.collect()
    if out.height == 0:
        return None
    return Param(("d",), out)


def p_ladder_cum_realized_mwh_from_workdir(
    workdir: Path | None,
    *,
    provider: "object | None" = None,
) -> "Param | None":
    """Per-(c, i, d) cumulative realized MWh handoff carrier
    (audit §3.17.2).

    This is a pure rolling-handoff state — written by the chain runner
    after each solve summing ``v_trade`` per tier into the next solve's
    starting accumulator.  Single-solve fixtures emit a header-only CSV
    (returns None).
    """
    if workdir is None:
        return None
    # Phase 4.1a / 4.2-1g — prior-roll accumulator now rides on
    # ``handoff/cumulative_commodity`` (canonical schema
    # ``[commodity, tier, period, p_ladder_cum_realized_mwh]``).  The
    # legacy ``solve_data/ladder_cum_realized_mwh`` CSV fallback was
    # dropped in 4.2-1g — the translator pipeline is the only source.
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    raw = read_handoff_frame(provider, K.HANDOFF_CUMULATIVE_COMMODITY)
    if raw is None:
        return None
    if raw.height == 0:
        return None
    value_col = ("p_ladder_cum_realized_mwh"
                 if "p_ladder_cum_realized_mwh" in raw.columns else "value")
    if value_col not in raw.columns:
        return None
    out = (raw.pipe(rename_to_axis, {"commodity": "c", "tier": "i", "period": "d",
                         value_col: "value"})
              .with_columns(pl.col("i").cast(pl.Utf8))
              .with_columns(pl.col("value").cast(pl.Float64, strict=False))
              .select("c", "i", "d", "value"))
    if out.height == 0:
        return None
    return Param(("c", "i", "d"), out)


# ---------------------------------------------------------------------------
# Public — Γ.3.G field list (for selective scope-checking).
# ---------------------------------------------------------------------------


G_PUBLIC_FIELDS: tuple[str, ...] = (
    "p_f_d_k",
    "p_ladder_cum_realized_mwh",
    "prundt",
    "dtt__delay_duration",
    "p_process_delay_weight",
    "process_delayed__duration",
    "pd_branch_weight",
    "pdt_branch_weight",
)


def apply_derived_g(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.G residual Derived Params, mutating ``flex_data``
    in place.

    Scope:
      * §3.17 commodity ladder — ``p_f_d_k``,
        ``p_ladder_cum_realized_mwh``.
      * §3.13 reserves — ``prundt``.
      * §3.15 delay — ``dtt__delay_duration``, ``p_process_delay_weight``,
        ``process_delayed__duration``.
      * §3.18 multi-branch normalisation — full cascade for
        ``pd_branch_weight`` and ``pdt_branch_weight``.

    Δ.3 replaced the previous ``derived_overrides_g`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)
    dt = getattr(flex_data, "dt", None)

    # Δ.12b — assignment is unconditional; helpers are authoritative
    # producers.  ``None`` is the explicit "feature inactive" signal
    # (no commodity-ladder commodities, no reserve relationships,
    # no delayed processes, etc.).  Hard exceptions propagate.

    # ─── §3.17.1 p_f_d_k ───────────────────────────────────────────
    flex_data.p_f_d_k = p_f_d_k_from_source(source, active_solve, workdir,
                                                  ctx=ctx, provider=provider)

    # ─── §3.17.2 p_ladder_cum_realized_mwh ─────────────────────────
    flex_data.p_ladder_cum_realized_mwh = (
        p_ladder_cum_realized_mwh_from_workdir(workdir, provider=provider))

    # ─── §3.13.1 prundt ────────────────────────────────────────────
    flex_data.prundt = prundt_from_source(source, active_solve, dt)

    # ─── §3.15 dtt__delay_duration, p_process_delay_weight and
    #     process_delayed__duration (one source read, same td labels) ──
    apply_delay_params(flex_data, source)

    # ─── §3.18 cluster D — multi-branch normalisation + non-anticipativity ──
    # Δ.8 consolidation: delegate the full cluster D port to
    # ``_branch.apply_branch_cluster``.  The helper covers
    # ``pd_branch_weight``, ``pdt_branch_weight``, and (already
    # CSV-loaded by the flex_data loaders) ``period_branch_full``,
    # ``period_in_use_set``, ``dt_non_anticipativity`` — the latter
    # three are kept as overrides so the lazy port becomes the single
    # producer once the CSV cascade retires (Δ.12).  R-O6: the helper
    # never touches ``invest_periods`` / ``v_invest``.
    # Δ.12b: defensive try/except removed; apply_branch_cluster is
    # parity-bound (cluster D tests gate it).
    from flextool.engine.preprocess.params._branch import apply_branch_cluster
    apply_branch_cluster(flex_data, source, workdir, active_solve,
                          ctx=ctx, provider=provider)
