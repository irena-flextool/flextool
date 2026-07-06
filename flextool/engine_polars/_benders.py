"""Benders (Option C) regional decomposition — master + multi-cut loop.

This module implements the Phase-2 Benders scheme described in
``specs/benders_option_c.md`` (design, Phase-2 revised, and the
PROCEED-WITH-CHANGES critique).  It REPLACES the dual-subgradient scheme's
false-convergence behaviour on GREENFIELD cross-region trade connections: the
Lagrangian splitter severs each cross-region arc into invest-less half-flows
bounded to ~0, collapsing to an autarkic solution with an INVALID bound ABOVE
the true optimum.  Benders puts the trade investment + the trade flow / capacity
coupling in a MASTER and feeds each region the master's chosen flow as a pinned
boundary injection, returning a VALID lower bound and the true optimum.

Architecture (per the spec's locked decisions):

* **Hand-built persistent master** — a single ``polar_high.Problem`` wrapped in a
  ``WarmProblem``, built ONCE and grown by appended optimality cut rows
  (``WarmProblem.add_cut_row``).  Structure MIRRORS the monolith's trade layer:

    - trade flow vars ``f[arc, d, t]`` for every cross-region directed arc, in the
      SAME unitsize-normalised units as the region half-flow ``v_flow`` (so cut
      slopes — reduced costs of the pinned columns — drop in with no rescale);
    - invest vars ``C[conn]`` per cross-region connection (single invest period in
      the prototype);
    - capacity coupling ``f[arc,d,t] ≤ C[conn]`` (NORMALISED — unitsize cancels,
      Phase-1 §A.5);
    - one recourse var ``η_r`` per region, lower-bounded by a large-NEGATIVE
      finite floor sized from the bootstrap region costs (NOT a hard ``η≥0`` —
      FlexTool region costs can be negative, so a blind 0 floor could cut off the
      optimum; the finite floor is a provably valid global under-estimate that
      keeps the cut-less iter-0 master kOptimal, and the f̄=0 bootstrap seeds the
      first real cuts before the first LB-bearing master solve);
    - objective ``Σ_conn C[conn]·unitsize·annu  +  Σ_r η_r`` where ``annu`` is read
      from the SAME source ``build_flextool`` uses
      (``ed_entity_annual_discounted`` + ``ed_lifetime_fixed_cost``) — NOT a
      hand-derived annuity (a mismatch silently yields a wrong-but-plausible
      optimum).

  Master is built AUTOSCALE-OFF (the test path never applies Layer 2), so the
  appended cut rows live on the built-column scale.

* **Region subproblems** are normal FlexTool models via the splitter, with the
  cross-region half-flows UNCAPPED (``benders_uncap_cross_region=True``) so a
  positive master pin is feasible.  Each iteration pins every region's forward
  cross-region half-flows to the current f̄ per-``(d,t)`` (reverse pinned to 0) and
  solves; the cut slope per ``(arc,d,t)`` is the reduced cost of the pinned
  forward column ``Solution.col_dual[pin_col_id]`` (Phase-1-verified =
  ``∂cost_r/∂f̄``, basis-correct, no monolith reference).

Loop (multi-cut Benders): bootstrap f̄=0 → first cuts → master → new f̄ → regions
→ cuts → master → … until ``gap = (best_UB − LB)/|best_UB| ≤ tol``.  ``LB`` =
master objective (a valid lower bound — the whole point vs the Lagrangian bug);
``UB`` = master native cost + Σ cost_r(f̄), incumbent = best (min) UB.

The loop MECHANICS (bootstrap pass, η-floor sizing, cut bookkeeping, LB/UB +
sandwich/monotonicity self-checks, in-out stabilization, stall guard, cut
compaction placement, parallel fan-out, convergence) live in the generic
coordinator :func:`polar_high.benders.solve_benders_loop`.  This module keeps
everything domain-specific: the split/build path (incl. the region autoscale
stack), arc discovery, the subproblem adapter whose ``solve_at`` owns pin+solve
(Layer-2 pin transform, ``fix_cols``, ``retry_on_unknown``, dual unscale, slope
aggregation — order is load-bearing, plan risk R13), the FlexTool-built master
adapter (:class:`_BendersMaster`), the capacity clamp (``project_point``), the
Tier-1 invest handoff, env-knob resolution, and the rendering of the
coordinator's structured exceptions into plain-English diagnostics.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, NoReturn

import highspy
import numpy as np
import polars as pl

from polar_high import (
    Param,
    Problem,
    Sum,
    WarmProblem,
    resolve_worker_count,
)
from polar_high.benders import (
    BendersBoundInvalid,
    BendersLoopOptions,
    BendersStalled,
    SubproblemHandle,
    SubproblemNotOptimal,
    SubproblemResult,
    evaluate_at_point,
    solve_benders_loop,
)
from polar_high.benders import _check_cuts_satisfied as _ph_check_cuts_satisfied
from polar_high.benders import _cut_separates as _ph_cut_separates

from flextool.engine_polars import _region_filter
from flextool.engine_polars import build_flextool as _build_flextool
from flextool.engine_polars._axis_enums import (
    get_global_axis_enums,
    reset_global_axis_enums,
    set_global_axis_enums,
)
from flextool.engine_polars.autoscale import (
    Layer2Plan,
    ScalingMode,
    apply_layer2,
    apply_scaling,
    detect_ranges,
    mode_enables_layer1,
    mode_enables_layer3,
    recommend_scaling,
    resolve_scaling_config,
    unscale_solution,
)
from flextool.engine_polars._region_filter import HalfFlow, RegionSplit
from flextool.engine_polars.input import FlexData

_logger = logging.getLogger(__name__)


def _benders_quiet() -> bool:
    """Whether the per-solve HiGHS native log should be silenced.

    Benders solves a master plus N region subproblems (cold build + a warm
    re-solve per iteration, regions in parallel); leaving HiGHS verbose
    floods the console with many interleaved native logs.  By DEFAULT we
    mute every HiGHS solve and let the orchestrator's per-iteration
    LB/UB/gap log be the visible progress.  Set ``FLEXTOOL_BENDERS_VERBOSE``
    to restore the full native HiGHS output (the pre-silencing behaviour).
    """
    return not os.environ.get("FLEXTOOL_BENDERS_VERBOSE")


def _silence_if_quiet(wp: WarmProblem) -> None:
    """Mute the HiGHS native log on ``wp`` when Benders runs quietly.

    Output-flag persists on the WarmProblem's HiGHS handle across all of
    its subsequent cold / warm / retry solves, so a single call right after
    construction covers the whole lifetime.  Silencing changes no solve
    numerics — results are byte-identical with and without it.
    """
    if _benders_quiet():
        wp.set_output_flag(False)

# The four investment/divestment decision variables assembled into a
# whole-system handoff for the TIER-1 invest->dispatch chain.  Each is
# declared in ``model.py`` over a 2-tuple of dims whose FIRST element is
# the entity axis ("p" for process / connection vars, "n" for node vars)
# and whose second is the period axis "d".  ``Var.dims`` is read at
# runtime to recover the exact entity column name — we do not hard-code it.
_INVEST_VAR_NAMES = ("v_invest_p", "v_invest_n", "v_divest_p", "v_divest_n")

# Absolute tolerance below which a non-owner region's invest value for an
# entity is treated as a numerically-collapsed zero (the expected case for
# an out-of-region invest var).  A non-owner value above this triggers a
# (non-fatal) canary warning.
_NONOWNER_NONZERO_ABS_TOL = 1e-6

# Relative band that separates benign numerical noise from a GENUINE invalid-
# bound bug on the lower-bound self-checks (LB monotonicity, LB ≤ best-UB
# sandwich).  A drop/overshoot smaller than this (and smaller than the
# convergence tolerance — a difference below ``tol`` cannot change the answer
# beyond tolerance) is recovered fail-safe: the loop keeps its best feasible
# incumbent instead of aborting the whole run.  A larger anomaly hard-fails,
# since that is the Lagrangian-style invalid-lower-bound pathology this scheme
# exists to catch.
_LB_GROSS_SLACK = 1e-3


def _benders_failure_message(summary: str, meaning: str, how_to_avoid: str) -> str:
    """Format a Benders hard-failure into a plain-English diagnostic: a one-line
    technical summary, what it means for the user, and how to avoid it."""
    return (
        f"{summary}\n\n"
        f"What this means:\n  {meaning}\n\n"
        f"How to avoid it:\n  {how_to_avoid}"
    )

# A large-NEGATIVE finite floor on each recourse var ``η_r``.  A truly free
# (lower=-inf) η leaves the cut-less master UNBOUNDED; an unbounded HiGHS solve
# then corrupts the warm basis so the first appended cut row triggers a
# kSolveError.  The floor must be a PROVABLY valid global under-estimate so it
# never cuts off the optimum (the spec's point-2 warning: FlexTool region costs
# can be negative via commodity-sell / storage-revenue, so a blind 0 is unsafe).
#
# The floor is derived per-run from the bootstrap (autarkic, f̄=0) region costs:
# ``eta_floor = -_ETA_FLOOR_MULT · max_r |cost_r^autarky|`` (sized by the
# coordinator's bootstrap pass — ``BendersLoopOptions.eta_floor_mult`` carries
# this constant), computed in the SAME (scaled) space the master objective
# lives in.  Validity: at the optimum ``η_r = cost_r(f̄*)``, the
# region's recourse cost at the optimal trade schedule.  ``cost_r^autarky``
# (zero import) is the no-trade reference; trade only relaxes a region's balance
# (a free injection it may ignore within finite slack), so the minimum
# achievable ``cost_r`` is bounded below by ``-(region sell-revenue capacity)``,
# a finite quantity no larger in magnitude than a small multiple of the autarkic
# cost for these models.  ``-1.1·max|cost^autarky|`` therefore sits BELOW any
# achievable ``cost_r`` (with margin) and can never cut off the optimum, while
# being ~90× tighter than the old ``-100·max|cost|``: that narrows the
# floor/cut-coef dynamic range (the over-wide range, e.g. 1e11 vs O(1e9) cut
# coefs, is what drove the Phase-3c "Optimal→Unknown" warning on the warm
# post-append re-solve).  The runtime ``LB ≤ best_UB`` sandwich guard is the
# safety net confirming the floor never produced an invalid bound.
_ETA_FLOOR_MULT = 1.1

# Env override for the per-iteration region-solve worker count (machine-local;
# NO schema/DB knob — avoids another migration).  ``FLEXTOOL_BENDERS_WORKERS``
# pins the thread-pool size for the parallel region recourse pass; unset/<=0
# leaves the ``workers`` argument (default: auto = min(n_regions, cpu-1)) in
# charge.  ``workers=1`` (or 1 region) keeps the fully-sequential path.
_BENDERS_WORKERS_ENV = "FLEXTOOL_BENDERS_WORKERS"


def _resolve_benders_workers(n_regions: int, workers) -> int:
    """Resolve the effective region-solve worker count.

    Precedence: explicit ``FLEXTOOL_BENDERS_WORKERS`` env (machine-local) wins;
    otherwise the ``workers`` argument (``None`` ⇒ auto ``min(n, cpu-1)``).
    Clamped to ``[1, n_regions]`` by :func:`resolve_worker_count`.
    """
    env = os.environ.get(_BENDERS_WORKERS_ENV)
    if env:
        try:
            env_n = int(env)
        except ValueError:
            _logger.warning(
                "Benders: ignoring non-integer %s=%r", _BENDERS_WORKERS_ENV, env
            )
        else:
            if env_n > 0:
                workers = env_n
    return resolve_worker_count(n_regions, workers)


# Default trailing window ``K`` for the stall guard: the best feasible cost must
# be frozen (no relative improvement) for this many consecutive iterations
# BEFORE a stall can be declared (in conjunction with a high gap + a blown-up
# incumbent; see :class:`polar_high.StallMonitor`).  Empirical, validated on
# spatial-decomposition traces up to N=10 regions (benign frozen windows there
# are <= 4 iters, so K=8 has a 2x margin).
_STALL_WINDOW_DEFAULT = 8

# Env override for the stall-guard window ``K`` (machine-local; NO schema/DB
# knob — mirrors the ``FLEXTOOL_BENDERS_WORKERS`` design note, avoids another
# migration).  Unset / <= 0 leaves the ``_STALL_WINDOW_DEFAULT``.  This is NOT a
# whole-guard opt-out (a correctness guard must not be disableable): a user who
# genuinely wants more iterations before the guard fires sets this HIGH.
_BENDERS_MAX_STALL_ENV = "FLEXTOOL_BENDERS_MAX_STALL"


def _resolve_benders_max_stall() -> int:
    """Resolve the stall-guard trailing window ``K``.

    Precedence: an explicit positive ``FLEXTOOL_BENDERS_MAX_STALL`` env wins;
    otherwise the ``_STALL_WINDOW_DEFAULT``.  Mirrors
    :func:`_resolve_benders_workers` EXACTLY, including the non-integer-warning
    branch (a malformed value is ignored, not fatal).
    """
    k = _STALL_WINDOW_DEFAULT
    env = os.environ.get(_BENDERS_MAX_STALL_ENV)
    if env:
        try:
            env_n = int(env)
        except ValueError:
            _logger.warning(
                "Benders: ignoring non-integer %s=%r", _BENDERS_MAX_STALL_ENV, env
            )
        else:
            if env_n > 0:
                k = env_n
    return k


# Env knob for the periodic MASTER CUT COMPACTION threshold (machine-local; NO
# schema/DB knob for the first ship — mirrors ``FLEXTOOL_BENDERS_WORKERS`` /
# ``FLEXTOOL_BENDERS_MAX_STALL``, avoids a migration).  When the master's
# accumulated cut-row count reaches this positive threshold, the master is
# COMPACTED via ``WarmProblem.compact_cuts`` — polar-high classifies each
# retained cut row by PRIMAL slack at the current master optimum, deletes the
# strictly-slack rows in place (LB-preserving, with a verify-restore belt), and
# keeps only the binding ones.  ``0`` / unset = OFF = today's unbounded cut
# growth = byte-identical to the pre-compaction path.  A malformed or negative
# value is IGNORED with a warning (OFF is used), mirroring the sibling resolvers.
_BENDERS_CUT_COMPACT_AT_ENV = "FLEXTOOL_BENDERS_CUT_COMPACT_AT"


def _resolve_benders_cut_compact_at() -> int:
    """Resolve the master cut-compaction threshold.

    Reads ``FLEXTOOL_BENDERS_CUT_COMPACT_AT``: unset or ``0`` (or negative)
    ⇒ ``0`` = OFF (unbounded cut growth, byte-identical to the pre-compaction
    path); a positive integer ⇒ the active cut-row count at which the master is
    compacted (via ``WarmProblem.compact_cuts``) keeping only the
    currently-binding cuts.  A non-integer OR negative value is IGNORED (not
    fatal, warned), mirroring :func:`_resolve_benders_max_stall` EXACTLY,
    including the malformed-value warning branch.
    """
    compact_at = 0
    env = os.environ.get(_BENDERS_CUT_COMPACT_AT_ENV)
    if env:
        try:
            env_n = int(env)
        except ValueError:
            _logger.warning(
                "Benders: ignoring non-integer %s=%r",
                _BENDERS_CUT_COMPACT_AT_ENV, env,
            )
        else:
            if env_n >= 0:
                compact_at = env_n
            else:
                _logger.warning(
                    "Benders: ignoring negative %s=%r (must be >= 0; "
                    "0 = OFF)", _BENDERS_CUT_COMPACT_AT_ENV, env,
                )
    return compact_at


# Default trial-point WINDOW ``W`` for the dominance cut-compaction policy.  The
# ``compact_cuts(policy="dominance")`` selection keeps, per recourse group, the
# oldest max-achiever at EACH of the last ``W`` master trial points — so ``W``
# trades master size (≈ ``W``·regions retained) for convergence robustness (a
# too-small window starves the recourse approximation and stalls).  Empirical
# starting point; tune per model.
_BENDERS_CUT_WINDOW_DEFAULT = 10

# Env override for the dominance-policy trial-point window ``W`` (machine-local;
# NO schema/DB knob — mirrors ``FLEXTOOL_BENDERS_CUT_COMPACT_AT`` /
# ``FLEXTOOL_BENDERS_MAX_STALL``, avoids a migration).  Unset / <= 0 leaves the
# ``_BENDERS_CUT_WINDOW_DEFAULT``; a malformed value is IGNORED with a warning.
_BENDERS_CUT_WINDOW_ENV = "FLEXTOOL_BENDERS_CUT_WINDOW"


def _resolve_benders_cut_window() -> int:
    """Resolve the dominance-policy trial-point window ``W``.

    Reads ``FLEXTOOL_BENDERS_CUT_WINDOW``: an explicit positive value wins;
    otherwise the ``_BENDERS_CUT_WINDOW_DEFAULT``.  Mirrors
    :func:`_resolve_benders_max_stall` EXACTLY, including the
    non-integer-warning branch (a malformed value is ignored, not fatal).
    """
    w = _BENDERS_CUT_WINDOW_DEFAULT
    env = os.environ.get(_BENDERS_CUT_WINDOW_ENV)
    if env:
        try:
            env_n = int(env)
        except ValueError:
            _logger.warning(
                "Benders: ignoring non-integer %s=%r", _BENDERS_CUT_WINDOW_ENV, env
            )
        else:
            if env_n > 0:
                w = env_n
    return w


# Cut-compaction SELECTION POLICY.  ``slack`` (the DEFAULT) drops cuts strictly
# slack at the current optimum — cheap and LB-safe, effective when the master
# carries genuinely-redundant cuts.  ``dominance`` is a NON-DEFAULT alternative
# (env opt-in) that groups cuts by recourse column and keeps only the oldest
# group-max achiever over the trailing window, dropping dominated cuts AND
# degenerate ties; it costs a trial-point sweep + can trigger the verify-restore
# re-solve, and it is INEFFECTIVE where the cuts are load-bearing (a degenerate
# optimum with cheap inter-temporal storage, e.g. the H2-trade N=10 case), so it
# is not the default.  An unrecognised value is IGNORED with a warning.
_BENDERS_CUT_POLICY_DEFAULT = "slack"
_BENDERS_CUT_POLICY_ENV = "FLEXTOOL_BENDERS_CUT_POLICY"


def _resolve_benders_cut_policy() -> str:
    """Resolve the cut-compaction selection policy (``slack`` | ``dominance``).

    Reads ``FLEXTOOL_BENDERS_CUT_POLICY``; a recognised value wins, otherwise the
    ``_BENDERS_CUT_POLICY_DEFAULT`` (``slack``).  An unrecognised value is ignored
    with a warning (not fatal), mirroring the other benders env knobs."""
    policy = _BENDERS_CUT_POLICY_DEFAULT
    env = os.environ.get(_BENDERS_CUT_POLICY_ENV)
    if env:
        if env in ("slack", "dominance"):
            policy = env
        else:
            _logger.warning(
                "Benders: ignoring unrecognised %s=%r (expected 'slack' or "
                "'dominance'); using %r",
                _BENDERS_CUT_POLICY_ENV, env, _BENDERS_CUT_POLICY_DEFAULT,
            )
    return policy


# Env override for the Benders in-out separation weight ``λ`` (machine-local; NO
# schema/DB knob in Phase 1 — mirrors ``FLEXTOOL_BENDERS_WORKERS`` /
# ``FLEXTOOL_BENDERS_MAX_STALL``, avoids a migration).  ``λ`` is the weight on
# the stable interior CENTRE in ``f_sep = λ·centre + (1-λ)·f_out`` fed to
# :class:`polar_high.InOutStabilizer` (one instance PER REGION).  ``0.0``
# (default) ⇒ OFF: the in-out block is skipped entirely and the loop is
# byte-identical to exact Benders.  ``λ ∈ (0, 1)`` ⇒ ON, larger = more
# stabilisation.  ``λ ≥ 1`` ("never query the master") is non-convergent and
# ``λ < 0`` is meaningless — both are IGNORED with a warning (the default 0.0 is
# used), mirroring the non-float-warning branch below and in
# :func:`_resolve_benders_max_stall`.
_BENDERS_IN_OUT_WEIGHT_ENV = "FLEXTOOL_BENDERS_IN_OUT_WEIGHT"

# Experimental (specs/benders_research_master_stabilization.md §6): which dual
# the region subproblems report for cut slopes.  ``basic`` (default) keeps the
# ``run_crossover=on`` pin — reproducible vertex duals.  ``interior`` solves
# regions with barrier and NO crossover: on a degenerate (storage-flat) optimal
# face the barrier limit is a central point of the optimal-dual face, i.e. an
# "averaged" cut slope instead of an arbitrary vertex choice.  Trade-offs: no
# warm start (every region re-solve is a cold IPM), possible imprecise
# termination (surfaces as the existing not-optimal subproblem error).  Any
# other value is IGNORED with a warning and ``basic`` is used.
_BENDERS_REGION_DUALS_ENV = "FLEXTOOL_BENDERS_REGION_DUALS"


def _resolve_benders_region_duals() -> str:
    """Resolve the region dual mode: ``basic`` (default) or ``interior``."""
    mode = os.environ.get(_BENDERS_REGION_DUALS_ENV, "basic").strip().lower()
    if mode not in ("basic", "interior"):
        _logger.warning(
            "Benders: ignoring unknown %s=%r (must be 'basic' or 'interior')",
            _BENDERS_REGION_DUALS_ENV, mode,
        )
        mode = "basic"
    return mode


def _apply_region_autoscale(pb: Problem, cfg, region_name: str) -> "Layer2Plan | None":
    """Apply the monolith's full autoscale (Layer-2 + Layer-3) to a Benders
    region subproblem ``pb``, in place, and return the Layer-2 inverse
    :class:`Layer2Plan` (or ``None`` when Layer-2 was skipped/failed).

    Mirrors the monolith's two pre-solve hooks
    (:func:`_orchestration._autoscale_apply_layer2_pre_solve` +
    :func:`_orchestration._autoscale_apply_layer3_pre_solve`) so the region
    subproblems condition exactly like the monolith LP — which runs this same
    ``scale_the_objective`` + Layer-2 + Layer-3 stack on every solve.

    * **Layer 2** (per-type var/constraint rescale) fires only in
      :class:`ScalingMode.FULL` and only when the pre-solve Layer-1 range
      detector trips.  Its :class:`Layer2Plan` is a *fixed* transform that must
      persist for the life of the region ``WarmProblem`` (built once, then only
      grown by master cuts) — the caller stores it on the :class:`_Region` and
      uses it to scale the boundary-flow pins and unscale the region solution.
    * **Layer 3** (HiGHS-native top-up: ``user_objective_scale`` /
      ``user_bound_scale`` / ``simplex_scale_strategy``) runs unconditionally
      whenever the mode enables it, derived from the *post-Layer-2* coefficient
      ranges.  It needs NO stored plan and is transparent to the cut math:
      ``user_*_scale`` are HiGHS-internal power-of-two options that HiGHS
      un-scales on output, so ``sol.obj`` and the boundary-flow duals come back
      in the same coordinates the Layer-2-only path produces (the monolith
      relies on this exact stacking — see ``_LEGACY_DEFAULT_OBJECTIVE_SCALE``).

    Layer 2 re-raises under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (CLAUDE.md invariant
    #1): a var/cstr/parameter family missing from the autoscale registries would
    otherwise be swallowed here and silently revert the region to an un-scaled
    LP, defeating the fix.  Layer 3, like the monolith, is best-effort (a
    failure just leaves HiGHS' own equilibration to fill in) and never fatal.
    """
    plan: "Layer2Plan | None" = None
    # --- Layer 2 (FULL mode + Layer-1 trigger). ----------------------------
    if mode_enables_layer1(cfg.mode):
        try:
            ranges_pre = detect_ranges(pb, cfg)
        except Exception:  # pragma: no cover — guard against future API drift
            if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
                raise
            _logger.exception(
                "Benders: Layer-2 range readout failed for region %r; "
                "solving it un-scaled", region_name,
            )
            ranges_pre = None
        if (
            ranges_pre is not None
            and cfg.mode is ScalingMode.FULL
            and ranges_pre.trigger
        ):
            try:
                plan = apply_layer2(pb, cfg)
            except Exception:  # pragma: no cover
                if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
                    raise
                _logger.exception(
                    "Benders: Layer-2 apply failed for region %r; solving it "
                    "un-scaled", region_name,
                )
                plan = None
            else:
                # Log the chosen per-type exponents.  The Benders splitter
                # builds the cross-region half-flows with a 1e12 uncap sentinel
                # (a bound the monolith LP never carries); this line lets us
                # confirm on the benchmark that the sentinel is not hijacking
                # the BOUND-range decision away from physical flows.
                _logger.info(
                    "Benders Layer-2 [region %s]: exponents=%s, rows=%d, "
                    "skipped_rows=%d, integer_cols=%d",
                    region_name,
                    {t.value: e for t, e in plan.type_exponents.items()},
                    plan.row_factors.shape[0],
                    len(plan.skipped_rows),
                    len(plan.skipped_integer_cols),
                )
    # --- Layer 3 (HiGHS-native top-up; unconditional when enabled). --------
    # Derived from POST-Layer-2 ranges (the residual spread after Layer 2's
    # per-type rescale) — so this re-reads the ranges off the now-mutated ``pb``.
    if mode_enables_layer3(cfg.mode):
        try:
            ranges_post = detect_ranges(pb, cfg)
            l3 = recommend_scaling(ranges_post, cfg, problem=pb)
            apply_scaling(pb, l3)
        except Exception:  # pragma: no cover — best-effort, mirrors monolith
            _logger.exception(
                "Benders: Layer-3 apply failed for region %r; HiGHS internal "
                "scaling will fill in", region_name,
            )
        else:
            _logger.info(
                "Benders Layer-3 [region %s]: user_objective_scale=%d, "
                "user_bound_scale=%d, simplex_scale_strategy=%d",
                region_name, l3.user_objective_scale, l3.user_bound_scale,
                l3.simplex_scale_strategy,
            )
    return plan


def _resolve_benders_in_out_weight(db_value: float = 0.0) -> float:
    """Resolve the Benders in-out separation weight ``λ``.

    Precedence (mirrors :func:`_resolve_benders_workers`' env-first
    resolution): an explicit, valid ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT`` in
    ``[0, 1)`` (machine-local) OVERRIDES the per-solve DB value; otherwise
    the DB ``db_value`` (the ``solve.benders_in_out_weight`` parameter,
    default ``0.0`` = OFF).  A non-float OR an out-of-``[0, 1)`` env value
    is IGNORED (not fatal, warned) and the DB value is used — ``λ ≥ 1``
    never queries the master (non-convergent) and ``λ < 0`` is meaningless,
    both config mistakes worth surfacing rather than silently clamping.
    """
    weight = db_value
    env = os.environ.get(_BENDERS_IN_OUT_WEIGHT_ENV)
    if env:
        try:
            env_w = float(env)
        except ValueError:
            _logger.warning(
                "Benders: ignoring non-float %s=%r",
                _BENDERS_IN_OUT_WEIGHT_ENV, env,
            )
        else:
            if 0.0 <= env_w < 1.0:
                weight = env_w
            else:
                _logger.warning(
                    "Benders: ignoring out-of-range %s=%r (must be in [0, 1); "
                    ">= 1 never queries the master, < 0 is meaningless)",
                    _BENDERS_IN_OUT_WEIGHT_ENV, env,
                )
    return weight


def _stall_worst_offenders(
    autarky_by_region: dict[str, float],
    region_costs: dict[str, float],
) -> tuple[str, float, str, float]:
    """Pick the ROOT and SYMPTOM node groups for the stall diagnostic.

    Both are derived purely from two ``{region: cost}`` maps — no solver state —
    so this is unit-testable in isolation (à la ``_check_cuts_satisfied``).

    * **Root** = ``argmax_r |autarky_r|``: the node group whose STAND-ALONE cost
      dominates — the one that cannot meet its own demand without imports.  Its
      ``autarky_ratio`` is ``|autarky_root| / max(1, second-largest |autarky|)``
      (how many times larger than the next node group; ``1.0`` when there is
      only one).
    * **Symptom** = ``argmax_r ratio_r``, ``ratio_r = |region_cost_r| /
      max(1, |autarky_r|)``: the node group forced worst into penalty/slack flow
      at the failing iteration.  ``max(1, ·)`` guards a near-zero-autarky node
      group from dividing by ~0.

    Returns ``(root, autarky_ratio, symptom, symptom_ratio)``.  Ties break on
    name (``sorted``) for determinism.  Empty inputs are not expected (the loop
    only calls this with a populated bootstrap), but degrade gracefully to
    ``("", 1.0, "", 1.0)``.
    """
    if not autarky_by_region:
        return "", 1.0, "", 1.0

    # Root: largest |autarky|, ties broken by name.
    ranked = sorted(
        autarky_by_region.items(), key=lambda kv: (-abs(kv[1]), kv[0])
    )
    root, root_autarky = ranked[0]
    second = abs(ranked[1][1]) if len(ranked) > 1 else 0.0
    autarky_ratio = abs(root_autarky) / max(1.0, second)

    # Symptom: largest current-cost / autarky ratio, ties broken by name.
    def _ratio(name: str) -> float:
        return abs(region_costs.get(name, 0.0)) / max(1.0, abs(autarky_by_region[name]))

    symptom = min(
        autarky_by_region, key=lambda name: (-_ratio(name), name)
    )
    symptom_ratio = _ratio(symptom)
    return root, autarky_ratio, symptom, symptom_ratio


@dataclass
class Coupling:
    """One cross-region ``(p, source, sink)`` coupling pair.

    Pairs an export half-flow with the matching import half-flow and
    carries the per-region ``v_flow`` column ids for each cell of the
    arc.  This is the SHARED decomposition substrate consumed by
    :func:`_build_arcs` (and re-exported for tests).  The dual-subgradient
    ``lam`` multipliers that the old Lagrangian scheme stored here are NOT
    part of the Benders contract and have been dropped.

    A region↔master coupling arc (master-hosted mode) is SINGLE-SIDED:
    the master side has no half-flow (the master keeps the whole original
    arc natively), so exactly one of ``export_region`` / ``import_region``
    is ``None`` (= the master) and that side's cols array is empty.
    """

    pipeline_key: tuple[str, str, str]
    export_region: str | None
    import_region: str | None
    export_cols: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    import_cols: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


def _identify_coupling_cols(
    splits: list[RegionSplit],
    warm: list[WarmProblem],
    master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
) -> list[Coupling]:
    """Pair :class:`HalfFlow`s on ``(p, source, sink)`` and resolve the
    ``v_flow`` column ids per region.  Used by :func:`_build_arcs` and by
    the decomposition tests directly.

    With a non-empty ``master_hosted_nodes`` the pair requirement relaxes
    to SINGLE-SIDED for region↔master arcs: a half-flow whose off-region
    terminal is master-hosted has no matching half-flow on the other side
    (the master keeps the whole original arc), so it yields a Coupling
    with the master side ``None``/empty.  With the default empty set the
    behaviour is byte-identical to the historical pair-only path.
    """
    by_e: dict[tuple, tuple[str, list[HalfFlow]]] = {}
    by_i: dict[tuple, tuple[str, list[HalfFlow]]] = {}
    for s in splits:
        for hf in s.half_flows:
            key = (hf.original_p, hf.original_source, hf.original_sink)
            (by_e if hf.side == "export" else by_i).setdefault(
                key, (s.region, []))[1].append(hf)

    region_idx = {s.region: i for i, s in enumerate(splits)}

    def _cols(vf, hf):
        return (vf.frame.filter(
            (pl.col("p") == hf.virtual_p)
            & (pl.col("source") == hf.virtual_arc_source)
            & (pl.col("sink") == hf.virtual_arc_sink)
        ).sort("d", "t"))["col_id"].to_numpy().astype(np.int64)

    out: list[Coupling] = []
    for key, (er, hfs_e) in by_e.items():
        if key not in by_i:
            # Single-sided EXPORT half-flow: a region node feeding a
            # master-hosted sink (region↔master arc).  Anything else
            # unpaired keeps today's silent skip.
            if key[2] in master_hosted_nodes:
                v_flow_e = warm[region_idx[er]]._p._vars["v_flow"]
                e_cols = _cols(v_flow_e, hfs_e[0])
                if e_cols.size == 0:
                    raise RuntimeError(
                        f"Benders: empty coupling columns for "
                        f"region↔master arc {key!r} (export side)."
                    )
                out.append(Coupling(
                    pipeline_key=key, export_region=er, import_region=None,
                    export_cols=e_cols,
                ))
            continue
        ir, hfs_i = by_i[key]
        v_flow_e = warm[region_idx[er]]._p._vars["v_flow"]
        v_flow_i = warm[region_idx[ir]]._p._vars["v_flow"]
        ehf, ihf = hfs_e[0], hfs_i[0]
        e_cols, i_cols = _cols(v_flow_e, ehf), _cols(v_flow_i, ihf)
        if e_cols.size == 0 or i_cols.size == 0:
            raise RuntimeError(
                f"Benders: empty coupling columns for arc {key!r} "
                f"(export={e_cols.size}, import={i_cols.size}).")
        if e_cols.size != i_cols.size:
            raise RuntimeError(
                f"Benders: pair size mismatch for {key!r}: "
                f"export={e_cols.size} vs import={i_cols.size}.")
        out.append(Coupling(
            pipeline_key=key, export_region=er, import_region=ir,
            export_cols=e_cols, import_cols=i_cols,
        ))
    # Single-sided IMPORT half-flows: a master-hosted source feeding a
    # region node.  Every by_i key already paired above was handled in
    # the by_e loop.
    for key, (ir, hfs_i) in by_i.items():
        if key in by_e:
            continue
        if key[1] not in master_hosted_nodes:
            continue
        v_flow_i = warm[region_idx[ir]]._p._vars["v_flow"]
        i_cols = _cols(v_flow_i, hfs_i[0])
        if i_cols.size == 0:
            raise RuntimeError(
                f"Benders: empty coupling columns for region↔master "
                f"arc {key!r} (import side)."
            )
        out.append(Coupling(
            pipeline_key=key, export_region=None, import_region=ir,
            import_cols=i_cols,
        ))
    return out


@dataclass
class _ArcMaster:
    """Master-side bookkeeping for one cross-region directed arc.

    For a region↔master coupling arc (master-hosted mode) exactly one of
    ``export_region`` / ``import_region`` is ``None`` — that side is the
    MASTER (no region pin columns; its pin-cols array is empty).
    """

    key: tuple  # (p, source, sink)
    conn: str  # the connection entity == key[0]
    export_region: str | None
    import_region: str | None
    # Ordered (by d,t) dim-tuples + master flow col-ids for f[arc, d, t].
    dim_tuples: list[tuple]
    f_col_ids: np.ndarray  # master f columns, aligned to dim_tuples
    # Per-region pinned-column ids (in the REGION's v_flow var), aligned to
    # dim_tuples.  Export region pins the export half-flow; import the import.
    export_pin_cols: np.ndarray
    import_pin_cols: np.ndarray


@dataclass
class BendersResult:
    """Outcome of :func:`solve_benders`."""

    converged: bool
    iterations: int
    total_objective: float  # best (min) UB = the recovered optimum
    lower_bound: float
    upper_bound: float
    gap: float
    region_costs: dict[str, float]  # cost_r at the incumbent f̄
    # Recovered master decisions at the incumbent.
    invest: dict[str, float]  # connection -> normalised invested capacity C
    # arc-key -> polars frame (p, source, sink, d, t, value) of the trade flow.
    trade_flow: dict[tuple, pl.DataFrame] = field(default_factory=dict)
    # Whole-system, owner-de-duplicated invest/divest decision frames for the
    # TIER-1 invest->dispatch handoff.  Keys are a subset of
    # ``_INVEST_VAR_NAMES``; each value is a long-form frame whose columns
    # match ``polar_high.Solution.value(name)`` exactly (``(entity_col, "d",
    # "value")``), so a downstream ``SnapshotSolution`` can expose them to
    # ``build_handoff_from_solution``.  UNION of region in-region invest +
    # master trade-connection invest.  Empty dict when the model has no
    # investment.
    invest_solution_vars: dict[str, pl.DataFrame] = field(default_factory=dict)
    # Convergence contract, carried so callers (the CLI exit-code scan) can
    # report a NON-convergence — a feasible incumbent whose ``gap`` never fell
    # to ``tol`` within ``max_iters`` — as a loud warning distinct from a
    # genuine infeasible / unbounded LP.  ``tol`` is the relative-gap target;
    # ``max_iters`` the iteration cap that was in force.
    tol: float = 0.0
    max_iters: int = 0


# ---------------------------------------------------------------------------
# Master construction (hand-built persistent WarmProblem).
# ---------------------------------------------------------------------------


class _BendersMaster:
    """Persistent master mirroring the monolith trade layer.

    Owns the trade flow vars, invest vars, the capacity coupling, and the
    ``η_r`` recourse vars; grows by appended cut rows.  Built ONCE;
    re-solved warm each iteration after appending cuts.

    Two construction paths, selected by ``master=``:

    * ``"flextool"`` (default, Phase 3a): the trade flow / invest / capacity
      / cost layer is generated by ``build_flextool`` over the network-only
      reduced :class:`FlexData` (:func:`_region_filter.master_network_data`)
      — the SAME emit the monolith uses, so the master picks up the
      ``maxFlow`` capacity coupling, the invest annuity, AND the connection
      flow cost natively / RP-consistently (the last is decisive at scale,
      Phase-3 §1.5 / §6).  ``v_flow[conn,d,t]`` and ``v_invest_p[conn,d]``
      are FlexTool-built; the ``η_r`` recourse vars + cut rows are appended
      via the polar-high primitives (:meth:`add_recourse_col` /
      :meth:`add_cut_row`).
    * ``"hand"`` (Phase 2): the hand-built ``f[arc,d,t]`` / ``C[conn]`` /
      normalised ``trade_capacity`` / invest-only objective.  Kept for the
      Phase-2 acceptance gate and master-vs-master comparison.

    The public surface (``solve``, ``set_eta_floor``, ``add_cut``,
    ``read_master``, ``master_native_cost``, the ``_eta_col`` /
    ``a.f_col_ids`` / ``_C_cols`` id maps) is identical across both paths so
    :func:`solve_benders` is path-agnostic.  On top of it the class
    implements the :class:`polar_high.benders.BendersMaster` protocol
    (``read_point`` / ``native_cost`` / ``project_point`` /
    ``relax_recourse`` / ``set_recourse_floor`` / ``compact_cuts``) so the
    generic coordinator can drive it directly.
    """

    def __init__(self, data: FlexData, arcs: list[_ArcMaster],
                 regions: list[str], eta_floor: float,
                 *, master: str = "flextool", obj_scale: float = 1.0,
                 master_hosted_nodes: frozenset[str] = frozenset(),
                 region_membership: dict | None = None):
        self.arcs = arcs
        self.regions = list(regions)
        self._eta_floor = eta_floor
        self._master_kind = master
        # Master-hosted node mode (empty default = byte-identical today's
        # path): the named balance/state nodes live natively in THIS master
        # (balances, penalties, storage, invest) — forwarded to
        # ``master_network_data`` on the flextool path.  The hand path
        # cannot host nodes (it builds only the trade layer).
        self._master_hosted_nodes = frozenset(master_hosted_nodes)
        self._region_membership = region_membership
        if master == "hand" and self._master_hosted_nodes:
            raise ValueError(
                "_BendersMaster: master='hand' cannot host nodes "
                f"(master_hosted_nodes={sorted(self._master_hosted_nodes)}); "
                "use the flextool master path."
            )
        # Objective scale ``s`` shared with the region subproblems.  The master
        # FlexTool objective is built ×s and each η enters at coef 1.0, so the
        # master objective lives in scaled space; the region cut slopes (region
        # objective duals) are ∂(s·currency) and drop in homogeneously.  See
        # ``_build_flextool_master`` and ``solve_benders``.
        self._obj_scale = float(obj_scale)
        # Master NATIVE cost (its own FlexTool objective minus the recourse
        # terms — invest annuity + flow cost today; master-hosted balance /
        # penalty / storage terms too once those land) at the last
        # ``read_master`` — stashed for the flextool path's
        # ``master_native_cost`` (which reads it from the FlexTool objective
        # rather than a hand coefficient sum).  Renamed from the misleading
        # ``_last_trade_cost`` (plan D-d).
        self._last_master_native_cost: float = 0.0
        # ``C_by_conn`` at the last ``read_point`` — consumed by
        # ``project_point`` (capacity clamp), ``native_cost`` and the
        # incumbent capture, all of which the coordinator calls with the
        # SAME master solution ``read_point`` just consumed.
        self._last_C_by_conn: dict[str, float] = {}
        # Count of LOOP master solves (``solve()``; the build-time solves go
        # through ``self._wp.solve()`` directly) — equals the coordinator's
        # 1-based iteration index, used to label ``project_point`` /
        # compaction diagnostics exactly as the in-module loop did.
        self._solve_count: int = 0
        if master == "flextool":
            self._build_flextool_master(data)
        elif master == "hand":
            self._build(data)
        else:
            raise ValueError(
                f"_BendersMaster: unknown master={master!r} "
                f"(expected 'flextool' or 'hand')"
            )

    # ------------------------------------------------------------------
    # FlexTool-generated master (Phase 3a).
    # ------------------------------------------------------------------

    def _build_flextool_master(self, data: FlexData) -> None:
        """Build the master trade layer via ``build_flextool`` over the
        network-only reduced FlexData, then append the ``η_r`` recourse
        columns and resolve the stable f / C / η col-id maps."""
        reduced = _region_filter.master_network_data(
            data, self.regions,
            region_membership=self._region_membership,
            master_hosted_nodes=self._master_hosted_nodes,
        )
        m = Problem()
        # AUTOSCALE-OFF: ``build_flextool`` never applies Layer 2 (that lives
        # only in ``_orchestration``); the appended cut rows then sit on the
        # built-column scale with no pre-scaling, identical to the hand path
        # and to the region subproblems (Phase-3 §2.3).
        #
        # Objective SCALING: build the master objective ×s (the SAME ``s`` the
        # regions are built with).  The master invest coef, flow-cost, and the
        # FlexTool objective are all multiplied by s natively (model.py:4128);
        # the appended η cols (coef 1.0) and the region cut slopes (∂(s·currency))
        # then live at the same scale, so the cut row is homogeneous in s and
        # cut validity is preserved (an exact monotone re-expression).  For the
        # small fixtures s=1.0 ⇒ byte-identical to the un-scaled build.
        _build_flextool(m, reduced, scale_the_objective=self._obj_scale)

        self._m = m
        self._wp = WarmProblem(m)
        _silence_if_quiet(self._wp)
        # Initial build so the warm handle exists and col ids resolve.  The
        # iter-0 master (no cuts) is bounded: η has the finite floor and
        # ``v_invest_p ≤ p_entity_max_units`` (FlexTool-emitted) bounds the
        # capacity (and hence ``v_flow`` via ``maxFlow``).
        self._wp.solve()
        self._built = True

        # Resolve f col-ids per arc by (p, source, sink, d, t) — the SAME
        # triple the region forward half-flow carries, so master f cell and
        # region pin cell map 1:1 (Phase-3 §1.4).
        for a in self.arcs:
            a.f_col_ids = np.array(
                [int(self._wp.col_id_of_var("v_flow", dt))
                 for dt in a.dim_tuples],
                dtype=np.int64,
            )
        # Resolve C col-ids per connection by (conn, d).  The reduced
        # FlexData's ``pd_invest_set`` gives one invest period per
        # INVESTABLE pipe; we sum the master invest value over those periods
        # in ``read_master``.  A trade connection of FIXED EXISTING capacity
        # only (``existing>0`` + no ``invest_method``) carries NO
        # ``v_invest_p`` column — it contributes an empty ``_C_cols`` entry
        # (zero invested capacity, zero handoff invest), and its flow is
        # bounded natively by the FlexTool ``maxFlow`` row at
        # ``p_flow_upper_existing = existing/unitsize`` (model.py:2468-2473;
        # ``has_invest_p`` False ⇒ no ``-v_invest_p`` LHS term ⇒
        # ``v_flow ≤ existing/unitsize``).  Mixed masters (some pipes
        # investable, some existing-only) are handled in ONE master: the
        # FlexTool emit bounds each arc by ``existing + invested``, the
        # invest var existing only for the investable subset.
        conns = sorted({a.conn for a in self.arcs})
        invest_periods: dict[str, list[str]] = {}
        if reduced.pd_invest_set is not None:
            for r in reduced.pd_invest_set.iter_rows(named=True):
                invest_periods.setdefault(r["p"], []).append(r["d"])
        has_invest_var = "v_invest_p" in self._wp._p._vars
        self._C_cols: dict[str, list[int]] = {}
        for c in conns:
            periods = invest_periods.get(c, [])
            if not periods or not has_invest_var:
                # Existing-only trade connection: no invest var/column.  An
                # empty list ⇒ ``read_master`` reports invested capacity 0
                # for ``c`` and ``trade_invest_frame`` carries no invest row
                # for it.  The existing capacity term lives in the FlexTool
                # ``maxFlow`` RHS (``_existing_cap_by_col`` below), not in C.
                self._C_cols[c] = []
                continue
            self._C_cols[c] = [
                int(self._wp.col_id_of_var("v_invest_p", (c, d)))
                for d in periods
            ]

        # Per master f col-id, the EXISTING flow capacity ``existing/unitsize``
        # the FlexTool ``maxFlow`` RHS (``p_flow_upper_existing``) enforces.
        # The master-side capacity self-check (``solve_benders``) bounds each
        # cell's chosen flow by ``existing_cap + invested`` — so an
        # existing-only arc (C=0) is correctly allowed flow up to its existing
        # capacity, while a greenfield arc (existing=0) is unchanged.  The
        # reduced data's ``p_flow_upper_existing`` is keyed
        # ``(p, source, sink, d[, t])`` and already unitsize-normalised
        # (existing/unitsize), the SAME normalisation the master ``v_flow``
        # lives in (Phase-1 §A.5), so it drops in cell-for-cell.
        self._existing_cap_by_col: dict[int, float] = {}
        fue = getattr(reduced, "p_flow_upper_existing", None)
        ex_lookup: dict[tuple, float] = {}
        if fue is not None:
            fr = fue.frame
            has_t = "t" in fr.columns
            for r in fr.iter_rows(named=True):
                if has_t:
                    ex_lookup[(r["p"], r["source"], r["sink"],
                               r["d"], r["t"])] = float(r["value"])
                else:
                    ex_lookup[(r["p"], r["source"], r["sink"], r["d"])] = (
                        float(r["value"]))
        for a in self.arcs:
            for dt, cid in zip(a.dim_tuples, a.f_col_ids):
                p, s, k, d, t = dt
                cap = ex_lookup.get((p, s, k, d, t))
                if cap is None:
                    cap = ex_lookup.get((p, s, k, d), 0.0)
                self._existing_cap_by_col[int(cid)] = cap

        # Append one η_r recourse column per region (cost=1.0 ⇒ each η
        # enters the master objective with coef 1.0 ON TOP of FlexTool's
        # own invest + flow cost — so the master objective is
        # FlexTool-invest+flow-cost + Σ η_r, exactly the Benders master).
        self._eta_col: dict[str, int] = {}
        for r in self.regions:
            self._eta_col[r] = int(self._wp.add_recourse_col(
                f"eta_{r}", cost=1.0, lower=self._eta_floor,
            ))
        # Re-solve so the appended η columns are part of the live model
        # before the loop's first cut append.
        self._wp._h.clearSolver()
        self._wp.solve()

        self._f_var_dims = ("p", "source", "sink", "d", "t")

    def _build(self, data: FlexData) -> None:
        arcs = self.arcs
        conns = sorted({a.conn for a in arcs})

        # --- invest annuity coefficient, read from the SAME source the
        # monolith uses: obj += Sum(v_invest_p * p_unitsize * annu)
        #                       + Sum(v_invest_p * p_unitsize * lifetime_fixed)
        # (model.py ~3900-3910).  We collapse the two e-d cost params into a
        # single per-connection coefficient `C_cost = unitsize * (annu + lf)`.
        # Single invest period in the prototype; we sum over the periods the
        # connection is invest-eligible (pd_invest_set).
        unitsize = self._param_map(data.p_unitsize, ("p",), "value")
        annu = self._param_map(
            data.ed_entity_annual_discounted, ("e", "d"), "value"
        )
        lf = self._param_map(data.ed_lifetime_fixed_cost, ("e", "d"), "value")
        invest_periods: dict[str, list[str]] = {}
        if data.pd_invest_set is not None:
            for r in data.pd_invest_set.iter_rows(named=True):
                invest_periods.setdefault(r["p"], []).append(r["d"])

        # Per-connection invest upper bound = p_entity_max_units (the monolith's
        # `maxInvest_var_bound`: v_invest_p <= p_entity_max_units, NORMALISED).
        # Mirroring it bounds the master (a single steep cut would otherwise
        # leave the LP unbounded until enough cuts accumulate).
        max_units = self._param_map(data.p_entity_max_units, ("e", "d"), "value")

        self._conn_cost: dict[str, float] = {}
        self._conn_cap: dict[str, float] = {}
        for c in conns:
            us = float(unitsize.get((c,), 0.0))
            periods = invest_periods.get(c, [])
            if not periods:
                raise RuntimeError(
                    f"Benders master: connection {c!r} has no pd_invest_set "
                    f"period — cannot form its invest cost coefficient"
                )
            coef = 0.0
            cap = 0.0
            for d in periods:
                coef += us * (float(annu.get((c, d), 0.0)) + float(lf.get((c, d), 0.0)))
                cap += float(max_units.get((c, d), 0.0))
            # Scale the invest cost coef by ``s`` so the hand master objective
            # (Σ C_cost·C + Σ η, η at coef 1.0 carrying the scaled recourse cost)
            # lives in the same scaled space as the regions / the flextool path.
            self._conn_cost[c] = coef * self._obj_scale
            self._conn_cap[c] = cap

        # --- build the hand master as a polar_high.Problem -----------------
        m = Problem()

        # Invest vars C[conn] >= 0 — one row per connection (single invest col).
        c_frame = pl.DataFrame({"conn": conns})
        C = m.add_var("C", ("conn",), c_frame, lower=0.0)

        # Trade flow vars f[arc_p, arc_source, arc_sink, d, t] >= 0 over the
        # union of every arc's (d,t) grid, keyed by the SAME (p,source,sink)
        # triple the region half-flows carry so the pin values map 1:1.
        f_rows = []
        for a in arcs:
            for (p, s, k, d, t) in a.dim_tuples:
                f_rows.append({"p": p, "source": s, "sink": k, "d": d, "t": t})
        f_frame = pl.DataFrame(f_rows)
        f = m.add_var("f", ("p", "source", "sink", "d", "t"), f_frame, lower=0.0)

        # eta_r recourse vars, lower-bounded by a large-negative finite floor
        # (see _ETA_FLOOR — keeps the cut-less iter-0 master kOptimal without
        # cutting off the optimum; the f̄=0 bootstrap then seeds the real cuts).
        eta_frame = pl.DataFrame({"region": self.regions})
        eta = m.add_var("eta", ("region",), eta_frame, lower=self._eta_floor)

        # Capacity coupling  f[arc,d,t] <= C[conn]   <=>   C[conn] - f >= 0
        # (NORMALISED: unitsize cancels — both f and C are unitsize-normalised).
        # Build one row per (arc, d, t).  We attach the connection key as a
        # join column so `Where` aligns f with its connection's C.
        cap_idx = pl.DataFrame(
            [
                {"conn": a.conn, "p": p, "source": s, "sink": k, "d": d, "t": t}
                for a in arcs
                for (p, s, k, d, t) in a.dim_tuples
            ]
        )
        # C[conn] - f[arc,d,t] >= 0  (f and C broadcast onto `over` by their
        # shared dims — `conn` for C, the 5 arc dims for f).
        m.add_cstr(
            "trade_capacity",
            over=cap_idx,
            sense=">=",
            lhs_terms={"cap": C, "flow_neg": -f},
            rhs_terms={"zero": 0.0},
        )

        # Invest upper bound  C[conn] <= max_units[conn]  (mirrors the
        # monolith maxInvest_var_bound; bounds the master).
        cap_param = Param(
            ("conn",),
            pl.DataFrame(
                {"conn": conns, "value": [self._conn_cap[c] for c in conns]}
            ),
        )
        m.add_cstr(
            "maxInvest",
            over=c_frame,
            sense="<=",
            lhs_terms={"invest": C},
            rhs_terms={"max_units": cap_param},
        )

        # Objective: Σ_conn C_cost[conn]·C[conn] + Σ_r eta_r.
        cost_param = Param(
            ("conn",),
            pl.DataFrame(
                {
                    "conn": conns,
                    "value": [self._conn_cost[c] for c in conns],
                }
            ),
        )
        m.set_objective(Sum(C * cost_param) + Sum(eta))

        self._m = m
        self._wp = WarmProblem(m)
        _silence_if_quiet(self._wp)
        # Initial build so the warm handle exists and col ids resolve.  The
        # iter-0 master (no cuts, η at its finite floor) is kOptimal; we never
        # read this solve's objective — the loop seeds the first cuts BEFORE the
        # first LB-bearing master solve.  But the build must succeed.
        self._wp.solve()
        self._built = True

        # Resolve stable master col-ids we reference in cut rows.
        self._eta_col: dict[str, int] = {
            r: int(self._wp.col_id_of_var("eta", (r,))) for r in self.regions
        }
        # f col-ids per arc were captured at arc construction time; re-resolve
        # them from the built master so they index the live HiGHS columns.
        for a in arcs:
            a.f_col_ids = np.array(
                [int(self._wp.col_id_of_var("f", dt)) for dt in a.dim_tuples],
                dtype=np.int64,
            )
        self._C_cols: dict[str, int] = {
            c: int(self._wp.col_id_of_var("C", (c,))) for c in conns
        }
        # The hand master is GREENFIELD-only (Phase-2 prototype: existing=0 ⇒
        # all trade capacity is invested ``C``).  No existing-capacity term,
        # so the per-col existing cap is uniformly 0 — the capacity self-check
        # then reduces to ``f ≤ C`` exactly as before.
        self._existing_cap_by_col: dict[int, float] = {}
        self._f_var_dims = ("p", "source", "sink", "d", "t")

    @staticmethod
    def _param_map(param, dims: tuple, value_col: str) -> dict[tuple, float]:
        """Build a ``{dim_tuple: value}`` lookup from a Param (or empty)."""
        if param is None:
            return {}
        fr = param.frame
        out: dict[tuple, float] = {}
        for r in fr.iter_rows(named=True):
            out[tuple(r[d] for d in dims)] = float(r[value_col])
        return out

    # -- per-iteration interface ----------------------------------------

    def set_eta_floor(self, floor: float) -> None:
        """Update every η column's lower bound on the live model.

        Skips regions whose η has already been relaxed to free (-inf) after
        contributing its first cut (see :meth:`relax_eta_after_cut`)."""
        relaxed = getattr(self, "_eta_relaxed", set())
        cols = np.array(
            [self._eta_col[r] for r in self.regions if r not in relaxed],
            dtype=np.int32,
        )
        if cols.size:
            lows = np.full(cols.size, float(floor), dtype=np.float64)
            highs = np.full(cols.size, highspy.kHighsInf, dtype=np.float64)
            self._wp._h.changeColsBounds(int(cols.size), cols, lows, highs)
        self._eta_floor = float(floor)

    def relax_eta_after_cut(self, region: str) -> None:
        """Relax ``η_region`` to free (lower=-inf) once it has at least one
        cut.  The cut(s) now bound ``η_region`` from below, so the finite
        bootstrap floor is no longer needed and removing it both tightens the
        master (no spurious floor-active corner) and narrows the bound dynamic
        range that drives the warm-resolve kUnknown."""
        relaxed = getattr(self, "_eta_relaxed", None)
        if relaxed is None:
            relaxed = self._eta_relaxed = set()
        if region in relaxed:
            return
        col = np.array([self._eta_col[region]], dtype=np.int32)
        lows = np.array([-highspy.kHighsInf], dtype=np.float64)
        highs = np.array([highspy.kHighsInf], dtype=np.float64)
        self._wp._h.changeColsBounds(1, col, lows, highs)
        relaxed.add(region)

    def solve(self):
        # Loop-iteration counter (1-based, one master solve per iteration):
        # labels the ``project_point`` clamp diagnostic and the compaction
        # timing line with the same iteration index the loop reports.
        self._solve_count += 1
        # Warm-restart: the master objective is scaled (scale_the_objective),
        # so appending a cut row and re-solving WARM stays kOptimal — no need
        # to throw away the basis every iteration.  WarmProblem.solve runs warm
        # first and only falls back to a cold clearSolver()+re-run if the warm
        # path fails to certify kOptimal (the proven cold fallback).
        sol = self._wp.solve(retry_on_unknown=True)
        if not sol.optimal:
            status = self._wp._h.getModelStatus()
            raise RuntimeError(_benders_failure_message(
                summary=(
                    f"Benders master problem did not solve to optimality "
                    f"(solver status {status}; "
                    f"{self._wp._h.getNumCol()} columns, "
                    f"{self._wp._h.getNumRow()} rows)."
                ),
                meaning=(
                    "The master problem — which decides how much capacity to "
                    "invest in the connections that couple the node groups — "
                    "could not be solved even after a cold restart. An "
                    "'infeasible' status usually means the connection bounds "
                    "cannot be satisfied together; an 'unbounded' status means "
                    "a connection investment has no (or a negative) cost and "
                    "the solver can build it for free."
                ),
                how_to_avoid=(
                    "Check the connections that couple the node groups for "
                    "contradictory bounds (e.g. a forced minimum above the "
                    "allowed maximum) and for missing or non-positive "
                    "investment costs. Re-run to rule out a transient solver "
                    "state. If it persists, please report it with the model."
                ),
            ))
        return sol

    def add_cut(self, region: str, f_bar: dict[int, float], cost_r: float,
                slopes: dict[int, float]) -> int:
        """Append the optimality cut for ``region``::

            eta_r  -  Σ_cell slope[cell]·f[cell]   >=   cost_r - Σ slope·f̄

        ``f_bar`` and ``slopes`` are keyed by MASTER f col-id.  Returns the
        appended row id.
        """
        eta_col = self._eta_col[region]
        col_ids: list[int] = [eta_col]
        coefs: list[float] = [1.0]
        rhs = cost_r
        for fcol, slope in slopes.items():
            if slope == 0.0:
                continue
            col_ids.append(int(fcol))
            coefs.append(-float(slope))
            rhs -= slope * f_bar[fcol]
        rhs = float(rhs)
        return self._wp.add_cut_row(col_ids, coefs, rhs)

    def compact_cuts(
        self, solution, *, policy: str = "slack", trial_col_values=None
    ) -> dict:
        """Compact the master's accumulated cut rows via
        :meth:`polar_high.WarmProblem.compact_cuts`.

        polar-high tracks every ``add_cut_row`` internally and prunes rows by
        one of two LB-preserving policies, then (verify) re-solves + rolls back
        if the objective drifted (the degenerate belt) — an operation FlexTool
        no longer reimplements:

        * ``policy="slack"`` classifies each cut by PRIMAL slack at
          ``solution.col_value`` and deletes the strictly-slack rows;
        * ``policy="dominance"`` groups cuts by their recourse (``η``) column
          and, over the ``trial_col_values`` window of recent master vertices,
          keeps per group only the oldest max-achiever at each trial point,
          dropping the dominated cuts AND the redundant degenerate ties that
          the slack policy would keep.

        Forwards ``policy`` / ``trial_col_values`` verbatim and returns the
        polar-high ``{"kept", "dropped", "restored"}`` report."""
        res = self._wp.compact_cuts(
            solution, policy=policy, trial_col_values=trial_col_values
        )
        # FlexTool-logger compaction report (the coordinator logs its own on
        # ``polar_high.benders``): log-following users and the compaction
        # tests read the kept/dropped counts off THIS module's logger, so the
        # line stays here after the loop extraction.
        _logger.info(
            "[benders timing] iter %d: cut compaction kept=%d dropped=%d "
            "restored=%s", self._solve_count, res["kept"], res["dropped"],
            res["restored"],
        )
        return res

    def read_master(self, sol) -> tuple[dict[str, dict[int, float]],
                                        dict[str, float], dict[str, float]]:
        """Return (f̄ per region-arc-cell, C per connection, eta per region)
        from a master solution.  f̄ is returned BOTH per arc and flattened by
        master col-id for cut bookkeeping."""
        f_by_col: dict[int, float] = {}
        for a in self.arcs:
            vals = sol.col_value[a.f_col_ids]
            for cid, v in zip(a.f_col_ids, vals):
                f_by_col[int(cid)] = float(v)
        # ``_C_cols`` carries a single col id per connection on the hand
        # path and a LIST (one per invest period) on the flextool path;
        # sum over the period columns either way.
        C_by_conn: dict[str, float] = {}
        for c, col in self._C_cols.items():
            cols = col if isinstance(col, (list, tuple, np.ndarray)) else [col]
            C_by_conn[c] = float(sum(sol.col_value[int(ci)] for ci in cols))
        eta_by_region = {r: float(sol.col_value[col]) for r, col in self._eta_col.items()}
        # Stash the master NATIVE cost at this solution.  On the flextool
        # path it is FlexTool's own objective MINUS Σ η_r (each η enters obj
        # with coef 1.0) — everything the master LP carries natively; on the
        # hand path it is the invest-only coefficient sum (computed in
        # ``master_native_cost``).
        if self._master_kind == "flextool":
            eta_sum = sum(eta_by_region.values())
            self._last_master_native_cost = float(sol.obj) - eta_sum
        return f_by_col, C_by_conn, eta_by_region

    def master_invest_frames(self, sol) -> dict[str, pl.DataFrame]:
        """Return every invest/divest decision var PRESENT in the master
        (``v_invest_p`` / ``v_invest_n`` / ``v_divest_p`` / ``v_divest_n``)
        as long-form ``(entity, d, value)`` frames in the SAME
        ``Solution.value`` semantics / unitsize-normalisation FlexTool
        emits — for the TIER-1 invest handoff.  ``{}`` on the hand master
        (no FlexTool invest Vars).

        Generalises the historical ``trade_invest_frame`` (which returned
        the trade-connection ``v_invest_p`` only): with master-hosted nodes
        the master also invests in its hosted STORAGE nodes
        (``v_invest_n``/``v_divest_n``) and in master-local units/
        connections, all of which must ride the handoff.  Each frame is
        built directly from ``Solution.value(name)`` (which indexes the
        master's Var frame by ``col_id``), so it is byte-identical in
        shape/units to what the monolith's value returns for those
        entities.  Master entities are in NO region's membership, so they
        are disjoint from every region's in-region invest by construction
        (see :func:`_assemble_benders_invest_vars`)."""
        if self._master_kind != "flextool":
            return {}
        out: dict[str, pl.DataFrame] = {}
        for name in _INVEST_VAR_NAMES:
            if name not in self._wp._p._vars:
                continue
            frame = sol.value(name)
            if frame is not None and frame.height > 0:
                out[name] = frame
        return out

    def master_native_cost(self, C_by_conn: dict[str, float]) -> float:
        """The master's OWN (native) cost at the last ``read_master`` — its
        full objective minus the recourse terms (``sol.obj − Σ η_r``).

        On the flextool path this is read from FlexTool's own objective, so
        it includes EVERYTHING the master LP carries natively: the coupling-
        connection invest annuity AND flow cost today (Phase-3 §2.4), plus —
        once master-hosted nodes land — their balance-penalty slack and
        storage-invest terms, with no arithmetic change here.  That is why
        the old name ``invest_cost`` (and the ``_last_trade_cost`` stash)
        was misleading and was renamed (plan D-d).  On the hand path it is
        the invest-only hand coefficient sum (the prototype's pipe flow cost
        is 0, so both agree)."""
        if self._master_kind == "flextool":
            return self._last_master_native_cost
        return sum(self._conn_cost[c] * C_by_conn[c] for c in C_by_conn)

    # -- polar_high.benders.BendersMaster protocol ------------------------
    # Thin adapters over the per-iteration interface above, so the generic
    # coordinator can drive this master directly.  All are called by the
    # coordinator with the SAME master solution within one iteration, in the
    # order read_point → project_point → native_cost.

    def read_point(self, sol) -> tuple[dict[int, float], dict[str, float]]:
        """Coordinator protocol: ``(coupling point by master f col-id,
        recourse value η per node group)`` at ``sol``.  Wraps
        :meth:`read_master` and stashes ``C_by_conn`` for
        :meth:`project_point` / :meth:`native_cost` / the incumbent
        capture."""
        f_by_col, C_by_conn, eta_by_region = self.read_master(sol)
        self._last_C_by_conn = C_by_conn
        return f_by_col, eta_by_region

    def native_cost(self, sol, recourse: dict[str, float]) -> float:
        """Coordinator protocol: the master's own cost at ``sol`` (objective
        minus recourse).  Delegates to :meth:`master_native_cost` at the
        ``C_by_conn`` stashed by :meth:`read_point` — bit-identical to the
        pre-coordinator in-module loop's call."""
        return self.master_native_cost(self._last_C_by_conn)

    def native_cost_at(self, point: dict[int, float]) -> float:
        """OPTIONAL coordinator protocol member (:meth:`polar_high.benders.
        BendersMaster.native_cost_at`): the master's NATIVE cost with every
        coupling flow PINNED at ``point`` (``{master f col-id -> value}``).

        The exact machinery of :func:`_master_autarky_cost` (save the
        coupling-column bounds, pin, solve once, read ``obj − Σ η``, restore
        the bounds in a ``finally``), but pinned to ``point`` values instead
        of 0 — so it evaluates the master's own cost at an ARBITRARY feasible
        coupling point.  Used ONLY by the off-loop single-point pin
        diagnostic (:func:`polar_high.benders.evaluate_at_point`); it is never
        reached from :func:`solve_benders_loop` (which reads the native cost
        at the master's own vertex via :meth:`native_cost`), so it adds no
        byte-parity risk to the loop.  ``point`` must carry a value for every
        coupling f col-id (the diagnostic supplies the full arc universe).
        """
        wp = self._wp
        col_ids = np.concatenate(
            [a.f_col_ids for a in self.arcs]
        ).astype(np.int64)
        vals = np.array(
            [float(point[int(c)]) for c in col_ids], dtype=np.float64
        )
        lo, hi = wp.get_col_bounds(col_ids)
        wp.fix_col_ids(col_ids, vals)
        try:
            sol = wp.solve(retry_on_unknown=True)
            if not sol.optimal:
                status = wp._h.getModelStatus()
                raise RuntimeError(
                    "Benders pin diagnostic: the master problem did not "
                    "solve to optimality with every node-group coupling "
                    "flow pinned at the diagnostic point (solver status "
                    f"{status})."
                )
            eta_sum = sum(
                float(sol.col_value[col]) for col in self._eta_col.values()
            )
            return float(sol.obj) - eta_sum
        finally:
            wp.set_col_bounds(col_ids, lo, hi)

    def project_point(self, f: dict[int, float], sol, *,
                      hard_fail: bool = True) -> float:
        """Coordinator protocol: clamp the coupling point ``f`` DOWN to the
        capacity the master chose (invested + existing), IN PLACE, returning
        the max clamp slack.  ``hard_fail=True`` (the master vertex) hard-
        fails on a GROSS overshoot with the domain diagnostic;
        ``hard_fail=False`` (in-out interior separation points, legitimately
        beyond the CURRENT capacity) clamps silently — see
        :func:`_clamp_flow_to_capacity`."""
        return _clamp_flow_to_capacity(
            f, self.arcs, self._last_C_by_conn, self._existing_cap_by_col,
            sol.max_primal_infeasibility,
            iterations=self._solve_count,
            hard_fail_gross=hard_fail,
        )

    def relax_recourse(self, sub_name: str) -> None:
        """Coordinator protocol name for :meth:`relax_eta_after_cut`."""
        self.relax_eta_after_cut(sub_name)

    def set_recourse_floor(self, floor: float) -> None:
        """Coordinator protocol name for :meth:`set_eta_floor`."""
        self.set_eta_floor(floor)


def _master_autarky_cost(master: _BendersMaster) -> float:
    """The master's NATIVE cost with every coupling flow pinned to 0 —
    the D-c "master autarky" term of the stall-guard reference scale.

    Definition (documented per plan D-c): *autarky is the cost of the
    whole decomposed system with every coupling flow pinned to zero —
    each node group self-supplies, and the master serves its hosted
    demand from its own storage / penalty slack without any node-group
    contribution.*  The master term is what this computes: pin all
    coupling-arc master ``v_flow`` columns (``a.f_col_ids`` of every
    coupling arc; master-local arcs stay free — they are the master's
    own dispatch) to 0, solve the master once, and read
    ``sol.obj − Σ η`` (each η enters the objective at coef 1.0, so the
    subtraction is exact regardless of the provisional recourse floor).

    The pinned bounds are captured beforehand (``get_col_bounds``) and
    RESTORED afterwards (``set_col_bounds``) — in a ``finally`` so a
    failed solve cannot leave the master corrupted for the rest of the
    run (plan risk R6).  Called ONCE, post-bootstrap, and ONLY when
    master-hosted nodes exist (the byte-parity gate: an extra master
    solve perturbs warm-basis state, so the empty-set path never runs
    this).  Feasibility of the pinned solve is guaranteed by the F9
    precondition (:func:`_assert_finite_boundary_penalties`): every
    master-hosted balance node carries finite penalty slack.
    """
    wp = master._wp
    col_ids = np.concatenate(
        [a.f_col_ids for a in master.arcs]
    ).astype(np.int64)
    lo, hi = wp.get_col_bounds(col_ids)
    wp.fix_col_ids(col_ids, np.zeros(col_ids.size, dtype=np.float64))
    try:
        sol = wp.solve(retry_on_unknown=True)
        if not sol.optimal:
            status = wp._h.getModelStatus()
            raise RuntimeError(_benders_failure_message(
                summary=(
                    f"Benders master problem did not solve to optimality "
                    f"with all node-group coupling flows pinned to zero "
                    f"(solver status {status})."
                ),
                meaning=(
                    "Before the main iterations, the master problem is "
                    "solved once with every flow between the node groups "
                    "and the master-hosted nodes forced to zero, to "
                    "measure the stand-alone cost of the master-hosted "
                    "nodes (their demand served from their own storage "
                    "and penalty slack). That solve failed, which "
                    "normally points to a master-hosted node whose "
                    "balance cannot be relaxed (missing or non-finite "
                    "penalty prices) or to a numerical problem in the "
                    "master."
                ),
                how_to_avoid=(
                    "Give every master-hosted node finite, moderate "
                    "penalty_up / penalty_down prices so its balance can "
                    "always be met with slack. Re-run to rule out a "
                    "transient solver state. If it persists, please "
                    "report it with the model."
                ),
            ))
        eta_sum = sum(
            float(sol.col_value[col]) for col in master._eta_col.values()
        )
        return float(sol.obj) - eta_sum
    finally:
        # Restore the pre-pin coupling-column bounds unconditionally.
        wp.set_col_bounds(col_ids, lo, hi)


# ---------------------------------------------------------------------------
# Region subproblem assembly.
# ---------------------------------------------------------------------------


@dataclass
class _Region:
    name: str
    wp: WarmProblem
    # forward arcs this region touches, with (region pin col-ids, master f
    # col-ids) aligned by (d,t).
    forward: list[tuple[_ArcMaster, np.ndarray, np.ndarray]]  # (arc, region_cols, master_cols)
    # reverse half-flow region col-ids to pin to 0.
    reverse_cols: np.ndarray
    # Layer-2 autoscale inverse plan for this region's subproblem (a FIXED
    # transform for the life of ``wp``), or ``None`` when Layer-2 did not fire.
    # Used to (a) scale the physical boundary-flow pins into the region's
    # scaled column space before ``fix_cols`` and (b) unscale the region
    # Solution (duals/cost/primal) back to master space after solve.
    plan: "Layer2Plan | None" = None


# ---------------------------------------------------------------------------
# TIER-1 whole-system invest assembly (region in-region + master trade).
# ---------------------------------------------------------------------------


def _resolve_entity_owner(
    region_membership: dict[str, dict[str, set[str]]],
    regions: list[str],
) -> dict[str, str]:
    """Build an ``entity -> owning-region`` map from region membership.

    Covers BOTH node entities (consumed by ``v_invest_n`` / ``v_divest_n``)
    and process/connection entities (consumed by ``v_invest_p`` /
    ``v_divest_p``).  A process is owned by the region that lists it in its
    ``"processes"`` set (the splitter assigns a process to a region via
    ``group_entity`` membership, i.e. the region containing its node(s)).

    *region_membership* is the EXCLUSIVE per-region membership returned by
    :func:`_region_filter.load_region_membership` — a region's OWN
    nodes/processes, NOT the shared set every region carries.  Ownership is
    therefore unambiguous for any entity claimed by exactly one region.

    An entity claimed by MORE than one region (shared, no unique owner) is
    assigned a deterministic owner: the first region in sorted region order
    that claims it (and a warning is emitted).  Iteration is over
    ``sorted(regions)`` so the tie-break is stable regardless of caller list
    order.

    Returns
    -------
    dict[str, str]
        ``{entity_name: region_name}`` for every node/process appearing in
        any region's membership.
    """
    owner: dict[str, str] = {}
    claims: dict[str, list[str]] = {}
    for region in sorted(regions):
        m = region_membership.get(region, {})
        for entity in m.get("nodes", set()) | m.get("processes", set()):
            claims.setdefault(entity, []).append(region)
    for entity, claiming in claims.items():
        owner[entity] = claiming[0]
        if len(claiming) > 1:
            _logger.warning(
                "Benders invest assembly: entity %r is shared across "
                "regions %r (no unique owner); assigning deterministic "
                "owner %r (first in sorted region order).  Shared "
                "invest-eligible entities are an untested edge case.",
                entity, claiming, owner[entity],
            )
    return owner


def _assemble_region_invest_vars(
    subproblems: list[Problem],
    subproblem_col_values: list[np.ndarray],
    owner_of_entity: Callable[[int, str], bool],
) -> dict[str, pl.DataFrame]:
    """Assemble whole-system invest/divest frames from the per-region
    recovered primal, keeping only owner-region rows.

    Parameters
    ----------
    subproblems
        Per-region :class:`polar_high.Problem` objects (region-index
        aligned).  Their ``_vars[name].frame`` carries ``(*dims, col_id)``
        and ``_vars[name].dims`` gives the natural dim order
        (``(entity_col, "d")`` for the invest/divest vars).
    subproblem_col_values
        Per-region recovered-primal ``col_value`` arrays
        (``Solution.col_value`` of each region's incumbent solve),
        region-index aligned with *subproblems*.  An empty / missing entry
        causes that region to be skipped.
    owner_of_entity
        Predicate ``(region_idx, entity) -> bool`` — ``True`` iff region
        ``region_idx`` OWNS ``entity``.  Only owned rows are kept, so the
        concatenated per-var frame has disjoint entity keys.

    Returns
    -------
    dict[str, pl.DataFrame]
        ``{name: frame}`` for each invest/divest var present in at least
        one region with >=1 owned row.  Each frame's columns exactly match
        ``polar_high.Solution.value(name)`` — ``(entity_col, "d",
        "value")`` — so a ``SnapshotSolution`` can serve them via
        ``.value(name)``.
    """
    out: dict[str, pl.DataFrame] = {}
    n_regions = len(subproblems)
    for name in _INVEST_VAR_NAMES:
        per_region_kept: list[pl.DataFrame] = []
        entity_col: str | None = None
        for i in range(n_regions):
            pb = subproblems[i]
            var = pb._vars.get(name)
            if var is None:
                continue
            if i >= len(subproblem_col_values):
                continue
            col_values = subproblem_col_values[i]
            if col_values is None or len(col_values) == 0:
                continue
            # Materialize this region's long-form frame exactly as
            # ``Solution.value(name)`` does: index the region's recovered
            # ``col_value`` by the Var's ``col_id`` and attach as "value".
            dims = tuple(var.dims)
            ent_col = dims[0]
            entity_col = ent_col
            frame = var.frame
            ids = frame["col_id"].to_numpy()
            vals = np.asarray(col_values)[ids]
            region_frame = frame.select(*dims).with_columns(
                value=pl.Series("value", vals)
            )
            # Owner-select: keep only rows whose entity is owned by this
            # region, so the concatenated frame has disjoint entity keys.
            entities = region_frame[ent_col].to_list()
            owned_mask = [bool(owner_of_entity(i, e)) for e in entities]
            # Canary: a NON-owner region carrying a materially non-zero
            # value violates the owner-selection assumption.  Warn, keep
            # only the owner's value.
            value_series = region_frame["value"].to_list()
            for e, owned, v in zip(entities, owned_mask, value_series):
                if (not owned) and v is not None and abs(v) > _NONOWNER_NONZERO_ABS_TOL:
                    _logger.warning(
                        "Benders invest assembly: non-owner region index "
                        "%d carries non-zero %s value %.6g for entity %r "
                        "(expected ~0 for an out-of-region invest var); "
                        "keeping only the owner's value.",
                        i, name, v, e,
                    )
            kept = region_frame.filter(pl.Series("__owned", owned_mask))
            if kept.height > 0:
                per_region_kept.append(kept)
        if per_region_kept:
            frame = pl.concat(per_region_kept, how="vertical")
            sort_cols = [c for c in (entity_col, "d") if c in frame.columns]
            if sort_cols:
                frame = frame.sort(sort_cols, maintain_order=True)
            out[name] = frame
    return out


def _assemble_benders_invest_vars(
    *,
    subproblems: list[Problem],
    region_of_index: list[str],
    region_membership: dict[str, dict[str, set[str]]],
    regions: list[str],
    region_col_values: list[np.ndarray] | None,
    master_invest_frames: dict[str, pl.DataFrame] | None,
    trade_conns: set[str],
) -> dict[str, pl.DataFrame]:
    """Assemble the whole-system TIER-1 invest handoff: the UNION of each
    region's owner-selected in-region invest/divest frames and the master's
    own invest/divest frames (:meth:`_BendersMaster.master_invest_frames`).

    The two contributions are DISJOINT by construction, but the partition is
    NOT pure region-membership ownership: a cross-region trade connection
    typically appears in BOTH regions' ``group_entity`` membership (it
    touches a node in each), so ``_resolve_entity_owner`` would otherwise
    hand it to one of them — clobbering the master's correct invested value
    with that region's pinned half-flow model's ZERO invest var.  The MASTER
    owns the trade-connection invest (it is the variable the capacity
    coupling acts on), so we EXCLUDE the trade connections from the region
    invest and take their value SOLELY from the master frames.  The master's
    OTHER invest entities — master-hosted storage nodes (``v_invest_n`` /
    ``v_divest_n``) and master-local units/connections — are in NO region's
    membership and carry no rows in any region frame (the splitter scrubs
    them, F3), so their union is disjoint by construction too; the
    defensive ``unique`` is a belt-and-braces guard only.

    Returns the same-shaped dict the downstream ``SnapshotSolution`` /
    ``build_handoff_from_solution`` expects (each frame's columns match
    ``Solution.value(name)`` exactly: ``(entity_col, "d", "value")``).
    """
    # (a) region in-region invest, owner-de-duplicated, with the cross-region
    # trade connections EXCLUDED (they are the master's, see docstring).
    out: dict[str, pl.DataFrame] = {}
    if region_col_values is not None:
        owner_by_entity = _resolve_entity_owner(region_membership, regions)

        def _owner_of_entity(region_idx: int, entity: str) -> bool:
            if entity in trade_conns:
                return False  # master-owned; never claimed by a region
            return owner_by_entity.get(entity) == region_of_index[region_idx]

        out = _assemble_region_invest_vars(
            subproblems, region_col_values, _owner_of_entity
        )

    # (b) master invest/divest frames — the trade-connection ``v_invest_p``
    # plus (master-hosted mode) the hosted storage nodes' ``v_invest_n`` /
    # ``v_divest_n`` and the master-local procs' invest.  Union each frame
    # into the matching region frame; deterministic ``_INVEST_VAR_NAMES``
    # order.  Ownership is disjoint (see docstring); ``unique`` is a belt.
    frames = master_invest_frames or {}
    for name in _INVEST_VAR_NAMES:
        frame = frames.get(name)
        if frame is None or frame.height == 0:
            continue
        existing = out.get(name)
        if existing is None:
            out[name] = frame
            continue
        ent_col = frame.columns[0]
        frame = frame.select(existing.columns)
        merged = pl.concat([existing, frame], how="vertical")
        merged = merged.unique(
            subset=[ent_col, "d"], keep="first", maintain_order=True
        )
        sort_cols = [c for c in (ent_col, "d") if c in merged.columns]
        out[name] = (
            merged.sort(sort_cols, maintain_order=True)
            if sort_cols else merged
        )
    return out


def _build_arcs(
    splits, warm,
    master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
) -> list[_ArcMaster]:
    """Discover the cross-region directed arcs + per-region pin columns.

    ``master_hosted_nodes`` is forwarded to
    :func:`_identify_coupling_cols`: non-empty, region↔master arcs come
    back SINGLE-SIDED (one region side ``None``) and the ``(d, t)``
    dim-tuples are recovered from whichever region side exists."""
    couplings = _identify_coupling_cols(
        splits, warm, master_hosted_nodes=master_hosted_nodes
    )
    region_idx = {s.region: i for i, s in enumerate(splits)}
    arcs: list[_ArcMaster] = []
    for cpl in couplings:
        # Recover the (d,t) dim-tuples in the export region's column order
        # (import region's for a master→region single-sided arc — the only
        # region side that exists there).
        if cpl.export_region is not None:
            vf = warm[region_idx[cpl.export_region]]._p._vars["v_flow"]
            side_cols = cpl.export_cols
        else:
            vf = warm[region_idx[cpl.import_region]]._p._vars["v_flow"]
            side_cols = cpl.import_cols
        hf_rows = vf.frame.filter(
            pl.col("col_id").is_in(side_cols)
        ).sort("d", "t")
        # Master arc dims use the ORIGINAL (p, source, sink) triple.
        p, s, k = cpl.pipeline_key
        dim_tuples = [
            (p, s, k, r["d"], r["t"]) for r in hf_rows.iter_rows(named=True)
        ]
        arcs.append(
            _ArcMaster(
                key=cpl.pipeline_key,
                conn=p,
                export_region=cpl.export_region,
                import_region=cpl.import_region,
                dim_tuples=dim_tuples,
                f_col_ids=np.zeros(len(dim_tuples), dtype=np.int64),  # filled by master
                export_pin_cols=cpl.export_cols.astype(np.int64),
                import_pin_cols=cpl.import_cols.astype(np.int64),
            )
        )
    return arcs


def _reverse_cols(split, warm: WarmProblem) -> np.ndarray:
    """All half-flow v_flow columns in ``split`` whose virtual arc is a
    REVERSE cross-region direction (so we pin them to 0)."""
    vf = warm._p._vars["v_flow"]
    # A reverse half-flow is one whose (original_source, original_sink) is the
    # reverse of a forward coupling; but simpler: pin EVERY half-flow that is
    # not a forward-pinned one.  We compute forward virtual cols separately, so
    # here gather all half-flow virtual cols and let the caller subtract.
    cols = []
    for hf in split.half_flows:
        sub = vf.frame.filter(
            (pl.col("p") == hf.virtual_p)
            & (pl.col("source") == hf.virtual_arc_source)
            & (pl.col("sink") == hf.virtual_arc_sink)
        )
        cols.append(sub["col_id"].to_numpy().astype(np.int64))
    return np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def solve_benders(
    data: FlexData,
    regions: list[str],
    *,
    max_iters: int = 20,
    tol: float = 1e-4,
    monolith_objective: float | None = None,
    build_problem=None,
    master: str = "flextool",
    scale_the_objective: float = 1.0,
    in_out_weight: float = 0.0,
    progress_callback: Callable[[dict], None] | None = None,
    subsolve_callback: Callable[[dict], None] | None = None,
    workers: int | None = None,
) -> BendersResult:
    """Run the multi-cut Benders loop on the regional decomposition of
    ``data``.

    Parameters
    ----------
    data
        The undecomposed :class:`FlexData`.
    regions
        Region names (the group entities the splitter partitions on).
    max_iters
        Iteration cap.
    tol
        Relative gap tolerance ``(best_UB − LB)/|best_UB|`` for convergence.
    monolith_objective
        If given (not ``None``), the loop asserts ``LB ≤ M·(1+1e-9)`` every
        iteration (an OPTIONAL test-time valid-lower-bound check against a
        known monolith optimum).  Pass ``None`` (the at-scale driver does) to
        skip it when no trustworthy/up-to-date M is available — the ALWAYS-ON
        ``LB ≤ best_UB`` sandwich guard is the monolith-free invalid-bound
        detector and needs no M.  ``M`` is interpreted in REAL units (same as
        the returned ``total_objective``/bounds), regardless of
        ``scale_the_objective``.
    build_problem
        Region builder.  Defaults to :func:`build_flextool` with the loop's
        ``scale_the_objective`` applied (so the regions build at the SAME
        scale as the master).  A custom builder is used as-is.
    master
        Master construction path: ``"flextool"`` (default, Phase 3a) builds
        the trade layer via ``build_flextool`` over the network-only reduced
        FlexData; ``"hand"`` (Phase 2) hand-builds it.  See
        :class:`_BendersMaster`.
    scale_the_objective
        Objective scale ``s`` applied to BOTH the master and the region
        subproblems (the solve's ``scale_the_objective``; ``s=1e-6`` for the
        real H2_trade solve, ``s=1.0`` for the small fixtures).  Because the
        cut coefficients are region OBJECTIVE duals, scaling both objectives by
        the same ``s`` scales the cuts consistently — an exact monotone
        re-expression that preserves cut validity and conditions the master LP
        (curing the at-scale warm-resolve kUnknown).  The loop's internal LB/UB
        arithmetic runs in scaled space; the returned ``total_objective``,
        ``lower_bound`` and ``upper_bound`` are UNSCALED back to real units
        (``÷s``).  ``s=1.0`` ⇒ byte-identical to the un-scaled path.
    in_out_weight
        The per-solve DB value of ``solve.benders_in_out_weight`` (the
        in-out separation weight ``λ``; default ``0.0`` = OFF = exact
        Benders, byte-identical).  Passed through to
        :func:`_resolve_benders_in_out_weight`, where the machine-local
        ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT`` env — when set and valid —
        overrides it.
    progress_callback
        Optional ``Callable[[dict], None]`` invoked ONCE per outer Benders
        iteration (after that iteration's master + region solves), so a
        caller (e.g. the orchestrator) can stream live per-iter lines.  The
        dict carries at least ``iter`` (1-based), ``lower_bound``,
        ``upper_bound`` (this iter's UB), ``best_upper_bound``, ``gap``,
        ``converged`` (bool), and ``region_costs`` — all bounds/costs in
        REAL units (÷s), matching the returned :class:`BendersResult`.
        No-op when ``None``; the loop behaviour is byte-identical when
        omitted.
    workers
        Worker-thread count for the per-iteration region recourse pass (the
        N independent region subproblem solves).  ``None`` (default)
        auto-resolves to ``min(n_regions, cpu_count - 1)``; ``<= 1`` (or a
        single region) keeps the fully-sequential path.  The machine-local
        env override ``FLEXTOOL_BENDERS_WORKERS`` takes precedence when set.
        The region solves are independent (each region owns its own HiGHS
        handle) and each solves single-threaded, so the parallel result is
        DETERMINISTIC — bit-identical ``(cost_r, cut slopes)`` and therefore
        identical LB/UB/iteration-count to ``workers=1`` (see the
        determinism gate test).

    Returns
    -------
    BendersResult
    """
    if build_problem is None:
        def build_problem(pb, d):
            _build_flextool(pb, d, scale_the_objective=scale_the_objective)

    _data_enums = getattr(data, "_axis_enums", None)
    _enums_token = None
    if _data_enums is not None and _data_enums != get_global_axis_enums():
        _enums_token = set_global_axis_enums(_data_enums)
    try:
        return _solve_benders_inner(
            data, regions, max_iters=max_iters, tol=tol,
            monolith_objective=monolith_objective, build_problem=build_problem,
            master=master, obj_scale=scale_the_objective,
            in_out_weight=in_out_weight,
            progress_callback=progress_callback,
            subsolve_callback=subsolve_callback, workers=workers,
        )
    finally:
        if _enums_token is not None:
            reset_global_axis_enums(_enums_token)


_BENDERS_PIN_DIAGNOSTIC_ENV = "FLEXTOOL_BENDERS_PIN_DIAGNOSTIC"
_BENDERS_PIN_MONOLITH_OBJ_ENV = "FLEXTOOL_BENDERS_PIN_MONOLITH_OBJ"
_BENDERS_PIN_BLOWUP_MULT_ENV = "FLEXTOOL_BENDERS_PIN_BLOWUP_MULT"
# Exactness band for the pin diagnostic's check (a): |Σcost − monolith| /
# monolith must be within this to call the decomposition "exact at the
# optimum".
_BENDERS_PIN_EXACT_BAND = 1e-2


def _maybe_run_pin_diagnostic(
    *, master, sub_adapters, arcs, f_bar,
    obj_scale: float, inv_s: float, eff_workers: int,
) -> "BendersResult | None":
    """The reusable go/no-go pin diagnostic (handoff §3).

    Gated ENTIRELY by the ``FLEXTOOL_BENDERS_PIN_DIAGNOSTIC`` env var (a path
    to a point file); UNSET ⇒ returns ``None`` and the caller runs the normal
    loop, byte-identical to today.  When set, it does NOT run the Benders loop
    — it evaluates the decomposed system ONCE at the pinned point and returns a
    :class:`BendersResult` carrying the diagnostic totals.

    The point file is a parquet with columns ``(p, source, sink, d, t,
    value)`` — the monolith's optimal flows on the coupling connections, keyed
    by the SAME ``(p, source, sink, d, t)`` identity each arc's ``dim_tuples``
    carry.  We map those onto the master coupling col-ids via
    ``zip(a.dim_tuples, a.f_col_ids)`` (the col-id ↔ (connection, source,
    sink, d, t) map), build the full coupling point over every arc f col-id
    (missing cells default to 0 and are counted), then:

      1. evaluate at ZERO coupling (``f_bar``) → per-region stand-alone
         reference cost;
      2. evaluate at the monolith point → per-region cost + master native
         cost, with each region flagged "blew up" iff it exceeds
         ``blowup_mult × stand-alone``.

    Verdict (handoff §3): (b) NO region blew up ⇒ the decomposition
    reproduces the monolith at its optimum with bounded recourse ⇒ **GO**;
    any region blew up ⇒ **STRUCTURAL**.  Check (a) — ``Σ region cost +
    master native ≈ monolith`` — is a corroborating exactness re-proof,
    printed when the monolith objective is supplied via
    ``FLEXTOOL_BENDERS_PIN_MONOLITH_OBJ`` (real units).
    """
    point_path = os.environ.get(_BENDERS_PIN_DIAGNOSTIC_ENV)
    if not point_path:
        return None

    blowup_mult = float(os.environ.get(_BENDERS_PIN_BLOWUP_MULT_ENV, "100"))
    mono_env = os.environ.get(_BENDERS_PIN_MONOLITH_OBJ_ENV)
    monolith_real = float(mono_env) if mono_env else None

    def _emit(line: str) -> None:
        print(line, flush=True)

    _emit(
        f"[benders pin-diagnostic] loading monolith handover flows from "
        f"{point_path!r}"
    )
    pts = pl.read_parquet(point_path)
    # Lookup keyed by the arc's (p, source, sink, d, t) identity.
    lookup: dict[tuple, float] = {
        (r["p"], r["source"], r["sink"], r["d"], r["t"]): float(r["value"])
        for r in pts.iter_rows(named=True)
    }
    # Build the full coupling point over EVERY arc f col-id; missing cells
    # default to 0 (the monolith carried no flow there) and are counted.
    point: dict[int, float] = {}
    n_cells = 0
    n_missing = 0
    for a in arcs:
        for dt, cid in zip(a.dim_tuples, a.f_col_ids):
            n_cells += 1
            v = lookup.get(tuple(dt))
            if v is None:
                n_missing += 1
                v = 0.0
            point[int(cid)] = v
    _emit(
        f"[benders pin-diagnostic] mapped {n_cells} coupling cell(s) onto "
        f"master col-ids ({n_missing} not present in the monolith flows → "
        f"pinned to 0)"
    )

    # (1) stand-alone reference at zero coupling; (2) the monolith point.
    ref = evaluate_at_point(
        master, sub_adapters, dict(f_bar), workers=eff_workers,
    )
    res = evaluate_at_point(
        master, sub_adapters, point,
        reference_costs=ref.sub_costs, blowup_mult=blowup_mult,
        workers=eff_workers,
    )

    # Report in REAL units (÷ obj_scale).
    total_real = res.total_cost * inv_s
    master_real = (
        res.master_native_cost * inv_s
        if res.master_native_cost is not None else None
    )
    _emit("[benders pin-diagnostic] === region + master breakdown "
          "(monolith optimal handover flows pinned) ===")
    for name in sorted(res.sub_costs):
        cost_real = res.sub_costs[name] * inv_s
        ref_real = ref.sub_costs.get(name, 0.0) * inv_s
        ratio = (
            abs(res.sub_costs[name]) / abs(ref.sub_costs[name])
            if ref.sub_costs.get(name) else float("inf")
        )
        flag = "BLEW UP" if res.blew_up.get(name) else "ok"
        _emit(
            f"[benders pin-diagnostic]   region {name!r}: cost={cost_real:.6e} "
            f"(stand-alone={ref_real:.6e}, ratio={ratio:.3g}×) [{flag}]"
        )
    if master_real is not None:
        _emit(
            f"[benders pin-diagnostic]   master native cost = {master_real:.6e}"
        )
    _emit(
        f"[benders pin-diagnostic]   TOTAL (Σ region + master native) = "
        f"{total_real:.6e}"
    )

    any_blew_up = any(res.blew_up.values())
    # (a) exactness re-proof (corroborating).
    if monolith_real is not None:
        rel = abs(total_real - monolith_real) / max(1.0, abs(monolith_real))
        exact_ok = rel <= _BENDERS_PIN_EXACT_BAND
        _emit(
            f"[benders pin-diagnostic]   (a) exactness: total {total_real:.6e} "
            f"vs monolith {monolith_real:.6e} → rel diff {rel:.3e} "
            f"({'MATCH' if exact_ok else 'MISMATCH'} at band "
            f"{_BENDERS_PIN_EXACT_BAND:g})"
        )
    else:
        exact_ok = None
        _emit(
            "[benders pin-diagnostic]   (a) exactness: monolith objective not "
            f"supplied ({_BENDERS_PIN_MONOLITH_OBJ_ENV} unset) — skipping the "
            "re-proof; (b) is the decisive check"
        )
    # (b) the decisive structural check.
    _emit(
        f"[benders pin-diagnostic]   (b) recourse: "
        f"{'a region HIT its penalty (blew up)' if any_blew_up else 'no region hit its penalty'}"
    )

    verdict = "STRUCTURAL" if any_blew_up else "GO"
    if verdict == "GO":
        _emit(
            "[benders pin-diagnostic] VERDICT: GO — the decomposed system "
            "reproduces the monolith at its optimal handover flows with "
            "bounded recourse; the stall is stabilization/volume, not "
            "structural."
        )
        if exact_ok is False:
            _emit(
                "[benders pin-diagnostic]   NOTE: (b) holds but (a) mismatched "
                "— check the point mapping / objective scale before trusting "
                "the exactness re-proof."
            )
    else:
        blown = sorted(n for n, b in res.blew_up.items() if b)
        _emit(
            "[benders pin-diagnostic] VERDICT: STRUCTURAL — region(s) "
            f"{blown} cannot reproduce the monolith even at the optimal "
            "handover flows (hit penalty/slack). Rank 1/2 will NOT save it; "
            "re-open the partition. STOP and surface this."
        )

    return BendersResult(
        converged=(verdict == "GO"),
        iterations=0,
        total_objective=total_real,
        lower_bound=total_real,
        upper_bound=total_real,
        gap=0.0,
        region_costs={n: c * inv_s for n, c in res.sub_costs.items()},
        invest={},
    )


def _solve_benders_inner(data, regions, *, max_iters, tol, monolith_objective,
                         build_problem, master="flextool",
                         obj_scale: float = 1.0,
                         in_out_weight: float = 0.0,
                         progress_callback=None, subsolve_callback=None,
                         workers=None) -> BendersResult:
    # --- master-hosted node set (plan §3 membership rule): every balance/
    # state node in NO region group is hosted natively in the MASTER.
    # Empty on every all-nodes-grouped model ⇒ every branch below keyed on
    # it takes today's exact path (byte-parity gate).  Logged LOUDLY here
    # (and _emit-announced by ``_run_benders_solve``, since the cascade pins
    # per-solve loggers to ERROR): an unexpectedly non-empty set on a chain
    # that groups everything is the R10 tripwire — a silently re-partitioned
    # node (replicate → master-hosted) must never go unannounced.
    region_membership = _region_filter.load_region_membership(data, regions)
    master_hosted = frozenset(
        _region_filter.compute_master_hosted_nodes(data, region_membership)
    )
    if master_hosted:
        _logger.info(
            "Benders: %d master-hosted node(s) — balance/state nodes in no "
            "region group, hosted natively in the master: %s",
            len(master_hosted), sorted(master_hosted),
        )

    # --- split with the cross-region half-flows UNCAPPED so the master pin is
    # feasible (Phase-2 splitter Benders mode).
    splits = _region_filter.split(
        data, regions=regions, region_membership=region_membership,
        benders_uncap_cross_region=True,
        master_hosted_nodes=master_hosted,
    )
    # Mixed-resolution coupling guard (master-hosted mode only, before any
    # LP build): a region↔master boundary connection must join SAME-time-
    # resolution nodes, or the single-sided half-flow pin silently distorts
    # the model (a leak, not a crash) — hard-error instead.
    _assert_no_mixed_resolution_coupling(data, splits, master_hosted)
    region_duals = _resolve_benders_region_duals()
    subproblems = [Problem() for _ in splits]
    # Full autoscale (Layer-2 per-type rescale + Layer-3 HiGHS-native top-up)
    # per region, aligned to ``splits``/``subproblems`` order.  Applied here —
    # after ``build_problem`` populates the FlexTool families and BEFORE
    # ``WarmProblem`` bakes the scaled bounds/coeffs into the canonical matrix —
    # so each region conditions like the (autoscaled) monolith instead of
    # grinding on raw 1e6-penalty columns.  Layer 3's ``simplex_scale_strategy``
    # is set via ``set_solver_option`` (which MERGES), so it composes with the
    # ``run_crossover`` pin below.  Only the Layer-2 plan is kept (to scale the
    # pins / unscale the solution); Layer 3 is HiGHS-internal and transparent to
    # the cut math.  ``resolve_scaling_config(None)`` reads ``FLEXTOOL_SCALING``
    # from the environment (same cascade-internal convention the monolith uses).
    _scale_cfg = resolve_scaling_config(None)
    region_plans: list[Layer2Plan | None] = []
    for s, pb in zip(splits, subproblems):
        build_problem(pb, s.data)
        region_plans.append(_apply_region_autoscale(pb, _scale_cfg, s.region))
        # PIN ``run_crossover=on`` on every region subproblem.  The Benders cut
        # slope is the reduced cost of the pinned boundary-flow column, which is
        # only a well-defined *basic* dual when the solve ends on a simplex basis.
        # Warm re-solves already run dual simplex and HiGHS' default is
        # crossover-on — but if a region grows large enough that ``solver=choose``
        # picks interior-point, crossover-OFF would return drifting interior duals
        # and non-reproducible, ill-conditioned cut slopes.  Pinning it guarantees
        # basic cut duals regardless of size or a future HiGHS default change.  Set
        # AFTER ``build_problem`` and via ``set_solver_option`` (which MERGES) so
        # the autoscaler's options (user_bound_scale / simplex_scale_strategy /
        # presolve) are preserved; ``WarmProblem`` reads this dict on its cold
        # build and the option persists across warm re-solves.  ``on`` is already
        # the HiGHS default, so this is byte-identical today — it is regression-
        # proofing, not a numerics change.
        # EXPERIMENTAL override (env, default OFF): ``interior`` swaps the pin
        # for barrier-without-crossover so cut slopes come from a central dual
        # of the (possibly flat) optimal face — see _BENDERS_REGION_DUALS_ENV.
        if region_duals == "interior":
            pb.set_solver_option("solver", "ipm")
            pb.set_solver_option("run_crossover", "off")
            pb.set_solver_option("ipm_optimality_tolerance", 1e-9)
        else:
            pb.set_solver_option("run_crossover", "on")
    warm = [WarmProblem(p) for p in subproblems]
    # Silence each region's per-solve HiGHS log by default (output_flag
    # persists across the cold build below and every warm parallel re-solve).
    for w in warm:
        _silence_if_quiet(w)
    # Initial build of every region (fix_cols / col_dual need a built model).
    for w in warm:
        w.solve()

    arcs = _build_arcs(splits, warm, master_hosted_nodes=master_hosted)
    if not arcs:
        raise RuntimeError(
            "Benders: no cross-region coupling arcs found — nothing to "
            "decompose"
        )

    # Boundary-node penalty finiteness precondition (optimality-cuts-only) —
    # extended (F9) to master-hosted balance nodes, which must carry BOTH
    # penalty params PRESENT and finite so the master's balance can always
    # be met with slack (incl. the D-c autarky pin below).
    _assert_finite_boundary_penalties(
        data, arcs, master_hosted_nodes=master_hosted
    )

    region_idx = {s.region: i for i, s in enumerate(splits)}

    # NOTE: ``_build_arcs`` filled each arc's ``f_col_ids`` with zeros; the
    # master rewrites them to its live HiGHS f columns at build time.  The
    # region-meta below references the REGION pin columns (export/import
    # half-flows), which are already resolved, so it can be built before the
    # master; we wire the master f col-ids into the meta after the master build.
    regions_meta: list[_Region] = []
    for s in splits:
        w = warm[region_idx[s.region]]
        forward: list[tuple[_ArcMaster, np.ndarray, np.ndarray]] = []
        fwd_region_cols: set[int] = set()
        for a in arcs:
            if s.region == a.export_region:
                region_cols = a.export_pin_cols
            elif s.region == a.import_region:
                region_cols = a.import_pin_cols
            else:
                continue
            forward.append((a, region_cols, a.f_col_ids))  # f_col_ids: see below
            fwd_region_cols.update(int(c) for c in region_cols)
        all_hf_cols = _reverse_cols(s, w)
        reverse_cols = np.array(
            [int(c) for c in all_hf_cols if int(c) not in fwd_region_cols],
            dtype=np.int64,
        )
        regions_meta.append(
            _Region(
                name=s.region, wp=w, forward=forward, reverse_cols=reverse_cols,
                plan=region_plans[region_idx[s.region]],
            )
        )

    # Pre-resolve dim-tuples for fix_cols (by region var frame, aligned to the
    # region pin col order).
    pin_dim_cache: dict[int, list[tuple]] = {}

    def _region_dim_tuples(w: WarmProblem, col_ids: np.ndarray) -> list[tuple]:
        vf = w._p._vars["v_flow"]
        fr = vf.frame.filter(pl.col("col_id").is_in(col_ids))
        # Order by the given col_ids.
        order = {int(c): i for i, c in enumerate(col_ids)}
        fr = fr.with_columns(
            pl.col("col_id").replace_strict(order, default=-1).alias("__o")
        ).sort("__o")
        return [tuple(r) for r in fr.select(*vf.dims).iter_rows()]

    def _pin_and_solve(rm: _Region, f_bar_local: dict[int, float]):
        """Pin region ``rm``'s forward half-flows to ``f_bar_local`` (reverse to
        0) and solve; return (cost_r, {master_f_col: slope}, region Solution).

        The returned :class:`polar_high.Solution` carries the region's
        recovered primal (incl. its in-region ``v_invest_p``/``v_invest_n``),
        used to assemble the whole-system TIER-1 invest handoff at the
        incumbent."""
        w = rm.wp
        for a, region_cols, master_cols in rm.forward:
            vals = np.array(
                [f_bar_local[int(mc)] for mc in master_cols], dtype=np.float64
            )
            # ``f_bar_local`` is a PHYSICAL boundary flow in master space, but
            # when the region is Layer-2 scaled the built model lives in scaled
            # column coordinates (``x_scaled = col_factor·x``).  ``fix_cols`` →
            # ``changeColsBounds`` writes the bound verbatim, so pin the SCALED
            # value ``col_factor·f̄``.  ``plan.col_factors`` is the forward
            # factor indexed by dense col-id; ``region_cols`` are those col-ids
            # in the same order as ``vals``/``master_cols``.
            if rm.plan is not None:
                vals = vals * rm.plan.col_factors[region_cols]
            dt = pin_dim_cache.setdefault(
                id(region_cols), _region_dim_tuples(w, region_cols)
            )
            w.fix_cols("v_flow", dt, vals)
        if rm.reverse_cols.size:
            dt_rev = pin_dim_cache.setdefault(
                id(rm.reverse_cols), _region_dim_tuples(w, rm.reverse_cols)
            )
            w.fix_cols(
                "v_flow", dt_rev, np.zeros(rm.reverse_cols.size, dtype=np.float64)
            )
        # ``retry_on_unknown=True`` mirrors the master's self-healing pattern
        # (see ``_BendersMaster.solve``): on any non-certified WARM status
        # (kUnknown/kSolveError off an ill-conditioned warm basis) drop the
        # basis and re-solve once cold, instead of crashing immediately.
        sol_r = w.solve(retry_on_unknown=True)
        if not sol_r.optimal:
            # Domain-side not-optimal: raised HERE, inside the adapter's
            # ``solve_at`` — it never reaches the coordinator's own checks
            # (which propagate it untouched, in region index order).  The
            # structured ``SubproblemNotOptimal`` type (a RuntimeError) lets
            # callers catch it while the message keeps the exact FlexTool
            # prose.
            raise SubproblemNotOptimal(rm.name, message=_benders_failure_message(
                summary=(
                    f"Benders node-group subproblem for {rm.name!r} did not "
                    "solve to optimality."
                ),
                meaning=(
                    "A single node group's dispatch+investment subproblem could "
                    "not be solved with the flows on its connections to other "
                    "node groups pinned to the master's values. Flow across the "
                    "node-group boundary is penalised rather than hard-"
                    "constrained, so this normally points to an infeasibility "
                    "WITHIN the node group (e.g. demand that cannot be met by "
                    "any combination of its own capacity, inflow and "
                    "investments) or a numerical problem there."
                ),
                how_to_avoid=(
                    f"Check node group {rm.name!r} in isolation: can its demand "
                    "be served by its own units/storages/inflows and allowed "
                    "investments? Look for missing capacity, an over-tight "
                    "constraint, or extreme parameter magnitudes. Re-run to rule "
                    "out a transient solver state. If it persists, please report "
                    "it with the model."
                ),
            ))
        # Undo the Layer-2 transform IN PLACE so cost/duals/primal come back
        # into master space: ``col_dual ×= col_factors`` (scaled reduced cost →
        # physical-space cut slope), ``col_value ÷= col_factors`` (→ physical,
        # what the TIER-1 invest handoff wants), ``obj`` invariant (already in
        # ``obj_scale`` space, = ``cost_r``).  After this the cut math in
        # ``_BendersMaster.add_cut`` is byte-consistent with the un-autoscaled
        # path and needs no change.  Detach the live HiGHS handle first so
        # ``unscale_solution``'s ``setSolution`` mirror is a no-op: the region
        # ``Solution.highs`` handle is never read downstream (unlike the
        # monolith's ``read_highs_solution`` path), and mirroring an unscaled
        # primal onto the WARM handle every iteration would perturb the next
        # warm re-solve — the exact conditioning this fix is protecting.
        if rm.plan is not None:
            sol_r.highs = None
            unscale_solution(sol_r, rm.plan)
        slopes: dict[int, float] = {}
        for a, region_cols, master_cols in rm.forward:
            rc = sol_r.col_dual[region_cols]
            for mc, g in zip(master_cols, rc):
                slopes[int(mc)] = slopes.get(int(mc), 0.0) + float(g)
        return float(sol_r.obj), slopes, sol_r

    # --- Build the master with a PROVISIONAL η floor (refined after the
    # bootstrap once we know the real cost scale).  The master assigns the live
    # f column ids (rewriting each arc's ``f_col_ids``), which the cut keying
    # below depends on, so it must be built BEFORE the bootstrap region solve.
    master_kind = master
    # Provisional floor in SCALED space (×obj_scale): just keeps the cut-less
    # initial build bounded; replaced post-bootstrap by the tight
    # -1.1·max|cost^autarky| floor below.  No cuts exist yet, so no warm-append
    # range concern at build time.
    master = _BendersMaster(
        data, arcs, [s.region for s in splits],
        eta_floor=-_ETA_FLOOR_MULT * 1e9 * obj_scale,
        master=master_kind, obj_scale=obj_scale,
        master_hosted_nodes=master_hosted,
        region_membership=region_membership,
    )
    # Re-bind the region-meta forward tuples to the master-rewritten f col-ids.
    for rm in regions_meta:
        rm.forward = [
            (a, region_cols, a.f_col_ids) for (a, region_cols, _old) in rm.forward
        ]

    # --- f̄ bootstrap point, keyed by the LIVE master f col-id: the all-zero
    # (autarky) dict over every arc's forward master columns.  Its KEY SET is
    # the coupling column universe for the whole run — the coordinator derives
    # the universe from nothing else (``initial_point`` contract).
    f_bar: dict[int, float] = {int(cid): 0.0 for a in arcs for cid in a.f_col_ids}

    # --- Parallel region recourse plumbing.  The N region subproblems are
    # independent (each owns its own WarmProblem / HiGHS handle, and HiGHS
    # run() releases the GIL); the coordinator fans the ``solve_at`` closures
    # out over a thread pool via ``solve_indexed_parallel`` exactly as this
    # driver's in-module loop used to.  Every region's cold first build
    # already ran SEQUENTIALLY above (the ``for w in warm: w.solve()`` loop),
    # so only WARM re-solves are parallelized and the per-region
    # ``(cost_r, slopes)`` are bit-identical to the sequential path.
    # ``_resolve_benders_workers`` applies the machine-local env override and
    # clamps to ``[1, n]``; the coordinator's own ``resolve_worker_count`` is
    # a fixed point on the resolved value, so the effective count is
    # identical.
    eff_workers = _resolve_benders_workers(len(regions_meta), workers)
    _logger.info(
        "Benders: region recourse pass over %d region(s) with %d worker(s)",
        len(regions_meta), eff_workers,
    )

    # Inverse objective scale: the coordinator (like the old in-module loop)
    # runs its LB/UB arithmetic in scaled space (objectives built ×s);
    # callers/tests (and the progress callback) expect REAL-unit costs (÷s).
    # s=1.0 ⇒ no-op.
    inv_s = 1.0 / obj_scale

    # In-out separation weight λ (machine-local env overrides the per-solve
    # DB value; 0.0 = OFF = exact Benders, byte-identical inside the
    # coordinator by construction).
    in_out_weight = _resolve_benders_in_out_weight(in_out_weight)
    if in_out_weight > 0.0 and master_hosted:
        # λ>0 evaluates the UB as master-native-cost(master vertex) +
        # Σ cost_r(interior f_sep).  That mixed point is a valid bound
        # only while the master's native cost is INDEPENDENT of the
        # coupling flows; a master hosting balance/storage nodes serves
        # real demand through them, so the mixed UB UNDER-COUNTS cost
        # (measured: LB legitimately crosses it → sandwich hard-fail).
        # Reject the combination up front — loud, never a silent
        # invalid bound.
        raise RuntimeError(_benders_failure_message(
            summary=(
                f"Benders in-out stabilization (weight "
                f"{in_out_weight:g}) is not supported together with "
                f"master-hosted nodes ({len(master_hosted)} balance/"
                f"storage node(s) outside every node group)."
            ),
            meaning=(
                "With in-out stabilization the node-group subproblems "
                "are evaluated at interior points while the master "
                "problem stays at its own solution. That mixed "
                "evaluation is a valid upper bound only while the "
                "master's own cost does not depend on the coupling "
                "flows — but a master hosting balance/storage nodes "
                "serves real demand through those flows, so the "
                "combination reports upper bounds BELOW the true cost "
                "(an invalid bound that later fails the bound checks) "
                "instead of merely converging slower."
            ),
            how_to_avoid=(
                "Set the solve's benders_in_out_weight to 0 (and unset "
                "any FLEXTOOL_BENDERS_IN_OUT_WEIGHT override) when the "
                "model has master-hosted nodes, or add every balance/"
                "storage node to a node group so nothing is hosted in "
                "the master."
            ),
        ))
    if in_out_weight > 0.0:
        _logger.info(
            "Benders: in-out separation ON (weight λ=%.3f) over %d region(s)",
            in_out_weight, len(regions_meta),
        )

    # Periodic MASTER CUT COMPACTION threshold (env-resolved; 0 = OFF =
    # byte-identical to the pre-compaction path — the coordinator skips the
    # whole compaction block at 0).
    compact_at = _resolve_benders_cut_compact_at()
    cut_policy = _resolve_benders_cut_policy()  # 'slack' (default) | 'dominance'
    # Capability guard: master cut compaction needs
    # ``polar_high.WarmProblem.compact_cuts`` (polar-high >= 3.5.0).  The
    # pyproject pin now guarantees it for pip installs, but an editable / dev
    # environment can still lag the pin — keep the belt so a stale install
    # degrades with a clear one-time message instead of crashing mid-solve
    # with an ``AttributeError``; the run then proceeds exactly like the
    # default (OFF) path.
    if compact_at > 0 and not hasattr(
        getattr(master, "_wp", None), "compact_cuts"
    ):
        _logger.warning(
            "benders: master cut compaction was requested "
            "(FLEXTOOL_BENDERS_CUT_COMPACT_AT=%d) but the installed "
            "polar-high has no WarmProblem.compact_cuts (needs >= 3.5.0); "
            "continuing without compaction.",
            compact_at,
        )
        compact_at = 0

    # --- Subproblem adapters.  ``solve_at`` is ``_pin_and_solve`` verbatim
    # (plan risk R13: the pin ×col_factors → solve → highs detach →
    # unscale_solution → read col_dual sequence is load-bearing and must not
    # be split); the coordinator never touches a region column itself.
    def _make_solve_at(rm: _Region):
        def _solve_at(point: dict[int, float]) -> SubproblemResult:
            cost_r, slopes, sol_r = _pin_and_solve(rm, point)
            return SubproblemResult(cost=cost_r, slopes=slopes, payload=sol_r)
        return _solve_at

    sub_adapters = [
        SubproblemHandle(rm.name, rm.wp, _make_solve_at(rm))
        for rm in regions_meta
    ]

    def _on_iteration(info: dict) -> None:
        # DIAGNOSTIC: surface the accumulated master cut-row count on THIS
        # module's logger every iteration (the master row count grows by
        # one-per-region-per-iteration, so this line makes the O(cuts) growth
        # of the master solve directly observable; with compaction ON it is
        # the post-append peak, before the loop-end compaction resets it).
        # The per-solve wall times live on the coordinator's own
        # ``polar_high.benders`` timing line.
        _logger.info(
            "[benders timing] iter %d: master_cut_rows=%d",
            info["iter"], info["cut_rows"],
        )
        if progress_callback is not None:
            # Stream one live per-iteration summary.  Bounds are reported in
            # REAL units (÷s) so the orchestrator's lines match the returned
            # ``BendersResult`` fields regardless of ``scale_the_objective``.
            progress_callback({
                "iter": info["iter"],
                "lower_bound": info["lower_bound"] * inv_s,
                "upper_bound": info["upper_bound"] * inv_s,
                "best_upper_bound": info["best_upper_bound"] * inv_s,
                "gap": info["gap"],
                "converged": info["converged"],
                "region_costs": {
                    r: c * inv_s for r, c in info["sub_costs"].items()
                },
            })

    _on_subsolve = None
    if subsolve_callback is not None:
        def _on_subsolve(info: dict) -> None:
            # Per-region FINISH event (worker thread; iter 0 = bootstrap).
            # ``cost`` arrives in scaled space → REAL units for the caller.
            subsolve_callback({
                "iter": info["iter"],
                "region": info["sub"],
                "obj": info["cost"] / obj_scale,  # cost_r → REAL units
            })

    def _on_incumbent(msol, sub_results, info) -> dict:
        # Incumbent capture for the TIER-1 invest handoff.  Everything is
        # MATERIALIZED here: the master ``master_invest_frames`` are fresh
        # DataFrames (the warm-restart reuses the master's ``col_value``
        # buffer across iterations, so a stashed Solution would read a later
        # iteration's values), and each region primal is ``.copy()``-d (the
        # region Solutions' buffers are likewise reused by later warm
        # re-solves).  ``sub_results`` is regions_meta/splits-index aligned
        # (the coordinator returns subproblem results in adapter order).
        return {
            "C": dict(master._last_C_by_conn),
            "master_invest_frames": master.master_invest_frames(msol),
            "region_col_values": [
                np.asarray(r.payload.col_value).copy() for r in sub_results
            ],
        }

    # --- D-c master-autarky stall reference — ONLY when master-hosted nodes
    # exist (byte-parity gate: an extra master solve perturbs warm-basis
    # state, so the empty-set path passes ``extra_reference_cost=None`` and
    # the coordinator adds nothing).  The coordinator calls the closure ONCE,
    # post-bootstrap, and folds |value| into the StallMonitor reference
    # scale: Σ_r|autarky_r| + |master_autarky|.  The computed value is kept
    # (``_master_autarky_holder``) so a later ``BendersStalled`` can carry
    # the master pseudo-entry into the stall diagnostics (D-c/D-d).
    _master_autarky_holder: list[float] = []
    extra_reference_cost = None
    if master_hosted:
        def extra_reference_cost() -> float:
            value = _master_autarky_cost(master)
            _master_autarky_holder.append(value)
            _logger.info(
                "Benders: master autarky (native cost at zero coupling "
                "flow) = %.6e (scaled space); added to the stall-guard "
                "reference scale", value,
            )
            return value

    options = BendersLoopOptions(
        max_iters=max_iters,
        tol=tol,
        in_out_weight=in_out_weight,
        workers=eff_workers,
        stall_window=_resolve_benders_max_stall(),
        # Raised gap floor so a loose ``tol`` never lets the floor fall below
        # the gap it must clear (same derivation the coordinator would apply
        # at None; passed explicitly to pin today's exact value).
        gap_floor=max(20.0 * tol, 0.02),
        compact_at=compact_at,
        cut_policy=cut_policy,
        cut_window=_resolve_benders_cut_window(),
        eta_floor_mult=_ETA_FLOOR_MULT,
        obj_scale=obj_scale,
        lb_gross_slack=_LB_GROSS_SLACK,
        # OPTIONAL test-time guard: M is supplied in REAL units; the
        # coordinator compares in the caller's (scaled) space, so pass M·s.
        monolith_objective=(
            monolith_objective * obj_scale
            if monolith_objective is not None else None
        ),
    )

    # --- OFF-LOOP pin diagnostic (handoff §3), env-gated by
    # ``FLEXTOOL_BENDERS_PIN_DIAGNOSTIC``.  When set, evaluate the decomposed
    # system ONCE at the monolith's optimal handover flows and return the
    # go/no-go verdict WITHOUT running the loop; UNSET ⇒ ``None`` and the
    # normal loop runs, byte-identical to today (nothing below reads the env
    # var).  Placed AFTER the split/subproblems/master are built and BEFORE
    # the coordinator, so it reuses the exact same adapters the loop would.
    _pin_result = _maybe_run_pin_diagnostic(
        master=master, sub_adapters=sub_adapters, arcs=arcs,
        f_bar=f_bar, obj_scale=obj_scale,
        inv_s=inv_s, eff_workers=eff_workers,
    )
    if _pin_result is not None:
        return _pin_result

    # --- Run the generic coordinator.  Its structured exceptions are
    # rendered below into the exact plain-English diagnostics this driver has
    # always raised (one ``_benders_failure_message`` call site per failure
    # kind — pinned by ``test_benders_failure_messages``).  The domain-side
    # not-optimal errors (region ``solve_at`` / ``_BendersMaster.solve``) and
    # the ``project_point`` capacity-clamp error carry their prose already
    # and propagate through the coordinator untouched.
    try:
        loop = solve_benders_loop(
            master,
            sub_adapters,
            options=options,
            initial_point=f_bar,
            extra_reference_cost=extra_reference_cost,
            on_iteration=_on_iteration,
            on_subsolve=_on_subsolve,
            on_incumbent=_on_incumbent,
        )
    except BendersBoundInvalid as exc:
        _raise_bound_invalid(
            exc, inv_s=inv_s, obj_scale=obj_scale,
            monolith_objective=monolith_objective,
        )
    except BendersStalled as exc:
        # D-c/D-d: hand the master pseudo-entry ("master-hosted nodes")
        # into the stall rendering — its stand-alone (autarky) cost next
        # to its per-iteration native cost, alongside the region maps —
        # so the diagnostic can name the master as root/symptom.  Both
        # are ``None`` (no entry injected) on the empty-set path.
        _raise_stalled(
            exc,
            master_autarky=(
                _master_autarky_holder[0]
                if _master_autarky_holder else None
            ),
            master_native_cost=master._last_master_native_cost,
        )

    # --- assemble result from the incumbent.  UNSCALE cost-valued outputs
    # back to real units (÷s): the loop's internal LB/UB/cost arithmetic ran
    # in scaled space (objectives built ×s), but callers/tests expect
    # real-unit costs.  ``invest`` (capacity C, MW) and ``trade_flow`` (MW)
    # are NOT costs and stay in their native (scale-invariant) units.
    # s=1.0 ⇒ no-op.
    payload = loop.incumbent_payload
    if payload is None:
        # No incumbent was recorded — only reachable on a zero-iteration run
        # (every real iteration improves the initial +inf best-UB).  Degrade
        # to an empty handoff rather than crash.
        payload = {
            "C": dict(master._last_C_by_conn),
            "master_invest_frames": {},
            "region_col_values": None,
        }
    trade_flow = _flow_frames(arcs, loop.incumbent_point)

    # --- TIER-1 whole-system invest handoff (GAP-a).  Assemble the same-shaped
    # ``{v_invest_p/v_invest_n/v_divest_p/v_divest_n -> (entity, d, value)}``
    # dict the downstream rolling-dispatch consumes, as the UNION of:
    #   (a) each REGION's in-region invest (owner-de-duplicated so each entity
    #       is claimed exactly once), AND
    #   (b) the MASTER's own invest/divest frames — the coupling-connection
    #       ``v_invest_p`` (disjoint from any region's in-region invest since
    #       the splitter never assigns a coupling connection to a region's
    #       membership) plus, with master-hosted nodes, the hosted storage
    #       nodes' ``v_invest_n``/``v_divest_n`` and the master-local procs'
    #       invest (in NO region's membership by construction).
    # NORMALISATION: both the region subproblems and the master are FlexTool-
    # built (``build_flextool``), so their ``v_invest_p`` carry IDENTICAL
    # p_unitsize-normalised units — the same units ``Solution.value("v_invest_p")``
    # returns on the monolith.  The assembled frames therefore drop straight
    # into ``build_handoff_from_solution`` with no rescale.
    invest_solution_vars = _assemble_benders_invest_vars(
        subproblems=subproblems,
        region_of_index=[s.region for s in splits],
        region_membership=region_membership,
        regions=regions,
        region_col_values=payload.get("region_col_values"),
        master_invest_frames=payload.get("master_invest_frames"),
        trade_conns={a.conn for a in arcs},
    )

    return BendersResult(
        converged=loop.converged,
        iterations=loop.iterations,
        total_objective=loop.best_upper_bound * inv_s,
        lower_bound=loop.lower_bound * inv_s,
        upper_bound=loop.best_upper_bound * inv_s,
        gap=loop.gap,
        region_costs={r: c * inv_s for r, c in loop.sub_costs.items()},
        invest=payload["C"],
        trade_flow=trade_flow,
        invest_solution_vars=invest_solution_vars,
        tol=tol,
        max_iters=max_iters,
    )


def _flow_frames(arcs: list[_ArcMaster], f_bar: dict[int, float]) -> dict[tuple, pl.DataFrame]:
    out: dict[tuple, pl.DataFrame] = {}
    for a in arcs:
        rows = []
        for dt, cid in zip(a.dim_tuples, a.f_col_ids):
            p, s, k, d, t = dt
            rows.append(
                {"p": p, "source": s, "sink": k, "d": d, "t": t,
                 "value": f_bar[int(cid)]}
            )
        out[a.key] = pl.DataFrame(rows)
    return out


def _clamp_flow_to_capacity(
    f_dict: dict[int, float],
    arcs: list[_ArcMaster],
    C_by_conn: dict[str, float],
    existing_cap_by_col: dict[int, float],
    solver_feas: float,
    *,
    iterations: int,
    hard_fail_gross: bool = True,
) -> float:
    """Clamp each coupling flow in ``f_dict`` DOWN to the capacity the master
    chose (``invested + existing``), IN PLACE, and return the max clamp slack.

    The master's chosen capacity must support its chosen flow: the coupling row
    ``C − f ≥ 0`` (≡ ``f ≤ existing_cap + Σ v_invest_p``) holds at the master
    optimum.  For a GREENFIELD arc the existing term is 0 (cap = invested C);
    for an EXISTING-only arc the invested C is 0 (cap = existing/unitsize); for
    a BOTH arc both contribute.

    The solver returns a vertex only within its feasibility tolerance of the
    active rows, and HiGHS enforces that on the INTERNALLY-SCALED problem — so
    the UNSCALED slack on ``f ≤ cap`` can exceed BOTH the nominal tolerance AND
    the reported ``max_primal_infeasibility`` (measured in scaled space), and it
    grows as cuts accumulate and the master gets more ill-conditioned.  Tuning a
    fixed tolerance is therefore chasing a moving target.  Instead CLAMP the
    flow down to the capacity it chose: any UB evaluated at the clamped point
    ``(C, min(f, cap))`` is a strictly capacity-feasible primal (a valid
    whole-problem upper bound — clamping flow DOWN can only raise region
    recourse cost, never invalidate it).  A GROSS overshoot — orders of
    magnitude beyond any plausible solver slack — still signals a real
    read/stale-state bug and hard-fails.

    Used for BOTH the master vertex ``f_out`` and (with in-out separation on)
    the interior separation point ``f_sep``.  For ``f_out`` a gross overshoot
    signals a real read/stale-state bug and HARD-FAILS (``hard_fail_gross=True``,
    the default — byte-identical to the original inline clamp).  For ``f_sep``
    (``hard_fail_gross=False``) a large overshoot is EXPECTED and legitimate: the
    interior centre is an OLD incumbent flow feasible against a PAST capacity, so
    when a later master picks a much smaller cap the convex combo can sit far
    above the CURRENT cap through no fault of the solver.  We just clamp it DOWN
    (raising recourse, keeping the UB valid) without the gross-bug guard.
    """
    max_clamp = 0.0
    for a in arcs:
        invested = C_by_conn.get(a.conn, 0.0)
        for cid in a.f_col_ids:
            cid = int(cid)
            cap = invested + existing_cap_by_col.get(cid, 0.0)
            f_val = f_dict[cid]
            slack = f_val - cap
            if slack <= 0.0:
                continue
            # Gross overshoot ⇒ genuine bug, not solver slack (only for the
            # master vertex; a stale-cap ``f_sep`` overshoot is legitimate).
            gross_tol = max(
                1e-2 * max(1.0, abs(cap), abs(f_val)), 1e3 * solver_feas
            )
            if hard_fail_gross and slack > gross_tol:
                raise RuntimeError(_benders_failure_message(
                    summary=(
                        f"Benders master chose a flow {f_val:.6e} on "
                        f"connection '{a.conn}' that exceeds the capacity "
                        f"{cap:.6e} it invested in (slack {slack:.3e} > "
                        f"{gross_tol:.3e}, solver feasibility "
                        f"{solver_feas:.3e}) at iteration {iterations}."
                    ),
                    meaning=(
                        "The master picked a flow on this connection larger "
                        "than the capacity it invested in, by far more than "
                        "solver rounding can explain (small overshoots are "
                        "absorbed automatically). That points to a real "
                        "inconsistency on this connection — typically a "
                        "units / normalisation mismatch or an extreme "
                        "cost/capacity magnitude — rather than solver noise."
                    ),
                    how_to_avoid=(
                        f"Check connection '{a.conn}' for an investment-cost, "
                        "capacity, efficiency or unitsize inconsistency or "
                        "extreme magnitude. Re-run to rule out a transient "
                        "solver state. If it persists, please report it with "
                        "the model."
                    ),
                ))
            f_dict[cid] = cap  # clamp to the supported capacity
            max_clamp = max(max_clamp, slack)
    return max_clamp


def _cut_separates(
    cost_r: float,
    slopes: dict[int, float],
    f_out: dict[int, float],
    f_sep: dict[int, float],
    eta_r: float,
) -> bool:
    """In-out separation test: does the cut GENERATED at the interior
    ``f_sep`` strictly separate the master vertex ``(f_out, eta_r)``?

    Delegated verbatim to the coordinator's
    :func:`polar_high.benders._cut_separates` — the arithmetic (row-scale
    ``tol_sep``, load-bearing against livelock on a degenerate vertex) moved
    there with the loop.  This wrapper keeps the FlexTool-side unit-test
    surface (and its ``eta_r`` vocabulary) stable."""
    return _ph_cut_separates(cost_r, slopes, f_out, f_sep, eta_r)


def _check_cuts_satisfied(cuts, f_bar, new_f_bar, eta_by_region, *,
                          iterations, inv_s) -> None:
    """Post-master cut self-check: at the NEW master point each just-appended
    cut must be SATISFIED, i.e.

        eta_r  >=  cost_r(f̄) + Σ_cell slope[cell]·(f_master[cell] − f̄[cell])

    The arithmetic — row-scale tolerance (keyed off the cut ROW's coefficient
    magnitude, not the possibly heavily-cancelled rhs), warn-and-continue
    below the gross band, hard-fail beyond it — moved verbatim into the
    coordinator (:func:`polar_high.benders._check_cuts_satisfied`), which the
    loop now runs internally.  This wrapper is the DOMAIN-RENDERING surface
    kept in FlexTool: it normalises the historical 3-tuple cut shape
    (``(region, cost_r, slopes)``, generation point = the shared ``f_bar``)
    to the coordinator's 4-tuple, delegates, and renders the structured
    ``BendersBoundInvalid`` into this driver's exact plain-English
    diagnostics — the same render sites the loop's exception handling uses.
    """
    norm = [
        cut if len(cut) == 4 else (cut[0], f_bar, cut[1], cut[2])
        for cut in cuts
    ]
    try:
        _ph_check_cuts_satisfied(
            norm, new_f_bar, eta_by_region, iteration=iterations
        )
    except BendersBoundInvalid as exc:
        _raise_cut_check_failure(exc, inv_s=inv_s)


# ---------------------------------------------------------------------------
# Structured-exception rendering: polar_high.benders → FlexTool prose.
#
# The coordinator raises structured exceptions carrying the numeric fields;
# FlexTool renders them into the exact plain-English summary / what-it-means /
# how-to-avoid texts this driver has always raised.  Each exception KIND gets
# its OWN ``_benders_failure_message`` call site with literal keyword strings
# (pinned by ``test_benders_failure_messages`` — the AST walk needs >= 8 such
# sites; together with the domain-side region/master not-optimal and the
# capacity-clamp sites above, these make up the full set).
# ---------------------------------------------------------------------------


def _raise_cut_check_failure(exc: BendersBoundInvalid, *, inv_s: float) -> NoReturn:
    """Render the two cut-self-check kinds (``cut_nonfinite`` /
    ``cut_violated``); re-raise anything else untouched."""
    if exc.kind == "cut_nonfinite":
        raise RuntimeError(_benders_failure_message(
            summary=(
                f"Benders recourse estimate η for node group {exc.sub_name!r} is "
                f"not a finite number ({exc.recourse_value!r}) after the master "
                f"solve at iteration {exc.iteration}."
            ),
            meaning=(
                "The master problem returned a non-finite value for a node "
                "group's recourse cost, which means that master solve did "
                "not produce a usable solution — almost always severe "
                "numerical ill-conditioning of the master (extreme "
                "connection investment-cost or capacity magnitudes) or a "
                "corrupted solver state."
            ),
            how_to_avoid=(
                "Re-run the solve to rule out a transient solver state. If "
                "it recurs, check the connections that couple the node "
                "groups for extreme or mismatched cost/capacity magnitudes "
                "and rescale the outliers. If it persists, please report it "
                "with the model."
            ),
        )) from exc
    if exc.kind == "cut_violated":
        raise RuntimeError(_benders_failure_message(
            summary=(
                f"Benders cut for node group {exc.sub_name!r} is violated at the "
                f"new master point: recourse estimate "
                f"η={exc.recourse_value * inv_s:.6e} is "
                f"below the cut floor {exc.cut_rhs * inv_s:.6e} (by "
                f"{exc.violation:.3e}, > {exc.gross_tol:.3e} at row scale "
                f"{exc.row_scale:.3e}) at iteration {exc.iteration}."
            ),
            meaning=(
                "A just-added Benders cut, which lower-bounds this node "
                "group's recourse cost, is not honoured by the master's own "
                "solution — by far more than solver rounding can explain "
                "(small violations are absorbed automatically). That points "
                "to a stale solver basis or a corrupted cut append, not to "
                "your model's economics."
            ),
            how_to_avoid=(
                "Re-run the solve first — a fresh basis usually clears a "
                "one-off warm-restart glitch. If it recurs, loosen the "
                "solver tolerance (e.g. --solver-mip-gap 0.01) and check the "
                "connections coupling the node groups for extreme "
                "investment-cost or capacity magnitudes that make the master "
                "ill-conditioned. If it persists, please report it with the "
                "model."
            ),
        )) from exc
    raise exc


def _raise_bound_invalid(
    exc: BendersBoundInvalid, *, inv_s: float, obj_scale: float,
    monolith_objective: float | None,
) -> NoReturn:
    """Render a coordinator ``BendersBoundInvalid`` into the exact
    plain-English diagnostics of the pre-coordinator in-module loop.

    All numeric fields on ``exc`` are in the loop's SCALED space; the
    cost-valued ones are unscaled (÷s) for display exactly as before.  The
    ``monolith`` kind reproduces the historical bare-RuntimeError text using
    the caller's REAL-unit ``monolith_objective`` (formatting the exception's
    scaled field ×inv_s could drift in the last bit when s != 1)."""
    if exc.kind == "lb_drop":
        raise RuntimeError(_benders_failure_message(
            summary=(
                f"Benders lower bound dropped "
                f"{exc.prev_lower_bound * inv_s:.6e} → "
                f"{exc.lower_bound * inv_s:.6e} "
                f"(by {exc.rel_drop:.2e}, > {exc.gross_band:.0e}) at iteration "
                f"{exc.iteration}."
            ),
            meaning=(
                "The Benders lower bound must never decrease — each "
                "iteration only adds cuts, which can only tighten it. A "
                "large drop means the master problem returned an "
                "inconsistent solution, almost always from a stale "
                "solver basis (warm restart) or severe numerical ill-"
                "conditioning of that master — not from your model's "
                "economics."
            ),
            how_to_avoid=(
                "Re-run the solve first — a fresh basis usually clears a "
                "one-off warm-restart glitch. If it recurs: loosen the "
                "solver tolerance (e.g. --solver-mip-gap 0.01); check the "
                "connections that couple the node groups for extreme "
                "investment-cost or capacity magnitudes / unit "
                "mismatches that make the master ill-conditioned. If it "
                "persists, please report it with the model."
            ),
        )) from exc
    if exc.kind == "sandwich":
        raise RuntimeError(_benders_failure_message(
            summary=(
                f"Benders lower bound {exc.lower_bound * inv_s:.6e} exceeds the "
                f"best feasible cost found {exc.best_upper_bound * inv_s:.6e} "
                f"(by {exc.rel_over:.2e}, > {exc.gross_band:.0e}) at iteration "
                f"{exc.iteration}."
            ),
            meaning=(
                "The lower bound has risen above a solution we already "
                "know is achievable, which is impossible for a valid "
                "bound. A gap this large means the master's bound is "
                "invalid — typically from severe numerical ill-"
                "conditioning of the master problem (very large or very "
                "small connection investment costs / capacities), not "
                "from your model being wrong."
            ),
            how_to_avoid=(
                "Check the connections that couple the node groups "
                "(investment cost, capacity, efficiency) for extreme "
                "magnitudes or unit errors and rescale/correct the "
                "outliers so the master is better conditioned. Loosen "
                "the solver tolerance (e.g. --solver-mip-gap 0.01) and "
                "re-run to rule out a transient solver state. If it "
                "persists, please report it with the model."
            ),
        )) from exc
    if exc.kind == "monolith":
        # OPTIONAL test-time guard (kept as the historical bare RuntimeError,
        # not a 3-section diagnostic — it only fires when a caller supplies a
        # known monolith optimum, i.e. in tests).
        raise RuntimeError(
            f"Benders LB {exc.lower_bound / obj_scale:.10e} exceeds monolith M "
            f"{monolith_objective:.10e} at iter {exc.iteration} — "
            f"INVALID lower bound (the bug this scheme fixes)"
        ) from exc
    _raise_cut_check_failure(exc, inv_s=inv_s)


#: Key of the MASTER pseudo-entry in the stall-diagnostic cost maps (plan
#: D-c/D-d): with master-hosted nodes, the master's own stand-alone
#: (autarky) cost and per-iteration native cost enter
#: :func:`_stall_worst_offenders` under this literal key — rendered as
#: "the master-hosted nodes", NEVER as a fake node-group name.
_MASTER_STALL_KEY = "master-hosted nodes"


def _raise_stalled(
    exc: BendersStalled,
    *,
    master_autarky: float | None = None,
    master_native_cost: float | None = None,
) -> NoReturn:
    """Render the coordinator's ``BendersStalled`` into the plain-English
    stall diagnostic, naming the worst-offender node group(s) exactly as the
    pre-coordinator loop did (the exception carries the bootstrap/autarky and
    stalled-iteration cost maps for :func:`_stall_worst_offenders`).

    With master-hosted nodes (``master_autarky`` not ``None``) the master
    joins the offender selection as a pseudo-entry keyed
    :data:`_MASTER_STALL_KEY` — its stand-alone (D-c autarky) cost against
    its last-iteration native cost — so "the master is the blown-up side"
    is expressible.  ``None`` (the empty-set path) injects nothing and the
    rendering is byte-identical to before."""
    reference_costs = dict(exc.sub_reference_costs)
    current_costs = dict(exc.sub_costs)
    if master_autarky is not None:
        reference_costs[_MASTER_STALL_KEY] = master_autarky
        current_costs[_MASTER_STALL_KEY] = (
            master_native_cost if master_native_cost is not None else 0.0
        )
    root, autarky_ratio, symptom, symptom_ratio = _stall_worst_offenders(
        reference_costs, current_costs
    )
    k = exc.window
    # Name the ROOT primarily; add the SYMPTOM only if it differs.  The
    # master pseudo-entry is rendered as "the master-hosted nodes" (its
    # key), never as a node-group name.
    root_disp = (
        "the master-hosted nodes" if root == _MASTER_STALL_KEY
        else repr(root)
    )
    if root == _MASTER_STALL_KEY:
        root_clause = (
            f"The master-hosted nodes are the likely cause — their "
            f"combined stand-alone cost is already {autarky_ratio:.0f}x "
            f"the next largest, i.e. they cannot meet their own demand "
            f"without flows from the node groups."
        )
    else:
        root_clause = (
            f"Node group {root!r} is the likely cause — its stand-alone "
            f"cost is already {autarky_ratio:.0f}x the next largest, i.e. it "
            f"cannot meet its own demand without imports."
        )
    if symptom == root:
        symptom_clause = ""
    elif symptom == _MASTER_STALL_KEY:
        symptom_clause = (
            f" At the stalled iteration the master-hosted nodes are the "
            f"ones forced worst into penalty/slack flow "
            f"({symptom_ratio:.0f}x their stand-alone cost)."
        )
    else:
        symptom_clause = (
            f" At the stalled iteration node group {symptom!r} is the "
            f"one forced worst into penalty/slack flow "
            f"({symptom_ratio:.0f}x its stand-alone cost)."
        )
    raise RuntimeError(_benders_failure_message(
        summary=(
            f"Benders stalled at iteration {exc.iteration}: the best "
            f"feasible cost has not improved for {k} iterations and the "
            f"relative gap is stuck at ~{exc.gap:.2f}, far from the {exc.tol} "
            "tolerance."
        ),
        meaning=(
            "The master keeps proposing node-group coupling flows that "
            "force one or more node groups into large penalty/slack "
            f"flow (recourse ~{symptom_ratio:.0f}x their stand-alone "
            "cost), so no improving feasible solution is found and the "
            f"bound cannot close. {root_clause}{symptom_clause}"
        ),
        how_to_avoid=(
            f"First, give the import/boundary nodes of {root_disp} a "
            "finite, moderate import price (penalty) a small multiple "
            "above the real marginal supply cost — an over-large penalty "
            "is what inflates the recourse and freezes the bound (any "
            "price above the true import cost gives the same optimum). "
            f"Then check {root_disp} in isolation for missing local "
            "capacity or imports, and rescale any extreme "
            "coupling-connection cost or capacity magnitudes. Only raise "
            "the iteration limit if the gap is still slowly improving "
            "(it is not here). If it persists, please report it with the "
            "model."
        ),
    )) from exc

def _assert_no_mixed_resolution_coupling(
    data: FlexData,
    splits,
    master_hosted_nodes: "frozenset[str] | set[str]",
) -> None:
    """Hard-error on a region↔master coupling arc whose two terminal
    nodes live on DIFFERENT time resolutions (one on aggregated
    ``new_stepduration`` blocks, the other on plain timesteps).

    The single-sided half-flow injection mirrors the original arc's
    cell-level terms per side, and for a MIXED-resolution arc the
    monolith's emit is asymmetric (the block side aggregates to the
    block-first cell while the fine side stays per-step), so the pinned
    region cells and the master's native cells no longer describe the
    same physical flow — the decomposition silently becomes a RELAXATION
    (measured on the lh2 fixture: the exactness gap, not a crash).
    Until the mirroring supports it, this is a loud unsupported-data
    error, never a silent degrade (plan D-a / R5 family).  The C9
    handover pattern side-steps it by design: the handover node MIRRORS
    the hosted node's time-aggregation group memberships (plan F10), so
    every boundary connection joins same-resolution nodes.

    Block membership is read from ``data.nodeStateBlock`` (the set of
    nodes whose balance/state lives on aggregated blocks).  Empty
    ``master_hosted_nodes`` returns immediately (byte-parity gate);
    region↔region arcs are untouched (both sides pinned — today's
    proven machinery).
    """
    if not master_hosted_nodes:
        return
    nsb = getattr(data, "nodeStateBlock", None)
    if nsb is None or nsb.height == 0:
        return
    block_nodes = set(nsb["n"].cast(pl.Utf8).to_list())
    seen: set[tuple] = set()
    for s in splits:
        for hf in s.half_flows:
            key = (hf.original_p, hf.original_source, hf.original_sink)
            if key in seen:
                continue
            seen.add(key)
            src, snk = hf.original_source, hf.original_sink
            # Single-sided (region↔master) arcs only: exactly one
            # terminal is master-hosted.
            if (src in master_hosted_nodes) == (snk in master_hosted_nodes):
                continue
            if (src in block_nodes) == (snk in block_nodes):
                continue
            coarse = src if src in block_nodes else snk
            fine = snk if coarse == src else src
            raise RuntimeError(_benders_failure_message(
                summary=(
                    f"Benders coupling connection {hf.original_p!r} "
                    f"joins nodes with different time resolutions "
                    f"across the master boundary: node {coarse!r} is "
                    f"on aggregated (multi-hour) time blocks while "
                    f"node {fine!r} is on plain timesteps."
                ),
                meaning=(
                    "A connection between a node group and a "
                    "master-hosted node is decomposed into a pinned "
                    "boundary flow, and that split is only exact when "
                    "both end nodes share the same time resolution. "
                    "With one end on aggregated time blocks, the "
                    "pinned per-timestep flows and the aggregated "
                    "block flows no longer describe the same physical "
                    "quantity, which silently distorts costs instead "
                    "of failing — so it is rejected up front."
                ),
                how_to_avoid=(
                    "Give both end nodes of the boundary connection "
                    "the same time resolution: put them in the same "
                    "time-aggregation (new_stepduration) group — the "
                    "recommended boundary pattern inserts a dedicated "
                    "handover node that mirrors the hosted node's "
                    "time-aggregation group memberships — or remove "
                    "the aggregation from one side."
                ),
            ))


def _assert_finite_boundary_penalties(
    data: FlexData,
    arcs: list[_ArcMaster],
    *,
    master_hosted_nodes: "frozenset[str] | set[str]" = frozenset(),
) -> None:
    """Optimality-cuts-only feasibility precondition: every boundary node
    (source/sink of a cross-region arc) must carry FINITE up/down slack
    penalties, so the recourse is always feasible.

    Extended (plan F9) to MASTER-HOSTED balance nodes: each must carry
    BOTH penalty params PRESENT and finite — with the coupling flows at
    zero (every early master iteration, and the D-c autarky reference
    solve, which literally pins them there) the master can only serve a
    hosted node's balance through priced slack, so a node whose penalty
    rows are MISSING is the same authored mistake as one whose penalty
    is infinite and gets the same plain-English error (the historical
    boundary-node branch above deliberately SKIPS nodes with no penalty
    rows — that skip must not carry over here)."""
    boundary_nodes = set()
    for a in arcs:
        _, s, k = a.key
        boundary_nodes.add(s)
        boundary_nodes.add(k)
    for pname in ("p_penalty_up", "p_penalty_down"):
        param = getattr(data, pname, None)
        if param is None:
            continue
        fr = param.frame
        if "n" not in fr.columns or "value" not in fr.columns:
            continue
        sub = fr.filter(pl.col("n").is_in(list(boundary_nodes)))
        if sub.height == 0:
            continue
        vals = sub["value"].to_numpy()
        if not np.all(np.isfinite(vals)):
            bad = sub.filter(~pl.col("value").is_finite())
            raise RuntimeError(
                f"Benders: non-finite {pname} on a boundary node — "
                f"optimality-cuts-only feasibility precondition violated:\n{bad}"
            )
    if not master_hosted_nodes:
        return
    balance_nodes: set[str] = set()
    if data.nodeBalance is not None and data.nodeBalance.height > 0:
        balance_nodes = set(
            data.nodeBalance["n"].cast(pl.Utf8).to_list()
        )
    schema_name = {"p_penalty_up": "penalty_up",
                   "p_penalty_down": "penalty_down"}
    for node in sorted(set(master_hosted_nodes) & balance_nodes):
        for pname in ("p_penalty_up", "p_penalty_down"):
            param = getattr(data, pname, None)
            sub = None
            if param is not None:
                fr = param.frame
                if "n" in fr.columns and "value" in fr.columns:
                    sub = fr.filter(pl.col("n").cast(pl.Utf8) == node)
            missing = sub is None or sub.height == 0
            if not missing and bool(
                np.all(np.isfinite(sub["value"].to_numpy()))
            ):
                continue
            problem = (
                "carries no value at all" if missing
                else "is not finite"
            )
            raise RuntimeError(_benders_failure_message(
                summary=(
                    f"Benders master-hosted node {node!r} has no usable "
                    f"{schema_name[pname]} penalty price — the value "
                    f"{problem}."
                ),
                meaning=(
                    "A balance node outside every node group is hosted "
                    "in the Benders master, and the master must be able "
                    "to serve or spill that node's balance with PRICED "
                    "slack whenever the coupling flows are still zero "
                    "(every early iteration, and the stand-alone "
                    "reference solve that pins them there). Both "
                    "penalty_up and penalty_down must therefore be "
                    "present and finite on the node; a missing or "
                    "infinite value would make those master solves "
                    "infeasible and fail later with a raw solver error "
                    "instead of this message."
                ),
                how_to_avoid=(
                    f"Author finite, moderate penalty_up and "
                    f"penalty_down prices on node {node!r} — a small "
                    "multiple of its real marginal supply cost is "
                    "enough (any price above the true cost gives the "
                    "same optimum). Alternatively, add the node to one "
                    "of the node groups so it is solved inside that "
                    "group's subproblem instead of the master."
                ),
            ))
