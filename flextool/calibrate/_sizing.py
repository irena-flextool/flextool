"""Adder sizing for the adequacy calibrator (C1b) — the numerically
load-bearing slice.

The engine param ``energy_margin_adder`` is a per-timestep MWh value
SUBTRACTED from a node's inflow in the invest solve, broadcast CONSTANT
over the invest ``(d, t)`` grid (P1).  ``node_slack_up_d_e`` reports the
resulting unserved energy as **annual MWh**, via
:func:`flextool.process_outputs._annualize.annualize_dt_to_d`::

    annual(d) = ( Σ_t value(d, t) · timestep_weight(d, t) ) / period_share(d)

A CONSTANT per-timestep adder ``a`` (MWh/step) therefore adds annual demand
``ΔE = a · W`` where

    W = Σ_{d ∈ invest periods} ( Σ_t timestep_weight(d, t) ) / period_share(d).

So to inject ``X`` MWh of annual demand the scalar adder is ``a = X / W``.
The calibrator's per-node target is ``X = λ · residual_unserved(node)``, and
hence ``increment_adder(node) = λ · residual(node) / W``.

Why ``W`` and not the ``/n_invest_steps`` shorthand
---------------------------------------------------
``/n_invest_steps`` is only correct for a full-year, weight≡1 timeline.  For
the representative-period timelines the calibrator TARGETS, ``period_share``
is far below 1 (e.g. a 72-step year fraction of ``72/8760``), so the true
divisor ``W`` is ``8760`` per period, not ``72`` — the naive divisor would
overshoot the first correction by the rep-weight factor (~121× here).

The W source (why it is robust)
-------------------------------
``W`` is a fixed property of the scenario's invest timeline, computed ONCE
per run straight from the input DB by REUSING the engine's own per-solve
derivations — the same code paths preprocessing uses — never the
``--csv-dump`` files (``solve_data/*.csv``), which are gated AND overwritten
by the LAST (dispatch) sub-solve and so do NOT reflect the invest grid:

* the invest ``(d, t)`` grid and per-period
  ``complete_period_share_of_year`` come from
  :func:`flextool.engine_polars._per_solve_sets.derive_per_solve_aggregates`;
* ``Σ_t timestep_weight(d, t)`` per period comes from the ACTUAL per-``(d,
  t)`` weights the engine builds, NOT a step-count shortcut —
  :func:`flextool.engine_polars._derived_params.p_timestep_weight_from_source`
  for the default / ``timeset_weights`` regimes, and
  :func:`flextool.engine_polars._emit_solve_writers._compute_rp_frames` for
  ``representative_period_weights`` (RP).

Why the ACTUAL weight, not the step count ``n_d``
-------------------------------------------------
``annualize_dt_to_d`` weights every ``(d, t)`` by ``p_timestep_weight``, and
RP weights flow into ``p_timestep_weight`` (``_compute_rp_frames`` folds
``representative_period_weights`` into ``timestep_weight.csv``; the CSV
loader puts it in ``p_timestep_weight``; ``out_node.py`` annualises
``node_slack_up_d_e`` with it) — so ``W`` MUST use the same weights.  For the
default and ``timeset_weights`` regimes the weights normalise to ``Σ_t = n_d``
(the step count), so those reduce to ``n_d``; but a general RP timeset with
UNEQUAL rep-block lengths or un-normalised weights does NOT, so ``W`` reads
the real ``Σ_t timestep_weight`` and is correct for every regime.
"""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from flextool.engine_polars._derived_params import (
    p_timestep_weight_from_source,
)
from flextool.engine_polars._emit_solve_writers import _compute_rp_frames
from flextool.engine_polars._per_solve_sets import derive_per_solve_aggregates
from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._spinedb_reader import SpineDbReader
from flextool.engine_polars._timeline import TimelineConfig

# Per-node absolute shed tolerance (MWh).  Nodes whose residual unserved
# energy is at/below this are treated as non-shedding and get NO increment,
# so numerical dust never seeds a spurious adder.
_SHED_TOL_MWH = 1e-6


def _normalise_url(url: str) -> str:
    """Promote a bare filesystem path to a ``sqlite:///`` URL; pass through
    anything already carrying a ``"://"`` scheme."""
    return url if "://" in url else f"sqlite:///{url}"


def _invest_solves(sc: "SolveConfig") -> list[str]:
    """Return the model's top-level solve names — the invest solves.

    ``model.solves`` (``SolveConfig.model_solve``) lists the solves the model
    runs; in a nested-invest model the top-level solve IS the invest solve
    (it CONTAINS the dispatch sub-solves), and in a flat single-solve model
    that one solve is both invest and dispatch.  Either way the top-level
    solve carries the ``(d, t)`` grid the ``energy_margin_adder`` is
    broadcast over, which is exactly the grid ``W`` must be measured on.
    """
    solves: list[str] = []
    for solve_list in sc.model_solve.values():
        for s in solve_list:
            if s not in solves:
                solves.append(s)
    return solves


def w_from_grids(
    weight_by_period: dict[str, int | float],
    share_by_period: dict[str, float],
) -> float:
    """Return ``W = Σ_d weight_d / share_d`` from per-period weight-sums + shares.

    ``weight_by_period[d]`` is the ACTUAL ``Σ_t timestep_weight(d, t)`` and
    ``share_by_period[d]`` is ``complete_period_share_of_year(d)``.  Pure
    arithmetic — the DB-reading :func:`invest_weight_W` builds the two dicts
    and calls this.

    Reductions the tests pin: with ``weight ≡ 1`` and ``share ≡ 1``
    (full-year, evenly-sampled) ``W`` collapses to the total step count
    ``Σ_d n_d``; with NON-unit weights (a general RP grid) ``W`` uses
    ``Σ_t weight``, NOT the step count.
    """
    total = 0.0
    for d, w in weight_by_period.items():
        share = share_by_period[d]
        if share <= 0.0:
            raise ValueError(
                f"period {d!r}: non-positive period_share {share!r} — cannot "
                "annualise."
            )
        total += float(w) / float(share)
    return total


def _rp_weight_sum_by_period(
    tc: "TimelineConfig", period: str, timeset: str,
) -> dict[str, float]:
    """Actual ``Σ_t timestep_weight`` per period for one RP *timeset*.

    Reuses the engine writer :func:`._emit_solve_writers._compute_rp_frames`
    — the single source of truth that folds
    ``representative_period_weights`` into ``timestep_weight.csv`` — so the
    weights match those the annualiser applies to ``node_slack_up_d_e``
    byte-for-byte.  The per-``(d, t)`` weight is independent of the RP chain
    TOPOLOGY (``within_solve`` vs ``within_period`` produce identical
    ``timestep_weight`` rows; the ``within_period`` writer itself calls this
    same single-timeset path per period), so calling it once per active RP
    timeset and summing is correct for both.
    """
    timeline_name = tc.timesets__timeline[timeset]
    timeline_steps = [step for step, _dur in tc.timelines[timeline_name]]
    frames = _compute_rp_frames(
        tc.rp_weights[timeset],
        tc.timeset_durations[timeset],
        period,
        timeline_steps,
    )
    tw = frames["timestep_weight.csv"].with_columns(
        pl.col("weight").cast(pl.Float64)
    )
    return {
        str(p): float(w)
        for p, w in tw.group_by("period").agg(pl.col("weight").sum()).iter_rows()
    }


def _weight_sum_by_period(
    source: "SpineDbReader",
    sc: "SolveConfig",
    tc: "TimelineConfig",
    solve: str,
    dt_complete: "pl.DataFrame",
) -> dict[str, float]:
    """Return ``{period: Σ_t timestep_weight(d, t)}`` for *solve*.

    Uses the ACTUAL engine-built weights for every regime:

    * **default / ``timeset_weights``** — the native
      :func:`._derived_params.p_timestep_weight_from_source` (returns dense
      1.0 or the normalised ``timeset_weights``), summed per period;
    * **``representative_period_weights`` (RP)** — that helper returns
      ``None`` (RP weights live in the CSV the writer folds), so each active
      RP timeset is folded via :func:`_rp_weight_sum_by_period`.  Any non-RP
      period sharing an RP solve falls back to its step count ``n_d`` (which
      the non-RP normalisation makes equal to ``Σ_t weight``).
    """
    active = sc.timesets_used_by_solves.get(solve, [])
    rp_present = any(ts in tc.rp_weights for _period, ts in active)

    if not rp_present:
        param = p_timestep_weight_from_source(source, dt_complete, solve)
        if param is not None and param.frame.height > 0:
            agg = param.frame.group_by("d").agg(pl.col("value").sum())
            return {str(d): float(v) for d, v in agg.iter_rows()}
        # No period_timeset / weights resolvable → default 1.0 ⇒ Σ = n_d.
        return {
            str(d): float(n)
            for d, n in dt_complete.group_by("d").len().iter_rows()
        }

    # At least one active timeset is RP.
    n_by_d = {
        str(d): float(n)
        for d, n in dt_complete.group_by("d").len().iter_rows()
    }
    out: dict[str, float] = {}
    for period, ts in active:
        period = str(period)
        if ts in tc.rp_weights:
            for p, w in _rp_weight_sum_by_period(tc, period, ts).items():
                out[p] = out.get(p, 0.0) + w
        else:
            out[period] = out.get(period, 0.0) + n_by_d.get(period, 0.0)
    return out


def invest_weight_W(url: str, scenario: str) -> float:
    """Compute the annualisation weight ``W`` for *scenario*'s invest solve.

    ``W = Σ_{d ∈ invest periods} (Σ_t timestep_weight(d, t)) / period_share(d)``
    using the ACTUAL per-``(d, t)`` ``timestep_weight`` the engine builds
    (default / ``timeset_weights`` / ``representative_period_weights``), so a
    constant per-timestep adder ``a`` injects annual demand ``a · W`` for
    every weighting regime; see the module docstring.

    Computed ONCE per calibration run (``W`` is independent of the adder).
    Reuses the engine's own per-solve derivations so it is correct-by-
    construction against the running engine version, never the gated /
    dispatch-overwritten ``solve_data`` CSVs.

    Raises
    ------
    ValueError
        If no invest solve / grid can be resolved, or ``W`` is non-positive.
    """
    url = _normalise_url(url)
    sc = SolveConfig.load_from_db_url(url, scenario)
    invest_solves = _invest_solves(sc)
    if not invest_solves:
        raise ValueError(
            f"scenario {scenario!r}: no model.solves — cannot resolve the "
            "invest solve to size the adder against."
        )

    source = SpineDbReader(url, scenario)
    tc = TimelineConfig.load_from_db_url(url, scenario)

    total = 0.0
    for solve in invest_solves:
        agg = derive_per_solve_aggregates(source, solve)
        if agg is None:
            raise ValueError(
                f"scenario {scenario!r}, solve {solve!r}: could not derive the "
                "per-solve (d, t) grid / period share from the DB "
                "(missing solve.period_timeset / timeline.timestep_duration). "
                "W cannot be computed."
            )
        # Σ_t timestep_weight per period (actual engine weights, all regimes).
        weight_by_d = _weight_sum_by_period(source, sc, tc, solve, agg.dt_complete)
        share_by_d = {
            str(d): float(v)
            for d, v in agg.complete_period_share_of_year.select(
                "d", "value"
            ).iter_rows()
        }
        # Restrict to the periods that actually have a share (the grid).
        weight_by_d = {d: weight_by_d[d] for d in share_by_d if d in weight_by_d}
        total += w_from_grids(weight_by_d, share_by_d)

    if total <= 0.0:
        raise ValueError(
            f"scenario {scenario!r}: computed W={total!r} is non-positive."
        )
    return total


def scalar_adder(residual_mwh: float, W: float, lam: float) -> float:
    """Return the per-timestep adder that injects ``lam · residual`` annual MWh.

    ``a = lam · residual_mwh / W`` (the exact inverse of ``ΔE = a · W``).
    ``W`` must be positive (a valid invest-timeline annualiser).
    """
    if W <= 0.0:
        raise ValueError(f"W must be positive, got {W!r}")
    return lam * residual_mwh / W


def sized_increments(
    residual: dict[str, float],
    *,
    W: float,
    lam: float,
    overshoot: float = 1.0,
    tol: float = _SHED_TOL_MWH,
) -> dict[str, float]:
    """Return ``{node: adder_increment}`` for every SHEDDING node.

    A node is shedding when its residual unserved energy exceeds *tol*;
    non-shedding nodes are skipped entirely (no key), so the loop bumps only
    the nodes that are actually short.  Each increment is
    ``overshoot · scalar_adder(residual[node], W, lam)`` — the constant
    per-timestep adder that (undamped, ``lam=1``, ``overshoot=1``) would
    inject exactly that node's residual annual MWh back as demand.

    ``overshoot`` (default ``1.0`` = off) is a ``>1`` planning-margin SAFETY
    multiplier: a single-year (or single-year RP) solve under-estimates true
    multi-year severity, so ``overshoot`` deliberately provisions beyond the
    measured slack (``overshoot=1.2`` ⇒ ~20 % extra headroom).  The right
    value is MODEL-DEPENDENT.
    """
    if W <= 0.0:
        raise ValueError(f"W must be positive, got {W!r}")
    out: dict[str, float] = {}
    for node, res in residual.items():
        if res > tol:
            out[node] = overshoot * scalar_adder(res, W, lam)
    return out


# ===========================================================================
# T2 — the ``timed`` sizer: place the additive margin at the low-VRE stress
# hours (per-timestep) instead of spreading it flat.
# ===========================================================================
#
# The uniform sizer above injects ``ΔE = λ·residual`` annual MWh as a CONSTANT
# per-timestep adder ``a = λ·residual/W``.  The ``timed`` sizer injects the
# SAME total energy but distributes it by the stress SHAPE — the per-cell
# ``node_slack_up_dt_e`` profile — so the demand lands exactly at the hours the
# invest solve could not serve.
#
# Timeline fold (base → representative)
# -------------------------------------
# ``node_slack_up_dt_e`` is the invest-solve up-slack UNFOLDED onto the full
# base timeline.  For a representative-period (RP) timeset every base block
# ``b`` decomposes convexly over representative blocks ``rep`` with hull weights
# ``rp_weights[timeset][base_start][rep_start]`` (``Σ_rep weight(b→rep)=1`` per
# base block — the SAME source ``invest_weight_W`` reuses).  For a real hour =
# (base block ``b``, within-block offset ``h``) carrying slack ``e(b, h)`` we
# fold it back onto the representative cell that shares its offset::
#
#     slack_rep(rep, h) = Σ_b weight(b→rep) · e(b, h)
#
# Total is conserved: ``Σ_rep slack_rep = Σ_b e(b,·) = residual`` (because
# ``Σ_rep weight(b→rep)=1``), so the folded stress carries the node's full
# residual — never the ~15 % that a raw subset of the invest-grid timestamps
# would (those base rows are convex combinations that sum to a fraction of the
# residual).
#
# Sizing (SHAPE from the fold, MAGNITUDE from the true annual residual)
# ---------------------------------------------------------------------
# The fold gives the per-cell stress SHAPE, but its raw magnitude is NOT a
# reliable proxy for the node's annual residual: ``node_slack_up_dt_e`` equals
# the annual ``node_slack_up_d_e`` only when the dt table is UNFOLDED onto the
# full base timeline (as it is for the H2 model, dt≈annual).  On a model whose
# dt table stays on the REPRESENTATIVE grid, ``Σ_dt slack ≠ annual residual``
# (measured ~0.10 on one RP model), so folding the raw dt total would
# under-inject ~10×.  We therefore use the fold only for the shape and
# NORMALISE it to the true annual residual ``res = node_slack_up_d_e[node]``.
#
# With the ENGINE per-cell timestep weight ``tw_rep(rep, h)`` (from
# ``_compute_rp_frames``'s ``timestep_weight.csv`` — the same weights the
# annualiser applies) and the folded shape ``slack_rep(rep, h)`` summing to
# ``S = Σ_cell slack_rep`` over the cells we can inject on::
#
#     adder(rep, h) = overshoot · λ · res · (slack_rep(rep, h) / S) / tw_rep(rep, h)
#
# The annualised injected energy is then EXACT and independent of both the
# dt-table form and ``tw_rep`` (it CANCELS in the weighted sum)::
#
#     Σ_cell adder·tw_rep = overshoot · λ · res · (Σ_cell slack_rep / S)
#                         = overshoot · λ · res
#
# — the SAME total energy the uniform sizer injects (``overshoot·λ·res``),
# placed at the stressed cells, correct whether the dt table is pre-unfolded
# (``S == res`` ⇒ ``adder = overshoot·λ·slack_rep/tw_rep``, unchanged from the
# naive fold) or on the representative grid (``S ≠ res`` ⇒ the shape is scaled
# up to the annual magnitude).  ``λ`` is ``damping_first`` on the first
# correction else ``damping_remaining`` (identical to uniform); ``overshoot``
# is the same ``>1`` planning-margin safety multiplier the uniform sizer uses.
#
# Non-RP invest solve
# -------------------
# When the invest solve is NOT representative-period the base timeline IS the
# invest grid; the fold degenerates to identity (each cell maps to itself with
# weight 1) and ``tw_rep`` is the per-cell ``p_timestep_weight`` (dense 1.0 or
# the normalised ``timeset_weights``).  The formula then places the adder at
# each timestep in proportion to that timestep's own slack, normalised to the
# annual residual — the sensible single-representative reduction.


def size_timed(
    dt_slack: dict[str, dict[tuple[str, str], float]],
    residual: dict[str, float],
    fold_edges: list[tuple[str, str, str, float]],
    tw_rep: dict[tuple[str, str], float],
    *,
    lam: float,
    overshoot: float = 1.0,
    tol: float = _SHED_TOL_MWH,
) -> dict[str, dict[tuple[str, str], float]]:
    """Pure timed sizer — fold + normalised per-cell sizing, no DB / no solver.

    The fold supplies only the per-cell stress SHAPE; the injected MAGNITUDE
    is taken from the TRUE annual residual *residual[node]* (see the module's
    T2 comment).  Each shedding node's folded shape is normalised so the
    annualised injected energy equals ``overshoot · λ · residual[node]`` —
    correct whether ``dt_slack`` was pre-unfolded onto the base timeline
    (``Σ shape == residual``, so the per-cell adder is unchanged from the
    naive fold) or left on the representative grid (``Σ shape ≠ residual``, so
    the shape is scaled to the annual magnitude).

    Parameters
    ----------
    dt_slack:
        ``{node: {(period, time): slack}}`` — the base-timeline up-slack
        profile (``read_residual_unserved_dt``, converted to a lookup).  Used
        for SHAPE only; its total need not equal the annual residual.
    residual:
        ``{node: annual_MWh}`` — the per-node TRUE annual residual
        (``node_slack_up_d_e``); a node is SHEDDING (and thus sized) only when
        it exceeds *tol*, matching the uniform :func:`sized_increments` gate,
        and it also sets the injected magnitude the shape is normalised to.
    fold_edges:
        ``[(period, base_time, rep_time, weight), ...]`` — the sparse fold
        operator: each edge sends ``weight · e(period, base_time)`` onto the
        representative cell ``(period, rep_time)``.  For a total-conserving
        fold every base cell's out-edge weights sum to 1.
    tw_rep:
        ``{(period, rep_time): timestep_weight}`` — the engine per-cell weight
        the adder is divided by (and the annualiser multiplies back).
    lam:
        Damping factor λ.
    overshoot:
        Planning-margin safety multiplier (default ``1.0`` = off); ``>1``
        provisions beyond the measured slack, exactly as in the uniform sizer.

    Returns
    -------
    ``{node: {(period, rep_time): adder}}`` for every shedding node.  Cells
    with zero folded slack (or a non-positive ``tw_rep``) are omitted, so the
    map carries only the stressed representative cells, and
    ``Σ_cell adder·tw_rep == overshoot · λ · residual[node]`` over the kept
    cells.
    """
    out: dict[str, dict[tuple[str, str], float]] = {}
    for node, res in residual.items():
        if res <= tol:
            continue
        node_slack = dt_slack.get(node)
        if not node_slack:
            continue
        # Fold the base-timeline slack onto the representative cells → the
        # per-cell stress SHAPE (its raw magnitude is unreliable; see below).
        slack_rep: dict[tuple[str, str], float] = defaultdict(float)
        for period, base_time, rep_time, weight in fold_edges:
            e = node_slack.get((period, base_time))
            if e:
                slack_rep[(period, rep_time)] += weight * e
        # Keep only the cells we can actually inject on (positive folded slack,
        # positive engine weight) BEFORE normalising, so the annualised
        # injected total is EXACTLY overshoot·λ·res over the kept cells.
        valid: dict[tuple[str, str], tuple[float, float]] = {}
        for cell, sr in slack_rep.items():
            if sr <= 0.0:
                continue
            tw = tw_rep.get(cell)
            if tw is None or tw <= 0.0:
                continue
            valid[cell] = (sr, tw)
        shape_total = sum(sr for sr, _tw in valid.values())
        if shape_total <= 0.0:
            continue
        # NORMALISE the shape to the true annual residual: distribute
        # overshoot·λ·res over the cells by their shape fraction, then divide
        # by tw_rep so the annualiser recovers exactly that energy.
        scale = overshoot * lam * res / shape_total
        adder = {cell: scale * sr / tw for cell, (sr, tw) in valid.items()}
        if adder:
            out[node] = adder
    return out


def _timeline_steps_and_index(
    tc: "TimelineConfig", timeset: str,
) -> tuple[list[str], dict[str, int]]:
    """Ordered timeline steps + ``{step: idx}`` for *timeset*'s timeline."""
    timeline_name = tc.timesets__timeline[timeset]
    steps = [step for step, _dur in tc.timelines[timeline_name]]
    return steps, {s: i for i, s in enumerate(steps)}


def _rp_fold_edges_and_tw(
    tc: "TimelineConfig", period: str, timeset: str,
) -> tuple[list[tuple[str, str, str, float]], dict[tuple[str, str], float]]:
    """Build the RP fold edges + per-cell ``tw_rep`` for one (period, timeset).

    ``edges`` maps every base-timeline cell ``(period, base_time)`` onto the
    representative cell ``(period, rep_time)`` that shares its within-block
    offset ``h``, weighted by the hull weight ``rp_weights[base][rep]``.
    ``tw_rep`` comes straight from the engine writer's ``timestep_weight.csv``
    (:func:`._emit_solve_writers._compute_rp_frames`) so the sizer divides by
    exactly the weight the annualiser multiplies back.

    Two representative-period regimes are handled:

    * **Hull / equal-length blocks** (the ``timed`` sizer's target, e.g.
      ``hull_5rp_168h``): many base blocks each the SAME length as the
      representative blocks, and ``node_slack_up_dt_e`` is unfolded onto the
      FULL base timeline.  The offset-preserving fold above applies.
    * **``representative_period_weights``**: a base period REPRESENTED by a
      few weighted sub-blocks (base-block length ≫ rep-block length), where the
      invest slack already lives on the representative grid.  There is nothing
      to unfold — the fold degenerates to identity on the representative cells
      (each rep cell maps to itself, weight 1), which the total-conservation
      contract (Σ out-weight per source cell = 1) still satisfies.

    The regime is decided by block-length alignment; a mismatch is NOT an
    error (it is the second regime), so no valid RP model is ever rejected.
    """
    steps, idx = _timeline_steps_and_index(tc, timeset)
    rpw = tc.rp_weights[timeset]  # {base_start: {rep_start: weight}}

    # Representative block ranges, anchored to real timeline steps.
    rep_count: dict[str, int] = {}
    for start, count in tc.timeset_durations[timeset]:
        rep_count[str(start)] = int(float(count))

    frames = _compute_rp_frames(
        rpw, tc.timeset_durations[timeset], period, steps,
    )
    tw = frames["timestep_weight.csv"].with_columns(
        pl.col("weight").cast(pl.Float64)
    )
    tw_rep = {(str(p), str(t)): float(w) for p, t, w in tw.iter_rows()}

    # Base blocks TILE the timeline: sort the rp_weights base starts by
    # timeline position; each base block spans from its start to the next
    # base start (the last runs to the timeline end).  Deriving the length
    # from the tiling handles a short trailing block.
    base_starts = sorted(rpw.keys(), key=lambda s: idx.get(s, len(steps)))
    base_ranges: list[tuple[str, int, int]] = []  # (base_start, start_idx, count)
    for i, bs in enumerate(base_starts):
        if bs not in idx:
            raise ValueError(
                f"RP fold: base block start {bs!r} not in timeline "
                f"(period={period!r}, timeset={timeset!r})."
            )
        bstart_idx = idx[bs]
        bend_idx = idx[base_starts[i + 1]] if i + 1 < len(base_starts) else len(steps)
        base_ranges.append((bs, bstart_idx, bend_idx - bstart_idx))

    # Aligned iff every base block is no longer than every rep block it maps to
    # (offset h has a representative counterpart).  Otherwise this is the
    # representative_period_weights regime → identity fold on the rep grid.
    aligned = all(
        bcount <= rep_count.get(str(rep_start), 0)
        for _bs, _si, bcount in base_ranges
        for rep_start, w in rpw[_bs].items()
        if w > 1e-12
    )

    if not aligned:
        # Identity on the representative cells: the invest slack is already
        # representative, so each rep cell maps to itself with weight 1.
        edges = [(str(p), str(t), str(t), 1.0) for (p, t) in tw_rep]
        return edges, tw_rep

    edges: list[tuple[str, str, str, float]] = []
    for bs, bstart_idx, bcount in base_ranges:
        for rep_start, weight in rpw[bs].items():
            if weight <= 1e-12:
                continue
            r_idx = idx.get(rep_start)
            if r_idx is None:
                raise ValueError(
                    f"RP fold: representative block start {rep_start!r} not in "
                    f"timeline (period={period!r}, timeset={timeset!r})."
                )
            for h in range(bcount):
                edges.append(
                    (period, steps[bstart_idx + h], steps[r_idx + h], float(weight))
                )
    return edges, tw_rep


def _identity_edges_and_tw(
    source: "SpineDbReader",
    sc: "SolveConfig",
    solve: str,
    dt_complete: "pl.DataFrame",
    periods: set[str] | None = None,
) -> tuple[list[tuple[str, str, str, float]], dict[tuple[str, str], float]]:
    """Identity fold (non-RP): each invest cell maps to itself, weight 1.

    ``tw_rep`` is the per-cell ``p_timestep_weight`` (dense 1.0 or the
    normalised ``timeset_weights``); *periods*, when given, restricts the grid
    to the non-RP periods of a mixed solve.
    """
    grid = dt_complete
    if periods is not None:
        grid = grid.filter(pl.col("d").cast(pl.Utf8).is_in(list(periods)))
    edges = [
        (str(d), str(t), str(t), 1.0)
        for d, t in grid.select("d", "t").iter_rows()
    ]
    param = p_timestep_weight_from_source(source, dt_complete, solve)
    tw_rep: dict[tuple[str, str], float] = {}
    if param is not None and param.frame.height > 0:
        for d, t, v in param.frame.select("d", "t", "value").iter_rows():
            tw_rep[(str(d), str(t))] = float(v)
    # Any cell without an explicit weight defaults to the trivial 1.0.
    for period, base_time, _rt, _w in edges:
        tw_rep.setdefault((period, base_time), 1.0)
    return edges, tw_rep


def timed_increments(
    residual: dict[str, float],
    dt_slack: dict[str, dict[tuple[str, str], float]],
    url: str,
    scenario: str,
    *,
    lam: float,
    overshoot: float = 1.0,
    tol: float = _SHED_TOL_MWH,
) -> dict[str, dict[tuple[str, str], float]]:
    """Return ``{node: {(period, time): adder}}`` for every SHEDDING node.

    Assembles the RP (or identity) fold for *scenario*'s invest solve(s) from
    the DB and applies the pure :func:`size_timed`.  Each shedding node's
    base-timeline slack supplies the stress SHAPE, normalised to the node's
    TRUE annual residual and converted to a per-cell ``energy_margin_adder``
    that (weighted by ``tw_rep``) injects exactly ``overshoot · λ · residual``
    annual MWh — the same total as uniform, placed at the stressed hours.
    ``overshoot`` (default ``1.0``) is the planning-margin safety multiplier.

    The fold reuses the engine's own rep-weight machinery
    (``rp_weights`` + ``_compute_rp_frames``), never the on-disk
    ``timeline_matching_map.csv`` (the wrong, unreliable artifact).
    """
    url = _normalise_url(url)
    sc = SolveConfig.load_from_db_url(url, scenario)
    tc = TimelineConfig.load_from_db_url(url, scenario)
    source = SpineDbReader(url, scenario)
    invest_solves = _invest_solves(sc)
    if not invest_solves:
        raise ValueError(
            f"scenario {scenario!r}: no model.solves — cannot resolve the "
            "invest solve for timed sizing."
        )

    all_edges: list[tuple[str, str, str, float]] = []
    tw_rep: dict[tuple[str, str], float] = {}
    for solve in invest_solves:
        active = sc.timesets_used_by_solves.get(solve, [])
        rp_pairs = [(str(p), ts) for p, ts in active if ts in tc.rp_weights]
        nonrp_periods = {str(p) for p, ts in active if ts not in tc.rp_weights}
        if rp_pairs:
            for period, ts in rp_pairs:
                edges, tw = _rp_fold_edges_and_tw(tc, period, ts)
                all_edges.extend(edges)
                tw_rep.update(tw)
            # Mixed solve: any non-RP period folds through the identity path.
            if nonrp_periods:
                agg = derive_per_solve_aggregates(source, solve)
                if agg is not None:
                    edges, tw = _identity_edges_and_tw(
                        source, sc, solve, agg.dt_complete, nonrp_periods,
                    )
                    all_edges.extend(edges)
                    tw_rep.update(tw)
        else:
            agg = derive_per_solve_aggregates(source, solve)
            if agg is None:
                raise ValueError(
                    f"scenario {scenario!r}, solve {solve!r}: could not derive "
                    "the invest (d, t) grid for identity-fold timed sizing."
                )
            edges, tw = _identity_edges_and_tw(source, sc, solve, agg.dt_complete)
            all_edges.extend(edges)
            tw_rep.update(tw)

    if not all_edges:
        raise ValueError(
            f"scenario {scenario!r}: timed sizing resolved no fold cells from "
            "the invest solve(s)."
        )
    return size_timed(
        dt_slack, residual, all_edges, tw_rep,
        lam=lam, overshoot=overshoot, tol=tol,
    )


__all__ = [
    "invest_weight_W",
    "scalar_adder",
    "size_timed",
    "sized_increments",
    "timed_increments",
    "w_from_grids",
]
