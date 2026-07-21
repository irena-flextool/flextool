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
    tol: float = _SHED_TOL_MWH,
) -> dict[str, float]:
    """Return ``{node: adder_increment}`` for every SHEDDING node.

    A node is shedding when its residual unserved energy exceeds *tol*;
    non-shedding nodes are skipped entirely (no key), so the loop bumps only
    the nodes that are actually short.  Each increment is
    ``scalar_adder(residual[node], W, lam)`` — the constant per-timestep
    adder that (undamped, ``lam=1``) would inject exactly that node's
    residual annual MWh back as demand.
    """
    if W <= 0.0:
        raise ValueError(f"W must be positive, got {W!r}")
    out: dict[str, float] = {}
    for node, res in residual.items():
        if res > tol:
            out[node] = scalar_adder(res, W, lam)
    return out


__all__ = [
    "invest_weight_W",
    "scalar_adder",
    "sized_increments",
    "w_from_grids",
]
