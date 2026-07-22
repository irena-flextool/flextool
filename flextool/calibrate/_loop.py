"""Driver skeleton for the adequacy-margin calibrator (C1a).

This is the loop that COMPOSES P1's ``energy_margin_adder`` knob and P2's
solve-success detector into an iterate-until-adequate cycle, plus the pure
readers of :mod:`flextool.calibrate._readers`.  It runs end to end today —
solve, verify, read the per-node residual unserved energy — but it does
**not** yet size the adder or guard against over-build.  Those are C1b
(sizing) and C1c (the over-build guard); this slice leaves a clean,
signature-stable seam for them in :func:`compute_step`.

Loop shape (per iteration ``k`` in ``range(iterations + 1)``; ``k=0`` is the
BASELINE)::

    1. write_calib_alt(url, scenario, adders)     # k=0 writes empty/zero
    2. clear (or, in debug, archive) the prior output_parquet
    3. run = run_solve(...);  outcome = assess_solve(run..., started_at=...)
       -> not outcome.succeeded  ⇒  raise CalibError   (fail-closed)
    4. residual  = read_residual_unserved(run.assess_dir)
       curtailment, penalty read too; record the iteration
    5. total_unserved <= slack_threshold_mwh  ⇒  converged, break
    6. increments = compute_step(..., W=W)        # C1b sizing (no guard)
       adders[node] += increment                  # bumps shedding nodes

The solve is *fail-closed*: an unverified solve (missing/stale/empty
required outputs, or an unoverridable nonzero exit) raises rather than
letting the calibrator step on numbers it cannot trust.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from flextool.calibrate._db_alt import calib_alt_name, write_calib_alt
from flextool.calibrate._guard import guard_freeze
from flextool.calibrate._readers import (
    read_curtailment_by_sink,
    read_residual_unserved,
    read_residual_unserved_dt,
    read_slack_penalty,
)
from flextool.calibrate._sizing import (
    _SHED_TOL_MWH,
    invest_weight_W,
    sized_increments,
    timed_increments,
)
from flextool.calibrate._solve import run_solve
from flextool.calibrate._solve_status import assess_solve


class CalibError(RuntimeError):
    """Raised when an iteration's solve cannot be trusted (fail-closed).

    Carries the solve-success detector's ``reason`` so the operator sees
    exactly why the run was rejected (missing/stale outputs, unoverridable
    nonzero exit, …) instead of a bare failure.
    """


@dataclass
class CalibConfig:
    """Configuration for a calibration run.

    ``iterations``          — number of ADJUSTMENT iterations after the
                              baseline; the loop runs ``iterations + 1``
                              solves (iteration 0 is the baseline).
    ``slack_threshold_mwh`` — total residual unserved energy at or below
                              which the run is considered converged.
    ``damping_first`` /
    ``damping_remaining``   — sizing damping factors (consumed by C1b).
    ``overshoot``           — planning-margin SAFETY multiplier on every sized
                              increment (default 1.0 = off; >1 provisions
                              beyond the measured slack for unmodeled
                              multi-year risk; MODEL-DEPENDENT).
    ``stall_fraction``      — over-build-guard STALL fraction (C1c): a shedding
                              node whose residual drops by less than this
                              fraction of its prior gap in response to its bump
                              is frozen as resource-capped.  Higher freezes
                              sooner.  Default 0.05.
    ``over_build_tightness``— RETAINED for CLI/config compatibility only; the
                              C1c guard no longer gates on curtailment
                              efficiency, so this value is not consulted (the
                              freeze is driven by ``stall_fraction``).
    ``warm_start_cache_dir``— stable basis-cache dir shared across
                              iterations.
    ``work_dir``            — subprocess working directory.
    ``out_root``            — output-location root; results land under
                              ``out_root/output_parquet/<scenario>/``.
    ``debug``               — when True, archive each iteration's outputs to
                              ``out_root/out_iter_<k>/`` instead of clearing
                              them.
    ``sizing``              — adder placement mode: ``"uniform"`` (default) —
                              a constant per-timestep margin sized ``λ·res/W``;
                              or ``"timed"`` — the same total energy placed
                              per-cell at the low-VRE stress hours (folded from
                              ``node_slack_up_dt_e``).
    """

    iterations: int
    slack_threshold_mwh: float
    damping_first: float
    damping_remaining: float
    over_build_tightness: float
    warm_start_cache_dir: Path
    work_dir: Path
    out_root: Path
    debug: bool = False
    sizing: str = "uniform"
    overshoot: float = 1.0
    stall_fraction: float = 0.05


@dataclass
class IterRecord:
    """One iteration's observed state (the calibration trajectory element).

    ``adders`` is the per-node adder snapshot that was WRITTEN and SOLVED
    for this iteration (captured before any increment is applied), so the
    trajectory pairs each observation with the input that produced it.  For
    ``uniform`` sizing each value is a scalar float; for ``timed`` sizing it
    is a ``{(period, time): float}`` per-cell map.

    ``solve_seconds`` is this iteration's wall-clock solve time (end −
    ``started_at``), so the report surfaces the per-iteration cost (and the
    warm-start speedup across iterations) directly.
    """

    iteration: int
    adders: "dict[str, float | dict[tuple[str, str], float]]"
    residual: dict[str, float]
    curtailment: dict[str, float]
    penalty_total: float
    penalty_by_node: dict[str, float]
    solve_seconds: float = 0.0

    @property
    def total_unserved(self) -> float:
        """Total residual unserved energy (MWh) across all nodes."""
        return float(sum(self.residual.values()))


@dataclass
class CalibResult:
    """Outcome of a calibration run.

    ``converged``           — whether total unserved fell to/under the
                              threshold within the iteration budget.
    ``stop_reason``         — the finer three-way signal of WHY the loop
                              stopped: ``"converged"`` (total unserved met
                              the threshold), ``"stalled"`` (no further bump
                              was possible — every remaining shedding node is
                              resource-capped, so the demand-margin lever is
                              exhausted — while still above threshold), or
                              ``"budget_exhausted"`` (ran the full
                              ``iterations`` budget without converging or
                              stalling).  ``converged`` stays ``True`` only
                              for the threshold case, so downstream flags keep
                              working; ``stop_reason`` distinguishes the two
                              non-converged exits.
    ``iterations_run``      — number of solves actually performed.
    ``final_adders``        — the per-node adders after the last step.
    ``trajectory``          — per-iteration :class:`IterRecord` list.
    ``guard_flagged_nodes`` — nodes the over-build guard flagged (C1c;
                              empty for now).
    """

    converged: bool
    iterations_run: int
    final_adders: dict[str, float]
    trajectory: list[IterRecord]
    guard_flagged_nodes: list[str] = field(default_factory=list)
    stop_reason: str = "budget_exhausted"


def compute_step(
    residual: dict[str, float],
    penalty_by_node: dict[str, float],
    prev_record: IterRecord | None,
    config: CalibConfig,
    *,
    W: float,
    flagged: set[str],
    url: str | None = None,
    scenario: str | None = None,
    dt_slack: "dict[str, dict[tuple[str, str], float]] | None" = None,
) -> "tuple[dict[str, float | dict[tuple[str, str], float]], set[str]]":
    """Compute per-node adder INCREMENTS for the next iteration.

    **C1b — sizing.**  Each shedding node's residual unserved energy is
    converted into an ``energy_margin_adder`` increment that (undamped) would
    inject that residual annual MWh back as demand, in one of two modes:

    * ``config.sizing == "uniform"`` — a CONSTANT per-timestep increment
      ``increment(node) = λ · residual(node) / W`` (see
      :func:`flextool.calibrate._sizing.sized_increments`); ``W`` is the
      invest-timeline annualisation weight, computed ONCE by the loop.
    * ``config.sizing == "timed"`` — the SAME total energy folded from the
      node's ``node_slack_up_dt_e`` stress profile onto the representative
      cells, so ``increment(node)`` is a ``{(period, time): float}`` map placed
      at the stressed hours (see
      :func:`flextool.calibrate._sizing.timed_increments`).  Requires *url*,
      *scenario* and this iteration's *dt_slack* profile.

    Every sized increment carries the ``config.overshoot`` planning-margin
    SAFETY multiplier (default 1.0 = off; >1 provisions beyond the measured
    slack) — applied identically in both sizing modes.

    ``λ`` is the damping factor: ``config.damping_first`` on the FIRST
    correction (no prior bump yet — ``prev_record is None``) and
    ``config.damping_remaining`` thereafter.  Non-shedding nodes get no
    increment.

    **C1c — over-build guard.**  On top of sizing this step:

    1. drops any node already in *flagged* — a node flagged resource-capped
       stays frozen for the rest of the run and is never bumped again;
    2. from the SECOND correction onward (``prev_record is not None``) runs
       :func:`flextool.calibrate._guard.guard_freeze` to REMOVE and FLAG any
       node whose residual FAILED to respond to its prior bump — the freeze
       keys off *residual* (this iteration) and *prev_record.residual* (the
       prior iteration): a node that was shedding last round
       (``prev_record.residual > shed_tol``, the same tolerance the sizer used
       above) but whose gap dropped by less than ``config.stall_fraction`` of
       its prior value is resource-capped (margin buys it no adequacy) and is
       frozen.  Curtailment is NOT consulted (a demand node never curtails, so
       keying the freeze on curtailment could never flag it).  A node that only
       STARTS shedding this round was never bumped, so it gets its first bump
       rather than being frozen.  On the first correction there is no prior to
       diff, so nothing is flagged and every shedding node is bumped.

    ``penalty_by_node`` is this iteration's monetised slack, carried for
    reporting/diagnostics.

    Returns ``(increments, newly_flagged)``: ``{node: increment_MWh}`` to ADD
    to the running adders (a missing node means "no change"), and the set of
    nodes newly flagged this round for the loop to union into its persistent
    flagged set.
    """
    lam = config.damping_first if prev_record is None else config.damping_remaining
    if config.sizing == "timed":
        if url is None or scenario is None or dt_slack is None:
            raise ValueError(
                "timed sizing requires url, scenario and dt_slack to be "
                "threaded into compute_step."
            )
        increments = timed_increments(
            residual, dt_slack, url, scenario,
            lam=lam, overshoot=config.overshoot,
        )
    else:
        increments = sized_increments(
            residual, W=W, lam=lam, overshoot=config.overshoot,
        )
    # A persistently-flagged node is resource-capped: never bump it again.
    increments = {n: v for n, v in increments.items() if n not in flagged}

    # The guard can only diff against a prior iteration, so it acts from the
    # SECOND correction onward; the first correction bumps all shedding nodes.
    if prev_record is None:
        return increments, set()

    return guard_freeze(
        increments,
        residual=residual,
        prev_residual=prev_record.residual,
        stall_fraction=config.stall_fraction,
        # Same shedding tolerance the sizer used above, so guard and sizer
        # agree on which nodes were shedding (and thus bumped) last round.
        shed_tol=_SHED_TOL_MWH,
    )


def _dt_slack_lookup(
    assess_dir: Path,
) -> dict[str, dict[tuple[str, str], float]]:
    """Read ``node_slack_up_dt_e`` into a ``{node: {(period, time): slack}}``
    lookup for the timed sizer (drops null / zero cells)."""
    out: dict[str, dict[tuple[str, str], float]] = {}
    for node, frame in read_residual_unserved_dt(assess_dir).items():
        cells: dict[tuple[str, str], float] = {}
        for period, time_, value in frame.itertuples(index=False):
            v = float(value)
            if v != 0.0:
                cells[(str(period), str(time_))] = v
        if cells:
            out[str(node)] = cells
    return out


def _copy_adders(
    adders: "dict[str, float | dict[tuple[str, str], float]]",
) -> "dict[str, float | dict[tuple[str, str], float]]":
    """Snapshot the adder state (per-cell maps copied, scalars passed through)."""
    return {
        node: (dict(val) if isinstance(val, dict) else val)
        for node, val in adders.items()
    }


def _accumulate_adder(
    adders: "dict[str, float | dict[tuple[str, str], float]]",
    node: str,
    inc: "float | dict[tuple[str, str], float]",
) -> None:
    """Add *inc* into ``adders[node]`` in place.

    Scalar increments accumulate arithmetically (uniform sizing); per-cell map
    increments accumulate CELL-WISE (timed sizing), so a node's stressed cells
    keep rising across corrections just as the scalar adder does.
    """
    if isinstance(inc, dict):
        cur = adders.get(node)
        if not isinstance(cur, dict):
            cur = {}
        for cell, v in inc.items():
            cur[cell] = cur.get(cell, 0.0) + v
        adders[node] = cur
    else:
        base = adders.get(node, 0.0)
        adders[node] = (base if isinstance(base, float) else 0.0) + inc


def _prepare_out_root(out_root: Path, iteration: int, debug: bool) -> None:
    """Clear (or, in debug, archive) the prior ``output_parquet`` tree.

    A clean output directory each iteration is what makes P2's freshness
    check meaningful: a genuinely failed solve skips output writing and so
    would leave the previous iteration's files in place.  In debug mode the
    prior tree is preserved under ``out_iter_<k-1>/`` for inspection; a
    stray tree present before the baseline (k=0) is simply removed.
    """
    prior = Path(out_root) / "output_parquet"
    if not prior.exists():
        return
    if debug and iteration > 0:
        dest = Path(out_root) / f"out_iter_{iteration - 1}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(prior), str(dest))
    else:
        shutil.rmtree(prior)


def run_calibration(
    url: str, scenario: str, config: CalibConfig,
) -> CalibResult:
    """Run the adequacy-margin calibration loop for *scenario*.

    Solves ``config.iterations + 1`` times (iteration 0 is the baseline),
    verifying every solve with P2's detector and reading the per-node
    residual unserved energy each time.  With the C1b :func:`compute_step`
    each shedding node's residual is sized into an ``energy_margin_adder``
    increment (``λ · residual / W``) and accumulated, so the run actually
    raises adders and drives slack down across iterations.  C1c adds the
    over-build guard on top without changing the loop.

    Raises
    ------
    CalibError
        If any iteration's solve cannot be trusted (fail-closed).
    """
    # W — the invest-timeline annualisation weight; a fixed property of the
    # scenario's invest grid, so compute it ONCE and thread it to every
    # sizing step.  Reads the DB only here (see _sizing.invest_weight_W).
    W = invest_weight_W(url, scenario)

    adders: dict[str, float] = {}
    trajectory: list[IterRecord] = []
    prev_record: IterRecord | None = None
    # C1c over-build guard state: the PERSISTENT set of nodes frozen as
    # resource-capped (never bumped again once flagged).  The freeze keys off
    # the residual RESPONSE to a bump, not curtailment, so no baseline spill is
    # tracked.
    flagged: set[str] = set()
    converged = False
    # Why the loop stopped; stays "budget_exhausted" unless an exit below
    # sets it (threshold met -> "converged"; no bump possible -> "stalled").
    stop_reason = "budget_exhausted"
    iterations_run = 0

    for k in range(config.iterations + 1):
        # 1. Materialise the current adders in the calibration alternative
        #    (k=0 writes an empty/zero alt so the scenario is solved THROUGH
        #    the calib alt from the very first iteration; the alt stack is
        #    then constant across the run).
        write_calib_alt(url, scenario, adders)

        # 2. Give this iteration a clean output dir (archive in debug mode).
        _prepare_out_root(config.out_root, k, config.debug)

        # 3. Solve, then verify fail-closed (always pass started_at).
        run = run_solve(
            url,
            scenario,
            work_dir=config.work_dir,
            out_root=config.out_root,
            cache_dir=config.warm_start_cache_dir,
        )
        solve_seconds = time.time() - run.started_at
        iterations_run = k + 1
        outcome = assess_solve(
            run.assess_dir,
            exit_code=run.returncode,
            started_at=run.started_at,
        )
        if not outcome.succeeded:
            raise CalibError(
                f"iteration {k} solve not trusted: {outcome.reason}"
            )

        # 4. Read the signals and record this iteration.
        residual = read_residual_unserved(run.assess_dir)
        curtailment = read_curtailment_by_sink(run.assess_dir)
        penalty_total, penalty_by_node = read_slack_penalty(run.assess_dir)
        # The per-cell stress profile is only needed for timed sizing.
        dt_slack = (
            _dt_slack_lookup(run.assess_dir)
            if config.sizing == "timed"
            else None
        )
        record = IterRecord(
            iteration=k,
            adders=_copy_adders(adders),
            residual=residual,
            curtailment=curtailment,
            penalty_total=penalty_total,
            penalty_by_node=penalty_by_node,
            solve_seconds=solve_seconds,
        )
        trajectory.append(record)

        # 5. Converge on total residual unserved energy.
        if record.total_unserved <= config.slack_threshold_mwh:
            converged = True
            stop_reason = "converged"
            break

        # 6. Size the next step, apply the over-build guard, and accumulate —
        #    but ONLY when a further solve will VALIDATE it.  On the FINAL
        #    iteration (k == config.iterations) there is no subsequent solve,
        #    so applying an increment here would leave a phantom, unverified
        #    adder in final_adders that _report.py would mispair against the
        #    LAST solve's residual (the misleading budget_exhausted case).
        #    Skipping the step keeps final_adders == the adders actually SOLVED
        #    at the last iteration; converged/stalled still break earlier, so
        #    only the terminal budget_exhausted path changes.
        if k < config.iterations:
            increments, newly_flagged = compute_step(
                residual, penalty_by_node, prev_record, config,
                W=W, flagged=flagged,
                url=url, scenario=scenario, dt_slack=dt_slack,
            )
            # A newly-flagged node is resource-capped from here on: persist it
            # so no future iteration bumps it again.  Union FIRST so the
            # flagged set is complete before the stall check reads it.
            flagged |= newly_flagged

            # Early stop — STALLED.  An empty increment set means no node will
            # be bumped this round: either every remaining shedding node is
            # flagged resource-capped, or no single node exceeds the per-node
            # sizing tolerance.  Since the adders are then frozen and the guard
            # set is monotone, the next solve would run on an IDENTICAL model
            # and no future iteration could differ.  We are above the threshold
            # (the converge check above did not fire), so this is the
            # "converged modulo the resource-capped nodes" stop: end the run
            # now instead of burning the rest of the --iterations budget on
            # identical solves.
            if not increments:
                stop_reason = "stalled"
                break

            for node, inc in increments.items():
                _accumulate_adder(adders, node, inc)
            prev_record = record

    return CalibResult(
        converged=converged,
        stop_reason=stop_reason,
        iterations_run=iterations_run,
        final_adders=dict(adders),
        trajectory=trajectory,
        guard_flagged_nodes=sorted(flagged),
    )


__all__ = [
    "CalibConfig",
    "CalibError",
    "CalibResult",
    "IterRecord",
    "calib_alt_name",
    "compute_step",
    "guard_freeze",
    "run_calibration",
]
