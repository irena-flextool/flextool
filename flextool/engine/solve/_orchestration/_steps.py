"""The per-solve result types (:class:`SnapshotSolution`, :class:`OrchestrationStep`) and the objective-scale
outputs of a sub-solve."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from flextool.engine.solve._handoff import SolveHandoff

if TYPE_CHECKING:
    import polars as pl
    from polar_high import Solution
    from flextool.engine.preprocess.flex_data import FlexData


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SnapshotSolution:
    """Lightweight Solution-API stand-in carrying only the captured
    decision-variable frames a sub-solve needs to survive polar-high's
    internal between-solve release of ``Solution._vars``.

    The cascade captures :attr:`OrchestrationStep.captured_vars` at the
    earliest reliable point in the per-solve loop (immediately before
    the step is deposited in ``self._all_steps``).  Polar-high may
    release the live :class:`polar_high.Solution._vars` dict for
    earlier sub-solves between the deposit and the end-of-cascade
    consumers (e.g. :func:`flextool.process_outputs.read_parameters.
    _entity_all_capacity`), so end-of-cascade writers that look at
    every sub-solve's decision variables fall back to this snapshot.

    The wrapper only satisfies the duck-typed contract that
    ``_entity_all_capacity`` needs:

    * ``solution._vars`` — dict-like with the captured names as keys
      (supports ``"v_invest_p" in solution._vars``).
    * ``solution.value(name)`` — returns the captured polars long-form
      DataFrame.

    All other ``Solution`` attributes (``obj``, ``optimal``, ``highs``,
    ``col_value``, …) are intentionally absent; callers that need
    those should read the live :attr:`OrchestrationStep.solution`
    (last step only by default, or every step under
    ``keep_solutions=True``).
    """

    _vars: "dict[str, pl.DataFrame]"

    def value(self, name: str) -> "pl.DataFrame":
        return self._vars[name]


@dataclass
class OrchestrationStep:
    """Per-solve result of :func:`run_orchestration`.

    Mirrors :class:`flextool.engine.solve.chain.ChainStep` but produced
    by the native orchestrator path.  ``handoff`` is the carrier used to
    seed the *next* solve's preprocessing.

    Attributes
    ----------
    solve_name : str
        The complete (sub-)solve identifier emitted by flextool's
        orchestration loop (e.g. ``"y2025_5week"`` or
        ``"dispatch_fullYear_roll_roll_3"``).
    solution : polar_high.Solution | None
        The HiGHS solution.  By default (``keep_solutions=False`` on
        :func:`run_chain_from_db` / :func:`run_orchestration`) only the
        LAST sub-solve in a cascade retains its ``solution`` — earlier
        steps clear this slot to release the HiGHS instance + variable
        arrays.  Set ``keep_solutions=True`` to retain ``solution`` on
        every step (Phase C.5 — memory discipline).  Also ``None`` on
        the failed-solve path.
    handoff : SolveHandoff
        polar_high-derived handoff carriers, threaded forward.  Always
        populated (kilobyte-sized; safe to retain across the cascade).
    obj : float | None
        Objective value (cached for quick comparison; equal to
        ``solution.obj``).  Always populated when the solve succeeded —
        survives the ``keep_solutions=False`` slim pass, so cascade-
        wide objective sweeps work without ``keep_solutions=True``.
    optimal : bool | None
        Phase C.5 — slim summary mirror of ``solution.optimal`` that
        survives the per-step memory release.  ``None`` only on the
        failed-solve path (where ``solution`` is also ``None``).  This is
        the STRICT solver verdict (HiGHS ``kOptimal``): a near-optimal
        crossover-off solve that :func:`classify_acceptance` accepted for
        consumption is ``False`` here — read :attr:`near_optimal` (or the
        ``optimal or near_optimal`` union) when the question is "is this
        step's solution usable?", not "did the solver certify optimality?".
    near_optimal : bool
        True when the solve was NOT strictly ``kOptimal`` yet
        :func:`classify_acceptance` accepted it as in-practice-optimal
        (feasible primal, small primal--dual gap) — the crossover-off
        interior-point case.  Such a solve wrote a usable solution and must
        NOT be treated as a failure by the CLI exit-code scan.  Always
        ``False`` for the Benders path (its acceptance is governed by
        ``is_benders`` + gap/tol) and for the failed-solve path.
    warm_used : bool
        True if this solve re-solved the prior solve's
        :class:`polar_high.WarmProblem` (an identical sub-solve); False
        if it was a cold rebuild.  Always False for the first solve
        and for ``warm=False`` runs.  Always populated (slim summary).
    flex_data : FlexData | None
        Δ.31 — the polars input bundle this sub-solve consumed.  Held
        on the step so downstream :func:`flextool.process_outputs.
        write_outputs` can build the parameter / set namespaces in
        memory instead of re-parsing the workdir CSVs.  Subject to the
        same ``keep_solutions`` gating as ``solution`` (Phase C.5):
        only the LAST step retains ``flex_data`` by default.  ``None``
        on the failed-load path.
    flex_data_provider : FlexDataProvider | None
        The per-sub-solve :class:`FlexDataProvider` populated by the
        cascade's writers.  Subject to the same ``keep_solutions``
        gating as ``solution`` / ``flex_data`` (Phase C.5): only the
        LAST step retains it by default.  Consumed by ``--csv-dump``
        in ``cmd_run_flextool`` to snapshot the cascade's derived
        frames to disk.
    """

    solve_name: str
    solution: "Solution | None"
    handoff: SolveHandoff
    obj: float | None = None
    optimal: bool | None = None
    near_optimal: bool = False
    warm_used: bool = False
    flex_data: "FlexData | None" = None
    flex_data_provider: "object | None" = None
    is_benders: bool = False
    """True when this step was produced by the Benders region driver.  It
    carries a full :class:`Solution` when its final solve was accepted
    (``benders_results == "full"``), else only a :class:`SnapshotSolution`
    invest carrier."""
    benders_gap: float | None = None
    """Benders steps only — the relative optimality gap reached at the
    incumbent ``(best_UB − LB)/max(1, |best_UB|)``.  Paired with
    ``benders_tol`` / ``benders_iterations`` so the CLI exit-code scan can
    report a NON-convergence (feasible incumbent, gap never met ``tol``) as a
    loud warning rather than a bogus "infeasible/unbounded"."""
    benders_tol: float | None = None
    """Benders steps only — the relative-gap tolerance that was in force."""
    benders_iterations: int | None = None
    """Benders steps only — the number of master/subproblem iterations run."""
    benders_results: str | None = None
    """Benders steps only — what the solve gives the result tables (item 287): ``"full"`` (its final solve, the
    whole model with the Benders investments fixed, was accepted: ``solution`` is that solve's and every table is
    written as for a monolithic solve), ``"invest_slice"`` (its investments and their costs only: final solve off
    or not accepted, followed by other solves) or ``"none"``.  ``optimal`` of a Benders step stays the Benders
    convergence (``near_optimal`` ``False``), whatever the final solve gave, so the CLI exit-code scan still
    reports a non-converged Benders solve."""
    captured_vars: "dict[str, pl.DataFrame]" = field(default_factory=dict)
    """Per-sub-solve snapshot of the decision-variable frames that
    end-of-cascade writers (``_entity_all_capacity`` and friends) need
    after polar-high has released the live ``Solution._vars`` dict for
    earlier sub-solves.

    Captured at the latest reliable point in the per-solve loop —
    immediately before the step is deposited — so it always reflects
    the sub-solve's own values regardless of how polar-high or
    downstream slimming touches ``self.solution``.  Empty dict when
    the solve failed or when there was no Solution to capture from.

    Access via :attr:`effective_solution` rather than reading directly:
    callers normally want the live Solution when it's still populated
    and the snapshot only as a fallback.
    """

    @property
    def effective_solution(self) -> "Solution | SnapshotSolution | None":
        """Return the live :attr:`solution` when its ``_vars`` is still
        populated; otherwise a :class:`SnapshotSolution` over the
        captured frames.  ``None`` only when both are unavailable
        (failed-solve path, or solution slimmed AND no capture taken).

        End-of-cascade consumers that walk every sub-solve's decision
        variables (e.g. ``read_parameters_multi`` →
        ``_entity_all_capacity``) should read this instead of
        :attr:`solution` so non-last sub-solves remain observable
        after polar-high's between-solve ``_vars`` release.
        """
        sol = self.solution
        live_vars = getattr(sol, "_vars", None) if sol is not None else None
        if live_vars:
            return sol
        if self.captured_vars:
            return SnapshotSolution(_vars=dict(self.captured_vars))
        return sol


# ---------------------------------------------------------------------------
# Scaling-output helper  (shared by cascade & single-solve paths)
# ---------------------------------------------------------------------------


def _write_scale_csv(
    *,
    solve_data_dir: Path,
    solve_name: str,
    effective_obj_scale: float,
    logger: logging.Logger,
) -> None:
    """Emit ``solve_data/scale_the_objective.csv`` for the given solve.

    Required by the downstream parquet / CSV writers — they read it via
    :func:`flextool.process_outputs.read_highs_solution.
    _resolve_inv_scale_the_objective` to un-scale variable values and
    duals back to user-facing units.  Best-effort: a write failure logs
    a warning but does not raise.

    The legacy human-readable ``scaling_report.txt`` diagnostic was
    retired in Phase 2b; the autoscaler's ``solve_data/autoscale_<solve>.yaml``
    (written from :func:`_autoscale_emit_layer1`) is the new
    machine-readable audit.  Callers should write the CSV exactly once
    per base solve — its value is invariant across rolls of the same
    base solve.
    """
    try:
        from flextool.engine.preprocess.emit._solve_writers import (
            derive_scale_the_objective,
        )
        sd = Path(solve_data_dir)
        sd.mkdir(parents=True, exist_ok=True)
        path = sd / "scale_the_objective.csv"
        derive_scale_the_objective(effective_obj_scale).write_csv(
            path, line_terminator="\r\n",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scale_the_objective.csv write failed for %s: %s",
            solve_name, exc,
        )


def _emit_provider_obj_scale(state, effective_obj_scale: float) -> None:
    """Put the objective scale this sub-solve's LP is built with into its provider.

    ``solve_data/scale_the_objective`` in the sub-solve provider holds exactly the value passed to
    ``build_flextool``, so a provider snapshot (``--csv-dump``, a test run's dump) carries the solve's real scale.
    No reader in the run uses the key (the output writers get the scale as an argument); it is there for the dump.
    """
    provider = getattr(state, "current_provider", None)
    if provider is None:
        return
    from flextool.engine.preprocess.emit._solve_writers import emit_scale_the_objective
    emit_scale_the_objective(
        state.paths.work_folder / "solve_data", effective_obj_scale, provider=provider,
    )
