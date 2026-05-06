"""Native flexpy orchestrator — Γ.8.D conductor.

This module is the master loop replacement for
``flextool/flextoolrunner/orchestration.py:run_model`` (638 LOC).
The Γ.8.D port lands a flexpy-native driver that:

* Combines the foundation modules (Γ.8.A ``_solve_config`` +
  Γ.8.B ``_timeline`` + Γ.8.C ``_recursive_solve`` + ``_stochastic``).
* Drives the per-solve preprocessing using flextool's existing
  preprocessing modules (the L0-L9 batch + ``preprocessing_solve_time``
  + ``solve_writers``) — those CSV writers stay the source of truth
  while flexpy's `load_flextool` continues to read them.
* Runs the actual solve via ``polar_high.Problem.solve`` (HiGHS)
  instead of glpsol/AMPL.
* Captures :class:`SolveHandoff` per solve via the native
  :func:`build_handoff_from_flexpy`, threads it forward as
  ``prior_handoff``, and routes it into the in-memory handoff slot of
  flextool's runner so the consume side (``preprocessing_solve_time``,
  ``handoff_writers``, ``cumulative_handoffs``) reads from it.

Design choices
--------------

* **CSV writers are still flextool's** — replacing them is a separate
  phase (Γ.7 / Γ.9 in the audit numbering).  Γ.8.D's job is to run the
  master loop natively, not to retire CSVs.  This means the orchestrator
  drives ``flextoolrunner.orchestration.run_model`` once per top-level
  invocation, but with a **flexpy-as-inner-solver** wrapper that:
    - Reads the per-solve snapshot via ``load_flextool``.
    - Builds the LP via ``build_flextool``.
    - Solves via ``polar_high`` (HiGHS).
    - Captures handoff via ``build_handoff_from_flexpy``.
    - Deposits handoff into ``state.handoffs`` for the next iteration's
      preprocessing.

* **Storage-fixing handoff is in-memory by default** when
  ``state.handoffs`` is non-None.  The legacy file-copy path
  (``shutil.copy`` of ``solve_data_<parent>/fix_storage_*.csv``) is
  consulted only when ``state.handoffs is None`` — see
  ``flextool/flextoolrunner/orchestration.py:362-431``.  This is a
  documented behaviour divergence: flexpy's preferred path is in-memory
  and avoids touching disk.

* **Roll-counter reset semantics**: every top-level
  :func:`run_orchestration` call invokes
  ``state.solve.roll_counter = state.solve.make_roll_counter()`` first
  so test re-use of the same SolveConfig doesn't desync (R-O5).

* **`model_solve` validation**: enforced loud-and-early.  Empty
  ``model_solve`` or more-than-one model raises
  :class:`FlexToolConfigError` per the audit's
  ``orchestration.py:634-637``.

* **Δ.12e**: the legacy file-symlink ``run_chain(native=False)`` driver
  retired once the native cascade reached feature parity (warm-LP,
  handoff carriers, output writer, override chain).  ``run_chain`` is
  now a thin compat shim that always delegates here.

Reference: ``flextool/flextoolrunner/orchestration.py``.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from flextool.engine_polars._solve_handoff import SolveHandoff
from flextool.engine_polars._solve_state import (
    FlexToolConfigError,
    PathConfig,
    RunnerState,
)

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig
    from flextool.engine_polars.input import FlexData


# Repository pin — flextool's preprocessing modules live in the same
# tree as the engine_polars package, but the legacy ``flextool``
# top-level (separate repo at ``/home/jkiviluo/sources/flextool``) is
# also available for some Spine fixtures.  Mirror the path-shim used
# elsewhere so imports of ``flextool.flextoolrunner.flextoolrunner``
# resolve when running against either tree.
_REPO_ROOT = Path("/home/jkiviluo/sources/flextool")


def _ensure_flextool_importable() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.append(str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class OrchestrationStep:
    """Per-solve result of :func:`run_orchestration`.

    Mirrors :class:`flextool.engine_polars.chain.ChainStep` but produced
    by the native orchestrator path.  ``handoff`` is the carrier used to
    seed the *next* solve's preprocessing.

    Attributes
    ----------
    solve_name : str
        The complete (sub-)solve identifier emitted by flextool's
        orchestration loop (e.g. ``"y2025_5week"`` or
        ``"dispatch_fullYear_roll_roll_3"``).
    solution : polar_high.Solution | None
        The HiGHS solution.  ``None`` only on the failed-solve path.
    handoff : SolveHandoff
        Flexpy-derived handoff carriers, threaded forward.
    obj : float | None
        Objective value (cached for quick comparison; equal to
        ``solution.obj``).
    warm_used : bool
        Δ.12d — True if this solve was produced by warm-updating the
        prior solve's :class:`polar_high.WarmProblem` instance; False
        if it was a cold rebuild.  Always False for the first solve
        and for ``warm=False`` runs.
    """

    solve_name: str
    solution: "Solution | None"
    handoff: SolveHandoff
    obj: float | None = None
    warm_used: bool = False


# ---------------------------------------------------------------------------
# Master loop
# ---------------------------------------------------------------------------


def _validate_model_solve(state: RunnerState) -> list[str]:
    """Validate ``state.solve.model_solve`` and return the solve list.

    Mirrors the guard at ``flextool/flextoolrunner/orchestration.py:78``
    and ``:634-637``: there must be exactly one model with at least one
    solve.  Multi-model is documented as unsupported.
    """
    if not state.solve.model_solve:
        raise FlexToolConfigError(
            "No model. Make sure the 'model' class defines solves [Array]."
        )
    if len(state.solve.model_solve) > 1:
        raise FlexToolConfigError(
            "Trying to run more than one model — not supported. "
            "model_solve must contain exactly one model."
        )
    solves = next(iter(state.solve.model_solve.values()))
    if not solves:
        raise FlexToolConfigError("No solves in model.")
    return solves


def _bootstrap_dirs(work_folder: Path, logger: logging.Logger) -> None:
    """Create ``solve_data/``, ``output_raw/``, ``output_plots/`` under
    *work_folder* if they don't exist.

    Mirrors lines 63-76 of the flextool reference.
    """
    for sub in ("solve_data", "output_raw", "output_plots"):
        try:
            (work_folder / sub).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.debug(f"{sub} folder existed")


def run_orchestration(
    state: RunnerState,
    work_folder: Path | str,
    *,
    runner_factory=None,
    db_url: str | None = None,
    scenario_name: str | None = None,
    warm: bool = False,
) -> dict[str, OrchestrationStep]:
    """Drive the master loop natively.

    Per-step:

    1. Bootstrap directories (idempotent).
    2. Validate ``state.solve.model_solve`` (exactly one model, ≥1 solve).
    3. Reset ``state.solve.roll_counter`` for repeatable test runs (R-O5).
    4. Drive flextool's ``orchestration.run_model`` with a flexpy
       cascade solver — each per-solve iteration loads the snapshot via
       ``load_flextool``, builds the LP via ``build_flextool``, solves
       via HiGHS, captures the handoff, and deposits it into
       ``state.handoffs`` (which the consume side already reads from).

    Parameters
    ----------
    state : RunnerState
        Native flexpy state carrier.  ``state.solve`` and
        ``state.timeline`` must be populated (call
        :func:`run_chain_from_db` for the canonical end-to-end path that
        sets these up from a DB).  The function may flip
        ``state.handoffs`` from ``None`` to ``{}`` to enable the in-memory
        capture/consume path; this is done unconditionally — the native
        orchestrator always uses in-memory handoff (storage-fixing falls
        through to the file-copy path only when ``state.handoffs`` is
        explicitly set to ``None`` after this returns).
    work_folder : Path | str
        Directory the snapshot tree lives under.  Created if missing.
        Per-solve preprocessing CSVs are emitted under
        ``work_folder/solve_data/``.
    runner_factory : callable | None
        Optional override for constructing the underlying
        :class:`FlexToolRunner` — used by tests that want to short-
        circuit flextool's preprocessing.  Default uses the canonical
        constructor.
    warm : bool, default False
        Δ.12d — when True, attempt warm LP updates between consecutive
        structurally-compatible per-solve iterations using
        :class:`polar_high.WarmProblem`.  Reuses one WarmProblem across
        the cascade, applying ``_apply_warm_updates`` between solves
        and falling back to a cold rebuild whenever the structural
        fingerprint changes or any unmapped Param differs.  Decisions
        are recorded per-step on :attr:`OrchestrationStep.warm_used`.
        Default ``False`` preserves the original cold-rebuild
        behaviour.

    Returns
    -------
    dict[str, OrchestrationStep]
        Mapping ``complete_solve_name → OrchestrationStep`` in solve
        order (Python dict insertion order preserved).

    Raises
    ------
    FlexToolConfigError
        Empty / multi-model ``model_solve``.
    FlexToolSolveError
        Any per-solve LP infeasibility / non-optimal status.
    """
    _ensure_flextool_importable()
    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    state.paths = PathConfig(work_folder=work_folder)

    logger = state.logger
    _bootstrap_dirs(work_folder, logger)
    solves = _validate_model_solve(state)

    # Reset roll-counter so repeated calls with the same SolveConfig
    # don't desync.  R-O5 in the orchestration risk register.
    state.solve.roll_counter = state.solve.make_roll_counter()

    # Always enable in-memory handoff for the native path.
    if state.handoffs is None:
        state.handoffs = {}

    # The cascade solver runs flexpy on each solve and captures the
    # handoff.  We use flextool's orchestration loop driver because it
    # encodes the recursive/rolling/stochastic expansion + per-solve
    # preprocessing chain we still consume.  Our cascade solver is the
    # `solver.run(...)` callback inside that loop.
    return _drive_cascade(state, work_folder, solves, runner_factory,
                          db_url=db_url, scenario_name=scenario_name,
                          warm=warm)


def _drive_cascade(
    state: RunnerState,
    work_folder: Path,
    solves: list[str],
    runner_factory,
    *,
    db_url: str | None = None,
    scenario_name: str | None = None,
    warm: bool = False,
) -> dict[str, OrchestrationStep]:
    """Drive the flextool master loop with a flexpy cascade solver.

    For every per-solve iteration:

    1. Read the snapshot via ``load_flextool``.
    2. Build the LP via ``build_flextool`` (cold rebuild) OR warm-update
       the prior iteration's :class:`polar_high.WarmProblem`.
    3. Solve via HiGHS.
    4. Build the handoff via ``build_handoff_from_flexpy``.
    5. Deposit it into ``state.handoffs`` so the next iteration's
       preprocessing picks it up.

    Parameter ``warm`` toggles per-iteration warm-LP updates: when True,
    the cascade reuses one ``WarmProblem`` across consecutive
    structurally-compatible iterations.  See
    :mod:`flextool.engine_polars._warm` for the structural-fingerprint
    + Param-classification machinery.  Cold rebuild (``warm=False``)
    remains the default for backward compatibility with every existing
    caller.

    This mirrors :func:`flextool.engine_polars.input.load_flextool_from_db`'s
    multi-solve cascade but emits an :class:`OrchestrationStep` per solve
    and runs the LAST solve too (the flexpy-side bookkeeping is the
    deliverable, not just an intermediate).
    """
    # Late imports — keep the orchestration module's import surface narrow
    # for callers that only need the dataclass.
    from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
    from flextool.flextoolrunner import orchestration as _flx_orch
    from flextool.flextoolrunner.solver_runner import SolverRunner

    from polar_high import Problem, WarmProblem
    from flextool.engine_polars.input import (
        build_handoff_from_flexpy,
        load_flextool,
    )
    from flextool.engine_polars.model import build_flextool
    from flextool.engine_polars._output_writer import (
        OutputWriterState,
        write_outputs_for_solve,
    )
    from flextool.engine_polars._warm import (
        _IncompatibleUpdate,
        _apply_warm_updates,
        _build_warm_problem,
        _fingerprint,
    )

    results: dict[str, OrchestrationStep] = {}
    # Δ.1: adapter that reuses flextool's process_outputs writers.  The
    # state carrier collects ``periods_already_emitted`` across the
    # cascade so we don't have to round-trip through SolveHandoff.
    writer_state = OutputWriterState()

    # The runner_factory hook lets tests inject a mock; the default uses
    # FlexToolRunner constructed against the same DB the state was
    # loaded from.  Since RunnerState in flexpy doesn't carry a DB URL
    # by default, callers must supply this via runner_factory or use
    # ``run_chain_from_db`` which constructs the runner explicitly.
    if runner_factory is None:
        raise FlexToolConfigError(
            "run_orchestration requires a runner_factory to construct "
            "the underlying FlexToolRunner.  Use run_chain_from_db for "
            "the canonical end-to-end path that wires this for you."
        )
    runner = runner_factory()
    # Push our state's handoff slot onto the runner's state so flextool's
    # consume hooks (preprocessing_solve_time + capture_post_solve) read
    # from the same dict.
    runner.state.handoffs = state.handoffs
    runner.state.logger.setLevel(logger_level := logging.ERROR)

    # Δ.12c — build a SpineDbReader once and reuse it across the cascade.
    # When db_url + scenario_name are supplied (run_chain_from_db wires
    # them), the override chain fires for every per-solve load — covering
    # the seeds the workdir CSV path can't provide once Δ.12-drop /
    # Δ.12c have retired the redundant CSVs.  When the caller didn't
    # supply them, we fall back to load_flextool's per-call autoresolve
    # (which works for fixtures whose work_folder follows the
    # ``work_<scenario>`` convention).
    cascade_db_reader = None
    if db_url is not None and scenario_name is not None:
        from flextool.engine_polars._spinedb_reader import SpineDbReader
        try:
            cascade_db_reader = SpineDbReader(db_url, scenario=scenario_name)
        except Exception:  # noqa: BLE001
            cascade_db_reader = None

    class _FlexpyCascadeSolver(SolverRunner):
        def __init__(self, runner_state):
            super().__init__(runner_state)
            self._all_steps: dict[str, OrchestrationStep] = results
            # Δ.12d — warm-LP carry-over state.  ``_warm_problem`` holds
            # the live :class:`polar_high.WarmProblem` reused across
            # consecutive structurally-compatible iterations; ``_prior_data``
            # / ``_prior_fp`` snapshot the previous iteration's FlexData +
            # fingerprint for the diff scan in
            # :func:`_apply_warm_updates`.  All three stay None when
            # ``warm=False`` (the existing cold-cascade behaviour) AND
            # are reset to None on every cold rebuild.
            self._warm_problem: "WarmProblem | None" = None
            self._prior_data = None
            self._prior_fp: "tuple | None" = None

        def run(self, complete_solve_name: str) -> int:
            # Δ.12 — wire ``handoff=`` through ``load_flextool`` so the
            # in-memory carriers from the prior solve flow into this
            # solve's FlexData directly.  Replaces the previous
            # implicit dependency on flextool's per-solve preprocessing
            # rewriting ``solve_data/p_entity_*.csv`` between solves.
            # After Δ.12 the cascade reads these five carrier-derived
            # fields from the in-memory ``SolveHandoff`` rather than the
            # workdir CSVs:
            #
            #   * ``p_entity_invested``
            #   * ``p_entity_divested``
            #   * ``p_entity_previously_invested_capacity``
            #   * ``p_roll_continue_state``
            #   * ``p_fix_storage_quantity``
            prior_for_load = (
                self.state.handoffs.get(self.state.last_captured_solve)
                if self.state.last_captured_solve is not None else None
            )
            data = load_flextool(
                self.state.paths.work_folder,
                handoff=prior_for_load,
                db_reader=cascade_db_reader,
            )

            # Δ.12d — warm-LP per-iteration decision.  When ``warm`` is
            # True AND the prior iteration left a live WarmProblem
            # whose fingerprint matches this iteration's data, attempt
            # to push the Param diff into the live LP.  Any
            # _IncompatibleUpdate (unmapped Param differs, gate
            # transitions to inactive while Param contributed cells,
            # …) drops back to a cold rebuild.  Cold rebuild also
            # fires on the first iteration and on any structural
            # fingerprint mismatch.
            warm_used = False
            if warm:
                fp = _fingerprint(data)
                tried_warm = (
                    self._warm_problem is not None
                    and self._prior_data is not None
                    and self._prior_fp == fp
                )
                if tried_warm:
                    try:
                        _apply_warm_updates(self._warm_problem,
                                            self._prior_data, data)
                        warm_used = True
                    except _IncompatibleUpdate:
                        # Drop the stale warm problem so the next
                        # branch builds a fresh one.
                        self._warm_problem = None
                if not warm_used:
                    self._warm_problem = _build_warm_problem(data)
                # ``WarmProblem.solve`` always keeps the HiGHS instance
                # alive on ``Solution.highs`` — that's the whole point
                # of warm reuse — so the output writer adapter
                # (``write_all_variables`` / ``write_all_handoffs``)
                # sees the live solver as it does for cold rebuilds
                # under ``keep_solver=True``.  No extra kwarg required.
                sol = self._warm_problem.solve()
                self._prior_data = data
                self._prior_fp = fp
            else:
                pb = Problem()
                build_flextool(pb, data)
                # Δ.12c-fix: ``keep_solver=True`` so ``sol.highs``
                # carries the live HiGHS instance the output writer
                # adapter consumes (``write_all_variables`` /
                # ``write_all_handoffs`` read MPS column / row names
                # directly off the solver).  Without this the adapter
                # no-ops and the native cascade emits zero parquets —
                # the regression
                # ``test_native_cascade_emits_reference_output_raw_files``
                # surfaced in Δ.12c-fix.
                sol = pb.solve(keep_solver=True)
            if not sol.optimal:
                self.state.logger.error(
                    f"flexpy non-optimal for {complete_solve_name}"
                )
                return 1

            prior = prior_for_load
            # Δ.1 — emit TIER A output_raw artefacts BEFORE the in-memory
            # handoff is built.  ``write_all_handoffs`` (called by the
            # adapter) refreshes ``solve_data/period_capacity.csv`` and
            # other handoff CSVs; ``build_handoff_from_flexpy`` then
            # reads those refreshed files for the in-memory handoff.
            try:
                write_outputs_for_solve(
                    sol,
                    work_folder=self.state.paths.work_folder,
                    solve_name=complete_solve_name,
                    prior_handoff=prior,
                    writer_state=writer_state,
                )
            except Exception as exc:  # noqa: BLE001
                self.state.logger.warning(
                    f"write_outputs_for_solve failed for "
                    f"{complete_solve_name}: {exc}"
                )

            handoff = build_handoff_from_flexpy(
                sol, self.state.paths.work_folder, complete_solve_name,
                prior_handoff=prior,
            )
            # Deposit so flextool's consume side picks it up on the next
            # iteration's preprocessing AND so we have it for the result
            # dict.
            #
            # IMPORTANT: flextool's ``orchestration.run_model`` also
            # calls ``capture_post_solve(state, complete_solve[solve])``
            # immediately after ``solver.run`` returns.  That helper
            # reads the current ``solve_data/*.csv`` files and OVERWRITES
            # the carriers on the same handoff object — which would
            # silently replace our flexpy-derived ``realized_invest``
            # with whatever the prior-handoff-seeded preprocessing
            # wrote to disk for THIS solve (i.e., the prior solve's
            # state, not our actual flexpy result).  We monkey-patch
            # ``capture_post_solve`` to a no-op for the duration of
            # this loop in ``_drive_cascade`` below — the override
            # restores it on exit.
            self.state.handoffs[complete_solve_name] = handoff
            self._all_steps[complete_solve_name] = OrchestrationStep(
                solve_name=complete_solve_name,
                solution=sol,
                handoff=handoff,
                obj=sol.obj,
                warm_used=warm_used,
            )
            return 0

    # Monkey-patch ``capture_post_solve`` at the orchestration module's
    # binding so the file-based capture doesn't overwrite our flexpy-
    # extracted handoffs.  The patch is restored on exit so other
    # callers (legacy file-based path) keep their behaviour.
    _real_capture = _flx_orch.capture_post_solve
    _flx_orch.capture_post_solve = lambda state, solve_name: None
    try:
        _flx_orch.run_model(runner.state, _FlexpyCascadeSolver(runner.state))
    finally:
        _flx_orch.capture_post_solve = _real_capture
    # Mirror the in-memory handoff dict back onto our state in case
    # callers want to inspect it.
    state.handoffs = runner.state.handoffs
    return results


# ---------------------------------------------------------------------------
# run_chain_from_db — top-level entry point
# ---------------------------------------------------------------------------


def run_chain_from_db(
    input_db_url: str | Path,
    scenario_name: str | None = None,
    work_folder: Path | str | None = None,
    *,
    flextool_dir: Path | str | None = None,
    bin_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
    warm: bool = False,
) -> dict[str, OrchestrationStep]:
    """Run a flextool multi-solve scenario end-to-end natively.

    Combines:

    1. :func:`flextool.engine_polars._native_input_writer.write_workdir_inputs`
       (Δ.20 — engine_polars-owned) populates the workdir's ``input/``
       and ``solve_data/`` CSVs.  Replaces the legacy
       ``FlexToolRunner.write_input`` call from earlier phases — the
       cascade no longer calls into the FlexToolRunner.write_input
       method directly.
    2. ``_orchestration.run_orchestration`` to drive the per-solve loop
       with a flexpy-as-inner-solver wrapper.
    3. Returns one :class:`OrchestrationStep` per per-solve iteration.

    For tests / scripts that want a single function call to go from a
    DB scenario to a dict of (Solution, SolveHandoff) pairs.

    Parameters
    ----------
    input_db_url : str | Path
        Spine SQLite URL or path.  A bare path is upgraded to ``sqlite:///``.
    scenario_name : str, optional
        Scenario filter to apply.  ``None`` picks the first scenario.
    work_folder : Path | str, optional
        Where to materialise the CSVs.  ``None`` uses an auto-cleaned
        tempdir.
    flextool_dir, bin_dir : Path, optional
        Override the default flextool install location.  Default:
        ``/home/jkiviluo/sources/flextool/{flextool,bin}``.
    logger : logging.Logger, optional
        Logger to use.  ``None`` constructs one named after the scenario.
    warm : bool, default False
        Δ.12d — when True, reuse one :class:`polar_high.WarmProblem`
        across consecutive structurally-compatible per-solve iterations
        in the cascade, applying ``_apply_warm_updates`` between solves
        rather than cold-rebuilding.  See
        :func:`run_orchestration` for full semantics.

    Returns
    -------
    dict[str, OrchestrationStep]
        Mapping ``complete_solve_name → OrchestrationStep``.  Iterate
        in insertion order to walk the chain.
    """
    _ensure_flextool_importable()
    from flextool.flextoolrunner.flextoolrunner import FlexToolRunner

    if logger is None:
        logger = logging.getLogger(
            f"flexpy.run_chain_from_db[{scenario_name}]"
        )

    db_url = str(input_db_url)
    if not db_url.startswith("sqlite:") and not db_url.startswith("postgresql"):
        db_url = f"sqlite:///{db_url}"

    if work_folder is None:
        work_folder = Path(tempfile.mkdtemp(prefix="flexpy_run_chain_"))
    else:
        work_folder = Path(work_folder)
        work_folder.mkdir(parents=True, exist_ok=True)

    flextool_dir_resolved = (
        Path(flextool_dir) if flextool_dir is not None
        else _REPO_ROOT / "flextool"
    )
    bin_dir_resolved = (
        Path(bin_dir) if bin_dir is not None else _REPO_ROOT / "bin"
    )

    # Δ.20 — workdir CSV population is now owned by engine_polars.
    # ``_native_input_writer.write_workdir_inputs`` invokes the input
    # writers from inside engine_polars; the live cascade no longer
    # references ``FlexToolRunner.write_input``.  Future Δ.20+ phases
    # progressively replace the underlying writers with InputSource-
    # driven helpers.
    from flextool.engine_polars._native_input_writer import (
        write_workdir_inputs,
    )

    write_workdir_inputs(
        db_url,
        scenario_name,
        work_folder,
        logger=logger,
    )

    # Construct the underlying FlexToolRunner — still needed to carry
    # the cross-cutting ``RunnerState`` (timeline, solve config, handoff
    # dict) through ``flextool.flextoolrunner.orchestration.run_model``,
    # which the native cascade still drives for the per-solve
    # preprocessing chain (``preprocessing_solve_time``,
    # ``solve_writers``, ``handoff_writers``).  No write_input call.
    def _runner_factory() -> "FlexToolRunner":
        runner = FlexToolRunner(
            input_db_url=db_url,
            scenario_name=scenario_name,
            flextool_dir=flextool_dir_resolved,
            bin_dir=bin_dir_resolved,
            work_folder=work_folder,
        )
        runner.state.logger.setLevel(logging.ERROR)
        return runner

    # Build a minimal native RunnerState so callers can introspect
    # state.handoffs after the run.  The real per-solve mutation happens
    # on the underlying flextool runner's state (driven inside
    # _drive_cascade); we mirror the handoffs dict back.
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig

    sc = SolveConfig.load_from_db_url(db_url, scenario_name, logger=logger)
    tc = TimelineConfig.load_from_db_url(db_url, scenario_name, logger=logger)
    tc.create_assumptive_parts(sc)
    tc.create_timeline_from_timestep_duration(sc)

    state = RunnerState(
        paths=PathConfig(work_folder=work_folder),
        solve=sc,
        logger=logger,
        timeline=tc,
        handoffs={},
    )

    return run_orchestration(
        state, work_folder, runner_factory=_runner_factory,
        db_url=db_url, scenario_name=scenario_name, warm=warm,
    )


__all__ = [
    "OrchestrationStep",
    "run_orchestration",
    "run_chain_from_db",
]
