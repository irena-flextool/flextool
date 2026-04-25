"""
SolverRunner — invokes external solver binaries (glpsol, HiGHS, CPLEX).

Manages the three-phase glpsol/HiGHS workflow and filters/reformats
solver output.

Entry point: ``SolverRunner.run(current_solve) -> int``
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from typing import IO, Optional

import highspy

from flextool.flextoolrunner.runner_state import RunnerState, FlexToolSolveError
from flextool.flextoolrunner.scaling import (
    apply_bound_scale_decision,
    update_bound_scale_in_cache,
)


# ---------------------------------------------------------------------------
# Agent 18d: user-facing solver-option knobs
# ---------------------------------------------------------------------------
#
# These two knobs are orthogonal to the LP-scaling infrastructure (Agents
# 5/8/12/18b/18c): they don't change the LP itself, they change how
# HiGHS solves it.  Both are aimed at rivendell-shaped wide-bound
# structurally-degenerate models where dual simplex stalls just below
# the default feasibility tolerance.
#
# ``--relax-feasibility[=TOL]``: loosens HiGHS' primal + dual
# feasibility tolerances from ``1e-7`` to ``1e-5`` (or a user value),
# so a solve that reaches ``Pr=0, Du=0`` within the new tolerance is
# accepted without the Markowitz-bump retry loop.
#
# ``--ipm``: switches HiGHS from simplex to interior-point, which has
# no basis and therefore cannot stall on Markowitz pivoting.

RELAX_FEASIBILITY_ENV_VAR = "FLEXTOOL_RELAX_FEASIBILITY"
"""Environment-variable fallback for ``--relax-feasibility``.  Set to
an empty value or ``1`` / ``yes`` / ``on`` / ``true`` to request the
default relaxed tolerance (``1e-5``); set to a floating-point number to
request an explicit tolerance."""

IPM_ENV_VAR = "FLEXTOOL_IPM"
"""Environment-variable fallback for ``--ipm``.  Truthy values
(``1`` / ``yes`` / ``on`` / ``true``) switch HiGHS to the interior-point
solver; unset / falsy leaves HiGHS' default simplex method in place."""

DEFAULT_RELAX_FEASIBILITY = 1e-5
"""Default tolerance used when ``--relax-feasibility`` is passed
without a value.  Two orders of magnitude looser than HiGHS'
``1e-7`` default — loose enough to absorb the rivendell S19
``5.67e-7`` residual without being irresponsibly loose."""


def resolve_relax_feasibility(cli_value) -> Optional[float]:
    """Resolve ``--relax-feasibility`` into an explicit tolerance.

    Accepts the CLI value (which may be ``None`` for absent, the string
    ``"default"`` for flag-with-no-value, or a float/numeric string for
    an explicit tolerance) and falls back to the
    :data:`RELAX_FEASIBILITY_ENV_VAR` env var when the CLI is silent.

    Returns ``None`` when the user did not request relaxation
    (HiGHS keeps its defaults), otherwise a positive float tolerance.
    Invalid values (non-numeric, <=0) return ``None``.
    """
    if cli_value is None:
        raw = os.environ.get(RELAX_FEASIBILITY_ENV_VAR, "").strip()
        if not raw:
            return None
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "on"):
            return DEFAULT_RELAX_FEASIBILITY
        try:
            tol = float(raw)
        except ValueError:
            return None
        return tol if tol > 0 else None
    # CLI path.  argparse sets ``cli_value`` to the sentinel string
    # "default" when the flag is passed without ``=TOL``; otherwise it
    # is already a float.
    if cli_value == "default":
        return DEFAULT_RELAX_FEASIBILITY
    try:
        tol = float(cli_value)
    except (TypeError, ValueError):
        return None
    return tol if tol > 0 else None


def resolve_ipm(cli_flag: bool) -> bool:
    """True iff ``--ipm`` is set OR :data:`IPM_ENV_VAR` is truthy."""
    if cli_flag:
        return True
    raw = os.environ.get(IPM_ENV_VAR, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


class SolverRunner:
    """Invoke external solver binaries and manage the solve workflow."""

    def __init__(self, state: RunnerState) -> None:
        self.state = state
        self.logger = state.logger

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, current_solve: str) -> int:
        """Run the model executable once and return the process exit code."""
        timer_in_model_run = time.perf_counter()

        try:
            solver = self.state.solve.solver_settings.solvers[current_solve]
        except KeyError:
            self.logger.warning(f"No solver defined for {current_solve}. Defaulting to highs.")
            solver = "highs"

        glpsol_file, highs_file = self._platform_binaries()
        flextool_model_file = str(self.state.paths.flextool_dir / "flextool.mod")
        flextool_base_data_file = str(self.state.paths.flextool_dir / "flextool_base.dat")
        wf = self.state.paths.work_folder
        glp_solution_file = str(wf / "glpsol_solution.txt")
        mps_file = str(wf / "flextool.mps")
        highs_option_file = self._ensure_highs_opt()
        cplex_sol_file = str(wf / "cplex.sol")
        flextool_sol_file = str(wf / "flextool.sol")

        if solver == "glpsol":
            returncode = self._run_glpsol_solver(
                glpsol_file, flextool_model_file, flextool_base_data_file,
                glp_solution_file, timer_in_model_run,
            )
        elif solver in ("highs", "cplex"):
            returncode = self._run_highs_or_cplex(
                solver, current_solve, glpsol_file, highs_file,
                flextool_model_file, flextool_base_data_file,
                mps_file, highs_option_file, cplex_sol_file,
                flextool_sol_file, timer_in_model_run,
            )
        else:
            message = f"Unknown solver '{solver}'. Currently supported options: highs, glpsol, cplex."
            self.logger.error(message)
            raise FlexToolSolveError(message)

        return returncode

    # ------------------------------------------------------------------
    # Solver workflows
    # ------------------------------------------------------------------

    def _ensure_highs_opt(self) -> str:
        """Return the path to ``bin/highs.opt``, seeding it from the
        tracked template on first use.

        ``bin/highs.opt`` is a user-editable tuning file; local edits must
        persist across updates, so the repo only ships the defaults as
        ``bin/highs.opt.template``.  On every solve we check the live file
        exists; if missing, copy the template over.  This keeps the setup
        self-healing without blowing away user edits on subsequent runs.
        """
        bin_dir = self.state.paths.bin_dir
        opt_path = bin_dir / "highs.opt"
        if not opt_path.exists():
            template = bin_dir / "highs.opt.template"
            if template.exists():
                import shutil
                shutil.copy(template, opt_path)
                self.logger.info(
                    "bin/highs.opt missing; seeded from bin/highs.opt.template"
                )
        return str(opt_path)

    def _platform_binaries(self) -> tuple[str, str]:
        """Return (glpsol_path, highs_path) and set executable permissions on Linux.

        highs_path is kept for backward compatibility but is no longer used
        (HiGHS is called via highspy Python API).
        """
        if sys.platform.startswith("linux"):
            glpsol_file = str(self.state.paths.bin_dir / "glpsol")
            highs_file = str(self.state.paths.bin_dir / "highs")
        elif sys.platform.startswith("win32"):
            glpsol_file = str(self.state.paths.bin_dir / "glpsol.exe")
            highs_file = str(self.state.paths.bin_dir / "highs.exe")
        elif sys.platform == "darwin":
            glpsol_file = str(self.state.paths.bin_dir / "glpsol_macos15_arm64")
            highs_file = str(self.state.paths.bin_dir / "highs")
        else:
            glpsol_file = str(self.state.paths.bin_dir / "glpsol")
            highs_file = str(self.state.paths.bin_dir / "highs")
        if sys.platform != "win32":
            if os.path.exists(glpsol_file):
                current_permissions = os.stat(glpsol_file).st_mode & 0o777
                if current_permissions != 0o755:
                    os.chmod(glpsol_file, 0o755)
        return glpsol_file, highs_file

    def _run_glpsol_solver(
        self,
        glpsol_file: str,
        flextool_model_file: str,
        flextool_base_data_file: str,
        glp_solution_file: str,
        timer_start: float,
    ) -> int:
        """Run a pure-GLPSOL solve (no HiGHS/CPLEX)."""
        wf = self.state.paths.work_folder
        with open(wf / "solve_data/glpsol_phase.csv", 'w') as p_model_file:
            p_model_file.write("phase\nread\n")

        only_glpsol = [
            glpsol_file, '--model', flextool_model_file,
            '-d', flextool_base_data_file, '--cbg', '-w', glp_solution_file,
        ]
        try:
            returncode = self._run_glpsol(only_glpsol)
            if returncode != 0:
                raise FlexToolSolveError(f"glpsol failed with exit code: {returncode}")
        except FlexToolSolveError:
            raise
        except Exception as e:
            self.logger.exception(f"Error occurred: {e}")
            raise FlexToolSolveError(f"Error occurred: {e}") from e
        if returncode != 0:
            message = f'glpsol failed: {returncode}'
            self.logger.error(message)
            raise FlexToolSolveError(message)

        # Check if solution is infeasible
        with open(glp_solution_file, 'r', encoding='utf-8') as inf_file:
            inf_content = inf_file.read()
            if 'INFEASIBLE' in inf_content:
                message = "The model is infeasible. Check the constraints."
                self.logger.error(message)
                raise FlexToolSolveError(message)

        timing = time.perf_counter() - timer_start
        self.logger.info(f"--- Solve with GLPSOL: {timing:.4f} seconds ---")
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',,' + str(round(timing, 4)) + ',')

        return returncode

    def _run_highs_or_cplex(
        self,
        solver: str,
        current_solve: str,
        glpsol_file: str,
        highs_file: str,
        flextool_model_file: str,
        flextool_base_data_file: str,
        mps_file: str,
        highs_option_file: str,
        cplex_sol_file: str,
        flextool_sol_file: str,
        timer_start: float,
    ) -> int:
        """Run the three-phase glpsol→HiGHS/CPLEX→glpsol workflow."""
        wf = self.state.paths.work_folder
        timer_in_model_run = timer_start

        # Phase 1: GLPSOL creates MPS file
        with open(wf / "solve_data/glpsol_phase.csv", 'w') as p_model_file:
            p_model_file.write("phase\nread\n")

        highs_step1 = [
            glpsol_file, '--check', '--model', flextool_model_file,
            '-d', flextool_base_data_file, '--wfreemps', mps_file,
        ]
        returncode = self._run_glpsol(highs_step1)
        if returncode != 0:
            raise FlexToolSolveError(f"glpsol MPS generation failed with exit code: {returncode}")

        # Check if the problem has columns (nodes)
        with open(mps_file, 'r', encoding='utf-8') as mps_file_handle:
            mps_content = mps_file_handle.read()
            if 'Columns:    0' in mps_content:
                message = (
                    "The problem has no columns. Check that the model has nodes "
                    "with entity alternative: true"
                )
                self.logger.error(message)
                raise FlexToolSolveError(message)

        timing = time.perf_counter() - timer_in_model_run
        self.logger.info(f"--- GLPSOL created sol file: {timing:.4f} seconds ---")
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',' + str(round(timing, 4)))
        timer_in_model_run = timer_in_model_run + timing

        # Phase 2: Run solver
        highs_instance = None  # kept alive for post-phase-3 handoff extraction
        if solver == "highs":
            highs_instance = self._run_highs(current_solve, highs_file, mps_file, highs_option_file, flextool_sol_file)
            timing = time.perf_counter() - timer_in_model_run
            self.logger.info(f"--- Solver (HiGHS): {timing:.4f} seconds ---")
            with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
                solve_progress.write(',' + str(round(timing, 4)))
            timer_in_model_run = timer_in_model_run + timing

        elif solver == "cplex":
            with open(wf / "solve_data/glpsol_phase.csv", 'w') as p_model_file:
                p_model_file.write("phase\nread\n")
            self._run_cplex(current_solve, mps_file, cplex_sol_file, flextool_sol_file)
            timing = time.perf_counter() - timer_in_model_run
            self.logger.info(f"--- Solver (CPLEX or Gurobi): {timing:.4f} seconds ---")
            with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
                solve_progress.write(',' + str(round(timing, 4)))
            timer_in_model_run = timer_in_model_run + timing

        # Phase 3: glpsol re-reads the model + ``.sol`` and writes the
        # legacy ``output_raw/*.csv`` dumps.  Only runs under
        # ``--use-old-raw-csv``; the Python writers
        # (:mod:`read_highs_solution` + :mod:`handoff_writers`) have
        # replaced phase 3 in the default path.
        returncode = 0
        if self.state.use_old_raw_csv:
            returncode = self._run_phase_3(
                glpsol_file, flextool_model_file, flextool_base_data_file,
                flextool_sol_file, wf, timer_in_model_run,
            )
            if returncode != 0:
                raise FlexToolSolveError(
                    f"glpsol output writing failed with exit code: {returncode}"
                )

        # Phase 4 (HiGHS-only): extract outputs directly from the live
        # ``Highs`` instance.  Always runs when HiGHS is the solver —
        # parquets are the new pathway and ``--use-old-raw-csv`` only
        # affects whether phase 3 ALSO writes its legacy CSVs.
        if highs_instance is not None:
            from flextool.process_outputs.read_highs_solution import (
                _actual_solve_name, write_all_variables,
            )
            from flextool.process_outputs.handoff_writers import write_all_handoffs
            # In rolling / nested scenarios ``current_solve`` is the PARENT
            # complete-solve while every CSV in ``solve_data/`` keys its
            # ``solve`` column off the ROLL name (``solve_current`` set in
            # the model).  Use that same roll name so parquet files are
            # per-roll and downstream CSV joins line up byte-for-byte.
            roll_name = _actual_solve_name(wf, current_solve)
            try:
                write_all_variables(
                    highs_instance,
                    solve_name=roll_name,
                    output_dir=wf / "output_raw",
                    realized_dispatch_csv=wf / "solve_data/realized_dispatch.csv",
                    realized_periods_csv=wf / "solve_data/realized_invest_periods_of_current_solve.csv",
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"parquet variable extraction failed: {exc}")
            try:
                write_all_handoffs(
                    highs_instance,
                    solve_name=roll_name,
                    work_folder=wf,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"handoff writers failed: {exc}")
            try:
                from flextool.process_outputs.cumulative_handoffs import (
                    write_cumulative_handoffs,
                )
                write_cumulative_handoffs(
                    highs_instance,
                    solve_name=roll_name,
                    work_folder=wf,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"cumulative handoff writers failed: {exc}")

        return returncode

    def _run_phase_3(
        self,
        glpsol_file: str,
        flextool_model_file: str,
        flextool_base_data_file: str,
        flextool_sol_file: str,
        wf,  # pathlib.Path
        timer_in_model_run: float,
    ) -> int:
        """Invoke glpsol phase 3: re-read model + ``.sol`` and write the
        legacy ``output_raw/*.csv`` dumps.

        Only called when ``state.use_old_raw_csv`` is set — the new
        pathway synthesises these outputs directly (pre-solve printfs
        for parameters/sets, :mod:`read_highs_solution` for solver values,
        :mod:`handoff_writers` for per-entity capacity).
        """
        with open(wf / "solve_data/glpsol_phase.csv", 'w') as p_model_file:
            p_model_file.write("phase\nwrite\n")

        highs_step3 = [
            glpsol_file, '--model', flextool_model_file,
            '-d', flextool_base_data_file, '-r', flextool_sol_file,
        ]
        returncode = self._run_glpsol(highs_step3)

        timing = time.perf_counter() - timer_in_model_run
        self.logger.info(f"--- GLPSOL wrote outputs: {timing:.4f} seconds ---")
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',' + str(round(timing, 4)) + '\n')

        return returncode

    def _run_highs(
        self,
        current_solve: str,
        highs_file: str,
        mps_file: str,
        highs_option_file: str,
        flextool_sol_file: str,
    ) -> "highspy.Highs":
        """Run HiGHS solver on an MPS file via highspy Python API.

        Returns the live ``Highs`` instance so the caller (still inside
        ``_run_highs_or_cplex``) can run the post-phase-3 handoff
        extraction without re-reading the model.
        """
        wf = self.state.paths.work_folder
        h = highspy.Highs()

        # Apply options from highs.opt file
        # Paths for solution_file and log_file are made absolute to work folder
        # Solution file writing options are tracked but not passed to HiGHS,
        # because the highspy API does not honor them — writeSolution() is
        # called explicitly after a successful solve instead.
        #
        # Agent 15 (LP-scaling): we also record which option keys the opt
        # file set so the subsequent DB/default overrides can skip
        # ``parallel`` / ``threads`` when the opt file already specified
        # them.  This unifies the highspy in-memory path with the external
        # HiGHS binary path (both now honor the opt file for determinism
        # knobs).  Production ``bin/highs.opt`` sets neither key, so the
        # production defaults (parallel=on, threads=4) are unchanged; the
        # test harness ``tests/highs.opt`` sets ``threads=1`` and now
        # actually gets it, eliminating alternate-optimum non-determinism
        # across full-suite runs.
        _PATH_OPTIONS = {'solution_file', 'log_file'}
        _SOLUTION_WRITE_OPTIONS = {'write_solution_to_file', 'solution_file', 'write_solution_style'}
        solution_style = 2  # default: glpsol-compatible style
        keys_from_opt: set[str] = set()
        if os.path.exists(highs_option_file):
            with open(highs_option_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    key, _, value = line.partition('=')
                    key, value = key.strip(), value.strip()
                    if key == 'write_solution_style':
                        solution_style = int(value)
                    if key in _SOLUTION_WRITE_OPTIONS:
                        continue
                    if key in _PATH_OPTIONS:
                        value = str(wf / value)
                    h.setOptionValue(key, self._parse_highs_option(value))
                    keys_from_opt.add(key)

        # Ensure log goes to work folder
        h.setOptionValue('log_file', str(wf / 'HiGHS.log'))

        # Apply per-solve overrides from database.  DB values always win
        # over the opt file; defaults apply only when neither DB nor opt
        # file has spoken.
        h.setOptionValue('presolve', self.state.solve.highs.presolve.get(current_solve, 'on'))
        h.setOptionValue('solver', self.state.solve.highs.method.get(current_solve, 'choose'))

        # Agent 18d (LP-scaling): user-facing solver-option knobs — IPM
        # switch + feasibility-tolerance relaxation.  Both are orthogonal
        # to Agents 5/8/12/18b/18c scaling infrastructure and aimed at
        # rivendell-shaped wide-bound models whose default-tolerance
        # simplex solve stalls on sub-tolerance residuals.  CLI/env-var
        # precedence matches Agent 9's ``FLEXTOOL_FORCE_ROW_SCALING``
        # pattern: an explicit value wins over any DB/opt-file default.
        relax_feasibility = getattr(self.state, 'relax_feasibility', None)
        if relax_feasibility is not None:
            h.setOptionValue('primal_feasibility_tolerance', float(relax_feasibility))
            h.setOptionValue('dual_feasibility_tolerance', float(relax_feasibility))
        use_ipm = getattr(self.state, 'use_ipm', False)
        if use_ipm:
            h.setOptionValue('solver', 'ipm')
        if relax_feasibility is not None or use_ipm:
            solver_opt = 'ipm' if use_ipm else self.state.solve.highs.method.get(current_solve, 'choose')
            tol_str = (
                f"{float(relax_feasibility):g}" if relax_feasibility is not None
                else "default"
            )
            self.logger.info(
                "Solver options: primal_feasibility_tolerance=%s, "
                "dual_feasibility_tolerance=%s, solver=%s",
                tol_str, tol_str, solver_opt,
            )
        # Parallel dual simplex (PAMI) off by default. HiGHS 1.11, 1.12, and
        # 1.14 all stall indefinitely on a class of degenerate LPs that
        # reach ``Pr=0 Du=0`` and then have a tiny residual reappear during
        # the post-optimality rescan: concurrent simplex trajectories
        # cannot agree on the clearing pivot, the Markowitz-bump loop runs
        # forever. Serial EKK takes exactly one extra pivot and terminates.
        # HiGHS upstream (issue #1547, open since 2024) and maintainer
        # @jajhall's own recommendation in #780 / #1044 is ``threads=1`` /
        # ``parallel=off``. Note that ``threads=1`` alone is NOT enough:
        # HiGHS still engages PAMI at hardcoded concurrency 8 against a
        # single thread — the worst oversubscription regime. We must also
        # set ``parallel=off`` explicitly. Precedence: DB > opt file > default.
        if current_solve in self.state.solve.highs.parallel:
            h.setOptionValue('parallel', self.state.solve.highs.parallel[current_solve])
        elif 'parallel' not in keys_from_opt:
            h.setOptionValue('parallel', 'off')
        # Thread count precedence: CLI/runner override > opt file > default (1).
        # ``state.highs_threads`` is only non-None when a caller (e.g. the CLI
        # ``--highs-threads`` flag) explicitly set it; the default-init path
        # leaves it unset so the opt file can win in test context.
        highs_threads = getattr(self.state, 'highs_threads', None)
        if highs_threads is not None:
            h.setOptionValue('threads', int(highs_threads))
        elif 'threads' not in keys_from_opt:
            h.setOptionValue('threads', 1)

        # Read and solve
        status = h.readModel(mps_file)
        if status != highspy.HighsStatus.kOk:
            message = f'HiGHS failed to read model: {mps_file}'
            self.logger.error(message)
            raise FlexToolSolveError(message)

        # Agent 18c (LP-scaling): variable-bound scaling via HiGHS's
        # built-in ``user_bound_scale`` option.  Row scaling (Agent 5 /
        # 18b) cannot compress the variable-bound spread — on models
        # like rivendell the bound range stays at ~9 decades despite
        # row scaling, and HiGHS itself hints "Consider setting the
        # user_bound_scale option to -8".  This block queries the
        # loaded LP's column bounds, decides an integer ``N``, and
        # calls ``h.setOptionValue('user_bound_scale', N)`` before
        # ``h.run()``.  HiGHS multiplies bounds by ``2^N`` internally
        # and un-scales on output — solution-invariant.
        #
        # Precedence mirrors Agent 18b's row-scaling decision: explicit
        # user setting in ``highs.opt`` wins; env-var force override is
        # next (test hook); analyser auto-decision only fires when
        # ``--auto-scale`` is active.
        try:
            lp = h.getLp()
            col_lower = list(lp.col_lower_)
            col_upper = list(lp.col_upper_)
        except Exception as exc:  # defensive — never fail the solve
            self.logger.warning(
                f"[scaling] could not query LP bounds for bound scaling: {exc}"
            )
            col_lower, col_upper = [], []
        auto_scale = getattr(self.state, 'auto_scale', False)
        # The analyser caches its ScaleTable under the roll name (``solve``
        # in orchestration's loop), which can differ from ``current_solve``
        # (= ``complete_solve[solve]``) in rolling / nested scenarios.
        # Fall back to ``current_solve`` when the orchestration hook has
        # not been set (e.g. direct ``SolverRunner.run`` call in tests).
        scale_key = getattr(self.state, 'current_scale_solve_name', None) or current_solve
        n_bound, bound_min, bound_max, bound_spread, source = apply_bound_scale_decision(
            solve_name=scale_key,
            col_lower=col_lower,
            col_upper=col_upper,
            auto_scale=auto_scale,
            user_opt_set=('user_bound_scale' in keys_from_opt),
            logger=self.logger,
        )
        if n_bound != 0:
            h.setOptionValue('user_bound_scale', int(n_bound))
        update_bound_scale_in_cache(
            scale_key, n_bound, bound_min, bound_max, bound_spread,
        )

        status = h.run()
        model_status = h.getModelStatus()

        if model_status == highspy.HighsModelStatus.kInfeasible:
            message = "The model is infeasible. Check the constraints."
            self.logger.error(message)
            raise FlexToolSolveError(message)

        if status != highspy.HighsStatus.kOk:
            message = f'HiGHS solver failed with status: {model_status}'
            self.logger.error(message)
            raise FlexToolSolveError(message)

        # Write solution file explicitly — the highspy API does not honor the
        # write_solution_to_file / solution_file options from highs.opt.
        h.writeSolution(flextool_sol_file, solution_style)
        self.logger.debug("HiGHS solved the problem")

        # NOTE: parquet extraction moved out of this method — it now runs
        # AFTER phase 3 inside ``_run_highs_or_cplex`` so that the
        # Category C custom writers (v_dual_node_balance, CO2 duals)
        # can read the parameter CSVs phase 3 produces.  The live
        # ``Highs`` instance is returned so the caller can do the
        # extraction at the right point.
        #
        # HiGHS-only by construction — only ``_run_highs`` creates a
        # ``Highs`` instance.  CPLEX takes a different path.
        return h

    @staticmethod
    def _parse_highs_option(value: str) -> int | float | str:
        """Parse a HiGHS option value string to the appropriate Python type."""
        if value.lower() in ('true', 'false'):
            return value.lower()
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def _run_cplex(
        self,
        current_solve: str,
        mps_file: str,
        cplex_sol_file: str,
        flextool_sol_file: str,
    ) -> None:
        """Run CPLEX solver on an MPS file.  (S10: consolidated precommand handling)"""
        # Build the CPLEX command once
        cplex_cmd: list[str] = ['cplex', '-c', 'read', mps_file]
        if current_solve in self.state.solve.solver_settings.arguments:
            cplex_cmd += self.state.solve.solver_settings.arguments[current_solve]
        cplex_cmd += ['opt', 'write', cplex_sol_file, 'quit']
        # Conditionally prepend wrapper
        if current_solve in self.state.solve.solver_settings.precommand:
            cplex_cmd = [self.state.solve.solver_settings.precommand[current_solve]] + cplex_cmd

        completed = subprocess.run(cplex_cmd, cwd=str(self.state.paths.work_folder))
        if completed.returncode != 0:
            message = f'Cplex solver failed: {completed.returncode}'
            self.logger.error(message)
            raise FlexToolSolveError(message)

        self._cplex_to_glpsol(cplex_sol_file, flextool_sol_file)

    # ------------------------------------------------------------------
    # GLPSOL subprocess with filtered output
    # ------------------------------------------------------------------

    def _run_glpsol(self, command_args: list[str]) -> int:
        """Run glpsol with filtered output and return the process exit code."""
        process = subprocess.Popen(
            command_args, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(self.state.paths.work_folder),
        )

        buffer: list[str] = []
        previous: str | bool = False
        counter = 0
        already_stripped_end_of_line = True

        for line in process.stdout:
            if line.startswith('Reading data...'):
                continue

            # Take note if Generating has been followed by Display statement
            if line.startswith('Timer - '):
                if already_stripped_end_of_line:
                    print(line.rstrip(), end='  ')
                else:
                    print('\n' + line.rstrip(), end='  ')
                counter = 0
                continue

            if line.startswith('Generating ') or line.startswith('Write '):
                previous = 'generate'
                counter += 1
                already_stripped_end_of_line = False
                if line.startswith('Generating '):
                    if command_args[5] == '-r':
                        continue
                    line = line.replace('Generating ', '  ').rstrip()
                elif line.startswith('Write '):
                    line = line.replace('Write ', '  ').rstrip()
                if counter == 3:
                    line = line + '\n'
                    counter = 0
                    already_stripped_end_of_line = True
                print(line, end='')
                continue

            if line.startswith('Checking'):
                if line.startswith('Checking:'):
                    previous = 'check'
                    buffer = [line]
            elif previous == 'check':
                buffer.append(line)
                if 'error' in line.lower() or 'failed' in line.lower() or 'assertion' in line.lower():
                    output = '\n' + ''.join(buffer)
                    output = output.replace('Checking (line', ' (flextool/flextool.mod line')
                    print(output)
                    buffer = []
                    previous = False
                elif line.strip() == '' or (not line.startswith(' ') and not line.startswith('Created')):
                    buffer = []
                    previous = False
            else:
                print(line, end='')

        process.wait()

        if process.returncode != 0 and self.logger:
            self.logger.error(f'glpsol failed with exit code: {process.returncode}')

        return process.returncode

    # ------------------------------------------------------------------
    # CPLEX solution → GLPSOL format conversion
    # ------------------------------------------------------------------

    def _cplex_to_glpsol(self, cplexfile: str, solutionfile: str) -> int:
        """Convert CPLEX XML solution to GLPSOL solution format.

        S03: shared helper ``_write_glpsol_solution`` eliminates the
        near-duplicate branches for 'optimal' vs 'integer optimal solution'.
        """
        try:
            tree = ET.parse(cplexfile)
        except OSError:
            message = (
                'The CPLEX solver does not produce a solution file if the problem '
                'is infeasible. Check the constraints, more info at cplex.log'
            )
            self.logger.error(message)
            raise FlexToolSolveError(message)
        root = tree.getroot()

        status = root.find('header').get('solutionStatusString')
        if status == "optimal":
            self._write_glpsol_solution(root, solutionfile, is_mip=False)
        elif status == "integer optimal solution":
            self._write_glpsol_solution(root, solutionfile, is_mip=True)
        else:
            message = "Optimality could not be reached. Check the flextool.sol file for more"
            self.logger.error(message)
            raise FlexToolSolveError(message)

        return 0

    @staticmethod
    def _write_glpsol_solution(
        root: ET.Element, solutionfile: str, *, is_mip: bool,
    ) -> None:
        """Write a GLPSOL-format solution file from a parsed CPLEX XML tree.

        S03: shared helper for both LP-optimal and MIP-optimal branches.
        """
        obj = root.find('header').get('objectiveValue')

        # Count rows and columns from last index
        for constraint in root.iter('constraint'):
            rows = constraint.get('index')
        rows = int(rows) + 2

        for variable in root.iter('variable'):
            col = variable.get('index')
        col = int(col) + 1

        with open(solutionfile, 'w') as f:
            if is_mip:
                f.write(f"s mip {rows} {col} o {obj}\n")
                # First constraint row is the objective function value
                f.write(f"i 1 {obj}\n")
                for constraint in root.iter("constraint"):
                    slack = constraint.get('slack')
                    index = int(constraint.get('index')) + 2
                    f.write(f"i {index} {slack}\n")
                for variable in root.iter('variable'):
                    val = variable.get('value')
                    index = int(variable.get('index')) + 1
                    f.write(f"j {index} {val}\n")
            else:
                f.write(f"s bas {rows} {col} f f {obj}\n")
                # First constraint row is the objective function value
                f.write(f"i 1 b {obj} 0\n")
                for constraint in root.iter("constraint"):
                    slack = constraint.get('slack')
                    index = int(constraint.get('index')) + 2
                    status = constraint.get('status')
                    dual = constraint.get('dual')
                    status = {'BS': 'b', 'LL': 'l', 'UL': 'u'}.get(status, status)
                    f.write(f"i {index} {status} {slack} {dual}\n")
                for variable in root.iter('variable'):
                    val = variable.get('value')
                    index = int(variable.get('index')) + 1
                    status = variable.get('status')
                    reduced = variable.get('reducedCost')
                    status = {'BS': 'b', 'LL': 'l', 'UL': 'u'}.get(status, status)
                    f.write(f"j {index} {status} {val} {reduced}\n")
            f.write("e o f")
