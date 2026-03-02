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
from typing import IO

from flextool.flextoolrunner.runner_state import RunnerState, FlexToolSolveError


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
        glp_solution_file = str(self.state.paths.root_dir / "glpsol_solution.txt")
        mps_file = str(self.state.paths.root_dir / "flextool.mps")
        highs_option_file = str(self.state.paths.bin_dir / "highs.opt")
        cplex_sol_file = str(self.state.paths.root_dir / "cplex.sol")
        flextool_sol_file = str(self.state.paths.root_dir / "flextool.sol")

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

    def _platform_binaries(self) -> tuple[str, str]:
        """Return (glpsol_path, highs_path) and set executable permissions on Linux."""
        if sys.platform.startswith("linux"):
            glpsol_file = str(self.state.paths.bin_dir / "glpsol")
            highs_file = str(self.state.paths.bin_dir / "highs")
            for binary in (glpsol_file, highs_file):
                if os.path.exists(binary):
                    current_permissions = os.stat(binary).st_mode & 0o777
                    if current_permissions != 0o755:
                        os.chmod(binary, 0o755)
        elif sys.platform.startswith("win32"):
            glpsol_file = str(self.state.paths.bin_dir / "glpsol.exe")
            highs_file = str(self.state.paths.bin_dir / "highs.exe")
        else:
            glpsol_file = str(self.state.paths.bin_dir / "glpsol")
            highs_file = str(self.state.paths.bin_dir / "highs")
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
        with open("solve_data/glpsol_phase.csv", 'w') as p_model_file:
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
        with open('glpsol_solution.txt', 'r') as inf_file:
            inf_content = inf_file.read()
            if 'INFEASIBLE' in inf_content:
                message = "The model is infeasible. Check the constraints."
                self.logger.error(message)
                raise FlexToolSolveError(message)

        timing = time.perf_counter() - timer_start
        self.logger.info(f"--- Solve with GLPSOL: {timing:.4f} seconds ---")
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
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
        timer_in_model_run = timer_start

        # Phase 1: GLPSOL creates MPS file
        with open("solve_data/glpsol_phase.csv", 'w') as p_model_file:
            p_model_file.write("phase\nread\n")

        highs_step1 = [
            glpsol_file, '--check', '--model', flextool_model_file,
            '-d', flextool_base_data_file, '--wfreemps', mps_file,
        ]
        returncode = self._run_glpsol(highs_step1)
        if returncode != 0:
            raise FlexToolSolveError(f"glpsol MPS generation failed with exit code: {returncode}")

        # Check if the problem has columns (nodes)
        with open(mps_file, 'r') as mps_file_handle:
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
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',' + str(round(timing, 4)))
        timer_in_model_run = timer_in_model_run + timing

        # Phase 2: Run solver
        if solver == "highs":
            self._run_highs(current_solve, highs_file, mps_file, highs_option_file)
            timing = time.perf_counter() - timer_in_model_run
            self.logger.info(f"--- Solver (HiGHS): {timing:.4f} seconds ---")
            with open("solve_data/solve_progress.csv", "a") as solve_progress:
                solve_progress.write(',' + str(round(timing, 4)))
            timer_in_model_run = timer_in_model_run + timing

        elif solver == "cplex":
            with open("solve_data/glpsol_phase.csv", 'w') as p_model_file:
                p_model_file.write("phase\nread\n")
            self._run_cplex(current_solve, mps_file, cplex_sol_file, flextool_sol_file)
            timing = time.perf_counter() - timer_in_model_run
            self.logger.info(f"--- Solver (CPLEX or Gurobi): {timing:.4f} seconds ---")
            with open("solve_data/solve_progress.csv", "a") as solve_progress:
                solve_progress.write(',' + str(round(timing, 4)))
            timer_in_model_run = timer_in_model_run + timing

        # Phase 3: GLPSOL writes outputs
        with open("solve_data/glpsol_phase.csv", 'w') as p_model_file:
            p_model_file.write("phase\nwrite\n")

        highs_step3 = [
            glpsol_file, '--model', flextool_model_file,
            '-d', flextool_base_data_file, '-r', flextool_sol_file,
        ]
        returncode = self._run_glpsol(highs_step3)

        timing = time.perf_counter() - timer_in_model_run
        self.logger.info(f"--- GLPSOL wrote outputs: {timing:.4f} seconds ---")
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write(',' + str(round(timing, 4)) + '\n')

        if returncode != 0:
            raise FlexToolSolveError(f"glpsol output writing failed with exit code: {returncode}")

        return returncode

    def _run_highs(
        self,
        current_solve: str,
        highs_file: str,
        mps_file: str,
        highs_option_file: str,
    ) -> None:
        """Run HiGHS solver on an MPS file."""
        highs_step2 = [
            highs_file, mps_file, f"--options_file={highs_option_file}",
            f"--presolve={self.state.solve.highs.presolve.get(current_solve, 'on')}",
            f"--solver={self.state.solve.highs.method.get(current_solve, 'choose')}",
            f"--parallel={self.state.solve.highs.parallel.get(current_solve, 'off')}",
        ]
        completed = subprocess.run(highs_step2)
        if completed.returncode != 0:
            message = f'Highs solver failed: {completed.returncode}'
            self.logger.error(message)
            raise FlexToolSolveError(message)
        self.logger.info("HiGHS solved the problem")

        # Check if solution is infeasible
        with open('HiGHS.log', 'r') as inf_file:
            inf_content = inf_file.read()
            if 'Infeasible' in inf_content:
                message = "The model is infeasible. Check the constraints."
                self.logger.error(message)
                raise FlexToolSolveError(message)

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

        completed = subprocess.run(cplex_cmd)
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
