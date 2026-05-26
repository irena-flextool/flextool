"""Subprocess-HiGHS driver for --save-memory.

The default HiGHS path in :mod:`flextool.engine_polars._solver_dispatch`
runs in-process via ``highspy``.  For large models that don't fit in
RAM when stacked alongside FlexTool's preprocessing state, this module
offers an opt-in alternative: build the LP into a temp MPS file via
:meth:`polar_high.Problem.build_only`, drop everything polar-side AND
the live HiGHS instance, then spawn :mod:`flextool.cli.cmd_solve_mps`
as a subprocess to actually solve.

The child process has a clean address space — none of FlexTool's
~7-11 GB of polars frames, no glibc fragmentation from upstream
preprocessing.  When it finishes, the parent reads the solution back
via a fresh ``highspy.Highs`` (read-only, lightweight: just the LP
storage + the produced solution arrays) and wraps it in a
``polar_high.Solution`` that downstream output writers consume
identically to an in-process solve.

Loses warm-LP reuse for the cascade — the Problem is in ``_released``
state after ``build_only`` and can't be resolved.  Already documented
on the ``--save-memory`` flag.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from polar_high import Problem, Solution


def _format_opt_value(v: object) -> str:
    """Render *v* in HiGHS .opt file syntax (``key=value`` per line)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _read_objective_from_sol(sol_path: Path) -> float:
    """Parse the ``Objective <value>`` line from a HiGHS style=0 sol file.

    ``highspy.Highs.getObjectiveValue()`` only reflects the most recent
    ``run()`` and is zero after a bare ``readModel + readSolution``, so
    we read the value the subprocess wrote directly.  Returns ``0.0``
    when the line is absent (caller should treat as non-optimal).
    """
    try:
        with open(sol_path) as f:
            for line in f:
                if line.startswith("Objective "):
                    return float(line.split(None, 1)[1])
    except (OSError, ValueError):
        pass
    return 0.0


def solve_via_subprocess(
    problem: "Problem",
    options: dict[str, Any] | None,
    *,
    solve_name: str,
    logger: logging.Logger | None = None,
    work_folder: Path | None = None,
) -> "Solution":
    """Solve *problem* via a HiGHS subprocess; return a Solution.

    ``options`` is the effective HiGHS solver-option dict (post any
    ``build_solver_options`` translation).  ``solve_name`` is used to
    name the temp MPS/sol files.  ``work_folder``, when given, holds
    the files under ``<work_folder>/solve_data/subprocess/`` for post-
    mortem inspection on failure; ``None`` uses a self-cleaning
    tempdir.
    """
    import highspy
    from polar_high import Solution

    cleanup = work_folder is None
    if work_folder is not None:
        out_dir = Path(work_folder) / "solve_data" / "subprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="flextool_subprocess_"))

    safe_name = solve_name.replace("/", "_").replace(" ", "_") or "solve"
    mps_path = out_dir / f"{safe_name}.mps"
    sol_path = out_dir / f"{safe_name}.sol"
    opts_path = out_dir / f"{safe_name}.opt"

    try:
        # Write effective options as a HiGHS .opt file.
        opts = options or {}
        with open(opts_path, "w") as f:
            for k, v in opts.items():
                f.write(f"{k}={_format_opt_value(v)}\n")

        if logger is not None:
            logger.info(
                "save_memory: building LP for %r, writing MPS to %s",
                solve_name, mps_path,
            )

        # Build LP into HiGHS, write MPS, release polar-side AND the
        # live HiGHS instance.  The Problem is in _released state from
        # here on — warm-LP reuse is impossible.  Var.col_id frames
        # survive on problem._vars so we can construct a Solution
        # below from the subprocess's col_value array.
        problem.build_only(str(mps_path), options=opts)

        cmd = [
            sys.executable, "-m", "flextool.cli.cmd_solve_mps",
            "--mps", str(mps_path),
            "--solution", str(sol_path),
            "--options", str(opts_path),
        ]
        if logger is not None:
            logger.info(
                "save_memory: spawning subprocess HiGHS for %r", solve_name,
            )
        # ``stdout`` / ``stderr`` pass through so HiGHS' own banner +
        # progress lines appear in the parent's log, matching the
        # in-process feel.
        cp = subprocess.run(cmd)
        # cmd_solve_mps exit codes:
        #   0 = kOptimal
        #   1 = non-optimal but completed (solution still written)
        #  >1 = hard error (no solution to read back)
        optimal = cp.returncode == 0
        if cp.returncode > 1:
            raise RuntimeError(
                f"subprocess HiGHS for solve {solve_name!r} failed with "
                f"exit code {cp.returncode}; MPS+options preserved at "
                f"{out_dir} for inspection"
            )

        if logger is not None:
            logger.info(
                "save_memory: subprocess complete (exit=%d, optimal=%s); "
                "reading solution from %s",
                cp.returncode, optimal, sol_path,
            )

        # Parent-side read-back.  This fresh Highs holds the LP and
        # the solution but never runs the simplex/IPM — so peak RSS
        # here is the LP storage (~5-15 GB for big models) plus the
        # solution arrays.  Far below the ~50 GB the child needed
        # during its actual solve.
        h = highspy.Highs()
        try:
            h.silent()
        except Exception:
            pass
        ok = (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)
        if h.readModel(str(mps_path)) not in ok:
            raise RuntimeError(
                f"parent failed to read MPS back from {mps_path} "
                f"after subprocess solve",
            )
        if h.readSolution(str(sol_path), 0) not in ok:
            raise RuntimeError(
                f"parent failed to read solution from {sol_path}",
            )

        hs = h.getSolution()
        col_value = np.asarray(hs.col_value, dtype=np.float64)
        n_cols = len(col_value)
        # ``readSolution`` populates duals only when the solution file
        # carries them.  HiGHS' style=0 writer includes duals for LPs;
        # default to zeros otherwise so callers see uniform shapes.
        row_dual = (
            np.asarray(hs.row_dual, dtype=np.float64)
            if hs.row_dual else np.zeros(0, dtype=np.float64)
        )
        col_dual = (
            np.asarray(hs.col_dual, dtype=np.float64)
            if hs.col_dual else np.zeros(n_cols, dtype=np.float64)
        )
        # See _read_objective_from_sol — getObjectiveValue() is zero
        # after readModel+readSolution (it only reflects the last run()),
        # so read the value the subprocess wrote to the .sol file.
        obj = _read_objective_from_sol(sol_path)

        # The Solution carries:
        # - col_value / duals → from the subprocess via the fresh h
        # - vars=problem._vars → Var.frame col_id maps survived
        #   build_only's _release_python_lp_inputs (see polar-high
        #   engine.py)
        # - highs=h → the fresh, read-only Highs handle for output
        #   writers (handoff_writers, read_highs_solution) that
        #   consume .getLp() / .allVariableNames() / .getSolution()
        # - col_names / row_names empty (same convention as
        #   polar-high's in-process save_memory path)
        return Solution(
            optimal=optimal,
            obj=obj,
            col_value=col_value,
            row_dual=row_dual,
            col_dual=col_dual,
            col_names=[],
            row_names=[],
            vars=dict(problem._vars),
            highs=h,
        )
    finally:
        if cleanup:
            for p in (mps_path, sol_path, opts_path):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
            try:
                out_dir.rmdir()
            except OSError:
                pass


__all__ = ["solve_via_subprocess"]
