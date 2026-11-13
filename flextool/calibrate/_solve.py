"""Shell out to ``cmd_run_flextool`` for one calibrator iteration.

The calibrator does **not** solve in-process: it launches a fresh
:mod:`flextool.cli.cmd_run_flextool` subprocess per iteration (a clean
address space per solve, matching how the model is run in production) and
then reads the produced parquet outputs.  This module owns that launch —
building the argv (with the warm-start flags), capturing the launch
time (needed by the solve-success detector's freshness check), and running
the subprocess with its stdout+stderr merged into one captured stream.

Warm start
----------
Warm start is requested with the ``--warm-start --basis-cache-dir <dir>``
flags of the child run; the child inherits the calibrator's environment
unchanged, so an environment variable the user set reaches it as set.  A
*stable* basis-cache directory shared across iterations lets HiGHS reuse
the previous iteration's basis when the structural model is unchanged (the
adder is RHS-only, so the warm-start fingerprint is stable across
iterations).

``--save-memory`` is deliberately **not** passed: it releases the live
HiGHS instance after each sub-solve and so DISABLES warm-LP reuse.  It must
also stay constant across every iteration — flipping it mid-run would
invalidate the shared basis cache.

This module does not judge success: it returns the raw
:class:`SolveRun` and lets the loop call
:func:`flextool.calibrate.assess_solve` so the loop owns the
``required_outputs`` choice.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from flextool._solver_config_source import split_solver_config_dirs
from flextool.run_options import Resolved, RunOptions
from flextool._child_python import child_env, python_module_argv

# Repo root = <root>/flextool/calibrate/_solve.py → parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SolveRun:
    """Raw record of one ``cmd_run_flextool`` subprocess.

    ``returncode`` — the subprocess exit code (a weak success signal; see
                     :mod:`flextool.calibrate._solve_status`).
    ``started_at`` — POSIX wall-clock time captured immediately before the
                     subprocess launched, for the detector's freshness check.
    ``assess_dir`` — the directory that directly holds this run's result
                     parquets (``<out_root>/output_parquet/<scenario>``).
    ``stdout``     — the merged stdout+stderr text of the subprocess.
    """

    returncode: int
    started_at: float
    assess_dir: Path
    stdout: str


def run_solve(
    url: str,
    scenario: str,
    *,
    work_dir: Path,
    out_root: Path,
    cache_dir: Path,
    threads: int | None = None,
    launcher: str = "calibrator",
    solver_config_dir: Path | Sequence[Path] | None = None,
    workspace_solver_config_dir: Path | Sequence[Path] | None = None,
) -> SolveRun:
    """Run one calibrator solve and return its raw :class:`SolveRun`.

    Parameters
    ----------
    url:
        Input SpineDB — a bare path (promoted to ``sqlite:///``) or a full
        SQLAlchemy URL.
    scenario:
        The model scenario to solve.
    work_dir:
        Working directory for the subprocess's intermediate files
        (``--work-folder``).
    out_root:
        Output-location root (``--output-location``); results land under
        ``out_root/output_parquet/<scenario>/``.
    cache_dir:
        Warm-start basis-cache directory, shared across iterations
        (``--basis-cache-dir``).  Keep it stable across the whole
        calibration run so HiGHS can reuse the prior iteration's basis.
    threads:
        Thread budget of each solve (``--threads``); ``None`` sends no flag (the child inherits the
        environment's ``FLEXTOOL_THREADS``, else uses the available cores).
    launcher:
        Who runs the solve, recorded in the run's ``_run_status.json`` (``--run-launcher``): ``calibrator``, or
        ``netload`` for the representative-period iteration.
    solver_config_dir:
        Folder, or folders highest first, of the solver option files (``--solver-config-dir`` once per folder);
        ``None`` sends no flag (the child resolves the folders itself: ``FLEXTOOL_SOLVER_CONFIG_DIR``, else from
        the input database's location).
    workspace_solver_config_dir:
        The folders the FlexTool window found, highest first (``--workspace-solver-config-dir`` once per folder:
        each trusted only when this computer set its workspace or project up); ``None`` sends no flag.

    The solve runs under the ``flextool-run`` supervisor, which writes the run's ``_run.log``; its output still
    comes back here in ``stdout``.  The child inherits this process's environment, plus ``PYTHONSAFEPATH=1``
    (:mod:`flextool._child_python`).
    """
    argv = solve_argv(
        url, scenario, work_dir=work_dir, out_root=out_root, cache_dir=cache_dir, threads=threads, launcher=launcher,
        solver_config_dir=solver_config_dir, workspace_solver_config_dir=workspace_solver_config_dir,
    )
    assess_dir = Path(out_root) / "output_parquet" / scenario

    started_at = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(_REPO_ROOT),
        env=child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return SolveRun(
        returncode=proc.returncode,
        started_at=started_at,
        assess_dir=assess_dir,
        stdout=proc.stdout or "",
    )


def solve_argv(
    url: str,
    scenario: str,
    *,
    work_dir: Path,
    out_root: Path,
    cache_dir: Path,
    threads: int | None = None,
    launcher: str = "calibrator",
    solver_config_dir: Path | Sequence[Path] | None = None,
    workspace_solver_config_dir: Path | Sequence[Path] | None = None,
) -> list[str]:
    """The command line of one calibrator iteration (parameters as :func:`run_solve`).

    ``python -m flextool_run_supervisor`` followed by ``cmd_run_flextool`` arguments: the supervisor passes them
    to the engine unchanged and owns the run's ``_run.log``.
    """
    url_norm = url if "://" in url else f"sqlite:///{url}"
    chosen = {
        "warm_start": Resolved(True, "cli"),
        "basis_cache_dir": Resolved(str(cache_dir), "cli"),
    }
    if threads is not None:
        chosen["threads"] = Resolved(int(threads), "cli")
    folders = split_solver_config_dirs(solver_config_dir)
    if folders:
        chosen["solver_config_dir"] = Resolved(tuple(str(folder) for folder in folders), "cli")
    found = split_solver_config_dirs(workspace_solver_config_dir)
    if found:
        chosen["workspace_solver_config_dir"] = Resolved(tuple(str(folder) for folder in found), "cli")
    warm = RunOptions(resolved=chosen)
    return [
        *python_module_argv("flextool_run_supervisor"),
        url_norm,
        "--scenario-name",
        scenario,
        "--work-folder",
        str(work_dir),
        "--output-location",
        str(out_root),
        "--write-methods",
        "parquet",
        *warm.to_argv(),
        "--run-launcher",
        launcher,
    ]


__all__ = ["SolveRun", "run_solve", "solve_argv"]
