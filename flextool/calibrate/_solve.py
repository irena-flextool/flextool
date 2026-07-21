"""Shell out to ``cmd_run_flextool`` for one calibrator iteration.

The calibrator does **not** solve in-process: it launches a fresh
:mod:`flextool.cli.cmd_run_flextool` subprocess per iteration (a clean
address space per solve, matching how the model is run in production) and
then reads the produced parquet outputs.  This module owns that launch —
building the argv, wiring the warm-start environment, capturing the launch
time (needed by the solve-success detector's freshness check), and running
the subprocess with its stdout+stderr merged into one captured stream.

Warm start
----------
Warm start is enabled via the environment, not the ``--warm-start`` CLI
flag, because it is the env vars the engine actually reads
(``FLEXTOOL_WARM_START`` / ``FLEXTOOL_BASIS_CACHE_DIR`` in
``flextool.engine_polars._orchestration``).  A *stable* basis-cache
directory shared across iterations lets HiGHS reuse the previous
iteration's basis when the structural model is unchanged (the adder is
RHS-only, so the warm-start fingerprint is stable across iterations).

``FLEXTOOL_SAVE_MEMORY`` is deliberately **not** set here: it releases the
live HiGHS instance after each sub-solve and so DISABLES warm-LP reuse.  It
must also stay constant (unset) across every iteration — flipping it
mid-run would invalidate the shared basis cache.

This module does not judge success: it returns the raw
:class:`SolveRun` and lets the loop call
:func:`flextool.calibrate.assess_solve` so the loop owns the
``required_outputs`` choice.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
        (``FLEXTOOL_BASIS_CACHE_DIR``).  Keep it stable across the whole
        calibration run so HiGHS can reuse the prior iteration's basis.
    """
    url_norm = url if "://" in url else f"sqlite:///{url}"
    argv = [
        sys.executable,
        "-m",
        "flextool.cli.cmd_run_flextool",
        url_norm,
        "--scenario-name",
        scenario,
        "--work-folder",
        str(work_dir),
        "--output-location",
        str(out_root),
        "--write-methods",
        "parquet",
    ]

    env = os.environ.copy()
    env["FLEXTOOL_WARM_START"] = "1"
    env["FLEXTOOL_BASIS_CACHE_DIR"] = str(cache_dir)
    # FLEXTOOL_SAVE_MEMORY is intentionally left untouched: setting it would
    # disable warm-LP reuse, and it must stay constant across iterations.

    assess_dir = Path(out_root) / "output_parquet" / scenario

    started_at = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(_REPO_ROOT),
        env=env,
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


__all__ = ["SolveRun", "run_solve"]
