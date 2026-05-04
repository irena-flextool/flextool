"""Re-generate flextool ``input/`` + ``solve_data/`` for a named
scenario into ``tests/data/work_<scenario>/``.  Read by tests but not
itself a test (leading underscore keeps pytest from collecting it).

Usage:
    python tests/_gen_input.py <scenario>            # default subdir
    python tests/_gen_input.py <scenario> <subdir>

Per-sub-solve snapshots
-----------------------
For chained-solve scenarios flextool's ``solve_data/`` is overwritten
each sub-solve, so by the time generation returns only the LAST
sub-solve's snapshot is on disk.  flexpy single-solve replay needs
each sub-solve's snapshot independently.

This generator installs a monkey-patch around
``flextool.flextoolrunner.orchestration.capture_post_solve`` that, in
addition to whatever the upstream function does (handoff capture or
no-op), copies the current ``solve_data/`` directory to
``solve_data_<solve_name>/`` immediately after each sub-solve's
``solver.run()`` returns.  The matching per-sub-solve parquet outputs
(``output_raw/v_*__<solve_name>.parquet``) are already written
per-sub-solve by the solver.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# REPO_ROOT still points at the live flextool checkout because flextool's
# Python package (FlexToolRunner, SolverRunner) and its bin/ scripts /
# .mod files are still consumed live — only the JSON fixture data and
# json_to_db helper have been vendored into flexpy_spike.
REPO_ROOT  = Path("/home/jkiviluo/sources/flextool")
TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = TESTS_DIR / "data"
FIXTURES   = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))

from _db_utils import json_to_db                                      # noqa: E402  vendored
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner    # noqa: E402
from flextool.flextoolrunner import solver_runner as _solver_runner_mod  # noqa: E402


def _install_per_sub_solve_snapshot_hook() -> None:
    """Wrap ``SolverRunner.run`` to snapshot ``solve_data/`` to
    ``solve_data_<solve_name>/`` immediately BEFORE the solver is
    invoked.

    The snapshot must be taken at this exact moment because that's the
    LP-input state of ``solve_data/``: per-iteration pre-solve writers
    have just finished (orchestration lines 255..502), preprocessing
    has just resolved the inherited / handed-off existing capacity
    (orchestration line 537), and the next thing that happens is the
    solver reading the CSVs.  flexpy's single-solve replay will read
    the snapshot the same way the .mod does.

    Idempotent: a second call is a no-op."""
    SolverRunner = _solver_runner_mod.SolverRunner
    if getattr(SolverRunner.run, "_snapshot_wrapped", False):
        return
    original_run = SolverRunner.run

    def _resolve_roll_name(wf: Path, fallback: str) -> str:
        """Same logic as flextool's ``_actual_solve_name``: in rolling /
        nested scenarios ``solver.run`` is invoked with the PARENT
        complete-solve name while ``solve_data/solve_current.csv`` keys
        off the per-roll name (e.g. ``dispatch_fullYear_roll_roll_0``).
        flextool's parquet output uses the roll name; align our snapshot
        directory with that so multiple rolls don't collide."""
        path = wf / "solve_data" / "solve_current.csv"
        if not path.exists():
            return fallback
        try:
            with path.open() as fh:
                lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
            if len(lines) < 2:
                return fallback
            # solve_current.csv is a single-column CSV (header + one value).
            return lines[1].split(",")[0]
        except Exception:
            return fallback

    def _wrapped_run(self, current_solve: str) -> int:
        try:
            wf = self.state.paths.work_folder
            src = wf / "solve_data"
            roll_name = _resolve_roll_name(wf, current_solve)
            dst = wf / f"solve_data_{roll_name}"
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        except Exception as exc:  # diagnostic only — don't break the solve
            try:
                self.logger.warning(
                    f"per-sub-solve snapshot failed for {current_solve}: {exc}"
                )
            except Exception:
                pass
        return original_run(self, current_solve)

    _wrapped_run._snapshot_wrapped = True  # type: ignore[attr-defined]
    SolverRunner.run = _wrapped_run


def generate(scenario: str, workdir: Path | None = None) -> Path:
    workdir = workdir or (DATA_DIR / f"work_{scenario}")
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "tests.sqlite"
    if db_path.exists(): db_path.unlink()
    db_url = json_to_db(FIXTURES / "tests.json", db_path)

    _install_per_sub_solve_snapshot_hook()

    runner = FlexToolRunner(
        input_db_url   = db_url,
        scenario_name  = scenario,
        flextool_dir   = REPO_ROOT / "flextool",
        bin_dir        = REPO_ROOT / "bin",
        work_folder    = workdir,
    )
    runner.write_input(db_url, scenario)
    rc = runner.run_model()
    if rc != 0:
        raise SystemExit(f"flextool failed for {scenario!r}: rc={rc}")
    print(f"\n{scenario}: generated under {workdir}")
    return workdir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    sub = Path(__file__).parent / "data" / sys.argv[2] if len(sys.argv) > 2 else None
    generate(sys.argv[1], sub)
