"""End-to-end CLI test for ``flextool.cli.cmd_run_flextool``.

The scenario tests in ``tests/test_scenarios.py`` exercise the in-process
``run_chain_from_db`` + ``write_outputs(write_methods=['csv'])`` path
directly.  That's faster and gives byte-level golden comparisons, but
it doesn't exercise:

  * The CLI argument parser, default handling, and exit-code contract.
  * The combined ``parquet + plot + csv`` writer matrix that the GUI
    actually invokes.
  * Wide-default ``--active-configs`` config selection that surfaces
    writer paths the smoke fixtures don't touch.
  * The full subprocess module-import path
    (``python -m flextool.cli.cmd_run_flextool``).

This test fills that gap with a single end-to-end subprocess run of the
``coal`` smoke scenario (the simplest fixture in the suite), mirroring
the invocation produced by Spine Toolbox / the FlexTool GUI.  When it
fails, the failure shape is "real-run reproducer": directly comparable
to the GUI's terminal output.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# Active-configs list mirrors the GUI's default selection seen in real
# user invocations (see the Rivendell run captured in the May 2026
# incident).  Spans the same writer surface area that triggered the
# regressions caught by adding this test.
DEFAULT_ACTIVE_CONFIGS = (
    "chunks",
    "connection_details",
    "connection_time_plots",
    "debug",
    "default",
    "lines",
    "lines_weekly",
    "reserve",
    "sum_period",
    "sum_periods",
    "unit_details",
    "unit_time_plots",
)


def test_cli_end_to_end_coal(test_db_url: str, tmp_path: Path) -> None:
    """Run the ``coal`` scenario through the CLI subprocess entry point.

    Mirrors the GUI's invocation: ``--write-methods parquet plot csv``,
    a representative ``--active-configs`` set, ``--highs-threads 1``.

    Pass criteria:
      * Exit code 0.
      * ``output_csv/coal/summary_solve.csv`` exists.
      * At least one parquet under ``output_raw/`` is present.
      * At least one ``.html`` / ``.png`` plot artefact under
        ``output_plot/`` is present (parquet+plot writers actually ran).
    """
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    output_location = tmp_path / "out"
    output_location.mkdir()
    output_subdir = "coal_e2e"

    cmd = [
        sys.executable,
        "-m", "flextool.cli.cmd_run_flextool",
        test_db_url,
        "--scenario-name", "coal",
        "--work-folder", str(work_folder),
        "--output-location", str(output_location),
        "--output-subdir", output_subdir,
        "--write-methods", "parquet", "plot", "csv",
        "--active-configs", *DEFAULT_ACTIVE_CONFIGS,
        "--highs-threads", "1",
    ]

    env = os.environ.copy()
    # Force unbuffered IO so we can read failure messages live if the
    # subprocess hangs or aborts mid-stream.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"CLI exited with code {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )

    csv_out = output_location / "output_csv" / output_subdir / "summary_solve.csv"
    assert csv_out.exists(), (
        f"CSV writer did not produce summary_solve.csv at {csv_out}\n"
        f"output_location contents:\n"
        + "\n".join(str(p.relative_to(output_location)) for p in output_location.rglob("*"))
    )

    parquet_dir = output_location / "output_parquet"
    parquet_files = list(parquet_dir.rglob("*.parquet")) if parquet_dir.exists() else []
    # Some FlexTool versions emit parquet under ``output_raw/`` inside
    # the work folder instead — accept either location.
    if not parquet_files:
        raw_dir = work_folder / "output_raw"
        parquet_files = list(raw_dir.rglob("*.parquet")) if raw_dir.exists() else []
    assert parquet_files, (
        f"Parquet writer produced no .parquet files under "
        f"{parquet_dir} or {work_folder / 'output_raw'}"
    )

    plot_dir = output_location / "output_plots"
    plot_files: list[Path] = []
    if plot_dir.exists():
        plot_files = [
            p for p in plot_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".html", ".png", ".svg"}
        ]
    assert plot_files, (
        f"Plot writer produced no .html/.png/.svg files under {plot_dir}"
    )
