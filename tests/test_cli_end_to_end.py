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


@pytest.mark.parametrize(
    "scenario",
    [
        # ``coal``: simplest fixture in the suite — exercises the basic
        # CLI surface (argument parsing, all three writers, plot config).
        "coal",
        # ``multi_year_one_solve``: single-solve multi-period invest —
        # exercises the per-period existing-capacity pivot in
        # ``_pdX_per_entity`` that the GUI's Rivendell run trips when the
        # entity × period axis has duplicates from nested-cascade filter
        # leakage.  Same general layout as ``rivendell_invest``.
        "multi_year_one_solve",
    ],
)
def test_cli_end_to_end(scenario: str, test_db_url: str, tmp_path: Path) -> None:
    """Run a scenario through the CLI subprocess entry point.

    Mirrors the GUI's invocation: ``--write-methods parquet plot csv``,
    a representative ``--active-configs`` set, ``--highs-threads 1``.

    Pass criteria:
      * Exit code 0.
      * ``output_csv/<subdir>/summary_solve.csv`` exists.
      * At least one parquet artefact is present (in ``output_parquet/``
        or ``work/output_raw/``).
      * At least one ``.html`` / ``.png`` plot artefact under
        ``output_plots/`` is present (parquet+plot writers actually ran).
    """
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    output_location = tmp_path / "out"
    output_location.mkdir()
    output_subdir = f"{scenario}_e2e"

    cmd = [
        sys.executable,
        "-m", "flextool.cli.cmd_run_flextool",
        test_db_url,
        "--scenario-name", scenario,
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


# ---------------------------------------------------------------------------
# Rivendell-shaped reproducer for the Map-index-name pivot bug
# ---------------------------------------------------------------------------
#
# Rivendell-derived fixtures author the per-period ``existing`` parameter as
# a ``Map`` whose ``index_name`` is set to ``x`` (rather than the canonical
# ``period`` default).  ``_per_entity_param_lf`` previously checked only
# ``"period" in cols`` to detect the period axis — when the source surfaced
# the column as ``x`` it fell through to the scalar-broadcast branch,
# emitting one fake-scalar row per source row.  ``_resolve_per_period_lf``'s
# ``scalar.join(on="e")`` then exploded each (e, d) grid pair by the number
# of source rows for that entity, and the eventual
# ``p_entity_all_existing`` pivot in ``read_parameters`` hit
# ``ValueError: Index contains duplicate entries, cannot reshape``.
#
# The Rivendell DB lives outside the repo (rivendell-build-db tooling) — we
# probe its cached location and skip cleanly when it isn't available so the
# rest of the suite stays self-contained.
_RIVENDELL_DB = Path.home() / ".cache" / "rivendell_to_flextool" / "rivendell.sqlite"


@pytest.mark.skipif(
    not _RIVENDELL_DB.exists(),
    reason=f"Rivendell cache DB not present at {_RIVENDELL_DB}",
)
def test_cli_end_to_end_rivendell_map_index_name(tmp_path: Path) -> None:
    """B0_base_slice: single-solve multi-period invest with a ``Map``-
    valued ``existing`` parameter whose ``index_name`` is ``x``.

    Regression guard for the
    ``Index contains duplicate entries, cannot reshape`` pivot crash in
    ``read_parameters.read_parameters`` — the fix is in
    ``_derived_existing._per_entity_param_lf`` /
    ``_derived_npv._per_entity_param_lf`` /
    ``_derived_params.p_entity_all_existing_from_source``: treat any
    extra non-name/value column as the period dim, not just ``period``.
    """
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    output_location = tmp_path / "out"
    output_location.mkdir()
    output_subdir = "rivendell_b0_e2e"

    cmd = [
        sys.executable,
        "-m", "flextool.cli.cmd_run_flextool",
        f"sqlite:///{_RIVENDELL_DB}",
        "--scenario-name", "B0_base_slice",
        "--work-folder", str(work_folder),
        "--output-location", str(output_location),
        "--output-subdir", output_subdir,
        "--write-methods", "parquet", "plot", "csv",
        "--active-configs", *DEFAULT_ACTIVE_CONFIGS,
        "--highs-threads", "1",
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=300,
    )

    # Narrow regression guard: assert the original
    # ``Index contains duplicate entries, cannot reshape`` symptom is
    # gone.  We don't require RC=0 because the Rivendell fixture exposes
    # other unrelated downstream issues (e.g. reserve column mismatches
    # in ``calc_slacks``) — but the pivot crash this test was built for
    # must not return.
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert "Index contains duplicate entries, cannot reshape" not in combined, (
        "Regression: ``_pdX_per_entity`` pivot hit the duplicate-entries "
        "error again.\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )
