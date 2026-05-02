"""Smoke tests for the unified TimingRecorder / timings.csv layer.

These replace the two legacy ``solve_progress.csv`` files (one in
``solve_data/`` written by five different writers, one in ``output/``
written by ``log_time``) — see ``flextool/flextoolrunner/timing_recorder.py``.

Each test runs a small scenario end-to-end and validates the resulting
``solve_data/timings.csv`` schema, total/phase consistency, and (for
the multi-roll case) per-roll coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.process_outputs.write_outputs import write_outputs


_EXPECTED_COLUMNS = [
    "phase",
    "subphase",
    "solve",
    "roll_index",
    "seconds",
    "started_at_iso",
    "cumulative_s",
]


def _run_one(scenario: str, test_db_url: str, test_bin_dir: Path,
             workdir: Path) -> Path:
    """Run write_input → run_model → write_outputs for ``scenario``.

    Returns the workdir.  Mirrors the test_scenarios.py harness — the
    timing recorder bootstraps inside ``FlexToolRunner.__init__`` since
    we don't go through the CLI.
    """
    runner = FlexToolRunner(
        input_db_url=test_db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(test_db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Model run failed for scenario '{scenario}'"
    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
        timing_recorder=runner.state.timing_recorder,
    )
    return workdir


@pytest.mark.smoke
def test_timings_csv_exists_and_has_expected_schema(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    """``solve_data/timings.csv`` exists with the documented columns."""
    _run_one("coal", test_db_url, test_bin_dir, workdir)

    csv_path = workdir / "solve_data" / "timings.csv"
    assert csv_path.exists(), (
        f"timings.csv missing at {csv_path}; the TimingRecorder did not "
        f"bootstrap or did not flush any rows"
    )
    df = pd.read_csv(csv_path)
    assert list(df.columns) == _EXPECTED_COLUMNS, (
        f"timings.csv columns differ: got {list(df.columns)}, "
        f"expected {_EXPECTED_COLUMNS}"
    )
    # cumulative_s is monotonically non-decreasing.
    assert df["cumulative_s"].is_monotonic_increasing, (
        "cumulative_s should be monotonically non-decreasing across rows"
    )
    # All rows have a non-empty phase name.
    assert (df["phase"].astype(str).str.len() > 0).all(), (
        "every row must carry a phase name"
    )


@pytest.mark.smoke
def test_timings_csv_has_solve_subphases(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    """The recorder captures both Python-side and mod-side solve subphases.

    Python writers (``solver_runner.py``) emit ``mps_gen`` and
    ``lp_solve``; the .mod's per-phase printfs emit ``mod_setup`` /
    ``mod_balance`` / etc.  This test confirms both pathways feed the
    unified CSV — losing either one would silently regress coverage.
    """
    _run_one("coal", test_db_url, test_bin_dir, workdir)

    csv_path = workdir / "solve_data" / "timings.csv"
    df = pd.read_csv(csv_path)
    solve_rows = df[df["phase"] == "solve"]
    subphases = set(solve_rows["subphase"].astype(str).unique())

    # Python-side: mps generation + LP solve.
    assert "mps_gen" in subphases, (
        f"missing 'mps_gen' subphase under phase='solve'; got {sorted(subphases)}"
    )
    assert "lp_solve" in subphases, (
        f"missing 'lp_solve' subphase under phase='solve'; got {sorted(subphases)}"
    )
    # Mod-side: at least the two early-LP phases (setup + total_obj_cost).
    assert "mod_setup" in subphases, (
        f"missing 'mod_setup' subphase; the .mod-side phase printfs are "
        f"not feeding mod_phases.csv → timings.csv. got={sorted(subphases)}"
    )


@pytest.mark.smoke
def test_timings_csv_multi_roll(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    """Rolling / multi-solve scenarios get one ``solve`` row per roll.

    Uses ``fullYear_roll`` (a rolling fullYear scenario in scenarios.yaml)
    which iterates the solver multiple times.  Asserts the recorder
    distinguishes per-roll entries via the ``roll_index`` column.
    """
    _run_one("fullYear_roll", test_db_url, test_bin_dir, workdir)

    csv_path = workdir / "solve_data" / "timings.csv"
    df = pd.read_csv(csv_path)
    # Each per-roll iteration yields a 'roll_setup' row (orchestration).
    roll_setup_rows = df[df["phase"] == "roll_setup"]
    assert len(roll_setup_rows) >= 2, (
        f"expected ≥2 'roll_setup' rows for a rolling scenario; "
        f"got {len(roll_setup_rows)}"
    )
    distinct_rolls = roll_setup_rows["roll_index"].astype(str).unique()
    assert len(distinct_rolls) >= 2, (
        f"roll_index should differ across rolls; got {sorted(distinct_rolls)}"
    )
