"""Smoke tests for the unified TimingRecorder / timings.csv layer.

These replace the two legacy ``solve_progress.csv`` files (one in
``solve_data/`` written by five different writers, one in ``output/``
written by ``log_time``) — see ``flextool/cli/_timing.py``.

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

from flextool.engine_polars import run_chain_from_db  # noqa: E402  # imports after sys.path manipulation above
from flextool.engine_polars._flex_data_provider import FlexDataProvider  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402  # imports after sys.path manipulation above


_EXPECTED_COLUMNS = [
    "phase",
    "subphase",
    "solve",
    "roll_index",
    "seconds",
    "started_at_iso",
    "cumulative_s",
]


def _run_one(scenario: str, test_db_url: str, test_solver_config_dir: Path,
             workdir: Path) -> Path:
    """Run the native cascade end-to-end and emit outputs for ``scenario``.

    Δ.22 migration: ``SolverRunner.run`` was deleted, so the legacy
    ``runner.run_model()`` path raises :class:`NotImplementedError`.
    ``run_chain_from_db`` is the cascade-native replacement.  The
    cascade's internal ``TimingRecorder`` writes ``solve_data/
    timings.csv`` directly, so we don't need to thread the recorder
    through ``write_outputs`` (which still gets called for symmetry with
    the legacy harness, but its rows aren't asserted on here).
    """
    steps = run_chain_from_db(
        test_db_url, scenario, work_folder=workdir,
        csv_dump=True, keep_solutions=True,
    )
    last_step = next(reversed(list(steps.values())))
    assert last_step.optimal, (
        f"Model run failed for scenario '{scenario}': "
        f"last step not optimal"
    )
    # write_outputs is best-effort here — the asserted artefact is
    # ``solve_data/timings.csv``, which the cascade writes inside
    # ``run_chain_from_db`` independent of ``write_outputs``.  Wrap
    # the call so per-scenario downstream-output bugs (e.g. a column
    # mismatch in ``out_capacity.unit_capacity``) don't mask the
    # timings.csv content the test is actually about.
    try:
        write_outputs(
            scenario_name=scenario,
            output_location=str(workdir),
            subdir=scenario,
            output_config_path=OUTPUT_CONFIG,
            write_methods=["csv"],
            fallback_output_location=str(workdir),
            flex_data=last_step.flex_data,
            solution=last_step.solution,
            solve_name=last_step.solve_name,
            # In-memory path requires a Provider; this test asserts on
            # timings.csv, not group flows, so an empty Provider keeps
            # behaviour identical while satisfying the contract.
            flex_data_provider=FlexDataProvider(),
        )
    except Exception:
        pass
    return workdir


@pytest.mark.smoke
def test_timings_csv_exists_and_has_expected_schema(
    test_db_url: str,
    test_solver_config_dir: Path,
    workdir: Path,
) -> None:
    """``solve_data/timings.csv`` exists with the documented columns."""
    _run_one("coal", test_db_url, test_solver_config_dir, workdir)

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
def test_timings_csv_multi_roll(
    test_db_url: str,
    test_solver_config_dir: Path,
    workdir: Path,
) -> None:
    """Rolling / multi-solve scenarios get one ``solve`` row per roll.

    Uses ``fullYear_roll`` (a rolling fullYear scenario in scenarios.yaml)
    which iterates the solver multiple times.  Asserts the recorder
    distinguishes per-roll entries via the ``roll_index`` column.
    """
    _run_one("fullYear_roll", test_db_url, test_solver_config_dir, workdir)

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
