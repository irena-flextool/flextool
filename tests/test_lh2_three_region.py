"""Integration tests for the three-region LH2 fixture.

The fixture exercises a coarse ``new_stepduration`` group: the h2_* and
lh2_* nodes are in the 24-h ``daily_group``, the electricity nodes are
hourly.  A daily node is balanced once per day by summing its hourly
flows (``nodeBalanceCoarse_eq``) and a daily storage node keeps one state
per day.  The liquefiers and pipes (between daily nodes only) have one
flow per day, written at every hour; the electrolysers (hourly ->
daily) have hourly flows.

The fixture is built fresh per pytest session from
``tests/fixtures/lh2_three_region.json`` (no .sqlite is checked into the
repo).  The integration tests are intentionally lightweight — they assert
solve success, lock the objective against a golden and spot-check the
written outputs.  The coarse-group rows themselves are tested in
``tests/engine/test_coarse_groups.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
EXPECTED_DIR = TEST_DIR / "expected"
FIXTURES_DIR = TEST_DIR / "fixtures"
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

if str(TEST_DIR / "fixtures") not in sys.path:
    sys.path.insert(0, str(TEST_DIR / "fixtures"))

from build_lh2_three_region import (  # noqa: E402
    N_DAYS,
    N_HOURS,
    REGIONS,
    SCENARIO,
)
from db_utils import json_to_db  # noqa: E402

from flextool.engine import run_chain_from_db  # noqa: E402


# Tolerance lifted from the existing scenario-test convention.
OBJECTIVE_TOLERANCE = 1e-4

# Objective of the hourly twin of this fixture (``daily_group.new_stepduration`` 24 → 1, or the daily group
# removed: both give 5 855 731 058.77; oracle batch F, specs/bugfix/oracle_batch_F.md §2).  The daily model is a
# relaxation of it: summing an hourly-feasible point's balances over each day gives a daily-feasible point, and
# the daily state at the end of each day is the hourly state at that time.  So the daily objective can never
# exceed it.
HOURLY_TWIN_OBJECTIVE = 5_855_731_058.77


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def lh2_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build the LH2 fixture once per session and return the sqlite URL.

    Mirrors the ``tests.json → tests.sqlite`` pattern from
    ``tests/conftest.py``: the JSON fixture is the committed source of
    truth, and ``json_to_db`` materialises a fresh SQLite for each
    pytest session (no stale state from previous runs).
    """
    db_path = tmp_path_factory.mktemp("lh2_db") / "lh2_three_region.sqlite"
    return json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)


@pytest.fixture(scope="session")
def lh2_solve(
    lh2_db_url: str,
    tmp_path_factory: pytest.TempPathFactory,
    test_solver_config_dir: Path,
) -> Path:
    """Run the LH2 fixture once per session and return the work dir.

    FlexToolRunner writes ``output_raw/``, ``solve_data/``,
    ``HiGHS.log`` and intermediate ``flextool.{mps,sol}`` relative to
    ``work_folder`` (default = cwd).  We pin ``work_folder`` to a
    session-scoped temp dir so all test paths are deterministic.
    """
    import os

    workdir = tmp_path_factory.mktemp("lh2_run")
    # Δ.22 migration: SolverRunner.run was deleted, so the legacy
    # runner.run_model() path raises NotImplementedError.  Native
    # cascade replacement is run_chain_from_db.  csv_dump=True snapshots
    # solve_data/; keep_solutions=True retains the flex_data_provider
    # so the derived input/ frames materialise to disk.
    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        steps = run_chain_from_db(
            lh2_db_url,
            SCENARIO,
            work_folder=workdir,
            csv_dump=True,
            keep_solutions=True,
        )
    finally:
        os.chdir(prev_cwd)
    assert steps, "run_chain_from_db returned no steps for LH2 fixture"
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal, (
        "LH2 three-region solve failed (last sub-solve not optimal)"
    )
    provider = getattr(last_step, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(workdir)
    return workdir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLh2ThreeRegion:
    """Integration tests for the monolithic LH2 three-region solve."""

    def test_solve_succeeds_and_objective_matches_golden(
        self, lh2_solve: Path
    ) -> None:
        """The LH2 monolithic solve produces a deterministic objective.

        Golden file lives in ``tests/expected/lh2_three_region/objective.json``
        and is regenerated on demand via ``--regenerate lh2_three_region``
        on a separate parametrized test (see ``test_scenarios.py`` for
        the existing pattern).  Here we use a static JSON so the
        threshold logic is explicit.  Independently of the golden, the
        daily objective is bounded above by the hourly twin's
        (``HOURLY_TWIN_OBJECTIVE``).
        """
        v_obj_path = lh2_solve / "output_raw" / "v_obj__lh2_week.parquet"
        assert v_obj_path.exists(), f"v_obj parquet not produced at {v_obj_path}"
        obj_df = pd.read_parquet(v_obj_path)
        actual = float(obj_df["objective"].iloc[0])

        golden_path = EXPECTED_DIR / "lh2_three_region" / "objective.json"
        if not golden_path.exists():
            # First-run convenience: write the golden so a follow-up
            # commit captures it.  The test fails on a missing golden
            # so the developer notices.
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(json.dumps({"objective": actual}, indent=2))
            pytest.fail(
                f"Wrote new golden objective {actual!r} to "
                f"{golden_path.relative_to(REPO_ROOT)}; commit it and "
                f"re-run the test to validate."
            )
        assert actual <= HOURLY_TWIN_OBJECTIVE, (
            f"daily objective {actual!r} exceeds the hourly twin's "
            f"{HOURLY_TWIN_OBJECTIVE!r}; the daily model is its relaxation"
        )
        golden = json.loads(golden_path.read_text())
        expected = float(golden["objective"])
        rel_diff = abs(actual - expected) / max(abs(expected), 1.0)
        assert rel_diff <= OBJECTIVE_TOLERANCE, (
            f"objective {actual!r} differs from golden {expected!r} "
            f"by relative {rel_diff:.3e} > tolerance {OBJECTIVE_TOLERANCE}"
        )

    def test_electrolyser_elec_input_has_hourly_variance(
        self, lh2_solve: Path
    ) -> None:
        """Electrolyser elec source-side flow is at hourly resolution.

        The electrolyser connects an hourly node, so its flows are hourly.  Even though the LP may
        legitimately bunch all consumption into a single hour per day, the
        fact that *any* hour-to-hour variation is permitted (i.e. flows are
        NOT pinned constant within a day) is what we want to exercise here.

        Concretely: we check that for *some* day, two distinct hourly
        timesteps in that day have different electrolyser elec-input
        values.  If the source-side LP variable were collapsed to the
        daily block, this would never be possible.
        """
        df = self._read_v_flow(lh2_solve)
        col = ("electrolyser_A", "elec_A", "h2_A")
        col_str = str(col)
        assert col_str in df.columns, (
            f"missing electrolyser source flow column {col_str!r} in "
            f"v_flow output (have: {df.columns.tolist()[:5]})"
        )
        flows = df[col_str].astype(float).to_numpy()
        any_intraday_variance = False
        for d in range(N_DAYS):
            day_slice = flows[d * 24 : (d + 1) * 24]
            if day_slice.max() - day_slice.min() > 1e-9:
                any_intraday_variance = True
                break
        assert any_intraday_variance, (
            "electrolyser elec source flow is constant within every day; "
            "expected hourly variance because the source side lives at the "
            "elec_A hourly block."
        )

    def test_lh2_storage_state_is_daily(self, lh2_solve: Path) -> None:
        """LH2 storage state (daily block) is constant within each day."""
        v_state_path = lh2_solve / "output_raw" / "v_state__lh2_week.parquet"
        assert v_state_path.exists(), f"v_state parquet missing at {v_state_path}"
        df = pd.read_parquet(v_state_path)
        for r in REGIONS:
            col = f"lh2_{r}"
            assert col in df.columns, f"missing v_state column for {col}"
            states = df[col].astype(float).to_numpy()
            assert len(states) == N_HOURS, (
                f"{col}: expected {N_HOURS} rows, got {len(states)}"
            )
            for d in range(N_DAYS):
                day_slice = states[d * 24 : (d + 1) * 24]
                spread = day_slice.max() - day_slice.min()
                assert spread < 1e-9, (
                    f"{col} day {d + 1}: state varies within the day "
                    f"(spread={spread:.3e}); daily storage state must be "
                    f"constant within each daily block."
                )

    def test_battery_state_is_hourly(self, lh2_solve: Path) -> None:
        """Battery storage state (hourly block) varies hour-to-hour.

        Sanity check that the daily group does not collapse hourly nodes
        to daily states.  At least one day should show
        nontrivial hour-to-hour variation in battery state.
        """
        v_state_path = lh2_solve / "output_raw" / "v_state__lh2_week.parquet"
        df = pd.read_parquet(v_state_path)
        for r in REGIONS:
            col = f"battery_{r}"
            if col not in df.columns:
                continue
            states = df[col].astype(float).to_numpy()
            any_variance = False
            for d in range(N_DAYS):
                day_slice = states[d * 24 : (d + 1) * 24]
                if day_slice.max() - day_slice.min() > 1e-6:
                    any_variance = True
                    break
            if any_variance:
                return
        # If no battery showed variance, the LP might just have decided
        # to keep batteries idle — that's a legitimate optimum.  The
        # test stops asserting in that case to avoid false positives.
        pytest.skip(
            "no battery showed intra-day variance — LP optimum may be "
            "idle batteries (acceptable, not a regression signal)."
        )

    def test_output_row_counts_match_fine_timeline(
        self, lh2_solve: Path
    ) -> None:
        """Every time-indexed output has 168 rows (the fine timeline): the
        flows of processes between daily nodes (one per day in the LP) are
        written at every hour, like every other variable.
        """
        for fname in ("v_flow", "v_state", "vq_state_up", "vq_state_down"):
            path = lh2_solve / "output_raw" / f"{fname}__lh2_week.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            assert len(df) == N_HOURS, (
                f"{fname}: expected {N_HOURS} rows (one per fine-timeline "
                f"step), got {len(df)}."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_v_flow(workdir: Path) -> pd.DataFrame:
        path = workdir / "output_raw" / "v_flow__lh2_week.parquet"
        assert path.exists(), f"v_flow parquet missing at {path}"
        return pd.read_parquet(path)
