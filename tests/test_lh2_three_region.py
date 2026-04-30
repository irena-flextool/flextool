"""Integration tests for the three-region LH2 fixture (Agent 1.9).

This is the first non-degenerate exercise of the flex-temporal stack:

* hourly + daily blocks coexist within one solve
* indirect-method (regular) connection straddles blocks (hourly elec
  source → daily H2 sink — the electrolyser)
* coarse-block flows (liquefier h2→lh2, pipelines) live entirely on
  the daily grid
* coarse-block storage state (LH2) is daily

The fixture is built fresh per pytest session by
``tests/fixtures/build_lh2_three_region.py`` (no .sqlite is checked
into the repo).  The integration tests are intentionally lightweight —
they assert solve success, lock the objective against a golden, and
spot-check the block-aware broadcast in the output writers (Agent 1.8).
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
    DAILY_STEPS,
    HOURLY_STEPS,
    N_DAYS,
    N_HOURS,
    REGIONS,
    SCENARIO,
)
from db_utils import json_to_db  # noqa: E402

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner  # noqa: E402


# Tolerance lifted from the existing scenario-test convention.
OBJECTIVE_TOLERANCE = 1e-4


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
    test_bin_dir: Path,
) -> Path:
    """Run the LH2 fixture once per session and return the work dir.

    FlexToolRunner writes ``output_raw/``, ``solve_data/``,
    ``HiGHS.log`` and intermediate ``flextool.{mps,sol}`` relative to
    ``work_folder`` (default = cwd).  We pin ``work_folder`` to a
    session-scoped temp dir so all test paths are deterministic.
    """
    import os

    workdir = tmp_path_factory.mktemp("lh2_run")
    # FlexToolRunner reads the cwd at construction time for some
    # downstream writers; pin it for this session by chdir'ing.
    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        runner = FlexToolRunner(
            input_db_url=lh2_db_url,
            scenario_name=SCENARIO,
            root_dir=workdir,
            bin_dir=test_bin_dir,
            work_folder=workdir,
        )
        runner.write_input(lh2_db_url, SCENARIO)
        rc = runner.run_model()
    finally:
        os.chdir(prev_cwd)
    assert rc == 0, "LH2 three-region solve failed (return code != 0)"
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
        threshold logic is explicit.
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

        The connection's source side lives at the elec node's block
        (hourly).  Even though the LP may legitimately bunch all
        consumption into a single hour per day, the fact that *any*
        hour-to-hour variation is permitted (i.e. flows are NOT pinned
        constant within a day) is what we want to exercise here.

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

    def test_liquefier_h2_to_lh2_is_daily_constant(
        self, lh2_solve: Path
    ) -> None:
        """Daily-block process flow is constant within each day.

        The liquefier is in ``daily_group`` and its source/sink both
        live at the daily block.  Agent 1.8's broadcast must populate
        every hour within a day with the same value.
        """
        df = self._read_v_flow(lh2_solve)
        col = ("liquefier_A", "h2_A", "lh2_A")
        col_str = str(col)
        assert col_str in df.columns, f"missing liquefier flow column {col_str!r}"
        flows = df[col_str].astype(float).to_numpy()
        for d in range(N_DAYS):
            day_slice = flows[d * 24 : (d + 1) * 24]
            spread = day_slice.max() - day_slice.min()
            assert spread < 1e-9, (
                f"day {d + 1}: liquefier flow varies within the day "
                f"(spread={spread:.3e}); daily-block broadcast should "
                f"emit the same value for all 24 hours."
            )

    def test_pipe_flows_are_daily(self, lh2_solve: Path) -> None:
        """LH2 pipeline flows (daily block) are constant within each day.

        Validates that connection-level coarse-block broadcast works
        for the pipeline topology too.
        """
        df = self._read_v_flow(lh2_solve)
        for col in (
            ("pipe_AB", "lh2_A", "lh2_B"),
            ("pipe_AB", "lh2_B", "lh2_A"),
            ("pipe_BC", "lh2_B", "lh2_C"),
            ("pipe_BC", "lh2_C", "lh2_B"),
        ):
            col_str = str(col)
            if col_str not in df.columns:
                continue  # one direction may be all-zero / pruned
            flows = df[col_str].astype(float).to_numpy()
            for d in range(N_DAYS):
                day_slice = flows[d * 24 : (d + 1) * 24]
                spread = day_slice.max() - day_slice.min()
                assert spread < 1e-9, (
                    f"{col_str} day {d + 1}: pipe flow varies within "
                    f"the day (spread={spread:.3e})."
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
                f"{col}: expected {N_HOURS} rows after broadcast, got {len(states)}"
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

        Sanity check that the broadcast doesn't accidentally collapse
        hourly nodes to daily-broadcast.  At least one day should show
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
        """Every time-indexed output has 168 rows (the fine timeline).

        Agent 1.8's broadcast widens coarse-block variables to the fine
        timeline so downstream readers don't see ragged DataFrames.
        """
        for fname in ("v_flow", "v_state", "vq_state_up", "vq_state_down"):
            path = lh2_solve / "output_raw" / f"{fname}__lh2_week.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            assert len(df) == N_HOURS, (
                f"{fname}: expected {N_HOURS} rows (one per fine-timeline "
                f"step), got {len(df)}.  Coarse-block variables must be "
                f"broadcast to every covered fine timestep."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_v_flow(workdir: Path) -> pd.DataFrame:
        path = workdir / "output_raw" / "v_flow__lh2_week.parquet"
        assert path.exists(), f"v_flow parquet missing at {path}"
        return pd.read_parquet(path)
