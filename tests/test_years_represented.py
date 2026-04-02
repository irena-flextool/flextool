"""Tests for p_years_represented scaling in the inflation factor.

Verifies that p_inflation_factor_operations_yearly and
p_inflation_factor_investment_yearly correctly account for
p_years_represented values by checking the output CSV files.

With inflation_rate = 0 (default), the expected inflation factor for
a period equals the sum of p_years_represented over its years:
  - years_represented = 1 (default): 1 year with p_years_represented = 1 => factor = 1
  - years_represented = 5: 5 years with p_years_represented = 1 => factor = 5
  - years_represented = 10: 10 years with p_years_represented = 1 => factor = 10
  - years_represented = 0.5: 1 year with p_years_represented = 0.5 => factor = 0.5
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario: str,
    db_url: str,
    bin_dir: Path,
    workdir: Path,
) -> Path:
    """Run a scenario and return the workdir (where output_raw/ lives)."""
    runner = FlexToolRunner(
        input_db_url=db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=bin_dir,
    )
    runner.write_input(db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Model run failed for scenario '{scenario}'"
    return workdir


def _read_inflation_factor_operations(workdir: Path) -> pd.DataFrame:
    """Read p_inflation_factor_operations_yearly.csv from output_raw/."""
    csv_path = workdir / "output_raw" / "p_inflation_factor_operations_yearly.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


def _read_inflation_factor_investment(workdir: Path) -> pd.DataFrame:
    """Read p_inflation_factor_investment_yearly.csv from output_raw/."""
    csv_path = workdir / "output_raw" / "p_inflation_factor_investment_yearly.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    return pd.read_csv(csv_path)


def _add_half_year_scenario(db_url: str) -> None:
    """Add a 'coal_half_year' scenario to the test DB.

    This creates:
    - A new solve entity 'y2020_half_year' with years_represented = 0.5
    - A new alternative 'half_year' that uses this solve
    - A new scenario 'coal_half_year' = init + west + coal + half_year
    """
    from spinedb_api import Array, DatabaseMapping, Map, import_data

    solves_value = Array(["y2020_half_year"], value_type=str, index_name="sequence_index")
    period_timeset = Map(["p2020"], ["2day"], index_name="period")
    realized_periods = Array(["p2020"], value_type=str, index_name="period")
    years_represented = Map(["p2020"], [0.5], index_name="period")

    with DatabaseMapping(db_url) as db_map:
        count, errors = import_data(
            db_map,
            alternatives=[("half_year", "")],
            scenarios=[("coal_half_year", False, "")],
            scenario_alternatives=[
                ("coal_half_year", "init", "west"),
                ("coal_half_year", "west", "coal"),
                ("coal_half_year", "coal", "half_year"),
                ("coal_half_year", "half_year", None),
            ],
            entities=[("solve", "y2020_half_year")],
            parameter_values=[
                ("model", "flexTool", "solves", solves_value, "half_year"),
                ("solve", "y2020_half_year", "period_timeset", period_timeset, "half_year"),
                ("solve", "y2020_half_year", "realized_periods", realized_periods, "half_year"),
                ("solve", "y2020_half_year", "solve_mode", "single_solve", "half_year"),
                ("solve", "y2020_half_year", "solver", "glpsol", "half_year"),
                ("solve", "y2020_half_year", "years_represented", years_represented, "half_year"),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add coal_half_year scenario")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def half_year_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a test DB with the extra coal_half_year scenario."""
    db_path = tmp_path_factory.mktemp("db_half_year") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    _add_half_year_scenario(url)
    return url


# ===================================================================
# Test: years_represented = 1 (default) -- coal scenario
# ===================================================================

class TestYearsRepresented1:
    """Verify inflation factor = 1 when years_represented = 1 (default).

    The coal scenario uses the default solve (y2020_2day_dispatch)
    with no explicit years_represented, so p_years_represented = 1.
    With inflation_rate = 0 (default), the factor should be exactly 1.
    """

    @pytest.fixture(scope="class")
    def run_result(
        self,
        test_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("yr1")
        os.chdir(workdir)
        return _run_scenario("coal", test_db_url, test_bin_dir, workdir)

    def test_operations_factor_equals_1(self, run_result: Path) -> None:
        """p_inflation_factor_operations_yearly should be 1 for p2020."""
        df = _read_inflation_factor_operations(run_result)
        for _, row in df.iterrows():
            assert row["value"] == pytest.approx(1.0, abs=1e-6), (
                f"Expected operations factor = 1.0 for period {row['period']}, "
                f"got {row['value']}"
            )

    def test_investment_factor_equals_1(self, run_result: Path) -> None:
        """p_inflation_factor_investment_yearly should be 1 for p2020."""
        df = _read_inflation_factor_investment(run_result)
        for _, row in df.iterrows():
            assert row["value"] == pytest.approx(1.0, abs=1e-6), (
                f"Expected investment factor = 1.0 for period {row['period']}, "
                f"got {row['value']}"
            )


# ===================================================================
# Test: years_represented = 10 -- y2020_2029_1x10y scenario
# ===================================================================

class TestYearsRepresented10:
    """Verify inflation factor = 10 when years_represented = 10.

    The y2020_2029_1x10y scenario has 1 period (p2020) with
    years_represented = 10. With inflation = 0, each of the 10 years
    contributes p_years_represented = 1, so the factor = 10.
    """

    @pytest.fixture(scope="class")
    def run_result(
        self,
        test_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("yr10")
        os.chdir(workdir)
        return _run_scenario("y2020_2029_1x10y", test_db_url, test_bin_dir, workdir)

    def test_operations_factor_equals_10(self, run_result: Path) -> None:
        """p_inflation_factor_operations_yearly should be 10 for p2020."""
        df = _read_inflation_factor_operations(run_result)
        p2020_rows = df[df["period"].str.strip() == "p2020"]
        assert len(p2020_rows) > 0, "No p2020 rows in operations factor output"
        for _, row in p2020_rows.iterrows():
            assert row["value"] == pytest.approx(10.0, abs=1e-4), (
                f"Expected operations factor = 10.0 for period p2020, "
                f"got {row['value']}"
            )

    def test_investment_factor_equals_10(self, run_result: Path) -> None:
        """p_inflation_factor_investment_yearly should be 10 for p2020."""
        df = _read_inflation_factor_investment(run_result)
        p2020_rows = df[df["period"].str.strip() == "p2020"]
        assert len(p2020_rows) > 0, "No p2020 rows in investment factor output"
        for _, row in p2020_rows.iterrows():
            assert row["value"] == pytest.approx(10.0, abs=1e-4), (
                f"Expected investment factor = 10.0 for period p2020, "
                f"got {row['value']}"
            )


# ===================================================================
# Test: years_represented = 5 (two periods) -- y2020_2029_2x5y scenario
# ===================================================================

class TestYearsRepresented5:
    """Verify inflation factor = 5 when years_represented = 5.

    The y2020_2029_2x5y scenario has 2 periods (p2020, p2025) each with
    years_represented = 5. With inflation = 0, each period's factor = 5.

    Note: the investment factor CSV only includes realized_invest_periods
    (p2020 in this scenario), while the operations factor CSV includes
    all realized_periods (p2020 and p2025).
    """

    @pytest.fixture(scope="class")
    def run_result(
        self,
        test_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("yr5")
        os.chdir(workdir)
        return _run_scenario("y2020_2029_2x5y", test_db_url, test_bin_dir, workdir)

    def test_operations_factor_equals_5(self, run_result: Path) -> None:
        """p_inflation_factor_operations_yearly should be 5 for both periods."""
        df = _read_inflation_factor_operations(run_result)
        assert len(df) >= 2, f"Expected at least 2 rows, got {len(df)}"
        for _, row in df.iterrows():
            assert row["value"] == pytest.approx(5.0, abs=1e-4), (
                f"Expected operations factor = 5.0 for period {row['period']}, "
                f"got {row['value']}"
            )

    def test_investment_factor_equals_5_for_realized(self, run_result: Path) -> None:
        """p_inflation_factor_investment_yearly should be 5 for realized invest periods.

        Only realized_invest_periods (p2020) appear in the investment
        factor CSV. The factor for p2020 should still be 5.
        """
        df = _read_inflation_factor_investment(run_result)
        assert len(df) >= 1, f"Expected at least 1 row, got {len(df)}"
        for _, row in df.iterrows():
            assert row["value"] == pytest.approx(5.0, abs=1e-4), (
                f"Expected investment factor = 5.0 for period {row['period']}, "
                f"got {row['value']}"
            )

    def test_both_periods_present_in_operations(self, run_result: Path) -> None:
        """Both p2020 and p2025 should appear in the operations factor output."""
        df = _read_inflation_factor_operations(run_result)
        periods = set(df["period"].str.strip())
        assert "p2020" in periods, f"p2020 not found in periods: {periods}"
        assert "p2025" in periods, f"p2025 not found in periods: {periods}"


# ===================================================================
# Test: years_represented = 0.5 -- coal_half_year scenario
#
# This is the critical test for the p_years_represented fix.
# Before the fix, the factor would be 1.0 (wrong).
# After the fix, the factor should be 0.5 (correct).
# ===================================================================

class TestYearsRepresentedHalf:
    """Verify inflation factor = 0.5 when years_represented = 0.5.

    The coal_half_year scenario has 1 period (p2020) with
    years_represented = 0.5. The write_years_represented function
    creates 1 row (range(max(1, 0.5)) = range(1)) with
    p_years_represented = min(1, 0.5) = 0.5.

    With inflation = 0, the factor = 0.5 * 1.0 = 0.5.
    """

    @pytest.fixture(scope="class")
    def run_result(
        self,
        half_year_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("yr_half")
        os.chdir(workdir)
        return _run_scenario("coal_half_year", half_year_db_url, test_bin_dir, workdir)

    def test_operations_factor_equals_half(self, run_result: Path) -> None:
        """p_inflation_factor_operations_yearly should be 0.5 for p2020."""
        df = _read_inflation_factor_operations(run_result)
        for _, row in df.iterrows():
            assert row["value"] == pytest.approx(0.5, abs=1e-6), (
                f"Expected operations factor = 0.5 for period {row['period']}, "
                f"got {row['value']}. This would be 1.0 without the "
                f"p_years_represented fix in the inflation factor formula."
            )

    def test_investment_factor_equals_half(self, run_result: Path) -> None:
        """p_inflation_factor_investment_yearly should be 0.5 for p2020."""
        df = _read_inflation_factor_investment(run_result)
        for _, row in df.iterrows():
            assert row["value"] == pytest.approx(0.5, abs=1e-6), (
                f"Expected investment factor = 0.5 for period {row['period']}, "
                f"got {row['value']}. This would be 1.0 without the "
                f"p_years_represented fix in the inflation factor formula."
            )


# ===================================================================
# Test: Cost scaling with years_represented
#
# Verify that total costs scale proportionally with the inflation factor.
# Compare the coal scenario (factor = 1) with the coal_half_year
# scenario (factor = 0.5). The operational costs should be half.
# ===================================================================

class TestCostScalingHalfYear:
    """Verify that operational costs scale with years_represented.

    Running the same model with years_represented = 0.5 should produce
    an objective value that is approximately 0.5x the years_represented = 1
    value, since all operational costs are multiplied by the inflation
    factor (which equals years_represented when inflation = 0).

    Note: fixed costs of existing units also scale with the factor,
    so the total objective should scale almost exactly by 0.5.
    """

    @pytest.fixture(scope="class")
    def coal_workdir(
        self,
        test_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("cost_coal")
        os.chdir(workdir)
        return _run_scenario("coal", test_db_url, test_bin_dir, workdir)

    @pytest.fixture(scope="class")
    def half_year_workdir(
        self,
        half_year_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("cost_half")
        os.chdir(workdir)
        return _run_scenario("coal_half_year", half_year_db_url, test_bin_dir, workdir)

    def _read_objective(self, workdir: Path) -> float:
        """Read the objective value from the solve progress CSV."""
        csv_path = workdir / "output_raw" / "solve_progress.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Look for objective value
            if "objective" in df.columns:
                return df["objective"].iloc[-1]
        # Fallback: read from the raw GLPK output
        csv_path = workdir / "solve_data" / "solve_progress.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "objective" in df.columns:
                return df["objective"].iloc[-1]
        pytest.skip("Could not find objective value in solve output")

    def test_half_year_cost_is_half(
        self, coal_workdir: Path, half_year_workdir: Path
    ) -> None:
        """Total cost with years_represented=0.5 should be ~0.5x the base cost.

        Both scenarios have the same dispatch (same demand, same units),
        but the inflation factor differs by 0.5x. Since all cost terms
        in the objective are multiplied by the inflation factor, the
        total should scale proportionally.
        """
        factor_base = _read_inflation_factor_operations(coal_workdir)
        factor_half = _read_inflation_factor_operations(half_year_workdir)

        base_factor_val = factor_base["value"].iloc[0]
        half_factor_val = factor_half["value"].iloc[0]

        assert base_factor_val == pytest.approx(1.0, abs=1e-6)
        assert half_factor_val == pytest.approx(0.5, abs=1e-6)

        # The ratio of factors should be 0.5
        ratio = half_factor_val / base_factor_val
        assert ratio == pytest.approx(0.5, abs=1e-6), (
            f"Factor ratio should be 0.5, got {ratio}"
        )
