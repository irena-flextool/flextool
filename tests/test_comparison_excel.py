"""Tests for comparison Excel output.

Verifies that the scenario comparison orchestrator writes the Excel file
to the correct directory (plot_dir / output_plot_comparisons).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd


class TestComparisonExcelDirectory(unittest.TestCase):
    """Test that comparison Excel is written to the plot_dir."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.plot_dir = self.test_dir / "output_plot_comparisons"
        self.plot_dir.mkdir()

        # Create a minimal config file
        self.config_path = self.test_dir / "config.yaml"
        self.config_path.write_text(
            "plots:\n  test:\n    plot_name: test\n"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_excel_written_to_plot_dir(self) -> None:
        """Comparison Excel file should be created inside plot_dir."""
        from flextool.scenario_comparison.data_models import TimeSeriesResults

        # Create minimal fake results
        fake_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        fake_results = TimeSeriesResults()
        fake_results.unit_flow__dt = fake_df

        fake_scenario_folders = {"scen_a": "/fake/a", "scen_b": "/fake/b"}

        # Mock the heavy-lifting functions to avoid needing real data
        with (
            patch(
                "flextool.scenario_comparison.orchestrator.get_scenario_results",
                return_value=(fake_scenario_folders, fake_results),
            ),
            patch(
                "flextool.scenario_comparison.orchestrator.combine_dispatch_mappings",
            ),
            patch(
                "flextool.scenario_comparison.orchestrator.plot_dict_of_dataframes",
            ),
        ):
            from flextool.scenario_comparison.orchestrator import run

            run(
                db_url=None,
                parquet_subdir="",
                plot_dir=str(self.plot_dir),
                output_config_path=str(self.config_path),
                active_configs=["default"],
                plot_rows=[0, 167],
                write_to_xlsx=True,
                write_dispatch_xlsx=False,
                write_to_ods=False,
                show_plots=False,
                dispatch_plots=False,
                scenario_folders=fake_scenario_folders,
            )

        # The Excel file should be in plot_dir, not in a separate directory
        xlsx_files = list(self.plot_dir.glob("*.xlsx"))
        self.assertTrue(
            len(xlsx_files) > 0,
            f"No .xlsx files found in {self.plot_dir}. "
            f"Contents: {list(self.plot_dir.iterdir())}",
        )

        # Verify the file name matches expected pattern
        self.assertEqual(xlsx_files[0].name, "compare_2_scens.xlsx")

    def test_excel_not_written_to_separate_directory(self) -> None:
        """Ensure no 'output_excel_comparison' directory is created."""
        from flextool.scenario_comparison.data_models import TimeSeriesResults

        fake_df = pd.DataFrame({"a": [1]})
        fake_results = TimeSeriesResults()
        fake_results.unit_flow__dt = fake_df

        fake_scenario_folders = {"scen_a": "/fake/a"}

        old_cwd = os.getcwd()
        os.chdir(str(self.test_dir))
        try:
            with (
                patch(
                    "flextool.scenario_comparison.orchestrator.get_scenario_results",
                    return_value=(fake_scenario_folders, fake_results),
                ),
                patch(
                    "flextool.scenario_comparison.orchestrator.combine_dispatch_mappings",
                ),
                patch(
                    "flextool.scenario_comparison.orchestrator.plot_dict_of_dataframes",
                ),
            ):
                from flextool.scenario_comparison.orchestrator import run

                run(
                    db_url=None,
                    parquet_subdir="",
                    plot_dir=str(self.plot_dir),
                    output_config_path=str(self.config_path),
                    active_configs=["default"],
                    plot_rows=[0, 167],
                    write_to_xlsx=True,
                    write_dispatch_xlsx=False,
                    write_to_ods=False,
                    show_plots=False,
                    dispatch_plots=False,
                    scenario_folders=fake_scenario_folders,
                )
        finally:
            os.chdir(old_cwd)

        # The old buggy directory should NOT exist
        wrong_dir = self.test_dir / "output_excel_comparison"
        self.assertFalse(
            wrong_dir.exists(),
            "output_excel_comparison directory should not be created",
        )


if __name__ == "__main__":
    unittest.main()
