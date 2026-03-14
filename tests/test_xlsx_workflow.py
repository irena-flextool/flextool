"""Tests for the full FlexTool workflow starting from an xlsx input file.

The ``test_full_workflow_from_xlsx`` test is a slow integration test that
actually runs the FlexTool model.  It may take several minutes.
"""

import os
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path


FLEXTOOL_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = FLEXTOOL_ROOT / "templates" / "example_input_template.xlsx"


@unittest.skipUnless(XLSX_PATH.exists(), f"Test input file not found: {XLSX_PATH}")
class TestXlsxWorkflow(unittest.TestCase):
    """Test the full FlexTool workflow starting from an xlsx input file."""

    def setUp(self) -> None:
        """Create a temporary working directory."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_xlsx_to_sqlite_conversion(self) -> None:
        """Test that xlsx can be converted to a valid sqlite database."""
        target_db = self.test_dir / "converted.sqlite"
        target_url = f"sqlite:///{target_db}"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_url,
                "--tabular-file-path",
                str(XLSX_PATH),
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
        )

        self.assertEqual(
            result.returncode,
            0,
            f"Conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertTrue(target_db.exists(), "SQLite database was not created")

        # Verify scenarios can be read from the created database
        from spinedb_api import DatabaseMapping

        with DatabaseMapping(target_url) as db:
            scenarios = [x["name"] for x in db.get_scenario_items()]

        self.assertTrue(len(scenarios) > 0, "No scenarios found in converted database")

    def test_conversion_creates_target_directory(self) -> None:
        """Test that conversion creates parent directories if they don't exist."""
        nested_dir = self.test_dir / "a" / "b" / "c"
        target_db = nested_dir / "deep.sqlite"
        target_url = f"sqlite:///{target_db}"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_url,
                "--tabular-file-path",
                str(XLSX_PATH),
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
        )

        self.assertEqual(
            result.returncode,
            0,
            f"Conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertTrue(target_db.exists(), "SQLite database was not created in nested directory")

    def test_conversion_overwrites_existing_db(self) -> None:
        """Test that converting into an existing database works (re-import)."""
        target_db = self.test_dir / "existing.sqlite"
        target_url = f"sqlite:///{target_db}"

        # First conversion -- creates the db
        result1 = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_url,
                "--tabular-file-path",
                str(XLSX_PATH),
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
        )
        self.assertEqual(result1.returncode, 0, "First conversion failed")

        # Second conversion -- should succeed (db already exists)
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_url,
                "--tabular-file-path",
                str(XLSX_PATH),
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
        )
        self.assertEqual(
            result2.returncode,
            0,
            f"Second conversion failed:\nstdout: {result2.stdout}\nstderr: {result2.stderr}",
        )

    # Slow integration test -- runs the actual FlexTool model
    def test_full_workflow_from_xlsx(self) -> None:
        """Test the full workflow: xlsx -> sqlite -> model run -> outputs.

        NOTE: This is a slow integration test.  The model execution step
        may take several minutes depending on the machine.
        """
        # Step 1: Convert xlsx to sqlite
        target_db = self.test_dir / "input.sqlite"
        target_url = f"sqlite:///{target_db}"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_tabular_input",
                target_url,
                "--tabular-file-path",
                str(XLSX_PATH),
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

        # Step 2: Get the first scenario name
        from spinedb_api import DatabaseMapping

        with DatabaseMapping(target_url) as db:
            scenarios = [x["name"] for x in db.get_scenario_items()]

        self.assertTrue(len(scenarios) > 0, "No scenarios in converted DB")
        scenario_name = scenarios[0]  # Use first available scenario

        # Step 3: Run FlexTool
        work_folder = self.test_dir / "work"
        work_folder.mkdir()
        output_parquet = self.test_dir / "output_parquet"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_run_flextool",
                target_url,
                "--scenario-name",
                scenario_name,
                "--work-folder",
                str(work_folder),
                "--write-methods",
                "parquet",
                "--output-location",
                str(self.test_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
            timeout=300,  # 5 minute timeout
        )

        self.assertEqual(
            result.returncode,
            0,
            f"FlexTool run failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

        # Step 4: Verify outputs exist
        self.assertTrue(output_parquet.exists(), "output_parquet directory was not created")
        parquet_files = list(output_parquet.rglob("*.parquet"))
        self.assertTrue(
            len(parquet_files) > 0,
            "No parquet files were generated",
        )


if __name__ == "__main__":
    unittest.main()
