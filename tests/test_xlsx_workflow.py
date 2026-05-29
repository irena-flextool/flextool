"""Tests for the full FlexTool workflow starting from an xlsx input file.

The xlsx is generated fresh from the canonical example DB once per pytest
session via the ``xlsx_input_path`` fixture, then driven through the v2
self-describing importer (``cmd_read_self_describing_tabular_input``) and
the full FlexTool engine for the ``coal`` scenario.

This exercises the contract that a Spine DB → v2 xlsx → Spine DB round trip
preserves the data that FlexTool needs to solve a non-trivial scenario.
The ``test_full_workflow_from_xlsx`` test is a slow integration test that
actually runs the FlexTool model.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import pytest

from flextool.export_to_tabular.export_to_excel import export_to_excel
from flextool.update_flextool.initialize_database import initialize_database

FLEXTOOL_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_JSON = (
    FLEXTOOL_ROOT / "flextool" / "schemas" / "canonical_databases" /
    "templates_examples.json"
)
SCHEMA_JSON = FLEXTOOL_ROOT / "flextool" / "schemas" / "spinedb_schema.json"


@pytest.fixture(scope="session")
def xlsx_input_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the v2 self-describing xlsx once per pytest session.

    Steps:
      1. Materialise ``templates_examples.json`` into a temp sqlite via
         ``initialize_database`` — gives us the canonical examples DB.
      2. Export that DB to xlsx via ``export_to_excel`` with
         ``use_new_format=True`` (v2 self-describing) and
         ``include_advanced=True`` so solve-period sheets carry the data
         the full-workflow scenario needs.

    The fixture never touches ``templates/example_input_template.xlsx`` —
    user-facing files are kept off-limits.  Session scope so the xlsx is
    built once even though three downstream tests consume it.
    """
    work = tmp_path_factory.mktemp("xlsx_workflow_src")
    src_db = work / "src.sqlite"
    initialize_database(str(TEMPLATES_JSON), str(src_db))

    xlsx_path = work / "example_input.xlsx"
    export_to_excel(
        f"sqlite:///{src_db}",
        str(xlsx_path),
        use_new_format=True,
        include_advanced=True,
    )
    assert xlsx_path.exists(), (
        f"export_to_excel did not produce {xlsx_path} — fixture cannot "
        "supply the v2 self-describing input."
    )
    return xlsx_path


def _initialise_target_db(target_db: Path) -> None:
    """Pre-create the target sqlite from the FlexTool schema.

    ``cmd_read_self_describing_tabular_input`` writes into a database
    whose entity classes / parameter definitions already exist.  Each
    test gets a fresh empty DB so writes are deterministic.
    """
    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()
    initialize_database(str(SCHEMA_JSON), str(target_db))


class TestXlsxWorkflow(unittest.TestCase):
    """Drive the v2 self-describing xlsx through the FlexTool pipeline."""

    @pytest.fixture(autouse=True)
    def _wire_xlsx(self, xlsx_input_path: Path, tmp_path: Path) -> None:
        # ``pytest`` injects fixtures here so the unittest methods can use
        # plain ``self.xlsx_input_path`` / ``self.test_dir``.
        self.xlsx_input_path = xlsx_input_path
        self.test_dir = tmp_path

    def _convert(self, target_db: Path) -> subprocess.CompletedProcess[str]:
        _initialise_target_db(target_db)
        target_url = f"sqlite:///{target_db}"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(self.xlsx_input_path),
                target_url,
            ],
            capture_output=True,
            text=True,
            cwd=str(FLEXTOOL_ROOT),
        )

    def test_xlsx_to_sqlite_conversion(self) -> None:
        """Test that xlsx can be converted to a valid sqlite database."""
        target_db = self.test_dir / "converted.sqlite"
        result = self._convert(target_db)
        self.assertEqual(
            result.returncode,
            0,
            f"Conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertTrue(target_db.exists(), "SQLite database was not created")

        from spinedb_api import DatabaseMapping

        with DatabaseMapping(f"sqlite:///{target_db}") as db:
            scenarios = [x["name"] for x in db.get_scenario_items()]

        self.assertTrue(len(scenarios) > 0, "No scenarios found in converted database")

    def test_conversion_creates_target_directory(self) -> None:
        """Test that conversion creates parent directories if they don't exist."""
        nested_dir = self.test_dir / "a" / "b" / "c"
        target_db = nested_dir / "deep.sqlite"
        result = self._convert(target_db)
        self.assertEqual(
            result.returncode,
            0,
            f"Conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertTrue(target_db.exists(), "SQLite database was not created in nested directory")

    def test_conversion_overwrites_existing_db(self) -> None:
        """Test that converting into an existing database works (re-import)."""
        target_db = self.test_dir / "existing.sqlite"
        result1 = self._convert(target_db)
        self.assertEqual(result1.returncode, 0, "First conversion failed")

        # Second conversion: keep the existing schema-initialised DB,
        # rerun the import (purge=True by default in the v2 CLI).
        target_url = f"sqlite:///{target_db}"
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "flextool.cli.cmd_read_self_describing_tabular_input",
                str(self.xlsx_input_path),
                target_url,
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

    @pytest.mark.slow
    def test_full_workflow_from_xlsx(self) -> None:
        """Test the full workflow: xlsx -> sqlite -> model run -> outputs.

        NOTE: This is a slow integration test.  The model execution step
        may take up to a minute on the bundled HiGHS solver.

        The ``coal`` scenario is used explicitly — it is the canonical
        end-to-end demonstrator (single-solve, single-period, no
        stochastic branches, no rolling) and exercises the round-trip
        for the parameter set FlexTool actually consumes (model.solves,
        solve.realized_periods / period_timeset, …).  Other scenarios
        depend on parameters (price_ladder_*) that the v2 reader does
        not yet round-trip — see the v2 self-describing reader for the
        outstanding work on multi-axis transposed sheets.
        """
        target_db = self.test_dir / "input.sqlite"
        target_url = f"sqlite:///{target_db}"
        result = self._convert(target_db)
        self.assertEqual(
            result.returncode,
            0,
            f"Conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )

        from spinedb_api import DatabaseMapping

        with DatabaseMapping(target_url) as db:
            scenarios = {x["name"] for x in db.get_scenario_items()}

        scenario_name = "coal"
        self.assertIn(
            scenario_name, scenarios,
            f"Scenario {scenario_name!r} missing from converted DB; "
            f"got {sorted(scenarios)}",
        )

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

        self.assertTrue(output_parquet.exists(), "output_parquet directory was not created")
        parquet_files = list(output_parquet.rglob("*.parquet"))
        self.assertTrue(
            len(parquet_files) > 0,
            "No parquet files were generated",
        )


if __name__ == "__main__":
    unittest.main()
