"""Parametrized end-to-end scenario tests for FlexTool.

Each test:
  1. Uses the session-scoped SQLite DB (imported from test/fixtures/tests.json)
  2. Runs FlexToolRunner for the given scenario in an isolated tmp workdir
  3. Calls write_outputs(write_methods=['csv'])
  4. Compares selected CSVs against golden files in test/expected/<scenario>/

Scenario definitions live in test/scenarios.yaml — edit that file to add
or modify scenarios. See test/README.md for the full workflow.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

TEST_DIR = Path(__file__).parent
EXPECTED_DIR = TEST_DIR / "expected"
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import round_for_comparison  # noqa: E402

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.process_outputs.write_outputs import write_outputs


def _load_scenarios() -> list[tuple[str, list[str]]]:
    with open(TEST_DIR / "scenarios.yaml") as f:
        entries = yaml.safe_load(f)
    return [(e["scenario"], e["csvs"]) for e in entries]


# CSVs with non-standard formatting that pd.read_csv cannot parse.
# These are compared as plain text instead.
_FREEFORM_CSVS = {"summary_solve.csv"}


def _is_freeform_csv(csv_name: str) -> bool:
    return csv_name in _FREEFORM_CSVS


def _strip_timestamps(text: str) -> str:
    """Remove run-specific timestamps so freeform CSVs can be compared across runs."""
    return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2}", "<TIMESTAMP>", text)


SCENARIOS = _load_scenarios()


@pytest.mark.parametrize("scenario,csvs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_scenario(
    scenario: str,
    csvs: list[str],
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
    request: pytest.FixtureRequest,
) -> None:
    regenerate = request.config.getoption("--regenerate")

    # Run the model
    runner = FlexToolRunner(
        input_db_url=test_db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(test_db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Model run failed for scenario '{scenario}'"

    # Write CSV outputs
    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
    )

    # Compare (or regenerate) each expected CSV
    if regenerate == scenario:
        for csv_name in csvs:
            actual_path = workdir / "output_csv" / scenario / csv_name
            assert actual_path.exists(), (
                f"Expected output not found: {actual_path}\n"
                f"Check that '{csv_name}' is a valid filename from templates/default_plots.yaml"
            )
            expected_path = EXPECTED_DIR / scenario / csv_name
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            if _is_freeform_csv(csv_name):
                shutil.copy2(actual_path, expected_path)
            else:
                round_for_comparison(pd.read_csv(actual_path)).to_csv(expected_path, index=False)
        pytest.skip(f"Regenerated {len(csvs)} file(s) for scenario '{scenario}'")
    else:
        for csv_name in csvs:
            actual_path = workdir / "output_csv" / scenario / csv_name
            expected_path = EXPECTED_DIR / scenario / csv_name

            assert actual_path.exists(), (
                f"Expected output not found: {actual_path}\n"
                f"Check that '{csv_name}' is a valid filename from templates/default_plots.yaml"
            )
            assert expected_path.exists(), (
                f"No golden file at {expected_path.relative_to(REPO_ROOT)}\n"
                f"Generate it with: pytest test/ --regenerate {scenario}"
            )

            if _is_freeform_csv(csv_name):
                actual_text = _strip_timestamps(actual_path.read_text())
                expected_text = _strip_timestamps(expected_path.read_text())
                assert actual_text == expected_text, (
                    f"{scenario}/{csv_name} content differs from expected"
                )
            else:
                actual = round_for_comparison(pd.read_csv(actual_path))
                expected = round_for_comparison(pd.read_csv(expected_path))
                pd.testing.assert_frame_equal(
                    actual,
                    expected,
                    check_exact=False,
                    rtol=1e-4,
                    obj=f"{scenario}/{csv_name}",
                )
