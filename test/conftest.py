"""pytest fixtures for FlexTool scenario tests.

CWD notes (Task 1 findings)
---------------------------
FlexToolRunner writes several files relative to CWD:
  solve_data/          — auto-created by __init__; holds solve progress CSVs
  output_raw/          — created by glpsol Phase 3; holds raw solver output CSVs
  HiGHS.log            — written by HiGHS
  output/              — created by write_outputs; holds output solve_progress.csv

Intermediate solver files go to root_dir (not CWD):
  flextool.mps, flextool.sol, glpsol_solution.txt

The workdir fixture uses monkeypatch.chdir() to isolate each test's CWD.
root_dir is set to workdir so intermediate files also stay in the temp dir.

bin_dir must contain highs.opt (+ solver binaries).
A session-scoped test_bin_dir fixture creates a temp dir with the test
highs.opt and symlinks to the actual solver binaries.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
REPO_ROOT = TEST_DIR.parent

# Make db_utils importable from this directory
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db, round_for_comparison  # noqa: E402

__all__ = ["round_for_comparison"]  # re-export for test modules


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate",
        metavar="SCENARIO",
        default=None,
        help=(
            "Regenerate expected CSVs for the given scenario name instead of comparing. "
            "Example: pytest test/ --regenerate coal"
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """When --regenerate is given, run only the matching scenario test."""
    regenerate = config.getoption("--regenerate", default=None)
    if not regenerate:
        return
    selected, deselected = [], []
    for item in items:
        params = getattr(item, "callspec", None) and item.callspec.params
        if params and params.get("scenario") == regenerate:
            selected.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Import JSON fixture → fresh SQLite DB once per test session."""
    db_path = tmp_path_factory.mktemp("db") / "tests.sqlite"
    return json_to_db(FIXTURES_DIR / "tests.json", db_path)


@pytest.fixture(scope="session")
def test_bin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temp bin dir with test highs.opt and symlinked solver binaries.

    Using test/highs.opt instead of bin/highs.opt ensures deterministic,
    tight-precision solves for golden file comparisons.
    """
    repo_bin = REPO_ROOT / "bin"
    bin_dir = tmp_path_factory.mktemp("bin")

    for binary in ["highs", "glpsol", "highs.exe", "glpsol.exe"]:
        src = repo_bin / binary
        if src.exists():
            (bin_dir / binary).symlink_to(src)

    shutil.copy(TEST_DIR / "highs.opt", bin_dir / "highs.opt")
    return bin_dir


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test isolated working directory.

    FlexToolRunner writes to CWD (solve_data/, output_raw/, HiGHS.log).
    monkeypatch.chdir ensures each test gets a clean CWD and leaves no
    files in the repo root.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


