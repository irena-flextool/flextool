"""Tests for the calibrator's pure post-solve readers.

These point the readers at REAL solve parquets.  The recon left a live
solve at ``output_parquet/<SCENARIO>/`` in the repo; when present (the
common case) the reader assertions run against it fast.  If it is absent, a
tiny solve is generated once from the nested-invest fixture — in that case
the whole module is marked ``solver``/``slow`` (the generation path drives
HiGHS).

The missing-``cost_node_discounted`` case is exercised directly with a
directory that holds no cost parquet, asserting the zero-slack tolerance
``(0.0, {})`` rather than an error.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _TESTS_DIR.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from flextool.calibrate._readers import (  # noqa: E402
    read_curtailment_by_sink,
    read_residual_unserved,
    read_slack_penalty,
)

SCENARIO = "multi_fullYear_battery_nested_24h_invest_one_solve"
_REAL_DIR = _REPO_ROOT / "output_parquet" / SCENARIO

# The checked-in real solve is the fast path.  Only when it is absent do we
# generate one (which drives HiGHS) — mark the whole module slow then.
pytestmark = (
    [] if _REAL_DIR.is_dir() else [pytest.mark.solver, pytest.mark.slow]
)


def _generate_solve(dest_root: Path) -> Path:
    """Solve the nested-invest fixture once; return its assess dir."""
    from db_utils import json_to_db  # noqa: PLC0415

    from flextool.calibrate._solve import run_solve  # noqa: PLC0415

    url = json_to_db(
        _TESTS_DIR / "fixtures" / "tests.json", dest_root / "readers.sqlite"
    )
    run = run_solve(
        url,
        SCENARIO,
        work_dir=dest_root / "work",
        out_root=dest_root / "out",
        cache_dir=dest_root / "cache",
    )
    assert run.assess_dir.is_dir(), run.stdout[-2000:]
    return run.assess_dir


@pytest.fixture(scope="module")
def assess_dir(tmp_path_factory) -> Path:
    if _REAL_DIR.is_dir():
        return _REAL_DIR
    dest = tmp_path_factory.mktemp("readers_solve")
    return _generate_solve(dest)


def test_residual_is_per_node_nonneg_floats(assess_dir: Path):
    residual = read_residual_unserved(assess_dir)
    assert isinstance(residual, dict)
    assert residual, "at least one node in the slack table"
    for node, val in residual.items():
        assert isinstance(node, str)
        assert isinstance(val, float)
        assert val >= 0.0, f"unserved energy must be non-negative ({node})"


def test_curtailment_groups_by_sink(assess_dir: Path):
    curt = read_curtailment_by_sink(assess_dir)
    assert isinstance(curt, dict)
    # Every key is a sink-node name; every value a non-negative float.
    for sink, val in curt.items():
        assert isinstance(sink, str)
        assert isinstance(val, float)
        assert val >= 0.0


def test_penalty_reads_total_and_per_node(assess_dir: Path):
    total, by_node = read_slack_penalty(assess_dir)
    assert isinstance(total, float)
    assert isinstance(by_node, dict)
    # Total is the sum of the per-node penalties (within float tolerance).
    assert total == pytest.approx(sum(by_node.values()), abs=1e-9)


def test_penalty_tolerates_missing_cost_file():
    """A dir without the cost parquet yields ``(0.0, {})``, not an error."""
    empty = Path(tempfile.mkdtemp())
    assert read_slack_penalty(empty) == (0.0, {})


def test_curtailment_tolerates_missing_file():
    """A dir without the curtailment parquet yields ``{}``."""
    empty = Path(tempfile.mkdtemp())
    assert read_curtailment_by_sink(empty) == {}
