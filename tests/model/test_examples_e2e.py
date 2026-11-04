"""End-to-end cascade smoke tests for scenarios of the shipped example database.

The database is built per module from the canonical
``schemas/canonical_databases/templates_examples.json`` (never from a
local or checked-in ``examples.sqlite``, which lags the schema and the
example data — CLAUDE.md invariant #3). Three of its scenarios each exercise a wide swath of the FlexTool cascade and each surfaced a
distinct, previously-uncaught bug when first run by a human against the
native cascade entry point:

* ``test_a_lot`` — exercises a large fraction of parameter shapes and
  cross-feature interactions; uncovered ARITH-1.
* ``fullYear_roll`` — full-year rolling solve with active reserves;
  uncovered RESERVE-1.
* ``multi_fullYear_battery_nested_multi_invest`` — multi-solve handoff
  with nested-multi-invest period structure; uncovered PERIODS-1
  (and REGRESS-1).

These tests assert the bare minimum smoke contract: the cascade runs
to completion and every sub-solve reports ``optimal``.  Until the bugs
above are fixed, the tests are expected to FAIL with the recorded
symptoms — that is the regression gate the bug-fix work will close.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool.engine import run_chain_from_db
from flextool.update_flextool.initialize_database import initialize_database


pytestmark = pytest.mark.solver


@pytest.fixture(scope="module")
def examples_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialise ``templates_examples.json`` into a fresh tmp SQLite."""
    json_src = package_data_path("schemas/canonical_databases/templates_examples.json")
    sqlite_path = tmp_path_factory.mktemp("examples_db") / "examples.sqlite"
    initialize_database(str(json_src), str(sqlite_path))
    return sqlite_path


@pytest.mark.parametrize(
    "scenario",
    [
        "test_a_lot",
        "fullYear_roll",
        "multi_fullYear_battery_nested_multi_invest",
    ],
)
def test_examples_scenario_solves(scenario: str, examples_db: Path, tmp_path: Path) -> None:
    """End-to-end cascade smoke test against the shipped example database.

    The minimum bar: ``run_chain_from_db`` returns at least one
    sub-solve and every sub-solve reports ``optimal``.  Tightening
    (objective / output-frame assertions) is a follow-up; this is the
    coverage gap.
    """
    # Solve from a per-test copy so concurrent parametrisations never share
    # one sqlite file (Windows read-lock deadlocks).
    db_copy = tmp_path / "examples.sqlite"
    db_copy.write_bytes(examples_db.read_bytes())

    steps = run_chain_from_db(
        db_copy,
        scenario_name=scenario,
        work_folder=tmp_path,
        csv_dump=False,
    )
    assert steps, (
        f"cascade returned no sub-solves for scenario {scenario!r}"
    )
    non_optimal = [
        name for name, step in steps.items()
        if not step.optimal
    ]
    assert not non_optimal, (
        f"scenario {scenario!r}: the following sub-solves were not "
        f"optimal: {non_optimal}"
    )
