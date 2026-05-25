"""COVERAGE-1 — end-to-end cascade smoke tests for `examples.sqlite` scenarios.

Three scenarios from ``projects/examples/input_sources/examples.sqlite``
each exercise a wide swath of the FlexTool cascade and each surfaced a
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

from flextool.engine_polars import run_chain_from_db


pytestmark = pytest.mark.solver


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DB = _REPO_ROOT / "projects" / "examples" / "input_sources" / "examples.sqlite"


@pytest.mark.parametrize(
    "scenario",
    [
        "test_a_lot",
        "fullYear_roll",
        "multi_fullYear_battery_nested_multi_invest",
    ],
)
def test_examples_scenario_solves(scenario: str, tmp_path: Path) -> None:
    """End-to-end cascade smoke test against ``examples.sqlite``.

    The minimum bar: ``run_chain_from_db`` returns at least one
    sub-solve and every sub-solve reports ``optimal``.  Tightening
    (objective / output-frame assertions) is a follow-up; this is the
    coverage gap.
    """
    if not EXAMPLES_DB.is_file():
        pytest.skip(f"examples.sqlite missing: {EXAMPLES_DB}")

    steps = run_chain_from_db(
        EXAMPLES_DB,
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
