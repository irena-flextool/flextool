"""Slice D (I-a, §11.3 / §15.2 D8) — post-hoc invest-cost reconciliation.

Runs the ``recourse`` scenario of ``stoch_two_period_invest`` through the
FULL output pipeline (``write_outputs``) and asserts that the reported
investment cost reconciles to the LP objective's REALIZED invest
contribution — i.e. ``calc_costs`` probability-weights the committed
invest by ``pd_branch_weight`` under recourse (choice I-a), so the cost
breakdown sums back to the objective rather than over-reporting by
``1/w_realized``.

Hand values (design §15.1): committed realized invest is 60 MW at p2035
(annuity 2100) and 30 MW at p2040 (annuity 1050), realized branch weight
0.25:

    invest cost = 0.25·(60·2100 + 30·1050) = 39 375  (0.039375 M CUR)

The UNWEIGHTED (I-b) recompute would report 60·2100 + 30·1050 = 157 500
(0.1575 M) — 1/0.25 = 4× larger — so the assertion discriminates the
weighting.  The full-expectation objective (0.196875 M) additionally
carries the non-realized low-branch contribution, which the committed
cost path correctly omits (§11.1).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.engine_polars._flex_data_provider import (  # noqa: E402
    FlexDataProvider,
)
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402


def _summary_row(csv_dir: Path, label: str) -> float:
    """Return the numeric value on the ``summary_solve.csv`` row whose
    first cell equals ``label`` (the file is an irregular, multi-shape
    CSV, so scan row-by-row)."""
    with open(csv_dir / "summary_solve.csv", newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == label:
                return float(row[1])
    raise AssertionError(f"summary_solve.csv has no {label!r} row")


@pytest.mark.slow
@pytest.mark.solver
def test_recourse_invest_cost_reconciles_to_objective(
    stoch_two_period_invest_db_url: str,
    test_solver_config_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    import os

    wf = tmp_path_factory.mktemp("recourse_cost")
    os.chdir(wf)
    steps = run_chain_from_db(
        stoch_two_period_invest_db_url,
        "recourse",
        work_folder=wf,
        solver_config_dir=test_solver_config_dir,
        csv_dump=True,
        keep_solutions=True,
    )
    last = next(reversed(steps.values()))
    assert last.solution is not None and last.solution.optimal
    provider = getattr(last, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(wf)

    write_outputs(
        scenario_name="recourse",
        output_location=str(wf),
        subdir="recourse",
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(wf),
        raw_output_dir=str(wf / "output_raw"),
        solution=last.solution,
        solve_name=last.solve_name,
        solve_steps=[
            (s.solve_name, s.flex_data, s.effective_solution)
            for s in steps.values()
        ],
        flex_data_provider=FlexDataProvider(),
    )
    csv_dir = wf / "output_csv" / "recourse"

    objective = _summary_row(csv_dir, "stoch_2p_inv")
    invest_realized = _summary_row(
        csv_dir, "Investment costs for realized periods (M CUR)")

    # I-a: the reported invest cost is the probability-weighted realized
    # contribution 0.25·(60·2100 + 30·1050) = 0.039375 M — NOT the
    # unweighted 0.1575 M (which would not reconcile to the objective).
    assert invest_realized == pytest.approx(0.039375, rel=1e-6)
    # The full-expectation objective carries the non-realized branches too.
    assert objective == pytest.approx(0.196875, rel=1e-6)
