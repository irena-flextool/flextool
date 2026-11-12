"""Reported investment costs of a recourse fan reconcile to the LP objective.

Runs the ``recourse`` scenario of ``stoch_two_period_invest`` through the
FULL output pipeline (``write_outputs``) and asserts the reported
investment-cost breakdown against the LP objective.

``stoch_two_period_invest`` is a 2-forecast fan (``high`` / ``low``, weights
0.25 / 0.75, no realization solve) revealing at its first period (p2035).
Under recourse each forecast decides its own investment; the reported
investment costs inside a fan are each branch's investment weighted by its
branch weight, at the fan's period labels.  The LP objective (0.196875 M)
is exactly that probability-weighted investment::

    0.25·(60·2100 + 30·1050) + 0.75·(80·2100 + 40·1050) = 196 875

(high builds 60 MW in p2035 and 30 more in p2040, low 80 and 40; annuity
1050 per MW and year, a p2035 decision pays two years), so the
"Investment costs for realized periods" total equals the objective.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "flextool" / "schemas" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.engine import run_chain_from_db  # noqa: E402
from flextool.engine.preprocess._flex_data_provider import (  # noqa: E402
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

    # The investment inside the fan is reported branch-weighted, so the
    # reported investment equals the objective (which holds nothing else).
    assert objective == pytest.approx(0.196875, rel=1e-6)
    assert invest_realized == pytest.approx(objective, rel=1e-6)
