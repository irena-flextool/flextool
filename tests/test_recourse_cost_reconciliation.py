"""Reported investment costs of a stochastic investment solve reconcile to the LP objective.

Runs the ``after_reveal`` scenario of ``stoch_two_period_invest`` through the
FULL output pipeline (``write_outputs``) and asserts the reported
investment-cost breakdown against the LP objective.

``stoch_two_period_invest`` is a 2-forecast fan (``high`` / ``low``, weights
0.25 / 0.75, no realization solve) revealing at its first period (p2035).
Under ``investments_after_reveal_period`` the p2035 investment is one decision
of both forecasts and is realized; the p2040 investments are per forecast and
are not (realized periods [p2035]).  Annuity 1 · 1000 · 0.05 / (1 − 1.05^−1) /
1.05 = 1000 per MW and year, paid at the start of the year; a p2035 decision
pays two years (2000 / MW), a p2040 one one year (1000 / MW).  Residual demand
high 60 / 90, low 80 / 120 MW; the shortfall penalty dominates.

The shared p2035 build covers low's 80 MW; p2040 adds high 10, low 40::

    objective = 80·2000 + 0.25·10·1000 + 0.75·40·1000 = 160 000 + 32 500 = 192 500

The reported investment lines (investment and replacement: lifetime 1, the
p2035 plant is replaced in p2040) hold the realized decision only, 80·2000 =
160 000; the rest of the objective is the per-forecast p2040 investment,
32 500, which only the ``output_horizon`` dump reports.
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
    with open(csv_dir / "summary_solve.csv", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == label:
                return float(row[1])
    raise AssertionError(f"summary_solve.csv has no {label!r} row")


@pytest.mark.slow
@pytest.mark.solver
def test_shared_reveal_investment_cost_reconciles_to_objective(
    stoch_two_period_invest_db_url: str,
    test_solver_config_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    import os

    wf = tmp_path_factory.mktemp("recourse_cost")
    os.chdir(wf)
    steps = run_chain_from_db(
        stoch_two_period_invest_db_url,
        "after_reveal",
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
        scenario_name="after_reveal",
        output_location=str(wf),
        subdir="after_reveal",
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
    csv_dir = wf / "output_csv" / "after_reveal"

    objective = _summary_row(csv_dir, "stoch_2p_inv")
    invested = (_summary_row(csv_dir, "Investment (M CUR)")
                + _summary_row(csv_dir, "Replacement (M CUR)"))
    counted = _summary_row(csv_dir, "Total cost counted in the objective (M CUR)")

    assert objective == pytest.approx(0.1925, rel=1e-6)
    assert invested == pytest.approx(0.16, rel=1e-6)
    # Operations cost nothing and no demand is short, so the counted total is the realized investment alone.
    assert counted == pytest.approx(invested, rel=1e-6)
