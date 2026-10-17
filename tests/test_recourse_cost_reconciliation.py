"""Slice D (§11.3 / §15.2 D8), R3 re-grounded — post-hoc invest-cost
reconciliation under the forecasts-only model.

Runs the ``recourse`` scenario of ``stoch_two_period_invest`` through the
FULL output pipeline (``write_outputs``) and asserts the reported
investment-cost breakdown against the LP objective.

``stoch_two_period_invest`` is now a genuine 2-forecast fan (``high`` /
``low``, n_real = 0) revealing FAN-AT-FIRST (p2035).  Under the corrected
forecasts-only model there is no realized member and no pre-reveal
here-and-now stage, so the real-named trunk periods p2035/p2040 are
weight-0 phantoms: the COMMITTED (realized-only) investment is 0, and the
cost breakdown's "Investment costs for realized periods" is therefore
0 M CUR — the committed-cost path does NOT fold the per-forecast
branch investment (which is wait-and-see, surfaced only under
``output_horizon``) into the realized total.

The LP objective (0.196875 M) carries the full probability-weighted
per-forecast invest expectation
``0.25·(60·2100 + 30·1050) + 0.75·(80·2100 + 40·1050)`` — all of it on the
forecast branches, none of it committed to a realized period.
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

    # Forecasts-only fan-at-first: the committed (realized-only) invest is
    # 0 (the real-named trunk is a weight-0 phantom — no here-and-now
    # investment), so the "Investment costs for realized periods" total is
    # 0 M.  The per-forecast branch investment is wait-and-see and is NOT
    # folded into the realized cost.
    assert invest_realized == pytest.approx(0.0, abs=1e-9)
    # The objective carries the full probability-weighted per-forecast
    # invest expectation.
    assert objective == pytest.approx(0.196875, rel=1e-6)
