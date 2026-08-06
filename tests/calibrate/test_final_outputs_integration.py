"""End-to-end check that the calibrator's post-loop replay actually works.

The unit tests in ``test_final_outputs.py`` stub the engine writer; this one
drives the REAL path: solve a small scenario, write its ``output_parquet``
tree exactly as a calibration iteration does (``--write-methods parquet``),
then call :func:`write_final_outputs` — the same call the CLI makes after the
loop — and confirm it regenerates the csv (and excel) formats from that
parquet, with no second solve.

Marked ``solver`` (needs HiGHS); reuses the session-scoped ``tests.json`` DB
and solver-config fixtures from ``tests/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flextool.calibrate import write_final_outputs

TEST_DIR = Path(__file__).resolve().parent.parent  # tests/
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402

pytestmark = pytest.mark.solver

# Small single-solve scenario in tests.json (units + a unidirectional
# connection) — the same one the spinedb-replay round-trip test uses.
SCENARIO = "unidirectional_connection"


def test_write_final_outputs_replays_real_parquet(
    test_db_url: str,
    test_solver_config_dir: Path,
    tmp_path: Path,
) -> None:
    # 1. Solve the scenario.
    steps = run_chain_from_db(
        input_db_url=test_db_url,
        scenario_name=SCENARIO,
        work_folder=tmp_path,
        solver_config_dir=test_solver_config_dir,
        warm=True,
        keep_solutions=True,
    )
    assert steps, f"no steps for scenario {SCENARIO!r}"
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal

    # 2. Write ONLY parquet, mirroring what a calibration iteration leaves at
    #    <out_root>/output_parquet/<scenario>/.
    write_outputs(
        scenario_name=SCENARIO,
        output_location=str(tmp_path),
        subdir=SCENARIO,
        output_config_path=str(REPO_ROOT / "templates" / "default_plots.yaml"),
        fallback_output_location=str(tmp_path),
        raw_output_dir=str(tmp_path / "output_raw"),
        write_methods=["parquet"],
        solution=last_step.solution,
        solve_name=last_step.solve_name,
        solve_steps=[
            (s.solve_name, s.flex_data, s.effective_solution)
            for s in steps.values()
        ],
        flex_data_provider=last_step.flex_data_provider,
    )
    parquet_dir = tmp_path / "output_parquet" / SCENARIO
    assert list(parquet_dir.glob("*.parquet")), "no parquet produced to replay"

    # 3. The post-loop step under test: regenerate csv + excel from that parquet
    #    alone (no solve object passed).
    written = write_final_outputs(tmp_path, SCENARIO, ["csv", "excel"])
    assert written == ["csv", "excel"]

    # 4. The regular output formats now exist alongside the parquet.
    csv_dir = tmp_path / "output_csv" / SCENARIO
    csvs = list(csv_dir.glob("*.csv"))
    assert csvs, "csv replay produced no files"

    excel_file = tmp_path / "output_excel" / f"output_{SCENARIO}.xlsx"
    assert excel_file.is_file(), "excel replay produced no workbook"
