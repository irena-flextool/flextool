"""Tier 8 — obj-decomposition parity tests.

For every scenario in ``tests/scenarios.yaml``:

* run ``FlexToolRunner.write_input`` → ``run_model`` → ``write_outputs(['csv'])``;
* read ``output_csv/<scenario>/costs_discounted.csv`` and sum every
  numeric value (the per-category objective breakdown);
* read ``output_csv/<scenario>/summary_solve.csv`` and parse the
  "Total cost (calculated) full horizon (M CUR)" row;
* assert the two values agree to ``1e-6`` relative.

A failure pinpoints the obj writer summing something twice, dropping a
term, or getting a sign wrong — the diagnostic gap that the existing
golden-CSV regression suite cannot localise on its own.

If a scenario does not emit ``costs_discounted.csv`` (pure-dispatch
runs where the writer drops it), the test is skipped — the
decomposition assertion is meaningless without a per-category file.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent.parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from test_scenarios import (  # noqa: E402
    OUTPUT_CONFIG,
    SCENARIOS,
)

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.engine_polars._flex_data_provider import FlexDataProvider  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402

from tests.decomposition._helpers import (  # noqa: E402
    parse_costs_discounted,
    parse_summary_obj,
)


# Re-wrap each SCENARIOS entry to carry just (scenario_name, db_fixture).
# csvs / expected_objective / time_budget_seconds are irrelevant here.
# The db_fixture column (last in SCENARIOS' parametrize tuple) selects
# which DB fixture the scenario needs — main vs. stochastic etc. Pytest
# `-k '<scenario>'` still selects the same case as in test_scenarios.py
# because the `id` is preserved.
_DECOMP_PARAMS = [
    pytest.param(p.values[0], p.values[-1], marks=p.marks, id=p.id)
    for p in SCENARIOS
]


@pytest.mark.decomposition
@pytest.mark.parametrize("scenario,db_fixture", _DECOMP_PARAMS)
def test_obj_decomposition(
    scenario: str,
    db_fixture: str,
    scenario_db_url: str,
    test_solver_config_dir: Path,
    workdir: Path,
) -> None:
    # Δ.22: SolverRunner.run is gone; run via the native cascade.
    # keep_solutions=True so the solve_steps bundle below carries
    # per-sub-solve flex_data + solution; csv_dump=True snapshots
    # solve_data/ so costs_discounted.csv etc. land on disk.
    steps = run_chain_from_db(
        scenario_db_url,
        scenario,
        work_folder=workdir,
        csv_dump=True,
        keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for scenario '{scenario}'"
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal, (
        f"Model run failed for scenario '{scenario}'"
    )

    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
        raw_output_dir=str(workdir / "output_raw"),
        solution=last_step.solution,
        solve_name=last_step.solve_name,
        solve_steps=[
            (s.solve_name, s.flex_data, s.effective_solution)
            for s in steps.values()
        ],
        # In-memory path requires a Provider; this test decomposes the
        # objective, not group flows, so an empty Provider keeps behaviour
        # identical while satisfying the contract.
        flex_data_provider=FlexDataProvider(),
    )

    out_dir = workdir / "output_csv" / scenario
    costs_path = out_dir / "costs_discounted.csv"
    summary_path = out_dir / "summary_solve.csv"

    if not costs_path.exists():
        pytest.skip(
            f"costs_discounted.csv not emitted for {scenario!r} "
            f"(pure-dispatch / writer drops it)"
        )

    assert summary_path.exists(), (
        f"summary_solve.csv missing for scenario '{scenario}' "
        f"— required for decomposition check"
    )

    decomposition_total = parse_costs_discounted(costs_path)
    obj_total = parse_summary_obj(summary_path)

    assert math.isclose(
        decomposition_total, obj_total, rel_tol=1e-6, abs_tol=1e-9
    ), (
        f"obj decomposition mismatch: scenario={scenario} "
        f"sum(costs_discounted.csv)={decomposition_total!r} "
        f"summary_solve obj={obj_total!r} "
        f"abs_err={abs(decomposition_total - obj_total)!r}"
    )
