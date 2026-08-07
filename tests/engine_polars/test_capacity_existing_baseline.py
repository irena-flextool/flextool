"""Regression: the capacity output ``existing`` column is the static
pre-invest baseline, NOT the cumulative chain-sum that folds in prior
sub-solves' investments.

The bug (reported 2026-08-06): in a multi-solve run the invest→dispatch
handoff folds each earlier sub-solve's ``v_invest`` into
``p_entity_all_existing``.  ``out_capacity`` used to read that cumulative
value for the ``existing`` column, so a later (dispatch / rolling)
sub-solve reported previously-invested capacity under ``existing`` — e.g.
``multi_year`` coal_plant ``p2035`` showed ``existing=823.32`` when the
true pre-invest baseline is ``500``.

The fix sources ``existing`` from ``par.entity_pre_existing`` (baseline);
the cumulative capacity still shows in ``total``.  This test pins the
invariant directly (independent of the golden CSV) so a future blind
golden re-regeneration cannot silently reintroduce the fold.

See ``tests/expected/REGEN_LOG.md`` (2026-08-07 entry) and
``flextool/process_outputs/out_capacity.py``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

pytestmark = pytest.mark.solver

# ``multi_year`` is a 4-sub-solve rolling invest chain (one period per
# sub-solve) with a coal_plant that ships at 500 MW existing and invests
# additional capacity in p2025/p2030 that carries forward into the later
# sub-solves — exactly the fold that produced the bug.
SCENARIO = "multi_year"
COAL_BASELINE = 500.0


@pytest.fixture(scope="module")
def multi_year_unit_capacity(test_db_url, tmp_path_factory) -> pd.DataFrame:
    """Solve ``multi_year`` once and return its ``unit_capacity__d.csv``."""
    from flextool.engine_polars import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    work = tmp_path_factory.mktemp("cap_existing_baseline")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = run_chain_from_db(
            input_db_url=test_db_url, scenario_name=SCENARIO,
            work_folder=work, keep_solutions=True,
        )
        last = next(reversed(steps.values()))
        write_outputs(
            scenario_name=SCENARIO, output_location=str(work), subdir=SCENARIO,
            write_methods=["csv"], fallback_output_location=str(work),
            raw_output_dir=str(work / "output_raw"),
            solution=last.solution, solve_name=last.solve_name,
            solve_steps=[(s.solve_name, s.flex_data, s.effective_solution)
                         for s in steps.values()],
            flex_data_provider=last.flex_data_provider,
        )
    csv = work / "output_csv" / SCENARIO / "unit_capacity__d.csv"
    assert csv.exists(), f"unit_capacity__d.csv not written under {csv.parent}"
    return pd.read_csv(csv)


def test_existing_is_pre_invest_baseline(multi_year_unit_capacity):
    """coal_plant ``existing`` stays at the 500 MW baseline every period,
    even in the sub-solves that inherit prior-solve investment."""
    coal = multi_year_unit_capacity[
        multi_year_unit_capacity["unit"] == "coal_plant"
    ].sort_values("period")
    assert len(coal) == 4, f"expected 4 coal_plant rows, got {len(coal)}"
    max_dev = (coal["existing"].astype(float) - COAL_BASELINE).abs().max()
    assert max_dev < 1e-6, (
        "existing column folded in carried-forward invest — expected the "
        f"{COAL_BASELINE} MW pre-invest baseline on every period:\n"
        f"{coal[['period', 'existing', 'invested', 'total']]}"
    )


def test_total_is_cumulative_above_baseline(multi_year_unit_capacity):
    """The cumulative (invest-folded) capacity still appears in ``total`` —
    the fix moves it out of ``existing``, it does not drop it."""
    coal = multi_year_unit_capacity[
        multi_year_unit_capacity["unit"] == "coal_plant"
    ].sort_values("period")
    # p2030 / p2035 inherit p2025's + p2030's investment: total climbs well
    # above the 500 baseline while existing stays at 500.
    late = coal[coal["period"].isin(["p2030", "p2035"])]
    assert (late["total"] > COAL_BASELINE + 1.0).all(), (
        f"total should carry the cumulative invest:\n{late[['period', 'total']]}"
    )
    # Regression guard: the old bug set existing == total on the pure-carry
    # period p2035 (existing=823.32=total).  Assert they now differ.
    p2035 = coal[coal["period"] == "p2035"].iloc[0]
    assert p2035["existing"] < p2035["total"] - 1.0, (
        "existing must not equal the folded cumulative total on p2035 "
        f"(existing={p2035['existing']}, total={p2035['total']})"
    )
