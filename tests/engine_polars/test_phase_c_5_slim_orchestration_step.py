"""Phase C.5 — slim ``OrchestrationStep`` memory-discipline contract.

The cascade default (``keep_solutions=False``) holds full per-step
state — ``solution`` + ``flex_data`` + ``flex_data_accumulator`` —
ONLY on the LAST step.  Earlier steps store slim summary fields
(``solve_name``, ``obj``, ``optimal``, ``warm_used``, ``handoff``)
and clear the heavy slots to release the per-sub-solve HiGHS
instance + variable arrays + writer-frame snapshot.

Opt-in: ``run_chain_from_db(..., keep_solutions=True)`` retains the
full per-step state for tests that need parity sweeps over every
sub-solve's ``solution`` / ``flex_data``.

This test exercises both branches on a small 2-sub-solve fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import run_chain_from_db


pytestmark = pytest.mark.solver


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# 2-sub-solve fixture: ``invest_1year_5weeks`` then
# ``y2020_fullYear_dispatch``.  Small + fast; exercises a non-rolling
# 2-step chain so the "all but last" assertion has at least one
# earlier step.
WORK_NAME = "work_5weeks_invest_fullYear_dispatch_coal_wind"
SCENARIO = "5weeks_invest_fullYear_dispatch_coal_wind"


def _db_or_skip() -> Path:
    fixture = DATA / WORK_NAME
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    return db


def _common_slim_fields_populated(step) -> None:
    """All slim summary fields must survive the per-step memory release."""
    assert step.solve_name, "slim solve_name must remain populated"
    assert step.handoff is not None, "slim handoff must remain populated"
    assert step.obj is not None, "slim obj must remain populated"
    assert step.optimal is True, (
        f"slim optimal must remain populated and True; got {step.optimal!r}"
    )


def test_slim_default_releases_per_step_state(tmp_path: Path) -> None:
    """Default ``keep_solutions=False`` keeps ``solution`` / ``flex_data``
    / ``flex_data_accumulator`` ONLY on the LAST step; clears them on
    earlier steps to release memory.
    """
    db = _db_or_skip()
    work = tmp_path / WORK_NAME
    sols = run_chain_from_db(db, scenario_name=SCENARIO, work_folder=work)
    assert len(sols) >= 2, (
        f"fixture must produce >=2 sub-solves to exercise the slim pass "
        f"(got {len(sols)}: {list(sols)})"
    )

    keys = list(sols)
    last_key = keys[-1]

    # Every step has the slim summary fields populated.
    for step in sols.values():
        _common_slim_fields_populated(step)

    # All but the last step have the heavy slots released.
    for k in keys[:-1]:
        step = sols[k]
        assert step.solution is None, (
            f"slim cascade must release step.solution on non-last step "
            f"{k!r}; still references {type(step.solution).__name__}"
        )
        assert step.flex_data is None, (
            f"slim cascade must release step.flex_data on non-last step "
            f"{k!r}"
        )
        assert step.flex_data_accumulator is None, (
            f"slim cascade must release step.flex_data_accumulator on "
            f"non-last step {k!r}"
        )

    # The last step keeps the heavy slots — cmd_run_flextool reads
    # last_step.flex_data + last_step.solution for write_outputs.
    last = sols[last_key]
    assert last.solution is not None, (
        "last step must retain solution under slim default "
        "(cmd_run_flextool depends on this)"
    )
    assert last.flex_data is not None, (
        "last step must retain flex_data under slim default "
        "(cmd_run_flextool depends on this)"
    )
    # Step 1-f — ``flex_data_accumulator`` is no longer populated by the
    # cascade (the per-sub-solve Provider replaced it).  Step 2 deletes
    # the field outright; for now it stays ``None`` on every step.
    assert last.flex_data_accumulator is None, (
        "Step 1-f — flex_data_accumulator must be None (replaced by Provider)"
    )


def test_keep_solutions_true_retains_every_step(tmp_path: Path) -> None:
    """``keep_solutions=True`` retains ``solution`` / ``flex_data`` /
    ``flex_data_accumulator`` on EVERY step — used by tests that need
    per-step parity / debug access.
    """
    db = _db_or_skip()
    work = tmp_path / WORK_NAME
    sols = run_chain_from_db(
        db, scenario_name=SCENARIO, work_folder=work, keep_solutions=True,
    )
    assert len(sols) >= 2

    for k, step in sols.items():
        _common_slim_fields_populated(step)
        assert step.solution is not None, (
            f"keep_solutions=True must retain step.solution on every "
            f"step; missing on {k!r}"
        )
        assert step.flex_data is not None, (
            f"keep_solutions=True must retain step.flex_data on every "
            f"step; missing on {k!r}"
        )
        # Step 1-f — Provider replaced the accumulator; the field is
        # always None pending Step 2 deletion.
        assert step.flex_data_accumulator is None, (
            f"Step 1-f — flex_data_accumulator must be None (Provider "
            f"replaced it); got non-None on {k!r}"
        )
