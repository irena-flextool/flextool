"""Tier 7 emission test — ``storage_state_start_binding`` row count.

Δ.22 ported from MPS-parsing to direct polar_high ``Problem`` inspection.

Invariant
---------
The cascade emits ``storage_state_start_binding`` (see
``flextool/engine_polars/model.py:2224``) over

    fixed_first_dt = nodeState_first_dt ⋈ storage_fix_start_filtered

where ``storage_fix_start_filtered`` is ``storage_fix_start`` anti-joined
with ``storage_bind_forward_only`` and ``storage_bind_within_solve``.
The constraint is gated on ``is_solve_first`` plus the presence of
``p_state_start``, ``p_state_existing_capacity`` and ``p_state_unitsize``
— all populated for the ``wind_battery`` fixture.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, run_chain_from_db

_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.emission
def test_storage_state_start_binding_emits_only_in_period_first(
    test_db_url: str,
) -> None:
    scenario = "wind_battery"
    with tempfile.TemporaryDirectory() as wd:
        steps = run_chain_from_db(
            test_db_url, scenario,
            work_folder=Path(wd), csv_dump=False, keep_solutions=True,
        )
        last = next(reversed(steps.values()))
        assert last.flex_data is not None

        pb = Problem()
        build_flextool(pb, last.flex_data)

    fd = last.flex_data

    # Expected per the cascade's derivation in model.py:2186-2226.
    fix_start = fd.storage_fix_start
    assert fix_start is not None and fix_start.height > 0, (
        "wind_battery fixture must populate storage_fix_start for this "
        "constraint to fire — fixture regression if 0")
    for excl in (fd.storage_bind_forward_only, fd.storage_bind_within_solve):
        if excl is not None and excl.height > 0:
            fix_start = fix_start.join(excl, on="n", how="anti")
    expected_over = fd.nodeState_first_dt.join(fix_start, on="n", how="inner")
    expected = expected_over.height

    actual = pb.cstr_row_count("storage_state_start_binding")
    # cstr_row_count is prefix-matched and also covers
    # ``storage_state_start_binding_cyclic_period``.  Restrict to the
    # exact-name family.
    bare = next(
        (r for r in pb.cstrs_named("storage_state_start_binding")
         if r.name == "storage_state_start_binding"),
        None,
    )
    bare_count = len(bare.over) if bare is not None else 0
    assert bare_count == expected, (
        f"storage_state_start_binding row count mismatch: actual="
        f"{bare_count} expected={expected} (|nodeState_first_dt ⋈ "
        f"filtered_storage_fix_start|={expected_over.height}); "
        f"prefix-total {actual}"
    )
