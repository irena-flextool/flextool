"""Regression test for ``BUG p_online_dt_empty_no_blocks``
(specs/model_bugs.md).

Bug: ``_writer_per_solve.write_per_solve_sets`` derived ``p_online_dt``
strictly from ``process_block ⋈ block_step_duration``.  When the input
data did not emit a ``process_block`` row for a UC process (true of the
single-solve ``_native_input_writer`` path, which ships a header-only
``process_block.csv``), ``p_online_dt`` collapsed to zero rows and the
entire UC machinery — ``v_online_linear``, ``v_startup_linear``,
``minimum_uptime``, ``maxOnline_linear``, etc. — built over an empty
index.  HiGHS then reduced the model to empty and the LP paid only the
VOLL slack penalty.

Fix: when a process in ``process_online`` has no ``process_block`` row,
fall back to the per-step timeline (``steps_in_use``).  This mirrors
the .mod's pre-v51 behaviour where UC processes without an explicit
block defaulted to the full (d, t) grid, and matches
``_native_input_writer``'s documented intent that empty stubs "fall
through to identity (every entity mapped to 'default')".

This regression test asserts the fallback fires: for the
``coal_wind_min_uptime`` scenario (UC-without-process_block),
``p_online_dt`` is non-empty and equals the per-step timeline size.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from flextool.engine_polars import run_chain_from_db

_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.emission
def test_p_online_dt_fallback_for_uc_without_process_block(
    test_db_url: str,
) -> None:
    """``p_online_dt`` must be populated for UC processes whose input
    data has no ``process_block`` row — see fallback in
    ``_writer_per_solve.write_per_solve_sets``.
    """
    scenario = "coal_wind_min_uptime"
    with tempfile.TemporaryDirectory() as wd:
        steps = run_chain_from_db(
            test_db_url, scenario,
            work_folder=Path(wd), csv_dump=False, keep_solutions=True,
        )
        last = next(reversed(steps.values()))

    fd = last.flex_data
    assert fd is not None

    # There is exactly one UC process (``coal_plant``) in this scenario.
    assert fd.process_online_linear is not None
    assert fd.process_online_linear.height == 1, (
        "Test fixture changed: expected exactly one process in "
        "process_online_linear for coal_wind_min_uptime."
    )

    # Without the fallback, ``p_online_dt`` would be 0 rows.  With the
    # fix, it expands to the per-step timeline size per online process.
    assert fd.p_online_dt is not None
    n_online = fd.process_online_linear.height + (
        fd.process_online_integer.height
        if fd.process_online_integer is not None else 0
    )
    n_dt = fd.dt.height if fd.dt is not None else 0
    assert n_dt > 0, "scenario has empty dt; test is meaningless"
    assert fd.p_online_dt.height == n_online * n_dt, (
        f"p_online_dt rows {fd.p_online_dt.height} != "
        f"online_processes ({n_online}) × dt_size ({n_dt}) — the "
        "fallback in _writer_per_solve.write_per_solve_sets has "
        "regressed.  See specs/model_bugs.md "
        "'BUG p_online_dt_empty_no_blocks'."
    )
