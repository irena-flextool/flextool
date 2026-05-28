"""Tier 7 emission test — ``minimum_uptime`` row count.

Δ.22 ported from MPS-parsing to direct polar_high ``Problem`` inspection.

Invariant
---------
The cascade emits ``minimum_uptime_<linear|integer>`` constraints over
``up_idx = pdt_uptime_set ⋈ online_set`` where ``online_set`` is
``process_online_linear`` for the linear block and
``process_online_integer`` for the integer block.  ``v_online`` (and
therefore the constraint) only materialises at ``(p, d, t)`` tuples in
``p_online_dt`` — so the actual row count is

    |pdt_uptime_set ⋈ (process_online_linear ∪ process_online_integer)
     ⋈ p_online_dt|

summed across the two suffixes.

Bug fixed: ``BUG p_online_dt_empty_no_blocks`` (specs/model_bugs.md) —
``_emit_per_solve.write_per_solve_sets`` now falls back to the
per-step timeline (``steps_in_use``) for UC processes without a
``process_block`` row, so ``p_online_dt`` is non-empty and the UC
constraints (``minimum_uptime``, ``maxOnline_linear`` …) are
populated.  See also ``test_p_online_dt_fallback.py`` for a direct
regression assertion on this fallback.
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
def test_minimum_uptime_emits_correct_rows(test_db_url: str) -> None:
    scenario = "coal_wind_min_uptime"
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

    # Expected per the cascade's derivation:
    #   |pdt_uptime_set ⋈ online_set ⋈ p_online_dt|
    # for each of online_set ∈ {process_online_linear, process_online_integer}.
    expected = 0
    online_blocks = (
        ("linear", fd.process_online_linear),
        ("integer", fd.process_online_integer),
    )
    for _, online_set in online_blocks:
        if (fd.pdt_uptime_set is None or fd.pdt_uptime_set.height == 0
                or online_set is None or online_set.height == 0
                or fd.p_online_dt is None or fd.p_online_dt.height == 0):
            continue
        up_idx = fd.pdt_uptime_set.join(online_set, on="p", how="inner")
        # v_online materialises at (p, d, t) ∈ p_online_dt ⋈ online_set.
        # up_idx is already (p, d, t)-keyed; intersect with p_online_dt
        # for the live-variable index.
        live = up_idx.join(
            fd.p_online_dt.select("p", "d", "t").unique(),
            on=["p", "d", "t"], how="inner")
        expected += live.height

    actual = pb.cstr_row_count("minimum_uptime")
    assert actual == expected, (
        f"minimum_uptime row count mismatch: actual={actual} "
        f"expected={expected} "
        f"(|pdt_uptime_set|="
        f"{fd.pdt_uptime_set.height if fd.pdt_uptime_set is not None else 0}, "
        f"|p_online_dt|="
        f"{fd.p_online_dt.height if fd.p_online_dt is not None else 0})"
    )
