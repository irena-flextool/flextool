"""Tier 7 emission test — ``maxToSink`` row count.

Δ.22 ported from MPS-parsing to direct polar_high ``Problem`` inspection.

Invariant
---------
``maxToSink`` is registered with ``over = pss_dt`` (the cartesian product
of ``process_source_sink`` × ``dt``); see
``flextool/engine_polars/model.py:1359``.  The materialised LP-row count
must equal ``|pss_dt|`` — one row per (p, source, sink, d, t) tuple.
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
def test_maxToSink_emits_one_row_per_pss_dt(test_db_url: str) -> None:
    scenario = "coal"
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
    expected = fd.pss_dt.height

    actual = pb.cstr_row_count("maxToSink")
    # ``cstr_row_count`` is prefix-matched (covers maxToSink_negCap,
    # maxToSink_online_*).  We only assert the bare-family count via
    # ``cstrs_named`` exact-name lookup.
    bare = next((r for r in pb.cstrs_named("maxToSink") if r.name == "maxToSink"),
                 None)
    bare_count = len(bare.over) if bare is not None else 0
    assert bare_count == expected, (
        f"maxToSink row count mismatch: actual={bare_count} "
        f"expected={expected} (|pss_dt|={fd.pss_dt.height}); "
        f"prefix-total (includes _negCap / _online_*): {actual}"
    )
