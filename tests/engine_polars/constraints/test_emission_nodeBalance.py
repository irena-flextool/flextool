"""Tier 7 emission test — ``nodeBalance_eq`` row count.

Δ.22 ported from MPS-parsing to direct polar_high ``Problem`` inspection.
The cascade builds the LP via :func:`flextool.engine_polars.build_flextool`
through polar_high; row counts are obtained from
``Problem.cstr_row_count(name)``.

Invariant
---------
``nodeBalance_eq`` is registered with ``over = nodeBalance_dt`` anti-joined
with ``nodeStateBlock`` (see ``flextool/engine_polars/model.py:858``).
The .mod's domain ``n in nodeBalance × dt`` minus ``n in nodeStateBlock``
is reproduced by the cascade by construction; the test pins the
materialised LP-row count to that derivation rule so a future writer
regression that drops rows (or duplicates them) is caught.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, run_chain_from_db
from flextool.engine_polars._pdt_join import compute_nodeBalance_dt

_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))


@pytest.mark.emission
def test_nodeBalance_emits_one_row_per_n_dt(test_db_url: str) -> None:
    scenario = "coal"
    with tempfile.TemporaryDirectory() as wd:
        steps = run_chain_from_db(
            test_db_url, scenario,
            work_folder=Path(wd), csv_dump=False, keep_solutions=True,
        )
        last = next(reversed(steps.values()))
        assert last.flex_data is not None, (
            f"FlexData missing on last step for scenario {scenario!r}; "
            f"keep_solutions=True should preserve it")

        pb = Problem()
        build_flextool(pb, last.flex_data)

    fd = last.flex_data
    nb_dt = compute_nodeBalance_dt(fd)
    expected_over = nb_dt
    if fd.nodeStateBlock is not None and fd.nodeStateBlock.height > 0:
        expected_over = expected_over.join(
            fd.nodeStateBlock, on="n", how="anti")
    expected = expected_over.height

    actual = pb.cstr_row_count("nodeBalance_eq")
    assert actual == expected, (
        f"nodeBalance_eq row count mismatch: actual={actual} "
        f"expected={expected} "
        f"(|nodeBalance_dt|={nb_dt.height} "
        f"|nodeStateBlock|="
        f"{fd.nodeStateBlock.height if fd.nodeStateBlock is not None else 0})"
    )
