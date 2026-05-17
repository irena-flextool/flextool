"""Multi-roll cascade runs in-memory by default.

After Step 2 of the FlexDataProvider migration the cascade no longer
toggles a global CSV-emission flag — writers always populate the
:class:`FlexDataProvider` and never write CSVs themselves.  Disk
emission is only via ``--csv-dump`` calling
:meth:`FlexDataProvider.snapshot_processed_inputs` from
``cmd_run_flextool``.

This module exercises the regression Phase E-f originally guarded
against: the loader-side ``if not path.exists()`` short-circuits.
Those guards used to drop seeded frames; today they read through the
Provider via ``provider.has(name)`` and never look at the disk at all
in the in-memory path.

We exercise the parity gate:

* Run ``work_fullYear_roll`` (a multi-roll fixture).
* Assert: the cascade completes without exception, with > 1 sub-solve.
* Assert: the final objective is finite.
* Assert: no CSVs in ``solve_data/cross_solve/`` (cross-solve carriers
  ride in ``state.cross_solve_carriers``).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from flextool.engine_polars import run_chain_from_db


pytestmark = pytest.mark.solver


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


_FIX = ("work_fullYear_roll", "fullYear_roll")


def _db_or_skip(work_name: str) -> Path:
    db = DATA / work_name / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    return db


def _final_objective(sols) -> float:
    last = next(reversed(sols.values()))
    if last.obj is not None:
        return float(last.obj)
    return float(last.solution.obj)


def test_multi_roll_cascade_completes(tmp_path: Path) -> None:
    """``work_fullYear_roll`` must complete in-memory with >= 2 sub-solves."""
    db = _db_or_skip(_FIX[0])
    work = tmp_path / "multi_roll"
    work.mkdir(parents=True, exist_ok=True)
    sols = run_chain_from_db(
        db, scenario_name=_FIX[1], work_folder=work,
    )
    assert len(sols) >= 2, (
        f"expected multi-roll fixture to yield >= 2 sub-solves; got "
        f"{len(sols)}: {list(sols)}"
    )


def test_multi_roll_final_objective_finite(tmp_path: Path) -> None:
    """Multi-roll cascade produces a finite final objective."""
    db = _db_or_skip(_FIX[0])

    work = tmp_path / "multi_roll"
    work.mkdir(parents=True, exist_ok=True)
    sols = run_chain_from_db(
        db, scenario_name=_FIX[1], work_folder=work,
    )
    assert len(sols) >= 2, (
        f"expected multi-roll fixture to yield >= 2 sub-solves; got "
        f"{len(sols)}: {list(sols)}"
    )
    obj = _final_objective(sols)
    assert math.isfinite(obj), (
        f"final objective not finite: {obj}"
    )


def test_multi_roll_no_cross_solve_csvs(tmp_path: Path) -> None:
    """In-memory multi-roll run leaves no CSVs in ``cross_solve/``
    (cross-solve carriers ride in ``state.cross_solve_carriers``)."""
    db = _db_or_skip(_FIX[0])
    work = tmp_path / "multi_roll"
    work.mkdir(parents=True, exist_ok=True)
    run_chain_from_db(
        db, scenario_name=_FIX[1], work_folder=work,
    )
    cs = work / "cross_solve"
    if not cs.is_dir():
        return
    csvs = sorted(p.name for p in cs.glob("*.csv"))
    assert not csvs, (
        f"cross_solve/ contains CSVs after in-memory cascade: {csvs}"
    )
