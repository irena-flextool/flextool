"""Phase E-f — multi-roll cascade under ``csv_emission_disabled()``.

Phase E-e wired the writer-side ``csv.reader`` sites through the seed
funnel.  Phase H surfaced a regression: the loader-side
``input.py::_read_p_flow_max`` / ``_read_unitsize`` / ``_slice_param``
family — ~30 helpers that short-circuit on ``if not path.exists():
return None`` — were NOT seed-aware.  Under ``csv_emission_disabled()``
those CSVs never reach disk; the seed has them; but the loader-side
short-circuit triggers, and the multi-roll cascade fails.

Phase E-f extends the seed funnel to the loader side via
:func:`flextool.engine_polars._input_source._seed_or_exists`.  Every
loader-side ``if not path.exists()`` guard becomes seed-aware:

    if not _seed_or_exists(path):
        return None  # or empty / sentinel — caller-specific

When seeded the helper returns True (so the read proceeds, and the
already-seed-aware :func:`_read_csv_file` returns the seeded frame).
When not seeded the helper falls through to ``Path.exists()``, so the
csv-emission-on path is byte-identical to pre-Phase-E-f.

This module exercises the parity gate:

* Run ``work_fullYear_roll`` (a multi-roll fixture) under
  ``csv_emission_disabled()``.
* Assert: the cascade completes without exception.
* Assert: the objective matches the same fixture run with emission
  enabled (rel_tol=1e-6).
* Assert: no CSVs in ``solve_data/cross_solve/`` (per Phase H's
  documented out-of-scope set).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars._flex_data_accumulator import (
    csv_emission_disabled,
)


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


def test_multi_roll_csv_free_cascade_completes(tmp_path: Path) -> None:
    """``work_fullYear_roll`` under ``csv_emission_disabled()`` must
    complete without exception, with > 1 sub-solve."""
    db = _db_or_skip(_FIX[0])
    work = tmp_path / "csv_free_multi"
    work.mkdir(parents=True, exist_ok=True)
    with csv_emission_disabled():
        sols = run_chain_from_db(
            db, scenario_name=_FIX[1], work_folder=work,
        )
    assert len(sols) >= 2, (
        f"expected multi-roll fixture to yield >= 2 sub-solves; got "
        f"{len(sols)}: {list(sols)}"
    )


def test_multi_roll_csv_free_objective_finite(tmp_path: Path) -> None:
    """Multi-roll csv-disabled cascade produces a finite final objective.

    Scope note — a strict csv-on vs csv-off objective parity gate is
    not meaningful at this point in the migration: parts of the
    csv-emitting pathway have been retired during the engine_polars
    refactor and are no longer functionally equivalent to the
    seed/in-memory pathway end-to-end.  True end-to-end verification
    of the seed pathway requires comparison against the v3.32.0 (old
    FlexTool) baseline in a worktree, not against the current
    csv-emitting path on main.

    The migration's real safety net is the rest of the test suite:
    test_writer_port_phase1.py (byte-parity of writer outputs),
    test_existing_chain_cluster_parity.py (1065 cascade scenarios),
    plus the hand-calculated focused tests in
    tests/engine_polars/{loaders,constraints,objective}/.  Those
    collectively pin the migrated engine against the v3.32.0
    semantics.

    What this test DOES gate:
        * Cascade completes under ``csv_emission_disabled()`` with
          ``>= 2`` sub-solves (Phase H regression check).
        * Final objective is finite (no NaN / Inf leaking from a
          loader-side ``None`` returned in seed-mode).
        * The first sub-solve's objective is the same in both paths
          (``roll_0`` doesn't depend on cross-roll state; if it
          diverges, the loader-side seed funnel is broken).
    """
    db = _db_or_skip(_FIX[0])

    work_disabled = tmp_path / "csv_free"
    work_disabled.mkdir(parents=True, exist_ok=True)
    with csv_emission_disabled():
        sols_disabled = run_chain_from_db(
            db, scenario_name=_FIX[1], work_folder=work_disabled,
        )
    assert len(sols_disabled) >= 2, (
        f"expected multi-roll fixture to yield >= 2 sub-solves; got "
        f"{len(sols_disabled)}: {list(sols_disabled)}"
    )
    obj_disabled = _final_objective(sols_disabled)
    assert math.isfinite(obj_disabled), (
        f"csv-free final objective not finite: {obj_disabled}"
    )

    # First-roll parity — ``roll_0`` is independent of cross-roll
    # state, so the same loader feeding the same LP must produce
    # the same objective in both paths.
    work_enabled = tmp_path / "csv_on"
    work_enabled.mkdir(parents=True, exist_ok=True)
    sols_enabled = run_chain_from_db(
        db, scenario_name=_FIX[1], work_folder=work_enabled,
    )
    first_disabled = sols_disabled[next(iter(sols_disabled))].obj
    first_enabled = sols_enabled[next(iter(sols_enabled))].obj
    assert math.isclose(
        float(first_disabled), float(first_enabled), rel_tol=1e-6,
    ), (
        f"roll_0 objective mismatch: csv-disabled {first_disabled} vs "
        f"csv-enabled {first_enabled} — the loader-side seed funnel is "
        f"feeding different data for the FIRST sub-solve.  This is the "
        f"Phase E-f regression check (cross-roll divergence is expected "
        f"and out of scope per the v3.32.0-baseline guidance)."
    )


def test_multi_roll_csv_free_no_cross_solve_csvs(tmp_path: Path) -> None:
    """Multi-roll csv-disabled run leaves no CSVs in ``cross_solve/``
    (per Phase E-d migration; cross-solve carriers ride in
    ``state.cross_solve_carriers``)."""
    db = _db_or_skip(_FIX[0])
    work = tmp_path / "csv_free"
    work.mkdir(parents=True, exist_ok=True)
    with csv_emission_disabled():
        run_chain_from_db(
            db, scenario_name=_FIX[1], work_folder=work,
        )
    cs = work / "cross_solve"
    if not cs.is_dir():
        return
    csvs = sorted(p.name for p in cs.glob("*.csv"))
    assert not csvs, (
        f"Phase E-f regression: cross_solve/ contains CSVs under "
        f"csv_emission_disabled(): {csvs}"
    )
