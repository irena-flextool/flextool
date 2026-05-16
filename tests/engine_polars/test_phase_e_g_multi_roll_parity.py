"""Phase E-g — multi-roll csv-off parity with v3.32.0 golden.

The Phase H handoff doc explicitly carved this out of scope: under
``csv_emission_disabled()`` the multi-roll cascade diverged from the
csv-on path starting at roll_1 because the post-solve
:func:`build_handoff_from_flexpy` reads several ``solve_data/*.csv``
support files (``nodeState.csv``, ``realized_dispatch.csv``,
``entityDivest.csv``, ``period_first.csv``, …) without an active
in-memory seed.  Under ``csv_emission_disabled()`` those files are
suppressed on disk; the per-sub-solve accumulator carries them in
memory but only when the seed is installed for the duration of the
read.

Phase E-g installs the per-sub-solve accumulator as the active seed
around the :func:`build_handoff_from_flexpy` call (mirroring the
existing seed install around :func:`write_outputs_for_solve` from
Phase E-f).  The fix is purely additive — it does not introduce new
disk writes and the csv-on cascade is byte-identical to pre-Phase-E-g.

These tests assert:

1. Under ``csv_emission_disabled()`` the 72-roll ``fullYear_roll``
   cascade reproduces the v3.32.0-golden final-roll objective.
2. Every per-roll objective matches the csv-on path within
   ``rel_tol=1e-6``.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars._flex_data_accumulator import csv_emission_disabled


DATA = (
    Path(__file__).parent / "data"
)
_FIX = "work_fullYear_roll"
_SCEN = "fullYear_roll"


def _db_or_skip() -> Path:
    db = DATA / _FIX / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    return db


# v3.32.0 golden — see ``output_raw/v_obj__dispatch_fullYear_roll_roll_71.parquet``
# in the work_fullYear_roll fixture.
_V3320_ROLL_71_OBJ = 3.432825e8


def test_fullYear_roll_csv_off_matches_v3320_golden() -> None:
    """Under ``csv_emission_disabled()`` the 72-roll cascade reproduces
    the v3.32.0 golden final-roll objective.

    Pre Phase E-g the csv-off path reported 1.54e9 (~4.5× too high) for
    roll_71 because the chain-cumulative storage handoff was broken
    starting at roll_1.
    """
    db = _db_or_skip()
    with tempfile.TemporaryDirectory() as t:
        wf = Path(t) / "csv_off"
        with csv_emission_disabled():
            sols = run_chain_from_db(
                db, scenario_name=_SCEN, work_folder=wf,
                keep_solutions=True,
            )
    last = next(reversed(sols.values()))
    obj_off = float(last.obj if last.obj is not None else last.solution.obj)
    assert obj_off == pytest.approx(_V3320_ROLL_71_OBJ, rel=1e-6), (
        f"csv-off roll_71 objective {obj_off:.6e} does not match "
        f"v3.32.0 golden {_V3320_ROLL_71_OBJ:.6e}"
    )


def test_fullYear_roll_csv_off_per_roll_matches_csv_on() -> None:
    """Every per-roll objective matches between csv-on and csv-off paths
    within ``rel_tol=1e-6``.

    Pre Phase E-g all rolls after roll_0 diverged — roll_1 was 3.2e2
    relative-different (8.76e6 vs 2.12e9), and the divergence grew over
    the cascade.  Post Phase E-g the per-roll objective table is
    byte-equal between paths.
    """
    db = _db_or_skip()
    with tempfile.TemporaryDirectory() as t:
        on_wf = Path(t) / "csv_on"
        off_wf = Path(t) / "csv_off"

        sols_on = run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=on_wf,
            keep_solutions=True,
        )
        with csv_emission_disabled():
            sols_off = run_chain_from_db(
                db, scenario_name=_SCEN, work_folder=off_wf,
                keep_solutions=True,
            )

    assert list(sols_on.keys()) == list(sols_off.keys()), (
        "csv-on / csv-off cascades produced different solve-name sets"
    )

    mismatches: list[tuple[str, float, float, float]] = []
    for name in sols_on:
        on_step = sols_on[name]
        off_step = sols_off[name]
        obj_on = float(
            on_step.obj if on_step.obj is not None else on_step.solution.obj
        )
        obj_off = float(
            off_step.obj if off_step.obj is not None else off_step.solution.obj
        )
        if obj_on == 0.0:
            if abs(obj_off) > 1e-6:
                mismatches.append((name, obj_on, obj_off, float("inf")))
            continue
        rel = abs(obj_off - obj_on) / abs(obj_on)
        if rel > 1e-6:
            mismatches.append((name, obj_on, obj_off, rel))

    assert not mismatches, (
        "Per-roll objective parity failed under csv_emission_disabled():\n"
        + "\n".join(
            f"  {n}: on={o:.6e} off={f:.6e} rel={r:.2e}"
            for n, o, f, r in mismatches
        )
    )
