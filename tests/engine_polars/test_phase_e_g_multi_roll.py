"""Multi-roll cascade parity with v3.32.0 golden — in-memory path.

After Step 2 of the FlexDataProvider migration the cascade has exactly
one data pathway: Source → Provider → LP/writers/post-processing.  The
72-roll ``fullYear_roll`` cascade must reproduce the v3.32.0 golden
final-roll objective end-to-end without writing CSVs from the writers.

These tests assert:

1. The 72-roll cascade reproduces the v3.32.0-golden final-roll
   objective (Phase E-g baseline).
2. Every per-roll objective matches between the default (in-memory)
   path and the ``--csv-dump`` path within ``rel_tol=1e-6`` — both
   feed identical data into identical LPs; the dump is a one-way
   debug snapshot, not a re-entry point.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from flextool.engine_polars import run_chain_from_db


_SCEN = "fullYear_roll"


# v3.32.0 golden — see ``output_raw/v_obj__dispatch_fullYear_roll_roll_71.parquet``
# in the work_fullYear_roll fixture.
_V3320_ROLL_71_OBJ = 3.432825e8


def test_fullYear_roll_matches_v3320_golden(scenario_workdir) -> None:
    """The 72-roll in-memory cascade reproduces the v3.32.0 golden
    final-roll objective.
    """
    db = scenario_workdir(_SCEN) / "tests.sqlite"
    with tempfile.TemporaryDirectory() as t:
        wf = Path(t) / "default"
        sols = run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=wf,
            keep_solutions=True,
        )
    last = next(reversed(sols.values()))
    obj = float(last.obj if last.obj is not None else last.solution.obj)
    assert obj == pytest.approx(_V3320_ROLL_71_OBJ, rel=1e-6), (
        f"roll_71 objective {obj:.6e} does not match "
        f"v3.32.0 golden {_V3320_ROLL_71_OBJ:.6e}"
    )


def test_fullYear_roll_csv_dump_per_roll_parity(scenario_workdir) -> None:
    """Every per-roll objective matches between default (in-memory)
    and ``--csv-dump`` paths within ``rel_tol=1e-6``.

    ``--csv-dump`` is a one-way Provider snapshot to disk — it must not
    influence the cascade's numeric output.
    """
    db = scenario_workdir(_SCEN) / "tests.sqlite"
    with tempfile.TemporaryDirectory() as t:
        default_wf = Path(t) / "default"
        dump_wf = Path(t) / "dump"

        sols_default = run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=default_wf,
            keep_solutions=True,
        )
        sols_dump = run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=dump_wf,
            keep_solutions=True, csv_dump=True,
        )

    assert list(sols_default.keys()) == list(sols_dump.keys()), (
        "default / csv-dump cascades produced different solve-name sets"
    )

    mismatches: list[tuple[str, float, float, float]] = []
    for name in sols_default:
        on_step = sols_default[name]
        off_step = sols_dump[name]
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
        "Per-roll objective parity failed between default and csv_dump:\n"
        + "\n".join(
            f"  {n}: default={o:.6e} dump={f:.6e} rel={r:.2e}"
            for n, o, f, r in mismatches
        )
    )
