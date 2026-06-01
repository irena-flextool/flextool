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

import numpy as np
import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.lean_parquet import read_lean_parquet


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


def _step_obj(step) -> float:
    """Objective for a step, robust to keep_solutions slimming (the scalar
    ``obj`` survives the slim even when ``solution`` is dropped)."""
    o = step.obj if step.obj is not None else getattr(
        getattr(step, "solution", None), "obj", None
    )
    assert o is not None, "step has no objective"
    return float(o)


def _compare_output_raw(wf_ref: Path, wf_sub: Path) -> list[str]:
    """Cell-by-cell compare every ``output_raw/*.parquet`` between two
    work folders.  Returns a list of human-readable mismatch strings
    (empty == full output parity)."""
    raw_ref = wf_ref / "output_raw"
    raw_sub = wf_sub / "output_raw"
    ref_files = {p.name for p in raw_ref.glob("*.parquet")}
    sub_files = {p.name for p in raw_sub.glob("*.parquet")}
    problems: list[str] = []
    if ref_files != sub_files:
        only_ref = sorted(ref_files - sub_files)
        only_sub = sorted(sub_files - ref_files)
        problems.append(
            f"output_raw file set differs: only_ref={only_ref} "
            f"only_sub={only_sub}"
        )
    # We assert on at least one variable output beyond v_obj so that the
    # comparison covers decision-variable values, not just the scalar.
    assert any(
        n.startswith("v_flow") or n.startswith("v_trade")
        or n.startswith("v_state")
        for n in ref_files
    ), (
        "fullYear_roll produced no decision-variable output_raw parquets — "
        "output-parity assertion would be vacuous"
    )
    for name in sorted(ref_files & sub_files):
        df_ref = read_lean_parquet(raw_ref / name)
        df_sub = read_lean_parquet(raw_sub / name)
        if list(df_ref.columns) != list(df_sub.columns):
            problems.append(f"{name}: column layout differs")
            continue
        if df_ref.shape != df_sub.shape:
            problems.append(
                f"{name}: shape {df_ref.shape} vs {df_sub.shape}"
            )
            continue
        a = df_ref.to_numpy()
        b = df_sub.to_numpy()
        # Numeric cells: exact-ish compare (the slim is memory-only, so
        # the LP and its solution are bit-identical between the two
        # modes; allow only floating round-trip noise).
        try:
            af = a.astype(float)
            bf = b.astype(float)
        except (ValueError, TypeError):
            # Mixed / object cells (index labels embedded) — fall back to
            # element-wise equality.
            if not (a == b).all():
                problems.append(f"{name}: object-cell mismatch")
            continue
        if not np.allclose(af, bf, rtol=1e-9, atol=1e-9, equal_nan=True):
            diff = np.nanmax(np.abs(af - bf))
            problems.append(f"{name}: max abs cell diff {diff:.3e}")
    return problems


def test_inloop_solution_null_preserves_results(scenario_workdir) -> None:
    """In-loop whole-``solution`` nulling (the per-roll floor-ratchet
    release) must not change any cascade result.

    ``fullYear_roll`` is a 72-roll SAME-LEVEL warm cascade, so on the
    ``keep_solutions=False`` run every prior roll's parked
    ``OrchestrationStep.solution`` is nulled IN-LOOP by the warm slim
    (``_same_level_older`` predicate) the moment the next roll parks —
    long before the post-cascade final slim runs.  This is the path the
    end-state-only tests do NOT cover.  ``keep_solutions=True`` retains
    every solution (slim gated off) and is the ground truth.

    Asserts both halves of the parity contract:
      (i)  per-roll objective parity, and
      (ii) full decision-variable output parity (every
           ``output_raw/*.parquet`` cell matches between the two modes).
    """
    db = scenario_workdir(_SCEN) / "tests.sqlite"
    with tempfile.TemporaryDirectory() as t:
        wf_keep = Path(t) / "keep"
        wf_slim = Path(t) / "slim"

        # Reference: keep_solutions=True — slim gated off, every parked
        # step retains its full Solution.
        ref = run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=wf_keep,
            warm=True, keep_solutions=True,
        )
        # Subject: keep_solutions=False — in-loop whole-solution nulling
        # active (the fix under test).
        sub = run_chain_from_db(
            db, scenario_name=_SCEN, work_folder=wf_slim,
            warm=True, keep_solutions=False,
        )

        # Compare the on-disk outputs WHILE the temp dirs still exist.
        out_problems = _compare_output_raw(wf_keep, wf_slim)

    # Sanity: this scenario really is a multi-roll cascade (>=3 rolls so
    # multiple prior steps are slimmed in-loop), and the modes produced
    # the same solve-name set.
    assert list(ref.keys()) == list(sub.keys())
    assert len(ref) >= 3, (
        f"fullYear_roll yielded only {len(ref)} rolls; need >=3 same-level "
        "rolls to exercise in-loop slimming of multiple prior steps"
    )

    # (i) per-roll objective parity.
    obj_mismatches: list[str] = []
    for name in ref:
        o_ref = _step_obj(ref[name])
        o_sub = _step_obj(sub[name])
        if o_ref == 0.0:
            if abs(o_sub) > 1e-6:
                obj_mismatches.append(f"{name}: {o_ref} vs {o_sub}")
            continue
        if abs(o_sub - o_ref) / abs(o_ref) > 1e-9:
            obj_mismatches.append(
                f"{name}: {o_ref:.9e} vs {o_sub:.9e}"
            )
    assert not obj_mismatches, (
        "in-loop solution nulling changed per-roll objectives "
        "(release must be memory-only):\n  " + "\n  ".join(obj_mismatches)
    )

    # (ii) full decision-variable output parity (computed in-block above,
    # while the temp work folders still existed).
    assert not out_problems, (
        "in-loop solution nulling changed decision-variable output "
        "(release must be memory-only):\n  " + "\n  ".join(out_problems)
    )
