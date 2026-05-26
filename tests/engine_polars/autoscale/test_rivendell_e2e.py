"""Rivendell guard — autoscaler must NOT regress the col-only escape.

The Rivendell ``S17_horizon2035_hourly_rp`` scenario is the original
trigger for the col-only ``user_bound_scale`` heuristic fix (see
``tests/engine_polars/test_rivendell_bug1_5_6_user_bound_scale.py``).
Its LP has tightly-clustered column bounds (all 1.0) and a wide RHS
spread (large cumulative-resource rows), so the pre-fix pooled
heuristic over-clamped to ``user_bound_scale = -10`` and presolve
declared the resulting LP infeasible.

Phase 1h's contract: the new autoscale stack — running on top of
Layer 2's semantic per-type rescale — must still produce a *near-zero*
``user_bound_scale`` for this fixture, and the objective must match
the autoscale-OFF baseline bit-for-bit.  A bound-scale of ``-1`` is
acceptable (Layer 3's ``clamp-large`` branch picks a single-decade
pull when the worst-bound magnitude is just barely above HiGHS' 1e+6
ceiling — that's the well-conditioned regime, not the pathological
``-10`` clamp the pre-fix heuristic produced).  Anything below ``-3``
indicates a regression.

We keep this file separate from the original
``test_rivendell_bug1_5_6_user_bound_scale.py`` because the assertion
shape is different — the original test guards against
*presolve-infeasible*, while this one guards against
*objective drift* under the new autoscale stack — and parametrising
the original across autoscale states would make the failure modes
harder to diagnose.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest


_RIVENDELL_DB_SRC = Path(
    "/home/jkiviluo/sources/flextool-engine/projects/Rivendell/input_sources/rivendell.sqlite"
)


@pytest.fixture(scope="session")
def rivendell_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped read-write copy under /tmp.

    Spine's mappers open the DB read-write; never run against the
    reference DB in ``projects/Rivendell/input_sources/`` directly.
    """
    if not _RIVENDELL_DB_SRC.exists():
        pytest.skip(f"Rivendell DB not present at {_RIVENDELL_DB_SRC}")
    dst_dir = tmp_path_factory.mktemp("rivendell_e2e_db")
    dst = dst_dir / "rivendell.sqlite"
    shutil.copy(_RIVENDELL_DB_SRC, dst)
    return dst


def _run_chain(
    db_path: Path,
    work_folder: Path,
    *,
    scaling: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Invoke the cascade under the requested ``--scaling`` mode."""
    monkeypatch.setenv("FLEXTOOL_SCALING", scaling)
    monkeypatch.setenv("FLEXTOOL_HIGHS_THREADS", "1")
    monkeypatch.delenv("FLEXTOOL_USER_BOUND_SCALE", raising=False)

    from flextool.engine_polars._orchestration import run_chain_from_db

    work_folder.mkdir(parents=True, exist_ok=True)
    return run_chain_from_db(
        input_db_url=f"sqlite:///{db_path}",
        scenario_name="S17_horizon2035_hourly_rp",
        work_folder=work_folder,
        keep_solutions=True,
    )


def _parse_user_bound_scale(yaml_path: Path) -> int:
    """Pull the ``user_bound_scale`` value from an autoscale YAML.

    The autoscale report is the small in-tree YAML dumper from
    ``flextool.engine_polars.autoscale._report`` — flat ``key: value``
    pairs, no anchors.  We can do a string scan because the file
    structure is fixed and a missing field is an actionable failure.
    """
    text = yaml_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("user_bound_scale:"):
            return int(stripped.split(":", 1)[1].strip())
    raise AssertionError(
        f"{yaml_path}: no 'user_bound_scale:' field found.  Layer 3 "
        "should have written one regardless of trigger state."
    )


def test_rivendell_s17_autoscale_no_aggressive_clamp(
    rivendell_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer 3 must not return to the ``-10`` clamp on Rivendell S17.

    Pre-fix the pooled row+col bound heuristic returned ``N=-10`` for
    every Rivendell hourly_rp scenario and HiGHS presolve declared the
    user-scaled model infeasible.  Post-fix (Layer 3 derives ``N`` from
    the post-Layer-2 ranges and applies the col-only spread guard
    inside its escape branch), ``N`` lands at 0 (no scaling needed) or
    ``-1`` (single-decade pull because the large RHS just clears HiGHS'
    1e+6 ceiling).

    We assert ``user_bound_scale >= -3`` to cover both observed values
    while still catching any regression that re-introduces a 6+ decade
    pull from a pooled row+col spread.
    """
    work = tmp_path / "on"
    steps = _run_chain(
        rivendell_db, work, scaling="full", monkeypatch=monkeypatch,
    )
    assert steps, "no orchestration steps produced — cascade aborted"

    for solve_name, step in steps.items():
        assert step.optimal is True, (
            f"sub-solve {solve_name!r}: not optimal under autoscale ON. "
            "Pre-fix HEAD presolve-infeasible'd here with N=-10 from the "
            "pooled bound heuristic — a non-optimal result now likely "
            "means Layer 3 has regressed back to the row-RHS-driven path."
        )

    # YAML report — exactly one per sub-solve.
    yamls = sorted((work / "solve_data").glob("autoscale_*.yaml"))
    assert yamls, (
        f"no autoscale_*.yaml emitted under {work / 'solve_data'} — "
        "Layer 3's apply hook is supposed to write one per sub-solve."
    )
    for yp in yamls:
        n = _parse_user_bound_scale(yp)
        assert n >= -3, (
            f"{yp.name}: Layer 3 picked user_bound_scale={n}.  Rivendell "
            "S17 has all-1.0 column bounds and the col-only spread is "
            "0 decades — Layer 3 should land at 0 (no clamp) or -1 "
            "(clamp-large for the ~9e+5 RHS).  A value <= -4 is the "
            "pre-fix pooled-heuristic failure mode (typically -10) and "
            "means presolve will declare the model infeasible."
        )


def test_rivendell_s17_objective_bit_for_bit(
    rivendell_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Objective must match between ``--scaling=full`` and ``--scaling=solver_only``.

    Both modes leave HiGHS' internal matrix equilibration on (the only
    difference is whether Layer 1/2/3 fire), so HiGHS unscales the
    internal solution back out before reporting ``obj`` and the two
    paths must agree byte-for-byte.  This catches the failure mode
    where Layer 3 silently introduces a non-power-of-two factor (e.g.
    if a future refactor swaps the exponent rounding for a ``log10``
    that happens to land on an integer for some scenarios).

    We do NOT compare against ``--scaling=off`` here — that mode forces
    ``simplex_scale_strategy=0`` and HiGHS will solve a numerically
    different LP (the no-matrix-equilibration path), so an exact match
    is not guaranteed.
    """
    on_work = tmp_path / "full"
    off_work = tmp_path / "solver_only"
    steps_on = _run_chain(
        rivendell_db, on_work, scaling="full", monkeypatch=monkeypatch,
    )
    steps_off = _run_chain(
        rivendell_db, off_work, scaling="solver_only", monkeypatch=monkeypatch,
    )

    assert set(steps_on) == set(steps_off), (
        f"--scaling=full / solver_only produced different sub-solve sets: "
        f"full={list(steps_on)} solver_only={list(steps_off)}"
    )

    for solve_name in steps_on:
        s_on = steps_on[solve_name]
        s_off = steps_off[solve_name]
        assert s_on.optimal is True
        assert s_off.optimal is True
        assert s_on.obj == s_off.obj, (
            f"sub-solve {solve_name!r}: autoscale-ON obj {s_on.obj!r} "
            f"≠ autoscale-OFF obj {s_off.obj!r}.  Power-of-two factors "
            "must round-trip exactly; non-zero delta is a correctness "
            "regression."
        )

        # Primal parity — same atol convention as the H2_trade e2e test.
        sol_on = s_on.solution
        sol_off = s_off.solution
        assert sol_on is not None and sol_off is not None
        assert sol_on.col_names == sol_off.col_names, (
            f"sub-solve {solve_name!r}: col_names differ between ON / "
            "OFF — the LP build itself diverged."
        )
        np.testing.assert_allclose(
            sol_on.col_value, sol_off.col_value, atol=1e-9, rtol=0.0,
            err_msg=(
                f"sub-solve {solve_name!r}: primal column values differ "
                "by more than 1e-9 between autoscale ON / OFF."
            ),
        )
