"""End-to-end bit-for-bit autoscale tests on the H2_trade fixture.

Phase 1h gate.  These tests pin the autoscaler's correctness on the
exact regression scenario that motivated the whole stack
(``y2050_h2_supply_curves`` against
``projects/test-engine/input_sources/H2_trade.sqlite``).

What we assert:

* **Bit-for-bit objective** — scaling=full and scaling=solver_only must
  produce the *identical* float for ``step.obj``.  Layer 2 / Layer 3
  exponents are powers of two; Layer 3 uses HiGHS' own
  ``user_*_scale`` (internally unscaled on output), so any drift from
  the unscaled solve would be a correctness regression rather than a
  floating-point reorder artefact.
* **Primal solution** — every column value matches within ``1e-9``
  absolute.  Same rationale as above — bit-for-bit is the target;
  the tiny tolerance allows for the one rounding HiGHS does when
  it unscales its internal solution back out for the caller.
* **Duals** — both ``row_dual`` and ``col_dual`` arrays are pulled
  from :class:`polar_high.Solution` and compared with the same
  ``1e-9`` tolerance.  These are populated only when
  ``keep_solutions=True`` keeps the live :class:`Solution` on the
  :class:`OrchestrationStep`.
* **Parallel-mode regression check** — scaling=full with HiGHS
  ``threads=2`` must reach Optimal.  Pre-fix HEAD failed here with
  ``non-optimal for lt_rp`` because polar-high's stream-time
  ``user_bound_scale`` heuristic over-clamped under the parallel
  scheduler; with Layer 3 driving the bound-scale exponent from
  post-Layer-2 ranges that pathology no longer fires.

The H2_trade fixture is tiny (197 cols / 233 rows / ~435 nnz) and
the entire test set runs in ~10 s on a developer laptop.  No need
for ``@pytest.mark.slow``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest


_H2_TRADE_DB_SRC = Path(
    "/home/jkiviluo/sources/flextool-engine/projects/test-engine/input_sources/H2_trade.sqlite"
)
_SCENARIO = "y2050_h2_supply_curves"


@pytest.fixture(scope="session")
def h2_trade_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped copy of the H2_trade DB.

    We copy once per test session into a tempdir under ``/tmp`` so the
    reference DB in ``projects/test-engine/input_sources/`` is never
    mutated by the run (Spine's mappers open the file in read-write
    mode).  All scenarios within the session share the copy.
    """
    if not _H2_TRADE_DB_SRC.exists():
        pytest.skip(f"H2_trade DB not present at {_H2_TRADE_DB_SRC}")
    dst_dir = tmp_path_factory.mktemp("h2_e2e_db")
    dst = dst_dir / "H2_trade.sqlite"
    shutil.copy(_H2_TRADE_DB_SRC, dst)
    return dst


def _run_chain(
    db_path: Path,
    work_folder: Path,
    *,
    scaling: str,
    highs_threads: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Invoke ``run_chain_from_db`` under a controlled env.

    The orchestration entry point reads the autoscale mode from
    ``FLEXTOOL_SCALING`` and the thread count from
    ``FLEXTOOL_HIGHS_THREADS`` (see
    ``flextool/engine_polars/_orchestration.py`` — the CLI mirrors
    its parsed args into these env vars before calling here, so the
    test exercises the same wiring).  ``keep_solutions=True`` is
    required to retain :class:`Solution` on every step so we can read
    ``col_value`` / ``row_dual`` / ``col_dual`` directly.
    """
    monkeypatch.setenv("FLEXTOOL_SCALING", scaling)
    monkeypatch.setenv("FLEXTOOL_HIGHS_THREADS", str(highs_threads))
    # Make sure no stray user-bound override leaks in from a parent
    # process — Layer 3 must derive its own recommendation.
    monkeypatch.delenv("FLEXTOOL_USER_BOUND_SCALE", raising=False)

    from flextool.engine_polars._orchestration import run_chain_from_db

    work_folder.mkdir(parents=True, exist_ok=True)
    return run_chain_from_db(
        input_db_url=f"sqlite:///{db_path}",
        scenario_name=_SCENARIO,
        work_folder=work_folder,
        keep_solutions=True,
    )


def test_h2_trade_autoscale_full_matches_solver_only_bit_for_bit(
    h2_trade_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bit-for-bit objective + primal + dual match between full and solver_only.

    With serial HiGHS (``--highs-threads 1``) both modes solve to
    optimality from the same LP arrays (full mode mutates them
    deterministically via power-of-two factors that Layer 2 undoes
    post-solve, while solver_only mode skips Layer 1/2/3 but keeps
    HiGHS' internal equilibration on).  Therefore the objective, the
    column values, and the row / column duals must agree to within
    numerical roundoff — we use ``1e-9`` absolute so the test fails
    loudly if any future change inadvertently introduces non-power-of-two
    scaling.

    We do NOT compare against ``--scaling=off`` here — that mode forces
    ``simplex_scale_strategy=0`` and HiGHS solves a numerically different
    LP (no internal matrix equilibration), so an exact match is not
    guaranteed.
    """
    on_work = tmp_path / "full"
    off_work = tmp_path / "solver_only"

    steps_on = _run_chain(
        h2_trade_db, on_work,
        scaling="full", highs_threads=1, monkeypatch=monkeypatch,
    )
    steps_off = _run_chain(
        h2_trade_db, off_work,
        scaling="solver_only", highs_threads=1, monkeypatch=monkeypatch,
    )

    assert set(steps_on) == set(steps_off), (
        "scaling full / solver_only produced different sub-solve sets: "
        f"on={list(steps_on)} off={list(steps_off)}"
    )

    for solve_name in steps_on:
        s_on = steps_on[solve_name]
        s_off = steps_off[solve_name]
        # Status: both must be Optimal.
        assert s_on.optimal is True, (
            f"scaling=full sub-solve {solve_name!r} not optimal "
            f"(obj={s_on.obj!r}).  The autoscaler must not introduce "
            "an infeasibility on a well-conditioned baseline."
        )
        assert s_off.optimal is True, (
            f"scaling=solver_only sub-solve {solve_name!r} not optimal "
            f"(obj={s_off.obj!r}).  Pre-Layer-3, serial-mode HiGHS "
            "should still solve the H2_trade LP — if this fails, the "
            "baseline itself has regressed."
        )
        # Objective: bit-for-bit identical.  Power-of-two scaling
        # round-trips through HiGHS' user_*_scale options exactly.
        assert s_on.obj == s_off.obj, (
            f"sub-solve {solve_name!r}: autoscale-ON obj {s_on.obj!r} "
            f"≠ autoscale-OFF obj {s_off.obj!r}.  Layer 2 + Layer 3 use "
            "powers of two so the unscaled objective must round-trip "
            "byte-for-byte; non-zero delta means a non-power-of-two "
            "factor leaked in or the unscale step lost precision."
        )
        # Primal: per-column values must match.
        sol_on = s_on.solution
        sol_off = s_off.solution
        assert sol_on is not None and sol_off is not None, (
            f"sub-solve {solve_name!r}: keep_solutions=True did not "
            "retain Solution on one of the step objects."
        )
        assert sol_on.col_names == sol_off.col_names, (
            f"sub-solve {solve_name!r}: col_names differ between ON / "
            "OFF — the LP build itself diverged (this is a build-side "
            "regression, not a scaling issue)."
        )
        np.testing.assert_allclose(
            sol_on.col_value, sol_off.col_value, atol=1e-9, rtol=0.0,
            err_msg=(
                f"sub-solve {solve_name!r}: primal column values "
                "differ by more than 1e-9 between scaling full / solver_only."
            ),
        )
        # Duals: row_dual and col_dual.
        np.testing.assert_allclose(
            sol_on.row_dual, sol_off.row_dual, atol=1e-9, rtol=0.0,
            err_msg=(
                f"sub-solve {solve_name!r}: row duals differ by more "
                "than 1e-9 between scaling full / solver_only."
            ),
        )
        np.testing.assert_allclose(
            sol_on.col_dual, sol_off.col_dual, atol=1e-9, rtol=0.0,
            err_msg=(
                f"sub-solve {solve_name!r}: reduced-cost duals differ "
                "by more than 1e-9 between scaling full / solver_only."
            ),
        )


def test_h2_trade_autoscale_on_parallel_mode_reaches_optimal(
    h2_trade_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the original H2_trade regression case.

    Pre-fix HEAD (no autoscale stack) hit ``non-optimal for lt_rp``
    when run with ``--highs-threads 2``.  The cause was polar-high's
    stream-time ``user_bound_scale`` heuristic over-clamping the LP
    under the parallel scheduler.

    Post-fix (this branch), Layer 3 picks ``user_bound_scale`` from
    the *post-Layer-2* coefficient ranges — Layer 2 has already pulled
    the ``ladder_tier_cap_annual_roll`` RHS values from ~1.4e+9 down
    to ~1.9e+5 by the time Layer 3 makes its recommendation, so the
    bound-scale exponent is sane and parallel-mode HiGHS solves the LP.
    """
    work = tmp_path / "parallel"
    steps = _run_chain(
        h2_trade_db, work,
        scaling="full", highs_threads=2, monkeypatch=monkeypatch,
    )
    assert steps, "no orchestration steps produced — cascade aborted"
    for solve_name, step in steps.items():
        assert step.optimal is True, (
            f"sub-solve {solve_name!r}: not optimal under scaling=full "
            f"+ threads=2 (obj={step.obj!r}).  This is the original H2_trade "
            "regression — pre-fix HEAD failed identically here.  The "
            "autoscaler exists so that this case solves; a failure means "
            "Layer 2 / Layer 3 are not engaging or are misconfigured."
        )
        # Sanity: objective is finite and positive (LP is a min-cost
        # supply-chain so any reasonable solution has obj > 0).
        assert step.obj is not None and step.obj > 0, (
            f"sub-solve {solve_name!r}: optimal but obj={step.obj!r}."
        )
