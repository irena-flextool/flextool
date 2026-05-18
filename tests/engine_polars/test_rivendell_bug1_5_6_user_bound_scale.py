"""Rivendell bug 1+5+6 regression — col-only ``user_bound_scale`` heuristic.

The Rivendell scenarios ``B0_base_hourly_rp`` (bug 1), ``S17_horizon2035_hourly_rp``
(bug 5) and ``S22_techdisc_hourly_rp`` (bug 6) all failed identically on
pre-fix HEAD ``c6ffd27e``::

    Coefficient ranges:
      Matrix  [1e+00, 1e+03]
      Cost    [6e-03, 1e+02]
      Bound   [1e-03, 1e-03]
      RHS     [8e-04, 3e+03]
    Presolving model
    Problem status detected on presolve: Infeasible

with HiGHS auto-applying ``user_bound_scale=-10`` from the
``recommend_user_bound_scale_from_lp`` heuristic and presolve declaring
the user-scaled model infeasible after 0 iterations.

Root cause: the heuristic pooled row + column bound ranges into one
geometric-midpoint computation, so any Rivendell-shaped LP whose RHS
included a cumulative-resource bound up to ~3.3e+6 picked up a 6.6-decade
spread and a recommendation of ``N=-10`` — even though the column bounds
themselves were tightly clustered at ``[1.0, 1.0]`` and required no
scaling at all.

HiGHS' ``user_bound_scale=N`` multiplies BOTH col bounds AND row bounds
(RHS) by ``2**N``, but **does not scale the constraint matrix or the
costs**.  With ``N=-10`` on a Rivendell B0/S17/S22 LP:

* col bound ``v ≤ 1.0`` becomes ``v ≤ ~9.77e-4``;
* matrix coefficients on ``v`` remain in ``[1, 1000]``;
* a balance constraint like ``Σ coef × v_i = rhs / 1024`` cannot be
  satisfied because each ``coef × v_i`` term is now capped at
  ``1000 × 9.77e-4 ≈ 0.977`` per variable, while the constraints expect
  unscaled flow magnitudes — presolve detects structural infeasibility.

The fix restricts the spread calculation to column bounds only (matching
the flextoolrunner-side ``decide_user_bound_scale`` which is fed only
``col_lower``/``col_upper`` via ``compute_bound_stats``).  On all three
Rivendell scenarios the col-bound spread is 0 decades (every column has
upper bound 1.0), so the recommender returns ``0`` and HiGHS' own
default scaling solves the model in ~1 s with an optimal objective on
the order of ``1e+12``.

This test runs the smallest of the three affected scenarios end-to-end
via :func:`run_chain_from_db` and asserts an optimal solve with a
positive objective.  Pre-fix it raises ``FlexToolSolveError`` on the
presolve-infeasible code path; post-fix it reaches optimality.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_RIVENDELL_DB = Path(
    "/home/jkiviluo/sources/flextool-engine/projects/Rivendell/input_sources/rivendell.sqlite"
)


@pytest.mark.parametrize(
    "scenario_name,expected_obj_min,expected_obj_max",
    [
        # B0 / S22 share the LP shape (8485×4726 nnz pre-migration; current
        # native cascade is 92354 rows × 92352 cols × ~595k nnz at the
        # full-year, hourly resolution they declare).  Both should converge
        # in the ~2.5e+12 range (annualised cost).  The bounds are loose
        # because intermediate cascade refactors may shift the objective
        # slightly; what we guard is "non-zero, finite, > 1e+10".
        ("B0_base_hourly_rp", 1e10, 1e13),
        # S17 has a horizon-restricted slice; the test handoff note
        # documents pre-migration v3.32.0 was 9.95e+9 on the slice variant.
        # The hourly_rp scenario is the full year, so the objective is
        # ~4.17e+11; we still only assert > 1e+10.
        ("S17_horizon2035_hourly_rp", 1e10, 1e13),
        ("S22_techdisc_hourly_rp", 1e10, 1e13),
    ],
)
def test_rivendell_bug1_5_6_reaches_optimal_with_positive_objective(
    tmp_path: Path, scenario_name: str, expected_obj_min: float,
    expected_obj_max: float,
) -> None:
    """Pre-fix: HiGHS presolve declared the model infeasible after
    applying ``user_bound_scale=-10`` from the pooled row+col bound
    heuristic.  Post-fix: the heuristic considers col bounds only,
    returns ``N=0`` for these scenarios (whose col bounds are all 1.0),
    and HiGHS solves to optimality with the unscaled model.
    """
    if not _RIVENDELL_DB.exists():
        pytest.skip(f"Rivendell DB not present at {_RIVENDELL_DB}")
    from flextool.engine_polars._orchestration import run_chain_from_db

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    steps = run_chain_from_db(
        input_db_url=f"sqlite:///{_RIVENDELL_DB}",
        scenario_name=scenario_name,
        work_folder=work_dir,
    )
    assert steps, (
        f"scenario {scenario_name!r}: no orchestration steps produced — "
        f"cascade aborted before any solve completed."
    )
    for solve_name, step in steps.items():
        assert step.optimal, (
            f"scenario {scenario_name!r} sub-solve {solve_name!r}: "
            f"not optimal (obj={step.obj}).  Pre-fix HEAD failed here "
            f"with 'flexpy non-optimal' because user_bound_scale=-10 was "
            f"applied from the row+col pooled bound spread, dragging col "
            f"bounds to ~1e-3 while leaving matrix coefficients at ~1e+3 "
            f"and making the LP presolve-infeasible.  Post-fix the "
            f"col-only spread for this scenario is 0 decades → no scaling."
        )
        # Objective must be a positive finite number in the expected band.
        obj = step.obj
        assert obj is not None, (
            f"scenario {scenario_name!r} sub-solve {solve_name!r}: "
            f"step.obj is None despite optimal=True."
        )
        assert expected_obj_min <= float(obj) <= expected_obj_max, (
            f"scenario {scenario_name!r} sub-solve {solve_name!r}: "
            f"objective {obj!r} outside expected band "
            f"[{expected_obj_min:.2e}, {expected_obj_max:.2e}].  A drift "
            f"outside the band is worth investigating — the band is "
            f"loose (1e+10 .. 1e+13) and pre-fix the value was 0."
        )
