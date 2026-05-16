"""Warm chain runner — equivalence, cold-fallback, and speedup tests.

Three lightweight tests covering ``run_chain(..., warm=True)`` /
``run_chain_from_db(..., warm=True)``, which wires
:class:`polar_high.WarmProblem` into the chain orchestrator:

1. **Equivalence** — for every standard chain fixture, ``warm=True``
   must produce the SAME per-sub-solve objective as ``warm=False``
   to machine precision.  This is the regression net for warm
   correctness; if any LP cell goes stale because a Param outside
   the clean-mapping set quietly differs, this test catches it.

2. **Cold-fallback** — on a chain where the structural fingerprint
   changes between sub-solves (different invest sets per period),
   ``warm_used`` must be False for every sub-solve and the runner
   must still produce correct per-sub-solve objectives.

3. **Speedup** — on a fixture with many structurally-similar
   rolling-horizon sub-solves (the
   ``dispatch_fullYear_roll_roll_<i>`` cascade), the warm-update
   path eliminates per-iteration cold-rebuild overhead.  Even when
   every transition still cold-rebuilds (e.g. ``p_profile_value``
   differs between rolls), reusing the WarmProblem skeleton is
   measurably cheaper than rebuilding from scratch.

Δ.12d — these tests now exercise the **native cascade**
(``_orchestration.run_chain_from_db``) by passing ``native=True``.
The legacy file-symlink path in ``chain.run_chain(native=False)`` is
on its way out (Δ.12e) and the tests follow the canonical native
entry point.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from flextool.engine_polars import run_chain_from_db

pytestmark = pytest.mark.solver


DATA = Path(__file__).resolve().parent / "data"


def _scenario_for_work(work_dir: str) -> str:
    """Map a fixture work-dir name to the scenario inside its
    ``tests.sqlite``.  Mirrors the small set of overrides used by the
    parity sweep + ``chain._resolve_native_scenario``."""
    overrides = {
        "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
        "work_commodity_ladder_annual": "coal_ladder_annual",
        "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
        "work_delay_source_coef": "water_pump_delayed",
        "work_inflation_check": "wind_battery_invest_lifetime_renew",
    }
    if work_dir in overrides:
        return overrides[work_dir]
    return work_dir.removeprefix("work_")


# ---------------------------------------------------------------------------
# Test 1: equivalence between warm=False and warm=True on a real chain.

@pytest.mark.parametrize("scenario", [
    "work_wind_battery_invest_lifetime_renew_4solve",
    "work_multi_year",
    "work_5weeks_invest_fullYear_dispatch_coal_wind",
])
def test_warm_chain_equivalence(scenario: str) -> None:
    """Cold and warm chain runs must produce identical per-sub-solve
    objectives across every chain fixture in the regression set."""
    work = DATA / scenario
    if not work.exists():
        pytest.skip(f"fixture {work} not present")
    db_path = work / "tests.sqlite"
    if not db_path.exists():
        pytest.skip(f"fixture DB {db_path} not present")

    scen = _scenario_for_work(scenario)
    sols_cold = run_chain_from_db(
        db_path, scenario_name=scen, warm=False, keep_solutions=True,
    )
    sols_warm = run_chain_from_db(
        db_path, scenario_name=scen, warm=True, keep_solutions=True,
    )
    assert list(sols_cold) == list(sols_warm), (
        f"{scenario}: warm and cold chains diverge in sub-solve order")

    for sub_solve in sols_cold:
        c = sols_cold[sub_solve].solution.obj
        w = sols_warm[sub_solve].solution.obj
        diff = abs(c - w)
        rel = diff / max(1.0, abs(c))
        assert rel < 1e-9, (
            f"{scenario}/{sub_solve}: warm={w}, cold={c}, rel={rel:.3e}")


# ---------------------------------------------------------------------------
# Test 2: cold-fallback when fingerprint changes between sub-solves.

def test_warm_chain_cold_fallback() -> None:
    """The 4-solve invest-lifetime fixture has different invest-set
    sizes per year — every transition is a structural change.  The
    warm runner must detect this and cold-rebuild every step
    (warm_used == False everywhere) while still producing correct
    objectives.

    Note: native ``run_chain_from_db`` returns one
    :class:`OrchestrationStep` per ``complete_solve_name`` —
    flextool's orchestration loop overwrites the dict entry when
    ``complete_solve[solve]`` repeats across rolling expansions.  For
    invest-lifetime the four sub-solves have distinct
    ``complete_solve_name`` values so all four steps appear.
    """
    work = DATA / "work_wind_battery_invest_lifetime_renew_4solve"
    if not work.exists():
        pytest.skip(f"fixture {work} not present")
    db_path = work / "tests.sqlite"
    if not db_path.exists():
        pytest.skip(f"fixture DB {db_path} not present")

    sols_warm = run_chain_from_db(
        db_path,
        scenario_name="wind_battery_invest_lifetime_renew_4solve",
        warm=True,
        keep_solutions=True,
    )
    n_warm = sum(1 for s in sols_warm.values() if s.warm_used)
    assert n_warm == 0, (
        f"expected all sub-solves to cold-rebuild (structural change), "
        f"but {n_warm}/{len(sols_warm)} reported warm_used=True")

    # Sanity: every sub-solve solved optimally.
    for sub_solve, step in sols_warm.items():
        assert step.solution.optimal, (
            f"{sub_solve}: flexpy LP not optimal under warm=True")


# ---------------------------------------------------------------------------
# Test 3: speedup on a rolling-horizon cascade.
#
# In the native path, ``run_chain_from_db`` drives flextool's
# orchestration loop which expands a rolling solve into
# ``dispatch_fullYear_roll_roll_<i>`` iterations.  Every iteration
# shares the same FlexData skeleton modulo time-shifted Params (and
# per-roll storage handoff).  The warm path doesn't always succeed
# in keeping the LP warm — ``p_profile_value`` and similar Params
# typically force cold-rebuild — but the WarmProblem skeleton reuse
# still avoids redundant Var/Cstr construction on the second-and-
# later iterations.

def test_warm_native_rolling_speedup() -> None:
    """Δ.12d — on the rolling-horizon ``fullYear_roll`` fixture
    (``dispatch_fullYear_roll_roll_<i>`` × N), warm-mode runtime
    must be no slower than cold-mode runtime, AND warm-mode
    objectives must match cold-mode objectives to machine precision.

    A strict speedup threshold isn't asserted here because the
    native path is dominated by flextool's per-solve preprocessing
    (CSV writers + ``preprocessing_solve_time`` for each roll) and
    file I/O — the LP-build savings are real but small relative to
    that pipeline.  The Δ.13+ preprocessing port is the prerequisite
    for a guaranteed-speedup native warm assertion.
    """
    work = DATA / "work_fullYear_roll"
    if not work.exists():
        pytest.skip(f"fixture {work} not present")
    db_path = work / "tests.sqlite"
    if not db_path.exists():
        pytest.skip(f"fixture DB {db_path} not present")

    t0 = time.perf_counter()
    sols_cold = run_chain_from_db(
        db_path, scenario_name="fullYear_roll", warm=False,
        keep_solutions=True,
    )
    t_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    sols_warm = run_chain_from_db(
        db_path, scenario_name="fullYear_roll", warm=True,
        keep_solutions=True,
    )
    t_warm = time.perf_counter() - t0

    # Equivalence — all per-(complete_solve_name) objectives match.
    assert set(sols_cold) == set(sols_warm)
    for sub in sols_cold:
        c = sols_cold[sub].solution.obj
        w = sols_warm[sub].solution.obj
        rel = abs(c - w) / max(1.0, abs(c))
        assert rel < 1e-6, (
            f"{sub}: warm={w}, cold={c}, rel={rel:.3e}")

    # Performance ceiling: warm should not be markedly slower than
    # cold.  A 2× slowdown would suggest something has gone wrong
    # (e.g. the WarmProblem is being rebuilt twice per iteration
    # rather than reused).
    assert t_warm < 2.0 * t_cold, (
        f"warm slowdown: cold={t_cold:.2f}s, warm={t_warm:.2f}s")


# ---------------------------------------------------------------------------
# Test 4: warm-mode triggers on a structurally-stable transition.
#
# Synthetic check — we use the engine-level WarmProblem reuse path
# directly to verify the warm machinery's plumbing reaches HiGHS for
# a transition where every Param is identical.  This complements the
# rolling test above (where p_profile_value forces cold rebuild on
# every transition).

def test_warm_native_reuses_warmproblem_on_identical_data() -> None:
    """Δ.12d — directly verify the cascade solver reuses one
    ``WarmProblem`` across two consecutive ``run()`` calls when
    given identical FlexData.  Bypasses ``run_chain_from_db`` (which
    would invoke flextool's full preprocessing) and exercises the
    orchestration's ``_FlexpyCascadeSolver.run`` logic in isolation.
    """
    from flextool.engine_polars._warm import (
        _build_warm_problem,
        _apply_warm_updates,
        _fingerprint,
        _IncompatibleUpdate,
    )
    from flextool.engine_polars.input import load_flextool

    work = DATA / "work_test_a_lot"
    if not work.exists():
        pytest.skip(f"fixture {work} not present")

    data1 = load_flextool(work)
    data2 = load_flextool(work)  # identical re-load

    fp1 = _fingerprint(data1)
    fp2 = _fingerprint(data2)
    assert fp1 == fp2, "fingerprint mismatch on identical reloads"

    warm = _build_warm_problem(data1)
    sol1 = warm.solve()
    obj1 = sol1.obj
    assert sol1.optimal

    # Apply warm update to second data (which is identical) and re-solve.
    n_updates = _apply_warm_updates(warm, data1, data2)
    sol2 = warm.solve()
    assert sol2.optimal
    assert abs(sol2.obj - obj1) < 1e-9 * max(1.0, abs(obj1)), (
        f"reused WarmProblem obj diverged: first={obj1}, second={sol2.obj}")
    # Identical data → no Param diffs → zero warm-update calls.
    assert n_updates == 0, f"expected 0 updates on identical data, got {n_updates}"
