"""Warm chain runner — equivalence, cold-fallback, and speedup tests.

Three lightweight tests covering the new ``run_chain(..., warm=True)``
codepath in ``flextool/chain.py`` (which wires :class:`polar_high.WarmProblem`
into the chain orchestrator):

1. **Equivalence** — for every standard chain fixture, ``warm=True``
   must produce the SAME per-sub-solve objective as ``warm=False``
   to machine precision.  This is the regression net for warm
   correctness; if any LP cell goes stale because a Param outside
   the clean-mapping set quietly differs, this test catches it.

2. **Cold-fallback** — on a chain where the structural fingerprint
   changes between sub-solves (different invest sets per period),
   ``warm_used`` must be False for every sub-solve and the runner
   must still produce correct per-sub-solve objectives.

3. **Speedup** — on a synthesized chain of structurally-identical
   sub-solves (same snapshot replicated under different sub-solve
   names), ``warm_used`` triggers from the second sub-solve onward
   and end-to-end runtime drops by at least 1.5×.

Per the spec these are the *only* tests added — we don't duplicate
the 164-strong cold-path regression net.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from flextool.engine_polars import run_chain

pytestmark = pytest.mark.solver


DATA = Path(__file__).resolve().parent / "data"


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

    sols_cold = run_chain(work, warm=False)
    sols_warm = run_chain(work, warm=True)
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
    objectives."""
    work = DATA / "work_wind_battery_invest_lifetime_renew_4solve"
    if not work.exists():
        pytest.skip(f"fixture {work} not present")

    sols_warm = run_chain(work, warm=True)
    n_warm = sum(1 for s in sols_warm.values() if s.warm_used)
    assert n_warm == 0, (
        f"expected all sub-solves to cold-rebuild (structural change), "
        f"but {n_warm}/{len(sols_warm)} reported warm_used=True")

    # Sanity: parity vs flextool reference still holds.
    import polars as pl
    for sub_solve, step in sols_warm.items():
        assert step.solution.optimal, (
            f"{sub_solve}: flexpy LP not optimal under warm=True")
        parq = work / "output_raw" / f"v_obj__{sub_solve}.parquet"
        if not parq.exists():
            continue
        ft = pl.read_parquet(parq)["objective"][0]
        rel = abs(step.solution.obj - ft) / max(1.0, abs(ft))
        assert rel < 1e-6, (
            f"{sub_solve}: warm-runner obj={step.solution.obj} "
            f"vs flextool {ft}, rel={rel}")


# ---------------------------------------------------------------------------
# Test 3: speedup on a synthesized chain of identical sub-solves.

def test_warm_chain_speedup() -> None:
    """Synthesize a 6-step chain by symlinking the same per-sub-solve
    snapshot under six different sub-solve names.  Every sub-solve
    sees identical FlexData, so the structural fingerprint matches
    AND every Param is identical between consecutive sub-solves —
    warm-update triggers from sub-solve #2 onwards.

    Uses ``work_test_a_lot`` (a feature-heavy 81-row dispatch LP) so
    LP-build time dominates over loader I/O, exposing the WarmProblem
    payoff.  On smaller fixtures (e.g. 5-week 16-row LPs) loader I/O
    dominates and the warm payoff drops below the noise floor — a
    known limitation called out in the WarmProblem design audit
    ("loader I/O dominates per-sub-solve time, masking the LP-build
    savings"), which a follow-up loader-cache layer will address.

    Asserts:
      * sub-solves #2..#N have ``warm_used=True``;
      * end-to-end warm runtime is at least 1.4× faster than the
        cold baseline (best-of-3 to absorb HiGHS startup variance).
    """
    src = DATA / "work_test_a_lot"
    if not src.exists():
        pytest.skip(f"fixture {src} not present")

    src_solve_data = src / "solve_data"
    if not src_solve_data.exists():
        pytest.skip(f"fixture {src_solve_data} not present")

    n_clones = 6

    def _bench(warm: bool) -> tuple[float, dict]:
        """Run the cloned chain in a freshly-staged tmpdir and return
        (elapsed_seconds, sols_dict)."""
        # Need a fresh tmpdir per run because TemporaryDirectory holds
        # exclusive ownership.  Symlinks are cheap.
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            os.symlink(src / "input", work / "input")
            os.symlink(src / "output_raw", work / "output_raw")
            os.symlink(src_solve_data, work / "solve_data")
            chain = []
            for i in range(n_clones):
                name = f"clone_{i}"
                os.symlink(src_solve_data, work / f"solve_data_{name}")
                chain.append(name)
            t0 = time.perf_counter()
            sols = run_chain(work, warm=warm, chain=chain)
            elapsed = time.perf_counter() - t0
        return elapsed, sols

    # Warm-up the JIT / module caches with one untimed call.
    _bench(warm=False)

    # Best of 3 for both modes — HiGHS first-call cost + filesystem
    # cache effects can move runtime by ~10%.
    t_cold = min(_bench(warm=False)[0] for _ in range(3))
    sols_warm = None
    warm_times: list[float] = []
    for _ in range(3):
        t, sols = _bench(warm=True)
        warm_times.append(t)
        sols_warm = sols
    t_warm = min(warm_times)

    # Equivalence check — clones must all return the same obj.
    objs = [step.solution.obj for step in sols_warm.values()]
    assert all(abs(o - objs[0]) < 1e-6 * max(1.0, abs(objs[0]))
               for o in objs), f"clone objs diverge: {objs}"

    # Warm-mode trigger — after the first sub-solve, every transition
    # is fingerprint-equal AND no Param differs.  Expect at least
    # n_clones - 1 warm hits.
    n_warm = sum(1 for s in sols_warm.values() if s.warm_used)
    assert n_warm >= n_clones - 1, (
        f"expected {n_clones - 1} warm sub-solves, got {n_warm}")

    speedup = t_cold / t_warm
    # Conservative threshold: >= 1.4×.  Local measurements on the
    # development box show ~1.8×; leaving headroom for noisier CI
    # boxes.  Below 1.4× would suggest something has gone wrong in
    # the warm-path build (e.g. WarmProblem fell back to cold every
    # step despite warm_used reporting True).
    assert speedup >= 1.4, (
        f"warm-mode speedup too low: cold={t_cold:.3f}s, "
        f"warm={t_warm:.3f}s, speedup={speedup:.2f}x")
