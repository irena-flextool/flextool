"""Warm-start basis-cache tests for the PARENT subprocess solver.

Exercises the parent half of the ``save_memory`` warm-start arm in
:func:`flextool.engine_polars._subprocess_solve.solve_via_subprocess`:
a fingerprint-keyed native HiGHS ``.bas`` basis cache that survives the
per-solve ``out_dir`` cleanup, is injected via the child's
``--warm-basis`` on a matching re-solve, refreshed atomically
(``os.replace``) on success, and fails safe to a cold solve on any
mismatch.  Driven end-to-end through the real ``cmd_solve_mps`` child.

The child (``cmd_solve_mps`` with ``--warm-basis``/``--basis``/
``--stats``) is already committed and tested separately; here we only
consume it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

import polar_high as fp
from flextool.engine_polars._subprocess_solve import solve_via_subprocess

# A single-roll storage-dispatch LP large enough to take a non-trivial
# number of simplex iterations cold, so an injected basis measurably cuts
# them (mirrors the child-side warm-vs-cold bench).
_N_T = 200


def _build_problem(seed: int) -> fp.Problem:
    """Build a ``_N_T``-step storage-dispatch LP for one roll."""
    rng = np.random.default_rng(seed)
    cost = rng.uniform(1.0, 10.0, size=_N_T)
    demand = rng.uniform(20.0, 80.0, size=_N_T)

    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(_N_T, dtype=np.int64)})
    v_flow = p.add_var("v_flow", "t", t_idx, lower=0.0, upper=1.0e6)
    v_state = p.add_var("v_state", "t", t_idx, lower=0.0, upper=1.0e6)

    lag = pl.DataFrame(
        {
            "t": np.arange(1, _N_T, dtype=np.int64),
            "t_prev": np.arange(0, _N_T - 1, dtype=np.int64),
        }
    )
    s_lag = fp.Lag(v_state, lag, time_dim="t", lag_col="t_prev")

    demand_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(_N_T, dtype=np.int64), "value": demand}),
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={
            "v_flow": v_flow,
            "s_lag": s_lag,
            "minus_s": -v_state.to_expr(),
        },
        rhs_terms={"demand": demand_p},
    )

    cost_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(_N_T, dtype=np.int64), "value": cost}),
    )
    p.set_objective(cost_p * v_flow, sense="min")
    return p


def _opts() -> dict[str, str]:
    """Force simplex + no presolve so the injected basis is directly
    usable and its iteration savings are observable."""
    return {"solver": "simplex", "presolve": "off"}


def _stats_path(work: Path, solve_name: str = "s") -> Path:
    return work / "solve_data" / "subprocess" / f"{solve_name}.stats.json"


def test_cold_then_warm_reuses_cached_basis(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    work = tmp_path / "work"
    work.mkdir()

    # --- Call 1: cold, populates the cache ---
    p1 = _build_problem(seed=1)
    sol1 = solve_via_subprocess(
        p1, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol1.optimal
    fp1 = p1._last_mps_fingerprint
    assert fp1 is not None
    cache_file = work / "basis_cache" / f"{fp1}.bas"
    assert cache_file.exists(), "call 1 did not publish a basis cache"

    cold = json.loads(_stats_path(work).read_text())
    iters_cold = cold["simplex_iteration_count"]
    assert iters_cold > 0, f"cold solve took 0 simplex iters: {cold}"
    assert cold["warm_basis_used"] is False

    # --- Call 2: identical model, SAME work_folder → cache hit ---
    p2 = _build_problem(seed=1)
    sol2 = solve_via_subprocess(
        p2, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol2.optimal
    assert p2._last_mps_fingerprint == fp1, "fingerprint drifted across builds"

    warm = json.loads(_stats_path(work).read_text())
    iters_warm = warm["simplex_iteration_count"]
    assert warm["warm_basis_used"] is True, "call 2 did not consume the cache"
    assert iters_warm < iters_cold, (
        f"warm basis did not reduce simplex iters: "
        f"cold={iters_cold} warm={iters_warm}"
    )
    assert abs(sol1.obj - sol2.obj) < 1e-6, (
        f"objectives diverged: cold={sol1.obj} warm={sol2.obj}"
    )

    # Atomic publish: no leftover *.bas.tmp.* debris in the cache dir.
    leftovers = list((work / "basis_cache").glob("*.bas.tmp.*"))
    assert not leftovers, f"leftover tmp basis files: {leftovers}"


def test_off_path_creates_no_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FLEXTOOL_WARM_START", raising=False)
    work = tmp_path / "work"
    work.mkdir()

    p = _build_problem(seed=1)
    sol = solve_via_subprocess(
        p, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol.optimal
    assert np.isfinite(sol.obj)
    # Zero behaviour change: no cache dir, no stats sidecar.
    assert not (work / "basis_cache").exists()
    assert not _stats_path(work).exists()


def test_stale_cache_falls_back_to_cold(tmp_path: Path, monkeypatch) -> None:
    """A garbage/mismatched ``.bas`` under the fingerprint filename must be
    rejected by the child (readBasis kError) → cold solve at the correct
    optimum, no crash."""
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")

    # Reference objective from a clean (empty-cache) solve of the model.
    work_ref = tmp_path / "ref"
    work_ref.mkdir()
    p_ref = _build_problem(seed=7)
    sol_ref = solve_via_subprocess(
        p_ref, "highs", _opts(), solve_name="s", work_folder=work_ref
    )
    assert sol_ref.optimal
    obj_ref = sol_ref.obj
    fp7 = p_ref._last_mps_fingerprint
    assert fp7 is not None

    # Pre-place garbage under the fingerprint filename in a fresh cache.
    work = tmp_path / "work"
    work.mkdir()
    cache_dir = work / "basis_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{fp7}.bas").write_bytes(
        b"HiGHS v1\nnot a real basis\n\x00\x01garbage"
    )

    p = _build_problem(seed=7)
    sol = solve_via_subprocess(
        p, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol.optimal
    stats = json.loads(_stats_path(work).read_text())
    assert stats["warm_basis_used"] is False, "garbage basis was not rejected"
    assert abs(sol.obj - obj_ref) < 1e-6, (
        f"stale-cache solve missed the optimum: got={sol.obj} ref={obj_ref}"
    )
    # The rejected garbage is overwritten by this run's fresh capture.
    assert (cache_dir / f"{fp7}.bas").exists()
    assert not list(cache_dir.glob("*.bas.tmp.*"))
