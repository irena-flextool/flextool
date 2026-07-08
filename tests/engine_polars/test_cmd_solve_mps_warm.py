"""Warm-start (basis-injection) contract tests for ``cmd_solve_mps``.

Exercises the SUBPROCESS-CHILD half of the ``save_memory`` warm-start arm:
inject a prior HiGHS basis via ``readBasis`` before solving, capture this
run's basis, and emit a measurement sidecar.  Driven in-process via
``flextool.cli.cmd_solve_mps.main([...])`` with LP inputs built by
polar-high in the same process.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

import polar_high as fp
from flextool.cli import cmd_solve_mps

# A single-roll rolling-horizon LP, large enough to take a non-trivial
# number of simplex iterations cold (so a warm basis can measurably cut
# them).  Structure mirrors polar-high's synthetic warm-vs-cold bench.
_N_T = 200


def _build_problem(cost: np.ndarray, demand: np.ndarray) -> fp.Problem:
    """Build a ``_N_T``-step storage-dispatch LP for one roll."""
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


def _write_mps(tmp_path: Path, name: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    cost = rng.uniform(1.0, 10.0, size=_N_T)
    demand = rng.uniform(20.0, 80.0, size=_N_T)
    p = _build_problem(cost, demand)
    mps = tmp_path / f"{name}.mps"
    p.write_mps(mps, release=True)
    return mps


def _options_file(tmp_path: Path) -> Path:
    """Force simplex + no presolve so a warm basis is directly usable and
    its iteration savings are observable (presolve would remap the basis)."""
    opt = tmp_path / "highs.opt"
    opt.write_text("solver=simplex\npresolve=off\n")
    return opt


def _run(*argv: str) -> int:
    """Call the child entrypoint, tolerating a possible ``SystemExit``."""
    try:
        rc = cmd_solve_mps.main(list(argv))
    except SystemExit as exc:  # pragma: no cover - main returns int directly
        code = exc.code
        rc = 0 if code is None else code
    return int(rc)


def test_warm_basis_cuts_simplex_iterations(tmp_path: Path) -> None:
    mps = _write_mps(tmp_path, "model", seed=1)
    opts = _options_file(tmp_path)

    sol1 = tmp_path / "cold.sol"
    bas = tmp_path / "run.bas"
    stats1 = tmp_path / "cold.json"

    # --- Capture (cold) run ---
    rc = _run(
        "--mps", str(mps),
        "--solution", str(sol1),
        "--basis", str(bas),
        "--stats", str(stats1),
        "--options", str(opts),
    )
    assert rc == 0
    assert sol1.exists()
    assert bas.exists()
    assert stats1.exists()

    cold = json.loads(stats1.read_text())
    iters_cold = cold["simplex_iteration_count"]
    obj_cold = cold["objective"]
    assert iters_cold > 0, f"cold solve took 0 simplex iters: {cold}"
    assert cold["warm_basis_used"] is False

    # --- Warm run (inject the captured basis) ---
    sol2 = tmp_path / "warm.sol"
    stats2 = tmp_path / "warm.json"
    rc = _run(
        "--mps", str(mps),
        "--solution", str(sol2),
        "--warm-basis", str(bas),
        "--stats", str(stats2),
        "--options", str(opts),
    )
    assert rc == 0
    warm = json.loads(stats2.read_text())
    iters_warm = warm["simplex_iteration_count"]

    assert warm["warm_basis_used"] is True
    assert iters_warm < iters_cold, (
        f"warm basis did not reduce simplex iters: "
        f"cold={iters_cold} warm={iters_warm}"
    )
    assert warm["objective"] == obj_cold


def test_missing_warm_basis_solves_cold(tmp_path: Path) -> None:
    mps = _write_mps(tmp_path, "model", seed=2)
    opts = _options_file(tmp_path)
    sol = tmp_path / "out.sol"
    stats = tmp_path / "out.json"

    rc = _run(
        "--mps", str(mps),
        "--solution", str(sol),
        "--warm-basis", str(tmp_path / "nonexistent.bas"),
        "--stats", str(stats),
        "--options", str(opts),
    )
    assert rc == 0
    assert sol.exists()
    data = json.loads(stats.read_text())
    assert data["warm_basis_used"] is False
    assert data["simplex_iteration_count"] > 0


def test_mismatched_warm_basis_solves_cold(tmp_path: Path) -> None:
    """A basis captured from a DIFFERENT model must fall back to a cold
    solve (readBasis kError), not crash, and reach the correct objective."""
    opts = _options_file(tmp_path)

    # Reference cold solve of the target model, for the objective.
    mps = _write_mps(tmp_path, "target", seed=3)
    sol_ref = tmp_path / "ref.sol"
    stats_ref = tmp_path / "ref.json"
    rc = _run(
        "--mps", str(mps),
        "--solution", str(sol_ref),
        "--stats", str(stats_ref),
        "--options", str(opts),
    )
    assert rc == 0
    obj_ref = json.loads(stats_ref.read_text())["objective"]

    # A basis from a structurally different (smaller) model.
    other = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(5, dtype=np.int64)})
    y = other.add_var("y", "t", t_idx, lower=0.0, upper=10.0)
    ones = fp.Param(
        ("t",), pl.DataFrame({"t": np.arange(5, dtype=np.int64), "value": np.ones(5)})
    )
    other.add_cstr(
        "cap",
        over=t_idx,
        sense="<=",
        lhs_terms={"y": y},
        rhs_terms={"rhs": ones},
    )
    other.set_objective(-ones * y, sense="min")
    other_mps = tmp_path / "other.mps"
    other.write_mps(other_mps, release=True)
    other_sol = tmp_path / "other.sol"
    other_bas = tmp_path / "other.bas"
    rc = _run(
        "--mps", str(other_mps),
        "--solution", str(other_sol),
        "--basis", str(other_bas),
        "--options", str(opts),
    )
    assert rc == 0
    assert other_bas.exists()

    # Inject the mismatched basis into the target model.
    sol = tmp_path / "mismatch.sol"
    stats = tmp_path / "mismatch.json"
    rc = _run(
        "--mps", str(mps),
        "--solution", str(sol),
        "--warm-basis", str(other_bas),
        "--stats", str(stats),
        "--options", str(opts),
    )
    assert rc == 0
    data = json.loads(stats.read_text())
    assert data["warm_basis_used"] is False
    assert data["objective"] == obj_ref


def test_truncated_warm_basis_solves_cold(tmp_path: Path) -> None:
    mps = _write_mps(tmp_path, "model", seed=4)
    opts = _options_file(tmp_path)
    garbage = tmp_path / "garbage.bas"
    garbage.write_text("HiGHS v1\nnot a real basis\n\x00\x01garbage")

    sol = tmp_path / "out.sol"
    stats = tmp_path / "out.json"
    rc = _run(
        "--mps", str(mps),
        "--solution", str(sol),
        "--warm-basis", str(garbage),
        "--stats", str(stats),
        "--options", str(opts),
    )
    assert rc == 0
    data = json.loads(stats.read_text())
    assert data["warm_basis_used"] is False
    assert data["simplex_iteration_count"] > 0
