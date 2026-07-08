"""First-transfer A/B GATING tests for the PARENT subprocess solver.

Exercises the warm-start A/B measurement gate in
:func:`flextool.engine_polars._subprocess_solve._solve_highs_subprocess`
(spec §7 B / §6.2): on a fingerprint's FIRST warm transfer the parent
runs one extra COLD timing probe child alongside the warm main run, then
compares wall times.  A regression (warm slower than
``cold * (1 + _WARM_AB_MARGIN)``) writes ``<fp>.nowarm`` and disables
future injection; otherwise ``<fp>.abtested`` is written and later solves
trust warm without re-probing.  Everything fails safe to a correct cold
solve.

Child spawns are counted/inspected by spying on ``subprocess.run`` in the
``_subprocess_solve`` module (delegating to the real one).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import polar_high as fp
from flextool.engine_polars import _subprocess_solve as _subp
from flextool.engine_polars._subprocess_solve import solve_via_subprocess

# A single-roll storage-dispatch LP large enough to take a non-trivial
# number of simplex iterations cold (so both the cold probe and the warm
# run produce a measurable, nonzero wall time).
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
    usable (matches the child-side warm-vs-cold bench)."""
    return {"solver": "simplex", "presolve": "off"}


class _RunSpy:
    """Records every ``subprocess.run`` argv, then delegates to the real one.

    ``calls`` holds the argv (first positional arg) of every spawn since
    the last :meth:`reset`.  Used to count child spawns per solve and to
    inspect whether a given spawn carried ``--warm-basis``.
    """

    def __init__(self, real):
        self._real = real
        self.calls: list[list[str]] = []

    def reset(self) -> None:
        self.calls = []

    def __call__(self, cmd, *args, **kwargs):
        # ``cmd`` is the argv list for the child; snapshot it.
        try:
            self.calls.append(list(cmd))
        except TypeError:  # pragma: no cover - defensive
            self.calls.append([str(cmd)])
        return self._real(cmd, *args, **kwargs)


def _has_warm(argv: list[str]) -> bool:
    return "--warm-basis" in argv


def _install_spy(monkeypatch) -> _RunSpy:
    spy = _RunSpy(_subp.subprocess.run)
    monkeypatch.setattr(_subp.subprocess, "run", spy)
    return spy


def test_first_transfer_probes_then_trusts(tmp_path: Path, monkeypatch) -> None:
    """Solve #1 cold (miss); solve #2 first-transfer runs a cold probe +
    warm run (2 spawns, one w/o and one w/ ``--warm-basis``) and writes
    ``.abtested``; solve #3 trusts warm (1 spawn, warm injected)."""
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    # Positive margin: on an identical model the warm run is never slower
    # than cold by 10%, so the verdict is "beneficial" (.abtested).
    monkeypatch.setattr(_subp, "_WARM_AB_MARGIN", 0.10)
    spy = _install_spy(monkeypatch)

    work = tmp_path / "work"
    work.mkdir()

    # --- Solve #1: cache miss → cold, publishes the basis, no probe. ---
    spy.reset()
    p1 = _build_problem(seed=1)
    sol1 = solve_via_subprocess(
        p1, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol1.optimal
    fp1 = p1._last_mps_fingerprint
    assert fp1 is not None
    cache_dir = work / "basis_cache"
    cache_file = cache_dir / f"{fp1}.bas"
    assert cache_file.exists(), "solve #1 did not publish a basis cache"
    assert len(spy.calls) == 1, f"miss should spawn once, got {spy.calls}"
    assert not _has_warm(spy.calls[0]), "cold miss must not inject a basis"
    assert not (cache_dir / f"{fp1}.abtested").exists()
    assert not (cache_dir / f"{fp1}.nowarm").exists()

    # --- Solve #2: cache hit, FIRST transfer → cold probe + warm run. ---
    spy.reset()
    p2 = _build_problem(seed=1)
    sol2 = solve_via_subprocess(
        p2, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol2.optimal
    assert p2._last_mps_fingerprint == fp1, "fingerprint drifted across builds"
    assert len(spy.calls) == 2, (
        f"first transfer must spawn a cold probe + the warm run, "
        f"got {len(spy.calls)}: {spy.calls}"
    )
    warm_flags = [_has_warm(c) for c in spy.calls]
    assert warm_flags.count(True) == 1 and warm_flags.count(False) == 1, (
        f"expected exactly one probe (no --warm-basis) and one warm run "
        f"(with --warm-basis), got {warm_flags}"
    )
    assert (cache_dir / f"{fp1}.abtested").exists(), (
        "first transfer did not record the .abtested verdict"
    )
    assert not (cache_dir / f"{fp1}.nowarm").exists(), (
        ".nowarm must NOT be written when warm is beneficial"
    )
    assert abs(sol1.obj - sol2.obj) < 1e-6, (
        f"objectives diverged: cold={sol1.obj} warm={sol2.obj}"
    )

    # --- Solve #3: cache hit, .abtested present → trust warm, NO probe. ---
    spy.reset()
    p3 = _build_problem(seed=1)
    sol3 = solve_via_subprocess(
        p3, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol3.optimal
    assert len(spy.calls) == 1, (
        f"trusted (.abtested) solve must NOT re-probe, got {spy.calls}"
    )
    assert _has_warm(spy.calls[0]), "trusted solve must inject the warm basis"
    assert abs(sol1.obj - sol3.obj) < 1e-6, (
        f"objectives diverged: cold={sol1.obj} trusted-warm={sol3.obj}"
    )
    # No leftover probe / tmp debris.
    assert not list(cache_dir.glob("*.bas.tmp.*"))


def test_regression_writes_nowarm_and_disables(
    tmp_path: Path, monkeypatch
) -> None:
    """A forced-regression margin makes the first transfer write ``.nowarm``;
    the next solve then never injects the basis (solves cold) and still
    reaches the correct optimum."""
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    # NEGATIVE margin: warm is judged a regression for any positive cold
    # time (warm_t > cold_t * (1 - 2.0) = cold_t * -1.0 is always true).
    monkeypatch.setenv("FLEXTOOL_WARM_AB_MARGIN", "-2.0")
    monkeypatch.setattr(_subp, "_WARM_AB_MARGIN", -2.0)
    spy = _install_spy(monkeypatch)

    work = tmp_path / "work"
    work.mkdir()

    # --- Solve #1: miss → cold, publishes basis. ---
    p1 = _build_problem(seed=3)
    sol1 = solve_via_subprocess(
        p1, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol1.optimal
    fp3 = p1._last_mps_fingerprint
    assert fp3 is not None
    cache_dir = work / "basis_cache"
    assert (cache_dir / f"{fp3}.bas").exists()

    # --- Solve #2: first transfer, forced regression → .nowarm. ---
    spy.reset()
    p2 = _build_problem(seed=3)
    sol2 = solve_via_subprocess(
        p2, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol2.optimal
    assert len(spy.calls) == 2, (
        f"first transfer must still run the cold probe + warm run, "
        f"got {spy.calls}"
    )
    assert (cache_dir / f"{fp3}.nowarm").exists(), (
        "forced-regression first transfer did not write .nowarm"
    )
    assert not (cache_dir / f"{fp3}.abtested").exists()
    assert abs(sol1.obj - sol2.obj) < 1e-6

    # --- Solve #3: .nowarm present → NO injection, cold, correct optimum. ---
    spy.reset()
    p3 = _build_problem(seed=3)
    sol3 = solve_via_subprocess(
        p3, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol3.optimal
    assert len(spy.calls) == 1, (
        f".nowarm solve must not probe, got {spy.calls}"
    )
    assert not _has_warm(spy.calls[0]), (
        ".nowarm must disable warm-basis injection"
    )
    assert abs(sol1.obj - sol3.obj) < 1e-6, (
        f"disabled-warm solve missed the optimum: got={sol3.obj} "
        f"ref={sol1.obj}"
    )


def test_preexisting_nowarm_never_injects(tmp_path: Path, monkeypatch) -> None:
    """A pre-placed ``<fp>.nowarm`` (from an earlier regression verdict)
    disables injection with no A/B probe and still reaches the optimum."""
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    spy = _install_spy(monkeypatch)

    # Reference objective + a real cached basis from a clean solve.
    work_ref = tmp_path / "ref"
    work_ref.mkdir()
    p_ref = _build_problem(seed=5)
    sol_ref = solve_via_subprocess(
        p_ref, "highs", _opts(), solve_name="s", work_folder=work_ref
    )
    assert sol_ref.optimal
    fp5 = p_ref._last_mps_fingerprint
    assert fp5 is not None
    ref_bas = (work_ref / "basis_cache" / f"{fp5}.bas").read_bytes()

    # Fresh cache: seed it with a real basis AND a pre-existing .nowarm.
    work = tmp_path / "work"
    work.mkdir()
    cache_dir = work / "basis_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{fp5}.bas").write_bytes(ref_bas)
    (cache_dir / f"{fp5}.nowarm").write_text("")

    spy.reset()
    p = _build_problem(seed=5)
    sol = solve_via_subprocess(
        p, "highs", _opts(), solve_name="s", work_folder=work
    )
    assert sol.optimal
    assert len(spy.calls) == 1, (
        f"pre-existing .nowarm must skip the A/B probe, got {spy.calls}"
    )
    assert not _has_warm(spy.calls[0]), (
        "pre-existing .nowarm must disable warm-basis injection"
    )
    assert not (cache_dir / f"{fp5}.abtested").exists()
    assert abs(sol.obj - sol_ref.obj) < 1e-6, (
        f"nowarm solve missed the optimum: got={sol.obj} ref={sol_ref.obj}"
    )
    # The capture still refreshes the cached basis on a .nowarm solve.
    assert (cache_dir / f"{fp5}.bas").exists()
    assert not list(cache_dir.glob("*.bas.tmp.*"))
