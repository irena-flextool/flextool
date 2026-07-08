"""Phase-4 Step 4b — IN-PROCESS warm-start basis cache (cascade tests).

Unlike ``test_warm_start_cascade.py`` (which drives the ``save_memory``
subprocess ``.bas`` arm), this file exercises the DEFAULT in-process warm
path: ``run_chain_from_db(..., warm=True)`` (the default) builds a live
:class:`polar_high.WarmProblem`, and under ``FLEXTOOL_WARM_START=1`` the
orchestrator captures the solved optimal basis as a
:class:`polar_high.NamedBasis`, keyed by the LP name-set fingerprint
(``Problem.basis_name_fingerprint``), storing it both in an in-process dict
and as ``<cache_dir>/<fp>.nbasis`` (JSON).  A later cross-run solve of the
same structural model injects that cached basis via
``WarmProblem.set_named_basis`` before its first solve.

We deliberately do NOT set ``FLEXTOOL_SAVE_MEMORY`` — that would route the
solve through the subprocess arm instead of the in-process WarmProblem.

The model is the same genuine, migrating fixture used by
``test_solver_integration.py`` / ``test_warm_start_cascade.py`` — a single
HiGHS sub-solve that resolves to a fresh WarmProblem build (CLAUDE.md
invariant: build the DB from JSON/schema, never a checked-in ``.sqlite``).

Assertions key on the observable contract of the in-process arm:
``<fp>.nbasis`` on disk, the ``WarmProblem.set_named_basis`` call (spied),
and objective equality across the cold-capture / warm-inject runs
(iteration counts are not exposed at the cascade level).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import polar_high  # noqa: E402

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.engine_polars._orchestration import (  # noqa: E402
    _basis_cache_active,
)
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

FIXTURES = TESTS_DIR / "fixtures"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"
# Same single-sub-solve HiGHS scenario the multi-solver + subprocess
# warm-start cascade tests drive.
SCENARIO = "2_day_stochastic_dispatch"


def _make_migrated_db(tmp_path: Path, name: str = "v52.sqlite") -> str:
    """Import ``stochastics.json`` into a fresh sqlite and migrate."""
    db_path = tmp_path / name
    url = json_to_db(STOCHASTICS_JSON, db_path)
    migrate_database(url)
    return url


def _run_cascade(db_url: str, work_folder: Path) -> float:
    """Drive the single ``2day_dispatch`` in-process warm solve; return obj.

    ``warm`` defaults to True (in-process WarmProblem); we do NOT set
    ``FLEXTOOL_SAVE_MEMORY`` so the solve stays in-process.
    """
    steps = run_chain_from_db(
        db_url, SCENARIO, work_folder=work_folder, keep_solutions=True,
    )
    assert steps, "no solve steps produced"
    step = list(steps.values())[-1]
    assert step.obj is not None, "cascade step carried no objective"
    obj = float(step.obj)
    assert obj == obj, "objective is NaN"  # noqa: PLR0124 - NaN check
    return obj


class _SetNamedBasisSpy:
    """Wraps ``WarmProblem.set_named_basis`` to record every call.

    Installed via ``monkeypatch.setattr`` so it auto-reverts.  Delegates to
    the real implementation so the warm-start transfer actually happens.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[object, str]] = []
        self._orig = polar_high.WarmProblem.set_named_basis
        spy = self

        def _wrapper(wp_self, nb, *, policy: str = "exact"):
            spy.calls.append((nb, policy))
            return spy._orig(wp_self, nb, policy=policy)

        monkeypatch.setattr(
            polar_high.WarmProblem, "set_named_basis", _wrapper,
        )


# ---------------------------------------------------------------------------
# Test 1 — Cross-run reuse: run 1 captures a .nbasis, run 2 injects it.
# ---------------------------------------------------------------------------


def test_inprocess_cross_run_capture_then_inject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two independent in-process warm cascades over the SAME structural
    model share one explicit ``FLEXTOOL_BASIS_CACHE_DIR``.

    Run 1 is a fresh build (empty in-process dict + empty on-disk cache):
    it captures the optimal basis to ``<fp>.nbasis`` and does NOT inject
    (nothing to inject).  Run 2 is a distinct ``run_chain_from_db`` — a
    fresh SolverRunner whose in-process dict is empty — so it must load the
    ``<fp>.nbasis`` from disk and inject it via ``WarmProblem`` before its
    first solve.  We assert on the file, the ``set_named_basis`` call
    (spied), and objective equality (warm-start must never change the
    answer).
    """
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    cache_dir = tmp_path / "shared_cache"
    monkeypatch.setenv("FLEXTOOL_BASIS_CACHE_DIR", str(cache_dir))

    db_url = _make_migrated_db(tmp_path)

    # --- Run 1: fresh build → capture, no inject ---
    spy1 = _SetNamedBasisSpy(monkeypatch)
    obj1 = _run_cascade(db_url, tmp_path / "work1")

    assert cache_dir.exists(), "run 1 did not create the shared cache dir"
    nbasis_files = list(cache_dir.glob("*.nbasis"))
    assert nbasis_files, (
        f"run 1 captured no <fp>.nbasis in {cache_dir}; "
        f"contents: {list(cache_dir.iterdir())}"
    )
    assert not spy1.calls, (
        "run 1 (empty cache) must NOT inject a basis — nothing was cached "
        f"yet, but set_named_basis was called {len(spy1.calls)} time(s)"
    )
    # Atomic publish: no leftover tmp debris after a captured run.
    assert not list(cache_dir.glob("*.nbasis.tmp.*")), (
        "leftover *.nbasis.tmp.* debris after run 1"
    )

    fp = nbasis_files[0].stem  # cache key = LP name-set fingerprint

    # --- Run 2: distinct process-like cascade, SAME cache → inject ---
    spy2 = _SetNamedBasisSpy(monkeypatch)
    obj2 = _run_cascade(db_url, tmp_path / "work2")

    assert spy2.calls, (
        "run 2 did not inject the cached basis — WarmProblem."
        "set_named_basis was never called despite a <fp>.nbasis on disk"
    )
    # The injected carrier's fingerprint must match the cached slot.
    injected_nb, injected_policy = spy2.calls[0]
    assert injected_policy == "exact"
    assert injected_nb.fingerprint == fp, (
        f"injected basis fingerprint {injected_nb.fingerprint!r} != cache "
        f"key {fp!r}"
    )
    # Warm-start must never change the answer.
    assert abs(obj1 - obj2) < 1e-6, (
        f"objectives diverged across capture/inject runs: "
        f"capture={obj1} inject={obj2}"
    )
    # Atomic publish holds across both runs.
    assert not list(cache_dir.glob("*.nbasis.tmp.*")), (
        "leftover *.nbasis.tmp.* debris after two runs"
    )


# ---------------------------------------------------------------------------
# Test 2 — OFF is a zero-behaviour-change no-op.
# ---------------------------------------------------------------------------


def test_inprocess_off_creates_no_nbasis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process warm solve WITHOUT ``FLEXTOOL_WARM_START``: the solve
    succeeds, ``set_named_basis`` is never called, and NO ``.nbasis`` file
    is created anywhere under the cache dir (byte-identical off-path)."""
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    monkeypatch.delenv("FLEXTOOL_WARM_START", raising=False)
    cache_dir = tmp_path / "shared_cache"
    monkeypatch.setenv("FLEXTOOL_BASIS_CACHE_DIR", str(cache_dir))

    db_url = _make_migrated_db(tmp_path)
    work = tmp_path / "work"

    spy = _SetNamedBasisSpy(monkeypatch)
    obj = _run_cascade(db_url, work)
    assert obj == obj  # finite  # noqa: PLR0124

    assert not spy.calls, (
        "warm-start OFF must never call set_named_basis; got "
        f"{len(spy.calls)} call(s)"
    )
    # The cache dir is only created lazily by the (gated) capture/inject
    # code, so with the opt-in off it must not exist — and certainly no
    # .nbasis file.
    if cache_dir.exists():
        assert not list(cache_dir.glob("*.nbasis")), (
            "warm-start OFF must not create any <fp>.nbasis file"
        )


# ---------------------------------------------------------------------------
# Test 3 — Benders-safe: the gate predicate disables the arm.
# ---------------------------------------------------------------------------


def test_gate_predicate_benders_and_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_basis_cache_active`` — the single gate guarding every new
    behaviour — is False for Benders and for non-HiGHS solvers, and False
    whenever the opt-in is off.

    A full Benders cascade is NOT built here because it cannot reach the
    warm-start code at all: ``_orchestration.run`` returns early through
    ``_run_benders_solve`` when ``decomposition_for(...) == "benders"``,
    BEFORE the warm branch that hosts the capture/inject blocks.  The
    ``decomposition != "benders"`` term in this predicate is the explicit
    belt-and-suspenders guard on top of that early return, so proving the
    predicate is False for Benders proves the arm is disabled for it.
    """
    # Opt-in on, HiGHS, non-Benders → the only True combination.
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    assert _basis_cache_active(None, "highs") is True
    assert _basis_cache_active("monolithic", "highs") is True

    # Benders disables the arm even with the opt-in on and HiGHS solver.
    assert _basis_cache_active("benders", "highs") is False

    # Non-HiGHS solvers disable the arm (no setBasis/getBasis handle).
    assert _basis_cache_active(None, "gurobi") is False
    assert _basis_cache_active("monolithic", "cplex") is False

    # Opt-in off disables everything regardless of solver / decomposition.
    monkeypatch.delenv("FLEXTOOL_WARM_START", raising=False)
    assert _basis_cache_active(None, "highs") is False
    assert _basis_cache_active("monolithic", "highs") is False


# ---------------------------------------------------------------------------
# Test 4 — Stale / garbage .nbasis: load failure falls back to cold.
# ---------------------------------------------------------------------------


def test_inprocess_garbage_nbasis_falls_back_to_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed ``<fp>.nbasis`` pre-placed in the cache dir must not
    crash the solve: the inject block's ``json.loads`` raises, is caught,
    and the solve proceeds cold to the correct optimum.  The corrupt file
    is then overwritten by a fresh, valid capture."""
    monkeypatch.delenv("FLEXTOOL_SAVE_MEMORY", raising=False)
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    cache_dir = tmp_path / "shared_cache"
    monkeypatch.setenv("FLEXTOOL_BASIS_CACHE_DIR", str(cache_dir))

    db_url = _make_migrated_db(tmp_path)

    # Run 1 (clean) to learn the true fingerprint + a valid objective.
    obj_ref = _run_cascade(db_url, tmp_path / "work0")
    nbasis_files = list(cache_dir.glob("*.nbasis"))
    assert nbasis_files, "reference run captured no <fp>.nbasis"
    fp_path = nbasis_files[0]

    # Corrupt the cached basis in place (invalid JSON).
    fp_path.write_text("{ this is not valid json ]]]")

    # Run 2 — fresh SolverRunner (empty in-process dict) forces a disk
    # load of the now-corrupt file → json.loads raises → caught → cold.
    obj = _run_cascade(db_url, tmp_path / "work1")
    assert abs(obj - obj_ref) < 1e-6, (
        f"garbage-.nbasis run diverged from the clean optimum: "
        f"clean={obj_ref} garbage={obj}"
    )
    # The cold solve still captured a fresh, valid basis, overwriting the
    # corrupt file — and no tmp debris remains.
    assert not list(cache_dir.glob("*.nbasis.tmp.*")), (
        "leftover *.nbasis.tmp.* debris after the garbage-recovery run"
    )
