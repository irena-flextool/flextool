"""End-to-end CASCADE tests for the warm-start ``.bas`` basis-cache arm.

The mechanism itself (fingerprint-keyed native ``.bas`` cache, ``--warm-basis``
injection, atomic publish, first-transfer A/B gate) is implemented and
unit-tested at the subprocess level in
``tests/engine_polars/test_subprocess_warm_cache.py`` /
``test_subprocess_warm_gating.py``.  Those drive ``solve_via_subprocess``
directly.

Here we prove the OPT-IN wiring fires through a *real flextool solve
cascade* — ``run_chain_from_db`` → orchestration save-memory gate →
``run_one_solve`` → ``_solve_highs_subprocess`` — and that it is a
zero-behaviour-change no-op when ``FLEXTOOL_WARM_START`` is off.

Env-var opt-in mirrors the CLI (``--warm-start`` sets
``FLEXTOOL_WARM_START='1'`` in ``flextool/cli/cmd_run_flextool.py``); in a
programmatic test we set the env vars directly via ``monkeypatch.setenv``
so they don't leak, exactly as the subprocess-level tests do.

The save-memory subprocess route is opted into with
``FLEXTOOL_SAVE_MEMORY=1`` (read by the orchestration gate at
``_orchestration.py``); under it every sub-solve cold-rebuilds, writes MPS
and dispatches to a subprocess HiGHS, which is where the warm-start arm
lives.  We reuse the exact fixture/DB helper and scenario constant from
``test_solver_integration.py`` so the model is a genuine, migrating build
(CLAUDE.md invariant: tests build the DB from JSON/schema, never a
checked-in ``.sqlite``).

We assert on cache files + gating markers + solve success only — no
iteration-count access at the cascade level (that reduction is already
proven at the subprocess level).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

FIXTURES = TESTS_DIR / "fixtures"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"
# Same scenario the multi-solver cascade integration test drives — a small,
# genuine, migrating model.  See ``test_solver_integration.py``.
SCENARIO = "2_day_stochastic_dispatch"


def _make_migrated_db(tmp_path: Path, name: str = "v52.sqlite") -> str:
    """Import ``stochastics.json`` into a fresh sqlite and migrate to v52.

    Byte-for-byte the helper used by ``test_solver_integration.py`` so the
    cascade builds the identical model.
    """
    db_path = tmp_path / name
    url = json_to_db(STOCHASTICS_JSON, db_path)
    migrate_database(url)
    return url


def _run_cascade(db_url: str, work_folder: Path) -> dict:
    """Drive the single ``2day_dispatch`` solve cascade and return the steps.

    ``keep_solutions=True`` so each step retains ``obj`` for the
    success / objective-parity assertions.
    """
    steps = run_chain_from_db(
        db_url, SCENARIO, work_folder=work_folder, keep_solutions=True,
    )
    assert steps, "no solve steps produced"
    return steps


def _assert_solved(steps: dict) -> float:
    """Assert the cascade produced a real, finite objective; return it."""
    step = list(steps.values())[-1]
    assert step.obj is not None, "cascade step carried no objective"
    obj = float(step.obj)
    assert obj == obj, "objective is NaN"  # noqa: PLR0124 - NaN check
    return obj


# ---------------------------------------------------------------------------
# Test 1 — OFF is a zero-behaviour-change no-op.
# ---------------------------------------------------------------------------


def test_warm_start_off_creates_no_basis_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_memory subprocess path WITHOUT ``FLEXTOOL_WARM_START``: the solve
    succeeds and NO ``basis_cache`` directory is created under the cascade's
    work_folder (zero behaviour change when the opt-in is off)."""
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    monkeypatch.delenv("FLEXTOOL_WARM_START", raising=False)
    monkeypatch.delenv("FLEXTOOL_BASIS_CACHE_DIR", raising=False)

    db_url = _make_migrated_db(tmp_path)
    work = tmp_path / "work"

    steps = _run_cascade(db_url, work)
    _assert_solved(steps)

    assert not (work / "basis_cache").exists(), (
        "warm-start OFF must not create a basis_cache directory"
    )


# ---------------------------------------------------------------------------
# Test 2 — ON, first run captures a basis under the cascade work_folder.
# ---------------------------------------------------------------------------


def test_warm_start_on_first_run_captures_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_memory subprocess path WITH ``FLEXTOOL_WARM_START=1``: the solve
    succeeds and at least one ``<fp>.bas`` file lands in
    ``<work_folder>/basis_cache/`` — proving the opt-in reaches the parent
    warm arm AND that the cache dir is derived from the cascade's
    work_folder (no explicit ``FLEXTOOL_BASIS_CACHE_DIR``)."""
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    monkeypatch.delenv("FLEXTOOL_BASIS_CACHE_DIR", raising=False)

    db_url = _make_migrated_db(tmp_path)
    work = tmp_path / "work"

    steps = _run_cascade(db_url, work)
    _assert_solved(steps)

    cache_dir = work / "basis_cache"
    assert cache_dir.exists(), (
        "warm-start ON did not create <work_folder>/basis_cache — the "
        "cascade work_folder did not reach the subprocess warm arm"
    )
    bas_files = list(cache_dir.glob("*.bas"))
    assert bas_files, (
        f"no <fp>.bas captured in {cache_dir} on first warm-start run; "
        f"contents: {list(cache_dir.iterdir())}"
    )
    # Atomic publish: no leftover tmp debris.
    assert not list(cache_dir.glob("*.bas.tmp.*")), (
        "leftover *.bas.tmp.* debris in the cache dir"
    )


# ---------------------------------------------------------------------------
# Test 3 — ON, second run reuses the cache and the first-transfer A/B gate
# fires (a gating marker appears), across two full cascade runs sharing an
# explicit cache dir.
# ---------------------------------------------------------------------------


def test_warm_start_second_run_reuses_and_ab_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two full ``run_chain_from_db`` cascades over the SAME structural model
    share one explicit ``FLEXTOOL_BASIS_CACHE_DIR``.

    Run 1 (cold) captures ``<fp>.bas``.  Run 2 is a cache HIT with no
    verdict marker yet, so the parent takes the FIRST-TRANSFER branch:
    it injects the cached basis (``--warm-basis``) into the warm main run,
    runs a cold timing probe, and records a verdict marker
    (``<fp>.abtested`` or ``<fp>.nowarm``).  We assert a marker appears —
    proving the warm injection + A/B measurement actually RAN inside the
    cascade, not just that a file sits on disk.  Both runs must succeed at
    the same objective.

    An explicit shared cache dir (not the per-run work_folder) is used so
    the ``.bas`` + markers survive across the two independent cascades and
    the fingerprint keys the same slot in both runs.
    """
    monkeypatch.setenv("FLEXTOOL_SAVE_MEMORY", "1")
    monkeypatch.setenv("FLEXTOOL_WARM_START", "1")
    cache_dir = tmp_path / "shared_cache"
    monkeypatch.setenv("FLEXTOOL_BASIS_CACHE_DIR", str(cache_dir))

    db_url = _make_migrated_db(tmp_path)

    # --- Run 1: cold, populates the shared cache ---
    steps1 = _run_cascade(db_url, tmp_path / "work1")
    obj1 = _assert_solved(steps1)

    assert cache_dir.exists(), "run 1 did not create the shared cache dir"
    bas_after_1 = list(cache_dir.glob("*.bas"))
    assert bas_after_1, (
        f"run 1 captured no <fp>.bas in {cache_dir}; "
        f"contents: {list(cache_dir.iterdir())}"
    )
    # No verdict marker yet — run 1 was a cold miss, not a transfer.
    assert not list(cache_dir.glob("*.abtested")), (
        "run 1 must not record an A/B verdict (no transfer happened)"
    )
    assert not list(cache_dir.glob("*.nowarm")), (
        "run 1 must not record an A/B verdict (no transfer happened)"
    )

    # --- Run 2: identical model, SAME shared cache → hit + first transfer ---
    steps2 = _run_cascade(db_url, tmp_path / "work2")
    obj2 = _assert_solved(steps2)

    markers = (
        list(cache_dir.glob("*.abtested")) + list(cache_dir.glob("*.nowarm"))
    )
    assert markers, (
        "no A/B gating marker (<fp>.abtested / <fp>.nowarm) after run 2 — "
        "the first-transfer warm injection + A/B probe did not fire in the "
        f"cascade; cache contents: {list(cache_dir.iterdir())}"
    )
    # Warm-start must never change the answer.
    assert abs(obj1 - obj2) < 1e-6, (
        f"objectives diverged across cold/warm cascade runs: "
        f"cold={obj1} warm={obj2}"
    )
    # Atomic publish holds across both runs.
    assert not list(cache_dir.glob("*.bas.tmp.*")), (
        "leftover *.bas.tmp.* debris after two runs"
    )
