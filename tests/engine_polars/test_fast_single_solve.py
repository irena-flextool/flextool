"""Δ.25 — focused tests for the surgical fast single-solve path.

Validates that
:func:`flextool.engine_polars.run_single_solve_from_db` produces the
same objective as the slow path
(:func:`flextool.engine_polars.run_chain_from_db`) on the
``work_base`` single-solve fixture.

Scope is intentionally narrow: this is the experimental fast path the
user flagged for ``test_24h_shipping``-style cold-start latency, NOT
production parity coverage.  The full parity sweep lives in
:mod:`test_orchestration_parity`.

Per the Δ.25 design (non-production, raise loudly), additional
fixtures will surface helper coverage gaps as
:class:`flextool.engine_polars.FastLoadError` — those are documented
in the Δ.25 close stanza, not bolted on here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import (
    FastLoadError,
    run_single_solve_from_db,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# work_base — the canonical simple single-solve fixture.
# ---------------------------------------------------------------------------


def test_fast_single_solve_work_base_obj_parity(tmp_path: Path) -> None:
    """``work_base`` solves with the same objective on the fast path
    as on the slow path.

    The slow-path objective on ``work_base`` is ``4780167750`` (see
    ``progress.md`` Δ.16+).  We accept any rel-error < 1e-9 since the
    LP construction is identical from a HiGHS perspective.
    """
    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast"
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="base",
        work_folder=work,
    )

    assert step.solution is not None, "fast path returned no Solution"
    assert step.solution.optimal, (
        f"fast path: HiGHS non-optimal "
        f"(status={getattr(step.solution, 'status', None)})"
    )
    expected_obj = 4780167750.0
    rel_err = abs(step.obj - expected_obj) / expected_obj
    assert rel_err < 1e-9, (
        f"fast path obj={step.obj} differs from expected "
        f"{expected_obj} by rel_err={rel_err:.3e}"
    )

    # Output writer adapter ran — output_raw should exist.
    assert (work / "output_raw").exists(), (
        "expected output_raw/ directory produced by the writer adapter"
    )


def test_fast_single_solve_requires_scenario_name(tmp_path: Path) -> None:
    """Fast path raises when no scenario_name supplied.

    The fast path doesn't auto-pick scenarios — scenario resolution
    is a slow-path-only convenience.  Verify the requirement is loud.
    """
    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast"
    # SpineDbReader requires a scenario string; an empty string is the
    # closest stand-in for "missing".  Either raises some flavour of
    # error inside spinedb_api or our own FastLoadError downstream.
    with pytest.raises((Exception,)):
        run_single_solve_from_db(
            f"sqlite:///{db}",
            scenario_name="",
            work_folder=work,
        )


def test_fast_single_solve_skip_output_emit(tmp_path: Path) -> None:
    """``emit_output=False`` short-circuits the writer adapter.

    Useful for benchmarking the LP-build path in isolation; verify
    no output_raw parquets appear when disabled.
    """
    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_no_emit"
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="base",
        work_folder=work,
        emit_output=False,
    )
    assert step.solution is not None and step.solution.optimal

    output_raw = work / "output_raw"
    if output_raw.exists():
        # Created at workdir-bootstrap time (mkdir -p) but should be
        # empty when emit_output=False.
        contents = list(output_raw.iterdir())
        assert not contents, (
            f"expected empty output_raw/ when emit_output=False; "
            f"got {[p.name for p in contents]}"
        )
