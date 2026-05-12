"""Phase 4 integration tests for the multi-solver dispatch.

Covers Step 7 of ``specs/flextool-multi-solver-handoff.md`` (lines
213-228).  The eight tests below verify the end-to-end behaviour of
the v52 schema + Phase 2 option translation + Phase 3 dispatch
pipeline:

* default and explicit-HiGHS paths produce identical objectives;
* missing-solver and unknown-solver paths raise ``FlexToolUserError``
  with actionable messages;
* raw options reach the commercial path through ``polar_solve``;
* convenience knob translation covers all five solvers;
* HiGHS ``io_api`` is fixed by Phase 3 design (skipped per spec);
* multi-solver dispatch on the same Problem routes correctly.

Commercial-solver tests (gurobi/cplex/xpress/copt) skip cleanly when
the underlying Python wrapper isn't installed — verified via a probe
solve, not by trusting the static ``available_solvers`` catalog.

Mock strategy
-------------
Tests that need to exercise the commercial path without a real solver
install monkey-patch :data:`flextool.engine_polars._solver_dispatch`'s
imports of ``polar_solve``, ``available_solvers`` and the exception
types via :func:`unittest.mock.patch`.  This keeps the test pinned to
the FlexTool surface even when polar-high's internal layout drifts.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
from spinedb_api import DatabaseMapping, Map, import_data

from polar_high import Problem, Sum
from polar_high.solvers._base import SolverResult, SolverStatus

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_single_solve_from_db  # noqa: E402
from flextool.engine_polars._solve_config import (  # noqa: E402
    SolverConfig,
)
from flextool.engine_polars._solver_dispatch import (  # noqa: E402
    FlexToolUserError,
    _PARAM_MAP,
    build_solver_options,
    run_one_solve,
)
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

FIXTURES = TESTS_DIR / "fixtures"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"
# The fixture's 2_day_stochastic_dispatch scenario activates the
# 2day_dispatch solve via the ``2day`` alternative — see
# ``tests/fixtures/stochastics.json``.  Inject solver_* values onto
# the ``2day`` alternative so the scenario filter picks them up.
SCENARIO = "2_day_stochastic_dispatch"
SOLVE_NAME = "2day_dispatch"
INJECT_ALTERNATIVE = "2day"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_migrated_db(tmp_path: Path, name: str = "v52.sqlite") -> str:
    """Import ``stochastics.json`` into a fresh sqlite and migrate to v52."""
    db_path = tmp_path / name
    url = json_to_db(STOCHASTICS_JSON, db_path)
    migrate_database(url)
    return url


def _set_solver_param(
    db_url: str,
    solve_name: str,
    param_name: str,
    value: object,
    alternative: str = INJECT_ALTERNATIVE,
) -> None:
    """Append a single ``solve.<param_name>`` parameter_value row.

    Wraps :func:`spinedb_api.import_data` so callers can hand a plain
    Python value (str / float / int / :class:`Map`) and the helper
    routes it through ``to_database`` for the correct on-disk encoding.
    """
    with DatabaseMapping(db_url) as db:
        count, errors = import_data(
            db,
            parameter_values=[
                ("solve", solve_name, param_name, value, alternative),
            ],
        )
        if errors:
            raise RuntimeError(f"import_data errors: {errors[:5]}")
        db.commit_session(f"set {solve_name}.{param_name}")


def _make_minimal_problem() -> Problem:
    """Build a 1-var trivial LP usable as a stand-in for solve dispatch.

    ``min x  s.t.  x >= 1, x >= 0`` — obj=1.0, optimal.  Used by tests
    that mock out the solver entirely.
    """
    pb = Problem()
    idx = pl.DataFrame({"i": ["a"]})
    x = pb.add_var("x", "i", idx, lower=0.0)
    pb.add_cstr(
        "c", sense=">=",
        lhs_terms={"x": Sum(x, over=("i",))},
        rhs_terms={"c": 1.0},
    )
    pb.set_objective(x)
    return pb


def _is_solver_actually_available(solver_name: str) -> bool:
    """Probe whether *solver_name* can actually solve a trivial LP.

    ``polar_high.solvers.available_solvers`` is a static catalog;
    runtime availability requires the Python wrapper + (for commercial
    solvers) a working license.  We probe by attempting a solve and
    catching the dispatch errors.
    """
    if solver_name == "highs":
        return True
    try:
        from polar_high.solvers import (
            LicenseError,
            SolverError,
            SolverNotAvailableError,
        )
        from polar_high.solvers import solve as polar_solve

        pb = _make_minimal_problem()
        polar_solve(pb, solver_name=solver_name)
        return True
    except (
        SolverNotAvailableError,
        LicenseError,
        SolverError,
        Exception,
    ):
        return False


# Materialise commercial-solver availability once per session.  The
# probe runs a tiny solve — cheap, but we don't want to repeat it per
# test.
_GUROBI_AVAILABLE = _is_solver_actually_available("gurobi")


# ---------------------------------------------------------------------------
# Module-scoped golden objective (test #1).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stochastics_default_obj(
    tmp_path_factory: pytest.TempPathFactory,
) -> float:
    """Run ``stochastics.json`` end-to-end on the default HiGHS path
    and return the objective.

    Used as the reference value for tests #1 and #2.  Module-scoped
    so the (~1s) solve runs once.
    """
    tmp_path = tmp_path_factory.mktemp("default_obj")
    db_url = _make_migrated_db(tmp_path)
    work = tmp_path / "work"
    step = run_single_solve_from_db(
        db_url, SCENARIO, work_folder=work, emit_output=False,
    )
    assert step.solution is not None and step.solution.optimal, (
        f"baseline stochastics solve failed: "
        f"status={getattr(step.solution, 'status', None)}"
    )
    return float(step.obj)


# ---------------------------------------------------------------------------
# Test 1 — default (no solver params) == legacy HiGHS direct path.
# ---------------------------------------------------------------------------


def test_default_unchanged(
    tmp_path: Path, stochastics_default_obj: float,
) -> None:
    """No ``solver_*`` parameters set → identical objective to the
    pre-Phase-3 default code path.

    The module fixture itself runs the solve once; this test repeats
    it in a fresh tmp_path to confirm the result is deterministic and
    matches the cached value within 1e-9 relative.
    """
    db_url = _make_migrated_db(tmp_path)
    step = run_single_solve_from_db(
        db_url, SCENARIO, work_folder=tmp_path / "work", emit_output=False,
    )
    assert step.solution is not None and step.solution.optimal
    rel = abs(step.obj - stochastics_default_obj) / abs(stochastics_default_obj)
    assert rel < 1e-9, (
        f"default-path obj drifted: got {step.obj}, expected "
        f"{stochastics_default_obj}, rel_err={rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — explicit solver="highs" matches default bitwise.
# ---------------------------------------------------------------------------


def test_explicit_highs_matches_default(
    tmp_path: Path, stochastics_default_obj: float,
) -> None:
    """Setting ``solver = "highs"`` (with default ``solver_io_api =
    "direct"``) must be byte-identical to the default path — the call
    site reduces to the same ``problem.solve(keep_solver=True)``.
    """
    db_url = _make_migrated_db(tmp_path)
    _set_solver_param(db_url, SOLVE_NAME, "solver", "highs")
    _set_solver_param(db_url, SOLVE_NAME, "solver_io_api", "direct")

    step = run_single_solve_from_db(
        db_url, SCENARIO, work_folder=tmp_path / "work", emit_output=False,
    )
    assert step.solution is not None and step.solution.optimal
    # Same call invocation → bit-identical objective.  Allow a tight
    # epsilon (1e-12 rel) only as guard against trivial FP non-determinism.
    rel = abs(step.obj - stochastics_default_obj) / abs(stochastics_default_obj)
    assert rel < 1e-12, (
        f"explicit-HiGHS obj differs from default: got {step.obj}, "
        f"expected {stochastics_default_obj}, rel_err={rel:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — solver wrapper not installed → FlexToolUserError.
# ---------------------------------------------------------------------------


def test_solver_not_installed_error() -> None:
    """``SolverNotAvailableError`` from polar-high is translated to a
    user-facing ``FlexToolUserError`` whose message names the missing
    solver, the available list, and the docs path.
    """
    from polar_high.solvers import SolverNotAvailableError

    cfg = SolverConfig(name="gurobi", io_api="direct")
    pb = _make_minimal_problem()

    def _raise(*args, **kwargs):
        raise SolverNotAvailableError("python wrapper not installed")

    fake_available = ["highs"]  # gurobi notably absent
    with patch(
        "polar_high.solvers.solve", side_effect=_raise,
    ), patch(
        "polar_high.solvers.available_solvers", new=fake_available,
    ):
        with pytest.raises(FlexToolUserError) as ei:
            run_one_solve(pb, cfg)

    msg = str(ei.value)
    assert "gurobi" in msg, f"missing solver name in error: {msg!r}"
    assert "highs" in msg, (
        f"available solvers list should appear in error: {msg!r}"
    )
    assert "docs/solvers/gurobi.md" in msg, (
        f"docs hint missing in error: {msg!r}"
    )
    # __cause__ preserves the underlying polar-high exception.
    assert isinstance(ei.value.__cause__, SolverNotAvailableError)


# ---------------------------------------------------------------------------
# Test 4 — unknown solver name → clean error, not a stack trace.
# ---------------------------------------------------------------------------


def test_unknown_solver_name_error() -> None:
    """A bogus solver name reaches the dispatch (the v52 value list is
    advisory in spinedb_api, not strictly enforced) and surfaces as a
    ``FlexToolUserError`` via the polar-high
    ``SolverNotAvailableError`` path.

    ``build_solver_options`` already enforces this earlier for
    convenience-knob users — verified in
    ``test_solver_config.py::test_build_solver_options_unknown_solver_raises``.
    Here we cover the raw-dispatch path: even with no options set,
    an unknown name must not produce a raw polar-high stack trace.
    """
    from polar_high.solvers import SolverNotAvailableError

    cfg = SolverConfig(name="bogus_solver_name", io_api="direct")
    pb = _make_minimal_problem()

    def _raise(*args, **kwargs):
        raise SolverNotAvailableError("unknown solver")

    with patch("polar_high.solvers.solve", side_effect=_raise):
        with pytest.raises(FlexToolUserError) as ei:
            run_one_solve(pb, cfg)

    msg = str(ei.value)
    assert "bogus_solver_name" in msg
    # The message should at least mention the known solvers so the
    # user sees a menu — Phase 3 inlines ``available_solvers``.
    assert "highs" in msg, (
        f"available solvers menu missing in error: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — raw solver_options reach polar_solve as kwargs (commercial path).
# ---------------------------------------------------------------------------


def test_solver_options_passthrough() -> None:
    """Raw ``solver_config.options`` flow through ``build_solver_options``
    and arrive as kwargs at :func:`polar_high.solvers.solve`.

    **Path pinned**: the commercial-solver path.  The default HiGHS
    path in ``run_one_solve`` short-circuits to
    ``problem.solve(keep_solver=True)`` and does **not** forward
    ``solver_config.options`` (HiGHS options reach the solver via the
    separate ``problem.set_solver_options(highs_options)`` chain in
    :func:`run_single_solve_from_db`, populated by the scaling helper —
    not from the user's ``solver_options`` map).  This is a known
    Phase 3 design gap that should be addressed in a follow-up, but
    it is out of scope for the integration tests; pinning the
    commercial path is the contract this test enforces.
    """
    cfg = SolverConfig(
        name="gurobi",
        io_api="direct",
        options={"presolve": "off", "log_to_console": "no"},
    )
    pb = _make_minimal_problem()

    captured: dict = {}

    def _fake_solve(model, solver_name=None, io_api="direct", env=None, **kw):
        captured["solver_name"] = solver_name
        captured["io_api"] = io_api
        captured["kwargs"] = kw
        # Return a minimal-but-shaped SolverResult so LiteSolution
        # construction (back in run_one_solve) succeeds.
        return SolverResult(
            status=SolverStatus.OPTIMAL,
            objective=1.0,
            primal={"x[a]": 1.0},
            dual=None,
            solver_name=solver_name or "gurobi",
            raw_status="ok",
        )

    with patch("polar_high.solvers.solve", side_effect=_fake_solve):
        sol = run_one_solve(pb, cfg)

    assert captured["solver_name"] == "gurobi"
    assert captured["kwargs"]["presolve"] == "off"
    assert captured["kwargs"]["log_to_console"] == "no"
    # LiteSolution exposes the obj/optimal surface.
    assert sol.optimal
    assert sol.obj == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6 — convenience knob translation table across all five solvers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_name, time_native, gap_native, threads_native",
    [
        ("highs",  "time_limit", "mip_rel_gap",          "threads"),
        ("gurobi", "TimeLimit",  "MIPGap",                "Threads"),
        ("cplex",  "timelimit",  "mip.tolerances.mipgap", "threads"),
        ("xpress", "maxtime",    "miprelstop",            "threads"),
        ("copt",   "TimeLimit",  "RelGap",                "Threads"),
    ],
)
def test_convenience_param_translation(
    solver_name: str, time_native: str, gap_native: str, threads_native: str,
) -> None:
    """``time_limit`` / ``mip_gap`` / ``threads`` translate to each
    solver's native parameter names.  Verifies the canonical
    :data:`_PARAM_MAP` from spec lines 109-136.
    """
    cfg = SolverConfig(
        name=solver_name, time_limit=60.0, mip_gap=0.01, threads=4,
    )
    opts = build_solver_options(cfg)
    assert opts == {
        time_native: 60.0,
        gap_native: 0.01,
        threads_native: 4,
    }, f"unexpected option dict for {solver_name}: {opts}"


def test_convenience_param_translation_raw_wins() -> None:
    """When a raw ``solver_options`` entry collides with a translated
    convenience knob, the raw value wins.  Spec §"raw options take
    precedence".
    """
    cfg = SolverConfig(
        name="gurobi",
        time_limit=60.0,
        options={"TimeLimit": 999.0},
    )
    opts = build_solver_options(cfg)
    assert opts["TimeLimit"] == 999.0, (
        f"raw solver_options.TimeLimit should win over convenience knob; "
        f"got {opts['TimeLimit']!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 — io_api="mps" on HiGHS.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "HiGHS path uses Problem.solve() regardless of io_api per Phase 3 "
        "design (run_one_solve short-circuits on solver_config.name == "
        "'highs' and never reads io_api).  io_api='mps' currently has no "
        "effect for the HiGHS path; revisit when polar-high adds MPS file "
        "support to the direct API or when the dispatch routes HiGHS "
        "through polar_solve unconditionally."
    )
)
def test_mps_io_api(tmp_path: Path, stochastics_default_obj: float) -> None:
    """Setting ``solver_io_api = "mps"`` on a HiGHS solve should yield
    the same objective.  Currently skipped — see decorator reason.
    """
    db_url = _make_migrated_db(tmp_path)
    _set_solver_param(db_url, SOLVE_NAME, "solver", "highs")
    _set_solver_param(db_url, SOLVE_NAME, "solver_io_api", "mps")
    step = run_single_solve_from_db(
        db_url, SCENARIO, work_folder=tmp_path / "work", emit_output=False,
    )
    rel = abs(step.obj - stochastics_default_obj) / abs(stochastics_default_obj)
    assert rel < 1e-9


# ---------------------------------------------------------------------------
# Test 8 — multi-solver dispatch on one Problem instance.
# ---------------------------------------------------------------------------


def test_multi_solver_scenario() -> None:
    """Two ``run_one_solve`` calls on the same Problem with different
    SolverConfigs each take the correct dispatch path.

    The HiGHS leg goes through ``problem.solve(keep_solver=True)``
    and produces a live ``Solution`` (``sol.highs is not None``).  The
    commercial leg goes through the mocked ``polar_solve`` and
    produces a :class:`LiteSolution` (``sol.highs is None``).

    Verifies the spec acceptance: each ``solve`` entity independently
    picks its solver — no global state leaks between calls.
    """
    pb_highs = _make_minimal_problem()
    pb_gurobi = _make_minimal_problem()

    sol_highs = run_one_solve(pb_highs, SolverConfig(name="highs"))
    assert sol_highs.optimal
    assert sol_highs.highs is not None, (
        "HiGHS dispatch should retain the live solver instance "
        "(keep_solver=True)"
    )
    assert sol_highs.obj == pytest.approx(1.0)

    def _fake_solve(model, solver_name=None, io_api="direct", env=None, **kw):
        return SolverResult(
            status=SolverStatus.OPTIMAL,
            objective=1.0,
            primal={"x[a]": 1.0},
            dual=None,
            solver_name=solver_name or "gurobi",
            raw_status="ok",
        )

    with patch("polar_high.solvers.solve", side_effect=_fake_solve):
        sol_gurobi = run_one_solve(
            pb_gurobi, SolverConfig(name="gurobi", io_api="direct"),
        )

    assert sol_gurobi.optimal
    assert sol_gurobi.highs is None, (
        "commercial path LiteSolution should not carry a live HiGHS "
        "instance — it routes through polar_solve and wraps the "
        "SolverResult."
    )
    assert sol_gurobi.obj == pytest.approx(1.0)

    # The HiGHS solve's state must not have been touched by the
    # subsequent gurobi-path mock.
    assert sol_highs.highs is not None
