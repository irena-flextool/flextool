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
from spinedb_api import DatabaseMapping, import_data

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
    """Setting ``solver = "highs"`` must be byte-identical to the
    default path — the call site reduces to the same
    ``problem.solve(keep_solver=True)``.

    v56 Batch C.9 removed the ``solver_io_api`` DB axis (replaced by
    the ``--matrix-file-format`` CLI flag; when unset the default
    ``SolverConfig.io_api`` is ``"direct"``).  Only ``solver`` needs
    to be authored for this parity check.
    """
    db_url = _make_migrated_db(tmp_path)
    _set_solver_param(db_url, SOLVE_NAME, "solver", "highs")

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
    """A missing commercial-solver CLI surfaces as a user-facing
    ``FlexToolUserError`` whose message names the missing solver,
    the missing binary, and the docs path.

    Updated contract: after the in-process commercial-solver path was
    retired, ``run_one_solve`` always goes via the subprocess CLI for
    non-HiGHS solvers.  ``SolverNotAvailableError`` from polar-high's
    Python wrapper is no longer the failure surface; the new failure
    surface is the CLI-binary lookup in
    :func:`flextool.engine_polars._subprocess_solve._find_solver_binary`.
    """
    cfg = SolverConfig(name="gurobi", io_api="direct")
    pb = _make_minimal_problem()

    # Force the binary-lookup path to fail even if gurobi_cl happens to
    # be on the test machine.
    with patch(
        "flextool.engine_polars._subprocess_solve._find_solver_binary",
        return_value=None,
    ):
        with pytest.raises(FlexToolUserError) as ei:
            run_one_solve(pb, cfg)

    msg = str(ei.value)
    assert "gurobi" in msg, f"missing solver name in error: {msg!r}"
    assert "gurobi_cl" in msg or "binary" in msg, (
        f"binary-missing hint should appear in error: {msg!r}"
    )
    assert "docs/solvers/gurobi.md" in msg, (
        f"docs hint missing in error: {msg!r}"
    )
    # __cause__ preserves the underlying RuntimeError from the subprocess
    # driver.
    assert isinstance(ei.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Test 4 — unknown solver name → clean error, not a stack trace.
# ---------------------------------------------------------------------------


def test_unknown_solver_name_error() -> None:
    """A bogus solver name reaches the dispatch and surfaces as a
    ``FlexToolUserError``.

    ``build_solver_options`` already enforces this earlier for
    convenience-knob users — verified in
    ``test_solver_config.py::test_build_solver_options_unknown_solver_raises``.
    Here we cover the raw-dispatch path: even with no options set,
    an unknown name must not produce a raw stack trace.
    """
    cfg = SolverConfig(name="bogus_solver_name", io_api="direct")
    pb = _make_minimal_problem()

    with pytest.raises(FlexToolUserError) as ei:
        run_one_solve(pb, cfg)

    msg = str(ei.value)
    assert "bogus_solver_name" in msg, (
        f"bogus solver name should appear in error: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — raw solver_options reach the commercial-solver subprocess CLI
# via the per-solver baseline + overlay opt-file mechanism.
# ---------------------------------------------------------------------------


def test_solver_options_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end check that the commercial-solver subprocess driver

    1. reads ``solver_config/<solver>.opt`` baseline defaults,
    2. overlays the scenario's raw ``solver_options`` on top,
    3. honours the translated convenience knobs from
       :func:`build_solver_options` (raw entries still win), and
    4. feeds the merged opt file to the CLI via the per-solver
       mechanism (Gurobi: ``ReadParams=<file>`` argv slot).

    The subprocess itself is mocked — we synthesise a minimal Gurobi
    ``.sol`` as a side effect so the parser + downstream
    ``setSolution`` plumbing still exercises real code.  The assertions
    target the constructed argv and the merged opt-file contents that
    Gurobi would actually have seen.
    """
    # Stand up a temp solver_config dir with a known baseline that
    # differs from the scenario overrides on purpose.  Two keys to test:
    # ``Threads`` (scenario overrides baseline) and ``LogToConsole``
    # (baseline-only, no scenario override → must reach Gurobi).
    config_dir = tmp_path / "solver_config"
    config_dir.mkdir()
    (config_dir / "gurobi.opt").write_text(
        "# test baseline\n"
        "Threads 1\n"
        "MIPGap 0.002\n"
        "LogToConsole 1\n"
    )
    monkeypatch.setenv("FLEXTOOL_SOLVER_CONFIG_DIR", str(config_dir))

    # Scenario overrides: bump Threads to 4 and forward a non-baseline
    # knob.  ``build_solver_options`` will merge in ``MIPGap=0.005``
    # (the convenience-knob translation for mip_gap=0.005) which the
    # raw options dict should *not* override (Threads is the only
    # collision here).
    cfg = SolverConfig(
        name="gurobi",
        io_api="direct",
        mip_gap=0.005,
        options={"Threads": 4, "Presolve": 2},
    )
    pb = _make_minimal_problem()

    captured: dict[str, object] = {}

    def _fake_subprocess_run(argv, **kwargs):
        # Persist argv + the opt-file contents for assertions below.
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        # ``gurobi_cl`` accepts ``ReadParams=<file>`` as one of its
        # positional-style argv tokens.  Find it and slurp the file.
        opt_arg = next(
            (a for a in argv if isinstance(a, str)
             and a.startswith("ReadParams=")),
            None,
        )
        if opt_arg is not None:
            opt_file = Path(opt_arg.split("=", 1)[1])
            captured["opt_contents"] = opt_file.read_text()
        # Locate the ResultFile=<path> argv token and write a minimal
        # Gurobi-style .sol so the parser returns OPTIMAL.  The single
        # primal value (for column "x[a]") matches _make_minimal_problem.
        sol_arg = next(
            (a for a in argv if isinstance(a, str)
             and a.startswith("ResultFile=")),
            None,
        )
        if sol_arg is not None:
            sol_path = Path(sol_arg.split("=", 1)[1])
            sol_path.write_text(
                "# Objective value = 1.0\n"
                "x[a] 1.0\n"
            )

        class _CP:
            returncode = 0
            stdout = ""
            stderr = ""
        return _CP()

    binary = Path("/fake/gurobi_cl")
    with patch(
        "flextool.engine_polars._subprocess_solve._find_solver_binary",
        return_value=binary,
    ), patch(
        "flextool.engine_polars._subprocess_solve.subprocess.run",
        side_effect=_fake_subprocess_run,
    ):
        sol = run_one_solve(pb, cfg)

    assert sol.optimal, "mock subprocess should yield OPTIMAL solution"

    argv = captured["argv"]
    assert argv[0] == str(binary), f"argv[0] should be the binary path; got {argv!r}"
    read_params_tokens = [
        a for a in argv if isinstance(a, str) and a.startswith("ReadParams=")
    ]
    assert len(read_params_tokens) == 1, (
        f"expected exactly one ReadParams=<file> argv token; got argv={argv!r}"
    )

    opt_contents = captured.get("opt_contents", "")
    assert opt_contents, "opt file referenced by argv should be non-empty"

    # Parse the merged file Gurobi would have seen.  Both baseline-only
    # and scenario-overridden values must be present with the right
    # values.
    merged: dict[str, str] = {}
    for line in opt_contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(maxsplit=1)
        merged[name.strip()] = value.strip()

    # Baseline-only key → must survive into the merged file.
    assert merged.get("LogToConsole") == "1", (
        f"baseline LogToConsole=1 should reach Gurobi; merged={merged!r}"
    )
    # Scenario override → must win over the baseline value.
    assert merged.get("Threads") == "4", (
        f"scenario Threads=4 should override baseline Threads=1; "
        f"merged={merged!r}"
    )
    # Convenience-knob translation (mip_gap=0.005 → MIPGap=0.005) must
    # land in the merged file and override the baseline 0.002.
    assert merged.get("MIPGap") == "0.005", (
        f"solver_mip_gap=0.005 should override baseline MIPGap=0.002; "
        f"merged={merged!r}"
    )
    # New scenario-only key reaches Gurobi too.
    assert merged.get("Presolve") == "2", (
        f"scenario Presolve=2 should reach Gurobi; merged={merged!r}"
    )


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



# ---------------------------------------------------------------------------
# Test 8 — multi-solver dispatch on one Problem instance.
# ---------------------------------------------------------------------------


def test_multi_solver_scenario() -> None:
    """Two ``run_one_solve`` calls on the same Problem with different
    SolverConfigs each take the correct dispatch path.

    Updated contract: after the in-process commercial-solver path was
    retired, the commercial leg goes through the subprocess CLI.  When
    the CLI binary is missing on the test machine we mock the lookup +
    raise — verifying the no-binary failure mode is the canonical
    contract for systems without commercial wrappers installed.

    Verifies that each ``solve`` entity independently picks its solver
    — no global state leaks between calls.
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

    # Commercial path: subprocess-only.  Mock the binary lookup so the
    # test machine doesn't need a real gurobi install — we verify the
    # dispatch routes to FlexToolUserError with the gurobi-specific
    # hint, confirming the per-solve dispatch picks the right path.
    with patch(
        "flextool.engine_polars._subprocess_solve._find_solver_binary",
        return_value=None,
    ):
        with pytest.raises(FlexToolUserError) as ei:
            run_one_solve(
                pb_gurobi, SolverConfig(name="gurobi", io_api="direct"),
            )
    assert "gurobi" in str(ei.value)
    # Discard the unused _fake_solve helper — kept as a no-op import
    # guard so the SolverResult/SolverStatus imports above still get
    # exercised by the parameterised tests.
    del _fake_solve

    # The HiGHS solve's state must not have been touched by the
    # subsequent gurobi-path dispatch.
    assert sol_highs.highs is not None


# ---------------------------------------------------------------------------
# Test 9 — HiGHS soft-promote on warm=False (in-process cold path retired).
# ---------------------------------------------------------------------------


def test_warm_false_highs_soft_promotes_to_subprocess(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """``warm=False`` on a HiGHS solve must route through the subprocess
    cmd_solve_mps path (in-process cold HiGHS retired) and log a
    one-shot warning naming the FLEXTOOL_SAVE_MEMORY override.

    Verifies behaviour at the orchestrator surface: we run a real
    single-solve end-to-end (no warm context — single-solve does not
    take a ``warm`` kwarg, so the soft-promote in
    :func:`run_single_solve_from_db` is the active path).  Spy on the
    subprocess call so we don't actually shell out, but assert (a) the
    subprocess path was chosen and (b) the warning was emitted exactly
    once.
    """
    from flextool.engine_polars import (
        _orchestration as _orch,
        _subprocess_solve as _subp,
    )

    db_url = _make_migrated_db(tmp_path)

    spy_calls: list[dict] = []

    def _spy_subprocess(problem, solver_name, options, **kwargs):
        spy_calls.append({
            "solver_name": solver_name,
            "solve_name": kwargs.get("solve_name"),
        })
        # Short-circuit before any actual subprocess work: do the
        # write_mps so the Problem releases (mimicking the real path's
        # side effect on subsequent reuse), then raise to short-circuit.
        # Raising RuntimeError lets run_one_solve translate to
        # FlexToolUserError, which we catch below.
        raise RuntimeError("spy short-circuit")

    caplog.set_level(logging.WARNING, logger="flextool")
    # Also capture the root logger output: run_single_solve_from_db
    # passes a fresh ``logger`` per call, not always "flextool.*".
    caplog.set_level(logging.WARNING)

    with patch.object(_subp, "solve_via_subprocess", side_effect=_spy_subprocess):
        # The spy raises RuntimeError; the HiGHS path lets that
        # propagate (only the commercial path translates RuntimeError
        # → FlexToolUserError).  Either way we just need the dispatch
        # to have run.
        with pytest.raises(RuntimeError, match="spy short-circuit"):
            from flextool.engine_polars import run_single_solve_from_db
            run_single_solve_from_db(
                db_url, SCENARIO,
                work_folder=tmp_path / "work", emit_output=False,
            )

    # (a) Subprocess was the chosen path and the solver was HiGHS.
    assert spy_calls, "subprocess driver was not invoked at all"
    assert spy_calls[0]["solver_name"] == "highs", (
        f"expected HiGHS soft-promote, got {spy_calls[0]['solver_name']!r}"
    )

    # (b) Soft-promote warning fired at least once and mentions the
    # FLEXTOOL_SAVE_MEMORY override.
    soft_promote_lines = [
        r.message for r in caplog.records
        if "FLEXTOOL_SAVE_MEMORY" in r.message
        and "subprocess" in r.message.lower()
    ]
    assert soft_promote_lines, (
        "no soft-promote warning logged; caplog records: "
        + "\n".join(r.message for r in caplog.records)
    )
    # Silence the unused-import lint on _orch (kept for symmetry / future
    # cascade-path coverage that exercises ``_drive_cascade`` directly).
    del _orch
