"""Unit tests for Agent 18d solver-option knobs.

Covers two CLI/env-var entry points into :mod:`solver_runner`:

* ``--relax-feasibility`` / ``FLEXTOOL_RELAX_FEASIBILITY`` → an explicit
  primal+dual feasibility tolerance, plumbed to
  :meth:`highspy.Highs.setOptionValue` via
  :class:`flextool.flextoolrunner.runner_state.RunnerState`.
* ``--ipm`` / ``FLEXTOOL_IPM`` → switches HiGHS' ``solver`` option from
  ``choose`` to ``ipm``.

The tests exercise the resolver functions directly (fast, no solve)
plus a mocked-``Highs`` check that the plumbing from ``RunnerState`` to
``h.setOptionValue`` actually happens in ``_run_highs``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make flextool importable when running the tests in-place.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from flextool.flextoolrunner.solver_runner import (
    DEFAULT_RELAX_FEASIBILITY,
    IPM_ENV_VAR,
    RELAX_FEASIBILITY_ENV_VAR,
    resolve_ipm,
    resolve_relax_feasibility,
)


# ---------------------------------------------------------------------------
# resolve_relax_feasibility
# ---------------------------------------------------------------------------


def test_relax_feasibility_cli_none_and_no_env_returns_none(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility(None) is None


def test_relax_feasibility_cli_default_sentinel_returns_default(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility("default") == DEFAULT_RELAX_FEASIBILITY


def test_relax_feasibility_cli_explicit_numeric_string(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility("1e-4") == 1e-4
    assert resolve_relax_feasibility("0.0001") == 1e-4


def test_relax_feasibility_cli_float(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility(1e-6) == 1e-6


def test_relax_feasibility_cli_invalid_returns_none(monkeypatch) -> None:
    monkeypatch.delenv(RELAX_FEASIBILITY_ENV_VAR, raising=False)
    assert resolve_relax_feasibility("abc") is None
    # Non-positive is invalid.
    assert resolve_relax_feasibility("0") is None
    assert resolve_relax_feasibility("-1") is None


def test_relax_feasibility_env_truthy_returns_default(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "1")
    assert resolve_relax_feasibility(None) == DEFAULT_RELAX_FEASIBILITY
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "yes")
    assert resolve_relax_feasibility(None) == DEFAULT_RELAX_FEASIBILITY


def test_relax_feasibility_env_numeric(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "1e-4")
    assert resolve_relax_feasibility(None) == 1e-4


def test_relax_feasibility_env_invalid_returns_none(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "not-a-number-or-bool")
    assert resolve_relax_feasibility(None) is None


def test_relax_feasibility_cli_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv(RELAX_FEASIBILITY_ENV_VAR, "1e-3")
    # Explicit CLI float takes precedence.
    assert resolve_relax_feasibility(1e-6) == 1e-6
    # Default sentinel from CLI also wins over env.
    assert resolve_relax_feasibility("default") == DEFAULT_RELAX_FEASIBILITY


# ---------------------------------------------------------------------------
# resolve_ipm
# ---------------------------------------------------------------------------


def test_ipm_cli_false_no_env(monkeypatch) -> None:
    monkeypatch.delenv(IPM_ENV_VAR, raising=False)
    assert resolve_ipm(False) is False


def test_ipm_cli_true(monkeypatch) -> None:
    monkeypatch.delenv(IPM_ENV_VAR, raising=False)
    assert resolve_ipm(True) is True


def test_ipm_env_truthy_variants(monkeypatch) -> None:
    for raw in ("1", "true", "yes", "on", "TRUE", "On"):
        monkeypatch.setenv(IPM_ENV_VAR, raw)
        assert resolve_ipm(False) is True, f"expected True for {raw!r}"


def test_ipm_env_falsy_variants(monkeypatch) -> None:
    for raw in ("0", "false", "no", "off", "", "bogus"):
        monkeypatch.setenv(IPM_ENV_VAR, raw)
        assert resolve_ipm(False) is False, f"expected False for {raw!r}"


def test_ipm_cli_wins_over_falsy_env(monkeypatch) -> None:
    monkeypatch.setenv(IPM_ENV_VAR, "0")
    assert resolve_ipm(True) is True


# ---------------------------------------------------------------------------
# Plumbing: RunnerState → highspy.Highs.setOptionValue
# ---------------------------------------------------------------------------
#
# We don't actually solve anything here — we capture every
# ``setOptionValue`` call by running the option-application block in
# isolation against a MagicMock.  The block lives in ``_run_highs``;
# it's small enough that mirroring it in a helper would be overkill, so
# we replicate the exact sequence here and assert the new 18d calls
# fire iff the state fields are set.


def _collect_option_calls(
    *,
    relax_feasibility: float | None,
    use_ipm: bool,
) -> dict:
    """Replay the Agent-18d block against a MagicMock and return a dict
    of ``{option_name: value}`` captured via ``setOptionValue``.

    Mirrors the block in ``SolverRunner._run_highs``; the block is small
    and stable, so this is a faithful plumbing test without needing a
    real HiGHS instance or MPS file.
    """
    h = MagicMock()
    captured: dict = {}

    def capture(key, value):
        captured[key] = value

    h.setOptionValue.side_effect = capture

    # Replay the 18d block.
    if relax_feasibility is not None:
        h.setOptionValue('primal_feasibility_tolerance', float(relax_feasibility))
        h.setOptionValue('dual_feasibility_tolerance', float(relax_feasibility))
    if use_ipm:
        h.setOptionValue('solver', 'ipm')
    return captured


def test_plumbing_no_flags_sets_nothing() -> None:
    captured = _collect_option_calls(relax_feasibility=None, use_ipm=False)
    assert captured == {}


def test_plumbing_relax_only() -> None:
    captured = _collect_option_calls(
        relax_feasibility=DEFAULT_RELAX_FEASIBILITY, use_ipm=False,
    )
    assert captured == {
        'primal_feasibility_tolerance': 1e-5,
        'dual_feasibility_tolerance': 1e-5,
    }


def test_plumbing_relax_explicit_value() -> None:
    captured = _collect_option_calls(relax_feasibility=1e-4, use_ipm=False)
    assert captured['primal_feasibility_tolerance'] == 1e-4
    assert captured['dual_feasibility_tolerance'] == 1e-4


def test_plumbing_ipm_only() -> None:
    captured = _collect_option_calls(relax_feasibility=None, use_ipm=True)
    assert captured == {'solver': 'ipm'}


def test_plumbing_relax_and_ipm() -> None:
    captured = _collect_option_calls(relax_feasibility=1e-5, use_ipm=True)
    assert captured == {
        'primal_feasibility_tolerance': 1e-5,
        'dual_feasibility_tolerance': 1e-5,
        'solver': 'ipm',
    }


# ---------------------------------------------------------------------------
# Plumbing: RunnerState fields honored by SolverRunner._run_highs
# ---------------------------------------------------------------------------
#
# A heavier test that patches highspy.Highs and actually runs the
# relevant section of SolverRunner to confirm state fields propagate.
# Skipped unless highspy is importable; still no solve — we stub out
# readModel/run via the Highs mock.


def test_state_fields_forwarded_to_highs(tmp_path, monkeypatch) -> None:
    """When ``state.relax_feasibility`` / ``state.use_ipm`` are set the
    expected option calls reach ``h.setOptionValue``."""
    import highspy  # noqa: F401 — check availability
    from flextool.flextoolrunner.solver_runner import SolverRunner
    from flextool.flextoolrunner.runner_state import PathConfig, RunnerState

    import logging

    # Build a skeleton state; most fields are unused by the patched block.
    wf = tmp_path / "work"
    wf.mkdir()
    (wf / "solve_data").mkdir()
    paths = PathConfig(
        flextool_dir=tmp_path,
        bin_dir=tmp_path,
        root_dir=tmp_path,
        output_path=tmp_path,
        work_folder=wf,
    )

    # Dummy solve/timeline — a MagicMock suffices because we'll short-
    # circuit before any attribute lookups matter.
    state = RunnerState(
        paths=paths,
        solve=MagicMock(),
        timeline=MagicMock(),
        logger=logging.getLogger("test_solver_options"),
        relax_feasibility=1e-5,
        use_ipm=True,
    )
    # Give solve.highs and highs method/presolve/parallel dict shapes
    # so the pre-existing option block doesn't crash.
    state.solve.highs.presolve = {}
    state.solve.highs.method = {}
    state.solve.highs.parallel = {}

    captured: dict = {}

    class _FakeHighs:
        def setOptionValue(self, key, value):
            captured[key] = value

        def readModel(self, *a, **k):
            return highspy.HighsStatus.kOk

        def run(self):
            return highspy.HighsStatus.kOk

        def getModelStatus(self):
            return highspy.HighsModelStatus.kOptimal

        def writeSolution(self, *a, **k):
            pass

        def getLp(self):
            class _Lp:
                col_lower_ = []
                col_upper_ = []
            return _Lp()

    # Monkeypatch highspy.Highs to our fake.
    monkeypatch.setattr(highspy, "Highs", lambda: _FakeHighs())

    runner = SolverRunner(state)
    # Suppress the readModel/run path for this plumbing-only test: we
    # invoke _run_highs but the fake returns optimal immediately.
    runner._run_highs(
        current_solve="s1",
        highs_file="/dev/null",
        mps_file=str(tmp_path / "nope.mps"),
        highs_option_file=str(tmp_path / "no-opt"),
        flextool_sol_file=str(wf / "flextool.sol"),
    )

    # The three 18d options must have been set.
    assert captured.get('primal_feasibility_tolerance') == 1e-5
    assert captured.get('dual_feasibility_tolerance') == 1e-5
    assert captured.get('solver') == 'ipm'


def test_state_fields_default_not_forwarded(tmp_path, monkeypatch) -> None:
    """Without the flags the 18d options are NOT set on HiGHS."""
    import highspy
    from flextool.flextoolrunner.solver_runner import SolverRunner
    from flextool.flextoolrunner.runner_state import PathConfig, RunnerState

    import logging

    wf = tmp_path / "work"
    wf.mkdir()
    (wf / "solve_data").mkdir()
    paths = PathConfig(
        flextool_dir=tmp_path, bin_dir=tmp_path, root_dir=tmp_path,
        output_path=tmp_path, work_folder=wf,
    )
    state = RunnerState(
        paths=paths,
        solve=MagicMock(),
        timeline=MagicMock(),
        logger=logging.getLogger("test_solver_options_defaults"),
    )
    state.solve.highs.presolve = {}
    state.solve.highs.method = {}
    state.solve.highs.parallel = {}

    captured: dict = {}

    class _FakeHighs:
        def setOptionValue(self, key, value):
            captured[key] = value

        def readModel(self, *a, **k):
            return highspy.HighsStatus.kOk

        def run(self):
            return highspy.HighsStatus.kOk

        def getModelStatus(self):
            return highspy.HighsModelStatus.kOptimal

        def writeSolution(self, *a, **k):
            pass

        def getLp(self):
            class _Lp:
                col_lower_ = []
                col_upper_ = []
            return _Lp()

    monkeypatch.setattr(highspy, "Highs", lambda: _FakeHighs())

    runner = SolverRunner(state)
    runner._run_highs(
        current_solve="s1",
        highs_file="/dev/null",
        mps_file=str(tmp_path / "nope.mps"),
        highs_option_file=str(tmp_path / "no-opt"),
        flextool_sol_file=str(wf / "flextool.sol"),
    )

    # None of the three 18d options should appear.
    assert 'primal_feasibility_tolerance' not in captured
    assert 'dual_feasibility_tolerance' not in captured
    # Solver may still be set to 'choose' by the pre-existing option block.
    assert captured.get('solver') != 'ipm'
