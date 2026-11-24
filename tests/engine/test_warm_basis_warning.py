"""The hidden warm-start lines are coded (WP-B of 4.6.1).

``run_orchestration`` sets the cascade's ``state.logger`` to ERROR, so its warnings never showed.  The warm-basis
inject / capture failures are now :data:`W-0550` (a warning, once per complete solve and step), and the three
run-mode lines are notes once per run: save-memory mode (W-0551), a solver without warm start (W-0552) and
``warm=False`` (W-0553).  A normal warm run prints none of them.
"""
from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from flextool import run_warnings  # noqa: E402
from flextool.engine import run_chain_from_db  # noqa: E402

pytestmark = pytest.mark.solver

# One small solve; and a rolling solve of 7 rolls, each with a realized-dispatch sibling (the rolls fold into
# their complete solve for the warnings).
SMALL = "2_day_stochastic_dispatch"
ROLLING = "1_week_rolling_wind"
CODES = ("W-0550", "W-0551", "W-0552", "W-0553")


def _run(db_url: str, work: Path, caplog: pytest.LogCaptureFixture, *, scenario: str = SMALL,
         warm: bool = True) -> list[logging.LogRecord]:
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=run_warnings.LOGGER_NAME):
        steps = run_chain_from_db(db_url, scenario, work_folder=work, warm=warm)
    assert steps, "no solve steps"
    return [r for r in caplog.records if getattr(r, run_warnings.CODE_ATTR, None) in CODES]


def _env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key in ("FLEXTOOL_SAVE_MEMORY", "FLEXTOOL_WARM_START", "FLEXTOOL_BASIS_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_a_normal_warm_run_prints_none(stochastic_db_url, tmp_path, monkeypatch, caplog):
    _env(monkeypatch, FLEXTOOL_WARM_START="1")
    assert _run(stochastic_db_url, tmp_path / "work", caplog) == []
    assert (tmp_path / "work" / "basis_cache").exists(), "the warm-basis cache did not run"
    assert not any(code in run_warnings.tally() for code in CODES)


def _lines(records: list[logging.LogRecord], code: str) -> list[str]:
    """The console lines of *code* (the debug-level details are not lines)."""
    return [r.getMessage() for r in records
            if getattr(r, run_warnings.CODE_ATTR, None) == code and r.levelno >= logging.INFO]


def test_a_failed_basis_load_is_one_warning_per_solve(stochastic_db_url, tmp_path, monkeypatch, caplog):
    from polar_high import Problem

    def broken(self):
        raise RuntimeError("basis cache broken (test)")

    _env(monkeypatch, FLEXTOOL_WARM_START="1")
    monkeypatch.setattr(Problem, "basis_name_fingerprint", broken)
    records = _run(stochastic_db_url, tmp_path / "work", caplog, scenario=ROLLING)
    lines = _lines(records, "W-0550")
    # One line per complete solve, although all 7 sub-solves failed (since 4.7.0 each roll of the rolling
    # within_solve solve realizes its own jump: one complete solve, no realization solve).
    assert lines == ["Warning: warm-start basis not loaded for solve '1week_rolling'; solved from scratch [W-0550]"], lines
    assert "basis cache broken (test)" in caplog.text            # the error goes to the debug log
    assert run_warnings.tally()["W-0550"] == len(lines)
    assert not [c for c in ("W-0551", "W-0552", "W-0553") if _lines(records, c)]


def test_a_failed_basis_save_is_one_warning(stochastic_db_url, tmp_path, monkeypatch, caplog):
    from polar_high import Solution

    def broken(self):
        raise RuntimeError("cannot read the basis (test)")

    _env(monkeypatch, FLEXTOOL_WARM_START="1")
    monkeypatch.setattr(Solution, "get_named_basis", broken)
    records = _run(stochastic_db_url, tmp_path / "work", caplog)
    assert _lines(records, "W-0550") == [
        "Warning: warm-start basis not saved for solve '2day_dispatch'; a later solve starts cold [W-0550]"]
    assert "cannot read the basis (test)" in caplog.text
    assert not [c for c in ("W-0551", "W-0552", "W-0553") if _lines(records, c)]


def test_save_memory_mode_is_one_note(stochastic_db_url, tmp_path, monkeypatch, caplog):
    _env(monkeypatch, FLEXTOOL_SAVE_MEMORY="1")
    records = _run(stochastic_db_url, tmp_path / "work", caplog)
    assert _lines(records, "W-0551") == [
        "Note: save-memory mode: every solve is rebuilt and solved by HiGHS in a separate process [W-0551]"]
    assert [r.getMessage() for r in records if getattr(r, run_warnings.CODE_ATTR) != "W-0551"] == []


def test_warm_false_is_one_note(stochastic_db_url, tmp_path, monkeypatch, caplog):
    _env(monkeypatch)
    records = _run(stochastic_db_url, tmp_path / "work", caplog, warm=False)
    assert _lines(records, "W-0553") == [
        "Note: warm reuse is off (warm=False); HiGHS solves run in a separate process [W-0553]"]
    assert [r.getMessage() for r in records if getattr(r, run_warnings.CODE_ATTR) != "W-0553"] == []


def test_a_solver_without_warm_start_is_one_note(stochastic_db_url, tmp_path, monkeypatch, caplog):
    """The cascade sees a commercial solver name; the solve itself still goes to HiGHS (no licence needed)."""
    from flextool.engine.input._solve_config import SolveConfig
    from flextool.engine.solve import _solver_dispatch

    real_config_for = SolveConfig.solver_config_for
    real_run_one_solve = _solver_dispatch.run_one_solve
    highs = {}

    def config_for(self, solve_name):
        config = real_config_for(self, solve_name)
        highs[solve_name] = config
        return dataclasses.replace(config, name="cplex")

    def run_one_solve(pb, config, **kwargs):
        return real_run_one_solve(pb, highs[kwargs["solve_name"]], **kwargs)

    _env(monkeypatch)
    monkeypatch.setattr(SolveConfig, "solver_config_for", config_for)
    monkeypatch.setattr(_solver_dispatch, "run_one_solve", run_one_solve)
    records = _run(stochastic_db_url, tmp_path / "work", caplog)
    assert _lines(records, "W-0552") == [
        "Note: warm start is not available for solver 'cplex'; every solve is built from scratch [W-0552]"]
