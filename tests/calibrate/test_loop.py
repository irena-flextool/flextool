"""Non-solver tests for the calibration loop's stop logic (early-stop + status).

These drive :func:`flextool.calibrate.run_calibration` end to end but with
every I/O boundary MONKEYPATCHED — the DB weight read, the solve subprocess,
the fail-closed assessor, and the three signal readers — so no real solve is
launched.  The REAL :func:`compute_step` (sizing + over-build guard) runs, so
the tests exercise the actual stall decision: when the guard flags every
remaining shedding node, ``compute_step`` yields ``{}`` and the loop must
STOP early rather than burn the rest of the ``--iterations`` budget on an
identical model.

Three trajectories are pinned, one per ``stop_reason``:

* ``"stalled"``          — the guard flags the last shedding node, so the
  next round would bump nothing → early stop BELOW the budget;
* ``"converged"``        — total unserved falls under the threshold;
* ``"budget_exhausted"`` — keeps making progress (no flag, no convergence)
  until the iteration budget is spent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import flextool.calibrate._loop as loop_mod
from flextool.calibrate._loop import CalibConfig, run_calibration


def _config(tmp_path: Path, *, iterations: int, tightness: float) -> CalibConfig:
    return CalibConfig(
        iterations=iterations,
        slack_threshold_mwh=1.0,
        damping_first=1.0,
        damping_remaining=1.0,
        over_build_tightness=tightness,
        warm_start_cache_dir=tmp_path / "cache",
        work_dir=tmp_path / "work",
        out_root=tmp_path / "out",
        debug=False,
    )


def _install_scripted_solve(monkeypatch, signals):
    """Patch the loop's I/O so each iteration reads the next scripted signal.

    *signals* is a list of ``{"residual": {...}, "curtailment": {...}}`` dicts,
    one per iteration.  ``run_solve`` advances an internal cursor (it is called
    once at the top of each iteration, before the reads), and the readers
    return that iteration's scripted dicts.  ``W`` and the solve verification
    are stubbed constant/success; the slack penalty is an unused zero.
    """
    state = {"i": -1}

    def fake_invest_weight_W(url, scenario):
        return 1000.0

    def fake_write_calib_alt(url, scenario, adders):
        return None

    def fake_run_solve(url, scenario, *, work_dir, out_root, cache_dir):
        state["i"] += 1
        return SimpleNamespace(
            assess_dir=Path(out_root) / f"iter_{state['i']}",
            returncode=0,
            started_at=0.0,
        )

    def fake_assess_solve(assess_dir, *, exit_code, started_at):
        return SimpleNamespace(succeeded=True, reason="")

    def fake_read_residual(assess_dir):
        return dict(signals[state["i"]]["residual"])

    def fake_read_curtailment(assess_dir):
        return dict(signals[state["i"]]["curtailment"])

    def fake_read_penalty(assess_dir):
        return 0.0, {}

    monkeypatch.setattr(loop_mod, "invest_weight_W", fake_invest_weight_W)
    monkeypatch.setattr(loop_mod, "write_calib_alt", fake_write_calib_alt)
    monkeypatch.setattr(loop_mod, "run_solve", fake_run_solve)
    monkeypatch.setattr(loop_mod, "assess_solve", fake_assess_solve)
    monkeypatch.setattr(loop_mod, "read_residual_unserved", fake_read_residual)
    monkeypatch.setattr(loop_mod, "read_curtailment_by_sink", fake_read_curtailment)
    monkeypatch.setattr(loop_mod, "read_slack_penalty", fake_read_penalty)
    return state


def test_run_calibration_stalls_and_stops_early(tmp_path, monkeypatch):
    """Guard flags the last shedding node -> loop STOPS below the budget."""
    # k=0 baseline sheds (bumped, no guard yet); k=1 makes real progress
    # (ΔSlack = 40, 40% of the prior gap → not stalled, keeps bumping); k=2 the
    # residual has stalled (ΔSlack = 1, only ~1.7% of the prior 60 → below the
    # 5% stall fraction) despite the bump, so the guard freezes 'west' -> next
    # round would bump nothing -> stall.  Curtailment is irrelevant to the
    # freeze now (kept in the signals only because the loop reads it for the
    # report).
    signals = [
        {"residual": {"west": 100.0}, "curtailment": {"west": 0.0}},   # k=0
        {"residual": {"west": 60.0}, "curtailment": {"west": 10.0}},   # k=1
        {"residual": {"west": 59.0}, "curtailment": {"west": 80.0}},   # k=2
    ]
    _install_scripted_solve(monkeypatch, signals)

    # Budget is 5 adjustments (6 solves); the stall must cut it short at 3.
    config = _config(tmp_path, iterations=5, tightness=0.1)
    result = run_calibration("db.sqlite", "scenA", config)

    assert result.stop_reason == "stalled"
    assert result.converged is False
    # Early stop: only the baseline + two corrections ran, NOT the full budget.
    assert result.iterations_run == 3
    assert result.iterations_run < config.iterations + 1
    # The guard-flagged node is surfaced.
    assert result.guard_flagged_nodes == ["west"]
    # 'west' is resource-capped, so it was never bumped a third time; the final
    # adder reflects exactly the two pre-flag corrections.
    assert "west" in result.final_adders


def test_run_calibration_converged(tmp_path, monkeypatch):
    """Total unserved falling under the threshold sets stop_reason=converged."""
    signals = [
        {"residual": {"west": 100.0}, "curtailment": {"west": 0.0}},  # k=0
        {"residual": {"west": 0.0}, "curtailment": {"west": 0.0}},    # k=1 -> <=thr
    ]
    _install_scripted_solve(monkeypatch, signals)

    config = _config(tmp_path, iterations=5, tightness=0.1)
    result = run_calibration("db.sqlite", "scenA", config)

    assert result.stop_reason == "converged"
    assert result.converged is True
    assert result.iterations_run == 2
    assert result.guard_flagged_nodes == []


def test_run_calibration_budget_exhausted(tmp_path, monkeypatch):
    """Progress without convergence or a flag runs the full budget."""
    # Residual keeps DROPPING by 10% of the prior gap each round (100→90→80→70),
    # comfortably above the 5% stall fraction, so the guard never freezes; but
    # it stays well above the threshold so it never converges -> the loop must
    # run every iteration.  (A FLAT residual would now correctly read as a
    # stall, so budget-exhausted requires genuine ongoing progress.)
    signals = [
        {"residual": {"west": r}, "curtailment": {"west": 0.0}}
        for r in (100.0, 90.0, 80.0, 70.0)  # iterations=3 -> 4 solves
    ]
    _install_scripted_solve(monkeypatch, signals)

    config = _config(tmp_path, iterations=3, tightness=0.1)
    result = run_calibration("db.sqlite", "scenA", config)

    assert result.stop_reason == "budget_exhausted"
    assert result.converged is False
    assert result.iterations_run == 4  # full budget consumed
    assert result.guard_flagged_nodes == []


def test_budget_exhausted_final_adders_match_last_solved_state(
    tmp_path, monkeypatch,
):
    """No phantom final increment: final_adders == adders SOLVED at the last
    iteration.

    iterations=3 → 4 solves at k=0..3.  The residual drops each round
    (100→90→80→70, always above the 5% stall fraction so nothing freezes).
    The sizing step is applied only at k=0,1,2 (injecting residual/W of
    100/1000, 90/1000, 80/1000 = 0.1, 0.09, 0.08); on the FINAL iteration k=3
    no further solve validates a step, so it is skipped.  Thus the adders
    written and solved at k=3 (three corrections = 0.27) are exactly what
    ``final_adders`` returns — and the trajectory's last residual pairs against
    those solved adders, not a phantom fourth increment that the old loop
    applied but never validated.
    """
    signals = [
        {"residual": {"west": r}, "curtailment": {"west": 0.0}}
        for r in (100.0, 90.0, 80.0, 70.0)
    ]
    _install_scripted_solve(monkeypatch, signals)

    config = _config(tmp_path, iterations=3, tightness=0.1)
    result = run_calibration("db.sqlite", "scenA", config)

    assert result.stop_reason == "budget_exhausted"
    assert result.iterations_run == 4
    # Three validated corrections of residual/W (0.1 + 0.09 + 0.08 = 0.27); the
    # phantom fourth increment is gone.
    assert result.final_adders == {"west": pytest.approx(0.27)}
    # final_adders equals the adders that were WRITTEN and SOLVED at the last
    # iteration (captured in the trajectory before any step is applied).
    assert result.trajectory[-1].adders == pytest.approx(result.final_adders)
    # …and pairs with the last solve's residual (the report's per-node pairing).
    assert result.trajectory[-1].residual == {"west": 70.0}
