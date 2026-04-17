"""Tests for the timeset_weights feature.

`timeset_weights` is an optional per-timestep weight map applied to cost
and slack terms in the objective (populates rp_cost_weight.csv in the
non-RP pathway). The runner must normalize so the weights sum to 1 per
period and then scale by the number of active timesteps, so that a
uniform input reproduces the default weight = 1 per step.

These tests cover the writer directly; see scenario-level tests for
end-to-end verification.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from flextool.flextoolrunner.runner_state import ActiveTimeEntry
from flextool.flextoolrunner.solve_writers import write_timeset_cost_weight


def _atl(period_steps: dict[str, list[str]]) -> dict[str, list[ActiveTimeEntry]]:
    return {
        period: [ActiveTimeEntry(timestep=s, index=i, duration=1.0) for i, s in enumerate(steps)]
        for period, steps in period_steps.items()
    }


def _read(path: Path) -> list[tuple[str, str, float]]:
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["period", "time", "weight"]
    return [(r[0], r[1], float(r[2])) for r in rows[1:]]


class TestWriteTimesetCostWeight:
    def test_handoff_example(self, tmp_path: Path):
        """The case in HANDOFF_timeset_weights.md: {0.1, 0.2, 0.3, 0.4}
        already sums to 1, so after ×n the weights become 0.4, 0.8, 1.2, 1.6."""
        (tmp_path / "solve_data").mkdir()
        wrote = write_timeset_cost_weight(
            active_time_list=_atl({"p1": ["t1", "t2", "t3", "t4"]}),
            timesets_used_by_solve=[("p1", "ts4")],
            timeset_weights={"ts4": {"t1": 0.1, "t2": 0.2, "t3": 0.3, "t4": 0.4}},
            work_folder=tmp_path,
        )
        assert wrote is True
        rows = _read(tmp_path / "solve_data" / "rp_cost_weight.csv")
        assert rows == [
            ("p1", "t1", 0.4),
            ("p1", "t2", 0.8),
            ("p1", "t3", 1.2),
            ("p1", "t4", 1.6),
        ]

    def test_uniform_input_reproduces_default(self, tmp_path: Path):
        """Uniform weights must end up as weight = 1 per step (the default)."""
        (tmp_path / "solve_data").mkdir()
        write_timeset_cost_weight(
            active_time_list=_atl({"p1": ["t1", "t2", "t3", "t4"]}),
            timesets_used_by_solve=[("p1", "ts4")],
            timeset_weights={"ts4": {"t1": 0.25, "t2": 0.25, "t3": 0.25, "t4": 0.25}},
            work_folder=tmp_path,
        )
        rows = _read(tmp_path / "solve_data" / "rp_cost_weight.csv")
        assert [r[2] for r in rows] == [1.0, 1.0, 1.0, 1.0]

    def test_unnormalized_input_is_normalized(self, tmp_path: Path):
        """User hands raw year-fraction weights (not summing to 1). Runner scales."""
        (tmp_path / "solve_data").mkdir()
        # OSeMOSYS Rivendell yearsplit (abridged): 2 wet + 2 dry steps
        write_timeset_cost_weight(
            active_time_list=_atl({"p": ["TS01", "TS02", "TS09", "TS10"]}),
            timesets_used_by_solve=[("p", "rivendell_timeset")],
            timeset_weights={"rivendell_timeset": {
                "TS01": 0.05125, "TS02": 0.05125,
                "TS09": 0.07375, "TS10": 0.07375,
            }},
            work_folder=tmp_path,
        )
        rows = _read(tmp_path / "solve_data" / "rp_cost_weight.csv")
        total = 0.05125 * 2 + 0.07375 * 2
        n = 4
        expected = [
            ("p", "TS01", 0.05125 * n / total),
            ("p", "TS02", 0.05125 * n / total),
            ("p", "TS09", 0.07375 * n / total),
            ("p", "TS10", 0.07375 * n / total),
        ]
        assert rows == pytest.approx(expected, rel=1e-9)
        # Sum of weights across the period = n (so the RHS scaling by
        # /complete_period_share_of_year divides evenly).
        assert sum(r[2] for r in rows) == pytest.approx(n, rel=1e-9)

    def test_missing_steps_are_treated_as_zero(self, tmp_path: Path):
        """A step not listed in the map gets weight 0 before normalization."""
        (tmp_path / "solve_data").mkdir()
        write_timeset_cost_weight(
            active_time_list=_atl({"p": ["t1", "t2", "t3", "t4"]}),
            timesets_used_by_solve=[("p", "ts4")],
            timeset_weights={"ts4": {"t1": 1.0, "t3": 1.0}},  # t2 and t4 absent
            work_folder=tmp_path,
        )
        rows = _read(tmp_path / "solve_data" / "rp_cost_weight.csv")
        # Two non-zero entries at 1.0 each; total=2, n=4; scale=2; so
        # non-zero steps get weight 2.0, absent steps get 0.
        assert rows == [
            ("p", "t1", 2.0),
            ("p", "t2", 0.0),
            ("p", "t3", 2.0),
            ("p", "t4", 0.0),
        ]

    def test_returns_false_when_no_timeset_has_weights(self, tmp_path: Path):
        (tmp_path / "solve_data").mkdir()
        wrote = write_timeset_cost_weight(
            active_time_list=_atl({"p": ["t1", "t2"]}),
            timesets_used_by_solve=[("p", "ts")],
            timeset_weights={},
            work_folder=tmp_path,
        )
        assert wrote is False
        assert not (tmp_path / "solve_data" / "rp_cost_weight.csv").exists()

    def test_multiple_periods_independent_normalization(self, tmp_path: Path):
        """Each period's active steps are normalized within that period."""
        (tmp_path / "solve_data").mkdir()
        write_timeset_cost_weight(
            active_time_list=_atl({"p1": ["t1", "t2"], "p2": ["t3", "t4", "t5"]}),
            timesets_used_by_solve=[("p1", "ts"), ("p2", "ts")],
            timeset_weights={"ts": {"t1": 0.25, "t2": 0.75, "t3": 0.1, "t4": 0.4, "t5": 0.5}},
            work_folder=tmp_path,
        )
        rows = _read(tmp_path / "solve_data" / "rp_cost_weight.csv")
        # p1: n=2, total=1 ⇒ weights become 0.5 and 1.5
        # p2: n=3, total=1 ⇒ weights become 0.3, 1.2, 1.5
        assert rows == pytest.approx([
            ("p1", "t1", 0.5),
            ("p1", "t2", 1.5),
            ("p2", "t3", 0.3),
            ("p2", "t4", 1.2),
            ("p2", "t5", 1.5),
        ], rel=1e-9)

    def test_only_some_periods_have_weights(self, tmp_path: Path):
        """One period with weights, another without — only the weighted one
        contributes rows."""
        (tmp_path / "solve_data").mkdir()
        wrote = write_timeset_cost_weight(
            active_time_list=_atl({"p1": ["t1", "t2"], "p2": ["t3", "t4"]}),
            timesets_used_by_solve=[("p1", "ts_weighted"), ("p2", "ts_plain")],
            timeset_weights={"ts_weighted": {"t1": 0.2, "t2": 0.8}},
            work_folder=tmp_path,
        )
        assert wrote is True
        rows = _read(tmp_path / "solve_data" / "rp_cost_weight.csv")
        assert [r[0] for r in rows] == ["p1", "p1"]
        assert rows == pytest.approx([("p1", "t1", 0.4), ("p1", "t2", 1.6)], rel=1e-9)
