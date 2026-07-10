"""Unit tests for SolveConfig — solve duplication and period mapping."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass


from flextool.engine_polars._solve_config import SolverSettings, SolveConfig


@dataclass
class _HiGHS:
    """Minimal stand-in matching the fields duplicate_solve accesses."""
    presolve: dict
    method: dict
    parallel: dict


def _make_solve_config(**overrides) -> SolveConfig:
    """Build a SolveConfig with sensible empty defaults."""
    defaults = dict(
        model=["flexTool"],
        model_solve=defaultdict(list, {"flexTool": ["invest"]}),
        solve_modes={"invest": "single_solve", "dispatch": "single_solve"},
        rolling_times=defaultdict(list),
        highs=_HiGHS(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(solvers={}, precommand={}, arguments=defaultdict(list)),
        solve_period_years_represented=defaultdict(list),
        hole_multipliers=defaultdict(str),
        contains_solves=defaultdict(list, {"invest": ["dispatch"]}),
        stochastic_branches=defaultdict(list),
        periods_available={},
        delay_durations={},
        logger=logging.getLogger("test"),
    )
    defaults.update(overrides)
    return SolveConfig(**defaults)


class TestDuplicateSolve:
    """duplicate_solve must propagate period maps to the new name."""

    def test_realized_periods_propagated(self) -> None:
        """When a solve is duplicated, realized_periods must have the new key."""
        sc = _make_solve_config()
        sc.realized_periods = defaultdict(list, {
            "dispatch": [("p2020", "p2020")],
        })

        sc.duplicate_solve("dispatch", "dispatch_p2020")

        assert "dispatch_p2020" in sc.realized_periods
        assert sc.realized_periods["dispatch_p2020"] == [("p2020", "p2020")]
        # Original key should also be preserved
        assert "dispatch" in sc.realized_periods

    def test_realized_invest_periods_propagated(self) -> None:
        sc = _make_solve_config()
        sc.realized_invest_periods = defaultdict(list, {
            "invest": [("p2020", "p2020")],
        })

        sc.duplicate_solve("invest", "invest_p2020")

        assert "invest_p2020" in sc.realized_invest_periods
        assert sc.realized_invest_periods["invest_p2020"] == [("p2020", "p2020")]

    def test_invest_periods_propagated(self) -> None:
        sc = _make_solve_config()
        sc.invest_periods = defaultdict(list, {
            "invest": [("p2020", "p2020"), ("p2030", "p2030")],
        })

        sc.duplicate_solve("invest", "invest_p2020")

        assert "invest_p2020" in sc.invest_periods
        assert len(sc.invest_periods["invest_p2020"]) == 2

    def test_fix_storage_periods_propagated(self) -> None:
        sc = _make_solve_config()
        sc.fix_storage_periods = defaultdict(list, {
            "dispatch": [("p2020", "p2020")],
        })

        sc.duplicate_solve("dispatch", "dispatch_p2020")

        assert "dispatch_p2020" in sc.fix_storage_periods

    def test_existing_maps_still_propagated(self) -> None:
        """Sanity check: solve_modes (already in dup_map_list) still works."""
        sc = _make_solve_config()

        sc.duplicate_solve("dispatch", "dispatch_p2020")

        assert sc.solve_modes.get("dispatch_p2020") == "single_solve"

    def test_empty_periods_no_error(self) -> None:
        """duplicate_solve should not fail if period maps have no entry for the solve."""
        sc = _make_solve_config()
        # realized_periods is empty — no entry for "dispatch"

        sc.duplicate_solve("dispatch", "dispatch_p2020")

        # Should just not add anything, no error
        assert "dispatch_p2020" not in sc.realized_periods

    def test_scaling_propagated(self) -> None:
        """duplicate_solve carries the per-solve ``scaling`` mode over."""
        sc = _make_solve_config(scaling={"dispatch": "basic"})

        sc.duplicate_solve("dispatch", "dispatch_p2020")

        assert sc.scaling.get("dispatch_p2020") == "basic"


class TestScalingFor:
    """scaling_for resolves the per-solve autoscaler mode (v64)."""

    def test_authored_value_returned(self) -> None:
        sc = _make_solve_config(scaling={"invest": "basic"})
        assert sc.scaling_for("invest") == "basic"

    def test_absent_returns_none(self) -> None:
        """Unauthored solve -> None so the caller falls back to CLI/env/full."""
        sc = _make_solve_config()
        assert sc.scaling_for("invest") is None

    def test_case_and_whitespace_normalised(self) -> None:
        sc = _make_solve_config(scaling={"invest": "  Solver_Only "})
        assert sc.scaling_for("invest") == "solver_only"

    def test_unrecognised_value_returns_none(self) -> None:
        """A typo / unknown mode falls through to None (defer to default)."""
        sc = _make_solve_config(scaling={"invest": "aggressive"})
        assert sc.scaling_for("invest") is None

    def test_all_four_modes_recognised(self) -> None:
        for mode in ("off", "solver_only", "basic", "full"):
            sc = _make_solve_config(scaling={"invest": mode})
            assert sc.scaling_for("invest") == mode
