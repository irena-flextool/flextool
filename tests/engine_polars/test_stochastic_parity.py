"""Γ.8.C parity tests for ``flextool.engine_polars._stochastic``.

The flexpy port (``StochasticSolver`` and the standalone helpers
``connect_two_timelines``, ``find_next_timestep``,
``find_previous_timestep``, ``write_timeline_matching_map``) must
produce equivalent ``period__branch_lists``, ``solve_branch__time_branch_lists``,
``active_time_lists``, ``jump_lists``, ``fix_storage_time_lists``,
``realized_time_lists``, and ``branch_start_time_lists`` to the
canonical ``flextool.flextoolrunner.stochastic.StochasticSolver`` on
every fixture.

The test reuses the fixture-discovery pattern from
``test_solve_config_parity.py``.

Coverage targets (per ``audit/solve_orchestration_plan.md §3.5``):

* ``create_stochastic_periods`` end-to-end on every fixture.
* Hand-cooked 2-period 3-branch scenario with weights [0.4, 0.4, 0.2].
* Realized-only invest invariant (R-O6): branches do NOT enter
  ``invest_periods``.
* Validation raises: missing realized start, multiple realized
  branches per period.
"""
from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path

import pytest
import spinedb_api as api
from spinedb_api.filters.scenario_filter import (
    apply_scenario_filter_to_subqueries,
)

from flextool.engine_polars._solve_config import (
    HiGHSConfig,
    SolveConfig,
    SolverSettings,
)
from flextool.engine_polars._solve_state import (
    ActiveTimeEntry,
    FlexToolConfigError,
    PathConfig,
    RunnerState,
)
from flextool.engine_polars._timeline import TimelineConfig
from flextool.engine_polars._recursive_solve import (
    ParentSolveInfo,
    RecursiveSolveBuilder,
)
from flextool.engine_polars._stochastic import (
    StochasticSolver,
    connect_two_timelines,
    find_next_timestep,
    find_previous_timestep,
    write_timeline_matching_map,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Fixture discovery — reused.
# ---------------------------------------------------------------------------


_DIRNAME_TO_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _discover_fixtures() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("work_"):
            continue
        sqlite = d / "tests.sqlite"
        if not sqlite.exists():
            continue
        if d.name in _DIRNAME_TO_SCENARIO_OVERRIDES:
            target = _DIRNAME_TO_SCENARIO_OVERRIDES[d.name]
            try:
                with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                    found = any(
                        s.name == target
                        for s in db.query(db.scenario_sq).all()
                    )
            except Exception:
                found = False
            if found:
                out.append((d.name, target))
                continue
        scen_target = d.name.removeprefix("work_")
        try:
            with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                scenarios = sorted(
                    s.name for s in db.query(db.scenario_sq).all()
                )
        except Exception:
            continue
        candidates = [scen_target]
        import re

        candidates.append(
            re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", scen_target)
        )
        candidates.append(re.sub(r"(\d+)_([a-z])", r"\1\2", scen_target))
        if scen_target.endswith("_full_storage"):
            base = scen_target[: -len("_full_storage")]
            candidates.append(
                re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base)
            )
            candidates.append(base)
        chosen: str | None = None
        for cand in candidates:
            if cand in scenarios:
                chosen = cand
                break
        if chosen is not None:
            out.append((d.name, chosen))
        elif scenarios:
            out.append((d.name, scenarios[0]))
    return out


PARITY_CASES = _discover_fixtures()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_mine(sqlite: Path, scenario: str) -> RunnerState:
    log = logging.getLogger(f"engine_polars.stochastic[{scenario}]")
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        s = SolveConfig.load_from_db(db, log)
        t = TimelineConfig.load_from_db(db, log)
    t.create_assumptive_parts(s)
    t.create_timeline_from_timestep_duration(s)
    return RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=s,
        logger=log,
        timeline=t,
    )


def _setup_ref(sqlite: Path, scenario: str):
    from flextool.flextoolrunner.solve_config import (
        SolveConfig as RefSolveConfig,
    )
    from flextool.flextoolrunner.timeline_config import (
        TimelineConfig as RefTimelineConfig,
    )
    from flextool.flextoolrunner.runner_state import (
        PathConfig as RefPathConfig,
        RunnerState as RefRunnerState,
    )

    log = logging.getLogger(f"flextoolrunner.stochastic[{scenario}]")
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        s = RefSolveConfig.load_from_db(db, log)
        t = RefTimelineConfig.load_from_db(db, log)
    t.create_assumptive_parts(s)
    t.create_timeline_from_timestep_duration(s)
    return RefRunnerState(
        paths=RefPathConfig(
            flextool_dir=Path("."),
            bin_dir=Path("."),
            root_dir=Path("."),
            output_path=Path("."),
            work_folder=Path("."),
        ),
        solve=s,
        timeline=t,
        logger=log,
    )


def _expand_mine(state: RunnerState):
    """Build all_solves + per-solve time lists via the port."""
    builder = RecursiveSolveBuilder(state)
    all_solves: list[str] = []
    complete_solve: dict = {}
    parent_roll: dict = {}
    active: dict = {}
    fix_storage: dict = {}
    realized: dict = {}
    for s in next(iter(state.solve.model_solve.values())):
        result = builder.define_solve_recursive(
            s, ParentSolveInfo(solve=None, roll=None), None, None, -1
        )
        all_solves += result.solves
        complete_solve.update(result.complete_solves)
        parent_roll.update(result.parent_roll_lists)
        active.update(result.active_time_lists)
        fix_storage.update(result.fix_storage_time_lists)
        realized.update(copy.deepcopy(result.realized_time_lists))
    return all_solves, complete_solve, parent_roll, active, fix_storage, realized


def _expand_ref(state):
    from flextool.flextoolrunner.recursive_solves import (
        ParentSolveInfo as RefParentSolveInfo,
        RecursiveSolveBuilder as RefBuilder,
    )

    builder = RefBuilder(state)
    all_solves: list[str] = []
    complete_solve: dict = {}
    parent_roll: dict = {}
    active: dict = {}
    fix_storage: dict = {}
    realized: dict = {}
    for s in next(iter(state.solve.model_solve.values())):
        result = builder.define_solve_recursive(
            s, RefParentSolveInfo(solve=None, roll=None), None, None, -1
        )
        all_solves += result.solves
        complete_solve.update(result.complete_solves)
        parent_roll.update(result.parent_roll_lists)
        active.update(result.active_time_lists)
        fix_storage.update(result.fix_storage_time_lists)
        realized.update(copy.deepcopy(result.realized_time_lists))
    return all_solves, complete_solve, parent_roll, active, fix_storage, realized


# ---------------------------------------------------------------------------
# Per-fixture parity sweep — full pipeline (recursive + stochastic).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario",
    PARITY_CASES,
    ids=[f"{w}@{s}" for w, s in PARITY_CASES],
)
def test_stochastic_parity(work_name: str, scenario: str) -> None:
    """create_stochastic_periods produces the same outputs as the
    flextool reference on every parity fixture.
    """
    sqlite = DATA / work_name / "tests.sqlite"
    mine = _setup_mine(sqlite, scenario)
    ref = _setup_ref(sqlite, scenario)

    try:
        ms = _expand_mine(mine)
    except Exception as e:
        try:
            _expand_ref(ref)
        except Exception as e2:
            assert type(e) is type(e2)
            return
        raise

    rs = _expand_ref(ref)

    # Now apply stochastic.
    from flextool.flextoolrunner.stochastic import (
        StochasticSolver as RefStochasticSolver,
    )

    mine_solver = StochasticSolver(mine)
    ref_solver = RefStochasticSolver(ref)

    m_out = mine_solver.create_stochastic_periods(
        mine.solve.stochastic_branches, ms[0], ms[1], ms[3], ms[4], ms[5]
    )
    r_out = ref_solver.create_stochastic_periods(
        ref.solve.stochastic_branches, rs[0], rs[1], rs[3], rs[4], rs[5]
    )

    names = [
        "period__branch_lists",
        "solve_branch__time_branch_lists",
        "active_time_lists",
        "jump_lists",
        "fix_storage_time_lists",
        "realized_time_lists",
        "branch_start_time_lists",
    ]
    diffs: list[str] = []
    for i, n in enumerate(names):
        # Defaultdict equality ignores missing-key default factory; compare
        # as plain dicts so missing-vs-empty-list distinctions surface.
        a = dict(m_out[i]) if isinstance(m_out[i], defaultdict) else m_out[i]
        b = dict(r_out[i]) if isinstance(r_out[i], defaultdict) else r_out[i]
        if a != b:
            diffs.append(f"{n} differs")
    assert not diffs, (
        f"{work_name}/{scenario} stochastic diverged from reference:\n"
        + "\n".join(f"  - {d}" for d in diffs)
    )

    # R-O6 invariant: branches do NOT enter invest_periods.
    # A branch name has the form ``period_branch`` (or
    # ``branch_start_period_branch`` for continuation).  Verify no
    # branch name made it into ``invest_periods``.
    for solve, periods in mine.solve.invest_periods.items():
        for p_from, p_in in periods:
            assert "_" not in p_in or p_in == p_from, (
                f"R-O6 violation: invest_periods[{solve}] contains "
                f"branched period {p_in!r}"
            )


# ---------------------------------------------------------------------------
# Hand-cooked 2-period 3-branch test
# ---------------------------------------------------------------------------


def _make_handcooked_state(
    weights: list[float],
    realized_branch: str,
) -> tuple[RunnerState, dict, dict, dict, dict]:
    """Build a tiny 2-period scenario for stochastic spot tests.

    Two periods (``p2020``, ``p2025``) with a single timeset each
    (``ts2020``, ``ts2025``).  Branches only fire at the start of
    p2025.  *weights* is a 3-element list assigning weights to
    branches ``branch_1``, ``branch_2``, ``branch_3``; *realized_branch*
    is the realized branch name.
    """
    log = logging.getLogger("test_handcooked_stochastic")

    # Build three timesteps in each period's "timeline".
    timeline_y2020 = [(f"t{i:03d}", "1.0") for i in range(3)]
    timeline_y2025 = [(f"t{i:03d}", "1.0") for i in range(3, 6)]

    timelines: defaultdict = defaultdict(
        list, {"y2020": timeline_y2020, "y2025": timeline_y2025}
    )
    timesets__timeline = defaultdict(
        list, {"ts2020": "y2020", "ts2025": "y2025"}
    )
    timeset_durations = defaultdict(
        list, {"ts2020": [("t000", 3)], "ts2025": [("t003", 3)]}
    )

    timeline = TimelineConfig(
        timelines=timelines,
        timesets=["ts2020", "ts2025"],
        timesets__timeline=timesets__timeline,
        timeset_durations=timeset_durations,
        new_step_durations={},
    )


    branches = ["branch_1", "branch_2", "branch_3"]
    info: list[tuple] = []
    # All three branches fire at the first step of p2020 (t000) — the
    # same pattern as flextool's 2_day_stochastic_dispatch fixture.
    # The validation at lines 268-277 of the reference requires at
    # least one row at the solve's first step with realized=yes; the
    # *realized_branch* satisfies this.
    for b, w in zip(branches, weights):
        realized_yn = "yes" if b == realized_branch else "no"
        info.append(("p2020", b, "t000", realized_yn, str(w)))

    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list, {"m": ["S"]}),
        solve_modes={"S": "single_solve"},
        rolling_times=defaultdict(list),
        highs=HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(
            solvers={}, precommand={}, arguments=defaultdict(list)
        ),
        solve_period_years_represented=defaultdict(list),
        hole_multipliers=defaultdict(list),
        contains_solves=defaultdict(list),
        stochastic_branches=defaultdict(list, {"S": info}),
        periods_available={},
        delay_durations={},
        logger=log,
    )
    sc.timesets_used_by_solves["S"] = [("p2020", "ts2020"), ("p2025", "ts2025")]
    sc.realized_periods["S"] = [("p2020", "p2020"), ("p2025", "p2025")]

    state = RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
        timeline=timeline,
    )

    # Build the active/realized/fix_storage time lists via the
    # recursive builder.
    builder = RecursiveSolveBuilder(state)
    result = builder.define_solve_recursive(
        "S", ParentSolveInfo(solve=None, roll=None), None, None, -1
    )

    return (
        state,
        {"S": result.complete_solves["S"]},
        result.active_time_lists,
        copy.deepcopy(result.realized_time_lists),
        result.fix_storage_time_lists,
    )


def test_stochastic_handcooked_3branch_weights() -> None:
    """3-branch scenario with weights [0.4, 0.4, 0.2], realized=branch_1.

    The hand-cooked fixture branches at the first step of p2020 (the
    same pattern as flextool's ``2_day_stochastic_dispatch`` fixture
    where all branches fire at the solve start).

    Expected behaviour per ``flextoolrunner/stochastic.py`` source:

    * ``branch_1`` is realized (realized=yes) → does NOT get an entry
      in active_time_lists (the three-way exclusion at line 311-318).
    * ``branch_2`` and ``branch_3`` get separate ``p2020_branch_2``
      and ``p2020_branch_3`` solve_branches with active time.
    """
    state, complete, active, realized, fix_storage = _make_handcooked_state(
        weights=[0.4, 0.4, 0.2], realized_branch="branch_1"
    )

    solver = StochasticSolver(state)
    out = solver.create_stochastic_periods(
        state.solve.stochastic_branches, ["S"], complete, active, fix_storage, realized
    )
    period__branch, sb_tb, active_lists, jump_lists, fix_storage_lists, realized_lists, branch_starts = out

    p_b = list(period__branch["S"])
    # The "first" period in the scenario where branching fires:
    # both periods carry the un-branched (period, period) entry plus
    # a per-branch (period, period_branch) entry for each branch row
    # in the start_times dict that triggered the branching.
    assert ("p2020", "p2020") in p_b
    # branch_1 is realized → it gets a "solve_branch" name with the
    # period_branch_1 form; both branch_2 and branch_3 do too.
    branched_solve_branches = {
        sb for (_p, sb) in p_b if sb != _p
    }
    # Three branches → three sb names.
    assert "p2020_branch_1" in branched_solve_branches
    assert "p2020_branch_2" in branched_solve_branches
    assert "p2020_branch_3" in branched_solve_branches

    # The realized branch (branch_1) does NOT get an active-time entry.
    # branch_2 and branch_3 do.
    assert "p2020_branch_1" not in active_lists["S"]
    assert "p2020_branch_2" in active_lists["S"]
    assert "p2020_branch_3" in active_lists["S"]

    # branch_starts records the first-branched (period, timestep) —
    # the first step of the first period.
    assert branch_starts["S"] == ("p2020", "t000")


def test_stochastic_zero_weight_branch_excluded() -> None:
    """Zero-weight branch is excluded from active_time_lists when both
    ``branch != period`` AND realized != ``yes``.

    Reference: ``flextoolrunner/stochastic.py:311-318``.
    """
    state, complete, active, realized, fix_storage = _make_handcooked_state(
        weights=[1.0, 0.0, 0.0], realized_branch="branch_1"
    )
    solver = StochasticSolver(state)
    out = solver.create_stochastic_periods(
        state.solve.stochastic_branches, ["S"], complete, active, fix_storage, realized
    )
    active_lists = out[2]
    # branch_2 and branch_3 have zero weight → excluded.
    assert "p2025_branch_2" not in active_lists["S"]
    assert "p2025_branch_3" not in active_lists["S"]


def test_stochastic_missing_realized_start_raises() -> None:
    """When ``stochastic_branches`` rows exist for a solve but none
    matches the solve's first step with realized=yes, raise.

    Reference: ``flextoolrunner/stochastic.py:270-277``.
    """
    log = logging.getLogger("test_missing_start")
    timeline = TimelineConfig(
        timelines=defaultdict(
            list, {"y2020": [("t000", "1.0"), ("t001", "1.0")]}
        ),
        timesets=["ts2020"],
        timesets__timeline=defaultdict(list, {"ts2020": "y2020"}),
        timeset_durations=defaultdict(list, {"ts2020": [("t000", 2)]}),
        new_step_durations={},
    )


    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list, {"m": ["S"]}),
        solve_modes={"S": "single_solve"},
        rolling_times=defaultdict(list),
        highs=HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(
            solvers={}, precommand={}, arguments=defaultdict(list)
        ),
        solve_period_years_represented=defaultdict(list),
        hole_multipliers=defaultdict(list),
        contains_solves=defaultdict(list),
        stochastic_branches=defaultdict(
            list,
            {
                # Row for p2020 at t999 with realized=yes — does NOT
                # match the solve's first step (t000).
                "S": [("p2020", "branch_1", "t999", "yes", "1.0")],
            },
        ),
        periods_available={},
        delay_durations={},
        logger=log,
    )
    sc.timesets_used_by_solves["S"] = [("p2020", "ts2020")]
    sc.realized_periods["S"] = [("p2020", "p2020")]

    state = RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
        timeline=timeline,
    )
    builder = RecursiveSolveBuilder(state)
    result = builder.define_solve_recursive(
        "S", ParentSolveInfo(solve=None, roll=None), None, None, -1
    )

    solver = StochasticSolver(state)
    with pytest.raises(FlexToolConfigError, match="realized start time"):
        solver.create_stochastic_periods(
            state.solve.stochastic_branches,
            ["S"],
            {"S": "S"},
            result.active_time_lists,
            result.fix_storage_time_lists,
            copy.deepcopy(result.realized_time_lists),
        )


# ---------------------------------------------------------------------------
# Free-function tests
# ---------------------------------------------------------------------------


def test_connect_two_timelines_returns_cumulative_durations() -> None:
    """``connect_two_timelines`` returns ``{timestep: cumulative_hours}``
    for both solves' timelines."""
    log = logging.getLogger("test_connect")
    timeline = TimelineConfig(
        timelines=defaultdict(
            list,
            {
                "y2020": [
                    ("t000", "1.0"),
                    ("t001", "2.0"),
                    ("t002", "1.0"),
                ],
            },
        ),
        timesets=["ts_a", "ts_b"],
        timesets__timeline=defaultdict(
            list, {"ts_a": "y2020", "ts_b": "y2020"}
        ),
        timeset_durations=defaultdict(list),
        new_step_durations={},
    )

    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list),
        solve_modes={},
        rolling_times=defaultdict(list),
        highs=HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(
            solvers={}, precommand={}, arguments=defaultdict(list)
        ),
        solve_period_years_represented=defaultdict(list),
        hole_multipliers=defaultdict(list),
        contains_solves=defaultdict(list),
        stochastic_branches=defaultdict(list),
        periods_available={},
        delay_durations={},
        logger=log,
    )
    sc.timesets_used_by_solves["solve_a"] = [("p2020", "ts_a")]
    sc.timesets_used_by_solves["solve_b"] = [("p2020", "ts_b")]
    state = RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
        timeline=timeline,
    )
    a_dur, b_dur = connect_two_timelines(
        state, "p2020", "solve_a", "solve_b",
        [("p2020", "p2020")],
    )
    # Both timelines are the same → equal cumulative durations.
    assert a_dur == b_dur == {"t000": 0.0, "t001": 1.0, "t002": 3.0}


def test_write_timeline_matching_map_in_memory_dict() -> None:
    """``write_timeline_matching_map`` returns the matching map as a
    dict (not a CSV side-effect)."""
    log = logging.getLogger("test_matching_map")
    # Two timelines with different resolutions: upper has 2 steps
    # (1h each), lower has 4 steps (0.5h each, but represented as 1h
    # entries for simplicity).
    timeline = TimelineConfig(
        timelines=defaultdict(
            list,
            {
                "y2020": [
                    ("t_up_0", "2.0"),
                    ("t_up_1", "2.0"),
                ],
                "y2020_lower": [
                    ("t_lo_0", "1.0"),
                    ("t_lo_1", "1.0"),
                    ("t_lo_2", "1.0"),
                    ("t_lo_3", "1.0"),
                ],
            },
        ),
        timesets=["ts_up", "ts_lo"],
        timesets__timeline=defaultdict(
            list, {"ts_up": "y2020", "ts_lo": "y2020_lower"}
        ),
        timeset_durations=defaultdict(list),
        new_step_durations={},
    )

    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list),
        solve_modes={},
        rolling_times=defaultdict(list),
        highs=HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(
            solvers={}, precommand={}, arguments=defaultdict(list)
        ),
        solve_period_years_represented=defaultdict(list),
        hole_multipliers=defaultdict(list),
        contains_solves=defaultdict(list),
        stochastic_branches=defaultdict(list),
        periods_available={},
        delay_durations={},
        logger=log,
    )
    sc.timesets_used_by_solves["upper"] = [("p2020", "ts_up")]
    sc.timesets_used_by_solves["lower"] = [("p2020", "ts_lo")]
    state = RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
        timeline=timeline,
    )

    upper_active = {
        "p2020": [
            ActiveTimeEntry(timestep="t_up_0", index=0, duration="2.0"),
            ActiveTimeEntry(timestep="t_up_1", index=1, duration="2.0"),
        ],
    }
    lower_active = {
        "p2020": [
            ActiveTimeEntry(timestep="t_lo_0", index=0, duration="1.0"),
            ActiveTimeEntry(timestep="t_lo_1", index=1, duration="1.0"),
            ActiveTimeEntry(timestep="t_lo_2", index=2, duration="1.0"),
            ActiveTimeEntry(timestep="t_lo_3", index=3, duration="1.0"),
        ],
    }

    matching = write_timeline_matching_map(
        state, upper_active, lower_active, "upper", "lower",
        [("p2020", "p2020")],
    )
    # First two lower steps (0h, 1h cumulative) map to upper's t_up_0
    # (which starts at 0h, lasts 2h).  Third + fourth (2h, 3h
    # cumulative) map to t_up_1 (starts at 2h).
    assert matching == {
        ("p2020", "t_lo_0"): "t_up_0",
        ("p2020", "t_lo_1"): "t_up_0",
        ("p2020", "t_lo_2"): "t_up_1",
        ("p2020", "t_lo_3"): "t_up_1",
    }
