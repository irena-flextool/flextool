"""Γ.8.C parity tests for ``flextool.engine_polars._recursive_solve``.

The flexpy port (``RecursiveSolveBuilder`` and the
``ParentSolveInfo`` namedtuple) must produce the same flat
``all_solves`` ordering, complete-solve mapping, parent-roll mapping,
and per-solve time lists as the canonical
``flextool.flextoolrunner.recursive_solves.RecursiveSolveBuilder``
on every fixture.  Any divergence indicates a port bug; this is the
parity oracle for downstream orchestration phases (Γ.8.D).

The test reuses the fixture-discovery pattern from
``test_solve_config_parity.py`` and ``test_timeline_parity.py`` so
adding a new ``work_<S>/`` fixture is automatic.

Coverage targets (per ``audit/solve_orchestration_plan.md §3.4``):

* ``define_solve_recursive`` end-to-end on every parity fixture —
  including the previously-regressing
  ``work_multi_fullYear_battery_nested_*`` and ``work_fullYear_roll``
  fixtures.
* ``_extract_time_range`` four boundary cases (single-, start-, end-,
  middle-period).
* Hand-cooked rolling-window: 4-period, jump=1, horizon=2, duration=4.
* Roll-counter resets between top-level invocations (R-O5).
* Renamed solves keep their complete-solve pointer mapped to the
  un-renamed parent.
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

from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._solve_state import (
    ActiveTimeEntry,
    PathConfig,
    RunnerState,
)
from flextool.engine_polars._timeline import TimelineConfig
from flextool.engine_polars._recursive_solve import (
    ParentSolveInfo,
    RecursiveSolveBuilder,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Fixture discovery — same pattern as test_solve_config_parity.py.
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
    log = logging.getLogger(f"engine_polars.recursive[{scenario}]")
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
    """Set up the read-only flextoolrunner reference state."""
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

    log = logging.getLogger(f"flextoolrunner.recursive[{scenario}]")
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
# Per-fixture parity sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario",
    PARITY_CASES,
    ids=[f"{w}@{s}" for w, s in PARITY_CASES],
)
def test_recursive_solve_parity(work_name: str, scenario: str) -> None:
    """Zero diff vs flextool reference on every fixture.

    Specifically validates the previously-regressing
    ``work_multi_fullYear_battery_nested_*`` and ``work_fullYear_roll``
    fixtures.
    """
    sqlite = DATA / work_name / "tests.sqlite"
    mine = _setup_mine(sqlite, scenario)
    ref = _setup_ref(sqlite, scenario)

    try:
        ms = _expand_mine(mine)
    except Exception as e:
        # Reference may also fail on broken fixtures; in that case
        # both sides should fail with the same exception type.
        try:
            _expand_ref(ref)
        except Exception as e2:
            assert type(e) is type(e2), (
                f"{work_name}/{scenario}: mine raised {type(e).__name__} "
                f"but ref raised {type(e2).__name__}"
            )
            return
        raise

    rs = _expand_ref(ref)

    diffs: list[str] = []
    if ms[0] != rs[0]:
        diffs.append(
            f"all_solves differ: mine={ms[0][:10]}... "
            f"(len={len(ms[0])}) ref={rs[0][:10]}... (len={len(rs[0])})"
        )
    if ms[1] != rs[1]:
        diffs.append(f"complete_solve differs: mine={ms[1]} ref={rs[1]}")
    if ms[2] != rs[2]:
        diffs.append(f"parent_roll differs: mine={ms[2]} ref={rs[2]}")
    if ms[3] != rs[3]:
        # Drill into the first divergent solve for a useful diagnostic.
        for key in ms[3].keys() | rs[3].keys():
            if ms[3].get(key) != rs[3].get(key):
                diffs.append(
                    f"active_time_lists[{key!r}] differs"
                )
                break
    if ms[4] != rs[4]:
        diffs.append("fix_storage_time_lists differs")
    if ms[5] != rs[5]:
        diffs.append("realized_time_lists differs")

    assert not diffs, (
        f"{work_name}/{scenario} recursive solve diverged from reference:\n"
        + "\n".join(f"  - {d}" for d in diffs)
    )


# ---------------------------------------------------------------------------
# _extract_time_range — four boundary cases
# ---------------------------------------------------------------------------


def _make_time_list() -> dict:
    """Build a 3-period synthetic active-time list for boundary tests."""
    return {
        "p2020": [
            ActiveTimeEntry(timestep=f"t{i:03d}", index=i, duration="1")
            for i in range(5)
        ],
        "p2025": [
            ActiveTimeEntry(timestep=f"t{i:03d}", index=i, duration="1")
            for i in range(5, 10)
        ],
        "p2030": [
            ActiveTimeEntry(timestep=f"t{i:03d}", index=i, duration="1")
            for i in range(10, 15)
        ],
    }


def test_extract_time_range_single_period() -> None:
    """Slice within a single period: ``[start[1] : end[1] + 1]``."""
    full = _make_time_list()
    period_order = list(full.keys())
    period_pos = {p: i for i, p in enumerate(period_order)}

    out = RecursiveSolveBuilder._extract_time_range(
        full, period_order, period_pos,
        ["p2025", 1], ["p2025", 3]
    )
    assert list(out.keys()) == ["p2025"]
    assert [e.timestep for e in out["p2025"]] == ["t006", "t007", "t008"]


def test_extract_time_range_start_period() -> None:
    """Start period is sliced; subsequent periods are full."""
    full = _make_time_list()
    period_order = list(full.keys())
    period_pos = {p: i for i, p in enumerate(period_order)}

    out = RecursiveSolveBuilder._extract_time_range(
        full, period_order, period_pos,
        ["p2020", 2], ["p2030", 2]
    )
    assert list(out.keys()) == ["p2020", "p2025", "p2030"]
    assert [e.timestep for e in out["p2020"]] == ["t002", "t003", "t004"]
    assert [e.timestep for e in out["p2025"]] == [
        "t005", "t006", "t007", "t008", "t009"
    ]
    assert [e.timestep for e in out["p2030"]] == ["t010", "t011", "t012"]


def test_extract_time_range_end_period() -> None:
    """End period is sliced ``[0 : end[1] + 1]``; preceding are full."""
    full = _make_time_list()
    period_order = list(full.keys())
    period_pos = {p: i for i, p in enumerate(period_order)}

    out = RecursiveSolveBuilder._extract_time_range(
        full, period_order, period_pos,
        ["p2020", 0], ["p2025", 3]
    )
    assert list(out.keys()) == ["p2020", "p2025"]
    assert len(out["p2020"]) == 5
    assert [e.timestep for e in out["p2025"]] == [
        "t005", "t006", "t007", "t008"
    ]


def test_extract_time_range_middle_period_only_full() -> None:
    """Multi-period range: middle periods come through unchanged."""
    full = _make_time_list()
    period_order = list(full.keys())
    period_pos = {p: i for i, p in enumerate(period_order)}

    out = RecursiveSolveBuilder._extract_time_range(
        full, period_order, period_pos,
        ["p2020", 1], ["p2030", 4]
    )
    assert list(out.keys()) == ["p2020", "p2025", "p2030"]
    # p2025 is the "middle" period — full.
    assert out["p2025"] == full["p2025"]


# ---------------------------------------------------------------------------
# _filter_time_list_by_periods + _filter_time_list_by_parent_scope
# ---------------------------------------------------------------------------


def test_filter_time_list_by_periods() -> None:
    full = _make_time_list()
    period_dict = defaultdict(list, {"S": [("p2020", "p2020"), ("p2030", "p2030")]})
    out = RecursiveSolveBuilder._filter_time_list_by_periods(full, period_dict, "S")
    assert set(out.keys()) == {"p2020", "p2030"}


def test_filter_time_list_by_parent_scope() -> None:
    full = _make_time_list()
    out = RecursiveSolveBuilder._filter_time_list_by_parent_scope(
        full, {"p2020", "p2030"}
    )
    assert set(out.keys()) == {"p2020", "p2030"}
    assert out["p2020"] == full["p2020"]


# ---------------------------------------------------------------------------
# Roll-counter R-O5 invariant
# ---------------------------------------------------------------------------


def test_roll_counter_advances_per_roll_within_one_call() -> None:
    """``state.solve.roll_counter[solve]`` increments once per roll
    produced by ``create_rolling_solves``.

    Reference: ``audit/solve_orchestration_plan.md §R-O5``.
    """
    sqlite = DATA / "work_fullYear_roll" / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("fullYear_roll fixture missing")
    state = _setup_mine(sqlite, "fullYear_roll")

    builder = RecursiveSolveBuilder(state)
    parent_info = ParentSolveInfo(solve=None, roll=None)
    solve_name = next(iter(state.solve.model_solve.values()))[0]
    result = builder.define_solve_recursive(
        solve_name, parent_info, None, None, -1
    )
    n_rolls = len(result.solves)
    # roll_counter should be exactly equal to n_rolls (started at 0).
    assert state.solve.roll_counter[solve_name] == n_rolls, (
        f"roll_counter desync: expected {n_rolls}, got "
        f"{state.solve.roll_counter[solve_name]}"
    )


# ---------------------------------------------------------------------------
# Single-matching-period rename path — duplicate_solve carbon-copy
# ---------------------------------------------------------------------------


def test_renamed_child_keeps_complete_solve_pointer() -> None:
    """When a child solve gets renamed (single-matching-period rename
    path), ``complete_solves[renamed]`` still points back to the
    original (un-renamed) name so the orchestrator's
    ``solve_period_history`` can find the parent's accumulated
    history.

    Reference: ``flextoolrunner/recursive_solves.py:455-475`` (rename
    path) + ``recursive_solves.py:500-505`` (complete_solve_name pass-
    through).
    """
    sqlite = (
        DATA
        / "work_multi_fullYear_battery_nested_24h_invest_one_solve"
        / "tests.sqlite"
    )
    if not sqlite.exists():
        pytest.skip(
            "multi_fullYear_battery_nested_24h_invest_one_solve missing"
        )
    state = _setup_mine(
        sqlite, "multi_fullYear_battery_nested_24h_invest_one_solve"
    )
    ms = _expand_mine(state)
    all_solves, complete_solves = ms[0], ms[1]

    # If any solve name was renamed (contains "_p"), its complete-solve
    # pointer should drop the per-period suffix.
    rename_seen = False
    for s in all_solves:
        if s in complete_solves and complete_solves[s] != s:
            # The rename path produces ``<name>_p2020``,
            # ``<name>_roll_0``, etc.; the complete_solves entry should
            # always be a real-solve name (not a renamed/rolled child).
            assert complete_solves[s] in state.solve.real_solves, (
                f"{s!r}'s complete_solve {complete_solves[s]!r} not in "
                f"real_solves"
            )
            rename_seen = True
    # We don't strictly require rename to happen in this fixture, but
    # if it did the invariant above must hold.
    if not rename_seen:
        # Ensure the test wasn't a no-op for the wrong reason — we
        # should at least see SOME complete_solves mapping.
        assert complete_solves


# ---------------------------------------------------------------------------
# Hand-cooked rolling-window: 4-period, jump=1, horizon=2, duration=4
# ---------------------------------------------------------------------------


def test_create_rolling_solves_handcooked_4period() -> None:
    """4-period solve with jump=1, horizon=2, duration=4 produces 4
    rolls named ``solve_roll_0``..``solve_roll_3`` with 2-period
    horizons each.

    Each roll's realized list covers 1 period (the jump) and the
    active list covers 2 periods (the horizon), modulo the last roll
    which gets clipped at the duration boundary.
    """
    log = logging.getLogger("test_handcooked")
    # Build a 4-period active time list with 1-hour timesteps,
    # 1 timestep per period (so jump=1h means 1 step per roll).
    full = {
        "p2020": [ActiveTimeEntry(timestep="t01", index=0, duration="1")],
        "p2025": [ActiveTimeEntry(timestep="t02", index=1, duration="1")],
        "p2030": [ActiveTimeEntry(timestep="t03", index=2, duration="1")],
        "p2035": [ActiveTimeEntry(timestep="t04", index=3, duration="1")],
    }

    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list, {"m": ["S"]}),
        solve_modes={"S": "rolling_window"},
        rolling_times=defaultdict(list, {"S": [1, 2, 4]}),
        highs=__import__(
            "flextool.engine_polars._solve_config",
            fromlist=["HiGHSConfig"],
        ).HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=__import__(
            "flextool.engine_polars._solve_config",
            fromlist=["SolverSettings"],
        ).SolverSettings(
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
    sc.roll_counter["S"] = 0

    state = RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
        timeline=None,
    )

    builder = RecursiveSolveBuilder(state)
    solves, active_lists, realized_lists = builder.create_rolling_solves(
        "S", full, jump=1, horizon=2, start=None, duration=4
    )

    assert solves == ["S_roll_0", "S_roll_1", "S_roll_2", "S_roll_3"]
    # Each realized list covers 1 period (the jump portion).
    for roll in solves:
        assert len(realized_lists[roll]) == 1
    # Each active list covers 2 periods (the horizon) except the last,
    # which gets clipped when the duration ends.
    for roll in solves[:-1]:
        assert len(active_lists[roll]) == 2
    # The roll counter advanced by 4.
    assert sc.roll_counter["S"] == 4


def test_create_rolling_solves_start_not_found_raises() -> None:
    """Bad ``start`` triggers ``FlexToolConfigError``."""
    log = logging.getLogger("test_bad_start")
    full = {
        "p2020": [ActiveTimeEntry(timestep="t01", index=0, duration="1")],
    }
    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list, {"m": ["S"]}),
        solve_modes={"S": "rolling_window"},
        rolling_times=defaultdict(list, {"S": [1, 1, 1]}),
        highs=__import__(
            "flextool.engine_polars._solve_config",
            fromlist=["HiGHSConfig"],
        ).HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=__import__(
            "flextool.engine_polars._solve_config",
            fromlist=["SolverSettings"],
        ).SolverSettings(
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
    sc.roll_counter["S"] = 0
    state = RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
        timeline=None,
    )
    builder = RecursiveSolveBuilder(state)

    from flextool.engine_polars._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="Start point not found"):
        builder.create_rolling_solves(
            "S", full, jump=1, horizon=1,
            start=["p2020", "t999_nonexistent"],
            duration=1,
        )


# ---------------------------------------------------------------------------
# Specific regression coverage — the previously-regressing fixtures
# ---------------------------------------------------------------------------


_REGRESSION_FIXTURES = [
    ("work_multi_fullYear_battery_nested_multi_invest", 80),
    ("work_multi_fullYear_battery_nested_24h_invest_one_solve", 74),
    ("work_fullYear_roll", 72),
]


@pytest.mark.parametrize("work_name,expected_n_solves", _REGRESSION_FIXTURES)
def test_regression_fixture_solve_count(work_name: str, expected_n_solves: int) -> None:
    """The previously-regressing fixtures (``handoff_csv_retirement``)
    expand to the same number of solves as the flextool reference.

    Reference: the CSV retirement agent surfaced parity gaps on these
    fixtures; this guards against a re-introduction.
    """
    from test_solve_config_parity import _DIRNAME_TO_SCENARIO_OVERRIDES as _O
    sqlite = DATA / work_name / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip(f"{work_name} fixture missing")
    scenario = _O.get(work_name, work_name.removeprefix("work_"))
    state = _setup_mine(sqlite, scenario)
    ms = _expand_mine(state)
    assert len(ms[0]) == expected_n_solves, (
        f"{work_name}: expected {expected_n_solves} solves, got "
        f"{len(ms[0])} (mine_first10={ms[0][:10]})"
    )
