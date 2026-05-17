"""Γ.8.B foundation parity tests for ``flextool.engine_polars._timeline``.

The flexpy port (``_timeline.TimelineConfig``) must be field-by-field
equivalent to the canonical ``flextool.flextoolrunner.timeline_config.TimelineConfig``
on every fixture in ``tests/engine_polars/data/work_*``.  Any
divergence indicates a port bug; this is the parity oracle for
downstream orchestration phases (Γ.8.C / Γ.8.D).

Coverage targets (per ``audit/solve_orchestration_plan.md §4 Γ.8.B``):

* ``load_from_db`` field parity — every dict / list field (incl. the
  RP-weights and timeset_weights decoders for both ``api.Map`` and
  ``list[tuple]`` Spine API shapes).
* ``create_assumptive_parts`` — six fallback rules; idempotency.
* ``create_timeline_from_timestep_duration`` — synthesised
  ``"{timeline}_{solve}"`` timelines and the ``original_timeline``
  reverse map.
* ``create_averaged_timeseries`` — ``pt_*.csv`` / ``pbt_*.csv``
  aggregation with sum vs. average semantics, including the
  ``storage_state_reference_value`` bypass.
* Free functions — ``get_active_time``, ``make_step_jump``,
  ``make_period_block``, ``make_steps``, ``make_timeset_timeline``.

The test reuses the fixture-discovery pattern from
``test_solve_config_parity.py`` so adding a new ``work_<S>/`` fixture
is automatic.
"""
from __future__ import annotations

import csv
import logging
import shutil
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
    FlexToolConfigError,
)
from flextool.engine_polars._timeline import (
    TimelineConfig,
    get_active_time,
    make_period_block,
    make_step_jump,
    make_steps,
    make_timeset_timeline,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Fixture discovery — reused from test_solve_config_parity.py
# ---------------------------------------------------------------------------


_DIRNAME_TO_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _discover_fixtures() -> list[tuple[str, str]]:
    """Return ``[(work_dirname, scenario_name), ...]`` for parity cases.

    Mirrors the discovery pattern from ``test_solve_config_parity.py``.
    """
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


def _load_mine(sqlite: Path, scenario: str) -> TimelineConfig:
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        return TimelineConfig.load_from_db(
            db, logging.getLogger(f"engine_polars.timeline[{scenario}]")
        )


def _load_ref(sqlite: Path, scenario: str):
    """Load via the read-only flextoolrunner reference port."""
    from flextool.flextoolrunner.timeline_config import (
        TimelineConfig as RefTimelineConfig,
    )

    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        return RefTimelineConfig.load_from_db(
            db, logging.getLogger(f"flextoolrunner.timeline[{scenario}]")
        )


def _load_mine_solve(sqlite: Path, scenario: str) -> SolveConfig:
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        return SolveConfig.load_from_db(
            db, logging.getLogger(f"engine_polars.solve[{scenario}]")
        )


def _load_ref_solve(sqlite: Path, scenario: str):
    from flextool.flextoolrunner.solve_config import (
        SolveConfig as RefSolveConfig,
    )

    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        return RefSolveConfig.load_from_db(
            db, logging.getLogger(f"flextoolrunner.solve[{scenario}]")
        )


_PLAIN_FIELDS = (
    "timesets",
    "new_step_durations",
    "rp_weights",
    "timeset_weights",
)
_DICT_FIELDS = (
    "timelines",
    "timesets__timeline",
    "timeset_durations",
)


def _diff(name: str, mine, ref) -> str | None:
    a = dict(mine) if isinstance(mine, defaultdict) else mine
    b = dict(ref) if isinstance(ref, defaultdict) else ref
    if a == b:
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        missing = kb - ka
        extra = ka - kb
        bad_vals = {k: (a[k], b[k]) for k in (ka & kb) if a[k] != b[k]}
        return (
            f"{name} differs: missing_keys={sorted(missing)} "
            f"extra_keys={sorted(extra)} value_diffs={bad_vals}"
        )
    return f"{name} differs: mine={a!r} ref={b!r}"


# ---------------------------------------------------------------------------
# Round-trip parity (load_from_db)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario",
    PARITY_CASES,
    ids=[f"{w}@{s}" for w, s in PARITY_CASES],
)
def test_timeline_load_field_parity(work_name: str, scenario: str) -> None:
    """Field-by-field zero diff vs flextool's reference TimelineConfig."""
    sqlite = DATA / work_name / "tests.sqlite"
    mine = _load_mine(sqlite, scenario)
    ref = _load_ref(sqlite, scenario)

    diffs: list[str] = []
    for f in _PLAIN_FIELDS + _DICT_FIELDS:
        d = _diff(f, getattr(mine, f), getattr(ref, f))
        if d:
            diffs.append(d)

    assert not diffs, (
        f"{work_name}/{scenario} timeline diverged from reference:\n"
        + "\n".join(f"  - {d}" for d in diffs)
    )


# ---------------------------------------------------------------------------
# create_assumptive_parts parity + idempotency
# ---------------------------------------------------------------------------


def _run_create_assumptive_parts_mine(
    work_name: str, scenario: str
) -> tuple[TimelineConfig, SolveConfig]:
    sqlite = DATA / work_name / "tests.sqlite"
    timeline = _load_mine(sqlite, scenario)
    solve = _load_mine_solve(sqlite, scenario)
    timeline.create_assumptive_parts(solve)
    return timeline, solve


def _run_create_assumptive_parts_ref(work_name: str, scenario: str):
    sqlite = DATA / work_name / "tests.sqlite"
    timeline = _load_ref(sqlite, scenario)
    solve = _load_ref_solve(sqlite, scenario)
    timeline.create_assumptive_parts(solve)
    return timeline, solve


@pytest.mark.parametrize(
    "work_name,scenario",
    PARITY_CASES,
    ids=[f"{w}@{s}" for w, s in PARITY_CASES],
)
def test_create_assumptive_parts_parity(
    work_name: str, scenario: str
) -> None:
    """create_assumptive_parts produces the same mutations on both
    ``self`` and ``solve_config`` as the flextool reference."""
    mine_tl, mine_sc = _run_create_assumptive_parts_mine(work_name, scenario)
    ref_tl, ref_sc = _run_create_assumptive_parts_ref(work_name, scenario)

    diffs: list[str] = []
    for f in _PLAIN_FIELDS + _DICT_FIELDS:
        d = _diff(f, getattr(mine_tl, f), getattr(ref_tl, f))
        if d:
            diffs.append(d)
    # Solve-side mutations made by Rules 4-6.
    for f in (
        "model_solve",
        "realized_periods",
        "timesets_used_by_solves",
        "solve_period_years_represented",
    ):
        d = _diff(
            f"solve.{f}",
            getattr(mine_sc, f),
            getattr(ref_sc, f),
        )
        if d:
            diffs.append(d)

    assert not diffs, (
        f"{work_name}/{scenario} assumptive_parts diverged:\n"
        + "\n".join(f"  - {d}" for d in diffs)
    )


def test_create_assumptive_parts_idempotent() -> None:
    """Running ``create_assumptive_parts`` twice is a no-op.

    The six rules each gate on "target already populated" so a second
    pass should produce the same state as the first.  Reference:
    ``audit/solve_orchestration_plan.md §3.2 — idempotency invariant``.
    """
    if not PARITY_CASES:
        pytest.skip("no fixtures to test idempotency on")

    work_name, scenario = PARITY_CASES[0]
    sqlite = DATA / work_name / "tests.sqlite"
    timeline = _load_mine(sqlite, scenario)
    solve = _load_mine_solve(sqlite, scenario)

    timeline.create_assumptive_parts(solve)
    after_first = (
        dict(timeline.timesets__timeline),
        dict(timeline.timeset_durations),
        list(timeline.timesets),
        dict(solve.model_solve),
        dict(solve.realized_periods),
        dict(solve.timesets_used_by_solves),
        dict(solve.solve_period_years_represented),
    )

    timeline.create_assumptive_parts(solve)
    after_second = (
        dict(timeline.timesets__timeline),
        dict(timeline.timeset_durations),
        list(timeline.timesets),
        dict(solve.model_solve),
        dict(solve.realized_periods),
        dict(solve.timesets_used_by_solves),
        dict(solve.solve_period_years_represented),
    )

    assert after_first == after_second, (
        "create_assumptive_parts is not idempotent on "
        f"{work_name}/{scenario}: a second invocation mutated state."
    )


# ---------------------------------------------------------------------------
# create_timeline_from_timestep_duration parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario",
    PARITY_CASES,
    ids=[f"{w}@{s}" for w, s in PARITY_CASES],
)
def test_create_timeline_from_timestep_duration_parity(
    work_name: str, scenario: str
) -> None:
    """create_timeline_from_timestep_duration synthesises the same
    per-solve aggregated timelines (and original_timeline reverse
    map) as the flextool reference.

    Most fixtures don't activate ``new_stepduration`` for their
    primary scenario but the parametrised loop still exercises the
    "no-op when not set" path.
    """
    mine_tl, mine_sc = _run_create_assumptive_parts_mine(work_name, scenario)
    ref_tl, ref_sc = _run_create_assumptive_parts_ref(work_name, scenario)

    mine_tl.create_timeline_from_timestep_duration(mine_sc)
    ref_tl.create_timeline_from_timestep_duration(ref_sc)

    diffs: list[str] = []
    for f in (
        "timelines",
        "timesets__timeline",
        "timeset_durations",
        "original_timeline",
    ):
        d = _diff(f, getattr(mine_tl, f), getattr(ref_tl, f))
        if d:
            diffs.append(d)

    assert not diffs, (
        f"{work_name}/{scenario} timeline_from_timestep_duration "
        f"diverged:\n" + "\n".join(f"  - {d}" for d in diffs)
    )


def test_new_stepduration_conflicting_timeset_raises(tmp_path: Path) -> None:
    """Defensive guard: if the same timeset is shared between two
    solves with different ``new_stepduration`` values, raise.

    Flextool's source comment says this should be rejected at
    migration time but doesn't enforce.  Per
    ``audit/handoff_post_split_todo.md`` "fail loudly" invariant the
    flexpy port enforces it.
    """
    timeline = TimelineConfig(
        timelines=defaultdict(
            list,
            {"y2020": [(f"t{i:04d}", "1.0") for i in range(1, 13)]},
        ),
        timesets=["shared_ts"],
        timesets__timeline=defaultdict(
            list, {"shared_ts": "y2020"}
        ),
        timeset_durations=defaultdict(
            list, {"shared_ts": [("t0001", 12)]}
        ),
        new_step_durations={"solve_a": "2.0", "solve_b": "3.0"},
    )

    class _SolveStub:
        timesets_used_by_solves = {
            "solve_a": [("p2020", "shared_ts")],
            "solve_b": [("p2020", "shared_ts")],
        }

    with pytest.raises(
        FlexToolConfigError, match="shared between solves"
    ):
        timeline.create_timeline_from_timestep_duration(_SolveStub())


# ---------------------------------------------------------------------------
# create_averaged_timeseries — semantics
# ---------------------------------------------------------------------------


def _build_minimal_workdir(tmp_path: Path, n_steps: int = 24) -> Path:
    """Create input/ + solve_data/ dirs with one pt_node_inflow.csv
    and the matching pt_*.csv stubs that ``create_averaged_timeseries``
    expects to copy/aggregate.
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir()
    sd.mkdir()

    # pt_node_inflow.csv: production shape is 3-col (node, time,
    # pt_node_inflow); sum semantics.
    with open(inp / "pt_node_inflow.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "time", "pt_node_inflow"])
        for i in range(1, n_steps + 1):
            w.writerow(["west", f"t{i:04d}", float(i)])

    # All other pt_*.csv files: 4-col header, average semantics.
    for fn in (
        "pt_commodity.csv",
        "pt_group.csv",
        "pt_node.csv",
        "pt_process.csv",
        "pt_profile.csv",
        "pt_process_source.csv",
        "pt_process_sink.csv",
        "pt_reserve__upDown__group.csv",
        "pbt_node_inflow.csv",
        "pbt_node.csv",
        "pbt_process.csv",
        "pbt_profile.csv",
        "pbt_process_source.csv",
        "pbt_process_sink.csv",
        "pbt_reserve__upDown__group.csv",
    ):
        with open(inp / fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entity", "param", "time", "value"])
            for i in range(1, n_steps + 1):
                w.writerow(["e1", "p1", f"t{i:04d}", float(i)])

    # p_node.csv: constant inflow row used after aggregation.
    with open(inp / "p_node.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "nodeParam", "value"])
        w.writerow(["west", "inflow", "10.0"])

    return tmp_path


def _provider_with_workdir(work: Path) -> "FlexDataProvider":
    """Seed a FlexDataProvider with every ``input/*.csv`` under *work*.

    Mirrors the cascade's pre-Phase-E behavior: every ``input/<file>``
    on disk is loaded as an all-Utf8 polars frame and put under the
    canonical ``input/<stem>`` Provider key.  The Provider is then
    passed to ``create_averaged_timeseries`` (Phase C contract).
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    import polars as pl

    p = FlexDataProvider()
    for csv_path in (work / "input").glob("*.csv"):
        frame = pl.read_csv(csv_path, infer_schema_length=0)
        p.put(f"input/{csv_path.stem}", frame)
    return p


def test_create_averaged_timeseries_no_op_when_no_new_stepduration(
    tmp_path: Path,
) -> None:
    """When solve doesn't set new_stepduration, frames pass through
    unchanged from ``input/<file>`` to ``solve_data/<file>`` on the
    Provider.
    """
    work = _build_minimal_workdir(tmp_path, n_steps=8)
    provider = _provider_with_workdir(work)

    timeline = TimelineConfig(
        timelines=defaultdict(list),
        timesets=[],
        timesets__timeline=defaultdict(list),
        timeset_durations=defaultdict(list),
        new_step_durations={},
    )

    class _SolveStub:
        timesets_used_by_solves: dict = {}

    timeline.create_averaged_timeseries(
        "solve_a",
        _SolveStub(),
        logging.getLogger("test"),
        provider=provider,
        work_folder=work,
    )

    src_frame = provider.get("input/pt_node_inflow")
    dst_frame = provider.get("solve_data/pt_node_inflow")
    assert dst_frame is not None
    assert src_frame.equals(dst_frame)


def test_create_averaged_timeseries_sum_vs_average(
    tmp_path: Path,
) -> None:
    """new_stepduration=4 over an hourly base: pt_node_inflow sums
    each 4-step group; pt_commodity averages."""
    n_base = 8
    work = _build_minimal_workdir(tmp_path, n_steps=n_base)
    provider = _provider_with_workdir(work)

    # Base timeline: 8 hourly steps t0001..t0008.
    base = [(f"t{i:04d}", "1.0") for i in range(1, n_base + 1)]
    # New aggregated timeline: two 4-hour blocks, t0001 (covers
    # t0001..t0004) and t0005 (covers t0005..t0008).
    new_tl = [("t0001", "4.0"), ("t0005", "4.0")]

    timeline = TimelineConfig(
        timelines=defaultdict(
            list, {"y2020": base, "y2020_solve_a": new_tl}
        ),
        timesets=["full"],
        timesets__timeline=defaultdict(
            list, {"full": "y2020_solve_a"}
        ),
        timeset_durations=defaultdict(list, {"full": [("t0001", 8)]}),
        new_step_durations={"solve_a": "4.0"},
    )
    timeline.original_timeline["y2020_solve_a"] = "y2020"

    class _SolveStub:
        timesets_used_by_solves = {"solve_a": [("p2020", "full")]}

    timeline.create_averaged_timeseries(
        "solve_a",
        _SolveStub(),
        logging.getLogger("test"),
        provider=provider,
        work_folder=work,
    )

    inflow_frame = provider.get("solve_data/pt_node_inflow")
    # Production shape: 3-col (node, time, pt_node_inflow); aggregated
    # rows + appended rows share the same width.
    assert inflow_frame.columns == ["node", "time", "pt_node_inflow"]
    inflow_rows = inflow_frame.rows()
    # Aggregated rows come first (sums of 4-hour blocks);
    # appended-from-p_node rows come second (one per new step).
    # ``1+2+3+4=10`` at t0001 (aggregated), ``5+6+7+8=26`` at t0005.
    # ``p_node[west, 'inflow']=10`` × duration 4 = 40 at each new
    # step (appended).
    inflow_at = {(r[0], r[1]): float(r[2]) for r in inflow_rows}
    # Both aggregated AND appended write to the same key, but appended
    # rows come after in the row list and are NOT overwritten by the
    # dict comprehension; this matches the legacy CSV which contains
    # both shapes interleaved.  Verify by partitioning on row order:
    agg_rows = inflow_rows[:2]
    appended_rows = inflow_rows[2:]
    assert len(appended_rows) == 2
    agg_at = {(r[0], r[1]): float(r[2]) for r in agg_rows}
    assert agg_at[("west", "t0001")] == 10.0
    assert agg_at[("west", "t0005")] == 26.0
    appended_at = {r[1]: float(r[2]) for r in appended_rows}
    # 10.0 * 4 = 40.0 at each new-timeline step.
    assert appended_at["t0001"] == 40.0
    assert appended_at["t0005"] == 40.0
    del inflow_at  # silence linter

    com_frame = provider.get("solve_data/pt_commodity")
    com_rows = com_frame.rows()
    # 4-col header pass-through; the aggregated values match legacy.
    assert com_rows[0] == ("e1", "p1", "t0001", "2.5")
    assert com_rows[1] == ("e1", "p1", "t0005", "6.5")


def test_create_averaged_timeseries_storage_state_reference_value_bypass(
    tmp_path: Path,
) -> None:
    """``storage_state_reference_value`` rows must NOT be aggregated.

    Reference: ``flextoolrunner/timeline_config.py:474``.
    """
    n_base = 8
    work = _build_minimal_workdir(tmp_path, n_steps=n_base)

    # Replace pt_node.csv with a storage_state_reference_value row.
    with open(work / "input" / "pt_node.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity", "param", "time", "value"])
        # Mix a storage_state row (special) with a regular param row.
        # Per the source, special rows preserve a single-value output
        # mapped to the new timeline's nearest at-or-before step.
        w.writerow(["n1", "storage_state_reference_value", "t0003", 99.0])
        for i in range(1, n_base + 1):
            w.writerow(["e1", "regular_param", f"t{i:04d}", float(i)])

    base = [(f"t{i:04d}", "1.0") for i in range(1, n_base + 1)]
    new_tl = [("t0001", "4.0"), ("t0005", "4.0")]
    provider = _provider_with_workdir(work)

    timeline = TimelineConfig(
        timelines=defaultdict(
            list, {"y2020": base, "y2020_solve_a": new_tl}
        ),
        timesets=["full"],
        timesets__timeline=defaultdict(
            list, {"full": "y2020_solve_a"}
        ),
        timeset_durations=defaultdict(list, {"full": [("t0001", 8)]}),
        new_step_durations={"solve_a": "4.0"},
    )
    timeline.original_timeline["y2020_solve_a"] = "y2020"

    class _SolveStub:
        timesets_used_by_solves = {"solve_a": [("p2020", "full")]}

    timeline.create_averaged_timeseries(
        "solve_a",
        _SolveStub(),
        logging.getLogger("test"),
        provider=provider,
        work_folder=work,
    )

    pt_node_frame = provider.get("solve_data/pt_node")
    rows = pt_node_frame.rows()
    # The storage_state_reference_value row should be present once,
    # mapped to t0001 (the nearest at-or-before new-timeline step
    # for the original t0003 target).
    storage_rows = [
        r for r in rows if r[1] == "storage_state_reference_value"
    ]
    assert len(storage_rows) == 1
    assert storage_rows[0] == (
        "n1",
        "storage_state_reference_value",
        "t0001",
        "99.0",
    )


# ---------------------------------------------------------------------------
# get_active_time + companions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,scenario",
    PARITY_CASES,
    ids=[f"{w}@{s}" for w, s in PARITY_CASES],
)
def test_get_active_time_parity(work_name: str, scenario: str) -> None:
    """get_active_time returns the same period -> ActiveTimeEntry list
    structure for the active solve as flextool's reference.
    """
    sqlite = DATA / work_name / "tests.sqlite"
    mine_tl = _load_mine(sqlite, scenario)
    mine_sc = _load_mine_solve(sqlite, scenario)
    ref_tl = _load_ref(sqlite, scenario)
    ref_sc = _load_ref_solve(sqlite, scenario)
    mine_tl.create_assumptive_parts(mine_sc)
    ref_tl.create_assumptive_parts(ref_sc)

    if not mine_sc.model_solve:
        pytest.skip(f"{work_name} has no model:solves")
    active_solves: list[str] = []
    for solves in mine_sc.model_solve.values():
        active_solves.extend(solves)
    if not active_solves:
        pytest.skip(f"{work_name} model_solve has no solves")

    for solve in active_solves:
        if solve not in mine_sc.timesets_used_by_solves:
            continue
        from flextool.flextoolrunner.timeline_config import (
            get_active_time as ref_get_active_time,
        )

        mine_at = get_active_time(
            solve,
            mine_sc.timesets_used_by_solves,
            mine_tl.timeset_durations,
            mine_tl.timelines,
            mine_tl.timesets__timeline,
        )
        ref_at = ref_get_active_time(
            solve,
            ref_sc.timesets_used_by_solves,
            ref_tl.timeset_durations,
            ref_tl.timelines,
            ref_tl.timesets__timeline,
        )
        assert dict(mine_at) == dict(ref_at), (
            f"{work_name}/{solve}: get_active_time diverged"
        )


def test_get_active_time_unknown_solve_raises() -> None:
    """An unknown solve raises ``ValueError`` with the documentation
    message from the flextool source (used as a configuration hint
    by callers).
    """
    with pytest.raises(ValueError, match="period_timeset"):
        get_active_time(
            "nope",
            {"other": [("p2020", "ts")]},
            {},
            {},
            {},
        )


def test_get_active_time_no_match_raises() -> None:
    """When the solve exists but its timesets don't resolve to any
    timeline data, raise the second message from the source.
    """
    with pytest.raises(ValueError, match="Failed to map timeset"):
        get_active_time(
            "s",
            {"s": [("p2020", "ts")]},
            {"ts": [("t0001", 4)]},
            {"y2020": []},  # empty timeline
            {"ts": "y2020"},
        )


def test_make_step_jump_simple_two_period() -> None:
    """A two-period contiguous block: each period yields N rows where
    every internal jump is 1 and the last row maps to the previous
    period's last step.  Reference: flextool source lines 608-703.
    """
    active_time_list = {
        "p2020": [
            ActiveTimeEntry("t0001", 0, "1.0"),
            ActiveTimeEntry("t0002", 1, "1.0"),
            ActiveTimeEntry("t0003", 2, "1.0"),
        ],
        "p2021": [
            ActiveTimeEntry("t0004", 3, "1.0"),
            ActiveTimeEntry("t0005", 4, "1.0"),
        ],
    }
    period__branch = [("p2020", "p2020"), ("p2021", "p2021")]
    rows = make_step_jump(active_time_list, period__branch, [])
    # Each row has 7 columns and there should be one row per active
    # step (3 + 2 = 5 rows).  ``r[6]`` is the jump in timeline-index
    # units; it can be negative when the previous-period link wraps
    # around (first row of period N points backwards to last row of
    # period N-1's timeline index).
    assert len(rows) == 5
    for r in rows:
        assert len(r) == 7
        assert isinstance(r[6], int)
    # Internal jumps within p2020 (steps 2 and 3) are jump=1.
    internal_p2020 = [r for r in rows if r[0] == "p2020" and r[1] != "t0001"]
    assert all(r[6] == 1 for r in internal_p2020), internal_p2020
    # The first row of p2020 wraps to the last row of p2021 (the
    # previous_period in reverse iteration).  This is the cross-period
    # link.  Jump magnitude depends on indices.
    p2020_first = [r for r in rows if r[0] == "p2020" and r[1] == "t0001"]
    assert len(p2020_first) == 1


def test_make_period_block_breaks_on_gap() -> None:
    """make_period_block detects gaps via index discontinuity (>1)."""
    active_time_list = {
        "p2020": [
            ActiveTimeEntry("t0001", 0, "1.0"),
            ActiveTimeEntry("t0002", 1, "1.0"),
            # gap (index 3 missing → jump = 4 - 1 = 3)
            ActiveTimeEntry("t0005", 4, "1.0"),
            ActiveTimeEntry("t0006", 5, "1.0"),
        ]
    }
    pbt, pbs = make_period_block(active_time_list)
    # Two blocks: one starting at t0001, one at t0005.
    block_firsts = sorted({r[1] for r in pbt})
    assert block_firsts == ["t0001", "t0005"]
    # Successor pairs are cyclic.
    assert ("p2020", "t0001", "t0005") in pbs
    assert ("p2020", "t0005", "t0001") in pbs


def test_make_steps_inclusive_slice() -> None:
    """make_steps returns ``steplist[start:stop+1]`` (inclusive)."""
    s = ["a", "b", "c", "d", "e"]
    assert make_steps(s, 1, 3) == ["b", "c", "d"]
    assert make_steps(s, 0, 0) == ["a"]


def test_make_timeset_timeline_ceil_length() -> None:
    """Length is rounded up via ``math.ceil`` so non-integer step
    counts include the final partial step.
    """
    s = ["t1", "t2", "t3", "t4", "t5"]
    # start='t2', length=2.4 → indices 1..ceil(1+2.4)=4 (exclusive)
    assert make_timeset_timeline(s, "t2", 2.4) == ["t2", "t3", "t4"]


# ---------------------------------------------------------------------------
# RP-weight non-uniform fixture: bug-hunt validation
# ---------------------------------------------------------------------------


def test_timeset_weights_non_uniform_round_trip() -> None:
    """``work_base_weighted`` is the canonical non-uniform weight
    fixture: timeset_weights map varies in 4 distinct values across
    48 timesteps.  TimelineConfig must round-trip the values verbatim.

    This is the bug-hunt validation pinned in
    ``audit/solve_orchestration_plan.md`` Γ.8.B acceptance — the
    derived-params override chain currently returns the trivial 1.0
    stub for ``p_rp_cost_weight``; this test verifies the timeline
    port produces the correct non-uniform frame that Γ.8.D will
    eventually wire into the override chain.
    """
    sqlite = DATA / "work_base_weighted" / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("work_base_weighted fixture not present")

    timeline = _load_mine(sqlite, "base_weighted")
    weights = timeline.timeset_weights
    assert "2day" in weights
    distinct_values = sorted(set(weights["2day"].values()))
    # The fixture has 4 distinct weights spanning a 4x range.
    assert distinct_values == [0.5, 1.0, 1.5, 2.0], (
        f"non-uniform weight signature changed: got {distinct_values}"
    )
    # Specific anchor values from the source DB.
    assert weights["2day"]["t0001"] == 0.5
    assert weights["2day"]["t0024"] == 1.0
    assert weights["2day"]["t0036"] == 1.5
    assert weights["2day"]["t0048"] == 2.0


def test_representative_period_weights_decoder_handles_both_shapes() -> None:
    """``_decode_rp_weights`` decodes both ``api.Map`` and
    ``list[tuple]`` shapes into the same nested dict.  Reference:
    lines 100-130 of flextool source — the Spine API quirk.
    """
    from flextool.engine_polars._timeline import _decode_rp_weights

    # api.Map shape (outer is a Map of Maps).
    inner1 = api.Map(["t0001", "t0005"], [0.7, 0.3])
    inner2 = api.Map(["t0001"], [1.0])
    outer_map = api.Map(["t0001", "t0009"], [inner1, inner2])

    # list[tuple] shape (what api.convert_map_to_table produces).
    list_shape = [
        ("t0001", [("t0001", 0.7), ("t0005", 0.3)]),
        ("t0009", [("t0001", 1.0)]),
    ]

    decoded_map = _decode_rp_weights({"ts": outer_map})
    decoded_list = _decode_rp_weights({"ts": list_shape})

    assert decoded_map == decoded_list
    assert decoded_map == {
        "ts": {
            "t0001": {"t0001": 0.7, "t0005": 0.3},
            "t0009": {"t0001": 1.0},
        }
    }


def test_timeset_weights_decoder_handles_both_shapes() -> None:
    """``_decode_timeset_weights`` decodes both ``api.Map`` and
    ``list[tuple]`` shapes.  Mirror of the rp_weights test.
    """
    from flextool.engine_polars._timeline import _decode_timeset_weights

    map_shape = api.Map(["t0001", "t0002"], [0.5, 1.5])
    list_shape = [("t0001", 0.5), ("t0002", 1.5)]

    decoded_map = _decode_timeset_weights({"ts": map_shape})
    decoded_list = _decode_timeset_weights({"ts": list_shape})

    assert decoded_map == decoded_list
    assert decoded_map == {"ts": {"t0001": 0.5, "t0002": 1.5}}


# ---------------------------------------------------------------------------
# Module-level smoke tests
# ---------------------------------------------------------------------------


def test_module_exports() -> None:
    """The module exports the expected public surface for downstream
    Γ.8.C / Γ.8.D consumers.
    """
    from flextool.engine_polars import _timeline as tl

    expected = {
        "TimelineConfig",
        "get_active_time",
        "make_step_jump",
        "make_period_block",
        "make_steps",
        "make_timeset_timeline",
        "separate_period_and_timeseries_data",
    }
    assert expected.issubset(set(tl.__all__))


def test_load_from_source_via_spinedb_reader() -> None:
    """``load_from_source`` accepts a SpineDbReader and produces a
    TimelineConfig field-equivalent to ``load_from_db`` on the same DB.
    Mirrors the SolveConfig.load_from_source pattern.
    """
    from flextool.engine_polars._spinedb_reader import SpineDbReader

    if not PARITY_CASES:
        pytest.skip("no fixtures for load_from_source")
    work_name, scenario = PARITY_CASES[0]
    sqlite = DATA / work_name / "tests.sqlite"
    reader = SpineDbReader("sqlite:///" + str(sqlite), scenario)
    via_source = TimelineConfig.load_from_source(reader)
    via_db = TimelineConfig.load_from_db_url(
        "sqlite:///" + str(sqlite), scenario
    )

    assert dict(via_source.timelines) == dict(via_db.timelines)
    assert list(via_source.timesets) == list(via_db.timesets)
    assert dict(via_source.timeset_weights) == dict(via_db.timeset_weights)


def test_load_from_source_unknown_source_raises() -> None:
    """Non-SpineDbReader sources raise NotImplementedError pointing at
    Γ.8.D.  No defensive silent fallback.
    """

    class FakeSource:
        pass

    with pytest.raises(NotImplementedError, match="Γ.8.D"):
        TimelineConfig.load_from_source(FakeSource())
