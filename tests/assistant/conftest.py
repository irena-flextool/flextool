"""Every assistant test runs with the user config and cache folders in its own tmp folder and a fresh docs store.

``assistant_project`` is a FlexTool project in a folder whose path has a space, ``#``, ``%25`` and ``ä`` (the
characters that break an unencoded SQLite URI): ``input_sources/tests.sqlite`` is built from
``tests/fixtures/tests.json`` (and migrated to this FlexTool), and ``settings.yaml`` numbers it 1.  Its result
folders have run records written by the engine's own recorder (``run_record.RunRecorder``): ``coal_1`` succeeded,
``coal_chp_1`` succeeded once and then failed (its result files belong to the earlier run), ``base_1`` has result
files but no run record (an older FlexTool); ``output_parquet_comparison`` was built from the run of ``coal_1``.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT_DIR_NAME = "G #2 ä%25x project"


def _user_home_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``FLEXTOOL_USER_HOME`` of this test: in this process and in the processes it starts.  At teardown it goes
    back to the session's temp folder (``tests/user_dirs_guard.py``), never to the real user folders."""
    from flextool.user_dirs import USER_HOME_ENV

    home = tmp_path / "user_home"
    monkeypatch.setenv(USER_HOME_ENV, str(home))
    return home


@pytest.fixture(autouse=True)
def fake_backend_faults() -> Iterator[None]:
    """A fake backend whose script ran out, or that got an invalid history, fails the test even when the session
    turned the raise into a failed answer (``tests/assistant/llm/fake_backend.py``)."""
    from tests.assistant.llm import fake_backend

    fake_backend.FAULTS.clear()
    yield
    faults = list(fake_backend.FAULTS)
    fake_backend.FAULTS.clear()
    assert not faults, f"fake backend faults: {faults}"


@pytest.fixture(autouse=True)
def user_cache_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """FlexTool's user cache folder is in ``tmp_path``: no test writes the real user cache.  The model facts the
    probes keep in this process, and the pacing the process learned, are forgotten too (they stand for the same
    cache)."""
    from flextool.assistant import docs_store, probes
    from flextool.assistant.pacing import PROCESS_PACER
    from flextool.user_dirs import user_cache_dir

    _user_home_in_tmp(tmp_path, monkeypatch)
    cache = user_cache_dir()
    docs_store.reset()
    probes._RECENT.clear()
    PROCESS_PACER.reset()
    yield cache
    docs_store.reset()
    probes._RECENT.clear()
    PROCESS_PACER.reset()


@pytest.fixture(autouse=True)
def user_config_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """FlexTool's user config folder (assistant settings, conversations, usage, key file) is in ``tmp_path``, in
    this process and in the processes a test starts: no test reads or writes the real one."""
    from flextool.assistant import _files

    _user_home_in_tmp(tmp_path, monkeypatch)
    root = _files.config_root()
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """An in-memory keyring in place of the system keyring; a blocked one is released when the test ends."""
    import keyring.core

    from tests.assistant.fake_keyring import FakeKeyring

    fake = FakeKeyring()
    monkeypatch.setattr(keyring.core, "_keyring_backend", fake)
    yield fake
    if fake.block is not None:
        fake.block.set()


@pytest.fixture(autouse=True)
def fresh_input_cache() -> Iterator[None]:
    """Each test loads its databases itself (a cached load would hide what the test checks)."""
    from flextool.assistant.inputs import _db
    from flextool.assistant.results import _runs

    _db.clear_cache()
    _runs.clear_cache()
    yield
    _db.clear_cache()
    _runs.clear_cache()


@pytest.fixture(scope="session")
def tests_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``tests/fixtures/tests.json`` as a database at this FlexTool's version, in a plain tmp folder."""
    return build_tests_db(tmp_path_factory.mktemp("tests_db") / "tests.sqlite")


def make_project(root: Path, db: Path, name: str = "tests.sqlite", number: int = 1) -> Path:
    """A project folder under *root* with *db* copied to ``input_sources/<name>`` and numbered *number*."""
    project = root / PROJECT_DIR_NAME
    (project / "input_sources").mkdir(parents=True)
    shutil.copy(db, project / "input_sources" / name)
    (project / "settings.yaml").write_text(f"input_source_numbers:\n  {name}: {number}\n", encoding="utf-8")
    return project


def write_parquet(folder: Path, name: str = "node_d_ee") -> None:
    """One small result table, written as the engine writes them (lean parquet)."""
    import pandas as pd

    from flextool.lean_parquet import write_lean_parquet

    frame = pd.DataFrame({"west": [1.0, 2.0]}, index=pd.Index(["p2020", "p2025"], name="period"))
    frame.columns.name = "node"
    folder.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(frame, folder / f"{name}.parquet")


def succeed(rec) -> None:
    rec.solve_started("y2020", "y2020", 0, "Solve start: y2020, 1/1")
    rec.solve_classified(model_status="kOptimal", acceptance="optimal", message="", objective=12.5, warm_used=False)
    rec.solve_finished(0)
    rec.outputs_cleared()
    write_parquet(rec.folder)
    rec.outputs_written()
    rec.finish("succeeded", exit_code=0, error=None)


def fail_with(text: str) -> Callable:
    def fail(rec) -> None:
        rec.solve_started("y2020", "y2020", 0, "Solve start: y2020, 1/1")
        rec.solve_classified(model_status="kInfeasible", acceptance="rejected", message=text, objective=None,
                             warm_used=False)
        rec.solve_finished(1)
        rec.finish("failed", exit_code=1, error={"type": "FlexToolSolveError", "message": "Error: 1"})
    return fail


def record_run(project: Path, folder_name: str, scenario: str, outcome: Callable,
               write_methods: tuple[str, ...] = ("parquet", "plot")) -> str:
    """Run *outcome* on an engine run record of ``output_parquet/<folder_name>``; return the run id."""
    from flextool.run_options import RunOptions
    from flextool.run_record import RunRecorder

    folder = project / "output_parquet" / folder_name
    run_options = RunOptions.from_env({})
    rec = RunRecorder.start(
        parquet_dir=folder,
        input={"db_url": f"sqlite:///{(project / 'input_sources' / 'tests.sqlite').as_posix()}",
               "scenario": scenario},
        output={"location": str(project), "subdir": folder_name, "work_folder": str(project / "work"),
                "write_methods": list(write_methods)},
        options=run_options.summary_lines(),
        argv=["cmd_run_flextool", "tests.sqlite"],
        log_capture="parent",
        launcher="gui",
        run_options=run_options,
    )
    try:
        outcome(rec)
    finally:
        rec.close()
    return rec.run_id


def compare_rows(out: dict) -> dict[tuple, dict]:
    """compare_inputs' grouped rows as ``{(class, entity, parameter): row as a dict}``, one per entity."""
    columns = out["columns"]
    rows = {}
    for row in out["differences"]:
        d = dict(zip(columns, row))
        for entity in d["entities"]:
            rows[(d["entity_class"], tuple(entity), d["parameter"])] = d
    return rows


def build_tests_db(path: Path) -> Path:
    """``tests/fixtures/tests.json`` as a database at this FlexTool's version, at *path*."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    migrate_database(json_to_db(FIXTURES / "tests.json", path))
    return path


def build_assistant_project(root: Path, db: Path) -> Path:
    """The synthetic project of the assistant tests (see the module text) under *root*; also used by
    ``tools/record_llm_replies.py``, so recorded replies come from this project only."""
    from flextool.run_record import now_iso
    from flextool.scenario_comparison.db_reader import comparison_members, new_write_id, write_comparison_metadata

    project = make_project(root, db)
    coal = record_run(project, "coal_1", "coal", succeed)
    record_run(project, "coal_chp_1", "coal_chp", succeed)
    record_run(project, "coal_chp_1", "coal_chp", fail_with(f"solve y2020 is infeasible; see {project}/work/log"))
    write_parquet(project / "output_parquet" / "base_1")
    comparison = project / "output_parquet_comparison"
    write_parquet(comparison)
    members = comparison_members({"coal_1": project / "output_parquet" / "coal_1"})
    assert members[0]["run_id"] == coal
    write_comparison_metadata(comparison, ["coal_1"], members, now_iso(), new_write_id())
    return project


@pytest.fixture(scope="session")
def assistant_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    return build_assistant_project(tmp_path_factory.mktemp("assistant"), tests_db)


@pytest.fixture
def ctx(assistant_project: Path):
    from flextool.assistant.context import make_context

    return make_context(assistant_project)


#: The many-runs project (agent feedback 2026-11-09, Cycle 1): one run folder ``<scenario>_1`` for each of the
#: first 67 scenarios of ``tests/fixtures/tests.json`` (in the file's order): the first 43 hold a result file but no
#: run record (an older FlexTool), the next 20 succeeded, the last 4 failed.
MANY_RUNS = 67
MANY_NO_RECORD = 43
MANY_SUCCEEDED = 20
#: cost_unit_annualized_d_ec of every succeeded run: 3 periods × 16 units rows, 12 categories (576 values).
MANY_COST_PERIODS = ("p2020", "p2025", "p2030")
MANY_COST_UNITS = tuple(f"u{i:02d}" for i in range(1, 17))
MANY_COST_CATEGORIES = tuple(f"category {i:02d}" for i in range(1, 13))


def many_run_scenarios() -> list[str]:
    import json

    data = json.loads((FIXTURES / "tests.json").read_text(encoding="utf-8"))
    return [s[0] for s in data["scenarios"]][:MANY_RUNS]


def _write_costs(folder: Path, scenario: str, scale: float) -> None:
    import numpy as np
    import pandas as pd

    from tests.assistant.results_tables import write_result

    index = pd.MultiIndex.from_tuples([(p, u) for p in MANY_COST_PERIODS for u in MANY_COST_UNITS],
                                      names=["period", "unit"])
    values = np.arange(len(index) * len(MANY_COST_CATEGORIES), dtype=float).reshape(len(index), -1) * scale
    frame = pd.DataFrame(values, index=index, columns=pd.Index(MANY_COST_CATEGORIES, name="category"))
    write_result(folder, "cost_unit_annualized_d_ec", frame, scenario)


def build_many_runs_project(root: Path, db: Path) -> Path:
    from tests.assistant.results_tables import succeed_writing

    project = make_project(root, db)
    scenarios = many_run_scenarios()
    for i, scenario in enumerate(scenarios):
        folder = f"{scenario}_1"
        if i < MANY_NO_RECORD:
            write_parquet(project / "output_parquet" / folder)
        elif i < MANY_NO_RECORD + MANY_SUCCEEDED:
            def tables(out: Path, scenario=scenario, scale=1.0 + i / 100) -> None:
                write_parquet(out)
                _write_costs(out, scenario, scale)
            record_run(project, folder, scenario, succeed_writing(tables))
        else:
            record_run(project, folder, scenario, fail_with("solve y2020 is infeasible"))
    return project


@pytest.fixture(scope="session")
def many_runs_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    return build_many_runs_project(tmp_path_factory.mktemp("many_runs"), tests_db)


#: The latest-runs project (Cycle 2): the many-runs project plus a second input source ``tests_b.sqlite``
#: (number 2), fixed run times, discounted cost tables and, for its newest run, capacity and slack tables.
#:
#: * succeeded run ``j`` (0…19, the scenarios after the 43 record-less ones) finished at 2026-10-01 j:00 UTC, but
#:   ``j = 11`` at 10:00 (the time of ``j = 10``: equal times go by folder name); its ``costs_discounted_p_`` is
#:   :func:`latest_costs` (``j = 5`` has none);
#: * failed run ``f`` (0…3) finished at 2026-10-08 f:00 UTC; ``f = 1`` was killed;
#: * the scenario of ``j = 19`` was also run from the second source as ``<scenario>_2``, finished 2026-10-09 00:00
#:   UTC (the newest run), with :data:`OVERVIEW_COSTS`, the capacity tables of :func:`write_overview_tables` and
#:   two slack tables;
#: * ``j = 3`` was made with FlexTool 0.0.1 (another version);
#: * the 43 record-less folders' files were written 2026-11-01 00:00 UTC + i minutes (i = their index; i = 6 at
#:   the time of i = 5): newer than every run record, as in a copied project (critique M2).
LATEST_SUCCEEDED_HOUR = {j: (10 if j == 11 else j) for j in range(MANY_SUCCEEDED)}
LATEST_NO_COST_TABLE = 5
LATEST_OTHER_VERSION = 3
LATEST_KILLED = 1
#: The costs of the newest run: 18 categories, in this table order.
OVERVIEW_COSTS = (
    ("commodity_cost", 600.0), ("commodity_sales", -50.0), ("co2", 80.0), ("other operational", 5.0),
    ("starts", 0.5), ("upward slack penalty", 300.0), ("downward slack penalty", 0.0),
    ("inertia slack penalty", 0.0), ("non-synchronous slack penalty", 0.0), ("upward reserve slack penalty", 2.0),
    ("downward reserve slack penalty", 0.0), ("capacity margin slack penalty", 0.0),
    ("unit investment/retirement", 200.0), ("connection investment/retirement", 40.0),
    ("storage investment/retirement", 30.0), ("fixed cost of existing assets", 10.0),
    ("fixed cost of pre-existing capacity", 0.25), ("profile penalty", 0.0),
)
OVERVIEW_PERIODS = ("p2020", "p2025", "p2030")
#: Units ``k`` = 1…16: existing 10 k in every period; k in (1, 4, 7, 10, 13) invested 100 in p2025 (p2020 empty:
#: no decision), so total 10 k + 100 from p2025; unit 16 divested 20 in p2030 (total 140 there).
OVERVIEW_UNIT_NAMES = tuple(f"wind_offshore_farm_north_{k:02d}" for k in range(1, 17))
OVERVIEW_INVESTED_UNITS = (1, 4, 7, 10, 13)
#: Connections ``k`` = 1…6: existing and total 100 k, nothing invested.
OVERVIEW_CONNECTION_NAMES = tuple(f"interconnector_link_{k}_to_{k + 1}" for k in range(1, 7))


def latest_costs(j: int) -> tuple[tuple[str, float], ...]:
    """``costs_discounted_p_`` of succeeded run *j*: commodity_cost 100 + 10 j, commodity_sales −20 (−500 for
    j = 4), investment 50, upward slack penalty 40 for j in (2, 7), upward reserve slack penalty 5 for j = 7."""
    return (("commodity_cost", 100.0 + 10 * j), ("commodity_sales", -500.0 if j == 4 else -20.0),
            ("investment", 50.0), ("upward slack penalty", 40.0 if j in (2, 7) else 0.0),
            ("upward reserve slack penalty", 5.0 if j == 7 else 0.0))


def write_costs_table(folder: Path, scenario: str, costs: tuple[tuple[str, float], ...]) -> None:
    """``costs_discounted_p_`` as write_outputs writes it: a series by category under a ``scenario`` level."""
    import pandas as pd

    from flextool.lean_parquet import write_lean_parquet

    series = pd.Series(dict(costs), dtype=float)
    series.index.name = "category"
    folder.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(pd.concat({scenario: series}, axis=1, names=["scenario"]), folder / "costs_discounted_p_.parquet")


def write_overview_tables(folder: Path, scenario: str) -> None:
    """The capacity and slack tables of the newest run of the latest-runs project."""
    import numpy as np
    import pandas as pd

    from tests.assistant.results_tables import write_result

    def capacity(stem: str, level: str, rows: dict[str, list[tuple[float, float, float, float, float]]]) -> None:
        index = pd.MultiIndex.from_tuples([(e, p) for e in rows for p in OVERVIEW_PERIODS], names=[level, "period"])
        values = [cell for e in rows for cell in rows[e]]
        frame = pd.DataFrame(values, index=index,
                             columns=pd.Index(["existing", "invested", "in_construction", "divested", "total"],
                                              name="parameter"))
        write_result(folder, stem, frame, scenario)

    nan = np.nan
    units = {}
    for k, name in enumerate(OVERVIEW_UNIT_NAMES, start=1):
        e = 10.0 * k
        if k in OVERVIEW_INVESTED_UNITS:
            units[name] = [(e, nan, 0.0, nan, e), (e, 100.0, 0.0, nan, e + 100), (e, 0.0, 0.0, nan, e + 100)]
        elif k == 16:
            units[name] = [(e, 0.0, 0.0, nan, e), (e, 0.0, 0.0, nan, e), (e, 0.0, 0.0, 20.0, e - 20)]
        else:
            units[name] = [(e, 0.0, 0.0, nan, e)] * 3
    capacity("unit_capacity_ed_p", "unit", units)
    capacity("connection_capacity_ed_p", "connection",
             {name: [(100.0 * k, 0.0, 0.0, nan, 100.0 * k)] * 3
              for k, name in enumerate(OVERVIEW_CONNECTION_NAMES, start=1)})
    capacity("node_capacity_ed_p", "node", {
        "pumped_hydro_reservoir_03": [(1000.0, 0.0, 0.0, nan, 1000.0)] * 3,
        "battery_storage_west_01": [(0.0, nan, 0.0, nan, 0.0), (0.0, 0.0, 0.0, nan, 0.0),
                                    (0.0, 455.5, 0.0, nan, 455.5)],
        "battery_storage_east_02": [(0.0, nan, 0.0, nan, 0.0), (0.0, nan, 0.0, nan, 0.0),
                                    (0.0, nan, 80.0, nan, 0.0)],
    })
    periods = pd.Index(list(OVERVIEW_PERIODS), name="period")
    up = pd.DataFrame({"west_grid_node": [0.0, 1520.5, 3.0], "east_grid_node": [0.0, 0.0, 0.0]}, index=periods)
    up.columns.name = "node"
    write_result(folder, "node_slack_up_d_e", up, scenario)
    down = pd.DataFrame({"west_grid_node": [0.0, 0.0, 0.0]}, index=periods)
    down.columns.name = "node"
    write_result(folder, "node_slack_down_d_e", down, scenario)
    reserve = pd.DataFrame([[0.0], [2.25], [0.0]], index=periods,
                           columns=pd.MultiIndex.from_tuples([("primary_reserve", "up", "all_nodes_group")],
                                                             names=["reserve", "updown", "node_group"]))
    write_result(folder, "nodeGroup_slack_reserve_d_eeg", reserve, scenario)


def set_record(folder: Path, **fields: object) -> None:
    """Rewrite fields of the run record of *folder* (``versions`` merges)."""
    import json

    from flextool.run_record import RUN_STATUS_FILE

    path = folder / RUN_STATUS_FILE
    status = json.loads(path.read_text(encoding="utf-8"))
    for key, value in fields.items():
        if key == "versions":
            status.setdefault("versions", {}).update(value)
        else:
            status[key] = value
    path.write_text(json.dumps(status), encoding="utf-8")


def set_finished(folder: Path, iso: str) -> None:
    """The run record of *folder* finished at *iso* (started ten minutes before)."""
    from datetime import datetime, timedelta

    end = datetime.fromisoformat(iso)
    set_record(folder, finished_at=end.isoformat(), started_at=(end - timedelta(minutes=10)).isoformat())


def latest_no_record_time(i: int) -> float:
    """When the files of record-less folder *i* were written (a POSIX time)."""
    from datetime import datetime, timezone

    minutes = 5 if i == 6 else i
    return datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() + 60 * minutes


def build_latest_runs_project(root: Path, db: Path) -> Path:
    import os

    from tests.assistant.results_tables import succeed_writing

    project = build_many_runs_project(root, db)
    shutil.copy(db, project / "input_sources" / "tests_b.sqlite")
    (project / "settings.yaml").write_text("input_source_numbers:\n  tests.sqlite: 1\n  tests_b.sqlite: 2\n",
                                           encoding="utf-8")
    scenarios = many_run_scenarios()
    out = project / "output_parquet"
    for i in range(MANY_NO_RECORD):
        stamp = latest_no_record_time(i)
        for path in (out / f"{scenarios[i]}_1").glob("*.parquet"):
            os.utime(path, (stamp, stamp))
    for j in range(MANY_SUCCEEDED):
        scenario = scenarios[MANY_NO_RECORD + j]
        folder = out / f"{scenario}_1"
        if j != LATEST_NO_COST_TABLE:
            write_costs_table(folder, scenario, latest_costs(j))
        set_finished(folder, f"2026-10-01T{LATEST_SUCCEEDED_HOUR[j]:02d}:00:00+00:00")
        if j == LATEST_OTHER_VERSION:
            set_record(folder, versions={"flextool": "0.0.1"})
    for f in range(MANY_RUNS - MANY_NO_RECORD - MANY_SUCCEEDED):
        folder = out / f"{scenarios[MANY_NO_RECORD + MANY_SUCCEEDED + f]}_1"
        set_finished(folder, f"2026-10-08T{f:02d}:00:00+00:00")
        if f == LATEST_KILLED:
            set_record(folder, state="killed")
    newest = scenarios[MANY_NO_RECORD + MANY_SUCCEEDED - 1]

    def tables(folder: Path) -> None:
        write_parquet(folder)
        write_costs_table(folder, newest, OVERVIEW_COSTS)
        write_overview_tables(folder, newest)

    record_run(project, f"{newest}_2", newest, succeed_writing(tables))
    set_finished(out / f"{newest}_2", "2026-10-09T00:00:00+00:00")
    return project


@pytest.fixture(scope="session")
def latest_runs_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    return build_latest_runs_project(tmp_path_factory.mktemp("latest_runs"), tests_db)
