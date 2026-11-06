"""Every assistant test runs with the user cache folder in its own tmp folder and a fresh docs store.

``assistant_project`` is a FlexTool project in a folder whose path has a space, ``#``, ``%25`` and ``ä`` (the
characters that break an unencoded SQLite URI): ``input_sources/tests.sqlite`` is built from
``tests/fixtures/tests.json`` (and migrated to this FlexTool), and ``settings.yaml`` numbers it 1.  Its result
folders have run records written by the engine's own recorder (``run_record.RunRecorder``): ``coal_1`` succeeded,
``coal_chp_1`` succeeded once and then failed (its result files belong to the earlier run), ``base_1`` has result
files but no run record (an older FlexTool); ``output_parquet_comparison`` was built from the run of ``coal_1``.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT_DIR_NAME = "G #2 ä%25x project"


@pytest.fixture(autouse=True)
def user_cache_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``platformdirs.user_cache_dir`` points into ``tmp_path``: no test writes the real user cache."""
    import platformdirs

    from flextool.assistant import docs_store

    cache = tmp_path / "user_cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *args, **kwargs: str(cache))
    docs_store.reset()
    yield cache
    docs_store.reset()


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
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    path = tmp_path_factory.mktemp("tests_db") / "tests.sqlite"
    migrate_database(json_to_db(FIXTURES / "tests.json", path))
    return path


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


@pytest.fixture(scope="session")
def assistant_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    from flextool.run_record import now_iso

    project = make_project(tmp_path_factory.mktemp("assistant"), tests_db)
    coal = record_run(project, "coal_1", "coal", succeed)
    record_run(project, "coal_chp_1", "coal_chp", succeed)
    record_run(project, "coal_chp_1", "coal_chp", fail_with(f"solve y2020 is infeasible; see {project}/work/log"))
    write_parquet(project / "output_parquet" / "base_1")
    comparison = project / "output_parquet_comparison"
    write_parquet(comparison)
    (comparison / "_metadata.json").write_text(json.dumps({
        "scenarios": ["coal_1"], "members": [{"folder": "coal_1", "run_id": coal}], "written_at": now_iso()}),
        encoding="utf-8")
    return project


@pytest.fixture
def ctx(assistant_project: Path):
    from flextool.assistant.context import make_context

    return make_context(assistant_project)
