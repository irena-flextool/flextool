"""The GUI and the run record.

* The GUI owns its engine runs' ``_run.log`` and finishes a run record the engine could not finish.  The engine is
  ``tests/run_record/_stub_engine.py`` (it writes a ``running`` status, prints the marker like the engine with a
  pipe owner, then exits or sleeps until it is killed); the manager is built without a window and ``_run_job``
  runs in the test's own thread or a plain thread.
* The executed list, the result viewer and the comparison offer only folders whose latest run succeeded and
  wrote its parquet (``run_record.results_usable``).  The executed-list test binds ``MainWindow`` methods to a
  bare Tk root (run it under ``xvfb-run -a``).
"""
from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from types import MethodType, SimpleNamespace

import pytest

from flextool.cli.cmd_run_flextool import build_parser
from flextool.gui import execution_manager as em
from flextool.gui import main_window as mw
from flextool.gui.check_tree import CheckTreeController
from flextool.gui.cli_format import format_cmd_for_log, terminal_command
from flextool.gui.data_models import ProjectSettings
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager, JobStatus, JobType
from flextool.gui.output_actions import OutputActionManager
from flextool.gui.result_viewer import ResultViewer
from flextool.gui.scenario_lists import ExecutedScenarioManager
from flextool.run_record import MARKER, RUN_LOG_FILE, read_status_file, write_status_atomic

STUB = Path(__file__).resolve().parents[1] / "run_record" / "_stub_engine.py"


@pytest.fixture
def manager(tmp_path, monkeypatch) -> ExecutionManager:
    monkeypatch.setattr(em, "_wrap_for_memory_cap", lambda cmd, estimate_gb: (list(cmd), {}, None))
    return ExecutionManager(project_path=tmp_path, settings=ProjectSettings())


def _job(subdir: str, status: JobStatus = JobStatus.RUNNING, job_id: int = 0) -> ExecutionJob:
    return ExecutionJob(job_id=job_id, job_type=JobType.SCENARIO, scenario_name=subdir, output_subdir=subdir,
                        status=status)


def _stub_job(manager: ExecutionManager, folder: Path, mode: str) -> ExecutionJob:
    job = _job("scen")
    manager._jobs = [job]
    manager._running_count = 1
    manager._build_run_command = lambda job, work_folder: [sys.executable, str(STUB), str(folder), mode]
    return job


def test_the_gui_writes_the_log_and_hides_the_marker(manager, tmp_path):
    folder = tmp_path / "output_parquet" / "scen"
    job = _stub_job(manager, folder, "exit0")

    manager._run_job(job)

    assert job.status is JobStatus.SUCCESS
    assert not any(line.startswith(MARKER) for line in job.stdout_lines)
    record = read_status_file(folder)
    lines = (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# FlexTool run {record['run_id']} started {record['started_at']}"
    # The command line the GUI shows comes before the engine's output, as in the GUI's progress view.
    shown = format_cmd_for_log([sys.executable, str(STUB), str(folder), "exit0"])
    assert job.stdout_lines[0] == shown
    assert lines[1:1 + len(shown.splitlines())] == shown.splitlines()
    assert "done" in lines and not any(line.startswith(MARKER) for line in lines)
    # The engine exited 0 without finishing its record (a stub): the GUI finished it.
    assert record["state"] == "succeeded" and record["error"]["type"] == "RecordIncomplete"


def test_a_job_killed_from_the_gui_is_recorded_as_killed(manager, tmp_path):
    folder = tmp_path / "output_parquet" / "scen"
    job = _stub_job(manager, folder, "sleep")
    worker = threading.Thread(target=manager._run_job, args=(job,), daemon=True)
    worker.start()
    deadline = time.monotonic() + 60
    while "sleeping" not in manager.get_stdout(job.job_id):
        assert time.monotonic() < deadline, manager.get_stdout(job.job_id)
        time.sleep(0.05)

    manager.kill_job(job.job_id)
    worker.join(60)

    assert not worker.is_alive()
    assert job.status is JobStatus.KILLED
    record = read_status_file(folder)
    assert record["state"] == "killed"
    assert record["error"]["message"] == "stopped from the FlexTool window"
    assert [s["state"] for s in record["solves"]] == ["error"]
    assert "sleeping" in (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()


def test_a_job_waits_while_another_job_uses_its_output_folder(manager):
    running = _job("a", JobStatus.RUNNING, 0)
    same_folder = _job("a", JobStatus.PENDING, 1)
    other = _job("b", JobStatus.PENDING, 2)
    manager._jobs = [running, same_folder, other]
    manager._running_count = 1
    manager._max_workers = 4
    manager._memory_admits = lambda candidate, source, estimate: True
    with manager._lock:
        assert manager._pick_next_pending() is other
        assert manager._pick_next_pending() is None
    assert same_folder.status is JobStatus.PENDING


def test_of_two_pending_jobs_on_one_folder_the_first_runs_first(manager):
    first = _job("a", JobStatus.PENDING, 0)
    second = _job("a", JobStatus.PENDING, 1)
    manager._jobs = [first, second]
    manager._max_workers = 4
    manager._memory_admits = lambda candidate, source, estimate: True
    second.force_start = True  # not even a forced start runs it ahead of the first
    with manager._lock:
        assert manager._pick_next_pending() is first
        assert manager._pick_next_pending() is None


def test_the_gui_command_names_its_launcher_and_the_spreadsheet(manager, tmp_path):
    xlsx = tmp_path / "input_sources" / "model.xlsx"
    job = ExecutionJob(job_id=0, scenario_name="scen", output_subdir="scen_1", is_xlsx_source=True, xlsx_path=xlsx,
                       input_db_url="sqlite:///" + str(tmp_path / "intermediate" / "model.sqlite"))
    cmd = manager._build_run_command(job, tmp_path / "work")

    assert cmd[1:3] == ["-m", "flextool.cli.cmd_run_flextool"]
    parsed = build_parser().parse_args(cmd[3:])
    assert parsed.run_launcher == "gui"
    assert parsed.input_source_file == str(xlsx)
    # What a user copies into a terminal: the same run under flextool-run, which records its own launcher.
    copy = terminal_command(cmd)
    assert copy[0] == "flextool-run"
    assert build_parser().parse_args(copy[1:]).run_launcher == "direct"
    assert copy[1:] == [tok for tok in cmd[3:] if tok not in ("--run-launcher", "gui")]


# ---------------------------------------------------------------------------
# The executed list, the result viewer and the comparison read the run record
# ---------------------------------------------------------------------------

_FINISHED = datetime(2026, 11, 5, 14, 2, 40).astimezone().isoformat(timespec="seconds")
_STARTED = datetime(2026, 11, 5, 14, 2, 11).astimezone().isoformat(timespec="seconds")


def _run_folder(project: Path, subdir: str, *, state: str | None, parquet_state: str = "written",
                parquet: bool = True) -> Path:
    """A run folder with one parquet file and (unless *state* is None) a run record."""
    folder = project / "output_parquet" / subdir
    folder.mkdir(parents=True)
    if parquet:
        (folder / "unit_flow.parquet").write_bytes(b"PAR1")
    if state is not None:
        write_status_atomic(folder, {
            "run_id": f"run-{subdir}", "state": state,
            "started_at": _STARTED, "finished_at": None if state == "running" else _FINISHED,
            "heartbeat_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "error": {"type": "FlexToolSolveError", "message": "solve for y2030 is infeasible"}
            if state == "failed" else None,
            "outputs": {"parquet_state": parquet_state},
        })
    return folder


@pytest.fixture
def project(tmp_path) -> Path:
    # good_1: succeeded and wrote its parquet; bad_1: failed, the parquet is an earlier run's; old_1: no record
    # (before run records); live_1: running; csv_1: succeeded with csv output only.
    _run_folder(tmp_path, "good_1", state="succeeded")
    _run_folder(tmp_path, "bad_1", state="failed", parquet_state="previous_run")
    _run_folder(tmp_path, "old_1", state=None)
    _run_folder(tmp_path, "live_1", state="running", parquet_state="previous_run")
    _run_folder(tmp_path, "csv_1", state="succeeded", parquet_state="not_requested", parquet=False)
    return tmp_path


def test_a_failed_folder_with_an_earlier_runs_parquet_is_not_checkable(project):
    rows = {info.name: info for info in ExecutedScenarioManager(project).scan_executed()}

    assert {name for name, info in rows.items() if info.checkable} == {"good", "old"}
    bad = rows["bad"]
    assert (bad.run_state, bad.parquet_state) == ("failed", "previous_run")
    assert bad.unusable_reason == "latest run failed: solve for y2030 is infeasible"
    # The pytest process is not an engine run, and its heartbeat is fresh: "unknown", shown as still running.
    assert rows["live"].run_state == "unknown"
    assert rows["csv"].run_state == "succeeded" and not rows["csv"].checkable
    # The time of a recorded run is its finish (else start) time, in the list's format.
    assert bad.timestamp == "05.11.26 14:02" and rows["live"].timestamp == "05.11.26 14:02"
    assert rows["old"].status is None


def test_the_viewer_leaves_out_failed_folders_and_keeps_their_place(project):
    settings = ProjectSettings()
    settings.executed_scenario_order = ["bad_1", "good_1", "old_1"]
    saves: list[int] = []
    viewer = SimpleNamespace(_project_path=project, _settings=settings,
                             _schedule_settings_save=lambda: saves.append(1))

    listed = ResultViewer._scan_scenarios(viewer)

    # Nothing checked: every folder on disk, minus those whose latest run did not succeed, in the saved order
    # with the new folders (alphabetical) after it.
    assert listed == ["good_1", "old_1"]
    assert settings.executed_scenario_order == ["bad_1", "good_1", "old_1", "csv_1", "live_1"]
    assert saves == [1]


def test_the_comparison_takes_only_folders_whose_results_are_usable(project):
    actions = OutputActionManager(project, ProjectSettings())
    cmd = actions._build_comparison_cmd([(1, "bad"), (1, "good"), (1, "old"), (1, "live")], plots=True, excel=False)
    start = cmd.index("--alternatives") + 1
    end = cmd.index("--plot-dir")
    assert cmd[start:end] == ["good_1", "old_1"]


def test_the_post_batch_comparison_drops_a_folder_whose_latest_job_failed(manager):
    manager.settings.auto_generate_comp_plots = False
    manager.settings.auto_generate_comp_excel = False
    earlier = _job("a", JobStatus.SUCCESS, 0)
    earlier.end_time = datetime(2026, 11, 5, 10, 0)
    later = _job("a", JobStatus.FAILED, 1)
    later.end_time = datetime(2026, 11, 5, 11, 0)
    manager._jobs = [earlier, later, _job("b", JobStatus.SUCCESS, 2)]
    manager._comparison_scenarios = ["a", "c"]

    manager._run_post_execution()

    assert manager._comparison_scenarios == ["c", "b"]


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    yield root
    root.destroy()


def test_the_executed_list_unchecks_and_refuses_rows_whose_run_did_not_succeed(project, tk_root, monkeypatch):
    host = tk_root
    host.executed_tree = ttk.Treeview(
        tk_root, columns=("check", "source_num", "scenario_name", "view", "timestamp"), show="headings",
    )
    settings = ProjectSettings()
    settings.checked_executed_scenarios = ["1|bad", "1|good", "1|live"]
    host.project_settings = settings
    host.exec_scenario_mgr = ExecutedScenarioManager(project, settings)
    host._executed_sort_mode = "timestamp"
    host._executed_row_info = {}
    host._sort_tree_items = MethodType(mw.MainWindow._sort_tree_items, host)
    host._update_output_status = lambda: None
    host._save_checked_executed_scenarios = MethodType(mw.MainWindow._collect_checked_executed_scenarios, host)
    host._executed_row_checkable = MethodType(mw.MainWindow._executed_row_checkable, host)

    mw.MainWindow._refresh_executed_scenarios(host)

    tree = host.executed_tree
    rows = {tree.set(iid, "scenario_name"): iid for iid in tree.get_children()}
    check = {name: tree.set(iid, "check") for name, iid in rows.items()}
    assert check == {"good": mw.CHECK_ON, "old": mw.CHECK_OFF, "bad": "✗", "live": "…", "csv": "–"}
    # The rescan dropped the rows whose latest run stopped succeeding from the saved checked set.
    assert settings.checked_executed_scenarios == ["1|good"]
    assert "run_failed" in tree.item(rows["bad"], "tags")
    # The name and timestamp columns are untouched (sorting and lookups use them).
    assert tree.set(rows["bad"], "timestamp") == "05.11.26 14:02"
    # Neither a click nor the space bar checks such a row.
    ctrl = CheckTreeController(tree, check_column="check", checked_glyph=mw.CHECK_ON,
                               unchecked_glyph=mw.CHECK_OFF, can_check=host._executed_row_checkable)
    tree.selection_set([rows["bad"], rows["old"]])
    ctrl.toggle_selected()
    assert tree.set(rows["bad"], "check") == "✗" and tree.set(rows["old"], "check") == mw.CHECK_ON
    # ▶ on such a row opens its run log, not the result viewer.
    (project / "output_parquet" / "bad_1" / RUN_LOG_FILE).write_text("# FlexTool run x\nModel status: Infeasible\n",
                                                                       encoding="utf-8")
    host.current_project = project.name
    host._open_run_log = MethodType(mw.MainWindow._open_run_log, host)
    host._view_scenario_plots = lambda *a: pytest.fail("the result viewer was opened for a failed run")
    monkeypatch.setattr(mw, "get_projects_dir", lambda: project.parent)
    mw.MainWindow._open_executed_row(host, rows["bad"])
    (window,) = [w for w in tk_root.winfo_children() if isinstance(w, tk.Toplevel)]
    text = window._text.get("1.0", "end")
    assert "latest run failed: solve for y2030 is infeasible" in text
    assert "Model status: Infeasible" in text
