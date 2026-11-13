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

import os
import socket
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
from flextool.run_record import MARKER, RUN_LOG_FILE, read_status_file, results_usable, write_status_atomic

STUB = Path(__file__).resolve().parents[1] / "run_record" / "_stub_engine.py"


@pytest.fixture
def manager(tmp_path, monkeypatch) -> ExecutionManager:
    monkeypatch.setattr(em, "_wrap_for_memory_cap", lambda cmd, estimate_gb: (list(cmd), {}, None))
    return ExecutionManager(project_path=tmp_path, settings=ProjectSettings())


def _job(subdir: str, status: JobStatus = JobStatus.RUNNING, job_id: int = 0) -> ExecutionJob:
    return ExecutionJob(job_id=job_id, job_type=JobType.SCENARIO, scenario_name=subdir, output_subdir=subdir,
                        status=status)


def _stub_job(manager: ExecutionManager, folder: Path, mode: str) -> ExecutionJob:
    job = _job(folder.name)  # the job's output folder is output_parquet/<folder name>
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


# ---------------------------------------------------------------------------
# The end of a job: who stopped it, the order of the record and the next job
# ---------------------------------------------------------------------------


def _wait_for(predicate, what, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.02)


def _start_sleeping_job(manager: ExecutionManager, folder: Path) -> tuple[ExecutionJob, threading.Thread]:
    """A ``sleep`` stub job in a plain thread, returned once its engine has written its record."""
    job = _stub_job(manager, folder, "sleep")
    job.source_name = "model.sqlite"
    worker = threading.Thread(target=manager._run_job, args=(job,), daemon=True)
    worker.start()
    _wait_for(lambda: job.run_record_folder is not None and "sleeping" in manager.get_stdout(job.job_id),
              "the stub engine did not start")
    return job, worker


def test_removing_the_source_records_the_kill_and_closing_waits_for_it(manager, tmp_path):
    folder = tmp_path / "output_parquet" / "scen"
    job, worker = _start_sleeping_job(manager, folder)

    manager.remove_jobs_for_source("model.sqlite")
    assert manager.get_jobs() == []
    # Closing FlexTool right away: cleanup waits for the worker of the job already taken off the list.
    manager.cleanup()

    record = read_status_file(folder)
    assert record["state"] == "killed"
    assert record["error"]["message"] == "its input source was removed"
    assert job.worker_done.is_set()
    worker.join(10)


def test_closing_flextool_finishes_the_record_before_cleanup_returns(manager, tmp_path):
    folder = tmp_path / "output_parquet" / "scen"
    job, _worker = _start_sleeping_job(manager, folder)

    manager.cleanup()  # no join: cleanup itself waits

    assert job.status is JobStatus.KILLED
    record = read_status_file(folder)
    assert (record["state"], record["error"]["message"]) == ("killed", "FlexTool was closed")


def test_cleanup_keeps_calling_its_progress_callback_while_it_waits(manager):
    pending = threading.Event()
    manager._live_waits.add(pending)
    calls: list[int] = []

    def progress() -> None:
        calls.append(1)
        pending.set()  # the job finishes after the first call

    manager.cleanup(progress=progress)

    assert calls == [1]


def test_the_list_is_refreshed_after_the_record_is_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "_wrap_for_memory_cap", lambda cmd, estimate_gb: (list(cmd), {}, None))
    folder = tmp_path / "output_parquet" / "scen"
    at_status: list[str] = []
    at_close: list[str] = []

    def on_status_change(job):
        if job.status is JobStatus.FAILED:
            at_status.append(read_status_file(folder)["state"])

    manager = ExecutionManager(project_path=tmp_path, settings=ProjectSettings(), on_status_change=on_status_change,
                               on_run_record_closed=lambda job: at_close.append(read_status_file(folder)["state"]))
    job = _stub_job(manager, folder, "exit1")

    manager._run_job(job)

    assert job.status is JobStatus.FAILED
    # The job's own notify comes before the record is finished; the second call comes after it.
    assert at_status == ["running"]
    assert at_close == ["failed"]


def test_a_folder_stays_busy_until_the_worker_has_closed_its_owner(manager, tmp_path, monkeypatch):
    folder = tmp_path / "output_parquet" / "a"
    first, second = _job("a", JobStatus.PENDING, 0), _job("a", JobStatus.PENDING, 1)
    manager._jobs = [first, second]
    manager._max_workers = 4
    manager._memory_admits = lambda candidate, source, estimate: True
    manager._build_run_command = lambda job, work_folder: [sys.executable, str(STUB), str(folder), "exit1"]
    closing, release = threading.Event(), threading.Event()
    real_close = em.RunLogOwner.close

    def slow_close(self, *args, **kwargs):
        closing.set()
        assert release.wait(60)
        return real_close(self, *args, **kwargs)

    monkeypatch.setattr(em.RunLogOwner, "close", slow_close)
    with manager._lock:
        assert manager._pick_next_pending() is first
    worker = threading.Thread(target=manager._run_job, args=(first,), daemon=True)
    worker.start()
    assert closing.wait(60)
    assert first.status is JobStatus.FAILED

    with manager._lock:
        assert manager._pick_next_pending() is None  # the first job's record is not finished yet
    release.set()
    worker.join(60)
    with manager._lock:
        assert manager._pick_next_pending() is second


def test_a_failure_while_finishing_the_record_still_frees_the_folder(manager, tmp_path, monkeypatch):
    folder = tmp_path / "output_parquet" / "a"
    first, second = _job("a", JobStatus.PENDING, 0), _job("a", JobStatus.PENDING, 1)
    manager._jobs = [first, second]
    manager._max_workers = 4
    manager._memory_admits = lambda candidate, source, estimate: True
    manager._build_run_command = lambda job, work_folder: [sys.executable, str(STUB), str(folder), "exit1"]

    def broken_close(self, *args, **kwargs):
        raise ValueError("a bug while finishing the record")

    monkeypatch.setattr(em.RunLogOwner, "close", broken_close)
    with manager._lock:
        assert manager._pick_next_pending() is first
    manager._run_job(first)

    assert first.worker_done.is_set()
    assert manager._running_count == 0
    with manager._lock:
        assert manager._pick_next_pending() is second


def test_a_job_stopped_between_its_pick_and_its_start_never_starts(manager, monkeypatch):
    job = _job("a", JobStatus.PENDING, 0)
    manager._jobs = [job]
    with manager._lock:
        assert manager._pick_next_pending() is job
    manager.kill_job(job.job_id)

    def no_popen(*args, **kwargs):
        raise AssertionError("the engine of a stopped job was started")

    monkeypatch.setattr(em.subprocess, "Popen", no_popen)
    manager._run_job(job)

    assert job.status is JobStatus.KILLED
    assert job.kill_reason == "stopped from the FlexTool window"
    assert manager._running_count == 0 and job.worker_done.is_set()


# ---------------------------------------------------------------------------
# A run that ends before the engine starts its record
# ---------------------------------------------------------------------------


def test_a_job_failing_before_its_record_withdraws_the_old_results(manager, tmp_path):
    folder = _run_folder(tmp_path, "scen", state="succeeded")  # run "run-scen" wrote its parquet
    job = _stub_job(manager, folder, "norecord-exit1")

    manager._run_job(job)

    assert job.status is JobStatus.FAILED
    record = read_status_file(folder)
    assert record["state"] == "failed"
    assert record["error"] == {"type": "ProcessEnded",
                               "message": "the run ended before it started its run record (exit code 1)"}
    assert not results_usable(record)
    assert record["outputs"]["parquet_state"] == "previous_run"
    assert record["outputs"]["previous"]["run_id"] == "run-scen"
    lines = (folder / RUN_LOG_FILE).read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# FlexTool run {record['run_id']} started {record['started_at']}"
    assert "early error" in lines
    assert lines[-1] == "# gui: the run ended before it started its run record (exit code 1): failed"


def test_a_job_killed_before_its_record_withdraws_the_old_results(manager, tmp_path):
    folder = _run_folder(tmp_path, "scen", state="succeeded")
    job = _stub_job(manager, folder, "norecord-sleep")
    worker = threading.Thread(target=manager._run_job, args=(job,), daemon=True)
    worker.start()
    _wait_for(lambda: "waiting" in manager.get_stdout(job.job_id), "the stub engine did not start")

    manager.kill_job(job.job_id)
    worker.join(60)

    record = read_status_file(folder)
    assert (record["state"], record["error"]["message"]) == ("killed", "stopped from the FlexTool window")
    assert record["outputs"]["previous"]["run_id"] == "run-scen"


def test_a_run_stopped_before_its_marker_is_finished_as_killed(manager, tmp_path):
    folder = tmp_path / "output_parquet" / "scen"
    job = _stub_job(manager, folder, "nomarker-sleep")
    worker = threading.Thread(target=manager._run_job, args=(job,), daemon=True)
    worker.start()
    _wait_for(lambda: "sleeping" in manager.get_stdout(job.job_id), "the stub engine did not start")
    run_id = read_status_file(folder)["run_id"]

    manager.kill_job(job.job_id)
    worker.join(60)

    record = read_status_file(folder)
    assert (record["run_id"], record["state"]) == (run_id, "killed")
    assert [s["state"] for s in record["solves"]] == ["error"]


def test_a_minimal_record_replaces_only_an_older_finished_record(manager, tmp_path):
    # A folder without a record but with parquet files (an older FlexTool): withdrawn, the parquet's run unknown.
    legacy = _run_folder(tmp_path, "legacy", state=None)
    manager._run_job(_stub_job(manager, legacy, "norecord-exit1"))
    record = read_status_file(legacy)
    assert record["state"] == "failed"
    assert (record["outputs"]["parquet_state"], record["outputs"]["previous"]["run_id"]) == ("previous_run", None)

    # (a) A run that started after the launch and finished its own record (here: a stub with a lost marker).
    folder = _run_folder(tmp_path, "scen", state="succeeded")
    manager._run_job(_stub_job(manager, folder, "foreign-done"))
    record = read_status_file(folder)
    assert record["state"] == "succeeded" and record["run_id"] != "run-scen"

    # (b) Another run, still running, owns the folder.
    live = _run_folder(tmp_path, "live", state="running")
    write_status_atomic(live, {**read_status_file(live), "process": {"pid": os.getpid(), "host": socket.gethostname(),
                                                                      "platform": sys.platform}})
    before = (live / "_run_status.json").read_bytes()
    manager._run_job(_stub_job(manager, live, "norecord-exit1"))
    assert (live / "_run_status.json").read_bytes() == before

    # (c) No folder: none is created.
    missing = tmp_path / "output_parquet" / "missing"
    manager._run_job(_stub_job(manager, missing, "norecord-exit1"))
    assert not missing.exists()


def test_the_gui_command_names_its_launcher_and_the_spreadsheet(manager, tmp_path):
    xlsx = tmp_path / "input_sources" / "model.xlsx"
    job = ExecutionJob(job_id=0, scenario_name="scen", output_subdir="scen_1", is_xlsx_source=True, xlsx_path=xlsx,
                       input_db_url="sqlite:///" + str(tmp_path / "intermediate" / "model.sqlite"))
    cmd = manager._build_run_command(job, tmp_path / "work")

    assert cmd[1:4] == ["-P", "-m", "flextool.cli.cmd_run_flextool"]
    parsed = build_parser().parse_args(cmd[4:])
    assert parsed.run_launcher == "gui"
    assert parsed.input_source_file == str(xlsx)
    # What a user copies into a terminal: the same run under flextool-run, which records its own launcher.
    copy = terminal_command(cmd)
    assert copy[0] == "flextool-run"
    assert build_parser().parse_args(copy[1:]).run_launcher == "direct"
    assert copy[1:] == [tok for tok in cmd[4:] if tok not in ("--run-launcher", "gui")]


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


def test_the_post_batch_comparison_leaves_out_a_success_without_new_parquet(manager, monkeypatch):
    # a exited 0 but wrote no parquet (its folder keeps an earlier run's files); b wrote its own; c has no run
    # record (an older FlexTool) and stays in.
    _run_folder(manager.project_path, "a", state="succeeded", parquet_state="skipped")
    _run_folder(manager.project_path, "b", state="succeeded")
    _run_folder(manager.project_path, "c", state=None)
    manager.settings.auto_generate_comp_plots = True
    manager.settings.auto_generate_comp_excel = False
    manager._jobs = [_job(s, JobStatus.SUCCESS, i) for i, s in enumerate("abc", start=10)]
    commands: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            commands.append(list(cmd))
            self.stdout = iter(())
            self.pid = 0

        def wait(self):
            return 0

    monkeypatch.setattr(em.subprocess, "Popen", FakePopen)
    manager._run_post_execution()

    (cmd,) = commands
    assert cmd[cmd.index("--alternatives") + 1:cmd.index("--plot-dir")] == ["b", "c"]
    # Told in the Auto-comparison job's output; a stays among the folders a later batch checks again.
    (aux,) = [j for j in manager._jobs if j.job_type is JobType.OUTPUT_ACTION]
    assert aux.stdout_lines[0] == "Left out a: the run wrote no parquet"
    assert manager._comparison_scenarios == ["a", "b", "c"]


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    yield root
    root.destroy()


def _executed_list_host(tk_root, project: Path, settings: ProjectSettings):
    """A bare Tk root with the ``MainWindow`` methods and state the executed list needs."""
    host = tk_root
    host.executed_tree = ttk.Treeview(
        tk_root, columns=("check", "source_num", "scenario_name", "view", "timestamp"), show="headings",
    )
    host.project_settings = settings
    host.exec_scenario_mgr = ExecutedScenarioManager(project, settings)
    host._executed_sort_mode = "timestamp"
    host._executed_row_info = {}
    host._unreadable_rescan_id = None
    host._unreadable_rescans_left = mw.UNREADABLE_RESCANS
    host._sort_tree_items = MethodType(mw.MainWindow._sort_tree_items, host)
    host._update_output_status = lambda: None
    host._save_checked_executed_scenarios = MethodType(mw.MainWindow._collect_checked_executed_scenarios, host)
    host._executed_row_checkable = MethodType(mw.MainWindow._executed_row_checkable, host)
    host._refresh_executed_scenarios = MethodType(mw.MainWindow._refresh_executed_scenarios, host)
    host._rescan_unreadable_rows = MethodType(mw.MainWindow._rescan_unreadable_rows, host)
    return host


def test_the_executed_list_unchecks_and_refuses_rows_whose_run_did_not_succeed(project, tk_root, monkeypatch):
    settings = ProjectSettings()
    settings.checked_executed_scenarios = ["1|bad", "1|good", "1|live"]
    host = _executed_list_host(tk_root, project, settings)

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


def test_the_gui_shows_times_in_one_format():
    from flextool.gui.data_models import TIMESTAMP_FORMAT
    from flextool.gui.dialogs.file_picker import _format_mtime

    assert TIMESTAMP_FORMAT == "%d.%m.%y %H:%M"
    assert _format_mtime(datetime(2026, 11, 5, 14, 2).timestamp()) == "05.11.26 14:02"


def test_the_run_log_window_shows_a_long_log_as_its_head_and_tail(project, tk_root, monkeypatch):
    from flextool.run_record import _owner

    log_path = project / "output_parquet" / "bad_1" / RUN_LOG_FILE
    # 2000 lines of 10 bytes (20 000 bytes); the window shows at most 1000: the first 500 (head, half of it) and
    # the last 500, so 19 000 bytes are left out.
    log_path.write_bytes(b"".join(f"line {i:04d}\n".encode() for i in range(2000)))
    monkeypatch.setattr(_owner, "LOG_VIEW_MAX_BYTES", 1000)
    host = _executed_list_host(tk_root, project, ProjectSettings())
    mw.MainWindow._refresh_executed_scenarios(host)
    rows = {host.executed_tree.set(iid, "scenario_name"): iid for iid in host.executed_tree.get_children()}
    host.current_project = project.name
    monkeypatch.setattr(mw, "get_projects_dir", lambda: project.parent)
    mw.MainWindow._open_run_log(host, rows["bad"])
    (window,) = [w for w in tk_root.winfo_children() if isinstance(w, tk.Toplevel)]
    lines = window._text.get("1.0", "end").splitlines()
    shown = [line for line in lines if line.startswith("line ")]
    assert shown == [f"line {i:04d}" for i in (*range(50), *range(1950, 2000))]
    assert f"[... 19000 bytes not shown here; the whole log is {log_path} ...]" in lines


# ---------------------------------------------------------------------------
# A run record that cannot be read now, or is damaged
# ---------------------------------------------------------------------------


def test_scan_survives_a_malformed_or_unreadable_status(project):
    bad_shape = _run_folder(project, "shape_1", state="succeeded")
    write_status_atomic(bad_shape, {"run_id": "x", "state": "succeeded", "outputs": [1]})
    cut = _run_folder(project, "cut_1", state="failed", parquet_state="previous_run")
    (cut / "_run_status.json").write_text('{"state": "failed", "outputs": {"parq', encoding="utf-8")

    rows = {info.name: info for info in ExecutedScenarioManager(project).scan_executed()}

    assert set(rows) == {"good", "bad", "old", "live", "csv", "shape", "cut"}
    assert {name for name, info in rows.items() if info.checkable} == {"good", "old"}
    # Wrong shape: damaged (a cross).  Cut short just now: probably being written (an ellipsis).
    assert (rows["shape"].run_state, mw.row_check_glyph(rows["shape"], False)) == ("malformed", "\u2717")
    assert (rows["cut"].run_state, mw.row_check_glyph(rows["cut"], False)) == ("unreadable", "\u2026")
    assert rows["cut"].unusable_reason == ("the run record could not be read (it may be being written); try again "
                                           "in a moment")


def test_an_unreadable_row_is_scanned_again_at_most_three_times(project, tk_root):
    cut = _run_folder(project, "cut_1", state="failed", parquet_state="previous_run")
    status = cut / "_run_status.json"
    status.write_text('{"state": "failed", "outputs": {"parq', encoding="utf-8")
    host = _executed_list_host(tk_root, project, ProjectSettings())
    scheduled: list[tuple[int, object]] = []
    host.after = lambda ms, fn: scheduled.append((ms, fn)) or f"after#{len(scheduled)}"

    host._refresh_executed_scenarios()
    assert [ms for ms, _ in scheduled] == [1500]
    host._refresh_executed_scenarios()          # another refresh while one is pending: no second rescan
    assert len(scheduled) == 1
    for _ in range(3):
        scheduled[-1][1]()                      # the rescan runs; the record still does not read
    assert len(scheduled) == 3                  # three rescans, then the row is left as it is
    # Once the record reads, the count starts again.
    write_status_atomic(cut, {"run_id": "r", "state": "failed", "outputs": {"parquet_state": "previous_run"}})
    host._refresh_executed_scenarios()
    tree = host.executed_tree
    rows = {tree.set(iid, "scenario_name"): iid for iid in tree.get_children()}
    assert tree.set(rows["cut"], "check") == "\u2717"
    assert host._unreadable_rescans_left == mw.UNREADABLE_RESCANS


def test_closing_with_jobs_hides_the_window_and_keeps_its_event_loop_going(tk_root):
    tk_root.deiconify()
    seen: list[tuple[str, bool]] = []

    class FakeManager:
        def has_pending_or_running(self):
            return True

        def cleanup(self, progress=None):
            # While cleanup waits: the main window is hidden, a note is shown, and events are processed.
            fired: list[int] = []
            tk_root.after(0, lambda: fired.append(1))
            progress()
            notes = [w for w in tk_root.winfo_children() if isinstance(w, tk.Toplevel)]
            seen.append((tk_root.state(), bool(notes) and bool(fired)))

    tk_root.execution_mgr = FakeManager()
    mw.MainWindow._cleanup_execution_jobs(tk_root)

    assert seen == [("withdrawn", True)]
    assert not [w for w in tk_root.winfo_children() if isinstance(w, tk.Toplevel)]
