"""The GUI owns its engine runs' ``_run.log`` and finishes a run record the engine could not finish.

The engine is ``tests/run_record/_stub_engine.py`` (it writes a ``running`` status, prints the marker like the
engine with a pipe owner, then exits or sleeps until it is killed).  No Tk: the manager is built without a window
and ``_run_job`` runs in the test's own thread or a plain thread.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from flextool.cli.cmd_run_flextool import build_parser
from flextool.gui import execution_manager as em
from flextool.gui.cli_format import format_cmd_for_log, terminal_command
from flextool.gui.data_models import ProjectSettings
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager, JobStatus, JobType
from flextool.run_record import MARKER, RUN_LOG_FILE, read_status_file

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
