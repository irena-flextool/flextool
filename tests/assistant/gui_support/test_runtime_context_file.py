"""The runtime context file and the live ``get_gui_context`` of an AI agent (P2 §13.1 test 2, §6.4, §6.5).

A writer's file round-trips (UTF-8 with non-ASCII kept, LF, no absolute path); of two windows' files (two
processes) the reader takes the window in use (focus, then the latest change; files without those by the newest
write); a file of a gone process and one older than 120 s are ignored (a live process's stale file reads as a
window that is open but not answering); the heartbeat writes only while the window's poll ticks;
``stop()`` leaves no file even with a write under way; ``cleanup()`` removes the files of gone processes; and
``flextool-assistant --call get_gui_context`` (run in this process) answers with what the window published, or
with the startup options when no window is open.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flextool.assistant import _files, cli, runtime_context
from flextool.assistant.context import GuiState, ResultRef, gui_snapshot
from flextool.assistant.runtime_context import RuntimeContextWriter, cleanup, read_runtime_context, runtime_path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "Vetyprojekti ä€"
    path.mkdir()
    return path


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot(project: Path, folder: str = "wind_1"):
    state = GuiState(sources_checked=(("base.sqlite", 1),), source_selected=None,
                     scenarios_checked=(("base.sqlite", "tuuli"),), scenario_selected=None,
                     results_checked=(ResultRef(folder, 1, "tuuli", "failed", False, "failed",
                                                f"latest run failed: see {project}/work/log"),),
                     results_selected=(), viewer=None)
    return gui_snapshot(project.name, state, taken_at="2026-11-07T10:00:00Z")


def wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def set_written_at(path: Path, moment: datetime) -> None:
    """Set the file's ``written_at``, as an older FlexTool wrote it (no ``changed_at`` / ``active_at``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["written_at"] = iso(moment)
    data.pop("changed_at", None)
    data.pop("active_at", None)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class Clock:
    """A writer's clock, stepped by the test."""

    def __init__(self, moment: datetime) -> None:
        self.now = moment

    def __call__(self) -> datetime:
        return self.now


def test_round_trip_utf8_lf_and_no_absolute_path(project, tmp_path):
    writer = RuntimeContextWriter(project, project.name, heartbeat_s=3600)
    writer.start()
    try:
        writer.publish_snapshot(snapshot(project))
        assert wait_for(lambda: writer.path.exists() and b"wind_1" in writer.path.read_bytes())
        raw = writer.path.read_bytes()
        text = raw.decode("utf-8")
        assert b"\r\n" not in raw and "Vetyprojekti ä€" in text
        assert str(tmp_path) not in text and "<project>/work/log" in text
        assert writer.path.name == f"{_files.project_key(project)}.{os.getpid()}.json"
        data = json.loads(text)
        assert (data["format"], data["pid"], data["project"]) == (
            1, os.getpid(), {"key": _files.project_key(project), "name": "Vetyprojekti ä€"})
        context, running = read_runtime_context(project)
        assert running == "running"
        assert context["written_at"] == data["written_at"] and context["gui"] == "running"
        assert context["input_source"] == "base.sqlite" and context["scenario"] == "tuuli"
        assert context["results_checked"] == [{
            "folder": "wind_1", "source_number": 1, "scenario": "tuuli", "state": "failed",
            "results_usable": False, "unusable_code": "failed", "reason": "latest run failed: see <project>/work/log"}]
    finally:
        assert writer.stop()
    assert not writer.path.exists()
    assert read_runtime_context(project) == (None, "not running")


def test_the_file_is_written_at_start_and_by_the_heartbeat(project):
    writer = RuntimeContextWriter(project, project.name, heartbeat_s=0.05)
    writer.start()
    try:
        assert wait_for(writer.path.exists)
        context, running = read_runtime_context(project)
        assert running == "running" and context["project"] == project.name and context["input_source"] is None
        first = writer.path.stat().st_mtime_ns
        assert wait_for(lambda: (writer.tick(), writer.path.stat().st_mtime_ns != first)[1])    # the poll ticks
    finally:
        assert writer.stop()


def test_the_newest_live_file_wins_and_gone_or_stale_ones_are_ignored(project):
    mine = RuntimeContextWriter(project, project.name, heartbeat_s=3600)
    other = RuntimeContextWriter(project, project.name, heartbeat_s=3600, pid=os.getppid())   # another window
    for writer, folder in ((mine, "mine_1"), (other, "other_1")):
        writer.start()
        writer.publish_snapshot(snapshot(project, folder))
    try:
        for writer, folder in ((mine, "mine_1"), (other, "other_1")):
            assert wait_for(lambda w=writer, f=folder: w.path.exists() and f.encode() in w.path.read_bytes())
        now = datetime.now(timezone.utc)
        set_written_at(other.path, now - timedelta(seconds=10))
        set_written_at(mine.path, now - timedelta(seconds=5))
        assert read_runtime_context(project, now=now)[0]["results_checked"][0]["folder"] == "mine_1"
        set_written_at(mine.path, now - timedelta(seconds=20))
        assert read_runtime_context(project, now=now)[0]["results_checked"][0]["folder"] == "other_1"
        gone = runtime_path(project, pid=dead_pid())
        gone.write_text(mine.path.read_text(encoding="utf-8").replace("mine_1", "gone_1"), encoding="utf-8")
        set_written_at(gone, now)
        assert read_runtime_context(project, now=now)[0]["results_checked"][0]["folder"] == "other_1"
        set_written_at(other.path, now - timedelta(seconds=121))
        set_written_at(mine.path, now - timedelta(seconds=121))
        # Both processes live but their files stale: open, not answering (review G18 critique m4).
        assert read_runtime_context(project, now=now) == (None, "running")
        assert read_runtime_context(project.parent, now=now) == (None, "not running")    # another project
        assert cleanup() == 1
        assert not gone.exists() and mine.path.exists() and other.path.exists()
    finally:
        assert mine.stop() and other.stop()
    assert not mine.path.exists() and not other.path.exists()


def test_stop_leaves_no_file_even_with_a_write_under_way(project, monkeypatch):
    writer = RuntimeContextWriter(project, project.name, heartbeat_s=3600)
    entered, release = threading.Event(), threading.Event()
    real = _files.write_text_atomic

    def slow(path, text, **kwargs):
        if "slow_1" in text:
            entered.set()
            release.wait(10)
        return real(path, text, **kwargs)

    monkeypatch.setattr(_files, "write_text_atomic", slow)
    writer.start()
    assert wait_for(writer.path.exists)
    writer.publish_snapshot(snapshot(project, "slow_1"))
    assert entered.wait(5)
    assert writer.stop(wait=0.1) is False          # the write under way is finished first
    release.set()
    assert writer.stop(wait=5)
    assert not writer.path.exists()
    assert [p for p in runtime_context.runtime_dir().iterdir()] == []      # no temp file either


def test_get_gui_context_from_the_command_line_reads_the_window(project, capsys):
    argv = ["--call", "get_gui_context", "--project-path", str(project)]
    assert cli.main(argv) == 0
    closed = json.loads(capsys.readouterr().out)
    assert closed["gui"] == "not running" and closed["front_end"] == "agent"
    assert closed["note"] == "The FlexTool window is not open for this project; showing the startup options."
    writer = RuntimeContextWriter(project, project.name, heartbeat_s=3600)
    writer.start()
    try:
        writer.publish_snapshot(snapshot(project))
        assert wait_for(lambda: writer.path.exists() and b"wind_1" in writer.path.read_bytes())
        assert cli.main(argv) == 0
        live = json.loads(capsys.readouterr().out)
    finally:
        assert writer.stop()
    assert live["gui"] == "running" and live["written_at"].endswith("Z")
    assert (live["project"], live["input_source"], live["scenario"]) == ("Vetyprojekti ä€", "base.sqlite", "tuuli")
    assert live["results_checked"][0]["results_usable"] is False
    assert live["results_checked"][0]["reason"] == "latest run failed: see <project>/work/log"
    assert live["scenarios_checked"] == [{"source": "base.sqlite", "scenario": "tuuli"}]
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["gui"] == "not running"


T0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)


def folder_of(context: dict | None) -> str | None:
    return None if context is None else context["results_checked"][0]["folder"]


def test_the_window_in_use_wins_over_a_later_heartbeat(project):
    """Review G18, C3-m5 and critique m3: A's selections changed at 10:00:00, B only heartbeated at 10:00:30 —
    A is the window in use (before: B, the newest write).  Then B gains the focus at 10:00:35: B."""
    clock_a, clock_b = Clock(T0 - timedelta(seconds=30)), Clock(T0 - timedelta(seconds=60))
    a = RuntimeContextWriter(project, project.name, heartbeat_s=3600, clock=clock_a)
    b = RuntimeContextWriter(project, project.name, heartbeat_s=0.05, pid=os.getppid(), clock=clock_b)
    try:
        b.start()
        b.publish_snapshot(snapshot(project, "b_1"))
        assert wait_for(lambda: b.path.exists() and b"b_1" in b.path.read_bytes())
        a.start()
        assert wait_for(lambda: a.writes >= 1)
        clock_a.now = T0
        a.publish_snapshot(snapshot(project, "a_1"))
        assert wait_for(lambda: a.path.exists() and b"a_1" in a.path.read_bytes())
        clock_b.now = T0 + timedelta(seconds=30)
        written = b.writes
        b.tick(active=False)
        assert wait_for(lambda: b.writes > written)
        data_b = json.loads(b.path.read_text(encoding="utf-8"))
        assert data_b["written_at"] == "2026-11-07T10:00:30Z" and data_b["changed_at"] == "2026-11-07T09:59:00Z"
        context, running = read_runtime_context(project, now=T0 + timedelta(seconds=40))
        assert running == "running" and folder_of(context) == "a_1"
        assert context["changed_at"] == "2026-11-07T10:00:00Z" and context["responding"] is True
        clock_b.now = T0 + timedelta(seconds=35)
        written = b.writes
        b.tick(active=True)                         # gaining the focus is written at once
        assert wait_for(lambda: b.writes > written)
        context, _running = read_runtime_context(project, now=T0 + timedelta(seconds=40))
        assert folder_of(context) == "b_1" and context["active_at"] == "2026-11-07T10:00:35Z"
    finally:
        assert a.stop() and b.stop()


def test_a_window_whose_poll_stops_is_not_heartbeated(project):
    """Review G18, C3-m5: the heartbeat writes only when the window's poll ticked since the last write, so a hung
    Tk thread's file goes stale (before: the writer's own thread kept it fresh)."""
    writer = RuntimeContextWriter(project, project.name, heartbeat_s=0.05)
    writer.start()
    try:
        assert wait_for(lambda: writer.heartbeats >= 3)
        assert writer.writes == 1                   # the start only
        writer.tick()
        assert wait_for(lambda: writer.writes == 2)
        beats = writer.heartbeats
        assert wait_for(lambda: writer.heartbeats >= beats + 3)
        assert writer.writes == 2
    finally:
        assert writer.stop()


def test_a_stale_file_of_a_live_window_says_busy_to_the_agent(project, capsys):
    """Review G18 critique m4: a window whose Tk thread has not ticked for over 120 s (a long operation) is not
    "not running": the agent gets the startup options with ``gui: running``, ``responding: false``."""
    writer = RuntimeContextWriter(project, project.name, heartbeat_s=3600, pid=os.getppid())
    writer.start()
    try:
        assert wait_for(writer.path.exists)
        set_written_at(writer.path, datetime.now(timezone.utc) - timedelta(seconds=121))
        assert read_runtime_context(project) == (None, "running")
        assert cli.main(["--call", "get_gui_context", "--project-path", str(project)]) == 0
        busy = json.loads(capsys.readouterr().out)
    finally:
        assert writer.stop()
    assert (busy["gui"], busy["responding"]) == ("running", False)
    assert busy["note"] == ("The FlexTool window is open for this project but has not answered for over two "
                            "minutes (it may be busy with a long operation); showing the startup options.")


def test_the_reader_orders_by_focus_then_change_then_write(project):
    """Hand-written files of two live windows: the latest ``active_at`` wins, then ``changed_at``, then
    ``written_at``; a missing field counts as earliest (before: always the newest ``written_at``)."""
    folder = runtime_context.runtime_dir()
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(runtime_context.snapshot_payload(snapshot(project), gui_running="running")))

    def put(pid: int, name: str, written: int, changed: int | None, active: int | None) -> None:
        context = json.loads(json.dumps(payload).replace("wind_1", name))
        data = {"format": 1, "pid": pid, "written_at": iso(T0 + timedelta(seconds=written)),
                "flextool_version": "x", "project": {"key": _files.project_key(project), "name": project.name},
                "context": context}
        for field, value in (("changed_at", changed), ("active_at", active)):
            if value is not None:
                data[field] = iso(T0 + timedelta(seconds=value))
        runtime_path(project, pid=pid).write_text(json.dumps(data), encoding="utf-8")

    now = T0 + timedelta(seconds=60)
    put(os.getpid(), "a_1", written=0, changed=0, active=None)
    put(os.getppid(), "b_1", written=30, changed=-60, active=None)
    assert folder_of(read_runtime_context(project, now=now)[0]) == "a_1"         # changed later
    put(os.getppid(), "b_1", written=30, changed=-60, active=20)
    assert folder_of(read_runtime_context(project, now=now)[0]) == "b_1"         # had the focus later
    put(os.getpid(), "a_1", written=0, changed=None, active=None)
    put(os.getppid(), "b_1", written=30, changed=None, active=None)
    assert folder_of(read_runtime_context(project, now=now)[0]) == "b_1"         # an older FlexTool's files
