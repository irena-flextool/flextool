"""The runtime context file and the live ``get_gui_context`` of an AI agent (P2 §13.1 test 2, §6.4, §6.5).

A writer's file round-trips (UTF-8 with non-ASCII kept, LF, no absolute path); of two windows' files (two
processes) the reader takes the newest live one; a file of a gone process and one older than 120 s are ignored;
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
    data = json.loads(path.read_text(encoding="utf-8"))
    data["written_at"] = iso(moment)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


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
        assert wait_for(lambda: writer.path.stat().st_mtime_ns != first)
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
        assert read_runtime_context(project, now=now) == (None, "not running")
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
