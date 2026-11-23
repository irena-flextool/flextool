"""The Add dialog's FlexTool 2.0 conversion job moves the converted database into input_sources.

Until 4.6.3 the job's worker called ``_os.replace``, a name bound only in the no-execution-manager branch of the same
method, so in the GUI (which always has an execution manager) every successful conversion ended with a NameError:
the job failed and the converted database, still in ``intermediate/``, was deleted.  Tk-free: the method runs
unbound on a stand-in host, the importer is a stand-in process and the worker thread runs at once.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from flextool.gui.dialogs import add_dialog


class _Manager:
    """The parts of ExecutionManager the conversion job uses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.finished: list[tuple[str, bool]] = []
        self.lines: list[str] = []

    def add_auxiliary_job(self, _type, _label, _key):
        return SimpleNamespace(job_id="job1", process=None)

    def append_stdout(self, _job_id: str, line: str) -> None:
        self.lines.append(line)

    def finish_job(self, job_id: str, ok: bool) -> None:
        self.finished.append((job_id, ok))


class _Thread:
    """Runs the worker at start(), so the test sees its result at once."""

    def __init__(self, target, daemon: bool = False) -> None:
        self._target = target

    def start(self) -> None:
        self._target()


def test_converted_database_is_moved_into_input_sources(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    input_dir = project / "input_sources"
    input_dir.mkdir(parents=True)
    source = tmp_path / "old_model.xlsm"
    source.write_bytes(b"ft2")
    work = project / "intermediate" / "old_model.ft2import.sqlite"
    converted = b"converted database"

    class _Process:
        """The importer: it writes the working database and exits with 0."""

        def __init__(self, cmd, **_kwargs) -> None:
            assert cmd[-1].endswith("old_model.ft2import.sqlite")
            work.write_bytes(converted)
            self.stdout = iter(["imported\n"])
            self.returncode = 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(add_dialog, "FilePickerDialog", lambda *a, **k: SimpleNamespace(result=str(source)))
    monkeypatch.setattr(add_dialog, "ask_old_flextool_master", lambda *a, **k: None)   # no sensitivities
    monkeypatch.setattr(add_dialog.subprocess, "Popen", _Process)
    monkeypatch.setattr(add_dialog.threading, "Thread", _Thread)

    scheduled: list = []
    main_window = SimpleNamespace(
        _open_or_raise_execution_window=lambda: None, execution_window=None,
        after=lambda _ms, fn: scheduled.append(fn),
    )
    manager = _Manager()
    host = SimpleNamespace(
        _flextool_root=tmp_path, _input_dir=input_dir, _project_path=project, _input_source_mgr=None,
        _execution_mgr=manager, master=main_window, result=False, old_convert_started=False,
    )

    add_dialog.AddDialog._on_convert_old_flextool(host)

    # The job succeeded; the database the importer wrote is now input_sources/old_model.sqlite.
    assert manager.finished == [("job1", True)]
    assert (input_dir / "old_model.sqlite").read_bytes() == converted
    assert not work.exists()
    assert len(scheduled) == 1          # the new source is refreshed and ticked on the Tk thread
    assert host.result and host.old_convert_started
