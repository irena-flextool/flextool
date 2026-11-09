"""The assistant's store, lock and ledger files when the file system misbehaves (review fix group G12).

Windows is simulated on Linux: ``_files._WINDOWS`` is set and ``os.unlink`` / ``os.open`` raise ``PermissionError``
(a sharing violation, or a delete still pending) for one path while a test says so.  ``_files._sleep`` and
``_files._clock`` are replaced by a fake clock that the waits advance, so nothing here depends on how fast the
machine is.  Where a lock must hold between processes the other owner is a real process (``spawn``), waited for
with bounded timeouts.  Expected values are worked out by hand from the docstrings of ``_files``, ``store`` and
``usage``, never read back from the code under test.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from flextool.assistant import _files, settings, store as store_module
from flextool.assistant.llm_adapter import Message
from flextool.assistant.store import Store
from tests.assistant import store_workers

ORIGIN = ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "model-a")
SPAWN = multiprocessing.get_context("spawn")


@pytest.fixture
def root(user_config_in_tmp: Path) -> Path:
    return user_config_in_tmp


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "Projekti"
    path.mkdir()
    return path


@pytest.fixture
def st(root: Path) -> Store:
    return Store(root)


def set_expiry(root: Path, days: int) -> None:
    settings.save_settings(settings.Settings(conversation_expiry_days=days), root)


def saved(st: Store, project: Path, text: str = "Miksi?", number: int | None = None) -> store_module.Conversation:
    conv = st.new_conversation(project, ORIGIN, "standard", number=number)
    conv.messages.append(Message("user", text))
    assert st.save(conv)
    return conv


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class FakeClock:
    """``_files._clock`` and ``_files._sleep``: every wait is recorded and moves the clock on by itself."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Windows sharing rules in ``_files``, with the fake clock; the background retry thread is not started (the
    tests run the retry themselves), and nothing a test leaves pending reaches the next test."""
    fake = FakeClock()
    for name, value in (("_WINDOWS", True), ("_sleep", fake.sleep), ("_clock", fake.clock),
                        ("_start_retry_thread", lambda: None)):
        monkeypatch.setattr(_files, name, value, raising=False)
    yield fake
    pending = getattr(_files, "_PENDING", None)
    if pending is not None:
        with _files._PENDING_GUARD:
            pending.clear()


class Refuse:
    """``os.<name>`` raising ``PermissionError`` on *path* while ``times`` is above 0 (``None``: until ``stop``);
    *flags_with*: only for an ``os.open`` whose flags include these.  Every other call goes to the real function."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str, path: Path, times: int | None = None, *,
                 flags_with: int = 0, before=None) -> None:
        self.real = getattr(os, name)
        self.path = os.path.normcase(os.fspath(path))
        self.times = times
        self.flags_with = flags_with
        self.before = before            # called once, at the first refused call (a swap by another owner)
        self.refused = 0
        self.active = True
        monkeypatch.setattr(os, name, self)

    def stop(self) -> None:
        self.active = False

    def __call__(self, target, *args, **kwargs):
        flags = args[0] if args else kwargs.get("flags", 0)
        if (self.active and os.path.normcase(os.fspath(target)) == self.path
                and (not self.flags_with or (isinstance(flags, int) and flags & self.flags_with == self.flags_with))
                and (self.times is None or self.times > 0)):
            if self.before is not None:
                before, self.before = self.before, None
                before()
            if self.times is not None:
                self.times -= 1
            self.refused += 1
            raise PermissionError(13, "The process cannot access the file because it is being used", str(target))
        return self.real(target, *args, **kwargs)


def lock_data(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_old(path: Path, seconds: float = 120.0) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


# --- C5-2: Windows PermissionError on lock delete and create ------------------------------------------------------
def test_a_release_retries_a_sharing_violation(tmp_path, monkeypatch, windows):
    lock = _files.PidLock(tmp_path / "usage.json.lock", stale_after=30)
    assert lock.try_acquire()
    Refuse(monkeypatch, "unlink", lock.path, times=2)
    lock.release()
    assert not lock.path.exists()
    assert windows.waits == [0.01, 0.02]
    assert not (tmp_path / "usage.json.lock.break").exists()


def test_a_lock_that_cannot_be_deleted_is_marked_released_and_others_take_it(tmp_path, monkeypatch, windows):
    lock = _files.PidLock(tmp_path / "projects.json.lock", stale_after=30)
    assert lock.try_acquire()
    refuse = Refuse(monkeypatch, "unlink", lock.path)
    lock.release()                              # never raises
    owner, _raw = _files.read_owner(lock.path)
    assert owner is not None and owner.released and owner.pid == os.getpid()
    assert not _files._PENDING                  # the mark was written: nothing to retry
    refuse.stop()
    other = _files.PidLock(lock.path, stale_after=30)
    assert other.try_acquire()                  # at once, although the marked lock is fresh and its process runs
    assert lock_data(lock.path)["token"] == other.token
    other.release()
    assert not lock.path.exists()


def test_a_conversation_closed_that_way_is_not_shown_as_open(st, root, project, monkeypatch, windows):
    conv = saved(st, project)
    lock_path = st.project_dir(conv.project_key) / f"{conv.id}.json.lock"
    refuse = Refuse(monkeypatch, "unlink", lock_path)
    st.close(conv)
    assert lock_path.exists()
    assert [i.locked for i in Store(root).list_conversations(project)] == [None]
    refuse.stop()
    again = st.open(project, conv.id, if_locked="raise")
    assert again.id == conv.id
    st.close(again)
    assert not lock_path.exists()


def test_a_lock_being_deleted_counts_as_taken_on_windows_only(tmp_path, monkeypatch, windows):
    path = tmp_path / "usage.json.lock"
    Refuse(monkeypatch, "open", path, flags_with=os.O_CREAT | os.O_EXCL)
    lock = _files.PidLock(path, stale_after=30)
    assert lock.try_acquire() is False
    assert lock.token is None
    with pytest.raises(_files.LockTimeout):
        lock.acquire(0.05)
    monkeypatch.setattr(_files, "_WINDOWS", False)
    with pytest.raises(PermissionError):        # POSIX: an unwritable folder, as before
        lock.try_acquire()


def test_breaking_a_stale_lock_that_cannot_be_deleted_gives_up_quietly(tmp_path, monkeypatch, windows):
    path = tmp_path / "usage.json.lock"
    raw = json.dumps({"pid": dead_pid(), "host": socket.gethostname(), "started": None, "token": "t"})
    path.write_text(raw, encoding="utf-8")
    Refuse(monkeypatch, "unlink", path)
    _files._break_stale(path, raw)
    assert path.read_text(encoding="utf-8") == raw
    assert not (tmp_path / "usage.json.lock.break").exists()


def test_one_short_lock_wait_stays_within_its_timeout_and_one_retry_round(tmp_path, monkeypatch, windows):
    # A stale lock whose delete always fails: breaking it is tried, but the retries of every try share the
    # acquire's deadline, so the wait is the timeout plus at most one round of sharing retries (0.63 s).
    path = tmp_path / "usage.json.lock"
    path.write_text(json.dumps({"pid": dead_pid(), "host": socket.gethostname(), "started": None, "token": "t"}),
                    encoding="utf-8")
    Refuse(monkeypatch, "unlink", path)
    start = windows.now
    with pytest.raises(_files.LockTimeout):
        _files.PidLock(path, stale_after=30).acquire(_files.LOCK_TIMEOUT)
    assert windows.now - start <= _files.LOCK_TIMEOUT + 0.63 + 1e-9


def test_a_later_retry_never_removes_the_lock_another_process_took(tmp_path, monkeypatch, windows):
    # Neither the delete nor the mark works, so the release is left for the retry.  Meanwhile the lock turns
    # stale and another process breaks it and takes the same path.  The retry must leave that new lock alone.
    lock = _files.PidLock(tmp_path / "usage.json.lock", stale_after=30)
    assert lock.try_acquire()
    mine = lock.token
    unlink = Refuse(monkeypatch, "unlink", lock.path)
    opener = Refuse(monkeypatch, "open", lock.path, flags_with=os.O_RDWR)
    lock.release()
    assert list(_files._PENDING) == [(os.path.normcase(str(lock.path)), mine)]
    unlink.stop()
    opener.stop()
    make_old(lock.path)
    go, taken, finish, out = SPAWN.Event(), SPAWN.Event(), SPAWN.Event(), SPAWN.Queue()
    proc = SPAWN.Process(target=store_workers.take_stale_lock, args=(str(lock.path), go, taken, finish, out))
    proc.start()
    try:
        go.set()
        theirs = out.get(timeout=60)
        assert _files.retry_pending_releases() == 0
        data = lock_data(lock.path)
        assert (data["token"], data["pid"]) == (theirs, proc.pid)
    finally:
        finish.set()
        proc.join(30)
    assert proc.exitcode == 0
    assert not lock.path.exists()


def test_the_released_mark_never_overwrites_a_lock_another_process_took_meanwhile(tmp_path, monkeypatch, windows):
    # The delete fails; between it and the mark, the lock is broken and taken by another process.  The mark
    # checks the token on the handle it writes through, so it finds the other lock and leaves it unchanged.
    lock = _files.PidLock(tmp_path / "usage.json.lock", stale_after=30)
    assert lock.try_acquire()
    Refuse(monkeypatch, "unlink", lock.path)
    go, taken, finish, out = SPAWN.Event(), SPAWN.Event(), SPAWN.Event(), SPAWN.Queue()
    proc = SPAWN.Process(target=store_workers.take_stale_lock, args=(str(lock.path), go, taken, finish, out))
    proc.start()
    real_open = os.open

    def swap() -> None:     # the other process breaks the (now old) lock and takes it, before the mark opens it
        make_old(lock.path)
        go.set()
        assert taken.wait(60)

    def opener(target, flags, *args, **kwargs):
        if os.path.normcase(os.fspath(target)) == os.path.normcase(str(lock.path)) and flags & os.O_RDWR:
            pending = getattr(opener, "swap", None)
            if pending is not None:
                opener.swap = None
                pending()
        return real_open(target, flags, *args, **kwargs)
    opener.swap = swap
    monkeypatch.setattr(os, "open", opener)
    try:
        lock.release()
        theirs = out.get(timeout=60)
        data = lock_data(lock.path)
        assert (data["token"], data["pid"], data.get("released")) == (theirs, proc.pid, None)
        assert not _files._PENDING
    finally:
        go.set()
        finish.set()
        proc.join(30)
    assert proc.exitcode == 0


def test_the_background_retry_removes_a_lock_once_windows_lets_go(tmp_path, monkeypatch):
    monkeypatch.setattr(_files, "_WINDOWS", True, raising=False)
    lock = _files.PidLock(tmp_path / "projects.json.lock", stale_after=30)
    assert lock.try_acquire()
    unlink = Refuse(monkeypatch, "unlink", lock.path)
    opener = Refuse(monkeypatch, "open", lock.path, flags_with=os.O_RDWR)
    lock.release()
    assert _files._PENDING
    unlink.stop()               # the other program closes its handle
    opener.stop()
    waited = threading.Event()
    for _ in range(500):        # bounded: the thread tries every 50 ms
        with _files._PENDING_GUARD:
            if not _files._PENDING:
                break
        waited.wait(0.01)
    assert not _files._PENDING
    assert not lock.path.exists()


def test_an_opener_waits_for_a_released_lock_that_windows_has_not_deleted_yet(st, project, monkeypatch, windows):
    conv = saved(st, project)
    st.close(conv)
    lock_path = st.project_dir(conv.project_key) / f"{conv.id}.json.lock"
    lock_path.write_text(json.dumps({"pid": os.getppid(), "host": socket.gethostname(), "started": None,
                                     "token": "other", "purpose": "owner", "released": True}), encoding="utf-8")
    refuse = Refuse(monkeypatch, "unlink", lock_path)
    waits: list[float] = []

    def wait(seconds: float) -> None:       # meanwhile the other process's handle is closed
        waits.append(seconds)
        refuse.stop()
    monkeypatch.setattr(store_module, "_sleep", wait, raising=False)
    opened = st.open(project, conv.id, if_locked="raise")
    assert opened.id == conv.id and waits == [0.02]
    assert lock_data(lock_path)["pid"] == os.getpid()
    st.close(opened)

