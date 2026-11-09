"""The assistant's store, lock and ledger files when the file system misbehaves (review fix group G12).

Windows is simulated on Linux: ``_files._WINDOWS`` is set and ``os.unlink`` / ``os.open`` raise ``PermissionError``
(a sharing violation, or a delete still pending) for one path while a test says so.  ``_files._sleep`` and
``_files._clock`` are replaced by a fake clock that the waits advance, so nothing here depends on how fast the
machine is.  Where a lock must hold between processes the other owner is a real process (``spawn``), waited for
with bounded timeouts.  Expected values are worked out by hand from the docstrings of ``_files``, ``store`` and
``usage``, never read back from the code under test.
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flextool.assistant import _files, settings, store as store_module
from flextool.assistant.llm_adapter import Message, Usage
from flextool.assistant.store import Store
from flextool.assistant.usage import Ledger, add_usage
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



# --- C5-4: files of the wrong shape; damage decided on stable evidence ---------------------------------------------
@pytest.fixture
def no_wait(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """``_files._sleep`` records instead of sleeping (the quarantine's second look)."""
    waits: list[float] = []
    monkeypatch.setattr(_files, "_sleep", waits.append, raising=False)
    return waits


def damaged_copies(folder: Path, name: str) -> list[str]:
    return sorted(n for n in os.listdir(folder) if n.startswith(f"{name}.damaged-"))


NOV = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)


def test_a_ledger_of_the_wrong_shape_is_set_aside_and_its_good_counts_carried(root, no_wait, capsys):
    good = {"requests": 2, "input_uncached": 20, "cache_read": 0, "cache_write": 0, "output": 4, "not_reported": 0,
            "rejected_attempts": 0, "cost": None}
    path = root / "usage.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 1, "months": {"2026-10": {"p": {"label": "P", "models": {
        "bad": {"requests": "7"}, "good": good}}}}}), encoding="utf-8")
    assert Ledger(root).read()["months"] == {"2026-10": {"p": {"label": "P", "models": {"good": good}}}}
    assert Ledger(root).record("gemini", "Google Gemini", "m", Usage(5, 0, 0, 1), now=NOV) is True
    assert len(damaged_copies(root, "usage.json")) == 1
    months = json.loads(path.read_text(encoding="utf-8"))["months"]
    assert months["2026-10"] == {"p": {"label": "P", "models": {"good": good}}}
    assert months["2026-11"]["gemini"]["models"]["m"]["requests"] == 1
    assert "not a usage record of the known form" in capsys.readouterr().err
    path.write_text(json.dumps({"format": 1, "months": []}), encoding="utf-8")
    assert Ledger(root).record("gemini", "Google Gemini", "m", Usage(5, 0, 0, 1), now=NOV) is True
    assert len(damaged_copies(root, "usage.json")) == 2
    assert list(json.loads(path.read_text(encoding="utf-8"))["months"]) == ["2026-11"]
    path.write_text(json.dumps({"format": 1, "months": {"2026-10": {"p": {"models": {"m": {"output": -1}}}}}}),
                    encoding="utf-8")
    assert Ledger(root).read()["months"] == {}


def test_a_ledger_read_that_looks_damaged_once_is_not_set_aside(root, no_wait, monkeypatch, capsys):
    path = root / "usage.json"
    root.mkdir(parents=True, exist_ok=True)
    Ledger(root).record("gemini", "Google Gemini", "m", Usage(5, 0, 0, 1), now=NOV)
    before = path.read_bytes()
    real = Path.read_text
    first = [True]

    def half_read(self, *args, **kwargs):           # the first read meets a file another program is writing
        if self == path and first[0]:
            first[0] = False
            return before.decode("utf-8")[:12]
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", half_read)
    assert Ledger(root).record("gemini", "Google Gemini", "m", Usage(5, 0, 0, 1), now=NOV) is False
    assert damaged_copies(root, "usage.json") == []
    assert path.read_bytes() == before
    assert "left as it is" in capsys.readouterr().err


def test_projects_json_is_set_aside_only_on_a_second_identical_read(st, root, no_wait, monkeypatch):
    path = root / "projects.json"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 1, "projects": {}}), encoding="utf-8")
    before = path.read_bytes()
    real = Path.read_text
    first = [True]

    def half_read(self, *args, **kwargs):
        if self == path and first[0]:
            first[0] = False
            return '{"format": 1, "proj'
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", half_read)
    with pytest.raises(OSError):
        st._change_projects(lambda rows: rows.setdefault("k", {}))
    assert damaged_copies(root, "projects.json") == [] and path.read_bytes() == before
    monkeypatch.setattr(Path, "read_text", real)
    # Damaged for real, twice within one second: two copies (they used to share one name), no write over either.
    for text in ("{", "[1]"):
        path.write_text(text, encoding="utf-8")
        st._change_projects(lambda rows: rows.setdefault("k", {}))
    copies = damaged_copies(root, "projects.json")
    assert len(copies) == 2 and all(_files.DAMAGED.match(c) for c in copies)
    assert sorted((root / c).read_text(encoding="utf-8") for c in copies) == ["[1]", "{"]


def test_a_conversation_of_the_wrong_shape_is_damaged_keeps_no_lock_and_stays_put(st, project):
    older = saved(st, project, "the older one")
    st.close(older)
    bad = saved(st, project, "the newer one")
    st.close(bad)
    file = st.project_dir(bad.project_key) / f"{bad.id}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["messages"] = ["oops"]
    file.write_text(json.dumps(data), encoding="utf-8")
    os.utime(file, (time.time() + 60, time.time() + 60))           # surely the newest
    for _ in range(2):
        with pytest.raises(store_module.ConversationDamaged):
            st.open(project, bad.id, if_locked="raise")
        assert not (file.parent / f"{bad.id}.json.lock").exists()
    assert file.exists()
    opened = st.open(project)                       # the newest that can be read
    assert opened.id == older.id
    st.close(opened)
    data["messages"] = 5
    file.write_text(json.dumps(data), encoding="utf-8")
    assert [i.id for i in st.list_conversations(project)] == [older.id]


def test_stored_usage_counts_that_are_not_numbers_read_as_zero(st, project):
    conv = saved(st, project)
    st.close(conv)
    file = st.project_dir(conv.project_key) / f"{conv.id}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["usage"] = {"output": "12", "input_uncached": 30, "cost": "x"}
    file.write_text(json.dumps(data), encoding="utf-8")
    opened = st.open(project, conv.id)
    assert (opened.usage["output"], opened.usage["input_uncached"], opened.usage["cost"]) == (0, 30, None)
    add_usage(opened.usage, Usage(5, 0, 0, 2))
    assert (opened.usage["output"], opened.usage["input_uncached"]) == (2, 35)
    st.close(opened)


def test_a_conversation_of_a_newer_flextool_opens_read_only_with_unknown_messages_skipped(st, project):
    conv = saved(st, project, "first")
    st.close(conv)
    file = st.project_dir(conv.project_key) / f"{conv.id}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["format"] = 2
    data["messages"].append({"role": "planner", "text": "a newer kind"})
    file.write_text(json.dumps(data), encoding="utf-8")
    before = file.read_bytes()
    opened = st.open(project, conv.id)
    assert opened.read_only and [m.text for m in opened.messages] == ["first"]
    assert not (file.parent / f"{conv.id}.json.lock").exists()
    assert file.read_bytes() == before and damaged_copies(file.parent, f"{conv.id}.json") == []


# --- C5-7: chat numbers after projects.json is lost ---------------------------------------------------------------
def numbered_chats(st: Store, project: Path, count: int) -> list[store_module.Conversation]:
    convs = []
    for _ in range(count):
        number = st.allocate_chat_number(project)
        convs.append(saved(st, project, f"chat {number}", number=number))
    return convs


def test_numbers_go_on_from_the_saved_and_open_chats_after_projects_json_is_lost(st, root, project, no_wait):
    convs = numbered_chats(st, project, 3)
    assert [c.number for c in convs] == [1, 2, 3]
    for conv in convs:
        st.close(conv)
    (root / "projects.json").write_text("{", encoding="utf-8")
    assert st.allocate_chat_number(project) == 4
    (root / "projects.json").unlink()
    unsaved = st.new_conversation(project, ORIGIN, None, number=4)     # registered, never saved
    assert st.allocate_chat_number(project) == 5
    st.close(unsaved)


def test_numbers_go_on_from_an_unsaved_chat_of_another_process(st, root, project):
    for conv in numbered_chats(st, project, 2):
        st.close(conv)
    ready, finish, out = SPAWN.Event(), SPAWN.Event(), SPAWN.Queue()
    proc = SPAWN.Process(target=store_workers.new_unsaved_chat, args=(str(root), str(project), ready, finish, out))
    proc.start()
    try:
        assert out.get(timeout=60) == 3
        assert ready.wait(30)
        (root / "projects.json").unlink()
        assert st.allocate_chat_number(project) == 4       # 3 is only in the other process's lock file
    finally:
        finish.set()
        proc.join(30)
    assert proc.exitcode == 0


def test_a_busy_project_list_still_gives_a_new_chat_its_number(st, root, project, monkeypatch, capsys):
    for conv in numbered_chats(st, project, 2):
        st.close(conv)

    def busy(path, **kwargs):
        raise _files.LockTimeout(path)
    monkeypatch.setattr(_files, "locked", busy)
    assert [st.allocate_chat_number(project) for _ in range(2)] == [3, 4]
    assert "this chat is number 3" in capsys.readouterr().err
    monkeypatch.undo()
    assert st.allocate_chat_number(project) == 5           # the numbers counted in memory are not given again


# --- C5-12: no parse before a steady save; listings parse a file once; open does not block the registry ----------
def test_a_save_of_an_open_chat_does_not_parse_the_file_again(st, project, monkeypatch):
    conv = saved(st, project)
    file = st.project_dir(conv.project_key) / f"{conv.id}.json"
    real = _files.read_json
    reads: list[Path] = []

    def spy(path, *args, **kwargs):
        reads.append(Path(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(_files, "read_json", spy)
    conv.messages.append(Message("user", "again"))
    assert st.save(conv)
    assert file not in reads
    monkeypatch.undo()
    st.close(conv)


def test_a_save_after_the_lock_was_lost_keeps_a_newer_flextools_file(st, project, capsys):
    conv = saved(st, project)
    folder = st.project_dir(conv.project_key)
    (folder / f"{conv.id}.json.lock").unlink()               # lost (removed by another FlexTool)
    file = folder / f"{conv.id}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    data["format"] = 2
    file.write_text(json.dumps(data), encoding="utf-8")       # a newer FlexTool opened, wrote and closed it
    before = file.read_bytes()
    conv.messages.append(Message("user", "after the loss"))
    assert st.save(conv) is False
    assert file.read_bytes() == before and conv.read_only
    assert "newer FlexTool" in capsys.readouterr().err
    st.close(conv)


def test_listing_parses_a_conversation_file_again_only_after_it_changed(st, project, monkeypatch):
    conv = saved(st, project)
    file = st.project_dir(conv.project_key) / f"{conv.id}.json"
    real = _files.read_json
    reads: list[Path] = []

    def spy(path, *args, **kwargs):
        reads.append(Path(path))
        return real(path, *args, **kwargs)
    monkeypatch.setattr(_files, "read_json", spy)
    assert [i.first_question for i in st.list_conversations(project)] == ["Miksi?"]
    assert [i.first_question for i in st.list_conversations(project)] == ["Miksi?"]
    assert reads.count(file) == 1
    conv.messages.append(Message("assistant", "Siksi, ja vähän pidempi vastaus."))
    assert st.save(conv)
    st.list_conversations(project)
    assert reads.count(file) == 2
    st.close(conv)


def test_an_opener_waiting_for_a_cleanup_does_not_block_the_registry(st, project, monkeypatch):
    conv = saved(st, project)
    st.close(conv)
    lock_path = st.project_dir(conv.project_key) / f"{conv.id}.json.lock"
    lock_path.write_text(json.dumps({"pid": os.getppid(), "host": socket.gethostname(), "started": None,
                                     "token": "cleanup", "purpose": "purge"}), encoding="utf-8")
    entered, let_go = threading.Event(), threading.Event()

    def wait(_seconds: float) -> None:      # the opener waits for the cleanup here, until the test lets it go
        entered.set()
        assert let_go.wait(30)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()              # the cleanup is done
    monkeypatch.setattr(store_module, "_sleep", wait, raising=False)
    opened: list = []
    opener = threading.Thread(target=lambda: opened.append(st.open(project, conv.id, if_locked="raise")))
    opener.start()
    try:
        assert entered.wait(30)
        looked: list = []
        reader = threading.Thread(target=lambda: looked.append(st._registered(conv.project_key, conv.id)))
        reader.start()
        reader.join(10)
        assert not reader.is_alive() and looked == [None]      # finished while the opener still waited
        assert not let_go.is_set()
    finally:
        let_go.set()
        opener.join(30)
    assert opened and opened[0].id == conv.id
    st.close(opened[0])
