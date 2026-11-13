"""The assistant's local files: settings, the conversation store with its locks and purge, and the usage ledger.

Everything lives in ``tmp_path`` (``user_config_in_tmp``).  The races run in real processes (``spawn``) where the
store must hold between processes: a purge while another process saves, two processes opening one conversation,
two processes counting into the ledger.  Expected values are written out by hand from P1 §10, never read back
from the code under test.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flextool.assistant import _files, settings, store as store_module, usage
from flextool.assistant.llm_adapter import Message, ToolCall, Usage, not_reported
from flextool.assistant.store import ConversationLocked, ConversationNotFound, Store
from flextool.assistant.usage import (Ledger, format_footer, format_token_line, format_totals, format_turn_line, ledger_key,
                                      short_number)
from tests.assistant import store_workers

ORIGIN = ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "model-a")
SPAWN = multiprocessing.get_context("spawn")


@pytest.fixture
def root(user_config_in_tmp: Path) -> Path:
    return user_config_in_tmp


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "Pröjekt Ääni €"
    path.mkdir()
    return path


@pytest.fixture
def st(root: Path) -> Store:
    return Store(root)


def key_of(path: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(path.resolve())).encode("utf-8")).hexdigest()[:16]


def set_expiry(root: Path, days: int) -> None:
    settings.save_settings(settings.Settings(conversation_expiry_days=days), root)


def saved(st: Store, project: Path, text: str = "Miksi säästö?") -> store_module.Conversation:
    conv = st.new_conversation(project, ORIGIN, "standard")
    conv.messages.append(Message("user", text))
    assert st.save(conv)
    return conv


def projects_rows(root: Path) -> dict:
    path = root / "projects.json"
    return json.loads(path.read_text(encoding="utf-8"))["projects"] if path.exists() else {}


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# --- files, settings ---------------------------------------------------------------------------------------------
def test_files_are_utf8_lf_and_the_project_key_is_the_resolved_path_hash(st, root, project):
    conv = saved(st, project, "Mikä on hiilen hinta? €/MWh\r\nrivi kaksi")
    conv.messages.append(Message("assistant", "", tool_calls=[ToolCall("abc123XYZ", "get_entity", {"name": "ä"},
                                                                       '{"name": "ä"}')]))
    assert st.save(conv)
    folder = root / "conversations" / key_of(project)
    raw = (folder / f"{conv.id}.json").read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert "Mikä on hiilen hinta? €/MWh" in text          # not escaped
    data = json.loads(text)
    assert data["format"] == 1
    assert data["project"] == {"key": key_of(project), "name": "Pröjekt Ääni €"}
    assert projects_rows(root)[key_of(project)]["name"] == "Pröjekt Ääni €"
    # The same folder spelled another way has the same key.
    assert _files.project_key(project / ".." / project.name) == key_of(project)
    # No temp file is left behind.
    assert not [n for n in os.listdir(folder) if n.endswith(".tmp")]
    st.close(conv)


def test_ids_and_times_are_utc(st, project):
    before = datetime.now(timezone.utc).replace(microsecond=0)
    conv = saved(st, project)
    after = datetime.now(timezone.utc)
    match = re.fullmatch(r"(\d{8}T\d{6}Z)-[0-9a-f]{6}", conv.id)
    assert match
    stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    assert before <= stamp <= after
    assert conv.created.endswith("Z") and conv.updated.endswith("Z")
    st.close(conv)


def test_settings_defaults_round_trip_and_notes(root):
    s = settings.load_settings(root)
    assert (s.conversation_expiry_days, s.compaction_every, s.keep_recent_turns, s.parallel_requests) == (
        30, None, None, 1)         # compaction from the model's context (P3 §4.5)
    assert s.provider.knobs == {"openrouter.data_collection": "deny", "openrouter.zdr": False}
    assert s.provider.key_storage == "keyring"
    path = root / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": 1, "conversation_expiry_days": -3, "parallel_requests": 9,
                                "provider": {"preset": "mistral", "tier": "huge", "colour": "blue"},
                                "future_key": [1, 2]}), encoding="utf-8")
    s = settings.load_settings(root)
    assert s.conversation_expiry_days == 30 and s.parallel_requests == 1 and s.provider.tier is None
    assert s.provider.preset == "mistral"
    assert len(s.notes) == 3
    s.conversation_expiry_days = 7
    settings.save_settings(s, root)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["conversation_expiry_days"] == 7
    assert data["future_key"] == [1, 2] and data["provider"]["colour"] == "blue"     # unknown keys kept
    path.write_text(json.dumps({"format": 2, "conversation_expiry_days": 5}), encoding="utf-8")
    s = settings.load_settings(root)
    assert s.read_only and s.conversation_expiry_days == 5
    with pytest.raises(_files.NewerFormat):
        settings.save_settings(s, root)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == 2                # never overwritten


# --- open, continue, locks ---------------------------------------------------------------------------------------
def test_continue_opens_the_newest_or_the_given_conversation(st, project):
    first = saved(st, project, "first")
    st.close(first)
    second = saved(st, project, "second")
    st.close(second)
    os.utime(st.project_dir(key_of(project)) / f"{first.id}.json", (time.time() + 60, time.time() + 60))
    newest = st.open(project)
    assert newest.id == first.id and newest.messages[0].text == "first"
    st.close(newest)
    given = st.open(project, second.id)
    assert given.messages[0].text == "second" and given.origin == ORIGIN and given.notice is None
    st.close(given)
    with pytest.raises(ConversationNotFound):
        st.open(project, "20200101T000000Z-abcdef")
    with pytest.raises(ConversationNotFound):
        st.open(project, "../../etc")
    with pytest.raises(ConversationNotFound):
        Store(st.root).open(project.parent)          # a project with no conversations


def test_a_conversation_keeps_only_the_users_level(st, project):
    """Owner GUI feedback 2026-11-08 (2): "level" is the user's choice; the "tier" older files kept was the level
    resolved at their last question, so it reads as the automatic level and is not written again."""
    chosen = st.new_conversation(project, ORIGIN, "standard")
    chosen.messages.append(Message("user", "q", turn=1))
    st.save(chosen)
    st.close(chosen)
    data = json.loads((st.project_dir(key_of(project)) / f"{chosen.id}.json").read_text(encoding="utf-8"))
    assert data["level"] == "standard" and "tier" not in data
    reopened = st.open(project, chosen.id)
    assert reopened.level == "standard"
    st.close(reopened)
    old = saved(st, project, "old")
    st.close(old)
    path = st.project_dir(key_of(project)) / f"{old.id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("level")
    data["tier"] = "small"
    path.write_text(json.dumps(data), encoding="utf-8")
    reopened = st.open(project, old.id)
    assert reopened.level is None
    st.save(reopened)
    st.close(reopened)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["level"] is None and "tier" not in data


def test_a_second_opener_in_this_process_gets_a_new_conversation_or_the_error(st, project):
    conv = saved(st, project)
    other = st.open(project, conv.id)
    assert other.id != conv.id and other.messages == []
    assert other.notice == "This conversation is open in another FlexTool window or terminal; starting a new one."
    with pytest.raises(ConversationLocked) as caught:
        st.open(project, conv.id, if_locked="raise")
    assert caught.value.owner == "this_process"
    st.close(other)
    st.close(conv)
    again = st.open(project, conv.id, if_locked="raise")
    assert again.id == conv.id
    st.close(again)


def test_a_stale_lock_of_a_gone_process_is_ignored(st, project):
    conv = saved(st, project)
    st.close(conv)
    lock = st.project_dir(key_of(project)) / f"{conv.id}.json.lock"
    lock.write_text(json.dumps({"pid": dead_pid(), "host": socket.gethostname(), "started": None, "token": "t"}),
                    encoding="utf-8")
    opened = st.open(project, conv.id, if_locked="raise")
    assert opened.id == conv.id
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()
    st.close(opened)
    assert not lock.exists()


def test_two_processes_opening_one_conversation_at_once(st, root, project):
    conv = saved(st, project)
    st.close(conv)
    barrier, finish, out = SPAWN.Barrier(3), SPAWN.Event(), SPAWN.Queue()
    procs = [SPAWN.Process(target=store_workers.open_at_once,
                           args=(str(root), str(project), conv.id, barrier, finish, out)) for _ in range(3)]
    for p in procs:
        p.start()
    try:
        results = sorted(out.get(timeout=60) for _ in procs)
        assert results[0] == ("locked", "other_process") and results[1] == ("locked", "other_process")
        assert results[2] == ("opened", conv.id)
        # Held by another process: this one cannot open or delete it.
        with pytest.raises(ConversationLocked) as caught:
            st.open(project, conv.id, if_locked="raise")
        assert caught.value.owner == "other_process"
        assert st.delete_conversation(project, conv.id) == "locked_elsewhere"
        assert [i.locked for i in st.list_conversations(project)] == ["other_process"]
    finally:
        finish.set()
        for p in procs:
            p.join(30)
    assert all(p.exitcode == 0 for p in procs)
    assert st.delete_conversation(project, conv.id) == "deleted"


# --- purge ------------------------------------------------------------------------------------------------------
def test_purge_after_the_expiry_keeps_open_conversations_and_the_ledger(st, root, project, tmp_path):
    old = saved(st, project, "old")
    st.set_title(project, old.id, "Vanha")
    st.close(old)
    still_open = saved(st, project, "open")
    Ledger(root).record("gemini", "Google Gemini", "model-a", Usage(5, 0, 0, 1))
    now = datetime.now(timezone.utc)
    assert st.purge_expired(now + timedelta(days=29)) == 0
    assert st.purge_expired(now + timedelta(days=31)) == 1
    folder = st.project_dir(key_of(project))
    assert sorted(os.listdir(folder)) == [f"{still_open.id}.json", f"{still_open.id}.json.lock"]
    assert key_of(project) in projects_rows(root)
    st.close(still_open)
    assert st.purge_expired(now + timedelta(days=31)) == 1
    assert not folder.exists()
    assert projects_rows(root) == {}
    assert (root / "usage.json").exists()


def test_expiry_zero_writes_nothing_and_deletes_everything_not_open(st, root, project):
    kept_open = saved(st, project, "open")
    closed = saved(st, project, "closed")
    st.close(closed)
    set_expiry(root, 0)
    Ledger(root).record("gemini", "Google Gemini", "model-a", Usage(5, 0, 0, 1))
    fresh = st.new_conversation(project, ORIGIN, "basic")
    fresh.messages.append(Message("user", "not saved"))
    assert st.save(fresh) is False
    assert st.purge_expired() == 1
    names = os.listdir(st.project_dir(key_of(project)))
    assert f"{closed.id}.json" not in names and f"{fresh.id}.json" not in names
    assert f"{kept_open.id}.json" in names
    st.set_open_chats(project, [kept_open.id], kept_open.id)
    assert st.open_chats(project) == {"ids": [], "selected": None}
    for conv in (kept_open, fresh):
        st.close(conv)
    assert st.purge_expired() == 1
    assert not (root / "conversations").exists() or not os.listdir(root / "conversations")
    assert json.loads((root / "usage.json").read_text(encoding="utf-8"))["months"]


def test_purge_removes_dead_writers_temp_files_orphan_titles_and_stale_locks(st, root, project):
    conv = saved(st, project)
    folder = st.project_dir(key_of(project))
    gone = dead_pid()
    (folder / f"{conv.id}.json.{gone}.0a1b2c3d.tmp").write_text("{", encoding="utf-8")
    (folder / f"{conv.id}.json.{os.getpid()}.0a1b2c3d.tmp").write_text("{", encoding="utf-8")     # a live writer
    (root / f"settings.json.{gone}.00000000.tmp").write_text("{", encoding="utf-8")
    orphan = "20260101T000000Z-aaaaaa"
    (folder / f"{orphan}.title.json").write_text('{"format": 1, "title": "x"}', encoding="utf-8")
    stale = "20260101T000000Z-bbbbbb"
    (folder / f"{stale}.json.lock").write_text(
        json.dumps({"pid": gone, "host": socket.gethostname(), "started": None, "token": "t"}), encoding="utf-8")
    st.purge_expired()
    names = set(os.listdir(folder))
    assert names == {f"{conv.id}.json", f"{conv.id}.json.lock", f"{conv.id}.json.{os.getpid()}.0a1b2c3d.tmp"}
    assert not (root / f"settings.json.{gone}.00000000.tmp").exists()
    st.close(conv)


def test_purge_and_saves_at_the_same_time_lose_no_save_and_no_project_row(root, tmp_path):
    """One process saves a conversation 60 times and another starts conversations in six projects while a third
    purges as if a year had passed: every open conversation and its project row survive, the last save is on
    disk."""
    saving = tmp_path / "saving project"
    others = [tmp_path / f"project {i}" for i in range(6)]
    for p in (saving, *others):
        p.mkdir()
    ready = [SPAWN.Event() for _ in range(3)]
    go, finish, stop = SPAWN.Event(), SPAWN.Event(), SPAWN.Event()
    save_out, new_out, purge_out = SPAWN.Queue(), SPAWN.Queue(), SPAWN.Queue()
    procs = [
        SPAWN.Process(target=store_workers.save_loop, args=(str(root), str(saving), 60, ready[0], go, finish, save_out)),
        SPAWN.Process(target=store_workers.new_projects,
                      args=(str(root), [str(p) for p in others], ready[1], go, finish, new_out)),
        SPAWN.Process(target=store_workers.purge_loop, args=(str(root), ready[2], stop, purge_out)),
    ]
    for p in procs:
        p.start()
    try:
        assert all(e.wait(60) for e in ready)
        conversation_id = save_out.get(timeout=60)
        go.set()
        assert save_out.get(timeout=60) == 60
        new_ids = new_out.get(timeout=60)
        stop.set()
        assert purge_out.get(timeout=60) > 0
        data = json.loads((root / "conversations" / key_of(saving) / f"{conversation_id}.json")
                          .read_text(encoding="utf-8"))
        assert [m["text"] for m in data["messages"]] == [f"question {i}" for i in range(60)]
        rows = projects_rows(root)
        assert key_of(saving) in rows
        for p, cid in zip(others, new_ids):
            assert key_of(p) in rows, p.name
            assert (root / "conversations" / key_of(p) / f"{cid}.json").exists()
    finally:
        stop.set()
        go.set()
        finish.set()
        for p in procs:
            p.join(30)
    assert all(p.exitcode == 0 for p in procs)


# --- delete, titles, numbers, open chats (the window's additions) ------------------------------------------------
def test_delete_during_a_finishing_save_leaves_no_file(st, project, monkeypatch):
    conv = saved(st, project)
    path = st.project_dir(key_of(project)) / f"{conv.id}.json"
    entered, release = threading.Event(), threading.Event()
    real = _files.write_json_atomic

    def slow_write(target, data, **kwargs):
        if Path(target) == path:
            entered.set()
            release.wait(10)
        return real(target, data, **kwargs)

    monkeypatch.setattr(_files, "write_json_atomic", slow_write)
    conv.messages.append(Message("assistant", "answer"))
    saver = threading.Thread(target=st.save, args=(conv,))
    saver.start()
    assert entered.wait(5)
    result: list[str] = []
    deleter = threading.Thread(target=lambda: result.append(st.delete_conversation(project, conv.id)))
    deleter.start()
    deleter.join(0.2)
    assert deleter.is_alive()                  # waits for the one write that is under way, no longer
    release.set()
    saver.join(5)
    deleter.join(5)
    assert result == ["deleted"]
    assert conv.deleted
    assert not path.exists() and not path.with_name(path.name + ".lock").exists()
    assert st.save(conv) is False              # a later save (checkpoint, repair) brings nothing back
    assert not path.exists()
    st.close(conv)
    assert st.delete_conversation(project, conv.id) == "missing"


def test_a_delete_does_not_wait_for_a_running_turn(st, project):
    conv = saved(st, project)
    started = time.monotonic()
    assert st.delete_conversation(project, conv.id) == "deleted"
    assert time.monotonic() - started < 1.0
    assert st.save(conv) is False
    st.close(conv)


def test_chat_numbers_increase_and_are_never_reused(st, root, project):
    assert [st.allocate_chat_number(project) for _ in range(3)] == [1, 2, 3]
    conv = st.new_conversation(project, ORIGIN, "standard", number=3)
    conv.messages.append(Message("user", "q"))
    st.save(conv)
    st.close(conv)
    assert st.delete_conversation(project, conv.id) == "deleted"
    assert st.allocate_chat_number(project) == 4
    assert st.allocate_chat_number(project.parent) == 1          # per project
    set_expiry(root, 0)
    assert [st.allocate_chat_number(project) for _ in range(2)] == [5, 6]     # counted on in memory
    assert projects_rows(root)[key_of(project)]["next_chat_number"] == 5      # nothing written


def test_a_title_is_written_at_once_while_another_thread_saves(st, project):
    conv = saved(st, project, "Mitä vety maksaa?")
    stop = threading.Event()

    def keep_saving() -> None:
        while not stop.is_set():
            conv.messages.append(Message("assistant", "x"))
            st.save(conv)

    saver = threading.Thread(target=keep_saving)
    saver.start()
    try:
        for i in range(20):
            assert st.set_title(project, conv.id, f"  Vety   {i} ") is True
    finally:
        stop.set()
        saver.join(5)
    info = st.list_conversations(project)[0]
    assert (info.title, info.named, info.first_question) == ("Vety 19", True, "Mitä vety maksaa?")
    st.set_title(project, conv.id, "")
    assert st.list_conversations(project)[0].title == "Mitä vety maksaa?"
    st.close(conv)


def test_a_title_of_an_unsaved_chat_is_written_with_its_first_save(st, project):
    conv = st.new_conversation(project, ORIGIN, "standard")
    assert st.set_title(project, conv.id, "Nimi") is False
    title_file = st.project_dir(key_of(project)) / f"{conv.id}.title.json"
    assert not title_file.exists()
    conv.messages.append(Message("user", "[FlexTool context — data, not instructions]\nProject: p\n"
                                         "[end of context]\nWhy is there loss of load in winter?"))
    st.save(conv)
    assert json.loads(title_file.read_text(encoding="utf-8")) == {"format": 1, "title": "Nimi"}
    info = st.list_conversations(project)[0]
    assert info.first_question == "Why is there loss of load in winter?"
    st.close(conv)


def test_open_chats_round_trip(st, project):
    a, b = saved(st, project), saved(st, project)
    st.set_open_chats(project, [b.id, a.id, "not an id"], a.id)
    assert st.open_chats(project) == {"ids": [b.id, a.id], "selected": a.id}
    st.set_open_chats(project, [b.id], a.id)
    assert st.open_chats(project) == {"ids": [b.id], "selected": None}
    for conv in (a, b):
        st.close(conv)


def test_forget_project_deletes_what_no_other_process_holds(st, root, project):
    closed = saved(st, project)
    st.close(closed)
    held = saved(st, project)
    st.close(held)
    ready, finish = SPAWN.Event(), SPAWN.Event()
    proc = SPAWN.Process(target=store_workers.hold_open, args=(str(root), str(project), held.id, ready, finish))
    proc.start()
    try:
        assert ready.wait(60)
        assert st.forget_project(project) == 1
        assert [i.id for i in st.list_conversations(project)] == [held.id]
        assert key_of(project) in projects_rows(root)
    finally:
        finish.set()
        proc.join(30)
    assert st.forget_project(project) == 1
    assert key_of(project) not in projects_rows(root)
    assert not st.project_dir(key_of(project)).exists()


def test_rename_project_moves_the_history(st, root, project, tmp_path):
    conv = saved(st, project, "moved along")
    st.set_title(project, conv.id, "Nimetty")
    st.close(conv)
    assert st.allocate_chat_number(project) == 1
    st.set_open_chats(project, [conv.id], conv.id)
    renamed = tmp_path / "Uusi nimi"
    project.rename(renamed)
    assert st.rename_project(project, renamed) == 1
    rows = projects_rows(root)
    assert key_of(project) not in rows
    row = rows[_files.project_key(renamed)]
    assert (row["name"], row["next_chat_number"]) == ("Uusi nimi", 2)
    assert row["open_chats"] == {"ids": [conv.id], "selected": conv.id}
    reopened = st.open(renamed, conv.id)
    assert reopened.messages[0].text == "moved along" and reopened.title == "Nimetty"
    assert reopened.project_name == "Uusi nimi"
    st.close(reopened)
    assert not (root / "conversations" / key_of(project)).exists()


def test_a_newer_conversation_file_is_read_only(st, project):
    conv = saved(st, project)
    st.close(conv)
    path = st.project_dir(key_of(project)) / f"{conv.id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    opened = st.open(project, conv.id)
    assert opened.read_only and "newer FlexTool" in (opened.notice or "")
    opened.messages.append(Message("user", "more"))
    assert st.save(opened) is False
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == 2
    st.close(opened)


# --- the usage ledger --------------------------------------------------------------------------------------------
def test_a_stopped_request_counts_with_its_output_not_reported(root):
    ledger = Ledger(root)
    now = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    ledger.record("anthropic", "Anthropic", "model-b", Usage(120, 30, 5, None), now=now)       # stopped mid-answer
    ledger.record("anthropic", "Anthropic", "model-b", not_reported(), now=now)                 # nothing reported
    ledger.record("anthropic", "Anthropic", "model-b", Usage(100, 0, 0, 40), now=now)
    ledger.record_rejected("anthropic", "Anthropic", "model-b", 2, now=now)
    counts = ledger.read()["months"]["2026-11"]["anthropic"]["models"]["model-b"]
    assert counts == {"requests": 3, "input_uncached": 220, "cache_read": 30, "cache_write": 5, "output": 40,
                      "not_reported": 2, "rejected_attempts": 2, "cost": None}


def test_cost_only_from_the_provider_and_keys_per_endpoint(root):
    ledger = Ledger(root)
    now = datetime(2026, 12, 1, tzinfo=timezone.utc)
    ledger.record("openrouter", "OpenRouter", "m", Usage(10, 0, 0, 2, cost=0.25), now=now)
    ledger.record("openrouter", "OpenRouter", "m", Usage(10, 0, 0, 2, cost=0.5), now=now)
    assert ledger.read()["months"]["2026-12"]["openrouter"]["models"]["m"]["cost"] == 0.75
    assert ledger_key("openai_compatible", "https://llm.example.org:8443/v1") == "openai_compatible:llm.example.org:8443"
    assert ledger_key("ollama", "http://localhost:11434/v1", forwarded_to="ollama.com") == "ollama_cloud"
    assert ledger_key("mistral", "https://api.mistral.ai/v1") == "mistral"
    text = format_totals(ledger.read())
    assert "    m: 2 requests · 20 in · 4 out · cost 0.75 (reported by OpenRouter, in credits)" in text


@pytest.fixture
def local_time(monkeypatch):
    """The local time zone for the test: ``local_time("Europe/Helsinki")`` (POSIX only: time.tzset)."""
    if not hasattr(time, "tzset"):
        pytest.skip("the local time zone can be set only where time.tzset exists")

    def set_zone(zone: str) -> None:
        monkeypatch.setenv("TZ", zone)
        time.tzset()
    yield set_zone
    monkeypatch.undo()
    time.tzset()


def _counts(input_uncached: int, output: int, cache_read: int = 0, requests: int = 1) -> dict:
    return {"requests": requests, "input_uncached": input_uncached, "cache_read": cache_read, "cache_write": 0,
            "output": output, "not_reported": 0, "rejected_attempts": 0, "cost": None}


def test_days_count_by_the_local_calendar_date(root, local_time):
    """23:30 UTC on 8 November is 01:30 on 9 November in Helsinki (UTC+2): the day is the 9th."""
    local_time("EET-2")                                         # Helsinki's winter offset, no tz database needed
    ledger = Ledger(root)
    ledger.record("anthropic", "Anthropic", "m", Usage(100, 20, 0, 7), now=datetime(2026, 11, 8, 23, 30,
                                                                                   tzinfo=timezone.utc))
    ledger.record_rejected("anthropic", "Anthropic", "m", 2, now=datetime(2026, 11, 9, 8, tzinfo=timezone.utc))
    data = ledger.read()
    assert list(data["days"]) == ["2026-11-09"]
    assert list(data["months"]) == ["2026-11"]
    day = data["days"]["2026-11-09"]["anthropic"]
    assert day["label"] == "Anthropic"
    assert day["models"]["m"] == {"requests": 1, "input_uncached": 100, "cache_read": 20, "cache_write": 0,
                                  "output": 7, "not_reported": 0, "rejected_attempts": 2, "cost": None}


def test_the_month_counts_by_the_local_calendar_too(root, local_time):
    """22:30 UTC on 30 November is 00:30 on 1 December in Helsinki (UTC+2): the request goes to December, the
    month of today, so today never exceeds this month.  The UTC-keyed November of an older ledger stays as it
    was (not re-bucketed)."""
    local_time("EET-2")
    root.mkdir(parents=True, exist_ok=True)
    (root / "usage.json").write_text(json.dumps({"format": 1, "months": {"2026-11": {"anthropic": {
        "label": "Anthropic", "models": {"m": _counts(3_000, 300)}}}}}), encoding="utf-8")
    now = datetime(2026, 11, 30, 22, 30, tzinfo=timezone.utc)
    ledger = Ledger(root)
    assert ledger.record("anthropic", "Anthropic", "m", Usage(12_345, 0, 0, 1_234), now=now) is True
    data = ledger.read()
    assert sorted(data["days"]) == ["2026-12-01"]
    assert sorted(data["months"]) == ["2026-11", "2026-12"]
    assert data["months"]["2026-11"]["anthropic"]["models"]["m"] == _counts(3_000, 300)
    assert data["months"]["2026-12"]["anthropic"]["models"]["m"] == _counts(12_345, 1_234)
    assert format_footer(data, "Anthropic", now) == (
        "Anthropic today 12k in, 1k out · this month 12k in, 1k out (this FlexTool installation)")


def test_only_the_last_40_days_are_kept(root, local_time):
    """On 9 November the 40 days kept are 1 October … 9 November: 30 September goes, its month stays."""
    local_time("UTC0")
    ledger = Ledger(root)
    for moment in (datetime(2026, 9, 30, 12), datetime(2026, 10, 1, 12), datetime(2026, 11, 9, 12)):
        ledger.record("gemini", "Google Gemini", "m", Usage(5, 0, 0, 1), now=moment.replace(tzinfo=timezone.utc))
    data = ledger.read()
    assert sorted(data["days"]) == ["2026-10-01", "2026-11-09"]
    assert sorted(data["months"]) == ["2026-09", "2026-10", "2026-11"]
    stored = json.loads((root / "usage.json").read_text(encoding="utf-8"))
    assert sorted(stored["days"]) == ["2026-10-01", "2026-11-09"]


def test_a_ledger_without_days_counts_today_from_now(root, local_time, capsys):
    """A ledger written before the days were kept is of the known form: nothing is set aside, today starts at 0."""
    local_time("UTC0")
    root.mkdir(parents=True, exist_ok=True)
    (root / "usage.json").write_text(json.dumps({"format": 1, "months": {"2026-11": {"anthropic": {
        "label": "Anthropic", "models": {"m": _counts(200_000, 9_456, cache_read=56_123)}}}}}), encoding="utf-8")
    now = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
    ledger = Ledger(root)
    assert ledger.read()["days"] == {}
    assert format_footer(ledger.read(), "Anthropic", now) == (
        "Anthropic today 0 in, 0 out · this month 256k in, 9k out (this FlexTool installation)")
    assert ledger.record("anthropic", "Anthropic", "m", Usage(12_345, 0, 0, 1_234), now=now) is True
    assert not [n for n in os.listdir(root) if n.startswith("usage.json.damaged-")]
    assert "known form" not in capsys.readouterr().err
    data = ledger.read()
    assert data["days"] == {"2026-11-09": {"anthropic": {"label": "Anthropic", "models": {"m": _counts(12_345, 1_234)}}}}
    # month: 256 123 + 12 345 = 268 468 in, 9 456 + 1 234 = 10 690 out
    assert format_footer(data, "Anthropic", now) == (
        "Anthropic today 12k in, 1k out · this month 268k in, 11k out (this FlexTool installation)")


def test_the_footer_and_the_usage_listing_show_today_and_this_month(root, local_time):
    local_time("UTC0")
    ledger = Ledger(root)
    now = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
    ledger.record("anthropic", "Anthropic", "a", Usage(800, 0, 0, 999), now=datetime(2026, 11, 2, tzinfo=timezone.utc))
    ledger.record("anthropic", "Anthropic", "a", Usage(450, 0, 0, 40), now=now)
    ledger.record_rejected("anthropic", "Anthropic", "a", 3, now=now)
    ledger.record("gemini", "Google Gemini", "g", Usage(1_500_000, 0, 0, 2_000), now=datetime(2026, 10, 3,
                                                                                              tzinfo=timezone.utc))
    data = ledger.read()
    # today 450 / 40 (exact below 1 000); this month 1 250 → 1k, 1 039 → 1k
    assert format_footer(data, "Anthropic", now) == ("Anthropic today 450 in, 40 out · this month 1k in, 1k out · "
                                                     "rate-limited attempts this month: 3 (this FlexTool installation)")
    assert format_footer(data, "Google Gemini", now) == (
        "Google Gemini this month: nothing used yet (this FlexTool installation)")
    lines = format_totals(data, now).splitlines()
    assert lines[:3] == [format_footer(data, "Anthropic", now), "", "2026-11"]
    assert "2026-10" in lines and "    g: 1 requests · 1 500 000 in · 2 000 out" in lines


def test_two_processes_count_into_one_ledger(root):
    go = SPAWN.Event()
    procs = [SPAWN.Process(target=store_workers.ledger_loop, args=(str(root), 40, go)) for _ in range(2)]
    for p in procs:
        p.start()
    go.set()
    for p in procs:
        p.join(60)
    assert all(p.exitcode == 0 for p in procs)
    months = Ledger(root).read()["months"]
    counts = next(iter(months.values()))["gemini"]["models"]["model-a"]
    assert (counts["requests"], counts["input_uncached"], counts["output"]) == (80, 800, 80)


def test_a_failing_ledger_write_is_a_warning_and_the_count_is_kept(root, capsys, monkeypatch):
    """A busy lock or a replace Windows refuses (PermissionError while another process or a scanner has the file
    open) is a warning, never a lost count: PR #369 Windows CI lost 1 of 80 paid requests to a LockTimeout."""
    real_locked, real_write = _files.locked, _files.write_json_atomic

    def busy(*args, **kwargs):
        raise _files.LockTimeout(root / "usage.json")

    def refused(*args, **kwargs):
        raise PermissionError(13, "The process cannot access the file because it is being used by another process")

    ledger = Ledger(root)
    monkeypatch.setattr(_files, "locked", busy)
    assert ledger.record("gemini", "Google Gemini", "m", Usage(1, 0, 0, 1)) is False
    assert "usage record could not be updated" in capsys.readouterr().err
    monkeypatch.setattr(_files, "locked", real_locked)
    monkeypatch.setattr(_files, "write_json_atomic", refused)
    assert ledger.record("gemini", "Google Gemini", "m", Usage(2, 0, 0, 2)) is False
    assert usage.pending_count(ledger.path) == 2
    assert ledger.read()["months"] == {}
    monkeypatch.setattr(_files, "write_json_atomic", real_write)
    assert ledger.record("gemini", "Google Gemini", "m", Usage(4, 0, 0, 4)) is True
    assert usage.pending_count(ledger.path) == 0
    counts = next(iter(ledger.read()["months"].values()))["gemini"]["models"]["m"]
    assert (counts["requests"], counts["input_uncached"], counts["output"]) == (3, 7, 7)
    day = next(iter(ledger.read()["days"].values()))["gemini"]["models"]["m"]
    assert (day["requests"], day["input_uncached"], day["output"]) == (3, 7, 7)


def test_a_count_kept_from_a_busy_lock_is_written_at_exit(root):
    """Another process holds the lock longer than a writer waits (the starvation Windows CI showed): the count
    waits in the recording process and its exit writes it once the lock is free."""
    recorded, out = SPAWN.Event(), SPAWN.Queue()
    ledger = Ledger(root)
    with _files.locked(ledger.path):
        proc = SPAWN.Process(target=store_workers.ledger_once_while_locked, args=(str(root), recorded, out))
        proc.start()
        assert recorded.wait(60)
        assert out.get(timeout=10) == (False, 1)
        assert ledger.read()["months"] == {}
    proc.join(60)
    assert proc.exitcode == 0
    counts = next(iter(ledger.read()["months"].values()))["gemini"]["models"]["model-a"]
    assert (counts["requests"], counts["input_uncached"], counts["output"]) == (1, 10, 1)


def test_turn_line_wording():
    turn = [Usage(310, 2900, 0, 412)]
    conversation = {"input_uncached": 970, "cache_read": 8900, "cache_write": 0, "output": 1230}
    assert format_turn_line(turn, conversation, "Gemini") == \
        "tokens 3k in (3k from cache) · 412 out · conversation 10k in · 1k out"
    assert format_turn_line([not_reported()], conversation, "Mistral") == "tokens not reported by Mistral"


@pytest.mark.parametrize(("value", "text"), [
    (0, "0"), (999, "999"), (1000, "1k"), (1499, "1k"), (1500, "2k"), (26_400, "26k"), (249_600, "250k"),
    (999_499, "999k"), (999_500, "1M"), (1_000_000, "1M"), (1_249_999, "1.2M"), (1_250_000, "1.3M"),
    (12_345_678, "12.3M"),
])
def test_token_numbers_are_rounded(value, text):
    assert short_number(value) == text


def test_the_window_token_line():
    """Owner GUI feedback 2026-11-08 (2): "in" counts every input token (uncached, read from and written to the
    cache)."""
    turn = [Usage(20_000, 5_000, 1_000, 6_000), Usage(0, 0, 0, 2_000)]
    conversation = {"input_uncached": 200_000, "cache_read": 40_000, "cache_write": 10_000, "output": 5_000}
    assert format_token_line(turn, conversation, "Anthropic") == \
        "This question: 26k in, 8k out · Whole chat: 250k in, 5k out"
    assert format_token_line(None, conversation, "Anthropic") == "Whole chat: 250k in, 5k out"
    assert format_token_line([not_reported()], {}, "Mistral") == "This question: tokens not reported by Mistral"
    assert format_token_line(None, {}, "Mistral") == ""


# --- imports -----------------------------------------------------------------------------------------------------
def test_the_store_and_its_purge_import_no_ai_library(root):
    code = ("import sys; from pathlib import Path; from flextool.assistant import store; "
            f"store.purge_at_start(Path({str(root)!r})); "
            "print(sorted(m for m in ('anthropic', 'openai', 'keyring', 'httpx2', 'flextool.assistant.session', "
            "'flextool.assistant.probes', 'flextool.assistant.keys') if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
