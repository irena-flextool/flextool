"""Functions the store tests run in other processes (``multiprocessing`` with ``spawn``: importable, top level).

Each worker opens the store at the root it is given, never the user's config folder.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def save_loop(root: str, project: str, rounds: int, ready, go, finish, out) -> None:
    """Start a conversation, report its id, then save it *rounds* times (a new message each time) as fast as it
    can; keep it open until *finish*."""
    from flextool.assistant.llm_adapter import Message
    from flextool.assistant.store import Store

    store = Store(Path(root))
    conv = store.new_conversation(Path(project), ("ollama", "http://localhost:11434/v1", "m"), "standard")
    out.put(conv.id)
    ready.set()
    go.wait(30)
    written = 0
    for i in range(rounds):
        conv.messages.append(Message("user", f"question {i}"))
        written += bool(store.save(conv))
    out.put(written)
    finish.wait(30)
    store.close(conv)


def purge_loop(root: str, ready, stop, out) -> None:
    """Purge again and again, as if a year had passed, until *stop*."""
    from flextool.assistant.store import Store

    store = Store(Path(root))
    later = datetime.now(timezone.utc) + timedelta(days=365)
    ready.set()
    count = 0
    while not stop.is_set():
        store.purge_expired(later)
        count += 1
    out.put(count)


def new_projects(root: str, projects: list[str], ready, go, finish, out) -> None:
    """Start and save one conversation in each project, keeping all open until *finish*."""
    from flextool.assistant.llm_adapter import Message
    from flextool.assistant.store import Store

    store = Store(Path(root))
    ready.set()
    go.wait(30)
    convs = []
    for project in projects:
        conv = store.new_conversation(Path(project), None, None)
        conv.messages.append(Message("user", "hello"))
        store.save(conv)
        convs.append(conv)
        time.sleep(0.005)
    out.put([c.id for c in convs])
    finish.wait(30)
    for conv in convs:
        store.close(conv)


def open_at_once(root: str, project: str, conversation_id: str, barrier, finish, out) -> None:
    """Open one conversation at the same moment as the other workers (``if_locked="raise"``); report the outcome,
    keep a won conversation open until *finish*."""
    from flextool.assistant.store import ConversationLocked, Store

    store = Store(Path(root))
    barrier.wait(30)
    try:
        conv = store.open(Path(project), conversation_id, if_locked="raise")
    except ConversationLocked as exc:
        out.put(("locked", exc.owner))
        return
    out.put(("opened", conv.id))
    finish.wait(30)
    store.close(conv)


def hold_open(root: str, project: str, conversation_id: str, ready, finish) -> None:
    """Keep one conversation open until *finish*."""
    from flextool.assistant.store import Store

    store = Store(Path(root))
    conv = store.open(Path(project), conversation_id, if_locked="raise")
    ready.set()
    finish.wait(30)
    store.close(conv)


def ledger_loop(root: str, requests: int, go) -> None:
    """Count *requests* requests of 10 input and 1 output token each in the ledger."""
    from flextool.assistant.llm_adapter import Usage
    from flextool.assistant.usage import Ledger

    ledger = Ledger(Path(root))
    go.wait(30)
    for _ in range(requests):
        ledger.record("gemini", "Google Gemini", "model-a", Usage(10, 0, 0, 1))


def ledger_once_while_locked(root: str, recorded, out) -> None:
    """Count one request while the parent holds the ledger's lock (longer than a writer waits): the update fails,
    the count is kept, and the exit of this process writes it once the lock is free."""
    from flextool.assistant import usage
    from flextool.assistant.llm_adapter import Usage

    written = usage.Ledger(Path(root)).record("gemini", "Google Gemini", "model-a", Usage(10, 0, 0, 1))
    out.put((written, usage.pending_count()))
    recorded.set()


def record_consents(root: str, tag: str, count: int, ready, go, out) -> None:
    """Accept the data notice for *count* endpoints of its own (``https://{tag}-{i}.example.test/v1``) as fast as
    it can, as two FlexTool windows or a window and a terminal would at the same moment."""
    from flextool.assistant import endpoint
    from flextool.assistant.providers import PRESETS

    preset = PRESETS["openai_compatible"]
    ready.set()
    go.wait(30)
    for i in range(count):
        base_url = f"https://{tag}-{i}.example.test/v1"
        route = endpoint.route_for(preset, base_url)
        endpoint.record_consent(endpoint.consent_key(preset.id, base_url, route), Path(root))
    out.put(count)


def take_stale_lock(lock_path: str, go, taken, finish, out) -> None:
    """When *go* is set, take the short lock at *lock_path* (the test has made it older than the stale age), report
    the new lock's token and keep it until *finish*."""
    from flextool.assistant._files import LOCK_STALE_AFTER, PidLock

    go.wait(30)
    lock = PidLock(Path(lock_path), stale_after=LOCK_STALE_AFTER)
    lock.acquire(10)
    out.put(lock.token)
    taken.set()
    finish.wait(30)
    lock.release()


def new_unsaved_chat(root: str, project: str, ready, finish, out) -> None:
    """Allocate a chat number and start that chat without saving it (as the window does at a first question);
    report the number and keep the chat open until *finish*."""
    from flextool.assistant.store import Store

    store = Store(Path(root))
    number = store.allocate_chat_number(Path(project))
    conv = store.new_conversation(Path(project), None, None, number=number)
    out.put(number)
    ready.set()
    finish.wait(30)
    store.close(conv)
