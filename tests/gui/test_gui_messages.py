"""The GUI message list's Tk-free core (``flextool/gui/messages.py``; design ``gui_message_list_design.md`` §3, §5).

Stub sources stand in for the real ones (packages M3a, M3b, P7b); every store lives in ``tmp_path`` (the shared
``FLEXTOOL_USER_HOME`` of a worker would leak dismissals between tests).  Expected values follow from the rules
of the design, written next to each case, never from this code's output.
"""
from __future__ import annotations

import json
import ntpath
import os
import posixpath
import re
import subprocess
import sys
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pytest

from flextool.gui import messages as m
from flextool.gui.messages import (
    MESSAGE_SPECS,
    GuiMessage,
    MessageCenter,
    MessageSpec,
    MessageStore,
    database_key,
    docs_page_url,
    message_docs_url,
    project_key,
)

REPO = Path(__file__).resolve().parents[2]
DAY0 = date(2026, 1, 1)
UPGRADED = MESSAGE_SPECS["database_upgraded"]
W0490 = MESSAGE_SPECS["value_list_check"]
UNREADABLE = MESSAGE_SPECS["database_unreadable"]
UPDATE = MESSAGE_SPECS["update_available"]
#: W-0491's spec (P7b registers the real one): a non-dismissable note, not an attention message.
NOTE_STUB = MessageSpec("list_defaults_lost_stub", "database", dismissable=False, attention=False, docs="W-0490")


def _msg(spec: MessageSpec, subject: str, instance: str = "") -> GuiMessage:
    return GuiMessage(spec, subject, subject.rsplit("/", 1)[-1], "A title", ("One sentence.",), instance=instance)


def _upgraded_source(store: MessageStore):
    """The shape of M3a's source: one message per pending ``database_upgraded`` entry, instance = its run."""
    for subject, entry in store.pending("database_upgraded").items():
        yield GuiMessage(UPGRADED, subject, entry["label"], "This database was upgraded",
                         (f"From {entry['facts'].get('from')} to {entry['facts'].get('to')}.",), instance=entry["run"])


def _static(states: list[GuiMessage]):
    """A source returning the current content of *states* (tests change the list between recomputes)."""
    return lambda _store: list(states)


def _db(folder: Path, name: str) -> str:
    """The key of a database file that exists (a key of a missing file in an existing folder is pruned)."""
    (folder / name).write_bytes(b"")
    return database_key(folder / name)


def _center(store: Path, *sources, today: date = DAY0, rank=None) -> MessageCenter:
    return MessageCenter(store, sources, today=lambda: today, rank=rank)


def _open_project(store: Path, *sources, today: date = DAY0) -> str | None:
    """One project open: recompute, and when the auto-show opens the window, close it after it listed everything."""
    center = _center(store, *sources, today=today)
    key = center.recompute("project_open")
    if key is not None:
        center.window_closed([msg.key for msg in center.messages], opened_by="project_open")
    return key


# -- no Tk, the registry, docs links -----------------------------------------------------------------------------
def test_messages_imports_no_tk():
    code = ("import sys; sys.modules['tkinter'] = None; import flextool.gui.messages; "
            "assert not any(k == 'tkinter' or k.startswith('tkinter.') for k in sys.modules if sys.modules[k])")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def _slug(heading: str) -> str:
    """Python-Markdown's default toc slug (mkdocs.yml sets no other)."""
    text = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _anchors(page: Path) -> set[str]:
    anchors, fenced = set(), False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and (hit := re.match(r"#{1,6}\s+(.*?)\s*$", line)):
            anchors.add(_slug(hit.group(1)))
    return anchors


def test_registry_kinds_and_their_docs_exist():
    from flextool.run_warnings.registry import WARNINGS

    # Design §3.1: W-0490 and the unreadable file are attention messages without Dismiss; the event kinds and
    # the update state are dismissable and capped.
    assert (W0490.dismissable, W0490.attention) == (False, True)
    assert (UNREADABLE.dismissable, UNREADABLE.attention) == (False, True)
    for spec_id in ("database_upgraded", "solver_defaults_updated", "update_available"):
        assert (MESSAGE_SPECS[spec_id].dismissable, MESSAGE_SPECS[spec_id].attention) == (True, False)
    for spec in MESSAGE_SPECS.values():
        if spec.docs.startswith("W-"):
            assert spec.docs in WARNINGS
            continue
        page, _, anchor = spec.docs.partition("#")
        assert (REPO / "docs" / page).is_file(), spec
        assert anchor in _anchors(REPO / "docs" / page), spec


def test_docs_urls():
    from flextool.docs_build.parameters import WEB_PAGE_URL
    from flextool.run_warnings.registry import DOCS_BASE_URL, WARNINGS_DOCS_URL

    site = "https://irena-flextool.github.io/flextool/"
    assert DOCS_BASE_URL == site == WEB_PAGE_URL.rsplit("parameter_catalogue/", 1)[0]
    assert WARNINGS_DOCS_URL == site + "reference/warnings/"
    assert (docs_page_url("flextool_gui_interface.md#migration-and-frozen-databases")
            == site + "flextool_gui_interface/#migration-and-frozen-databases")
    assert docs_page_url("how_to/stochastics.md") == site + "how_to/stochastics/"
    assert docs_page_url("index.md") == site
    assert message_docs_url(W0490) == site + "reference/warnings/#w-0490"


def test_register_spec_refuses_a_different_spec_under_a_used_id():
    m.register_spec(UPGRADED)       # the same spec again is fine
    with pytest.raises(ValueError):
        m.register_spec(MessageSpec("database_upgraded", "database", dismissable=False, attention=False, docs="x"))


# -- keys ------------------------------------------------------------------------------------------------------------
def test_windows_keys_ignore_case_and_slashes_but_not_unc():
    # ntpath.normcase lowercases and turns "/" into "\": one file under one drive letter is one key.
    assert database_key(r"Z:\Proj\Model.sqlite", pathmod=ntpath) == r"db:z:\proj\model.sqlite"
    assert database_key("z:/proj/model.sqlite", pathmod=ntpath) == r"db:z:\proj\model.sqlite"
    # The mapped drive and its UNC path are not unified (never resolved: design §3.3).
    assert database_key(r"\\server\share\proj\model.sqlite", pathmod=ntpath) == r"db:\\server\share\proj\model.sqlite"
    assert project_key(r"Z:\Projects\P", pathmod=ntpath) == r"project:z:\projects\p"


def test_posix_keys_are_absolute_case_kept_and_not_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert database_key("x.sqlite", pathmod=posixpath) == "db:" + posixpath.join(os.getcwd(), "x.sqlite")
    assert database_key("/a/Model.sqlite", pathmod=posixpath) != database_key("/a/model.sqlite", pathmod=posixpath)
    if sys.platform != "win32":
        (tmp_path / "real.sqlite").write_bytes(b"")
        (tmp_path / "link.sqlite").symlink_to(tmp_path / "real.sqlite")
        assert database_key(tmp_path / "link.sqlite") == f"db:{tmp_path}/link.sqlite"


# -- store -----------------------------------------------------------------------------------------------------------
def test_store_round_trip_and_dismiss(tmp_path):
    path = tmp_path / "gui_messages.json"
    store = MessageStore(path, today=lambda: DAY0)
    store.post("database_upgraded", "db:/p/a.sqlite", "a.sqlite", run="r1", facts={"from": 80, "to": 82})
    again = MessageStore(path)
    assert again.pending("database_upgraded") == {
        "db:/p/a.sqlite": {"run": "r1", "posted": "2026-01-01", "facts": {"from": 80, "to": 82}, "label": "a.sqlite"}}
    # A copy of the file (another path) is another subject: nothing pending for it.
    assert "db:/p/a_repaired.sqlite" not in again.pending("database_upgraded")
    center = _center(path, _upgraded_source)
    center.recompute("project_open")
    center.dismiss(center.messages[0])
    assert MessageStore(path).pending("database_upgraded") == {} and center.messages == ()


def test_corrupt_file_counts_as_empty_and_is_replaced(tmp_path):
    path = tmp_path / "gui_messages.json"
    path.write_text("{not json", encoding="utf-8")
    store = MessageStore(path)
    assert store.pending("database_upgraded") == {}
    store.post("database_upgraded", "db:/p/a.sqlite", "a.sqlite", run="r1")
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == 1


def test_unknown_format_is_read_as_empty_and_never_overwritten(tmp_path):
    path = tmp_path / "gui_messages.json"
    newer = '{"format": 2, "subjects": {}}'
    path.write_text(newer, encoding="utf-8")
    store = MessageStore(path)
    store.post("database_upgraded", "db:/p/a.sqlite", "a.sqlite", run="r1")
    assert path.read_text(encoding="utf-8") == newer
    assert list(store.pending("database_upgraded")) == ["db:/p/a.sqlite"]       # held for the session


def test_two_writers_keep_each_others_entries(tmp_path):
    path = tmp_path / "gui_messages.json"
    first, second = MessageStore(path), MessageStore(path)      # both loaded before either writes
    first.post("database_upgraded", "db:/p/a.sqlite", "a.sqlite", run="r1")
    second.post("database_upgraded", "db:/p/b.sqlite", "b.sqlite", run="r2")
    assert set(MessageStore(path).pending("database_upgraded")) == {"db:/p/a.sqlite", "db:/p/b.sqlite"}


def test_replace_retried_once_then_kept_in_memory(tmp_path, monkeypatch):
    path = tmp_path / "gui_messages.json"
    real_replace, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise PermissionError("another window reads it")
        real_replace(src, dst)

    monkeypatch.setattr(m.os, "replace", flaky)
    MessageStore(path).post("database_upgraded", "db:/p/a.sqlite", "a.sqlite", run="r1")
    assert len(calls) == 2 and "db:/p/a.sqlite" in MessageStore(path).pending("database_upgraded")

    def locked(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(m.os, "replace", locked)
    store = MessageStore(path)
    store.post("database_upgraded", "db:/p/b.sqlite", "b.sqlite", run="r2")      # must not raise
    assert set(store.pending("database_upgraded")) == {"db:/p/a.sqlite", "db:/p/b.sqlite"}
    assert set(MessageStore(path).pending("database_upgraded")) == {"db:/p/a.sqlite"}
    assert [p.name for p in tmp_path.iterdir()] == ["gui_messages.json"]          # no temporary file left
    monkeypatch.setattr(m.os, "replace", real_replace)
    store.post("database_upgraded", "db:/p/c.sqlite", "c.sqlite", run="r3")      # the unsaved post is replayed
    assert set(MessageStore(path).pending("database_upgraded")) == {"db:/p/a.sqlite", "db:/p/b.sqlite",
                                                                    "db:/p/c.sqlite"}


def test_unwritable_folder_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    store = MessageStore(blocker / "gui_messages.json")         # its folder cannot be made: a file is in the way
    store.post("database_upgraded", "db:/p/a.sqlite", "a.sqlite", run="r1")
    assert list(store.pending("database_upgraded")) == ["db:/p/a.sqlite"]


def test_offered_version(tmp_path):
    path = tmp_path / "gui_messages.json"
    MessageStore(path).set_offered_version("4.6.4")
    assert MessageStore(path).offered_version() == "4.6.4"
    MessageStore(path).set_offered_version(None)
    assert MessageStore(path).offered_version() is None


def test_pruning(tmp_path):
    path = tmp_path / "gui_messages.json"
    folder = tmp_path / "projects"
    folder.mkdir()
    kept_file = folder / "kept.sqlite"
    kept_file.write_bytes(b"")
    old_file = folder / "old.sqlite"
    old_file.write_bytes(b"")
    subjects = {
        "deleted": database_key(folder / "deleted.sqlite"),            # folder exists, file gone → dropped
        "offline": database_key(tmp_path / "offline_drive" / "x.sqlite"),   # folder gone too → kept
        "kept": database_key(kept_file),
        "old": database_key(old_file),
    }
    store = MessageStore(path, today=lambda: DAY0)
    for name, key in subjects.items():
        if name != "old":
            store.post("database_upgraded", key, name, run="r")
    MessageStore(path, today=lambda: DAY0 - timedelta(days=1)).post("database_upgraded", subjects["old"], "old",
                                                                     run="r")
    # 180 days after DAY0 the entries written on DAY0 stay; 181 days after the day before DAY0, "old" goes.
    _center(path, today=DAY0 + timedelta(days=180))
    assert set(MessageStore(path).pending("database_upgraded")) == {subjects["offline"], subjects["kept"]}
    _center(path, today=DAY0 + timedelta(days=181))
    assert set(MessageStore(path).pending("database_upgraded")) == set()


def test_rename_prefix_moves_pending_shown_and_dismissals(tmp_path):
    path = tmp_path / "gui_messages.json"
    old, new = tmp_path / "P", tmp_path / "Q"
    db_old, db_sibling = database_key(old / "input_sources" / "x.sqlite"), database_key(tmp_path / "P2" / "y.sqlite")
    store = MessageStore(path)
    store.post("database_upgraded", db_old, "x.sqlite", run="r1")
    store.post("database_upgraded", db_sibling, "y.sqlite", run="r1")
    store.post("solver_defaults_updated", project_key(old), "P")
    store.count_shown([_msg(UPGRADED, db_old, "r1")])
    store.resolve(_msg(UPDATE, project_key(old), "4.6.4"))     # a dismissed state under the project
    store.rename_prefix(old, new)
    after = MessageStore(path)
    db_new = database_key(new / "input_sources" / "x.sqlite")
    assert set(after.pending("database_upgraded")) == {db_new, db_sibling}       # "P2" is not under "P"
    assert after.pending("solver_defaults_updated") == {
        project_key(new): {"run": "", "posted": date.today().isoformat(), "facts": {}, "label": "Q"}}
    assert after.shown(_msg(UPGRADED, db_new, "r1")) == 1
    assert after.is_dismissed(_msg(UPDATE, project_key(new), "4.6.4"))


# -- the center: order, decisions, dismiss -----------------------------------------------------------------------
def test_order_attention_first_then_database_by_rank_then_project_then_user(tmp_path):
    states = [_msg(UPDATE, "user", "4.6.4"), _msg(MESSAGE_SPECS["solver_defaults_updated"], "project:/P"),
              _msg(UPGRADED, "db:/P/b.sqlite", "r"), _msg(UPGRADED, "db:/P/a.sqlite", "r"),
              _msg(W0490, "db:/P/b.sqlite"), _msg(UNREADABLE, "db:/P/a.sqlite")]
    rank = {"db:/P/a.sqlite": 1, "db:/P/b.sqlite": 2}.get
    center = _center(tmp_path / "s.json", _static(states), rank=lambda k: rank(k, 0))
    center.recompute()
    assert [(msg.spec.id, msg.subject_key) for msg in center.messages] == [
        ("database_unreadable", "db:/P/a.sqlite"), ("value_list_check", "db:/P/b.sqlite"),
        ("database_upgraded", "db:/P/a.sqlite"), ("database_upgraded", "db:/P/b.sqlite"),
        ("solver_defaults_updated", "project:/P"), ("update_available", "user")]


def test_auto_show_decision_table(tmp_path):
    path = tmp_path / "s.json"
    states = [_msg(UPGRADED, "db:/P/a.sqlite", "r1")]
    center = _center(path, _static(states))
    first = states[0].key
    assert center.recompute("project_open") == first                     # new at project open → shown at it
    center.window_closed([first], opened_by="project_open")
    assert center.recompute("refresh") is None                           # closed for now, same set → none
    states.append(_msg(UPGRADED, "db:/P/b.sqlite", "r1"))
    assert center.recompute("refresh") == states[1].key                  # refresh with a new key → at that one
    center.window_closed([states[1].key])
    states.append(_msg(UPGRADED, "db:/P/c.sqlite", "r2"))
    assert center.recompute("other") is None                             # other, new non-attention → indicator
    assert center.indicator() == (3, True)                               # 3 active, one unseen
    states.append(_msg(W0490, "db:/P/d.sqlite"))
    assert center.recompute("other") == states[3].key                    # other, new attention → shown
    center.window_closed([msg.key for msg in center.messages])
    assert center.recompute("other") is None                             # no longer new
    assert center.indicator() == (4, True)                               # an attention message keeps the accent
    assert _center(path, _static(states)).recompute("project_open") == states[3].key    # reopened → again
    states[:] = []
    assert _center(path, _static(states)).recompute("project_open") is None             # nothing active → none


def test_indicator_without_accent_once_all_seen(tmp_path):
    states = [_msg(UPGRADED, "db:/P/a.sqlite", "r1")]
    center = _center(tmp_path / "s.json", _static(states))
    center.recompute("project_open")
    center.window_closed([states[0].key])
    assert center.indicator() == (1, False)


def test_dismissed_state_hides_only_that_instance(tmp_path):
    path = tmp_path / "s.json"
    states = [_msg(UPDATE, "user", "4.6.4")]
    center = _center(path, _static(states))
    center.recompute("project_open")
    center.dismiss(center.messages[0])
    assert _center(path, _static(states)).recompute("project_open") is None
    states[:] = [_msg(UPDATE, "user", "4.6.5")]                           # a newer offered version → shown
    assert _center(path, _static(states)).recompute("project_open") == states[0].key


def test_non_dismissable_message_cannot_be_dismissed(tmp_path):
    center = _center(tmp_path / "s.json", _static([_msg(W0490, "db:/P/a.sqlite")]))
    center.recompute()
    with pytest.raises(ValueError):
        center.dismiss(center.messages[0])
    with pytest.raises(ValueError):
        center.post_event("value_list_check", "db:/P/a.sqlite", "a.sqlite")
    assert not (tmp_path / "s.json").exists()        # an attention message has no store entry at all


def test_a_failing_source_is_skipped(tmp_path):
    def broken(_store):
        raise RuntimeError("boom")

    center = _center(tmp_path / "s.json", broken, _static([_msg(W0490, "db:/P/a.sqlite")]))
    assert center.recompute("project_open") == "value_list_check|db:/P/a.sqlite|"


# -- the auto-show cap (decisions 18 and 20) ---------------------------------------------------------------------
def test_dismissable_event_stops_auto_opening_at_the_third_open(tmp_path):
    path = tmp_path / "s.json"
    key = _db(tmp_path, "a.sqlite")
    MessageStore(path).post("database_upgraded", key, "a.sqlite", run="r1", facts={"from": 80, "to": 82})
    assert _open_project(path, _upgraded_source) is not None             # open 1 (shown = 1)
    assert _open_project(path, _upgraded_source) is not None             # open 2 (shown = 2)
    center = _center(path, _upgraded_source)
    assert center.recompute("project_open") is None                      # open 3: capped
    assert center.recompute("refresh") is None                           # Refresh does not lift the cap
    assert [msg.key for msg in center.messages] == [f"database_upgraded|{key}|r1"]
    assert center.indicator() == (1, True)                               # still listed, counted, unseen
    # The same file upgraded again: a new run id is a new message, uncapped again.
    MessageStore(path).post("database_upgraded", key, "a.sqlite", run="r2")
    assert _open_project(path, _upgraded_source) == f"database_upgraded|{key}|r2"
    assert MessageStore(path).shown(_msg(UPGRADED, key, "r2")) == 1


def test_only_the_project_open_auto_show_counts(tmp_path):
    path = tmp_path / "s.json"
    states = [_msg(UPGRADED, _db(tmp_path, "a.sqlite"), "r1")]
    for _ in range(3):                    # each time the user opens the list from the indicator / a Refresh
        center = _center(path, _static(states))
        center.recompute("project_open")
        center.window_closed([states[0].key], opened_by=None)
        center.window_closed([states[0].key], opened_by="refresh")
    assert _center(path, _static(states)).recompute("project_open") == states[0].key


def test_attention_messages_are_never_capped(tmp_path):
    path = tmp_path / "s.json"
    for spec in (W0490, UNREADABLE):
        states = [_msg(spec, database_key(tmp_path / f"{spec.id}.sqlite"))]
        for _ in range(3):
            assert _open_project(path, _static(states)) == states[0].key
        assert _open_project(path, _static(states)) == states[0].key     # the 4th open still shows it
    assert not path.exists()                                             # and nothing was stored for them


def test_two_upgraded_databases_are_two_messages(tmp_path):
    path = tmp_path / "s.json"
    a, b = _db(tmp_path, "a.sqlite"), _db(tmp_path, "b.sqlite")
    store = MessageStore(path)
    store.post("database_upgraded", a, "a.sqlite", run="r1")
    store.post("database_upgraded", b, "b.sqlite", run="r1")
    center = _center(path, _upgraded_source)
    center.recompute("project_open")
    assert len(center.messages) == 2
    center.dismiss(next(msg for msg in center.messages if msg.subject_key == a))
    assert [msg.subject_key for msg in center.messages] == [b]
    assert set(MessageStore(path).pending("database_upgraded")) == {b}


def test_non_dismissable_note_is_capped_but_stays(tmp_path):
    path = tmp_path / "s.json"
    key = _db(tmp_path, "a.sqlite")
    states = [_msg(NOTE_STUB, key, "hash1")]
    assert _open_project(path, _static(states)) is not None
    assert _open_project(path, _static(states)) is not None
    center = _center(path, _static(states))
    assert center.recompute("project_open") is None                      # the third open does not show it
    assert [msg.key for msg in center.messages] == [states[0].key]       # still active and counted
    assert center.indicator() == (1, True)
    with pytest.raises(ValueError):                                      # and has no Dismiss
        center.dismiss(center.messages[0])
    states[:] = [_msg(NOTE_STUB, key, "hash2")]                          # a changed set of empty defaults
    assert _open_project(path, _static(states)) == states[0].key
