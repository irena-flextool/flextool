"""The GUI message list's Tk-free core (``flextool/gui/messages.py``; design ``gui_message_list_design.md`` §3, §5).

Stub sources stand in for the sources of later packages (M3b, P7b), next to the real M3a ones
(``message_sources.py``, at the end); every store lives in ``tmp_path`` (the shared
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
from types import SimpleNamespace

import pytest

from flextool.gui import message_sources as ms
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
from flextool.gui.message_sources import UpgradeResult

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


# == M3a: the real sources (flextool/gui/message_sources.py) =========================================================
# Expected texts are the inventory's (specs/gui_message_list_inventory.md L2a, L2b, L3, L4); the data versions come
# from how each case is built.
_SENTENCE_END = re.compile(r"[.!?](\s|$)")


def _check_style(msg: GuiMessage) -> None:
    """Design §4.5 / §5: a title of at most 60 characters, at most 4 sentences in the whole message."""
    assert len(msg.title) <= 60, msg.title
    assert sum(len(_SENTENCE_END.findall(p)) for p in msg.paragraphs) <= 4, msg.paragraphs


def _result(path: Path, before, after, *, upgraded=True, failed=False, copy_of=None, messages=()) -> UpgradeResult:
    return UpgradeResult(path.name, path, before, after, copy_of, upgraded, failed, list(messages))


def test_upgraded_messages_from_upgrade_results(tmp_path):
    store_path = tmp_path / "s.json"
    project = tmp_path / "P" / "input_sources"
    project.mkdir(parents=True)
    for name in ("a.sqlite", "schema.sqlite", "a_v82.sqlite", "failed.sqlite", "same.sqlite"):
        (project / name).write_bytes(b"")
    other = tmp_path / "other.sqlite"                        # upgraded, but not a source of this project
    other.write_bytes(b"")
    results = [
        _result(project / "a.sqlite", 80, 82),
        _result(project / "schema.sqlite", 82, 82),          # SpineDB schema only: same data version
        _result(project / "a_v82.sqlite", 80, 82, copy_of="frozen.sqlite"),
        _result(project / "failed.sqlite", 80, 80, upgraded=False, failed=True, messages=["failed.sqlite: x"]),
        _result(project / "same.sqlite", 82, 82, upgraded=False),
        _result(other, 80, 82),
    ]
    center = _center(store_path, ms.upgraded_source(lambda: [database_key(p) for p in project.iterdir()]))
    ms.post_upgrades(center, results, "2026-11-23T10:15:02")
    center.recompute("project_open")
    by_label = {msg.subject_label: msg for msg in center.messages}
    # One message per upgraded database of the project; neither the failed, the unchanged nor the other one.
    assert sorted(by_label) == ["a.sqlite", "a_v82.sqlite", "schema.sqlite"]
    assert by_label["a.sqlite"].title == "This database was upgraded"
    assert "from data version 80 to 82" in by_label["a.sqlite"].paragraphs[0]
    assert "data version" not in by_label["schema.sqlite"].paragraphs[0]
    assert "database format" in by_label["schema.sqlite"].paragraphs[0]
    assert by_label["a_v82.sqlite"].title == "An upgraded copy was added"
    assert "frozen database frozen.sqlite (from data version 80 to 82)" in by_label["a_v82.sqlite"].paragraphs[0]
    for msg in center.messages:
        assert msg.instance == "2026-11-23T10:15:02" and msg.spec.dismissable and not msg.spec.attention
        _check_style(msg)
    facts = MessageStore(store_path).pending("database_upgraded")
    assert facts[database_key(project / "schema.sqlite")]["facts"]["schema"] is True
    assert "schema" not in facts[database_key(project / "a.sqlite")]["facts"]
    # A newer run of the same file replaces its message (one per database).
    ms.post_upgrades(center, [_result(project / "a.sqlite", 82, 83)], "2026-11-24T08:00:00")
    center.recompute("other")
    again = [msg for msg in center.messages if msg.subject_label == "a.sqlite"]
    assert len(again) == 1 and "from data version 82 to 83" in again[0].paragraphs[0]


def test_failure_lines_route_failed_and_unexplained_results():
    p = Path("/p")
    # The shape of check_and_upgrade_database's failure message.
    tb = "x.sqlite: migration FAILED — copy the traceback below.\n\nError: boom\n\nTraceback:\n..."
    results = [
        _result(p / "x.sqlite", 80, 80, upgraded=False, failed=True, messages=[tb]),
        _result(p / "y.sqlite", 80, 80, upgraded=False, messages=["Cannot check database version (spinedb_api)"]),
        _result(p / "z.sqlite", 80, 82, messages=["z.sqlite: FlexTool data upgraded from version 80 to 82."]),
    ]
    # The traceback stays in the Execution window (and the text says so); the error line is kept.
    failed = "x.sqlite: migration FAILED — copy the traceback in the Execution window. Error: boom"
    assert ms.failure_lines(results, cancelled=False) == [failed, "Cannot check database version (spinedb_api)"]
    # After the user's Cancel a restored file is not a failure; a failed one still is.
    assert ms.failure_lines(results, cancelled=True) == [failed]


def _empty_sqlite(path: Path) -> Path:
    """A valid SQLite file that is not a Spine database: its data version cannot be read."""
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    with con:
        con.execute("CREATE TABLE t (x)")
    con.close()
    return path


def test_planning_note_kinds(tmp_path, monkeypatch):
    from flextool.gui.data_models import ProjectSettings
    from flextool.gui.input_sources import InputSourceManager
    from flextool.update_flextool.frozen import FrozenCheckError

    project = tmp_path / "project"
    _empty_sqlite(project / "input_sources" / "internal.sqlite")
    _empty_sqlite(tmp_path / "shared" / "external.sqlite")
    settings = ProjectSettings()
    settings.external_refs = {"external.sqlite": "../shared/external.sqlite"}
    mgr = InputSourceManager(project, settings)
    internal, _external, _frozen, notes = mgr.plan_db_migrations()
    # The internal file is still migrated (no note); the external one is skipped with an "unreadable" note.
    assert [name for name, _ in internal] == ["internal.sqlite"]
    assert [(n.kind, n.name) for n in notes] == [("unreadable", "external.sqlite")]
    assert mgr.planning_notes == notes
    center = _center(tmp_path / "s.json", ms.unreadable_source(lambda: mgr.planning_notes))
    assert center.recompute("other") is not None             # a new attention message opens the list
    (msg,) = center.messages
    assert (msg.title, msg.subject_label) == ("FlexTool cannot read this database file", "external.sqlite")
    assert msg.subject_key == database_key((tmp_path / "shared" / "external.sqlite").resolve())
    assert not msg.spec.dismissable and msg.spec.attention
    _check_style(msg)

    # A frozen marker that cannot be read: the frozen variant, for an internal file too.
    def locked(path, _target):
        raise FrozenCheckError(Path(path), "database is locked")

    monkeypatch.setattr(InputSourceManager, "_frozen_needing_migration", staticmethod(locked))
    notes = mgr.plan_db_migrations()[3]
    assert sorted((n.kind, n.name) for n in notes) == [("frozen_unreadable", "external.sqlite"),
                                                      ("frozen_unreadable", "internal.sqlite")]
    center.recompute("refresh")
    assert {msg.title for msg in center.messages} == {"FlexTool cannot tell whether this file is frozen"}
    assert all("database is locked" in msg.paragraphs[-1] for msg in center.messages)
    for msg in center.messages:
        _check_style(msg)

    # A frozen file the user declined: nothing (the tree shows it); notes are replaced at each plan.
    def needs(path, target):
        return SimpleNamespace(label="L", version=target - 1), target - 1, False

    monkeypatch.setattr(InputSourceManager, "_frozen_needing_migration", staticmethod(needs))
    for path in mgr._registered_sqlite_paths():
        mgr.decline_frozen(path)
    assert mgr.plan_db_migrations() == ([], [], [], []) and mgr.planning_notes == []
    center.recompute("refresh")
    assert center.messages == ()


def test_solver_settings_message_is_the_shared_notice(tmp_path):
    from flextool.gui.settings_io import solver_defaults_message, solver_defaults_notice

    paragraphs = solver_defaults_message("my_project")
    # With the "Affected projects" part (the subject line) put back, it is the printed line word for word.
    assert " ".join((paragraphs[0], "Affected projects: my_project.", *paragraphs[1:])) == solver_defaults_notice(
        "my_project")
    assert paragraphs[0].startswith("Behaviour changed in FlexTool 4.2.0")
    assert "else mip_rel_gap in solver_config/highs.opt" in paragraphs[0]
    assert paragraphs[-1].startswith("To keep the old behaviour") and "shown once" not in " ".join(paragraphs)

    project_dir = tmp_path / "projects" / "my_project"
    project_dir.mkdir(parents=True)
    center = _center(tmp_path / "s.json", ms.solver_defaults_source(project_dir, "my_project"))
    assert center.recompute("project_open") is None
    center.post_event("solver_defaults_updated", project_key(project_dir), "Project my_project", run="r1")
    assert center.recompute("project_open") is not None
    (msg,) = center.messages
    assert (msg.title, msg.subject_label, msg.paragraphs) == (
        "This project's solver settings were updated", "Project my_project", paragraphs)
    _check_style(msg)
    center.dismiss(msg)
    assert MessageStore(tmp_path / "s.json").pending("solver_defaults_updated") == {}


def test_update_source(tmp_path, monkeypatch):
    path = tmp_path / "s.json"
    flags = {"enabled": True, "git": False, "installed": "4.6.3"}

    def newer(v):
        return tuple(map(int, v.split("."))) > tuple(map(int, flags["installed"].split(".")))

    source = ms.update_source(enabled=lambda: flags["enabled"], is_git=lambda: flags["git"],
                              installed=lambda: flags["installed"], is_newer=newer)
    center = _center(path, source)
    center.recompute("other")
    assert center.messages == ()                             # nothing stored
    MessageStore(path).set_offered_version("4.6.4")
    center = _center(path, source)
    center.recompute("project_open")
    (msg,) = center.messages
    assert msg.title == "A newer FlexTool is available" and msg.instance == "4.6.4"
    assert msg.paragraphs[0].startswith("FlexTool 4.6.4 is available; you have 4.6.3.")
    _check_style(msg)
    flags["installed"] = "4.6.4"                             # installed meanwhile: gone by itself
    center.recompute("other")
    assert center.messages == ()
    flags.update(installed="4.6.3", git=True)                # a git checkout: the button only
    center.recompute("other")
    assert center.messages == ()
    flags.update(git=False, enabled=False)                   # the start-up check is off
    center.recompute("other")
    assert center.messages == ()
    monkeypatch.setenv("FLEXTOOL_NO_UPDATE_CHECK", "1")
    assert not ms.update_check_enabled(SimpleNamespace(check_updates_on_startup=True))
    monkeypatch.delenv("FLEXTOOL_NO_UPDATE_CHECK")
    assert ms.update_check_enabled(SimpleNamespace(check_updates_on_startup=True))
    assert not ms.update_check_enabled(SimpleNamespace(check_updates_on_startup=False))


def test_update_check_answer_stores_or_removes_the_offered_version(tmp_path):
    from flextool.gui.main_window import MainWindow

    path = tmp_path / "s.json"
    recomputed: list[str] = []

    class Host:
        _apply_update_indicator = MainWindow._apply_update_indicator
        _store_offered_version = MainWindow._store_offered_version

        def __init__(self):
            self._messages = SimpleNamespace(center=None, store_path=lambda: path)

        def _messages_recompute(self, trigger="other"):
            recomputed.append(trigger)

    host = Host()
    host._apply_update_indicator(True, "4.6.4", True)        # a conclusive newer version: stored
    assert MessageStore(path).offered_version() == "4.6.4" and recomputed == ["other"]
    host._apply_update_indicator(False, None, False)         # offline: kept
    assert MessageStore(path).offered_version() == "4.6.4"
    host._apply_update_indicator(False, "4.6.4", True)       # a conclusive "nothing newer": removed
    assert MessageStore(path).offered_version() is None


def test_install_info_version_helpers(monkeypatch):
    from flextool.update_flextool import install_info

    monkeypatch.setattr(install_info, "flextool_version", lambda: "4.6.3")
    assert install_info.is_newer_than_installed("4.6.4")
    assert not install_info.is_newer_than_installed("4.6.3")
    assert not install_info.is_newer_than_installed("not a version")

    def offline(*_a, **_k):
        raise OSError("offline")

    monkeypatch.setattr(install_info, "urlopen", offline)
    assert install_info.latest_pypi_version(timeout=0.1) is None


def test_project_rename_keeps_its_messages(tmp_path, monkeypatch):
    from flextool import user_dirs
    from flextool.gui import project_utils

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(user_dirs, "user_config_dir", lambda: tmp_path / "config")
    db = tmp_path / "projects" / "Alpha" / "input_sources" / "a.sqlite"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    store_path = tmp_path / "config" / m.STORE_NAME
    store = MessageStore(store_path)
    store.post("database_upgraded", database_key(db), "a.sqlite", run="r1")
    store.post("solver_defaults_updated", project_key(tmp_path / "projects" / "Alpha"), "Project Alpha", run="r1")
    project_utils.rename_project("Alpha", "Beta")
    after = MessageStore(store_path)
    assert set(after.pending("database_upgraded")) == {
        database_key(tmp_path / "projects" / "Beta" / "input_sources" / "a.sqlite")}
    assert set(after.pending("solver_defaults_updated")) == {project_key(tmp_path / "projects" / "Beta")}


def _set_version(path: Path, version: int) -> None:
    """As ``tests/update_flextool/test_frozen_guard.py::_set_version``: the ``model.version`` default."""
    from spinedb_api import DatabaseMapping, import_data

    db = DatabaseMapping(f"sqlite:///{path}", upgrade=False)
    try:
        with db:
            import_data(db, object_parameters=[["model", "version", version, None, "test"]])
            db.commit_session("test: set version")
    finally:
        db.close()


def test_two_databases_one_version_back_give_two_upgraded_messages(schema_db_url, tmp_path):
    """M3a acceptance: two schema databases set one data version back, migrated headless by the worker's per-file
    step, give two ``database_upgraded`` messages from that version to the current one."""
    import shutil

    from flextool.gui.main_window import upgrade_one
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    old = FLEXTOOL_DB_VERSION - 1
    folder = tmp_path / "P" / "input_sources"
    folder.mkdir(parents=True)
    paths = []
    for name in ("one.sqlite", "two.sqlite"):
        path = folder / name
        shutil.copy2(schema_db_url.removeprefix("sqlite:///"), path)
        _set_version(path, old)
        paths.append(path)
    results = [upgrade_one(p.name, p, "plain", None, progress_callback=None, cancel_check=None) for p in paths]
    assert [(r.before, r.after, r.upgraded, r.failed) for r in results] == [
        (old, FLEXTOOL_DB_VERSION, True, False)] * 2
    center = _center(tmp_path / "s.json", ms.upgraded_source(lambda: [database_key(p) for p in paths]))
    ms.post_upgrades(center, results, "run1")
    center.recompute("project_open")
    assert sorted(msg.subject_label for msg in center.messages) == ["one.sqlite", "two.sqlite"]
    for msg in center.messages:
        assert f"from data version {old} to {FLEXTOOL_DB_VERSION}" in msg.paragraphs[0]
