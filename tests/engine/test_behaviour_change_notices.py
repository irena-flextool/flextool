"""Behaviour-change notices: printed only with the database's marker, each entity once per run, naming the
version (or the not-yet-published release) and how to silence it."""
from __future__ import annotations

import dataclasses

import pytest

from flextool.engine import _behaviour_changes as bc

CHANGE = "storage_default_fix_nothing"


@pytest.fixture(autouse=True)
def _clean_run():
    yield
    bc.start_run(None)


def test_no_marker_no_notice(capsys) -> None:
    bc.start_run({})
    assert not bc.notice(CHANGE, "storage nodes", ["a"])
    assert capsys.readouterr().out == ""


def test_each_entity_once_per_run(capsys) -> None:
    bc.start_run({CHANGE: "77"})
    assert bc.notice(CHANGE, "storage nodes", ["a", "b"])
    assert not bc.notice(CHANGE, "storage nodes", ["b", "a"])  # a later roll of the same run
    assert bc.notice(CHANGE, "storage nodes", ["b", "c"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "Affected storage nodes: a, b." in lines[0] and "Affected storage nodes: c." in lines[1]
    bc.start_run({CHANGE: "77"})  # a new run announces again
    assert bc.notice(CHANGE, "storage nodes", ["a"])


def test_text_names_version_and_silence(monkeypatch) -> None:
    change = bc.BEHAVIOUR_CHANGES[CHANGE]
    monkeypatch.setattr(bc, "_installed_version", lambda: "4.1.1")
    text = bc.notice_text(CHANGE, "storage nodes", ["a"])
    assert text.startswith("Behaviour changed in the FlexTool release after 4.1.1 (not yet published): ")
    assert text.endswith(change.silence)
    released = dataclasses.replace(change, since="4.2.0", published="1.12.2026")
    monkeypatch.setitem(bc.BEHAVIOUR_CHANGES, CHANGE, released)
    assert bc.notice_text(CHANGE, "storage nodes", ["a"]).startswith(
        "Behaviour changed in FlexTool 4.2.0 (published 1.12.2026): ")


def test_marker_json() -> None:
    bc.start_run({"storage_fix_method_rewritten": '{"since": 77, "rewrites": [["n", "A", "fix_start", "x"]]}',
                  CHANGE: "77"})
    assert bc.marker_json("storage_fix_method_rewritten")["rewrites"] == [["n", "A", "fix_start", "x"]]
    assert bc.marker_json(CHANGE) == {}
