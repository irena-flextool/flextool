"""``flextool-repair-value-list --checked-by-hand``: the record for a file the tool cannot read after the checklist.

A file stays ``unclear`` (exit 4) after the W-0490 checklist when nothing in its list tells the tool what happened.
With ``--checked-by-hand`` the user confirms the checklist; the tool changes no value for W-0490 and writes only the
part A record into a repaired copy, so W-0490 ends for that copy.

The ``unclear`` file (:func:`_unclear`, the P7 follow-up's case U3a) is ``make_hit(legacy=True)`` with R's commit
moved after the session: R reads ``cumulative_limits`` but its commit is no anchor, every shipped member is
present (``not_allowed`` is the legacy layout's row at index 14, commit NULL, so it is not a hand-restored row), so
no lost member is known → ``unclear``.  The three ``invest_method`` class defaults already read ``not_allowed``
(``make_hit`` points them to that row), and no value refers to R, so the checklist's steps 2-5 change nothing: this
is the file after the checklist.  The schema database has every shipped default, so part B asks nothing.

Oracles (from the request and the record format, never from the tool's output):

- the record is ``{"v":1,"step":22,"end_commit":<the session's →22 commit>,"end_date":"2024-09-19","set":0,
  "kept":0,"checked_by_hand":true}`` (no part A item, so nothing set or kept);
- the copy's values, defaults and list rows equal the original's; the original is byte-identical (sha256) and
  keeps W-0490; a frozen original gives a copy that is not frozen;
- W-0490 ends on the copy and comes back when a shipped member is deleted from the copy after the record (the
  record's list check), and the record does not end another session's finding (another ``end_commit``);
- each refusal leaves the folder as it was.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from flextool.cli import cmd_repair_value_list as cli
from flextool.update_flextool import frozen as fz
from flextool.update_flextool import value_list_history as vlh
from tests.update_flextool._value_list_cases import (HIT_DATE, LIST_STEP, STEP_22, VERSION_PREFIX, add_commit,
                                                     make_hit, member_id, sql)

AFTER_SESSION = "2025-01-10"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unclear(tmp_path: Path, schema_db_url):
    case = make_hit(tmp_path, schema_db_url, legacy=True)
    sql(case.path, "UPDATE list_value SET commit_id = ? WHERE id = ?",
        (add_commit(case.path, "Updated", AFTER_SESSION), case.r_id))
    return case


def _finding(case) -> vlh.Finding:
    return vlh.Finding(22, ("invest_methods",), 21, 22, HIT_DATE, end_commit=case.end_commit)


def _rows(path: Path) -> tuple:
    with contextlib.closing(sqlite3.connect(fz.ro_uri(path), uri=True)) as con:
        return tuple(con.execute(q).fetchall() for q in (
            "SELECT id, value, type FROM parameter_value ORDER BY id",
            "SELECT id, default_value, default_type FROM parameter_definition ORDER BY id",
            'SELECT id, parameter_value_list_id, "index", value FROM list_value ORDER BY id'))


class _Ask:
    def __init__(self, *replies: str) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise EOFError
        return self.replies.pop(0)


@pytest.mark.parametrize("interactive", [False, True])
def test_checked_by_hand_records_the_repair_in_a_copy(tmp_path, schema_db_url, interactive) -> None:
    case = _unclear(tmp_path, schema_db_url)
    fz.freeze(case.path, "reference 1.0")
    assert cli.main([str(case.path)]) == cli.EXIT_UNCLEAR, "unclear after the checklist"
    sha, listing, before = _sha(case.path), sorted(p.name for p in tmp_path.iterdir()), _rows(case.path)
    if interactive:
        ask = _Ask("yes", "y")
        assert cli.main([str(case.path), "--checked-by-hand"], ask=ask) == cli.EXIT_OK
        assert ask.prompts[0] == cli.BY_HAND_PROMPT and len(ask.prompts) == 2, "the confirmation, then the write"
    else:
        assert cli.main([str(case.path), "--checked-by-hand", "--yes"]) == cli.EXIT_OK
    output = case.path.with_name("hit_repaired.sqlite")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        listing + [output.name, output.name + ".value_list_repair.csv", output.name + ".value_list_repair.txt"])

    assert _sha(case.path) == sha, "the original is untouched"
    assert vlh.possible_value_list_overwrite_ro(case.path) == [_finding(case)]
    assert fz.read_frozen(case.path).label == "reference 1.0" and fz.read_frozen(output) is None
    assert _rows(output) == before, "no value, default or list row changed"
    record = vlh.read_record(output)
    with contextlib.closing(sqlite3.connect(fz.ro_uri(output), uri=True)) as con:
        value = vlh.record_row(con, vlh.RECORD_KEY)[0]
    assert json.loads(value) == {
        "v": 1, "step": 22, "end_commit": case.end_commit, "end_date": HIT_DATE, "set": 0, "kept": 0,
        "checked_by_hand": True}
    assert record is not None and record.value["checked_by_hand"] is True
    assert vlh.possible_value_list_overwrite_ro(output) == []
    assert "checked by hand" in output.with_name(output.name + ".value_list_repair.txt").read_text(encoding="utf-8")

    # a rerun on the copy asks nothing and changes nothing
    copy_sha = _sha(output)
    assert cli.main([str(output), "--output", str(tmp_path / "again.sqlite")]) == cli.EXIT_OK
    assert _sha(output) == copy_sha and not (tmp_path / "again.sqlite").exists()

    # another session's finding is not ended by this record; damage after the record brings W-0490 back
    other = vlh.Finding(22, ("invest_methods",), 21, 22, HIT_DATE, end_commit=case.end_commit + 1)
    assert not vlh.record_ends(other, output)
    sql(output, "DELETE FROM list_value WHERE id = ?", (member_id(output, "invest_methods", "retire_total"),))
    assert vlh.possible_value_list_overwrite_ro(output) == [_finding(case)]


def _refused(tmp_path: Path, path: Path, argv: list[str], code: int, text: str, capsys, ask=None) -> None:
    sha, listing = _sha(path), sorted(p.name for p in tmp_path.iterdir())
    assert cli.main([str(path), *argv], ask=ask) == code
    out = capsys.readouterr()
    assert text in out.out + out.err
    assert _sha(path) == sha and sorted(p.name for p in tmp_path.iterdir()) == listing


def test_refused_on_a_file_the_tool_can_read(tmp_path, schema_db_url, capsys) -> None:
    """An anchored hit is ``hit``, not ``unclear``: the normal run repairs it."""
    case = make_hit(tmp_path, schema_db_url)
    _refused(tmp_path, case.path, ["--checked-by-hand", "--yes"], cli.EXIT_ERROR,
             "run the tool without --checked-by-hand", capsys)


def test_refused_with_several_sessions(tmp_path, schema_db_url, capsys) -> None:
    """A second 21 → 22 session in the window (2025-01-05) gives two findings; one record covers one session."""
    case = make_hit(tmp_path, schema_db_url)
    for comment in (VERSION_PREFIX + "21", LIST_STEP, STEP_22, VERSION_PREFIX + "22"):
        add_commit(case.path, comment, "2025-01-05")
    assert len(vlh.session_findings_ro(case.path)) == 2
    _refused(tmp_path, case.path, ["--checked-by-hand", "--yes"], cli.EXIT_ERROR, "several W-0490 sessions", capsys)


def test_refused_while_the_list_lacks_a_member(tmp_path, schema_db_url, capsys) -> None:
    """The bug did not fire but ``cumulative_limits`` was deleted later (unclear): steps 2-3 are not done yet."""
    case = make_hit(tmp_path, schema_db_url, overwrite=False)
    sql(case.path, "DELETE FROM list_value WHERE id = ?", (member_id(case.path, "invest_methods",
                                                                     "cumulative_limits"),))
    _refused(tmp_path, case.path, ["--checked-by-hand", "--yes"], cli.EXIT_ERROR, "still lacks cumulative_limits",
             capsys)


def test_refused_without_a_confirmation(tmp_path, schema_db_url, capsys) -> None:
    """No terminal and no ``--yes`` → 3; at the prompt anything but ``yes`` (here ``y``, and EOF) → 1."""
    case = _unclear(tmp_path, schema_db_url)
    _refused(tmp_path, case.path, ["--checked-by-hand"], cli.EXIT_NEEDS_ANSWERS, "add --yes", capsys)
    _refused(tmp_path, case.path, ["--checked-by-hand"], cli.EXIT_ERROR, "Not confirmed", capsys, ask=_Ask("y"))
    _refused(tmp_path, case.path, ["--checked-by-hand"], cli.EXIT_ERROR, "Not confirmed", capsys, ask=_Ask())


@pytest.mark.parametrize("argv", [["--yes"], ["--checked-by-hand", "--archive", "old.sqlite"]])
def test_usage_errors(tmp_path, argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path / "x.sqlite"), *argv])
    assert exc.value.code == 2
