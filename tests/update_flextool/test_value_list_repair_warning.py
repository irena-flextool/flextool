"""The repair record ends W-0490 and the GUI's "Please check this database" box (package P7).

Every database is a made-up copy of the schema database (``_value_list_cases.make_hit``: a defaults-only hit, the
unit / connection / node ``invest_method`` defaults read ``cumulative_limits``, ``not_allowed`` is missing from
``invest_methods``).  The repaired copy answers the three default suspects ``set:not_allowed``.

Oracles (from the construction, never from the tool's output):

- the original hit file gives one finding (step 22, 21 → 22, 2024-09-19) before and after the repair;
- the repaired copy gives none, because it holds a valid record (P5's rules: same step / ``end_commit`` /
  ``end_date``, written by a later commit with the comment ``REPAIR_COMMIT``, and the list complete);
- a hand-built record (the JSON written out below) ends a finding only when every rule holds: each negative case
  has a positive control that differs in that one rule.
"""
from __future__ import annotations

import shlex
import shutil
from pathlib import Path

import pytest

from flextool import run_warnings
from flextool.cli import cmd_repair_value_list as cli
from flextool.run_warnings.registry import WARNINGS
from flextool.update_flextool import value_list_answers as va
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_write as vw
from tests._support.coded_warnings import capture_coded
from tests.update_flextool._value_list_cases import HIT_DATE, add_commit, make_hit, sql

DOCS = Path(__file__).resolve().parents[2] / "docs" / "terminal_workflow.md"
HIT = vlh.Finding(22, ("invest_methods",), 21, 22, HIT_DATE)


def _repaired(tmp_path: Path, schema_db_url: str):
    """``(hit case, repaired copy)``: every suspect (the three class defaults) answered ``set:not_allowed``."""
    case = make_hit(tmp_path, schema_db_url)
    output = tmp_path / "out" / "hit_repaired.sqlite"
    output.parent.mkdir()
    a = vr.analyse(vr.snapshot(case.path, output.parent, name=output.name))
    found = vr.all_suspects(a)
    assert sorted(s.cls for s in found) == ["connection", "node", "unit"]
    plan = va.make_plan(a, found, {s.item: va.Answer(va.SET, "not_allowed") for s in found})
    w = vw.write_copy(a, plan, output, source=case.path)
    assert w.part_a_record
    return case, output


def _restore_member(path: Path, list_id: int) -> None:
    """Put ``not_allowed`` back as the 14th member (index 13), so the list check of the record holds."""
    sql(path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type) VALUES (?, 13, ?, \'str\')',
        (list_id, b'"not_allowed"'))


def _add_record(path: Path, end_commit: int, comment: str) -> int:
    """A record row written by a new commit *comment*, as an import would add it; returns that commit's id."""
    commit = add_commit(path, comment, "2026-11-23 12:00:00")
    value = f'{{"v":1,"step":22,"end_commit":{end_commit},"end_date":"{HIT_DATE}","set":3,"kept":0}}'
    sql(path, "INSERT INTO metadata (name, value, commit_id) VALUES (?, ?, ?)", (vlh.RECORD_KEY, value, commit))
    return commit


def test_repaired_copy_ends_the_warning_and_the_original_keeps_it(tmp_path, schema_db_url) -> None:
    case, output = _repaired(tmp_path, schema_db_url)
    assert vlh.possible_value_list_overwrite_ro(output) == []
    assert vlh.possible_value_list_overwrite_ro(case.path) == [HIT]
    assert vlh.session_findings_ro(output) == [HIT], "the tool still sees the session it repaired"
    for path, expected in ((output, []), (case.path, ["W-0490"])):
        run_warnings.start_run()
        with capture_coded() as got:
            vlh.warn_at_run_start(f"sqlite:///{path}")
        assert [c.code for c in got.calls] == expected, path.name


def test_deleting_the_restored_member_brings_the_warning_back(tmp_path, schema_db_url) -> None:
    _, output = _repaired(tmp_path, schema_db_url)
    sql(output, "DELETE FROM list_value WHERE value = ? AND parameter_value_list_id = (SELECT id FROM "
                "parameter_value_list WHERE name = 'invest_methods')", (b'"not_allowed"',))
    assert vlh.possible_value_list_overwrite_ro(output) == [HIT]


def test_a_corrupt_record_keeps_the_warning(tmp_path, schema_db_url) -> None:
    _, output = _repaired(tmp_path, schema_db_url)
    sql(output, "UPDATE metadata SET value = '{\"v\":1,' WHERE name = ?", (vlh.RECORD_KEY,))
    assert vlh.possible_value_list_overwrite_ro(output) == [HIT]


def test_an_error_while_reading_the_record_keeps_the_warning(tmp_path, schema_db_url, monkeypatch) -> None:
    _, output = _repaired(tmp_path, schema_db_url)

    def boom(finding, path):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(vlh, "record_ends", boom)
    assert vlh.possible_value_list_overwrite_ro(output) == [HIT]


def test_a_record_for_another_session_does_not_end_the_warning(tmp_path, schema_db_url) -> None:
    """A record copied in from another hit file names that file's ``end_commit``: it ends nothing here.  Control:
    the same record with this file's own ``end_commit`` ends the finding."""
    other = make_hit(tmp_path, schema_db_url, name="other.sqlite", before=[("unit", "U1", "base", "not_allowed")])
    first = make_hit(tmp_path, schema_db_url, name="first.sqlite")
    assert first.end_commit != other.end_commit, "the 'before' value's commits shift the ids"
    _restore_member(other.path, other.list_id)
    copy = tmp_path / "control.sqlite"
    shutil.copy2(other.path, copy)
    _add_record(other.path, first.end_commit, vlh.REPAIR_COMMIT)
    (finding,) = vlh.possible_value_list_overwrite_ro(other.path)
    assert finding.end_commit == other.end_commit
    _add_record(copy, other.end_commit, vlh.REPAIR_COMMIT)
    assert vlh.possible_value_list_overwrite_ro(copy) == []


def test_a_record_imported_into_a_sibling_copy_does_not_end_the_warning(tmp_path, schema_db_url) -> None:
    """Sibling copies of one database share their commit ids, so a record imported from a repaired sibling (commit
    "Import 3 items", as Spine Toolbox writes it) matches the finding in everything but the commit's comment: the
    comment is the guard.  Control: the same row with the tool's comment ends the finding."""
    case = make_hit(tmp_path, schema_db_url)
    _restore_member(case.path, case.list_id)
    commit = _add_record(case.path, case.end_commit, "Import 3 items")
    assert vlh.possible_value_list_overwrite_ro(case.path) == [HIT]
    sql(case.path, 'UPDATE "commit" SET comment = ? WHERE id = ?', (vlh.REPAIR_COMMIT, commit))
    assert vlh.possible_value_list_overwrite_ro(case.path) == []


def test_gui_check_flags_the_original_only(tmp_path, schema_db_url) -> None:
    """The 4.6.2 GUI box reads the same filtered findings: a project with the original and its repaired copy flags
    the original only."""
    from flextool.gui.data_models import ProjectSettings
    from flextool.gui.input_sources import InputSourceManager

    case, output = _repaired(tmp_path, schema_db_url)
    sources = tmp_path / "project" / "input_sources"
    sources.mkdir(parents=True)
    shutil.copy2(case.path, sources / "model.sqlite")
    shutil.copy2(output, sources / "model_repaired.sqlite")
    mgr = InputSourceManager(tmp_path / "project", ProjectSettings())
    mgr.plan_db_migrations()
    assert [(name, f) for name, f in mgr.take_check_findings()] == [("model.sqlite", HIT)]


@pytest.mark.parametrize("text", [pytest.param(lambda: WARNINGS["W-0490"].body, id="w0490_body"),
                                  pytest.param(lambda: DOCS.read_text(encoding="utf-8"), id="terminal_workflow")])
def test_texts_give_the_command_that_repair_command_builds(tmp_path, text) -> None:
    db = tmp_path / "model.sqlite"
    command = cli.repair_command(db, python="python", shell="posix")
    module_form = command.removesuffix(" " + shlex.quote(str(db.resolve())))
    assert module_form == "python -P -m flextool.cli.cmd_repair_value_list"
    body = text()
    assert f'{module_form} "' in body and f'{cli.PROG} "' in body
