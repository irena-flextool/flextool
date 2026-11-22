"""The value-list repair tool's snapshot and analysis (W-0490 package P1): states, lost member, position proof.

Every database is a made-up copy of the schema database (``_value_list_cases.make_hit``).  Expected states, rows
and commits come from the construction: R is the schema's ``not_allowed`` row (index 0, siblings at indexes 1-12
with consecutive ids and the schema's commit), C the session's value-list commit, ``end_commit`` the →22 version
commit.  Case names follow the design: "defaults-only hit" (the shipped first member lost, only defaults refer to
it), "legacy-member hit" (the overwritten row held a member FlexTool never shipped).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest

from flextool._resources import package_data_path
from flextool.update_flextool import db_migration
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr
from tests.update_flextool._value_list_cases import (
    LEGACY, VERSION_PREFIX, add_commit, make_hit, member_id, sql)

PRIOR = ("not_allowed", "invest_no_limit", "invest_period", "invest_total", "invest_period_total",
         "retire_no_limit", "retire_period", "retire_total", "retire_period_total", "invest_retire_no_limit",
         "invest_retire_period", "invest_retire_total", "invest_retire_period_total")
AFTER_SESSION = "2025-01-10"


def _analyse(path: Path, tmp_path: Path) -> vr.Analysis:
    folder = tmp_path / "snap"
    folder.mkdir(exist_ok=True)
    return vr.analyse(vr.snapshot(path, folder))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_defaults_only_hit(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url)
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.lost, a.lost_row, a.restore, a.position_proof) == (
        vr.HIT, "not_allowed", None, ("not_allowed",), True)
    assert (a.r.id, a.r.index, a.r.value, a.r.commit, a.c, a.h) == (
        case.r_id, 0, "cumulative_limits", case.c, case.c, None)
    assert (a.finding.start_commit, a.finding.end_commit, a.finding.list_commit) == (
        case.start_commit, case.end_commit, case.c)
    assert a.list_id == case.list_id and len(a.members) == 13 and a.pre_pass.repaired == 0


def test_not_hit(tmp_path, schema_db_url) -> None:
    """The append worked: ``cumulative_limits`` is a new row with commit C, every shipped member keeps its row."""
    a = _analyse(make_hit(tmp_path, schema_db_url, overwrite=False).path, tmp_path)
    assert (a.state, a.lost, a.restore) == (vr.NOT_HIT, None, ())


def test_legacy_member_hit(tmp_path, schema_db_url) -> None:
    """Every shipped member is present (``not_allowed`` at index 14, commit NULL): the lost member was legacy."""
    a = _analyse(make_hit(tmp_path, schema_db_url, legacy=True).path, tmp_path)
    assert (a.state, a.lost, a.restore, a.position_proof) == (vr.HIT, None, (), False)


def test_deleted_shipped_member_and_legacy_overwrite_is_not_proven(tmp_path, schema_db_url) -> None:
    """The one missing shipped member (``invest_total``, deleted earlier) is not at R's template position: the
    proof fails, so nothing can be certain for it."""
    case = make_hit(tmp_path, schema_db_url, legacy=True)
    sql(case.path, "DELETE FROM list_value WHERE id = ?", (member_id(case.path, "invest_methods", "invest_total"),))
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.lost, a.restore, a.position_proof) == (vr.HIT, "invest_total", ("invest_total",), False)


@pytest.mark.parametrize("change", [
    pytest.param("UPDATE list_value SET commit_id = 1 WHERE id = ?", id="mixed_sibling_commits"),
    pytest.param("UPDATE list_value SET id = 9000 WHERE id = ?", id="non_consecutive_id"),
])
def test_position_proof_fails(tmp_path, schema_db_url, change) -> None:
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, change, (case.r_id + 5,))
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.lost, a.position_proof) == (vr.HIT, "not_allowed", False)


def test_null_sibling_commits_are_older_than_c(tmp_path, schema_db_url) -> None:
    """Siblings from before commit tracking (NULL commit) count as older than C and share one commit."""
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, "UPDATE list_value SET commit_id = NULL WHERE parameter_value_list_id = ? AND id != ?",
        (case.list_id, case.r_id))
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.position_proof) == (vr.HIT, True)


def test_hit_unanchored_when_r_was_rewritten_later(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url)
    later = add_commit(case.path, "Updated", AFTER_SESSION)
    sql(case.path, "UPDATE list_value SET commit_id = ? WHERE id = ?", (later, case.r_id))
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.lost, a.c, a.r.commit) == (vr.HIT_UNANCHORED, "not_allowed", case.c, later)


def test_hit_unanchored_after_a_retried_migration(tmp_path, schema_db_url) -> None:
    """D-1: the migration failed after its value-list commit (in the window, which dates the finding) and was
    retried after the window.  R keeps the failed attempt's commit C - 1, whose next commit is the retry's
    value-list commit, not ``Added cumulative investments``: no anchor, so never certain."""
    case = make_hit(tmp_path, schema_db_url, retried=True)
    a = _analyse(case.path, tmp_path)
    assert a.finding is not None and a.finding.date == "2024-09-19"
    assert (a.state, a.lost, a.r.commit, a.c) == (vr.HIT_UNANCHORED, "not_allowed", case.c - 1, case.c)


def test_hand_restored_member_is_the_lost_one(tmp_path, schema_db_url) -> None:
    """The checklist's hand fix: ``not_allowed`` added again after the session, above every sibling."""
    case = make_hit(tmp_path, schema_db_url)
    later = add_commit(case.path, "Updated", AFTER_SESSION)
    sql(case.path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type, commit_id) '
                   "VALUES (?, 13, ?, 'str', ?)", (case.list_id, b'"not_allowed"', later))
    restored = member_id(case.path, "invest_methods", "not_allowed")
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.lost, a.lost_row, a.restore, a.position_proof) == (
        vr.HIT, "not_allowed", restored, (), True)


@pytest.mark.parametrize("re_added", [False, True], ids=["cumulative_limits_missing", "cumulative_limits_re_added"])
def test_renamed_back(tmp_path, schema_db_url, re_added) -> None:
    """R was renamed back to ``not_allowed`` at H, after the session; no row is anchored to C any more, so C comes
    from the commit table.  With ``cumulative_limits`` re-added by hand there is nothing to restore."""
    case = make_hit(tmp_path, schema_db_url)
    h = add_commit(case.path, "Updated", AFTER_SESSION)
    sql(case.path, "UPDATE list_value SET value = ?, commit_id = ? WHERE id = ?", (b'"not_allowed"', h, case.r_id))
    if re_added:
        sql(case.path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type, commit_id) '
                       "VALUES (?, 13, ?, 'str', ?)", (case.list_id, b'"cumulative_limits"', h))
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.c, a.h, a.lost, a.position_proof) == (vr.RENAMED_BACK, case.c, h, "not_allowed", True)
    assert a.restore == (() if re_added else ("cumulative_limits",))


def test_unclear(tmp_path, schema_db_url) -> None:
    """R reads ``cumulative_limits`` but its commit is after the session and no shipped member is missing."""
    case = make_hit(tmp_path, schema_db_url, legacy=True)
    later = add_commit(case.path, "Updated", AFTER_SESSION)
    sql(case.path, "UPDATE list_value SET commit_id = ? WHERE id = ?", (later, case.r_id))
    a = _analyse(case.path, tmp_path)
    assert a.state == vr.UNCLEAR and a.reason and a.members


def test_pre_v77_group_shape(tmp_path, schema_db_url) -> None:
    """Before v77 the group ``invest_method`` is bound to ``invest_methods`` and defaults to R."""
    case = make_hit(tmp_path, schema_db_url, pre_v77_group=True)
    assert sql(case.path, "SELECT CAST(pd.default_value AS TEXT) FROM parameter_definition pd JOIN entity_class ec "
                          "ON pd.entity_class_id = ec.id WHERE ec.name = 'group' AND pd.name = 'invest_method'") == [
        (str(case.r_id),)]
    a = _analyse(case.path, tmp_path)
    assert (a.state, a.lost, a.position_proof) == (vr.HIT, "not_allowed", True)


def test_values_from_before_and_after_keep_the_state(tmp_path, schema_db_url) -> None:
    """Values written with spinedb_api before the session (``not_allowed`` → R) and after it (``cumulative_limits``
    → R) do not change the list analysis."""
    case = make_hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")],
                    after=[("unit", "U2", "base", "cumulative_limits")])
    refs = sql(case.path, "SELECT CAST(value AS TEXT) FROM parameter_value WHERE type = 'list_value_ref'")
    assert refs == [(str(case.r_id),)] * 2
    assert _analyse(case.path, tmp_path).state == vr.HIT


def test_legacy_before_values_name_their_own_rows(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url, legacy=True, before=[("unit", "U1", "base", LEGACY),
                                                                  ("unit", "U2", "base", "not_allowed")])
    refs = dict(sql(case.path, "SELECT e.name, CAST(pv.value AS TEXT) FROM parameter_value pv JOIN entity e "
                               "ON pv.entity_id = e.id"))
    assert refs == {"U1": str(case.r_id), "U2": str(member_id(case.path, "invest_methods", "not_allowed"))}


def test_no_history_and_no_finding(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, 'DELETE FROM "commit" WHERE comment LIKE ?', (VERSION_PREFIX + "%",))
    assert _analyse(case.path, tmp_path).state == vr.NO_HISTORY
    other = make_hit(tmp_path, schema_db_url, name="late.sqlite")
    sql(other.path, 'UPDATE "commit" SET date = ? WHERE id > ?', ("2025-08-01 12:00:00", other.start_commit))
    assert _analyse(other.path, tmp_path).state == vr.NO_FINDING


def test_newer_data_version_is_refused(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, "UPDATE parameter_definition SET default_value = ?, default_type = 'float' WHERE name = 'version' "
                   "AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'model')", (b"999.0",))
    with pytest.raises(vr.RepairError, match="update FlexTool"):
        _analyse(case.path, tmp_path)


def test_a_finding_for_another_step_is_refused(tmp_path, schema_db_url, monkeypatch) -> None:
    case = make_hit(tmp_path, schema_db_url)
    monkeypatch.setattr(vlh, "possible_value_list_overwrite_ro",
                        lambda path: [vlh.Finding(27, ("minimum_time_methods",), 26, 27, "2024-09-19")])
    with pytest.raises(vr.RepairError, match="step 27"):
        _analyse(case.path, tmp_path)


def test_source_is_never_written(tmp_path, schema_db_url) -> None:
    """The 279 pre-pass repairs a text-form default on the snapshot only; the source stays byte-identical."""
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, "UPDATE parameter_definition SET default_value = ? WHERE name = 'invest_method' AND "
                   "entity_class_id = (SELECT id FROM entity_class WHERE name = 'unit')", (b'"cumulative_limits"',))
    before = _sha(case.path)
    folder = tmp_path / "snap"
    folder.mkdir()
    snap = vr.snapshot(case.path, folder)
    assert snap.name == f"hit.sqlite.partial-{__import__('os').getpid()}"
    a = vr.analyse(snap)
    assert a.pre_pass.repaired == 1 and a.state == vr.HIT
    assert _sha(case.path) == before
    query = ("SELECT CAST(pd.default_value AS TEXT) FROM parameter_definition pd JOIN entity_class ec ON "
             "pd.entity_class_id = ec.id WHERE ec.name = 'unit' AND pd.name = 'invest_method'")
    assert sql(snap, query) == [(str(case.r_id),)]
    snap.unlink()
    case.path.unlink()


def test_snapshot_of_a_locked_file_says_close_the_gui(tmp_path, schema_db_url, monkeypatch) -> None:
    import contextlib
    import sqlite3

    case = make_hit(tmp_path, schema_db_url)
    monkeypatch.setattr(vr, "_BUSY_TIMEOUT_S", 0.1)
    with contextlib.closing(sqlite3.connect(case.path, isolation_level=None)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        with pytest.raises(vr.RepairError, match="close Spine Toolbox"):
            vr.snapshot(case.path, tmp_path)
        lock.execute("ROLLBACK")
    assert not list(tmp_path.glob("*.partial-*"))


def test_data_tables_are_pinned() -> None:
    template = json.loads(Path(package_data_path("schemas/pre_v26/flextool_template_v2.json")).read_text(
        encoding="utf-8"))
    assert vlh.PRIOR_MEMBERS[22] == PRIOR == tuple(m for name, m in template["parameter_value_lists"]
                                                   if name == "invest_methods")
    schema = json.loads(Path(package_data_path("schemas/spinedb_schema.json")).read_text(encoding="utf-8"))
    assert set(PRIOR) <= {m for name, m in schema["parameter_value_lists"] if name == "invest_methods"}
    assert set(vlh.PRIOR_MEMBERS) == set(vlh.APPENDED) == set(vlh.STEP_COMMIT) == {22}
    assert vlh.APPEND[22] == ("invest_methods",)
    block = re.split(r"elif next_version == (\d+):", inspect.getsource(db_migration._migrate_open))
    step22 = block[block.index("22") + 1]
    assert re.findall(r'_commit_step\(db,\s*"([^"]+)"\)', step22) == [vlh.STEP_COMMIT[22]]
    assert f'["invest_methods","{vlh.APPENDED[22]}"]' in step22.replace(" ", "")
