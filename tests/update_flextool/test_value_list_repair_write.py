"""The value-list repair tool's write: apply, verify, publish, records (package P5).

Every database is a made-up copy of the schema database (``_value_list_cases``).  Oracles from the construction
(written by hand, never from the tool's output):

- ``make_hit``: R (the schema's ``not_allowed`` row, index 0) reads ``cumulative_limits``; the shipped
  ``cumulative_limits`` row is gone, so the list has 13 rows and the write adds ``not_allowed`` as a 14th row
  (``import_data`` appends; R keeps its id and its text).  The unit, connection and node ``invest_method``
  defaults point to R (three default suspects); the group default is ``not_active`` in ``group_invest_methods``
  (shipped, no suspect).  "before" values (U1) are certain, "after" values (U2, U3) have no suggestion.  An entity
  with no own value (U4) reads the class default.
- ``make_lost_defaults`` (part B): the named defaults are ``NULL``; storage nodes A (start before the upgrade) and
  B (start after it), no method, scenario s1 = [base] → two storage suspects (the v77 rule, see
  ``test_list_default_restore``).  ``node.storage_start_end_method`` ships ``fix_nothing``,
  ``unit.conversion_method`` ``constant_efficiency``, ``solve.solve_mode`` ``single_solve``.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from flextool.update_flextool import frozen as fz
from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_answers as va
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from flextool.update_flextool import value_list_write as vw
from tests.update_flextool._value_list_cases import (add_commit, clear_defaults, make_hit, make_lost_defaults, sql,
                                                     write_items)

TODAY = "2026-11-23"
SET, KEEP, SKIP = va.Answer(va.SET, "not_allowed"), va.Answer(va.KEEP), va.Answer(va.SKIP)


def _open_files_under(folder: Path) -> list[str]:
    out = []
    for fd in Path("/proc/self/fd").iterdir():
        with contextlib.suppress(OSError):
            if (target := os.readlink(fd)).startswith(str(folder)):
                out.append(target)
    return out


def _prepare(source: Path, output: Path):
    a = vr.analyse(vr.snapshot(source, output.parent, name=output.name))
    return a, vr.all_suspects(a)


def _by(found, cls: str, kind: str = vs.DEFAULT, entity: str = ""):
    (s,) = [s for s in found if (s.cls, s.kind, s.entity) == (cls, kind, entity)]
    return s.item


def _commits(path: Path) -> tuple[int, str]:
    return sql(path, 'SELECT count(*) FROM "commit"')[0][0], sql(path, 'SELECT comment FROM "commit" ORDER BY id '
                                                                     "DESC LIMIT 1")[0][0]


def _metadata(path: Path) -> list[str]:
    return sorted(n for (n,) in sql(path, "SELECT name FROM metadata"))


def _invest_members(path: Path) -> list[tuple[int, int, bytes]]:
    return sql(path, 'SELECT lv.id, lv."index", lv.value FROM list_value lv JOIN parameter_value_list l ON l.id = '
                     "lv.parameter_value_list_id WHERE l.name = 'invest_methods' ORDER BY lv.\"index\"")


def _effective(path: Path, cls: str, param: str) -> dict[str, str]:
    from flextool.engine.input._spinedb_reader import SpineDbReader

    frame = SpineDbReader(f"sqlite:///{path}", "s1").parameter(cls, param)
    return dict(zip(frame["name"].to_list(), frame["value"].to_list()))


def _hit(tmp_path, schema_db_url, **kwargs):
    case = make_hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")],
                    after=[("unit", "U2", "base", "cumulative_limits"), ("unit", "U3", "base", "cumulative_limits")],
                    **kwargs)
    # entities active in base (the scenario filter drops the others); U4 has no own invest_method
    write_items(case.path, [("unit", u, "efficiency", "base", 0.5) for u in ("U1", "U2", "U3", "U4")],
                {"s1": ["base"]})
    return case


def test_apply_hit_record_and_rerun(tmp_path, schema_db_url) -> None:
    """Suggested answers for the defaults and U1, chosen ones for U2 (``set:invest_total``) and U3 (``keep``).
    Effective unit ``invest_method`` in s1 (hand table): U1 not_allowed, U2 invest_total, U3 cumulative_limits (kept:
    R), U4 the default not_allowed."""
    case = _hit(tmp_path, schema_db_url)
    source_sha, (n_commits, _) = fz.sha256_of(case.path), _commits(case.path)
    output = tmp_path / "out" / "hit_repaired.sqlite"
    output.parent.mkdir()
    a, found = _prepare(case.path, output)
    answers = {_by(found, c): SET for c in ("unit", "connection", "node")}
    answers |= {_by(found, "unit", vs.VALUE, "U1"): SET, _by(found, "unit", vs.VALUE, "U2"): va.Answer(
        va.SET, "invest_total"), _by(found, "unit", vs.VALUE, "U3"): KEEP}
    assert len(found) == len(answers) == 6
    plan = va.make_plan(a, found, answers)
    w = vw.write_copy(a, plan, output, source=case.path, today=TODAY)
    assert fz.sha256_of(case.path) == source_sha
    assert sorted(p.name for p in output.parent.iterdir()) == [output.name, output.name + vw.SIDE_CSV,
                                                                output.name + vw.SIDE_TXT]
    assert (w.part_a_record, w.part_b_record) == (True, False)
    members = _invest_members(output)
    assert len(members) == 14 and members[0] == (case.r_id, 0, b'"cumulative_limits"')
    assert members[-1][1:] == (13, b'"not_allowed"')
    assert _commits(output) == (n_commits + 1, vlh.REPAIR_COMMIT)
    (finding,) = vlh.possible_value_list_overwrite_ro(output)
    assert vlh.record_ends(finding, output) and not vlh.record_ends(finding, case.path)
    assert vlh.read_record(output).value == {"v": 1, "step": 22, "end_commit": case.end_commit,
                                             "end_date": "2024-09-19", "set": 5, "kept": 1}
    assert _effective(output, "unit", "invest_method") == {"U1": "not_allowed", "U2": "invest_total",
                                                           "U3": "cumulative_limits", "U4": "not_allowed"}
    assert "state: hit" in w.report.read_text(encoding="utf-8")
    # rerun on the copy: asks nothing, writes nothing
    out_sha = fz.sha256_of(output)
    again = tmp_path / "again" / "x.sqlite"
    again.parent.mkdir()
    a2, found2 = _prepare(output, again)
    prev = vw.previous_repair(a2, found2)
    assert re.fullmatch(r"repaired on \d{4}-\d\d-\d\d \(5 set, 1 kept\)", prev.part_a)
    assert prev.suspects == () and [(s.entity, s.current) for s in prev.kept] == [("U3", "cumulative_limits")]
    vw.discard(a2.path)
    assert fz.sha256_of(output) == out_sha and list(again.parent.iterdir()) == []
    if Path("/proc/self/fd").is_dir():
        assert _open_files_under(tmp_path) == []
    for path in (case.path, output, w.answers, w.report):
        path.unlink()


def test_skip_keep_not_hit_and_legacy(tmp_path, schema_db_url) -> None:
    """One skip → the other answers written, no record.  All keep → record (``not_allowed`` still restored).
    Not-hit and a legacy hit whose defaults point to ``not_allowed`` (no suspects) → the record alone."""
    case = _hit(tmp_path, schema_db_url)
    out = tmp_path / "skip.sqlite"
    a, found = _prepare(case.path, out)
    answers = {s.item: SET if s.suggestion else KEEP for s in found} | {_by(found, "node"): SKIP}
    w = vw.write_copy(a, va.make_plan(a, found, answers), out, source=case.path)
    assert not w.part_a_record and vlh.RECORD_KEY not in _metadata(out)
    assert sql(out, "SELECT count(*) FROM parameter_definition WHERE name = 'invest_method' AND default_value = ?",
               (str(case.r_id).encode(),)) == [(1,)]  # the node default still points to R
    out = tmp_path / "keep.sqlite"
    a, found = _prepare(case.path, out)
    w = vw.write_copy(a, va.make_plan(a, found, {s.item: KEEP for s in found}), out, source=case.path)
    assert w.part_a_record and len(_invest_members(out)) == 14
    assert vlh.record_ends(vlh.possible_value_list_overwrite_ro(out)[0], out)
    for name, kwargs in (("nothit", {"overwrite": False}), ("legacy", {"legacy": True})):
        folder = tmp_path / name
        folder.mkdir()
        other = make_hit(folder, schema_db_url, **kwargs)
        out = folder / "out.sqlite"
        a, found = _prepare(other.path, out)
        assert found == () and a.restore == ()
        meta, (n, _) = _metadata(other.path), _commits(other.path)
        w = vw.write_copy(a, va.make_plan(a, found, {}), out, source=other.path)
        assert w.part_a_record and _metadata(out) == sorted(meta + [vlh.RECORD_KEY])
        assert _commits(out) == (n + 1, vlh.REPAIR_COMMIT)
        assert vlh.record_ends(vlh.possible_value_list_overwrite_ro(out)[0], out)


@pytest.mark.parametrize("re_added", [False, True])
def test_renamed_back(tmp_path, schema_db_url, re_added) -> None:
    """R renamed back to ``not_allowed`` by hand at H: ``cumulative_limits`` is put back as a new row (index 13), or
    nothing is restored when it was re-added by hand; the record is written either way."""
    case = make_hit(tmp_path, schema_db_url)
    h = add_commit(case.path, "Updated", "2025-01-10")
    sql(case.path, "UPDATE list_value SET value = ?, commit_id = ? WHERE id = ?", (b'"not_allowed"', h, case.r_id))
    if re_added:
        sql(case.path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type, commit_id) '
                       "VALUES (?, 13, ?, 'str', ?)", (case.list_id, b'"cumulative_limits"', h))
    out = tmp_path / "out.sqlite"
    a, found = _prepare(case.path, out)
    assert a.state == vr.RENAMED_BACK
    before = _invest_members(case.path)
    w = vw.write_copy(a, va.make_plan(a, found, {s.item: KEEP for s in found}), out, source=case.path)
    after = _invest_members(out)
    assert after[:len(before)] == before and w.part_a_record
    assert [m[1:] for m in after[len(before):]] == ([] if re_added else [(13, b'"cumulative_limits"')])
    assert vlh.record_ends(vlh.possible_value_list_overwrite_ro(out)[0], out)


def test_frozen_source_and_old_spinedb_revision(tmp_path, schema_db_url) -> None:
    """A frozen source: the copy is not frozen, the source still is, the label is in the ``.txt``.  An old SpineDB
    revision (the parameter group migration taken out by hand): the copy is upgraded and written, the source keeps
    its revision and bytes (R7).  An unknown (newer) revision is refused with "update FlexTool"."""
    case = make_hit(tmp_path, schema_db_url)
    fz.freeze(case.path, "reference 1.0")
    out = tmp_path / "out.sqlite"
    a, found = _prepare(case.path, out)
    sha = fz.sha256_of(case.path)
    w = vw.write_copy(a, va.make_plan(a, found, {s.item: SET for s in found}), out, source=case.path)
    assert fz.read_frozen(out) is None and fz.read_frozen(case.path).label == "reference 1.0"
    assert fz.FROZEN_KEY not in _metadata(out) and fz.sha256_of(case.path) == sha
    assert "frozen: reference 1.0" in w.report.read_text(encoding="utf-8")

    (tmp_path / "old").mkdir()
    old = make_hit(tmp_path / "old", schema_db_url)
    _set_back_revision(old.path)
    assert fz.schema_outdated_ro(old.path)
    sha = fz.sha256_of(old.path)
    out = tmp_path / "old" / "out.sqlite"
    a, found = _prepare(old.path, out)
    w = vw.write_copy(a, va.make_plan(a, found, {s.item: SET for s in found}), out, source=old.path)
    assert w.part_a_record and not fz.schema_outdated_ro(out)
    assert fz.sha256_of(old.path) == sha and fz.schema_outdated_ro(old.path)
    sql(out, "UPDATE alembic_version SET version_num = 'ffffffffffff'")  # a revision of a newer Spine library
    with pytest.raises(vr.RepairError, match="update FlexTool"):
        vw.check_schema(out)


def _set_back_revision(path: Path) -> None:
    """Undo the newest spinedb_api migration (``parameter_group``) by raw SQL and set the revision back."""
    with contextlib.closing(sqlite3.connect(path, isolation_level=None)) as con:
        (ddl,) = con.execute("SELECT sql FROM sqlite_master WHERE name = 'parameter_definition'").fetchone()
        ddl = re.sub(r",\s*parameter_group_id INTEGER", "", ddl)
        ddl = re.sub(r",\s*CONSTRAINT fk_parameter_definition_parameter_group_id_parameter_group[^\n]*\)", "", ddl)
        cols = ",".join(r[1] for r in con.execute("PRAGMA table_info(parameter_definition)")
                        if r[1] != "parameter_group_id")
        con.execute("PRAGMA foreign_keys = OFF")
        con.execute("BEGIN")
        con.execute(ddl.replace("CREATE TABLE parameter_definition", "CREATE TABLE pd_old", 1))
        con.execute(f"INSERT INTO pd_old ({cols}) SELECT {cols} FROM parameter_definition")
        con.execute("DROP TABLE parameter_definition")
        con.execute("ALTER TABLE pd_old RENAME TO parameter_definition")
        con.execute("DROP TABLE parameter_group")
        con.execute("UPDATE alembic_version SET version_num = 'e9f2c2330cf8'")
        con.execute("COMMIT")


def test_refusals_and_failures_leave_nothing(tmp_path, schema_db_url, monkeypatch) -> None:
    """Output == source / an existing output → refused.  A file created at the output while the copy is made →
    refused, that file kept.  A write the plan does not hold → the verify refuses it.  That, or a
    ``KeyboardInterrupt`` in the verify → no output, no partial, no side file.
    A dry run's snapshot in a temporary folder leaves nothing next to the source."""
    case = make_hit(tmp_path, schema_db_url)
    with pytest.raises(vr.RepairError, match="new file"):
        vw.check_output(case.path, case.path)
    with pytest.raises(vr.RepairError, match="exists already"):
        vw.check_output(tmp_path / "other.sqlite", case.path)
    assert vw.default_output(case.path) == tmp_path / "hit_repaired.sqlite"
    folder = tmp_path / "w"
    folder.mkdir()
    out = folder / "out.sqlite"
    real_verify, real_apply = vw._verify, vw._apply

    def late_file(*args):
        real_verify(*args)
        out.write_bytes(b"someone else's")

    def stray_write(a, *args):  # a write the plan does not hold: the verify must refuse it
        done = real_apply(a, *args)
        sql(a.path, "UPDATE parameter_definition SET default_value = ?, default_type = 'float' WHERE name = "
                    "'efficiency' AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'unit')", (b"2.0",))
        return done

    def interrupt(*args):
        raise KeyboardInterrupt

    for name, inject, error in (("_verify", late_file, "appeared"), ("_apply", stray_write, "a definition no answer"),
                                ("_verify", interrupt, None)):
        monkeypatch.setattr(vw, name, inject)
        a, found = _prepare(case.path, out)
        with pytest.raises(KeyboardInterrupt if error is None else vr.RepairError, match=error):
            vw.write_copy(a, va.make_plan(a, found, {}), out, source=case.path)
        monkeypatch.undo()
        assert sorted(p.name for p in folder.iterdir()) == (["out.sqlite"] if inject is late_file else [])
        if inject is late_file:
            assert out.read_bytes() == b"someone else's"
            out.unlink()
    import tempfile

    names = sorted(p.name for p in tmp_path.iterdir())
    with tempfile.TemporaryDirectory(dir=tmp_path) as dry:
        a = vr.analyse(vr.snapshot(case.path, dry))
        vw.discard(a.path)
        assert list(Path(dry).iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == names


def test_record_parsers(tmp_path, schema_db_url) -> None:
    """Part A: valid → ends the finding; bad JSON, ``v`` 2, another ``end_commit``, an import's commit comment → not;
    several rows → the newest decides.  Part B: the same rules for ``parse_part_b_record``."""
    case = make_hit(tmp_path, schema_db_url, overwrite=False)
    (finding,) = vlh.possible_value_list_overwrite_ro(case.path)
    ok = vlh.record_value(finding, 0, 0)
    good_b = ldr.part_b_record_value(TODAY, 1, 2, 0, "abc")

    def rows(*items) -> None:
        sql(case.path, "DELETE FROM metadata")
        for key, value, comment in items:
            commit = add_commit(case.path, comment, "2026-11-23")
            sql(case.path, "INSERT INTO metadata (name, value, commit_id) VALUES (?, ?, ?)", (key, value, commit))

    def part_b():
        with contextlib.closing(fz._connect_ro(case.path)) as con:
            return ldr.parse_part_b_record(con)

    rows((vlh.RECORD_KEY, ok, vlh.REPAIR_COMMIT))
    assert vlh.record_ends(finding, case.path)
    other = ok.replace(f'"end_commit":{finding.end_commit}', f'"end_commit":{finding.end_commit + 1}')
    for value, comment in (("{not json", vlh.REPAIR_COMMIT), (ok.replace('"v":1', '"v":2'), vlh.REPAIR_COMMIT),
                           (other, vlh.REPAIR_COMMIT), (ok, "Import 3 items")):
        rows((vlh.RECORD_KEY, value, comment))
        assert not vlh.record_ends(finding, case.path), value
    rows((vlh.RECORD_KEY, ok, vlh.REPAIR_COMMIT), (vlh.RECORD_KEY, "{not json", vlh.REPAIR_COMMIT))
    assert not vlh.record_ends(finding, case.path)
    rows((vlh.RECORD_KEY, "{not json", vlh.REPAIR_COMMIT), (vlh.RECORD_KEY, ok, vlh.REPAIR_COMMIT))
    assert vlh.record_ends(finding, case.path)

    rows((ldr.PART_B_RECORD_KEY, good_b, vlh.REPAIR_COMMIT))
    rec = part_b()
    assert (rec.date, rec.set, rec.kept, rec.storage, rec.empty) == (TODAY, 1, 2, 0, "abc")
    for value, comment in (("[1]", vlh.REPAIR_COMMIT), (good_b.replace('"v":1', '"v":2'), vlh.REPAIR_COMMIT),
                           (good_b, "Import 3 items")):
        rows((ldr.PART_B_RECORD_KEY, value, comment))
        assert part_b() is None, value
    rows((ldr.PART_B_RECORD_KEY, good_b, vlh.REPAIR_COMMIT), (ldr.PART_B_RECORD_KEY, "[1]", vlh.REPAIR_COMMIT))
    assert part_b() is None


STORAGE = "node.storage_start_end_method"
LOST = ["unit.conversion_method", "solve.solve_mode", STORAGE]


def _lost(tmp_path, schema_db_url):
    """Three bug-reachable defaults and one reference default the bug could not reach (the user keeps it empty)."""
    (extra,) = sorted(f"{c}.{p}" for c, p in ldr.reference().keys() - ldr.bug_reachable())[:1]
    case = make_lost_defaults(tmp_path, LOST + [extra], schema_db=schema_db_url,
                              before=[("node", "A", "storage_state_start", "base", 0.5)],
                              after=[("node", "B", "storage_state_start", "base", 0.3)], scenarios={"s1": ["base"]})
    return case, extra


def _part_b_answers(found, keep: tuple[str, ...], storage_b: va.Answer | None):
    answers = {}
    for s in found:
        if s.kind == ldr.STORAGE_KIND:
            ans = va.Answer(va.SET, "fix_start") if s.entity == "A" else storage_b
        else:
            ans = KEEP if f"{s.cls}.{s.parameter}" in keep else va.Answer(va.SET, s.suggestion[4:])
        if ans is not None:
            answers[s.item] = ans
    return answers


def test_part_b_restore_storage_and_rerun(tmp_path, schema_db_url) -> None:
    """Part B only.  Three defaults restored (set 3), the extra one kept, node A ``fix_start`` (storage 1), node B
    kept (kept 2).  Hash over the full empty set after the write = the extra name alone (R1)."""
    case, extra = _lost(tmp_path, schema_db_url)
    meta, (n, _) = _metadata(case.path), _commits(case.path)
    out = tmp_path / "out.sqlite"
    a, found = _prepare(case.path, out)
    assert sorted(s.entity for s in found if s.kind == ldr.STORAGE_KIND) == ["A", "B"]
    plan = va.make_plan(a, found, _part_b_answers(found, (extra,), KEEP))
    assert va.file_key(a).startswith("k0-00000000-")
    w = vw.write_copy(a, plan, out, source=case.path, today=TODAY)
    assert (w.part_a_record, w.part_b_record) == (False, True)
    assert _commits(out) == (n + 1, vlh.REPAIR_COMMIT)
    assert _metadata(out) == sorted(meta + [ldr.PART_B_RECORD_KEY])  # no behaviour marker
    with contextlib.closing(fz._connect_ro(out)) as con:
        rec = ldr.parse_part_b_record(con)
        assert ldr.empty_shipped_defaults(con) == [extra]
    assert (rec.date, rec.set, rec.kept, rec.storage, rec.empty) == (TODAY, 3, 2, 1, ldr.empty_hash([extra]))
    shipped = {"unit.conversion_method": "constant_efficiency", "solve.solve_mode": "single_solve",
               STORAGE: "fix_nothing"}
    for name, member in shipped.items():
        cls, param = name.split(".")
        ((payload, typ),) = sql(out, "SELECT pd.default_value, pd.default_type FROM parameter_definition pd JOIN "
                                     "entity_class ec ON ec.id = pd.entity_class_id WHERE ec.name = ? AND pd.name = ?",
                                (cls, param))
        assert bytes(payload).isdigit() and typ == "list_value_ref", name
    from spinedb_api import from_database

    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.list_default_repair import list_defaults_repairable_ro

    with open_database_mapping(f"sqlite:///{out}") as db:
        for name, member in shipped.items():
            cls, param = name.split(".")
            item = db.get_item("parameter_definition", entity_class_name=cls, name=param)
            assert from_database(item["default_value"], item["default_type"]) == member
    assert not list_defaults_repairable_ro(out)
    assert _effective(out, "node", "storage_start_end_method") == {"A": "fix_start", "B": "fix_nothing"}
    # rerun on the copy: asks nothing
    again = tmp_path / "again" / "x.sqlite"
    again.parent.mkdir()
    a2, found2 = _prepare(out, again)
    prev = vw.previous_repair(a2, found2)
    assert (prev.part_a, prev.part_b, prev.suspects) == ("", f"defaults restored on {TODAY} (3 set, 2 kept)", ())
    vw.discard(a2.path)
    # a default emptied after the repair: asked again, the kept one pre-selected keep
    later = tmp_path / "later.sqlite"
    shutil.copy2(out, later)
    clear_defaults(later, ["unit.startup_method"], add_commit(later, "Updated", "2026-11-24"))
    a3, found3 = _prepare(later, tmp_path / "again" / "y.sqlite")
    prev = vw.previous_repair(a3, found3)
    assert prev.part_b == ""
    assert sorted((f"{s.cls}.{s.parameter}", s.suggestion) for s in prev.suspects) == sorted(
        [(extra, va.KEEP), ("unit.startup_method", "set:no_startup")])
    vw.discard(a3.path)


def test_unanswered_storage_holds_the_storage_default(tmp_path, schema_db_url) -> None:
    """R2: every default answered ``set:``, node B's storage question skipped → the storage default stays ``NULL``,
    no part B record, the summary says so; a rerun asks B again and the empty set still names the storage default."""
    case, extra = _lost(tmp_path, schema_db_url)
    out = tmp_path / "out.sqlite"
    a, found = _prepare(case.path, out)
    answers = _part_b_answers(found, (), None)
    w = vw.write_copy(a, va.make_plan(a, found, answers), out, source=case.path)
    assert not w.part_b_record and ldr.PART_B_RECORD_KEY not in _metadata(out)
    assert va.STORAGE_HELD in w.lines and va.STORAGE_HELD in w.report.read_text(encoding="utf-8")
    assert sql(out, "SELECT pd.default_value FROM parameter_definition pd JOIN entity_class ec ON ec.id = "
                    "pd.entity_class_id WHERE ec.name = 'node' AND pd.name = 'storage_start_end_method'") == [(None,)]
    with contextlib.closing(fz._connect_ro(out)) as con:
        assert ldr.empty_shipped_defaults(con) == [STORAGE]
    a2, found2 = _prepare(out, tmp_path / "again.sqlite")
    prev = vw.previous_repair(a2, found2)
    assert sorted((s.kind, s.entity) for s in prev.suspects) == [(vs.DEFAULT, ""), (ldr.STORAGE_KIND, "B")]
    vw.discard(a2.path)
