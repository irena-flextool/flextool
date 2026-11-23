"""The value-list repair tool's answers CSV, answer checks and plan (package P4).

Every database is a made-up copy of the schema database (``_value_list_cases``).  Oracles from the construction
(written by hand, never from the tool's output):

- ``make_hit``: R is the schema's ``not_allowed`` row, now reading ``cumulative_limits``; ``not_allowed`` is the
  member the write restores, so ``set:not_allowed`` is allowed on R's list; the unit, connection and node
  ``invest_method`` defaults point to R (three default suspects, recommended ``set:not_allowed`` with the reason);
  "before" values are certain ``set:not_allowed``; "after" values have no suggestion.  The group default is
  ``not_active`` in ``group_invest_methods`` (v82), whose members (schema) hold ``not_active`` and
  ``cumulative_limits`` but not ``not_allowed``.
- ``file_key`` = ``k<end_commit>-<session date 20240919>-<max commit id>``; without a W-0490 finding
  ``k0-00000000-<max commit id>``.  The max id is read here with raw SQL.
- ``make_lost_defaults``: a storage node A with its own start written before the upgrade, no method, scenario
  s1 = [base] → one storage suspect (``classify_storage_default``, see ``test_list_default_restore``); members
  allowed for it: ``fix_start`` only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_answers as va
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from tests.update_flextool._value_list_cases import (HIT_DATE, add_commit, clear_defaults, make_hit,
                                                     make_lost_defaults, sql, write_values)

STORAGE = "node.storage_start_end_method"
LATER = "2025-01-10"


def _analyse(path: Path, tmp_path: Path, name: str = "snap"):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    a = vr.analyse(vr.snapshot(path, folder))
    return a, vr.all_suspects(a)


def _max_commit(path: Path) -> int:
    return sql(path, 'SELECT max(id) FROM "commit"')[0][0]


def _rows(path: Path, delim: str = ",") -> list[list[str]]:
    import csv

    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=delim))


def _save(path: Path, rows, delim: str = ",", encoding: str = "utf-8-sig") -> None:
    import csv

    with open(path, "w", encoding=encoding, newline="") as f:
        csv.writer(f, delimiter=delim).writerows(rows)


def _hit(tmp_path, schema_db_url, **kwargs):
    case = make_hit(tmp_path, schema_db_url, before=[("unit", "001", "base", "not_allowed"),
                                                     ("unit", "U3", "base", "not_allowed")],
                    after=[("unit", "U2", "base", "cumulative_limits")], **kwargs)
    a, found = _analyse(case.path, tmp_path)
    return case, a, found


def _item(found, cls: str, kind: str = vs.DEFAULT, entity: str = "") -> str:
    (s,) = [s for s in found if (s.cls, s.kind, s.entity) == (cls, kind, entity)]
    return s.item


def test_round_trip_and_a_preselection_is_no_answer(tmp_path, schema_db_url) -> None:
    case, a, found = _hit(tmp_path, schema_db_url)
    key = f"k{case.end_commit}-{HIT_DATE.replace('-', '')}-{_max_commit(a.path)}"
    assert va.file_key(a) == key
    unit, conn, node = (_item(found, c) for c in ("unit", "connection", "node"))
    u1, u2, u3 = (_item(found, "unit", vs.VALUE, e) for e in ("001", "U2", "U3"))
    assert len(found) == 6
    # dry run: suggestions filled, every answer empty; nothing counts as answered
    dry = tmp_path / "dry.csv"
    va.write_answers(dry, found, key)
    assert dry.read_bytes().startswith(b"\xef\xbb\xbf" + ",".join(va.COLUMNS).encode())
    rows = {r[0]: dict(zip(va.COLUMNS, r)) for r in _rows(dry)[1:]}
    assert all(r["answer"] == "" and r["file_key"] == key for r in rows.values())
    assert rows[unit]["suggestion"] == f"set:not_allowed ({vs.DEFAULT_REASON})"
    assert rows[u1]["suggestion"] == "set:not_allowed" and rows[u2]["suggestion"] == ""
    assert (rows[u1]["current"], rows[u1]["entity"], rows[u1]["evidence"]) == ("cumulative_limits", "001",
                                                                               vs.CERTAIN)
    read = va.read_answers(dry, a, found)
    assert (read.answers, read.stale, read.warnings) == ({}, (), ())
    plan = va.make_plan(a, found, read.answers)
    assert (plan.part_a_record, plan.part_b_record, plan.sets, plan.answers) == (False, False, (), {})
    assert {d.action for d in plan.decisions} == {va.SKIP} and plan.restore == ("not_allowed",)
    # --accept-certain answers the two certain values only
    assert va.accept_certain(found) == {u1: va.Answer(va.SET, "not_allowed"), u3: va.Answer(va.SET, "not_allowed")}
    # explicit answers round trip
    answers = {unit: va.Answer(va.SET, "not_allowed"), conn: va.Answer(va.KEEP), node: va.Answer(va.SKIP),
               u1: va.Answer(va.SET, "not_allowed"), u2: va.Answer(va.KEEP)}
    out = tmp_path / "answers.csv"
    va.write_answers(out, found, key, answers)
    read = va.read_answers(out, a, found)
    assert read.answers == answers
    assert va.make_plan(a, found, read.answers) == va.make_plan(a, found, answers)
    plan = va.make_plan(a, found, answers)
    assert sorted(d.suspect.item for d in plan.sets) == sorted([unit, u1])
    assert plan.part_a_record is False  # node skipped, U3 unanswered
    assert plan.counts(part_b=False) == {"set": 2, "kept": 2}


def test_excel_semicolons_case_and_number_names(tmp_path, schema_db_url) -> None:
    """Finnish Excel: ``;``, ``Keep``, entity ``001`` turned into ``1`` → warning, applied.  Another entity text, or
    a changed ``current``, makes the row stale: not applied, no record."""
    _, a, found = _hit(tmp_path, schema_db_url)
    key = va.file_key(a)
    path = tmp_path / "a.csv"
    va.write_answers(path, found, key)
    rows = _rows(path)
    col = {c: i for i, c in enumerate(va.COLUMNS)}
    for r in rows[1:]:
        r[col["answer"]] = "Keep"
        if r[col["entity"]] == "001":
            r[col["entity"]], r[col["alternative"]] = "1", " base"
            r[col["answer"]] = "SET:Not_Allowed"
    _save(path, rows + [[""] * len(va.COLUMNS)], ";")
    assert b";" in path.read_bytes().splitlines()[0]
    read = va.read_answers(path, a, found)
    u1 = _item(found, "unit", vs.VALUE, "001")
    assert read.stale == () and len(read.warnings) == 2 and all(w.startswith(u1) for w in read.warnings)
    assert read.answers[u1] == va.Answer(va.SET, "not_allowed")
    assert {v for k, v in read.answers.items() if k != u1} == {va.Answer(va.KEEP)} and len(read.answers) == 6
    assert va.make_plan(a, found, read.answers).part_a_record is True

    u2 = _item(found, "unit", vs.VALUE, "U2")
    for change in (("entity", "U2b"), ("current", "not_allowed")):
        bad = [list(r) for r in rows]
        for r in bad:
            if r[0] == u2:
                r[col[change[0]]] = change[1]
        _save(path, bad, ";")
        read = va.read_answers(path, a, found)
        assert [item for item, _ in read.stale] == [u2] and u2 not in read.answers
        assert va.make_plan(a, found, read.answers, stale=read.stale).part_a_record is False


def test_refused_files_and_answers(tmp_path, schema_db_url) -> None:
    case, a, found = _hit(tmp_path, schema_db_url)
    key = va.file_key(a)
    path = tmp_path / "a.csv"
    va.write_answers(path, found, key)
    rows = _rows(path)
    unit = _item(found, "unit")
    # cp1252 (Excel's plain "CSV" in Finland) with a non-ASCII name
    _save(path, rows[:1] + [[*r[:5], "Hämeenlinna", *r[6:]] for r in rows[1:]], ";", "cp1252")
    with pytest.raises(va.AnswersError, match="CSV UTF-8"):
        va.read_answers(path, a, found)
    for row, match in (([*rows[1][:1], *rows[1][1:]], "twice"), (["pv:999999", *rows[1][1:]], "not an item"),
                       ([rows[1][0], "k1-20240919-1", *rows[1][2:]], "another copy")):
        _save(path, rows + [row])
        with pytest.raises(va.AnswersError, match=match):
            va.read_answers(path, a, found)
    allowed = va.choices(a, found)[unit]
    assert {"not_allowed", "cumulative_limits"} <= set(allowed) and "not_active" not in allowed
    for answer, match in (("set:not_active", "not a member"), ("set:bogus", "not a member"), ("yes", "not an answer")):
        bad = [list(r) for r in rows]
        bad[[r[0] for r in bad].index(unit)][8] = answer
        _save(path, bad)
        with pytest.raises(va.AnswersError, match=match):
            va.read_answers(path, a, found)
    with pytest.raises(va.AnswersError, match="not a member"):
        va.make_plan(a, found, {unit: va.Answer(va.SET, "not_active")})
    # a sibling copy with one more commit refuses the file written for the first one
    sibling = tmp_path / "sibling.sqlite"
    shutil.copy2(case.path, sibling)
    add_commit(sibling, "Updated", LATER)
    va.write_answers(path, found, key)
    b, found_b = _analyse(sibling, tmp_path, "snap_b")
    with pytest.raises(va.AnswersError, match="another copy"):
        va.read_answers(path, b, found_b)


def test_carried_group_value_is_checked_against_its_own_list(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url)
    write_values(case.path, [("group", "G1", "base", "cumulative_limits")])
    v77 = add_commit(case.path, "v77: schema batch", LATER)
    sql(case.path, "UPDATE parameter_value SET commit_id = ? WHERE entity_id = (SELECT id FROM entity WHERE "
                   "name = 'G1')", (v77,))
    a, found = _analyse(case.path, tmp_path)
    g1 = _item(found, "group", vs.CARRIED, "G1")
    s = next(s for s in found if s.item == g1)
    allowed = va.choices(a, found)[g1]
    assert va.check_answer(s, "set:cumulative_limits", allowed) == va.Answer(va.SET, "cumulative_limits")
    assert va.check_answer(s, "set:not_active", allowed) == va.Answer(va.SET, "not_active")
    with pytest.raises(va.AnswersError):
        va.check_answer(s, "set:not_allowed", allowed)


def test_all_keep_gives_a_record_one_skip_none(tmp_path, schema_db_url) -> None:
    """Defaults-only hit: three default suspects.  All ``keep`` → record due; one skip → none.  A file with no
    overwrite (not-hit) and no suspects → record due with no answers."""
    case = make_hit(tmp_path, schema_db_url)
    a, found = _analyse(case.path, tmp_path)
    assert [s.kind for s in found] == [vs.DEFAULT] * 3
    keep = {s.item: va.Answer(va.KEEP) for s in found}
    plan = va.make_plan(a, found, keep)
    assert (plan.part_a_record, plan.part_b_record, plan.counts(False)) == (True, False, {"set": 0, "kept": 3})
    assert va.make_plan(a, found, {**keep, found[0].item: va.Answer(va.SKIP)}).part_a_record is False
    (tmp_path / "n").mkdir()
    not_hit = make_hit(tmp_path / "n", schema_db_url, overwrite=False)
    b, none = _analyse(not_hit.path, tmp_path, "snap_n")
    assert (b.state, none) == (vr.NOT_HIT, ())
    assert va.make_plan(b, none, {}).part_a_record is True


def _lost(tmp_path, schema_db_url, name="lost.sqlite"):
    case = make_lost_defaults(tmp_path, ["unit.conversion_method", STORAGE], schema_db=schema_db_url, name=name,
                              before=[("node", "A", "storage_state_start", "base", 0.5)],
                              scenarios={"s1": ["base"]})
    a, found = _analyse(case.path, tmp_path, f"snap_{name}")
    by = {(s.kind, f"{s.cls}.{s.parameter}"): s.item for s in found}
    return case, a, found, by[(vs.DEFAULT, "unit.conversion_method")], by[(vs.DEFAULT, STORAGE)], \
        by[(ldr.STORAGE_KIND, STORAGE)]


def test_part_b_only_file(tmp_path, schema_db_url) -> None:
    case, a, found, conv, default, node = _lost(tmp_path, schema_db_url)
    assert len(found) == 3 and node.startswith("sn:")
    key = va.file_key(a)
    assert key == f"k0-00000000-{_max_commit(a.path)}"
    path = tmp_path / "b.csv"
    va.write_answers(path, found, key)
    row = {r[0]: dict(zip(va.COLUMNS, r)) for r in _rows(path)[1:]}[node]
    assert (row["kind"], row["class"], row["parameter"], row["current"], row["answer"], row["suggestion"]) == (
        "storage", "node", "storage_start_end_method", "", "", "")
    s = next(x for x in found if x.item == node)
    allowed = va.choices(a, found)[node]
    assert allowed == ("fix_start",)
    assert va.check_answer(s, "SET:FIX_START", allowed) == va.Answer(va.SET, "fix_start")
    assert va.check_answer(s, "keep", allowed) == va.Answer(va.KEEP)
    with pytest.raises(va.AnswersError):
        va.check_answer(s, "set:fix_nothing", allowed)
    answers = {conv: va.Answer(va.SET, "constant_efficiency"), default: va.Answer(va.SET, "fix_nothing"),
               node: va.Answer(va.SET, "fix_start")}
    va.write_answers(path, found, key, answers)
    read = va.read_answers(path, a, found)
    assert read.answers == answers
    plan = va.make_plan(a, found, read.answers)
    assert (plan.part_a_record, plan.part_b_record, plan.held) == (False, True, ())
    assert plan.counts(part_b=True) == {"set": 2, "kept": 0, "storage": 1}
    # the same file on a sibling copy with one more commit: refused
    sibling = tmp_path / "sibling.sqlite"
    shutil.copy2(case.path, sibling)
    add_commit(sibling, "Updated", LATER)
    b, found_b = _analyse(sibling, tmp_path, "snap_sibling")
    assert va.file_key(b) == f"k0-00000000-{_max_commit(a.path) + 1}"
    with pytest.raises(va.AnswersError, match="another copy"):
        va.read_answers(path, b, found_b)


def test_storage_default_held_while_a_storage_question_is_open(tmp_path, schema_db_url) -> None:
    """R2: the storage default answered ``set:`` but the storage node skipped → the default is not set, no part B
    record; the other default is set."""
    _, a, found, conv, default, node = _lost(tmp_path, schema_db_url)
    answers = {conv: va.Answer(va.SET, "constant_efficiency"), default: va.Answer(va.SET, "fix_nothing")}
    for extra in ({}, {node: va.Answer(va.SKIP)}):
        plan = va.make_plan(a, found, {**answers, **extra})
        assert plan.held == (default,) and [d.suspect.item for d in plan.sets] == [conv]
        assert plan.part_b_record is False and plan.counts(True) == {"set": 1, "kept": 0, "storage": 0}
    plan = va.make_plan(a, found, {**answers, node: va.Answer(va.KEEP)})
    assert plan.held == () and plan.part_b_record is True
    assert "left empty until every storage question" in va.STORAGE_HELD


def test_part_a_and_part_b_records_are_due_independently(tmp_path, schema_db_url) -> None:
    """A hit at v82 with one empty part B default (``unit.conversion_method``): three part A defaults, one part B."""
    case = make_hit(tmp_path, schema_db_url)
    clear_defaults(case.path, ["unit.conversion_method"], None)
    a, found = _analyse(case.path, tmp_path)
    part_a = {s.item: va.Answer(va.KEEP) for s in found if not s.part_b}
    part_b = {s.item: va.Answer(va.SET, "constant_efficiency") for s in found if s.part_b}
    assert (len(part_a), len(part_b)) == (3, 1)
    for answers, due in ((part_a, (True, False)), (part_b, (False, True)), ({**part_a, **part_b}, (True, True))):
        plan = va.make_plan(a, found, answers)
        assert (plan.part_a_record, plan.part_b_record) == due
