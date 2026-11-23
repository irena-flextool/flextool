"""Part B of the value-list repair tool (package P2b): empty shipped list defaults and storage suspects.

Every database is a made-up v82 copy of the schema database (``_value_list_cases.make_lost_defaults``): the named
defaults are ``NULL`` with the v78 step commit, as on files upgraded before 4.6.3.  Oracles (written by hand):

- shipped members, from ``spinedb_schema.json``: ``unit.conversion_method`` = constant_efficiency,
  ``solve.solve_mode`` = single_solve, ``node.storage_binding_method`` = bind_forward_only,
  ``node.storage_start_end_method`` = fix_nothing; all four had a member default in the v25 template (bug-reachable);
  ``unit.end_of_life_method`` (example name below) is not in the v25 template;
- storage: the v77 rule (``classify_storage_default`` docstring) with the old default fix_start writes fix_start in
  the alternative that gives a node's own start value, for a node with no stored method in some scenario, unless
  that write changes how the node starts in a scenario where it becomes the method (a lower alternative's stored
  method).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from tests.update_flextool._value_list_cases import (add_commit, clear_defaults, make_hit, make_lost_defaults,
                                                     member_id, repair_part_b, sql)

SCHEMAS = Path(__file__).resolve().parents[2] / "flextool" / "schemas"
THREE = ["unit.conversion_method", "solve.solve_mode", "node.storage_binding_method"]
SHIPPED = {"unit.conversion_method": "set:constant_efficiency", "solve.solve_mode": "set:single_solve",
           "node.storage_binding_method": "set:bind_forward_only"}
STORAGE = "node.storage_start_end_method"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all(path: Path, tmp_path: Path):
    folder = tmp_path / "snap"
    folder.mkdir(parents=True, exist_ok=True)
    source = _sha(path)
    a = vr.analyse(vr.snapshot(path, folder))
    before = _sha(a.path)
    found = vr.all_suspects(a)
    assert _sha(a.path) == before and _sha(path) == source  # read-only
    return a, found


def _defaults(found) -> dict[str, vs.Suspect]:
    return {f"{s.cls}.{s.parameter}": s for s in found if s.kind == vs.DEFAULT}


def test_three_empty_defaults_with_the_fingerprint(tmp_path, schema_db_url) -> None:
    case = make_lost_defaults(tmp_path, THREE, schema_db=schema_db_url)
    a, found = _all(case.path, tmp_path)
    assert a.state == vr.NO_FINDING
    got = _defaults(found)
    assert {k: s.suggestion for k, s in got.items()} == SHIPPED and len(found) == 3
    for s in found:
        assert (s.label, s.current, s.reason, s.part_b, s.certain) == (ldr.PROBABLY, "", ldr.RESTORE_REASON, True,
                                                                        False)
        assert s.item.startswith("def:") and s.commit == case.emptied_commit
    assert len(vs.groups(found)) == 1  # one question "restore 3 shipped defaults"


def test_without_the_fingerprint_no_sign_of_the_bug(tmp_path, schema_db_url) -> None:
    case = make_lost_defaults(tmp_path, THREE, fingerprint=False, schema_db=schema_db_url)
    _, found = _all(case.path, tmp_path)
    assert {k: (s.suggestion, s.label) for k, s in _defaults(found).items()} == {
        k: (v, ldr.NO_SIGN) for k, v in SHIPPED.items()}


def test_emptied_by_hand_after_the_upgrade(tmp_path, schema_db_url) -> None:
    case = make_lost_defaults(tmp_path, THREE[:1], schema_db=schema_db_url)
    clear_defaults(case.path, ["solve.solve_mode"], add_commit(case.path, "Updated", "2025-11-03"))
    _, found = _all(case.path, tmp_path)
    labels = {k: s.label for k, s in _defaults(found).items()}
    assert labels == {"unit.conversion_method": ldr.PROBABLY, "solve.solve_mode": "emptied by hand on 2025-11-03 "
                      "(Updated)"}


def test_other_member_other_list_and_user_definitions_are_not_suspects(tmp_path, schema_db_url) -> None:
    case = make_lost_defaults(tmp_path, [], schema_db=schema_db_url)
    p = case.path
    none_id = member_id(p, "conversion_methods", "none")
    sql(p, "UPDATE parameter_definition SET default_value = ?, default_type = 'list_value_ref' WHERE name = "
           "'conversion_method'", (str(none_id).encode(),))
    # solve_mode bound to another list, NULL: not the shipped list, never touched
    sql(p, "UPDATE parameter_definition SET parameter_value_list_id = (SELECT id FROM parameter_value_list WHERE "
           "name = 'yes_no'), default_value = NULL, default_type = NULL WHERE name = 'solve_mode'")
    sql(p, "INSERT INTO parameter_definition (entity_class_id, name, parameter_value_list_id, default_value, "
           "default_type, description, commit_id) SELECT ec.id, 'my_method', l.id, NULL, NULL, '', 2 FROM "
           "entity_class ec, parameter_value_list l WHERE ec.name = 'unit' AND l.name = 'conversion_methods'")
    _, found = _all(p, tmp_path)
    assert found == ()


def test_w0490_hit_invest_defaults_are_part_a_only(tmp_path, schema_db_url) -> None:
    """A hit with NULL unit / connection / node ``invest_method`` defaults: each is one suspect, part A's label
    (``value_list_suspects.SHIPPED``); the group default (``not_active`` in its own list) is no suspect."""
    case = make_hit(tmp_path, schema_db_url)
    clear_defaults(case.path, ["unit.invest_method", "connection.invest_method", "node.invest_method"], None)
    _, found = _all(case.path, tmp_path)
    assert len({s.item for s in found}) == len(found) == 3
    assert sorted((s.cls, s.label, s.suggestion, s.part_b) for s in found) == [
        (cls, vs.SHIPPED, "set:not_allowed", False) for cls in ("connection", "node", "unit")]


def test_storage_suspects(tmp_path, schema_db_url) -> None:
    """Scenarios s1 = [base], s2 = [low, base] (base ranks higher).  A: start 0.5 in base before the upgrade, no
    method → write in base (start fixed under fix_start and under the written fix_start in s1 and s2).  B: the same,
    its start added after the upgrade.  C: stored method fix_nothing in base → no candidate.  D: start in base,
    fix_nothing in low: candidate in s1 only, but the write makes base D's method in s2 too, where it started free
    (low's fix_nothing) and would be fixed → not kept."""
    before = [("node", "A", "storage_state_start", "base", 0.5), ("node", "C", "storage_state_start", "base", 0.5),
              ("node", "C", "storage_start_end_method", "base", "fix_nothing"),
              ("node", "D", "storage_state_start", "base", 0.5),
              ("node", "D", "storage_start_end_method", "low", "fix_nothing")]
    case = make_lost_defaults(tmp_path, [STORAGE], schema_db=schema_db_url, before=before,
                              after=[("node", "B", "storage_state_start", "base", 0.3)],
                              scenarios={"s1": ["base"], "s2": ["low", "base"]})
    _, found = _all(case.path, tmp_path)
    storage = {s.entity: s for s in found if s.kind == ldr.STORAGE_KIND}
    assert sorted(storage) == ["A", "B"]
    entity = dict(sql(case.path, "SELECT name, id FROM entity"))
    base = sql(case.path, "SELECT id FROM alternative WHERE name = 'base'")[0][0]
    for name, label in (("A", ldr.STORAGE_FIXED), ("B", ldr.STORAGE_LATER)):
        s = storage[name]
        assert (s.item, s.alternative, s.parameter, s.current, s.label, s.suggestion, s.reason) == (
            f"sn:{entity[name]}:{base}", "base", "storage_start_end_method", "", label, None, ldr.STORAGE_NOTE)
        assert not s.certain and s.context.scenarios == ("s1", "s2")
    assert "keeps this node's start fixed" in storage["A"].evidence
    assert _defaults(found)[STORAGE].suggestion == "set:fix_nothing"
    # no v77 commit or no fingerprint: no storage suspects
    for kwargs in ({"v77": False}, {"fingerprint": False}):
        folder = tmp_path / str(kwargs)
        folder.mkdir()
        other = make_lost_defaults(folder, [STORAGE], schema_db=schema_db_url, before=before,
                                   scenarios={"s1": ["base"]}, **kwargs)
        _, found = _all(other.path, folder)
        assert [s.kind for s in found] == [vs.DEFAULT]


def test_old_storage_default_is_the_v25_shipped_default() -> None:
    v25 = json.loads((SCHEMAS / "pre_v26" / "flextool_template_v25.json").read_text(encoding="utf-8"))
    (default,) = [d for c, n, d, *_ in v25["parameter_definitions"] if (c, n) == ("node", "storage_start_end_method")]
    assert ldr.OLD_STORAGE_DEFAULT == default == "fix_start"


def test_empty_shipped_defaults_over_the_full_table(tmp_path, schema_db_url) -> None:
    import sqlite3

    case = make_lost_defaults(tmp_path, ["unit.conversion_method", "solve.solve_mode"], schema_db=schema_db_url)
    assert ("unit", "end_of_life_method") in ldr.reference() and ("unit", "end_of_life_method") not in \
        ldr.bug_reachable()
    clear_defaults(case.path, ["unit.end_of_life_method"], None)
    with sqlite3.connect(case.path) as con:
        names = ldr.empty_shipped_defaults(con)
    con.close()
    assert names == ["solve.solve_mode", "unit.conversion_method", "unit.end_of_life_method"]
    assert ldr.empty_hash(names) == hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
    # the sets, computed here from the two JSONs: a member default in the schema, and also in the v25 template
    def member_defaults(name):
        data = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
        lists = {}
        for lst, value, *_ in data["parameter_value_lists"]:
            lists.setdefault(lst, []).append(value)
        return {(c, n) for c, n, d, lst, *_ in data["parameter_definitions"] if lst and d in lists.get(lst, [])}
    current = member_defaults("spinedb_schema.json")
    assert set(ldr.reference()) == current
    assert ldr.bug_reachable() == current & member_defaults("pre_v26/flextool_template_v25.json")
    assert len(ldr.bug_reachable()) == 21


def test_module_imports_only_the_standard_library() -> None:
    """Loaded on its own (``import flextool`` itself imports far more), the module pulls in no third-party or
    FlexTool module."""
    code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('ldr', sys.argv[1]); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.bug_reachable(); "
            "print([n for n in sys.modules if n.split('.')[0] in ('spinedb_api', 'polars', 'flextool', 'sqlalchemy')])")
    out = subprocess.run([sys.executable, "-c", code, ldr.__file__], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_older_snapshot_gets_the_migrate_first_line(tmp_path, schema_db_url) -> None:
    case = make_lost_defaults(tmp_path, THREE, schema_db=schema_db_url)
    sql(case.path, "UPDATE parameter_definition SET default_value = ? WHERE name = 'version'", (b"76.0",))
    a, found = _all(case.path, tmp_path)
    assert a.version == 76 and found == () and vr.part_b_line(a) == ldr.MIGRATE_FIRST


# -- W-0491: the run / GUI note (package P7b) ------------------------------------------------------------------------
# Oracles (by hand): the note names the empty bug-reachable ``class.parameter`` names (sorted; in the line the
# storage one first) and carries sha1("\n".join(sorted full empty set))[:12]; it is off without the fingerprint,
# below the current data version, on an unreadable file, and when a valid part B record (comment ``REPAIR_COMMIT``)
# carries that hash.  ``unit.end_of_life_method`` is a reference default the bug could not reach (not in the v25
# template).
TWO = ["unit.conversion_method", "solve.solve_mode"]
NOT_REACHABLE = "unit.end_of_life_method"


def _hash(names) -> str:
    return hashlib.sha1("\n".join(sorted(names)).encode("utf-8")).hexdigest()[:12]


def test_note_names_the_empty_bug_reachable_defaults_until_the_repair(tmp_path, schema_db_url) -> None:
    import os

    case = make_lost_defaults(tmp_path, TWO, schema_db=schema_db_url)
    source = _sha(case.path)
    assert ldr.list_defaults_lost_ro(case.path) == ldr.LostDefaults(
        ("solve.solve_mode", "unit.conversion_method"), _hash(TWO))
    assert _sha(case.path) == source  # read-only
    fd = Path("/proc/self/fd")
    if fd.is_dir():  # no handle left open (Linux)
        assert not [p for p in fd.iterdir() if os.path.realpath(p) == str(case.path.resolve())]
    repaired = repair_part_b(case.path, tmp_path / "out" / "lost_repaired.sqlite")
    assert ldr.list_defaults_lost_ro(repaired) is None
    # a default emptied after the repair: the note is back, with the hash of the new set
    clear_defaults(repaired, ["unit.startup_method"], add_commit(repaired, "Updated", "2026-11-24"))
    assert ldr.list_defaults_lost_ro(repaired) == ldr.LostDefaults(("unit.startup_method",),
                                                                   _hash(["unit.startup_method"]))
    # the record copied into the original by an import: not the tool's commit, the note stays
    ((value,),) = sql(repaired, "SELECT value FROM metadata WHERE name = ?", (ldr.PART_B_RECORD_KEY,))
    commit = add_commit(case.path, "Import 3 items", "2026-11-24")
    sql(case.path, "INSERT INTO metadata (name, value, commit_id) VALUES (?, ?, ?)",
        (ldr.PART_B_RECORD_KEY, value, commit))
    assert ldr.list_defaults_lost_ro(case.path).names == ("solve.solve_mode", "unit.conversion_method")
    case.path.unlink()  # nothing holds the file (Windows)


def test_note_is_off_without_the_sign_of_the_bug_or_when_unreadable(tmp_path, schema_db_url) -> None:
    for kwargs in ({"fingerprint": False}, {}):
        folder = tmp_path / str(len(kwargs))
        folder.mkdir()
        case = make_lost_defaults(folder, TWO, schema_db=schema_db_url, **kwargs)
        if not kwargs:  # an older file (data version 76): not checked
            sql(case.path, "UPDATE parameter_definition SET default_value = ?, default_type = 'float' WHERE name = "
                           "'version' AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'model')",
                (b"76.0",))
        assert ldr.list_defaults_lost_ro(case.path) is None, kwargs
    # only a default the bug could not reach is empty
    assert (tuple(NOT_REACHABLE.split(".")) in ldr.reference()
            and tuple(NOT_REACHABLE.split(".")) not in ldr.bug_reachable())
    case = make_lost_defaults(tmp_path, [NOT_REACHABLE], schema_db=schema_db_url, name="unreach.sqlite")
    assert ldr.list_defaults_lost_ro(case.path) is None
    broken = tmp_path / "broken.sqlite"
    broken.write_bytes(b"not a database")
    assert ldr.list_defaults_lost_ro(broken) is None
    assert ldr.list_defaults_lost_ro(tmp_path / "missing.sqlite") is None
    with pytest.raises(sqlite3.Error):   # the GUI's check: a failed read is not cached as "nothing to say"
        ldr.list_defaults_lost_ro(broken, strict=True)


def test_note_record_hash_covers_the_full_empty_set(tmp_path, schema_db_url) -> None:
    """R1: a repair that keeps a default empty (a bug-reachable one, or one the bug could not reach) writes the hash
    of what stays empty, so the note ends; R2: a skipped storage question writes no record, so the note stays and
    names the storage default."""
    case = make_lost_defaults(tmp_path, TWO + [NOT_REACHABLE], schema_db=schema_db_url)
    assert ldr.list_defaults_lost_ro(case.path) == ldr.LostDefaults(
        ("solve.solve_mode", "unit.conversion_method"), _hash(TWO + [NOT_REACHABLE]))
    for keep in ([NOT_REACHABLE], ["solve.solve_mode"]):
        out = repair_part_b(case.path, tmp_path / keep[0] / "out.sqlite", keep=keep)
        assert ldr.list_defaults_lost_ro(out) is None, keep
    held = make_lost_defaults(tmp_path, [STORAGE], schema_db=schema_db_url, name="held.sqlite",
                              before=[("node", "A", "storage_state_start", "base", 0.5)], scenarios={"s1": ["base"]})
    out = repair_part_b(held.path, tmp_path / "held_out" / "out.sqlite", skip_storage=True)
    assert ldr.list_defaults_lost_ro(out) == ldr.LostDefaults((STORAGE,), _hash([STORAGE]))


def test_note_at_run_start(tmp_path, schema_db_url) -> None:
    """One Note line on the hit file, storage name first and three names shown; none on the repaired copy."""
    from flextool import run_warnings
    from flextool.run_warnings.registry import NOTE, WARNINGS
    from tests._support.coded_warnings import capture_coded

    case = make_lost_defaults(tmp_path, TWO + [STORAGE, "node.invest_method"], schema_db=schema_db_url)
    repaired = repair_part_b(case.path, tmp_path / "out" / "lost_repaired.sqlite")
    from flextool.cli import cmd_repair_value_list as cli

    spec = WARNINGS["W-0491"]
    assert spec.severity == NOTE and "`storage_start_end_method`" in spec.body
    assert f'python -P -m {cli.MODULE} "' in spec.body and 'flextool-repair-value-list "' in spec.body
    for path, expected in ((case.path, 1), (repaired, 0)):
        run_warnings.start_run()
        with capture_coded() as got:
            ldr.note_at_run_start(f"sqlite:///{path}")
            ldr.note_at_run_start(f"sqlite:///{path}")  # once per run
        assert len(got.printed) == expected, path.name
    assert [c.names for c in got.calls] == []
    run_warnings.start_run()
    with capture_coded() as got:
        ldr.note_at_run_start(f"sqlite:///{case.path}")
    (call,) = got.calls
    assert call.names == [STORAGE, "node.invest_method", "solve.solve_mode", "unit.conversion_method"]
    assert got.printed == ["Note: defaults emptied by an old upgrade (4 cases, e.g. 'node.storage_start_end_method', "
                           "'node.invest_method', 'solve.solve_mode') [W-0491]"]


def test_a_failing_w0490_check_does_not_stop_w0491(tmp_path, monkeypatch) -> None:
    """The run start calls W-0491 in its own ``try``: W-0490 raising still lets it run (the run is stopped right
    after, at the next call of the run start)."""
    from flextool.engine.solve import _orchestration as orch
    from flextool.update_flextool import value_list_history as vlh

    class Stop(Exception):
        pass

    def boom(db_url):
        raise RuntimeError("W-0490 check failed")

    def stop():
        raise Stop

    seen = []
    monkeypatch.setattr(vlh, "warn_at_run_start", boom)
    monkeypatch.setattr(ldr, "note_at_run_start", seen.append)
    monkeypatch.setattr(orch, "_reset_solver_args_echo", stop)
    db = tmp_path / "x.sqlite"
    with pytest.raises(Stop):
        orch.run_chain_from_db(f"sqlite:///{db}", "s", work_folder=tmp_path / "w")
    assert seen == [f"sqlite:///{db}"]
