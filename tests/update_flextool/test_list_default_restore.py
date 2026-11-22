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
import subprocess
import sys
from pathlib import Path

from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from tests.update_flextool._value_list_cases import (add_commit, clear_defaults, make_hit, make_lost_defaults,
                                                     member_id, sql)

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
