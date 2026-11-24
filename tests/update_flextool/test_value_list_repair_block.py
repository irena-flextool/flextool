"""The repair tool's one question for all method parameter defaults (design ``repair_partB_prompt_design.md`` rev 3
§1b, §1c and the addendum): which entities rely on a default, which defaults part A asks about and suggests, the
branch and the block's text.

Every database is a made-up copy of the schema database (``_value_list_cases``).  Oracles from the construction:

- ``make_hit``: the unit, connection and node ``invest_method`` defaults reference R (``cumulative_limits`` now),
  commit 2 (the schema's, before every version commit); the schema has no entities; no FlexTool 3.x v22
  fingerprint commit, so an empty default is "no sign of the bug";
- ``make_lost_defaults``: the named defaults are ``NULL`` with the v78 step commit, after the fingerprint:
  "probably emptied by a Spine library bug"; FlexTool's defaults from ``spinedb_schema.json``: ``solve.solve_mode``
  single_solve, ``unit.conversion_method`` constant_efficiency, ``node.storage_binding_method``
  bind_forward_only;
- a commit added after the last version commit with a comment of the user's ("Updated") is a hand edit; one with
  ``db_migration.REFRESH_COMMIT`` is FlexTool's own.

The expected texts are the design's text parts (§1c table and the addendum) filled in by hand.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from flextool.update_flextool import db_migration as dbm
from flextool.update_flextool import frozen as fz
from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from flextool.update_flextool import value_list_write as vw
from tests.update_flextool._value_list_cases import (VERSION_PREFIX, add_commit, clear_defaults, make_hit,
                                                     make_lost_defaults, repair_part_b, sql, write_items)

W_A = ("Some invest_method defaults read cumulative_limits instead of not_allowed: a Spine library bug overwrote "
       "not_allowed in the invest_methods list (W-0490).")
O1_O2 = ("The default values for all or some methods are not set in this database. This is probably due to a Spine "
         "library bug (spinedb_api 0.31.0–0.32.2, databases upgraded 2024–2026).")
O1 = "The default values for all or some methods are not set in this database."
SETS_A = "{}: {} on the default, which reads cumulative_limits (FlexTool's default: not_allowed)"
KEPT_A = ("{}: 0 entities rely on the default, which reads cumulative_limits; changed by hand on 2026-12-01, so it is "
          "kept")
KEEP_IT = "If you chose cumulative_limits as a default yourself, type e and keep it."
EACH = "e goes through them here, one by one."
AS_THEY_ARE = "x skips: the defaults stay as they are, which may cause unwanted behaviour later."
UNSET = "x skips: the defaults stay unset, which may cause unwanted behaviour later."
HAND_DATE = "2026-12-01"
_DEFAULT_COMMIT = ("UPDATE parameter_definition SET commit_id = ? WHERE name = 'invest_method' AND entity_class_id = "
                   "(SELECT id FROM entity_class WHERE name = ?)")


def _analyse(path: Path, tmp_path: Path, name: str = "snap"):
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    a = vr.analyse(vr.snapshot(path, folder))
    return a, vr.all_suspects(a)


def _block(found):
    block = vs.defaults_block(found)
    change = vs.results_change(block)
    return block, change, ldr.block_lines(block, change), ldr.block_prompt(block, change)


def _by_hand(path: Path, classes=("unit", "connection", "node"), comment: str = "Updated") -> None:
    """Commit the *classes*' ``invest_method`` defaults (still on R) in a commit after the last version commit."""
    commit = add_commit(path, comment, HAND_DATE)
    for cls in classes:
        sql(path, _DEFAULT_COMMIT, (commit, cls))


def test_reliance_per_scenario_and_the_check_list(tmp_path, schema_db_url) -> None:
    """B1: U1 has its own value in ``base`` (in both scenarios); U5 only in ``inv``, which ``s1 = [base]`` lacks, so
    U5 relies on the unit default in s1.  Enter sets the unit default with an entity relying on it: results change,
    and the check list names U5 (no cumulative cap) and not U1."""
    case = make_hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")])
    write_items(case.path, [("unit", "U5", "invest_method", "inv", "invest_total")],
                {"s1": ["base"], "s2": ["base", "inv"]})
    _, found = _analyse(case.path, tmp_path)
    unit = next(s for s in found if s.kind == vs.DEFAULT and s.cls == "unit")
    assert (unit.context.relying, unit.context.relying_names) == (1, (("U5", False),))
    block, change, lines, prompt = _block(found)
    assert sorted(s.cls for s in block) == ["connection", "node", "unit"] and change is True
    assert lines == [
        W_A,
        "  " + SETS_A.format("connection.invest_method", "0 entities rely"),
        "  " + SETS_A.format("node.invest_method", "0 entities rely"),
        "  " + SETS_A.format("unit.invest_method", "1 entity relies"),
        "Enter restores FlexTool's method parameter defaults. Afterwards, check these entities and give them their "
        "own value by hand where needed:",
        "  unit.invest_method, no cumulative cap: U5 could invest and retire up to 1e6 MW until now",
        "  after the restore they neither invest nor retire; give them their own invest_method if they should",
        KEEP_IT, EACH, AS_THEY_ARE]
    assert prompt == "  [Enter = restore; e = one by one; x = skip; q = quit] > "


def test_a_default_changed_by_hand_is_kept(tmp_path, schema_db_url) -> None:
    """S1 pins.  An R default committed after the last version commit by the user: "changed by hand", ``keep``.  The
    same, then one more upgrade (a version commit): FlexTool's default again.  The same edit in FlexTool's refresh
    commit: not by hand.  A value written after the overwrite and after the last version commit is still asked with
    no suggestion (4.6.3 behaviour)."""
    case = make_hit(tmp_path, schema_db_url, name="hand.sqlite", after=[("unit", "U2", "base", "cumulative_limits")])
    _by_hand(case.path, ("unit",))
    _, found = _analyse(case.path, tmp_path, "s1")
    got = {(s.kind, s.cls, s.entity): (s.label, s.suggestion) for s in found}
    assert got == {("default", "unit", ""): ("changed by hand on 2026-12-01", "keep"),
                   ("default", "connection", ""): (vs.SHIPPED, "set:not_allowed"),
                   ("default", "node", ""): (vs.SHIPPED, "set:not_allowed"),
                   ("value", "unit", "U2"): (vs.AFTER, None)}
    assert next(s for s in found if s.cls == "unit" and s.kind == vs.DEFAULT).reason == vs.HAND_REASON

    add_commit(case.path, VERSION_PREFIX + "83", "2026-12-05")
    _, found = _analyse(case.path, tmp_path, "s2")
    assert {s.cls: s.suggestion for s in found if s.kind == vs.DEFAULT}["unit"] == "set:not_allowed"

    refreshed = make_hit(tmp_path, schema_db_url, name="refresh.sqlite")
    _by_hand(refreshed.path, ("unit",), dbm.REFRESH_COMMIT)
    _, found = _analyse(refreshed.path, tmp_path, "s3")
    assert {(s.label, s.suggestion) for s in found} == {(vs.SHIPPED, "set:not_allowed")}


def test_hit_at_an_older_data_version_and_no_entity_relying(tmp_path, schema_db_url) -> None:
    """B2: a hit at data version 82 (part B is off: no O1 line, W says "a" bug); no entity relies on any default, so
    Enter changes no result and R(i) takes its second clause; ``e`` is offered (a part A default), the defaults
    "stay as they are" on ``x``."""
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, "UPDATE parameter_definition SET default_value = ?, default_type = 'float' WHERE name = 'version' "
                   "AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'model')", (b"82.0",))
    a, found = _analyse(case.path, tmp_path)
    assert a.version == 82
    _, change, lines, prompt = _block(found)
    assert change is False
    assert lines == [W_A] + ["  " + SETS_A.format(f"{c}.invest_method", "0 entities rely")
                             for c in ("connection", "node", "unit")] + [
        "Enter restores FlexTool's method parameter defaults. This does not change results: no entity relies on the "
        "invest_method defaults that Enter changes.", KEEP_IT, EACH, AS_THEY_ARE]
    assert prompt == "  [Enter = restore; e = one by one; x = skip; q = quit] > "


def test_part_b_only_block_offers_no_e(tmp_path, schema_db_url) -> None:
    """B2: only empty defaults (all "probably"): O1 with O2, one line each, R(i)'s first clause, no ``e``."""
    case = make_lost_defaults(tmp_path, ["unit.conversion_method", "solve.solve_mode", "node.storage_binding_method"],
                              schema_db=schema_db_url)
    _, found = _analyse(case.path, tmp_path)
    _, change, lines, prompt = _block(found)
    assert change is False
    assert lines == [
        O1_O2,
        "  node.storage_binding_method: 0 entities rely on the default (bind_forward_only)",
        "  solve.solve_mode: 0 entities rely on the default (single_solve)",
        "  unit.conversion_method: 0 entities rely on the default (constant_efficiency)",
        "Enter restores FlexTool's method parameter defaults. This does not change results: FlexTool already uses "
        "these defaults when none is set.", UNSET]
    assert prompt == "  [Enter = restore; x = skip; q = quit] > "


def test_default_kept_in_an_earlier_repair(tmp_path, schema_db_url) -> None:
    """S2: a repair kept ``solve.solve_mode`` empty; ``unit.conversion_method`` was emptied by hand afterwards, so the
    record's hash no longer matches.  The kept one is suggested ``keep`` with its suffix, E names it, the hand-emptied
    one is restored."""
    case = make_lost_defaults(tmp_path, ["unit.conversion_method", "solve.solve_mode"], schema_db=schema_db_url)
    copy = repair_part_b(case.path, tmp_path / "out" / "copy.sqlite", keep=("solve.solve_mode",))
    with contextlib.closing(fz._connect_ro(copy)) as con:
        date = ldr.parse_part_b_record(con).date
    clear_defaults(copy, ["unit.conversion_method"], add_commit(copy, "Updated", "2026-12-02"))
    a, found = _analyse(copy, tmp_path)
    block, change, lines, _ = _block(vw.previous_repair(a, found).suspects)
    assert change is False and [s.suggestion for s in block] == ["keep", "set:constant_efficiency"]
    assert lines == [
        O1_O2,
        f"  solve.solve_mode: 0 entities rely on the default (single_solve); kept in the repair of {date}, so it "
        "stays unset",
        "  unit.conversion_method: 0 entities rely on the default (constant_efficiency); emptied by hand on "
        "2026-12-02 (Updated)",
        "Enter restores FlexTool's method parameter defaults and leaves the ones kept in an earlier repair. This "
        "does not change results: FlexTool already uses these defaults when none is set.", UNSET]


def test_block_of_kept_defaults_only(tmp_path, schema_db_url) -> None:
    """Addendum (CR-1 S8): every part A default changed by hand and nothing else: Enter keeps them, "This does not
    change results." without a clause, no "type e and keep it", ``e`` offered, "Enter = keep".  Plus one part B line
    that Enter sets: E restores and keeps, R(i) takes its first clause, still no K (no part A default is set)."""
    case = make_hit(tmp_path, schema_db_url)
    _by_hand(case.path)
    _, found = _analyse(case.path, tmp_path, "s1")
    _, change, lines, prompt = _block(found)
    assert change is False
    assert lines == [W_A] + ["  " + KEPT_A.format(f"{c}.invest_method") for c in ("connection", "node", "unit")] + [
        "Enter keeps the defaults changed by hand. This does not change results.", EACH, AS_THEY_ARE]
    assert prompt == "  [Enter = keep; e = one by one; x = skip; q = quit] > "

    clear_defaults(case.path, ["unit.conversion_method"], None)
    _, found = _analyse(case.path, tmp_path, "s2")
    _, change, lines, prompt = _block(found)
    assert change is False
    assert lines == [
        O1, W_A, "  " + KEPT_A.format("connection.invest_method"), "  " + KEPT_A.format("node.invest_method"),
        "  unit.conversion_method: 0 entities rely on the default (constant_efficiency); no sign of the bug",
        "  " + KEPT_A.format("unit.invest_method"),
        "Enter restores FlexTool's method parameter defaults, keeps the ones changed by hand. This does not change "
        "results: FlexTool already uses these defaults when none is set.", EACH, AS_THEY_ARE]
    assert prompt == "  [Enter = restore; e = one by one; x = skip; q = quit] > "
