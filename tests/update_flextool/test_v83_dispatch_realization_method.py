"""Part 5 of v83: ``dispatch_realization_methods`` gets ``deterministic`` as the new shipped default, and every
stochastic solve that relied on the default gets ``within_solve`` written so that it keeps its results (plan §2a).

The engine reads no solve definition default: a solve without a ``dispatch_realization_method`` value ran
within_solve.  So the rule (derived from plan §2a step 2 alone, not from the code): a scenario is a candidate for a
solve when its winning (highest-ranked) ``stochastic_branches`` value is non-empty and no method value wins; the write
goes into the alternative of that winning ``stochastic_branches`` value; with the writes in place every scenario where
a method won before must read the same method, else nothing is written for the solve.  A null method row (``b'null'``,
no type) that wins reads as absent, as the engine reads it (within_solve today); when it ranks above the branches'
alternative, the write goes into it (the row is overwritten), so the candidate reads within_solve.

Pure planner rows first, then the part called directly on a database made from the v82 schema (``schema_db_url``,
copied into ``tmp_path``): ``dispatch_realization_methods`` is [within_solve, separate_dispatch] with the shipped
default within_solve and the one alternative ``base``.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, to_database
from spinedb_api.exception import NothingToCommit

from flextool.update_flextool import db_migration as m

LIST = "dispatch_realization_methods"
MEMBERS = ["within_solve", "separate_dispatch", "deterministic"]
#: The texts, copied from plan §2a step 4.
OLD_TEXT = ("[choice] How the committed part of a stochastic solve is realized. within_solve (default): a realized: "
            "yes branch gets a realization solve automatically. separate_dispatch: always runs a separate realized "
            "dispatch solve; needs a realized: yes branch (error otherwise).")
NEW_TEXT = ("[choice] How a solve with stochastic_branches is solved. deterministic (default): the forecasts are "
            "ignored and the solve uses the data of the realized: yes branch, with a warning (an error without that "
            "branch). within_solve: the forecasts are solved together; a rolling solve realizes each jump with the "
            "realized: yes branch and looks ahead with the forecasts made at the end of the jump; a single solve with "
            "a realized: yes branch gets a separate realization solve. separate_dispatch: a first, stochastic solve "
            "fixes the units online of _fix_in_uc units; a realization solve then dispatches the realized: yes branch "
            "over the jump (a single solve: its realized periods) and looks ahead with the forecasts made at their "
            "end; needs a realized: yes branch. Without stochastic_branches the three are the same.")
UNDECIDED_PRINT = ("v83: dispatch_realization_method not written for solve v: a value in alternative base would "
                   "replace your dispatch_realization_method in alternative E in scenario S1. In scenario S2 it now "
                   "runs deterministic: the forecasts are ignored (results change), or the run stops when there is "
                   "no realized: yes branch. Set dispatch_realization_method within_solve yourself where it should "
                   "apply.")


# --- the pure planner -------------------------------------------------------------------------------------------
# Each row: branches {(solve, alternative): non-empty}, methods {(solve, alternative): method}, scenarios, expected
# writes, expected undecided solves.  The derivation of each row is its comment.
ROWS = {
    # S1 = [base]: branches win in base, no method → candidate, target base.
    "one_scenario": ({("v", "base"): True}, {}, {"S1": ["base"]}, (("v", "base"),), ()),
    # No stochastic_branches → never a candidate.
    "no_branches": ({}, {("v", "base"): "separate_dispatch"}, {"S1": ["base"]}, (), ()),
    # An explicit separate_dispatch wins in S1 → not a candidate.
    "explicit_method": ({("v", "base"): True}, {("v", "base"): "separate_dispatch"}, {"S1": ["base"]}, (), ()),
    # The winning stochastic_branches is empty → not stochastic there.
    "empty_branches": ({("v", "base"): False}, {}, {"S1": ["base"]}, (), ()),
    # The method sits in U, which no scenario uses → nothing resolves in S1 → written into base.
    "method_in_unused": ({("v", "base"): True}, {("v", "U"): "separate_dispatch"}, {"S1": ["base"]},
                         (("v", "base"),), ()),
    # S1 = [base, E]: E's separate_dispatch wins; S2 = [base]: candidate → base.  After: S1 still wins E → kept.
    "method_above_target": ({("v", "base"): True}, {("v", "E"): "separate_dispatch"},
                            {"S1": ["base", "E"], "S2": ["base"]}, (("v", "base"),), ()),
    # S1 = [E, base]: base ranks above E, so a write in base (for S2) would replace E's method in S1 → undecided.
    "target_above_method": ({("v", "base"): True}, {("v", "E"): "separate_dispatch"},
                            {"S1": ["E", "base"], "S2": ["base"]}, (), ("v",)),
    # E holds within_solve ranked below base: the write in base reads the same method in S1 → allowed.
    "target_above_same_method": ({("v", "base"): True}, {("v", "E"): "within_solve"},
                                 {"S1": ["E", "base"], "S2": ["base"]}, (("v", "base"),), ()),
    # Branches won in a1 in S1 and in a2 in S2 → one write per target.
    "two_targets": ({("v", "a1"): True, ("v", "a2"): True}, {}, {"S1": ["base", "a1"], "S2": ["base", "a2"]},
                    (("v", "a1"), ("v", "a2")), ()),
    # Branches only in a1: S1 = [base, a1] is a candidate, S2 = [base] is not stochastic → only a1.
    "branches_in_some_scenarios": ({("v", "a1"): True}, {}, {"S1": ["base", "a1"], "S2": ["base"]},
                                   (("v", "a1"),), ()),
    # S1 = [base, a1]: a1's empty map wins → not stochastic; S2 = [base] → base.  The base value reaches S1 too
    # (no method resolved there before: nothing to keep).
    "empty_above": ({("v", "base"): True, ("v", "a1"): False}, {}, {"S1": ["base", "a1"], "S2": ["base"]},
                    (("v", "base"),), ()),
    # base's null method row reads as no method (the engine skips it: within_solve today) → candidate, target base
    # (the row is overwritten).
    "null_in_target": ({("v", "base"): True}, {("v", "base"): None}, {"S1": ["base"]}, (("v", "base"),), ()),
    # S1 = [base, L, H]: H's null wins over L's separate_dispatch → absent (within_solve today) → candidate; H ranks
    # above base (the branches) → H is overwritten, S1 reads within_solve.
    "null_above_lower_method": ({("v", "base"): True}, {("v", "L"): "separate_dispatch", ("v", "H"): None},
                                {"S1": ["base", "L", "H"]}, (("v", "H"),), ()),
    # The same with no method in L: a write into base would be hidden by H → H.
    "null_above_target": ({("v", "base"): True}, {("v", "H"): None}, {"S1": ["base", "H"]}, (("v", "H"),), ()),
    # Two solves are decided apart: w undecided (as target_above_method) does not stop v.
    "two_solves": ({("v", "base"): True, ("w", "base"): True}, {("w", "E"): "separate_dispatch"},
                   {"S1": ["E", "base"], "S2": ["base"]}, (("v", "base"),), ("w",)),
}


@pytest.mark.parametrize("row", sorted(ROWS))
def test_planner(row: str) -> None:
    branches, methods, scenarios, writes, undecided = ROWS[row]
    got_writes, got_undecided = m.plan_dispatch_realization_method(branches, methods, scenarios)
    assert got_writes == writes
    assert tuple(u[0] for u in got_undecided) == undecided


def test_planner_undecided_reason() -> None:
    branches, methods, scenarios, _w, _u = ROWS["target_above_method"]
    assert m.plan_dispatch_realization_method(branches, methods, scenarios)[1] == ((
        "v", "a value in alternative base would replace your dispatch_realization_method in alternative E in "
        "scenario S1", ("S2",)),)


def test_planner_reason_names_every_failing_scenario() -> None:
    """S3 = [base] is the candidate; the base write would replace E's method in S1 and F's in S2: both named."""
    branches = {("v", "base"): True}
    methods = {("v", "E"): "separate_dispatch", ("v", "F"): "separate_dispatch"}
    scenarios = {"S1": ["E", "base"], "S2": ["F", "base"], "S3": ["base"]}
    assert m.plan_dispatch_realization_method(branches, methods, scenarios) == ((), ((
        "v", "a value in alternative base would replace your dispatch_realization_method in alternative E in "
        "scenario S1; a value in alternative base would replace your dispatch_realization_method in alternative F "
        "in scenario S2", ("S3",)),))


def test_scenarios_text() -> None:
    assert m._v83_scenarios_text(("S2",)) == "scenario S2"
    assert m._v83_scenarios_text(("S1", "S2")) == "scenarios S1, S2"
    assert m._v83_scenarios_text((m._V77_NO_SCENARIO,)) == "the database"


# --- on a database --------------------------------------------------------------------------------------------
def _copy(schema_db_url: str, folder: Path) -> str:
    path = folder / "db.sqlite"
    shutil.copy2(schema_db_url.removeprefix("sqlite:///"), path)
    return f"sqlite:///{path.resolve().as_posix()}"


def _raw(url: str, sql: str) -> None:
    con = sqlite3.connect(url.removeprefix("sqlite:///"))
    try:
        con.execute(sql)
        con.commit()
    finally:
        con.close()


def _branches() -> Map:
    """One forecast and its realized branch in p2020 (period → branch → analysis_time → realized → weight)."""
    def fan(realized: str) -> Map:
        return Map(["t0001"], [Map([realized], [1.0])])
    return Map(["p2020"], [Map(["f1", "real"], [fan("no"), fan("yes")])])


def _add(db, solve: str, name: str, alternative: str, value) -> None:
    raw, typ = to_database(value)
    db.add_item("parameter_value", entity_class_name="solve", entity_byname=(solve,), parameter_definition_name=name,
                alternative_name=alternative, value=raw, type=typ)


def _seed(url: str, scenarios: dict[str, list[str]], branches=(), methods=()) -> None:
    """Alternatives and scenarios (lowest rank first), solves with their values."""
    with DatabaseMapping(url) as db:
        alternatives = {a for alts in scenarios.values() for a in alts} | {a for _s, a in branches} | {
            a for _s, a, _m in methods}
        for alternative in sorted(alternatives - {"base"}):
            db.add_item("alternative", name=alternative)
        for scenario, alts in scenarios.items():
            db.add_item("scenario", name=scenario)
            for rank, alternative in enumerate(alts, start=1):
                db.add_item("scenario_alternative", scenario_name=scenario, alternative_name=alternative, rank=rank)
        for solve in sorted({s for s, _a in branches} | {s for s, _a, _m in methods}):
            db.add_item("entity", entity_class_name="solve", name=solve)
        for solve, alternative in branches:
            _add(db, solve, "stochastic_branches", alternative, _branches())
        for solve, alternative, method in methods:
            _add(db, solve, "dispatch_realization_method", alternative, method)
        db.commit_session("seed")


def _members(db) -> list[str]:
    items = sorted(db.get_items("list_value", parameter_value_list_name=LIST), key=lambda x: x["index"])
    return [x["parsed_value"] for x in items]


def _methods(db) -> dict[tuple[str, str], str]:
    return {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"] for pv in
            db.find_parameter_values(entity_class_name="solve", parameter_definition_name="dispatch_realization_method")}


def _pdef(db):
    return db.get_item("parameter_definition", entity_class_name="solve", name="dispatch_realization_method")


def _set_default(db, value) -> None:
    raw, typ = to_database(value)
    db.update_item("parameter_definition", id=_pdef(db)["id"], default_value=raw, default_type=typ)


def test_no_scenario_writes_into_the_branches_alternative(schema_db_url: str, tmp_path: Path,
                                                          capsys: pytest.CaptureFixture) -> None:
    """No scenario: the pseudo-scenario of all alternatives.  st has branches and no method → within_solve in base;
    det has none → nothing; sd has separate_dispatch → kept."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {}, branches=[("st", "base"), ("sd", "base")], methods=[("sd", "base", "separate_dispatch")])
    with DatabaseMapping(url) as db:
        db.add_item("entity", entity_class_name="solve", name="det")
        assert (_members(db), _pdef(db)["parsed_value"]) == (["within_solve", "separate_dispatch"], "within_solve")
        assert _pdef(db)["description"] == OLD_TEXT
        m._v83_dispatch_realization_method(db)
        db.commit_session("v83 part 5")
    with DatabaseMapping(url) as db:
        assert _members(db) == MEMBERS
        assert _methods(db) == {("st", "base"): "within_solve", ("sd", "base"): "separate_dispatch"}
        assert _pdef(db)["parsed_value"] == "deterministic"
        assert _pdef(db)["description"] == NEW_TEXT
    assert "v83: dispatch_realization_method within_solve written" in capsys.readouterr().out


def test_scenarios(schema_db_url: str, tmp_path: Path) -> None:
    """a: method in E above the target in S1, S2 without E → written into base, E kept.  b: branches in a1 (S1) and
    a2 (S2) → one write per target.  c: branches only in a1 → written into a1 only.  d: the method only in U, which
    no scenario uses → written into base."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["base", "E", "a1"], "S2": ["base", "a2"]},
          branches=[("a", "base"), ("b", "a1"), ("b", "a2"), ("c", "a1"), ("d", "base")],
          methods=[("a", "E", "separate_dispatch"), ("d", "U", "separate_dispatch")])
    with DatabaseMapping(url) as db:
        m._v83_dispatch_realization_method(db)
        db.commit_session("v83 part 5")
    with DatabaseMapping(url) as db:
        assert _methods(db) == {("a", "E"): "separate_dispatch", ("a", "base"): "within_solve",
                                ("b", "a1"): "within_solve", ("b", "a2"): "within_solve",
                                ("c", "a1"): "within_solve",
                                ("d", "U"): "separate_dispatch", ("d", "base"): "within_solve"}


def test_undecided_writes_nothing_and_prints(schema_db_url: str, tmp_path: Path,
                                             capsys: pytest.CaptureFixture) -> None:
    """S1 = [E, base]: a write into base (for S2 = [base]) would rank above E's separate_dispatch → undecided."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["E", "base"], "S2": ["base"]}, branches=[("v", "base")],
          methods=[("v", "E", "separate_dispatch")])
    with DatabaseMapping(url) as db:
        m._v83_dispatch_realization_method(db)
        assert _methods(db) == {("v", "E"): "separate_dispatch"}
        assert _members(db) == MEMBERS
    assert UNDECIDED_PRINT in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("default", [None, "separate_dispatch"])
def test_null_and_user_default_are_kept(schema_db_url: str, tmp_path: Path, default) -> None:
    """The engine never reads the default: a NULL or user-set default stays, the writes are as with the shipped one,
    and the description is replaced whatever the default."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["base"]}, branches=[("v", "base")])
    if default is None:
        _raw(url, "UPDATE parameter_definition SET default_value = NULL, default_type = NULL WHERE name = "
                  "'dispatch_realization_method'")
    else:
        with DatabaseMapping(url) as db:
            _set_default(db, default)
            db.commit_session("user default")
    with DatabaseMapping(url) as db:
        assert _pdef(db)["parsed_value"] == default
        m._v83_dispatch_realization_method(db)
        db.commit_session("v83 part 5")
    with DatabaseMapping(url) as db:
        assert _pdef(db)["parsed_value"] == default
        assert _methods(db) == {("v", "base"): "within_solve"}
        assert _pdef(db)["description"] == NEW_TEXT
        assert _members(db) == MEMBERS


def test_a_null_row_in_the_target_is_overwritten(schema_db_url: str, tmp_path: Path) -> None:
    """A null method row (b'null', no type) in the branches' alternative: the engine reads it as absent (within_solve
    today) → candidate, and the row itself gets within_solve (no second row in base)."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["base"]}, branches=[("v", "base")], methods=[("v", "base", None)])
    with DatabaseMapping(url) as db:
        assert _methods(db) == {("v", "base"): None}
        m._v83_dispatch_realization_method(db)
        db.commit_session("v83 part 5")
    with DatabaseMapping(url) as db:
        assert _methods(db) == {("v", "base"): "within_solve"}


def test_a_null_row_above_a_method_is_overwritten(schema_db_url: str, tmp_path: Path) -> None:
    """S1 = [base, L, H]: branches in base, separate_dispatch in L, a null row in H.  The engine reads H's null as
    absent (within_solve today), so S1 is a candidate; H ranks above base → H gets within_solve, L is kept."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["base", "L", "H"]}, branches=[("v", "base")],
          methods=[("v", "L", "separate_dispatch"), ("v", "H", None)])
    with DatabaseMapping(url) as db:
        m._v83_dispatch_realization_method(db)
        db.commit_session("v83 part 5")
    with DatabaseMapping(url) as db:
        assert _methods(db) == {("v", "L"): "separate_dispatch", ("v", "H"): "within_solve"}


def test_edited_description_is_kept(schema_db_url: str, tmp_path: Path) -> None:
    with DatabaseMapping(_copy(schema_db_url, tmp_path)) as db:
        db.update_item("parameter_definition", id=_pdef(db)["id"], description=OLD_TEXT + " My note.")
        m._v83_dispatch_realization_method(db)
        assert _pdef(db)["description"] == OLD_TEXT + " My note."
        assert _pdef(db)["parsed_value"] == "deterministic"


def test_rerun_writes_nothing(schema_db_url: str, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """In the session (a rerun of the part) and after the commit: no write, one already-applied log per rerun and
    none on the first run."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["base"]}, branches=[("v", "base")])
    after = {("v", "base"): "within_solve"}
    caplog.set_level(logging.INFO)

    def logged() -> int:
        return sum("dispatch_realization_method: already written" in r.getMessage() for r in caplog.records)

    with DatabaseMapping(url) as db:
        m._v83_dispatch_realization_method(db)
        assert logged() == 0
        m._v83_dispatch_realization_method(db)
        assert logged() == 1
        assert (_members(db), _methods(db), _pdef(db)["parsed_value"]) == (MEMBERS, after, "deterministic")
        db.commit_session("v83 part 5")
    with DatabaseMapping(url) as db:
        m._v83_dispatch_realization_method(db)
        assert logged() == 2
        assert (_members(db), _methods(db), _pdef(db)["parsed_value"]) == (MEMBERS, after, "deterministic")
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


def test_guard_stops_on_a_changed_member(schema_db_url: str, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    url = _copy(schema_db_url, tmp_path)
    _seed(url, {"S1": ["base"]}, branches=[("v", "base")])

    def overwrite_first(db, name, members):
        first = min(db.get_items("list_value", parameter_value_list_name=name), key=lambda x: x["index"])
        raw, typ = to_database("overwritten")  # the old spinedb_api bug rewrote a first member
        db.update_item("list_value", id=first["id"], value=raw, type=typ)

    monkeypatch.setattr(m, "_append_list_members", overwrite_first)
    with DatabaseMapping(url) as db:
        with pytest.raises(m.FlexToolMigrationError,
                           match=r"A version 83 upgrade step changed an existing member of the value list "
                                 r"dispatch_realization_methods \('within_solve'\); "
                                 r"the upgrade stops"):
            m._v83_dispatch_realization_method(db)
    with DatabaseMapping(url) as db:
        assert _members(db) == ["within_solve", "separate_dispatch"]
        assert _methods(db) == {}
