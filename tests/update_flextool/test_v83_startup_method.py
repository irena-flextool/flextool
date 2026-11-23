"""Part 2 of v83: ``startup_methods`` gets linear_fix_in_uc, integer and integer_fix_in_uc, every ``binary`` value
and default becomes ``integer_fix_in_uc`` (the same model as 4.6.1 ``binary``), ``binary`` leaves the list, and the
six descriptions that name the members get their v83 text by text alone (plan §2, §0).

The part is called directly on a database made from the v82 schema (``schema_db_url``, copied into ``tmp_path``),
whose ``startup_methods`` is [no_startup, linear, binary] with the shipped default no_startup on unit and connection.
Expected values follow from the plan's rules alone: members are appended at the end in the order linear_fix_in_uc,
integer, integer_fix_in_uc and binary is removed, so the list reads no_startup, linear, linear_fix_in_uc, integer,
integer_fix_in_uc; only ``binary`` values / defaults change; a description changes only when it is the old text.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database
from spinedb_api.exception import NothingToCommit

from flextool.update_flextool import db_migration as m

MEMBERS = ["no_startup", "linear", "linear_fix_in_uc", "integer", "integer_fix_in_uc"]
#: The v83 texts, copied from plan §2 (the definitions not listed there keep their text).
OLD_UNIT_STARTUP = ("[choice] Whether the unit has online variables. linear: any share of the capacity can be online. "
                    "binary: whole units only (slower to solve). Online status links to the previous time step; "
                    "that link stops at a gap in the timeline.")
NEW = {
    ("unit", "startup_method"):
        "[choice] Whether the unit has online variables. linear: any share of the capacity can be online. integer: "
        "whole units only (a MIP, slower to solve); several units of one entity are counted as whole units. "
        "linear_fix_in_uc and integer_fix_in_uc: the same, and in a separate realization solve (a rolling "
        "separate_dispatch solve, or a single solve with a realized: yes branch) the units online over the jump (the "
        "realized periods) are kept at what the stochastic (unit commitment) solve chose; without _fix_in_uc the realization decides them "
        "again with the realized data. Elsewhere, including a rolling within_solve solve, a member and its "
        "_fix_in_uc variant are the same. Online status links to the previous time step; that link stops at a gap "
        "in the timeline.",
    ("connection", "startup_method"):
        "[choice] Gives the connection online capacity in both directions (linear: any share; integer: whole units, "
        "a MIP; linear_fix_in_uc and integer_fix_in_uc: the same, and in a separate realization solve the online "
        "capacity over the jump (the realized periods) is kept at what the stochastic solve chose). Flows and reserves are limited by the "
        "online capacity, and startup_cost is paid per MW brought online. No minimum load or minimum up/down time. "
        "Not allowed for dc_power_flow_with_angles lines.",
    ("unit", "constraint_online_coeff"):
        "[unit of the constraint per unit online] A map of coefficients (Index: constraint name, value: "
        "coefficient). Puts the number of units online in each time step on the left side of the user constraint. "
        "One unit is virtual_unitsize; without it, the largest existing capacity, else 1000 MW (set "
        "virtual_unitsize when the capacity is invested). Needs online variables (a startup_method other than "
        "no_startup, or a minimum_time_method); a coefficient without them is an error. With startup_method linear "
        "or linear_fix_in_uc the number is continuous. For MW online, use the unit size (MW per unit) as the "
        "coefficient.",
    ("connection", "constraint_online_coeff"):
        "[unit of the constraint per unit online] A map of coefficients (Index: constraint name, value: "
        "coefficient). Puts the number of units of the connection online in each time step on the left side of the "
        "user constraint. One unit is virtual_unitsize; without it, the largest existing capacity, else 1000 MW "
        "(set virtual_unitsize when the capacity is invested). Needs online variables (a startup_method other than "
        "no_startup); a coefficient without them is an error. With startup_method linear or linear_fix_in_uc the "
        "number is continuous. For MW online, use the unit size (MW per unit) as the coefficient.",
    ("unit__inputNode", "startup_input_method"):
        "[choice] What the input arc supplies. just_input (default): the normal input flow. startup_and_input: the "
        "normal input flow, and each start-up of the unit also takes startup_energy x MW started from this node. "
        "startup_only: only the start-up energy; the arc has no flow and takes no part in the conversion, the "
        "capacity, input shares or ramps. The start-up input can be fuel (oil to start a coal unit), heat (from the "
        "tank of a CSP plant) or electricity. Needs online variables (a startup_method other than no_startup, or a "
        "minimum_time_method). The start-up input is priced as the node's commodity, counts in CO2 prices and caps "
        "and is dropped with the start-ups by reduce_unit_detail_method online. It is not part of "
        "constraint_flow_coeff, flow groups or the capacity margin.",
}
REDUCE_OLD, REDUCE_NEW = ("integers: binary startup_method is solved as continuous",
                          "integers: startup_method integer and integer_fix_in_uc are solved as continuous")


def _copy(schema_db_url: str, folder: Path) -> str:
    path = folder / "db.sqlite"
    shutil.copy2(schema_db_url.removeprefix("sqlite:///"), path)
    return f"sqlite:///{path.resolve().as_posix()}"


def _raw(url: str, sql: str, *args) -> None:
    con = sqlite3.connect(url.removeprefix("sqlite:///"))
    try:
        con.execute(sql, args)
        con.commit()
    finally:
        con.close()


def _members(db) -> list[str]:
    items = sorted(db.get_items("list_value", parameter_value_list_name="startup_methods"), key=lambda x: x["index"])
    return [x["parsed_value"] for x in items]


def _values(db) -> dict[tuple[str, str, str, str], object]:
    out = {}
    for pdef in db.get_items("parameter_definition"):
        if pdef["parameter_value_list_name"] == "startup_methods":
            for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                               parameter_definition_name=pdef["name"]):
                out[(pv["entity_class_name"], pv["entity_byname"][-1], pdef["name"], pv["alternative_name"])] = \
                    pv["parsed_value"]
    return out


def _default(db, cls: str, name: str = "startup_method"):
    return db.get_item("parameter_definition", entity_class_name=cls, name=name)["parsed_value"]


def _description(db, cls: str, name: str) -> str:
    return db.get_item("parameter_definition", entity_class_name=cls, name=name)["description"]


def _add(db, cls: str, byname: tuple, name: str, alternative: str, value) -> None:
    raw, typ = to_database(value)
    db.add_item("parameter_value", entity_class_name=cls, entity_byname=byname, parameter_definition_name=name,
                alternative_name=alternative, value=raw, type=typ)


def _seed(url: str) -> None:
    """Unit u1 binary in base and alt, u2 linear, u3 no_startup + binary in alt, connection c binary, user
    definition unit.my_method on startup_methods binary; u4's binary in alt is stored as a plain string."""
    with DatabaseMapping(url) as db:
        db.add_item("alternative", name="alt")
        db.add_item("entity", entity_class_name="node", name="n1")
        db.add_item("entity", entity_class_name="node", name="n2")
        for unit in ("u1", "u2", "u3", "u4"):
            db.add_item("entity", entity_class_name="unit", name=unit)
        db.add_item("entity", entity_class_name="connection", name="c")
        db.add_item("entity", entity_class_name="connection__node__node", entity_byname=("c", "n1", "n2"))
        db.add_item("parameter_definition", entity_class_name="unit", name="my_method",
                    parameter_value_list_name="startup_methods")
        _add(db, "unit", ("u1",), "startup_method", "base", "binary")
        _add(db, "unit", ("u1",), "startup_method", "alt", "binary")
        _add(db, "unit", ("u2",), "startup_method", "base", "linear")
        _add(db, "unit", ("u3",), "startup_method", "base", "no_startup")
        _add(db, "unit", ("u3",), "startup_method", "alt", "binary")
        _add(db, "unit", ("u4",), "startup_method", "base", "linear")
        _add(db, "connection", ("c",), "startup_method", "base", "binary")
        _add(db, "unit", ("u2",), "my_method", "alt", "binary")
        db.commit_session("seed")
    # A plain string (not a list reference), as an external writer could store it.
    _raw(url, "INSERT INTO parameter_value (parameter_definition_id, entity_class_id, entity_id, alternative_id, "
              "value, type) SELECT pd.id, ec.id, e.id, a.id, ?, 'str' FROM parameter_definition pd "
              "JOIN entity_class ec ON pd.entity_class_id = ec.id JOIN entity e ON e.class_id = ec.id "
              "JOIN alternative a ON a.name = 'alt' WHERE ec.name = 'unit' AND pd.name = 'startup_method' "
              "AND e.name = 'u4'", b'"binary"')


#: After the part: each binary becomes integer_fix_in_uc, linear and no_startup stay.
AFTER = {
    ("unit", "u1", "startup_method", "base"): "integer_fix_in_uc",
    ("unit", "u1", "startup_method", "alt"): "integer_fix_in_uc",
    ("unit", "u2", "startup_method", "base"): "linear",
    ("unit", "u3", "startup_method", "base"): "no_startup",
    ("unit", "u3", "startup_method", "alt"): "integer_fix_in_uc",
    ("unit", "u4", "startup_method", "base"): "linear",
    ("unit", "u4", "startup_method", "alt"): "integer_fix_in_uc",
    ("connection", "c", "startup_method", "base"): "integer_fix_in_uc",
    ("unit", "u2", "my_method", "alt"): "integer_fix_in_uc",
}


def test_binary_values_and_members(schema_db_url: str, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    url = _copy(schema_db_url, tmp_path)
    _seed(url)
    with DatabaseMapping(url) as db:
        assert _members(db) == ["no_startup", "linear", "binary"]
        assert _values(db)[("unit", "u4", "startup_method", "alt")] == "binary"
        m._v83_startup_method(db)
        assert _members(db) == MEMBERS
        assert _values(db) == AFTER
        db.commit_session("v83 part 2")
    with DatabaseMapping(url) as db:
        assert _members(db) == MEMBERS
        assert _values(db) == AFTER
        assert (_default(db, "unit"), _default(db, "connection")) == ("no_startup", "no_startup")
    printed = capsys.readouterr().out
    assert "v83: the startup_methods member binary is now integer_fix_in_uc" in printed
    # one line per binary value: u1 base, u1 alt, u3 alt, u4 alt (plain string), c base, u2 my_method alt
    assert len([line for line in printed.splitlines() if line.endswith("= binary (startup_method)")
                or line.endswith("= binary (my_method)")]) == 6


def test_binary_default_is_rewritten(schema_db_url: str, tmp_path: Path) -> None:
    url = _copy(schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="unit", name="startup_method")
        raw, typ = to_database("binary")
        db.update_item("parameter_definition", id=pdef["id"], default_value=raw, default_type=typ)
        db.commit_session("binary default")
    with DatabaseMapping(url) as db:
        assert _default(db, "unit") == "binary"
        m._v83_startup_method(db)
        db.commit_session("v83 part 2")
    with DatabaseMapping(url) as db:
        assert (_default(db, "unit"), _default(db, "connection")) == ("integer_fix_in_uc", "no_startup")
        assert _members(db) == MEMBERS
        assert _description(db, "unit", "startup_method") == NEW[("unit", "startup_method")]


def test_null_default_gets_the_new_text(schema_db_url: str, tmp_path: Path) -> None:
    url = _copy(schema_db_url, tmp_path)
    _raw(url, "UPDATE parameter_definition SET default_value = NULL, default_type = NULL WHERE name = "
              "'startup_method' AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'unit')")
    with DatabaseMapping(url) as db:
        assert _default(db, "unit") is None
        assert _description(db, "unit", "startup_method") == OLD_UNIT_STARTUP
        m._v83_startup_method(db)
        assert _default(db, "unit") is None
        assert _description(db, "unit", "startup_method") == NEW[("unit", "startup_method")]


def test_each_old_text_is_replaced(schema_db_url: str, tmp_path: Path) -> None:
    with DatabaseMapping(_copy(schema_db_url, tmp_path)) as db:
        reduce_old = _description(db, "solve", "reduce_unit_detail_method")
        assert REDUCE_OLD in reduce_old
        assert not any(_description(db, *key) == text for key, text in NEW.items())
        m._v83_startup_method(db)
        for key, text in NEW.items():
            assert _description(db, *key) == text, key
        assert _description(db, "solve", "reduce_unit_detail_method") == reduce_old.replace(REDUCE_OLD, REDUCE_NEW)


def test_edited_description_is_kept(schema_db_url: str, tmp_path: Path) -> None:
    with DatabaseMapping(_copy(schema_db_url, tmp_path)) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="unit", name="startup_method")
        db.update_item("parameter_definition", id=pdef["id"], description=OLD_UNIT_STARTUP + " My note.")
        m._v83_startup_method(db)
        assert _description(db, "unit", "startup_method") == OLD_UNIT_STARTUP + " My note."
        assert _description(db, "connection", "startup_method") == NEW[("connection", "startup_method")]


def test_rerun_writes_nothing(schema_db_url: str, tmp_path: Path) -> None:
    """In the session (a rerun of the part) and after the commit (a kill after the v83 commit, then a rerun)."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url)
    with DatabaseMapping(url) as db:
        m._v83_startup_method(db)
        m._v83_startup_method(db)
        assert (_members(db), _values(db)) == (MEMBERS, AFTER)
        db.commit_session("v83 part 2")
    with DatabaseMapping(url) as db:
        m._v83_startup_method(db)
        assert (_members(db), _values(db)) == (MEMBERS, AFTER)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


def test_guard_stops_on_a_changed_member_without_a_commit(schema_db_url: str, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    url = _copy(schema_db_url, tmp_path)
    _seed(url)

    def overwrite_first(db, name, members):
        first = min(db.get_items("list_value", parameter_value_list_name=name), key=lambda x: x["index"])
        raw, typ = to_database("overwritten")  # the old spinedb_api bug rewrote a first member
        db.update_item("list_value", id=first["id"], value=raw, type=typ)

    monkeypatch.setattr(m, "_append_list_members", overwrite_first)
    with DatabaseMapping(url) as db:
        with pytest.raises(m.FlexToolMigrationError,
                           match=r"A version 83 upgrade step changed an existing member of the value list "
                                 r"startup_methods \('no_startup'\); the upgrade stops"):
            m._v83_startup_method(db)
    with DatabaseMapping(url) as db:
        assert _members(db) == ["no_startup", "linear", "binary"]
        assert _values(db)[("unit", "u1", "startup_method", "base")] == "binary"


def test_binary_removal_passes_the_guard(schema_db_url: str, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """The last guard call is the removal form with exactly the id of binary."""
    calls = []
    check = m._v83_check_members

    def recording(db, name, before, removed=frozenset()):
        calls.append({before[i][1] for i in removed})
        check(db, name, before, removed)

    monkeypatch.setattr(m, "_v83_check_members", recording)
    with DatabaseMapping(_copy(schema_db_url, tmp_path)) as db:
        m._v83_startup_method(db)
        assert _members(db) == MEMBERS
    assert calls == [set(), {b'"binary"'}]


def _binary_default(url: str) -> None:
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="unit", name="startup_method")
        raw, typ = to_database("binary")
        db.update_item("parameter_definition", id=pdef["id"], default_value=raw, default_type=typ)
        db.commit_session("binary default")


def test_leftover_default_reference_stops_the_removal(schema_db_url: str, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """A binary default left in place (the detach skipped) would read as linear_fix_in_uc after the commit, which
    takes the freed id: the part raises before the removal instead."""
    url = _copy(schema_db_url, tmp_path)
    _binary_default(url)
    monkeypatch.setattr(m, "_detach_list_defaults", lambda db, name, renames: [])
    with DatabaseMapping(url) as db:
        with pytest.raises(m.FlexToolMigrationError,
                           match=r"remove the member\(s\) 'binary' of the value list startup_methods while still "
                                 r"in use \(unit\.startup_method default\)"):
            m._v83_startup_method(db)
        assert "binary" in _members(db)


def test_leftover_value_reference_stops_the_removal(schema_db_url: str, tmp_path: Path) -> None:
    """A binary value the rewrite missed (the write skipped) stops the part before the removal."""
    url = _copy(schema_db_url, tmp_path)
    _seed(url)
    with DatabaseMapping(url) as db:
        db.update_parameter_value = lambda **kwargs: None
        with pytest.raises(m.FlexToolMigrationError, match=r"unit u1 \[base\] = binary \(startup_method\)"):
            m._v83_startup_method(db)
        assert "binary" in _members(db)
