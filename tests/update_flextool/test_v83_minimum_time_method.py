"""Part 1 of v83: ``minimum_time_method`` written where a unit relied on a min up/down time alone (C rev 3 §6).

M1 (C rev 3 §10.3): every row on the pure planner :func:`plan_minimum_time_method`; the DB rows also with part 1
called directly on a database made from the v82 schema (``v82_schema_db_url``, copied into ``tmp_path``).  M3: the
planner's model of today's engine (``online``, ``need``) equals the engine's classification on the same database.

Rules the expectations are derived from (C rev 3 §6.2-§6.3; scenarios lowest rank first, the last wins; no
``startup_method`` default ``D`` = ``no_startup``, no method default unless stated):

* online: ``startup_method`` in {linear, binary, ...}, or ``no_startup`` with an enabling method, or ``no_startup``
  with ``piecewise_linear`` and a curve that burns fuel at zero output; need: the times > 0 when online;
* candidate scenario: need non-empty and no method resolves (an explicit value wins, or a default other than none);
* M(u): ``both`` if the candidates need both times, else the one; written into the ``startup_method`` winner of each
  candidate scenario, else into its highest alternative holding any value of u;
* ``none`` where u is not online but M(u) would win: into the alternative whose value keeps u off there (its
  ``startup_method``, else ``conversion_method``, else curve winner), else, when that one ranks below the write
  (or there is none), into the scenario's highest alternative;
* a null value that wins reads as unset: the definition default applies (not a lower alternative's value); a null
  ``minimum_time_method`` row that wins above the write target would hide the write, so M(u) goes into that row;
* then every scenario is checked (a resolving method unchanged, the needed times enabled, no enabling method where
  u is not online); a failure leaves u undecided with no write.
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, to_database
from spinedb_api.exception import NothingToCommit

from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import common, v77, v83
from tests.update_flextool._rerun import patch_migration

MARKER = m.BEHAVIOUR_CHANGE_MARKER_PREFIX + v83.V83_MIN_TIME_MARKER
RISING = Map([0.4, 0.7, 1.0], [0.30, 0.40, 0.42])  # fuel 1.33, 1.75, 2.38: two lines, rising efficiency
FLAT = Map([0.5, 1.0], [0.4, 0.4])  # fuel 1.25, 2.5: one line through 0, no fuel at zero output → no auto_startup
BAD = Map([0.5, 0.9], [0.4, 0.4])  # ends at 0.9, not 1.0: a curve with a problem counts as no curve
#: The auto_startup of each curve, derived above.
AUTO = ((RISING, True), (FLAT, False), (BAD, False))
N0 = {"startup_method": "no_startup"}
D_ON = {"startup_method": "linear", "min_uptime": 3.0}
#: name -> (unit values {param: {alternative: value}}, scenarios, defaults, writes {alternative: method},
#: undecided (reason, candidate scenarios) or None, a DB row)
CASES = {
    # S: linear online, min_uptime needed, nothing resolves → M = min_uptime into the startup winner base.
    "plain": ({"min_uptime": {"base": 4.0}, "startup_method": {"base": "linear"}}, {"S": ["base"]}, N0,
              {"base": "min_uptime"}, None, True),
    # SA needs up, SB down, SAB both; binary (pre-part-2 name) is online → M = both into base (startup winner).
    "split": ({"min_uptime": {"A": 2.0}, "min_downtime": {"B": 3.0}, "startup_method": {"base": "binary"}},
              {"SA": ["base", "A"], "SB": ["base", "B"], "SAB": ["base", "A", "B"]}, N0, {"base": "both"}, None, True),
    # S1 candidate (down) → base; S2: off's no_startup wins, not online, but base's write would win → none into off.
    "ft2_off": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear", "off": "no_startup"}},
                {"S1": ["base"], "S2": ["base", "off"]}, N0, {"base": "min_downtime", "off": "none"}, None, True),
    # top's explicit none resolves in the only scenario: no candidate.
    "explicit_none": ({"min_uptime": {"base": 2.0}, "min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"},
                       "minimum_time_method": {"top": "none"}}, {"S": ["base", "top"]}, N0, {}, None, False),
    # base's own min_uptime resolves: no candidate (its min_downtime goes: the user's method).
    "explicit_single": ({"min_uptime": {"base": 2.0}, "min_downtime": {"base": 2.0},
                         "startup_method": {"base": "linear"}, "minimum_time_method": {"base": "min_uptime"}},
                        {"S": ["base"]}, N0, {}, None, False),
    # no startup, no method, no curve: not online, no need.
    "no_online": ({"min_uptime": {"base": 2.0}}, {"S": ["base"]}, N0, {}, None, False),
    # 0 is not a time.
    "zero": ({"min_uptime": {"base": 0.0}, "startup_method": {"base": "linear"}}, {"S": ["base"]}, N0, {}, None,
             False),
    # scratch is in no scenario, so its none never counts: as plain with min_downtime.
    "explicit_unused_alt": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"},
                             "minimum_time_method": {"scratch": "none"}}, {"S": ["base"]}, N0,
                            {"base": "min_downtime"}, None, True),
    # S1 [base, E]: E's none resolves and still wins over the base write; S2 [base] is the candidate.
    "explicit_some_scenarios": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"},
                                 "minimum_time_method": {"E": "none"}}, {"S1": ["base", "E"], "S2": ["base"]}, N0,
                                {"base": "min_downtime"}, None, True),
    # S2 [base] is the candidate → base; in S1 [E, base] base then wins over E's none: R1 fails in S1.
    "masking_conflict": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"},
                          "minimum_time_method": {"E": "none"}}, {"S1": ["E", "base"], "S2": ["base"]}, N0, {},
                         ("a value in alternative base would replace your minimum_time_method in alternative E in "
                          "scenario S1", ("S2",)), True),
    # no_startup by default, piecewise with a rising curve → online; both times; no startup value → base (held).
    "pw_auto": ({"conversion_method": {"base": "piecewise_linear"}, "piecewise_linear_efficiency": {"base": RISING},
                 "min_uptime": {"base": 2.0}, "min_downtime": {"base": 2.0}}, {"S": ["base"]}, N0,
                {"base": "both"}, None, True),
    # top's explicit no_startup is the startup winner, the curve keeps it online → top: both.
    "pw_auto_off": ({"conversion_method": {"base": "piecewise_linear"},
                     "piecewise_linear_efficiency": {"base": RISING}, "min_uptime": {"base": 2.0},
                     "min_downtime": {"base": 2.0}, "startup_method": {"top": "no_startup"}},
                    {"S": ["base", "top"]}, N0, {"top": "both"}, None, True),
    # S1 curve-online (up) → base (held); S2: C's constant_efficiency keeps it off (C ranks above base) → none
    # into C.
    "pw_mixed": ({"conversion_method": {"base": "piecewise_linear", "C": "constant_efficiency"},
                  "piecewise_linear_efficiency": {"base": RISING}, "min_uptime": {"base": 2.0}},
                 {"S1": ["base"], "S2": ["base", "C"]}, N0, {"base": "min_uptime", "C": "none"}, None, True),
    # FLAT burns no fuel at zero output (no auto_startup), no startup: not online, nothing.
    "pw_flat": ({"conversion_method": {"base": "piecewise_linear"}, "piecewise_linear_efficiency": {"base": FLAT},
                 "min_uptime": {"base": 2.0}}, {"S": ["base"]}, N0, {}, None, True),
    # BAD has a problem: no curve, so not online, nothing.
    "pw_bad_curve": ({"conversion_method": {"base": "piecewise_linear"}, "piecewise_linear_efficiency": {"base": BAD},
                      "min_uptime": {"base": 2.0}}, {"S": ["base"]}, N0, {}, None, True),
    # the time comes from the min_uptime default 3 → candidate → base (startup winner).
    "default_time": ({"startup_method": {"base": "linear"}}, {"S": ["base"]}, {**N0, "min_uptime": 3.0},
                     {"base": "min_uptime"}, None, True),
    # online by D = linear, time by default; no startup value → base holds existing.
    "default_time_D_on": ({"existing": {"base": 1.0}}, {"S": ["base"]}, D_ON, {"base": "min_uptime"}, None, False),
    # the same with no value at all: no target.
    "default_time_D_on_no_values": ({}, {"S": ["base"]}, D_ON, {},
                                    ("no alternative of scenario S holds a value of the unit", ("S",)), False),
    # S1 online by D (up) → base (held); S2 off's no_startup → not online, base would win → none into off.
    "D_on_with_off": ({"min_uptime": {"base": 2.0}, "startup_method": {"off": "no_startup"}},
                      {"S1": ["base"], "S2": ["base", "off"]}, {"startup_method": "linear"},
                      {"base": "min_uptime", "off": "none"}, None, False),
    # S2 [off, base]: not online, base would win, but S2's top is base itself, the target.
    "D_on_off_below": ({"min_uptime": {"base": 2.0}, "startup_method": {"off": "no_startup"}},
                       {"S1": ["base"], "S2": ["off", "base"]}, {"startup_method": "linear"}, {},
                       ("a value in alternative base would give the unit online variables in scenario S2", ("S1",)),
                       False),
    # a none default is no default: as plain.
    "none_default": ({"min_uptime": {"base": 4.0}, "startup_method": {"base": "linear"}}, {"S": ["base"]},
                     {**N0, "minimum_time_method": "none"}, {"base": "min_uptime"}, None, True),
    # a min_uptime default resolves everywhere: the user's method, no candidate.
    "single_default": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"}}, {"S": ["base"]},
                       {**N0, "minimum_time_method": "min_uptime"}, {}, None, False),
    # S1 [base, cfg] candidate (up) → base; S2 [base, off, cfg]: off's no_startup keeps it off and ranks above base
    # → none into off, not into the shared top cfg (which would hide base's write in S1).
    "shared_top": ({"min_uptime": {"base": 2.0}, "startup_method": {"base": "linear", "off": "no_startup"}},
                   {"S1": ["base", "cfg"], "S2": ["base", "off", "cfg"]}, N0,
                   {"base": "min_uptime", "off": "none"}, None, True),
    # top's null method wins and reads the default (none): S is a candidate (down); the startup winner is base, but
    # the null row ranks above it (it would hide base's write) → min_downtime into top's null row (updated); S then
    # reads top's min_downtime: down enabled, unit already online, no other scenario → written.
    "null_method_above_target": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"},
                                  "minimum_time_method": {"top": None}}, {"S": ["base", "top"]}, N0,
                                 {"top": "min_downtime"}, None, True),
    # base's own null method row reads the default: candidate (down) → base, the null row is updated.
    "null_method_in_target": ({"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear"},
                               "minimum_time_method": {"base": None}}, {"S": ["base"]}, N0,
                              {"base": "min_downtime"}, None, True),
    # top's null min_uptime reads the default 3 (not base's 2, either way a time): candidate (up) → base.
    "null_time_reads_default": ({"min_uptime": {"base": 2.0, "top": None}, "startup_method": {"base": "linear"}},
                                {"S": ["base", "top"]}, {**N0, "min_uptime": 3.0}, {"base": "min_uptime"}, None,
                                True),
    # top's null startup_method reads D = linear (not no_startup): online, base's time → candidate (up); the
    # startup winner is top (its null row) → top.
    "null_startup_reads_default": ({"min_uptime": {"base": 2.0}, "startup_method": {"top": None}},
                                   {"S": ["base", "top"]}, {"startup_method": "linear"}, {"top": "min_uptime"},
                                   None, True),
    # "2" is a time for the engine (float("2") > 0).
    "str_time": ({"min_uptime": {"base": "2"}, "startup_method": {"base": "linear"}}, {"S": ["base"]}, N0,
                 {"base": "min_uptime"}, None, False),
}
DB_ROWS = [name for name, case in CASES.items() if case[5]]


def _planner_values(units: dict[str, dict]) -> dict:
    """``{param: {(unit, alternative): value}}`` as :func:`m._v83_unit_inputs` gives it (a curve as its
    ``auto_startup``)."""
    out: dict = {}
    for unit, params in units.items():
        for param, by_alt in params.items():
            for alt, value in by_alt.items():
                if isinstance(value, Map):
                    value = next(auto for curve, auto in AUTO if curve is value)
                out.setdefault(param, {})[(unit, alt)] = value
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_m1_planner(name: str) -> None:
    values, scenarios, defaults, writes, undecided, _db = CASES[name]
    plan = v83.plan_minimum_time_method(["u"], _planner_values({"u": values}), scenarios, defaults)
    assert {a: method for _u, a, method in plan.writes} == writes
    expected = () if undecided is None else (("u", undecided[0], undecided[1]),)
    assert tuple((u, r, c) for u, r, c, _n in plan.undecided) == expected


# --- databases ---------------------------------------------------------------------------------------------------


def _schema_copy(v82_schema_db_url: str, folder: Path) -> str:
    path = folder / "db.sqlite"
    shutil.copy2(v82_schema_db_url.removeprefix("sqlite:///"), path)
    return f"sqlite:///{path.resolve().as_posix()}"


def _set_default(db, name: str, value) -> None:
    pdef = db.get_item("parameter_definition", entity_class_name="unit", name=name)
    raw, typ = to_database(value)
    db.update_item("parameter_definition", id=pdef["id"], default_value=raw, default_type=typ)


def _add(db, item_type: str, **fields) -> None:
    """``db.add_item`` that fails on a refused item (it returns its error instead of raising)."""
    _item, error = db.add_item(item_type, **fields)
    assert error is None, (item_type, fields, error)


def _fill(db, units: dict[str, dict], scenarios: dict[str, list[str]], defaults: dict, *,
          inactive: dict[str, set[str]] | None = None) -> None:
    """*units* ``{unit: {param: {alternative: value}}}``, each active in every alternative (except *inactive*
    ``{unit: {alternative}}``), with an output arc to node ``elec``."""
    alts = {a for alist in scenarios.values() for a in alist}
    alts |= {a for params in units.values() for by_alt in params.values() for a in by_alt}
    for alt in sorted(alts - {"base"}):
        _add(db, "alternative", name=alt)
    for scen, alist in scenarios.items():
        _add(db, "scenario", name=scen)
        for rank, alt in enumerate(alist, start=1):
            _add(db, "scenario_alternative", scenario_name=scen, alternative_name=alt, rank=rank)
    _add(db, "entity", entity_class_name="node", name="elec")
    for alt in sorted(alts):
        _add(db, "entity_alternative", entity_class_name="node", entity_byname=("elec",), alternative_name=alt,
                    active=True)
    for unit, params in units.items():
        _add(db, "entity", entity_class_name="unit", name=unit)
        _add(db, "entity", entity_class_name="unit__outputNode", entity_byname=(unit, "elec"))
        for alt in sorted(alts):
            _add(db, "entity_alternative", entity_class_name="unit", entity_byname=(unit,), alternative_name=alt,
                        active=alt not in (inactive or {}).get(unit, set()))
        for param, by_alt in params.items():
            for alt, value in by_alt.items():
                raw, typ = to_database(value)
                _add(db, "parameter_value", entity_class_name="unit", entity_byname=(unit,),
                            parameter_definition_name=param, alternative_name=alt, value=raw, type=typ)
    for name, value in defaults.items():
        if name != "startup_method" or value != "no_startup":  # the shipped default
            _set_default(db, name, value)


def _methods(db) -> dict[tuple[str, str], object]:
    return {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"]
            for pv in db.find_parameter_values(entity_class_name="unit", parameter_definition_name="minimum_time_method")}


def _members(db) -> list[str]:
    items = sorted(db.get_items("list_value", parameter_value_list_name="minimum_time_methods"),
                   key=lambda x: x["index"])
    return [x["parsed_value"] for x in items]


def _marker(db) -> dict | None:
    for md in db.get_items("metadata"):
        if md["name"] == MARKER:
            return json.loads(md["value"])
    return None


@pytest.mark.parametrize("name", DB_ROWS)
def test_m1_part_on_a_v82_database(name: str, v82_schema_db_url: str, tmp_path: Path) -> None:
    values, scenarios, defaults, writes, undecided, _db = CASES[name]
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        _fill(db, {"u": values}, scenarios, defaults)
        own = _methods(db)
        v83._v83_minimum_time_method(db)
        assert _methods(db) == own | {("u", a): method for a, method in writes.items()}
        expected = {"since": 83, "writes": [["u", a, method] for a, method in sorted(writes.items())],
                    "undecided": [] if undecided is None else [["u", undecided[0], list(undecided[1])]]}
        assert _marker(db) == (expected if writes or undecided else None)
        assert _members(db) == ["min_uptime", "min_downtime", "both", "none"]


@pytest.mark.parametrize("name", DB_ROWS)
def test_m3_planner_model_matches_the_engine(name: str, v82_schema_db_url: str, tmp_path: Path) -> None:
    """``online`` / ``need`` of the planner equal the engine's ``_classify_process_method`` (startup ≠ no_startup)
    and the ungated time reading (``_try_param``, ``float(v) > 0``) in every scenario of the row."""
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.preprocess.params import _derived as drv

    values, scenarios, defaults, *_rest = CASES[name]
    url = _schema_copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _fill(db, {"u": values}, scenarios, defaults)
        db.commit_session("M3")
        units, vals, scen, dflt, _inactive = v83._v83_unit_inputs(db)
    for s in scenarios:
        online, need, *_r = v83._v83_unit_state("u", scen[s], vals, dflt)
        reader = SpineDbReader(url, s)
        cls = drv._classify_process_method(reader)
        engine_online = any(str(r["p"]) == "u" and r["startup"] != "no_startup" for r in cls.to_dicts())
        engine_need = set()
        for k in ("min_uptime", "min_downtime"):
            df = drv._try_param(reader, "unit", k)
            if df is not None and engine_online and any(r["name"] == "u" and float(r["value"]) > 0
                                                        for r in df.iter_rows(named=True)):
                engine_need.add(k)
        assert (online, set(need)) == (engine_online, engine_need), s


@pytest.mark.parametrize("name", DB_ROWS)
def test_m3_gated_rows_after_part_1_keep_the_needed_times(name: str, v82_schema_db_url: str, tmp_path: Path) -> None:
    """M3 after the engine gate (C rev 3 §10.3): on the database part 1 migrated, the gated ``_process_min_time``
    of ``u`` in each scenario equals the times the engine built before (the planner's ``need`` on the v82
    database), outside C §10.1: where a method of the user's resolves (§10.1 (1)) or in an undecided unit's
    candidate scenarios (§10.1 (2)) the gate may only drop times, never add one."""
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.preprocess.params import _derived as drv

    values, scenarios, defaults, *_rest = CASES[name]
    url = _schema_copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _fill(db, {"u": values}, scenarios, defaults)
        db.commit_session("M3")
        _units, vals, scen, dflt, _inactive = v83._v83_unit_inputs(db)
        before = {s: v83._v83_unit_state("u", scen[s], vals, dflt) for s in scenarios}
        v83._v83_minimum_time_method(db)
        undecided = {s for row in (_marker(db) or {}).get("undecided", []) for s in row[2]}
        try:
            db.commit_session("v83 part 1")
        except NothingToCommit:  # a row part 1 writes nothing for
            pass
    for s, state in before.items():
        reader = SpineDbReader(url, s)
        gated = {k for k in ("min_uptime", "min_downtime") if any(p == "u" for p, _v in drv._process_min_time(reader, k))}
        if state.resolves or s in undecided:
            assert gated <= set(state.need), s
        else:
            assert gated == set(state.need), s


# --- further DB checks -------------------------------------------------------------------------------------------

PLAIN = CASES["plain"][0]
FT2_OFF = CASES["ft2_off"][0]


def test_a_both_default_writes_nothing(v82_schema_db_url: str, tmp_path: Path) -> None:
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        _fill(db, {"u": PLAIN, "f": FT2_OFF}, {"S1": ["base"], "S2": ["base", "off"]}, {"minimum_time_method": "both"})
        v83._v83_minimum_time_method(db)
        assert _methods(db) == {}
        assert _marker(db) is None


def test_a_raw_null_default_is_no_default(v82_schema_db_url: str, tmp_path: Path) -> None:
    """The v3.x shape: ``default_value = b'null'``, ``default_type NULL`` (not NULL as a raw column)."""
    url = _schema_copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _fill(db, {"u": PLAIN}, {"S": ["base"]}, {})
        db.commit_session("plain")
    con = sqlite3.connect(url.removeprefix("sqlite:///"))
    try:
        con.execute("UPDATE parameter_definition SET default_value = ?, default_type = NULL "
                    "WHERE name = 'minimum_time_method'", (b"null",))
        con.commit()
    finally:
        con.close()
    with DatabaseMapping(url) as db:
        v83._v83_minimum_time_method(db)
        assert _methods(db) == {("u", "base"): "min_uptime"}


def _raw(url: str, sql: str, *args) -> None:
    con = sqlite3.connect(url.removeprefix("sqlite:///"))
    try:
        con.execute(sql, args)
        con.commit()
    finally:
        con.close()


def test_a_list_without_none_gets_it_and_off_reads_none(v82_schema_db_url: str, tmp_path: Path) -> None:
    """The v3.33.0 shape: [min_uptime, min_downtime, both]."""
    url = _schema_copy(v82_schema_db_url, tmp_path)
    _raw(url, "DELETE FROM list_value WHERE value = ? AND parameter_value_list_id = (SELECT id FROM "
              "parameter_value_list WHERE name = 'minimum_time_methods')", b'"none"')
    with DatabaseMapping(url) as db:
        assert _members(db) == ["min_uptime", "min_downtime", "both"]
        _fill(db, {"u": FT2_OFF}, {"S1": ["base"], "S2": ["base", "off"]}, {})
        v83._v83_minimum_time_method(db)
        assert _members(db) == ["min_uptime", "min_downtime", "both", "none"]
        assert _methods(db) == {("u", "base"): "min_downtime", ("u", "off"): "none"}
        db.commit_session("v83 part 1")
    with DatabaseMapping(url) as db:
        assert _methods(db) == {("u", "base"): "min_downtime", ("u", "off"): "none"}


def test_a_corrupted_list_gets_min_uptime_back(v82_schema_db_url: str, tmp_path: Path) -> None:
    """Item D: the first member overwritten with none, [none, min_downtime, both].  A value that was min_uptime now
    reads none and stays (explicit); min_uptime is appended and the plain unit gets it."""
    url = _schema_copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _fill(db, {"u": PLAIN, "old": PLAIN | {"minimum_time_method": {"base": "min_uptime"}}}, {"S": ["base"]}, {})
        db.commit_session("before the corruption")
    lst = "(SELECT id FROM parameter_value_list WHERE name = 'minimum_time_methods')"
    _raw(url, f"DELETE FROM list_value WHERE value = ? AND parameter_value_list_id = {lst}", b'"none"')
    _raw(url, f"UPDATE list_value SET value = ? WHERE value = ? AND parameter_value_list_id = {lst}",
         b'"none"', b'"min_uptime"')
    with DatabaseMapping(url) as db:
        assert _members(db) == ["none", "min_downtime", "both"]
        assert _methods(db) == {("old", "base"): "none"}
        v83._v83_minimum_time_method(db)
        assert _members(db) == ["none", "min_downtime", "both", "min_uptime"]
        assert _methods(db) == {("old", "base"): "none", ("u", "base"): "min_uptime"}


def test_a_scenario_where_the_unit_is_inactive_is_ignored(v82_schema_db_url: str, tmp_path: Path) -> None:
    """In S2 [base, dead] u is switched off; counted, its no_startup in dead would get a none."""
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        _fill(db, {"u": {"min_downtime": {"base": 2.0}, "startup_method": {"base": "linear", "dead": "no_startup"}}},
              {"S1": ["base"], "S2": ["base", "dead"]}, {}, inactive={"u": {"dead"}})
        v83._v83_minimum_time_method(db)
        assert _methods(db) == {("u", "base"): "min_downtime"}


def test_no_scenario_reads_all_alternatives_as_one(v82_schema_db_url: str, tmp_path: Path) -> None:
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        _fill(db, {"u": PLAIN}, {}, {})
        v83._v83_minimum_time_method(db)
        assert _methods(db) == {("u", "base"): "min_uptime"}


def test_marker_merges_writes_and_replaces_undecided(v82_schema_db_url: str, tmp_path: Path) -> None:
    masking = CASES["masking_conflict"][0]
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        _fill(db, {"u": PLAIN, "mc": masking}, {"S1": ["E", "base"], "S2": ["base"]}, {})
        v77._v77_set_marker(db, v83.V83_MIN_TIME_MARKER, json.dumps(
            {"since": 83, "writes": [["x", "A", "both"]], "undecided": [["y", "gone", ["S9"]]]}))
        v83._v83_minimum_time_method(db)
        # u: S1 [E, base] and S2 [base] both candidates (u has no method) → base: min_uptime.
        assert _marker(db) == {
            "since": 83, "writes": [["x", "A", "both"], ["u", "base", "min_uptime"]],
            "undecided": [["mc", "a value in alternative base would replace your minimum_time_method in alternative "
                                 "E in scenario S1", ["S2"]]]}


def test_rerun_writes_nothing_and_logs_already_applied(v82_schema_db_url: str, tmp_path: Path,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    """In the session (a rerun of the part) and after the commit (a kill after the v83 commit, then a rerun)."""
    url = _schema_copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _fill(db, {"u": FT2_OFF, "mc": CASES["masking_conflict"][0]},
              {"S1": ["base"], "S2": ["base", "off"], "S3": ["E", "base"]}, {})
        with caplog.at_level(logging.INFO):
            v83._v83_minimum_time_method(db)
        assert not [r for r in caplog.records if r.getMessage().startswith(m.ALREADY_APPLIED_PREFIX)]
        methods, marker = _methods(db), _marker(db)
        assert methods[("u", "base")] == "min_downtime" and marker["undecided"]
        caplog.clear()
        with caplog.at_level(logging.INFO):
            v83._v83_minimum_time_method(db)
        assert (_methods(db), _marker(db)) == (methods, marker)
        assert [r.getMessage() for r in caplog.records if r.getMessage().startswith(m.ALREADY_APPLIED_PREFIX)] == [
            f"{m.ALREADY_APPLIED_PREFIX} v83 minimum_time_method: already written"]
        db.commit_session("v83")
    caplog.clear()
    with DatabaseMapping(url) as db:
        with caplog.at_level(logging.INFO):
            v83._v83_minimum_time_method(db)
        assert (_methods(db), _marker(db)) == (methods, marker)
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")
    applied = [r for r in caplog.records if r.getMessage().startswith(m.ALREADY_APPLIED_PREFIX)]
    assert [r.getMessage() for r in applied] == [f"{m.ALREADY_APPLIED_PREFIX} v83 minimum_time_method: already written"]


def test_guard_stops_on_a_changed_member_without_a_commit(v82_schema_db_url: str, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    url = _schema_copy(v82_schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        _fill(db, {"u": PLAIN}, {"S": ["base"]}, {})
        db.commit_session("plain")

    def overwrite_first(db, name, members):
        first = min(db.get_items("list_value", parameter_value_list_name=name), key=lambda x: x["index"])
        raw, typ = to_database("overwritten")  # the old spinedb_api bug rewrote a first member
        db.update_item("list_value", id=first["id"], value=raw, type=typ)

    patch_migration(monkeypatch, "_append_list_members", overwrite_first)
    with DatabaseMapping(url) as db:
        with pytest.raises(m.FlexToolMigrationError,
                           match=r"A version 83 upgrade step changed an existing member of the value list "
                                 r"minimum_time_methods \('min_uptime'\); the upgrade stops"):
            v83._v83_minimum_time_method(db)
    with DatabaseMapping(url) as db:
        assert _members(db) == ["min_uptime", "min_downtime", "both", "none"]
        assert _methods(db) == {}
        assert _marker(db) is None


def test_guard_removal_form(v82_schema_db_url: str, tmp_path: Path) -> None:
    """A planned removal passes; a planned removal that did not happen stops."""
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        before = common._list_members(db, ["minimum_time_methods"])
        none_id = next(i for i, (_l, v, _t) in before.items() if v == b'"none"')
        with pytest.raises(m.FlexToolMigrationError, match=r"\('none' \(its planned removal did not happen\)\)"):
            v83._v83_check_members(db, "minimum_time_methods", before, removed={none_id})
        item = next(x for x in db.get_items("list_value", parameter_value_list_name="minimum_time_methods")
                    if x["parsed_value"] == "none")
        db.remove_item("list_value", item["id"])
        v83._v83_check_members(db, "minimum_time_methods", before, removed={none_id})
        with pytest.raises(m.FlexToolMigrationError, match="'none'"):
            v83._v83_check_members(db, "minimum_time_methods", before)


def test_undecided_unit_is_printed(v82_schema_db_url: str, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    with DatabaseMapping(_schema_copy(v82_schema_db_url, tmp_path)) as db:
        _fill(db, {"mc": CASES["masking_conflict"][0]}, {"S1": ["E", "base"], "S2": ["base"]}, {})
        v83._v83_minimum_time_method(db)
    assert ("v83: minimum_time_method not written for unit mc: a value in alternative base would replace your "
            "minimum_time_method in alternative E in scenario S1. In scenario S2 its min_downtime no longer applies "
            "(results change). Set minimum_time_method yourself (both, min_uptime or min_downtime) where it should "
            "apply.") in capsys.readouterr().out
