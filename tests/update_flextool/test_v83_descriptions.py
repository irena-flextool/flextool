"""Data version 83, part 4 (item 278): descriptions replaced by text alone and the unused ``solver`` value list
deleted (plan §3, §7 V4).

Databases: the v82 schema (``v82_schema_db_url``: the bundled v25 template migrated to 82) and the current schema
(``schema_db_url``, at 83), copied into ``tmp_path``.  Expected values: the texts of ``spinedb_schema.json`` (the
shipped schema the plan names as the target), and two texts written out here from the plan: the ``group.co2_method``
member list with ``none`` in place of ``no_method`` (the member name since the list rename), and the
``virtual_unitsize`` sentence naming the ``startup_method`` members that commit whole units (owner decision D5).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m

from tests.update_flextool.test_v83_schema_batch import (
    _assert_second_refresh_commits_nothing, _copy, _definition, _migrate, _path, _raw, _shipped_texts, _values)

NEW_CO2_METHOD = ("[choice] Choice of the CO2 method: none, price, period, total, price_period, price_total, "
                  "period_total, price_period_total")
OLD_CO2_METHOD = NEW_CO2_METHOD.replace(": none,", ": no_method,")
NEW_UNIT_SIZE = ("[MW] Size of one unit. With it, existing is the number of units. Integer startups (startup_method "
                 "integer or integer_fix_in_uc) commit whole units of this size. Investments and retirements are "
                 "continuous; their limits are in MW. No value: the existing capacity, or 1000 MW without it.")
OLD_UNIT_SIZE = NEW_UNIT_SIZE.replace("Integer startups (startup_method integer or integer_fix_in_uc)",
                                      "Binary startups")
OLD_NODE_AVAILABILITY = "[-] e.g. 0.9 means 90%. Fraction of capacity available for storage. Constant or time."


def _add_solver_list(url: str) -> None:
    """The unused ``solver`` value list a template older than v52 shipped (members glpsol, highs, cplex)."""
    with DatabaseMapping(url) as db:
        db.add_parameter_value_list(name="solver")
        for i, member in enumerate(("glpsol", "highs", "cplex")):
            raw, typ = to_database(member)
            db.add_list_value(parameter_value_list_name="solver", value=raw, type=typ, index=i)
        db.commit_session("solver list")


def test_v25_to_83_descriptions_are_the_shipped_ones(v82_schema_db_url: str, tmp_path: Path) -> None:
    """The v25 template at 82 (with the unused ``solver`` list) migrated to 83: every description is the
    ``spinedb_schema.json`` text, the ``solver`` list is gone and ``solve.solver`` stays on ``solvers``; a text the
    user edited is kept."""
    url = _copy(v82_schema_db_url, tmp_path)
    _add_solver_list(url)
    with DatabaseMapping(url) as db:
        assert _definition(db, "unit", "virtual_unitsize")["description"] == OLD_UNIT_SIZE
        assert _definition(db, "node", "availability")["description"] == OLD_NODE_AVAILABILITY
        pdef = _definition(db, "node", "penalty_up")
        db.update_parameter_definition(id=pdef["id"], description="my own text")
        db.commit_session("user text")
    _migrate(url)
    shipped = _shipped_texts()
    with DatabaseMapping(url) as db:
        held = {(d["entity_class_name"], d["name"]): d["description"] or ""
                for d in db.get_items("parameter_definition")}
        lists = {x["name"] for x in db.get_items("parameter_value_list")}
        solver = _definition(db, "solve", "solver")["parameter_value_list_name"]
    diffs = {key for key, text in held.items() if key in shipped and text != (shipped[key] or "")}
    assert diffs == {("node", "penalty_up")}
    assert held[("node", "penalty_up")] == "my own text"
    assert held[("unit", "virtual_unitsize")] == held[("connection", "virtual_unitsize")] == NEW_UNIT_SIZE
    assert held[("group", "co2_method")] == NEW_CO2_METHOD
    assert "solver" not in lists and solver == "solvers"
    assert m._current_version_ro(_path(url), None) == FLEXTOOL_DB_VERSION


@pytest.mark.parametrize("default", ["null", "price"])
def test_co2_method_text_whatever_the_default(v82_schema_db_url: str, tmp_path: Path, default: str) -> None:
    """``group.co2_method`` with a ``NULL`` default and with a user-set default both get the new text (CR-1 B2);
    the default stays."""
    url = _copy(v82_schema_db_url, tmp_path)
    if default == "null":
        _raw(url, "UPDATE parameter_definition SET default_value = NULL, default_type = NULL WHERE name = "
                  "'co2_method' AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'group')")
    else:
        with DatabaseMapping(url) as db:
            raw, typ = to_database(default)
            db.update_item("parameter_definition", id=_definition(db, "group", "co2_method")["id"],
                           default_value=raw, default_type=typ)
            db.commit_session("user default")
    with DatabaseMapping(url) as db:
        assert _definition(db, "group", "co2_method")["description"] == OLD_CO2_METHOD
    _migrate(url)
    with DatabaseMapping(url) as db:
        pdef = _definition(db, "group", "co2_method")
        assert pdef["description"] == NEW_CO2_METHOD
        assert pdef["parsed_value"] == (None if default == "null" else default)


def test_old_texts_at_83_are_refreshed_once(schema_db_url: str, tmp_path: Path, monkeypatch) -> None:
    """A database at 83 with old part-4 texts (``node.availability``; ``unit.virtual_unitsize``) is not current;
    one open replaces both (the values are unchanged), and a second open is current, takes no lock and commits
    nothing."""
    url = _copy(schema_db_url, tmp_path)
    with DatabaseMapping(url) as db:
        for (cls, name), text in ((("node", "availability"), OLD_NODE_AVAILABILITY),
                                  (("unit", "virtual_unitsize"), OLD_UNIT_SIZE)):
            db.update_parameter_definition(id=_definition(db, cls, name)["id"], description=text)
        db.commit_session("old texts")
        values_before = _values(db)
    path = _path(url)
    assert m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) is None
    _migrate(url)
    shipped = _shipped_texts()
    with DatabaseMapping(url) as db:
        assert _definition(db, "node", "availability")["description"] == shipped[("node", "availability")]
        assert _definition(db, "unit", "virtual_unitsize")["description"] == NEW_UNIT_SIZE
    assert not m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION

    def no_lock(*_args, **_kwargs):
        raise AssertionError("a current database took the migration lock")

    monkeypatch.setattr(importlib.import_module("flextool.update_flextool.migrate_before_run"), "migration_lock",
                        no_lock)
    _assert_second_refresh_commits_nothing(url, values_before)


#: The descriptions that databases made from a release's shipped database or template (FlexTool 3.22.0-3.43.0) hold
#: at 82, read from those databases (``tools/migration_sweep.py``; ``""``: no description written), written out here.
RELEASE_OLD_TEXTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("unit", "minimum_time_method"): (
        "[choice] Not functional yet. Choice between minimum up- and downtimes (<empty>, min_downtime, min_uptime, "
        "both).",),
    ("node", "penalty_up"): ("[CUR/MW] Penalty cost for decreasing consumption in the node. Constant, Period or Time.",),
    ("node", "penalty_down"): (
        "[CUR/MW] Penalty cost for increasing consumption in the node. Constant, Period or Time.",),
    ("group", "penalty_inertia"): ("[CUR/MWs per hour] Penalty for violating the inertia constraint. Constant or period.",),
    ("connection", "reactance"): ("",),
    ("connection", "transfer_method"): ("[choice] Choice of transfer method",),
    ("group", "transfer_method"): (
        "[choice] Override transfer_method for all connections within this nodeGroup. Options: "
        "use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, "
        "variable_cost_only, dc_power_flow_with_angles. When set to dc_power_flow_with_angles, connections between "
        "member nodes use B-theta DC power flow (requires reactance parameter on connections).",),
    ("commodity", "price_ladder_cumulative"): (
        "[CUR/MWh; MWh]",
        "[CUR/MWh; MWh] Stepped supply curve for price_method='price_ladder_cumulative'.  Structure: Map(tier -> "
        "{price, quantity}).  1-based integer tier index.  quantity=inf marks an unbounded tail tier.  "
        "Period-agnostic — the cap is a single total across the full model horizon."),
}


def _write_texts(url: str, texts: dict[tuple[str, str], str]) -> None:
    with DatabaseMapping(url) as db:
        for (cls, name), text in texts.items():
            db.update_parameter_definition(id=_definition(db, cls, name)["id"], description=text or None)
        db.commit_session("old texts")


@pytest.mark.parametrize("which", [0, -1], ids=["first", "last"])
def test_release_texts_at_82_become_the_shipped_ones(v82_schema_db_url: str, tmp_path: Path, which: int) -> None:
    """Each old text a release database holds at 82 (the first and the last where a definition has two) is replaced
    by the shipped text at 83; a second refresh commits nothing."""
    url = _copy(v82_schema_db_url, tmp_path)
    _write_texts(url, {key: olds[which] for key, olds in RELEASE_OLD_TEXTS.items()})
    _migrate(url)
    shipped = _shipped_texts()
    with DatabaseMapping(url) as db:
        held = {key: _definition(db, *key)["description"] for key in RELEASE_OLD_TEXTS}
        values_before = _values(db)
    assert held == {key: shipped[key] for key in RELEASE_OLD_TEXTS}
    _assert_second_refresh_commits_nothing(url, values_before)


def test_release_text_edited_by_the_user_is_kept(v82_schema_db_url: str, tmp_path: Path) -> None:
    """A ``minimum_time_method`` description the user edited from the old release text is kept at 83."""
    url = _copy(v82_schema_db_url, tmp_path)
    edited = RELEASE_OLD_TEXTS[("unit", "minimum_time_method")][0] + " Mine."
    _write_texts(url, {("unit", "minimum_time_method"): edited})
    _migrate(url)
    with DatabaseMapping(url) as db:
        assert _definition(db, "unit", "minimum_time_method")["description"] == edited


def test_release_text_at_83_is_refreshed(schema_db_url: str, tmp_path: Path) -> None:
    """A database at 83 holding an old release text (``unit.minimum_time_method``; ``connection.reactance`` with no
    description) is not current; one open writes the shipped texts and it is current."""
    url = _copy(schema_db_url, tmp_path)
    _write_texts(url, {("unit", "minimum_time_method"): RELEASE_OLD_TEXTS[("unit", "minimum_time_method")][0],
                       ("connection", "reactance"): ""})
    path = _path(url)
    assert m._head_defaults_stale_ro(path)
    _migrate(url)
    shipped = _shipped_texts()
    with DatabaseMapping(url) as db:
        assert _definition(db, "unit", "minimum_time_method")["description"] == shipped[("unit", "minimum_time_method")]
        assert _definition(db, "connection", "reactance")["description"] == shipped[("connection", "reactance")]
    assert not m._head_defaults_stale_ro(path)
    assert m._current_version_ro(path, None) == FLEXTOOL_DB_VERSION
