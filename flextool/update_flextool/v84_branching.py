"""Data version 84 (inside FlexTool 4.7.0): ``solve.invest_solve_branching_method`` replaces
``stochastic_invest_method``, ``shared_invest_periods`` and ``shared_operation_periods`` (item 1024).  The step
:func:`_migrate_v84_schema_batch` is run by ``db_migration``, which imports this module where it uses it."""
from __future__ import annotations

import json

from spinedb_api import Map, to_database

from flextool.update_flextool.db_migration import (
    _already_applied,
    _append_list_members,
    _commit_step,
    _ensure_parameter_group,
    _pdef_or_none,
    _remove_items_bounded,
    _replace_text_pairs,
    _v77_set_marker,
    _v83_dispatch_realization_texts,
    _v83_write_values,
)


#: The value list and the members of ``solve.invest_solve_branching_method``, default first (the engine's
#: ``_solve_config.INVEST_SOLVE_BRANCHING_METHODS``; pinned by ``tests/update_flextool/test_v84_branching.py``).
_V84_LIST = "invest_solve_branching_methods"
_V84_METHODS: tuple[str, ...] = (
    "operations_only", "investments_after_reveal_period", "investments_at_reveal_period")
#: The parameters (and the value list) v84 replaces.
_V84_OLD_PARAMETERS: tuple[str, ...] = (
    "stochastic_invest_method", "shared_invest_periods", "shared_operation_periods")
_V84_OLD_LIST = "stochastic_invest_methods"
#: The behaviour-change marker of v84 (W-0637).
V84_MARKER = "stochastic_invest_controls_rewritten"
_V84_DESCRIPTION = (
    "[choice] Investment solves only (not rolling, with invest_periods or realized_invest_periods, and "
    "stochastic_branches): which decisions all forecast branches share. The reveal period is the period where the "
    "forecasts start. operations_only (default): the investments of every period are one plan for all branches; "
    "each branch operates on its own from the reveal period on. investments_after_reveal_period: the investments "
    "up to and including the reveal period are shared; each branch invests on its own in the later periods, which "
    "cannot be realized. investments_at_reveal_period: the investments before the reveal period are shared; each "
    "branch invests on its own from the reveal period on, and those periods cannot be realized.")


def _v84_definitions() -> dict[tuple[str, str], dict]:
    """The definition v84 adds, with its fields (the default is the list member ``operations_only``)."""
    return {("solve", "invest_solve_branching_method"): {
        "parameter_type_list": ("str",), "parameter_value_list_name": _V84_LIST,
        "parameter_group_name": "solve_advanced", "description": _V84_DESCRIPTION}}


def _v84_dispatch_realization_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the ``dispatch_realization_method`` description as v83 wrote it and its v84
    text (what an investment solve reads)."""
    (key, (_old, v83)), = _v83_dispatch_realization_texts().items()
    return {key: (v83, v83 + (
        " An investment solve (not rolling, with invest_periods or realized_invest_periods) does not read it: its "
        "forecasts are solved (invest_solve_branching_method), and separate_dispatch there is ignored with a "
        "warning."))}


def _v84_definitions_part(db, *, update: bool = True) -> None:
    """Part 1 of v84: the value list ``invest_solve_branching_methods`` (members appended when missing) and the
    definition ``solve.invest_solve_branching_method`` (default ``operations_only``, group ``solve_advanced``),
    or (``update``) its fields.  ``update=False`` only adds what is missing.  No value is written.  Idempotent."""
    if not db.get_item("entity_class", name="solve"):
        return
    if not db.get_item("parameter_value_list", name=_V84_LIST):
        db.add_parameter_value_list(name=_V84_LIST)
    _append_list_members(db, _V84_LIST, _V84_METHODS)
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    for (cls, name), fields in _v84_definitions().items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            val, typ = to_database(_V84_METHODS[0])
            db.add_parameter_definition(entity_class_name=cls, name=name, default_value=val, default_type=typ,
                                        **fields)
        elif update and any((tuple(pdef[k] or ()) if k == "parameter_type_list" else (pdef[k] or "")) != v
                            for k, v in fields.items()):
            db.update_parameter_definition(id=pdef["id"], **fields)


def _v84_branching(db) -> None:
    """Part 2 of v84.  A solve is *eligible* when its own rows, in any alternative, can make it a stochastic
    investment solve: a ``solve_mode`` row other than ``rolling_window`` (or none), a non-empty ``invest_periods`` or
    ``realized_invest_periods`` row and a ``stochastic_branches`` row with an entry.  On an eligible solve with a
    ``stochastic_invest_method`` ``recourse`` row, each ``recourse`` row becomes ``investments_after_reveal_period``
    and each other row ``operations_only``, in its own alternative (the alternatives layer as before; where the solve
    reads differently in a scenario, the run refuses it there).  Nothing else is written.  The solves with a row of an
    old parameter are printed and listed in the marker ``stochastic_invest_controls_rewritten`` (``{"since": 84,
    "solves": [...]}``, W-0637); the old parameters and the list ``stochastic_invest_methods`` are removed with their
    values.  Without the old definitions (a rerun) nothing is written.  Idempotent; no commit."""
    if not db.get_item("entity_class", name="solve"):
        return
    old = [pdef for name in _V84_OLD_PARAMETERS if (pdef := _pdef_or_none(db, "solve", name)) is not None]
    if not old:
        _already_applied(84, "invest_solve_branching_method: already converted")
    else:
        rows: dict[str, dict] = {}
        for name in ("solve_mode", "invest_periods", "realized_invest_periods", "stochastic_branches",
                     *_V84_OLD_PARAMETERS):
            for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name=name):
                rows.setdefault((name, pv["entity_byname"][0]), {})[pv["alternative_name"]] = pv["parsed_value"]

        def of(solve: str, *names: str) -> list:
            return [v for name in names for v in rows.get((name, solve), {}).values()]

        def text(value) -> str:
            return str(value or "").strip().lower()

        writes = []
        for (name, solve), methods in sorted(rows.items()):
            modes = of(solve, "solve_mode")
            if (name == "stochastic_invest_method" and "recourse" in map(text, methods.values())
                    and (not modes or any(text(m) != "rolling_window" for m in modes))
                    and any(v is not None and getattr(v, "values", True)
                            for v in of(solve, "invest_periods", "realized_invest_periods"))
                    and any(isinstance(b, Map) and b.indexes for b in of(solve, "stochastic_branches"))):
                writes += [(solve, alt, _V84_METHODS[1] if text(v) == "recourse" else _V84_METHODS[0])
                           for alt, v in sorted(methods.items())]
        _v83_write_values(db, "solve", "invest_solve_branching_method", writes)
        listed = sorted({solve for name, solve in rows if name in _V84_OLD_PARAMETERS})
        if listed:
            print("v84: stochastic_invest_method, shared_invest_periods and shared_operation_periods are replaced by "
                  "invest_solve_branching_method (recourse became investments_after_reveal_period on a non-rolling "
                  "solve with investments and stochastic_branches); check invest_solve_branching_method on solve "
                  f"{', '.join(listed)}")
            _v77_set_marker(db, V84_MARKER, json.dumps({"since": 84, "solves": listed}))
        _remove_items_bounded(db, "parameter_definition", [pdef["id"] for pdef in old],
                              "v84: removed solve.stochastic_invest_method, shared_invest_periods and "
                              "shared_operation_periods")
    old_list = db.get_item("parameter_value_list", name=_V84_OLD_LIST)
    if old_list and not any(pd["parameter_value_list_name"] == _V84_OLD_LIST
                            for pd in db.get_items("parameter_definition")):
        db.remove_item("parameter_value_list", old_list["id"])


def _migrate_v84_schema_batch(db) -> None:
    """All schema changes of database version 84 (v83 -> v84), in one step committed at the end; each part is
    idempotent: the definition (:func:`_v84_definitions_part`), the old parameters replaced (:func:`_v84_branching`)
    and the ``dispatch_realization_method`` description (an investment solve does not read it; by text alone)."""
    _v84_definitions_part(db)
    _v84_branching(db)
    _replace_text_pairs(db, _v84_dispatch_realization_texts())
    _commit_step(db, "v84: invest_solve_branching_method replaces stochastic_invest_method, shared_invest_periods "
                     "and shared_operation_periods")
