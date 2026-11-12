"""Workbook round-trip of the value shapes database version 77 allows.

Builds a database from the live master template (never a checked-in
.sqlite), writes one value of every newly allowed shape, exports it to xlsx,
imports the workbook into a fresh template and compares the values.  Also
checks that the behaviour-change markers (database metadata) survive the
round-trip, as the xlsx "update version" flow relies on it.
"""
from __future__ import annotations

import pytest
from spinedb_api import Array, DatabaseMapping, Map, from_database, to_database

from flextool._resources import package_data_path
from flextool.export_to_tabular.export_to_excel import export_to_excel
from flextool.process_inputs.read_self_describing_excel import (
    read_behaviour_markers,
    read_self_describing_excel,
)
from flextool.process_inputs.write_self_describing_to_db import (
    restore_behaviour_markers,
    write_sheet_data_to_db,
)
from flextool.update_flextool.initialize_database import initialize_database

MASTER_TEMPLATE = package_data_path("schemas/spinedb_schema.json")
MARKER = ("flextool_behaviour_change:user_constraint_prebuilt_nets_retirements", "77")


def _period_time(values: dict[str, dict[str, float]]) -> Map:
    return Map(
        indexes=list(values),
        values=[Map(indexes=list(v), values=list(v.values()), index_name="time") for v in values.values()],
        index_name="period",
    )


def _period(values: dict[str, float]) -> Map:
    return Map(indexes=list(values), values=list(values.values()), index_name="period")


def _time(values: dict[str, float]) -> Map:
    return Map(indexes=list(values), values=list(values.values()), index_name="time")


# (entity_class, entity_byname, parameter, value)
VALUES = [
    ("profile", ("p_const",), "profile", 0.4),
    ("profile", ("p_time",), "profile", _time({"t01": 0.1, "t02": 0.2})),
    ("profile", ("p_2d",), "profile", _period_time({"p2030": {"t01": 0.3, "t02": 0.4},
                                                     "p2040": {"t01": 0.5, "t02": 0.6}})),
    ("unit", ("u1",), "efficiency", _period_time({"p2030": {"t01": 0.4, "t02": 0.45},
                                                  "p2040": {"t01": 0.5, "t02": 0.55}})),
    ("unit", ("u1",), "efficiency_at_min_load", _period_time({"p2030": {"t01": 0.3, "t02": 0.35}})),
    ("connection", ("c_line",), "efficiency", _period_time({"p2030": {"t01": 0.9, "t02": 0.95}})),
    ("node", ("n1",), "self_discharge_loss", _period_time({"p2030": {"t01": 0.01, "t02": 0.02}})),
    ("unit", ("u1",), "invest_max_period", 250.0),
    ("unit", ("u1",), "startup_cost", _period({"p2030": 10.0, "p2040": 12.0})),
    ("node", ("n1",), "penalty_method", "off"),
    ("node", ("n1",), "energy_margin_adder", _period_time({"p2030": {"t01": 1.0, "t02": 2.0}})),
    ("solve", ("s_empty",), "shared_operation_periods", Array([], value_type=str)),
    ("solve", ("s_one",), "shared_operation_periods", Array(["p2030"])),
    ("group", ("g_retire",), "retire_max_period", 75.0),
    ("group", ("g_retire",), "invest_method", "retire_period"),
    ("group", ("g_off",), "invest_method", "not_active"),
    ("constraint", ("c1",), "constant", _period({"p2030": 5.0, "p2040": 6.0})),
    # v79: the group inertia and non-synchronous limits take time and period + time maps.
    ("group", ("g_inertia",), "inertia_limit", _time({"t01": 200.0, "t02": 800.0})),
    ("group", ("g_inertia",), "non_synchronous_limit", _period_time({"p2030": {"t01": 0.2, "t02": 0.6}})),
]


@pytest.fixture(scope="module")
def round_tripped(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, list, str]:
    out = tmp_path_factory.mktemp("v77_rt")
    src = out / "src.sqlite"
    initialize_database(str(MASTER_TEMPLATE), str(src))
    src_url = f"sqlite:///{src}"
    with DatabaseMapping(src_url) as db:
        for cls, byname, _, _ in VALUES:
            if not db.get_item("entity", entity_class_name=cls, entity_byname=byname):
                db.add_entity(entity_class_name=cls, entity_byname=byname)
        for cls, byname, name, value in VALUES:
            v, t = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=name,
                                   alternative_name="base", value=v, type=t)
        db.add_item("metadata", name=MARKER[0], value=MARKER[1])
        db.commit_session("v77 shapes")

    xlsx = out / "rt.xlsx"
    export_to_excel(src_url, str(xlsx), include_advanced=True)
    rt = out / "rt.sqlite"
    initialize_database(str(MASTER_TEMPLATE), str(rt))
    rt_url = f"sqlite:///{rt}"
    sheets = read_self_describing_excel(str(xlsx), skip_sheets={"navigate", "version"})
    write_sheet_data_to_db(sheets, rt_url, purge_first=True, keep_entities=True)
    restore_behaviour_markers(rt_url, read_behaviour_markers(str(xlsx)))

    result: dict = {}
    with DatabaseMapping(rt_url) as db:
        for p in db.get_parameter_value_items():
            result[(p["entity_class_name"], tuple(p["entity_byname"]), p["parameter_definition_name"],
                    p["alternative_name"])] = from_database(p["value"], p["type"])
        markers = [(m["name"], m["value"]) for m in db.get_items("metadata")]
    return result, markers, str(xlsx)


def _flatten(value) -> object:
    """Comparable form: nested (index, value) tuples for Maps, a list for Arrays."""
    if isinstance(value, Map):
        return tuple((str(i), _flatten(v)) for i, v in zip(value.indexes, value.values))
    if isinstance(value, Array):
        return ("array", tuple(value.values))
    return value


@pytest.mark.parametrize("cls,byname,name,value", VALUES, ids=[f"{v[0]}.{v[2]}.{v[1][0]}" for v in VALUES])
def test_value_survives_workbook_round_trip(round_tripped, cls, byname, name, value) -> None:
    result, _, _ = round_tripped
    key = (cls, byname, name, "base")
    assert key in result, f"{cls}.{name} of {byname[0]} was lost on the workbook round-trip"
    assert _flatten(result[key]) == _flatten(value)


def test_behaviour_markers_survive_workbook_round_trip(round_tripped) -> None:
    _, markers, _ = round_tripped
    assert MARKER in markers


def test_group_limits_by_time_are_on_the_time_sheets(round_tripped) -> None:
    """inertia_limit's time map is written to the group time-series sheet (group_t), non_synchronous_limit's
    period + time map to the group (period, time) sheet (group_s…), as unit.efficiency's; neither is left on the
    period-only sheets.  The round-trip itself is test_value_survives_workbook_round_trip."""
    import openpyxl

    _, _, xlsx = round_tripped
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    found: dict[str, set[str]] = {}
    for ws in wb.worksheets:
        if ws.title.startswith("group_"):
            for row in ws.iter_rows(values_only=True):
                for cell in row:
                    if cell in ("inertia_limit", "non_synchronous_limit"):
                        found.setdefault(cell, set()).add(ws.title)
    wb.close()
    assert "group_t" in found.get("inertia_limit", set()), found
    assert any(t.startswith("group_s") for t in found.get("non_synchronous_limit", set())), found


def test_restored_markers_keep_one_item_per_name(tmp_path) -> None:
    """The import's own upgrade may already have written a marker the workbook also carries, with another value.
    Restoring matches by name: one item per name, the JSON rewrite lists merged (database items first), a plain
    value taken from the workbook."""
    import json

    rt = tmp_path / "target.sqlite"
    initialize_database(str(MASTER_TEMPLATE), str(rt))
    url = f"sqlite:///{rt}"
    rewritten = "flextool_behaviour_change:storage_fix_method_rewritten"
    stored = {"since": 77, "rewrites": [["ev", "A", "fix_start", "fix_nothing"]], "undecided": []}
    restored = {"since": 77,
                "rewrites": [["ev", "A", "fix_start", "fix_nothing"], ["bat", "B", "fix_end", "fix_nothing"]],
                "undecided": [["n", "C", "fix_start", ["S1", "S2"]]]}
    with DatabaseMapping(url) as db:
        db.add_item("metadata", name=rewritten, value=json.dumps(stored))
        db.add_item("metadata", name=MARKER[0], value="76")
        db.commit_session("markers of the import's own upgrade")
    restore_behaviour_markers(url, [(rewritten, json.dumps(restored)), MARKER])
    with DatabaseMapping(url) as db:
        items = [(m["name"], m["value"]) for m in db.get_items("metadata")
                 if m["name"] in (rewritten, MARKER[0])]
    assert sorted(name for name, _ in items) == sorted([rewritten, MARKER[0]])
    values = dict(items)
    assert values[MARKER[0]] == MARKER[1]
    merged = json.loads(values[rewritten])
    assert merged["rewrites"] == [["ev", "A", "fix_start", "fix_nothing"], ["bat", "B", "fix_end", "fix_nothing"]]
    assert merged["undecided"] == [["n", "C", "fix_start", ["S1", "S2"]]]
