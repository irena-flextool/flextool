"""Tests for the calibrator's per-scenario calibration-alternative writer.

These build a tmp SpineDB from the JSON fixture (never a checked-in
``.sqlite``) and exercise :func:`flextool.calibrate._db_alt.write_calib_alt`
against real ``import_data`` semantics: the alternative lands at the TOP
rank of the scenario's stack, its value WINS when the scenario filter is
materialised, a changed adder UPDATEs the single row in place (no
duplicate), and an identical re-write is a no-op that does not crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.calibrate._db_alt import (  # noqa: E402
    calib_alt_name,
    write_calib_alt,
)

FIXTURE_JSON = _TESTS_DIR / "fixtures" / "tests.json"
SCENARIO = "multi_fullYear_battery_nested_24h_invest_one_solve"
NODE = "west"  # a real node active in the target scenario


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """A tmp DB built from the JSON fixture; returns the sqlite URL."""
    return json_to_db(FIXTURE_JSON, tmp_path / "calib.sqlite")


# --- read-back helpers ------------------------------------------------------


def _materialized(url: str, scenario: str, node: str, param: str):
    """Value of *param* on *node* as MATERIALISED under the scenario filter.

    Applies the real ``scenario_filter`` and exports the filtered values,
    so this is exactly the value the engine's scenario read would see —
    i.e. it reflects alternative ranking (the winner of the stack).
    """
    from spinedb_api import DatabaseMapping, export_data
    from spinedb_api.filters.scenario_filter import (
        apply_scenario_filter_to_subqueries,
    )

    with DatabaseMapping(url) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        data = export_data(db)
        for cls, ent, p, *_rest in data.get("parameter_values", []):
            name = ent[0] if isinstance(ent, (list, tuple)) else ent
            if p == param and name == node:
                return _rest[0]  # value
    return None


def _ranked_alts(url: str, scenario: str) -> list[tuple[str, int]]:
    """The scenario's ``(alt_name, rank)`` links, ascending by rank."""
    from spinedb_api import DatabaseMapping

    with DatabaseMapping(url) as db:
        scen_id = next(
            s.id for s in db.query(db.scenario_sq).all() if s.name == scenario
        )
        alts = {a.id: a.name for a in db.query(db.alternative_sq).all()}
        links = [
            (alts[s.alternative_id], s.rank)
            for s in db.query(db.scenario_alternative_sq).all()
            if s.scenario_id == scen_id
        ]
    return sorted(links, key=lambda x: x[1])


def _row_count(url: str, alt: str, node: str, param: str) -> int:
    """Number of stored parameter_value rows for (alt, node, param)."""
    from spinedb_api import DatabaseMapping

    with DatabaseMapping(url) as db:
        alt_id = next(
            a.id for a in db.query(db.alternative_sq).all() if a.name == alt
        )
        ent_id = next(
            e.id for e in db.query(db.entity_sq).all() if e.name == node
        )
        pdef_id = next(
            p.id
            for p in db.query(db.parameter_definition_sq).all()
            if p.name == param
        )
        return sum(
            1
            for r in db.query(db.parameter_value_sq).all()
            if r.entity_id == ent_id
            and r.alternative_id == alt_id
            and r.parameter_definition_id == pdef_id
        )


# --- tests ------------------------------------------------------------------


def test_calib_alt_name():
    assert calib_alt_name(SCENARIO) == f"{SCENARIO}_adeq_calib"


def test_alt_created_at_top_rank(db_url: str):
    write_calib_alt(db_url, SCENARIO, {NODE: 1234.0})
    ranked = _ranked_alts(db_url, SCENARIO)
    assert ranked, "scenario has alternatives"
    top_name, _top_rank = ranked[-1]
    assert top_name == calib_alt_name(SCENARIO), (
        f"calib alt must be at the highest rank; stack: {ranked}"
    )


def test_calib_value_wins_under_scenario_filter(db_url: str):
    write_calib_alt(db_url, SCENARIO, {NODE: 1234.0})
    assert _materialized(db_url, SCENARIO, NODE, "energy_margin_adder") == 1234.0
    assert (
        _materialized(db_url, SCENARIO, NODE, "energy_margin_method")
        == "inflow_adder"
    )


def test_rewrite_updates_single_row(db_url: str):
    alt = calib_alt_name(SCENARIO)
    write_calib_alt(db_url, SCENARIO, {NODE: 1234.0})
    write_calib_alt(db_url, SCENARIO, {NODE: 5678.0})
    # Value updated in place...
    assert _materialized(db_url, SCENARIO, NODE, "energy_margin_adder") == 5678.0
    # ...and there is exactly one stored row per parameter (no duplicate).
    assert _row_count(db_url, alt, NODE, "energy_margin_adder") == 1
    assert _row_count(db_url, alt, NODE, "energy_margin_method") == 1


def test_identical_rewrite_is_noop(db_url: str):
    write_calib_alt(db_url, SCENARIO, {NODE: 1234.0})
    # Re-writing the identical state must not raise (NothingToCommit).
    write_calib_alt(db_url, SCENARIO, {NODE: 1234.0})
    assert _materialized(db_url, SCENARIO, NODE, "energy_margin_adder") == 1234.0


def test_empty_adder_still_materialises_alt(db_url: str):
    """A baseline (k=0) empty write still creates the alt + scenario link."""
    write_calib_alt(db_url, SCENARIO, {})
    ranked = _ranked_alts(db_url, SCENARIO)
    assert calib_alt_name(SCENARIO) == ranked[-1][0]


# --- timed (Map-valued) adder ----------------------------------------------


def _materialized_map(url: str, scenario: str, node: str):
    """Decode the materialised ``energy_margin_adder`` as ``{(period,time): float}``.

    Reads the value under the scenario filter and, when it is a 2-D
    ``period → time → float`` Map, flattens it to a per-cell dict so the test
    can assert the authored values round-tripped.
    """
    from spinedb_api import DatabaseMapping, export_data
    from spinedb_api.filters.scenario_filter import (
        apply_scenario_filter_to_subqueries,
    )
    from spinedb_api.parameter_value import Map

    with DatabaseMapping(url) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        data = export_data(db)
        for cls, ent, p, value, *_rest in data.get("parameter_values", []):
            name = ent[0] if isinstance(ent, (list, tuple)) else ent
            if p == "energy_margin_adder" and name == node:
                assert isinstance(value, Map), f"expected a Map, got {type(value)}"
                cells: dict[tuple[str, str], float] = {}
                for period, inner in zip(value.indexes, value.values):
                    for t, v in zip(inner.indexes, inner.values):
                        cells[(str(period), str(t))] = float(v)
                return cells
    return None


def test_timed_map_adder_round_trips(db_url: str):
    """A ``timed`` per-cell increment writes a period→time Map that reads back
    as exactly the authored per-cell values."""
    cells = {
        ("y2020", "t0001"): 12.5,
        ("y2020", "t0002"): 7.25,
        ("y2030", "t0001"): 3.0,
    }
    write_calib_alt(db_url, SCENARIO, {NODE: cells})
    got = _materialized_map(db_url, SCENARIO, NODE)
    assert got == pytest.approx(cells)
    # Method still set to inflow_adder.
    assert (
        _materialized(db_url, SCENARIO, NODE, "energy_margin_method")
        == "inflow_adder"
    )


def test_timed_map_adder_idempotent_overwrite(db_url: str):
    """Re-writing a changed per-cell map overwrites in place (one row)."""
    alt = calib_alt_name(SCENARIO)
    write_calib_alt(db_url, SCENARIO, {NODE: {("y2020", "t0001"): 1.0}})
    write_calib_alt(db_url, SCENARIO, {NODE: {("y2020", "t0001"): 9.0,
                                              ("y2020", "t0002"): 4.0}})
    got = _materialized_map(db_url, SCENARIO, NODE)
    assert got == pytest.approx({("y2020", "t0001"): 9.0, ("y2020", "t0002"): 4.0})
    assert _row_count(db_url, alt, NODE, "energy_margin_adder") == 1
