"""``lifetime_technical``, ``payback_time``, ``net_salvage_value`` and ``no_reinvestment`` end to end (data version 81).

* A database with the version-80 names (``lifetime``, ``salvage_value``, ``no_investment``), migrated, has the
  schema and the values of the same database written with the new names, and builds the same LP in every solve.
* ``payback_time`` sets the annuity factor and nothing else: with the same technical lifetime (so the same
  window), the objective coefficient of every investment variable scales by the ratio of the annuity factors.
* Without ``payback_time`` the payback time is the technical lifetime: ``lifetime_technical`` L alone builds the
  LP of L and ``payback_time`` L, in a single, a rolling and a stochastic solve.

Fixtures are built from JSON: ``tests.json`` and ``stoch_two_period_invest.json``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database

from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, RunCache, b64, build
from tests.engine.test_constraint_constant_period_map import ROLL_LONG_JUMP
from tests.engine.test_lifetime_shapes import ROLL_CHAIN, STOCH_CHAIN, STOCH_JSON, _lp
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

TESTS_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json"
NEW_TO_OLD = {"lifetime_technical": "lifetime", "net_salvage_value": "salvage_value"}

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _packed(value) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode(), "str"]


def _as_version_80(data: dict) -> dict:
    """``data`` (a fixture JSON at version 81) written with the version-80 names: ``lifetime_technical`` is
    ``lifetime`` and ``net_salvage_value`` is ``salvage_value`` (definitions, types and values), ``payback_time`` is
    dropped (definitions, types and values: the migration has to make it), the ``lifetime_methods`` member and the
    values ``no_reinvestment`` are ``no_investment``, and the version is 80."""
    old = json.loads(json.dumps(data))
    new_member, old_member = _packed("no_reinvestment"), _packed("no_investment")
    old["parameter_definitions"] = [[r[0], NEW_TO_OLD.get(r[1], r[1]), *r[2:]] for r in old["parameter_definitions"]
                                    if r[1] != "payback_time"]
    for r in old["parameter_definitions"]:
        if r[:2] == ["model", "version"]:
            r[2] = b64(80, "float")
    old["parameter_types"] = [[r[0], NEW_TO_OLD.get(r[1], r[1]), *r[2:]] for r in old["parameter_types"]
                              if r[1] != "payback_time"]
    old["parameter_value_lists"] = [[r[0], old_member] if r[0] == "lifetime_methods" and r[1] == new_member else r
                                    for r in old["parameter_value_lists"]]
    values = []
    for r in old["parameter_values"]:
        if r[2] == "payback_time":
            continue
        r = [r[0], r[1], NEW_TO_OLD.get(r[2], r[2]), r[3], *r[4:]]
        if r[2] == "lifetime_method" and r[3] == new_member:
            r[3] = old_member
        values.append(r)
    old["parameter_values"] = values
    return old


def _schema_and_values(url: str) -> dict:
    with DatabaseMapping(url) as db:
        return {
            "defs": {(d["entity_class_name"], d["name"]): (d["description"], tuple(d["parameter_type_list"] or ()),
                                                           d["parameter_group_name"], d["parameter_value_list_name"],
                                                           from_database(d["default_value"], d["default_type"]))
                     for d in db.get_items("parameter_definition")},
            "lists": sorted((x["parameter_value_list_name"], x["index"], x["parsed_value"])
                            for x in db.get_items("list_value")),
            "values": {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
                        v["alternative_name"]): (bytes(v["value"]), v["type"])
                       for v in db.get_items("parameter_value")},
        }


@pytest.mark.solver
@pytest.mark.parametrize("scenario", ["wind_battery_invest_lifetime_choice", "coal_retire",
                                      "multi_year_wind_no_investment"])
def test_v80_database_builds_the_same_lp_after_migration(tmp_path_factory, scenario) -> None:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp(f"v81_e2e_{scenario}")
    data = json.loads(TESTS_JSON.read_text(encoding="utf-8"))
    (work / "old.json").write_text(json.dumps(_as_version_80(data)), encoding="utf-8")
    old_url = json_to_db(work / "old.json", work / "old.sqlite")
    with DatabaseMapping(old_url) as db:
        assert db.get_items("parameter_definition", entity_class_name="unit", name="lifetime")
        assert not db.get_items("parameter_definition", entity_class_name="unit", name="payback_time")
    migrate_database(old_url)
    new_url = json_to_db(TESTS_JSON, work / "new.sqlite")
    migrated, direct = _schema_and_values(old_url), _schema_and_values(new_url)
    assert migrated["defs"] == direct["defs"]
    assert migrated["lists"] == direct["lists"]
    lifetimes = {k[:2] + k[3:]: v for k, v in migrated["values"].items() if k[2] == "lifetime_technical"}
    assert lifetimes and lifetimes == {k[:2] + k[3:]: v for k, v in migrated["values"].items()
                                       if k[2] == "payback_time"}
    assert migrated["values"] == direct["values"]

    runs = {name: run_chain_from_db(url, scenario, work_folder=work / f"{name}_w", keep_solutions=True)
            for name, url in (("migrated", old_url), ("direct", new_url))}
    assert list(runs["migrated"]) == list(runs["direct"]) and runs["direct"]
    for step in runs["direct"]:
        assert _lp(runs["migrated"][step].flex_data) == _lp(runs["direct"][step].flex_data), step


# ---------------------------------------------------------------------------
# payback_time sets the annuity factor
#
# coal_plant invests (invest_no_limit) with invest_cost 1000 CUR/kW, discount_rate r = 0.05, fixed_cost 0 and
# lifetime_technical 20 (the window: every period of the run), payback_time n = 10 or 20.  The objective
# coefficient of v_invest[coal_plant, d] is invest_cost · 1000 · f(r, n) · (the discounting of the window, the same
# for both n) · unitsize, so the 10-year coefficient over the 20-year one is
#   f(r, n) = r / (1 − (1 + r)^−n):  f(0.05, 10) = 0.05 / (1 − 0.6139133) = 0.1295046,
#                                    f(0.05, 20) = 0.05 / (1 − 0.3768895) = 0.0802426,  ratio 1.6139133.


def _annuity_factor(r: float, n: float) -> float:
    return r / (1.0 - (1.0 + r) ** -n)


def _invest_objective(tmp_path_factory, chain: list[str], payback: float) -> dict[str, dict[str, float]]:
    """``{solve: {v_invest column of coal_plant: objective coefficient}}`` (unscaled build)."""
    values = [["unit", PLANT, name, b64(v, "float")] for name, v in (
        ("invest_cost", 1000.0), ("discount_rate", 0.05), ("fixed_cost", 0.0), ("lifetime_technical", 20.0),
        ("payback_time", payback))] + [["unit", PLANT, "invest_method", b64("invest_no_limit", "str")]]
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"payback_{len(chain)}_{int(payback)}", values=values,
                            base_chain=chain)
    out = {}
    for name, step in steps.items():
        m = build(step.flex_data).canonicalise()
        out[name] = {m.col_names[c]: float(m.col_obj[c]) for c in range(m.n_cols)
                     if m.col_names[c].startswith("v_invest") and f"{PLANT}," in m.col_names[c]}
    return out


@pytest.mark.solver
@pytest.mark.parametrize("chain", [SINGLE, CHAIN], ids=["single", "chain"])
def test_payback_time_sets_the_annuity_factor(tmp_path_factory, chain) -> None:
    ten = _invest_objective(tmp_path_factory, chain, 10.0)
    twenty = _invest_objective(tmp_path_factory, chain, 20.0)
    ratio = _annuity_factor(0.05, 10.0) / _annuity_factor(0.05, 20.0)
    assert ratio == pytest.approx(1.6139133, rel=1e-7)
    assert list(ten) == list(twenty)
    checked = 0
    for solve, cols in twenty.items():
        assert set(ten[solve]) == set(cols), solve
        for col, coef in cols.items():
            assert coef != 0.0, (solve, col)
            assert ten[solve][col] == pytest.approx(ratio * coef, rel=1e-9), (solve, col)
            checked += 1
    assert checked >= 4


# ---------------------------------------------------------------------------
# No payback_time: the payback time is the technical lifetime


def _fixture_without_payback(tmp_path_factory, fixture: Path, entity: str) -> Path:
    """``fixture`` without the ``payback_time`` values of ``entity`` (it keeps its ``lifetime_technical``)."""
    data = json.loads(fixture.read_text(encoding="utf-8"))
    data["parameter_values"] = [r for r in data["parameter_values"]
                                if not (r[1] == entity and r[2] == "payback_time")]
    path = tmp_path_factory.mktemp(f"no_payback_{fixture.stem}") / fixture.name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


CASES = {
    # case: (entity, fixture, base chain, extra values, technical lifetime)
    "single": (PLANT, TESTS_JSON, SINGLE,
               [["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
                ["unit", PLANT, "invest_method", b64("invest_retire_no_limit", "str")],
                ["unit", PLANT, "net_salvage_value", b64(50.0, "float")]], 10.0),
    "rolling": (PLANT, TESTS_JSON, ROLL_CHAIN,
                ROLL_LONG_JUMP + [["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")]], 10.0),
    "stochastic": ("peaker", STOCH_JSON, STOCH_CHAIN,
                   [["unit", "peaker", "lifetime_method", b64("reinvest_choice", "str")],
                    ["unit", "peaker", "invest_method", b64("invest_retire_no_limit", "str")],
                    ["unit", "peaker", "net_salvage_value", b64(0.1, "float")]], 1.0),
}


@pytest.mark.solver
@pytest.mark.parametrize("case", list(CASES))
def test_technical_lifetime_alone_equals_both(tmp_path_factory, case) -> None:
    entity, fixture, chain, extra, life = CASES[case]
    fixture = _fixture_without_payback(tmp_path_factory, fixture, entity)
    runs = {}
    for with_payback in (False, True):
        values = list(extra) + [["unit", entity, "lifetime_technical", b64(life, "float")]]
        if with_payback:
            values.append(["unit", entity, "payback_time", b64(life, "float")])
        steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"alone_{case}_{with_payback}", values=values,
                                base_chain=chain, fixture_json=fixture)
        runs[with_payback] = steps
    alone, both = runs[False], runs[True]
    assert list(alone) == list(both) and both
    if case == "rolling":
        assert len(both) == 8
    for name in both:
        assert _lp(alone[name].flex_data) == _lp(both[name].flex_data), name
