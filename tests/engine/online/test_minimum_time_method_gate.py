"""The min up / down rows are built only when ``minimum_time_method`` enables the time (C rev 3 §5, §10.3 T1-T4).

``_process_min_time`` keeps a unit's ``min_uptime`` with ``minimum_time_method`` ``min_uptime`` / ``both``, its
``min_downtime`` with ``min_downtime`` / ``both``; an unset method is ``none``, which enables neither.  W-0231 names a
unit whose time its method leaves out (except a deliberate, explicit ``none``).

Hand-derived oracles.  T1 / T2 on ``tests.json`` (``coal_plant``: one 500 MW unit, efficiency 0.4, coal 20 €/MWh,
so 50 €/MWh of fuel; ``west``: penalty_up 900, penalty_down 800 €/MWh; ``startup_method linear``, ``min_load 0.5``,
no start-up cost), 1 h steps.  A unit online at min load with no demand dumps 250 MW: 250 · 800 + 250 · 50 =
212 500 per step.  The first step of a solve is a free start (no start-up, so no minimum up time from it).

* T1, demand 500 / 0 / 500 MW, ``min_downtime 2``.  t1 and t3 need the whole unit (unmet demand 900 > 50).
  Enabled: going off at t2 (share x) keeps it off at t3, 500 · x MW unmet there (450 000 · x) against 212 500 · x
  for staying on: online (1, 1, 1).  Disabled: nothing binds t3, so off at t2: (1, 0, 1).
* T2, demand 0 / 500 / 0 / 0 MW, ``min_uptime 3``.  t2 needs the whole unit.  Enabled: a share a online at the free
  t1 makes no start-up at t2; the rest (1 - a) starts at t2 and must stay on at t3 and t4: 212 500 · a +
  425 000 · (1 - a), least at a = 1: (1, 1, 0, 0).  Disabled: t1 online only costs: (0, 1, 0, 0).

T3 (two-stage, C §10.3 U5 / S8) on ``stoch_t0_commit.json`` reshaped (wind removed): one ``integer_fix_in_uc``
``peaker`` (``existing`` 1, ``virtual_unitsize`` 100 MW, ``min_load 0.5``, 10 €/MWh, no investment), demand
100 / 0 / 100 MW, penalty_up = penalty_down = 10 000 €/MWh, ``min_downtime 2``, ``both`` in ``base`` and ``none`` in
the top alternative.  Each forecast sees the T1 shape: off at t2 costs nothing (none), on costs 50 · 10 000, so
(1, 0, 1); under the old ungated rows the shutdown at t2 would keep it off at t3 (100 · 10 000 unmet), so (1, 1, 1).
The rolling solve (horizon 2 h, jump 1 h, the ``realized: yes`` branch at each roll start) gives the same per
realized step.  The realization solve of the single solve keeps the reference forecast's units online.
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from spinedb_api import DatabaseMapping, to_database

from tests._support.coded_warnings import capture_coded
from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, make_db, time_map, uc, utf8

FLEXTOOL_ROOT = Path(__file__).resolve().parents[3]
T0_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_t0_commit.json"
UNSET = None


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _steps_of(hours: int) -> list:
    return [["timeset", "2day", "timeset_duration",
             b64({"index_type": "str", "index_name": "time", "data": [["t0001", float(hours)]]}, "map")]]


def _solve_plant(tmp_path_factory, alt: str, demand: list[float], times: dict[str, float], method):
    """Solve ``coal_plant`` (linear, min_load 0.5) with *times* and *method* (None: not set) over ``len(demand)``
    steps; return ``(v_online_linear of coal_plant by step, captured coded warnings)``."""
    from flextool.engine import run_chain_from_db

    values = uc(PLANT, 0.5, "linear") + _steps_of(len(demand)) + [
        ["node", "west", "inflow", time_map({f"t{i + 1:04d}": -x for i, x in enumerate(demand)}, 0.0)]]
    values += [["unit", PLANT, k, b64(v, "float")] for k, v in times.items()]
    if method is not None:
        values.append(["unit", PLANT, "minimum_time_method", b64(method, "str")])
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN)
    with capture_coded() as got:
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)
    (step,) = steps.values()
    assert step.solution.optimal
    online = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == PLANT).sort("t")
    return [round(x, 6) + 0.0 for x in online["value"].to_list()], got


# ─── T1 / T4: down-time gate and W-0231 ─────────────────────────────────────


@pytest.mark.solver
@pytest.mark.parametrize("method, expected", [
    ("both", [1.0, 1.0, 1.0]), ("min_downtime", [1.0, 1.0, 1.0]),
    ("none", [1.0, 0.0, 1.0]), ("min_uptime", [1.0, 0.0, 1.0]), (UNSET, [1.0, 0.0, 1.0])])
def test_t1_down_time_applies_only_when_enabled(tmp_path_factory, method, expected) -> None:
    online, got = _solve_plant(tmp_path_factory, f"t1_{method}", [500.0, 0.0, 500.0], {"min_downtime": 2.0}, method)
    assert online == expected
    # T4 inside T1: the method leaves the time out → one W-0231 naming min_downtime; an explicit none or an
    # enabling method → none.
    if method in ("min_uptime", UNSET):
        assert got.names("W-0231") == [PLANT]
        shown = "min_uptime" if method else "none (not set)"
        assert got.detail("W-0231", PLANT) == (
            f"min_downtime 2 h, minimum_time_method {shown}; no minimum down time constraint")
    else:
        assert got.names("W-0231") == []


@pytest.mark.solver
def test_startup_default_without_a_time_builds(tmp_path_factory) -> None:
    """``coal_plant`` with ``min_load 0.5``, no ``startup_method`` of its own and the definition default ``linear``,
    no min up / down time; demand 500 / 0 / 500 MW.  The input derivation reads the default as the engine does, so the
    unit gets online variables and the model builds (reading explicit values only, the input derivation gave the
    unit no online method while the engine built online rows, and the build stopped).  Oracle as T1 disabled: t1 and
    t3 need the whole unit; at t2 a share x online dumps 250 · x MW (212 500 · x), off is free: (1, 0, 1)."""
    from flextool.engine import run_chain_from_db

    values = [["unit", PLANT, "min_load", b64(0.5, "float")]] + _steps_of(3) + [
        ["node", "west", "inflow", time_map({"t0001": -500.0, "t0002": 0.0, "t0003": -500.0}, 0.0)]]
    url, scen = make_db(tmp_path_factory, alt="startup_dflt", values=values, base_chain=PLANT_CHAIN)
    with DatabaseMapping(url) as db:
        _set_default(db, "startup_method", "linear")
        db.commit_session("startup_method default")
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("startup_dflt_w"), keep_solutions=True)
    (step,) = steps.values()
    assert step.solution.optimal
    online = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == PLANT).sort("t")
    assert [round(x, 6) + 0.0 for x in online["value"].to_list()] == [1.0, 0.0, 1.0]


# ─── T2: up-time gate ───────────────────────────────────────────────────────


@pytest.mark.solver
@pytest.mark.parametrize("method, expected", [
    ("both", [1.0, 1.0, 0.0, 0.0]), ("min_uptime", [1.0, 1.0, 0.0, 0.0]),
    ("none", [0.0, 1.0, 0.0, 0.0]), ("min_downtime", [0.0, 1.0, 0.0, 0.0]), (UNSET, [0.0, 1.0, 0.0, 0.0])])
def test_t2_up_time_applies_only_when_enabled(tmp_path_factory, method, expected) -> None:
    online, _got = _solve_plant(tmp_path_factory, f"t2_{method}", [0.0, 500.0, 0.0, 0.0], {"min_uptime": 3.0},
                                method)
    assert online == expected


# ─── T4: W-0231 on the input derivation ─────────────────────────────────────


class _Provider:
    def __init__(self) -> None:
        self.frames: dict[str, pl.DataFrame] = {}

    def has(self, key: str) -> bool:
        return key in self.frames

    def get(self, key: str) -> pl.DataFrame:
        return self.frames[key]

    def put(self, key: str, frame: pl.DataFrame) -> None:
        self.frames[key] = frame


def _set_default(db: DatabaseMapping, parameter: str, value) -> None:
    """Set the definition default of ``unit.<parameter>``."""
    pdef = db.get_item("parameter_definition", entity_class_name="unit", name=parameter)
    raw, typ = to_database(value)
    db.update_item("parameter_definition", id=pdef["id"], default_value=raw, default_type=typ)


def _w0231(schema_db_url: str, tmp_path: Path, units: dict[str, dict], *, method_default=None,
           startup_default=None, corrupt: bool = False, markers: dict | None = None) -> dict[str, str]:
    """``{unit: W-0231 detail}`` of ``derive_process_method`` on a current-schema database with *units*
    ``{unit: {param: value}}`` in ``base``."""
    import logging
    import sqlite3

    from flextool.engine import _behaviour_changes as bc
    from flextool.input_derivation._process_method import derive_process_method

    path = tmp_path / "w0231.sqlite"
    shutil.copy(schema_db_url.removeprefix("sqlite:///"), path)
    url = f"sqlite:///{path}"
    with DatabaseMapping(url) as db:
        db.add_entity(entity_class_name="node", entity_byname=("elec",))
        for unit, params in units.items():
            db.add_entity(entity_class_name="unit", entity_byname=(unit,))
            db.add_entity(entity_class_name="unit__outputNode", entity_byname=(unit, "elec"))
            for param, value in params.items():
                raw, typ = to_database(value)
                db.add_parameter_value(entity_class_name="unit", entity_byname=(unit,),
                                       parameter_definition_name=param, alternative_name="base", value=raw, type=typ)
        if method_default is not None:
            _set_default(db, "minimum_time_method", method_default)
        if startup_default is not None:
            _set_default(db, "startup_method", startup_default)
        db.commit_session("w0231")
    if corrupt:  # item D: the first member overwritten with none, [none, min_downtime, both]
        con = sqlite3.connect(path)
        try:
            lst = "(SELECT id FROM parameter_value_list WHERE name = 'minimum_time_methods')"
            con.execute(f"DELETE FROM list_value WHERE value = ? AND parameter_value_list_id = {lst}", (b'"none"',))
            con.execute(f"UPDATE list_value SET value = ? WHERE value = ? AND parameter_value_list_id = {lst}",
                        (b'"none"', b'"min_uptime"'))
            con.commit()
        finally:
            con.close()
    bc.start_run(markers or {})
    try:
        with capture_coded() as got, DatabaseMapping(url) as db:
            derive_process_method(db, _Provider(), logging.getLogger(__name__), ct_method_overrides={})
    finally:
        bc.start_run({})
    return {n: got.detail("W-0231", n) for n in got.names("W-0231")}


LINEAR_DOWN = {"startup_method": "linear", "min_downtime": 2.0}


def test_t4_w0231_cases(schema_db_url: str, tmp_path: Path) -> None:
    """One database: the user's min_uptime method (names min_downtime), an explicit none (silent), unset (named),
    an enabling both (silent), a unit without online variables (no effect)."""
    got = _w0231(schema_db_url, tmp_path, {
        "own": LINEAR_DOWN | {"minimum_time_method": "min_uptime"},
        "off": LINEAR_DOWN | {"minimum_time_method": "none"},
        "unset": LINEAR_DOWN,
        "on": LINEAR_DOWN | {"minimum_time_method": "both"},
        "flow_only": {"min_downtime": 2.0}})
    assert got == {
        "own": "min_downtime 2 h, minimum_time_method min_uptime; no minimum down time constraint",
        "unset": "min_downtime 2 h, minimum_time_method none (not set); no minimum down time constraint",
        "flow_only": ("min_downtime 2 h, minimum_time_method none (not set); the unit has no online variables, so "
                      "the time has no effect")}


def test_t4_a_none_default_is_no_default(schema_db_url: str, tmp_path: Path) -> None:
    """A ``none`` definition default (U2) is not a deliberate switch-off: the unit is named (N3)."""
    got = _w0231(schema_db_url, tmp_path, {"u": LINEAR_DOWN}, method_default="none")
    assert list(got) == ["u"]


def test_t4_a_user_method_default_is_labelled(schema_db_url: str, tmp_path: Path) -> None:
    """A ``min_uptime`` definition default leaves a min downtime out; the detail says the method is the default."""
    got = _w0231(schema_db_url, tmp_path, {"u": LINEAR_DOWN}, method_default="min_uptime")
    assert got == {"u": "min_downtime 2 h, minimum_time_method min_uptime (default); no minimum down time constraint"}


def test_t4_online_through_a_startup_default(schema_db_url: str, tmp_path: Path) -> None:
    """A ``startup_method`` definition default ``linear`` gives the unit online variables (the engine broadcasts it),
    so W-0231 names the lost constraint, not "no online variables"."""
    got = _w0231(schema_db_url, tmp_path, {"u": {"min_downtime": 2.0}}, startup_default="linear")
    assert got == {"u": "min_downtime 2 h, minimum_time_method none (not set); no minimum down time constraint"}


def test_t4_a_corrupted_list_names_the_fingerprint(schema_db_url: str, tmp_path: Path) -> None:
    """``minimum_time_methods`` without ``min_uptime``: a value that was ``min_uptime`` reads ``none``, and the
    explicit-none exception is lifted with the fingerprint sentence."""
    got = _w0231(schema_db_url, tmp_path,
                 {"old": {"startup_method": "linear", "min_uptime": 2.0, "minimum_time_method": "min_uptime"}},
                 corrupt=True)
    assert got == {"old": (
        "min_uptime 2 h, minimum_time_method none; no minimum up time constraint. The value list "
        "minimum_time_methods has no min_uptime: an old database upgrade overwrote it, and values that were "
        "min_uptime now read none. Check these units.")}


def test_t4_an_undecided_unit_gets_the_reason(schema_db_url: str, tmp_path: Path) -> None:
    """A unit in the marker's ``undecided`` list is named with the upgrade's reason, also under an explicit none."""
    reason = "a value in alternative base would replace your minimum_time_method in alternative E in scenario S"
    marker = {"minimum_time_method_written": json.dumps(
        {"since": 83, "writes": [], "undecided": [["u", reason, ["S"]]]})}
    got = _w0231(schema_db_url, tmp_path, {"u": LINEAR_DOWN | {"minimum_time_method": "none"}}, markers=marker)
    assert got == {"u": ("min_downtime 2 h, minimum_time_method none; no minimum down time constraint. The database "
                         f"upgrade could not write minimum_time_method for this unit: {reason}.")}


# ─── T3: two-stage ──────────────────────────────────────────────────────────

PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"
STEPS = ("t0001", "t0002", "t0003")


def _enc(value, kind: str) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode(), kind]


def _leaf(data) -> dict:
    return {"type": "map", "index_type": "str", "data": data}


def _two_stage(rolling: bool):
    """``stoch_t0_commit`` as the T3 model (module docstring); *rolling*: 3 rolls (horizon 2 h, jump 1 h) with the
    forecasts and the realized branch at each roll start."""
    def mutate(data: dict) -> None:
        wind = {"wind", "wprof", "stoch_g"}
        data["entities"] = [e for e in data["entities"] if not (set(e[1] if isinstance(e[1], list) else [e[1]])
                                                                 & wind)]
        drop = {"invest_method", "invest_cost", "invest_max_total", "lifetime_technical", "discount_rate",
                "min_uptime", "startup_cost", "existing", "virtual_unitsize", "startup_method", "min_downtime"}
        keep = []
        for r in data["parameter_values"]:
            names = set(r[1] if isinstance(r[1], list) else [r[1]])
            if names & wind or (r[1] == "peaker" and r[2] in drop):
                continue
            if r[1] == "elec" and r[2] == "inflow":
                r[3] = _enc({"index_type": "str", "data": [["t0001", -100.0], ["t0002", 0.0], ["t0003", -100.0]]},
                            "map")
            if r[0] == "solve" and r[2] == "stochastic_branches" and rolling:
                starts = list(STEPS)
                r[3] = _enc({"index_type": "str", "data": [["p2035", _leaf([
                    [b, _leaf([[t, _leaf([[flag, w]])] for t in starts])]
                    for b, flag, w in (("rlz", "yes", 0.0), ("high", "no", 0.5), ("low", "no", 0.5))])]]}, "map")
            keep.append(r)
        keep += [["unit", "peaker", "existing", _enc(1.0, "float"), "base"],
                 ["unit", "peaker", "virtual_unitsize", _enc(100.0, "float"), "base"],
                 ["unit", "peaker", "startup_method", _enc("integer_fix_in_uc", "str"), "base"],
                 ["unit", "peaker", "min_downtime", _enc(2.0, "float"), "base"],
                 ["unit", "peaker", "minimum_time_method", _enc("none", "str"), "top"]]
        if rolling:
            keep += [["solve", PARENT, "solve_mode", _enc("rolling_window", "str"), "top"],
                     ["solve", PARENT, "rolling_solve_horizon", _enc(2.0, "float"), "top"],
                     ["solve", PARENT, "rolling_solve_jump", _enc(1.0, "float"), "top"]]
        data["parameter_values"] = keep
        data["entity_alternatives"] = [e for e in data.get("entity_alternatives", [])
                                       if not (set(e[1]) & wind)]
        data["alternatives"].append(["top", "minimum_time_method none over both"])
        data["scenario_alternatives"] = [[s, a, "top" if b is None else b] for s, a, b in data["scenario_alternatives"]]
        data["scenario_alternatives"].append(["stoch_t0_commit", "top", None])
    return mutate


def _run_two_stage(tmp_path: Path, rolling: bool) -> dict:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads(T0_FIXTURE.read_text(encoding="utf-8"))
    _two_stage(rolling)(data)
    fixture = tmp_path / "t3.json"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(fixture, tmp_path / "t3.sqlite")
    migrate_database(url)
    work = tmp_path / "work"
    work.mkdir()
    return run_chain_from_db(url, "stoch_t0_commit", work_folder=work, warm=False, keep_solutions=True)


def _online(step) -> pl.DataFrame:
    return utf8(step.solution.value("v_online_integer")).filter(pl.col("p") == "peaker")


@pytest.mark.solver
def test_t3_single_two_stage_forecasts_and_realization(tmp_path: Path) -> None:
    steps = _run_two_stage(tmp_path, rolling=False)
    assert list(steps) == [PARENT, SIBLING]
    assert all(s.solution.optimal for s in steps.values())
    parent = _online(steps[PARENT])
    forecasts = sorted(d for d in set(parent["d"]) if d != "p2035")
    assert forecasts == ["p2035_high", "p2035_low"]
    for d in forecasts:
        assert [round(x, 9) + 0.0 for x in parent.filter(pl.col("d") == d).sort("t")["value"]] == [1.0, 0.0, 1.0], d
    # The realization keeps the reference forecast's (the first sorted) units online.
    sib = _online(steps[SIBLING]).filter(pl.col("d") == "p2035").sort("t")
    assert [round(x, 9) + 0.0 for x in sib["value"]] == [1.0, 0.0, 1.0]


@pytest.mark.solver
def test_t3_rolling_two_stage_forecasts(tmp_path: Path) -> None:
    steps = _run_two_stage(tmp_path, rolling=True)
    rolls = [k for k in steps if "__realized_dispatch" not in k]
    assert len(rolls) == 3, list(steps)
    assert all(s.solution.optimal for s in steps.values())
    for k, t in zip(rolls, STEPS):
        online = _online(steps[k]).filter((pl.col("t") == t) & (pl.col("d") != "p2035"))
        assert sorted(online["d"].to_list()) == ["p2035_high", "p2035_low"], (k, online)
        expected = 0.0 if t == "t0002" else 1.0
        assert [round(x, 9) + 0.0 for x in online["value"]] == [expected, expected], (k, t)
