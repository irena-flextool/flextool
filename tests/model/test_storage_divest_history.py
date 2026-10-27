"""Storage-node divestment history: retired capacity stays retired.

The storage-state cap (``maxState``) and the cumulative-capacity constraints
read the active-divest set to subtract every earlier retirement from the
built capacity.  Before this fix, the node consumer sites read the
process-only active-divest set and cast node ids against its process Enum
(CLAUDE.md invariant #2: never broadcast across an axis by casting a wrong
vocabulary) — the cast produced nulls, the semi-join matched zero rows, and
a storage-node divest never reached any ``maxState`` row, not even its own
period's.  A retirement earned its salvage value without ever reducing the
storage capacity bound, and the state stayed as free to cycle as before the
retirement.

Hand derivation: with ``invest_method = retire_period`` (no invest
variable) and ``retire_min_period == retire_max_period`` forced to 40 in
p2020 only (0 elsewhere), the storage-state cap

    v_state[n,d,t] + Σ_{d′≤d} v_divest_n[n,d′] ≤ existing[n] / state_unitsize[n]

with ``existing = 50`` and ``state_unitsize = 1`` forces
``v_divest_n[battery,p2020] == 40`` and caps
``v_state[battery,d,t] ≤ 50 − 40 = 10`` for every period d ∈ {p2020, p2025,
p2030, p2035} (the retirement is permanent, no lifetime / reinvest).  Before
the fix, only p2020's own ``maxState`` row carried the divest term, so later
periods stayed capped at the un-retired 50 MWh and could cycle up to it.

Every DB is built from a JSON fixture plus an override alternative (never a
checked-in ``.sqlite``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
BASE = "network_coal_wind_battery_invest_cumulative"
PERIODS = ("p2020", "p2025", "p2030", "p2035")


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _overlay(data: dict) -> tuple[dict, str]:
    lps = _lp_snapshot()
    retire = lps._pmap({"p2020": 40.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0})
    data = lps._overlay(
        data, name="storage_divest_history", base=BASE,
        values=[
            ["node", "battery", "invest_method", "retire_period"],
            ["node", "battery", "retire_min_period", retire],
            ["node", "battery", "retire_max_period", retire],
            ["node", "battery", "salvage_value", 1000.0],
            ["node", "battery", "discount_rate", 0.05],
        ])
    return data, "storage_divest_history"


def _solve(tmp_path_factory, monkeypatch):
    """Build the DB from the overlay and run it, under autoscale-strict."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    data, scenario = _overlay(json.loads((FIXTURES / "tests.json").read_text()))
    work = tmp_path_factory.mktemp("storage_divest_history")
    json_path = work / "fixture.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / "fixture.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "work", keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    assert len(steps) == 1, f"expected a single solve, got {list(steps)}"
    step = next(iter(steps.values()))
    assert step.solution is not None, "solve produced no solution"
    return step


def test_divested_storage_capacity_stays_retired_in_later_periods(tmp_path_factory, monkeypatch):
    step = _solve(tmp_path_factory, monkeypatch)

    divest = step.solution.value("v_divest_n").filter(pl.col("n") == "battery")
    p2020_divest = divest.filter(pl.col("d") == "p2020")["value"][0]
    assert p2020_divest == pytest.approx(40.0, abs=1e-6)

    state = step.solution.value("v_state").filter(pl.col("n") == "battery")
    for d in PERIODS:
        max_state = state.filter(pl.col("d") == d)["value"].max()
        assert max_state <= 10.0 + 1e-6, f"{d}: v_state reaches {max_state}, expected <= 10"
