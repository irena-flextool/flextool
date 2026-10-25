"""non_anticipativity_storage_use holds exactly the flow terms of the node balance (#21).

The tie says the net flow into a stochastic node is the same in every forecast over the tied steps.  It must
therefore carry every flow term of nodeBalance_eq: the reverse flow ``v_flow_back`` of two-way one-variable and DC
arcs, and the section input ``v_online · section`` of min_load_efficiency units drawing from the node, as well as
the plain flows.  Databases are built from ``tests/fixtures/stochastics.json`` plus an override alternative (the
overlays of ``tools/lp_snapshot.py``), never from a checked-in ``.sqlite``.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build

REPO = Path(__file__).resolve().parents[2]
STOCHASTICS = REPO / "tests" / "fixtures" / "stochastics.json"
TIE = "non_anticipativity_storage_use"
FLOW_PREFIXES = ("v_flow[", "v_flow_back[", "v_online_linear[", "v_online_integer[")
BALANCE_FAMILIES = ("nodeBalance_eq", "nodeBalanceCoarse_eq")


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _backflow_section_overlay(data: dict) -> dict:
    """2-day stochastic dispatch + a lossless two-way connection ``spill`` from the stochastic hydro_reservoir to
    a new storage ``res2`` (reverse flow ``v_flow_back``), and a direct unit ``turb2`` (hydro_reservoir →
    demand_node, min_load_efficiency, linear online) whose section input is drawn from the stochastic node."""
    return _lp_snapshot()._overlay(
        data, name="custom", base="2_day_stochastic_dispatch",
        entities=[["node", "res2"], ["connection", "spill"],
                  ["connection__node__node", ["spill", "hydro_reservoir", "res2"]],
                  ["unit", "turb2"], ["unit__inputNode", ["turb2", "hydro_reservoir"]],
                  ["unit__outputNode", ["turb2", "demand_node"]]],
        values=[["node", "res2", "node_type", "storage"],
                ["node", "res2", "existing", 1000.0],
                ["node", "res2", "storage_state_start", 0.5],
                ["node", "res2", "storage_start_end_method", "fix_start"],
                ["node", "res2", "penalty_up", 10000.0],
                ["node", "res2", "penalty_down", 10000.0],
                ["connection", "spill", "existing", 100.0],
                ["connection", "spill", "transfer_method", "no_losses_no_variable_cost"],
                ["unit", "turb2", "existing", 200.0],
                ["unit", "turb2", "conversion_method", "min_load_efficiency"],
                ["unit", "turb2", "efficiency", 0.9],
                ["unit", "turb2", "efficiency_at_min_load", 0.7],
                ["unit", "turb2", "min_load", 0.3],
                ["unit", "turb2", "startup_method", "linear"]])


def _run(tmp_path_factory, data: dict, scenario: str) -> list:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    work = tmp_path_factory.mktemp(f"na_terms_{scenario}")
    json_path = work / "fixture.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / "fixture.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "work", keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal, "solve did not finish optimal"
    return list(steps.values())


def _inner(name: str) -> list[str]:
    return name[name.index("[") + 1:-1].split(",")


def _assert_tie_equals_balance_flow_terms(pb) -> int:
    """Every tie row (n, d_ref, b, t): its flow columns at (d_ref, t) have the coefficients of the node's balance
    at (d_ref, t), its flow columns at (b, t) the negated coefficients of the balance at (b, t), and no flow column
    of those balances is missing.  The balance of a coarse node is its nodeBalanceCoarse_eq row, which carries
    the fine-step coefficients unchanged.  Returns the number of tie rows checked."""
    tie = row_terms(pb, TIE)
    # (n, d) → {column: coefficient} over all balance rows of n in d; a flow column enters exactly one of them.
    balance: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)
    for fam in BALANCE_FAMILIES:
        for row, cols in row_terms(pb, fam).items():
            n, d = _inner(row)[:2]
            for col, v in cols.items():
                if col.startswith(FLOW_PREFIXES):
                    assert col not in balance[(n, d)], (col, row)
                    balance[(n, d)][col] = v
    for row, cols in tie.items():
        n, d_ref, b, t = _inner(row)
        for side, sign in ((d_ref, 1.0), (b, -1.0)):
            want = {c: sign * v for c, v in balance[(n, side)].items() if _inner(c)[-2:] == [side, t]}
            got = {c: v for c, v in cols.items() if c.startswith(FLOW_PREFIXES) and _inner(c)[-2:] == [side, t]}
            assert set(got) == set(want), (row, side, sorted(set(want) ^ set(got)))
            for c, v in want.items():
                assert got[c] == pytest.approx(v, rel=1e-12, abs=0.0), (row, c, got[c], v)
        assert all(c.startswith(FLOW_PREFIXES) for c in cols), (row, cols)
    return len(tie)


@pytest.fixture(scope="module")
def backflow_lp(tmp_path_factory):
    data = _backflow_section_overlay(json.loads(STOCHASTICS.read_text()))
    steps = _run(tmp_path_factory, data, "custom")
    step = steps[-1]
    return build(step.flex_data), step.flex_data


def _tie_row(pb, t: str = "t0001") -> tuple[str, dict[str, float]]:
    rows = {r: c for r, c in row_terms(pb, TIE, "hydro_reservoir,").items() if r.endswith(f",{t}]")}
    assert rows, "no tie row for hydro_reservoir"
    row = sorted(rows)[0]
    return row, rows[row]


def test_tie_row_contains_back_flow(backflow_lp):
    pb, _ = backflow_lp
    row, cols = _tie_row(pb)
    _, d_ref, b, t = _inner(row)
    assert cols[f"v_flow_back[spill,hydro_reservoir,res2,{d_ref},{t}]"] == pytest.approx(100.0)
    assert cols[f"v_flow_back[spill,hydro_reservoir,res2,{b},{t}]"] == pytest.approx(-100.0)


def test_tie_row_contains_section_input(backflow_lp):
    pb, fd = backflow_lp
    row, cols = _tie_row(pb)
    _, d_ref, b, t = _inner(row)
    balance = row_terms(pb, "nodeBalance_eq", f"hydro_reservoir,{d_ref},{t}")
    section = balance[f"nodeBalance_eq[hydro_reservoir,{d_ref},{t}]"][f"v_online_linear[turb2,{d_ref},{t}]"]
    assert section < 0.0
    assert cols[f"v_online_linear[turb2,{d_ref},{t}]"] == pytest.approx(section)
    assert cols[f"v_online_linear[turb2,{b},{t}]"] == pytest.approx(-section)


def test_tie_equals_balance_flow_terms(backflow_lp):
    pb, _ = backflow_lp
    assert _assert_tie_equals_balance_flow_terms(pb) > 0


@pytest.mark.parametrize("fixture", ["stochastic_2_day_dispatch", "stoch_na_reserve", "conn_online_stoch",
                                     "coarse_stoch"])
def test_tie_equals_balance_flow_terms_on_snapshot_fixtures(tmp_path_factory, fixture):
    """The same generic check on the lp_snapshot fixtures that emit the tie (coarse_stoch: coarse node rows)."""
    lps = _lp_snapshot()
    data = json.loads(STOCHASTICS.read_text())
    scenario = "2_day_stochastic_dispatch"
    if fixture != "stochastic_2_day_dispatch":
        data = lps.PATCHES[fixture](data)
        scenario = fixture
    checked = 0
    for step in _run(tmp_path_factory, data, scenario):
        checked += _assert_tie_equals_balance_flow_terms(build(step.flex_data))
    assert checked > 0


def test_backflow_overlay_solves_optimal(backflow_lp):
    """The extra terms keep the fixture feasible: the solve of the fixture already finished optimal."""
    pb, _ = backflow_lp
    assert row_terms(pb, TIE)


def test_tie_only_at_balance_nodes(tmp_path_factory):
    """A commodity node in the stochastic group has no energy balance, so it has no balance flow terms to tie
    (before, its tie held only the flows into it: an asymmetric half of a balance it does not have)."""
    data = _lp_snapshot()._overlay(
        json.loads(STOCHASTICS.read_text()), name="custom", base="2_day_stochastic_dispatch",
        entities=[["group__node", ["add_stochastics", "coal_market"]]], values=[])
    step = _run(tmp_path_factory, data, "custom")[-1]
    tie = row_terms(build(step.flex_data), TIE)
    assert tie
    assert not [r for r in tie if r.startswith(f"{TIE}[coal_market,")]
    assert all(r.startswith(f"{TIE}[hydro_reservoir,") for r in tie)
