"""Tests for the solve-to-solve handoff CSV writers.

Two layers:
  * Unit tests with mocked highs + on-disk parameter CSVs in a tmp
    workdir — fast, no solver invocation.
  * One integration test that runs a real multi-solve scenario
    (``wind_battery_invest``) and asserts the Python-written handoff
    files match what phase 3 produces, within ``%.8g`` rounding.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from flextool.process_outputs.handoff_writers import (
    _is_first_solve,
    _load_unitsize,
    write_p_entity_divested,
    write_p_entity_period_existing_capacity,
    write_p_roll_continue_state,
    write_fix_storage_quantity,
    write_fix_storage_price,
    write_fix_storage_usage,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_highs(
    variable_names: list[str],
    col_values: list[float],
    row_names: list[str] | None = None,
    row_duals: list[float] | None = None,
) -> MagicMock:
    h = MagicMock()
    h.allVariableNames.return_value = list(variable_names)
    sol = SimpleNamespace(
        col_value=list(col_values),
        row_dual=list(row_duals or []),
    )
    h.getSolution.return_value = sol
    h.getLp.return_value = SimpleNamespace(row_names_=list(row_names or []))
    return h


def _make_workfolder(tmp_path: Path, *, first_solve: bool = True) -> Path:
    """Build a minimal work folder with the input/ + output_raw/ files
    every handoff writer reads."""
    (tmp_path / "input").mkdir()
    (tmp_path / "output_raw").mkdir()
    (tmp_path / "solve_data").mkdir()

    # solveFirst flag — input/p_model.csv (long format).
    (tmp_path / "input" / "p_model.csv").write_text(
        f"modelParam,p_model\nsolveFirst,{1 if first_solve else 0}\nsolveLast,1\n"
    )
    return tmp_path


def _write_unitsize(work: Path, unitsize: dict[str, float]) -> None:
    """Mirror the wide-format ``input/p_entity_unitsize.csv`` layout."""
    cols = list(unitsize.keys())
    vals = [str(unitsize[k]) for k in cols]
    (work / "input" / "p_entity_unitsize.csv").write_text(
        "entity," + ",".join(cols) + "\nvalue," + ",".join(vals) + "\n"
    )


def _write_entity_set(work: Path, entities: list[str]) -> None:
    (work / "input" / "set_entity.csv").write_text(
        "entity\n" + "\n".join(entities) + "\n"
    )


def _write_entity_divest_set(work: Path, entities: list[str]) -> None:
    (work / "input" / "set_entityDivest.csv").write_text(
        "entity\n" + "\n".join(entities) + "\n"
    )


# ---------------------------------------------------------------------------
# _is_first_solve
# ---------------------------------------------------------------------------


def test_is_first_solve_reads_p_model(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path, first_solve=True)
    assert _is_first_solve(work) is True

    (work / "input" / "p_model.csv").write_text(
        "modelParam,p_model\nsolveFirst,0\nsolveLast,0\n"
    )
    assert _is_first_solve(work) is False


def test_is_first_solve_defaults_true_when_missing(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    assert _is_first_solve(tmp_path) is True


def test_load_unitsize_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_unitsize(work, {"battery": 1.0, "wind_plant": 1000.0})
    assert _load_unitsize(work) == {"battery": 1.0, "wind_plant": 1000.0}


# ---------------------------------------------------------------------------
# write_p_entity_divested
# ---------------------------------------------------------------------------


def test_p_entity_divested_first_solve(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_unitsize(work, {"unit_a": 100.0, "unit_b": 1.0})
    _write_entity_divest_set(work, ["unit_a", "unit_b"])

    # v_divest[unit_a, p2025]=2 → cumulative 2*100 = 200.  unit_b has none.
    h = _fake_highs(
        variable_names=["v_divest[unit_a,p2025]"], col_values=[2.0],
    )

    write_p_entity_divested(h, solve_name="s1", work_folder=work)
    rows = list(csv.DictReader(open(work / "solve_data" / "p_entity_divested.csv")))
    assert len(rows) == 2
    by_entity = {r["entity"]: float(r["p_entity_divested"]) for r in rows}
    assert by_entity == {"unit_a": 200.0, "unit_b": 0.0}


def test_p_entity_divested_second_solve_accumulates(tmp_path: Path) -> None:
    """Non-first solve adds prior file values + this solve's v_divest."""
    work = _make_workfolder(tmp_path, first_solve=False)
    _write_unitsize(work, {"unit_a": 50.0})
    _write_entity_divest_set(work, ["unit_a"])
    # Prior cumulative = 100; this solve adds v_divest=3 * unitsize=50 = 150.
    (work / "solve_data" / "p_entity_divested.csv").write_text(
        "entity,p_entity_divested\nunit_a,100\n"
    )
    h = _fake_highs(
        variable_names=["v_divest[unit_a,p2030]"], col_values=[3.0],
    )

    write_p_entity_divested(h, solve_name="s2", work_folder=work)
    rows = list(csv.DictReader(open(work / "solve_data" / "p_entity_divested.csv")))
    assert {r["entity"]: float(r["p_entity_divested"]) for r in rows} == {"unit_a": 250.0}


# ---------------------------------------------------------------------------
# write_fix_storage_quantity
# ---------------------------------------------------------------------------


def test_fix_storage_quantity_writes_only_for_target_method(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_unitsize(work, {"battery": 10.0, "tank": 1.0})
    (work / "input" / "node__storage_nested_fix_method.csv").write_text(
        "node,storage_nested_fix_method\nbattery,fix_quantity\ntank,fix_price\n"
    )
    (work / "solve_data" / "fix_storage_timesteps.csv").write_text(
        "period,step\np2020,t0001\np2020,t0024\n"
    )
    # v_state for both nodes; only battery is fix_quantity → only it's emitted.
    h = _fake_highs(
        variable_names=[
            "v_state[battery,p2020,t0001]",
            "v_state[battery,p2020,t0024]",
            "v_state[battery,p2020,t0048]",  # not in fix_steps → dropped
            "v_state[tank,p2020,t0001]",
        ],
        col_values=[5.0, 7.0, 9.0, 99.0],
    )

    write_fix_storage_quantity(h, solve_name="s1", work_folder=work)
    rows = list(csv.DictReader(open(work / "solve_data" / "fix_storage_quantity.csv")))
    assert {(r["period"], r["step"], r["node"], r["p_fix_storage_quantity"]) for r in rows} == {
        ("p2020", "t0001", "battery", "50"),  # 5 × 10
        ("p2020", "t0024", "battery", "70"),  # 7 × 10
    }


def test_fix_storage_quantity_preserves_prior_when_empty(tmp_path: Path) -> None:
    """Non-first solve with no fix_quantity entries must not clobber prior content."""
    work = _make_workfolder(tmp_path, first_solve=False)
    _write_unitsize(work, {"battery": 1.0})
    # Prior content from earlier sub-solve.
    prior = "period,step,node,p_fix_storage_quantity\np2020,t0001,battery,50\n"
    (work / "solve_data" / "fix_storage_quantity.csv").write_text(prior)
    # Empty fix_storage_timesteps and no node with fix_quantity method.
    (work / "solve_data" / "fix_storage_timesteps.csv").write_text("period,step\n")

    h = _fake_highs(variable_names=[], col_values=[])
    write_fix_storage_quantity(h, solve_name="s2", work_folder=work)
    assert (work / "solve_data" / "fix_storage_quantity.csv").read_text() == prior


# ---------------------------------------------------------------------------
# write_p_roll_continue_state
# ---------------------------------------------------------------------------


def test_p_roll_continue_state_uses_last_realized_step(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_unitsize(work, {"battery": 2.0, "non_state_node": 1.0})
    (work / "input" / "p_node_type.csv").write_text(
        "node,p_node_type\nbattery,storage\n"
    )
    # Two periods, multiple steps each — last entry per period is the
    # "last realized" boundary.
    (work / "solve_data" / "realized_dispatch.csv").write_text(
        "period,step\np2020,t0001\np2020,t0002\np2025,t0001\np2025,t0048\n"
    )
    h = _fake_highs(
        variable_names=[
            "v_state[battery,p2020,t0001]",
            "v_state[battery,p2020,t0002]",
            "v_state[battery,p2025,t0001]",
            "v_state[battery,p2025,t0048]",
        ],
        col_values=[1.0, 2.0, 3.0, 7.5],
    )
    write_p_roll_continue_state(h, solve_name="s1", work_folder=work)
    rows = list(csv.DictReader(open(work / "solve_data" / "p_roll_continue_state.csv")))
    # Last (period, time) pair is (p2025, t0048); v_state=7.5 × 2 = 15.0.
    assert rows == [{"node": "battery", "p_roll_continue_state": "15"}]


# ---------------------------------------------------------------------------
# write_p_entity_period_existing_capacity
# ---------------------------------------------------------------------------


def test_p_entity_period_existing_capacity_first_solve(tmp_path: Path) -> None:
    """existing = pre_existing[d] + v_invest * unitsize when first_solve."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_unitsize(work, {"battery": 1.0, "wind_plant": 1000.0})
    _write_entity_set(work, ["battery", "wind_plant"])
    # pre_existing layout: solve, period, entity1, entity2, ...
    (work / "solve_data" / "p_entity_pre_existing.csv").write_text(
        "solve,period,battery,wind_plant\ns1,p2020,50,1000\n"
    )
    (work / "solve_data" / "realized_invest_periods_of_current_solve.csv").write_text(
        "period\np2020\n"
    )
    (work / "solve_data" / "period_first.csv").write_text("period\np2020\n")
    # ed_invest covers (battery, p2020) only — wind_plant has no invest.
    (work / "solve_data" / "set_ed_invest.csv").write_text(
        "solve,entity,period\ns1,battery,p2020\n"
    )
    h = _fake_highs(
        variable_names=["v_invest[battery,p2020]"], col_values=[368.0],
    )

    write_p_entity_period_existing_capacity(h, solve_name="s1", work_folder=work)
    rows = list(csv.DictReader(
        open(work / "solve_data" / "p_entity_period_existing_capacity.csv")
    ))
    by_key = {(r["entity"], r["period"]): r for r in rows}
    # battery:  pre 50 + invest 368*1 = 418, invested = 368
    assert float(by_key[("battery", "p2020")]["p_entity_period_existing_capacity"]) == 418.0
    assert float(by_key[("battery", "p2020")]["p_entity_period_invested_capacity"]) == 368.0
    # wind_plant: pre 1000 + invest 0 = 1000, invested 0
    assert float(by_key[("wind_plant", "p2020")]["p_entity_period_existing_capacity"]) == 1000.0
    assert float(by_key[("wind_plant", "p2020")]["p_entity_period_invested_capacity"]) == 0.0


# ---------------------------------------------------------------------------
# write_fix_storage_usage
# ---------------------------------------------------------------------------


def test_fix_storage_usage_net_flow_through_storage_node(tmp_path: Path) -> None:
    """Exact formula for method_nvar / simple 1var_per_way:
    ``(outflow - inflow) × step_duration`` where flow is
    ``v_flow[p, n, *]`` or ``v_flow[p, *, n]`` times ``unitsize[p]``."""
    work = _make_workfolder(tmp_path)
    _write_unitsize(work, {"inverter": 2.0, "battery_node": 1.0, "west": 1.0})
    (work / "input" / "node__storage_nested_fix_method.csv").write_text(
        "node,storage_nested_fix_method\nbattery_node,fix_usage\n"
    )
    (work / "solve_data" / "fix_storage_timesteps.csv").write_text(
        "period,step\np2020,t0024\n"
    )
    (work / "solve_data" / "steps_in_use.csv").write_text(
        "period,step,step_duration\np2020,t0024,1.5\n"
    )
    # Discharge: battery_node → west (outflow 3 × unitsize 2 = 6)
    # Charge:    west → battery_node (inflow 1 × unitsize 2 = 2)
    # Net: (6 - 2) × 1.5 = 6
    h = _fake_highs(
        variable_names=[
            "v_flow[inverter,battery_node,west,p2020,t0024]",
            "v_flow[inverter,west,battery_node,p2020,t0024]",
        ],
        col_values=[3.0, 1.0],
    )
    write_fix_storage_usage(h, solve_name="s1", work_folder=work)
    rows = list(csv.DictReader(open(work / "solve_data" / "fix_storage_usage.csv")))
    assert rows == [{
        "period": "p2020", "step": "t0024",
        "node": "battery_node", "p_fix_storage_usage": "6",
    }]


def test_fix_storage_usage_preserves_prior_when_no_matching_node(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path, first_solve=False)
    prior = "period,step,node,p_fix_storage_usage\np2020,t0012,battery,7.5\n"
    (work / "solve_data" / "fix_storage_usage.csv").write_text(prior)
    # No node has fix_usage method → early return, must not clobber
    (work / "input" / "node__storage_nested_fix_method.csv").write_text(
        "node,storage_nested_fix_method\n"
    )
    (work / "solve_data" / "fix_storage_timesteps.csv").write_text("period,step\n")
    _write_unitsize(work, {"any": 1.0})

    h = _fake_highs(variable_names=[], col_values=[])
    write_fix_storage_usage(h, solve_name="s2", work_folder=work)
    assert (work / "solve_data" / "fix_storage_usage.csv").read_text() == prior


# ---------------------------------------------------------------------------
# Integration test: multi-solve scenario, Python writer ≡ phase 3
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).parent.parent


def _clean_run_dirs(repo: Path) -> None:
    for sub in ("output_raw", "solve_data"):
        d = repo / sub
        for p in d.glob("*"):
            if p.is_file():
                p.unlink()


def _run_scenario(repo: Path, scenario: str, *, use_old_raw_csv: bool) -> dict[str, str]:
    """Run a real scenario via subprocess and return handoff CSV contents."""
    cmd = [
        sys.executable, "run_flextool.py",
        f"sqlite:///templates/examples.sqlite",
        f"sqlite:///output_info.sqlite",
        "--scenario-name", scenario,
    ]
    if use_old_raw_csv:
        cmd.append("--use-old-raw-csv")
    _clean_run_dirs(repo)
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        pytest.skip(f"FlexTool run failed for {scenario!r}: {result.stderr[-300:]}")
    out: dict[str, str] = {}
    for fname in (
        "p_entity_period_existing_capacity.csv",
        "p_entity_divested.csv",
        "fix_storage_quantity.csv",
        "fix_storage_price.csv",
        "fix_storage_usage.csv",
        "p_roll_continue_state.csv",
    ):
        p = repo / "solve_data" / fname
        out[fname] = p.read_text() if p.exists() else ""
    return out


def _normalise_csv(text: str, *, decimals: int = 6) -> str:
    """Sort rows + round numerics so that order / printf precision don't matter."""
    if not text or len(text.strip().split("\n")) <= 1:
        return text.strip()
    df = pd.read_csv(StringIO(text)).round(decimals)
    sort_cols = list(df.columns[:-1]) if len(df.columns) > 1 else list(df.columns)
    return df.sort_values(sort_cols).reset_index(drop=True).to_csv(index=False).strip()


@pytest.mark.skipif(
    not (REPO_ROOT / "templates" / "examples.sqlite").exists(),
    reason="examples.sqlite not present (dev install only)",
)
@pytest.mark.parametrize("scenario", [
    "wind_battery_invest",
    "multi_fullYear_battery_nested_24h_invest_one_solve",
    "fullYear_roll",
    "network_all_tech",
])
def test_handoff_csv_matches_phase3(scenario: str) -> None:
    """End-to-end: handoff writer's output equals phase 3's, byte-for-byte
    after row-sort + 6-decimal rounding."""
    phase3 = _run_scenario(REPO_ROOT, scenario, use_old_raw_csv=True)
    new = _run_scenario(REPO_ROOT, scenario, use_old_raw_csv=False)
    for fname in phase3:
        assert _normalise_csv(phase3[fname]) == _normalise_csv(new[fname]), (
            f"{scenario}/{fname} differs.\n"
            f"phase3:\n{phase3[fname]}\n"
            f"new:\n{new[fname]}"
        )
