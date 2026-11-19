"""The total CO2 price of a stochastic solve (item 183).

A stochastic solve with several scenario paths has one cumulative CO2 cap row per path
(``co2_max_total_path[g, leaf]``, each with the whole budget and only its own path's emissions) instead of the
one ``co2_max_total[g]`` row of a deterministic solve, and the output used to read only the latter:
``co2_price_total_d_g`` was missing.  It is now the price of one more tonne of the budget: one tonne more raises
every path's row, so the price is the sum of the path duals (each carries its path's branch weight).

Oracle (relational, independent of the new code): a stochastic solve whose two forecasts (weights 1 : 3) are
identical is the deterministic solve twice over, weighted 0.25 / 0.75 — the same dispatch, the same objective,
and so the same marginal value of the budget.  Its ``co2_price_total_d_g`` must equal the deterministic solve's
(read through the existing ``co2_max_total`` dual, unchanged by item 183), and each path's raw dual must be its
branch weight times the deterministic dual (the weighted-copies identity, which also pins that the paths are
summed, not averaged or taken from one path).

Model (``stoch_two_period_midhorizon_nreal0`` shape, investment removed): demand 120 MW, wind 60 MW in p2035 and
85 MW in p2040 (both forecasts the same), the residual from ``base`` (gas, 0.5 tCO2/MWh, 11 CUR/MWh) or
``clean`` (no CO2, 41 CUR/MWh).  Base-only emissions are 60·0.5·8760 + 35·0.5·8760 = 416 100 t against a
300 000 t budget, so the cap binds and ``base`` runs part-load (an interior point: the dual is unique); the
price is 30 CUR/MWh / 0.5 t/MWh = 60 CUR/t (the model has no discounting).  The fan's forecast period p2040 is
not realized, so the stochastic output has the p2035 row only.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "flextool" / "schemas" / "default_plots.yaml")
FIXTURE = TEST_DIR / "fixtures" / "stoch_two_period_midhorizon_nreal0.json"
SOLVE = "stoch_2p_mid"
GROUP = "co2g"
CAP = 300_000.0
WEIGHTS = {"high": 0.25, "low": 0.75}

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine import run_chain_from_db  # noqa: E402
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402


def _b64(value: object, value_type: str) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode("ascii"), value_type]


_INVEST = {"invest_method", "invest_cost", "discount_rate", "lifetime_technical", "invest_max_total",
           "virtual_unitsize"}


def _fixture(tmp: Path) -> Path:
    """The mid-horizon fan with a CO2 budget, no investment and an ``det`` control scenario."""
    data = json.loads(FIXTURE.read_text())
    data["parameter_values"] = [
        pv for pv in data["parameter_values"]
        if not (pv[0] == "unit" and pv[1] == "base" and pv[2] in _INVEST | {"existing"})
        and not (pv[0] == "solve" and pv[2] in {"invest_periods", "realized_invest_periods"})]
    data["alternatives"].append(["co2", "CO2 budget, gas base unit, clean backstop"])
    data["entities"] += [
        ["commodity", "gas", None], ["node", "gas_node", None], ["unit", "clean", None],
        ["group", GROUP, None],
        ["commodity__node", ["gas", "gas_node"], None],
        ["unit__inputNode", ["base", "gas_node"], None],
        ["unit__outputNode", ["clean", "elec"], None],
        ["group__node", [GROUP, "gas_node"], None],
    ]
    data["entity_alternatives"] += [["node", ["gas_node"], "init", True], ["unit", ["clean"], "init", True]]
    data["parameter_values"] += [
        ["unit", "base", "existing", _b64(200.0, "float"), "co2"],
        ["unit", "clean", "existing", _b64(200.0, "float"), "co2"],
        ["unit", "clean", "efficiency", _b64(1.0, "float"), "co2"],
        ["unit__outputNode", ["clean", "elec"], "other_operational_cost", _b64(41.0, "float"), "co2"],
        ["node", "gas_node", "node_type", _b64("commodity", "str"), "co2"],
        ["commodity", "gas", "price", _b64(10.0, "float"), "co2"],
        ["commodity", "gas", "co2_content", _b64(0.5, "float"), "co2"],
        ["group", GROUP, "co2_method", _b64("total", "str"), "co2"],
        ["group", GROUP, "co2_max_total", _b64(CAP, "float"), "co2"],
    ]
    # det = the same model without the fan; its p2040 wind is the forecasts' plain 0.85 profile.
    data["alternatives"].append(["plain", "The forecasts' plain wind profile, no fan"])
    prof = next(pv for pv in data["parameter_values"]
                if pv[0] == "profile" and pv[2] == "profile" and pv[4] == "stoch")
    data["parameter_values"].append([*prof[:4], "plain"])
    data["scenarios"] += [["co2_det", False, "Deterministic control"], ["co2_stoch", False, "Identical forecasts"]]
    data["scenario_alternatives"] += [
        ["co2_det", "init", "base"], ["co2_det", "base", "plain"], ["co2_det", "plain", "co2"],
        ["co2_det", "co2", None],
        ["co2_stoch", "init", "base"], ["co2_stoch", "base", "stoch"], ["co2_stoch", "stoch", "co2"],
        ["co2_stoch", "co2", None],
    ]
    out = tmp / "stoch_co2_total.json"
    out.write_text(json.dumps(data))
    return out


def _run(url: str, scenario: str, wf: Path, solver_config_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Run ``scenario`` through the full output pipeline; ``co2_price_total_d_g`` and the raw output folder."""
    import os

    os.chdir(wf)
    steps = run_chain_from_db(url, scenario, work_folder=wf, solver_config_dir=solver_config_dir,
                              csv_dump=True, keep_solutions=True)
    last = next(reversed(steps.values()))
    assert last.solution is not None and last.solution.optimal
    provider = getattr(last, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(wf)
    write_outputs(
        scenario_name=scenario, output_location=str(wf), subdir=scenario,
        output_config_path=OUTPUT_CONFIG, write_methods=["csv"], fallback_output_location=str(wf),
        raw_output_dir=str(wf / "output_raw"), solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider(),
    )
    # co2_price_total_d_g is written as group_CO2_price_total__d (default_plots.yaml).
    price = pd.read_csv(wf / "output_csv" / scenario / "group_CO2_price_total__d.csv").set_index("period")
    return price, wf / "output_raw"


def _raw_dual(raw: Path, name: str) -> pd.DataFrame:
    from flextool.lean_parquet import read_lean_parquet

    return read_lean_parquet(raw / f"{name}__{SOLVE}.parquet")


@pytest.fixture(scope="module")
def runs(tmp_path_factory, test_solver_config_dir) -> dict:
    from flextool.update_flextool.db_migration import migrate_database

    tmp = tmp_path_factory.mktemp("co2_total_price")
    url = json_to_db(_fixture(tmp), tmp / "stoch_co2_total.sqlite")
    migrate_database(url)
    return {scen: _run(url, scen, tmp_path_factory.mktemp(scen), test_solver_config_dir)
            for scen in ("co2_det", "co2_stoch")}


@pytest.mark.slow
@pytest.mark.solver
def test_stochastic_total_co2_price_equals_the_deterministic_one(runs) -> None:
    """Identical forecasts: the stochastic ``co2_price_total_d_g`` is the deterministic one, period by period."""
    det, _ = runs["co2_det"]
    stoch, _ = runs["co2_stoch"]
    assert GROUP in det.columns and det[GROUP].abs().min() > 1.0   # the cap binds (positive price)
    assert GROUP in stoch.columns, "no total CO2 price in the stochastic solve (item 183)"
    common = det.index.intersection(stoch.index)
    assert len(common) > 0
    for d in common:
        assert stoch.at[d, GROUP] == pytest.approx(det.at[d, GROUP], rel=1e-6), d


@pytest.mark.slow
@pytest.mark.solver
def test_each_path_dual_is_its_branch_weight_times_the_deterministic_dual(runs) -> None:
    """Weighted copies of the deterministic LP: path b's raw dual = w_b · the deterministic raw dual, so the
    price (minus their sum) is the deterministic price, not an average or one path's share."""
    _, det_raw = runs["co2_det"]
    _, stoch_raw = runs["co2_stoch"]
    det_dual = float(_raw_dual(det_raw, "v_dual_co2_max_total")[GROUP].iloc[0])
    path = _raw_dual(stoch_raw, "v_dual_co2_max_total_path")
    assert det_dual < 0.0                                       # a binding <= cap: negative raw dual
    by_leaf = {c[1]: float(path[c].iloc[0]) for c in path.columns if c[0] == GROUP}
    assert set(by_leaf) == set(WEIGHTS)
    for leaf, w in WEIGHTS.items():
        assert by_leaf[leaf] == pytest.approx(w * det_dual, rel=1e-6), leaf
    assert not (stoch_raw / f"v_dual_co2_max_total__{SOLVE}.parquet").exists() or \
        _raw_dual(stoch_raw, "v_dual_co2_max_total").empty


def test_fold_sums_the_paths_and_keeps_solve_order() -> None:
    """Unit check of the reader fold: per-path duals are summed per group; a chain that mixes a stochastic and a
    deterministic solve keeps one row per solve in creation order (hand values)."""
    from flextool.process_outputs.read_variables import _fold_co2_total_path

    total = pd.DataFrame({"g1": [-5.0, 0.0]}, index=pd.Index(["s0", "s1"], name="solve"))
    total.columns.name = "group"
    path = pd.DataFrame([[-1.0, -3.0, -2.0]], index=pd.Index(["s1"], name="solve"),
                        columns=pd.MultiIndex.from_tuples([("g1", "a"), ("g1", "b"), ("g2", "a")],
                                                          names=["group", "leaf"]))
    out = _fold_co2_total_path(total, path, {"s0": 0, "s1": 1})
    assert list(out.index) == ["s0", "s1"]
    assert out.loc["s0", "g1"] == -5.0
    assert out.loc["s1", "g1"] == -4.0 and out.loc["s1", "g2"] == -2.0
    # The concat of a chain's path parquets gives a deterministic solve an all-NaN row (read without
    # zero-filling): it keeps its own total row.
    with_det = pd.concat([pd.DataFrame(float("nan"), index=pd.Index(["s0"], name="solve"),
                                       columns=path.columns), path])
    out = _fold_co2_total_path(total, with_det, {"s0": 0, "s1": 1})
    assert out.loc["s0", "g1"] == -5.0 and out.loc["s1", "g1"] == -4.0
    # A deterministic run (no path parquet) is untouched.
    assert _fold_co2_total_path(total, path.iloc[0:0, 0:0], {}) is total

