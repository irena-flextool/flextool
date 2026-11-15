"""The written results of piecewise_linear units and the post-solve warning W5 (design §5.1, chunk 4).

The oracle cases (``specs/piecewise_efficiency_oracle.md``) are solved as in ``test_piecewise_oracles.py`` and their
results written (parquet) the way the run does: once with every step's solution in memory (``solve_steps``) and once
from the per-roll slices persisted under ``output_raw/`` (the normal run, no ``solve_steps``).  Expected values are
the oracle's hand calculation (exact fractions; rtol 1e-7):

* O1 (thermal, integer online): the fuel input per step 1450/9, 3025/9, 10000/21 MWh, the output 60 / 130 / 200 MW,
  the units online 1, 2, 2; the CO2 of the period 0.2 t/MWh × Σq × 2920 = 0.2 · 61325/63 · 2920 t, the fuel cost
  10 CUR/MWh · Σq · 2920 / 1e6 M CUR (the step weight 2920 of oracle §0.1).  No W5 (the fuel has a price).
* O3 (disposal): W5 for E — 3 steps above the curve, the largest 1220/77 MWh, the total 3660/77 MWh (oracle §9; the
  step of the largest is a tie over t01-t03, §0.2: not asserted).
* O3-c (flat curve = constant efficiency): no W5 (single-line units keep the equality).
"""
from __future__ import annotations

import logging
import warnings
from fractions import Fraction as Fr

import pandas as pd
import pytest

from tests.model._piecewise_oracle_db import make_oracle_db
from tests.model.test_piecewise_oracles import CASES

pytestmark = pytest.mark.solver

RTOL = 1e-7
_W5_LOGGER = "flextool.process_outputs.calc_piecewise"
_RUNS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _write(tmp_path_factory, name: str, *, in_memory: bool, caplog):
    """Solve the case and write its parquet results; ``(steps, parquet folder, W5 findings, W5 messages)``."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs import calc_piecewise
    from flextool.process_outputs.write_outputs import write_outputs

    key = (name, in_memory)
    if key in _RUNS:
        return _RUNS[key]
    ents, vals = CASES[name]()
    label = f"{name.replace('-', '_')}_{'mem' if in_memory else 'persisted'}"
    url, scen = make_oracle_db(tmp_path_factory, label, ents, vals)
    work = tmp_path_factory.mktemp(f"{label}_w")
    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=in_memory)
    last = next(reversed(steps.values()))
    found: list = []
    real = calc_piecewise.warn_piecewise_excess

    def spy(*args, **kwargs):
        found.extend(real(*args, **kwargs))
        return found

    caplog.clear()
    caplog.set_level(logging.WARNING, logger=_W5_LOGGER)
    extra = ({"solve_steps": [(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()]}
             if in_memory else {"flex_data": last.flex_data})
    with warnings.catch_warnings(), pytest.MonkeyPatch.context() as mp:
        warnings.simplefilter("ignore")
        mp.setattr(calc_piecewise, "warn_piecewise_excess", spy)
        write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["parquet"],
                      fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                      solution=last.solution, solve_name=last.solve_name,
                      flex_data_provider=last.flex_data_provider, **extra)
    messages = [r.getMessage() for r in caplog.records if r.name == _W5_LOGGER]
    _RUNS[key] = (steps, work / "output_parquet" / scen, found, messages)
    return _RUNS[key]


def _column(out_dir, table: str, *key: str) -> list[float]:
    frame = pd.read_parquet(out_dir / f"{table}.parquet")
    col = str((out_dir.name, *key))
    assert col in frame.columns, f"{table}: no column {col} in {list(frame.columns)}"
    return frame[col].astype(float).to_list()


def _close(got: list[float], want: list, *, what: str) -> None:
    assert len(got) == len(want), f"{what}: {got} vs {want}"
    for g, w in zip(got, want):
        assert g == pytest.approx(float(w), rel=RTOL), f"{what}: {got} vs {want}"


@pytest.mark.parametrize("in_memory", [True, False], ids=["in_memory", "persisted"])
def test_O1_result_tables(tmp_path_factory, caplog, in_memory) -> None:
    _steps, out, found, messages = _write(tmp_path_factory, "O1", in_memory=in_memory, caplog=caplog)
    q = [Fr(1450, 9), Fr(3025, 9), Fr(10000, 21)]
    _close([-v for v in _column(out, "unit_inputNode_dt_ee", "G", "fuel_node")], q, what="fuel input [MW]")
    _close(_column(out, "unit_outputNode_dt_ee", "G", "elec"), [60, 130, 200], what="output [MW]")
    _close(_column(out, "unit_online_dt_e", "G"), [1, 2, 2], what="units online")
    _close(_column(out, "process_co2_d_eee", "unit", "G", "fuel", "fuel_node"), [Fr(2, 10) * sum(q) * 2920],
           what="CO2 [t]")
    cost = pd.read_parquet(out / "cost_unit_annualized_d_ec.parquet")
    _close(cost[str((out.name, "commodity_cost"))].astype(float).to_list(), [10 * sum(q) * 2920 / Fr(10**6)],
           what="fuel cost [M CUR]")
    assert found == [] and messages == []


@pytest.mark.parametrize("in_memory", [True, False], ids=["in_memory", "persisted"])
def test_W5_O3_input_above_the_curve(tmp_path_factory, caplog, in_memory) -> None:
    _steps, _out, found, messages = _write(tmp_path_factory, "O3", in_memory=in_memory, caplog=caplog)
    assert [f.unit for f in found] == ["E"]
    f = found[0]
    assert f.steps == 3
    _close([f.largest, f.total], [Fr(1220, 77), Fr(3660, 77)], what="largest, total [MWh]")
    assert len(messages) == 1
    msg = messages[0]
    assert msg.startswith(f"Unit 'E' used more input than its curve needs in 3 time steps (largest "
                          f"{float(Fr(1220, 77)):.4g} MWh in p2030/")
    assert f"total {float(Fr(3660, 77)):.4g} MWh): its input had no value or a negative value there" in msg


def test_W5_not_for_the_flat_curve_O3_c(tmp_path_factory, caplog) -> None:
    steps, _out, found, messages = _write(tmp_path_factory, "O3-c", in_memory=True, caplog=caplog)
    fd = next(iter(steps.values())).flex_data
    assert fd.process_piecewise is None or fd.process_piecewise.height == 0
    assert found == [] and messages == []
