"""A unit ``efficiency`` above 1 is a plain output / input ratio (C-6).

``efficiency`` is not bounded to 1: a heat pump with COP 3 has efficiency 3, and a desalination unit with an
electricity input and a water output in m³ has the m³ per MWh ratio (e.g. 250).  No engine code clamps or rounds
it; this pins the slope (single and rolling seed paths), the solve under strict autoscale and
the written flows, including the large ratio.

Hand-derived oracle (direct unit ``coal_plant``: input ``coal_market``, output ``demand_node``,
``constant_efficiency``, conversion coefficients 1):

    p_slope = 1 / efficiency                         (3 -> 1/3, 250 -> 0.004)
    input[d, t] = output[d, t] / efficiency          (every reported time step)

Checked in the rolling solve (synthetic rolls seed ``p_slope`` from the CSV cascade) and in the single solve of
the ``1week_rolling`` vehicle of ``test_rolling_period_maps.py``, with ``FLEXTOOL_AUTOSCALE_STRICT=1`` (an
autoscale registry gap fails instead of falling back to an unscaled LP).  The input and output flows are read
from the written ``unit_inputNode_dt_ee`` / ``unit_outputNode_dt_ee`` parquet results
(full precision; the CSV results round to 6 decimals).
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from tests.engine._unit_db import b64, make_db
from tests.model.test_rolling_period_maps import ROLL_CHAIN, _deterministic_rolling_json, _value, scalar

COAL = "coal_plant"
COAL_IN = "coal_market"
COAL_OUT = "demand_node"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _solve_and_write(tmp_path_factory, eta: float, single: bool):
    """Steps of the solve and the folder of its written CSV results."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    alt = f"c6_eta{int(eta)}_{'single' if single else 'rolling'}"
    values = [["unit", COAL, "efficiency", scalar(eta)]]
    if single:
        values.append(["solve", "1week_rolling", "solve_mode", b64("single_solve", "str")])
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=ROLL_CHAIN,
                        fixture_json=_deterministic_rolling_json(tmp_path_factory, alt))
    work = tmp_path_factory.mktemp(f"{alt}_w")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
        last = next(reversed(steps.values()))
        write_outputs(
            scenario_name=scen, output_location=str(work), subdir=scen,
            write_methods=["parquet"], fallback_output_location=str(work),
            raw_output_dir=str(work / "output_raw"),
            solution=last.solution, solve_name=last.solve_name,
            solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
            flex_data_provider=last.flex_data_provider,
        )
    return steps, work / "output_parquet" / scen


@pytest.mark.parametrize("single", [False, True], ids=["rolling", "single"])
@pytest.mark.parametrize("eta", [3.0, 250.0], ids=["cop3", "m3_per_mwh_250"])
def test_efficiency_above_one_gives_input_output_over_efficiency(tmp_path_factory, eta, single) -> None:
    steps, out_dir = _solve_and_write(tmp_path_factory, eta, single)
    assert len(steps) == (1 if single else 7)
    for name, step in steps.items():
        assert step.optimal, name
        assert _value(step.flex_data.p_slope, p=COAL) == {round(1.0 / eta, 9)}, name

    def flow(name: str, node: str) -> pd.Series:
        frame = pd.read_parquet(out_dir / f"{name}.parquet").set_index(["period", "time"])
        return frame[str((out_dir.name, COAL, node))].astype(float)

    fuel = flow("unit_inputNode_dt_ee", COAL_IN).abs()
    out = flow("unit_outputNode_dt_ee", COAL_OUT)
    assert len(out) == 168 and list(fuel.index) == list(out.index)
    assert out.sum() > 0.0, "the unit must produce for the ratio to be tested"
    assert fuel.to_list() == pytest.approx((out / eta).to_list(), rel=1e-9, abs=1e-9)
