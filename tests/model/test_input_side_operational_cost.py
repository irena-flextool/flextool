"""``other_operational_cost`` on the input side of a unit with an efficiency is a cost.

docs/reference/units.md: ``other_operational_cost`` on unit--inputNode / unit--outputNode is
"[CUR/MWh] Other operational variable costs for energy flows".  For a unit with a constant
efficiency, an input-side cost ``c`` per MWh of fuel is economically the same as an
output-side cost ``c·slope`` per MWh of output (``slope = 1 / efficiency``).  The objective
inherited from GMPL subtracted the input-side term (``variable_cost_eff_source``), so a
higher input O&M made the unit cheaper to run.

``coal_plant`` (``coal_market -> west``, efficiency 0.4, no ``min_load_efficiency`` section in
the ``coal`` chain) is the direct unit under test.  DBs are ``tests.json`` plus an override
alternative (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import COAL, PLANT, PLANT_CHAIN, b64, make_db, utf8

INPUT_COST = 5.0
FLOW_COLUMN = f"v_flow[{PLANT},{COAL},west,"


def _run_capturing_lp(tmp_path_factory, alt: str, values: list):
    """Solve ``tests.json`` + ``values`` and return ``(step, {column: objective coefficient})``.

    The objective coefficients of the ``coal_plant`` input-flow columns are read from the
    Problem right after ``build_flextool`` returns (before autoscale), written as MPS.
    """
    import flextool.engine.model.core as core
    from flextool.engine import run_chain_from_db

    url, scenario = make_db(tmp_path_factory, alt=alt, values=values, base_chain=PLANT_CHAIN)
    mps_dir = tmp_path_factory.mktemp(f"{alt}_mps")
    built: list = []
    original = core.build_flextool

    def capturing(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        path = mps_dir / f"build_{len(built)}.mps"
        m.write_mps(path, emit_names=True)
        built.append(path)
        return result

    mp = pytest.MonkeyPatch()
    mp.setattr(core, "build_flextool", capturing)
    try:
        steps = run_chain_from_db(url, scenario, work_folder=tmp_path_factory.mktemp(f"{alt}_w"),
                                  keep_solutions=True, warm=False)
    finally:
        mp.undo()
    assert len(steps) == 1 and len(built) == 1, (list(steps), built)
    step = next(iter(steps.values()))
    assert step.solution is not None
    return step, _objective_coefficients(built[0])


def _objective_coefficients(mps_path) -> dict[str, float]:
    """``{column: cost coefficient}`` for the coal_plant input-flow columns."""
    coeffs: dict[str, float] = {}
    in_columns = False
    for line in mps_path.read_text().splitlines():
        if line.startswith("COLUMNS"):
            in_columns = True
            continue
        if in_columns and not line.startswith(" "):
            break
        if not in_columns:
            continue
        parts = line.split()
        if parts[0].startswith(FLOW_COLUMN):
            for row, value in zip(parts[1::2], parts[2::2]):
                if row == "cost":
                    coeffs[parts[0]] = float(value)
    return coeffs


def _flow(step) -> pl.Series:
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == PLANT) & (pl.col("source") == COAL) & (pl.col("sink") == "west"))
            .sort("d", "t"))
    assert rows.height > 0
    return rows["value"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base_step, base_coef = _run_capturing_lp(tmp_path_factory, "plant_om_base", [])
    slopes = (utf8(base_step.flex_data.p_slope.frame).filter(pl.col("p") == PLANT)["value"]
              .unique().to_list())
    assert len(slopes) == 1, f"coal_plant slope is not constant: {slopes}"
    slope = float(slopes[0])
    assert slope == pytest.approx(2.5)
    assert base_step.flex_data.process_min_load_eff is None or (
        PLANT not in utf8(base_step.flex_data.process_min_load_eff)["p"].to_list())
    in_step, in_coef = _run_capturing_lp(tmp_path_factory, "plant_om_input", [
        ["unit__inputNode", [PLANT, COAL], "other_operational_cost", b64(INPUT_COST, "float")]])
    out_step, out_coef = _run_capturing_lp(tmp_path_factory, "plant_om_output", [
        ["unit__outputNode", [PLANT, "west"], "other_operational_cost",
         b64(INPUT_COST * slope, "float")]])
    return {"base": (base_step, base_coef), "input": (in_step, in_coef),
            "output": (out_step, out_coef)}


def test_input_cost_raises_flow_objective_coefficient(runs):
    """Solve-independent pin: input cost c adds c·slope·unitsize·w_op to the coefficient of
    the input flow, exactly what the output-side cost c·slope adds."""
    _, base = runs["base"]
    _, with_input = runs["input"]
    _, with_output = runs["output"]
    assert base and set(base) == set(with_input) == set(with_output)
    for col, b in base.items():
        added_in = with_input[col] - b
        added_out = with_output[col] - b
        assert added_out > 0.0, col
        assert added_in == pytest.approx(added_out, rel=1e-12), col


def test_input_cost_equals_output_cost_times_slope(runs):
    base_step, _ = runs["base"]
    in_step, _ = runs["input"]
    out_step, _ = runs["output"]
    assert in_step.obj > base_step.obj
    assert in_step.obj == pytest.approx(out_step.obj, rel=1e-9)
    flow_in, flow_out = _flow(in_step), _flow(out_step)
    assert flow_in.len() == flow_out.len()
    assert (flow_in - flow_out).abs().max() == pytest.approx(0.0, abs=1e-6)
