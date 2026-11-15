"""``reduce_unit_detail_method online`` keeps the full capacity of an indirect unit whose weighted output at full
capacity ``W`` is not 1 (``preprocess._reduce_detail._refold_input_caps``).

The case (``tests/model/_piecewise_oracle_db.py``, built from the schema of ``tests.json``): unit ``CHP``, gas
(20 CUR/MWh, input ``conversion_flow_coeff`` 1) -> elec (``conversion_flow_coeff`` 2) + heat (0.2), one 100-MW unit
(``virtual_unitsize`` 100, ``existing`` 1), binary start-ups at no cost, ``min_load`` 0.3; elec demand 100 MW and heat
demand 0 in both one-hour steps of p2030 (weight 8760 / 2 = 4380 per step).  ``W`` fills the outputs in
descending ``conversion_flow_coeff`` order up to the sum cap 1: elec 1 × 2 = 2, heat 0, so W = 2.  At 100 MW elec
the weighted output is 2 × 100 = 200 MW on 1 unit online.

Hand values (exact fractions).

Piecewise curve (0.3, 0.25), (0.6, 0.35), (1.0, 0.40): F(ℓ) = ℓ/η = 6/5, 12/7, 5/2.

* Online form: seg 1 α = (12/7 − 6/5)/0.3 = 12/7, β = 6/5 − 0.3·12/7 = 24/35; seg 2 α = (5/2 − 12/7)/0.4 = 55/28,
  β = 5/2 − 55/28 = 15/28 (η rises, no origin line).  Input limit per unit of capacity
  max(12/7·2 + 24/35, 55/28·2 + 15/28) = max(144/35, 125/28) = 125/28.  At 200 MW weighted output, 1 unit online:
  fuel = max(12/7·200 + 24/35·100, 55/28·200 + 15/28·100) = 12500/28 = 3125/7 MW = 125/28 × 100: within the limit.
  Cost 3125/7 × 2 × 4380 × 20 = 547500000/7.
* Reduced (capacity form): origin line 1/max η = 5/2, no segment with a negative intercept.  Fuel 5/2 × 200 =
  500 MW; input limit 5/2 × W = 5 per unit of capacity (500 MW).  The kept online-form limit 125/28 would leave
  125/28 × 100 / 5 = 625/7 ≈ 89.29 MW of elec and 75/7 ≈ 10.71 MW to slack.  Cost 500 × 2 × 4380 × 20 = 87600000.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import polars as pl
import pytest

from tests.engine._unit_db import utf8
from tests.model._piecewise_oracle_db import (
    balance_node, commodity_node, curve, make_oracle_db, merge, scalar, text, time_settings, unit,
)

pytestmark = pytest.mark.solver

RTOL = 1e-9


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _case(level: str | None) -> tuple[list, list]:
    extra = [["solve", "s1", "reduce_unit_detail_method", text(level)]] if level else []
    ents, vals = merge(
        time_settings(2, ["p2030"], extra_solve=extra), commodity_node("gas", "gas_node", 20.0),
        balance_node("elec", {"t01": -100.0, "t02": -100.0}), balance_node("heat", {"t01": 0.0, "t02": 0.0}),
        unit("CHP", ["gas_node"], ["elec", "heat"], {
            "conversion_method": text("piecewise_linear"),
            "piecewise_linear_efficiency": curve([(0.3, 0.25), (0.6, 0.35), (1.0, 0.40)]),
            "virtual_unitsize": scalar(100.0), "existing": scalar(1.0), "startup_method": text("binary"),
            "startup_cost": scalar(0.0), "min_load": scalar(0.3)}))
    vals += [["unit__outputNode", ["CHP", "elec"], "conversion_flow_coeff", scalar(2.0)],
             ["unit__outputNode", ["CHP", "heat"], "conversion_flow_coeff", scalar(0.2)]]
    return ents, vals


def _run(tmp_path_factory, name: str, level: str | None):
    from flextool.engine import run_chain_from_db

    url, scen = make_oracle_db(tmp_path_factory, name, *_case(level))
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{name}_w"), keep_solutions=True)
    assert len(steps) == 1, list(steps)
    step = next(iter(steps.values()))
    assert step.solution is not None
    return step


def _mw(step, source: str, sink: str) -> list[float]:
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == "CHP") & (pl.col("source") == source) & (pl.col("sink") == sink)).sort("d", "t"))
    return [v * 100.0 for v in rows["value"].to_list()]


def _input_rows(param) -> list[float]:
    return (utf8(param.frame).filter((pl.col("p") == "CHP") & (pl.col("source") == "gas_node"))
            .sort("d", "t")["value"].to_list())


@pytest.mark.parametrize("level, fuel, cap, obj", [
    (None, Fr(3125, 7), Fr(125, 28), Fr(547500000, 7)),
    ("online", Fr(500), Fr(5), Fr(87600000)),
])
def test_piecewise_reduced_unit_keeps_its_capacity(tmp_path_factory, level, fuel, cap, obj) -> None:
    """Reduced and unreduced serve the whole 100 MW; the reduced input limit is 5/2 × W = 5 (module docstring)."""
    s = _run(tmp_path_factory, f"pw_cap_{level or 'none'}", level)
    assert _mw(s, "CHP", "elec") == pytest.approx([100.0, 100.0], rel=RTOL)
    slack = utf8(s.solution.value("vq_state_up")).filter(pl.col("n") == "elec")["value"].to_list()
    assert slack == pytest.approx([0.0, 0.0], abs=1e-9)
    assert _mw(s, "gas_node", "CHP") == pytest.approx([float(fuel)] * 2, rel=RTOL)
    assert _input_rows(s.flex_data.p_indirect_input_cap) == pytest.approx([float(cap)] * 2, rel=RTOL)
    # one unit of capacity: the arc's structural upper bound is the limit itself
    assert _input_rows(s.flex_data.p_flow_upper) == pytest.approx([float(cap)] * 2, rel=RTOL)
    assert float(s.solution.obj) == pytest.approx(float(obj), rel=RTOL)
