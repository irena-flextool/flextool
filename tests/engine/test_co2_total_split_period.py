"""``co2_max_total`` in a chain whose solves each hold part of one period (item 1042, D49): the budget is split by the
hour weight of each solve's window, so the chain's realized emissions equal a single solve's limit.

Model: ``test_flowgroup_cumulative_carry._spec`` with ``chain`` and ``split`` 3: solve ``s1`` holds p1 on t01-t03
(period share 3 / 8760, objective factor 2920), ``s2`` p1 on t04 (share 1 / 8760, factor 8760).  ``coal_u`` draws
coal (1 t CO2 per MWh, price 0) from ``coal_n`` in group ``co2g``; demand 1 MW; coal 10 EUR/MWh.
``co2_max_total`` 2190 t: a single solve of p1 on t01-t04 caps E·8760 / 4 ≤ 2190, i.e. 1 t over the 4 h.

s1 holds its 3 h as the whole period (f_d_k = 3 / 3 = 1): E_1·2920 ≤ 2190, so E_1 ≤ 0.75 t.  s2 reaches
f_d_k = (3 realized + 1) / 1 = 4: its share is the hours reached, 4 / 8760, so its hour weighs 8760 / 4 = 2190, and
s1's realized tonnes are weighed the same: (E_1 + E_2)·2190 ≤ 2190, E_2 ≤ 1 − E_1.  Before, s2 weighed its hour
8760 and s1's tonnes 2920: E_2·8760 ≤ 2190 − E_1·2920.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine import test_flowgroup_cumulative_carry as fc

pack = fc.pack
pytestmark = pytest.mark.solver


def _co2_chain(gas_s1: float) -> dict:
    spec = fc._spec(gas={"t01": gas_s1, "t02": gas_s1, "t03": gas_s1, "t04": 40.0}, coal=10.0, limits=[],
                    chain=True, fuel_arc=True, split=3.0)
    spec["entities"] = [e for e in spec["entities"] if not e[0].startswith("flowGroup")]
    spec["entities"] += [["group", "co2g", None], ["group__node", ["co2g", "coal_n"], None]]
    for v in spec["parameter_values"]:
        if v[:3] == ["unit", "coal_u", "efficiency"]:
            v[3] = pack(1.0, "float")
    spec["parameter_values"] += [["commodity", "coal", "co2_content", pack(1.0, "float"), "init"],
                                 ["group", "co2g", "co2_method", pack("total", "str"), "init"],
                                 ["group", "co2g", "co2_max_total", pack(2190.0, "float"), "init"]]
    return spec


def _coal(step) -> float:
    """MWh of coal_u's output in *step* (unit size 10 MW, 1 h steps)."""
    frame = step.solution.value("v_flow").with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "source"))
    rows = frame.filter((pl.col("p") == "coal_u") & (pl.col("source") == "coal_n"))
    return float(rows["value"].sum()) * 10.0


def test_split_period_budget_is_shared_by_hour_weight(tmp_path):
    """Gas 40 everywhere: coal is cheaper, so s1 burns its 0.75 t and s2 the remaining 0.25 t (1 t in total).
    s1: 0.75 x 10 + 2.25 x 40 = 97.5 -> x 2920 = 284 700.  s2: 0.25 x 10 + 0.75 x 40 = 32.5 -> x 8760 = 284 700
    (before: s2's RHS 2190 - 0.75 x 2920 = 0, gas only, 40 -> 350 400; 0.75 t in total).  The carrier s2 hands on
    weighs the period's 1 t by s2's share: represented 1 x 1 / (4 / 8760) = 2190 (before: 0.75 x 2920 + 0 = 2190
    from s1 alone, which a later solve would add to 0.25 x 8760 of a burning s2)."""
    steps = fc._run(_co2_chain(40.0), tmp_path)
    assert len(steps) == 2
    assert [_coal(s) for s in steps] == pytest.approx([0.75, 0.25], abs=1e-9)
    assert [s.solution.obj for s in steps] == pytest.approx([97.5 * 2920.0, 32.5 * 8760.0], rel=1e-9)
    carried = steps[1].handoff.cumulative_co2.select(pl.col("period").cast(pl.Utf8), "value", "represented").rows()
    assert carried == [("p1", pytest.approx(1.0, abs=1e-9), pytest.approx(2190.0, rel=1e-9))]


def test_split_period_unused_budget_carries_to_the_later_part(tmp_path):
    """Gas 5 in s1: s1 burns no coal, 3 x 5 = 15 -> 43 800.  s2 may burn 1 - 0 = 1 t: coal 1, 10 -> 87 600 (before:
    E_2 x 8760 <= 2190, 0.25 t: 2.5 + 30 = 32.5 -> 284 700)."""
    steps = fc._run(_co2_chain(5.0), tmp_path)
    assert [_coal(s) for s in steps] == pytest.approx([0.0, 1.0], abs=1e-9)
    assert [s.solution.obj for s in steps] == pytest.approx([15.0 * 2920.0, 10.0 * 8760.0], rel=1e-9)
