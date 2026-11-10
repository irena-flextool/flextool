"""The ``solve`` column of the per-period output tables (#210).

Before, a rolling solve's period tables carried the label of the roll that happened to realize the period last
(``dispatch_fullYear_roll_roll_17``).  Now a period is labelled by the last complete solve that realized its
dispatch, and the investment tables (capacities, investment marginal values) by the solve that realized its
investments.
"""
from __future__ import annotations

import pandas as pd

from flextool.process_outputs.solve_order import complete_solve_name, period_solve_labels


def _sp(rows: list[tuple[str, str]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(rows, names=["solve", "period"])


def test_roll_suffix_is_removed() -> None:
    assert complete_solve_name("dispatch_fullYear_roll_roll_17") == "dispatch_fullYear_roll"
    assert complete_solve_name("y2030_dispatch") == "y2030_dispatch"
    assert complete_solve_name("s__realized_dispatch_roll_3") == "s__realized_dispatch"


def test_a_period_realized_by_many_rolls_names_the_complete_solve() -> None:
    sp = _sp([(f"disp_roll_{k}", "p2020") for k in range(18)] + [("disp_roll_18", "p2025")])
    assert period_solve_labels(sp) == {"p2020": "disp", "p2025": "disp"}


def test_the_last_complete_solve_wins_in_cascade_order() -> None:
    # Cascade order, not name order: 'b' realizes p1 after 'z'.
    sp = _sp([("z_roll_0", "p1"), ("z_roll_1", "p1"), ("b", "p1"), ("b", "p2")])
    assert period_solve_labels(sp) == {"p1": "b", "p2": "b"}


def test_investment_tables_name_the_investment_solve() -> None:
    dispatch = _sp([("dispatch_roll_0", "p2020"), ("dispatch_roll_1", "p2025")])
    invest = _sp([("invest_p2020", "p2020")])
    # p2025 has no realized investment: the dispatch label.
    assert period_solve_labels(dispatch, invest, table="unit_capacity_ed_p") == {
        "p2020": "invest_p2020", "p2025": "dispatch"}
    # A dispatch table ignores the investment solves.
    assert period_solve_labels(dispatch, invest, table="unit_outputNode_d_ee") == {
        "p2020": "dispatch", "p2025": "dispatch"}
    assert period_solve_labels(dispatch, invest) == {"p2020": "dispatch", "p2025": "dispatch"}
