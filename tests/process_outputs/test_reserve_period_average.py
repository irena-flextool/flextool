"""Item 188: the period average of the reserve a process holds is time-step weighted.

``process_reserve_average_d_eppe`` = Σ_t w·Δt·MW / Σ_t w·Δt over the period's realized steps (``w`` the timestep
weight).  Before it was Σ_t Δt·MW / (complete_period_share_of_year · 8760), the unweighted mean of the steps.

Hand values (one provider, steps t1-t4 of Δt = 1 h, reserve 10, 10, 30, 30 MW): w ≡ 1 gives 80 / 4 = 20 MW;
w = (1.5, 1.5, 0.5, 0.5) (one representative block standing for 3 base blocks, another for 1: w_r · n_rp / n_base
with n_rp = 2, n_base = 4) gives (15 + 15 + 15 + 15) / 4 = 15 MW, where the old formula gave 80 / 4 = 20 MW.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from flextool.process_outputs.calc_slacks import reserve_period_average
from flextool.process_outputs.reduced_cells import ReducedCells

STEPS = pd.MultiIndex.from_tuples([("p1", f"t{i}") for i in range(1, 5)], names=["period", "time"])
COLS = pd.MultiIndex.from_tuples([("u1", "r", "up", "n")], names=["process", "reserve", "updown", "node"])


def _reserve(values) -> pd.DataFrame:
    return pd.DataFrame({COLS[0]: values}, index=STEPS).set_axis(COLS, axis=1)


def _par(weights) -> SimpleNamespace:
    return SimpleNamespace(timestep_weight=pd.Series(weights, index=STEPS),
                           step_duration=pd.Series(1.0, index=STEPS))


def test_unit_weights_give_the_plain_mean() -> None:
    avg = reserve_period_average(_reserve([10.0, 10.0, 30.0, 30.0]), _par([1.0] * 4))
    assert avg.loc["p1", COLS[0]] == pytest.approx(20.0, rel=1e-12)


def test_representative_weights_weight_the_steps() -> None:
    avg = reserve_period_average(_reserve([10.0, 10.0, 30.0, 30.0]), _par([1.5, 1.5, 0.5, 0.5]))
    assert avg.loc["p1", COLS[0]] == pytest.approx(15.0, rel=1e-12)


def test_a_step_without_a_weight_raises() -> None:
    par = SimpleNamespace(timestep_weight=pd.Series([1.0, 1.0, 1.0], index=STEPS[:3]),
                          step_duration=pd.Series(1.0, index=STEPS))
    with pytest.raises(ValueError, match=r"Reserve results: no timestep weight .*\(p1, t4\)"):
        reserve_period_average(_reserve([10.0, 10.0, 30.0, 30.0]), par)


def test_steps_of_a_solve_without_reserves_are_left_out() -> None:
    """t3, t4 belong to a solve that left reserves out (the reader fills them with 0): the average is over t1, t2,
    10 MW, not (10 + 10 + 0 + 0) / 4 = 5 MW."""
    cells = ReducedCells(steps={"reserves": STEPS[2:]}, solves={"reserves": {"s2"}})
    avg = reserve_period_average(_reserve([10.0, 10.0, 0.0, 0.0]), _par([1.0] * 4), cells)
    assert avg.loc["p1", COLS[0]] == pytest.approx(10.0, rel=1e-12)
