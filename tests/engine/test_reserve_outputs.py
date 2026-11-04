"""Reserve reservations are reported in MW per step, also with steps longer than an hour.

Fixture ``tests.json``, scenario chain of ``network_coal_wind_reserve`` (built from JSON): the ``primary`` up
reserve of ``electricity`` has ``reservation`` 10 MW (``timeseries_only``), wind_plant (``existing`` 1000 MW, no
``virtual_unitsize`` → unit size 1000 MW) is the only provider, ``reliability`` 0.5, ``max_share`` 0.2.  The
override alternative sets ``new_stepduration`` 2 h on the dispatch solve.

Hand derivation.  ``reserveBalance_timeseries_eq``: reserve · reliability ≥ reservation, so wind holds
10 / 0.5 = 20 MW in every step (holding reserve has no cost; 20 MW ≤ max_share · 1000 = 200 MW).  The model's
``v_reserve`` counts in unit sizes (MW = v_reserve · 1000), so the reported reservation is 20 MW per step — not
20 MW × 2 h (before: ``v_reserve`` × step duration = 0.04, without the unit size), and the period average
Σ_t MW · Δt / Σ_t Δt is 20 MW too.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db

pytestmark = pytest.mark.solver

CHAIN = ["init", "west", "coal", "wind", "network", "reserve"]
SOLVE = "y2020_2day_dispatch"
HELD_MW = 10.0 / 0.5
UNIT_SIZE = 1000.0


@pytest.fixture(scope="module")
def two_hour_run(tmp_path_factory):
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt="reserve_2h",
                        values=[["solve", SOLVE, "new_stepduration", b64(2.0, "float")]], base_chain=CHAIN)
    work = tmp_path_factory.mktemp("reserve_2h_w")
    steps = run_chain_from_db(url, scenario_name=scen, work_folder=work, keep_solutions=True)
    assert steps and all(s.optimal for s in steps.values())
    last = next(reversed(list(steps.values())))
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
                  flex_data_provider=getattr(last, "flex_data_provider", None),
                  raw_output_dir=str(work / "output_raw"))
    return last, work / "output_csv" / scen


def _reported(csv_dir, stem: str, index_cols: int) -> pd.Series:
    frame = pd.read_csv(csv_dir / f"{stem}.csv", header=[0, 1, 2, 3], index_col=list(range(index_cols)))
    assert list(frame.columns) == [("wind_plant", "primary", "up", "west")]
    return frame.iloc[:, 0]


def test_reservation_is_mw_per_step(two_hour_run) -> None:
    step, csv_dir = two_hour_run
    durations = step.flex_data.p_step_duration.frame["value"].unique().to_list()
    assert durations == [2.0], "fixture must have 2-hour steps"
    per_step = _reported(csv_dir, "process__reserve__upDown__node__dt", 3)
    assert len(per_step) > 1
    assert per_step.to_numpy() == pytest.approx(HELD_MW, rel=1e-9)
    solved = step.solution.value("v_reserve").filter(pl.col("p").cast(pl.Utf8) == "wind_plant")["value"]
    assert per_step.to_numpy() == pytest.approx(solved.to_numpy() * UNIT_SIZE, rel=1e-9)


def test_period_average_is_mw(two_hour_run) -> None:
    _, csv_dir = two_hour_run
    average = _reported(csv_dir, "process__reserve__upDown__node__d_average", 2)
    assert average.to_numpy() == pytest.approx(HELD_MW, rel=1e-9)
