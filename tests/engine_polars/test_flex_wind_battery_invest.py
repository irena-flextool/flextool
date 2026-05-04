"""flexpy ``wind_battery_invest`` — invest+storage with a user-defined
constraint linking battery (kWh, node) and battery_inverter (kW, process)
capacities.

Parity is BLOCKED on:
  * user constraints with ``invested_capacity_coefficient`` (the
    ``battery_tie_kW_kWh`` constraint forces battery_invest =
    8 × battery_inverter_invest).
  * multi-period invest accounting (``edd_invest`` mapping carries
    p2020 invests forward into p2025 dispatch).

The infrastructure for invest+storage on storage nodes is wired
(``v_invest_n``, maxState tightening, invest-cost obj contribution),
but without the user-constraint extension flexpy lands on a different
local optimum than flextool."""

from pathlib import Path

import polars as pl
import pytest

from flexpy import Problem
from flextool.engine_polars import load_flextool, build_flextool


DATA = Path(__file__).resolve().parent / "data"


def test_wind_battery_invest_parity():
    work = DATA / "work_wind_battery_invest"
    d = load_flextool(work)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    assert sol.optimal
    # Γ.4: prefer golden_obj.json if present, fall back to parquet.
    from _golden import assert_obj_within
    assert_obj_within(sol.obj, work,
                       parquet_glob="v_obj__y2020_5week.parquet")
