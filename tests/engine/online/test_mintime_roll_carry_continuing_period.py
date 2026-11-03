"""The minimum up-time obligation is carried across a rolling seam at the boundary of a continuing period.

Two periods on one timeline: p2020 on t0001-t0024 and p2025 on t0025-t0048, so p2025's first step continues
p2020's last step (``jump == 1``).  Rolled one period per roll (horizon = jump = 24 h), the seam falls exactly on
the period boundary.  The second roll's first step (p2025, t0025) is the seam root (``dt_ramp_carry``), the start-up
that the first roll realized within the minimum up-time before its end is carried onto the steps it still covers,
and the rolled commitment equals the single solve, whose look-back walks across the same boundary.  Fixtures are
built from ``tests.json`` under the pytest tmp dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, TIMESTEPS, b64, make_db, run, time_map, uc, utf8

SOLVE = "y2020_2day_dispatch"
L_UP = 6.0            # h, minimum up-time
HIGH = 400.0          # MW, within the 500 MW coal_plant and above its 200 MW minimum load
ENTS = [["timeset", "ta", None], ["timeset", "tb", None]]


def _pmap(data, index_name="period"):
    return {"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in data.items()]}


def _values(rolling: bool) -> list:
    # Demand only in t0021-t0024 (the last four hours of p2020): the unit starts up at t0021 and min up-time keeps
    # it online until t0026 (6 h), although p2025 has no demand.  The efficiency is halved before t0021, so running
    # the two extra hours before the demand (t0019-t0020) costs more fuel than running them after it (t0025-t0026):
    # the single solve has the same unique optimum as the rolled one, which cannot see the demand ahead.
    demand = {t: (-HIGH if t in TIMESTEPS[20:24] else 0.0) for t in TIMESTEPS}
    efficiency = {t: (0.2 if t in TIMESTEPS[:20] else 0.4) for t in TIMESTEPS}
    vals = uc(PLANT, 0.4, "binary") + [
        ["unit", PLANT, "efficiency", time_map(efficiency, 0.4)],
        ["unit", PLANT, "startup_cost", b64(50.0, "float")],
        ["unit", PLANT, "min_uptime", b64(L_UP, "float")],
        ["unit", PLANT, "minimum_time_method", b64("min_uptime", "str")],
        ["node", "west", "inflow", time_map(demand, 0.0)],
        ["timeset", "ta", "timeline", b64("y2020", "str")],
        ["timeset", "tb", "timeline", b64("y2020", "str")],
        ["timeset", "ta", "timeset_duration", b64(_pmap({"t0001": 24.0}, "time"), "map")],
        ["timeset", "tb", "timeset_duration", b64(_pmap({"t0025": 24.0}, "time"), "map")],
        ["solve", SOLVE, "period_timeset", b64(_pmap({"p2020": "ta", "p2025": "tb"}), "map")],
        ["solve", SOLVE, "realized_periods", b64({"value_type": "str", "data": ["p2020", "p2025"]}, "array")]]
    if rolling:
        vals += [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
                 ["solve", SOLVE, "rolling_solve_horizon", b64(24.0, "float")],
                 ["solve", SOLVE, "rolling_solve_jump", b64(24.0, "float")]]
    return vals


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for name, rolling in (("mt_cont_single", False), ("mt_cont_rolling", True)):
        url, scen = make_db(tmp_path_factory, alt=name, values=_values(rolling), base_chain=PLANT_CHAIN,
                            entities=ENTS)
        out[name] = list(run(tmp_path_factory, url, scen).values())
    return out


def _uc(step, var, realized=False):
    v = utf8(step.solution.value(f"{var}_integer")).filter(pl.col("p") == PLANT)
    if realized:
        real = utf8(step.flex_data.realized_dispatch.rename({"period": "d", "step": "t"}))
        v = v.join(real.select("d", "t"), on=["d", "t"], how="semi")
    value = pl.col("value").round(6)
    return v.select("d", "t", pl.when(value == 0.0).then(0.0).otherwise(value).alias("value")).sort("d", "t")


@pytest.mark.solver
def test_continuing_period_carries_the_minimum_up_time(runs):
    """Hand derivation: the start-up at p2020 t0021 (the only one, 1 unit) has hours_to_end 4 h at the end of
    roll 1 (t0021-t0024).  Carried onto the steps of roll 2 that follow the root p2025 t0025 while elapsed + 4 < 6:
    t0025 (elapsed 0) and t0026 (elapsed 1), each with a carried start-up sum of 1; t0027 (elapsed 2) has none.
    These are the steps whose single-solve look-back reaches back to t0021 (6 h windows t0020-t0025, t0021-t0026).
    So the unit is online in t0021-t0026 in both runs and offline from t0027."""
    (single,), (first, second) = runs["mt_cont_single"], runs["mt_cont_rolling"]
    assert first.flex_data.dt_ramp_carry is None
    root = utf8(second.flex_data.dt_ramp_carry).select("d", "t", "d_previous", "t_previous").rows()
    assert root == [("p2025", "t0025", "p2020", "t0024")]
    assert utf8(second.flex_data.dt_online_carry).rows() == [("p2025", "t0025")]
    carry = utf8(second.flex_data.uptime_carry).filter(pl.col("p") == PLANT).sort("d", "t")
    assert carry.select("d", "t").rows() == [("p2025", "t0025"), ("p2025", "t0026")]
    assert carry["value"].cast(pl.Float64).to_list() == [1.0, 1.0]
    look = (utf8(single.flex_data.uptime_lookback)
            .filter((pl.col("p") == PLANT) & (pl.col("d_back") == "p2020") & (pl.col("t_back") == "t0021")
                    & (pl.col("d") == "p2025"))
            .select("d", "t").sort("t").rows())
    assert look == carry.select("d", "t").rows()
    for var in ("v_online", "v_startup", "v_shutdown"):
        rolled = pl.concat([_uc(first, var, realized=True), _uc(second, var, realized=True)]).sort("d", "t")
        assert rolled.equals(_uc(single, var)), var
    on = _uc(single, "v_online").filter(pl.col("value") > 0.5).select("d", "t").rows()
    assert on == [("p2020", t) for t in TIMESTEPS[20:24]] + [("p2025", "t0025"), ("p2025", "t0026")]
