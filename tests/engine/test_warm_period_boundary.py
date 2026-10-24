"""Rolls across a period boundary: warm == cold, roll by roll, objectives and written outputs.

The fingerprint once held frame heights only, so two rolls with the same number of steps and periods
but a different split between the periods (p2020 18 h + p2025 12 h, then 12 h + 18 h) looked
identical and the previous roll's LP was reused with the wrong (d, t) -> period membership.  The
fingerprint carries the per-period run lengths of the sorted ``dt`` and the positional offsets of the
``dtttdt`` links (the cheap pre-filter; two tests below pin it).  Since the reuse contract of
``solve/_warm`` no roll is reused at all (``dt`` labels shift): a reused LP kept the labels of the
roll it was built for, so the outputs of the within-period rolls 1-3 and 9-11 came from other hours.

Fixture: ``tests.json`` ``init/west/coal/co2_price`` plus an override alternative (built from JSON,
never a checked-in DB): CO2 price 10 / 100 per tonne in p2020 / p2025, ``y2020_2day_dispatch`` over
both periods (2 days each, both realized), rolled with a 30 h horizon and a 6 h jump (16 rolls; rolls
4-7 straddle the period boundary).  No ``co2_max_period``: the test checks the fingerprint, not the
co2-cap Param ``p_window_fraction``.  Magnitudes stay below the autoscale Layer-2 trigger.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from tests.engine._unit_db import assert_outputs_equal, b64, make_db, utf8

pytestmark = pytest.mark.solver

SOLVE = "y2020_2day_dispatch"
VALUES = [
    ["group", "co2_price", "co2_price",
     b64({"index_type": "str", "index_name": "period", "data": [["p2020", 10.0], ["p2025", 100.0]]}, "map")],
    ["solve", SOLVE, "period_timeset",
     b64({"index_type": "str", "index_name": "period", "data": [["p2020", "2day"], ["p2025", "2day"]]}, "map")],
    ["solve", SOLVE, "realized_periods",
     b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")],
    ["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
    ["solve", SOLVE, "rolling_solve_horizon", b64(30.0, "float")],
    ["solve", SOLVE, "rolling_solve_jump", b64(6.0, "float")],
]


def _roll(i: int) -> str:
    return f"{SOLVE}_roll_{i}"


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="warm_period_boundary", values=VALUES,
                        base_chain=["init", "west", "coal", "co2_price"])
    out = {}
    for warm in (True, False):
        wf = tmp_path_factory.mktemp(f"wpb_{int(warm)}")
        out[warm] = run_chain_from_db(url, scen, warm=warm, keep_solutions=True, work_folder=wf)
        out[("wf", warm)] = wf
    assert list(out[True]) == list(out[False])
    assert list(out[False]) == [_roll(i) for i in range(16)], list(out[False])
    return out


def _hours(step) -> dict[str, float]:
    sd = utf8(step.flex_data.p_step_duration.frame)
    return dict(sd.group_by("d").agg(pl.col("value").sum()).sort("d").iter_rows())


def test_fixture_straddles_the_boundary(runs) -> None:
    """Rolls 4-7 cover both periods with the same 30 steps but a shifting split."""
    cold = runs[False]
    assert [_hours(cold[_roll(i)]) for i in range(4, 8)] == [
        {"p2020": 24.0, "p2025": 6.0}, {"p2020": 18.0, "p2025": 12.0},
        {"p2020": 12.0, "p2025": 18.0}, {"p2020": 6.0, "p2025": 24.0}]


def test_warm_equals_cold_across_period_boundary(runs) -> None:
    hot, cold = runs[True], runs[False]
    for i in range(16):
        h, c = hot[_roll(i)].solution, cold[_roll(i)].solution
        assert h.obj == pytest.approx(c.obj, rel=1e-9), f"roll {i}: warm {h.obj} vs cold {c.obj}"
        # The written outputs (read by label) agree, not only the objective.
        assert_outputs_equal(runs[("wf", True)], runs[("wf", False)], _roll(i))
        # Positional comparison in LP column / row order.
        hv, cv = h.value("v_flow")["value"].to_numpy(), c.value("v_flow")["value"].to_numpy()
        assert hv.shape == cv.shape, f"roll {i}: v_flow sizes differ"
        np.testing.assert_allclose(hv, cv, rtol=0, atol=1e-6, err_msg=f"roll {i}: v_flow")
        hd = h.constraint_dual("nodeBalance_eq")["dual"].to_numpy()
        cd = c.constraint_dual("nodeBalance_eq")["dual"].to_numpy()
        assert hd.shape == cd.shape, f"roll {i}: nodeBalance_eq sizes differ"
        np.testing.assert_allclose(hd, cd, rtol=1e-7, atol=1e-9, err_msg=f"roll {i}: nodeBalance_eq duals")
    # No roll reuses the previous roll's LP: its dt labels shift.
    for i in range(16):
        assert not hot[_roll(i)].warm_used, f"roll {i} reused an LP built for another window"


def test_fingerprint_sees_period_split(runs) -> None:
    """Unit level: same heights, different period split -> different fingerprint; within a period equal."""
    from flextool.engine.solve._warm import _fingerprint

    cold = runs[False]
    fd = {i: cold[_roll(i)].flex_data for i in (1, 2, 5, 6)}
    assert all(f is not None for f in fd.values())
    # The plain heights are the same in rolls 5 and 6: only the split of dt between the periods differs.
    assert fd[5].dt.height == fd[6].dt.height
    assert _fingerprint(fd[5]) != _fingerprint(fd[6])
    # Two consecutive rolls inside p2020 keep the same fingerprint (the pre-filter passes; the
    # full comparison then rejects the shifted dt).
    assert _fingerprint(fd[1]) == _fingerprint(fd[2])


def test_fingerprint_sees_moved_block_start(runs) -> None:
    """Unit level: a timeblock start that moves inside the window (same heights) changes the fingerprint."""
    from flextool.engine.solve._warm import _fingerprint

    fd = runs[False][_roll(1)].flex_data
    lk = fd.dtttdt.sort("d", "t")
    assert "t_previous_within_timeset" in lk.columns
    # Make step 10 of the window the first step of a block: its previous-within-timeset link wraps to
    # the last step of the window instead of pointing at step 9.
    last_t = lk["t"][-1]
    moved = lk.with_columns(
        pl.when(pl.int_range(pl.len()) == 10).then(pl.lit(last_t)).otherwise(pl.col("t_previous_within_timeset"))
        .cast(lk.schema["t_previous_within_timeset"]).alias("t_previous_within_timeset"))
    assert moved.height == lk.height
    assert _fingerprint(dataclasses.replace(fd, dtttdt=moved)) != _fingerprint(fd)
    assert _fingerprint(dataclasses.replace(fd, dtttdt=lk)) == _fingerprint(fd)
