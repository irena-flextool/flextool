"""A solve without ``new_stepduration`` after one with it reads its own inflow scaling inputs.

``create_averaged_timeseries`` puts the full-timeline inflow energy (``solve_data/node_full_timeline_flow``) for a
solve with ``new_stepduration``; ``scale_to_annual_flow`` then divides by it instead of the solve's own period
share.  A later solve without ``new_stepduration`` must not read the earlier solve's frame.

Databases are ``tests/fixtures/tests.json`` + one override alternative on ``init, west, wind, battery, fullYear``:
model solves ``y2020_fullYear_dispatch`` (all 72 steps, new_stepduration 2) then ``y2020_2day_dispatch`` (the first
48 steps, no new_stepduration); west's profile is scaled with ``scale_to_annual_flow`` to 1e6.  The 48 steps hold
−29 103 of the 72 steps' total, so the full-timeline denominator gives a different multiplier than the 2day solve's
own period share.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, run, utf8

pytestmark = pytest.mark.solver

CHAIN = ["init", "west", "wind", "battery", "fullYear"]
SCALING = [["node", "west", "inflow_method", b64("scale_to_annual_flow", "str")],
           ["node", "west", "annual_flow", b64(1.0e6, "float")]]


def _solves(names: list[str]) -> list:
    return ["model", "flexTool", "solves", b64({"value_type": "str", "data": names}, "array")]


def _two_day(tmp_path_factory, alt: str, values: list):
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=CHAIN)
    steps = run(tmp_path_factory, url, scen)
    step = steps["y2020_2day_dispatch"]
    inflow = sorted(utf8(step.flex_data.p_inflow.frame).filter(pl.col("n") == "west")
                    .select("d", "t", "value").rows())
    mult = utf8(step.flex_data_provider.get("solve_data/period_flow_annual_multiplier")).rows()
    return inflow, mult


def test_uncoarsened_solve_after_a_coarsened_one_equals_the_solve_alone(tmp_path_factory) -> None:
    """The 2day solve run after the fullYear solve with new_stepduration 2 has the same scaled west inflow and
    multiplier as the 2day solve run alone (period share 48/8760 over its own −29 103).  Before: it read the
    fullYear solve's full-timeline energy and scaled by 1e6 / |full-timeline energy| instead."""
    after_inflow, after_mult = _two_day(tmp_path_factory, "ctu_after", SCALING + [
        _solves(["y2020_fullYear_dispatch", "y2020_2day_dispatch"]),
        ["solve", "y2020_fullYear_dispatch", "new_stepduration", b64(2.0, "float")]])
    alone_inflow, alone_mult = _two_day(tmp_path_factory, "ctu_alone", SCALING + [_solves(["y2020_2day_dispatch"])])
    # Alone: annual_flow × period share / |Σ inflow| = 1e6 · (48/8760) / 29 103 = 0.188278.
    assert len(alone_mult) == 1 and alone_mult[0][:2] == ("west", "p2020")
    assert float(alone_mult[0][2]) == pytest.approx(1.0e6 * 48 / (8760 * 29103.0), rel=1e-12)
    assert after_mult == alone_mult
    assert len(after_inflow) == len(alone_inflow) >= 48
    assert after_inflow == alone_inflow
