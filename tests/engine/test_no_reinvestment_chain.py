"""A chain of solves with an entity whose investment window closes at its end of life (``no_reinvestment``).

Fixture ``tests.json`` (built from JSON), the chain ``invest_solveSequence_5weeks`` (``CHAIN``): the solves
y2020 … y2035 each invest in their own period and the next one (y2035 in p2035 only).  Run years: p2020 0,
p2025 5, p2030 10, p2035 15.  coal_plant has ``lifetime_method = no_reinvestment``: its end of life is
EOL = yr[p2020] + lifetime = lifetime, and a period d is closed to investment iff yr[d] ≥ EOL.

The class partitions of the investment set (``pd_invest_set`` for units and connections, ``nd_invest_set`` for
storage nodes) are what the model builds its investment variables over; they must hold exactly the
(entity, period) pairs of ``ed_invest_set``, which the ``no_reinvestment`` window filters.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import b64, build, make_db, run, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_retire_end_of_life import _lifetime
from tests.engine.test_user_constraint_capacity_terms import CHAIN, PLANT

YEARS = {"p2020": 0.0, "p2025": 5.0, "p2030": 10.0, "p2035": 15.0}
#: The investment periods of each solve of ``invest_solveSequence_5weeks``.
INVEST_PERIODS = {"y2020_5week": ["p2020", "p2025"], "y2025_5week": ["p2025", "p2030"],
                  "y2030_5week": ["p2030", "p2035"], "y2035_5week": ["p2035"]}


def _pairs(frame: pl.DataFrame | None, axis: str, entity: str | None = None) -> set[tuple[str, str]]:
    if frame is None:
        return set()
    rows = utf8(frame).select(pl.col(axis).alias("e"), "d")
    if entity is not None:
        rows = rows.filter(pl.col("e") == entity)
    return set(rows.iter_rows())


def _solve_of(step_name: str) -> str:
    return next(s for s in INVEST_PERIODS if step_name.startswith(s))


@pytest.mark.solver
@pytest.mark.parametrize("life", [10.0, 15.0])
def test_chain_runs_after_the_window_closes(tmp_path_factory, life) -> None:
    """The chain solves to the end, and in each solve coal_plant may invest exactly in the solve's investment
    periods with yr < EOL = ``life``.  Hand: life 10 → y2020 {p2020, p2025}, y2025 {p2025}, y2030 and y2035
    none; life 15 → y2020 {p2020, p2025}, y2025 {p2025, p2030}, y2030 {p2030}, y2035 none.  (Before: once every
    coal period of a solve was closed, the solve stopped with a ``p_entity_max_units`` error.)"""
    url, scen = make_db(tmp_path_factory, alt=f"noinv_chain_{int(life)}", values=_lifetime("no_reinvestment", life),
                        base_chain=_no_battery_invest(CHAIN))
    steps = run(tmp_path_factory, url, scen)
    assert {_solve_of(name) for name in steps} == set(INVEST_PERIODS)
    for name, step in steps.items():
        want = {(PLANT, d) for d in INVEST_PERIODS[_solve_of(name)] if YEARS[d] < life}
        assert _pairs(step.flex_data.pd_invest_set, "p", PLANT) == want, name


@pytest.mark.solver
def test_partitions_follow_the_filtered_invest_set(tmp_path_factory) -> None:
    """coal_plant ``no_reinvestment`` with lifetime 10 beside a storage node that still invests
    (``battery_inverter`` not allowed to invest): in every solve the union of the class partitions equals
    ``ed_invest_set``, and the LP of y2030 and y2035 (where coal's window is closed, yr ≥ 10) has no coal
    investment variable.  (Before: the partitions kept coal's closed periods from the unfiltered seed.)"""
    values = _lifetime("no_reinvestment", 10.0) + [
        ["connection", "battery_inverter", "invest_method", b64("not_allowed", "str")]]
    url, scen = make_db(tmp_path_factory, alt="noinv_partitions", values=values, base_chain=CHAIN)
    steps = run(tmp_path_factory, url, scen)
    closed_seen = 0
    for name, step in steps.items():
        fd = step.flex_data
        assert fd.ed_invest_set is not None, name
        parts = _pairs(fd.pd_invest_set, "p") | _pairs(fd.nd_invest_set, "n")
        assert parts == _pairs(fd.ed_invest_set, "e"), name
        if _solve_of(name) in ("y2030_5week", "y2035_5week"):
            closed_seen += 1
            pb = build(fd)
            var = pb._vars.get("v_invest_p")
            if var is not None:
                assert utf8(var.frame).filter(pl.col("p") == PLANT).height == 0, name
    assert closed_seen == 2
