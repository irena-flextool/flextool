"""Rule-6 ``years_represented`` default must seed EVERY solve period.

DEFECT PIN: ``TimelineConfig.create_assumptive_parts`` Rule 6
(``_timeline.py``) is supposed to seed "each period represents 1 year"
when a solve does not author ``years_represented``.  Pre-fix it checked
solve membership INSIDE its per-period loop; the first ``append``
inserts the solve key into the ``defaultdict``, so every later period
saw the key present and was skipped — exactly one (arbitrary) period
got seeded.  Multi-period solves relying on the default then failed
the years-represented completeness check in ``_native_run_model.py``
with ``FlexToolConfigError`` ("defined, but not to all of the
periods"), including plain deterministic multi-period solves.

Solver-free: constructs ``TimelineConfig`` / ``SolveConfig`` directly
and calls ``create_assumptive_parts``.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from flextool.engine_polars._solve_config import (
    HiGHSConfig,
    SolveConfig,
    SolverSettings,
)
from flextool.engine_polars._timeline import TimelineConfig

SOLVE = "s2p"
PERIODS = ["p2035", "p2040", "p2045"]


def _timeline_config() -> TimelineConfig:
    """Minimal config that makes Rules 1-3 no-ops."""
    timelines: defaultdict = defaultdict(list)
    timelines["tl"] = [("t0001", 1.0), ("t0002", 1.0)]
    timesets__timeline: defaultdict = defaultdict(str)
    timesets__timeline["ts1"] = "tl"
    timeset_durations: defaultdict = defaultdict(list)
    timeset_durations["ts1"] = [("t0001", 2)]
    return TimelineConfig(
        timelines=timelines,
        timesets=["ts1"],
        timesets__timeline=timesets__timeline,
        timeset_durations=timeset_durations,
        new_step_durations={},
    )


def _solve_config(
    years_represented: list | None = None,
    *,
    realized: list[str] | None = None,
    invest: list[str] | None = None,
) -> SolveConfig:
    """Minimal SolveConfig that makes Rules 4-5 no-ops (solve already
    has periods and timesets) so only Rule 6 mutates."""
    ypr: defaultdict = defaultdict(list)
    if years_represented is not None:
        ypr[SOLVE] = list(years_represented)
    model_solve: defaultdict = defaultdict(list)
    model_solve["flextool"] = [SOLVE]
    cfg = SolveConfig(
        model=["flextool"],
        model_solve=model_solve,
        solve_modes={},
        rolling_times=defaultdict(list),
        highs=HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(
            solvers={}, precommand={}, arguments={}
        ),
        solve_period_years_represented=ypr,
        hole_multipliers=defaultdict(list),
        contains_solves=defaultdict(list),
        stochastic_branches=defaultdict(list),
        periods_available={},
        delay_durations={},
        logger=logging.getLogger(__name__),
    )
    realized = PERIODS if realized is None else realized
    cfg.realized_periods[SOLVE] = [(p, p) for p in realized]
    if invest:
        cfg.invest_periods[SOLVE] = [(p, p) for p in invest]
    cfg.timesets_used_by_solves[SOLVE] = [
        (p, "ts1") for p in (realized + (invest or []))
    ]
    return cfg


def test_default_seeds_all_periods_with_one_year() -> None:
    """DEFECT PIN: no ``years_represented`` → EVERY period of the solve
    gets a 1.0-year default row (pre-fix: exactly one)."""
    cfg = _solve_config(None)
    _timeline_config().create_assumptive_parts(cfg)
    assert cfg.solve_period_years_represented[SOLVE] == [
        [p, 1.0] for p in PERIODS
    ]


def test_default_covers_invest_only_periods() -> None:
    """Invest-only periods are part of the solve and must be seeded
    too, after the realized periods (first-occurrence order)."""
    cfg = _solve_config(None, realized=["p2035"], invest=["p2040"])
    _timeline_config().create_assumptive_parts(cfg)
    assert cfg.solve_period_years_represented[SOLVE] == [
        ["p2035", 1.0],
        ["p2040", 1.0],
    ]


def test_full_definition_untouched() -> None:
    """A solve that authors ``years_represented`` for all periods keeps
    exactly its authored values — Rule 6 must not touch them."""
    authored = [("p2035", 1.0), ("p2040", 5.0), ("p2045", 10.0)]
    cfg = _solve_config(authored)
    _timeline_config().create_assumptive_parts(cfg)
    assert cfg.solve_period_years_represented[SOLVE] == authored


def test_partial_definition_untouched() -> None:
    """A PARTIAL authored definition is preserved verbatim — Rule 6
    does not backfill the missing periods (the completeness check in
    ``_native_run_model.py`` later raises for them; pre-existing
    contract, unchanged by the fix)."""
    authored = [("p2035", 2.0)]
    cfg = _solve_config(authored)
    _timeline_config().create_assumptive_parts(cfg)
    assert cfg.solve_period_years_represented[SOLVE] == authored


def test_default_is_idempotent() -> None:
    """Running ``create_assumptive_parts`` twice must not duplicate the
    seeded rows (the rules document idempotency)."""
    cfg = _solve_config(None)
    tc = _timeline_config()
    tc.create_assumptive_parts(cfg)
    tc.create_assumptive_parts(cfg)
    assert cfg.solve_period_years_represented[SOLVE] == [
        [p, 1.0] for p in PERIODS
    ]
