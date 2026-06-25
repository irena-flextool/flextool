"""Benders (Option C) — parallel region recourse DETERMINISM gate.

The per-iteration region recourse pass solves the N independent region
subproblems across a thread pool (``solve_benders(..., workers=...)``, backed by
``polar_high.solve_indexed_parallel``).  The Benders trajectory depends on the
region ``col_dual`` cut slopes, so the parallel solves MUST produce IDENTICAL
``(cost_r, slopes)`` and therefore identical LB / UB / iteration-count / recovered
invest+trade to the sequential (``workers=1``) path.

This module is the HARD determinism gate: run ``solve_benders`` on the prototype
``lh2_three_region_trade_invest`` fixture with ``workers=1`` and with
``workers=3`` and assert the outcomes are EXACTLY (bit-for-bit) equal — not just
close.  A divergence means the parallelism is non-deterministic and must not
ship.
"""
from __future__ import annotations

import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars._benders import solve_benders

_REGIONS = ["region_A", "region_B", "region_C"]


@pytest.fixture(scope="module")
def ti_data(scenario_workdir):
    workdir = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )
    return load_flextool(workdir)


def _run(ti_data, workers):
    return solve_benders(
        ti_data,
        _REGIONS,
        max_iters=20,
        tol=1e-4,
        monolith_objective=None,
        workers=workers,
    )


def test_parallel_equals_sequential_exact(ti_data) -> None:
    seq = _run(ti_data, workers=1)
    par = _run(ti_data, workers=3)

    # Scalar trajectory: EXACT equality (bit-for-bit), not approximate.
    assert par.iterations == seq.iterations, (
        f"iteration count differs: parallel={par.iterations} "
        f"sequential={seq.iterations}"
    )
    assert par.converged == seq.converged
    assert par.total_objective == seq.total_objective, (
        f"objective differs: parallel={par.total_objective!r} "
        f"sequential={seq.total_objective!r}"
    )
    assert par.lower_bound == seq.lower_bound, (
        f"LB differs: parallel={par.lower_bound!r} sequential={seq.lower_bound!r}"
    )
    assert par.upper_bound == seq.upper_bound, (
        f"UB differs: parallel={par.upper_bound!r} sequential={seq.upper_bound!r}"
    )
    assert par.gap == seq.gap, (
        f"gap differs: parallel={par.gap!r} sequential={seq.gap!r}"
    )

    # Per-region recourse costs: EXACT.
    assert par.region_costs.keys() == seq.region_costs.keys()
    for r in seq.region_costs:
        assert par.region_costs[r] == seq.region_costs[r], (
            f"region cost[{r}] differs: parallel={par.region_costs[r]!r} "
            f"sequential={seq.region_costs[r]!r}"
        )

    # Recovered investment per connection: EXACT.
    assert par.invest.keys() == seq.invest.keys()
    for c in seq.invest:
        assert par.invest[c] == seq.invest[c], (
            f"invest[{c}] differs: parallel={par.invest[c]!r} "
            f"sequential={seq.invest[c]!r}"
        )

    # Recovered trade-flow frames: EXACT (same arcs, same per-cell values).
    assert par.trade_flow.keys() == seq.trade_flow.keys()
    for key in seq.trade_flow:
        assert par.trade_flow[key].equals(seq.trade_flow[key]), (
            f"trade_flow[{key}] differs between parallel and sequential"
        )

    # The whole-system invest handoff frames: EXACT.
    assert par.invest_solution_vars.keys() == seq.invest_solution_vars.keys()
    for name in seq.invest_solution_vars:
        assert par.invest_solution_vars[name].equals(
            seq.invest_solution_vars[name]
        ), f"invest_solution_vars[{name}] differs between parallel and sequential"


def test_env_override_resolves(ti_data, monkeypatch) -> None:
    """The machine-local ``FLEXTOOL_BENDERS_WORKERS`` override drives the same
    deterministic result as the explicit ``workers`` arg / the sequential path."""
    monkeypatch.setenv("FLEXTOOL_BENDERS_WORKERS", "2")
    env = _run(ti_data, workers=None)  # env wins over the auto default
    monkeypatch.delenv("FLEXTOOL_BENDERS_WORKERS")
    seq = _run(ti_data, workers=1)
    assert env.total_objective == seq.total_objective
    assert env.lower_bound == seq.lower_bound
    assert env.iterations == seq.iterations
