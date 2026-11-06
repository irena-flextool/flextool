"""Regression test: ``co2_price`` feature with missing data downgrades
to a warning, not an error.

The CO2-price feature gate is driven by *topology* (``flow_from_co2_priced``
is populated whenever a process source-sink connects to a node in a
priced commodity group), but the *data* fields (``p_co2_price``,
``p_co2_content``) are authored separately on ``group.co2_price`` and
``commodity.co2_content``.  Pre-fix HEAD raised a ``ValueError`` from
``build_flextool`` when the data was missing — a hard abort that broke
otherwise-valid solves (e.g. a debug run with the CO2 group temporarily
emptied, or a fixture where ``group.co2_price`` carried
extra periods that the silent-default resolver couldn't classify).

Post-fix: ``build_flextool`` logs a warning naming the missing field(s)
and skips the CO2 cost term.  The LP build proceeds and the solve
completes; the only behavioural difference is that CO2 emissions are
no longer priced in the objective.

Fixture: the session-cached ``coal_co2_price`` cascade run (built from
``tests/fixtures/tests.json``).  Each test rebuilds the solve's FlexData
with the production loader call (``load_scenario``), clears one field and
calls ``build_flextool`` on it.
"""
from __future__ import annotations

import logging

import pytest

from polar_high import Problem
from flextool.engine import build_flextool
from tests._support.load import ScenarioRun, load_scenario


pytestmark = pytest.mark.solver

SCENARIO = "coal_co2_price"


@pytest.fixture(scope="module")
def co2_price_run(scenario_run) -> ScenarioRun:
    """The ``coal_co2_price`` cascade run (one cascade, shared by the three tests)."""
    return scenario_run(SCENARIO)


def test_missing_p_co2_price_warns_and_builds(
    co2_price_run: ScenarioRun, caplog,
) -> None:
    """Clear ``p_co2_price`` after loading; build_flextool must warn
    and still produce a solvable LP (with the CO2 term omitted)."""
    data = load_scenario(co2_price_run)
    # Fixture sanity: the topology activates co2_price.
    assert data.flow_from_co2_priced is not None
    assert data.flow_from_co2_priced.height > 0
    assert data.p_co2_price is not None, (
        "fixture pre-condition: p_co2_price is populated"
    )
    # Simulate the bug: clear the price (e.g. user disabled the
    # group.co2_price alternative without removing the topology).
    data.p_co2_price = None

    pb = Problem()
    with caplog.at_level(logging.WARNING,
                         logger="flextool.engine.model.core"):
        # Pre-fix: this raised ValueError on the CO2_PRICE invariant.
        build_flextool(pb, data)

    warnings = [r for r in caplog.records
                if "CO2 price method" in r.getMessage()]
    assert warnings, (
        "expected a warning naming the missing CO2 data field; "
        f"got records: {[r.getMessage() for r in caplog.records]}"
    )
    assert "p_co2_price" in warnings[0].getMessage()

    sol = pb.solve()
    assert sol.optimal, (
        f"LP must still solve with the CO2 cost term skipped; got {sol!r}"
    )


def test_missing_p_co2_content_warns_and_builds(
    co2_price_run: ScenarioRun, caplog,
) -> None:
    """Symmetric: clearing ``p_co2_content`` instead of ``p_co2_price``
    also triggers the warning + skip (the CO2 term needs both)."""
    data = load_scenario(co2_price_run)
    assert data.p_co2_content is not None
    data.p_co2_content = None

    pb = Problem()
    with caplog.at_level(logging.WARNING,
                         logger="flextool.engine.model.core"):
        build_flextool(pb, data)

    warnings = [r for r in caplog.records
                if "CO2 price method" in r.getMessage()]
    assert warnings
    assert "p_co2_content" in warnings[0].getMessage()

    sol = pb.solve()
    assert sol.optimal


def test_objective_drops_co2_term_when_price_missing(
    co2_price_run: ScenarioRun,
) -> None:
    """When p_co2_price is missing, the objective EXCLUDES the CO2 cost
    contribution.  Compare against the fully-priced baseline: the
    missing-price objective is strictly smaller (no CO2 cost)."""
    # Baseline with CO2 price.
    data_full = load_scenario(co2_price_run)
    pb_full = Problem()
    build_flextool(pb_full, data_full)
    sol_full = pb_full.solve()
    assert sol_full.optimal

    # Same model, p_co2_price cleared.
    data_nop = load_scenario(co2_price_run)
    data_nop.p_co2_price = None
    pb_nop = Problem()
    build_flextool(pb_nop, data_nop)
    sol_nop = pb_nop.solve()
    assert sol_nop.optimal

    # CO2 carries a non-zero price in the fixture, so dropping the term
    # strictly reduces the objective (the solver may also redispatch,
    # making the gap larger than just the CO2 cost — but never smaller).
    assert sol_nop.obj < sol_full.obj - 1e-6, (
        f"baseline obj={sol_full.obj}, no-CO2 obj={sol_nop.obj}; "
        "expected the no-CO2 objective to be strictly smaller"
    )
