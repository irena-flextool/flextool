"""Hand-computed step_duration LPs loaded from hand-written CSV folders.

Moved from ``tests/engine/test_native_cascade.py`` (c24 S1-8): the
folders ``tests/engine/data/work_step_duration_{1,3}`` are gitignored
hand-written CSV inputs loaded without a DB or a provider, so these are
CSV-reload tests; they skip wherever the folders are absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine import load_flextool
from tests.engine.test_native_cascade import _solve_data

pytestmark = pytest.mark.solver

_DATA = Path(__file__).resolve().parents[1] / "engine" / "data"


def _solve_csv_folder(workdir: Path) -> float:
    """Load a hand-written CSV folder (no DB, no provider) and solve it.

    Only the two ``*_handcomputed`` tests use it: their folders
    (``data/work_step_duration_{1,3}``) are gitignored hand-written CSV
    inputs, so these are CSV-reload tests, not production-path loads; they
    skip wherever the folders are absent.
    """
    return _solve_data(load_flextool(workdir), str(workdir))


def test_step_duration_1_handcomputed():
    """Trivial single-step LP: one demand node (-100 MWh), one
    cost-linear process (coal @ 20 EUR/MWh, efficiency=1).

    LP objective = commodity_cost = price x flow x step_duration
                 = 20 x 100 x 1 = 2000.0 EUR.

    Hand-derived value with period_share=1, inflation=1,
    timestep_weight=1, so op_factor reduces to step_duration.
    """
    workdir = _DATA / "work_step_duration_1"
    if not workdir.exists():
        pytest.skip(f"fixture missing: {workdir}")
    obj = _solve_csv_folder(workdir)
    assert obj == pytest.approx(2000.0, rel=1e-9), (
        f"baseline single-step LP did not match hand-computed "
        f"objective: got {obj}, expected 2000.0"
    )


def test_step_duration_3_handcomputed():
    """Same physical scenario as the step=1 test, expressed as
    step_duration=3 with pdtNodeInflow tripled.

    LP objective = 20 x 100 x 3 = 6000.0 EUR.

    A regression that re-enables the cascade p_inflow override AND
    that override is still step_duration-unaware (the lt_rp bug
    class) would silently divide demand by step_duration and
    produce 2000.0 EUR instead.
    """
    workdir = _DATA / "work_step_duration_3"
    if not workdir.exists():
        pytest.skip(f"fixture missing: {workdir}")
    obj = _solve_csv_folder(workdir)
    assert obj == pytest.approx(6000.0, rel=1e-9), (
        f"step_duration=3 LP did not match hand-computed objective: "
        f"got {obj}, expected 6000.0.  This is the p_inflow lt_rp "
        f"bug class - see specs/native_inflow_pathway_disabled.md."
    )

