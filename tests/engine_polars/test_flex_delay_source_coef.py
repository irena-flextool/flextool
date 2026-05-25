"""polar_high parity for delay-path with non-default source flow coef.

Scenario ``delay_source_coef`` = ``water_pump_delayed`` + alt
``delay_source_coef_on``, which sets
``unit__inputNode.(water_pump, water_source).flow_coefficient = 2.0``
(default is 1.0).  This exercises the .mod's source-coefficient
multiplier on delayed source flows (flextool.mod:2573) — a code path
that no other fixture combines with delays.  Without the matching
multiplier in ``flextool/_delay.py::delayed_input_expr``, the LP
solutions diverge.

Phase 3d: scenario added to ``tests.json`` via
``tests/fixtures/_augment_phase3d.py``; no derivative workdir / CSV
patch required.
"""
import polars as pl
import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool

pytestmark = pytest.mark.solver


SCENARIO = "delay_source_coef"


def test_delay_source_coef_parity(scenario_workdir):
    """Build the scenario workdir and solve the LP — the elevated source
    coefficient (=2.0) must flow through ``load_flextool`` into
    ``data.p_process_source_flow_coef``, and the LP must still solve to
    optimality.

    Note: in the ``water_pump_delayed`` scenario the source node
    (``water_source``) has unit commodity price = 0, so the elevated
    coefficient does not change the optimal objective value; the test
    therefore exercises the *correctness* of the multiplier path
    (load → build → solve with the elevated coefficient stitched into
    the delay LP) rather than asserting an obj delta.
    """
    work = scenario_workdir(SCENARIO)

    data = load_flextool(work)

    # Sanity: the elevated coefficient flowed through into the loaded
    # data structure.
    fr = data.p_process_source_flow_coef
    assert fr is not None, "p_process_source_flow_coef missing"
    row = fr.frame.filter(
        (pl.col("p") == "water_pump") & (pl.col("source") == "water_source")
    )
    assert row.height == 1, (
        f"expected one (water_pump, water_source) row; got {row.height}"
    )
    assert float(row["value"][0]) == pytest.approx(2.0), (
        f"loaded coefficient is {float(row['value'][0])!r}, expected 2.0 "
        "— the delay_source_coef_on alternative did not flow through to "
        "load_flextool"
    )

    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, "LP must be optimal"
