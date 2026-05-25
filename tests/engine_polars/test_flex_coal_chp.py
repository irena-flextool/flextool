"""``coal_chp`` scenario — multi-flow CHP unit.  Adds two features
polar_high doesn't yet model:

  * a user-defined ``process_constraint_equal`` that enforces the
    backpressure ratio between heat and elec output, and
  * the noEff-source commodity buy term (now wired in but the
    parity test still fails because of the missing user-constraint).

Marked ``xfail`` until ``process_constraint_equal/less_than/greater_than``
support lands in flex_coal_model."""

import polars as pl
import pytest

from polar_high import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool

pytestmark = pytest.mark.solver


SCENARIO = "coal_chp"

def test_coal_chp_parity(scenario_workdir):
    work = scenario_workdir(SCENARIO)
    data = load_flextool(work)
    assert data.process_indirect is not None, "fixture should have indirect process"
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
