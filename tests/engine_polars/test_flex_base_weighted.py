"""``base_weighted`` — base with non-uniform timeset_weights.  Just
re-weights `p_rp_cost_weight` to non-1.0 values; the rest of the LP
is identical to ``base``.  Validates that flexpy's load_flextool
correctly reads and applies the weights."""

from pathlib import Path
import polars as pl

from polar_high import Problem
from flextool.engine_polars import load_flextool
from flextool.engine_polars import build_flextool
import pytest

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_base_weighted"


def test_base_weighted_parity():
    data = load_flextool(WORK)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
