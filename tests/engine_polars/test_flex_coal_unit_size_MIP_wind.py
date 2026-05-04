"""flexpy ``coal_unit_size_MIP_wind`` — two-stage solve where the
``y2020_fullYear_dispatch`` run inherits invested capacity from a prior
``y2020_5week`` MIP solve.  The inheritance lands in
``p_entity_period_existing_capacity`` (already includes prior invest);
``p_entity_period_invested_capacity`` is a record of "this came from
prior invest" — must NOT be summed into existing."""

from pathlib import Path

import polars as pl

from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool


DATA = Path(__file__).resolve().parent / "data"


def test_coal_unit_size_MIP_wind_parity():
    work = DATA / "work_coal_unit_size_MIP_wind"
    d = load_flextool(work)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_fullYear_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
