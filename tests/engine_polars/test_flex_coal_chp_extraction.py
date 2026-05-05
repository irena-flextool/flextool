"""``coal_chp_extraction`` scenario — extraction-condensing CHP.

Differs from the back-pressure baseline (``coal_chp``) in that the
back-pressure user constraint is *deactivated* (``constraint__sense.csv``
has no row for ``coal_chp_fix``) and the iso-fuel tradeoff between heat
and electricity is encoded entirely through non-default values of
``p_process_sink_flow_coefficient`` ({heat: 0.2, west: 2.0}) and
``p_process_sink_max_capacity_coefficient`` ({heat: 1.0, west: 0.8}).

The .mod's ``conversion_indirect`` (flextool.mod:2557-2580) multiplies
each sink term by ``p_process_sink_flow_coefficient[p, sink]``; the
matching multiplier landed in ``flextool/model.py``'s
``conversion_indirect`` emission via the ``p_process_sink_flow_coef``
Param plumbed through ``_load_indirect``.
"""

from pathlib import Path

import polars as pl

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
import pytest

pytestmark = pytest.mark.solver


WORK = Path(__file__).resolve().parent / "data" / "work_coal_chp_extraction"


def test_coal_chp_extraction_parity():
    data = load_flextool(WORK)
    assert data.process_indirect is not None, (
        "fixture should have indirect process")
    # Sanity: the fixture should populate the sink-side flow_coef Param
    # (heat=0.2, west=2.0).  Without it the conversion equation
    # collapses to back-pressure semantics and parity fails at ~96%.
    assert data.p_process_sink_flow_coef is not None
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    flextool_obj = pl.read_parquet(
        WORK / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    assert sol.optimal
    assert abs(sol.obj - flextool_obj) / max(1.0, flextool_obj) < 1e-6
