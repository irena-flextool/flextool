"""polar_high lh2_three_region parity — JSON-fixture scenario imported from
flextool's tests/fixtures/lh2_three_region.json (regen via
tests/_gen_lh2_three_region.py).

Closes at machine epsilon via:
  * arc-side block-aware aggregation in nodeBalanceBlock_eq (per-arc
    weight = block_step_duration of the relevant side block).
  * arc-block-compatibility filter on flow_to_n / flow_from_n /
    flow_from_nodeBalance_*: arcs whose side-block doesn't connect to
    a node's block via overlap_set are dropped from that node's
    nodeBalance_eq (the .mod's process_side_block restriction).
"""
from pathlib import Path

import polars as pl

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
import pytest

pytestmark = pytest.mark.solver


def _flextool_obj(work: Path) -> float:
    sc_file = work / "solve_data" / "solve_current.csv"
    if sc_file.exists():
        solve = pl.read_csv(sc_file)["solve"][0]
        parq = work / "output_raw" / f"v_obj__{solve}.parquet"
        if parq.exists():
            return pl.read_parquet(parq)["objective"][0]
    parq = list(work.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(sorted(parq)[-1])["objective"][0]
    return pl.read_csv(work / "output_raw" / "v_obj.csv")["objective"][0]


def test_lh2_three_region_parity(scenario_workdir):
    work = scenario_workdir("lh2_three_region", db_fixture="lh2")
    data = load_flextool(work)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    # Γ.4: prefer golden_obj.json if present, fall back to parquet.
    from _golden import has_golden, assert_obj_within
    if has_golden(work):
        assert_obj_within(sol.obj, work)
        return
    flextool_obj = _flextool_obj(work)
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"obj mismatch: polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )
