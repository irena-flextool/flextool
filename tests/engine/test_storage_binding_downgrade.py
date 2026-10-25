"""The plain storage binding sets keep the per-solve downgrade of blended-weights nodes (#127).

A solve without representative periods turns every ``*_blended_weights`` storage_binding_method into its plain
method (``_run_model._downgrade_rp_methods_for_non_rp_solve``, in the per-solve frame).  The binding sets must come
from that frame: a DB-direct projection of the raw DB value used to replace them.  Databases are built from
``tests/fixtures/lh2_three_region_rp_invest.json`` plus an override alternative, never a checked-in sqlite.
"""
from __future__ import annotations

import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build
from tests.engine.test_bind_within_period import RP_SOLVE, _run, _state_cols, _week168



def test_downgraded_within_solve_node_keeps_storage_next_to_plain_within_solve_nodes(tmp_path_factory):
    """A non-RP solve turns bind_within_solve_blended_weights into bind_within_solve.  The fixture's batteries
    are plain bind_within_solve in the DB; before, the DB-direct projection of that method replaced the
    solve's set, so the downgraded lh2 nodes dropped out and their storage had no state chain."""
    down = _run(tmp_path_factory, "lh2_three_region_rp_invest.json",
                _week168("bind_within_solve_blended_weights"))[RP_SOLVE]
    plain = _run(tmp_path_factory, "lh2_three_region_rp_invest.json", _week168("bind_within_solve"))[RP_SOLVE]
    members = sorted(str(n) for n in down.flex_data.storage_bind_within_solve["n"].to_list())
    assert {"lh2_A", "lh2_B", "lh2_C", "battery_A"} <= set(members)
    cols = _state_cols(row_terms(build(down.flex_data), "nodeBalanceCoarse_eq", "lh2_A,y2040,t0001")[
        "nodeBalanceCoarse_eq[lh2_A,y2040,t0001]"])
    assert cols["v_state[lh2_A,y2030,t0096]"] > 0          # within_solve: y2040 continues from y2030
    assert float(down.solution.obj) == pytest.approx(float(plain.solution.obj), rel=1e-9)
