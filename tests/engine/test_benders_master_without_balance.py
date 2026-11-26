"""The Benders master without a balance node (item 302).

On the plain path (every node in a region) the master holds only the flows between regions: none of its nodes has
a balance there.  Before, the master kept the region nodes that are no end of a cross-region arc as inert balance
rows, because the build refused a model without a balance node; with every balance node at the end of a
cross-region arc there was none left and the run stopped with "No node has an energy balance".

Toy: :mod:`tests.engine._benders_two_region` (``regular`` link): rA and rB are both ends of ``link``.  Hand oracle
(the helper's docstring): genB 100 MW, objective = genB's annuity ``hand_objective(100)`` = 764 215.116.  The
master LP has no ``nodeBalance_eq`` row and no balance slack column.  A user's model without a balance node is
still refused, when it is loaded.
"""
from __future__ import annotations

import pytest

import flextool.engine.solve._benders as fx_benders
from flextool.engine.solve import _region_filter
from tests.engine import _benders_two_region as two_region

pytestmark = pytest.mark.solver


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from flextool.engine import run_chain_from_db

    url, scen = two_region.build(tmp_path_factory, transfer_method="regular", alt="master_without_balance")
    reduced, masters = [], []
    real_reduce, real_build = _region_filter.master_network_data, fx_benders._build_flextool

    def _reduce(*args, **kwargs):
        reduced.append(real_reduce(*args, **kwargs))
        return reduced[-1]

    def _build(m, d, *args, **kwargs):
        real_build(m, d, *args, **kwargs)
        if any(d is r for r in reduced):
            masters.append(m)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_region_filter, "master_network_data", _reduce)
        mp.setattr(fx_benders, "_build_flextool", _build)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("master_without_balance_w"),
                                  keep_solutions=True)
    return steps[two_region.SOLVE], masters


def test_benders_solves_to_the_hand_objective(run):
    step, _ = run
    assert step.is_benders and step.optimal, (step.benders_gap, step.benders_iterations)
    assert float(step.solution.obj) == pytest.approx(two_region.hand_objective(100.0), rel=1e-6)


def test_master_has_no_balance_row_or_slack(run):
    _, masters = run
    assert len(masters) == 1
    m = masters[0]
    assert sum(r.over.height for r in m.cstrs_named("nodeBalance_eq")) == 0
    for name in ("vq_state_up", "vq_state_down"):
        var = m._vars.get(name)
        assert var is None or var.frame.height == 0, name
    assert m._vars["v_flow"].frame.height > 0


def test_model_without_balance_node_is_refused_at_load(tmp_path_factory):
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import b64, make_db

    url, scen = make_db(tmp_path_factory, alt="no_balance_node", base_chain=["init"],
                        entities=[["node", "only_commodity", None]],
                        values=[["node", "only_commodity", "node_type", b64("commodity", "str")]])
    with pytest.raises(Exception, match="No node has an energy balance"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("no_balance_node_w"))
