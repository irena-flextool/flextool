"""Benders with region units not listed in their region group (the documented setup lists nodes only).

The split keeps an unlisted unit in every region as a shared process; only the region of its nodes has balance
rows for it to serve, so that region decides its investment.  Before the fix the investment assembly after the
loop took a region's investments only from listed units: an investable unlisted unit had no value and the final
solve stopped with "no Benders value for v_invest_p".

The toy (:mod:`tests.engine._benders_two_region`, ``regular`` link, units unlisted) is solved with Benders and as
a monolith.  Oracles: the hand derivation in the helper (genB 100 MW, objective = its annuity) and the monolith's
objective.  This is also the ``regular``-link control of the one-variable-connection tests (item 301): it shows
the toy converges on the plain region-region path.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.solve import _region_filter
from tests.engine import _benders_two_region as two_region
from tests.engine._unit_db import flow, unitsize, utf8

pytestmark = pytest.mark.solver

CAPACITY_MW = 100.0       # hand: genB's capacity (helper docstring)


def _run(tmp_path_factory, decomposition: str):
    from flextool.engine import run_chain_from_db

    url, scen = two_region.build(tmp_path_factory, transfer_method="regular", decomposition=decomposition)
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"unlisted_{decomposition}_w"),
                              keep_solutions=True)
    return steps[two_region.SOLVE]


@pytest.fixture(scope="module")
def benders(tmp_path_factory):
    return _run(tmp_path_factory, "benders")


@pytest.fixture(scope="module")
def mono(tmp_path_factory):
    return _run(tmp_path_factory, "none")


def test_unlisted_units_take_the_region_of_their_nodes():
    pss = pl.DataFrame({"p": ["genA", "genB", "link", "link", "genX", "genX", "genM"],
                        "source": ["genA", "genB", "rA", "rB", "rA", "genX", "genM"],
                        "sink": ["rA", "rB", "rB", "rA", "genX", "rB", "hub"]})
    got = _region_filter.process_region_by_nodes(pss, {"region_A": {"rA", "rA_aux"}, "region_B": {"rB"}})
    # The cross-region link, the unit with nodes in two regions (genX) and the one with none (genM) are left out.
    assert got == {"genA": "region_A", "genB": "region_B"}


def test_benders_converges_to_the_hand_investment(benders):
    assert benders.is_benders and benders.optimal, (benders.benders_gap, benders.benders_iterations)
    assert benders.benders_results == "full"
    invest = utf8(benders.solution.value("v_invest_p")).filter(pl.col("p") == "genB")
    assert invest.height == 1
    assert invest["value"][0] * unitsize(benders, "genB") == pytest.approx(CAPACITY_MW, abs=1e-6)


def test_objective_is_the_hand_annuity_and_the_monolith(benders, mono):
    hand = two_region.hand_objective(CAPACITY_MW)
    assert hand == pytest.approx(764_215.116, abs=1e-3)   # the formula, evaluated by hand
    assert mono.optimal and not mono.is_benders
    assert float(mono.obj) == pytest.approx(hand, rel=1e-6)
    assert float(benders.solution.obj) == pytest.approx(hand, rel=1e-6)
    assert float(benders.solution.obj) == pytest.approx(float(mono.obj), rel=1e-6)


def test_hourly_flows_are_the_hand_values(benders):
    first = [1.0 if i < 24 else 0.0 for i in range(two_region.HOURS)]
    gen_a = flow(benders, "genA", "genA", "rA").to_list()
    gen_b = flow(benders, "genB", "genB", "rB").to_list()
    signed = (flow(benders, "link", "rA", "rB") - flow(benders, "link", "rB", "rA")).to_list()
    assert gen_a == pytest.approx([100.0] * two_region.HOURS, abs=1e-6)
    assert gen_b == pytest.approx([100.0] * two_region.HOURS, abs=1e-6)
    assert signed == pytest.approx([-100.0 if f else 100.0 for f in first], abs=1e-6)
    for name in ("vq_state_up", "vq_state_down"):
        frame = benders.solution.value(name)
        assert frame.height == 0 or float(frame["value"].abs().max()) <= 1e-6, name
