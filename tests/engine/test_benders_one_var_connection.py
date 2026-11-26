"""Benders with a one-variable connection between regions (item 301).

A ``no_losses_no_variable_cost`` connection has one arc ``(link, rA, rB)``: the forward flow ``v_flow`` and the
reverse flow ``v_flow_back`` (signed flow = v_flow - v_flow_back).  The split cuts the forward flow into a
half-flow pair coupled through the master; before the fix it left the reverse column in every region with no
capacity, no cost and no coupling: a free unlimited source at one end of the connection.  Benders then found
genB = 0, and the final solve paid unserved energy at both nodes.  Now each half-flow is one-variable too and its
``v_flow_back`` is coupled like ``v_flow``; the master holds both directions and the capacity.

Toy: :mod:`tests.engine._benders_two_region` (units unlisted, ``benders_max_iter`` 400, tolerance 1e-6).

Hand oracle (independent of the code; the helper's docstring has the efficiency-1 derivation):

* eq (efficiency 1).  Hours 1-24: rA needs 200 = genA (<= 100) + reverse flow w, and w <= genB's capacity C, so
  C >= 100.  Hours 25-48: rB needs 200 = genB + forward flow g, g <= genA's 100, so C >= 100.  VOLL (1000 per
  MWh) is far above the annuity: C = 100 MW, objective = genB's annuity ``hand_objective(100)`` = 764 215.116.
  genA = genB = 100 every hour, signed link flow -100 (hours 1-24) and +100 (25-48), no unserved energy.
* conv (efficiency 0.5, rA in MWh, rB in t: a unit conversion, 1 t at rB = 2 MWh at rA).  Forward: rA loses 2g,
  rB gains g.  Reverse: rB loses w, rA gains 2w.  Hours 1-24: rA short 100 MWh = 2w -> w = 50 t <= C.  Hours
  25-48: rB needs 100 t = genB + g, genA's spare 100 MWh gives g = 50 t, so genB = 50 -> C = 50.  The cost is
  linear in C: objective = hand_objective(50) = 764 215.116 / 2 = 382 107.558.  genA = 100, genB = 50 every hour,
  signed flow -50 / +50 t.
* region<->master (rB hosted in the master, a third region region_C {rC} with genC serving rC's 10 MW at no cost):
  the same C = 100 and objective; the coupling is single-sided (only region_A has half-flows).

Second oracle: the monolith (``decomposition`` none) of the same model.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve import _benders, _region_filter
from tests.engine import _benders_two_region as two_region
from tests.engine._unit_db import b64, cstr_over, flow, unitsize, utf8

pytestmark = pytest.mark.solver

ONE_VAR = "no_losses_no_variable_cost"
HOURS = two_region.HOURS
FIRST = [i < 24 for i in range(HOURS)]           # hours 1-24
# Hand values (module docstring).
CASES = {
    "eq": dict(capacity=100.0, objective=764_215.116, gen_b=100.0, signed=100.0),
    "conv": dict(capacity=50.0, objective=382_107.558, gen_b=50.0, signed=50.0),
    "mh": dict(capacity=100.0, objective=764_215.116, gen_b=100.0, signed=100.0),
}
BUILD = {
    "eq": dict(transfer_method=ONE_VAR),
    "conv": dict(transfer_method=ONE_VAR, efficiency=0.5, uom=True),
    "mh": dict(transfer_method=ONE_VAR, master_hosted_rB=True),
    "mh_regular": dict(transfer_method="regular", master_hosted_rB=True),
}


def _run(tmp_path_factory, case: str, decomposition: str = "benders"):
    """Run the case; returns ``(final solve step, BendersResult or None)``."""
    from flextool.engine import run_chain_from_db

    url, scen = two_region.build(tmp_path_factory, decomposition=decomposition,
                                 alt=f"one_var_{case}_{decomposition}", **BUILD[case])
    results = []
    real = _benders.solve_benders

    def _capture(*args, **kwargs):
        res = real(*args, **kwargs)
        results.append(res)
        return res

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_benders, "solve_benders", _capture)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"one_var_{case}_w"),
                                  keep_solutions=True)
    return steps[two_region.SOLVE], (results[-1] if results else None)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return {case: _run(tmp_path_factory, case) for case in ("eq", "conv", "mh", "mh_regular")}


@pytest.fixture(scope="module")
def monos(tmp_path_factory):
    return {case: _run(tmp_path_factory, case, "none")[0] for case in ("eq", "mh")}


def _gen_b_mw(step) -> float:
    invest = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == "genB")
    assert invest.height == 1
    return float(invest["value"][0]) * unitsize(step, "genB")


def _signed_link(step) -> list[float]:
    back = (utf8(step.solution.value("v_flow_back"))
            .filter((pl.col("p") == "link") & (pl.col("source") == "rA") & (pl.col("sink") == "rB"))
            .sort("d", "t"))
    assert back.height == HOURS
    return (flow(step, "link", "rA", "rB") - back["value"] * unitsize(step, "link")).to_list()


@pytest.mark.parametrize("case", ["eq", "conv", "mh"])
def test_benders_converges_to_the_hand_investment(runs, case):
    step, _ = runs[case]
    assert step.is_benders and step.optimal, (step.benders_gap, step.benders_iterations)
    assert step.benders_results == "full"
    assert _gen_b_mw(step) == pytest.approx(CASES[case]["capacity"], abs=1e-6)


@pytest.mark.parametrize("case", ["eq", "conv", "mh"])
def test_objective_is_the_hand_value(runs, case):
    hand = CASES[case]["objective"]
    assert two_region.hand_objective(CASES[case]["capacity"]) == pytest.approx(hand, abs=1e-3)
    assert float(runs[case][0].solution.obj) == pytest.approx(hand, rel=1e-6)


@pytest.mark.parametrize("case", ["eq", "mh"])
def test_objective_is_the_monolith(runs, monos, case):
    mono = monos[case]
    assert mono.optimal and not mono.is_benders
    assert float(mono.obj) == pytest.approx(CASES[case]["objective"], rel=1e-6)
    assert float(runs[case][0].solution.obj) == pytest.approx(float(mono.obj), rel=1e-6)


def test_conversion_costs_half_the_monolith_of_efficiency_one(runs, monos):
    # The cost is linear in genB's capacity, which the conversion halves (module docstring).
    assert float(runs["conv"][0].solution.obj) == pytest.approx(float(monos["eq"].obj) / 2.0, rel=1e-6)


@pytest.mark.parametrize("case", ["eq", "conv", "mh"])
def test_hourly_flows_are_the_hand_values(runs, case):
    step, _ = runs[case]
    hand = CASES[case]
    # Only the signed flow is unique (v_flow and v_flow_back may both be positive at efficiency 1).
    assert _signed_link(step) == pytest.approx([-hand["signed"] if f else hand["signed"] for f in FIRST], abs=1e-6)
    assert flow(step, "genA", "genA", "rA").to_list() == pytest.approx([100.0] * HOURS, abs=1e-6)
    assert flow(step, "genB", "genB", "rB").to_list() == pytest.approx([hand["gen_b"]] * HOURS, abs=1e-6)
    for name in ("vq_state_up", "vq_state_down"):
        frame = step.solution.value(name)
        assert frame.height == 0 or float(frame["value"].abs().max()) <= 1e-6, name


def test_trade_flow_has_both_directions(runs):
    """The reverse coupling is keyed by its direction of travel: it does not overwrite the forward one."""
    step, res = runs["eq"]
    assert res is not None
    assert set(res.trade_flow) == {("link", "rA", "rB"), ("link", "rB", "rA")}
    fwd = res.trade_flow[("link", "rA", "rB")].sort("d", "t")
    back = res.trade_flow[("link", "rB", "rA")].sort("d", "t")
    assert set(utf8(back)["source"].to_list()) == {"rB"} and set(utf8(back)["sink"].to_list()) == {"rA"}
    signed = ((fwd["value"] - back["value"]) * unitsize(step, "link")).to_list()
    assert signed == pytest.approx([-100.0 if f else 100.0 for f in FIRST], abs=1e-6)


def test_region_master_regular_link_control(runs):
    """Diagnostic control: the same region<->master toy with a ``regular`` link (two arcs, no v_flow_back)
    converges to the hand values; a failure of the one-variable case above with this green is item 301's."""
    step, _ = runs["mh_regular"]
    assert step.is_benders and step.optimal
    assert _gen_b_mw(step) == pytest.approx(100.0, abs=1e-6)
    assert float(step.solution.obj) == pytest.approx(764_215.116, rel=1e-6)


# ---------------------------------------------------------------------------
# The split (no solve): the data handed to solve_benders, captured before it runs.


class _Stop(Exception):
    pass


def _benders_input(tmp_path_factory, alt: str, extra_entities=(), extra_values=()):
    """The FlexData and regions ``solve_benders`` gets for the eq toy plus the extras (it is not run)."""
    from flextool.engine import run_chain_from_db

    url, scen = two_region.build(tmp_path_factory, transfer_method=ONE_VAR, alt=alt,
                                 extra_entities=extra_entities, extra_values=extra_values)
    got = {}

    def _capture(data, regions, *args, **kwargs):
        got["data"], got["regions"] = data, regions
        raise _Stop

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_benders, "solve_benders", _capture)
        with pytest.raises(_Stop):
            run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    return got["data"], got["regions"]


def _dc_group(nodes: list[str], lines: list[str]) -> tuple[list, list]:
    ents = [["group", "dc_net", None]] + [["group__node", ["dc_net", n], None] for n in nodes]
    vals = ([["group", "dc_net", "transfer_method", b64("dc_power_flow_with_angles", "str")]]
            + [["connection", c, "reactance", b64(0.1, "float")] for c in lines])
    return ents, vals


def _arcs(frame) -> set[tuple[str, str, str]]:
    if frame is None:
        return set()
    return set(utf8(frame.select("p", "source", "sink")).iter_rows())


def test_split_couples_the_reverse_direction(tmp_path_factory):
    """In-region one-variable DC line ``link_in`` (rA -> rA_aux, a balance node of region_A) next to the
    cross-region ``link``."""
    ents, vals = _dc_group(["rA", "rA_aux"], ["link_in"])
    ents += [["node", "rA_aux", None], ["group__node", ["region_A", "rA_aux"], None],
             ["connection", "link_in", None], ["connection__node__node", ["link_in", "rA", "rA_aux"], None]]
    vals += [["node", "rA_aux", "node_type", b64("balance", "str")],
             ["connection", "link_in", "transfer_method", b64(ONE_VAR, "str")],
             ["connection", "link_in", "existing", b64(10.0, "float")]]
    data, regions = _benders_input(tmp_path_factory, "one_var_split", ents, vals)
    membership = _region_filter.load_region_membership(data, regions)
    splits = {s.region: s for s in _region_filter.split(data, regions=regions, region_membership=membership,
                                                        benders_uncap_cross_region=True)}
    assert set(splits) == {"region_A", "region_B"}       # an in-region DC line is allowed
    hf = {s: {(h.side, h.one_var) for h in splits[s].half_flows} for s in splits}
    assert hf == {"region_A": {("export", True)}, "region_B": {("import", True)}}
    exp = splits["region_A"].half_flows[0]
    imp = splits["region_B"].half_flows[0]
    a_arcs = _arcs(splits["region_A"].data.process_source_sink_2way_1var)
    b_arcs = _arcs(splits["region_B"].data.process_source_sink_2way_1var)
    assert a_arcs == {("link_in", "rA", "rA_aux"), (exp.virtual_p, exp.virtual_arc_source, exp.virtual_arc_sink)}
    # link_in is not listed in a region group, so like its v_flow arc it is in region_B too, inert there: neither
    # of its nodes has a balance in region_B.
    assert b_arcs == {(imp.virtual_p, imp.virtual_arc_source, imp.virtual_arc_sink), ("link_in", "rA", "rA_aux")}
    assert not {"rA", "rA_aux"} & set(utf8(splits["region_B"].data.nodeBalance)["n"].to_list())
    reduced = _region_filter.master_network_data(data, regions, region_membership=membership)
    assert _arcs(reduced.process_source_sink_2way_1var) == {("link", "rA", "rB")}

    from polar_high import Problem
    from flextool.engine import build_flextool

    m = Problem()
    build_flextool(m, reduced)
    back = utf8(m._vars["v_flow_back"].frame).filter(pl.col("p") == "link")
    assert set(back.select("p", "source", "sink").iter_rows()) == {("link", "rA", "rB")}
    assert back.height == HOURS
    # (The master also gets a v_flow_back on the in-region DC line from the DC arcs, which it does not filter:
    # a column in no balance and no capacity row, inert.)
    rows = cstr_over(m, "maxFlow_back")
    assert rows is not None and set(rows.select("p", "source", "sink").iter_rows()) == {("link", "rA", "rB")}
    assert rows.height == HOURS


def test_dc_line_between_regions_is_refused(tmp_path_factory):
    ents, vals = _dc_group(["rA", "rB"], ["link"])
    data, regions = _benders_input(tmp_path_factory, "one_var_dc_cross", ents, vals)
    with pytest.raises(FlexToolConfigError, match=r"DC power-flow line\(s\) between regions.*: link\..*"
                                                  r"Put both nodes of the line in one region, or take the "
                                                  r"nodes out of the dc_power_flow_with_angles group"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)
