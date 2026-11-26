"""Benders: a unit whose nodes are in more than one region is solved in the master (item 303).

Before, the split kept a unit that a region group lists in that region only: its arcs at another region's nodes
had no balance there, so that flow was free disposal and the Benders loop optimised a wrong model (an unlisted
such unit was refused).  Now the unit is in the master: its flows, conversion, capacity, investment and costs.
Each region with a node of the unit sees its flow there as a half-flow pinned to the master's flow.

Toy: :mod:`tests.engine._benders_two_region` with ``link`` 50 MW and the unit ``dual``: input ``fuelX`` (commodity
``fuelc``, price 0.1 per MWh), outputs rA and rB, efficiency 1, 1000 MW existing; listed in region_A (or unlisted).

Hand oracle.  The 48-hour solve's operational cost is scaled to a year: x 8760/48 = 182.5.  genB's annuity is
7 642.15 per MW (``hand_objective(1)``); one MW of genB saves at most 48 MWh of dual's fuel, 48 x 0.1 x 182.5 =
876, so genB = 0.  Hours 1-24: rA needs 200 = genA 100 + dual 100.  Hours 25-48: rB needs 200 = link 50 (genA's
spare) + dual 150.  Dual's fuel is 24 x 100 + 24 x 150 = 6 000 MWh, the objective 6 000 x 0.1 x 182.5 = 109 500,
no unserved energy.  Before the fix region_B did not see dual: Benders invested 150 MW in genB (objective about
1.5 x 764 215.116 + 1 200 MWh x 0.1 x 182.5 = 1 168 222.7).  Second oracle: the monolith of the same model.
"""
from __future__ import annotations

import contextlib
import io

import polars as pl
import pytest

import flextool.engine.solve._benders as fx_benders
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.solve import _region_filter
from tests.engine import _benders_two_region as two_region
from tests.engine._unit_db import b64, flow, unitsize, utf8

pytestmark = pytest.mark.solver

LINK_MW = 50.0
HAND_OBJECTIVE = 6_000 * 0.1 * 8760 / 48          # 109 500


def _dual(listed: bool = True, extra_entities=(), extra_values=()):
    ents = [["commodity", "fuelc", None], ["node", "fuelX", None], ["commodity__node", ["fuelc", "fuelX"], None],
            ["unit", "dual", None], ["unit__inputNode", ["dual", "fuelX"], None],
            ["unit__outputNode", ["dual", "rA"], None], ["unit__outputNode", ["dual", "rB"], None]]
    if listed:
        ents.append(["group__unit", ["region_A", "dual"], None])
    vals = [["node", "fuelX", "node_type", b64("commodity", "str")],
            ["commodity", "fuelc", "price", b64(0.1, "float")],
            ["unit", "dual", "existing", b64(1000.0, "float")],
            ["unit", "dual", "efficiency", b64(1.0, "float")]]
    return ents + list(extra_entities), vals + list(extra_values)


def _run(tmp_path_factory, alt: str, decomposition: str = "benders", listed: bool = True):
    from flextool.engine import run_chain_from_db

    ents, vals = _dual(listed)
    url, scen = two_region.build(tmp_path_factory, transfer_method="regular", link_mw=LINK_MW,
                                 decomposition=decomposition, alt=alt, extra_entities=ents, extra_values=vals)
    results = []
    real = fx_benders.solve_benders

    def _capture(*args, **kwargs):
        results.append(real(*args, **kwargs))
        return results[-1]

    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        mp.setattr(fx_benders, "solve_benders", _capture)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)
    return steps[two_region.SOLVE], (results[-1] if results else None), out.getvalue()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return {"listed": _run(tmp_path_factory, "span_listed"),
            "unlisted": _run(tmp_path_factory, "span_unlisted", listed=False),
            "mono": _run(tmp_path_factory, "span_mono", decomposition="none")}


def _benders_input(tmp_path_factory, alt: str, listed: bool = True, extra_entities=(), extra_values=()):
    """The whole-system FlexData and the regions ``solve_benders`` gets (the run stops right there)."""
    from flextool.engine import run_chain_from_db

    ents, vals = _dual(listed, extra_entities, extra_values)
    url, scen = two_region.build(tmp_path_factory, transfer_method="regular", link_mw=LINK_MW, alt=alt,
                                 extra_entities=ents, extra_values=vals)
    got = {}

    class _Stop(Exception):
        pass

    def _capture(data, regions, *args, **kwargs):
        got["data"], got["regions"] = data, regions
        raise _Stop

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fx_benders, "solve_benders", _capture)
        with pytest.raises(_Stop):
            run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    return got["data"], got["regions"]


@pytest.mark.parametrize("case", ["listed", "unlisted"])
def test_benders_total_is_the_hand_value_and_the_monolith(runs, case):
    step, res, _ = runs[case]
    mono = runs["mono"][0]
    assert step.is_benders and step.optimal, (step.benders_gap, step.benders_iterations)
    assert res.total_objective == pytest.approx(HAND_OBJECTIVE, rel=1e-6)
    assert float(step.solution.obj) == pytest.approx(HAND_OBJECTIVE, rel=1e-6)
    assert float(mono.obj) == pytest.approx(HAND_OBJECTIVE, rel=1e-6)


@pytest.mark.parametrize("case", ["listed", "unlisted"])
def test_no_investment_and_hand_flows(runs, case):
    step = runs[case][0]
    invest = utf8(step.solution.value("v_invest_p")).filter(pl.col("p") == "genB")
    assert invest["value"].sum() * unitsize(step, "genB") == pytest.approx(0.0, abs=1e-6)
    first = [i < 24 for i in range(two_region.HOURS)]
    fuel = flow(step, "dual", "fuelX", "dual").to_list()
    assert fuel == pytest.approx([100.0 if f else 150.0 for f in first], abs=1e-6)
    for name in ("vq_state_up", "vq_state_down"):
        frame = step.solution.value(name)
        assert frame.height == 0 or float(frame["value"].abs().max()) <= 1e-6, name


def test_run_log_counts_the_master_arcs(runs):
    # link; dual <- fuelX; dual -> rA, dual -> rB
    assert "Benders master: 1 connection, 1 unit__inputNode arc, 2 unit__outputNode arcs" in runs["listed"][2]


def test_split_puts_the_unit_in_the_master(tmp_path_factory):
    data, regions = _benders_input(tmp_path_factory, "span_split")
    membership = _region_filter.load_region_membership(data, regions)
    assert _region_filter.master_units(data, {r: m["nodes"] for r, m in membership.items()}) == {"dual"}
    splits = {s.region: s for s in _region_filter.split(data, regions=regions, region_membership=membership,
                                                        benders_uncap_cross_region=True)}
    for region, node in (("region_A", "rA"), ("region_B", "rB")):
        dual_hf = [h for h in splits[region].half_flows if h.original_p == "dual"]
        assert [(h.side, h.in_region_node) for h in dual_hf] == [("import", node)], region
        pss = utf8(splits[region].data.process_source_sink)
        assert pss.filter(pl.col("p") == "dual").height == 0, region
    master = _region_filter.master_network_data(data, regions, region_membership=membership)
    arcs = set(utf8(master.process_source_sink.select("p", "source", "sink")).iter_rows())
    assert {a for a in arcs if a[0] == "dual"} == {("dual", "fuelX", "dual"), ("dual", "dual", "rA"),
                                                  ("dual", "dual", "rB")}
    assert _region_filter.master_arc_counts(data, regions, membership) == (1, 1, 2)


def test_unit_listed_in_a_region_without_its_nodes_is_refused(tmp_path_factory):
    """genA (output rA, region_A) listed in region_B."""
    data, regions = _benders_input(tmp_path_factory, "span_misplaced",
                                   extra_entities=[["group__unit", ["region_B", "genA"], None]])
    with pytest.raises(FlexToolConfigError, match=r"genA.*region_B.*region_A"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)


def test_reserve_on_a_master_unit_is_refused(tmp_path_factory):
    ents = [["reserve", "upR", None], ["group", "res_grp", None], ["group__node", ["res_grp", "rB"], None],
            ["reserve__upDown__group", ["upR", "up", "res_grp"], None],
            ["reserve__upDown__unit__node", ["upR", "up", "dual", "rB"], None]]
    vals = [["reserve__upDown__group", ["upR", "up", "res_grp"], "reservation", b64(10.0, "float")],
            ["reserve__upDown__group", ["upR", "up", "res_grp"], "reserve_method", b64("timeseries_only", "str")],
            ["reserve__upDown__group", ["upR", "up", "res_grp"], "penalty_reserve", b64(5000.0, "float")],
            ["reserve__upDown__unit__node", ["upR", "up", "dual", "rB"], "max_share", b64(1.0, "float")],
            ["reserve__upDown__unit__node", ["upR", "up", "dual", "rB"], "reliability", b64(1.0, "float")],
            ["reserve__upDown__unit__node", ["upR", "up", "dual", "rB"], "is_enabled", b64("yes", "str")]]
    data, regions = _benders_input(tmp_path_factory, "span_reserve", extra_entities=ents, extra_values=vals)
    with pytest.raises(FlexToolConfigError, match=r"dual.*reserve"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)


def test_integer_online_on_a_master_unit_is_refused(tmp_path_factory):
    vals = [["unit", "dual", "startup_method", b64("integer", "str")]]
    data, regions = _benders_input(tmp_path_factory, "span_integer", extra_values=vals)
    with pytest.raises(FlexToolConfigError, match=r"dual.*integer"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)


# ---------------------------------------------------------------------------
# lh2_master_build on the plain path (D20): liquefier_B, listed in region_B, takes h2_B and feeds lh2_B and lh2_A
# (region_A).  From the fixture: the master holds pipe_AB, pipe_BC and liquefier_B (input h2_B; outputs lh2_B,
# lh2_A); region_A sees liquefier_B as one import into lh2_A, region_B as an export from h2_B and an import into
# lh2_B.  Oracle: with the monolith's flows pinned the decomposition reproduces the monolith's objective M (before,
# region_A did not see the lh2_A output, so it could not); a short loop keeps its lower bound <= M.
# ---------------------------------------------------------------------------

LH2_REGIONS = ["region_A", "region_B", "region_C"]
OBJ_SCALE = 1e-6       # the production scale_the_objective (cf. test_benders_master_hosted)


@pytest.fixture(scope="module")
def mb_data(scenario_run):
    from tests._support.load import load_scenario

    return load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_master_build"))


def test_lh2_liquefier_b_is_in_the_master(mb_data):
    membership = _region_filter.load_region_membership(mb_data, LH2_REGIONS)
    region_nodes = {r: m["nodes"] for r, m in membership.items()}
    assert _region_filter.master_units(mb_data, region_nodes) == {"liquefier_B"}
    assert _region_filter.master_arc_counts(mb_data, LH2_REGIONS, membership) == (2, 1, 2)
    splits = {s.region: s for s in _region_filter.split(mb_data, regions=LH2_REGIONS, region_membership=membership,
                                                        benders_uncap_cross_region=True)}
    sides = {r: {(h.side, h.in_region_node) for h in s.half_flows if h.original_p == "liquefier_B"}
             for r, s in splits.items()}
    assert sides == {"region_A": {("import", "lh2_A")}, "region_B": {("export", "h2_B"), ("import", "lh2_B")},
                     "region_C": set()}
    for r, s in splits.items():
        assert utf8(s.data.process_source_sink).filter(pl.col("p") == "liquefier_B").height == 0, r
    master = _region_filter.master_network_data(mb_data, LH2_REGIONS, region_membership=membership)
    arcs = set(utf8(master.process_source_sink.select("p", "source", "sink")).iter_rows())
    assert {a for a in arcs if a[0] == "liquefier_B"} == {
        ("liquefier_B", "h2_B", "liquefier_B"), ("liquefier_B", "liquefier_B", "lh2_B"),
        ("liquefier_B", "liquefier_B", "lh2_A")}
    assert "liquefier_B" in set(utf8(master.pd_invest_set)["p"].to_list())


def test_lh2_decomposition_reproduces_the_monolith(mb_data, tmp_path, monkeypatch):
    from polar_high import Problem

    from flextool.engine import build_flextool

    pb = Problem()
    build_flextool(pb, mb_data)
    mono = pb.solve()
    assert mono.optimal
    M = mono.obj
    flows = mono.value("v_flow").select(
        [pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink", "d", "t")] + [pl.col("value")])
    # A coarse process of the monolith (liquefier_B and the pipes, every node daily) has one rate per day, held
    # over its hours (model/_coarse_flows); the master keeps every flow at the solve's step, so the point carries
    # that rate at each hour of the day.
    steps = utf8(mb_data.process_coarse_step.select("p", "d", "b_first", "t"))
    coarse = flows.join(steps.select("p").unique(), on="p", how="semi")
    expanded = (coarse.rename({"t": "b_first"}).join(steps, on=["p", "d", "b_first"], how="inner")
                .select("p", "source", "sink", "d", "t", "value"))
    flows = pl.concat([flows.join(steps.select("p").unique(), on="p", how="anti"), expanded])
    point = tmp_path / "monolith_flows.parquet"
    flows.write_parquet(point)
    monkeypatch.setenv("FLEXTOOL_BENDERS_PIN_DIAGNOSTIC", str(point))
    pinned = fx_benders.solve_benders(mb_data, LH2_REGIONS, max_iters=5, tol=1e-4, monolith_objective=M,
                                      scale_the_objective=OBJ_SCALE)
    monkeypatch.delenv("FLEXTOOL_BENDERS_PIN_DIAGNOSTIC")
    assert pinned.converged, "a region hit its penalty with the monolith's flows pinned"
    assert pinned.total_objective == pytest.approx(M, rel=1e-9)

    res = fx_benders.solve_benders(mb_data, LH2_REGIONS, max_iters=10, tol=1e-4, monolith_objective=M,
                                   scale_the_objective=OBJ_SCALE)
    assert res.lower_bound <= M * (1 + 1e-9), (res.lower_bound, M)


# ---------------------------------------------------------------------------
# A group investment limit over the master and a region (review R1).  dual (in the master) investing next to genB
# (region_B): before, region_B bounded genB and the master bounded dual, each by the whole limit (Benders 50 + 120 =
# 170 MW under a maximum of 120; the final solve with those investments was infeasible).  Now refused; a group with
# only master members is allowed and applied once.
# ---------------------------------------------------------------------------

DUAL_INVEST = [["unit", "dual", "invest_method", b64("invest_period", "str")],
               ["unit", "dual", "invest_cost", b64(10.0, "float")],
               ["unit", "dual", "lifetime_technical", b64(two_region.LIFETIME, "float")],
               ["unit", "dual", "invest_max_period",
                b64({"index_type": "str", "index_name": "period", "data": [["p2020", 1000.0]]}, "map")]]


def _limit_group(members: list[str], param: str, value: float) -> tuple[list, list]:
    ents = [["group", "lim", None]] + [["group__unit", ["lim", m], None] for m in members]
    vals = [["group", "lim", "invest_method", b64("invest_period", "str")],
            ["group", "lim", param, b64({"index_type": "str", "index_name": "period", "data": [["p2020", value]]},
                                        "map")]]
    return ents, vals


def _dual_invest_values() -> list:
    # dual greenfield (existing 0 instead of 1000): the toy's values with that one replaced
    return [v for v in _dual()[1] if v[:3] != ["unit", "dual", "existing"]] + [
        ["unit", "dual", "existing", b64(0.0, "float")]] + DUAL_INVEST


@pytest.mark.parametrize("case", ["max_unlisted", "min_unlisted", "max_listed_with_genB"])
def test_group_limit_over_the_master_and_a_region_is_refused(tmp_path_factory, case):
    param, value = ("invest_min_period", 1.0) if case.startswith("min") else ("invest_max_period", 120.0)
    ents, vals = _limit_group(["dual", "genB"], param, value)
    ents += [["group__unit", ["region_B", "genB"], None]]
    if case == "max_listed_with_genB":
        ents += [["group__unit", ["region_B", "dual"], None]]
    data, regions = _benders_input_raw(tmp_path_factory, f"lim_{case}", ents, vals, listed=False)
    word = "limit" if case.startswith("min") else "maximum"
    with pytest.raises(RuntimeError, match=rf"investment {word} group 'lim' has entities in regions region_B and the "
                                           r"Benders master \(region_B: genB; the Benders master: dual\)"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)


def _benders_input_raw(tmp_path_factory, alt, extra_entities, extra_values, listed=True):
    """:func:`_benders_input` with dual greenfield and investable."""
    from flextool.engine import run_chain_from_db

    ents = _dual(listed)[0] + list(extra_entities)
    vals = _dual_invest_values() + list(extra_values)
    url, scen = two_region.build(tmp_path_factory, transfer_method="regular", link_mw=LINK_MW, alt=alt,
                                 extra_entities=ents, extra_values=vals)
    got = {}

    class _Stop(Exception):
        pass

    def _capture(data, regions, *args, **kwargs):
        got["data"], got["regions"] = data, regions
        raise _Stop

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fx_benders, "solve_benders", _capture)
        with pytest.raises(_Stop):
            run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    return got["data"], got["regions"]


#: Hand oracle of the allowed case: lim = {dual} with a maximum of 100 MW in p2020; dual greenfield, invest_cost 10
#: per kW (annuity of 100 MW = hand_objective(100) / 10 = 76 421.51), fuel 0.1 per MWh (x 182.5 = 18.25).
#: dual's capacity C bounds each output arc (units.md: per-edge capacity).  Hours 25-48: rB needs 200 = link 50 +
#: dual + genB, so dual <= C <= 100 leaves genB >= 50 (G); one MW of genB (7 642.15) is worth more than one of dual
#: (764.22) plus the fuel it saves (at most 48 x 18.25 = 876), so C = 100, G = 50.  Hours 1-24: rA needs 100 beyond
#: genA = dual + the link's reverse flow (<= 50, from genB's free 50 MW): dual 50.  Fuel 24 x 50 + 24 x 100 = 3 600
#: MWh.  Objective = 76 421.512 + hand_objective(50) + 3 600 x 0.1 x 182.5.
ALLOWED_OBJECTIVE = (two_region.hand_objective(100.0) / 10 + two_region.hand_objective(50.0)
                     + 3_600 * 0.1 * 8760 / 48)


@pytest.fixture(scope="module")
def allowed(tmp_path_factory):
    from flextool.engine import run_chain_from_db

    out = {}
    for dec in ("benders", "none"):
        ents, vals = _limit_group(["dual"], "invest_max_period", 100.0)
        url, scen = two_region.build(tmp_path_factory, transfer_method="regular", link_mw=LINK_MW,
                                     decomposition=dec, alt=f"lim_allowed_{dec}",
                                     extra_entities=_dual(False)[0] + ents,
                                     extra_values=_dual_invest_values() + vals)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"lim_allowed_{dec}_w"),
                                  keep_solutions=True)
        out[dec] = steps[two_region.SOLVE]
    return out


def test_group_limit_of_master_members_only_is_applied_once(allowed):
    assert ALLOWED_OBJECTIVE == pytest.approx(524_229.07, abs=0.01)   # the formula, evaluated by hand
    step, mono = allowed["benders"], allowed["none"]
    assert step.is_benders and step.optimal, (step.benders_gap, step.benders_iterations)
    assert float(mono.obj) == pytest.approx(ALLOWED_OBJECTIVE, rel=1e-6)
    assert float(step.solution.obj) == pytest.approx(ALLOWED_OBJECTIVE, rel=1e-6)
    invest = utf8(step.solution.value("v_invest_p"))
    mw = {p: invest.filter(pl.col("p") == p)["value"].sum() * unitsize(step, p) for p in ("dual", "genB")}
    assert mw == pytest.approx({"dual": 100.0, "genB": 50.0}, abs=1e-6)


# ---------------------------------------------------------------------------
# The other refusals of a master unit (Q4): each error names the unit.
# ---------------------------------------------------------------------------

def test_start_up_draw_from_a_region_node_is_refused(tmp_path_factory):
    ents = [["unit__inputNode", ["dual", "rA"], None]]
    vals = [["unit", "dual", "startup_method", b64("linear", "str")],
            ["unit__inputNode", ["dual", "rA"], "startup_input_method", b64("startup_only", "str")],
            ["unit__inputNode", ["dual", "rA"], "startup_energy", b64(1.0, "float")]]
    data, regions = _benders_input(tmp_path_factory, "span_startup", extra_entities=ents, extra_values=vals)
    with pytest.raises(FlexToolConfigError, match=r"dual \(start-up energy from node 'rA' of region_A"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)


def test_capacity_margin_group_at_a_region_node_is_refused(tmp_path_factory):
    ents = [["group", "cm_b", None], ["group__node", ["cm_b", "rB"], None]]
    vals = [["group", "cm_b", "capacity_margin_method", b64("absolute", "str")],
            ["group", "cm_b", "capacity_margin_absolute", b64(10.0, "float")],
            ["group", "cm_b", "penalty_capacity_margin", b64(100.0, "float")]]
    data, regions = _benders_input(tmp_path_factory, "span_margin", extra_entities=ents, extra_values=vals)
    with pytest.raises(FlexToolConfigError, match=r"dual \(capacity margin group 'cm_b' at one of its nodes"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)


def test_flow_group_limit_on_a_master_unit_is_refused(tmp_path_factory):
    ents = [["flowGroup", "fg_dual", None], ["flowGroup__unit__node", ["fg_dual", "dual", "rB"], None]]
    vals = [["flowGroup", "fg_dual", "max_instant_flow", b64(1.0e6, "float")]]
    data, regions = _benders_input(tmp_path_factory, "span_flowgroup", extra_entities=ents, extra_values=vals)
    with pytest.raises(FlexToolConfigError, match=r"dual \(flowGroup 'fg_dual' \(unit 'dual'\)"):
        _region_filter.split(data, regions=regions, benders_uncap_cross_region=True)
