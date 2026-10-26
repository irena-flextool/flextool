"""Which flows the non-synchronous limit counts (FA-11).

    infeed  = flagged unit arcs and DC connections into the group (v_flow, and v_flow_back of one-variable /
              DC arcs, which arrives at the arc's source) + positive inflow
    supply  = flows out of the group's nodes (v_flow, and v_flow_back leaving the arc's sink) + demand
    inside  = processes with a source and a different sink node in the group: counted in neither

Before: the inside set was empty on the single-solve path (internal lines and storage connections counted
both ways, and the optimiser circulated energy through them to loosen the limit), DC connections were
dropped as soon as one unit was flagged, and reverse flows were ignored.  The engine also warns about a
one-variable connection that crosses the boundary of a non-sync group (simultaneous flow both ways is free).
Every database is a JSON fixture plus an override alternative (canonical how-to JSON:
``initialize_database``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8

pytestmark = pytest.mark.solver

NET = ["init", "west", "coal", "wind", "network"]
LL = ["Base", "lossless_2way"]
WIND = [["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", b64("yes", "str")]]
DC_WN = [["connection", "west_north", "is_DC", b64("yes", "str")]]
ROLLING = [["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
           ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(36.0, "float")],
           ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(24.0, "float")]]
HOWTO = (Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "canonical_databases"
         / "howto_non_sync_and_curtailment.json")


def _ns_group(g: str, nodes: list[str], limit: float) -> tuple[list, list]:
    ents = [["group", g, None]] + [["group__node", [g, n], None] for n in nodes]
    vals = [["group", g, "has_non_synchronous", b64("yes", "str")],
            ["group", g, "non_synchronous_limit", b64(limit, "float")],
            ["group", g, "penalty_non_synchronous", b64(4000.0, "float")]]
    return ents, vals


def _steps(tmp_path_factory, alt: str, ents: list, vals: list, chain: list[str]) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=vals, base_chain=chain, entities=ents)
    return run(tmp_path_factory, url, scen)


def _pairs(frame: pl.DataFrame | None) -> set[tuple[str, str]]:
    if frame is None:
        return set()
    f = utf8(frame)
    return set(f.select(f.columns[0], f.columns[1]).iter_rows()) if f.height else set()


def _emitted(step, key: str) -> set[tuple[str, str]]:
    """The emit-side derive of ``key`` on the step's Provider (the legacy rule)."""
    frame = step.flex_data_provider.get(key)
    return _pairs(frame)


def _gwe(tmp_path_factory, alt: str, extra: list, chain: list[str] = NET) -> dict:
    ents, vals = _ns_group("gwe", ["west", "east"], 0.3)
    return _steps(tmp_path_factory, alt, ents, vals + WIND + extra, chain)


# ---------------------------------------------------------------------------
# internal transfers (inside) and DC connections; single and rolling solves


def test_internal_line_not_counted_single_solve(tmp_path_factory) -> None:
    """{west, east} with west_east inside: the inside set is {(west_east, gwe)} (= the emit derive), no
    west_east term in any of the 48 rows, and the optimum no longer circulates energy over the line."""
    step = list(_gwe(tmp_path_factory, "ns_net", []).values())[-1]
    fd = step.flex_data
    inside = _pairs(fd.process_group_inside_nonSync)
    assert inside == {("west_east", "gwe")}
    assert inside == _emitted(step, "solve_data/process__group_inside_group_nonSync")
    rows = row_terms(build(fd), "non_sync_constraint", "gwe,")
    assert len(rows) == 48
    for row, terms in rows.items():
        assert not [c for c in terms if c.startswith("v_flow[west_east,")], row
    assert step.solution.obj == pytest.approx(1_588_494_588.0718954, rel=1e-8)


def test_dc_connection_counts_with_flagged_unit(tmp_path_factory) -> None:
    """west_north is_DC with wind flagged: process_sink_nonSync holds both of its nodes (= the emit derive) and
    its flow north → west is non-synchronous infeed (+unitsize · Δt · icap = 100) in every row."""
    step = list(_gwe(tmp_path_factory, "ns_dc", DC_WN).values())[-1]
    fd = step.flex_data
    sink_ns = _pairs(fd.process_sink_nonSync)
    assert {("west_north", "west"), ("west_north", "north"), ("wind_plant", "west")} <= sink_ns
    assert sink_ns == _emitted(step, "solve_data/process__sink_nonSync")
    rows = row_terms(build(fd), "non_sync_constraint", "gwe,")
    assert len(rows) == 48
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[west_north,north,west,p2020,{t}]") == pytest.approx(100.0), row
    assert step.solution.obj == pytest.approx(1_590_055_502.7108438, rel=1e-8)


@pytest.mark.parametrize("variant", ["net", "dc"])
def test_rolling_rows_equal_single_solve(tmp_path_factory, variant: str) -> None:
    """Rolling dispatch (36 h / 24 h) of the same model: every roll has the single solve's inside set and
    non-sync row coefficients at its steps (the synthetic-solve path follows the same rule)."""
    extra = DC_WN if variant == "dc" else []
    single = list(_gwe(tmp_path_factory, f"ns_{variant}_1", extra).values())[-1]
    ref = row_terms(build(single.flex_data), "non_sync_constraint", "gwe,")
    rolls = _gwe(tmp_path_factory, f"ns_{variant}_r", extra + ROLLING)
    assert len(rolls) >= 2, list(rolls)
    for name, step in rolls.items():
        fd = step.flex_data
        assert _pairs(fd.process_group_inside_nonSync) == {("west_east", "gwe")}, name
        assert _pairs(fd.process_sink_nonSync) == _pairs(single.flex_data.process_sink_nonSync), name
        rows = row_terms(build(fd), "non_sync_constraint", "gwe,")
        assert rows, name
        for row, terms in rows.items():
            assert terms.keys() == ref[row].keys(), (name, row)
            for col, v in terms.items():
                # rel 1e-6: the roll sub-solves read p_slope rounded to 7 digits (1.111111 for 1 / 0.9), a separate
                # issue that touches every slope row, not only this one
                assert v == pytest.approx(ref[row][col], rel=1e-6), (name, row, col)


# ---------------------------------------------------------------------------
# reverse flows of a one-variable DC line (lossless_2way: ll exports 100 MW from lw at t0001, imports at t0002)


def test_reverse_flow_out_of_group_is_supply(tmp_path_factory) -> None:
    """Group {le}, ll is_DC: the reverse flow le → lw (v_flow_back[ll,lw,le]) leaves le, so it is supply
    (−0.5 · 200 = −100 in every row); at t0002 le exports 100 MW and the shortfall halves (vq 50, was 100)."""
    ents, vals = _ns_group("gle", ["le"], 0.5)
    step = list(_steps(tmp_path_factory, "ns_back_sink", ents, vals + [
        ["connection", "ll", "is_DC", b64("yes", "str")]], LL).values())[-1]
    rows = row_terms(build(step.flex_data), "non_sync_constraint", "gle,")
    assert len(rows) == 2
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(-100.0), row
    vq = {t: x for _g, _d, t, x in utf8(step.solution.value("vq_non_synchronous")).rows()}
    assert vq["t0002"] == pytest.approx(50.0, abs=1e-6)
    assert step.solution.obj == pytest.approx(1_752_000_000.0, rel=1e-9)


def test_reverse_flow_into_group_is_infeed(tmp_path_factory) -> None:
    """Group {lw}, ll is_DC: the reverse flow le → lw arrives at lw through a DC line, so it is non-synchronous
    infeed (+200 = unitsize · Δt · icap), and the forward flow lw → le is supply (−100).  Before: no reverse
    term, and the free circulation of the one-variable line hid the infeed (objective 0)."""
    ents, vals = _ns_group("glw", ["lw"], 0.5)
    step = list(_steps(tmp_path_factory, "ns_back_src", ents, vals + [
        ["connection", "ll", "is_DC", b64("yes", "str")]], LL).values())[-1]
    rows = row_terms(build(step.flex_data), "non_sync_constraint", "glw,")
    assert len(rows) == 2
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(200.0), row
        assert coef(terms, f"v_flow[ll,lw,le,p2020,{t}]") == pytest.approx(-100.0), row
    vq = {t: x for _g, _d, t, x in utf8(step.solution.value("vq_non_synchronous")).rows()}
    assert vq == pytest.approx({"t0001": 50.0, "t0002": 50.0}, abs=1e-6)


# ---------------------------------------------------------------------------
# canonical how-to (non_sync_and_curtailment): the battery connection is inside the group


def test_howto_battery_connection_inside(tmp_path_factory) -> None:
    """``curtailment``: nodeA_group = {nodeA, battery}, so con_battery is inside and has no term in the rows;
    the battery no longer cycles to inflate the supply side (objective 6.580699 → 8.788234 × 10⁶)."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp("ns_howto")
    db = work / "howto.sqlite"
    initialize_database(str(HOWTO), str(db))
    url = f"sqlite:///{db}"
    migrate_database(url)
    steps = run_chain_from_db(url, "curtailment", work_folder=work / "w", keep_solutions=True)
    step = list(steps.values())[-1]
    fd = step.flex_data
    assert ("con_battery", "nodeA_group") in _pairs(fd.process_group_inside_nonSync)
    rows = row_terms(build(fd), "non_sync_constraint")
    assert rows
    for row, terms in rows.items():
        assert not [c for c in terms if c.startswith("v_flow[con_battery,")], row
    assert step.solution.obj == pytest.approx(8_788_234.09095786, rel=1e-7)


# ---------------------------------------------------------------------------
# one-variable connection across the boundary (owner question Q-c15d-B: warning by default)


_WARN = "crosses the boundary of group"


@pytest.mark.parametrize("case,expect", [
    ("1var_crossing", True), ("regular_crossing", False), ("exact_crossing", False),
    ("1var_inside", False), ("1var_dc_crossing", False)])
def test_one_variable_crossing_warning(tmp_path_factory, caplog, case: str, expect: bool) -> None:
    """A one-variable west_north with one end in {west, east} warns once (naming the connection and the group,
    recommending exact); a regular or exact line, a one-variable line with both ends in the group and an is_DC
    one-variable line (its two-way flow tightens the limit) do not."""
    nodes = ["west", "east", "north"] if case == "1var_inside" else ["west", "east"]
    ents, vals = _ns_group("gwe", nodes, 0.3)
    method = {"1var": "no_losses_no_variable_cost", "regular": "regular", "exact": "exact"}[case.split("_")[0]]
    vals = vals + WIND + [["connection", "west_north", "transfer_method", b64(method, "str")]]
    if method == "no_losses_no_variable_cost":
        vals.append(["connection", "west_north", "efficiency", b64(1.0, "float")])
    if case == "1var_dc_crossing":
        vals += DC_WN
    with caplog.at_level(logging.WARNING):
        _steps(tmp_path_factory, f"ns_w_{case}", ents, vals, NET)
    hits = [r.getMessage() for r in caplog.records if _WARN in r.getMessage()]
    if expect:
        assert len(hits) == 1, hits
        assert "'west_north'" in hits[0] and "'gwe'" in hits[0] and "exact" in hits[0]
    else:
        assert not hits, hits


# ---------------------------------------------------------------------------
# two groups joined by a line (#128): what an export / import is in each group


@pytest.mark.parametrize("dc", [False, True])
def test_two_groups_joined_by_a_line(tmp_path_factory, dc: bool) -> None:
    """{west} and {east}, each with a limit 0.3, wind flagged at west, west_east between them.  In each group
    the line's flow out is supply (negative, × limit), its flow in is infeed only when the line is is_DC: wind
    exported over an AC line arrives in east as synchronous (the documented semantics, #128); over a DC line it
    is non-synchronous infeed of east (+unitsize · Δt · icap = 400)."""
    e1, v1 = _ns_group("gw", ["west"], 0.3)
    e2, v2 = _ns_group("ge", ["east"], 0.3)
    extra = [["connection", "west_east", "is_DC", b64("yes", "str")]] if dc else []
    step = list(_steps(tmp_path_factory, f"ns_two_{int(dc)}", e1 + e2, v1 + v2 + WIND + extra, NET).values())[-1]
    pb = build(step.flex_data)
    for g, out_arc, in_arc in (("gw", "west,east", "east,west"), ("ge", "east,west", "west,east")):
        rows = row_terms(pb, "non_sync_constraint", f"{g},")
        assert len(rows) == 48, g
        for row, terms in rows.items():
            t = row.split(",")[-1].rstrip("]")
            assert coef(terms, f"v_flow[west_east,{out_arc},p2020,{t}]") < 0.0, (g, row)
            infeed = [v for c, v in terms.items() if c == f"v_flow[west_east,{in_arc},p2020,{t}]"]
            if dc:
                assert infeed == [pytest.approx(400.0)], (g, row, terms)
            else:
                assert not infeed, (g, row)
        if g == "gw":
            for row, terms in rows.items():
                t = row.split(",")[-1].rstrip("]")
                assert [c for c in terms if c.startswith("v_flow[wind_plant,") and c.endswith(f",west,p2020,{t}]")]


def test_dc_power_flow_line_crossing_warning(tmp_path_factory, caplog) -> None:
    """west_north is a line of a dc_power_flow_with_angles group {west, north} (classified one-variable through
    the group override) and crosses {west, east}: one warning names the line and the group and the loophole,
    without recommending exact (a DC power-flow line has no exact option)."""
    ents, vals = _ns_group("gwe", ["west", "east"], 0.3)
    ents += [["group", "gdc", None], ["group__node", ["gdc", "west"], None], ["group__node", ["gdc", "north"], None]]
    vals = vals + WIND + [["group", "gdc", "transfer_method", b64("dc_power_flow_with_angles", "str")]]
    with caplog.at_level(logging.WARNING):
        _steps(tmp_path_factory, "ns_w_dcpf", ents, vals, NET)
    hits = [r.getMessage() for r in caplog.records if _WARN in r.getMessage()]
    assert len(hits) == 1, hits
    assert hits[0].startswith("DC power-flow line 'west_north'") and "'gwe'" in hits[0]
    assert "exact" not in hits[0]
