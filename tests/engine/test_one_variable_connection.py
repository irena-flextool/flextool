"""A one-variable connection (``no_losses_no_variable_cost``, DC power-flow lines) has one signed flow (package K).

* ``other_operational_cost`` is ignored (it would be a cost one way and a revenue the other): the LP and the
  objective are those without it, with W-0590 once per solve.
* ``efficiency`` ≠ 1 cannot be a loss (it would be a gain in the reverse direction).  It is ignored (slope 1,
  W-0591) on a DC power-flow line and when the two nodes share a ``unit_of_measure`` or either has none; between
  nodes in different units it is a conversion, the same in both directions: the forward flow takes f · slope from
  the source and the reverse flow gives f · slope back there (W-0589).

The model is ``lossless_2way`` of ``tests.json``: ``ll`` (200 MW) between lw and le, lw exports 100 MW at t0001
and imports 100 MW at t0002.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests._support.coded_warnings import capture_coded
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import b64, build, make_db, run

pytestmark = pytest.mark.solver

LL = ["Base", "lossless_2way"]
ROLLING = [["solve", "lossless_2way_solve", "solve_mode", b64("rolling_window", "str")],
           ["solve", "lossless_2way_solve", "rolling_solve_horizon", b64(1.0, "float")],
           ["solve", "lossless_2way_solve", "rolling_solve_jump", b64(1.0, "float")]]


#: lw in MWh and le in t: an efficiency of ll is a unit conversion.
UNITS = [["node", "lw", "unit_of_measure", b64("MWh", "str")], ["node", "le", "unit_of_measure", b64("t", "str")]]


def _solve(tmp_path_factory, alt: str, efficiency: float, cost: float | None, extra: list | None = None,
           entities: list | None = None):
    vals = [["connection", "ll", "efficiency", b64(efficiency, "float")]]
    if cost is not None:
        vals.append(["connection", "ll", "other_operational_cost", b64(cost, "float")])
    with capture_coded() as got:
        url, scen = make_db(tmp_path_factory, alt=alt, values=vals + (extra or []), base_chain=LL,
                            entities=entities)
        steps = run(tmp_path_factory, url, scen)
    return steps, got


def _lp(step) -> tuple[dict, dict]:
    """(nodeBalance_eq rows, objective coefficients by column) of the step's LP."""
    pb = build(step.flex_data)
    m = pb.canonicalise()
    return row_terms(pb, "nodeBalance_eq"), {m.col_names[i]: float(m.col_obj[i]) for i in range(m.n_cols)}


def test_variable_cost_is_ignored(tmp_path_factory) -> None:
    """With other_operational_cost 5 the LP (balance rows and objective coefficients) and the objective are
    those without a cost (before: 5 · Δt on the forward flow only, objective 2.19e9 with efficiency 0.5); one
    W-0590 names ll, no W-0589 with efficiency 1."""
    with_cost, got = _solve(tmp_path_factory, "k_cost", 1.0, 5.0)
    without, got0 = _solve(tmp_path_factory, "k_nocost", 1.0, None)
    a, b = list(with_cost.values())[-1], list(without.values())[-1]
    assert _lp(a) == _lp(b)
    assert a.solution.obj == pytest.approx(b.solution.obj, abs=1e-6)
    assert a.flex_data.p_pdt_varCost_process is None
    assert [c.names for c in got.calls if c.code == "W-0590"] == [["ll"]]
    assert not [c for c in got.calls + got0.calls if c.code == "W-0589"]
    assert not [c for c in got0.calls if c.code == "W-0590"]


@pytest.mark.parametrize("efficiency", [0.5, 2.0])
def test_efficiency_is_the_same_conversion_both_ways(tmp_path_factory, efficiency: float) -> None:
    """Nodes in different units: at lw (the source) the forward flow takes 200 / efficiency per unit and the reverse
    flow gives the same back; at le (the sink) both are ±200.  So the net transfer is one signed flow and sending
    the same amount both ways changes no node (before: the reverse flow gave 200 at lw, with efficiency 2 a free
    100 MW there).  One W-0589 names ll with its efficiency."""
    steps, got = _solve(tmp_path_factory, f"k_eff_{int(efficiency * 10)}", efficiency, None, UNITS)
    rows, _obj = _lp(list(steps.values())[-1])
    for t in ("t0001", "t0002"):
        lw = rows[f"nodeBalance_eq[lw,p2020,{t}]"]
        le = rows[f"nodeBalance_eq[le,p2020,{t}]"]
        fwd, back = f"v_flow[ll,lw,le,p2020,{t}]", f"v_flow_back[ll,lw,le,p2020,{t}]"
        assert lw[fwd] == pytest.approx(-200.0 / efficiency) and lw[back] == pytest.approx(200.0 / efficiency)
        assert le[fwd] == pytest.approx(200.0) and le[back] == pytest.approx(-200.0)
    calls = [c for c in got.calls if c.code == "W-0589"]
    assert [c.names for c in calls] == [["ll"]]
    assert got.detail("W-0589", "ll") == f"efficiency {efficiency:g}"


def test_warnings_once_per_rolling_solve(tmp_path_factory) -> None:
    """A rolling solve of two rolls prints each warning once (the rolls count as one solve)."""
    steps, got = _solve(tmp_path_factory, "k_roll", 0.5, 5.0, ROLLING)
    assert len(steps) >= 2, list(steps)
    for code in ("W-0591", "W-0590"):
        lines = [line for line in got.lines() if line.endswith(f"[{code}]")]
        assert len(lines) == 1, (code, got.lines())
        assert "'ll'" in lines[0], lines


def test_dc_power_flow_line_cost_and_efficiency_are_ignored(tmp_path_factory) -> None:
    """A line of a dc_power_flow_with_angles group is a one-variable connection too: its other_operational_cost
    is dropped with W-0590, and its efficiency is ignored (slope 1, W-0591) even between nodes in different
    units."""
    ents = [["group", "gdc", None], ["group__node", ["gdc", "west"], None], ["group__node", ["gdc", "north"], None]]
    vals = [["group", "gdc", "transfer_method", b64("dc_power_flow_with_angles", "str")],
            ["connection", "west_north", "other_operational_cost", b64(3.0, "float")],
            ["connection", "west_north", "efficiency", b64(0.9, "float")],
            ["node", "west", "unit_of_measure", b64("MWh", "str")],
            ["node", "north", "unit_of_measure", b64("MWh_th", "str")]]
    with capture_coded() as got:
        url, scen = make_db(tmp_path_factory, alt="k_dcpf", values=vals, entities=ents,
                            base_chain=["init", "west", "coal", "wind", "network"])
        steps = run(tmp_path_factory, url, scen)
    assert "west_north" in got.names("W-0590")
    assert "west_north" in got.names("W-0591") and "west_north" not in got.names("W-0589")
    for step in steps.values():
        slope = step.flex_data.p_slope.frame.filter(pl.col("p").cast(pl.Utf8) == "west_north")
        assert slope.height > 0 and (slope.get_column("value") == 1.0).all()
        cost = step.flex_data.p_pdt_varCost_process
        assert cost is None or "west_north" not in {str(p) for p in cost.frame.get_column("p").to_list()}


@pytest.mark.parametrize("units", [None, "same", "one"], ids=["no_units", "same_units", "one_unit"])
def test_efficiency_without_a_unit_conversion_is_ignored(tmp_path_factory, units: str | None) -> None:
    """Efficiency 0.98 between nodes without units, in the same unit, or with a unit on one node only: the LP and
    the objective are those of efficiency 1 (the reverse flow cannot gain what the forward flow loses); one W-0591
    names ll with the ignored efficiency, no W-0589."""
    extra = {None: [], "same": [["node", "lw", "unit_of_measure", b64("t", "str")],
                                ["node", "le", "unit_of_measure", b64(" t", "str")]],
             "one": [["node", "le", "unit_of_measure", b64("t", "str")]]}[units]
    lossy, got = _solve(tmp_path_factory, f"k_ign_{units}", 0.98, None, extra)
    unit, _got1 = _solve(tmp_path_factory, f"k_one_{units}", 1.0, None, extra)
    a, b = list(lossy.values())[-1], list(unit.values())[-1]
    assert _lp(a) == _lp(b)
    assert a.solution.obj == pytest.approx(b.solution.obj, abs=1e-9)
    assert [c.names for c in got.calls if c.code == "W-0591"] == [["ll"]]
    assert got.detail("W-0591", "ll") == "efficiency 0.98"
    assert not [c for c in got.calls if c.code == "W-0589"]


#: A ring lw → le → lx → lw of one-variable connections (ll of the fixture, l2, l3).
RING_ENTITIES = [["node", "lx", None], ["connection", "l2", None], ["connection", "l3", None],
                 ["connection__node__node", ["l2", "le", "lx"], None],
                 ["connection__node__node", ["l3", "lx", "lw"], None]]
RING_VALUES = [v for c in ("l2", "l3") for v in (
    ["connection", c, "transfer_method", b64("no_losses_no_variable_cost", "str")],
    ["connection", c, "existing", b64(200.0, "float")])] + [["node", "lx", "node_type", b64("balance", "str")]]


@pytest.mark.parametrize("efficiency", [0.98, 1.0])
def test_a_ring_cannot_circulate_energy(tmp_path_factory, efficiency: float) -> None:
    """A ring of one-variable connections, all with the efficiency: every flow column (forward and reverse) puts
    as much into the balance of one node as it takes from another, so no direction and no loop creates or
    destroys energy; the LP is that of efficiency 1."""
    vals = RING_VALUES + [["connection", c, "efficiency", b64(efficiency, "float")] for c in ("l2", "l3")]
    steps, got = _solve(tmp_path_factory, f"k_ring_{int(efficiency * 100)}", efficiency, None, vals, RING_ENTITIES)
    rows, _obj = _lp(list(steps.values())[-1])
    by_column: dict[str, float] = {}
    for terms in rows.values():
        for col, val in terms.items():
            if col.startswith(("v_flow[", "v_flow_back[")):
                by_column[col] = by_column.get(col, 0.0) + val
    assert {col.split("[")[1].split(",")[0] for col in by_column} == {"ll", "l2", "l3"}
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in by_column.values()), by_column
    if efficiency != 1.0:
        assert sorted(got.names("W-0591")) == ["l2", "l3", "ll"]


def test_results_match_the_objective(tmp_path_factory) -> None:
    """Efficiency 0.5 between lw (MWh) and le (t) and cost 5: the reported connection cost is 0, the flow results
    are the signed flow at le and twice it at lw, the balance results of lw add up to 0, and the varCost frame a
    CSV reload reads has no row of ll (the cost is not charged again)."""
    import pandas as pd

    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    vals = UNITS + [["connection", "ll", "efficiency", b64(0.5, "float")],
                    ["connection", "ll", "other_operational_cost", b64(5.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="k_results", values=vals, base_chain=LL)
    work = tmp_path_factory.mktemp("k_results_w")
    steps = run_chain_from_db(url, scenario_name=scen, work_folder=work, csv_dump=True)
    last = next(reversed(list(steps.values())))
    provider = last.flex_data_provider
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
                  flex_data_provider=provider, raw_output_dir=str(work / "output_raw"))
    out = work / "output_csv" / scen
    cost = pd.read_csv(out / "cost_connection_discounted__d.csv")
    assert cost.loc[cost["connection"] == "ll", "other operational"].tolist() == [0]
    flow = pd.read_csv(out / "connection__dt.csv").set_index("time")["ll"]
    assert flow.to_dict() == pytest.approx({"t0001": 50.0, "t0002": -50.0})
    node = pd.read_csv(out / "node__dt.csv", header=[0, 1], index_col=[0, 1, 2])
    lw = node["lw"].droplevel([0, 1])
    assert lw["To connections"].to_dict() == pytest.approx({t: -2.0 * f for t, f in flow.items()})
    assert lw.sum(axis=1).to_dict() == pytest.approx({"t0001": 0.0, "t0002": 0.0}, abs=1e-9)
    key = "solve_data/pdtProcess__source__sink__dt_varCost"
    assert provider.has(key)
    assert "ll" not in provider.get(key).get_column("process").cast(pl.Utf8).to_list()
