"""Flow-group flow limits count a one-variable connection's reverse flow (item 138).

docs/reference/flow_groups.md: a flow limit applies to the net flow into the group, and a connection counts by its
net flow at the member node.  A two-variable connection (``variable_cost_only`` / ``regular`` / ``exact``) carries
its reverse flow on its own reverse arc, which ``_group_flow`` already netted.  A one-variable connection
(``no_losses_no_variable_cost``, and every DC power-flow line) carries it in ``v_flow_back`` (signed flow =
v_flow − v_flow_back), which ``{max,min}Instant_flow`` and ``{max,min}Cumulative_flow_{period,solve}`` never saw:
the reverse flow passed any flow limit unseen.

Fixture: ``tests.json`` ``lossless_2way`` (from JSON): line ``ll`` lw → le, 200 MW, two 1-h steps, no units and no
costs; inflow lw +100 / le −100 at t0001 and lw −100 / le +100 at t0002, so without a limit ll carries 100 MW
lw → le at t0001 and 100 MW le → lw at t0002 and nothing is unserved (objective 0).  A flow limit that holds the
line back leaves the same energy unserved at lw and spilled at le, priced by the nodes' penalties, so the
objective is linear in the energy the limit holds back.

Hand values (net flow into the member node, MW; "held back" = energy left unserved at the receiving node):

* ``min_instant_flow`` L at (ll, le): F[le] = forward − back ≥ L.  t0001: forward 100 (≥ L).  t0002: back
  = −L, so −40 → 40 MW back and 60 held back; −70 → 70 back, 30 held back.  obj(−40) = 2·obj(−70).
* ``max_cumulative_flow`` C at (ll, lw) (one period, step_duration 1, timestep_weight 1, r = 1):
  F[lw] = −forward + back, Σ_t F ≤ 2·C.  C = −25: Σ_t F[lw] = −50 at the optimum, 50 MWh held back.  How the
  50 MWh split between the steps is degenerate (forward 100 + back 50, or forward 150 at t0001 with 50 MWh left
  unserved at lw then + back 100 cost the same), so only the sum is asserted.  The same per period (a period
  map) and per solve (a constant).
* Before the fix the reverse flow was invisible: F[le] = forward only, F[lw] = −forward only, so each limit
  above was met with nothing held back (objective 0), while the two-variable line held back the energy.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import BASE_FIXTURE_JSON, add_override_alternative, b64, build, utf8

pytestmark = pytest.mark.solver

LL = ["Base", "lossless_2way"]
GROUP = "llg"
TIMES = ("t0001", "t0002")
REL = 1e-9
TOL = 1e-6
_WORK: dict[str, Path] = {}         # step solve name + alt -> the run's work folder (its output_raw/)


def _group(members: list[tuple[str, str]], values: list) -> tuple[list, list]:
    ents = [["flowGroup", GROUP, None]] + [["flowGroup__connection__node", [GROUP, c, n], None]
                                           for c, n in members]
    return ents, [["flowGroup", GROUP, p, v] for p, v in values]


def _method(method: str, conn: str = "ll") -> list:
    vals = [["connection", conn, "transfer_method", b64(method, "str")]]
    if method == "variable_cost_only":
        vals.append(["connection", conn, "other_operational_cost", b64(0.0, "float")])
    return vals


def _dc_values() -> tuple[list, list]:
    """lw and le in a dc_power_flow_with_angles group (reference lw, base 100 MVA); ll reactance 0.1 p.u.: a
    susceptance of 1000 MW/rad, so 100 MW is a 0.1 rad angle difference, far inside the ±π angle bounds."""
    ents = [["group", "dcnet", None], ["group__node", ["dcnet", "lw"], None],
            ["group__node", ["dcnet", "le"], None]]
    vals = [["group", "dcnet", "transfer_method", b64("dc_power_flow_with_angles", "str")],
            ["group", "dcnet", "reference_node", b64("lw", "str")],
            ["group", "dcnet", "base_MVA", b64(100.0, "float")],
            ["connection", "ll", "reactance", b64(0.1, "float")]]
    return ents, vals


def _reversed_line() -> tuple[list, list]:
    """The same 200 MW lossless line authored le → lw (``ll_rev``), ``ll`` inactive."""
    ents = [["connection", "ll_rev", None], ["connection__node__node", ["ll_rev", "le", "lw"], None]]
    vals = [["connection", "ll_rev", "existing", b64(200.0, "float")],
            ["connection", "ll_rev", "efficiency", b64(1.0, "float")]]
    return ents, vals


def _run(tpf, alt: str, values: list, entities: list | None = None, *, inactive: tuple[str, ...] = ()):
    """``lossless_2way`` + override ``alt`` (``inactive`` connections switched off in it), solved under
    FLEXTOOL_AUTOSCALE_STRICT=1; returns the single step."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads(Path(BASE_FIXTURE_JSON).read_text(encoding="utf-8"))
    scenario = add_override_alternative(data, alt, values, LL, entities)
    for conn in inactive:
        data["entity_alternatives"].append(["connection", [conn], alt, False])
    work = tpf.mktemp(f"{alt}_db")
    (work / f"{alt}.json").write_text(json.dumps(data))
    url = json_to_db(work / f"{alt}.json", work / f"{alt}.sqlite")
    migrate_database(url)
    wf = tpf.mktemp(f"{alt}_w")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
        steps = run_chain_from_db(url, scenario, work_folder=wf, keep_solutions=True)
    assert len(steps) == 1, list(steps)
    step = next(iter(steps.values()))
    assert step.solution is not None and step.solution.optimal
    _WORK[alt] = wf
    return step


def _net_into(step, node: str) -> dict[str, float]:
    """{t: MW} net flow into ``node`` over every connection, signed as nodeBalance (v_flow into the node +, out −;
    v_flow_back the other way), from the solution."""
    fd, sol = step.flex_data, step.solution
    us = dict(utf8(fd.p_unitsize.frame).select("p", "value").iter_rows())
    out = dict.fromkeys(TIMES, 0.0)
    for name, sign in (("v_flow", 1.0), ("v_flow_back", -1.0)):
        try:
            frame = sol.value(name)
        except (KeyError, ValueError):
            continue
        if frame is None:
            continue
        for p, s, k, _d, t, x in utf8(frame).select("p", "source", "sink", "d", "t", "value").rows():
            if p not in ("ll", "ll_rev"):
                continue
            if k == node:
                out[t] += sign * x * us[p]
            elif s == node:
                out[t] -= sign * x * us[p]
    return out


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict:
    tpf = tmp_path_factory
    period_map = b64({"index_type": "str", "index_name": "period", "data": [["p2020", -25.0]]}, "map")
    limits = {
        "inst40": ([("ll", "le")], [("min_instant_flow", b64(-40.0, "float")),
                                    ("flow_aggregator", b64("both", "str"))]),
        "inst70": ([("ll", "le")], [("min_instant_flow", b64(-70.0, "float"))]),
        "cum_period": ([("ll", "lw")], [("max_cumulative_flow", period_map)]),
        "cum_solve": ([("ll", "lw")], [("max_cumulative_flow", b64(-25.0, "float"))]),
    }
    out = {"none_1var": _run(tpf, "fgr_none_1var", [])}
    for key, (members, vals) in limits.items():
        ents, gvals = _group(members, vals)
        out[f"{key}_1var"] = _run(tpf, f"fgr_{key}_1var", gvals, ents)
        out[f"{key}_2var"] = _run(tpf, f"fgr_{key}_2var", gvals + _method("variable_cost_only"), ents)
    ents, gvals = _group([("ll_rev", "le")], [("min_instant_flow", b64(-40.0, "float"))])
    rents, rvals = _reversed_line()
    out["inst40_rev"] = _run(tpf, "fgr_inst40_rev", gvals + rvals + _method("no_losses_no_variable_cost", "ll_rev"),
                             ents + rents, inactive=("ll",))
    ents, gvals = _group([("ll", "lw"), ("ll", "le")], [("min_instant_flow", b64(0.0, "float"))])
    out["both_ends"] = _run(tpf, "fgr_both_ends", gvals, ents)
    dents, dvals = _dc_values()
    out["none_dc"] = _run(tpf, "fgr_none_dc", dvals, dents)
    ents, gvals = _group([("ll", "le")], [("min_instant_flow", b64(-40.0, "float"))])
    out["inst40_dc"] = _run(tpf, "fgr_inst40_dc", gvals + dvals, ents + dents)
    return out


def test_fixture_without_limit(cases) -> None:
    """Without a flow limit nothing is held back (objective 0) and ll carries the inflow: +100 MW into le at t0001,
    −100 MW at t0002 (the reverse flow, v_flow_back on the one-variable line)."""
    step = cases["none_1var"]
    assert step.solution.obj == pytest.approx(0.0, abs=TOL)
    net = _net_into(step, "le")
    assert net["t0001"] == pytest.approx(100.0, abs=TOL)
    assert net["t0002"] == pytest.approx(-100.0, abs=TOL)


@pytest.mark.parametrize("case", ["inst40", "inst70", "cum_period", "cum_solve"])
def test_one_variable_line_equals_two_variable_line(cases, case) -> None:
    """(a) The same line as ``no_losses_no_variable_cost`` (one variable, reverse in v_flow_back) and as a zero-cost
    lossless ``variable_cost_only`` line (two arcs) gives the same objective and the same net flows under every flow
    limit family.  The limit holds energy back (objective > 0).  Before: the one-variable line ignored the limit
    (objective 0)."""
    one, two = cases[f"{case}_1var"], cases[f"{case}_2var"]
    assert two.solution.obj > 1.0
    assert one.solution.obj == pytest.approx(two.solution.obj, rel=REL)
    for node in ("lw", "le"):
        n1, n2 = _net_into(one, node), _net_into(two, node)
        if case.startswith("inst"):
            for t in TIMES:
                assert n1[t] == pytest.approx(n2[t], abs=TOL), (case, node, t)
        else:       # the split between the steps is degenerate (module docstring), the period sum is not
            assert sum(n1.values()) == pytest.approx(sum(n2.values()), abs=TOL), (case, node)


def test_instant_flow_hand_values(cases) -> None:
    """``min_instant_flow`` L at (ll, le): the reverse flow at t0002 is exactly −L, so 40 / 70 MW go back and
    60 / 30 MW are held back: obj(−40) = 2·obj(−70); t0001 keeps its full 100 MW forward."""
    for case, back in (("inst40_1var", 40.0), ("inst70_1var", 70.0)):
        net = _net_into(cases[case], "le")
        assert net["t0001"] == pytest.approx(100.0, abs=TOL), case
        assert net["t0002"] == pytest.approx(-back, abs=TOL), case
    assert cases["inst40_1var"].solution.obj == pytest.approx(2.0 * cases["inst70_1var"].solution.obj, rel=1e-7)


@pytest.mark.parametrize("case", ["cum_period_1var", "cum_solve_1var"])
def test_cumulative_flow_hand_values(cases, case) -> None:
    """``max_cumulative_flow`` −25 MW at (ll, lw) over 2 h: Σ_t (−forward + back) = −50 MWh at lw (the reverse flow
    counts + at the line's source node; before the fix the limit read −forward only and held nothing back); 50 MWh
    held back = the inst70 case's 30 × 5/3."""
    net = _net_into(cases[case], "lw")
    assert sum(net.values()) == pytest.approx(-50.0, abs=TOL)
    assert cases[case].solution.obj == pytest.approx(cases["inst70_1var"].solution.obj * 50.0 / 30.0, rel=1e-7)


def test_orientation_invariance(cases) -> None:
    """(b) The line authored le → lw (its reverse flow is now the lw → le direction) under the same limit at le
    gives the same objective and net flow: the sign follows the node, not the arc's authoring direction."""
    fwd, rev = cases["inst40_1var"], cases["inst40_rev"]
    assert rev.solution.obj == pytest.approx(fwd.solution.obj, rel=REL)
    assert _net_into(rev, "le")["t0002"] == pytest.approx(-40.0, abs=TOL)


def test_both_ends_in_group_cancel(cases) -> None:
    """(c) A group with both ends of the line, (ll, lw) and (ll, le): the line's flow leaves one member node and
    enters the other, so F ≡ 0 and ``min_instant_flow`` 0 holds nothing back (objective as without a group); the row
    carries no v_flow_back column with a nonzero coefficient."""
    step = cases["both_ends"]
    assert step.solution.obj == pytest.approx(cases["none_1var"].solution.obj, abs=TOL)
    for row, terms in row_terms(build(step.flex_data), "minInstant_flow").items():
        for col, c in terms.items():
            assert abs(c) <= 1e-12, (row, col, c)


def test_dc_power_flow_line(cases) -> None:
    """(d) ll as a DC power-flow line (two buses, free angles: the angle equation only fixes θ_le): no limit →
    objective 0; ``min_instant_flow`` −40 at (ll, le) → 40 MW back at t0002 (θ_le − θ_lw = 0.04 rad), the same
    objective as the lossless one-variable line."""
    assert cases["none_dc"].solution.obj == pytest.approx(0.0, abs=TOL)
    step = cases["inst40_dc"]
    net = _net_into(step, "le")
    assert net["t0001"] == pytest.approx(100.0, abs=TOL)
    assert net["t0002"] == pytest.approx(-40.0, abs=TOL)
    assert step.solution.obj == pytest.approx(cases["inst40_1var"].solution.obj, rel=1e-7)
    angle = utf8(step.solution.value("v_angle"))
    theta = {(n, t): x for n, _d, t, x in angle.select("n", "d", "t", "value").rows()}
    assert theta[("le", "t0002")] - theta[("lw", "t0002")] == pytest.approx(0.04, abs=1e-6)


def test_flowgroup_output_matches_row(cases, tmp_path_factory) -> None:
    """(e) The flowGroup output ``net_flow`` (``flow_aggregator: both``) reads the line by its net flow, as the row's
    F: in inst40 it is ≥ the limit −40 at every step, +100 at t0001 and −40 at t0002 (the reverse flow counted)."""
    net = _flowgroup_net_flow(cases["inst40_1var"], _WORK["fgr_inst40_1var"], tmp_path_factory)
    assert set(net) == set(TIMES)
    assert all(v >= -40.0 - TOL for v in net.values()), net
    assert net["t0001"] == pytest.approx(100.0, abs=TOL)
    assert net["t0002"] == pytest.approx(-40.0, abs=TOL)


def _flowgroup_net_flow(step, work: Path, tpf) -> dict[str, float]:
    """``flowGroup_gd_t.net_flow`` of GROUP per step, from the output pipeline (published as
    ``group_flow__dt.csv``, default_plots.yaml)."""
    from flextool.process_outputs.write_outputs import write_outputs

    out = tpf.mktemp("fgr_out")
    write_outputs(scenario_name="fgr", output_location=str(out), subdir="fgr", write_methods=["csv"],
                  fallback_output_location=str(out), raw_output_dir=str(work / "output_raw"),
                  solution=step.solution, solve_name=step.solve_name,
                  solve_steps=[(step.solve_name, step.flex_data, step.effective_solution)],
                  flex_data_provider=step.flex_data_provider)
    files = list(out.rglob("group_flow__dt.csv"))
    assert files, sorted(str(p.relative_to(out)) for p in out.rglob("*"))
    frame = pl.read_csv(files[0])
    return {t: v for g, t, v in frame.select("group", "time", "net_flow").rows() if g == GROUP}
