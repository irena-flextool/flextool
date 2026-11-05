"""Coarse processes: a unit or connection whose every balance node is coarse takes the coarse step of its nodes.

Owner decision 2026-11-03 ("Coarse new_stepduration groups"): a process between coarse nodes has one flow per
coarse step (a rate constant over the step); a process with a fine balance node keeps the solve's step, and so
does a process with a feature whose rows are per fine step (a delay,
DC power flow, an exact connection, a user-constraint flow term, a flow-group instant limit, a capacity-margin /
inertia / non-synchronous group), or, in a stochastic solve, a forecast tie that covers only part of one of its
coarse steps.  Online processes: test_coarse_online_reserves.py.

Fixture: ``tests/fixtures/lh2_three_region.json`` (h2_* / lh2_* in the 24-h ``daily_group``, elec_* hourly; the
liquefiers h2 -> lh2 and the pipes lh2 <-> lh2 are between daily nodes, the electrolysers elec -> h2 are not),
with override alternatives built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.params import _coarse
from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, last_stochastic, make_db, run, utf8, without_realization

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"
LH2 = FIX / "lh2_three_region.json"
LH2_COARSE = {"liquefier_A", "liquefier_B", "liquefier_C", "pipe_AB", "pipe_BC"}


def _chain(fixture: Path, scenario: str) -> list[str]:
    data = json.loads(fixture.read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


def _f(x) -> list:
    return b64(float(x), "float")


def _s(x) -> list:
    return b64(x, "str")


def _lh2(tmp_path_factory, alt, values=(), entities=None) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=list(values), base_chain=_chain(LH2, "lh2_three_region"),
                        entities=entities, fixture_json=LH2)
    return run(tmp_path_factory, url, scen)


def _coarse_procs(fd) -> set[str]:
    return set() if fd.processCoarse is None else set(utf8(fd.processCoarse)["p"].to_list())


# ---------------------------------------------------------------------------
# The rule (pure function)


def _frames(process_nodes, node_step, balance=None, excluded=None, steps_of=None):
    """``coarse_process_frames`` on one 48-h period of 1-h steps; ``steps_of`` node -> its group step (h)."""
    rows = []
    for n, step in (steps_of or node_step).items():
        for h in range(48):
            first = int(h // step * step)
            rows.append((n, "d1", f"t{first:02d}", f"t{h:02d}"))
    ncs = pl.DataFrame(rows, schema={"n": pl.Utf8, "d": pl.Utf8, "b_first": pl.Utf8, "t": pl.Utf8}, orient="row")
    sd = pl.DataFrame({"d": ["d1"] * 48, "t": [f"t{h:02d}" for h in range(48)], "value": [1.0] * 48})
    nodes = set().union(*process_nodes.values())
    return _coarse.coarse_process_frames(
        process_nodes=process_nodes, balance_nodes=nodes if balance is None else balance, node_step=node_step,
        node_coarse_step=ncs, step_duration=sd, excluded=excluded or {}, solve_step=1.0)


def test_rule_all_balance_nodes_coarse() -> None:
    """A process is coarse only when every balance node is coarse; a node without a balance is neutral; a
    process without a balance node keeps the solve's step."""
    fr = _frames({"pipe": {"a", "b"}, "elz": {"a", "fine"}, "fuel_unit": {"a", "fuel"}, "free": {"fuel"}},
                 {"a": 24.0, "b": 24.0}, balance={"a", "b", "fine"})
    assert utf8(fr.processCoarse)["p"].to_list() == ["fuel_unit", "pipe"]
    assert fr.step_hours == {"fuel_unit": 24.0, "pipe": 24.0}
    pipe = fr.process_coarse_step.filter(pl.col("p") == "pipe")
    assert pipe.height == 48 and pipe["b_first"].unique().sort().to_list() == ["t00", "t24"]
    sums = pipe.group_by("b_first").agg(pl.col("w").sum())["w"].to_list()
    assert sums == pytest.approx([1.0, 1.0])
    assert pipe["w"].to_list() == pytest.approx([1.0 / 24.0] * 48)


def test_rule_shortest_dividing_step() -> None:
    """24 h + 12 h nodes: the process takes the 12-h steps; 24 h + 36 h: the steps do not divide, it stays fine
    with the reason."""
    fr = _frames({"p12": {"a", "b"}, "p36": {"a", "c"}}, {"a": 24.0, "b": 12.0, "c": 36.0})
    assert utf8(fr.processCoarse)["p"].to_list() == ["p12"]
    assert fr.step_hours == {"p12": 12.0}
    assert fr.process_coarse_step["b_first"].unique().sort().to_list() == ["t00", "t12", "t24", "t36"]
    assert "do not divide" in fr.kept_fine["p36"]


def test_rule_excluded_stays_fine() -> None:
    """A process with a per-step feature keeps the solve's step even when all its nodes are coarse; the reason
    is kept for the info line."""
    fr = _frames({"liq": {"a", "b"}}, {"a": 24.0, "b": 24.0}, excluded={"liq": "it has online variables"})
    assert fr.processCoarse.height == 0
    assert fr.kept_fine == {"liq": "it has online variables"}


# ---------------------------------------------------------------------------
# lh2


def test_lh2_coarse_processes(tmp_path_factory) -> None:
    """lh2: the liquefiers (h2 -> lh2) and the pipes (lh2 <-> lh2) are coarse at 24 h; the electrolysers
    (elec hourly -> h2 daily) and every hourly process are not.  Each coarse step's weights sum to 1."""
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_rule")).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    steps = utf8(fd.process_coarse_step.drop("w")).with_columns(w=fd.process_coarse_step["w"])
    assert set(steps["p"].unique().to_list()) == LH2_COARSE
    liq = steps.filter(pl.col("p") == "liquefier_A")
    assert liq.height == 168 and liq["b_first"].n_unique() == 7
    assert liq.group_by("d", "b_first").agg(pl.col("w").sum())["w"].to_list() == pytest.approx([1.0] * 7)
    node = utf8(fd.node_coarse_step).filter(pl.col("n") == "lh2_A").select("d", "b_first", "t")
    assert liq.select("d", "b_first", "t").equals(node)


def test_lh2_online_unit_with_ramps_is_coarse(tmp_path_factory, caplog, monkeypatch) -> None:
    """liquefier_A with online variables and a ramp limit is coarse (its units online once per day,
    test_coarse_online_reserves): its start-up / shut-down capacity reduction is one averaged row per day
    (maxFlow_online_linear_coarse, maxFlow_online_shutdown_linear_coarse for the six days with a successor), its ramps
    one row per day with a predecessor (ρ̄ = 2e-4·60·24 = 0.288 < 1: days 2–7), no hourly row; the info line names it
    with the other coarse processes and the behaviour notice names the reduction."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    uc = [["unit", "liquefier_A", "startup_method", _s("linear")], ["unit", "liquefier_A", "min_load", _f(0.3)],
          ["unit", "liquefier_A", "virtual_unitsize", _f(50.0)]]
    monkeypatch.setattr(_coarse, "_WARNED", set())
    caplog.set_level(logging.INFO, logger=_coarse.__name__)
    rel = ["liquefier_A", "lh2_A"]
    ramp = [["unit__outputNode", rel, "ramp_method", _s("ramp_limit")],
            ["unit__outputNode", rel, "ramp_speed_up", _f(2e-4)], ["unit__outputNode", rel, "ramp_speed_down", _f(2e-4)]]
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_uc_ramp", uc + ramp)).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    msgs = [r.message for r in caplog.records]
    assert not any("'liquefier_A' connects only nodes of coarse" in m for m in msgs)
    assert any("use the 24 h time step of their nodes" in m and "'liquefier_A'" in m for m in msgs)
    assert "its start-up / shut-down capacity reduction uses the coarse step" in coarse_process_reasons(
        fd, fd.process_coarse_step)["liquefier_A"]
    pb = build(fd)
    days = [f"t{24 * k + 1:04d}" for k in range(7)]
    arc = "liquefier_A,h2_A,lh2_A,"
    assert sorted(r.rsplit(",", 1)[1][:-1] for r in row_terms(pb, "maxFlow_online_linear_coarse", arc)) == days
    assert sorted(r.rsplit(",", 1)[1][:-1]
                  for r in row_terms(pb, "maxFlow_online_shutdown_linear_coarse", arc)) == days[:-1]
    for fam in ("ramp_sink_up_constraint", "ramp_sink_down_constraint"):
        assert sorted(r.rsplit(",", 1)[1][:-1] for r in row_terms(pb, fam, arc)) == days[1:], fam
    for fam in ("maxFlow_online_linear", "maxFlow_online_shutdown_linear"):
        assert not row_terms(pb, fam, "liquefier_A,"), fam


def test_lh2_user_constraint_flow_stays_fine(tmp_path_factory) -> None:
    """A user constraint with a flow coefficient on pipe_AB keeps pipe_AB at the solve's step: the constraint's
    rows are per fine step, and with a time-varying constant a constant flow could not meet them."""
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_uc_flow", [
        ["constraint", "cap_ab", "sense", _s("less_than")], ["constraint", "cap_ab", "constant", _f(1000.0)],
        ["connection__node", ["pipe_AB", "lh2_B"], "constraint_flow_coeff",
         b64({"index_type": "str", "index_name": "constraint", "data": [["cap_ab", 1.0]]}, "map")]],
        [["constraint", "cap_ab", None], ["connection__node", ["pipe_AB", "lh2_B"], None]])).flex_data
    assert _coarse_procs(fd) == LH2_COARSE - {"pipe_AB"}


# ---------------------------------------------------------------------------
# Output expansion (flextool.engine.output._coarse_flows)


class _Steps:
    """A FlexData stand-in carrying only ``process_coarse_step``: pipe has 3-h coarse steps on 6 hourly steps."""

    process_coarse_step = pl.DataFrame({"p": ["pipe"] * 6, "d": ["d1"] * 6,
                                        "b_first": ["t1", "t1", "t1", "t4", "t4", "t4"],
                                        "t": [f"t{h}" for h in range(1, 7)], "w": [1.0 / 3.0] * 6})


def test_expand_coarse_copies_to_missing_steps() -> None:
    """A coarse flow (one LP row per coarse step) is copied to the other fine steps of its step, right after its
    own row; a fine process is unchanged; a step that has its own LP row keeps it (an LP whose flows are fine
    comes back unchanged)."""
    from flextool.engine.output._coarse_flows import expand_coarse

    lp = pl.DataFrame({"p": ["elz", "elz", "pipe", "pipe"], "source": ["a", "a", "x", "x"],
                       "sink": ["b", "b", "y", "y"], "d": ["d1"] * 4, "t": ["t1", "t2", "t1", "t4"],
                       "value": [5.0, 6.0, 2.0, 3.0]})
    out = expand_coarse(lp, _Steps())
    assert out.columns == lp.columns and out.schema == lp.schema
    assert out.filter(pl.col("p") == "elz").equals(lp.filter(pl.col("p") == "elz"))
    pipe = out.filter(pl.col("p") == "pipe")
    assert pipe["t"].to_list() == [f"t{h}" for h in range(1, 7)]
    assert pipe["value"].to_list() == [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    fine_lp = pl.DataFrame({"p": ["pipe"] * 6, "source": ["x"] * 6, "sink": ["y"] * 6, "d": ["d1"] * 6,
                            "t": [f"t{h}" for h in range(1, 7)], "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert expand_coarse(fine_lp, _Steps()).equals(fine_lp)
    assert expand_coarse(lp, None).equals(lp)


def test_expand_coarse_first_step_mode() -> None:
    """An event variable (start-ups): the value stays at the coarse step's first fine step and the other fine steps
    of the step get 0, so the sum over the step is the LP's value; a fine process and the dtypes are unchanged."""
    from flextool.engine.output._coarse_flows import COARSE_VARS, expand_coarse

    lp = pl.DataFrame({"p": ["elz", "elz", "pipe", "pipe"], "d": ["d1"] * 4, "t": ["t1", "t2", "t1", "t4"],
                       "value": [5.0, 6.0, 2.0, 3.0]})
    out = expand_coarse(lp, _Steps(), mode="first")
    assert out.schema == lp.schema
    assert out.filter(pl.col("p") == "elz").equals(lp.filter(pl.col("p") == "elz"))
    pipe = out.filter(pl.col("p") == "pipe")
    assert pipe["t"].to_list() == [f"t{h}" for h in range(1, 7)]
    assert pipe["value"].to_list() == [2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    assert {n for n, m in COARSE_VARS.items() if m == "first"} == {
        "v_startup_linear", "v_startup_integer", "v_shutdown_linear", "v_shutdown_integer"}
    with pytest.raises(ValueError):
        expand_coarse(lp, _Steps(), mode="spread")


def test_expand_value_keys_copies_to_missing_steps() -> None:
    """The HiGHS-name reader's keys ``(d, t, p, source, sink)``: the same rule as expand_coarse."""
    from flextool.engine.output._coarse_flows import coarse_steps, expand_value_keys

    keys = {("d1", "t1", "pipe", "x", "y"): 2.0, ("d1", "t4", "pipe", "x", "y"): 3.0,
            ("d1", "t2", "elz", "a", "b"): 6.0, ("d1", "t5", "pipe", "x", "y"): 9.0}
    expand_value_keys(keys, coarse_steps(_Steps()))
    pipe = {k[1]: v for k, v in keys.items() if k[2] == "pipe"}
    assert pipe == {"t1": 2.0, "t2": 2.0, "t3": 2.0, "t4": 3.0, "t5": 9.0, "t6": 3.0}
    assert keys[("d1", "t2", "elz", "a", "b")] == 6.0
    events = {("d1", "t1", "pipe"): 1.0, ("d1", "t2", "elz"): 6.0}
    expand_value_keys(events, coarse_steps(_Steps()), mode="first")
    assert {k[1]: v for k, v in events.items() if k[2] == "pipe"} == {
        "t1": 1.0, "t2": 0.0, "t3": 0.0}


def _gate_patterns():
    from flextool.engine.output._coarse_flows import COARSE_VARS

    names = "|".join(sorted(COARSE_VARS, key=len, reverse=True))
    literal = re.compile(rf"""\.value\(\s*["']({names})["']|_vars\[\s*["']({names})["']\s*\]"""
                         rf"""|col_id_of_var\(\s*["']({names})|fix_cols\(\s*["']({names})""")
    # A read whose variable name is not a literal (a loop over names): flagged when a coarse variable's name is
    # quoted within the eight lines above it.
    dynamic = re.compile(r"""\.value\(\s*[A-Za-z_]\w*\s*\)|_vars\[\s*[A-Za-z_]\w*\s*\]"""
                         r"""|col_id_of_var\(\s*[A-Za-z_]\w*|fix_cols\(\s*[A-Za-z_]\w*""")
    quoted = re.compile(rf"""["']({names})["']""")
    return literal, dynamic, quoted


# Files that may read the LP's columns as they are: the expansion helper itself, and Benders, which reads only the
# half-flows of cross-region arcs (a cross-region process is never coarse).
_RAW_READERS = {"flextool/engine/output/_coarse_flows.py", "flextool/engine/solve/_benders.py"}


def test_readers_expand_coarse_variables() -> None:
    """Every reader of the values of a variable of _coarse_flows.COARSE_VARS (flows, units online, start-ups,
    shut-downs) outside the expansion helper goes through it: a raw read would see a coarse process only at its
    coarse step's first fine step (outputs lose the other steps, the CO2 carry under-counts, the roll carry of
    units online reads no value at an interior last realized step)."""
    literal, dynamic, quoted = _gate_patterns()
    hits = []
    for path in sorted((FLEXTOOL_ROOT / "flextool").rglob("*.py")):
        rel = path.relative_to(FLEXTOOL_ROOT).as_posix()
        if rel in _RAW_READERS:
            continue
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if literal.search(line) or (dynamic.search(line)
                                        and any(quoted.search(x) for x in lines[max(0, i - 9):i])):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, "raw coarse-variable reads (use flextool.engine.output._coarse_flows):\n" + "\n".join(hits)


# ---------------------------------------------------------------------------
# The model: one flow per coarse step


def _fine_flows(monkeypatch) -> None:
    """Make every process keep the solve's step (the model before coarse processes)."""
    monkeypatch.setattr(_coarse, "coarse_process_frames", lambda **kw: None)


def _total_objective(steps: dict) -> float:
    return sum(float(s.obj) for s in without_realization(steps).values())


def _flow_cols(pb, prefix: str) -> list[str]:
    return [c for c in pb.canonicalise().col_names if c.startswith(prefix)]


LH2_RP = FIX / "lh2_three_region_rp_invest.json"
_ROLLING = [["solve", "lh2_week", "solve_mode", _s("rolling_window")],
            ["solve", "lh2_week", "rolling_solve_horizon", _f(72.0)],
            ["solve", "lh2_week", "rolling_solve_jump", _f(48.0)]]


def _case(tmp_path_factory, case: str, alt: str) -> dict:
    if case == "rp":
        url, scen = make_db(tmp_path_factory, alt=alt, values=[], fixture_json=LH2_RP,
                            base_chain=_chain(LH2_RP, "lh2_three_region_rp_invest"))
        return run(tmp_path_factory, url, scen)
    return _lh2(tmp_path_factory, alt, _ROLLING if case == "rolling" else [])


@pytest.mark.parametrize("case", ["single", "rolling", "rp"])
def test_lh2_same_optimum_as_fine_flows(tmp_path_factory, monkeypatch, case) -> None:
    """lh2 (every per-step parameter of the liquefiers and pipes is constant within a day): the model with one
    flow per day has the objective of the model with hourly flows, in a single solve, a rolling solve (72 h /
    48 h) and a representative-period invest solve, with fewer columns.  Only the objective is compared: with
    storage at every lh2 node and pipes in both directions the daily energy of each arc is not unique."""
    coarse = _case(tmp_path_factory, case, f"cf_same_{case}")
    _fine_flows(monkeypatch)
    fine = _case(tmp_path_factory, case, f"cf_same_{case}_fine")
    assert len(coarse) == len(fine)
    assert _total_objective(coarse) == pytest.approx(_total_objective(fine), rel=1e-9)
    c_step, f_step = last_stochastic(coarse), last_stochastic(fine)
    assert _coarse_procs(c_step.flex_data) and not _coarse_procs(f_step.flex_data)
    assert len(_flow_cols(build(c_step.flex_data), "v_flow[")) < len(_flow_cols(build(f_step.flex_data), "v_flow["))


def test_lh2_coarse_columns_and_rows(tmp_path_factory) -> None:
    """One week of 24-h coarse steps: 7 v_flow columns per coarse arc (none on the other 161 hours), 7
    maxFlow_coarse rows and no hourly maxFlow row per arc; the electrolyser (elec hourly -> h2 daily) keeps 168
    columns; the daily balance of lh2_A holds liquefier_A's column of the day with 24 × unitsize."""
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_cols")).flex_data
    pb = build(fd)
    cols = pb.canonicalise().col_names
    for arc in ("liquefier_A,h2_A,lh2_A", "pipe_AB,lh2_A,lh2_B", "pipe_AB,lh2_B,lh2_A"):
        mine = [c for c in cols if c.startswith(f"v_flow[{arc},")]
        assert [c.rstrip("]").split(",")[-1] for c in mine] == [f"t{24 * k + 1:04d}" for k in range(7)]
        assert len(row_terms(pb, "maxFlow_coarse", f"{arc},")) == 7
        assert not row_terms(pb, "maxFlow", f"{arc},")
    assert len([c for c in cols if c.startswith("v_flow[electrolyser_A,elec_A,h2_A,")]) == 168
    us = float(utf8(fd.p_unitsize.frame).filter(pl.col("p") == "liquefier_A")["value"][0])
    day = row_terms(pb, "nodeBalanceCoarse_eq", "lh2_A,y2030,t0025")["nodeBalanceCoarse_eq[lh2_A,y2030,t0025]"]
    assert day["v_flow[liquefier_A,h2_A,lh2_A,y2030,t0025]"] == pytest.approx(24.0 * us)
    assert not any(c.startswith("v_flow[liquefier_A,") and not c.endswith("t0025]") for c in day)


def test_registry_has_explicit_coarse_families() -> None:
    """Every <family>_coarse resolves to its own registry entry (a prefix match would silently take the fine
    family's)."""
    from flextool.engine.autoscale._layer2_types import CONSTRAINT_FAMILIES, lookup_cstr

    for name in ("maxFlow", "maxFlow_negCap", "maxFlow_back", "maxOutputSum", "maxOutputSum_negCap",
                 "conversion_indirect", "minInputShare", "profile_flow_upper_limit", "profile_flow_lower_limit",
                 "profile_flow_fixed"):
        assert lookup_cstr(f"{name}_coarse") is CONSTRAINT_FAMILIES[f"{name}_coarse"]
        assert CONSTRAINT_FAMILIES[f"{name}_coarse"] == CONSTRAINT_FAMILIES[name]


def _tmap_lh2(fn) -> list:
    """A ``time`` map over lh2's 168 hourly steps, value ``fn(h)`` for hour h = 0 … 167."""
    return b64({"index_type": "str", "index_name": "time", "data": [[f"t{h + 1:04d}", fn(h)] for h in range(168)]},
               "map")


# Every arc-local family on coarse processes: liquefier_A with a second output (indirect: conversion_indirect,
# maxOutputSum) and an hourly availability, efficiency and output profile; liquefier_B with a second input and
# an input_share_min (minInputShare); pipe_BC without losses (one variable: v_flow_back, maxFlow_back).
_FEATURES_VALUES = [
    ["unit__outputNode", ["liquefier_A", "lh2_B"], "capacity_max_coeff", _f(0.5)],
    ["unit__outputNode", ["liquefier_A", "lh2_A"], "capacity_max_coeff", _f(0.5)],
    ["unit", "liquefier_A", "availability", _tmap_lh2(lambda h: 0.5 + 0.5 * ((h // 3) % 2))],
    ["unit", "liquefier_A", "efficiency", _tmap_lh2(lambda h: 0.8 + 0.1 * ((h // 6) % 2))],
    ["profile", "liq_prof", "profile", _tmap_lh2(lambda h: 0.2 + 0.05 * (h % 4))],
    ["unit__node__profile", ["liquefier_A", "lh2_A", "liq_prof"], "profile_method", _s("upper_limit")],
    ["unit__inputNode", ["liquefier_B", "h2_B"], "input_share_min", _f(0.3)],
    ["connection", "pipe_BC", "transfer_method", _s("no_losses_no_variable_cost")],
]
_FEATURES_ENTITIES = [["unit__outputNode", ["liquefier_A", "lh2_B"], None],
                      ["unit__inputNode", ["liquefier_B", "h2_C"], None],
                      ["profile", "liq_prof", None],
                      ["unit__node__profile", ["liquefier_A", "lh2_A", "liq_prof"], None]]
_TWINS = ("maxFlow", "maxFlow_back", "maxOutputSum", "conversion_indirect", "minInputShare",
          "profile_flow_upper_limit")


def _keys(name: str) -> list[str]:
    return name[name.index("[") + 1:-1].split(",")


def test_averaged_rows_are_step_means_of_fine_rows(tmp_path_factory, monkeypatch) -> None:
    """EQ. <family>_coarse: each coarse row is Σ_{t∈b} w_t · (fine row at t), w_t = Δt/Δ_b — checked against the
    rows the same data gives with hourly flows: every coefficient on the coarse step's column (the fine
    coefficients of its 24 columns, weighted), every other coefficient and the bound.  Covers maxFlow,
    maxFlow_back, maxOutputSum, conversion_indirect (an hourly efficiency), minInputShare and
    profile_flow_upper_limit (an hourly profile and availability)."""
    coarse = last_stochastic(_lh2(tmp_path_factory, "cf_avg", _FEATURES_VALUES, _FEATURES_ENTITIES)).flex_data
    _fine_flows(monkeypatch)
    fine = last_stochastic(_lh2(tmp_path_factory, "cf_avg_fine", _FEATURES_VALUES, _FEATURES_ENTITIES)).flex_data
    steps = utf8(coarse.process_coarse_step.drop("w")).with_columns(w=coarse.process_coarse_step["w"])
    step_of = {(p, d, t): b for p, d, b, t, _w in steps.iter_rows()}
    members: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for p, d, b, t, w in steps.iter_rows():
        members.setdefault((p, d, b), []).append((t, w))
    pc, pf = build(coarse), build(fine)

    def to_coarse(col: str) -> str:
        if not col.startswith(("v_flow[", "v_flow_back[")):
            return col
        k = _keys(col)
        b = step_of.get((k[0], k[-2], k[-1]))
        return col if b is None else col[:col.index("[") + 1] + ",".join([*k[:-1], b]) + "]"

    fm = pf.canonicalise()
    col_ub = dict(zip(fm.col_names, fm.col_ub))
    seen = set()
    for fam in _TWINS:
        c_rows, c_bounds = row_terms(pc, f"{fam}_coarse"), row_bounds(pc, f"{fam}_coarse")
        f_rows, f_bounds = row_terms(pf, fam), row_bounds(pf, fam)
        for name, terms in c_rows.items():
            k = _keys(name)
            expect: dict[str, float] = {}
            lo = hi = 0.0
            for t, w in members[(k[0], k[-2], k[-1])]:
                fname = f"{fam}[{','.join([*k[:-1], t])}]"
                if fname not in f_rows and fam == "maxFlow":
                    # An input arc of an indirect unit without investment: the hourly model bounds the variable
                    # instead of writing a row (v_flow ≤ upper); the coarse model caps the coarse column by a row.
                    col = f"v_flow[{','.join([*k[:-1], t])}]"
                    f_rows[fname], f_bounds[fname] = {col: 1.0}, (-float("inf"), float(col_ub[col]))
                for col, v in f_rows[fname].items():
                    expect[to_coarse(col)] = expect.get(to_coarse(col), 0.0) + w * v
                lo += w * f_bounds[fname][0]
                hi += w * f_bounds[fname][1]
            expect = {c: v for c, v in expect.items() if abs(v) > 1e-12}
            assert terms.keys() == expect.keys(), (name, terms, expect)
            for col, v in expect.items():
                assert terms[col] == pytest.approx(v, rel=1e-9, abs=1e-12), (name, col)
            assert c_bounds[name] == pytest.approx((lo, hi), rel=1e-9, abs=1e-9), name
            seen.add(fam)
        # The fine family keeps no row of a coarse process.
        assert not [r for r in row_terms(pc, fam) if _keys(r)[0] in LH2_COARSE]
    assert seen == set(_TWINS)


def test_one_varying_limit_same_optimum(tmp_path_factory, monkeypatch) -> None:
    """pipe_AB with an hourly availability (one time-varying limit per arc): the step-averaged limit lets the
    pipe move the energy the hourly limits allow over the day, so the optimum is the same."""
    vals = [["connection", "pipe_AB", "availability", _tmap_lh2(lambda h: 0.05 + 0.3 * (h % 2))]]
    coarse = _lh2(tmp_path_factory, "cf_avail", vals)
    _fine_flows(monkeypatch)
    fine = _lh2(tmp_path_factory, "cf_avail_fine", vals)
    assert "pipe_AB" in _coarse_procs(last_stochastic(coarse).flex_data)
    assert _total_objective(coarse) == pytest.approx(_total_objective(fine), rel=1e-9)


# ---------------------------------------------------------------------------
# Hand-built model (tests.json, 2 days of 1-h steps): fuel -> src -> ha =hpipe=> hb <- src2 (fixed profile)
#
# ha, hb: balance nodes in a 24-h group; src (coal_market -> ha, efficiency η) and src2 (coal_market -> hb,
# 100 MW, fixed profile f) take fuel from the commodity node coal_market (no balance: neutral), so src, src2 and
# hpipe (ha -> hb, efficiency 0.9, regular) are coarse.  hb's demand D: day 1 10 MW for 12 h then 30 MW
# (480 MWh), day 2 15 MW (360 MWh); f: 0.01 for 12 h then 0.03 in each day (mean 0.02).
# Hand derivation (one flow per day):
#   src2 = mean(f) · 100 MW = 2 MW every hour (profile_flow_fixed_coarse), 48 MWh a day;
#   hpipe delivers the rest (a connection's flow is what reaches its sink; its source gives flow / efficiency):
#   F_k · 24 h = D_k − 48  ->  F_1 = 432 / 24 = 18 MW, F_2 = 312 / 24 = 13 MW;
#   src = F_k / 0.9 (ha's daily balance) = 20 MW and 14.4444 MW; the reverse pipe flow is 0.
# A pipe cost c_t (1 for 12 h, then 5) costs the coarse flow its daily mean (3); an efficiency η_t (0.5 for 12 h,
# then 1) costs it the mean fuel per MWh, mean(1/η) = 1.5, i.e. η = 2/3.

_HAND_CHAIN = ["init", "west", "coal"]
_HOURS = [f"t{h:04d}" for h in range(1, 49)]


def _hmap(fn) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, fn(h)] for h, t in enumerate(_HOURS)]}, "map")


def _half(first: float, second: float) -> list:
    return _hmap(lambda h: first if h % 24 < 12 else second)


_HAND_ENTITIES = [
    ["node", "ha", None], ["node", "hb", None], ["unit", "src", None], ["unit", "src2", None],
    ["connection", "hpipe", None], ["group", "daily", None], ["profile", "pf", None],
    ["unit__inputNode", ["src", "coal_market"], None], ["unit__outputNode", ["src", "ha"], None],
    ["unit__inputNode", ["src2", "coal_market"], None], ["unit__outputNode", ["src2", "hb"], None],
    ["connection__node__node", ["hpipe", "ha", "hb"], None],
    ["group__node", ["daily", "ha"], None], ["group__node", ["daily", "hb"], None],
    ["unit__node__profile", ["src2", "hb", "pf"], None],
]


def _hand_values(cost, efficiency) -> list:
    out = [["group", "daily", "new_stepduration", _f(24.0)]]
    for n in ("ha", "hb"):
        out += [["node", n, "node_type", _s("balance")], ["node", n, "penalty_up", _f(10000.0)],
                ["node", n, "penalty_down", _f(10000.0)]]
    out += [["node", "hb", "inflow", _hmap(lambda h: -(10.0 if h < 12 else 30.0) if h < 24 else -15.0)],
            ["unit", "src", "conversion_method", _s("constant_efficiency")], ["unit", "src", "efficiency", efficiency],
            ["unit", "src", "existing", _f(1000.0)],
            ["unit", "src2", "conversion_method", _s("constant_efficiency")], ["unit", "src2", "efficiency", _f(1.0)],
            ["unit", "src2", "existing", _f(100.0)],
            ["profile", "pf", "profile", _half(0.01, 0.03)],
            ["unit__node__profile", ["src2", "hb", "pf"], "profile_method", _s("fixed")],
            ["connection", "hpipe", "efficiency", _f(0.9)], ["connection", "hpipe", "existing", _f(1000.0)],
            ["connection", "hpipe", "transfer_method", _s("regular")],
            ["connection", "hpipe", "other_operational_cost", cost]]
    return out


def _hand(tmp_path_factory, alt: str, cost=None, efficiency=None):
    vals = _hand_values(_f(3.0) if cost is None else cost, _f(2.0 / 3.0) if efficiency is None else efficiency)
    url, scen = make_db(tmp_path_factory, alt=alt, values=vals, base_chain=_HAND_CHAIN, entities=_HAND_ENTITIES)
    return last_stochastic(run(tmp_path_factory, url, scen))


def _mw(step, p: str, source: str, sink: str) -> list[float]:
    from flextool.engine.output._coarse_flows import var_values

    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == p)["value"][0])
    flows = var_values(step.solution, step.flex_data, "v_flow")
    f = utf8(flows.drop("value")).with_columns(value=flows["value"])
    rows = f.filter((pl.col("p") == p) & (pl.col("source") == source) & (pl.col("sink") == sink)).sort("t")
    assert rows["t"].to_list() == _HOURS
    return [v * us for v in rows["value"].to_list()]


def test_hand_built_coarse_flows(tmp_path_factory) -> None:
    """The flows of the hand-built model (derivation above): src2 2 MW, hpipe 18 / 13 MW, src 20 / 14.4444 MW,
    constant through each day; no reverse flow."""
    step = _hand(tmp_path_factory, "cf_hand")
    assert _coarse_procs(step.flex_data) == {"src", "src2", "hpipe"}
    assert _mw(step, "src2", "coal_market", "hb") == pytest.approx([2.0] * 48, rel=1e-9)
    assert _mw(step, "hpipe", "ha", "hb") == pytest.approx([18.0] * 24 + [13.0] * 24, rel=1e-9)
    assert _mw(step, "src", "coal_market", "ha") == pytest.approx([20.0] * 24 + [13.0 / 0.9] * 24, rel=1e-9)
    assert _mw(step, "hpipe", "hb", "ha") == pytest.approx([0.0] * 48, abs=1e-9)


def test_hand_built_cost_and_efficiency_are_step_means(tmp_path_factory, monkeypatch) -> None:
    """A pipe cost that varies within the day costs the coarse flow its daily mean, and an efficiency that varies
    costs the mean fuel per MWh: the objective equals that of the constant mean values.  With hourly flows the
    model picks the cheap and efficient hours, so it is cheaper (the results change, and the notice says why)."""
    varying = _hand(tmp_path_factory, "cf_hand_var", cost=_half(1.0, 5.0), efficiency=_half(0.5, 1.0))
    mean = _hand(tmp_path_factory, "cf_hand_mean")
    assert float(varying.obj) == pytest.approx(float(mean.obj), rel=1e-9)
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    why = coarse_process_reasons(varying.flex_data, varying.flex_data.process_coarse_step)
    assert any("variable cost varies" in r for r in why["hpipe"])
    assert any("efficiency varies" in r for r in why["src"])
    assert "src2" not in why
    assert coarse_process_reasons(mean.flex_data, mean.flex_data.process_coarse_step) == {}
    _fine_flows(monkeypatch)
    fine = _hand(tmp_path_factory, "cf_hand_var_fine", cost=_half(1.0, 5.0), efficiency=_half(0.5, 1.0))
    assert float(fine.obj) < float(varying.obj) * (1 - 1e-6)


# ---------------------------------------------------------------------------
# Ramps between coarse steps

_LIQ = "liquefier_A,h2_A,lh2_A"


def _liq_ramp(speed: float) -> list:
    rel = ["liquefier_A", "lh2_A"]
    return [["unit__outputNode", rel, "ramp_method", _s("ramp_limit")],
            ["unit__outputNode", rel, "ramp_speed_up", _f(speed)],
            ["unit__outputNode", rel, "ramp_speed_down", _f(speed)]]


def _count(fd, p: str) -> float:
    return float(utf8(fd.p_process_existing_count.frame).filter(pl.col("p") == p)["value"][0])


def test_coarse_ramp_between_coarse_steps(tmp_path_factory) -> None:
    """liquefier_A (24-h steps) with ramp speed 2e-4 per minute: one ramp row per day after the first, from the
    previous day's column, with room ρ̄ = 2e-4 · 60 · (24 + 24) / 2 = 0.288 of the capacity; no row inside a day.
    With a speed of 1e-3 the room is 1.44 ≥ 1: no row can bind within the period and none is built."""
    fd = last_stochastic(_lh2(tmp_path_factory, "cf_ramp", _liq_ramp(2e-4))).flex_data
    pb = build(fd)
    rows = row_terms(pb, "ramp_sink_up_constraint", f"{_LIQ},")
    days = [f"t{24 * k + 1:04d}" for k in range(1, 7)]
    assert sorted(rows) == [f"ramp_sink_up_constraint[{_LIQ},y2030,{t}]" for t in days]
    bounds = row_bounds(pb, "ramp_sink_up_constraint", f"{_LIQ},")
    for k, t in enumerate(days):
        name = f"ramp_sink_up_constraint[{_LIQ},y2030,{t}]"
        prev = f"t{24 * k + 1:04d}"
        assert rows[name] == pytest.approx({f"v_flow[{_LIQ},y2030,{t}]": 1.0, f"v_flow[{_LIQ},y2030,{prev}]": -1.0})
        assert bounds[name][1] == pytest.approx(0.288 * _count(fd, "liquefier_A"), rel=1e-9)
    down = row_bounds(pb, "ramp_sink_down_constraint", f"{_LIQ},")
    assert len(down) == 6 and all(lo == pytest.approx(-0.288 * _count(fd, "liquefier_A"), rel=1e-9)
                                  for lo, _hi in down.values())
    fast = last_stochastic(_lh2(tmp_path_factory, "cf_ramp_fast", _liq_ramp(1e-3))).flex_data
    assert not row_terms(build(fast), "ramp_sink_up_constraint", f"{_LIQ},")


def test_coarse_ramp_roll_carry(tmp_path_factory) -> None:
    """Rolling 72 h / 36 h: roll 1 realizes t0001–t0036, so liquefier_A's last realized coarse step is the
    12-h step t0025–t0036.  Roll 2 starts at t0037 (its first coarse step t0037–t0060, 24 h) and ramps from the
    flow roll 1 has there, over the carried 12 h: room ρ̄ = 2e-4 · 60 · (12 + 24) / 2 = 0.216 of the capacity, so
    the up row reads v_flow[t0037] ≤ f̂ + 0.216 · capacity."""
    vals = _liq_ramp(2e-4) + [["solve", "lh2_week", "solve_mode", _s("rolling_window")],
                              ["solve", "lh2_week", "rolling_solve_horizon", _f(72.0)],
                              ["solve", "lh2_week", "rolling_solve_jump", _f(36.0)]]
    steps = list(without_realization(_lh2(tmp_path_factory, "cf_ramp_roll", vals)).values())
    first, second = steps[0], steps[1]
    f_hat = float(utf8(first.solution.value("v_flow")).filter(
        (pl.col("p") == "liquefier_A") & (pl.col("sink") == "lh2_A") & (pl.col("t") == "t0025"))["value"][0])
    fd = second.flex_data
    carried = utf8(fd.p_roll_continue_flow_duration.frame).filter(pl.col("p") == "liquefier_A")
    assert carried["value"].to_list() == [12.0]
    pb = build(fd)
    name = f"ramp_sink_up_constraint[{_LIQ},y2030,t0037]"
    assert row_terms(pb, "ramp_sink_up_constraint", f"{_LIQ},y2030,t0037")[name] == pytest.approx(
        {f"v_flow[{_LIQ},y2030,t0037]": 1.0})
    hi = row_bounds(pb, "ramp_sink_up_constraint", f"{_LIQ},y2030,t0037")[name][1]
    assert hi == pytest.approx(f_hat + 0.216 * _count(fd, "liquefier_A"), rel=1e-9)


# ---------------------------------------------------------------------------
# CO2 roll accumulator and the behaviour notice


def test_co2_accumulator_counts_every_step_of_a_coarse_flow(tmp_path) -> None:
    """compute_co2_rolling_accumulator reads a coarse flow (one LP column for the 3 steps of its coarse step) at
    every step: the realized tonnes equal those of the same flow written per step, 100 t/MWh · 10 · unitsize 5 ·
    Δt 2 h · 3 steps = 30000 t (the LP column alone would count one step, 10000 t)."""
    from types import SimpleNamespace

    from polar_high import Param

    from flextool.engine.preprocess.emit._co2_accumulators import compute_co2_rolling_accumulator
    from tests.engine.test_co2_accumulator_entity_axis import _Provider, _Sol

    steps = ["t1", "t2", "t3"]
    fd = SimpleNamespace(
        realized_dispatch=pl.DataFrame({"period": ["p1"] * 3, "step": steps}),
        p_step_duration=Param(("d", "t"), pl.DataFrame({"d": ["p1"] * 3, "t": steps, "value": [2.0] * 3})),
        p_timestep_weight=Param(("d", "t"), pl.DataFrame({"d": ["p1"] * 3, "t": steps, "value": [1.0] * 3})),
        process_source_sink_noEff=pl.DataFrame({"p": ["u1"], "source": ["coal_node"], "sink": ["elec"]}),
        process_source_sink_eff=None, process_unit=pl.DataFrame({"p": ["u1"]}), process_min_load_eff=None,
        p_co2_content=Param(("c",), pl.DataFrame({"c": ["coal"], "value": [100.0]})), p_slope=None,
        p_process_source_conversion_flow_coeff=None, p_process_sink_conversion_flow_coeff=None,
        group_node=pl.DataFrame({"g": ["cap_grp"], "n": ["coal_node"]}),
        p_period_share=Param(("d",), pl.DataFrame({"d": ["p1"], "value": [1.0]})),
        p_years_represented_d=Param(("d",), pl.DataFrame({"d": ["p1"], "value": [1.0]})),
        p_all_entity_unitsize=None, p_unitsize=Param(("p",), pl.DataFrame({"p": ["u1"], "value": [5.0]})),
        process_coarse_step=pl.DataFrame({"p": ["u1"] * 3, "d": ["p1"] * 3, "b_first": ["t1"] * 3, "t": steps,
                                          "w": [1.0 / 3.0] * 3}))
    provider = _Provider({
        "input/group__co2_method": pl.DataFrame({"group": ["cap_grp"], "co2_method": ["total"]}),
        "solve_data/commodity_node_co2": pl.DataFrame({"commodity": ["coal"], "node": ["coal_node"]})})

    def flows(ts):
        return pl.DataFrame({"p": ["u1"] * len(ts), "source": ["coal_node"] * len(ts), "sink": ["elec"] * len(ts),
                             "d": ["p1"] * len(ts), "t": ts, "value": [10.0] * len(ts)})

    tonnes = {}
    for kind, ts in (("coarse", ["t1"]), ("fine", steps)):
        out = compute_co2_rolling_accumulator(fd, _Sol(flows(ts)), work_folder=tmp_path, prior_cumulative_co2=None,
                                              provider=provider)
        tonnes[kind] = out["p_co2_cum_realized_tonnes"][0]
    assert tonnes["coarse"] == pytest.approx(30000.0, rel=1e-12)
    assert tonnes["fine"] == pytest.approx(tonnes["coarse"], rel=1e-12)


_MARKER = "flextool_behaviour_change:coarse_process_flows"


@pytest.mark.parametrize("marker", [True, False], ids=["marker", "no_marker"])
def test_behaviour_notice(tmp_path_factory, capsys, marker) -> None:
    """An upgraded database (marker present) prints one notice naming each coarse process whose results can
    change and why (here hpipe's varying cost and src's varying efficiency; src2 is exact and not named).
    Without the marker, nothing."""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db

    alt = f"cf_notice_{marker}"
    url, scen = make_db(tmp_path_factory, alt=alt, values=_hand_values(_half(1.0, 5.0), _half(0.5, 1.0)),
                        base_chain=_HAND_CHAIN, entities=_HAND_ENTITIES)
    if marker:
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=_MARKER, value="77")
            db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "one flow per coarse step" in ln]
    if not marker:
        assert not lines
        return
    assert len(lines) == 1 and _MARKER in lines[0]
    assert "hpipe (its variable cost varies within a coarse step)" in lines[0]
    assert "src (its efficiency varies within a coarse step)" in lines[0]
    assert "src2" not in lines[0]


# ---------------------------------------------------------------------------
# Stochastic and Benders solves

STOCH = FIX / "stochastics.json"
# stochastics.json with a second reservoir 'upper' feeding hydro_reservoir through 'upper_turbine'; both in the
# coarse group 'daily', so the turbine is a coarse process.  hydro_reservoir is in the stochastic group
# add_stochastics: its net flow (the turbine's included) is tied between the forecasts over the committed steps.
_STOCH_ENTITIES = [["group", "daily", None], ["group__node", ["daily", "hydro_reservoir"], None],
                   ["node", "upper", None], ["group__node", ["daily", "upper"], None],
                   ["unit", "upper_turbine", None], ["unit__inputNode", ["upper_turbine", "upper"], None],
                   ["unit__outputNode", ["upper_turbine", "hydro_reservoir"], None]]


def _stoch_values(step: float) -> list:
    return [["group", "daily", "new_stepduration", _f(step)],
            ["node", "upper", "node_type", _s("storage")], ["node", "upper", "existing", _f(3000.0)],
            ["node", "upper", "inflow", _f(200.0)], ["node", "upper", "storage_start_end_method", _s("fix_start")],
            ["node", "upper", "storage_state_start", _f(0.5)],
            ["node", "upper", "penalty_up", _f(5000.0)], ["node", "upper", "penalty_down", _f(5000.0)],
            ["unit", "upper_turbine", "efficiency", _f(1.0)], ["unit", "upper_turbine", "existing", _f(300.0)]]


def _stoch(tmp_path_factory, alt: str, scenario: str, step: float) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_stoch_values(step), base_chain=_chain(STOCH, scenario),
                        entities=_STOCH_ENTITIES, fixture_json=STOCH)
    return run(tmp_path_factory, url, scen)


def _forecasts(fd) -> dict[str, list[str]]:
    from flextool.engine.model._non_anticipativity import reference_forecasts

    return {a: f for a, f in reference_forecasts(fd) if len(f) >= 2}


def test_stochastic_coarse_arcs_in_every_forecast(tmp_path_factory, monkeypatch) -> None:
    """2-day stochastic dispatch (four forecasts, every step tied): upper_turbine has two daily columns in each
    forecast and none on the other hours, and the solve has the objective of the model with hourly flows (the
    turbine's parameters are constant)."""
    coarse = _stoch(tmp_path_factory, "cf_sto", "2_day_stochastic_dispatch", 24.0)
    step = last_stochastic(coarse)
    fd = step.flex_data
    forecasts = sorted({f for fs in _forecasts(fd).values() for f in fs})
    assert len(forecasts) == 4 and _coarse_procs(fd) == {"upper_turbine"}
    cols = _flow_cols(build(fd), "v_flow[upper_turbine,")
    assert sorted((_keys(c)[-2], _keys(c)[-1]) for c in cols) == [(d, t) for d in forecasts for t in ("t0001", "t0025")]
    _fine_flows(monkeypatch)
    fine = _stoch(tmp_path_factory, "cf_sto_fine", "2_day_stochastic_dispatch", 24.0)
    assert not _coarse_procs(last_stochastic(fine).flex_data)
    assert _total_objective(coarse) == pytest.approx(_total_objective(fine), rel=1e-9)


def _tie_reach(fd) -> list[tuple[str, str, str]]:
    """``(column, d, t)``: each flow column of a non_anticipativity_storage_use row with a fine step it holds
    that is not tied (a coarse column holds every fine step of its coarse step)."""
    tie = utf8(fd.dt_non_anticipativity)
    tied = {(f, t) for a, fs in _forecasts(fd).items() for f in fs
            for t in tie.filter(pl.col("d") == a)["t"].to_list()}
    held: dict[tuple[str, str, str], list[str]] = {}
    if fd.process_coarse_step is not None:
        for p, d, b, t in utf8(fd.process_coarse_step).select("p", "d", "b_first", "t").iter_rows():
            held.setdefault((p, d, b), []).append(t)
    out = []
    for terms in row_terms(build(fd), "non_anticipativity_storage_use").values():
        for col in terms:
            if not col.startswith("v_flow"):
                continue
            keys = _keys(col)
            p, d, t = keys[0], keys[-2], keys[-1]
            out += [(col, d, s) for s in held.get((p, d, t), [t]) if (d, s) not in tied]
    return out


@pytest.mark.parametrize("step", [24.0, 48.0])
def test_stochastic_tie_stays_within_the_committed_steps(tmp_path_factory, step) -> None:
    """Rolling stochastic dispatch (96-h horizon, 24-h jump): each roll ties the forecasts over its 24 committed
    hours only.  With 24-h coarse steps a committed day is one coarse step: upper_turbine is coarse and its tied
    columns hold only committed hours.  With 48-h steps the first coarse step of a roll holds 24 committed and 24
    uncommitted hours, and one flow per coarse step would tie the uncommitted ones too: upper_turbine keeps the
    solve's step in those rolls, so no tie row holds an uncommitted hour."""
    steps = without_realization(_stoch(tmp_path_factory, f"cf_sto_roll{int(step)}", "1_week_rolling_wind", step))
    rolls = [s.flex_data for s in steps.values() if _forecasts(s.flex_data)]
    assert len(rolls) >= 5
    for fd in rolls:
        assert row_terms(build(fd), "non_anticipativity_storage_use")
        assert _tie_reach(fd) == []
    coarse_in = ["upper_turbine" in _coarse_procs(fd) for fd in rolls]
    if step == 24.0:
        assert all(coarse_in)
    else:
        assert not coarse_in[0] and not all(coarse_in)


def test_benders_regions_keep_cross_region_flows_fine(tmp_path_factory, monkeypatch) -> None:
    """lh2_three_region_invest, Benders invest solve over region_A/B/C: in each region the liquefier (h2 -> lh2,
    both daily nodes of the region) is coarse, 7 columns per week; the pipes cross regions, so their half-flows
    keep 168 hourly columns (they are pinned to the master per hour); the master keeps every flow hourly.  The
    split runs as the Benders solve runs it (_benders._solve_benders_inner).  The liquefiers' parameters are
    constant within a day, so the Benders objective is that of hourly flows (within the gap tolerance)."""
    from flextool.engine.solve import _region_filter as rf

    url, scen = make_db(tmp_path_factory, alt="cf_benders", values=[], fixture_json=LH2,
                        base_chain=_chain(LH2, "lh2_three_region_invest"))
    steps = run(tmp_path_factory, url, scen)
    invest = next(s for k, s in steps.items() if "lh2_invest" in k)
    assert invest.is_benders
    fd = invest.flex_data
    regions = ["region_A", "region_B", "region_C"]
    membership = rf.load_region_membership(fd, regions)
    hosted = frozenset(rf.compute_master_hosted_nodes(fd, membership))
    splits = rf.split(fd, regions=regions, region_membership=membership, benders_uncap_cross_region=True,
                      master_hosted_nodes=hosted)
    for sp in splits:
        r = sp.region[-1]
        assert _coarse_procs(sp.data) == {f"liquefier_{r}"}
        cols = _flow_cols(build(sp.data), "v_flow[")
        assert len([c for c in cols if c.startswith(f"v_flow[liquefier_{r},")]) == 7
        assert sp.half_flows
        for hf in sp.half_flows:
            assert hf.original_p.startswith("pipe_")
            assert len([c for c in cols if c.startswith(f"v_flow[{hf.virtual_p},")]) == 168
    master = rf.master_network_data(fd, regions, region_membership=membership, master_hosted_nodes=hosted)
    assert master.processCoarse is None and master.process_coarse_step is None
    _fine_flows(monkeypatch)
    url, scen = make_db(tmp_path_factory, alt="cf_benders_fine", values=[], fixture_json=LH2,
                        base_chain=_chain(LH2, "lh2_three_region_invest"))
    fine = next(s for k, s in run(tmp_path_factory, url, scen).items() if "lh2_invest" in k)
    assert float(invest.obj) == pytest.approx(float(fine.obj), rel=1e-6)
