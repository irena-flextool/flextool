"""Coarse processes in rows that couple several entities: exact connections, DC power flow and the row rule.

Owner decision 2026-11-04 ("Coarse flows for units with online variables / reserves", (8) and the row rule): an
exact connection whose nodes are all coarse is coarse, with one direction per coarse step; a DC line whose nodes
are all coarse is coarse, and the nodes of a DC network whose every line is coarse (its nodes with the same coarse
steps) have one voltage angle per coarse step (Q3: a DC network with a line at the solve's step keeps every angle
per solve step).  A row whose every member is coarse
— the capacity-margin, inertia, non-synchronous and flow-group instant-flow rows of a group, a user constraint, the
forecast storage-use tie, a DC power-flow row between coarse-angle nodes — is one averaged row per coarse step
(``<family>_coarse``); a row that mixes coarse and fine members stays per solve step and reads the coarse terms at
their constant value.  The inertia / non-synchronous slack of an averaged row is one value per coarse step.

Exact checks are relational: the coarse model has the objective of the model whose processes keep the solve's step
with their values tied over each coarse step (every parameter constant within each step), and the structure of
the rows.  Fixtures: ``tests/fixtures/lh2_three_region.json`` (h2_* / lh2_* daily nodes; the liquefiers h2 -> lh2
and the pipes lh2 <-> lh2 are between daily nodes, the electrolysers elec -> h2 are not; one week of hourly steps),
``case14_dc_power_flow.json`` (4 hourly steps) and ``stochastics.json``, with override alternatives built from
JSON under the pytest tmp dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, last_stochastic, make_db, run, utf8, without_realization
from tests.engine.test_coarse_flows import (
    LH2_COARSE, STOCH, _STOCH_ENTITIES, _chain, _coarse_procs, _f, _forecasts, _keys, _lh2, _s, _stoch_values,
    _tmap_lh2, _total_objective,
)
from tests.engine.test_coarse_groups import DC, _pmap

pytestmark = pytest.mark.solver

_DAYS = [f"t{24 * k + 1:04d}" for k in range(7)]


def _cols(pb, prefix: str) -> list[str]:
    return [c for c in pb.canonicalise().col_names if c.startswith(prefix)]


def _steps_of(rows: dict) -> list[str]:
    return sorted(_keys(r)[-1] for r in rows)


# ---------------------------------------------------------------------------
# The fine model with every coarse value tied over its coarse step

_LEVELS = ("v_flow", "v_flow_back", "v_online_linear", "v_online_integer", "v_reserve", "v_exact_forward",
           "v_angle", "vq_inertia", "vq_non_synchronous")


def _add_ties(pb, fd) -> None:
    """In the fine LP ``pb`` of ``fd`` (every entity at the solve's step): ``x[…,t] = x[…,b]`` for each level x of
    a coarse entity and each interior fine step t of its coarse step b (the variable's coarse steps:
    ``engine.output._coarse_flows.coarse_steps`` — the coarse processes, the coarse-angle nodes, the groups of an
    averaged inertia / non-synchronous row)."""
    from polar_high.engine import Var

    from flextool.engine.output._coarse_flows import coarse_steps

    for name in _LEVELS:
        var = pb._vars.get(name)
        steps = coarse_steps(fd, name)
        if var is None or steps is None:
            continue
        dims = var.dims
        key = steps.columns[0]
        st = steps.rename({key: f"_{dims[0]}", "d": "_d", "b_first": "_b_first", "t": "_t"})
        fr = var.frame.with_columns(*(pl.col(c).cast(pl.Utf8).alias(f"_{c}") for c in dims))
        inner = (fr.join(st, on=[f"_{dims[0]}", "_d", "_t"], how="inner")
                 .filter(pl.col("_t") != pl.col("_b_first")))
        if inner.height == 0:
            continue
        first = fr.select(*[f"_{c}" for c in dims], pl.col("col_id").alias("__first"))
        keys = [f"_{c}" for c in dims[:-1]]
        paired = inner.join(first.rename({"_t": "_b_first"}), on=[*keys, "_b_first"], how="inner")
        assert paired.height == inner.height, name
        pb.add_cstr(f"tie_{name}", over=inner.select(dims), sense="==",
                    lhs_terms={"a": Var(name, dims, inner.select(*dims, "col_id")),
                               "b": -Var(name, dims, paired.select(*dims, pl.col("__first").alias("col_id")))},
                    rhs_terms={})


def _tied_fine(monkeypatch) -> None:
    """Build every solve fine (the model ignores the coarse processes) with the coarse values tied
    (:func:`_add_ties`).  The tie rows are a test device outside the autoscale registry: not strict there."""
    import flextool.engine.model.core as core
    from flextool.engine.model import _coarse_flows

    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    monkeypatch.setattr(_coarse_flows, "prepare", lambda b: None)
    original = core.build_flextool

    def build_tied(pb, data, *args, **kwargs):
        out = original(pb, data, *args, **kwargs)
        _add_ties(pb, data)
        return out

    monkeypatch.setattr(core, "build_flextool", build_tied)


def _assert_equals_tied(coarse: dict, tied: dict) -> None:
    assert len(coarse) == len(tied)
    assert _total_objective(coarse) == pytest.approx(_total_objective(tied), rel=1e-9)


# ---------------------------------------------------------------------------
# Exact connections: one direction per coarse step

_EXACT = [["connection", "pipe_AB", "transfer_method", _s("exact")],
          ["solve", "lh2_week", "solver_mip_gap", _f(1e-9)]]


def test_exact_connection_is_coarse_with_one_direction_per_day(tmp_path_factory) -> None:
    """pipe_AB (lh2_A <-> lh2_B, both daily) with the exact transfer method is coarse: its direction switch has 7
    columns (one per day), its direction rows are one averaged row per day (maxFlow_exact_{forward,reverse}_coarse:
    v_flow ≤ Ū_b · v_exact_forward[pipe_AB,d,b]) and none per hour; its flows have 7 columns per arc.  Before: an
    exact connection kept the solve's step (168 columns, 168 rows per direction)."""
    fd = last_stochastic(_lh2(tmp_path_factory, "rr_exact_struct", _EXACT)).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    pb = build(fd)
    assert [_keys(c)[-1] for c in _cols(pb, "v_exact_forward[pipe_AB,")] == _DAYS
    for fam in ("maxFlow_exact_forward_coarse", "maxFlow_exact_reverse_coarse"):
        assert _steps_of(row_terms(pb, fam, "pipe_AB,")) == _DAYS, fam
    for fam in ("maxFlow_exact_forward", "maxFlow_exact_reverse"):
        assert not row_terms(pb, fam, "pipe_AB,"), fam
    for arc in ("pipe_AB,lh2_A,lh2_B,", "pipe_AB,lh2_B,lh2_A,"):
        assert [_keys(c)[-1] for c in _cols(pb, f"v_flow[{arc}")] == _DAYS
    # One averaged row: the flow and the switch of the same day.
    row = row_terms(pb, "maxFlow_exact_forward_coarse", "pipe_AB,lh2_A,lh2_B,y2030,t0025")
    terms = next(iter(row.values()))
    assert set(terms) == {"v_flow[pipe_AB,lh2_A,lh2_B,y2030,t0025]", "v_exact_forward[pipe_AB,y2030,t0025]"}


def test_coarse_exact_equals_tied_fine_model(tmp_path_factory, monkeypatch) -> None:
    """With every parameter constant within a day, the coarse exact connection has the objective of the hourly
    model whose flows and direction switch are tied over each day."""
    coarse = _lh2(tmp_path_factory, "rr_exact_tie", _EXACT)
    assert "pipe_AB" in _coarse_procs(last_stochastic(coarse).flex_data)
    _tied_fine(monkeypatch)
    _assert_equals_tied(coarse, _lh2(tmp_path_factory, "rr_exact_tie_fine", _EXACT))


# ---------------------------------------------------------------------------
# DC power flow: angles per coarse step where every line of the DC network is coarse

_BUSES = [f"bus_{i}" for i in range(1, 15)]


def _dc_case(tmp_path_factory, alt: str, buses: list[str], load3: "list[float] | None" = None,
             buses_4h: tuple[str, ...] = ()) -> dict:
    """case14 on 4 hourly steps with ``buses`` in the 2-h group blk and ``buses_4h`` in the 4-h group blk4
    (constant inflows; ``load3`` the hourly inflow of bus_3)."""
    four = _pmap({f"t000{i}": 1.0 for i in range(1, 5)}, "time")
    ents = [["timeline", "tl_dispatch", None], ["group", "blk", None]]
    ents += [["group__node", ["blk", n], None] for n in buses]
    vals = [["timeline", "tl_dispatch", "timestep_duration", four],
            ["timeset", "ts_dispatch", "timeset_duration", _pmap({"t0001": 4.0}, "time")],
            ["group", "blk", "new_stepduration", _f(2.0)]]
    if buses_4h:
        ents += [["group", "blk4", None], *(["group__node", ["blk4", n], None] for n in buses_4h)]
        vals.append(["group", "blk4", "new_stepduration", _f(4.0)])
    if load3 is not None:
        vals.append(["node", "bus_3", "inflow", _pmap({f"t000{i}": v for i, v in enumerate(load3, 1)}, "time")])
    url, scen = make_db(tmp_path_factory, alt=alt, values=vals, base_chain=_chain(DC, "dc_opf_test"), entities=ents,
                        fixture_json=DC)
    return run(tmp_path_factory, url, scen)


def test_all_coarse_dc_network_has_angles_per_coarse_step(tmp_path_factory) -> None:
    """Every bus of case14 in a 2-h group: every line is coarse, every bus a coarse-angle node with 2 angle
    columns (t0001, t0003); each line has one dc_flow_eq_coarse row per coarse step on its flow, reverse flow and
    the two end angles of that step, and no hourly row; the reference angle is pinned once per coarse step.
    Before: DC lines kept the solve's step (angles, flows and rows on all 4 hours)."""
    fd = last_stochastic(_dc_case(tmp_path_factory, "rr_dc_all", _BUSES)).flex_data
    lines = set(utf8(fd.connection_dc_power_flow)["p"].to_list())
    assert lines and _coarse_procs(fd) == lines
    pb = build(fd)
    for n in _BUSES:
        assert [_keys(c)[-1] for c in _cols(pb, f"v_angle[{n},")] == ["t0001", "t0003"], n
    assert not row_terms(pb, "dc_flow_eq")
    rows = row_terms(pb, "dc_flow_eq_coarse", "line_1_2,")
    assert _steps_of(rows) == ["t0001", "t0003"]
    terms = rows["dc_flow_eq_coarse[line_1_2,bus_1,bus_2,p1,t0003]"]
    assert {c for c in terms if c.startswith("v_angle")} == {"v_angle[bus_1,p1,t0003]", "v_angle[bus_2,p1,t0003]"}
    assert {c for c in terms if c.startswith("v_flow")} == {"v_flow[line_1_2,bus_1,bus_2,p1,t0003]",
                                                            "v_flow_back[line_1_2,bus_1,bus_2,p1,t0003]"}
    assert _steps_of(row_terms(pb, "dc_reference_angle_eq")) == ["t0001", "t0003"]


def test_coarse_dc_network_equals_tied_fine_model(tmp_path_factory, monkeypatch) -> None:
    """The all-coarse DC network (constant inflows) has the objective of the hourly model whose flows and angles
    are tied over each 2-h step; and so does the mixed case (bus_13 and bus_14 coarse, line_13_14 coarse, the
    other lines at the solve's step)."""
    coarse = {k: _dc_case(tmp_path_factory, f"rr_dc_tie_{k}", b) for k, b in
              (("all", _BUSES), ("mixed", ["bus_13", "bus_14"]))}
    _tied_fine(monkeypatch)
    for k, b in (("all", _BUSES), ("mixed", ["bus_13", "bus_14"])):
        _assert_equals_tied(coarse[k], _dc_case(tmp_path_factory, f"rr_dc_tie_{k}_fine", b))


def test_dc_line_at_a_node_with_a_fine_line_keeps_per_step_rows(tmp_path_factory) -> None:
    """bus_13 and bus_14 in the 2-h group: line_13_14 is coarse (2 flow columns per arc), but bus_13 also has the
    hourly lines line_12_13 and line_6_13, and bus_14 the hourly line_9_14, so both keep an angle per solve step
    (Q3); line_13_14's dc_flow_eq stays per solve step (4 rows) with its flow at the coarse step's column (the row
    rule: a mixed row) and the behaviour notice names it."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    fd = last_stochastic(_dc_case(tmp_path_factory, "rr_dc_mixed", ["bus_13", "bus_14"])).flex_data
    assert _coarse_procs(fd) == {"line_13_14"}
    pb = build(fd)
    assert [_keys(c)[-1] for c in _cols(pb, "v_flow[line_13_14,")] == ["t0001", "t0003"]
    for n in ("bus_13", "bus_14"):
        assert len(_cols(pb, f"v_angle[{n},")) == 4
    rows = row_terms(pb, "dc_flow_eq", "line_13_14,")
    assert _steps_of(rows) == ["t0001", "t0002", "t0003", "t0004"]
    assert "v_flow[line_13_14,bus_13,bus_14,p1,t0001]" in rows["dc_flow_eq[line_13_14,bus_13,bus_14,p1,t0002]"]
    assert not row_terms(pb, "dc_flow_eq_coarse")
    why = coarse_process_reasons(fd, fd.process_coarse_step)["line_13_14"]
    assert any("DC power-flow row stays at the solve's step" in w for w in why)


def _per_step_angles(monkeypatch) -> None:
    """Build every model with every voltage angle per solve step (no coarse-angle node); flows stay coarse."""
    import flextool.engine.preprocess.flex_data as fdm

    monkeypatch.setattr(fdm, "coarse_angle_steps",
                        lambda d: pl.DataFrame(schema={c: pl.Utf8 for c in ("n", "d", "b_first", "t")}))


@pytest.mark.parametrize("buses, buses_4h, load3", [
    # bus_12's lines (line_6_12, line_12_13) are coarse, but bus_6 and bus_13 also have hourly lines.
    (["bus_6", "bus_12", "bus_13"], (), [-94.2, -250.0, -94.2, -250.0]),
    # Every line coarse, but bus_14 has 4-h steps and the lines at it 2-h steps.
    ([f"bus_{i}" for i in range(1, 14)], ("bus_14",), [-94.2, -94.2, -250.0, -250.0]),
], ids=["fine-line-in-network", "different-coarse-steps"])
def test_dc_network_not_all_coarse_keeps_per_step_angles(tmp_path_factory, monkeypatch, buses, buses_4h,
                                                         load3) -> None:
    """A coarse angle is held only where the whole DC network is coarse with one set of coarse steps.  In a network
    with an hourly line (or with nodes of different coarse steps) a coarse angle would force the coarse lines' rows
    (flow and one end angle constant over the step) to hold the neighbouring angles constant too, and through
    them the hourly lines' flows.  So every angle stays per solve step: the model equals the same coarse model
    built with every angle per solve step.  Before: bus_12 (bus_14) had one angle per coarse step and the objective
    was higher (the hourly lines lost their per-step physics)."""
    coarse = _dc_case(tmp_path_factory, f"rr_dc_net_{len(buses_4h)}", buses, load3, buses_4h)
    fd = last_stochastic(coarse).flex_data
    pb = build(fd)
    for n in ("bus_12", "bus_14"):
        assert len(_cols(pb, f"v_angle[{n},")) == 4, n
    assert not row_terms(pb, "dc_flow_eq_coarse")
    _per_step_angles(monkeypatch)
    per_step = _dc_case(tmp_path_factory, f"rr_dc_net_ps_{len(buses_4h)}", buses, load3, buses_4h)
    assert _total_objective(coarse) == pytest.approx(_total_objective(per_step), rel=1e-9)


def test_all_coarse_dc_network_equals_per_step_angles(tmp_path_factory, monkeypatch) -> None:
    """In an all-coarse DC network with one set of coarse steps the per-step rows of the coarse lines already hold
    every angle difference constant over a coarse step, and the reference angle is 0: one angle per coarse step
    is exact (equal objective with every angle per solve step), here with a bus_3 inflow that changes between the
    two coarse steps."""
    load3 = [-94.2, -94.2, -250.0, -250.0]
    coarse = _dc_case(tmp_path_factory, "rr_dc_allc", _BUSES, load3)
    _per_step_angles(monkeypatch)
    per_step = _dc_case(tmp_path_factory, "rr_dc_allc_ps", _BUSES, load3)
    assert _total_objective(coarse) == pytest.approx(_total_objective(per_step), rel=1e-9)


def test_angle_outputs_are_constant_within_a_coarse_step(tmp_path_factory) -> None:
    """The readers write a coarse-angle node's angle on every solve step, constant through its coarse step
    (``engine.output._coarse_flows.var_values``, copy mode): 4 values per bus, equal in pairs."""
    from flextool.engine.output._coarse_flows import var_values

    step = last_stochastic(_dc_case(tmp_path_factory, "rr_dc_out", _BUSES))
    ang = utf8(var_values(step.solution, step.flex_data, "v_angle"))
    for n in ("bus_5", "bus_9"):
        v = ang.filter(pl.col("n") == n).sort("t")
        assert v["t"].to_list() == ["t0001", "t0002", "t0003", "t0004"]
        vals = v["value"].to_list()
        assert vals[0] == vals[1] and vals[2] == vals[3]


# ---------------------------------------------------------------------------
# The row rule: group rows, flow-group instant limits and user constraints

# lh2_A / lh2_B with a constant demand (their hourly inflow would vary within a day).
_CONST_LH2 = [["node", "lh2_A", "inflow", _f(-60.0)], ["node", "lh2_B", "inflow", _f(-60.0)]]


def _group(g: str, nodes: list[str]) -> list:
    return [["group", g, None]] + [["group__node", [g, n], None] for n in nodes]


# Inertia: liquefier_A / liquefier_B carry an inertia constant on their lh2 outputs; the group lh2_in (lh2_A,
# lh2_B) needs more inertia than the liquefiers give at their usual load, so the slack is used.
_INERTIA_ENTS = _group("lh2_in", ["lh2_A", "lh2_B"])
_INERTIA = _CONST_LH2 + [
    ["group", "lh2_in", "has_inertia", _s("yes")], ["group", "lh2_in", "inertia_limit", _f(5000.0)],
    ["group", "lh2_in", "penalty_inertia", _f(10.0)],
    ["unit__outputNode", ["liquefier_A", "lh2_A"], "inertia_constant", _f(2.0)],
    ["unit__outputNode", ["liquefier_B", "lh2_B"], "inertia_constant", _f(3.0)]]
# Non-synchronous: liquefier_A's output is non-synchronous, the group lh2_ns (lh2_A, lh2_B) allows 2 % (binding).
_NONSYNC_ENTS = _group("lh2_ns", ["lh2_A", "lh2_B"])
_NONSYNC = _CONST_LH2 + [
    ["group", "lh2_ns", "has_non_synchronous", _s("yes")], ["group", "lh2_ns", "non_synchronous_limit", _f(0.02)],
    ["group", "lh2_ns", "penalty_non_synchronous", _f(10.0)],
    ["unit__outputNode", ["liquefier_A", "lh2_A"], "is_non_synchronous", _s("yes")]]


def _case(tmp_path_factory, alt: str, values: list, entities: list) -> dict:
    return _lh2(tmp_path_factory, alt, values, entities)


@pytest.mark.parametrize("family,ents,vals,slack", [
    ("inertia_constraint", _INERTIA_ENTS, _INERTIA, "vq_inertia[lh2_in,"),
    ("non_sync_constraint", _NONSYNC_ENTS, _NONSYNC, "vq_non_synchronous[lh2_ns,"),
])
def test_group_row_of_coarse_processes_is_one_row_per_day(tmp_path_factory, family, ents, vals, slack) -> None:
    """A group of daily nodes whose row members are all coarse (the liquefiers and pipes at lh2_A / lh2_B) has one
    averaged row per day (<family>_coarse) and one slack column per day; no hourly row.  Before: a process at a
    node of an inertia / non-synchronous group kept the solve's step (168 rows and slack columns)."""
    fd = last_stochastic(_case(tmp_path_factory, f"rr_{family}", vals, ents)).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    pb = build(fd)
    assert _steps_of(row_terms(pb, f"{family}_coarse")) == _DAYS
    assert not row_terms(pb, family)
    assert [_keys(c)[-1] for c in _cols(pb, slack)] == _DAYS


@pytest.mark.parametrize("family,ents,vals", [
    ("inertia", _INERTIA_ENTS, _INERTIA), ("non_sync", _NONSYNC_ENTS, _NONSYNC)])
def test_group_row_equals_tied_fine_model(tmp_path_factory, monkeypatch, family, ents, vals) -> None:
    """With every parameter constant within a day, the averaged inertia / non-synchronous rows give the objective
    of the hourly model whose flows and slacks are tied over each day; the slack outputs are constant within each
    day (copy mode) and equal the tied model's."""
    from flextool.engine.output._coarse_flows import var_values

    slack = {"inertia": "vq_inertia", "non_sync": "vq_non_synchronous"}[family]
    coarse = _case(tmp_path_factory, f"rr_tie_{family}", vals, ents)
    _tied_fine(monkeypatch)
    tied = _case(tmp_path_factory, f"rr_tie_{family}_fine", vals, ents)
    _assert_equals_tied(coarse, tied)
    c, f = (last_stochastic(x) for x in (coarse, tied))
    vc = utf8(var_values(c.solution, c.flex_data, slack)).sort("g", "d", "t")
    vf = utf8(var_values(f.solution, f.flex_data, slack)).sort("g", "d", "t")
    assert vc.height == vf.height == 168
    assert vc["value"].to_list() == pytest.approx(vf["value"].to_list(), abs=1e-6)
    assert vc["value"].sum() > 1e-6


_FG_ENTS = [["flowGroup", "fg_pipe", None], ["flowGroup__connection__node", ["fg_pipe", "pipe_AB", "lh2_B"], None]]


def _fg(cap) -> list:
    return [["flowGroup", "fg_pipe", "max_instant_flow", cap]]


def test_instant_flow_of_coarse_flow_group_is_one_row_per_day(tmp_path_factory) -> None:
    """flowGroup fg_pipe (pipe_AB into lh2_B) with a max_instant_flow that is 10 MW in the first 12 hours of each
    day and 30 MW in the last 12: one maxInstant_flow_coarse row per day and none per hour.  Hand derivation
    (EQ. <family>_coarse, w = Δt/Δ_b = 1/24): the row's bound is the step mean of the cap,
    (12·10 + 12·30)/24 = 20 MW, on the day's one pipe flow column.  The behaviour notice names the varying limit."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    cap = _tmap_lh2(lambda h: 10.0 if h % 24 < 12 else 30.0)
    fd = last_stochastic(_case(tmp_path_factory, "rr_fg_struct", _fg(cap), _FG_ENTS)).flex_data
    assert "pipe_AB" in _coarse_procs(fd)
    pb = build(fd)
    assert not row_terms(pb, "maxInstant_flow")
    bounds = row_bounds(pb, "maxInstant_flow_coarse", "fg_pipe,")
    assert _steps_of(bounds) == _DAYS
    assert all(hi == pytest.approx(20.0) for _, hi in bounds.values())
    terms = row_terms(pb, "maxInstant_flow_coarse",
                      "fg_pipe,y2030,t0025")["maxInstant_flow_coarse[fg_pipe,y2030,t0025]"]
    assert set(terms) == {"v_flow[pipe_AB,lh2_A,lh2_B,y2030,t0025]", "v_flow[pipe_AB,lh2_B,lh2_A,y2030,t0025]"}
    why = coarse_process_reasons(fd, fd.process_coarse_step)["pipe_AB"]
    assert any("flow-group instant-flow row of 'fg_pipe' uses the step mean" in w for w in why)


def _partial_cap() -> list:
    """max_instant_flow 10 MW authored on the first 12 hours of each day only (no limit on the other hours)."""
    return b64({"index_type": "str", "index_name": "time",
                "data": [[f"t{h + 1:04d}", 10.0] for h in range(168) if h % 24 < 12]}, "map")


def test_instant_flow_on_part_of_a_coarse_step_stays_per_step(tmp_path_factory, monkeypatch) -> None:
    """fg_pipe's max_instant_flow is authored on the first 12 hours of each day only, so the fine model has a row
    on those hours and none on the others.  An averaged row would count the 12 hours without a row as a limit of 0
    (bound (12·10 + 12·0)/24 = 5 MW); instead the row stays per solve step on the hours it covers: 12 rows per day
    with the authored bound 10 MW, each on the day's pipe flow column, no maxInstant_flow_coarse row; and the model
    equals the hourly model with the pipe flows tied over each day."""
    coarse = _case(tmp_path_factory, "rr_fg_partial", _fg(_partial_cap()), _FG_ENTS)
    fd = last_stochastic(coarse).flex_data
    assert "pipe_AB" in _coarse_procs(fd)
    pb = build(fd)
    assert not row_terms(pb, "maxInstant_flow_coarse")
    bounds = row_bounds(pb, "maxInstant_flow", "fg_pipe,")
    assert len(bounds) == 7 * 12
    assert all(hi == pytest.approx(10.0) for _, hi in bounds.values())
    terms = row_terms(pb, "maxInstant_flow", "fg_pipe,y2030,t0030")["maxInstant_flow[fg_pipe,y2030,t0030]"]
    assert set(terms) == {"v_flow[pipe_AB,lh2_A,lh2_B,y2030,t0025]", "v_flow[pipe_AB,lh2_B,lh2_A,y2030,t0025]"}
    _tied_fine(monkeypatch)
    _assert_equals_tied(coarse, _case(tmp_path_factory, "rr_fg_partial_fine", _fg(_partial_cap()), _FG_ENTS))


def test_instant_flow_equals_tied_fine_model(tmp_path_factory, monkeypatch) -> None:
    """A constant max_instant_flow (15 MW, binding: the pipe would carry more): coarse == tied hourly model."""
    coarse = _case(tmp_path_factory, "rr_fg_tie", _fg(_f(15.0)), _FG_ENTS)
    _tied_fine(monkeypatch)
    _assert_equals_tied(coarse, _case(tmp_path_factory, "rr_fg_tie_fine", _fg(_f(15.0)), _FG_ENTS))


def test_instant_flow_with_a_fine_member_stays_per_step(tmp_path_factory) -> None:
    """fg_pipe also holds electrolyser_A (elec_A hourly -> h2_A): the group mixes coarse and fine flows, so its
    maxInstant_flow rows stay per solve step (168), each reading pipe_AB's flow at its day's column, and the
    behaviour notice names pipe_AB's mixed row."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    ents = _FG_ENTS + [["flowGroup__connection__node", ["fg_pipe", "electrolyser_A", "h2_A"], None]]
    fd = last_stochastic(_case(tmp_path_factory, "rr_fg_mixed", _fg(_f(15.0)), ents)).flex_data
    pb = build(fd)
    rows = row_terms(pb, "maxInstant_flow", "fg_pipe,")
    assert len(rows) == 168 and not row_terms(pb, "maxInstant_flow_coarse")
    assert "v_flow[pipe_AB,lh2_A,lh2_B,y2030,t0025]" in rows["maxInstant_flow[fg_pipe,y2030,t0030]"]
    why = coarse_process_reasons(fd, fd.process_coarse_step)["pipe_AB"]
    assert any("flow-group instant-flow row of 'fg_pipe' stays at the solve's step" in w for w in why)


def _uc(coeffs: list[tuple[str, str]]) -> tuple[list, list]:
    """User constraint cap_ab (≤ 30) with flow coefficient 1 on each ``(connection, node)``."""
    ents = [["constraint", "cap_ab", None]] + [["connection__node", [c, n], None] for c, n in coeffs]
    vals = [["constraint", "cap_ab", "sense", _s("less_than")], ["constraint", "cap_ab", "constant", _f(30.0)]]
    vals += [["connection__node", [c, n], "constraint_flow_coeff",
              b64({"index_type": "str", "index_name": "constraint", "data": [["cap_ab", 1.0]]}, "map")]
             for c, n in coeffs]
    return ents, vals


def test_user_constraint_on_coarse_flows_is_one_row_per_day(tmp_path_factory, monkeypatch) -> None:
    """A user constraint with a flow coefficient on pipe_AB only: pipe_AB is coarse and the constraint has one
    process_constraint_less_than_coarse row per day, none per hour; with constant coefficients it is the hourly
    row itself, so the objective equals the tied hourly model's.  Before: a flow term in a user constraint kept
    pipe_AB at the solve's step."""
    ents, vals = _uc([("pipe_AB", "lh2_B")])
    coarse = _case(tmp_path_factory, "rr_uc", vals, ents)
    fd = last_stochastic(coarse).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    pb = build(fd)
    assert _steps_of(row_terms(pb, "process_constraint_less_than_coarse", "cap_ab,")) == _DAYS
    assert not row_terms(pb, "process_constraint_less_than", "cap_ab,")
    _tied_fine(monkeypatch)
    _assert_equals_tied(coarse, _case(tmp_path_factory, "rr_uc_fine", vals, ents))


def test_user_constraint_mixing_coarse_and_fine_flows_stays_per_step(tmp_path_factory) -> None:
    """The user constraint also has a flow coefficient on electrolyser_A (hourly): its rows stay per solve step
    (168) and read pipe_AB's flow at one column per day (the row rule for a mixed row)."""
    ents, vals = _uc([("pipe_AB", "lh2_B"), ("electrolyser_A", "h2_A")])
    fd = last_stochastic(_case(tmp_path_factory, "rr_uc_mixed", vals, ents)).flex_data
    assert "pipe_AB" in _coarse_procs(fd)
    pb = build(fd)
    rows = row_terms(pb, "process_constraint_less_than", "cap_ab,")
    assert len(rows) == 168 and not row_terms(pb, "process_constraint_less_than_coarse")
    pipe = {c for terms in rows.values() for c in terms if c.startswith("v_flow[pipe_AB,lh2_B,")}
    assert sorted(_keys(c)[-1] for c in pipe) == _DAYS


# Capacity margin of h2_A in an investment solve: its consuming unit liquefier_A is coarse.
_CM_ENTS = _group("cm_h2", ["h2_A"])
_CM = [["group", "cm_h2", "capacity_margin_method", _s("manual")], ["group", "cm_h2", "capacity_margin", _f(10.0)],
       ["group", "cm_h2", "penalty_capacity_margin", _f(100.0)],
       ["solve", "lh2_week", "invest_periods", b64({"value_type": "str", "data": ["y2030"]}, "array")],
       ["solve", "lh2_week", "realized_invest_periods", b64({"value_type": "str", "data": ["y2030"]}, "array")]]


def test_capacity_margin_of_coarse_consumers_is_one_row_per_day(tmp_path_factory, monkeypatch) -> None:
    """Group cm_h2 (h2_A) with a capacity margin: the unit consuming from h2_A (liquefier_A) is coarse, so the
    margin has one capacityMargin_coarse row per day and none per hour; without a time-varying profile or inflow
    at h2_A it is the hourly row, so the objective equals the tied hourly model's."""
    coarse = _case(tmp_path_factory, "rr_cm", _CM, _CM_ENTS)
    fd = last_stochastic(coarse).flex_data
    pb = build(fd)
    assert _steps_of(row_terms(pb, "capacityMargin_coarse", "cm_h2,")) == _DAYS
    assert not row_terms(pb, "capacityMargin", "cm_h2,")
    _tied_fine(monkeypatch)
    _assert_equals_tied(coarse, _case(tmp_path_factory, "rr_cm_fine", _CM, _CM_ENTS))


# ---------------------------------------------------------------------------
# The forecast storage-use tie, averaged over the coarse step

# stochastics.json with 'upper' (a daily storage node whose only process is upper_turbine, coarse) also in the
# stochastic group: its tie has only coarse members.  hydro_reservoir also has the hourly hydro plant: mixed.
_STOCH_UPPER = _STOCH_ENTITIES + [["group__node", ["add_stochastics", "upper"], None]]


def _stoch_upper(tmp_path_factory, alt: str) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_stoch_values(24.0),
                        base_chain=_chain(STOCH, "2_day_stochastic_dispatch"), entities=_STOCH_UPPER,
                        fixture_json=STOCH)
    return run(tmp_path_factory, url, scen)


def test_storage_use_tie_of_a_coarse_node_is_averaged(tmp_path_factory, monkeypatch) -> None:
    """2-day stochastic dispatch, every step tied: the tie of 'upper' (members: upper_turbine, coarse) is one
    non_anticipativity_storage_use_coarse row per forecast pair and day (the row's Δt-weighted mean, so a slope
    that varies within the day enters by its step mean instead of over-determining 24 hourly ties of one flow
    column); hydro_reservoir, with the hourly hydro plant, keeps its hourly ties, each reading upper_turbine's
    flow at its day's column.  The objective equals the tied hourly model's."""
    coarse = _stoch_upper(tmp_path_factory, "rr_na")
    fd = last_stochastic(coarse).flex_data
    pairs = sum(len(f) - 1 for f in _forecasts(fd).values())
    assert pairs >= 1 and "upper_turbine" in _coarse_procs(fd)
    pb = build(fd)
    rows = row_terms(pb, "non_anticipativity_storage_use_coarse", "upper,")
    assert len(rows) == 2 * pairs
    assert {_keys(r)[-1] for r in rows} == {"t0001", "t0025"}
    assert not row_terms(pb, "non_anticipativity_storage_use", "upper,")
    hydro = row_terms(pb, "non_anticipativity_storage_use", "hydro_reservoir,")
    assert len(hydro) == 48 * pairs
    _tied_fine(monkeypatch)
    _assert_equals_tied(without_realization(coarse), without_realization(_stoch_upper(tmp_path_factory, "rr_na_f")))
