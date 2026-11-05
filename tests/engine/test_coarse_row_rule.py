"""Coarse processes in rows that couple several entities: exact connections, DC power flow and the row rule.

Owner decision 2026-11-04 ("Coarse flows for units with online variables / reserves", (8) and the row rule): an
exact connection whose nodes are all coarse is coarse, with one direction per coarse step; a DC line whose nodes
are all coarse is coarse, and a coarse node whose every DC line is coarse has one voltage angle per coarse step
(Q3: a node with a DC line at the solve's step keeps its angle per solve step); a DC power-flow row between
coarse-angle nodes is one averaged row per coarse step, any other stays per solve step and reads a coarse flow at its
constant value (the row rule).

Exact checks are relational: the coarse model has the objective of the model whose processes keep the solve's step
with their values tied over each coarse step (every parameter constant within each step), and the structure of
the rows.  Fixtures: ``tests/fixtures/lh2_three_region.json`` (h2_* / lh2_* daily nodes; the liquefiers h2 -> lh2
and the pipes lh2 <-> lh2 are between daily nodes, the electrolysers elec -> h2 are not; one week of hourly steps),
``case14_dc_power_flow.json`` (4 hourly steps), with override alternatives built from
JSON under the pytest tmp dir.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build, last_stochastic, make_db, run, utf8
from tests.engine.test_coarse_flows import (
    LH2_COARSE, _chain, _coarse_procs, _f, _keys, _lh2, _s, _total_objective,
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
# DC power flow: angles per coarse step where every DC line at the node is coarse

_BUSES = [f"bus_{i}" for i in range(1, 15)]


def _dc_case(tmp_path_factory, alt: str, buses: list[str]) -> dict:
    """case14 on 4 hourly steps with ``buses`` in the 2-h group blk (constant inflows)."""
    four = _pmap({f"t000{i}": 1.0 for i in range(1, 5)}, "time")
    ents = [["timeline", "tl_dispatch", None], ["group", "blk", None]]
    ents += [["group__node", ["blk", n], None] for n in buses]
    vals = [["timeline", "tl_dispatch", "timestep_duration", four],
            ["timeset", "ts_dispatch", "timeset_duration", _pmap({"t0001": 4.0}, "time")],
            ["group", "blk", "new_stepduration", _f(2.0)]]
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

