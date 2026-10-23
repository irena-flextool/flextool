"""Upward reserve of a process with online variables comes out of its online capacity.

Reserve held upward at a node the process delivers into competes with its flow for the units online:

* ``maxFlow_online_<kind>``: ``v_flow + Σ_r v_reserve_up ≤ v_online · availability · capacity_max_coeff``
  per capped arc (the reserve at the arc's sink);
* ``maxOutputSum_online_<kind>``: ``Σ_k v_flow + Σ_k Σ_r v_reserve_up ≤ v_online · availability`` for an
  indirect (multi-output) unit.

Before, only the built-capacity rows (``maxFlow``, ``maxOutputSum``) carried the reserve, so an offline
unit could hold upward reserve for free.  Processes without online variables keep the built-capacity limit.

Fixtures: ``tests.json`` + override alternatives (built from JSON).  Units: the helpers return MW
(variable × unitsize); availability and ``capacity_max_coeff`` are 1 for the units used here.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import (
    CHP, CHP_CHAIN, COAL, PLANT, PLANT_CHAIN, b64, build, cstr_over, demand,
    flow, no_ratio_constraint, online, reserve, solve, uc, utf8,
)

TOL = 1e-5


def _up_reserve(providers: list, *, reservation: float, penalty: float) -> tuple[list, list]:
    """Timeseries up reserve ``ru`` on group ``electricity``, offered by ``providers`` =
    [(class, process, node), ...] with max_share 1 and reliability 1."""
    grp = ["ru", "up", "electricity"]
    ents = [["reserve", "ru", None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64("timeseries_only", "str")],
            ["reserve__upDown__group", grp, "reservation", b64(reservation, "float")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(penalty, "float")]]
    for cls, p, n in providers:
        rel = f"reserve__upDown__{cls}__node"
        ents.append([rel, ["ru", "up", p, n], None])
        vals += [[rel, ["ru", "up", p, n], "is_enabled", b64("yes", "str")],
                 [rel, ["ru", "up", p, n], "max_share", b64(1.0, "float")],
                 [rel, ["ru", "up", p, n], "reliability", b64(1.0, "float")]]
    return ents, vals


def _vq(step) -> pl.Series:
    return (utf8(step.solution.value("vq_reserve"))
            .filter(pl.col("r") == "ru").sort("d", "t")["value"])


def _row_terms(pb, family: str) -> dict[str, dict[str, float]]:
    """``{row: {column: coefficient}}`` of the canonical rows of ``family``."""
    import collections

    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(family + "[")}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            ri = m.row_idx[k]
            if ri in rows:
                out[rows[ri]][m.col_names[c]] = m.val[k]
    return out


def _reserve_coefs(terms: dict[str, float], process: str, node: str) -> list[float]:
    return [v for c, v in terms.items() if c.startswith(f"v_reserve[{process},ru,up,{node},")]


# The offline-unit case: wind covers the 20 MW load, the coal plant (linear or integer UC, min_load 0.5,
# startup_cost 1e5) is the only provider of a 50 MW upward reserve priced at 1e5 €/MWh.
def _offline_case(tpf, alt: str, method: str):
    ents, vals = _up_reserve([("unit", PLANT, "west")], reservation=50.0, penalty=1.0e5)
    return solve(tpf, alt=alt, base_chain=PLANT_CHAIN + ["wind"], entities=ents,
                 values=vals + uc(PLANT, 0.5, method) + [
                     ["unit", PLANT, "startup_cost", b64(1.0e5, "float")],
                     demand("west", 20.0)])


@pytest.mark.parametrize("method,kind", [("linear", "linear"), ("binary", "integer")])
def test_offline_unit_holds_no_up_reserve(tmp_path_factory, method, kind) -> None:
    """An offline unit offers no upward reserve; the reserve requirement drives commitment.

    Before the fix the linear solve kept the coal plant offline (online 0) while it held the whole 50 MW
    reserve (review reproduction ``i25_offline``)."""
    step = _offline_case(tmp_path_factory, f"uro_{kind}", method)
    out = flow(step, PLANT, COAL, "west")
    up = reserve(step, PLANT, "ru", "west")
    on = online(step, PLANT, kind)
    assert ((on - out - up) >= -TOL).all()
    assert (up.filter(on <= TOL).abs() <= TOL).all()
    # The requirement is met by committing the unit, not by an unconstrained reserve.
    met = _vq(step) <= 1e-9
    assert met.any()
    assert ((on - out - up).filter(met) >= -1e-6).all()
    assert (on.filter(up > TOL) > 0.0).all()
    assert up.filter(met).min() == pytest.approx(50.0, abs=1e-4)

    terms = _row_terms(build(step.flex_data), f"maxFlow_online_{kind}")
    plant_rows = {r: t for r, t in terms.items() if r.startswith(f"maxFlow_online_{kind}[{PLANT},{COAL},west,")}
    assert plant_rows
    for row, t in plant_rows.items():
        assert _reserve_coefs(t, PLANT, "west") == [pytest.approx(1.0)], row


def test_indirect_online_unit_up_reserve_sum(tmp_path_factory) -> None:
    """Two-output CHP (linear UC, min_load 0.8): Σ outputs + upward reserve ≤ online capacity.

    Demands: west 100 MW, heat 300 MW, so Σ outputs = 400 MW; the 200 MW reserve at west is offered only
    by the CHP.  Per arc, west 100 + reserve 200 ≤ online already holds whenever online ≥ 400 (the heat
    output alone needs that), so only the summed row can tie the reserve to the units online.  Before the
    fix that row had no reserve term: min_load 0.8 caps online at 400 / 0.8 = 500 MW while the reserve
    stayed 200 MW (outputs + reserve = 600 MW > online), limited only by the 1000 MW built capacity."""
    ents, vals = _up_reserve([("unit", CHP, "west")], reservation=200.0, penalty=1.0e5)
    step = solve(tmp_path_factory, alt="uro_chp", base_chain=CHP_CHAIN, entities=ents,
                 values=no_ratio_constraint() + vals + uc(CHP, 0.8)
                 + [demand("west", 100.0), demand("heat", 300.0)])
    west = flow(step, CHP, CHP, "west")
    heat = flow(step, CHP, CHP, "heat")
    up = reserve(step, CHP, "ru", "west")
    on = online(step, CHP)
    assert ((on - west - heat - up) >= -TOL).all()
    assert up.max() > 1.0

    pb = build(step.flex_data)
    for row, t in _row_terms(pb, "maxOutputSum_online_linear").items():
        assert _reserve_coefs(t, CHP, "west") == [pytest.approx(1.0)], row
    west_rows = {r: t for r, t in _row_terms(pb, "maxFlow_online_linear").items()
                 if r.startswith(f"maxFlow_online_linear[{CHP},{CHP},west,")}
    assert west_rows
    for row, t in west_rows.items():
        assert _reserve_coefs(t, CHP, "west") == [pytest.approx(1.0)], row
    heat_rows = {r: t for r, t in _row_terms(pb, "maxFlow_online_linear").items()
                 if r.startswith(f"maxFlow_online_linear[{CHP},{CHP},heat,")}
    assert heat_rows
    for row, t in heat_rows.items():
        assert not [c for c in t if c.startswith("v_reserve[")], row


def test_online_connection_up_reserve(tmp_path_factory) -> None:
    """The reserve term is process-generic: a connection with online variables gets it on its arc
    west→east for the upward reserve held at its sink east (coefficient +1), not for the one held at its
    source west.

    The input path gives connections no online variables today (``process_online_*`` read only
    ``unit.startup_method``; see specs/github_issues.md), so the unidirectional connection ``west_east``
    is added to the online sets of the built FlexData directly.  Only unidirectional connections map to
    an online method, so there is no reverse arc to cover; two-way one-variable arcs keep
    ``maxFlow_back`` without a reserve term, unchanged."""
    import dataclasses

    ents, vals = _up_reserve([("connection", "west_east", "east"), ("connection", "west_east", "west")],
                             reservation=10.0, penalty=1.0e3)
    step = solve(tmp_path_factory, alt="uro_conn",
                 base_chain=PLANT_CHAIN + ["coal_min_load", "wind", "network", "unidirectional"], entities=ents,
                 values=vals)
    fd = step.flex_data
    p_dtype = fd.process_online_linear.schema["p"]
    conn = pl.DataFrame({"p": ["west_east"]}).with_columns(pl.col("p").cast(p_dtype))
    conn_dt = conn.join(fd.dt.select("d", "t"), how="cross").select(fd.p_online_dt.columns)
    fd = dataclasses.replace(
        fd,
        process_online=pl.concat([fd.process_online, conn.select(fd.process_online.columns)]),
        process_online_linear=pl.concat([fd.process_online_linear, conn]),
        p_online_dt=pl.concat([fd.p_online_dt, conn_dt]))
    terms = _row_terms(build(fd), "maxFlow_online_linear")
    fwd = {r: t for r, t in terms.items() if r.startswith("maxFlow_online_linear[west_east,west,east,")}
    assert fwd
    for row, t in fwd.items():
        assert _reserve_coefs(t, "west_east", "east") == [pytest.approx(1.0)], row
        assert _reserve_coefs(t, "west_east", "west") == [], row


def test_non_online_unit_keeps_built_capacity_limit(tmp_path_factory) -> None:
    """Without online variables the reserve stays limited by the built capacity (``maxFlow``): with an
    unmeetable requirement the coal plant fills its 500 MW with flow + reserve at every step."""
    ents, vals = _up_reserve([("unit", PLANT, "west")], reservation=5000.0, penalty=1.0)
    step = solve(tmp_path_factory, alt="uro_noon", base_chain=PLANT_CHAIN, entities=ents,
                 values=vals + [demand("west", 100.0)])
    pb = build(step.flex_data)
    assert cstr_over(pb, "maxFlow_online_linear") is None
    assert cstr_over(pb, "maxFlow_online_integer") is None
    out = flow(step, PLANT, COAL, "west")
    up = reserve(step, PLANT, "ru", "west")
    assert (out + up).to_list() == pytest.approx([500.0] * out.len(), abs=1e-4)
