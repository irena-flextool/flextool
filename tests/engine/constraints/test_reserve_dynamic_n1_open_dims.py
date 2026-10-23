"""Regression: reserveBalance_dynamic_eq / _n_1_eq must not leave the
v_flow-native ``sink`` / ``source`` axis open in the RHS flow term.

Before the fix, the RHS pieces renamed ``pss``'s sink/source column to
``n`` for the join frame but kept ``v_flow``'s own ``source``/``sink``
columns, which were never tied to ``n`` nor summed out.  The residual
open axis was harmless until a model actually populated the
``dynamic`` / ``n_1`` method with a real ``increase_reserve_ratio`` /
``large_failure_ratio`` — then ``canonicalise()`` (invoked by every
MPS write / commercial-subprocess solve) raised::

    ValueError: constraint 'reserveBalance_dynamic_eq': term has open
    dims ('sink', ...), aggregate ['sink'] via Sum() before adding.
"""
from __future__ import annotations

import dataclasses

import polars as pl
import pytest
from polar_high import Param, Problem

from flextool.engine import build_flextool

from .conftest import solver_options


def _add_increase_reserve_ratio(d):
    """Turn the timeseries reserve into a *dynamic* reserve whose RHS is
    driven by the producer's flow into ``n1``."""
    irr_set = pl.DataFrame(
        {"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"]})
    irr_param = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"],
                      "value": [0.1]}))
    return dataclasses.replace(
        d,
        reserve_upDown_group_method_timeseries=pl.DataFrame(
            schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8,
                    "method": pl.Utf8}),
        reserve_upDown_group_method_dynamic=pl.DataFrame(
            {"r": ["r1"], "ud": ["up"], "g": ["g"], "method": ["dynamic"]}),
        process_reserve_upDown_node_increase_reserve_ratio=irr_set,
        p_process_reserve_upDown_node_increase_reserve_ratio_value=irr_param,
    )


def _row_terms(m, needle: str):
    """Return {row_name: {col_name: coeff}} for canonical rows matching
    ``needle`` (CSC walk of the canonical matrix)."""
    import collections
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if needle in nm}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            ri = m.row_idx[k]
            if ri in rows:
                out[rows[ri]][m.col_names[c]] = m.val[k]
    return out


def test_reserve_dynamic_canonicalises(toy_group_reserve):
    d = _add_increase_reserve_ratio(toy_group_reserve)
    pb = Problem()
    build_flextool(pb, d)
    # This is what crashed on a real model (MPS write / commercial subprocess).
    m = pb.canonicalise()
    assert "reserveBalance_dynamic_eq" in set(pb.cstr_names())

    # The dynamic RHS must pull in exactly the producer flow whose *sink*
    # is the reserve node n1, with coefficient -(unitsize · irr) moved to
    # the LHS = -(100 · 0.1) = -10.  Regression guard: before the fix the
    # sink axis stayed open and the term crashed canonicalise(); a naive
    # "fix" that dropped the axis would instead drop / duplicate the term.
    rows = _row_terms(m, "reserveBalance_dynamic_eq")
    assert rows, "no dynamic rows in canonical matrix"
    for rname, terms in rows.items():
        flow_terms = {c: v for c, v in terms.items() if c.startswith("v_flow[")}
        assert len(flow_terms) == 1, f"{rname}: {flow_terms}"
        (coeff,) = flow_terms.values()
        assert coeff == pytest.approx(-10.0, rel=1e-9), f"{rname}: {coeff}"


def test_reserve_dynamic_solves(toy_group_reserve):
    d = _add_increase_reserve_ratio(toy_group_reserve)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options=solver_options())
    assert sol.optimal


def _add_large_failure_ratio(d, ud: str):
    """Turn the timeseries reserve into an *n-1* reserve; ``ud`` picks the
    up (sink-side) or down (source-side) failure branch.  For ``down`` the
    failing flow is ``u``'s consumption from ``FUEL_n``, so ``FUEL_n`` joins
    group ``g`` and carries the ratio."""
    n_fail = "n1" if ud == "up" else "FUEL_n"
    lfr_set = pl.DataFrame(
        {"p": ["u"], "r": ["r1"], "ud": [ud], "n": [n_fail]})
    lfr_param = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": [ud], "n": [n_fail],
                      "value": [0.1]}))
    group_node = (d.group_node if ud == "up" else
                  pl.concat([d.group_node, pl.DataFrame({"g": ["g"], "n": ["FUEL_n"]})]))
    return dataclasses.replace(
        d,
        group_node=group_node,
        reserve_upDown_group=pl.DataFrame(
            {"r": ["r1"], "ud": [ud], "g": ["g"]}),
        reserve_upDown_group_method_timeseries=pl.DataFrame(
            schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8,
                    "method": pl.Utf8}),
        reserve_upDown_group_method_n_1=pl.DataFrame(
            {"r": ["r1"], "ud": [ud], "g": ["g"], "method": ["n_1"]}),
        prundt=d.prundt.with_columns(pl.lit(ud).alias("ud")),
        process_reserve_upDown_node_active=pl.DataFrame(
            {"p": ["u"], "r": ["r1"], "ud": [ud], "n": ["n1"]}),
        p_process_reserve_upDown_node_reliability=Param(("p", "r", "ud", "n"),
            pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": [ud], "n": ["n1"],
                          "value": [1.0]})),
        pdtReserve_upDown_group_reservation=Param(("r", "ud", "g", "d", "t"),
            pl.DataFrame({"r": ["r1"]*2, "ud": [ud]*2, "g": ["g"]*2,
                          "d": ["d1"]*2, "t": ["t01", "t02"],
                          "value": [10.0, 10.0]})),
        p_process_reserve_upDown_node_max_share=Param(("p", "r", "ud", "n"),
            pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": [ud], "n": ["n1"],
                          "value": [1.0]})),
        process_reserve_upDown_node_large_failure_ratio=lfr_set,
        p_process_reserve_upDown_node_large_failure_ratio_value=lfr_param,
    )


@pytest.mark.parametrize("ud,name", [
    ("up", "reserveBalance_up_n_1_eq"),
    ("down", "reserveBalance_down_n_1_eq"),
])
def test_reserve_n_1_canonicalises(toy_group_reserve, ud, name):
    d = _add_large_failure_ratio(toy_group_reserve, ud)
    pb = Problem()
    build_flextool(pb, d)
    m = pb.canonicalise()
    assert name in set(pb.cstr_names())

    # One row per failing process (p_fail = u) and step, with the failing
    # flow's coefficient -(unitsize · ratio) = -(100 · 0.1) = -10, the
    # group total v_reserve_total (1) and minus u's own reserve
    # -(unitsize · reliability) = -100: u is the only provider and fails, so
    # it cannot cover its own loss.
    rows = _row_terms(m, name)
    assert sorted(rows) == [f"{name}[r1,{ud},g,u,d1,t01]", f"{name}[r1,{ud},g,u,d1,t02]"]
    for rname, terms in rows.items():
        t = rname.split(",")[-1].rstrip("]")
        flow_terms = {c: v for c, v in terms.items() if c.startswith("v_flow[")}
        assert len(flow_terms) == 1, f"{rname}: {flow_terms}"
        (coeff,) = flow_terms.values()
        assert coeff == pytest.approx(-10.0, rel=1e-9), f"{rname}: {coeff}"
        reserve_terms = {c: v for c, v in terms.items() if c.startswith("v_reserve[")}
        assert reserve_terms == {f"v_reserve[u,r1,{ud},n1,d1,{t}]": pytest.approx(-100.0, rel=1e-12)}, rname
        assert terms[f"v_reserve_total[r1,{ud},g,d1,{t}]"] == 1.0, rname
