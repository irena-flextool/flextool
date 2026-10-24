"""Dynamic and n-1-down requirements count the flow a unit draws from the node (#23(b)).

At a node a unit consumes from, ``nodeBalance_eq`` subtracts ``v_flow · unitsize · slope`` (plus
``v_online · section · unitsize`` for a min_load_efficiency unit with online variables) on an efficiency arc.
The dynamic requirement (``increase_reserve_ratio`` × the flow between the unit and the node) and the
downward n-1 requirement (the loss of the unit's consumption) now measure that flow the same way, and so do
their shortfall scales (full-load input = capacity × (slope + section)).  Before, both counted the output
``v_flow · unitsize``.

Fixture: ``tests.json`` (``coal`` 500 MW, ``wind`` 1000 MW, ``heat`` −300, ``dr_increase_demand`` 50 MW
consumer at west) plus ``e_boiler``: west → heat, efficiency 0.5, 100 MW.  ``dyn`` (up, dynamic_only): wind's
output (ratio 0.5) and e_boiler's consumption (ratio 1) at west drive it; ``n1d`` (down, large_failure_only):
e_boiler and dr_increase_demand fail (ratio 1).  Coal provides both.  Built from JSON.
"""
from __future__ import annotations

import collections

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, build, solve, unitsize, utf8

pytestmark = pytest.mark.solver

REL = "reserve__upDown__unit__node"
WIND = "wind_plant"
EB = "e_boiler"
DR = "dr_increase_demand"
BASE = PLANT_CHAIN + ["wind", "heat", "dr_increase_demand"]
EB_ENTS = [["unit", EB, None], ["unit__inputNode", [EB, "west"], None], ["unit__outputNode", [EB, "heat"], None]]


def _eb_values(method: str = "constant_efficiency") -> list:
    return [["unit", EB, "efficiency", b64(0.5, "float")],
            ["unit", EB, "existing", b64(100.0, "float")],
            ["unit", EB, "conversion_method", b64(method, "str")]]


def _reserves() -> tuple[list, list]:
    ents: list = []
    vals: list = []
    for r, ud, method in (("dyn", "up", "dynamic_only"), ("n1d", "down", "large_failure_only")):
        grp = [r, ud, "electricity"]
        ents += [["reserve", r, None], ["reserve__upDown__group", grp, None]]
        vals += [["reserve__upDown__group", grp, "reserve_method", b64(method, "str")],
                 ["reserve__upDown__group", grp, "penalty_reserve", b64(1000.0, "float")]]
    for key, params in (((["dyn", "up", WIND, "west"]), {"increase_reserve_ratio": 0.5}),
                        ((["dyn", "up", EB, "west"]), {"increase_reserve_ratio": 1.0}),
                        ((["dyn", "up", PLANT, "west"]), {"max_share": 0.1, "reliability": 1.0}),
                        ((["n1d", "down", EB, "west"]), {"large_failure_ratio": 1.0}),
                        ((["n1d", "down", DR, "west"]), {"large_failure_ratio": 1.0}),
                        ((["n1d", "down", PLANT, "west"]), {"max_share": 1.0, "reliability": 1.0})):
        ents.append([REL, key, None])
        vals.append([REL, key, "is_enabled", b64("yes", "str")])
        vals += [[REL, key, k, b64(v, "float")] for k, v in params.items()]
    return ents, vals


def _row_terms(pb, family: str) -> dict[str, dict[str, float]]:
    """``{row: {column: coefficient}}`` of the canonical rows of ``family``."""
    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(family + "[")}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            ri = m.row_idx[k]
            if ri in rows:
                out[rows[ri]][m.col_names[c]] = m.val[k]
    return out


def _coef(terms: dict[str, float], prefix: str) -> float:
    hit = [v for c, v in terms.items() if c.startswith(prefix)]
    assert len(hit) == 1, (prefix, terms)
    return hit[0]


def _balance(pb) -> dict[str, dict[str, float]]:
    return {r.split(",", 1)[1]: t for r, t in _row_terms(pb, "nodeBalance_eq").items()
            if r.startswith("nodeBalance_eq[west,")}


@pytest.fixture(scope="module")
def eff(tmp_path_factory):
    ents, vals = _reserves()
    return solve(tmp_path_factory, alt="drawn_eff", base_chain=BASE, entities=EB_ENTS + ents,
                 values=_eb_values() + vals)


def test_dynamic_requirement_counts_input_drawn(eff) -> None:
    """e_boiler's v_flow coefficient is −unitsize · slope · ratio (= −200 per unit of v_flow) in the dynamic
    row and in its n-1-down row, the same as in nodeBalance_eq[west] (Δt = 1).  Before: −100."""
    pb = build(eff.flex_data)
    us = unitsize(eff, EB)
    bal = _balance(pb)
    dyn = _row_terms(pb, "reserveBalance_dynamic_eq")
    n1 = {r: t for r, t in _row_terms(pb, "reserveBalance_down_n_1_eq").items() if f",{EB}," in r}
    assert dyn and n1
    for row, terms in dyn.items():
        dt = row.split(",", 3)[3]
        expected = _coef(bal[dt], f"v_flow[{EB},west,heat,")
        assert expected == pytest.approx(-us * 2.0)
        assert _coef(terms, f"v_flow[{EB},west,heat,") == pytest.approx(expected, rel=1e-12), row
    for row, terms in n1.items():
        dt = row.split(",", 4)[4]
        assert _coef(terms, f"v_flow[{EB},west,heat,") == pytest.approx(_coef(bal[dt], f"v_flow[{EB},west,heat,"),
                                                                         rel=1e-12), row


def test_sink_side_flows_unchanged(eff) -> None:
    """Guard: a flow into the node (wind's output) still counts at unitsize × ratio; dr_increase_demand's
    consumption (no efficiency) at unitsize × ratio."""
    pb = build(eff.flex_data)
    for row, terms in _row_terms(pb, "reserveBalance_dynamic_eq").items():
        assert _coef(terms, f"v_flow[{WIND},{WIND},west,") == pytest.approx(-unitsize(eff, WIND) * 0.5), row
    for row, terms in _row_terms(pb, "reserveBalance_down_n_1_eq").items():
        if f",{DR}," in row:
            assert _coef(terms, f"v_flow[{DR},west,{DR},") == pytest.approx(-unitsize(eff, DR)), row


def test_scale_uses_full_load_input(eff) -> None:
    """Shortfall scales: dynamic = wind 1000 · 0.5 + e_boiler 100 · 2 · 1 = 700 MW; n-1 down = max(dr 50,
    e_boiler 200) = 200 MW.  Before: 600 / 100."""
    from flextool.engine.model._reserve import reserve_shortfall_scale

    sc = utf8(reserve_shortfall_scale(eff.flex_data).frame)
    dyn = sc.filter(pl.col("r") == "dyn")["value"]
    n1d = sc.filter(pl.col("r") == "n1d")["value"]
    assert len(dyn) == len(n1d) == eff.flex_data.dt.height
    assert dyn.to_list() == pytest.approx([700.0] * len(dyn))
    assert n1d.to_list() == pytest.approx([200.0] * len(n1d))


def test_section_counts_in_requirement(tmp_path_factory) -> None:
    """A min_load_efficiency e_boiler with linear online variables draws v_online · section · unitsize too:
    the dynamic and n-1-down rows carry v_online_linear[e_boiler] with the coefficient nodeBalance_eq has
    (× ratio 1), and the scales use slope + section."""
    ents, vals = _reserves()
    step = solve(tmp_path_factory, alt="drawn_sec", base_chain=BASE, entities=EB_ENTS + ents,
                 values=_eb_values("min_load_efficiency") + vals + [
                     ["unit", EB, "efficiency_at_min_load", b64(0.4, "float")],
                     ["unit", EB, "min_load", b64(0.3, "float")],
                     ["unit", EB, "startup_method", b64("linear", "str")]])
    pb = build(step.flex_data)
    bal = _balance(pb)
    rows = list(_row_terms(pb, "reserveBalance_dynamic_eq").items())
    rows += [(r, t) for r, t in _row_terms(pb, "reserveBalance_down_n_1_eq").items() if f",{EB}," in r]
    assert rows
    for row, terms in rows:
        dt = ",".join(row.rstrip("]").split(",")[-2:]) + "]"
        sec = _coef(bal[dt], f"v_online_linear[{EB},")
        assert sec < 0.0
        assert _coef(terms, f"v_online_linear[{EB},") == pytest.approx(sec, rel=1e-12), row
        assert _coef(terms, f"v_flow[{EB},west,heat,") == pytest.approx(_coef(bal[dt], f"v_flow[{EB},west,heat,"),
                                                                         rel=1e-12), row
    from flextool.engine.model._reserve import reserve_shortfall_scale

    us = unitsize(step, EB)
    full = -(_coef(next(iter(bal.values())), f"v_flow[{EB},west,heat,"))
    full += -(_coef(next(iter(bal.values())), f"v_online_linear[{EB},"))
    sc = utf8(reserve_shortfall_scale(step.flex_data).frame)
    n1d = sc.filter(pl.col("r") == "n1d")["value"]
    assert full > us * 2.0  # section adds to the full-load input
    assert n1d.to_list() == pytest.approx([full] * len(n1d), rel=1e-9)
