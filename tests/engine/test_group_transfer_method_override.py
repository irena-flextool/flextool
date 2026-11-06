"""A group's ``transfer_method`` overrides the transfer method of its connections on every path.

docs/reference.md: a group ``transfer_method`` other than ``use_connection_transfer_methods`` applies to the
connections with both endpoints in the group (``dc_power_flow_with_angles``: the DC power-flow lines, one
signed flow each).  The input derivation applied it (``derived/ct_method_overrides`` →
``input/process_method``), but the engine's own classifier (``_derived._classify_process_method``) read
``connection.transfer_method`` only, so the DB-path topology ignored it: every case14 DC line kept a second
``v_flow`` arc in the reverse direction that is in ``nodeBalance_eq`` and ``maxFlow`` but not in
``dc_flow_eq`` (a lossless path around the DC power-flow physics), and a group ``unidirectional`` override
left the connection two-way.

Fixtures (JSON, never a checked-in .sqlite): ``case14_dc_power_flow.json`` ``dc_opf_test``; ``tests.json``
network chain with a group ``gwe`` over west and east overriding west_east (regular) to ``unidirectional``
and to ``exact``.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db, run, utf8

pytestmark = pytest.mark.solver

NET = ["init", "west", "coal", "wind", "network"]
CASE14 = FLEXTOOL_ROOT / "tests" / "fixtures" / "case14_dc_power_flow.json"


def _group(method: str) -> tuple[list, list]:
    ents = [["group", "gwe", None], ["group__node", ["gwe", "west"], None],
            ["group__node", ["gwe", "east"], None]]
    return ents, [["group", "gwe", "transfer_method", b64(method, "str")]]


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict:
    out = {}
    url, scen = make_db(tmp_path_factory, alt="gto_case14", values=[], base_chain=["base"], fixture_json=CASE14)
    out["case14"] = (url, scen, list(run(tmp_path_factory, url, scen).values())[-1])
    for method in ("unidirectional", "exact"):
        ents, vals = _group(method)
        url, scen = make_db(tmp_path_factory, alt=f"gto_{method}", values=vals, base_chain=NET, entities=ents)
        out[method] = (url, scen, list(run(tmp_path_factory, url, scen).values())[-1])
    return out


def _connection_methods(step, url: str, scenario: str) -> tuple[dict, dict]:
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.preprocess.params._derived import _classify_process_method

    cls = _classify_process_method(SpineDbReader(url, scenario))
    engine = dict(utf8(cls.filter(pl.col("klass") == "connection").select("p", "method")).iter_rows())
    pm = utf8(step.flex_data_provider.get("input/process_method"))
    derived = {p: m for p, m in pm.select("process", "method").iter_rows() if p in engine}
    return engine, derived


@pytest.mark.parametrize("case", ["case14", "unidirectional", "exact"])
def test_classifier_applies_group_override(cases, case) -> None:
    """The engine classifier's connection methods equal the input derivation's override-aware
    ``input/process_method``."""
    url, scen, step = cases[case]
    engine, derived = _connection_methods(step, url, scen)
    assert engine == derived
    expected = {"case14": "method_2way_1var_off", "unidirectional": "method_1way_1var_off",
                "exact": "method_2way_2var_MIP_exclude"}[case]
    conn = "line_10_11" if case == "case14" else "west_east"
    assert engine[conn] == expected


def test_dc_lines_have_one_arc(cases) -> None:
    """Every DC power-flow line has exactly one v_flow arc (its reverse flow is v_flow_back, tied to the
    angles by dc_flow_eq)."""
    _url, _scen, step = cases["case14"]
    pss = utf8(step.flex_data.process_source_sink)
    lines = pss.filter(pl.col("p").str.starts_with("line_"))
    assert lines.height > 0
    assert lines.group_by("p").len()["len"].to_list() == [1] * lines["p"].n_unique()
    back = utf8(step.flex_data.process_source_sink_2way_1var)
    assert set(back["p"].to_list()) == set(lines["p"].to_list())


def test_unidirectional_override_is_one_way(cases) -> None:
    _url, _scen, step = cases["unidirectional"]
    pss = utf8(step.flex_data.process_source_sink).filter(pl.col("p") == "west_east")
    assert pss.select("source", "sink").rows() == [("west", "east")]


def _two_groups(m1: str, m2: str) -> tuple[list, list]:
    ents, vals = [], []
    for g, m in (("gwe", m1), ("gwe2", m2)):
        ents += [["group", g, None], ["group__node", [g, "west"], None], ["group__node", [g, "east"], None]]
        vals.append(["group", g, "transfer_method", b64(m, "str")])
    return ents, vals


def test_conflicting_group_overrides_raise(tmp_path_factory) -> None:
    """Two groups over west and east with different overrides: both the input derivation (the run) and the engine
    classifier raise, naming the connection and the groups."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.preprocess.params._derived import _classify_process_method

    ents, vals = _two_groups("unidirectional", "exact")
    url, scen = make_db(tmp_path_factory, alt="gto_conflict", values=vals, base_chain=NET, entities=ents)
    with pytest.raises(FlexToolConfigError, match=r"west_east \(gwe: unidirectional, gwe2: exact\)"):
        _classify_process_method(SpineDbReader(url, scen))
    with pytest.raises(FlexToolConfigError, match="west_east"):
        run(tmp_path_factory, url, scen)


def test_equal_group_overrides_agree(tmp_path_factory) -> None:
    """Two groups with the same override are no conflict: one arc, as with a single group."""
    ents, vals = _two_groups("unidirectional", "unidirectional")
    url, scen = make_db(tmp_path_factory, alt="gto_same", values=vals, base_chain=NET, entities=ents)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    engine, derived = _connection_methods(step, url, scen)
    assert engine == derived and engine["west_east"] == "method_1way_1var_off"
