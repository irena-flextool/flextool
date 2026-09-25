"""Slice D — Phase D2: invest-axis branch expansion + per-period anchor map.

Design: ``specs/sliceD_recourse_invest_design.md`` §3 (A) / §4 (B) / §18 D2.
Solver-free unit coverage of the two mechanisms; the full flag-on cascade
objective gate lands at D5.
"""
from __future__ import annotations

import polars as pl
import pytest

import flextool.engine_polars._derived_params as dp
from flextool.engine_polars._derived_params import (
    _anchor_expand_explicit,
    _build_recourse_anchor_pairs,
    _expand_invest_branch_periods,
)
from flextool.engine_polars._emit_solve_writers import (
    emit_stochastic_invest_method,
)


@pytest.fixture(autouse=True)
def _reset_anchor_scope():
    """Never leak the module-global recourse holders to another test."""
    dp._RECOURSE_ANCHOR_PAIRS = None
    dp._RECOURSE_WALK_PROVIDER = None
    yield
    dp._RECOURSE_ANCHOR_PAIRS = None
    dp._RECOURSE_WALK_PROVIDER = None


def _emit(prov, key, cols, rows):
    from flextool.engine_polars._emit_provider_io import _emit as _e
    _e(prov, key, pl.DataFrame({c: [r[i] for r in rows]
                                for i, c in enumerate(cols)}))


def _provider(tmp_path, *, flag: str, pb, piu):
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    prov = FlexDataProvider()
    emit_stochastic_invest_method(
        flag, str(tmp_path / "solve_data/stochastic_invest_method.csv"),
        provider=prov)
    _emit(prov, "solve_data/period__branch.csv", ("period", "branch"), pb)
    _emit(prov, "solve_data/period_in_use_set.csv", ("period",),
          [(p,) for p in piu])
    return prov


PB = [
    ("p2035", "p2035"), ("p2035", "p2035_rlz"), ("p2035", "p2035_low"),
    ("p2040", "p2040"), ("p2040", "p2040_rlz"), ("p2040", "p2040_low"),
]
PIU = ["p2035", "p2035_low", "p2040", "p2040_low"]


# ---------------------------------------------------------------------------
# (A) invest-side branch expansion
# ---------------------------------------------------------------------------


def test_expand_invest_axis_recourse(tmp_path):
    prov = _provider(tmp_path, flag="recourse", pb=PB, piu=PIU)
    out = _expand_invest_branch_periods(
        ["p2035", "p2040"], tmp_path, provider=prov)
    assert out == ["p2035", "p2035_low", "p2040", "p2040_low"]


def test_expand_invest_axis_flag_off_noop(tmp_path):
    prov = _provider(tmp_path, flag="none", pb=PB, piu=PIU)
    out = _expand_invest_branch_periods(
        ["p2035", "p2040"], tmp_path, provider=prov)
    assert out == ["p2035", "p2040"]


def test_expand_invest_axis_empty_passthrough(tmp_path):
    prov = _provider(tmp_path, flag="recourse", pb=PB, piu=PIU)
    assert _expand_invest_branch_periods([], tmp_path, provider=prov) == []
    assert _expand_invest_branch_periods(None, tmp_path, provider=prov) is None


# ---------------------------------------------------------------------------
# (B) anchor pairs + explicit expansion
# ---------------------------------------------------------------------------


def test_anchor_pairs_recourse(tmp_path):
    prov = _provider(tmp_path, flag="recourse", pb=PB, piu=PIU)
    pairs = _build_recourse_anchor_pairs(tmp_path, provider=prov)
    got = {(r["anchor"], r["br"]) for r in pairs.iter_rows(named=True)}
    # only in-use synthetic branches (p2035_rlz not in PIU → excluded).
    assert got == {("p2035", "p2035_low"), ("p2040", "p2040_low")}


def test_anchor_pairs_flag_off_none(tmp_path):
    prov = _provider(tmp_path, flag="none", pb=PB, piu=PIU)
    assert _build_recourse_anchor_pairs(tmp_path, provider=prov) is None


def test_anchor_expand_inherits_branch_rows(tmp_path):
    prov = _provider(tmp_path, flag="recourse", pb=PB, piu=PIU)
    dp._enter_recourse_anchor_scope(tmp_path, provider=prov)
    explicit = pl.LazyFrame({
        "e": ["peaker", "peaker"],
        "d": ["p2035", "p2040"],
        "value": [7.0, 9.0],
    })
    out = _anchor_expand_explicit(explicit).collect().sort("e", "d")
    got = {(r["d"], r["value"]) for r in out.iter_rows(named=True)}
    assert got == {
        ("p2035", 7.0), ("p2035_low", 7.0),
        ("p2040", 9.0), ("p2040_low", 9.0),
    }


def test_anchor_expand_noop_when_scope_none(tmp_path):
    prov = _provider(tmp_path, flag="none", pb=PB, piu=PIU)
    dp._enter_recourse_anchor_scope(tmp_path, provider=prov)  # sets None
    explicit = pl.LazyFrame({"e": ["peaker"], "d": ["p2035"], "value": [7.0]})
    out = _anchor_expand_explicit(explicit).collect()
    assert out.height == 1 and out["d"].to_list() == ["p2035"]
