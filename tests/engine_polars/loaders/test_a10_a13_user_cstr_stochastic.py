"""Surface A.10 (user-defined flow / investment constraint loader) and
A.13 (stochastic-branching loader) tests.

A.10 covers ``flextool.engine_polars.input._load_user_constraints`` —
the helper that turns ``constraint__sense.csv`` +
``p_process_node_constraint_flow_coeff.csv`` into the (cdt_eq,
cdt_le, cdt_ge) constraint-axis triple plus the ``flow_cstr_idx``
index frame.  Tests pin the blank-shortcut gates, the dt-cardinality
cross join and the source/sink dedup that drives ``flow_cstr_idx``.

A.13 covers ``flextool.engine_polars.input._load_stochastics`` — the
helper that reads four solve_data CSVs (pdt/pd branch weights,
non-anticipativity timesteps, ``period_in_use_set``) plus the
``groupIncludeStochastics`` input.  Tests pin the null-fill / empty-
file fallbacks and the canonical ``g`` rename.

Both helpers are pure: hand-built ``Path`` directories + small CSV
overlays isolate one transformation per test (mirrors the A.8 pattern).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars.input import _load_user_constraints, _load_stochastics


# --- helpers --------------------------------------------------------

def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    return inp, sd


# ====================================================================
# A.10 — _load_user_constraints
# ====================================================================

def test_user_constraints_blank_shortcut_gates(tmp_path: Path):
    """Covers A10-blank_when_pss_none + A10-blank_when_constraint_sense_missing
    + A10-blank_when_constraint_sense_empty.

    Three blank-shortcut paths share one fixture; each must yield the
    canonical ``[None]*12`` so no user-cstr family fires.
    """
    inp, _ = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})

    # (1) pss=None: earliest gate (no disk read at all).
    out1 = _load_user_constraints(inp, pss=None, dt=dt)
    # Hand-calc: pss is None -> [None]*12.
    assert out1 == [None] * 12

    pss = pl.DataFrame({"p": ["p1"], "source": ["n_src"], "sink": ["n_snk"]})

    # (2) pss populated but constraint__sense.csv absent.
    out2 = _load_user_constraints(inp, pss=pss, dt=dt)
    # Hand-calc: cs_path.exists() False -> [None]*12.
    assert out2 == [None] * 12

    # (3) constraint__sense.csv present but header-only (height=0).
    _write(inp / "constraint__sense.csv", "constraint,sense\n")
    out3 = _load_user_constraints(inp, pss=pss, dt=dt)
    # Hand-calc: cs.height==0 -> [None]*12 (distinct branch from (2)).
    assert out3 == [None] * 12


def test_cdt_cross_with_dt_cardinality(tmp_path: Path):
    """Covers A10-cdt_cross_with_dt_cardinality.

    Single (c1, equal) sense row crossed against 3 dt rows -> cdt_eq
    height=3, cdt_le and cdt_ge stay None (no rows of their senses).
    """
    inp, _ = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1", "d1", "d2"], "t": ["t1", "t2", "t1"]})
    pss = pl.DataFrame({"p": ["p1"], "source": ["n_src"], "sink": ["n_snk"]})
    _write(inp / "constraint__sense.csv", "constraint,sense\nc1,equal\n")

    out = _load_user_constraints(inp, pss=pss, dt=dt)
    # Hand-calc: cs filtered to sense=='equal' -> 1 row (c1); cross-join
    # with 3 dt rows -> height=3.  No le/ge sense rows -> those slots None.
    cdt_eq, cdt_le, cdt_ge = out[3], out[4], out[5]
    assert cdt_eq is not None and cdt_eq.height == 3
    assert set(cdt_eq.columns) == {"cn", "d", "t"}
    assert sorted(cdt_eq.rows()) == sorted(
        [("c1", "d1", "t1"), ("c1", "d1", "t2"), ("c1", "d2", "t1")])
    assert cdt_le is None and cdt_ge is None
    # has_user_cstr is True once the live path runs.
    assert out[11] is True


def test_flow_cstr_idx_source_plus_sink_dedup(tmp_path: Path):
    """Covers A10-flow_cstr_idx_source_plus_sink_sum.

    coef rows (p1,n_x,c1,2.0) and (p1,n_y,c1,3.0) with a single pss row
    (p1,n_x,n_y) — n_x matches as source, n_y matches as sink.  After
    concat + group_by, ``flow_cstr_idx`` emerges only ONCE despite two
    contributing legs (the coefficient sum lives downstream in the
    Δ.12 override frame).
    """
    inp, _ = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss = pl.DataFrame({"p": ["p1"], "source": ["n_x"], "sink": ["n_y"]})
    _write(inp / "constraint__sense.csv", "constraint,sense\nc1,equal\n")
    _write(inp / "p_process_node_constraint_flow_coeff.csv",
           "process,node,constraint,p_process_node_constraint_flow_coeff\n"
           "p1,n_x,c1,2.0\n"
           "p1,n_y,c1,3.0\n")

    out = _load_user_constraints(inp, pss=pss, dt=dt)
    # Hand-calc: src_match -> (p1,n_x,n_y,c1,2.0); sink_match ->
    # (p1,n_x,n_y,c1,3.0); concat+group_by on (p,source,sink,c) collapses
    # the two legs into ONE index row.  flow_cstr_coef stays None
    # (Δ.12-drop, owned by apply_derived_b).
    fci = out[0]
    assert fci is not None and fci.height == 1
    assert fci.columns == ["p", "source", "sink", "cn"]
    assert fci.row(0) == ("p1", "n_x", "n_y", "c1")
    assert out[1] is None  # flow_cstr_coef


def test_flow_cstr_idx_via_provider_utf8_coef_column(tmp_path: Path):
    """Regression: A2 / Rivendell-2 — ``p_process_node_constraint_flow_coeff``
    arrives from the Provider with an all-Utf8 schema (the canonical
    ``_rows_to_frame`` output of :class:`SpineDBBackend`).  The loader
    must cast ``coef`` to Float64 before ``group_by(...).agg(coef.sum())``
    or polars raises ``InvalidOperationError: 'sum' operation not supported
    for dtype 'str'`` (input.py:1574).

    Reproduces the cascade-path schema by seeding a ``FlexDataProvider``
    directly — disk CSV via ``read_csv_fallback`` parses ``"2.0"`` as
    Float64 and masks the bug.
    """
    from flextool.engine_polars._flex_data_provider import FlexDataProvider

    inp, _ = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss = pl.DataFrame({"p": ["p1"], "source": ["n_x"], "sink": ["n_y"]})
    _write(inp / "constraint__sense.csv", "constraint,sense\nc1,equal\n")

    # All-Utf8 schema as the SpineDBBackend / _rows_to_frame produces.
    coef_frame = pl.DataFrame(
        {
            "process": ["p1", "p1"],
            "node": ["n_x", "n_y"],
            "constraint": ["c1", "c1"],
            "p_process_node_constraint_flow_coeff": ["2.0", "3.0"],
        },
        schema={
            "process": pl.Utf8,
            "node": pl.Utf8,
            "constraint": pl.Utf8,
            "p_process_node_constraint_flow_coeff": pl.Utf8,
        },
    )
    provider = FlexDataProvider()
    provider.put("input/p_process_node_constraint_flow_coeff", coef_frame)

    out = _load_user_constraints(inp, pss=pss, dt=dt, provider=provider)
    # Hand-calc: with the cast in place, group_by on (p,source,sink,c)
    # collapses src_match + sink_match to a single index row.
    fci = out[0]
    assert fci is not None and fci.height == 1
    assert fci.columns == ["p", "source", "sink", "cn"]
    assert fci.row(0) == ("p1", "n_x", "n_y", "c1")


def test_flow_cstr_idx_none_when_no_match(tmp_path: Path):
    """Covers A10-flow_cstr_idx_none_when_no_match.

    coef row (p1,n_other,c1,2.0) cannot join either side of pss
    (p1,n_src,n_snk).  src_match.height + sink_match.height == 0 ->
    short-circuit before the concat, ``flow_cstr_idx`` stays None.
    """
    inp, _ = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    pss = pl.DataFrame({"p": ["p1"], "source": ["n_src"], "sink": ["n_snk"]})
    _write(inp / "constraint__sense.csv", "constraint,sense\nc1,equal\n")
    _write(inp / "p_process_node_constraint_flow_coeff.csv",
           "process,node,constraint,p_process_node_constraint_flow_coeff\n"
           "p1,n_other,c1,2.0\n")

    out = _load_user_constraints(inp, pss=pss, dt=dt)
    # Hand-calc: n_other not in {n_src, n_snk} -> both joins height=0 ->
    # gate `src_match.height + sink_match.height > 0` is False ->
    # flow_cstr_idx stays at its initial None.
    assert out[0] is None
    # cdt_eq still emitted (sense row drove it).
    assert out[3] is not None and out[3].height == 1


# ====================================================================
# A.13 — _load_stochastics
# ====================================================================

def test_branch_weights_null_fill_and_empty_csv(tmp_path: Path):
    """Covers A13-pdt_branch_weight_null_value_fills_to_one
    + A13-pd_branch_weight_none_when_csv_empty.

    Two CSV-fallback gates in one fixture:
      * pdt_branch_weight.csv has one row (d1,t1) with NULL value ->
        cast(strict=False) yields null, fill_null(1.0) restores it; the
        dt-base join keeps that 1.0 at (d1,t1).
      * pd_branch_weight.csv present but header-only -> df.height==0
        gate -> Param stays None.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    # Empty value cell — Polars CSV reader parses as null.
    _write(sd / "pdt_branch_weight.csv",
           "period,time,value\nd1,t1,\n")
    _write(sd / "pd_branch_weight.csv", "period,value\n")

    out = _load_stochastics(inp, sd, dt)
    # Hand-calc: null value -> fill_null(1.0) -> 1.0; left-join onto
    # base (d1,t1)=1.0 -> coalesce(value__r=1.0, value=1.0) -> 1.0.
    pdt = out["pdt_branch_weight"]
    assert pdt is not None
    assert pdt.dims == ("d", "t")
    fr = pdt.frame
    assert fr.height == 1
    assert fr["value"][0] == pytest.approx(1.0, rel=1e-7)
    # Hand-calc: pd csv height==0 -> the `if df.height > 0` gate skips
    # the wrap, pd_branch_weight stays at its initial None.
    assert out["pd_branch_weight"] is None


def test_period_in_use_set_none_when_no_period_column(tmp_path: Path):
    """Covers A13-period_in_use_set_none_when_no_period_column.

    period_in_use_set.csv has a row but lacks the ``period`` column —
    the schema-guard ``"period" in df.columns`` skips the rename rather
    than KeyError-ing the loader.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    # 1 row, only `solve` column — no `period`.
    _write(sd / "period_in_use_set.csv", "solve\nsolve1\n")

    out = _load_stochastics(inp, sd, dt)
    # Hand-calc: df.height>0 True, but "period" not in columns -> gate
    # skips, period_in_use_set stays at its initial None.
    assert out["period_in_use_set"] is None


def test_groupStochastic_renames_first_column_to_g(tmp_path: Path):
    """Covers A13-groupStochastic_renames_first_column_to_g.

    groupIncludeStochastics.csv with header ``group`` and 2 rows ->
    df.rename({df.columns[0]: "g"}) + select("g").unique() yields a
    canonical (g,) frame regardless of source header spelling.
    """
    inp, sd = _make_dirs(tmp_path)
    dt = pl.DataFrame({"d": ["d1"], "t": ["t1"]})
    _write(inp / "groupIncludeStochastics.csv", "group\ngrp1\ngrp2\n")

    out = _load_stochastics(inp, sd, dt)
    # Hand-calc: rename group->g, unique() over 2 distinct rows -> height=2.
    gs = out["groupStochastic"]
    assert gs is not None
    assert gs.columns == ["g"]
    assert gs.height == 2
    assert sorted(gs["g"].to_list()) == ["grp1", "grp2"]
