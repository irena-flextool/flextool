"""Emission test — indirect (multi-flow) converter ``maxFlow`` RHS split.

Regression guard for the *indirect free-capacity* bug: INDIRECT process
converters (internal process-node topology, e.g. ``H2 + elec → [proc] →
NH3``) used to take their capacity-enforcing ``maxFlow`` RHS from
``p_flow_upper`` (the ``existing + invest_max`` ceiling).  For a greenfield
converter (``existing = 0``) that made ``v_flow − Σv_invest ≤
invest_max/unitsize`` — satisfied at ``v_invest = 0`` → free (unpaid)
capacity.

The fix binds the indirect **OUTPUT** arc (``source == p``) with the
``maxToSink`` existing-only RHS (``existing/unitsize``, = 0 for greenfield),
so any positive output flow forces paid ``v_invest > 0`` — matching direct
units.  The **input/fuel** arcs (``sink == p``) and any zero-flow-coef aux
arcs keep the loose ``p_flow_upper`` bound (pinning them to 0 would make
fuel hard-infeasible).

Assertions (critique B2 — cover EVERY indirect converter in the fixture,
including the ones with zero-flow-coef input arcs):
  (a) every indirect OUTPUT-arc ``maxFlow`` RHS = 0 (greenfield), and the
      row carries a ``−v_invest`` LHS term;
  (b) every indirect INPUT/fuel arc keeps a loose (> 0) RHS — feasibility;
  (c) every zero-flow-coef aux arc keeps its unconstrained (loose) RHS;
  (d) ``flow_upper_rhs`` row-count parity with ``pss_dt`` (no dropped /
      duplicated rows).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, run_chain_from_db
from flextool.engine_polars._pdt_join import compute_pss_dt

_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

_SCENARIO = "scenario_test_6h_no_carrier_storage"


def _build(db_url: str):
    """Build the h2_trade_parity model and return (flex_data, Problem)."""
    with tempfile.TemporaryDirectory() as wd:
        steps = run_chain_from_db(
            db_url, _SCENARIO,
            work_folder=Path(wd), csv_dump=False, keep_solutions=True,
        )
        last = next(reversed(steps.values()))
        assert last.flex_data is not None
        pb = Problem()
        build_flextool(pb, last.flex_data)
    return last.flex_data, pb


def _maxflow_record(pb: Problem):
    rec = next((r for r in pb.cstrs_named("maxFlow") if r.name == "maxFlow"),
               None)
    assert rec is not None, "maxFlow constraint not emitted"
    return rec


@pytest.mark.emission
def test_indirect_output_arcs_existing_only_rhs(
        h2_trade_parity_db_url: str) -> None:
    """(a) Every indirect greenfield OUTPUT arc gets RHS = 0 and carries a
    ``−v_invest`` LHS term."""
    d, pb = _build(h2_trade_parity_db_url)
    rec = _maxflow_record(pb)
    rhs = rec.proto.rhs.frame  # Param frame (post coef/availability folds)

    out_arcs = d.process_output_flows
    assert out_arcs is not None and out_arcs.height > 0, (
        "fixture must have indirect output arcs")

    # Sanity: the fixture's converters are greenfield (existing = 0), so
    # every output-arc row must be exactly 0 — NOT the invest ceiling.
    out_rows = rhs.join(out_arcs, on=("p", "source", "sink"), how="semi")
    assert out_rows.height > 0
    assert out_rows.get_column("value").max() == 0.0, (
        "indirect OUTPUT-arc maxFlow RHS is non-zero — the greenfield "
        "converter still gets free (unpaid) capacity from the invest "
        "ceiling. Per-arc max: "
        f"{out_rows.sort('value', descending=True).head(3)}")
    assert out_rows.get_column("value").min() == 0.0

    # Every distinct indirect output process must appear in the RHS (no arc
    # silently dropped from flow_upper_rhs).
    out_ps = set(str(x) for x in out_arcs.get_column("p").unique().to_list())
    rhs_out_ps = set(str(x) for x in
                     out_rows.get_column("p").unique().to_list())
    assert out_ps == rhs_out_ps, (
        f"output-arc processes missing from RHS: {out_ps - rhs_out_ps}")

    # (a-ii) the maxFlow LHS carries a ``−v_invest`` term keyed on (p, d)
    # that broadcasts to the output arcs.  The expr has two term families:
    # v_flow over (p, source, sink, d, t) and the invest term over (p, d).
    term_dims = {t.dims for t in rec.proto.expr.terms}
    assert ("p", "d") in term_dims, (
        "maxFlow LHS is missing the (p, d) −v_invest term families "
        f"(got {term_dims}) — greenfield output arcs would not force "
        "paid invest")


@pytest.mark.emission
def test_indirect_input_arcs_stay_loose(
        h2_trade_parity_db_url: str) -> None:
    """(b) Every indirect INPUT/fuel arc keeps a loose (> 0) RHS — pinning
    them to existing (= 0) would make fuel hard-infeasible."""
    d, pb = _build(h2_trade_parity_db_url)
    rec = _maxflow_record(pb)
    rhs = rec.proto.rhs.frame

    in_arcs = d.process_input_flows
    assert in_arcs is not None and in_arcs.height > 0

    in_rows = rhs.join(in_arcs, on=("p", "source", "sink"), how="semi")
    assert in_rows.height > 0
    assert in_rows.get_column("value").min() > 0.0, (
        "an indirect INPUT/fuel arc has RHS <= 0 — fuel would be pinned to "
        "0 (hard infeasibility). Offending rows: "
        f"{in_rows.filter(pl.col('value') <= 0.0)}")

    # No input arc may draw from the existing-only (= 0) slice: assert the
    # input-arc RHS is NOT identically 0 for any process (B1 belt).
    per_proc_max = (in_rows.group_by("p")
                    .agg(pl.col("value").max().alias("mx")))
    zeroed = per_proc_max.filter(pl.col("mx") <= 0.0)
    assert zeroed.height == 0, (
        f"input arcs of these processes were pinned to 0: {zeroed}")


@pytest.mark.emission
def test_indirect_zero_coef_aux_arcs_unconstrained(
        h2_trade_parity_db_url: str) -> None:
    """(c) Zero-flow-coef aux arcs (in neither process_input_flows nor
    process_output_flows) keep their loose ``p_flow_upper`` value."""
    d, pb = _build(h2_trade_parity_db_url)
    rec = _maxflow_record(pb)
    rhs = rec.proto.rhs.frame

    out_arcs = d.process_output_flows
    in_arcs = d.process_input_flows

    # aux arcs = indirect-process arcs that are neither output nor input.
    indir_arcs = rec.over.join(d.process_indirect, on="p", how="semi")
    aux = (indir_arcs
           .join(out_arcs, on=("p", "source", "sink"), how="anti")
           .join(in_arcs, on=("p", "source", "sink"), how="anti")
           .select("p", "source", "sink").unique())
    if aux.height == 0:
        pytest.skip("no zero-coef aux arcs in this fixture")

    aux_rows = rhs.join(aux, on=("p", "source", "sink"), how="semi")
    assert aux_rows.height > 0
    assert aux_rows.get_column("value").min() > 0.0, (
        "a zero-coef aux arc was pinned to 0 instead of keeping its loose "
        f"p_flow_upper value: {aux_rows.filter(pl.col('value') <= 0.0)}")


@pytest.mark.emission
def test_sinkless_indirect_raises(h2_trade_parity_db_url: str) -> None:
    """S1 guard — a sink-less indirect invest unit (no output arc) must
    raise a clear error, never emit silently (which would grant free
    capacity on its un-tightened source arc).

    No fixture authors a legal sink-less indirect topology (≥2 inputs, 0
    outputs → 1way_nvar), so we synthesise the condition by clearing
    ``process_output_flows`` on a real indirect build — every indirect
    process then has zero output arcs and the guard must fire.
    """
    with tempfile.TemporaryDirectory() as wd:
        steps = run_chain_from_db(
            h2_trade_parity_db_url, _SCENARIO,
            work_folder=Path(wd), csv_dump=False, keep_solutions=True,
        )
        d = next(reversed(steps.values())).flex_data
        assert d is not None
        # Drop every output arc → all indirect processes become sink-less.
        d.process_output_flows = d.process_output_flows.head(0)
        pb = Problem()
        with pytest.raises(NotImplementedError, match="without an output"):
            build_flextool(pb, d)


@pytest.mark.solver
def test_indirect_model_stays_feasible(h2_trade_parity_db_url: str) -> None:
    """B2 — the RHS split must not make any indirect converter infeasible.

    The fixture has multiple converters (``*_H2toNH3``, ``*_H2toMeOH``,
    ``*_oretoHBI``, …) including ones with zero-flow-coef input arcs.
    Pinning an input/fuel arc to the existing (= 0) RHS would make fuel
    hard-infeasible; assert the whole model still solves to optimality and
    that no converter carries output flow without paying invest (the
    ``v_invest > 0 ⟺ output flow > 0`` capacity discipline).
    """
    with tempfile.TemporaryDirectory() as wd:
        steps = run_chain_from_db(
            h2_trade_parity_db_url, _SCENARIO,
            work_folder=Path(wd), csv_dump=False, keep_solutions=True,
        )
        d = next(reversed(steps.values())).flex_data
        assert d is not None
        pb = Problem()
        build_flextool(pb, d)
        sol = pb.solve()

    assert sol.optimal, "indirect RHS split made the model infeasible"

    out_arcs = d.process_output_flows
    out_ps = set(str(x) for x in out_arcs.get_column("p").unique().to_list())

    inv = (sol.value("v_invest_p")
           .with_columns(pl.col("p").cast(pl.Utf8).alias("_p")))
    invested = set(
        inv.filter((pl.col("value") > 1e-6)
                   & pl.col("_p").is_in(list(out_ps)))
        .get_column("_p").to_list())

    # Output-arc flow per converter (magnitude, summed over d, t).
    flow = (sol.value("v_flow")
            .with_columns(pl.col("p").cast(pl.Utf8).alias("_p"),
                          pl.col("source").cast(pl.Utf8).alias("_s"),
                          pl.col("sink").cast(pl.Utf8).alias("_k")))
    oa = out_arcs.with_columns(
        pl.col("p").cast(pl.Utf8).alias("_p"),
        pl.col("source").cast(pl.Utf8).alias("_s"),
        pl.col("sink").cast(pl.Utf8).alias("_k"))
    out_flow = (flow.join(oa.select("_p", "_s", "_k"),
                          on=["_p", "_s", "_k"], how="semi")
                .group_by("_p")
                .agg(pl.col("value").abs().sum().alias("f")))
    flowing = set(
        out_flow.filter(pl.col("f") > 1e-6).get_column("_p").to_list())

    # Capacity discipline: any converter that carries OUTPUT flow must have
    # paid for capacity (v_invest > 0), since existing = 0 → RHS = 0.
    free_riders = flowing - invested
    assert not free_riders, (
        "greenfield indirect converters carry output flow WITHOUT paid "
        f"invest (free capacity leaked): {sorted(free_riders)}")


@pytest.mark.emission
def test_indirect_rhs_row_count_parity(
        h2_trade_parity_db_url: str) -> None:
    """(d) ``flow_upper_rhs`` covers exactly ``pss_dt`` — no dropped or
    duplicated rows introduced by the RHS split."""
    d, pb = _build(h2_trade_parity_db_url)
    rec = _maxflow_record(pb)
    rhs = rec.proto.rhs.frame

    pss_dt = compute_pss_dt(d)
    assert rec.over.height == pss_dt.height, (
        f"maxFlow over-height {rec.over.height} != |pss_dt| {pss_dt.height}")
    # RHS frame must have exactly one value per (p, source, sink, d, t) and
    # cover every over-row.
    assert rhs.height == rec.over.height, (
        f"RHS frame height {rhs.height} != over-height {rec.over.height}")
    uniq = rhs.select("p", "source", "sink", "d", "t").unique().height
    assert uniq == rhs.height, (
        f"RHS frame has duplicated keys: {rhs.height - uniq} dupes")
