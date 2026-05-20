"""Δ.25 — focused tests for the surgical fast single-solve path.

Validates that
:func:`flextool.engine_polars.run_single_solve_from_db` produces the
same objective as the slow path
(:func:`flextool.engine_polars.run_chain_from_db`) on the
``work_base`` single-solve fixture.

Scope is intentionally narrow: this is the experimental fast path the
user flagged for ``test_24h_shipping``-style cold-start latency, NOT
production parity coverage.  The full parity sweep lives in
:mod:`test_orchestration_parity`.

Per the Δ.25 design (non-production, raise loudly), additional
fixtures will surface helper coverage gaps as
:class:`flextool.engine_polars.FastLoadError` — those are documented
in the Δ.25 close stanza, not bolted on here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import (
    FastLoadError,
    run_single_solve_from_db,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# work_base — the canonical simple single-solve fixture.
# ---------------------------------------------------------------------------


def test_fast_single_solve_work_base_obj_parity(tmp_path: Path) -> None:
    """``work_base`` solves with the same objective on the fast path
    as on the slow path.

    The slow-path objective on ``work_base`` is ``4780167750`` (see
    ``progress.md`` Δ.16+).  We accept any rel-error < 1e-9 since the
    LP construction is identical from a HiGHS perspective.
    """
    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast"
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="base",
        work_folder=work,
    )

    assert step.solution is not None, "fast path returned no Solution"
    assert step.solution.optimal, (
        f"fast path: HiGHS non-optimal "
        f"(status={getattr(step.solution, 'status', None)})"
    )
    expected_obj = 4780167750.0
    rel_err = abs(step.obj - expected_obj) / expected_obj
    assert rel_err < 1e-9, (
        f"fast path obj={step.obj} differs from expected "
        f"{expected_obj} by rel_err={rel_err:.3e}"
    )

    # Output writer adapter ran — output_raw should exist.
    assert (work / "output_raw").exists(), (
        "expected output_raw/ directory produced by the writer adapter"
    )


def test_fast_single_solve_requires_scenario_name(tmp_path: Path) -> None:
    """Fast path raises when no scenario_name supplied.

    The fast path doesn't auto-pick scenarios — scenario resolution
    is a slow-path-only convenience.  Verify the requirement is loud.
    """
    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast"
    # SpineDbReader requires a scenario string; an empty string is the
    # closest stand-in for "missing".  Either raises some flavour of
    # error inside spinedb_api or our own FastLoadError downstream.
    with pytest.raises((Exception,)):
        run_single_solve_from_db(
            f"sqlite:///{db}",
            scenario_name="",
            work_folder=work,
        )


def test_fast_single_solve_dump_csvs_populates_workdir(
    tmp_path: Path, caplog
) -> None:
    """Δ.30 — the fast path calls ``data.dump_csvs(work_folder)`` so
    ``input/`` and ``solve_data/`` carry the FlexData-derived CSVs that
    handoff_writers + read_parameters consume.

    Acceptance:
    1. ``input/`` contains the four entity-class set CSVs and the
       wide-format ``p_entity_unitsize.csv``.
    2. ``solve_data/`` carries the header-only
       ``solve__p_entity_pre_existing.csv`` stub.
    3. No ``[Errno 2]`` log records for the input/handoff CSVs the
       writers consume.
    4. ``v_dual_node_balance__base.parquet`` is non-empty (Δ.30 piece 1).
    """
    import logging

    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_dump"
    with caplog.at_level(logging.WARNING):
        step = run_single_solve_from_db(
            f"sqlite:///{db}",
            scenario_name="base",
            work_folder=work,
        )
    assert step.solution is not None and step.solution.optimal

    # 1. input/ entity-class set CSVs + wide-format unitsize.
    input_dir = work / "input"
    for name in ("entity.csv", "node.csv", "process_unit.csv",
                  "process_connection.csv", "p_entity_unitsize.csv"):
        assert (input_dir / name).exists(), (
            f"missing input/{name} after fast-path dump_csvs"
        )

    # 2. solve_data/ pre-existing stub.
    assert (work / "solve_data" / "solve__p_entity_pre_existing.csv").exists()

    # 3. No [Errno 2] warnings for the previously-broken inputs.
    forbidden_paths = (
        "input/p_entity_unitsize.csv",
        "solve__p_entity_pre_existing.csv",
    )
    bad_records = [
        rec.message for rec in caplog.records
        if "[Errno 2]" in rec.message
        and any(p in rec.message for p in forbidden_paths)
    ]
    assert not bad_records, (
        f"unexpected [Errno 2] warnings for handoff CSVs: {bad_records}"
    )

    # 4. v_dual_node_balance non-empty (piece 1 fix).
    parquet = work / "output_raw" / "v_dual_node_balance__base.parquet"
    assert parquet.exists()
    from flextool.lean_parquet import read_lean_parquet
    df = read_lean_parquet(parquet)
    assert not df.empty, "v_dual_node_balance parquet is empty"
    assert (df != 0).any().any(), "v_dual_node_balance has only zeros"


def test_fast_single_solve_skip_output_emit(tmp_path: Path) -> None:
    """``emit_output=False`` short-circuits the writer adapter.

    Useful for benchmarking the LP-build path in isolation; verify
    no output_raw parquets appear when disabled.
    """
    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_no_emit"
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="base",
        work_folder=work,
        emit_output=False,
    )
    assert step.solution is not None and step.solution.optimal

    output_raw = work / "output_raw"
    if output_raw.exists():
        # Created at workdir-bootstrap time (mkdir -p) but should be
        # empty when emit_output=False.
        contents = list(output_raw.iterdir())
        assert not contents, (
            f"expected empty output_raw/ when emit_output=False; "
            f"got {[p.name for p in contents]}"
        )


# ---------------------------------------------------------------------------
# Δ.28 — broadcast-needing Direct Params populate on the fast path.
# ---------------------------------------------------------------------------


def test_fast_single_solve_p_commodity_price_lh2(tmp_path: Path) -> None:
    """Δ.28 — ``p_commodity_price`` must populate on the fast path for
    ``work_lh2_three_region``.

    Spine carries ``commodity.price=30`` (scalar) for ``coal``.
    Pre-Δ.28 the dispatcher ran ``apply_direct_params`` BEFORE
    ``apply_derived_a`` populated ``flex_data.dt`` — so the broadcast
    helper saw an empty ``period_filter`` and returned None.  After
    Δ.28 splits the pass into 1a (dt-independent) / 1b
    (broadcast-needing, post-derived_a), the helper sees a populated
    ``dt`` and broadcasts the scalar across (d, t).
    """
    from flextool.engine_polars._fast_load import load_flextool_source_only
    from flextool.engine_polars import SpineDbReader

    fixture = DATA / "work_lh2_three_region"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_lh2"
    work.mkdir()
    reader = SpineDbReader(db, scenario="lh2_three_region")
    fd = load_flextool_source_only(reader, work)

    assert fd.p_commodity_price is not None, (
        "fast path: p_commodity_price is None — Δ.28 dt-ordering bug "
        "regression?  Spine carries commodity.price=30 (scalar) for "
        "coal on this fixture."
    )
    # Phase E.1: scalar source stays (c,) — one row per commodity.
    # polar_high broadcasts to (c, d, t) lazily at constraint emission.
    assert fd.p_commodity_price.dims == ("c",), (
        f"fast path: p_commodity_price dims={fd.p_commodity_price.dims}, "
        "expected (c,) for scalar commodity.price=30."
    )
    assert fd.p_commodity_price.frame.height == 1, (
        f"fast path: p_commodity_price height={fd.p_commodity_price.frame.height}, "
        "expected 1 (single commodity scalar) under Phase E.1."
    )
    assert (fd.p_commodity_price.frame["value"]
            == 30.0).all(), (
        "fast path: every p_commodity_price row should carry the "
        "scalar 30.0 broadcast value."
    )


def test_fast_single_solve_p_process_availability_lh2(tmp_path: Path) -> None:
    """Δ.28 — ``p_process_availability`` must populate on the fast path
    for ``work_lh2_three_region``.

    Spine carries no explicit ``unit/connection.availability`` rows
    on this fixture; every entity inherits the schema default 1.0.
    Pre-Δ.28 the resolver's ``_try_parameter_frame`` returned the
    empty explicit-only frame and the helper produced ``None``.  After
    Δ.28's fall-through to ``parameter()`` (which applies the schema
    default) the helper broadcasts 1.0 to every (entity, d, t) — the
    same flat frame the slow path's ``pdtProcess_availability.csv``
    encodes.
    """
    from flextool.engine_polars._fast_load import load_flextool_source_only
    from flextool.engine_polars import SpineDbReader

    fixture = DATA / "work_lh2_three_region"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_lh2"
    work.mkdir()
    reader = SpineDbReader(db, scenario="lh2_three_region")
    fd = load_flextool_source_only(reader, work)

    assert fd.p_process_availability is not None, (
        "fast path: p_process_availability is None — Δ.28 "
        "default-broadcast fall-through regression?"
    )
    # Phase E.1: scalar-default source stays (p,) — one row per process
    # (15 unit + 5 connection = 20).  polar_high broadcasts to (p, d, t)
    # lazily at constraint emission.
    assert fd.p_process_availability.dims == ("p",), (
        f"fast path: p_process_availability dims="
        f"{fd.p_process_availability.dims}, expected (p,) for scalar default."
    )
    assert fd.p_process_availability.frame.height == 20, (
        f"fast path: p_process_availability height="
        f"{fd.p_process_availability.frame.height}, expected 20 "
        "(20 processes, scalar default) under Phase E.1."
    )


def test_fast_single_solve_process_profile_upper_lh2(tmp_path: Path) -> None:
    """Δ.29 — ``process_profile_upper`` must populate on the fast path
    for ``work_lh2_three_region``.

    The lh2 fixture has 3 wind units (wind_A/B/C) with
    ``unit__node__profile.profile_method = upper_limit`` against their
    respective elec_X output nodes.  Without this set on the fast path,
    the LP's ``profile_flow_upper_limit`` constraint family is
    unconstructed (model.py:2311 gates on ``height > 0``) — so the LP
    treats wind capacity as unlimited and satisfies all demand from
    "free" wind, driving obj to 0.  With this set populated the LP
    forces coal usage and obj > 0.

    Δ.28 close stanza diagnosed this as the load-bearing residual gap
    on lh2; Δ.29 wires the existing ``_projection_params`` helper into
    ``apply_projection_params``.
    """
    from flextool.engine_polars._fast_load import load_flextool_source_only
    from flextool.engine_polars import SpineDbReader

    fixture = DATA / "work_lh2_three_region"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_lh2"
    work.mkdir()
    reader = SpineDbReader(db, scenario="lh2_three_region")
    fd = load_flextool_source_only(reader, work)

    assert fd.process_profile_upper is not None, (
        "fast path: process_profile_upper is None — Δ.29 wiring "
        "regression?  Spine carries 3 unit__node__profile rows for "
        "wind_{A,B,C} on this fixture with profile_method=upper_limit."
    )
    assert fd.process_profile_upper.height == 3, (
        f"fast path: process_profile_upper height="
        f"{fd.process_profile_upper.height}, expected 3 "
        "(wind_A/B/C → elec_A/B/C upper-limit profiles)."
    )
    # Schema parity with the slow path's _load_profiles output.
    assert set(fd.process_profile_upper.columns) == {
        "p", "source", "sink", "f"}, (
        f"fast path: process_profile_upper schema={fd.process_profile_upper.columns}, "
        "expected {p, source, sink, f}."
    )


def test_fast_single_solve_p_node_capacity_for_scaling_lh2(tmp_path: Path) -> None:
    """Δ.29 — ``p_node_capacity_for_scaling`` must default to 1.0 across
    ``(nodeBalance, period_in_use)`` on the fast path.

    The slow path's :func:`_load_node_capacity_for_scaling` reads
    ``solve_data/node_capacity_for_scaling.csv`` and inner-joins with
    nodeBalance.  Without a fast-path source helper, the field stayed
    None, leaving the LP's slack-penalty objective term without the
    row-scaling factor — when the pow10 cascade is active this makes
    penalty a free source of energy.  Δ.29's tactical default (1.0)
    matches the inactive-scaling path and is bit-equal to the slow
    path's CSV.
    """
    from flextool.engine_polars._fast_load import load_flextool_source_only
    from flextool.engine_polars import SpineDbReader

    fixture = DATA / "work_lh2_three_region"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / "fast_lh2"
    work.mkdir()
    reader = SpineDbReader(db, scenario="lh2_three_region")
    fd = load_flextool_source_only(reader, work)

    assert fd.p_node_capacity_for_scaling is not None, (
        "fast path: p_node_capacity_for_scaling is None — Δ.29 "
        "default-1.0 helper regression?"
    )
    # 12 nodeBalance nodes × 1 period (y2030) = 12.
    assert fd.p_node_capacity_for_scaling.frame.height == 12, (
        f"fast path: p_node_capacity_for_scaling height="
        f"{fd.p_node_capacity_for_scaling.frame.height}, expected 12 "
        "(12 nodeBalance nodes × 1 period)."
    )
    assert (fd.p_node_capacity_for_scaling.frame["value"]
            == 1.0).all(), (
        "fast path: every p_node_capacity_for_scaling row should "
        "carry the default scalar 1.0 (scaling-inactive path)."
    )


# ---------------------------------------------------------------------------
# Multi-block fast-path sentinel — Phase 0 of the multi-block plan.
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_lh2_three_region_fast_path_obj_parity(tmp_path: Path) -> None:
    """Multi-block fast-path obj must match slow-path within 1e-6 rel err.

    Anchors the multi-block fast-path parity bar from
    specs/multi_block_fast_path_handoff.md.  Closed by Phase 1 +
    Phase 2 (``BlockLayout.from_source`` + threading through
    ``apply_derived_b/e``); current rel_err ≤ 1e-14 on LH2.
    """
    import json

    from polar_high import Problem
    from flextool.engine_polars import build_flextool, SpineDbReader
    from flextool.engine_polars._fast_load import load_flextool_source_only

    fixture = DATA / "work_lh2_three_region"
    db = fixture / "tests.sqlite"
    golden = fixture / "golden_obj.json"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    if not golden.exists():
        pytest.skip(f"fixture golden missing: {golden}")

    work = tmp_path / "fast_lh2"
    work.mkdir()
    reader = SpineDbReader(db, scenario="lh2_three_region")
    fd = load_flextool_source_only(reader, work)

    pb = Problem()
    build_flextool(pb, fd)
    sol = pb.solve()

    assert sol is not None, "fast path: solve() returned no Solution"
    assert sol.optimal, (
        f"fast path: HiGHS non-optimal "
        f"(status={getattr(sol, 'status', None)})"
    )

    golden_obj = json.loads(golden.read_text())["obj"]
    rel_err = abs(sol.obj - golden_obj) / abs(golden_obj)
    assert rel_err < 1e-6, (
        f"fast path obj={sol.obj} differs from golden "
        f"{golden_obj} by rel_err={rel_err:.3e}"
    )
