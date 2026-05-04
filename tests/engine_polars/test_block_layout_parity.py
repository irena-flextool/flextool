"""Δ.2 parity tests for ``flextool.engine_polars._block_layout``.

The flexpy port (``BlockLayout``) must produce the same per-solve
block frames as the canonical
``flextool.flextoolrunner.blocks`` reference functions on every
fixture in ``tests/engine_polars/data/work_*``.

Coverage
--------

* **Per-fixture parity** — every fixture exercising blocks (every
  ``work_*`` directory with ``solve_data/{entity_block,
  process_side_block, block_step_duration, overlap_set,
  block_step_previous, block_period_time_first, _last}.csv``)
  must field-by-field match the reference.

* **Single-block degenerate test** — ``work_base`` exercises the
  trivial case (one ``default`` block, identity-only overlap).

* **Multi-block test** — ``work_lh2_three_region`` exercises 3
  blocks (default / hourly_group / daily_group); asserts
  ``overlap_set``, per-block step-duration, and mixed-block arcs
  match the reference.

The test reuses the fixture-discovery pattern from
``test_solve_config_parity.py``.

Inputs are reconstructed from each fixture's ``input/`` CSVs (entity
class lists, group memberships, process source/sink, ct_method) and
``solve_data/`` CSVs (``step_previous.csv`` for ``default_jump_list``
and ``block_step_duration.csv``'s default rows for the
``active_time_list``).  This lets the parity test run without
re-driving flextool's full orchestration loop.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._block_layout import (
    DEFAULT_BLOCK,
    BlockLayout,
)
from flextool.engine_polars._solve_state import ActiveTimeEntry


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------


def _discover_fixtures() -> list[str]:
    """Return ``[work_dirname, ...]`` for every fixture with the full
    block-CSV set under ``solve_data/``.

    Fixtures missing one or more of the block CSVs (rare; only when
    flextool's orchestration didn't fire ``write_block_data_for_solve``)
    are skipped — there's nothing to compare against.
    """
    out: list[str] = []
    required = (
        "entity_block.csv",
        "process_side_block.csv",
        "process_block.csv",
        "block_step_duration.csv",
        "overlap_set.csv",
        "block_step_previous.csv",
        "block_period_time_first.csv",
        "block_period_time_last.csv",
        "step_previous.csv",
    )
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("work_"):
            continue
        sd = d / "solve_data"
        inp = d / "input"
        if not sd.is_dir() or not inp.is_dir():
            continue
        if not all((sd / f).exists() for f in required):
            continue
        out.append(d.name)
    return out


PARITY_CASES = _discover_fixtures()


# ---------------------------------------------------------------------------
# Input reconstruction from on-disk fixtures
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[list[str]]:
    """Read ``path`` as a CSV, return rows minus the header (or [] if
    the file is empty)."""
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        return [row for row in reader if row]


def _read_csv_dict(path: Path) -> list[dict[str, str]]:
    """Read ``path`` as a CSV via ``DictReader``, returning ``[{col: val,
    ...}]``.  Empty file → []."""
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _build_inputs_from_fixture(workdir: Path):
    """Reconstruct all ``BlockLayout.build`` inputs from a fixture's
    ``input/`` and ``solve_data/`` CSVs.

    The reconstruction matches what ``write_block_data_for_solve``
    reads from the same files, so calling the reference vs the port
    on the reconstructed inputs is a true apples-to-apples comparison.

    Returns
    -------
    dict
        Keyword arguments suitable for ``BlockLayout.build`` and the
        reference functions.
    """
    inp = workdir / "input"
    sd = workdir / "solve_data"

    # Entity class lists ---------------------------------------------------
    nodes = [r[0] for r in _read_csv_rows(inp / "node.csv")]
    units = [r[0] for r in _read_csv_rows(inp / "process_unit.csv")]
    connections = [r[0] for r in _read_csv_rows(inp / "process_connection.csv")]
    if not nodes and not units and not connections:
        # Pre-v42 fallback — entity.csv is the catch-all.
        nodes = [r[0] for r in _read_csv_rows(inp / "entity.csv")]

    # Resolution groups (numeric new_stepduration) -------------------------
    resolution_groups: dict[str, float] = {}
    for row in _read_csv_dict(inp / "p_group.csv"):
        if row.get("groupParam") == "new_stepduration":
            try:
                resolution_groups[row["group"]] = float(row["p_group"])
            except (TypeError, ValueError):
                continue

    # Decomposition groups (string method enum) ---------------------------
    decomposition_groups: dict[str, str] = {}
    for row in _read_csv_dict(inp / "p_group_decomposition.csv"):
        if row.get("groupParam") == "decomposition_method":
            decomposition_groups[row["group"]] = str(row["p_group"])
    if not decomposition_groups:
        for row in _read_csv_dict(inp / "p_group.csv"):
            if row.get("groupParam") == "decomposition_method":
                decomposition_groups[row["group"]] = str(row["p_group"])

    # Group memberships ----------------------------------------------------
    group_node = [(r[0], r[1]) for r in _read_csv_rows(inp / "group__node.csv")]
    group_process_all = [
        (r[0], r[1]) for r in _read_csv_rows(inp / "group__process.csv")
    ]
    unit_set = set(units)
    conn_set = set(connections)
    group_unit = [(g, p) for g, p in group_process_all if p in unit_set]
    group_connection = [
        (g, p) for g, p in group_process_all if p in conn_set
    ]

    # Reserve membership (Agent 1.7) --------------------------------------
    reserve_upDown_group = [
        (r[0], r[1], r[2])
        for r in _read_csv_rows(inp / "reserve__upDown__group__method.csv")
        if len(r) >= 4 and r[3] != "no_reserve"
    ]
    process_reserve_upDown_node = [
        (r[0], r[1], r[2], r[3])
        for r in _read_csv_rows(inp / "process__reserve__upDown__node.csv")
        if len(r) >= 4
    ]

    # Process source/sink: first source/sink per process ------------------
    sources: dict[str, list[str]] = defaultdict(list)
    sinks: dict[str, list[str]] = defaultdict(list)
    for r in _read_csv_rows(inp / "process__source.csv"):
        if len(r) >= 2:
            sources[r[0]].append(r[1])
    for r in _read_csv_rows(inp / "process__sink.csv"):
        if len(r) >= 2:
            sinks[r[0]].append(r[1])
    process_source_sink = []
    for p in set(list(sources.keys()) + list(sinks.keys())):
        src_list = sources.get(p, [""])
        snk_list = sinks.get(p, [""])
        process_source_sink.append((p, src_list[0], snk_list[0]))

    # Process ct_method ----------------------------------------------------
    process_ct_method: dict[str, str] = {
        r[0]: r[1]
        for r in _read_csv_rows(inp / "process__ct_method.csv")
        if len(r) >= 2
    }

    # active_time_list: reconstructed from block_step_duration's default
    # rows.  This is what the reference would build when given the
    # default-block timeline directly.
    bsd_rows = _read_csv_dict(sd / "block_step_duration.csv")
    active_time_list: dict[str, list[ActiveTimeEntry]] = {}
    idx_per_period: dict[str, int] = defaultdict(int)
    for row in bsd_rows:
        if row["block"] != DEFAULT_BLOCK:
            continue
        period = row["period"]
        active_time_list.setdefault(period, []).append(
            ActiveTimeEntry(
                timestep=row["step"],
                index=idx_per_period[period],
                duration=row["step_duration"],
            )
        )
        idx_per_period[period] += 1

    # default_jump_list: read step_previous.csv as 7-tuples (period,
    # step, previous, prev_within_ts, prev_period, prev_within_solve,
    # jump).
    sp_rows = _read_csv_dict(sd / "step_previous.csv")
    default_jump_list = [
        (
            r["period"], r["time"], r["previous"],
            r["previous_within_timeset"], r["previous_period"],
            r["previous_within_solve"], r["jump"],
        )
        for r in sp_rows
    ]

    return dict(
        nodes=nodes,
        units=units,
        connections=connections,
        resolution_groups=resolution_groups,
        decomposition_groups=decomposition_groups,
        group_unit=group_unit,
        group_connection=group_connection,
        group_node=group_node,
        reserve_upDown_group=reserve_upDown_group,
        process_reserve_upDown_node=process_reserve_upDown_node,
        process_source_sink=process_source_sink,
        process_ct_method=process_ct_method,
        active_time_list=active_time_list,
        default_jump_list=default_jump_list,
    )


# ---------------------------------------------------------------------------
# Reference invocation
# ---------------------------------------------------------------------------


def _run_reference(inputs: dict) -> dict:
    """Drive the reference ``flextool/flextoolrunner/blocks.py``
    helpers with the reconstructed inputs and return the same set of
    dataclass objects as our port produces.
    """
    from flextool.flextoolrunner import blocks as ref_blocks

    ref_blocks.validate_group_membership(
        inputs["group_unit"], inputs["group_connection"], inputs["group_node"],
        inputs["resolution_groups"], inputs["decomposition_groups"],
        reserve_upDown_group=inputs["reserve_upDown_group"],
        process_reserve_upDown_node=inputs["process_reserve_upDown_node"],
    )

    ba = ref_blocks.derive_blocks(
        solve="ref",
        solve_config=None,
        timeline_config=None,
        nodes=inputs["nodes"],
        units=inputs["units"],
        connections=inputs["connections"],
        resolution_groups=inputs["resolution_groups"],
        group_unit=inputs["group_unit"],
        group_connection=inputs["group_connection"],
        group_node=inputs["group_node"],
        process_source_sink=inputs["process_source_sink"],
        process_ct_method=inputs["process_ct_method"],
    )
    bt = ref_blocks._build_block_timelines(
        solve="ref",
        solve_config=None,
        timeline_config=None,
        block_assignments=ba,
        active_time_list=inputs["active_time_list"],
    )
    overlap = ref_blocks.derive_overlap_set(
        solve="ref",
        block_assignments=ba,
        block_timelines=bt,
    )
    pred = ref_blocks.derive_block_predecessors(
        solve="ref",
        block_assignments=ba,
        block_timelines=bt,
        default_jump_list=inputs["default_jump_list"],
    )
    bnd = ref_blocks.derive_block_boundaries(
        block_assignments=ba,
        block_timelines=bt,
    )
    return dict(
        node_block=ba.node_block,
        process_block_in=ba.process_block_in,
        process_block_out=ba.process_block_out,
        process_block=ba.process_block,
        block_step_duration=ba.block_step_duration,
        per_block=bt.per_block,
        overlap_rows=list(overlap.rows),
        predecessor_rows=list(pred.rows),
        boundary_first=list(bnd.first),
        boundary_last=list(bnd.last),
    )


def _run_port(inputs: dict) -> BlockLayout:
    return BlockLayout.build(
        solve="port",
        solve_config=None,
        timeline_config=None,
        nodes=inputs["nodes"],
        units=inputs["units"],
        connections=inputs["connections"],
        resolution_groups=inputs["resolution_groups"],
        group_unit=inputs["group_unit"],
        group_connection=inputs["group_connection"],
        group_node=inputs["group_node"],
        process_source_sink=inputs["process_source_sink"],
        process_ct_method=inputs["process_ct_method"],
        decomposition_groups=inputs["decomposition_groups"],
        reserve_upDown_group=inputs["reserve_upDown_group"],
        process_reserve_upDown_node=inputs["process_reserve_upDown_node"],
        active_time_list=inputs["active_time_list"],
        default_jump_list=inputs["default_jump_list"],
    )


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def _diff_dict(name: str, mine, ref) -> str | None:
    if mine == ref:
        return None
    a = mine if isinstance(mine, dict) else dict(mine)
    b = ref if isinstance(ref, dict) else dict(ref)
    ka, kb = set(a), set(b)
    missing = kb - ka
    extra = ka - kb
    bad_vals = {k: (a[k], b[k]) for k in (ka & kb) if a[k] != b[k]}
    return (
        f"{name} differs: missing={sorted(missing)} "
        f"extra={sorted(extra)} bad_vals={bad_vals}"
    )


def _diff_rows(name: str, mine, ref) -> str | None:
    """Compare list-of-tuple frames as multisets (order-insensitive)."""
    a = sorted(mine)
    b = sorted(ref)
    if a == b:
        return None
    only_mine = [r for r in a if r not in set(b)]
    only_ref = [r for r in b if r not in set(a)]
    return (
        f"{name} differs: "
        f"len(mine)={len(a)} len(ref)={len(b)} "
        f"only_mine[:5]={only_mine[:5]} only_ref[:5]={only_ref[:5]}"
    )


# ---------------------------------------------------------------------------
# Per-fixture parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name", PARITY_CASES, ids=lambda w: w,
)
def test_block_layout_field_parity(work_name: str) -> None:
    """Field-by-field zero diff vs ``flextoolrunner.blocks`` reference."""
    workdir = DATA / work_name
    inputs = _build_inputs_from_fixture(workdir)
    ref = _run_reference(inputs)
    port = _run_port(inputs)

    diffs: list[str] = []

    # Internal dicts (BlockAssignments).
    for f in (
        "node_block",
        "process_block_in",
        "process_block_out",
        "process_block",
        "block_step_duration",
    ):
        d = _diff_dict(f, getattr(port, f), ref[f])
        if d:
            diffs.append(d)

    # Per-block timelines (dict of dict of list-of-tuples).
    if port.per_block_timeline != ref["per_block"]:
        # Render diff summary at the (block, period) level.
        details = []
        for block in set(port.per_block_timeline) | set(ref["per_block"]):
            mp = port.per_block_timeline.get(block, {})
            rp = ref["per_block"].get(block, {})
            for period in set(mp) | set(rp):
                if mp.get(period) != rp.get(period):
                    details.append(
                        f"({block}, {period}): "
                        f"mine[:3]={mp.get(period, [])[:3]} "
                        f"ref[:3]={rp.get(period, [])[:3]}"
                    )
        diffs.append("per_block_timeline differs: " + "; ".join(details[:5]))

    # Tuple lists — overlap, predecessors, boundaries.
    # Note: overlap_set order matters in the reference (insertion-
    # determined), but for parity-correctness purposes we compare as
    # multisets — the LP build is order-insensitive.
    d = _diff_rows("overlap_rows", port._overlap_rows, ref["overlap_rows"])
    if d:
        diffs.append(d)
    d = _diff_rows(
        "predecessor_rows", port._predecessor_rows, ref["predecessor_rows"]
    )
    if d:
        diffs.append(d)
    d = _diff_rows(
        "boundary_first", port._boundary_first_rows, ref["boundary_first"]
    )
    if d:
        diffs.append(d)
    d = _diff_rows(
        "boundary_last", port._boundary_last_rows, ref["boundary_last"]
    )
    if d:
        diffs.append(d)

    assert not diffs, (
        f"{work_name} block layout diverged from flextoolrunner reference:\n"
        + "\n".join(f"  - {d}" for d in diffs)
    )


# ---------------------------------------------------------------------------
# Spot tests: degenerate single-block + multi-block lh2_three_region
# ---------------------------------------------------------------------------


def test_single_block_degeneracy_work_base() -> None:
    """``work_base`` exercises the trivial case (one ``default``
    block, identity-only overlap)."""
    workdir = DATA / "work_base"
    if not (workdir / "solve_data" / "block_step_duration.csv").exists():
        pytest.skip("work_base block CSVs missing")

    inputs = _build_inputs_from_fixture(workdir)
    port = _run_port(inputs)

    # Only the default block.
    assert set(port.block_step_duration) == {DEFAULT_BLOCK}

    # Every node maps to default.
    assert set(port.node_block.values()) == {DEFAULT_BLOCK}

    # process_side_block — every entry is default.
    psb = port.process_side_block_frame
    if psb.height > 0:
        assert set(psb["block"].to_list()) == {DEFAULT_BLOCK}

    # overlap_set — all identity rows (block_coarse == block_fine,
    # step_coarse == step_fine, fraction == 1.0).
    ov = port.overlap_set_frame
    assert ov.height > 0
    assert (ov["block_coarse"] == DEFAULT_BLOCK).all()
    assert (ov["block_fine"] == DEFAULT_BLOCK).all()
    assert (ov["step_coarse"] == ov["step_fine"]).all()
    assert (ov["fraction"] == 1.0).all()


def test_multi_block_lh2_three_region() -> None:
    """``work_lh2_three_region`` exercises 3 blocks (default /
    hourly_group / daily_group)."""
    workdir = DATA / "work_lh2_three_region"
    if not (workdir / "solve_data" / "block_step_duration.csv").exists():
        pytest.skip("lh2_three_region block CSVs missing")

    inputs = _build_inputs_from_fixture(workdir)
    port = _run_port(inputs)

    HOURLY = "hourly_group"
    DAILY = "daily_group"
    assert set(port.block_step_duration) == {DEFAULT_BLOCK, HOURLY, DAILY}
    assert port.block_step_duration[HOURLY] == 1.0
    assert port.block_step_duration[DAILY] == 24.0

    # Per-side block assignment: hourly nodes (electrolyser sources) on
    # hourly_group; daily nodes (h2 / lh2) on daily_group.
    nb = port.node_block
    for n in ("h2_A", "h2_B", "h2_C", "lh2_A", "lh2_B", "lh2_C"):
        assert nb.get(n) == DAILY, n
    for n in ("battery_A", "battery_B", "battery_C", "elec_A"):
        assert nb.get(n) == HOURLY, n

    # process_side_block: indirect electrolysers split per side
    # (source=hourly, sink=daily); pipes are daily↔daily.
    psb = port.process_side_block_frame
    elec_sides = (
        psb.filter(pl.col("process").str.starts_with("electrolyser_"))
        .pivot(values="block", index="process", on="side")
        .to_dicts()
    )
    assert len(elec_sides) >= 1
    for r in elec_sides:
        assert r["source"] == HOURLY, r
        assert r["sink"] == DAILY, r

    # overlap_set: 168 daily↔hourly rows per period (7 coarse × 24 fine).
    ov = port.overlap_set_frame
    d2h = ov.filter(
        (pl.col("block_coarse") == DAILY)
        & (pl.col("block_fine") == HOURLY)
    )
    assert d2h.height == 168, d2h.height
    # Self-identity for every block.
    for b in (DEFAULT_BLOCK, HOURLY, DAILY):
        self_rows = ov.filter(
            (pl.col("block_coarse") == b) & (pl.col("block_fine") == b)
        )
        assert self_rows.height > 0
        assert (self_rows["step_coarse"] == self_rows["step_fine"]).all()
        assert (self_rows["fraction"] == 1.0).all()


# ---------------------------------------------------------------------------
# Frame-shape sanity (column names + dtypes)
# ---------------------------------------------------------------------------


def test_frame_columns_match_canonical_csv_headers() -> None:
    """Every frame uses the canonical CSV column names so downstream
    helpers can join directly without renaming."""
    if not PARITY_CASES:
        pytest.skip("no parity fixtures discovered")

    workdir = DATA / PARITY_CASES[0]
    inputs = _build_inputs_from_fixture(workdir)
    port = _run_port(inputs)

    assert port.entity_block_frame.columns == ["entity", "block"]
    assert port.process_side_block_frame.columns == ["process", "side", "block"]
    assert port.process_block_frame.columns == ["process", "block"]
    assert port.block_step_duration_frame.columns == [
        "block", "period", "step", "step_duration",
    ]
    assert port.overlap_set_frame.columns == [
        "period", "block_coarse", "step_coarse",
        "block_fine", "step_fine", "fraction",
    ]
    assert port.block_step_previous_frame.columns == [
        "block", "period", "step", "step_previous",
        "step_previous_within_timeset", "period_previous",
        "step_previous_within_solve",
    ]
    assert port.block_period_time_first_frame.columns == [
        "block", "period", "step",
    ]
    assert port.block_period_time_last_frame.columns == [
        "block", "period", "step",
    ]


def test_validate_group_membership_rejects_dual_resolution_membership() -> None:
    """Validation must raise on an entity in two resolution-groups."""
    from flextool.engine_polars._block_layout import (
        validate_group_membership,
    )
    from flextool.engine_polars._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="multiple resolution-groups"):
        validate_group_membership(
            group_unit=[("hourly_group", "u1"), ("daily_group", "u1")],
            group_connection=[],
            group_node=[],
            resolution_groups={"hourly_group": 1.0, "daily_group": 24.0},
            decomposition_groups={},
        )


def test_validate_group_membership_rejects_reserve_in_resolution_group() -> None:
    """V1 rule: reserve participants must NOT sit in a resolution-group."""
    from flextool.engine_polars._block_layout import (
        validate_group_membership,
    )
    from flextool.engine_polars._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="participates in reserves"):
        validate_group_membership(
            group_unit=[],
            group_connection=[],
            group_node=[("daily_group", "n1")],
            resolution_groups={"daily_group": 24.0},
            decomposition_groups={},
            reserve_upDown_group=[("res", "up", "g_res")],
            process_reserve_upDown_node=[("p1", "res", "up", "n1")],
        )


def test_aligned_subsets_only_raises_on_non_aligned() -> None:
    """The reference's aligned-subsets assumption raises on non-aligned
    fine→coarse aggregation; our port preserves that behaviour."""
    from flextool.engine_polars._block_layout import _aggregate_timeline

    # 3h coarse against 2h fine rows — not aligned.
    rows = [("t1", 2.0), ("t2", 2.0), ("t3", 2.0)]
    with pytest.raises(NotImplementedError, match="Non-aligned subset"):
        _aggregate_timeline(rows, coarse_duration=3.0)


def test_load_from_solve_data_bridge_work_lh2() -> None:
    """``BlockLayout.load_from_solve_data`` reads the on-disk CSVs into
    the same frame surface as the live ``build`` would produce.

    Used as a transitional bridge while the orchestrator still drives
    flextool's ``write_block_data_for_solve``.  This test pins the
    invariant on a multi-block fixture so the bridge stays compatible
    with the live builder (i.e., callers can swap one for the other).
    """
    workdir = DATA / "work_lh2_three_region"
    if not (workdir / "solve_data" / "block_step_duration.csv").exists():
        pytest.skip("lh2_three_region block CSVs missing")

    inputs = _build_inputs_from_fixture(workdir)
    built = _run_port(inputs)
    loaded = BlockLayout.load_from_solve_data(workdir / "solve_data")

    # Frames match (multiset comparison — row order in some frames is
    # insertion-determined and not load-bearing).
    def _eq(a: pl.DataFrame, b: pl.DataFrame) -> bool:
        return sorted(a.iter_rows()) == sorted(b.iter_rows())

    assert _eq(built.entity_block_frame, loaded.entity_block_frame)
    assert _eq(built.process_side_block_frame, loaded.process_side_block_frame)
    assert _eq(built.process_block_frame, loaded.process_block_frame)
    assert _eq(built.block_step_duration_frame, loaded.block_step_duration_frame)
    assert _eq(built.overlap_set_frame, loaded.overlap_set_frame)
    assert _eq(built.block_period_time_first_frame,
               loaded.block_period_time_first_frame)
    assert _eq(built.block_period_time_last_frame,
               loaded.block_period_time_last_frame)
    assert _eq(built.block_step_previous_frame,
               loaded.block_step_previous_frame)

    # Reconstructed dicts.
    assert built.node_block == loaded.node_block
    assert built.process_block == loaded.process_block
    assert built.block_step_duration == loaded.block_step_duration


def test_load_from_solve_data_missing_directory_returns_empty() -> None:
    """``missing_ok=True`` (default): a non-existent solve_data/
    yields an empty BlockLayout (every frame height 0)."""
    layout = BlockLayout.load_from_solve_data(
        DATA / "no_such_directory_for_blockslayout",
    )
    assert layout.is_empty()


def test_load_from_solve_data_block_compat_lh2() -> None:
    """``block_compat`` returns the (b, b_f) compatibility set used to
    filter flow_to_n / flow_from_n in input.py."""
    workdir = DATA / "work_lh2_three_region"
    if not (workdir / "solve_data" / "overlap_set.csv").exists():
        pytest.skip("lh2_three_region overlap_set missing")

    layout = BlockLayout.load_from_solve_data(workdir / "solve_data")
    compat = layout.block_compat()
    pairs = set(zip(compat["b"].to_list(), compat["b_f"].to_list()))

    # daily_group → fine (hourly / default) emitted in both senses
    # (coarse↔default symmetric, coarse↔fine canonical-direction).
    assert ("daily_group", "hourly_group") in pairs
    assert ("daily_group", "default") in pairs
    # default↔coarse symmetric row exists (default→daily) so node-side
    # filtering on a default-block node picks up daily-side arcs.
    assert ("default", "daily_group") in pairs
    # Self-identity for every block.
    assert ("daily_group", "daily_group") in pairs
    assert ("hourly_group", "hourly_group") in pairs
    assert ("default", "default") in pairs


def test_load_from_solve_data_coarse_blocks_lh2() -> None:
    """``coarse_blocks`` selects blocks with step_duration > threshold."""
    workdir = DATA / "work_lh2_three_region"
    if not (workdir / "solve_data" / "block_step_duration.csv").exists():
        pytest.skip("lh2_three_region block_step_duration missing")

    layout = BlockLayout.load_from_solve_data(workdir / "solve_data")
    coarse = set(layout.coarse_blocks(threshold=1.0))
    # daily_group is the only block with step_duration > 1h.
    assert coarse == {"daily_group"}


def test_default_block_only_emits_identity_overlap() -> None:
    """A fixture with no resolution-groups produces identity-only
    overlap rows (degenerate case)."""
    layout = BlockLayout.build(
        solve="t",
        solve_config=None,
        timeline_config=None,
        nodes=["n1", "n2"],
        units=["u1"],
        connections=[],
        resolution_groups={},
        group_unit=[],
        group_connection=[],
        group_node=[],
        process_source_sink=[("u1", "n1", "n2")],
        process_ct_method={"u1": "constant_efficiency"},
        active_time_list={
            "p1": [
                ActiveTimeEntry(timestep="t1", index=0, duration="1.0"),
                ActiveTimeEntry(timestep="t2", index=1, duration="1.0"),
            ],
        },
    )
    assert set(layout.block_step_duration) == {DEFAULT_BLOCK}
    assert layout.node_block == {"n1": DEFAULT_BLOCK, "n2": DEFAULT_BLOCK}
    assert layout.process_block_in == {"u1": DEFAULT_BLOCK}
    assert layout.process_block_out == {"u1": DEFAULT_BLOCK}
    ov = layout.overlap_set_frame
    assert ov.height == 2
    assert (ov["block_coarse"] == DEFAULT_BLOCK).all()
    assert (ov["step_coarse"] == ov["step_fine"]).all()
