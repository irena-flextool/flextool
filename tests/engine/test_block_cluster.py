"""Block-layout consumers in :mod:`flextool.engine.preprocess.params._block`.

Per-fixture parity check against flextool's canonical preprocessed CSVs in
``solve_data/``:

* ``flow_to_n`` / ``flow_from_n`` — every arc ``(p, source, sink)`` with
  ``n`` = its sink / source (no block filter: a node of a coarse
  ``new_stepduration`` group sums its fine flows, ``nodeBalanceCoarse_eq``).
* ``nodeState_last_dt`` — the default block's last step of each period for
  every storage node.
* ``dtttdt_block_interior`` — interior-of-block dtttdt rows.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess.params._block_layout import DEFAULT_BLOCK
from flextool.engine.preprocess.params._block import (
    dtttdt_block_interior_lf,
    flow_from_n_frame,
    flow_to_n_frame,
    load_block_bundle,
    nodeState_last_dt_lf,
)
from flextool.engine.input._input_source import _read_csv_file


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------


# Phase 3d: curated parity sweep — replaces the legacy disk-discovery.
from _parity_sweep import PARITY_SWEEP_CASES  # noqa: E402

# Each tuple is (legacy_workdir_dirname, scenario, db_fixture).  The
# block-cluster tests parametrise over the legacy dirname only — keeping
# the test IDs stable — but consume the workdir via ``scenario_workdir``
# and the per-case scenario/db_fixture.
PARITY_CASES = [c[0] for c in PARITY_SWEEP_CASES]
_PARITY_DETAILS: dict[str, tuple[str, str]] = {
    legacy: (scen, dbf) for legacy, scen, dbf in PARITY_SWEEP_CASES
}


def _resolve_work(work: str, scenario_workdir) -> "Path":
    """Map a legacy ``work_<X>`` dirname → on-demand workdir via
    ``scenario_workdir``.  Returns the materialised tmp Path."""
    scen, dbf = _PARITY_DETAILS[work]
    return scenario_workdir(scen, db_fixture=dbf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frames_equal(a: pl.DataFrame, b: pl.DataFrame, *,
                   tol: float = 1e-9) -> tuple[bool, str]:
    """Compare two frames after sorting by every column."""
    if a is None and b is None:
        return True, "both None"
    if a is None or b is None:
        return False, f"one None: a={a is None}, b={b is None}"
    if set(a.columns) != set(b.columns):
        return False, f"columns differ: {a.columns} vs {b.columns}"
    cols = sorted(a.columns)
    a = a.select(*cols).sort(*cols)
    b = b.select(*cols).sort(*cols)
    if a.height != b.height:
        return False, f"heights differ: {a.height} vs {b.height}"
    for c in cols:
        if a[c].dtype.is_numeric() and b[c].dtype.is_numeric():
            la = a[c].cast(pl.Float64, strict=False)
            lb = b[c].cast(pl.Float64, strict=False)
            diff = (la - lb).abs().max()
            if diff is not None and diff > tol:
                return False, f"col {c!r} max-diff = {diff} > {tol}"
        else:
            la = a[c].cast(pl.Utf8, strict=False)
            lb = b[c].cast(pl.Utf8, strict=False)
            if not la.equals(lb):
                for i, (xa, xb) in enumerate(zip(la.to_list(), lb.to_list())):
                    if xa != xb:
                        return False, (
                            f"col {c!r} differ at row {i}: {xa!r} vs {xb!r}"
                        )
                return False, f"col {c!r} differ"
    return True, "ok"


def _csv_path_flow_n(workdir_path: Path, side: str) -> pl.DataFrame:
    """``flow_to_n`` (``side='sink'``) / ``flow_from_n`` (``'source'``) from
    the canonical CSVs: every arc, ``n`` = its ``side`` node."""
    sd = workdir_path / "solve_data"
    pss = _read_csv_file(sd / "process_source_sink.csv").rename(
        {"process": "p"})
    return (pss.with_columns(n=pl.col(side))
            .select("p", "source", "sink", "n").unique()
            .sort("p", "source", "sink", "n"))


# ---------------------------------------------------------------------------
# Per-fixture parity — flow_to_n block filter (Δ.3 gap closure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("work", PARITY_CASES, ids=PARITY_CASES)
@pytest.mark.parametrize("side", ["sink", "source"])
def test_flow_n_parity(work: str, side: str, scenario_workdir) -> None:
    """``flow_to_n_frame`` / ``flow_from_n_frame`` hold every arc once per
    end node (multi-block fixtures included)."""
    workdir_path = _resolve_work(work, scenario_workdir)
    sd = workdir_path / "solve_data"
    pss = _read_csv_file(sd / "process_source_sink.csv").rename(
        {"process": "p"})
    fn = flow_to_n_frame if side == "sink" else flow_from_n_frame
    actual = fn(pss).sort("p", "source", "sink", "n")
    expected = _csv_path_flow_n(workdir_path, side)
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work}: flow_n ({side}) differs: {diag}\n"
        f"expected (head):\n{expected.head(8)}\n"
        f"actual   (head):\n{actual.head(8)}"
    )


# ---------------------------------------------------------------------------
# Per-fixture parity — nodeState_last_dt (Δ.9 lift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("work", PARITY_CASES, ids=PARITY_CASES)
def test_nodeState_last_dt_parity(work: str, scenario_workdir) -> None:
    """``nodeState_last_dt`` matches the CSV oracle for fixtures with a
    ``nodeState`` set."""
    workdir_path = _resolve_work(work, scenario_workdir)
    sd = workdir_path / "solve_data"
    ns_path = sd / "nodeState.csv"
    if not ns_path.exists():
        return
    nodeState = _read_csv_file(ns_path).rename({"node": "n"})
    if nodeState.height == 0:
        return
    bundle = load_block_bundle(workdir_path)
    if bundle is None:
        return
    bptl_f = bundle.layout.block_period_time_last_frame
    eb_f = bundle.layout.entity_block_frame
    if bptl_f.height == 0 or eb_f.height == 0:
        return

    # Reference: the default block's last step of each period × every storage node.
    bptl = bptl_f.rename({"block": "bk", "period": "d", "step": "t"})
    expected = (nodeState.select("n").unique()
        .join(bptl.filter(pl.col("bk") == DEFAULT_BLOCK).select("d", "t"), how="cross")
        .select("n", "d", "t"))

    actual = nodeState_last_dt_lf(nodeState, bundle).collect()
    if expected.height == 0 and actual.height == 0:
        return
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work}: nodeState_last_dt differs: {diag}\n"
        f"expected (head):\n{expected.head(8)}\n"
        f"actual   (head):\n{actual.head(8)}"
    )


# ---------------------------------------------------------------------------
# Hand-cooked invariants — single-block + multi-block
# ---------------------------------------------------------------------------


def test_multi_block_flow_to_n_keeps_every_arc(scenario_workdir) -> None:
    """On ``lh2_three_region`` (hourly + daily nodes) every arc enters the
    balance of its sink: the electrolysers' hourly→daily arcs and the
    liquefiers' daily→daily arcs included.  Before c15b the block filter
    dropped the arcs whose side block did not overlap the node's block."""
    work_lh2 = scenario_workdir("lh2_three_region", db_fixture="lh2")
    sd = work_lh2 / "solve_data"
    pss = _read_csv_file(sd / "process_source_sink.csv").rename(
        {"process": "p"})
    actual = flow_to_n_frame(pss)
    assert actual.height == pss.select("p", "source", "sink").unique().height
    sinks = set(zip(actual["p"].cast(pl.Utf8).to_list(), actual["n"].cast(pl.Utf8).to_list()))
    assert ("electrolyser_A", "h2_A") in sinks and ("liquefier_A", "lh2_A") in sinks


def test_no_workdir_returns_none() -> None:
    """``load_block_bundle(None)`` returns ``None`` cleanly."""
    bundle = load_block_bundle(None)
    assert bundle is None


def test_dtttdt_block_interior_default_branch() -> None:
    """Default branch: keep dtttdt rows with
    ``t_previous_within_timeset == t_previous``."""
    df = pl.DataFrame({
        "d": ["d1"] * 4,
        "t": ["t1", "t2", "t3", "t4"],
        "t_previous": ["t0", "t1", "t2", "t3"],
        "t_previous_within_timeset": ["t0", "t1", "t2", "t3"],
        "d_previous": ["d1"] * 4,
        "t_previous_within_solve": ["t0", "t1", "t2", "t3"],
    })
    out = dtttdt_block_interior_lf(df, period_block_time=None).collect()
    assert out.height == 4
    assert set(out.columns) == {"d", "t", "t_previous"}


def test_dtttdt_block_interior_synthesised_branch() -> None:
    """Synthesised branch: rebuild interior pairs from coarse-block
    period_block_time."""
    pbt = pl.DataFrame({
        "d": ["d1"] * 6,
        "b_first": ["block1"] * 3 + ["block2"] * 3,
        "t": ["t1", "t2", "t3", "t4", "t5", "t6"],
    })
    dtttdt = pl.DataFrame({
        "d": ["d1"] * 6,
        "t": ["t1", "t2", "t3", "t4", "t5", "t6"],
        "t_previous": ["t0", "t1", "t2", "t3", "t4", "t5"],
        "t_previous_within_timeset": ["t0", "t1", "t2", "t3", "t4", "t5"],
        "d_previous": ["d1"] * 6,
        "t_previous_within_solve": ["t0", "t1", "t2", "t3", "t4", "t5"],
    })
    out = dtttdt_block_interior_lf(dtttdt, pbt).collect()
    # Two blocks of 3 steps each → 2 interior pairs per block = 4 rows.
    assert out.height == 4
    # Boundaries between blocks are NOT included (t4 follows t3 across
    # blocks, but the synthesised branch only emits intra-block pairs).
    assert (out.filter(pl.col("t") == "t4").height == 0)


def test_block_bundle_group_steps(scenario_workdir) -> None:
    """The bundle's layout carries each node's group and the group steps:
    lh2's daily_group is 24 h; a fixture without groups has the default block
    only."""
    single = load_block_bundle(scenario_workdir("coal"))
    multi = load_block_bundle(
        scenario_workdir("lh2_three_region", db_fixture="lh2"))
    assert single is not None and multi is not None
    assert set(single.layout.block_step_duration) == {DEFAULT_BLOCK}
    assert multi.layout.block_step_duration["daily_group"] == 24.0
    assert multi.layout.node_block["h2_A"] == "daily_group"
