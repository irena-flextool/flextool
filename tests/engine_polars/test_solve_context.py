"""Δ.12a — focused unit tests for :class:`SolveContext`.

The cluster sweeps verify ``load_flextool`` end-to-end with the
context activated; this file pins down the SolveContext API itself
(typed fields, ``read_csv`` cache, activate / deactivate semantics).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._input_source import (
    _install_csv_cache,
    _read_csv_file,
    seed_provider_from_dir,
)
from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._solve_context import SolveContext


def _seed_workdir_provider(wd: Path) -> FlexDataProvider:
    """Build an in-memory provider from a test workdir's input/ +
    solve_data/ for SolveContext, post Step 2.5 (disk fallback removed).

    Seeds empty placeholder frames for SolveContext's lazy-field
    carriers whose absence was previously treated as "empty default"
    via the disk-fallback path.  Post Step 2.5 the lazy readers call
    ``_provider_fetch_or_raise``; an empty placeholder keeps the
    test's "empty workdir → empty fields" semantics intact under the
    new strict contract.
    """
    provider = FlexDataProvider()
    if (wd / "input").exists():
        seed_provider_from_dir(provider, wd / "input", "input")
    if (wd / "solve_data").exists():
        seed_provider_from_dir(provider, wd / "solve_data", "solve_data")
    # Empty placeholders for the SolveContext lazy carriers — only
    # seeded when the on-disk seed didn't supply them.
    for stem in ("period_in_use_set", "period__branch", "edd_history",
                  "p_entity_period_existing_capacity",
                  "p_entity_pre_existing"):
        key = f"solve_data/{stem}"
        if not provider.has(key):
            provider.put(key, pl.DataFrame())
    return provider


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(c) for c in r))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    sd = tmp_path / "solve_data"
    inp = tmp_path / "input"

    _write_csv(sd / "solve_current.csv", ["solve"], [["s1"]])
    _write_csv(
        sd / "p_model.csv",
        ["modelParam", "p_model"],
        [["solveFirst", 1]],
    )
    _write_csv(
        sd / "realized_dispatch.csv",
        ["solve", "period", "step"],
        [["s1", "p2020", "t01"], ["s1", "p2020", "t02"], ["s1", "p2025", "t01"]],
    )
    _write_csv(
        sd / "realized_invest_periods_of_current_solve.csv",
        ["period"],
        [["p2020"], ["p2025"]],
    )
    _write_csv(
        sd / "period_in_use_set.csv",
        ["period"],
        [["p2020"], ["p2025"]],
    )
    _write_csv(
        sd / "period__branch.csv",
        ["period", "branch"],
        [["p2020", "p2020"], ["p2020", "p2020_alt"]],
    )
    _write_csv(
        sd / "edd_history.csv",
        ["entity", "period_history", "period"],
        [["wind", "p2020", "p2025"]],
    )
    _write_csv(
        sd / "p_entity_period_existing_capacity.csv",
        ["entity", "period", "p_entity_period_existing_capacity",
         "p_entity_period_invested_capacity"],
        [["wind", "p2020", 100, 0], ["wind", "p2025", 100, 0]],
    )
    _write_csv(
        sd / "p_entity_pre_existing.csv",
        ["entity", "period", "value"],
        [["wind", "p2020", 100]],
    )
    inp.mkdir(parents=True, exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Typed-field correctness
# ---------------------------------------------------------------------------


def test_solve_context_typed_fields_loaded(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    assert ctx.solve_name == "s1"
    assert ctx.solveFirst is True
    assert ctx.realized_periods == {"p2020", "p2025"}
    assert ctx.realized_invest_periods == {"p2020", "p2025"}
    assert ctx.period_in_use.height == 2
    assert set(ctx.period_in_use["d"].to_list()) == {"p2020", "p2025"}
    assert ctx.period_branch.height == 2
    assert ctx.edd_history.height == 1
    assert ctx.p_entity_period_existing_capacity.height == 2
    assert ctx.p_entity_pre_existing.height == 1


def test_solve_context_handles_missing_workdir(tmp_path: Path) -> None:
    """Empty workdir → all fields default to empty."""
    ctx = SolveContext.from_workdir(tmp_path, provider=_seed_workdir_provider(tmp_path))
    assert ctx.solve_name is None
    assert ctx.solveFirst is True  # default policy
    assert ctx.realized_periods == set()
    assert ctx.realized_invest_periods == set()
    assert ctx.period_in_use.height == 0
    assert ctx.period_branch.height == 0


def test_solve_context_solve_first_zero(tmp_path: Path) -> None:
    sd = tmp_path / "solve_data"
    _write_csv(
        sd / "p_model.csv",
        ["modelParam", "p_model"],
        [["solveFirst", 0]],
    )
    ctx = SolveContext.from_workdir(tmp_path, provider=_seed_workdir_provider(tmp_path))
    assert ctx.solveFirst is False


# ---------------------------------------------------------------------------
# read_csv cache semantics
# ---------------------------------------------------------------------------


def test_read_csv_caches_repeats(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    df1 = ctx.read_csv("solve_current.csv")
    df2 = ctx.read_csv("solve_current.csv")
    # Same identity → cache hit.
    assert df1 is df2


def test_read_csv_returns_none_for_missing(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    assert ctx.read_csv("does_not_exist.csv") is None


def test_read_csv_kind_input(workdir: Path) -> None:
    inp = workdir / "input"
    _write_csv(inp / "p_node.csv", ["node", "value"], [["n1", 1.0]])
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    df = ctx.read_csv("p_node.csv", kind="input")
    assert df is not None
    assert df.height == 1


# ---------------------------------------------------------------------------
# Activation / cache scoping
# ---------------------------------------------------------------------------


def test_activate_installs_global_cache(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    csv_path = workdir / "solve_data" / "solve_current.csv"
    # Pre-activation: each _read_csv_file is a fresh polars.read_csv.
    df_a = _read_csv_file(csv_path)
    df_b = _read_csv_file(csv_path)
    assert df_a is not df_b

    ctx.activate()
    try:
        df_c = _read_csv_file(csv_path)
        df_d = _read_csv_file(csv_path)
        # Active cache → identity-stable on repeats.
        assert df_c is df_d
    finally:
        ctx.deactivate()

    # Post-deactivation: fresh reads again.
    df_e = _read_csv_file(csv_path)
    df_f = _read_csv_file(csv_path)
    assert df_e is not df_f


def test_context_manager_protocol(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    csv_path = workdir / "solve_data" / "solve_current.csv"
    with ctx:
        df1 = _read_csv_file(csv_path)
        df2 = _read_csv_file(csv_path)
        assert df1 is df2
    # After exit cache is uninstalled.
    df3 = _read_csv_file(csv_path)
    df4 = _read_csv_file(csv_path)
    assert df3 is not df4


def test_install_csv_cache_clears(workdir: Path) -> None:
    """Direct assertion against the install / uninstall API."""
    cache: dict[str, pl.DataFrame] = {}
    csv_path = workdir / "solve_data" / "solve_current.csv"
    _install_csv_cache(cache)
    try:
        _read_csv_file(csv_path)
        # First read populates cache.
        assert len(cache) == 1
    finally:
        _install_csv_cache(None)


def test_solve_data_dir_property(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    assert ctx.solve_data_dir == workdir / "solve_data"
    assert ctx.input_dir == workdir / "input"
