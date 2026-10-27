"""Δ.12a — focused unit tests for :class:`SolveContext`.

The cluster sweeps verify ``load_flextool`` end-to-end; this file pins
down the SolveContext API itself: typed fields read from the provider,
and the provider-only contract (``provider=None`` raises, a CSV in the
work folder is never read).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.preprocess import _solve_context as sc
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess._solve_context import SolveContext
from flextool.engine.preprocess.params import _derived
from tests._support.provider import provider_from_csv_dirs


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
    provider = provider_from_csv_dirs(
        wd / "input" if (wd / "input").is_dir() else None,
        wd / "solve_data" if (wd / "solve_data").is_dir() else None,
    )
    # Empty placeholders for the SolveContext lazy carriers — only
    # seeded when the on-disk seed didn't supply them.
    for stem in ("period_in_use_set", "period__branch", "edd_history",
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
# Provider-only contract (c24 stage 3): no disk arm, provider=None raises
# ---------------------------------------------------------------------------


def test_from_workdir_requires_provider(workdir: Path) -> None:
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        SolveContext.from_workdir(workdir)


@pytest.mark.parametrize("loader", ["_maybe_read", "_read_frame"])
def test_frame_loaders_require_provider(workdir: Path, loader: str) -> None:
    path = workdir / "solve_data" / "edd_history.csv"
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        getattr(sc, loader)(path)


def test_period_share_requires_provider(workdir: Path) -> None:
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        sc._load_period_share(workdir / "solve_data")


def test_read_active_solve_requires_provider(workdir: Path) -> None:
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        sc._read_active_solve(workdir)


def test_lazy_field_without_provider_raises(workdir: Path) -> None:
    """A hand-built context without a provider cannot fall back to the CSVs on disk."""
    ctx = SolveContext(workdir=workdir)
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        ctx.period_in_use


def test_csv_on_disk_is_not_read(workdir: Path) -> None:
    """The fixture's CSVs sit on disk; an empty provider sees none of them."""
    provider = FlexDataProvider()
    assert sc._read_active_solve(workdir, provider=provider) is None
    ctx = SolveContext.from_workdir(workdir, provider=provider)
    assert ctx.solve_name is None
    assert ctx.realized_periods == set()
    assert ctx.period_share_of_year.height == 0
    with pytest.raises(sc.FlexDataError):
        ctx.period_in_use


def test_ctx_read_requires_provider(workdir: Path) -> None:
    with pytest.raises(ValueError, match="requires a FlexDataProvider"):
        _derived._ctx_read(None, workdir, "period__branch.csv")


def test_ctx_read_reads_the_provider_not_disk(workdir: Path) -> None:
    """Disk has ``period__branch.csv``; the provider decides what ``_ctx_read`` returns."""
    empty = FlexDataProvider()
    assert _derived._ctx_read(None, workdir, "period__branch.csv", provider=empty) is None
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    assert _derived._ctx_read(ctx, workdir, "period__branch.csv", provider=empty) is None
    frame = pl.DataFrame({"period": ["p2030"], "branch": ["p2030"]})
    provider = FlexDataProvider()
    provider.put("solve_data/period__branch", frame)
    got = _derived._ctx_read(None, workdir, "period__branch.csv", provider=provider)
    assert got is not None and got["period"].to_list() == ["p2030"]
    assert _derived._ctx_read(None, None, "period__branch.csv", provider=provider) is None


def test_solve_data_dir_property(workdir: Path) -> None:
    ctx = SolveContext.from_workdir(workdir, provider=_seed_workdir_provider(workdir))
    assert ctx.solve_data_dir == workdir / "solve_data"
    assert ctx.input_dir == workdir / "input"
