"""Phase 3 — Provider carrier round-trip tests.

After Phase 3 migration, ``SolveContext`` consumes its per-solve
preprocessing carriers directly from the :class:`FlexDataProvider`
instead of re-reading the writer-produced CSVs.  This module verifies:

* Each carrier round-trips through the Provider (writer-side ``put``
  → reader-side ``SolveContext`` attribute access).
* A missing carrier raises :class:`FlexDataError` with an informative
  message naming the carrier.

The tests build minimal in-memory frames matching each carrier's
canonical schema and seed them via :meth:`FlexDataProvider.put` under
the ``"solve_data/<stem>"`` key — the same key the
``capture_frames`` monkey-patch uses when routing writer emissions.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._solve_context import (
    FlexDataError,
    SolveContext,
    _load_edd_history,
    _load_period_branch,
    _load_period_in_use,
    _load_period_share,
    _load_solve_branch_weight,
    _load_steps_in_use,
    _maybe_read,
)


# ---------------------------------------------------------------------------
# Helpers — match the canonical writer output of each carrier.
# ---------------------------------------------------------------------------


def _seed(provider: FlexDataProvider, name: str, frame: pl.DataFrame) -> None:
    """Seed *provider* under the canonical ``solve_data/<stem>`` key."""
    provider.put(f"solve_data/{name}", frame)


# ---------------------------------------------------------------------------
# A) SolveContext lazy DataFrame loaders — one round-trip per carrier.
# ---------------------------------------------------------------------------


def test_period_in_use_round_trip(tmp_path):
    provider = FlexDataProvider()
    _seed(
        provider,
        "period_in_use_set",
        pl.DataFrame({"period": ["d2025", "d2030", "d2025"]}),
    )
    path = tmp_path / "solve_data" / "period_in_use_set.csv"
    out = _load_period_in_use(path, provider=provider)
    assert out.columns == ["d"]
    assert out["d"].to_list() == ["d2025", "d2030"]


def test_period_branch_round_trip(tmp_path):
    provider = FlexDataProvider()
    _seed(
        provider,
        "period__branch",
        pl.DataFrame({"period": ["d2025", "d2025"], "branch": ["b0", "b1"]}),
    )
    path = tmp_path / "solve_data" / "period__branch.csv"
    out = _load_period_branch(path, provider=provider)
    assert out.columns == ["d_anchor", "b"]
    assert out["d_anchor"].to_list() == ["d2025", "d2025"]
    assert out["b"].to_list() == ["b0", "b1"]


def test_edd_history_round_trip(tmp_path):
    provider = FlexDataProvider()
    frame = pl.DataFrame(
        {"entity": ["e1"], "period_history": ["d2025"], "period": ["d2030"]},
    )
    _seed(provider, "edd_history", frame)
    path = tmp_path / "solve_data" / "edd_history.csv"
    out = _load_edd_history(path, provider=provider)
    assert out.equals(frame)


def test_steps_in_use_round_trip(tmp_path):
    provider = FlexDataProvider()
    _seed(
        provider,
        "steps_in_use",
        pl.DataFrame(
            {
                "period": ["d2025", "d2025"],
                "step": ["t0001", "t0002"],
                "step_duration": ["1.0", "1.0"],
            },
        ),
    )
    path = tmp_path / "solve_data" / "steps_in_use.csv"
    out = _load_steps_in_use(path, provider=provider)
    assert out.columns == ["d", "t", "step_duration"]
    assert out.height == 2
    assert out.schema["step_duration"] == pl.Float64


def test_period_share_round_trip(tmp_path):
    provider = FlexDataProvider()
    _seed(
        provider,
        "complete_period_share_of_year_calc",
        pl.DataFrame({"period": ["d2025"], "value": ["1.0"]}),
    )
    sd_dir = tmp_path / "solve_data"
    out = _load_period_share(sd_dir, provider=provider)
    assert out.columns == ["d", "value"]
    assert out["value"].to_list() == [1.0]


def test_solve_branch_weight_round_trip(tmp_path):
    provider = FlexDataProvider()
    _seed(
        provider,
        "solve_branch_weight",
        pl.DataFrame(
            {"branch": ["b0", "b1"], "p_branch_weight_input": ["0.5", "0.5"]},
        ),
    )
    path = tmp_path / "solve_data" / "solve_branch_weight.csv"
    out = _load_solve_branch_weight(path, provider=provider)
    assert out.columns == ["b", "p_branch_weight_input"]
    assert out["p_branch_weight_input"].to_list() == [0.5, 0.5]


def test_maybe_read_round_trip(tmp_path):
    """The generic ``_maybe_read`` loader (powers
    p_entity_period_existing_capacity + p_entity_pre_existing +
    p_entity_all_existing) routes through the Provider too."""
    provider = FlexDataProvider()
    frame = pl.DataFrame(
        {"entity": ["e1"], "period": ["d2025"], "value": [42.0]},
    )
    _seed(provider, "p_entity_pre_existing", frame)
    path = tmp_path / "solve_data" / "p_entity_pre_existing.csv"
    out = _maybe_read(path, provider=provider)
    assert out.equals(frame)


# ---------------------------------------------------------------------------
# B) Strict semantics — a missing carrier raises an informative error.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader, fname",
    [
        (_load_period_in_use, "period_in_use_set.csv"),
        (_load_period_branch, "period__branch.csv"),
        (_load_edd_history, "edd_history.csv"),
        (_load_steps_in_use, "steps_in_use.csv"),
        (_load_solve_branch_weight, "solve_branch_weight.csv"),
    ],
)
def test_loader_raises_when_provider_missing_carrier(tmp_path, loader, fname):
    """Each Provider-strict loader raises FlexDataError on miss, with the
    carrier filename mentioned in the message."""
    provider = FlexDataProvider()
    path = tmp_path / "solve_data" / fname
    with pytest.raises(FlexDataError) as exc:
        loader(path, provider=provider)
    msg = str(exc.value)
    assert fname in msg, f"error message must name carrier {fname}, got: {msg}"


def test_maybe_read_raises_when_provider_missing_carrier(tmp_path):
    provider = FlexDataProvider()
    path = tmp_path / "solve_data" / "p_entity_pre_existing.csv"
    with pytest.raises(FlexDataError) as exc:
        _maybe_read(path, provider=provider)
    assert "p_entity_pre_existing.csv" in str(exc.value)


# ---------------------------------------------------------------------------
# C) SolveContext property accessor goes through provider.
# ---------------------------------------------------------------------------


def test_solve_context_lazy_property_uses_provider(tmp_path):
    """SolveContext stashes the provider; property access routes through
    it for the lazy DataFrame fields."""
    provider = FlexDataProvider()
    _seed(
        provider,
        "period_in_use_set",
        pl.DataFrame({"period": ["d2025"]}),
    )
    _seed(
        provider,
        "period__branch",
        pl.DataFrame({"period": ["d2025"], "branch": ["b0"]}),
    )
    workdir = tmp_path / "wd"
    (workdir / "solve_data").mkdir(parents=True)
    ctx = SolveContext(workdir=workdir, provider=provider)
    # period_in_use reads from the Provider — no file on disk exists.
    assert ctx.period_in_use["d"].to_list() == ["d2025"]
    # period_branch likewise.
    assert ctx.period_branch["d_anchor"].to_list() == ["d2025"]
    assert ctx.period_branch["b"].to_list() == ["b0"]


def test_solve_context_missing_carrier_raises(tmp_path):
    """Provider-supplied SolveContext raises when the requested carrier
    is absent (Phase 3 strict semantics — no silent disk fallback)."""
    provider = FlexDataProvider()
    # Seed period_in_use but NOT period__branch.
    _seed(
        provider,
        "period_in_use_set",
        pl.DataFrame({"period": ["d2025"]}),
    )
    workdir = tmp_path / "wd"
    (workdir / "solve_data").mkdir(parents=True)
    ctx = SolveContext(workdir=workdir, provider=provider)
    # period_in_use should work.
    _ = ctx.period_in_use
    # period_branch should fail loudly.
    with pytest.raises(FlexDataError) as exc:
        _ = ctx.period_branch
    assert "period__branch.csv" in str(exc.value)


# ---------------------------------------------------------------------------
# D) Legacy disk path still works when provider is None.
# ---------------------------------------------------------------------------


def test_loader_disk_path_when_provider_is_none(tmp_path):
    """The legacy disk-fallback path remains when ``provider=None`` —
    this is the test-only contract that goes away with Phase 4."""
    workdir = tmp_path / "wd"
    sd = workdir / "solve_data"
    sd.mkdir(parents=True)
    pl.DataFrame({"period": ["d2025"]}).write_csv(sd / "period_in_use_set.csv")
    out = _load_period_in_use(sd / "period_in_use_set.csv", provider=None)
    assert out.columns == ["d"]
    assert out["d"].to_list() == ["d2025"]


def test_loader_disk_path_missing_file_returns_empty(tmp_path):
    """Disk path — missing file yields a typed empty frame (legacy
    contract, retained for the ``provider=None`` test path)."""
    out = _load_period_in_use(
        tmp_path / "no_such_file.csv", provider=None,
    )
    assert out.height == 0
    assert out.columns == ["d"]
