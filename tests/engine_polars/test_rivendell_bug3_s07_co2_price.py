"""Rivendell bug 3 regression — S07_co2price_slice ``p_co2_price`` resolver.

The Rivendell ``S07_co2price_slice`` scenario regressed (HEAD ``fcfc0d11``)
with::

    ValueError: build_flextool: feature 'co2_price' is active but
    data fields are not populated (None): ['p_co2_price'].  Either
    fill them in the data or don't enable the feature.

Root cause: ``group.co2_price`` in the Rivendell DB is authored as a 1d
Map whose ``index_name`` is the spinedb_api silent default (``"x"``),
NOT an explicit ``"period"`` or ``"time"``.  Unlike ``co2_max_period``
(see ``test_phase_e_i_s08_co2cap_smoke.py``) whose allow-list
``{SCALAR, MAP_PERIOD}`` admits a unique shape at depth 1 (so the
silent default disambiguates structurally), ``co2_price`` admits
``{SCALAR, MAP_PERIOD, MAP_TIME, MAP_PERIOD_TIME}`` — depth 1 covers
both ``MAP_PERIOD`` and ``MAP_TIME``, so the silent default cannot be
resolved structurally.

The fix: when structural inference is ambiguous, the resolver probes
the Map's actual index values against the active solve's known periods
/ timesteps (mirroring the legacy CSV pipeline's
:func:`flextool.engine_polars._timeline.separate_period_and_timeseries_data`
discriminator).  A Map keyed by ``y2019, y2020, ...`` is unambiguously
:class:`Shape.MAP_PERIOD`; a Map keyed by ``t00001, t00002, ...`` is
:class:`Shape.MAP_TIME`.

This test exercises the offending resolver path directly — no full LP
solve required.  It fails on pre-fix HEAD (returns ``None``) and passes
after the fix (returns a populated ``Param``).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._direct_params import (
    p_co2_price_from_source,
)
from flextool.engine_polars._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    Shape,
    _disambiguate_shape_by_value_domain,
    resolve_param_shape,
)


# ---------------------------------------------------------------------------
# A. Helper-level — value-domain disambiguation (no DB required).
# ---------------------------------------------------------------------------


def _frame_with_index(values: list[str]) -> pl.DataFrame:
    """Build a (name, x, value) parameter frame with the given index
    values — mirrors what :meth:`SpineDbReader.parameter` returns for a
    1d Map whose index_name is the silent default ``"x"``.
    """
    return pl.DataFrame({
        "name":  ["g0"] * len(values),
        "x":     values,
        "value": [float(i) for i in range(len(values))],
    })


def test_value_domain_disambiguates_period_indexed_map() -> None:
    """A Map keyed by ``y2019, y2020, y2021`` against a period_filter
    whose ``d`` column lists those years resolves to ``MAP_PERIOD``.
    """
    df = _frame_with_index(["y2019", "y2020", "y2021"])
    period_filter = pl.DataFrame({
        "d": ["y2019", "y2020", "y2021"],
        "t": ["t00001", "t00001", "t00001"],
    })
    allowed = {Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
               Shape.MAP_PERIOD_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape == Shape.MAP_PERIOD


def test_value_domain_disambiguates_time_indexed_map() -> None:
    """A Map keyed by ``t00001, t00002`` against the same filter
    resolves to ``MAP_TIME``.
    """
    df = _frame_with_index(["t00001", "t00002"])
    period_filter = pl.DataFrame({
        "d": ["y2019", "y2020"],
        "t": ["t00001", "t00002"],
    })
    allowed = {Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_TIME,
               Shape.MAP_PERIOD_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape == Shape.MAP_TIME


def test_value_domain_no_filter_returns_none() -> None:
    """Without a ``period_filter`` (off-cascade call sites), the
    fallback cannot probe and returns ``None`` — the caller must then
    accept ambiguity (resolver returns None, field drops from FlexData).
    """
    df = _frame_with_index(["y2019"])
    allowed = {Shape.MAP_PERIOD, Shape.MAP_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=None,
    )
    assert shape is None


def test_value_domain_mixed_values_returns_none() -> None:
    """When index values match NEITHER the period set NOR the timestep
    set exclusively, we don't guess: the resolver stays at ``None`` and
    the caller falls back.  Guards against silently misclassifying
    fixtures with novel indexing.
    """
    df = _frame_with_index(["foo", "bar"])
    period_filter = pl.DataFrame({
        "d": ["y2019"],
        "t": ["t00001"],
    })
    allowed = {Shape.MAP_PERIOD, Shape.MAP_TIME}
    shape = _disambiguate_shape_by_value_domain(
        df, ent_cols=["name"], resolved_labels=[None],
        allowed=allowed, period_filter=period_filter,
    )
    assert shape is None


# ---------------------------------------------------------------------------
# B. End-to-end on the Rivendell DB — the S07 regression case.
# ---------------------------------------------------------------------------


_RIVENDELL_DB = Path(
    "/home/jkiviluo/sources/flextool-engine/projects/Rivendell/input_sources/rivendell.sqlite"
)


@pytest.fixture(scope="module")
def rivendell_source_s07():
    """SpineDbReader against Rivendell scenario ``S07_co2price_slice``.

    Skips when the DB isn't checked out (e.g. CI doesn't ship it).
    """
    if not _RIVENDELL_DB.exists():
        pytest.skip(f"Rivendell DB not present at {_RIVENDELL_DB}")
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    return SpineDbReader(f"sqlite:///{_RIVENDELL_DB}", "S07_co2price_slice")


def test_s07_co2_price_has_silent_default_index_name(
    rivendell_source_s07,
) -> None:
    """Sanity-check: the DB authoring really uses the silent default
    ``"x"`` for ``group.co2_price``.  If a future DB regen sets
    ``index_name="period"`` explicitly, this test becomes a no-op
    (structural inference handles it) and can be removed alongside the
    rest of the suite.
    """
    info = rivendell_source_s07.parameter_shape_info(
        "group", "co2_price")
    assert info == ["x"], (
        f"Rivendell S07 DB Map index_name is {info!r}; "
        "the regression test assumes the silent default 'x'. "
        "If the DB has been re-authored with an explicit 'period' "
        "index_name, remove this assertion."
    )


def test_s07_co2_price_allowlist_admits_both_period_and_time() -> None:
    """The fix only matters because ``("group", "co2_price")`` admits
    BOTH ``MAP_PERIOD`` and ``MAP_TIME`` at depth 1 — structural
    disambiguation (``_infer_silent_default_labels``) cannot fix this
    case; value-domain probing must.
    """
    allowed = PARAM_ALLOWED_SHAPES[("group", "co2_price")]
    assert Shape.MAP_PERIOD in allowed
    assert Shape.MAP_TIME in allowed


def test_s07_resolve_param_shape_with_period_filter_returns_map_period(
    rivendell_source_s07,
) -> None:
    """Pre-fix: ``resolve_param_shape("group", "co2_price")`` returns
    ``None`` — silent default at depth 1 with two allowed shapes is
    structurally ambiguous.  Post-fix: passing a ``period_filter``
    whose ``d`` column carries the active periods triggers value-domain
    probing and the resolver returns a ``ResolvedShape(MAP_PERIOD)``
    with the silent-default ``"x"`` column renamed to ``"period"``.
    """
    # Pre-fix arm: no period_filter ⇒ still None (genuine ambiguity).
    resolved_no_filter = resolve_param_shape(
        rivendell_source_s07, "group", "co2_price")
    assert resolved_no_filter is None, (
        "Off-cascade resolver call (no period_filter) MUST still return "
        "None for the genuinely ambiguous structural case — the fix only "
        "activates when the consumer site supplies a period_filter."
    )

    # The Rivendell DB declares periods y2019..y2050 — synthesise a
    # period_filter that covers (a superset of) the Map's keys.
    period_names = [f"y{2019 + i}" for i in range(32)]
    period_filter = pl.DataFrame({
        "d": period_names,
        "t": ["t00001"] * len(period_names),
    })
    resolved = resolve_param_shape(
        rivendell_source_s07, "group", "co2_price",
        period_filter=period_filter,
    )
    assert resolved is not None, (
        "Pre-fix HEAD returned None here — value-domain probing was "
        "missing.  Post-fix the resolver consults the period_filter's "
        "`d` column and recognises the Map keys (y2019, ...) as periods."
    )
    assert resolved.shape == Shape.MAP_PERIOD
    assert resolved.period_index_column == "period", (
        f"Frame columns: {resolved.frame.columns}; "
        "expected the silent-default 'x' column to be renamed to 'period'."
    )
    assert "period" in resolved.frame.columns
    assert resolved.frame.height > 0


def test_s07_p_co2_price_from_source_is_populated(
    rivendell_source_s07,
) -> None:
    """End-to-end resolver assertion: the consumer site in
    ``apply_direct_params_b`` must receive a non-None ``Param`` for
    ``p_co2_price`` so that ``build_flextool`` doesn't raise on the
    ``co2_price`` feature gate.
    """
    period_names = [f"y{2019 + i}" for i in range(32)]
    dt = pl.DataFrame({
        "d": period_names,
        "t": ["t00001"] * len(period_names),
    })
    result = p_co2_price_from_source(
        rivendell_source_s07, period_filter=dt)
    assert result is not None, (
        "Pre-fix HEAD: returned None → build_flextool raised "
        "'co2_price feature active but p_co2_price is None'."
    )
    frame = result.frame.collect() if hasattr(result.frame, "collect") else result.frame
    assert {"g", "d", "t", "value"}.issubset(set(frame.columns)), (
        f"Param frame columns: {frame.columns}"
    )
    assert frame.height > 0
    assert (frame.select(pl.col("g") == "co2_group").to_series().any())


# ---------------------------------------------------------------------------
# C. Full cascade smoke — guard against regressions in the consumer wiring.
# ---------------------------------------------------------------------------


def test_s07_native_cascade_reaches_solver(tmp_path: Path) -> None:
    """The whole point: with the resolver fix in place, the S07 scenario
    must get past ``build_flextool``'s ``CO2_PRICE`` invariant when run
    through the standard ``run_chain_from_db`` entry point (which the
    ``cmd_run_flextool`` CLI also uses).  We don't assert objective
    parity here — just that the LP builds and the solve completes.
    """
    if not _RIVENDELL_DB.exists():
        pytest.skip(f"Rivendell DB not present at {_RIVENDELL_DB}")
    from flextool.engine_polars._orchestration import run_chain_from_db

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    steps = run_chain_from_db(
        input_db_url=f"sqlite:///{_RIVENDELL_DB}",
        scenario_name="S07_co2price_slice",
        work_folder=work_dir,
    )
    # Pre-fix HEAD raised on build_flextool before any step completed.
    # Post-fix at least one step is reported with optimal=True.
    assert steps, "no orchestration steps produced — cascade aborted early"
    for solve_name, step in steps.items():
        assert step.optimal, (
            f"solve {solve_name!r} did not reach an optimal solution "
            f"(obj={step.obj}); expected the LP build to succeed "
            "after the resolver fix populates p_co2_price."
        )
