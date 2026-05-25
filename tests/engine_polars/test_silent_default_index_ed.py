"""Silent-default index regression — Tier-2 ``ed_*`` multi-class union helpers.

The six helpers below (``ed_invest_max_period``, ``ed_divest_max_period``,
``ed_invest_min_period``, ``ed_divest_min_period``,
``ed_cumulative_max_capacity``, ``ed_cumulative_min_capacity``) union
``parameter`` rows across ``unit / node / connection`` into a single
``Param(("e", "d"))`` frame keyed on the entity union axis.

Pre-migration, the underlying ``_e_period_param_union`` helper had a
partial ``"x" → "period"`` workaround but:

1. didn't value-domain probe, so silent-default Maps whose ``x`` column
   contained timestep tokens (rather than period tokens) would have
   misrouted — low risk for these specific parameters in practice, but
   the resolver does it right;
2. silently DROPPED scalar-authored entities (``period_col`` was
   ``None``, so the row was skipped entirely) — so a user-authored
   ``unit.invest_max_period = 5`` would never reach the LP.

Δ.17c-Tier2 routes each class through
:func:`flextool.engine_polars._param_shapes.resolve_param_shape` +
:func:`flextool.engine_polars._param_shapes.broadcast_to_period`.  The
``{SCALAR, MAP_PERIOD}`` allow-list resolves silent-default
``index_name`` at depth 1 structurally; SCALAR rows are now broadcast
across the active solve's periods.

Cyprus DB authoring snapshot (2026-05-25): the Cyprus migration test
fixture ``/tmp/cyprus_test.sqlite`` has ZERO authored rows for these
six parameters across ``unit / node / connection``.  This test
therefore guards the dedup-invariant (no duplicate ``(e, d)`` keys)
across the migrated helpers and verifies the SCALAR-broadcast path
against a minimal synthetic stub source (Cyprus can't exercise it
end-to-end for these params).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


_CYPRUS_DB = Path("/tmp/cyprus_test.sqlite")
_SCENARIO = "D 5W 24-30_35_40_45_50"
_PERIOD_NAMES = ["p2024", "p2025", "p2026", "p2027", "p2028", "p2029",
                 "p2030", "p2035", "p2040", "p2045", "p2050"]


_ED_HELPERS = [
    "ed_invest_max_period_from_source",
    "ed_divest_max_period_from_source",
    "ed_invest_min_period_from_source",
    "ed_divest_min_period_from_source",
    "ed_cumulative_max_capacity_from_source",
    "ed_cumulative_min_capacity_from_source",
]


@pytest.fixture(scope="module")
def cyprus_source():
    """SpineDbReader against the Cyprus migrated DB.

    Skips when the DB isn't present on this host (CI / other devs).
    """
    if not _CYPRUS_DB.exists():
        pytest.skip(f"Cyprus DB not present at {_CYPRUS_DB}")
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    return SpineDbReader(f"sqlite:///{_CYPRUS_DB}", _SCENARIO)


@pytest.fixture(scope="module")
def period_filter() -> pl.DataFrame:
    """A ``[d, t]`` frame covering the scenario's periods (one timestep
    per period; only the ``d`` column matters for these helpers).
    """
    return pl.DataFrame({
        "d": _PERIOD_NAMES,
        "t": ["t00001"] * len(_PERIOD_NAMES),
    })


def _collect(param) -> pl.DataFrame:
    """Materialise a ``Param.frame`` whether it's eager or lazy."""
    frame = param.frame
    return frame.collect() if hasattr(frame, "collect") else frame


def _assert_no_e_d_duplicates(param) -> None:
    """No two rows share the same ``(e, d)`` key.

    Pre-migration the union helper would have surfaced duplicates only
    if BOTH the silent-default ``"x"`` workaround AND a same-named entity
    in another class were authored; Cyprus doesn't exercise that combo.
    Post-migration the resolver's structural dedup guarantees uniqueness
    regardless.  This assertion is the safety net.
    """
    frame = _collect(param)
    frame = frame.with_columns(
        pl.col("e").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
    )
    dupes = (
        frame.group_by(["e", "d"])
             .len()
             .filter(pl.col("len") > 1)
    )
    assert dupes.height == 0, (
        f"Duplicate (e, d) keys after Tier-2 union:\n{dupes.head(10)}"
    )


@pytest.mark.parametrize("helper_name", _ED_HELPERS)
def test_cyprus_ed_helper_no_duplicates(
    cyprus_source, period_filter, helper_name,
) -> None:
    """Each of the six Tier-2 helpers must return either ``None``
    (legitimately empty for the Cyprus scenario; ZERO authored rows
    confirmed by SQL inspection on 2026-05-25) or a Param with no
    duplicate ``(e, d)`` keys.
    """
    import flextool.engine_polars._direct_params as dp
    fn = getattr(dp, helper_name)
    result = fn(cyprus_source, period_filter=period_filter)
    if result is None:
        # Legitimately empty — Cyprus doesn't author this parameter on
        # any of unit / node / connection.  The pre-migration helper
        # would also return None for the no-authoring case (no
        # ``period_col`` AND no entity loop iterations).
        return
    assert result.dims == ("e", "d"), (
        f"{helper_name} returned a Param with unexpected dims "
        f"{result.dims}; expected ('e', 'd')."
    )
    _assert_no_e_d_duplicates(result)


@pytest.mark.parametrize("helper_name", _ED_HELPERS)
def test_cyprus_ed_helper_returns_none(
    cyprus_source, period_filter, helper_name,
) -> None:
    """Document the Cyprus authoring state: zero rows for the Tier-2
    parameters across ``unit / node / connection`` means each helper
    returns ``None``.  This test pins that observation so a future
    fixture update that introduces authoring surfaces explicitly
    instead of being absorbed silently by the dedup test above.

    If this test FAILS post-fixture-update, that's good news — but the
    test must then be updated to lock the actual period → value
    associations Cyprus introduces.
    """
    import flextool.engine_polars._direct_params as dp
    fn = getattr(dp, helper_name)
    result = fn(cyprus_source, period_filter=period_filter)
    assert result is None, (
        f"Cyprus authored {helper_name!r} (expected None per the "
        f"2026-05-25 audit).  Result frame:\n{_collect(result)}"
    )


# ---------------------------------------------------------------------------
# SCALAR-broadcast regression — synthetic InputSource
#
# Cyprus carries no Map / scalar authoring for any of the six Tier-2
# parameters across ``unit / node / connection``.  To lock the
# scalar-broadcast invariant (pre-migration this row was DROPPED, post-
# migration it broadcasts across all active periods) we use a minimal
# stub ``InputSource`` that surfaces a scalar row for one
# ``(class, entity, param)`` triple.
# ---------------------------------------------------------------------------


class _ScalarStubSource:
    """Minimal InputSource that surfaces a single scalar row.

    Mirrors the stub idiom used in ``test_a04_a05_online_ramp.py`` and
    the Phase-1 inertia regressions.  Only overrides what the resolver
    consults: ``parameter_explicit``, ``parameter_shape_info``, and
    ``entities`` (for the entity-dim discovery in
    :func:`_entity_dim_columns_for_frame`).
    """

    def __init__(self, entity_class: str, parameter_name: str,
                 entity_name: str, value: float) -> None:
        self._entity_class = entity_class
        self._parameter_name = parameter_name
        self._entity_name = entity_name
        self._value = value

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if (entity_class == self._entity_class
                and parameter_name == self._parameter_name):
            return pl.DataFrame({
                "name": [self._entity_name],
                "value": [self._value],
            })
        raise KeyError((entity_class, parameter_name))

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        # Scalar — no index labels.
        return []

    def entities(self, entity_class: str):
        if entity_class == self._entity_class:
            return pl.DataFrame({"name": [self._entity_name]})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


@pytest.mark.parametrize("helper_name,entity_class,parameter_name", [
    ("ed_invest_max_period_from_source", "unit", "invest_max_period"),
    ("ed_divest_max_period_from_source", "node", "retire_max_period"),
    ("ed_invest_min_period_from_source", "connection", "invest_min_period"),
    ("ed_divest_min_period_from_source", "unit", "retire_min_period"),
    ("ed_cumulative_max_capacity_from_source", "node", "cumulative_max_capacity"),
    ("ed_cumulative_min_capacity_from_source", "connection", "cumulative_min_capacity"),
])
def test_scalar_authored_broadcasts_across_periods(
    period_filter, helper_name, entity_class, parameter_name,
) -> None:
    """Pre-migration: scalar-authored rows (no Map index) were SILENTLY
    DROPPED because ``period_col`` was ``None`` and the helper
    ``continue``-d past the entity.

    Post-migration: ``resolve_param_shape`` detects ``Shape.SCALAR`` and
    ``broadcast_to_period`` cross-joins with the active solve's period
    axis, emitting one row per active period.  We assert all 11
    Cyprus-scenario periods are present and carry the authored value.
    """
    import flextool.engine_polars._direct_params as dp
    # filter_zero on invest_min / retire_min / cumulative_* would drop
    # value=0.0 — pick a nonzero stub value for ALL helpers so the
    # regression is uniform.
    stub = _ScalarStubSource(entity_class, parameter_name,
                              entity_name=f"stub_{entity_class}",
                              value=7.0)
    fn = getattr(dp, helper_name)
    result = fn(stub, period_filter=period_filter)
    assert result is not None, (
        f"{helper_name} dropped a scalar-authored row — pre-migration "
        f"helper bug (silent ``continue`` on period_col=None)."
    )
    assert result.dims == ("e", "d"), result.dims
    frame = _collect(result).with_columns(
        pl.col("e").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
    )
    rows = frame.filter(pl.col("e") == f"stub_{entity_class}").sort("d")
    assert rows.height == len(_PERIOD_NAMES), (
        f"Scalar broadcast should emit one row per active period "
        f"({len(_PERIOD_NAMES)}); got {rows.height}:\n{rows}"
    )
    values = rows.get_column("value").to_list()
    assert all(v == 7.0 for v in values), (
        f"All scalar-broadcast values should be 7.0; got {values}"
    )
    got_periods = set(rows.get_column("d").to_list())
    assert got_periods == set(_PERIOD_NAMES), (
        f"Scalar broadcast covered {got_periods}, expected "
        f"{set(_PERIOD_NAMES)}"
    )


# ---------------------------------------------------------------------------
# Silent-default Map regression — synthetic InputSource
#
# Cyprus doesn't author these parameters as Maps either, so we
# additionally lock the silent-default ``index_name`` ("x") resolution
# path against a stub source that mimics what spinedb_api emits when
# the author omitted ``index_name``.
# ---------------------------------------------------------------------------


class _SilentDefaultMapStubSource:
    """Surfaces a 1d_map(period) authored with the silent-default
    ``index_name`` ``"x"``.

    The SpineDbReader fallback names the index column ``"x"`` (not
    ``"period"``) in this case.  Pre-migration ``_e_period_param_union``
    had a partial ``"x" → "period"`` workaround so this test wouldn't
    have failed on the dedup invariant pre-migration; what it locks is
    that the post-migration resolver path (via
    ``_infer_silent_default_labels`` + structural depth-1
    disambiguation) lands on the same period → value mapping.
    """

    def __init__(self, entity_class: str, parameter_name: str,
                 entity_name: str, period_values: dict[str, float]) -> None:
        self._entity_class = entity_class
        self._parameter_name = parameter_name
        self._entity_name = entity_name
        self._period_values = period_values

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        if (entity_class == self._entity_class
                and parameter_name == self._parameter_name):
            periods = list(self._period_values.keys())
            values = [self._period_values[p] for p in periods]
            return pl.DataFrame({
                "name": [self._entity_name] * len(periods),
                # silent-default index_name → column named "x"
                "x": periods,
                "value": values,
            })
        raise KeyError((entity_class, parameter_name))

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        # Silent default — spinedb_api emits "x".
        return ["x"]

    def entities(self, entity_class: str):
        if entity_class == self._entity_class:
            return pl.DataFrame({"name": [self._entity_name]})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


def test_silent_default_index_resolves_to_period(period_filter) -> None:
    """A 1d_map authored with silent-default ``index_name`` ("x") and
    period-token keys must resolve to ``Shape.MAP_PERIOD`` and lock the
    authored period → value association — no scalar cross-join, no
    dropped rows.
    """
    import flextool.engine_polars._direct_params as dp
    period_values = {
        "p2024": 1.5,
        "p2025": 2.5,
        "p2030": 3.5,
    }
    stub = _SilentDefaultMapStubSource(
        entity_class="unit",
        parameter_name="invest_max_period",
        entity_name="stub_unit_map",
        period_values=period_values,
    )
    result = dp.ed_invest_max_period_from_source(
        stub, period_filter=period_filter)
    assert result is not None, (
        "ed_invest_max_period_from_source returned None for a "
        "silent-default Map source; expected a populated Param."
    )
    assert result.dims == ("e", "d")
    frame = _collect(result).with_columns(
        pl.col("e").cast(pl.Utf8),
        pl.col("d").cast(pl.Utf8),
    )
    rows = frame.filter(pl.col("e") == "stub_unit_map").sort("d")
    # Active periods are the 11 in period_filter; only 3 are authored.
    assert rows.height == len(period_values), (
        f"Map-authored entity should surface exactly {len(period_values)} "
        f"rows (the authored periods), got {rows.height}:\n{rows}"
    )
    got = dict(zip(
        rows.get_column("d").to_list(),
        rows.get_column("value").to_list(),
    ))
    assert got == period_values, (
        f"Period → value association lost: got {got}, expected "
        f"{period_values}"
    )
    _assert_no_e_d_duplicates(result)
