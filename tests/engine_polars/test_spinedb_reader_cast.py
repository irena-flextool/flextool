"""Phase 2 tests for :class:`SpineDbReader` cast-on-emit.

The reader's constructor gains optional ``axis_enums`` + ``contract``
kwargs.  When supplied, every frame returned by :meth:`entities`,
:meth:`parameter`, :meth:`parameter_explicit` has its dim columns cast
to the canonical :class:`pl.Enum` dtypes.

Map-inner-key dim columns (axis ``source_type == "parameter_keys"``)
are silently filtered to the scenario-active vocabulary so Map data
authored beyond the active scenario does not break the strict cast.
Entity-class dim columns remain strict — vocabulary misses there
raise :class:`FlexDataIntegrityError` (covered by the backend-level
breadcrumb suite in ``tests/spinedb_backend/test_backend_cast.py``).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import SpineDbReader
from flextool.spinedb_backend import SpineDBBackend
from flextool.spinedb_backend._axis_enums import (
    AxisContract,
    build_axis_enums,
    load_axis_contract,
)


@pytest.fixture(scope="module")
def TEST_A_LOT_DB(scenario_workdir) -> Path:
    return scenario_workdir("test_a_lot") / "tests.sqlite"


# ---------------------------------------------------------------------------
# Module-scoped fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contract() -> AxisContract:
    return load_axis_contract()


@pytest.fixture(scope="module")
def axis_enums(TEST_A_LOT_DB: Path, contract: AxisContract) -> dict[str, pl.Enum]:
    """Build axis enums against an unscoped Backend.

    The Phase-1 scenario-filter session-binding blocker means we open
    the Backend without a scenario for vocabulary discovery.  Enum
    vocabulary is the union across scenarios anyway — see the
    test_axis_enums module docstring.
    """
    b = SpineDBBackend(str(TEST_A_LOT_DB))
    try:
        return build_axis_enums(b, contract)
    finally:
        b.close()


# ---------------------------------------------------------------------------
# 1. Back-compat: no kwargs → Utf8.
# ---------------------------------------------------------------------------


def test_spinedb_reader_returns_utf8_when_axis_enums_none(
    TEST_A_LOT_DB: Path,
) -> None:
    """Pre-Phase-2 behaviour: constructed without ``axis_enums``, every
    dim column is Utf8."""
    r = SpineDbReader(str(TEST_A_LOT_DB), "test_a_lot")
    frame = r.entities("node__profile")
    assert frame.schema["node"] == pl.Utf8
    assert frame.schema["profile"] == pl.Utf8


# ---------------------------------------------------------------------------
# 2. Happy path: axis_enums supplied → Enum dim columns.
# ---------------------------------------------------------------------------


def test_spinedb_reader_entities_returns_enum_when_axis_enums_supplied(
    TEST_A_LOT_DB: Path,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """Constructed with ``axis_enums``, :meth:`entities` casts the
    ``node`` AND ``profile`` columns of the ``node__profile`` multi-dim
    class — both are recognised column synonyms in the contract
    (``node → n``, ``profile → f``).
    """
    r = SpineDbReader(
        str(TEST_A_LOT_DB),
        "test_a_lot",
        axis_enums=axis_enums,
        contract=contract,
    )
    frame = r.entities("node__profile")
    # ``node`` IS a column synonym for the ``n`` axis → cast.
    assert frame.schema["node"] == axis_enums["n"]
    # ``profile`` IS a column synonym for the ``f`` axis → cast (Phase 4).
    assert frame.schema["profile"] == axis_enums["f"]


def test_spinedb_reader_parameter_returns_enum_when_axis_enums_supplied(
    TEST_A_LOT_DB: Path,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """:meth:`parameter` on a map parameter unrolls to ``(name, period,
    value)``; the ``period`` index column is cast to the ``d`` enum."""
    r = SpineDbReader(
        str(TEST_A_LOT_DB),
        "test_a_lot",
        axis_enums=axis_enums,
        contract=contract,
    )
    frame = r.parameter("solve", "years_represented")
    # ``name`` is the 0-dim entity column for class ``solve`` — not in
    # contract synonyms, stays Utf8.
    assert frame.schema["name"] == pl.Utf8
    # ``period`` is a synonym for ``d`` — cast.
    assert frame.schema["period"] == axis_enums["d"]
    # ``value`` is non_dim — stays Float64.
    assert frame.schema["value"] == pl.Float64


def test_spinedb_reader_parameter_explicit_casts_dim_columns(
    TEST_A_LOT_DB: Path,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """:meth:`parameter_explicit` returns rows with overrides only; the
    dim columns still get cast.
    """
    r = SpineDbReader(
        str(TEST_A_LOT_DB),
        "test_a_lot",
        axis_enums=axis_enums,
        contract=contract,
    )
    frame = r.parameter_explicit("solve", "years_represented")
    if frame.height == 0:
        return
    assert frame.schema["period"] == axis_enums["d"]


# ---------------------------------------------------------------------------
# 3. Integrity-error path.
# ---------------------------------------------------------------------------


def test_spinedb_reader_filters_scenario_trimmed_parameter_keys(
    TEST_A_LOT_DB: Path,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """Map-inner-key dim columns (axis ``source_type == "parameter_keys"``)
    are silently filtered to the scenario-active vocabulary in
    ``_maybe_cast_frame``, NOT raised — Maps may legitimately carry
    inner keys for periods/timesteps that the active scenario does
    not enable, and the strict cast against the scenario-narrow enum
    would otherwise reject perfectly valid data.

    Entity-class dim columns remain strict — see the backend-level
    breadcrumb suite in ``tests/spinedb_backend/test_backend_cast.py``.
    """
    # Probe the real DB to find a period token to sacrifice.
    probe = SpineDbReader(str(TEST_A_LOT_DB), "test_a_lot")
    frame = probe.parameter("solve", "years_represented")
    if frame.height == 0:
        return
    sacrifice = frame["period"][0]
    d_vocab = [t for t in axis_enums["d"].categories.to_list()
               if t != sacrifice]
    bad_enums = dict(axis_enums)
    bad_enums["d"] = pl.Enum(d_vocab)

    r = SpineDbReader(
        str(TEST_A_LOT_DB),
        "test_a_lot",
        axis_enums=bad_enums,
        contract=contract,
    )
    filtered = r.parameter("solve", "years_represented")
    # The sacrificed token is silently dropped — no exception, no row
    # carrying that token survives the cast.
    assert filtered.schema["period"] == bad_enums["d"]
    assert sacrifice not in filtered["period"].cast(pl.Utf8).to_list()
    # All surviving period tokens are members of the narrowed vocabulary.
    survivors = set(filtered["period"].cast(pl.Utf8).to_list()) - {None}
    assert survivors <= set(d_vocab)


def test_spinedb_reader_loads_contract_automatically(
    TEST_A_LOT_DB: Path,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """When constructed with only ``axis_enums`` (no ``contract``),
    the reader loads the default contract automatically — smoke check
    the cast still runs.
    """
    r = SpineDbReader(
        str(TEST_A_LOT_DB),
        "test_a_lot",
        axis_enums=axis_enums,
    )
    frame = r.entities("node__profile")
    assert frame.schema["node"] == axis_enums["n"]
