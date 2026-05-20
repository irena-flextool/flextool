"""Phase 2 tests for :class:`SpineDbReader` cast-on-emit.

The reader's constructor gains optional ``axis_enums`` + ``contract``
kwargs.  When supplied, every frame returned by :meth:`entities`,
:meth:`parameter`, :meth:`parameter_explicit` has its dim columns cast
to the canonical :class:`pl.Enum` dtypes.

Vocabulary misses raise :class:`FlexDataIntegrityError` with
``(parameter, entity_class, scenario)`` origin breadcrumbs threaded in.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import SpineDbReader
from flextool.spinedb_backend import SpineDBBackend
from flextool.spinedb_backend._axis_enums import (
    AxisContract,
    FlexDataIntegrityError,
    build_axis_enums,
    load_axis_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_A_LOT_DB = (
    REPO_ROOT / "tests" / "engine_polars" / "data" / "work_test_a_lot"
    / "tests.sqlite"
)


pytestmark = pytest.mark.skipif(
    not TEST_A_LOT_DB.exists(),
    reason=f"test_a_lot DB not present at {TEST_A_LOT_DB}",
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contract() -> AxisContract:
    return load_axis_contract()


@pytest.fixture(scope="module")
def axis_enums(contract: AxisContract) -> dict[str, pl.Enum]:
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


def test_spinedb_reader_returns_utf8_when_axis_enums_none() -> None:
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
        pytest.skip("solve.years_represented has no rows for this scenario")
    assert frame.schema["period"] == axis_enums["d"]


# ---------------------------------------------------------------------------
# 3. Integrity-error path.
# ---------------------------------------------------------------------------


def test_spinedb_reader_integrity_error_carries_origin(
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """A truncated period enum forces a miss in
    ``solve.years_represented``; the raised
    :class:`FlexDataIntegrityError` must mention the parameter name +
    entity_class as origin breadcrumbs.
    """
    # Probe the real DB to find a period token to sacrifice.
    probe = SpineDbReader(str(TEST_A_LOT_DB), "test_a_lot")
    frame = probe.parameter("solve", "years_represented")
    if frame.height == 0:
        pytest.skip("no years_represented rows in scenario")
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
    with pytest.raises(FlexDataIntegrityError) as exc:
        r.parameter("solve", "years_represented")
    msg = str(exc.value)
    assert sacrifice in msg
    # Origin breadcrumb: parameter_name was threaded in.
    assert "years_represented" in msg
    # Origin breadcrumb: entity_class.
    assert "solve" in msg
    # Origin breadcrumb: scenario.
    assert "test_a_lot" in msg


def test_spinedb_reader_loads_contract_automatically(
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
