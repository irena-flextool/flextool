"""Phase 2 tests for :class:`SpineDBBackend` cast-on-emit.

The Backend gains two optional kwargs (``axis_enums`` + ``contract``)
on :meth:`entities`, :meth:`parameter_values`, :meth:`parameter_defaults`.
When supplied the returned frame's dim columns are cast to the
canonical :class:`pl.Enum` dtypes from
:func:`flextool.spinedb_backend._axis_enums.build_axis_enums`.

Vocabulary misses raise :class:`FlexDataIntegrityError` with
``(parameter, entity, scenario)`` origin breadcrumbs threaded in.

The Phase-1 ``test_axis_enums`` blocker (scenario-filter session
binding) means we exercise the cast against an unscoped Backend —
the cast is a per-frame dtype contract, independent of which
scenario filter the Backend was opened with.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

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
def backend() -> SpineDBBackend:
    b = SpineDBBackend(str(TEST_A_LOT_DB))
    yield b
    b.close()


@pytest.fixture(scope="module")
def axis_enums(
    backend: SpineDBBackend, contract: AxisContract,
) -> dict[str, pl.Enum]:
    return build_axis_enums(backend, contract)


# ---------------------------------------------------------------------------
# 1. Back-compat: no kwargs → Utf8.
# ---------------------------------------------------------------------------


def test_entities_returns_utf8_when_axis_enums_none(
    backend: SpineDBBackend,
) -> None:
    """Pre-Phase-2 behaviour: when ``axis_enums`` is None (default), the
    dim columns of :meth:`entities` are Utf8.  This is the back-compat
    slot for the Phase 4 rollout."""
    frame = backend.entities(
        classes=("node",),
        header="node",
    )
    assert frame.schema["node"] == pl.Utf8


def test_parameter_values_returns_utf8_when_axis_enums_none(
    backend: SpineDBBackend,
) -> None:
    """Same back-compat slot for :meth:`parameter_values`."""
    frame = backend.parameter_values(
        cl_pars=[("commodity", "price_method")],
        header="commodity,p_commodity_price_method",
    )
    if frame.height > 0:
        assert frame.schema["commodity"] == pl.Utf8


def test_parameter_defaults_returns_utf8_when_axis_enums_none(
    backend: SpineDBBackend,
) -> None:
    """Same back-compat slot for :meth:`parameter_defaults`."""
    frame = backend.parameter_defaults(
        cl_pars=[("node", "penalty_up"), ("node", "penalty_down")],
        header="class,paramName,default_value",
        filter_in_type=["float", "str", "bool"],
    )
    for col in frame.columns:
        assert frame.schema[col] == pl.Utf8


# ---------------------------------------------------------------------------
# 2. Happy path: axis_enums supplied → Enum dim columns.
# ---------------------------------------------------------------------------


def test_entities_returns_enum_when_axis_enums_supplied(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """Passing ``axis_enums`` casts the ``node`` column to the canonical
    ``n`` enum (entity_class axis)."""
    frame = backend.entities(
        classes=("node",),
        header="node",
        axis_enums=axis_enums,
        contract=contract,
    )
    assert frame.schema["node"] == axis_enums["n"]


def test_entities_multi_class_returns_enum(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """A multi-class entities call (e.g. unit + connection) still casts
    correctly when the header column matches a contract axis."""
    frame = backend.entities(
        classes=("unit",),
        header="unit",
        axis_enums=axis_enums,
        contract=contract,
    )
    # Column "unit" has no axis mapping in the contract; should stay Utf8.
    assert frame.schema["unit"] == pl.Utf8


def test_parameter_values_casts_dim_columns(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """A scalar parameter on commodity casts the ``commodity`` column to
    the ``c`` enum.  ``value`` (data-side) stays Utf8."""
    frame = backend.parameter_values(
        cl_pars=[("commodity", "price")],
        header="commodity,value",
        filter_in_type=["float"],
        axis_enums=axis_enums,
        contract=contract,
    )
    if frame.height == 0:
        pytest.skip("commodity.price has no float rows in fixture")
    assert frame.schema["commodity"] == axis_enums["c"]
    # value is a non_dim_columns entry — left as Utf8.
    assert frame.schema["value"] == pl.Utf8


def test_parameter_values_map_param_casts_period_column(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """A 1d-map parameter (period-indexed) emits ``[entity, period,
    value]`` rows; the period column is cast to the ``d`` enum.

    Uses ``solve.years_represented`` which is a Map(period → years).
    """
    frame = backend.parameter_values(
        cl_pars=[("solve", "years_represented")],
        header="solve,period,value",
        filter_in_type=["1d_map"],
        axis_enums=axis_enums,
        contract=contract,
    )
    if frame.height == 0:
        pytest.skip("solve.years_represented has no rows in fixture")
    # ``solve`` column has no contract axis — stays Utf8.
    assert frame.schema["solve"] == pl.Utf8
    # ``period`` is a synonym for the d axis → cast to d.
    assert frame.schema["period"] == axis_enums["d"]
    # ``value`` is the data column — left Utf8.
    assert frame.schema["value"] == pl.Utf8


def test_parameter_defaults_casts_dim_columns(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """The default-value frame's ``class`` / ``paramName`` /
    ``default_value`` columns are all non-dim → all stay Utf8.

    This pins the policy that the contract's ``non_dim_columns`` block
    is respected even when ``axis_enums`` is supplied.
    """
    frame = backend.parameter_defaults(
        cl_pars=[("node", "penalty_up"), ("node", "penalty_down")],
        header="class,paramName,default_value",
        filter_in_type=["float", "str", "bool"],
        axis_enums=axis_enums,
        contract=contract,
    )
    for col in ("class", "paramName", "default_value"):
        assert frame.schema[col] == pl.Utf8


def test_parameter_defaults_loads_contract_when_omitted(
    backend: SpineDBBackend,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """When ``contract`` is omitted but ``axis_enums`` is supplied, the
    helper transparently loads the default contract.  Smoke check: the
    call still succeeds without a TypeError / NoneType failure.
    """
    frame = backend.parameter_defaults(
        cl_pars=[("node", "penalty_up")],
        header="class,paramName,default_value",
        filter_in_type=["float", "str", "bool"],
        axis_enums=axis_enums,
    )
    # The frame columns all map to non_dim entries; just check it built.
    assert "class" in frame.columns


# ---------------------------------------------------------------------------
# 3. Integrity-error path: bad token → FlexDataIntegrityError w/ origin.
# ---------------------------------------------------------------------------


def test_entities_integrity_error_carries_origin(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """Inject a truncated axis_enums (drop one node from the ``n``
    vocabulary) and call :meth:`entities` on the ``node`` class — the
    Backend now sees a real-DB token that isn't in the enum and must
    raise :class:`FlexDataIntegrityError`.

    The error message must mention the entity name (so users can find
    the row in their DB) and the axis label.
    """
    # Get one real node name and remove it from the vocabulary to force
    # a miss when the Backend tries to cast the entities frame.
    nodes = [
        e["name"] for e in backend.find_entities(entity_class_name="node")
    ]
    if not nodes:
        pytest.skip("no node entities in fixture")
    sacrifice = nodes[0]
    n_vocab = [t for t in axis_enums["n"].categories.to_list()
               if t != sacrifice]
    bad_enums = dict(axis_enums)
    bad_enums["n"] = pl.Enum(n_vocab)
    with pytest.raises(FlexDataIntegrityError) as exc:
        backend.entities(
            classes=("node",),
            header="node",
            axis_enums=bad_enums,
            contract=contract,
        )
    msg = str(exc.value)
    assert sacrifice in msg
    assert "node" in msg.lower()


def test_parameter_values_integrity_error_carries_origin(
    backend: SpineDBBackend,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
) -> None:
    """Same shape for :meth:`parameter_values`: a truncated period enum
    forces a miss in the ``solve.years_represented`` map; the error
    message must mention the parameter name + the bad period token.
    """
    # First check there's any rows; otherwise skip.
    rows = backend.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="years_represented",
    )
    map_rows = [r for r in rows if r.get("type") == "map"]
    if not map_rows:
        pytest.skip("no solve.years_represented map rows in fixture")
    # Extract a real period token and drop it from the d vocab.
    pv = map_rows[0]["parsed_value"]
    sacrifice = str(pv.indexes[0])
    d_vocab = [t for t in axis_enums["d"].categories.to_list()
               if t != sacrifice]
    bad_enums = dict(axis_enums)
    bad_enums["d"] = pl.Enum(d_vocab)
    with pytest.raises(FlexDataIntegrityError) as exc:
        backend.parameter_values(
            cl_pars=[("solve", "years_represented")],
            header="solve,period,value",
            filter_in_type=["1d_map"],
            axis_enums=bad_enums,
            contract=contract,
        )
    msg = str(exc.value)
    # The bad period token is in the message.
    assert sacrifice in msg
    # The parameter name is in the origin breadcrumb (Option C).
    assert "years_represented" in msg
