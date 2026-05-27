"""Cast Map value columns to the contract dtype, not the sniffed one.

Spine DB does not enforce types inside Map values: the same numeric
coefficient may be authored as the Python ``float`` ``1.0`` in one cell
and as the Python ``str`` ``"1"`` in the next.  Sample-sniffing the
first non-None leaf (the pre-contract behaviour) makes the resulting
polars ``value`` column ``Utf8`` whenever a string leaf is seen first,
which silently breaks downstream lazy arithmetic in polar-high — the
error surfaces deep inside ``collect_schema()`` as
``InvalidOperationError: arithmetic on string and numeric not allowed``.

These tests build an isolated SpineDB at test time via
:mod:`spinedb_api`, populate it with the minimal shape that reproduces
the bug (one Map with float leaves + one Map with string leaves of the
same parameter), and assert that
:meth:`flextool.engine_polars._spinedb_reader.SpineDbReader.parameter`
returns a ``Float64`` value column for both rows.

A second test pins the "unparseable string" path: a Map authored with
``"abc"`` in a value cell that the contract types as ``Float64`` must
surface as :class:`FlexDataIntegrityError` with the entity name and
offending token in the message so the user can locate the bad cell.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


# Late imports inside the helpers so the module collects even when
# ``spinedb_api`` is unavailable in the runner; the tests themselves
# would xfail / skip in that case via the helper's import.


def _build_db(
    sqlite_path: Path,
    *,
    rows: list[tuple[str, object]],
    parameter_name: str = "constraint_invested_capacity_coefficient",
) -> str:
    """Build a one-class / one-parameter SpineDB at *sqlite_path*.

    The class is ``widget`` (an arbitrary single-dim placeholder, not a
    FlexTool class — keeps the fixture self-contained).  The parameter
    defaults to ``constraint_invested_capacity_coefficient`` — a real
    FlexTool parameter so the contract's ``Float64`` default applies
    without an explicit override.

    ``rows`` is a list of ``(entity_name, value)`` pairs.  ``value``
    can be a :class:`spinedb_api.parameter_value.Map` (Map / Array
    leaf goes through the contract) or a Python scalar (str / float /
    bool — bypasses the contract).

    Returns the resulting Spine URL.
    """
    from spinedb_api import DatabaseMapping
    from spinedb_api.parameter_value import to_database

    url = f"sqlite:///{sqlite_path}"
    with DatabaseMapping(url, create=True) as db:
        # SpineDbReader requires a named scenario to apply the filter.
        # The default "Base" alternative is created automatically; wrap
        # it in a scenario of the same name so the filter resolves.
        db.add_scenario_item(name="Base")
        db.add_scenario_alternative_item(
            scenario_name="Base", alternative_name="Base", rank=1,
        )
        db.add_entity_class_item(name="widget")
        db.add_parameter_definition_item(
            name=parameter_name,
            entity_class_name="widget",
        )
        for entity_name, _v in rows:
            db.add_entity_item(entity_class_name="widget", name=entity_name)
        for entity_name, value in rows:
            db_val, db_type = to_database(value)
            db.add_parameter_value_item(
                entity_class_name="widget",
                parameter_definition_name=parameter_name,
                entity_byname=(entity_name,),
                alternative_name="Base",
                value=db_val,
                type=db_type,
            )
        db.commit_session("seed")
    return url


def test_value_column_is_float_when_some_cells_are_strings(tmp_path: Path) -> None:
    """Mixed float / string leaves cast to the contract's Float64.

    The bug: sample-sniffing the first leaf of the FIRST row determined
    the whole column's dtype.  When the first row had a string leaf
    (``"1"`` — perfectly numeric as text), every row's value came back
    Utf8.  Downstream lazy arithmetic then chokes.
    """
    from spinedb_api.parameter_value import Map

    from flextool.engine_polars._spinedb_reader import SpineDbReader

    rows = [
        ("W_str",   Map(indexes=["c1"], values=["1"],  index_name="constraint")),
        ("W_float", Map(indexes=["c1"], values=[1.0],  index_name="constraint")),
    ]
    url = _build_db(tmp_path / "mixed.sqlite", rows=rows)

    # No scenario filter: pass the default "Base" alternative as a
    # scenario name.  spinedb-api's filter is a no-op for a single-
    # alternative DB.
    reader = SpineDbReader(url, scenario="Base")
    df = reader.parameter(
        "widget", "constraint_invested_capacity_coefficient",
    )

    assert df.schema["value"] == pl.Float64, (
        f"value column dtype {df.schema['value']!r} != Float64; "
        "the string '1' on the first row poisoned the sniff"
    )
    # Both rows survived the cast with the right numeric value.
    assert sorted(df["value"].to_list()) == [1.0, 1.0]


def test_unparseable_string_value_raises_with_breadcrumb(tmp_path: Path) -> None:
    """A genuinely non-numeric leaf (``"abc"``) for a Float64-typed
    parameter must NOT pass silently — it must raise with enough
    context for the user to locate the offending Spine cell.
    """
    from spinedb_api.parameter_value import Map

    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.spinedb_backend._axis_enums import (
        FlexDataIntegrityError,
    )

    rows = [
        ("W_good", Map(indexes=["c1"], values=[2.5],   index_name="constraint")),
        ("W_typo", Map(indexes=["c1"], values=["abc"], index_name="constraint")),
    ]
    url = _build_db(tmp_path / "typo.sqlite", rows=rows)

    reader = SpineDbReader(url, scenario="Base")
    with pytest.raises(FlexDataIntegrityError) as excinfo:
        reader.parameter(
            "widget", "constraint_invested_capacity_coefficient",
        )

    msg = str(excinfo.value)
    # Breadcrumb: entity + parameter + bad token must all be visible.
    assert "abc" in msg
    assert "W_typo" in msg
    assert "constraint_invested_capacity_coefficient" in msg


def test_scalar_string_parameter_bypasses_contract_dtype(tmp_path: Path) -> None:
    """Top-level scalar parameters do NOT go through the contract dtype.

    The contract defaults to ``Float64`` for any parameter without an
    explicit override, but that policy must only apply to indexed
    (Map / Array) parameters.  Scalar parameters like ``node_type``
    ("commodity" / "storage" / ...) come from spinedb-api with their
    native Python type already set; if the contract default was
    applied here, every scalar string parameter would be rejected at
    read time.

    Regression: an earlier draft did apply the contract dtype to
    scalars, which broke ``projection_params.nodeBalance`` on the
    first DB that hit it.
    """
    from flextool.engine_polars._spinedb_reader import SpineDbReader

    rows = [
        ("W1", "commodity"),
        ("W2", "storage"),
    ]
    # ``node_type`` is a real FlexTool parameter that is scalar-str;
    # the contract has no override for it, so the default Float64
    # would (incorrectly) apply if scalars weren't bypassed.
    url = _build_db(
        tmp_path / "scalar.sqlite",
        rows=rows,
        parameter_name="node_type",
    )

    reader = SpineDbReader(url, scenario="Base")
    df = reader.parameter("widget", "node_type")

    assert df.schema["value"] == pl.Utf8
    assert set(df["value"].to_list()) == {"commodity", "storage"}


def test_array_string_parameter_bypasses_contract_dtype(tmp_path: Path) -> None:
    """``Array`` parameters bypass the contract dtype.

    Every FlexTool ``Array`` parameter (``model.solves``,
    ``solve.invest_periods``, ``solve.realized_periods``,
    ``solve.solver_arguments``, ...) holds Utf8 leaves — there are no
    numeric ``Array`` parameters in the schema.  Applying the
    contract default ``Float64`` to them would reject every row.

    Regression: an earlier draft enforced the contract on any
    indexed shape, which broke ``model.solves`` (cells like
    ``"y2020_2day_dispatch"``) for every fixture that read it.
    """
    from spinedb_api.parameter_value import Array

    from flextool.engine_polars._spinedb_reader import SpineDbReader

    rows = [
        ("W1", Array(["y2020_2day_dispatch", "y2030_2day_dispatch"])),
    ]
    url = _build_db(
        tmp_path / "array.sqlite",
        rows=rows,
        # ``solves`` is the real-world Array-of-str case (model.solves).
        parameter_name="solves",
    )

    reader = SpineDbReader(url, scenario="Base")
    df = reader.parameter("widget", "solves")

    assert df.schema["value"] == pl.Utf8
    assert df["value"].to_list() == [
        "y2020_2day_dispatch", "y2030_2day_dispatch",
    ]
