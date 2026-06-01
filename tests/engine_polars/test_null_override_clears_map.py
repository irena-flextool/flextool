"""Regression: an explicit ``null`` value clears a base Map, it must
not crash the per-(node, constraint) coefficient reader.

Root cause (Zimbabwe ``S4_Dry``): a scenario stacks a higher-priority
alternative that sets ``node.constraint_invested_capacity_coeff`` to a
bare Spine ``null`` (``type=None, value=b'null'``), overriding the base
alternative's Map.  ``from_database`` parses that to Python ``None``.

Before the fix, :meth:`SpineDbReader._unroll_rows` kept the whole-null
value as a value-less *scalar* row, so :meth:`parameter` returned a
``[name, value]`` frame with no index column.  The downstream shape
resolver
(:func:`flextool.engine_polars._direct_params._resolve_constraint_index_col`)
then raised ``Unrecognised Map index column(s) []`` — the misleading
"malformed Map" error in the bug report.

The fix drops *whole-null* values (``v is None``) before shape
discovery, so a cleared entity resolves to "unset" — exactly like an
absent parameter.  It deliberately does NOT drop null *leaves* inside a
Map (``Map{a: 1, b: null}``): those parse to a container object, the
sibling structure still yields the index column, and dropping a key
would silently shrink a Map's domain.  This test pins both halves of
that contract.

Fixture: ``tests/fixtures/null_override_clears_map.json`` — a ``node``
class with three entities and three composable base-Map alternatives
plus a ``clear`` alternative that nulls ``node_cleared``:

* ``with_map``      — all three Maps active (baseline).
* ``nulled_mixed``  — ``node_cleared`` nulled; ``node_keep`` and
  ``node_partial`` survive (cases a + c).
* ``nulled_only``   — only ``node_cleared`` present, then nulled →
  empty (case b).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.db_utils import json_to_db

from flextool.engine_polars import _direct_params as dp
from flextool.engine_polars._spinedb_reader import SpineDbReader

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "null_override_clears_map.json"
PARAM = "constraint_invested_capacity_coeff"


@pytest.fixture(scope="module")
def db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    db_path = tmp_path_factory.mktemp("null_override") / "tests.sqlite"
    return json_to_db(FIXTURE, db_path)


def _reader(db_url: str, scenario: str) -> SpineDbReader:
    return SpineDbReader(db_url, scenario)


def test_baseline_all_maps_present(db_url: str) -> None:
    """``with_map``: every base Map resolves, including ``node_partial``'s
    null *leaf* — the fixture is non-degenerate and null leaves are
    retained (not the thing the fix drops)."""
    df = _reader(db_url, "with_map").parameter("node", PARAM).sort("name", "constraint")
    assert df.columns == ["name", "constraint", "value"]
    rows = list(zip(df["name"].to_list(), df["constraint"].to_list(), df["value"].to_list()))
    assert rows == [
        ("node_cleared", "cleared_link", -1.0),
        ("node_keep", "keep_link", 2.0),
        ("node_partial", "partial_a", 3.0),
        ("node_partial", "partial_b", None),  # null leaf retained
    ]


def test_null_override_drops_cleared_entity_siblings_survive(db_url: str) -> None:
    """``nulled_mixed`` (cases a + c): ``node_cleared``'s whole-null
    override removes it entirely; ``node_keep`` keeps its Map; the
    ``constraint`` index column is intact; ``node_partial`` keeps BOTH
    leaves (the null leaf is NOT dropped — no domain shrinkage)."""
    df = _reader(db_url, "nulled_mixed").parameter("node", PARAM).sort("name", "constraint")
    assert df.columns == ["name", "constraint", "value"]
    assert "node_cleared" not in df["name"].to_list()
    rows = list(zip(df["name"].to_list(), df["constraint"].to_list(), df["value"].to_list()))
    assert rows == [
        ("node_keep", "keep_link", 2.0),
        ("node_partial", "partial_a", 3.0),
        ("node_partial", "partial_b", None),  # null leaf retained (scope boundary)
    ]


def test_null_override_mixed_node_constraint_coef_no_raise(db_url: str) -> None:
    """The actual crash site: ``_node_constraint_coef`` no longer raises
    on a frame where one entity was nulled; surviving rows are returned."""
    p = dp._node_constraint_coef(_reader(db_url, "nulled_mixed"), PARAM)
    assert p is not None
    out = p.lazy.collect().sort("n", "cn")
    assert out.columns == ["n", "cn", "value"]
    rows = list(zip(out["n"].to_list(), out["cn"].to_list(), out["value"].to_list()))
    assert rows == [
        ("node_keep", "keep_link", 2.0),
        ("node_partial", "partial_a", 3.0),
        ("node_partial", "partial_b", None),
    ]


def test_only_entity_nulled_resolves_to_empty(db_url: str) -> None:
    """``nulled_only`` (case b): the only param-bearing node is nulled →
    :meth:`parameter` returns an empty frame and the constraint-coef
    reader resolves to ``None`` (cleared, not crashed)."""
    df = _reader(db_url, "nulled_only").parameter("node", PARAM)
    assert df.height == 0
    assert dp._node_constraint_coef(_reader(db_url, "nulled_only"), PARAM) is None
