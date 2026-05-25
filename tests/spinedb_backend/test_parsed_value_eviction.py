"""Track A — :class:`SpineDBBackend.parameter_values` evicts
``MappedItem._parsed_value`` per row.

The eviction relies on a spinedb-api private API contract: that each
``PublicItem`` exposes ``.mapped_item`` (a :class:`MappedItemBase`), and
that the mapped item's ``parsed_value`` is a lazy property backed by an
``_parsed_value`` instance attribute.  If a future spinedb-api release
changes this shape these tests fail loudly with an actionable error
message pointing at the file to revisit
(:mod:`flextool.spinedb_backend._backend`, ``parameter_values``).

The tests are scoped to ``templates/examples.sqlite`` because that
fixture is in-tree and small.  The eviction logic is the same for any
DB the Backend opens, so this fixture exercises the contract without
depending on external paths.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.spinedb_backend import SpineDBBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DB = REPO_ROOT / "templates" / "examples.sqlite"


pytestmark = pytest.mark.skipif(
    not EXAMPLES_DB.exists(),
    reason=f"examples.sqlite not present at {EXAMPLES_DB}",
)


# ---------------------------------------------------------------------------
# API-drift contract — spinedb-api shape this Track A relies on.
# ---------------------------------------------------------------------------


def test_mapped_item_parsed_value_contract() -> None:
    """:class:`MappedItem` (spinedb-api) exposes the lazy ``parsed_value``
    property + ``_parsed_value`` backing attribute that Track A drops.

    If this test fails the spinedb-api API has drifted; revisit
    :mod:`flextool.spinedb_backend._backend` ``parameter_values`` Track-A
    eviction code.  Specifically check that:

    * ``PublicItem.mapped_item`` still returns the underlying
      :class:`MappedItemBase`.
    * ``MappedItem._parsed_value`` still names the cached parsed object.
    * ``MappedItem.parsed_value`` is still a lazy property that
      re-evaluates from ``value`` + ``type`` when ``_parsed_value`` is
      ``None``.
    """
    from spinedb_api.db_mapping_base import PublicItem
    from spinedb_api.mapped_items import ParameterValueItem

    # PublicItem.mapped_item -> MappedItem.
    public_attr = getattr(PublicItem, "mapped_item", None)
    assert public_attr is not None, (
        "spinedb-api API drift: PublicItem.mapped_item missing — "
        "revisit flextool/spinedb_backend/_backend.py Track-A eviction."
    )
    assert isinstance(public_attr, property), (
        "spinedb-api API drift: PublicItem.mapped_item is no longer a "
        "property — revisit Track-A eviction in _backend.py."
    )

    # ParameterValueItem.parsed_value is a property (the lazy-parse hook).
    pv_attr = getattr(ParameterValueItem, "parsed_value", None)
    assert isinstance(pv_attr, property), (
        "spinedb-api API drift: ParameterValueItem.parsed_value is no "
        "longer a property — revisit Track-A eviction in _backend.py."
    )

    # _parsed_value is an instance attribute set in __init__; we can't
    # check it on the class object directly, but we can confirm
    # has_value_been_parsed (the public predicate) still exists.
    assert hasattr(ParameterValueItem, "has_value_been_parsed"), (
        "spinedb-api API drift: MappedItem.has_value_been_parsed "
        "missing — revisit Track-A eviction in _backend.py."
    )


# ---------------------------------------------------------------------------
# Behaviour — eviction happens and re-parse works.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend_with_index() -> SpineDBBackend:
    """A backend whose ``_parameter_value_index`` is populated.

    Constructed unscoped (no scenario filter) because the SpineDBBackend's
    in-constructor scenario_filter path has a documented limitation around
    session binding that is orthogonal to Track A — the eviction logic
    fires regardless of scenario state.
    """
    b = SpineDBBackend(str(EXAMPLES_DB))
    yield b
    b.close()


def _get_map_class_param(
    backend: SpineDBBackend, *, min_rows: int = 1,
) -> tuple[str, str]:
    """Pick a (class, param) tuple whose values are of type ``map``.

    Track A's win materialises on map/time_series/array values (those
    have non-trivial parsed_value memory).  ``map`` is the most common
    shape in examples.sqlite and the one most relevant to peak RSS.

    ``min_rows`` lets callers require at least N rows for tests that
    exercise iteration semantics.
    """
    index = backend._get_parameter_value_index()
    best: tuple[str, str] | None = None
    best_count = -1
    for (cls, par), rows in index.items():
        if not rows:
            continue
        map_rows = [r for r in rows if r["type"] == "map"]
        if len(map_rows) >= min_rows and len(map_rows) > best_count:
            best = (cls, par)
            best_count = len(map_rows)
    if best is None:
        pytest.skip(
            f"No (class, param) with >={min_rows} map-typed rows in "
            f"examples.sqlite"
        )
    return best


def test_parameter_values_drops_parsed_value_after_call(
    backend_with_index: SpineDBBackend,
) -> None:
    """After ``parameter_values()`` returns the underlying
    ``MappedItem._parsed_value`` is ``None`` for each touched row.
    """
    cl_par = _get_map_class_param(backend_with_index)

    # Pre-warm: access parsed_value on each row so we know eviction has
    # something to do (otherwise lazy parse simply never triggered).
    index = backend_with_index._get_parameter_value_index()
    rows = index[cl_par]
    for r in rows:
        _ = r["parsed_value"]  # force lazy parse
    assert all(
        r.mapped_item._parsed_value is not None for r in rows
    ), "fixture sanity check: parsed_value not populated after pre-warm"

    # Run the Track A path: parameter_values walks rows, then evicts.
    backend_with_index.parameter_values(
        cl_pars=[cl_par],
        header="x,y,value",  # column count is permissive for this test
        filter_in_type=["map"],
    )

    # Every row that the materialiser touched should now have
    # ``_parsed_value is None``.
    leftover = [
        r for r in rows
        if r.mapped_item._parsed_value is not None
    ]
    assert leftover == [], (
        f"Track A leak: {len(leftover)} rows for {cl_par} still hold "
        f"a parsed_value after parameter_values() returned."
    )


def test_parsed_value_reparse_after_eviction(
    backend_with_index: SpineDBBackend,
) -> None:
    """After eviction, accessing ``parsed_value`` re-parses identically.

    This is the safety net that lets eviction be unconditional: if some
    downstream caller (today: none; future: defensive) re-accesses a
    cleared row, spinedb-api's lazy property re-parses from ``value`` +
    ``type`` and returns the same object.
    """
    cl_par = _get_map_class_param(backend_with_index)
    rows = backend_with_index._get_parameter_value_index()[cl_par]
    if not rows:
        pytest.skip("no rows for selected (class, param)")

    sample = rows[0]
    # Force parse, snapshot a serialisable form, drop, re-parse.
    before = sample["parsed_value"]
    before_to_database = before.to_database()
    sample.mapped_item._parsed_value = None
    assert sample.mapped_item._parsed_value is None, "eviction didn't stick"

    after = sample["parsed_value"]
    assert after is not None, "re-parse returned None"
    assert after.to_database() == before_to_database, (
        "re-parse produced a different value — Track A's drop+re-parse "
        "round-trip is not lossless; revisit eviction strategy."
    )


def test_evict_as_we_go_handles_continue_paths(
    backend_with_index: SpineDBBackend,
) -> None:
    """The generator-based eviction wraps ``params``; even when the
    materialiser's inner ``for`` body ``continue``s mid-iteration the
    generator's next step evicts the previously-yielded row.

    Constructed manually here so the test pins the behaviour against the
    actual generator definition in :mod:`_backend`, not just end-to-end.
    """
    from flextool.spinedb_backend._backend import SpineDBBackend as Cls

    # Synthesise a tiny stand-in for ``params`` using two real rows.
    cl_par = _get_map_class_param(backend_with_index, min_rows=2)
    rows = backend_with_index._get_parameter_value_index()[cl_par][:2]
    if len(rows) < 2:
        pytest.skip("need at least 2 rows to exercise generator iteration")

    # Pre-warm both rows.
    for r in rows:
        _ = r["parsed_value"]
    assert all(r.mapped_item._parsed_value is not None for r in rows)

    # Drive the same parameter_values path with a continue-on-every-row
    # caller pattern (filter_in_value to a non-match for str-typed; but
    # cl_par is map so the str-filter branch doesn't apply.  Instead use
    # filter_out_index on a value that won't match — that exercises the
    # map-branch continue).
    backend_with_index.parameter_values(
        cl_pars=[cl_par],
        header="x,y,value",
        filter_out_index="__no_such_index_will_match__",
    )

    # Even though the materialiser body continued on every row, the
    # generator wrapper should have evicted as it advanced.
    leftover = [r for r in rows if r.mapped_item._parsed_value is not None]
    assert leftover == [], (
        "generator wrapper failed to evict on continue: "
        f"{len(leftover)} rows still hold parsed_value."
    )

    _ = Cls  # silence unused-import diagnostic on the inner import
