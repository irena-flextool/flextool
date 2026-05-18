"""Phase 1 tests for ``flextool.spinedb_backend._axis_enums``.

The Phase-1 module (sibling test ``test_axis_contract.py`` covers the
JSON shape) implements:

  * :func:`load_axis_contract` — parse the JSON.
  * :func:`build_axis_enums` — emit per-axis :class:`pl.Enum`.
  * :func:`cast_against_contract` — boundary cast helper.
  * :class:`FlexDataIntegrityError` — beginner-friendly cast failure.
  * :func:`_lookup_similar_classes` — suggestion-builder helper.

These tests pin every behaviour above against the canonical contract
and a small DB fixture (``work_test_a_lot/tests.sqlite``).

The Backend's per-scenario filter is not exercised here — it has a
pre-existing session-binding bug that's unrelated to Phase 1 (see
``test_spinedb_backend_smoke.py`` which also skips scenarios).  The
unscoped vocabulary (no scenario filter) is the right vocabulary for
the Enum bootstrap anyway: an Enum dtype must cover EVERY token any
scenario emits, and the unscoped EAV view is the union.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.spinedb_backend import SpineDBBackend
from flextool.spinedb_backend._axis_enums import (
    AxisContract,
    AxisSpec,
    FlexDataIntegrityError,
    _lookup_similar_classes,
    build_axis_enums,
    cast_against_contract,
    load_axis_contract,
)


# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_A_LOT_DB = (
    REPO_ROOT / "tests" / "engine_polars" / "data" / "work_test_a_lot"
    / "tests.sqlite"
)
LADDER_ANNUAL_DB = (
    REPO_ROOT / "tests" / "engine_polars" / "data"
    / "work_commodity_ladder_annual" / "tests.sqlite"
)


pytestmark = pytest.mark.skipif(
    not TEST_A_LOT_DB.exists(),
    reason=f"test_a_lot DB not present at {TEST_A_LOT_DB}",
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — open the Backend once, reuse across tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contract() -> AxisContract:
    return load_axis_contract()


@pytest.fixture(scope="module")
def backend() -> SpineDBBackend:
    """A long-lived SpineDBBackend on test_a_lot.

    No scenario filter — see the module docstring for the rationale.
    """
    b = SpineDBBackend(str(TEST_A_LOT_DB))
    yield b
    b.close()


@pytest.fixture(scope="module")
def axis_enums(
    backend: SpineDBBackend, contract: AxisContract,
) -> dict[str, pl.Enum]:
    return build_axis_enums(backend, contract)


# ---------------------------------------------------------------------------
# 1. load_axis_contract
# ---------------------------------------------------------------------------


def test_load_axis_contract_parses_and_returns_dataclass() -> None:
    """``load_axis_contract`` parses the canonical JSON, returns an
    :class:`AxisContract` instance with populated fields.

    The contract currently declares 17 axes; the assertion is
    written as ``>= 15`` so new axis additions don't churn the test.
    """
    c = load_axis_contract()
    assert isinstance(c, AxisContract)
    assert len(c.axes) >= 15, (
        f"contract should declare ≥15 axes; got {len(c.axes)}"
    )
    # Every axis is an AxisSpec with the required fields populated.
    for axis in c.axes:
        assert isinstance(axis, AxisSpec)
        assert axis.name
        assert axis.source_type in (
            "entity_class", "entity_class_union",
            "parameter_keys", "parameter_value_list", "synthetic",
        )
    # mixed_vocab + synthetic_allowlist + non_dim blocks are present.
    assert "confirmed" in c.mixed_vocab_columns
    assert c.synthetic_token_allowlist  # non-empty
    assert "method" in c.non_dim_columns.get("confirmed", [])


def test_column_to_axis_lookup(contract: AxisContract) -> None:
    """``column_to_axis`` resolves friendly column names + mixed-vocab
    columns to the right axis."""
    assert contract.column_to_axis("entity").name == "e"
    assert contract.column_to_axis("source").name == "e"
    assert contract.column_to_axis("sink").name == "e"
    assert contract.column_to_axis("node").name == "n"
    assert contract.column_to_axis("process").name == "p"
    assert contract.column_to_axis("period").name == "d"
    # Direct axis-name match.
    assert contract.column_to_axis("n").name == "n"
    assert contract.column_to_axis("cn").name == "constraint"
    assert contract.column_to_axis("bk").name == "block"
    # Non-dim columns are not mapped.
    assert contract.column_to_axis("method") is None
    assert contract.column_to_axis("value") is None
    assert contract.column_to_axis("scenario") is None
    # Unknown columns return None (no axis match).
    assert contract.column_to_axis("definitely_not_a_dim") is None


# ---------------------------------------------------------------------------
# 2. build_axis_enums — vocabulary sourcing
# ---------------------------------------------------------------------------


def test_build_axis_enums_entity_class(
    backend: SpineDBBackend, axis_enums: dict[str, pl.Enum],
) -> None:
    """For ``n`` (node entity_class), the enum vocab matches
    ``backend.find_entities(entity_class_name="node")``."""
    expected = {e["name"] for e in backend.find_entities(entity_class_name="node")}
    actual = set(axis_enums["n"].categories.to_list())
    assert expected.issubset(actual), (
        f"node enum missing entities: {expected - actual}"
    )


def test_build_axis_enums_entity_class_union(
    backend: SpineDBBackend, axis_enums: dict[str, pl.Enum],
) -> None:
    """For ``p`` (entity_class_union), the enum vocab is the union of
    the ``unit`` and ``connection`` entity classes."""
    units = {e["name"] for e in backend.find_entities(entity_class_name="unit")}
    conns = {e["name"] for e in backend.find_entities(entity_class_name="connection")}
    expected = units | conns
    actual = set(axis_enums["p"].categories.to_list())
    assert actual == expected, (
        f"process enum mismatch: missing={expected - actual}, "
        f"extra={actual - expected}"
    )


def test_build_axis_enums_parameter_keys_t(
    backend: SpineDBBackend, axis_enums: dict[str, pl.Enum],
) -> None:
    """``t`` enum sources from the keys of
    ``timeline.timestep_duration`` map."""
    tokens = set(axis_enums["t"].categories.to_list())
    # test_a_lot has timeline ``y2020`` whose timestep_duration map has
    # keys ``t0001``, ``t0002``, ....
    assert "t0001" in tokens
    assert "t0002" in tokens
    # Cardinality matches the largest timestep set in the DB.
    rows = backend.find_parameter_values(
        entity_class_name="timeline",
        parameter_definition_name="timestep_duration",
    )
    expected_keys: set[str] = set()
    for r in rows:
        pv = r.get("parsed_value")
        if pv is not None and hasattr(pv, "indexes"):
            for idx in pv.indexes:
                expected_keys.add(str(idx))
    assert tokens == expected_keys


def test_build_axis_enums_parameter_keys_d(
    backend: SpineDBBackend, axis_enums: dict[str, pl.Enum],
) -> None:
    """``d`` enum unions period keys from the four solve parameters
    (years_represented, realized_periods, invest_periods,
    realized_invest_periods)."""
    tokens = set(axis_enums["d"].categories.to_list())
    # test_a_lot has periods p2020..p2035.
    assert "p2020" in tokens
    assert "p2025" in tokens
    assert "p2030" in tokens
    assert "p2035" in tokens


def test_build_axis_enums_parameter_keys_i_runtime_depth(
    contract: AxisContract,
) -> None:
    """``i`` enum cardinality matches the runtime-discovered tier count
    of the ``commodity.price_ladder_*`` maps.  Uses a fixture that has
    a known 2-tier ladder so this gates the discovery logic instead of
    measuring "≥1"."""
    if not LADDER_ANNUAL_DB.exists():
        pytest.skip(f"ladder annual DB not at {LADDER_ANNUAL_DB}")
    backend_ladder = SpineDBBackend(str(LADDER_ANNUAL_DB))
    try:
        enums = build_axis_enums(backend_ladder, contract)
        i_vocab = enums["i"].categories.to_list()
        # work_commodity_ladder_annual has a 2-tier ladder (keys "1", "2").
        assert i_vocab == ["1", "2"], (
            f"i axis vocab should be ['1','2']; got {i_vocab}"
        )
    finally:
        backend_ladder.close()


def test_build_axis_enums_synthetic(axis_enums: dict[str, pl.Enum]) -> None:
    """Synthetic axes use the contract's ``tokens`` field verbatim."""
    assert axis_enums["side"].categories.to_list() == ["source", "sink"]
    assert axis_enums["ud"].categories.to_list() == ["up", "down"]
    assert axis_enums["klass"].categories.to_list() == ["unit", "connection"]


def test_synthetic_allowlist_merged_into_branch(
    axis_enums: dict[str, pl.Enum],
) -> None:
    """The ``branch`` enum carries the schema-derived branch labels
    PLUS the ``eff`` / ``noEff`` synthetic allowlist tokens introduced
    by ``_writer_co2_accumulators``.

    test_a_lot has no ``solve.stochastic_branches`` rows, so the only
    tokens in the branch enum are the allowlist entries.  The pin
    asserts that the allowlist is folded in regardless of whether the
    DB has stochastic data — the writer literally introduces these
    tokens and the enum must accept them.
    """
    tokens = axis_enums["branch"].categories.to_list()
    assert "eff" in tokens
    assert "noEff" in tokens


def test_build_axis_enums_parameter_value_list_branch(
    contract: AxisContract, axis_enums: dict[str, pl.Enum],
) -> None:
    """``branch`` axis sources from ``solve.stochastic_branches``
    scalar values per the contract.  On a fixture with no stochastic
    rows the enum equals only the allowlist.

    The shape — that the loader RAN ``parameter_value_list`` discovery
    against the contract — is gated by the existing
    :func:`test_synthetic_allowlist_merged_into_branch`; this test
    additionally pins the axis ``source_type`` so future contract
    edits can't silently swap to ``synthetic`` and lose DB-sourced
    branch labels.
    """
    branch_spec = contract.by_name("branch")
    assert branch_spec.source_type == "parameter_value_list"


# ---------------------------------------------------------------------------
# 3. cast_against_contract — happy path + miss handling
# ---------------------------------------------------------------------------


def test_cast_against_contract_happy_path(
    contract: AxisContract, axis_enums: dict[str, pl.Enum],
    backend: SpineDBBackend,
) -> None:
    """A frame with ``n, p, d, t`` columns whose values are in the DB
    casts cleanly to Enum dtypes."""
    nodes = [e["name"] for e in backend.find_entities(entity_class_name="node")][:2]
    units = [e["name"] for e in backend.find_entities(entity_class_name="unit")][:2]
    frame = pl.DataFrame({
        "n": nodes,
        "p": units,
        "d": ["p2020", "p2025"],
        "t": ["t0001", "t0002"],
        "value": [1.0, 2.0],
    })
    out = cast_against_contract(
        frame, contract=contract, axis_enums=axis_enums,
    )
    assert out.schema["n"] == axis_enums["n"]
    assert out.schema["p"] == axis_enums["p"]
    assert out.schema["d"] == axis_enums["d"]
    assert out.schema["t"] == axis_enums["t"]
    # value column is untouched.
    assert out.schema["value"] == pl.Float64


def test_cast_against_contract_skips_non_dim_columns(
    contract: AxisContract, axis_enums: dict[str, pl.Enum],
) -> None:
    """Columns named ``method`` / ``value`` / ``scenario`` stay Utf8."""
    frame = pl.DataFrame({
        "method": ["m1", "m2"],
        "value": ["v1", "v2"],
        "scenario": ["s1", "s2"],
    })
    out = cast_against_contract(
        frame, contract=contract, axis_enums=axis_enums,
    )
    assert out.schema["method"] == pl.Utf8
    assert out.schema["value"] == pl.Utf8
    assert out.schema["scenario"] == pl.Utf8


def test_cast_against_contract_mixed_vocab(
    contract: AxisContract, axis_enums: dict[str, pl.Enum],
    backend: SpineDBBackend,
) -> None:
    """``source`` and ``sink`` columns hold entity names (mixed
    node + process vocab) — cast against the ``e`` (entity union)
    enum, not against ``n`` or ``p``."""
    # source is a unit, sink is a node — both must be in the 'e' union.
    src = [e["name"] for e in backend.find_entities(entity_class_name="unit")][0]
    snk = [e["name"] for e in backend.find_entities(entity_class_name="node")][0]
    frame = pl.DataFrame({"source": [src], "sink": [snk], "v": [1.0]})
    out = cast_against_contract(
        frame, contract=contract, axis_enums=axis_enums,
    )
    assert out.schema["source"] == axis_enums["e"]
    assert out.schema["sink"] == axis_enums["e"]


def test_cast_against_contract_raises_integrity_error_typo(
    contract: AxisContract, axis_enums: dict[str, pl.Enum],
    backend: SpineDBBackend,
) -> None:
    """A typo'd node name (close to a real node by Levenshtein ≤ 2)
    raises :class:`FlexDataIntegrityError`.  The message names the
    bad token, the axis label, and surfaces a ``did you mean`` hint.

    test_a_lot has a node called ``coal_market``; we feed
    ``coal_market2`` (distance 1)."""
    nodes = {e["name"] for e in backend.find_entities(entity_class_name="node")}
    # Pick a real node, mutate it slightly.
    base = next(n for n in nodes if len(n) >= 4)
    bad = base + "X"  # distance 1; should still be in the suggestion list
    frame = pl.DataFrame({"n": [bad]})
    with pytest.raises(FlexDataIntegrityError) as exc:
        cast_against_contract(
            frame, contract=contract, axis_enums=axis_enums,
            origin={"parameter": "p_inflow", "entity": base},
            backend=backend,
        )
    msg = str(exc.value)
    assert bad in msg
    assert "node" in msg  # axis label
    assert "did you mean" in msg.lower()


def test_cast_against_contract_raises_integrity_error_wrong_class(
    contract: AxisContract, axis_enums: dict[str, pl.Enum],
    backend: SpineDBBackend,
) -> None:
    """A constraint name in a node column raises with an "exists as
    constraint" hint (cross-class lookup wins over typo suggestions)."""
    cstr_name = next(
        e["name"]
        for e in backend.find_entities(entity_class_name="constraint")
    )
    frame = pl.DataFrame({"n": [cstr_name]})
    with pytest.raises(FlexDataIntegrityError) as exc:
        cast_against_contract(
            frame, contract=contract, axis_enums=axis_enums, backend=backend,
        )
    msg = str(exc.value)
    assert cstr_name in msg
    assert "constraint" in msg.lower(), (
        "cross-class lookup hint missing: did the helper find the "
        "constraint match?"
    )


# ---------------------------------------------------------------------------
# 4. _lookup_similar_classes — Levenshtein + cross-class
# ---------------------------------------------------------------------------


def test_lookup_similar_classes_levenshtein(
    contract: AxisContract, backend: SpineDBBackend,
) -> None:
    """A typo within distance 2 of a real node yields a ``did you mean``
    hint; distance > 2 returns no typo suggestions (cross-class lookup
    only)."""
    node_axis = contract.by_name("n")
    # Pick a real node name, then mutate it by 1 character.
    nodes = [
        e["name"]
        for e in backend.find_entities(entity_class_name="node")
        if len(e["name"]) >= 4
    ]
    base = nodes[0]
    typo = base + "x"  # distance 1
    hints_close = _lookup_similar_classes(
        typo, backend, node_axis, ["node", "unit", "connection", "commodity"],
    )
    assert any(base in h for h in hints_close), (
        f"typo {typo!r} (distance 1 from {base!r}) should be flagged; "
        f"got {hints_close}"
    )
    # Far-away token: no Levenshtein hits.
    far = "qwertyuiop_zzzz"
    hints_far = _lookup_similar_classes(
        far, backend, node_axis, ["node", "unit", "connection"],
    )
    # No typo suggestion since edit distance > 2.
    assert all("did you mean" not in h.lower() for h in hints_far)


def test_flex_data_integrity_error_message_format() -> None:
    """Snapshot the 4-paragraph rendering — every section header and
    every breadcrumb line is present, regardless of which suggestions
    apply."""
    err = FlexDataIntegrityError.from_cast_failure(
        axis_name="n",
        axis_friendly="node",
        bad_token="mystery_node",
        vocabulary_size=11,
        parameter="p_inflow",
        entity="north",
        map_index="t0001",
        scenario="test_a_lot",
        suggestions=[
            "'mystery_node' exists as a commodity, not a node.",
            "did you mean 'north'?",
        ],
    )
    msg = str(err)
    # Paragraph 1: opener with bad token + axis name.
    assert "Found an unknown node name 'mystery_node'" in msg
    # Paragraph 2: breadcrumb block.
    assert "Where it appeared:" in msg
    assert "p_inflow" in msg
    assert "north" in msg
    assert "t0001" in msg
    assert "test_a_lot" in msg
    # Paragraph 3: cardinality sentence.
    assert "11 nodes" in msg
    # Paragraph 4: bullet list with both supplied hints.
    assert "What to do:" in msg
    assert "exists as a commodity" in msg
    assert "did you mean 'north'?" in msg
