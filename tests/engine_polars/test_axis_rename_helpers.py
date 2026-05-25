"""Tests for the FlexTool axis rename + literal helpers.

The Phase-4 cluster sweep landed two convention-enforcing helpers on
``flextool.engine_polars._axis_enums``:

* :func:`rename_to_axis` — rename + cast in one call, for any
  ``.rename({old: new_axis_column_name})`` site.
* :func:`lit_axis` — :func:`pl.lit` with the axis Enum dtype, for any
  literal injection into an axis-aware column.

Both read from the cascade-wide :data:`_LIVE_AXIS_ENUMS` set by
``load_flextool``.  When unset (``None``), they reduce to plain
``.rename`` / ``pl.lit`` — the pre-activation behaviour cascade tests
rely on.

These tests cover both paths (activation off + activation on) plus the
synonym + mixed-vocab resolution that makes the helper transparent at
the call site.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars._axis_enums import (
    alias_to_axis,
    get_global_axis_enums,
    lit_axis,
    rename_to_axis,
    schema_dtype,
    set_global_axis_enums,
)


# ---------------------------------------------------------------------------
# Fixture — populate _LIVE_AXIS_ENUMS with a tiny canonical vocabulary for
# the activation-on tests, and reset on teardown.
# ---------------------------------------------------------------------------


@pytest.fixture
def axis_enums():
    """Small canonical-vocabulary axis enum dict.

    Mirrors what ``build_axis_enums`` would produce for a minimal
    fixture: node + process + commodity + group + period + step + entity
    (union) + side (synthetic) + klass (synthetic).
    """
    return {
        "n": pl.Enum(["coal", "gas", "wind"]),
        "p": pl.Enum(["coal_plant", "gas_chp", "wind_park"]),
        "c": pl.Enum(["electricity", "heat"]),
        "g": pl.Enum(["group_a", "group_b"]),
        "d": pl.Enum(["2030", "2040", "2050"]),
        "t": pl.Enum(["t0", "t1", "t2"]),
        "e": pl.Enum([  # union: node + process
            "coal", "gas", "wind",
            "coal_plant", "gas_chp", "wind_park",
        ]),
        "side": pl.Enum(["source", "sink"]),
        "klass": pl.Enum(["unit", "connection"]),
        "ud": pl.Enum(["up", "down"]),
    }


@pytest.fixture
def activated(axis_enums):
    """Activate the global axis enum vocabulary; reset on teardown.

    Use this fixture for any test that exercises the "activation on"
    code path of ``rename_to_axis`` / ``lit_axis`` / ``schema_dtype``.
    """
    set_global_axis_enums(axis_enums)
    try:
        yield axis_enums
    finally:
        set_global_axis_enums(None)


# ---------------------------------------------------------------------------
# Global-setter / getter
# ---------------------------------------------------------------------------


def test_global_starts_none() -> None:
    """The module-level global defaults to None (no activation)."""
    assert get_global_axis_enums() is None


def test_set_global_axis_enums_roundtrip(axis_enums) -> None:
    """``set_global_axis_enums`` is read back by
    ``get_global_axis_enums``."""
    set_global_axis_enums(axis_enums)
    try:
        assert get_global_axis_enums() is axis_enums
    finally:
        set_global_axis_enums(None)
    assert get_global_axis_enums() is None


# ---------------------------------------------------------------------------
# schema_dtype — synonym resolution
# ---------------------------------------------------------------------------


def test_schema_dtype_resolves_canonical(axis_enums) -> None:
    """Direct canonical-name lookup returns the matching Enum."""
    assert schema_dtype(axis_enums, "n") == axis_enums["n"]
    assert schema_dtype(axis_enums, "e") == axis_enums["e"]


def test_schema_dtype_resolves_synonym(axis_enums) -> None:
    """A column-name synonym resolves to the canonical axis Enum."""
    # period → d
    assert schema_dtype(axis_enums, "period") == axis_enums["d"]
    # node → n
    assert schema_dtype(axis_enums, "node") == axis_enums["n"]
    # source → e (mixed-vocab → union axis)
    assert schema_dtype(axis_enums, "source") == axis_enums["e"]
    # sink → e
    assert schema_dtype(axis_enums, "sink") == axis_enums["e"]


def test_schema_dtype_unknown_falls_back_utf8(axis_enums) -> None:
    """Unknown axis names fall back to pl.Utf8."""
    assert schema_dtype(axis_enums, "made_up") == pl.Utf8


def test_schema_dtype_none_enums_returns_utf8() -> None:
    """When the enums dict is None, every lookup returns pl.Utf8."""
    assert schema_dtype(None, "n") == pl.Utf8
    assert schema_dtype(None, "period") == pl.Utf8
    assert schema_dtype(None, "source") == pl.Utf8


# ---------------------------------------------------------------------------
# rename_to_axis — activation OFF (pre-load_flextool)
# ---------------------------------------------------------------------------


def test_rename_to_axis_passthrough_when_global_unset() -> None:
    """With no global axis_enums set, behaves like a plain
    ``.rename(...)`` — no cast applied.
    """
    df = pl.DataFrame({"node": ["coal", "gas"]})
    out = df.pipe(rename_to_axis, {"node": "n"})
    assert out.columns == ["n"]
    assert out.schema["n"] == pl.Utf8


def test_rename_to_axis_preserves_lazyframe_when_unset() -> None:
    lf = pl.LazyFrame({"node": ["coal"]})
    out = lf.pipe(rename_to_axis, {"node": "n"})
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().columns == ["n"]


# ---------------------------------------------------------------------------
# rename_to_axis — activation ON
# ---------------------------------------------------------------------------


def test_rename_to_axis_casts_canonical_target(activated) -> None:
    """Rename to a canonical axis name casts the column to the
    matching Enum."""
    df = pl.DataFrame({"node": ["coal", "gas"]})
    out = df.pipe(rename_to_axis, {"node": "n"})
    assert out.columns == ["n"]
    assert out.schema["n"] == activated["n"]


def test_rename_to_axis_casts_via_synonym(activated) -> None:
    """Rename to a column-name synonym resolves to the canonical axis
    enum."""
    df = pl.DataFrame({"x": ["2030", "2040"]})
    out = df.pipe(rename_to_axis, {"x": "period"})
    assert out.columns == ["period"]
    assert out.schema["period"] == activated["d"]


def test_rename_to_axis_casts_mixed_vocab_to_entity_union(activated) -> None:
    """source/sink renames cast against the 'e' (entity union) axis,
    not against a same-named single-class axis."""
    df = pl.DataFrame({"node_col": ["coal", "gas_chp"]})
    out = df.pipe(rename_to_axis, {"node_col": "source"})
    assert out.columns == ["source"]
    # The 'e' union axis accepts both node and process names; the cast
    # picks up that enum, not the 'n' (node-only) enum.
    assert out.schema["source"] == activated["e"]


def test_rename_to_axis_skips_non_dim_columns(activated) -> None:
    """Renames whose target name doesn't resolve to a canonical axis
    (data columns like 'method', 'value') leave dtype alone."""
    df = pl.DataFrame({"x": ["constant_efficiency", "linear"]})
    out = df.pipe(rename_to_axis, {"x": "method"})
    assert out.columns == ["method"]
    assert out.schema["method"] == pl.Utf8


def test_rename_to_axis_handles_multiple_columns(activated) -> None:
    """Multiple renames in one call each get their own cast."""
    df = pl.DataFrame({
        "node": ["coal", "gas"],
        "period": ["2030", "2040"],
        "extra": ["a", "b"],
    })
    out = df.pipe(rename_to_axis, {
        "node": "n",
        "period": "d",
        "extra": "method",  # non-dim → no cast
    })
    assert out.columns == ["n", "d", "method"]
    assert out.schema["n"] == activated["n"]
    assert out.schema["d"] == activated["d"]
    assert out.schema["method"] == pl.Utf8


def test_rename_to_axis_lazyframe(activated) -> None:
    """LazyFrame in, LazyFrame out, with the cast pushed into the
    plan."""
    lf = pl.LazyFrame({"node": ["coal", "gas"]})
    out = lf.pipe(rename_to_axis, {"node": "n"})
    assert isinstance(out, pl.LazyFrame)
    collected = out.collect()
    assert collected.schema["n"] == activated["n"]


def test_rename_to_axis_nulls_unknown_tokens(activated) -> None:
    """Non-strict cast: tokens outside the Enum vocabulary become
    null, consistent with cast_frame_axes defaults."""
    df = pl.DataFrame({"node": ["coal", "made_up_node"]})
    out = df.pipe(rename_to_axis, {"node": "n"})
    assert out.schema["n"] == activated["n"]
    # Real token preserved; unknown token nulled.
    assert out["n"].to_list() == ["coal", None]


# ---------------------------------------------------------------------------
# lit_axis — activation OFF and ON
# ---------------------------------------------------------------------------


def test_lit_axis_passthrough_when_unset() -> None:
    """With no global axis_enums, lit_axis is plain pl.lit (Utf8)."""
    df = pl.DataFrame({"x": [1, 2]})
    out = df.with_columns(side=lit_axis("source", "side"))
    assert out.schema["side"] == pl.Utf8
    assert out["side"].to_list() == ["source", "source"]


def test_lit_axis_emits_enum_when_activated(activated) -> None:
    """When activation is on, lit_axis emits the axis Enum dtype."""
    df = pl.DataFrame({"x": [1, 2]})
    out = df.with_columns(side=lit_axis("source", "side"))
    assert out.schema["side"] == activated["side"]
    assert out["side"].to_list() == ["source", "source"]


def test_lit_axis_resolves_synonym(activated) -> None:
    """lit_axis(value, 'node') resolves to the n-axis enum."""
    df = pl.DataFrame({"x": [1]})
    out = df.with_columns(n=lit_axis("coal", "node"))
    assert out.schema["n"] == activated["n"]


def test_lit_axis_unknown_axis_passes_through(activated) -> None:
    """Literal with an unknown axis name returns plain pl.lit (Utf8)."""
    df = pl.DataFrame({"x": [1]})
    out = df.with_columns(method=lit_axis("constant_efficiency", "method"))
    assert out.schema["method"] == pl.Utf8


# ---------------------------------------------------------------------------
# alias_to_axis — activation OFF and ON
# ---------------------------------------------------------------------------


def test_alias_to_axis_passthrough_when_unset() -> None:
    """With no global axis_enums set, behaves like plain
    ``pl.col(X).alias(Y)`` — no cast applied."""
    df = pl.DataFrame({"name": ["coal", "gas"]})
    out = df.select(alias_to_axis("name", "n"))
    assert out.columns == ["n"]
    assert out.schema["n"] == pl.Utf8


def test_alias_to_axis_casts_canonical_target(activated) -> None:
    """Activation on + canonical target → cast to that Enum."""
    df = pl.DataFrame({"name": ["coal", "gas"]})
    out = df.select(alias_to_axis("name", "n"))
    assert out.columns == ["n"]
    assert out.schema["n"] == activated["n"]


def test_alias_to_axis_casts_via_synonym(activated) -> None:
    """Synonym target resolves to canonical axis enum."""
    df = pl.DataFrame({"x": ["2030", "2040"]})
    out = df.select(alias_to_axis("x", "period"))
    assert out.schema["period"] == activated["d"]


def test_alias_to_axis_mixed_vocab_uses_entity_union(activated) -> None:
    """Aliasing as source/sink casts against the 'e' union axis."""
    df = pl.DataFrame({"process_col": ["coal_plant", "gas_chp"]})
    out = df.select(alias_to_axis("process_col", "source"))
    assert out.schema["source"] == activated["e"]


def test_alias_to_axis_non_dim_passthrough(activated) -> None:
    """Aliasing to a data column (non-axis) skips the cast."""
    df = pl.DataFrame({"x": [1.0, 2.0]})
    out = df.select(alias_to_axis("x", "value"))
    assert out.schema["value"] == pl.Float64  # original dtype preserved


def test_alias_to_axis_accepts_expression(activated) -> None:
    """Source can be an arbitrary pl.Expr, not just a column name."""
    df = pl.DataFrame({"a": ["2030"], "b": ["2040"]})
    # Conditional projection: pick "a" when row 0, else "b"
    expr = pl.col("a")  # trivial; real callers use arithmetic / when/then
    out = df.select(alias_to_axis(expr, "period"))
    assert out.schema["period"] == activated["d"]


def test_alias_to_axis_unknown_target_passthrough(activated) -> None:
    """Target name that doesn't resolve to a canonical axis (and isn't
    a known synonym) leaves the alias dtype-untouched."""
    df = pl.DataFrame({"x": [1]})
    out = df.select(alias_to_axis("x", "made_up_target"))
    assert out.schema["made_up_target"] == pl.Int64


def test_alias_to_axis_nulls_unknown_tokens(activated) -> None:
    """Non-strict cast: tokens outside vocabulary become null."""
    df = pl.DataFrame({"name": ["coal", "made_up_node"]})
    out = df.select(alias_to_axis("name", "n"))
    assert out["n"].to_list() == ["coal", None]


# ---------------------------------------------------------------------------
# Documentation-style end-to-end example: rename + literal in a join.
# ---------------------------------------------------------------------------


def test_helpers_compose_in_realistic_join(activated) -> None:
    """The whole-point integration: a rename + a literal injection
    feed into a join that previously SchemaError'd in Enum mode.

    Frame A: process flows; column 'p' is the process axis.
    Frame B: derived flow rows with 'unit' → renamed to 'p' + side
        literal injected.
    Join on 'p' must compose without Enum/Utf8 mismatch.
    """
    frame_a = pl.DataFrame({
        "p": pl.Series(["coal_plant", "gas_chp"], dtype=activated["p"]),
        "qty": [10.0, 20.0],
    })
    frame_b = (
        pl.DataFrame({"unit": ["coal_plant", "gas_chp"]})
        .pipe(rename_to_axis, {"unit": "p"})
        .with_columns(side=lit_axis("source", "side"))
    )
    # Schema check before the join — confirms both helpers are doing
    # their job.
    assert frame_b.schema["p"] == activated["p"]
    assert frame_b.schema["side"] == activated["side"]
    # The join itself.
    joined = frame_a.join(frame_b, on="p")
    assert joined.height == 2
    assert joined.schema["p"] == activated["p"]
