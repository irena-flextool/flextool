"""Canonical axis :class:`pl.Enum` dtypes for FlexData dimension columns.

Phase 2 of the ``pl.Enum`` dtype refactor (see
``specs/enum_dtype_refactor_handoff.md``).  This module is pure — no
side effects on import, no globals mutated, no string-cache enablement.
Downstream phases (3+) will:

1.  Call :func:`build_axis_enums` once, immediately after the
    foundational sets (``_load_time`` → ``flex_data.dt`` and
    ``_load_node`` → ``flex_data.nodeBalance``) are populated but
    BEFORE the heavy ``_load_*`` calls run.

2.  Thread the resulting ``dict[str, pl.Enum]`` into every loader
    that returns a frame with dimension columns, and at each loader
    exit call :func:`cast_frame_axes` to convert the dim-column
    dtypes from ``pl.String`` to the canonical Enum.

3.  Use the same dict in :mod:`flextool.engine_polars._derived_params`
    when constructing scratch LazyFrames so their schemas match the
    cast loader outputs (avoids ``SchemaMismatchError`` at join).

The vocabulary choices below were determined by Phase 1's audit
(see ``specs/axis_vocabulary_audit.md``).  Highlights:

* ``d`` / ``t`` — drawn from the canonical ``flex_data.dt`` set.
* ``t_previous`` and variants — share the ``t`` vocabulary.
* ``n`` — from ``flex_data.nodeBalance`` (master field per audit).
* ``p`` — discovered by union over every Param/DataFrame field on
  FlexData that has a ``p`` column (resilient against fixtures
  where ``process_source_sink`` is absent).
* ``source`` / ``sink`` — **union** of ``n`` and ``p``.  The audit
  showed that on the H2_trade y2050 fixture ``source`` / ``sink``
  values are all nodes, but the handoff spec warns the
  ``process_source_sink`` family can carry process names too in
  other fixtures.  A union Enum is correctness-safe and only
  costs a few hundred extra category slots.
* ``c`` / ``g`` / ``e`` / ``f`` / ``i`` — single canonical
  vocabulary per audit (no mixing flagged within a single
  fixture).
* ``d_invest`` / ``d_divest`` / ``d_previous`` / ``d_upper`` /
  ``d_back`` — share the ``d`` vocabulary.
* ``t_upper`` / ``t_back`` — share the ``t`` vocabulary.

Public API
----------

* :func:`build_axis_enums` — produce the dict.
* :func:`cast_frame_axes` — cast a frame's dim columns in-place.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    # Avoid a circular import at module-load time — FlexData lives in
    # ``input.py`` which itself transitively imports this module.
    from flextool.engine_polars.input import FlexData


# ---------------------------------------------------------------------------
# Axis families
#
# Every column name on the LHS gets the Enum dtype of the column name on
# the RHS.  Keeps the public mapping concise: callers see one dtype per
# logical axis, and synonym columns (``d_invest``, ``t_previous``, …)
# pick up the matching dtype automatically.

_AXIS_SYNONYMS: dict[str, str] = {
    # Period synonyms
    "d_invest": "d",
    "d_divest": "d",
    "d_previous": "d",
    "d_upper": "d",
    "d_back": "d",
    # Time-step synonyms
    "t_previous": "t",
    "t_previous_within_block": "t",
    "t_previous_within_timeset": "t",
    "t_previous_within_solve": "t",
    "t_upper": "t",
    "t_back": "t",
    # Friendly long names that occasionally leak from helper code.
    "node": "n",
    "process": "p",
    "commodity": "c",
    "group": "g",
    "entity": "e",
}


def _unique_values_for_column(flex_data: "FlexData", column: str) -> list[str]:
    """Walk every Param / DataFrame field on ``flex_data`` and collect
    the union of unique values seen in ``column``.

    Used as the discovery routine for ``p``, ``c``, ``g``, ``e``, ``f``,
    ``i`` — axes whose canonical source is harder to pin down to a
    single FlexData field (different fixtures populate different
    subsets of FlexData).  Pure read; never modifies a frame.
    """
    # Local import to avoid the circular dependency at module-load.
    from polar_high import Param

    vocab: set[str] = set()
    for f in dataclasses.fields(flex_data):
        val = getattr(flex_data, f.name)
        if val is None:
            continue
        if isinstance(val, Param):
            try:
                frame = val.frame
            except Exception:
                continue
        elif isinstance(val, (pl.DataFrame, pl.LazyFrame)):
            frame = val
        else:
            continue

        # Cheap schema check first — avoids materialising LazyFrames
        # that don't have the column at all.
        try:
            schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) \
                else frame.schema
        except Exception:
            continue
        if column not in schema:
            continue

        try:
            if isinstance(frame, pl.LazyFrame):
                col_vals = (frame.select(pl.col(column).drop_nulls()
                                          .unique())
                                  .collect()[column].to_list())
            else:
                col_vals = frame[column].drop_nulls().unique().to_list()
        except Exception:
            continue
        vocab.update(map(str, col_vals))
    return sorted(vocab)


def _unique_from_frame(frame: pl.DataFrame | None, column: str) -> list[str]:
    """Sorted unique values of ``column`` in ``frame``; ``[]`` if the
    frame is missing or doesn't have the column."""
    if frame is None or column not in frame.columns:
        return []
    return sorted(frame[column].drop_nulls().unique().to_list())


def build_axis_enums(flex_data: "FlexData",
                      workdir: Path | None = None,
                      ) -> dict[str, pl.Enum]:
    """Return a mapping ``{column_name -> pl.Enum}`` covering every
    dimension column that downstream loaders / cascade steps will
    cast.

    Parameters
    ----------
    flex_data
        A ``FlexData`` instance.  At minimum ``flex_data.dt`` and
        ``flex_data.nodeBalance`` must be populated — typical caller
        is right after ``_load_time`` and ``_load_node`` in
        :func:`flextool.engine_polars.input.load_flextool`.  Heavier
        fields (process topology, etc.) may still be ``None`` —
        :func:`build_axis_enums` walks whatever is populated and
        unions the observed vocabularies.
    workdir
        Reserved.  Future variants of this function may consult the
        original workdir CSVs (``input/process.csv``, ``entity.csv``,
        …) to pad the Enum categories beyond what FlexData currently
        materialises — useful when a loader is going to introduce
        entities that aren't yet visible on the bootstrapping
        FlexData snapshot.  Currently unused; signature kept for
        forward compatibility.

    Returns
    -------
    dict[str, pl.Enum]
        Keys are *column names* (``"d"``, ``"t"``, ``"n"``, …),
        including the synonym columns (``"d_invest"``, ``"t_previous"``,
        ``"node"``, …) — every synonym shares the dtype of its
        canonical axis so downstream code can blindly look up the
        column name without worrying about variants.

    Notes
    -----
    * No side effects.  Safe to call repeatedly.
    * ``pl.Enum`` requires globally-unique categories.  We sort and
      de-duplicate every vocabulary.
    * The ``source`` / ``sink`` Enum is the **union** of node and
      process vocabularies.  Phase 1's audit shows this fixture's
      ``source``/``sink`` values are all nodes, but
      ``process_source_sink.csv`` is known to mix node and process
      names in the broader fleet — a union avoids
      ``SchemaMismatchError`` when Phase 3+ lands on those fixtures.
    """
    _ = workdir  # reserved; see docstring

    # ── canonical "time" axes ────────────────────────────────────────
    d_vocab = _unique_from_frame(flex_data.dt, "d")
    t_vocab = _unique_from_frame(flex_data.dt, "t")

    # ── canonical "node" axis ────────────────────────────────────────
    n_vocab = _unique_from_frame(flex_data.nodeBalance, "n")
    # In some fixtures nodeBalance is a strict subset of all nodes ever
    # referenced (e.g., storage-only nodes appearing in nodeState that
    # aren't slack-eligible).  Pad with whatever else we observe.
    n_vocab_extra = _unique_values_for_column(flex_data, "n")
    n_vocab = sorted(set(n_vocab).union(n_vocab_extra))

    # ── discovered axes (vocab unioned from every populated field) ──
    p_vocab = _unique_values_for_column(flex_data, "p")
    c_vocab = _unique_values_for_column(flex_data, "c")
    g_vocab = _unique_values_for_column(flex_data, "g")
    e_vocab = _unique_values_for_column(flex_data, "e")
    f_vocab = _unique_values_for_column(flex_data, "f")
    i_vocab = _unique_values_for_column(flex_data, "i")
    b_vocab = _unique_values_for_column(flex_data, "b")
    b_first_vocab = _unique_values_for_column(flex_data, "b_first")
    b_next_vocab = _unique_values_for_column(flex_data, "b_next")
    r_vocab = _unique_values_for_column(flex_data, "r")
    ud_vocab = _unique_values_for_column(flex_data, "ud")
    td_vocab = _unique_values_for_column(flex_data, "td")

    # ── source / sink union vocabulary ──────────────────────────────
    # Audit: source values are all currently nodes on H2_trade y2050,
    # BUT process_source_sink.csv in other fixtures mixes node and
    # process names (handoff spec §2.3).  Union both to be safe.
    # Also include any values *actually observed* in source/sink
    # columns that aren't in either pure vocabulary (defensive — would
    # indicate a fixture-specific quirk like a synthetic dummy entity).
    src_observed = _unique_values_for_column(flex_data, "source")
    snk_observed = _unique_values_for_column(flex_data, "sink")
    source_sink_vocab = sorted(
        set(n_vocab) | set(p_vocab) | set(src_observed) | set(snk_observed)
    )

    # ── build the Enum dtypes (categories MUST be unique, non-null) ──
    def _enum(values: list[str]) -> pl.Enum:
        # ``pl.Enum`` raises on duplicates; we de-dup defensively.
        return pl.Enum(sorted(set(values)))

    enums: dict[str, pl.Enum] = {}

    # Canonical axes — only register an Enum if we actually have at
    # least one value.  An empty-categories Enum is technically legal
    # but downstream casts of any real string would null out, which
    # is just a foot-gun.
    if d_vocab:
        enums["d"] = _enum(d_vocab)
    if t_vocab:
        enums["t"] = _enum(t_vocab)
    if n_vocab:
        enums["n"] = _enum(n_vocab)
    if p_vocab:
        enums["p"] = _enum(p_vocab)
    if c_vocab:
        enums["c"] = _enum(c_vocab)
    if g_vocab:
        enums["g"] = _enum(g_vocab)
    if e_vocab:
        enums["e"] = _enum(e_vocab)
    if f_vocab:
        enums["f"] = _enum(f_vocab)
    if i_vocab:
        enums["i"] = _enum(i_vocab)
    if b_vocab:
        enums["b"] = _enum(b_vocab)
    if b_first_vocab:
        enums["b_first"] = _enum(b_first_vocab)
    if b_next_vocab:
        enums["b_next"] = _enum(b_next_vocab)
    if r_vocab:
        enums["r"] = _enum(r_vocab)
    if ud_vocab:
        enums["ud"] = _enum(ud_vocab)
    if td_vocab:
        enums["td"] = _enum(td_vocab)
    if source_sink_vocab:
        ss_enum = _enum(source_sink_vocab)
        enums["source"] = ss_enum
        enums["sink"] = ss_enum

    # ── attach synonym columns to their canonical axis dtype ────────
    for synonym, canonical in _AXIS_SYNONYMS.items():
        if canonical in enums:
            enums[synonym] = enums[canonical]

    return enums


def cast_frame_axes(
    frame: "pl.DataFrame | pl.LazyFrame",
    enums: dict[str, pl.Enum],
    *,
    strict: bool = False,
) -> "pl.DataFrame | pl.LazyFrame":
    """Cast every dim column in ``frame`` whose name is in ``enums`` to
    the matching Enum dtype.  Columns absent from ``enums`` (value
    columns, unmapped axes, etc.) are left untouched.

    Parameters
    ----------
    frame
        Eager ``pl.DataFrame`` or lazy ``pl.LazyFrame``.  The return
        type matches the input type — no eager materialisation of
        LazyFrames.
    enums
        The mapping returned by :func:`build_axis_enums`.
    strict
        If ``False`` (default), unknown values silently become null
        on cast — useful during the Phase 3 rollout so a missing
        vocabulary entry surfaces as nulls rather than an exception.
        Flip to ``True`` once the loader cascade is clean.

    Notes
    -----
    Skips columns that are already in the target Enum dtype, so the
    cast is idempotent.
    """
    if isinstance(frame, pl.LazyFrame):
        schema = frame.collect_schema()
        columns = list(schema.keys())
    else:
        schema = frame.schema
        columns = frame.columns

    exprs = []
    for col in columns:
        target = enums.get(col)
        if target is None:
            continue
        if schema[col] == target:
            continue
        exprs.append(pl.col(col).cast(target, strict=strict))
    if not exprs:
        return frame
    return frame.with_columns(exprs)


__all__ = ["build_axis_enums", "cast_frame_axes"]
