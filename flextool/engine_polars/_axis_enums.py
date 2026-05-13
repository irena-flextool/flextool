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
    "t_source": "t",
    "t_sink": "t",
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


def _read_csv_column(path: "Path", column: str) -> list[str]:
    """Read a single column from a workdir CSV.  Tolerant of missing
    files / missing column / parse errors — returns ``[]``.

    Used during seed-time vocabulary discovery before any ``_load_*``
    function has run.  We can't lean on
    :func:`flextool.engine_polars._input_source._read_csv_file` because
    we want a cheap one-column read and don't care about caching.
    """
    try:
        if not path.exists():
            return []
        df = pl.read_csv(path, infer_schema_length=0)
    except Exception:
        return []
    if column not in df.columns:
        return []
    try:
        return [str(v) for v in df[column].drop_nulls().unique().to_list()]
    except Exception:
        return []


def _seed_vocab_from_workdir(workdir: "Path | None") -> dict[str, set[str]]:
    """Read a handful of canonical workdir CSVs and produce per-axis
    vocabulary seeds.  Pads the Enum categories so casts performed
    early in the loader (before the full FlexData is materialised)
    cover entities those loaders will introduce.

    Tolerant of every kind of missing file — empty dict on failure.
    """
    out: dict[str, set[str]] = {}
    if workdir is None:
        return out
    inp = workdir / "input"
    sd  = workdir / "solve_data"

    # entity.csv — superset of every entity-typed axis (p, n, e, …).
    e_vocab: set[str] = set()
    e_vocab.update(_read_csv_column(inp / "entity.csv", "entity"))

    # process.csv would be the canonical p source but it's not always
    # present; instead get the p-vocabulary from process_source_sink.csv
    # in solve_data (always emitted by preprocessing for non-trivial
    # solves) plus entity-class membership.
    p_vocab: set[str] = set()
    p_vocab.update(_read_csv_column(sd / "process_source_sink.csv", "process"))
    p_vocab.update(_read_csv_column(inp / "process.csv", "process"))

    # node — sd/nodeBalance is read by _load_node directly; for the
    # Enum we also pad with entities of class node (input/node.csv).
    n_vocab: set[str] = set()
    n_vocab.update(_read_csv_column(inp / "node.csv", "node"))
    n_vocab.update(_read_csv_column(sd / "nodeBalance.csv", "node"))

    # commodity / group / etc.
    c_vocab: set[str] = set()
    c_vocab.update(_read_csv_column(inp / "commodity.csv", "commodity"))
    c_vocab.update(_read_csv_column(sd / "p_commodity.csv", "commodity"))

    g_vocab: set[str] = set()
    g_vocab.update(_read_csv_column(inp / "group.csv", "group"))

    f_vocab: set[str] = set()
    f_vocab.update(_read_csv_column(inp / "profile.csv", "profile"))

    if e_vocab or n_vocab or p_vocab:
        # Every entity also belongs to ``e``.  Seed e with the union of
        # node + process + raw entity vocab.
        e_vocab |= n_vocab | p_vocab
        out["e"] = e_vocab
    if n_vocab:
        out["n"] = n_vocab
    if p_vocab:
        out["p"] = p_vocab
    if c_vocab:
        out["c"] = c_vocab
    if g_vocab:
        out["g"] = g_vocab
    if f_vocab:
        out["f"] = f_vocab
    return out


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
    # Seed vocabularies from the canonical workdir CSVs *before* we walk
    # the (possibly empty) FlexData.  The walking below then unions in
    # anything FlexData has already materialised — they should be a
    # subset, but the union is defensive against fixture quirks.
    workdir_seeds = _seed_vocab_from_workdir(workdir)

    # ── canonical "time" axes ────────────────────────────────────────
    d_vocab = _unique_from_frame(flex_data.dt, "d")
    t_vocab = _unique_from_frame(flex_data.dt, "t")

    # ── canonical "node" axis ────────────────────────────────────────
    n_vocab = _unique_from_frame(flex_data.nodeBalance, "n")
    # In some fixtures nodeBalance is a strict subset of all nodes ever
    # referenced (e.g., storage-only nodes appearing in nodeState that
    # aren't slack-eligible).  Pad with whatever else we observe.
    n_vocab_extra = _unique_values_for_column(flex_data, "n")
    n_vocab = sorted(set(n_vocab).union(n_vocab_extra)
                      .union(workdir_seeds.get("n", set())))

    # ── discovered axes (vocab unioned from every populated field) ──
    p_vocab = sorted(set(_unique_values_for_column(flex_data, "p"))
                      .union(workdir_seeds.get("p", set())))
    c_vocab = sorted(set(_unique_values_for_column(flex_data, "c"))
                      .union(workdir_seeds.get("c", set())))
    g_vocab = sorted(set(_unique_values_for_column(flex_data, "g"))
                      .union(workdir_seeds.get("g", set())))
    e_vocab = sorted(set(_unique_values_for_column(flex_data, "e"))
                      .union(workdir_seeds.get("e", set())))
    f_vocab = sorted(set(_unique_values_for_column(flex_data, "f"))
                      .union(workdir_seeds.get("f", set())))
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
    # NOTE on entity-like axes: ``e``, ``p``, ``n``, ``source``, ``sink``
    # all reference *entities* (nodes / processes / connections).  Code
    # in the loader frequently renames between these column names
    # (``cap_long.rename({"e": "p"})``) — but rename does NOT change
    # the column dtype.  If ``e`` and ``p`` carry different Enum dtypes,
    # the post-rename ``p`` column carries ``Enum(e-vocab)`` and any
    # downstream join against an Enum(p-vocab) column raises
    # ``SchemaMismatchError``.  To eliminate this entire failure mode
    # we use a SINGLE entity Enum — the union of node + process +
    # observed source/sink — for every entity-like axis.  The category
    # set is only modestly larger than per-axis Enums and the memory
    # cost of one entity-dictionary entry is trivial vs. the row-level
    # savings.
    entity_vocab = sorted(
        set(n_vocab) | set(p_vocab) | set(e_vocab) | set(source_sink_vocab)
    )
    # ``b`` is period-like — ``period__branch`` in the stochastic
    # helper renames between ``d`` and ``b`` freely.  Unify with the
    # period vocabulary for the same rename-safety reason.
    #
    # ``b_first``, ``b_next`` are TIME-like — they appear in
    # ``period_block_succ`` and ``model.py``'s storage-block algebra
    # carries time values (renames of ``v_state.t``).  We unify
    # ``b_first`` / ``b_next`` with the ``t`` vocabulary instead.
    period_vocab = sorted(
        set(d_vocab) | set(b_vocab)
    )
    time_vocab = sorted(
        set(t_vocab) | set(b_first_vocab) | set(b_next_vocab)
    )

    if period_vocab:
        per_enum = _enum(period_vocab)
        enums["d"] = per_enum
        enums["b"] = per_enum
    if time_vocab:
        t_enum = _enum(time_vocab)
        enums["t"] = t_enum
        enums["b_first"] = t_enum
        enums["b_next"] = t_enum
    if entity_vocab:
        ent_enum = _enum(entity_vocab)
        enums["e"] = ent_enum
        enums["p"] = ent_enum
        enums["n"] = ent_enum
        enums["source"] = ent_enum
        enums["sink"] = ent_enum
    if c_vocab:
        enums["c"] = _enum(c_vocab)
    if g_vocab:
        enums["g"] = _enum(g_vocab)
    if f_vocab:
        enums["f"] = _enum(f_vocab)
    if i_vocab:
        enums["i"] = _enum(i_vocab)
    if r_vocab:
        enums["r"] = _enum(r_vocab)
    if ud_vocab:
        enums["ud"] = _enum(ud_vocab)
    if td_vocab:
        enums["td"] = _enum(td_vocab)

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


def align_join_dtypes(left: "pl.LazyFrame | pl.DataFrame",
                        right: "pl.LazyFrame | pl.DataFrame",
                        cols: list[str] | tuple[str, ...]):
    """Make the dtypes of ``cols`` on ``left`` and ``right`` match.

    Strategy: prefer the Enum dtype if either side has one (Enum joins
    are zero-copy when both sides match exactly); fall back to
    ``pl.Utf8`` otherwise.  Returns the (possibly modified) (left,
    right) pair — no schema rebuild when the dtypes already agree.

    Used to bridge the cascade-helper boundary where one side comes
    from a fresh CSV read (String) and the other from an Enum-cast
    FlexData field.
    """
    if isinstance(left, pl.LazyFrame):
        lschema = left.collect_schema()
    else:
        lschema = left.schema
    if isinstance(right, pl.LazyFrame):
        rschema = right.collect_schema()
    else:
        rschema = right.schema

    left_casts = []
    right_casts = []
    for c in cols:
        ld = lschema.get(c)
        rd = rschema.get(c)
        if ld is None or rd is None or ld == rd:
            continue
        # Prefer Enum side; otherwise fall back to String coercion.
        if isinstance(ld, pl.Enum):
            target = ld
            right_casts.append(pl.col(c).cast(target, strict=False))
        elif isinstance(rd, pl.Enum):
            target = rd
            left_casts.append(pl.col(c).cast(target, strict=False))
        else:
            # Two non-Enum non-matching dtypes — cast both to String.
            left_casts.append(pl.col(c).cast(pl.Utf8, strict=False))
            right_casts.append(pl.col(c).cast(pl.Utf8, strict=False))
    if left_casts:
        left = left.with_columns(left_casts)
    if right_casts:
        right = right.with_columns(right_casts)
    return left, right


def empty_like(frame: "pl.DataFrame | pl.LazyFrame",
                 columns: list[str] | tuple[str, ...],
                 extra: dict[str, "pl.DataType"] | None = None,
                 *,
                 lazy: bool = False) -> "pl.DataFrame | pl.LazyFrame":
    """Build an empty frame whose dim-column dtypes match ``frame``.

    Designed for cascade scratch frames that were previously declared
    as ``pl.LazyFrame(schema={"e": pl.Utf8, "d": pl.Utf8, ...})`` —
    those break the moment ``frame`` is cast to Enum.  This helper
    inspects ``frame``'s schema for each requested column and reuses
    its dtype, falling back to ``pl.Utf8`` only when the column is
    absent from ``frame``.  ``extra`` lets the caller add value-typed
    columns (typically ``"value": pl.Float64``) that aren't in
    ``frame``.

    Usage::

        bounded_walk = empty_like(anchor_lf, ["e", "d", "d_all"], lazy=True)

    ``frame`` may itself be eager or lazy.
    """
    if isinstance(frame, pl.LazyFrame):
        schema = frame.collect_schema()
    else:
        schema = frame.schema
    out_schema: dict[str, "pl.DataType"] = {}
    for c in columns:
        out_schema[c] = schema.get(c, pl.Utf8)
    if extra:
        out_schema.update(extra)
    if lazy:
        return pl.LazyFrame(schema=out_schema)
    return pl.DataFrame(schema=out_schema)


def cast_dim(col_expr: "pl.Expr",
              enums: "dict[str, pl.Enum] | None",
              axis: str) -> "pl.Expr":
    """Align a populated-frame ``pl.Expr`` with its empty-frame
    :func:`schema_dtype` counterpart.

    Used at every cascade-helper site that pairs an empty-frame branch
    declared with ``schema_dtype(_enums, axis)`` against a populated
    branch that emits dim columns from raw String CSV reads.  When
    ``enums`` is populated, casts ``col_expr`` to the canonical Enum
    dtype for ``axis``; otherwise (and for axes absent from the
    mapping) returns the expression unchanged — preserving current
    behaviour when ``_enums`` is ``None``.

    The non-strict cast nulls out values not in the Enum vocabulary
    (consistent with :func:`cast_frame_axes` defaults).
    """
    if enums is None:
        return col_expr
    dt = enums.get(axis)
    if dt is None:
        return col_expr
    return col_expr.cast(dt, strict=False)


def schema_dtype(enums: "dict[str, pl.Enum] | None",
                   axis: str) -> "pl.DataType":
    """Return the Enum dtype for ``axis`` if ``enums`` is populated.

    Otherwise fall back to :class:`pl.Utf8` (the current default).

    Designed for scratch-frame schema declarations in the broadcast
    cascade and adjacent helpers.  Each site that previously hard-coded
    ``schema={"e": pl.Utf8, "d": pl.Utf8}`` becomes::

        _enums = getattr(flex_data, "_axis_enums", None)
        return pl.DataFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
        })

    With ``flex_data._axis_enums is None`` (the current default during
    the cascade) the lookup returns ``pl.Utf8`` and behavior is
    identical to the hard-coded form.  When a future dispatch sets
    ``flex_data._axis_enums`` before the cascade runs, the scratch
    frames pick up the canonical Enum dtype automatically.
    """
    if enums is None:
        return pl.Utf8
    return enums.get(axis, pl.Utf8)


def cast_value_axes(value, enums: dict[str, pl.Enum], *, strict: bool = False):
    """Recursively cast a value's dim columns to the canonical Enums.

    Handles:
      * :class:`polar_high.Param` — rebuilds with cast frame.
      * :class:`pl.DataFrame` / :class:`pl.LazyFrame` — applies
        :func:`cast_frame_axes`.
      * ``dict`` / ``tuple`` / ``list`` — recursively walks and rebuilds
        the same container shape with cast children.
      * Everything else — returned unchanged.

    Used by :func:`load_flextool` to wrap the return values of each
    ``_load_*`` function in a single shot at the call site, so the
    interior of every loader doesn't need an individual cast injection.
    """
    # Late import to avoid circular dependency.
    from polar_high import Param

    if value is None:
        return value
    if isinstance(value, Param):
        # Operate on the lazy form to avoid forcing a collect() — every
        # Param keeps an internal ``lazy`` LazyFrame regardless of whether
        # ``.frame`` has been materialised yet.
        try:
            cast_lazy = cast_frame_axes(value.lazy, enums, strict=strict)
        except Exception:
            return value
        if cast_lazy is value.lazy:
            return value
        return Param(value.dims, cast_lazy,
                      name=getattr(value, "name", None),
                      _sources=getattr(value, "_sources", None))
    if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
        try:
            return cast_frame_axes(value, enums, strict=strict)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: cast_value_axes(v, enums, strict=strict)
                for k, v in value.items()}
    if isinstance(value, list):
        return [cast_value_axes(v, enums, strict=strict) for v in value]
    if isinstance(value, tuple):
        return tuple(cast_value_axes(v, enums, strict=strict) for v in value)
    return value


def cast_flexdata_axes(flex_data: "FlexData",
                        enums: dict[str, pl.Enum],
                        *, strict: bool = False) -> "FlexData":
    """Walk every Param / DataFrame / LazyFrame field on ``flex_data``
    and cast dim columns to the canonical Enums in place.  Returns the
    same FlexData (mutated) for fluent use.

    Skips ``block_layout`` and other non-frame fields.  Skips fields
    whose value is ``None``.
    """
    from polar_high import Param

    for f in dataclasses.fields(flex_data):
        val = getattr(flex_data, f.name, None)
        if val is None:
            continue
        if isinstance(val, (Param, pl.DataFrame, pl.LazyFrame)):
            new_val = cast_value_axes(val, enums, strict=strict)
            if new_val is not val:
                setattr(flex_data, f.name, new_val)
    return flex_data


__all__ = [
    "build_axis_enums",
    "cast_dim",
    "cast_frame_axes",
    "cast_value_axes",
    "cast_flexdata_axes",
    "empty_like",
    "align_join_dtypes",
    "schema_dtype",
]
