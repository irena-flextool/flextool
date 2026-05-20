"""Production :class:`InputSource` implementation backed by SpineDB.

Wraps :class:`spinedb_api.DatabaseMapping`, applies a scenario filter
once at construction, and exposes per-(entity_class, parameter_name)
polars frames on demand.  See ``audit/db_direct_param_map.md §5`` for
the binding spec.

Lazy-evaluation pattern
-----------------------
The constructor materialises three eager caches: the per-class entity
universe, the per-(class, parameter) row list, and the per-(class,
parameter) default.  Subsequent :meth:`parameter` calls walk the
cached rows once, build the result with :class:`polars.LazyFrame`
chains (default broadcast / left-join / cast), and ``.collect()`` once
at the boundary.  Callers receive eager DataFrames (the API contract);
internal composition stays lazy so polars can fuse / optimise the
chain.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import polars as pl

from flextool.spinedb_backend._axis_enums import (
    AxisContract,
    FlexDataIntegrityError,
    cast_against_contract,
    load_axis_contract,
)

# Late imports of spinedb_api at construction time keep the import
# graph free of an unconditional dependency for users who only consume
# CSVs.


# ---------------------------------------------------------------------------
# Helpers


def _coerce_index(idx: Any) -> Any:
    """Coerce a single index into a Python type polars likes.

    ``Map.indexes`` returns ``numpy.str_`` (and equivalent numpy
    scalar types); polars happily accepts these but constructing a
    :class:`pl.DataFrame` mixed-source from them is more reliable when
    they're plain Python.  Datetimes (``numpy.datetime64`` from
    :class:`TimeSeries`) get returned to polars as-is — polars converts
    them on construction.
    """
    if isinstance(idx, np.str_):
        return str(idx)
    if isinstance(idx, np.generic):
        # numpy.float64 / int64 / bool_ → Python scalar.
        return idx.item()
    return idx


def _coerce_value(v: Any) -> Any:
    """Coerce a leaf value into a Python type with reasonable polars
    dtype inference.  Strings stay strings; numeric numpy scalars become
    Python floats / ints; booleans stay booleans; everything else
    passes through (polars will reject blob-like values via the schema
    check downstream).
    """
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


# Default canonical column names per spec §4.2.  Used when a `Map.index_name`
# is empty / generic.  Probed in order of nesting depth so the topmost
# (period) gets the first slot.
_DEFAULT_INDEX_NAMES = ("period", "tier", "branch", "t", "sub_index")


# ---------------------------------------------------------------------------
# SpineDbReader


class SpineDbReader:
    """Read flextool input data directly from a SpineDB scenario.

    Parameters
    ----------
    db_url : str
        Spine sqlite URL.  Bare paths are auto-prefixed with
        ``sqlite:///``.
    scenario : str
        Scenario name to apply via
        :func:`spinedb_api.filters.scenario_filter.apply_scenario_filter_to_subqueries`.

    The constructor opens the DB, applies the filter, and pre-builds
    three caches:

    * ``_entities[class_id]``: list of (entity_id, name, element_name_list).
    * ``_param_rows[(class_id, pdef_id)]``: list of (entity_id, value, type).
    * ``_param_defs[(class_id, pdef_name)]``: pdef row + parsed default.

    The DB is closed at construction time after the caches are built —
    no DB handle is held across calls.
    """

    # ------------------------------------------------------------------
    # Construction

    def __init__(
        self,
        db_url: str,
        scenario: str,
        *,
        axis_enums: dict[str, pl.Enum] | None = None,
        contract: AxisContract | None = None,
    ):
        url = str(db_url)
        if not url.startswith("sqlite:") and not url.startswith("postgresql"):
            url = f"sqlite:///{url}"
        self._db_url = url
        self._scenario = scenario
        # Phase 2 cast-on-emit: when ``axis_enums`` is supplied, every
        # frame returned by entities / parameter / parameter_explicit is
        # cast against the contract before return.  ``None`` keeps the
        # pre-Phase-2 behaviour (Utf8 dim columns) so callers that
        # haven't opted in yet see no change.
        self._axis_enums = axis_enums
        if contract is None and axis_enums is not None:
            contract = load_axis_contract()
        self._contract = contract

        # Build caches once.
        from spinedb_api import DatabaseMapping, from_database
        from spinedb_api.filters.scenario_filter import (
            apply_scenario_filter_to_subqueries,
        )

        with DatabaseMapping(url) as db:
            apply_scenario_filter_to_subqueries(db, scenario)

            # Class id ↔ name map.
            self._class_id_to_name: dict[int, str] = {}
            self._class_name_to_id: dict[str, int] = {}
            for c in db.query(db.entity_class_sq).all():
                self._class_id_to_name[c.id] = c.name
                self._class_name_to_id[c.name] = c.id

            # Class dim names (from wide subquery).  None for 0-dim.
            self._class_dim_names: dict[int, list[str] | None] = {}
            for c in db.query(db.wide_entity_class_sq).all():
                if c.dimension_name_list:
                    self._class_dim_names[c.id] = c.dimension_name_list.split(",")
                else:
                    self._class_dim_names[c.id] = None

            # Per-class entities (via wide_entity_sq for element_name_list).
            self._entities_by_class: dict[int, list[tuple[int, str, list[str] | None]]] = (
                defaultdict(list)
            )
            for e in db.query(db.wide_entity_sq).all():
                if e.element_name_list:
                    elements = e.element_name_list.split(",")
                else:
                    elements = None
                self._entities_by_class[e.class_id].append(
                    (e.id, e.name, elements)
                )

            # Parameter definitions (cached eagerly, parsed defaults).
            self._pdef_by_class_name: dict[tuple[int, str], dict] = {}
            for p in db.query(db.wide_parameter_definition_sq).all():
                default_val = from_database(p.default_value, p.default_type)
                self._pdef_by_class_name[(p.entity_class_id, p.name)] = {
                    "id": p.id,
                    "name": p.name,
                    "default_value": default_val,
                    "default_type": p.default_type,
                    "value_list_id": p.parameter_value_list_id,
                }

            # Per-(class_id, pdef_id) value rows.  We load eagerly,
            # parsing values via ``from_database`` once, so per-call
            # ``parameter()`` is pure polars assembly.
            self._param_rows: dict[tuple[int, int], list[tuple[int, Any]]] = (
                defaultdict(list)
            )
            for r in db.query(db.parameter_value_sq).all():
                v = from_database(r.value, r.type)
                self._param_rows[(r.entity_class_id, r.parameter_definition_id)] \
                    .append((r.entity_id, v))

            # entity_id → (class_id, name) for joining.
            self._entity_by_id: dict[int, tuple[int, str]] = {}
            for cls_id, ents in self._entities_by_class.items():
                for eid, name, _ in ents:
                    self._entity_by_id[eid] = (cls_id, name)

    # ------------------------------------------------------------------
    # Public diagnostics

    @property
    def db_url(self) -> str:
        return self._db_url

    @property
    def scenario(self) -> str:
        return self._scenario

    def __repr__(self) -> str:
        return f"SpineDbReader(db_url={self._db_url!r}, scenario={self._scenario!r})"

    # ------------------------------------------------------------------
    # Entity columns: ``[name]`` for 0-dim classes; one column per dim
    # (named after the dim class) for n-dim relationships.  Repeated
    # dim classes get a 1-based suffix.

    def _entity_columns(self, class_id: int) -> list[str]:
        """Resolve the column names for the given class's entity frame.

        For 0-dim classes returns ``["name"]``.  For n-dim relationships
        returns the dim-class names, with repeats disambiguated by a
        ``_N`` suffix (e.g. ``connection__node__node`` →
        ``["connection", "node_1", "node_2"]``).
        """
        dims = self._class_dim_names.get(class_id)
        if dims is None:
            return ["name"]
        # Disambiguate duplicates.
        seen: dict[str, int] = defaultdict(int)
        cols: list[str] = []
        # First pass: count occurrences.
        counts: dict[str, int] = defaultdict(int)
        for d in dims:
            counts[d] += 1
        # Second pass: emit, suffixing only those with multiplicity > 1.
        for d in dims:
            if counts[d] > 1:
                seen[d] += 1
                cols.append(f"{d}_{seen[d]}")
            else:
                cols.append(d)
        return cols

    # ------------------------------------------------------------------
    # InputSource Protocol — entities

    def entities(self, entity_class: str) -> pl.DataFrame:
        cls_id = self._class_name_to_id.get(entity_class)
        if cls_id is None:
            # Unknown class — empty frame with a sensible schema.
            return pl.DataFrame(schema={"name": pl.Utf8})
        cols = self._entity_columns(cls_id)
        rows = self._entities_by_class.get(cls_id, [])
        if not rows:
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
        if len(cols) == 1:
            frame = (
                pl.DataFrame({cols[0]: [name for _, name, _ in rows]},
                             schema={cols[0]: pl.Utf8})
                .sort(cols)
            )
            return self._maybe_cast_frame(
                frame,
                entity_class=entity_class,
                parameter_name=None,
            )
        # Multi-dim: split element_name_list into N columns.
        data: dict[str, list[str]] = {c: [] for c in cols}
        for _, _ent_name, elements in rows:
            if elements is None or len(elements) != len(cols):
                # Defensive: skip malformed rows rather than crash.
                continue
            for c, v in zip(cols, elements):
                data[c].append(v)
        schema = {c: pl.Utf8 for c in cols}
        frame = pl.DataFrame(data, schema=schema).sort(cols)
        return self._maybe_cast_frame(
            frame,
            entity_class=entity_class,
            parameter_name=None,
        )

    # ------------------------------------------------------------------
    # InputSource Protocol — parameter

    def parameter_default(self, entity_class: str, parameter_name: str) -> Any:
        cls_id = self._class_name_to_id.get(entity_class)
        if cls_id is None:
            return None
        pdef = self._pdef_by_class_name.get((cls_id, parameter_name))
        if pdef is None:
            return None
        return pdef["default_value"]

    # ------------------------------------------------------------------
    # Δ.17c — raw per-level index_name labels for a parameter.
    #
    # Used by :func:`flextool.engine_polars._param_shapes.resolve_param_shape`
    # to validate the parameter's shape against an explicit allow-list
    # (per the user's directive "read from database the dimensionality
    # of the parameter" + "read the dimension index label from the
    # database").
    #
    # Returns the list of raw ``Map.index_name`` labels per nesting
    # depth: empty list for scalars; one entry per Map level for
    # n-dim Maps.  Labels are returned exactly as the DB authored them
    # (no normalisation, no canonical default substitution) so the
    # caller can detect "wrong index_name" and raise loudly.
    #
    # ``TimeSeries`` / ``Array`` collapse to a single canonical label
    # (``"time"`` / ``""`` respectively) — neither is valid for the
    # registry-routed parameters today, but rejecting them with an
    # accurate label is better than silently treating them as a Map.
    def parameter_shape_info(self, entity_class: str,
                              parameter_name: str) -> "list[str | None]":
        cls_id = self._class_name_to_id.get(entity_class)
        if cls_id is None:
            raise KeyError(f"unknown entity_class {entity_class!r}")
        pdef = self._pdef_by_class_name.get((cls_id, parameter_name))
        if pdef is None:
            raise KeyError(
                f"unknown parameter ({entity_class!r}, {parameter_name!r})"
            )
        rows = self._param_rows.get((cls_id, pdef["id"]), [])
        # Probe the deepest-nested row to capture the widest schema.
        # Spine schema invariant: all rows for a parameter under a
        # scenario share the same shape, but we're defensive in case
        # a fixture mixes shapes (the resolver caller will raise on
        # ambiguity downstream).
        deepest: list[str | None] = []
        for _eid, v in rows:
            cand = self._index_name_path(v)
            if len(cand) > len(deepest):
                deepest = cand
        if deepest:
            return deepest
        # No explicit rows — inspect the parameter's default_value (the
        # schema default may itself be a Map).  Otherwise treat as
        # scalar (depth 0).
        default = pdef["default_value"]
        if default is not None:
            return self._index_name_path(default)
        return []

    @staticmethod
    def _index_name_path(v: Any) -> "list[str | None]":
        """Walk *v*'s nesting and return raw ``Map.index_name`` labels
        per depth level.  Used by :meth:`parameter_shape_info`.

        Differs from :meth:`_discover_index_cols` in that it returns
        the raw DB labels (``None`` / empty when unset) instead of the
        canonical defaults — that's the whole point of the Δ.17c
        resolver path.
        """
        from spinedb_api.parameter_value import (
            Map, TimeSeries, Array,
        )
        out: list[str | None] = []
        cur = v
        while True:
            if isinstance(cur, Map):
                out.append(cur.index_name if cur.index_name else None)
                if len(cur.values) == 0:
                    break
                cur = cur.values[0]
                continue
            if isinstance(cur, TimeSeries):
                out.append("time")
                break
            if isinstance(cur, Array):
                out.append("")
                break
            break
        return out

    def parameter_explicit(self, entity_class: str,
                             parameter_name: str) -> pl.DataFrame:
        """Like :meth:`parameter` but suppresses default-broadcast rows.

        Returns ONLY entities that have a row in the parameter_value
        table for the active scenario — i.e. an explicit override.  The
        Spine schema default never appears.

        Useful for helpers that mirror flextool's preprocessing
        ``p_unit.get(name, None)`` semantic, where "absent" must remain
        distinguishable from "present with the default value".
        """
        cls_id = self._class_name_to_id.get(entity_class)
        if cls_id is None:
            raise KeyError(f"unknown entity_class {entity_class!r}")
        pdef = self._pdef_by_class_name.get((cls_id, parameter_name))
        if pdef is None:
            raise KeyError(
                f"unknown parameter ({entity_class!r}, {parameter_name!r})"
            )
        ent_cols = self._entity_columns(cls_id)
        rows = self._param_rows.get((cls_id, pdef["id"]), [])
        columns, index_cols, leaf_dtype = self._unroll_rows(
            rows, ent_cols, parameter_name,
        )
        if not columns or not columns["value"]:
            schema_in = {c: pl.Utf8 for c in ent_cols}
            for ic in index_cols:
                schema_in[ic] = pl.Utf8
            schema_in["value"] = leaf_dtype or pl.Float64
            return self._maybe_cast_frame(
                pl.DataFrame(schema=schema_in),
                entity_class=entity_class,
                parameter_name=parameter_name,
            )
        overrides = {"value": leaf_dtype} if leaf_dtype is not None else None
        # ``strict=False`` preserves the tolerance of the pre-columnar
        # ``pl.DataFrame(list_of_dicts)`` constructor for heterogeneous
        # Map-index columns (e.g. solve.invest_periods with mixed
        # String / Int64 indexes across entities).
        frame = self._finalize(
            pl.DataFrame(columns, schema_overrides=overrides, strict=False).lazy(),
            ent_cols + index_cols,
        )
        return self._maybe_cast_frame(
            frame,
            entity_class=entity_class,
            parameter_name=parameter_name,
        )

    def parameter(self, entity_class: str, parameter_name: str) -> pl.DataFrame:
        cls_id = self._class_name_to_id.get(entity_class)
        if cls_id is None:
            raise KeyError(f"unknown entity_class {entity_class!r}")
        pdef = self._pdef_by_class_name.get((cls_id, parameter_name))
        if pdef is None:
            raise KeyError(
                f"unknown parameter ({entity_class!r}, {parameter_name!r})"
            )

        ent_cols = self._entity_columns(cls_id)
        rows = self._param_rows.get((cls_id, pdef["id"]), [])
        default = pdef["default_value"]

        # Step 1 — unroll each row into per-column lists.  The shape
        # depends on the value's runtime type (scalar / Map / TimeSeries
        # / Array).  We don't know up-front whether the parameter is
        # "scalar across all rows"; we infer per-row.
        columns, index_cols, leaf_dtype = self._unroll_rows(
            rows, ent_cols, parameter_name,
        )

        # Step 2 — assemble the per-parameter LazyFrame.  Empty rows
        # collapse to a 0-row frame with the correct schema.
        if not columns or not columns["value"]:
            schema_in = {c: pl.Utf8 for c in ent_cols}
            for ic in index_cols:
                schema_in[ic] = pl.Utf8
            schema_in["value"] = leaf_dtype or pl.Float64
            v_lf = pl.DataFrame(schema=schema_in).lazy()
        else:
            overrides = {"value": leaf_dtype} if leaf_dtype is not None else None
            # ``strict=False`` preserves pre-columnar tolerance of mixed
            # Map-index types (see _spinedb_reader.py:415).
            v_lf = pl.DataFrame(
                columns, schema_overrides=overrides, strict=False,
            ).lazy()

        # Step 3 — apply the §4.5 default policy lazily.
        if default is None:
            # None-skip: return rows as-is.
            frame = self._finalize(v_lf, ent_cols + index_cols)
            return self._maybe_cast_frame(
                frame,
                entity_class=entity_class,
                parameter_name=parameter_name,
            )

        if not index_cols:
            # Scalar-default + scalar-parameter → broadcast.  Build the
            # entities frame from the raw cache (not via self.entities())
            # so the join keys stay Utf8 — casting happens once at the
            # end against the joined frame.
            E = self._entities_frame_raw(cls_id, ent_cols).lazy()
            joined = (E.join(v_lf, on=ent_cols, how="left")
                       .with_columns(pl.col("value").fill_null(default)))
            frame = self._finalize(joined, ent_cols)
            return self._maybe_cast_frame(
                frame,
                entity_class=entity_class,
                parameter_name=parameter_name,
            )

        # Scalar-default + indexed-parameter → return overrides only;
        # the default is consumed via parameter_default() upstream.
        frame = self._finalize(v_lf, ent_cols + index_cols)
        return self._maybe_cast_frame(
            frame,
            entity_class=entity_class,
            parameter_name=parameter_name,
        )

    def _entities_frame_raw(
        self, cls_id: int, cols: list[str],
    ) -> pl.DataFrame:
        """Internal — build the raw Utf8 entities frame for *cls_id*
        without going through :meth:`entities` (which would cast).

        Used by :meth:`parameter` to keep the join keys Utf8 so the
        join is a String/String op (polars 1.40 won't auto-coerce
        Utf8↔Enum on join keys); the final frame is cast once before
        return.
        """
        rows = self._entities_by_class.get(cls_id, [])
        if not rows:
            return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
        if len(cols) == 1:
            return (
                pl.DataFrame({cols[0]: [name for _, name, _ in rows]},
                             schema={cols[0]: pl.Utf8})
                .sort(cols)
            )
        data: dict[str, list[str]] = {c: [] for c in cols}
        for _, _ent_name, elements in rows:
            if elements is None or len(elements) != len(cols):
                continue
            for c, v in zip(cols, elements):
                data[c].append(v)
        return pl.DataFrame(
            data, schema={c: pl.Utf8 for c in cols},
        ).sort(cols)

    def _maybe_cast_frame(
        self,
        frame: pl.DataFrame,
        *,
        entity_class: str,
        parameter_name: str | None,
    ) -> pl.DataFrame:
        """Cast *frame* against the configured axis enums + contract.

        When the reader was constructed with ``axis_enums=None``
        (default), the frame passes through unchanged — pre-Phase-2
        back-compat.  When ``axis_enums`` is non-None, every dim
        column resolved via :meth:`AxisContract.column_to_axis` is
        cast strictly; a vocabulary miss raises
        :class:`FlexDataIntegrityError` with
        ``(parameter, entity_class, scenario)`` threaded as the
        origin breadcrumb.

        Phase 4 — for 0-dim entity frames whose single column is
        ``name``, cast that column against the axis whose source class
        is ``entity_class`` (looked up in the contract).  Without this
        hook the ``name`` column remains Utf8 and every downstream
        ``pl.col("name").alias("p")`` produces a Utf8 ``p`` that mixes
        SchemaError-fully with the Enum-typed cascade.
        """
        if self._axis_enums is None:
            return frame
        # Pre-cast: handle entity-class element columns.  For 0-dim
        # classes the single column is ``name``; for n-dim relationships
        # the columns are the dim-class names (``unit``, ``node``,
        # ``node_1``, ``node_2``, ``connection`` …).  Each such column
        # carries entities of its dim class, so we cast against the
        # axis whose source claims that class.
        if self._contract is not None:
            element_casts: list[pl.Expr] = []
            for col in frame.columns:
                if col == "name":
                    axis = self._axis_for_entity_class(entity_class)
                elif col == "value":
                    # ``value`` is data, never a dim column.
                    axis = None
                else:
                    # ``unit``, ``node``, ``connection``, … — element of
                    # a relationship class.  Strip a trailing ``_N``
                    # disambiguator (``node_1`` / ``node_2`` → ``node``).
                    base = col.rsplit("_", 1)[0] if (
                        "_" in col and col.rsplit("_", 1)[1].isdigit()
                    ) else col
                    axis = self._axis_for_entity_class(base)
                if axis is None:
                    continue
                target = self._axis_enums.get(axis.name)
                if target is None or frame.schema[col] == target:
                    continue
                # Polars' numeric → Enum cast interprets the source value
                # as a POSITIONAL INDEX into the enum's categories.  When
                # the source column is numeric (e.g. a Spine 1d_map with
                # numeric keys whose level was labelled "constraint" /
                # other axis-synonym name), casting to the matching axis
                # enum would silently produce category-by-position
                # tokens.  Skip the cast — numeric columns are never dim
                # columns under this contract.
                if frame.schema[col].is_numeric():
                    continue
                element_casts.append(
                    pl.col(col).cast(target, strict=False)
                )
            if element_casts:
                frame = frame.with_columns(element_casts)
        origin = {
            "parameter": parameter_name,
            "entity": entity_class,
            "scenario": self._scenario,
        }
        return cast_against_contract(
            frame,
            contract=self._contract,
            axis_enums=self._axis_enums,
            origin=origin,
        )

    def _axis_for_entity_class(self, entity_class: str):
        """Return the contract axis whose source includes *entity_class*.

        Walks the contract's axes for ``source_type == "entity_class"``
        matching ``entity_class`` directly, or ``source_type ==
        "entity_class_union"`` whose list includes ``entity_class``.
        Returns the first matching :class:`AxisSpec`; ``None`` when no
        axis claims this class (synthetic classes, methods, etc.).

        When two axes claim the same class (e.g. ``node`` is sourced
        by both ``n`` directly AND by ``e`` (the entity union)), the
        single-class axis wins — its enum is narrower and more
        precise than the union.
        """
        if self._contract is None:
            return None
        single: object | None = None
        union: object | None = None
        for axis in self._contract.axes:
            if axis.source_type == "entity_class":
                if axis.source == entity_class:
                    single = axis
            elif axis.source_type == "entity_class_union":
                if entity_class in (axis.source or []):
                    if union is None:
                        union = axis
        return single if single is not None else union

    # ------------------------------------------------------------------
    # Unrolling the per-row blob into rectangular rows

    def _unroll_rows(
        self,
        rows: list[tuple[int, Any]],
        ent_cols: list[str],
        parameter_name: str,
    ) -> tuple[dict[str, list], list[str], pl.DataType | None]:
        """Walk each (entity_id, parsed_value) pair and emit a columnar
        dict of per-column lists ready for ``pl.DataFrame``.

        Columnar layout: ``columns[col_name]`` is a list, one entry per
        scalar leaf.  Every list has the same length.  Building columns
        directly is ~9x faster than the previous row-of-dicts pattern
        when handed to ``pl.DataFrame`` (per the
        ``arrow_value_direct_read_study.md`` benchmark).  The recursion
        uses a positional ``idx_path`` mutated in place via append /
        pop, avoiding ``dict(base)`` copies at every Map node.

        Returns the columns dict, the index column names (uniform
        across rows for the parameter), and the inferred leaf dtype.
        Empty input returns ``({}, [], None)``.

        Dim columns leave as Utf8 lists; the caller's
        ``_maybe_cast_frame`` (i.e.
        :func:`flextool.spinedb_backend._axis_enums.cast_against_contract`)
        applies the contract-axis Enum cast with its
        (parameter, entity, scenario) error breadcrumbs.  Don't wire
        enum dtypes here — keep the construction step orthogonal to
        the Phase 4 enum refactor for forward-compatibility with the
        future Arrow-native read.
        """
        if not rows:
            return {}, [], None

        # Discover index columns from the widest-shaped row (spec
        # §5.2.5).  In practice flextool's params don't mix shapes
        # within one scenario — but we're defensive.
        index_cols: list[str] = []
        for _eid, v in rows:
            cand = self._discover_index_cols(v, parameter_name)
            if len(cand) > len(index_cols):
                index_cols = cand

        # Pre-allocate one list per output column.
        col_names = ent_cols + index_cols + ["value"]
        columns: dict[str, list] = {name: [] for name in col_names}

        # Walk each (entity_id, parsed_value) pair.  ``idx_path`` is a
        # positional list mirroring ``index_cols`` — mutated via
        # append/pop inside the recursion, no per-node dict copy.
        idx_path: list[Any] = []
        for eid, v in rows:
            cls_id, ent_name = self._entity_by_id[eid]
            ent_values = self._entity_dim_values(cls_id, ent_name)
            self._unroll_value(
                v, index_cols, columns, ent_cols, ent_values, idx_path,
            )

        # Infer leaf dtype from the first non-None value.
        leaf_dtype: pl.DataType | None = None
        sample = next((x for x in columns["value"] if x is not None), None)
        if isinstance(sample, bool):
            leaf_dtype = pl.Boolean
        elif isinstance(sample, (int, float)):
            leaf_dtype = pl.Float64
        elif isinstance(sample, str):
            leaf_dtype = pl.Utf8

        return columns, index_cols, leaf_dtype

    def _entity_dim_values(self, class_id: int, ent_name: str) -> list[str]:
        """Return the dim-element values for *ent_name* in class
        *class_id*.  For 0-dim classes returns ``[ent_name]``; for
        n-relationships, returns the cached element list.
        """
        dims = self._class_dim_names.get(class_id)
        if dims is None:
            return [ent_name]
        for eid, name, elements in self._entities_by_class[class_id]:
            if name == ent_name:
                if elements is None:
                    return [ent_name]
                return elements
        return [ent_name]

    def _discover_index_cols(self, v: Any, parameter_name: str) -> list[str]:
        """Walk the value's nesting structure and return the index
        column names.  Scalar values return ``[]``.  Nested
        :class:`Map` / :class:`TimeSeries` / :class:`Array` return one
        name per level, falling back to canonical defaults from
        ``_DEFAULT_INDEX_NAMES`` when ``index_name`` is empty.
        """
        from spinedb_api.parameter_value import (
            Map, TimeSeries, Array,
        )

        cols: list[str] = []
        depth = 0
        cur = v
        while True:
            if isinstance(cur, Map):
                name = cur.index_name or _DEFAULT_INDEX_NAMES[
                    depth if depth < len(_DEFAULT_INDEX_NAMES) else -1
                ]
                # Map's "time" index is conventionally 't' in flextool.
                if name == "time":
                    name = "t"
                # Disambiguate when this level's index_name collides
                # with one already produced at an outer level — e.g.
                # ``examples.sqlite::invest_5weeks.invest_periods`` is
                # a 2D Map where BOTH levels carry ``index_name='x'``.
                # Without disambiguation the outer (anchor) and inner
                # (period) indexes both land in a single ``x`` column
                # and the outer is silently overwritten (deepest-wins
                # in _emit_leaf).  That loses the anchor → window
                # mapping the synthetic per-sub-solve cascade needs
                # (see ``_derived_params._solve_periods``).  Append a
                # depth suffix to keep both levels addressable.
                if name in cols:
                    name = f"{name}_{depth + 1}"
                cols.append(name)
                depth += 1
                # Probe the first child value to continue.
                if len(cur.values) == 0:
                    break
                cur = cur.values[0]
                continue
            if isinstance(cur, TimeSeries):
                cols.append("t")
                break
            if isinstance(cur, Array):
                # Use a non-axis name so cast_against_contract leaves
                # this column alone.  "i" collides with the canonical
                # tier_index axis ("i") whose vocabulary comes from
                # commodity.price_ladder_*; Array indices are
                # position-only and have no semantic axis.
                cols.append("_array_index")
                break
            break
        return cols

    def _unroll_value(self, v: Any, index_cols: list[str],
                       columns: dict[str, list],
                       ent_cols: list[str],
                       ent_values: list[str],
                       idx_path: list[Any]) -> None:
        """Recursively unroll *v* into per-column lists in ``columns``.

        ``idx_path`` is a positional list parallel to ``index_cols`` —
        mutated via append/pop as we descend.  Each scalar leaf
        triggers an ``_emit_leaf`` that appends entity values + the
        current ``idx_path`` (padded with ``None`` for any
        unreached trailing index columns) + the coerced leaf into the
        relevant column lists.
        """
        from spinedb_api.parameter_value import (
            Map, TimeSeries, Array,
        )

        # Scalar leaf.
        if not isinstance(v, (Map, TimeSeries, Array)):
            self._emit_leaf(
                columns, ent_cols, ent_values, idx_path, index_cols,
                _coerce_value(v),
            )
            return

        # Map / TimeSeries / Array all recurse identically once we know
        # which index column applies at the current depth.  Map's
        # ``index_name`` is already resolved into ``index_cols`` at
        # discovery time; TimeSeries / Array fall through to the
        # depth-indexed slot (or ``i`` if we've outrun the discovered
        # index_cols, e.g. a shape-mixed parameter).
        depth = len(idx_path)
        if isinstance(v, Map):
            for idx, child in zip(v.indexes, v.values):
                idx_path.append(_coerce_index(idx))
                self._unroll_value(
                    child, index_cols, columns, ent_cols, ent_values, idx_path,
                )
                idx_path.pop()
            return

        if isinstance(v, TimeSeries):
            for idx, val in zip(v.indexes, v.values):
                idx_path.append(_coerce_index(idx))
                self._emit_leaf(
                    columns, ent_cols, ent_values, idx_path, index_cols,
                    _coerce_value(val),
                )
                idx_path.pop()
            return

        if isinstance(v, Array):
            for i, val in enumerate(v.values):
                idx_path.append(i)
                self._emit_leaf(
                    columns, ent_cols, ent_values, idx_path, index_cols,
                    _coerce_value(val),
                )
                idx_path.pop()
            return

    def _emit_leaf(self, columns: dict[str, list],
                    ent_cols: list[str],
                    ent_values: list[str],
                    idx_path: list[Any],
                    index_cols: list[str],
                    value: Any) -> None:
        """Append one scalar leaf row across all per-column lists.

        Pads trailing ``index_cols`` (those beyond ``len(idx_path)``)
        with ``None`` so every column list ends up at the same length
        even when a parameter mixes shallow and deep shapes within a
        scenario.

        Duplicate names in ``index_cols`` (e.g. a nested Map where
        every level shares the same ``index_name='x'``) collapse to a
        single column via the polars dict-of-lists deduplication.  In
        that case we use a *last-wins* policy per leaf so the deepest
        Map level's index value survives — matching the pre-columnar
        ``pl.DataFrame(list_of_dicts)`` behaviour where each
        ``child_base[col_name] = ...`` overwrote the outer entry.
        """
        for col, val in zip(ent_cols, ent_values):
            columns[col].append(val)
        # Resolve duplicate column names by keeping the last index_path
        # value per name (deepest-wins; pad with None when idx_path is
        # shorter than index_cols).
        n_path = len(idx_path)
        per_col_value: dict[str, Any] = {}
        for i, col in enumerate(index_cols):
            if i < n_path:
                per_col_value[col] = idx_path[i]
            else:
                per_col_value.setdefault(col, None)
        for col, v in per_col_value.items():
            columns[col].append(v)
        columns["value"].append(value)

    # ------------------------------------------------------------------
    # Materialisation

    def _finalize(self, lf: pl.LazyFrame, sort_cols: list[str]) -> pl.DataFrame:
        """Sort by ``sort_cols`` (deterministic row order — §4.3) and
        collect.  Single materialisation point per :meth:`parameter`
        call so the polars optimiser can fuse the chain.
        """
        # Only sort by columns actually present (defensive).
        cols_present = lf.collect_schema().names()
        keep = [c for c in sort_cols if c in cols_present]
        if keep:
            lf = lf.sort(keep)
        return lf.collect()
