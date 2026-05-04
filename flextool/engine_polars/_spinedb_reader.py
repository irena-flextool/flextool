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

    def __init__(self, db_url: str, scenario: str):
        url = str(db_url)
        if not url.startswith("sqlite:") and not url.startswith("postgresql"):
            url = f"sqlite:///{url}"
        self._db_url = url
        self._scenario = scenario

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
            return (
                pl.DataFrame({cols[0]: [name for _, name, _ in rows]},
                             schema={cols[0]: pl.Utf8})
                .sort(cols)
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
        return pl.DataFrame(data, schema=schema).sort(cols)

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
        out_rows, index_cols, leaf_dtype = self._unroll_rows(
            rows, ent_cols, parameter_name,
        )
        if not out_rows:
            schema_in = {c: pl.Utf8 for c in ent_cols}
            for ic in index_cols:
                schema_in[ic] = pl.Utf8
            schema_in["value"] = leaf_dtype or pl.Float64
            return pl.DataFrame(schema=schema_in)
        return self._finalize(pl.DataFrame(out_rows).lazy(),
                               ent_cols + index_cols)

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

        # Step 1 — unroll each row into a list of dict rows.  The shape
        # depends on the value's runtime type (scalar / Map / TimeSeries
        # / Array).  We don't know up-front whether the parameter is
        # "scalar across all rows"; we infer per-row.
        out_rows, index_cols, leaf_dtype = self._unroll_rows(
            rows, ent_cols, parameter_name,
        )

        # Step 2 — assemble the per-parameter LazyFrame.  Empty rows
        # collapse to a 0-row frame with the correct schema.
        if not out_rows:
            schema_in = {c: pl.Utf8 for c in ent_cols}
            for ic in index_cols:
                schema_in[ic] = pl.Utf8
            schema_in["value"] = leaf_dtype or pl.Float64
            v_lf = pl.DataFrame(schema=schema_in).lazy()
        else:
            v_lf = pl.DataFrame(out_rows).lazy()

        # Step 3 — apply the §4.5 default policy lazily.
        if default is None:
            # None-skip: return rows as-is.
            return self._finalize(v_lf, ent_cols + index_cols)

        if not index_cols:
            # Scalar-default + scalar-parameter → broadcast.
            E = self.entities(entity_class).lazy()
            joined = (E.join(v_lf, on=ent_cols, how="left")
                       .with_columns(pl.col("value").fill_null(default)))
            return self._finalize(joined, ent_cols)

        # Scalar-default + indexed-parameter → return overrides only;
        # the default is consumed via parameter_default() upstream.
        return self._finalize(v_lf, ent_cols + index_cols)

    # ------------------------------------------------------------------
    # Unrolling the per-row blob into rectangular rows

    def _unroll_rows(
        self,
        rows: list[tuple[int, Any]],
        ent_cols: list[str],
        parameter_name: str,
    ) -> tuple[list[dict], list[str], pl.DataType | None]:
        """Walk each (entity_id, parsed_value) pair and emit a list of
        dict rows ready for ``pl.DataFrame``.

        Returns the rows, the index column names (uniform across rows
        for the parameter), and the inferred leaf dtype.  Empty input
        returns ``([], [], None)``.
        """
        if not rows:
            return [], [], None

        from spinedb_api.parameter_value import (
            Map, TimeSeries, Array, DateTime, Duration,
        )

        # Probe the first non-None value to decide structure.  We
        # require all rows to share the same structure (Spine schema
        # invariant under a scenario).
        index_cols: list[str] = []
        leaf_dtype: pl.DataType | None = None

        # Discover index columns from the first non-trivial row.  This
        # is the spec's "pick the widest schema" rule applied per
        # parameter.
        for _eid, v in rows:
            cand = self._discover_index_cols(v, parameter_name)
            if len(cand) > len(index_cols):
                index_cols = cand
        # Pad narrower rows by replicating their leaf at the last index
        # level (spec §5.2.5: 1d_map vs 2d_map for the same param name).
        # In practice flextool's params don't mix shapes within one
        # scenario — but we're defensive.

        out_rows: list[dict] = []
        for eid, v in rows:
            cls_id, ent_name = self._entity_by_id[eid]
            ent_values = self._entity_dim_values(cls_id, ent_name)
            base = dict(zip(ent_cols, ent_values))
            self._unroll_value(v, index_cols, base, out_rows)

        # Infer leaf dtype from a sample of out_rows' value column.
        if out_rows:
            sample = next((r["value"] for r in out_rows if r["value"] is not None),
                          None)
            if isinstance(sample, bool):
                leaf_dtype = pl.Boolean
            elif isinstance(sample, (int, float)):
                leaf_dtype = pl.Float64
            elif isinstance(sample, str):
                leaf_dtype = pl.Utf8
            else:
                leaf_dtype = None

        return out_rows, index_cols, leaf_dtype

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
                cols.append("i")
                break
            break
        return cols

    def _unroll_value(self, v: Any, index_cols: list[str],
                       base: dict, out: list[dict]) -> None:
        """Recursively unroll *v* into ``out`` using ``index_cols`` as
        the per-level column names.  ``base`` holds the entity dim
        values + already-decided index columns from outer levels.
        Scalars become a single row.
        """
        from spinedb_api.parameter_value import (
            Map, TimeSeries, Array,
        )

        # Scalar leaf.
        if not isinstance(v, (Map, TimeSeries, Array)):
            row = dict(base)
            row["value"] = _coerce_value(v)
            out.append(row)
            return

        # Map: recurse on each (index, value) pair.
        if isinstance(v, Map):
            depth = sum(1 for c in index_cols if c in base)
            col_name = index_cols[depth] if depth < len(index_cols) else "i"
            for idx, child in zip(v.indexes, v.values):
                child_base = dict(base)
                child_base[col_name] = _coerce_index(idx)
                self._unroll_value(child, index_cols, child_base, out)
            return

        # TimeSeries: emit one row per (datetime, float).
        if isinstance(v, TimeSeries):
            depth = sum(1 for c in index_cols if c in base)
            col_name = index_cols[depth] if depth < len(index_cols) else "t"
            for idx, val in zip(v.indexes, v.values):
                row = dict(base)
                row[col_name] = _coerce_index(idx)
                row["value"] = _coerce_value(val)
                out.append(row)
            return

        # Array: 1-d sequence, indexed by integer position.
        if isinstance(v, Array):
            depth = sum(1 for c in index_cols if c in base)
            col_name = index_cols[depth] if depth < len(index_cols) else "i"
            for i, val in enumerate(v.values):
                row = dict(base)
                row[col_name] = i
                row["value"] = _coerce_value(val)
                out.append(row)
            return

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
