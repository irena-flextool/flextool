"""In-memory implementation of the :class:`InputSource` Protocol.

Used by unit tests for processing-layer Param helpers.  The caller
supplies hand-crafted entity / parameter frames already in the
post-resolution shape — no SpineDB, no scenario filtering, no default
fill is applied by the reader (the caller is responsible for shaping
the data exactly as a real source would have produced it).

This is the migration-velocity unlock for Γ.1/Γ.2/Γ.3: every Direct /
Projection / Derived helper takes an :class:`InputSource`, so test
coverage no longer requires standing up sqlite or generating fixtures.
See ``audit/db_direct_param_map.md §4.4`` and ``§8.3``.
"""
from __future__ import annotations

from typing import Any, Mapping

import polars as pl


class InMemoryReader:
    """Trivial dict-backed :class:`InputSource`.

    Parameters
    ----------
    entities : Mapping[str, pl.DataFrame]
        ``{entity_class_name: frame}``.  Frame schema follows
        :meth:`InputSource.entities`: one ``[name]`` column for 0-dim
        classes; one column per dim (named after the dim class) for
        n-relationship classes.
    parameters : Mapping[tuple[str, str], pl.DataFrame]
        ``{(entity_class, parameter_name): frame}``.  Frame schema
        follows :meth:`InputSource.parameter`.
    defaults : Mapping[tuple[str, str], Any] | None
        Optional ``{(entity_class, parameter_name): default_value}``.
        Absent keys imply ``None`` default (§4.5 None-skip branch).
    startup_arcs : pl.DataFrame | None
        Optional ``(unit, node, method)`` frame returned by :meth:`startup_arcs`.  Without it the arcs come
        from the ``("unit__inputNode", "startup_input_method")`` frame, if given.  No arc is hidden: the
        caller leaves the ``startup_only`` arcs out of the entity and parameter frames, as the database
        readers would.

    Lookups raise :class:`KeyError` on unknown classes / parameters.
    """

    def __init__(
        self,
        entities: Mapping[str, pl.DataFrame],
        parameters: Mapping[tuple[str, str], pl.DataFrame],
        defaults: Mapping[tuple[str, str], Any] | None = None,
        startup_arcs: pl.DataFrame | None = None,
    ):
        # Defensive copy: the caller may mutate their inputs after
        # constructing us.  Polars frames are cheap to wrap.
        self._entities: dict[str, pl.DataFrame] = dict(entities)
        self._parameters: dict[tuple[str, str], pl.DataFrame] = dict(parameters)
        self._defaults: dict[tuple[str, str], Any] = (
            dict(defaults) if defaults is not None else {}
        )
        self._startup_arcs = startup_arcs

    # ------------------------------------------------------------------
    # InputSource Protocol

    def entities(self, entity_class: str) -> pl.DataFrame:
        try:
            return self._entities[entity_class]
        except KeyError:
            raise KeyError(
                f"InMemoryReader: unknown entity_class {entity_class!r}"
            ) from None

    def parameter(self, entity_class: str, parameter_name: str) -> pl.DataFrame:
        key = (entity_class, parameter_name)
        try:
            return self._parameters[key]
        except KeyError:
            raise KeyError(
                f"InMemoryReader: unknown parameter "
                f"({entity_class!r}, {parameter_name!r})"
            ) from None

    def parameter_default(self, entity_class: str, parameter_name: str) -> Any:
        return self._defaults.get((entity_class, parameter_name))

    def startup_arcs(self) -> pl.DataFrame:
        """``(unit, node, method)`` (Utf8) of the arcs with ``startup_input_method`` ``startup_and_input`` or
        ``startup_only``: the frame given to the constructor, else the rows of the
        ``("unit__inputNode", "startup_input_method")`` frame with one of those values."""
        schema = {"unit": pl.Utf8, "node": pl.Utf8, "method": pl.Utf8}
        if self._startup_arcs is not None:
            return self._startup_arcs.select(*(pl.col(c).cast(pl.Utf8) for c in schema))
        frame = self._parameters.get(("unit__inputNode", "startup_input_method"))
        if frame is None or frame.height == 0:
            return pl.DataFrame(schema=schema)
        return (frame.filter(pl.col("value").is_in(["startup_and_input", "startup_only"]))
                .select(pl.col("unit").cast(pl.Utf8), pl.col("node").cast(pl.Utf8),
                        pl.col("value").cast(pl.Utf8).alias("method"))
                .sort("unit", "node"))

    def parameter_explicit(self, entity_class: str,
                            parameter_name: str) -> pl.DataFrame:
        """Mirror of :meth:`SpineDbReader.parameter_explicit`.

        InMemoryReader holds frames the caller passed in directly — the
        Protocol treats those frames as already containing only
        explicit values (no default broadcast).  So this is identical
        to :meth:`parameter` for the in-memory case.
        """
        return self.parameter(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str,
                              parameter_name: str) -> "list[str | None]":
        """Δ.17c — raw per-level ``index_name`` labels for the
        parameter.

        InMemoryReader callers author frames already in the post-
        resolution shape (column names like ``period`` / ``t`` / etc.)
        — no DB-level metadata is held.  We infer the labels from the
        frame's column names: any column named ``period`` →
        ``"period"``; any column named ``t`` / ``time`` → ``"time"``;
        anything else is propagated as-is so the resolver can flag it.

        The frame's entity-dim columns are excluded by walking the
        registered entities frame for the same class.
        """
        df = self.parameter(entity_class, parameter_name)
        try:
            ent_df = self.entities(entity_class)
            ent_cols = set(ent_df.columns)
        except KeyError:
            ent_cols = {"name"}
        out: list[str | None] = []
        for c in df.columns:
            if c in ent_cols or c == "value":
                continue
            if c == "period":
                out.append("period")
            elif c in ("t", "time"):
                out.append("time")
            else:
                # Unknown column → caller raises; pass through raw.
                out.append(c)
        return out

    def parameter_index_label_paths(self, entity_class: str,
                                    parameter_name: str) -> "set[tuple[str | None, ...]]":
        """Mirror of :meth:`SpineDbReader.parameter_index_label_paths`: the
        frame's index columns (non-entity, non-value) as one label path."""
        if self.parameter(entity_class, parameter_name).height == 0:
            return set()
        return {tuple(self.parameter_shape_info(entity_class, parameter_name))}

    def _entity_columns(self, entity_class: str) -> list[str]:
        try:
            cols = list(self.entities(entity_class).columns)
        except KeyError:
            cols = []
        return cols or ["name"]

    def parameter_entity_label_paths(self, entity_class: str, parameter_name: str,
                                     ) -> "dict[tuple[str, ...], tuple[str | None, ...]]":
        """Mirror of :meth:`SpineDbReader.parameter_entity_label_paths`: per entity, the names of the index
        columns its rows populate (non-null), in column order; the deepest row's when they differ."""
        df = self.parameter(entity_class, parameter_name)
        ent_cols = [c for c in self._entity_columns(entity_class) if c in df.columns]
        idx_cols = [c for c in df.columns if c not in ent_cols and c != "value"]
        out: dict[tuple[str, ...], tuple[str | None, ...]] = {}
        for row in df.select(ent_cols + idx_cols).iter_rows(named=True):
            ent = tuple(str(row[c]) for c in ent_cols)
            path = tuple(c for c in idx_cols if row[c] is not None)
            if len(path) >= len(out.get(ent, ())):
                out[ent] = path
        return out

    def without_entities(self, entity_class: str, parameter_name: str,
                         entities) -> "InMemoryReader":
        """Mirror of :meth:`SpineDbReader.without_entities`: a copy whose ``(entity_class, parameter_name)``
        frame lacks the rows of ``entities`` (class-dim value tuples).  Index columns that only those rows
        populated are dropped, as a database without those values would not have them."""
        df = self.parameter(entity_class, parameter_name)
        ent_cols = [c for c in self._entity_columns(entity_class) if c in df.columns]
        drop = pl.DataFrame([list(e) for e in sorted(entities)], schema=ent_cols, orient="row")
        kept = (df.with_row_index("__row")
                .join(df.select(pl.col(c).cast(pl.Utf8) for c in ent_cols).with_row_index("__row")
                      .join(drop, on=ent_cols, how="semi").select("__row"), on="__row", how="anti")
                .drop("__row"))
        empty = [c for c in kept.columns
                 if c not in ent_cols and c != "value" and kept.get_column(c).null_count() == kept.height]
        params = dict(self._parameters)
        params[(entity_class, parameter_name)] = kept.drop(empty)
        return InMemoryReader(self._entities, params, self._defaults, startup_arcs=self._startup_arcs)

    def parameter_shape_variants(self, entity_class: str,
                                  parameter_name: str,
                                  ) -> "set[tuple[str, ...]]":
        """Spec fix_map_reading.md §3.2 — distinct per-row classified axis
        paths for a PERIOD_TIME_PARAMS parameter, inferred from the
        frame's null patterns.

        InMemoryReader callers author frames already in post-resolution
        shape (columns ``period`` / ``t`` / ``time``).  Each row's variant
        is the tuple of canonical axes whose column is non-null, in
        canonical order (period before time).  Returns an empty set for
        non-PERIOD_TIME keys so the resolver falls back to its combined
        allow-list check.
        """
        from flextool.engine.input._param_shapes import PERIOD_TIME_PARAMS
        if (entity_class, parameter_name) not in PERIOD_TIME_PARAMS:
            return set()
        df = self.parameter(entity_class, parameter_name)
        has_period = "period" in df.columns
        tcol = "t" if "t" in df.columns else (
            "time" if "time" in df.columns else None
        )
        if not has_period and tcol is None:
            # Scalar-only frame (or no period/time columns).
            return {()} if df.height > 0 else set()
        sel_cols = ([c for c in ("period",) if has_period]
                    + ([tcol] if tcol is not None else []))
        variants: set[tuple[str, ...]] = set()
        for row in df.select(sel_cols).iter_rows(named=True):
            path: list[str] = []
            if has_period and row.get("period") is not None:
                path.append("period")
            if tcol is not None and row.get(tcol) is not None:
                path.append("time")
            variants.add(tuple(path))
        return variants

    # ------------------------------------------------------------------
    # Diagnostics

    def __repr__(self) -> str:
        return (
            f"InMemoryReader(classes={len(self._entities)}, "
            f"params={len(self._parameters)}, "
            f"defaults={len(self._defaults)})"
        )

    @property
    def known_classes(self) -> list[str]:
        return sorted(self._entities)

    @property
    def known_parameters(self) -> list[tuple[str, str]]:
        return sorted(self._parameters)
