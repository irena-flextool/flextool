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

    Lookups raise :class:`KeyError` on unknown classes / parameters.
    """

    def __init__(
        self,
        entities: Mapping[str, pl.DataFrame],
        parameters: Mapping[tuple[str, str], pl.DataFrame],
        defaults: Mapping[tuple[str, str], Any] | None = None,
    ):
        # Defensive copy: the caller may mutate their inputs after
        # constructing us.  Polars frames are cheap to wrap.
        self._entities: dict[str, pl.DataFrame] = dict(entities)
        self._parameters: dict[tuple[str, str], pl.DataFrame] = dict(parameters)
        self._defaults: dict[tuple[str, str], Any] = (
            dict(defaults) if defaults is not None else {}
        )

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
