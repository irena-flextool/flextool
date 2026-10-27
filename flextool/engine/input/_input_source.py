"""Source abstractions for flextool's input data.

This module hosts **two** Protocols, used by separate phases of the
DB-direct migration:

* :class:`FlexInputSource` — the **CSV-shaped** source used for the
  fixture / pre-built-workdir path (:class:`CsvSource`).  Materialises
  flextool's ``input/`` + ``solve_data/`` CSV layout on disk;
  ``load_flextool`` walks them with ``polars.read_csv``.
* :class:`InputSource` — the **per-(entity_class, parameter_name)
  frame** Protocol introduced in Γ.1 of the deeper DB-direct migration
  (audit/db_direct_param_map.md §4.3).  Implementations
  (:class:`flextool._spinedb_reader.SpineDbReader`,
  :class:`flextool._inmemory_reader.InMemoryReader`) return individual
  parameter frames in their natural shape, scenario-resolved, with
  defaults applied per §4.5.  This is the abstraction Γ.1/Γ.2/Γ.3
  helpers compose against.

The two Protocols coexist: ``FlexInputSource`` keeps the existing
CSV-shaped loader for fixture workdirs, while ``InputSource`` is the
DB-direct abstraction used by the live cascade
(:func:`flextool.engine.run_chain_from_db`).

CSV-shaped source notes:

Today's downstream consumer (:func:`flextool.engine.preprocess.flex_data.load_flextool`) reads
CSVs via ``polars.read_csv`` directly off the directory tree, so the
Protocol exposes both:

* :pyattr:`FlexInputSource.input_dir` and
  :pyattr:`FlexInputSource.solve_data_dir` — Paths to the materialised
  CSV directories (the existing reader walks these as before).
* :meth:`FlexInputSource.get` — convenience accessor for callers that
  want a frame by ``(kind, name)`` without dealing with paths.

For :class:`CsvSource` the directories are just ``workdir/input`` and
``workdir/solve_data`` with no materialisation work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl


Kind = Literal["input", "solve_data"]


def _read_csv_file(path: "Path | str") -> pl.DataFrame:
    """The plain ``polars.read_csv`` behind :meth:`CsvSource.get`.

    CSV-retirement (Γ.8.F): the package-wide grep for ``pl.read_csv``
    returns only the ``CsvSource``-internal sites (``CsvSource.get`` plus
    this helper).  The loader never calls it: it reads the provider.
    """
    return pl.read_csv(path)


# ---------------------------------------------------------------------------
# Γ.1 — per-(entity_class, parameter_name) Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InputSource(Protocol):
    """Source-agnostic per-(entity_class, parameter_name) read API.

    Implementations are bound to a single scenario at construction;
    :meth:`entities` and :meth:`parameter` return scenario-resolved
    frames with defaults applied per §4.5 of the audit spec.

    Frames are deterministic in row order (sorted by entity dim columns
    first, then index columns) so per-Param parity assertions are
    stable across runs.
    """

    def entities(self, entity_class: str) -> pl.DataFrame:
        """Return the entity universe for *entity_class*.

        Schema:
            * 0-dim object class (e.g. ``"node"``): one column ``[name]``.
            * n-relationship class (e.g.
              ``"commodity__node"``, ``"connection__node__node"``):
              one column per dim, named after the dim's class.  Repeated
              dim classes are disambiguated by appending a 1-based
              suffix (e.g. ``connection__node__node`` →
              ``[connection, node_1, node_2]``).
        """

    def parameter(self,
                  entity_class: str,
                  parameter_name: str,
                  ) -> pl.DataFrame:
        """Return the parameter frame for ``(entity_class, parameter_name)``.

        Schema:
            * Entity dim columns from :meth:`entities`,
            * Followed by index columns implied by the parameter's
              value type (period / tier / t / branch / sub_index, in
              the parameter's natural index order),
            * Followed by a single ``value`` column (typed:
              ``pl.Float64`` for numerics, ``pl.Boolean`` /
              ``pl.Utf8`` for the occasional non-numeric).

        Default policy (§4.5):
            * ``parameter_definition.default_value is None`` → return
              only entities with explicit overrides; no fill-in rows.
            * Scalar default + scalar parameter → broadcast: one row
              per entity with the default for entities that have no
              override.
            * Scalar default + indexed parameter → return only entities
              with overrides; the default is exposed via
              :meth:`parameter_default` so helpers can ``fill_null``
              against their own index frames.
        """

    def parameter_default(self,
                           entity_class: str,
                           parameter_name: str,
                           ) -> Any:
        """Return the parameter's scalar default, or ``None``.

        Used by helpers to ``fill_null`` against their own index
        frames in the scalar-default-on-indexed case (§4.5).
        """

    def parameter_shape_info(self,
                              entity_class: str,
                              parameter_name: str,
                              ) -> "list[str | None]":
        """Return the raw per-level ``Map.index_name`` labels for the
        parameter (Δ.17c).

        Schema:

        * Empty list (``[]``) — scalar parameter (no Map nesting).
        * One entry per Map nesting level, in order from outermost to
          innermost.  Entries are the raw labels exactly as authored
          in the source database (``None`` when unset / empty).

        Used by :func:`flextool.engine.input._param_shapes.resolve_param_shape`
        to validate a parameter's actual shape against an explicit
        per-parameter allow-list.  See the Δ.17c dispatch / open-issues
        doc for the user advice that mandated this.

        Implementations that lack explicit DB metadata (e.g.
        :class:`InMemoryReader` in unit tests) infer labels from the
        parameter frame's column names — see the per-implementation
        docstring for details.
        """

    def parameter_shape_variants(self,
                                  entity_class: str,
                                  parameter_name: str,
                                  ) -> "set[tuple[str, ...]]":
        """Return the distinct per-row classified axis paths for a
        period/time registry parameter (spec ``fix_map_reading.md`` §3.2).

        Each element is a tuple of canonical axis labels
        (``("period",)`` / ``("time",)`` / ``("period", "time")`` /
        ``()`` for scalar).  Returns an empty set for parameters outside
        ``PERIOD_TIME_PARAMS`` (or when classification is not possible),
        in which case :func:`resolve_param_shape` falls back to checking
        the combined shape against the allow-list.

        This member is OPTIONAL: the resolver consumes it via ``getattr``
        so test-only stubs that implement only :meth:`parameter_shape_info`
        keep working.  It is declared here so the two production readers
        (:class:`SpineDbReader`, :class:`InMemoryReader`) satisfy the
        ``@runtime_checkable`` isinstance check used at the
        ``load_flextool`` boundary.
        """


@runtime_checkable
class FlexInputSource(Protocol):
    """Protocol every input-source must satisfy.

    Once :pyattr:`input_dir` / :pyattr:`solve_data_dir` return, the
    directories must be populated and ready for the existing CSV
    reader to walk.
    """

    @property
    def input_dir(self) -> Path: ...
    @property
    def solve_data_dir(self) -> Path: ...

    def get(self, kind: Kind, name: str) -> pl.DataFrame | None:
        """Return the named frame from ``input/`` or ``solve_data/``.

        ``name`` may be given with or without the ``.csv`` suffix.
        Returns ``None`` when the file is absent.  Empty (header-only)
        files yield an empty DataFrame, consistent with
        ``polars.read_csv`` behaviour.
        """
        ...


class CsvSource:
    """Wraps a flextool workdir on disk (the pre-DB-migration layout).

    Construction is trivial and read-only; both directories must exist
    or be missing in the same way they would be when calling
    ``load_flextool(workdir)`` directly.
    """

    def __init__(self, workdir: Path | str):
        self._workdir = Path(workdir)

    @property
    def workdir(self) -> Path:
        return self._workdir

    @property
    def input_dir(self) -> Path:
        return self._workdir / "input"

    @property
    def solve_data_dir(self) -> Path:
        return self._workdir / "solve_data"

    def get(self, kind: Kind, name: str) -> pl.DataFrame | None:
        d = self.input_dir if kind == "input" else self.solve_data_dir
        fname = name if name.endswith(".csv") else f"{name}.csv"
        path = d / fname
        if not path.exists():
            return None
        return _read_csv_file(path)

    def __repr__(self) -> str:
        return f"CsvSource(workdir={self._workdir!s})"
