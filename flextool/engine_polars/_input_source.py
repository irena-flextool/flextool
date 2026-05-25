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
(:func:`flextool.engine_polars.run_chain_from_db`).

CSV-shaped source notes:

Today's downstream consumer (:func:`flextool.input.load_flextool`) reads
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

import logging
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl
import polars.exceptions as pl_exc


_LOGGER = logging.getLogger(__name__)


Kind = Literal["input", "solve_data"]


_active_cache: dict[Path, pl.DataFrame] | None = None


def _read_csv_file(path: "Path | str") -> pl.DataFrame:
    """Single residual ``polars.read_csv`` site for the engine_polars
    package.

    CSV-retirement (Γ.8.F) gates every workdir CSV read in the loader
    path through this helper so the package-wide grep for
    ``pl.read_csv`` returns only the ``CsvSource``-internal sites
    (``CsvSource.get`` plus this helper).

    Δ.12a — when a per-solve cache is active (set via
    :func:`_install_csv_cache` from the ``SolveContext`` constructor),
    repeated reads of the same absolute path hit memory.
    """
    if _active_cache is not None:
        # Use the str form as cache key — avoids the per-call ``Path.resolve``
        # syscall (which adds ~50µs each and dominates the cache miss path
        # for small fixtures with few duplicate reads).  Different string
        # forms of the same file (e.g. ``./x.csv`` vs ``x.csv``) miss the
        # cache but the loader path always constructs paths from the same
        # workdir prefix so collisions are negligible in practice.
        key = str(path)
        cached = _active_cache.get(key)
        if cached is not None:
            return cached
        df = pl.read_csv(path)
        _active_cache[key] = df
        return df
    return pl.read_csv(path)


def read_csv_fallback(path: "Path | str") -> pl.DataFrame:
    """Off-cascade disk read of a single CSV.

    Reserved for callers in :pyfile:`flextool/engine_polars/input.py`
    that still serve workdir-only loader-unit tests.  Cascade code MUST
    go through :class:`FlexDataProvider`; this is the single sanctioned
    entry point for the residual disk-fallback path so the Rule 1
    invariant scan can confirm input.py never calls ``_read_csv_file``
    or ``pl.read_csv`` directly.
    """
    return _read_csv_file(path)


def seed_provider_from_dir(
    provider,
    directory: "Path | str",
    kind: str,
    *,
    names: "tuple[str, ...] | None" = None,
) -> int:
    """Off-cascade test/bridge helper: populate *provider* by reading
    CSV files under *directory* and keying them under both
    ``"<stem>"`` and ``"{kind}/<stem>"``.

    Mirrors the dual-key convention used by :func:`capture_frames`.
    Returns the count of files seeded.  Missing directories are a
    no-op (return 0).  Callers in cascade code must NOT reach for
    this helper: it exists for test fixtures, region-decomposition
    seeding, and the off-cascade workdir bridge.

    Parameters
    ----------
    provider
        :class:`FlexDataProvider` to populate.
    directory
        Source directory.
    kind
        Prefix for the parent-qualified key (``"input"`` or
        ``"solve_data"``).
    names
        Optional explicit allow-list of stems (without ``.csv``) to
        consume.  When ``None`` every ``*.csv`` is read.  Use the
        explicit form to skip non-canonical files in directories that
        also carry ragged or human-readable artefacts (e.g.
        ``solve_progress.csv``).
    """
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return 0
    if names is not None:
        targets = [d / f"{n}.csv" for n in names if (d / f"{n}.csv").exists()]
    else:
        targets = sorted(d.glob("*.csv"))
    seeded = 0
    for p in targets:
        try:
            df = _read_csv_file(p)
            provider.put(f"{kind}/{p.stem}", df)
        except (pl_exc.ComputeError, pl_exc.NoDataError) as exc:
            _LOGGER.warning(
                "seed_provider_from_dir: skipping malformed CSV %s "
                "(%s: %s)",
                p,
                type(exc).__name__,
                exc,
            )
            continue
        except Exception as exc:  # noqa: BLE001 — log + continue for stray files
            _LOGGER.warning(
                "seed_provider_from_dir: skipping unreadable CSV %s "
                "(%s: %s)",
                p,
                type(exc).__name__,
                exc,
            )
            continue
        seeded += 1
    return seeded


def _install_csv_cache(cache: "dict[Path, pl.DataFrame] | None") -> None:
    """Δ.12a — install / clear the process-level CSV-read cache.

    Called by ``SolveContext.__enter__`` / ``__exit__`` (or the
    explicit ``activate_cache`` / ``deactivate_cache`` helpers) to
    install the per-solve cache so :func:`_read_csv_file` calls in any
    helper hit memory on repeats.

    Pass ``None`` to disable caching (default).  Multiple
    activate/deactivate cycles within a single process are supported;
    nesting is the caller's responsibility (typically via the SolveContext
    context-manager boundary which is one-deep per solve).
    """
    global _active_cache
    _active_cache = cache


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

        Used by :func:`flextool.engine_polars._param_shapes.resolve_param_shape`
        to validate a parameter's actual shape against an explicit
        per-parameter allow-list.  See the Δ.17c dispatch / open-issues
        doc for the user advice that mandated this.

        Implementations that lack explicit DB metadata (e.g.
        :class:`InMemoryReader` in unit tests) infer labels from the
        parameter frame's column names — see the per-implementation
        docstring for details.
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
