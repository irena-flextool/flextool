"""Per-solve in-memory state — the typed replacement for ``solve_data/``.

Δ.12a — port per-solve preprocessing metadata into engine_polars.

This module hosts :class:`SolveContext`, the typed dataclass the
:doc:`audit/native_data_path_design_solve_context` schematic specifies.
It wraps the per-solve preprocessing artefacts that today live as CSVs
under ``workdir/solve_data/`` and exposes them as in-memory polars
frames so the override helpers (``apply_derived_a..g``) and the
``_load_*`` family in :mod:`flextool.engine_polars.input` no longer have
to re-read the same files dozens of times per solve.

:class:`SolveContext` reads the ``solve_data/*.csv`` files written by
the per-solve preprocessing.  The point of going through this object
is:

1. **Single funnel** — every workdir CSV read goes through one place,
   so the gross :func:`flextool.engine_polars._input_source._read_csv_file`
   call count drops dramatically (helpers consume cached frames instead
   of re-issuing ``pl.read_csv`` every call).
2. **Typed entry points** — the most-consumed metadata
   (``solveFirst``, ``realized_periods``, ``period_in_use``,
   ``period_branch``, ``edd_history``, etc.) is parsed once into typed
   Python state and exposed as named attributes.  Helpers that today
   re-derive these from raw CSVs each call (e.g.
   ``_read_active_solve``, ``_read_realize_invest_periods``) consume the
   typed fields.
3. **Future cutover seam** — when Δ.12c lands, the typed-fields
   constructor swaps to populate from
   :class:`~flextool.engine_polars._solve_state.RunnerState` directly
   (``state.solve.realized_periods[solve_name]``,
   ``state.timeline.dt_for_solve(solve_name)``).  Helpers don't change.

See :doc:`audit/native_data_path_design_solve_context` for the design
rationale and :doc:`audit/handoff_csv_retirement.md` for the broader
CSV-retirement plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ._axis_enums import schema_dtype
from ._input_source import _install_csv_cache, _read_csv_file


# ---------------------------------------------------------------------------
# SolveContext
# ---------------------------------------------------------------------------


@dataclass
class SolveContext:
    """Typed in-memory carrier of per-solve preprocessing state.

    Construct via :meth:`from_workdir` to populate the typed fields from
    a flextool workdir (the CSV-bridge path used by Δ.12a) or via the
    plain constructor to hand-build for tests.

    All ``DataFrame`` fields are eager (``pl.DataFrame``) — helpers can
    convert via ``df.lazy()`` at the call site to keep the rest of the
    pipeline lazy per the project invariant.

    Attributes
    ----------
    workdir : Path
        The flextool workdir.  Kept around so :meth:`read_csv` can
        service helpers that still need a file the typed fields don't
        cover (e.g. flexible-shape "user-defined" CSVs that genuinely
        have no Spine analogue).  Each such read is cached so repeated
        calls hit memory.
    solve_name : str | None
        Active solve as written to ``solve_data/solve_current.csv``.
        ``None`` for fixtures without per-solve preprocessing.
    solveFirst : bool
        ``True`` iff this is the first sub-solve in a multi-solve cascade
        (per ``solve_data/p_model.csv:solveFirst``).  Default ``True``
        (single-solve fixtures behave as first-of-chain).
    realized_periods : set[str]
        Periods realized this solve (``realized_dispatch.csv`` distinct
        ``period`` column).  Empty when the file is missing.
    realized_invest_periods : set[str]
        Periods where invest was realized this solve
        (``realized_invest_periods_of_current_solve.csv``).
    period_in_use : pl.DataFrame
        ``[d]`` distinct frame from ``period_in_use_set.csv``.  This is
        the authoritative active-period set INCLUDING any stochastic-
        branch periods (per ``per_solve_sets.py:95-101``).  Empty frame
        when the file is missing.
    period_branch : pl.DataFrame
        ``[d_anchor, b]`` from ``period__branch.csv`` — full unfiltered
        anchor → sibling map.  Empty frame when the file is missing.
    edd_history : pl.DataFrame
        ``[entity, d_decided, d_apply]`` from ``edd_history.csv`` — the
        invest-decision-date → apply-date map used by handoff-derived
        ``p_entity_previously_invested_capacity``.  Empty frame when the
        file is missing.
    p_entity_pre_existing : pl.DataFrame
        ``[entity, period, value]`` from ``p_entity_pre_existing.csv``.
        Empty when missing.

    The remaining workdir CSVs (block-layout, dtttdt, etc.) are still
    accessed via :meth:`read_csv` — they're either covered by other
    typed objects (:class:`BlockLayout`) or read few enough times that
    string-keyed access is fine.
    """

    workdir: Path
    solve_name: str | None = None
    solveFirst: bool = True
    realized_periods: set[str] = field(default_factory=set)
    realized_invest_periods: set[str] = field(default_factory=set)
    # Phase 3 — Provider stashed at construction so the lazy DataFrame
    # property loaders can consult it without re-plumbing the kwarg
    # through every caller.  When ``provider`` is None the legacy disk
    # path remains active (test-only fallback that goes away with
    # Phase 4 activation).
    provider: "object | None" = field(default=None, repr=False)
    # ``period_in_use`` / ``period_branch`` / ``edd_history`` /
    # ``p_entity_pre_existing`` are populated on first access via the
    # descriptor machinery below — they're DataFrame-shaped fields and
    # each requires a CSV read.  Most
    # parity tests only consume one or two of them, so deferring the
    # reads until an attribute access asks for them avoids paying the
    # IO cost upfront.
    _period_in_use_loaded: bool = field(default=False, repr=False)
    _period_in_use: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"d": schema_dtype(None, "d")}
        ),
        repr=False,
    )
    _period_branch_loaded: bool = field(default=False, repr=False)
    _period_branch: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "d_anchor": schema_dtype(None, "d_anchor"),
                "b": schema_dtype(None, "b"),
            }
        ),
        repr=False,
    )
    _edd_history_loaded: bool = field(default=False, repr=False)
    _edd_history: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={}), repr=False,
    )
    _ppe_loaded: bool = field(default=False, repr=False)
    _ppe: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={}), repr=False,
    )

    # ── Path B Cat B (WriterSnapshot top-7) extensions ──────────────────
    # Per-solve preprocessing artefacts written by ``_emit_solve_writers``
    # / ``_emit_period_calc`` and consumed by the cascade (audit
    # `specs/in_cascade_csv_audit.md` §Category B).  Lazy-loaded on first
    # attribute access — same pattern as ``period_in_use`` /
    # ``period_branch`` above.  Schemas mirror the renamed canonical form
    # the cascade helpers themselves construct, so callers can swap a
    # ``_read_csv_file + rename + select + unique`` chain for a single
    # attribute access.
    _steps_in_use_loaded: bool = field(default=False, repr=False)
    _steps_in_use: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "d": schema_dtype(None, "d"),
                "t": schema_dtype(None, "t"),
                "step_duration": pl.Float64,
            }
        ),
        repr=False,
    )
    _period_share_loaded: bool = field(default=False, repr=False)
    _period_share: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"d": schema_dtype(None, "d"), "value": pl.Float64}
        ),
        repr=False,
    )
    _p_entity_all_existing_loaded: bool = field(default=False, repr=False)
    _p_entity_all_existing: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={}), repr=False,
    )
    _solve_branch_weight_loaded: bool = field(default=False, repr=False)
    _solve_branch_weight: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={
                "b": schema_dtype(None, "b"),
                "p_branch_weight_input": pl.Float64,
            }
        ),
        repr=False,
    )

    # Internal: ad-hoc CSV cache.  Keyed by ``str(path)`` (no syscall).
    # Values are ``None`` when the file is absent so we don't re-stat
    # repeatedly.
    _csv_cache: dict[str, pl.DataFrame | None] = field(
        default_factory=dict, repr=False
    )
    # Internal: process-level cache view installed by ``activate`` —
    # a strict subset of ``_csv_cache`` (None-valued entries excluded
    # because the ``_read_csv_file`` cache only stores successfully-
    # read frames).  ``None`` means caching is not currently active.
    _active_cache: "dict[str, pl.DataFrame] | None" = field(
        default=None, repr=False
    )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_workdir(cls, workdir: Path | str,
                      *, provider: "object | None" = None) -> "SolveContext":
        """Construct a SolveContext from a flextool workdir.

        Eagerly reads only the cheap typed scalars
        (``solve_name``, ``solveFirst``, ``realized_periods``,
        ``realized_invest_periods``) — these come from small single-
        column CSVs and are consumed by the apply_derived_* boundary
        checks that gate the rest of the cascade.

        The DataFrame-shaped fields (``period_in_use``, ``period_branch``,
        ``edd_history``, ``p_entity_pre_existing``) are loaded **lazily**
        on first attribute access.  Most parity tests only consume one or
        two,
        so paying the IO cost upfront for a bag of frames the test
        never touches dominates the cache savings on small fixtures.

        Step 1-g-4b — *provider* threads the live
        :class:`FlexDataProvider` through to the four eager scalar
        readers so they hit the in-memory frames before falling back
        to the seed funnel / disk.
        """
        wd = Path(workdir)
        sd = wd / "solve_data"
        ctx = cls(workdir=wd, provider=provider)
        ctx.solve_name = _read_active_solve(wd, provider=provider)
        ctx.solveFirst = _read_solve_first(wd, provider=provider)
        ctx.realized_periods = _read_realized_dispatch_periods(
            sd / "realized_dispatch.csv", provider=provider,
        )
        ctx.realized_invest_periods = _read_period_set(
            sd / "realized_invest_periods_of_current_solve.csv",
            provider=provider,
        )
        return ctx

    # ------------------------------------------------------------------
    # Cached CSV reader
    # ------------------------------------------------------------------

    def read_csv(
        self,
        relative: str | Path,
        *,
        kind: str = "solve_data",
    ) -> pl.DataFrame | None:
        """Cached read for an ad-hoc workdir CSV.

        Parameters
        ----------
        relative : str | Path
            Filename relative to the directory selected by *kind*.  May
            be a bare ``foo.csv`` or a relative subpath.
        kind : str
            ``"solve_data"`` (default) or ``"input"`` — selects the
            subdirectory under :attr:`workdir`.  Anything else is
            interpreted as a workdir-rooted relative path.

        Returns
        -------
        pl.DataFrame | None
            The frame, or ``None`` when the file is missing.  The result
            is cached by absolute path so subsequent calls hit memory.
        """
        rel = Path(relative)
        if kind == "solve_data":
            path = self.workdir / "solve_data" / rel
        elif kind == "input":
            path = self.workdir / "input" / rel
        else:
            path = self.workdir / rel
        # Cache key = str(path).  Same key as ``_read_csv_file``'s active-
        # cache so SolveContext.read_csv and direct ``_read_csv_file``
        # calls share the cache when ``activate`` is in effect.  Avoids
        # the per-call ``Path.resolve`` syscall.
        key = str(path)
        if key in self._csv_cache:
            return self._csv_cache[key]
        if not path.exists():
            self._csv_cache[key] = None
            return None
        try:
            df = _read_csv_file(path)
        except pl.exceptions.NoDataError:
            df = pl.DataFrame()
        self._csv_cache[key] = df
        return df

    # ------------------------------------------------------------------
    # Cache activation (process-level, scoped via context manager)
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Δ.12a — install the read-cache so helper modules' direct
        ``_read_csv_file`` calls hit this context's cache on repeat.

        Idempotent: calling ``activate`` twice on the same context is
        a no-op.  Pair with :meth:`deactivate` to clear, or use the
        context-manager protocol.
        """
        # Build the active-cache view (DataFrame-only; absent-file
        # entries from ``read_csv`` are dropped because
        # ``_read_csv_file`` doesn't track absences).  Subsequent
        # cache misses populate ``cache_view`` directly through
        # ``_read_csv_file``; ``read_csv`` continues to populate
        # ``_csv_cache`` with the broader (None-included) shape.
        cache_view: dict[str, pl.DataFrame] = {
            k: v for k, v in self._csv_cache.items() if v is not None
        }
        _install_csv_cache(cache_view)
        self._active_cache = cache_view

    def deactivate(self) -> None:
        """Δ.12a — uninstall the read-cache."""
        _install_csv_cache(None)
        self._active_cache = None

    def __enter__(self) -> "SolveContext":
        self.activate()
        return self

    def __exit__(self, *exc) -> None:
        self.deactivate()

    # ------------------------------------------------------------------
    # Lazy DataFrame fields
    # ------------------------------------------------------------------

    @property
    def period_in_use(self) -> pl.DataFrame:
        if not self._period_in_use_loaded:
            self._period_in_use = _load_period_in_use(
                self.solve_data_dir / "period_in_use_set.csv",
                provider=self.provider,
            )
            self._period_in_use_loaded = True
        return self._period_in_use

    @property
    def period_branch(self) -> pl.DataFrame:
        if not self._period_branch_loaded:
            self._period_branch = _load_period_branch(
                self.solve_data_dir / "period__branch.csv",
                provider=self.provider,
            )
            self._period_branch_loaded = True
        return self._period_branch

    @property
    def edd_history(self) -> pl.DataFrame:
        if not self._edd_history_loaded:
            self._edd_history = _load_edd_history(
                self.solve_data_dir / "edd_history.csv",
                provider=self.provider,
            )
            self._edd_history_loaded = True
        return self._edd_history

    @property
    def p_entity_pre_existing(self) -> pl.DataFrame:
        if not self._ppe_loaded:
            self._ppe = _maybe_read(
                self.solve_data_dir / "p_entity_pre_existing.csv",
                provider=self.provider,
            )
            self._ppe_loaded = True
        return self._ppe

    # ── Path B Cat B (WriterSnapshot top-7) accessors ─────────────────
    @property
    def steps_in_use(self) -> pl.DataFrame:
        """``[d, t, step_duration]`` from ``steps_in_use.csv`` —
        canonical column names.  Empty when the file is missing.

        Path B / audit §Category B: cascade helpers that previously
        re-read the CSV (``_derived_params.py:5421, 7974``;
        ``_derived_branch.py:285``) should prefer this typed accessor.
        """
        if not self._steps_in_use_loaded:
            self._steps_in_use = _load_steps_in_use(
                self.solve_data_dir / "steps_in_use.csv",
                provider=self.provider,
            )
            self._steps_in_use_loaded = True
        return self._steps_in_use

    @property
    def period_share_of_year(self) -> pl.DataFrame:
        """``[d, value]`` from ``complete_period_share_of_year_calc.csv``
        (canonical form: ``period`` → ``d``).  Falls back to the non-
        ``_calc`` variant when only that file exists.  Empty when neither
        is present.

        Path B / audit §Category B: cascade helpers that previously
        re-read the CSV (``_derived_params.py:7990``) should prefer this
        typed accessor.
        """
        if not self._period_share_loaded:
            self._period_share = _load_period_share(
                self.solve_data_dir, provider=self.provider,
            )
            self._period_share_loaded = True
        return self._period_share

    @property
    def p_entity_all_existing(self) -> pl.DataFrame:
        """``p_entity_all_existing.csv`` — preserved schema (canonical
        column names already present at writer side).  Empty when the
        file is missing.

        Path B / audit §Category B: cascade helpers that previously
        re-read the CSV (``_derived_params.py:4940``) should prefer this
        typed accessor.
        """
        if not self._p_entity_all_existing_loaded:
            self._p_entity_all_existing = _maybe_read(
                self.solve_data_dir / "p_entity_all_existing.csv",
                provider=self.provider,
            )
            self._p_entity_all_existing_loaded = True
        return self._p_entity_all_existing

    @property
    def solve_branch_weight(self) -> pl.DataFrame:
        """``[b, p_branch_weight_input]`` from
        ``solve_branch_weight.csv`` (rename ``branch`` → ``b``).  Empty
        when the file is missing.

        Path B / audit §Category B: cascade helpers that previously
        re-read the CSV (``_derived_params.py:7623``) should prefer
        this typed accessor.
        """
        if not self._solve_branch_weight_loaded:
            self._solve_branch_weight = _load_solve_branch_weight(
                self.solve_data_dir / "solve_branch_weight.csv",
                provider=self.provider,
            )
            self._solve_branch_weight_loaded = True
        return self._solve_branch_weight

    # ------------------------------------------------------------------
    # Typed-field convenience accessors
    # ------------------------------------------------------------------

    @property
    def solve_data_dir(self) -> Path:
        return self.workdir / "solve_data"

    @property
    def input_dir(self) -> Path:
        return self.workdir / "input"


# ---------------------------------------------------------------------------
# Loaders for the typed fields
# ---------------------------------------------------------------------------


def _carrier_miss(path: Path, consumer: str) -> "FlexDataError":
    """Build the strict-Provider miss error for a SolveContext loader.

    The producer-side writer should have populated the carrier before
    the loader runs.  A miss here is a cascade bug — likely the writer
    didn't run, or its frame was emitted under a non-canonical key.
    """
    return FlexDataError(
        f"FlexDataProvider has no carrier for '{path.name}' "
        f"(consumer: {consumer}).  This per-solve frame must be "
        f"populated by its emitter before SolveContext consumes it.  "
        f"Check that the producer ran and registered under the canonical "
        f"key '{_provider_key_for(path)}'."
    )


def _provider_key_for(path: Path) -> str:
    """Return the canonical Provider key for *path* (parent/stem)."""
    parent = path.parent.name
    stem = path.stem
    if parent:
        return f"{parent}/{stem}"
    return stem


class FlexDataError(RuntimeError):
    """Raised when the Provider lacks a carrier the cascade requires.

    This is a programming/wiring error — surfaced rather than worked
    around so the broken edge is visible.  See ``specs/enum_dtype_refactor_plan.md``
    §Phase 3 for the rationale (no silent disk-fallback arms).
    """


def _provider_fetch_or_raise(
    provider: "object", path: Path, consumer: str,
) -> pl.DataFrame:
    """Fetch *path*'s carrier from *provider* strictly; raise on miss."""
    from ._emit_provider_io import _provider_key
    key = _provider_key(path)
    if provider.has(key):
        df = provider.get(key)
        if df is not None:
            return df
    raise _carrier_miss(path, consumer)


def _maybe_read(path: Path,
                 *, provider: "object | None" = None) -> pl.DataFrame:
    """Eager read returning empty frame on missing / empty CSV.

    Provider-first: when *provider* is supplied, the carrier is fetched
    strictly from the Provider; a missing key raises :class:`FlexDataError`.
    Empty frames (height 0) are returned as ``pl.DataFrame()`` so callers
    that ``if df.height == 0`` continue to short-circuit cleanly.

    Disk fallback only runs when *provider* is None (the test-only path
    that remains until Phase 4 activation makes provider mandatory).
    """
    if provider is not None:
        df = _provider_fetch_or_raise(provider, path, "SolveContext._maybe_read")
        if df.height == 0:
            return pl.DataFrame()
        return df
    if not path.exists():
        return pl.DataFrame()
    try:
        return _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def _read_active_solve(workdir: Path,
                        *, provider: "object | None" = None) -> str | None:
    """Mirror of ``_derived_params._read_active_solve``.

    Provider-first read; falls back to disk when the Provider is absent
    or doesn't carry the frame.
    """
    from flextool.engine_polars._emit_provider_io import _provider_key
    p = workdir / "solve_data" / "solve_current.csv"
    if provider is not None and provider.has(_provider_key(p)):
        df = provider.get(_provider_key(p))
        if df is None or df.height == 0:
            return None
        col = df.columns[0]
        return df[col][0]
    if not p.exists():
        return None
    try:
        df = _read_csv_file(p)
    except pl.exceptions.NoDataError:
        return None
    if df.height == 0:
        return None
    col = df.columns[0]
    return df[col][0]


def _read_solve_first(work_folder: Path,
                       *, provider: "object | None" = None) -> bool:
    """Mirror of ``input._read_solve_first``.

    Reads ``modelParam == 'solveFirst'`` from ``p_model.csv``.  Resolves
    in order: ``solve_data/p_model.csv`` → ``input/p_model.csv`` → True.
    """
    from flextool.engine_polars._emit_provider_io import _provider_key

    def _cell_str(value: "object | None") -> str:
        return "" if value is None else str(value)

    for cand in ("solve_data/p_model.csv", "input/p_model.csv"):
        path = work_folder / cand
        key = _provider_key(path)
        if not provider.has(key):
            continue
        df = provider.get(key)
        if "modelParam" not in df.columns or "p_model" not in df.columns:
            return True
        for r in df.iter_rows(named=True):
            if _cell_str(r["modelParam"]) == "solveFirst":
                try:
                    return bool(int(_cell_str(r["p_model"])))
                except (ValueError, TypeError):
                    return True
        # Frame existed but didn't contain the flag — treat as default.
        return True
    return True


def _read_period_set(path: Path,
                      *, provider: "object | None" = None) -> set[str]:
    """Read a single-column period CSV (header row, then one period per row)."""
    from flextool.engine_polars._emit_provider_io import _provider_key

    def _cell_str(value: "object | None") -> str:
        return "" if value is None else str(value)

    key = _provider_key(path)
    if not provider.has(key):
        return set()
    df = provider.get(key)
    out: set[str] = set()
    if df.width == 0:
        return out
    col0 = df.columns[0]
    for value in df.get_column(col0):
        c0 = _cell_str(value)
        if c0:
            out.add(c0)
    return out


def _read_realized_dispatch_periods(path: Path,
                                     *, provider: "object | None" = None) -> set[str]:
    """Read distinct periods from ``realized_dispatch.csv``."""
    from flextool.engine_polars._emit_provider_io import _provider_key

    def _cell_str(value: "object | None") -> str:
        return "" if value is None else str(value)

    key = _provider_key(path)
    if not provider.has(key):
        return set()
    df = provider.get(key)
    if "period" not in df.columns:
        return set()
    out: set[str] = set()
    for value in df.get_column("period"):
        c = _cell_str(value)
        if c:
            out.add(c)
    return out


def _read_frame(path: Path,
                 *, provider: "object | None" = None,
                 consumer: str = "SolveContext") -> pl.DataFrame | None:
    """Provider-first frame fetch with strict semantics.

    Returns the frame when present (Provider lookup if *provider* is
    supplied; disk read when *provider* is None — the legacy test-only
    path).  Returns ``None`` when the source genuinely yields an empty
    body (height 0 or NoDataError).  Raises :class:`FlexDataError` when
    a Provider is supplied but lacks the carrier.
    """
    if provider is not None:
        df = _provider_fetch_or_raise(provider, path, consumer)
        if df.height == 0:
            return None
        return df
    if not path.exists():
        return None
    try:
        df = _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return None
    if df.height == 0:
        return None
    return df


def _load_period_in_use(path: Path,
                         *, provider: "object | None" = None) -> pl.DataFrame:
    """Load ``period_in_use_set.csv`` and rename to canonical ``[d]``.

    Preserves CSV row order (``unique(maintain_order=True)``) so callers
    that consume the frame as an ordered list of periods (e.g. the
    canonical-order reorder in ``_dt_period_active_steps``) get the
    same active_time_list ordering the workdir CSV exposes.

    Phase 4.6 — cast the renamed ``d`` column to the canonical axis
    Enum when activation is on so downstream joins against cascade
    frames (which are Enum-typed) match dtypes.
    """
    from flextool.engine_polars._axis_enums import (
        get_global_axis_enums,
        cast_frame_axes,
    )
    _live = get_global_axis_enums()
    _empty_d_dtype = (
        _live.get("d", pl.Utf8) if _live is not None else pl.Utf8
    )
    empty = pl.DataFrame(schema={"d": _empty_d_dtype})
    df = _read_frame(path, provider=provider,
                      consumer="SolveContext.period_in_use")
    if df is None:
        return empty
    df = df.rename({df.columns[0]: "d"})
    df = df.select("d").unique(maintain_order=True)
    if _live is not None:
        df = cast_frame_axes(df, _live)
    return df


def _load_period_branch(path: Path,
                         *, provider: "object | None" = None) -> pl.DataFrame:
    """Load ``period__branch.csv`` as ``[d_anchor, b]``.

    Phase 4.6 — under activation cast the dim columns to the canonical
    axis enums so downstream joins see Enum dtypes consistently.
    """
    from flextool.engine_polars._axis_enums import (
        get_global_axis_enums,
        cast_frame_axes,
    )
    _live = get_global_axis_enums()
    _d_anchor = _live.get("d_anchor", pl.Utf8) if _live is not None else pl.Utf8
    _b = _live.get("branch", pl.Utf8) if _live is not None else pl.Utf8
    empty = pl.DataFrame(schema={"d_anchor": _d_anchor, "b": _b})
    df = _read_frame(path, provider=provider,
                      consumer="SolveContext.period_branch")
    if df is None:
        return empty
    rename = {}
    if "period" in df.columns:
        rename["period"] = "d_anchor"
    if "branch" in df.columns:
        rename["branch"] = "b"
    df = df.rename(rename)
    cols = [c for c in ("d_anchor", "b") if c in df.columns]
    df = df.select(cols).unique(maintain_order=True) if cols else empty
    if _live is not None:
        df = cast_frame_axes(df, _live)
    return df


def _load_edd_history(path: Path,
                       *, provider: "object | None" = None) -> pl.DataFrame:
    """Load ``edd_history.csv`` — schema preserved as-is."""
    df = _read_frame(path, provider=provider,
                      consumer="SolveContext.edd_history")
    if df is None:
        return pl.DataFrame()
    return df


def _load_steps_in_use(path: Path,
                        *, provider: "object | None" = None) -> pl.DataFrame:
    """Load ``steps_in_use.csv`` and rename to canonical ``[d, t,
    step_duration]``.  Empty frame when missing.

    Audit §Category B (WriterSnapshot top-7): the cascade helpers
    consume this frame after renaming ``period`` → ``d`` and
    ``step`` / ``time`` → ``t``; this helper centralises that rename
    so the cascade sees the canonical form directly.

    Phase 4.6 — cast ``d``/``t`` to canonical axis enums when
    activation is on so downstream joins (against cascade frames that
    are Enum-typed) match dtypes.
    """
    from flextool.engine_polars._axis_enums import (
        get_global_axis_enums,
        cast_frame_axes,
    )
    _live = get_global_axis_enums()
    _d_dt = _live.get("d", pl.Utf8) if _live is not None else pl.Utf8
    _t_dt = _live.get("t", pl.Utf8) if _live is not None else pl.Utf8
    empty = pl.DataFrame(
        schema={"d": _d_dt, "t": _t_dt, "step_duration": pl.Float64}
    )
    df = _read_frame(path, provider=provider,
                      consumer="SolveContext.steps_in_use")
    if df is None:
        return empty
    period_col = next((c for c in ("period", "d") if c in df.columns), None)
    step_col = next(
        (c for c in ("step", "t", "time") if c in df.columns), None
    )
    if period_col is None or step_col is None:
        return empty
    out = df.rename({period_col: "d", step_col: "t"})
    if "step_duration" in out.columns:
        out = out.with_columns(
            pl.col("step_duration").cast(pl.Float64, strict=False)
        )
    cols = [c for c in ("d", "t", "step_duration") if c in out.columns]
    out = out.select(cols)
    if _live is not None:
        out = cast_frame_axes(out, _live)
    return out


def _load_period_share(solve_data_dir: Path,
                        *, provider: "object | None" = None) -> pl.DataFrame:
    """Load ``complete_period_share_of_year_calc.csv`` (preferred) or
    its non-``_calc`` variant — rename ``period`` → ``d``.

    Audit §Category B (WriterSnapshot top-7): the cascade currently
    falls through both names at its read sites (e.g.
    ``_derived_params.py:7984-7988``).  Centralising the fallback here
    gives every consumer the same semantics.
    """
    empty = pl.DataFrame(schema={"d": pl.Utf8, "value": pl.Float64})
    cand_paths = (
        solve_data_dir / "complete_period_share_of_year_calc.csv",
        solve_data_dir / "complete_period_share_of_year.csv",
    )
    if provider is not None:
        # Provider-strict: at least one of the canonical variants must
        # carry the frame.  We probe ``_calc`` first (the producer's
        # default), then the legacy non-``_calc`` name as a fallback.
        from ._emit_provider_io import _provider_key
        for path in cand_paths:
            key = _provider_key(path)
            if provider.has(key):
                df = provider.get(key)
                if df is None or df.height == 0:
                    continue
                out = df
                if "period" in out.columns:
                    out = out.rename({"period": "d"})
                if "value" in out.columns:
                    out = out.with_columns(
                        pl.col("value").cast(pl.Float64, strict=False)
                    )
                cols = [c for c in ("d", "value") if c in out.columns]
                if cols:
                    return out.select(cols)
        # Neither variant carried a usable frame — treat as empty.
        # ``period_share_of_year`` is genuinely optional (some fixtures
        # skip the inflation-factor pipeline entirely); empty is the
        # legacy semantics.
        return empty
    # Legacy disk path — kept until Phase 4 makes provider mandatory.
    for path in cand_paths:
        if path.exists():
            try:
                df = _read_csv_file(path)
            except pl.exceptions.NoDataError:
                continue
            if df.height == 0:
                continue
            out = df
            if "period" in out.columns:
                out = out.rename({"period": "d"})
            if "value" in out.columns:
                out = out.with_columns(
                    pl.col("value").cast(pl.Float64, strict=False)
                )
            cols = [c for c in ("d", "value") if c in out.columns]
            if cols:
                return out.select(cols)
    return empty


def _load_solve_branch_weight(path: Path,
                                *, provider: "object | None" = None) -> pl.DataFrame:
    """Load ``solve_branch_weight.csv`` and rename to canonical
    ``[b, p_branch_weight_input]``.

    Audit §Category B (WriterSnapshot top-7).
    """
    empty = pl.DataFrame(
        schema={"b": pl.Utf8, "p_branch_weight_input": pl.Float64}
    )
    df = _read_frame(path, provider=provider,
                      consumer="SolveContext.solve_branch_weight")
    if df is None:
        return empty
    ren = {}
    if "branch" in df.columns:
        ren["branch"] = "b"
    out = df.rename(ren)
    if "p_branch_weight_input" in out.columns:
        out = out.with_columns(
            pl.col("p_branch_weight_input").cast(pl.Float64, strict=False)
        )
    cols = [c for c in ("b", "p_branch_weight_input") if c in out.columns]
    return out.select(cols) if cols else empty


__all__ = ["SolveContext", "FlexDataError"]
