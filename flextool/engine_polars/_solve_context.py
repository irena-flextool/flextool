"""Per-solve in-memory state — the typed replacement for ``solve_data/``.

Δ.12a — port per-solve preprocessing metadata into engine_polars.

This module hosts :class:`SolveContext`, the typed dataclass the
:doc:`audit/native_data_path_design_solve_context` schematic specifies.
It wraps the per-solve preprocessing artefacts that today live as CSVs
under ``workdir/solve_data/`` and exposes them as in-memory polars
frames so the override helpers (``apply_derived_a..g``) and the
``_load_*`` family in :mod:`flextool.engine_polars.input` no longer have
to re-read the same files dozens of times per solve.

Until :func:`flextool.flextoolrunner.flextoolrunner.FlexToolRunner.write_input`
is fully replaced (Δ.12c), :class:`SolveContext` reads the
``solve_data/*.csv`` files that flextool's preprocessing already wrote.
The point of going through this object is:

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
from typing import Any

import polars as pl

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
    p_entity_period_existing_capacity : pl.DataFrame
        ``[entity, period, p_entity_period_existing_capacity,
        p_entity_period_invested_capacity]`` from
        ``p_entity_period_existing_capacity.csv``.  Empty when missing.
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
    # ``period_in_use`` / ``period_branch`` / ``edd_history`` /
    # ``p_entity_period_existing_capacity`` / ``p_entity_pre_existing``
    # are populated on first access via the descriptor machinery below —
    # they're DataFrame-shaped fields and each requires a CSV read.  Most
    # parity tests only consume one or two of them, so deferring the
    # reads until an attribute access asks for them avoids paying the
    # IO cost upfront.
    _period_in_use_loaded: bool = field(default=False, repr=False)
    _period_in_use: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={"d": pl.Utf8}),
        repr=False,
    )
    _period_branch_loaded: bool = field(default=False, repr=False)
    _period_branch: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"d_anchor": pl.Utf8, "b": pl.Utf8}
        ),
        repr=False,
    )
    _edd_history_loaded: bool = field(default=False, repr=False)
    _edd_history: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={}), repr=False,
    )
    _ppec_loaded: bool = field(default=False, repr=False)
    _ppec: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={}), repr=False,
    )
    _ppe_loaded: bool = field(default=False, repr=False)
    _ppe: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema={}), repr=False,
    )

    # ── Path B Cat B (WriterSnapshot top-7) extensions ──────────────────
    # Per-solve preprocessing artefacts written by ``_writer_solve_writers``
    # / ``_writer_period_calc`` and consumed by the cascade (audit
    # `specs/in_cascade_csv_audit.md` §Category B).  Lazy-loaded on first
    # attribute access — same pattern as ``period_in_use`` /
    # ``period_branch`` above.  Schemas mirror the renamed canonical form
    # the cascade helpers themselves construct, so callers can swap a
    # ``_read_csv_file + rename + select + unique`` chain for a single
    # attribute access.
    _steps_in_use_loaded: bool = field(default=False, repr=False)
    _steps_in_use: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"d": pl.Utf8, "t": pl.Utf8, "step_duration": pl.Float64}
        ),
        repr=False,
    )
    _period_share_loaded: bool = field(default=False, repr=False)
    _period_share: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            schema={"d": pl.Utf8, "value": pl.Float64}
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
            schema={"b": pl.Utf8, "p_branch_weight_input": pl.Float64}
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
    def from_workdir(cls, workdir: Path | str) -> "SolveContext":
        """Construct a SolveContext from a flextool workdir.

        Eagerly reads only the cheap typed scalars
        (``solve_name``, ``solveFirst``, ``realized_periods``,
        ``realized_invest_periods``) — these come from small single-
        column CSVs and are consumed by the apply_derived_* boundary
        checks that gate the rest of the cascade.

        The DataFrame-shaped fields (``period_in_use``, ``period_branch``,
        ``edd_history``, ``p_entity_period_existing_capacity``,
        ``p_entity_pre_existing``) are loaded **lazily** on first
        attribute access.  Most parity tests only consume one or two,
        so paying the IO cost upfront for a bag of frames the test
        never touches dominates the cache savings on small fixtures.
        """
        wd = Path(workdir)
        sd = wd / "solve_data"
        ctx = cls(workdir=wd)
        ctx.solve_name = _read_active_solve(wd)
        ctx.solveFirst = _read_solve_first(wd)
        ctx.realized_periods = _read_realized_dispatch_periods(
            sd / "realized_dispatch.csv"
        )
        ctx.realized_invest_periods = _read_period_set(
            sd / "realized_invest_periods_of_current_solve.csv"
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
                self.solve_data_dir / "period_in_use_set.csv"
            )
            self._period_in_use_loaded = True
        return self._period_in_use

    @property
    def period_branch(self) -> pl.DataFrame:
        if not self._period_branch_loaded:
            self._period_branch = _load_period_branch(
                self.solve_data_dir / "period__branch.csv"
            )
            self._period_branch_loaded = True
        return self._period_branch

    @property
    def edd_history(self) -> pl.DataFrame:
        if not self._edd_history_loaded:
            self._edd_history = _load_edd_history(
                self.solve_data_dir / "edd_history.csv"
            )
            self._edd_history_loaded = True
        return self._edd_history

    @property
    def p_entity_period_existing_capacity(self) -> pl.DataFrame:
        if not self._ppec_loaded:
            self._ppec = _maybe_read(
                self.solve_data_dir / "p_entity_period_existing_capacity.csv"
            )
            self._ppec_loaded = True
        return self._ppec

    @property
    def p_entity_pre_existing(self) -> pl.DataFrame:
        if not self._ppe_loaded:
            self._ppe = _maybe_read(
                self.solve_data_dir / "p_entity_pre_existing.csv"
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
                self.solve_data_dir / "steps_in_use.csv"
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
            self._period_share = _load_period_share(self.solve_data_dir)
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
                self.solve_data_dir / "p_entity_all_existing.csv"
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
                self.solve_data_dir / "solve_branch_weight.csv"
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


def _maybe_read(path: Path) -> pl.DataFrame:
    """Eager read returning empty frame on missing / empty CSV."""
    if not path.exists():
        return pl.DataFrame()
    try:
        return _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def _read_active_solve(workdir: Path) -> str | None:
    """Mirror of ``_derived_params._read_active_solve``."""
    p = workdir / "solve_data" / "solve_current.csv"
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


def _read_solve_first(work_folder: Path) -> bool:
    """Mirror of ``input._read_solve_first``.

    Reads ``modelParam == 'solveFirst'`` from ``p_model.csv``.  Resolves
    in order: ``solve_data/p_model.csv`` → ``input/p_model.csv`` → True.
    """
    import csv as _csv

    for cand in ("solve_data/p_model.csv", "input/p_model.csv"):
        path = work_folder / cand
        if not path.exists():
            continue
        with path.open() as fh:
            reader = _csv.reader(fh)
            header = next(reader, None) or []
            try:
                param_idx = header.index("modelParam")
                value_idx = header.index("p_model")
            except ValueError:
                return True
            for r in reader:
                if (
                    len(r) > max(param_idx, value_idx)
                    and r[param_idx] == "solveFirst"
                ):
                    try:
                        return bool(int(r[value_idx]))
                    except (ValueError, TypeError):
                        return True
        return True
    return True


def _read_period_set(path: Path) -> set[str]:
    """Read a single-column period CSV (header row, then one period per row)."""
    import csv as _csv

    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as fh:
        reader = _csv.reader(fh)
        next(reader, None)
        for r in reader:
            if r and r[0]:
                out.add(r[0])
    return out


def _read_realized_dispatch_periods(path: Path) -> set[str]:
    """Read distinct periods from ``realized_dispatch.csv``."""
    import csv as _csv

    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as fh:
        reader = _csv.reader(fh)
        header = next(reader, None) or []
        try:
            i = header.index("period")
        except ValueError:
            return set()
        for r in reader:
            if len(r) > i and r[i]:
                out.add(r[i])
    return out


def _load_period_in_use(path: Path) -> pl.DataFrame:
    """Load ``period_in_use_set.csv`` and rename to canonical ``[d]``.

    Preserves CSV row order (``unique(maintain_order=True)``) so callers
    that consume the frame as an ordered list of periods (e.g. the
    canonical-order reorder in ``_dt_period_active_steps``) get the
    same active_time_list ordering the workdir CSV exposes.
    """
    if not path.exists():
        return pl.DataFrame(schema={"d": pl.Utf8})
    try:
        df = _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame(schema={"d": pl.Utf8})
    if df.height == 0:
        return pl.DataFrame(schema={"d": pl.Utf8})
    df = df.rename({df.columns[0]: "d"})
    return df.select("d").unique(maintain_order=True)


def _load_period_branch(path: Path) -> pl.DataFrame:
    """Load ``period__branch.csv`` as ``[d_anchor, b]``."""
    if not path.exists():
        return pl.DataFrame(schema={"d_anchor": pl.Utf8, "b": pl.Utf8})
    try:
        df = _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame(schema={"d_anchor": pl.Utf8, "b": pl.Utf8})
    if df.height == 0:
        return pl.DataFrame(schema={"d_anchor": pl.Utf8, "b": pl.Utf8})
    rename = {}
    if "period" in df.columns:
        rename["period"] = "d_anchor"
    if "branch" in df.columns:
        rename["branch"] = "b"
    df = df.rename(rename)
    cols = [c for c in ("d_anchor", "b") if c in df.columns]
    return df.select(cols).unique(maintain_order=True) if cols else pl.DataFrame(
        schema={"d_anchor": pl.Utf8, "b": pl.Utf8}
    )


def _load_edd_history(path: Path) -> pl.DataFrame:
    """Load ``edd_history.csv`` — schema preserved as-is."""
    if not path.exists():
        return pl.DataFrame()
    try:
        return _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return pl.DataFrame()


def _load_steps_in_use(path: Path) -> pl.DataFrame:
    """Load ``steps_in_use.csv`` and rename to canonical ``[d, t,
    step_duration]``.  Empty frame when missing.

    Audit §Category B (WriterSnapshot top-7): the cascade helpers
    consume this frame after renaming ``period`` → ``d`` and
    ``step`` / ``time`` → ``t``; this helper centralises that rename
    so the cascade sees the canonical form directly.
    """
    empty = pl.DataFrame(
        schema={"d": pl.Utf8, "t": pl.Utf8, "step_duration": pl.Float64}
    )
    if not path.exists():
        return empty
    try:
        df = _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return empty
    if df.height == 0:
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
    return out.select(cols)


def _load_period_share(solve_data_dir: Path) -> pl.DataFrame:
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


def _load_solve_branch_weight(path: Path) -> pl.DataFrame:
    """Load ``solve_branch_weight.csv`` and rename to canonical
    ``[b, p_branch_weight_input]``.

    Audit §Category B (WriterSnapshot top-7).
    """
    empty = pl.DataFrame(
        schema={"b": pl.Utf8, "p_branch_weight_input": pl.Float64}
    )
    if not path.exists():
        return empty
    try:
        df = _read_csv_file(path)
    except pl.exceptions.NoDataError:
        return empty
    if df.height == 0:
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


__all__ = ["SolveContext"]
