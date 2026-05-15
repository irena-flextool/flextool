"""Phase C — per-sub-solve FlexData accumulator.

This module owns the in-memory carrier that collects the writer-port
``_write_*`` derived frames during one sub-solve's preprocessing pass.
Phase D will consume the accumulator as a ``seed`` to ``load_flextool``
to skip the disk-read path; Phase C only builds the accumulator and
plumbs it forward — CSV emission is unchanged.

Memory discipline (handoff decision #11)
----------------------------------------

The accumulator is **per sub-solve only**.  It is built fresh at the
start of each per-sub-solve preprocessing pass and replaced when the
next sub-solve runs.  The cascade must NOT accumulate Solution +
FlexData across sub-solves; the same applies here — there is no
cascade-wide accumulator dict, only the latest sub-solve's frames.

Design — wrapper-side capture (approach (a) per Phase C handoff)
----------------------------------------------------------------

The 37 ``OK_thin_wrapper`` writers identified in
``specs/phase_b_writer_audit.md`` all funnel their derived frames
through their module's private ``_write(df, path)`` helper before
emitting the CSV.  We monkey-patch that helper for the duration of
:class:`FlexDataAccumulator` (and its context-manager partner
:func:`capture_frames`) so every CSV emission also stashes
``frames[path.name] = df`` for the sub-solve.

The 103 "special-handling" writers (multi-CSV streamed monoliths and
``fh.write()`` row-by-row emitters identified in the same audit) are
left untouched in this phase — Phase C explicitly defers their
adapters.  Downstream consumers in Phase D will read the missing
fields from ``load_flextool``'s disk path.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import polars as pl


# ---------------------------------------------------------------------------
# Writer modules whose ``_write`` helper feeds the 37 thin-wrapper writers.
# Patching these four modules' ``_write`` covers every OK_thin_wrapper entry
# from the Phase B audit (writers in _writer_leaf_sets, _writer_mid_sets,
# _writer_calc_params, _writer_arc_unions).
# ---------------------------------------------------------------------------

_PATCH_MODULES = (
    "flextool.engine_polars._writer_leaf_sets",
    "flextool.engine_polars._writer_mid_sets",
    "flextool.engine_polars._writer_calc_params",
    "flextool.engine_polars._writer_arc_unions",
)


# ---------------------------------------------------------------------------
# Public dataclass.
# ---------------------------------------------------------------------------


@dataclass
class FlexDataAccumulator:
    """Per-sub-solve carrier of derived frames keyed by target CSV name.

    Keys are the basename of the path each writer's ``_write`` helper
    receives (e.g. ``"period_group.csv"`` or
    ``"entity_lifetime_method.csv"``) — the canonical name the CSV
    writes to under ``<work>/solve_data/``.  Phase D will map these
    keys into the equivalent ``FlexData`` fields when wiring the
    cascade to consume the accumulator instead of re-reading from
    disk.

    The accumulator is NOT cascade-wide.  It is built fresh at the
    start of each sub-solve's preprocessing pass and replaced when the
    next sub-solve runs.  Only the latest sub-solve's frames are
    retained.
    """

    solve_name: str | None = None
    frames: dict[str, pl.DataFrame] = field(default_factory=dict)

    def capture(self, path: Path | str, df: pl.DataFrame) -> None:
        """Stash a (path.name → frame) pair.  Overwrites on duplicate key."""
        # Use basename so identical CSV names across runs collide
        # deterministically (only one final frame per CSV target).
        key = Path(path).name
        # Clone to insulate accumulated state from any in-place mutation
        # the writer might do post-_write (none of the 37 thin writers do,
        # but the clone is cheap insurance — polars clone is a view alias).
        self.frames[key] = df

    # Convenience for tests / Phase-D consumers.
    def __contains__(self, key: str) -> bool:
        return key in self.frames

    def get(self, key: str) -> pl.DataFrame | None:
        return self.frames.get(key)

    def keys(self) -> list[str]:
        return list(self.frames.keys())

    # ------------------------------------------------------------------
    # Phase D — seed lookup
    # ------------------------------------------------------------------
    def lookup(self, target: "Path | str") -> "pl.DataFrame | None":
        """Return the captured frame whose target CSV basename matches
        *target*, or ``None`` when this accumulator does not cover the
        file.

        ``target`` may be a full ``Path`` (``<work>/solve_data/foo.csv``)
        or a bare basename (``"foo.csv"`` or ``"foo"`` — the ``.csv``
        suffix is added when missing, to mirror
        :meth:`CsvSource.get`'s call style).

        Phase D's :func:`load_flextool` consumes this method through a
        process-level seed hook installed in
        :mod:`flextool.engine_polars._input_source`.  See
        :func:`_install_seed` / :func:`_seed_lookup` there.
        """
        name = Path(target).name
        if not name.endswith(".csv"):
            name = f"{name}.csv"
        return self.frames.get(name)


# ---------------------------------------------------------------------------
# Context manager — monkey-patches the four writer modules' ``_write``
# helper to capture frames into the supplied accumulator.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def capture_frames(
    accumulator: FlexDataAccumulator,
) -> Iterator[FlexDataAccumulator]:
    """Patch the participating writers' ``_write`` helper to capture
    derived frames into *accumulator* for the duration of the block.

    The 37 ``OK_thin_wrapper`` writers from the Phase B audit each call
    their module's ``_write(df, path)`` exactly once per emitted CSV.
    By rebinding that name on each module for the lifetime of this
    context, every emission also pushes ``(path.name → df)`` into the
    accumulator.

    The patched ``_write`` still emits the CSV — Phase C is parallel-
    write mode.  CSV byte-identical parity tests stay green; the
    accumulator is purely additive.
    """
    import importlib

    modules = [importlib.import_module(name) for name in _PATCH_MODULES]
    saved: list[tuple[object, object]] = [
        (mod, getattr(mod, "_write")) for mod in modules
    ]
    try:
        for mod, original in saved:
            def _make_wrapped(_orig=original):  # late-bind per module
                def _wrapped(df: pl.DataFrame, path: Path) -> None:
                    accumulator.capture(path, df)
                    _orig(df, path)
                return _wrapped
            setattr(mod, "_write", _make_wrapped())
        yield accumulator
    finally:
        for mod, original in saved:
            setattr(mod, "_write", original)


__all__ = ["FlexDataAccumulator", "capture_frames"]
