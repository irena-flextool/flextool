"""Step 1-a — :class:`FlexDataProvider` scaffolding.

The Provider is the single abstraction that eventually replaces both the
in-memory :class:`FlexDataAccumulator` carrier and the path-based seed
funnel (:func:`_seed_lookup` / :func:`_seed_or_exists` / :func:`_seed_or_pick`
/ :func:`_seed_open`).  See
``specs/flex_data_provider_migration_handoff.md`` for the multi-step plan.

This file is the *first* step (S1-a): a plain dict-backed class with
``get`` / ``has`` / ``put`` plus stub ``snapshot_*`` methods.  It is NOT
yet wired into the cascade — the loaders, writers, post-processing
readers, and orchestration still use the seed funnel.  Subsequent steps
migrate consumers one file at a time, then delete the seed funnel
wholesale.

Name conventions
----------------

* The Provider keys frames by *name*: the CSV basename **without** the
  ``.csv`` suffix (e.g. ``"p_flow_max"``, ``"pdtNode"``,
  ``"realized_dispatch"``).  This mirrors the public interface contract
  in the handoff (Step 1 section) and is one step removed from the
  accumulator's path-style keys (``"p_flow_max.csv"``).
* Parent-qualified variants (``"solve_data/p_flow_max"``,
  ``"input/timeline"``) are supported.  When the same basename appears
  in two source directories (the canonical example is ``timeline.csv``
  which exists in both ``input/`` and ``solve_data/``), the caller can
  disambiguate by passing the qualified key.
* Lookup is bidirectional and mirrors the seed funnel's ``_seed_or_pick``
  pattern.  ``get`` first tries the exact key as supplied; failing that:

  * If the supplied key is *bare* (no ``/``), it falls back to scanning
    for any stored qualified key whose tail matches.  This is the
    "promote bare lookup to whatever qualified frame is present" path.
  * If the supplied key is *qualified*, it falls back to the bare tail.
    This is the "demote qualified lookup to the basename" path; same
    behaviour as the seed funnel passing a qualified path to
    ``_seed_lookup`` when only the bare basename was stashed.

  The two-direction fallback is intentional for the scaffolding stage —
  consumers being migrated in Step 1 read with whichever form the
  legacy seed call used (bare for ``_seed_or_exists``, qualified for
  ``_seed_open``).  Once every consumer is migrated and we settle on a
  single style, the fallback can be tightened.

* ``has(name)`` returns ``True`` iff ``get(name)`` would return a
  non-``None`` frame, and uses the same lookup logic.

Suffix handling
---------------

The Provider stores keys *without* the ``.csv`` suffix.  ``get`` /
``has`` / ``put`` accept either form — a trailing ``.csv`` is stripped
before keying.  This matches the handoff's interface contract: "Names
are basenames without the .csv suffix" while still being lenient about
callers that pass full filenames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import polars as pl


def _strip_csv(name: str) -> str:
    """Drop a trailing ``.csv`` suffix if present."""
    if name.endswith(".csv"):
        return name[: -len(".csv")]
    return name


def _split_qualified(key: str) -> tuple[str, str]:
    """Return ``(parent, tail)`` for *key*.

    ``parent`` is ``""`` for a bare key.  ``tail`` is always the
    basename without parent.  Only the rightmost ``/`` is used so deeper
    paths (uncommon in this codebase) collapse to ``parent`` ≠ ``""``.
    """
    if "/" in key:
        parent, _, tail = key.rpartition("/")
        return parent, tail
    return "", key


class FlexDataProvider:
    """Dict-backed carrier of preprocessing artefacts.

    Interface contract — see the handoff (Step 1 section).  This is the
    scaffolding class for the migration; subsequent steps will replace
    its consumers and eventually evolve it into the single data pathway
    of the cascade.

    The class is intentionally minimal and "dumb" — no caching beyond
    the dict, no lazy loading, no eviction, no indexing.  Optimisations
    come later (Step 3) once measurements show where the peaks are.
    """

    def __init__(self) -> None:
        self._frames: dict[str, pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def put(self, name: str, frame: pl.DataFrame) -> None:
        """Store *frame* under *name*.

        *name* may be bare (``"p_flow_max"``), qualified
        (``"solve_data/p_flow_max"``), and may include the ``.csv``
        suffix in either form — the suffix is stripped before keying.
        """
        key = _strip_csv(name)
        self._frames[key] = frame

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> pl.DataFrame | None:
        """Return the frame for *name*, or ``None`` if unavailable.

        Lookup logic:

        1. Exact match on the supplied (suffix-stripped) key.
        2. If the supplied key is bare, scan for any stored qualified
           key whose tail matches.  Returns the first match in
           insertion order.
        3. If the supplied key is qualified, fall back to the bare
           tail.

        Returns ``None`` if no candidate matches.
        """
        key = _strip_csv(name)
        if key in self._frames:
            return self._frames[key]
        parent, tail = _split_qualified(key)
        if parent == "":
            # Bare lookup — promote to whatever qualified frame matches
            # by tail.  Mirrors ``_seed_or_pick`` which accepts the
            # first hit across multiple parent dirs.
            for stored_key, frame in self._frames.items():
                stored_parent, stored_tail = _split_qualified(stored_key)
                if stored_parent and stored_tail == key:
                    return frame
            return None
        # Qualified lookup — fall back to bare tail.  Mirrors the seed
        # funnel's behaviour when only the bare basename was stashed
        # but the caller passes a full path.
        return self._frames.get(tail)

    def has(self, name: str) -> bool:
        """Return ``True`` iff :meth:`get` would return a non-``None``
        frame for *name*."""
        return self.get(name) is not None

    # ------------------------------------------------------------------
    # Iteration helpers (handy for tests + future snapshot impls).
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        return list(self._frames.keys())

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return self.has(name)

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - trivial
        return iter(self._frames)

    def items(self) -> Iterable[tuple[str, pl.DataFrame]]:  # pragma: no cover
        return self._frames.items()

    # ------------------------------------------------------------------
    # Snapshot stubs — implemented in Step 2 when ``--csv-dump`` is
    # rewired to dump from the live Provider rather than from the seed
    # funnel / direct writer emission.
    # ------------------------------------------------------------------

    # TODO Step 2: dump raw input frames the Provider received from
    # the source DB (pre-preprocessing).
    def snapshot_raw_inputs(self, work_folder: Path) -> None:
        """Write raw input frames to *work_folder*.

        Step 1 stub — implemented in Step 2.  See the handoff doc.
        """
        pass

    # TODO Step 2: dump derived / processed frames the writers populated
    # into the Provider during the cascade.
    def snapshot_processed_inputs(self, work_folder: Path) -> None:
        """Write processed (derived) frames to *work_folder*.

        Step 1 stub — implemented in Step 2.  See the handoff doc.
        """
        pass


__all__ = ["FlexDataProvider"]
