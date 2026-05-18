""":class:`FlexDataProvider` — single carrier for preprocessing artefacts.

The Provider is the single abstraction by which loaders, writers,
post-processing readers, and orchestration exchange preprocessing
artefacts.  Dict-backed: ``put(name, frame)`` stores a frame;
``get(name)`` / ``has(name)`` look it up by basename (without the
``.csv`` suffix) or by parent-qualified key (``"<parent>/<stem>"``).

Name conventions
----------------

* The Provider keys frames by *name*: the CSV basename **without** the
  ``.csv`` suffix (e.g. ``"p_flow_max"``, ``"pdtNode"``,
  ``"realized_dispatch"``).
* Parent-qualified variants (``"solve_data/p_flow_max"``,
  ``"input/timeline"``) are supported.  When the same basename appears
  in two source directories (the canonical example is ``timeline.csv``
  which exists in both ``input/`` and ``solve_data/``), the caller can
  disambiguate by passing the qualified key.
* Lookup is bidirectional.  ``get`` first tries the exact key; failing
  that:

  * If the supplied key is *bare* (no ``/``), it falls back to scanning
    for any stored qualified key whose tail matches.
  * If the supplied key is *qualified*, it falls back to the bare tail.

* ``has(name)`` returns ``True`` iff ``get(name)`` would return a
  non-``None`` frame, and uses the same lookup logic.

Suffix handling
---------------

The Provider stores keys *without* the ``.csv`` suffix.  ``get`` /
``has`` / ``put`` accept either form — a trailing ``.csv`` is stripped
before keying.
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
        # Phase 4 — axis enum vocabulary + contract.  Populated by
        # ``input_derivation.run`` against the active SpineDBBackend, or
        # lazy-built by ``load_flextool`` against the workdir sqlite when
        # the cascade entry point bypasses input_derivation.  ``None``
        # means activation is off (legacy behaviour).
        self.axis_enums: "dict[str, pl.Enum] | None" = None
        self.contract: "object | None" = None

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
            # by tail.
            for stored_key, frame in self._frames.items():
                stored_parent, stored_tail = _split_qualified(stored_key)
                if stored_parent and stored_tail == key:
                    return frame
            return None
        # Qualified lookup — fall back to bare tail.
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
    # Snapshot — one-way disk dumps for ``--csv-dump`` debug runs.
    # ------------------------------------------------------------------

    def snapshot_raw_inputs(self, work_folder: Path) -> None:
        """Write raw input frames to *work_folder*.

        Currently a no-op — the Provider populates with *derived* /
        processed frames during the cascade; "raw inputs" don't have a
        well-defined separate carrier in the current pipeline.  Left as
        a deliberate stub so the contract surface stays stable; revisit
        if a real raw-input snapshot becomes useful.
        """
        pass

    def snapshot_processed_inputs(self, work_folder: Path) -> None:
        """Write every stored frame to ``work_folder/{name}.csv``.

        Bare-keyed frames go directly under *work_folder*; parent-qualified
        keys (``"solve_data/foo"``) materialise into subdirectories
        (``work_folder/solve_data/foo.csv``).
        """
        work_folder = Path(work_folder)
        work_folder.mkdir(parents=True, exist_ok=True)
        for name, frame in self._frames.items():
            target = work_folder / f"{name}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame.write_csv(target)


__all__ = ["FlexDataProvider"]
