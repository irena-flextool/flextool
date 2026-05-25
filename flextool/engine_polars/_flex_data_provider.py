""":class:`FlexDataProvider` — single carrier for preprocessing artefacts.

The Provider is the single abstraction by which loaders, writers,
post-processing readers, and orchestration exchange preprocessing
artefacts.  Dict-backed: ``put(name, frame)`` stores a frame;
``get(name)`` / ``has(name)`` look it up by exact-match key.

Name conventions
----------------

* The Provider keys frames by their **parent-qualified key**
  (``"<parent>/<stem>"``).  Producers emit qualified keys
  (``"solve_data/p_flow_max"``, ``"input/timeline"``, ``"derived/..."``,
  ``"handoff/..."``) and consumers query the same form — typically via
  the typed constants in ``_provider_keys.py`` (``K.SOLVE_DATA_...``)
  or via ``_emit_provider_io._provider_key(path)``.
* The same basename can appear under multiple parents (the canonical
  example is ``timeline.csv`` which exists in both ``input/`` and
  ``solve_data/``); each is stored under its qualified key and looked
  up by the same.
* ``get`` is an exact-match lookup against the stored key (after
  ``.csv``-suffix stripping).  A typo or unqualified key returns
  ``None`` — the Phase 0a-era bare↔qualified fallback was dropped in
  Phase 4.2-2 to give one canonical key form and one lookup path.
* ``has(name)`` returns ``True`` iff ``get(name)`` would return a
  non-``None`` frame.

Suffix handling
---------------

The Provider stores keys *without* the ``.csv`` suffix.  ``get`` /
``has`` / ``put`` accept either form — a trailing ``.csv`` is stripped
before keying.

Source tagging
--------------

``put(key, frame, *, source=None)`` accepts an optional free-form
``source`` string that is retained alongside the frame and surfaced by
:meth:`FlexDataProvider.get_source`.  The natural cascade leaves
``source`` unset; external-override writes (see
:func:`flextool.engine_polars._provider_translators.translate_overrides_to_provider`)
tag their entries with ``"external_override"`` so downstream audit
tooling can distinguish overridden keys from naturally-produced ones.
Eviction by :meth:`release_unused` drops the source entry alongside the
frame.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import polars as pl


class EvictedFrameError(KeyError):
    """Raised by :meth:`FlexDataProvider.get` when *name* has been
    evicted by :meth:`FlexDataProvider.release_unused`.

    Catching this *silently* (e.g. ``except (KeyError, EvictedFrameError)``)
    is almost certainly a bug — eviction is sound by construction
    (driven by the static lifetime map computed from declared ``READS``),
    so a hit here means either the ``READS`` declaration is incomplete
    for the active handler or a handler is reading a frame outside its
    declared scope.  Either way, surface the failure.

    The exception carries the offending frame name and the
    ``last_needed`` item-group token so the caller can pinpoint the
    drift.
    """

    def __init__(self, name: str, last_needed: Any) -> None:
        super().__init__(
            f"FlexDataProvider: frame {name!r} has been evicted "
            f"(last_needed={last_needed!r}).  This typically means "
            f"the active handler's READS declaration is incomplete "
            f"or a handler is reading a frame outside its declared "
            f"scope.  See specs/step_3_handoff.md Track B for details."
        )
        self.frame_name = name
        self.last_needed = last_needed


def _strip_csv(name: str) -> str:
    """Drop a trailing ``.csv`` suffix if present."""
    if name.endswith(".csv"):
        return name[: -len(".csv")]
    return name


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

    def __init__(
        self,
        *,
        rss_budget_mb: float | None = None,
        retain_all: bool = False,
    ) -> None:
        self._frames: dict[str, pl.DataFrame] = {}
        # Phase 6a — opt-in source-tagging.  ``put(key, frame, source=...)``
        # records the tag here; ``get_source(key)`` looks it up.  Entries
        # with ``source=None`` (the default) are *not* added to this map
        # so the natural cascade pays zero memory overhead.  Eviction
        # clears the matching entry alongside the frame.
        self._sources: dict[str, str] = {}
        # Phase 4 — axis enum vocabulary + contract.  Populated by
        # ``input_derivation.run`` against the active SpineDBBackend, or
        # lazy-built by ``load_flextool`` against the workdir sqlite when
        # the cascade entry point bypasses input_derivation.  ``None``
        # means activation is off (legacy behaviour).
        self.axis_enums: "dict[str, pl.Enum] | None" = None
        self.contract: "object | None" = None

        # Track B — READS-driven lifetime + threshold-gated eviction.
        #
        # ``_reads[handler_id]`` is the list of frame names the handler
        # declares it consumes.  ``precompute_lifetimes`` walks the
        # handler-to-item-group map and the READS declarations to compute
        # ``_last_needed[name] = max item-group that still needs it``.
        # ``release_unused(after=group)`` drops every frame whose
        # ``_last_needed`` is past ``group`` in iteration order.  After
        # eviction a subsequent ``get(name)`` raises
        # :class:`EvictedFrameError` instead of silently re-fetching —
        # we deliberately reject re-derivation (Step 3 design call).
        #
        # Eviction is threshold-gated by ``rss_budget_mb``: when the
        # currently-cached frame footprint stays below the budget, the
        # gate is closed and ``release_unused`` is a no-op.  Small
        # problems therefore pay zero eviction overhead; large problems
        # hit the memory bound.  ``retain_all`` (set by ``--csv-dump``)
        # disables eviction unconditionally so debug-snapshot runs see
        # every frame intact.
        #
        # ``rss_budget_mb`` resolution order:
        # 1. constructor arg, if non-None.
        # 2. ``FLEXTOOL_RSS_BUDGET_MB`` env var, if set.
        # 3. ``None`` (eviction never fires; threshold gate always closed).
        env_budget = os.environ.get("FLEXTOOL_RSS_BUDGET_MB")
        self.rss_budget_mb: float | None = (
            float(rss_budget_mb) if rss_budget_mb is not None
            else (float(env_budget) if env_budget else None)
        )
        self.retain_all: bool = bool(retain_all)

        self._reads: dict[str, list[str]] = {}
        # The handler-to-item-group mapping; populated by
        # ``register_handler``.  Each handler is invoked at one or more
        # item-groups (often exactly one).
        self._handler_groups: dict[str, list[Any]] = {}
        # The item-group iteration order; populated by
        # ``precompute_lifetimes``.  Maps each group token to its
        # position index (0-based).
        self._group_order: dict[Any, int] | None = None
        # The lifetime map: frame name → (item_group_token, index).
        # ``_last_needed`` is only populated after
        # ``precompute_lifetimes`` runs; before that, all frames are
        # treated as live indefinitely.
        self._last_needed: dict[str, tuple[Any, int]] = {}
        # Names that have been evicted by ``release_unused``.  Reads on
        # these raise :class:`EvictedFrameError`.
        self._evicted: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def put(
        self,
        name: str,
        frame: pl.DataFrame,
        *,
        source: str | None = None,
    ) -> None:
        """Store *frame* under *name*.

        *name* may be bare (``"p_flow_max"``), qualified
        (``"solve_data/p_flow_max"``), and may include the ``.csv``
        suffix in either form — the suffix is stripped before keying.

        Parameters
        ----------
        source:
            Optional free-form tag retained alongside the frame and
            surfaced via :meth:`get_source`.  ``None`` (the default)
            leaves no source entry — the natural-cascade case.  Phase 6a
            of ``specs/provider_consolidation.md`` adds this so the
            external-override layer can mark its writes with a stable
            ``"external_override"`` tag for downstream audit.
        """
        key = _strip_csv(name)
        self._frames[key] = frame
        if source is None:
            # Overwriting a previously-tagged entry with an untagged put
            # must clear the stale tag — the new frame is the new truth.
            self._sources.pop(key, None)
        else:
            self._sources[key] = source

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> pl.DataFrame | None:
        """Return the frame for *name*, or ``None`` if unavailable.

        Lookup logic:

        1. If *name* is in the evicted set (Track B), raise
           :class:`EvictedFrameError`.  We deliberately do NOT silently
           re-fetch — eviction is sound by static analysis when
           ``READS`` declarations are complete; a hit here points at a
           READS-declaration drift or a handler reading outside its
           declared scope.
        2. Exact match on the supplied (suffix-stripped) key.

        Returns ``None`` if no candidate matches.  The Phase 0a-era
        bare↔qualified fallback was dropped in Phase 4.2-2 — callers
        must pass the canonical parent-qualified key (typically a
        ``K.*`` constant from ``_provider_keys.py`` or the result of
        ``_emit_provider_io._provider_key(path)``).
        """
        key = _strip_csv(name)
        if key in self._evicted:
            raise EvictedFrameError(key, self._evicted[key])
        return self._frames.get(key)

    def has(self, name: str) -> bool:
        """Return ``True`` iff :meth:`get` would return a non-``None``
        frame for *name*."""
        return self.get(name) is not None

    def get_source(self, name: str) -> str | None:
        """Return the source tag recorded for *name*, or ``None``.

        Phase 6a — opt-in source-tagging.  ``None`` is returned both for
        keys that were ``put`` without a ``source`` argument and for
        keys that have never been ``put`` (or have been evicted).  This
        accessor never raises.
        """
        return self._sources.get(_strip_csv(name))

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
    # Track B — READS registry + lifetime + eviction.
    # ------------------------------------------------------------------

    def register_handler(
        self,
        handler_id: str,
        *,
        reads: Iterable[str],
        groups: Iterable[Any] | None = None,
    ) -> None:
        """Declare a handler's frame reads and (optionally) the
        item-groups it fires in.

        *handler_id* is a stable identifier — typically the fully-
        qualified function name (e.g.
        ``"flextool.engine_polars._emit_arc_unions.write_process_arc_unions"``).
        *reads* is the list of (suffix-stripped) frame names the handler
        consumes via ``get``/``has``.  *groups* is the item-group tokens
        at which the handler fires; ``None`` means "fires once at the
        sentinel single-pass group token".

        Calling twice with the same *handler_id* replaces the previous
        declaration.  The Provider does not enforce that registered
        handlers are the only ones reading frames — that's the job of
        the CI tracing mode (Phase B.3).
        """
        key = str(handler_id)
        self._reads[key] = [_strip_csv(r) for r in reads]
        self._handler_groups[key] = list(groups) if groups is not None else []

    def precompute_lifetimes(self, item_groups: Iterable[Any]) -> None:
        """Compute ``_last_needed[name] = (group, group_idx)`` from the
        currently-registered handler ``READS`` and the iteration order
        of *item_groups*.

        *item_groups* defines the order in which item-groups will be
        visited by the build loop.  For each registered handler, its
        ``groups`` are intersected with this order; the maximum group
        index across all handlers reading *name* becomes the frame's
        ``last_needed``.

        Handlers that registered with ``groups=None`` (single-pass) are
        assumed to read at every group, so they pin their READS to the
        *last* group in iteration order — effectively "retain for the
        whole build."  This is the safe default; refine the registration
        once a handler's actual group span is known.

        Must be called exactly once per build loop.  After this, the
        Provider knows which frames can be evicted *if* the memory
        threshold triggers.  Eviction itself is the job of
        :meth:`release_unused`.

        Re-running ``precompute_lifetimes`` (e.g. between sub-solves)
        re-derives the map from scratch and clears the evicted set —
        eviction state is per-build-loop, not persistent.
        """
        order = list(item_groups)
        if not order:
            # No groups → nothing to evict.
            self._group_order = {}
            self._last_needed = {}
            self._evicted = {}
            return
        group_idx: dict[Any, int] = {g: i for i, g in enumerate(order)}
        last_group_idx = len(order) - 1
        last_group_token = order[-1]

        last_needed: dict[str, tuple[Any, int]] = {}
        for handler_id, reads in self._reads.items():
            handler_groups = self._handler_groups.get(handler_id, [])
            if not handler_groups:
                # Single-pass / unknown: pin to last group.
                handler_last_idx = last_group_idx
                handler_last_token = last_group_token
            else:
                # Find the latest of the handler's groups in iteration
                # order.  Unknown groups are treated as 'past the end',
                # i.e. retain to last (defensive).
                idxs = [
                    group_idx.get(g, last_group_idx) for g in handler_groups
                ]
                handler_last_idx = max(idxs)
                handler_last_token = order[handler_last_idx]
            for name in reads:
                prev = last_needed.get(name)
                if prev is None or prev[1] < handler_last_idx:
                    last_needed[name] = (handler_last_token, handler_last_idx)
        self._group_order = group_idx
        self._last_needed = last_needed
        self._evicted = {}

    def release_unused(self, *, after: Any) -> list[str]:
        """Drop every frame whose ``_last_needed`` group index is ``≤``
        the index of *after* in the precomputed iteration order.

        Returns the list of evicted frame names (callers can log /
        assert on it).  Frames that are still needed downstream are
        untouched.

        No-op when:

        * ``retain_all`` is True (e.g. ``--csv-dump`` mode).
        * ``precompute_lifetimes`` hasn't been called.
        * The current footprint (``rss_estimate_mb``) is below the
          ``rss_budget_mb`` threshold — small problems pay no eviction
          overhead.

        Tests can force eviction by setting ``rss_budget_mb=0`` or by
        constructing the Provider with ``rss_budget_mb=0``.
        """
        if self.retain_all or self._group_order is None:
            return []
        if self.rss_budget_mb is not None:
            current_mb = self.rss_estimate_mb()
            if current_mb <= self.rss_budget_mb:
                return []
        try:
            after_idx = self._group_order[after]
        except KeyError:
            # Unknown group token → conservative: don't evict.
            return []
        evicted: list[str] = []
        for name in list(self._frames.keys()):
            ln = self._last_needed.get(name)
            if ln is None:
                continue  # frame not declared in any READS — keep
            _, idx = ln
            if idx <= after_idx:
                del self._frames[name]
                # Phase 6a — the source tag is per-frame metadata, so it
                # must vacate when the frame does.
                self._sources.pop(name, None)
                self._evicted[name] = ln[0]
                evicted.append(name)
        return evicted

    def rss_estimate_mb(self) -> float:
        """Sum of ``frame.estimated_size()`` across the live cache, in
        megabytes.  Used as the threshold gate for
        :meth:`release_unused`.

        ``polars`` tracks frame sizes cheaply (no full walk required).
        Empty / not-yet-materialised frames contribute 0.
        """
        total = 0.0
        for frame in self._frames.values():
            try:
                total += float(frame.estimated_size())
            except Exception:  # pragma: no cover — defensive
                # If polars ever changes the API, fall back to 0 rather
                # than crash the budget check.
                pass
        return total / (1024.0 * 1024.0)

    def is_evicted(self, name: str) -> bool:
        """Return True iff *name* has been evicted by
        :meth:`release_unused`.

        Test helper; production callers use :meth:`get` and let
        :class:`EvictedFrameError` surface naturally.
        """
        return _strip_csv(name) in self._evicted

    def reset_lifetimes(self) -> None:
        """Clear the registered READS, item-group order, lifetime map,
        and evicted-frame markers.

        Use this when transitioning between LP-build phases (e.g.
        between sub-solves in a cascade) so each phase computes its own
        lifetime map.  Does NOT touch the actual frame cache.
        """
        self._reads = {}
        self._handler_groups = {}
        self._group_order = None
        self._last_needed = {}
        self._evicted = {}

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
