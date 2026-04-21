"""Cross-scenario axis-bounds manifest for the result viewer.

Background
----------
:class:`~flextool.plot_outputs.plan.PlotPlan` records a per-scenario
``subplot_y_ranges`` list so the y-axis stays stable while the user
scrolls the time slider **within** a scenario.  Switching scenarios in
single-viewer mode loads a different plan whose y-ranges are computed
from that scenario alone, so the axis can still jump.

This module produces a side-car manifest, written once per batch run at
``<project_path>/output_parquet/_shared/axis_bounds.json``, that stores
each scenario's per-subplot y-ranges.  The viewer takes the union at
read time, filtered to the scenarios the user has currently selected —
so checking/unchecking scenarios in the executed list naturally adjusts
the shared axis without a manifest rewrite.

Schema
------

.. code-block:: json

    {
      "<result_key>": {
        "<sub_config>": {
          "<subplot_key>": {
            "<scenario_name>": [min, max],
            ...
          }
        }
      }
    }

``<subplot_key>`` is chosen as the subplot *title*.  See the comment on
:meth:`ManifestAccumulator.add_plan` for the rationale and the fallback
used for untitled (single-subplot) plans.

Bar-chart plans are intentionally skipped — bars don't use time-scrolling
so cross-scenario y-axis stability isn't a concern for them.

Atomicity
---------
:meth:`ManifestAccumulator.write` serialises via a temp-file + rename so
concurrent readers (e.g. the viewer polling the file) never observe a
half-written JSON document.

Backward compatibility
----------------------
An older version of this module stored bounds as 2-element lists directly
under the subplot title (``{"subplot": [min, max]}``).  Such entries are
silently ignored on load — the next batch run rewrites the file in the
new per-scenario schema.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flextool.plot_outputs.plan import PlotPlan


logger = logging.getLogger(__name__)


# Sentinel for subplots with no title (single-subplot plans where no
# subplot levels are present).  These plans always have exactly one
# subplot, so a single sentinel per (result_key, sub_config) is safe.
_UNTITLED_KEY = "<untitled>"


class ManifestAccumulator:
    """Collects per-subplot y-axis bounds across scenarios.

    One accumulator instance lives for the entire batch run.  The caller
    invokes :meth:`add_plan` with a ``scenario_name`` so the per-scenario
    slice is recorded distinctly.  Subsequent runs of the same scenario
    replace that slice (rather than union it), making the newest
    computation authoritative for that scenario.

    The accumulator seeds from any existing on-disk manifest so that
    re-running a single scenario's batch doesn't wipe out other
    scenarios' entries.

    The target path is
    ``<project_path>/output_parquet/_shared/axis_bounds.json``.  The
    ``_shared`` directory is created on :meth:`write`.
    """

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self._shared_dir = self.project_path / "output_parquet" / "_shared"
        self._manifest_path = self._shared_dir / "axis_bounds.json"
        # Four-level nested dict:
        #   result_key -> sub_config -> subplot_key -> scenario_name -> [lo, hi]
        self._data: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        # Scenarios this accumulator has seen during the current batch.
        # Any previously-stored entries for these scenarios are cleared on
        # first add so the new plans fully replace the old slice.
        self._replaced_scenarios: set[str] = set()
        # Seed accumulator with any bounds already on disk so re-running a
        # single scenario doesn't wipe the others' slices.
        self._load_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_plan(
        self,
        result_key: str,
        sub_config: str,
        plan: "PlotPlan",
        scenario_name: str,
    ) -> None:
        """Record one plan's subplot_y_ranges under *scenario_name*.

        Subplot keying: we use the subplot *title* (first element of each
        ``effective_plot_specs`` tuple).  Rationale:

        - ``title`` is a plain string or ``None`` — directly JSON-safe.
        - When the same entity is present in two scenarios, the title is
          identical, so the reader's union picks up both ranges.
        - When one scenario drops an entity (e.g. zero-filtered), its
          subplot simply doesn't contribute — the other scenario's
          bounds still govern, which is the desired behaviour.
        - Subplot *indices* would misalign in exactly that dropped-entity
          case, so we avoid index-based keying.
        - ``None`` titles (single-subplot plans without subplot levels)
          are mapped to a single ``<untitled>`` sentinel, which is safe
          because such plans always have exactly one subplot.

        Bar-chart plans are skipped — see module docstring.

        Replacement semantics: the first call for a given
        ``scenario_name`` within this accumulator's lifetime clears any
        previously-stored entries for that scenario across *all*
        (result_key, sub_config, subplot) triples.  Subsequent calls for
        the same scenario simply add/overwrite entries for the specific
        subplots touched.  This way a re-run authoritative picture of the
        scenario doesn't leave stale subplot entries lingering from a
        previous run.
        """
        if plan.chart_type == 'bar':
            return
        if not plan.subplot_y_ranges:
            return
        if not scenario_name:
            logger.warning(
                "ManifestAccumulator.add_plan called with empty scenario_name; "
                "dropping entries for %s/%s",
                result_key, sub_config,
            )
            return

        # effective_plot_specs and subplot_y_ranges are parallel lists of
        # the same length (see _compute_time_plan).  Guard anyway to avoid
        # silent misalignment if that invariant ever breaks.
        specs = plan.effective_plot_specs
        ranges = plan.subplot_y_ranges
        if len(specs) != len(ranges):
            logger.warning(
                "PlotPlan %s/%s: effective_plot_specs (%d) and "
                "subplot_y_ranges (%d) length mismatch — skipping",
                result_key, sub_config, len(specs), len(ranges),
            )
            return

        # On first touch of this scenario in this accumulator lifetime,
        # drop any disk-seeded entries that belong to it — the current
        # batch is now the authoritative picture of the scenario.
        if scenario_name not in self._replaced_scenarios:
            self._purge_scenario_in_memory(scenario_name)
            self._replaced_scenarios.add(scenario_name)

        result_entry = self._data.setdefault(result_key, {})
        sub_entry = result_entry.setdefault(sub_config, {})

        for (title, _selector), (lo, hi) in zip(specs, ranges):
            key = _UNTITLED_KEY if title is None else str(title)
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except (TypeError, ValueError):
                continue
            subplot_entry = sub_entry.setdefault(key, {})
            subplot_entry[scenario_name] = [lo_f, hi_f]

    def write(self) -> None:
        """Atomically write the accumulated manifest to disk.

        Creates the ``_shared`` directory if missing.  Writes to a temp
        file in the same directory, then ``os.replace``-renames it into
        place so concurrent readers never see a partial document.
        """
        if not self._data:
            # Nothing to write — don't create an empty file.
            return
        _atomic_write_manifest(self._shared_dir, self._manifest_path, self._data)

    # ------------------------------------------------------------------
    # Introspection helpers (primarily for tests)
    # ------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def data(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
        """Read-only view of the accumulated data.  Don't mutate."""
        return self._data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _purge_scenario_in_memory(self, scenario_name: str) -> None:
        """Drop *scenario_name*'s entries from ``self._data`` in place.

        Removes empty subplot / sub_config / result_key entries as it
        prunes.  Used when a scenario's first ``add_plan`` call arrives
        and the disk-seeded data for that scenario needs to be cleared
        before the new batch's values take its place.
        """
        for result_key in list(self._data.keys()):
            sub_map = self._data[result_key]
            for sub_config in list(sub_map.keys()):
                subplot_map = sub_map[sub_config]
                for subplot_key in list(subplot_map.keys()):
                    scen_map = subplot_map[subplot_key]
                    if scenario_name in scen_map:
                        del scen_map[scenario_name]
                    if not scen_map:
                        del subplot_map[subplot_key]
                if not subplot_map:
                    del sub_map[sub_config]
            if not sub_map:
                del self._data[result_key]

    def _load_existing(self) -> None:
        """Seed the accumulator from an on-disk manifest if present.

        Tolerant of malformed files and the legacy flat schema — anything
        that doesn't fit the four-level per-scenario layout is silently
        dropped.  A corrupt previous write can't permanently break the
        next batch run; a legacy manifest is simply ignored and will be
        rewritten in the new schema on the next :meth:`write`.
        """
        loaded = load_axis_bounds_manifest(self.project_path)
        if loaded is None:
            return
        saw_legacy = False
        # Shape-validate as we copy in so downstream code can trust the structure.
        for rk, rk_val in loaded.items():
            if not isinstance(rk_val, dict):
                continue
            rk_entry: dict[str, dict[str, dict[str, list[float]]]] = {}
            for sc, sc_val in rk_val.items():
                if not isinstance(sc_val, dict):
                    continue
                sc_entry: dict[str, dict[str, list[float]]] = {}
                for sp, sp_val in sc_val.items():
                    # Legacy schema: sp_val is a 2-element list [lo, hi].
                    # Ignore (and flag) — the per-scenario attribution is
                    # missing, so there's no safe way to retain it.
                    if isinstance(sp_val, (list, tuple)):
                        saw_legacy = True
                        continue
                    if not isinstance(sp_val, dict):
                        continue
                    scen_entry: dict[str, list[float]] = {}
                    for scenario_name, bounds in sp_val.items():
                        if not isinstance(scenario_name, str) or not scenario_name:
                            continue
                        if (
                            isinstance(bounds, (list, tuple))
                            and len(bounds) == 2
                        ):
                            try:
                                scen_entry[scenario_name] = [
                                    float(bounds[0]), float(bounds[1]),
                                ]
                            except (TypeError, ValueError):
                                continue
                    if scen_entry:
                        sc_entry[str(sp)] = scen_entry
                if sc_entry:
                    rk_entry[str(sc)] = sc_entry
            if rk_entry:
                self._data[str(rk)] = rk_entry
        if saw_legacy:
            logger.warning(
                "Shared axis manifest %s uses the legacy (pre-per-scenario) "
                "schema; those entries are being dropped and will be "
                "rewritten on the next batch run.",
                self._manifest_path,
            )


# ---------------------------------------------------------------------------
# Reader API (used by the result viewer — Chunk C)
# ---------------------------------------------------------------------------

def load_axis_bounds_manifest(project_path: Path) -> dict | None:
    """Load the shared axis-bounds manifest for *project_path*.

    Returns the parsed JSON object (a nested ``dict``) or ``None`` when the
    manifest does not exist, cannot be read, or is malformed.  The function
    never raises — a corrupt or missing manifest simply means "no
    cross-scenario axis data available", and the viewer falls back to the
    per-plan y-ranges.

    The expected manifest path is
    ``<project_path>/output_parquet/_shared/axis_bounds.json``.
    """
    manifest_path = (
        Path(project_path) / "output_parquet" / "_shared" / "axis_bounds.json"
    )
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Ignoring unreadable shared manifest %s: %s",
            manifest_path, exc,
        )
        return None
    if not isinstance(loaded, dict):
        logger.warning(
            "Shared manifest %s is not a JSON object — ignoring",
            manifest_path,
        )
        return None
    return loaded


def apply_manifest_to_plan(
    plan: "PlotPlan",
    manifest: dict | None,
    result_key: str,
    sub_config: str,
    active_scenarios: set[str] | None = None,
) -> bool:
    """Override ``plan.subplot_y_ranges`` with values from *manifest*.

    Mutates *plan* in-place.  Returns ``True`` when at least one subplot
    range was replaced, ``False`` when nothing changed (missing manifest,
    bar plan, no matching entry, filtered-out scenarios, etc.).

    *active_scenarios* controls which scenarios contribute to the union:

    - ``None`` — union over *all* scenarios present in the manifest (the
      full cross-scenario fallback).
    - A non-empty ``set`` — union only over the scenarios in the set.
    - An empty ``set`` — no scenarios to union from; the call becomes a
      no-op (the plan's per-scenario y-ranges remain in effect).

    Subplot keying: for each ``(title, _)`` in
    ``plan.effective_plot_specs``, if the manifest has a per-scenario map
    keyed by that title (with ``None`` mapped to ``_UNTITLED_KEY`` exactly
    as the writer does), the entries for the active scenarios are unioned
    and written to ``plan.subplot_y_ranges`` at the matching index.
    Subplots without a manifest entry keep their per-scenario range.

    Bar-chart plans are skipped (matching the writer).  Legacy flat-schema
    entries (``list``/``tuple`` instead of per-scenario ``dict``) are
    ignored — the next batch run rewrites them.  Extra entries in the
    manifest that don't correspond to any subplot in *plan* are ignored.
    """
    if manifest is None or plan is None:
        return False
    if plan.chart_type == "bar":
        return False
    if not plan.effective_plot_specs:
        return False
    if active_scenarios is not None and len(active_scenarios) == 0:
        return False
    sub_entry = manifest.get(result_key)
    if not isinstance(sub_entry, dict):
        return False
    title_map = sub_entry.get(sub_config)
    if not isinstance(title_map, dict):
        return False

    # Pad subplot_y_ranges if shorter than effective_plot_specs so we can
    # overwrite any index.  (In practice they are parallel lists, but the
    # plan file format is JSON — defend against asymmetric loads.)
    ranges: list[tuple[float, float]] = [
        tuple(r) for r in plan.subplot_y_ranges
    ]
    while len(ranges) < len(plan.effective_plot_specs):
        ranges.append((0.0, 0.0))

    changed = False
    for i, (title, _selector) in enumerate(plan.effective_plot_specs):
        key = _UNTITLED_KEY if title is None else str(title)
        scen_map = title_map.get(key)
        if not isinstance(scen_map, dict) or not scen_map:
            # Missing, legacy (list/tuple), or empty — no override.
            continue
        new_range = _union_scenarios(scen_map, active_scenarios)
        if new_range is None:
            continue
        if ranges[i] != new_range:
            ranges[i] = new_range
            changed = True

    if changed:
        plan.subplot_y_ranges = ranges
    return changed


def _union_scenarios(
    scen_map: dict, active_scenarios: set[str] | None,
) -> tuple[float, float] | None:
    """Union per-scenario bounds, filtering to *active_scenarios*.

    Returns ``None`` when no valid bounds remain after filtering.
    """
    los: list[float] = []
    his: list[float] = []
    for scenario_name, bounds in scen_map.items():
        if active_scenarios is not None and scenario_name not in active_scenarios:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        try:
            lo = float(bounds[0])
            hi = float(bounds[1])
        except (TypeError, ValueError):
            continue
        los.append(lo)
        his.append(hi)
    if not los:
        return None
    return (min(los), max(his))


# ---------------------------------------------------------------------------
# Scenario removal helper
# ---------------------------------------------------------------------------

def remove_scenario_from_manifest(
    project_path: Path, scenario_name: str,
) -> bool:
    """Strip all entries for *scenario_name* from the on-disk manifest.

    Atomically rewrites ``<project_path>/output_parquet/_shared/axis_bounds.json``
    with the scenario's slice removed.  Empty subplot / sub_config /
    result_key entries are pruned.  Returns ``True`` if the file was
    modified (including being deleted when empty), ``False`` when the
    manifest is missing, unreadable, or already free of the scenario.

    Safe to call when the manifest doesn't exist — it just returns
    ``False``.  Intended for callers that delete a scenario's output so
    its slice doesn't continue to pull y-range data in the viewer.
    """
    if not scenario_name:
        return False
    shared_dir = Path(project_path) / "output_parquet" / "_shared"
    manifest_path = shared_dir / "axis_bounds.json"
    if not manifest_path.is_file():
        return False

    loaded = load_axis_bounds_manifest(project_path)
    if not isinstance(loaded, dict):
        return False

    changed = False
    for result_key in list(loaded.keys()):
        sub_map = loaded.get(result_key)
        if not isinstance(sub_map, dict):
            continue
        for sub_config in list(sub_map.keys()):
            subplot_map = sub_map.get(sub_config)
            if not isinstance(subplot_map, dict):
                continue
            for subplot_key in list(subplot_map.keys()):
                scen_map = subplot_map.get(subplot_key)
                # Legacy flat entries have no scenario dimension; skip.
                if not isinstance(scen_map, dict):
                    continue
                if scenario_name in scen_map:
                    del scen_map[scenario_name]
                    changed = True
                if not scen_map:
                    del subplot_map[subplot_key]
            if not subplot_map:
                del sub_map[sub_config]
        if not sub_map:
            del loaded[result_key]

    if not changed:
        return False

    if not loaded:
        # Nothing left — drop the file entirely.
        try:
            manifest_path.unlink()
        except OSError as exc:
            logger.warning(
                "Failed to delete now-empty manifest %s: %s",
                manifest_path, exc,
            )
            return False
        return True

    _atomic_write_manifest(shared_dir, manifest_path, loaded)
    return True


# ---------------------------------------------------------------------------
# Shared atomic writer
# ---------------------------------------------------------------------------

def _atomic_write_manifest(shared_dir: Path, manifest_path: Path, data: dict) -> None:
    """Temp-file + rename write shared by the accumulator and remove helper."""
    try:
        shared_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Failed to create shared manifest directory %s: %s",
            shared_dir, exc,
        )
        return

    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(shared_dir),
            prefix="axis_bounds_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path)
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            tmp_path.replace(manifest_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    except Exception as exc:
        logger.warning(
            "Failed to write shared axis-bounds manifest %s: %s",
            manifest_path, exc,
        )
