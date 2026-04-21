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
the **union** of subplot y-ranges across every scenario processed by the
batch.  Chunk C of the refactor teaches the viewer to consult this
manifest before falling back to the per-plan ranges, giving cross-scenario
axis stability.

Schema
------

.. code-block:: json

    {
      "<result_key>": {
        "<sub_config>": {
          "<subplot_key>": [min, max]
        }
      }
    }

``<subplot_key>`` is chosen as the subplot *title*.  See the module-level
comment on :meth:`ManifestAccumulator.add_plan` for the rationale and
the fallback used for untitled (single-subplot) plans.

Bar-chart plans are intentionally skipped — bars don't use time-scrolling
so cross-scenario y-axis stability isn't a concern for them.

Atomicity
---------
:meth:`ManifestAccumulator.write` serialises via a temp-file + rename so
concurrent readers (e.g. the viewer polling the file) never observe a
half-written JSON document.
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

    One accumulator instance lives for the entire batch run.
    :meth:`add_plan` is called once per ``save_plot_plan`` call; the
    bounds are unioned (element-wise min/max) against whatever is
    already stored for that ``(result_key, sub_config, subplot_key)``
    triple — including bounds loaded from an existing on-disk manifest,
    so re-running one scenario doesn't wipe the others.

    The target path is
    ``<project_path>/output_parquet/_shared/axis_bounds.json``.  The
    ``_shared`` directory is created on :meth:`write`.
    """

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path)
        self._shared_dir = self.project_path / "output_parquet" / "_shared"
        self._manifest_path = self._shared_dir / "axis_bounds.json"
        # Three-level nested dict: result_key -> sub_config -> subplot_key -> [lo, hi]
        self._data: dict[str, dict[str, dict[str, list[float]]]] = {}
        # Seed accumulator with any bounds already on disk so re-running a
        # single scenario unions into the full cross-scenario view rather
        # than replacing it.
        self._load_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_plan(
        self, result_key: str, sub_config: str, plan: "PlotPlan",
    ) -> None:
        """Merge one plan's subplot_y_ranges into the accumulator.

        Subplot keying: we use the subplot *title* (first element of each
        ``effective_plot_specs`` tuple).  Rationale:

        - ``title`` is a plain string or ``None`` — directly JSON-safe.
        - When the same entity is present in two scenarios, the title is
          identical, so their ranges union correctly.
        - When one scenario drops an entity (e.g. zero-filtered), its
          subplot simply doesn't contribute — the other scenario's
          bounds still govern, which is the desired behaviour.
        - Subplot *indices* would misalign in exactly that dropped-entity
          case, so we avoid index-based keying.
        - ``None`` titles (single-subplot plans without subplot levels)
          are mapped to a single ``<untitled>`` sentinel, which is safe
          because such plans always have exactly one subplot.

        Bar-chart plans are skipped — see module docstring.
        """
        if plan.chart_type == 'bar':
            return
        if not plan.subplot_y_ranges:
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

        result_entry = self._data.setdefault(result_key, {})
        sub_entry = result_entry.setdefault(sub_config, {})

        for (title, _selector), (lo, hi) in zip(specs, ranges):
            key = _UNTITLED_KEY if title is None else str(title)
            try:
                lo_f = float(lo)
                hi_f = float(hi)
            except (TypeError, ValueError):
                continue
            existing = sub_entry.get(key)
            if existing is None:
                sub_entry[key] = [lo_f, hi_f]
            else:
                sub_entry[key] = [
                    min(existing[0], lo_f),
                    max(existing[1], hi_f),
                ]

    def write(self) -> None:
        """Atomically write the accumulated manifest to disk.

        Creates the ``_shared`` directory if missing.  Writes to a temp
        file in the same directory, then ``os.replace``-renames it into
        place so concurrent readers never see a partial document.
        """
        if not self._data:
            # Nothing to write — don't create an empty file.
            return
        try:
            self._shared_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to create shared manifest directory %s: %s",
                self._shared_dir, exc,
            )
            return

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._shared_dir),
                prefix="axis_bounds_",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_path)
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, sort_keys=True)
                tmp_path.replace(self._manifest_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        except Exception as exc:
            logger.warning(
                "Failed to write shared axis-bounds manifest %s: %s",
                self._manifest_path, exc,
            )

    # ------------------------------------------------------------------
    # Introspection helpers (primarily for tests)
    # ------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def data(self) -> dict[str, dict[str, dict[str, list[float]]]]:
        """Read-only view of the accumulated data.  Don't mutate."""
        return self._data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        """Seed the accumulator from an on-disk manifest if present.

        Tolerant of malformed files — any issue is logged and the
        accumulator starts fresh, so a corrupt previous write can't
        permanently break the next batch run.
        """
        loaded = load_axis_bounds_manifest(self.project_path)
        if loaded is None:
            return
        # Shape-validate as we copy in so downstream code can trust the structure.
        for rk, rk_val in loaded.items():
            if not isinstance(rk_val, dict):
                continue
            rk_entry: dict[str, dict[str, list[float]]] = {}
            for sc, sc_val in rk_val.items():
                if not isinstance(sc_val, dict):
                    continue
                sc_entry: dict[str, list[float]] = {}
                for sp, bounds in sc_val.items():
                    if (
                        isinstance(bounds, (list, tuple))
                        and len(bounds) == 2
                    ):
                        try:
                            sc_entry[str(sp)] = [float(bounds[0]), float(bounds[1])]
                        except (TypeError, ValueError):
                            continue
                if sc_entry:
                    rk_entry[str(sc)] = sc_entry
            if rk_entry:
                self._data[str(rk)] = rk_entry


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
) -> bool:
    """Override ``plan.subplot_y_ranges`` with values from *manifest*.

    Mutates *plan* in-place.  Returns ``True`` when at least one subplot
    range was replaced, ``False`` when nothing changed (missing manifest,
    bar plan, no matching entry, etc.).

    The override is per-subplot: for each ``(title, _)`` in
    ``plan.effective_plot_specs``, if the manifest has a range keyed by
    that title (with ``None`` mapped to ``_UNTITLED_KEY`` exactly as the
    writer does), the corresponding entry in ``plan.subplot_y_ranges`` is
    replaced.  Subplots without a manifest entry keep their per-scenario
    range, which is the desired fallback behaviour.

    Bar-chart plans are skipped (matching the writer).  Extra entries in
    the manifest that don't correspond to any subplot in *plan* are
    ignored.
    """
    if manifest is None or plan is None:
        return False
    if plan.chart_type == "bar":
        return False
    if not plan.effective_plot_specs:
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
        bounds = title_map.get(key)
        if bounds is None:
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue
        try:
            lo = float(bounds[0])
            hi = float(bounds[1])
        except (TypeError, ValueError):
            continue
        new_range = (lo, hi)
        if ranges[i] != new_range:
            ranges[i] = new_range
            changed = True

    if changed:
        plan.subplot_y_ranges = ranges
    return changed
