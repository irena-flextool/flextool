"""User-facing scaling diagnostic report — polars / in-memory.

After every solve, ``_orchestration.run_model`` calls
:func:`write_scaling_report` to emit
``<work_folder>/solve_data/scaling_report.txt`` containing:

1. Header with solve name, timestamp, HiGHS version.
2. Scaling decisions taken (from :class:`scaling.ScaleTable`).
3. Coefficient-family ranges (log10 spread, absolute min/max).
4. Bimodal detection — flags coefficient families with bimodal shape.
5. Composite-scale-mismatch detector — the load-bearing user diagnostic
   for the LP-scaling project: finds directly-connected entities whose
   unitsizes span more than 3 orders of magnitude.  Reports the top-10
   offending pairs and prints the recommendation text locked by the
   2026-04-22 design discussion.
6. Near-duplicate parameter clusters (skipped — no CSV dir available;
   section is omitted gracefully).
7. Escape-slack activity — any slack total above a tolerance is reported
   along with top-5 offending cells.
8. HiGHS matrix-range summary derived from the live ``polar_high.Solution``
   object (or omitted gracefully when ``solution`` is ``None``).
9. Summary line — well-scaled / acceptably / poorly-scaled.

ASCII-only output (no unicode / emoji) for easy diffing.

Inputs:

* ``flex_data`` — FlexData bag (topology source for the composite-
  mismatch detector).
* ``solution`` — polar_high.Solution (HiGHS matrix ranges come from
  ``solution.highs.getLp()``).
"""

from __future__ import annotations

import io
import logging
import math
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from flextool.engine_polars.scaling import FamilyStats, ScaleTable

if TYPE_CHECKING:
    from flextool.engine_polars.input import FlexData
    from polar_high import Solution


# ---------------------------------------------------------------------------
# Thresholds — identical to the original module.
# ---------------------------------------------------------------------------

FAMILY_SPREAD_WARN_DECADES = 5.0
"""Within-family log10 spread that triggers a '!' warning mark."""

BIMODAL_GAP_DECADES = 2.0
"""Minimum log10 gap between two clusters to flag a family as bimodal."""

BIMODAL_MIN_CLUSTER_SHARE = 0.10
"""Each of the two clusters must hold this fraction of the values."""

COMPOSITE_MISMATCH_LOG10_THRESHOLD = 3.0
"""Unitsize ratio (log10) above this flags a composite-scale mismatch."""

MATRIX_RANGE_WARN_DECADES = 6.0
"""Matrix coefficient range beyond this (log10) triggers a matrix warning."""

COST_RANGE_WARN_DECADES = 8.0
"""Cost coefficient range beyond this (log10) triggers a cost warning."""

ESCAPE_TIER_TOLERANCE = 1e-8
"""Absolute threshold below which slack activity is treated as zero."""


# ---------------------------------------------------------------------------
# Bimodal detection  (identical to original)
# ---------------------------------------------------------------------------


@dataclass
class BimodalSplit:
    """Describe a bimodal split of a coefficient family."""

    lower_center_log10: float
    upper_center_log10: float
    gap_decades: float
    n_lower: int
    n_upper: int
    lower_share: float
    upper_share: float
    split_log10: float


def detect_bimodal(log10_values: list[float]) -> Optional[BimodalSplit]:
    """Return a :class:`BimodalSplit` if *log10_values* is bimodal, else None.

    Rule (locked in the Agent 10 spec):

    1. Sort ascending.
    2. Find the largest adjacent gap.
    3. If the gap exceeds :data:`BIMODAL_GAP_DECADES` *and* both sides
       hold more than :data:`BIMODAL_MIN_CLUSTER_SHARE` of the values,
       return the split; otherwise ``None``.
    """
    if len(log10_values) < 4:
        return None
    xs = sorted(log10_values)
    best_gap = 0.0
    best_idx = -1
    for i in range(len(xs) - 1):
        gap = xs[i + 1] - xs[i]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    if best_idx < 0 or best_gap <= BIMODAL_GAP_DECADES:
        return None
    n = len(xs)
    n_lower = best_idx + 1
    n_upper = n - n_lower
    share_lower = n_lower / n
    share_upper = n_upper / n
    if share_lower < BIMODAL_MIN_CLUSTER_SHARE:
        return None
    if share_upper < BIMODAL_MIN_CLUSTER_SHARE:
        return None
    if n_lower < 2 or n_upper < 2:
        return None
    lower = xs[:n_lower]
    upper = xs[n_lower:]
    return BimodalSplit(
        lower_center_log10=lower[len(lower) // 2],
        upper_center_log10=upper[len(upper) // 2],
        gap_decades=best_gap,
        n_lower=n_lower,
        n_upper=n_upper,
        lower_share=share_lower,
        upper_share=share_upper,
        split_log10=(xs[best_idx] + xs[best_idx + 1]) / 2.0,
    )


# ---------------------------------------------------------------------------
# Composite-scale-mismatch detector — FlexData variant
# ---------------------------------------------------------------------------


@dataclass
class MismatchPair:
    """A directly-connected (process -> node) pair with a large unitsize ratio."""

    process: str
    node: str
    role: str  # "source" or "sink"
    process_unitsize: float
    node_unitsize: float
    ratio: float  # always >= 1
    small_entity: str
    small_size: float
    large_entity: str
    large_size: float


def _build_unitsize_map(flex_data: "FlexData") -> dict[str, float]:
    """Build a ``{entity_name: unitsize}`` mapping from ``flex_data.p_all_entity_unitsize``.

    Returns an empty dict when the param is absent or empty.
    """
    p_unitsize = flex_data.p_all_entity_unitsize
    if p_unitsize is None:
        return {}
    try:
        frame = p_unitsize.frame
    except Exception:
        return {}
    if frame is None or frame.is_empty():
        return {}
    if "e" not in frame.columns or "value" not in frame.columns:
        return {}
    p_col = frame["e"].to_list()
    v_col = frame["value"].to_list()
    result: dict[str, float] = {}
    for name, v in zip(p_col, v_col):
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv > 0.0:
            result[name] = fv
    return result


def _build_process_node_pairs(
    flex_data: "FlexData",
) -> list[tuple[str, str, str]]:
    """Return ``(process, node, role)`` pairs from the canonical topology DataFrames.

    Draws from ``flex_data.process_source_canonical`` (schema: ``p``, ``source``)
    and ``flex_data.process_sink_canonical`` (schema: ``p``, ``sink``).
    Missing DataFrames contribute nothing.
    """
    pairs: list[tuple[str, str, str]] = []

    src_df = flex_data.process_source_canonical
    if src_df is not None and src_df.height > 0:
        for p, node in zip(src_df["p"].to_list(), src_df["source"].to_list()):
            if p and node:
                pairs.append((str(p), str(node), "source"))

    snk_df = flex_data.process_sink_canonical
    if snk_df is not None and snk_df.height > 0:
        for p, node in zip(snk_df["p"].to_list(), snk_df["sink"].to_list()):
            if p and node:
                pairs.append((str(p), str(node), "sink"))

    return pairs


def find_composite_mismatches(
    flex_data: "FlexData",
    threshold_log10: float = COMPOSITE_MISMATCH_LOG10_THRESHOLD,
) -> list[MismatchPair]:
    """Find directly-connected entity pairs spanning > *threshold_log10* decades.

    Reads topology and unitsizes from *flex_data*.  Returns list sorted
    by ratio descending.
    """
    unitsizes = _build_unitsize_map(flex_data)
    if not unitsizes:
        return []
    pairs = _build_process_node_pairs(flex_data)
    if not pairs:
        return []

    # Collect process entries per node.
    per_node: dict[str, list[tuple[str, str]]] = {}
    for proc, node, role in pairs:
        per_node.setdefault(node, []).append((proc, role))

    seen: set[tuple[str, str, str]] = set()
    mismatches: list[MismatchPair] = []
    for node, procs in per_node.items():
        # Entity cloud = node itself + all connected processes.
        cloud: list[tuple[str, str]] = [(node, "node")]
        cloud.extend(procs)
        cloud_with_sz: list[tuple[str, str, float]] = [
            (ename, role, unitsizes[ename])
            for ename, role in cloud
            if ename in unitsizes
        ]
        if len(cloud_with_sz) < 2:
            continue
        for i in range(len(cloud_with_sz)):
            for j in range(i + 1, len(cloud_with_sz)):
                e1, r1, s1 = cloud_with_sz[i]
                e2, r2, s2 = cloud_with_sz[j]
                if e1 == e2 or s1 <= 0 or s2 <= 0:
                    continue
                if s1 >= s2:
                    large_e, large_s = e1, s1
                    small_e, small_s = e2, s2
                else:
                    large_e, large_s = e2, s2
                    small_e, small_s = e1, s1
                ratio = large_s / small_s
                if math.log10(ratio) < threshold_log10:
                    continue
                key = (node, small_e, large_e)
                if key in seen:
                    continue
                seen.add(key)
                if e1 == node or e2 == node:
                    proc_e = e2 if e1 == node else e1
                    proc_role = r2 if e1 == node else r1
                    proc_size = s2 if e1 == node else s1
                    node_size = s1 if e1 == node else s2
                    mismatches.append(
                        MismatchPair(
                            process=proc_e,
                            node=node,
                            role=proc_role,
                            process_unitsize=proc_size,
                            node_unitsize=node_size,
                            ratio=ratio,
                            small_entity=small_e,
                            small_size=small_s,
                            large_entity=large_e,
                            large_size=large_s,
                        )
                    )
                else:
                    small_role = r1 if small_e == e1 else r2
                    mismatches.append(
                        MismatchPair(
                            process=small_e,
                            node=node,
                            role=small_role,
                            process_unitsize=small_s,
                            node_unitsize=large_s,
                            ratio=ratio,
                            small_entity=small_e,
                            small_size=small_s,
                            large_entity=large_e,
                            large_size=large_s,
                        )
                    )
    mismatches.sort(key=lambda m: m.ratio, reverse=True)
    return mismatches


MISMATCH_RECOMMENDATION = """\
Composite-scale mismatch detected: node {node} connects entities of wildly different scales
(e.g., {small_entity} at {small_size} MW and {large_entity} at {large_size} MW \
— a ratio of {ratio}:1).
No linear scaling can eliminate matrix coefficient spread arising from such connections.
Recommendations:
  (1) Aggregate the small-side units: e.g., use 1000 buildings instead of 1 to match
      the order of magnitude of the connected system. Accept that aggregation introduces
      some inaccuracy in the small-scale dynamics.
  (2) Run the two subsystems as sequential models: optimise the large system first, then
      use its results as boundary conditions for a detailed small-system run (invest →
      dispatch handoff, or whichever staging fits your use case)."""


def _format_mismatch_recommendation(mismatch: MismatchPair) -> str:
    """Render the locked recommendation text with ASCII substitutions."""
    s = MISMATCH_RECOMMENDATION.format(
        node=mismatch.node,
        small_entity=mismatch.small_entity,
        small_size=_fmt_num(mismatch.small_size),
        large_entity=mismatch.large_entity,
        large_size=_fmt_num(mismatch.large_size),
        ratio=_fmt_ratio(mismatch.ratio),
    )
    s = s.replace("—", "--").replace("→", "->")
    return s


def _fmt_num(v: float) -> str:
    if not math.isfinite(v):
        return "?"
    if v == 0.0:
        return "0"
    if v == int(v) and abs(v) < 1e7:
        return str(int(v))
    return f"{v:g}"


def _fmt_ratio(r: float) -> str:
    if not math.isfinite(r):
        return "?"
    if r >= 1000:
        return f"{r:.3g}"
    return f"{r:.1f}"


# ---------------------------------------------------------------------------
# HiGHS matrix-range extraction from Solution object
# ---------------------------------------------------------------------------


def _extract_highs_matrix_ranges(
    solution: "Solution",
) -> dict[str, Any]:
    """Extract coefficient ranges and version from a polar_high Solution.

    Returns a dict with keys matching the original ``parse_highs_log`` output:
    ``version``, ``matrix_range``, ``cost_range``, ``bound_range``, ``rhs_range``.

    ``cost_range`` is derived from LP column costs; ``bound_range`` and
    ``rhs_range`` are derived from LP row/column bounds.  Returns ``None``
    for any range that cannot be computed.
    """
    out: dict[str, Any] = {
        "version": None,
        "matrix_range": None,
        "cost_range": None,
        "bound_range": None,
        "rhs_range": None,
    }
    if solution is None:
        return out
    highs = solution.highs
    if highs is None:
        return out

    # Try to get the HiGHS version string.
    try:
        import highspy
        out["version"] = getattr(highspy, "__version__", None)
    except ImportError:
        pass

    # Read the LP object from the Highs instance.
    try:
        lp = highs.getLp()
    except Exception:
        return out

    # Matrix coefficient range.
    try:
        import numpy as np
        raw_vals = lp.a_matrix_.value_
        mat_arr = np.abs(np.asarray(raw_vals, dtype=np.float64))
        nonzero = mat_arr[mat_arr > 0.0]
        if nonzero.size > 0:
            out["matrix_range"] = (float(nonzero.min()), float(nonzero.max()))
    except Exception:
        pass

    # Cost coefficient range.
    try:
        import numpy as np
        costs = np.abs(np.asarray(lp.col_cost_, dtype=np.float64))
        nz_costs = costs[costs > 0.0]
        if nz_costs.size > 0:
            out["cost_range"] = (float(nz_costs.min()), float(nz_costs.max()))
    except Exception:
        pass

    # Bound range (column bounds — lower and upper, finite only).
    try:
        import numpy as np
        lo_raw = np.asarray(lp.col_lower_, dtype=np.float64)
        hi_raw = np.asarray(lp.col_upper_, dtype=np.float64)
        import highspy as _highspy
        inf = _highspy.kHighsInf
        finite_lo = np.abs(lo_raw[(lo_raw != 0.0) & np.isfinite(lo_raw) & (np.abs(lo_raw) < inf)])
        finite_hi = np.abs(hi_raw[(hi_raw != 0.0) & np.isfinite(hi_raw) & (np.abs(hi_raw) < inf)])
        combined = np.concatenate([finite_lo, finite_hi])
        nz_bounds = combined[combined > 0.0]
        if nz_bounds.size > 0:
            out["bound_range"] = (float(nz_bounds.min()), float(nz_bounds.max()))
    except Exception:
        pass

    # RHS range (row bounds — lower and upper, finite only).
    try:
        import numpy as np
        import highspy as _highspy
        inf = _highspy.kHighsInf
        rlo = np.abs(np.asarray(lp.row_lower_, dtype=np.float64))
        rhi = np.abs(np.asarray(lp.row_upper_, dtype=np.float64))
        finite_rlo = rlo[(rlo > 0.0) & np.isfinite(rlo) & (rlo < inf)]
        finite_rhi = rhi[(rhi > 0.0) & np.isfinite(rhi) & (rhi < inf)]
        rhs_combined = np.concatenate([finite_rlo, finite_rhi])
        if rhs_combined.size > 0:
            out["rhs_range"] = (float(rhs_combined.min()), float(rhs_combined.max()))
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Coefficient-source diagnostic — pinpoint which variables / constraints
# produce the smallest and largest matrix / cost / bound / RHS coefficients.
# ---------------------------------------------------------------------------


def _name_at(names: Any, idx: int, prefix: str) -> str:
    """Return the name at *idx* with a safe fallback like ``f"{prefix}_{idx}"``."""
    try:
        if names is None:
            return f"{prefix}_{idx}"
        if idx < len(names):
            n = names[idx]
            if n:
                return str(n)
    except Exception:
        pass
    return f"{prefix}_{idx}"


def _topk_indices(arr: "Any", k: int, *, largest: bool) -> "Any":
    """Return up to *k* indices of the largest (or smallest) values in *arr*.

    Uses :func:`numpy.argpartition` to avoid sorting the full array.  The
    returned indices are sorted by value (descending for ``largest=True``,
    ascending otherwise).  Assumes *arr* is a 1-D numpy array of non-negative
    floats with no zeros (callers are expected to filter beforehand).
    """
    import numpy as np
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, n)
    if largest:
        # Indices of the k largest values (unsorted).
        if k < n:
            part = np.argpartition(-arr, k - 1)[:k]
        else:
            part = np.arange(n)
        # Sort those k by descending value.
        return part[np.argsort(-arr[part])]
    else:
        if k < n:
            part = np.argpartition(arr, k - 1)[:k]
        else:
            part = np.arange(n)
        return part[np.argsort(arr[part])]


def _collect_coefficient_sources(
    highs: Any, top: int = 5
) -> dict[str, Any]:
    """Identify the variables / constraints producing the smallest and
    largest matrix / cost / bound / RHS coefficients in the live HiGHS LP.

    Returns a dict with keys ``matrix_smallest`` / ``matrix_largest``
    (entries: ``(value, col_name, row_name)``), ``cost_smallest`` /
    ``cost_largest`` (entries: ``(value, col_name)``), ``bound_smallest`` /
    ``bound_largest`` (entries: ``(value, col_name, "lower"|"upper")``),
    ``rhs_smallest`` / ``rhs_largest`` (entries: ``(value, row_name,
    "lower"|"upper")``).  When extraction fails or the LP cannot be read,
    returns ``{"error": "<message>"}``.
    """
    out: dict[str, Any] = {
        "matrix_smallest": [],
        "matrix_largest": [],
        "cost_smallest": [],
        "cost_largest": [],
        "bound_smallest": [],
        "bound_largest": [],
        "rhs_smallest": [],
        "rhs_largest": [],
    }
    if highs is None:
        out["error"] = "no live HiGHS solver instance"
        return out

    try:
        import numpy as np
        import highspy as _highspy
        inf = _highspy.kHighsInf
    except Exception as exc:
        out["error"] = f"numpy/highspy unavailable: {exc}"
        return out

    try:
        lp = highs.getLp()
    except Exception as exc:
        out["error"] = f"could not read LP: {exc}"
        return out

    col_names = getattr(lp, "col_names_", None)
    row_names = getattr(lp, "row_names_", None)

    # --- Matrix coefficients (CSC: value_, index_, start_) ---
    try:
        a = lp.a_matrix_
        vals = np.asarray(a.value_, dtype=np.float64)
        rows = np.asarray(a.index_, dtype=np.int64)
        starts = np.asarray(a.start_, dtype=np.int64)
        abs_vals = np.abs(vals)
        nz_mask = abs_vals > 0.0
        nz_idx = np.flatnonzero(nz_mask)
        if nz_idx.size > 0:
            nz_abs = abs_vals[nz_idx]
            small = _topk_indices(nz_abs, top, largest=False)
            large = _topk_indices(nz_abs, top, largest=True)
            # Build a column lookup: for nonzero positions in the CSC value_
            # array, find which column they belong to using starts.
            # `np.searchsorted(starts, k, side="right") - 1` maps nonzero
            # offset k → column index.
            n_cols = max(starts.size - 1, 0)
            for picks, target in (
                (small, out["matrix_smallest"]),
                (large, out["matrix_largest"]),
            ):
                for local in picks:
                    k = int(nz_idx[local])
                    col = int(np.searchsorted(starts, k, side="right") - 1)
                    if col < 0:
                        col = 0
                    if col >= n_cols:
                        col = n_cols - 1 if n_cols > 0 else 0
                    row = int(rows[k])
                    target.append(
                        (
                            float(vals[k]),
                            _name_at(col_names, col, "col"),
                            _name_at(row_names, row, "row"),
                        )
                    )
    except Exception as exc:
        out.setdefault("error", f"matrix scan failed: {exc}")

    # --- Cost coefficients ---
    try:
        costs = np.asarray(lp.col_cost_, dtype=np.float64)
        abs_c = np.abs(costs)
        nz = np.flatnonzero(abs_c > 0.0)
        if nz.size > 0:
            nz_abs = abs_c[nz]
            small = _topk_indices(nz_abs, top, largest=False)
            large = _topk_indices(nz_abs, top, largest=True)
            for picks, target in (
                (small, out["cost_smallest"]),
                (large, out["cost_largest"]),
            ):
                for local in picks:
                    col = int(nz[local])
                    target.append(
                        (float(costs[col]), _name_at(col_names, col, "col"))
                    )
    except Exception as exc:
        out.setdefault("error", f"cost scan failed: {exc}")

    # --- Bound coefficients (col_lower_ / col_upper_, finite, nonzero) ---
    try:
        lo = np.asarray(lp.col_lower_, dtype=np.float64)
        hi = np.asarray(lp.col_upper_, dtype=np.float64)
        # Combine into a single ranking of (abs_value, col_idx, side).
        lo_abs = np.abs(lo)
        hi_abs = np.abs(hi)
        lo_mask = (lo_abs > 0.0) & np.isfinite(lo) & (lo_abs < inf)
        hi_mask = (hi_abs > 0.0) & np.isfinite(hi) & (hi_abs < inf)
        lo_idx = np.flatnonzero(lo_mask)
        hi_idx = np.flatnonzero(hi_mask)
        combined_abs = np.concatenate([lo_abs[lo_idx], hi_abs[hi_idx]])
        combined_col = np.concatenate([lo_idx, hi_idx]).astype(np.int64)
        # side[i] == 0 → lower, 1 → upper.
        combined_side = np.concatenate(
            [np.zeros(lo_idx.size, dtype=np.int8),
             np.ones(hi_idx.size, dtype=np.int8)]
        )
        if combined_abs.size > 0:
            small = _topk_indices(combined_abs, top, largest=False)
            large = _topk_indices(combined_abs, top, largest=True)
            for picks, target in (
                (small, out["bound_smallest"]),
                (large, out["bound_largest"]),
            ):
                for local in picks:
                    col = int(combined_col[local])
                    side = "lower" if combined_side[local] == 0 else "upper"
                    raw = float(lo[col]) if side == "lower" else float(hi[col])
                    target.append(
                        (raw, _name_at(col_names, col, "col"), side)
                    )
    except Exception as exc:
        out.setdefault("error", f"bound scan failed: {exc}")

    # --- RHS coefficients (row_lower_ / row_upper_, finite, nonzero) ---
    try:
        rlo = np.asarray(lp.row_lower_, dtype=np.float64)
        rhi = np.asarray(lp.row_upper_, dtype=np.float64)
        rlo_abs = np.abs(rlo)
        rhi_abs = np.abs(rhi)
        rlo_mask = (rlo_abs > 0.0) & np.isfinite(rlo) & (rlo_abs < inf)
        rhi_mask = (rhi_abs > 0.0) & np.isfinite(rhi) & (rhi_abs < inf)
        rlo_idx = np.flatnonzero(rlo_mask)
        rhi_idx = np.flatnonzero(rhi_mask)
        combined_abs = np.concatenate([rlo_abs[rlo_idx], rhi_abs[rhi_idx]])
        combined_row = np.concatenate([rlo_idx, rhi_idx]).astype(np.int64)
        combined_side = np.concatenate(
            [np.zeros(rlo_idx.size, dtype=np.int8),
             np.ones(rhi_idx.size, dtype=np.int8)]
        )
        if combined_abs.size > 0:
            small = _topk_indices(combined_abs, top, largest=False)
            large = _topk_indices(combined_abs, top, largest=True)
            for picks, target in (
                (small, out["rhs_smallest"]),
                (large, out["rhs_largest"]),
            ):
                for local in picks:
                    row = int(combined_row[local])
                    side = "lower" if combined_side[local] == 0 else "upper"
                    raw = float(rlo[row]) if side == "lower" else float(rhi[row])
                    target.append(
                        (raw, _name_at(row_names, row, "row"), side)
                    )
    except Exception as exc:
        out.setdefault("error", f"rhs scan failed: {exc}")

    return out


def _render_coefficient_sources(sources: dict[str, Any]) -> list[str]:
    """Render Section 8.5 — coefficient-source diagnostic.

    Always returns at least the section header.  When the source dict
    contains an ``error`` entry (or all categories are empty), emits a
    one-line "(coefficient sources unavailable: ...)" notice instead of
    the per-category lists.
    """
    lines: list[str] = [
        "-- 8.5 Coefficient sources -------------------------------------------------"
    ]
    if not sources:
        lines.append("(coefficient sources unavailable: no data)")
        lines.append("")
        return lines

    err = sources.get("error")
    has_any = any(
        sources.get(k)
        for k in (
            "matrix_smallest", "matrix_largest",
            "cost_smallest", "cost_largest",
            "bound_smallest", "bound_largest",
            "rhs_smallest", "rhs_largest",
        )
    )
    if err and not has_any:
        lines.append(f"(coefficient sources unavailable: {err})")
        lines.append("")
        return lines

    name_w = 38

    def _trunc(s: str, w: int) -> str:
        if len(s) <= w:
            return s
        # Keep the start, since FlexTool MPS names embed entity name first.
        return s[: w - 1] + "~"

    def _emit_pair(title: str, entries: list[tuple], fmt: str) -> None:
        lines.append(title)
        if not entries:
            lines.append("  (none)")
            return
        for entry in entries:
            if fmt == "matrix":
                v, col, row = entry
                lines.append(
                    f"  {abs(v):.2e}  {_trunc(col, name_w):<{name_w}s} x "
                    f"{_trunc(row, name_w):<{name_w}s}"
                )
            elif fmt == "cost":
                v, col = entry
                lines.append(
                    f"  {abs(v):.2e}  {_trunc(col, name_w):<{name_w}s}"
                )
            elif fmt == "bound":
                v, col, side = entry
                lines.append(
                    f"  {abs(v):.2e}  {_trunc(col, name_w):<{name_w}s} "
                    f"({side})"
                )
            elif fmt == "rhs":
                v, row, side = entry
                lines.append(
                    f"  {abs(v):.2e}  {_trunc(row, name_w):<{name_w}s} "
                    f"({side})"
                )

    _emit_pair(
        "Matrix smallest |values|:", sources.get("matrix_smallest", []), "matrix"
    )
    _emit_pair(
        "Matrix largest |values|:", sources.get("matrix_largest", []), "matrix"
    )
    lines.append("")
    _emit_pair(
        "Cost smallest |coef|:", sources.get("cost_smallest", []), "cost"
    )
    _emit_pair(
        "Cost largest |coef|:", sources.get("cost_largest", []), "cost"
    )
    lines.append("")
    _emit_pair(
        "Bound smallest |values|:", sources.get("bound_smallest", []), "bound"
    )
    _emit_pair(
        "Bound largest |values|:", sources.get("bound_largest", []), "bound"
    )
    lines.append("")
    _emit_pair(
        "RHS smallest |values|:", sources.get("rhs_smallest", []), "rhs"
    )
    _emit_pair(
        "RHS largest |values|:", sources.get("rhs_largest", []), "rhs"
    )
    lines.append("")
    if err:
        lines.append(f"(partial: {err})")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Escape-slack activity reader  (unchanged from original)
# ---------------------------------------------------------------------------


@dataclass
class SlackActivity:
    """Per-slack summary: total activity and offending cells."""

    slack_name: str
    total: float
    n_nonzero: int
    top_cells: list[tuple[str, str, float]]  # (entity, time, value)


SLACK_VARS: tuple[str, ...] = (
    "vq_state_up",
    "vq_state_down",
    "vq_reserve",
    "vq_inertia",
    "vq_non_synchronous",
    "vq_capacity_margin",
    "vq_state_up_group",
)


def read_slack_activity(
    output_raw_dir: Path,
    solve_name: str,
    top: int = 5,
) -> list[SlackActivity]:
    """Inspect the slack parquet files and return per-slack summaries.

    Unchanged from the original module — the parquet files are already in
    absolute user-facing units (Agent 9 un-scaling is applied before write).
    """
    try:
        import pandas as pd  # noqa: F401
        from flextool.lean_parquet import read_lean_parquet
    except ImportError:
        return [SlackActivity(slack, 0.0, 0, []) for slack in SLACK_VARS]

    results: list[SlackActivity] = []
    for slack in SLACK_VARS:
        shards = list(output_raw_dir.glob(f"{slack}__*.parquet"))
        total = 0.0
        n_nonzero = 0
        top_cells: list[tuple[str, str, float]] = []
        for shard in shards:
            try:
                df = read_lean_parquet(shard)
            except Exception:
                continue
            if df.empty or df.shape[1] == 0:
                continue
            try:
                abs_df = df.abs()
                total += float(abs_df.sum().sum())
                mask = abs_df > ESCAPE_TIER_TOLERANCE
                n_nonzero += int(mask.sum().sum())
                try:
                    stacked = abs_df.where(mask).stack()
                    stacked = stacked.dropna().sort_values(ascending=False)
                    for idx, val in stacked.head(top).items():
                        if isinstance(idx, tuple):
                            row_part = idx[:-1]
                            col_part = idx[-1]
                        else:
                            row_part = ()
                            col_part = idx
                        time_label = (
                            str(row_part[-1]) if len(row_part) >= 1 else ""
                        )
                        if isinstance(col_part, tuple):
                            entity_label = " / ".join(str(x) for x in col_part)
                        else:
                            entity_label = str(col_part)
                        top_cells.append((entity_label, time_label, float(val)))
                except Exception:
                    pass
            except Exception:
                pass
        top_cells.sort(key=lambda t: t[2], reverse=True)
        results.append(
            SlackActivity(
                slack_name=slack,
                total=total,
                n_nonzero=n_nonzero,
                top_cells=top_cells[:top],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Bimodal log10-value collector — FlexData variant
# ---------------------------------------------------------------------------


def _collect_family_log10_values_inmemory(
    family_ranges: dict[str, Any],
    flex_data: "FlexData",
) -> dict[str, list[float]]:
    """Build per-family log10 sample lists from FlexData Params.

    Mirrors ``_collect_family_log10_values`` from the original but pulls
    values directly from the in-memory Params rather than re-reading CSVs.
    Only families that have a corresponding Param are populated; bimodal
    detection is skipped for families with no raw values available.
    """
    from flextool.engine_polars.scaling import _extract_param_values
    import numpy as np

    # Map family name -> list of numpy arrays to pool.
    def _to_log10(arr: np.ndarray) -> list[float]:
        out: list[float] = []
        for v in arr.tolist():
            if v == 0.0 or not math.isfinite(v):
                continue
            out.append(math.log10(abs(v)))
        return out

    family_arrays: dict[str, np.ndarray] = {}

    # entity_unitsize
    if flex_data.p_all_entity_unitsize is not None:
        family_arrays["entity_unitsize"] = _extract_param_values(flex_data.p_all_entity_unitsize)

    # node_inflow / node_annual_flow — same source param.
    if flex_data.p_inflow is not None:
        inflow_arr = _extract_param_values(flex_data.p_inflow)
        family_arrays["node_inflow"] = inflow_arr
        family_arrays["node_annual_flow"] = inflow_arr

    # vom_and_op_costs
    vom_parts = []
    for attr in (
        "p_pssdt_varCost",
        "p_pdt_varCost_source",
        "p_pdt_varCost_sink",
        "p_pdt_varCost_process",
        "p_startup_cost",
        "p_commodity_price",
    ):
        p = getattr(flex_data, attr, None)
        if p is not None:
            vom_parts.append(_extract_param_values(p))
    if vom_parts:
        family_arrays["vom_and_op_costs"] = np.concatenate(vom_parts)

    # capex_invest
    if getattr(flex_data, "ed_lifetime_fixed_cost", None) is not None:
        family_arrays["capex_invest"] = _extract_param_values(
            flex_data.ed_lifetime_fixed_cost
        )

    # node_penalty
    penalty_parts = []
    for attr in ("p_penalty_up", "p_penalty_down"):
        p = getattr(flex_data, attr, None)
        if p is not None:
            penalty_parts.append(_extract_param_values(p))
    if penalty_parts:
        family_arrays["node_penalty"] = np.concatenate(penalty_parts)

    out: dict[str, list[float]] = {}
    for name in family_ranges:
        arr = family_arrays.get(name)
        if arr is not None and arr.size > 0:
            out[name] = _to_log10(arr)
        else:
            out[name] = []
    return out


# ---------------------------------------------------------------------------
# Formatting helpers  (identical to original)
# ---------------------------------------------------------------------------


def _fmt_log(v: Optional[float]) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "    n/a"
    return f"{v:+.2f}"


def _fmt_float(v: Optional[float]) -> str:
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "n/a"
    return f"{v:.3g}"


def _fmt_range(pair: Optional[tuple[float, float]]) -> str:
    if pair is None:
        return "n/a"
    return f"[{pair[0]:.3g}, {pair[1]:.3g}]"


def _spread_decades(pair: Optional[tuple[float, float]]) -> Optional[float]:
    if pair is None:
        return None
    lo, hi = pair
    if lo <= 0 or hi <= 0:
        return None
    return math.log10(hi / lo)


# ---------------------------------------------------------------------------
# Section renderers  (mostly identical to original; near-dup section skipped)
# ---------------------------------------------------------------------------


def _render_header(solve_name: str, highs_version: Optional[str]) -> list[str]:
    lines: list[str] = [
        "=" * 78,
        "FlexTool scaling diagnostic report",
        "=" * 78,
        f"Solve name      : {solve_name}",
        f"Report time     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"HiGHS version   : {highs_version or 'unknown'}",
        "",
        "This report diagnoses numerical scaling of the LP/MIP matrix that HiGHS just",
        "solved.  It flags coefficient-range spread, bimodal coefficient distributions,",
        "composite-scale mismatches between directly-connected entities, near-duplicate",
        "parameter values, and slack activity.  See flextool/SLACK_CONVENTION.md and",
        "project_lp_scaling_2026-04.md for background.",
        "",
    ]
    return lines


def _render_scaling_decisions(
    table: ScaleTable,
    applied_row_scaling: Optional[str],
    applied_obj_scale: Optional[float],
    override_source: Optional[str],
) -> list[str]:
    lines: list[str] = [
        "-- 2. Scaling decisions -----------------------------------------------------"
    ]
    applied_row = (
        applied_row_scaling if applied_row_scaling is not None else "no"
    )
    override_note = (
        f" (source: {override_source})" if override_source else ""
    )
    lines.append(
        f"use_row_scaling        recommended={table.use_row_scaling}  "
        f"applied={applied_row}{override_note}"
    )
    applied_obj_str = (
        f"{applied_obj_scale:g}"
        if applied_obj_scale is not None
        else f"{table.scale_the_objective:g}"
    )
    lines.append(
        f"scale_the_objective    recommended={table.scale_the_objective:g}  "
        f"applied={applied_obj_str}"
    )
    lines.append(f"rough_obj_estimate     {table.rough_obj_estimate:.3g}")
    lines.append(
        f"unitsize_spread_log10  {table.unitsize_spread_log10:.2f} decades"
    )
    rhs_spread = getattr(table, "rhs_spread_log10", 0.0) or 0.0
    cost_spread = getattr(table, "cost_spread_log10", 0.0) or 0.0
    trigger = getattr(table, "row_scaling_trigger", "none") or "none"
    lines.append(f"rhs_spread_log10       {rhs_spread:.2f} decades")
    lines.append(f"cost_spread_log10      {cost_spread:.2f} decades")
    if trigger != "none":
        if trigger == "unitsize":
            spread_val = table.unitsize_spread_log10
        elif trigger == "rhs":
            spread_val = rhs_spread
        else:
            spread_val = cost_spread
        lines.append(
            f"Row scaling triggered by: {trigger} "
            f"(spread = {spread_val:.2f} decades)"
        )
    else:
        lines.append(
            "Row scaling triggered by: none "
            "(all spreads below threshold)"
        )
    user_bound_scale = getattr(table, "user_bound_scale", 0) or 0
    bound_spread = getattr(table, "bound_spread_log10", 0.0) or 0.0
    bound_min = getattr(table, "bound_abs_min", None)
    bound_max = getattr(table, "bound_abs_max", None)
    bound_range_str = (
        f"[{_fmt_float(bound_min)}, {_fmt_float(bound_max)}]"
        if (bound_min is not None and bound_max is not None)
        else "n/a"
    )
    lines.append(
        f"Variable bound scaling: user_bound_scale={user_bound_scale} "
        f"(bound range {bound_range_str}, spread {bound_spread:.2f} decades)"
    )
    lines.append("")
    return lines


def _render_family_table(
    family_ranges: dict[str, Any],
) -> tuple[list[str], int]:
    """Render the coefficient-family table; returns (lines, n_warnings)."""
    lines: list[str] = [
        "-- 3. Coefficient-family ranges --------------------------------------------"
    ]
    lines.append(
        f"{'family':22s} {'n_nz':>6s} {'lg10_min':>10s} {'p10':>7s} "
        f"{'median':>8s} {'p90':>7s} {'lg10_max':>10s} {'abs_min':>10s} {'abs_max':>11s}"
    )
    n_warn = 0
    for name, stats_obj in family_ranges.items():
        if isinstance(stats_obj, dict):
            stats = FamilyStats(**stats_obj)
        else:
            stats = stats_obj
        spread_warn = ""
        if (
            stats.log10_min is not None
            and stats.log10_max is not None
            and (stats.log10_max - stats.log10_min) > FAMILY_SPREAD_WARN_DECADES
        ):
            spread_warn = "  !"
            n_warn += 1
        lines.append(
            f"{name:22s} {stats.n_nonzero:6d} "
            f"{_fmt_log(stats.log10_min):>10s} "
            f"{_fmt_log(stats.log10_p10):>7s} "
            f"{_fmt_log(stats.log10_median):>8s} "
            f"{_fmt_log(stats.log10_p90):>7s} "
            f"{_fmt_log(stats.log10_max):>10s} "
            f"{_fmt_float(stats.abs_min):>10s} "
            f"{_fmt_float(stats.abs_max):>11s}"
            f"{spread_warn}"
        )
    lines.append("")
    if n_warn:
        lines.append(
            f"(! = log10 spread > {FAMILY_SPREAD_WARN_DECADES:.0f} decades -- "
            "family likely needs attention)"
        )
        lines.append("")
    return lines, n_warn


def _render_bimodal(
    family_ranges: dict[str, Any],
    family_log10_values: dict[str, list[float]],
) -> tuple[list[str], int]:
    """Render bimodal-detection section.

    Unlike the original, entity-name representatives are not shown when
    family is not ``entity_unitsize`` (no CSV rescan available) — and even
    for ``entity_unitsize`` we skip the file re-read since FlexData is not
    passed into this renderer.  The section still correctly flags bimodal
    families and reports their cluster parameters.
    """
    lines: list[str] = [
        "-- 4. Bimodal coefficient distributions ------------------------------------"
    ]
    n_warn = 0
    any_flagged = False
    for name, log10_values in family_log10_values.items():
        split = detect_bimodal(log10_values)
        if split is None:
            continue
        any_flagged = True
        n_warn += 1
        lines.append(
            f"[{name}]  ! bimodal: gap = {split.gap_decades:.2f} decades"
        )
        lines.append(
            f"  lower cluster: {split.n_lower} values "
            f"({split.lower_share:.0%}), center log10 = {split.lower_center_log10:+.2f} "
            f"(~{10 ** split.lower_center_log10:.3g})"
        )
        lines.append(
            f"  upper cluster: {split.n_upper} values "
            f"({split.upper_share:.0%}), center log10 = {split.upper_center_log10:+.2f} "
            f"(~{10 ** split.upper_center_log10:.3g})"
        )
        lines.append("  (entity-name representatives unavailable in polars engine path)")
        lines.append("")
    if not any_flagged:
        lines.append("(no bimodal distributions detected)")
        lines.append("")
    return lines, n_warn


def _render_composite_mismatch(
    mismatches: list[MismatchPair],
    top: int = 10,
) -> tuple[list[str], bool]:
    lines: list[str] = [
        "-- 5. Composite-scale mismatch ---------------------------------------------",
    ]
    if not mismatches:
        lines.append(
            f"(no directly-connected pairs span > "
            f"{COMPOSITE_MISMATCH_LOG10_THRESHOLD:.0f} decades in unitsize)"
        )
        lines.append("")
        return lines, False
    lines.append(
        f"Found {len(mismatches)} mismatched pair(s).  Top {min(top, len(mismatches))}:"
    )
    lines.append("")
    lines.append(
        f"{'node':20s} {'small entity':24s} {'size':>10s}   "
        f"{'large entity':24s} {'size':>10s}   {'ratio':>12s}"
    )
    for m in mismatches[:top]:
        lines.append(
            f"{m.node:20s} "
            f"{m.small_entity:24s} {_fmt_num(m.small_size):>10s}   "
            f"{m.large_entity:24s} {_fmt_num(m.large_size):>10s}   "
            f"{_fmt_ratio(m.ratio) + ':1':>12s}"
        )
    lines.append("")
    lines.append(_format_mismatch_recommendation(mismatches[0]))
    lines.append("")
    return lines, True


def _render_near_duplicates_skipped() -> list[str]:
    """Section 6 placeholder — near-duplicate scan requires CSV files."""
    lines: list[str] = [
        "-- 6. Near-duplicate parameter clusters ------------------------------------"
    ]
    lines.append(
        "(near-duplicate scan not available in the polars engine path -- "
        "no CSV directory to scan)"
    )
    lines.append("")
    return lines


def _render_slack_activity(
    activity: list[SlackActivity],
) -> tuple[list[str], int]:
    lines: list[str] = [
        "-- 7. Escape-tier / slack activity -----------------------------------------",
    ]
    firing = [a for a in activity if a.total > ESCAPE_TIER_TOLERANCE]
    if not firing:
        lines.append("(no slack activity above tolerance)")
        lines.append("")
        return lines, 0
    lines.append(
        "Slack variables with total activity (in absolute user-facing units):"
    )
    lines.append("")
    lines.append(
        f"{'slack name':22s} {'total':>14s} {'n_nonzero':>12s}   top-5 offending cells"
    )
    for a in firing:
        top_str = (
            "; ".join(
                f"{ent}@{t}={_fmt_num(v)}" for ent, t, v in a.top_cells
            )
            if a.top_cells
            else "(none recorded)"
        )
        lines.append(
            f"{a.slack_name:22s} {_fmt_float(a.total):>14s} "
            f"{a.n_nonzero:12d}   {top_str}"
        )
    lines.append("")
    lines.append("Any non-zero slack indicates saturation of physical constraints.")
    lines.append("Review the offending cells' inputs.")
    lines.append("")
    return lines, len(firing)


def _render_highs_matrix(parsed: dict[str, Any]) -> tuple[list[str], int]:
    lines: list[str] = [
        "-- 8. HiGHS matrix-range summary -------------------------------------------"
    ]
    n_warn = 0
    matrix = parsed.get("matrix_range")
    cost = parsed.get("cost_range")
    bound = parsed.get("bound_range")
    rhs = parsed.get("rhs_range")
    matrix_decades = _spread_decades(matrix)
    cost_decades = _spread_decades(cost)

    def _mark(decades: Optional[float], warn_threshold: float) -> str:
        if decades is None:
            return ""
        if decades > warn_threshold:
            return "  !"
        return ""

    m_mark = _mark(matrix_decades, MATRIX_RANGE_WARN_DECADES)
    c_mark = _mark(cost_decades, COST_RANGE_WARN_DECADES)
    if m_mark:
        n_warn += 1
    if c_mark:
        n_warn += 1

    lines.append(
        f"Matrix  range : {_fmt_range(matrix)}"
        + (
            f"  (spread {matrix_decades:.1f} decades)"
            if matrix_decades is not None
            else ""
        )
        + m_mark
    )
    lines.append(
        f"Cost    range : {_fmt_range(cost)}"
        + (
            f"  (spread {cost_decades:.1f} decades)"
            if cost_decades is not None
            else ""
        )
        + c_mark
    )
    lines.append(f"Bound   range : {_fmt_range(bound)}")
    lines.append(f"RHS     range : {_fmt_range(rhs)}")
    lines.append("")
    if n_warn:
        lines.append(
            f"(! = spread > {MATRIX_RANGE_WARN_DECADES:.0f} decades matrix / "
            f"{COST_RANGE_WARN_DECADES:.0f} decades cost)"
        )
        lines.append("")
    return lines, n_warn


def _render_summary(
    warnings: int,
    mismatch_fired: bool,
    matrix: Optional[tuple[float, float]],
) -> list[str]:
    lines: list[str] = [
        "-- 9. Summary --------------------------------------------------------------"
    ]
    if warnings == 0 and not mismatch_fired:
        lines.append("Model well-scaled: no significant diagnostics")
    elif mismatch_fired:
        mat_str = _fmt_range(matrix)
        lines.append(
            f"Model poorly scaled: {warnings} warning(s) - matrix range {mat_str}, "
            "see composite-scale-mismatch section"
        )
    else:
        lines.append(
            f"Model scaled acceptably: {warnings} warning(s) - see sections above"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def write_scaling_report(
    scale_table: ScaleTable,
    flex_data: "FlexData",
    solve_data_dir: "Path | str",
    solve_name: str,
    solution: "Solution | None" = None,
    output_raw_dir: "Path | str | None" = None,
    applied_row_scaling: Optional[str] = None,
    applied_obj_scale: Optional[float] = None,
    override_source: Optional[str] = None,
    stdout_summary: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Build ``solve_data/scaling_report.txt`` and return its path.

    Parameters
    ----------
    scale_table:
        Pre-computed :class:`~flextool.engine_polars.scaling.ScaleTable`.
    flex_data:
        In-memory :class:`~flextool.engine_polars.input.FlexData` bag —
        used to read topology and unitsizes for the composite-mismatch
        detector, and to re-scan raw values for bimodal detection.
    solve_data_dir:
        Directory where ``scaling_report.txt`` is written.
    solve_name:
        Human-readable solve identifier (shown in the header).
    solution:
        Optional ``polar_high.Solution`` object from the completed solve.
        When provided, HiGHS matrix / cost / bound / RHS ranges are
        extracted directly from the live LP.  When ``None``, section 8 is
        still rendered but all range fields show "n/a".
    output_raw_dir:
        Directory containing per-variable parquet shards for the slack
        activity scan.  Defaults to ``<solve_data_dir>/../output_raw``.
    applied_row_scaling:
        The ``use_row_scaling`` value actually passed to HiGHS (may differ
        from ``scale_table.use_row_scaling`` if the user overrode it).
    applied_obj_scale:
        The ``scale_the_objective`` value actually used (may differ from
        ``scale_table.scale_the_objective``).
    override_source:
        Description of where *applied_row_scaling* came from (e.g.
        ``"user_db_setting"``).
    stdout_summary:
        When True, print a short summary to stdout after writing.
    logger:
        Optional logger for a one-line debug message.

    Returns
    -------
    Path
        The path to the written report file.
    """
    solve_data_path = Path(solve_data_dir)
    solve_data_path.mkdir(parents=True, exist_ok=True)

    # --- Extract HiGHS ranges (section 8) from the live solution object ---
    if solution is not None:
        parsed_ranges = _extract_highs_matrix_ranges(solution)
    else:
        parsed_ranges = {
            "version": None,
            "matrix_range": None,
            "cost_range": None,
            "bound_range": None,
            "rhs_range": None,
        }

    # --- Build per-family log10 values for bimodal detection ---
    family_log10 = _collect_family_log10_values_inmemory(
        scale_table.family_ranges, flex_data
    )

    # --- Build composite-mismatch pairs ---
    mismatches = find_composite_mismatches(flex_data)

    # --- Slack activity ---
    if output_raw_dir is None:
        output_raw_dir = solve_data_path.parent / "output_raw"
    output_raw_path = Path(output_raw_dir)
    slack_activity = read_slack_activity(output_raw_path, solve_name)

    # --- Render all sections ---
    lines: list[str] = []
    lines.extend(_render_header(solve_name, parsed_ranges.get("version")))
    lines.extend(
        _render_scaling_decisions(
            scale_table,
            applied_row_scaling,
            applied_obj_scale,
            override_source,
        )
    )
    fam_lines, fam_warn = _render_family_table(scale_table.family_ranges)
    lines.extend(fam_lines)
    bimodal_lines, bimodal_warn = _render_bimodal(
        scale_table.family_ranges, family_log10
    )
    lines.extend(bimodal_lines)
    mismatch_lines, mismatch_fired = _render_composite_mismatch(mismatches)
    lines.extend(mismatch_lines)
    lines.extend(_render_near_duplicates_skipped())
    slack_lines, slack_warn = _render_slack_activity(slack_activity)
    lines.extend(slack_lines)
    highs_lines, highs_warn = _render_highs_matrix(parsed_ranges)
    lines.extend(highs_lines)

    # --- Section 8.5: coefficient-source diagnostic (defensive) ---
    try:
        if solution is not None and getattr(solution, "highs", None) is not None:
            coef_sources = _collect_coefficient_sources(solution.highs)
        else:
            coef_sources = {
                "error": "no live HiGHS solver instance",
            }
    except Exception as exc:  # pragma: no cover - safety net
        coef_sources = {"error": f"unexpected failure: {exc}"}
    lines.extend(_render_coefficient_sources(coef_sources))

    total_warnings = (
        fam_warn + bimodal_warn + slack_warn + highs_warn
        + (1 if mismatch_fired else 0)
    )
    lines.extend(
        _render_summary(
            total_warnings,
            mismatch_fired,
            parsed_ranges.get("matrix_range"),
        )
    )

    report_text = "\n".join(lines).rstrip("\n") + "\n"
    out_path = solve_data_path / "scaling_report.txt"
    out_path.write_text(report_text)

    if logger is not None:
        logger.debug(
            "[scaling-report] %s written (%d warnings, mismatch=%s)",
            out_path,
            total_warnings,
            mismatch_fired,
        )

    if stdout_summary:
        _echo_stdout_summary(
            solve_name=solve_name,
            highs_version=parsed_ranges.get("version"),
            mismatch_fired=mismatch_fired,
            mismatches=mismatches,
            total_warnings=total_warnings,
            matrix_range=parsed_ranges.get("matrix_range"),
            out_path=out_path,
        )

    return out_path


def _echo_stdout_summary(
    *,
    solve_name: str,
    highs_version: Optional[str],
    mismatch_fired: bool,
    mismatches: list[MismatchPair],
    total_warnings: int,
    matrix_range: Optional[tuple[float, float]],
    out_path: Path,
) -> None:
    """Print a short stdout summary; expand when mismatch fires."""
    print("")
    print(
        f"[scaling-report] solve={solve_name!r}  HiGHS={highs_version or '?'}  "
        f"warnings={total_warnings}"
    )
    if mismatch_fired:
        print(_format_mismatch_recommendation(mismatches[0]))
    if matrix_range is not None:
        spread = _spread_decades(matrix_range)
        if spread is not None:
            print(
                f"[scaling-report] matrix range "
                f"{matrix_range[0]:.3g}..{matrix_range[1]:.3g} "
                f"({spread:.1f} decades)"
            )
    if total_warnings == 0 and not mismatch_fired:
        print(f"[scaling-report] model well-scaled; full report: {out_path}")
    elif mismatch_fired:
        print(f"[scaling-report] model poorly scaled; full report: {out_path}")
    else:
        print(
            f"[scaling-report] model scaled acceptably; "
            f"full report: {out_path}"
        )
