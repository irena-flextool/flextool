"""Agent 10: user-facing scaling diagnostic report.

After every solve, ``orchestration.run_model`` calls
:func:`write_scaling_report` to emit
``<work_folder>/solve_data/scaling_report.txt`` containing:

1. Header with solve name, timestamp, HiGHS version.
2. Scaling decisions taken (from :class:`scaling.ScaleTable`).
3. Coefficient-family ranges (log10 spread, absolute min/max).
4. Bimodal detection — flags coefficient families with bimodal shape.
5. Composite-scale-mismatch detector — **the load-bearing user
   diagnostic for the LP-scaling project**: finds directly-connected
   entities whose unitsizes span more than 3 orders of magnitude.
   Reports the top-10 offending pairs and prints the recommendation
   text locked by the 2026-04-22 design discussion (see
   ``project_lp_scaling_2026-04.md``).
6. Near-duplicate parameter clusters (reuses
   :func:`precision.report_near_duplicates`).
7. Escape-slack activity — any slack total above a tolerance is
   reported along with top-5 offending cells.
8. HiGHS matrix-range summary parsed from ``HiGHS.log``.
9. Summary line — well-scaled / acceptably / poorly-scaled.

ASCII-only output (no unicode / emoji) for easy diffing.  Stdlib +
pandas only — pandas is already a hard dependency via the rest of the
runner.  The report is always generated (cheap; no opt-in).
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from flextool.flextoolrunner.scaling import FamilyStats, ScaleTable
from flextool.flextoolrunner.precision import report_near_duplicates


# ---------------------------------------------------------------------------
# Thresholds — tuned to match the project design document.
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
# Bimodal detection
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
        # Need a meaningful denominator for the 10% share test.
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
    # Ignore single-value "clusters" — one outlier does not make a mode.
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
# Composite-scale-mismatch detector
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


def _read_entity_unitsize_wide(path: Path) -> dict[str, float]:
    """Parse the wide-format ``p_entity_unitsize.csv``.

    Returns ``{entity_name: unitsize}``.  Empty when the file is absent
    or malformed.
    """
    if not path.exists():
        return {}
    try:
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return {}
    if len(rows) < 2:
        return {}
    header, data = rows[0], rows[1]
    out: dict[str, float] = {}
    for name, cell in zip(header[1:], data[1:]):
        s = (cell or "").strip()
        if not s:
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if math.isfinite(v) and v > 0.0:
            out[name] = v
    return out


def _read_process_node_pairs(
    input_dir: Path,
) -> list[tuple[str, str, str]]:
    """Read process__source.csv and process__sink.csv.

    Returns a list of ``(process, node, role)`` tuples where ``role`` is
    ``"source"`` or ``"sink"``.  Missing files contribute nothing.
    """
    pairs: list[tuple[str, str, str]] = []
    for filename, role in (
        ("process__source.csv", "source"),
        ("process__sink.csv", "sink"),
    ):
        path = input_dir / filename
        if not path.exists():
            continue
        try:
            with path.open(newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if header is None:
                    continue
                for row in reader:
                    if len(row) < 2:
                        continue
                    proc = row[0].strip()
                    node = row[1].strip()
                    if not proc or not node:
                        continue
                    pairs.append((proc, node, role))
        except OSError:
            continue
    return pairs


def find_composite_mismatches(
    input_dir: Path,
    threshold_log10: float = COMPOSITE_MISMATCH_LOG10_THRESHOLD,
) -> list[MismatchPair]:
    """Find directly-connected entity pairs spanning > *threshold_log10* decades.

    Two entities are considered directly connected when they share a
    balance row — i.e. they both appear in the same node's row of the
    ``process_source_sink`` tuple set.  For each node ``n``, we collect
    the processes connected to ``n`` (via either ``process__source`` or
    ``process__sink``), plus the node itself, and look for the largest
    unitsize ratio among the resulting entity cloud.

    The returned list is sorted by ratio descending.  Pairs are
    deduplicated — the same (small, large) entity combination appears
    at most once per shared node.
    """
    unitsizes = _read_entity_unitsize_wide(input_dir / "p_entity_unitsize.csv")
    if not unitsizes:
        return []
    pairs = _read_process_node_pairs(input_dir)
    if not pairs:
        return []

    # Collect entity -> neighbours per node.
    per_node: dict[str, list[tuple[str, str]]] = {}
    for proc, node, role in pairs:
        per_node.setdefault(node, []).append((proc, role))

    seen: set[tuple[str, str, str]] = set()
    mismatches: list[MismatchPair] = []
    for node, procs in per_node.items():
        # Entity cloud = node itself + all connected processes.
        cloud: list[tuple[str, str]] = [(node, "node")]
        cloud.extend(procs)
        # Filter to entities we have a unitsize for.
        cloud_with_sz: list[tuple[str, str, float]] = [
            (ename, role, unitsizes[ename])
            for ename, role in cloud
            if ename in unitsizes
        ]
        if len(cloud_with_sz) < 2:
            continue
        # Pairwise scan — short lists (tens of entities per node at
        # worst), so O(k^2) is fine.
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
                # Pick the "process" side (if exactly one is the node
                # itself, label the other as the process / role from the
                # cloud).  For the report, prefer to show the smaller
                # entity as the process and the bigger as the node — but
                # not all pairs follow that pattern; fall back to the
                # smaller role.
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
                    # Both are processes on the same node — label the
                    # "process" as the smaller one for display, role
                    # becomes the smaller's role.
                    small_role = (
                        r1 if small_e == e1 else r2
                    )
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
\u2014 a ratio of {ratio}:1).
No linear scaling can eliminate matrix coefficient spread arising from such connections.
Recommendations:
  (1) Aggregate the small-side units: e.g., use 1000 buildings instead of 1 to match
      the order of magnitude of the connected system. Accept that aggregation introduces
      some inaccuracy in the small-scale dynamics.
  (2) Run the two subsystems as sequential models: optimise the large system first, then
      use its results as boundary conditions for a detailed small-system run (invest \u2192
      dispatch handoff, or whichever staging fits your use case)."""


def _format_mismatch_recommendation(mismatch: MismatchPair) -> str:
    """Render the locked recommendation text with this pair's values.

    Replaces the two unicode characters (em-dash, rightwards arrow)
    with ASCII so the emitted report stays easily-diffable.
    """
    s = MISMATCH_RECOMMENDATION.format(
        node=mismatch.node,
        small_entity=mismatch.small_entity,
        small_size=_fmt_num(mismatch.small_size),
        large_entity=mismatch.large_entity,
        large_size=_fmt_num(mismatch.large_size),
        ratio=_fmt_ratio(mismatch.ratio),
    )
    # ASCII substitutions for the two non-ASCII glyphs in the template.
    s = s.replace("\u2014", "--").replace("\u2192", "->")
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
# HiGHS log parser — lightweight; harness has a richer one.
# ---------------------------------------------------------------------------


_HIGHS_VERSION = re.compile(r"Running HiGHS\s+([^\s]+)")
_RANGES = re.compile(
    r"Coefficient ranges:\s*\n"
    r"\s*Matrix\s*\[([^,]+),\s*([^\]]+)\]\s*\n"
    r"\s*Cost\s*\[([^,]+),\s*([^\]]+)\]\s*\n"
    r"\s*Bound\s*\[([^,]+),\s*([^\]]+)\]\s*\n"
    r"\s*RHS\s*\[([^,]+),\s*([^\]]+)\]"
)
_PRESOLVE_MATRIX = re.compile(
    r"Presolve:\s*\n.*?Matrix\s*\[([^,]+),\s*([^\]]+)\]",
    re.DOTALL,
)


def parse_highs_log(log_text: str) -> dict[str, Any]:
    """Pull matrix / cost / bound / RHS ranges and HiGHS version from a log."""
    out: dict[str, Any] = {
        "version": None,
        "matrix_range": None,
        "cost_range": None,
        "bound_range": None,
        "rhs_range": None,
    }
    m = _HIGHS_VERSION.search(log_text)
    if m:
        out["version"] = m.group(1).strip()
    m = _RANGES.search(log_text)
    if m:
        g = [float(x) for x in m.groups()]
        out["matrix_range"] = (g[0], g[1])
        out["cost_range"] = (g[2], g[3])
        out["bound_range"] = (g[4], g[5])
        out["rhs_range"] = (g[6], g[7])
    return out


# ---------------------------------------------------------------------------
# Escape-slack activity reader
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

    The parquet values are already in absolute user-facing units
    (Agent 9 un-scaling is applied before parquet write).  Totals are
    absolute-value sums; *n_nonzero* counts cells above
    :data:`ESCAPE_TIER_TOLERANCE`.

    If ``pandas`` / ``pyarrow`` fail to read a file (e.g., empty frame
    written by AMPL printf), the slack is reported as zero activity —
    never raises.
    """
    try:
        import pandas as pd  # noqa: F401
        from flextool.lean_parquet import read_lean_parquet
    except ImportError:
        return [
            SlackActivity(slack, 0.0, 0, [])
            for slack in SLACK_VARS
        ]

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
                # Extract top-k cells.  ``stack`` is deprecated-kw
                # gymnastics across pandas versions — fall back
                # gracefully to a flat scan when the fast path fails.
                try:
                    stacked = abs_df.where(mask).stack()
                    stacked = stacked.dropna().sort_values(ascending=False)
                    for idx, val in stacked.head(top).items():
                        # idx is (row_index_tuple..., col_label_or_tuple)
                        if isinstance(idx, tuple):
                            row_part = idx[:-1]
                            col_part = idx[-1]
                        else:
                            row_part = ()
                            col_part = idx
                        # Row is (solve, period, time) or (solve, period)
                        # — pick the last element for "time" display, or
                        # blank when period-only.
                        time_label = (
                            str(row_part[-1])
                            if len(row_part) >= 1
                            else ""
                        )
                        if isinstance(col_part, tuple):
                            entity_label = " / ".join(
                                str(x) for x in col_part
                            )
                        else:
                            entity_label = str(col_part)
                        top_cells.append(
                            (entity_label, time_label, float(val))
                        )
                except Exception:
                    pass
            except Exception:
                pass
        # Keep just the global top-k across shards.
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
# Formatting helpers
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
# Section renderers
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
    lines: list[str] = ["-- 2. Scaling decisions -----------------------------------------------------"]
    # When the caller does not pass an explicit applied value (e.g. user
    # has no DB setting AND auto-scale is off), the runtime default in
    # ``solve_writers.write_p_use_row_scaling`` is "no" — report that,
    # NOT the recommendation, so the user sees what actually got applied
    # to the solve.
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
        f"{applied_obj_scale:g}" if applied_obj_scale is not None
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
    # Agent 18b: additional row-scaling triggers.  Older JSON files
    # round-tripped into this struct may lack the new fields; fall back
    # to 0.0 / "none" so the report still renders.
    rhs_spread = getattr(table, "rhs_spread_log10", 0.0) or 0.0
    cost_spread = getattr(table, "cost_spread_log10", 0.0) or 0.0
    trigger = getattr(table, "row_scaling_trigger", "none") or "none"
    lines.append(
        f"rhs_spread_log10       {rhs_spread:.2f} decades"
    )
    lines.append(
        f"cost_spread_log10      {cost_spread:.2f} decades"
    )
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
    # Agent 18c — variable-bound scaling line.  Populated by
    # ``solver_runner`` after HiGHS has loaded the LP.  For older
    # JSON round-trips these fields may be absent; fall back to 0.
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
    lines: list[str] = ["-- 3. Coefficient-family ranges --------------------------------------------"]
    lines.append(
        f"{'family':22s} {'n_nz':>6s} {'lg10_min':>10s} {'p10':>7s} "
        f"{'median':>8s} {'p90':>7s} {'lg10_max':>10s} {'abs_min':>10s} {'abs_max':>11s}"
    )
    n_warn = 0
    for name, stats_obj in family_ranges.items():
        # stats_obj may be a FamilyStats instance OR a dict (after JSON round-trip).
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
            f"(! = log10 spread > {FAMILY_SPREAD_WARN_DECADES:.0f} decades — "
            "family likely needs attention)"
        )
        lines.append("")
    return lines, n_warn


def _collect_family_log10_values(
    family_ranges: dict[str, Any],
    input_dir: Path,
) -> dict[str, list[float]]:
    """Rescan the CSVs in *input_dir* for per-family log10 sample values.

    The :class:`FamilyStats` summary keeps only quantiles, not the raw
    values, so bimodal detection needs its own pass.  We import the
    family definitions lazily and reuse the same pool-all-cells
    convention as ``scaling._scan_family``.
    """
    from flextool.flextoolrunner import scaling

    out: dict[str, list[float]] = {}
    for name in family_ranges:
        if name == "entity_unitsize":
            values = scaling._read_entity_unitsizes(input_dir)
        else:
            files = scaling.FAMILIES.get(name, [])
            values = scaling._scan_family(input_dir, files)
        log10_values: list[float] = []
        for v in values:
            if v == 0.0 or not math.isfinite(v):
                continue
            log10_values.append(math.log10(abs(v)))
        out[name] = log10_values
    return out


def _top_representatives(
    input_dir: Path,
    family: str,
    split_log10: float,
    side: str,
    k: int = 3,
) -> list[tuple[str, float]]:
    """Return up to *k* (entity-name, value) pairs on the given side.

    ``side == "lower"`` takes entities with log10(value) <= split; ``upper``
    takes entities above.  Currently only supports ``family == "entity_unitsize"``
    — the only family where per-entity names are cheap to obtain.  Other
    families return an empty list (the bimodal report then lists "(names
    unavailable)").
    """
    if family != "entity_unitsize":
        return []
    path = input_dir / "p_entity_unitsize.csv"
    unitsizes = _read_entity_unitsize_wide(path)
    if not unitsizes:
        return []
    entries: list[tuple[str, float]] = []
    for ename, v in unitsizes.items():
        if v <= 0:
            continue
        lv = math.log10(v)
        if side == "lower" and lv <= split_log10:
            entries.append((ename, v))
        elif side == "upper" and lv > split_log10:
            entries.append((ename, v))
    entries.sort(key=lambda t: t[1], reverse=(side == "upper"))
    return entries[:k]


def _render_bimodal(
    family_ranges: dict[str, Any],
    family_log10_values: dict[str, list[float]],
    input_dir: Path,
) -> tuple[list[str], int]:
    lines: list[str] = ["-- 4. Bimodal coefficient distributions ------------------------------------"]
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
        lower_reps = _top_representatives(input_dir, name, split.split_log10, "lower")
        upper_reps = _top_representatives(input_dir, name, split.split_log10, "upper")
        if lower_reps:
            pretty = ", ".join(f"{n}={_fmt_num(v)}" for n, v in lower_reps)
            lines.append(f"  lower representatives: {pretty}")
        else:
            lines.append("  lower representatives: (names unavailable)")
        if upper_reps:
            pretty = ", ".join(f"{n}={_fmt_num(v)}" for n, v in upper_reps)
            lines.append(f"  upper representatives: {pretty}")
        else:
            lines.append("  upper representatives: (names unavailable)")
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
    # User-facing recommendation — locked wording from project memo.
    lines.append(_format_mismatch_recommendation(mismatches[0]))
    lines.append("")
    return lines, True


def _render_near_duplicates(input_dir: Path) -> list[str]:
    lines: list[str] = ["-- 6. Near-duplicate parameter clusters ------------------------------------"]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            report_near_duplicates(input_dir, top=10)
    except Exception as exc:  # defensive — diagnostic never fails the run
        lines.append(f"(near-duplicate scan failed: {exc})")
        lines.append("")
        return lines
    body = buf.getvalue().rstrip("\n")
    if not body:
        lines.append("(no output)")
    else:
        for row in body.splitlines():
            lines.append(row)
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
                f"{ent}@{t}={_fmt_num(v)}"
                for ent, t, v in a.top_cells
            )
            if a.top_cells else "(none recorded)"
        )
        lines.append(
            f"{a.slack_name:22s} {_fmt_float(a.total):>14s} "
            f"{a.n_nonzero:12d}   {top_str}"
        )
    lines.append("")
    lines.append(
        "Any non-zero slack indicates saturation of physical constraints."
    )
    lines.append(
        "Review the offending cells' inputs."
    )
    lines.append("")
    return lines, len(firing)


def _render_highs_matrix(parsed: dict[str, Any]) -> tuple[list[str], int]:
    lines: list[str] = ["-- 8. HiGHS matrix-range summary -------------------------------------------"]
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
        + (f"  (spread {matrix_decades:.1f} decades)" if matrix_decades is not None else "")
        + m_mark
    )
    lines.append(
        f"Cost    range : {_fmt_range(cost)}"
        + (f"  (spread {cost_decades:.1f} decades)" if cost_decades is not None else "")
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
    lines: list[str] = ["-- 9. Summary --------------------------------------------------------------"]
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
    input_dir: Path | str,
    solve_data_dir: Path | str,
    solve_name: str,
    highs_log_path: Path | str | None = None,
    output_raw_dir: Path | str | None = None,
    applied_row_scaling: Optional[str] = None,
    applied_obj_scale: Optional[float] = None,
    override_source: Optional[str] = None,
    stdout_summary: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Build ``solve_data/scaling_report.txt`` and return its path.

    When *stdout_summary* is True, a short echo (header + mismatch
    section if fired + summary) is printed to stdout via ``print``.
    """
    input_path = Path(input_dir)
    solve_data_path = Path(solve_data_dir)
    solve_data_path.mkdir(parents=True, exist_ok=True)

    # --- Gather inputs ---
    if highs_log_path is not None:
        highs_log_path = Path(highs_log_path)
        log_text = (
            highs_log_path.read_text(errors="replace")
            if highs_log_path.exists()
            else ""
        )
    else:
        log_text = ""
    parsed_log = parse_highs_log(log_text)

    family_log10 = _collect_family_log10_values(
        scale_table.family_ranges, input_path
    )
    mismatches = find_composite_mismatches(input_path)

    if output_raw_dir is None:
        # Guess from the solve_data convention: ``output_raw`` is its sibling.
        output_raw_dir = solve_data_path.parent / "output_raw"
    output_raw_path = Path(output_raw_dir)
    slack_activity = read_slack_activity(output_raw_path, solve_name)

    # --- Render sections ---
    lines: list[str] = []
    lines.extend(_render_header(solve_name, parsed_log.get("version")))
    lines.extend(_render_scaling_decisions(
        scale_table,
        applied_row_scaling,
        applied_obj_scale,
        override_source,
    ))
    fam_lines, fam_warn = _render_family_table(scale_table.family_ranges)
    lines.extend(fam_lines)
    bimodal_lines, bimodal_warn = _render_bimodal(
        scale_table.family_ranges, family_log10, input_path
    )
    lines.extend(bimodal_lines)
    mismatch_lines, mismatch_fired = _render_composite_mismatch(mismatches)
    lines.extend(mismatch_lines)
    lines.extend(_render_near_duplicates(input_path))
    slack_lines, slack_warn = _render_slack_activity(slack_activity)
    lines.extend(slack_lines)
    highs_lines, highs_warn = _render_highs_matrix(parsed_log)
    lines.extend(highs_lines)
    total_warnings = (
        fam_warn + bimodal_warn + slack_warn + highs_warn
        + (1 if mismatch_fired else 0)
    )
    lines.extend(_render_summary(
        total_warnings, mismatch_fired, parsed_log.get("matrix_range")
    ))

    report_text = "\n".join(lines).rstrip("\n") + "\n"
    out_path = solve_data_path / "scaling_report.txt"
    out_path.write_text(report_text)

    if logger is not None:
        logger.info(
            "[scaling-report] %s written (%d warnings, mismatch=%s)",
            out_path, total_warnings, mismatch_fired,
        )

    if stdout_summary:
        _echo_stdout_summary(
            solve_name=solve_name,
            highs_version=parsed_log.get("version"),
            mismatch_fired=mismatch_fired,
            mismatches=mismatches,
            total_warnings=total_warnings,
            matrix_range=parsed_log.get("matrix_range"),
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
    """Print a 5-10 line stdout summary; expand when mismatch fires."""
    print("")
    print(
        f"[scaling-report] solve={solve_name!r}  HiGHS={highs_version or '?'}  "
        f"warnings={total_warnings}"
    )
    if mismatch_fired:
        # When the load-bearing diagnostic fires, emit the full
        # recommendation so the user sees it immediately without
        # opening the report file.
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
        print(
            f"[scaling-report] model poorly scaled; full report: {out_path}"
        )
    else:
        print(
            f"[scaling-report] model scaled acceptably; "
            f"full report: {out_path}"
        )
