"""Compute per-node-per-period bounds for the two-tier state slack.

The two-tier slack convention (see ``flextool/SLACK_CONVENTION.md``) splits
every ``vq_*`` into a bounded primary (``<= K_rel``) plus an unbounded
escape (``>= 0``).  For ``vq_state_up`` / ``vq_state_down``, the primary
cap ``K_rel[n, d]`` must be large enough to absorb the node's expected
demand shortfall per timestep *in units of the row scaler* so the
escape tier only fires on pathological inputs.

Definition::

    K_rel[n, d] = max(
        1,
        ceil_pow10(max_demand_per_step[n]
                   / (node_cap[n, d] * min_step_duration[d]))
    )

where

* ``node_cap[n, d]`` is ``node_capacity_for_scaling[n, d]``.  In the
  pre-Agent-5 state this defaults to 1 everywhere, so ``K_rel``
  collapses to ``ceil_pow10(max |inflow|)``.
* ``max_demand_per_step[n]`` is the max absolute value of the node's
  timeseries inflow / demand profile across ``pt_node_inflow`` and
  ``pbt_node_inflow`` and any constant ``p_node[inflow]`` +
  ``pd_node[inflow]`` values.
* ``min_step_duration[d]`` is the shortest ``step_duration`` across
  timesteps present in period ``d`` (from
  ``solve_data/steps_in_use.csv``).
* ``ceil_pow10(x)`` rounds ``x > 0`` up to the nearest power of 10 so
  structurally-identical nodes share the same cap — preserves the HiGHS
  symmetry detection the model relies on.

Nodes with no demand profile get ``K_rel = 1`` (default).  The resulting
CSV ``solve_data/p_state_slack_k_rel.csv`` has one row per
``(solve, period, node)`` — long format — matching the shape of
``pdtNode_penalty_up.csv`` for per-node-per-period writes in the model.
The AMPL reader declares ``p_state_slack_k_rel[n, d]`` with default 1
so any node missing from the CSV retains the minimum cap.

This module is deliberately minimal.  Agent 8 will land a proper
``ScaleAnalyzer`` that covers the full set of objective and row scalers;
``K_rel`` lives here until then.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Primitive: round up to next power of 10
# ---------------------------------------------------------------------------


def ceil_pow10(x: float) -> float:
    """Return the smallest power of 10 ``>= x``.

    ``ceil_pow10(1) == 1``, ``ceil_pow10(9) == 10``, ``ceil_pow10(10) == 10``,
    ``ceil_pow10(11) == 100``.  ``x <= 0`` returns 1 — this keeps callers
    safe when a node has no demand.
    """
    if x <= 0:
        return 1.0
    if x <= 1:
        return 1.0
    # log10 rounding — guard against tiny fp wobble at exact powers.
    exp = math.log10(x)
    exp_ceil = math.ceil(exp - 1e-12)
    return 10.0 ** exp_ceil


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _read_pt_node_inflow(path: Path) -> dict[str, float]:
    """Return ``{node: max |inflow|}`` from a long-format ``pt_node_inflow.csv``.

    Headers: ``node,time,pt_node_inflow``.
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row.get("node") or ""
            if not node:
                continue
            raw = row.get("pt_node_inflow") or ""
            if raw == "" or raw is None:
                continue
            try:
                v = abs(float(raw))
            except ValueError:
                continue
            prev = out.get(node, 0.0)
            if v > prev:
                out[node] = v
    return out


def _read_pbt_node_inflow(path: Path) -> dict[str, float]:
    """Return ``{node: max |inflow|}`` from ``pbt_node_inflow.csv``.

    Headers: ``node,branch,time_start,time,pbt_node_inflow``.
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row.get("node") or ""
            if not node:
                continue
            raw = row.get("pbt_node_inflow") or ""
            if raw == "" or raw is None:
                continue
            try:
                v = abs(float(raw))
            except ValueError:
                continue
            prev = out.get(node, 0.0)
            if v > prev:
                out[node] = v
    return out


def _read_scalar_inflow(p_node_path: Path, pd_node_path: Path) -> dict[str, float]:
    """Return ``{node: max |inflow|}`` from the scalar ``p_node`` +
    per-period ``pd_node`` ``inflow`` parameter values.

    A node that uses a constant inflow (no timeseries) carries its value
    here; we keep it so the K_rel computation is robust to model
    variants that don't use ``pt_node_inflow``.
    """
    out: dict[str, float] = {}
    if p_node_path.exists():
        with p_node_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("nodeParam") or "") != "inflow":
                    continue
                node = row.get("node") or ""
                raw = row.get("p_node") or ""
                if not node or not raw:
                    continue
                try:
                    v = abs(float(raw))
                except ValueError:
                    continue
                if v > out.get(node, 0.0):
                    out[node] = v
    if pd_node_path.exists():
        with pd_node_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("nodeParam") or "") != "inflow":
                    continue
                node = row.get("node") or ""
                raw = row.get("pd_node") or ""
                if not node or not raw:
                    continue
                try:
                    v = abs(float(raw))
                except ValueError:
                    continue
                if v > out.get(node, 0.0):
                    out[node] = v
    return out


def _read_min_step_duration(steps_in_use_csv: Path) -> dict[str, float]:
    """Return ``{period: min step_duration}`` from ``solve_data/steps_in_use.csv``.

    Headers: ``period,step,step_duration``.
    """
    out: dict[str, float] = {}
    if not steps_in_use_csv.exists():
        return out
    with steps_in_use_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            period = row.get("period") or ""
            raw = row.get("step_duration") or ""
            if not period or not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if v <= 0:
                continue
            prev = out.get(period)
            if prev is None or v < prev:
                out[period] = v
    return out


def _read_periods_in_use(
    set_period_in_use_csv: Path,
    steps_in_use_csv: Path,
    solve: str | None = None,
) -> list[str]:
    """Return the list of periods active in the current solve.

    Prefers ``solve_data/set_period_in_use.csv`` when present; falls
    back to the distinct periods in ``solve_data/steps_in_use.csv``.

    The CSV layout written by ``flextool.mod`` is ``solve,period`` with
    a header row (see ``flextool.mod`` around line 5422).  On
    multi-solve scenarios the file accumulates rows across solves, so
    when ``solve`` is provided we filter to that solve; otherwise we
    return every unique period in first-seen order.
    """
    if set_period_in_use_csv.exists():
        periods: list[str] = []
        seen: set[str] = set()
        with set_period_in_use_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = (row.get("period") or "").strip()
                if not p:
                    continue
                if solve is not None:
                    s = (row.get("solve") or "").strip()
                    if s != solve:
                        continue
                if p in seen:
                    continue
                seen.add(p)
                periods.append(p)
        if periods:
            return periods
    # Fallback: distinct periods from steps_in_use.csv in first-seen order.
    if not steps_in_use_csv.exists():
        return []
    seen_list: list[str] = []
    seen_set: set[str] = set()
    with steps_in_use_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("period") or ""
            if p and p not in seen_set:
                seen_list.append(p)
                seen_set.add(p)
    return seen_list


def _read_node_capacity_for_scaling(path: Path) -> dict[tuple[str, str], float]:
    """Return ``{(node, period): node_capacity_for_scaling}`` from a
    wide CSV with layout ``solve,period,<node>,<node>,...``.

    Missing ⇒ 1.0 is applied by the caller.  Pre-Agent-5 the file is
    written with every cell = 1, so this is a no-op today — the hook is
    in place for when Agent 5 activates the formula.
    """
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return out
        if len(header) < 3:
            return out
        node_cols = header[2:]
        for row in reader:
            if len(row) < 3:
                continue
            period = row[1]
            for i, node in enumerate(node_cols):
                idx = 2 + i
                if idx >= len(row):
                    break
                raw = row[idx]
                if raw == "":
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    continue
                out[(node, period)] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_k_rel(
    *,
    max_demand_per_step: dict[str, float],
    min_step_duration: dict[str, float],
    node_capacity_for_scaling: dict[tuple[str, str], float],
    nodes: Iterable[str],
    periods: Iterable[str],
) -> dict[tuple[str, str], float]:
    """Compute ``K_rel[n, d]`` for every (node, period) combination.

    Pure function — all inputs are explicit dicts so the core algorithm
    is trivial to unit test.  The caller is responsible for reading
    them from CSV (see :func:`write_p_state_slack_k_rel`).

    Applies the formula described at the top of this module; defaults
    missing ``node_capacity_for_scaling`` and ``min_step_duration`` to 1.
    """
    out: dict[tuple[str, str], float] = {}
    nodes_list = list(nodes)
    periods_list = list(periods)
    for n in nodes_list:
        demand = max_demand_per_step.get(n, 0.0)
        for d in periods_list:
            cap = node_capacity_for_scaling.get((n, d), 1.0)
            step = min_step_duration.get(d, 1.0)
            denom = cap * step
            if denom <= 0:
                denom = 1.0
            ratio = demand / denom
            k = max(1.0, ceil_pow10(ratio))
            out[(n, d)] = k
    return out


def write_p_state_slack_k_rel(
    solve: str,
    *,
    work_folder: Path,
) -> Path:
    """Compute and write ``solve_data/p_state_slack_k_rel.csv``.

    Long format: ``solve,period,node,p_state_slack_k_rel``.  Only rows
    where ``K_rel != 1.0`` are emitted — nodes missing from the file
    inherit the AMPL default of 1 via the reader, keeping the file
    small on scenarios with a few large-demand nodes embedded in many
    trivial ones.

    Reads every data source directly from the filesystem — ``input/*``
    for the raw node definitions and ``solve_data/*`` for the
    solve-specific schedule and scaler.  Must be called AFTER the
    orchestration loop has written ``steps_in_use.csv`` and
    ``node_capacity_for_scaling.csv`` (the former is written by
    :func:`solve_writers.write_active_timelines`; the latter is
    emitted by the AMPL model itself, so on the very first solve of a
    model it is absent and the helper falls back to the default = 1).
    """
    wf = Path(work_folder)
    input_dir = wf / "input"
    solve_data = wf / "solve_data"

    # Profile sources (raw input)
    max_demand_pt  = _read_pt_node_inflow(input_dir / "pt_node_inflow.csv")
    max_demand_pbt = _read_pbt_node_inflow(input_dir / "pbt_node_inflow.csv")
    max_demand_sc  = _read_scalar_inflow(
        input_dir / "p_node.csv", input_dir / "pd_node.csv",
    )
    # Merge: per-node max across all three sources.
    max_demand_per_step: dict[str, float] = {}
    for source in (max_demand_pt, max_demand_pbt, max_demand_sc):
        for n, v in source.items():
            if v > max_demand_per_step.get(n, 0.0):
                max_demand_per_step[n] = v

    min_step_duration = _read_min_step_duration(solve_data / "steps_in_use.csv")
    periods = _read_periods_in_use(
        solve_data / "set_period_in_use.csv",
        solve_data / "steps_in_use.csv",
        solve=solve,
    )
    node_cap = _read_node_capacity_for_scaling(
        solve_data / "node_capacity_for_scaling.csv",
    )

    # Every node that has any demand + every node mentioned in the
    # scaler map; that's the universe the reader cares about.  Nodes
    # the model defines but with no demand and no scaler row will
    # simply inherit the default (1) via omission from the CSV.
    nodes: set[str] = set(max_demand_per_step)
    for (n, _d) in node_cap:
        nodes.add(n)

    k_rel = compute_k_rel(
        max_demand_per_step=max_demand_per_step,
        min_step_duration=min_step_duration,
        node_capacity_for_scaling=node_cap,
        nodes=nodes,
        periods=periods,
    )

    out_path = solve_data / "p_state_slack_k_rel.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solve", "period", "node", "p_state_slack_k_rel"])
        # Stable iteration order: periods as listed in the schedule,
        # nodes alphabetically.  Only write non-default values.
        for d in periods:
            for n in sorted(nodes):
                val = k_rel.get((n, d), 1.0)
                if val != 1.0:
                    writer.writerow([solve, d, n, format(val, "g")])
    return out_path
