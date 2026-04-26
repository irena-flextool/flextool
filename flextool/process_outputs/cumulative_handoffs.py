"""
Rolling per-period accumulator writers for price-ladder tiers and the
model-wide CO2 cap.

These writers maintain three accumulators across rolling solves so the
LP's cumulative caps (ladder annual/cumulative and ``co2_max_total``)
can partition a period's quota across rolls that share that period:

* ``solve_data/ladder_cum_realized_mwh.csv`` — ``(commodity, tier, period)
  → realized sim-MWh of that tier of that commodity that fell into that
  period across all prior rolls``.  Loaded by the mod into
  ``p_ladder_cum_realized_mwh[c, i, d]``.

* ``solve_data/ladder_cum_sim_hours.csv`` — ``period → realized sim-hours
  of that period across all prior rolls``.  Loaded by the mod into
  ``p_ladder_cum_sim_hours[d]``.  Drives ``f_d_k[d]`` — shared between
  the ladder and CO2 cumulative caps.

* ``solve_data/co2_cum_realized_tonnes.csv`` — ``(group, period) →
  realized sim-window tonnes of CO2 emitted across all prior rolls``.
  Loaded by the mod into ``p_co2_cum_realized_tonnes[g, d]`` (already
  scaled by /1000, matching the mod's RHS convention).

Together with the current roll's horizon the mod computes ``f_d_k[d]``,
the fraction of period d "filled" by prior-realized hours plus this-roll
hours, and caps v_trade / CO2 on a rolling-partition basis (see
``flextool.mod`` ladder_tier_cap_annual_roll / _cumulative_roll /
_annual_overspent / _cumulative_overspent and co2_max_total).

Uniform-split assumption (ladder only — Bug #1 fix, commit #7)
--------------------------------------------------------------
``v_trade[c, n, d, i]`` is period-level in the LP (no time index — the
LP makes one trade decision per (commodity, node, period, tier)).  A
roll's horizon covers only some of period d's hours, realizes a subset
(the dispatch window), and looks ahead at the rest.  When attributing
a roll's v_trade MWh to the "realized" slice, we assume the LP's
period-level decision distributes uniformly over the horizon's hours:

    realized_mwh_this_roll[c, i, d] =
        sum_n v_trade[c, n, d, i] * p_commodity_unitsize[c]
          * (realized_sim_hours_d / horizon_sim_hours_d)

This is a modelling choice — the LP itself cannot distinguish "realized"
from "lookahead" hours within a single v_trade number — but it matches
how cost and dispatch are already attributed at the period-level in the
rest of the model.  Under it the accumulators form a proper partition:
summing across all rolls of a run reconstructs the full realized MWh.

CO2 is derived directly (no uniform-split needed)
-------------------------------------------------
Unlike v_trade, CO2 emissions come from ``v_flow[p, source, sink, d, t]``
which is per-timestep — so realized tonnes can be accumulated directly
over realized (d, t) pairs without any uniform-split assumption:

    realized_tonnes_this_roll[g, d] =
      sum over realized (d, t) pairs, per (c, n) in group's co2 nodes,
        of  p_commodity[c, 'co2_content'] / 1000
          * (buy-side flows - sell-side flows)
          * step_duration[d, t] * p_rp_cost_weight[d, t]

MVP scope: no-eff + simple-eff branches only.  Min-load-efficiency and
online-section corrections are deferred; if a model uses them on a
priced-commodity node the accumulator under-reports tonnes (cap slightly
over-enforces — conservative) and a warning is logged.

Hooked into ``solver_runner._run_highs_or_cplex`` after
:func:`flextool.process_outputs.handoff_writers.write_all_handoffs`.
HiGHS-only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from flextool.process_outputs.handoff_writers import (
    _is_first_solve,
    _load_step_duration,
)
from flextool.process_outputs.read_highs_solution import (
    _load_realized_set,
    extract_variable,
)

if TYPE_CHECKING:
    import highspy

_logger = logging.getLogger(__name__)

# Matches flextool.mod's "infinite / inactive" sentinel.  Tiers with
# total_cap at or above this value are the infinite tail — they have no
# cap, so no accumulator is needed for them.
_INFINITE_TIER_THRESHOLD = 1e29

# Commodities that need accumulator tracking: anything using a ladder
# price method (both annual and cumulative).  `price` commodities are
# skipped entirely.
_LADDER_PRICE_METHODS = frozenset(
    {"price_ladder_annual", "price_ladder_cumulative"}
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_price_methods(work_folder: Path) -> dict[str, str]:
    """Return ``{commodity: price_method}`` from
    ``input/p_commodity_price_method.csv``.  Missing file → empty
    (every commodity defaults to ``'price'`` in the mod)."""
    path = work_folder / "input" / "p_commodity_price_method.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "commodity" not in df.columns:
        return {}
    return {
        str(r["commodity"]): str(r["p_commodity_price_method"])
        for _, r in df.iterrows()
    }


def _load_ladder_commodities(work_folder: Path) -> set[str]:
    """Return the set of commodities using any ``price_ladder_*`` method.

    Both annual and cumulative tiers need rolling accumulators now — the
    mod's per-period cap partitions a period's quota across rolls that
    share that period.  A non-ladder commodity uses scalar price and is
    skipped.
    """
    return {
        c for c, m in _load_price_methods(work_folder).items()
        if m in _LADDER_PRICE_METHODS
    }


def _load_finite_ladder_tiers(
    work_folder: Path,
) -> dict[tuple[str, int], float]:
    """Return ``{(commodity, tier): quantity}`` for every ladder tier of
    any ``price_ladder_*`` commodity with a finite cap.

    Reads both ``input/commodity_ladder_cumulative.csv`` and
    ``input/commodity_ladder_annual.csv``.  Infinite tiers (quantity >=
    1e29 sentinel) are dropped — they never bind and their accumulator
    row would contribute nothing.  For the annual CSV the quantity can
    vary per period; we treat a tier as "finite" if ANY period has a
    finite quantity (the accumulator writer emits per-period rows, and
    the annual constraint also evaluates per period).
    """
    ladder_commodities = _load_ladder_commodities(work_folder)
    if not ladder_commodities:
        return {}
    out: dict[tuple[str, int], float] = {}

    cum_path = work_folder / "input" / "commodity_ladder_cumulative.csv"
    if cum_path.exists():
        df = pd.read_csv(cum_path)
        if not df.empty:
            for _, row in df.iterrows():
                c = str(row["commodity"])
                if c not in ladder_commodities:
                    continue
                try:
                    tier = int(row["tier"])
                    q = float(row["quantity"])
                except (ValueError, TypeError):
                    continue
                if q != q or q >= _INFINITE_TIER_THRESHOLD:
                    continue
                out[(c, tier)] = q

    ann_path = work_folder / "input" / "commodity_ladder_annual.csv"
    if ann_path.exists():
        df = pd.read_csv(ann_path)
        if not df.empty:
            # Per-period rows; keep the minimum finite quantity seen
            # across periods (structural: any finite period makes the
            # accumulator relevant for that tier).
            for _, row in df.iterrows():
                c = str(row["commodity"])
                if c not in ladder_commodities:
                    continue
                try:
                    tier = int(row["tier"])
                    q = float(row["quantity"])
                except (ValueError, TypeError):
                    continue
                if q != q or q >= _INFINITE_TIER_THRESHOLD:
                    continue
                key = (c, tier)
                prev = out.get(key)
                out[key] = q if prev is None else min(prev, q)
    return out


def _load_commodity_unitsize(work_folder: Path) -> dict[str, float]:
    """Return ``{commodity: unitsize}`` from
    ``input/p_commodity_unitsize.csv``.  Unknown commodities default to
    1.0 at lookup time (matches the mod's default)."""
    path = work_folder / "input" / "p_commodity_unitsize.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "commodity" not in df.columns:
        return {}
    return {
        str(r["commodity"]): float(r["p_commodity_unitsize"])
        for _, r in df.iterrows()
    }


def _load_prior_cum_realized_mwh(
    path: Path,
) -> dict[tuple[str, int, str], float]:
    """Return ``{(commodity, tier, period): cum_mwh}`` from the previous
    roll's accumulator CSV.  Header-only seed → empty.
    """
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"commodity", "tier", "period", "p_ladder_cum_realized_mwh"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    out: dict[tuple[str, int, str], float] = {}
    for _, row in df.iterrows():
        try:
            tier = int(row["tier"])
            val = float(row["p_ladder_cum_realized_mwh"])
        except (ValueError, TypeError):
            continue
        out[(str(row["commodity"]), tier, str(row["period"]))] = val
    return out


def _load_prior_cum_sim_hours(path: Path) -> dict[str, float]:
    """Return ``{period: cum_hours}`` from the previous roll's accumulator."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"period", "p_ladder_cum_sim_hours"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    return {
        str(r["period"]): float(r["p_ladder_cum_sim_hours"])
        for _, r in df.iterrows()
    }


# ---------------------------------------------------------------------------
# CO2 accumulator loaders
# ---------------------------------------------------------------------------


def _load_prior_co2_cum_realized_tonnes(
    path: Path,
) -> dict[tuple[str, str], float]:
    """Return ``{(group, period): cum_tonnes}`` from the previous roll's
    CO2 accumulator CSV.  Header-only seed → empty."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    needed = {"group", "period", "p_co2_cum_realized_tonnes"}
    if df.empty or not needed.issubset(df.columns):
        return {}
    out: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        try:
            val = float(row["p_co2_cum_realized_tonnes"])
        except (ValueError, TypeError):
            continue
        out[(str(row["group"]), str(row["period"]))] = val
    return out


def _load_co2_max_total_groups(work_folder: Path) -> set[str]:
    """Return ``{group}`` for groups subject to the ``co2_max_total`` cap.

    Mirrors the mod's ``group_co2_max_total`` set definition (line
    ~187): every group whose ``co2_method`` is one of the
    ``co2_max_total_method`` variants.  Source:
    ``input/group__co2_method.csv``.  For robustness the set of
    "total" methods covers the common user-facing value ``total``.
    A future alternative-name can be added here without churning the
    mod.
    """
    path = work_folder / "input" / "group__co2_method.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "group" not in df.columns or "co2_method" not in df.columns:
        return set()
    # Matches flextool_base.dat:186
    #   co2_max_total_method := total, price_total, period_total, price_period_total
    total_methods = {
        "total", "price_total", "period_total", "price_period_total",
    }
    mask = df["co2_method"].astype(str).isin(total_methods)
    return {str(g) for g in df.loc[mask, "group"]}


def _load_commodity_co2_content(work_folder: Path) -> dict[str, float]:
    """Return ``{commodity: co2_content}`` from
    ``input/p_commodity_co2_content.csv``.  Wide format:
    header row = ``commodity,<c1>,<c2>,...``;
    row 2 = ``value,<v1>,<v2>,...``.
    """
    path = work_folder / "input" / "p_commodity_co2_content.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0)
    if df.empty or "value" not in df.index:
        return {}
    row = df.loc["value"]
    out: dict[str, float] = {}
    for c, v in row.items():
        try:
            out[str(c)] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def _load_commodity_node_co2(work_folder: Path) -> set[tuple[str, str]]:
    """Return ``{(commodity, node)}`` from ``solve_data/commodity_node_co2.csv``.

    These are the (c, n) pairs that contribute to CO2 caps.  Only
    process flows into/out of these nodes are aggregated by the writer.
    """
    path = work_folder / "solve_data" / "commodity_node_co2.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "commodity" not in df.columns or "node" not in df.columns:
        return set()
    return {
        (str(r["commodity"]), str(r["node"]))
        for _, r in df.iterrows()
    }


def _load_group_node(work_folder: Path) -> dict[str, set[str]]:
    """Return ``{group: {node, ...}}`` from ``solve_data/group_node.csv``.

    Missing file → empty dict; callers fall back to "attribute to every
    CO2 group" (conservative over-count when multiple groups share a
    node).
    """
    path = work_folder / "solve_data" / "group_node.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "group" not in df.columns or "node" not in df.columns:
        return {}
    out: dict[str, set[str]] = {}
    for _, r in df.iterrows():
        out.setdefault(str(r["group"]), set()).add(str(r["node"]))
    return out


def _load_process_source_sink_partition(
    work_folder: Path,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    """Return ``(noEff, eff)`` sets of ``(process, source, sink)``.

    Both dumps come from phase-1 printfs next to
    ``process_source_sink.csv``.  Absent files → empty sets (older
    workdirs pre-dating the dump — writer then conservatively
    under-reports emissions for that branch).
    """
    noeff: set[tuple[str, str, str]] = set()
    eff: set[tuple[str, str, str]] = set()
    ne_path = work_folder / "solve_data" / "process_source_sink_noEff.csv"
    if ne_path.exists():
        df = pd.read_csv(ne_path)
        if not df.empty and {"process", "source", "sink"}.issubset(df.columns):
            noeff = {
                (str(r["process"]), str(r["source"]), str(r["sink"]))
                for _, r in df.iterrows()
            }
    eff_path = work_folder / "solve_data" / "process_source_sink_eff.csv"
    if eff_path.exists():
        df = pd.read_csv(eff_path)
        if not df.empty and {"process", "source", "sink"}.issubset(df.columns):
            eff = {
                (str(r["process"]), str(r["source"]), str(r["sink"]))
                for _, r in df.iterrows()
            }
    return noeff, eff


def _load_entity_unitsize(work_folder: Path) -> dict[str, float]:
    """Return ``{entity: unitsize}`` from ``input/p_entity_unitsize.csv``.

    Wide CSV: header = entity names, second row labelled ``value`` with
    per-entity values.  Mirrors ``handoff_writers._load_unitsize``.
    """
    path = work_folder / "input" / "p_entity_unitsize.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0)
    if "value" not in df.index:
        return {}
    row = df.loc["value"]
    out: dict[str, float] = {}
    for e, v in row.items():
        try:
            out[str(e)] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def _load_rp_cost_weight(
    work_folder: Path,
) -> dict[tuple[str, str], float]:
    """Return ``{(period, time): rp_cost_weight}`` from
    ``solve_data/p_rp_cost_weight.csv``.  Absent/empty → empty dict;
    callers default to 1.0 at lookup time (matches the mod's default).
    """
    path = work_folder / "solve_data" / "p_rp_cost_weight.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    value_cols = [c for c in df.columns if c not in ("solve", "period", "time")]
    if not value_cols:
        return {}
    v = value_cols[0]
    out: dict[tuple[str, str], float] = {}
    for _, r in df.iterrows():
        try:
            out[(str(r["period"]), str(r["time"]))] = float(r[v])
        except (ValueError, TypeError):
            continue
    return out


def _load_pdtProcess_slope(
    work_folder: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Return ``{(period, time): {process: slope}}`` from
    ``solve_data/pdtProcess_slope.csv``.  Wide format: cols after
    (solve, period, time) are process names.  Missing → empty."""
    path = work_folder / "solve_data" / "pdtProcess_slope.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    idx_cols = ["solve", "period", "time"]
    if not all(c in df.columns for c in idx_cols):
        return {}
    process_cols = [c for c in df.columns if c not in idx_cols]
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (str(row["period"]), str(row["time"]))
        out[key] = {
            p: float(row[p])
            for p in process_cols
            if pd.notna(row[p])
        }
    return out


def _load_process_flow_coefficient_wide(
    path: Path,
) -> dict[tuple[str, str], float]:
    """Load a ``p_process_{source,sink}_flow_coefficient.csv`` (wide).

    Layout (3 rows): ``process,<p1>,<p2>,...`` / ``{source|sink},<s1>,
    <s2>,...`` / ``value,<v1>,<v2>,...``.  Returns
    ``{(process, source_or_sink_name): coefficient}``.  Missing file →
    empty dict (callers default to 1.0 at lookup time).
    """
    if not path.exists():
        return {}
    with open(path) as f:
        lines = [ln.rstrip("\n").split(",") for ln in f if ln.strip()]
    if len(lines) < 3:
        return {}
    processes = lines[0][1:]
    keys = lines[1][1:]
    vals = lines[2][1:]
    out: dict[tuple[str, str], float] = {}
    for p, k, v in zip(processes, keys, vals):
        try:
            out[(str(p), str(k))] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def _load_process_unit_set(work_folder: Path) -> set[str]:
    """Return ``{process}`` from ``solve_data/process_unit.csv``.

    Matches the mod's ``if p in process_unit then ... else 1`` guard in
    the ``_eff`` branch of the CO2 LHS.
    """
    path = work_folder / "solve_data" / "process_unit.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "process" not in df.columns:
        return set()
    return {str(p) for p in df["process"]}


def _load_process_min_load_eff_set(work_folder: Path) -> set[str]:
    """Return processes using ``min_load_efficiency`` (deferred branch).

    Reads ``solve_data/process__ct_method.csv``.  When a process in this
    set appears as a CO2 source in the current model, the writer emits
    a warning and under-reports tonnes for the min-load-section part —
    conservative (LP cap slightly over-enforces).
    """
    path = work_folder / "solve_data" / "process__ct_method.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "method" not in df.columns:
        return set()
    rows = df.loc[df["method"].astype(str) == "min_load_efficiency"]
    return {str(p) for p in rows.get("process", [])}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _horizon_and_realized_hours(
    step_duration: dict[tuple[str, str], float],
    realized_set: set[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-period ``(horizon_hours, realized_hours)``.

    Horizon hours sum ``step_duration`` over every (d, t) this roll's
    dt loaded from ``steps_in_use.csv``.  Realized hours sum only over
    the (d, t) pairs listed in ``realized_dispatch.csv``.
    """
    horizon: dict[str, float] = {}
    realized: dict[str, float] = {}
    for (d, t), dur in step_duration.items():
        horizon[d] = horizon.get(d, 0.0) + dur
        if (d, t) in realized_set:
            realized[d] = realized.get(d, 0.0) + dur
    # Make sure every horizon period has a realized entry (possibly 0.0)
    # so iteration below never hits a missing key.
    for d in horizon:
        realized.setdefault(d, 0.0)
    return horizon, realized


def _v_trade_realized_mwh_this_roll(
    v_trade_df: pd.DataFrame,
    unitsize: dict[str, float],
    horizon_hours: dict[str, float],
    realized_hours: dict[str, float],
    ladder_commodities: set[str],
) -> dict[tuple[str, int, str], float]:
    """Sum realized MWh contributions for THIS roll per (commodity, tier, period).

    Uniform-split assumption: v_trade's period-level decision attributes
    proportionally to realized hours of the roll's view of the period.
    ``v_trade_df`` row index is ``(solve, period)``; columns are
    ``(commodity, node, tier)``.  Empty frame → empty dict.
    """
    out: dict[tuple[str, int, str], float] = {}
    if v_trade_df.empty:
        return out
    for row_key, row in v_trade_df.iterrows():
        period = str(row_key[-1]) if isinstance(row_key, tuple) else str(row_key)
        hz = horizon_hours.get(period, 0.0)
        if hz <= 0.0:
            continue
        rz = realized_hours.get(period, 0.0)
        if rz <= 0.0:
            # This roll didn't realize any of period d — v_trade in d is
            # lookahead only, so no MWh to accumulate.
            continue
        fraction = rz / hz
        for col_key, val in row.items():
            if pd.isna(val) or val == 0.0:
                continue
            if not isinstance(col_key, tuple) or len(col_key) < 3:
                continue
            commodity = str(col_key[0])
            if commodity not in ladder_commodities:
                continue
            try:
                tier = int(col_key[-1])
            except (ValueError, TypeError):
                continue
            us = unitsize.get(commodity, 1.0)
            key = (commodity, tier, period)
            out[key] = out.get(key, 0.0) + float(val) * us * fraction
    return out


def write_ladder_rolling_accumulators(
    h: "highspy.Highs",
    *,
    solve_name: str,
    work_folder: Path,
) -> list[Path]:
    """Write both ladder rolling accumulator CSVs.

    Uses uniform-split assumption: realized MWh of period d in this roll =
    v_trade_k[c, n, d, i] × unitsize × (realized_sim_hours_d / horizon_sim_hours_d)
    — i.e., the LP's period-level v_trade decision is assumed to distribute
    uniformly across the horizon's hours, and the realized portion is a
    proportional slice.

    Returns [path_to_cum_realized_mwh_csv, path_to_cum_sim_hours_csv].
    """
    mwh_path = work_folder / "solve_data" / "ladder_cum_realized_mwh.csv"
    hrs_path = work_folder / "solve_data" / "ladder_cum_sim_hours.csv"

    ladder_tiers = _load_finite_ladder_tiers(work_folder)
    ladder_commodities = {c for (c, _i) in ladder_tiers}

    # Always write at least header-only CSVs so the mod's table-data-in
    # blocks always find the files (mod defaults cover the empty case).
    if not ladder_commodities:
        mwh_path.write_text(
            "commodity,tier,period,p_ladder_cum_realized_mwh\n"
        )
        # Still need cum_sim_hours — even without ladders the file must
        # exist.  Zero rows → mod default 0 → f_d_k[d] = horizon / (share
        # * 8760), which is 1.0 on a full single solve.
        hrs_path.write_text("period,p_ladder_cum_sim_hours\n")
        return [mwh_path, hrs_path]

    first_solve = _is_first_solve(work_folder)
    prior_mwh = {} if first_solve else _load_prior_cum_realized_mwh(mwh_path)
    prior_hrs = {} if first_solve else _load_prior_cum_sim_hours(hrs_path)

    step_duration = _load_step_duration(work_folder)
    realized_set = _load_realized_set(
        work_folder / "solve_data" / "realized_dispatch.csv"
    ) or set()
    horizon_hours, realized_hours = _horizon_and_realized_hours(
        step_duration, realized_set
    )

    unitsize = _load_commodity_unitsize(work_folder)

    v_trade_df = extract_variable(
        h,
        "v_trade",
        ("commodity", "node"),
        solve_name=solve_name,
        has_time=False,
        trailing_col_names=("tier",),
    )
    this_roll_mwh = _v_trade_realized_mwh_this_roll(
        v_trade_df, unitsize, horizon_hours, realized_hours,
        ladder_commodities,
    )

    # --- Build updated cum_realized_mwh table -----------------------------
    # Seed with prior rows, then bump by this roll's contributions.  Only
    # emit rows for (c, i) actually in ladder_tiers (finite tiers of
    # ladder commodities) — infinite / non-ladder tiers don't need a row.
    combined: dict[tuple[str, int, str], float] = dict(prior_mwh)
    for key, val in this_roll_mwh.items():
        # Only write rows for finite tiers; this_roll_mwh is already
        # filtered to ladder commodities via ladder_commodities.
        if (key[0], key[1]) not in ladder_tiers:
            continue
        combined[key] = combined.get(key, 0.0) + val

    mwh_rows = sorted(
        ((c, i, d, v) for (c, i, d), v in combined.items()
         if (c, i) in ladder_tiers),
        key=lambda r: (r[0], r[1], r[2]),
    )
    mwh_df = pd.DataFrame(
        mwh_rows,
        columns=["commodity", "tier", "period", "p_ladder_cum_realized_mwh"],
    )
    mwh_df.to_csv(mwh_path, index=False, float_format="%.8g")

    # --- Build updated cum_sim_hours table --------------------------------
    # Prior + realized hours from THIS roll.  Include every period
    # appearing in either.
    updated_hrs: dict[str, float] = dict(prior_hrs)
    for d, hrs in realized_hours.items():
        updated_hrs[d] = updated_hrs.get(d, 0.0) + hrs
    hrs_rows = sorted(updated_hrs.items(), key=lambda kv: kv[0])
    hrs_df = pd.DataFrame(
        hrs_rows, columns=["period", "p_ladder_cum_sim_hours"],
    )
    hrs_df.to_csv(hrs_path, index=False, float_format="%.8g")

    _logger.info(
        "wrote %s (%d rows) and %s (%d rows)",
        mwh_path, len(mwh_df), hrs_path, len(hrs_df),
    )
    return [mwh_path, hrs_path]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _attribute_groups(
    node: str,
    co2_groups: set[str],
    group_node: dict[str, set[str]],
) -> set[str]:
    """Return the CO2 groups that own ``node``.

    Falls back to ``co2_groups`` when ``group_node`` is empty (no dump
    available — conservative: the writer over-attributes, cap binds
    slightly tighter).  When ``group_node`` is present, returns the
    intersection with ``co2_groups`` (only groups that both carry the
    cap AND own this node contribute).
    """
    if not group_node:
        return co2_groups
    return {g for g in co2_groups if node in group_node.get(g, set())}


def _co2_tonnes_this_roll(
    v_flow_df: pd.DataFrame,
    *,
    realized_set: set[tuple[str, str]],
    co2_groups: set[str],
    co2_content: dict[str, float],
    commodity_node_co2: set[tuple[str, str]],
    group_node: dict[str, set[str]],
    noeff_set: set[tuple[str, str, str]],
    eff_set: set[tuple[str, str, str]],
    process_unit_set: set[str],
    process_source_flow_coeff: dict[tuple[str, str], float],
    process_sink_flow_coeff: dict[tuple[str, str], float],
    step_duration: dict[tuple[str, str], float],
    rp_weight: dict[tuple[str, str], float],
    slope: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str], float]:
    """Compute realized sim-window tonnes added by THIS roll per (group, period).

    Mirrors the mod's ``co2_max_total`` LHS exactly for the no-eff and
    simple-eff branches.  Min-load-efficiency and online-section
    corrections are intentionally omitted — callers log a warning when
    any contributing process uses those methods.  Group/node filtering:
    we use ``commodity_node_co2`` (the (c, n) CO2-priced pairs) and
    attribute to every CO2 group — the LHS filter is ``(g, n) in
    group_node``; since the mod-side ``group_commodity_node_period_co2_total``
    set already enforces that, and we don't have a group→node map
    dumped Python-side, we attribute equally to all CO2 groups per node
    — this matches the typical single-group-per-CO2-node case.  When
    multiple groups share the same node the writer over-attributes
    (conservative — the cap binds tighter); a future follow-up can add
    an explicit ``group_node.csv`` dump if needed.

    Returns ``{(group, period): realized_tonnes_in_window}``.  Units:
    tonnes (post-/1000 scaling).  Key absent → zero contribution.
    """
    out: dict[tuple[str, str], float] = {}
    if not co2_groups or not commodity_node_co2 or v_flow_df.empty:
        return out

    # Build a lookup by sink-node → (c, n) for incoming flows (emissions)
    # and by source-node → (c, n) for outgoing flows (removals).
    nodes_with_co2 = {n for (_c, n) in commodity_node_co2}
    commodity_by_node = {n: c for (c, n) in commodity_node_co2}

    # Iterate the wide DataFrame.  Rows are (solve, period, time); cols
    # are (process, source, sink).
    for row_key, row in v_flow_df.iterrows():
        if not isinstance(row_key, tuple):
            continue
        # Row index may be (solve, period, time) or (solve, period).
        if len(row_key) == 3:
            _, period, time = row_key
        elif len(row_key) == 2:
            _, period = row_key
            time = ""
        else:
            continue
        period_s = str(period)
        time_s = str(time)
        if (period_s, time_s) not in realized_set:
            continue
        dur = step_duration.get((period_s, time_s))
        if dur is None or dur <= 0:
            continue
        rpw = rp_weight.get((period_s, time_s), 1.0)
        slope_pt = slope.get((period_s, time_s), {})

        for col_key, val in row.items():
            if pd.isna(val) or val == 0.0:
                continue
            if not isinstance(col_key, tuple) or len(col_key) < 3:
                continue
            p = str(col_key[0])
            source = str(col_key[1])
            sink = str(col_key[2])
            us = 1.0  # entity_unitsize looked up below
            # Branch discrimination: only process flows whose source OR
            # sink is a CO2 commodity_node contribute.
            # Case A: sink in commodity_node_co2 → emissions into sink (a
            # node), from process source n=sink, sign = + (CO2 increase).
            # In the mod this is ``sum {(p, n, sink) in process_source_sink_*}
            # v_flow[p, n, sink, d, t]`` where n is the CO2 node.
            # Here the column is (process, source, sink); the CO2 node is
            # the SOURCE of the flow if flow is "out of n into the process"
            # (what the mod treats as a CO2 increase — source burns
            # commodity).  Re-read the mod: "process_source_sink_noEff
            # ... v_flow[p, n, sink, d, t]" where n replaces ``source`` =
            # the commodity node.  So: emissions when (p, source=n, sink)
            # has n in nodes_with_co2.
            #
            # Removals: "sum {(p, source, n) in process_source_sink}
            # v_flow[p, source, n, d, t]" where n is the CO2 node.  Here
            # n is the SINK of the flow.
            emission_node = source if source in nodes_with_co2 else None
            removal_node = sink if sink in nodes_with_co2 else None

            if emission_node is None and removal_node is None:
                continue

            if emission_node is not None:
                c = commodity_by_node.get(emission_node)
                if c is None:
                    continue
                content = co2_content.get(c, 0.0)
                if content == 0.0:
                    continue
                # Discriminate noEff vs eff branch.
                pss = (p, source, sink)
                if pss in noeff_set:
                    flow_piece = float(val) * us * dur * rpw
                elif pss in eff_set:
                    sl = slope_pt.get(p, 1.0)
                    if p in process_unit_set:
                        fc_sink = process_sink_flow_coeff.get((p, sink), 1.0)
                        fc_source = process_source_flow_coeff.get((p, source), 1.0)
                        coeff = fc_sink / fc_source if fc_source else 1.0
                    else:
                        coeff = 1.0
                    flow_piece = float(val) * us * dur * rpw * sl * coeff
                else:
                    # Not categorised: skip (conservative).
                    continue
                contribution = content / 1000.0 * flow_piece
                groups_for_node = _attribute_groups(
                    emission_node, co2_groups, group_node,
                )
                for g in groups_for_node:
                    key = (g, period_s)
                    out[key] = out.get(key, 0.0) + contribution

            if removal_node is not None:
                c = commodity_by_node.get(removal_node)
                if c is None:
                    continue
                content = co2_content.get(c, 0.0)
                if content == 0.0:
                    continue
                # Removals: sign = negative; mod uses process_source_sink
                # (i.e. all, not just noEff/eff) for the subtraction.
                flow_piece = float(val) * us * dur * rpw
                contribution = -content / 1000.0 * flow_piece
                groups_for_node = _attribute_groups(
                    removal_node, co2_groups, group_node,
                )
                for g in groups_for_node:
                    key = (g, period_s)
                    out[key] = out.get(key, 0.0) + contribution

    return out


def write_co2_rolling_accumulators(
    h: "highspy.Highs",
    *,
    solve_name: str,
    work_folder: Path,
) -> list[Path]:
    """Write ``solve_data/co2_cum_realized_tonnes.csv`` — the CO2 cap
    rolling-window realized-emissions accumulator.

    Mirrors :func:`write_ladder_rolling_accumulators` for the
    ``co2_max_total`` cap: sums prior-roll tonnes with this roll's
    realized (d, t) contributions, writes one row per ``(group, period)``.

    Simplified MVP — no-eff + simple-eff branches only.  Min-load-
    efficiency and online-section corrections are deferred; if a model
    uses them in a priced-commodity CO2 node the writer logs a warning
    and under-reports realized emissions (conservative — LP cap
    slightly over-enforces).

    Returns ``[path_to_co2_cum_realized_tonnes_csv]``.
    """
    out_path = work_folder / "solve_data" / "co2_cum_realized_tonnes.csv"

    co2_groups = _load_co2_max_total_groups(work_folder)
    # No active CO2 cap → header-only CSV (mod default 0 leaves the
    # cap inactive).  Still emit the file so the mod's table-data-in
    # block always has a file to open.
    if not co2_groups:
        out_path.write_text("group,period,p_co2_cum_realized_tonnes\n")
        return [out_path]

    first_solve = _is_first_solve(work_folder)
    prior = (
        {} if first_solve
        else _load_prior_co2_cum_realized_tonnes(out_path)
    )

    co2_content = _load_commodity_co2_content(work_folder)
    commodity_node_co2 = _load_commodity_node_co2(work_folder)
    group_node = _load_group_node(work_folder)
    noeff_set, eff_set = _load_process_source_sink_partition(work_folder)
    process_unit_set = _load_process_unit_set(work_folder)
    process_source_flow_coeff = _load_process_flow_coefficient_wide(
        work_folder / "solve_data" / "p_process_source_flow_coefficient.csv"
    )
    process_sink_flow_coeff = _load_process_flow_coefficient_wide(
        work_folder / "solve_data" / "p_process_sink_flow_coefficient.csv"
    )
    step_duration = _load_step_duration(work_folder)
    rp_weight = _load_rp_cost_weight(work_folder)
    slope = _load_pdtProcess_slope(work_folder)
    realized_set = _load_realized_set(
        work_folder / "solve_data" / "realized_dispatch.csv"
    ) or set()

    # Warn about deferred branches in priced-CO2 models.
    min_load_procs = _load_process_min_load_eff_set(work_folder)
    if min_load_procs and commodity_node_co2:
        # Any min-load-efficiency process whose source is a CO2 node?
        # We don't have an easy (process → CO2 node) map here; warn
        # liberally whenever both exist.  False positives are cheap.
        _logger.warning(
            "write_co2_rolling_accumulators: model uses min_load_efficiency "
            "on %d process(es); MVP under-reports their section-term "
            "emissions (LP cap slightly over-enforces). Affected: %s",
            len(min_load_procs), sorted(min_load_procs),
        )

    v_flow_df = extract_variable(
        h, "v_flow", ("process", "source", "sink"),
        solve_name=solve_name, has_time=True,
    )

    # v_flow is per-entity-flow in MW of process-scale — the mod LHS
    # multiplies by p_entity_unitsize.  Scale the DataFrame here.
    entity_unitsize = _load_entity_unitsize(work_folder)
    if not v_flow_df.empty and entity_unitsize:
        # Multiply each column by the corresponding process unitsize.
        def _us_for_col(col: tuple[str, ...]) -> float:
            p = col[0] if isinstance(col, tuple) else col
            return entity_unitsize.get(str(p), 1.0)

        col_scales = [
            _us_for_col(c) for c in v_flow_df.columns
        ]
        if col_scales:
            v_flow_df = v_flow_df.mul(col_scales, axis=1)

    this_roll = _co2_tonnes_this_roll(
        v_flow_df,
        realized_set=realized_set,
        co2_groups=co2_groups,
        co2_content=co2_content,
        commodity_node_co2=commodity_node_co2,
        group_node=group_node,
        noeff_set=noeff_set,
        eff_set=eff_set,
        process_unit_set=process_unit_set,
        process_source_flow_coeff=process_source_flow_coeff,
        process_sink_flow_coeff=process_sink_flow_coeff,
        step_duration=step_duration,
        rp_weight=rp_weight,
        slope=slope,
    )

    combined: dict[tuple[str, str], float] = dict(prior)
    for key, val in this_roll.items():
        combined[key] = combined.get(key, 0.0) + val

    rows = sorted(
        ((g, d, v) for (g, d), v in combined.items()),
        key=lambda r: (r[0], r[1]),
    )
    df = pd.DataFrame(
        rows, columns=["group", "period", "p_co2_cum_realized_tonnes"],
    )
    df.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info(
        "wrote %s (%d rows)", out_path, len(df),
    )
    return [out_path]


def write_cumulative_handoffs(
    h: "highspy.Highs",
    *,
    solve_name: str,
    work_folder: Path,
) -> list[Path]:
    """Write every cumulative-quota handoff CSV.

    Parallels :func:`flextool.process_outputs.handoff_writers.write_all_handoffs`.
    Currently covers:

    * the ladder rolling accumulators (v_trade-based per-period
      cumulative/annual cap partitions), and
    * the CO2 rolling accumulator (v_flow-based per-period
      ``co2_max_total`` cap partition).
    """
    written: list[Path] = []
    try:
        written.extend(
            write_ladder_rolling_accumulators(
                h, solve_name=solve_name, work_folder=work_folder,
            )
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "write_ladder_rolling_accumulators failed: %s", exc
        )
    try:
        written.extend(
            write_co2_rolling_accumulators(
                h, solve_name=solve_name, work_folder=work_folder,
            )
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "write_co2_rolling_accumulators failed: %s", exc
        )
    return written
