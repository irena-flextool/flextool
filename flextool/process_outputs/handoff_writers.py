"""
Solve-to-solve handoff CSV writers.

These writers produce the six CSVs that ``flextool.mod`` reads at the
START of each solve to receive state from the PREVIOUS solve.  Today
those files are written by the second ``glpsol`` invocation (phase 3)
which re-reads the model and the ``.sol`` solution; the goal here is to
produce them directly from the ``Highs`` instance + the input/parameter
data, so phase 3 can eventually be retired.

The six handoff files (per ``ARCHITECTURE.md`` "Solve-to-solve
handoff"):

    1. solve_data/p_entity_period_existing_capacity.csv
    2. solve_data/p_entity_divested.csv
    3. solve_data/fix_storage_quantity.csv
    4. solve_data/fix_storage_price.csv
    5. solve_data/fix_storage_usage.csv
    6. solve_data/p_roll_continue_state.csv

All six are implemented.  #5 (``fix_storage_usage``) uses a simplified
``v_flow × unitsize`` formula that is exact for ``method_nvar`` and for
``method_1var_per_way`` with trivial coefficients / no min_load_efficiency
— enough for the typical battery / inverter topology.  See
:func:`write_fix_storage_usage` for the caveat on more exotic process
methods.

Current sourcing strategy (transitional)
----------------------------------------
The writers source SOLVER values directly from ``highspy.Highs``:

* variable values via :func:`extract_variable` (already used by
  ``read_highs_solution.py``).
* constraint duals — same machinery, with ``source="row_dual"``.

PARAMETER values are read from ``output_raw/`` — i.e. from what
``glpsol`` phase 3 just wrote.  This is intentional for the PoC:

* These CSVs are pure parameter snapshots (input data after model-side
  derivations).  Reading them is cheaper than re-implementing the GMPL
  derivations in Python.
* Phase 3 currently runs anyway, so the files are guaranteed to exist
  when our writers execute (we wire after phase 3).
* Once phase 3 is dropped, the same params can be sourced from
  ``input/`` + Python-side derivation — a separable follow-up.

Hooking into the solve loop
---------------------------
Call :func:`write_all_handoffs` from ``solver_runner._run_highs_or_cplex``
AFTER phase 3 completes.  Two pathways then write the same files; ours
runs second and overwrites with values computed from the live solver
state.  Tests assert byte-equivalence (within format precision).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from flextool.process_outputs.read_highs_solution import (
    _load_realized_periods,
    _load_realized_set,
    extract_variable,
)

if TYPE_CHECKING:
    import highspy

_logger = logging.getLogger(__name__)

# Same constant that the model uses; multiply duals by this to undo the
# 1e-6 scaling baked into the objective.
_INV_SCALE_THE_OBJECTIVE = 1e6


# ---------------------------------------------------------------------------
# Parameter / set loaders
#
# All parameter CSVs in ``output_raw/`` are written by glpsol phase 3.
# These small helpers normalise their varied shapes into Python dicts.
# ---------------------------------------------------------------------------


def _load_unitsize(work_folder: Path) -> dict[str, float]:
    """Return ``{entity: unitsize}`` from ``output_raw/p_entity_unitsize.csv``.

    The CSV is wide: header row = entity names, second row labelled
    ``value`` = the unitsize per entity.
    """
    path = work_folder / "output_raw" / "p_entity_unitsize.csv"
    df = pd.read_csv(path, index_col=0)
    return df.loc["value"].astype(float).to_dict()


def _load_pre_existing(work_folder: Path) -> dict[tuple[str, str], float]:
    """Return ``{(period, entity): value}`` from ``p_entity_pre_existing.csv``.

    CSV layout: rows indexed by (solve, period); entity columns.  Solve
    level is collapsed — the same value applies for any solve.
    """
    path = work_folder / "output_raw" / "p_entity_pre_existing.csv"
    df = pd.read_csv(path, index_col=[0, 1])
    if df.empty:
        return {}
    df = df.droplevel(0)  # drop solve level → period only
    out: dict[tuple[str, str], float] = {}
    for period, row in df.iterrows():
        for entity, value in row.items():
            if pd.notna(value):
                out[(str(period), str(entity))] = float(value)
    return out


def _load_prior_existing(
    work_folder: Path,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Return prior ``p_entity_period_existing_capacity`` + ``_invested_capacity``.

    Source: the previous solve's ``solve_data/p_entity_period_existing_capacity.csv``.
    For the first solve this file does not yet exist — return empty dicts.
    """
    path = work_folder / "solve_data" / "p_entity_period_existing_capacity.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    if df.empty:
        return {}, {}
    existing: dict[tuple[str, str], float] = {}
    invested: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        key = (str(row["entity"]), str(row["period"]))
        existing[key] = float(row["p_entity_period_existing_capacity"])
        invested[key] = float(row["p_entity_period_invested_capacity"])
    return existing, invested


def _load_prior_divested(work_folder: Path) -> dict[str, float]:
    """Return ``{entity: cumulative_divested}`` from prior solve, or empty."""
    path = work_folder / "solve_data" / "p_entity_divested.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "p_entity_divested" not in df.columns:
        return {}
    return {
        str(row["entity"]): float(row["p_entity_divested"])
        for _, row in df.iterrows()
    }


def _load_storage_fix_methods(
    work_folder: Path, target_method: str,
) -> set[str]:
    """Return ``{node, ...}`` whose nested fix method == *target_method*.

    Reads ``input/node__storage_nested_fix_method.csv`` (long format
    ``node, storage_nested_fix_method``).  Empty when the file doesn't
    exist or no node has that method.
    """
    path = work_folder / "input" / "node__storage_nested_fix_method.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(
        df.loc[df["storage_nested_fix_method"] == target_method, "node"]
        .astype(str)
        .tolist()
    )


def _load_node_state(work_folder: Path) -> set[str]:
    """Return the set of nodes that maintain a state variable."""
    path = work_folder / "input" / "nodeState.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df["nodeState"].astype(str).tolist())


def _load_entity(work_folder: Path) -> set[str]:
    """Return the full ``entity`` set from ``output_raw/set_entity.csv``."""
    path = work_folder / "output_raw" / "set_entity.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        return set()
    return set(df.iloc[:, 0].astype(str).tolist())


def _load_entity_divest(work_folder: Path) -> set[str]:
    """Return ``entityDivest`` — entities allowed to divest.

    Sourced from ``output_raw/set_entityDivest.csv`` (phase 3 dump);
    matches the ``setof {(e,m) in entity__invest_method : m not in
    divest_method_not_allowed} (e)`` derivation in the model.
    """
    path = work_folder / "output_raw" / "set_entityDivest.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        return set()
    return set(df.iloc[:, 0].astype(str).tolist())


def _load_realized_period_time_last(
    work_folder: Path,
) -> list[tuple[str, str]]:
    """Return one ``(period, time)`` pair per realized period — the LAST
    realized timestep within that period.

    Derived from ``solve_data/realized_dispatch.csv`` (period, step) by
    taking the last row for each period (rows are written in dispatch
    order).
    """
    path = work_folder / "solve_data" / "realized_dispatch.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    time_col = "step" if "step" in df.columns else "time"
    return [
        (str(period), str(group[time_col].iloc[-1]))
        for period, group in df.groupby("period", sort=False)
    ]


def _load_complete_period_share_of_year(
    work_folder: Path,
) -> dict[str, float]:
    """Return ``{period: share_of_year}`` from
    ``output_raw/complete_period_share_of_year.csv``."""
    path = work_folder / "output_raw" / "complete_period_share_of_year.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    # Expected layout: solve,period,value or period,value
    period_col = "period"
    value_col = [c for c in df.columns if c not in ("solve", "period")][0]
    return dict(zip(df[period_col].astype(str), df[value_col].astype(float)))


def _load_inflation_factor_operations_yearly(
    work_folder: Path,
) -> dict[str, float]:
    """Return ``{period: inflation_factor}`` from
    ``output_raw/p_inflation_factor_operations_yearly.csv``."""
    path = work_folder / "output_raw" / "p_inflation_factor_operations_yearly.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    period_col = "period"
    value_col = [c for c in df.columns if c not in ("solve", "period")][0]
    return dict(zip(df[period_col].astype(str), df[value_col].astype(float)))


def _load_step_duration(work_folder: Path) -> dict[tuple[str, str], float]:
    """Return ``{(period, time): step_duration}`` from ``solve_data/steps_in_use.csv``."""
    path = work_folder / "solve_data" / "steps_in_use.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    time_col = "step" if "step" in df.columns else "time"
    dur_col = "step_duration"
    return {
        (str(p), str(t)): float(d)
        for p, t, d in zip(df["period"], df[time_col], df[dur_col])
    }


def _is_first_solve(work_folder: Path) -> bool:
    """True iff this is the first solve in the model.

    Source: ``input/p_model.csv`` (long ``modelParam,p_model`` pairs)
    written by ``write_solve_status``.  Same flag the GMPL writer reads
    via ``p_model['solveFirst']``.
    """
    path = work_folder / "input" / "p_model.csv"
    if not path.exists():
        return True
    df = pd.read_csv(path)
    matches = df.loc[df["modelParam"] == "solveFirst", "p_model"]
    if matches.empty:
        return True
    return bool(int(matches.iloc[0]))


# ---------------------------------------------------------------------------
# Handoff writers
# ---------------------------------------------------------------------------


def write_p_entity_divested(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> Path:
    """Write ``solve_data/p_entity_divested.csv`` from v_divest + prior.

    ``cumulative_divested[e] = prior_divested[e] + sum_d v_divest[e,d] * unitsize[e]``
    over every divest period (sum_d means all (e,d) declared in ed_divest).
    """
    out_path = work_folder / "solve_data" / "p_entity_divested.csv"

    entities = _load_entity_divest(work_folder)
    unitsize = _load_unitsize(work_folder)
    prior = {} if _is_first_solve(work_folder) else _load_prior_divested(work_folder)
    divest_df = extract_variable(
        h, "v_divest", ("entity",), solve_name=solve_name, has_time=False,
    )

    # divest_df: rows (solve, period), columns = entity.  Sum across periods.
    if divest_df.empty:
        per_entity_sum: dict[str, float] = {}
    else:
        per_entity_sum = divest_df.sum(axis=0).to_dict()

    rows = []
    for e in sorted(entities):
        cumulative = prior.get(e, 0.0) + per_entity_sum.get(e, 0.0) * unitsize.get(e, 1.0)
        rows.append((e, cumulative))

    out = pd.DataFrame(rows, columns=["entity", "p_entity_divested"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_fix_storage_quantity(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> Path:
    """Write ``solve_data/fix_storage_quantity.csv`` from v_state for fix_quantity nodes.

    ``v_state[n, d, t].val * p_entity_unitsize[n]`` for every
    ``(n, d, t)`` with ``n in fix_quantity_nodes`` and
    ``(d, t) in dt_fix_storage_timesteps``.
    """
    out_path = work_folder / "solve_data" / "fix_storage_quantity.csv"

    target_nodes = _load_storage_fix_methods(work_folder, "fix_quantity")
    fix_steps_path = work_folder / "solve_data" / "fix_storage_timesteps.csv"
    fix_steps = _load_realized_set(fix_steps_path)
    unitsize = _load_unitsize(work_folder)
    state_df = extract_variable(h, "v_state", ("node",), solve_name=solve_name)

    rows: list[tuple[str, str, str, float]] = []
    if not state_df.empty and target_nodes:
        for (_solve, period, time), row in state_df.iterrows():
            if fix_steps is not None and (period, time) not in fix_steps:
                continue
            for node in row.index:
                if node not in target_nodes:
                    continue
                rows.append((period, time, node, float(row[node]) * unitsize.get(node, 1.0)))

    # Match phase-3 semantics: only OVERWRITE the file when this solve
    # actually has fix-storage entries to record.  An empty dispatch
    # sub-solve must NOT clobber rows the upper storage solve wrote
    # earlier (the next solve reads them back).
    if not rows:
        if _is_first_solve(work_folder) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_quantity\n")
            _logger.info("wrote %s (header only — first solve)", out_path)
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    out = pd.DataFrame(rows, columns=["period", "step", "node", "p_fix_storage_quantity"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_p_roll_continue_state(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> Path:
    """Write ``solve_data/p_roll_continue_state.csv``.

    For every ``n in nodeState`` and the LAST realized timestep of each
    realized period, store ``v_state[n,d,t].val * p_entity_unitsize[n]``.
    Output file has only the latest period's entries (the model writer
    iterates and overwrites; the next read uses the final one).
    """
    out_path = work_folder / "solve_data" / "p_roll_continue_state.csv"

    nodes = _load_node_state(work_folder)
    last_pairs = _load_realized_period_time_last(work_folder)
    unitsize = _load_unitsize(work_folder)
    state_df = extract_variable(h, "v_state", ("node",), solve_name=solve_name)

    rows: list[tuple[str, float]] = []
    if not state_df.empty and nodes and last_pairs:
        # The mod re-opens the file inside the loop so only the LAST
        # period's entries survive.  Replicate by taking the last
        # (period, time) pair.
        last_period, last_time = last_pairs[-1]
        try:
            row = state_df.loc[(solve_name, last_period, last_time)]
            for node in row.index:
                if node not in nodes:
                    continue
                rows.append((node, float(row[node]) * unitsize.get(node, 1.0)))
        except KeyError:
            pass

    # Match phase-3 semantics — the mod only writes this file when both
    # ``nodeState`` and ``realized_period__time_last`` are non-empty.
    # When this solve has nothing to record, do NOT overwrite a prior
    # solve's content.
    if not rows:
        _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    out = pd.DataFrame(rows, columns=["node", "p_roll_continue_state"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_p_entity_period_existing_capacity(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> Path:
    """Write ``solve_data/p_entity_period_existing_capacity.csv``.

    For each (entity, period) in ``ed_history_realized`` ∪ (entity ×
    d_realize_invest), compute::

        existing = (first_solve & period in period_first → p_entity_pre_existing[e,d])
                 + (not first_solve & (e,d) in history → prior_existing[e,d])
                 + ((e,d) in ed_invest & d in d_realize_invest → v_invest[e,d] * unitsize[e])
        invested = (not first_solve & (e,d) in history → prior_invested[e,d])
                 + ((e,d) in ed_invest & d in d_realize_invest → v_invest[e,d] * unitsize[e])

    The phase-3 ``ed_history_realized`` set is read from
    ``output_raw/set_ed_invest.csv`` (entity, period) for the
    "(e,d) in ed_invest" predicate, plus the prior history file's keys.
    """
    out_path = work_folder / "solve_data" / "p_entity_period_existing_capacity.csv"

    unitsize = _load_unitsize(work_folder)
    pre_existing = _load_pre_existing(work_folder)
    prior_existing, prior_invested = (
        ({}, {}) if _is_first_solve(work_folder)
        else _load_prior_existing(work_folder)
    )
    first_solve = _is_first_solve(work_folder)

    invest_df = extract_variable(
        h, "v_invest", ("entity",), solve_name=solve_name, has_time=False,
    )

    # Periods to include in the iteration set.  For the FIRST solve the
    # mod uses ``d_realize_invest ∪ d_fix_storage_period ∪
    # d_realized_period`` (via ``ed_history_realized_first``); for later
    # solves only ``d_realize_invest`` adds new keys (the realized /
    # fix-storage periods only contribute on the first solve).
    realize_invest = _load_realized_periods(
        work_folder / "solve_data" / "realized_invest_periods_of_current_solve.csv"
    ) or set()
    if first_solve:
        realized_periods: set[str] = set()
        rd_path = work_folder / "solve_data" / "realized_dispatch.csv"
        if rd_path.exists():
            realized_periods.update(
                pd.read_csv(rd_path)["period"].astype(str).unique()
            )
        fix_storage_periods: set[str] = set()
        fs_path = work_folder / "solve_data" / "fix_storage_timesteps.csv"
        if fs_path.exists():
            fs_df = pd.read_csv(fs_path)
            if not fs_df.empty:
                fix_storage_periods.update(fs_df["period"].astype(str).unique())
        iteration_periods = realize_invest | realized_periods | fix_storage_periods
    else:
        iteration_periods = set(realize_invest)

    # period_first is written by Python orchestration to solve_data/.
    # Fallback for the (unlikely) case it's missing: take min(realize_invest).
    period_first = _load_realized_periods(
        work_folder / "solve_data" / "period_first.csv"
    ) or ({min(realize_invest)} if realize_invest else set())

    # ed_invest set (entity, period) — phase-3 dump.  CSV layout is
    # ``solve, entity, period`` (3 columns); we drop the solve column.
    ed_invest: set[tuple[str, str]] = set()
    ei_path = work_folder / "output_raw" / "set_ed_invest.csv"
    if ei_path.exists():
        ei_df = pd.read_csv(ei_path)
        if not ei_df.empty and {"entity", "period"}.issubset(ei_df.columns):
            ed_invest = {
                (str(r["entity"]), str(r["period"])) for _, r in ei_df.iterrows()
            }
        elif not ei_df.empty and len(ei_df.columns) >= 2:
            # Fallback for files that don't carry headers
            ed_invest = {
                (str(r.iloc[-2]), str(r.iloc[-1])) for _, r in ei_df.iterrows()
            }

    # Iteration set: ed_history_realized ∪ (entity × d_realize_invest).
    # On the FIRST solve, ed_history_realized = entity × realized_or_invest
    # periods (within the same branch).  On subsequent solves it grows
    # with the prior-history file's keys.  Approximation that matches
    # phase 3 in the common cases the tests cover: union prior keys +
    # all entities × realize_invest.
    iter_keys: set[tuple[str, str]] = set(prior_existing.keys())
    entities = _load_entity(work_folder)
    for e in entities:
        for d in iteration_periods:
            iter_keys.add((e, d))

    rows: list[tuple[str, str, float, float]] = []
    for e, d in sorted(iter_keys):
        existing = 0.0
        invested = 0.0
        if first_solve and d in period_first:
            existing += pre_existing.get((d, e), 0.0)
        if not first_solve:
            existing += prior_existing.get((e, d), 0.0)
            invested += prior_invested.get((e, d), 0.0)
        if (e, d) in ed_invest and d in realize_invest:
            try:
                v = float(invest_df.loc[(solve_name, d), e])
            except KeyError:
                v = 0.0
            existing += v * unitsize.get(e, 1.0)
            invested += v * unitsize.get(e, 1.0)
        rows.append((e, d, existing, invested))

    out = pd.DataFrame(
        rows,
        columns=[
            "entity", "period",
            "p_entity_period_existing_capacity",
            "p_entity_period_invested_capacity",
        ],
    )
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_fix_storage_price(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> Path:
    """Write ``solve_data/fix_storage_price.csv``.

    For each ``n`` with method ``fix_price`` and each
    ``(d, t) in dt_fix_storage_timesteps``::

        price = -nodeBalance_eq[c, n, d, t, ...].dual
                / p_inflation_factor_operations_yearly[d]
                * complete_period_share_of_year[d]
                / scale_the_objective

    The ``nodeBalance_eq`` constraint is indexed by 8 fields
    ``(solve, node, period, time, t_prev, t_prev_within_timeset,
    d_prev, t_prev_within_solve)``.  We accept rows for any value of
    the four "previous" indices — exactly one constraint exists per
    ``(c, n, d, t)`` so the dual is well-defined.
    """
    out_path = work_folder / "solve_data" / "fix_storage_price.csv"

    target_nodes = _load_storage_fix_methods(work_folder, "fix_price")
    fix_steps = _load_realized_set(
        work_folder / "solve_data" / "fix_storage_timesteps.csv"
    )
    inflation = _load_inflation_factor_operations_yearly(work_folder)
    period_share = _load_complete_period_share_of_year(work_folder)

    # Empty short-circuit — most scenarios have no fix_price nodes.
    if not target_nodes or not fix_steps:
        if _is_first_solve(work_folder) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_price\n")
            _logger.info("wrote %s (header only — first solve)", out_path)
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    # nodeBalance_eq has 8 indices.  Use the generic extractor with
    # 6 col_names so the trailing two become the (period, time) row index;
    # then we filter rows to fix_steps and entity-name match to
    # target_nodes (col_names[1] == 'node').
    df = extract_variable(
        h, "nodeBalance_eq",
        col_names=("c", "node", "t_prev", "t_prev_within_timeset",
                   "d_prev", "t_prev_within_solve"),
        solve_name=solve_name,
        has_time=True,
        source="row_dual",
        value_scale=1.0,  # we apply the full transform manually below
    )

    if df.empty:
        # Same preserve-prior semantics as the empty short-circuit above.
        if _is_first_solve(work_folder) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_price\n")
        return out_path

    rows: list[tuple[str, str, str, float]] = []
    # df: row index (solve, period, time); columns MultiIndex
    # (c, node, t_prev, t_prev_within_timeset, d_prev, t_prev_within_solve).
    # The constraint uses ``c in solve_current`` so c == solve_name; pick
    # the only matching slice.  For each (period, time) ∈ fix_steps and
    # each node ∈ target_nodes, sum the duals across the (typically
    # single) "previous" combinations.
    for (_s, period, time), row in df.iterrows():
        if (period, time) not in fix_steps:
            continue
        for col, dual_val in row.items():
            node = col[1]
            if node not in target_nodes:
                continue
            scale = (
                -1.0
                / inflation.get(period, 1.0)
                * period_share.get(period, 1.0)
                / (1.0 / _INV_SCALE_THE_OBJECTIVE)
            )
            rows.append((period, time, node, float(dual_val) * scale))

    if not rows:
        if _is_first_solve(work_folder) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_price\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    out = pd.DataFrame(rows, columns=["period", "step", "node", "p_fix_storage_price"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_fix_storage_usage(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> Path:
    """Write ``solve_data/fix_storage_usage.csv``.

    Net energy flow through each ``fix_usage`` storage node over the
    step::

        usage[n, d, t] = (sum_{p: n is source} v_flow[p, n, *] × unitsize[p]
                        - sum_{p: n is sink}   v_flow[p, *, n] × unitsize[p])
                        × step_duration[d, t]

    This is the *simple* form of the model's ``r_storage_usage_dt`` —
    exact for ``method_nvar`` processes (line 5421 of the model) and
    for ``method_1var_per_way`` processes whose ``pdtProcess_slope == 1``
    and that have no ``min_load_efficiency`` and unit coefficients
    equal to 1.  That covers the typical battery / storage-inverter
    topology; more exotic process methods connected to a ``fix_usage``
    node are NOT reproduced byte-for-byte here — the slope/section
    corrections in the full formula (lines 5389–5429) are skipped.

    If your model pairs ``fix_usage`` nodes with min_load_efficiency or
    non-unity unit coefficients, fall back to ``--use-old-raw-csv`` for
    this file until the full flow derivation is ported over.
    """
    out_path = work_folder / "solve_data" / "fix_storage_usage.csv"

    target_nodes = _load_storage_fix_methods(work_folder, "fix_usage")
    fix_steps = _load_realized_set(
        work_folder / "solve_data" / "fix_storage_timesteps.csv"
    )

    if not target_nodes or not fix_steps:
        if _is_first_solve(work_folder) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_usage\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    unitsize = _load_unitsize(work_folder)
    step_duration = _load_step_duration(work_folder)
    flow_df = extract_variable(
        h, "v_flow", ("process", "source", "sink"), solve_name=solve_name,
    )

    rows: list[tuple[str, str, str, float]] = []
    for (_solve, period, time), row in flow_df.iterrows():
        if (period, time) not in fix_steps:
            continue
        dt = step_duration.get((period, time), 1.0)
        per_node: dict[str, float] = {n: 0.0 for n in target_nodes}
        for (process, source, sink), v in row.items():
            us = unitsize.get(process, 1.0)
            if source in per_node:
                per_node[source] += float(v) * us  # outflow from node
            if sink in per_node:
                per_node[sink] -= float(v) * us  # inflow to node
        for node, net in per_node.items():
            if net == 0.0:
                continue
            rows.append((period, time, node, net * dt))

    if not rows:
        if _is_first_solve(work_folder) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_usage\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    out = pd.DataFrame(rows, columns=["period", "step", "node", "p_fix_storage_usage"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def write_all_handoffs(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
) -> list[Path]:
    """Write all six handoff files.

    Each writer is independent — failure on one is logged and does not
    abort the rest, mirroring :func:`write_all_variables`.
    """
    written: list[Path] = []
    for fn in (
        write_p_entity_divested,
        write_fix_storage_quantity,
        write_p_roll_continue_state,
        write_p_entity_period_existing_capacity,
        write_fix_storage_price,
        write_fix_storage_usage,
    ):
        try:
            written.append(fn(h, solve_name=solve_name, work_folder=work_folder))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("handoff writer %s failed: %s", fn.__name__, exc)
    return written
