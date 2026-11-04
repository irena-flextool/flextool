"""
Solve-to-solve handoff CSV writers — DEBUG OUTPUT ONLY (item 163, c24).

These writers produce five CSVs that MIRROR the engine's real,
in-memory solve-to-solve handoff (``flextool.engine.preprocess.flex_data
.build_handoff_from_solution`` / :class:`flextool.engine.solve._handoff
.SolveHandoff`).  Nothing in the engine reads them back — the mirrors
are written for `--csv-dump` debugging only (owner decision, design_163
.md §11 Q1): ``csv_dump=False`` (every normal run) computes nothing and
writes nothing.

The five handoff files (per ``ARCHITECTURE.md`` "Solve-to-solve
handoff"):

    1. solve_data/p_entity_period_existing_capacity.csv
    2. solve_data/p_entity_divested.csv
    3. solve_data/fix_storage_quantity.csv
    4. solve_data/fix_storage_usage.csv
    5. solve_data/p_roll_continue_state.csv

The nested ``fix_price`` carrier is built in memory by
``flextool.engine.preprocess.flex_data.build_handoff_from_solution`` (the
same price definition as the node price output,
``flextool.engine.output._prices``); there is no CSV for it.
``fix_storage_usage`` uses a simplified ``v_flow × unitsize`` formula that
is exact for ``method_nvar`` and for ``method_1var_per_way`` with trivial
coefficients / no min_load_efficiency — enough for the typical battery /
inverter topology.  See :func:`write_fix_storage_usage` for the caveat on
more exotic process methods.

Sourcing strategy
-----------------
The writers source SOLVER values directly from ``highspy.Highs``:

* variable values via :func:`extract_variable` (already used by
  ``read_highs_solution.py``).
* constraint duals — same machinery, with ``source="row_dual"``.

Every other input comes from the sub-solve's ``flex_data``, ``provider``
and :class:`flextool.engine.output._writer.OutputWriterState` — never
from the work folder.  ``flex_data``, ``is_first_solve`` and
``writer_state`` are REQUIRED on :func:`write_all_handoffs` (a ``None``
value raises ``ValueError`` naming the parameter): production threads
all three on every call, so a caller missing one is a wiring bug, not a
tolerated gap.  When a writer needs the value its own PREVIOUS call
wrote (typically because this is the first solve, so ``prior_handoff``
carries nothing yet), it falls back to ``writer_state`` — see
:class:`OutputWriterState`'s docstring.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from flextool.engine.output import _coarse_flows
from flextool.process_outputs.read_highs_solution import (
    _load_realized_periods,
    extract_variable,
)

if TYPE_CHECKING:
    import highspy
    from flextool.engine.solve._handoff import SolveHandoff
    from flextool.engine.preprocess.flex_data import FlexData
    from flextool.engine.output._writer import OutputWriterState

_logger = logging.getLogger(__name__)

# The six basenames the handoff mirrors write under ``solve_data/`` (five
# handoff files + the vestigial ``period_capacity.csv`` accumulator).  Used
# by ``_bootstrap_dirs`` (item 163 §4, 163-5) to clear a reused work folder's
# leftover mirrors at run start, and by ``tools/output_snapshot.py``.
MIRROR_FILES: tuple[str, ...] = (
    "p_entity_period_existing_capacity.csv",
    "p_entity_divested.csv",
    "fix_storage_quantity.csv",
    "fix_storage_usage.csv",
    "p_roll_continue_state.csv",
    "period_capacity.csv",
)

# ---------------------------------------------------------------------------
# Provider-aware lookup helper — Provider-first, then ``None`` (caller
# falls back to its own disk read).  Provider key uses the
# parent-qualified convention (``"<parent>/<basename>"`` without
# ``.csv``).

def _provider_lookup_df(provider: "object | None", path: "Path | str"):
    """Return the polars frame for *path* sourced from the Provider, or
    ``None`` when the Provider doesn't carry it.
    """
    p = Path(path)
    parent = p.parent.name
    stem = p.stem
    name = f"{parent}/{stem}" if parent else stem
    if provider is not None and provider.has(name):
        return provider.get(name)
    return None


# ---------------------------------------------------------------------------
# Parameter / set loaders
#
# Parameter CSVs come from ``input/`` (write-once) or ``solve_data/``
# (per-solve).  These small helpers normalise their varied shapes into
# Python dicts.
# ---------------------------------------------------------------------------


def _load_pre_existing(*, provider: "object | None" = None) -> dict[tuple[str, str], float]:
    """Return ``{(period, entity): value}`` for ``p_entity_pre_existing``.

    Provider-only (item 163: no work-folder CSV fallback): the canonical
    in-memory frame at ``solve_data/p_entity_pre_existing`` (populated by
    ``_chain_params.emit_p_entity_pre_existing``) is tall —
    ``(entity, period, value)``.  A ``None`` provider or a missing key
    gives ``{}`` — ``write_p_entity_period_existing_capacity`` treats an
    empty pre_existing as "no pre-existing capacity recorded for this
    solve", which is the correct semantics when nothing upstream produced
    it.
    """
    if provider is None or not provider.has("solve_data/p_entity_pre_existing"):
        return {}
    df_pl = provider.get("solve_data/p_entity_pre_existing")
    if df_pl is None or df_pl.is_empty():
        return {}
    # Tall ``(entity, period, value)`` schema — ``value`` is a
    # repr-encoded string per ``_ed_value_frame``.  Convert
    # directly to the ``{(period, entity): float}`` dict the
    # downstream consumers expect.
    out: dict[tuple[str, str], float] = {}
    for row in df_pl.iter_rows(named=True):
        period_v = row.get("period")
        entity_v = row.get("entity")
        value_v = row.get("value")
        if period_v is None or entity_v is None or value_v is None:
            continue
        try:
            out[(str(period_v), str(entity_v))] = float(value_v)
        except (TypeError, ValueError):
            continue
    return out


def _load_prior_existing(
    *, prior_handoff: "SolveHandoff | None" = None,
    writer_state: "OutputWriterState | None" = None,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Return prior ``p_entity_period_existing_capacity`` + ``_invested_capacity``.

    With ``prior_handoff`` populated, read the two dicts from
    ``realized_existing`` / ``realized_invest`` (in-memory, production's
    real source).  Else fall back to ``writer_state.existing`` /
    ``.invested`` — the previous call of THIS writer, within this same
    cascade, wrote them (item 163: no work-folder CSV; see the module
    docstring).  Empty when neither is available (e.g. the first solve).
    """
    if prior_handoff is not None and (
        prior_handoff.realized_existing is not None
        or prior_handoff.realized_invest is not None
    ):
        existing: dict[tuple[str, str], float] = {}
        invested: dict[tuple[str, str], float] = {}
        if prior_handoff.realized_existing is not None:
            for r in prior_handoff.realized_existing.iter_rows(named=True):
                existing[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        if prior_handoff.realized_invest is not None:
            for r in prior_handoff.realized_invest.iter_rows(named=True):
                invested[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        return existing, invested
    if writer_state is not None:
        return dict(writer_state.existing), dict(writer_state.invested)
    return {}, {}


def _load_prior_divested(
    *, prior_handoff: "SolveHandoff | None" = None,
    writer_state: "OutputWriterState | None" = None,
) -> dict[str, float]:
    """Return ``{entity: cumulative_divested}`` from prior solve, or empty.

    Reads from ``prior_handoff.divest_cumulative`` when populated (production's
    real source); else ``writer_state.divested`` — the previous call of THIS
    writer, within this same cascade, wrote it (item 163: no work-folder CSV).
    """
    if prior_handoff is not None and prior_handoff.divest_cumulative is not None:
        return {
            str(r["entity"]): float(r["value"])
            for r in prior_handoff.divest_cumulative.iter_rows(named=True)
        }
    if writer_state is not None:
        return dict(writer_state.divested)
    return {}


# ---------------------------------------------------------------------------
# FlexData-only resolvers (item 163: no work-folder disk fallback)
#
# Each ``_resolve_*`` reads only ``flex_data``.  ``flex_data=None`` or the
# field itself being ``None`` both give the same empty default the old
# disk fallback gave on a missing file — there is no separate disk arm
# left to fall through to.
# ---------------------------------------------------------------------------


def _resolve_unitsize(flex_data: "FlexData | None" = None) -> dict[str, float]:
    """``{entity: unitsize}`` from ``flex_data.p_all_entity_unitsize``."""
    if flex_data is None:
        return {}
    param = getattr(flex_data, "p_all_entity_unitsize", None)
    if param is None:
        return {}
    try:
        f = param.frame
        entity_col = f.columns[0]
        return dict(zip(
            f[entity_col].cast(str).to_list(),
            f["value"].cast(float).to_list(),
        ))
    except Exception:  # noqa: BLE001
        return {}


def _resolve_storage_fix_methods(
    target_method: str, flex_data: "FlexData | None" = None,
) -> set[str]:
    """``{node, ...}`` whose nested fix method == *target_method*, from
    ``flex_data.node__storage_nested_fix_method``."""
    if flex_data is None:
        return set()
    f = getattr(flex_data, "node__storage_nested_fix_method", None)
    if f is None:
        return set()
    try:
        sub = f.filter(pl.col("method") == target_method)
        return set(sub["node"].cast(str).to_list())
    except Exception:  # noqa: BLE001
        return set()


def _resolve_node_state(flex_data: "FlexData | None" = None) -> set[str]:
    """``{node, ...}`` carrying a state variable, from ``flex_data.nodeState``."""
    if flex_data is None:
        return set()
    f = getattr(flex_data, "nodeState", None)
    if f is None:
        return set()
    try:
        col = f.columns[0]
        return set(f[col].cast(str).to_list())
    except Exception:  # noqa: BLE001
        return set()


def _resolve_entity(flex_data: "FlexData | None" = None) -> set[str]:
    """Full entity set (every process + connection + node) — the entity column
    of ``flex_data.p_all_entity_unitsize``."""
    if flex_data is None:
        return set()
    param = getattr(flex_data, "p_all_entity_unitsize", None)
    if param is None:
        return set()
    try:
        f = param.frame
        entity_col = f.columns[0]
        return set(f[entity_col].cast(str).to_list())
    except Exception:  # noqa: BLE001
        return set()


def _resolve_entity_divest(flex_data: "FlexData | None" = None) -> set[str]:
    """``entityDivest`` — entities allowed to divest, from ``flex_data.ed_divest_set``."""
    if flex_data is None:
        return set()
    f = getattr(flex_data, "ed_divest_set", None)
    if f is None:
        return set()
    try:
        entity_col = f.columns[0]
        return set(f[entity_col].cast(str).to_list())
    except Exception:  # noqa: BLE001
        return set()


def _resolve_realized_period_time_last(
    flex_data: "FlexData | None" = None,
) -> list[tuple[str, str]]:
    """One ``(period, time)`` pair per realized period — the LAST step in
    each.  From ``flex_data.realized_dispatch`` (polars frame of
    ``(period, step)``); empty/None means "no realized dispatch this solve"."""
    if flex_data is None:
        return []
    rd = getattr(flex_data, "realized_dispatch", None)
    if rd is None:
        return []
    try:
        cols = rd.columns
        time_col = "step" if "step" in cols else ("time" if "time" in cols else cols[1])
        # Keep LAST step per period, preserving period order of first occurrence.
        seen_order: list[str] = []
        last_per: dict[str, str] = {}
        for p, t in zip(rd["period"].cast(str).to_list(), rd[time_col].cast(str).to_list()):
            if p not in last_per:
                seen_order.append(p)
            last_per[p] = t
        return [(p, last_per[p]) for p in seen_order]
    except Exception:  # noqa: BLE001
        return []


def _resolve_step_duration(flex_data: "FlexData | None" = None) -> dict[tuple[str, str], float]:
    """``{(period, time): step_duration}`` from ``flex_data.p_step_duration``."""
    if flex_data is None:
        return {}
    param = getattr(flex_data, "p_step_duration", None)
    if param is None:
        return {}
    try:
        f = param.frame
        cols = f.columns
        period_col = cols[0]
        time_col = cols[1]
        return {
            (str(p), str(t)): float(v)
            for p, t, v in zip(
                f[period_col].cast(str).to_list(),
                f[time_col].cast(str).to_list(),
                f["value"].cast(float).to_list(),
            )
        }
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Handoff writers
# ---------------------------------------------------------------------------


def write_p_entity_divested(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    writer_state: "OutputWriterState | None" = None,
    csv_dump: bool = False,
) -> Path:
    """Write ``solve_data/p_entity_divested.csv`` from v_divest + prior (debug mirror; item 163: only
    under ``csv_dump=True`` — see the module docstring).

    ``cumulative_divested[e] = prior_divested[e] + sum_d v_divest[e,d] * unitsize[e]``
    over every divest period (sum_d means all (e,d) declared in ed_divest).

    ``prior_handoff`` (when populated) is the real, in-memory ``divest_cumulative`` carrier; else
    ``writer_state.divested`` (the previous call of this writer, within this cascade) — see
    :func:`_load_prior_divested`.  After writing, ``writer_state.divested`` is updated to this call's
    cumulative, for the NEXT call to fall back to.
    """
    out_path = work_folder / "solve_data" / "p_entity_divested.csv"
    if not csv_dump:
        return out_path

    entities = _resolve_entity_divest(flex_data)
    unitsize = _resolve_unitsize(flex_data)
    first = bool(is_first_solve)
    prior = (
        {} if first
        else _load_prior_divested(prior_handoff=prior_handoff, writer_state=writer_state)
    )
    divest_df = extract_variable(
        h, "v_divest", ("entity",), solve_name=solve_name, has_time=False,
    )

    # divest_df: rows (solve, period), columns = entity.  Sum across periods.
    if divest_df.empty:
        per_entity_sum: dict[str, float] = {}
    else:
        per_entity_sum = divest_df.sum(axis=0).to_dict()

    rows = []
    cumulative_by_entity: dict[str, float] = {}
    for e in sorted(entities):
        cumulative = prior.get(e, 0.0) + per_entity_sum.get(e, 0.0) * unitsize.get(e, 1.0)
        rows.append((e, cumulative))
        cumulative_by_entity[e] = cumulative
    if writer_state is not None:
        writer_state.divested = cumulative_by_entity

    out = pd.DataFrame(rows, columns=["entity", "p_entity_divested"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_fix_storage_quantity(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    writer_state: "OutputWriterState | None" = None,
    csv_dump: bool = False,
) -> Path:
    """Write ``solve_data/fix_storage_quantity.csv`` from v_state for fix_quantity nodes (debug mirror;
    item 163: only under ``csv_dump=True``).

    ``v_state[n, d, t].val * p_entity_unitsize[n]`` for every ``(n, d, t)`` with ``n in
    fix_quantity_nodes``.  Item 163 drops the ``dt_fix_storage_timesteps`` filter: the production
    caller never threaded a provider through it (F5 in design_163.md), so it was already a no-op on
    every fresh run — removing it changes no production output.
    """
    out_path = work_folder / "solve_data" / "fix_storage_quantity.csv"
    if not csv_dump:
        return out_path
    name = out_path.name

    target_nodes = _resolve_storage_fix_methods("fix_quantity", flex_data)
    unitsize = _resolve_unitsize(flex_data)
    state_df = extract_variable(h, "v_state", ("node",), solve_name=solve_name)

    rows: list[tuple[str, str, str, float]] = []
    if not state_df.empty and target_nodes:
        for (_solve, period, time), row in state_df.iterrows():
            for node in row.index:
                if node not in target_nodes:
                    continue
                rows.append((period, time, node, float(row[node]) * unitsize.get(node, 1.0)))

    # Match phase-3 semantics: only OVERWRITE the file when this solve
    # actually has fix-storage entries to record.  An empty dispatch
    # sub-solve must NOT clobber rows the upper storage solve wrote
    # earlier this cascade (the next solve reads them back).  ``writer_state
    # .mirrors_written`` replaces the ``out_path.exists()`` disk check — a
    # leftover file from an earlier run must not look like "already written
    # this cascade".
    already_written = writer_state is not None and name in writer_state.mirrors_written
    if not rows:
        if bool(is_first_solve) and not already_written:
            out_path.write_text("period,step,node,p_fix_storage_quantity\n")
            _logger.info("wrote %s (header only — first solve)", out_path)
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
    else:
        out = pd.DataFrame(rows, columns=["period", "step", "node", "p_fix_storage_quantity"])
        out.to_csv(out_path, index=False, float_format="%.8g")
        _logger.info("wrote %s (%d rows)", out_path, len(out))
    if writer_state is not None:
        writer_state.mirrors_written.add(name)
    return out_path


def write_p_roll_continue_state(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    flex_data: "FlexData | None" = None,
    writer_state: "OutputWriterState | None" = None,
    csv_dump: bool = False,
) -> Path:
    """Write ``solve_data/p_roll_continue_state.csv`` (debug mirror; item 163: only under
    ``csv_dump=True``).

    For every ``n in nodeState`` and the LAST realized timestep of each
    realized period, store ``v_state[n,d,t].val * p_entity_unitsize[n]``.
    Output file has only the latest period's entries (the model writer
    iterates and overwrites; the next read uses the final one).
    """
    out_path = work_folder / "solve_data" / "p_roll_continue_state.csv"
    if not csv_dump:
        return out_path
    name = out_path.name

    nodes = _resolve_node_state(flex_data)
    last_pairs = _resolve_realized_period_time_last(flex_data)
    unitsize = _resolve_unitsize(flex_data)
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
    # solve's content written earlier THIS cascade.
    if not rows:
        _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    out = pd.DataFrame(rows, columns=["node", "p_roll_continue_state"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    if writer_state is not None:
        writer_state.mirrors_written.add(name)
    return out_path


def write_p_entity_period_existing_capacity(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    writer_state: "OutputWriterState | None" = None,
    provider: "object | None" = None,
    csv_dump: bool = False,
) -> Path:
    """Write ``solve_data/p_entity_period_existing_capacity.csv`` (debug mirror; item 163: only under
    ``csv_dump=True`` — the rows are still computed and ``writer_state`` is still updated every call,
    because the next call's ``prior_existing``/``prior_invested`` fallback depends on it; only the disk
    write is gated).

    For each (entity, period) in ``ed_history_realized`` ∪ (entity × d_realize_invest), compute::

        existing = (first_solve & period in period_first → p_entity_pre_existing[e,d])
                 + (not first_solve & (e,d) in history → prior_existing[e,d])
        invested = (not first_solve & (e,d) in history → prior_invested[e,d])

    Item 163 drops the ``ed_invest`` / ``v_invest`` term: ``solve_data/solve__ed_invest.csv`` has had no
    producer since v4 (F5 in design_163.md), so that term was always ∅ in a fresh run — this is the same
    output every production run has always produced.
    """
    out_path = work_folder / "solve_data" / "p_entity_period_existing_capacity.csv"

    pre_existing = _load_pre_existing(provider=provider)
    first_solve = bool(is_first_solve)
    prior_existing, prior_invested = (
        ({}, {}) if first_solve
        else _load_prior_existing(prior_handoff=prior_handoff, writer_state=writer_state)
    )

    # Periods to include in the iteration set.  For the FIRST solve the
    # mod uses ``d_realize_invest ∪ d_fix_storage_period ∪
    # d_realized_period`` (via ``ed_history_realized_first``); for later
    # solves only ``d_realize_invest`` adds new keys (the realized /
    # fix-storage periods only contribute on the first solve).
    realize_invest = _load_realized_periods(
        work_folder / "solve_data" / "realized_invest_periods_of_current_solve.csv",
        provider=provider,
    ) or set()
    if first_solve:
        realized_periods: set[str] = set()
        rd_path = work_folder / "solve_data" / "realized_dispatch.csv"
        _seeded_rd = _provider_lookup_df(provider, rd_path)
        if _seeded_rd is not None:
            realized_periods.update(
                _seeded_rd["period"].cast(str).unique().to_list()
            )
        fix_storage_periods: set[str] = set()
        fs_path = work_folder / "solve_data" / "fix_storage_timesteps.csv"
        _seeded_fs = _provider_lookup_df(provider, fs_path)
        if _seeded_fs is not None and _seeded_fs.height > 0:
            fix_storage_periods.update(
                _seeded_fs["period"].cast(str).unique().to_list()
            )
        iteration_periods = realize_invest | realized_periods | fix_storage_periods
    else:
        iteration_periods = set(realize_invest)

    # period_first is written by Python orchestration to solve_data/.
    # Fallback for the (unlikely) case it's missing: take min(realize_invest).
    period_first = _load_realized_periods(
        work_folder / "solve_data" / "period_first.csv",
        provider=provider,
    ) or ({min(realize_invest)} if realize_invest else set())

    # Iteration set: the prior-history keys ∪ (entity × iteration_periods).
    iter_keys: set[tuple[str, str]] = set(prior_existing.keys())
    entities = _resolve_entity(flex_data)
    for e in entities:
        for d in iteration_periods:
            iter_keys.add((e, d))

    rows: list[tuple[str, str, float, float]] = []
    existing_by_key: dict[tuple[str, str], float] = {}
    invested_by_key: dict[tuple[str, str], float] = {}
    for e, d in sorted(iter_keys):
        existing = 0.0
        invested = 0.0
        if first_solve and d in period_first:
            existing += pre_existing.get((d, e), 0.0)
        if not first_solve:
            existing += prior_existing.get((e, d), 0.0)
            invested += prior_invested.get((e, d), 0.0)
        rows.append((e, d, existing, invested))
        existing_by_key[(e, d)] = existing
        invested_by_key[(e, d)] = invested
    if writer_state is not None:
        writer_state.existing = existing_by_key
        writer_state.invested = invested_by_key

    out = pd.DataFrame(
        rows,
        columns=[
            "entity", "period",
            "p_entity_period_existing_capacity",
            "p_entity_period_invested_capacity",
        ],
    )
    if csv_dump:
        out.to_csv(out_path, index=False, float_format="%.8g")
        _logger.info("wrote %s (%d rows)", out_path, len(out))
    if writer_state is not None:
        writer_state.mirrors_written.add(out_path.name)
    return out_path


def write_fix_storage_usage(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    writer_state: "OutputWriterState | None" = None,
    csv_dump: bool = False,
) -> Path:
    """Write ``solve_data/fix_storage_usage.csv`` (debug mirror; item 163: only under
    ``csv_dump=True``).

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

    Item 163 drops the ``dt_fix_storage_timesteps`` filter for the same reason
    :func:`write_fix_storage_quantity` does — the production caller never
    threaded a provider through it, so it was already a no-op.
    """
    out_path = work_folder / "solve_data" / "fix_storage_usage.csv"
    if not csv_dump:
        return out_path
    name = out_path.name

    target_nodes = _resolve_storage_fix_methods("fix_usage", flex_data)
    already_written = writer_state is not None and name in writer_state.mirrors_written

    if not target_nodes:
        if bool(is_first_solve) and not already_written:
            out_path.write_text("period,step,node,p_fix_storage_usage\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        if writer_state is not None:
            writer_state.mirrors_written.add(name)
        return out_path

    unitsize = _resolve_unitsize(flex_data)
    step_duration = _resolve_step_duration(flex_data)
    flow_df = extract_variable(
        h, "v_flow", ("process", "source", "sink"), solve_name=solve_name,
        coarse_steps=_coarse_flows.coarse_steps(flex_data),
    )

    rows: list[tuple[str, str, str, float]] = []
    for (_solve, period, time), row in flow_df.iterrows():
        dt = step_duration.get((period, time), 1.0)
        per_node: dict[str, float] = {n: 0.0 for n in target_nodes}
        for (process, source, sink), v in row.items():
            us = unitsize.get(process, 1.0)
            if source in per_node:
                per_node[source] += float(v) * us  # outflow from node
            if sink in per_node:
                per_node[sink] -= float(v) * us  # inflow to node
        for node, net in per_node.items():
            # Legacy file only: no engine code reads it.  The engine's source is ``handoff/fix_storage_usage``
            # (flex_data.build_handoff_from_solution), which keeps zero usages (a zero window still caps, item
            # 150), so this file can differ from it.
            if net == 0.0:
                continue
            rows.append((period, time, node, net * dt))

    if not rows:
        if bool(is_first_solve) and not already_written:
            out_path.write_text("period,step,node,p_fix_storage_usage\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        if writer_state is not None:
            writer_state.mirrors_written.add(name)
        return out_path

    out = pd.DataFrame(rows, columns=["period", "step", "node", "p_fix_storage_usage"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    if writer_state is not None:
        writer_state.mirrors_written.add(name)
    return out_path


# ---------------------------------------------------------------------------
# Cross-solve capacity accumulators
#
# These four writers are conceptually handoffs too — the file they emit
# is the cumulative output-so-far across every solve that has run.  On
# the first solve they truncate + write the header; later solves append
# only the rows whose period is not already in
# ``solve_data/period_capacity.csv`` (so rolling windows don't write
# the same period twice).  Together they replicate the tail of phase 3
# that produced ``solve_data/unit_capacity__period.csv`` etc. — so when
# phase 3 is skipped, the user-facing capacity outputs stay intact.
# ---------------------------------------------------------------------------


def _append_period_capacity(
    work_folder: Path, new_periods: list[str],
    writer_state: "OutputWriterState",
) -> None:
    """Write ``solve_data/period_capacity.csv`` = prior_set ∪ new_periods.

    Truncate + re-emit union so the next roll sees an up-to-date set.
    ``writer_state`` is required (item 163: no disk fallback) — the prior
    set is always sourced from the in-memory accumulator.
    """
    path = work_folder / "solve_data" / "period_capacity.csv"
    existing = set(writer_state.periods_already_emitted)
    union = existing | set(new_periods)
    # Emit sorted for determinism (downstream must not rely on row order).
    with open(path, "w", encoding="utf-8") as f:
        f.write("period\n")
        for p in sorted(union):
            f.write(p + "\n")


def _bump_period_capacity(
    work_folder: Path,
    writer_state: "OutputWriterState",
) -> None:
    """Write ``solve_data/period_capacity.csv`` — header-only union of periods already emitted.

    Called once per solve at the end of :func:`write_all_handoffs`.  Item 163: ``new_periods`` is
    always ``[]`` now — the set that fed it (``d_realize_dispatch_or_invest.csv``, keyed off
    ``_actual_solve_name``) was already vestigial (the per-period capacity dumps that consumed it
    were removed 2026-08-07) and had no provider wiring (F5 in design_163.md), so a fresh run's file
    was always header-only already; this is byte-identical.  ``period_capacity.csv`` is NOT gated
    behind ``csv_dump`` — unlike the other five mirrors (design_163.md §11 Q1), it is written every
    run.
    """
    new_periods: list[str] = []
    _append_period_capacity(work_folder, new_periods, writer_state=writer_state)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def write_all_handoffs(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    writer_state: "OutputWriterState | None" = None,
    provider: "object | None" = None,
    csv_dump: bool = False,
) -> list[Path]:
    """Write the handoff files.

    Each writer is independent — failure on one is logged and does not
    abort the rest, mirroring :func:`write_all_variables`.

    ``prior_handoff`` (when provided) sources the prior-roll state
    (``realized_existing`` / ``realized_invest`` / ``divest_cumulative``)
    from the in-memory ``SolveHandoff`` instead of the writer's own
    previous call; forwarded to the two writers that consume prior state,
    the remainder ignore it.

    ``flex_data``, ``is_first_solve`` and ``writer_state`` are REQUIRED
    (item 163: no work-folder disk fallback left in any writer) — a
    ``None`` value raises ``ValueError`` naming the parameter.  Production
    threads all three on every call; a caller missing one is a wiring bug,
    never silently tolerated by relaxing a writer back to a disk read.

    ``csv_dump`` gates the five handoff-mirror CSVs' disk WRITE (design_163
    .md §11 Q1: they are debug output only; ``csv_dump=False`` computes
    nothing and writes nothing for those five).  ``period_capacity.csv`` is
    not one of them — it is written every call regardless of ``csv_dump``.
    """
    if flex_data is None:
        raise ValueError("write_all_handoffs needs flex_data")
    if is_first_solve is None:
        raise ValueError("write_all_handoffs needs is_first_solve")
    if writer_state is None:
        raise ValueError("write_all_handoffs needs writer_state")

    # Writers that take additional kwargs beyond the base (h, solve_name,
    # work_folder).  The dispatch table routes per-writer.
    written: list[Path] = []
    fd_kwargs = {
        "flex_data": flex_data, "is_first_solve": is_first_solve,
        "writer_state": writer_state, "csv_dump": csv_dump,
    }
    dispatch: list[tuple[object, dict]] = [
        (write_p_entity_divested,
         {**fd_kwargs, "prior_handoff": prior_handoff}),
        (write_fix_storage_quantity, fd_kwargs),
        (write_p_roll_continue_state,
         {"flex_data": flex_data, "writer_state": writer_state, "csv_dump": csv_dump}),
        (write_p_entity_period_existing_capacity,
         {**fd_kwargs, "prior_handoff": prior_handoff, "provider": provider}),
        (write_fix_storage_usage, fd_kwargs),
    ]
    for fn, extra in dispatch:
        try:
            written.append(fn(
                h, solve_name=solve_name, work_folder=work_folder, **extra,
            ))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("handoff writer %s failed: %s", fn.__name__, exc)
    # ``period_capacity.csv``: written every call, independent of ``csv_dump``
    # (design_163.md §11 Q1 exempts it — see ``_bump_period_capacity``).
    try:
        _bump_period_capacity(work_folder, writer_state=writer_state)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("period_capacity accumulation failed: %s", exc)
    return written
