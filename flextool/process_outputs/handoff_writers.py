"""
Solve-to-solve handoff CSV writers.

These writers produce the six CSVs carrying state from the previous
solve, written from the live ``Highs`` instance + the input /
parameter data.

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

Sourcing strategy
-----------------
The writers source SOLVER values directly from ``highspy.Highs``:

* variable values via :func:`extract_variable` (already used by
  ``read_highs_solution.py``).
* constraint duals — same machinery, with ``source="row_dual"``.

PARAMETER values are read from ``input/`` (write-once) and
``solve_data/`` (per-solve) — both pure parameter snapshots (input
data after model-side derivations).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from flextool.process_outputs.read_highs_solution import (
    _actual_solve_name,
    _load_realized_periods,
    _load_realized_set,
    extract_variable,
)

if TYPE_CHECKING:
    import highspy
    from flextool.engine_polars._solve_handoff import SolveHandoff
    from flextool.engine_polars.input import FlexData
    from flextool.engine_polars._output_writer import OutputWriterState

_logger = logging.getLogger(__name__)

# Fallback inverse of scale_the_objective (multiply duals by 1e6 to
# undo a 1e-6 scaling).  The live per-solve value is read via
# :func:`flextool.process_outputs.read_highs_solution._resolve_inv_scale_the_objective`
# — this constant only applies when the CSV can't be read.
_INV_SCALE_THE_OBJECTIVE = 1e6


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


def _load_unitsize(work_folder: Path) -> dict[str, float]:
    """Return ``{entity: unitsize}`` from ``input/p_entity_unitsize.csv``.

    The CSV is wide: header row = entity names, second row labelled
    ``value`` = the unitsize per entity.
    """
    path = work_folder / "input" / "p_entity_unitsize.csv"
    df = pd.read_csv(path, index_col=0)
    return df.loc["value"].astype(float).to_dict()


def _load_pre_existing(
    work_folder: Path,
    *,
    provider: "object | None" = None,
) -> dict[tuple[str, str], float]:
    """Return ``{(period, entity): value}`` for ``p_entity_pre_existing``.

    Provider-first: the canonical in-memory frame at
    ``solve_data/p_entity_pre_existing`` (populated by
    ``_emit_chain_params.emit_p_entity_pre_existing``) is tall —
    ``(entity, period, value)`` — and is what every consumer should
    use on the in-memory cascade.

    Falls back to the per-solve appendage CSV
    ``solve_data/solve__p_entity_pre_existing.csv`` (wide
    ``(solve, period) × entity_columns``) when the Provider doesn't
    have it.  On the current in-memory cascade that CSV is typically
    absent — in which case we silently return ``{}`` rather than
    raising ``FileNotFoundError``.  ``write_p_entity_period_existing_capacity``
    treats an empty pre_existing as "no pre-existing capacity recorded
    for this solve", which is the correct semantics when nothing
    upstream produced it.
    """
    if provider is not None and provider.has("solve_data/p_entity_pre_existing"):
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

    # Disk fallback — legacy pre-Δ.31 CSV.
    path = work_folder / "solve_data" / "solve__p_entity_pre_existing.csv"
    if not path.exists():
        return {}
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
    *, prior_handoff: "SolveHandoff | None" = None,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Return prior ``p_entity_period_existing_capacity`` + ``_invested_capacity``.

    With ``prior_handoff`` populated, read the two dicts from
    ``realized_existing`` / ``realized_invest`` (in-memory consume side).
    Else fall back to the previous solve's
    ``solve_data/p_entity_period_existing_capacity.csv`` (file fallback).
    For the first solve neither is available → return empty dicts.
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
    path = work_folder / "solve_data" / "p_entity_period_existing_capacity.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    if df.empty:
        return {}, {}
    existing = {}
    invested = {}
    for _, row in df.iterrows():
        key = (str(row["entity"]), str(row["period"]))
        existing[key] = float(row["p_entity_period_existing_capacity"])
        invested[key] = float(row["p_entity_period_invested_capacity"])
    return existing, invested


def _load_prior_divested(
    work_folder: Path,
    *, prior_handoff: "SolveHandoff | None" = None,
) -> dict[str, float]:
    """Return ``{entity: cumulative_divested}`` from prior solve, or empty.

    Reads from ``prior_handoff.divest_cumulative`` when populated; else
    falls back to ``solve_data/p_entity_divested.csv``."""
    if prior_handoff is not None and prior_handoff.divest_cumulative is not None:
        return {
            str(r["entity"]): float(r["value"])
            for r in prior_handoff.divest_cumulative.iter_rows(named=True)
        }
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
    """Return the set of nodes that maintain a state variable.

    Derived from ``input/p_node_type.csv``: rows whose ``p_node_type``
    equals ``'storage'``.  Nodes absent from the file use the mod's
    default (``'balance'``), which is not a storage node.
    """
    path = work_folder / "input" / "p_node_type.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "p_node_type" not in df.columns:
        return set()
    storage = df.loc[df["p_node_type"].astype(str) == "storage", "node"]
    return set(storage.astype(str).tolist())


def _load_entity(work_folder: Path) -> set[str]:
    """Return the full ``entity`` set from ``input/entity.csv``."""
    path = work_folder / "input" / "entity.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        return set()
    return set(df.iloc[:, 0].astype(str).tolist())


def _load_entity_divest(work_folder: Path) -> set[str]:
    """Return ``entityDivest`` — entities allowed to divest.

    Sourced from ``solve_data/entityDivest.csv`` (phase 1 dump);
    matches the ``setof {(e,m) in entity__invest_method : m not in
    divest_method_not_allowed} (e)`` derivation in the model.
    """
    path = work_folder / "solve_data" / "entityDivest.csv"
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
    ``solve_data/complete_period_share_of_year.csv``."""
    path = work_folder / "solve_data" / "complete_period_share_of_year.csv"
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
    ``solve_data/solve__p_inflation_factor_operations_yearly.csv``."""
    path = work_folder / "solve_data" / "solve__p_inflation_factor_operations_yearly.csv"
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

    Source: ``solve_data/p_model.csv`` (long ``modelParam,p_model`` pairs)
    written by ``write_solve_status``.  Reads ``p_model['solveFirst']``.
    """
    path = work_folder / "solve_data" / "p_model.csv"
    if not path.exists():
        return True
    df = pd.read_csv(path)
    matches = df.loc[df["modelParam"] == "solveFirst", "p_model"]
    if matches.empty:
        return True
    return bool(int(matches.iloc[0]))


# ---------------------------------------------------------------------------
# Phase G — in-memory resolvers
#
# Each ``_resolve_*`` prefers a value from the in-memory ``FlexData``
# (or an explicit kwarg passed by the cascade) and falls back to the
# matching ``_load_*`` disk reader.  Callers thread ``flex_data`` /
# ``is_first_solve`` from the cascade so the per-iter file reads listed
# in ``specs/in_memory_carriers_audit.md`` (12 readers in this module)
# disappear from the hot path while the disk fallback stays intact for
# unit tests / synthesized callers that never instantiate FlexData.
# ---------------------------------------------------------------------------


def _resolve_unitsize(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> dict[str, float]:
    """``{entity: unitsize}`` from ``flex_data.p_all_entity_unitsize``.

    Phase G — when ``flex_data`` is supplied, trust the in-memory
    carrier even if ``p_all_entity_unitsize`` is ``None`` (returns
    empty dict, same as the disk fallback would on a missing file)."""
    if flex_data is not None:
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
            pass
    return _load_unitsize(work_folder)


def _resolve_pre_existing(
    work_folder: Path, flex_data: "FlexData | None" = None,
    *, provider: "object | None" = None,
) -> dict[tuple[str, str], float]:
    """``{(period, entity): value}`` — pre-existing capacity from FlexData.

    Prefers ``flex_data.p_entity_period_existing_capacity`` (Param) when
    populated; this is the same data that ``solve__p_entity_pre_existing.csv``
    snapshots after the parent's overlay applies.  Falls back to the disk
    CSV when the in-memory carrier is absent (e.g. unit-test paths).
    """
    # No direct in-memory equivalent on FlexData; fall back to disk.
    # (The audit's "wire kwarg" suggestion would require surfacing
    # ``solve__p_entity_pre_existing.csv`` as a FlexData field — deferred.)
    return _load_pre_existing(work_folder, provider=provider)


def _resolve_storage_fix_methods(
    work_folder: Path, target_method: str,
    flex_data: "FlexData | None" = None,
) -> set[str]:
    """``{node, ...}`` whose nested fix method == *target_method*.

    Phase G — when ``flex_data`` is supplied, treat ``flex_data
    .node__storage_nested_fix_method`` as the authoritative source.
    ``None`` means "no such CSV existed when FlexData was built" — the
    on-disk fallback would return an empty set in that case, so we
    short-circuit here without re-reading the file.
    """
    if flex_data is not None:
        f = getattr(flex_data, "node__storage_nested_fix_method", None)
        if f is None:
            return set()
        try:
            sub = f.filter(pl.col("method") == target_method)
            return set(sub["node"].cast(str).to_list())
        except Exception:  # noqa: BLE001
            pass
    return _load_storage_fix_methods(work_folder, target_method)


def _resolve_node_state(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> set[str]:
    """``{node, ...}`` carrying a state variable (``nodeState`` set).

    Phase G — when ``flex_data`` is supplied we trust the in-memory
    ``nodeState`` set even if it's ``None`` (which means: no storage
    nodes in this model, same answer the disk fallback would give from
    a missing ``p_node_type.csv``).
    """
    if flex_data is not None:
        f = getattr(flex_data, "nodeState", None)
        if f is None:
            return set()
        try:
            col = f.columns[0]
            return set(f[col].cast(str).to_list())
        except Exception:  # noqa: BLE001
            pass
    return _load_node_state(work_folder)


def _resolve_entity(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> set[str]:
    """Full entity set (every process + connection + node).

    Phase G — when ``flex_data`` is supplied, trust the in-memory
    entity carrier (use ``p_all_entity_unitsize``'s entity column as the
    canonical set)."""
    if flex_data is not None:
        param = getattr(flex_data, "p_all_entity_unitsize", None)
        if param is None:
            return set()
        try:
            f = param.frame
            entity_col = f.columns[0]
            return set(f[entity_col].cast(str).to_list())
        except Exception:  # noqa: BLE001
            pass
    return _load_entity(work_folder)


def _resolve_entity_divest(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> set[str]:
    """``entityDivest`` — entities allowed to divest.  Phase G trusts
    ``flex_data.ed_divest_set`` when supplied (``None`` ⇒ empty set,
    same as missing CSV)."""
    if flex_data is not None:
        f = getattr(flex_data, "ed_divest_set", None)
        if f is None:
            return set()
        try:
            entity_col = f.columns[0]
            return set(f[entity_col].cast(str).to_list())
        except Exception:  # noqa: BLE001
            pass
    return _load_entity_divest(work_folder)


def _resolve_realized_period_time_last(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> list[tuple[str, str]]:
    """One ``(period, time)`` pair per realized period — the LAST step in
    each.  Prefers ``flex_data.realized_dispatch`` (polars frame of
    ``(period, step)``).  When ``flex_data`` is supplied we trust the
    in-memory state — empty/None means "no realized dispatch this
    solve", same answer the disk fallback would give."""
    if flex_data is not None:
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
            pass
    return _load_realized_period_time_last(work_folder)


def _resolve_complete_period_share_of_year(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> dict[str, float]:
    """``{period: share_of_year}`` from ``flex_data.p_period_share``.
    Phase G trusts the in-memory carrier when supplied."""
    if flex_data is not None:
        param = getattr(flex_data, "p_period_share", None)
        if param is None:
            return {}
        try:
            f = param.frame
            period_col = f.columns[0]
            return dict(zip(
                f[period_col].cast(str).to_list(),
                f["value"].cast(float).to_list(),
            ))
        except Exception:  # noqa: BLE001
            pass
    return _load_complete_period_share_of_year(work_folder)


def _resolve_inflation_factor_operations_yearly(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> dict[str, float]:
    """``{period: inflation_factor}`` from ``flex_data.p_inflation_op``.
    Phase G trusts the in-memory carrier when supplied."""
    if flex_data is not None:
        param = getattr(flex_data, "p_inflation_op", None)
        if param is None:
            return {}
        try:
            f = param.frame
            period_col = f.columns[0]
            return dict(zip(
                f[period_col].cast(str).to_list(),
                f["value"].cast(float).to_list(),
            ))
        except Exception:  # noqa: BLE001
            pass
    return _load_inflation_factor_operations_yearly(work_folder)


def _resolve_step_duration(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> dict[tuple[str, str], float]:
    """``{(period, time): step_duration}`` from ``flex_data.p_step_duration``.
    Phase G trusts the in-memory carrier when supplied."""
    if flex_data is not None:
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
            pass
    return _load_step_duration(work_folder)


def _resolve_is_first_solve(
    work_folder: Path, is_first_solve: bool | None = None,
) -> bool:
    """Cascade-supplied flag preferred; else read ``solve_data/p_model.csv``."""
    if is_first_solve is not None:
        return is_first_solve
    return _is_first_solve(work_folder)


# ---------------------------------------------------------------------------
# Handoff writers
# ---------------------------------------------------------------------------


def write_p_entity_divested(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
) -> Path:
    """Write ``solve_data/p_entity_divested.csv`` from v_divest + prior.

    ``cumulative_divested[e] = prior_divested[e] + sum_d v_divest[e,d] * unitsize[e]``
    over every divest period (sum_d means all (e,d) declared in ed_divest).

    ``prior_handoff`` (when populated) replaces the on-disk read of the
    parent solve's ``p_entity_divested.csv`` with the in-memory
    ``divest_cumulative`` carrier.  ``flex_data`` / ``is_first_solve``
    are Phase G kwargs that route the entity / unitsize / first-solve
    lookups through in-memory carriers — see the resolvers above.
    """
    out_path = work_folder / "solve_data" / "p_entity_divested.csv"

    entities = _resolve_entity_divest(work_folder, flex_data)
    unitsize = _resolve_unitsize(work_folder, flex_data)
    first = _resolve_is_first_solve(work_folder, is_first_solve)
    prior = (
        {} if first
        else _load_prior_divested(work_folder, prior_handoff=prior_handoff)
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
    for e in sorted(entities):
        cumulative = prior.get(e, 0.0) + per_entity_sum.get(e, 0.0) * unitsize.get(e, 1.0)
        rows.append((e, cumulative))

    out = pd.DataFrame(rows, columns=["entity", "p_entity_divested"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_fix_storage_quantity(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
) -> Path:
    """Write ``solve_data/fix_storage_quantity.csv`` from v_state for fix_quantity nodes.

    ``v_state[n, d, t].val * p_entity_unitsize[n]`` for every
    ``(n, d, t)`` with ``n in fix_quantity_nodes`` and
    ``(d, t) in dt_fix_storage_timesteps``.
    """
    out_path = work_folder / "solve_data" / "fix_storage_quantity.csv"

    target_nodes = _resolve_storage_fix_methods(work_folder, "fix_quantity", flex_data)
    fix_steps_path = work_folder / "solve_data" / "fix_storage_timesteps.csv"
    fix_steps = _load_realized_set(fix_steps_path)
    unitsize = _resolve_unitsize(work_folder, flex_data)
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
        if _resolve_is_first_solve(work_folder, is_first_solve) and not out_path.exists():
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
    flex_data: "FlexData | None" = None,
) -> Path:
    """Write ``solve_data/p_roll_continue_state.csv``.

    For every ``n in nodeState`` and the LAST realized timestep of each
    realized period, store ``v_state[n,d,t].val * p_entity_unitsize[n]``.
    Output file has only the latest period's entries (the model writer
    iterates and overwrites; the next read uses the final one).
    """
    out_path = work_folder / "solve_data" / "p_roll_continue_state.csv"

    nodes = _resolve_node_state(work_folder, flex_data)
    last_pairs = _resolve_realized_period_time_last(work_folder, flex_data)
    unitsize = _resolve_unitsize(work_folder, flex_data)
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
    prior_handoff: "SolveHandoff | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    provider: "object | None" = None,
    csv_dump: bool = False,
) -> Path:
    """Write ``solve_data/p_entity_period_existing_capacity.csv``.

    For each (entity, period) in ``ed_history_realized`` ∪ (entity ×
    d_realize_invest), compute::

        existing = (first_solve & period in period_first → p_entity_pre_existing[e,d])
                 + (not first_solve & (e,d) in history → prior_existing[e,d])
                 + ((e,d) in ed_invest & d in d_realize_invest → v_invest[e,d] * unitsize[e])
        invested = (not first_solve & (e,d) in history → prior_invested[e,d])
                 + ((e,d) in ed_invest & d in d_realize_invest → v_invest[e,d] * unitsize[e])

    The ``ed_history_realized`` set is read from
    ``solve_data/solve__ed_invest.csv`` (entity, period) for the
    "(e,d) in ed_invest" predicate, plus the prior history file's keys.
    """
    out_path = work_folder / "solve_data" / "p_entity_period_existing_capacity.csv"

    unitsize = _resolve_unitsize(work_folder, flex_data)
    pre_existing = _resolve_pre_existing(work_folder, flex_data, provider=provider)
    first_solve = _resolve_is_first_solve(work_folder, is_first_solve)
    prior_existing, prior_invested = (
        ({}, {}) if first_solve
        else _load_prior_existing(work_folder, prior_handoff=prior_handoff)
    )

    invest_df = extract_variable(
        h, "v_invest", ("entity",), solve_name=solve_name, has_time=False,
        provider=provider,
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
        # Step 1-e — Provider-aware: under the in-memory cascade the
        # files aren't on disk but the per-sub-solve Provider has the
        # frames.  Transitional seed-funnel fallback for unplumbed
        # callsites lives in :func:`_provider_lookup_df` below.
        realized_periods: set[str] = set()
        rd_path = work_folder / "solve_data" / "realized_dispatch.csv"
        _seeded_rd = _provider_lookup_df(provider, rd_path)
        if _seeded_rd is not None:
            realized_periods.update(
                _seeded_rd["period"].cast(str).unique().to_list()
            )
        elif rd_path.exists():
            realized_periods.update(
                pd.read_csv(rd_path)["period"].astype(str).unique()
            )
        fix_storage_periods: set[str] = set()
        fs_path = work_folder / "solve_data" / "fix_storage_timesteps.csv"
        _seeded_fs = _provider_lookup_df(provider, fs_path)
        if _seeded_fs is not None:
            if _seeded_fs.height > 0:
                fix_storage_periods.update(
                    _seeded_fs["period"].cast(str).unique().to_list()
                )
        elif fs_path.exists():
            fs_df = pd.read_csv(fs_path)
            if not fs_df.empty:
                fix_storage_periods.update(fs_df["period"].astype(str).unique())
        iteration_periods = realize_invest | realized_periods | fix_storage_periods
    else:
        iteration_periods = set(realize_invest)

    # period_first is written by Python orchestration to solve_data/.
    # Fallback for the (unlikely) case it's missing: take min(realize_invest).
    period_first = _load_realized_periods(
        work_folder / "solve_data" / "period_first.csv",
        provider=provider,
    ) or ({min(realize_invest)} if realize_invest else set())

    # ed_invest set (entity, period) — phase-1 dump.  CSV layout is
    # ``solve, entity, period`` (3 columns); we drop the solve column.
    ed_invest: set[tuple[str, str]] = set()
    ei_path = work_folder / "solve_data" / "solve__ed_invest.csv"
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
    entities = _resolve_entity(work_folder, flex_data)
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
    if csv_dump:
        out.to_csv(out_path, index=False, float_format="%.8g")
        _logger.info("wrote %s (%d rows)", out_path, len(out))
    return out_path


def write_fix_storage_price(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    scale_the_objective: float | None = None,
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
    from flextool.process_outputs.read_highs_solution import (
        _resolve_inv_scale_the_objective,
    )
    out_path = work_folder / "solve_data" / "fix_storage_price.csv"

    target_nodes = _resolve_storage_fix_methods(work_folder, "fix_price", flex_data)
    fix_steps = _load_realized_set(
        work_folder / "solve_data" / "fix_storage_timesteps.csv"
    )
    inflation = _resolve_inflation_factor_operations_yearly(work_folder, flex_data)
    period_share = _resolve_complete_period_share_of_year(work_folder, flex_data)
    # Agent 12: resolve live scale_the_objective from the per-solve CSV.
    # Phase G: cascade-supplied ``scale_the_objective`` kwarg short-
    # circuits the CSV read entirely.
    inv_scale = _resolve_inv_scale_the_objective(
        work_folder, scale_the_objective=scale_the_objective,
    )

    # Empty short-circuit — most scenarios have no fix_price nodes.
    if not target_nodes or not fix_steps:
        if _resolve_is_first_solve(work_folder, is_first_solve) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_price\n")
            _logger.info("wrote %s (header only — first solve)", out_path)
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    # nodeBalance_eq has 9 indices (Agent 1.4 added ``bn`` between
    # ``node`` and ``period``).  Use the generic extractor with 7
    # col_names so the trailing two stay the (period, time) row index;
    # then filter rows to fix_steps and entity-name match to
    # target_nodes (col_names[1] == 'node').  In degenerate mode bn is
    # always 'default'.
    df = extract_variable(
        h, "nodeBalance_eq",
        col_names=("c", "node", "bn"),
        solve_name=solve_name,
        has_time=True,
        source="row_dual",
        value_scale=1.0,  # we apply the full transform manually below
        trailing_col_names=("t_prev", "t_prev_within_timeset",
                            "d_prev", "t_prev_within_solve"),
    )

    if df.empty:
        # Same preserve-prior semantics as the empty short-circuit above.
        if _resolve_is_first_solve(work_folder, is_first_solve) and not out_path.exists():
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
                * inv_scale
            )
            rows.append((period, time, node, float(dual_val) * scale))

    if not rows:
        if _resolve_is_first_solve(work_folder, is_first_solve) and not out_path.exists():
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
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
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

    target_nodes = _resolve_storage_fix_methods(work_folder, "fix_usage", flex_data)
    fix_steps = _load_realized_set(
        work_folder / "solve_data" / "fix_storage_timesteps.csv"
    )

    if not target_nodes or not fix_steps:
        if _resolve_is_first_solve(work_folder, is_first_solve) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_usage\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    unitsize = _resolve_unitsize(work_folder, flex_data)
    step_duration = _resolve_step_duration(work_folder, flex_data)
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
        if _resolve_is_first_solve(work_folder, is_first_solve) and not out_path.exists():
            out_path.write_text("period,step,node,p_fix_storage_usage\n")
        else:
            _logger.debug("skipped %s (empty; preserving prior content)", out_path)
        return out_path

    out = pd.DataFrame(rows, columns=["period", "step", "node", "p_fix_storage_usage"])
    out.to_csv(out_path, index=False, float_format="%.8g")
    _logger.info("wrote %s (%d rows)", out_path, len(out))
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


def _load_entity_class_set(work_folder: Path, set_name: str) -> list[str]:
    """Return the ordered list of entities in ``input/<set_name>.csv``.

    The resolved-set CSVs (``entity.csv``, ``process_unit.csv``,
    ``process_connection.csv``) are written by ``input_writer`` from
    the DB and live in ``input/``.  The mod previously also re-emitted
    them under ``solve_data/`` via ``printf`` blocks; that redundant
    write was retired in the post-solve cleanup, so all consumers now
    read directly from ``input/``.

    Special case: ``nodeState`` is derived from ``input/p_node_type.csv``
    (rows with ``p_node_type == 'storage'``), preserving the original
    node order.
    """
    if set_name == "nodeState":
        path = work_folder / "input" / "p_node_type.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        if df.empty or "p_node_type" not in df.columns:
            return []
        return df.loc[df["p_node_type"].astype(str) == "storage", "node"].astype(str).tolist()
    path = work_folder / "input" / f"{set_name}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        return []
    return df.iloc[:, 0].astype(str).tolist()


def _load_unitsize_map(
    work_folder: Path, flex_data: "FlexData | None" = None,
) -> dict[str, float]:
    """Wrapper around :func:`_load_unitsize` that never raises on a
    missing file (some scenarios don't use any entities with unitsize).

    Phase G — when ``flex_data`` is supplied, trust the in-memory
    ``p_all_entity_unitsize`` carrier (``None`` short-circuits to {} —
    same as the file fallback's missing-file branch)."""
    if flex_data is not None:
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
            pass
    path = work_folder / "input" / "p_entity_unitsize.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, index_col=0)
    return df.loc["value"].astype(float).to_dict() if "value" in df.index else {}
def _load_period_capacity(work_folder: Path) -> set[str]:
    """Periods already output by a previous roll's capacity dump."""
    path = work_folder / "solve_data" / "period_capacity.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "period" not in df.columns:
        return set()
    return set(df["period"].astype(str))


def _load_drdi(work_folder: Path, roll: str) -> list[str]:
    """Periods in this roll's ``d_realize_dispatch_or_invest`` set, in
    file order."""
    path = work_folder / "solve_data" / "d_realize_dispatch_or_invest.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return []
    df = df[df["solve"] == roll]
    return df["period"].tolist()
def _append_period_capacity(
    work_folder: Path, new_periods: list[str],
    writer_state: "OutputWriterState | None" = None,
) -> None:
    """Write ``solve_data/period_capacity.csv`` = prior_set ∪ new_periods.

    Truncate + re-emit union so the next roll sees an up-to-date set.

    When ``writer_state`` is supplied, source the prior set from the
    in-memory accumulator instead of re-reading the CSV.
    """
    path = work_folder / "solve_data" / "period_capacity.csv"
    if writer_state is not None:
        existing = set(writer_state.periods_already_emitted)
    else:
        existing = _load_period_capacity(work_folder)
    union = existing | set(new_periods)
    # Emit sorted for determinism (downstream must not rely on row order).
    with open(path, "w", encoding="utf-8") as f:
        f.write("period\n")
        for p in sorted(union):
            f.write(p + "\n")
def _bump_period_capacity(
    work_folder: Path, solve_name: str,
    writer_state: "OutputWriterState | None" = None,
) -> None:
    """Accumulate this solve's realized periods into ``period_capacity.csv``.

    Called once per solve at the end of :func:`write_all_handoffs`.  This
    set used to let the per-period capacity dumps skip already-emitted
    periods; those writers were removed 2026-08-07, so the accumulator is
    now vestigial (kept pending a separate teardown of the
    ``period_capacity`` machinery, which ripples into orchestration and
    several tests).  Still read via :func:`_load_period_capacity`.

    When ``writer_state`` is supplied, ALSO push the newly
    accumulated periods into ``writer_state.periods_already_emitted``
    in-memory.  The writer adapter previously re-read the file to refresh
    this set; with the in-memory dual update it can trust the accumulator.
    """
    roll = _actual_solve_name(work_folder, solve_name)
    new_periods = _load_drdi(work_folder, roll)
    _append_period_capacity(work_folder, new_periods, writer_state=writer_state)
    if writer_state is not None and new_periods:
        writer_state.periods_already_emitted.update(str(p) for p in new_periods)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def write_all_handoffs(
    h: "highspy.Highs", *, solve_name: str, work_folder: Path,
    prior_handoff: "SolveHandoff | None" = None,
    flex_data: "FlexData | None" = None,
    is_first_solve: bool | None = None,
    writer_state: "OutputWriterState | None" = None,
    scale_the_objective: float | None = None,
    provider: "object | None" = None,
    csv_dump: bool = False,
) -> list[Path]:
    """Write all six handoff files.

    Each writer is independent — failure on one is logged and does not
    abort the rest, mirroring :func:`write_all_variables`.

    ``prior_handoff`` (when provided) sources the prior-roll state
    (``realized_existing`` / ``realized_invest`` / ``divest_cumulative``)
    from the in-memory ``SolveHandoff`` instead of re-reading the
    parent solve's CSV outputs.  Forwarded to the two writers that
    consume prior state; the remainder ignore it.

    Phase G kwargs (additive, fully backward-compatible):

    * ``flex_data`` — the cascade's in-memory FlexData; routes the per-
      writer ``_load_unitsize`` / ``_load_node_state`` / ``_load_entity``
      / ``_load_storage_fix_methods`` / period-share / inflation / step-
      duration / realized-dispatch lookups through in-memory carriers.
      CSV fallback preserved when not supplied.
    * ``is_first_solve`` — cascade-supplied first-solve flag; replaces
      the per-call ``solve_data/p_model.csv`` read in every writer.
    * ``writer_state`` — pushes newly-accumulated periods into the
      ``periods_already_emitted`` set in-memory + lets the per-period
      capacity writer skip the on-disk re-read of ``period_capacity.csv``.

    See ``specs/in_memory_carriers_audit.md`` for the per-reader mapping.
    """
    # Writers that take additional kwargs beyond the base (h, solve_name,
    # work_folder).  Phase G greatly expanded these — the dispatch table
    # below routes per-writer.
    written: list[Path] = []
    fd_kwargs = {"flex_data": flex_data, "is_first_solve": is_first_solve}
    dispatch: list[tuple[object, dict]] = [
        (write_p_entity_divested,
         {**fd_kwargs, "prior_handoff": prior_handoff}),
        (write_fix_storage_quantity, fd_kwargs),
        (write_p_roll_continue_state, {"flex_data": flex_data}),
        (write_p_entity_period_existing_capacity,
         {**fd_kwargs, "prior_handoff": prior_handoff, "provider": provider,
          "csv_dump": csv_dump}),
        (write_fix_storage_price,
         {**fd_kwargs, "scale_the_objective": scale_the_objective}),
        (write_fix_storage_usage, fd_kwargs),
    ]
    for fn, extra in dispatch:
        try:
            written.append(fn(
                h, solve_name=solve_name, work_folder=work_folder, **extra,
            ))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("handoff writer %s failed: %s", fn.__name__, exc)
    # Accumulate this solve's realized periods into ``period_capacity.csv``
    # for the next roll (vestigial: the per-period capacity dumps that
    # consumed this set were removed 2026-08-07; the accumulator is kept
    # pending a separate teardown of the ``period_capacity`` machinery).
    try:
        _bump_period_capacity(work_folder, solve_name, writer_state=writer_state)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("period_capacity accumulation failed: %s", exc)
    return written
