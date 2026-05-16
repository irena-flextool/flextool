"""Writer-port Phase 1 follow-up — pdtProcess / pdtNode / pdtProcess_{source,sink}.

Native ports of the four ``write_pdt*`` helpers in
:mod:`flextool.flextoolrunner.preprocessing.entity_period_calc_params`
that consume the ``PdtLookup`` / ``PdtLookupPerSide`` class hierarchy.

The lookup classes themselves are ported in :mod:`._pdt_lookup`; this
module just orchestrates the per-row resolution against fixture inputs
and emits the legacy CSV format.

Ported writers (legacy LOC ~123):

* ``write_pdtProcess``         — flextool.mod L1227 (7-branch).
* ``write_pdtNode``            — flextool.mod L1176 (9-branch: time-first
  + class-default fallback).
* ``write_pdtProcess_source``  — flextool.mod L1265 (6-branch, per-side).
* ``write_pdtProcess_sink``    — flextool.mod L1279 (6-branch, per-side).

Phase E-b lift — derive_X / _write split
----------------------------------------

Each writer is now a thin ``_write(derive_X(...), path)`` wrapper around
its ``derive_*`` companion that materialises a canonical polars frame.
The accumulator in :mod:`._flex_data_accumulator` captures these frames
via its patched ``_write`` helper, so downstream Phase D / E-a
consumers can read them straight out of memory instead of round-tripping
through ``solve_data/*.csv`` on disk.

These four writers are the high-memory hotspots flagged in
``specs/sparse_writer_lessons_for_engine_polars.md`` §1 — a real
``write_pdtProcess`` can emit a 441 MB / 280k-row dense CSV.  The
sparse-emit optimisation (``filter(value != 0.0)`` + consumer-side
``fill_null(0.0)``) is intentionally NOT applied here: byte-parity with
the legacy dense emit must hold for ``test_writer_port_phase1.py``.
The sparse-emit rework is a separate future dispatch that needs
goldens regenerated and consumer-side overlays.

Value-column semantics: the legacy emitters wrote ``f"...,{repr(v)}\\n"``
which preserves the Python-type-as-emitted distinction
between ``0`` (int) and ``0.0`` (float).  We mirror that by building
``value`` as a ``Utf8`` column with ``repr(v)`` applied per-row (the
same pattern :mod:`._writer_chain_params` uses for its
``_ed_value_frame`` helper).
"""
from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

from flextool.engine_polars._pdt_lookup import (
    NODE_PARAM_DEF1,
    PROCESS_PARAM_DEF1,
    PdtLookup,
    PdtLookupPerSide,
    read_class_defaults,
)


# ---------------------------------------------------------------------------
# Canonical writer-port emitter — mirrors the ``_write(df, path)`` idiom
# in :mod:`._writer_arc_unions` and the other patched modules.  All
# writers in this module funnel their derived frames through this helper
# so :mod:`._flex_data_accumulator` can capture them via its monkey-patch.
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1] and row[2]:
                out.append((row[0], row[1], row[2]))
    return out


# ---------------------------------------------------------------------------
# Family — pdtProcess (mod L1227).
# ---------------------------------------------------------------------------

def derive_pdtProcess(input_dir: Path, solve_data_dir: Path) -> pl.DataFrame:
    """Materialise the ``pdtProcess`` frame.

    Columns: ``process, param, period, time, value`` — all ``Utf8``
    (value cells are ``repr(v)`` so int/float distinction round-trips
    byte-identically to the legacy ``f",{repr(v)}\\n"`` emit).

    Domain: ``process_TimeParam_in_use × steps_in_use`` (dense; the
    sparse-emit optimisation from
    ``specs/sparse_writer_lessons_for_engine_polars.md`` §1 is NOT
    applied — preserving byte-parity with the legacy writer is the
    Phase E-b gate).
    """
    lookup = PdtLookup(
        pbt_csv=input_dir / "pbt_process.csv",
        pd_csv=input_dir / "pd_process.csv",
        pt_csv=input_dir / "pt_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_entity_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        param_def1=PROCESS_PARAM_DEF1,
    )
    domain = _read_pairs(solve_data_dir / "process_TimeParam_in_use.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    n = len(domain) * len(dt)
    processes: list[str] = [""] * n
    params: list[str] = [""] * n
    periods: list[str] = [""] * n
    times: list[str] = [""] * n
    values: list[str] = [""] * n
    i = 0
    for (p, param) in domain:
        for (d, t) in dt:
            v = lookup.get(p, param, d, t)
            processes[i] = p
            params[i] = param
            periods[i] = d
            times[i] = t
            values[i] = repr(v)
            i += 1
    return pl.DataFrame(
        {
            "process": processes,
            "param": params,
            "period": periods,
            "time": times,
            "value": values,
        },
        schema={
            "process": pl.Utf8,
            "param": pl.Utf8,
            "period": pl.Utf8,
            "time": pl.Utf8,
            "value": pl.Utf8,
        },
    )


def write_pdtProcess(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtProcess.csv`` (see derive docstring)."""
    _write(
        derive_pdtProcess(input_dir, solve_data_dir),
        solve_data_dir / "pdtProcess.csv",
    )


# ---------------------------------------------------------------------------
# Family — pdtNode (mod L1176; time-first + class default).
# ---------------------------------------------------------------------------

def derive_pdtNode(input_dir: Path, solve_data_dir: Path) -> pl.DataFrame:
    """Materialise the ``pdtNode`` frame.

    Columns: ``node, param, period, time, value`` — all ``Utf8``.
    Domain: ``node__TimeParam_in_use × steps_in_use``.
    """
    lookup = PdtLookup(
        pbt_csv=input_dir / "pbt_node.csv",
        pd_csv=input_dir / "pd_node.csv",
        pt_csv=input_dir / "pt_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_entity_csv=input_dir / "group__node.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        param_def1=NODE_PARAM_DEF1,
        time_first_priority=True,
        class_default_values=read_class_defaults(
            input_dir / "default_values.csv", "node"
        ),
    )
    domain = _read_pairs(solve_data_dir / "node__TimeParam_in_use.csv")
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")

    n = len(domain) * len(dt)
    nodes: list[str] = [""] * n
    params: list[str] = [""] * n
    periods: list[str] = [""] * n
    times: list[str] = [""] * n
    values: list[str] = [""] * n
    i = 0
    for (nd, param) in domain:
        for (d, t) in dt:
            v = lookup.get(nd, param, d, t)
            nodes[i] = nd
            params[i] = param
            periods[i] = d
            times[i] = t
            values[i] = repr(v)
            i += 1
    return pl.DataFrame(
        {
            "node": nodes,
            "param": params,
            "period": periods,
            "time": times,
            "value": values,
        },
        schema={
            "node": pl.Utf8,
            "param": pl.Utf8,
            "period": pl.Utf8,
            "time": pl.Utf8,
            "value": pl.Utf8,
        },
    )


def write_pdtNode(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtNode.csv`` (see derive docstring)."""
    _write(
        derive_pdtNode(input_dir, solve_data_dir),
        solve_data_dir / "pdtNode.csv",
    )


# ---------------------------------------------------------------------------
# Family — pdtProcess_source / pdtProcess_sink (mod L1265 / L1279).
# ---------------------------------------------------------------------------

def _derive_pdtProcess_side(
    *,
    pbt_csv: Path,
    pd_csv: Path,
    pt_csv: Path,
    p_csv: Path,
    period_time_first_csv: Path,
    solve_branch_csv: Path,
    period_branch_csv: Path,
    group_process_csv: Path,
    group_stochastic_csv: Path,
    domain_csv: Path,
    dt_csv: Path,
    side_col: str,
) -> pl.DataFrame:
    lookup = PdtLookupPerSide(
        pbt_csv=pbt_csv,
        pd_csv=pd_csv,
        pt_csv=pt_csv,
        p_csv=p_csv,
        period_time_first_csv=period_time_first_csv,
        solve_branch_csv=solve_branch_csv,
        period_branch_csv=period_branch_csv,
        group_process_csv=group_process_csv,
        group_stochastic_csv=group_stochastic_csv,
    )
    domain = _read_triples(domain_csv)
    dt = _read_pairs(dt_csv)

    n = len(domain) * len(dt)
    processes: list[str] = [""] * n
    sides: list[str] = [""] * n
    params: list[str] = [""] * n
    periods: list[str] = [""] * n
    times: list[str] = [""] * n
    values: list[str] = [""] * n
    i = 0
    for (p, side, param) in domain:
        for (d, t) in dt:
            v = lookup.get(p, side, param, d, t)
            processes[i] = p
            sides[i] = side
            params[i] = param
            periods[i] = d
            times[i] = t
            values[i] = repr(v)
            i += 1
    return pl.DataFrame(
        {
            "process": processes,
            side_col: sides,
            "param": params,
            "period": periods,
            "time": times,
            "value": values,
        },
        schema={
            "process": pl.Utf8,
            side_col: pl.Utf8,
            "param": pl.Utf8,
            "period": pl.Utf8,
            "time": pl.Utf8,
            "value": pl.Utf8,
        },
    )


def derive_pdtProcess_source(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """Materialise ``pdtProcess_source`` (mod L1265, 6-branch per-side).

    Columns: ``process, source, param, period, time, value`` (all Utf8).
    Domain: ``process_source_sourceSinkTimeParam_in_use × steps_in_use``.
    """
    return _derive_pdtProcess_side(
        pbt_csv=input_dir / "pbt_process_source.csv",
        pd_csv=input_dir / "pd_process_source.csv",
        pt_csv=input_dir / "pt_process_source.csv",
        p_csv=input_dir / "p_process_source.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        domain_csv=solve_data_dir / "process_source_sourceSinkTimeParam_in_use.csv",
        dt_csv=solve_data_dir / "steps_in_use.csv",
        side_col="source",
    )


def write_pdtProcess_source(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtProcess_source.csv`` (see derive docstring)."""
    _write(
        derive_pdtProcess_source(input_dir, solve_data_dir),
        solve_data_dir / "pdtProcess_source.csv",
    )


def derive_pdtProcess_sink(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """Materialise ``pdtProcess_sink`` (mod L1279, 6-branch per-side).

    Columns: ``process, sink, param, period, time, value`` (all Utf8).
    Domain: ``process_sink_sourceSinkTimeParam_in_use × steps_in_use``.
    """
    return _derive_pdtProcess_side(
        pbt_csv=input_dir / "pbt_process_sink.csv",
        pd_csv=input_dir / "pd_process_sink.csv",
        pt_csv=input_dir / "pt_process_sink.csv",
        p_csv=input_dir / "p_process_sink.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        domain_csv=solve_data_dir / "process_sink_sourceSinkTimeParam_in_use.csv",
        dt_csv=solve_data_dir / "steps_in_use.csv",
        side_col="sink",
    )


def write_pdtProcess_sink(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit ``solve_data/pdtProcess_sink.csv`` (see derive docstring)."""
    _write(
        derive_pdtProcess_sink(input_dir, solve_data_dir),
        solve_data_dir / "pdtProcess_sink.csv",
    )
