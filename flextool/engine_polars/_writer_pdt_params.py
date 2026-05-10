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

Output CSVs are written row-by-row with ``repr(v)`` on the value column
to preserve precision parity with the legacy writers (see
:mod:`._writer_calc_params` module docstring for the rationale).
"""
from __future__ import annotations

import csv
from pathlib import Path

from flextool.engine_polars._pdt_lookup import (
    NODE_PARAM_DEF1,
    PROCESS_PARAM_DEF1,
    PdtLookup,
    PdtLookupPerSide,
    read_class_defaults,
)


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

def write_pdtProcess(input_dir: Path, solve_data_dir: Path) -> None:
    """7-branch fallback over pbt/pd/pt/p + processParam_def1 + 0.

    Output: ``solve_data/pdtProcess.csv`` indexed by (process, param,
    period, time).  Domain: ``process_TimeParam_in_use × dt``.
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
    out_path = solve_data_dir / "pdtProcess.csv"
    with out_path.open("w") as fh:
        fh.write("process,param,period,time,value\n")
        for (p, param) in domain:
            for (d, t) in dt:
                v = lookup.get(p, param, d, t)
                fh.write(f"{p},{param},{d},{t},{repr(v)}\n")


# ---------------------------------------------------------------------------
# Family — pdtNode (mod L1176; time-first + class default).
# ---------------------------------------------------------------------------

def write_pdtNode(input_dir: Path, solve_data_dir: Path) -> None:
    """9-branch fallback over pbt/pt/pd/p + nodeParam_def1 +
    class_paramName_default + 0.  Time axis precedes period.
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
    out_path = solve_data_dir / "pdtNode.csv"
    with out_path.open("w") as fh:
        fh.write("node,param,period,time,value\n")
        for (n, param) in domain:
            for (d, t) in dt:
                v = lookup.get(n, param, d, t)
                fh.write(f"{n},{param},{d},{t},{repr(v)}\n")


# ---------------------------------------------------------------------------
# Family — pdtProcess_source / pdtProcess_sink (mod L1265 / L1279).
# ---------------------------------------------------------------------------

def write_pdtProcess_source(input_dir: Path, solve_data_dir: Path) -> None:
    """6-branch fallback (no def1).  Domain:
    ``process_source_sourceSinkTimeParam_in_use × dt``."""
    lookup = PdtLookupPerSide(
        pbt_csv=input_dir / "pbt_process_source.csv",
        pd_csv=input_dir / "pd_process_source.csv",
        pt_csv=input_dir / "pt_process_source.csv",
        p_csv=input_dir / "p_process_source.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
    )
    domain = _read_triples(
        solve_data_dir / "process_source_sourceSinkTimeParam_in_use.csv"
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    out_path = solve_data_dir / "pdtProcess_source.csv"
    with out_path.open("w") as fh:
        fh.write("process,source,param,period,time,value\n")
        for (p, src, param) in domain:
            for (d, t) in dt:
                v = lookup.get(p, src, param, d, t)
                fh.write(f"{p},{src},{param},{d},{t},{repr(v)}\n")


def write_pdtProcess_sink(input_dir: Path, solve_data_dir: Path) -> None:
    """6-branch fallback (no def1).  Domain:
    ``process_sink_sourceSinkTimeParam_in_use × dt``."""
    lookup = PdtLookupPerSide(
        pbt_csv=input_dir / "pbt_process_sink.csv",
        pd_csv=input_dir / "pd_process_sink.csv",
        pt_csv=input_dir / "pt_process_sink.csv",
        p_csv=input_dir / "p_process_sink.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
    )
    domain = _read_triples(
        solve_data_dir / "process_sink_sourceSinkTimeParam_in_use.csv"
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv")
    out_path = solve_data_dir / "pdtProcess_sink.csv"
    with out_path.open("w") as fh:
        fh.write("process,sink,param,period,time,value\n")
        for (p, snk, param) in domain:
            for (d, t) in dt:
                v = lookup.get(p, snk, param, d, t)
                fh.write(f"{p},{snk},{param},{d},{t},{repr(v)}\n")
