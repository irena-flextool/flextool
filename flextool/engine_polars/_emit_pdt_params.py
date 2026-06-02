"""pdtProcess / pdtNode / pdtProcess_{source,sink} writers.

Four ``write_pdt*`` helpers that consume the ``PdtLookup`` /
``PdtLookupPerSide`` class hierarchy.

The lookup classes live in :mod:`._pdt_lookup`; this module
orchestrates the per-row resolution against fixture inputs and emits
the canonical CSV format.

Writers:

* ``write_pdtProcess``         — 7-branch.
* ``write_pdtNode``            — 9-branch (time-first + class-default
  fallback).
* ``write_pdtProcess_source``  — 6-branch, per-side.
* ``write_pdtProcess_sink``    — 6-branch, per-side.

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
same pattern :mod:`._emit_chain_params` uses for its
``_ed_value_frame`` helper).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine_polars._pdt_lookup import (
    NODE_PARAM_DEF1,
    PROCESS_PARAM_DEF1,
    PdtLookup,
    PdtLookupPerSide,
    read_class_defaults,
)
from flextool.engine_polars._vectorize import (
    build_entity_dt_grid,
    build_fold_frame,
    coalesce_value,
    collect_value_frame,
    lift_dict_to_lookup,
)


def _cell_str(value: "object | None") -> str:
    """Reproduce a ``csv.reader`` cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; ``csv.reader`` then reads those
    strings back.  Mirror that so structural string keys stay
    byte-identical to the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


def _read_pairs(path: Path,
                *,
                provider: "object | None" = None) -> list[tuple[str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str]] = []
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.append((c0, c1))
    return out


def _read_triples(path: Path,
                  *,
                  provider: "object | None" = None) -> list[tuple[str, str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str, str]] = []
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
        if c0 and c1 and c2:
            out.append((c0, c1, c2))
    return out


# ---------------------------------------------------------------------------
# Family — pdtProcess (mod L1227).
# ---------------------------------------------------------------------------

def derive_pdtProcess(input_dir: Path, solve_data_dir: Path,
                       *,
                       provider: "object | None" = None) -> pl.DataFrame:
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
        provider=provider,
    )
    domain = _read_pairs(
        solve_data_dir / "process_TimeParam_in_use.csv", provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

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


def derive_pdtProcess_vectorized(input_dir: Path, solve_data_dir: Path,
                                  *,
                                  provider: "object | None" = None,
                                  engine: str = "eager") -> pl.DataFrame:
    """Vectorized ``pdtProcess`` derive — byte-parity with the legacy.

    Replaces the per-cell ``PdtLookup.get`` loop in
    :func:`derive_pdtProcess` with vectorized polars (left-joins +
    ``coalesce`` in cascade-priority order + the group-by-sum folds),
    still per roll over the roll's own window.  Output is byte-identical
    to :func:`derive_pdtProcess`: columns ``process, param, period,
    time, value`` all ``Utf8``, entity-major row order, ``repr(v)``
    value cells.

    Cascade (``PdtLookup`` 7-branch, period-first):
    fold(branch1 stoch + branch2 parent) → ``pd`` → ``pt`` → ``p`` →
    ``param ∈ PROCESS_PARAM_DEF1`` → ``1.0`` → ``0.0``.
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
        provider=provider,
    )
    domain = _read_pairs(
        solve_data_dir / "process_TimeParam_in_use.csv", provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

    key_cols = ["process", "param"]
    out_cols = [*key_cols, "period", "time"]

    grid = build_entity_dt_grid(domain, dt, key_cols=key_cols)

    pd_df = lift_dict_to_lookup(
        lookup._pd, ["process", "param", "period"], "v_pd")
    pt_df = lift_dict_to_lookup(
        lookup._pt, ["process", "param", "time"], "v_pt")
    p_df = lift_dict_to_lookup(
        lookup._p, ["process", "param"], "v_p")

    periods = [d for (d, _t) in dt]
    fold = build_fold_frame(
        pbt=lookup._pbt,
        pbt_key_cols=key_cols,
        out_key_cols=out_cols,
        ts_for_d=lookup._ts_for_d,
        tb_for_d=lookup._tb_for_d,
        pe_for_d=lookup._pe_for_d,
        stoch_entities=lookup._stoch_entity,
        stoch_filter_cols=["process"],
        periods=periods,
    )

    out = (
        grid
        .join(pd_df, on=["process", "param", "period"], how="left")
        .join(pt_df, on=["process", "param", "time"], how="left")
        .join(p_df, on=["process", "param"], how="left")
    )
    if fold is not None:
        out = out.join(fold, on=out_cols, how="left")
    else:
        out = out.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("v_fold"))

    def1 = list(PROCESS_PARAM_DEF1)
    out = out.with_columns(
        coalesce_value([
            pl.col("v_fold"),   # branches 1-2 (fold)
            pl.col("v_pd"),     # branch 3 (period-first)
            pl.col("v_pt"),     # branch 4
            pl.col("v_p"),      # branch 5
            pl.when(pl.col("param").is_in(def1))
              .then(pl.lit(1.0)).otherwise(pl.lit(0.0)),  # branch 6 / 8
        ])
    )
    return collect_value_frame(
        out, key_cols=out_cols, engine=engine)


def emit_pdtProcess(input_dir: Path, solve_data_dir: Path,
                     *, provider) -> None:
    """Emit ``pdtProcess`` to the Provider."""
    _emit(provider, "solve_data/pdtProcess.csv",
          derive_pdtProcess_vectorized(
              input_dir, solve_data_dir, provider=provider))


# ---------------------------------------------------------------------------
# Family — pdtNode (mod L1176; time-first + class default).
# ---------------------------------------------------------------------------

def derive_pdtNode(input_dir: Path, solve_data_dir: Path,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
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
            input_dir / "default_values.csv", "node", provider=provider,
        ),
        provider=provider,
    )
    domain = _read_pairs(
        solve_data_dir / "node__TimeParam_in_use.csv", provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)

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


def emit_pdtNode(input_dir: Path, solve_data_dir: Path,
                  *, provider) -> None:
    """Emit ``pdtNode`` to the Provider."""
    _emit(provider, "solve_data/pdtNode.csv",
          derive_pdtNode(input_dir, solve_data_dir, provider=provider))


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
    provider: "object | None" = None,
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
        provider=provider,
    )
    domain = _read_triples(domain_csv, provider=provider)
    dt = _read_pairs(dt_csv, provider=provider)

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
    *,
    provider: "object | None" = None,
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
        provider=provider,
    )


def emit_pdtProcess_source(input_dir: Path, solve_data_dir: Path,
                            *, provider) -> None:
    """Emit ``pdtProcess_source`` to the Provider."""
    _emit(
        provider, "solve_data/pdtProcess_source.csv",
        derive_pdtProcess_source(input_dir, solve_data_dir, provider=provider),
    )


def derive_pdtProcess_sink(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
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
        provider=provider,
    )


def emit_pdtProcess_sink(input_dir: Path, solve_data_dir: Path,
                          *, provider) -> None:
    """Emit ``pdtProcess_sink`` to the Provider."""
    _emit(
        provider, "solve_data/pdtProcess_sink.csv",
        derive_pdtProcess_sink(input_dir, solve_data_dir, provider=provider),
    )
