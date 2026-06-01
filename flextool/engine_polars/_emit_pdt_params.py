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
    _provider_key,
)
from flextool.engine_polars._pdt_lookup import (
    NODE_PARAM_DEF1,
    PROCESS_PARAM_DEF1,
    PdtLookup,
    PdtLookupPerSide,
    read_class_defaults,
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

def _derive_pdtProcess_over(
    input_dir: Path,
    solve_data_dir: Path,
    *,
    dt: list[tuple[str, str]],
    period_time_first_csv: Path,
    solve_branch_csv: Path,
    period_branch_csv: Path,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Core of :func:`derive_pdtProcess`: build the frame over an explicit
    ``(period, time)`` list *dt* and explicit branch-map CSV paths.

    The value at ``(process, param, d, t)`` is a pure coordinate lookup
    (the :class:`PdtLookup` cascade) over fixed inputs — ZERO window
    aggregation — so the result is coordinate-stable (specs §9.5):
    deriving over the complete (branch-expanded) timeline and slicing to a
    roll yields exactly the same rows as deriving over the roll window
    directly, PROVIDED the branch maps span the branch-periods the value
    depends on.  :func:`derive_pdtProcess` passes the per-roll
    ``steps_in_use`` + per-roll branch maps; :func:`derive_pdtProcess_full_dt`
    passes the branch-EXPANDED complete timeline + complete branch maps.
    """
    lookup = PdtLookup(
        pbt_csv=input_dir / "pbt_process.csv",
        pd_csv=input_dir / "pd_process.csv",
        pt_csv=input_dir / "pt_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_time_first_csv=period_time_first_csv,
        solve_branch_csv=solve_branch_csv,
        period_branch_csv=period_branch_csv,
        group_entity_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        param_def1=PROCESS_PARAM_DEF1,
        provider=provider,
    )
    domain = _read_pairs(
        solve_data_dir / "process_TimeParam_in_use.csv", provider=provider,
    )

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


def derive_pdtProcess(input_dir: Path, solve_data_dir: Path,
                      *,
                      provider: "object | None" = None) -> pl.DataFrame:
    """Materialise the ``pdtProcess`` frame over the ROLL window.

    Columns: ``process, param, period, time, value`` — all ``Utf8``
    (value cells are ``repr(v)`` so int/float distinction round-trips
    byte-identically to the legacy ``f",{repr(v)}\\n"`` emit).

    Domain: ``process_TimeParam_in_use × steps_in_use`` (dense; the
    sparse-emit optimisation from
    ``specs/sparse_writer_lessons_for_engine_polars.md`` §1 is NOT
    applied — preserving byte-parity with the legacy writer is the
    Phase E-b gate).  This is the legacy per-roll derivation, retained as
    the fallback for :func:`emit_pdtProcess` when no cached full-domain
    frame is present.
    """
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)
    return _derive_pdtProcess_over(
        input_dir, solve_data_dir, dt=dt,
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )


def derive_pdtProcess_full_dt(input_dir: Path, solve_data_dir: Path,
                              *,
                              provider: "object | None" = None,
                              ) -> pl.DataFrame:
    """Materialise the ``pdtProcess`` frame over the BRANCH-EXPANDED COMPLETE
    timeline (R4).

    Identical to :func:`derive_pdtProcess` except the ``(d, t)`` grid is the
    branch-expanded complete timeline and the ``PdtLookup`` branch maps are
    the complete-timeline ``period__branch`` / ``solve_branch__time_branch`` /
    ``first_timesteps`` — all read from the ``_full_dt_branched/`` Provider
    keys the runner populates once per ``complete_solve`` (see
    :func:`flextool.engine_polars._full_dt.build_branch_expanded_complete_frames`).
    Because pdtProcess resolves DISTINCT per-branch values at the
    branch-periods, deriving over the BASE ``steps_complete_solve`` would drop
    those rows; the branch-expanded complete timeline carries them, so the
    per-roll slice is value-identical to the legacy per-roll derivation.

    Computed once per ``complete_solve`` and sliced per roll via
    :func:`flextool.engine_polars._full_dt.slice_full_dt`.
    """
    from flextool.engine_polars import _full_dt as _fdt
    dt = _read_pairs(
        _fdt.branched_path(solve_data_dir, "steps_complete_solve_branched"),
        provider=provider,
    )
    return _derive_pdtProcess_over(
        input_dir, solve_data_dir, dt=dt,
        period_time_first_csv=_fdt.branched_path(
            solve_data_dir, "first_timesteps"),
        solve_branch_csv=_fdt.branched_path(
            solve_data_dir, "solve_branch__time_branch"),
        period_branch_csv=_fdt.branched_path(
            solve_data_dir, "period__branch"),
        provider=provider,
    )


def emit_pdtProcess(input_dir: Path, solve_data_dir: Path,
                    *, provider,
                    full_dt_frame: "pl.DataFrame | None" = None) -> None:
    """Emit ``pdtProcess`` to the Provider.

    When *full_dt_frame* is supplied (the branch-expanded complete-timeline
    frame computed once per ``complete_solve`` and cached on the runner
    state) the roll's frame is obtained by slicing it to the roll's
    ``steps_in_use`` window via :func:`slice_full_dt` —
    compute-once-over-full_dt + slice-per-roll.  When it is absent the legacy
    per-roll :func:`derive_pdtProcess` runs as a fallback.  Either way the
    emitted key and frame content are byte-identical (the slice is
    value-preserving for this coordinate-stable family; specs §9.5).
    """
    from flextool.engine_polars import _full_dt as _fdt
    _family = next(
        f for f in _fdt.FULL_DT_FAMILIES if f.name == "pdtProcess"
    )
    _fdt.emit_via_full_dt(
        _family, input_dir, solve_data_dir,
        provider=provider, full_dt_frame=full_dt_frame,
    )


# ---------------------------------------------------------------------------
# Family — pdtNode (mod L1176; time-first + class default).
# ---------------------------------------------------------------------------

def _derive_pdtNode_over(
    input_dir: Path,
    solve_data_dir: Path,
    *,
    dt: list[tuple[str, str]],
    domain: list[tuple[str, str]],
    period_time_first_csv: Path,
    solve_branch_csv: Path,
    period_branch_csv: Path,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Core of :func:`derive_pdtNode`: build the frame over an explicit
    ``(period, time)`` list *dt*, an explicit ``(node, param)`` *domain*, and
    explicit branch-map CSV paths.

    The value at ``(node, param, d, t)`` is a pure coordinate lookup (the
    :class:`PdtLookup` cascade) over fixed inputs — ZERO window aggregation —
    so the result is coordinate-stable (specs §9.5): deriving over the
    complete (branch-expanded) timeline and slicing to a roll yields exactly
    the same rows as deriving over the roll window directly, PROVIDED the
    branch maps span the branch-periods the value depends on.
    :func:`derive_pdtNode` passes the per-roll ``steps_in_use`` + per-roll
    branch maps; :func:`derive_pdtNode_full_dt` passes the branch-EXPANDED
    complete timeline + complete branch maps.
    """
    lookup = PdtLookup(
        pbt_csv=input_dir / "pbt_node.csv",
        pd_csv=input_dir / "pd_node.csv",
        pt_csv=input_dir / "pt_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_time_first_csv=period_time_first_csv,
        solve_branch_csv=solve_branch_csv,
        period_branch_csv=period_branch_csv,
        group_entity_csv=input_dir / "group__node.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        param_def1=NODE_PARAM_DEF1,
        time_first_priority=True,
        class_default_values=read_class_defaults(
            input_dir / "default_values.csv", "node", provider=provider,
        ),
        provider=provider,
    )

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


def derive_pdtNode(input_dir: Path, solve_data_dir: Path,
                    *,
                    provider: "object | None" = None) -> pl.DataFrame:
    """Materialise the ``pdtNode`` frame over the ROLL window.

    Columns: ``node, param, period, time, value`` — all ``Utf8``.
    Domain: ``node__TimeParam_in_use × steps_in_use``.  This is the legacy
    per-roll derivation, retained as the :func:`emit_pdtNode` fallback when no
    cached full-domain frame is present; its domain is read from the per-solve
    ``node__TimeParam_in_use.csv`` written by
    :func:`flextool.engine_polars._emit_arc_unions.emit_node_time_param_in_use`.
    """
    domain = _read_pairs(
        solve_data_dir / "node__TimeParam_in_use.csv", provider=provider,
    )
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)
    return _derive_pdtNode_over(
        input_dir, solve_data_dir, dt=dt, domain=domain,
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )


def derive_pdtNode_full_dt(input_dir: Path, solve_data_dir: Path,
                           *,
                           provider: "object | None" = None,
                           ) -> pl.DataFrame:
    """Materialise the ``pdtNode`` frame over the BRANCH-EXPANDED COMPLETE
    timeline (R4 / R6a).

    Identical to :func:`derive_pdtNode` except:

    * the ``(d, t)`` grid is the branch-expanded complete timeline and the
      ``PdtLookup`` branch maps are the complete-timeline ``period__branch`` /
      ``solve_branch__time_branch`` / ``first_timesteps`` — all read from the
      ``_full_dt_branched/`` Provider keys the runner populates once per
      ``complete_solve`` (see
      :func:`flextool.engine_polars._full_dt.build_branch_expanded_complete_frames`),
      exactly like :func:`derive_pdtProcess_full_dt`; and
    * the ``(node, param)`` DOMAIN is RECONSTRUCTED here (R6a) by calling
      :func:`flextool.engine_polars._emit_arc_unions.derive_node_time_param_in_use`
      directly, rather than reading the per-solve
      ``node__TimeParam_in_use.csv``.  That per-solve CSV is emitted by the
      per-roll dispatcher (``_emit_solve_time`` L2 batch 22) and is therefore
      ABSENT at full_dt compute time (first roll, before the dispatcher runs).
      Its inputs — ``node.csv`` / ``node__storage_solve_horizon_method.csv``
      (solve-invariant inputs) and the node-type partitions
      ``nodeBalance`` / ``nodeBalancePeriod`` / ``nodeState`` (emitted by
      ``emit_node_type_sets`` purely from ``node.csv`` + ``p_node_type.csv``,
      discarding ``solve_data_dir``, hence solve-INVARIANT) — are all present
      in the Provider at compute time, so the domain reconstructs identically
      to the per-solve value.  The domain is solve-invariant, so a single
      complete-timeline reconstruction is value-identical to every roll's
      ``node__TimeParam_in_use``.

    Because pdtNode resolves DISTINCT per-branch values at the branch-periods,
    deriving over the BASE ``steps_complete_solve`` would drop those rows; the
    branch-expanded complete timeline carries them, so the per-roll slice is
    value-identical to the legacy per-roll derivation.

    Computed once per ``complete_solve`` and sliced per roll via
    :func:`flextool.engine_polars._full_dt.slice_full_dt`.
    """
    from flextool.engine_polars import _full_dt as _fdt
    from flextool.engine_polars._emit_arc_unions import (
        derive_node_time_param_in_use,
    )
    # R6a: reconstruct the (node, param) domain from solve-invariant inputs
    # present at full_dt compute time, NOT from the per-solve CSV (absent here).
    domain_df = derive_node_time_param_in_use(
        input_dir, solve_data_dir, provider=provider,
    )
    domain = list(
        zip(
            domain_df.get_column("node").to_list(),
            domain_df.get_column("param").to_list(),
        )
    )
    dt = _read_pairs(
        _fdt.branched_path(solve_data_dir, "steps_complete_solve_branched"),
        provider=provider,
    )
    return _derive_pdtNode_over(
        input_dir, solve_data_dir, dt=dt, domain=domain,
        period_time_first_csv=_fdt.branched_path(
            solve_data_dir, "first_timesteps"),
        solve_branch_csv=_fdt.branched_path(
            solve_data_dir, "solve_branch__time_branch"),
        period_branch_csv=_fdt.branched_path(
            solve_data_dir, "period__branch"),
        provider=provider,
    )


def emit_pdtNode(input_dir: Path, solve_data_dir: Path,
                  *, provider,
                  full_dt_frame: "pl.DataFrame | None" = None) -> None:
    """Emit ``pdtNode`` to the Provider.

    When *full_dt_frame* is supplied (the branch-expanded complete-timeline
    frame computed once per ``complete_solve`` and cached on the runner state)
    the roll's frame is obtained by slicing it to the roll's ``steps_in_use``
    window via :func:`slice_full_dt` — compute-once-over-full_dt +
    slice-per-roll.  When it is absent the legacy per-roll
    :func:`derive_pdtNode` runs as a fallback.  Either way the emitted key and
    frame content are byte-identical (the slice is value-preserving for this
    coordinate-stable family; specs §9.5).
    """
    from flextool.engine_polars import _full_dt as _fdt
    _family = next(
        f for f in _fdt.FULL_DT_FAMILIES if f.name == "pdtNode"
    )
    _fdt.emit_via_full_dt(
        _family, input_dir, solve_data_dir,
        provider=provider, full_dt_frame=full_dt_frame,
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
    dt: list[tuple[str, str]],
    side_col: str,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Core per-side derivation over an explicit ``(period, time)`` list *dt*
    and explicit branch-map CSV paths.

    The value at ``(process, side, param, d, t)`` is a pure coordinate lookup
    (the :class:`PdtLookupPerSide` cascade) over fixed inputs — ZERO window
    aggregation — so the result is coordinate-stable (specs §9.5): deriving
    over the complete (branch-expanded) timeline and slicing to a roll yields
    exactly the same rows as deriving over the roll window directly, PROVIDED
    the branch maps span the branch-periods the value depends on.
    """
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


def _derive_pdtProcess_side_roll(
    input_dir: Path, solve_data_dir: Path,
    *,
    side: str,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Per-roll per-side derivation (over the roll's ``steps_in_use`` window
    + per-roll branch maps).  Shared body for ``source`` / ``sink``; *side*
    selects the input shard prefix, domain CSV, and side column name."""
    dt = _read_pairs(solve_data_dir / "steps_in_use.csv", provider=provider)
    return _derive_pdtProcess_side(
        pbt_csv=input_dir / f"pbt_process_{side}.csv",
        pd_csv=input_dir / f"pd_process_{side}.csv",
        pt_csv=input_dir / f"pt_process_{side}.csv",
        p_csv=input_dir / f"p_process_{side}.csv",
        period_time_first_csv=solve_data_dir / "first_timesteps.csv",
        solve_branch_csv=solve_data_dir / "solve_branch__time_branch.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        group_process_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        domain_csv=solve_data_dir
        / f"process_{side}_sourceSinkTimeParam_in_use.csv",
        dt=dt,
        side_col=side,
        provider=provider,
    )


def _derive_pdtProcess_side_full_dt(
    input_dir: Path, solve_data_dir: Path,
    *,
    side: str,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Per-side derivation over the BRANCH-EXPANDED COMPLETE timeline (R4).

    Identical to :func:`_derive_pdtProcess_side_roll` except the ``(d, t)``
    grid is the branch-expanded complete timeline and the branch maps are the
    complete-timeline ``period__branch`` / ``solve_branch__time_branch`` /
    ``first_timesteps`` — all read from the ``_full_dt_branched/`` Provider
    keys the runner populates once per ``complete_solve`` (see
    :func:`flextool.engine_polars._full_dt.build_branch_expanded_complete_frames`).
    Because the per-side lookup resolves DISTINCT per-branch values at the
    branch-periods, deriving over the BASE ``steps_complete_solve`` would drop
    those rows; the branch-expanded complete timeline carries them, so the
    per-roll slice is value-identical to the legacy per-roll derivation."""
    from flextool.engine_polars import _full_dt as _fdt
    dt = _read_pairs(
        _fdt.branched_path(solve_data_dir, "steps_complete_solve_branched"),
        provider=provider,
    )
    return _derive_pdtProcess_side(
        pbt_csv=input_dir / f"pbt_process_{side}.csv",
        pd_csv=input_dir / f"pd_process_{side}.csv",
        pt_csv=input_dir / f"pt_process_{side}.csv",
        p_csv=input_dir / f"p_process_{side}.csv",
        period_time_first_csv=_fdt.branched_path(
            solve_data_dir, "first_timesteps"),
        solve_branch_csv=_fdt.branched_path(
            solve_data_dir, "solve_branch__time_branch"),
        period_branch_csv=_fdt.branched_path(
            solve_data_dir, "period__branch"),
        group_process_csv=input_dir / "group__process.csv",
        group_stochastic_csv=input_dir / "groupIncludeStochastics.csv",
        domain_csv=solve_data_dir
        / f"process_{side}_sourceSinkTimeParam_in_use.csv",
        dt=dt,
        side_col=side,
        provider=provider,
    )


def derive_pdtProcess_source(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise ``pdtProcess_source`` over the ROLL window (mod L1265,
    6-branch per-side).

    Columns: ``process, source, param, period, time, value`` (all Utf8).
    Domain: ``process_source_sourceSinkTimeParam_in_use × steps_in_use``.
    Legacy per-roll derivation, retained as the :func:`emit_pdtProcess_source`
    fallback when no cached full-domain frame is present.
    """
    return _derive_pdtProcess_side_roll(
        input_dir, solve_data_dir, side="source", provider=provider,
    )


def derive_pdtProcess_source_full_dt(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise ``pdtProcess_source`` over the BRANCH-EXPANDED COMPLETE
    timeline (R4); computed once per ``complete_solve`` and sliced per roll
    via :func:`flextool.engine_polars._full_dt.slice_full_dt`."""
    return _derive_pdtProcess_side_full_dt(
        input_dir, solve_data_dir, side="source", provider=provider,
    )


def emit_pdtProcess_source(input_dir: Path, solve_data_dir: Path,
                           *, provider,
                           full_dt_frame: "pl.DataFrame | None" = None) -> None:
    """Emit ``pdtProcess_source`` to the Provider.

    When *full_dt_frame* is supplied (the branch-expanded complete-timeline
    frame cached on the runner state) the roll's frame is the slice of it to
    the roll's ``steps_in_use`` window; else the legacy per-roll derivation
    runs.  Either way the result is byte-identical (coordinate-stable family;
    specs §9.5).
    """
    from flextool.engine_polars import _full_dt as _fdt
    _family = next(
        f for f in _fdt.FULL_DT_FAMILIES if f.name == "pdtProcess_source"
    )
    _fdt.emit_via_full_dt(
        _family, input_dir, solve_data_dir,
        provider=provider, full_dt_frame=full_dt_frame,
    )


def derive_pdtProcess_sink(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise ``pdtProcess_sink`` over the ROLL window (mod L1279,
    6-branch per-side).

    Columns: ``process, sink, param, period, time, value`` (all Utf8).
    Domain: ``process_sink_sourceSinkTimeParam_in_use × steps_in_use``.
    Legacy per-roll derivation, retained as the :func:`emit_pdtProcess_sink`
    fallback when no cached full-domain frame is present.
    """
    return _derive_pdtProcess_side_roll(
        input_dir, solve_data_dir, side="sink", provider=provider,
    )


def derive_pdtProcess_sink_full_dt(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Materialise ``pdtProcess_sink`` over the BRANCH-EXPANDED COMPLETE
    timeline (R4); computed once per ``complete_solve`` and sliced per roll
    via :func:`flextool.engine_polars._full_dt.slice_full_dt`."""
    return _derive_pdtProcess_side_full_dt(
        input_dir, solve_data_dir, side="sink", provider=provider,
    )


def emit_pdtProcess_sink(input_dir: Path, solve_data_dir: Path,
                         *, provider,
                         full_dt_frame: "pl.DataFrame | None" = None) -> None:
    """Emit ``pdtProcess_sink`` to the Provider.

    When *full_dt_frame* is supplied (the branch-expanded complete-timeline
    frame cached on the runner state) the roll's frame is the slice of it to
    the roll's ``steps_in_use`` window; else the legacy per-roll derivation
    runs.  Either way the result is byte-identical (coordinate-stable family;
    specs §9.5).
    """
    from flextool.engine_polars import _full_dt as _fdt
    _family = next(
        f for f in _fdt.FULL_DT_FAMILIES if f.name == "pdtProcess_sink"
    )
    _fdt.emit_via_full_dt(
        _family, input_dir, solve_data_dir,
        provider=provider, full_dt_frame=full_dt_frame,
    )
