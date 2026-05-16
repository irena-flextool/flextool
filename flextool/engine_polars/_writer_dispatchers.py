"""Writer-port Phase 1 closeout — top-level dispatcher own-compute.

Native ports of the two remaining top-level "dispatcher" functions in
the legacy preprocessing tree.  Both functions are *own-compute*: they
contain inline derivations (loops, projections, unions, gated
constructions) rather than delegating to sibling ``write_*`` helpers.

* :func:`write_process_arc_unions` — mirrors
  ``flextool.flextoolrunner.preprocessing.process_arc_unions
  .write_process_arc_unions`` (~216 LOC).

  Migrates the 14-set L1 arc-union batch in dependency order:
  ``process__profileProcess__toSink``,
  ``process__source__toProfileProcess``,
  ``process_profile``,
  ``process_source_toProcess``,
  ``process_process_toSink``,
  ``process_source_sink_eff``,
  ``process_source_sink_noEff``,
  ``process_online``,
  ``process_minload``,
  ``process__commodity__node_co2``,
  ``process_co2``,
  ``process_source_sink``,
  ``process_source_sink_alwaysProcess``,
  ``process__source__sink__profile__profile_method_direct``.

* :func:`write_entity_period_calc_params` — mirrors
  ``flextool.flextoolrunner.preprocessing.entity_period_calc_params
  .write_entity_period_calc_params`` (~138 LOC).

  Emits ``pdProcess.csv``, ``pdNode.csv``, ``edEntity_lifetime.csv``,
  ``ed_fixed_cost.csv``, ``p_entity_unitsize.csv`` via the
  ``PdLookup`` machinery (native shim in :mod:`._pdt_lookup`).

These are byte-for-byte mirrors of the legacy emitters — same CSV
header, same per-row formatting, same iteration order.  The parity
tests under ``tests/engine_polars/test_writer_port_phase1.py`` assert
the file-level equivalence with ``filecmp``.

Note on Phase 1 vs Phase 2 boundary
-----------------------------------

``write_process_arc_unions`` is called from BOTH the Phase 1
top-level chain (``input_writer.write_input``) AND the Phase 2
per-solve chain (``preprocessing.solve_time``); the override hook
intercepts both call sites because they bind the same module attribute.

``write_entity_period_calc_params`` is currently only called from the
Phase 2 ``preprocessing.solve_time.preprocessing_solve_time`` chain
(no top-level call).  Porting it here keeps the dispatcher symmetric
and primes Phase 2 with a native implementation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Polars-frame _write helper — patched by Phase E-b accumulator.
#
# This is the single emission funnel for the converted derive_* family in
# this module.  The patched variant in
# :mod:`._flex_data_accumulator.capture_frames` rebinds this name to also
# capture (path.name -> df) into the per-sub-solve accumulator.
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


# ---------------------------------------------------------------------------
# CSV I/O — same helpers as the sibling legacy modules.
# ---------------------------------------------------------------------------


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


def _read_n_col(path: Path, n: int) -> list[tuple[str, ...]]:
    if not path.exists():
        return []
    out: list[tuple[str, ...]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= n and all(row[i] for i in range(n)):
                out.append(tuple(row[:n]))
    return out


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _write_csv(path: Path, header: tuple[str, ...], rows) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(",".join(r) + "\n" for r in rows))


# ---------------------------------------------------------------------------
# Method-constant frozensets — re-exported from the canonical legacy
# module ``flextool.flextoolrunner.preprocessing._method_constants`` to
# avoid drift.  These are model invariants from flextool_base.dat:60-95
# and are not user-editable.
# ---------------------------------------------------------------------------

from flextool.flextoolrunner.preprocessing._method_constants import (  # noqa: E402
    METHOD_INDIRECT as _METHOD_INDIRECT,
    METHOD_DIRECT as _METHOD_DIRECT,
)


# ---------------------------------------------------------------------------
# write_process_arc_unions — top-level dispatcher own-compute.
# Mirrors flextool.flextoolrunner.preprocessing.process_arc_unions
# .write_process_arc_unions lines 80-293 of the legacy module.
# ---------------------------------------------------------------------------


def write_process_arc_unions(input_dir: Path, solve_data_dir: Path) -> None:
    """Migrate the 14-set L1 arc-union batch in dependency order.

    Byte-for-byte mirror of the legacy emitter.
    """
    METHOD_INDIRECT = _METHOD_INDIRECT
    METHOD_DIRECT = _METHOD_DIRECT

    # ---- 1) process__profileProcess__toSink (project from 5-tuple set)
    five_tuple_to_sink = _read_n_col(
        solve_data_dir
        / "process__profileProcess__toSink__profile__profile_method.csv",
        5,
    )
    profile_to_sink_3 = list(dict.fromkeys(
        (p_outer, p, sink) for p_outer, p, sink, _f, _fm in five_tuple_to_sink
    ))
    _write_csv(
        solve_data_dir / "process__profileProcess__toSink.csv",
        ("process_outer", "process", "sink"),
        profile_to_sink_3,
    )

    # ---- 2) process__source__toProfileProcess (project from 5-tuple set)
    five_tuple_to_source = _read_n_col(
        solve_data_dir
        / "process__source__toProfileProcess__profile__profile_method.csv",
        5,
    )
    source_to_profile_3 = list(dict.fromkeys(
        (p, source, p_aux) for p, source, p_aux, _f, _fm in five_tuple_to_source
    ))
    _write_csv(
        solve_data_dir / "process__source__toProfileProcess.csv",
        ("process", "source", "process_aux"),
        source_to_profile_3,
    )

    # ---- 3) process_profile = setof p from (1) ∪ setof p from (2)
    seen_profile: dict[str, None] = {}
    for p, _, _ in source_to_profile_3:
        seen_profile.setdefault(p, None)
    for p, _, _ in profile_to_sink_3:
        seen_profile.setdefault(p, None)
    _write_csv(
        solve_data_dir / "process_profile.csv",
        ("process",),
        [(p,) for p in seen_profile.keys()],
    )

    # ---- 4) process_source_toProcess
    process_method = _read_pairs(input_dir / "process_method.csv")
    sources = _read_pairs(input_dir / "process__source.csv")
    sinks = _read_pairs(input_dir / "process__sink.csv")
    p_with_indirect = frozenset(p for p, m in process_method if m in METHOD_INDIRECT)
    p_with_direct = frozenset(p for p, m in process_method if m in METHOD_DIRECT)
    has_sink = frozenset(p for p, _ in sinks)
    has_source = frozenset(p for p, _ in sources)
    excluded_to_profile = frozenset(source_to_profile_3)
    rows_source_toProcess: list[tuple[str, str, str]] = []
    for p, source in sources:
        if p in p_with_indirect:
            rows_source_toProcess.append((p, source, p))
        elif (p in p_with_direct
              and p not in has_sink
              and (p, source, p) not in excluded_to_profile):
            rows_source_toProcess.append((p, source, p))
    _write_csv(
        solve_data_dir / "process_source_toProcess.csv",
        ("process", "source", "process_aux"),
        list(dict.fromkeys(rows_source_toProcess)),
    )

    # ---- 5) process_process_toSink (symmetric)
    excluded_profile_to_sink = frozenset(profile_to_sink_3)
    rows_process_toSink: list[tuple[str, str, str]] = []
    for p, sink in sinks:
        if p in p_with_indirect:
            rows_process_toSink.append((p, p, sink))
        elif (p in p_with_direct
              and p not in has_source
              and (p, p, sink) not in excluded_profile_to_sink):
            rows_process_toSink.append((p, p, sink))
    _write_csv(
        solve_data_dir / "process_process_toSink.csv",
        ("process_outer", "process", "sink"),
        list(dict.fromkeys(rows_process_toSink)),
    )

    # ---- 6) process_source_sink_eff = source_toSink ∪ sink_toSource
    sst = _read_n_col(solve_data_dir / "process_source_toSink.csv", 3)
    sts = _read_n_col(solve_data_dir / "process_sink_toSource.csv", 3)
    union: dict[tuple[str, ...], None] = {}
    for r in sst:
        union.setdefault(r, None)
    for r in sts:
        union.setdefault(r, None)
    _write_csv(
        solve_data_dir / "process_source_sink_eff.csv",
        ("process", "source", "sink"),
        list(union.keys()),
    )

    # ---- 7) process_source_sink_noEff = 8-way union
    src_to_proc = rows_source_toProcess
    proc_to_snk = rows_process_toSink
    snk_to_proc = _read_n_col(solve_data_dir / "process_sink_toProcess.csv", 3)
    proc_to_src = _read_n_col(solve_data_dir / "process_process_toSource.csv", 3)
    proc_to_snk_noConv = _read_n_col(
        solve_data_dir / "process_process_toSink_noConversion.csv", 3
    )
    src_to_proc_noConv = _read_n_col(
        solve_data_dir / "process_source_toProcess_noConversion.csv", 3
    )
    union2: dict[tuple[str, ...], None] = {}
    for src in (src_to_proc, proc_to_snk, snk_to_proc, proc_to_src,
                profile_to_sink_3, source_to_profile_3,
                proc_to_snk_noConv, src_to_proc_noConv):
        for r in src:
            union2.setdefault(tuple(r), None)
    _write_csv(
        solve_data_dir / "process_source_sink_noEff.csv",
        ("process", "source", "sink"),
        list(union2.keys()),
    )

    # ---- 8) process_online = online_linear ∪ online_integer
    a = _read_singles(solve_data_dir / "process_online_linear.csv")
    b = _read_singles(solve_data_dir / "process_online_integer.csv")
    seen_o: dict[str, None] = {}
    for p in a + b:
        seen_o.setdefault(p, None)
    _write_csv(
        solve_data_dir / "process_online.csv",
        ("process",),
        [(p,) for p in seen_o.keys()],
    )

    # ---- 9) process_minload — filter on process__ct_method
    ctm = _read_pairs(solve_data_dir / "process__ct_method.csv")
    p_with_min_load = frozenset(p for p, m in ctm if m == "min_load_efficiency")
    processes = _read_singles(input_dir / "process.csv")
    minload = [p for p in processes if p in p_with_min_load]
    _write_csv(
        solve_data_dir / "process_minload.csv",
        ("process",),
        [(p,) for p in minload],
    )

    # ---- 10) process__commodity__node_co2
    cn_co2 = _read_pairs(solve_data_dir / "commodity_node_co2.csv")
    arc_endpoints_acc: dict[str, dict[str, None]] = {}
    for p, n in sources + sinks:
        arc_endpoints_acc.setdefault(p, {})[n] = None
    arc_endpoints: dict[str, frozenset[str]] = {
        p: frozenset(d.keys()) for p, d in arc_endpoints_acc.items()
    }
    rows_pcn_co2: list[tuple[str, str, str]] = []
    for p in processes:
        nodes_for_p = arc_endpoints.get(p, frozenset())
        for c, n in cn_co2:
            if n in nodes_for_p:
                rows_pcn_co2.append((p, c, n))
    _write_csv(
        solve_data_dir / "process__commodity__node_co2.csv",
        ("process", "commodity", "node"),
        list(dict.fromkeys(rows_pcn_co2)),
    )

    # ---- 11) process_co2 = setof p from process__commodity__node_co2
    seen_pco2: dict[str, None] = {}
    for p, _, _ in rows_pcn_co2:
        seen_pco2.setdefault(p, None)
    _write_csv(
        solve_data_dir / "process_co2.csv",
        ("process",),
        [(p,) for p in seen_pco2.keys()],
    )

    # ---- 12) process_source_sink (10-way union)
    pss_union: dict[tuple[str, ...], None] = {}
    for r in (sst + sts + src_to_proc + proc_to_snk
              + snk_to_proc + proc_to_src
              + profile_to_sink_3 + source_to_profile_3
              + proc_to_snk_noConv + src_to_proc_noConv):
        pss_union.setdefault(tuple(r), None)
    _write_csv(
        solve_data_dir / "process_source_sink.csv",
        ("process", "source", "sink"),
        list(pss_union.keys()),
    )

    # ---- 13) process_source_sink_alwaysProcess
    src_to_proc_d = _read_n_col(
        solve_data_dir / "process_source_toProcess_direct.csv", 3
    )
    proc_to_snk_d = _read_n_col(
        solve_data_dir / "process_process_toSink_direct.csv", 3
    )
    snk_to_proc_d = _read_n_col(
        solve_data_dir / "process_sink_toProcess_direct.csv", 3
    )
    proc_to_src_d = _read_n_col(
        solve_data_dir / "process_process_toSource_direct.csv", 3
    )
    pssa: dict[tuple[str, ...], None] = {}
    for r in (src_to_proc_d + proc_to_snk_d + snk_to_proc_d + proc_to_src_d
              + src_to_proc + proc_to_snk + snk_to_proc + proc_to_src
              + profile_to_sink_3 + source_to_profile_3
              + proc_to_snk_noConv + src_to_proc_noConv):
        pssa.setdefault(tuple(r), None)
    _write_csv(
        solve_data_dir / "process_source_sink_alwaysProcess.csv",
        ("process", "source", "sink"),
        list(pssa.keys()),
    )

    # ---- 14) process__source__sink__profile__profile_method_direct
    profiles = _read_n_col(
        input_dir / "process__node__profile__profile_method.csv", 4
    )
    p_n_to_fm: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for p, n, f, fm in profiles:
        p_n_to_fm.setdefault((p, n), []).append((f, fm))
    rows_direct: list[tuple[str, ...]] = []
    for p, source, sink in sst:  # process_source_toSink
        if p not in p_with_direct:
            continue
        seen_fm: dict[tuple[str, str], None] = {}
        for f, fm in p_n_to_fm.get((p, source), ()):
            seen_fm.setdefault((f, fm), None)
        for f, fm in p_n_to_fm.get((p, sink), ()):
            seen_fm.setdefault((f, fm), None)
        for f, fm in seen_fm:
            rows_direct.append((p, source, sink, f, fm))
    _write_csv(
        solve_data_dir
        / "process__source__sink__profile__profile_method_direct.csv",
        ("process", "source", "sink", "profile", "profile_method"),
        list(dict.fromkeys(rows_direct)),
    )


# ---------------------------------------------------------------------------
# write_entity_period_calc_params — top-level dispatcher own-compute.
# Mirrors flextool.flextoolrunner.preprocessing.entity_period_calc_params
# .write_entity_period_calc_params lines 67-202 of the legacy module.
# ---------------------------------------------------------------------------


# ---- Phase E-b — derive_X family for each emitted CSV --------------------
#
# Each derive_X is standalone (rebuilds its own lookups) so it can be
# called independently for accumulator capture.  The wrapper
# :func:`write_entity_period_calc_params` constructs the shared input
# bundle once via :func:`_entity_period_inputs` and feeds private
# ``_compute_*`` helpers to avoid recomputing the PdLookup scans across
# the five outputs.


def _entity_period_inputs(input_dir: Path, solve_data_dir: Path) -> dict:
    """Shared input bundle for the 5 entity-period derives."""
    from flextool.engine_polars._pdt_lookup import PdLookup

    pp = PdLookup(
        pd_csv=input_dir / "pd_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
    )
    pn = PdLookup(
        pd_csv=input_dir / "pd_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
    )

    return {
        "pp": pp,
        "pn": pn,
        "process_set": frozenset(_read_singles(input_dir / "process.csv")),
        "node_set": frozenset(_read_singles(input_dir / "node.csv")),
        "process_period_in_use": _read_pairs(
            solve_data_dir / "process__PeriodParam_in_use.csv"),
        "node_period_in_use": _read_pairs(
            solve_data_dir / "node__PeriodParam_in_use.csv"),
        "period_with_history": _read_singles(
            solve_data_dir / "period_with_history.csv"),
        "entities": _read_singles(input_dir / "entity.csv"),
    }


def _read_p_table(path: Path) -> dict[tuple[str, str], float]:
    """Read a 3-col (entity, paramName, value) table, silently skipping
    rows whose value isn't a float.  Mirrors the legacy local-loop in
    :func:`write_entity_period_calc_params`."""
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1]:
                try:
                    out[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
    return out


def _compute_pdProcess(inp: dict) -> pl.DataFrame:
    rows: list[tuple[str, str, str, str]] = []
    pp = inp["pp"]
    for (p, param) in inp["process_period_in_use"]:
        for d in inp["period_with_history"]:
            v = pp.get(p, param, d)
            rows.append((p, param, d, repr(v)))
    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "param":   [r[1] for r in rows],
            "period":  [r[2] for r in rows],
            "value":   [r[3] for r in rows],
        },
        schema={"process": pl.Utf8, "param": pl.Utf8,
                "period": pl.Utf8, "value": pl.Utf8},
    )


def _compute_pdNode(inp: dict) -> pl.DataFrame:
    rows: list[tuple[str, str, str, str]] = []
    pn = inp["pn"]
    for (n, param) in inp["node_period_in_use"]:
        for d in inp["period_with_history"]:
            v = pn.get(n, param, d)
            rows.append((n, param, d, repr(v)))
    return pl.DataFrame(
        {
            "node":   [r[0] for r in rows],
            "param":  [r[1] for r in rows],
            "period": [r[2] for r in rows],
            "value":  [r[3] for r in rows],
        },
        schema={"node": pl.Utf8, "param": pl.Utf8,
                "period": pl.Utf8, "value": pl.Utf8},
    )


def _compute_edEntity_lifetime(inp: dict) -> pl.DataFrame:
    pp = inp["pp"]
    pn = inp["pn"]
    process_set = inp["process_set"]
    node_set = inp["node_set"]
    rows: list[tuple[str, str, str]] = []
    for e in inp["entities"]:
        for d in inp["period_with_history"]:
            if e in process_set:
                v = pp.get(e, "lifetime", d)
            elif e in node_set:
                v = pn.get(e, "lifetime", d)
            else:
                v = 0.0
            rows.append((e, d, repr(v)))
    return pl.DataFrame(
        {"entity": [r[0] for r in rows],
         "period": [r[1] for r in rows],
         "value":  [r[2] for r in rows]},
        schema={"entity": pl.Utf8, "period": pl.Utf8, "value": pl.Utf8},
    )


def _compute_ed_fixed_cost(inp: dict) -> pl.DataFrame:
    pp = inp["pp"]
    pn = inp["pn"]
    process_set = inp["process_set"]
    node_set = inp["node_set"]
    rows: list[tuple[str, str, str]] = []
    for e in inp["entities"]:
        for d in inp["period_with_history"]:
            v = (1000.0 if e in node_set else 0.0) * pn.get(e, "fixed_cost", d) \
                + (1000.0 if e in process_set else 0.0) * pp.get(e, "fixed_cost", d)
            rows.append((e, d, repr(v)))
    return pl.DataFrame(
        {"entity": [r[0] for r in rows],
         "period": [r[1] for r in rows],
         "value":  [r[2] for r in rows]},
        schema={"entity": pl.Utf8, "period": pl.Utf8, "value": pl.Utf8},
    )


def _compute_p_entity_unitsize(input_dir: Path, inp: dict) -> pl.DataFrame:
    p_process = _read_p_table(input_dir / "p_process.csv")
    p_node = _read_p_table(input_dir / "p_node.csv")
    process_set = inp["process_set"]
    node_set = inp["node_set"]
    rows: list[tuple[str, str]] = []
    for e in inp["entities"]:
        if e in process_set:
            v = (p_process.get((e, "virtual_unitsize"), 0.0)
                 or p_process.get((e, "existing"), 0.0)
                 or 1000.0)
        elif e in node_set:
            v = (p_node.get((e, "virtual_unitsize"), 0.0)
                 or p_node.get((e, "existing"), 0.0)
                 or 1000.0)
        else:
            v = 0.0
        rows.append((e, repr(v)))
    return pl.DataFrame(
        {"entity": [r[0] for r in rows],
         "value":  [r[1] for r in rows]},
        schema={"entity": pl.Utf8, "value": pl.Utf8},
    )


# ---- Public derive_X (each rebuilds its own input bundle) ----

def derive_pdProcess(input_dir: Path, solve_data_dir: Path) -> pl.DataFrame:
    """``pdProcess.csv`` — (process, param, period, value) for every
    (process, param) in process__PeriodParam_in_use × period_with_history,
    value pulled from PdLookup over (pd_process, p_process, period__branch).
    """
    return _compute_pdProcess(_entity_period_inputs(input_dir, solve_data_dir))


def derive_pdNode(input_dir: Path, solve_data_dir: Path) -> pl.DataFrame:
    """``pdNode.csv`` — node-side analogue of pdProcess."""
    return _compute_pdNode(_entity_period_inputs(input_dir, solve_data_dir))


def derive_edEntity_lifetime(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``edEntity_lifetime.csv`` — per-entity lifetime per period_with_history.
    Process entities pull from the process PdLookup, node entities from the
    node PdLookup, others get 0.0.
    """
    return _compute_edEntity_lifetime(
        _entity_period_inputs(input_dir, solve_data_dir),
    )


def derive_ed_fixed_cost(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_fixed_cost.csv`` — entity fixed cost summed across the
    process and node side, each side scaled by 1000 if the entity is
    a member of that side."""
    return _compute_ed_fixed_cost(
        _entity_period_inputs(input_dir, solve_data_dir),
    )


def derive_p_entity_unitsize(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``p_entity_unitsize.csv`` — per-entity ``virtual_unitsize`` first,
    falling back to ``existing`` then to 1000.0; pulled from the
    side-appropriate p_*.csv table."""
    return _compute_p_entity_unitsize(
        input_dir,
        _entity_period_inputs(input_dir, solve_data_dir),
    )


def write_entity_period_calc_params(input_dir: Path,
                                    solve_data_dir: Path) -> None:
    """Migrate pdProcess/pdNode + edEntity_lifetime + ed_fixed_cost +
    p_entity_unitsize in one pass.

    Byte-for-byte mirror of the legacy emitter.  Each output flows
    through ``_write(derive_X(...), path)`` so Phase E-b's accumulator
    captures every frame.
    """
    inp = _entity_period_inputs(input_dir, solve_data_dir)
    _write(_compute_pdProcess(inp), solve_data_dir / "pdProcess.csv")
    _write(_compute_pdNode(inp), solve_data_dir / "pdNode.csv")
    _write(_compute_edEntity_lifetime(inp),
           solve_data_dir / "edEntity_lifetime.csv")
    _write(_compute_ed_fixed_cost(inp),
           solve_data_dir / "ed_fixed_cost.csv")
    _write(_compute_p_entity_unitsize(input_dir, inp),
           solve_data_dir / "p_entity_unitsize.csv")
