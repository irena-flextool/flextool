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

from flextool.engine_polars._writer_provider_io import (
    _provider_key,
    _provider_open,
)


# ---------------------------------------------------------------------------
# Polars-frame _write helper — patched by Phase E-b accumulator.
#
# This is the single emission funnel for the converted derive_* family in
# this module.  The patched variant in
# :mod:`._flex_data_accumulator.capture_frames` rebinds this name to also
# capture (path.name -> df) into the per-sub-solve accumulator.
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    # Phase E-c — gate disk emission behind ``emit_csvs_enabled``.
    from flextool.engine_polars._flex_data_accumulator import emit_csvs_enabled
    if not emit_csvs_enabled():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


# ---------------------------------------------------------------------------
# CSV I/O — same helpers as the sibling legacy modules.
# ---------------------------------------------------------------------------


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    out: list[tuple[str, str]] = []
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _read_n_col(path: Path, n: int,
                *, provider: "object | None" = None) -> list[tuple[str, ...]]:
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    out: list[tuple[str, ...]] = []
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= n and all(row[i] for i in range(n)):
                out.append(tuple(row[:n]))
    return out


def _read_singles(path: Path,
                  *, provider: "object | None" = None) -> list[str]:
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    with seeded as fh:
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
#
# Phase E-b lift: each of the 14 emitted CSVs flows through ``_write``
# (via a private ``_compute_*`` helper) so the accumulator monkey-patch
# captures every frame.  Public ``derive_*`` functions are exposed for
# standalone seed lookups; the wrapper builds the shared input bundle
# once and threads intermediate frames so dependent CSVs don't re-scan.
# ---------------------------------------------------------------------------


def _to_frame(rows, header: tuple[str, ...]) -> pl.DataFrame:
    """Dedup rows + materialise as an all-Utf8 polars frame.

    ``polars.write_csv`` on an all-Utf8 frame produces byte-identical
    output to the legacy ``_write_csv`` helper for plain ASCII data
    (the same shape the dispatcher operates on).
    """
    deduped = list(dict.fromkeys(tuple(r) for r in rows))
    cols = {h: [r[i] for r in deduped] for i, h in enumerate(header)}
    return pl.DataFrame(cols, schema={h: pl.Utf8 for h in header})


def _arc_unions_inputs(input_dir: Path, solve_data_dir: Path,
                        *, provider: "object | None" = None) -> dict:
    """Shared input bundle for the 14 derive_* in this monolith."""
    METHOD_INDIRECT = _METHOD_INDIRECT
    METHOD_DIRECT = _METHOD_DIRECT
    process_method = _read_pairs(input_dir / "process_method.csv",
                                  provider=provider)
    sources = _read_pairs(input_dir / "process__source.csv",
                          provider=provider)
    sinks = _read_pairs(input_dir / "process__sink.csv",
                        provider=provider)
    return {
        "process_method": process_method,
        "sources": sources,
        "sinks": sinks,
        "p_with_indirect": frozenset(
            p for p, m in process_method if m in METHOD_INDIRECT),
        "p_with_direct": frozenset(
            p for p, m in process_method if m in METHOD_DIRECT),
        "has_sink": frozenset(p for p, _ in sinks),
        "has_source": frozenset(p for p, _ in sources),
        "processes": _read_singles(input_dir / "process.csv",
                                    provider=provider),
        "five_tuple_to_sink": _read_n_col(
            solve_data_dir
            / "process__profileProcess__toSink__profile__profile_method.csv",
            5,
            provider=provider,
        ),
        "five_tuple_to_source": _read_n_col(
            solve_data_dir
            / "process__source__toProfileProcess__profile__profile_method.csv",
            5,
            provider=provider,
        ),
        "input_dir": input_dir,
        "solve_data_dir": solve_data_dir,
    }


# ---- (1) process__profileProcess__toSink ------------------------------------

def _profile_to_sink_3(inp: dict) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys(
        (p_outer, p, sink)
        for p_outer, p, sink, _f, _fm in inp["five_tuple_to_sink"]
    ))


def _compute_process__profileProcess__toSink(inp: dict) -> pl.DataFrame:
    return _to_frame(
        _profile_to_sink_3(inp), ("process_outer", "process", "sink"),
    )


def derive_process__profileProcess__toSink(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process__profileProcess__toSink.csv`` — 3-col projection from
    the 5-tuple ``process__profileProcess__toSink__profile__profile_method``."""
    return _compute_process__profileProcess__toSink(
        _arc_unions_inputs(input_dir, solve_data_dir),
    )


# ---- (2) process__source__toProfileProcess ----------------------------------

def _source_to_profile_3(inp: dict) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys(
        (p, source, p_aux)
        for p, source, p_aux, _f, _fm in inp["five_tuple_to_source"]
    ))


def _compute_process__source__toProfileProcess(inp: dict) -> pl.DataFrame:
    return _to_frame(
        _source_to_profile_3(inp), ("process", "source", "process_aux"),
    )


def derive_process__source__toProfileProcess(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process__source__toProfileProcess.csv`` — 3-col projection from
    the 5-tuple ``process__source__toProfileProcess__profile__profile_method``."""
    return _compute_process__source__toProfileProcess(
        _arc_unions_inputs(input_dir, solve_data_dir),
    )


# ---- (3) process_profile ----------------------------------------------------

def _compute_process_profile(inp: dict) -> pl.DataFrame:
    seen: dict[str, None] = {}
    for p, _, _ in _source_to_profile_3(inp):
        seen.setdefault(p, None)
    for p, _, _ in _profile_to_sink_3(inp):
        seen.setdefault(p, None)
    return _to_frame([(p,) for p in seen.keys()], ("process",))


def derive_process_profile(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_profile.csv`` — set of ``process`` appearing in either
    profile-source or profile-sink projections."""
    return _compute_process_profile(
        _arc_unions_inputs(input_dir, solve_data_dir),
    )


# ---- (4) process_source_toProcess + (5) process_process_toSink --------------

def _rows_source_toProcess(inp: dict) -> list[tuple[str, str, str]]:
    excluded_to_profile = frozenset(_source_to_profile_3(inp))
    rows: list[tuple[str, str, str]] = []
    for p, source in inp["sources"]:
        if p in inp["p_with_indirect"]:
            rows.append((p, source, p))
        elif (p in inp["p_with_direct"]
              and p not in inp["has_sink"]
              and (p, source, p) not in excluded_to_profile):
            rows.append((p, source, p))
    return list(dict.fromkeys(rows))


def _rows_process_toSink(inp: dict) -> list[tuple[str, str, str]]:
    excluded_profile_to_sink = frozenset(_profile_to_sink_3(inp))
    rows: list[tuple[str, str, str]] = []
    for p, sink in inp["sinks"]:
        if p in inp["p_with_indirect"]:
            rows.append((p, p, sink))
        elif (p in inp["p_with_direct"]
              and p not in inp["has_source"]
              and (p, p, sink) not in excluded_profile_to_sink):
            rows.append((p, p, sink))
    return list(dict.fromkeys(rows))


def _compute_process_source_toProcess(inp: dict) -> pl.DataFrame:
    return _to_frame(
        _rows_source_toProcess(inp), ("process", "source", "process_aux"),
    )


def _compute_process_process_toSink(inp: dict) -> pl.DataFrame:
    return _to_frame(
        _rows_process_toSink(inp), ("process_outer", "process", "sink"),
    )


def derive_process_source_toProcess(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_toProcess.csv`` — METHOD_INDIRECT or METHOD_DIRECT
    sources gated by exclusion from the profile projection."""
    return _compute_process_source_toProcess(
        _arc_unions_inputs(input_dir, solve_data_dir),
    )


def derive_process_process_toSink(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_process_toSink.csv`` — symmetric counterpart of
    process_source_toProcess for sinks."""
    return _compute_process_process_toSink(
        _arc_unions_inputs(input_dir, solve_data_dir),
    )


# ---- (6) process_source_sink_eff -------------------------------------------

def _compute_process_source_sink_eff(
    solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    sst = _read_n_col(solve_data_dir / "process_source_toSink.csv", 3,
                       provider=provider)
    sts = _read_n_col(solve_data_dir / "process_sink_toSource.csv", 3,
                       provider=provider)
    union: dict[tuple[str, ...], None] = {}
    for r in sst:
        union.setdefault(r, None)
    for r in sts:
        union.setdefault(r, None)
    return _to_frame(list(union.keys()), ("process", "source", "sink"))


def derive_process_source_sink_eff(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink_eff.csv`` — union of ``process_source_toSink``
    and ``process_sink_toSource``."""
    return _compute_process_source_sink_eff(solve_data_dir)


# ---- Helper: shared 6-list bundle for 7/12/13 -------------------------------

def _disk_arc_lists(solve_data_dir: Path,
                     *, provider: "object | None" = None) -> dict:
    return {
        "sst": _read_n_col(solve_data_dir / "process_source_toSink.csv", 3,
                            provider=provider),
        "sts": _read_n_col(solve_data_dir / "process_sink_toSource.csv", 3,
                            provider=provider),
        "snk_to_proc": _read_n_col(
            solve_data_dir / "process_sink_toProcess.csv", 3,
            provider=provider),
        "proc_to_src": _read_n_col(
            solve_data_dir / "process_process_toSource.csv", 3,
            provider=provider),
        "proc_to_snk_noConv": _read_n_col(
            solve_data_dir / "process_process_toSink_noConversion.csv", 3,
            provider=provider),
        "src_to_proc_noConv": _read_n_col(
            solve_data_dir / "process_source_toProcess_noConversion.csv", 3,
            provider=provider),
    }


# ---- (7) process_source_sink_noEff -----------------------------------------

def _compute_process_source_sink_noEff(
    inp: dict, disk: dict,
) -> pl.DataFrame:
    src_to_proc = _rows_source_toProcess(inp)
    proc_to_snk = _rows_process_toSink(inp)
    profile_to_sink_3 = _profile_to_sink_3(inp)
    source_to_profile_3 = _source_to_profile_3(inp)
    union2: dict[tuple[str, ...], None] = {}
    for src in (src_to_proc, proc_to_snk, disk["snk_to_proc"],
                disk["proc_to_src"], profile_to_sink_3, source_to_profile_3,
                disk["proc_to_snk_noConv"], disk["src_to_proc_noConv"]):
        for r in src:
            union2.setdefault(tuple(r), None)
    return _to_frame(list(union2.keys()), ("process", "source", "sink"))


def derive_process_source_sink_noEff(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink_noEff.csv`` — 8-way union over arc relations."""
    return _compute_process_source_sink_noEff(
        _arc_unions_inputs(input_dir, solve_data_dir),
        _disk_arc_lists(solve_data_dir),
    )


# ---- (8) process_online ----------------------------------------------------

def _compute_process_online(solve_data_dir: Path,
                             *, provider: "object | None" = None) -> pl.DataFrame:
    a = _read_singles(solve_data_dir / "process_online_linear.csv",
                       provider=provider)
    b = _read_singles(solve_data_dir / "process_online_integer.csv",
                       provider=provider)
    seen: dict[str, None] = {}
    for p in a + b:
        seen.setdefault(p, None)
    return _to_frame([(p,) for p in seen.keys()], ("process",))


def derive_process_online(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_online.csv`` — union of online_linear and online_integer."""
    return _compute_process_online(solve_data_dir)


# ---- (9) process_minload ---------------------------------------------------

def _compute_process_minload(inp: dict, solve_data_dir: Path,
                              *, provider: "object | None" = None) -> pl.DataFrame:
    ctm = _read_pairs(solve_data_dir / "process__ct_method.csv",
                       provider=provider)
    p_with_min_load = frozenset(
        p for p, m in ctm if m == "min_load_efficiency"
    )
    minload = [p for p in inp["processes"] if p in p_with_min_load]
    return _to_frame([(p,) for p in minload], ("process",))


def derive_process_minload(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_minload.csv`` — processes whose ct_method is
    ``min_load_efficiency``, ordered by ``process.csv`` ordering."""
    return _compute_process_minload(
        _arc_unions_inputs(input_dir, solve_data_dir), solve_data_dir,
    )


# ---- (10) process__commodity__node_co2 + (11) process_co2 ------------------

def _rows_pcn_co2(inp: dict, solve_data_dir: Path,
                   *, provider: "object | None" = None) -> list[tuple[str, str, str]]:
    cn_co2 = _read_pairs(solve_data_dir / "commodity_node_co2.csv",
                          provider=provider)
    arc_endpoints_acc: dict[str, dict[str, None]] = {}
    for p, n in inp["sources"] + inp["sinks"]:
        arc_endpoints_acc.setdefault(p, {})[n] = None
    arc_endpoints: dict[str, frozenset[str]] = {
        p: frozenset(d.keys()) for p, d in arc_endpoints_acc.items()
    }
    rows: list[tuple[str, str, str]] = []
    for p in inp["processes"]:
        nodes_for_p = arc_endpoints.get(p, frozenset())
        for c, n in cn_co2:
            if n in nodes_for_p:
                rows.append((p, c, n))
    return list(dict.fromkeys(rows))


def _compute_process__commodity__node_co2(
    inp: dict, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    return _to_frame(
        _rows_pcn_co2(inp, solve_data_dir, provider=provider),
        ("process", "commodity", "node"),
    )


def _compute_process_co2(inp: dict, solve_data_dir: Path,
                          *, provider: "object | None" = None) -> pl.DataFrame:
    seen: dict[str, None] = {}
    for p, _, _ in _rows_pcn_co2(inp, solve_data_dir, provider=provider):
        seen.setdefault(p, None)
    return _to_frame([(p,) for p in seen.keys()], ("process",))


def derive_process__commodity__node_co2(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process__commodity__node_co2.csv`` — process-commodity-node
    triples where the node is a CO2 commodity node and the process touches
    that node via either a source or sink."""
    return _compute_process__commodity__node_co2(
        _arc_unions_inputs(input_dir, solve_data_dir), solve_data_dir,
    )


def derive_process_co2(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_co2.csv`` — set of ``process`` from process__commodity__node_co2."""
    return _compute_process_co2(
        _arc_unions_inputs(input_dir, solve_data_dir), solve_data_dir,
    )


# ---- (12) process_source_sink ----------------------------------------------

def _compute_process_source_sink(
    inp: dict, disk: dict,
) -> pl.DataFrame:
    src_to_proc = _rows_source_toProcess(inp)
    proc_to_snk = _rows_process_toSink(inp)
    profile_to_sink_3 = _profile_to_sink_3(inp)
    source_to_profile_3 = _source_to_profile_3(inp)
    pss_union: dict[tuple[str, ...], None] = {}
    for r in (disk["sst"] + disk["sts"] + src_to_proc + proc_to_snk
              + disk["snk_to_proc"] + disk["proc_to_src"]
              + profile_to_sink_3 + source_to_profile_3
              + disk["proc_to_snk_noConv"] + disk["src_to_proc_noConv"]):
        pss_union.setdefault(tuple(r), None)
    return _to_frame(list(pss_union.keys()), ("process", "source", "sink"))


def derive_process_source_sink(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink.csv`` — 10-way union of every (process,
    source, sink) triple appearing across the eff/noEff/profile families."""
    return _compute_process_source_sink(
        _arc_unions_inputs(input_dir, solve_data_dir),
        _disk_arc_lists(solve_data_dir),
    )


# ---- (13) process_source_sink_alwaysProcess --------------------------------

def _compute_process_source_sink_alwaysProcess(
    inp: dict, disk: dict, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    src_to_proc = _rows_source_toProcess(inp)
    proc_to_snk = _rows_process_toSink(inp)
    profile_to_sink_3 = _profile_to_sink_3(inp)
    source_to_profile_3 = _source_to_profile_3(inp)
    src_to_proc_d = _read_n_col(
        solve_data_dir / "process_source_toProcess_direct.csv", 3,
        provider=provider,
    )
    proc_to_snk_d = _read_n_col(
        solve_data_dir / "process_process_toSink_direct.csv", 3,
        provider=provider,
    )
    snk_to_proc_d = _read_n_col(
        solve_data_dir / "process_sink_toProcess_direct.csv", 3,
        provider=provider,
    )
    proc_to_src_d = _read_n_col(
        solve_data_dir / "process_process_toSource_direct.csv", 3,
        provider=provider,
    )
    pssa: dict[tuple[str, ...], None] = {}
    for r in (src_to_proc_d + proc_to_snk_d + snk_to_proc_d + proc_to_src_d
              + src_to_proc + proc_to_snk
              + disk["snk_to_proc"] + disk["proc_to_src"]
              + profile_to_sink_3 + source_to_profile_3
              + disk["proc_to_snk_noConv"] + disk["src_to_proc_noConv"]):
        pssa.setdefault(tuple(r), None)
    return _to_frame(list(pssa.keys()), ("process", "source", "sink"))


def derive_process_source_sink_alwaysProcess(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink_alwaysProcess.csv`` — 12-way union including
    the ``_direct`` arc lists."""
    return _compute_process_source_sink_alwaysProcess(
        _arc_unions_inputs(input_dir, solve_data_dir),
        _disk_arc_lists(solve_data_dir),
        solve_data_dir,
    )


# ---- (14) process__source__sink__profile__profile_method_direct ------------

def _compute_process__source__sink__profile__profile_method_direct(
    inp: dict, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    sst = _read_n_col(solve_data_dir / "process_source_toSink.csv", 3,
                       provider=provider)
    profiles = _read_n_col(
        inp["input_dir"] / "process__node__profile__profile_method.csv", 4,
        provider=provider,
    )
    p_n_to_fm: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for p, n, f, fm in profiles:
        p_n_to_fm.setdefault((p, n), []).append((f, fm))
    rows: list[tuple[str, ...]] = []
    for p, source, sink in sst:
        if p not in inp["p_with_direct"]:
            continue
        seen_fm: dict[tuple[str, str], None] = {}
        for f, fm in p_n_to_fm.get((p, source), ()):
            seen_fm.setdefault((f, fm), None)
        for f, fm in p_n_to_fm.get((p, sink), ()):
            seen_fm.setdefault((f, fm), None)
        for f, fm in seen_fm:
            rows.append((p, source, sink, f, fm))
    return _to_frame(
        list(dict.fromkeys(rows)),
        ("process", "source", "sink", "profile", "profile_method"),
    )


def derive_process__source__sink__profile__profile_method_direct(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process__source__sink__profile__profile_method_direct.csv`` —
    cross-product of process_source_toSink × profile/profile_method gated
    by METHOD_DIRECT."""
    return _compute_process__source__sink__profile__profile_method_direct(
        _arc_unions_inputs(input_dir, solve_data_dir), solve_data_dir,
    )


# ---- Wrapper ---------------------------------------------------------------


def write_process_arc_unions(input_dir: Path, solve_data_dir: Path,
                              *, provider: "object | None" = None) -> None:
    """Migrate the 14-set L1 arc-union batch in dependency order.

    Byte-for-byte mirror of the legacy emitter.  Each output flows through
    ``_write(_compute_X(inp, ...), path)`` so the Phase E-b accumulator
    captures every emitted frame.

    Step 1-g — *provider* threads the per-sub-solve Provider through
    every ``_arc_unions_inputs`` / ``_disk_arc_lists`` / ``_compute_*``
    call so internal ``_read_*`` helpers resolve their frames in-memory
    before falling back to disk.
    """
    inp = _arc_unions_inputs(input_dir, solve_data_dir, provider=provider)

    # 1
    _write(_compute_process__profileProcess__toSink(inp),
           solve_data_dir / "process__profileProcess__toSink.csv")
    # 2
    _write(_compute_process__source__toProfileProcess(inp),
           solve_data_dir / "process__source__toProfileProcess.csv")
    # 3
    _write(_compute_process_profile(inp),
           solve_data_dir / "process_profile.csv")
    # 4
    _write(_compute_process_source_toProcess(inp),
           solve_data_dir / "process_source_toProcess.csv")
    # 5
    _write(_compute_process_process_toSink(inp),
           solve_data_dir / "process_process_toSink.csv")
    # 6
    _write(_compute_process_source_sink_eff(solve_data_dir, provider=provider),
           solve_data_dir / "process_source_sink_eff.csv")
    # 7
    disk = _disk_arc_lists(solve_data_dir, provider=provider)
    _write(_compute_process_source_sink_noEff(inp, disk),
           solve_data_dir / "process_source_sink_noEff.csv")
    # 8
    _write(_compute_process_online(solve_data_dir, provider=provider),
           solve_data_dir / "process_online.csv")
    # 9
    _write(_compute_process_minload(inp, solve_data_dir, provider=provider),
           solve_data_dir / "process_minload.csv")
    # 10
    _write(_compute_process__commodity__node_co2(inp, solve_data_dir,
                                                   provider=provider),
           solve_data_dir / "process__commodity__node_co2.csv")
    # 11
    _write(_compute_process_co2(inp, solve_data_dir, provider=provider),
           solve_data_dir / "process_co2.csv")
    # 12
    _write(_compute_process_source_sink(inp, disk),
           solve_data_dir / "process_source_sink.csv")
    # 13
    _write(_compute_process_source_sink_alwaysProcess(
               inp, disk, solve_data_dir, provider=provider),
           solve_data_dir / "process_source_sink_alwaysProcess.csv")
    # 14
    _write(
        _compute_process__source__sink__profile__profile_method_direct(
            inp, solve_data_dir, provider=provider),
        solve_data_dir
        / "process__source__sink__profile__profile_method_direct.csv",
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


def _entity_period_inputs(input_dir: Path, solve_data_dir: Path,
                            *, provider: "object | None" = None) -> dict:
    """Shared input bundle for the 5 entity-period derives."""
    from flextool.engine_polars._pdt_lookup import PdLookup

    pp = PdLookup(
        pd_csv=input_dir / "pd_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )
    pn = PdLookup(
        pd_csv=input_dir / "pd_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )

    return {
        "pp": pp,
        "pn": pn,
        "process_set": frozenset(_read_singles(input_dir / "process.csv", provider=provider)),
        "node_set": frozenset(_read_singles(input_dir / "node.csv", provider=provider)),
        "process_period_in_use": _read_pairs(
            solve_data_dir / "process__PeriodParam_in_use.csv", provider=provider),
        "node_period_in_use": _read_pairs(
            solve_data_dir / "node__PeriodParam_in_use.csv", provider=provider),
        "period_with_history": _read_singles(
            solve_data_dir / "period_with_history.csv", provider=provider),
        "entities": _read_singles(input_dir / "entity.csv", provider=provider),
    }


def _read_p_table(path: Path,
                  *, provider: "object | None" = None,
                  ) -> dict[tuple[str, str], float]:
    """Read a 3-col (entity, paramName, value) table, silently skipping
    rows whose value isn't a float.  Mirrors the legacy local-loop in
    :func:`write_entity_period_calc_params`."""
    out: dict[tuple[str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
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

def derive_pdProcess(input_dir: Path, solve_data_dir: Path,
                       *, provider: "object | None" = None) -> pl.DataFrame:
    """``pdProcess.csv`` — (process, param, period, value) for every
    (process, param) in process__PeriodParam_in_use × period_with_history,
    value pulled from PdLookup over (pd_process, p_process, period__branch).
    """
    return _compute_pdProcess(
        _entity_period_inputs(input_dir, solve_data_dir, provider=provider))


def derive_pdNode(input_dir: Path, solve_data_dir: Path,
                   *, provider: "object | None" = None) -> pl.DataFrame:
    """``pdNode.csv`` — node-side analogue of pdProcess."""
    return _compute_pdNode(
        _entity_period_inputs(input_dir, solve_data_dir, provider=provider))


def derive_edEntity_lifetime(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """``edEntity_lifetime.csv`` — per-entity lifetime per period_with_history.
    Process entities pull from the process PdLookup, node entities from the
    node PdLookup, others get 0.0.
    """
    return _compute_edEntity_lifetime(
        _entity_period_inputs(input_dir, solve_data_dir, provider=provider),
    )


def derive_ed_fixed_cost(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """``ed_fixed_cost.csv`` — entity fixed cost summed across the
    process and node side, each side scaled by 1000 if the entity is
    a member of that side."""
    return _compute_ed_fixed_cost(
        _entity_period_inputs(input_dir, solve_data_dir, provider=provider),
    )


def derive_p_entity_unitsize(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    """``p_entity_unitsize.csv`` — per-entity ``virtual_unitsize`` first,
    falling back to ``existing`` then to 1000.0; pulled from the
    side-appropriate p_*.csv table."""
    return _compute_p_entity_unitsize(
        input_dir,
        _entity_period_inputs(input_dir, solve_data_dir, provider=provider),
    )


def write_entity_period_calc_params(input_dir: Path,
                                    solve_data_dir: Path,
                                    *, provider: "object | None" = None) -> None:
    """Migrate pdProcess/pdNode + edEntity_lifetime + ed_fixed_cost +
    p_entity_unitsize in one pass.

    Byte-for-byte mirror of the legacy emitter.  Each output flows
    through ``_write(derive_X(...), path)`` so Phase E-b's accumulator
    captures every frame.
    """
    inp = _entity_period_inputs(input_dir, solve_data_dir, provider=provider)
    _write(_compute_pdProcess(inp), solve_data_dir / "pdProcess.csv")
    _write(_compute_pdNode(inp), solve_data_dir / "pdNode.csv")
    _write(_compute_edEntity_lifetime(inp),
           solve_data_dir / "edEntity_lifetime.csv")
    _write(_compute_ed_fixed_cost(inp),
           solve_data_dir / "ed_fixed_cost.csv")
    _write(_compute_p_entity_unitsize(input_dir, inp),
           solve_data_dir / "p_entity_unitsize.csv")
