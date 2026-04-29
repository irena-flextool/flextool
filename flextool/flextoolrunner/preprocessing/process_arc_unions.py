"""Process-arc derivations for L1 — projections, joining sets, and unions
of already-Python-side arc tuple sets.

Migrated from flextool.mod (line numbers shift slightly across batches):

    L1007  process__profileProcess__toSink   (project (p, p2, sink) from
                                               the 5-tuple Python set)
    L1008  process__source__toProfileProcess (project (p, source, p2)
                                               from the 5-tuple Python set)
    L1009  process_profile                   (union of projections of
                                               the above two)
    L1010  process_source_toProcess          ((p, source, p) for p with
                                               method_indirect, OR with
                                               method_direct + no sinks
                                               + (p, source, p) ∉ process__source__toProfileProcess)
    L1018  process_process_toSink            (symmetric counterpart)
    L1042  process__source__sink__profile__profile_method_direct
                                              (5-tuple gated by method_direct)
    L1099  process_source_sink_eff           (process_source_toSink ∪
                                               process_sink_toSource)
    L1089  process_source_sink_noEff         (8-way union)
    L1120  process_online                    (process_online_linear ∪
                                               process_online_integer)
    L1743  process_minload                   ({p : (p, 'min_load_efficiency')
                                               in process__ct_method})
    L2010  process__commodity__node_co2      (process × commodity_node_co2
                                               filtered by arc membership)
    L2011  process_co2                       (project p from above)
"""
from __future__ import annotations

import csv
from pathlib import Path

from flextool.flextoolrunner.preprocessing._method_constants import (
    METHOD_INDIRECT, METHOD_DIRECT,
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


def write_process_arc_unions(input_dir: Path, solve_data_dir: Path) -> None:
    """Migrates the 16-set L1 batch in dependency order."""
    # ---- 1) process__profileProcess__toSink (project from 5-tuple Python set)
    five_tuple_to_sink = _read_n_col(
        solve_data_dir / "process__profileProcess__toSink__profile__profile_method.csv",
        5,
    )
    profile_to_sink_3 = list(dict.fromkeys(
        (p_outer, p, sink) for p_outer, p, sink, _f, _fm in five_tuple_to_sink
    ))
    _write_csv(solve_data_dir / "process__profileProcess__toSink.csv",
               ("process_outer", "process", "sink"),
               profile_to_sink_3)

    # ---- 2) process__source__toProfileProcess (project from 5-tuple Python set)
    five_tuple_to_source = _read_n_col(
        solve_data_dir / "process__source__toProfileProcess__profile__profile_method.csv",
        5,
    )
    source_to_profile_3 = list(dict.fromkeys(
        (p, source, p_aux) for p, source, p_aux, _f, _fm in five_tuple_to_source
    ))
    _write_csv(solve_data_dir / "process__source__toProfileProcess.csv",
               ("process", "source", "process_aux"),
               source_to_profile_3)

    # ---- 3) process_profile = setof p from (1) ∪ setof p from (2)
    seen_profile: dict[str, None] = {}
    for p, _, _ in source_to_profile_3:
        seen_profile.setdefault(p, None)
    for p, _, _ in profile_to_sink_3:
        seen_profile.setdefault(p, None)
    _write_csv(solve_data_dir / "process_profile.csv",
               ("process",),
               [(p,) for p in seen_profile.keys()])

    # ---- 4) process_source_toProcess
    # { (p, source) in process_source, p2 in process : p == p2
    #     AND (p2, source) in process_source [redundant since p==p2]
    #     AND ( has any indirect method
    #           OR ( has any direct method AND |sinks| < 1
    #                AND (p, source, p2) ∉ process__source__toProfileProcess ) )
    # }
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
    _write_csv(solve_data_dir / "process_source_toProcess.csv",
               ("process", "source", "process_aux"),
               list(dict.fromkeys(rows_source_toProcess)))

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
    _write_csv(solve_data_dir / "process_process_toSink.csv",
               ("process_outer", "process", "sink"),
               list(dict.fromkeys(rows_process_toSink)))

    # ---- 6) process_source_sink_eff = source_toSink ∪ sink_toSource
    sst = _read_n_col(solve_data_dir / "process_source_toSink.csv", 3)
    sts = _read_n_col(solve_data_dir / "process_sink_toSource.csv", 3)
    union: dict[tuple[str, ...], None] = {}
    for r in sst:
        union.setdefault(r, None)
    for r in sts:
        union.setdefault(r, None)
    _write_csv(solve_data_dir / "process_source_sink_eff.csv",
               ("process", "source", "sink"),
               list(union.keys()))

    # ---- 7) process_source_sink_noEff = 8-way union
    src_to_proc = rows_source_toProcess
    proc_to_snk = rows_process_toSink
    snk_to_proc = _read_n_col(solve_data_dir / "process_sink_toProcess.csv", 3)
    proc_to_src = _read_n_col(solve_data_dir / "process_process_toSource.csv", 3)
    proc_to_snk_noConv = _read_n_col(solve_data_dir / "process_process_toSink_noConversion.csv", 3)
    src_to_proc_noConv = _read_n_col(solve_data_dir / "process_source_toProcess_noConversion.csv", 3)
    union2: dict[tuple[str, ...], None] = {}
    for src in (src_to_proc, proc_to_snk, snk_to_proc, proc_to_src,
                profile_to_sink_3, source_to_profile_3,
                proc_to_snk_noConv, src_to_proc_noConv):
        for r in src:
            union2.setdefault(tuple(r), None)
    _write_csv(solve_data_dir / "process_source_sink_noEff.csv",
               ("process", "source", "sink"),
               list(union2.keys()))

    # ---- 8) process_online = online_linear ∪ online_integer (1-col sets)
    a = _read_singles(solve_data_dir / "process_online_linear.csv")
    b = _read_singles(solve_data_dir / "process_online_integer.csv")
    seen_o: dict[str, None] = {}
    for p in a + b:
        seen_o.setdefault(p, None)
    _write_csv(solve_data_dir / "process_online.csv",
               ("process",),
               [(p,) for p in seen_o.keys()])

    # ---- 9) process_minload — filter on process__ct_method
    ctm = _read_pairs(solve_data_dir / "process__ct_method.csv")
    p_with_min_load = frozenset(p for p, m in ctm if m == "min_load_efficiency")
    processes = _read_singles(input_dir / "process.csv")
    minload = [p for p in processes if p in p_with_min_load]
    _write_csv(solve_data_dir / "process_minload.csv",
               ("process",),
               [(p,) for p in minload])

    # ---- 10) process__commodity__node_co2
    # = {p in process, (c, n) in commodity_node_co2 : (p, n) in process_source ∨ process_sink}
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
    _write_csv(solve_data_dir / "process__commodity__node_co2.csv",
               ("process", "commodity", "node"),
               list(dict.fromkeys(rows_pcn_co2)))

    # ---- 11) process_co2 = setof p from process__commodity__node_co2
    seen_pco2: dict[str, None] = {}
    for p, _, _ in rows_pcn_co2:
        seen_pco2.setdefault(p, None)
    _write_csv(solve_data_dir / "process_co2.csv",
               ("process",),
               [(p,) for p in seen_pco2.keys()])

    # ---- 7b) process_source_sink (10-way union of arc tuple sets)
    # mod L1040: process_source_toSink ∪ process_sink_toSource ∪
    #   process_source_toProcess ∪ process_process_toSink ∪
    #   process_sink_toProcess ∪ process_process_toSource ∪
    #   process__profileProcess__toSink ∪ process__source__toProfileProcess ∪
    #   process_process_toSink_noConversion ∪ process_source_toProcess_noConversion
    pss_union: dict[tuple[str, ...], None] = {}
    for r in (sst + sts + src_to_proc + proc_to_snk
              + snk_to_proc + proc_to_src
              + profile_to_sink_3 + source_to_profile_3
              + proc_to_snk_noConv + src_to_proc_noConv):
        pss_union.setdefault(tuple(r), None)
    _write_csv(solve_data_dir / "process_source_sink.csv",
               ("process", "source", "sink"),
               list(pss_union.keys()))

    # ---- 7c) process_source_sink_alwaysProcess
    # mod L1052: 10-way union with process_*_direct + process_*_to_* +
    # noConversion + profile sets.
    src_to_proc_d = _read_n_col(solve_data_dir / "process_source_toProcess_direct.csv", 3)
    proc_to_snk_d = _read_n_col(solve_data_dir / "process_process_toSink_direct.csv", 3)
    snk_to_proc_d = _read_n_col(solve_data_dir / "process_sink_toProcess_direct.csv", 3)
    proc_to_src_d = _read_n_col(solve_data_dir / "process_process_toSource_direct.csv", 3)
    pssa: dict[tuple[str, ...], None] = {}
    for r in (src_to_proc_d + proc_to_snk_d + snk_to_proc_d + proc_to_src_d
              + src_to_proc + proc_to_snk + snk_to_proc + proc_to_src
              + profile_to_sink_3 + source_to_profile_3
              + proc_to_snk_noConv + src_to_proc_noConv):
        pssa.setdefault(tuple(r), None)
    _write_csv(solve_data_dir / "process_source_sink_alwaysProcess.csv",
               ("process", "source", "sink"),
               list(pssa.keys()))

    # ---- 12) process__source__sink__profile__profile_method_direct
    # { (p, source, sink) in process_source_toSink, f in profile,
    #   fm in profile_method
    #   : has any direct method
    #   AND ( (p, source, f, fm) in process__node__profile__profile_method
    #         OR (p, sink, f, fm) in process__node__profile__profile_method )
    # }
    profiles = _read_n_col(input_dir / "process__node__profile__profile_method.csv", 4)
    # Index by (p, n) → list of (f, fm)
    p_n_to_fm: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for p, n, f, fm in profiles:
        p_n_to_fm.setdefault((p, n), []).append((f, fm))
    rows_direct: list[tuple[str, ...]] = []
    for p, source, sink in sst:  # process_source_toSink
        if p not in p_with_direct:
            continue
        # Profile combinations that match either source-side or sink-side
        seen_fm: dict[tuple[str, str], None] = {}
        for f, fm in p_n_to_fm.get((p, source), ()):
            seen_fm.setdefault((f, fm), None)
        for f, fm in p_n_to_fm.get((p, sink), ()):
            seen_fm.setdefault((f, fm), None)
        for f, fm in seen_fm:
            rows_direct.append((p, source, sink, f, fm))
    _write_csv(
        solve_data_dir / "process__source__sink__profile__profile_method_direct.csv",
        ("process", "source", "sink", "profile", "profile_method"),
        list(dict.fromkeys(rows_direct)),
    )


def write_param_in_use_sets(input_dir: Path, solve_data_dir: Path) -> None:
    """node__PeriodParam_in_use, process__PeriodParam_in_use, process_TimeParam_in_use,
    plus the four sourceSink*_in_use partner sets.

    Each filters the universe of (entity, param) pairs by membership in
    the Required / Invest enum subsets in flextool_base.dat. We use
    Python constants from `_param_taxonomy.py` to mirror those
    invariants — no ``input/<ParamName>.csv`` exists in the runtime
    layout.
    """
    from flextool.flextoolrunner.preprocessing._param_taxonomy import (
        PROCESS_PERIOD_PARAM, PROCESS_PERIOD_PARAM_REQUIRED, PROCESS_PERIOD_PARAM_INVEST,
        PROCESS_TIME_PARAM, PROCESS_TIME_PARAM_REQUIRED,
        NODE_PERIOD_PARAM, NODE_PERIOD_PARAM_REQUIRED, NODE_PERIOD_PARAM_INVEST,
        SOURCE_SINK_TIME_PARAM, SOURCE_SINK_TIME_PARAM_REQUIRED,
        SOURCE_SINK_PERIOD_PARAM, SOURCE_SINK_PERIOD_PARAM_REQUIRED,
    )
    nodes = _read_singles(input_dir / "node.csv")
    processes = _read_singles(input_dir / "process.csv")
    invest_set = frozenset(_read_singles(solve_data_dir / "entityInvest.csv"))
    divest_set = frozenset(_read_singles(solve_data_dir / "entityDivest.csv"))
    ctm = _read_pairs(solve_data_dir / "process__ct_method.csv")
    p_with_min_load = frozenset(p for p, m in ctm if m == "min_load_efficiency")
    sources = _read_pairs(input_dir / "process__source.csv")
    sinks = _read_pairs(input_dir / "process__sink.csv")

    # node__PeriodParam_in_use — flextool.mod L1247
    rows: list[tuple[str, str]] = []
    for n in nodes:
        is_invest = n in invest_set or n in divest_set
        for param in NODE_PERIOD_PARAM:
            if param in NODE_PERIOD_PARAM_REQUIRED:
                rows.append((n, param))
            elif is_invest and param in NODE_PERIOD_PARAM_INVEST:
                rows.append((n, param))
    _write_csv(solve_data_dir / "node__PeriodParam_in_use.csv",
               ("node", "param"),
               list(dict.fromkeys(rows)))

    # process__PeriodParam_in_use — has a third clause for online processes:
    #   || (p in process_online && param == 'startup_cost')
    process_online = frozenset(_read_singles(solve_data_dir / "process_online.csv"))
    rows = []
    for p in processes:
        is_invest = p in invest_set or p in divest_set
        is_online = p in process_online
        for param in PROCESS_PERIOD_PARAM:
            if param in PROCESS_PERIOD_PARAM_REQUIRED:
                rows.append((p, param))
            elif is_invest and param in PROCESS_PERIOD_PARAM_INVEST:
                rows.append((p, param))
            elif is_online and param == "startup_cost":
                rows.append((p, param))
    _write_csv(solve_data_dir / "process__PeriodParam_in_use.csv",
               ("process", "param"),
               list(dict.fromkeys(rows)))

    # process_TimeParam_in_use
    rows = []
    for p in processes:
        for param in PROCESS_TIME_PARAM:
            if param in PROCESS_TIME_PARAM_REQUIRED:
                rows.append((p, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows.append((p, param))
    _write_csv(solve_data_dir / "process_TimeParam_in_use.csv",
               ("process", "param"),
               list(dict.fromkeys(rows)))

    # process_source_sourceSinkTimeParam_in_use — flextool.mod L1369
    # { (p, source) in process_source, param in sourceSinkTimeParam :
    #     param in sourceSinkTimeParamRequired
    #     OR ((p, 'min_load_efficiency') in process__ct_method AND param in {min_load, efficiency_at_min_load})
    # }
    rows3: list[tuple[str, str, str]] = []
    for p, source in sources:
        for param in SOURCE_SINK_TIME_PARAM:
            if param in SOURCE_SINK_TIME_PARAM_REQUIRED:
                rows3.append((p, source, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows3.append((p, source, param))
    _write_csv(solve_data_dir / "process_source_sourceSinkTimeParam_in_use.csv",
               ("process", "source", "param"),
               list(dict.fromkeys(rows3)))

    # process_sink_sourceSinkTimeParam_in_use
    rows3 = []
    for p, sink in sinks:
        for param in SOURCE_SINK_TIME_PARAM:
            if param in SOURCE_SINK_TIME_PARAM_REQUIRED:
                rows3.append((p, sink, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows3.append((p, sink, param))
    _write_csv(solve_data_dir / "process_sink_sourceSinkTimeParam_in_use.csv",
               ("process", "sink", "param"),
               list(dict.fromkeys(rows3)))

    # process_source_sourceSinkPeriodParam_in_use — same shape, period axis
    rows3 = []
    for p, source in sources:
        for param in SOURCE_SINK_PERIOD_PARAM:
            if param in SOURCE_SINK_PERIOD_PARAM_REQUIRED:
                rows3.append((p, source, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows3.append((p, source, param))
    _write_csv(solve_data_dir / "process_source_sourceSinkPeriodParam_in_use.csv",
               ("process", "source", "param"),
               list(dict.fromkeys(rows3)))

    # process_sink_sourceSinkPeriodParam_in_use
    rows3 = []
    for p, sink in sinks:
        for param in SOURCE_SINK_PERIOD_PARAM:
            if param in SOURCE_SINK_PERIOD_PARAM_REQUIRED:
                rows3.append((p, sink, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows3.append((p, sink, param))
    _write_csv(solve_data_dir / "process_sink_sourceSinkPeriodParam_in_use.csv",
               ("process", "sink", "param"),
               list(dict.fromkeys(rows3)))


def write_group_commodity_node_period_co2_total(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1981 — joining over group_node + commodity__node + co2.

    {g in group, (c, n) in commodity_node :
       (g, n) in group_node
       AND p_commodity[c, 'co2_content'] != 0
       AND g in group_co2_max_total}
    """
    cn = _read_pairs(input_dir / "commodity__node.csv")
    gn = _read_pairs(input_dir / "group__node.csv")
    g_with_n: dict[str, frozenset[str]] = {}
    _gn_acc: dict[str, dict[str, None]] = {}
    for g, n in gn:
        _gn_acc.setdefault(g, {})[n] = None
    g_with_n = {g: frozenset(d.keys()) for g, d in _gn_acc.items()}
    p_commodity: dict[tuple[str, str], float] = {}
    pc_path = input_dir / "p_commodity.csv"
    if pc_path.exists():
        with pc_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1]:
                    try:
                        p_commodity[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue
    co2_max_total = frozenset(
        _read_singles(solve_data_dir / "group_co2_max_total.csv")
    )
    rows: list[tuple[str, str, str]] = []
    for g in co2_max_total:
        nodes = g_with_n.get(g, frozenset())
        for c, n in cn:
            if n in nodes and p_commodity.get((c, "co2_content"), 0.0) != 0.0:
                rows.append((g, c, n))
    _write_csv(solve_data_dir / "group_commodity_node_period_co2_total.csv",
               ("group", "commodity", "node"),
               list(dict.fromkeys(rows)))


def write_process_source_delayed_partition(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1092-1093 — partition process_source by process_delayed.

        set process_source_undelayed := {(p, e) in process_source : p not in process_delayed};
        set process_source_delayed   := {(p, e) in process_source : p in process_delayed};
    """
    pairs = _read_pairs(input_dir / "process__source.csv")
    delayed = frozenset(_read_singles(solve_data_dir / "process_delayed.csv"))
    delayed_rows = [(p, src) for p, src in pairs if p in delayed]
    undelayed_rows = [(p, src) for p, src in pairs if p not in delayed]
    _write_csv(solve_data_dir / "process_source_delayed.csv",
               ("process", "source"), delayed_rows)
    _write_csv(solve_data_dir / "process_source_undelayed.csv",
               ("process", "source"), undelayed_rows)
