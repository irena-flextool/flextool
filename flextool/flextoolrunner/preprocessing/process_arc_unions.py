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


def write_process_source_sink_param_t(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1197 — process_source_sink_eff × processTimeParam
    filtered by (p, param) ∈ process__param_t.

        set process_source_sink_param_t :=
            {(p, source, sink) in process_source_sink_eff, param in processTimeParam :
             (p, param) in process__param_t};

    process__param_t is the projection of process__param__time, which is
    loaded from solve_data/pt_process.csv. processTimeParam is a constant
    enum from flextool_base.dat:153 (PROCESS_TIME_PARAM in our taxonomy).
    """
    from flextool.flextoolrunner.preprocessing._param_taxonomy import (
        PROCESS_TIME_PARAM,
    )
    pss_eff = _read_n_col(solve_data_dir / "process_source_sink_eff.csv", 3)
    process_param_t: set[tuple[str, str]] = set()
    pt_path = solve_data_dir / "pt_process.csv"
    if pt_path.exists():
        with pt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    process_param_t.add((row[0], row[1]))
    rows: list[tuple[str, str, str, str]] = []
    for p, source, sink in pss_eff:
        for param in PROCESS_TIME_PARAM:
            if (p, param) in process_param_t:
                rows.append((p, source, sink, param))
    _write_csv(solve_data_dir / "process_source_sink_param_t.csv",
               ("process", "source", "sink", "param"),
               list(dict.fromkeys(rows)))


def write_node_time_param_in_use(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1208-1214 — node × nodeTimeParam filtered.

        set node__TimeParam_in_use :=
          { n in node, param in nodeTimeParam:
            (n in nodeBalance && param in nodeTimeParamRequired)
            || (n in nodeBalancePeriod && param in nodeTimeParamRequired)
            || (n in nodeState && param in {self_discharge_loss, availability})
            || ((n, 'use_reference_value') in node__storage_solve_horizon_method
                 && param == 'storage_state_reference_value')
          };
    """
    from flextool.flextoolrunner.preprocessing._param_taxonomy import (
        NODE_TIME_PARAM, NODE_TIME_PARAM_REQUIRED,
    )
    nodes = _read_singles(input_dir / "node.csv")
    n_balance = frozenset(_read_singles(solve_data_dir / "nodeBalance.csv"))
    n_balance_period = frozenset(_read_singles(solve_data_dir / "nodeBalancePeriod.csv"))
    n_state = frozenset(_read_singles(solve_data_dir / "nodeState.csv"))
    n_storage_use_ref = frozenset(
        n for n, m in _read_pairs(input_dir / "node__storage_solve_horizon_method.csv")
        if m == "use_reference_value"
    )
    rows: list[tuple[str, str]] = []
    for n in nodes:
        is_bal = n in n_balance
        is_bal_period = n in n_balance_period
        is_state = n in n_state
        is_use_ref = n in n_storage_use_ref
        for param in NODE_TIME_PARAM:
            if (is_bal or is_bal_period) and param in NODE_TIME_PARAM_REQUIRED:
                rows.append((n, param))
            elif is_state and param in ("self_discharge_loss", "availability"):
                rows.append((n, param))
            elif is_use_ref and param == "storage_state_reference_value":
                rows.append((n, param))
    _write_csv(solve_data_dir / "node__TimeParam_in_use.csv",
               ("node", "param"),
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


def write_process_source_sink_param(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1185-1189 — process_source_sink × sourceSinkParam
    filtered by parameter membership on either side or via process_connection.

        { (p, src, sink) in process_source_sink, param in SOURCE_SINK_PARAM :
          (p, src,  param) in process__source__param
          OR (p, sink, param) in process__sink__param
          OR ((p, param) in process__param AND p in process_connection) }

    process__source__param ← input/p_process_source.csv columns
        [process, source, sourceSinkParam]
    process__sink__param   ← input/p_process_sink.csv   columns [process, sink, …]
    process__param         ← input/p_process.csv columns [process, processParam]
    process_connection     ← input/process_connection.csv [process_connection]
    """
    from flextool.flextoolrunner.preprocessing._param_taxonomy import (
        SOURCE_SINK_PARAM,
    )
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)

    src_param: set[tuple[str, str, str]] = set()
    pps_path = input_dir / "p_process_source.csv"
    if pps_path.exists():
        with pps_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    src_param.add((r[0], r[1], r[2]))
    sink_param: set[tuple[str, str, str]] = set()
    ppk_path = input_dir / "p_process_sink.csv"
    if ppk_path.exists():
        with ppk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    sink_param.add((r[0], r[1], r[2]))
    proc_param: set[tuple[str, str]] = set()
    pp_path = input_dir / "p_process.csv"
    if pp_path.exists():
        with pp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] and r[1]:
                    proc_param.add((r[0], r[1]))
    proc_conn = frozenset(_read_singles(input_dir / "process_connection.csv"))

    rows: list[tuple[str, str, str, str]] = []
    for p, src, sink in triples:
        for param in SOURCE_SINK_PARAM:
            if ((p, src, param) in src_param
                    or (p, sink, param) in sink_param
                    or ((p, param) in proc_param and p in proc_conn)):
                rows.append((p, src, sink, param))
    _write_csv(solve_data_dir / "process__source__sink__param.csv",
               ("process", "source", "sink", "param"), rows)


def write_process_source_sink_profile_method_connection(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1060-1063 — process_source_sink × profile ×
    profile_method, filtered by (p, profile, method) membership in
    process__profile__profile_method.

        { (p, src, sink) in process_source_sink, f in profile, m in profile_method :
          (p, f, m) in process__profile__profile_method }

    Mod's iteration uses misleading var names `(p, sink, source)` —
    MathProg binds positionally, so the OUTPUT tuple is still
    [process, source, sink, profile, profile_method] matching the
    direct counterpart that's unioned with this set at L1064-1068.

    process__profile__profile_method ← input/process__profile__profile_method.csv
        (3-col loader at mod L697).
    """
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    pp_pm = _read_n_col(input_dir / "process__profile__profile_method.csv", 3)
    # Group profile_methods by process for O(N + M) join.
    fm_for_p: dict[str, list[tuple[str, str]]] = {}
    for p, f, m in pp_pm:
        fm_for_p.setdefault(p, []).append((f, m))
    rows: list[tuple[str, str, str, str, str]] = []
    for p, src, sink in triples:
        for f, m in fm_for_p.get(p, ()):
            rows.append((p, src, sink, f, m))
    _write_csv(
        solve_data_dir / "process__source__sink__profile__profile_method_connection.csv",
        ("process", "source", "sink", "profile", "profile_method"),
        rows,
    )


def write_process_source_sink_param_with_time(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1187-1195 — process_source_sink × sourceSinkTimeParam
    filtered by EITHER static or time-variant param membership on either
    side, or via process_connection.

    Distinct from `write_process_source_sink_param_t` (batch 25) which
    handles a different mod set: `process_source_sink_param_t`
    (single-underscore name) gates by the simpler condition
    `(p, param) in process__param_t`. This function handles
    `process__source__sink__param_t` (double-underscore name).

        { (p, src, sink) in process_source_sink, param in SOURCE_SINK_TIME_PARAM :
          (p, src,  param) in process__source__param
          OR (p, src,  param) in process__source__param_t
          OR (p, sink, param) in process__sink__param
          OR (p, sink, param) in process__sink__param_t
          OR ((p, param) in process__param   AND p in process_connection)
          OR ((p, param) in process__param_t AND p in process_connection) }

    The _t projections come from solve_data/pt_process*.csv (already
    written by solve_writers.py before this preprocessing hook).
    """
    from flextool.flextoolrunner.preprocessing._param_taxonomy import (
        SOURCE_SINK_TIME_PARAM,
    )
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)

    src_param: set[tuple[str, str, str]] = set()
    pps_path = input_dir / "p_process_source.csv"
    if pps_path.exists():
        with pps_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    src_param.add((r[0], r[1], r[2]))
    sink_param: set[tuple[str, str, str]] = set()
    ppk_path = input_dir / "p_process_sink.csv"
    if ppk_path.exists():
        with ppk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    sink_param.add((r[0], r[1], r[2]))
    proc_param: set[tuple[str, str]] = set()
    pp_path = input_dir / "p_process.csv"
    if pp_path.exists():
        with pp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] and r[1]:
                    proc_param.add((r[0], r[1]))
    proc_conn = frozenset(_read_singles(input_dir / "process_connection.csv"))

    # _t variants from solve_data/pt_process*.csv (process__source__param_t
    # = setof {(p, src, param, t)} (p, src, param), etc.)
    src_param_t: set[tuple[str, str, str]] = set()
    pts_path = solve_data_dir / "pt_process_source.csv"
    if pts_path.exists():
        with pts_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    src_param_t.add((r[0], r[1], r[2]))
    sink_param_t: set[tuple[str, str, str]] = set()
    ptk_path = solve_data_dir / "pt_process_sink.csv"
    if ptk_path.exists():
        with ptk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    sink_param_t.add((r[0], r[1], r[2]))
    proc_param_t: set[tuple[str, str]] = set()
    pt_path = solve_data_dir / "pt_process.csv"
    if pt_path.exists():
        with pt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] and r[1]:
                    proc_param_t.add((r[0], r[1]))

    rows: list[tuple[str, str, str, str]] = []
    for p, src, sink in triples:
        for param in SOURCE_SINK_TIME_PARAM:
            if ((p, src, param) in src_param
                    or (p, src, param) in src_param_t
                    or (p, sink, param) in sink_param
                    or (p, sink, param) in sink_param_t
                    or ((p, param) in proc_param and p in proc_conn)
                    or ((p, param) in proc_param_t and p in proc_conn)):
                rows.append((p, src, sink, param))
    _write_csv(solve_data_dir / "process__source__sink__param_t.csv",
               ("process", "source", "sink", "param"), rows)


def write_process_method_sources_sinks(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1046-1053 — 3-way join across process_source_sink_alwaysProcess,
    process_source_sink, and process_method, filtered by per-side source/sink
    aliasing (alias to the orig leg or to p itself, but not both = p).

        setof { (p, always_src, always_snk) in process_source_sink_alwaysProcess,
                (p, orig_src,   orig_snk)   in process_source_sink,
                (p, m)                       in process_method
              : (always_src == orig_src OR always_src == p)
                AND (always_snk == orig_snk OR always_snk == p)
                AND NOT (always_src == p AND always_snk == p) }
            (p, m, orig_src, orig_snk, always_src, always_snk)

    Output column order matches mod's setof projection.
    """
    always = _read_n_col(solve_data_dir / "process_source_sink_alwaysProcess.csv", 3)
    pss = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    pm = _read_pairs(input_dir / "process_method.csv")

    # Group rows of each input by process for O(N + M + K) join.
    always_for_p: dict[str, list[tuple[str, str]]] = {}
    for p, asrc, asnk in always:
        always_for_p.setdefault(p, []).append((asrc, asnk))
    pss_for_p: dict[str, list[tuple[str, str]]] = {}
    for p, osrc, osnk in pss:
        pss_for_p.setdefault(p, []).append((osrc, osnk))
    methods_for_p: dict[str, list[str]] = {}
    for p, m in pm:
        methods_for_p.setdefault(p, []).append(m)

    seen: dict[tuple[str, str, str, str, str, str], None] = {}
    for p, alist in always_for_p.items():
        olist = pss_for_p.get(p, ())
        mlist = methods_for_p.get(p, ())
        if not olist or not mlist:
            continue
        for asrc, asnk in alist:
            if asrc == p and asnk == p:
                continue
            for osrc, osnk in olist:
                if not (asrc == osrc or asrc == p):
                    continue
                if not (asnk == osnk or asnk == p):
                    continue
                for m in mlist:
                    seen.setdefault((p, m, osrc, osnk, asrc, asnk), None)
    _write_csv(
        solve_data_dir / "process_method_sources_sinks.csv",
        ("process", "method", "orig_source", "orig_sink",
         "always_source", "always_sink"),
        list(seen.keys()),
    )


def write_ed_history_realized_first(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L993 — entity × realized periods, first solve only.

        { (e, d) in entity × (d_realize_invest ∪ d_fix_storage_period
                              ∪ d_realized_period)
          : (d, d) in period__branch
            AND p_model['solveFirst'] }

    On non-first solves the set is empty (the AND on solveFirst short-
    circuits). Reads p_model.csv (input/) for the solveFirst flag.
    """
    # Honour solveFirst: empty result on non-first solves.
    solve_first = False
    pm_path = input_dir / "p_model.csv"
    if pm_path.exists():
        with pm_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] == "solveFirst":
                    try:
                        solve_first = bool(int(r[1]))
                    except ValueError:
                        pass
                    break
    if not solve_first:
        _write_csv(solve_data_dir / "ed_history_realized_first.csv",
                   ("entity", "period"), [])
        return

    entities = _read_singles(input_dir / "entity.csv")
    d_realize_invest = frozenset(_read_singles(
        solve_data_dir / "realized_invest_periods_of_current_solve.csv"
    ))
    d_fix_storage = frozenset(_read_singles(
        solve_data_dir / "d_fix_storage_period_set.csv"
    ))
    d_realized = frozenset(_read_singles(
        solve_data_dir / "d_realized_period_set.csv"
    ))
    realized_periods = d_realize_invest | d_fix_storage | d_realized

    pb = _read_pairs(solve_data_dir / "period__branch.csv")
    same_branch = frozenset((d, b) for d, b in pb if d == b)
    diag_periods = frozenset(d for d, _ in same_branch)

    rows = [
        (e, d) for e in entities
        for d in realized_periods if d in diag_periods
    ]
    _write_csv(solve_data_dir / "ed_history_realized_first.csv",
               ("entity", "period"), rows)


def write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1152-1155 — process_source_sink filtered for 1way
    processes whose source is a node, with no sink OR multiple sources.

        { (p, src, sink) in process_source_sink :
          p has any method in METHOD_1WAY
          AND (p, src) in process_source
          AND (no sinks for p  OR  >= 2 sources for p) }

    process_method ← input/process_method.csv (canonical; mod L1003).
    process_source / process_sink ← input/process__source.csv /
        input/process__sink.csv (canonical; mod L686-687).
    """
    from flextool.flextoolrunner.preprocessing._method_constants import (
        METHOD_1WAY,
    )
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    pm = _read_pairs(input_dir / "process_method.csv")
    methods_of_p: dict[str, set[str]] = {}
    for p, m in pm:
        methods_of_p.setdefault(p, set()).add(m)
    has_1way = {
        p: bool(ms & METHOD_1WAY) for p, ms in methods_of_p.items()
    }
    proc_source_pairs = frozenset(_read_pairs(input_dir / "process__source.csv"))
    sources_of_p: dict[str, int] = {}
    for p, _ in proc_source_pairs:
        sources_of_p[p] = sources_of_p.get(p, 0) + 1
    sinks_of_p: dict[str, int] = {}
    for p, _ in _read_pairs(input_dir / "process__sink.csv"):
        sinks_of_p[p] = sinks_of_p.get(p, 0) + 1

    rows: list[tuple[str, str, str]] = []
    for p, src, sink in triples:
        if not has_1way.get(p, False):
            continue
        if (p, src) not in proc_source_pairs:
            continue
        # no sinks OR >= 2 sources
        if sinks_of_p.get(p, 0) == 0 or sources_of_p.get(p, 0) >= 2:
            rows.append((p, src, sink))
    _write_csv(
        solve_data_dir / "process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.csv",
        ("process", "source", "sink"),
        rows,
    )


def write_node_group_dispatch_process_fully_inside(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1789-1794 — for each (g, p) where g is a dispatch
    nodeGroup and p is a process, include if BOTH source and sink nodes
    of p are members of g, AND p isn't a self-loop (e.g. a battery
    storage with source==sink).

        { g in nodeGroupDispatch, p in process :
          sum {(p, src) in process_source : (g, src) in group_node} 1
          AND sum {(p, snk) in process_sink : (g, snk) in group_node} 1
          AND NOT sum {(p, src, snk) in process_source_sink : src == snk} 1 }
    """
    ngd = _read_singles(input_dir / "nodeGroupDispatch.csv")
    procs = _read_singles(input_dir / "process.csv")
    process_source_pairs = _read_pairs(input_dir / "process__source.csv")
    process_sink_pairs = _read_pairs(input_dir / "process__sink.csv")
    gn = _read_pairs(input_dir / "group__node.csv")
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)

    nodes_in_g: dict[str, set[str]] = {}
    for g, n in gn:
        nodes_in_g.setdefault(g, set()).add(n)
    sources_of_p: dict[str, set[str]] = {}
    for p, src in process_source_pairs:
        sources_of_p.setdefault(p, set()).add(src)
    sinks_of_p: dict[str, set[str]] = {}
    for p, snk in process_sink_pairs:
        sinks_of_p.setdefault(p, set()).add(snk)
    self_loop_processes = frozenset(
        p for p, src, snk in triples if src == snk
    )

    rows: list[tuple[str, str]] = []
    for g in ngd:
        gnodes = nodes_in_g.get(g, set())
        if not gnodes:
            continue
        for p in procs:
            if p in self_loop_processes:
                continue
            srcs = sources_of_p.get(p, set())
            snks = sinks_of_p.get(p, set())
            if (srcs & gnodes) and (snks & gnodes):
                rows.append((g, p))
    _write_csv(solve_data_dir / "nodeGroupDispatch__process_fully_inside.csv",
               ("group", "process"), rows)


def write_process_source_sink_ramp_method(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1205-1209 — process_source_sink × ramp_method
    filtered by per-side ramp_method membership.

        { (p, src, sink) in process_source_sink, m in ramp_method :
          (p, src, m)  in process_node_ramp_method
          OR (p, sink, m) in process_node_ramp_method }
    """
    from flextool.flextoolrunner.preprocessing._method_constants import (
        RAMP_METHOD,
    )
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    pnrm: set[tuple[str, str, str]] = set()
    pnrm_path = input_dir / "process__node__ramp_method.csv"
    if pnrm_path.exists():
        with pnrm_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    pnrm.add((r[0], r[1], r[2]))
    rows: list[tuple[str, str, str, str]] = []
    for p, src, sink in triples:
        for m in RAMP_METHOD:
            if (p, src, m) in pnrm or (p, sink, m) in pnrm:
                rows.append((p, src, sink, m))
    _write_csv(solve_data_dir / "process__source__sink__ramp_method.csv",
               ("process", "source", "sink", "ramp_method"), rows)


def write_process_source_sink_coeff_zero(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1973 — process_source_sink filtered by zero
    flow_coefficient on either side.

        { (p, src, sink) in process_source_sink :
          (p, src) in process_source_coeff_zero
          OR (p, sink) in process_sink_coeff_zero }

    Used downstream to skip flow-coefficient multiplications when the
    coefficient is zero.
    """
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    src_zero = frozenset(_read_pairs(solve_data_dir / "process_source_coeff_zero.csv"))
    sink_zero = frozenset(_read_pairs(solve_data_dir / "process_sink_coeff_zero.csv"))
    rows = [
        (p, src, sink) for p, src, sink in triples
        if (p, src) in src_zero or (p, sink) in sink_zero
    ]
    _write_csv(solve_data_dir / "process_source_sink_coeff_zero.csv",
               ("process", "source", "sink"), rows)


def write_process_source_sink_ramp_family(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1660-1688 — process_source_sink filtered by per-arc
    ramp_method membership and (for the limit variants) a positive
    ramp_speed parameter.

        process_source_sink_ramp_limit_source_up   — (p,src,sink) where
            (p,src,m) in process_node_ramp_method with m in RAMP_LIMIT_METHOD
            AND p_process_source[p,src,'ramp_speed_up'] > 0
        process_source_sink_ramp_limit_sink_up     — symmetric on sink
        process_source_sink_ramp_limit_source_down — symmetric on speed_down
        process_source_sink_ramp_limit_sink_down   — symmetric
        process_source_sink_ramp_cost              — OR of source-side
            and sink-side membership in RAMP_COST_METHOD (no speed gate)

    Reads input/process__node__ramp_method.csv (mod loader L684) and
    input/p_process_source.csv / p_process_sink.csv (canonical sources;
    solve_data variants only exist after first solve via mod printf).
    """
    from flextool.flextoolrunner.preprocessing._method_constants import (
        RAMP_LIMIT_METHOD, RAMP_COST_METHOD,
    )
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)

    # process_node_ramp_method (process, node, ramp_method)
    pnrm: dict[tuple[str, str], set[str]] = {}
    pnrm_path = input_dir / "process__node__ramp_method.csv"
    if pnrm_path.exists():
        with pnrm_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    pnrm.setdefault((r[0], r[1]), set()).add(r[2])

    def _has_ramp_method(p: str, n: str, methods: frozenset[str]) -> bool:
        return bool(pnrm.get((p, n), set()) & methods)

    # p_process_source / p_process_sink: (process, side, param, value)
    p_proc_side: dict[tuple[str, str, str, str], float] = {}
    for filename, side_label in (
        ("p_process_source.csv", "source"),
        ("p_process_sink.csv",   "sink"),
    ):
        path = input_dir / filename
        if not path.exists():
            continue
        with path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] and r[2]:
                    try:
                        p_proc_side[(side_label, r[0], r[1], r[2])] = float(r[3])
                    except ValueError:
                        continue

    def _ramp_speed(side: str, p: str, n: str, dir_: str) -> float:
        return p_proc_side.get((side, p, n, f"ramp_speed_{dir_}"), 0.0)

    rsu_rows = [(p, src, sink) for p, src, sink in triples
                if _has_ramp_method(p, src, RAMP_LIMIT_METHOD)
                and _ramp_speed("source", p, src, "up") > 0]
    _write_csv(solve_data_dir / "process_source_sink_ramp_limit_source_up.csv",
               ("process", "source", "sink"), rsu_rows)

    siu_rows = [(p, src, sink) for p, src, sink in triples
                if _has_ramp_method(p, sink, RAMP_LIMIT_METHOD)
                and _ramp_speed("sink", p, sink, "up") > 0]
    _write_csv(solve_data_dir / "process_source_sink_ramp_limit_sink_up.csv",
               ("process", "source", "sink"), siu_rows)

    rsd_rows = [(p, src, sink) for p, src, sink in triples
                if _has_ramp_method(p, src, RAMP_LIMIT_METHOD)
                and _ramp_speed("source", p, src, "down") > 0]
    _write_csv(solve_data_dir / "process_source_sink_ramp_limit_source_down.csv",
               ("process", "source", "sink"), rsd_rows)

    sid_rows = [(p, src, sink) for p, src, sink in triples
                if _has_ramp_method(p, sink, RAMP_LIMIT_METHOD)
                and _ramp_speed("sink", p, sink, "down") > 0]
    _write_csv(solve_data_dir / "process_source_sink_ramp_limit_sink_down.csv",
               ("process", "source", "sink"), sid_rows)

    cost_rows = [(p, src, sink) for p, src, sink in triples
                 if _has_ramp_method(p, src, RAMP_COST_METHOD)
                 or _has_ramp_method(p, sink, RAMP_COST_METHOD)]
    _write_csv(solve_data_dir / "process_source_sink_ramp_cost.csv",
               ("process", "source", "sink"), cost_rows)


def write_process_source_sink_ramp_unions(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1506-1528 — process_source_sink_ramp (5-way union of
    the ramp_limit_*/ramp_cost sets) plus 4 dtttdt-filtered ramp limit
    variants.

    The 5 input ramp_*.csv files are written by
    write_process_source_sink_ramp_family above. Must be called after it.

    The dtttdt-filtered sets aren't referenced by any constraint today
    (mod L3468-3543's ramp_*_constraint families inline the same filter
    on block_dtttdt instead), but the bare-decl is still present so we
    migrate faithfully.

    dtttdt-filter for each side/direction:
        p_process_<side>[p, n, 'ramp_speed_<dir>'] * 60 < step_duration[d, t]
        AND dt_jump[d, t] == 1
    """
    ramp_files = (
        "process_source_sink_ramp_limit_source_up.csv",
        "process_source_sink_ramp_limit_sink_up.csv",
        "process_source_sink_ramp_limit_source_down.csv",
        "process_source_sink_ramp_limit_sink_down.csv",
        "process_source_sink_ramp_cost.csv",
    )
    seen: dict[tuple[str, ...], None] = {}
    for fname in ramp_files:
        for r in _read_n_col(solve_data_dir / fname, 3):
            seen.setdefault(r, None)
    _write_csv(solve_data_dir / "process_source_sink_ramp.csv",
               ("process", "source", "sink"),
               list(seen.keys()))

    header9 = ("process", "source", "sink", "period", "time", "previous",
               "previous_within_timeset", "previous_period",
               "previous_within_solve")

    sp_path = solve_data_dir / "step_previous.csv"
    if not sp_path.exists():
        for tail in ("source_up", "sink_up", "source_down", "sink_down"):
            _write_csv(
                solve_data_dir
                / f"process_source_sink_dtttdt_ramp_limit_{tail}.csv",
                header9, [],
            )
        return

    # step_previous.csv cols: period, time, previous, previous_within_timeset,
    #                         previous_period, previous_within_solve, jump
    dtttdt_jump1: list[tuple[str, ...]] = []
    with sp_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) < 7 or not r[0] or not r[1]:
                continue
            try:
                jump = int(r[6])
            except ValueError:
                try:
                    jump = int(float(r[6]))
                except ValueError:
                    continue
            if jump == 1:
                dtttdt_jump1.append(tuple(r[:6]))

    step_duration: dict[tuple[str, str], float] = {}
    siu_path = solve_data_dir / "steps_in_use.csv"
    if siu_path.exists():
        with siu_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        step_duration[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue

    ramp_speed: dict[tuple[str, str, str, str], float] = {}
    for fname, side_label in (
        ("p_process_source.csv", "source"),
        ("p_process_sink.csv",   "sink"),
    ):
        path = input_dir / fname
        if not path.exists():
            continue
        with path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if (len(r) >= 4 and r[0] and r[1]
                        and r[2] in ("ramp_speed_up", "ramp_speed_down")):
                    try:
                        ramp_speed[(side_label, r[0], r[1], r[2])] = float(r[3])
                    except ValueError:
                        continue

    def _gen(filename: str, side: str, dir_: str) -> list[tuple[str, ...]]:
        triples = _read_n_col(solve_data_dir / filename, 3)
        if not triples:
            return []
        out: list[tuple[str, ...]] = []
        for p, src, sink in triples:
            n = src if side == "source" else sink
            rs60 = ramp_speed.get((side, p, n, f"ramp_speed_{dir_}"), 0.0) * 60.0
            for tup in dtttdt_jump1:
                d, t = tup[0], tup[1]
                if rs60 < step_duration.get((d, t), 0.0):
                    out.append((p, src, sink, *tup))
        return out

    _write_csv(
        solve_data_dir / "process_source_sink_dtttdt_ramp_limit_source_up.csv",
        header9,
        _gen("process_source_sink_ramp_limit_source_up.csv", "source", "up"),
    )
    _write_csv(
        solve_data_dir / "process_source_sink_dtttdt_ramp_limit_sink_up.csv",
        header9,
        _gen("process_source_sink_ramp_limit_sink_up.csv", "sink", "up"),
    )
    _write_csv(
        solve_data_dir / "process_source_sink_dtttdt_ramp_limit_source_down.csv",
        header9,
        _gen("process_source_sink_ramp_limit_source_down.csv", "source", "down"),
    )
    _write_csv(
        solve_data_dir / "process_source_sink_dtttdt_ramp_limit_sink_down.csv",
        header9,
        _gen("process_source_sink_ramp_limit_sink_down.csv", "sink", "down"),
    )


def write_process_source_sink_is_node_family(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1071, L1153-1158 — process_source_sink filtered by
    sink ∈ process_sink (i.e. sink is a real node-side endpoint), then
    further partitioned by the process's method bucket.

        process__source__sinkIsNode             := no method filter
        process__source__sinkIsNode_not2way1var := process p has any method NOT in METHOD_2WAY_1VAR
        process__source__sinkIsNode_2way1var    := process p has any method IN METHOD_2WAY_1VAR
        process__source__sinkIsNode_2way2var    := process p has any method IN METHOD_2WAY_2VAR

    The mod's `sum{(p,m) in process_method : m in S} 1` is true iff p has
    AT LEAST ONE method in S — equivalent to set-membership on the
    per-process method set.
    """
    from flextool.flextoolrunner.preprocessing._method_constants import (
        METHOD_2WAY_1VAR, METHOD_2WAY_2VAR,
    )
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    sinks = frozenset(_read_pairs(input_dir / "process__sink.csv"))
    # process_method canonical source is input/ (mod loader L1003).
    # solve_data/process_method.csv is mod's printf copy, only exists
    # after the first solve.
    pm = _read_pairs(input_dir / "process_method.csv")
    methods_of_p: dict[str, set[str]] = {}
    for p, m in pm:
        methods_of_p.setdefault(p, set()).add(m)
    has_2way_1var = {p: bool(ms & METHOD_2WAY_1VAR) for p, ms in methods_of_p.items()}
    has_not_2way_1var = {p: bool(ms - METHOD_2WAY_1VAR) for p, ms in methods_of_p.items()}
    has_2way_2var = {p: bool(ms & METHOD_2WAY_2VAR) for p, ms in methods_of_p.items()}

    base_rows = [(p, src, sink) for p, src, sink in triples if (p, sink) in sinks]
    _write_csv(solve_data_dir / "process__source__sinkIsNode.csv",
               ("process", "source", "sink"), base_rows)
    _write_csv(solve_data_dir / "process__source__sinkIsNode_2way1var.csv",
               ("process", "source", "sink"),
               [r for r in base_rows if has_2way_1var.get(r[0], False)])
    _write_csv(solve_data_dir / "process__source__sinkIsNode_not2way1var.csv",
               ("process", "source", "sink"),
               [r for r in base_rows if has_not_2way_1var.get(r[0], False)])
    _write_csv(solve_data_dir / "process__source__sinkIsNode_2way2var.csv",
               ("process", "source", "sink"),
               [r for r in base_rows if has_2way_2var.get(r[0], False)])


def write_process_source_sink_delayed_partition(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1096-1097 — partition process_source_sink by process_delayed.

        set process_source_sink_undelayed := {(p, source, sink) in process_source_sink : p not in process_delayed};
        set process_source_sink_delayed   := {(p, source, sink) in process_source_sink : p in process_delayed};
    """
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    delayed = frozenset(_read_singles(solve_data_dir / "process_delayed.csv"))
    delayed_rows = [(p, src, sink) for p, src, sink in triples if p in delayed]
    undelayed_rows = [(p, src, sink) for p, src, sink in triples if p not in delayed]
    _write_csv(solve_data_dir / "process_source_sink_delayed.csv",
               ("process", "source", "sink"), delayed_rows)
    _write_csv(solve_data_dir / "process_source_sink_undelayed.csv",
               ("process", "source", "sink"), undelayed_rows)


def write_node_group_dispatch_sets(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1596-1657 — 12 nodeGroupDispatch sets joining
    process_source_sink_alwaysProcess with nodeGroupDispatch + group_node
    + group_process_node + flowAggregator + process_unit/connection.

    Eight 'base' sets partition the (g, p, source, sink) space by
    side (which leg of the arc is in the group), process kind
    (unit vs connection) and aggregation (with vs without flow
    aggregator). Four 'projection' sets project pairs (g, p) or
    (g, ga) from the relevant base sets.

    All sets share a common prefilter:
        (g, p) not in nodeGroupDispatch__process_fully_inside

    Mod's printfs at L5181-5253 (inside the `if solveFirst` block)
    write the same files but only on the first solve. They're
    redundant after migration but harmless — mod will load the
    Python-written CSV via table-data-IN, then iterate it back out.
    Row order in the output may differ between mod's printf and
    Python's emission, but `read_sets.py` reads via pandas with
    `set_index(...).index` so row order doesn't matter downstream.
    """
    ngd = _read_singles(input_dir / "nodeGroupDispatch.csv")
    fag = frozenset(_read_singles(input_dir / "flowAggregator.csv"))
    p_unit = frozenset(_read_singles(input_dir / "process_unit.csv"))
    p_conn = frozenset(_read_singles(input_dir / "process_connection.csv"))

    g_nodes_acc: dict[str, dict[str, None]] = {}
    for g, n in _read_pairs(input_dir / "group__node.csv"):
        g_nodes_acc.setdefault(g, {})[n] = None
    g_nodes: dict[str, frozenset[str]] = {
        g: frozenset(d.keys()) for g, d in g_nodes_acc.items()
    }

    # group_process_node restricted to flowAggregator groups: (p, n) -> [ga, ...]
    pn_to_aggregators: dict[tuple[str, str], list[str]] = {}
    for g, p, n in _read_n_col(input_dir / "group__process__node.csv", 3):
        if g in fag:
            pn_to_aggregators.setdefault((p, n), []).append(g)

    pss_always = _read_n_col(
        solve_data_dir / "process_source_sink_alwaysProcess.csv", 3
    )
    fully_inside = frozenset(_read_pairs(
        solve_data_dir / "nodeGroupDispatch__process_fully_inside.csv"
    ))

    def _emit_4tuple(*, kind: frozenset[str], side: str,
                     not_aggregated: bool) -> list[tuple[str, ...]]:
        """side ∈ {'sink', 'source'} — which leg must be in group_node."""
        out: list[tuple[str, ...]] = []
        for g in ngd:
            gnodes = g_nodes.get(g, frozenset())
            if not gnodes:
                continue
            for p, src, sink in pss_always:
                if p not in kind:
                    continue
                if (g, p) in fully_inside:
                    continue
                n = sink if side == "sink" else src
                if n not in gnodes:
                    continue
                if not_aggregated:
                    if (p, n) in pn_to_aggregators:
                        continue
                out.append((g, p, src, sink))
        return out

    def _emit_5tuple(*, kind: frozenset[str], side: str
                     ) -> list[tuple[str, ...]]:
        """5-tuple variant: include each ga ∈ flowAggregator with
        (ga, p, n_side) ∈ group_process_node."""
        out: list[tuple[str, ...]] = []
        for g in ngd:
            gnodes = g_nodes.get(g, frozenset())
            if not gnodes:
                continue
            for p, src, sink in pss_always:
                if p not in kind:
                    continue
                if (g, p) in fully_inside:
                    continue
                n = sink if side == "sink" else src
                if n not in gnodes:
                    continue
                for ga in pn_to_aggregators.get((p, n), ()):
                    out.append((g, ga, p, src, sink))
        return out

    # Set 1: process__unit__to_node_Not_in_aggregate — sink ∈ group, no ga
    rows1 = _emit_4tuple(kind=p_unit, side="sink", not_aggregated=True)
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv",
        ("group", "process", "unit", "node"), rows1,
    )

    # Set 2: process__node__to_unit_Not_in_aggregate — source ∈ group, no ga
    rows2 = _emit_4tuple(kind=p_unit, side="source", not_aggregated=True)
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv",
        ("group", "process", "node", "unit"), rows2,
    )

    # Set 3: group_aggregate__process__unit__to_node — sink ∈ group, ga
    rows3 = _emit_5tuple(kind=p_unit, side="sink")
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__group_aggregate__process__unit__to_node.csv",
        ("group", "group_aggregate", "unit", "source", "sink"), rows3,
    )

    # Set 4: group_aggregate__process__node__to_unit — source ∈ group, ga
    rows4 = _emit_5tuple(kind=p_unit, side="source")
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__group_aggregate__process__node__to_unit.csv",
        ("group", "group_aggregate", "unit", "source", "sink"), rows4,
    )

    # Set 5: process__node__to_connection_Not_in_aggregate — source ∈ group, no ga
    rows5 = _emit_4tuple(kind=p_conn, side="source", not_aggregated=True)
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv",
        ("group", "process", "node", "connection"), rows5,
    )

    # Set 6: process__connection__to_node_Not_in_aggregate — sink ∈ group, no ga
    rows6 = _emit_4tuple(kind=p_conn, side="sink", not_aggregated=True)
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv",
        ("group", "process", "connection", "node"), rows6,
    )

    # Set 7: connection_Not_in_aggregate (g, p) — projection of 5 ∪ 6
    seen7: dict[tuple[str, str], None] = {}
    for g, p, _, _ in rows5:
        seen7.setdefault((g, p), None)
    for g, p, _, _ in rows6:
        seen7.setdefault((g, p), None)
    _write_csv(solve_data_dir / "nodeGroupDispatch__connection_Not_in_aggregate.csv",
               ("group", "connection"), list(seen7.keys()))

    # Set 8: group_aggregate__process__connection__to_node — sink ∈ group, ga
    rows8 = _emit_5tuple(kind=p_conn, side="sink")
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__group_aggregate__process__connection__to_node.csv",
        ("group", "group_aggregate", "connection", "source", "sink"), rows8,
    )

    # Set 9: group_aggregate__process__node__to_connection — source ∈ group, ga
    rows9 = _emit_5tuple(kind=p_conn, side="source")
    _write_csv(
        solve_data_dir / "nodeGroupDispatch__group_aggregate__process__node__to_connection.csv",
        ("group", "group_aggregate", "connection", "source", "sink"), rows9,
    )

    # Set 10: group_aggregate_Connection (g, ga) — projection of 8 ∪ 9
    seen10: dict[tuple[str, str], None] = {}
    for g, ga, _, _, _ in rows8:
        seen10.setdefault((g, ga), None)
    for g, ga, _, _, _ in rows9:
        seen10.setdefault((g, ga), None)
    _write_csv(solve_data_dir / "nodeGroupDispatch__group_aggregate_Connection.csv",
               ("group", "group_aggregate"), list(seen10.keys()))

    # Set 11: group_aggregate_Unit_to_group (g, ga) — projection of 3
    seen11: dict[tuple[str, str], None] = {}
    for g, ga, _, _, _ in rows3:
        seen11.setdefault((g, ga), None)
    _write_csv(solve_data_dir / "nodeGroupDispatch__group_aggregate_Unit_to_group.csv",
               ("group", "group_aggregate"), list(seen11.keys()))

    # Set 12: group_aggregate_Group_to_unit (g, ga) — projection of 4
    seen12: dict[tuple[str, str], None] = {}
    for g, ga, _, _, _ in rows4:
        seen12.setdefault((g, ga), None)
    _write_csv(solve_data_dir / "nodeGroupDispatch__group_aggregate_Group_to_unit.csv",
               ("group", "group_aggregate"), list(seen12.keys()))


def write_small_set_derivations(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L999, L1061, L1132, L1174, L1222-1223 — 6 small
    derived sets that depend on already-migrated solve_data CSVs.

      * ed_history_realized              (mod L999) = ed_history_realized_read
                                                    ∪ ed_history_realized_first
      * process__source__sink__profile__profile_method
                                          (mod L1061) = 4-way union of
                                          profile_method connection/direct +
                                          profileProcess source/sink sets
      * process_sinkIsNode_2way1var       (mod L1132) = setof p from
                                          process__source__sinkIsNode_2way1var
      * nodeSelfDischarge                 (mod L1174) = {n ∈ nodeState : ∃
                                          (d, t) ∈ dt with pdtNode[n,
                                          'self_discharge_loss', d, t] != 0}
      * pdt_online_linear/integer         (mod L1222-3) = {p ∈ process_online_*,
                                          (d, t) ∈ dt : pdProcess[p,
                                          'startup_cost', d] != 0}
    """
    # ed_history_realized (mod L999)
    ed_read = _read_pairs(
        solve_data_dir / "p_entity_period_existing_capacity.csv"
    )
    ed_first = _read_pairs(solve_data_dir / "ed_history_realized_first.csv")
    seen_ed: dict[tuple[str, str], None] = {}
    for r in ed_read:
        seen_ed.setdefault(r, None)
    for r in ed_first:
        seen_ed.setdefault(r, None)
    _write_csv(solve_data_dir / "ed_history_realized.csv",
               ("entity", "period"), list(seen_ed.keys()))

    # process__source__sink__profile__profile_method (mod L1061) = 4-way union
    seen_pf: dict[tuple[str, ...], None] = {}
    for fname in (
        "process__profileProcess__toSink__profile__profile_method.csv",
        "process__source__toProfileProcess__profile__profile_method.csv",
        "process__source__sink__profile__profile_method_connection.csv",
        "process__source__sink__profile__profile_method_direct.csv",
    ):
        for r in _read_n_col(solve_data_dir / fname, 5):
            seen_pf.setdefault(r, None)
    _write_csv(
        solve_data_dir / "process__source__sink__profile__profile_method.csv",
        ("process", "source", "sink", "profile", "profile_method"),
        list(seen_pf.keys()),
    )

    # process_sinkIsNode_2way1var (mod L1132) = projection of column 0
    triples = _read_n_col(
        solve_data_dir / "process__source__sinkIsNode_2way1var.csv", 3
    )
    seen_p: dict[str, None] = {}
    for p, _, _ in triples:
        seen_p.setdefault(p, None)
    _write_csv(solve_data_dir / "process_sinkIsNode_2way1var.csv",
               ("process",), [(p,) for p in seen_p.keys()])

    # nodeSelfDischarge (mod L1174) — exists filter on pdtNode
    nodeState = frozenset(_read_singles(solve_data_dir / "nodeState.csv"))
    nodes_with_selfdischarge: set[str] = set()
    pdtn_path = solve_data_dir / "pdtNode.csv"
    if pdtn_path.exists():
        with pdtn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if (len(r) >= 5 and r[0] in nodeState
                        and r[1] == "self_discharge_loss"):
                    try:
                        if float(r[4]) != 0.0:
                            nodes_with_selfdischarge.add(r[0])
                    except ValueError:
                        continue
    _write_csv(solve_data_dir / "nodeSelfDischarge.csv",
               ("node",),
               [(n,) for n in _read_singles(solve_data_dir / "nodeState.csv")
                if n in nodes_with_selfdischarge])

    # pdt_online_linear / pdt_online_integer (mod L1222-1223)
    # Filter dt × process_online_* by pdProcess[p, 'startup_cost', d] != 0.
    pd_startup: set[tuple[str, str]] = set()  # (process, period) where startup_cost != 0
    pdp_path = solve_data_dir / "pdProcess.csv"
    if pdp_path.exists():
        with pdp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] == "startup_cost" and r[2]:
                    try:
                        if float(r[3]) != 0.0:
                            pd_startup.add((r[0], r[2]))
                    except ValueError:
                        continue
    dt_pairs = _read_n_col(solve_data_dir / "steps_in_use.csv", 2)
    for fname_in, fname_out in (
        ("process_online_linear.csv",  "pdt_online_linear.csv"),
        ("process_online_integer.csv", "pdt_online_integer.csv"),
    ):
        procs = _read_singles(solve_data_dir / fname_in)
        rows: list[tuple[str, str, str]] = []
        for p in procs:
            for d, t in dt_pairs:
                if (p, d) in pd_startup:
                    rows.append((p, d, t))
        _write_csv(solve_data_dir / fname_out,
                   ("process", "period", "time"), rows)


def write_param_t_projections_and_time_params(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L393-L1149 — 8-set family around the process__param
    projections and their sourceSinkTimeParam joins.

    Projections (drop the time column from each *__time set):
      * process__param_t          (mod L393)
      * connection__param__time   (mod L398, filter by process_connection)
      * connection__param_t       (mod L399)
      * process__source__param_t  (mod L403)
      * process__sink__param_t    (mod L406)

    Joins with SOURCE_SINK_TIME_PARAM (3 sets):
      * process__source__timeParam (mod L1134)
      * process__sink__timeParam   (mod L1140)
      * process__timeParam         (mod L1146; only for connection processes)

    Reads:
    - solve_data/pt_process.csv (process, param, time, value) →
      process__param__time
    - solve_data/pt_process_source.csv (process, source, param, time, value)
    - solve_data/pt_process_sink.csv (process, sink, param, time, value)
    - input/p_process_source.csv / p_process_sink.csv / p_process.csv
      (provide the static *__param sets)
    - input/process_connection.csv
    """
    from flextool.flextoolrunner.preprocessing._param_taxonomy import (
        SOURCE_SINK_TIME_PARAM,
    )
    proc_conn = frozenset(
        _read_singles(input_dir / "process_connection.csv")
    )

    # process__param__time → projection (process, param) [drop time]
    pp_t_seen: dict[tuple[str, str], None] = {}
    conn_pt_rows: list[tuple[str, str, str]] = []
    conn_pt_seen: dict[tuple[str, str], None] = {}
    pt_path = solve_data_dir / "pt_process.csv"
    if pt_path.exists():
        with pt_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    pp_t_seen.setdefault((r[0], r[1]), None)
                    if r[0] in proc_conn:
                        conn_pt_rows.append((r[0], r[1], r[2]))
                        conn_pt_seen.setdefault((r[0], r[1]), None)
    _write_csv(solve_data_dir / "process__param_t.csv",
               ("process", "param"), list(pp_t_seen.keys()))
    _write_csv(solve_data_dir / "connection__param__time.csv",
               ("connection", "param", "time"), conn_pt_rows)
    _write_csv(solve_data_dir / "connection__param_t.csv",
               ("connection", "param"), list(conn_pt_seen.keys()))

    # process__source__param_t (drop time)
    pps_t_seen: dict[tuple[str, str, str], None] = {}
    pts_path = solve_data_dir / "pt_process_source.csv"
    if pts_path.exists():
        with pts_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] and r[2]:
                    pps_t_seen.setdefault((r[0], r[1], r[2]), None)
    _write_csv(solve_data_dir / "process__source__param_t.csv",
               ("process", "source", "param"), list(pps_t_seen.keys()))

    # process__sink__param_t
    ppk_t_seen: dict[tuple[str, str, str], None] = {}
    ptk_path = solve_data_dir / "pt_process_sink.csv"
    if ptk_path.exists():
        with ptk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] and r[2]:
                    ppk_t_seen.setdefault((r[0], r[1], r[2]), None)
    _write_csv(solve_data_dir / "process__sink__param_t.csv",
               ("process", "sink", "param"), list(ppk_t_seen.keys()))

    # process__source__param (static) from input/p_process_source.csv
    src_param: set[tuple[str, str, str]] = set()
    pps_path = input_dir / "p_process_source.csv"
    if pps_path.exists():
        with pps_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    src_param.add((r[0], r[1], r[2]))
    sink_param: set[tuple[str, str, str]] = set()
    ppk_path = input_dir / "p_process_sink.csv"
    if ppk_path.exists():
        with ppk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] and r[2]:
                    sink_param.add((r[0], r[1], r[2]))
    proc_param: set[tuple[str, str]] = set()
    pp_path = input_dir / "p_process.csv"
    if pp_path.exists():
        with pp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] and r[1]:
                    proc_param.add((r[0], r[1]))

    # process__source__timeParam: { (p, source) in process_source, param :
    #   (p, src, param) in process__source__param OR in process__source__param_t }
    proc_sources = _read_pairs(input_dir / "process__source.csv")
    proc_sinks = _read_pairs(input_dir / "process__sink.csv")
    pps_t_set = frozenset(pps_t_seen.keys())
    ppk_t_set = frozenset(ppk_t_seen.keys())
    pp_t_set = frozenset(pp_t_seen.keys())

    rows_src_tp: list[tuple[str, str, str]] = []
    for p, src in proc_sources:
        for param in SOURCE_SINK_TIME_PARAM:
            if ((p, src, param) in src_param
                    or (p, src, param) in pps_t_set):
                rows_src_tp.append((p, src, param))
    _write_csv(solve_data_dir / "process__source__timeParam.csv",
               ("process", "source", "param"), rows_src_tp)

    rows_snk_tp: list[tuple[str, str, str]] = []
    for p, snk in proc_sinks:
        for param in SOURCE_SINK_TIME_PARAM:
            if ((p, snk, param) in sink_param
                    or (p, snk, param) in ppk_t_set):
                rows_snk_tp.append((p, snk, param))
    _write_csv(solve_data_dir / "process__sink__timeParam.csv",
               ("process", "sink", "param"), rows_snk_tp)

    # process__timeParam: { p in process, param in sourceSinkTimeParam :
    #     ((p, param) in process__param   AND p in process_connection)
    #  OR ((p, param) in process__param_t AND p in process_connection) }
    processes = _read_singles(input_dir / "process.csv")
    rows_p_tp: list[tuple[str, str]] = []
    for p in processes:
        if p not in proc_conn:
            continue
        for param in SOURCE_SINK_TIME_PARAM:
            if (p, param) in proc_param or (p, param) in pp_t_set:
                rows_p_tp.append((p, param))
    _write_csv(solve_data_dir / "process__timeParam.csv",
               ("process", "param"), rows_p_tp)


def write_gdt_instant_flow_sets(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1131-1132 — gdt_maxInstantFlow + gdt_minInstantFlow.

        gdt_maxInstantFlow := {g in group, (d, t) in dt :
                                pdtGroup[g, 'max_instant_flow', d, t]};
        gdt_minInstantFlow := {g in group, (d, t) in dt :
                                pdtGroup[g, 'min_instant_flow', d, t]};

    Reads solve_data/pdtGroup.csv (cols: group, param, period, time, value).
    Filter is "value != 0" — mathprog `pdtGroup[...]` in a boolean
    context is true iff non-zero.
    """
    max_rows: list[tuple[str, str, str]] = []
    min_rows: list[tuple[str, str, str]] = []
    pdtg_path = solve_data_dir / "pdtGroup.csv"
    if pdtg_path.exists():
        with pdtg_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if (len(r) >= 5 and r[0] and r[2] and r[3]):
                    try:
                        v = float(r[4])
                    except ValueError:
                        continue
                    if v == 0.0:
                        continue
                    if r[1] == "max_instant_flow":
                        max_rows.append((r[0], r[2], r[3]))
                    elif r[1] == "min_instant_flow":
                        min_rows.append((r[0], r[2], r[3]))
    _write_csv(solve_data_dir / "gdt_maxInstantFlow.csv",
               ("group", "period", "time"), max_rows)
    _write_csv(solve_data_dir / "gdt_minInstantFlow.csv",
               ("group", "period", "time"), min_rows)


def write_p_process_delay_weight(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1096-1099 — p_process_delay_weight.

        param p_process_delay_weight {(p, td) in process_delayed__duration}
            := if (p, td) in process_delay_single__delay_duration
               then 1
               else p_process_delay_weighted[p, td];

    Reads solve_data/process_delayed__duration.csv (already migrated),
    input/process_delay_single.csv, input/p_process_delay_weighted.csv.
    """
    delayed_duration = _read_pairs(
        solve_data_dir / "process_delayed__duration.csv"
    )
    delay_single = frozenset(
        _read_pairs(input_dir / "process_delay_single.csv")
    )
    weighted: dict[tuple[str, str], float] = {}
    pdw_path = input_dir / "p_process_delay_weighted.csv"
    if pdw_path.exists():
        with pdw_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        weighted[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue
    rows: list[tuple[str, str, str]] = []
    for p, td in delayed_duration:
        v = 1.0 if (p, td) in delay_single else weighted.get((p, td), 0.0)
        rows.append((p, td, repr(v)))
    _write_csv(solve_data_dir / "p_process_delay_weight.csv",
               ("process", "delay_duration", "value"), rows)


def write_gcndt_co2_price(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1542-1548 — gcndt_co2_price (5-tuple set).

        {g in group, (c,n) in commodity_node, d in period_in_use,
         t in time_in_use:
             (d,t) in dt
             AND (g, n) in group_node
             AND p_commodity[c, 'co2_content']
             AND g in group_co2_price
             AND pdtGroup[g, 'co2_price', d, t]
        };

    Note: the period_in_use × time_in_use × `(d,t) in dt` triple gate
    is equivalent to iterating (d, t) ∈ dt directly (dt ⊆ period_in_use
    × time_in_use by construction).
    """
    g_co2_price = frozenset(
        _read_singles(solve_data_dir / "group_co2_price.csv")
    )
    cn = _read_pairs(input_dir / "commodity__node.csv")

    gn_acc: dict[str, set[str]] = {}
    for g, n in _read_pairs(input_dir / "group__node.csv"):
        gn_acc.setdefault(g, set()).add(n)

    p_commodity_co2: dict[str, float] = {}
    pc_path = input_dir / "p_commodity.csv"
    if pc_path.exists():
        with pc_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] == "co2_content":
                    try:
                        p_commodity_co2[r[0]] = float(r[2])
                    except ValueError:
                        continue

    co2_price_dt: set[tuple[str, str, str]] = set()
    pdtg_path = solve_data_dir / "pdtGroup.csv"
    if pdtg_path.exists():
        with pdtg_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if (len(r) >= 5 and r[0] and r[1] == "co2_price"
                        and r[2] and r[3]):
                    try:
                        if float(r[4]) != 0.0:
                            co2_price_dt.add((r[0], r[2], r[3]))
                    except ValueError:
                        continue

    dt_pairs = _read_n_col(solve_data_dir / "steps_in_use.csv", 2)

    rows: list[tuple[str, str, str, str, str]] = []
    for g in g_co2_price:
        gnodes = gn_acc.get(g, set())
        if not gnodes:
            continue
        for c, n in cn:
            if n not in gnodes:
                continue
            if p_commodity_co2.get(c, 0.0) == 0.0:
                continue
            for d, t in dt_pairs:
                if (g, d, t) in co2_price_dt:
                    rows.append((g, c, n, d, t))
    _write_csv(solve_data_dir / "gcndt_co2_price.csv",
               ("group", "commodity", "node", "period", "time"), rows)


def write_group_commodity_node_period_co2_period(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1550-1555 — group_commodity_node_period_co2_period.

        {g in group, (c, n) in commodity_node, d in period_in_use:
            (g, n) in group_node
            AND p_commodity[c, 'co2_content']
            AND g in group_co2_max_period
        };
    """
    g_co2_max_period = frozenset(
        _read_singles(solve_data_dir / "group_co2_max_period.csv")
    )
    cn = _read_pairs(input_dir / "commodity__node.csv")

    gn_acc: dict[str, set[str]] = {}
    for g, n in _read_pairs(input_dir / "group__node.csv"):
        gn_acc.setdefault(g, set()).add(n)

    p_commodity_co2: dict[str, float] = {}
    pc_path = input_dir / "p_commodity.csv"
    if pc_path.exists():
        with pc_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1] == "co2_content":
                    try:
                        p_commodity_co2[r[0]] = float(r[2])
                    except ValueError:
                        continue

    period_in_use = _read_singles(
        solve_data_dir / "period_in_use_set.csv"
    )

    rows: list[tuple[str, str, str, str]] = []
    for g in g_co2_max_period:
        gnodes = gn_acc.get(g, set())
        if not gnodes:
            continue
        for c, n in cn:
            if n not in gnodes:
                continue
            if p_commodity_co2.get(c, 0.0) == 0.0:
                continue
            for d in period_in_use:
                rows.append((g, c, n, d))
    _write_csv(solve_data_dir / "group_commodity_node_period_co2_period.csv",
               ("group", "commodity", "node", "period"), rows)


def write_peedt(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1084 — peedt is the cross-product of arcs × timesteps.

        set peedt := {(p, source, sink) in process_source_sink, (d, t) in dt};

    Used as the index set for v_flow / p_flow_max / p_flow_min / d_flow*.
    On full-year hourly fixtures this produces hundreds of thousands of rows,
    so the writer streams output in chunks rather than building the full
    list in memory.
    """
    triples = _read_n_col(solve_data_dir / "process_source_sink.csv", 3)
    dt_pairs = _read_n_col(solve_data_dir / "steps_in_use.csv", 2)
    out_path = solve_data_dir / "peedt.csv"
    with out_path.open("w") as fh:
        fh.write("process,source,sink,period,time\n")
        for p, src, snk in triples:
            for d, t in dt_pairs:
                fh.write(f"{p},{src},{snk},{d},{t}\n")
