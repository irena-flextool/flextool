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
