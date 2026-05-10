"""Writer-port Phase 1 follow-up — process-arc-union + period-param subset.

Native polars port of the simpler, leaf-like ``write_*`` helpers in
:mod:`flextool.flextoolrunner.preprocessing.process_arc_unions` plus a
single self-contained writer from
:mod:`flextool.flextoolrunner.preprocessing.entity_period_calc_params`.

The two source modules are large (~2.3 kLOC and ~2.4 kLOC respectively);
this dispatch chips the cheap, leaf-like writers out — i.e. those whose
inputs are already-native L0-L9 ``solve_data/*.csv`` outputs (or plain
``input/*.csv``) and whose semantics are pure projection / join / filter
with no ``PdtLookup``-class machinery behind them.

Ported writers (legacy LOC budget ~535):

From ``process_arc_unions.py``:

* ``write_process_source_sink_param_t``                              (~38 LOC)
* ``write_node_time_param_in_use``                                   (~44 LOC)
* ``write_process_source_delayed_partition``                         (~18 LOC)
* ``write_process_source_sink_param``                                (~62 LOC)
* ``write_process_source_sink_profile_method_connection``            (~35 LOC)
* ``write_process_method_sources_sinks``                             (~56 LOC)
* ``write_ed_history_realized_first``                                (~56 LOC)
* ``write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source``
                                                                     (~50 LOC)
* ``write_process_source_sink_ramp_method``                          (~32 LOC)
* ``write_process_source_sink_coeff_zero``                           (~24 LOC)
* ``write_process_source_sink_is_node_family``                       (~46 LOC)
* ``write_process_source_sink_delayed_partition``                    (~18 LOC)

From ``entity_period_calc_params.py``:

* ``write_pProcess_source_sink``                                     (~56 LOC)

The four ``write_pdtProcess`` / ``write_pdtNode`` / ``write_pdtProcess_source``
/ ``write_pdtProcess_sink`` writers from ``entity_period_calc_params``
were *also* on the candidate list but lean on the ~200 LOC ``PdtLookup``
class hierarchy from ``preprocessing/pd_lookups.py``.  Porting them
sensibly requires lifting that whole machinery — deferred to the next
dispatch.

Each ``write_*`` is a thin wrapper around a ``derive_*`` (or, where the
legacy emits multiple CSVs from one shared computation, around a small
``_compute_*`` helper).  The ``derive_*`` returns a fresh
``pl.DataFrame`` in the same in-memory contract as
:mod:`._writer_leaf_sets` / :mod:`._writer_mid_sets` /
:mod:`._writer_calc_params`.

Style mirrors :mod:`._writer_calc_params` — eager polars reads of tiny
CSVs with positional column renames, expression chains where natural,
small python loops where the iteration order is precision-load-bearing
(matches the legacy ``dict.fromkeys`` ordered-dedup pattern).

Precision-parity pattern
------------------------

``write_pProcess_source_sink`` writes a value column.  Legacy formats
it via ``f"{repr(v)}"`` with ``v`` already a python float — we mirror
exactly by pre-stringifying with ``repr(float(v))``.  See
:mod:`._writer_calc_params` module docstring for the precision-parity
rationale (round-trip-exactness of ``repr(float)`` and divergence
from polars' default float formatting).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# CSV I/O — same conventions as the sibling _writer_*.py modules.
# ---------------------------------------------------------------------------

def _read_csv(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read a tiny flextool CSV with positional column rename."""
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame({c: [] for c in columns}, schema={c: pl.Utf8 for c in columns})
    df = pl.read_csv(
        path,
        has_header=True,
        schema_overrides={c: pl.Utf8 for c in columns},
        truncate_ragged_lines=True,
    )
    keep = df.columns[: len(columns)]
    df = df.select(keep)
    df.columns = columns
    return df


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _drop_blank_rows(df: pl.DataFrame, required_cols: list[str]) -> pl.DataFrame:
    expr = pl.col(required_cols[0]) != ""
    for c in required_cols[1:]:
        expr = expr & (pl.col(c) != "")
    return df.filter(expr)


def _read_n_col_rows(path: Path, columns: list[str]) -> list[tuple[str, ...]]:
    """Read a CSV as a list of tuples preserving CSV row order.

    Mirrors the legacy ``_read_n_col`` helper — used where the legacy
    code iterates a list (not a set) to drive deterministic dedupe via
    ``dict.fromkeys``.
    """
    df = _read_csv(path, columns)
    df = _drop_blank_rows(df, columns)
    return list(df.iter_rows())


# ---------------------------------------------------------------------------
# Constants — mirror flextool_base.dat enums.  Pinned here as frozensets
# so the native module has no transitive import from the legacy
# preprocessing tree.  If those enums change, update both sites in
# lockstep; the parity tests will catch drift.
# ---------------------------------------------------------------------------

# NOTE on enum-iteration parity: legacy code stores these as frozensets
# and iterates them directly with ``for param in FOO``.  Python's
# string-hash randomization makes the resulting iteration order
# session-dependent, BUT within a single pytest process both the legacy
# and native writers see the SAME randomized order — and the native
# writer's job is byte-identical parity within the same process, not
# stable order across runs.  We therefore mirror the legacy storage
# (frozenset, with the exact same element tuple) to guarantee identical
# iteration order in any given session.

# flextool_base.dat:153 — PROCESS_TIME_PARAM
_PROCESS_TIME_PARAM: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load",
    "other_operational_cost", "availability",
))

# flextool_base.dat:155 — SOURCE_SINK_PARAM
_SOURCE_SINK_PARAM: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load", "coefficient",
    "flow_unitsize", "other_operational_cost", "ramp_cost",
    "ramp_speed_up", "ramp_speed_down", "inertia_constant",
))

# flextool_base.dat:178 — NODE_TIME_PARAM
_NODE_TIME_PARAM: frozenset[str] = frozenset((
    "inflow", "penalty_down", "penalty_up", "self_discharge_loss",
    "availability", "storage_state_reference_value",
))

# flextool_base.dat:179 — NODE_TIME_PARAM_REQUIRED
_NODE_TIME_PARAM_REQUIRED: frozenset[str] = frozenset((
    "inflow", "penalty_down", "penalty_up",
))

# flextool_base.dat:28-31 — RAMP_METHOD
_RAMP_METHOD: frozenset[str] = frozenset(("ramp_limit", "ramp_cost", "both"))

# preprocessing/_method_constants.py L90 / L93 — speed/cost-gated subsets.
_RAMP_LIMIT_METHOD: frozenset[str] = frozenset(("ramp_limit", "both"))
_RAMP_COST_METHOD: frozenset[str] = frozenset(("ramp_cost", "both"))

# flextool_base.dat:69-70 — METHOD_1WAY
_METHOD_1WAY: frozenset[str] = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
    "method_1way_nvar_off", "method_1way_nvar_LP", "method_1way_nvar_MIP",
))

# flextool_base.dat:84 — METHOD_2WAY_1VAR
_METHOD_2WAY_1VAR: frozenset[str] = frozenset(("method_2way_1var_off",))

# flextool_base.dat:85 — METHOD_2WAY_2VAR
_METHOD_2WAY_2VAR: frozenset[str] = frozenset((
    "method_2way_2var_off", "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))


# ===========================================================================
# process_arc_unions — leaf-like writers
# ===========================================================================


# ---- process_source_sink_param_t (mod L1197) ------------------------------

def derive_process_source_sink_param_t(solve_data_dir: Path) -> pl.DataFrame:
    """``process_source_sink_eff`` × ``processTimeParam`` filtered by
    ``(p, param) in process__param_t`` (loaded from ``pt_process.csv``).
    """
    pss_eff = _read_n_col_rows(
        solve_data_dir / "process_source_sink_eff.csv",
        ["process", "source", "sink"],
    )
    pt_pairs = _read_csv(solve_data_dir / "pt_process.csv", ["process", "param"])
    pt_pairs = _drop_blank_rows(pt_pairs, ["process", "param"])
    pt_set: set[tuple[str, str]] = set(pt_pairs.iter_rows())

    rows: list[tuple[str, str, str, str]] = []
    for p, source, sink in pss_eff:
        for param in _PROCESS_TIME_PARAM:
            if (p, param) in pt_set:
                rows.append((p, source, sink, param))
    deduped = list(dict.fromkeys(rows))
    return pl.DataFrame(
        {
            "process": [r[0] for r in deduped],
            "source":  [r[1] for r in deduped],
            "sink":    [r[2] for r in deduped],
            "param":   [r[3] for r in deduped],
        },
        schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "param": pl.Utf8},
    )


def write_process_source_sink_param_t(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process_source_sink_param_t(solve_data_dir),
        solve_data_dir / "process_source_sink_param_t.csv",
    )


# ---- node__TimeParam_in_use (mod L1208-1214) ------------------------------

def derive_node_time_param_in_use(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """node × nodeTimeParam filtered by per-node membership in
    nodeBalance / nodeBalancePeriod / nodeState
    or by ``(n, 'use_reference_value') in node__storage_solve_horizon_method``.
    """
    nodes = (
        _drop_blank_rows(_read_csv(input_dir / "node.csv", ["node"]), ["node"])
        .get_column("node").to_list()
    )
    n_balance = frozenset(
        _drop_blank_rows(_read_csv(solve_data_dir / "nodeBalance.csv", ["node"]), ["node"])
        .get_column("node").to_list()
    )
    n_balance_period = frozenset(
        _drop_blank_rows(_read_csv(solve_data_dir / "nodeBalancePeriod.csv", ["node"]), ["node"])
        .get_column("node").to_list()
    )
    n_state = frozenset(
        _drop_blank_rows(_read_csv(solve_data_dir / "nodeState.csv", ["node"]), ["node"])
        .get_column("node").to_list()
    )
    storage_method = _drop_blank_rows(
        _read_csv(
            input_dir / "node__storage_solve_horizon_method.csv",
            ["node", "method"],
        ),
        ["node", "method"],
    )
    n_storage_use_ref = frozenset(
        storage_method.filter(pl.col("method") == "use_reference_value")
                      .get_column("node").to_list()
    )

    rows: list[tuple[str, str]] = []
    for n in nodes:
        is_bal = n in n_balance
        is_bal_period = n in n_balance_period
        is_state = n in n_state
        is_use_ref = n in n_storage_use_ref
        for param in _NODE_TIME_PARAM:
            if (is_bal or is_bal_period) and param in _NODE_TIME_PARAM_REQUIRED:
                rows.append((n, param))
            elif is_state and param in ("self_discharge_loss", "availability"):
                rows.append((n, param))
            elif is_use_ref and param == "storage_state_reference_value":
                rows.append((n, param))
    deduped = list(dict.fromkeys(rows))
    return pl.DataFrame(
        {"node": [r[0] for r in deduped], "param": [r[1] for r in deduped]},
        schema={"node": pl.Utf8, "param": pl.Utf8},
    )


def write_node_time_param_in_use(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_node_time_param_in_use(input_dir, solve_data_dir),
        solve_data_dir / "node__TimeParam_in_use.csv",
    )


# ---- process_source_{delayed,undelayed} (mod L1092-1093) -------------------

def derive_process_source_delayed_partition(
    input_dir: Path, solve_data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition ``process__source`` by membership in ``process_delayed``.

    Returns (delayed, undelayed) frames, each with columns (process, source).
    """
    pairs = _read_csv(input_dir / "process__source.csv", ["process", "source"])
    pairs = _drop_blank_rows(pairs, ["process", "source"])
    delayed_set = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "process_delayed.csv", ["process"]),
            ["process"],
        ).get_column("process").to_list()
    )
    delayed = pairs.filter(pl.col("process").is_in(list(delayed_set)))
    undelayed = pairs.filter(~pl.col("process").is_in(list(delayed_set)))
    return delayed, undelayed


def write_process_source_delayed_partition(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    delayed, undelayed = derive_process_source_delayed_partition(input_dir, solve_data_dir)
    _write(delayed, solve_data_dir / "process_source_delayed.csv")
    _write(undelayed, solve_data_dir / "process_source_undelayed.csv")


# ---- process__source__sink__param (mod L1185-1189) -------------------------

def derive_process_source_sink_param(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink × SOURCE_SINK_PARAM`` admitted if param row
    exists on EITHER side, OR via connection-process ``p_process``.
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )

    src_param = _drop_blank_rows(
        _read_csv(
            input_dir / "p_process_source.csv",
            ["process", "source", "param"],
        ),
        ["process", "source", "param"],
    )
    sink_param = _drop_blank_rows(
        _read_csv(
            input_dir / "p_process_sink.csv",
            ["process", "sink", "param"],
        ),
        ["process", "sink", "param"],
    )
    proc_param = _drop_blank_rows(
        _read_csv(input_dir / "p_process.csv", ["process", "param"]),
        ["process", "param"],
    )
    proc_conn = frozenset(
        _drop_blank_rows(
            _read_csv(input_dir / "process_connection.csv", ["process"]),
            ["process"],
        ).get_column("process").to_list()
    )

    src_set: set[tuple[str, str, str]] = set(src_param.iter_rows())
    sink_set: set[tuple[str, str, str]] = set(sink_param.iter_rows())
    proc_set: set[tuple[str, str]] = set(proc_param.iter_rows())

    rows: list[tuple[str, str, str, str]] = []
    for p, src, sink in triples:
        for param in _SOURCE_SINK_PARAM:
            if ((p, src, param) in src_set
                    or (p, sink, param) in sink_set
                    or ((p, param) in proc_set and p in proc_conn)):
                rows.append((p, src, sink, param))
    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "source":  [r[1] for r in rows],
            "sink":    [r[2] for r in rows],
            "param":   [r[3] for r in rows],
        },
        schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "param": pl.Utf8},
    )


def write_process_source_sink_param(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_process_source_sink_param(input_dir, solve_data_dir),
        solve_data_dir / "process__source__sink__param.csv",
    )


# ---- process__source__sink__profile__profile_method_connection
#      (mod L1060-1063) -------------------------------------------------------

def derive_process_source_sink_profile_method_connection(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink × profile × profile_method`` filtered by
    ``(p, profile, method) in process__profile__profile_method``.
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    pp_pm = _read_n_col_rows(
        input_dir / "process__profile__profile_method.csv",
        ["process", "profile", "profile_method"],
    )
    fm_for_p: dict[str, list[tuple[str, str]]] = {}
    for p, f, m in pp_pm:
        fm_for_p.setdefault(p, []).append((f, m))

    rows: list[tuple[str, str, str, str, str]] = []
    for p, src, sink in triples:
        for f, m in fm_for_p.get(p, ()):
            rows.append((p, src, sink, f, m))
    return pl.DataFrame(
        {
            "process":        [r[0] for r in rows],
            "source":         [r[1] for r in rows],
            "sink":           [r[2] for r in rows],
            "profile":        [r[3] for r in rows],
            "profile_method": [r[4] for r in rows],
        },
        schema={
            "process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
            "profile": pl.Utf8, "profile_method": pl.Utf8,
        },
    )


def write_process_source_sink_profile_method_connection(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_process_source_sink_profile_method_connection(input_dir, solve_data_dir),
        solve_data_dir / "process__source__sink__profile__profile_method_connection.csv",
    )


# ---- process_method_sources_sinks (mod L1046-1053) -------------------------

def derive_process_method_sources_sinks(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """3-way join over process_source_sink_alwaysProcess, process_source_sink
    and process_method, aliasing-gated.

    Output columns: (process, method, orig_source, orig_sink,
                     always_source, always_sink).
    """
    always = _read_n_col_rows(
        solve_data_dir / "process_source_sink_alwaysProcess.csv",
        ["process", "always_source", "always_sink"],
    )
    pss = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "orig_source", "orig_sink"],
    )
    pm = _read_n_col_rows(
        input_dir / "process_method.csv",
        ["process", "method"],
    )

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
    rows = list(seen.keys())
    return pl.DataFrame(
        {
            "process":       [r[0] for r in rows],
            "method":        [r[1] for r in rows],
            "orig_source":   [r[2] for r in rows],
            "orig_sink":     [r[3] for r in rows],
            "always_source": [r[4] for r in rows],
            "always_sink":   [r[5] for r in rows],
        },
        schema={
            "process": pl.Utf8, "method": pl.Utf8,
            "orig_source": pl.Utf8, "orig_sink": pl.Utf8,
            "always_source": pl.Utf8, "always_sink": pl.Utf8,
        },
    )


def write_process_method_sources_sinks(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_process_method_sources_sinks(input_dir, solve_data_dir),
        solve_data_dir / "process_method_sources_sinks.csv",
    )


# ---- ed_history_realized_first (mod L993) ---------------------------------

def derive_ed_history_realized_first(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """entity × realized periods, but only on the first solve.

    Honours the ``solveFirst`` flag on ``p_model``: non-first solves
    emit an empty frame.
    """
    # solveFirst gate: short-circuit to empty.
    solve_first = False
    pm = _read_csv(solve_data_dir / "p_model.csv", ["key", "value"])
    if pm.height > 0:
        row = pm.filter(pl.col("key") == "solveFirst")
        if row.height > 0:
            try:
                solve_first = bool(int(row.get_column("value")[0]))
            except (ValueError, TypeError):
                solve_first = False
    if not solve_first:
        return pl.DataFrame(
            {"entity": [], "period": []},
            schema={"entity": pl.Utf8, "period": pl.Utf8},
        )

    entities = (
        _drop_blank_rows(_read_csv(input_dir / "entity.csv", ["entity"]), ["entity"])
        .get_column("entity").to_list()
    )
    d_realize_invest = frozenset(
        _drop_blank_rows(
            _read_csv(
                solve_data_dir / "realized_invest_periods_of_current_solve.csv",
                ["period"],
            ),
            ["period"],
        ).get_column("period").to_list()
    )
    d_fix_storage = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "d_fix_storage_period_set.csv", ["period"]),
            ["period"],
        ).get_column("period").to_list()
    )
    d_realized = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "d_realized_period_set.csv", ["period"]),
            ["period"],
        ).get_column("period").to_list()
    )
    realized_periods = d_realize_invest | d_fix_storage | d_realized

    pb = _drop_blank_rows(
        _read_csv(solve_data_dir / "period__branch.csv", ["period", "branch"]),
        ["period", "branch"],
    )
    diag_periods = frozenset(
        d for d, b in pb.iter_rows() if d == b
    )

    rows: list[tuple[str, str]] = [
        (e, d) for e in entities
        for d in realized_periods if d in diag_periods
    ]
    return pl.DataFrame(
        {"entity": [r[0] for r in rows], "period": [r[1] for r in rows]},
        schema={"entity": pl.Utf8, "period": pl.Utf8},
    )


def write_ed_history_realized_first(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_ed_history_realized_first(input_dir, solve_data_dir),
        solve_data_dir / "ed_history_realized_first.csv",
    )


# ---- process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source
#      (mod L1152-1155) -----------------------------------------------------

def derive_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink`` filtered for 1-way processes whose source
    is a real node-side endpoint, with no sink OR ≥2 sources.
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    pm = _read_n_col_rows(input_dir / "process_method.csv", ["process", "method"])
    methods_of_p: dict[str, set[str]] = {}
    for p, m in pm:
        methods_of_p.setdefault(p, set()).add(m)
    has_1way = {p: bool(ms & _METHOD_1WAY) for p, ms in methods_of_p.items()}

    proc_source_pairs = frozenset(_read_n_col_rows(
        input_dir / "process__source.csv", ["process", "source"],
    ))
    sources_of_p: dict[str, int] = {}
    for p, _ in proc_source_pairs:
        sources_of_p[p] = sources_of_p.get(p, 0) + 1
    sinks_of_p: dict[str, int] = {}
    for p, _ in _read_n_col_rows(input_dir / "process__sink.csv", ["process", "sink"]):
        sinks_of_p[p] = sinks_of_p.get(p, 0) + 1

    rows: list[tuple[str, str, str]] = []
    for p, src, sink in triples:
        if not has_1way.get(p, False):
            continue
        if (p, src) not in proc_source_pairs:
            continue
        if sinks_of_p.get(p, 0) == 0 or sources_of_p.get(p, 0) >= 2:
            rows.append((p, src, sink))
    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "source":  [r[1] for r in rows],
            "sink":    [r[2] for r in rows],
        },
        schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8},
    )


def write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
            input_dir, solve_data_dir,
        ),
        solve_data_dir
        / "process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.csv",
    )


# ---- process__source__sink__ramp_method (mod L1205-1209) -------------------

def derive_process_source_sink_ramp_method(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink × ramp_method`` filtered by per-side membership."""
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    pnrm = _read_n_col_rows(
        input_dir / "process__node__ramp_method.csv",
        ["process", "node", "ramp_method"],
    )
    pnrm_set: set[tuple[str, str, str]] = set(pnrm)

    rows: list[tuple[str, str, str, str]] = []
    for p, src, sink in triples:
        for m in _RAMP_METHOD:
            if (p, src, m) in pnrm_set or (p, sink, m) in pnrm_set:
                rows.append((p, src, sink, m))
    return pl.DataFrame(
        {
            "process":     [r[0] for r in rows],
            "source":      [r[1] for r in rows],
            "sink":        [r[2] for r in rows],
            "ramp_method": [r[3] for r in rows],
        },
        schema={
            "process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
            "ramp_method": pl.Utf8,
        },
    )


def write_process_source_sink_ramp_method(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_process_source_sink_ramp_method(input_dir, solve_data_dir),
        solve_data_dir / "process__source__sink__ramp_method.csv",
    )


# ---- process_source_sink_coeff_zero (mod L1973) ---------------------------

def derive_process_source_sink_coeff_zero(
    solve_data_dir: Path,
) -> pl.DataFrame:
    """``process_source_sink`` filtered by zero flow coefficient on EITHER side."""
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    src_zero = frozenset(_read_n_col_rows(
        solve_data_dir / "process_source_coeff_zero.csv",
        ["process", "source"],
    ))
    sink_zero = frozenset(_read_n_col_rows(
        solve_data_dir / "process_sink_coeff_zero.csv",
        ["process", "sink"],
    ))
    rows = [
        (p, src, sink) for p, src, sink in triples
        if (p, src) in src_zero or (p, sink) in sink_zero
    ]
    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "source":  [r[1] for r in rows],
            "sink":    [r[2] for r in rows],
        },
        schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8},
    )


def write_process_source_sink_coeff_zero(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_process_source_sink_coeff_zero(solve_data_dir),
        solve_data_dir / "process_source_sink_coeff_zero.csv",
    )


# ---- process__source__sinkIsNode_* family (mod L1071, L1153-1158) ---------

def derive_process_source_sink_is_node_family(
    input_dir: Path, solve_data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Emit the 4-frame ``process__source__sinkIsNode*`` family.

    Returns (base, two_way_1var, not_two_way_1var, two_way_2var).
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    sinks = frozenset(_read_n_col_rows(
        input_dir / "process__sink.csv", ["process", "sink"],
    ))
    pm = _read_n_col_rows(input_dir / "process_method.csv", ["process", "method"])
    methods_of_p: dict[str, set[str]] = {}
    for p, m in pm:
        methods_of_p.setdefault(p, set()).add(m)
    has_2way_1var = {p: bool(ms & _METHOD_2WAY_1VAR) for p, ms in methods_of_p.items()}
    has_not_2way_1var = {p: bool(ms - _METHOD_2WAY_1VAR) for p, ms in methods_of_p.items()}
    has_2way_2var = {p: bool(ms & _METHOD_2WAY_2VAR) for p, ms in methods_of_p.items()}

    base_rows = [(p, src, sink) for p, src, sink in triples if (p, sink) in sinks]

    def _to_df(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "process": [r[0] for r in rows],
                "source":  [r[1] for r in rows],
                "sink":    [r[2] for r in rows],
            },
            schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8},
        )

    return (
        _to_df(base_rows),
        _to_df([r for r in base_rows if has_2way_1var.get(r[0], False)]),
        _to_df([r for r in base_rows if has_not_2way_1var.get(r[0], False)]),
        _to_df([r for r in base_rows if has_2way_2var.get(r[0], False)]),
    )


def write_process_source_sink_is_node_family(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    base, two1, not21, two2 = derive_process_source_sink_is_node_family(
        input_dir, solve_data_dir,
    )
    _write(base, solve_data_dir / "process__source__sinkIsNode.csv")
    _write(two1, solve_data_dir / "process__source__sinkIsNode_2way1var.csv")
    _write(not21, solve_data_dir / "process__source__sinkIsNode_not2way1var.csv")
    _write(two2, solve_data_dir / "process__source__sinkIsNode_2way2var.csv")


# ---- process_source_sink_{delayed,undelayed} (mod L1096-1097) --------------

def derive_process_source_sink_delayed_partition(
    solve_data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition ``process_source_sink`` by membership in ``process_delayed``."""
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    delayed_set = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "process_delayed.csv", ["process"]),
            ["process"],
        ).get_column("process").to_list()
    )
    delayed_rows = [r for r in triples if r[0] in delayed_set]
    undelayed_rows = [r for r in triples if r[0] not in delayed_set]

    def _to_df(rows: list[tuple[str, ...]]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "process": [r[0] for r in rows],
                "source":  [r[1] for r in rows],
                "sink":    [r[2] for r in rows],
            },
            schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8},
        )

    return _to_df(delayed_rows), _to_df(undelayed_rows)


def write_process_source_sink_delayed_partition(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    delayed, undelayed = derive_process_source_sink_delayed_partition(solve_data_dir)
    _write(delayed, solve_data_dir / "process_source_sink_delayed.csv")
    _write(undelayed, solve_data_dir / "process_source_sink_undelayed.csv")


# ===========================================================================
# entity_period_calc_params — self-contained subset
# ===========================================================================


def _read_value_lookup_3(path: Path) -> dict[tuple[str, str, str], float]:
    """Load a 4-col CSV (k1, k2, k3, value) as a dict keyed on the
    first 3 columns and floated on the 4th.  Silently skip rows whose
    value can't be parsed.
    """
    df = _read_csv(path, ["k1", "k2", "k3", "value"])
    out: dict[tuple[str, str, str], float] = {}
    for k1, k2, k3, v in df.iter_rows():
        if not k1 or not k2 or not k3:
            continue
        try:
            out[(k1, k2, k3)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def derive_p_process_source_sink(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``pProcess_source_sink``: prefer p_process_source, fall back to
    p_process_sink, then 0.  Domain = ``process__source__sink__param``.

    Returns a 5-col frame; value column is pre-stringified with
    ``repr(float(v))`` to preserve bit-exact precision parity with
    legacy code (see :mod:`._writer_calc_params` module docstring).
    """
    p_src = _read_value_lookup_3(input_dir / "p_process_source.csv")
    p_snk = _read_value_lookup_3(input_dir / "p_process_sink.csv")

    domain = _read_n_col_rows(
        solve_data_dir / "process__source__sink__param.csv",
        ["process", "source", "sink", "param"],
    )

    processes: list[str] = []
    sources: list[str] = []
    sinks: list[str] = []
    params: list[str] = []
    values: list[str] = []
    for p, src, snk, param in domain:
        if (p, src, param) in p_src:
            v = p_src[(p, src, param)]
        elif (p, snk, param) in p_snk:
            v = p_snk[(p, snk, param)]
        else:
            v = 0.0
        processes.append(p)
        sources.append(src)
        sinks.append(snk)
        params.append(param)
        values.append(repr(float(v)))
    return pl.DataFrame(
        {
            "process": processes, "source": sources, "sink": sinks,
            "param": params, "value": values,
        },
        schema={
            "process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
            "param": pl.Utf8, "value": pl.Utf8,
        },
    )


def write_pProcess_source_sink(input_dir: Path, solve_data_dir: Path) -> None:
    _write(
        derive_p_process_source_sink(input_dir, solve_data_dir),
        solve_data_dir / "pProcess_source_sink.csv",
    )


# ---------------------------------------------------------------------------
# process_source_sink_ramp_family (mod L1660-1688)
# ---------------------------------------------------------------------------

def _read_p_proc_side_lookup(path: Path) -> dict[tuple[str, str, str], float]:
    """Read p_process_source / p_process_sink: (process, side, param) → value."""
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    df = _read_csv(path, ["process", "side", "param", "value"])
    df = _drop_blank_rows(df, ["process", "side", "param"])
    for p, s, param, v in df.iter_rows():
        try:
            out[(p, s, param)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _compute_ramp_family(
    input_dir: Path, solve_data_dir: Path,
) -> dict[str, list[tuple[str, str, str]]]:
    """Emit the 5 ramp-family triple sets.

    Returns ``{filename → rows}``.  Rows preserve ``process_source_sink``
    order; legacy emits no dedup (input is already unique).
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )
    pnrm_rows = _read_n_col_rows(
        input_dir / "process__node__ramp_method.csv",
        ["process", "node", "ramp_method"],
    )
    pnrm: dict[tuple[str, str], set[str]] = {}
    for p, n, m in pnrm_rows:
        pnrm.setdefault((p, n), set()).add(m)

    p_proc_source = _read_p_proc_side_lookup(input_dir / "p_process_source.csv")
    p_proc_sink = _read_p_proc_side_lookup(input_dir / "p_process_sink.csv")

    def _has_method(p: str, n: str, methods: frozenset[str]) -> bool:
        return bool(pnrm.get((p, n), set()) & methods)

    rsu = [
        (p, src, sink) for p, src, sink in triples
        if _has_method(p, src, _RAMP_LIMIT_METHOD)
        and p_proc_source.get((p, src, "ramp_speed_up"), 0.0) > 0
    ]
    siu = [
        (p, src, sink) for p, src, sink in triples
        if _has_method(p, sink, _RAMP_LIMIT_METHOD)
        and p_proc_sink.get((p, sink, "ramp_speed_up"), 0.0) > 0
    ]
    rsd = [
        (p, src, sink) for p, src, sink in triples
        if _has_method(p, src, _RAMP_LIMIT_METHOD)
        and p_proc_source.get((p, src, "ramp_speed_down"), 0.0) > 0
    ]
    sid = [
        (p, src, sink) for p, src, sink in triples
        if _has_method(p, sink, _RAMP_LIMIT_METHOD)
        and p_proc_sink.get((p, sink, "ramp_speed_down"), 0.0) > 0
    ]
    cost = [
        (p, src, sink) for p, src, sink in triples
        if _has_method(p, src, _RAMP_COST_METHOD)
        or _has_method(p, sink, _RAMP_COST_METHOD)
    ]
    return {
        "process_source_sink_ramp_limit_source_up.csv": rsu,
        "process_source_sink_ramp_limit_sink_up.csv":   siu,
        "process_source_sink_ramp_limit_source_down.csv": rsd,
        "process_source_sink_ramp_limit_sink_down.csv":   sid,
        "process_source_sink_ramp_cost.csv":               cost,
    }


def _triples_frame(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "source":  [r[1] for r in rows],
            "sink":    [r[2] for r in rows],
        },
        schema={"process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8},
    )


def write_process_source_sink_ramp_family(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """5 per-arc ramp sets gated by method and (for limit variants) speed > 0."""
    by_file = _compute_ramp_family(input_dir, solve_data_dir)
    for fname, rows in by_file.items():
        _write(_triples_frame(rows), solve_data_dir / fname)


# ---------------------------------------------------------------------------
# process_source_sink_ramp_unions — 5-way union of ramp_*.csv files
# ---------------------------------------------------------------------------

def write_process_source_sink_ramp_unions(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Order-preserving union of the 5 ramp-family triple sets."""
    ramp_files = (
        "process_source_sink_ramp_limit_source_up.csv",
        "process_source_sink_ramp_limit_sink_up.csv",
        "process_source_sink_ramp_limit_source_down.csv",
        "process_source_sink_ramp_limit_sink_down.csv",
        "process_source_sink_ramp_cost.csv",
    )
    seen: dict[tuple[str, str, str], None] = {}
    for fname in ramp_files:
        for r in _read_n_col_rows(
            solve_data_dir / fname, ["process", "source", "sink"],
        ):
            seen.setdefault(r, None)
    _write(
        _triples_frame(list(seen.keys())),
        solve_data_dir / "process_source_sink_ramp.csv",
    )


# ---------------------------------------------------------------------------
# group_commodity_node_period_co2_total (mod L1981)
# ---------------------------------------------------------------------------

def write_group_commodity_node_period_co2_total(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Join group_co2_max_total × group__node × commodity__node × p_commodity.

    Emit rows ``(g, c, n)`` where:
      * (g, n) ∈ group__node and g ∈ group_co2_max_total
      * (c, n) ∈ commodity__node
      * ``p_commodity[c, 'co2_content'] != 0``
    """
    cn = _read_n_col_rows(
        input_dir / "commodity__node.csv", ["commodity", "node"],
    )
    gn = _read_n_col_rows(
        input_dir / "group__node.csv", ["group", "node"],
    )
    g_with_n: dict[str, set[str]] = {}
    for g, n in gn:
        g_with_n.setdefault(g, set()).add(n)

    p_commodity: dict[tuple[str, str], float] = {}
    pc_path = input_dir / "p_commodity.csv"
    if pc_path.exists() and pc_path.stat().st_size > 0:
        pc_df = _read_csv(pc_path, ["commodity", "param", "value"])
        pc_df = _drop_blank_rows(pc_df, ["commodity", "param"])
        for c, param, v in pc_df.iter_rows():
            try:
                p_commodity[(c, param)] = float(v)
            except (TypeError, ValueError):
                continue
    co2_max_total = frozenset(
        _read_n_col_rows(
            solve_data_dir / "group_co2_max_total.csv", ["group"],
        )
    )
    # NB: frozenset entries are 1-tuples — flatten.
    co2_max_total = frozenset(t[0] for t in co2_max_total)

    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for g in co2_max_total:
        nodes = g_with_n.get(g, set())
        for c, n in cn:
            if n in nodes and p_commodity.get((c, "co2_content"), 0.0) != 0.0:
                key = (g, c, n)
                if key not in seen:
                    seen.add(key)
                    rows.append(key)
    _write(
        pl.DataFrame(
            {
                "group":     [r[0] for r in rows],
                "commodity": [r[1] for r in rows],
                "node":      [r[2] for r in rows],
            },
            schema={"group": pl.Utf8, "commodity": pl.Utf8, "node": pl.Utf8},
        ),
        solve_data_dir / "group_commodity_node_period_co2_total.csv",
    )


# ===========================================================================
# Phase 1 follow-up 4 — param_in_use family + dispatch-fully-inside set.
# ===========================================================================


# Per-class param taxonomies — mirror flextool_base.dat.  Pinned here to
# avoid a transitive import from the legacy preprocessing tree (matches
# the pattern used for ramp / method constants above).
#
# Update both sites in lockstep if base.dat changes; the parity tests
# catch drift.

# flextool_base.dat:144-152 — processPeriodParam family.
_PROCESS_PERIOD_PARAM: frozenset[str] = frozenset((
    "fixed_cost", "other_operational_cost", "lifetime", "existing",
    "discount_rate", "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period", "startup_cost",
))
_PROCESS_PERIOD_PARAM_REQUIRED: frozenset[str] = frozenset((
    "fixed_cost", "other_operational_cost", "lifetime", "existing",
))
_PROCESS_PERIOD_PARAM_INVEST: frozenset[str] = frozenset((
    "discount_rate", "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period",
))

# flextool_base.dat:153-154 — processTimeParam family.
_PROCESS_TIME_PARAM_REQUIRED: frozenset[str] = frozenset((
    "efficiency", "other_operational_cost", "availability",
))

# flextool_base.dat:158-161 — sourceSinkTime/PeriodParam family
# (period == time taxonomy in this version of base.dat).
_SOURCE_SINK_TIME_PARAM: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load", "other_operational_cost",
))
_SOURCE_SINK_TIME_PARAM_REQUIRED: frozenset[str] = frozenset((
    "efficiency", "other_operational_cost",
))
_SOURCE_SINK_PERIOD_PARAM: frozenset[str] = _SOURCE_SINK_TIME_PARAM
_SOURCE_SINK_PERIOD_PARAM_REQUIRED: frozenset[str] = _SOURCE_SINK_TIME_PARAM_REQUIRED

# flextool_base.dat:168-177 — nodePeriodParam family.
_NODE_PERIOD_PARAM: frozenset[str] = frozenset((
    "annual_flow", "peak_inflow", "fixed_cost", "discount_rate",
    "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period", "lifetime",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period",
    "virtual_unitsize",
    "storage_state_reference_price", "existing", "penalty_up", "penalty_down",
))
_NODE_PERIOD_PARAM_REQUIRED: frozenset[str] = frozenset((
    "annual_flow", "peak_inflow", "fixed_cost", "lifetime",
    "storage_state_reference_price", "existing",
    "penalty_up", "penalty_down",
))
_NODE_PERIOD_PARAM_INVEST: frozenset[str] = frozenset((
    "discount_rate", "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period",
    "virtual_unitsize",
))


# ---- write_param_in_use_sets (mod L1247 / L1369) --------------------------
#
# Emits seven param-in-use CSVs.  Legacy implementation iterates a small
# python dictionary keyed by (entity, param) and dedupes with
# ``dict.fromkeys``; native polars wouldn't be faster for this shape —
# the inputs are tiny enum cross-products.  We mirror the legacy loops
# directly inside ``derive_*`` for code-shape parity.

def _read_singles_list(path: Path) -> list[str]:
    """Read column 0 of a small CSV into a list (preserves CSV order)."""
    return [
        r[0] for r in _read_n_col_rows(path, ["c0"])
    ]


def _derive_node_period_param_in_use(
    nodes: list[str], invest_set: frozenset[str], divest_set: frozenset[str],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for n in nodes:
        is_invest = n in invest_set or n in divest_set
        for param in _NODE_PERIOD_PARAM:
            if param in _NODE_PERIOD_PARAM_REQUIRED:
                rows.append((n, param))
            elif is_invest and param in _NODE_PERIOD_PARAM_INVEST:
                rows.append((n, param))
    return list(dict.fromkeys(rows))


def _derive_process_period_param_in_use(
    processes: list[str], invest_set: frozenset[str],
    divest_set: frozenset[str], process_online: frozenset[str],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for p in processes:
        is_invest = p in invest_set or p in divest_set
        is_online = p in process_online
        for param in _PROCESS_PERIOD_PARAM:
            if param in _PROCESS_PERIOD_PARAM_REQUIRED:
                rows.append((p, param))
            elif is_invest and param in _PROCESS_PERIOD_PARAM_INVEST:
                rows.append((p, param))
            elif is_online and param == "startup_cost":
                rows.append((p, param))
    return list(dict.fromkeys(rows))


def _derive_process_time_param_in_use(
    processes: list[str], p_with_min_load: frozenset[str],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for p in processes:
        for param in _PROCESS_TIME_PARAM:
            if param in _PROCESS_TIME_PARAM_REQUIRED:
                rows.append((p, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows.append((p, param))
    return list(dict.fromkeys(rows))


def _derive_pss_param_in_use(
    pairs: list[tuple[str, str]], p_with_min_load: frozenset[str],
    enum: frozenset[str], required: frozenset[str],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for p, side in pairs:
        for param in enum:
            if param in required:
                rows.append((p, side, param))
            elif (p in p_with_min_load
                  and param in ("min_load", "efficiency_at_min_load")):
                rows.append((p, side, param))
    return list(dict.fromkeys(rows))


def _rows_to_frame_2(rows: list[tuple[str, str]],
                     cols: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {cols[0]: [r[0] for r in rows], cols[1]: [r[1] for r in rows]},
        schema={cols[0]: pl.Utf8, cols[1]: pl.Utf8},
    )


def _rows_to_frame_3(rows: list[tuple[str, str, str]],
                     cols: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            cols[0]: [r[0] for r in rows],
            cols[1]: [r[1] for r in rows],
            cols[2]: [r[2] for r in rows],
        },
        schema={c: pl.Utf8 for c in cols},
    )


def write_param_in_use_sets(input_dir: Path, solve_data_dir: Path) -> None:
    """Emit the seven ``*_in_use`` CSVs that filter (entity, param) by
    Required/Invest enum membership.

    Outputs:
      * ``node__PeriodParam_in_use.csv``
      * ``process__PeriodParam_in_use.csv``
      * ``process_TimeParam_in_use.csv``
      * ``process_source_sourceSinkTimeParam_in_use.csv``
      * ``process_sink_sourceSinkTimeParam_in_use.csv``
      * ``process_source_sourceSinkPeriodParam_in_use.csv``
      * ``process_sink_sourceSinkPeriodParam_in_use.csv``
    """
    nodes = _read_singles_list(input_dir / "node.csv")
    processes = _read_singles_list(input_dir / "process.csv")
    invest_set = frozenset(
        _read_singles_list(solve_data_dir / "entityInvest.csv")
    )
    divest_set = frozenset(
        _read_singles_list(solve_data_dir / "entityDivest.csv")
    )
    ctm = _read_n_col_rows(
        solve_data_dir / "process__ct_method.csv", ["process", "method"],
    )
    p_with_min_load = frozenset(
        p for p, m in ctm if m == "min_load_efficiency"
    )
    process_online = frozenset(
        _read_singles_list(solve_data_dir / "process_online.csv")
    )
    sources = [
        (p, src) for p, src in _read_n_col_rows(
            input_dir / "process__source.csv", ["process", "source"],
        )
    ]
    sinks = [
        (p, snk) for p, snk in _read_n_col_rows(
            input_dir / "process__sink.csv", ["process", "sink"],
        )
    ]

    # node__PeriodParam_in_use
    _write(
        _rows_to_frame_2(
            _derive_node_period_param_in_use(nodes, invest_set, divest_set),
            ("node", "param"),
        ),
        solve_data_dir / "node__PeriodParam_in_use.csv",
    )
    # process__PeriodParam_in_use
    _write(
        _rows_to_frame_2(
            _derive_process_period_param_in_use(
                processes, invest_set, divest_set, process_online,
            ),
            ("process", "param"),
        ),
        solve_data_dir / "process__PeriodParam_in_use.csv",
    )
    # process_TimeParam_in_use
    _write(
        _rows_to_frame_2(
            _derive_process_time_param_in_use(processes, p_with_min_load),
            ("process", "param"),
        ),
        solve_data_dir / "process_TimeParam_in_use.csv",
    )
    # process_source / process_sink _sourceSinkTimeParam_in_use
    _write(
        _rows_to_frame_3(
            _derive_pss_param_in_use(
                sources, p_with_min_load,
                _SOURCE_SINK_TIME_PARAM, _SOURCE_SINK_TIME_PARAM_REQUIRED,
            ),
            ("process", "source", "param"),
        ),
        solve_data_dir / "process_source_sourceSinkTimeParam_in_use.csv",
    )
    _write(
        _rows_to_frame_3(
            _derive_pss_param_in_use(
                sinks, p_with_min_load,
                _SOURCE_SINK_TIME_PARAM, _SOURCE_SINK_TIME_PARAM_REQUIRED,
            ),
            ("process", "sink", "param"),
        ),
        solve_data_dir / "process_sink_sourceSinkTimeParam_in_use.csv",
    )
    # process_source / process_sink _sourceSinkPeriodParam_in_use
    _write(
        _rows_to_frame_3(
            _derive_pss_param_in_use(
                sources, p_with_min_load,
                _SOURCE_SINK_PERIOD_PARAM, _SOURCE_SINK_PERIOD_PARAM_REQUIRED,
            ),
            ("process", "source", "param"),
        ),
        solve_data_dir / "process_source_sourceSinkPeriodParam_in_use.csv",
    )
    _write(
        _rows_to_frame_3(
            _derive_pss_param_in_use(
                sinks, p_with_min_load,
                _SOURCE_SINK_PERIOD_PARAM, _SOURCE_SINK_PERIOD_PARAM_REQUIRED,
            ),
            ("process", "sink", "param"),
        ),
        solve_data_dir / "process_sink_sourceSinkPeriodParam_in_use.csv",
    )


# ---- write_node_group_dispatch_process_fully_inside (mod L1789-1794) ------

def derive_node_group_dispatch_process_fully_inside(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """For each ``g ∈ nodeGroupDispatch`` × ``p ∈ process``, include if
    BOTH some source and some sink of ``p`` are in ``group__node[g]``
    AND ``p`` is not a self-loop (no ``(p, n, n)`` in ``process_source_sink``).
    """
    ngd = _read_singles_list(input_dir / "nodeGroupDispatch.csv")
    procs = _read_singles_list(input_dir / "process.csv")
    process_source_pairs = _read_n_col_rows(
        input_dir / "process__source.csv", ["process", "source"],
    )
    process_sink_pairs = _read_n_col_rows(
        input_dir / "process__sink.csv", ["process", "sink"],
    )
    gn = _read_n_col_rows(input_dir / "group__node.csv", ["group", "node"])
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
    )

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
    return _rows_to_frame_2(rows, ("group", "process"))


def write_node_group_dispatch_process_fully_inside(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    _write(
        derive_node_group_dispatch_process_fully_inside(input_dir, solve_data_dir),
        solve_data_dir / "nodeGroupDispatch__process_fully_inside.csv",
    )
