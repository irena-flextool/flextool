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

def derive_process_source_sink_ramp_unions(
    solve_data_dir: Path,
) -> pl.DataFrame:
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
    return _triples_frame(list(seen.keys()))


def write_process_source_sink_ramp_unions(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Order-preserving union of the 5 ramp-family triple sets."""
    _write(
        derive_process_source_sink_ramp_unions(solve_data_dir),
        solve_data_dir / "process_source_sink_ramp.csv",
    )


# ---------------------------------------------------------------------------
# group_commodity_node_period_co2_total (mod L1981)
# ---------------------------------------------------------------------------

def derive_group_commodity_node_period_co2_total(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``group_commodity_node_period_co2_total`` 3-col frame; see writer."""
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
    return pl.DataFrame(
        {
            "group":     [r[0] for r in rows],
            "commodity": [r[1] for r in rows],
            "node":      [r[2] for r in rows],
        },
        schema={"group": pl.Utf8, "commodity": pl.Utf8, "node": pl.Utf8},
    )


def write_group_commodity_node_period_co2_total(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Join group_co2_max_total × group__node × commodity__node × p_commodity.

    Emit rows ``(g, c, n)`` where:
      * (g, n) ∈ group__node and g ∈ group_co2_max_total
      * (c, n) ∈ commodity__node
      * ``p_commodity[c, 'co2_content'] != 0``
    """
    _write(
        derive_group_commodity_node_period_co2_total(
            input_dir, solve_data_dir,
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


# ===========================================================================
# Phase 1 follow-up 5 — small_set_derivations + arc-union small writers
# ===========================================================================

# Helpers used by the small writers below.  These are byte-for-byte parity
# with the legacy ``_read_singles`` / ``_read_pairs`` / ``_write_csv``
# helpers in ``process_arc_unions``; we keep them local to this module
# rather than reaching into the legacy module so the native port has no
# transitive import from preprocessing.


def _read_singles_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    import csv
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _read_pairs_csv(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    import csv
    out: list[tuple[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _read_n_col_csv(path: Path, n: int) -> list[tuple[str, ...]]:
    if not path.exists():
        return []
    import csv
    out: list[tuple[str, ...]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= n and all(row[i] for i in range(n)):
                out.append(tuple(row[:n]))
    return out


def _write_csv_rows(path: Path, header: tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n"
                    + "".join(",".join(r) + "\n" for r in rows))


def _rows_to_frame(rows, header: tuple[str, ...]) -> pl.DataFrame:
    """Build an all-Utf8 ``pl.DataFrame`` from rows + a header tuple.

    Mirrors the byte shape of :func:`_write_csv_rows` exactly: header
    becomes column names; each tuple element a string cell.  Uses
    column-of-tuples projection so empty-row frames still carry the
    requested schema.
    """
    n = len(header)
    cols: list[list[str]] = [[] for _ in range(n)]
    for r in rows:
        for i in range(n):
            cols[i].append(r[i])
    return pl.DataFrame(
        {header[i]: cols[i] for i in range(n)},
        schema={h: pl.Utf8 for h in header},
    )


# ---- write_small_set_derivations (mod L999, L1061, L1132, L1174, L1222-3) --

def derive_ed_history_realized(solve_data_dir: Path) -> pl.DataFrame:
    """Order-preserving union of
    ``p_entity_period_existing_capacity`` + ``ed_history_realized_first``,
    projected to (entity, period).
    """
    ed_read = _read_pairs_csv(
        solve_data_dir / "p_entity_period_existing_capacity.csv"
    )
    ed_first = _read_pairs_csv(solve_data_dir / "ed_history_realized_first.csv")
    seen_ed: dict[tuple[str, str], None] = {}
    for r in ed_read:
        seen_ed.setdefault(r, None)
    for r in ed_first:
        seen_ed.setdefault(r, None)
    return _rows_to_frame(list(seen_ed.keys()), ("entity", "period"))


def derive_process_source_sink_profile_method(
    solve_data_dir: Path,
) -> pl.DataFrame:
    """4-way union of the *profile_method* sub-CSVs (5-col frame)."""
    seen_pf: dict[tuple[str, ...], None] = {}
    for fname in (
        "process__profileProcess__toSink__profile__profile_method.csv",
        "process__source__toProfileProcess__profile__profile_method.csv",
        "process__source__sink__profile__profile_method_connection.csv",
        "process__source__sink__profile__profile_method_direct.csv",
    ):
        for r in _read_n_col_csv(solve_data_dir / fname, 5):
            seen_pf.setdefault(r, None)
    return _rows_to_frame(
        list(seen_pf.keys()),
        ("process", "source", "sink", "profile", "profile_method"),
    )


def derive_process_sinkIsNode_2way1var(
    solve_data_dir: Path,
) -> pl.DataFrame:
    """Projection of column 0 of
    ``process__source__sinkIsNode_2way1var.csv``."""
    triples = _read_n_col_csv(
        solve_data_dir / "process__source__sinkIsNode_2way1var.csv", 3
    )
    seen_p: dict[str, None] = {}
    for p, _, _ in triples:
        seen_p.setdefault(p, None)
    return _rows_to_frame([(p,) for p in seen_p.keys()], ("process",))


def derive_nodeSelfDischarge(solve_data_dir: Path) -> pl.DataFrame:
    """Subset of nodeState whose ``pdtNode[n, 'self_discharge_loss', d, t]``
    is non-zero for at least one (d, t).
    """
    import csv
    nodeState = frozenset(_read_singles_csv(solve_data_dir / "nodeState.csv"))
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
    rows = [
        (n,)
        for n in _read_singles_csv(solve_data_dir / "nodeState.csv")
        if n in nodes_with_selfdischarge
    ]
    return _rows_to_frame(rows, ("node",))


def _scan_pd_startup(solve_data_dir: Path) -> set[tuple[str, str]]:
    """(process, period) pairs where ``pdProcess[p, 'startup_cost', d]`` != 0."""
    import csv
    pd_startup: set[tuple[str, str]] = set()
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
    return pd_startup


def _derive_pdt_online(
    solve_data_dir: Path, processes_csv: str,
) -> pl.DataFrame:
    pd_startup = _scan_pd_startup(solve_data_dir)
    dt_pairs = _read_n_col_csv(solve_data_dir / "steps_in_use.csv", 2)
    procs = _read_singles_csv(solve_data_dir / processes_csv)
    rows: list[tuple[str, str, str]] = []
    for p in procs:
        for d, t in dt_pairs:
            if (p, d) in pd_startup:
                rows.append((p, d, t))
    return _rows_to_frame(rows, ("process", "period", "time"))


def derive_pdt_online_linear(solve_data_dir: Path) -> pl.DataFrame:
    """``pdt_online_linear`` — process_online_linear × dt gated by startup_cost!=0."""
    return _derive_pdt_online(solve_data_dir, "process_online_linear.csv")


def derive_pdt_online_integer(solve_data_dir: Path) -> pl.DataFrame:
    """``pdt_online_integer`` — process_online_integer × dt gated by startup_cost!=0."""
    return _derive_pdt_online(solve_data_dir, "process_online_integer.csv")


def write_small_set_derivations(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L999, L1061, L1132, L1174, L1222-3 — 6 small derived sets.

    Emits ``ed_history_realized``,
    ``process__source__sink__profile__profile_method``,
    ``process_sinkIsNode_2way1var``, ``nodeSelfDischarge``,
    ``pdt_online_linear``, ``pdt_online_integer``.

    See the legacy docstring for the full set of math-prog derivations.
    """
    _write(
        derive_ed_history_realized(solve_data_dir),
        solve_data_dir / "ed_history_realized.csv",
    )
    _write(
        derive_process_source_sink_profile_method(solve_data_dir),
        solve_data_dir / "process__source__sink__profile__profile_method.csv",
    )
    _write(
        derive_process_sinkIsNode_2way1var(solve_data_dir),
        solve_data_dir / "process_sinkIsNode_2way1var.csv",
    )
    _write(
        derive_nodeSelfDischarge(solve_data_dir),
        solve_data_dir / "nodeSelfDischarge.csv",
    )
    _write(
        derive_pdt_online_linear(solve_data_dir),
        solve_data_dir / "pdt_online_linear.csv",
    )
    _write(
        derive_pdt_online_integer(solve_data_dir),
        solve_data_dir / "pdt_online_integer.csv",
    )


# ---- write_process_source_sink_param_with_time (mod L1187-1195) ------------

# preprocessing/_param_taxonomy.py — SOURCE_SINK_TIME_PARAM (smaller than
# the dat-level enum, which is the union of all sourceSink params).  We
# mirror the legacy taxonomy exactly so the rows + per-row param-iter
# order match within a single process.
_SOURCE_SINK_TIME_PARAM_WITH_TIME: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load",
    "other_operational_cost",
))


def derive_process_source_sink_param_with_time(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``process__source__sink__param_t`` 4-col frame.

    See :func:`write_process_source_sink_param_with_time` for semantics.
    """
    import csv
    triples = _read_n_col_csv(solve_data_dir / "process_source_sink.csv", 3)

    def _read_3(path: Path) -> set[tuple[str, str, str]]:
        out: set[tuple[str, str, str]] = set()
        if not path.exists():
            return out
        with path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] and row[1] and row[2]:
                    out.add((row[0], row[1], row[2]))
        return out

    def _read_2(path: Path) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        if not path.exists():
            return out
        with path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    out.add((row[0], row[1]))
        return out

    src_param = _read_3(input_dir / "p_process_source.csv")
    sink_param = _read_3(input_dir / "p_process_sink.csv")
    proc_param = _read_2(input_dir / "p_process.csv")
    proc_conn = frozenset(_read_singles_csv(input_dir / "process_connection.csv"))
    src_param_t = _read_3(solve_data_dir / "pt_process_source.csv")
    sink_param_t = _read_3(solve_data_dir / "pt_process_sink.csv")
    proc_param_t = _read_2(solve_data_dir / "pt_process.csv")

    rows: list[tuple[str, str, str, str]] = []
    for p, src, sink in triples:
        for param in _SOURCE_SINK_TIME_PARAM_WITH_TIME:
            if ((p, src, param) in src_param
                    or (p, src, param) in src_param_t
                    or (p, sink, param) in sink_param
                    or (p, sink, param) in sink_param_t
                    or ((p, param) in proc_param and p in proc_conn)
                    or ((p, param) in proc_param_t and p in proc_conn)):
                rows.append((p, src, sink, param))
    return _rows_to_frame(rows, ("process", "source", "sink", "param"))


def write_process_source_sink_param_with_time(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1187-1195 — process_source_sink × SOURCE_SINK_TIME_PARAM
    gated by static or time-variant param membership on either side, or via
    process_connection.

    Distinct from the sibling ``write_process_source_sink_param`` (3-col
    set, no _t variants).  This one is the double-underscore-named set
    ``process__source__sink__param_t``.
    """
    _write(
        derive_process_source_sink_param_with_time(input_dir, solve_data_dir),
        solve_data_dir / "process__source__sink__param_t.csv",
    )


# ---- write_gdt_instant_flow_sets (mod L1131-1132) -------------------------

def _scan_gdt_instant_flow_rows(
    solve_data_dir: Path,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """One scan over pdtGroup.csv, splitting max/min_instant_flow rows."""
    import csv
    max_rows: list[tuple[str, str, str]] = []
    min_rows: list[tuple[str, str, str]] = []
    pdtg_path = solve_data_dir / "pdtGroup.csv"
    if pdtg_path.exists():
        with pdtg_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 5 and r[0] and r[2] and r[3]:
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
    return max_rows, min_rows


def derive_gdt_max_instant_flow(solve_data_dir: Path) -> pl.DataFrame:
    """``gdt_maxInstantFlow`` — pdtGroup rows with param=max_instant_flow."""
    max_rows, _ = _scan_gdt_instant_flow_rows(solve_data_dir)
    return _rows_to_frame(max_rows, ("group", "period", "time"))


def derive_gdt_min_instant_flow(solve_data_dir: Path) -> pl.DataFrame:
    """``gdt_minInstantFlow`` — pdtGroup rows with param=min_instant_flow."""
    _, min_rows = _scan_gdt_instant_flow_rows(solve_data_dir)
    return _rows_to_frame(min_rows, ("group", "period", "time"))


def write_gdt_instant_flow_sets(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1131-1132 — gdt_maxInstantFlow + gdt_minInstantFlow.

    Each row is included iff the corresponding ``pdtGroup[g, P, d, t]``
    value is non-zero.  Both frames come from one scan of pdtGroup.csv.
    """
    max_rows, min_rows = _scan_gdt_instant_flow_rows(solve_data_dir)
    _write(
        _rows_to_frame(max_rows, ("group", "period", "time")),
        solve_data_dir / "gdt_maxInstantFlow.csv",
    )
    _write(
        _rows_to_frame(min_rows, ("group", "period", "time")),
        solve_data_dir / "gdt_minInstantFlow.csv",
    )


# ---- write_p_process_delay_weight (mod L1096-1099) ------------------------

def derive_p_process_delay_weight(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``p_process_delay_weight`` 3-col frame; see writer docstring."""
    import csv
    delayed_duration = _read_pairs_csv(
        solve_data_dir / "process_delayed__duration.csv"
    )
    delay_single = frozenset(
        _read_pairs_csv(input_dir / "process_delay_single.csv")
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
    return _rows_to_frame(rows, ("process", "delay_duration", "value"))


def write_p_process_delay_weight(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1096-1099 — ``p_process_delay_weight``.

    For each (p, td) in ``process_delayed__duration``: 1 if
    ``(p, td) in process_delay_single`` else ``p_process_delay_weighted``
    (default 0).
    """
    _write(
        derive_p_process_delay_weight(input_dir, solve_data_dir),
        solve_data_dir / "p_process_delay_weight.csv",
    )


# ---- write_gcndt_co2_price (mod L1542-1548) -------------------------------

def derive_gcndt_co2_price(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``gcndt_co2_price`` 5-col frame; see writer docstring."""
    import csv
    g_co2_price = frozenset(
        _read_singles_csv(solve_data_dir / "group_co2_price.csv")
    )
    cn = _read_pairs_csv(input_dir / "commodity__node.csv")

    gn_acc: dict[str, set[str]] = {}
    for g, n in _read_pairs_csv(input_dir / "group__node.csv"):
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

    dt_pairs = _read_n_col_csv(solve_data_dir / "steps_in_use.csv", 2)

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
    return _rows_to_frame(
        rows, ("group", "commodity", "node", "period", "time"),
    )


def write_gcndt_co2_price(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1542-1548 — gcndt_co2_price 5-tuple set."""
    _write(
        derive_gcndt_co2_price(input_dir, solve_data_dir),
        solve_data_dir / "gcndt_co2_price.csv",
    )


# ---- write_group_commodity_node_period_co2_period (mod L1550-1555) --------

def derive_group_commodity_node_period_co2_period(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``group_commodity_node_period_co2_period`` 4-col frame.

    See :func:`write_group_commodity_node_period_co2_period`.
    """
    import csv
    g_co2_max_period = frozenset(
        _read_singles_csv(solve_data_dir / "group_co2_max_period.csv")
    )
    cn = _read_pairs_csv(input_dir / "commodity__node.csv")

    gn_acc: dict[str, set[str]] = {}
    for g, n in _read_pairs_csv(input_dir / "group__node.csv"):
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

    period_in_use = _read_singles_csv(solve_data_dir / "period_in_use_set.csv")

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
    return _rows_to_frame(rows, ("group", "commodity", "node", "period"))


def write_group_commodity_node_period_co2_period(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1550-1555 — group_commodity_node_period_co2_period."""
    _write(
        derive_group_commodity_node_period_co2_period(input_dir, solve_data_dir),
        solve_data_dir / "group_commodity_node_period_co2_period.csv",
    )


# ---- write_peedt (mod L1084) ----------------------------------------------

def derive_peedt(solve_data_dir: Path) -> pl.DataFrame:
    """``peedt = process_source_sink × steps_in_use`` (5-col frame).

    Hot-path for full-year fixtures — up to ~280k rows.
    """
    triples = _read_n_col_csv(solve_data_dir / "process_source_sink.csv", 3)
    dt_pairs = _read_n_col_csv(solve_data_dir / "steps_in_use.csv", 2)
    procs: list[str] = []
    srcs: list[str] = []
    snks: list[str] = []
    ds: list[str] = []
    ts: list[str] = []
    for p, src, snk in triples:
        for d, t in dt_pairs:
            procs.append(p)
            srcs.append(src)
            snks.append(snk)
            ds.append(d)
            ts.append(t)
    return pl.DataFrame(
        {
            "process": procs,
            "source":  srcs,
            "sink":    snks,
            "period":  ds,
            "time":    ts,
        },
        schema={
            "process": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8,
            "period": pl.Utf8, "time": pl.Utf8,
        },
    )


def write_peedt(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1084 — peedt = process_source_sink × dt.

    280k-row hot path for full-year fixtures.
    """
    _write(derive_peedt(solve_data_dir), solve_data_dir / "peedt.csv")


# ===========================================================================
# Phase 1 follow-up 6 — flow-bound + state-slack + storage reference price
# + 12-CSV nodeGroupDispatch dispatch set family.
#
# All five writers in this section emit either a parameter table (long-form
# ``(keys..., value)`` with ``repr(float)`` precision parity) or a set of
# tuples; semantics mirror flextool.mod L1596-L1803 exactly.  Each native
# implementation reads the same input/solve_data CSVs as its legacy peer
# and writes byte-identical output (CSV row order + float formatting).
# ===========================================================================


# ---- write_p_flow_min (mod L1680-1684) ------------------------------------


def derive_p_flow_min(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``p_flow_min`` 6-col frame; see writer docstring."""
    import csv
    sinkIsNode = frozenset(_read_n_col_csv(
        solve_data_dir / "process__source__sinkIsNode_2way1var.csv", 3
    ))
    cols = ("process", "source", "sink", "period", "time", "value")
    if not sinkIsNode:
        return _rows_to_frame([], cols)

    dcm: dict[tuple[str, str], float] = {}
    pdcm_path = solve_data_dir / "p_entity_dispatch_capacity_max.csv"
    if pdcm_path.exists():
        with pdcm_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        dcm[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue
    unitsize: dict[str, float] = {}
    pus_path = solve_data_dir / "p_entity_unitsize.csv"
    if pus_path.exists():
        with pus_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0]:
                    try:
                        unitsize[r[0]] = float(r[1])
                    except ValueError:
                        continue

    peedt = _read_n_col_csv(solve_data_dir / "peedt.csv", 5)
    rows: list[tuple[str, ...]] = []
    for p, src, sink, d, t in peedt:
        if (p, src, sink) not in sinkIsNode:
            continue
        us = unitsize.get(p, 1.0)
        if us == 0.0:
            continue
        v = -dcm.get((p, d), 0.0) / us
        rows.append((p, src, sink, d, t, repr(v)))
    return _rows_to_frame(rows, cols)


def write_p_flow_min(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1680-1684 — ``p_flow_min{(p,source,sink,d,t) in peedt}``.

    Emits only the non-zero rows.  The value is
    ``-(p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p])``
    when ``(p, source, sink) in process__source__sinkIsNode_2way1var``,
    else 0 (skipped — mod's bare-decl provides ``default 0``).
    """
    _write(
        derive_p_flow_min(input_dir, solve_data_dir),
        solve_data_dir / "p_flow_min.csv",
    )


# ---- write_p_flow_max (mod L1661-1677) ------------------------------------


def derive_p_flow_max(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``p_flow_max`` 6-col frame; see :func:`write_p_flow_max`."""
    import csv
    coeff_zero = frozenset(_read_n_col_csv(
        solve_data_dir / "process_source_sink_coeff_zero.csv", 3
    ))
    has_indirect = frozenset(
        p for p, _m in _read_pairs_csv(
            solve_data_dir / "process__method_indirect.csv"
        )
    )
    process_source = frozenset(_read_pairs_csv(input_dir / "process__source.csv"))
    process_sink = frozenset(_read_pairs_csv(input_dir / "process__sink.csv"))
    has_min_load = frozenset(
        p for p, m in _read_pairs_csv(solve_data_dir / "process__ct_method.csv")
        if m == "min_load_efficiency"
    )

    dcm: dict[tuple[str, str], float] = {}
    pdcm_path = solve_data_dir / "p_entity_dispatch_capacity_max.csv"
    if pdcm_path.exists():
        with pdcm_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        dcm[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue
    unitsize: dict[str, float] = {}
    pus_path = solve_data_dir / "p_entity_unitsize.csv"
    if pus_path.exists():
        with pus_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0]:
                    try:
                        unitsize[r[0]] = float(r[1])
                    except ValueError:
                        continue

    slope: dict[tuple[str, str, str], float] = {}
    section: dict[tuple[str, str, str], float] = {}
    for fname, target in (
        ("pdtProcess_slope.csv", slope),
        ("pdtProcess_section.csv", section),
    ):
        path = solve_data_dir / fname
        if path.exists():
            with path.open() as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for r in reader:
                    if len(r) >= 4 and r[0] and r[1] and r[2]:
                        try:
                            target[(r[0], r[1], r[2])] = float(r[3])
                        except ValueError:
                            continue

    src_max_coef: dict[tuple[str, str], float] = {}
    pms_path = input_dir / "p_process_source_max_capacity_coefficient.csv"
    if pms_path.exists():
        with pms_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        src_max_coef[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue
    sink_max_coef: dict[tuple[str, str], float] = {}
    pmk_path = input_dir / "p_process_sink_max_capacity_coefficient.csv"
    if pmk_path.exists():
        with pmk_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        sink_max_coef[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue

    # p_unconstrained_flow_cap = max over models of
    # p_max_flow_for_unconstrained_variables[m]; default 1e6 if absent.
    p_uflow = 1_000_000.0
    pmfu_path = input_dir / "p_max_flow_for_unconstrained_variables.csv"
    if pmfu_path.exists():
        max_v: float | None = None
        with pmfu_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0]:
                    try:
                        v = float(r[1])
                    except ValueError:
                        continue
                    if max_v is None or v > max_v:
                        max_v = v
        if max_v is not None:
            p_uflow = max_v

    peedt = _read_n_col_csv(solve_data_dir / "peedt.csv", 5)
    rows: list[tuple[str, ...]] = []
    for p, src, sink, d, t in peedt:
        if (p, src, sink) in coeff_zero:
            value = p_uflow
        else:
            us = unitsize.get(p, 1.0)
            dcm_v = dcm.get((p, d), 0.0)
            if p in has_indirect and (p, src) in process_source:
                if p in has_min_load:
                    eff_term = (slope.get((p, d, t), 0.0)
                                + section.get((p, d, t), 0.0))
                else:
                    eff_term = slope.get((p, d, t), 0.0)
                src_coef = src_max_coef.get((p, src), 1.0)
                base = eff_term * (dcm_v / us) / src_coef
            else:
                base = dcm_v / us
            sink_coef = (sink_max_coef.get((p, sink), 1.0)
                         if (p, sink) in process_sink else 1.0)
            value = base * sink_coef
        rows.append((p, src, sink, d, t, repr(value)))
    return _rows_to_frame(
        rows, ("process", "source", "sink", "period", "time", "value"),
    )


def write_p_flow_max(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1661-1677 — ``p_flow_max{(p,source,sink,d,t) in peedt}``.

    Two-branch value formula, see legacy docstring at
    ``process_arc_unions.write_p_flow_max``.  Every peedt row gets a
    value (mod's bare-decl has no default).
    """
    _write(
        derive_p_flow_max(input_dir, solve_data_dir),
        solve_data_dir / "p_flow_max.csv",
    )


# ---- write_p_state_slack_share (mod L1689-1691) ---------------------------


def derive_p_state_slack_share(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``p_state_slack_share`` 5-col frame; see writer docstring."""
    import csv
    g_loss = frozenset(
        _read_singles_csv(solve_data_dir / "group_loss_share.csv")
    )
    g_type: dict[str, str] = {}
    for g, ty in _read_pairs_csv(input_dir / "group__loss_share_type.csv"):
        g_type[g] = ty
    nodes_in_g: dict[str, list[str]] = {}
    for g, n in _read_pairs_csv(input_dir / "group__node.csv"):
        nodes_in_g.setdefault(g, []).append(n)
    inflow: dict[tuple[str, str, str], float] = {}
    pdtni_path = solve_data_dir / "pdtNodeInflow.csv"
    if pdtni_path.exists():
        with pdtni_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] and r[2]:
                    try:
                        inflow[(r[0], r[1], r[2])] = float(r[3])
                    except ValueError:
                        continue
    dt_pairs = _read_n_col_csv(solve_data_dir / "steps_in_use.csv", 2)

    rows: list[tuple[str, str, str, str, str]] = []
    for g in g_loss:
        ngs = nodes_in_g.get(g, [])
        if not ngs:
            continue
        share_type = g_type.get(g)
        n_count = len(ngs)
        for n in ngs:
            for d, t in dt_pairs:
                if share_type == "inflow_weighted":
                    total = sum(inflow.get((ng, d, t), 0.0) for ng in ngs)
                    v = (inflow.get((n, d, t), 0.0) / total
                         if total != 0.0 else 0.0)
                elif share_type == "equal":
                    v = 1.0 / n_count
                else:
                    v = 0.0
                rows.append((g, n, d, t, repr(v)))
    return _rows_to_frame(
        rows, ("group", "node", "period", "time", "value"),
    )


def write_p_state_slack_share(input_dir: Path, solve_data_dir: Path) -> None:
    """flextool.mod L1689-1691 — ``p_state_slack_share[g, n, d, t]``.

    Inflow-weighted or equal share over the nodes of group ``g`` for
    each ``(d, t) ∈ dt``, restricted to ``g ∈ group_loss_share``.
    """
    _write(
        derive_p_state_slack_share(input_dir, solve_data_dir),
        solve_data_dir / "p_state_slack_share.csv",
    )


# ---- write_p_storage_state_reference_price (mod L1693-1698) ---------------


def derive_p_storage_state_reference_price(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``p_storage_state_reference_price`` 3-col frame; see writer docstring."""
    import csv
    # (n, d2, t2) → value, keyed by (node, period, step) from fix_storage_price.
    fix_price: dict[tuple[str, str, str], float] = {}
    fsp_path = solve_data_dir / "fix_storage_price.csv"
    if fsp_path.exists():
        with fsp_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1] and r[2]:
                    try:
                        fix_price[(r[2], r[0], r[1])] = float(r[3])
                    except ValueError:
                        continue

    ptl = _read_pairs_csv(solve_data_dir / "last_timesteps.csv")
    ptl_for_d: dict[str, list[str]] = {}
    for d, t in ptl:
        ptl_for_d.setdefault(d, []).append(t)
    pb_d2_for_d: dict[str, list[str]] = {}
    for d2, d in _read_pairs_csv(solve_data_dir / "period__branch.csv"):
        pb_d2_for_d.setdefault(d, []).append(d2)
    dtt_for_dt: dict[tuple[str, str], list[str]] = {}
    for d, t, t2 in _read_n_col_csv(
        solve_data_dir / "timeline_matching_map.csv", 3
    ):
        dtt_for_dt.setdefault((d, t), []).append(t2)

    use_ref = frozenset(
        n for n, m in _read_pairs_csv(
            input_dir / "node__storage_solve_horizon_method.csv"
        ) if m == "use_reference_price"
    )

    pd_ref_price: dict[tuple[str, str], float] = {}
    pdn_path = solve_data_dir / "pdNode.csv"
    if pdn_path.exists():
        with pdn_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if (len(r) >= 4 and r[0]
                        and r[1] == "storage_state_reference_price"
                        and r[2]):
                    try:
                        pd_ref_price[(r[0], r[2])] = float(r[3])
                    except ValueError:
                        continue

    nodes_state = _read_singles_csv(solve_data_dir / "nodeState.csv")
    period_in_use = _read_singles_csv(
        solve_data_dir / "period_in_use_set.csv"
    )

    rows: list[tuple[str, str, str]] = []
    for n in nodes_state:
        for d in period_in_use:
            sum_v = 0.0
            has_match = False
            for d2 in pb_d2_for_d.get(d, []):
                for t in ptl_for_d.get(d, []):
                    for t2 in dtt_for_dt.get((d, t), []):
                        v = fix_price.get((n, d2, t2))
                        if v is not None:
                            has_match = True
                            sum_v += v
            if has_match:
                value = sum_v
            elif n in use_ref:
                value = pd_ref_price.get((n, d), 0.0)
            else:
                value = 0.0
            rows.append((n, d, repr(value)))
    return _rows_to_frame(rows, ("node", "period", "value"))


def write_p_storage_state_reference_price(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1693-1698 — ``p_storage_state_reference_price[n, d]``.

    Sum of ``p_fix_storage_price`` over the ``(d2, t2)`` matched by
    ``period__branch`` + ``period__time_last`` + ``dtt_timeline_matching``
    if any match exists, otherwise fall back to
    ``pdNode[n, 'storage_state_reference_price', d]`` for nodes with
    ``(n, 'use_reference_price') ∈ node__storage_solve_horizon_method``,
    else 0.
    """
    _write(
        derive_p_storage_state_reference_price(input_dir, solve_data_dir),
        solve_data_dir / "p_storage_state_reference_price.csv",
    )


# ---- write_node_group_dispatch_sets (mod L1596-1657) ----------------------


def _compute_node_group_dispatch_sets(
    input_dir: Path, solve_data_dir: Path,
) -> dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]:
    """One shared scan; returns ``{filename → (header, rows)}`` for the
    12 nodeGroupDispatch CSVs.
    """
    ngd = _read_singles_csv(input_dir / "nodeGroupDispatch.csv")
    fag = frozenset(_read_singles_csv(input_dir / "flowAggregator.csv"))
    p_unit = frozenset(_read_singles_csv(input_dir / "process_unit.csv"))
    p_conn = frozenset(_read_singles_csv(input_dir / "process_connection.csv"))

    g_nodes_acc: dict[str, dict[str, None]] = {}
    for g, n in _read_pairs_csv(input_dir / "group__node.csv"):
        g_nodes_acc.setdefault(g, {})[n] = None
    g_nodes: dict[str, frozenset[str]] = {
        g: frozenset(d.keys()) for g, d in g_nodes_acc.items()
    }

    # group_process_node restricted to flowAggregator groups: (p, n) → [ga, ...]
    pn_to_aggregators: dict[tuple[str, str], list[str]] = {}
    for g, p, n in _read_n_col_csv(input_dir / "group__process__node.csv", 3):
        if g in fag:
            pn_to_aggregators.setdefault((p, n), []).append(g)

    pss_always = _read_n_col_csv(
        solve_data_dir / "process_source_sink_alwaysProcess.csv", 3
    )
    fully_inside = frozenset(_read_pairs_csv(
        solve_data_dir / "nodeGroupDispatch__process_fully_inside.csv"
    ))

    def _emit_4tuple(*, kind: frozenset[str], side: str,
                     not_aggregated: bool) -> list[tuple[str, ...]]:
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
                if not_aggregated and (p, n) in pn_to_aggregators:
                    continue
                out.append((g, p, src, sink))
        return out

    def _emit_5tuple(*, kind: frozenset[str], side: str
                     ) -> list[tuple[str, ...]]:
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

    rows1 = _emit_4tuple(kind=p_unit, side="sink", not_aggregated=True)
    rows2 = _emit_4tuple(kind=p_unit, side="source", not_aggregated=True)
    rows3 = _emit_5tuple(kind=p_unit, side="sink")
    rows4 = _emit_5tuple(kind=p_unit, side="source")
    rows5 = _emit_4tuple(kind=p_conn, side="source", not_aggregated=True)
    rows6 = _emit_4tuple(kind=p_conn, side="sink", not_aggregated=True)
    rows8 = _emit_5tuple(kind=p_conn, side="sink")
    rows9 = _emit_5tuple(kind=p_conn, side="source")

    # Set 7 — projection of 5 ∪ 6 to (g, connection).
    seen7: dict[tuple[str, str], None] = {}
    for g, p, _, _ in rows5:
        seen7.setdefault((g, p), None)
    for g, p, _, _ in rows6:
        seen7.setdefault((g, p), None)
    # Set 10 — projection of 8 ∪ 9 to (g, ga).
    seen10: dict[tuple[str, str], None] = {}
    for g, ga, _, _, _ in rows8:
        seen10.setdefault((g, ga), None)
    for g, ga, _, _, _ in rows9:
        seen10.setdefault((g, ga), None)
    # Set 11 — projection of rows3 to (g, ga).
    seen11: dict[tuple[str, str], None] = {}
    for g, ga, _, _, _ in rows3:
        seen11.setdefault((g, ga), None)
    # Set 12 — projection of rows4 to (g, ga).
    seen12: dict[tuple[str, str], None] = {}
    for g, ga, _, _, _ in rows4:
        seen12.setdefault((g, ga), None)

    return {
        "nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv": (
            ("group", "process", "unit", "node"), rows1,
        ),
        "nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv": (
            ("group", "process", "node", "unit"), rows2,
        ),
        "nodeGroupDispatch__group_aggregate__process__unit__to_node.csv": (
            ("group", "group_aggregate", "unit", "source", "sink"), rows3,
        ),
        "nodeGroupDispatch__group_aggregate__process__node__to_unit.csv": (
            ("group", "group_aggregate", "unit", "source", "sink"), rows4,
        ),
        "nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv": (
            ("group", "process", "node", "connection"), rows5,
        ),
        "nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv": (
            ("group", "process", "connection", "node"), rows6,
        ),
        "nodeGroupDispatch__connection_Not_in_aggregate.csv": (
            ("group", "connection"), list(seen7.keys()),
        ),
        "nodeGroupDispatch__group_aggregate__process__connection__to_node.csv": (
            ("group", "group_aggregate", "connection", "source", "sink"), rows8,
        ),
        "nodeGroupDispatch__group_aggregate__process__node__to_connection.csv": (
            ("group", "group_aggregate", "connection", "source", "sink"), rows9,
        ),
        "nodeGroupDispatch__group_aggregate_Connection.csv": (
            ("group", "group_aggregate"), list(seen10.keys()),
        ),
        "nodeGroupDispatch__group_aggregate_Unit_to_group.csv": (
            ("group", "group_aggregate"), list(seen11.keys()),
        ),
        "nodeGroupDispatch__group_aggregate_Group_to_unit.csv": (
            ("group", "group_aggregate"), list(seen12.keys()),
        ),
    }


def write_node_group_dispatch_sets(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """flextool.mod L1596-1657 — 12 nodeGroupDispatch sets.

    Joins ``process_source_sink_alwaysProcess`` with ``nodeGroupDispatch``
    + ``group__node`` + ``group__process__node`` + ``flowAggregator`` +
    ``process_unit`` / ``process_connection``.  Eight base sets partition
    the (g, p, source, sink) space by side (sink-in-group vs source-in-
    group), kind (unit vs connection) and aggregator presence (with-ga
    vs no-ga).  Four projection sets project pairs (g, p) or (g, ga) from
    the relevant base sets.  All sets share the prefilter
    ``(g, p) ∉ nodeGroupDispatch__process_fully_inside``.
    """
    by_file = _compute_node_group_dispatch_sets(input_dir, solve_data_dir)
    for fname, (header, rows) in by_file.items():
        _write(_rows_to_frame(rows, header), solve_data_dir / fname)


# ---------------------------------------------------------------------------
# write_param_t_projections_and_time_params — Phase 1 follow-up 7
# (legacy process_arc_unions.py:1914, ~150 LOC)
# ---------------------------------------------------------------------------

# flextool_base.dat:158 — SOURCE_SINK_TIME_PARAM (mirrored locally for the
# same enum-iteration-parity reasons as _PROCESS_TIME_PARAM above; legacy
# stores it as a frozenset and the native writer mirrors that storage so
# in-process iteration order matches byte-for-byte).
_SS_TIME_PARAM_ENUM: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load", "other_operational_cost",
))


def _read_pt_pp_t_seen(
    pt_path: Path, proc_conn: frozenset[str],
) -> tuple[dict[tuple[str, str], None],
           list[tuple[str, str, str]],
           dict[tuple[str, str], None]]:
    """Read pt_process.csv and return:

      * pp_t_seen   — ordered (process, param) projection of all rows
      * conn_pt_rows — full (process, param, time) rows where process is
                       a connection (preserving CSV order)
      * conn_pt_seen — ordered (process, param) projection restricted to
                       connection rows.

    Mirrors legacy ``setdefault`` ordered-dedup semantics.
    """
    pp_t_seen: dict[tuple[str, str], None] = {}
    conn_pt_rows: list[tuple[str, str, str]] = []
    conn_pt_seen: dict[tuple[str, str], None] = {}
    if not pt_path.exists():
        return pp_t_seen, conn_pt_rows, conn_pt_seen
    import csv
    with pt_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 3 and r[0] and r[1] and r[2]:
                pp_t_seen.setdefault((r[0], r[1]), None)
                if r[0] in proc_conn:
                    conn_pt_rows.append((r[0], r[1], r[2]))
                    conn_pt_seen.setdefault((r[0], r[1]), None)
    return pp_t_seen, conn_pt_rows, conn_pt_seen


def _read_pps_t_seen(
    path: Path,
) -> dict[tuple[str, str, str], None]:
    """Read pt_process_source.csv or pt_process_sink.csv and return the
    ordered (e0, e1, param) projection (legacy ``setdefault`` order)."""
    seen: dict[tuple[str, str, str], None] = {}
    if not path.exists():
        return seen
    import csv
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 4 and r[0] and r[1] and r[2]:
                seen.setdefault((r[0], r[1], r[2]), None)
    return seen


def _read_param_static_3(path: Path) -> set[tuple[str, str, str]]:
    """Read input/p_process_source.csv or p_process_sink.csv as a set
    of (process, side, param) triples (membership test only — order
    not load-bearing)."""
    out: set[tuple[str, str, str]] = set()
    if not path.exists():
        return out
    import csv
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 3 and r[0] and r[1] and r[2]:
                out.add((r[0], r[1], r[2]))
    return out


def _read_param_static_2(path: Path) -> set[tuple[str, str]]:
    """Read input/p_process.csv as a set of (process, param) pairs."""
    out: set[tuple[str, str]] = set()
    if not path.exists():
        return out
    import csv
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 2 and r[0] and r[1]:
                out.add((r[0], r[1]))
    return out


def write_param_t_projections_and_time_params(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Native port of
    :func:`flextool.flextoolrunner.preprocessing.process_arc_unions.write_param_t_projections_and_time_params`
    (legacy line 1914).

    Emits 8 ``solve_data/`` CSVs:

    * Projections (drop the time column from each *__time set):
        - ``process__param_t.csv``
        - ``connection__param__time.csv``
        - ``connection__param_t.csv``
        - ``process__source__param_t.csv``
        - ``process__sink__param_t.csv``

    * Joins with ``SOURCE_SINK_TIME_PARAM``:
        - ``process__source__timeParam.csv``
        - ``process__sink__timeParam.csv``
        - ``process__timeParam.csv``

    Reads ``solve_data/pt_process{,_source,_sink}.csv`` and
    ``input/p_process{,_source,_sink}.csv``,
    ``input/process{,_connection,__source,__sink}.csv``.
    """
    proc_conn = frozenset(_read_singles_list(input_dir / "process_connection.csv"))

    # process__param__time → projection (process, param) [drop time]
    pp_t_seen, conn_pt_rows, conn_pt_seen = _read_pt_pp_t_seen(
        solve_data_dir / "pt_process.csv", proc_conn,
    )
    _write(
        _rows_to_frame(list(pp_t_seen.keys()), ("process", "param")),
        solve_data_dir / "process__param_t.csv",
    )
    _write(
        _rows_to_frame(conn_pt_rows, ("connection", "param", "time")),
        solve_data_dir / "connection__param__time.csv",
    )
    _write(
        _rows_to_frame(list(conn_pt_seen.keys()), ("connection", "param")),
        solve_data_dir / "connection__param_t.csv",
    )

    # process__source__param_t (drop time)
    pps_t_seen = _read_pps_t_seen(solve_data_dir / "pt_process_source.csv")
    _write(
        _rows_to_frame(
            list(pps_t_seen.keys()), ("process", "source", "param"),
        ),
        solve_data_dir / "process__source__param_t.csv",
    )

    # process__sink__param_t
    ppk_t_seen = _read_pps_t_seen(solve_data_dir / "pt_process_sink.csv")
    _write(
        _rows_to_frame(
            list(ppk_t_seen.keys()), ("process", "sink", "param"),
        ),
        solve_data_dir / "process__sink__param_t.csv",
    )

    # Static parameter sets from input/p_process_source.csv, p_process_sink.csv,
    # p_process.csv.
    src_param = _read_param_static_3(input_dir / "p_process_source.csv")
    sink_param = _read_param_static_3(input_dir / "p_process_sink.csv")
    proc_param = _read_param_static_2(input_dir / "p_process.csv")

    # process__source__timeParam
    proc_sources = _read_n_col_rows(
        input_dir / "process__source.csv", ["process", "source"],
    )
    proc_sinks = _read_n_col_rows(
        input_dir / "process__sink.csv", ["process", "sink"],
    )
    pps_t_set = frozenset(pps_t_seen.keys())
    ppk_t_set = frozenset(ppk_t_seen.keys())
    pp_t_set = frozenset(pp_t_seen.keys())

    rows_src_tp: list[tuple[str, str, str]] = []
    for p, src in proc_sources:
        for param in _SS_TIME_PARAM_ENUM:
            if ((p, src, param) in src_param
                    or (p, src, param) in pps_t_set):
                rows_src_tp.append((p, src, param))
    _write(
        _rows_to_frame(rows_src_tp, ("process", "source", "param")),
        solve_data_dir / "process__source__timeParam.csv",
    )

    rows_snk_tp: list[tuple[str, str, str]] = []
    for p, snk in proc_sinks:
        for param in _SS_TIME_PARAM_ENUM:
            if ((p, snk, param) in sink_param
                    or (p, snk, param) in ppk_t_set):
                rows_snk_tp.append((p, snk, param))
    _write(
        _rows_to_frame(rows_snk_tp, ("process", "sink", "param")),
        solve_data_dir / "process__sink__timeParam.csv",
    )

    # process__timeParam — only for processes that are connections.
    processes = _read_singles_list(input_dir / "process.csv")
    rows_p_tp: list[tuple[str, str]] = []
    for p in processes:
        if p not in proc_conn:
            continue
        for param in _SS_TIME_PARAM_ENUM:
            if (p, param) in proc_param or (p, param) in pp_t_set:
                rows_p_tp.append((p, param))
    _write(
        _rows_to_frame(rows_p_tp, ("process", "param")),
        solve_data_dir / "process__timeParam.csv",
    )
