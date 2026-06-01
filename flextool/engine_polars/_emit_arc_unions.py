"""Process-arc-union + period-param leaf writers.

Cheap, leaf-like writers whose inputs are already-native L0-L9
``solve_data/*.csv`` outputs (or plain ``input/*.csv``) and whose
semantics are pure projection / join / filter with no ``PdtLookup``-class
machinery behind them.

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
:mod:`._emit_leaf_sets` / :mod:`._emit_mid_sets` /
:mod:`._emit_calc_params`.

Style mirrors :mod:`._emit_calc_params` — eager polars reads of tiny
CSVs with positional column renames, expression chains where natural,
small python loops where the iteration order is precision-load-bearing
(matches the legacy ``dict.fromkeys`` ordered-dedup pattern).

Precision-parity pattern
------------------------

``write_pProcess_source_sink`` writes a value column.  Legacy formats
it via ``f"{repr(v)}"`` with ``v`` already a python float — we mirror
exactly by pre-stringifying with ``repr(float(v))``.  See
:mod:`._emit_calc_params` module docstring for the precision-parity
rationale (round-trip-exactness of ``repr(float)`` and divergence
from polars' default float formatting).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# CSV I/O — same conventions as the sibling _emit_*.py modules.
# ---------------------------------------------------------------------------

# Provider-aware open helper — re-exported from the shared module.
# Step 2.5 Phase B collapsed the local copy that carried a disk-fallback
# arm; cascade code uses the Provider-only shim.

from flextool.engine_polars._emit_provider_io import (  # noqa: E402
    _emit,
    _provider_key,
)
from flextool.engine_polars import _provider_keys as K  # noqa: E402
from flextool.engine_polars._provider_translators import (  # noqa: E402
    read_handoff_frame,
)


def _cell_str(value: "object | None") -> str:
    """Reproduce a ``csv.reader`` cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; ``csv.reader`` then reads those
    strings back.  Mirror that here so dict/set keys and string
    comparisons stay byte-identical to the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


def _read_csv(path: Path, columns: list[str],
              *,
              provider: "object | None" = None) -> pl.DataFrame:
    """Provider-only — Step 2.5 Phase C dropped the disk-fallback arm.

    Returns the Provider's frame sliced to *columns* with positional
    rename; returns an empty all-Utf8 frame when the Provider misses
    the key (matches legacy missing-CSV behaviour).
    """
    from flextool.engine_polars._emit_provider_io import (
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    return pl.DataFrame(
        {c: [] for c in columns}, schema={c: pl.Utf8 for c in columns},
    )


def _drop_blank_rows(df: pl.DataFrame, required_cols: list[str]) -> pl.DataFrame:
    expr = pl.col(required_cols[0]) != ""
    for c in required_cols[1:]:
        expr = expr & (pl.col(c) != "")
    return df.filter(expr)


def _read_n_col_rows(path: Path, columns: list[str],
                      *,
                      provider: "object | None" = None) -> list[tuple[str, ...]]:
    """Read a CSV as a list of tuples preserving CSV row order.

    Mirrors the legacy ``_read_n_col`` helper — used where the legacy
    code iterates a list (not a set) to drive deterministic dedupe via
    ``dict.fromkeys``.
    """
    df = _read_csv(path, columns, provider=provider)
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

def derive_process_source_sink_param_t(solve_data_dir: Path,
                                         *,
                                         provider: "object | None" = None,
                                         ) -> pl.DataFrame:
    """``process_source_sink_eff`` × ``processTimeParam`` filtered by
    ``(p, param) in process__param_t`` (loaded from ``pt_process.csv``).
    """
    pss_eff = _read_n_col_rows(
        solve_data_dir / "process_source_sink_eff.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    pt_pairs = _read_csv(
        solve_data_dir / "pt_process.csv", ["process", "param"],
        provider=provider,
    )
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


def emit_process_source_sink_param_t(solve_data_dir: Path,
                                       *, provider) -> None:
    """Emit ``process_source_sink_param_t``."""
    _emit(provider, "solve_data/process_source_sink_param_t.csv",
          derive_process_source_sink_param_t(solve_data_dir, provider=provider))


# ---- node__TimeParam_in_use (mod L1208-1214) ------------------------------

def derive_node_time_param_in_use(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """node × nodeTimeParam filtered by per-node membership in
    nodeBalance / nodeBalancePeriod / nodeState
    or by ``(n, 'use_reference_value') in node__storage_solve_horizon_method``.
    """
    nodes = (
        _drop_blank_rows(
            _read_csv(input_dir / "node.csv", ["node"], provider=provider),
            ["node"],
        ).get_column("node").to_list()
    )
    n_balance = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "nodeBalance.csv", ["node"],
                      provider=provider),
            ["node"],
        ).get_column("node").to_list()
    )
    n_balance_period = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "nodeBalancePeriod.csv", ["node"],
                      provider=provider),
            ["node"],
        ).get_column("node").to_list()
    )
    n_state = frozenset(
        _drop_blank_rows(
            _read_csv(solve_data_dir / "nodeState.csv", ["node"],
                      provider=provider),
            ["node"],
        ).get_column("node").to_list()
    )
    storage_method = _drop_blank_rows(
        _read_csv(
            input_dir / "node__storage_solve_horizon_method.csv",
            ["node", "method"],
            provider=provider,
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


def emit_node_time_param_in_use(input_dir: Path, solve_data_dir: Path,
                                  *, provider) -> None:
    """Emit ``node_time_param_in_use`` to the Provider."""
    _emit(provider, "solve_data/node__TimeParam_in_use.csv",
          derive_node_time_param_in_use(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- process_source_{delayed,undelayed} (mod L1092-1093) -------------------

def derive_process_source_delayed_partition(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition ``process__source`` by membership in ``process_delayed``.

    Returns (delayed, undelayed) frames, each with columns (process, source).
    """
    pairs = _read_csv(
        input_dir / "process__source.csv", ["process", "source"],
        provider=provider,
    )
    pairs = _drop_blank_rows(pairs, ["process", "source"])
    delayed_set = frozenset(
        _drop_blank_rows(
            _read_csv(
                solve_data_dir / "process_delayed.csv", ["process"],
                provider=provider,
            ),
            ["process"],
        ).get_column("process").to_list()
    )
    delayed = pairs.filter(pl.col("process").is_in(list(delayed_set)))
    undelayed = pairs.filter(~pl.col("process").is_in(list(delayed_set)))
    return delayed, undelayed


def emit_process_source_delayed_partition(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_delayed_partition`` to the Provider."""
    delayed, undelayed = derive_process_source_delayed_partition(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/process_source_delayed.csv", delayed)
    _emit(provider, "solve_data/process_source_undelayed.csv", undelayed)


# ---- process__source__sink__param (mod L1185-1189) -------------------------

def derive_process_source_sink_param(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``process_source_sink × SOURCE_SINK_PARAM`` admitted if param row
    exists on EITHER side, OR via connection-process ``p_process``.
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )

    src_param = _drop_blank_rows(
        _read_csv(
            input_dir / "p_process_source.csv",
            ["process", "source", "param"],
            provider=provider,
        ),
        ["process", "source", "param"],
    )
    sink_param = _drop_blank_rows(
        _read_csv(
            input_dir / "p_process_sink.csv",
            ["process", "sink", "param"],
            provider=provider,
        ),
        ["process", "sink", "param"],
    )
    proc_param = _drop_blank_rows(
        _read_csv(
            input_dir / "p_process.csv", ["process", "param"],
            provider=provider,
        ),
        ["process", "param"],
    )
    proc_conn = frozenset(
        _drop_blank_rows(
            _read_csv(
                input_dir / "process_connection.csv", ["process"],
                provider=provider,
            ),
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


def emit_process_source_sink_param(input_dir: Path, solve_data_dir: Path,
                                     *, provider) -> None:
    """Emit ``process_source_sink_param`` to the Provider."""
    _emit(provider, "solve_data/process__source__sink__param.csv",
          derive_process_source_sink_param(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- process__source__sink__profile__profile_method_connection
#      (mod L1060-1063) -------------------------------------------------------

def derive_process_source_sink_profile_method_connection(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``process_source_sink × profile × profile_method`` filtered by
    ``(p, profile, method) in process__profile__profile_method``.
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    pp_pm = _read_n_col_rows(
        input_dir / "process__profile__profile_method.csv",
        ["process", "profile", "profile_method"],
        provider=provider,
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


def emit_process_source_sink_profile_method_connection(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_sink_profile_method_connection`` to the Provider."""
    _emit(
        provider,
        "solve_data/process__source__sink__profile__profile_method_connection.csv",
        derive_process_source_sink_profile_method_connection(
            input_dir, solve_data_dir, provider=provider,
        ),
    )


# ---- process_method_sources_sinks (mod L1046-1053) -------------------------

def derive_process_method_sources_sinks(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """3-way join over process_source_sink_alwaysProcess, process_source_sink
    and process_method, aliasing-gated.

    Output columns: (process, method, orig_source, orig_sink,
                     always_source, always_sink).
    """
    always = _read_n_col_rows(
        solve_data_dir / "process_source_sink_alwaysProcess.csv",
        ["process", "always_source", "always_sink"],
        provider=provider,
    )
    pss = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "orig_source", "orig_sink"],
        provider=provider,
    )
    pm = _read_n_col_rows(
        input_dir / "process_method.csv",
        ["process", "method"],
        provider=provider,
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


def emit_process_method_sources_sinks(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_method_sources_sinks`` to the Provider."""
    _emit(provider, "solve_data/process_method_sources_sinks.csv",
          derive_process_method_sources_sinks(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- ed_history_realized_first (mod L993) ---------------------------------

def derive_ed_history_realized_first(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """entity × realized periods, but only on the first solve.

    Honours the ``solveFirst`` flag on ``p_model``: non-first solves
    emit an empty frame.
    """
    # solveFirst gate: short-circuit to empty.
    solve_first = False
    pm = _read_csv(
        solve_data_dir / "p_model.csv", ["key", "value"], provider=provider,
    )
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
        _drop_blank_rows(
            _read_csv(input_dir / "entity.csv", ["entity"], provider=provider),
            ["entity"],
        ).get_column("entity").to_list()
    )
    d_realize_invest = frozenset(
        _drop_blank_rows(
            _read_csv(
                solve_data_dir / "realized_invest_periods_of_current_solve.csv",
                ["period"],
                provider=provider,
            ),
            ["period"],
        ).get_column("period").to_list()
    )
    d_fix_storage = frozenset(
        _drop_blank_rows(
            _read_csv(
                solve_data_dir / "d_fix_storage_period_set.csv", ["period"],
                provider=provider,
            ),
            ["period"],
        ).get_column("period").to_list()
    )
    d_realized = frozenset(
        _drop_blank_rows(
            _read_csv(
                solve_data_dir / "d_realized_period_set.csv", ["period"],
                provider=provider,
            ),
            ["period"],
        ).get_column("period").to_list()
    )
    realized_periods = d_realize_invest | d_fix_storage | d_realized

    pb = _drop_blank_rows(
        _read_csv(
            solve_data_dir / "period__branch.csv", ["period", "branch"],
            provider=provider,
        ),
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


def emit_ed_history_realized_first(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``ed_history_realized_first`` to the Provider."""
    _emit(provider, "solve_data/ed_history_realized_first.csv",
          derive_ed_history_realized_first(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source
#      (mod L1152-1155) -----------------------------------------------------

def derive_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``process_source_sink`` filtered for 1-way processes whose source
    is a real node-side endpoint, with no sink OR ≥2 sources.
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    pm = _read_n_col_rows(
        input_dir / "process_method.csv", ["process", "method"],
        provider=provider,
    )
    methods_of_p: dict[str, set[str]] = {}
    for p, m in pm:
        methods_of_p.setdefault(p, set()).add(m)
    has_1way = {p: bool(ms & _METHOD_1WAY) for p, ms in methods_of_p.items()}

    proc_source_pairs = frozenset(_read_n_col_rows(
        input_dir / "process__source.csv", ["process", "source"],
        provider=provider,
    ))
    sources_of_p: dict[str, int] = {}
    for p, _ in proc_source_pairs:
        sources_of_p[p] = sources_of_p.get(p, 0) + 1
    sinks_of_p: dict[str, int] = {}
    for p, _ in _read_n_col_rows(
        input_dir / "process__sink.csv", ["process", "sink"],
        provider=provider,
    ):
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


def emit_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_is_node_sink_1way_no_sink_or_more_than_1_source`` to the Provider."""
    _emit(
        provider,
        "solve_data/process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.csv",
        derive_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(
            input_dir, solve_data_dir, provider=provider,
        ),
    )


# ---- process__source__sink__ramp_method (mod L1205-1209) -------------------

def derive_process_source_sink_ramp_method(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``process_source_sink × ramp_method`` filtered by per-side membership."""
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    pnrm = _read_n_col_rows(
        input_dir / "process__node__ramp_method.csv",
        ["process", "node", "ramp_method"],
        provider=provider,
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


def emit_process_source_sink_ramp_method(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_sink_ramp_method`` to the Provider."""
    _emit(provider, "solve_data/process__source__sink__ramp_method.csv",
          derive_process_source_sink_ramp_method(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- process_source_sink_coeff_zero (mod L1973) ---------------------------

def derive_process_source_sink_coeff_zero(
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``process_source_sink`` filtered by zero flow coefficient on EITHER side."""
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    src_zero = frozenset(_read_n_col_rows(
        solve_data_dir / "process_source_coeff_zero.csv",
        ["process", "source"],
        provider=provider,
    ))
    sink_zero = frozenset(_read_n_col_rows(
        solve_data_dir / "process_sink_coeff_zero.csv",
        ["process", "sink"],
        provider=provider,
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


def emit_process_source_sink_coeff_zero(
    solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_sink_coeff_zero``."""
    _emit(provider, "solve_data/process_source_sink_coeff_zero.csv",
          derive_process_source_sink_coeff_zero(solve_data_dir,
                                                  provider=provider))


# ---- process__source__sinkIsNode_* family (mod L1071, L1153-1158) ---------

def derive_process_source_sink_is_node_family(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Emit the 4-frame ``process__source__sinkIsNode*`` family.

    Returns (base, two_way_1var, not_two_way_1var, two_way_2var).
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    sinks = frozenset(_read_n_col_rows(
        input_dir / "process__sink.csv", ["process", "sink"],
        provider=provider,
    ))
    pm = _read_n_col_rows(
        input_dir / "process_method.csv", ["process", "method"],
        provider=provider,
    )
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


def emit_process_source_sink_is_node_family(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_sink_is_node_family`` to the Provider."""
    base, two1, not21, two2 = derive_process_source_sink_is_node_family(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/process__source__sinkIsNode.csv", base)
    _emit(provider, "solve_data/process__source__sinkIsNode_2way1var.csv", two1)
    _emit(provider, "solve_data/process__source__sinkIsNode_not2way1var.csv", not21)
    _emit(provider, "solve_data/process__source__sinkIsNode_2way2var.csv", two2)


# ---- process_source_sink_{delayed,undelayed} (mod L1096-1097) --------------

def derive_process_source_sink_delayed_partition(
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition ``process_source_sink`` by membership in ``process_delayed``."""
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    delayed_set = frozenset(
        _drop_blank_rows(
            _read_csv(
                solve_data_dir / "process_delayed.csv", ["process"],
                provider=provider,
            ),
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


def emit_process_source_sink_delayed_partition(
    solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit the ``process_source_sink_delayed``/``_undelayed`` partition."""
    delayed, undelayed = derive_process_source_sink_delayed_partition(
        solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/process_source_sink_delayed.csv", delayed)
    _emit(provider, "solve_data/process_source_sink_undelayed.csv", undelayed)


# ===========================================================================
# entity_period_calc_params — self-contained subset
# ===========================================================================


def _read_value_lookup_3(path: Path,
                          *,
                          provider: "object | None" = None,
                          ) -> dict[tuple[str, str, str], float]:
    """Load a 4-col CSV (k1, k2, k3, value) as a dict keyed on the
    first 3 columns and floated on the 4th.  Silently skip rows whose
    value can't be parsed.
    """
    df = _read_csv(path, ["k1", "k2", "k3", "value"], provider=provider)
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
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``pProcess_source_sink``: prefer p_process_source, fall back to
    p_process_sink, then 0.  Domain = ``process__source__sink__param``.

    Returns a 5-col frame; value column is pre-stringified with
    ``repr(float(v))`` to preserve bit-exact precision parity with
    legacy code (see :mod:`._emit_calc_params` module docstring).
    """
    p_src = _read_value_lookup_3(
        input_dir / "p_process_source.csv", provider=provider,
    )
    p_snk = _read_value_lookup_3(
        input_dir / "p_process_sink.csv", provider=provider,
    )

    domain = _read_n_col_rows(
        solve_data_dir / "process__source__sink__param.csv",
        ["process", "source", "sink", "param"],
        provider=provider,
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


def emit_pProcess_source_sink(input_dir: Path, solve_data_dir: Path,
                                *, provider) -> None:
    """Emit ``pProcess_source_sink`` to the Provider."""
    _emit(provider, "solve_data/pProcess_source_sink.csv",
          derive_p_process_source_sink(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---------------------------------------------------------------------------
# process_source_sink_ramp_family (mod L1660-1688)
# ---------------------------------------------------------------------------

def _read_p_proc_side_lookup(path: Path,
                              *,
                              provider: "object | None" = None,
                              ) -> dict[tuple[str, str, str], float]:
    """Read p_process_source / p_process_sink: (process, side, param) → value.

    Provider-only after Step 2.5 Phase C — returns empty when the
    Provider misses the key (matches legacy missing-CSV behaviour).
    """
    out: dict[tuple[str, str, str], float] = {}
    if provider is None or not provider.has(_provider_key(path)):
        return out
    df = _read_csv(path, ["process", "side", "param", "value"], provider=provider)
    df = _drop_blank_rows(df, ["process", "side", "param"])
    for p, s, param, v in df.iter_rows():
        try:
            out[(p, s, param)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _compute_ramp_family(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> dict[str, list[tuple[str, str, str]]]:
    """Emit the 5 ramp-family triple sets.

    Returns ``{filename → rows}``.  Rows preserve ``process_source_sink``
    order; legacy emits no dedup (input is already unique).
    """
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
    )
    pnrm_rows = _read_n_col_rows(
        input_dir / "process__node__ramp_method.csv",
        ["process", "node", "ramp_method"],
        provider=provider,
    )
    pnrm: dict[tuple[str, str], set[str]] = {}
    for p, n, m in pnrm_rows:
        pnrm.setdefault((p, n), set()).add(m)

    p_proc_source = _read_p_proc_side_lookup(
        input_dir / "p_process_source.csv", provider=provider,
    )
    p_proc_sink = _read_p_proc_side_lookup(
        input_dir / "p_process_sink.csv", provider=provider,
    )

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


def emit_process_source_sink_ramp_family(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_sink_ramp_family`` to the Provider."""
    by_file = _compute_ramp_family(input_dir, solve_data_dir, provider=provider)
    for fname, rows in by_file.items():
        _emit(provider, f"solve_data/{fname}", _triples_frame(rows))


# ---------------------------------------------------------------------------
# process_source_sink_ramp_unions — 5-way union of ramp_*.csv files
# ---------------------------------------------------------------------------

def derive_process_source_sink_ramp_unions(
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
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
            provider=provider,
        ):
            seen.setdefault(r, None)
    return _triples_frame(list(seen.keys()))


def emit_process_source_sink_ramp_unions(
    solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``process_source_sink_ramp``."""
    _emit(provider, "solve_data/process_source_sink_ramp.csv",
          derive_process_source_sink_ramp_unions(solve_data_dir,
                                                   provider=provider))


# ---------------------------------------------------------------------------
# group_commodity_node_period_co2_total (mod L1981)
# ---------------------------------------------------------------------------

def derive_group_commodity_node_period_co2_total(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``group_commodity_node_period_co2_total`` 3-col frame; see writer."""
    cn = _read_n_col_rows(
        input_dir / "commodity__node.csv", ["commodity", "node"],
        provider=provider,
    )
    gn = _read_n_col_rows(
        input_dir / "group__node.csv", ["group", "node"],
        provider=provider,
    )
    g_with_n: dict[str, set[str]] = {}
    for g, n in gn:
        g_with_n.setdefault(g, set()).add(n)

    p_commodity: dict[tuple[str, str], float] = {}
    pc_path = input_dir / "p_commodity.csv"
    _pc_has = provider is not None and provider.has(_provider_key(pc_path))
    if _pc_has:
        pc_df = _read_csv(
            pc_path, ["commodity", "param", "value"], provider=provider,
        )
        pc_df = _drop_blank_rows(pc_df, ["commodity", "param"])
        for c, param, v in pc_df.iter_rows():
            try:
                p_commodity[(c, param)] = float(v)
            except (TypeError, ValueError):
                continue
    co2_max_total = frozenset(
        _read_n_col_rows(
            solve_data_dir / "group_co2_max_total.csv", ["group"],
            provider=provider,
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


def emit_group_commodity_node_period_co2_total(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``group_commodity_node_period_co2_total`` to the Provider."""
    _emit(provider, "solve_data/group_commodity_node_period_co2_total.csv",
          derive_group_commodity_node_period_co2_total(
              input_dir, solve_data_dir, provider=provider,
          ))


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

def _read_singles_list(path: Path,
                        *,
                        provider: "object | None" = None) -> list[str]:
    """Read column 0 of a small CSV into a list (preserves CSV order)."""
    return [
        r[0] for r in _read_n_col_rows(path, ["c0"], provider=provider)
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


def emit_param_in_use_sets(input_dir: Path, solve_data_dir: Path,
                            *, provider) -> None:
    """Emit ``param_in_use_sets`` to the Provider."""
    nodes = _read_singles_list(input_dir / "node.csv", provider=provider)
    processes = _read_singles_list(input_dir / "process.csv", provider=provider)
    invest_set = frozenset(
        _read_singles_list(
            solve_data_dir / "entityInvest.csv", provider=provider,
        )
    )
    divest_set = frozenset(
        _read_singles_list(
            solve_data_dir / "entityDivest.csv", provider=provider,
        )
    )
    ctm = _read_n_col_rows(
        solve_data_dir / "process__ct_method.csv", ["process", "method"],
        provider=provider,
    )
    p_with_min_load = frozenset(
        p for p, m in ctm if m == "min_load_efficiency"
    )
    process_online = frozenset(
        _read_singles_list(
            solve_data_dir / "process_online.csv", provider=provider,
        )
    )
    sources = [
        (p, src) for p, src in _read_n_col_rows(
            input_dir / "process__source.csv", ["process", "source"],
            provider=provider,
        )
    ]
    sinks = [
        (p, snk) for p, snk in _read_n_col_rows(
            input_dir / "process__sink.csv", ["process", "sink"],
            provider=provider,
        )
    ]

    _emit(provider, "solve_data/node__PeriodParam_in_use.csv",
          _rows_to_frame_2(
              _derive_node_period_param_in_use(nodes, invest_set, divest_set),
              ("node", "param"),
          ))
    _emit(provider, "solve_data/process__PeriodParam_in_use.csv",
          _rows_to_frame_2(
              _derive_process_period_param_in_use(
                  processes, invest_set, divest_set, process_online,
              ),
              ("process", "param"),
          ))
    _emit(provider, "solve_data/process_TimeParam_in_use.csv",
          _rows_to_frame_2(
              _derive_process_time_param_in_use(processes, p_with_min_load),
              ("process", "param"),
          ))
    _emit(provider, "solve_data/process_source_sourceSinkTimeParam_in_use.csv",
          _rows_to_frame_3(
              _derive_pss_param_in_use(
                  sources, p_with_min_load,
                  _SOURCE_SINK_TIME_PARAM, _SOURCE_SINK_TIME_PARAM_REQUIRED,
              ),
              ("process", "source", "param"),
          ))
    _emit(provider, "solve_data/process_sink_sourceSinkTimeParam_in_use.csv",
          _rows_to_frame_3(
              _derive_pss_param_in_use(
                  sinks, p_with_min_load,
                  _SOURCE_SINK_TIME_PARAM, _SOURCE_SINK_TIME_PARAM_REQUIRED,
              ),
              ("process", "sink", "param"),
          ))
    _emit(provider, "solve_data/process_source_sourceSinkPeriodParam_in_use.csv",
          _rows_to_frame_3(
              _derive_pss_param_in_use(
                  sources, p_with_min_load,
                  _SOURCE_SINK_PERIOD_PARAM, _SOURCE_SINK_PERIOD_PARAM_REQUIRED,
              ),
              ("process", "source", "param"),
          ))
    _emit(provider, "solve_data/process_sink_sourceSinkPeriodParam_in_use.csv",
          _rows_to_frame_3(
              _derive_pss_param_in_use(
                  sinks, p_with_min_load,
                  _SOURCE_SINK_PERIOD_PARAM, _SOURCE_SINK_PERIOD_PARAM_REQUIRED,
              ),
              ("process", "sink", "param"),
          ))


# ---- write_node_group_dispatch_process_fully_inside (mod L1789-1794) ------

def derive_node_group_dispatch_process_fully_inside(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """For each ``g ∈ nodeGroupDispatch`` × ``p ∈ process``, include if
    BOTH some source and some sink of ``p`` are in ``group__node[g]``
    AND ``p`` is not a self-loop (no ``(p, n, n)`` in ``process_source_sink``).
    """
    ngd = _read_singles_list(
        input_dir / "nodeGroupDispatch.csv", provider=provider,
    )
    procs = _read_singles_list(
        input_dir / "process.csv", provider=provider,
    )
    process_source_pairs = _read_n_col_rows(
        input_dir / "process__source.csv", ["process", "source"],
        provider=provider,
    )
    process_sink_pairs = _read_n_col_rows(
        input_dir / "process__sink.csv", ["process", "sink"],
        provider=provider,
    )
    gn = _read_n_col_rows(
        input_dir / "group__node.csv", ["group", "node"], provider=provider,
    )
    triples = _read_n_col_rows(
        solve_data_dir / "process_source_sink.csv",
        ["process", "source", "sink"],
        provider=provider,
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


def emit_node_group_dispatch_process_fully_inside(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``node_group_dispatch_process_fully_inside`` to the Provider."""
    _emit(provider, "solve_data/nodeGroupDispatch__process_fully_inside.csv",
          derive_node_group_dispatch_process_fully_inside(
              input_dir, solve_data_dir, provider=provider,
          ))


# ===========================================================================
# Phase 1 follow-up 5 — small_set_derivations + arc-union small writers
# ===========================================================================

# Helpers used by the small writers below.  These are byte-for-byte parity
# with the legacy ``_read_singles`` / ``_read_pairs`` / ``_write_csv``
# helpers in ``process_arc_unions``; we keep them local to this module
# rather than reaching into the legacy module so the native port has no
# transitive import from preprocessing.


def _read_singles_csv(path: Path,
                       *,
                       provider: "object | None" = None) -> list[str]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[str] = []
    for row in df.iter_rows():
        if not row:
            continue
        c0 = _cell_str(row[0])
        if c0:
            out.append(c0)
    return out


def _read_pairs_csv(path: Path,
                     *,
                     provider: "object | None" = None) -> list[tuple[str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str]] = []
    for row in df.iter_rows():
        if len(row) >= 2:
            c0, c1 = _cell_str(row[0]), _cell_str(row[1])
            if c0 and c1:
                out.append((c0, c1))
    return out


def _read_n_col_csv(path: Path, n: int,
                     *,
                     provider: "object | None" = None) -> list[tuple[str, ...]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, ...]] = []
    for row in df.iter_rows():
        if len(row) >= n:
            cells = tuple(_cell_str(row[i]) for i in range(n))
            if all(cells):
                out.append(cells)
    return out


def _rows_to_frame(rows, header: tuple[str, ...]) -> pl.DataFrame:
    """Build an all-Utf8 ``pl.DataFrame`` from rows + a header tuple.

    Header becomes column names; each tuple element a string cell.
    Uses column-of-tuples projection so empty-row frames still carry
    the requested schema.
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

def derive_ed_history_realized(solve_data_dir: Path,
                                 *,
                                 provider: "object | None" = None,
                                 ) -> pl.DataFrame:
    """Order-preserving union of realized-existing pair set +
    ``ed_history_realized_first``, projected to (entity, period).

    Phase 4.2-1c — the realized-existing pair set is now sourced from
    the ``handoff/realized_existing`` carrier (cross-solve handoff
    translator pipeline) rather than ``solve_data/
    p_entity_period_existing_capacity``.  The handoff carries the
    cumulative realized existing capacity per (entity, period); its
    unique pairs are the realized history.  Legacy parity: no value
    filter — every row contributes its (entity, period) pair, matching
    ``_read_pairs_csv`` behaviour on the legacy CSV.

    ``ed_history_realized_first`` stays sourced from ``solve_data/``
    (per-iter input, not a cross-iter handoff carrier).
    """
    seen_ed: dict[tuple[str, str], None] = {}
    realized_existing = read_handoff_frame(provider, K.HANDOFF_REALIZED_EXISTING)
    if realized_existing is not None and realized_existing.height > 0:
        for row in realized_existing.iter_rows(named=True):
            seen_ed.setdefault((row["entity"], row["period"]), None)
    ed_first = _read_pairs_csv(
        solve_data_dir / "ed_history_realized_first.csv", provider=provider,
    )
    for r in ed_first:
        seen_ed.setdefault(r, None)
    return _rows_to_frame(list(seen_ed.keys()), ("entity", "period"))


def derive_process_source_sink_profile_method(
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """4-way union of the *profile_method* sub-CSVs (5-col frame)."""
    seen_pf: dict[tuple[str, ...], None] = {}
    for fname in (
        "process__profileProcess__toSink__profile__profile_method.csv",
        "process__source__toProfileProcess__profile__profile_method.csv",
        "process__source__sink__profile__profile_method_connection.csv",
        "process__source__sink__profile__profile_method_direct.csv",
    ):
        for r in _read_n_col_csv(
            solve_data_dir / fname, 5, provider=provider,
        ):
            seen_pf.setdefault(r, None)
    return _rows_to_frame(
        list(seen_pf.keys()),
        ("process", "source", "sink", "profile", "profile_method"),
    )


def derive_process_sinkIsNode_2way1var(
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """Projection of column 0 of
    ``process__source__sinkIsNode_2way1var.csv``."""
    triples = _read_n_col_csv(
        solve_data_dir / "process__source__sinkIsNode_2way1var.csv", 3,
        provider=provider,
    )
    seen_p: dict[str, None] = {}
    for p, _, _ in triples:
        seen_p.setdefault(p, None)
    return _rows_to_frame([(p,) for p in seen_p.keys()], ("process",))


def derive_nodeSelfDischarge(solve_data_dir: Path,
                              *,
                              provider: "object | None" = None,
                              ) -> pl.DataFrame:
    """Subset of nodeState whose ``pdtNode[n, 'self_discharge_loss', d, t]``
    is non-zero for at least one (d, t).
    """
    nodeState = frozenset(_read_singles_csv(
        solve_data_dir / "nodeState.csv", provider=provider,
    ))
    nodes_with_selfdischarge: set[str] = set()
    pdtn_path = solve_data_dir / "pdtNode.csv"
    _df = provider.get(_provider_key(pdtn_path))
    if _df is not None:
        for r in _df.iter_rows():
            if (len(r) >= 5 and _cell_str(r[0]) in nodeState
                    and _cell_str(r[1]) == "self_discharge_loss"):
                try:
                    if float(r[4]) != 0.0:
                        nodes_with_selfdischarge.add(_cell_str(r[0]))
                except (ValueError, TypeError):
                    continue
    rows = [
        (n,)
        for n in _read_singles_csv(
            solve_data_dir / "nodeState.csv", provider=provider,
        )
        if n in nodes_with_selfdischarge
    ]
    return _rows_to_frame(rows, ("node",))


def _scan_pd_startup(solve_data_dir: Path,
                      *,
                      provider: "object | None" = None,
                      ) -> set[tuple[str, str]]:
    """(process, period) pairs where ``pdProcess[p, 'startup_cost', d]`` != 0."""
    pd_startup: set[tuple[str, str]] = set()
    pdp_path = solve_data_dir / "pdProcess.csv"
    df = provider.get(_provider_key(pdp_path))
    if df is not None:
        for r in df.iter_rows():
            if len(r) >= 4:
                c0, c1, c2 = _cell_str(r[0]), _cell_str(r[1]), _cell_str(r[2])
                if c0 and c1 == "startup_cost" and c2:
                    try:
                        if float(r[3]) != 0.0:
                            pd_startup.add((c0, c2))
                    except (ValueError, TypeError):
                        continue
    return pd_startup


def _derive_pdt_online(
    solve_data_dir: Path, processes_csv: str,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    pd_startup = _scan_pd_startup(solve_data_dir, provider=provider)
    dt_pairs = _read_n_col_csv(
        solve_data_dir / "steps_in_use.csv", 2, provider=provider,
    )
    procs = _read_singles_csv(
        solve_data_dir / processes_csv, provider=provider,
    )
    rows: list[tuple[str, str, str]] = []
    for p in procs:
        for d, t in dt_pairs:
            if (p, d) in pd_startup:
                rows.append((p, d, t))
    return _rows_to_frame(rows, ("process", "period", "time"))


def derive_pdt_online_linear(solve_data_dir: Path,
                              *,
                              provider: "object | None" = None) -> pl.DataFrame:
    """``pdt_online_linear`` — process_online_linear × dt gated by startup_cost!=0."""
    return _derive_pdt_online(
        solve_data_dir, "process_online_linear.csv", provider=provider,
    )


def derive_pdt_online_integer(solve_data_dir: Path,
                               *,
                               provider: "object | None" = None) -> pl.DataFrame:
    """``pdt_online_integer`` — process_online_integer × dt gated by startup_cost!=0."""
    return _derive_pdt_online(
        solve_data_dir, "process_online_integer.csv", provider=provider,
    )


def emit_small_set_derivations(solve_data_dir: Path,
                                 *, provider) -> None:
    """Emit the small per-solve set derivations consumed downstream."""
    _emit(provider, "solve_data/ed_history_realized.csv",
          derive_ed_history_realized(solve_data_dir, provider=provider))
    _emit(provider,
          "solve_data/process__source__sink__profile__profile_method.csv",
          derive_process_source_sink_profile_method(
              solve_data_dir, provider=provider,
          ))
    _emit(provider, "solve_data/process_sinkIsNode_2way1var.csv",
          derive_process_sinkIsNode_2way1var(solve_data_dir, provider=provider))
    _emit(provider, "solve_data/nodeSelfDischarge.csv",
          derive_nodeSelfDischarge(solve_data_dir, provider=provider))
    _emit(provider, "solve_data/pdt_online_linear.csv",
          derive_pdt_online_linear(solve_data_dir, provider=provider))
    _emit(provider, "solve_data/pdt_online_integer.csv",
          derive_pdt_online_integer(solve_data_dir, provider=provider))


# ---- write_gdt_instant_flow_sets (mod L1131-1132) -------------------------

def _scan_gdt_instant_flow_rows(
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """One scan over pdtGroup.csv, splitting max/min_instant_flow rows."""
    max_rows: list[tuple[str, str, str]] = []
    min_rows: list[tuple[str, str, str]] = []
    pdtg_path = solve_data_dir / "pdtGroup.csv"
    df = provider.get(_provider_key(pdtg_path))
    if df is not None:
        for r in df.iter_rows():
            if len(r) >= 5:
                c0, c2, c3 = _cell_str(r[0]), _cell_str(r[2]), _cell_str(r[3])
                if c0 and c2 and c3:
                    try:
                        v = float(r[4])
                    except (ValueError, TypeError):
                        continue
                    if v == 0.0:
                        continue
                    c1 = _cell_str(r[1])
                    if c1 == "max_instant_flow":
                        max_rows.append((c0, c2, c3))
                    elif c1 == "min_instant_flow":
                        min_rows.append((c0, c2, c3))
    return max_rows, min_rows


def derive_gdt_max_instant_flow(solve_data_dir: Path,
                                  *,
                                  provider: "object | None" = None,
                                  ) -> pl.DataFrame:
    """``gdt_maxInstantFlow`` — pdtGroup rows with param=max_instant_flow."""
    max_rows, _ = _scan_gdt_instant_flow_rows(solve_data_dir, provider=provider)
    return _rows_to_frame(max_rows, ("group", "period", "time"))


def derive_gdt_min_instant_flow(solve_data_dir: Path,
                                  *,
                                  provider: "object | None" = None,
                                  ) -> pl.DataFrame:
    """``gdt_minInstantFlow`` — pdtGroup rows with param=min_instant_flow."""
    _, min_rows = _scan_gdt_instant_flow_rows(solve_data_dir, provider=provider)
    return _rows_to_frame(min_rows, ("group", "period", "time"))


def emit_gdt_instant_flow_sets(
    solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``gdt_maxInstantFlow`` and ``gdt_minInstantFlow``."""
    max_rows, min_rows = _scan_gdt_instant_flow_rows(
        solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/gdt_maxInstantFlow.csv",
          _rows_to_frame(max_rows, ("group", "period", "time")))
    _emit(provider, "solve_data/gdt_minInstantFlow.csv",
          _rows_to_frame(min_rows, ("group", "period", "time")))


# ---- write_p_process_delay_weight (mod L1096-1099) ------------------------

def derive_p_process_delay_weight(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``p_process_delay_weight`` 3-col frame; see writer docstring."""
    delayed_duration = _read_pairs_csv(
        solve_data_dir / "process_delayed__duration.csv", provider=provider,
    )
    delay_single = frozenset(
        _read_pairs_csv(
            input_dir / "process_delay_single.csv", provider=provider,
        )
    )
    weighted: dict[tuple[str, str], float] = {}
    pdw_path = input_dir / "p_process_delay_weighted.csv"
    _df = provider.get(_provider_key(pdw_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1:
                    try:
                        weighted[(c0, c1)] = float(r[2])
                    except (ValueError, TypeError):
                        continue
    rows: list[tuple[str, str, str]] = []
    for p, td in delayed_duration:
        v = 1.0 if (p, td) in delay_single else weighted.get((p, td), 0.0)
        rows.append((p, td, repr(v)))
    return _rows_to_frame(rows, ("process", "delay_duration", "value"))


def emit_p_process_delay_weight(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``p_process_delay_weight`` to the Provider."""
    _emit(provider, "solve_data/p_process_delay_weight.csv",
          derive_p_process_delay_weight(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- write_gcndt_co2_price (mod L1542-1548) -------------------------------

def derive_gcndt_co2_price(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``gcndt_co2_price`` 5-col frame; see writer docstring."""
    g_co2_price = frozenset(
        _read_singles_csv(
            solve_data_dir / "group_co2_price.csv", provider=provider,
        )
    )
    cn = _read_pairs_csv(
        input_dir / "commodity__node.csv", provider=provider,
    )

    gn_acc: dict[str, set[str]] = {}
    for g, n in _read_pairs_csv(
        input_dir / "group__node.csv", provider=provider,
    ):
        gn_acc.setdefault(g, set()).add(n)

    p_commodity_co2: dict[str, float] = {}
    pc_path = input_dir / "p_commodity.csv"
    _df = provider.get(_provider_key(pc_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1 == "co2_content":
                    try:
                        p_commodity_co2[c0] = float(r[2])
                    except (ValueError, TypeError):
                        continue

    co2_price_dt: set[tuple[str, str, str]] = set()
    pdtg_path = solve_data_dir / "pdtGroup.csv"
    _df = provider.get(_provider_key(pdtg_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 5:
                c0, c2, c3 = _cell_str(r[0]), _cell_str(r[2]), _cell_str(r[3])
                if c0 and _cell_str(r[1]) == "co2_price" and c2 and c3:
                    try:
                        if float(r[4]) != 0.0:
                            co2_price_dt.add((c0, c2, c3))
                    except (ValueError, TypeError):
                        continue

    dt_pairs = _read_n_col_csv(
        solve_data_dir / "steps_in_use.csv", 2, provider=provider,
    )

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


def emit_gcndt_co2_price(input_dir: Path, solve_data_dir: Path,
                          *, provider) -> None:
    """Emit ``gcndt_co2_price`` to the Provider."""
    _emit(provider, "solve_data/gcndt_co2_price.csv",
          derive_gcndt_co2_price(input_dir, solve_data_dir, provider=provider))


# ---- write_group_commodity_node_period_co2_period (mod L1550-1555) --------

def derive_group_commodity_node_period_co2_period(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``group_commodity_node_period_co2_period`` 4-col frame.

    See :func:`write_group_commodity_node_period_co2_period`.
    """
    g_co2_max_period = frozenset(
        _read_singles_csv(
            solve_data_dir / "group_co2_max_period.csv", provider=provider,
        )
    )
    cn = _read_pairs_csv(
        input_dir / "commodity__node.csv", provider=provider,
    )

    gn_acc: dict[str, set[str]] = {}
    for g, n in _read_pairs_csv(
        input_dir / "group__node.csv", provider=provider,
    ):
        gn_acc.setdefault(g, set()).add(n)

    p_commodity_co2: dict[str, float] = {}
    pc_path = input_dir / "p_commodity.csv"
    _df = provider.get(_provider_key(pc_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1 == "co2_content":
                    try:
                        p_commodity_co2[c0] = float(r[2])
                    except (ValueError, TypeError):
                        continue

    period_in_use = _read_singles_csv(
        solve_data_dir / "period_in_use_set.csv", provider=provider,
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
    return _rows_to_frame(rows, ("group", "commodity", "node", "period"))


def emit_group_commodity_node_period_co2_period(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``group_commodity_node_period_co2_period`` to the Provider."""
    _emit(provider, "solve_data/group_commodity_node_period_co2_period.csv",
          derive_group_commodity_node_period_co2_period(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- write_peedt (mod L1084) ----------------------------------------------

def derive_peedt(solve_data_dir: Path,
                  *,
                  provider: "object | None" = None) -> pl.DataFrame:
    """``peedt = process_source_sink × steps_in_use`` (5-col frame).

    Hot-path for full-year fixtures — up to ~280k rows.
    """
    triples = _read_n_col_csv(
        solve_data_dir / "process_source_sink.csv", 3, provider=provider,
    )
    dt_pairs = _read_n_col_csv(
        solve_data_dir / "steps_in_use.csv", 2, provider=provider,
    )
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


def emit_peedt(solve_data_dir: Path,
                *, provider) -> None:
    """Emit ``peedt`` — period×entity×entity×dt index frame."""
    _emit(provider, "solve_data/peedt.csv",
          derive_peedt(solve_data_dir, provider=provider))


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
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``p_flow_min`` 6-col frame; see writer docstring."""
    sinkIsNode = frozenset(_read_n_col_csv(
        solve_data_dir / "process__source__sinkIsNode_2way1var.csv", 3,
        provider=provider,
    ))
    cols = ("process", "source", "sink", "period", "time", "value")
    if not sinkIsNode:
        return _rows_to_frame([], cols)

    dcm: dict[tuple[str, str], float] = {}
    pdcm_path = solve_data_dir / "p_entity_dispatch_capacity_max.csv"
    _df = provider.get(_provider_key(pdcm_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1:
                    try:
                        dcm[(c0, c1)] = float(r[2])
                    except (ValueError, TypeError):
                        continue
    unitsize: dict[str, float] = {}
    pus_path = solve_data_dir / "p_entity_unitsize.csv"
    _df = provider.get(_provider_key(pus_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 2:
                c0 = _cell_str(r[0])
                if c0:
                    try:
                        unitsize[c0] = float(r[1])
                    except (ValueError, TypeError):
                        continue

    peedt = _read_n_col_csv(
        solve_data_dir / "peedt.csv", 5, provider=provider,
    )
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


def emit_p_flow_min(input_dir: Path, solve_data_dir: Path,
                     *, provider) -> None:
    """Emit ``p_flow_min`` to the Provider."""
    _emit(provider, "solve_data/p_flow_min.csv",
          derive_p_flow_min(input_dir, solve_data_dir, provider=provider))


# ---- write_p_flow_max (mod L1661-1677) ------------------------------------


def derive_p_flow_max(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``p_flow_max`` 6-col frame; see :func:`write_p_flow_max`."""
    coeff_zero = frozenset(_read_n_col_csv(
        solve_data_dir / "process_source_sink_coeff_zero.csv", 3,
        provider=provider,
    ))
    has_indirect = frozenset(
        p for p, _m in _read_pairs_csv(
            solve_data_dir / "process__method_indirect.csv",
            provider=provider,
        )
    )
    process_source = frozenset(_read_pairs_csv(
        input_dir / "process__source.csv", provider=provider,
    ))
    process_sink = frozenset(_read_pairs_csv(
        input_dir / "process__sink.csv", provider=provider,
    ))
    has_min_load = frozenset(
        p for p, m in _read_pairs_csv(
            solve_data_dir / "process__ct_method.csv", provider=provider,
        )
        if m == "min_load_efficiency"
    )

    dcm: dict[tuple[str, str], float] = {}
    pdcm_path = solve_data_dir / "p_entity_dispatch_capacity_max.csv"
    _df = provider.get(_provider_key(pdcm_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1:
                    try:
                        dcm[(c0, c1)] = float(r[2])
                    except (ValueError, TypeError):
                        continue
    unitsize: dict[str, float] = {}
    pus_path = solve_data_dir / "p_entity_unitsize.csv"
    _df = provider.get(_provider_key(pus_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 2:
                c0 = _cell_str(r[0])
                if c0:
                    try:
                        unitsize[c0] = float(r[1])
                    except (ValueError, TypeError):
                        continue

    slope: dict[tuple[str, str, str], float] = {}
    section: dict[tuple[str, str, str], float] = {}
    for fname, target in (
        ("pdtProcess_slope.csv", slope),
        ("pdtProcess_section.csv", section),
    ):
        path = solve_data_dir / fname
        _df = provider.get(_provider_key(path))
        if _df is not None:
            for r in _df.iter_rows():
                if len(r) >= 4:
                    c0, c1, c2 = (
                        _cell_str(r[0]), _cell_str(r[1]), _cell_str(r[2]),
                    )
                    if c0 and c1 and c2:
                        try:
                            target[(c0, c1, c2)] = float(r[3])
                        except (ValueError, TypeError):
                            continue

    src_max_coef: dict[tuple[str, str], float] = {}
    pms_path = input_dir / "p_process_source_capacity_max_coeff.csv"
    _df = provider.get(_provider_key(pms_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1:
                    try:
                        src_max_coef[(c0, c1)] = float(r[2])
                    except (ValueError, TypeError):
                        continue
    sink_max_coef: dict[tuple[str, str], float] = {}
    pmk_path = input_dir / "p_process_sink_capacity_max_coeff.csv"
    _df = provider.get(_provider_key(pmk_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 3:
                c0, c1 = _cell_str(r[0]), _cell_str(r[1])
                if c0 and c1:
                    try:
                        sink_max_coef[(c0, c1)] = float(r[2])
                    except (ValueError, TypeError):
                        continue

    # p_unconstrained_flow_cap = max over models of
    # p_max_flow_for_unconstrained_variables[m]; default 1e6 if absent.
    p_uflow = 1_000_000.0
    pmfu_path = input_dir / "p_max_flow_for_unconstrained_variables.csv"
    _df = provider.get(_provider_key(pmfu_path))
    if _df is not None:
        max_v: float | None = None
        for r in _df.iter_rows():
            if len(r) >= 2 and _cell_str(r[0]):
                try:
                    v = float(r[1])
                except (ValueError, TypeError):
                    continue
                if max_v is None or v > max_v:
                    max_v = v
        if max_v is not None:
            p_uflow = max_v

    peedt = _read_n_col_csv(
        solve_data_dir / "peedt.csv", 5, provider=provider,
    )
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


def emit_p_flow_max(input_dir: Path, solve_data_dir: Path,
                     *, provider) -> None:
    """Emit ``p_flow_max`` to the Provider."""
    _emit(provider, "solve_data/p_flow_max.csv",
          derive_p_flow_max(input_dir, solve_data_dir, provider=provider))


# ---- write_p_state_slack_share (mod L1689-1691) ---------------------------


def derive_p_state_slack_share(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``p_state_slack_share`` 5-col frame; see writer docstring."""
    g_loss = frozenset(
        _read_singles_csv(
            solve_data_dir / "group_loss_share.csv", provider=provider,
        )
    )
    g_type: dict[str, str] = {}
    for g, ty in _read_pairs_csv(
        input_dir / "group__loss_share_type.csv", provider=provider,
    ):
        g_type[g] = ty
    nodes_in_g: dict[str, list[str]] = {}
    for g, n in _read_pairs_csv(
        input_dir / "group__node.csv", provider=provider,
    ):
        nodes_in_g.setdefault(g, []).append(n)
    inflow: dict[tuple[str, str, str], float] = {}
    pdtni_path = solve_data_dir / "pdtNodeInflow.csv"
    _df = provider.get(_provider_key(pdtni_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 4:
                c0, c1, c2 = _cell_str(r[0]), _cell_str(r[1]), _cell_str(r[2])
                if c0 and c1 and c2:
                    try:
                        inflow[(c0, c1, c2)] = float(r[3])
                    except (ValueError, TypeError):
                        continue
    dt_pairs = _read_n_col_csv(
        solve_data_dir / "steps_in_use.csv", 2, provider=provider,
    )

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


def emit_p_state_slack_share(input_dir: Path, solve_data_dir: Path,
                              *, provider) -> None:
    """Emit ``p_state_slack_share`` to the Provider."""
    _emit(provider, "solve_data/p_state_slack_share.csv",
          derive_p_state_slack_share(input_dir, solve_data_dir,
                                       provider=provider))


# ---- write_p_storage_state_reference_price (mod L1693-1698) ---------------


def derive_p_storage_state_reference_price(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> pl.DataFrame:
    """``p_storage_state_reference_price`` 3-col frame; see writer docstring."""
    # (n, d2, t2) → value, keyed by (node, period, step) from
    # ``handoff/fix_storage_price`` (canonical schema
    # ``[node, period, step, p_fix_storage_price]``).  Phase 4.1f —
    # replaces the legacy ``solve_data/fix_storage_price.csv`` Provider
    # read; the translator seeds the handoff key at iteration start
    # (parent's data shadowing sequential when nested).
    from flextool.engine_polars import _provider_keys as K
    from flextool.engine_polars._provider_translators import (
        read_handoff_frame,
    )
    fix_price: dict[tuple[str, str, str], float] = {}
    fsp_df = read_handoff_frame(provider, K.HANDOFF_FIX_STORAGE_PRICE)
    if fsp_df is not None and fsp_df.height > 0:
        for n_, d_, t_, v_ in fsp_df.select(
            "node", "period", "step", "p_fix_storage_price",
        ).iter_rows():
            if n_ and d_ and t_ and v_ is not None and v_ != "":
                try:
                    fix_price[(n_, d_, t_)] = float(v_)
                except (ValueError, TypeError):
                    continue

    ptl = _read_pairs_csv(
        solve_data_dir / "last_timesteps.csv", provider=provider,
    )
    ptl_for_d: dict[str, list[str]] = {}
    for d, t in ptl:
        ptl_for_d.setdefault(d, []).append(t)
    pb_d2_for_d: dict[str, list[str]] = {}
    for d2, d in _read_pairs_csv(
        solve_data_dir / "period__branch.csv", provider=provider,
    ):
        pb_d2_for_d.setdefault(d, []).append(d2)
    dtt_for_dt: dict[tuple[str, str], list[str]] = {}
    for d, t, t2 in _read_n_col_csv(
        solve_data_dir / "timeline_matching_map.csv", 3, provider=provider,
    ):
        dtt_for_dt.setdefault((d, t), []).append(t2)

    use_ref = frozenset(
        n for n, m in _read_pairs_csv(
            input_dir / "node__storage_solve_horizon_method.csv",
            provider=provider,
        ) if m == "use_reference_price"
    )

    pd_ref_price: dict[tuple[str, str], float] = {}
    pdn_path = solve_data_dir / "pdNode.csv"
    _df = provider.get(_provider_key(pdn_path))
    if _df is not None:
        for r in _df.iter_rows():
            if len(r) >= 4:
                c0, c2 = _cell_str(r[0]), _cell_str(r[2])
                if (c0
                        and _cell_str(r[1]) == "storage_state_reference_price"
                        and c2):
                    try:
                        pd_ref_price[(c0, c2)] = float(r[3])
                    except (ValueError, TypeError):
                        continue

    nodes_state = _read_singles_csv(
        solve_data_dir / "nodeState.csv", provider=provider,
    )
    period_in_use = _read_singles_csv(
        solve_data_dir / "period_in_use_set.csv", provider=provider,
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


def emit_p_storage_state_reference_price(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``p_storage_state_reference_price`` to the Provider."""
    _emit(provider, "solve_data/p_storage_state_reference_price.csv",
          derive_p_storage_state_reference_price(
              input_dir, solve_data_dir, provider=provider,
          ))


# ---- write_node_group_dispatch_sets (mod L1596-1657) ----------------------


def _compute_node_group_dispatch_sets(
    input_dir: Path, solve_data_dir: Path,
    *,
    provider: "object | None" = None,
) -> dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]:
    """One shared scan; returns ``{filename → (header, rows)}`` for the
    12 nodeGroupDispatch CSVs.
    """
    ngd = _read_singles_csv(
        input_dir / "nodeGroupDispatch.csv", provider=provider,
    )
    fag = frozenset(_read_singles_csv(
        input_dir / "flowAggregator.csv", provider=provider,
    ))
    p_unit = frozenset(_read_singles_csv(
        input_dir / "process_unit.csv", provider=provider,
    ))
    p_conn = frozenset(_read_singles_csv(
        input_dir / "process_connection.csv", provider=provider,
    ))

    g_nodes_acc: dict[str, dict[str, None]] = {}
    for g, n in _read_pairs_csv(
        input_dir / "group__node.csv", provider=provider,
    ):
        g_nodes_acc.setdefault(g, {})[n] = None
    g_nodes: dict[str, frozenset[str]] = {
        g: frozenset(d.keys()) for g, d in g_nodes_acc.items()
    }

    # group_process_node restricted to flowAggregator groups: (p, n) → [ga, ...]
    pn_to_aggregators: dict[tuple[str, str], list[str]] = {}
    for g, p, n in _read_n_col_csv(
        input_dir / "group__process__node.csv", 3, provider=provider,
    ):
        if g in fag:
            pn_to_aggregators.setdefault((p, n), []).append(g)

    pss_always = _read_n_col_csv(
        solve_data_dir / "process_source_sink_alwaysProcess.csv", 3,
        provider=provider,
    )
    fully_inside = frozenset(_read_pairs_csv(
        solve_data_dir / "nodeGroupDispatch__process_fully_inside.csv",
        provider=provider,
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


def emit_node_group_dispatch_sets(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``node_group_dispatch_sets`` to the Provider."""
    by_file = _compute_node_group_dispatch_sets(
        input_dir, solve_data_dir, provider=provider,
    )
    for fname, (header, rows) in by_file.items():
        _emit(provider, f"solve_data/{fname}", _rows_to_frame(rows, header))


