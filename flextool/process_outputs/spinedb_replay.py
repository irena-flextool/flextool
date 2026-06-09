"""Reconstruct the minimal engine ``s`` (sets) namespace needed by the
``spinedb`` writer from PROCESSED ``output_parquet`` results alone.

This supports the parquet-replay path of the ``spinedb`` output write-method:
build the results SpineDB from a previously processed ``output_parquet``
bundle with NO re-solve (and without ``output_raw`` or the input DB).

``write_spinedb`` consumes exactly five attributes of ``s`` (verified in
``write_spinedb.py``):

* ``s.solve_period_time`` — ``pd.MultiIndex(['solve','period','time'])``,
  read as a bare attribute in ``_ensure_solve_level`` (:737) to attach a
  ``solve`` level to every ``_dt`` frame lacking one.
* ``s.solve_period`` — ``pd.MultiIndex(['solve','period'])``, read as a bare
  attribute in ``_ensure_solve_level`` (:750-751) and ``_discount_factor_map``
  (:864) to attach a ``solve`` level to ``_d`` frames.
* ``s.process_unit`` — ``pd.Index(name='process')``, via ``getattr`` in
  ``_iter_columns`` (:560); the set of unit process names.
* ``s.process_connection`` — ``pd.Index(name='process')``, via ``getattr`` in
  ``_iter_columns`` (:561) and ``_connection_triple_lookup`` (:782); the set
  of connection process names.
* ``s.process_source_sink`` — ``pd.MultiIndex(['process','source','sink'])``,
  iterated as ``(process, source, sink)`` tuples in
  ``_connection_triple_lookup`` (:781) to form ``connection__node__node``
  bynames.

All structures are rebuilt purely from the ``results`` dict (pure function,
no randomness) and sorted canonically for cross-process determinism.

The two discount-factor params (#33/#34) are NOT recoverable from the
processed bundle; on the replay path ``par`` is ``None`` and the writer
skips them with a debug log.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

logger = logging.getLogger(__name__)


# Frames whose first column level (or, for ``_ed_p`` capacity frames, the
# entity index level) carries the unit / connection process name.
_UNIT_PREFIX = "unit_"
_CONNECTION_PREFIX = "connection_"

# Net-flow frames (column = bare connection name, value = signed net flow,
# positive = first→second node) used to detect bidirectional (2-way)
# connections on the replay path.
_NET_FLOW_KEYS = ("connection_dt_eee", "connection_d_eee")
# Net-flow magnitude below which a sign is treated as numerical noise (not a
# genuine reverse flow).
_FLOW_SIGN_EPS = 1e-9


def _frame_column_labels(df) -> list:
    """Return the column labels of ``df`` as a flat list, transparently
    handling the squeezed-``Series`` case.

    ``read_lean_parquet`` (via ``write_outputs``) squeezes any frame whose
    column axis is a single level / single column down to a ``pandas.Series``;
    such a Series has NO ``.columns`` attribute but preserves the original
    single column label in ``.name`` (a scalar for a 1-level frame, or a tuple
    for a multi-level frame squeezed to one column).  Scanning only
    ``df.columns`` would therefore silently drop the process carried by such a
    squeezed frame.  This helper yields the label list for both shapes so the
    process set / node map can never under-cover."""
    cols = getattr(df, "columns", None)
    if cols is not None:
        return list(cols)
    # Squeezed Series — the (only) column label survives in ``.name``.
    name = getattr(df, "name", None)
    if name is None:
        return []
    return [name]


def build_replay_s(results: dict) -> SimpleNamespace:
    """Build the minimal ``s`` shim for ``write_spinedb`` from ``results``.

    Parameters
    ----------
    results : dict[str, pandas.DataFrame | pandas.Series]
        Processed results keyed by RAW engine table name, exactly as the
        ``read_parquet_dir`` branch of ``write_outputs`` rebuilds them
        (``scenario`` column level already stripped / single levels
        squeezed).

    Returns
    -------
    types.SimpleNamespace
        Object exposing ``solve_period_time``, ``solve_period``,
        ``process_unit``, ``process_connection`` and ``process_source_sink``
        with the exact shapes ``write_spinedb`` expects, plus a
        ``replay_solve_name`` convenience attribute (the canonical solve
        label recovered from ``node_prices_dt_e``, or ``None`` when no solve
        axis is recoverable) so the replay caller can pass the real solve
        name to ``write_spinedb`` for the ``collapse_solve`` frames whose
        single ``solve`` Map key is taken from ``solve_name``.
    """
    solve_period_time, solve_period = _build_solve_indexes(results)
    process_unit = _collect_process_names(results, _UNIT_PREFIX)
    process_connection = _collect_process_names(results, _CONNECTION_PREFIX)
    process_source_sink = _build_process_source_sink(results)
    replay_solve_name = _last_solve_name(solve_period)

    return SimpleNamespace(
        solve_period_time=solve_period_time,
        solve_period=solve_period,
        process_unit=process_unit,
        process_connection=process_connection,
        process_source_sink=process_source_sink,
        replay_solve_name=replay_solve_name,
    )


def _last_solve_name(solve_period: pd.MultiIndex):
    """Return the last solve label in ``solve_period`` (creation order), or
    ``None`` when the index is empty.

    Frames written under ``collapse_solve`` (``costs_discounted_d_p``) carry
    no per-row solve axis, so the writer wraps the whole frame under a single
    ``solve_name`` key.  On the native path that key is the final cascade
    ``solve_name``; the final solve label in ``node_prices`` (= last creating
    solve, on-disk row order) reproduces it, so the parquet path emits the
    same ``solve`` Map key."""
    if solve_period is None or len(solve_period) == 0:
        return None
    return str(solve_period.get_level_values("solve")[-1])


# ---------------------------------------------------------------------------
# Solve reconstruction from node_prices_dt_e
# ---------------------------------------------------------------------------

def _build_solve_indexes(results: dict):
    """Return ``(solve_period_time, solve_period)`` MultiIndexes from the
    ``(solve, period, time)`` index of ``node_prices_dt_e``.

    Native ``s.solve_period_time`` / ``s.solve_period`` are the realized
    ``(solve, period, time)`` / ``(solve, period)`` index deduped on
    ``(period, time)`` / ``period`` with ``keep='last'`` (the last creating
    solve wins).  ``node_prices_dt_e`` retains the same pre-dedup
    ``(solve, period, time)`` index in solve-creation row order, so deduping
    it ``keep='last'`` reproduces the native winner exactly.

    When ``node_prices_dt_e`` is absent or empty, return empty MultiIndexes;
    the writer then falls back to wrapping every frame under the single
    ``solve_name`` key (``_series_to_map`` :669-673).
    """
    empty_spt = pd.MultiIndex.from_arrays(
        [[], [], []], names=["solve", "period", "time"])
    empty_sp = pd.MultiIndex.from_arrays(
        [[], []], names=["solve", "period"])

    obj = results.get("node_prices_dt_e")
    if obj is None:
        return empty_spt, empty_sp
    idx = obj.index
    if idx is None or len(idx) == 0:
        return empty_spt, empty_sp
    names = list(idx.names)
    if not {"solve", "period", "time"} <= set(names):
        logger.debug(
            "spinedb replay: node_prices_dt_e index %s lacks "
            "(solve, period, time) — solve recon degraded to single wrap",
            names)
        return empty_spt, empty_sp

    solve = idx.get_level_values("solve")
    period = idx.get_level_values("period")
    time = idx.get_level_values("time")

    # solve_period_time: dedup on (period, time), keep='last'.
    spt = pd.MultiIndex.from_arrays(
        [solve, period, time], names=["solve", "period", "time"])
    pt = pd.MultiIndex.from_arrays([period, time], names=["period", "time"])
    keep_spt = ~pt.duplicated(keep="last")
    solve_period_time = spt[keep_spt]

    # solve_period: collapse to (solve, period), dedup on period, keep='last'.
    sp_full = pd.MultiIndex.from_arrays(
        [solve, period], names=["solve", "period"])
    keep_sp = ~period.duplicated(keep="last")
    solve_period = sp_full[keep_sp]

    return solve_period_time, solve_period


# ---------------------------------------------------------------------------
# Unit / connection process-name sets
# ---------------------------------------------------------------------------

def _collect_process_names(results: dict, prefix: str) -> pd.Index:
    """Union the process names across all ``results`` frames whose key starts
    with ``prefix`` (``unit_`` or ``connection_``).

    The process name is the FIRST column level of the (scenario-stripped)
    frame — named ``unit`` / ``connection`` / ``process`` depending on the
    frame — or, for ``_ed_p`` capacity frames where the entity sits in the
    index, the index level named ``unit`` / ``connection``.  Returns a
    canonically sorted ``pd.Index(name='process')`` (matching native
    ``s.process_unit`` / ``s.process_connection``).
    """
    entity_level = "unit" if prefix == _UNIT_PREFIX else "connection"
    names: set[str] = set()

    for key, df in results.items():
        if not key.startswith(prefix):
            continue
        if df is None:
            continue
        # Capacity / indicator frames carry the entity in the index level.
        idx_names = list(getattr(df.index, "names", []) or [])
        if entity_level in idx_names:
            names.update(df.index.get_level_values(entity_level).tolist())
            continue
        # Otherwise the entity is the first column level — read it via
        # ``_frame_column_labels`` so a squeezed single-column Series (whose
        # label lives in ``.name``, not ``.columns``) still contributes its
        # process and the set never silently under-covers.
        for col in _frame_column_labels(df):
            first = col[0] if isinstance(col, tuple) else col
            names.add(first)

    return pd.Index(sorted(names), name="process")


# ---------------------------------------------------------------------------
# Connection (process, source, sink) triple
# ---------------------------------------------------------------------------

def _build_process_source_sink(results: dict) -> pd.MultiIndex:
    """Reconstruct ``(process, source, sink)`` triples for connections from
    the ``connection_leftward_*`` and ``connection_rightward_*`` column
    bynames.

    Node-order convention (verified against ``calc_connections.py`` and real
    parquet):

    * ``connection_leftward_*`` columns are ``(process, first_node)`` —
      ``first_node`` is the **source** in ``s.process_source_sink``.
    * ``connection_rightward_*`` columns are ``(process, second_node)`` —
      ``second_node`` is the **sink**.

    So the triple is ``(process, left_node, right_node)`` =
    ``(process, source, sink)``, matching the native
    ``_connection_triple_lookup`` which builds ``(process, source, sink)``
    from ``s.process_source_sink``.

    The ``_dt`` and ``_d`` variants carry identical ``(process, node)``
    columns; both are scanned and unioned so a missing variant cannot drop a
    connection.  Returns a canonically sorted
    ``pd.MultiIndex(['process','source','sink'])``.
    """
    left_map = _process_node_map(results, "leftward")   # process -> source
    right_map = _process_node_map(results, "rightward")  # process -> sink

    # Warn about any genuinely bidirectional (2-way) connection BEFORE the
    # triple is committed: its (source, sink) ordering is not recoverable
    # from the processed bundle (see ``_warn_two_way_connections``).
    _warn_two_way_connections(results, set(left_map) | set(right_map))

    triples: set[tuple[str, str, str]] = set()
    for process in set(left_map) | set(right_map):
        source = left_map.get(process)
        sink = right_map.get(process)
        if source is None or sink is None:
            logger.debug(
                "spinedb replay: connection '%s' missing %s node — triple "
                "skipped", process,
                "left/source" if source is None else "right/sink")
            continue
        triples.add((process, source, sink))

    if not triples:
        return pd.MultiIndex.from_arrays(
            [[], [], []], names=["process", "source", "sink"])

    ordered = sorted(triples)
    return pd.MultiIndex.from_tuples(
        ordered, names=["process", "source", "sink"])


def _process_node_map(results: dict, direction: str) -> dict:
    """Map ``process -> node`` from every ``connection_<direction>_*`` frame
    (``leftward`` / ``rightward``).  Columns are ``(process, node)``.

    Each connection has exactly one left node and one right node, so a single
    mapping per process is correct; the first encountered wins (frames agree).
    """
    needle = f"connection_{direction}_"
    out: dict[str, str] = {}
    for key, df in results.items():
        if not key.startswith(needle):
            continue
        if df is None:
            continue
        # ``_frame_column_labels`` covers the squeezed single-column Series
        # case (label in ``.name``), so a connection whose only directional
        # frame was squeezed still maps its node.
        for col in _frame_column_labels(df):
            if not isinstance(col, tuple) or len(col) < 2:
                # Expected (process, node); skip malformed.
                continue
            process, node = col[0], col[1]
            out.setdefault(process, node)
    return out


def _warn_two_way_connections(results: dict, connections: set) -> None:
    """Emit a runtime ``logging.warning`` for every BIDIRECTIONAL (2-way)
    connection reconstructed on the replay path, naming it.

    Why this matters
    ----------------
    For a 1-way connection the ``(source, sink)`` ordering is exact on the
    replay path: native ``s.process_source_sink`` holds the single arc
    ``(left_node, right_node)`` and the reconstruction reproduces it verbatim.

    For a TRUE 2-way connection native ``s.process_source_sink`` carries BOTH
    arcs ``(a, b)`` and ``(b, a)``; ``_connection_triple_lookup`` keeps the
    lexicographically-first.  The processed parquet retains only the
    ``(left_node, right_node)`` *geometry* (``connection_leftward_*`` /
    ``connection_rightward_*`` each name a single node) with NO surviving
    signal for the arc the writer would have picked, so the reconstructed
    ``connection__node__node`` byname may be SILENTLY swapped relative to the
    native DB.  We do not attempt to make it exact here (that needs either the
    input DB or a new persisted artifact — a decision still pending with the
    maintainer); we only make the possible swap LOUD.

    Detection predicate (parquet-only)
    ----------------------------------
    2-way-ness is not visible in the directional geometry alone — for a 1-way
    connection ``connection_leftward_*`` / ``connection_rightward_*`` BOTH
    carry the (loss-driven) directional flow components, so "appears in both
    roles" matches 1-way connections too and cannot be the test.  The one
    signal that survives into the processed bundle is the NET flow
    (``connection_dt_eee`` / ``connection_d_eee``, column = bare connection,
    value positive = first→second node): a genuinely bidirectional connection
    flows BOTH ways across the horizon, so its net flow takes BOTH signs,
    whereas a 1-way connection is strictly single-signed.  We therefore warn
    exactly for connections whose net flow has both a value ``> +eps`` and a
    value ``< -eps``.

    Known residual edge: a 2-way connection that happens to flow in only ONE
    direction across the entire realized horizon is indistinguishable from a
    1-way connection here (single-signed net flow) and is NOT warned — but in
    that case its single realized direction is exactly what the reconstruction
    emits, so the byname is at least self-consistent with the observed flow.
    Making 2-way fully exact is out of scope (no input-DB read, no new
    persisted artifact).
    """
    two_way = sorted(
        c for c in connections if _connection_is_bidirectional(results, c)
    )
    if not two_way:
        return
    logger.warning(
        "spinedb replay: %d bidirectional (2-way) connection(s) detected "
        "(%s). The (source, sink) node ordering of their "
        "'connection__node__node' byname — and therefore which physical node "
        "the flow_to_first_node / flow_to_second_node params describe — is "
        "NOT recoverable from the processed parquet bundle and may be swapped "
        "relative to a native solve. 1-way connections are exact.",
        len(two_way), ", ".join(two_way),
    )


def _connection_is_bidirectional(results: dict, connection: str) -> bool:
    """True when ``connection``'s net flow takes BOTH signs in any net-flow
    frame (``connection_dt_eee`` / ``connection_d_eee``) — the parquet-only
    signature of a genuinely 2-way connection (see
    ``_warn_two_way_connections``)."""
    for key in _NET_FLOW_KEYS:
        df = results.get(key)
        if df is None:
            continue
        series = _net_flow_series(df, connection)
        if series is None or len(series) == 0:
            continue
        has_pos = bool((series > _FLOW_SIGN_EPS).any())
        has_neg = bool((series < -_FLOW_SIGN_EPS).any())
        if has_pos and has_neg:
            return True
    return False


def _net_flow_series(df, connection: str):
    """Extract the net-flow Series for ``connection`` from a net-flow frame.

    Net-flow frames are keyed by the bare connection name (column level
    ``process``).  Handles the squeezed single-column ``Series`` case (label
    in ``.name``) and the multi-column frame case (select the column whose
    first level is ``connection``)."""
    cols = getattr(df, "columns", None)
    if cols is None:
        # Squeezed Series — usable only if its name identifies this connection.
        name = getattr(df, "name", None)
        first = name[0] if isinstance(name, tuple) else name
        return df if first == connection else None
    for col in cols:
        first = col[0] if isinstance(col, tuple) else col
        if first == connection:
            return df[col]
    return None
