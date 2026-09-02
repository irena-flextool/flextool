"""Pure post-solve readers for the adequacy calibrator.

Each function takes the directory that holds a solve's result parquets
(``<out_root>/output_parquet/<scenario>/``) and returns plain Python
dicts.  Nothing here solves, mutates the DB, or touches the network — they
are pure readers, so the loop can call them once per iteration and the
tests can point them straight at real parquets.

Why :func:`flextool.lean_parquet.read_lean_parquet` and not
``pd.read_parquet``
-------------------------------------------------------------------
FlexTool writes these tables with a compact custom footer that records the
MultiIndex level names; a raw ``pd.read_parquet`` returns flat
string-tuple column names and loses the level structure the group-bys here
rely on.  :func:`read_lean_parquet` reconstructs the real row/column
MultiIndex, so ``groupby(level=...)`` works.

Signals
-------
* ``node_slack_up_d_e`` — per-node unserved-energy up-slack (annual MWh,
  ≥0).  Dense and always present on a valid solve → the robust adequacy
  signal the loop steers on.
* ``unit_curtailment_outputNode_d_ee`` — per-(unit, sink) curtailment
  (MWh); OPTIONAL (absent when nothing curtails).  Provided now for the
  over-build guard (C1c) to consume.
* ``cost_node_discounted_d_ec`` — discounted per-(node, category) node
  cost (M€); its ``"upward slack penalty"`` category is the monetised
  slack.  This table is legitimately ABSENT on a zero-slack solve
  (out_costs.py skips an empty category) → treated as zero penalty, never
  an error.
"""

from __future__ import annotations

from pathlib import Path

from flextool.lean_parquet import read_lean_parquet

# Registry keys of the parquet tables read here.  Resolved to on-disk
# basenames through P2's registry helper so a schema rename breaks loudly
# here rather than silently missing a file.
_SLACK_KEY = "node_slack_up_d_e"
_SLACK_KEY_DT = "node_slack_up_dt_e"
_CURTAILMENT_KEY = "unit_curtailment_outputNode_d_ee"
_COST_KEY = "cost_node_discounted_d_ec"
_UNIT_CAPACITY_KEY = "unit_capacity_ed_p"
_TOTAL_COST_KEY = "costs_discounted_p_"

_SLACK_PENALTY_CATEGORY = "upward slack penalty"


def _resolve_filename(key: str) -> str:
    """Resolve an output *key* to its ``<key>.parquet`` basename.

    Prefers P2's registry helper
    (``flextool.calibrate._solve_status._registry_filename``) so an
    OUTPUT_TRANSFORM rename surfaces as a clear error; falls back to the
    ``<key>.parquet`` basename (which matches the OUTPUT_TRANSFORM keys) if
    that private helper is ever removed.
    """
    try:
        from flextool.calibrate._solve_status import _registry_filename

        return _registry_filename(key)
    except (ImportError, AttributeError):
        return f"{key}.parquet"


def read_residual_unserved(assess_dir: Path) -> dict[str, float]:
    """Return ``{node: total_unserved_MWh}`` from ``node_slack_up_d_e``.

    Sums each node's up-slack over all periods and drops the ``scenario``
    column level, so the result is keyed by node.  This is the robust,
    always-present adequacy signal the loop converges on.
    """
    path = Path(assess_dir) / _resolve_filename(_SLACK_KEY)
    df = read_lean_parquet(path)
    # columns: (scenario, node); index: period.  Sum over periods, then
    # collapse any scenario level so the key is the node name.
    totals = df.sum(axis=0).groupby(level="node").sum()
    return {str(node): float(val) for node, val in totals.items()}


def read_residual_unserved_dt(assess_dir: Path) -> dict[str, "object"]:
    """Return ``{node: DataFrame[period, time, value]}`` from ``node_slack_up_dt_e``.

    The per-``(period, time)`` companion to :func:`read_residual_unserved`:
    where that reader collapses the up-slack to one annual MWh per node, this
    one keeps the FULL per-timestep profile UNFOLDED onto the base timeline —
    the stress SHAPE the ``timed`` sizer redistributes the additive margin
    over.  ``node_slack_up_dt_e`` has row index ``(period, time)`` and column
    levels ``(scenario, node)``; per node the scenario level is collapsed
    (summed) and the result returned as a tidy pandas frame with columns
    ``period``, ``time``, ``value`` (one row per non-null base cell).

    Read with :func:`read_lean_parquet` (never ``pd.read_parquet``) so the
    ``(period, time)`` / ``(scenario, node)`` MultiIndex is reconstructed.
    A node absent from the table (no slack rows) simply has no key.
    """
    path = Path(assess_dir) / _resolve_filename(_SLACK_KEY_DT)
    df = read_lean_parquet(path)
    out: dict[str, object] = {}
    node_level = df.columns.get_level_values("node")
    for node in dict.fromkeys(node_level):  # ordered-unique node names
        sub = df.loc[:, node_level == node]
        # Collapse any scenario level → one value per (period, time) row.
        series = sub.sum(axis=1)
        frame = series.reset_index()
        frame.columns = ["period", "time", "value"]
        out[str(node)] = frame
    return out


def read_curtailment_by_sink(assess_dir: Path) -> dict[str, float]:
    """Return ``{sink_node: total_curtailment_MWh}`` or ``{}`` if absent.

    Reads ``unit_curtailment_outputNode_d_ee`` (columns
    ``(scenario, unit, sink)``), sums each column over periods and groups
    the totals by ``sink`` node.  The file is OPTIONAL (no curtailment ⇒ no
    table); a missing file yields ``{}``.  Consumed by the over-build guard
    (C1c).
    """
    path = Path(assess_dir) / _resolve_filename(_CURTAILMENT_KEY)
    if not path.is_file():
        return {}
    df = read_lean_parquet(path)
    totals = df.sum(axis=0).groupby(level="sink").sum()
    return {str(sink): float(val) for sink, val in totals.items()}


def read_slack_penalty(assess_dir: Path) -> tuple[float, dict[str, float]]:
    """Return ``(total_Meur, {node: penalty_Meur})`` from the node-cost table.

    Reads ``cost_node_discounted_d_ec`` (row index ``(period, node)``,
    columns ``(scenario, category)``), selects the
    ``"upward slack penalty"`` category and sums it per node (over periods
    and scenarios).  This table is legitimately ABSENT on a zero-slack
    solve (out_costs.py skips an empty cost category) → returns
    ``(0.0, {})`` rather than raising.
    """
    path = Path(assess_dir) / _resolve_filename(_COST_KEY)
    if not path.is_file():
        return 0.0, {}
    df = read_lean_parquet(path)
    is_penalty = df.columns.get_level_values("category") == _SLACK_PENALTY_CATEGORY
    penalty = df.loc[:, is_penalty]
    if penalty.shape[1] == 0:
        return 0.0, {}
    # One value per (period, node) row across the selected penalty
    # column(s); collapse periods (and any scenario level) per node.
    per_row = penalty.sum(axis=1)
    per_node = per_row.groupby(level="node").sum()
    by_node = {str(node): float(val) for node, val in per_node.items()}
    total = float(per_row.sum())
    return total, by_node


def read_unit_capacity_total(assess_dir: Path) -> dict[str, float]:
    """Return ``{unit: total_capacity_MW}`` from ``unit_capacity_ed_p``.

    Reads the post-invest fleet the solve actually committed to: the ``total``
    column of ``unit_capacity_ed_p`` (row index ``(unit, period)``, column
    levels ``(scenario, parameter)`` with ``parameter`` in
    ``{existing, invested, divested, total}`` — see
    ``flextool.process_outputs.out_capacity.unit_capacity``).  ``total`` is the
    CUMULATIVE ``entity_all_capacity`` (existing + carried-forward invest), so
    it is non-decreasing across periods.

    Per unit the capacity is taken as the MAX over periods — the *mature*
    (fully-built) fleet the net-load signal must be sized against; for a
    single-invest-period solve this is simply that period's total, and for a
    multi-period invest solve it is the final built-out capacity.  Any
    ``scenario`` column level is collapsed (summed; a single-scenario solve
    dir has exactly one) before the per-unit reduction.  Units whose ``total``
    is entirely null (never realized) are dropped, so a unit absent from the
    result cleanly falls back to its existing cap in
    :func:`flextool.representative_periods.netload.build_group_capacities`.
    Sorted by unit name for determinism.

    This is a plain reader (no solving / DB / network), used by the net-load
    solve-iteration driver to feed each iteration's invested caps back into the
    representative-period selection.
    """
    path = Path(assess_dir) / _resolve_filename(_UNIT_CAPACITY_KEY)
    df = read_lean_parquet(path)
    # Select the ``total`` parameter column, collapsing any scenario level so
    # the result is a Series indexed by (unit, period).
    total = df.xs("total", level="parameter", axis=1).sum(axis=1)
    per_unit = total.groupby(level="unit").max().dropna()
    return {str(unit): float(val) for unit, val in sorted(per_unit.items())}


def read_total_system_cost(assess_dir: Path) -> float:
    """Return the discounted total system cost (M€) from ``costs_discounted_p_``.

    ``costs_discounted_p_`` is the authoritative full-horizon cost summary
    (``flextool.process_outputs.out_costs.cost_summaries``): every investment
    AND dispatch cost category summed over all realized periods, discounted and
    years-represented-weighted exactly as the LP objective is.  It INCLUDES the
    ``upward slack penalty`` / ``downward slack penalty`` categories (the
    monetised unserved-energy penalties) and the negative ``commodity_sales``
    revenue term, so summing every category cell yields the same signed total
    the objective minimises — the correct comparability metric for the
    keep-best selection (same full timeline + same penalty prices each
    iteration).

    On disk the table is the per-scenario cost Series tagged with a single
    ``scenario`` column level (row index ``category``); every numeric cell is
    summed to the scalar total.
    """
    path = Path(assess_dir) / _resolve_filename(_TOTAL_COST_KEY)
    df = read_lean_parquet(path)
    return float(df.to_numpy().sum())


__all__ = [
    "read_curtailment_by_sink",
    "read_residual_unserved",
    "read_residual_unserved_dt",
    "read_slack_penalty",
    "read_total_system_cost",
    "read_unit_capacity_total",
]
