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


__all__ = [
    "read_curtailment_by_sink",
    "read_residual_unserved",
    "read_residual_unserved_dt",
    "read_slack_penalty",
]
