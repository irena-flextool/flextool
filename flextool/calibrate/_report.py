"""Pure rendering of a calibration run's outcome (C2).

This module turns an in-memory :class:`~flextool.calibrate._loop.CalibResult`
into (a) machine-readable CSV artifacts and (b) a compact human-readable
stdout summary.  It does **no** solving and re-reads **no** parquet — it
operates purely on the trajectory already carried by ``CalibResult`` — so it
is cheap, deterministic, and safe to call anywhere.

Two CSVs are written by :func:`write_report`:

* a per-``(iteration, node)`` **long** table — one row per node observed in
  each iteration, carrying that iteration's residual unserved energy, the
  adder snapshot that was solved, the per-sink curtailment, the monetised
  per-node slack, and whether the node ended the run flagged
  resource-capped;
* a per-iteration **summary** table — total unserved energy, total slack
  penalty, the count of shedding nodes, the run's flagged count, and the
  convergence flag.

Both are emitted in a fully deterministic order: iterations ascending,
nodes sorted lexicographically.
"""

from __future__ import annotations

import csv
from pathlib import Path

from flextool.calibrate._loop import CalibResult, IterRecord

# CSV column orders — pinned here so both the writer and any downstream
# consumer (and the tests) agree on the exact schema.
LONG_COLUMNS = [
    "iteration",
    "node",
    "residual_mwh",
    "adder_mwh",
    "curtailment_mwh",
    "penalty_meur",
    "flagged",
]
SUMMARY_COLUMNS = [
    "iteration",
    "total_unserved_mwh",
    "total_penalty_meur",
    "n_shedding",
    "n_flagged",
    "converged",
]

LONG_FILENAME = "calibration_by_iteration_node.csv"
SUMMARY_FILENAME = "calibration_summary.csv"


def _record_nodes(record: IterRecord) -> list[str]:
    """Sorted union of every node observed in one iteration's signals."""
    nodes: set[str] = set()
    nodes.update(record.residual)
    nodes.update(record.adders)
    nodes.update(record.curtailment)
    nodes.update(record.penalty_by_node)
    return sorted(nodes)


def _n_shedding(record: IterRecord) -> int:
    """Count of nodes with strictly-positive residual unserved energy."""
    return sum(1 for v in record.residual.values() if v > 0.0)


def write_report(result: CalibResult, *, out_dir: Path) -> list[Path]:
    """Write the two machine-readable CSV artifacts for *result*.

    Emits :data:`LONG_FILENAME` (per-``(iteration, node)``) and
    :data:`SUMMARY_FILENAME` (per-iteration) into *out_dir*, creating it if
    needed.  Rows are ordered iterations ascending then nodes sorted, so the
    output is byte-deterministic for a given result.

    Returns the two written paths in ``[long, summary]`` order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged = set(result.guard_flagged_nodes)

    long_path = out_dir / LONG_FILENAME
    with long_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(LONG_COLUMNS)
        for record in sorted(result.trajectory, key=lambda r: r.iteration):
            for node in _record_nodes(record):
                writer.writerow(
                    [
                        record.iteration,
                        node,
                        record.residual.get(node, 0.0),
                        record.adders.get(node, 0.0),
                        record.curtailment.get(node, 0.0),
                        record.penalty_by_node.get(node, 0.0),
                        node in flagged,
                    ]
                )

    summary_path = out_dir / SUMMARY_FILENAME
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_COLUMNS)
        ordered = sorted(result.trajectory, key=lambda r: r.iteration)
        last_iteration = ordered[-1].iteration if ordered else None
        for record in ordered:
            # ``converged`` is a run-level outcome; it is true only on the
            # final iteration of a converged run (the one that met the
            # threshold), false on every earlier row.
            row_converged = result.converged and record.iteration == last_iteration
            writer.writerow(
                [
                    record.iteration,
                    record.total_unserved,
                    record.penalty_total,
                    _n_shedding(record),
                    len(result.guard_flagged_nodes),
                    row_converged,
                ]
            )

    return [long_path, summary_path]


def format_summary(result: CalibResult) -> str:
    """Render a compact human-readable summary of *result* for stdout.

    Covers: convergence status and iterations run; initial→final total
    unserved energy (and slack penalty, when non-zero); the final per-node
    adders sorted descending with each node's final residual; and a clearly
    marked resource-capped section listing the guard-flagged nodes (the ones
    that need firm capacity / imports / storage rather than more demand
    margin).  Pure — reads only the in-memory *result*.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Adequacy-margin calibration summary")
    lines.append("=" * 60)

    n_adjust = result.iterations_run - 1
    if result.stop_reason == "converged":
        lines.append(
            f"Status: CONVERGED after {result.iterations_run} solve(s) "
            f"({n_adjust} adjustment iteration(s))."
        )
    elif result.stop_reason == "stalled":
        lines.append(
            f"Status: CONVERGED MODULO RESOURCE-CAPPED NODES after "
            f"{result.iterations_run} solve(s) ({n_adjust} adjustment "
            f"iteration(s))."
        )
        lines.append(
            "The energy-margin lever is exhausted: every remaining shedding "
            "node is resource-capped, so no further demand margin can be "
            "added and another solve would only reproduce this one."
        )
    else:  # "budget_exhausted"
        lines.append(
            f"Status: NOT CONVERGED — ran the full budget of "
            f"{result.iterations_run} solve(s) without reaching the slack "
            f"threshold."
        )

    trajectory = sorted(result.trajectory, key=lambda r: r.iteration)
    if trajectory:
        first = trajectory[0]
        last = trajectory[-1]
        lines.append("")
        lines.append(
            f"Total unserved energy: {first.total_unserved:,.1f} MWh "
            f"(baseline) -> {last.total_unserved:,.1f} MWh (final)."
        )
        if first.penalty_total or last.penalty_total:
            lines.append(
                f"Slack penalty:         {first.penalty_total:,.3f} M-EUR "
                f"(baseline) -> {last.penalty_total:,.3f} M-EUR (final)."
            )

    # Final per-node adders, largest first; pair each with its final residual.
    final_residual = trajectory[-1].residual if trajectory else {}
    lines.append("")
    if result.final_adders:
        lines.append("Final energy-margin adders (MWh/timestep), largest first:")
        for node, adder in sorted(
            result.final_adders.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            resid = final_residual.get(node, 0.0)
            lines.append(
                f"  {node:<24} adder={adder:>14,.4f}   "
                f"final residual={resid:>14,.1f} MWh"
            )
    else:
        lines.append("Final energy-margin adders: none (no node was bumped).")

    # Resource-capped section — nodes the over-build guard froze.
    lines.append("")
    lines.append(
        "Resource-capped nodes (need firm capacity / imports / storage, "
        "not more demand margin):"
    )
    if result.guard_flagged_nodes:
        for node in result.guard_flagged_nodes:
            resid = final_residual.get(node, 0.0)
            lines.append(f"  {node:<24} final residual={resid:>14,.1f} MWh")
    else:
        lines.append("  none")

    if result.stop_reason == "stalled" and trajectory:
        lines.append("")
        lines.append(
            f"Residual unserved energy still {trajectory[-1].total_unserved:,.1f} "
            f"MWh at exit, but the calibration is CONVERGED MODULO the "
            f"resource-capped nodes above: that residual sits on the "
            f"guard-flagged nodes, which need firm capacity / imports / "
            f"storage — not more demand margin. Raising --iterations will not "
            f"help; the demand-margin lever is exhausted."
        )
    elif result.stop_reason == "budget_exhausted" and trajectory:
        lines.append("")
        lines.append(
            f"Residual unserved energy still {trajectory[-1].total_unserved:,.1f} "
            f"MWh at exit — raise --iterations or investigate the flagged nodes."
        )

    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = [
    "LONG_COLUMNS",
    "LONG_FILENAME",
    "SUMMARY_COLUMNS",
    "SUMMARY_FILENAME",
    "format_summary",
    "write_report",
]
