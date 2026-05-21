"""In-memory carrier of state passed between solves.

Home of the canonical :class:`SolveHandoff` dataclass — the typed
record of "what one solve produced that the next solve(s) consume".
Built post-solve by :func:`flextool.engine_polars.input.build_handoff_from_flexpy`
directly from the flexpy ``Solution`` object; consumed by the next
sub-solve via the orchestrator's iteration-start translator
(:func:`_provider_translators.translate_handoff_to_provider`) which
fans each field into the Provider under a ``handoff/<field>`` key.

Phase 3 of ``specs/provider_consolidation.md`` retired the legacy
``capture_post_solve()`` disk-read constructor — the cascade was
already using ``build_handoff_from_flexpy`` exclusively; the
capture-from-disk function was dead code and has been removed,
together with the three carrier fields it was the only populator
for (``fix_storage_timesteps``, ``ed_history_realized_first``,
``edd_history``).  Consumers of those fields already fell through
to Provider/CSV reads when the carrier was ``None`` (the universal
state in the cascade path).

R-O2 mitigation
---------------

``flextool/process_outputs/handoff_writers.py:67`` and
``flextool/process_outputs/cumulative_handoffs.py:89`` both import
``SolveHandoff`` by absolute path
(``flextool.flextoolrunner.solve_handoff.SolveHandoff``).  To preserve
source compatibility WITHOUT duplicating the dataclass, the legacy
module re-exports the same class from this one — see
``flextoolrunner/solve_handoff.py``'s shim header.  Both import paths
therefore resolve to the SAME ``SolveHandoff`` class.

Carrier schemas (each is the on-disk equivalent's columns, renamed
for in-memory uniformity — string keys + ``value`` for single-value
carriers, named metric columns for multi-value):

    realized_invest        [entity, period, value]
    realized_existing      [entity, period, value]
    divest_cumulative      [entity, value]
    roll_end_state         [node, value]
    fix_storage            [node, period, time, quantity, price, usage]
    cumulative_co2         [group, period, value]
    cumulative_commodity   [commodity, tier, period, mwh]
    cum_sim_hours          [period, value]

(Δ.1 — ``periods_already_emitted`` was previously listed here; it
moved to ``_output_writer.OutputWriterState`` since it gates writer-
side emission and isn't a true solver-handoff carrier.)

``fix_storage`` collapses the three file-based carriers
(``fix_storage_quantity.csv``, ``..._price.csv``, ``..._usage.csv``)
into one wide row with NULL columns for unused metrics.  Per the
design discussion: less files is good, the trio is independent so
NULL-columns cleanly express which metric is active per (n, d, t).

``realized_invest`` and ``realized_existing`` together cover the two
columns of ``solve_data/p_entity_period_existing_capacity.csv``:
``realized_invest`` is what was *built this solve*, ``realized_existing``
is the resolved existing-capacity history (pre-existing decay +
divest enter this column and aren't reconstructible from
``realized_invest`` alone).

Reference: ``flextool/flextoolrunner/solve_handoff.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl



@dataclass
class SolveHandoff:
    """Per-solve output → dependent-solve input carrier.

    Each field is a polars DataFrame in the schema documented in this
    module's docstring, or ``None`` when that carrier kind isn't active
    for this handoff.  Solves are identified by full solve name; the
    handoff represents the *output* of one solve becoming the *input*
    to its dependent (child / next-roll) solves.
    """

    # Realized invest *built this solve* in absolute units (post-unitsize).
    # Producer: any solve with v_invest > 0.
    # Consumer: subsequent solves' preprocessing.
    # File equivalent: ``p_entity_period_invested_capacity`` column of
    #   solve_data/p_entity_period_existing_capacity.csv.
    realized_invest: pl.DataFrame | None = None

    # Realized existing-capacity history per (entity, period) — captures
    # pre-existing decay + divest that ``realized_invest`` doesn't.
    # File equivalent: ``p_entity_period_existing_capacity`` column of
    #   solve_data/p_entity_period_existing_capacity.csv.
    realized_existing: pl.DataFrame | None = None

    # Cumulative divested capacity per entity (scalar, not per-period).
    # Carries pre-existing decay forward across solves.
    # File equivalent: solve_data/p_entity_divested.csv.
    divest_cumulative: pl.DataFrame | None = None

    # End-of-roll storage state for ``bind_forward_only`` carry-over.
    # Producer: prior roll's v_state at its last (d, t).
    # Consumer: next roll's nodeBalance_eq first-timestep term.
    # File equivalent: solve_data/p_roll_continue_state.csv.
    roll_end_state: pl.DataFrame | None = None

    # Storage quota imposed by parent solve on child solve.  Wide row:
    # NULL columns mark inactive metrics (the trio is independent —
    # parent may set quantity-only, price-only, or any combination).
    # Producer: parent solve's v_state + cost duals.
    # Consumer: child's fix_storage_* constraints.
    # File equivalents: solve_data/fix_storage_{quantity,price,usage}.csv
    fix_storage: pl.DataFrame | None = None

    # Running CO2 totals carried across rolls for cumulative-cap constraint.
    # File equivalent: solve_data/co2_cum_realized_tonnes.csv.
    cumulative_co2: pl.DataFrame | None = None

    # Running per-tier commodity consumption for cumulative ladder pricing.
    # File equivalent: solve_data/commodity_ladder_cumulative.csv (per-tier mwh).
    cumulative_commodity: pl.DataFrame | None = None

    # Running simulated-hour total per period.  Independent 1-D
    # carrier shared by ladder + CO2-cap constraints.
    # File equivalent: solve_data/ladder_cum_sim_hours.csv.
    cum_sim_hours: pl.DataFrame | None = None

    _FIELDS = (
        "realized_invest", "realized_existing", "divest_cumulative",
        "roll_end_state", "fix_storage",
        "cumulative_co2", "cumulative_commodity", "cum_sim_hours",
    )

    def is_empty(self) -> bool:
        """True when no carrier is populated."""
        return all(getattr(self, f) is None for f in self._FIELDS)


def write_fix_storage_files_from_handoff(
    fix_storage: "pl.DataFrame", solve_data_dir,
) -> None:
    """Write the three ``solve_data/fix_storage_*.csv`` files from the
    in-memory wide handoff frame.

    Replaces the file-based ``shutil.copy`` propagation of the parent
    solve's archived ``fix_storage_*_<parent>.csv`` to the current
    solve's ``fix_storage_*.csv``.  The .mod still reads CSV at run
    time (per ``flextool.mod``'s ``table data IN`` blocks); the source
    becomes the in-memory handoff frame instead of the parent's
    archived copy.

    The handoff frame's schema is wide ``[node, period, time, quantity,
    price, usage]`` with NULLs for inactive metrics; this writer fans
    it back out to per-metric files in long format.  The on-disk
    column name for the time axis is ``step`` (not ``time``) — renamed
    accordingly.
    """
    for metric, on_disk_col, fname in (
        ("quantity", "p_fix_storage_quantity", "fix_storage_quantity.csv"),
        ("price",    "p_fix_storage_price",    "fix_storage_price.csv"),
        ("usage",    "p_fix_storage_usage",    "fix_storage_usage.csv"),
    ):
        out = (
            fix_storage
            .filter(pl.col(metric).is_not_null())
            .rename({"time": "step", metric: on_disk_col})
            .select("node", "period", "step", on_disk_col)
        )
        out.write_csv(solve_data_dir / fname)


__all__ = [
    "SolveHandoff",
    "write_fix_storage_files_from_handoff",
]
