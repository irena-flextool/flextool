"""In-memory carrier of state passed between solves — Γ.8.D port.

This module is a 1:1 port of
``flextool/flextoolrunner/solve_handoff.py`` (304 LOC) into the
``engine_polars`` subpackage.  It owns the canonical
:class:`SolveHandoff` dataclass + post-solve capture function for the
native flexpy orchestrator (``_orchestration.py``).

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
    periods_already_emitted [period]

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

from ._input_source import _read_csv_file


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

    # Periods whose capacity outputs have already been dumped by an
    # earlier solve — gates re-emission across rolls.
    # File equivalent: solve_data/period_capacity.csv.
    periods_already_emitted: pl.DataFrame | None = None

    _FIELDS = (
        "realized_invest", "realized_existing", "divest_cumulative",
        "roll_end_state", "fix_storage",
        "cumulative_co2", "cumulative_commodity", "cum_sim_hours",
        "periods_already_emitted",
    )

    def is_empty(self) -> bool:
        """True when no carrier is populated."""
        return all(getattr(self, f) is None for f in self._FIELDS)


def capture_post_solve(state, solve_name: str) -> None:
    """Populate ``state.handoffs[solve_name]`` from the just-completed
    solve's outputs.

    Called from the orchestration loop immediately after the solver
    returns successfully, but only when ``state.handoffs is not None``
    (the opt-in flag).  The capture is **additive** — existing
    post-solve CSV writes (e.g. ``p_entity_period_existing_capacity.csv``)
    continue unchanged, so file-based downstream consumers see the
    same bytes they always did.  This hook merely records the same
    data in memory for future in-memory consumers.

    Carriers not exercised by the current solve (file missing or
    empty) leave their slot at ``None``.

    Mirrors :func:`flextool.flextoolrunner.solve_handoff.capture_post_solve`.
    """
    if state.handoffs is None:
        return  # opt-in flag is off — no-op

    sd = state.paths.work_folder / "solve_data"
    handoff = state.handoffs.setdefault(solve_name, SolveHandoff())

    def _read(name: str) -> "pl.DataFrame | None":
        p = sd / name
        if not p.exists():
            return None
        df = _read_csv_file(p)
        return df if df.height > 0 else None

    # realized_invest + realized_existing: two columns of the same file.
    # ``p_entity_period_invested_capacity`` is what was built in *this*
    # solve.  ``p_entity_period_existing_capacity`` is the resolved
    # existing-capacity history (pre-existing decay + divest enter
    # this column and aren't reconstructible from realized_invest alone).
    ppec = _read("p_entity_period_existing_capacity.csv")
    if ppec is not None:
        if "p_entity_period_invested_capacity" in ppec.columns:
            handoff.realized_invest = (
                ppec.with_columns(
                    value=pl.col("p_entity_period_invested_capacity")
                            .cast(pl.Float64, strict=False)
                            .fill_null(0.0))
                    .select("entity", "period", "value"))
        if "p_entity_period_existing_capacity" in ppec.columns:
            handoff.realized_existing = (
                ppec.with_columns(
                    value=pl.col("p_entity_period_existing_capacity")
                            .cast(pl.Float64, strict=False)
                            .fill_null(0.0))
                    .select("entity", "period", "value"))

    # divest_cumulative: per-entity scalar.  Column name on disk is
    # ``p_entity_divested``; rename to canonical ``value``.
    div = _read("p_entity_divested.csv")
    if div is not None and "p_entity_divested" in div.columns:
        handoff.divest_cumulative = (
            div.with_columns(
                value=pl.col("p_entity_divested")
                        .cast(pl.Float64, strict=False)
                        .fill_null(0.0))
               .select("entity", "value"))

    # roll_end_state: per-node scalar (end-of-roll storage state).
    rcs = _read("p_roll_continue_state.csv")
    if rcs is not None and "p_roll_continue_state" in rcs.columns:
        handoff.roll_end_state = (
            rcs.with_columns(
                value=pl.col("p_roll_continue_state")
                        .cast(pl.Float64, strict=False)
                        .fill_null(0.0))
               .select("node", "value"))

    # fix_storage: outer-join the three independent files (quantity /
    # price / usage) into one wide row keyed by (node, period, time).
    # NULL columns mark inactive metrics — the trio is independent so
    # any combination may be set per (n, d, t).
    def _fix_one(name: str, value_col: str) -> "pl.DataFrame | None":
        df = _read(name)
        if df is None or value_col not in df.columns:
            return None
        return df.rename({"step": "time", value_col: value_col}) \
                 .select("node", "period", "time", value_col)

    fq = _fix_one("fix_storage_quantity.csv", "p_fix_storage_quantity")
    fp = _fix_one("fix_storage_price.csv", "p_fix_storage_price")
    fu = _fix_one("fix_storage_usage.csv", "p_fix_storage_usage")

    if fq is not None or fp is not None or fu is not None:
        # Outer-join on (node, period, time) so each metric independently
        # contributes its own rows.
        merged = None
        for src, col, out_col in [(fq, "p_fix_storage_quantity", "quantity"),
                                    (fp, "p_fix_storage_price", "price"),
                                    (fu, "p_fix_storage_usage", "usage")]:
            if src is None:
                continue
            r = src.rename({col: out_col})
            merged = r if merged is None else merged.join(
                r, on=["node", "period", "time"], how="full", coalesce=True)
        # Ensure all three metric columns exist (NULL where not provided).
        for c in ("quantity", "price", "usage"):
            if c not in merged.columns:
                merged = merged.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
        handoff.fix_storage = merged.select(
            "node", "period", "time", "quantity", "price", "usage")

    # cumulative_co2: per-(group, period) running total.
    co2 = _read("co2_cum_realized_tonnes.csv")
    if co2 is not None and "p_co2_cum_realized_tonnes" in co2.columns:
        handoff.cumulative_co2 = (
            co2.with_columns(
                value=pl.col("p_co2_cum_realized_tonnes")
                        .cast(pl.Float64, strict=False)
                        .fill_null(0.0))
               .select("group", "period", "value"))

    # cumulative_commodity: per-(commodity, tier, period) running mwh.
    # Schema not exercised by current fixtures; capture leniently.
    cc = _read("commodity_ladder_cumulative.csv")
    if cc is not None:
        # Tolerate either ``mwh`` or ``p_ladder_cum_realized_mwh`` as the
        # value column name — the file's writer may use either.
        if "mwh" in cc.columns:
            value_col = "mwh"
        elif "p_ladder_cum_realized_mwh" in cc.columns:
            value_col = "p_ladder_cum_realized_mwh"
        else:
            value_col = None
        if value_col is not None and {"commodity", "tier", "period"}.issubset(cc.columns):
            handoff.cumulative_commodity = (
                cc.with_columns(
                    mwh=pl.col(value_col).cast(pl.Float64, strict=False).fill_null(0.0))
                  .select("commodity", "tier", "period", "mwh"))

    # cum_sim_hours: per-period running simulated-hour total.
    csh = _read("ladder_cum_sim_hours.csv")
    if csh is not None and "p_ladder_cum_sim_hours" in csh.columns:
        handoff.cum_sim_hours = (
            csh.with_columns(
                value=pl.col("p_ladder_cum_sim_hours")
                        .cast(pl.Float64, strict=False)
                        .fill_null(0.0))
               .select("period", "value"))

    # periods_already_emitted: bare set of period strings.
    pae = _read("period_capacity.csv")
    if pae is not None and "period" in pae.columns:
        handoff.periods_already_emitted = pae.select("period").unique()


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
    "capture_post_solve",
    "write_fix_storage_files_from_handoff",
]
