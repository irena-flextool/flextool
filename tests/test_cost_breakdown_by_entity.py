"""Reconciliation test for the per-entity cost break-down outputs.

The new ``cost_breakdown_by_entity`` output (out_costs.py) collapses every
cost category per ENTITY (unit / connection / node) at the PERIOD level, in
both annualized (M CUR/a) and discounted (M CUR) flavours.  It is purely
additive over existing per-entity intermediates in ``calc_costs.py``.

This test proves the additivity invariant end-to-end against a live solve:

1.  **Broadcast guards** — the two per-process intermediates added in
    ``calc_costs.py`` must sum-over-process back to the system per-(d, t)
    totals they were derived from:

        Σ_process cost_commodity_process_dt == Σ_commodity cost_commodity_dt
        Σ_process cost_process_co2_dt        == cost_co2_dt

    A failure here means a commodity has node-varying prices (so the
    process-level price frame mis-aligns) — STOP and report, do not paper
    over it.

2.  **Reconciliation** — summing each per-entity table over its entities,
    per category and period, must equal the corresponding system summary
    (``annualized_costs_d_p`` / ``costs_discounted_d_p``).  Investment +
    retirement are compared against the fused system column
    ('<kind> investment & retirement').  Group-level penalties with no
    entity home are explicitly excluded.

The harness mirrors ``tests/test_cost_aggregation_semantics.py`` (run the
native cascade, snapshot the processed inputs), but instead of reading CSVs
back it reconstructs the in-memory ``par / s / v / r`` namespaces exactly as
``write_outputs`` does, then calls the two output functions directly.  This
keeps the assertions on the published pandas frames (not on round-tripped
CSV text).

CLAUDE.md invariant #3: the DB is built from JSON/schema via the
``test_db_url`` fixture (``json_to_db``); no checked-in ``.sqlite`` is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.process_outputs.out_costs import (  # noqa: E402
    cost_breakdown_by_entity,
    cost_summaries,
)
from flextool.process_outputs.process_results import post_process_results  # noqa: E402
from flextool.process_outputs.write_outputs import _read_outputs  # noqa: E402


# ---------------------------------------------------------------------------
# Harness — run the cascade and rebuild the in-memory par/s/v/r namespaces.
# ---------------------------------------------------------------------------

def _build_namespaces(scenario: str, test_db_url: str, workdir: Path):
    """Run the native cascade for ``scenario`` and return ``(par, s, v, r)``.

    Mirrors ``write_outputs``'s non-replay branch: ``_read_outputs`` builds
    ``par / s / v`` from the persisted realized slices + last step's
    flex_data/solution, and ``post_process_results`` (which calls
    ``compute_slacks`` then ``compute_costs``) derives ``r``.
    """
    import os

    os.chdir(workdir)

    steps = run_chain_from_db(
        test_db_url,
        scenario,
        work_folder=workdir,
        csv_dump=True,
        keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for scenario {scenario!r}"
    last_step = next(reversed(steps.values()))
    assert last_step.solution is not None and last_step.solution.optimal, (
        f"Last sub-solve for scenario {scenario!r} did not solve optimally"
    )
    provider = getattr(last_step, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(workdir)

    par, s, v = _read_outputs(
        str(workdir / "output_raw"),
        flex_data=last_step.flex_data,
        solution=last_step.solution,
        solve_name=last_step.solve_name,
        solve_steps=[
            (st.solve_name, st.flex_data, st.effective_solution)
            for st in steps.values()
        ],
        # The in-memory output path REQUIRES a Provider (empty group sets
        # must be stated by an explicit Provider, not a missing one).
        flex_data_provider=provider,
    )
    r = post_process_results(par, s, v)
    return par, s, v, r


def _outputs_dict(par, s, v, r) -> dict:
    """Run both cost output functions and return {key: frame}."""
    out: dict[str, pd.DataFrame | pd.Series] = {}
    for func in (cost_summaries, cost_breakdown_by_entity):
        for frame, key in func(par, s, v, r, debug=False):
            out[key] = frame
    return out


# ---------------------------------------------------------------------------
# Reconciliation bookkeeping.
# ---------------------------------------------------------------------------

# Categories that live only at the group/system level (no entity home) and so
# are intentionally NOT reproduced by the per-entity tables.  Excluded from
# the reconciliation column set.
SYSTEM_ONLY_CATEGORIES = {
    "inertia slack penalty",
    "non-synchronous slack penalty",
    "upward reserve slack penalty",
    "downward reserve slack penalty",
    "capacity margin penalty",
}

# Per-kind fused invest/retirement system column → the two per-entity
# categories that sum into it.
FUSED_INVEST_COLUMN = {
    "unit": "unit investment & retirement",
    "connection": "connection investment & retirement",
    "node": "storage investment & retirement",
}
INVEST_CATEGORIES = {
    "unit": ("investment", "retirement"),
    "connection": ("investment", "retirement"),
    "node": ("storage investment", "storage retirement"),
}

# Direct per-category matches (same column name on both sides).
DIRECT_CATEGORIES = [
    "commodity_cost",
    "commodity_sales",
    "co2",
    "other operational",
    "starts",
    "fixed cost pre-existing",
    "fixed cost invested",
    "fixed cost reduction of divestments",
    "upward slack penalty",
    "downward slack penalty",
]

KINDS = ("unit", "connection", "node")

ATOL = 1e-6
RTOL = 1e-9


def _entity_sum_per_category(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse a (period, entity) × category table over the entity level →
    (period × category) frame (sum over entities)."""
    # Index level 0 is 'period', level 1 is the entity level.
    return table.groupby(level="period").sum()


def _system_category_series(system: pd.DataFrame, category: str) -> pd.Series:
    """Per-period system value for ``category`` (0.0 where the column is
    absent — e.g. a category never charged in this scenario)."""
    if category in system.columns:
        return system[category].astype(float)
    return pd.Series(0.0, index=system.index)


def _assert_period_index_agrees(per_entity_periods, system_periods, ctx: str):
    """The set of periods carrying a per-entity charge must be a subset of
    the system periods (the system summary may carry extra all-zero
    invest-only periods, but never the reverse)."""
    extra = set(map(str, per_entity_periods)) - set(map(str, system_periods))
    assert not extra, (
        f"{ctx}: per-entity table carries periods absent from the system "
        f"summary: {sorted(extra)}"
    )


def _collapsed_by_kind(out: dict, flavour: str) -> dict:
    """{kind: (period × category) entity-sum frame} for the present tables."""
    collapsed = {}
    for kind in KINDS:
        key = f"cost_{kind}_{flavour}_d_ec"
        if key in out:
            collapsed[kind] = _entity_sum_per_category(out[key])
    return collapsed


def _reconcile_cross_kind(out: dict, system: pd.DataFrame, flavour: str):
    """Reconcile the per-entity tables against the system summary.

    The system summary carries ONE column per category that already sums over
    every entity kind (a coal unit's ``fixed cost pre-existing`` and a
    connection's both land in the single system column).  The additivity
    invariant is therefore the CROSS-KIND total: summing every entity table
    over its entities, then summing those across the three kinds, per category
    and period, must equal the system column.

    Returns the list of categories actually reconciled (non-vacuous guard).
    """
    collapsed = _collapsed_by_kind(out, flavour)
    for kind, frame in collapsed.items():
        _assert_period_index_agrees(frame.index, system.index, f"{kind} table")

    checked: list[str] = []

    # --- Direct categories: cross-kind sum vs the single system column ----
    for cat in DIRECT_CATEGORIES:
        if cat not in system.columns:
            continue
        total = pd.Series(0.0, index=system.index)
        present = False
        for frame in collapsed.values():
            if cat in frame.columns:
                total = total.add(frame[cat].astype(float), fill_value=0.0)
                present = True
        if not present:
            # No entity table emits this category in this scenario (the
            # system column may still be non-zero only via a system-only
            # producer — but every DIRECT category has an entity home, so a
            # non-zero system column with no entity producer is a real bug).
            sys_col = system[cat].astype(float)
            assert sys_col.abs().sum() <= ATOL, (
                f"{flavour} / {cat}: system column is non-zero "
                f"({sys_col.abs().sum():.6f}) but NO entity table reproduces "
                f"it — a cost lost its per-entity home."
            )
            continue
        total = total.reindex(system.index).fillna(0.0)
        # annualized_costs_d_p does NOT fillna its dispatch columns, so a
        # period that realizes investment but no dispatch carries NaN in the
        # operational categories.  The per-entity table fills those periods
        # with 0.0 (the operational pieces are absent there), which is the
        # correct additive identity — treat the system NaN as 0.0 to compare.
        rhs = system[cat].astype(float).fillna(0.0)
        np.testing.assert_allclose(
            total.to_numpy(), rhs.to_numpy(), rtol=RTOL, atol=ATOL,
            err_msg=(
                f"{flavour} / {cat}: cross-kind per-period entity-sum does not "
                f"match the system summary.\nlhs=\n{total}\nrhs=\n{rhs}"
            ),
        )
        np.testing.assert_allclose(
            total.sum(), rhs.sum(), rtol=RTOL, atol=ATOL,
            err_msg=f"{flavour} / {cat}: full-column sum mismatch",
        )
        checked.append(cat)

    # --- Fused invest + retirement: per-kind, vs the kind's fused column --
    for kind, frame in collapsed.items():
        inv_cat, div_cat = INVEST_CATEGORIES[kind]
        fused_col = FUSED_INVEST_COLUMN[kind]
        if fused_col not in system.columns:
            continue
        if inv_cat not in frame.columns and div_cat not in frame.columns:
            # No invest/retire for this kind: system fused column must be 0.
            sys_col = system[fused_col].astype(float)
            assert sys_col.abs().sum() <= ATOL, (
                f"{flavour} / {fused_col}: system column non-zero "
                f"({sys_col.abs().sum():.6f}) but the {kind} table has no "
                f"invest/retire categories."
            )
            continue
        total = pd.Series(0.0, index=frame.index)
        for cc in (inv_cat, div_cat):
            if cc in frame.columns:
                total = total.add(frame[cc].astype(float), fill_value=0.0)
        total = total.reindex(system.index).fillna(0.0)
        rhs = system[fused_col].astype(float).fillna(0.0)
        np.testing.assert_allclose(
            total.to_numpy(), rhs.to_numpy(), rtol=RTOL, atol=ATOL,
            err_msg=(
                f"{flavour} / {fused_col}: per-period (invest+retire) "
                f"entity-sum does not match the fused system column.\n"
                f"lhs=\n{total}\nrhs=\n{rhs}"
            ),
        )
        np.testing.assert_allclose(
            total.sum(), rhs.sum(), rtol=RTOL, atol=ATOL,
            err_msg=f"{flavour} / {fused_col}: full-column sum mismatch",
        )
        checked.append(fused_col)

    return checked


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------

# Three complementary fixtures give full, non-vacuous category coverage:
#   * wind_battery_invest — unit investment (wind) + storage investment
#     (battery state); exercises the 'investment' / 'storage investment'
#     and 'fixed cost invested' columns with non-zero values.
#   * coal_retire — commodity (coal) + retirement (divest 0.5 coal_plant)
#     + 'fixed cost reduction of divestments' + 'fixed cost pre-existing'.
#   * coal_co2_price — commodity + CO2 price + (online) startup, plus
#     node-state slack penalties.
SCENARIOS = [
    "wind_battery_invest",
    "coal_retire",
    "coal_co2_price",
]


@pytest.fixture(scope="module", params=SCENARIOS)
def namespaces(request, test_db_url, tmp_path_factory):
    workdir = tmp_path_factory.mktemp(f"cbe_{request.param.replace('-', '_')}")
    return request.param, _build_namespaces(request.param, test_db_url, workdir)


# ---------------------------------------------------------------------------
# 1. Broadcast guards (deterministic; STOP-and-report on failure).
# ---------------------------------------------------------------------------

def test_broadcast_guard_commodity(namespaces):
    scenario, (par, s, v, r) = namespaces
    per_process = r.cost_commodity_process_dt.sum(axis=1)
    per_commodity = r.cost_commodity_dt.sum(axis=1)
    per_process, per_commodity = per_process.align(per_commodity, fill_value=0.0)
    np.testing.assert_allclose(
        per_process.to_numpy(), per_commodity.to_numpy(),
        rtol=RTOL, atol=ATOL,
        err_msg=(
            f"[{scenario}] BROADCAST GUARD FAILED for commodity: "
            f"Σ_process cost_commodity_process_dt ({per_process.sum():.6f}) != "
            f"Σ_commodity cost_commodity_dt ({per_commodity.sum():.6f}); a "
            f"commodity likely has node-varying prices needing a two-level join."
        ),
    )
    # Same for sales.
    sp = r.sales_commodity_process_dt.sum(axis=1)
    sc = r.sales_commodity_dt.sum(axis=1)
    sp, sc = sp.align(sc, fill_value=0.0)
    np.testing.assert_allclose(
        sp.to_numpy(), sc.to_numpy(), rtol=RTOL, atol=ATOL,
        err_msg=f"[{scenario}] BROADCAST GUARD FAILED for commodity sales.",
    )


def test_broadcast_guard_co2(namespaces):
    scenario, (par, s, v, r) = namespaces
    per_process = r.cost_process_co2_dt.sum(axis=1)
    system = r.cost_co2_dt
    per_process, system = per_process.align(system, fill_value=0.0)
    np.testing.assert_allclose(
        per_process.to_numpy(), system.to_numpy(),
        rtol=RTOL, atol=ATOL,
        err_msg=(
            f"[{scenario}] BROADCAST GUARD FAILED for CO2: "
            f"Σ_process cost_process_co2_dt ({per_process.sum():.6f}) != "
            f"cost_co2_dt ({system.sum():.6f})."
        ),
    )


# ---------------------------------------------------------------------------
# 2. Reconciliation against the system summaries.
# ---------------------------------------------------------------------------

def test_reconciliation_annualized(namespaces):
    scenario, (par, s, v, r) = namespaces
    out = _outputs_dict(par, s, v, r)
    system = out["annualized_costs_d_p"]
    checked = _reconcile_cross_kind(out, system, "annualized")
    assert checked, (
        f"[{scenario}] no per-entity annualized category reconciled — the "
        f"test is vacuous; check the scenario actually emits cost-by-entity."
    )
    # Non-system-only categories present in the summary must ALL be covered.
    expected = {
        c for c in system.columns
        if c not in SYSTEM_ONLY_CATEGORIES
        and system[c].abs().sum() > ATOL
    }
    # Fused invest columns are reconciled under their own name.
    missing = expected - set(checked)
    assert not missing, (
        f"[{scenario}] annualized: non-zero system categories left "
        f"un-reconciled (no entity home found): {sorted(missing)}"
    )


def test_reconciliation_discounted(namespaces):
    scenario, (par, s, v, r) = namespaces
    out = _outputs_dict(par, s, v, r)
    system = out["costs_discounted_d_p"]
    checked = _reconcile_cross_kind(out, system, "discounted")
    assert checked, (
        f"[{scenario}] no per-entity discounted category reconciled — vacuous."
    )
    expected = {
        c for c in system.columns
        if c not in SYSTEM_ONLY_CATEGORIES
        and system[c].abs().sum() > ATOL
    }
    missing = expected - set(checked)
    assert not missing, (
        f"[{scenario}] discounted: non-zero system categories left "
        f"un-reconciled (no entity home found): {sorted(missing)}"
    )


def test_output_shape_and_levels(namespaces):
    """Lock the published shape: index (period, entity-level), single
    'category' column level, entity level named per kind."""
    scenario, (par, s, v, r) = namespaces
    out = _outputs_dict(par, s, v, r)
    for kind in KINDS:
        for flavour in ("annualized", "discounted"):
            key = f"cost_{kind}_{flavour}_d_ec"
            if key not in out:
                continue
            df = out[key]
            assert df.index.nlevels == 2, f"{key}: expected 2-level index"
            assert df.index.names[0] == "period", f"{key}: level 0 must be 'period'"
            assert df.index.names[1] == kind, (
                f"{key}: entity level must be named {kind!r}, "
                f"got {df.index.names[1]!r}"
            )
            assert df.columns.name == "category", (
                f"{key}: columns level must be named 'category'"
            )
            assert df.notna().all().all(), f"{key}: NaNs leaked into output"
