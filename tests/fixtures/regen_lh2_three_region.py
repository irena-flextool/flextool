"""Regenerate ``tests/fixtures/lh2_three_region.json`` from scratch.

This is the imperative builder for the three-region LH2 fixture.  It
constructs a fresh Spine SQLite DB programmatically, then exports it
to JSON via :func:`tests.db_utils.db_to_json`.  Tests do NOT invoke
this script — they consume the committed JSON via
:func:`tests.db_utils.json_to_db` (mirrors the ``tests.json`` pattern).

Usage::

    python tests/fixtures/regen_lh2_three_region.py
        # writes tests/fixtures/lh2_three_region.json

    python tests/fixtures/regen_lh2_three_region.py --out /tmp/foo.json
        # writes the JSON to a custom path

The build is byte-deterministic: re-running the script produces an
identical JSON file (the underlying SQLite IDs are seeded by
``spinedb_api`` in a stable way given an identical insertion order).

Layout (per region)
-------------------
    elec_<r>     (hourly_group)         — power balance node
    h2_<r>       (daily_group)          — gaseous H2 transit node
    lh2_<r>      (daily_group, storage) — liquid H2 storage

Per-region processes:
    wind_<r>     — variable RES (profile_method=upper_limit)
    coal_<r>     — dispatchable thermal (constant_efficiency, no UC)
    battery_<r> + battery_charge_<r> + battery_discharge_<r>
                 — storage node + bi-directional inverter
    electrolyser_<r> — *indirect* method: elec→H2.  Source side is
                       hourly (elec), sink side is daily (h2).
    liquefier_<r>    — direct constant_efficiency: h2→lh2 (both daily)

Inter-region:
    pipe_AB, pipe_BC — H2 pipelines (process_connection) between LH2
                       storage nodes.  All-daily.

Exogenous:
    coal_market   — single global commodity node (hourly) feeding every
                    coal plant.
    lh2_demand_<r> — fixed daily demand drawn off lh2 storage in
                     regions A and B (none in C).

Time:
    timeline ``y2030_168h``: 168 hourly steps t0001..t0168.
    timeset ``week168``    : 168 rows of duration 1.0.
    Single solve ``lh2_week`` over period ``y2030``.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from spinedb_api import DatabaseMapping, import_data, to_database
from spinedb_api.exception import SpineDBAPIError

# Allow running this script standalone from any cwd.
HERE = Path(__file__).parent
TESTS_DIR = HERE.parent
REPO_ROOT = TESTS_DIR.parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import db_to_json, json_to_db  # noqa: E402

# Reuse constants from the public module so tests and this script
# stay in lockstep.
from build_lh2_three_region import (  # noqa: E402
    ALT,
    Horizon,
    REGIONS,
    SCENARIO,
)


DEFAULT_OUT = HERE / "lh2_three_region.json"

# Benders Phase-0 sibling fixture (greenfield investable trade, 2-day
# horizon).  Lives in a SEPARATE JSON so the committed
# ``lh2_three_region.json`` stays byte-identical.
TRADE_INVEST_ALT: str = "lh2_three_region_trade_invest"
TRADE_INVEST_SCENARIO: str = "lh2_three_region_trade_invest"
TRADE_INVEST_OUT = HERE / "lh2_three_region_trade_invest.json"

# Additive invest layer (TIER 1 Lagrangian invest→dispatch chain).  This
# alternative + scenario layers ON TOP of the base ``ALT`` topology; the
# base ``lh2_three_region`` scenario is untouched so every existing
# lagrangian/parity test stays byte-stable.  See
# ``specs/lagrangian_solution_assembly.md`` "Tier1 test fixture plan".
INVEST_ALT: str = "lh2_three_region_invest"
INVEST_SCENARIO: str = "lh2_three_region_invest"

# Benders Phase-3b sibling fixture — representative-period invest with
# NON-UNIT ``representative_period_weights``.  Lives in a SEPARATE JSON
# (``lh2_three_region_rp_invest.json``) so the committed
# ``lh2_three_region.json`` / ``lh2_three_region_trade_invest.json`` stay
# byte-identical.  Regression vehicle for the RP-weight engine fix (RP
# weights must reach the objective via the folded ``timestep_weight.csv``).
RP_INVEST_ALT: str = "lh2_three_region_rp_invest"
RP_INVEST_SCENARIO: str = "lh2_three_region_rp_invest"
RP_INVEST_OUT = HERE / "lh2_three_region_rp_invest.json"


# --- Time series synthesis --------------------------------------------------


def _wind_profile(scale: float, phase: float, hz: Horizon) -> dict[str, float]:
    """Sinusoidal wind profile over the horizon, normalised to
    [0.05, 0.95]·scale.  The weekly term keeps the legacy ``/168.0``
    period so the DEFAULT 168h emit is byte-identical."""
    out: dict[str, float] = {}
    for i, ts in enumerate(hz.hourly_steps):
        diurnal = 0.5 + 0.4 * math.cos((i + phase) * 2 * math.pi / 24)
        weekly = 0.85 + 0.15 * math.sin((i / 168.0) * 2 * math.pi)
        v = scale * diurnal * weekly
        out[ts] = round(max(0.0, min(1.0, v)), 6)
    return out


def _step_wind_profile(hz: Horizon) -> dict[str, float]:
    """Deterministic 100/50/0 availability vector for the Benders Phase-0
    fixture: the first third of the horizon at 1.0, the middle third at
    0.5, the last third at 0.0.  For the canonical 48h horizon this is
    exactly ``[1.0]*16 + [0.5]*16 + [0.0]*16``.
    """
    n = hz.n_hours
    third = n // 3
    levels = [1.0] * third + [0.5] * third + [0.0] * (n - 2 * third)
    return {ts: levels[i] for i, ts in enumerate(hz.hourly_steps)}


def _zero_profile(hz: Horizon) -> dict[str, float]:
    """All-zero availability across the horizon."""
    return {ts: 0.0 for ts in hz.hourly_steps}


def _elec_demand(peak: float, base: float, hz: Horizon) -> dict[str, float]:
    """Hourly electricity demand profile (negative = demand)."""
    out: dict[str, float] = {}
    for i, ts in enumerate(hz.hourly_steps):
        diurnal = 0.7 + 0.3 * math.cos((i - 6) * 2 * math.pi / 24)
        v = base + (peak - base) * diurnal
        out[ts] = round(-v, 4)
    return out


def _daily_lh2_demand(daily_kw: float, hz: Horizon) -> dict[str, float]:
    """Daily LH2 demand at the lh2 storage node, indexed at the daily
    block's per-day step labels (one entry per day).
    """
    out: dict[str, float] = {}
    for d in range(hz.n_days):
        out[hz.daily_steps[d]] = round(-daily_kw * 24, 4)
    return out


# --- Spine import helpers ---------------------------------------------------


def _map(rows: list[tuple[str, float]] | dict[str, float]) -> dict[str, Any]:
    """Encode a Spine 1d-map literal."""
    if isinstance(rows, dict):
        rows = list(rows.items())
    return {
        "type": "map",
        "index_type": "str",
        "index_name": "time",
        "rank": 1,
        "data": [[k, float(v)] for k, v in rows],
    }


def _array_str(values: list[str], index_name: str = "sequence_index") -> dict[str, Any]:
    return {
        "type": "array",
        "value_type": "str",
        "data": list(values),
        "index_name": index_name,
    }


def _period_timeset_map(rows: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "type": "map",
        "index_type": "str",
        "index_name": "period",
        "rank": 1,
        "data": [[p, ts] for p, ts in rows],
    }


def _years_represented_map(rows: list[tuple[str, float]]) -> dict[str, Any]:
    """Encode a ``solve.years_represented`` Map (period -> #years)."""
    return {
        "type": "map",
        "index_type": "str",
        "index_name": "period",
        "rank": 1,
        "data": [[p, float(v)] for p, v in rows],
    }


def _rp_weights_map(
    base_to_reps: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Encode ``timeset.representative_period_weights`` — a rank-2 nested
    Map ``base_period_start -> {representative_period_start -> weight}``.

    Mirrors :func:`flextool.representative_periods.preprocess._build_weights_map`.
    """
    outer: list[list[Any]] = []
    for base, reps in base_to_reps.items():
        inner = {
            "type": "map",
            "index_type": "str",
            "index_name": "representative_period",
            "rank": 1,
            "data": [[rep, float(w)] for rep, w in reps.items()],
        }
        outer.append([base, inner])
    return {
        "type": "map",
        "index_type": "str",
        "index_name": "base_period",
        "rank": 1,
        "data": outer,
    }


# --- Builder ---------------------------------------------------------------


def _array_str_periods(values: list[str]) -> dict[str, Any]:
    """Encode a period Array (``invest_periods`` / ``realized_periods``)."""
    return _array_str(values, "period")


def _build_invest_overlay(base_entities: list[tuple]) -> dict[str, list]:
    """Return the additive invest layer (entities, parameter_values,
    alternative/scenario, entity_alternatives) for the
    ``lh2_three_region_invest`` scenario.

    The layer composes ON TOP of the base ``ALT`` topology: the invest
    scenario lists BOTH ``ALT`` (base entities + their parameter values)
    then ``INVEST_ALT`` (this layer's overrides + new solves), so the
    invest layer wins on overlapping ``(entity, parameter)`` keys.

    What the layer does
    -------------------
    * Makes ``wind_<r>`` invest-eligible in EVERY region
      (``invest_method=invest_no_limit`` + low ``invest_cost`` +
      ``lifetime`` / ``discount_rate``), so the assembled invest vars span
      all three regions and the owner-selection assembly is exercised.
    * TIGHTENS the binding ``existing`` capacity (drops ``coal_<r>`` from
      400 → a small floor) so each region has a genuine in-region capacity
      shortage.  With the per-hour annualised wind invest cost far below
      the ``penalty_up=8000`` unserved-energy slack price, the LP strictly
      prefers building wind over paying the penalty — i.e. it actually
      invests.  Verified empirically (see the test thresholds).
    * Defines a two-solve chain ``[lh2_invest, lh2_dispatch]``:
      ``lh2_invest`` is a single-solve Lagrangian invest solve
      (``decomposition=lagrangian`` + ``invest_periods=[y2030]``);
      ``lh2_dispatch`` is a monolithic single-solve dispatch over the same
      period (``decomposition`` unset) that consumes the invested capacity
      via the handoff overlay.
    """
    new_entities: list[tuple] = []
    pv: list[tuple] = []

    # --- New solves -----------------------------------------------------
    # Lagrangian investment solve.
    new_entities.append(("solve", "lh2_invest"))
    pv.extend([
        ("solve", "lh2_invest", "solve_mode", "single_solve", INVEST_ALT),
        ("solve", "lh2_invest", "decomposition", "lagrangian", INVEST_ALT),
        ("solve", "lh2_invest", "period_timeset",
         _period_timeset_map([("y2030", "week168")]), INVEST_ALT),
        ("solve", "lh2_invest", "realized_periods",
         _array_str_periods(["y2030"]), INVEST_ALT),
        ("solve", "lh2_invest", "invest_periods",
         _array_str_periods(["y2030"]), INVEST_ALT),
        ("solve", "lh2_invest", "realized_invest_periods",
         _array_str_periods(["y2030"]), INVEST_ALT),
        # Lagrangian knobs mirror the base-scenario db-driven test so the
        # subgradient converges on the same ~0.1 % gap floor.
        ("solve", "lh2_invest", "lagrangian_alpha", 10.0, INVEST_ALT),
        ("solve", "lh2_invest", "lagrangian_max_iter", 100.0, INVEST_ALT),
        ("solve", "lh2_invest", "lagrangian_tolerance", 0.5, INVEST_ALT),
    ])

    # Downstream monolithic dispatch solve (consumes the invested
    # capacity via the handoff overlay).  ``decomposition`` left unset ⇒
    # monolithic.
    new_entities.append(("solve", "lh2_dispatch"))
    pv.extend([
        ("solve", "lh2_dispatch", "solve_mode", "single_solve", INVEST_ALT),
        ("solve", "lh2_dispatch", "period_timeset",
         _period_timeset_map([("y2030", "week168")]), INVEST_ALT),
        ("solve", "lh2_dispatch", "realized_periods",
         _array_str_periods(["y2030"]), INVEST_ALT),
    ])

    # --- Chain: override base ``[lh2_week]`` with the 2-solve chain -----
    pv.append(
        ("model", "flexTool", "solves",
         _array_str(["lh2_invest", "lh2_dispatch"]), INVEST_ALT))

    # --- Per-region invest eligibility + binding capacity tightening ---
    for r in REGIONS:
        wind = f"wind_{r}"
        coal = f"coal_{r}"
        pv.extend([
            # Make wind invest-eligible (cheapest path: invest_no_limit,
            # no caps needed).  ``invest_cost`` is the OVERNIGHT capital
            # cost per kW; annualised over lifetime/discount it is a tiny
            # fraction of the 8000 slack price, so investing strictly beats
            # unserved energy whenever capacity is short.
            ("unit", wind, "invest_method", "invest_no_limit", INVEST_ALT),
            ("unit", wind, "invest_cost", 50.0, INVEST_ALT),
            ("unit", wind, "lifetime", 25.0, INVEST_ALT),
            ("unit", wind, "discount_rate", 0.05, INVEST_ALT),
            # Tighten the dispatchable thermal so each region faces a
            # genuine in-region capacity shortage that wind investment
            # must cover.  Base ``existing`` = 400; drop to 20 under the
            # invest scenario only (additive override — base scenario
            # keeps 400).
            ("unit", coal, "existing", 20.0, INVEST_ALT),
        ])

    # --- Entity-alternatives: activate the two NEW solves under the
    #     invest alternative (the base entities are activated by ALT,
    #     which the invest scenario also lists).
    ent_alts: list[tuple] = []
    for ent in new_entities:
        cl, name = ent[0], ent[1]
        ent_byname = (name,) if isinstance(name, str) else tuple(name)
        ent_alts.append((cl, ent_byname, INVEST_ALT, True))

    return {
        "entities": new_entities,
        "parameter_values": pv,
        "alternatives": [
            (INVEST_ALT, "Three-region LH2 invest→dispatch chain (TIER 1)"),
        ],
        "scenarios": [
            (INVEST_SCENARIO, False,
             "Three-region LH2 Lagrangian invest→dispatch chain (TIER 1)"),
        ],
        # Order matters: ALT first (base topology), INVEST_ALT second
        # (overrides + new solves win on overlapping keys).
        "scenario_alternatives": [
            (INVEST_SCENARIO, ALT, None),
            (INVEST_SCENARIO, INVEST_ALT, None),
        ],
        "entity_alternatives": ent_alts,
    }


def _build_trade_invest_overlay(
    base_entities: list[tuple], hz: Horizon
) -> dict[str, list]:
    """Return the additive Benders Phase-0 ``lh2_three_region_trade_invest``
    overlay (greenfield investable cross-region pipes + asymmetric
    wind/demand that makes A→B→C trade strictly optimal).

    Layered ON TOP of the base ``ALT`` topology: the new scenario lists
    ``ALT`` first then ``TRADE_INVEST_ALT``, so these overrides win on
    overlapping ``(entity, parameter)`` keys.  Mirrors
    :func:`_build_invest_overlay` structurally.

    See ``specs/benders_option_c_fixture_recipe.md`` for the numeric
    argument; the verified ground truth (monolith trades, current
    Lagrangian collapses to autarky) is exercised by
    ``tests/engine_polars/test_benders_phase0_fixture.py``.
    """
    new_entities: list[tuple] = []
    pv: list[tuple] = []

    # --- New single Lagrangian investment solve -------------------------
    new_entities.append(("solve", "lh2_trade_invest"))
    pv.extend([
        ("solve", "lh2_trade_invest", "solve_mode", "single_solve",
         TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "decomposition", "lagrangian",
         TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "period_timeset",
         _period_timeset_map([("y2030", "week168")]), TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "realized_periods",
         _array_str_periods(["y2030"]), TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "invest_periods",
         _array_str_periods(["y2030"]), TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "realized_invest_periods",
         _array_str_periods(["y2030"]), TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "lagrangian_alpha", 10.0,
         TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "lagrangian_max_iter", 100.0,
         TRADE_INVEST_ALT),
        ("solve", "lh2_trade_invest", "lagrangian_tolerance", 0.5,
         TRADE_INVEST_ALT),
    ])
    pv.append(
        ("model", "flexTool", "solves",
         _array_str(["lh2_trade_invest"]), TRADE_INVEST_ALT))

    # --- Greenfield investable pipes (override base existing=50) --------
    for pipe in ("pipe_AB", "pipe_BC"):
        pv.extend([
            ("connection", pipe, "existing", 0.0, TRADE_INVEST_ALT),
            ("connection", pipe, "invest_method", "invest_total",
             TRADE_INVEST_ALT),
            ("connection", pipe, "invest_cost", 10.0, TRADE_INVEST_ALT),
            ("connection", pipe, "invest_max_total", 5000.0,
             TRADE_INVEST_ALT),
            ("connection", pipe, "lifetime", 25.0, TRADE_INVEST_ALT),
            ("connection", pipe, "discount_rate", 0.05, TRADE_INVEST_ALT),
            # efficiency 0.95 inherited from base.
        ])

    # --- Region A: cheap wind, deterministic 100/50/0 availability -----
    pv.append(
        ("profile", "wind_profile_A", "profile",
         _map(_step_wind_profile(hz)), TRADE_INVEST_ALT))

    # --- Region C: demand-heavy import sink -----------------------------
    # (a) add a daily LH2 demand at lh2_C (absent in base).
    pv.append(
        ("node", "lh2_C", "inflow",
         _map(_daily_lh2_demand(120.0, hz)), TRADE_INVEST_ALT))
    # (b) kill C's local cheap wind so it cannot self-supply.
    pv.append(
        ("profile", "wind_profile_C", "profile",
         _map(_zero_profile(hz)), TRADE_INVEST_ALT))
    # (c) cap C's coal so even coal cannot fully cover C (mostly consumed
    #     locally by C's elec demand).
    pv.append(("unit", "coal_C", "existing", 60.0, TRADE_INVEST_ALT))

    # --- Entity-alternatives: activate the new solve under the overlay --
    ent_alts: list[tuple] = []
    for ent in new_entities:
        cl, name = ent[0], ent[1]
        ent_byname = (name,) if isinstance(name, str) else tuple(name)
        ent_alts.append((cl, ent_byname, TRADE_INVEST_ALT, True))

    return {
        "entities": new_entities,
        "parameter_values": pv,
        "alternatives": [
            (TRADE_INVEST_ALT,
             "Three-region LH2 greenfield-trade invest (Benders Phase 0)"),
        ],
        "scenarios": [
            (TRADE_INVEST_SCENARIO, False,
             "Three-region LH2 greenfield investable trade "
             "(Benders Phase 0 prototype)"),
        ],
        "scenario_alternatives": [
            (TRADE_INVEST_SCENARIO, ALT, None),
            (TRADE_INVEST_SCENARIO, TRADE_INVEST_ALT, None),
        ],
        "entity_alternatives": ent_alts,
    }


# Representative-period weight variants emitted by the Phase-3b RP fixture.
# Each is ``{rep_start: weight}`` per FlexTool period; the two reps start at
# ``t0001`` (rep1) and ``t0025`` (rep2) of the 48h horizon.  A single base
# period (``t0001``) folds to the two reps, so ``_compute_rp_frames``
# normalises ``w_r[rep] = weight · n_rp/n_base = weight · 2`` (uniform
# 0.5/0.5 ⇒ 1.0/1.0, the byte-identity baseline).
_RP_WEIGHT_VARIANTS: dict[str, dict[str, dict[str, float]]] = {
    # Base case — cost-asymmetric reps (rep1 cheap free-wind, rep2 forced
    # coal): non-unit weights so the RP fold strictly moves the objective.
    RP_INVEST_ALT: {
        "y2030": {"t0001": 0.7, "t0025": 0.3},
        "y2040": {"t0001": 0.55, "t0025": 0.45},
    },
    # Swapped case — the two y2030 reps' weights are exchanged (sum
    # preserved so ``period_share`` is unchanged).  Objective MUST move
    # vs the base case once the RP-weight fix lands (pre-fix delta == 0).
    f"{RP_INVEST_ALT}_swap": {
        "y2030": {"t0001": 0.3, "t0025": 0.7},
        "y2040": {"t0001": 0.55, "t0025": 0.45},
    },
    # Uniform case — unit weights (0.5/0.5 → folded 1.0/1.0).  Must be
    # byte-identical to a non-RP run: proves the fix is inert at w≡1.
    f"{RP_INVEST_ALT}_uniform": {
        "y2030": {"t0001": 0.5, "t0025": 0.5},
        "y2040": {"t0001": 0.5, "t0025": 0.5},
    },
}


def _build_rp_invest_overlay(
    base_entities: list[tuple], hz: Horizon
) -> dict[str, list]:
    """Return the additive Benders Phase-3b ``lh2_three_region_rp_invest``
    overlay — a representative-period, multi-period (y2030+y2040) invest
    model carrying NON-UNIT ``representative_period_weights``.

    Three sibling scenarios share one base topology, differing ONLY in the
    RP weight values (see :data:`_RP_WEIGHT_VARIANTS`):

    * ``lh2_three_region_rp_invest``        — 0.7/0.3, 0.55/0.45 (base)
    * ``lh2_three_region_rp_invest_swap``   — 0.3/0.7, 0.55/0.45 (reps swapped)
    * ``lh2_three_region_rp_invest_uniform``— 0.5/0.5, 0.5/0.5 (w≡1)

    Topology (layered ON TOP of the base ``ALT``):

    * Two FlexTool periods y2030+y2040, BOTH invest-eligible
      (``invest_periods=[y2030,y2040]``, ``years_represented={10,10}``).
    * A SEPARATE RP timeset per period (``rp_y2030`` / ``rp_y2040``); each
      splits the 48h horizon into two 24h representative blocks
      (starts ``t0001`` / ``t0025``) via ``timeset_duration`` and carries
      the variant's ``representative_period_weights``.
    * Greenfield investable pipes (``existing=0`` + ``invest_total``) with
      a NON-ZERO flow cost (``other_operational_cost=2.0``).
    * lh2 nodes bind with ``bind_within_period_blended_weights`` (the RP
      storage variant); battery left on the base ``bind_within_solve``.
    * Asymmetry forcing A→B→C trade: region A has cheap temporal wind
      (100/50/0 availability) + boosted existing wind/electrolyser/
      liquefier to supply the corridor; regions B/C have their
      electrolyser+liquefier capped at 10 (cannot self-make LH2) and their
      coal sized to cover LOCAL elec only; region C has zero local wind and
      a 120 kW/day lh2 demand it must import.

    Mirrors :func:`_build_trade_invest_overlay` structurally.  See
    ``specs/benders_option_c.md`` "RP-weight bug — fix design".
    """
    new_entities: list[tuple] = []
    pv: list[tuple] = []
    alternatives: list[tuple] = []
    scenarios: list[tuple] = []
    scenario_alternatives: list[tuple] = []
    ent_alts: list[tuple] = []

    # --- RP timesets: one per period per variant.  The timeset entities
    #     are SHARED across variants (same name) only when weights match;
    #     to keep each scenario self-contained we name the timesets per
    #     variant so swapping weights never bleeds across scenarios.
    for alt_name, per_period_weights in _RP_WEIGHT_VARIANTS.items():
        scen_name = alt_name  # scenario name == alternative name
        alternatives.append(
            (alt_name, f"Three-region LH2 RP-weight invest ({alt_name})"))
        scenarios.append(
            (scen_name, False,
             f"Three-region LH2 representative-period invest "
             f"({alt_name}) — RP-weight regression vehicle"))
        scenario_alternatives.append((scen_name, ALT, None))
        scenario_alternatives.append((scen_name, alt_name, None))

        period_timeset_rows: list[tuple[str, str]] = []
        for period, weights in per_period_weights.items():
            ts_name = f"rp_{period}__{alt_name}"
            new_entities.append(("timeset", ts_name))
            pv.extend([
                ("timeset", ts_name, "timeline", "y2030_168h", alt_name),
                ("timeset", ts_name, "timeset_duration",
                 _map([("t0001", 24.0), ("t0025", 24.0)]), alt_name),
                ("timeset", ts_name, "representative_period_weights",
                 _rp_weights_map({"t0001": weights}), alt_name),
            ])
            ent_alts.append(("timeset", (ts_name,), alt_name, True))
            period_timeset_rows.append((period, ts_name))

        # --- One Lagrangian invest solve per variant ----------------------
        solve_name = f"lh2_rp_invest__{alt_name}"
        new_entities.append(("solve", solve_name))
        pv.extend([
            ("solve", solve_name, "solve_mode", "single_solve", alt_name),
            ("solve", solve_name, "period_timeset",
             _period_timeset_map(period_timeset_rows), alt_name),
            ("solve", solve_name, "realized_periods",
             _array_str_periods(["y2030", "y2040"]), alt_name),
            ("solve", solve_name, "invest_periods",
             _array_str_periods(["y2030", "y2040"]), alt_name),
            ("solve", solve_name, "realized_invest_periods",
             _array_str_periods(["y2030", "y2040"]), alt_name),
            ("solve", solve_name, "years_represented",
             _years_represented_map([("y2030", 10.0), ("y2040", 10.0)]),
             alt_name),
        ])
        pv.append(
            ("model", "flexTool", "solves",
             _array_str([solve_name]), alt_name))
        ent_alts.append(("solve", (solve_name,), alt_name, True))

        # --- Shared topology overrides (per variant alternative) ----------
        # Greenfield investable pipes with non-zero flow cost.
        for pipe in ("pipe_AB", "pipe_BC"):
            pv.extend([
                ("connection", pipe, "existing", 0.0, alt_name),
                ("connection", pipe, "invest_method", "invest_total",
                 alt_name),
                ("connection", pipe, "invest_cost", 10.0, alt_name),
                ("connection", pipe, "invest_max_total", 5000.0, alt_name),
                ("connection", pipe, "lifetime", 25.0, alt_name),
                ("connection", pipe, "discount_rate", 0.05, alt_name),
                ("connection", pipe, "other_operational_cost", 2.0,
                 alt_name),
            ])
        # lh2 nodes: RP storage binding.
        for r in REGIONS:
            pv.append(
                ("node", f"lh2_{r}", "storage_binding_method",
                 "bind_within_period_blended_weights", alt_name))
        # Region A: cheap temporal wind + boosted supply chain.
        pv.append(
            ("profile", "wind_profile_A", "profile",
             _map(_step_wind_profile(hz)), alt_name))
        pv.append(("unit", "wind_A", "existing", 1600.0, alt_name))
        pv.append(
            ("connection", "electrolyser_A", "existing", 600.0, alt_name))
        pv.append(("unit", "liquefier_A", "existing", 600.0, alt_name))
        # Regions B/C: cannot self-make LH2; coal sized for local elec.
        for r in ("B", "C"):
            pv.append(
                ("connection", f"electrolyser_{r}", "existing", 10.0,
                 alt_name))
            pv.append(("unit", f"liquefier_{r}", "existing", 10.0, alt_name))
        pv.append(("unit", "coal_B", "existing", 620.0, alt_name))
        pv.append(("unit", "coal_C", "existing", 260.0, alt_name))
        pv.append(
            ("profile", "wind_profile_C", "profile",
             _map(_zero_profile(hz)), alt_name))
        pv.append(
            ("node", "lh2_C", "inflow",
             _map(_daily_lh2_demand(120.0, hz)), alt_name))

    return {
        "entities": new_entities,
        "parameter_values": pv,
        "alternatives": alternatives,
        "scenarios": scenarios,
        "scenario_alternatives": scenario_alternatives,
        "entity_alternatives": ent_alts,
    }


def _build_payload(
    hz: Horizon | None = None,
    extra_overlay: "callable | None" = None,
) -> dict[str, list]:
    """Return the import_data payload for the LH2 fixture.

    Composes onto the v51-migrated baseline tests.json (which already
    carries every entity_class, parameter_definition and value-list we
    need).

    Parameters
    ----------
    hz:
        Timeline horizon.  ``None`` (the default) reproduces the
        committed 168h / 7-day fixture byte-for-byte.
    extra_overlay:
        Optional callable ``f(base_entities) -> overlay_dict`` (same
        shape as :func:`_build_invest_overlay`) layered on top of the
        invest overlay.  Used by the Benders Phase-0 sibling fixture to
        add the greenfield-trade scenario.
    """
    if hz is None:
        hz = Horizon.default()
    entities: list[tuple] = []
    parameter_values: list[tuple] = []
    alternatives: list[tuple] = [(ALT, "Three-region LH2 fixture (Agent 1.9)")]
    scenarios: list[tuple] = [(SCENARIO, False, "Three-region LH2 fixture (Agent 1.9)")]
    scenario_alternatives: list[tuple] = [
        (SCENARIO, ALT, None),
    ]

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    entities.append(("timeline", "y2030_168h"))
    parameter_values.append(
        ("timeline", "y2030_168h", "timestep_duration",
         _map([(ts, 1.0) for ts in hz.hourly_steps]), ALT))

    entities.append(("timeset", "week168"))
    parameter_values.append(("timeset", "week168", "timeline", "y2030_168h", ALT))
    parameter_values.append(
        ("timeset", "week168", "timeset_duration",
         _map([(hz.hourly_steps[0], float(hz.n_hours))]), ALT))

    # ------------------------------------------------------------------
    # Solve / model
    # ------------------------------------------------------------------
    entities.append(("solve", "lh2_week"))
    parameter_values.extend([
        ("solve", "lh2_week", "solve_mode", "single_solve", ALT),
        # Batches C.3-C.5 retired the three string ``highs_*``
        # shortcuts entirely; per-solve HiGHS option overrides now
        # live on ``solver_arguments`` and are routed through the
        # engine-side _resolve_effective_highs_options.
        ("solve", "lh2_week", "period_timeset",
         _period_timeset_map([("y2030", "week168")]), ALT),
        ("solve", "lh2_week", "realized_periods",
         _array_str(["y2030"], "period"), ALT),
    ])
    parameter_values.append(
        ("model", "flexTool", "solves", _array_str(["lh2_week"]), ALT)
    )

    # ------------------------------------------------------------------
    # Resolution + decomposition groups
    # ------------------------------------------------------------------
    entities.append(("group", "hourly_group"))
    entities.append(("group", "daily_group"))
    parameter_values.append(("group", "hourly_group", "new_stepduration", 1.0, ALT))
    parameter_values.append(("group", "daily_group", "new_stepduration", 24.0, ALT))

    for r in REGIONS:
        entities.append(("group", f"region_{r}"))
        parameter_values.append(
            ("group", f"region_{r}", "decomposition_method",
             "lagrangian_region", ALT))

    # ------------------------------------------------------------------
    # Per-region nodes / processes
    # ------------------------------------------------------------------
    wind_scales = {"A": 0.55, "B": 0.45, "C": 0.85}
    wind_phases = {"A": 0.0, "B": 6.0, "C": 12.0}
    elec_peak = {"A": 700.0, "B": 600.0, "C": 200.0}
    elec_base = {"A": 400.0, "B": 350.0, "C": 100.0}
    lh2_daily_kw = {"A": 80.0, "B": 60.0, "C": 0.0}

    # Single global commodity node for coal fuel.
    entities.append(("commodity", "coal"))
    parameter_values.append(("commodity", "coal", "price", 30.0, ALT))
    entities.append(("node", "coal_market"))
    parameter_values.append(("node", "coal_market", "node_type", "commodity", ALT))
    entities.append(("group__node", ("hourly_group", "coal_market")))

    for r in REGIONS:
        elec = f"elec_{r}"
        h2 = f"h2_{r}"
        lh2 = f"lh2_{r}"
        wind = f"wind_{r}"
        coal = f"coal_{r}"
        battery = f"battery_{r}"
        battery_charge = f"battery_charge_{r}"
        battery_discharge = f"battery_discharge_{r}"
        electrolyser = f"electrolyser_{r}"
        liquefier = f"liquefier_{r}"
        wind_profile_name = f"wind_profile_{r}"

        # --- Nodes -----------------------------------------------------
        entities.extend([
            ("node", elec),
            ("node", h2),
            ("node", lh2),
            ("node", battery),
        ])
        parameter_values.extend([
            ("node", elec, "node_type", "balance", ALT),
            ("node", elec, "penalty_up", 8000.0, ALT),
            ("node", elec, "penalty_down", 8000.0, ALT),
            ("node", elec, "inflow", _map(_elec_demand(elec_peak[r], elec_base[r], hz)), ALT),
            ("node", h2, "node_type", "balance", ALT),
            ("node", h2, "penalty_up", 5000.0, ALT),
            ("node", h2, "penalty_down", 5000.0, ALT),
            ("node", lh2, "node_type", "storage", ALT),
            ("node", lh2, "existing", 5000.0, ALT),
            ("node", lh2, "storage_binding_method", "bind_within_solve", ALT),
            ("node", lh2, "storage_start_end_method", "fix_start", ALT),
            ("node", lh2, "storage_state_start", 0.5, ALT),
            ("node", lh2, "storage_solve_horizon_method", "free", ALT),
            ("node", lh2, "self_discharge_loss", 0.0, ALT),
            ("node", lh2, "penalty_up", 5000.0, ALT),
            ("node", lh2, "penalty_down", 5000.0, ALT),
            ("node", battery, "node_type", "storage", ALT),
            ("node", battery, "existing", 200.0, ALT),
            ("node", battery, "storage_binding_method", "bind_within_solve", ALT),
            ("node", battery, "storage_start_end_method", "fix_start", ALT),
            ("node", battery, "storage_state_start", 0.5, ALT),
            ("node", battery, "storage_solve_horizon_method", "free", ALT),
            ("node", battery, "self_discharge_loss", 0.0001, ALT),
            ("node", battery, "penalty_up", 3000.0, ALT),
            ("node", battery, "penalty_down", 3000.0, ALT),
        ])

        if lh2_daily_kw[r] > 0:
            parameter_values.append(
                ("node", lh2, "inflow",
                 _map(_daily_lh2_demand(lh2_daily_kw[r], hz)), ALT))

        # --- Resolution group memberships ------------------------------
        entities.append(("group__node", ("hourly_group", elec)))
        entities.append(("group__node", ("hourly_group", battery)))
        entities.append(("group__node", ("daily_group", h2)))
        entities.append(("group__node", ("daily_group", lh2)))

        # --- Decomposition group memberships ---------------------------
        for n in (elec, h2, lh2, battery):
            entities.append(("group__node", (f"region_{r}", n)))

        # --- Wind profile ---------------------------------------------
        entities.append(("profile", wind_profile_name))
        parameter_values.append(
            ("profile", wind_profile_name, "profile",
             _map(_wind_profile(wind_scales[r], wind_phases[r], hz)), ALT))

        # --- Wind unit ------------------------------------------------
        entities.append(("unit", wind))
        entities.append(("unit__outputNode", (wind, elec)))
        entities.append(("unit__node__profile", (wind, elec, wind_profile_name)))
        parameter_values.extend([
            ("unit", wind, "conversion_method", "none", ALT),
            ("unit", wind, "efficiency", 1.0, ALT),
            ("unit", wind, "existing", 800.0, ALT),
            ("unit__node__profile", (wind, elec, wind_profile_name),
             "profile_method", "upper_limit", ALT),
        ])
        entities.append(("group__unit", ("hourly_group", wind)))
        entities.append(("group__unit", (f"region_{r}", wind)))

        # --- Coal plant -----------------------------------------------
        entities.append(("unit", coal))
        entities.append(("unit__inputNode", (coal, "coal_market")))
        entities.append(("unit__outputNode", (coal, elec)))
        parameter_values.extend([
            ("unit", coal, "conversion_method", "constant_efficiency", ALT),
            ("unit", coal, "efficiency", 0.4, ALT),
            ("unit", coal, "existing", 400.0, ALT),
        ])
        entities.append(("group__unit", ("hourly_group", coal)))
        entities.append(("group__unit", (f"region_{r}", coal)))

        # --- Battery inverter (charge / discharge) --------------------
        entities.append(("unit", battery_charge))
        entities.append(("unit__inputNode", (battery_charge, elec)))
        entities.append(("unit__outputNode", (battery_charge, battery)))
        parameter_values.extend([
            ("unit", battery_charge, "conversion_method", "constant_efficiency", ALT),
            ("unit", battery_charge, "efficiency", 0.95, ALT),
            ("unit", battery_charge, "existing", 100.0, ALT),
        ])
        entities.append(("group__unit", ("hourly_group", battery_charge)))
        entities.append(("group__unit", (f"region_{r}", battery_charge)))

        entities.append(("unit", battery_discharge))
        entities.append(("unit__inputNode", (battery_discharge, battery)))
        entities.append(("unit__outputNode", (battery_discharge, elec)))
        parameter_values.extend([
            ("unit", battery_discharge, "conversion_method", "constant_efficiency", ALT),
            ("unit", battery_discharge, "efficiency", 0.95, ALT),
            ("unit", battery_discharge, "existing", 100.0, ALT),
        ])
        entities.append(("group__unit", ("hourly_group", battery_discharge)))
        entities.append(("group__unit", (f"region_{r}", battery_discharge)))

        # --- Electrolyser (process_connection, regular = indirect) ----
        entities.append(("connection", electrolyser))
        entities.append(("connection__node__node", (electrolyser, elec, h2)))
        parameter_values.extend([
            ("connection", electrolyser, "transfer_method", "regular", ALT),
            ("connection", electrolyser, "efficiency", 0.7, ALT),
            ("connection", electrolyser, "existing", 250.0, ALT),
        ])
        entities.append(("group__connection", (f"region_{r}", electrolyser)))

        # --- Liquefier (h2 → lh2, both daily) -------------------------
        entities.append(("unit", liquefier))
        entities.append(("unit__inputNode", (liquefier, h2)))
        entities.append(("unit__outputNode", (liquefier, lh2)))
        parameter_values.extend([
            ("unit", liquefier, "conversion_method", "constant_efficiency", ALT),
            ("unit", liquefier, "efficiency", 0.85, ALT),
            ("unit", liquefier, "existing", 200.0, ALT),
        ])
        entities.append(("group__unit", ("daily_group", liquefier)))
        entities.append(("group__unit", (f"region_{r}", liquefier)))

    # ------------------------------------------------------------------
    # Inter-region H2 pipelines (daily)
    # ------------------------------------------------------------------
    for src_r, dst_r in (("A", "B"), ("B", "C")):
        pipe = f"pipe_{src_r}{dst_r}"
        src = f"lh2_{src_r}"
        dst = f"lh2_{dst_r}"
        entities.append(("connection", pipe))
        entities.append(("connection__node__node", (pipe, src, dst)))
        parameter_values.extend([
            ("connection", pipe, "transfer_method", "regular", ALT),
            ("connection", pipe, "efficiency", 0.95, ALT),
            ("connection", pipe, "existing", 50.0, ALT),
        ])
        entities.append(("group__connection", ("daily_group", pipe)))
        entities.append(("group__connection", (f"region_{src_r}", pipe)))

    # Entity alternatives: every entity we created must be marked
    # active under the LH2 alternative so the scenario filter exposes
    # it.
    entity_alternatives: list[tuple] = []
    for ent in entities:
        cl = ent[0]
        name = ent[1]
        if isinstance(name, str):
            ent_byname: tuple = (name,)
        else:
            ent_byname = tuple(name)
        entity_alternatives.append((cl, ent_byname, ALT, True))

    # Additive invest layer (TIER 1).  Composes on top of the base
    # topology via a NEW alternative + scenario; the base scenario is
    # untouched.  New solves reference the existing base entities, so the
    # overlay only adds two ``solve`` entities + parameter values +
    # alternative/scenario rows.
    invest = _build_invest_overlay(entities)
    entities.extend(invest["entities"])
    parameter_values.extend(invest["parameter_values"])
    alternatives.extend(invest["alternatives"])
    scenarios.extend(invest["scenarios"])
    scenario_alternatives.extend(invest["scenario_alternatives"])
    entity_alternatives.extend(invest["entity_alternatives"])

    # Optional additional overlay (Benders Phase-0 greenfield-trade
    # scenario).  Layered last so it can override base values on the same
    # keys (the scenario lists ALT first, then the overlay's alternative).
    if extra_overlay is not None:
        overlay = extra_overlay(entities, hz)
        entities.extend(overlay["entities"])
        parameter_values.extend(overlay["parameter_values"])
        alternatives.extend(overlay["alternatives"])
        scenarios.extend(overlay["scenarios"])
        scenario_alternatives.extend(overlay["scenario_alternatives"])
        entity_alternatives.extend(overlay["entity_alternatives"])

    return {
        "entities": entities,
        "parameter_values": parameter_values,
        "alternatives": alternatives,
        "scenarios": scenarios,
        "scenario_alternatives": scenario_alternatives,
        "entity_alternatives": entity_alternatives,
    }


def _build_sqlite(
    db_path: Path,
    hz: Horizon | None = None,
    extra_overlay: "callable | None" = None,
) -> str:
    """Build the LH2 fixture as a fresh SQLite DB at *db_path*.

    ``hz`` / ``extra_overlay`` default to ``None`` ⇒ the committed
    168h / 7-day fixture, byte-identical to the legacy emit.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: import the baseline tests.json so we inherit every
    # entity_class, parameter_definition and value list FlexTool needs.
    json_to_db(HERE / "tests.json", db_path)

    # Step 2: apply just the v51 schema additions inline.  The full
    # migrate_database loop is not used here because the JSON snapshot
    # has stale ``model.version`` (38) but the schema is at v50, so
    # the loop would re-run already-applied migrations that mutate
    # value_lists no longer present.  The v51 step is purely additive.
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db:
        decomp_payload = {
            "parameter_value_lists": [
                ("decomposition_methods", "none"),
                ("decomposition_methods", "lagrangian_region"),
            ],
        }
        _, _ = import_data(db, **decomp_payload)
        default_val_none, default_type_none = to_database(None)
        default_val_str, default_type_str = to_database("none")
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="new_stepduration",
            default_value=default_val_none,
            default_type=default_type_none,
            parameter_type_list=("float",),
            description=(
                "Hours. Members of this group operate at this step "
                "duration. Overrides the solve-level new_stepduration "
                "for these entities."
            ),
        )
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="decomposition_method",
            default_value=default_val_str,
            default_type=default_type_str,
            parameter_value_list_name="decomposition_methods",
            parameter_type_list=("str",),
            description=(
                "Decomposition strategy.  'none' (default) leaves the "
                "group monolithic; 'lagrangian_region' marks it as an "
                "independent region for Agent 3.2's decomposition."
            ),
        )
        db.commit_session("Applied v51 group-block schema additions")

    # Step 3: layer the LH2 fixture on top.
    payload = _build_payload(hz=hz, extra_overlay=extra_overlay)
    with DatabaseMapping(url) as db:
        count, errors = import_data(db, **payload)
        if errors:
            raise RuntimeError(f"LH2 fixture import errors: {errors[:5]}")
        db.commit_session("Built LH2 three-region fixture")

    # Step 4: prune the baseline tests.json topology so the LH2 fixture
    # is the *only* input to the runner.
    _prune_baseline_topology(url)

    print(f"LH2 fixture built: {count} items at {db_path}")
    return url


def _prune_baseline_topology(url: str) -> None:
    """Delete every entity that is not part of the LH2 fixture.

    Keeps the entire schema (entity classes, parameter definitions,
    value lists, default values) intact — only entities and their
    parameter values are removed.  This sidesteps the
    ``_validate_timeline_timestep_duration`` failure on the residual
    ``y2020`` timeline from tests.json without rebuilding the schema.
    """
    keep_nodes: set[str] = set(["coal_market"])
    keep_units: set[str] = set()
    keep_connections: set[str] = set()
    keep_groups: set[str] = {"hourly_group", "daily_group"} | {f"region_{r}" for r in REGIONS}
    keep_profiles: set[str] = {f"wind_profile_{r}" for r in REGIONS}
    keep_commodities: set[str] = {"coal"}
    keep_timelines: set[str] = {"y2030_168h"}
    keep_timesets: set[str] = {"week168"}
    keep_solves: set[str] = {
        "lh2_week", "lh2_invest", "lh2_dispatch", "lh2_trade_invest",
    }
    # Phase-3b RP fixture: retain the per-variant RP timesets + invest
    # solves so the rp_invest scenarios survive the prune.  Names mirror
    # ``_build_rp_invest_overlay`` (``rp_<period>__<alt>`` /
    # ``lh2_rp_invest__<alt>``).  Harmless on the legacy / trade-invest
    # emits (those overlays never create these entities).
    for alt_name in _RP_WEIGHT_VARIANTS:
        keep_solves.add(f"lh2_rp_invest__{alt_name}")
        for period in ("y2030", "y2040"):
            keep_timesets.add(f"rp_{period}__{alt_name}")
    for r in REGIONS:
        keep_nodes |= {f"elec_{r}", f"h2_{r}", f"lh2_{r}", f"battery_{r}"}
        keep_units |= {
            f"wind_{r}", f"coal_{r}",
            f"battery_charge_{r}", f"battery_discharge_{r}",
            f"liquefier_{r}",
        }
        keep_connections |= {f"electrolyser_{r}"}
    for src_r, dst_r in (("A", "B"), ("B", "C")):
        keep_connections.add(f"pipe_{src_r}{dst_r}")

    keep_by_class: dict[str, set[str]] = {
        "node": keep_nodes,
        "unit": keep_units,
        "connection": keep_connections,
        "group": keep_groups,
        "profile": keep_profiles,
        "commodity": keep_commodities,
        "timeline": keep_timelines,
        "timeset": keep_timesets,
        "solve": keep_solves,
        "model": {"flexTool"},
        "reserve": set(),
        "constraint": set(),
        "upDown": set(),
    }

    with DatabaseMapping(url) as db:
        ids_to_remove: list[tuple[str, int]] = []
        for class_name, keepers in keep_by_class.items():
            for ent in db.find_entities(entity_class_name=class_name):
                name = ent["entity_byname"][0]
                if name not in keepers:
                    ids_to_remove.append(("entity", ent["id"]))
        all_to_remove_names: dict[str, set[str]] = {
            cl: {e["entity_byname"][0]
                 for e in db.find_entities(entity_class_name=cl)
                 if e["entity_byname"][0] not in keepers}
            for cl, keepers in keep_by_class.items()
        }
        for cl_def in [
            "commodity__node", "connection__node", "connection__profile",
            "group__connection", "group__node", "group__unit",
            "node__profile", "unit__inputNode", "unit__outputNode",
            "connection__node__node", "group__connection__node",
            "group__unit__node", "reserve__upDown__group",
            "unit__node__profile", "reserve__upDown__connection__node",
            "reserve__upDown__unit__node",
        ]:
            for ent in db.find_entities(entity_class_name=cl_def):
                bn = ent["entity_byname"]
                drop = False
                for comp in bn:
                    for cl_check, removed in all_to_remove_names.items():
                        if comp in removed:
                            drop = True
                            break
                    if drop:
                        break
                if drop:
                    ids_to_remove.append(("entity", ent["id"]))
        seen_ids: set[int] = set()
        to_remove: list[tuple[str, int]] = []
        for kind, ent_id in ids_to_remove:
            if ent_id in seen_ids:
                continue
            seen_ids.add(ent_id)
            to_remove.append((kind, ent_id))
        for kind, ent_id in to_remove:
            try:
                db.remove_items(kind, ent_id)
            except SpineDBAPIError:
                pass
        db.commit_session("Pruned baseline topology")


def regenerate_json(
    out_path: Path,
    hz: Horizon | None = None,
    extra_overlay: "callable | None" = None,
) -> int:
    """Rebuild the SQLite from scratch and export to JSON.

    ``hz`` / ``extra_overlay`` default to the committed 168h fixture.

    Returns the JSON file size in bytes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lh2_regen_") as td:
        db_path = Path(td) / "lh2_three_region.sqlite"
        _build_sqlite(db_path, hz=hz, extra_overlay=extra_overlay)
        db_to_json(db_path, out_path)
    return out_path.stat().st_size


def regenerate_trade_invest_json(out_path: Path = TRADE_INVEST_OUT) -> int:
    """Rebuild the Benders Phase-0 sibling fixture
    (``lh2_three_region_trade_invest.json``) at a 2-day / 48h horizon
    with the greenfield-trade overlay.  Independent of the committed
    ``lh2_three_region.json`` (which this function never touches)."""
    return regenerate_json(
        out_path,
        hz=Horizon(n_hours=48, n_days=2),
        extra_overlay=_build_trade_invest_overlay,
    )


def regenerate_rp_invest_json(out_path: Path = RP_INVEST_OUT) -> int:
    """Rebuild the Benders Phase-3b RP-weight fixture
    (``lh2_three_region_rp_invest.json``) at a 2-day / 48h horizon with
    the representative-period invest overlay (non-unit RP weights).
    Independent of ``lh2_three_region.json`` /
    ``lh2_three_region_trade_invest.json`` (never touched here)."""
    return regenerate_json(
        out_path,
        hz=Horizon(n_hours=48, n_days=2),
        extra_overlay=_build_rp_invest_overlay,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the LH2 three-region fixture and export it to JSON. "
            "Tests consume the JSON via tests.db_utils.json_to_db; this "
            "script is the source-of-truth regenerator."
        )
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output JSON path (default: tests/fixtures/lh2_three_region.json)",
    )
    parser.add_argument(
        "--trade-invest-out", type=Path, default=TRADE_INVEST_OUT,
        help=(
            "Output JSON path for the Benders Phase-0 sibling fixture "
            "(default: tests/fixtures/lh2_three_region_trade_invest.json)"
        ),
    )
    parser.add_argument(
        "--no-trade-invest", action="store_true",
        help="Skip emitting the Benders Phase-0 sibling fixture.",
    )
    parser.add_argument(
        "--rp-invest-out", type=Path, default=RP_INVEST_OUT,
        help=(
            "Output JSON path for the Benders Phase-3b RP-weight fixture "
            "(default: tests/fixtures/lh2_three_region_rp_invest.json)"
        ),
    )
    parser.add_argument(
        "--no-rp-invest", action="store_true",
        help="Skip emitting the Benders Phase-3b RP-weight fixture.",
    )
    args = parser.parse_args(argv)
    size = regenerate_json(args.out)
    print(f"Wrote {args.out}  ({size:,} bytes)")
    if not args.no_trade_invest:
        ti_size = regenerate_trade_invest_json(args.trade_invest_out)
        print(f"Wrote {args.trade_invest_out}  ({ti_size:,} bytes)")
    if not args.no_rp_invest:
        rp_size = regenerate_rp_invest_json(args.rp_invest_out)
        print(f"Wrote {args.rp_invest_out}  ({rp_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
