"""Build the three-region LH2 test fixture (Agent 1.9).

The fixture is the first non-degenerate exercise of the flex-temporal
infrastructure (Agents 1.1-1.8): three regions A/B/C, each with an
hourly electricity node and a daily H2 / LH2 storage node, wired by
H2 pipelines.

Layout per region
-----------------
    elec_<r>     (hourly_group)         — power balance node
    h2_<r>       (daily_group)          — gaseous H2 transit node
    lh2_<r>      (daily_group, storage) — liquid H2 storage

Per-region processes:
    wind_<r>     — variable RES (profile_method=upper_limit)
    coal_<r>     — dispatchable thermal (constant_efficiency, no UC)
    battery_<r> + battery_charge_<r> + battery_discharge_<r>
                 — storage node + bi-directional inverter
    electrolyser_<r> — *indirect* method: elec→H2.  Source side is
                       hourly (elec), sink side is daily (h2) — exactly
                       the case Agent 1.3 designed for.
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

import json
import math
import sys
from pathlib import Path
from typing import Any

from spinedb_api import DatabaseMapping, import_data, to_database

# Allow running this script standalone (python build_lh2_three_region.py out.sqlite)
HERE = Path(__file__).parent
TESTS_DIR = HERE.parent
REPO_ROOT = TESTS_DIR.parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from spinedb_api import to_database  # noqa: E402  (re-exported for clarity)
from spinedb_api.exception import SpineDBAPIError  # noqa: E402


# --- Time series synthesis --------------------------------------------------

N_HOURS = 168
N_DAYS = 7
HOURLY_STEPS: list[str] = [f"t{i:04d}" for i in range(1, N_HOURS + 1)]
DAILY_STEPS: list[str] = [HOURLY_STEPS[d * 24] for d in range(N_DAYS)]


def _wind_profile(scale: float, phase: float) -> dict[str, float]:
    """Sinusoidal wind profile over 168h, normalised to [0.05, 0.95]·scale.

    Period = 24h (one diurnal swing) plus a slow 168h drift to vary daily
    output.  Region C uses scale=0.85 (best site), A=0.55, B=0.45.
    """
    out: dict[str, float] = {}
    for i, ts in enumerate(HOURLY_STEPS):
        diurnal = 0.5 + 0.4 * math.cos((i + phase) * 2 * math.pi / 24)
        weekly = 0.85 + 0.15 * math.sin((i / 168.0) * 2 * math.pi)
        v = scale * diurnal * weekly
        out[ts] = round(max(0.0, min(1.0, v)), 6)
    return out


def _elec_demand(peak: float, base: float) -> dict[str, float]:
    """Hourly electricity demand profile.

    Returned as *negative* values (FlexTool convention: inflow<0 means
    energy leaves the node, i.e. demand).  Mild diurnal pattern.
    """
    out: dict[str, float] = {}
    for i, ts in enumerate(HOURLY_STEPS):
        diurnal = 0.7 + 0.3 * math.cos((i - 6) * 2 * math.pi / 24)
        v = base + (peak - base) * diurnal
        out[ts] = round(-v, 4)
    return out


def _daily_lh2_demand(daily_kw: float) -> dict[str, float]:
    """Daily LH2 demand at the lh2 storage node, indexed at the daily
    block's per-day step labels (one entry per day).

    The *value* is energy per timestep (FlexTool ``inflow`` semantics).
    Negative for demand.  Internally the daily block aggregates 24
    hourly steps; we report per-day energy directly because the daily
    block samples one inflow value per coarse step.
    """
    out: dict[str, float] = {}
    for d in range(N_DAYS):
        out[DAILY_STEPS[d]] = round(-daily_kw * 24, 4)
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


# --- Builder ---------------------------------------------------------------


REGIONS = ("A", "B", "C")
ALT = "lh2_three_region"
SCENARIO = "lh2_three_region"


def _build_payload() -> dict[str, list]:
    """Return the import_data payload for the LH2 fixture.

    Composes onto the v51-migrated baseline tests.json (which already
    carries every entity_class, parameter_definition and value-list we
    need).
    """
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
         _map([(ts, 1.0) for ts in HOURLY_STEPS]), ALT))

    entities.append(("timeset", "week168"))
    parameter_values.append(("timeset", "week168", "timeline", "y2030_168h", ALT))
    parameter_values.append(
        ("timeset", "week168", "timeset_duration",
         _map([(HOURLY_STEPS[0], float(N_HOURS))]), ALT))

    # ------------------------------------------------------------------
    # Solve / model
    # ------------------------------------------------------------------
    entities.append(("solve", "lh2_week"))
    parameter_values.extend([
        ("solve", "lh2_week", "solve_mode", "single_solve", ALT),
        ("solve", "lh2_week", "highs_method", "choose", ALT),
        ("solve", "lh2_week", "highs_parallel", "off", ALT),
        ("solve", "lh2_week", "highs_presolve", "on", ALT),
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

    # Single global commodity node for coal fuel.  Hourly so coal plants
    # (which run hourly) have their fuel-side block agree with the
    # process block — avoids the Agent 1.6 mismatched-side UC limitation
    # (no UC anywhere in this fixture, but worth keeping the topology
    # clean).
    entities.append(("commodity", "coal"))
    parameter_values.append(("commodity", "coal", "price", 30.0, ALT))
    entities.append(("node", "coal_market"))
    parameter_values.append(("node", "coal_market", "node_type", "commodity", ALT))
    # commodity_market is hourly by default → place in hourly_group so
    # its block matches its consumers' input side.
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
            ("node", elec, "inflow", _map(_elec_demand(elec_peak[r], elec_base[r])), ALT),
            ("node", h2, "node_type", "balance", ALT),
            ("node", h2, "penalty_up", 5000.0, ALT),
            ("node", h2, "penalty_down", 5000.0, ALT),
            # LH2 storage with a fixed start state — use_reference_value
            # so unbalanced solves do not snap the state to zero.
            ("node", lh2, "node_type", "storage", ALT),
            ("node", lh2, "existing", 5000.0, ALT),  # 5 MWh storage
            ("node", lh2, "storage_binding_method", "bind_within_solve", ALT),
            ("node", lh2, "storage_start_end_method", "fix_start", ALT),
            ("node", lh2, "storage_state_start", 0.5, ALT),
            ("node", lh2, "storage_solve_horizon_method", "free", ALT),
            ("node", lh2, "self_discharge_loss", 0.0, ALT),
            ("node", lh2, "penalty_up", 5000.0, ALT),
            ("node", lh2, "penalty_down", 5000.0, ALT),
            # Battery — small per-region storage at the elec node side.
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

        # LH2 demand — only A and B
        if lh2_daily_kw[r] > 0:
            parameter_values.append(
                ("node", lh2, "inflow",
                 _map(_daily_lh2_demand(lh2_daily_kw[r])), ALT))

        # --- Resolution group memberships ------------------------------
        # Elec + battery are hourly; H2, LH2 are daily.
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
             _map(_wind_profile(wind_scales[r], wind_phases[r])), ALT))

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
        # Place wind in hourly_group via its output node — but wind is a
        # process_unit, so use group__unit.
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
        # Two unidirectional units: charge (elec → battery) and discharge
        # (battery → elec).  Direct method, no UC.  Both at hourly block.
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
        # Source elec is hourly, sink h2 is daily — Agent 1.3's
        # M-matrix and overlap-set are exercised here.  Connections are
        # the only process kind that carry the indirect (regular)
        # transfer_method, so the electrolyser is modelled as a
        # ``connection`` rather than a ``unit`` (otherwise both sides
        # would collapse to the finer block under the unit's 1var
        # methods).
        entities.append(("connection", electrolyser))
        entities.append(("connection__node__node", (electrolyser, elec, h2)))
        parameter_values.extend([
            ("connection", electrolyser, "transfer_method", "regular", ALT),
            ("connection", electrolyser, "efficiency", 0.7, ALT),
            ("connection", electrolyser, "existing", 250.0, ALT),
        ])
        # No group__connection in hourly_group/daily_group → blocks.py
        # picks each side's block from its node's block (elec=hourly
        # source, h2=daily sink).  Process unified block = finer side
        # = hourly.
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
    # Connection between LH2 storage nodes.  connection__node__node
    # entity is (connection, source_node, sink_node).
    # ------------------------------------------------------------------
    for src_r, dst_r in (("A", "B"), ("B", "C")):
        pipe = f"pipe_{src_r}{dst_r}"
        src = f"lh2_{src_r}"
        dst = f"lh2_{dst_r}"
        entities.append(("connection", pipe))
        entities.append(("connection__node__node", (pipe, src, dst)))
        parameter_values.extend([
            # Bidirectional (regular) so v_flow_leftward exists.
            ("connection", pipe, "transfer_method", "regular", ALT),
            ("connection", pipe, "efficiency", 0.95, ALT),
            ("connection", pipe, "existing", 50.0, ALT),
        ])
        entities.append(("group__connection", ("daily_group", pipe)))
        # Pipe spans two regions — assign to source-region for tracking
        # (decomposition-method consumer only — not used by monolithic
        # solve).
        entities.append(("group__connection", (f"region_{src_r}", pipe)))

    # ------------------------------------------------------------------
    # Output flags
    # ------------------------------------------------------------------
    parameter_values.append(("model", "flexTool", "output_unit__node_flow_t",
                             "yes", ALT))
    parameter_values.append(("model", "flexTool", "output_node_balance_t",
                             "yes", ALT))
    parameter_values.append(("model", "flexTool", "output_connection__node__node_flow_t",
                             "yes", ALT))

    # Entity alternatives: every entity we created must be marked
    # active under the LH2 alternative so the scenario filter exposes
    # it.  The scenario_filter API overrides ``entity_sq`` such that an
    # entity is only visible if any active scenario alternative carries
    # an ``entity_alternative`` row marking it active.  Defaulting to
    # active is the natural intent for a fixture builder.
    entity_alternatives: list[tuple] = []
    for ent in entities:
        cl = ent[0]
        name = ent[1]
        # spinedb_api expects the entity name to be a tuple even for
        # single-dim classes.
        if isinstance(name, str):
            ent_byname: tuple = (name,)
        else:
            ent_byname = tuple(name)
        entity_alternatives.append((cl, ent_byname, ALT, True))

    return {
        "entities": entities,
        "parameter_values": parameter_values,
        "alternatives": alternatives,
        "scenarios": scenarios,
        "scenario_alternatives": scenario_alternatives,
        "entity_alternatives": entity_alternatives,
    }


def build(db_path: Path) -> str:
    """Build the LH2 fixture as a fresh SQLite DB at *db_path*.

    Returns the ``sqlite:///...`` URL.
    """
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: import the baseline tests.json so we inherit every
    # entity_class, parameter_definition and value list FlexTool needs.
    # The JSON snapshot was taken at v50 schema (its parameter
    # default_value for ``model.version`` is stale at 38, but the
    # schema is already v50 — solve.new_stepduration is present).
    json_to_db(HERE / "tests.json", db_path)

    # Step 2: apply just the v51 schema additions inline.  The full
    # migrate_database loop is not used here because the JSON snapshot
    # has stale ``model.version`` (38) but the schema is at v50, so
    # the loop would re-run already-applied migrations that mutate
    # value_lists no longer present.  The v51 step is purely additive
    # (two parameter_definitions on entity_class ``group`` + one new
    # value list), so duplicating it here keeps the fixture builder
    # self-contained.
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db:
        # Value list for decomposition_method.
        decomp_payload = {
            "parameter_value_lists": [
                ("decomposition_methods", "none"),
                ("decomposition_methods", "lagrangian_region"),
            ],
        }
        _, _ = import_data(db, **decomp_payload)
        # Two new group-level parameter_definitions.
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
    payload = _build_payload()
    with DatabaseMapping(url) as db:
        count, errors = import_data(db, **payload)
        if errors:
            raise RuntimeError(f"LH2 fixture import errors: {errors[:5]}")
        db.commit_session("Built LH2 three-region fixture")

    # Step 4: prune the baseline tests.json topology so the LH2 fixture
    # is the *only* input to the runner.  The validator
    # ``_validate_timeline_timestep_duration`` walks every ``timeline``
    # entity in the DB regardless of the scenario filter, and the
    # baseline ``y2020`` timeline only carries its ``timestep_duration``
    # under the ``init`` alternative — under ``lh2_three_region`` the
    # value is invisible and the validator trips.  Same logic applies to
    # other entity classes whose presence in the entity table
    # short-circuits scenario filtering (model.solves array,
    # entity.process_unit etc.).  Pruning is a fixture-builder concern,
    # not a runtime concern.
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
    keep_solves: set[str] = {"lh2_week"}
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

    # Map entity_class → set of names to keep (single-dim only — keep
    # all multi-dim entities whose components survive the prune).
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
        # Single ``model`` instance ``flexTool`` is preserved; any
        # other model entity (none in the fixture) would be dropped.
        "model": {"flexTool"},
        # Reserves / constraints / upDown — drop everything inherited
        # from tests.json (the LH2 fixture defines none).
        "reserve": set(),
        "constraint": set(),
        "upDown": set(),
    }

    # First pass: collect entity ids to remove for each single-dim class.
    with DatabaseMapping(url) as db:
        ids_to_remove: list[tuple[str, int]] = []
        for class_name, keepers in keep_by_class.items():
            for ent in db.find_entities(entity_class_name=class_name):
                name = ent["entity_byname"][0]
                if name not in keepers:
                    ids_to_remove.append(("entity", ent["id"]))
        # Second pass: any multi-dim entity referencing a removed
        # single-dim entity must also go.  spinedb_api cascades on
        # entity removal but explicit removal is safer for deeply
        # nested classes.
        # We iterate every multi-dim class and drop entities whose
        # components contain a name that is no longer kept.
        all_to_remove_names: dict[str, set[str]] = {
            cl: {e["entity_byname"][0]
                 for e in db.find_entities(entity_class_name=cl)
                 if e["entity_byname"][0] not in keepers}
            for cl, keepers in keep_by_class.items()
        }
        # Map of (single-dim class) → set of removed names, for fast
        # subset checks.
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
                # Check if any component name appears in
                # all_to_remove_names for the matching component class.
                # Use a permissive map: if a component name matches a
                # single-dim removed name in any class, drop the
                # multi-dim row.
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
        # Deduplicate
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
                # Already gone (cascade from a parent removal).
                pass
        db.commit_session("Pruned baseline topology")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lh2_three_region.sqlite")
    build(out)
