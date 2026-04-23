#!/usr/bin/env python3
"""Generate the composite benchmark scenario.

The composite scenario is the primary target of the LP-scaling work: it has
units spanning ~8 orders of magnitude in capacity, which exposes the
matrix-coefficient bimodality that solver-internal scaling cannot fix alone.

Approach: copy ``templates/examples.sqlite`` and add a new alternative
``composite_scales`` that layers these entities onto the ``network_all_tech``
system:

    building_node        (tiny demand node,    ~0.005 MW scale)
      └── tiny_heatpump   (0.01 MW heatpump, west → building_node)
      └── tiny_battery    (0.005 MW storage node attached to building)
    west (electricity hub, reused from examples.sqlite)
      └── mega_coal       (10000 MW coal unit — dwarfs existing 500 MW)
      └── mega_wind       (5000 MW VRE with profile)
    east (reused)
      └── interconnector  (existing ``west_east``, upgraded to 500 MW)

Then a scenario ``composite_benchmark`` uses the base ``network_all_tech``
chain plus the new alternative.

Timeline is the examples.sqlite 2-day timeset (48h); expanding to 720h is
out of scope for this generator (would require rebuilding profiles).

Usage:
    python benchmarks/scaling/scenarios/composite/generate.py

Output:
    benchmarks/scaling/scenarios/composite/input.sqlite
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from spinedb_api import DatabaseMapping

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DB = REPO_ROOT / "templates" / "examples.sqlite"
OUT_DIR = Path(__file__).resolve().parent
OUT_DB = OUT_DIR / "input.sqlite"

ALT_NAME = "composite_scales"
NEW_SCENARIO = "composite_benchmark"
SCENARIO_NAME = NEW_SCENARIO  # harness reads this attribute


def float_value(v: float) -> bytes:
    """Raw JSON-encoded float payload for a Spine DB ``float`` parameter value."""
    return json.dumps(v).encode("utf-8")


def str_value(v: str) -> bytes:
    return json.dumps(v).encode("utf-8")


def main() -> int:
    if not SRC_DB.exists():
        print(f"ERROR: source DB not found: {SRC_DB}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DB.unlink(missing_ok=True)
    shutil.copy2(SRC_DB, OUT_DB)

    url = f"sqlite:///{OUT_DB.resolve()}"
    with DatabaseMapping(url) as db:
        # Add composite alternative.
        db.add_item(
            "alternative",
            name=ALT_NAME,
            description="Composite scales: tiny building units + oversized continental units",
        )

        # ------------------------------------------------------------------
        # New entities (in entity classes that already exist in examples.sqlite)
        # ------------------------------------------------------------------
        def add_entity(cls: str, byname: tuple[str, ...]):
            _, err = db.add_item(
                "entity", entity_class_name=cls, entity_byname=byname
            )
            if err and "already exists" not in str(err).lower():
                print(f"WARN {cls}/{byname}: {err}")
            # Also activate this entity in the composite alternative so it
            # becomes part of the scenario.
            _, err2 = db.add_item(
                "entity_alternative",
                entity_class_name=cls,
                entity_byname=byname,
                alternative_name=ALT_NAME,
                active=True,
            )
            if err2 and "already exists" not in str(err2).lower():
                print(f"WARN entity_alt {cls}/{byname}/{ALT_NAME}: {err2}")

        def add_param(
            cls: str,
            byname: tuple[str, ...],
            pname: str,
            value: bytes,
            ptype: str = "float",
        ):
            _, err = db.add_item(
                "parameter_value",
                entity_class_name=cls,
                entity_byname=byname,
                parameter_definition_name=pname,
                alternative_name=ALT_NAME,
                value=value,
                type=ptype,
            )
            if err:
                print(f"WARN param {cls}/{byname}/{pname}: {err}")

        # --- Tiny building node: 0.005 MW demand scale ---
        add_entity("node", ("building_node",))
        add_param("node", ("building_node",), "has_balance", str_value("yes"), "str")
        add_param("node", ("building_node",), "penalty_up", float_value(900.0))
        add_param("node", ("building_node",), "penalty_down", float_value(800.0))
        # Constant demand of 3 kW, indexed by each timestep in the 2-day timeset.
        inflow_map = {
            "index_type": "str",
            "rank": 1,
            "data": [[f"t{i:04d}", -0.003] for i in range(1, 49)],
        }
        add_param(
            "node",
            ("building_node",),
            "inflow",
            json.dumps(inflow_map).encode("utf-8"),
            "map",
        )

        # --- Tiny heatpump: 0.01 MW from west → building_node ---
        add_entity("unit", ("tiny_heatpump",))
        add_param("unit", ("tiny_heatpump",), "existing", float_value(0.01))
        add_param("unit", ("tiny_heatpump",), "efficiency", float_value(3.0))
        add_entity("unit__inputNode", ("tiny_heatpump", "west"))
        add_entity("unit__outputNode", ("tiny_heatpump", "building_node"))

        # --- Tiny storage node: 0.005 MWh (5 Wh scale) ---
        add_entity("node", ("tiny_battery_storage",))
        add_param("node", ("tiny_battery_storage",), "has_balance", str_value("yes"), "str")
        add_param("node", ("tiny_battery_storage",), "has_storage", str_value("yes"), "str")
        add_param("node", ("tiny_battery_storage",), "storage_state_reference_value", float_value(0.0025))
        add_param("node", ("tiny_battery_storage",), "storage_state_reference_price", float_value(0.0))
        add_param("node", ("tiny_battery_storage",), "penalty_up", float_value(900.0))
        add_param("node", ("tiny_battery_storage",), "penalty_down", float_value(800.0))

        # --- Mega coal: 10 000 MW on west ---
        add_entity("unit", ("mega_coal",))
        add_param("unit", ("mega_coal",), "existing", float_value(10000.0))
        add_param("unit", ("mega_coal",), "efficiency", float_value(0.4))
        add_entity("unit__inputNode", ("mega_coal", "coal_market"))
        add_entity("unit__outputNode", ("mega_coal", "west"))

        # --- Mega wind: 5 000 MW dispatchable VRE proxy on west ---
        # Simplified as a dispatchable source with existing capacity only
        # (no availability profile — keeps the generator self-contained).
        add_entity("unit", ("mega_wind",))
        add_param("unit", ("mega_wind",), "existing", float_value(5000.0))
        add_param("unit", ("mega_wind",), "efficiency", float_value(1.0))
        add_entity("unit__outputNode", ("mega_wind", "west"))

        # ------------------------------------------------------------------
        # Scenario wiring — reuse network_all_tech chain + add composite alt
        # ------------------------------------------------------------------
        sas = list(db.get_scenario_alternative_items())
        base_chain = sorted(
            [sa for sa in sas if sa["scenario_name"] == "network_all_tech"],
            key=lambda x: x["rank"],
        )
        base_alts = [sa["alternative_name"] for sa in base_chain]

        _, err = db.add_item(
            "scenario",
            name=NEW_SCENARIO,
            active=False,
            description=(
                "Composite benchmark: network_all_tech + tiny building units "
                "+ mega continental units (~7 orders of magnitude)"
            ),
        )
        if err:
            print(f"WARN scenario {NEW_SCENARIO}: {err}")

        full_chain = base_alts + [ALT_NAME]
        for i, alt in enumerate(full_chain):
            _, err = db.add_item(
                "scenario_alternative",
                scenario_name=NEW_SCENARIO,
                alternative_name=alt,
                rank=i + 1,
            )
            if err:
                print(f"WARN scen_alt {NEW_SCENARIO}/{alt}: {err}")

        db.commit_session(f"Add composite benchmark scenario ({ALT_NAME})")

    print(f"Wrote {OUT_DB}")
    print(f"Scenario to run: {NEW_SCENARIO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
