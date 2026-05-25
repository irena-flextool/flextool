"""Stage 4 Phase 3d augmentations.

Adds scenarios/alternatives to ``tests/fixtures/tests.json`` and
``tests/fixtures/stochastics.json`` so that the remaining disk-only
``work_<scenario>/`` fixtures can be reproduced by ``scenario_workdir``.

Augmentations
-------------
``tests.json``:
  * alt ``ladder_ann_on`` — sets ``commodity.coal.price_method =
    'price_ladder_annual'`` and a 3d-map ``price_ladder_annual``.
  * alt ``ladder_cum_on`` — sets ``commodity.coal.price_method =
    'price_ladder_cumulative'`` and a 2d-map ``price_ladder_cumulative``.
  * scenario ``coal_ladder_annual`` = ``coal`` + ``ladder_ann_on``.
  * scenario ``coal_ladder_cumulative`` = ``coal`` + ``ladder_cum_on``.
  * alt ``inflation_2pct`` — overrides ``model.flexTool.inflation_rate``
    from 0.04 to 0.02.  This is the patch on top of
    ``wind_battery_invest_lifetime_renew`` used by
    ``test_db_direct_inflation_2pct.py``.  Add a copy scenario
    ``wind_battery_invest_lifetime_renew_inflation_2pct`` =
    ``wind_battery_invest_lifetime_renew`` + ``inflation_2pct``.
  * alt ``delay_source_coef_on`` — sets
    ``unit__inputNode.(water_pump, water_source).flow_coefficient = 2.0``
    on top of ``water_pump_delayed`` so the .mod source-coef multiplier
    on delayed flows is exercised (flextool.mod:2573).
  * scenario ``delay_source_coef`` = ``water_pump_delayed`` +
    ``delay_source_coef_on``.

``stochastics.json``:
  * alt ``no_storage_override`` — sets ``node.hydro_reservoir.node_type
    = 'commodity'``.  Drains storage so the LP exercises stochastic
    branches with no carrier between periods.
  * scenario ``2_day_stochastic_dispatch_no_storage`` =
    ``2_day_stochastic_dispatch`` + ``no_storage_override``.

All additions are append-only.  Idempotent: re-running is a no-op
(``import_data`` silently skips existing rows).

Run::

    python tests/fixtures/_augment_phase3d.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from db_utils import db_to_json, json_to_db  # noqa: E402

from spinedb_api import (  # noqa: E402
    DatabaseMapping,
    Map,
    to_database,
)

MIGRATE_SCRIPT = Path("/home/jkiviluo/sources/flextool/migrate_database.py")
WORK_DIR = Path("/tmp/phase3d_build")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _ladder_annual_map_3d() -> Map:
    """``period -> tier -> {price, quantity}`` 3d map.

    Reproduces ``work_commodity_ladder_annual``'s ``coal`` value::

        p2020:
          tier 1: price=20, quantity=1
          tier 2: price=30, quantity=Infinity
    """
    inner_t1 = Map(["price", "quantity"], [20.0, 1.0])
    inner_t2 = Map(["price", "quantity"], [30.0, float("inf")])
    tiers = Map(["1", "2"], [inner_t1, inner_t2], index_name="tier")
    return Map(["p2020"], [tiers], index_name="period")


def _ladder_cumulative_map_2d() -> Map:
    """``tier -> {price, quantity}`` 2d map.

    Reproduces ``work_commodity_ladder_cumulative``'s ``coal`` value::

        tier 1: price=20, quantity=1
        tier 2: price=30, quantity=Infinity
    """
    inner_t1 = Map(["price", "quantity"], [20.0, 1.0])
    inner_t2 = Map(["price", "quantity"], [30.0, float("inf")])
    return Map(["1", "2"], [inner_t1, inner_t2], index_name="tier")


# ---------------------------------------------------------------------------
# Augment tests.json
# ---------------------------------------------------------------------------


def _augment_tests_db(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db:
        # Alternatives
        for alt_name, desc in (
            ("ladder_ann_on", "Phase 3d: enable coal annual price ladder."),
            ("ladder_cum_on", "Phase 3d: enable coal cumulative price ladder."),
            ("inflation_2pct",
                "Phase 3d: override model.inflation_rate to 0.02 for "
                "test_db_direct_inflation_2pct.py"),
            ("delay_source_coef_on",
                "Phase 3d: override unit__inputNode."
                "(water_pump, water_source).flow_coefficient from 1.0 to "
                "2.0 for test_flex_delay_source_coef.py."),
        ):
            db.add_alternative_item(name=alt_name, description=desc)

        # Parameter values (alt ladder_ann_on)
        for alt, pd_name, val in (
            ("ladder_ann_on", "price_method", "price_ladder_annual"),
            ("ladder_cum_on", "price_method", "price_ladder_cumulative"),
        ):
            db_val, db_type = to_database(val)
            db.add_parameter_value_item(
                entity_class_name="commodity",
                entity_byname=("coal",),
                parameter_definition_name=pd_name,
                alternative_name=alt,
                value=db_val,
                type=db_type,
            )

        # Map-valued parameter values
        ann_map = _ladder_annual_map_3d()
        db_val, db_type = to_database(ann_map)
        db.add_parameter_value_item(
            entity_class_name="commodity",
            entity_byname=("coal",),
            parameter_definition_name="price_ladder_annual",
            alternative_name="ladder_ann_on",
            value=db_val,
            type=db_type,
        )

        cum_map = _ladder_cumulative_map_2d()
        db_val, db_type = to_database(cum_map)
        db.add_parameter_value_item(
            entity_class_name="commodity",
            entity_byname=("coal",),
            parameter_definition_name="price_ladder_cumulative",
            alternative_name="ladder_cum_on",
            value=db_val,
            type=db_type,
        )

        # inflation_2pct: override model.flexTool.inflation_rate
        db_val, db_type = to_database(0.02)
        db.add_parameter_value_item(
            entity_class_name="model",
            entity_byname=("flexTool",),
            parameter_definition_name="inflation_rate",
            alternative_name="inflation_2pct",
            value=db_val,
            type=db_type,
        )

        # delay_source_coef_on: override
        # unit__inputNode.(water_pump, water_source).flow_coefficient
        db_val, db_type = to_database(2.0)
        db.add_parameter_value_item(
            entity_class_name="unit__inputNode",
            entity_byname=("water_pump", "water_source"),
            parameter_definition_name="flow_coefficient",
            alternative_name="delay_source_coef_on",
            value=db_val,
            type=db_type,
        )

        # Scenarios
        scenarios_to_add = (
            ("coal_ladder_annual",
                ["init", "west", "coal", "ladder_ann_on"]),
            ("coal_ladder_cumulative",
                ["init", "west", "coal", "ladder_cum_on"]),
        )
        for sc_name, alts in scenarios_to_add:
            db.add_scenario_item(name=sc_name)
            for rank, alt_name in enumerate(alts):
                db.add_scenario_alternative_item(
                    scenario_name=sc_name,
                    alternative_name=alt_name,
                    rank=rank + 1,
                )

        # delay_source_coef: copy water_pump_delayed alternative chain
        # and append delay_source_coef_on.
        sc_wpd = db.get_scenario_item(name="water_pump_delayed")
        if sc_wpd is None:
            raise RuntimeError(
                "water_pump_delayed scenario missing")
        sas_wpd = db.find_scenario_alternatives(
            scenario_name="water_pump_delayed"
        )
        sas_wpd_sorted = sorted(sas_wpd, key=lambda r: r["rank"])
        new_chain_wpd = [
            r["alternative_name"] for r in sas_wpd_sorted
        ] + ["delay_source_coef_on"]
        new_scen_wpd = "delay_source_coef"
        db.add_scenario_item(name=new_scen_wpd)
        for rank, alt_name in enumerate(new_chain_wpd):
            db.add_scenario_alternative_item(
                scenario_name=new_scen_wpd,
                alternative_name=alt_name,
                rank=rank + 1,
            )

        # wind_battery_invest_lifetime_renew_inflation_2pct: copy
        # wind_battery_invest_lifetime_renew alternative chain and append
        # inflation_2pct.
        sc_renew = db.get_scenario_item(name="wind_battery_invest_lifetime_renew")
        if sc_renew is None:
            raise RuntimeError(
                "wind_battery_invest_lifetime_renew scenario missing")
        # Fetch its alternative chain in order.
        sas = db.find_scenario_alternatives(
            scenario_name="wind_battery_invest_lifetime_renew"
        )
        sas_sorted = sorted(sas, key=lambda r: r["rank"])
        new_chain = [r["alternative_name"] for r in sas_sorted] + ["inflation_2pct"]
        new_scen = "wind_battery_invest_lifetime_renew_inflation_2pct"
        db.add_scenario_item(name=new_scen)
        for rank, alt_name in enumerate(new_chain):
            db.add_scenario_alternative_item(
                scenario_name=new_scen,
                alternative_name=alt_name,
                rank=rank + 1,
            )

        db.commit_session("Phase 3d augmentations: ladders + inflation_2pct")
    print(f"augmented {db_path}")


def _augment_stochastics_db(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db:
        db.add_alternative_item(
            name="no_storage_override",
            description=(
                "Phase 3d: drain hydro_reservoir storage by setting "
                "node_type='commodity'."),
        )
        db_val, db_type = to_database("commodity")
        db.add_parameter_value_item(
            entity_class_name="node",
            entity_byname=("hydro_reservoir",),
            parameter_definition_name="node_type",
            alternative_name="no_storage_override",
            value=db_val,
            type=db_type,
        )

        # Scenario: 2_day_stochastic_dispatch + no_storage_override.
        sas = db.find_scenario_alternatives(
            scenario_name="2_day_stochastic_dispatch"
        )
        if not sas:
            raise RuntimeError(
                "2_day_stochastic_dispatch scenario missing from "
                "stochastics.json"
            )
        chain = [r["alternative_name"] for r in sorted(sas, key=lambda r: r["rank"])]
        chain = chain + ["no_storage_override"]
        new_scen = "2_day_stochastic_dispatch_no_storage"
        db.add_scenario_item(name=new_scen)
        for rank, alt_name in enumerate(chain):
            db.add_scenario_alternative_item(
                scenario_name=new_scen,
                alternative_name=alt_name,
                rank=rank + 1,
            )

        db.commit_session("Phase 3d augment: no_storage_override")
    print(f"augmented {db_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _round_trip(src_json: Path, augment_fn) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp_db = WORK_DIR / (src_json.stem + "_round_trip.sqlite")
    if tmp_db.exists():
        tmp_db.unlink()

    json_to_db(src_json, tmp_db)
    if MIGRATE_SCRIPT.exists():
        subprocess.run(
            [sys.executable, str(MIGRATE_SCRIPT), str(tmp_db)],
            check=True,
        )
    augment_fn(tmp_db)
    db_to_json(tmp_db, src_json)
    print(f"wrote augmented JSON -> {src_json}")


def main() -> None:
    _round_trip(REPO_ROOT / "tests/fixtures/tests.json", _augment_tests_db)
    _round_trip(
        REPO_ROOT / "tests/fixtures/stochastics.json",
        _augment_stochastics_db,
    )


if __name__ == "__main__":
    main()
