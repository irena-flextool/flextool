"""A tiny one-node model database built from the schema JSON in the test (never a checked-in ``.sqlite``).

Timeline ``tl``: t1..t4, 1 h steps.  Node ``n`` with demand 50, 60, 80, 70 MWh and an upward penalty of
10 000 CUR/MWh.  Unit ``gen`` -> ``n``: no existing capacity, invest_cost 100 CUR/kW, lifetime 1 year and discount
rate 0.05 (annuity 1.05 × 1e5 = 1.05e5 CUR/MW/a), other_operational_cost 1 CUR/MWh.  One period ``p1``, years_represented 1.

Timeset ``rp``: ``timeset_duration {t1: 2}`` with ``representative_period_weights`` {t1: {t1: 1}, t3: {t1: 1}}
(base periods t1 and t3 are both represented by the block t1..t2).  Timeset ``full``: ``timeset_duration {t1: 4}``.

The solves are given by the caller as ``{solve: {parameter: value}}``; ``model.solves`` lists them in that order.
Values are spinedb_api objects or plain floats / strings.
"""
from __future__ import annotations

from pathlib import Path

import spinedb_api as api

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_JSON = FLEXTOOL_ROOT / "flextool" / "schemas" / "spinedb_schema.json"
ALT = "tiny"
SCENARIO = "tiny_s"
DEMAND = {"t1": 50.0, "t2": 60.0, "t3": 80.0, "t4": 70.0}


def tmap(data: dict, index_name: str = "time") -> api.Map:
    return api.Map(list(data), list(data.values()), index_name=index_name)


def arr(values: list[str]) -> api.Array:
    return api.Array(list(values))


def build_db(tmp_dir: Path, solves: dict[str, dict]) -> str:
    """Build the database in *tmp_dir* and return its URL; the scenario is :data:`SCENARIO`."""
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    db_path = Path(tmp_dir) / "tiny.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db_path))
    url = f"sqlite:///{db_path.resolve().as_posix()}"

    entities = [
        ("model", "flextool"), ("timeline", "tl"), ("timeset", "rp"), ("timeset", "full"),
        ("node", "n"), ("unit", "gen"), ("unit__outputNode", ("gen", "n")),
    ] + [("solve", s) for s in solves]
    rp_weights = api.Map(
        ["t1", "t3"],
        [tmap({"t1": 1.0}, "representative_period"), tmap({"t1": 1.0}, "representative_period")],
        index_name="base_period",
    )
    values = [
        ("model", "flextool", "solves", arr(list(solves))),
        ("timeline", "tl", "timestep_duration", tmap({t: 1.0 for t in DEMAND})),
        ("timeset", "rp", "timeline", "tl"),
        ("timeset", "rp", "timeset_duration", tmap({"t1": 2.0})),
        ("timeset", "rp", "representative_period_weights", rp_weights),
        ("timeset", "full", "timeline", "tl"),
        ("timeset", "full", "timeset_duration", tmap({"t1": 4.0})),
        ("node", "n", "node_type", "balance"),
        ("node", "n", "inflow", tmap({t: -v for t, v in DEMAND.items()})),
        ("node", "n", "penalty_up", 10000.0),
        ("unit", "gen", "efficiency", 1.0),
        ("unit", "gen", "invest_method", "invest_no_limit"),
        ("unit", "gen", "invest_cost", 100.0),
        ("unit", "gen", "lifetime", 1.0),
        ("unit", "gen", "discount_rate", 0.05),
        ("unit__outputNode", ("gen", "n"), "other_operational_cost", 1.0),
    ]
    for solve, params in solves.items():
        for par, val in params.items():
            values.append(("solve", solve, par, val))

    with open_database_mapping(url) as db:
        count, errors = api.import_data(
            db,
            alternatives=[(ALT, "")],
            scenarios=[(SCENARIO, False, "")],
            scenario_alternatives=[(SCENARIO, ALT, None)],
            entities=entities,
            entity_alternatives=[("node", ("n",), ALT, True), ("unit", ("gen",), ALT, True)],
            parameter_values=[(c, e, p, v, ALT) for c, e, p, v in values],
        )
        if errors:
            raise RuntimeError(f"import errors: {errors[:5]}")
        db.commit_session("tiny model")
    return url


def invest_rp_solve(**extra) -> dict:
    """The investment solve on the representative timeset ``rp``."""
    return {
        "solve_mode": "single_solve",
        "period_timeset": tmap({"p1": "rp"}, "period"),
        "years_represented": tmap({"p1": 1.0}, "period"),
        "invest_periods": arr(["p1"]),
        "realized_invest_periods": arr(["p1"]),
        **extra,
    }


def dispatch_solve(**extra) -> dict:
    """The dispatch solve on the full timeset."""
    return {
        "solve_mode": "single_solve",
        "period_timeset": tmap({"p1": "full"}, "period"),
        "years_represented": tmap({"p1": 1.0}, "period"),
        "realized_periods": arr(["p1"]),
        **extra,
    }
