"""Small databases of the piecewise_linear oracle cases (``specs/piecewise_efficiency_oracle.md`` §1).

The schema (entity classes, parameter definitions, value lists) is that of ``tests/fixtures/tests.json``; the
data is written here from scratch: one alternative ``base`` and the scenario ``oracle``.  Databases are built under
the pytest tmp dir and migrated (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine._unit_db import b64

REPO = Path(__file__).resolve().parents[2]
SCHEMA_JSON = REPO / "tests" / "fixtures" / "tests.json"
SCHEMA_KEYS = ("entity_classes", "parameter_definitions", "parameter_types", "parameter_value_lists",
               "list_values", "parameter_groups")


def scalar(v: float) -> list:
    return b64(float(v), "float")


def text(v: str) -> list:
    return b64(v, "str")


def array(values: list[str], index_name: str | None = None) -> list:
    x: dict = {"value_type": "str", "data": list(values)}
    if index_name:
        x["index_name"] = index_name
    return b64(x, "array")


def str_map(pairs: dict, index_name: str | None = None) -> list:
    x: dict = {"index_type": "str", "data": [[k, v] for k, v in pairs.items()]}
    if index_name:
        x["index_name"] = index_name
    return b64(x, "map")


def curve(points: list[tuple[float, float]]) -> list:
    """A curve (map over load shares, float keys)."""
    return b64({"index_type": "float", "data": [[k, v] for k, v in points]}, "map")


def curve_per_period(per: dict[str, list[tuple[float, float]]]) -> list:
    """A curve per period (map over periods of maps over load shares)."""
    return b64({"index_type": "str", "index_name": "period", "data": [
        [d, {"type": "map", "index_type": "float", "data": [[k, v] for k, v in pts]}] for d, pts in per.items()]},
        "map")


def steps(n: int) -> list[str]:
    return [f"t{i:02d}" for i in range(1, n + 1)]


def time_settings(n_steps: int, periods: list[str], *, solve: str = "s1", extra_solve: list | None = None,
                  rolling: tuple[float, float] | None = None) -> tuple:
    """Entities and values of the common fixture's time structure: timeline ``tl`` of ``n_steps`` one-hour steps,
    timeset ``ts`` over all of them, model ``flexTool`` with the single solve ``solve`` over ``periods``
    (``years_represented`` 1 each); a rolling solve with ``rolling = (horizon, jump)`` hours."""
    ents = [["model", "flexTool", None], ["solve", solve, None], ["timeline", "tl", None], ["timeset", "ts", None]]
    vals = [
        ["timeline", "tl", "timestep_duration", str_map({t: 1.0 for t in steps(n_steps)})],
        ["timeset", "ts", "timeline", text("tl")],
        ["timeset", "ts", "timeset_duration", str_map({"t01": float(n_steps)})],
        ["model", "flexTool", "solves", array([solve], "sequence_index")],
        ["solve", solve, "solve_mode", text("rolling_window" if rolling else "single_solve")],
        ["solve", solve, "period_timeset", str_map({d: "ts" for d in periods}, "period")],
        ["solve", solve, "realized_periods", array(periods, "period")],
        ["solve", solve, "years_represented", str_map({d: 1.0 for d in periods}, "period")],
        ["solve", solve, "solver", text("highs")],
        ["solve", solve, "solver_mip_gap", scalar(1e-9)],
    ]
    if rolling:
        vals += [["solve", solve, "rolling_solve_horizon", scalar(rolling[0])],
                 ["solve", solve, "rolling_solve_jump", scalar(rolling[1])]]
    return ents, vals + list(extra_solve or [])


def balance_node(name: str, inflow, *, penalty_up: float = 10000.0, penalty_down: float = 10000.0) -> tuple:
    """A balance node with ``inflow`` (a number, or ``{step: MW}``)."""
    val = scalar(inflow) if isinstance(inflow, (int, float)) else str_map(inflow, "time")
    return [["node", name, None]], [
        ["node", name, "node_type", text("balance")], ["node", name, "inflow", val],
        ["node", name, "penalty_up", scalar(penalty_up)], ["node", name, "penalty_down", scalar(penalty_down)]]


def commodity_node(commodity: str, node: str, price: float, co2: float | None = None) -> tuple:
    ents = [["commodity", commodity, None], ["node", node, None], ["commodity__node", [commodity, node], None]]
    vals = [["node", node, "node_type", text("commodity")], ["commodity", commodity, "price", scalar(price)]]
    if co2 is not None:
        vals.append(["commodity", commodity, "co2_content", scalar(co2)])
    return ents, vals


def unit(name: str, inputs: list[str], outputs: list[str], params: dict) -> tuple:
    ents = [["unit", name, None]]
    ents += [["unit__inputNode", [name, n], None] for n in inputs]
    ents += [["unit__outputNode", [name, n], None] for n in outputs]
    vals = [["unit", name, k, v] for k, v in params.items()]
    return ents, vals


def merge(*parts: tuple) -> tuple[list, list]:
    ents: list = []
    vals: list = []
    for e, v in parts:
        ents += e
        vals += v
    return ents, vals


def make_oracle_db(tmp_path_factory: pytest.TempPathFactory, name: str, entities: list,
                   values: list) -> tuple[str, str]:
    """A database with the schema of ``tests.json`` and only *entities* / *values* (alternative ``base``, every
    entity active); returns ``(url, scenario)``."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    schema = json.loads(SCHEMA_JSON.read_text())
    data = {k: schema[k] for k in SCHEMA_KEYS if k in schema}
    data["alternatives"] = [["base", ""]]
    data["entities"] = list(entities)
    data["entity_alternatives"] = [[e[0], [e[1]] if isinstance(e[1], str) else e[1], "base", True]
                                   for e in entities if e[0] in ("node", "unit", "connection")]
    data["parameter_values"] = [[c, e, p, v, "base"] for c, e, p, v in values]
    data["scenarios"] = [["oracle", False, ""]]
    data["scenario_alternatives"] = [["oracle", "base", None]]
    work = tmp_path_factory.mktemp(f"{name}_db")
    path = work / f"{name}.json"
    path.write_text(json.dumps(data))
    url = json_to_db(path, work / f"{name}.sqlite")
    migrate_database(url)
    return url, "oracle"
