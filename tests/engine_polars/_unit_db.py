"""Shared helpers for the unit-floor / input-share / reserve tests.

Every database is ``tests/fixtures/tests.json`` plus one override
alternative appended to a base alternative chain, built under the pytest
tmp dir and migrated (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
BASE_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json"

CHP = "coal_chp"            # indirect: coal_market -> west + heat, 1000 MW, eff 0.9
PLANT = "coal_plant"        # direct:   coal_market -> west, 500 MW, eff 0.4
COAL = "coal_market"
GAS = "gas_market"
CHP_CHAIN = ["init", "west", "coal_chp", "heat"]
PLANT_CHAIN = ["init", "west", "coal"]
GAS_ENTITIES = [
    ["commodity", "gas", None],
    ["node", GAS, None],
    ["commodity__node", ["gas", GAS], None],
    ["unit__inputNode", [CHP, GAS], None],
]


def b64(x, kind: str) -> list:
    """Spine import-format packed value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


TIMESTEPS = [f"t{i:04d}" for i in range(1, 73)]


def time_map(values: dict[str, float], default: float) -> list:
    """A ``time`` map over the fixture's timesteps (``default`` where
    absent)."""
    return b64({"index_type": "str", "index_name": "time",
                "data": [[t, values.get(t, default)] for t in TIMESTEPS]},
               "map")


def make_db(tmp_path_factory: pytest.TempPathFactory, *, alt: str,
            values: list, base_chain: list[str],
            entities: list | None = None) -> tuple[str, str]:
    """``tests.json`` + override alternative ``alt`` appended to
    ``base_chain``; returns ``(url, scenario)``.  New ``node`` entities
    are activated in ``alt``."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([alt, ""])
    for ent in entities or []:
        data["entities"].append(ent)
        if ent[0] == "node":
            data["entity_alternatives"].append(["node", [ent[1]], alt, True])
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, val, alt])
    scenario = f"{alt}_test"
    chain = list(base_chain) + [alt]
    data["scenarios"].append([scenario, False, ""])
    for a, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([scenario, a, before])
    work = tmp_path_factory.mktemp(f"{alt}_db")
    json_path = work / f"{alt}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{alt}.sqlite")
    migrate_database(url)
    return url, scenario


def run(tmp_path_factory, url: str, scenario: str) -> dict:
    from flextool.engine_polars import run_chain_from_db

    steps = run_chain_from_db(
        url, scenario, work_folder=tmp_path_factory.mktemp(f"{scenario}_w"),
        keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


def solve(tmp_path_factory, *, alt: str, values: list, base_chain: list[str],
          entities: list | None = None):
    url, scen = make_db(tmp_path_factory, alt=alt, values=values,
                        base_chain=base_chain, entities=entities)
    return list(run(tmp_path_factory, url, scen).values())[-1]


def utf8(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col(c).cast(pl.Utf8) for c in df.columns
                           if c not in ("value",))


def unitsize(step, unit: str) -> float:
    return float(utf8(step.flex_data.p_unitsize.frame)
                 .filter(pl.col("p") == unit)["value"][0])


def flow(step, unit: str, source: str, sink: str) -> pl.Series:
    """Flow in MW (v_flow × unitsize), ordered by (d, t)."""
    rows = (utf8(step.solution.value("v_flow"))
            .filter((pl.col("p") == unit) & (pl.col("source") == source)
                    & (pl.col("sink") == sink))
            .sort("d", "t"))
    assert rows.height > 0, f"no v_flow rows for {unit} {source}->{sink}"
    return rows["value"] * unitsize(step, unit)


def online(step, unit: str, kind: str = "linear") -> pl.Series:
    """Online capacity in MW, ordered by (d, t)."""
    rows = (utf8(step.solution.value(f"v_online_{kind}"))
            .filter(pl.col("p") == unit).sort("d", "t"))
    assert rows.height > 0, f"no v_online_{kind} rows for {unit}"
    return rows["value"] * unitsize(step, unit)


def reserve(step, unit: str, r: str, node: str) -> pl.Series:
    """Reserve in MW at ``node`` (v_reserve × unitsize), ordered by (d, t)."""
    rows = (utf8(step.solution.value("v_reserve"))
            .filter((pl.col("p") == unit) & (pl.col("r") == r)
                    & (pl.col("n") == node))
            .sort("d", "t"))
    assert rows.height > 0, f"no v_reserve rows for {unit} {r} {node}"
    return rows["value"] * unitsize(step, unit)


def demand(node: str, mw: float) -> list:
    return ["node", node, "inflow", b64(-mw, "float")]


def no_ratio_constraint() -> list:
    return [["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")]]


def gas(price: float) -> list:
    return [["node", GAS, "node_type", b64("commodity", "str")],
            ["commodity", "gas", "price", b64(price, "float")]]


def uc(unit: str, min_load: float, method: str = "linear") -> list:
    return [["unit", unit, "startup_method", b64(method, "str")],
            ["unit", unit, "min_load", b64(min_load, "float")]]


def build(fd):
    """Build (do not solve) the LP for ``fd``."""
    from polar_high import Problem

    from flextool.engine_polars import build_flextool

    pb = Problem()
    build_flextool(pb, fd)
    return pb


def cstr_over(pb, name: str) -> pl.DataFrame | None:
    """The ``over`` frame (Utf8) of the constraint family ``name``."""
    recs = [r for r in pb.cstrs_named(name) if r.name == name]
    if not recs:
        return None
    return utf8(recs[0].over)
