"""Small models built in code for the solve-copy tests (a solve split by a 2-D period map against the same solves
written by hand), and the helpers that run them and compare their LPs.

Every model is a ``packed`` fixture spec (``schema_sections``) built from scratch, written to JSON and imported into
its own database (``json_to_db`` + ``migrate_database``): the split and the hand-written variants never share a
database, so the names of the split solves are never also ``solve`` entities.

Model A: one balance node ``elec`` with 100 MW constant demand (penalty 10 000 EUR/MWh), one investable unit ``gen``
(invest_cost 1 EUR/kW, lifetime 20 a, discount rate 0.05, other_operational_cost 10 EUR/MWh), two periods ``p1``
(steps t01-t02) and ``p2`` (t03-t04) on 1 h steps.

Model B: model A's node with a 200 MW unit ``gen`` (10 EUR/MWh) and 100 MW of wind whose profile has the
forecasts ``high`` (weight 1) and ``low`` (weight 3), declared at p1/t01 and p2/t03; dispatch only.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEPS = ("t01", "t02", "t03", "t04")
PERIODS = {"p1": "ts_a", "p2": "ts_b"}
#: The 2-D period parameters a split solve of model A / B carries.
SPLIT_PARAMS = ("invest_periods", "realized_invest_periods", "realized_periods")


def lp_snapshot():
    """``tools/lp_snapshot.py`` as a module (the MPS capture and the canonical LP diff)."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def pack(value, type_name: str) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode("ascii"), type_name]


def pmap(pairs, **kw) -> dict:
    """A 1-D str-indexed map value (no ``index_name`` = Spine's default ``x``)."""
    out = {"index_type": "str", "data": [[k, v] for k, v in pairs]}
    out.update(kw)
    return out


def nested(pairs, **kw) -> dict:
    """An inner map of a nested map."""
    return {"type": "map", **pmap(pairs, **kw)}


def array(values) -> list:
    return pack({"value_type": "str", "data": list(values)}, "array")


def two_d(entries: dict[str, list[str]], **kw) -> list:
    """A 2-D period map ``{outer key: [period, ...]}`` (inner value "yes")."""
    return pack(pmap([(outer, nested([(p, "yes") for p in inner])) for outer, inner in entries.items()], **kw),
                "map")


def _base_spec(solves: list[str], alternatives: list[str]) -> dict:
    from flextool.update_flextool.fixture_schema import schema_sections

    spec = schema_sections("packed")
    spec["alternatives"] = [["init", ""]] + [[a, ""] for a in alternatives]
    spec["scenarios"] = [["base", False, ""]]
    spec["scenario_alternatives"] = [["base", "init", alternatives[0] if alternatives else None]]
    for a, b in zip(alternatives, alternatives[1:] + [None]):
        spec["scenario_alternatives"].append(["base", a, b])
    spec["entities"] = [["model", "flextool", None], ["node", "elec", None], ["timeline", "tl", None],
                        ["timeset", "ts_a", None], ["timeset", "ts_b", None], ["unit", "gen", None],
                        ["unit__outputNode", ["gen", "elec"], None]] + [["solve", s, None] for s in solves]
    spec["entity_alternatives"] = [["node", ["elec"], "init", True], ["unit", ["gen"], "init", True]]
    spec["parameter_values"] = [
        ["model", "flextool", "solves", array(solves), "init"],
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in STEPS]), "map"), "init"],
        ["timeset", "ts_a", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts_a", "timeset_duration", pack(pmap([("t01", 2.0)]), "map"), "init"],
        ["timeset", "ts_b", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts_b", "timeset_duration", pack(pmap([("t03", 2.0)]), "map"), "init"],
        ["node", "elec", "node_type", pack("balance", "str"), "init"],
        ["node", "elec", "penalty_up", pack(10000.0, "float"), "init"],
        ["node", "elec", "penalty_down", pack(10000.0, "float"), "init"],
        ["node", "elec", "inflow", pack(-100.0, "float"), "init"],
        ["unit", "gen", "efficiency", pack(1.0, "float"), "init"],
        ["unit__outputNode", ["gen", "elec"], "other_operational_cost", pack(10.0, "float"), "init"],
    ]
    return spec


def model_a(mode: str, *, years: tuple[float, float] = (5.0, 5.0), years_index: str | None = None,
            years_own_periods: bool = False, solve_rows: list | None = None, model_rows: list | None = None,
            split_params: tuple[str, ...] = SPLIT_PARAMS) -> dict:
    """Model A.  *mode*: ``single`` (one solve ``inv`` over both periods), ``split`` (one solve ``inv`` whose
    *split_params* are 2-D maps ``{p1: [p1], p2: [p2]}``; the other period parameters 1-D over both periods) or
    ``hand`` (the solves ``inv_p1`` and ``inv_p2`` written out).  ``years_represented`` is ``{p1, p2}`` (*years*)
    on every solve, index name *years_index* (``None`` = Spine's default ``x``); with *years_own_periods* a
    hand-written solve carries only its own period's entry.  *solve_rows* (``[parameter, packed value]``) go on
    every solve, *model_rows* (``[class, entity, parameter, packed value]``) on the model."""
    solves = ["inv"] if mode in ("single", "split") else [f"inv_{p}" for p in PERIODS]
    spec = _base_spec(solves, [])
    pv = spec["parameter_values"]
    pv += [
        ["unit", "gen", "existing", pack(0.0, "float"), "init"],
        ["unit", "gen", "virtual_unitsize", pack(1.0, "float"), "init"],
        ["unit", "gen", "invest_method", pack("invest_total", "str"), "init"],
        ["unit", "gen", "invest_cost", pack(1.0, "float"), "init"],
        ["unit", "gen", "discount_rate", pack(0.05, "float"), "init"],
        ["unit", "gen", "lifetime", pack(20.0, "float"), "init"],
        ["unit", "gen", "invest_max_total", pack(1000.0, "float"), "init"],
    ]
    pv += [[*row[:3], row[3], "init"] for row in (model_rows or [])]
    yk = {} if years_index is None else {"index_name": years_index}
    year_of = dict(zip(PERIODS, years))

    def rows(solve: str, periods: list[str]) -> list:
        out = [["solve", solve, "solve_mode", pack("single_solve", "str"), "init"],
               ["solve", solve, "period_timeset", pack(pmap([(p, PERIODS[p]) for p in periods]), "map"), "init"]]
        for par in SPLIT_PARAMS:
            if mode == "split" and par in split_params:
                out.append(["solve", solve, par, two_d({p: [p] for p in PERIODS}), "init"])
            else:
                out.append(["solve", solve, par, array(periods), "init"])
        yr_periods = periods if (mode == "hand" and years_own_periods) else list(PERIODS)
        out.append(["solve", solve, "years_represented",
                    pack(pmap([(p, year_of[p]) for p in yr_periods], **yk), "map"), "init"])
        out += [["solve", solve, par, value, "init"] for par, value in (solve_rows or [])]
        return out

    if mode in ("single", "split"):
        pv += rows("inv", list(PERIODS))
    else:
        for p in PERIODS:
            pv += rows(f"inv_{p}", [p])
    return spec


def write_spec(spec: dict, folder: Path, stem: str = "f") -> Path:
    from flextool.update_flextool.fixture_schema import dump_fixture_json

    path = folder / f"{stem}.json"
    dump_fixture_json(spec, path)
    return path


def build_db(json_path: Path, folder: Path) -> str:
    """The migrated database of *json_path* in *folder*; returns its URL."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    url = json_to_db(json_path, folder / f"{json_path.stem}.sqlite")
    migrate_database(url)
    return url


def run_capturing(url: str, scenario: str, folder: Path):
    """Run *scenario* cold and capture every built Problem: returns ``(built, steps)`` with ``built`` the
    ``(solve, MPS path, objective offset)`` of each build in build order and ``steps`` the run's steps."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core
    import pytest

    lps = lp_snapshot()
    built: list[tuple[str, Path, float]] = []
    original = core.build_flextool

    def capturing(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        solve = lps._solve_name_from_stack()
        path = folder / f"{len(built):03d}_{lps._safe(solve)}.mps"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the objective offset is kept separately
            m.write_mps(path, emit_names=True)
        built.append((solve, path, float(m._obj_offset)))
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "build_flextool", capturing)
        steps = run_chain_from_db(url, scenario, work_folder=folder / "w", warm=False, keep_solutions=True)
    return built, steps


def run_spec(spec: dict, folder: Path, scenario: str = "base"):
    """:func:`run_capturing` of *spec* in its own database under *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    return run_capturing(build_db(write_spec(spec, folder), folder), scenario, folder)


def lp_differences(case: list, twin: list, rtol: float = 1e-12) -> dict[str, list[str]]:
    """``{solve: canonical differences}`` of two captured runs, compared build by build (same solve order)."""
    lps = lp_snapshot()
    assert [s for s, _, _ in case] == [s for s, _, _ in twin], ([s for s, _, _ in case], [s for s, _, _ in twin])
    out = {}
    for (solve, pa, off_a), (_, pb, off_b) in zip(case, twin):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), rtol, off_a, off_b)
        if msgs:
            out[solve] = msgs
    return out


#: FlexTool scales the objective by this factor in the LP it builds (``p_scale_the_objective``, not a database
#: parameter), so an objective coefficient in the MPS is the EUR coefficient times this.
OBJECTIVE_SCALE = 1e-6


def objective_coefficient(mps: Path, column: str) -> float:
    """The objective coefficient of *column* in *mps*, in EUR (the LP's objective scaling taken out)."""
    return lp_snapshot().parse_mps(mps).obj[column] / OBJECTIVE_SCALE
