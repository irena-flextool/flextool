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


def two_d(entries: dict[str, list[str]], inner_index: str | None = None, **kw) -> list:
    """A 2-D period map ``{outer key: [period, ...]}`` (inner value "yes"; *kw* for the outer map, *inner_index*
    the inner maps' index name)."""
    ik = {} if inner_index is None else {"index_name": inner_index}
    return pack(pmap([(outer, nested([(p, "yes") for p in inner], **ik)) for outer, inner in entries.items()],
                     **kw), "map")


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


def model_b(mode: str, *, fan_periods: tuple[str, ...] = ("p1", "p2"), look_ahead: bool = False,
            rows: tuple[str, str] = ("p1", "p2"), solve_rows: list | None = None,
            real_branch: "dict[str, float] | None" = None) -> dict:
    """Model B: dispatch only, forecasts ``high`` (weight 1) / ``low`` (weight 3) of the wind profile declared at
    the first step of each period in *fan_periods* (p1/t01, p2/t03).  The forecast authored at t01 covers all four
    steps, with the t03 forecast's values at t03 / t04 (high 0.4, low 0.1), so whichever forecast a p2 step reads,
    it reads the same values.  *mode*: ``split`` (one solve ``st`` whose ``realized_periods`` is the 2-D map
    ``{rows[0]: [p1], rows[1]: [p2]}``) or ``hand`` (``st_<row>`` written out, each with ``years_represented`` of
    both periods).  With *look_ahead* the first solve also covers p2 (2-D ``invest_periods`` {p1: [p1, p2],
    p2: [p2]}, ``realized_invest_periods`` as ``realized_periods``), and a hand-written solve carries only the
    ``stochastic_branches`` rows of the period it realizes; without, each carries the solve's whole
    ``stochastic_branches``.  *solve_rows* (``[parameter, packed value]``) go on every solve.  *real_branch*
    ({period: value}) adds a ``realized: yes`` branch ``real`` (weight 0) at each fan, whose forecast made at the
    period's first step is that constant value from there on."""
    first, second = rows
    row_of = {"p1": first, "p2": second}
    solves = ["st"] if mode == "split" else [f"st_{row_of[p]}" for p in PERIODS]
    spec = _base_spec(solves, [])
    spec["entities"] += [["unit", "wind", None], ["group", "sg", None], ["profile", "wp", None],
                         ["group__unit", ["sg", "wind"], None], ["unit__outputNode", ["wind", "elec"], None],
                         ["unit__node__profile", ["wind", "elec", "wp"], None]]
    spec["entity_alternatives"].append(["unit", ["wind"], "init", True])
    first_step = {"p1": "t01", "p2": "t03"}

    def fans(periods) -> dict:
        return pmap([(p, nested([("high", nested([(first_step[p], nested([("no", 1.0)]))])),
                                 ("low", nested([(first_step[p], nested([("no", 3.0)]))]))]
                                + ([("real", nested([(first_step[p], nested([("yes", 0.0)]))]))] if real_branch
                                   else [])))
                     for p in periods])

    branch_series = [("high", nested([("t01", nested([(t, 0.4) for t in STEPS])),
                                      ("t03", nested([(t, 0.4) for t in STEPS[2:]]))])),
                     ("low", nested([("t01", nested([(t, v) for t, v in zip(STEPS, (0.2, 0.2, 0.1, 0.1))])),
                                     ("t03", nested([(t, 0.1) for t in STEPS[2:]]))]))]
    if real_branch:
        start_index = {"p1": 0, "p2": 2}
        branch_series.append(("real", nested([(first_step[p], nested([(t, v) for t in STEPS[start_index[p]:]]))
                                              for p, v in real_branch.items()])))
    profile = pmap(branch_series)
    pv = spec["parameter_values"]
    pv += [
        ["unit", "gen", "existing", pack(200.0, "float"), "init"],
        ["unit", "wind", "efficiency", pack(1.0, "float"), "init"],
        ["unit", "wind", "existing", pack(100.0, "float"), "init"],
        ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
        ["profile", "wp", "profile", pack(profile, "map"), "init"],
        ["group", "sg", "include_stochastics", pack("yes", "str"), "init"],
    ]
    realized = {row_of[p]: [p] for p in PERIODS}
    invest = {first: list(PERIODS), second: ["p2"]}

    def solve_values(solve: str, periods: list[str], fan, values: dict) -> list:
        out = [["solve", solve, "solve_mode", pack("single_solve", "str"), "init"],
               ["solve", solve, "period_timeset", pack(pmap([(p, PERIODS[p]) for p in periods]), "map"), "init"],
               ["solve", solve, "years_represented",
                pack(pmap([(p, 1.0) for p in PERIODS], index_name="period"), "map"), "init"]]
        if fan["data"]:
            out.append(["solve", solve, "stochastic_branches", pack(fan, "map"), "init"])
        out += [["solve", solve, par, value, "init"] for par, value in values.items()]
        return out + [["solve", solve, par, value, "init"] for par, value in (solve_rows or [])]

    if mode == "split":
        values = {"realized_periods": two_d(realized)}
        if look_ahead:
            values.update(invest_periods=two_d(invest), realized_invest_periods=two_d(realized))
        pv += solve_values("st", list(PERIODS), fans(fan_periods), values)
    else:
        for p in PERIODS:
            row = row_of[p]
            values = {"realized_periods": array([p])}
            if look_ahead:
                values.update(invest_periods=array(invest[row]), realized_invest_periods=array([p]))
            own = invest[row] if look_ahead else [p]
            fan = fans([q for q in fan_periods if q == p]) if look_ahead else fans(fan_periods)
            pv += solve_values(f"st_{row}", own, fan, values)
    return spec


def rename_solve(data: dict, old: str, new: str) -> dict:
    """*data* (a packed fixture JSON) with the solve *old* renamed to *new* (its entity, its values and every
    ``model.solves`` / ``contains_solves`` array that names it)."""
    for entity in data["entities"]:
        if entity[0] == "solve" and entity[1] == old:
            entity[1] = new
    for row in data["parameter_values"]:
        if row[0] == "solve" and row[1] == old:
            row[1] = new
        if (row[0], row[2]) in (("model", "solves"), ("solve", "contains_solves")):
            value = json.loads(base64.b64decode(row[3][0]))
            if isinstance(value, dict):
                value["data"] = [new if s == old else s for s in value["data"]]
            elif value == old:
                value = new
            row[3] = [base64.b64encode(json.dumps(value).encode()).decode("ascii"), row[3][1]]
    return data


def model_a_ladder(mode: str, ladder: dict[str, dict[str, list[str]]], *, lists: dict[str, list[str]] | None = None,
                   unit_rows: list | None = None, years_index: str | None = None,
                   outer_index: str | None = None, contains: list[str] | None = None) -> dict:
    """Model A with any 2-D period maps.  *ladder*: ``{parameter: {outer key: [period, ...]}}`` of the 2-D
    parameters; *lists*: the 1-D period lists of the solve (``{parameter: [period, ...]}``).  ``split`` writes one
    solve ``inv`` with these values (``period_timeset`` over both periods); ``hand`` writes the solves the docs say
    the maps stand for, ``inv_<outer>`` in the order of the outer keys: each 2-D parameter its inner list, each 1-D
    list its periods that the solve has, ``period_timeset`` and ``years_represented`` (5 years per period) of its
    periods.  *unit_rows* (``[parameter, packed value]``) go on the unit ``gen``; *outer_index* names the outer
    level of the 2-D maps (``None`` = Spine's default ``x``).  *contains*: the ``contains_solves`` of ``inv`` (split)
    or of every ``inv_<outer>`` (hand); each named solve is a dispatch solve over both periods (``realized_periods``
    [p1, p2], 5 years per period), as ``invest_5weeks`` contains its storage solve in the shipped example."""
    lists = dict(lists or {})
    outers = list(dict.fromkeys(o for entries in ladder.values() for o in entries))
    solves = ["inv"] if mode == "split" else [f"inv_{o}" for o in outers]
    spec = model_a("single")
    spec["entities"] = [e for e in spec["entities"] if e[0] != "solve"] + [["solve", s, None]
                                                                           for s in solves + list(contains or [])]
    pv = [r for r in spec["parameter_values"] if r[0] != "solve" and (r[0], r[2]) != ("model", "solves")]
    pv.append(["model", "flextool", "solves", array(solves), "init"])
    pv += [["unit", "gen", par, value, "init"] for par, value in (unit_rows or [])]
    yk = {} if years_index is None else {"index_name": years_index}

    def rows(solve: str, periods: list[str], values: dict) -> list:
        out = [["solve", solve, "solve_mode", pack("single_solve", "str"), "init"],
               ["solve", solve, "period_timeset", pack(pmap([(p, PERIODS[p]) for p in periods]), "map"), "init"],
               ["solve", solve, "years_represented", pack(pmap([(p, 5.0) for p in periods], **yk), "map"),
                "init"]]
        if contains:
            out.append(["solve", solve, "contains_solves", array(contains), "init"])
        return out + [["solve", solve, par, value, "init"] for par, value in values.items()]

    for solve in contains or []:
        pv += [["solve", solve, "solve_mode", pack("single_solve", "str"), "init"],
               ["solve", solve, "period_timeset", pack(pmap(list(PERIODS.items())), "map"), "init"],
               ["solve", solve, "realized_periods", array(list(PERIODS)), "init"],
               ["solve", solve, "years_represented", pack(pmap([(p, 5.0) for p in PERIODS], **yk), "map"),
                "init"]]
    if mode == "split":
        ok = {} if outer_index is None else {"index_name": outer_index, "inner_index": "period"}
        values = {par: two_d(entries, **ok) for par, entries in ladder.items()}
        values.update({par: array(periods) for par, periods in lists.items()})
        pv += rows("inv", list(PERIODS), values)
    else:
        for outer in outers:
            own = list(dict.fromkeys(p for entries in ladder.values() for p in entries.get(outer, [])))
            values = {par: array(entries[outer]) for par, entries in ladder.items() if entries.get(outer)}
            values.update({par: array([p for p in periods if p in own]) for par, periods in lists.items()
                           if any(p in own for p in periods)})
            pv += rows(f"inv_{outer}", [p for p in PERIODS if p in own], values)
    spec["parameter_values"] = pv
    return spec
