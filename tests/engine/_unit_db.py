"""Shared helpers for the unit-floor / input-share / reserve tests.

Every database is ``tests/fixtures/tests.json`` plus one override
alternative appended to a base alternative chain, built under the pytest
tmp dir and migrated (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import base64
import contextlib
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


def add_override_alternative(data: dict, alt: str, values: list, base_chain: list[str],
                             entities: list | None = None) -> str:
    """Append the override alternative ``alt`` to the Spine import data
    ``data`` (changed in place; no files): the alternative, ``entities``
    (new ``node`` / ``unit`` / ``connection`` entities active in ``alt``),
    the parameter ``values`` ``[class, entity, parameter, packed value]`` in
    ``alt``, and the scenario ``<alt>_test`` = ``base_chain`` + ``alt``.
    Returns the scenario name."""
    data["alternatives"].append([alt, ""])
    for ent in entities or []:
        data["entities"].append(ent)
        if ent[0] in ("node", "unit", "connection"):
            data["entity_alternatives"].append([ent[0], [ent[1]], alt, True])
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, val, alt])
    scenario = f"{alt}_test"
    chain = list(base_chain) + [alt]
    data["scenarios"].append([scenario, False, ""])
    for a, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([scenario, a, before])
    return scenario


def make_db(tmp_path_factory: pytest.TempPathFactory, *, alt: str,
            values: list, base_chain: list[str],
            entities: list | None = None,
            fixture_json: Path = BASE_FIXTURE_JSON) -> tuple[str, str]:
    """``fixture_json`` (default ``tests.json``) + override alternative
    ``alt`` appended to ``base_chain`` (:func:`add_override_alternative`);
    returns ``(url, scenario)``."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads(Path(fixture_json).read_text())
    scenario = add_override_alternative(data, alt, values, base_chain, entities)
    work = tmp_path_factory.mktemp(f"{alt}_db")
    json_path = work / f"{alt}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{alt}.sqlite")
    migrate_database(url)
    return url, scenario


def run(tmp_path_factory, url: str, scenario: str) -> dict:
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(
        url, scenario, work_folder=tmp_path_factory.mktemp(f"{scenario}_w"),
        keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


class RunCache:
    """One cascade per distinct model within a test module.

    Tests of one module that solve the same model share its run: the key is the fixture JSON, the base chain, the
    override values and entities and the run options, not the override alternative's name (it only names the
    scenario).  A run made while a test patches the engine (``monkeypatch`` on the build, the scaling, ...) must not
    go through the cache.  Callers only read the cached steps and work folder (copy the folder before writing into
    it).  Hold one per module and clear it at module teardown, so a module's runs are freed when it ends and the
    cache never crosses modules (``--dist loadfile`` keeps a module on one worker)::

        _RUNS = RunCache()

        @pytest.fixture(scope="module", autouse=True)
        def _free_runs():
            yield
            _RUNS.clear()
    """

    def __init__(self) -> None:
        self._runs: dict[str, tuple[dict, Path, str]] = {}

    def get(self, tmp_path_factory: pytest.TempPathFactory, *, alt: str, values: list, base_chain: list[str],
            entities: list | None = None, fixture_json: Path = BASE_FIXTURE_JSON,
            **run_kwargs) -> tuple[dict, Path, str]:
        """``(steps, work_folder, scenario)`` of the model, solved on first use (``keep_solutions=True``)."""
        from flextool.engine import run_chain_from_db

        key = json.dumps([str(fixture_json), list(base_chain), values, entities, sorted(run_kwargs.items())],
                         default=str)
        if key not in self._runs:
            url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=base_chain,
                                entities=entities, fixture_json=fixture_json)
            wf = tmp_path_factory.mktemp(f"{alt}_w")
            steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True, **run_kwargs)
            self._runs[key] = (steps, wf, scen)
        return self._runs[key]

    def clear(self) -> None:
        self._runs.clear()


#: Name suffix of the realization solve the engine adds after a stochastic solve with a ``realized: yes``
#: designation (``solve._orchestration._spawn_realized_dispatch_siblings``); its rolls are ``<name>_roll_<k>``.
REALIZATION_SUFFIX = "__realized_dispatch"


def is_realization(name: str) -> bool:
    """True for a step of an engine-added realization solve (or one of its rolls)."""
    return REALIZATION_SUFFIX in name


def without_realization(steps: dict) -> dict:
    """*steps* without the engine-added realization solves: the stochastic (or deterministic) solves only."""
    return {k: s for k, s in steps.items() if not is_realization(k)}


def last_stochastic(steps: dict):
    """The last step that is not an engine-added realization solve (the stochastic solve of a single run)."""
    return list(without_realization(steps).values())[-1]


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


@contextlib.contextmanager
def axis_vocabulary(fd):
    """The run's axis vocabulary live for the duration (``fd._axis_enums``; None in Utf8 mode).

    In axis-Enum mode (``FLEXTOOL_AXIS_ENUMS=1``) the model builders and the region split read the axis vocabulary
    from the live context (``set_global_axis_enums``), which ``run_chain_from_db`` resets when the run ends; a test
    that rebuilds or splits a run's FlexData afterwards sets it as the run does.  A FlexData without the attribute
    (a Benders region's, from ``_region_filter.split``) is built under its parent's vocabulary, as the Benders
    solve builds it: the live one is left as it is."""
    from flextool.engine.input._axis_enums import reset_global_axis_enums, set_global_axis_enums

    if not hasattr(fd, "_axis_enums"):
        yield
        return
    token = set_global_axis_enums(fd._axis_enums)
    try:
        yield
    finally:
        reset_global_axis_enums(token)


def build(fd):
    """Build (do not solve) the LP for ``fd``, with the run's axis vocabulary (:func:`axis_vocabulary`)."""
    from polar_high import Problem

    from flextool.engine import build_flextool

    pb = Problem()
    with axis_vocabulary(fd):
        build_flextool(pb, fd)
    return pb


def cstr_over(pb, name: str) -> pl.DataFrame | None:
    """The ``over`` frame (Utf8) of the constraint family ``name``."""
    recs = [r for r in pb.cstrs_named(name) if r.name == name]
    if not recs:
        return None
    return utf8(recs[0].over)


def output_frames(work_folder: Path, solve_name: str, *, prefixes=("v_", "vq_")) -> dict[str, pl.DataFrame]:
    """The ``output_raw/<name>__<solve_name>.parquet`` frames whose name starts with one of
    ``prefixes`` (default: the primal variable outputs), keyed by ``<name>``.  Duals
    (``v_dual_*``) are left out: they are not unique at a degenerate optimum."""
    out = {}
    for f in sorted((Path(work_folder) / "output_raw").glob(f"*__{solve_name}.parquet")):
        name = f.name[: -len(f"__{solve_name}.parquet")]
        if name.startswith(tuple(prefixes)) and not name.startswith("v_dual_"):
            out[name] = pl.read_parquet(f)
    return out


def assert_outputs_equal(wf_a: Path, wf_b: Path, solve_name: str, *,
                         rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> None:
    """The written primal outputs of ``solve_name`` agree between two work folders: same files,
    same labels, values within ``rel_tol`` / ``abs_tol``."""
    from polars.testing import assert_frame_equal

    a, b = output_frames(wf_a, solve_name), output_frames(wf_b, solve_name)
    assert a, f"{solve_name}: no primal outputs written in {wf_a}"
    assert sorted(a) == sorted(b), f"{solve_name}: output files differ: {sorted(a)} vs {sorted(b)}"
    for name in a:
        fa, fb = a[name], b[name]
        keys = [c for c in fa.columns if not fa.schema[c].is_numeric()]
        try:
            assert_frame_equal(fa.sort(keys), fb.sort(keys), check_exact=False, rel_tol=rel_tol, abs_tol=abs_tol,
                               check_column_order=False)
        except AssertionError as exc:
            raise AssertionError(f"{solve_name}: output {name} differs: {exc}") from None


#: The ``fullYear_roll`` scenario of tests.json: dispatch_fullYear_roll, 8-h horizon / 4-h jump over the 72-h
#: fullYear timeset of four periods (72 rolls).
FULLYEAR_ROLL_CHAIN = ["init", "west", "battery", "wind", "fullYear", "dispatch_fullYear_roll"]


def fullyear_roll_first_hours(hours: float) -> list:
    """Override values that cut ``fullYear_roll`` to the first ``hours`` hours of p2020 (``hours / 4`` same-level
    rolls of 8 h / 4 h, the same solve, timeline and units)."""
    return [["solve", "dispatch_fullYear_roll", "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", "fullYear"]]}, "map")],
            ["solve", "dispatch_fullYear_roll", "realized_periods",
             b64({"value_type": "str", "data": ["p2020"]}, "array")],
            ["solve", "dispatch_fullYear_roll", "rolling_duration", b64(float(hours), "float")]]
