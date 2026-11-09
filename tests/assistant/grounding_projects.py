"""The synthetic project of the assistant's grounding questions (P3 §6.2).  Test support, not collected.

The input data is ``tests/fixtures/tests.json`` (through ``json_to_db``, migrated to this FlexTool) with the
override alternatives of :data:`GROUNDING_OVERRIDES`, each appended by ``add_override_alternative`` (the scenario
is ``<alternative>_test``):

* ``h2_lookup`` -- a storage node ``h2_tank`` whose description gives its own units (never solved);
* ``demand_update`` -- the failed-run case: the first build gives ``node west inflow`` the fixture's ``west``
  series as a ``time`` map and solves; the source is then rewritten with :data:`DEMAND_UPDATE_REWRITE` (inflow
  keyed by period) and the second run of the same scenario stops in the input checks before any solve, so its
  result folder holds the first run's files and a failed run record;
* ``invest_2day`` -- the investment case: the fixture's ``coal_invest`` alternative with ``p2020`` as an investment
  period of the 48-hour solve.

The fixture's own scenarios ``coal`` and ``coal_co2_price`` are solved as they are.  The oracle of the answer key
reads exactly this data (and ``tests.json``); nothing here states a result.

:func:`build_grounding_project` makes a FlexTool project the way the FlexTool window does (its folders, the source
numbered 1 in ``settings.yaml``) in a folder whose path has a space and an ``ä``, runs the scenarios with
``flextool-run`` and the window's command line (parquet results in ``output_parquet/<scenario>_1``) in the order
of :data:`RUN_ORDER`, so no later run is older than the rewritten source, and writes the marker
:data:`MARKER_FILE` that the calibration runner (``tools/assistant_eval.py``) requires.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "tests.json"

#: The project folder's name (a space and an ``ä``: paths the tools must handle).
PROJECT_NAME = "Grounding project ä"
SOURCE_NAME = "grounding.sqlite"
#: The marker the runner requires (P3-D10): ``{format, set_version, fingerprint, source_sha256, solved, runs}``;
#: ``runs`` maps each ``output_parquet`` folder to the ``run_id`` of its final status.
MARKER_FILE = ".flextool_grounding_project.json"
MARKER_FORMAT = 2
#: The version of the question set this project is built for (``grounding_set.yaml`` ``set_version``).
SET_VERSION = 3
#: The ``--threads`` of every run (the window sends its "Threads per run").
RUN_THREADS = 1


@dataclass(frozen=True)
class Packed:
    """A parameter value in the Spine JSON form of its ``kind`` (``map``, ``array``, ``time_series``...)."""

    kind: str
    data: dict


@dataclass(frozen=True)
class FixtureSeries:
    """The value ``tests.json`` holds for ``(cls, entity, parameter)`` in ``alternative``, written again as a
    ``time`` map over the fixture's 72 timesteps (``tests/engine/_unit_db.py:time_map``)."""

    cls: str
    entity: str
    parameter: str
    alternative: str


Value = Union[float, str, Packed, FixtureSeries]


@dataclass(frozen=True)
class Override:
    """An override alternative ``alt`` after ``base_chain`` (scenario ``<alt>_test``): ``values`` are
    ``(class, entity, parameter, value)``, ``entities`` new entities ``(class, name, description)``."""

    alt: str
    base_chain: tuple[str, ...]
    values: tuple[tuple[str, str, str, Value], ...]
    entities: tuple[tuple[str, str, str | None], ...] = ()

    @property
    def scenario(self) -> str:
        return f"{self.alt}_test"


H2_LOOKUP = Override(
    alt="h2_lookup",
    base_chain=("init", "west", "coal"),
    entities=(("node", "h2_tank", "Hydrogen tank. Quantities in kg, flows in kg/h."),),
    values=(("node", "h2_tank", "node_type", "storage"),
            ("node", "h2_tank", "existing", 2000.0)),
)
DEMAND_UPDATE = Override(
    alt="demand_update",
    base_chain=("init", "west", "coal"),
    values=(("node", "west", "inflow", FixtureSeries("node", "west", "inflow", "west")),),
)
#: The second build of ``demand_update``: the inflow keyed by period, which the input checks refuse.
DEMAND_UPDATE_REWRITE = Override(
    alt="demand_update",
    base_chain=("init", "west", "coal"),
    values=(("node", "west", "inflow",
             Packed("map", {"type": "map", "index_type": "str", "index_name": "period",
                            "data": [["p2020", -500.0]]})),),
)
INVEST_2DAY = Override(
    alt="invest_2day",
    base_chain=("init", "west", "coal", "coal_invest"),
    values=(("solve", "y2020_2day_dispatch", "invest_periods",
             Packed("array", {"value_type": "str", "data": ["p2020"], "index_name": "period"})),),
)
#: The overrides of the first build, in the order they are appended.
GROUNDING_OVERRIDES: tuple[Override, ...] = (H2_LOOKUP, DEMAND_UPDATE, INVEST_2DAY)

#: The runs, in order: ``(scenario, build)``; build 2 is the source after :data:`DEMAND_UPDATE_REWRITE`.
RUN_ORDER: tuple[tuple[str, int], ...] = (
    ("demand_update_test", 1),
    ("demand_update_test", 2),
    ("coal", 2),
    ("coal_co2_price", 2),
    ("invest_2day_test", 2),
)
#: The run expected to fail, and a part of its recorded reason (``validate_node_inflow_not_period``).
FAILED_RUN = ("demand_update_test", 2)
FAILED_REASON = "keyed by period"


class GroundingBuildError(RuntimeError):
    """A step of the build did not do what the project needs."""


@dataclass(frozen=True)
class GroundingRun:
    scenario: str
    build: int
    folder: str                 # output_parquet/<folder>
    run_id: str | None
    state: str | None
    exit_code: int
    error: str | None


@dataclass(frozen=True)
class GroundingProject:
    path: Path
    source: Path
    source_number: int
    runs: tuple[GroundingRun, ...]
    fingerprint: str
    source_sha256: str


# --- the input data -------------------------------------------------------------------------------------------
def _b64(value: object, kind: str) -> list:
    from tests.engine._unit_db import b64

    return b64(value, kind)


def _fixture_value(data: dict, series: FixtureSeries) -> list:
    import base64

    from tests.engine._unit_db import TIMESTEPS, time_map

    found = [pv for pv in data["parameter_values"]
             if pv[:3] == [series.cls, series.entity, series.parameter] and pv[4] == series.alternative]
    if len(found) != 1:
        raise GroundingBuildError(f"tests.json has {len(found)} values of {series}")
    decoded = json.loads(base64.b64decode(found[0][3][0]))
    values = {str(t): float(v) for t, v in decoded["data"]}
    if sorted(values) != TIMESTEPS:
        raise GroundingBuildError(f"{series} does not cover the fixture's timesteps")
    return time_map(values, 0.0)


def pack(data: dict, value: Value) -> list:
    """*value* in the Spine import form (``[base64 JSON, type]``)."""
    if isinstance(value, FixtureSeries):
        return _fixture_value(data, value)
    if isinstance(value, Packed):
        return _b64(value.data, value.kind)
    if isinstance(value, bool):
        raise TypeError("a boolean override value has no FlexTool type here")
    if isinstance(value, (int, float)):
        return _b64(float(value), "float")
    if isinstance(value, str):
        return _b64(value, "str")
    raise TypeError(f"not an override value: {value!r}")


def input_data(build: Literal[1, 2] = 1) -> dict:
    """``tests.json`` with the overrides of *build* appended (build 2: :data:`DEMAND_UPDATE_REWRITE` in place of
    :data:`DEMAND_UPDATE`)."""
    from tests.engine._unit_db import add_override_alternative

    data = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    for override in GROUNDING_OVERRIDES:
        if build == 2 and override is DEMAND_UPDATE:
            override = DEMAND_UPDATE_REWRITE
        values = [[c, e, p, pack(data, v)] for c, e, p, v in override.values]
        entities = [list(ent) for ent in override.entities]
        add_override_alternative(data, override.alt, values, list(override.base_chain), entities)
    return data


def build_inputs(work: Path, *, build: Literal[1, 2] = 1) -> Path:
    """The grounding database of *build* at ``<work>/input_sources/grounding.sqlite`` (migrated to this
    FlexTool); an existing one is replaced in one step.  Returns its path."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    target = Path(work) / "input_sources" / SOURCE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(work) / "intermediate" / f"grounding_build_{build}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    json_path = staging / "grounding.json"
    json_path.write_text(json.dumps(input_data(build)), encoding="utf-8")
    url = json_to_db(json_path, staging / SOURCE_NAME)
    migrate_database(url)
    os.replace(staging / SOURCE_NAME, target)
    shutil.rmtree(staging)
    return target


def override_data() -> list:
    """The override data as plain JSON values (what the fingerprint covers)."""
    return [dataclasses.asdict(o) for o in (*GROUNDING_OVERRIDES, DEMAND_UPDATE_REWRITE)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(source_sha256: str, run_ids: list[str | None]) -> str:
    """sha256 of the source database's hash, the override data and each run's ``run_id``, in run order."""
    payload = json.dumps({"source_sha256": source_sha256, "overrides": override_data(), "runs": run_ids},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- the project ----------------------------------------------------------------------------------------------
def _new_project(root: Path) -> Path:
    from flextool.gui.project_utils import PROJECT_SUBDIRS

    project = Path(root) / PROJECT_NAME
    if project.exists():
        raise GroundingBuildError(f"{project} exists already; build into an empty folder")
    project.mkdir(parents=True)
    for sub in PROJECT_SUBDIRS:
        (project / sub).mkdir()
    return project


def _number_sources(project: Path) -> int:
    """Number the input sources as the window does (``InputSourceManager.refresh``) and save the settings."""
    from flextool.gui.input_sources import InputSourceManager
    from flextool.gui.settings_io import load_project_settings, save_project_settings

    settings = load_project_settings(project)
    InputSourceManager(project, settings).refresh()
    save_project_settings(project, settings)
    number = settings.input_source_numbers.get(SOURCE_NAME)
    if number is None:
        raise GroundingBuildError(f"{SOURCE_NAME} got no source number")
    return int(number)


def run_command(project: Path, source: Path, scenario: str, subdir: str) -> list[str]:
    """``flextool-run`` with the window's command line (``gui/execution_manager.py`` ``_build_run_command``),
    parquet results only."""
    return [sys.executable, "-m", "flextool_run_supervisor",
            "sqlite:///" + str(source),
            "--scenario-name", scenario,
            "--work-folder", str(project / "work" / subdir),
            "--output-location", str(project),
            "--output-subdir", subdir,
            "--write-methods", "parquet",
            "--threads", str(RUN_THREADS)]


def _run(project: Path, source: Path, scenario: str, build: int, number: int) -> GroundingRun:
    from flextool import run_record

    subdir = f"{scenario}_{number}"
    work = project / "work" / subdir
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    proc = subprocess.run(run_command(project, source, scenario, subdir), cwd=str(project.parent),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", check=False)
    status = run_record.read_status_file(project / "output_parquet" / subdir) or {}
    error = status.get("error") or {}
    run = GroundingRun(scenario=scenario, build=build, folder=subdir, run_id=status.get("run_id"),
                       state=status.get("state"), exit_code=proc.returncode,
                       error=str(error.get("message")) if isinstance(error, dict) and error.get("message") else None)
    expected_failure = (scenario, build) == FAILED_RUN
    ok = (run.state == "failed" and run.exit_code != 0 and FAILED_REASON in (run.error or "")) if expected_failure \
        else (run.state == "succeeded" and run.exit_code == 0)
    if not ok:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        want = f"a failed run whose reason says {FAILED_REASON!r}" if expected_failure else "a successful run"
        raise GroundingBuildError(f"{scenario} (build {build}) was not {want}: state {run.state}, exit code "
                                  f"{run.exit_code}, error {run.error}\n{tail}")
    return run


def write_marker(project: Path, *, fingerprint_value: str, source_sha256: str, solved: bool,
                 runs: dict[str, str | None]) -> Path:
    path = project / MARKER_FILE
    path.write_text(json.dumps({"format": MARKER_FORMAT, "set_version": SET_VERSION,
                                "fingerprint": fingerprint_value, "source_sha256": source_sha256,
                                "solved": solved, "runs": runs}, indent=1) + "\n", encoding="utf-8")
    return path


def build_grounding_project(root: Path, *, solve: bool = True) -> GroundingProject:
    """The grounding project under *root* (``<root>/Grounding project ä``, which must not exist): the inputs, the
    source numbered as the window numbers it, the runs of :data:`RUN_ORDER` (``solve=False``: the first build's
    inputs only, no runs, and a marker the runner refuses unless a test allows it) and the marker."""
    root = Path(root).expanduser().resolve()  # the runs work in other folders: a relative root would break them
    project = _new_project(root)
    source = build_inputs(project, build=1)
    number = _number_sources(project)
    runs: list[GroundingRun] = []
    if solve:
        built = 1
        for scenario, build in RUN_ORDER:
            if build != built:
                source = build_inputs(project, build=build)
                built = build
            runs.append(_run(project, source, scenario, build, number))
    source_sha = sha256_file(source)
    value = fingerprint(source_sha, [r.run_id for r in runs])
    # A folder's final status is its last run's (demand_update_test_1 holds the failed second run).
    write_marker(project, fingerprint_value=value, source_sha256=source_sha, solved=solve,
                 runs={r.folder: r.run_id for r in runs})
    return GroundingProject(path=project, source=source, source_number=number, runs=tuple(runs),
                            fingerprint=value, source_sha256=source_sha)

