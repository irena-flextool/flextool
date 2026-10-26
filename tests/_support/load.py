"""Rebuild a finished cascade's FlexData through the production loader call.

Production builds every sub-solve's FlexData with one call (``_orchestration.py``, the per-solve loop)::

    load_flextool(work_folder, provider=<the sub-solve's provider>,
                  db_reader=SpineDbReader(db_url, scenario), handoff=<prior solve's SolveHandoff>)

Tests that need "the FlexData of sub-solve X" must make the same call, not reload a CSV dump of the
provider (the reload path misses the in-memory handoff carry, ``storage_solve_last`` and other
provider-only state, and builds axis Enums that production never builds).  This module is that call:

* :func:`run_scenario` runs the cascade once and keeps, per step, the post-solve provider and the
  handoff the step consumed (:class:`ScenarioRun`).
* :func:`load_scenario` repeats the production load for one step of that run.
* :func:`clone_provider` gives a test its own provider to change.

Loads happen in ``run.load_folder``, an empty sibling of the work folder.  Until the loader's disk
fallbacks are removed, a load in the work folder (which holds the provider snapshot CSVs and a
``tests.sqlite`` copy) would read those files for keys the provider lacks and would build axis
Enums.  Production's work folder has neither, so the empty folder is the faithful stand-in.

Mutation contract: the providers, handoffs and ``last_flex_data`` held by a :class:`ScenarioRun` are
shared by every test that uses the (session-cached) run.  Never mutate them; clone a provider with
:func:`clone_provider` before changing it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from flextool.engine.input._spinedb_reader import SpineDbReader
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.flex_data import FlexData, load_flextool
from flextool.engine.solve._orchestration import run_chain_from_db

__all__ = [
    "ScenarioRun",
    "StepInputs",
    "clone_provider",
    "load_scenario",
    "run_scenario",
    "sqlite_path_from_url",
]


@dataclass(frozen=True)
class StepInputs:
    """What production passed to ``load_flextool`` for one sub-solve, plus its objective.

    ``provider`` is the step's post-solve provider (``OrchestrationStep.flex_data_provider``); ``None``
    when the step has none (a failed or partial step).  ``prior_handoff`` is the handoff of the previous
    step in execution order (the one production loads with), ``None`` for the first step.  Both are
    shared with the session cache: never mutate them.
    """

    solve_name: str
    provider: FlexDataProvider | None
    prior_handoff: object | None
    obj: float | None


@dataclass(frozen=True)
class ScenarioRun:
    """A finished cascade run, kept so tests can rebuild any step's FlexData (see :func:`load_scenario`).

    ``work_folder`` is the cascade's ``work_<scenario>`` folder; it also holds the provider snapshot CSVs
    and a ``tests.sqlite`` copy of the DB (for tests that read those files).  ``load_folder`` is the empty
    folder every :func:`load_scenario` loads from.  ``last_flex_data`` is the FlexData the cascade built
    for its last step (read-only, shared).
    """

    scenario: str
    db_fixture: str | None
    db_url: str
    work_folder: Path
    load_folder: Path
    steps: tuple[StepInputs, ...]
    last_flex_data: FlexData | None
    last_obj: float | None

    @property
    def last(self) -> StepInputs:
        """The last step in execution order."""
        if not self.steps:
            raise ValueError(f"scenario {self.scenario!r}: the cascade produced no steps")
        return self.steps[-1]

    def step(self, solve_name: str) -> StepInputs:
        """The step called ``solve_name``; ``KeyError`` lists the available names."""
        for s in self.steps:
            if s.solve_name == solve_name:
                return s
        raise KeyError(
            f"scenario {self.scenario!r} has no step {solve_name!r}; steps: {[s.solve_name for s in self.steps]}"
        )

    def reader(self) -> SpineDbReader:
        """A fresh DB reader for the run's scenario, without axis Enums (as production builds it)."""
        return SpineDbReader(self.db_url, scenario=self.scenario)


def sqlite_path_from_url(db_url: str) -> Path:
    """The file path of a ``sqlite:///<path>`` URL (Windows ``/C:/…`` handled)."""
    path = urlparse(db_url).path
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def clone_provider(p: FlexDataProvider) -> FlexDataProvider:
    """A provider holding the same frames as ``p``; ``put()`` on the copy never reaches ``p``.

    Shallow: the frame dict and the source tags are copied, the (immutable) polars frames are shared.
    ``axis_enums`` / ``contract`` are carried over.  The copy has ``retain_all=True`` and no lifetimes,
    so it never evicts.  Raises if ``p`` has evicted frames: the copy would silently return ``None``
    for them where ``p`` raises ``EvictedFrameError``.
    """
    if p._evicted:
        raise ValueError(
            f"clone_provider: the provider has evicted frames ({sorted(p._evicted)[:5]}…); a clone would "
            f"read them as absent.  Is FLEXTOOL_RSS_BUDGET_MB set?"
        )
    q = FlexDataProvider(retain_all=True)
    q._frames = dict(p._frames)
    q._sources = dict(p._sources)
    q.axis_enums = p.axis_enums
    q.contract = p.contract
    return q


def run_scenario(
    db_url: str,
    scenario: str,
    work_folder: Path,
    *,
    solver_config_dir: Path,
    db_fixture: str | None = None,
) -> ScenarioRun:
    """Run the cascade for ``scenario`` in ``work_folder`` and keep what :func:`load_scenario` needs.

    Runs ``run_chain_from_db(csv_dump=False, keep_solutions=True)`` (``keep_solutions`` keeps every
    step's provider), copies the DB to ``work_folder/tests.sqlite``, snapshots the last step's provider
    to CSVs under ``work_folder`` and creates the empty sibling ``load_<scenario>`` folder.  A step
    without a provider is kept with ``provider=None``; loading it raises (see :func:`load_scenario`).
    """
    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    load_folder = work_folder.parent / f"load_{scenario}"
    load_folder.mkdir()
    steps = run_chain_from_db(
        input_db_url=db_url,
        scenario_name=scenario,
        work_folder=work_folder,
        solver_config_dir=solver_config_dir,
        csv_dump=False,
        keep_solutions=True,
    )
    shutil.copy(sqlite_path_from_url(db_url), work_folder / "tests.sqlite")
    ordered = list(steps.items())
    step_inputs: list[StepInputs] = []
    for i, (name, step) in enumerate(ordered):
        step_inputs.append(StepInputs(
            solve_name=name,
            provider=getattr(step, "flex_data_provider", None),
            prior_handoff=ordered[i - 1][1].handoff if i > 0 else None,
            obj=step.obj,
        ))
    if step_inputs and step_inputs[-1].provider is not None:
        step_inputs[-1].provider.snapshot_processed_inputs(work_folder)
    last_step = ordered[-1][1] if ordered else None
    return ScenarioRun(
        scenario=scenario,
        db_fixture=db_fixture,
        db_url=db_url,
        work_folder=work_folder,
        load_folder=load_folder,
        steps=tuple(step_inputs),
        last_flex_data=getattr(last_step, "flex_data", None),
        last_obj=last_step.obj if last_step is not None else None,
    )


def _scenario_axis_enums(db_url: str, scenario: str) -> tuple[dict, object]:
    """The scenario-filtered axis Enum vocabulary and contract, built as ``load_flextool``'s auto-build does."""
    from flextool.spinedb_backend import SpineDBBackend
    from flextool.spinedb_backend._axis_enums import build_axis_enums, load_axis_contract

    contract = load_axis_contract()
    with SpineDBBackend(db_url, scenario) as backend:
        enums = build_axis_enums(backend, contract)
    return enums, contract


def load_scenario(
    run: ScenarioRun,
    *,
    step: str | None = None,
    provider: FlexDataProvider | None = None,
    axis_enums: bool = False,
) -> FlexData:
    """The production loader call for one step of a finished run.

    ``load_flextool(run.load_folder, provider=<provider or clone_provider(step.provider)>,
    db_reader=run.reader(), handoff=step.prior_handoff)``.

    * ``step``: a sub-solve name (default: the last step).  Raises ``ValueError`` if a later step holds
      the same provider object (rolls of one level share one provider, so its frames are the later
      roll's), or if the step has no provider.
    * ``provider``: a provider the test prepared (normally ``clone_provider(run.last.provider)`` with a
      key changed); used as given.  It must not carry axis Enums unless ``axis_enums=True``.
    * ``axis_enums``: build the scenario's axis Enum vocabulary from ``run.db_url``, set it on the
      provider used for the load (the clone, never the cached provider) and give it to the reader.
      Only for tests of the Enum activation itself; the default ``False`` is what production does.
    """
    target = run.last if step is None else run.step(step)
    if target.provider is None:
        raise ValueError(
            f"scenario {run.scenario!r}: step {target.solve_name!r} has no flex_data_provider "
            f"(failed or partial step); it cannot be rebuilt"
        )
    idx = next(i for i, s in enumerate(run.steps) if s is target)
    sharing = [s.solve_name for s in run.steps[idx + 1:] if s.provider is target.provider]
    if sharing:
        raise ValueError(
            f"scenario {run.scenario!r}: step {target.solve_name!r} shares its provider with later step(s) "
            f"{sharing}; the provider holds the last of them, so step {target.solve_name!r} cannot be rebuilt"
        )
    leftovers = sorted(p.name for p in run.load_folder.iterdir()) if run.load_folder.exists() else None
    if leftovers is None:
        raise FileNotFoundError(f"load folder {run.load_folder} is missing")
    if leftovers:
        raise RuntimeError(
            f"load folder {run.load_folder} is not empty ({leftovers[:5]}); a file there would feed the "
            f"loader's disk fallbacks for every later load of this run"
        )
    p = clone_provider(target.provider) if provider is None else provider
    if axis_enums:
        enums, contract = _scenario_axis_enums(run.db_url, run.scenario)
        p.axis_enums = enums
        p.contract = contract
        reader = SpineDbReader(run.db_url, scenario=run.scenario, axis_enums=enums, contract=contract)
    else:
        if p.axis_enums is not None:
            raise ValueError(
                "load_scenario: the provider carries axis Enums but axis_enums=False; production loads "
                "without them.  Pass axis_enums=True or a provider without Enums."
            )
        reader = run.reader()
    return load_flextool(run.load_folder, provider=p, db_reader=reader, handoff=target.prior_handoff)
