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
* :func:`save_provider` / :func:`load_saved` carry a provider to a child process (which cannot receive
  the in-memory object) as parquet files, and repeat the production load there.

Loads happen in ``run.load_folder``, an empty sibling of the work folder (which holds the provider
snapshot CSVs and a ``tests.sqlite`` copy).  The loader reads neither any more (c24 stage 3: provider
only, no discovery), so the empty folder only keeps the load free of files production never has.

Mutation contract: the providers, handoffs and ``last_flex_data`` held by a :class:`ScenarioRun` are
shared by every test that uses the (session-cached) run.  Never mutate them; clone a provider with
:func:`clone_provider` before changing it.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from flextool.engine.input._spinedb_reader import SpineDbReader
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.flex_data import FlexData, load_flextool
from flextool.engine.solve._orchestration import run_chain_from_db

__all__ = [
    "ScenarioRun",
    "StepInputs",
    "clone_provider",
    "load_saved",
    "load_scenario",
    "read_saved_provider",
    "run_scenario",
    "save_provider",
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
    (the last that is not an engine-added realization solve) to CSVs under ``work_folder`` and creates the empty sibling ``load_<scenario>`` folder.  A step
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
    # The snapshot is of the last solve that is not an engine-added realization solve: after a stochastic solve
    # with a realized: yes designation the run ends with ``<solve>__realized_dispatch``, a deterministic dispatch
    # without the fan; the files tests read (branch sets, folds) are the stochastic solve's.
    snap = [s for s in step_inputs if "__realized_dispatch" not in s.solve_name] or step_inputs
    if snap and snap[-1].provider is not None:
        snap[-1].provider.snapshot_processed_inputs(work_folder)
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
    """The scenario-filtered axis Enum vocabulary and contract, built as an axis-Enum run builds them."""
    from flextool.spinedb_backend._axis_enums import build_scenario_axis_enums

    return build_scenario_axis_enums(db_url, scenario)


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
    p = clone_provider(target.provider) if provider is None else provider
    return _production_load(
        run.load_folder, p, run.db_url, run.scenario, handoff=target.prior_handoff, axis_enums=axis_enums,
    )


def _production_load(
    load_folder: Path,
    p: FlexDataProvider,
    db_url: str,
    scenario: str,
    *,
    handoff: object | None,
    axis_enums: bool,
) -> FlexData:
    """``load_flextool(load_folder, provider=p, db_reader=SpineDbReader(db_url, scenario), handoff=handoff)``.

    Refuses a missing or non-empty ``load_folder`` (a file there would feed the loader's disk fallbacks).
    ``axis_enums=True`` sets the scenario's Enum vocabulary on ``p`` (which must be the caller's own
    provider, never a cached one) and on the reader; ``False`` refuses a provider that carries Enums.
    """
    leftovers = sorted(f.name for f in load_folder.iterdir()) if load_folder.exists() else None
    if leftovers is None:
        raise FileNotFoundError(f"load folder {load_folder} is missing")
    if leftovers:
        raise RuntimeError(
            f"load folder {load_folder} is not empty ({leftovers[:5]}); a file there would feed the "
            f"loader's disk fallbacks for every later load of this run"
        )
    if axis_enums:
        enums, contract = _scenario_axis_enums(db_url, scenario)
        p.axis_enums = enums
        p.contract = contract
        reader = SpineDbReader(db_url, scenario=scenario, axis_enums=enums, contract=contract)
    else:
        if p.axis_enums is not None:
            raise ValueError(
                "load_scenario: the provider carries axis Enums but axis_enums=False; production loads "
                "without them.  Pass axis_enums=True or a provider without Enums."
            )
        reader = SpineDbReader(db_url, scenario=scenario)
    return load_flextool(load_folder, provider=p, db_reader=reader, handoff=handoff)


_MANIFEST = "manifest.json"


def save_provider(p: FlexDataProvider, folder: Path) -> None:
    """Write ``p``'s frames to ``folder`` so :func:`read_saved_provider` can rebuild it in another process.

    One parquet file per key (parquet keeps the dtypes; a CSV would not), plus ``manifest.json`` with the
    key → file map and the source tags.  ``folder`` must be missing or empty.  Raises for a provider
    with evicted frames or axis Enums: neither survives the files (the child builds Enums itself with
    ``load_saved(..., axis_enums=True)``).
    """
    if p._evicted:
        raise ValueError(f"save_provider: the provider has evicted frames ({sorted(p._evicted)[:5]}…)")
    if p.axis_enums is not None:
        raise ValueError("save_provider: the provider carries axis Enums, which are not saved")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise RuntimeError(f"save_provider: {folder} is not empty")
    frames: dict[str, str] = {}
    for i, (key, frame) in enumerate(p._frames.items()):
        name = f"{i:05d}.parquet"
        frame.write_parquet(folder / name)
        frames[key] = name
    (folder / _MANIFEST).write_text(json.dumps({"frames": frames, "sources": p._sources}, indent=1))


def read_saved_provider(folder: Path) -> FlexDataProvider:
    """The provider :func:`save_provider` wrote to ``folder`` (same keys, frames and source tags)."""
    folder = Path(folder)
    manifest = json.loads((folder / _MANIFEST).read_text())
    q = FlexDataProvider(retain_all=True)
    q._frames = {key: pl.read_parquet(folder / name) for key, name in manifest["frames"].items()}
    q._sources = dict(manifest["sources"])
    return q


def load_saved(
    folder: Path,
    db_url: str,
    scenario: str,
    load_folder: Path,
    *,
    axis_enums: bool = False,
) -> FlexData:
    """The production loader call on a provider saved with :func:`save_provider` (for a child process).

    ``load_flextool(load_folder, provider=read_saved_provider(folder), db_reader=SpineDbReader(db_url,
    scenario))``.  No handoff is passed, so save only a step that consumed none (``prior_handoff is
    None``: a single-solve run or the first step).  ``load_folder`` must exist and be empty;
    ``axis_enums`` as in :func:`load_scenario`.
    """
    p = read_saved_provider(folder)
    return _production_load(Path(load_folder), p, db_url, scenario, handoff=None, axis_enums=axis_enums)
