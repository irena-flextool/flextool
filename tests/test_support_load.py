"""Tests for ``tests/_support/load.py``: the production loader call that tests use to rebuild a sub-solve.

The rebuild tests are relational (rebuild from the post-solve provider == what the cascade solved), so their
expected values come from the cascade itself, not from literals.
"""
from __future__ import annotations

import dataclasses

import polars as pl
import pytest
from polar_high import Problem

from flextool.engine import build_flextool
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from polar_high import Param

from tests._support.load import (
    clone_provider,
    load_saved,
    load_scenario,
    read_saved_provider,
    save_provider,
)

pytestmark = pytest.mark.solver


@pytest.mark.parametrize(
    ("scenario", "db_fixture"),
    [
        ("base", "main"),
        ("multi_year", "main"),
        ("wind_battery_invest_lifetime_renew_4solve", "main"),
        ("1_week_rolling_wind", "stochastic"),
    ],
)
def test_rebuild_from_the_post_solve_provider_is_the_solved_lp(scenario_run, scenario, db_fixture):
    """Rebuilding the last sub-solve with ``load_scenario`` and solving it gives the cascade's own objective.

    Multi-solve scenarios (``multi_year``, the 4-solve chain, the stochastic rolling run) only match when the
    prior solve's handoff carry reaches the load, which a CSV reload of the snapshot misses.
    """
    run = scenario_run(scenario, db_fixture)
    assert run.last_obj is not None
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    assert sol.obj == pytest.approx(run.last_obj, rel=1e-9, abs=1e-9)


def test_scenario_workdir_is_the_runs_work_folder(scenario_run, scenario_workdir):
    """``scenario_workdir`` and ``scenario_run`` share one cached cascade per scenario."""
    run = scenario_run("base")
    work = scenario_workdir("base")
    assert work == run.work_folder
    assert scenario_run("base") is run
    assert (work / "tests.sqlite").is_file()
    assert (work / "input").is_dir() and (work / "solve_data").is_dir()
    assert run.load_folder.is_dir() and run.load_folder != run.work_folder


def test_clone_isolates_the_cached_provider(scenario_run):
    """A clone's ``put`` never reaches the session-cached provider; loads neither change the cached provider's
    keys nor set axis Enums on it, and repeated loads agree."""
    run = scenario_run("base")
    original = run.last.provider
    keys_before = set(original.keys())

    p = clone_provider(original)
    p.put("solve_data/x", pl.DataFrame({"a": [1]}))
    assert p.has("solve_data/x")
    assert not original.has("solve_data/x")

    first = load_scenario(run)
    second = load_scenario(run)
    assert first.p_inflow.frame.height > 0
    assert first.p_inflow.frame.equals(second.p_inflow.frame)

    load_scenario(run, axis_enums=True)
    assert original.axis_enums is None
    assert set(original.keys()) == keys_before
    assert list(run.load_folder.iterdir()) == []


def test_production_enum_state(scenario_run):
    """Production loads without axis Enums; ``axis_enums=True`` turns the activation on for Enum tests."""
    run = scenario_run("base")
    plain = load_scenario(run)
    assert plain._axis_enums is None
    assert plain.p_inflow.frame.schema["n"] == pl.String

    enum_data = load_scenario(run, axis_enums=True)
    assert enum_data._axis_enums is not None
    assert isinstance(enum_data.p_inflow.frame.schema["n"], pl.Enum)


def test_step_guard_rejects_a_roll_whose_provider_a_later_roll_shares(scenario_run):
    """Rolls of one level share one provider, so an earlier roll cannot be rebuilt from it."""
    run = scenario_run("fullYear_roll")
    shared = [
        s.solve_name for i, s in enumerate(run.steps)
        if s.provider is not None and any(t.provider is s.provider for t in run.steps[i + 1:])
    ]
    assert shared, f"fullYear_roll should have rolls sharing a provider; steps: {[s.solve_name for s in run.steps]}"
    with pytest.raises(ValueError, match="shares its provider"):
        load_scenario(run, step=shared[0])
    with pytest.raises(KeyError, match="no step"):
        load_scenario(run, step="no_such_solve")


def test_step_without_provider_raises_at_load(scenario_run):
    """A step without a provider is kept in the run (so snapshot-only users still work) and raises on load."""
    run = scenario_run("base")
    broken = dataclasses.replace(
        run, steps=run.steps[:-1] + (dataclasses.replace(run.last, provider=None),),
    )
    with pytest.raises(ValueError, match=f"step {run.last.solve_name!r} has no flex_data_provider"):
        load_scenario(broken)


def test_non_empty_load_folder_raises(scenario_run, tmp_path):
    """A file in the load folder would feed the loader's disk fallbacks, so the load refuses to run."""
    run = scenario_run("base")
    (tmp_path / "input").mkdir()
    moved = dataclasses.replace(run, load_folder=tmp_path)
    with pytest.raises(RuntimeError, match="is not empty"):
        load_scenario(moved)


def test_clone_refuses_an_evicted_provider():
    """A clone of a provider with evicted frames would read them as absent instead of raising."""
    p = FlexDataProvider()
    p.put("solve_data/x", pl.DataFrame({"a": [1]}))
    p._evicted["solve_data/y"] = "group_1"
    with pytest.raises(ValueError, match="evicted frames"):
        clone_provider(p)


def test_enum_provider_needs_axis_enums_flag(scenario_run):
    """A provider carrying axis Enums is rejected by a production-mode (``axis_enums=False``) load."""
    run = scenario_run("base")
    p = clone_provider(run.last.provider)
    p.axis_enums = {"node": pl.Enum(["n"])}
    with pytest.raises(ValueError, match="carries axis Enums"):
        load_scenario(run, provider=p)


def _same_field(a, b) -> bool:
    """Exact equality of two FlexData field values.  A dataclass value (``block_layout``) is compared field by
    field; a value of any other type raises, so a new field type is never compared by type alone."""
    if isinstance(a, Param) and isinstance(b, Param):
        return a.dims == b.dims and a.frame.schema == b.frame.schema and a.frame.equals(b.frame)
    if isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame):
        return a.schema == b.schema and a.equals(b)
    if a is None or b is None or isinstance(a, (bool, int, float, str, dict, list, tuple, set, frozenset)):
        return a == b
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _same_field(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    raise TypeError(f"_same_field: no comparison for {type(a).__name__}")


def test_saved_provider_round_trips_and_loads_like_load_scenario(scenario_run, tmp_path):
    """``save_provider`` → ``read_saved_provider`` gives back every key, dtype, frame and source tag, and
    ``load_saved`` (the child-process load) builds the same FlexData as ``load_scenario``."""
    run = scenario_run("base")
    original = run.last.provider
    assert run.last.prior_handoff is None
    saved = tmp_path / "provider"
    save_provider(original, saved)

    back = read_saved_provider(saved)
    assert back.keys() == original.keys()
    for key in original.keys():
        a, b = original.get(key), back.get(key)
        assert a.schema == b.schema, key
        assert a.equals(b), key
    assert back._sources == original._sources

    load_folder = tmp_path / "load"
    load_folder.mkdir()
    via_saved = load_saved(saved, run.db_url, run.scenario, load_folder)
    direct = load_scenario(run)
    differing = [
        f.name for f in dataclasses.fields(direct)
        if not _same_field(getattr(direct, f.name), getattr(via_saved, f.name))
    ]
    assert differing == []
    assert direct.block_layout is not None  # compared field by field above, not by type alone
    assert via_saved._axis_enums is None
    assert list(load_folder.iterdir()) == []

    with_enums = load_saved(saved, run.db_url, run.scenario, load_folder, axis_enums=True)
    assert isinstance(with_enums.p_inflow.frame.schema["n"], pl.Enum)


def test_save_provider_refuses_what_it_cannot_carry(scenario_run, tmp_path):
    """Axis Enums and a non-empty target folder are refused instead of being silently lost or mixed."""
    run = scenario_run("base")
    p = clone_provider(run.last.provider)
    p.axis_enums = {"node": pl.Enum(["n"])}
    with pytest.raises(ValueError, match="axis Enums"):
        save_provider(p, tmp_path / "enums")
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x").write_text("")
    with pytest.raises(RuntimeError, match="is not empty"):
        save_provider(run.last.provider, tmp_path / "used")
