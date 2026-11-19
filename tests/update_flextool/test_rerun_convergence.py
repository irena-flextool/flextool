"""A migration killed during or right after a step converges when it is run again (items 284 and 285, MIG-1).

Each step's commit carries the previous step's ``model.version`` stamp (``_stage_version``), so a hard kill (no
restore of the backup copy: the copy deleted, a remote database) leaves the stamp at most one step behind what was
committed: content k under stamp k-1, or step k part way under stamp k-1.  A rerun then repeats step k, which must
recognise its own effect.

The databases are built from the bundled template JSONs and the v75 seed JSON (never a checked-in ``.sqlite``).  The
oracle is the clean migration: content k reached step by step from the seed.  A rerun is compared with it at k
(``_rerun.snapshot``: every definition, value, list, class, entity, alternative, group and metadata row).  A clean
migration logs no ``_already_applied`` record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, import_data, to_database

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.fixture_schema import unpack_value
from flextool.update_flextool.initialize_database import initialize_database

from tests.update_flextool._rerun import (Kill, already_applied_records, copy_db, diff, kill_after_commit, migrate,
                                          quiet, restamp_file, snapshot, url_of, version_of)

TEMPLATES = package_data_path("schemas/pre_v26")
SEED75 = Path(__file__).parent / "data" / "v75_seed_stoch_two_period.json"

#: The steps whose rerun on their own result failed (an item looked up after the step removed or renamed it, or a
#: raw commit with nothing to commit) or changed values (v37 flipped the flipped coefficient back).
SELF_RERUN_STEPS = (35, 37, 38, 52, 57, 59, 61, 65)
#: The steps that detect their own effect (the others only tolerate an empty commit).
GUARDED_STEPS = (35, 37, 38, 52, 65)


def _seed_data(path: Path, *, base: bool) -> None:
    """A unit between two nodes, a connection, a group, a reserve, a profile and a commodity, their relations, and a
    few values in alternative ``a1`` (``coefficient`` 0.5 in, 2.0 out); with *base*, also a ``Base`` alternative."""
    with DatabaseMapping(url_of(path)) as db:
        db.add_alternative(name="a1")
        if base:
            db.add_alternative(name="Base")
        classes = {c["name"] for c in db.get_items("entity_class")}

        def entity(cls, *names):
            if cls not in classes:
                return
            if len(names) == 1:
                db.add_entity(entity_class_name=cls, name=names[0])
            else:
                db.add_entity(entity_class_name=cls, element_name_list=names)

        for cls, name in [("unit", "u1"), ("node", "n_in"), ("node", "n_out"), ("connection", "c1"), ("group", "g1"),
                          ("reserve", "r1"), ("profile", "p1"), ("commodity", "fuel")]:
            entity(cls, name)
        if "upDown" in classes and not db.get_item("entity", entity_class_name="upDown", name="up"):
            entity("upDown", "up")
        for cls, names in [("unit__inputNode", ("u1", "n_in")), ("unit__outputNode", ("u1", "n_out")),
                           ("connection__node__node", ("c1", "n_in", "n_out")),
                           ("reserve__upDown__group", ("r1", "up", "g1")), ("node__profile", ("n_in", "p1")),
                           ("connection__profile", ("c1", "p1")), ("unit__node__profile", ("u1", "n_out", "p1")),
                           ("group__node", ("g1", "n_in")), ("group__unit", ("g1", "u1")),
                           ("commodity__node", ("fuel", "n_in"))]:
            entity(cls, *names)
        definitions = {(d["entity_class_name"], d["name"]) for d in db.get_items("parameter_definition")}
        for cls, byname, name, value in [
                ("unit__inputNode", ("u1", "n_in"), "coefficient", 0.5),
                ("unit__outputNode", ("u1", "n_out"), "coefficient", 2.0),
                ("unit", ("u1",), "existing", 100.0), ("connection", ("c1",), "existing", 50.0),
                ("node", ("n_in",), "inflow", -10.0), ("unit__outputNode", ("u1", "n_out"), "ramp_speed_up", 0.5),
                ("unit__inputNode", ("u1", "n_in"), "ramp_speed_up", 0.3), ("group", ("g1",), "has_inertia", "yes"),
                ("connection", ("c1",), "is_DC", "yes"), ("unit", ("u1",), "efficiency", 0.4)]:
            if (cls, name) in definitions:
                val, typ = to_database(value)
                db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=name,
                                       alternative_name="a1", value=val, type=typ)
        db.commit_session("seed data")


@pytest.fixture(scope="module")
def v2_chain(tmp_path_factory) -> dict:
    """``{k: path}``: the v2 template (version 0) with the seed data and ``Base``, migrated one step at a time; the
    file at k is the clean content k.  ``"records"``: what the clean chain logged as already applied."""
    folder = tmp_path_factory.mktemp("v2_chain")
    current = folder / "current.sqlite"
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v2.json"), str(current))
    _seed_data(current, base=True)
    chain = {0: copy_db(current, folder / "k0.sqlite")}
    with already_applied_records() as records:
        for k in range(1, FLEXTOOL_DB_VERSION + 1):
            migrate(current, up_to=k)
            chain[k] = copy_db(current, folder / f"k{k}.sqlite")
    chain["records"] = records
    return chain


@pytest.fixture(scope="module")
def v75_chain(v2_chain, tmp_path_factory) -> dict:
    """``{k: path}`` for k 75..current: the clean v2 chain at 75 (a real v75 schema) with the data of the v75 seed
    JSON and a valueless ``unit__node__profile``, migrated one step at a time."""
    folder = tmp_path_factory.mktemp("v75_chain")
    current = copy_db(v2_chain[75], folder / "current.sqlite")
    with open(SEED75, encoding="utf-8") as f:
        data = json.load(f)
    with DatabaseMapping(url_of(current)) as db:
        _, errors = import_data(db, unparse_value=unpack_value, **{key: data[key] for key in (
            "alternatives", "scenarios", "scenario_alternatives", "entities", "entity_alternatives",
            "parameter_values")})
        assert not errors, errors[:3]
        db.add_entity(entity_class_name="unit__node__profile", element_name_list=("thermal", "city", "wprof"))
        db.commit_session("v75 seed data")
    chain = {75: copy_db(current, folder / "k75.sqlite")}
    with already_applied_records() as records:
        for k in range(76, FLEXTOOL_DB_VERSION + 1):
            migrate(current, up_to=k)
            chain[k] = copy_db(current, folder / f"k{k}.sqlite")
    chain["records"] = records
    return chain


def test_a_clean_migration_finds_nothing_already_applied(v2_chain, v75_chain, tmp_path) -> None:
    """The guards key on an effect a clean run does not have yet: the v2, v25 and v75 chains log none."""
    assert v2_chain["records"] == []
    assert v75_chain["records"] == []
    path = tmp_path / "v25.sqlite"
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v25.json"), str(path))
    _seed_data(path, base=False)
    with already_applied_records() as records:
        migrate(path)
    assert version_of(path) == FLEXTOOL_DB_VERSION
    assert records == []


def test_one_step_chain_equals_one_shot_migration(v2_chain, tmp_path) -> None:
    """The staged stamp changes no content: the seed migrated in one run equals the chain's end."""
    path = copy_db(v2_chain[0], tmp_path / "db.sqlite")
    migrate(path)
    assert diff(snapshot(v2_chain[FLEXTOOL_DB_VERSION]), snapshot(path)) == []


@pytest.mark.parametrize("k", SELF_RERUN_STEPS)
def test_step_rerun_on_its_own_result_converges(v2_chain, tmp_path, k: int) -> None:
    """Content k under stamp k-1 (a kill right after step k's last commit) migrated to k equals content k."""
    path = copy_db(v2_chain[k], tmp_path / "db.sqlite")
    restamp_file(path, k - 1)
    with already_applied_records() as records:
        migrate(path, up_to=k)
    assert version_of(path) == k
    assert diff(snapshot(v2_chain[k]), snapshot(path)) == []
    assert bool(records) == (k in GUARDED_STEPS), records
    assert all(f" v{k} " in r for r in records), records


def _commits(path: Path, up_to: int) -> int:
    """How many commits migrating a copy of *path* to *up_to* makes (the last one is the version commit)."""
    probe = copy_db(path, path.with_name("probe.sqlite"))
    with kill_after_commit(None) as count:
        migrate(probe, up_to=up_to)
    probe.unlink()
    return count[0]


def _kill_and_rerun(clean: dict, k: int, j: int, work: Path) -> list:
    """Run step k on clean k-1, kill it right after its j-th commit, migrate to k again: the differences from clean
    k (an error as one entry)."""
    path = copy_db(clean[k - 1], work / f"k{k}_j{j}.sqlite")
    try:
        try:
            with kill_after_commit(j):
                migrate(path, up_to=k)
            return [("not killed", j)]
        except Kill:
            pass
        if version_of(path) != k - 1:
            return [("stamp after the kill", version_of(path))]
        migrate(path, up_to=k)
        return diff(snapshot(clean[k]), snapshot(path))
    except Exception as exc:  # noqa: BLE001 - reported as the case's result
        return [("error", f"{type(exc).__name__}: {exc}")]
    finally:
        path.unlink(missing_ok=True)


def test_v52_rerun_after_its_first_commit_converges(v2_chain, tmp_path) -> None:
    """v52 commits its value lists (the ``solver`` list already renamed ``solvers``) before its definitions."""
    assert _commits(v2_chain[51], 52) == 3
    assert _kill_and_rerun(v2_chain, 52, 1, tmp_path) == []


def test_a_kill_in_a_step_leaves_the_previous_steps_stamp(v2_chain, tmp_path) -> None:
    """Step 81's stamp rides on step 82's first commit: a kill right after it leaves the file at 81, not at 80."""
    step_81 = _commits(v2_chain[80], 81) - 1
    path = copy_db(v2_chain[80], tmp_path / "db.sqlite")
    with pytest.raises(Kill), kill_after_commit(step_81 + 1):
        migrate(path, up_to=82)
    assert version_of(path) == 81
    migrate(path, up_to=82)
    assert diff(snapshot(v2_chain[82]), snapshot(path)) == []


@pytest.mark.nightly_heavy  # PR CI skips it; runs nightly (pyproject markers)
@pytest.mark.timeout(3600)
def test_every_kill_inside_a_step_converges(v2_chain, v75_chain, tmp_path) -> None:
    """For every step k (12..current on the v2 chain, 76..current on the v75 chain) and every commit j of it: the
    run of step k on clean k-1 killed right after commit j, migrated to k again, equals clean k."""
    failures = []
    for chain, first in ((v2_chain, 12), (v75_chain, 76)):
        for k in range(first, FLEXTOOL_DB_VERSION + 1):
            for j in range(1, _commits(chain[k - 1], k)):
                result = _kill_and_rerun(chain, k, j, tmp_path)
                if result:
                    failures.append((first, k, j, result[:3]))
    assert failures == []
