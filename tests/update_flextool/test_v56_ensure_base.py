"""The v56 backfills of a database without a ``Base`` alternative (item 292).

v56 writes ``profile_method`` / ``ramp_method`` = ``none`` on the entities without a value and ``is_enabled = no`` on
the reserve provisions that were inactive by default into ``Base``.  A database without ``Base`` (every one made from
the pre-v26 templates: the v25 template has only a lower-case ``base``) gets a ``Base`` at the lowest rank of every
scenario, so that the values keep the older behaviour in every scenario and any value of another alternative wins.

The seed: the v25 template, alternative ``a1``, profiles on a node, a connection and a unit flow, two reserve
provisions (one active in ``a1`` by an entity_alternative row, one without a row: inactive before v56), a node
inflow in ``base``, and four scenarios: ``s1 = [a1]``, ``s2 = [base, a1]``, ``s_base = [base]`` and ``s_empty = []``.
The v38 floor (``default_node_type_from_migration``) takes rank 0 in the three scenarios that have a rank 1.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import from_database, to_database

from flextool._resources import package_data_path
from flextool.spinedb_backend import SpineDBBackend
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.initialize_database import initialize_database

from tests.update_flextool._rerun import (commit_messages, copy_db, diff, kill_and_rerun, migrate, quiet, restamp_file,
                                          snapshot, url_of, version_of)

TEMPLATES = package_data_path("schemas/pre_v26")
FLOOR = "default_node_type_from_migration"
PROVISION_ACTIVE = ("r1", "up", "u1", "n_out")
PROVISION_INACTIVE = ("r1", "up", "u2", "n_out")
SCENARIOS = ("s1", "s2", "s_base", "s_empty")


def _seed(path: Path) -> None:
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v25.json"), str(path))
    with open_database_mapping(url_of(path)) as db:
        db.add_alternative(name="a1")
        for cls, name in [("unit", "u1"), ("unit", "u2"), ("node", "n_in"), ("node", "n_out"), ("connection", "c1"),
                          ("profile", "p1"), ("reserve", "r1")]:
            db.add_entity(entity_class_name=cls, name=name)
        for cls, names in [("unit__inputNode", ("u1", "n_in")), ("unit__outputNode", ("u1", "n_out")),
                           ("node__profile", ("n_in", "p1")), ("connection__profile", ("c1", "p1")),
                           ("unit__node__profile", ("u1", "n_out", "p1")),
                           ("reserve__upDown__unit__node", PROVISION_ACTIVE),
                           ("reserve__upDown__unit__node", PROVISION_INACTIVE)]:
            db.add_entity(entity_class_name=cls, element_name_list=names)
        db.add_entity_alternative(entity_class_name="reserve__upDown__unit__node", entity_byname=PROVISION_ACTIVE,
                                  alternative_name="a1", active=True)
        for cls, name in [("unit", "u1"), ("unit", "u2"), ("node", "n_in"), ("node", "n_out")]:
            for alternative in ("base", "a1"):
                db.add_entity_alternative(entity_class_name=cls, entity_byname=(name,), alternative_name=alternative,
                                          active=True)
        val, typ = to_database(-10.0)
        db.add_parameter_value(entity_class_name="node", entity_byname=("n_in",), parameter_definition_name="inflow",
                               alternative_name="base", value=val, type=typ)
        for scenario, alternatives in (("s1", ["a1"]), ("s2", ["base", "a1"]), ("s_base", ["base"]), ("s_empty", [])):
            db.add_scenario(name=scenario)
            for rank, alternative in enumerate(alternatives, start=1):
                db.add_scenario_alternative(scenario_name=scenario, alternative_name=alternative, rank=rank)
        db.commit_session("seed data")


@pytest.fixture(scope="module")
def chain(tmp_path_factory) -> dict:
    """The seed (``"seed"``), migrated to 55, 56 and the current version (``55``, ``56``, ``"end"``)."""
    folder = tmp_path_factory.mktemp("v56_base")
    current = folder / "current.sqlite"
    _seed(current)
    out = {"seed": copy_db(current, folder / "seed.sqlite")}
    for k in (55, 56):
        migrate(current, up_to=k)
        out[k] = copy_db(current, folder / f"k{k}.sqlite")
    migrate(current)
    out["end"] = copy_db(current, folder / "end.sqlite")
    return out


def _ranks(path: Path) -> dict:
    with open_database_mapping(url_of(path)) as db:
        return {s: sorted((a["rank"], a["alternative_name"]) for a in db.get_items("scenario_alternative")
                          if a["scenario_name"] == s) for s in SCENARIOS}


def test_base_is_at_the_bottom_of_every_scenario(chain) -> None:
    assert version_of(chain["end"]) == FLEXTOOL_DB_VERSION
    assert _ranks(chain["end"]) == {
        "s1": [(-1, "Base"), (0, FLOOR), (1, "a1")],
        "s2": [(-1, "Base"), (0, FLOOR), (1, "base"), (2, "a1")],
        "s_base": [(-1, "Base"), (0, FLOOR), (1, "base")],
        "s_empty": [(1, "Base")],
    }


def test_base_holds_the_v56_values(chain) -> None:
    with open_database_mapping(url_of(chain["end"])) as db:
        in_base = {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"]):
                   v["parsed_value"] for v in db.get_items("parameter_value") if v["alternative_name"] == "Base"}
    assert in_base == {
        ("node__profile", ("n_in", "p1"), "profile_method"): "none",
        ("connection__profile", ("c1", "p1"), "profile_method"): "none",
        ("unit__node__profile", ("u1", "n_out", "p1"), "profile_method"): "none",
        ("unit__inputNode", ("u1", "n_in"), "ramp_method"): "none",
        ("unit__outputNode", ("u1", "n_out"), "ramp_method"): "none",
        ("reserve__upDown__unit__node", PROVISION_ACTIVE, "is_enabled"): "no",
        ("reserve__upDown__unit__node", PROVISION_INACTIVE, "is_enabled"): "no",
    }


def _resolved(path: Path, scenario: str) -> tuple[set, dict]:
    """The enabled reserve provisions and the ``profile_method`` per profile entity in *scenario*, as the engine's
    reader resolves them."""
    backend = SpineDBBackend(url_of(path), scenario)
    try:
        enabled = {tuple(e["entity_byname"]) for e in backend.find_entities(
            entity_class_name="reserve__upDown__unit__node")}
        methods = {(cls, tuple(v["entity_byname"])): v["parsed_value"]
                   for cls in ("node__profile", "unit__node__profile")
                   for v in backend.find_parameter_values(entity_class_name=cls,
                                                          parameter_definition_name="profile_method")}
    finally:
        backend.close()
    return enabled, methods


def test_each_scenario_keeps_the_older_behaviour_and_another_alternative_wins(chain, tmp_path) -> None:
    """Before v56 the provision without an entity_alternative row was inactive everywhere, the one active in ``a1``
    active where ``a1`` is; the profiles had no method.  A value written in ``a1`` after the upgrade outranks Base.
    Units and nodes are inactive by default and active in ``base`` and ``a1``, so ``s_empty`` shows no entity."""
    path = copy_db(chain["end"], tmp_path / "db.sqlite")
    with open_database_mapping(url_of(path)) as db:
        val, typ = to_database("upper_limit")
        db.add_parameter_value(entity_class_name="node__profile", entity_byname=("n_in", "p1"),
                               parameter_definition_name="profile_method", alternative_name="a1", value=val, type=typ)
        db.commit_session("a user value")
    expected = {
        "s1": ({PROVISION_ACTIVE}, "upper_limit"),
        "s2": ({PROVISION_ACTIVE}, "upper_limit"),
        "s_base": (set(), "none"),
    }
    for scenario, (provisions, node_profile_method) in expected.items():
        assert _resolved(path, scenario) == (provisions, {
            ("node__profile", ("n_in", "p1")): node_profile_method,
            ("unit__node__profile", ("u1", "n_out", "p1")): "none"}), scenario
    assert _resolved(path, "s_empty") == (set(), {})


def test_the_lower_case_base_is_left_as_it_was(chain) -> None:
    def base_alternative(path: Path) -> tuple:
        with open_database_mapping(url_of(path)) as db:
            alternative = db.get_item("alternative", name="base")
            ranks = sorted((a["scenario_name"], a["rank"]) for a in db.get_items("scenario_alternative")
                           if a["alternative_name"] == "base")
            values = sorted((v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
                             from_database(v["value"], v["type"])) for v in db.get_items("parameter_value")
                            if v["alternative_name"] == "base")
        return alternative["description"], ranks, values

    assert base_alternative(chain["end"]) == base_alternative(chain["seed"])
    assert base_alternative(chain["seed"])[1:] == ([("s2", 1), ("s_base", 1)], [("node", ("n_in",), "inflow", -10.0)])


def test_step_56_rerun_on_its_own_result_converges(chain, tmp_path) -> None:
    path = copy_db(chain[56], tmp_path / "db.sqlite")
    restamp_file(path, 55)
    migrate(path, up_to=56)
    assert version_of(path) == 56
    assert diff(snapshot(chain[56]), snapshot(path)) == []


def test_a_kill_after_a_base_write_converges(chain, tmp_path) -> None:
    """A kill right after the commit that makes ``Base`` (D.1), after D.2 and after Batch F, and right before D.1:
    the rerun of step 56 equals the clean step."""
    messages = commit_messages(chain[55], 56)
    d1 = next(j for j, m in enumerate(messages, start=1) if m.startswith("v56 D.1"))
    d2 = next(j for j, m in enumerate(messages, start=1) if m.startswith("v56 D.2"))
    f = next(j for j, m in enumerate(messages, start=1) if m.startswith("v56 Batch F"))
    assert 1 < d1 < d2 < f < len(messages)
    clean = {55: chain[55], 56: chain[56]}
    assert {j: kill_and_rerun(clean, 56, j, tmp_path) for j in (d1 - 1, d1, d2, f)} == {
        j: [] for j in (d1 - 1, d1, d2, f)}


@pytest.mark.nightly_heavy  # PR CI skips it; runs nightly (pyproject markers)
def test_every_kill_inside_step_56_converges(chain, tmp_path) -> None:
    clean = {55: chain[55], 56: chain[56]}
    commits = len(commit_messages(chain[55], 56))
    assert {j: kill_and_rerun(clean, 56, j, tmp_path) for j in range(1, commits)} == {
        j: [] for j in range(1, commits)}
