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
from spinedb_api import to_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import (_DEFINITION_RENAMES, _REUSED_DEFINITIONS,
                                                   BEHAVIOUR_CHANGE_MARKER_PREFIX)
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.initialize_database import initialize_database

from tests.update_flextool._rerun import (TEMPLATES, Kill, already_applied_records, commit_messages, copy_db, diff,
                                          import_v75_seed_data, kill_after_commit, kill_and_rerun, migrate, quiet,
                                          restamp_file, snapshot, url_of, version_of)


#: The steps whose rerun on their own result failed (an item looked up after the step removed or renamed it, or a
#: raw commit with nothing to commit) or changed values (v37 flipped the flipped coefficient back).
SELF_RERUN_STEPS = (35, 37, 38, 52, 57, 59, 61, 65)
#: The steps that detect their own effect (the others only tolerate an empty commit).
GUARDED_STEPS = (35, 37, 38, 52, 65)
#: (stamp, content) of a database migrated by FlexTool 4.6.0 or older that failed or was killed after content was
#: committed (the stamp was written only at the end): steps stamp+1..content run again on their own later results.
#: Each needs the later names of definitions (``_DEFINITION_RENAMES``), parameter group ``output`` (v43) and the
#: v62 list members (v51, v60), and v36 / v37 must not rescale or flip ``flow_coefficient`` again ((35, 37), (14, 45)).
#: Item 284: (31, 56).
LEGACY_CASES = ((22, 35), (42, 44), (43, 56), (55, 65), (50, 62), (31, 56), (23, FLEXTOOL_DB_VERSION), (35, 37),
                (14, 45), (34, 38))


def _seed_data(path: Path, *, base: bool) -> None:
    """A unit between two nodes, a connection, a group, a reserve, a profile and a commodity, their relations, and a
    few values in alternative ``a1`` (``coefficient`` 0.5 in, 2.0 out); with *base*, also a ``Base`` alternative."""
    with open_database_mapping(url_of(path)) as db:
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


#: The online seed (plan §5): per case unit ``{unit: (values {param: {alternative: value}}, scenarios {scenario:
#: [alternative, ...]} lowest rank first)}``.  Each unit is active only in its own alternatives (``unit`` is inactive
#: by default), so the unit-wide defaults below reach only the unit of a scenario.  uE comes at 80 (its curve's
#: first version).
ONLINE_UNITS = {
    # split times: A_S1 needs up (2), A_S2 up (the default 3) and down (3).
    "uA": ({"startup_method": {"A_base": "linear"}, "min_uptime": {"A_up": 2.0}, "min_downtime": {"A_down": 3.0}},
           {"A_S1": ["A_base", "A_up"], "A_S2": ["A_base", "A_down"]}),
    # online by the binary default in B_S1 (both times); B_off's no_startup keeps it off in B_S2.
    "uB": ({"min_downtime": {"B_base": 2.0}, "startup_method": {"B_off": "no_startup"}},
           {"B_S1": ["B_base"], "B_S2": ["B_base", "B_off"]}),
    # binary, both times; the method in C_scratch is in no scenario.
    "uC": ({"startup_method": {"C_base": "binary"}, "min_downtime": {"C_base": 2.0},
            "minimum_time_method": {"C_scratch": "min_uptime"}}, {"C_S": ["C_base"]}),
    # D_E's method resolves in D_S1 only.
    "uD": ({"startup_method": {"D_base": "linear"}, "min_downtime": {"D_base": 2.0},
            "minimum_time_method": {"D_E": "min_downtime"}}, {"D_S1": ["D_base", "D_E"], "D_S2": ["D_base"]}),
}
#: The unit-wide defaults of the online seed: a binary startup (online) and a min up time.  Set at 11: steps 1-11
#: import the pre-v12 templates, which give every definition its template default.
ONLINE_DEFAULTS = {("unit", "startup_method"): "binary", ("unit", "min_uptime"): 3.0,
                   ("connection", "startup_method"): "binary"}


def _online_unit(db, unit: str, values: dict, scenarios: dict) -> None:
    alternatives = sorted({a for alts in scenarios.values() for a in alts} | {
        a for by_alt in values.values() for a in by_alt})
    for alternative in alternatives:
        db.add_alternative(name=alternative)
    for scenario, alts in scenarios.items():
        db.add_scenario(name=scenario)
        for rank, alternative in enumerate(alts, start=1):
            db.add_scenario_alternative(scenario_name=scenario, alternative_name=alternative, rank=rank)
    db.add_entity(entity_class_name="unit", name=unit)
    db.add_entity(entity_class_name="unit__outputNode", element_name_list=(unit, "n_out"))
    for alternative in alternatives:
        db.add_entity_alternative(entity_class_name="unit", entity_byname=(unit,), alternative_name=alternative,
                                  active=True)
    for name, by_alt in values.items():
        for alternative, value in by_alt.items():
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name="unit", entity_byname=(unit,), parameter_definition_name=name,
                                   alternative_name=alternative, value=val, type=typ)


def _online_data(path: Path, version: int) -> None:
    """The online seed's data that comes at *version*: at 0 the units of :data:`ONLINE_UNITS` and connection cX
    (``binary`` in X_base); at 11 :data:`ONLINE_DEFAULTS`; at 80 uE (``no_startup``, ``piecewise_linear`` with a rising
    curve, in E_S); at 82 solve sol_roll (branches and ``rolling_window`` in P_base, no method) and sol_sep (branches
    and ``separate_dispatch``) in P_S."""
    from tests.update_flextool.test_v83_dispatch_realization_method import _branches
    from tests.update_flextool.test_v83_minimum_time_method import RISING

    with open_database_mapping(url_of(path)) as db:
        if version == 0:
            for unit, (values, scenarios) in ONLINE_UNITS.items():
                _online_unit(db, unit, values, scenarios)
            db.add_alternative(name="X_base")
            db.add_entity(entity_class_name="connection", name="cX")
            val, typ = to_database("binary")
            db.add_parameter_value(entity_class_name="connection", entity_byname=("cX",), alternative_name="X_base",
                                   parameter_definition_name="startup_method", value=val, type=typ)
        elif version == 11:
            for (cls, name), value in ONLINE_DEFAULTS.items():
                pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
                val, typ = to_database(value)
                db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
        elif version == 80:
            _online_unit(db, "uE", {"startup_method": {"E_base": "no_startup"},
                                    "conversion_method": {"E_base": "piecewise_linear"},
                                    "piecewise_linear_efficiency": {"E_base": RISING}}, {"E_S": ["E_base"]})
        elif version == 82:
            db.add_alternative(name="P_base")
            db.add_scenario(name="P_S")
            db.add_scenario_alternative(scenario_name="P_S", alternative_name="P_base", rank=1)
            for solve, extra in (("sol_roll", ("solve_mode", "rolling_window")),
                                 ("sol_sep", ("dispatch_realization_method", "separate_dispatch"))):
                db.add_entity(entity_class_name="solve", name=solve)
                for name, value in (("stochastic_branches", _branches()), extra):
                    val, typ = to_database(value)
                    db.add_parameter_value(entity_class_name="solve", entity_byname=(solve,),
                                           alternative_name="P_base", parameter_definition_name=name, value=val,
                                           type=typ)
        db.commit_session(f"online seed data at {version}")


#: The versions after whose step :func:`_online_data` adds its later parts.
ONLINE_STAGES = (11, 80, 82)


@pytest.fixture(scope="module")
def online_chain(tmp_path_factory) -> dict:
    """``{k: path}``: the v2 template with the seed data, ``Base`` and the online seed data (:func:`_online_data`,
    each later part added right after the step of its version), migrated one step at a time.  ``"step only"``:
    ``{k: path}`` of step k's own result at the versions where data comes after it (the oracle of a rerun of step
    k)."""
    folder = tmp_path_factory.mktemp("online_chain")
    current = folder / "current.sqlite"
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v2.json"), str(current))
    _seed_data(current, base=True)
    _online_data(current, 0)
    chain = {0: copy_db(current, folder / "k0.sqlite"), "step only": {}}
    with already_applied_records() as records:
        for k in range(1, FLEXTOOL_DB_VERSION + 1):
            migrate(current, up_to=k)
            if k in ONLINE_STAGES:
                chain["step only"][k] = copy_db(current, folder / f"k{k}_step.sqlite")
                _online_data(current, k)
            chain[k] = copy_db(current, folder / f"k{k}.sqlite")
    chain["records"] = records
    return chain


@pytest.fixture(scope="module")
def v25_one_shot(tmp_path_factory) -> dict:
    """``{k: path}`` for k 55, 56, 82, current: the v25 template with the seed data (no ``Base``) migrated to each
    in turn."""
    folder = tmp_path_factory.mktemp("v25_one_shot")
    current = folder / "current.sqlite"
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v25.json"), str(current))
    _seed_data(current, base=False)
    out = {}
    for k in (55, 56, FLEXTOOL_DB_VERSION - 1, FLEXTOOL_DB_VERSION):
        migrate(current, up_to=k)
        out[k] = copy_db(current, folder / f"k{k}.sqlite")
    return out


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
    import_v75_seed_data(current)
    with open_database_mapping(url_of(current)) as db:
        db.add_entity(entity_class_name="unit__node__profile", element_name_list=("thermal", "city", "wprof"))
        db.commit_session("a valueless unit__node__profile")
    chain = {75: copy_db(current, folder / "k75.sqlite")}
    with already_applied_records() as records:
        for k in range(76, FLEXTOOL_DB_VERSION + 1):
            migrate(current, up_to=k)
            chain[k] = copy_db(current, folder / f"k{k}.sqlite")
    chain["records"] = records
    return chain


#: The v56 backfills of the seed data in a database without ``Base`` (V5): the entities of :func:`_seed_data` that
#: have no value of the parameter in any alternative (no reserve provision: no ``is_enabled``).
V56_BASE_VALUES = {
    ("node__profile", ("n_in", "p1"), "profile_method", "Base"): "'none'",
    ("connection__profile", ("c1", "p1"), "profile_method", "Base"): "'none'",
    ("unit__node__profile", ("u1", "n_out", "p1"), "profile_method", "Base"): "'none'",
    ("unit__inputNode", ("u1", "n_in"), "ramp_method", "Base"): "'none'",
    ("unit__outputNode", ("u1", "n_out"), "ramp_method", "Base"): "'none'",
}


def test_a_clean_migration_finds_nothing_already_applied(v2_chain, v75_chain, online_chain, tmp_path) -> None:
    """The guards key on an effect a clean run does not have yet: the v2, v25, v75 and online chains log none.
    The v25 template has no ``Base``: v56 makes one for its backfills, below every scenario (it has none)."""
    assert v2_chain["records"] == []
    assert v75_chain["records"] == []
    assert online_chain["records"] == []
    path = tmp_path / "v25.sqlite"
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v25.json"), str(path))
    _seed_data(path, base=False)
    with already_applied_records() as records:
        migrate(path)
    assert version_of(path) == FLEXTOOL_DB_VERSION
    assert records == []
    after = snapshot(path)
    assert "Base" in after["alternatives"]
    assert {key: value for key, value in after["values"].items() if key[3] == "Base"} == V56_BASE_VALUES
    assert [row for row in after["scenario_alternatives"] if row[1] == "Base"] == []


#: The v83 lists every chain ends with (the members the parts append, ``binary`` removed by part 2).
V83_LISTS = {
    "minimum_time_methods": ("min_uptime", "min_downtime", "both", "none"),
    "startup_methods": ("no_startup", "linear", "linear_fix_in_uc", "integer", "integer_fix_in_uc"),
    "dispatch_realization_methods": ("within_solve", "separate_dispatch", "deterministic"),
}


@pytest.mark.parametrize("chain", ("v2_chain", "v25_one_shot", "v75_chain", "online_chain"))
def test_every_chain_ends_with_the_v83_lists(request, chain: str) -> None:
    after = snapshot(request.getfixturevalue(chain)[FLEXTOOL_DB_VERSION])
    assert {name: tuple(m for _index, m in sorted(after["list_values"][name])) for name in V83_LISTS} == {
        name: tuple(repr(m) for m in members) for name, members in V83_LISTS.items()}
    assert "solver" not in after["lists"]


@pytest.mark.parametrize("chain", ("v2_chain", "v25_one_shot", "v75_chain", "online_chain"))
def test_v83_rerun_on_its_own_result_converges(request, tmp_path, chain: str) -> None:
    """Content 83 under stamp 82 (a hand edit, or a kill after the v83 commit with its stamp lost) migrated to 83
    equals content 83, markers included."""
    clean = request.getfixturevalue(chain)
    path = copy_db(clean[FLEXTOOL_DB_VERSION], tmp_path / "db.sqlite")
    restamp_file(path, FLEXTOOL_DB_VERSION - 1)
    migrate(path)
    assert version_of(path) == FLEXTOOL_DB_VERSION
    assert diff(snapshot(clean[FLEXTOOL_DB_VERSION]), snapshot(path)) == []


def test_online_seed_kill_after_the_v83_commit_converges(online_chain, tmp_path) -> None:
    """The v83 step commits once (parts 0-5), then the version commit; a kill after the first converges."""
    assert len(commit_messages(online_chain[82], 83)) == 2
    assert kill_and_rerun(online_chain, FLEXTOOL_DB_VERSION, 1, tmp_path) == []


def test_online_seed_at_83(online_chain) -> None:
    """The v83 parts on the online seed (rules of ``plan_minimum_time_method``; D = binary, min_uptime default 3).
    uA: A_S1 needs up, A_S2 up and down → both into A_base (its startup winner).  uB: online by D in B_S1, both times
    → both into B_base (no startup value: the highest alternative with a value); B_S2 is off by B_off's no_startup
    and B_base would win → none into B_off.  uC: binary, both times; C_scratch is in no scenario → both into C_base.
    uD: D_S1 resolves by D_E; D_S2 needs both → both into D_base (D_E still wins in D_S1).  uE: online by its rising
    curve, the default up time → min_uptime into E_base.  u1 is off in every scenario.  Part 2: every binary (uC,
    cX and both defaults) is integer_fix_in_uc.  Part 5: sol_roll gets within_solve in P_base; part 3 lists it and
    the separate_dispatch sol_sep."""
    writes = {("uA", "A_base"): "both", ("uB", "B_base"): "both", ("uB", "B_off"): "none", ("uC", "C_base"): "both",
              ("uD", "D_base"): "both", ("uE", "E_base"): "min_uptime"}
    after = snapshot(online_chain[FLEXTOOL_DB_VERSION])
    before = snapshot(online_chain[FLEXTOOL_DB_VERSION - 1])
    methods = {(key[1][0], key[3]): value for key, value in after["values"].items()
               if key[0] == "unit" and key[2] == "minimum_time_method"}
    own = {(key[1][0], key[3]): value for key, value in before["values"].items()
           if key[0] == "unit" and key[2] == "minimum_time_method"}
    assert methods == own | {key: repr(value) for key, value in writes.items()}
    markers = {name.removeprefix(BEHAVIOUR_CHANGE_MARKER_PREFIX): value for name, value in after["metadata"]}
    assert json.loads(markers["minimum_time_method_written"]) == {
        "since": 83, "writes": [[u, a, m] for (u, a), m in sorted(writes.items())], "undecided": []}
    assert json.loads(markers["rolling_within_solve_one_solve"]) == {"since": 83, "solves": ["sol_roll"]}
    assert json.loads(markers["separate_dispatch_realization_looks_ahead"]) == {"since": 83, "solves": ["sol_sep"]}
    assert {key: value for key, value in after["values"].items() if key[2] == "startup_method"} == {
        key: ("'integer_fix_in_uc'" if value == "'binary'" else value)
        for key, value in before["values"].items() if key[2] == "startup_method"}
    for cls in ("unit", "connection"):
        assert after["definitions"][(cls, "startup_method")][1] == "'integer_fix_in_uc'"
    assert after["values"][("solve", ("sol_roll",), "dispatch_realization_method", "P_base")] == "'within_solve'"
    assert after["values"][("solve", ("sol_sep",), "dispatch_realization_method", "P_base")] == "'separate_dispatch'"


def test_v25_v56_rerun_on_its_own_result_converges(v25_one_shot, tmp_path) -> None:
    """V5 on the v25 seed (no ``Base`` before v56): content 56 under stamp 55 migrated to 56 equals content 56."""
    path = copy_db(v25_one_shot[56], tmp_path / "db.sqlite")
    restamp_file(path, 55)
    migrate(path, up_to=56)
    assert diff(snapshot(v25_one_shot[56]), snapshot(path)) == []


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


@pytest.mark.parametrize(("s", "k"), LEGACY_CASES)
def test_steps_rerun_on_later_content_converge(v2_chain, tmp_path, s: int, k: int) -> None:
    """Content k under stamp s migrated to k equals content k, and the steps say what they found applied."""
    path = copy_db(v2_chain[k], tmp_path / "db.sqlite")
    restamp_file(path, s)
    with already_applied_records() as records:
        migrate(path, up_to=k)
    assert version_of(path) == k
    assert diff(snapshot(v2_chain[k]), snapshot(path)) == []
    assert records


def test_item_284_pre_v32_database_killed_after_v56_converges(v2_chain, tmp_path) -> None:
    """A v31 database whose migration by FlexTool 4.6.0 or older was killed after v56 (content 56, stamp 31, the
    backup copy deleted) migrates to the current version and equals the clean migration."""
    path = copy_db(v2_chain[56], tmp_path / "db.sqlite")
    restamp_file(path, 31)
    migrate(path)
    assert version_of(path) == FLEXTOOL_DB_VERSION
    assert diff(snapshot(v2_chain[FLEXTOOL_DB_VERSION]), snapshot(path)) == []


def test_a_rerun_keeps_the_definitions_v23_brought_back(v2_chain, tmp_path) -> None:
    """v13 and v15 removed ``node.storate_state_end`` and ``node.invest_forced``, v23 typed them again: steps 15..34
    rerun on content 34 keep those definitions and their values."""
    expected = copy_db(v2_chain[34], tmp_path / "expected.sqlite")
    with open_database_mapping(url_of(expected)) as db:
        for name, value in (("invest_forced", 25.0), ("storate_state_end", 0.5)):
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name="node", entity_byname=("n_in",), parameter_definition_name=name,
                                   alternative_name="a1", value=val, type=typ)
        db.commit_session("values of the v23 definitions")
    for stamp in (12, 14):
        path = copy_db(expected, tmp_path / f"db{stamp}.sqlite")
        restamp_file(path, stamp)
        with already_applied_records() as records:
            migrate(path, up_to=34)
        assert diff(snapshot(expected), snapshot(path)) == []
        assert any("brought back" in r for r in records), records


def test_v52_rerun_after_its_first_commit_converges(v2_chain, tmp_path) -> None:
    """v52 commits its value lists (the ``solver`` list already renamed ``solvers``) before its definitions."""
    assert len(commit_messages(v2_chain[51], 52)) == 3
    assert kill_and_rerun(v2_chain, 52, 1, tmp_path) == []


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc to list the open files")
def test_the_helpers_and_a_killed_migration_release_the_file(v2_chain, tmp_path) -> None:
    """Nothing keeps the file open after a killed migration, a rerun or the helpers that read it (Windows cannot
    delete or replace an open file: ``with DatabaseMapping`` closes only the session, not the engine's pooled
    connection).  Checked on the open file descriptors, so it fails here as well as on Windows."""
    import gc
    import os

    def held(path: Path) -> list[str]:
        target = os.path.realpath(path)
        return [fd for fd in os.listdir("/proc/self/fd")
                if os.path.realpath(f"/proc/self/fd/{fd}") in (target, target + "-wal", target + "-shm")]

    path = copy_db(v2_chain[51], tmp_path / "db.sqlite")
    gc.disable()  # an engine left open must not be closed by a collection that happens to run
    try:
        with pytest.raises(Kill), kill_after_commit(1):
            migrate(path, up_to=52)
        assert held(path) == []
        for read in (version_of, snapshot, lambda p: restamp_file(p, 50), lambda p: migrate(p, up_to=52)):
            read(path)
            assert held(path) == [], read
    finally:
        gc.enable()


def test_a_kill_in_a_step_leaves_the_previous_steps_stamp(v2_chain, tmp_path) -> None:
    """Step 81's stamp rides on step 82's first commit: a kill right after it leaves the file at 81, not at 80."""
    step_81 = len(commit_messages(v2_chain[80], 81)) - 1
    path = copy_db(v2_chain[80], tmp_path / "db.sqlite")
    with pytest.raises(Kill), kill_after_commit(step_81 + 1):
        migrate(path, up_to=82)
    assert version_of(path) == 81
    migrate(path, up_to=82)
    assert diff(snapshot(v2_chain[82]), snapshot(path)) == []


@pytest.mark.nightly_heavy  # PR CI skips it; runs nightly (pyproject markers)
@pytest.mark.timeout(3600)
def test_every_kill_inside_a_step_converges(v2_chain, v25_chain, v75_chain, online_chain, tmp_path) -> None:
    """For every step k (12..current on the v2 chain, 76..current on the v75 and online chains, 56 (V5: the one
    that makes ``Base``) and current on the v25 chain) and every commit j of it: the run of step k on clean k-1
    killed right after commit j, migrated to k again, equals clean k."""
    last = FLEXTOOL_DB_VERSION + 1
    failures = []
    for name, chain, steps in (("v2", v2_chain, range(12, last)), ("v75", v75_chain, range(76, last)),
                               ("online", online_chain, range(76, last)), ("v25", v25_chain, (56, last - 1))):
        for k in steps:
            clean = {**chain, **chain.get("step only", {})} if k in chain.get("step only", {}) else chain
            for j in range(1, len(commit_messages(chain[k - 1], k))):
                result = kill_and_rerun(clean, k, j, tmp_path)
                if result:
                    failures.append((name, k, j, result[:3]))
    assert failures == []


#: The restamp grid (review m12): stamps s × contents k, s < k.
GRID_STAMPS = (11, 18, 24, 31, 37, 43, 50, 55, 60, 64, 70, 75)
GRID_CONTENTS = (38, 45, 56, 62, 65, 75, FLEXTOOL_DB_VERSION)


@pytest.fixture(scope="module")
def v25_chain(tmp_path_factory) -> dict:
    """``{k: path}`` for k 25..current: the v25 template with the seed data (no ``Base``), migrated one step at a
    time."""
    folder = tmp_path_factory.mktemp("v25_chain")
    current = folder / "current.sqlite"
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v25.json"), str(current))
    _seed_data(current, base=False)
    chain = {int(version_of(current)): copy_db(current, folder / "k_seed.sqlite")}
    for k in range(min(chain) + 1, FLEXTOOL_DB_VERSION + 1):
        migrate(current, up_to=k)
        chain[k] = copy_db(current, folder / f"k{k}.sqlite")
    return chain


@pytest.mark.nightly_heavy  # PR CI skips it; runs nightly (pyproject markers)
@pytest.mark.timeout(3600)
def test_every_restamp_of_the_grid_converges(v2_chain, v25_chain, v75_chain, online_chain, tmp_path) -> None:
    """Content k of the v2, v25, v75 and online chains under every grid stamp s < k (a hand edit, a lost or a legacy
    stamp), migrated to k, equals the clean content k."""
    failures = []
    cases = 0
    for name, chain in (("v2", v2_chain), ("v25", v25_chain), ("v75", v75_chain), ("online", online_chain)):
        for k in GRID_CONTENTS:
            if k not in chain:
                continue
            for s in GRID_STAMPS:
                if s >= k:
                    continue
                cases += 1
                path = copy_db(chain[k], tmp_path / f"{name}_{s}_{k}.sqlite")
                restamp_file(path, s)
                try:
                    migrate(path, up_to=k)
                except Exception as exc:  # noqa: BLE001 - collect, report all
                    failures.append((name, s, k, repr(exc)[:200]))
                    continue
                found = diff(snapshot(chain[k]), snapshot(path))
                if version_of(path) != k or found:
                    failures.append((name, s, k, version_of(path), found[:3]))
                path.unlink()
    assert cases > 100
    assert failures == []


@pytest.mark.nightly_heavy  # PR CI skips it; runs nightly (pyproject markers)
def test_the_rename_registry_matches_the_migration_chain(v2_chain) -> None:
    """Every rename of ``_DEFINITION_RENAMES`` happens at its version in the clean chain (the old name there before,
    gone after; the new name there after), and every definition name that is absent between two presences is in
    ``_REUSED_DEFINITIONS``."""
    present: dict[tuple[str, str], set[int]] = {}
    for k in range(FLEXTOOL_DB_VERSION + 1):
        with open_database_mapping(url_of(v2_chain[k])) as db:
            for d in db.get_items("parameter_definition"):
                present.setdefault((d["entity_class_name"], d["name"]), set()).add(k)
    wrong = [entry for entry in _DEFINITION_RENAMES
             if not ({entry[0] - 1} <= present.get(entry[1:3], set()) and entry[0] not in present[entry[1:3]]
                     and entry[0] in present.get(entry[3:5], set()))]
    assert wrong == []
    reused = sorted(key for key, versions in present.items()
                    if max(versions) - min(versions) + 1 != len(versions))
    assert reused == sorted(_REUSED_DEFINITIONS)


@pytest.mark.parametrize("s", (22, 37, 55))
def test_v75_database_restamped_below_its_entities_converges(v75_chain, tmp_path, s: int) -> None:
    """Item 285: content current with entities made at v75 (after the v38 floor and the v56 ``none`` backfills),
    stamp 22, 37 or 55, migrated to current equals the clean content: no floor, no backfill on those entities (and
    v23 does not bring back the flags v38 removed)."""
    k = FLEXTOOL_DB_VERSION
    path = copy_db(v75_chain[k], tmp_path / "db.sqlite")
    restamp_file(path, s)
    with already_applied_records() as records:
        migrate(path)
    assert version_of(path) == k
    assert diff(snapshot(v75_chain[k]), snapshot(path)) == []
    assert any(" v56 " in r and "default is" in r for r in records), records
    assert (s <= 37) == any(" v38 " in r and "has_storage are removed" in r for r in records), records


@pytest.mark.parametrize("s", (0, 5, 10))
def test_template_steps_skip_on_later_content(v2_chain, tmp_path, s: int) -> None:
    """Owner Q1: a stamp 0-10 over current content (a hand edit) does not re-import the old templates."""
    path = copy_db(v2_chain[FLEXTOOL_DB_VERSION], tmp_path / "db.sqlite")
    restamp_file(path, s)
    with already_applied_records() as records:
        migrate(path)
    assert diff(snapshot(v2_chain[FLEXTOOL_DB_VERSION]), snapshot(path)) == []
    assert sum("share_loss_of_load" in r for r in records) == 11 - s, records


def test_v56_backfills_a_user_set_upper_limit_default(v2_chain, tmp_path) -> None:
    """A ``unit__node__profile.profile_method`` default set to ``upper_limit`` by hand before v56 (no ``none`` member
    yet) is not taken for v56's own: the entity still gets its explicit ``none``."""
    path = copy_db(v2_chain[55], tmp_path / "db.sqlite")
    with open_database_mapping(url_of(path)) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="unit__node__profile", name="profile_method")
        val, typ = to_database("upper_limit")
        db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
        db.commit_session("user default")
    with already_applied_records() as records:
        migrate(path, up_to=56)
    assert records == []
    with open_database_mapping(url_of(path)) as db:
        values = [(v["alternative_name"], v["parsed_value"]) for v in db.find_parameter_values(
            entity_class_name="unit__node__profile", parameter_definition_name="profile_method")]
    assert values == [("Base", "none")]
