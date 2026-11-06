"""Storage start / end rules of database version 77.

* The default ``storage_start_end_method`` is ``fix_nothing``.  Before, a node without its own method got
  ``fix_start``: with a ``storage_state_start`` on some node it was pinned at 0 (its own value if it had one), with
  none anywhere it was not fixed.  A database upgraded from that default carries the marker
  ``storage_default_fix_nothing``; runs then name the nodes whose results change.
* ``fix_start`` / ``fix_end`` / ``fix_start_end`` without the node's own ``storage_state_start`` /
  ``storage_state_end`` stop the run with one error naming every such node.
* The upgrade rewrites explicit fix methods without their value to what they did (``fix_nothing``, or an added 0)
  where every scenario agrees and nothing else changes, and announces it.  A node that used the old default with
  its own start value gets ``fix_start`` written in that value's alternative (results unchanged).

Databases: ``tests/fixtures/tests.json`` (battery: bind_within_timeblock, fix_start, storage_state_start 0.1;
ev_battery: bind_within_timeblock, fix_nothing, no start value), edited as JSON and built under the pytest tmp dir.
A database "from before version 77" is the same JSON with ``model.version`` 76 and the old default ``fix_start``;
the migration then runs the version 77 step on it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database
from spinedb_api.exception import NothingToCommit

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db
from tests.engine._unit_db import b64, build, run

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tests.json"
CHAIN = ["init", "west", "coal", "wind", "battery", "ev"]
NOTICE = "Behaviour changed in"
PREFIX = "flextool_behaviour_change:"


def _data() -> dict:
    """tests.json without its scenarios (each test adds its own: the upgrade looks at every scenario)."""
    data = json.loads(FIXTURE.read_text())
    data["scenarios"] = []
    data["scenario_alternatives"] = []
    return data


def _drop(data: dict, ent: str, param: str) -> None:
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if not (pv[0] == "node" and pv[1] == ent and pv[2] == param)]


def _put(data: dict, ent: str, param: str, value, kind: str, alt: str) -> None:
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if not (pv[0] == "node" and pv[1] == ent and pv[2] == param and pv[4] == alt)]
    data["parameter_values"].append(["node", ent, param, b64(value, kind), alt])


def _alt(data: dict, alt: str) -> None:
    if all(a[0] != alt for a in data["alternatives"]):
        data["alternatives"].append([alt, ""])


def _scenario(data: dict, name: str, chain: list[str]) -> str:
    data["scenarios"].append([name, False, ""])
    for a, before in zip(chain, chain[1:] + [None]):
        data["scenario_alternatives"].append([name, a, before])
    return name


def _as_v76(data: dict) -> None:
    """The database as version 76 wrote it: model.version 76, default storage_start_end_method fix_start."""
    for pd in data["parameter_definitions"]:
        if pd[0] == "model" and pd[1] == "version":
            pd[2] = b64(76, "float")
        if pd[0] == "node" and pd[1] == "storage_start_end_method":
            pd[2] = b64("fix_start", "str")


def _db(tmp_path_factory, name: str, data: dict, up_to: int | None = None) -> str:
    work = tmp_path_factory.mktemp(name)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url, up_to=up_to)
    return url


def _mps(tmp_path_factory, step) -> str:
    path = tmp_path_factory.mktemp("mps") / "lp.mps"
    build(step.flex_data).write_mps(str(path), emit_names=True)
    return path.read_text()


def _first_lp(tmp_path_factory, url: str, scenario: str) -> str:
    return _mps(tmp_path_factory, next(iter(run(tmp_path_factory, url, scenario).values())))


def _markers(url: str) -> dict[str, str]:
    with DatabaseMapping(url) as db:
        return {md["name"]: md["value"] for md in db.get_items("metadata")}


def _notice_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith(NOTICE)]


# ---------------------------------------------------------------------------
# (a) a default node where another node has a start value: was pinned at 0, now free + notice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binding", ["bind_within_timeblock", "bind_forward_only"])
def test_default_node_beside_a_start_value_starts_free_with_a_notice(tmp_path_factory, capsys, binding):
    """ev_battery without its own method in an upgraded database, battery fix_start 0.1: before version 77 the
    default fix_start pinned ev_battery at 0 (battery's start value made p_state_start non-empty).  Now it starts
    free: its LP equals the twin where ev_battery says fix_nothing explicitly; the notice names ev_battery only
    (battery keeps its own method and value)."""
    upgraded = _data()
    _drop(upgraded, "ev_battery", "storage_start_end_method")
    _alt(upgraded, "bind")
    _put(upgraded, "ev_battery", "storage_binding_method", binding, "str", "bind")
    scen = _scenario(upgraded, "case", CHAIN + ["bind"])
    twin = json.loads(json.dumps(upgraded))
    _put(twin, "ev_battery", "storage_start_end_method", "fix_nothing", "str", "bind")
    _as_v76(upgraded)
    url = _db(tmp_path_factory, f"upgraded_{binding}", upgraded)
    assert PREFIX + "storage_default_fix_nothing" in _markers(url)
    capsys.readouterr()
    lp = _first_lp(tmp_path_factory, url, scen)
    lines = [ln for ln in _notice_lines(capsys.readouterr().out) if "storage_start_end_method" in ln
             and "default" in ln]
    assert len(lines) == 1
    affected = re.search(r"Affected storage nodes: ([^.]*)\.", lines[0]).group(1).split(", ")
    assert affected == ["ev_battery"]
    assert lp == _first_lp(tmp_path_factory, _db(tmp_path_factory, f"twin_{binding}", twin), scen)


# ---------------------------------------------------------------------------
# (b) no start value anywhere: the old default did nothing -> unchanged, no notice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binding", ["bind_within_timeblock", "bind_forward_only"])
def test_default_nodes_without_any_start_value_are_unchanged_and_silent(tmp_path_factory, capsys, binding):
    """battery and ev_battery without a method and no storage_state_start anywhere: the old default fix_start
    fixed nothing, so the upgraded database builds the same LP as the twin with fix_nothing on both, and no
    notice is printed although the marker is present."""
    upgraded = _data()
    for node in ("battery", "ev_battery"):
        _drop(upgraded, node, "storage_start_end_method")
    _drop(upgraded, "battery", "storage_state_start")
    _alt(upgraded, "bind")
    for node in ("battery", "ev_battery"):
        _put(upgraded, node, "storage_binding_method", binding, "str", "bind")
    scen = _scenario(upgraded, "case", CHAIN + ["bind"])
    twin = json.loads(json.dumps(upgraded))
    for node in ("battery", "ev_battery"):
        _put(twin, node, "storage_start_end_method", "fix_nothing", "str", "bind")
    _as_v76(upgraded)
    url = _db(tmp_path_factory, f"nostart_{binding}", upgraded)
    assert PREFIX + "storage_default_fix_nothing" in _markers(url)
    capsys.readouterr()
    lp = _first_lp(tmp_path_factory, url, scen)
    assert not [ln for ln in _notice_lines(capsys.readouterr().out) if "storage" in ln]
    assert lp == _first_lp(tmp_path_factory, _db(tmp_path_factory, f"nostart_twin_{binding}", twin), scen)


def test_new_database_with_default_nodes_prints_no_notice(tmp_path_factory, capsys):
    """A version 77 database (default fix_nothing, no marker) with a default node beside a start value: no
    notice, no error."""
    data = _data()
    _drop(data, "ev_battery", "storage_start_end_method")
    for pd in data["parameter_definitions"]:
        if pd[0] == "node" and pd[1] == "storage_start_end_method":
            pd[2] = b64("fix_nothing", "str")
    scen = _scenario(data, "case", CHAIN)
    url = _db(tmp_path_factory, "fresh", data)
    assert not _markers(url)
    capsys.readouterr()
    run(tmp_path_factory, url, scen)
    assert not _notice_lines(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# (c) a fix method without its value is an error naming every node
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "missing"), [
    ("fix_start", ["storage_state_start"]),
    ("fix_end", ["storage_state_end"]),
    ("fix_start_end", ["storage_state_start", "storage_state_end"]),
])
def test_fix_method_without_its_value_is_an_error(tmp_path_factory, method, missing):
    """ev_battery has no storage_state_start / storage_state_end: an explicit fix of either stops the run."""
    data = _data()
    _alt(data, "fix")
    _put(data, "ev_battery", "storage_start_end_method", method, "str", "fix")
    scen = _scenario(data, "case", CHAIN + ["fix"])
    url = _db(tmp_path_factory, f"err_{method}", data)
    with pytest.raises(FlexToolConfigError) as exc:
        run(tmp_path_factory, url, scen)
    text = str(exc.value)
    assert f"ev_battery ({method}, missing {' and '.join(missing)})" in text
    assert "battery (" not in text.replace("ev_battery (", "")
    assert "fix_nothing" in text


def test_error_names_every_node_at_once(tmp_path_factory):
    """battery fix_start without its start value and ev_battery fix_end without an end value: one error, both
    nodes."""
    data = _data()
    _drop(data, "battery", "storage_state_start")
    _alt(data, "fix")
    _put(data, "ev_battery", "storage_start_end_method", "fix_end", "str", "fix")
    scen = _scenario(data, "case", CHAIN + ["fix"])
    url = _db(tmp_path_factory, "err_two", data)
    with pytest.raises(FlexToolConfigError) as exc:
        run(tmp_path_factory, url, scen)
    text = str(exc.value)
    assert "battery (fix_start, missing storage_state_start)" in text
    assert "ev_battery (fix_end, missing storage_state_end)" in text


# ---------------------------------------------------------------------------
# (d) the upgrade rewrites explicit fix methods without their value
# ---------------------------------------------------------------------------


def test_upgrade_rewrites_a_fix_that_did_nothing(tmp_path_factory, capsys):
    """ev_battery fix_start (no value) in two scenarios without any start value: the fix did nothing, the
    upgrade makes it fix_nothing; the LP equals the twin that says fix_nothing, and the run names the rewrite."""
    upgraded = _data()
    _drop(upgraded, "battery", "storage_state_start")
    _drop(upgraded, "battery", "storage_start_end_method")
    _put(upgraded, "ev_battery", "storage_start_end_method", "fix_start", "str", "ev")
    _alt(upgraded, "other")
    scen = _scenario(upgraded, "case", CHAIN)
    _scenario(upgraded, "case_other", CHAIN + ["other"])
    twin = json.loads(json.dumps(upgraded))
    _put(twin, "ev_battery", "storage_start_end_method", "fix_nothing", "str", "ev")
    _as_v76(upgraded)
    capsys.readouterr()
    url = _db(tmp_path_factory, "rewrite_nothing", upgraded)
    assert "node ev_battery [ev]: fix_start -> fix_nothing" in capsys.readouterr().out
    marker = json.loads(_markers(url)[PREFIX + "storage_fix_method_rewritten"])
    assert marker["rewrites"] == [["ev_battery", "ev", "fix_start", "fix_nothing"]]
    lp = _first_lp(tmp_path_factory, url, scen)
    lines = [ln for ln in _notice_lines(capsys.readouterr().out) if "rewrote" in ln]
    assert len(lines) == 1 and "ev_battery (alternative ev: fix_start -> fix_nothing)" in lines[0]
    assert lp == _first_lp(tmp_path_factory, _db(tmp_path_factory, "rewrite_nothing_twin", twin), scen)


def test_upgrade_adds_a_zero_where_the_fix_pinned_at_zero(tmp_path_factory, capsys):
    """battery fix_start without its own value beside ev_battery's start value 0.5 (ev_battery fix_nothing) in
    both scenarios: the fix pinned battery at 0; the upgrade adds storage_state_start 0 in battery's alternative;
    the LP equals the twin with that value."""
    upgraded = _data()
    _drop(upgraded, "battery", "storage_state_start")
    _put(upgraded, "ev_battery", "storage_state_start", 0.5, "float", "ev")
    _alt(upgraded, "other")
    scen = _scenario(upgraded, "case", CHAIN)
    _scenario(upgraded, "case_other", CHAIN + ["other"])
    twin = json.loads(json.dumps(upgraded))
    _put(twin, "battery", "storage_state_start", 0.0, "float", "battery")
    _as_v76(upgraded)
    url = _db(tmp_path_factory, "rewrite_zero", upgraded)
    marker = json.loads(_markers(url)[PREFIX + "storage_fix_method_rewritten"])
    assert marker["rewrites"] == [["battery", "battery", "fix_start", "storage_state_start = 0"]]
    capsys.readouterr()
    lp = _first_lp(tmp_path_factory, url, scen)
    assert any("battery (alternative battery: fix_start -> storage_state_start = 0)" in ln
               for ln in _notice_lines(capsys.readouterr().out))
    assert lp == _first_lp(tmp_path_factory, _db(tmp_path_factory, "rewrite_zero_twin", twin), scen)


def test_upgrade_leaves_a_fix_scenarios_used_differently(tmp_path_factory):
    """ev_battery fix_start (no value): pinned at 0 in a scenario with battery's start value (in its own
    alternative), not fixed in one without it.  The upgrade leaves it; the run stops naming both scenarios."""
    upgraded = _data()
    start = [pv for pv in upgraded["parameter_values"]
             if pv[:3] == ["node", "battery", "storage_state_start"]]
    _drop(upgraded, "battery", "storage_state_start")
    _drop(upgraded, "battery", "storage_start_end_method")
    _alt(upgraded, "bstart")
    upgraded["parameter_values"].append(["node", "battery", "storage_state_start", start[0][3], "bstart"])
    _put(upgraded, "ev_battery", "storage_start_end_method", "fix_start", "str", "ev")
    _scenario(upgraded, "with_start", CHAIN + ["bstart"])
    scen = _scenario(upgraded, "without_start", CHAIN)
    _as_v76(upgraded)
    url = _db(tmp_path_factory, "rewrite_mixed", upgraded)
    marker = json.loads(_markers(url)[PREFIX + "storage_fix_method_rewritten"])
    # battery used the old default with its own start value: the upgrade writes that default explicitly.
    assert marker["rewrites"] == [["battery", "bstart", "default fix_start", "fix_start"]]
    assert marker["undecided"] == [["ev_battery", "ev", "fix_start", ["with_start", "without_start"]]]
    with pytest.raises(FlexToolConfigError) as exc:
        run(tmp_path_factory, url, scen)
    text = str(exc.value)
    assert "ev_battery (fix_start, missing storage_state_start)" in text
    assert "scenarios with_start, without_start used it differently" in text


def test_upgrade_writes_the_old_default_on_a_node_with_its_own_start(tmp_path_factory, capsys):
    """battery without its own method but with its storage_state_start 0.1 (alternative battery), ev_battery
    fix_nothing: the old default fix_start pinned battery at 0.1.  The upgrade writes fix_start on battery in
    alternative battery, so the LP equals the twin that says fix_start (as before the upgrade); the run names
    the rewrite, and no node is left for the default notice."""
    upgraded = _data()
    _drop(upgraded, "battery", "storage_start_end_method")
    _alt(upgraded, "other")
    scen = _scenario(upgraded, "case", CHAIN)
    _scenario(upgraded, "case_other", CHAIN + ["other"])
    twin = json.loads(json.dumps(upgraded))
    _put(twin, "battery", "storage_start_end_method", "fix_start", "str", "battery")
    _as_v76(upgraded)
    capsys.readouterr()
    url = _db(tmp_path_factory, "default_explicit", upgraded)
    assert "node battery [battery]: default fix_start -> fix_start" in capsys.readouterr().out
    marker = json.loads(_markers(url)[PREFIX + "storage_fix_method_rewritten"])
    assert marker["rewrites"] == [["battery", "battery", "default fix_start", "fix_start"]]
    lp = _first_lp(tmp_path_factory, url, scen)
    lines = _notice_lines(capsys.readouterr().out)
    assert any("battery (alternative battery: default fix_start -> fix_start)" in ln for ln in lines), lines
    assert not [ln for ln in lines if "fix_nothing keeps the new behaviour" in ln], lines
    assert lp == _first_lp(tmp_path_factory, _db(tmp_path_factory, "default_explicit_twin", twin), scen)


def test_storage_steps_rerun_change_nothing(tmp_path_factory):
    """After the upgrade, re-running the storage sub-steps finds nothing to commit."""
    upgraded = _data()
    _put(upgraded, "ev_battery", "storage_start_end_method", "fix_start", "str", "ev")
    _scenario(upgraded, "case", CHAIN)
    _as_v76(upgraded)
    url = _db(tmp_path_factory, "rerun", upgraded, up_to=77)  # the v77 step only: v78 changes the description
    with DatabaseMapping(url) as db:
        m._v77_storage_fix_rewrite(db)
        m._v77_storage_default_explicit(db)
        m._v77_storage_default(db)
        with pytest.raises(NothingToCommit):
            db.commit_session("re-run")


@pytest.mark.parametrize(("default", "marked"), [("fix_start", True), (None, False)])
def test_storage_default_marker_only_where_the_old_default_fixed(tmp_path_factory, default, marked):
    data = _data()
    _as_v76(data)
    for pd in data["parameter_definitions"]:
        if pd[0] == "node" and pd[1] == "storage_start_end_method":
            pd[2] = b64(default, "str") if default else [None, None]
    url = _db(tmp_path_factory, f"marker_{default}", data)
    assert (PREFIX + "storage_default_fix_nothing" in _markers(url)) is marked
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="node", name="storage_start_end_method")
        assert pdef["description"] == m._v78_new_description(
            "node", "storage_start_end_method", m._V77_STORAGE_START_END_DESCRIPTION)
        new_default = from_database(pdef["default_value"], pdef["default_type"]) if pdef["default_value"] else None
        assert new_default == ("fix_nothing" if marked else None)


# ---------------------------------------------------------------------------
# classify_storage_fix: the rule itself
# ---------------------------------------------------------------------------


def test_classify_keeps_a_fix_with_its_value():
    """(1) own value in every scenario where the method applies → nothing to do."""
    from flextool.update_flextool.db_migration import classify_storage_fix

    plan = classify_storage_fix({("n", "A"): "fix_start"}, {("n", "A"): 0.4}, {}, {"S1": ["A"], "S2": ["A", "B"]})
    assert plan.methods == () and plan.values == () and plan.undecided == ()


def test_classify_drops_a_fix_that_did_nothing():
    """(2) no node has a value in any scenario: fix_start → fix_nothing; fix_start_end keeps the end that has
    its value (→ fix_end)."""
    from flextool.update_flextool.db_migration import classify_storage_fix

    plan = classify_storage_fix({("n", "A"): "fix_start", ("k", "A"): "fix_start_end"}, {}, {("k", "A"): 0.2},
                                {"S1": ["A"], "S2": ["A", "B"]})
    assert set(plan.methods) == {("n", "A", "fix_start", "fix_nothing"), ("k", "A", "fix_start_end", "fix_end")}
    assert plan.values == () and plan.undecided == ()


def test_classify_adds_zero_where_the_fix_pinned_at_zero():
    """(3) another node has a start value in every scenario: n was pinned at 0 → storage_state_start 0 at A."""
    from flextool.update_flextool.db_migration import classify_storage_fix

    plan = classify_storage_fix({("n", "A"): "fix_start"}, {("m", "A"): 0.3}, {}, {"S1": ["A"], "S2": ["A", "B"]})
    assert plan.values == (("n", "A", "storage_state_start"),) and plan.methods == ()
    assert plan.records == (("n", "A", "fix_start", "storage_state_start = 0"),)


def test_classify_leaves_mixed_scenarios():
    """(4) pinned at 0 in S1 (m has a value there), not fixed in S2 → left, naming both scenarios."""
    from flextool.update_flextool.db_migration import classify_storage_fix

    plan = classify_storage_fix({("n", "A"): "fix_start"}, {("m", "B"): 0.3}, {}, {"S1": ["A", "B"], "S2": ["A"]})
    assert plan.methods == () and plan.values == ()
    assert plan.undecided == (("n", "A", "fix_start", ("S1", "S2")),)


def test_classify_drops_a_zero_that_would_leak_into_another_scenario():
    """n's fix_start at A has no value in S1 (m's value pins it at 0) → candidate 0 at A.  In S2, n's method comes
    from B (above A) and its start 0.5 from L (below A): a 0 at A would override 0.5 there, so the candidate is
    dropped (left, naming S1 and S2)."""
    from flextool.update_flextool.db_migration import classify_storage_fix

    plan = classify_storage_fix(
        {("n", "A"): "fix_start", ("n", "B"): "fix_start"},
        {("n", "L"): 0.5, ("m", "A"): 0.3}, {},
        {"S1": ["A"], "S2": ["L", "A", "B"]})
    assert plan.values == () and plan.methods == ()
    assert plan.undecided == (("n", "A", "fix_start", ("S1", "S2")),)


def test_classify_ignores_values_of_nodes_switched_off():
    """m is inactive in S2: its start value does not count there, so n's fix did nothing in S2 but pinned at 0
    in S1 → left."""
    from flextool.update_flextool.db_migration import classify_storage_fix

    plan = classify_storage_fix({("n", "A"): "fix_start"}, {("m", "A"): 0.3}, {},
                                {"S1": ["A"], "S2": ["A"]}, inactive=frozenset({("S2", "m")}))
    assert plan.undecided == (("n", "A", "fix_start", ("S1", "S2")),)


def test_classify_default_writes_the_old_default_where_the_node_has_its_own_start():
    """n has no stored method and its start 0.4 at A in S1 and S2: fix_start is written at A.  k has no value:
    nothing is written for it (it starts free now; the run names it)."""
    from flextool.update_flextool.db_migration import classify_storage_default

    writes = classify_storage_default({}, {("n", "A"): 0.4}, {}, {"S1": ["A"], "S2": ["A", "B"]}, "fix_start")
    assert writes == (("n", "A"),)


def test_classify_default_never_overrides_a_lower_stored_method():
    """n relies on the default in S1 (start 0.4 at A).  In S2 its method fix_nothing comes from L, below A: a
    fix_start at A would pin it there, so nothing is written for n."""
    from flextool.update_flextool.db_migration import classify_storage_default

    writes = classify_storage_default({("n", "L"): "fix_nothing"}, {("n", "A"): 0.4}, {},
                                      {"S1": ["A"], "S2": ["L", "A"]}, "fix_start")
    assert writes == ()


def test_database_without_scenarios_is_one_scenario_of_all_alternatives(tmp_path_factory):
    data = _data()
    work = tmp_path_factory.mktemp("noscen")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    with DatabaseMapping(url) as db:
        _methods, _starts, _ends, scenarios, _inactive = m._v77_storage_inputs(db)
        alternatives = [a["name"] for a in db.get_items("alternative")]
    assert scenarios == {m._V77_NO_SCENARIO: alternatives}
