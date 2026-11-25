"""The value-list repair tool's suspects (W-0490 package P2): which references, their evidence and suggestion,
their context and similarity groups.

Every database is a made-up copy of the schema database (``_value_list_cases.make_hit``).  The oracle is the
construction: R is the schema's ``not_allowed`` row, now reading ``cumulative_limits``; the unit, connection and
node ``invest_method`` defaults point to R; the group one is ``not_active`` in ``group_invest_methods`` (v82) unless
``pre_v77_group`` binds it to ``invest_methods`` with the default R.  "before" values (written before the session)
meant the member they named; "after" values were written while R read ``cumulative_limits``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from tests.update_flextool._value_list_cases import LEGACY, add_commit, make_hit, member_id, sql, write_values

REASON = ("Recommended: FlexTool's default. A method parameter default that reads cumulative_limits can lead to "
          "unwanted behaviour (here: entities that rely on it invest and retire up to 1e6 MW).")
LATER = "2025-01-10"
_DEFAULT = ("UPDATE parameter_definition SET default_value = ?, default_type = ? WHERE name = 'invest_method' AND "
            "entity_class_id = (SELECT id FROM entity_class WHERE name = ?)")


def _suspects(path: Path, tmp_path: Path) -> tuple[vr.Analysis, tuple[vs.Suspect, ...]]:
    folder = tmp_path / "snap"
    folder.mkdir(parents=True, exist_ok=True)
    a = vr.analyse(vr.snapshot(path, folder))
    before = hashlib.sha256(a.path.read_bytes()).hexdigest()
    found = vs.suspects(a)
    assert hashlib.sha256(a.path.read_bytes()).hexdigest() == before  # read-only
    return a, found


def _by_entity(found) -> dict[str, vs.Suspect]:
    return {s.entity: s for s in found if s.kind != vs.DEFAULT}


def _list_id(path: Path, name: str) -> int:
    return sql(path, "SELECT id FROM parameter_value_list WHERE name = ?", (name,))[0][0]


def _add_numbers(path: Path, items) -> None:
    """``(class, entity, parameter, alternative, number)`` values, written with spinedb_api."""
    from spinedb_api import to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{path}") as db:
        for cls, entity, param, alternative, number in items:
            if not db.get_items("entity", entity_class_name=cls, name=entity):
                db.add_entity(entity_class_name=cls, name=entity)
            value, typ = to_database(number)
            db.add_parameter_value(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=param,
                                   alternative_name=alternative, value=value, type=typ)
        db.commit_session("Updated")


def test_defaults_only_hit_gives_four_default_suspects(tmp_path, schema_db_url) -> None:
    """Design F2 shape: the four defaults point to R, no value does.  Each is recommended back to FlexTool's
    default with the reason."""
    case = make_hit(tmp_path, schema_db_url, pre_v77_group=True)
    _, found = _suspects(case.path, tmp_path)
    assert sorted((s.kind, s.cls, s.current, s.suggestion, s.label, s.list_id) for s in found) == [
        ("default", cls, "cumulative_limits", "set:not_allowed", vs.SHIPPED, case.list_id)
        for cls in ("connection", "group", "node", "unit")]
    assert all(s.reason == REASON and s.item.startswith("def:") and not s.certain for s in found)


def test_null_and_text_defaults(tmp_path, schema_db_url) -> None:
    """Lost defaults are NULL (the realistic shape): not part A's (the bug did not make them read
    ``cumulative_limits``) but part B's, "no sign of the bug" (``make_hit`` has no FlexTool 3.x v22 fingerprint
    commit), recommended ``not_allowed``.  A text-form default is turned into a reference to R by the item-279
    pre-pass and is then a part A suspect like any reference to R.  The group default (``not_active`` in its own
    list) is FlexTool's: no suspect."""
    from flextool.update_flextool import list_default_restore as ldr

    case = make_hit(tmp_path, schema_db_url)
    for cls in ("connection", "node"):
        sql(case.path, _DEFAULT, (None, None, cls))
    sql(case.path, _DEFAULT, (b'"cumulative_limits"', "list_value_ref", "unit"))
    a, found = _suspects(case.path, tmp_path)
    assert a.pre_pass.repaired == 1
    assert [(s.cls, s.current, s.suggestion, s.keep_text, s.reason) for s in found] == [
        ("unit", "cumulative_limits", "set:not_allowed", "keep cumulative_limits", REASON)]
    assert sorted((s.cls, s.label, s.suggestion, s.part_b) for s in vr.all_suspects(a) if s.part_b) == [
        ("connection", ldr.NO_SIGN, "set:not_allowed", True), ("node", ldr.NO_SIGN, "set:not_allowed", True)]


def test_shipped_default_is_not_a_suspect_and_legacy_values_get_no_suggestion(tmp_path, schema_db_url) -> None:
    """Legacy-member hit: the defaults point to the ``not_allowed`` row (index 14), so they are FlexTool's; a
    default set to another member (``invest_total``) is the user's choice: no suspect; a default on R (it read the
    legacy member before the session, ``cumulative_limits`` now) is a suspect; a value written before the session
    read the legacy member: no suggestion."""
    case = make_hit(tmp_path, schema_db_url, legacy=True, before=[("unit", "U1", "base", LEGACY)])
    other = member_id(case.path, "invest_methods", "invest_total")
    sql(case.path, _DEFAULT, (str(other).encode(), "list_value_ref", "connection"))
    sql(case.path, _DEFAULT, (str(case.r_id).encode(), "list_value_ref", "node"))
    _, found = _suspects(case.path, tmp_path)
    assert [(s.kind, s.cls, s.current, s.suggestion) for s in found] == [
        ("default", "node", "cumulative_limits", "set:not_allowed"),
        ("value", "unit", "cumulative_limits", None)]
    assert found[1].label == vs.LEGACY


def test_values_before_c_are_certain_only_in_hit(tmp_path, schema_db_url) -> None:
    before = [("unit", "U1", "base", "not_allowed"), ("unit", "U3", "base", "not_allowed")]
    after = [("unit", "U2", "base", "cumulative_limits")]
    case = make_hit(tmp_path, schema_db_url, before=before, after=after)
    sql(case.path, "UPDATE parameter_value SET commit_id = NULL WHERE entity_id = (SELECT id FROM entity WHERE "
                   "name = 'U3')")  # from before commit tracking: older than C
    _, found = _suspects(case.path, tmp_path)
    values = _by_entity(found)
    assert {e: (s.label, s.suggestion, s.current, s.list_id) for e, s in values.items()} == {
        "U1": (vs.CERTAIN, "set:not_allowed", "cumulative_limits", case.list_id),
        "U3": (vs.CERTAIN, "set:not_allowed", "cumulative_limits", case.list_id),
        "U2": (vs.AFTER, None, "cumulative_limits", case.list_id)}
    assert values["U1"].certain and not values["U2"].certain

    retried = make_hit(tmp_path, schema_db_url, name="retried.sqlite", before=before[:1], after=after, retried=True)
    _, found = _suspects(retried.path, tmp_path / "r")
    values = _by_entity(found)
    assert {e: (s.label, s.suggestion) for e, s in values.items()} == {
        "U1": ("probably not_allowed", None), "U2": (vs.AFTER, None)}
    assert not any(s.certain for s in found)


def test_renamed_back_asks_only_about_values_between_c_and_h(tmp_path, schema_db_url) -> None:
    """R renamed back to ``not_allowed`` at H.  U1 (before C) and U3 (after H) read what they meant; U2 was saved
    while R read ``cumulative_limits``.  The defaults read ``not_allowed`` again: shipped, no suspect."""
    case = make_hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")],
                    after=[("unit", "U2", "base", "cumulative_limits")])
    h = add_commit(case.path, "Updated", LATER)
    sql(case.path, "UPDATE list_value SET value = ?, commit_id = ? WHERE id = ?", (b'"not_allowed"', h, case.r_id))
    write_values(case.path, [("unit", "U3", "base", "not_allowed")])
    a, found = _suspects(case.path, tmp_path)
    assert a.state == vr.RENAMED_BACK
    assert [(s.kind, s.entity, s.label, s.current, s.suggestion, s.keep_text) for s in found] == [
        ("value", "U2", vs.RENAMED, "not_allowed", None, "keep (stays not_allowed, the current text)")]


def test_carried_v77_group_values(tmp_path, schema_db_url) -> None:
    """A group value v77 rewrote by name to ``cumulative_limits`` (commit = the v77 step's) is a carried suspect in
    its own list; one the user wrote later is not.  An empty group default is part B's (recommended
    ``not_active``), not part A's."""
    case = make_hit(tmp_path, schema_db_url)
    write_values(case.path, [("group", "G1", "base", "cumulative_limits")])
    v77 = add_commit(case.path, "v77: schema batch", LATER)
    sql(case.path, "UPDATE parameter_value SET commit_id = ? WHERE entity_id = (SELECT id FROM entity WHERE "
                   "name = 'G1')", (v77,))
    write_values(case.path, [("group", "G2", "base", "cumulative_limits")])
    sql(case.path, _DEFAULT, (None, None, "group"))
    a, found = _suspects(case.path, tmp_path)
    group_list = _list_id(case.path, "group_invest_methods")
    group = [s for s in found if s.cls == "group"]
    assert [(s.kind, s.entity, s.list_id, s.current, s.label, s.suggestion) for s in group] == [
        ("carried", "G1", group_list, "cumulative_limits", vs.CARRIED_LABEL, None)]
    assert [(s.kind, s.list_id, s.suggestion, s.part_b) for s in vr.all_suspects(a) if s.cls == "group"
            and s.kind == vs.DEFAULT] == [("default", group_list, "set:not_active", True)]


def test_context_and_similarity_groups(tmp_path, schema_db_url) -> None:
    """U2 and U3 (no parameters, alternative ``base`` used by scenario ``s1``) are one group; U4 in ``spare`` (no
    scenario) is inert, its own group; U5 has an ``invest_cost``, its own group.  The unit default is relied on by
    R1 (``invest_cost``), R2 (a cumulative cap) and U4 (its own value is only in ``spare``, which ``s1`` does not
    use)."""
    case = make_hit(tmp_path, schema_db_url)
    sql(case.path, "INSERT INTO alternative (name, description) VALUES ('spare', '')")
    write_values(case.path, [("unit", "U2", "base", "cumulative_limits"), ("unit", "U3", "base", "cumulative_limits"),
                             ("unit", "U4", "spare", "cumulative_limits"), ("unit", "U5", "base", "cumulative_limits")])
    _add_numbers(case.path, [("unit", "U5", "invest_cost", "base", 100.0), ("unit", "R1", "invest_cost", "base", 50.0),
                             ("unit", "R2", "cumulative_max_capacity", "base", 10.0),
                             ("unit", "U3", "invest_max_total", "spare", 5.0)])
    sql(case.path, "INSERT INTO scenario (name, description, active) VALUES ('s1', '', 1)")
    sql(case.path, "INSERT INTO scenario_alternative (scenario_id, alternative_id, rank) VALUES ((SELECT id FROM "
                   "scenario WHERE name = 's1'), (SELECT id FROM alternative WHERE name = 'base'), 1)")
    _, found = _suspects(case.path, tmp_path)
    values = _by_entity(found)
    unit = next(s for s in found if s.kind == vs.DEFAULT and s.cls == "unit")
    assert (unit.context.relying, unit.context.relying_cost, unit.context.relying_cap) == (3, 1, 1)
    assert unit.context.relying_names == (("R1", False), ("R2", True), ("U4", False))
    assert unit.context.text().startswith("3 entities rely on it: 1 with invest_cost, 1 with a cumulative cap "
                                          "(without one: invests and retires up to 1e6 MW)")
    assert values["U5"].context.text().startswith(
        "invest_cost yes, no cumulative cap: invests and retires up to 1e6 MW; scenarios: s1; written ")
    assert values["U3"].context.limits == ("invest_max_total",)
    assert "scenarios: none (inert)" in values["U4"].context.text()
    value_groups = [tuple(s.entity for s in g.suspects) for g in vs.groups(found) if g.suspects[0].kind == vs.VALUE]
    assert sorted(value_groups) == [("U2", "U3"), ("U4",), ("U5",)]
    assert len(vs.groups(found)) == 3 + 3  # unit, connection and node defaults: one group each


def test_no_suspects_without_a_hit(tmp_path, schema_db_url) -> None:
    case = make_hit(tmp_path, schema_db_url, overwrite=False, after=[("unit", "U2", "base", "cumulative_limits")])
    a, found = _suspects(case.path, tmp_path)
    assert (a.state, found) == (vr.NOT_HIT, ())
