"""A migration step that removes many rows commits them in batches (``_remove_items_bounded``).  When it stops after
a batch (a crash or a kill, which the restore of :func:`migrate_database` cannot cover), a rerun of the step on the
same file must give the database a clean run gives.

Each case builds its pre-step database from data (``import_data``, never a checked-in ``.sqlite``), lowers the batch
bound to 2 id runs so that a handful of scattered rows commit in several batches, stops the step right after its
first (or last) batch commit and reruns it.  The clean run is checked against hand-derived expectations, and the rerun against
the clean run.  Last, :func:`migrate_database` restores the file when a step fails.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database, import_data

from flextool.update_flextool import db_migration as m

N = 8


class _Stop(RuntimeError):
    """The interruption: raised right after a batch commit."""


def _import(url: str, **data) -> None:
    with DatabaseMapping(url, create=True) as db:
        _, errors = import_data(db, **data)
        assert not errors, errors[:3]
        db.commit_session("pre-step data")


def _snapshot(url: str) -> dict:
    with DatabaseMapping(url) as db:
        return {
            "values": {(pv["entity_class_name"], tuple(pv["entity_byname"]), pv["parameter_definition_name"],
                        pv["alternative_name"]): from_database(pv["value"], pv["type"])
                       for pv in db.get_items("parameter_value")},
            "definitions": {(d["entity_class_name"], d["name"]): (d["parameter_value_list_name"],
                                                                  from_database(d["default_value"], d["default_type"]))
                            for d in db.get_items("parameter_definition")},
            "entity_alternatives": {(ea["entity_class_name"], tuple(ea["entity_byname"]), ea["alternative_name"]):
                                    ea["active"] for ea in db.get_items("entity_alternative")},
            "classes": {c["name"]: c["active_by_default"] for c in db.get_items("entity_class")},
            "lists": sorted(x["name"] for x in db.get_items("parameter_value_list")),
            "entities": sorted((e["entity_class_name"], tuple(e["entity_byname"])) for e in db.get_items("entity")),
        }


def _run(url: str, step) -> None:
    with DatabaseMapping(url) as db:
        step(db)
        m._commit_step(db, "the step")


def _clean_and_rerun(tmp_path: Path, build, step, monkeypatch, *, trigger: str = "",
                     last: bool = False) -> tuple[dict, dict, list]:
    """``(clean, rerun, batch commits)``: *step* run once on one database, and on another one stopped after its first
    (``last``: its last) batch commit whose message has *trigger*, then run again."""
    clean, partial = (f"sqlite:///{tmp_path / name}" for name in ("clean.sqlite", "partial.sqlite"))
    build(clean)
    build(partial)
    monkeypatch.setattr(m, "_MAX_REMOVED_RANGES", 2)
    _run(clean, step)
    real_commit = m._commit_step
    batches: list[str] = []

    def stop_after_first_batch(db, message):
        real_commit(db, message)
        batch = re.search(r"batch (\d+)/(\d+)\)", message)
        if batch and trigger in message and batch[1] == (batch[2] if last else "1"):
            batches.append(message)
            raise _Stop(message)

    monkeypatch.setattr(m, "_commit_step", stop_after_first_batch)
    with pytest.raises(_Stop):
        _run(partial, step)
    monkeypatch.setattr(m, "_commit_step", real_commit)
    _run(partial, step)
    return _snapshot(clean), _snapshot(partial), batches


# ── v56: the yes-only lists become yes_no ────────────────────────────────────────────────────────────────────────

def _build_v56_yes_no(url: str) -> None:
    """``N`` connections with ``is_DC = yes`` (list ``is_DC``), each followed by another value, so the is_DC ids are
    scattered."""
    values = []
    for i in range(N):
        values += [["connection", f"c{i}", "is_DC", "yes", "Base"], ["connection", f"c{i}", "other", 1.0, "Base"]]
    _import(url,
            entity_classes=[["connection", ()], ["group", ()], ["node", ()], ["unit", ()],
                            ["unit__inputNode", ("unit", "node")], ["unit__outputNode", ("unit", "node")]],
            parameter_value_lists=[["yes_no", "yes"], ["yes_no", "no"], ["is_DC", "yes"],
                                   ["has_capacity_margin", "yes"], ["has_inertia", "yes"],
                                   ["has_non_synchronous", "yes"], ["is_non_synchronous", "yes"]],
            parameter_definitions=[["connection", "is_DC", None, "is_DC", ""], ["connection", "other", None, None, ""],
                                   ["group", "has_capacity_margin", None, "has_capacity_margin", ""],
                                   ["group", "has_inertia", None, "has_inertia", ""],
                                   ["group", "has_non_synchronous", None, "has_non_synchronous", ""],
                                   ["unit__inputNode", "is_non_synchronous", None, "is_non_synchronous", ""],
                                   ["unit__outputNode", "is_non_synchronous", None, "is_non_synchronous", ""]],
            alternatives=[["Base", ""]], entities=[["connection", f"c{i}"] for i in range(N)],
            parameter_values=values)


@pytest.mark.parametrize("last", [False, True])
def test_v56_yes_no_rebind_rerun_keeps_every_value(tmp_path, monkeypatch, last: bool) -> None:
    """Every ``is_DC = yes`` survives the rebind to ``yes_no`` (default ``no``), also when the step stopped after
    removing the first or the last batch of them: the values wait in a stage definition (written in one run of ids,
    so it is removed in one commit), which is gone after the step."""
    clean, rerun, batches = _clean_and_rerun(tmp_path, _build_v56_yes_no, m._migrate_v56_retype_yes_only_to_yes_no,
                                             monkeypatch, trigger="is_DC values", last=last)
    assert batches
    assert {k: v for k, v in clean["values"].items() if k[2] == "is_DC"} == {
        ("connection", (f"c{i}",), "is_DC", "Base"): "yes" for i in range(N)}
    assert clean["definitions"][("connection", "is_DC")] == ("yes_no", "no")
    assert "is_DC" not in clean["lists"]
    assert rerun == clean
    assert not any(name.endswith("_stage") for _, name in clean["definitions"])


# ── v56: entity_alternative activity becomes is_enabled ──────────────────────────────────────────────────────────

def _build_v56_is_enabled(url: str) -> None:
    """``N`` reserve provisions, the even ones active in Base by an entity_alternative row, the odd ones without one
    (inactive before v56), and ``N`` constraints active in Base; the rows of the two classes alternate, so the ids of
    each class are scattered."""
    units = [f"u{i}" for i in range(N)]
    entity_alternatives = []
    for i, u in enumerate(units):
        if i % 2 == 0:
            entity_alternatives.append(["reserve__upDown__unit__node", ("r", "up", u, "n"), "Base", True])
        entity_alternatives.append(["constraint", f"k{i}", "Base", True])
    _import(url,
            entity_classes=[["reserve", ()], ["upDown", ()], ["unit", ()], ["node", ()], ["connection", ()],
                            ["constraint", ()], ["group", ()],
                            ["reserve__upDown__unit__node", ("reserve", "upDown", "unit", "node"), None, None, False],
                            ["reserve__upDown__connection__node", ("reserve", "upDown", "connection", "node"), None,
                             None, False]],
            parameter_value_lists=[["yes_no", "yes"], ["yes_no", "no"], ["is_active", "yes"]],
            parameter_groups=[["constraint", "ff0000", 1], ["reserve", "00ff00", 2]],
            alternatives=[["Base", ""]],
            entities=[["reserve", "r"], ["upDown", "up"], ["node", "n"]] + [["unit", u] for u in units]
            + [["reserve__upDown__unit__node", ("r", "up", u, "n")] for u in units]
            + [["constraint", f"k{i}"] for i in range(N)],
            entity_alternatives=entity_alternatives)


def _expected_is_enabled() -> dict:
    """Active in Base -> yes; a reserve provision without a Base row (inactive before v56) -> no."""
    out = {("reserve__upDown__unit__node", ("r", "up", f"u{i}", "n"), "is_enabled", "Base"):
           "yes" if i % 2 == 0 else "no" for i in range(N)}
    out.update({("constraint", (f"k{i}",), "is_enabled", "Base"): "yes" for i in range(N)})
    return out


@pytest.mark.parametrize("last", [False, True])
def test_v56_is_enabled_rerun_keeps_the_written_values(tmp_path, monkeypatch, last: bool) -> None:
    """A reserve provision active in Base is ``is_enabled = yes`` after the step, also when the step stopped after
    removing the first or the last batch of entity_alternative rows: the rerun does not take a provision whose row is gone for
    one without a row."""
    clean, rerun, batches = _clean_and_rerun(tmp_path, _build_v56_is_enabled,
                                             m._migrate_v56_reactivate_is_enabled_parameter, monkeypatch,
                                             trigger="reserve__upDown__unit__node", last=last)
    assert batches
    assert clean["values"] == _expected_is_enabled()
    assert clean["entity_alternatives"] == {}
    assert clean["classes"]["reserve__upDown__unit__node"] is True
    assert rerun == clean


def test_v56_is_enabled_rerun_after_the_whole_step(tmp_path) -> None:
    """A second run of the whole committed step (the chain failed at a later step, before the version stamp)
    changes nothing: the provisions active in Base stay ``yes``."""
    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    _build_v56_is_enabled(url)
    _run(url, m._migrate_v56_reactivate_is_enabled_parameter)
    once = _snapshot(url)
    _run(url, m._migrate_v56_reactivate_is_enabled_parameter)
    assert once["values"] == _expected_is_enabled()
    assert _snapshot(url) == once


# ── v70: output capacity_min_coeff, v36 leftovers ────────────────────────────────────────────────────────────────

def _build_v69_coefficients(url: str) -> None:
    """``N`` units with an input ``capacity_min_coeff`` 0.3 and an output ``capacity_max_coeff`` 0.8; the output
    ``capacity_min_coeff`` is 0.8 (a v36 leftover, equal to the maximum) on the even units and 2.0 (out of [0, 1])
    on the odd ones, so every output minimum goes and their ids are scattered."""
    units = [f"u{i}" for i in range(N)]
    values = []
    for i, u in enumerate(units):
        values += [["unit__inputNode", (u, "fuel"), "capacity_min_coeff", 0.3, "Base"],
                   ["unit__outputNode", (u, "elec"), "capacity_max_coeff", 0.8, "Base"],
                   ["unit__outputNode", (u, "elec"), "capacity_min_coeff", 0.8 if i % 2 == 0 else 2.0, "Base"]]
    _import(url,
            entity_classes=[["model", ()], ["node", ()], ["unit", ()], ["unit__inputNode", ("unit", "node")],
                            ["unit__outputNode", ("unit", "node")], ["reserve__upDown__group", ()]],
            parameter_definitions=[["unit__inputNode", "capacity_max_coeff", 1.0, None, ""],
                                   ["unit__inputNode", "capacity_min_coeff", 1.0, None, ""],
                                   ["unit__outputNode", "capacity_max_coeff", 1.0, None, ""],
                                   ["unit__outputNode", "capacity_min_coeff", 1.0, None, ""],
                                   ["model", "max_flow_for_unconstrained_variables", None, None, ""]],
            alternatives=[["Base", ""]],
            entities=[["node", "fuel"], ["node", "elec"]] + [["unit", u] for u in units]
            + [["unit__inputNode", (u, "fuel")] for u in units] + [["unit__outputNode", (u, "elec")] for u in units],
            parameter_values=values)


@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("trigger", ["unit__outputNode", "unit__inputNode"])
def test_v70_rerun_removes_every_leftover(tmp_path, monkeypatch, trigger: str, last: bool) -> None:
    """No output minimum is left after the step (the leftovers equal the maximum, the others are out of range) and
    no input minimum (its meaning changed), also when the step stopped after the first or last batch of either removal."""
    clean, rerun, batches = _clean_and_rerun(tmp_path, _build_v69_coefficients,
                                             m._migrate_v70_capacity_coefficient_descriptions, monkeypatch,
                                             trigger=trigger, last=last)
    assert batches
    assert not [k for k in clean["values"] if k[2] in ("capacity_min_coeff", "input_share_min")]
    assert {k: v for k, v in clean["values"].items() if k[2] == "capacity_max_coeff"} == {
        ("unit__outputNode", (f"u{i}", "elec"), "capacity_max_coeff", "Base"): 0.8 for i in range(N)}
    assert ("unit__inputNode", "input_share_min") in clean["definitions"]
    assert rerun == clean


# ── v58: flow memberships leave group ────────────────────────────────────────────────────────────────────────

def _build_v57_memberships(url: str) -> None:
    """``N`` flow memberships of ``flow_only`` (no other role) and of ``dual`` (also a ``group__node`` member), each
    between the unit or connection it names, so the removed ids are scattered."""
    from spinedb_api import to_database

    from tests.update_flextool.test_v58_flowgroup_carveout import _build_v57_base

    _build_v57_base(url)
    with DatabaseMapping(url) as db:
        db.add_entity(entity_class_name="node", name="n")
        for g in ("flow_only", "dual"):
            db.add_entity(entity_class_name="group", name=g)
        db.add_entity(entity_class_name="group__node", entity_byname=("dual", "n"))
        value, value_type = to_database(100.0)
        db.add_parameter_value(entity_class_name="group", entity_byname=("flow_only",),
                               parameter_definition_name="max_instant_flow", alternative_name="base",
                               value=value, type=value_type)
        db.add_entity_alternative(entity_class_name="group", entity_byname=("flow_only",), alternative_name="base",
                                  active=True)
        for i in range(N):
            db.add_entity(entity_class_name="unit", name=f"u{i}")
            db.add_entity(entity_class_name="group__unit__node", entity_byname=("flow_only", f"u{i}", "n"))
            db.add_entity(entity_class_name="connection", name=f"c{i}")
            db.add_entity(entity_class_name="group__connection__node", entity_byname=("dual", f"c{i}", "n"))
        db.commit_session("scattered flow memberships")


@pytest.mark.parametrize("last", [False, True])
def test_v58_rerun_rehomes_every_membership(tmp_path, monkeypatch, last: bool) -> None:
    """Every flow membership is under flowGroup after the step, ``flow_only`` (with its value and entity_alternative)
    leaves group and ``dual`` keeps its node membership, also when the step stopped after the first batch of the
    membership removal, or after the last one (every membership gone, the groups not yet)."""
    clean, rerun, batches = _clean_and_rerun(tmp_path, _build_v57_memberships,
                                             m._migrate_v58_carve_flowgroup_out_of_group, monkeypatch,
                                             trigger="flow memberships", last=last)
    assert batches
    entities = set(clean["entities"])
    assert {("flowGroup__unit__node", ("flow_only", f"u{i}", "n")) for i in range(N)} <= entities
    assert {("flowGroup__connection__node", ("dual", f"c{i}", "n")) for i in range(N)} <= entities
    assert [b for c, b in clean["entities"] if c == "group"] == [("dual",)]
    assert [b for c, b in clean["entities"] if c == "group__node"] == [("dual", "n")]
    assert not [c for c, _ in clean["entities"] if c in ("group__unit__node", "group__connection__node")]
    assert clean["values"][("flowGroup", ("flow_only",), "max_instant_flow", "base")] == 100.0
    assert clean["entity_alternatives"][("flowGroup", ("flow_only",), "base")] is True
    assert rerun == clean


# ── migrate_database: all or nothing ─────────────────────────────────────────────────────────────────────────────

def _v80_file(tmp_path: Path) -> tuple[Path, str]:
    from tests.db_utils import json_to_db

    seed = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
    url = json_to_db(seed, tmp_path / "db.sqlite")
    m.migrate_database(url, up_to=80)
    return tmp_path / "db.sqlite", url


def _failing_v81(db) -> None:
    """The v81 step, then a failure after it has committed."""
    m._v81_value_lists(db)
    m._v81_definitions_part(db)
    m._commit_step(db, "v81 part")
    raise RuntimeError("step failed")


def test_failed_migration_restores_the_file(tmp_path, monkeypatch) -> None:
    path, url = _v80_file(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(m, "_migrate_v81_schema_batch", _failing_v81)
    with pytest.raises(RuntimeError, match="step failed"):
        m.migrate_database(url)
    assert path.read_bytes() == before
    assert not (tmp_path / "db.sqlite.premigration.bak").exists()


def test_without_backup_the_failed_step_stays_committed(tmp_path, monkeypatch) -> None:
    """``backup=False`` (a caller with its own backup) leaves the file as the failed step left it: the restore above
    is what keeps it whole."""
    path, url = _v80_file(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(m, "_migrate_v81_schema_batch", _failing_v81)
    with pytest.raises(RuntimeError, match="step failed"):
        m.migrate_database(url, backup=False)
    assert path.read_bytes() != before
    assert not (tmp_path / "db.sqlite.premigration.bak").exists()


def test_the_window_upgrade_backs_up_once(tmp_path, monkeypatch) -> None:
    """The window's upgrade holds its own backup and restores the file; ``migrate_database`` makes none of its own."""
    from flextool.gui import db_version_check
    from flextool.update_flextool import migration_backup

    path, _ = _v80_file(tmp_path)
    before = path.read_bytes()
    copies: list[Path] = []
    real_copy = migration_backup.shutil.copy2

    def counting_copy(src, dst, *args, **kwargs):
        if str(dst).endswith(".premigration.bak"):
            copies.append(Path(dst))
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(migration_backup.shutil, "copy2", counting_copy)
    monkeypatch.setattr(m, "_migrate_v81_schema_batch", _failing_v81)
    was_upgraded, failed, messages = db_version_check.check_and_upgrade_database(path)
    assert failed and not was_upgraded
    assert "restored" in messages[-1]
    assert copies == [path.with_name(path.name + ".premigration.bak")]
    assert path.read_bytes() == before

