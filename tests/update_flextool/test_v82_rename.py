"""Database migration to version 82: the model's discount parameters are renamed.

A v80 database (built from the v75 seed JSON and migrated up to 80, never a checked-in ``.sqlite``) carries
``model.inflation_rate`` values in two alternatives (a constant and a period map), an ``inflation_offset_investment``
of 1 and a default of its own, 0.25, on ``inflation_offset_operations``.  Through 81 to 82:

* ``inflation_rate`` is ``social_discount_rate``, ``inflation_offset_investment`` is ``discount_offset_investment``
  and ``inflation_offset_operations`` is ``discount_offset_operations``: each value keeps its bytes, type and
  alternative, each definition keeps its stored default (an empty one, 0.0, the user's 0.25) and its group, and the
  old names are gone;
* the descriptions of the three and of ``discount_rate`` and ``payback_time`` (unit, connection, node) are the
  shipped ones, and no shipped description names an old name;
* nothing else differs from a migration of the same database without those values.

A database migrated to 81 by a development build (a superseded shipped ``payback_time`` description, a v81
definition missing) reaches the same state.  Each part, the whole step and the refresh of a database at 82 commit
nothing on a rerun; a version reset to 80 or 55 migrates to the same state.  An old definition that is back beside
the new one (the user added a new name, a template was re-applied, old-named data was imported) gives its values to
the new name where it has none, the ones it has are kept and printed, and the old definition goes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, Map, import_data, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import migrate_database
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
OLD = tuple(m.V82_PARAMETER_RENAMES)
NEW = tuple(m.V82_PARAMETER_RENAMES.values())
#: Stored at 80: (alternative index, old name, value).
STORED = [
    (0, "inflation_rate", 0.02),
    (1, "inflation_rate", Map(["p2020", "p2025"], [0.03, 0.04], index_name="period")),
    (0, "inflation_offset_investment", 1.0),
]
USER_DEFAULT = 0.25  # on inflation_offset_operations


def _defs(db) -> dict:
    return {(d["entity_class_name"], d["name"]): (d["description"] or "", tuple(d["parameter_type_list"] or ()),
                                                  d["parameter_group_name"], d["parameter_value_list_name"],
                                                  None if d["default_value"] is None else bytes(d["default_value"]),
                                                  d["default_type"])
            for d in db.get_items("parameter_definition")}


def _values(db) -> dict:
    return {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
             v["alternative_name"]): (bytes(v["value"]), v["type"]) for v in db.get_items("parameter_value")}


def _state(url: str) -> dict:
    with DatabaseMapping(url) as db:
        return {"defs": _defs(db), "values": _values(db)}


def _shipped() -> dict[tuple[str, str], str]:
    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        return {(r[0], r[1]): r[4] for r in json.load(f)["parameter_definitions"]}


def _model_entity(db) -> str:
    return sorted(e["name"] for e in db.get_items("entity", entity_class_name="model"))[0]


def _plant(db, *, user_default: bool = True) -> None:
    """Store :data:`STORED` and the user's default (old names)."""
    alts = sorted(a["name"] for a in db.get_items("alternative"))
    entity = _model_entity(db)
    for i, name, value in STORED:
        val, typ = to_database(value)
        db.add_parameter_value(entity_class_name="model", entity_byname=(entity,), parameter_definition_name=name,
                               alternative_name=alts[i], value=val, type=typ)
    if user_default:
        pdef = db.get_item("parameter_definition", entity_class_name="model", name="inflation_offset_operations")
        val, typ = to_database(USER_DEFAULT)
        db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)


def _seed(tmp_path_factory, tag: str, up_to: int) -> str:
    url = json_to_db(SEED, tmp_path_factory.mktemp(tag) / "seed.sqlite")
    migrate_database(url, up_to=up_to)
    return url


def _set_version(url: str, version: int) -> None:
    with DatabaseMapping(url) as db:
        import_data(db, object_parameters=[["model", "version", version, None,
                                            "[version number] Contains database version information."]])
        db.commit_session("test: set version")


@pytest.fixture(scope="module")
def renamed(tmp_path_factory) -> dict:
    url = _seed(tmp_path_factory, "v82", 80)
    with DatabaseMapping(url) as db:
        _plant(db)
        db.commit_session("v80 discount settings")
    migrate_database(url, up_to=81)
    before = _state(url)
    migrate_database(url)
    clean = _seed(tmp_path_factory, "v82_clean", FLEXTOOL_DB_VERSION)
    return {"url": url, "before": before, "after": _state(url), "clean": _state(clean)}


def test_values_keep_their_bytes_under_the_new_names(renamed) -> None:
    before, after = renamed["before"]["values"], renamed["after"]["values"]
    renames = m.V82_PARAMETER_RENAMES
    moved = {(c, e, renames[p], a): v for (c, e, p, a), v in before.items() if c == "model" and p in renames}
    assert len(moved) == len(STORED)
    assert {k: v for k, v in after.items() if k[0] == "model" and k[2] in NEW} == moved
    assert not [k for k in after if k[0] == "model" and k[2] in OLD]


def test_definitions_keep_their_defaults_and_get_the_shipped_fields(renamed) -> None:
    """The raw stored default of each definition is the one it had under its old name (never compared or
    rewritten): empty for the rate, 0.0, the user's 0.25; constant, group ``model``, the shipped description."""
    before, after = renamed["before"]["defs"], renamed["after"]["defs"]
    shipped = _shipped()
    for old, new in m.V82_PARAMETER_RENAMES.items():
        assert ("model", old) not in after
        assert after[("model", new)][4:] == before[("model", old)][4:], new
        assert after[("model", new)][:4] == (shipped[("model", new)], ("float",), "model", None), new
    for name, default in (("discount_offset_operations", USER_DEFAULT), ("discount_offset_investment", 0.0)):
        val, typ = to_database(default)
        assert after[("model", name)][4:] == (bytes(val), typ), name


def test_descriptions_are_the_shipped_ones(renamed) -> None:
    after, shipped = renamed["after"]["defs"], _shipped()
    for key, text in m._v82_texts().items():
        assert after[key][0] == shipped[key] == text, key
    assert "discount_offset_investment - 1" in after[("unit", "discount_rate")][0]
    assert "social_discount_rate" in after[("node", "payback_time")][0]


def test_no_shipped_description_names_an_old_name() -> None:
    """Neither the schema nor a text the v81 refresh writes on a database at 82 names an old parameter."""
    texts = list(_shipped().values()) + list({**m._v81_texts(), **m._v82_texts()}.values())
    for old in OLD:
        assert not [t for t in texts if old in (t or "")], old


def test_changes_nothing_else(renamed) -> None:
    """Without the old names and their values, the database equals a migration of the same seed without them (the
    user's default aside)."""
    after, clean = renamed["after"], renamed["clean"]
    own = ("model", "discount_offset_operations")
    assert ({k: v for k, v in after["defs"].items() if k != own}
            == {k: v for k, v in clean["defs"].items() if k != own})
    assert after["defs"][own][:4] == clean["defs"][own][:4]
    assert ({k: v for k, v in after["values"].items() if not (k[0] == "model" and k[2] in NEW)}
            == {k: v for k, v in clean["values"].items() if not (k[0] == "model" and k[2] in NEW)})


def test_development_v81_database_reaches_the_same_state(tmp_path_factory, renamed) -> None:
    """A database migrated to 81 by a development build of v81: the superseded shipped ``payback_time`` description
    on unit and a missing ``node.build_year``.  Migrated to 82 it equals the database above."""
    url = _seed(tmp_path_factory, "v82_dev81", 81)
    with DatabaseMapping(url) as db:
        _plant(db)
        pdef = db.get_item("parameter_definition", entity_class_name="unit", name="payback_time")
        db.update_parameter_definition(id=pdef["id"], description=m._v81_superseded()[("unit", "payback_time")][1])
        db.remove_item("parameter_definition",
                       db.get_item("parameter_definition", entity_class_name="node", name="build_year")["id"])
        db.commit_session("a development v81 database")
    migrate_database(url)
    assert _state(url) == renamed["after"]


def _no_commit(db, step, monkeypatch) -> None:
    commits: list[str] = []

    def commit_must_be_empty(db_, message) -> None:
        with pytest.raises(NothingToCommit):
            db_.commit_session(message)
        commits.append(message)

    monkeypatch.setattr(m, "_commit_step", commit_must_be_empty)
    step(db)
    if step is m._migrate_v82_schema_batch:
        assert len(commits) == 1, commits
    else:
        assert commits == []
        with pytest.raises(NothingToCommit):
            db.commit_session("rerun")


@pytest.mark.parametrize("part", ["_v82_rename_definitions", "_v82_definitions_part", "_v82_descriptions",
                                  "_migrate_v82_schema_batch", "_refresh_head_schema"])
def test_rerun_commits_nothing(renamed, part, monkeypatch) -> None:
    with DatabaseMapping(renamed["url"]) as db:
        _no_commit(db, getattr(m, part), monkeypatch)
    assert _state(renamed["url"]) == renamed["after"]


def _discount_part(state: dict) -> dict:
    """The definitions and values of the model's discount parameters (old and new names) in *state*."""
    names = set(OLD) | set(NEW)
    return {part: {k: v for k, v in state[part].items() if k[0] == "model" and k[1 if part == "defs" else 2] in names}
            for part in ("defs", "values")}


@pytest.mark.parametrize("version", [80, 55])
def test_version_reset_migrates_to_the_same_state(tmp_path_factory, renamed, version) -> None:
    """A run killed after the step committed, before the version was written, reruns from an earlier version.  From
    80 (v81 and v82 again) the database ends as it was.  From 55, v56's default fixes find only the new names (a
    silent no-op on them) and v82 finds no old name: the discount parameters end as they were (v56 and later steps
    rerun on the rest of the schema rewrite some defaults of their own, which is not v82's)."""
    url = _seed(tmp_path_factory, f"v82_reset_{version}", 80)
    with DatabaseMapping(url) as db:
        _plant(db)
        db.commit_session("v80 discount settings")
    migrate_database(url)
    _set_version(url, version)
    migrate_database(url)
    if version == 80:
        assert _state(url) == renamed["after"]
    else:
        assert _discount_part(_state(url)) == _discount_part(renamed["after"])


def test_old_definition_back_beside_the_new_one(tmp_path_factory, capsys, monkeypatch) -> None:
    """At 81, the user added ``social_discount_rate`` with a value in the first alternative: the old value of that
    alternative is kept out and printed, the other one moves, ``inflation_offset_investment`` (no collision) is
    renamed, and the old ``inflation_rate`` goes.  A rerun commits nothing."""
    url = _seed(tmp_path_factory, "v82_collision", 81)
    with DatabaseMapping(url) as db:
        _plant(db, user_default=False)
        alts = sorted(a["name"] for a in db.get_items("alternative"))
        entity = _model_entity(db)
        db.add_parameter_definition(entity_class_name="model", name="social_discount_rate",
                                    parameter_type_list=("float",))
        val, typ = to_database(0.07)
        db.add_parameter_value(entity_class_name="model", entity_byname=(entity,),
                               parameter_definition_name="social_discount_rate", alternative_name=alts[0],
                               value=val, type=typ)
        db.commit_session("the new name added at 81")
        before = _values(db)
    capsys.readouterr()
    migrate_database(url)
    printed = capsys.readouterr().out
    assert ("v82: model social_discount_rate already had a value where inflation_rate had one; kept the "
            "social_discount_rate value, the inflation_rate value was not moved to social_discount_rate:\n"
            f"  model {entity} [{alts[0]}] = 0.02\n") in printed
    with DatabaseMapping(url) as db:
        after = _values(db)
        assert not [d for d in db.get_items("parameter_definition", entity_class_name="model") if d["name"] in OLD]
        assert _defs(db)[("model", "social_discount_rate")][:3] == (_shipped()[("model", "social_discount_rate")],
                                                                    ("float",), "model")
    rate = ("model", (entity,), "social_discount_rate")
    assert after[(*rate, alts[0])] == before[(*rate, alts[0])]
    assert after[(*rate, alts[1])] == before[("model", (entity,), "inflation_rate", alts[1])]
    assert (after[("model", (entity,), "discount_offset_investment", alts[0])]
            == before[("model", (entity,), "inflation_offset_investment", alts[0])])
    with DatabaseMapping(url) as db:
        _no_commit(db, m._migrate_v82_schema_batch, monkeypatch)


def test_empty_old_definition_back_at_82_is_removed(tmp_path_factory, renamed) -> None:
    """An empty ``inflation_offset_investment`` re-added (a template re-applied) and the version reset to 81: the
    step removes it, and the database ends as it was."""
    url = _seed(tmp_path_factory, "v82_empty_old", 80)
    with DatabaseMapping(url) as db:
        _plant(db)
        db.commit_session("v80 discount settings")
    migrate_database(url)
    with DatabaseMapping(url) as db:
        db.add_parameter_definition(entity_class_name="model", name="inflation_offset_investment",
                                    parameter_type_list=("float",))
        db.commit_session("an old definition re-added")
    _set_version(url, 81)
    migrate_database(url)
    assert _state(url) == renamed["after"]


def test_head_refresh_writes_the_v82_text_for_a_superseded_pair(tmp_path_factory) -> None:
    """A database at 82 whose ``payback_time`` still has the superseded v81 description (which names the old rate)
    gets the v82 description, not the v81 one."""
    url = _seed(tmp_path_factory, "v82_head_superseded", FLEXTOOL_DB_VERSION)
    with DatabaseMapping(url) as db:
        for cls in ("unit", "connection", "node"):
            pdef = db.get_item("parameter_definition", entity_class_name=cls, name="payback_time")
            db.update_parameter_definition(id=pdef["id"], description=m._v81_superseded()[(cls, "payback_time")][1])
        db.commit_session("a superseded payback_time text")
    path = Path(url.removeprefix("sqlite:///"))
    assert m._head_defaults_stale_ro(path)
    migrate_database(url)
    with DatabaseMapping(url) as db:
        for cls in ("unit", "connection", "node"):
            text = db.get_item("parameter_definition", entity_class_name=cls, name="payback_time")["description"]
            assert text == m._v82_texts()[(cls, "payback_time")] == _shipped()[(cls, "payback_time")]
            assert "inflation_rate" not in text
    assert not m._head_defaults_stale_ro(path)


#: The ``discount_offset_operations`` description an earlier development build of v82 shipped (no 0-1 range).
_DEV_V82_OPERATIONS_TEXT = (
    "[years] Where in each year operating and fixed costs are weighed by social_discount_rate, counted from the start "
    "of the year: 0.5 (default) the middle of the year, standing for costs spread over the year. No effect when "
    "social_discount_rate is 0. Constant.")


@pytest.mark.parametrize("edited", [False, True])
def test_head_refresh_replaces_the_superseded_operations_offset_text(tmp_path_factory, edited) -> None:
    """A database at 82 with the discount_offset_operations description of an earlier development build (and the
    shipped default 0.5) gets the current one, which gives the 0-1 range; a description the user edited is kept."""
    url = _seed(tmp_path_factory, f"v82_ops_text_{edited}", FLEXTOOL_DB_VERSION)
    old = _DEV_V82_OPERATIONS_TEXT + (" My note." if edited else "")
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="model", name="discount_offset_operations")
        db.update_parameter_definition(id=pdef["id"], description=old)
        db.commit_session("the operations offset text of a development build")
    path = Path(url.removeprefix("sqlite:///"))
    assert m._head_defaults_stale_ro(path) is not edited
    migrate_database(url)
    with DatabaseMapping(url) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="model", name="discount_offset_operations")
        if edited:
            assert pdef["description"] == old
        else:
            assert pdef["description"] == _shipped()[("model", "discount_offset_operations")]
            assert "between 0 and 1" in pdef["description"]
    assert not m._head_defaults_stale_ro(path)
