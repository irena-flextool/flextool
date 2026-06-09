"""flextool-update back-fills new parameter DEFINITIONS into an existing
user settings DB (e.g. ``output-spinedb`` added after the DB was first made)
WITHOUT overwriting the user's existing parameter VALUES.

Regression for the gap where ``output_settings.sqlite`` made by an older
version never gained the ``output-spinedb`` option (create-if-absent only).
"""

import json

from spinedb_api import DatabaseMapping, import_data, from_database

from flextool._resources import package_data_path
from flextool.update_flextool.initialize_database import initialize_database
from flextool.update_flextool.self_update import _sync_settings_param_defs


def _settings_param_defs(db_url):
    with DatabaseMapping(db_url) as db:
        return {d["name"] for d in
                db.get_parameter_definition_items(entity_class_name="settings")}


def _default_values(db_url):
    with DatabaseMapping(db_url) as db:
        return {
            v["parameter_definition_name"]: from_database(v["value"], v["type"])
            for v in db.get_parameter_value_items(entity_class_name="settings")
            if v["entity_byname"] == ("default",)
        }


def test_sync_backfills_output_spinedb_and_preserves_user_values(tmp_path):
    template_path = str(package_data_path("schemas/output_settings_template.json"))
    with open(template_path) as f:
        template = json.load(f)
    assert any(d[1] == "output-spinedb" for d in template["parameter_definitions"]), \
        "fixture itself must define output-spinedb"

    # Build an OLD template that predates output-spinedb (drop its def/type/value).
    old = json.loads(json.dumps(template))
    old["parameter_definitions"] = [
        d for d in old["parameter_definitions"] if d[1] != "output-spinedb"]
    old["parameter_types"] = [
        t for t in old.get("parameter_types", []) if t[1] != "output-spinedb"]
    old["parameter_values"] = [
        v for v in old.get("parameter_values", []) if v[2] != "output-spinedb"]
    old_json = tmp_path / "old_output_settings.json"
    old_json.write_text(json.dumps(old))

    db_path = str(tmp_path / "output_settings.sqlite")
    db_url = "sqlite:///" + db_path
    initialize_database(str(old_json), db_path)

    # Precondition: the old DB lacks output-spinedb.
    assert "output-spinedb" not in _settings_param_defs(db_url)

    # The user sets a NON-default value, which the sync must not clobber.
    with DatabaseMapping(db_url) as db:
        import_data(db, parameter_values=[
            ["settings", "default", "output-csv", True, "default"]])
        db.commit_session("user edit")
    assert _default_values(db_url).get("output-csv") is True

    # Sync from the current (full) template.
    _sync_settings_param_defs(template_path, db_path)

    # The missing definition is back-filled...
    assert "output-spinedb" in _settings_param_defs(db_url)
    assert "output-results-db-url" in _settings_param_defs(db_url)
    # ...and the user's value is preserved (NOT reset to the template default).
    assert _default_values(db_url).get("output-csv") is True


def test_sync_is_idempotent(tmp_path):
    template_path = str(package_data_path("schemas/output_settings_template.json"))
    db_path = str(tmp_path / "output_settings.sqlite")
    initialize_database(template_path, db_path)
    before = _settings_param_defs("sqlite:///" + db_path)
    _sync_settings_param_defs(template_path, db_path)  # no-op on a current DB
    _sync_settings_param_defs(template_path, db_path)
    after = _settings_param_defs("sqlite:///" + db_path)
    assert before == after
