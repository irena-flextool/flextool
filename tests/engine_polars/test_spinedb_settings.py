"""Unit tests: ``output-spinedb`` from the settings DB is honored on BOTH the
native solve path and the parquet-replay path.

``_resolve_settings`` derives the effective ``write_methods`` from a settings
DB ONLY when the caller passed ``write_methods is None``; the ``output-<method>``
boolean flags map 1:1 to write-methods.  An explicit caller-supplied list
bypasses derivation entirely.  There is no path distinction for ``spinedb`` —
a True flag writes the results SpineDB on either path (the run produces it from
the live solve; re-create rebuilds/augments it from parquet).

Built against a minimal in-memory settings DB (CLAUDE.md invariant #3: no
checked-in ``.sqlite``).  No model is solved.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.process_outputs.write_outputs import _resolve_settings


def _build_settings_db(tmp_path: Path, output_flags: dict) -> str:
    """Create a minimal settings DB and return its sqlite URL.

    ``output_flags`` maps method name (e.g. ``'spinedb'``) → bool; each becomes
    an ``output-<method>`` boolean parameter value on the single ``settings``
    entity.
    """
    db_file = tmp_path / "output_settings.sqlite"
    url = "sqlite:///" + str(db_file)
    with DatabaseMapping(url, create=True) as db:
        db.add_or_update_entity_class(name="settings")
        db.add_or_update_entity(entity_class_name="settings", name="the_settings")
        for method, enabled in output_flags.items():
            pname = f"output-{method}"
            db.add_or_update_parameter_definition(
                entity_class_name="settings", name=pname
            )
            value, type_ = to_database(bool(enabled))
            db.add_or_update_parameter_value(
                entity_class_name="settings",
                entity_byname=("the_settings",),
                parameter_definition_name=pname,
                alternative_name="Base",
                value=value,
                type=type_,
            )
        db.commit_session("seed settings")
    return url


def _resolve(settings_url, write_methods):
    """Thin wrapper returning just the resolved ``write_methods``."""
    return _resolve_settings(
        write_methods=write_methods,
        output_config_path=None,
        active_configs=None,
        plot_rows=None,
        output_location=None,
        plot_file_format=None,
        settings_db_url=settings_url,
        fallback_output_location=str(Path(settings_url)),
        results_db_url=None,
    )[0]


def test_settings_spinedb_true_included(tmp_path):
    """settings output-spinedb=true → spinedb IS in the derived write_methods."""
    url = _build_settings_db(tmp_path, {"parquet": True, "spinedb": True})
    methods = _resolve(url, write_methods=None)
    assert "spinedb" in methods
    assert "parquet" in methods


def test_settings_spinedb_false_excluded(tmp_path):
    """settings output-spinedb=false → spinedb is NOT in the derived list."""
    url = _build_settings_db(tmp_path, {"parquet": True, "spinedb": False})
    methods = _resolve(url, write_methods=None)
    assert "spinedb" not in methods
    assert "parquet" in methods


def test_explicit_write_methods_bypass_settings(tmp_path):
    """An explicit caller-supplied list wins; the settings list is not derived."""
    url = _build_settings_db(tmp_path, {"spinedb": True})
    methods = _resolve(url, write_methods=["plot"])
    assert methods == ["plot"]


def test_no_settings_db_explicit_spinedb(tmp_path):
    """No settings DB + explicit ['spinedb'] → honoured unchanged."""
    methods = _resolve_settings(
        write_methods=["spinedb"],
        output_config_path=None,
        active_configs=None,
        plot_rows=None,
        output_location=None,
        plot_file_format=None,
        settings_db_url=None,
        fallback_output_location=str(tmp_path),
        results_db_url=None,
    )[0]
    assert methods == ["spinedb"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
