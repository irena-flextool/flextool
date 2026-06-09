"""Unit tests: ``output-spinedb`` derived from the settings DB is a
REPLAY-only output.

``_resolve_settings`` derives the effective ``write_methods`` from a
settings DB ONLY when the caller passed ``write_methods is None``.  For the
``spinedb`` write-method this derived list must depend on the run path:

  * NATIVE solve path (``read_parquet_dir=False``): the results SpineDB is
    produced by the re-create / replay step, NOT the solve — so a
    settings-derived ``spinedb`` is STRIPPED (and a one-time info log fires).
  * REPLAY path (``read_parquet_dir=True``): the settings-derived list KEEPS
    ``spinedb`` — that path is exactly where it runs.
  * An EXPLICIT caller-supplied ``write_methods`` is never filtered — passing
    ``['spinedb']`` forces it on the native path too.

These exercise ``_resolve_settings`` directly against a minimal in-memory
settings DB built here (CLAUDE.md invariant #3: no checked-in ``.sqlite``).
No model is solved.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.process_outputs.write_outputs import _resolve_settings


def _build_settings_db(tmp_path: Path, output_flags: dict) -> str:
    """Create a minimal settings DB and return its sqlite URL.

    ``output_flags`` maps method name (e.g. ``'spinedb'``) → bool; each
    becomes an ``output-<method>`` boolean parameter value on the single
    ``settings`` entity.
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


def _resolve(settings_url, write_methods, read_parquet_dir):
    """Thin wrapper returning just the resolved ``write_methods``."""
    resolved = _resolve_settings(
        write_methods=write_methods,
        output_config_path=None,
        active_configs=None,
        plot_rows=None,
        output_location=None,
        plot_file_format=None,
        settings_db_url=settings_url,
        fallback_output_location=str(Path(settings_url)),
        results_db_url=None,
        read_parquet_dir=read_parquet_dir,
    )
    # _resolve_settings returns a 7-tuple; write_methods is the first element.
    return resolved[0]


def test_native_path_strips_settings_derived_spinedb(tmp_path, caplog):
    """NATIVE path + settings output-spinedb=true → spinedb NOT in the
    derived write_methods, and the info log fires once."""
    url = _build_settings_db(
        tmp_path, {"parquet": True, "spinedb": True, "plot": False,
                   "csv": False, "excel": False}
    )
    with caplog.at_level(logging.INFO):
        methods = _resolve(url, write_methods=None, read_parquet_dir=False)
    assert "spinedb" not in methods
    assert "parquet" in methods  # other settings-derived methods survive
    assert any(
        "re-create/replay step" in r.getMessage() for r in caplog.records
    ), "expected the one-time skip info log on the native path"


def test_replay_path_keeps_settings_derived_spinedb(tmp_path):
    """REPLAY path + settings output-spinedb=true → spinedb IS included."""
    url = _build_settings_db(
        tmp_path, {"parquet": True, "spinedb": True, "plot": False,
                   "csv": False, "excel": False}
    )
    methods = _resolve(url, write_methods=None, read_parquet_dir=True)
    assert "spinedb" in methods
    assert "parquet" in methods


def test_explicit_cli_spinedb_honored_on_native(tmp_path):
    """An EXPLICIT caller-supplied write_methods=['spinedb'] is honoured on
    the native path — only the settings-DERIVED list is filtered."""
    # Settings DB present but irrelevant: an explicit list bypasses derivation.
    url = _build_settings_db(
        tmp_path, {"parquet": True, "spinedb": False}
    )
    methods = _resolve(url, write_methods=["spinedb"], read_parquet_dir=False)
    assert methods == ["spinedb"]


def test_no_settings_db_explicit_spinedb_native(tmp_path):
    """No settings DB at all + explicit ['spinedb'] on native → still
    honoured (filter only ever touches the settings-derived list)."""
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
        read_parquet_dir=False,
    )[0]
    assert methods == ["spinedb"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
