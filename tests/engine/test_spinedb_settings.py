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

import importlib
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.process_outputs.write_outputs import _resolve_settings


def _build_settings_db(tmp_path: Path, output_flags: dict, results_db_url: str | None = None) -> str:
    """Create a minimal settings DB and return its sqlite URL.

    ``output_flags`` maps method name (e.g. ``'spinedb'``) → bool; each becomes
    an ``output-<method>`` boolean parameter value on the single ``settings``
    entity.  ``results_db_url`` becomes ``output-results-db-url``.
    """
    db_file = tmp_path / "output_settings.sqlite"
    url = "sqlite:///" + str(db_file)
    values = {f"output-{method}": bool(enabled) for method, enabled in output_flags.items()}
    if results_db_url is not None:
        values["output-results-db-url"] = results_db_url
    with DatabaseMapping(url, create=True) as db:
        db.add_or_update_entity_class(name="settings")
        db.add_or_update_entity(entity_class_name="settings", name="the_settings")
        for pname, parameter_value in values.items():
            db.add_or_update_parameter_definition(
                entity_class_name="settings", name=pname
            )
            value, type_ = to_database(parameter_value)
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


def _resolve_results_url(settings_url, write_methods=None, results_db_url=None):
    """``(write_methods, results_db_url)`` resolved from the settings DB."""
    resolved = _resolve_settings(
        write_methods=write_methods,
        output_config_path=None,
        active_configs=None,
        plot_rows=None,
        output_location=None,
        plot_file_format=None,
        settings_db_url=settings_url,
        fallback_output_location=str(Path(settings_url)),
        results_db_url=results_db_url,
    )
    return resolved[0], resolved[6]


@pytest.mark.parametrize("results_url", [
    "sqlite:///Z:/results/results.sqlite",            # a mapped network drive
    "sqlite://///server/share/results.sqlite",        # a UNC path, as FlexTool writes it
    "sqlite:////server/share/results.sqlite",
    "SQLite:///relative/results.sqlite",
])
def test_settings_results_db_url_sqlite_any_drive_accepted(tmp_path, results_url):
    """A ``sqlite:`` results URL from the settings DB is used whatever the drive (item 230 D3)."""
    url = _build_settings_db(tmp_path, {"spinedb": True}, results_db_url=results_url)
    methods, resolved_url = _resolve_results_url(url)
    assert "spinedb" in methods
    assert resolved_url == results_url


def test_settings_results_db_server_url_refused_with_spinedb(tmp_path, monkeypatch):
    """A server URL from the settings DB is refused when spinedb runs, before any results database is opened;
    the error names the command-line flag."""
    url = _build_settings_db(tmp_path, {"spinedb": True}, results_db_url="http://host/db")
    wo = importlib.import_module("flextool.process_outputs.write_outputs")
    from flextool.engine.input._solve_state import FlexToolConfigError

    opened = []
    real_open = wo.open_database_mapping
    monkeypatch.setattr(wo, "open_database_mapping", lambda u, *a, **k: (opened.append(u), real_open(u, *a, **k))[1])
    with pytest.raises(FlexToolConfigError, match="--results-db-url"):
        _resolve_results_url(url)
    assert opened == [url]  # only the settings DB itself


def test_settings_results_db_server_url_ignored_without_spinedb(tmp_path, caplog):
    """spinedb is not a write-method: the server URL is not used, with a warning, and the run goes on."""
    url = _build_settings_db(tmp_path, {"csv": True, "spinedb": False}, results_db_url="postgresql://host/db")
    with caplog.at_level("WARNING"):
        methods, resolved_url = _resolve_results_url(url)
    assert methods == ["csv"]
    assert resolved_url is None
    assert "postgresql://host/db" in caplog.text


def test_cli_results_db_server_url_accepted(tmp_path):
    """The command line's server URL wins over the settings DB's and is used (D3: servers from the CLI only)."""
    url = _build_settings_db(tmp_path, {"spinedb": True}, results_db_url="http://host/db")
    methods, resolved_url = _resolve_results_url(url, results_db_url="postgresql://host/results")
    assert "spinedb" in methods
    assert resolved_url == "postgresql://host/results"


@pytest.mark.parametrize("settings_url, expected", [
    ("sqlite://///server/share/s.sqlite", "//server/share/s.sqlite"),
    ("sqlite:///\\\\server\\share\\s.sqlite", "\\\\server\\share\\s.sqlite"),
    ("sqlite:///Z:/settings/s.sqlite", "Z:/settings/s.sqlite"),
])
def test_settings_db_on_network_drive_is_read(monkeypatch, settings_url, expected):
    """A settings DB on a UNC share or a mapped drive is looked up at that path and read, not silently skipped."""
    wo = importlib.import_module("flextool.process_outputs.write_outputs")

    checked, opened = [], []
    monkeypatch.setattr(Path, "exists", lambda self: (checked.append(str(self)), True)[1])

    class _Settings:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_entity_items(self, entity_class_name):
            return [{"name": "s"}]

        def get_parameter_value_items(self, entity_class_name):
            return [{"entity_byname": ("s",), "parameter_definition_name": "output-csv",
                     "value": to_database(True)[0], "type": to_database(True)[1]}]

    monkeypatch.setattr(wo, "open_database_mapping", lambda u, *a, **k: (opened.append(u), _Settings())[1])
    methods, _ = _resolve_results_url(settings_url)
    assert checked[0] == str(Path(expected))
    assert opened == [settings_url]
    assert methods == ["csv"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
