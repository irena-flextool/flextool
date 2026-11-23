"""Shared fixtures of the migration tests."""
from __future__ import annotations

import contextlib
import io

import pytest


@pytest.fixture(scope="session")
def v82_schema_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A database with the schema of data version 82 and no entities: the bundled v25 template migrated to 82
    (about 12 s, once per session).  The v83 parts are called directly on copies of it.  Its ``startup_methods`` is
    [no_startup, linear, binary], ``dispatch_realization_methods`` [within_solve, separate_dispatch] with the default
    within_solve, and the descriptions the v83 parts replace are the shipped v82 texts."""
    from flextool._resources import package_data_path
    from flextool.update_flextool.db_migration import migrate_database
    from flextool.update_flextool.initialize_database import initialize_database

    path = tmp_path_factory.mktemp("v82_schema_db") / "v82.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(package_data_path("schemas/pre_v26/flextool_template_v25.json")), str(path))
        migrate_database(f"sqlite:///{path}", up_to=82)
    return f"sqlite:///{path}"
