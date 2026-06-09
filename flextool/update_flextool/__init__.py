"""Update and migration: GitHub update, database schema versioning."""

#: The current FlexTool database schema version.
#: This is the single source of truth — all other modules import from here.
#: Defined before submodule imports to avoid circular-import issues.
FLEXTOOL_DB_VERSION: int = 58

from flextool.update_flextool.self_update import update_flextool  # noqa: E402  # FLEXTOOL_DB_VERSION must precede submodule imports to break circular dep
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402  # see FLEXTOOL_DB_VERSION ordering note above
from flextool.update_flextool.initialize_database import initialize_database  # noqa: E402  # see FLEXTOOL_DB_VERSION ordering note above

__all__ = ['update_flextool', 'migrate_database', 'initialize_database', 'FLEXTOOL_DB_VERSION']
