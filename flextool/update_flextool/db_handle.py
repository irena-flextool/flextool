"""Open a Spine database and release its file when done.

``with DatabaseMapping(url) as db`` closes only the session: the engine's
pooled connection keeps the SQLite file open until ``close()`` disposes it
(or the garbage collector happens to).  On Windows an open file cannot be
deleted or replaced, so a temporary folder holding the database cannot be
removed.  :func:`open_database_mapping` closes the engine on exit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from spinedb_api import DatabaseMapping


@contextmanager
def open_database_mapping(url: str, **kwargs) -> Iterator[DatabaseMapping]:
    """``DatabaseMapping(url, **kwargs)`` as a context; the session and the engine are closed on exit."""
    db = DatabaseMapping(url, **kwargs)
    try:
        with db:
            yield db
    finally:
        db.close()
