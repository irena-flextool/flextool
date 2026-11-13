"""Open a Spine database and release its file when done.

``with DatabaseMapping(url) as db`` closes only the session: the engine's
pooled connection keeps the SQLite file open until ``close()`` disposes it
(or the garbage collector happens to).  On Windows an open file cannot be
deleted or replaced, so a temporary folder holding the database cannot be
removed.  :func:`open_database_mapping` closes the engine on exit.

Every FlexTool site opens a database through :func:`database_mapping` (or
:func:`open_database_mapping`): a ``sqlite:///<path>`` string is turned into a
SQLAlchemy URL object first (:func:`flextool.sqlite_url.to_sqlalchemy_url`),
so a path holding ``%`` or ``#`` opens with every spinedb-api version.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlencode

from spinedb_api import DatabaseMapping
from spinedb_api.filters.tools import FILTER_IDENTIFIER, apply_filter_stack, load_filters, pop_filter_configs
from sqlalchemy.engine import URL

from flextool.sqlite_url import SQLITE_PREFIX, to_sqlalchemy_url


def database_mapping(url: str | os.PathLike[str] | URL, **kwargs) -> DatabaseMapping:
    """``DatabaseMapping(url, **kwargs)`` with a ``sqlite:///`` string *url* read as a file path as it is
    (:func:`~flextool.sqlite_url.to_sqlalchemy_url`).

    A Spine filter query of such a string (``?spinedbfilter=...``, e.g. from Spine Toolbox) is taken off the
    URL and its filters applied to the opened mapping, as ``DatabaseMapping`` does for a string
    (``apply_filters=False`` skips them)."""
    target = to_sqlalchemy_url(url)
    configs: list = []
    if isinstance(url, str) and isinstance(target, URL) and FILTER_IDENTIFIER in target.query:
        values = target.query[FILTER_IDENTIFIER]
        values = [values] if isinstance(values, str) else list(values)
        # spinedb-api parses the filter values (a config file path, or a ``cfg:`` shorthand) of a string only.
        configs, _ = pop_filter_configs(SQLITE_PREFIX + "x?" + urlencode([(FILTER_IDENTIFIER, value) for value in values]))
        target = target.difference_update_query([FILTER_IDENTIFIER])
    apply_filters = kwargs.get("apply_filters", True)
    db = DatabaseMapping(target, **kwargs)
    if configs and apply_filters:
        try:
            apply_filter_stack(db, load_filters(configs))
        except BaseException:
            db.close()
            raise
    return db


@contextmanager
def open_database_mapping(url: str | os.PathLike[str] | URL, **kwargs) -> Iterator[DatabaseMapping]:
    """:func:`database_mapping` as a context; the session and the engine are closed on exit."""
    db = database_mapping(url, **kwargs)
    try:
        with db:
            yield db
    finally:
        db.close()
