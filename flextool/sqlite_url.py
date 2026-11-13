"""SQLite database URLs built from filesystem paths, safe for any path.

A ``sqlite:///`` URL string keeps the file path as it is: ``sqlite:////tmp/G #2 ä%25x/t.sqlite`` names the
folder ``G #2 ä%25x``.  FlexTool builds and shows URLs in that form (run records, messages, command lines).
spinedb-api 0.36.7 reads such a string the same way, but spinedb-api 0.36.8 percent-decodes every URL string
given to ``DatabaseMapping`` (``%25`` becomes ``%`` and the file is not found; an encoded ``%23`` in a read-only
``file:`` URI becomes ``#``, which cuts the path).  A :class:`sqlalchemy.engine.URL` object is used as it is by
both versions, so a database is opened through :func:`to_sqlalchemy_url`, which turns a ``sqlite:///`` string
into such an object without decoding anything.

This module imports the standard library only; SQLAlchemy is imported when a URL object is built.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, quote, unquote

if TYPE_CHECKING:
    from sqlalchemy.engine import URL

#: The prefix of a SQLite file URL; what follows it (up to a ``?query``) is the file path as it is.
SQLITE_PREFIX = "sqlite:///"


def sqlite_url(path: str | os.PathLike[str]) -> URL:
    """The SQLAlchemy URL of the SQLite file *path* (relative or absolute, any characters, a Windows drive or UNC
    path too).  The path is not encoded or decoded."""
    from sqlalchemy.engine import URL

    return URL.create("sqlite", database=os.fspath(path))


def sqlite_url_string(path: str | os.PathLike[str]) -> str:
    """The readable ``sqlite:///<path>`` URL string of *path*, for messages, records and command lines.  It is
    opened through :func:`to_sqlalchemy_url` (``open_database_mapping`` does that)."""
    return SQLITE_PREFIX + os.fspath(path)


def read_only_sqlite_url_string(path: str | os.PathLike[str]) -> str:
    """``sqlite:///file:/a/b%20c.sqlite?mode=ro&uri=true``: the URL that opens *path* read-only as an SQLite
    URI (the file is never created or written).  See :func:`read_only_uri_of_posix`."""
    return read_only_uri_of_posix(Path(path).absolute().as_posix())


def read_only_uri_of_posix(posix: str) -> str:
    """``sqlite:///file:/a/b%20c.sqlite?mode=ro&uri=true`` of an absolute POSIX-style path; a Windows drive path
    ``C:/a/b.sqlite`` gets a leading ``/`` (``file:/C:/a/b.sqlite``), the form SQLite reads as an absolute file
    URI there.  A UNC path ``//server/share/x.sqlite`` gets ``//`` more (``file:////server/share/x.sqlite``): an
    empty authority (SQLite refuses any other) and the UNC path as the path, which SQLite's Windows VFS opens as
    ``\\\\server\\share\\x.sqlite``.  Every part is percent-encoded (SQLite decodes the URI), the server and
    share names too."""
    if posix.startswith("//"):
        posix = "//" + posix
    elif not posix.startswith("/"):
        posix = "/" + posix
    return SQLITE_PREFIX + "file:" + quote(posix, safe="/:") + "?mode=ro&uri=true"


def read_only_sqlite_url(path: str | os.PathLike[str]) -> URL:
    """The SQLAlchemy URL of :func:`read_only_sqlite_url_string`."""
    url = to_sqlalchemy_url(read_only_sqlite_url_string(path))
    assert not isinstance(url, str)
    return url


def is_sqlite_url(url: str) -> bool:
    """Whether *url* is an SQLite URL: it starts with ``sqlite:`` (any case, leading blanks ignored).  A plain
    string test: ``sqlite:///Z:/a.sqlite``, ``sqlite://///server/share/x.sqlite`` (a UNC path on a network share)
    and ``sqlite:////server/share/x.sqlite`` are SQLite URLs whatever the file; a database server URL
    (``postgresql://``, ``http://``) is not."""
    return url.strip().lower().startswith("sqlite:")


def _split_sqlite_url(url: str) -> tuple[str, str]:
    """``(database, query)`` of a ``sqlite:///`` string: the text after ``sqlite:///`` up to the first ``?``, and
    the query after it.  A ``#`` in the query starts the rest of the path: spinedb-api's ``append_filter_config``
    puts a filter query in front of a ``#`` it reads as a fragment (``/a/G ?spinedbfilter=...#2/t.sqlite`` is
    the file ``/a/G #2/t.sqlite``)."""
    database, _, query = url[len(SQLITE_PREFIX):].partition("?")
    query, hash_, fragment = query.partition("#")
    return database + hash_ + fragment, query


def sqlite_file_path(url: str | os.PathLike[str] | None) -> Path | None:
    """The file of a ``sqlite:///`` URL string or of a bare path; ``None`` for an empty value, an in-memory
    ``sqlite://`` and any other database URL (a server).

    The path is the text after ``sqlite:///`` as it is (no decoding), without the query (``?spinedbfilter=...``
    of Spine Toolbox, ``?mode=ro``); a ``#`` that a filter query was put in front of is joined back
    (:func:`_split_sqlite_url`).  A read-only SQLite URI (``sqlite:///file:/a/b%20c.sqlite?mode=ro&uri=true``,
    :func:`read_only_uri_of_posix`) is decoded back to its file (``/a/b c.sqlite``; ``C:/...`` and
    ``//server/share/...`` too).  Pure string -> path; does not touch the filesystem."""
    if url is None:
        return None
    if not isinstance(url, str):
        text = os.fspath(url)
        return Path(text) if text else None
    if not url:
        return None
    if not url.startswith(SQLITE_PREFIX):
        return None if "://" in url else Path(url)
    database, query = _split_sqlite_url(url)
    if database.startswith("file:") and ("uri", "true") in parse_qsl(query, keep_blank_values=True):
        database = unquote(database[len("file:"):])
        if database.startswith("////"):
            database = database[2:]
        elif len(database) > 2 and database[0] == "/" and database[2] == ":":
            database = database[1:]
    return Path(database) if database else None


def to_sqlalchemy_url(url: str | os.PathLike[str] | URL) -> str | URL:
    """*url* in the form to give ``DatabaseMapping`` / ``create_engine``: a ``sqlite:///`` string becomes a
    :class:`~sqlalchemy.engine.URL` whose database is the text after ``sqlite:///`` as it is (up to the first
    ``?``; a file path cannot hold ``?`` on Windows) and whose query is the query of the string (``mode=ro``,
    ``spinedbfilter`` …, decoded as a query is).  A ``#`` after the query is part of the path: spinedb-api's
    ``append_filter_config`` puts the query in front of a ``#`` it reads as a fragment.  A path object becomes
    :func:`sqlite_url`.  A URL object and any other string (a database server, an in-memory ``sqlite://``) are
    returned unchanged.  ``DatabaseMapping`` does not read a filter query of a URL object as it reads one of a
    string: :func:`flextool.update_flextool.db_handle.database_mapping` applies the filters itself."""
    from sqlalchemy.engine import URL

    if isinstance(url, URL):
        return url
    if not isinstance(url, str):
        return sqlite_url(url)
    if not url.startswith(SQLITE_PREFIX):
        return url
    database, query = _split_sqlite_url(url)
    params: dict[str, list[str]] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        params.setdefault(key, []).append(value)
    return URL.create(
        "sqlite",
        database=database or None,
        query={key: values[0] if len(values) == 1 else tuple(values) for key, values in params.items()},
    )
