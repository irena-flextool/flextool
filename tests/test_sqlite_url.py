"""SQLite URLs of filesystem paths (item 262): a path holding ``#``, ``%25``, spaces or non-ASCII letters opens
with spinedb-api 0.36.7 (reads a URL string as it is) and 0.36.8 (percent-decodes a URL string once more)."""

from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api.filters.scenario_filter import scenario_filter_config
from spinedb_api.filters.tools import append_filter_config
from sqlalchemy.engine import URL

from flextool.sqlite_url import (
    read_only_uri_of_posix,
    read_only_sqlite_url,
    read_only_sqlite_url_string,
    sqlite_file_path,
    sqlite_url,
    sqlite_url_string,
    to_sqlalchemy_url,
)
from flextool.update_flextool.db_handle import open_database_mapping

HOSTILE_NAMES = ["G #2 ä%25x project", "x%25y", "a b", "ä", "100%", "%23hash"]


@pytest.mark.parametrize("path", [
    "/tmp/G #2 ä%25x project/t.sqlite",
    "/tmp/x%25y/t.sqlite",
    "rel dir/%41.sqlite",
    "C:/Users/A B/x#1.sqlite",
    "C:\\Users\\A B\\x%25.sqlite",
    "//server/my share/G #1/x.sqlite",
    "\\\\server\\share\\ä%2525.sqlite",
])
def test_string_url_keeps_the_path_as_it_is(path):
    url = sqlite_url_string(path)
    assert url == "sqlite:///" + path
    parsed = to_sqlalchemy_url(url)
    assert isinstance(parsed, URL)
    assert (parsed.drivername, parsed.database, dict(parsed.query)) == ("sqlite", path, {})
    assert sqlite_url(path).database == path
    assert sqlite_url(Path(path)).database == str(Path(path))


def test_read_only_url_keeps_its_encoding_and_query():
    url = read_only_sqlite_url_string(Path("/tmp/G #2 ä%25x/t.sqlite"))
    assert url == "sqlite:///file:/tmp/G%20%232%20%C3%A4%2525x/t.sqlite?mode=ro&uri=true"
    parsed = to_sqlalchemy_url(url)
    assert parsed.database == "file:/tmp/G%20%232%20%C3%A4%2525x/t.sqlite"
    assert dict(parsed.query) == {"mode": "ro", "uri": "true"}
    assert read_only_sqlite_url(Path("/tmp/G #2 ä%25x/t.sqlite")) == parsed


def test_filter_query_in_front_of_a_hash_belongs_to_the_path():
    # spinedb-api's append_filter_config reads '#' as a fragment and puts the query in front of it.
    url = append_filter_config("sqlite:////tmp/G #2 ä%25x/t.sqlite", scenario_filter_config("S"))
    parsed = to_sqlalchemy_url(url)
    assert parsed.database == "/tmp/G #2 ä%25x/t.sqlite"
    assert parsed.query["spinedbfilter"] == "cfg:scenario:S"


@pytest.mark.parametrize("path", [
    "/tmp/G #2 ä%25x project/t.sqlite",
    "/tmp/x%25y/t.sqlite",
    "rel dir/%41.sqlite",
    "C:/Users/A B/x#1.sqlite",
    "//server/my share/G #1/x.sqlite",
])
def test_file_path_of_every_url_form(path):
    """Item 263: the file of a URL is the path as written, also when Toolbox put a filter query in front of a
    ``#`` of a folder name, and for the read-only URI (decoded back)."""
    assert sqlite_file_path(sqlite_url_string(path)) == Path(path)
    assert sqlite_file_path(sqlite_url_string(path) + "?mode=ro") == Path(path)
    # The form spinedb-api's append_filter_config gives: the query goes in front of the first '#'.
    head, hash_, tail = path.partition("#")
    filtered = sqlite_url_string(head) + "?spinedbfilter=cfg%3Ascenario%3AS" + hash_ + tail
    assert sqlite_file_path(filtered) == Path(path)
    if not path.startswith("rel"):
        assert sqlite_file_path(read_only_uri_of_posix(path)) == Path(path)
    assert sqlite_file_path(path) == Path(path)
    assert sqlite_file_path(Path(path)) == Path(path)


def test_file_path_of_a_filter_before_a_hash_is_not_cut():
    url = append_filter_config("sqlite:////tmp/G #2 ä%25x/t.sqlite", scenario_filter_config("S"))
    assert url.index("?spinedbfilter=") < url.index("#2")
    assert sqlite_file_path(url) == Path("/tmp/G #2 ä%25x/t.sqlite")


def test_file_path_of_other_databases_is_none():
    for url in (None, "", "sqlite:///", "sqlite://", "postgresql://u@host/db", "http://127.0.0.1:5000"):
        assert sqlite_file_path(url) is None, url


def test_path_extractors_use_the_shared_helper():
    """The six URL -> path readers of review 262 give the whole path of a filtered ``#`` folder."""
    from flextool._solver_config_source import input_db_filesystem_path
    from flextool.run_record._status import db_path_from_url
    from flextool.update_flextool.frozen import sqlite_path_from

    url = append_filter_config("sqlite:////tmp/G #2 ä%25x/t.sqlite", scenario_filter_config("S"))
    expected = Path("/tmp/G #2 ä%25x/t.sqlite")
    assert db_path_from_url(url) == expected
    assert input_db_filesystem_path(url) == expected
    assert sqlite_path_from(url) == expected


def test_other_urls_pass_unchanged():
    for url in ("sqlite://", "postgresql://u@host/db", "http://127.0.0.1:5000"):
        assert to_sqlalchemy_url(url) == url
    obj = URL.create("sqlite", database="/x%25.sqlite")
    assert to_sqlalchemy_url(obj) is obj
    assert to_sqlalchemy_url(Path("/a b/x.sqlite")).database == str(Path("/a b/x.sqlite"))


@pytest.fixture(scope="module")
def hostile_dbs(tmp_path_factory):
    """One Spine database with alternatives Base and alt2 and scenario S (alt2 only) in each hostile folder."""
    root = tmp_path_factory.mktemp("urls")
    paths = []
    for name in HOSTILE_NAMES:
        path = root / name / "t.sqlite"
        path.parent.mkdir()
        with open_database_mapping(sqlite_url_string(path), create=True) as db:
            db.add_alternative(name="alt2")
            db.add_scenario(name="S")
            db.add_scenario_alternative(scenario_name="S", alternative_name="alt2", rank=1)
            db.commit_session("test data")
        paths.append(path)
    return paths


def _alternatives(url, **kwargs):
    with open_database_mapping(url, **kwargs) as db:
        return sorted(item["name"] for item in db.find_alternatives())


def test_every_url_form_opens_a_hostile_path(hostile_dbs):
    for path in hostile_dbs:
        assert path.is_file()
        everything = ["Base", "alt2"]
        assert _alternatives(sqlite_url_string(path)) == everything, path
        assert _alternatives(sqlite_url_string(path.as_posix())) == everything, path
        assert _alternatives(path) == everything, path
        assert _alternatives(sqlite_url(path)) == everything, path
        assert _alternatives(read_only_sqlite_url_string(path)) == everything, path
        assert _alternatives(read_only_sqlite_url(path)) == everything, path


def test_filtered_string_url_applies_its_filters(hostile_dbs):
    for path in hostile_dbs:
        url = append_filter_config(sqlite_url_string(path), scenario_filter_config("S"))
        with open_database_mapping(url) as db:
            assert sorted(item["name"] for item in db.find_alternatives()) == ["alt2"], path
            assert db.get_filter_configs() == [scenario_filter_config("S")]
        assert _alternatives(url, apply_filters=False) == ["Base", "alt2"], path
