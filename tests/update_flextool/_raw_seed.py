"""Definitions-only seeds as raw SQLite rows: dumped from a shipped ``.sqlite`` of an old release, rebuilt with
``sqlite3`` (never a spinedb export: a spinedb round trip reads a list default stored as text, item 279, as a lost
default).

A seed is a JSON object ``{"source": ..., "schema": [SQL, ...], "tables": {name: {"columns": [...], "rows":
[[...], ...]}}}``.  ``schema`` holds the ``CREATE`` statements of ``sqlite_master`` in their order; every table is
kept (an old schema needs its ``entity_class_type``, ``relationship_entity_class`` or ``next_id`` rows for the
SpineDB upgrade), but the rows of :data:`DATA_TABLES` (entities and values) are dropped.  A ``bytes`` cell is
``{"b64": ...}``.  The ``commit`` rows keep their dates and messages (the upgrade history ``value_list_history``
reads); their ``user`` is blanked unless asked otherwise.

Shared by the release-seed tests, ``tools/make_release_seeds.py`` and ``tools/migration_sweep.py``.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
from pathlib import Path

#: Tables whose rows are model data, not definitions: left empty in a seed.
DATA_TABLES = frozenset({
    "entity", "object", "relationship", "relationship_entity", "entity_element", "entity_alternative", "entity_group",
    "entity_metadata", "entity_location", "parameter_value", "parameter_value_metadata",
})


def _cell(value):
    if isinstance(value, bytes):
        return {"b64": base64.b64encode(value).decode("ascii")}
    return value


def _uncell(value):
    if isinstance(value, dict):
        return base64.b64decode(value["b64"])
    return value


def dump_rows(path: Path, source: str, *, blank_users: bool = True) -> dict:
    """The seed of the SQLite file *path* (*source* says where it came from)."""
    con = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        master = con.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL "
                             "AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall()
        tables: dict = {}
        for kind, name, _sql in master:
            if kind != "table":
                continue
            columns = [row[1] for row in con.execute(f'PRAGMA table_info("{name}")')]
            rows = [] if name in DATA_TABLES else [
                [_cell(v) for v in row] for row in con.execute(f'SELECT * FROM "{name}" ORDER BY rowid')]
            if name == "commit" and blank_users and "user" in columns:
                at = columns.index("user")
                rows = [row[:at] + [""] + row[at + 1:] for row in rows]
            tables[name] = {"columns": columns, "rows": rows}
    finally:
        con.close()
    return {"source": source, "schema": [sql for _kind, _name, sql in master], "tables": tables}


def write_seed(seed: dict, target: Path) -> None:
    """*seed* as JSON with one row per line (a stable, readable diff)."""
    lines = ["{", f'"source": {json.dumps(seed["source"])},', '"schema": [']
    lines += [json.dumps(sql) + ("," if i < len(seed["schema"]) - 1 else "") for i, sql in enumerate(seed["schema"])]
    lines += ["],", '"tables": {']
    names = list(seed["tables"])
    for i, name in enumerate(names):
        table = seed["tables"][name]
        lines.append(f'{json.dumps(name)}: {{"columns": {json.dumps(table["columns"])}, "rows": [')
        rows = table["rows"]
        lines += [json.dumps(row, ensure_ascii=False) + ("," if j < len(rows) - 1 else "") for j, row in enumerate(rows)]
        lines.append("]}" + ("," if i < len(names) - 1 else ""))
    lines += ["}", "}"]
    Path(target).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_seed(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(seed: dict, target: Path) -> Path:
    """A new SQLite file *target* with the tables, rows and indexes of *seed*."""
    con = sqlite3.connect(target)
    try:
        tables = [sql for sql in seed["schema"] if sql.lstrip().upper().startswith("CREATE TABLE")]
        others = [sql for sql in seed["schema"] if sql not in tables]
        for sql in tables:
            con.execute(sql)
        for name, table in seed["tables"].items():
            if table["rows"]:
                columns = ", ".join(f'"{c}"' for c in table["columns"])
                marks = ", ".join("?" for _c in table["columns"])
                con.executemany(f'INSERT INTO "{name}" ({columns}) VALUES ({marks})',
                                [[_uncell(v) for v in row] for row in table["rows"]])
        for sql in others:
            con.execute(sql)
        con.commit()
    finally:
        con.close()
    return target


def git_show(repo: Path, ref: str, path: str) -> bytes | None:
    """The bytes of *path* at *ref* in the git clone *repo*, or ``None`` when the ref has no such file."""
    result = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"], capture_output=True,
                            stdin=subprocess.DEVNULL, check=False)
    return result.stdout if result.returncode == 0 else None


def seed_from_git(repo: Path, ref: str, path: str, work: Path, *, blank_users: bool = True) -> dict:
    """The seed of the shipped SQLite *path* at *ref* (written to *work* to be read)."""
    data = git_show(repo, ref, path)
    if data is None:
        raise FileNotFoundError(f"{ref}:{path}")
    raw = Path(work) / "shipped.sqlite"
    raw.write_bytes(data)
    try:
        return dump_rows(raw, f"{ref}:{path}", blank_users=blank_users)
    finally:
        raw.unlink()
