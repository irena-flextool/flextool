"""Migrate the shipped schema of every release tag to the current data version and report what differs.

For each release tag of the local clone (``v3.0.0`` .. the newest) the schema sources the tag ships
(:func:`pick_sources`) are built definitions-only in a temporary folder: a shipped ``.sqlite`` through the raw-row
builder of ``tests/update_flextool/_raw_seed.py`` (entities and values dropped), a template JSON through
``initialize_database``, a canonical examples JSON cut to its definitions first.  Each one is migrated as it is and
with the synthetic seed data of ``test_rerun_convergence._seed_data`` (and ``Base`` where it has none).  Sources that are the same git
blob in several tags are run once.

Reported per source: an error; *drift*, the differences between the one-shot migration and the step-by-step one;
the ``_already_applied`` records of both (a clean migration has none); the definitions whose description at the
current version differs from ``spinedb_schema.json`` (reported, not counted as a failure: an older migration path
can leave a text the current schema has replaced); the time of the one-shot migration.  ``--triangle``: also every
content k of the step-by-step chain under every stamp s < k, migrated to k, compared with content k.  ``--large``:
also the time of the large v69 database of ``test_large_database_migration``.

Run by hand before each data-version release (CI has no git history)::

    python tools/migration_sweep.py [--tags 'v4.*'] [--workers 8] [--triangle] [--large] [--json report.json]

Exit code 1 when any source has an error, drift or an ``_already_applied`` record.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import multiprocessing
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: A release tag: ``v`` and a version (pre-releases included); ``archive/*``, ``backup/*``, ``parked/*`` are not.
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+([ab]\d+)?$")
SCHEMA_JSON = "flextool/schemas/spinedb_schema.json"
MASTER_JSON_PACKAGED = "flextool/version/flextool_template_master.json"
EXAMPLES_JSON = "flextool/textual_templates/canonical_databases/templates_examples.json"
MASTER_JSON = "version/flextool_template_master.json"
EXAMPLES_SQLITE = "templates/examples.sqlite"
INIT_SQLITE = "Init.sqlite"
#: Data sections of a canonical examples JSON: dropped, so that only the definitions are migrated.
EXAMPLES_DATA_KEYS = ("entities", "entity_alternatives", "parameter_values", "entity_groups", "entity_metadata",
                      "parameter_value_metadata")


@dataclass(frozen=True)
class Source:
    """One schema source of a tag: *kind* ``sqlite`` (raw rows), ``template`` (a template JSON) or ``examples``
    (a canonical examples JSON cut to its definitions), at *path* in the tag."""
    kind: str
    path: str


def pick_sources(paths) -> list[Source]:
    """The schema sources a tag with the files *paths* ships, in the order of the plan's table:

    * v3.43.1 and later: ``flextool/schemas/spinedb_schema.json``;
    * v3.41.3 - v3.43.0: ``flextool/version/flextool_template_master.json`` and the canonical examples JSON;
    * v3.9.0 - v3.41.2: ``templates/examples.sqlite`` and ``version/flextool_template_master.json``;
    * v3.2.0 - v3.8.x: ``Init.sqlite`` and ``version/flextool_template_master.json``;
    * v3.0.0 - v3.1.x: ``Init.sqlite`` only.
    """
    paths = set(paths)
    if SCHEMA_JSON in paths:
        return [Source("template", SCHEMA_JSON)]
    if MASTER_JSON_PACKAGED in paths:
        return [Source("template", MASTER_JSON_PACKAGED)] + (
            [Source("examples", EXAMPLES_JSON)] if EXAMPLES_JSON in paths else [])
    sqlite = next((p for p in (EXAMPLES_SQLITE, INIT_SQLITE) if p in paths), None)
    out = [Source("sqlite", sqlite)] if sqlite else []
    if MASTER_JSON in paths:
        out.append(Source("template", MASTER_JSON))
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8",
                          check=True, stdin=subprocess.DEVNULL).stdout


def _version_key(tag: str) -> list:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", tag)]


def release_tags(pattern: str = "*") -> list[str]:
    return sorted((t for t in _git("tag").split() if RELEASE_TAG.match(t) and fnmatch.fnmatch(t, pattern)),
                  key=_version_key)


def tag_sources(tag: str) -> list[tuple[Source, str]]:
    """``(source, git blob id)`` of *tag*."""
    blobs = {}
    for line in _git("ls-tree", "-r", tag).splitlines():
        meta, path = line.split("\t", 1)
        blobs[path] = meta.split()[2]
    return [(source, blobs[source.path]) for source in pick_sources(blobs)]


# --- one source --------------------------------------------------------------------------------------------------

def _build(tag: str, source: Source, folder: Path) -> Path:
    """The definitions-only database of *source* at *tag* in *folder*."""
    from flextool.update_flextool.initialize_database import initialize_database

    from tests.update_flextool._raw_seed import build, git_show, seed_from_git
    from tests.update_flextool._rerun import quiet

    target = folder / "seed.sqlite"
    if source.kind == "sqlite":
        return build(seed_from_git(ROOT, tag, source.path, folder), target)
    data = json.loads(git_show(ROOT, tag, source.path).decode("utf-8"))
    if source.kind == "examples":
        data = {key: value for key, value in data.items() if key not in EXAMPLES_DATA_KEYS}
    template = folder / "template.json"
    template.write_text(json.dumps(data), encoding="utf-8")
    quiet(initialize_database, str(template), str(target))
    return target


def _description_diffs(path: Path) -> dict[tuple[str, str], str]:
    """``{(class, name): description}`` of the definitions of *path* whose description differs from the shipped
    schema's (definitions the schema lacks are left out)."""
    from flextool._resources import package_data_path

    from tests.update_flextool._rerun import snapshot

    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        shipped = {(row[0], row[1]): row[4] for row in json.load(f)["parameter_definitions"]}
    return {key: fields[3] for key, fields in snapshot(path)["definitions"].items()
            if key in shipped and fields[3] != shipped[key]}


def _triangle(chain: dict[int, Path], folder: Path) -> list[tuple]:
    from tests.update_flextool._rerun import copy_db, diff, migrate, restamp_file, snapshot

    found = []
    first = min(chain)
    for k in sorted(chain):
        for s in range(first, k):
            path = copy_db(chain[k], folder / "triangle.sqlite")
            try:
                restamp_file(path, s)
                migrate(path, up_to=k)
                differences = diff(snapshot(chain[k]), snapshot(path))
                if differences:
                    found.append((s, k, [str(x)[:200] for x in differences[:3]]))
            except Exception as exc:  # noqa: BLE001 - reported per case
                found.append((s, k, [f"{type(exc).__name__}: {str(exc)[:200]}"]))
            finally:
                path.unlink(missing_ok=True)
    return found


def run_source(job: tuple) -> dict:
    """Build, migrate and compare one ``(tags, source, with_data, triangle)`` job (in a worker process): *source* as
    the first of *tags* ships it (the others ship the same blob)."""
    tags, source, with_data, triangle = job
    tag = tags[0]
    logging.disable(logging.WARNING)
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.frozen import read_db_version_ro
    from flextool.update_flextool.list_default_repair import repair_list_defaults

    from tests.update_flextool._rerun import (already_applied_records, copy_db, diff, migrate, snapshot, url_of,
                                              version_of)
    from tests.update_flextool.test_rerun_convergence import _seed_data

    result = {"tag": tag, "tags": list(tags), "kind": source.kind, "path": source.path, "with_data": with_data}
    with tempfile.TemporaryDirectory() as work:
        folder = Path(work)
        try:
            seed = _build(tag, source, folder)
            result["from"] = read_db_version_ro(seed) if _has_version(seed) else None
            if with_data:
                # As migrate_database does before its first SpineDB open: list defaults stored as text (item 279)
                # repaired; the data is added through the current spinedb_api, which upgrades an older SpineDB schema.
                repair_list_defaults(seed)
                with open_database_mapping(url_of(seed), upgrade=True) as db:
                    has_base = bool(db.get_item("alternative", name="Base"))
                _seed_data(seed, base=not has_base)
            one_shot = copy_db(seed, folder / "one_shot.sqlite")
            started = time.perf_counter()
            with already_applied_records() as records:
                migrate(one_shot)
            result["seconds"] = round(time.perf_counter() - started, 2)
            stepwise = copy_db(seed, folder / "stepwise.sqlite")
            start = int(result["from"] or 0)
            chain = {start: copy_db(seed, folder / f"k{start}.sqlite")} if triangle else {}
            with already_applied_records() as step_records:
                for k in range(start + 1, FLEXTOOL_DB_VERSION + 1):
                    migrate(stepwise, up_to=k)
                    if triangle:
                        chain[k] = copy_db(stepwise, folder / f"k{k}.sqlite")
            result["version"] = version_of(one_shot)
            result["records"] = list(records) + list(step_records)
            result["drift"] = [str(x)[:300] for x in diff(snapshot(stepwise), snapshot(one_shot))]
            result["descriptions"] = {f"{c}.{n}": text for (c, n), text in _description_diffs(one_shot).items()}
            if triangle:
                result["triangle"] = _triangle(chain, folder)
        except Exception as exc:  # noqa: BLE001 - reported per source
            result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    return result


def _has_version(path: Path) -> bool:
    import sqlite3

    con = sqlite3.connect(path)
    try:
        return bool(con.execute("SELECT 1 FROM parameter_definition WHERE name = 'version'").fetchone())
    finally:
        con.close()


def _large_seconds() -> float:
    from flextool.update_flextool.db_migration import migrate_database

    from tests.update_flextool._rerun import quiet
    from tests.update_flextool.test_large_database_migration import _build_v69_large

    with tempfile.TemporaryDirectory() as work:
        url = f"sqlite:///{(Path(work) / 'large.sqlite').as_posix()}"
        _build_v69_large(url)
        started = time.perf_counter()
        quiet(migrate_database, url)
        return round(time.perf_counter() - started, 2)


def failed(result: dict) -> bool:
    return bool(result.get("error") or result.get("drift") or result.get("records") or result.get("triangle")
                or result.get("version") is not None and result.get("version") != _head_version())


def _head_version() -> int:
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    return FLEXTOOL_DB_VERSION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tags", default="*", help="fnmatch pattern of the release tags (default: all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--triangle", action="store_true", help="also every stamp s < k of the chain")
    parser.add_argument("--large", action="store_true", help="also time the large v69 database")
    parser.add_argument("--json", type=Path, help="write every result to this file")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    tags = release_tags(args.tags)
    by_blob: dict[tuple[Source, str], list[str]] = defaultdict(list)
    for tag in tags:
        for source, blob in tag_sources(tag):
            by_blob[(source, blob)].append(tag)
    jobs = [(tuple(tags_of), source, with_data, args.triangle) for (source, _blob), tags_of in by_blob.items()
            for with_data in (False, True)]
    print(f"{len(tags)} release tags, {len(by_blob)} distinct sources, {len(jobs)} migrations", flush=True)
    results = []
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(run_source, jobs):
            results.append(result)
            flag = "FAIL" if failed(result) else "ok"
            print(f"{flag:4} {result['tag']:10} {result['kind']:8} data={int(result['with_data'])} "
                  f"from={result.get('from')} {result.get('seconds', '-')} s descriptions="
                  f"{len(result.get('descriptions', {}))} {result.get('error', '')}", flush=True)
            for line in result.get("drift", [])[:3] + result.get("records", [])[:3] + [
                    str(case) for case in result.get("triangle", [])[:3]]:
                print(f"       {line[:250]}", flush=True)
    texts: dict[str, set[str]] = defaultdict(set)
    for result in results:
        for key, text in result.get("descriptions", {}).items():
            texts[key].add(text)
    print(f"\nDescriptions at {_head_version()} that differ from spinedb_schema.json: {len(texts)} definitions")
    for key in sorted(texts):
        tags_with = [t for t in tags if any(key in r.get("descriptions", {}) and t in r["tags"] for r in results)]
        print(f"  {key}: {len(texts[key])} text(s), in {len(tags_with)} tag(s) {tags_with[0]}..{tags_with[-1]}")
    bad = [r for r in results if failed(r)]
    timing = sorted(r["seconds"] for r in results if "seconds" in r)
    if timing:
        print(f"\nOne-shot migration: median {timing[len(timing) // 2]} s, max {timing[-1]} s")
    if args.large:
        print(f"Large v69 database: {_large_seconds()} s")
    print(f"Failures: {len(bad)}; wall time {round(time.perf_counter() - started)} s")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
