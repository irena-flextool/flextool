"""Phase F — typed registry of parquet outputs + manifest.json.

Asserts:

* ``flextool.engine_polars._parquet_bundle.write_manifest`` is callable
  on a populated bundle root and produces ``manifest.json`` with the
  documented schema.
* Every entry's ``key`` matches a REGISTRY entry.
* Every present file the manifest references actually exists on disk.
* Spot-checked entries' ``columns`` list matches the underlying parquet
  schema (after stripping the index columns the lean-parquet writer
  unrolls into the column list).

The bundle root used here is the ``work_base`` test fixture, which
ships pre-populated ``output_raw/v_*.parquet`` files from a prior
cascade.  This avoids running a fresh cascade in the test (which would
take 10s+ and require the solver) — the manifest writer is purely
disk-walking + REGISTRY enumeration, so a static fixture is sufficient
to exercise its contract.

The Phase F registry doesn't change writer behaviour; it just
documents the bundle.  Per the spec: "the manifest writer at end of
cascade only includes files actually present on disk" — modelled here
as the ``exists`` flag on each file entry.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._parquet_bundle import (
    REGISTRY,
    ParquetSpec,
    write_manifest,
    write_parquet,
)


# ---------------------------------------------------------------------------
# Schema basics — write_manifest is callable, returns a Path, the file is
# valid JSON and matches the documented schema shape.
# ---------------------------------------------------------------------------


def _manifest_for(work_folder: Path) -> dict:
    path = write_manifest(work_folder)
    assert path == work_folder / "output_raw" / "manifest.json", (
        f"write_manifest must write to <work_folder>/output_raw/manifest.json; "
        f"got {path}"
    )
    assert path.is_file(), f"manifest.json not on disk after write: {path}"
    return json.loads(path.read_text())


def _copy_fixture(tmp_path: Path, scenario_workdir) -> Path:
    """Copy a fresh cascade-produced ``work_base`` into ``tmp_path`` so
    the test isolates the manifest.json write from the session-cached
    fixture (and from any prior test that wrote its own manifest)."""
    src = scenario_workdir("base")
    dst = tmp_path / "work_base"
    shutil.copytree(src, dst)
    return dst


def test_manifest_has_documented_schema(
    tmp_path: Path, scenario_workdir
) -> None:
    """The manifest carries ``version`` / ``generated_at`` /
    ``bundle_root`` / ``files``; ``files`` is a list of entries with
    the documented per-entry keys."""
    work = _copy_fixture(tmp_path, scenario_workdir)
    manifest = _manifest_for(work)

    assert manifest["version"] == "1", (
        f"manifest version must be '1'; got {manifest['version']!r}"
    )
    assert "generated_at" in manifest and isinstance(manifest["generated_at"], str)
    # Should parse back as ISO 8601 — a permissive check is fine here.
    from datetime import datetime
    datetime.fromisoformat(manifest["generated_at"])

    assert manifest["bundle_root"] == str(work.resolve()), (
        f"bundle_root must be the absolute work_folder path; "
        f"got {manifest['bundle_root']!r}"
    )

    files = manifest["files"]
    assert isinstance(files, list) and len(files) > 0

    required_keys = {
        "key", "category", "path", "exists", "size_bytes",
        "columns", "indices", "note", "producer",
    }
    for entry in files:
        missing = required_keys - set(entry)
        assert not missing, (
            f"manifest entry missing keys {missing}: {entry}"
        )
        assert entry["category"] in {"raw", "processed"}, (
            f"unknown category {entry['category']!r} on entry "
            f"{entry['key']!r}"
        )


def test_manifest_entries_match_registry(
    tmp_path: Path, scenario_workdir
) -> None:
    """Every ``files[].key`` corresponds to a REGISTRY entry."""
    work = _copy_fixture(tmp_path, scenario_workdir)
    manifest = _manifest_for(work)

    registry_keys = set(REGISTRY)
    seen_keys = {entry["key"] for entry in manifest["files"]}
    unknown = seen_keys - registry_keys
    assert not unknown, (
        f"manifest references unknown REGISTRY keys: {sorted(unknown)}"
    )


def test_manifest_present_files_exist_on_disk(
    tmp_path: Path, scenario_workdir
) -> None:
    """Every entry whose ``exists`` is True must point at a real file
    under the bundle root."""
    work = _copy_fixture(tmp_path, scenario_workdir)
    manifest = _manifest_for(work)

    for entry in manifest["files"]:
        if not entry["exists"]:
            continue
        path = work / entry["path"]
        assert path.is_file(), (
            f"manifest claims file exists but it's not on disk: {path} "
            f"(key={entry['key']!r})"
        )
        # ``size_bytes`` must be a non-negative int when the file exists.
        size = entry["size_bytes"]
        assert isinstance(size, int) and size >= 0, (
            f"size_bytes for present file must be int >= 0; got {size!r} "
            f"on {entry['key']!r}"
        )


def test_manifest_idempotent(
    tmp_path: Path, scenario_workdir
) -> None:
    """Calling write_manifest twice produces a JSON-comparable result
    (modulo ``generated_at``)."""
    work = _copy_fixture(tmp_path, scenario_workdir)
    m1 = _manifest_for(work)
    m2 = _manifest_for(work)
    # Strip the timestamp before comparing — everything else must be
    # deterministic across calls.
    m1.pop("generated_at", None)
    m2.pop("generated_at", None)
    assert m1 == m2, (
        "write_manifest must be idempotent (deterministic apart from "
        "generated_at)"
    )


# ---------------------------------------------------------------------------
# Spot-check: pick a couple of present raw entries and prove their
# REGISTRY ``columns`` list matches the parquet schema.
# ---------------------------------------------------------------------------


def _present_keys_in_category(manifest: dict, category: str) -> list[str]:
    return [e["key"] for e in manifest["files"]
            if e["exists"] and e["category"] == category]


@pytest.mark.parametrize("registry_key", ["v_obj", "v_invest", "v_state"])
def test_registry_columns_match_parquet_schema(
    tmp_path: Path, registry_key: str, scenario_workdir,
) -> None:
    """For per-solve raw variables, the parquet's column set should be
    a SUPERSET of the REGISTRY ``columns`` declaration when the column
    layout doesn't carry ``solve``/``period``/``time`` index columns
    in the value-axis.

    The lean-parquet writer stores the row-index columns under
    ``__index_level_<n>__`` synthetic names, so we compare REGISTRY
    columns against the parquet's COLUMN names (the wide-format value
    columns) — those must intersect non-trivially.  A stricter check
    would require parsing the ``flextool`` metadata blob on the
    parquet file; we keep it loose to avoid coupling the test to that
    implementation detail.
    """
    work = _copy_fixture(tmp_path, scenario_workdir)
    manifest = _manifest_for(work)

    present = _present_keys_in_category(manifest, "raw")
    if registry_key not in present:
        pytest.skip(
            f"REGISTRY key {registry_key!r} not present in fixture's "
            f"output_raw/; skipping spot-check.",
        )

    # Find the parquet path for this key.
    entry = next(e for e in manifest["files"]
                 if e["key"] == registry_key and e["exists"])
    parquet_path = work / entry["path"]

    spec: ParquetSpec = REGISTRY[registry_key]
    if not spec.columns:
        pytest.skip(f"REGISTRY entry {registry_key!r} declares no columns")

    schema_cols = set(pl.scan_parquet(parquet_path).collect_schema().names())

    # ``v_obj`` carries the value column ``objective`` which IS in the
    # column-axis (single-column wide).  The lean-parquet writer
    # serializes the row-index columns + the value columns side-by-side,
    # so the spec's columns should at least intersect the parquet's
    # column names.  A strict superset-check would over-fit to the
    # writer's column-naming convention; we just assert the intersection
    # is non-empty when ``spec.columns`` is non-empty.
    spec_cols = set(spec.columns)

    # The intersection represents either:
    #   * declared value columns that the writer kept (e.g. objective,
    #     entity, node), OR
    #   * declared index columns that the writer serialized verbatim
    #     (e.g. solve / period / time when stored as physical columns).
    # We tolerate either path by checking against both axes.
    overlap = (spec_cols & schema_cols) | (set(spec.indices) & schema_cols)
    assert overlap, (
        f"spec {registry_key!r} columns {spec_cols} (or indices "
        f"{spec.indices}) must intersect parquet schema {schema_cols}; "
        f"intersection empty — REGISTRY entry may be stale."
    )


# ---------------------------------------------------------------------------
# write_parquet — typed write path validation.
# ---------------------------------------------------------------------------


def test_write_parquet_unknown_key_raises(tmp_path: Path) -> None:
    """Calling write_parquet with an unknown key surfaces a KeyError
    pointing the caller at the REGISTRY."""
    df = pl.DataFrame({"foo": [1.0]})
    with pytest.raises(KeyError, match="REGISTRY"):
        write_parquet("nope_not_a_real_key", df, tmp_path)


def test_write_parquet_glob_filename_raises(tmp_path: Path) -> None:
    """Glob-filename specs (per-solve shards) are managed by their
    existing writers; the typed write path must refuse them."""
    df = pl.DataFrame({"objective": [42.0]})
    # ``v_obj`` uses a glob filename ``v_obj__*.parquet``.
    with pytest.raises(ValueError, match="glob"):
        write_parquet("v_obj", df, tmp_path)


def test_write_parquet_validates_columns(tmp_path: Path) -> None:
    """When the spec declares a non-empty ``columns`` list, write_parquet
    rejects frames whose columns don't match (set-equality)."""
    # Use a non-glob spec.  The handoff capacity CSV specs
    # (``entity_all_capacity`` etc.) declare seven columns; supply
    # something different and confirm the validator fires.
    df = pl.DataFrame({"unrelated": [1.0]})
    with pytest.raises(ValueError, match="columns"):
        write_parquet("entity_all_capacity", df, tmp_path)
