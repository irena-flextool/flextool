"""``tools/migration_sweep.py`` picks the schema sources a release tag ships (plan §5 table).

The path sets are those of the first tag of each range (listed with ``git ls-tree`` when the table was made; the
tests need no git history).  Other files of a tag (here ``README.md``, a how-to database) are never a source.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "migration_sweep.py"
_spec = importlib.util.spec_from_file_location("migration_sweep", _SCRIPT)
sweep = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("migration_sweep", sweep)  # its dataclass looks its module up
_spec.loader.exec_module(sweep)
Source = sweep.Source

OTHER = {"README.md", "how_to/example.sqlite"}
CASES = {
    # v3.0.0 - v3.1.x: the shipped database only.
    "v3.0.0": ({"Init.sqlite"}, [Source("sqlite", "Init.sqlite")]),
    # v3.2.0 - v3.8.x: the database, then the master template.
    "v3.2.0": ({"Init.sqlite", "version/flextool_template_master.json"},
               [Source("sqlite", "Init.sqlite"), Source("template", "version/flextool_template_master.json")]),
    # v3.9.0 - v3.41.2: the examples database replaces Init.sqlite.
    "v3.9.0": ({"templates/examples.sqlite", "version/flextool_template_master.json"},
               [Source("sqlite", "templates/examples.sqlite"),
                Source("template", "version/flextool_template_master.json")]),
    # v3.41.3 - v3.43.0: no shipped database; the packaged master template and the canonical examples JSON.
    "v3.41.3": ({"flextool/version/flextool_template_master.json",
                 "flextool/textual_templates/canonical_databases/templates_examples.json"},
                [Source("template", "flextool/version/flextool_template_master.json"),
                 Source("examples", "flextool/textual_templates/canonical_databases/templates_examples.json")]),
    # v3.43.1 - v4.6.3: the schema JSON only (the canonical examples moved next to it and are not a source).
    "v3.43.1": ({"flextool/schemas/spinedb_schema.json", "flextool/schemas/canonical_databases/templates_examples.json"},
                [Source("template", "flextool/schemas/spinedb_schema.json")]),
}


@pytest.mark.parametrize("tag", list(CASES))
def test_pick_sources(tag: str) -> None:
    paths, expected = CASES[tag]
    assert sweep.pick_sources(paths | OTHER) == expected


def test_a_tag_without_a_schema_has_no_source() -> None:
    assert sweep.pick_sources(OTHER) == []


@pytest.mark.parametrize(("tag", "release"), [("v3.0.0", True), ("v4.0.0b26", True), ("v4.0.0a2", True),
                                               ("archive/master-v3", False), ("parked/x", False), ("v4", False)])
def test_release_tags(tag: str, release: bool) -> None:
    assert bool(sweep.RELEASE_TAG.match(tag)) == release
