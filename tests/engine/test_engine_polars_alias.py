"""``flextool.engine_polars`` is the old name of ``flextool.engine``, kept as an alias.

The alias must expose the public API, and nothing in the repository may
use it: otherwise it would hide imports the rename missed.
"""

import re
from pathlib import Path

import flextool.engine as engine
import flextool.engine_polars as alias

REPO = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
ALIAS_DIR = REPO / "flextool" / "engine_polars"
OLD_NAME = re.compile(
    r"flextool[./]engine_polars\b|from\s+flextool\s+import\s.*\bengine_polars\b"
)


def test_alias_exposes_the_public_api():
    assert sorted(alias.__all__) == sorted(engine.__all__)
    for name in engine.__all__:
        assert getattr(alias, name) is getattr(engine, name), name


def test_provider_keys_import():
    from flextool.engine.preprocess import _provider_keys as current
    from flextool.engine_polars import _provider_keys

    assert _provider_keys is current


def test_nothing_uses_the_old_name():
    scanned = 0
    offenders = []
    for top in ("flextool", "tests", "tools"):
        for path in (REPO / top).rglob("*.py"):
            path = path.resolve()
            if path == THIS_FILE or ALIAS_DIR in path.parents:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                if OLD_NAME.search(line):
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert scanned > 500, f"scanned only {scanned} files"
    assert not offenders, "use flextool.engine instead:\n" + "\n".join(offenders)
