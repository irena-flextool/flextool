"""Write the committed release seeds of the migration tests from the shipped databases of old releases.

Each seed is the definitions-only raw-row JSON of one shipped ``.sqlite`` (``tests/update_flextool/_raw_seed.py``):
its tables and their ``CREATE`` SQL, the class, definition, value list, alternative, scenario and commit rows,
no entities and no values, the commit users blanked.  Needs the release tags in the local clone; the tests never do.

Usage::

    python tools/make_release_seeds.py          # write the seeds
    python tools/make_release_seeds.py --check  # exit 1 when a committed seed differs from its tag
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.update_flextool._raw_seed import read_seed, seed_from_git, write_seed  # noqa: E402

#: (tag, path in the tag): the four committed seeds (plan §5).
RELEASE_SEEDS = (
    ("v3.0.0", "Init.sqlite"),
    ("v3.10.0", "templates/examples.sqlite"),
    ("v3.22.0", "templates/examples.sqlite"),
    ("v3.33.0", "templates/examples.sqlite"),
)
SEED_DIR = ROOT / "tests" / "update_flextool" / "data" / "release_seeds"


def seed_path(tag: str) -> Path:
    return SEED_DIR / f"{tag.replace('.', '_')}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    differs = []
    with tempfile.TemporaryDirectory() as work:
        for tag, path in RELEASE_SEEDS:
            seed = seed_from_git(ROOT, tag, path, Path(work))
            target = seed_path(tag)
            if args.check:
                if not target.is_file() or read_seed(target) != seed:
                    differs.append(target.name)
            else:
                write_seed(seed, target)
                print(f"{tag}:{path} -> {target.relative_to(ROOT)}")
    if differs:
        print("differs from its tag:", ", ".join(differs))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
