"""``python -m flextool.docs_build --out DIR``: write the generated documentation pages into ``DIR``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flextool.docs_build import (
    REPO_ROOT,
    PrivateNameError,
    default_private_names_path,
    write_pages,
)
from flextool.docs_build.catalogue import unmatched_bullets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flextool.docs_build",
        description="Write the formulation page and the results catalogue, generated from the installed code.",
    )
    parser.add_argument("--out", type=Path, required=True, help="folder to write the pages into")
    parser.add_argument("--docs-dir", type=Path, default=REPO_ROOT / "docs",
                        help="the Markdown documentation they are built with (default: docs/ of this clone)")
    parser.add_argument("--private-names", type=Path, default=None,
                        help="list of private names to check the pages against (default: .private_names at the "
                             "repository root, skipped when absent)")
    args = parser.parse_args(argv)
    private_names = args.private_names if args.private_names is not None else default_private_names_path()
    try:
        written = write_pages(args.out, args.docs_dir, private_names)
    except PrivateNameError as exc:
        print(exc, file=sys.stderr)
        return 1
    for bullet in unmatched_bullets(args.docs_dir):
        print(f"results.md ({bullet.topic}): names no result table: {bullet.text}", file=sys.stderr)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
