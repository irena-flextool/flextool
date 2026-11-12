"""``flextool-freeze``: mark a FlexTool input database as frozen, remove the mark, or show it.

A frozen database is never migrated in place without the user saying so (``flextool-migrate-database`` refuses it
unless given ``--copy-to`` or ``--allow-migrate-frozen``; the FlexTool window asks).  Only a database at the current
FlexTool data version can be frozen.
"""
from __future__ import annotations

import argparse
import sys

from flextool.update_flextool.frozen import (
    MAX_LABEL_LENGTH,
    FreezeError,
    freeze,
    read_db_version_ro,
    read_frozen,
    read_migrated_from,
    sqlite_path_from,
    unfreeze,
)


def build_parser() -> argparse.ArgumentParser:
    """The ``flextool-freeze`` command line."""
    parser = argparse.ArgumentParser(
        prog="flextool-freeze",
        description="Freeze a FlexTool input database (it is then never migrated in place without consent), "
                    "unfreeze it, or show its frozen marker.",
    )
    parser.add_argument("database", help="The .sqlite file (a path or a sqlite:/// URL)")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--label", help=f"Freeze the database with this label (at most {MAX_LABEL_LENGTH} "
                                        "characters, no '|'), e.g. \"UAE reference 1.0\"")
    action.add_argument("--unfreeze", action="store_true", help="Remove the frozen marker")
    action.add_argument("--status", action="store_true", help="Print the frozen marker and the data version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = sqlite_path_from(args.database)
    if path is None or not path.is_file():
        print(f"No sqlite file at {args.database}", file=sys.stderr)
        return 1
    if args.status:
        info = read_frozen(path)
        version = read_db_version_ro(path)
        print(f"{path}: FlexTool data version {version if version is not None else 'unknown'}")
        if info is None:
            print("Not frozen.")
        else:
            print(f"Frozen: {info.describe()}.")
        origin = read_migrated_from(path)
        if origin is not None:
            print(f"Migrated copy of '{origin.label}' (version {origin.version}, sha256 {origin.sha256}).")
        return 0
    if args.unfreeze:
        info = unfreeze(path)
        print("Was not frozen; nothing changed." if info is None else f"Unfrozen (was {info.describe()}).")
        return 0
    try:
        info = freeze(path, args.label)
    except FreezeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Frozen: {info.describe()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
