"""``flextool-migrate-database``: migrate a FlexTool input database to the current version.

A frozen database (``flextool-freeze``) is not migrated in place: the command exits with code 2 unless it is given
``--copy-to PATH`` (migrate a copy; the original stays byte-identical) or ``--allow-migrate-frozen`` (migrate the
frozen file itself; its marker is kept with the new version).
"""
import argparse
import sys

from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.frozen import FrozenDatabaseError, migrate_copy, read_frozen, sqlite_path_from

#: The exit code of a refused frozen database.
EXIT_FROZEN = 2


def build_parser() -> argparse.ArgumentParser:
    """The ``flextool-migrate-database`` command line."""
    parser = argparse.ArgumentParser(
        prog="flextool-migrate-database",
        description="Migrate a FlexTool input database to the current FlexTool data version.",
    )
    parser.add_argument('filepath', help="Filepath of the database, absolute or relative to flextool folder")
    frozen = parser.add_mutually_exclusive_group()
    frozen.add_argument('--copy-to', metavar='PATH',
                        help="Copy the database to PATH and migrate the copy; the original is not changed. "
                             "For a frozen database (the copy is not frozen and records what it was migrated from).")
    frozen.add_argument('--allow-migrate-frozen', action='store_true',
                        help="Migrate a frozen database in place; its frozen marker is kept with the new version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.copy_to:
        source = sqlite_path_from(args.filepath)
        if source is None or not source.is_file():
            print(f"No sqlite file at {args.filepath}", file=sys.stderr)
            return 1
        destination = sqlite_path_from(args.copy_to)
        try:
            marker = migrate_copy(source, destination)
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Migrated copy written to {destination} (from '{marker.label}', version {marker.version}); "
              f"{source} was not changed.")
        return 0
    try:
        migrate_database(args.filepath, allow_frozen=args.allow_migrate_frozen)
    except FrozenDatabaseError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FROZEN
    if args.allow_migrate_frozen:
        path = sqlite_path_from(args.filepath)
        info = read_frozen(path) if path is not None else None
        if info is not None:
            print(f"The database is still frozen: {info.describe()}.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
