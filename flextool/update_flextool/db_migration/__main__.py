"""``python -m flextool.update_flextool.db_migration FILE`` migrates FILE."""
import argparse

from flextool.update_flextool.db_migration.runner import migrate_database


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="python -m flextool.update_flextool.db_migration")
    parser.add_argument('filename',help= "The filepath of the database to be migrated")
    args = parser.parse_args()
    migrate_database(args.filename)
