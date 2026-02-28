import argparse
from flextool.update_flextool.db_migration import migrate_database


def migrage_database(db_name):
    migrate_database(db_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help="Filepath of the database, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)


if __name__ == '__main__':
    main()
