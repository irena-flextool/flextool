import argparse
from flextool.migrate_database import migrate_database


def migrage_database(db_name):
    migrate_database(db_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help= "Filepath of the database, absolute or relative to flextool folder")
    args = parser.parse_args()
    migrate_database(args.filepath)