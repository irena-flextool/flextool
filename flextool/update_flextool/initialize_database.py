import json
import argparse
from spinedb_api import import_data, DatabaseMapping

from flextool.update_flextool.export_database import keep_serialized_unparse


def initialize_database(json_template, database_name="new_database.sqlite"):

    if not database_name.endswith(".sqlite"):
        print("Give a name with .sqlite file extension")
        exit(-1)

    with open(json_template) as json_file:
        template = json.load(json_file)

    with DatabaseMapping('sqlite:///' + database_name, create=True) as new_db:
        # Remove any default alternatives created by spinedb_api (e.g. 'Base')
        # so they don't conflict with the template's own alternatives.
        for alt in new_db.find_alternatives():
            try:
                new_db.remove_alternative(name=alt['name'])
            except Exception:
                pass
        # ``keep_serialized_unparse`` handles both author styles:
        #   * raw scalars (``"no_method"``, ``None``, ``5``) used by the
        #     hand-maintained ``schemas/spinedb_schema.json`` and
        #     other static templates.
        #   * ``[json_str, type_str]`` pairs emitted by ``export_database``
        #     for canonical databases that carry full parameter values.
        (num, log) = import_data(new_db, unparse_value=keep_serialized_unparse, **template)
        print(str(num) + " imports made")
        print("Initialized " + database_name)
        new_db.commit_session("Initialized")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_filepath',help= "The filepath of the source JSON")
    parser.add_argument('db_filepath',help= "The filepath of the new database")
    args = parser.parse_args()
    initialize_database(args.json_filepath, args.db_filepath)