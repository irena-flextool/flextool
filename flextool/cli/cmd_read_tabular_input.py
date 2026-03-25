import argparse
import os
from pathlib import Path
from flextool.process_inputs.read_tabular_with_specification import TabularReader
from flextool.process_inputs.write_to_input_db import write_to_flextool_input_db


def _ensure_target_db_exists(target_db_url: str) -> None:
    """Initialize the target database from the FlexTool template if it does not exist.

    The ``write_to_flextool_input_db`` function opens the database with
    ``create=False`` and expects a valid FlexTool schema (including a
    *version* parameter).  When the caller asks us to write into a brand-new
    sqlite file (e.g. during xlsx-to-sqlite conversion) the file won't exist
    yet, so we must create and initialise it first.
    """
    if not target_db_url.startswith("sqlite:///"):
        return  # Only handle local sqlite files

    db_path = target_db_url.replace("sqlite:///", "", 1)
    if os.path.exists(db_path):
        return  # Already exists -- nothing to do

    # Make sure the parent directory exists
    parent = Path(db_path).parent
    parent.mkdir(parents=True, exist_ok=True)

    # Locate the FlexTool master template
    flextool_root = Path(__file__).resolve().parent.parent.parent
    json_template = flextool_root / "version" / "flextool_template_master.json"
    if not json_template.exists():
        raise FileNotFoundError(
            f"FlexTool template not found at {json_template}. "
            "Cannot initialize the target database."
        )

    from flextool.update_flextool.initialize_database import initialize_database

    print(f"Target database does not exist. Initializing: {db_path}")
    initialize_database(str(json_template), db_path)


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument('target_db_url',help= "URL to FlexTool input database (e.g. sqlite:///input_data.sqlite)")

        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--tabular-file-path', help= "The file path of a FlexTool input file (either xlsx or ods).")
        input_group.add_argument('--csv-directory-path', help= "Input data as csv files in FlexTool format.")
        parser.add_argument('--migration-follows', action='store_true',
                            help="Accept a version mismatch because migration will run after import")

        args = parser.parse_args()

        # Initialize the target database if it doesn't exist yet
        _ensure_target_db_exists(args.target_db_url)

        # Get the path to import_excel_input.json relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "..", "import_excel_input.json")
        tabular_reader = TabularReader(json_path)

        if args.tabular_file_path:
            input_path = args.tabular_file_path
            input_type = 'excel'
        elif args.csv_directory_path:
            input_path = args.csv_directory_path
            input_type = 'csv'
        else:
            raise ValueError("Must provide either tabular_file_path or csv_directory_path")

        write_to_flextool_input_db(input_path, tabular_reader, args.target_db_url, input_type,
                                   migration_follows=args.migration_follows)



if __name__ == '__main__':
    main()
