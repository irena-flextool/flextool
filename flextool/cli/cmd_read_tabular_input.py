import argparse
import os
from flextool.process_inputs.read_tabular_with_specification import TabularReader
from flextool.process_inputs.write_to_input_db import write_to_flextool_input_db


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument('target_db_url',help= "URL to FlexTool input database (e.g. sqlite:///input_data.sqlite)")

        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--tabular-file-path', help= "The file path of a FlexTool input file (either xlsx or ods).")
        input_group.add_argument('--csv-directory-path', help= "Input data as csv files in FlexTool format.")

        args = parser.parse_args()

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

        write_to_flextool_input_db(input_path, tabular_reader, args.target_db_url, input_type)



if __name__ == '__main__':
    main()
