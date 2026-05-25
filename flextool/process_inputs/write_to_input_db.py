import os
from spinedb_api import DatabaseMapping, from_database, SpineDBAPIError, Asterisk
from spinedb_api.exception import NothingToCommit
from flextool.update_flextool import FLEXTOOL_DB_VERSION

flextool_db_version = FLEXTOOL_DB_VERSION

def write_to_flextool_input_db(input_path, tabular_reader, target_db_url, input_type='excel',
                               migration_follows: bool = False):
    """Write tabular data to FlexTool input database, reading one sheet/file at a time.

    Args:
        input_path: Path to Excel/ODS file or directory containing CSV files
        tabular_reader: TabularReader instance with loaded specification
        target_db_url: URL to target SQLite database
        input_type: Either 'excel' or 'csv' to determine reading method
        migration_follows: If True, accept a version mismatch because
            the caller will migrate the database after import

    This function reads sheets/files one at a time to minimize memory usage.
    """
    if target_db_url.startswith('sqlite://') or target_db_url.startswith('http://'):
        db_url = target_db_url
    elif os.path.exists(target_db_url) and target_db_url.endswith(".sqlite"):
        db_url = 'sqlite:///' + target_db_url
    else:
        print("No sqlite file at " + target_db_url)
        exit(-1)

    with DatabaseMapping(db_url, create = False, upgrade = True) as db:
        sq = db.object_parameter_definition_sq
        version_parameter = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "version").one_or_none()
        if version_parameter is None:
            #if no version assume version 0
            print(f"No FlexTool database version found. Needs to be proper FlexTool database at version: {flextool_db_version}")
            exit(-1)
        else:
            version = from_database(version_parameter.default_value, version_parameter.default_type)
            if version == flextool_db_version:
                print(f"Valid FlexTool input database with correct version: {flextool_db_version}")
            elif migration_follows:
                print(f"FlexTool input database version: {version} (will be migrated to {flextool_db_version})")
            else:
                print(f"Wrong FlexTool input database version: {version}. Should be: {flextool_db_version}")
                exit(-1)

        # Get list of tables to process
        selected_tables = tabular_reader.get_selected_table_names()
        print(f"\nProcessing {len(selected_tables)} selected tables")

        # Purge the database empty of data times
        try:
            db.remove_entity(id=Asterisk)
            db.remove_alternative(id=Asterisk)
            db.remove_scenario(id=Asterisk)
            db.remove_parameter_value(id=Asterisk)
            db.commit_session("Purged the db")
        except Exception as e:
            raise RuntimeError(f"Failed to purge the input database before writing. {e}")

        # Process each table one at a time
        for table_name in selected_tables:
            print(f"\nProcessing table {table_name + ': ':<30}", end="")

            # Read the sheet/file that contains the data
            try:
                if input_type == 'excel':
                    raw_df, column_types = tabular_reader.read_excel_sheet(input_path, table_name)
                elif input_type == 'csv':
                    csv_file = os.path.join(input_path, f"{table_name}.csv")
                    if not os.path.exists(csv_file):
                        print(f"  Warning: CSV file not found: {csv_file}")
                        continue
                    raw_df, column_types = tabular_reader.read_csv_file(csv_file)
                else:
                    raise ValueError(f"Unknown input_type: {input_type}")

                # Skip if sheet not found in the file
                if raw_df is None:
                    print(f"Skipping {table_name} - no sheet found")
                    continue

                # Skip if no data (e.g., sheet not selected)
                if raw_df.empty:
                    print(f"  Skipping {table_name} - no data")
                    continue

            except Exception as e:
                print(f"  Error reading table '{table_name}': {e}")
                continue  # Continue to next table on read error

            # Get mappings for this sheet
            mappings = tabular_reader.get_table_mappings(table_name)
            if not mappings:
                raise ValueError(f"No specification found for sheet {table_name}")

            table_options = tabular_reader.get_table_options(table_name)
            if 'row' in table_options:
                raw_df = raw_df.iloc[table_options['row']:]
            if 'column' in table_options:
                raw_df = raw_df.iloc[table_options['column']:]

            # Process each mapping in the table
            for mapping_name in mappings.keys():
                print(f"{mapping_name:>25}", end="")

                mapping_info = tabular_reader._parse_mapping(mappings[mapping_name])
                if not mapping_info:
                    continue

                # Read data and process the dataframe
                try:
                    (data_df, ent_zip_list, ent_act_zip, scen_array, scen_alt_df) = tabular_reader._extract_data(raw_df, mapping_info, table_options, column_types, table_name, mapping_name)
                except Exception as e:
                    print(f"\n  Error processing mapping '{mapping_name}': {e}")
                    continue  # Continue to next mapping on processing error

                if ent_zip_list:
                    tabular_reader._add_entities(ent_zip_list, db, table_name, mapping_name)

                # Add alternatives to the database
                if data_df is not None:
                    alt_array = data_df.columns.get_level_values('Alternative').unique().to_list()
                    tabular_reader._add_alternatives(alt_array, db, table_name, mapping_name)

                # Write re-organised dataframe to database
                if data_df is not None:
                #try:
                    # check_type:
                    value_type = mapping_info['rules'].get('ParameterValueType')
                    if value_type:
                        value_type = value_type['value']
                    else:
                        value_type = 'constant'
                    tabular_reader._add_parameters(data_df, db, table_name, mapping_name, value_type)

                # Ensure alternatives referenced by entity_alternatives exist
                if ent_act_zip is not None:
                    ent_act_list = list(ent_act_zip)
                    ea_alts = list({alt for _cls, _ent, alt, _act in ent_act_list})
                    if ea_alts:
                        tabular_reader._add_alternatives(ea_alts, db, table_name, mapping_name)
                    tabular_reader._add_entity_alternatives(ent_act_list, db, table_name, mapping_name)

                # Add scenarios to the database
                if scen_array is not None:
                    for scen in scen_array:
                        try:
                            db.add_scenario(name=scen)
                        except Exception as e:
                            raise SpineDBAPIError(f'Could not add scenario {scen} to the database: {e}')

                # Add scenario_alternatives to the database
                if scen_alt_df is not None:
                    tabular_reader._add_scenario_alternatives(scen_alt_df, db, table_name, mapping_name)

        # Commit the changes
        try:
            db.commit_session(f"Imported data from mapping '{mapping_name}'")
            tabular_reader.logger.info(f"Successfully committed data and items from mapping {mapping_name}")
        except NothingToCommit:
            pass
        except Exception as e:
            raise SpineDBAPIError(f"Could not commit data and items based on {mapping_name}: {e}")

    print("\nAll data processed successfully!    (* = No data found for the mapping)")
