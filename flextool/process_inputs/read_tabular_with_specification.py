import pandas as pd
import numpy as np
import json
from python_calamine import load_workbook
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import argparse
import re
from spinedb_api import to_database, Map, Array, SpineDBAPIError, dataframes
from spinedb_api.exception import NothingToCommit


class TabularReader:
    """Reads CSV, Excel (.xlsx), or ODS (.ods) files using specification mappings.

    This class handles:
    - Loading and parsing specification files
    - Reading data from CSV, Excel, and ODS formats
    - Applying type conversions and transformations based on specifications
    """

    def __init__(self, spec_file_path: str):
        """Initialize the TabularReader with a specification file.

        Args:
            spec_file_path: Path to the JSON specification file
        """
        self.spec_file_path = Path(spec_file_path)
        self.specifications = self._load_specifications()
        self.handler = logging.StreamHandler()
        self.handler.terminator = ""
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(self.handler)
        self.alternatives_added = set()
        self.entities_added = {}
        self.entity_alternatives_added = {}

    # ==================== Specification Methods ====================

    def _load_specifications(self) -> Dict[str, Any]:
        """Load the specification file."""
        with open(self.spec_file_path, 'r') as f:
            return json.load(f)

    def get_source_type(self) -> str:
        """Get the source type from specifications (ExcelReader or DatapackageReader)."""
        return self.specifications.get("source_type", "DatapackageReader")

    def get_table_names(self) -> List[str]:
        """Get all table names defined in the specification."""
        return list(self.specifications.get("tables", {}).keys())

    def get_selected_table_names(self) -> List[str]:
        """Get only the selected table names."""
        all_tables = self.get_table_names()
        selected_tables = self.specifications.get("selected_tables", [])
        if selected_tables:
            return [t for t in all_tables if t in selected_tables]
        return all_tables

    def get_table_mappings(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get specification for a specific table (CSV file or Excel sheet)."""
        # Normalize table name (remove .csv extension if present)
        table_key = table_name.replace('.csv', '').lower()
        table_mappings = self.specifications.get("tables", {}).get(table_key, {}).get("mappings", {})
        return table_mappings

    def get_table_options(self, table_name: str) -> Dict[str, Any]:
        """Get table options for a table."""
        table_key = table_name.replace('.csv', '').lower()
        table_options = self.specifications.get("tables", {}).get(table_key, {}).get("options", {})
        return table_options

    def get_table_column_types(self, table_name: str) -> Dict[str, str]:
        """Get column type specifications for a table."""
        table_key = table_name.replace('.csv', '').lower()
        table_column_types = self.specifications.get("tables", {}).get(table_key, {}).get("column_types", {})
        return table_column_types

    def get_table_row_types(self, table_name: str) -> Dict[str, str]:
        """Get row type specifications for a table."""
        table_key = table_name.replace('.csv', '').lower()
        table_row_types = self.specifications.get("tables", {}).get(table_key, {}).get("row_types", {})
        return table_row_types

    def is_table_selected(self, table_name: str) -> bool:
        """Check if a table is selected for processing."""
        table_key = table_name.replace('.csv', '').lower()
        selected_tables = self.specifications.get("selected_tables", [])
        if not selected_tables:
            return True  # If no selection specified, all tables are selected
        return table_key in selected_tables

    def _add_hidden_map_type_to_index(self, map_type, value, df_index) -> pd.Index:
                if isinstance(df_index, pd.MultiIndex):
                    # Get all arrays from existing multi-index
                    arrays = [df_index.get_level_values(i).to_list() for i in range(df_index.nlevels)]
                    arrays.append([value] * len(df_index))
                    names = list(df_index.names) + [map_type]
                    return pd.MultiIndex.from_arrays(arrays, names=names)
                else:
                    return pd.MultiIndex.from_arrays(
                        [df_index.values, [value] * len(df_index)],
                        names=[df_index.name, map_type]
                    )

    # ==================== Add to database methods ====================

    def _add_entities(self, ent_set_list, db_map, table_name, mapping_name):
        for ent_set in ent_set_list:
            for ent_class, ent in ent_set:
                key = (ent_class, ent)
                if key not in self.entities_added:
                    self.entities_added[key] = {
                        'entity_class_name': ent_class,
                        'entity_byname': ent
                    }
                    try:
                        db_map.add_entity(entity_class_name=ent_class,
                            entity_byname=ent)
                    except Exception as e:
                        raise SpineDBAPIError(f'Could not add entity {ent} from class {ent_class} based on table.mapping {table_name}.{mapping_name} to the database: {e}')

    def _add_alternatives(self, alt_array, db_map, table_name, mapping_name):
        for alt in alt_array:
            if alt not in self.alternatives_added:
                self.alternatives_added.add(alt)
                try:
                    db_map.add_alternative(name=alt)
                except Exception as e:
                    raise SpineDBAPIError(f'Could not add alternative {alt} based on table.mapping {table_name}.{mapping_name} to the database: {e}')

    def _add_entity_alternatives(self, ent_act_zip, db_map, table_name, mapping_name):
        for ent_class, ent, alt, value in ent_act_zip:
            key = (ent_class, ent, alt)
            if key not in self.entity_alternatives_added:
                self.entity_alternatives_added[key] = {
                    'entity_class_name': ent_class,
                    'entity_byname': ent,
                    'alternative': alt
                }
                try:
                    db_map.add_entity_alternative(entity_class_name=ent_class,
                        entity_byname=ent,
                        alternative_name=alt,
                        active=bool(value)),
                except Exception as e:
                    raise SpineDBAPIError(f'Could not add entity_alternative {ent}, {alt} from class {ent_class} based on table.mapping {table_name}.{mapping_name} to the database: {e}')

    def _add_scenario_alternatives(self, scen_alt_df, db_map, table_name, mapping_name):
        # First, add any missing alternatives
        alternatives = scen_alt_df.stack().unique()
        self._add_alternatives(alternatives, db_map, table_name, mapping_name)
        # Add scenario alternatives
        for (scen, alts) in scen_alt_df.items():
            alts_list = alts.dropna().to_list()
            for i, alt in enumerate(alts_list):
                try:
                    db_map.add_scenario_alternative(scenario_name=scen,
                        alternative_name=alt,
                        rank=i)
                except Exception as e:
                    raise SpineDBAPIError(f'Could not add scenario {scen} - alternative {alt} to the database: {e}')

    def _add_parameters(self, data_df: pd.DataFrame, db_map, table_name, mapping_name: str, value_type: str) -> None:
        """Add extracted parameter data to the Spine database.

        Args:
            data_df: DataFrame with multi-index columns (entity_class, parameter_definition, alternative, entity)
            db_map: DatabaseMapping instance
            mapping_name: Name of the mapping being processed
        """
        # Make the output look nice
        if data_df.empty:
            print("*", end='')  # Mark mappings with no data
            return
        else:
            print(' ', end='')

        errors = []
        
        # Order the dataframe column levels to be reliable
        data_df.columns = data_df.columns.reorder_levels(['EntityClass', 'ParameterDefinition', 'Alternative', 'Entity_byname'])
        data_df.columns.names = ['entity_class_name', 'parameter_definition_name', 'alternative_name', 'entity_byname']

        if isinstance(data_df.index, pd.MultiIndex):
            data_df.index = data_df.index.get_level_values(0)


        # Add parameter values based on value_type by iterating through columns
        if value_type == 'constant':
            # Scalar
            for col_name, col_data in data_df.items():
                value, type_ = to_database(col_data.iloc[0])
        
                # Add parameter value
                try:
                    db_map.add_parameter_value(
                        entity_class_name=col_name[0],
                        parameter_definition_name=col_name[1],
                        alternative_name=col_name[2],
                        entity_byname=col_name[3],
                        value=value,
                        type=type_
                    )
                except Exception as e:
                    errors.append(f"Failed to add a constant for (class, parameter, alternative, entity): {col_name[0]}, {col_name[1]}, {col_name[2]}, {col_name[3]} based on table.mapping {table_name}.{mapping_name}.\n    {e}")

        elif value_type == 'array':
            # Array
            for col_name, col_data in data_df.items():
                array_obj = Array(values=col_data.dropna().to_list(), index_name=col_data.index.name)
                value, type_ = to_database(array_obj)

                # Add parameter value
                try:
                    db_map.add_parameter_value(
                        entity_class_name=col_name[0],
                        parameter_definition_name=col_name[1],
                        alternative_name=col_name[2],
                        entity_byname=col_name[3],
                        value=value,
                        type=type_
                    )
                except Exception as e:
                    errors.append(f"Failed to add an array for (class, parameter, alternative, entity): {col_name[0]}, {col_name[1]}, {col_name[2]}, {col_name[3]} based on table.mapping {table_name}.{mapping_name}.\n    {e}")

        elif value_type == 'map':
            # Map
            for col_name, col_data in data_df.items():
                df_col = data_df[col_name].dropna()
                map_obj = Map(indexes=df_col.index.to_list(), values=df_col.values.tolist(), index_name=df_col.index.name)
                value, type_ = to_database(map_obj)

                # Add parameter value
                try:
                    db_map.add_parameter_value(
                        entity_class_name=col_name[0],
                        parameter_definition_name=col_name[1],
                        alternative_name=col_name[2],
                        entity_byname=col_name[3],
                        value=value,
                        type=type_
                    )
                except Exception as e:
                    errors.append(f"Failed to add a map for (class, parameter, alternative, entity): {col_name[0]}, {col_name[1]}, {col_name[2]}, {col_name[3]} based on table.mapping {table_name}.{mapping_name}.\n    {e}")

        # Raise errors if any occurred
        if errors:
            error_msg = f"\nErrors occurred while processing mapping '{mapping_name}':\n"
            error_msg += "\n".join(f"  - {err}" for err in errors)
            raise SpineDBAPIError(error_msg)


    # ==================== Reading Methods ====================

    def read_excel_sheet(self, excel_file_path: str, sheet_name: str) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Read a single Excel/ODS sheet and return raw DataFrame with metadata.

        Consolidates the functionality of read_excel_with_spec and _read_excel_with_options
        into a single method for simplicity.

        Args:
            excel_file_path: Path to Excel (.xlsx) or ODS (.ods) file
            sheet_name: Name of sheet to read

        Returns:
            Tuple of (raw_df, column_types)
            - raw_df: Raw pandas DataFrame with no header processing
            - column_types: Dictionary mapping column indices to types

        Raises:
            ValueError: If sheet not in specification
        """
        file_path = Path(excel_file_path)

        # Check if table is selected for processing
        if not self.is_table_selected(sheet_name):
            self.logger.info(f"Skipping {sheet_name} - not in selected tables")
            return pd.DataFrame(), {}, {}

        # Get column type specifications
        table_column_types = self.get_table_column_types(sheet_name.lower())
        default_column_type = self.specifications['tables']['node_c']['default_column_type']

        try:
            raw_df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='openpyxl', header=None, na_filter=True)
        except Exception as e1:
            try:
                print(f"The fast Calamine engine could not read sheet {sheet_name}, trying OpenPyxl.")
                raw_df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine='calamine', header=None, na_filter=True)
            except Exception as e2:
                raise ValueError(f"Both calamine and openpyxl engines failed for sheet {sheet_name}") from e2

        # Note: Column type conversions are applied after dataframe processing
        # in the extraction methods, not here on the raw dataframe

        if len(table_column_types) < len(raw_df.columns):
            for i in range(len(table_column_types),len(raw_df.columns),1):
                table_column_types[i] = default_column_type

        return raw_df, table_column_types

    def read_csv_file(self, csv_file_path: str) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Read a single CSV file and return raw DataFrame with metadata.

        Consolidates the functionality of read_csv_with_spec and _read_csv_with_options
        into a single method for simplicity.

        Args:
            csv_file_path: Path to CSV file

        Returns:
            Tuple of (raw_df, column_types)
            - raw_df: Raw pandas DataFrame with no header processing
            - column_types: Dictionary mapping column indices to types

        Raises:
            ValueError: If file not in specification
        """
        file_path = Path(csv_file_path)
        filename = file_path.name

        # Check if table is selected for processing
        if not self.is_table_selected(filename):
            self.logger.info(f"Skipping {filename} - not in selected tables")
            return pd.DataFrame(), {}, {}

        # Get column type specifications
        table_column_types = self.get_table_column_types(filename.lower())

        # Always read without headers - the mapping will extract headers from specific positions
        # This ensures all position references in the mapping are correct
        raw_df = pd.read_csv(file_path, header=None)

        # Note: Column type conversions are applied after dataframe processing
        # in the extraction methods, not here on the raw dataframe

        return raw_df, table_column_types

    # ==================== Spine DB Processing Methods ====================

    def _parse_mapping(self, mapping: List[Dict]) -> Dict[str, Any]:
        """Parse mapping structure to extract map_types and their configurations.

        Returns a dictionary with map_type as key and configuration as value.
        For map_types that appear multiple times (like Element, Dimension), stores them as lists.
        """
        parsed = {}
        parsed['skip_columns'] = []
        parsed['read_start_row'] = 0
        parsed['rules'] = {}

        if 'skip_columns' in mapping:
            parsed['skip_columns'] = mapping['skip_columns']
        if 'read_start_row' in mapping:
            parsed['read_start_row'] = mapping['read_start_row']

        for item in mapping['rules']:
            map_type = item.get('map_type')
            config = {
                'position': item.get('position'),
                'value': item.get('value'),
                'filter_re': item.get('filter_re'),
                'import_entities': item.get('import_entities')
            }
            # Handle multiple instances of same map_type (Element, Dimension, etc.)
            if map_type in ['Element', 'Dimension', 'IndexName', 'ParameterValueIndex']:
                if map_type not in parsed['rules']:
                    parsed['rules'][map_type] = []
                parsed['rules'][map_type].append(config)
            else:
                parsed['rules'][map_type] = config
            
        return parsed

    def _extract_data(self, df: pd.DataFrame, mapping_info: Dict, table_options: Dict,
                        table_column_types: Dict = None) -> (pd.DataFrame, list, zip):
        """Extract data from pivoted (tabular) format.

        In pivoted data:
        - Negative positions (-1, -2, ...) indicate HEADER ROWS at the top containing column labels
        - Positive positions (0, 1, 2, ...) indicate LABEL COLUMNS containing row identifiers
        - Data is in the rows AFTER header rows and columns NOT in label columns or skip_columns

        Args:
            df: Raw dataframe
            mapping_info: Parsed mapping configuration
            table_column_types: Dictionary mapping column indices (from raw df) to types

        Returns a DataFrame with multi-index columns: (entity_class, parameter_definition, alternative, entity)
        Rows contain parameter values, with multi-index for map/array parameters.
        """
        # Convenience
        rulez = mapping_info['rules']

        # optional return sets (entities and entity_alternatives to be added)
        ent_set_list = []
        ent_alt_zip = None

        # Separate header row and label column mappings
        col_level_positions = {}  # map_type -> row_index (negative positions)
        row_level_positions = {}  # map_type -> col_index (positive positions)
        hidden_mappings = {}  # map_type -> value


        # Find the maximum header row index to determine where data starts
        max_header_row = -1

        filters = {}
        for map_type, config in rulez.items():
            # Handle list of configs (Element, Dimension)
            if isinstance(config, list):
                positions = []
                counter = 0
                for cfg in config:
                    pos = cfg['position']
                    if isinstance(pos, int):
                        positions.append(pos)
                    if cfg['filter_re']:
                        filters[map_type + '_' + counter] = [cfg['filter_re'], pos]
                        counter =+ counter
                # Store as list
                if positions:
                    row_level_flag = False
                    col_level_flag = False
                    row_level_positions[map_type] = []
                    for i, pos in enumerate(positions):
                        # Check if negative (header rows) or positive (label columns)
                        if pos < 0:
                            # Header rows
                            col_pos = abs(pos) - 1
                            if col_level_flag is False:
                                col_level_positions[map_type] = [col_pos]
                                col_level_flag = True
                            else:
                                col_level_positions[map_type].append(col_pos)
                            max_header_row = max(max_header_row, col_pos)
                        else:
                            # Label columns
                            if row_level_flag is False:
                                row_level_positions[map_type] = [pos]    
                                row_level_flag = True
                            else:
                                row_level_positions[map_type].append(pos)
            else:
                pos = config['position']
                if pos == 'hidden':
                    if config.get('value') is not None:
                        hidden_mappings[map_type] = config['value']
                elif isinstance(pos, int):
                    if pos < 0:
                        # Header row - convert to 0-based row index
                        col_positions = abs(pos) - 1
                        col_level_positions[map_type] = col_positions
                        max_header_row = max(max_header_row, col_positions)
                    else:
                        # Label column
                        row_level_positions[map_type] = pos
                if config['filter_re']:
                    filters[map_type] = [config['filter_re'], pos]

        # Data rows start AFTER all header rows
        data_start_row = max(max_header_row + 1, mapping_info['read_start_row'])

        ent_alt_col = set()
        # Treat EntityAlternativeActivity separately
        if 'EntityAlternativeActivity' in row_level_positions:
            ent_alt_col = set([row_level_positions.pop('EntityAlternativeActivity')])

        # Identify which columns are label columns (to skip when extracting data)
        label_column_indices = set()
        for pos in row_level_positions.values():
            if isinstance(pos, list):
                label_column_indices.update(pos)
            else:
                label_column_indices.add(pos)

        # Step 0: Copy the dataframe
        df_work = df.copy()
        if table_options.get('header'):
            df_work = df_work.iloc[1:,:]
        skip_cols = set(mapping_info['skip_columns'])

        # Step 1a: Find columns based on regex filter that will be later removed (but not yet, to keep the column order)
        for map_type, rules in filters.items():
            position = rules[1]
            if not isinstance(position, int):
                raise TypeError(f"Regex filter for a mapping that does not have integer position: {mapping_info}")
            if position >= 0:  # Positive, drop rows
                df_work = df_work.drop(df_work.index[data_start_row:][~df_work.iloc[data_start_row:][position].str.contains(rules[0], regex=True, na=False)])
            else:              # Positive, drop columns
                avoid_cols = set(skip_cols.union(label_column_indices))
                examine_cols = list(set(range(len(df_work.columns))) - avoid_cols)
                filter_cols = (~df_work.iloc[-position-1,examine_cols].str.contains(rules[0], regex=True, na=False)).index.tolist()
                if filter_cols:
                    skip_cols = skip_cols.union(filter_cols)

        # Sidetrack 1b: Get scenarios and scenario_alternatives (if defined in the mapping)
        if 'Scenario' in rulez:
            scen_rule = rulez['Scenario']

            if scen_rule['position'] >= 0:
                scen_array = df_work.iloc[:,scen_rule['position']]
                if 'ScenarioAlternative' in rulez:
                    df_work.index = df_work.iloc[:,scen_rule['position']]
                    df_work = df_work.iloc[:,scen_rule['position'] + 1:]
                    return(None, None, None, None, df_work)
                else:
                    return(None, None, None, scen_array, None)
                    
            if scen_rule['position'] < 0:
                examine_cols = list(set(range(len(df_work.columns))) - skip_cols)
                df_work = df_work.iloc[:, examine_cols]
                scen_array = df_work.iloc[-scen_rule['position']-1]
                if 'ScenarioAlternative' in rulez:
                    df_work.columns = df_work.iloc[-scen_rule['position']-1]
                    df_work = df_work.iloc[-scen_rule['position']:]
                    return(None, None, None, None, df_work)
                else:
                    return(None, None, None, scen_array, None)

        # Step 2: Build column multi-index from 'header' rows
        skip_cols = label_column_indices.union(skip_cols).union(ent_alt_col)
        examine_cols = list(set(range(len(df_work.columns))) - skip_cols)

        col_index_names = []
        if col_level_positions:
            col_index_arrays = []
            nan_positions = set()

            for map_type, pos in col_level_positions.items():
                if isinstance(pos, list):
                    # Multiple column levels in this map_type
                    for i, row_idx in enumerate(pos):
                        col_index_arrays.append(df_work.iloc[row_idx, :].values)
                        col_index_names.append(f"{map_type}_{i}")
                else:
                    col_index_arrays.append(df_work.iloc[pos, :].values)
                    col_index_names.append(map_type)
                    nan_positions.update(df_work.iloc[pos, examine_cols][df_work.iloc[pos, examine_cols].isna()].index.tolist())

            df_work.columns = pd.MultiIndex.from_arrays(col_index_arrays, names=col_index_names)
            skip_cols = skip_cols.union(nan_positions)

        # Step 3a: Drop header rows (data starts at data_start_row)
        df_work = df_work.iloc[data_start_row:, :].reset_index(drop=True)

        # Sidetrack 3b: Add entities from elements if required
        if rulez.get('Element') is not None:
            for i, element_map in enumerate(rulez.get('Element')):
                if element_map['import_entities']:
                    if element_map['position'] >= 0:
                        ent_array = df_work.iloc[:,element_map['position']].to_list()
                    elif element_map['position'] < 0:
                        ent_array = df_work.iloc[element_map['position'] + 1,:].to_list()
                    # Add entity_class name for the dimension
                    ent_array = [(x,) for x in ent_array]
                    ent_set_list.append(zip([rulez['Dimension'][i]['value']]*len(ent_array), ent_array))

        # Step 4: Build row multi-index from 'header' columns
        row_index_names = []
        if row_level_positions:
            row_index_arrays = []

            for map_type, pos in row_level_positions.items():
                if isinstance(pos, list):
                    for i, col_idx in enumerate(pos):
                        row_index_arrays.append(df_work.iloc[:, col_idx].values)
                        row_index_names.append(f"{map_type}_{i}")
                else:
                    row_index_arrays.append(df_work.iloc[:, pos].values)
                    row_index_names.append(map_type)

            df_work.index = pd.MultiIndex.from_arrays(row_index_arrays, names=row_index_names)

        # Step 5: Apply column type conversions to data columns
        if table_column_types:
            for orig_col_idx, col_type in table_column_types.items():
                orig_col_idx = int(orig_col_idx)
                if orig_col_idx < len(df_work.columns):
                    # Check if this original column is still in the dataframe
                    if col_type == "float":
                        try:
                            df_work.iloc[:, orig_col_idx] = pd.to_numeric(
                                df_work.iloc[:, orig_col_idx], errors='raise')
                        except (ValueError, TypeError) as e:
                            self.logger.warning(
                                f" Column {orig_col_idx} contains non-numeric values: {e}. "
                                f"Converting invalid values to NaN and continuing.\n{' ':<30}"
                            )
                            df_work.iloc[:, orig_col_idx] = pd.to_numeric(
                                df_work.iloc[:, orig_col_idx], errors='coerce')
                    elif col_type == "string":
                        df_work.iloc[:, orig_col_idx] = df_work.iloc[:, orig_col_idx].astype('string')
                else:
                    continue

        # Step 6a: combine element names to entity_byname
        row_entity_levels = []
        col_entity_levels = []
        for row_index_name in row_index_names:
            if row_index_name.startswith('Element_') or row_index_name == 'Entity':
                row_entity_levels.append(row_index_name)
        for col_index_name in col_index_names:
            if col_index_name.startswith('Element_') or col_index_name == 'Entity':
                col_entity_levels.append(col_index_name)

        if row_entity_levels:
            joined = df_work.index.to_frame()[row_entity_levels].apply(tuple, axis=1)
            joined.name = 'Entity_byname'
            df_work = df_work.set_index(joined, append=True)
            df_work.index = df_work.index.droplevel(row_entity_levels)
            if not isinstance(df_work.index, pd.MultiIndex):
                df_work.index = pd.MultiIndex.from_arrays([df_work.index], names=[df_work.index.name])
            # entity_array = df_work.index.get_level_values('Entity_byname').unique().tolist()
        elif col_entity_levels:
            joined = df_work.columns.to_frame()[col_entity_levels].apply(tuple, axis=1).to_frame()
            joined.columns = ['Entity_byname']
            df_work.columns = joined.set_index('Entity_byname', append=True).index
            df_work.columns = df_work.columns.droplevel(col_entity_levels)
            # entity_array = df_work.columns.get_level_values('Entity_byname').unique().tolist()
        elif rulez.get('Entity') and rulez['Entity']['position'] == 'hidden' and rulez['Entity']['value']:
            df_work = df_work.drop(columns=df_work.columns)
            if rulez.get('ExpandedValue'):
                df_work = df_work.reset_index(level='ExpandedValue')
                df_work = df_work.set_index(pd.Index(data=range(len(df_work))), append=True)
            elif rulez.get('ParameterValue'):
                df_work = df_work.reset_index(level='ParameterValue')
            else:
                raise(f'No Entity nor Elements and also no ExpandedValue or Parametervalue for {mapping_info["rules"]}')
            columns_array = [[(rulez['Entity']['value'],)]]
            df_work.columns = pd.MultiIndex.from_arrays(columns_array, names=['Entity_byname'])
            skip_cols = set()
        else:
            raise RuntimeError(f'The import definition for {mapping_info} does not include Entity nor Dimension')

        # Sidetrack 6b: Add entities to the list
        # if rulez['EntityClass']['position'] == 'hidden':
        #     entity_class_array = [hidden_mappings['EntityClass']]*len(entity_array) 
        # elif rulez['EntityClass']['position'] >= 0:
        #     entity_class_array = df_work.index.get_level_values('EntityClass').tolist()
        # elif rulez['EntityClass']['position'] < 0:
        #     entity_class_array = df_work.columns.get_level_values('EntityClass').tolist()
        # ent_zip_list.append(zip(entity_class_array, entity_array))

        # Sidetrack 6c: EntityAlternative or Entity definition (now that df_work has been sufficiently cleaned up)
        if 'EntityAlternativeActivity' in rulez:
            df_ent_alt = df_work.iloc[:,rulez['EntityAlternativeActivity']['position']].dropna()
            if isinstance(df_ent_alt, pd.Series):
                df_ent_alt = df_ent_alt.to_frame()
            if 'Alternative' in df_ent_alt.columns:
                df_ent_alt = df_ent_alt.stack('Alternative')
            if 'Entity_byname' in df_ent_alt.columns:
                df_ent_alt = df_ent_alt.stack('Entity_byname')
            entity_class_array = [hidden_mappings['EntityClass']]*len(df_ent_alt) 
            ent_alt_zip = zip(entity_class_array, 
                df_ent_alt.index.get_level_values('Entity_byname').tolist(), 
                df_ent_alt.index.get_level_values('Alternative').tolist(), 
                df_ent_alt.squeeze(axis=1).values)

        # Step 6e: Drop label columns and skip_columns (in reverse order to preserve indices)
        #          Also drop metadata levels (both from columns and rows)
        # Combine label columns and skip columns and possible entityAlternative columns
        for map_type in ['EntityMetadata', 'ParameterValueMetadata']:
            if map_type in rulez:
                pos = rulez[map_type]['position']
                if isinstance(pos, int) and pos >= 0:
                    skip_cols.add(pos)
                    df_work.index = df_work.index.droplevel(map_type)
                elif isinstance(pos, int) and pos < 0:
                    df_work.columns = df_work.columns.droplevel(map_type)
        cols_to_keep = list(set(range(len(df_work.columns))) - skip_cols)
        df_work = df_work.iloc[:, cols_to_keep]

        # Special case 6f: Data in ExpandedValue column:
        exp_value_rules = rulez.get('ExpandedValue')
        if exp_value_rules:
            exp_value_position = exp_value_rules.get('position')
        if row_entity_levels and not col_entity_levels and exp_value_rules and exp_value_position and isinstance(exp_value_position, int):
            df_work = df_work.drop(columns=df_work.columns)
            df_work = df_work.reset_index(level='ExpandedValue')

        if 'Entity_byname' in df_work.index.names \
            and not any(x.startswith('ParameterValueIndex') for x in df_work.index.names) \
            and 'ParameterDefinition' not in df_work.columns.names \
            and 'ExpandedValue' not in df_work.columns.to_list():
            df_work = df_work.T

        # Step 7: Add hidden values as constant levels
        column_relevant_types = ['EntityClass', 'ParameterDefinition', 'Alternative', 'Entity_byname'] # , 'Element', 'Dimension'
        row_relevant_types = ['ParameterValueIndex']
        preferred_types = {'EntityClass': 'Entity_byname', 'Entity_byname': 'EntityClass', 'ParameterDefinition': 'Alternative', 'Alternative': 'ParameterDefinition'}

        # Step 7a: Add hidden column levels
        for map_type in column_relevant_types:
            if map_type in hidden_mappings:
                if preferred_types[map_type] in df_work.index.names:
                    df_work.index = self._add_hidden_map_type_to_index(map_type, hidden_mappings[map_type], df_work.index) 
                else:
                    df_work.columns = self._add_hidden_map_type_to_index(map_type, hidden_mappings[map_type], df_work.columns) 

        # Step 7b: Add hidden row levels
        for map_type in row_relevant_types:
            if map_type in hidden_mappings:
                if preferred_types[map_type] in df_work.columns.names:
                    df_work.columns = self._add_hidden_map_type_to_index(map_type, hidden_mappings[map_type], df_work.columns) 
                else:
                    df_work.index = self._add_hidden_map_type_to_index(map_type, hidden_mappings[map_type], df_work.index) 

        # Step 7c: Make zipped arrays to add entities (before any of them are dropped by stack/unstack operations when no data)
        if 'Entity_byname' in df_work.columns.names:
            ent_class = df_work.columns.get_level_values('EntityClass')
            ent = df_work.columns.get_level_values('Entity_byname')
        elif 'Entity_byname' in df_work.index.names:
            ent_class = df_work.index.get_level_values('EntityClass')
            ent = df_work.index.get_level_values('Entity_byname')
        ent_set_list.append(set(zip(ent_class, ent)))

        # Shortcut 7d:
        # If there are no parameter values to be imported, cut short:
        if 'ParameterDefinition' not in rulez:
            return None, ent_set_list, ent_alt_zip, None, None

        # Step 8: Stack column levels that belong in rows (IndexName, ParameterValueIndex)
        if isinstance(df_work.columns, pd.MultiIndex):
            levels_to_stack = [name for name in list(df_work.columns.names)
                              if name and name.startswith(('IndexName', 'ParameterValueIndex'))]
            if levels_to_stack:
                df_work = df_work.stack(levels_to_stack, future_stack=True)
                # Ensure result is a DataFrame (stack can return Series with single column)
                if isinstance(df_work, pd.Series):
                    df_work = df_work.to_frame().replace('', np.nan).dropna()

        # Step 9: Unstack row levels that belong in columns (EntityClass, ParameterDefinition, Alternative, Entity_byname)
        if isinstance(df_work.index, pd.MultiIndex):
            levels_to_unstack = [name for name in list(df_work.index.names)
                                if name and name.startswith(('EntityClass', 'ParameterDefinition', 'Alternative', 'Entity_byname'))]
            if levels_to_unstack:
                if df_work.index.duplicated().any():
                    df_work = df_work.set_index(pd.Index(data=range(len(df_work))), append=True)
                df_work = df_work.unstack(levels_to_unstack)
                # Ensure result is a DataFrame (unstack can return Series with single row)
                if isinstance(df_work, pd.Series):
                    df_work = df_work.to_frame().replace('', np.nan).dropna().T

        # Step 10: Add index names
        if rulez.get('IndexName') and not isinstance(rulez['IndexName'][0]['position'], int):
            if isinstance(df_work.index, pd.MultiIndex):
                for i, index_name in enumerate(rulez['IndexName']):
                    df_work.index = df_work.index.rename(index_name['value'], level=i)
            else:
                df_work.index.name = rulez['IndexName'][0]['value']

        # Might need to convert to 0 for scalars
        # Check if all values are empty/NaN
        # if df_work.index.isna().all() or (df_work.index == '').all():
        #     df_work.index = pd.Index([0] * len(df_work))

        # Step 11: Drop all empty or NaN columns (now, after data has been organised into Spine db format):
        df_work = df_work.dropna(axis=1, how='all')

        # Step 12: Clean possible None level from the column level multi-index
        if None in df_work.columns.names:
            none_level_pos = df_work.columns.names.index(None)
            df_work.columns = df_work.columns.droplevel(none_level_pos)

        return df_work, ent_set_list, ent_alt_zip, None, None

    def _build_column_key(self, item: Dict) -> Tuple:
        """Build column key tuple: (entity_class, parameter_definition, alternative, entity).

        entity_class and entity can be tuples for multi-dimensional entities.
        """
        entity_class = item.get('EntityClass')

        # Build entity_class tuple if multi-dimensional
        if 'Dimension' in item:
            # Get dimensions if they exist
            dimensions = []
            i = 0
            while f'Dimension_{i}' in item:
                dimensions.append(item[f'Dimension_{i}'])
                i += 1
            if dimensions:
                entity_class = tuple(dimensions)

        parameter_definition = item.get('ParameterDefinition')
        alternative = item.get('Alternative', 'Base')

        # Build entity tuple if multi-dimensional
        entity_elements = []
        i = 0
        while f'Element_{i}' in item:
            entity_elements.append(str(item[f'Element_{i}']))
            i += 1

        if entity_elements:
            entity = tuple(entity_elements)
        elif 'Entity' in item:
            entity = str(item['Entity'])
        else:
            entity = None

        return (entity_class, parameter_definition, alternative, entity)

    def _build_row_index(self, item: Dict, pv_type: str) -> Any:
        """Build row index for map/array parameters.

        For scalar values: returns 0 (default index)
        For maps: returns the index value from ParameterValueIndex
        For arrays: returns the array index
        """
        if pv_type in ['map', 'array']:
            # Get index from ParameterValueIndex
            if 'ParameterValueIndex' in item:
                return item['ParameterValueIndex']

        # Default: scalar value, use index 0
        return 0

    def _build_dataframe_from_column_data(self, column_data: Dict[Tuple, Dict]) -> pd.DataFrame:
        """Build DataFrame from column_data dictionary.

        Args:
            column_data: Dict mapping column_key -> {row_index -> value}

        Returns:
            DataFrame with multi-index columns
        """
        if not column_data:
            # Return empty DataFrame with appropriate structure
            return pd.DataFrame()

        # Create DataFrame from nested dict
        df = pd.DataFrame(column_data)

        # Set multi-index column names
        if len(df.columns) > 0:
            df.columns = pd.MultiIndex.from_tuples(
                df.columns,
                names=['entity_class', 'parameter_definition', 'alternative', 'entity']
            )

        return df


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Read data from CSV, Excel (.xlsx), or ODS (.ods) files using specification mappings'
    )
    parser.add_argument(
        '--mode',
        choices=['csv', 'excel'],
        default='csv',
        help='Source mode: csv for CSV files, excel for Excel/ODS files (supports .xlsx and .ods) (default: csv)'
    )
    parser.add_argument(
        '--spec',
        type=str,
        default='import_flex_mod.json',
        help='Path to the specification JSON file (default: import_flex_mod.json)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='output',
        help='Input path: directory for CSV mode, file path (.xlsx or .ods) for Excel mode (default: output)'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Initialize the TabularReader with the specification file
    reader = TabularReader(args.spec)

    # Get the source type from specification
    source_type = reader.get_source_type()
    logger.info(f"Specification source type: {source_type}")
    logger.info(f"Running in {args.mode} mode")

    # Process tables/sheets one at a time using streaming approach
    selected_tables = reader.get_selected_table_names()
    logger.info(f"Processing {len(selected_tables)} selected tables")

    tables_read = 0
    for table_name in selected_tables:
        try:
            if args.mode == 'csv':
                csv_file = Path(args.input) / f"{table_name}.csv"
                df, mappings, column_types = reader.read_csv_file(str(csv_file))
            elif args.mode == 'excel':
                df, mappings, column_types = reader.read_excel_sheet(args.input, table_name)
            else:
                continue

            if not df.empty:
                logger.info(f"Successfully read {table_name}: {df.shape}")
                tables_read += 1
                # Here you would process the data further if needed
            else:
                logger.info(f"Skipped {table_name} - no data")

        except Exception as e:
            logger.error(f"Error reading {table_name}: {e}")
            continue

    logger.info(f"Finished processing. Total tables/sheets read: {tables_read}")

