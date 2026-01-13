import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
import argparse
import re
from spinedb_api import to_database, Map, Array, SpineDBAPIError
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


    def _add_entities(self, ent_zip_list, db_map, mapping_name):
        for ent_zip in ent_zip_list:
            for ent_class, ent in ent_zip:
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
                        raise SpineDBAPIError(f'Could not add entity {ent} from class {ent_class} to the database: {e}')

        # Commit the changes
        try:
            db_map.commit_session(f"Imported entities from {mapping_name}")
            self.logger.info(f"Successfully committed entities '{mapping_name}'")
        except NothingToCommit:
            pass
        except Exception as e:
            raise SpineDBAPIError(f"Failed to commit session: {e}")

    def _add_alternatives(self, ent_zip_list, db_map, mapping_name):
        for ent_zip in ent_zip_list:
            for ent_class, ent in ent_zip:
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
                        raise SpineDBAPIError(f'Could not add entity {ent} from class {ent_class} to the database: {e}')

        # Commit the changes
        try:
            db_map.commit_session(f"Imported entities from {mapping_name}")
            self.logger.info(f"Successfully committed entities '{mapping_name}'")
        except NothingToCommit:
            pass
        except Exception as e:
            raise SpineDBAPIError(f"Failed to commit session: {e}")

    def _add_entity_alternatives(self, ent_act_zip, db_map, mapping_name):
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
                    raise SpineDBAPIError(f'Could not add entity_alternative {ent}, {alt} from class {ent_class} to the database: {e}')

        # Commit the changes
        try:
            db_map.commit_session(f"Imported entity_alternatives from {mapping_name}")
            self.logger.info(f"Successfully committed entity_alternatives '{mapping_name}'")
        except NothingToCommit:
            pass
        except Exception as e:
            raise SpineDBAPIError(f"Failed to commit session: {e}")


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

        # Auto-detect engine based on file extension
        file_extension = file_path.suffix.lower()
        if file_extension == '.ods':
            engine = 'odf'
        elif file_extension in ['.xlsx', '.xlsm', '.xltx', '.xltm']:
            engine = 'openpyxl'
        else:
            # Default to openpyxl for backward compatibility
            engine = 'openpyxl'
            self.logger.warning(f"Unknown file extension '{file_extension}', defaulting to openpyxl engine")

        # Always read without headers - the mapping will extract headers from specific positions
        # This ensures all position references in the mapping are correct
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name,
                              header=None, engine=engine, na_filter=True)

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
            if map_type in ['Element', 'Dimension']:
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
        ent_zip_list = []
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
                    # Check if negative (header rows) or positive (label columns)
                    if positions[0] < 0:
                        # Header rows
                        header_rows = [abs(p) - 1 for p in positions]
                        col_level_positions[map_type] = header_rows
                        max_header_row = max(max_header_row, max(header_rows))
                    else:
                        # Label columns
                        row_level_positions[map_type] = positions
            else:
                pos = config['position']
                if pos == 'hidden':
                    if config.get('value') is not None:
                        hidden_mappings[map_type] = config['value']
                elif isinstance(pos, int):
                    if pos < 0:
                        # Header row - convert to 0-based row index
                        header_row_idx = abs(pos) - 1
                        col_level_positions[map_type] = header_row_idx
                        max_header_row = max(max_header_row, header_row_idx)
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
                    df_work.index = df_work.iloc[:,0]
                    df_work = df_work.iloc[:,1:]
                    return(None, None, None, None, df_work)
                else:
                    return(None, None, None, scen_array, None)
                    
            if scen_rule['position'] < 0:
                scen_array = df_work.iloc[-scen_rule['position']-1]
                if 'ScenarioAlternative' in rulez:
                    df_work.columns = df_work.iloc[0]
                    df_work = df_work.iloc[1:]
                    return(None, None, None, None, df_work)
                else:
                    return(None, None, None, scen_array, None)

        # Step 2: Build column multi-index from 'header' rows
        col_index_names = []
        if col_level_positions:
            col_index_arrays = []

            for map_type, pos in col_level_positions.items():
                if isinstance(pos, list):
                    # Multiple column levels in this map_type
                    for i, row_idx in enumerate(pos):
                        col_index_arrays.append(df_work.iloc[row_idx, :].values)
                        col_index_names.append(f"{map_type}_{i}")
                else:
                    col_index_arrays.append(df_work.iloc[pos, :].values)
                    col_index_names.append(map_type)

            df_work.columns = pd.MultiIndex.from_arrays(col_index_arrays, names=col_index_names)

        # Step 3: Drop header rows (data starts at data_start_row)
        df_work = df_work.iloc[data_start_row:, :].reset_index(drop=True)

        # Sidetrack 3a: Make EntityAlternatives

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
                    ent_zip_list.append(zip([rulez['Dimension'][i]['value']]*len(ent_array), ent_array))

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

        # Step 6b: combine element names to entity_byname
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
        elif col_entity_levels:
            joined = df_work.columns.to_frame()[col_entity_levels].apply(tuple, axis=1).to_frame()
            joined.columns = ['Entity_byname']
            df_work.columns = joined.set_index('Entity_byname', append=True).index
            df_work.columns = df_work.columns.droplevel(col_entity_levels)
        elif rulez.get('Entity') and rulez['Entity']['position'] == 'hidden' and rulez['Entity']['value']:
            df_work = df_work.drop(columns=df_work.columns)
            if rulez.get('ExpandedValue'):
                df_work = df_work.reset_index(level='ExpandedValue')
                df_work = df_work.set_index(pd.Index(data=range(len(df_work)), name=rulez['IndexName']['value']), append=True)
            elif rulez.get('ParameterValue'):
                df_work = df_work.reset_index(level='ParameterValue')
            else:
                raise(f'No Entity nor Elements and also no ExpandedValue or Parametervalue for {mapping_info["rules"]}')
            columns_array = [[(rulez['Entity']['value'],)]]
            df_work.columns = pd.MultiIndex.from_arrays(columns_array, names=['Entity_byname'])
        else:
            raise RuntimeError(f'The import definition for {mapping_info} does not include Entity nor Dimension')
            return (None, ent_zip_list, None, None, None)

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
        elif 'ParameterDefinition' not in rulez:
            entity_array = df_work.drop(df_work.columns, axis=1).index.unique().to_list()
            if 'EntityClass' in hidden_mappings:
                entity_class_array = [hidden_mappings['EntityClass']]*len(entity_array) 
            ent_zip_list.append(zip(entity_class_array, entity_array))
            return None, ent_zip_list, None, None, None

        # Step 6a: Drop label columns and skip_columns (in reverse order to preserve indices)
        #          Also drop metadata levels (both from columns and rows)
        # Combine label columns and skip columns and possible entityAlternative columns
        cols_to_drop = label_column_indices.union(skip_cols).union(ent_alt_col)
        for map_type in ['EntityMetadata', 'ParameterValueMetadata']:
            if map_type in rulez:
                pos = rulez[map_type]['position']
                if isinstance(pos, int) and pos >= 0:
                    cols_to_drop.add(pos)
                    df_work.index = df_work.index.droplevel(map_type)
                elif isinstance(pos, int) and pos < 0:
                    df_work.columns = df_work.columns.droplevel(map_type)
        cols_to_drop_sorted = sorted(cols_to_drop, reverse=True)
        for col_idx in cols_to_drop_sorted:
            if col_idx < len(df_work.columns):
                df_work = df_work.drop(df_work.columns[col_idx], axis=1)

        # Special case 6d: Data in ExpandedValue column:
        exp_value_rules = rulez.get('ExpandedValue')
        if exp_value_rules:
            exp_value_position = exp_value_rules.get('position')
        if row_entity_levels and not col_entity_levels and exp_value_rules and exp_value_position and isinstance(exp_value_position, int):
            df_work = df_work.drop(columns=df_work.columns)
            df_work = df_work.reset_index(level='ExpandedValue')

        # Step 7: Add hidden values as constant levels
        column_relevant_types = ['EntityClass', 'ParameterDefinition', 'Alternative', 'Entity_byname'] # , 'Element', 'Dimension'
        row_relevant_types = ['ParameterValueIndex']

        # Add hidden column levels
        for map_type in column_relevant_types:
            if map_type in hidden_mappings:
                value = hidden_mappings[map_type]
                if isinstance(df_work.columns, pd.MultiIndex):
                    # Get all arrays from existing multi-index
                    arrays = [df_work.columns.get_level_values(i) for i in range(df_work.columns.nlevels)]
                    arrays.append([value] * len(df_work.columns))
                    names = list(df_work.columns.names) + [map_type]
                    df_work.columns = pd.MultiIndex.from_arrays(arrays, names=names)
                else:
                    df_work.columns = pd.MultiIndex.from_arrays(
                        [df_work.columns.values, [value] * len(df_work.columns)],
                        names=[df_work.columns.name, map_type]
                    )

        # Add hidden row levels
        for map_type in row_relevant_types:
            if map_type in hidden_mappings:
                value = hidden_mappings[map_type]
                if isinstance(df_work.index, pd.MultiIndex):
                    arrays = [df_work.index.get_level_values(i) for i in range(df_work.index.nlevels)]
                    arrays.append([value] * len(df_work.index))
                    names = list(df_work.index.names) + [map_type]
                    df_work.index = pd.MultiIndex.from_arrays(arrays, names=names)
                else:
                    df_work.index = pd.MultiIndex.from_arrays(
                        [df_work.index.values, [value] * len(df_work.index)],
                        names=[df_work.index.name, map_type]
                    )

        # Step 8: Stack column levels that belong in rows (IndexName, ParameterValueIndex)
        if isinstance(df_work.columns, pd.MultiIndex):
            levels_to_stack = [name for name in df_work.columns.names
                              if name in ['IndexName', 'ParameterValueIndex']]
            if levels_to_stack:
                df_work = df_work.stack(levels_to_stack, future_stack=True)
                # Ensure result is a DataFrame (stack can return Series with single column)
                if isinstance(df_work, pd.Series):
                    df_work = df_work.to_frame().replace('', np.nan).dropna()

        # Step 9: Stack and unstack to reorganize indices
        # Unstack row levels that belong in columns (EntityClass, ParameterDefinition, Alternative, Entity_byname)
        if isinstance(df_work.index, pd.MultiIndex):
            levels_to_unstack = [name for name in df_work.index.names
                                if name in ['EntityClass', 'ParameterDefinition', 'Alternative', 'Entity_byname']]
            if levels_to_unstack:
                if df_work.index.duplicated().any():
                    df_work = df_work.set_index(pd.Index(data=range(len(df_work))), append=True)
                df_work = df_work.unstack(levels_to_unstack)
                # Ensure result is a DataFrame (unstack can return Series with single row)
                if isinstance(df_work, pd.Series):
                    df_work = df_work.to_frame().replace('', np.nan).dropna().T

        # Step 10: Build final row index from ParameterValueIndex or IndexName
        if isinstance(df_work.index, pd.MultiIndex):
            # Combine Element dimensions into tuples for parameter value index
            df_work.index = df_work.index.map(lambda x: '-'.join(map(str, x)))
        else:
            # Single index - might need to convert to 0 for scalars
            # Check if all values are empty/NaN
            if df_work.index.isna().all() or (df_work.index == '').all():
                df_work.index = pd.Index([0] * len(df_work))

        # Step 11: Drop all empty or NaN columns (now, after data has been organised into Spine db format):
        df_work = df_work.dropna(axis=1, how='all')

        # Step 12: Clean possible None level from the column level multi-index
        if None in df_work.columns.names:
            none_level_pos = df_work.columns.names.index(None)
            df_work.columns = df_work.columns.droplevel(none_level_pos)

        return df_work, ent_zip_list, ent_alt_zip, None, None

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

    def _extract_linear_data(self, df: pd.DataFrame, mapping_info: Dict, table_options: Dict) -> pd.DataFrame:
        """Extract data from linear (column or row based) format.

        In linear data, ParameterValue/ExpandedValue has an integer position indicating
        where the actual data is located (either a column or row).

        Returns a DataFrame with multi-index columns: (entity_class, parameter_definition, alternative, entity)
        Rows contain parameter values, with multi-index for map/array parameters.
        """
        # Determine if data is in columns or rows
        pv_pos = mapping_info.get('ParameterValue', {}).get('position')
        if not pv_pos or pv_pos == 'hidden':
            pv_pos = mapping_info.get('ExpandedValue', {}).get('position')

        is_column_data = isinstance(pv_pos, int) and pv_pos >= 0

        # Collect hidden values
        hidden_mappings = {}
        for map_type, config in mapping_info.items():
            if not isinstance(config, list) and config['position'] == 'hidden' and config.get('value') is not None:
                hidden_mappings[map_type] = config['value']

        pv_type = hidden_mappings.get('ParameterValueType', 'float')

        # Build data structure: dict of column_key -> {row_index -> value}
        column_data = {}

        # Iterate through rows (for column data) or columns (for row data)
        if is_column_data:
            # Data is organized in columns
            for row_idx in range(len(df)):
                item = {**hidden_mappings}
                skip_row = False

                # Extract values from each mapped position
                for map_type, config in mapping_info.items():
                    # Handle list of configs (Element, Dimension)
                    if isinstance(config, list):
                        # For Element, collect all values in order
                        if map_type == 'Element':
                            for i, cfg in enumerate(config):
                                pos = cfg['position']
                                if isinstance(pos, int) and pos >= 0 and pos < len(df.columns):
                                    value = df.iloc[row_idx, pos]
                                    item[f'Element_{i}'] = value
                        elif map_type == 'Dimension':
                            for i, cfg in enumerate(config):
                                if cfg.get('value') is not None:
                                    item[f'Dimension_{i}'] = cfg['value']
                    else:
                        pos = config['position']
                        if isinstance(pos, int):
                            if pos >= 0 and pos < len(df.columns):
                                value = df.iloc[row_idx, pos]

                                # Apply filter if specified
                                filter_re = config.get('filter_re')
                                if filter_re:
                                    if pd.isna(value) or not re.match(filter_re, str(value)):
                                        skip_row = True
                                        break

                                item[map_type] = value
                            elif pos < 0:
                                # Negative position means row header
                                # Convert to 0-based column index
                                header_row = abs(pos) - 1
                                # This would need to be extracted from a header row
                                # For now, we'll use the column name
                                if header_row == 0 and hasattr(df.columns[0], '__iter__'):
                                    item[map_type] = df.columns[0][header_row]

                if skip_row:
                    continue

                # Get value
                if 'ParameterValue' in item and pd.isna(item.get('ParameterValue')):
                    continue
                if 'ExpandedValue' in item and pd.isna(item.get('ExpandedValue')):
                    continue

                # Build column key and row index
                col_key = self._build_column_key(item)
                row_key = self._build_row_index(item, pv_type)

                # Get the value
                if 'ExpandedValue' in item:
                    cell_value = item['ExpandedValue']
                elif 'ParameterValue' in item:
                    cell_value = item['ParameterValue']
                else:
                    continue

                # Store the value
                if col_key not in column_data:
                    column_data[col_key] = {}
                column_data[col_key][row_key] = cell_value

        else:
            # Data is organized in rows (less common)
            for col_idx in range(len(df.columns)):
                item = {**hidden_mappings}
                skip_col = False

                for map_type, config in mapping_info.items():
                    if not isinstance(config, list):
                        pos = config['position']
                        if isinstance(pos, int):
                            if pos < 0:
                                # Negative position is row index (0-based after conversion)
                                row_idx = abs(pos) - 1
                                if row_idx < len(df):
                                    value = df.iloc[row_idx, col_idx]

                                    # Apply filter if specified
                                    filter_re = config.get('filter_re')
                                    if filter_re:
                                        if pd.isna(value) or not re.match(filter_re, str(value)):
                                            skip_col = True
                                            break

                                    item[map_type] = value

                if skip_col:
                    continue

                # Get value
                if 'ParameterValue' in item and pd.isna(item.get('ParameterValue')):
                    continue
                if 'ExpandedValue' in item and pd.isna(item.get('ExpandedValue')):
                    continue

                # Build column key and row index
                col_key = self._build_column_key(item)
                row_key = self._build_row_index(item, pv_type)

                # Get the value
                if 'ExpandedValue' in item:
                    cell_value = item['ExpandedValue']
                elif 'ParameterValue' in item:
                    cell_value = item['ParameterValue']
                else:
                    continue

                # Store the value
                if col_key not in column_data:
                    column_data[col_key] = {}
                column_data[col_key][row_key] = cell_value

        # Convert to DataFrame with multi-index columns
        return self._build_dataframe_from_column_data(column_data)

    def _write_items_to_db(self, data_df: pd.DataFrame, db_map, mapping_name: str, value_type: str) -> None:
        """Write extracted data DataFrame to Spine database.

        This method:
        1. Collects unique alternatives and entities from DataFrame columns
        2. Adds alternatives and entities first
        3. Adds parameter values by iterating through DataFrame columns

        Args:
            data_df: DataFrame with multi-index columns (entity_class, parameter_definition, alternative, entity)
            db_map: DatabaseMapping instance
            mapping_name: Name of the mapping being processed
        """
        if data_df.empty:
            self.logger.warning("*")
            return
        else:
            print(' ', end='')

        errors = []

        # Track unique alternatives and entities to add
        alternatives_to_add = set()
        entities_to_add = {}  # (entity_class, entity_byname) -> entity info
        
        # Order the dataframe column levels to be reliable
        data_df.columns = data_df.columns.reorder_levels(['EntityClass', 'ParameterDefinition', 'Alternative', 'Entity_byname'])


        # First pass: collect alternatives and entities from column multi-index
        for col in data_df.columns:
            entity_class, parameter_definition, alternative, entity_byname = col

            # Collect alternatives
            if alternative and pd.notna(alternative):
                if alternative not in self.alternatives_added:
                    alternatives_to_add.add(str(alternative))
                    self.alternatives_added.add(str(alternative))

            key = (entity_class, entity_byname)
            if key not in self.entities_added:
                entities_to_add[key] = {
                    'entity_class_name': entity_class,
                    'entity_byname': entity_byname
                }
                self.entities_added[key] = {
                    'entity_class_name': entity_class,
                    'entity_byname': entity_byname
                }

        # Add alternatives
        for alt_name in alternatives_to_add:
            try:
                db_map.add_alternative(name=alt_name)
            except Exception as e:
                raise SpineDBAPIError(f"Failed to add alternative '{alt_name}': {e}")

        # Add entities
        for entity_info in entities_to_add.values():
            try:
                db_map.add_entity(entity_class_name=entity_info['entity_class_name'],
                    entity_byname=entity_info['entity_byname'])
            except Exception as e:
                raise SpineDBAPIError(f"Failed to add entity {entity_info}: {e}")
                #errors.append(f"Failed to add entity {entity_info}: {e}")

        # Second pass: add parameter values by iterating through columns
        # Determine value type based on series length and index
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
                    raise SpineDBAPIError(f"Failed to add parameter value {col_name}: {e}")

        elif value_type == 'array':
            # Array
            for col_name, col_data in data_df.items():
                array_obj = Array(values=col_data.dropna().to_list(), index_name=col_data.index.name)
                value, type_ = to_database(array_obj)

                # Add parameter value
                db_map.add_parameter_value(
                    entity_class_name=col_name[0],
                    parameter_definition_name=col_name[1],
                    alternative_name=col_name[2],
                    entity_byname=col_name[3],
                    value=value,
                    type=type_
                )

        elif value_type == 'map':
            # Map
            for col_name, col_data in data_df.items():
                df_col = data_df[col_name].dropna()
                map_obj = Map(indexes=df_col.index.to_list(), values=df_col.values.tolist(), index_name='')
                value, type_ = to_database(map_obj)

                # Add parameter value
                db_map.add_parameter_value(
                    entity_class_name=col_name[0],
                    parameter_definition_name=col_name[1],
                    alternative_name=col_name[2],
                    entity_byname=col_name[3],
                    value=value,
                    type=type_
                )

        # Commit the changes
        try:
            db_map.commit_session(f"Imported data from mapping '{mapping_name}'")
            self.logger.info(f"Successfully committed {len(data_df.columns)} columns for mapping '{mapping_name}'")
        except Exception as e:
            errors.append(f"Failed to commit session: {e}")

        # Raise errors if any occurred
        if errors:
            error_msg = f"\nErrors occurred while processing mapping '{mapping_name}':\n"
            error_msg += "\n".join(f"  - {err}" for err in errors)
            raise RuntimeError(error_msg)


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

