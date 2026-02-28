"""
Convert dataframes to Spine Toolbox database.

This module provides functions to write pandas DataFrames to a Spine database,
handling entity classes, entities, parameters, and time series data.
"""

from typing import Dict, Union
import pandas as pd
from spinedb_api import DatabaseMapping, parameter_value
from spinedb_api.exception import NothingToCommit
from spinedb_api.parameter_value import to_database


def dataframes_to_spine(
    dataframes: Dict[str, pd.DataFrame],
    db_url: str,
    import_datetime: str = None,
    purge_before_import: bool = True
):
    """
    Write dataframes to Spine database.
    
    Args:
        dataframes: Dict mapping dataframe names to DataFrames
                   Entity names are in index (or MultiIndex for multi-dimensional)
                   For time series, entity names are in columns
        db_url: Database URL (e.g., "sqlite:///path/to/database.sqlite")
        import_datetime: Datetime string for alternative name (format: yyyy-mm-dd_hh-mm)
                        If None, uses current datetime
        purge_before_import: If True, purge parameter values, entities, and alternatives 
                            before import (default: True)
    """
    from datetime import datetime
    
    # Generate alternative name with datetime
    if import_datetime is None:
        import_datetime = datetime.now().strftime('%Y_%m_%d-%H_%M')
    alternative_name = f'cesm-{import_datetime}'

    with DatabaseMapping(db_url) as db_map:
        # Phase -1: Purge if requested
        if purge_before_import:
            print("Phase -1: Purging database...")
            db_map.purge_items('parameter_value')
            db_map.purge_items('entity')
            db_map.purge_items('alternative')
            db_map.purge_items('scenario')
            db_map.refresh_session()
            db_map.commit_session("Purged parameter values, entities and alternatives")
            print("  Purged parameter values, entities, and alternatives")
        
        # Separate dataframes by type
        entity_dfs = {}
        ts_dfs = {}
        str_dfs = {}
        array_dfs = {}

        for name, df in dataframes.items():
            if '.ts.' in name:
                ts_dfs[name] = df
            elif '.str.' in name:
                str_dfs[name] = df
            elif '.array.' in name:
                array_dfs[name] = df
            else:
                entity_dfs[name] = df
        
        # Phase 0: Add alternative
        print(f"Phase 0: Adding a scenario and an alternative '{alternative_name}'...")
        try:
            db_map.add_alternative(name=alternative_name)
            db_map.add_scenario(name='base')
            db_map.add_scenario_alternative(scenario_name='base',
                                            alternative_name=alternative_name,
                                            rank=0)
            db_map.commit_session(f"Added alternative {alternative_name}")
            print(f"  Added alternative: {alternative_name}")
        except Exception as e:
            print(f"  Alternative {alternative_name} already exists or error: {e}")

       # Phase 1: Add entity classes and entities
        print("Phase 1: Adding entity classes and entities...")
        _add_entity_classes_and_entities(db_map, entity_dfs, alternative_name)
        try:
            db_map.commit_session("Added entity classes and entities")
        except NothingToCommit:
            print("No entities to commit")        
        
        # Phase 2: Add parameter definitions and values
        print("Phase 2: Adding parameter definitions and values...")
        _add_parameters(db_map, entity_dfs, alternative_name)
        try:
            db_map.commit_session("Added parameter definitions and values")
        except NothingToCommit:
            print("No parameters (constants) to commit")        

        # Phase 3: Add time series parameters
        if ts_dfs:
            print("Phase 3: Adding time series parameters...")
            _add_time_series(db_map, ts_dfs, dataframes.get("timeline"), alternative_name)
            try:
                db_map.commit_session("Added time series parameters")
            except NothingToCommit:
                print("No time series parameters to commit")
        
        # Phase 4: Add str (map) parameters
        if str_dfs:
            print("Phase 4: Adding str (map) parameters...")
            _add_strs(db_map, str_dfs, alternative_name)
            try:
                db_map.commit_session("Added str parameters")
            except NothingToCommit:
                print("No time series parameters to commit")
        
        if array_dfs:
            print("Phase 5: Adding array parameters...")
            _add_arrays(db_map, array_dfs, alternative_name)
            try:
                db_map.commit_session("Added str parameters")
            except NothingToCommit:
                print("No time series parameters to commit")
        
        print("Done!")


def _get_entity_names_from_index(idx: Union[pd.Index, pd.MultiIndex]) -> list:
    """
    Extract entity names from index, converting MultiIndex tuples to strings with '__'.
    For single Index, returns list of names as-is.
    For MultiIndex, joins levels with '__'.
    """
    if isinstance(idx, pd.MultiIndex):
        # Join multi-dimensional names with '__'
        return ['__'.join(map(str, t)) for t in idx]
    else:
        return [str(name) for name in idx]


def _add_entity_classes_and_entities(db_map: DatabaseMapping, entity_dfs: Dict[str, pd.DataFrame], alternative_name: str):
    """Add entity classes and their entities."""

    # List of entity_classes that require entity_alternative to be true
    ent_alt_classes = ['unit', 'node', 'connection', 'reserve__upDown__unit__node', 'reserve__upDown__connection__node']
    
    # Sort: single-dimensional classes first (no dots), then multi-dimensional
    sorted_classes = sorted(entity_dfs.keys(), key=lambda x: ('.' in x, x))
    
    for class_name in sorted_classes:
        df = entity_dfs[class_name]
        
        # Determine if multi-dimensional
        if '.' in class_name:
            dimensions = class_name.split('.')
            
            # Get dimension names from index
            if isinstance(df.index, pd.MultiIndex):
                dimension_name_list = tuple(df.index.names)
            else:
                # Fallback if names not set
                dimension_name_list = tuple(dimensions)
            
            class_name = '__'.join(dimensions)
        else:
            dimension_name_list = None
        
        # Add entity class
        try:
            db_map.add_entity_class(
                name=class_name,
                dimension_name_list=dimension_name_list
            )
            print(f"  Added entity class: {class_name}")
        except Exception as e:
            print(f"  Entity class {class_name} already exists or error: {e}")
        
        # Add entities
        if dimension_name_list:
            # Multi-dimensional: index levels are dimensions
            if isinstance(df.index, pd.MultiIndex):
                for element_tuple in df.index.unique():
                    element_name_list = tuple(str(elem) for elem in element_tuple[1:])
                    try:
                        db_map.add_entity(
                            entity_class_name=class_name,
                            element_name_list=element_name_list,
                            name=element_tuple[0]
                        )
                    except Exception as e:
                        pass  # Entity might already exist
                    
                    if class_name in ent_alt_classes:
                        try:
                            db_map.add_entity_alternative(
                                entity_class_name=class_name,
                                element_name_list=element_name_list,
                                alternative_name=alternative_name
                            )
                        except Exception as e:
                            pass  # Entity_alternative might already exist
            else:
                # Single index but class name has dots - treat as single entity per row
                for entity_name in df.index.unique():
                    try:
                        db_map.add_entity(
                            entity_class_name=class_name,
                            element_name_list=(str(entity_name),)
                        )
                    except Exception as e:
                        pass

        else:
            # Single-dimensional: index contains entity names
            for entity_name in df.index.unique():
                try:
                    db_map.add_entity(
                        entity_class_name=class_name,
                        name=str(entity_name)
                    )
                except Exception as e:
                    pass  # Entity might already exist

                if class_name in ent_alt_classes:
                    try:
                        db_map.add_entity_alternative(
                            entity_class_name=class_name,
                            entity_byname=(str(entity_name),),
                            alternative_name=alternative_name,
                            active=True
                        )
                    except Exception as e:
                        pass  # Entity_alternative might already exist


def _add_parameters(db_map: DatabaseMapping, entity_dfs: Dict[str, pd.DataFrame], alternative_name: str):
    """Add parameter definitions and constant values."""
    
    for class_name, df in entity_dfs.items():
        # Convert class name with dots to double underscore
        if '.' in class_name:
            dimensions = class_name.split('.')
            db_class_name = '__'.join(dimensions)
        else:
            db_class_name = class_name
        
        # Get parameter columns (all columns that aren't part of the structure)
        param_cols = df.columns.tolist()
        
        # Add parameter definitions
        for param_name in param_cols:
            try:
                db_map.add_parameter_definition(
                    entity_class_name=db_class_name,
                    name=param_name
                )
            except Exception:
                pass  # Already exists
        
        # Add parameter values
        for param_name in param_cols:
            for idx, value in df[param_name].items():
                # Skip if value is None or NaN
                if pd.isna(value):
                    continue
                
                # Build entity_byname tuple
                if isinstance(idx, tuple):
                    # MultiIndex - use tuple of strings
                    entity_byname = tuple(str(elem) for elem in idx[1:])
                else:
                    # Single index
                    entity_byname = (str(idx),)
                
                # Parse value
                if isinstance(value, (int, float)):
                    parsed_value = float(value)
                elif isinstance(value, str):
                    parsed_value = value
                elif isinstance(value, list):
                    if isinstance(value[0], (int, float)):  # Assume there is only one type in the array
                        parsed_value = parameter_value.Array(value, float, 'index')
                    elif isinstance(value[0], (str)):
                        parsed_value = parameter_value.Array(value, str, 'index')
                else:
                    parsed_value = value
                
                try:
                    db_map.add_parameter_value(
                        entity_class_name=db_class_name,
                        parameter_definition_name=param_name,
                        entity_byname=entity_byname,
                        alternative_name=alternative_name,
                        parsed_value=parsed_value
                    )
                except Exception as e:
                    print(f"  Warning: Could not add value for {db_class_name}.{param_name}: {e}")


def _add_column_based_params(
    db_map: DatabaseMapping,
    dfs: Dict[str, pd.DataFrame],
    separator: str,
    alternative_name: str,
    build_value,
    type_label: str,
):
    """Shared scaffolding for column-oriented parameter writers (time series, str maps, arrays).

    For each DataFrame in *dfs*, parses the name with *separator* to extract
    (class_name, param_name), converts dots in class_name to '__', adds the
    parameter definition, and then calls build_value(df, column_index) per entity
    column to obtain (db_value, value_type) before writing the parameter value.
    """
    for name, df in dfs.items():
        parts = name.split(separator)
        if len(parts) != 2:
            print(f"  Warning: Invalid {type_label} name format: {name}")
            continue

        class_name, param_name = parts
        db_class_name = '__'.join(class_name.split('.')) if '.' in class_name else class_name

        try:
            db_map.add_parameter_definition(entity_class_name=db_class_name, name=param_name)
        except Exception:
            pass  # Already exists

        if isinstance(df.columns, pd.MultiIndex):
            entity_names = ['__'.join(map(str, col)) for col in df.columns]
        else:
            entity_names = [str(col) for col in df.columns]

        for i, entity_name in enumerate(entity_names):
            db_value, value_type = build_value(df, i)
            try:
                db_map.add_parameter_value(
                    entity_class_name=db_class_name,
                    parameter_definition_name=param_name,
                    entity_byname=(entity_name,),
                    alternative_name=alternative_name,
                    value=db_value,
                    type=value_type,
                )
                print(f"  Added {type_label}: {db_class_name}.{param_name} for {entity_name}")
            except Exception as e:
                print(f"  Warning: Could not add {type_label} for {entity_name}: {e}")


def _add_time_series(
    db_map: DatabaseMapping,
    ts_dfs: Dict[str, pd.DataFrame],
    timeline_df: pd.DataFrame,
    alternative_name: str,
):
    """Add time series parameter values."""
    if timeline_df is not None and timeline_df.index.name == 'datetime':
        start_time = pd.to_datetime(timeline_df.index[0]).isoformat()
    elif timeline_df is not None and 'datetime' in timeline_df.columns:
        start_time = pd.to_datetime(timeline_df['datetime'].iloc[0]).isoformat()
    else:
        start_time = None

    def build_ts_value(df: pd.DataFrame, i: int):
        values = df.iloc[:, i].tolist()
        if start_time and df.index.name == 'datetime':
            ts_value = {
                "type": "time_series",
                "data": values,
                "index": {"start": start_time, "resolution": "1h"},
            }
        elif df.index.name == 'datetime':
            timestamps = pd.to_datetime(df.index).strftime('%Y-%m-%dT%H:%M:%S').tolist()
            ts_value = {
                "type": "time_series",
                "data": [[ts, val] for ts, val in zip(timestamps, values)],
            }
        else:
            ts_value = {"type": "time_series", "data": values}
        return to_database(ts_value)

    _add_column_based_params(db_map, ts_dfs, '.ts.', alternative_name, build_ts_value, 'time series')


def _add_strs(db_map: DatabaseMapping, str_dfs: Dict[str, pd.DataFrame], alternative_name: str):
    """Add string indexed (map) parameter values."""
    from spinedb_api.parameter_value import Map

    def build_map_value(df: pd.DataFrame, i: int):
        map_value = Map(
            indexes=df.index.astype(str).tolist(),
            values=df.iloc[:, i].tolist(),
            index_name=df.index.name,
        )
        return to_database(map_value)

    _add_column_based_params(db_map, str_dfs, '.str.', alternative_name, build_map_value, 'str map')


def _add_arrays(db_map: DatabaseMapping, array_dfs: Dict[str, pd.DataFrame], alternative_name: str):
    """Add array parameter values."""
    from spinedb_api.parameter_value import Array

    def build_array_value(df: pd.DataFrame, i: int):
        array_value = Array(values=df.iloc[:, i].tolist(), index_name=df.index.name)
        return to_database(array_value)

    _add_column_based_params(db_map, array_dfs, '.array.', alternative_name, build_array_value, 'array')

                
# Example usage
if __name__ == "__main__":
    # Example: Create sample dataframes with new index structure
    sample_dfs = {
        'node': pd.DataFrame({
            'annual_flow': [100000.0, 80000.0, None],
            'penalty_up': [10000.0, 10000.0, 10000.0]
        }, index=pd.Index(['west', 'east', 'heat'], name='node')),
        
        'connection': pd.DataFrame({
            'efficiency': [0.90, 0.98],
            'capacity': [750.0, 500.0]
        }, index=pd.Index(['charger', 'pony1'], name='connection')),
        
        'unit.outputNode': pd.DataFrame({
            'capacity': [100.0, 50.0],
            'efficiency': [0.9, 0.95]
        }, index=pd.MultiIndex.from_tuples(
            [('coal_plant', 'west'), ('gas_plant', 'east')],
            names=['unit', 'outputNode']
        )),
        
        'node.str.inflow': pd.DataFrame({
            'west': [-1002.1, -980.7, -968, -969.1, -971.9, -957.8, -975.2, -975.1, -973.2, -800],
            'east': [-1002.1, -980.7, -968, -969.1, -971.9, -957.8, -975.2, -975.1, -973.2, -800],
            'heat': [-30, -40, -50, -60, -50, -50, -50, -50, -50, -50]
        }, index=pd.date_range('2023-01-01', periods=10, freq='H', name='datetime'))
    }
    
    timeline = pd.DataFrame(
        index=pd.date_range('2023-01-01', periods=8760, freq='H', name='datetime')
    )
    
    # Write to database
    # dataframes_to_spine(sample_dfs, "sqlite:///test_flextool.sqlite", import_datetime='2025-10-02_15-30')