#!/usr/bin/env python3
"""
Script to convert JSON schema from flat mapping structure to nested table structure.
Converts import_flex_short.json format to import_flex_short_modified.json format.
"""

import json
import sys
from typing import Dict, Any, List


def convert_schema(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the input schema from flat mapping structure to nested table structure.
    
    Args:
        input_data: Dictionary containing the original schema structure
        
    Returns:
        Dictionary containing the converted schema structure
    """
    # Initialize the output structure with top-level fields
    output_data = {
        "name": input_data.get("name", ""),
        "description": input_data.get("description", ""),
        "item_type": input_data.get("item_type", ""),
        "source_type": input_data["mapping"].get("source_type", ""),
        "tables": {},
        "selected_tables": input_data["mapping"].get("selected_tables", [])
    }
    
    mapping_data = input_data.get("mapping", {})
    table_mappings = mapping_data.get("table_mappings", {})
    table_options = mapping_data.get("table_options", {})
    table_types = mapping_data.get("table_types", {})
    table_default_column_type = mapping_data.get("table_default_column_type", {})
    table_row_types = mapping_data.get("table_row_types", {})
    
    # Process each table
    for table_name, mappings_list in table_mappings.items():
        table_config = {}
        
        # Add table options if they exist
        if table_name in table_options:
            table_config["options"] = table_options[table_name]
        
        # Add column types if they exist
        if table_name in table_types:
            table_config["column_types"] = table_types[table_name]
        
        # Add default column type if it exists
        if table_name in table_default_column_type:
            table_config["default_column_type"] = table_default_column_type[table_name]
        
        # Add row types if they exist
        if table_name in table_row_types:
            table_config["row_types"] = table_row_types[table_name]
        
        # Process mappings for this table
        table_config["mappings"] = {}
        
        for mapping_dict in mappings_list:
            for mapping_name, mapping_content in mapping_dict.items():
                # Extract the mapping array from the nested structure
                mapping_array = mapping_content.get("mapping", [])
                table_config["mappings"][mapping_name] = mapping_array
        
        output_data["tables"][table_name] = table_config
    
    # Handle tables that might be in other sections but not in table_mappings
    # (like process__reserve__updown__node__period_average in the example)
    all_table_names = set(table_mappings.keys())
    all_table_names.update(table_options.keys())
    all_table_names.update(table_types.keys())
    all_table_names.update(table_default_column_type.keys())
    all_table_names.update(table_row_types.keys())
    
    for table_name in all_table_names:
        if table_name not in output_data["tables"]:
            table_config = {}
            
            # Add table options with proper key name for this table
            if table_name in table_options:
                table_config["table_options"] = table_options[table_name]
            
            # Add column types with proper key name for this table
            if table_name in table_types:
                table_config["table_types"] = table_types[table_name]
            
            # Add default column type
            if table_name in table_default_column_type:
                table_config["table_default_column_type"] = table_default_column_type[table_name]
            
            # Add row types with proper key name for this table
            if table_name in table_row_types:
                table_config["table_row_types"] = table_row_types[table_name]
            
            # Add empty mappings if no mappings exist for this table
            if table_name not in table_mappings:
                table_config["mappings"] = {}
            
            output_data["tables"][table_name] = table_config
    
    return output_data


def main():
    """
    Main function to handle command line arguments and file processing.
    """
    if len(sys.argv) != 3:
        print("Usage: python convert_schema.py <input_file.json> <output_file.json>")
        print("Example: python convert_schema.py import_flex_short.json import_flex_short_modified.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Convert the schema
        output_data = convert_schema(input_data)
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully converted {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file '{input_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()