"""YAML config file I/O with comment preservation for dispatch plots."""

import os
import yaml


def _yaml_quote(value: str) -> str:
    """Quote a string if it contains YAML-special characters."""
    special = set('(),[]{}&*#?|->!%@`"\'')
    if any(c in special for c in str(value)):
        # Use double quotes, escaping internal double quotes
        escaped = str(value).replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


def _yaml_color(color: str) -> str:
    """Quote hex color strings starting with #."""
    if str(color).startswith('#'):
        return f'"{color}"'
    return str(color)


def parse_config_with_comments(config_path: str) -> tuple[dict, dict[str, dict[str, str] | set[str]]]:
    """
    Parse a YAML config file while tracking commented-out entries.

    Handles both new dict-format scenarios and old list-format scenarios.
    For scenarios: commented_entries['scenarios'] is dict[str, str] (name->color).
    For nodes: commented_entries['nodes'] is set[str].
    No comment tracking for positive/negative sections.

    Returns:
    --------
    tuple : (config_dict, commented_entries)
        - config_dict: parsed YAML content
        - commented_entries: dict mapping section -> commented items
          'scenarios' -> dict[str, str] (name->color)
          'nodes' -> set[str]
    """
    commented_entries: dict[str, dict[str, str] | set[str]] = {
        'scenarios': {},
        'nodes': set(),
    }

    if not os.path.exists(config_path):
        return {}, commented_entries

    with open(config_path, 'r') as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        # Check for section headers (top-level keys)
        if line and not line.startswith(' ') and not line.startswith('#') and ':' in line:
            current_section = line.split(':')[0].strip()

        stripped = line.strip()
        if not current_section or not stripped.startswith('#'):
            continue

        # Only parse comments in scenarios and nodes sections
        if current_section == 'scenarios':
            # Commented scenario: #name: color or #- name
            item_line = stripped.lstrip('#').strip()
            if item_line.startswith('- '):
                # Old list format: #- name
                name = item_line[2:].strip().strip('"').strip("'")
                if ':' in name:
                    name = name.split(':')[0].strip()
                commented_entries['scenarios'][name] = ''
            elif ':' in item_line:
                # New dict format: #name: color
                parts = item_line.split(':', 1)
                name = parts[0].strip().strip('"').strip("'")
                color = parts[1].strip().strip('"').strip("'") if len(parts) > 1 else ''
                commented_entries['scenarios'][name] = color

        elif current_section == 'nodes':
            item_line = stripped.lstrip('#').strip()
            if item_line.startswith('- '):
                item = item_line[2:].strip().strip('"').strip("'")
                commented_entries['nodes'].add(item)

    # Load the actual YAML (uncommented parts)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f) or {}

    return config_dict, commented_entries


def _write_sign_section(section_name: str, config_dict: dict, lines: list[str]) -> None:
    """Write a positive/negative section with processGroups and processes_not_aggregated."""
    section = config_dict.get(section_name)
    if section is None:
        return
    lines.append(f"{section_name}:")

    pg = section.get('processGroups')
    if pg and isinstance(pg, dict):
        lines.append("  processGroups:")
        for name, color in pg.items():
            lines.append(f"    {_yaml_quote(name)}: {_yaml_color(color)}")
    else:
        lines.append("  processGroups: {}")

    pna = section.get('processes_not_aggregated')
    if pna and isinstance(pna, dict):
        lines.append("  processes_not_aggregated:")
        for name, color in pna.items():
            lines.append(f"    {_yaml_quote(name)}: {_yaml_color(color)}")
    else:
        lines.append("  processes_not_aggregated: {}")

    lines.append("")


def write_config_with_comments(config_path: str, config_dict: dict,
                               commented_entries: dict[str, dict[str, str] | set[str]]) -> None:
    """
    Write config to YAML file with commented entries preserved.

    New format:
    - time_to_plot: first_timestep, number_of_timesteps
    - scenarios: dict name->color (commented = #name: color)
    - positive: processGroups and processes_not_aggregated as dict name->color
    - negative: processGroups and processes_not_aggregated as dict name->color
    - nodes: list (commented = #- name)

    No nodeGroups section, no colors section.

    Parameters:
    -----------
    config_path : str or Path
        Path to write the config file
    config_dict : dict
        Main config dictionary with active entries
    commented_entries : dict
        'scenarios' -> dict[str, str] (name->color) for commented scenarios
        'nodes' -> set[str] for commented nodes
    """
    lines = []

    lines.append("# The color codes in the config file can be replaced by appropriate colors.")
    lines.append("# You can use named colors from: https://matplotlib.org/stable/gallery/color/named_colors.html")
    lines.append("# Deleting the config files resets the colors.")
    lines.append("")

    # time_to_plot
    if 'time_to_plot' in config_dict:
        value = config_dict['time_to_plot']
        lines.append("time_to_plot:")
        lines.append(f"  first_timestep: {value.get('first_timestep', 0)}")
        lines.append(f"  number_of_timesteps: {value.get('number_of_timesteps', 168)}")
        lines.append("")

    # scenarios (dict format with comments)
    if 'scenarios' in config_dict:
        lines.append("scenarios:")
        scenarios = config_dict['scenarios']
        if isinstance(scenarios, dict):
            for name, color in scenarios.items():
                lines.append(f"  {_yaml_quote(name)}: {_yaml_color(color)}")
        # Commented scenarios
        commented_scens = commented_entries.get('scenarios', {})
        if isinstance(commented_scens, dict):
            for name in sorted(commented_scens.keys()):
                color = commented_scens[name]
                lines.append(f"  #{_yaml_quote(name)}: {_yaml_color(color)}")
        lines.append("")

    _write_sign_section('positive', config_dict, lines)
    _write_sign_section('negative', config_dict, lines)

    # nodes (list format with comments)
    if 'nodes' in config_dict:
        lines.append("nodes:")
        for item in config_dict['nodes']:
            lines.append(f"  - {item}")
        for item in sorted(commented_entries.get('nodes', set())):
            lines.append(f"  #- {item}")
        lines.append("")

    with open(config_path, 'w') as f:
        f.write('\n'.join(lines))
