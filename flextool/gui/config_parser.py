from __future__ import annotations

import logging
from pathlib import Path

import yaml

from flextool.plot_outputs.config import flatten_new_format

logger = logging.getLogger(__name__)


def parse_plot_configs(yaml_path: Path) -> list[str]:
    """Read a plot config YAML file and return available config names.

    Supports both old format (result_key at top level) and new format
    (entry-name grouping with ``group``/``order`` keys).  New-format
    entries are flattened to result_key level before scanning.

    Returns a sorted list of config names
    (e.g. ``['chunks', 'default', 'reserve']``).
    """
    if not yaml_path.is_file():
        logger.warning("Plot config file not found: %s", yaml_path)
        return []

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Failed to read plot config %s: %s", yaml_path, exc)
        return []

    if not isinstance(data, dict):
        return []

    plots = data.get("plots")
    if not isinstance(plots, dict):
        return []

    # Flatten new-format entries so we always iterate flat result_keys
    plots = flatten_new_format(plots)

    config_names: set[str] = set()
    has_default = False

    for _result_key, result_value in plots.items():
        if not isinstance(result_value, dict):
            continue

        # Look at the first sub-key to decide whether this entry is a
        # bare plot definition (belongs to ``default``) or a mapping of
        # config-name -> plot definition.
        first_subkey = next(iter(result_value), None)
        if first_subkey == "plot_name":
            # Direct plot definition -> part of the implicit "default" config
            has_default = True
        else:
            # Each sub-key is a config name (e.g. "default", "chunks",
            # "debug", "reserve", ...).
            for config_name in result_value:
                config_names.add(config_name)

    # If any entries had a bare plot_name, they belong to "default".
    if has_default:
        config_names.add("default")

    return sorted(config_names)
