from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

GROUP_NAMES: dict[str, str] = {
    "0": "Infeasibilities",
    "1": "Curtailments",
    "2": "Flows",
    "3": "Costs",
    "4": "Capacities",
    "5": "Emissions",
    "6": "NodeGroups",
    "7": "Nodes",
    "8": "Unit/Conn details",
    "9": "Extras",
}


@dataclass
class PlotVariant:
    """A single variant (letter) within a plot entry."""

    letter: str  # e.g., "d", "t", "g", "a", "tf"
    result_key: str  # e.g., "node_slack_up_dt_e"
    sub_config: str  # e.g., "default", "chunks"
    full_name: str  # e.g., "Loss of load (upward slack)"


@dataclass
class PlotEntry:
    """A second-level entry in the plot tree (e.g., "0.0")."""

    number: str  # e.g., "0.0"
    short_name: str  # e.g., "Loss of load (upwar..."
    full_name: str  # e.g., "Loss of load (upward slack)"
    variants: list[PlotVariant] = field(default_factory=list)


@dataclass
class PlotGroup:
    """A first-level group in the plot tree (e.g., "0")."""

    number: str  # e.g., "0"
    name: str  # e.g., "Infeasibilities"
    entries: list[PlotEntry] = field(default_factory=list)


def _parse_plot_name(plot_name: str) -> tuple[str, str, str, str]:
    """Parse a plot_name string into (group, entry_number, variant, human_name).

    Examples::

        "6.0.t NodeGroup dispatch"  -> ("6", "6.0", "t", "NodeGroup dispatch")
        "5.0 Emissions CO2 total"   -> ("5", "5.0", "",  "Emissions CO2 total")
    """
    parts = plot_name.split(None, 1)
    token = parts[0]
    human_name = parts[1] if len(parts) > 1 else ""

    segments = token.split(".")
    group = segments[0]
    entry_num = segments[1] if len(segments) > 1 else "0"
    variant = segments[2] if len(segments) > 2 else ""
    number = f"{group}.{entry_num}"

    return group, number, variant, human_name


def _extract_plot_items(
    plots: dict,
) -> list[tuple[str, str, str, str, str, str]]:
    """Walk the YAML *plots* dict and yield parsed plot items.

    Each item is ``(group, number, variant, human_name, result_key, sub_config)``.
    Entries without ``map_dimensions_for_plots`` are skipped.
    """
    items: list[tuple[str, str, str, str, str, str]] = []
    for result_key, value in plots.items():
        if not isinstance(value, dict):
            continue

        if "plot_name" in value:
            # Direct config (no sub-config level)
            if "map_dimensions_for_plots" not in value:
                continue
            group, number, variant, human_name = _parse_plot_name(value["plot_name"])
            items.append((group, number, variant, human_name, result_key, "default"))
        else:
            # Sub-config level
            for sub_config, sub_value in value.items():
                if not isinstance(sub_value, dict):
                    continue
                if "plot_name" not in sub_value:
                    continue
                if "map_dimensions_for_plots" not in sub_value:
                    continue
                group, number, variant, human_name = _parse_plot_name(
                    sub_value["plot_name"]
                )
                items.append(
                    (group, number, variant, human_name, result_key, sub_config)
                )

    return items


def parse_plot_config(config_path: Path) -> list[PlotGroup]:
    """Parse a plot config YAML and return the tree structure.

    Returns a list of :class:`PlotGroup` objects sorted by group number.
    Each group contains :class:`PlotEntry` objects sorted by entry number,
    and each entry contains :class:`PlotVariant` objects for available
    variant letters.

    Entries without ``map_dimensions_for_plots`` are excluded.
    """
    if not config_path.is_file():
        logger.warning("Plot config file not found: %s", config_path)
        return []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        logger.error("Failed to read plot config %s: %s", config_path, exc)
        return []

    if not isinstance(data, dict):
        return []

    plots = data.get("plots")
    if not isinstance(plots, dict):
        return []

    items = _extract_plot_items(plots)

    # Group by entry number, collecting variants
    # Use dict to preserve insertion order; entries keyed by number
    entry_map: dict[str, PlotEntry] = {}
    for group, number, variant, human_name, result_key, sub_config in items:
        if number not in entry_map:
            entry_map[number] = PlotEntry(
                number=number,
                short_name=human_name,
                full_name=human_name,
                variants=[],
            )
        entry = entry_map[number]
        # Update full_name if this variant has a longer/better name
        # (keep the first one encountered for consistency)
        pv = PlotVariant(
            letter=variant,
            result_key=result_key,
            sub_config=sub_config,
            full_name=human_name,
        )
        # Avoid duplicate variant letters for the same entry
        existing_letters = {v.letter for v in entry.variants}
        if variant not in existing_letters:
            entry.variants.append(pv)

    # Build group structure
    group_map: dict[str, PlotGroup] = {}
    for number, entry in entry_map.items():
        group_num = number.split(".")[0]
        if group_num not in group_map:
            group_map[group_num] = PlotGroup(
                number=group_num,
                name=GROUP_NAMES.get(group_num, f"Group {group_num}"),
                entries=[],
            )
        group_map[group_num].entries.append(entry)

    # Sort groups by number, entries by number within each group
    groups = sorted(group_map.values(), key=lambda g: int(g.number))
    for group in groups:
        group.entries.sort(key=lambda e: [int(x) for x in e.number.split(".")])

    return groups
