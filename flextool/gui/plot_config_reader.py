from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from flextool.plot_outputs.config import _is_new_format_entry

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



def _derive_variant_letter(result_key: str, sub_config: str,
                           sub_value: dict | None = None) -> str:
    """Derive variant letter from result_key suffix and sub_config name.

    The variant letter determines the display order and label for each
    plot variant (p=period, h=hourly, w=weekly, a=aggregated).

    If *sub_value* contains an explicit ``variant`` key, that value is
    used directly.
    """
    # Explicit override
    if sub_value and 'variant' in sub_value:
        return str(sub_value['variant'])
    if sub_config == 'chunks':
        return 'w'
    if sub_config == 'sum_periods':
        return 'a'
    if sub_config == 'lines':
        return 'h'  # lines are always time-based
    # Default: derive from result_key data type
    is_dt = '_dt_' in result_key or result_key.endswith('_dt') or '_gdt_' in result_key
    if is_dt:
        return 'h'
    return 'p'


def _extract_plot_items(
    plots: dict,
) -> list[tuple[str, str, str, str, str, str]]:
    """Walk the YAML *plots* dict and yield parsed plot items.

    Each entry under ``plots:`` is an entry-name key with ``group``/``order``
    fields and nested result_keys.

    Each item is ``(group, number, variant, human_name, result_key, sub_config)``.
    Entries without ``map_dimensions_for_plots`` are skipped.
    """
    items: list[tuple[str, str, str, str, str, str]] = []
    for key, value in plots.items():
        if not isinstance(value, dict):
            continue
        if not _is_new_format_entry(value):
            continue

        entry_name = key
        group_num = str(value['group'])
        order_num = str(value['order'])
        number = f"{group_num}.{order_num}"

        for result_key, rk_value in value.items():
            if result_key in ('group', 'order') or not isinstance(rk_value, dict):
                continue
            for sub_config, sub_value in rk_value.items():
                if not isinstance(sub_value, dict):
                    continue
                if 'map_dimensions_for_plots' not in sub_value:
                    continue
                variant_letter = _derive_variant_letter(
                    result_key, sub_config, sub_value,
                )
                items.append(
                    (group_num, number, variant_letter, entry_name,
                     result_key, sub_config)
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
