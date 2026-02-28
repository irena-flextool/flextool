from dataclasses import dataclass, field

# Dimension rule characters come from templates/default_plots.yaml and must remain unchanged.
# This dict documents their meaning for readability in code (e.g. DIMENSION_RULES['t'] → 'time axis (x)').
DIMENSION_RULES: dict[str, str] = {
    't': 'time axis (x, datetime)',
    'i': 'time-like axis (x, chunk-average)',
    'b': 'bar/line column stack',
    'e': 'expand (separate y-axes)',
    'g': 'group (grouped bars)',
    's': 'stack (stacked segments)',
    'u': 'subplot (separate plot panel)',
    'l': 'line series',
    'm': 'merge (sum and collapse)',
    'a': 'average and collapse',
}

# Field names for plot settings — used by _is_single_config() to detect config dicts.
PLOT_FIELD_NAMES = {
    'plot_name', 'map_dimensions_for_plots', 'subplots_per_row', 'legend',
    'bar_orientation', 'base_length', 'max_subplots_per_file', 'max_items_per_file',
    'time_average_duration', 'xlabel', 'ylabel', 'value_label', 'axis_scale_min_max',
    'axis_tick_format', 'always_include_zero'
}


@dataclass
class PlotConfig:
    """Typed, default-carrying wrapper for a single plot's settings dict.

    Callers that pass raw dicts (e.g. scenario_results.py) need no change:
    ``PlotConfig(**raw_dict)`` works as long as the dict only contains known fields.
    """
    plot_name: str = ''
    map_dimensions_for_plots: dict = field(default_factory=dict)
    subplots_per_row: int = 3
    legend: str = 'right'
    bar_orientation: str = 'horizontal'
    base_length: float = 4.0
    max_subplots_per_file: int = 9
    max_items_per_file: int | None = None
    time_average_duration: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    value_label: str | None = None
    axis_scale_min_max: dict | list | None = None
    axis_tick_format: str | None = None
    always_include_zero: bool = True


def _is_single_config(d: dict) -> bool:
    """Return True if *d* is a single-plot config (has known field names), not a named-config dict."""
    return any(k in PLOT_FIELD_NAMES for k in d)
