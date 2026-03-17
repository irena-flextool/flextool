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
    'f': 'file (separate file per member)',
    'm': 'merge (sum and collapse)',
    'a': 'average and collapse',
}

# Field names for plot settings — used by _is_single_config() to detect config dicts.
PLOT_FIELD_NAMES = {
    'plot_name', 'map_dimensions_for_plots', 'subplots_per_row', 'legend',
    'bar_orientation', 'base_length', 'max_subplots_per_file', 'max_items_per_plot',
    'time_average_duration', 'xlabel', 'ylabel', 'value_label', 'axis_bounds',
    'axis_scale_min_max',  # backward compat alias for axis_bounds
    'axis_tick_format', 'always_include_zero_in_axis', 'skip_data_with_only_zeroes',
    'multiply_by',
}


@dataclass
class PlotConfig:
    """Typed, default-carrying wrapper for a single plot's settings dict.

    Callers that pass raw dicts (e.g. scenario_results.py) need no change:
    ``PlotConfig(**raw_dict)`` works as long as the dict only contains known fields.
    """
    plot_name: str = ''
    map_dimensions_for_plots: dict = field(default_factory=dict)
    subplots_per_row: int = 2
    legend: str = 'right'
    bar_orientation: str = 'horizontal'
    base_length: float = 4.0
    max_subplots_per_file: int = 6
    max_items_per_plot: int = 10
    time_average_duration: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    value_label: str | None = None
    axis_bounds: dict | list | str | None = None
    axis_tick_format: str = '1,.0f'
    always_include_zero_in_axis: bool = True
    skip_data_with_only_zeroes: bool = False
    multiply_by: float | None = None


def _is_single_config(d: dict) -> bool:
    """Return True if *d* is a single-plot config (has known field names), not a named-config dict."""
    return any(k in PLOT_FIELD_NAMES for k in d)
