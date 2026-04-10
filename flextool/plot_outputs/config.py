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
    'max_items_per_subplot_column',
    'time_average_duration', 'xlabel', 'ylabel', 'value_label', 'axis_bounds',
    'axis_scale_min_max',  # backward compat alias for axis_bounds
    'axis_tick_format', 'always_include_zero_in_axis', 'skip_data_with_only_zeroes',
    'multiply_by', 'full_timeline', 'subplots_by_magnitudes',
    'variant',  # used by plot_config_reader to override variant letter derivation
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
    legend: str = 'right'  # 'right', 'all', or 'shared'
    bar_orientation: str = 'horizontal'
    base_length: float = 4.0
    max_subplots_per_file: int = 6
    max_items_per_plot: int = 10
    max_items_per_subplot_column: int = 40
    time_average_duration: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    value_label: str | None = None
    axis_bounds: dict | list | str | None = None
    axis_tick_format: str = 'dynamic'
    always_include_zero_in_axis: bool = True
    skip_data_with_only_zeroes: bool = False
    multiply_by: float | None = None
    full_timeline: bool = False
    subplots_by_magnitudes: bool = False


def _is_single_config(d: dict) -> bool:
    """Return True if *d* is a single-plot config (has known field names), not a named-config dict."""
    return any(k in PLOT_FIELD_NAMES for k in d)


def _is_new_format_entry(d: dict) -> bool:
    """Return True if *d* is an entry-name-based config (has 'group' and 'order' keys)."""
    return 'group' in d and 'order' in d


def flatten_new_format(plots: dict) -> dict:
    """Convert entry-name-based format to flat result_key-based format.

    Entries have ``group`` and ``order`` keys; result_keys are nested
    underneath.  The flattened output maps each result_key directly to
    its sub-config dict (same shape the orchestrator/plan code expects).

    When the same result_key appears under multiple entry names (e.g.
    ``costs_dt_p`` under both "Costs" and "Costs lines"), the sub-configs
    from each entry are merged into a single dict for that result_key.

    Already-flat entries (result_key → config dict without group/order)
    pass through unchanged for programmatic callers.
    """
    flat: dict = {}
    for key, value in plots.items():
        if not isinstance(value, dict):
            continue
        if _is_new_format_entry(value):
            for rk, rk_val in value.items():
                if rk in ('group', 'order') or not isinstance(rk_val, dict):
                    continue
                if rk in flat:
                    flat[rk].update(rk_val)
                else:
                    flat[rk] = dict(rk_val)
        else:
            # Already flat (result_key → config dict) — pass through
            flat[key] = value
    return flat
