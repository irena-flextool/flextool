"""Color and column-name constants for dispatch and summary plots."""

# Default color mapping for special columns
DEFAULT_SPECIAL_COLORS = {
    # Positive special columns (at top of legend/plot)
    'LossOfLoad': 'crimson',
    'Discharge': 'aqua',
    'Import': 'indigo',
    # Negative special columns (at bottom of legend/plot)
    'Charge': 'lime',
    'Export': 'purple',
    'internal_losses': 'darkgray',
}

# Special columns that should be POSITIVE (at top of stacked plot, top of legend)
POSITIVE_SPECIAL = ['LossOfLoad', 'Discharge', 'Import']
# Special columns that should be NEGATIVE (at bottom of stacked plot, bottom of legend)
NEGATIVE_SPECIAL = ['Charge', 'Export', 'internal_losses']
# Columns plotted as lines, not stacked areas
LINE_COLUMNS = ['Curtailed', 'Demand']

# Ramp envelope band colours (one per band, up and down alike), the fallback of ``categories.ramp_envelope`` in
# plot_settings.yaml (scenario_comparison/ramp_plots.py).  Plots are drawn on a white background in light and
# dark GUI themes alike.
DEFAULT_RAMP_BAND_COLORS = {
    'units': '#4e79a7',        # steel blue — unit flows
    'storage': '#76b7b2',      # teal — flows from / to storages (the dispatch Discharge family)
    'vre': '#edc948',          # yellow — VRE curtailment release / extra curtailment
    'connections': '#b07aa1',  # purple — connection flows (the dispatch Import / Export family)
}
