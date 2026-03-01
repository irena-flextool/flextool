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
