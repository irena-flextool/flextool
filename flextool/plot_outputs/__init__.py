"""
Plot Outputs Module
===================

Entry point : orchestrator.plot_dict_of_dataframes()
Config      : config.PlotConfig, config.DIMENSION_RULES
Line plots  : plot_lines
Bar charts  : plot_bars  (orchestration), plot_bars_detail (rendering)
Shared util : subplot_helpers, axis_helpers, legend_helpers, format_helpers
Performance : perf
"""
from flextool.plot_outputs.orchestrator import plot_dict_of_dataframes, prepare_plot_data, compute_all_plot_plans

__all__ = ['plot_dict_of_dataframes', 'prepare_plot_data', 'compute_all_plot_plans']
