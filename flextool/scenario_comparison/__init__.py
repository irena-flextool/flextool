"""Cross-scenario analysis module.

Navigation:
- data_models.py      : TimeSeriesResults, DispatchMappings — start here to understand data shapes
- db_reader.py        : Load parquet files from scenario folders → TimeSeriesResults
- dispatch_mappings.py: Load dispatch mapping parquet files → DispatchMappings
- config_io.py        : Parse/write dispatch config YAML with commented entries
- config_builder.py   : Build/update dispatch config.yaml from data
- dispatch_data.py    : Prepare per-scenario dispatch DataFrames for plotting
- dispatch_plots.py   : Render stacked area dispatch plots
- summary_plots.py    : Summary bar chart plots
- orchestrator.py     : Top-level run() function tying all pieces together
"""
from flextool.scenario_comparison.db_reader import get_scenario_results
__all__ = ['get_scenario_results']
