"""Cross-scenario analysis: loads parquet files, combines results, generates comparison plots."""
from flextool.scenario_comparison.db_reader import get_scenario_results
__all__ = ['get_scenario_results']
