# Backward-compatibility shim — plot_dict_of_dataframes now lives in orchestrator.py
from flextool.plot_outputs.orchestrator import plot_dict_of_dataframes

__all__ = ['plot_dict_of_dataframes']
