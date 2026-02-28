"""Backward-compatibility shim. Import from the individual modules instead."""
from flextool.process_outputs.read_variables import read_variables
from flextool.process_outputs.read_parameters import read_parameters
from flextool.process_outputs.read_sets import read_sets

__all__ = ['read_variables', 'read_parameters', 'read_sets']
