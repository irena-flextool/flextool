"""Input data reading: converts CSV/Excel/ODS files to Spine database format."""
from flextool.process_inputs.read_tabular_with_specification import TabularReader
from flextool.process_inputs.write_to_input_db import write_to_flextool_input_db
__all__ = ['TabularReader', 'write_to_flextool_input_db']
