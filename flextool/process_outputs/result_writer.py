"""Backward-compatible re-export shim. Import from write_outputs instead."""
from flextool.process_outputs.write_outputs import (
    write_outputs,
    write_summary_csv,
    log_time,
    print_namespace_structure,
    ALL_OUTPUTS,
)

__all__ = ['write_outputs', 'write_summary_csv', 'log_time', 'print_namespace_structure', 'ALL_OUTPUTS']
