"""Parity tests for engine_polars/scaling.py and engine_polars/scaling_report.py.

Tests in this directory verify that the in-memory polars scaling analyzer
(analyze_solve) and report writer (write_scaling_report) produce correct,
well-formed output on the two reference models: work_all and
work_network_all_tech.

No full solver invocation is needed -- load_flextool is sufficient to
populate the FlexData bag that the scaling analyzer consumes.
"""
