"""Python preprocessing for derived sets and calculated parameters.

Modules here compute values that used to be derived inside
``flextool.mod`` via ``setof`` / filter expressions or calculated-
parameter blocks. Each module owns one related family of sets/params,
takes already-loaded DB inputs, and writes CSVs into ``solve_data/``
that ``flextool.mod`` reads via ``table data IN`` declarations.

Style discipline (enforced by tests/test_preprocessing_ordered_set_lint.py):

- No bare ``set()`` constructors, set literals, or set comprehensions.
- Use ``dict.fromkeys(iterable)`` for ordered, deduplicated containers.
- ``frozenset`` is permitted only when wrapping an already-ordered
  source for hot-path membership testing.

See ``migration/README.md`` for the per-step workflow.
"""
