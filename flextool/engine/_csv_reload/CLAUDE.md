# CSV reload (`flextool/engine/_csv_reload/`) — shelved

Rebuilding a model from a CSV dump without the database is shelved; it is revisited only when the owner names a
real need.  Don't extend this package, add `SEEDED_FIELDS` seeds or fix reload gaps you notice — note them in your
report instead.  The CSV **dump** (`flextool/engine/output/_dump_csvs.py`) is maintained: a new FlexData field gets
a dump entry or a stated reason why not (`tests/engine/test_flexdata_field_routes.py`).  The reload tests
(`tests/csv_reload/`, `-m csv_reload`) stay opt-in.
