"""Tier-7 constraint-emission introspection tests.

For every constraint family, one test that builds the LP from a fixture
exercising the constraint and asserts that it shows up in
``Problem.cstrs_named`` with the expected number of rows.  Catches
"constraint declared but never bound" silent failures — bugs that don't
necessarily change the optimal objective but leave the LP under-
constrained.

The shared helpers live in ``_helpers.py``.  Tests do **not** solve the
LP — they only inspect what ``build_flextool`` emits, which makes them
fast (build is ~10-20 ms vs. solve at ~50-200 ms) and excellent
candidates for the smoke layer.
"""
