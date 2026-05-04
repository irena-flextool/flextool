"""Tier-6 perturbation-as-diagnostic tests.

For each multiplier in every objective term, one test that scales it
by a non-trivial factor and asserts the new obj matches the closed-form
prediction within ``1e-6`` rel.  A failing test names the exact term
in the audit (``audit/objective_audit.md``).

The shared helpers live in ``_harness.py``.
"""
