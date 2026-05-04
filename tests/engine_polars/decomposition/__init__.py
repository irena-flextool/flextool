"""Tier-8 closed-form objective-decomposition tests.

For each major scenario fixture, solve the LP and *independently*
recompute every component of the objective from the solved variable
values.  Assert the sum equals ``sol.obj`` to ~1e-9 rel — catching
double-counts and sign flips structurally.

A failure prints a per-component diagnostic dict so the missing /
mis-counted term is immediately visible.

The component-by-component closed forms live in :mod:`_components` and
mirror, term-for-term, the obj construction in
``flextool/model.py:944-1212``.

NOTE: ``pdt_branch_weight`` is universally MISSING in flexpy and the
audit lists it as such (see ``audit/objective_audit.md`` §11).  Every
component therefore treats branch weight as 1.0 — which matches what
flexpy's model emits today.  When stochastic invest / branch weights
land, these helpers will need a `pdt_branch_weight` join too.
"""
