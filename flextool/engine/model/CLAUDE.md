# Model rules (`flextool/engine/model/`)

- **Headers.** Every equation, objective term and variable has its `EQ.` / `OBJ.` / `VAR.` header (formula, what it
  does, notes; format in `docs/dev/engine.md`).  Writing the formula is a review: if it disagrees with the code or
  the user docs, stop.  `tests/docs/test_header_grammar.py` checks that every LP row and column has a header.
- **Autoscale registries.** Every new `add_var` / `add_cstr` is registered in
  `flextool/engine/autoscale/_layer2_types.py` (`VARIABLE_FAMILIES` / `CONSTRAINT_FAMILIES`) and every new schema
  parameter in `_quantity_types.py` (`PARAMETER_TYPES`).  A gap does not crash a user run: the solve silently drops
  to the unscaled LP.  Tests run with `FLEXTOOL_AUTOSCALE_STRICT=1`; dynamic f-string constraint names are pinned in
  `tests/engine/autoscale/test_registry_coverage.py`.
- **`core/` is a package, one module per equation family** (map in `core/__init__.py`, step table in
  `core/_build.py`).  `core/__init__.py` re-exports the public names only: import a private helper from the module
  that defines it, and patch a name in a test on the module that reads it (a patch on `core` reaches no submodule).
  A new family goes into its family's module, or a new one ≤ 1 500 lines; the formulation page follows the table.
- **Prove the LP effect** with `tools/lp_snapshot.py`: byte-identical for a refactor, every diff explained for a fix.
- **Parity with the old GMPL model is not correctness** — several GMPL bugs were ported faithfully.  Check copied
  logic against the user docs.
- **Several shapes, one LP.** A parameter that accepts constant / period / time / map values needs a
  constant-vs-equivalent-map test giving the same LP in single, rolling and stochastic solves.
- **Economics, not heuristics.** Cost and objective terms follow first-principles economics (cash flows at their
  moments, financing at the investor's rate, time weighting at the model's rate).  Expected outcomes such as
  "retire-and-rebuild never pays" are checks of the principle, never targets to engineer a special case toward.
- `node.storage_binding_method` is one value per node.  Each `*_blended_weights` method owns its whole continuity
  (never leans on a `bind_within_*` method: double count); `bind_intraperiod_blocks` is an aggregation using
  `nodeBalanceBlock_eq`.
