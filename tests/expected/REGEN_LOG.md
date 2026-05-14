# Golden regeneration log

This file records cases where a `tests/expected/<scenario>/<csv>` golden
was regenerated against current code rather than left as the original
v3.32.0 reference.  The convention for this project is that
`tests/expected/` is the v3.32.0 frozen output of flextool's legacy
GMPL pipeline, used as the correctness oracle for the engine_polars
cascade at HEAD (see `tests/test_scenarios.py`).  Any departure from
that convention is documented here so future readers don't have to
re-investigate why the file moved.

## 2026-05-14 — `hyphenated_entity_names/costs__dt.csv`

**Context.** Δ.22 deleted `bin/glpsol` along with the GMPL solver
pipeline.  Six v3.32.0 test scenarios had `solver=glpsol` pinned per-
entity in `tests/fixtures/tests.json` as a documented temporary
workaround for an unrelated HiGHS handoff bug (see commit `d2018b3c`'s
message).  At HEAD these pins started raising `FlexToolUserError:
Solver 'glpsol' is not installed`.  Commit `e03e3047` switched the
pins to `solver=highs` (the planned cleanup per `d2018b3c`).

Five of the six scenarios then matched their v3.32.0 goldens
byte-identically under HiGHS — same LP vertex picked.  The sixth,
**`hyphenated_entity_names`**, has a degenerate optimum: there are
two equally-optimal LP solutions for the parallel `west-east-aux`
connection (VOM=0.1 cost would push flow through it only at exact
tie-breaks).  glpsol picked one vertex; HiGHS picks the other.

- **Total objective unchanged**: still `548.39350 M CUR`, matches
  the pinned `expected_objective` at `1e-4` tolerance.
- **Cost allocation shifted**: HiGHS's chosen vertex routes zero
  flow through `west-east-aux`, so `other_operational` is zero and
  `commodity_cost` absorbs the ~75 CUR/day previously appearing in
  `other_operational`.
- **`connection__dt.csv` for the same scenario was NOT regenerated**
  — it happens to be byte-identical under HiGHS.

**Decision** (recorded by the user, 2026-05-14): **accept the regen.**
Both LP vertices are equally optimal; v3.32.0's glpsol-specific
allocation has no physical preference over HiGHS's choice once
glpsol is gone.

Cross-references:
- `tests/scenarios.yaml` carries the same note inline next to the
  `hyphenated_entity_names` entry.
- `tests/fixtures/tests.json` solver pins for the six scenarios
  were updated in commit `e03e3047` (`"glpsol"` → `"highs"` for
  scenarios: `y2020_2029_1x10y`, `y2020_2029_2x5y`,
  `hyphenated_entity_names`, `years_represented_half`,
  `years_represented_2_5`, `unidirectional_connection`).

When to revisit:
- If a future change adds LP-side tie-breaking (e.g. ε-perturbation
  on the parallel connection) that forces a deterministic vertex,
  this golden may need a second regen to match the new vertex.
- If glpsol is ever reintroduced as a supported solver, the
  fixture pins could be reverted and the original v3.32.0 golden
  restored — at which point this log entry becomes historical.
