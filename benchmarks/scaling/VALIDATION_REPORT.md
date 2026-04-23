# Scaling validation

Quick reference for the LP-scaling benchmark harness. Detailed
per-agent history lives in the git log (grep for `Agent `). For
concept and user-facing flags, see `flextool/SCALING_USER_GUIDE.md`
and `flextool/SLACK_CONVENTION.md`.

## What's in this directory

- `run_benchmarks.py` — driver. Generates scenarios, solves them,
  captures objective + matrix/cost/bound/RHS ranges + slack totals,
  compares to committed baselines.
- `scenarios/<name>/generate.py` — one per scenario. Produces
  `input.sqlite` (gitignored).
- `baseline/<name>.json` — committed reference metrics per scenario.
- `baseline/CHANGELOG.md` — when and why the baselines were
  refreshed.
- `README.md` — how to run.

## Covered scenarios

- `small_building` — single node, 2–3 units, 48 h dispatch.
- `medium_national` — ~6 nodes, ~20 units, 48 h with investment.
- `continental` — rivendell-structure slice, 2 periods × 16
  timeslices.
- `composite` — directly connects a tiny-scale node (0.01 MW heatpump)
  to a large-scale node (10 000 MW coal + 5 000 MW wind). Primary
  stress test for composite-scale mismatch diagnostics.

## How to validate current state

```bash
for s in small_building medium_national continental composite; do
  python benchmarks/scaling/run_benchmarks.py \
    --scenario "$s" \
    --compare "benchmarks/scaling/baseline/${s}.json"
done
```

Exit 0 on all four = no regression vs committed baselines.

## When to refresh baselines

Only when a deliberate, explained change to the scaling stack
legitimately shifts the ranges or column counts. Record the reason in
`baseline/CHANGELOG.md`, regenerate with
`python benchmarks/scaling/run_benchmarks.py --write-baseline`.
Objectives and slack totals are **always invariant** to the scaling
stack — any objective delta is a real regression signal, not a
baseline refresh opportunity.

## Scope limits

This directory is a developer regression tool, not CI-wired. Rivendell
and other large real-world models are tested out-of-tree; see past
reports under `projects/rivendell/` (gitignored) if they still exist.
