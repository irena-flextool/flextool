# Test rules (`tests/`)

- **Which tests:** the modules your change touches or that test it (`tools/related_tests.py`), new tests, and
  `tools/lp_snapshot.py` for model changes; `-n` ≤ 8.  No directory or `-k` keyword sweeps, no "baseline before
  edits" runs, never the whole `tests/test_scenarios.py` (name the scenarios by node id).  The full suite runs at
  milestones only (a hook enforces this).  `tests/csv_reload/` (the shelved reload) runs only with
  `-m csv_reload`.  Anything that opens Tk windows (`tests/gui`) runs under `xvfb-run -a`, never on the owner's
  display.
- **Databases come from JSON** in `tmp_path` (`json_to_db`, the session-scoped `schema_db_url`).  Never read a
  template `.sqlite` in place: it lags the schema, and concurrent xdist opens of one SQLite file can hang a Windows
  shard.  If one must be read, open it read-only or from a `tmp_path` copy (`tests/test_invariant_guards.py`).
- **Expected values are derived independently** — from the fixture JSON, the docs and the EQ headers, written down in
  a comment — never copied from the new code's output.  Prefer relational assertions (twin == case, warm == cold,
  constant == map, conservation).  A regression test fails before the fix and passes after.
- **Test economy:** add tests for result correctness (one regression test per fix that changes LP results or
  invested amounts).  Validation-only and reader fixes are proven by `lp_snapshot`; other test ideas go to
  `specs/test_proposals.md`.  Keep finer-grained tests that localise a failure even when a broader gate covers it.
- **Goldens** change only with an explanation: `tests/expected/REGEN_LOG.md` entry, named scenarios only.
- **Folders** are named by what is tested (`tests/model/`, `tests/engine/<area>/`), never by implementation layer;
  avoid a basename that already exists elsewhere under `tests/` (collection clash).
- **Flakes:** check across separate processes (a loop of single pytest runs), since hash order differs per process.
- A new test module gets a `tests/ci_durations.json` entry (its seconds; `tools/ci_durations.py` rebuilds the file
  from CI logs) so the CI shards stay balanced.
