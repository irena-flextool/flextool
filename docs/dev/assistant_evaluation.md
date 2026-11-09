# Evaluating the AI assistant's levels

The assistant works at three levels: basic, standard and advanced. Each level has its own instructions, tools and
token budgets. FlexTool places a model at a level from its name, its size and its context window. The
**calibration** checks those placements and budgets on questions whose answers are known. It is run by hand, at
release time or when the levels change. It never runs in CI, and pytest never collects it.

Everything lives under `tools/`, which is not packaged:

| file | what it holds |
|---|---|
| `tools/assistant_eval.py` | the command line: `build`, `run`, `grade`, `summary`, `calibrate`, `dual` |
| `tools/assistant_eval_scoring.py` | the question set and answer key formats, and the automatic checks |
| `tools/assistant_eval_analysis.py` | the calibration tables and the dual grading |
| `tools/assistant_eval_data/grounding_set.yaml` | the questions, each with a difficulty class |
| `tools/assistant_eval_data/answer_key.yaml` | the answers, written by an independent oracle |
| `tools/assistant_eval_data/results.jsonl` | the committed summary, one line per graded answer |
| `tests/assistant/grounding_projects.py` | the synthetic project the questions are about |

## The questions and their classes

Each question has one of three difficulty classes:

- **lookup**: one value or fact. This includes units, inactive entities, a failed run, and questions the data
  cannot answer.
- **interpret**: a question across tables or scenarios, or about an equation.
- **diagnose**: a causal chain over several steps, such as why there is unserved energy or why an investment has
  its size.

The basic level is meant for lookup questions. Standard adds interpret questions, and advanced adds diagnose
questions. Every model answers every question at every level setting. This shows what a model does at its own
level, and whether it escalates cleanly at a lower one.

## The grounding project

`python tools/assistant_eval.py build --out DIR` builds the project the questions are about and solves it (about
half a minute). The project uses `tests/fixtures/tests.json` with three override alternatives:

- `h2_lookup`: a storage node whose description gives its own units;
- `demand_update`: a run that succeeds, after which the source is rewritten so that the next run fails in the
  input checks;
- `invest_2day`: an investment version of the 48-hour solve.

The fixture's scenarios `coal` and `coal_co2_price` are solved as they are. The project folder name has a space
and an `ä` in it.

`build` writes a marker file, `.flextool_grounding_project.json`. The marker holds a hash of the source database,
the run id of each result folder, and a fingerprint of the source, the overrides and the run ids. `run` refuses a
project without a valid marker, and one that has changed since it was built: another database, another input
source, results of another run, or a run of its own made again. It also refuses a project built without its runs.
Data from a real project therefore never reaches a calibration model.

Two builds are never the same project. The source database differs in its hash from build to build (the database
records its own times), and every run gets a new run id. A rebuild therefore starts a new matrix: results folders
made on an earlier build cannot be continued with `--resume` on the new one, nor summarised or sampled together
with results of the new one. Keep the build for as long as its results are being run and graded.

## Running the models

```
python tools/assistant_eval.py run --project DIR/"Grounding project ä" --provider PRESET --model MODEL \
    --out _audit_reports/assistant_eval/MODEL_NAME [--levels basic,standard,advanced] [--accept-data-notice]
```

- **Keys** are read from the system keyring or from the provider's environment variable. They are never given on
  the command line or in a file.
- **The data notice** is the one the chat shows. `--accept-data-notice` accepts it for this run only, and only for
  a grounding project: nothing is kept, so every run on a remote service needs the flag (also with `--resume`), an
  acceptance in the chat does not cover a calibration, and the chat still asks. When the probe shows that a model
  is forwarded (an Ollama cloud model), that route's notice is asked as well.
- **One record per cell.** A cell is one model at one level setting answering one question. The records go to the
  `--out` folder. They are redacted, and they stay local: `_audit_reports/` is git-ignored. Each question gets a
  fresh conversation. The eight turns of S01 share one conversation, and S01 runs only at the model's automatic
  level.
- **Local models (Ollama).** Make one variant of the model for each context size. For example, write a Modelfile
  with `FROM <model>` (a model you have pulled) and `PARAMETER num_ctx 32768`, then run
  `ollama create <model>-32k -f Modelfile`.
  Do the same for `-16k` and `-64k`. Then pass
  `--ollama-context basic=16384,standard=32768,advanced=65536`. Before the first cell of each setting, the runner
  loads that setting's variant and records its context and memory use. If the variant runs with a smaller context
  than asked, or does not fit fully in graphics memory, every cell of that setting is recorded as "not applicable
  (context)" and not run.
  On a card whose memory is partly used by the desktop, a model may never fit fully. `--allow-cpu-offload` runs
  a variant that loaded with the asked context but is partly on the CPU. Each cell records the offload: the
  model's size, its size in graphics memory and the share on the CPU (`cpu_offload` in the cell and in its
  `results.jsonl` line, null when the model was fully in graphics memory). A variant whose context is smaller
  than asked is still not applicable.
- **OpenRouter free models.** `--relax-data-collection` lets OpenRouter use providers that keep the data of the
  requests. The data is synthetic, and each cell records this. The runner stops after a daily number of requests:
  45 for a `:free` model by default, or `--max-requests-per-day N`. When it stops, it prints "Daily request budget
  used. Run again with --resume --accept-data-notice tomorrow." (without the flag for a local model). A provider's
  "quota used" reply stops it in the same way.
- **Errors.** A cell whose answer was cut off, was still too long for the context, or was refused by the provider
  is a graded outcome and is saved. Any other error (no connection, a timeout, a busy service, a key, model or
  routing error) stops the run with exit code 1. That cell is not saved, and the message says to run again with
  `--resume`. The requests it sent still count towards the day's budget (`spent.json` in the results folder).
- **`--resume`** skips finished cells and asks again the cells that stopped with such an error. It also runs a
  cell recorded as not applicable once its setting fits. It refuses to continue a folder made with another model,
  provider, address, free-plan setting, Ollama contexts, CPU-offload setting, data-collection setting or project. Other levels,
  questions or repeats are fine.
- **Usage.** The requests are counted in the usage ledger under the provider's name with "(calibration)" added,
  apart from chat use.

## Grading

```
python tools/assistant_eval.py grade RESULTS_M1 RESULTS_M2 ... --dir GRADING   # repeat until every cell is graded
python tools/assistant_eval.py summary RESULTS ... --record
```

`grade` runs the automatic checks against the answer key:

- facts, with their units and SI prefixes;
- that every number in the answer comes from a tool result;
- forbidden statements;
- "the data does not answer" questions;
- the failed run;
- escalation;
- the tools called and the run's health.

Some items need a person, and the owner grades those:

- a number with no tool source (the nearest source numbers are shown with it);
- statement facts;
- whether the answer separates facts from assumptions.

Grading is blind. `grade` takes the results folders of every model and shuffles their cells together, so give it
all of them; one folder alone is refused unless `--single-model` says so. The order and its seed are kept in
`GRADING/grading.json`. A folder added later is shuffled into the cells not yet shown. A cell is shown as
`Cell #n`, never by its name. The model, the provider, the level, the call caps, the notices and the times are
hidden, and model and family names become "[assistant]". Texts and phrases that would reveal the level are
replaced with neutral placeholders, and so are the sizes of cut tool results. The only hint at the level is the
expected behaviour: either "a full answer is expected" or "escalating to a higher level is acceptable". Before the
first question, `grade` checks everything the sitting would show. If something still names the model, the level, a
cap or the time of a run, it grades nothing and says which cell. A model's writing style can still give it away,
so the blinding is only partial.

A sitting is 37 cells (`--sitting N`). `grade` always continues where it stopped: grades are saved after each
cell, and `q` or Ctrl-C keeps the verdicts already given for the current cell. `--regrade #n` grades cell #n
again; the result is recorded as a new line, and the first pass is kept. An interrupted re-grade saves nothing.

Each cell gets one outcome:

- **answered**: every critical item passes, and at least 75 % of the other items pass;
- **escalated**: a clean escalation whose numbers all come from tools;
- **failed**: anything else.

The count of invented numbers is kept with each cell. `summary --record` appends the graded cells to
`tools/assistant_eval_data/results.jsonl` and refuses grades made with a placeholder key.

## The evidence for the owner

`python tools/assistant_eval.py calibrate` reads `results.jsonl` and prints a table for each model. The rows are
the level settings, and the columns are the classes. Each entry gives two rates, each with n and a Wilson 95 %
interval:

- **answered_rate**: answered cells as a share of the cells that ran;
- **safe_rate**: answered or escalated cells with no invented number.

It also prints, for each setting:

- the calls, rounds and output tokens of the answered cells (median and 90th percentile);
- cap hits, length stops, fit compactions and truncations;
- the ratio of the reported to the estimated prompt tokens.

Free-plan cells are shown in a separate block and are not used for the caps.

Then it lists the evidence:

- whether the models at each level answer that level's classes, and the models below escalate them;
- a proposed number of calls per question for each level: the 90th percentile of the answered cells plus one,
  never below 3;
- each model's evidence level next to its automatic level;
- notes on the token estimates.

The starting bars are 60 % answered at basic, 75 % at standard and advanced, and no invented number. In the first
round each class has only a few cells, so these tables are evidence for the owner's review, not decisions.

## Dual grading

A second grader, Claude Opus 5.5 (`claude-opus-5-5`), grades a subset of the cells. Its agreement with the owner
is then measured. The owner's grades stay authoritative.

```
python tools/assistant_eval.py dual sample RESULTS_M1 RESULTS_M2 RESULTS_M3 RESULTS_M4 --dir DUAL
python tools/assistant_eval.py dual grade-ai --dir DUAL --accept-data-notice
python tools/assistant_eval.py dual report --dir DUAL --record
```

- **`sample`** draws 60 cells with a recorded seed. It takes one cell from each model × setting × class, then 24
  more weighted towards cells with numbers to review and towards interpret and diagnose cells. Then it adds cells
  that Opus wrote until there are at least 15 of them. Only cells whose outcome depends on a person's judgement
  are drawn.
- **`grade-ai`** sends one request per cell. There are no tools and no sampling settings, and the answer can be at
  most 16 000 tokens. Opus sees the same blinded material as the owner, framed as data. An invalid reply gets one
  retry; a second invalid reply is recorded as "unparsed" and counts as a disagreement. Its verdicts stay sealed in
  `grader_verdicts.sealed.json`.
- **`report`** runs only once the owner has made a first-pass grade of every sampled cell. It reports, for review
  numbers, statements, facts-vs-assumptions and the cell outcome:
  - agreement and Cohen's kappa, with a bootstrap interval resampled by question;
  - the lenient and strict disagreements;
  - Opus's own cells apart from the rest, as raw counts;
  - every disagreement;
  - the conditions of the proposed rule for reducing manual grading.

  `--record` appends Opus's lines with `"authoritative": false`.

## The answer key

The oracle writes the key from `tests/fixtures/tests.json`, `tests/assistant/grounding_projects.py`, the user
docs, `flextool/process_outputs/_output_meta.py` and the model's equation headers. It never runs FlexTool, the
tools or the runner. Every value has a `derivation`.

The format is in the docstring of `tools/assistant_eval_scoring.py`. A number taken from a result table carries
that table's unit, and `tests/assistant/grounding/test_grounding_set_consistency.py` checks this. A disputed value
is derived again by a third agent. It is never edited to match a model or a tool.

## Reading results.jsonl over time

Each line is one graded answer: which model answered which question, at which assistant level, on which date,
with which FlexTool version, and how it was graded. To see how a model does, take its lines with the latest date
and group them by level and by question class (lookup, interpret, diagnose).

The date matters. Providers change models without changing their names, so an answer recorded under the same
model name a few months later may come from a different model. Compare lines of one model only when their dates
are close, and treat an old line as a record of that day, not of the model as it is now.

The question set and the answer key have version numbers. Compare scores only between lines with the same
`set_version` and `key_version`. When either changes, run the models again before comparing.

A FlexTool upgrade can change answers too. The `flextool_version` and `commit` fields tell which version produced
an answer.

Lines with `"authoritative": false` are a second grader's verdicts, kept to measure agreement. Count scores only
from authoritative lines. A line with outcome `not_applicable` is a cell that was not run because the model's
context did not fit; leave it out of the rates. `knobs_relaxed` marks runs that allowed the provider to keep the
synthetic data.

A single line is one try. Read rates over several questions, not one answer.
