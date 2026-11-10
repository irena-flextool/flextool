# Evaluating the AI assistant's levels

The assistant works at three levels: basic, standard and advanced. Each level has its own instructions, tools and
token budgets. FlexTool places a model at a level from its name, its size and its context window. The
**calibration** checks those placements and budgets on questions whose answers are known. It is run by hand, at
release time or when the levels change. It never runs in CI, and pytest never collects it.

Everything lives under `tools/`, which is not packaged:

| file | what it holds |
|---|---|
| `tools/assistant_eval.py` | the command line: `build`, `run`, `grade-ai`, `review`, `summary`, `calibrate` |
| `tools/assistant_eval_scoring.py` | the question set and answer key formats, the automatic checks, the graded items and the grader material |
| `tools/assistant_eval_analysis.py` | the calibration tables and the agreement figures |
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
- **What the provider said.** After FlexTool's own message the console shows the provider's own words: its HTTP
  status, its error code and its message, for example
  `Google Gemini said: 429 RESOURCE_EXHAUSTED — You exceeded your current quota. Quota exceeded for metric: …`,
  then each different text of the cell's earlier failed attempts once (`earlier attempts (2): 503 UNAVAILABLE —
  …`). The stop's entry in `spent.json` keeps it: `provider` (the stopping answer: `code`, FlexTool's error code
  such as `rpm` or `busy`; `status`; `provider_code`; `message`) and `attempts` (every failed attempt of the cell
  in the same form, the stopping one last), with the run's `pace` and `stream`. A saved turn keeps its failed
  attempts too (`failed_attempts`; the turn's `error` has `provider`). The message is redacted (no key, no local
  path) and at most 300 characters. An error that came with no provider answer (no connection, a timeout, a
  stream that ended early) has `null` status, code and message.
- **`--pace SECONDS`** keeps at least that many seconds between the starts of two requests of the run, on top of
  FlexTool's own pacing (the spacing is the largest of the pace, the interval learned from rate limits, the
  preset's seed and the provider's request-limit headers). The pace is not learned or kept for the chat. A
  provider that limits the tokens per minute refuses requests that are each small enough: every request of a cell
  carries about 5 000 tokens of instructions and tool descriptions, so a few back to back exceed the minute.
  For Gemma via Google (preset `gemini`) use `--pace 20`.
- **`--no-stream`** asks for whole answers instead of streamed ones (the OpenAI-compatible providers; refused for
  Anthropic). The answer is read as the streamed one is: the same text (reasoning blocks such as Gemma's
  `<thought>` removed), tool calls with their provider data (Gemini's thought signatures), usage and error
  classification. Gemma via Google answered whole requests more reliably than streamed ones in manual tests.
- **`--pace` and `--no-stream` are part of the folder's record, not of its identity**: `run.json` lists every
  run's `pace` and `stream` (`runs`), each cell records the ones it was asked with (in `model`), and a `--resume`
  may change them.
- **`--resume`** skips finished cells and asks again the cells that stopped with such an error. It also runs a
  cell recorded as not applicable once its setting fits. A saved cell that is asked again stays in the folder
  until its new record replaces it, so a stop on the way never leaves the cell missing. `run.json` records the
  questions, levels and repeats every run of the folder planned (its `matrix`), before any cell is asked. It refuses to continue a folder made with another model,
  provider, address, free-plan setting, Ollama contexts, CPU-offload setting, data-collection setting or project. Other levels,
  questions or repeats are fine.
- **Skipping a cell that never completes.** Each request of a cell can fail, and a failure discards the whole
  cell, so a cell of many requests (the eight turns of S01) on a flaky endpoint may never finish. The owner can
  leave it out:

  ```
  python tools/assistant_eval.py run --resume --project DIR/"Grounding project ä" --provider PRESET \
      --model MODEL --out RESULTS --skip S01__small__r1 --skip-reason "the endpoint breaks off"
  ```

  with the other options of the folder's first run (its identity is checked as for `--resume`). `--skip` can be
  given more than once. Nothing is asked and no notice or key is needed: `run.json` records each cell as "not
  applicable (skipped by owner)" (its `skipped` entry) with the reason, the time and the attempts and requests
  `spent.json` and a saved record show (null when nothing tells). The skip is part of the folder's record, not of
  its identity, so a later `--resume` is not refused; it leaves the cell out. A cell id the folder cannot have is
  refused, with the closest ids named; a finished or not-applicable cell cannot be skipped. `--unskip CELL` takes
  a skip back (the entry moves to `unskipped`) and the next `--resume` asks the cell again. `grade-ai`, `review`
  and `summary` accept a folder whose only gaps are skipped cells: a skipped cell is in no rate and no agreement
  figure. `summary` prints it apart ("skipped: 1", with the cell and the reason), `summary --record` appends it as
  outcome `skipped` (with `skip_reason` and `skip_attempts`) and `calibrate` lists it apart in the same way.
- **Usage.** The requests are counted in the usage ledger under the provider's name with "(calibration)" added,
  apart from chat use.

## Grading

Claude Opus 5.5 (`claude-opus-5-5`) grades every cell, blind. The owner then reviews part of the cells, always
giving their own verdict before seeing Opus's. The owner's review is final.

```
python tools/assistant_eval.py grade-ai _audit_reports/assistant_eval/{opus,gemini,gemma,gptoss} \
    --dir _audit_reports/assistant_eval/grading_v2 --accept-data-notice
python tools/assistant_eval.py review --dir _audit_reports/assistant_eval/grading_v2      # repeat until done
python tools/assistant_eval.py summary --dir _audit_reports/assistant_eval/grading_v2 --record
python tools/assistant_eval.py calibrate
```

All grading state lives in the grading folder (`--dir`): `grading.json` (the blind order and its seed, the results
folders, the key version), `cells/<n>.json` (Opus's grading and the owner's review of cell #n), `review.json` (the
review queue and its random sample) and `agreement.txt`. The results folders are only read. A grading folder made
by the earlier grading (no "missing" verdict, no "incomplete" outcome) is refused: grade into a new folder.

### Items and outcomes

The automatic checks run against the answer key: facts with their units and SI prefixes, that every number in the
answer comes from a tool result, forbidden claims, "the data does not answer" questions, the failed run and
escalation. A check can only settle that an item is **right**. Every other finding is an item a grader judges,
with the verdicts it allows:

- **right**: the answer gets this item right;
- **missing**: the answer does not give it, and says nothing false about it;
- **wrong**: the answer says something false about it, or gives another value.

A number fact the check did not find is judged with the numbers that carry the fact's unit in the answer. A
number with no tool source is right (a shown or plain calculation, or a general constant) or wrong (invented). A
match of a forbidden pattern is wrong, or right when the answer does not claim it (for example, negated).

Each cell gets one outcome:

- **answered**: nothing missing and nothing wrong;
- **incomplete**: nothing wrong and no invented number, but something missing (a critical item too: the line
  lists `critical_missing`);
- **escalated**: a clean hand-over to a higher level and nothing wrong (what it left out does not count);
- **failed**: something wrong, an invented number, or no final answer.

A missing unit next to a right value, or a true claim not marked as an assumption, does not make a cell
incomplete: it is counted as `form_missing`. A wrong unit or a false unmarked claim fails the cell. A cell whose
final answer is empty (for example, cut off while the model was still reasoning) is failed without asking Opus
(`no_answer` in its line). The tools called and the run's health are not items; the line keeps them as
diagnostics.

Each cell also gets an understandability rating, which never changes the outcome: 1 = clear (the user gets the
answer on the first reading), 2 = understandable with effort (the answer is there, but buried, out of order or
needs a second reading), 3 = hard to follow.

### What the graders see

Opus and the owner see the same material: the question, the expected behaviour ("a full answer is expected" or
"escalating to a higher level is acceptable"), the answer key, the items the checks found right, each tool call
with its arguments and its result (folded to 4 000 characters), and the final message of each turn. Reasoning
blocks (`<thought>`, `<think>`) are removed, and so are reasoning fields, continuation notes, notices, usage and
times. The answer the graders see is the final message; text a model streamed between tool rounds is not
recorded.

Grading is blind. The cells of every model are shuffled together, and a cell is shown as `Cell #n`. Model and
family names become "[assistant]", phrases that name the level become "[level]", the level texts' fixed
escalation sentences become "[escalation sentence]", and the sizes of cut tool results are hidden. Before
anything is sent or shown, a leak scan checks every request as it would be sent; if something still names the
model, the level, a cap or the time of a run, nothing is sent, and every cell it found is named. The blinding is
partial: style, tool names (the basic level has fewer tools) and answer length can still hint at the model.

Answers and tool texts are data. Every text a model wrote or a tool returned is inside a fence
`<<<DATA tag>>>` … `<<<END tag>>>` whose tag the model could not know; material that holds its own tag is
refused. Opus's reply must be exactly the asked JSON (every item once, each verdict among the item's allowed
ones, confidences, reasons of at most 200 characters). Two slips are accepted and recorded in the grade's
`relaxed` list: a reason over 200 characters is cut to 200, and a verdict for an item the automatic checks found
right is dropped. An otherwise invalid reply gets one retry that names what was wrong, field by field (only
FlexTool's own item ids and field names); a second invalid reply is saved as unparsed and goes to the owner. An answer whose text, tool names or arguments seem to address a grader
goes to the owner too, whatever Opus said.

### grade-ai

`grade-ai` sends one request per cell (none for a cell without an answer), through FlexTool's retries and pacing.
A cell whose items the automatic checks all found right is sent too: its request asks only for the
understandability.
The key comes from the keyring or `ANTHROPIC_API_KEY`; `--accept-data-notice` accepts Anthropic's data notice for
this run only. The requests are counted in the usage ledger as "Anthropic (calibration)". Give it the results
folders of every model; one folder alone is refused unless `--single-model`. The folder set is fixed by the first
`grade-ai`. It refuses results folders that miss a planned cell (one of the `run.json` matrix; for a folder from
before the matrix, any question, level or repeat another graded folder has) or have cells that are not finished
(a cell the owner skipped with `run --skip` is no gap). The refusal names the cells and each folder's `run --resume` command. A cell that came back later would change
the grading's cells, so `grade-ai`, `review` and `summary` refuse until every folder is complete.

Run the same command again to continue: a cell that was graded from the same material is skipped. If a cell's
answer, the answer key or the rubric has changed since, `grade-ai` stops; `--changed` grades those cells again and
keeps their earlier grading in the cell's history (their review starts over). An error that is not an answer
stops the run with exit code 1 and saves nothing for that cell; an error during the retry keeps the first reply
(already paid), and the next run sends only the retry. Grading requests are sent without prompt caching (each
prompt is unique). `grade-ai`, `review` and `summary --record` lock the grading folder: two of them cannot work on
it at once.

### review

`review` starts once Opus has graded every cell. The queue is drawn once:

- every cell with a low confidence, an unparsed reply or an injection hint;
- every cell with a number that has no tool source, whatever Opus said;
- every cell written by `claude-opus-5-5` (for self-preference);
- a random sample of the rest: 15 % and at least 35 cells (all of them when fewer), drawn with the grading's seed;
- a cell `--changed` graded again that is outside the sampled pool and no longer has another reason (`changed`).

For each queued cell, the owner sees the material and gives a verdict on each judged item (`r` / `m` / `w`, only
the allowed letters) and the understandability (`1` / `2` / `3`). Only then are Opus's verdicts, confidences and
reasons shown, with each disagreement marked. For each disagreement, Enter keeps the owner's verdict and a letter
changes it. Both the first and the final verdicts are kept, and agreement uses the first. The reasons the cell
was queued are never shown.

A sitting is 37 cells (`--sitting N`; 0 for no limit). `q` or Ctrl-C keeps the verdicts already given, and the
next `review` asks the rest. `--regrade #n` reviews a cell again; the latest review wins. `--all` queues every
cell (full manual grading).

### summary

`summary --dir` prints, per model, level setting and class: answered, incomplete, escalated, failed (and how many
had no answer), the cells still waiting, the invented numbers and the understandability ratings. It also writes
`agreement.txt`: Opus against the owner's first verdicts, per item (3 categories), per outcome (4) and for
understandability (also linear-weighted kappa), each with n, agreement, Cohen's kappa with a bootstrap interval by
question, the confusion table and the lenient (Opus kinder) and strict counts. It reports:

- **the random sample**: the estimate for the cells Opus alone graded. When the Wilson 95 % upper bound of the
  outcome disagreement is above 10 %, it says "Review every cell: review --all." With 35 cells this passes only
  with no disagreement (0 of 35: 9.9 %);
- **all reviewed cells**;
- **self-preference**: Opus's own cells that no other reason queued, against the random sample, as raw counts.

`summary --record` appends one line per cell to `tools/assistant_eval_data/results.jsonl` and refuses while
cells still wait for review. A cell's grade is the owner's when reviewed, else Opus's. A later review or a
`--changed` grading appends a new line, and the latest line of a cell is its grade.

A paid canary (a wrong answer that tells the grader to mark every item right, excluded from the results) would
test injection resistance on the real model; the tests can only check the fences with a fake backend.

## The evidence for the owner

`python tools/assistant_eval.py calibrate` reads `results.jsonl` and prints a table for each model. The rows are
the level settings, and the columns are the classes. Each entry gives three rates, each with n and a Wilson 95 %
interval:

- **answered_rate**: answered cells as a share of the cells that ran;
- **incomplete_rate**: incomplete cells: nothing false said, but something left out. They are never counted as
  safe, because unlike an escalation they do not say what they left out;
- **safe_rate**: answered or escalated cells with no invented number.

For each model it adds the cells incomplete on a critical item, the cells without an answer and how many cells
the owner graded (the rest were graded by Opus alone). `calibrate` reads only grading v2 lines (`line_format` 2)
and refuses a file with older lines.

It also prints, for each setting:

- the calls, rounds and output tokens of the answered cells (median and 90th percentile);
- cap hits, length stops, fit compactions and truncations;
- the ratio of the reported to the estimated prompt tokens;
- the understandability ratings and their median.

Free-plan cells are shown in a separate block and are not used for the caps.

Then it lists the evidence:

- whether the models at each level answer that level's classes, and the models below escalate them;
- a proposed number of calls per question for each level: the 90th percentile of the answered cells plus one,
  never below 3;
- each model's evidence level next to its automatic level;
- notes on the token estimates.

The starting bars are 60 % answered at basic, 75 % at standard and advanced, and no invented number. In the first
round each class has only a few cells, so these tables are evidence for the owner's review, not decisions.

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

Each line says who gave its grade (`grader`: `owner` when reviewed, `claude-opus-5-5` otherwise, `none` for a
cell without an answer or not run), why it was reviewed, and Opus's own outcome and understandability next to the
final ones. When a cell has several lines, the latest is its grade. A line with outcome `not_applicable` is a cell
that was not run because the model's context did not fit; leave it out of the rates. `knobs_relaxed` marks runs
that allowed the provider to keep the synthetic data.

A single line is one try. Read rates over several questions, not one answer.
