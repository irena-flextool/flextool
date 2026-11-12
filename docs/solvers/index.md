# Solver selection in FlexTool

FlexTool can dispatch each `solve` in a scenario tree to a different LP / MIP
solver. The default — and what every existing scenario continues to use
without any extra configuration — is **HiGHS**, the open-source solver that
ships with FlexTool. Commercial solvers (Gurobi, CPLEX, Xpress, COPT) are
opt-in: install the vendor's Python wrapper, obtain a licence, then set
`solver = "<name>"` on the affected `solve` entity in the Spine database.
Solver choice is per-solve, not global, so a scenario can mix (for example) a
large dispatch LP on Gurobi with a small validation solve on HiGHS.

## Supported solvers

| Solver  | License                                       |
|---------|-----------------------------------------------|
| HiGHS   | Free, open source                             |
| Gurobi  | Commercial / academic free / WLS              |
| CPLEX   | Commercial / academic free / subscription     |
| Xpress  | Commercial / community (5000 var/row cap)     |
| COPT    | Commercial / academic free                    |

Per-solver installation, licensing, and verification:

- [HiGHS](highs.md) — bundled, no setup
- [Gurobi](gurobi.md)
- [CPLEX](cplex.md)
- [Xpress](xpress.md)
- [COPT](copt.md)

PLEASE NOTE: The instructions can be out-of-date.
Let us know if this is the case by opening an issue.

## How to switch solvers

Solver selection lives on the `solve` entity in the Spine database, in the
`solve_advanced` parameter group. Each parameter is optional and falls back
to a documented default if omitted, so existing scenarios behave exactly as
they did before this feature was added.

The DB-level solver parameters are, with their per-solver defaults:

| Parameter            | Default     | One-line description                                         |
|----------------------|-------------|--------------------------------------------------------------|
| `solver`             | `"highs"`   | Solver name from `[highs, gurobi, cplex, xpress, copt]`      |
| `solver_arguments`   | empty Map   | Raw option name → value pairs forwarded verbatim to the solver |
| `solver_mip_gap`     | unset       | Relative MIP gap; normalised across solvers                  |
| `solver_precommand`  | unset       | Not used: FlexTool never runs it (kept so older databases still read); a value gives a warning, see [Solver precommand](#solver-precommand-is-not-run) |

A handful of options that used to be DB parameters are now controlled by
**CLI flags** on `flextool` (they apply to the whole run rather than per
solve):

| CLI flag               | Former DB param (removed in v56; a stored value was dropped) | Values / notes |
|------------------------|--------------------------|-------------------------------------------------|
| `--solver-time-limit`  | `solver_time_limit`      | Seconds per solve, on every solver (HiGHS `time_limit`, Gurobi `TimeLimit`, CPLEX `timelimit`, Xpress `MAXTIME`, COPT `TimeLimit`); unset means no limit. See [Time limit](#time-limit). |
| `--solver-mip-gap`     |                          | Relative MIP gap of every solve, on every solver (HiGHS `mip_rel_gap`, Gurobi `MIPGap`, CPLEX `mip tolerances mipgap`, Xpress `MIPRELSTOP`, COPT `RelGap`); wins over the solve's `solver_mip_gap` |
| `--solver-config-dir`  |                          | The folder of the solver option files; see [Where the option files come from](#where-the-option-files-come-from) |
| `--highs-threads`      | `solver_threads`         | HiGHS only: integer thread count (default 1); `>1` enables HiGHS parallel mode |
| `--solver-log-level`   | `solver_log_level`       | `silent` hides the log of every solver; `normal` (default) shows it; `verbose` is HiGHS only |

`--presolve`, `--highs-threads`, `--user-bound-scale` and
`--solver-log-level verbose` are HiGHS options. When one of them is given and a solve runs on a commercial
solver, the run warns once: set the solver's own parameter in the solve's
`solver_arguments` or in its option file instead. Commercial solvers take
their thread count from their option file (for example `Threads` in
`gurobi.opt`); `--threads` does not change it.

The environment variables `FLEXTOOL_SOLVER_TIME_LIMIT` and
`FLEXTOOL_SOLVER_MIP_GAP` are the twins of the two flags (the flag wins). The
older names `FLEXTOOL_HIGHS_TIME_LIMIT` and `FLEXTOOL_HIGHS_MIP_GAP` still
work, for HiGHS only. See [Environment variables](../env_vars.md).

Example — set Gurobi on a single solve, with a 60-second time limit on each
solve of the run:

```text
solve entity:  yearly_dispatch
  solve_advanced.solver = "gurobi"
```
```bash
flextool <input_db_url> --solver-time-limit 60
```

Set the DB parameter in Spine Toolbox by selecting the `solve` entity,
opening its parameter editor, finding the `solve_advanced` group, and
entering the values above. Unspecified parameters take their defaults.

### Solver precommand is not run

FlexTool 3 ran the solve's `solver_precommand` in front of the CPLEX call,
for example to reserve a floating licence for the solve. FlexTool 4 never
runs it: a command stored in a database would run on the computer of anyone
who opens that database and presses Run, so a database from someone else
could run any program. A database that still has a value loads and solves as
before, with a warning once per solve and run that the command is not run.

To hold a floating licence, use your solver's own licence server setup (for
example a Gurobi token server set in `gurobi.lic`, or the licence server in
the licence file of CPLEX, Xpress or COPT); the solver then checks a licence
out and back in for the solve by itself. A solver wrapper set on this
computer (never in a database) is planned for a later release.

## The parameters in detail

| Parameter            | Type   | Default     | Description                                                                                              |
|----------------------|--------|-------------|----------------------------------------------------------------------------------------------------------|
| `solver`             | str    | `"highs"`   | Solver name. Must be one of `highs`, `gurobi`, `cplex`, `xpress`, `copt`.                                |
| `solver_arguments`   | map    | empty       | Map of option name → value passed verbatim to the underlying solver. Use this for any solver-native option that is not one of the convenience knobs below. Raw options always override the convenience knobs on key collision. |
| `solver_mip_gap`     | float  | unset       | Relative MIP optimality gap (e.g. `0.01` for 1 %). Translated to each solver's native parameter (`MIPGap` / `mip tolerances mipgap` / `MIPRELSTOP` / `RelGap` / `mip_rel_gap`). `--solver-mip-gap` wins over it. |
| `solver_precommand`  | str    | unset       | Not used: FlexTool never runs this command (FlexTool 3 ran it in front of CPLEX). The parameter is kept only so that older databases still read; a value gives a warning once per solve. See [Solver precommand](#solver-precommand-is-not-run). |

The corresponding CLI flags (`--solver-time-limit`, `--solver-mip-gap`,
`--highs-threads`, `--solver-log-level`) are documented in the table above.

### Authoring `solver_arguments` in Spine

`solver_arguments` is a Spine **Map** of string keys to string-or-float values.
Keys are the solver's own parameter names (e.g. `Presolve` for Gurobi or
`presolve` for HiGHS), values are whatever that solver expects.

Example (Gurobi, in Spine Toolbox's map editor):

```text
solver_arguments:
  Presolve   = 2
  Method     = 2
  Heuristics = 0.1
```

These are passed through untouched, within the limits of
[What the database may set](#what-the-database-may-set). Parameter names match
the way the solver reads them: without regard to case (`mipgap` and `MIPGap`
are one parameter); Gurobi also ignores `_` (`MIP_Gap` is `MIPGap`) and Xpress
an `XPRS_` prefix; for CPLEX a dotted name such as `mip.tolerances.mipgap` is
the same as `mip tolerances mipgap`. Write CPLEX names in full: an abbreviation
is not matched to its full name when the layers are merged (the file and folder
parameters below excepted: they are recognised abbreviated too). What a solver
does with a misspelt name is the solver's own: Gurobi ignores it with a
message, and FlexTool repeats that message as a warning naming where the name
was set.

FlexTool's results are not affected by a solver's file parameters (they are in
`output_parquet/<scenario>/` and the outputs you chose). Gurobi's `ResultFile`
is FlexTool's own: FlexTool names the result file and reads the solution from
it, so a `ResultFile` set anywhere is ignored with a warning. A commercial
solver's log is saved as `<work folder>/solve_data/subprocess/<solve>.log`
(see below). HiGHS' log is part of the run's console output, which
`flextool-run` and the FlexTool window save in the run's log,
`output_parquet/<scenario>/_run.log`.

#### What the database may set

An input database can come from someone else, so FlexTool trusts
`solver_arguments` and `solver_mip_gap` less than the files and the command
line of your own computer. The solver option files (`solver_config/<solver>.opt`),
the flags and the environment variables keep full paths and server
parameters; from the database:

- **Ordinary parameters** (gaps, tolerances, methods, thread counts, switches)
  are taken as they are.
- **A parameter that names a file or folder** takes only a plain file name:
  ASCII letters, digits and `_ . + -`, starting with a letter, digit or `_`,
  with no `/`, `\`, `:` or `..`. The file is written in the solve's work
  folder, `<work folder>/solve_data/subprocess/`. A full path is refused.
  These are the parameters whose name contains *file*, *dir* or *path*, and
  also:

  | Solver | File and folder parameters |
  |--------|----------------------------|
  | HiGHS  | `log_file`, `solution_file`, `read_solution_file`, `read_basis_file`, `write_basis_file`, `write_model_file`, `write_presolved_model_file`, `write_iis_model_file`, `mip_improving_solution_file` |
  | Gurobi | `ResultFile`, `LogFile`, `SolFiles`, `NodefileDir`, `InputFile`, `ConcurrentSettings`, `MultiObjSettings`, `TuneBaseSettings`, `TuneIgnoreSettings`, `TuneParams`, `Record` |
  | CPLEX  | `logfile`, `workdir`, `output intsolfileprefix` (also abbreviated, as the Interactive Optimizer accepts) and names starting with `read` or `write` |
  | Xpress | `TUNERMETHODFILE`, `TUNEROUTPUTPATH`, `TUNERSESSIONNAME` |
  | COPT   | `LogFile` |

  Names are matched as the solver reads them (above), so `Log_File` for
  Gurobi or `XPRS_TUNEROUTPUTPATH` for Xpress is the same parameter. A number
  or a switch in such a parameter (Gurobi `NodefileStart 0.5`) passes.
  Every solver runs in the same work folder, so the file name may not be one
  that a solver reads by itself from there: `gurobi.env`, a name ending in
  `.lic`, `license.dat`, `license.key`, `client.ini`, `xpauth.xpr` or
  `nethasp.ini` (in any case).
- **Gurobi server, cloud and licence parameters** are refused:
  `ComputeServer`, `ServerPassword`, `ServerTimeout`, the `CS...` Cluster
  Manager parameters (`CSManager`, `CSAPIAccessID`, `CSAPISecret`, `CSGroup`,
  `CSPriority`, `CSRouter`, ...), the `Cloud...` Instant Cloud parameters, the
  `WLS...` Web License Service parameters, `TokenServer`, `TSPort`,
  `LicenseID`, `UserName`, `WorkerPool` and `WorkerPassword`.
- **Xpress remote-solving controls** are refused: `COMPUTE`, which sends the
  solve to a remote service, and the other `COMPUTE...` controls
  (`COMPUTEEXECSERVICE`, `COMPUTEJOBPRIORITY`, `COMPUTELOG`).

The error names the parameter, its value and the solve. Put a file path or a
server parameter in the solver's option file, `solver_config/<solver>.opt`
(or that file in a folder given with `--solver-config-dir`), which keeps them.
The checks against extra commands (no line breaks, spaces, quotes, `;`, `#`,
`=` or a backslash at the end) apply to every source. So does, for Gurobi,
CPLEX, Xpress and COPT, the rule that every parameter has a value: an empty
value, or an option-file line with a name and no value, is refused (for
CPLEX also a last word that is not a number where the parameter takes one,
such as `mip tolerances mipgap` without its value).

### Per-solver baseline `.opt` files

Every solver gets a baseline parameter file in a `solver_config/` folder
(which folder: see [Where the option files come from](#where-the-option-files-come-from)).
These are **user-local files, not committed to the repo**: each ships in the
wheel as a `solver_config/<solver>.opt.template`, and `flextool-update` and
the GUI seed the workspace's `solver_config/<solver>.opt` from its template
(and re-seed any that are missing).  The runtime files are gitignored, so you can edit them —
or delete one to reset it to the bundled default — with zero git-committed
change. The FlexTool window (at start) and `flextool-update` replace a
workspace `solver_config/<solver>.opt` that is an unedited copy of an earlier
FlexTool default with the current one; an edited file is never touched. The
option values are the same, so no solve changes.

| Solver  | Baseline file                | Native format                                       |
|---------|------------------------------|-----------------------------------------------------|
| HiGHS   | `solver_config/highs.opt`    | `key=value` per line, `#` comments                  |
| Gurobi  | `solver_config/gurobi.opt`   | `ParamName value` per line, `#` comments            |
| CPLEX   | `solver_config/cplex.opt`    | `name value` per line; `name` uses interactive `set` syntax (e.g. `mip tolerances mipgap`) |
| Xpress  | `solver_config/xpress.opt`   | `CONTROL value` per line                            |
| COPT    | `solver_config/copt.opt`     | `ParamName value` per line                          |

FlexTool reads results from the solver directly and needs no HiGHS solution
file, so `write_solution_to_file`, `solution_file` and `write_solution_style`
are ignored in `highs.opt` (older templates set them; projects created then
still have the lines). To have HiGHS write a solution file anyway, set these
options in the solve's `solver_arguments` (`solution_file` takes a plain file
name there; the file is written in `<work folder>/solve_data/subprocess/`).

FlexTool reads each solver's baseline once at the start of the run, overlays
the solve's `solver_mip_gap`, its `solver_arguments` Map and the run's flags
**parameter by parameter** (one value per parameter, the highest layer wins),
writes the merged result next to the MPS file (the record of what the solver
received) and feeds it to the solver via the per-solver mechanism:

| Solver  | How FlexTool feeds the merged file to the CLI                                                  |
|---------|------------------------------------------------------------------------------------------------|
| HiGHS   | `--options=<file>` argv flag on the `flextool.cli.cmd_solve_mps` subprocess                    |
| Gurobi  | `gurobi_cl <Name>=<value> ... ResultFile=<sol> <mps>`: one `Name=value` argument per line of the file |
| CPLEX   | Each line emitted as `set <name> <value>` on the interactive optimizer's stdin before `read`   |
| Xpress  | Each line emitted as `setControl <NAME> <value>` on the optimizer console's stdin before `readprob` |
| COPT    | Each line emitted as `set <ParamName> <value>` on `copt_cmd`'s stdin before `read`             |

**Override precedence** (highest wins), the same for every solver:

1. The run's flags: `--solver-time-limit` and `--solver-mip-gap` (or
   `FLEXTOOL_SOLVER_TIME_LIMIT` / `FLEXTOOL_SOLVER_MIP_GAP`; the GUI's time
   limit is sent as the flag). HiGHS also takes `--highs-threads`,
   `--presolve`, `--solver-log-level` and `--threads` here (a commercial
   solver's log is hidden by `--solver-log-level silent` too, but that is not
   one of its parameters).
2. Raw `solver_arguments` entries on the solve.
3. Translated convenience knobs (`solver_mip_gap`).
4. The baseline file `<solver>.opt`.

For HiGHS every solve receives this one layered set: warm and cold solves,
`--save-memory` and the subprocess solve of `warm=False` alike. FlexTool's own
determinism defaults sit below the baseline file.

Before each solve the run prints what the solver receives: `Solver args:` for
HiGHS (the options set above FlexTool's defaults), and for a commercial solver
`Solver args (<solver>):` with every parameter under its native name, for
example `Solver args (gurobi): Threads=1, MIPGap=0.001, TimeLimit=1800`.

A commercial solver's program runs in `<work folder>/solve_data/subprocess/`.
FlexTool writes there the model (`<solve>.mps`), the parameters passed
(`<solve>.opt`) and, for CPLEX, Xpress and COPT, the command script it pipes
to the program (`<solve>.script`). To replay a solve by hand, run the program
in that folder with the script as its input, for example
`cd solve_data/subprocess && cplex < y2020_dispatch.script`. The solver's
result file and its own log file (`gurobi.log`, `cplex.log`) are written there
too.

The solver's log is shown as the solver writes it, in the terminal, the
FlexTool window (greyed, like HiGHS' log) and the run log, after a line
`Running <solver> (<program>)`, and it is saved as `<solve>.log` in the same
folder. Like the other files there, `<solve>.log` holds the last roll of a
rolling solve. A failed solve's message points to the log instead of repeating
it. With `--solver-log-level silent` the log is not shown (nor the `Running`
line), but FlexTool still reads it to know how the solve ended, saves it, and a
failure message then carries its last 20 lines. If FlexTool stops the solver
(the time-limit watchdog, Ctrl-C, a failure), it also stops the processes the
solver started.

### Where the option files come from

A run takes all its `<solver>.opt` files from one `solver_config` folder, the
first of:

1. `--solver-config-dir PATH` (the GUI always sends it: the workspace's
   `solver_config`, or the folder of `FLEXTOOL_SOLVER_CONFIG_DIR` when that
   variable is set for the GUI);
2. the `FLEXTOOL_SOLVER_CONFIG_DIR` environment variable;
3. `solver_config/` next to the input database, then, for a database in a GUI
   project (`<workspace>/projects/<project>/input_sources/`), the workspace's
   `solver_config/`. The search goes no further up, so a `solver_config`
   folder somewhere above an unrelated database is never picked up;
4. for a Spine Toolbox run with `--project-folder-file`, the `solver_config/`
   of that Toolbox project;
5. `solver_config/` in the current folder;
6. none found: FlexTool's defaults (the shipped templates), with a warning
   that lists the folders tried.

Tiers 3 to 5 count a folder only when it holds at least one `<solver>.opt`.
They skip `solver_config/` in the current folder when every `.opt` in it is an
unedited FlexTool template (of this or an earlier version) and the current
folder is not a FlexTool folder: FlexTool 4.3 and older left such a copy in
the folder a run was started from. The run then takes the next tier and
warns, naming the skipped folder; to use your own option files, pass
`--solver-config-dir <the folder with your files>`, and delete the skipped
folder to stop the warning. The FlexTool folder itself (the workspace, which
holds `templates/project_folder.txt`) is never skipped: its `solver_config/`
is the folder the FlexTool window uses, and a run from there uses it without
a warning, edited or not.
In the chosen folder, a solver whose file is missing uses its shipped default.
A Spine Toolbox run (whose Data Store is the workspace's `input_data.sqlite`)
therefore uses the workspace's `solver_config`, even though Toolbox runs the
tool in a temporary folder.

The run header and the log say which folder was used and why, for example:

```text
Solver config:  /home/me/flextool/solver_config (next to the input database)
```

The run reads the files once at its start: every roll and solve of the run
uses them as they were then, and an edit during the run reaches only the next
run. The files the run used are copied to
`<work folder>/solve_data/solver_config/` as its record; the copy is written
afresh every run and never read back. Up to FlexTool 4.3 runs read a copy in
`<work folder>/solver_config/`; such a folder is no longer read as the work
folder's copy (the run warns and names it when it differs from the files the
run uses). When the work folder is also the current folder, the folder is
tier 5: it is used when you have edited a file in it (or it is the FlexTool
folder), and skipped, with a warning, while it holds only the templates 4.3
copied there. To run with the files of a particular folder, pass
`--solver-config-dir <folder>`.

### Time limit

`--solver-time-limit` (and the GUI's time limit) applies to each solve
separately: each roll of a rolling solve, and each master and region solve of
a Benders decomposition, gets the full limit. The solver enforces it. When a
solve stops at its time limit:

- a MIP with a solution found keeps that solution, with a warning naming the
  solve and, when the solver reports it, the gap reached; the run goes on;
- an LP, or a MIP without a solution, stops the run with an error;
- in a Benders decomposition, a master or region solve stopped by its limit
  stops the run with an error, MIP or not: the decomposition needs each of its
  solves to be optimal.

Another limit or an interrupt (a node, iteration, solution or memory limit, or
an aborted solve) is not a time limit: it stops the run, naming the solver's
status, even with a solution. So does a solution the solver does not certify
as optimal, such as CPLEX's "optimal with unscaled infeasibilities" or
"optimal face unbounded".

FlexTool reads how a solve ended from the solver's result file (HiGHS, Gurobi's
JSON result, CPLEX) or from its console log (COPT, Xpress; Gurobi when it
writes a text result file). With COPT's log turned off (`Logging 0` or
`LogToConsole 0`), or a text Gurobi result with `LogToConsole 0`, a finished
solve cannot be told from one stopped by a limit, so the run stops and says
so: keep the log on (the shipped option files do). Xpress: FlexTool knows
only its time limit and an unfinished solve, from the console and the print
solution; another Xpress limit with a solution is not told apart.

FlexTool also stops a solver that has not finished well after its limit
(after twice the limit plus 10 minutes), which only guards against a solver
that hangs: reading the model and writing the result are not counted in the
solver's own limit.

## Warm-start caveat

The cross-solver dispatch always builds a fresh solver instance per call.
HiGHS retains its existing warm-start fast path because FlexTool uses HiGHS
directly (not via the cross-solver `solve()` entry point) when no `solver`
parameter is set. **Picking any non-HiGHS solver disables the in-process
warm path for that solve**: each sub-solve is built and solved through the
cross-solver dispatch. Rolls of a rolling-horizon scenario are built anew on
every solver (HiGHS reuses a built model only for an identical sub-solve), so
the solver choice itself does not change how often the LP is built.

## Benders-decomposition caveat

`decomposition_method = benders_regional` is currently **HiGHS-only**.
Setting a non-HiGHS `solver` on a Benders-decomposed scenario raises a
`FlexToolUserError` at startup. The error names both the offending solve and
the two remedies (either set `solver = highs` for that solve, or remove the
Benders decomposition from the group).

This is an upstream constraint in polar-high: the Benders master is a
persistent `WarmProblem`, and warm-LP machinery is HiGHS-only by design.
Lifting the restriction is tracked separately; in the meantime FlexTool
fails loudly rather than silently running on HiGHS.

The layered options above also reach the Benders master and every region
subproblem. Benders' own settings stay on top of them: the region
subproblems always run crossover (or the interior-point settings of the
experimental interior-dual mode), and the per-solve HiGHS log is silenced.
While the regions are solved in parallel (more than one worker), `threads`
and `parallel` (`--highs-threads`) apply to the master only; the regions stay
single-threaded so that the result does not depend on the worker count. HiGHS
has one thread scheduler per process, so the master re-initialises it before
each of its solves (polar-high 3.10.0 or later; with an older polar-high the
thread count is not used while the regions run in parallel). A
time limit (`--solver-time-limit`, or `time_limit` in `solver_arguments`)
applies to each master and region solve separately, not to the whole
decomposition. A master or region solve that stops at its limit stops the run
(see "Time limit" above).

## Verifying which solvers are installed

To see which solver Python wrappers polar-high has detected on this system,
run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Example output on a developer machine with all five wrappers installed:

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

A solver appears in `available_solvers` if its **Python wrapper** is
importable. That is not the same as "the solver is usable" — license checks
happen later, when FlexTool actually dispatches a solve. The most common
failure mode is `available_solvers` lists (say) `gurobi` but the first solve
raises a `FlexToolUserError` about a missing licence file; see the relevant
per-solver page for the licence setup.

## Common pitfalls

- **Setting `solver` without installing the wrapper.** Run the verification
  command above before authoring `solver = "gurobi"` (or similar) on a
  solve. If the solver name is missing from `available_solvers`, the
  per-solver page has the `pip install` line you need.
- **Setting `solver` without a valid licence.** The wrapper installs without
  one; the licence check only fires at solve time. FlexTool's error will
  point you at the licensing section of the solver page.
- **Mixing solvers without checking.** A scenario tree can route each solve
  to a different solver, but every solve needs its own `solver` parameter
  if you want anything other than HiGHS. Unset solves stay on HiGHS.
- **Commercial solvers on very large models.** A non-HiGHS solver is always
  given the model as an MPS file and run in its own process. The trade-off is bounded peak RSS (the
  `Problem.write_mps` writer caps ~2-3 GB on the largest LPs) at the cost
  of file I/O plus a fresh solver process per sub-solve.
- **Setting the same parameter in several places.** One value reaches the
  solver, from the highest layer of the precedence list above. If you pass
  `--solver-time-limit 60` and also set `solver_arguments = {TimeLimit: 30}`
  on a Gurobi solve, Gurobi sees `TimeLimit = 60`; without the flag it sees
  `30`. A `solver_arguments` entry wins over `solver_mip_gap` and over the
  option file.
- **Xpress and integer variables.** FlexTool's Xpress driver solves linear
  problems only; a solve with integer variables (integer investments, unit
  online variables) stops with a message. Use another solver for such solves.
