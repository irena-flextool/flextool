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
| `--solver-config-dir`  |                          | A folder of the solver option files; repeat for several, highest first (file by file); see [Where the option files come from](#where-the-option-files-come-from) |
| `--threads`            |                          | Threads of the run (the FlexTool window sends its Threads per run); sets every solver's thread count (HiGHS `threads` unless `--highs-threads` is given, Gurobi `Threads`, CPLEX `threads`, Xpress `THREADS`, COPT `Threads`). See below |
| `--highs-threads`      | `solver_threads`         | HiGHS only: integer thread count (default 1); `>1` enables HiGHS parallel mode |
| `--solver-log-level`   | `solver_log_level`       | `silent` hides the log of every solver; `normal` (default) shows it; `verbose` is HiGHS only |

`--presolve`, `--highs-threads`, `--user-bound-scale` and
`--solver-log-level verbose` are HiGHS options. When one of them is given and a solve runs on a commercial
solver, the run warns once: set the solver's own parameter in the solve's
`solver_arguments` or in its option file instead.

`--threads` (and `FLEXTOOL_THREADS`; the FlexTool window always sends its
Threads per run, default 1) sets the thread count of every solver: HiGHS
`threads`, Gurobi `Threads`, CPLEX `threads`, Xpress `THREADS`, COPT
`Threads`. It replaces the value of the option file and of
`solver_arguments`, even a value meaning all cores (`0`; Xpress and COPT
`0` or negative). FlexTool splits the computer's cores between the runs it executes in
parallel, so the run's thread budget wins. The run prints one line when it
replaces a different value, for example `gurobi uses Threads 1, the run's
thread budget (flag --threads), instead of Threads 0 (all cores) in
<folder>/gurobi.opt.` For more solver threads raise `--threads` (Threads per
run in the FlexTool window), or `FLEXTOOL_THREADS` when the budget came from
it. Without `--threads` and `FLEXTOOL_THREADS` (a terminal
or Spine Toolbox run that gives neither) the option file and
`solver_arguments` keep the thread count. Per-algorithm thread controls
(Xpress `BARTHREADS` / `MIPTHREADS` / `CONCURRENTTHREADS`, COPT `BarThreads` /
`SimplexThreads` / `CrossoverThreads` / `MipTasks`, Gurobi `ConcurrentJobs`)
are not changed: a value set there is used as given, even when it is larger
than the run's threads.

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
out and back in for the solve by itself. For anything else that ran in front
of the solver, set a solver wrapper on this computer (never in a database):
see [Running a solver through a wrapper](#running-a-solver-through-a-wrapper)
and [FlexTool 3 precommand users](#flextool-3-precommand-users).

### Running a solver through a wrapper

A solver wrapper is a program that FlexTool starts in front of Gurobi's,
CPLEX's, Xpress's or COPT's program, for example to hold a floating licence
for the solve, to queue the solve, or to lower the solver's priority
(`nice`). It is set per solver on this computer only, never in a database, a
project, a workspace or on the command line (a Spine Toolbox project from
someone else carries command lines), in one of two ways (the first wins):

- the environment variable `FLEXTOOL_SOLVER_WRAPPER_GUROBI`,
  `FLEXTOOL_SOLVER_WRAPPER_CPLEX`, `FLEXTOOL_SOLVER_WRAPPER_XPRESS` or
  `FLEXTOOL_SOLVER_WRAPPER_COPT`: a JSON list of the program's full path and
  its arguments, or just the program's full path (never split at spaces).
  Empty or `[]` means no wrapper, also over the settings file (for CI);

  ```bash
  export FLEXTOOL_SOLVER_WRAPPER_GUROBI='["/usr/bin/nice", "-n", "10"]'
  ```

- `wrappers` in `solver_settings.json` in FlexTool's user settings folder
  (the FlexTool window writes it: Solver options → On this computer):

  ```json
  {"format": 1, "wrappers": {"cplex": ["/usr/local/bin/lic_hold", "--queue", "opt"]}}
  ```

Runs from the FlexTool window, the terminal and Spine Toolbox all use the
same setting: a Toolbox run reads the environment variable (set it where
Toolbox is started) or the settings file of the user running Toolbox.

FlexTool then runs, without a shell, the wrapper's items followed by the
solver's own command: the solver program's full path and its arguments.

```text
/usr/local/bin/lic_hold --queue opt /opt/ibm/cplex/bin/x86-64_linux/cplex
```

The wrapper must start its remaining arguments as a command, pass its
standard input and output through (CPLEX, Xpress and COPT get their commands
on standard input), and return the solver's exit code. A shell script that
ends in `exec "$@"` does; on Windows, and anywhere, a Python script does:

```python
# lic_hold.py: run with the full path of python as the wrapper program
import subprocess, sys

# ... reserve the licence here ...
code = subprocess.run(sys.argv[1:]).returncode  # stdin, stdout and stderr pass through
# ... release it here ...
sys.exit(code)
```

```json
{"format": 1, "wrappers": {"cplex": ["C:\\Python312\\python.exe", "C:\\tools\\lic_hold.py"]}}
```

The rules, checked before the first solve of a run and again just before the
solver starts:

- the first item is the wrapper program's full path (FlexTool never looks for
  it on `PATH` or in the current folder), to an existing file, symbolic links
  followed; executable on Linux and macOS, a `.exe` on Windows;
- a script that Windows runs through a command interpreter (`.bat`, `.cmd`,
  `.ps1`, `.vbs`, `.js`, ...) and a command interpreter itself (`cmd.exe`,
  `powershell`, `pwsh`, `wscript`, `cscript`) are refused on every system:
  they would read the solver's path and arguments as a command line. On
  Windows also `wsl.exe` and `bash.exe` (and `sh`, `zsh`, `mshta`,
  `rundll32`, `regsvr32`, `forfiles`, `conhost`, `msiexec`), which join
  their arguments into a command line or run what they name. Use a program,
  or `python` with a script as above. On Linux and macOS a shell script
  works, with the script as the program or as the shell's first argument,
  when it starts the solver with `exec "$@"` (never `sh -c` with the
  arguments pasted into a command text);
- at most 32 items, each at most 4096 characters.

The run's log has the line
`Running CPLEX (/opt/.../cplex) through the solver wrapper /usr/local/bin/lic_hold with 2 arguments (from environment variable FLEXTOOL_SOLVER_WRAPPER_CPLEX)`
(with `--solver-log-level silent` it goes to the log only): the program and
the number of its arguments, never the arguments, because the AI assistant
can send the run log to its provider. The run record (`_run_status.json`,
on this computer) has `solver_wrappers` with the items in full. A failure
message says the solve was started through the wrapper, since the wrapper
may be the cause. **Keep passwords and tokens out of the wrapper's items
anyway:** the run record sits in the project's result folder, which may be
synced or shared; let the wrapper read secrets itself. A project package leaves the run log out and
keeps only the wrapper's program in the run record, with the number of its
arguments. The solver program is found in the absolute folders of `PATH`
(and, on Linux and macOS, the usual install folders), never in the current
folder. HiGHS runs inside FlexTool and has no wrapper; to lower its priority,
start FlexTool itself under `nice`. Watchdog, Ctrl-C and the stop of the
solver's processes work through the wrapper: FlexTool stops the wrapper and
everything it started.

### FlexTool 3 precommand users

FlexTool 3 ran a solve's `solver_precommand` in front of CPLEX. FlexTool 4
does not run it (see [Solver precommand is not run](#solver-precommand-is-not-run));
set the same program once per computer as the CPLEX wrapper instead:

1. Find the program's full path (FlexTool 3 found it on `PATH`): `which
   lic_hold` on Linux and macOS, `where lic_hold` on Windows.
2. Set it: in the FlexTool window, Solver options → On this computer, CPLEX
   program; or `FLEXTOOL_SOLVER_WRAPPER_CPLEX=/full/path/lic_hold`. If the
   precommand had arguments, give them one per line in the window, or as a
   JSON list in the variable: `["/full/path/lic_hold", "--queue", "opt"]`.
3. Check what the program does with its arguments. FlexTool 3 gave it
   `cplex -c "read ..." ...`; FlexTool 4 gives it the full path of the CPLEX
   program alone and sends CPLEX's commands on standard input. A program that
   only runs `"$@"` (or `subprocess.run(sys.argv[1:])`) works unchanged; one
   that read the CPLEX arguments must be changed. A `.bat` or `.cmd` must
   become a program, for example the Python script above.
4. Run once and look for the `Running CPLEX (...) through the solver wrapper`
   line. The database's `solver_precommand` can stay (it is never run, and
   gives a warning per solve); clear it to stop the warning. The same works
   for Gurobi, Xpress and COPT with their own variables.

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
the same as `mip tolerances mipgap`. Write CPLEX names in full: from the
database an abbreviation is refused (the message gives the full name), and in
an option file it is not matched to its full name when the layers are merged.
A name that is not one of the solver's parameters is refused from the database
(see below); in your own option files it gives a warning and is passed on, and
what the solver does with it is the solver's own: Gurobi ignores it with a
message, which FlexTool repeats as a warning naming where the name was set.

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

- **Only the solver's own parameter names** are taken. FlexTool has a list of
  every parameter of each solver, made from the solver's own package:

  | Solver | List made from |
  |--------|----------------|
  | HiGHS  | HiGHS 1.15.1 (highspy) |
  | Gurobi | Gurobi 13.0.2 (gurobipy), with `gurobi_cl`'s own parameters |
  | CPLEX  | CPLEX 22.1.2 (the cplex package), with the Interactive Optimizer's `logfile`, `defaults` and `distmip config` from IBM's documentation |
  | Xpress | Xpress 46.01.02 (xpress 9.8.1) |
  | COPT   | COPT 8.0.4 (coptpy), with the console's `LogFile` |

  A name that is not in the list is refused, and the message suggests the
  closest listed name: a misspelling (`MIPGapp`), a name in the wrong case
  for HiGHS (`Presolve`; HiGHS takes `presolve` only), a CPLEX abbreviation
  (`mip tol mipgap`: the message says to write `mip tolerances mipgap`), or a
  parameter of a newer solver version than the list. For HiGHS, a name that
  the installed HiGHS knows as a number or a switch is taken as well, so
  updating HiGHS does not stop your runs; a text option of a newer HiGHS is
  not. Rare and advanced parameters of the listed versions are in the list.
  Names are matched as the solver reads them (above).
- **Ordinary parameters** (gaps, tolerances, methods, thread counts, switches)
  are taken as they are. A text parameter that names no file (CPLEX
  `read fileencoding` and `cpumask`, Xpress `MPSRHSNAME` and the like) takes
  no `/`, `\` or `:` from the database; none of their values needs them.
- **A parameter that names a file or folder** takes only a plain file name:
  ASCII letters, digits and `_ . + -`, starting with a letter, digit or `_`,
  with no `/`, `\`, `:` or `..`. The file is written in the solve's work
  folder, `<work folder>/solve_data/subprocess/`. A full path is refused.
  These are the parameters whose value is a file or folder name:

  | Solver | File and folder parameters |
  |--------|----------------------------|
  | HiGHS  | `log_file`, `solution_file`, `read_solution_file`, `read_basis_file`, `write_basis_file`, `write_model_file`, `write_presolved_model_file`, `write_iis_model_file`, `mip_improving_solution_file` |
  | Gurobi | `ResultFile`, `LogFile`, `SolFiles`, `NodefileDir`, `InputFile`, `ConcurrentSettings`, `MultiObjSettings`, `TuneBaseSettings`, `TuneIgnoreSettings`, `TuneParams` |
  | CPLEX  | `logfile`, `workdir`, `output intsolfileprefix` |
  | Xpress | `TUNERMETHODFILE`, `TUNEROUTPUTPATH`, `TUNERSESSIONNAME` |
  | COPT   | `LogFile` |

  A number or a switch in a parameter whose name has *file*, *dir* or *path*
  in it (Gurobi `NodefileStart 0.5`), or that writes files of its own (Gurobi
  and CPLEX `Record`), passes. Every solver runs in the same work folder, so
  the file name may not be one that a solver reads by itself from there:
  `gurobi.env`, a name ending in `.lic`, `license.dat`, `license.key`,
  `client.ini`, `xpauth.xpr` or `nethasp.ini` (in any case).
- **Server, cloud and licence parameters** are refused, whatever their type
  (an integer can switch remote solving on). Gurobi: `ComputeServer`,
  `ServerPassword`, `ServerTimeout`, the `CS...` Cluster Manager parameters
  (`CSManager`, `CSAPIAccessID`, `CSAPISecret`, `CSGroup`, `CSPriority`,
  `CSRouter`, ...), the `Cloud...` Instant Cloud parameters, the `WLS...` Web
  License Service parameters, `TokenServer`, `TSPort`, `LicenseID`,
  `Username`, `WorkerPool` and `WorkerPassword`. Xpress: `COMPUTE`, which
  sends the solve to a remote service, and the other `COMPUTE...` controls
  (`COMPUTEEXECSERVICE`, `COMPUTEJOBPRIORITY`, `COMPUTELOG`). CPLEX:
  `distmip config` (the machines of a distributed MIP solve).
- **A few others are refused** as well: Gurobi's `JobID` and `Dummy` (text
  parameters with no use from a database) and CPLEX's `defaults` (a command
  that resets every parameter, not a parameter).
- **FlexTool 3's HiGHS settings** in a solve that uses Gurobi, CPLEX, Xpress
  or COPT are left out with one warning per run: the database migration moved
  FlexTool 3's `highs_method`, `highs_parallel` and `highs_presolve` into
  `solver_arguments` as `solver`, `parallel` and `presolve` for every solve.
  A number for the solver's own parameter of that name (Gurobi `presolve 2`)
  is still passed.

The error names the parameter, its value and the solve. FlexTool checks the
`solver_arguments` of every solve the model runs (`model.solves` and the
solves they contain) before the first solve starts, so the run stops in
seconds, not after hours of earlier solves. A solve that the model does not
run is not checked: its options never reach a solver. Put a
file path, a server parameter or a parameter that is not in the list in the
solver's option file, `solver_config/<solver>.opt` in the project's or the
workspace's `solver_config` (or a folder given with `--solver-config-dir`),
which keeps them when the folder is trusted (see
[Which option folders are trusted](#which-option-folders-are-trusted)). An
option file that came in a project package from someone else gets these
rules in the project's runs until you trust the project's solver option
files, and is checked by them when it is imported: a file with a line they
refuse is kept as `<solver>.opt.review`, which no run reads. To use it as it
is, read it, rename it to `<solver>.opt` and tick "Trust this project's solver
option files" (Solver options → On this computer, with the project open), or
run `flextool-trust-workspace <project folder>` in a terminal.
Exporting a project applies the same rules to the option files it would
ship: a file with such a parameter is left out of the package, and the export
names its lines (a server password or a folder of your computer should not be
sent unseen). A CPLEX parameter name in `cplex.opt` is written without the
leading `set` (FlexTool adds it), and a name that starts with `set` is refused
from every source. A UTF-8 byte order mark at the start of an option file is
ignored.
The checks against extra commands (no line breaks, spaces, quotes, `;`, `#`,
`=` or a backslash at the end) apply to every source. So does the rule that
every parameter has a value: an empty value is refused for every solver
(HiGHS included), and for Gurobi, CPLEX, Xpress and COPT also an option-file
line with a name and no value (for CPLEX also a last word that is not a
number where the parameter takes one, such as `mip tolerances mipgap`
without its value, and a group of parameters given a value, such as
`mip tolerances 0.01`: CPLEX would ask which parameter and read the
script's next line as the answer). The solution-file lines of old `highs.opt` templates are
dropped, as before.

#### Allowing a parameter that is not in the list

A newer solver version can have parameters that FlexTool's list does not
have yet. Set such a parameter in `solver_config/<solver>.opt`, which takes
any name (with a warning). Or allow names that are not in the list from input
databases on your computer, in one of two ways (the first wins):

- the environment variable `FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS=1`
  (`0` refuses them again); or
- the file `solver_settings.json` in FlexTool's folder of your user settings
  (`~/.config/flextool/` on Linux, `%LOCALAPPDATA%\flextool\` on Windows,
  `~/Library/Application Support/flextool/` on macOS):

  ```json
  {"format": 1, "allow_unlisted_parameters": true}
  ```

The setting is on your computer only: no database, project, workspace or
command-line flag can switch it on, since those can come from someone else.
With it on, an unlisted name is passed with one warning per run that says
where the setting is. The rules for paths, servers and the files a solver
reads by itself still apply:

- an unlisted name takes only a number, a switch or a plain file name (as a
  file parameter above), never a path or a name such as `gurobi.env`;
- a name that looks like a server parameter (Gurobi `CS...`, `Cloud...`,
  `WLS...`, `Server...`, `Token...`, `Worker...`; Xpress `COMPUTE...`; CPLEX
  `distmip ...`) is still refused, as are the listed ones above;
- a CPLEX abbreviation of a listed name (`log` for `logfile`, `d c` for
  `distmip config`, `def` for `defaults`) is still refused: the switch is for
  a newer solver's parameters, not for abbreviations;
- a HiGHS text option that FlexTool's list does not have is still refused. The
run's record (`output_parquet/<scenario>/_run_status.json`, `solver_config`)
says whether the setting was on and where it came from.

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
writes the merged result next to the MPS file as `<solve>.opt` (the record of
what the solver received; nothing reads it back) and hands each parameter to
the solver via the per-solver mechanism:

| Solver  | How FlexTool hands the merged parameters to the CLI                                            |
|---------|------------------------------------------------------------------------------------------------|
| HiGHS   | `--options-json=<solve>.options.json` on the `flextool.cli.cmd_solve_mps` subprocess (`--save-memory`), each option set in order; `--options=<solve>.opt` reads the record for a replay by hand |
| Gurobi  | `gurobi_cl <Name>=<value> ... ResultFile=<sol> <mps>`: one `Name=value` argument per parameter |
| CPLEX   | Each parameter emitted as `set <name> <value>` on the interactive optimizer's stdin before `read` |
| Xpress  | Each parameter emitted as `setControl <NAME> <value>` on the optimizer console's stdin before `readprob` |
| COPT    | Each parameter emitted as `set <ParamName> <value>` on `copt_cmd`'s stdin before `read`        |

**Override precedence** (highest wins), the same for every solver:

1. The run's flags: `--solver-time-limit` and `--solver-mip-gap` (or
   `FLEXTOOL_SOLVER_TIME_LIMIT` / `FLEXTOOL_SOLVER_MIP_GAP`; the GUI's time
   limit is sent as the flag), and the run's `--threads` (or
   `FLEXTOOL_THREADS`; the GUI's Threads per run is sent as the flag) as the
   solver's thread count. HiGHS also takes `--highs-threads`, `--presolve` and
   `--solver-log-level` here (a commercial solver's log is hidden by
   `--solver-log-level silent` too, but that is not one of its parameters).
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

A run reads each solver's file (`highs.opt`, `gurobi.opt`, `cplex.opt`,
`xpress.opt`, `copt.opt`) **file by file** from an ordered list of
`solver_config` folders, the *ladder*: each solver's file comes from the first
folder of the ladder that has it, and a solver that no folder has uses
FlexTool's default (the shipped template). The ladder is the first of:

1. the folders you give or the FlexTool window gives, and only these:
   `--solver-config-dir PATH` (repeat the flag for several folders, highest
   first) or the `FLEXTOOL_SOLVER_CONFIG_DIR` environment variable (several
   folders separated by `:`, on Windows `;`, as in `PATH`), and the folders a
   launcher found, `--workspace-solver-config-dir PATH` (repeatable, above the
   others). Nothing from the database's folders or the current folder is
   added. The FlexTool window (and the calibration it starts) always sends the
   project's `solver_config/` (when the project has that folder), then the
   workspace's `solver_config/`, as `--workspace-solver-config-dir`; when
   `FLEXTOOL_SOLVER_CONFIG_DIR` is set for the window, its folders replace the
   workspace's folder (passed on as `--solver-config-dir`) and the project's
   stays on top;
2. the input database's folders:
   - `solver_config/` next to the input database, alone;
   - for a database in a GUI project
     (`<workspace>/projects/<project>/input_sources/`), the project's
     `solver_config/`, then the workspace's: the same files as a run of the
     project from the FlexTool window.

   The search goes no further up, so a `solver_config` folder somewhere above
   an unrelated database is never picked up;
3. for a Spine Toolbox run with `--project-folder-file`, the `solver_config/`
   of that Toolbox project;
4. `solver_config/` in the current folder;
5. none found: FlexTool's defaults (the shipped templates), with a warning
   that lists the folders tried.

So a project's runs take each solver's options from the project's
`solver_config/`, else the workspace's, else FlexTool's default. To give a
project its own Gurobi options, for example, put just a `gurobi.opt` in
`<workspace>/projects/<project>/solver_config/`: its runs still use the
workspace's `highs.opt`. Delete the project's file to use the workspace's
again.

A project's `solver_config/` never gives an unedited FlexTool template (of
this or an earlier version): that file holds the options FlexTool shipped then
rather than a choice made for the project, and
the workspace's file of that solver is used instead (the header names the
skipped file). Tiers 2 to 4 count only when one of their folders holds a
`<solver>.opt`. They skip `solver_config/` in the current folder when every
`.opt` in it is an unedited FlexTool template and the current folder is not a
FlexTool folder: FlexTool 4.3 and older left such a copy in the folder a run
was started from. The run then takes the next tier and warns, naming the
skipped folder; to use your own option files, pass
`--solver-config-dir <the folder with your files>`, and delete the skipped
folder to stop the warning. The FlexTool folder itself (the workspace, which
holds `templates/project_folder.txt`) is never skipped: its `solver_config/`
is the folder the FlexTool window uses, and a run from there uses it without
a warning, edited or not.
A Spine Toolbox run (whose Data Store is the workspace's `input_data.sqlite`)
therefore uses the workspace's `solver_config`, even though Toolbox runs the
tool in a temporary folder.

The run header and the log say which folder was used and why, for example:

```text
Solver config:  /home/me/flextool/solver_config (next to the input database)
```

With several folders they name the folder of each file, then the solvers that
use FlexTool's defaults:

```text
Solver config:  /home/me/flextool/projects/P/solver_config (project of the FlexTool window): gurobi.opt; /home/me/flextool/solver_config (workspace of the FlexTool window): highs.opt; FlexTool defaults: cplex.opt, xpress.opt, copt.opt
```

A folder of the ladder that does not exist gives a warning; its solvers use
the next folder or FlexTool's default.

#### Which option folders are trusted

The files of a folder you name (`--solver-config-dir`,
`FLEXTOOL_SOLVER_CONFIG_DIR`) and FlexTool's shipped defaults are trusted:
they may use full file paths, server and licence settings, and parameter
names that are not in FlexTool's list (with a warning). A folder that a run
finds by itself (the FlexTool window's `--workspace-solver-config-dir`
folders, and tiers 2 to 4) may have come from someone else, inside a
workspace, a project, a model folder or a Spine Toolbox project that was
shared or unzipped. Its files are trusted only when the folder it is in (the
workspace; for `projects/<project>/solver_config`, the project) is one this
computer set up; otherwise every line gets the input database's rules (see
"What the database may set"): a file option takes only a plain file name,
placed in the solve's work folder; no server or licence settings; only
parameter names in FlexTool's list. Trust is judged per folder, so file by
file: when a project's untrusted `gurobi.opt` is used over a trusted
workspace's, the project's file gets the database's rules, and the
workspace's `highs.opt` keeps its trust. A file that is byte for byte one of
FlexTool's shipped templates is trusted wherever it lies. A refused line stops
the run with a message that names the file, the line, and how to trust the
folder. The `Solver config:` line says `not a workspace this computer set up`
for such a folder, and the run record's `solver_config.layers[].trusted` and
`solver_config.files[].trusted` are `false`.

| Where the folder comes from | Trusted |
|---|---|
| `--solver-config-dir`, `FLEXTOOL_SOLVER_CONFIG_DIR` | always |
| `--workspace-solver-config-dir` (the FlexTool window), next to the database, the database's project or workspace, the Toolbox project, the current folder | when its workspace (or project) is on this computer's list |
| FlexTool's shipped defaults | always |

This computer's list is `trusted_workspaces` in `solver_settings.json` in
FlexTool's user settings folder (on Linux `~/.config/flextool/`, on Windows
`%LOCALAPPDATA%\flextool\`, on macOS `~/Library/Application Support/flextool/`),
outside every workspace, so nothing that comes with a workspace can add to it.
A workspace gets on the list:

- when `flextool-update`, or the FlexTool window at its start, creates its
  `solver_config/`, or finds nothing in it but FlexTool's own default files
  (unedited `.opt` files of this or an earlier version; a symbolic link does
  not count). A workspace received with edited option files is therefore
  never listed by being opened;
- when you tick "Trust this workspace's solver option files" in the FlexTool
  window (Solver options → On this computer), after reading the files.

You can also trust a folder from a terminal: `flextool-trust-workspace
<folder>` (the workspace, or a GUI project's folder) shows the option files of
its `solver_config/` that differ from FlexTool's defaults, line by line, and
asks before it lists the folder; `--remove` takes it off the list, `--list`
shows the list. It is a command you type and answer, not a flag of a run: a
Spine Toolbox project carries command lines for its tools, so a run flag could
come with a project from someone else, while anything that could start this
command unasked can already run any program as you.

The trust is for the folder as it was when it was listed, not for its path:

- a different folder put at the same path later (the trusted one deleted or
  moved away, and a copy from someone else unpacked or copied there) is not
  trusted; the message says the folder was replaced. FlexTool tells the
  folders apart by their identity: on Linux and macOS the device, the inode
  number and the time the folder was made; on Windows the volume serial
  number, the file ID and the creation time;
- an option file that changed since, or appeared since, is trusted when your
  own account wrote it and no other account can change it: on Linux and macOS
  you own it, no other account can write it, and its group can write it only
  when the group is your own private group (the one named after you, with no
  other members). Otherwise (another owner, writable by others, a link) it gets
  the database's rules, with a message naming it, until you trust the folder
  again (the box, or `flextool-trust-workspace`). On Windows FlexTool does not
  read files' owners and access lists: a file in a folder that is still the
  one you trusted counts as your own edit, as Windows gives a new file its
  writer as owner and, in your own folders, lets no other ordinary account
  write. A drive that gives files no ID (FAT, some network drives) gets no such
  pass: there any change asks to be trusted again;
- so editing your own files never asks again, but files you unpack or copy
  into a trusted folder yourself are your edits too, and so are changes that a
  sync tool (OneDrive, Dropbox, a network share) or `git pull` writes with
  your account. Do not trust a folder that other people can change through such
  a tool; name it with `--solver-config-dir` for the runs that should use it.

Entries in `trusted_workspaces` must be full folder paths: an entry such as `.`
or `~/flextool` makes the list unreadable (nothing is trusted, and the message
says why). Entries written by FlexTool 4.4 development versions (plain folder
names) are bound to the folder at that path, with its files as they are, the
first time FlexTool reads them. `FLEXTOOL_USER_HOME` or `XDG_CONFIG_HOME`
naming a relative folder, or one inside the current folder, is ignored with a
warning (the platform's folder is used): a workspace could otherwise carry its
own list.

**Spine Toolbox projects are code.** A Toolbox project runs the tool scripts
and command lines it carries, so it can do anything you can, solver options
included: open a Toolbox project, or a launcher script that comes with a
workspace, only from people you trust, as you would a program.

A GUI project (for its own `projects/<project>/solver_config/`) gets on the
list when the FlexTool window creates it, when the window opens it while its
`solver_config/` is missing or holds nothing but FlexTool's default files and
no received file waits for review (no `<solver>.opt.review`, no
`imported_solver_config/`), and when you tick "Trust this project's solver
option files" (Solver options → On this computer, with the project open). A
project imported from a package is never listed by being imported or opened
with its shipped files in place, and importing at the path of a project
deleted earlier first takes that path off the list; renaming a project in the
window keeps its trust, deleting it removes it. Copying a received file into
a trusted folder (the workspace's `solver_config/`, a listed project's, or a
folder you name with `--solver-config-dir`) makes it trusted there: read it
first.

For a workspace whose option files you edited before this check existed, tick
that box once (or pass the folder with `--solver-config-dir`). On CI or a
server, pass the folder with `--solver-config-dir` or set
`FLEXTOOL_SOLVER_CONFIG_DIR` in the job's environment: a folder named that way
is trusted on every run, and nothing needs to be listed. A fresh checkout
where `flextool-update` creates `solver_config/` is listed too (in the user
settings folder of the user who ran it).

The run reads the files once at its start: every roll and solve of the run
uses them as they were then, and an edit during the run reaches only the next
run. The files the run used are copied to
`<work folder>/solve_data/solver_config/` as its record; the copy is written
afresh every run and never read back. Up to FlexTool 4.3 runs read a copy in
`<work folder>/solver_config/`; such a folder is no longer read as the work
folder's copy (the run warns and names it when it differs from the files the
run uses). When the work folder is also the current folder, the folder is
tier 4: it is used when you have edited a file in it (or it is the FlexTool
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
