# Solver option trust (developer)

The code-level rules behind [Which option folders are trusted](index.md#which-option-folders-are-trusted) and
[What the database may set](index.md#what-the-database-may-set). Where the option files come from (the folder
ladder) and the option stack are in [architecture.md "Runtime options"](../dev/architecture.md#runtime-options),
item 6.

- **Which names are allowed.** The database layers (`solver_arguments`, `solver_mip_gap`) take only names in
  `engine/solve/solver_catalogue/<solver>.json` (`_solver_catalogue`, written by `tools/solver_catalogue.py` with a
  reviewed class per name: `plain` / `file` / `local_only` / `flextool`), checked for every solve before the first
  (`check_run_solver_options`) and again per solve (`check_database_solver_option`). HiGHS names the installed
  highspy knows as bool / int / double pass too. The per-user switch for unlisted names
  (`flextool/_local_solver_settings.py`: `FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS`, else
  `user_config_dir()/solver_settings.json`) is never read from a database, project, workspace or flag. Option
  files and flags warn on an unlisted name and pass it. Project import screens shipped files with the run's own
  rule for an untrusted file (`_solver_dispatch.untrusted_option_file_problems`: `validate_solver_option` +
  `check_database_solver_option` per line) and holds a refused one back as `<solver>.opt.review`; export's
  privacy screen is the engine-free `flextool/_solver_option_rules.py`.
- **Trust follows the source, per folder of the ladder (so per file).** A folder the run found by itself (ladder tiers
  2-4, and the GUI's `--workspace-solver-config-dir` folders) is trusted only when its parent (the workspace, or
  the project for `projects/<P>/solver_config`) is in `trusted_workspaces` of the settings file
  (`SolverConfigSource.layer_note`); otherwise `SolverOptionFiles.untrusted_note(solver)`, from the folder that
  gave that solver's file, is handed to `resolve_commercial_options` / `_resolve_effective_highs_options` and each
  file line goes through `check_database_solver_option(untrusted=...)`. `--solver-config-dir` /
  `FLEXTOOL_SOLVER_CONFIG_DIR` and Spine Toolbox projects (code) are trusted. A file that is byte for byte a
  shipped template is trusted wherever it lies; FlexTool's own file is told by content (`is_unedited_template`),
  never by the origin text.
- **Adding and removing trust.** Only `ensure_runtime_files` (when it creates `solver_config/` or finds only
  unedited templates, `holds_only_flextool_defaults`), the GUI's project create / open
  (`note_project_set_up_here`: only with nothing received waiting, `awaits_review`) and the GUI's explicit "trust
  this workspace / project" add to the list; project import, the first-open rename of `imported_solver_config/`
  and project deletion take the path off (`forget_trusted_folder`); nothing inside a workspace
  (`templates/project_folder.txt` included) is a trust anchor. An entry is bound to the folder trusted
  (`folder_identity`: device + inode + birth time, Windows volume serial + file ID + creation time, `st_ctime`
  before Python 3.12; ext4 reuses a deleted folder's inode at once, so the birth time is needed, and a stored one
  must be matched) and to its `.opt` hashes (`TrustedFolder`): a replaced folder is untrusted (`trust_status`),
  and a changed file is covered only when `only_you_can_change` (POSIX owner, not group/other-writable unless the
  user's private group; on Windows a plain file, as owner and ACL are not read) — `read_solver_option_files`
  keeps that per-file verdict in `SolverOptionFiles.file_notes`. Text entries of 4.4 development builds are bound
  on first read; relative entries are refused; a relative `FLEXTOOL_USER_HOME` / `XDG_CONFIG_HOME` is ignored
  (`user_dirs`; never depends on the start folder), and a settings file inside the folder judged
  (`_check_settings_outside`) makes its trust unreadable. The settings file is refused on read when not UTF-8 or
  (POSIX) another account owns or can write it (`_settings_file_problem`).
- **The terminal route** is `flextool-trust-workspace` (`cli/cmd_trust_workspace.py`): a typed, answered command,
  not a run flag (a Toolbox project carries tool command lines, and anything that can start the command unasked
  can already run code). It refuses a stdin that is not a terminal and has no `--yes`, escapes what it shows
  (`_safe_text.printable`), and trusts exactly the bytes it showed (`show_folder` →
  `apply_settings(workspace_shown=)`, `ShownFilesChanged`), as does the window's box with the files as the dialog
  found them.
- **Solver wrapper.** A commercial solver may run behind this computer's solver wrapper
  (`_local_solver_settings.solver_wrapper`: `FLEXTOOL_SOLVER_WRAPPER_GUROBI` and its CPLEX, XPRESS, COPT twins,
  else `wrappers` in the same file; `check_wrapper` holds the rules), checked for every solver of the run before
  the first solve (`check_run_solver_wrappers`) and again before each start; `_solve_commercial_subprocess`
  prepends it to the solver's argv (no shell; the solver path is absolute, `_find_solver_binary` skips relative
  and, on Windows, current-folder `PATH` entries), logs the `Running ... through the solver wrapper` line with
  the program and its argument count only (the AI assistant sends run logs out) and notes `solver_wrappers` in
  full in the run record (export keeps only the program and the count).
- The settings file (written 0600) is written only by the GUI's Solver options → On this computer,
  `flextool-trust-workspace` and `ensure_runtime_files`; no assistant tool, database value, project file or flag
  writes or sets it.
