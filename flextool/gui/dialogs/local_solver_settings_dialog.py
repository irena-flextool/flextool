"""Solver options → On this computer: the solver settings of this computer, not of the project (item 240 B1-B3).

Everything here is written to ``solver_settings.json`` in FlexTool's user config folder
(:mod:`flextool._local_solver_settings`), never to the project's ``settings.yaml``: these settings must not travel
with a project or a workspace.

- **Solver wrapper** per commercial solver: the wrapper program (a full path, with a file picker) and its
  arguments, one per line (stored as a list, never split).  OK checks each with
  :func:`~flextool._local_solver_settings.check_wrapper` and shows its message.  A wrapper set by
  ``FLEXTOOL_SOLVER_WRAPPER_<SOLVER>`` is shown read-only, since the variable wins.
- **Allow unlisted solver parameter names from input databases** (B1's switch); read-only when
  ``FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS`` is set.
- **Trust this workspace's solver option files**: puts the workspace (the folder FlexTool was started in) on this
  computer's list of workspaces it set up, so the files of its ``solver_config/`` may name paths and servers (B3).
  The option files that differ from FlexTool's defaults are listed next to it, to be read first.
- **Trust this project's solver option files** (with a project open): the same for the project's own
  ``solver_config/`` (``projects/<P>/solver_config``), which a run reads file by file over the workspace's.  A
  project that came as a package or a copied folder is not trusted until the box is ticked; its state is written
  only when the box was changed.
"""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from flextool import _local_solver_settings as local

_LABELS = {"gurobi": "Gurobi", "cplex": "CPLEX", "xpress": "Xpress", "copt": "COPT"}


def _edited_option_files(folder: Path) -> list[str]:
    """The ``<solver>.opt`` files in *folder* that are not an unedited FlexTool default."""
    from flextool._solver_config_source import SOLVER_NAMES, is_unedited_template

    edited = []
    for solver in SOLVER_NAMES:
        path = folder / f"{solver}.opt"
        try:
            if path.is_file() and not is_unedited_template(solver, path.read_bytes()):
                edited.append(path.name)
        except OSError:
            edited.append(path.name)
    return edited


class LocalSolverSettingsDialog(tk.Toplevel):
    """The modal "On this computer" dialog.  *workspace* is the folder whose trust it shows (default: the current
    folder, the FlexTool window's workspace); *project* the open GUI project, whose own ``solver_config`` trust
    it shows too (``None``: no project box)."""

    def __init__(self, parent: tk.Misc, *, workspace: Path | None = None, project: Path | None = None) -> None:
        data = local.read_settings_file()  # raises LocalSolverSettingsError (before any window); the opener shows it
        super().__init__(parent)
        self._parent = parent
        self.workspace = Path(workspace) if workspace is not None else Path.cwd()
        self.project = Path(project) if project is not None else None
        self.project_trust_var: tk.BooleanVar | None = None
        self._project_trusted_at_open: bool | None = None
        self.title("Solver settings on this computer")
        self.transient(parent)
        self.resizable(False, False)
        self.saved = False
        self.program_vars: dict[str, tk.StringVar] = {}
        self.args_texts: dict[str, tk.Text] = {}
        self.env_locked: dict[str, str] = {}

        wrappers = data.get(local.WRAPPERS_KEY) if isinstance(data.get(local.WRAPPERS_KEY), dict) else {}

        body = ttk.Frame(self, padding=12)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(
            body, justify="left",
            text=(f"These settings are saved on this computer in\n{local.settings_path()}\n"
                  "and are never part of a project, a workspace or a database."),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1

        ttk.Label(body, text="Solver wrapper", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Label(
            body, justify="left", foreground="#666666",
            text=("A program that starts in front of the solver, for example to hold a floating licence.\n"
                  "Give its full path; FlexTool adds the solver's path and arguments after its own arguments.\n"
                  "Keep passwords out of the arguments: they are shown in the run log."),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        row += 1
        for solver, label in _LABELS.items():
            env_name = local.wrapper_env_name(solver)
            value = wrappers.get(solver) if isinstance(wrappers.get(solver), list) else []
            env_value = os.environ.get(env_name)
            locked = env_value is not None
            if locked:
                self.env_locked[solver] = env_name
                try:
                    shown = local.solver_wrapper(solver)
                    value = list(shown.argv) if shown is not None else []
                except local.LocalSolverSettingsError:
                    value = [env_value]
            program = tk.StringVar(master=self, value=str(value[0]) if value else "")
            self.program_vars[solver] = program
            ttk.Label(body, text=f"{label} program:").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            entry = ttk.Entry(body, width=48, textvariable=program)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            browse = ttk.Button(body, text="Browse…", command=lambda v=program: self._browse(v))
            browse.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=2)
            row += 1
            ttk.Label(body, text="arguments, one per line:").grid(row=row, column=0, sticky="nw", padx=(0, 8))
            text = tk.Text(body, width=48, height=2, wrap="none")
            text.insert("1.0", "\n".join(str(item) for item in value[1:]))
            text.grid(row=row, column=1, sticky="ew", pady=(0, 2))
            self.args_texts[solver] = text
            row += 1
            if locked:
                entry.state(["disabled"])
                browse.state(["disabled"])
                text.configure(state="disabled")
                ttk.Label(body, text=f"set by {env_name}", foreground="#666666").grid(
                    row=row, column=1, sticky="w")
                row += 1

        ttk.Separator(body, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1
        allow = data.get("allow_unlisted_parameters") is True
        allow_env = os.environ.get(local.ALLOW_UNLISTED_ENV, "").strip()
        if allow_env:
            try:
                allow = local.allow_unlisted_parameters().value
            except local.LocalSolverSettingsError:
                pass
        self.allow_var = tk.BooleanVar(master=self, value=allow)
        allow_box = ttk.Checkbutton(
            body, variable=self.allow_var,
            text="Allow solver parameter names that are not in FlexTool's list from input databases")
        allow_box.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        if allow_env:
            allow_box.state(["disabled"])
            ttk.Label(body, text=f"set by {local.ALLOW_UNLISTED_ENV}", foreground="#666666").grid(
                row=row, column=0, columnspan=3, sticky="w")
            row += 1

        ttk.Separator(body, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1
        try:
            trusted = local.is_trusted_workspace(self.workspace)
        except local.LocalSolverSettingsError:
            trusted = False
        self.trust_var = tk.BooleanVar(master=self, value=trusted)
        ttk.Checkbutton(
            body, variable=self.trust_var, text="Trust this workspace's solver option files",
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        folder = self.workspace / "solver_config"
        edited = _edited_option_files(folder)
        ttk.Label(
            body, justify="left", foreground="#666666",
            text=(f"{folder}\n"
                  + (f"Edited: {', '.join(edited)}. Read them before you tick the box: trusted files may name\n"
                     "files anywhere and solver servers. Untrusted ones get the input database's rules."
                     if edited else "No file there differs from FlexTool's defaults.")),
        ).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        if self.project is not None:
            try:
                project_trusted: bool | None = local.is_trusted_workspace(self.project)
            except local.LocalSolverSettingsError:
                project_trusted = None  # unknown: the box writes nothing unless it is changed
            self._project_trusted_at_open = project_trusted
            self.project_trust_var = tk.BooleanVar(master=self, value=bool(project_trusted))
            ttk.Checkbutton(
                body, variable=self.project_trust_var,
                text=f"Trust this project's solver option files ({self.project.name})",
            ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
            row += 1
            own = self.project / "solver_config"
            edited = _edited_option_files(own)
            ttk.Label(
                body, justify="left", foreground="#666666",
                text=(f"{own}\n"
                      + (f"Edited: {', '.join(edited)}. Its runs use them over the workspace's files. Read them before\n"
                         "you tick the box, above all when the project came from someone else."
                         if edited else "No file there differs from FlexTool's defaults.")),
            ).grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1

        buttons = ttk.Frame(body)
        buttons.grid(row=row, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="OK", command=self.ok).pack(side="right")
        self.bind("<Escape>", lambda _e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self._install_grab()

    def _install_grab(self) -> None:
        """Take the input grab once the window is viewable (retried on the main loop)."""
        if not self.winfo_exists():
            return
        if self.winfo_viewable():
            try:
                self.grab_set()
                self.focus_set()
            except tk.TclError:
                pass
            return
        self.after(50, self._install_grab)

    def _browse(self, var: tk.StringVar) -> None:
        chosen = filedialog.askopenfilename(parent=self, title="Solver wrapper program")
        if chosen:
            var.set(chosen)

    def wrapper_argv(self, solver: str) -> list[str]:
        """The wrapper of *solver* as entered: the program, then one argument per non-empty line."""
        program = self.program_vars[solver].get().strip()
        if not program:
            return []
        lines = self.args_texts[solver].get("1.0", "end-1c").splitlines()
        return [program, *(line for line in lines if line.strip())]

    def ok(self) -> None:
        """Check every wrapper, then write the settings; on a problem show it and keep the dialog open."""
        entered = {s: self.wrapper_argv(s) for s in _LABELS if s not in self.env_locked}
        allow = None if os.environ.get(local.ALLOW_UNLISTED_ENV, "").strip() else self.allow_var.get()
        trust = self.trust_var.get()
        trust_project = None
        if self.project_trust_var is not None and self.project_trust_var.get() != self._project_trusted_at_open:
            trust_project = self.project_trust_var.get()
        try:
            for solver, argv in entered.items():
                if argv:
                    local.check_wrapper(argv, source=f"the {_LABELS[solver]} program field")
            # One write, under the file's lock: every setting or none.
            local.update_settings_file(
                lambda data: local.apply_settings(data, wrappers=entered, allow_unlisted=allow,
                                                  workspace=self.workspace, trust_workspace=trust,
                                                  project=self.project, trust_project=trust_project))
        except local.LocalSolverSettingsError as exc:
            messagebox.showerror("Solver settings on this computer", str(exc), parent=self)
            return
        self.saved = True
        self._close()

    def cancel(self) -> None:
        self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
        # Give the input grab back to the dialog this one was opened from (Solver options).
        try:
            if self._parent.winfo_exists() and isinstance(self._parent, tk.Toplevel):
                self._parent.grab_set()
        except tk.TclError:
            pass


def open_local_solver_settings(
    parent: tk.Misc, *, workspace: Path | None = None, project: Path | None = None,
) -> LocalSolverSettingsDialog | None:
    """Open the dialog; a settings file FlexTool cannot read is shown as an error instead."""
    try:
        return LocalSolverSettingsDialog(parent, workspace=workspace, project=project)
    except local.LocalSolverSettingsError as exc:
        messagebox.showerror("Solver settings on this computer", str(exc), parent=parent)
        return None


__all__ = ["LocalSolverSettingsDialog", "open_local_solver_settings"]
