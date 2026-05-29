"""Modal dialog for the in-app "Update FlexTool" action."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class UpdateDialog(tk.Toplevel):
    """Ask the user to confirm a FlexTool self-update.

    Exposes these attributes after it closes:

      - ``proceed`` -- ``True`` if the user clicked Update.
      - ``include_toolbox`` -- ``True`` if Spine Toolbox should be installed
        alongside FlexTool (the ``[toolbox]`` extra).
      - ``check_on_startup`` -- the (possibly changed) "check for updates on
        startup" preference; the caller persists it whether or not the user
        proceeded with the update.

    Show it by constructing it and calling ``parent.wait_window(self)``.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        install_description: str,
        is_git: bool,
        default_toolbox: bool,
        check_on_startup: bool,
    ) -> None:
        super().__init__(parent)
        self.title("Update FlexTool")
        self.transient(parent)
        self.resizable(False, False)

        self.proceed: bool = False
        self.include_toolbox: bool = default_toolbox
        self.check_on_startup: bool = check_on_startup
        self._toolbox_var = tk.BooleanVar(value=default_toolbox)
        self._check_startup_var = tk.BooleanVar(value=check_on_startup)

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text="Update FlexTool to the latest version.",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        method = (
            "git pull + editable reinstall"
            if is_git
            else "pip upgrade from PyPI"
        )
        ttk.Label(
            body,
            text=f"Current install: {install_description}\nUpdate method: {method}.",
            wraplength=440,
            justify="left",
        ).pack(anchor="w", pady=(8, 8))

        ttk.Checkbutton(
            body,
            text="Install Spine Toolbox",
            variable=self._toolbox_var,
        ).pack(anchor="w")
        ttk.Label(
            body,
            text=(
                "Spine Toolbox is a large optional dependency. It is required "
                "to open .sqlite input sources in the Spine DB Editor. Leave "
                "unchecked if you do not need it."
            ),
            wraplength=440,
            justify="left",
            foreground="#888888",
        ).pack(anchor="w", padx=(24, 0), pady=(0, 8))

        ttk.Label(
            body,
            text=(
                "FlexTool must be restarted after the update. Progress is shown "
                "in the Execution window."
            ),
            wraplength=440,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            body,
            text="Check for updates on startup",
            variable=self._check_startup_var,
        ).pack(anchor="w", pady=(0, 8))

        btns = ttk.Frame(body)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right")
        ttk.Button(btns, text="Update", command=self._on_update).pack(
            side="right", padx=(0, 8)
        )

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _e: self._on_cancel())

        self.grab_set()
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        dw, dh = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    def _on_update(self) -> None:
        self.proceed = True
        self.include_toolbox = bool(self._toolbox_var.get())
        self.check_on_startup = bool(self._check_startup_var.get())
        self._close()

    def _on_cancel(self) -> None:
        self.proceed = False
        self.check_on_startup = bool(self._check_startup_var.get())
        self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
