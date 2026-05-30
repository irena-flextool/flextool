"""Modal dialog for the in-app "Update FlexTool" action."""
from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk


class UpdateDialog(tk.Toplevel):
    """Ask the user to confirm a FlexTool self-update.

    Exposes these attributes after it closes:

      - ``proceed`` -- ``True`` if the user clicked Update.
      - ``include_toolbox`` -- ``True`` if Spine Toolbox should be installed
        alongside FlexTool (the ``[toolbox]`` extra).
      - ``check_on_startup`` -- the (possibly changed) "check for updates on
        startup" preference; persisted whether or not the user updated.
      - ``update_available`` -- the latest known availability (possibly
        refreshed by the in-dialog Check button); ``None`` if never determined.

    The **Update** button is disabled when we know there is no update; the
    **Check** button re-queries on demand (useful when the startup check is
    off) and re-enables Update if the situation changed.

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
        update_available: bool | None,
        check_fn: Callable[[], bool],
        post_to_main: Callable[..., None],
    ) -> None:
        super().__init__(parent)
        self.title("Update FlexTool")
        self.transient(parent)
        self.resizable(False, False)

        self.proceed: bool = False
        self.include_toolbox: bool = default_toolbox
        self.check_on_startup: bool = check_on_startup
        self.update_available: bool | None = update_available
        self._initial_check_on_startup: bool = check_on_startup
        self._check_fn = check_fn
        self._post_to_main = post_to_main

        self._toolbox_var = tk.BooleanVar(value=default_toolbox)
        self._check_startup_var = tk.BooleanVar(value=check_on_startup)
        self._status_var = tk.StringVar()

        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text="Update FlexTool to the latest version.",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        method = (
            "git pull + editable reinstall" if is_git else "pip upgrade from PyPI"
        )
        ttk.Label(
            body,
            text=f"Current install: {install_description}\nUpdate method: {method}.",
            wraplength=440,
            justify="left",
        ).pack(anchor="w", pady=(8, 8))

        ttk.Label(
            body, textvariable=self._status_var, wraplength=440, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Checkbutton(
            body, text="Install Spine Toolbox", variable=self._toolbox_var,
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
                "FlexTool must be restarted after an update. Progress is shown "
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
        # Check on the left; right-aligned left→right: Update | OK | Cancel.
        # Update = save settings and upgrade; OK = save settings only; Cancel =
        # close without changing anything.
        self._check_btn = ttk.Button(btns, text="Check", command=self._on_check)
        self._check_btn.pack(side="left")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right")
        ttk.Button(btns, text="OK", command=self._on_ok).pack(
            side="right", padx=(0, 8)
        )
        self._update_btn = ttk.Button(btns, text="Update", command=self._on_update)
        self._update_btn.pack(side="right", padx=(0, 8))

        self._render_availability()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _e: self._on_cancel())

        self.grab_set()
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        dw, dh = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    # ── Availability state ──────────────────────────────────────────

    def _render_availability(self) -> None:
        """Reflect ``update_available`` in the status label and Update button."""
        if self.update_available is True:
            self._status_var.set("Status: a newer version is available.")
            self._set_update_enabled(True)
        elif self.update_available is False:
            self._status_var.set("Status: you are up to date.")
            self._set_update_enabled(False)
        else:
            self._status_var.set(
                "Status: not checked — click Check to look for updates."
            )
            self._set_update_enabled(True)

    def _set_update_enabled(self, enabled: bool) -> None:
        try:
            self._update_btn.configure(state="normal" if enabled else "disabled")
        except tk.TclError:
            pass

    def _on_check(self) -> None:
        """Query for an update off the UI thread; update state when it returns."""
        self._check_btn.configure(state="disabled")
        self._status_var.set("Status: checking…")

        def _worker() -> None:
            try:
                available = bool(self._check_fn())
            except Exception:
                available = False
            self._post_to_main(self._apply_check_result, available)

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_check_result(self, available: bool) -> None:
        """Apply a Check result on the main thread."""
        if not self.winfo_exists():
            return
        self.update_available = available
        self._render_availability()
        try:
            self._check_btn.configure(state="normal")
        except tk.TclError:
            pass

    # ── Button handlers ─────────────────────────────────────────────

    def _on_update(self) -> None:
        self.proceed = True
        self.include_toolbox = bool(self._toolbox_var.get())
        self.check_on_startup = bool(self._check_startup_var.get())
        self._close()

    def _on_ok(self) -> None:
        """Save the settings (e.g. the startup-check toggle) without updating."""
        self.proceed = False
        self.check_on_startup = bool(self._check_startup_var.get())
        self._close()

    def _on_cancel(self) -> None:
        """Close without applying any change (discard the toggle edits)."""
        self.proceed = False
        self.check_on_startup = self._initial_check_on_startup
        self._close()

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
