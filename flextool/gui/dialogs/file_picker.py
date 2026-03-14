from __future__ import annotations

import logging
import os
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import ttk

logger = logging.getLogger(__name__)

# Prefixes for directory/file entries in the Treeview
_DIR_PREFIX = "\U0001f4c1 "   # folder emoji
_FILE_PREFIX = "\U0001f4c4 "  # page emoji
# Fallback plain-text prefixes if emoji cause issues
_DIR_PREFIX_PLAIN = "[DIR] "
_FILE_PREFIX_PLAIN = "      "


def _format_mtime(timestamp: float) -> str:
    """Format a modification timestamp as DD.MM.YY hh:mm."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%d.%m.%y %H:%M")


def _parse_extensions(pattern: str) -> set[str]:
    """Parse a filetype pattern string like ``'*.xlsx *.sqlite'`` into a set of lowercase extensions.

    Returns an empty set for wildcard patterns (``'*'``, ``'*.*'``).
    """
    extensions: set[str] = set()
    for part in pattern.split():
        part = part.strip()
        if part in ("*", "*.*"):
            return set()  # wildcard = show all
        if part.startswith("*."):
            extensions.add(part[1:].lower())  # e.g. ".xlsx"
    return extensions


class FilePickerDialog(tk.Toplevel):
    """Custom file picker with last-modified column and sorting.

    After the dialog closes, access ``result`` for the selected path(s).
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str = "Select File",
        initialdir: str | Path = ".",
        filetypes: list[tuple[str, str]] | None = None,
        multiple: bool = False,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """
        Args:
            parent: Parent window.
            title: Dialog title.
            initialdir: Starting directory.
            filetypes: List of (description, pattern) like
                ``[("Excel files", "*.xlsx"), ("All files", "*")]``.
            multiple: Allow selecting multiple files.
            width: Dialog width in pixels (``None`` = auto from parent).
            height: Dialog height in pixels (``None`` = auto).
        """
        super().__init__(parent)
        self.title(title)

        self._multiple = multiple
        self._result: Path | list[Path] | None = None

        # Resolve initial directory
        init = Path(initialdir).resolve()
        if not init.is_dir():
            init = Path.home()
        self._current_dir = init

        # File type filters
        if filetypes:
            self._filetypes = filetypes
        else:
            self._filetypes = [("All files", "*")]

        # Sorting state: (column_id, ascending)
        self._sort_col: str = "name"
        self._sort_asc: bool = True

        # Detect whether emoji render acceptably (best-effort)
        self._use_emoji = True
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            # If the font can measure the emoji without error, assume it works
            default_font.measure(_DIR_PREFIX)
        except Exception:
            self._use_emoji = False

        # ── Modal behaviour ──────────────────────────────────────
        self.transient(parent)
        self.grab_set()

        # ── Dialog size ──────────────────────────────────────────
        if width is None:
            try:
                width = parent.winfo_width()
            except Exception:
                width = 700
            if width < 400:
                width = 700
        if height is None:
            try:
                height = int(parent.winfo_screenheight() * 0.6)
            except Exception:
                height = 500
            if height < 300:
                height = 500

        self.geometry(f"{width}x{height}")
        self.resizable(True, True)
        self.minsize(400, 300)

        self._build_widgets()
        self._populate()

        # Close via window-manager "X"
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # ── Keyboard bindings ────────────────────────────────────
        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self._on_cancel())

        # Centre on parent
        self.update_idletasks()
        try:
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
        except Exception:
            px, py, pw, ph = 100, 100, 800, 600
        w = self.winfo_width()
        h = self.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

        # Block until closed
        parent.wait_window(self)

    # ── Public API ───────────────────────────────────────────────

    @property
    def result(self) -> Path | list[Path] | None:
        """Selected file path(s), or ``None`` if cancelled."""
        return self._result

    # ── Widget construction ──────────────────────────────────────

    def _build_widgets(self) -> None:
        pad = dict(padx=8, pady=4)

        # ── Location bar ─────────────────────────────────────────
        loc_frame = ttk.Frame(self)
        loc_frame.pack(fill="x", **pad)

        ttk.Label(loc_frame, text="Location:").pack(side="left")
        self._loc_var = tk.StringVar(value=str(self._current_dir))
        self._loc_entry = ttk.Entry(loc_frame, textvariable=self._loc_var)
        self._loc_entry.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self._loc_entry.bind("<Return>", self._on_location_enter)

        # ── Treeview ─────────────────────────────────────────────
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="both", expand=True, **pad)

        select_mode = "extended" if self._multiple else "browse"
        self._tree = ttk.Treeview(
            tree_frame,
            columns=("name", "modified"),
            show="headings",
            selectmode=select_mode,
        )

        # Column headings
        self._tree.heading(
            "name",
            text="Name \u25b2",
            command=lambda: self._on_sort("name"),
        )
        self._tree.heading(
            "modified",
            text="Last Modified",
            command=lambda: self._on_sort("modified"),
        )

        # Column widths
        self._tree.column("name", width=350, minwidth=150, stretch=True)
        self._tree.column("modified", width=130, minwidth=100, stretch=False)

        # Scrollbar
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Events
        self._tree.bind("<Double-1>", self._on_double_click)
        self._tree.bind("<<TreeviewSelect>>", self._on_selection_changed)

        # ── File type filter ─────────────────────────────────────
        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill="x", **pad)

        ttk.Label(filter_frame, text="File type:").pack(side="left")

        # Build display strings for the combobox
        self._filter_display: list[str] = []
        for desc, pattern in self._filetypes:
            if pattern in ("*", "*.*"):
                self._filter_display.append(f"{desc}")
            else:
                self._filter_display.append(f"{desc} ({pattern})")

        self._filter_var = tk.StringVar(value=self._filter_display[0])
        self._filter_combo = ttk.Combobox(
            filter_frame,
            textvariable=self._filter_var,
            values=self._filter_display,
            state="readonly",
            width=40,
        )
        self._filter_combo.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self._filter_combo.bind("<<ComboboxSelected>>", lambda e: self._populate())

        # ── Buttons ──────────────────────────────────────────────
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=(4, 8))

        self._cancel_btn = ttk.Button(
            btn_frame, text="Cancel", command=self._on_cancel
        )
        self._cancel_btn.pack(side="right", padx=(5, 0))

        self._ok_btn = ttk.Button(
            btn_frame, text="Ok", command=self._on_ok, state="disabled"
        )
        self._ok_btn.pack(side="right")

    # ── Directory listing ────────────────────────────────────────

    def _get_active_extensions(self) -> set[str]:
        """Return the set of allowed extensions based on current filter selection."""
        idx = 0
        current = self._filter_var.get()
        for i, display in enumerate(self._filter_display):
            if display == current:
                idx = i
                break
        _, pattern = self._filetypes[idx]
        return _parse_extensions(pattern)

    def _populate(self) -> None:
        """Read the current directory and fill the Treeview."""
        self._tree.delete(*self._tree.get_children())
        self._loc_var.set(str(self._current_dir))

        extensions = self._get_active_extensions()

        dirs: list[tuple[str, float]] = []
        files: list[tuple[str, float]] = []

        try:
            with os.scandir(self._current_dir) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=True):
                            stat = entry.stat(follow_symlinks=True)
                            dirs.append((entry.name, stat.st_mtime))
                        elif entry.is_file(follow_symlinks=True):
                            # Apply extension filter
                            if extensions:
                                suffix = Path(entry.name).suffix.lower()
                                if suffix not in extensions:
                                    continue
                            stat = entry.stat(follow_symlinks=True)
                            files.append((entry.name, stat.st_mtime))
                    except (OSError, PermissionError):
                        # Skip entries we cannot stat
                        continue
        except (OSError, PermissionError) as exc:
            logger.warning("Cannot read directory %s: %s", self._current_dir, exc)

        # Sort according to current sort state
        dirs = self._sort_entries(dirs)
        files = self._sort_entries(files)

        dir_prefix = _DIR_PREFIX if self._use_emoji else _DIR_PREFIX_PLAIN
        file_prefix = _FILE_PREFIX if self._use_emoji else _FILE_PREFIX_PLAIN

        # ".." entry for going up
        if self._current_dir.parent != self._current_dir:
            self._tree.insert(
                "",
                "end",
                iid="__parent__",
                values=(f"{dir_prefix}..", ""),
                tags=("dir",),
            )

        # Directories first
        for name, mtime in dirs:
            display_name = f"{dir_prefix}{name}"
            self._tree.insert(
                "",
                "end",
                values=(display_name, _format_mtime(mtime)),
                tags=("dir",),
            )

        # Then files
        for name, mtime in files:
            display_name = f"{file_prefix}{name}"
            self._tree.insert(
                "",
                "end",
                values=(display_name, _format_mtime(mtime)),
                tags=("file",),
            )

        self._update_ok_state()
        self._update_sort_headings()

    def _sort_entries(
        self, entries: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Sort a list of (name, mtime) tuples according to the current sort state."""
        if self._sort_col == "name":
            entries.sort(key=lambda e: e[0].lower(), reverse=not self._sort_asc)
        else:  # modified
            entries.sort(key=lambda e: e[1], reverse=not self._sort_asc)
        return entries

    # ── Sorting ──────────────────────────────────────────────────

    def _on_sort(self, col: str) -> None:
        """Toggle sort order for the given column."""
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        self._populate()

    def _update_sort_headings(self) -> None:
        """Update column heading text to show the sort indicator."""
        arrow = " \u25b2" if self._sort_asc else " \u25bc"
        for col_id, label in [("name", "Name"), ("modified", "Last Modified")]:
            if col_id == self._sort_col:
                self._tree.heading(col_id, text=f"{label}{arrow}")
            else:
                self._tree.heading(col_id, text=label)

    # ── Navigation ───────────────────────────────────────────────

    def _navigate_to(self, path: Path) -> None:
        """Change the current directory and refresh."""
        resolved = path.resolve()
        if resolved.is_dir():
            self._current_dir = resolved
            self._populate()

    def _on_location_enter(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Handle Enter key in the location bar."""
        typed = self._loc_var.get().strip()
        if typed:
            candidate = Path(typed)
            if candidate.is_dir():
                self._navigate_to(candidate)
            else:
                self._loc_var.set(str(self._current_dir))

    def _on_double_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Handle double-click on a Treeview item."""
        sel = self._tree.selection()
        if not sel:
            return
        item = sel[0]
        tags = self._tree.item(item, "tags")
        values = self._tree.item(item, "values")

        if "dir" in tags:
            # Navigate into directory
            raw_name = values[0]
            name = self._strip_prefix(raw_name)
            if name == "..":
                self._navigate_to(self._current_dir.parent)
            else:
                self._navigate_to(self._current_dir / name)
        elif "file" in tags:
            # Select file and confirm
            self._on_ok()

    # ── Selection ────────────────────────────────────────────────

    def _on_selection_changed(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Enable/disable Ok button based on selection."""
        self._update_ok_state()

    def _update_ok_state(self) -> None:
        """Enable Ok only when at least one file is selected."""
        has_file_selected = False
        for item in self._tree.selection():
            tags = self._tree.item(item, "tags")
            if "file" in tags:
                has_file_selected = True
                break
        self._ok_btn.configure(state="normal" if has_file_selected else "disabled")

    # ── Actions ──────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """Confirm selection and close."""
        selected_paths: list[Path] = []
        for item in self._tree.selection():
            tags = self._tree.item(item, "tags")
            if "file" not in tags:
                continue
            values = self._tree.item(item, "values")
            name = self._strip_prefix(values[0])
            selected_paths.append(self._current_dir / name)

        if not selected_paths:
            return  # nothing to do, Ok should be disabled anyway

        if self._multiple:
            self._result = selected_paths
        else:
            self._result = selected_paths[0]

        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        """Cancel and close."""
        self._result = None
        self.grab_release()
        self.destroy()

    # ── Helpers ──────────────────────────────────────────────────

    def _strip_prefix(self, display_name: str) -> str:
        """Remove the directory/file prefix from a display name."""
        for prefix in (_DIR_PREFIX, _FILE_PREFIX, _DIR_PREFIX_PLAIN, _FILE_PREFIX_PLAIN):
            if display_name.startswith(prefix):
                return display_name[len(prefix):]
        return display_name.strip()
