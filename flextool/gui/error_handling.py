from __future__ import annotations

import logging
import tkinter.messagebox as messagebox
import traceback
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def show_error(parent: Any, title: str, message: str) -> None:
    """Show error dialog."""
    messagebox.showerror(title, message, parent=parent)


def show_warning(parent: Any, title: str, message: str) -> None:
    """Show warning dialog."""
    messagebox.showwarning(title, message, parent=parent)


def show_info(parent: Any, title: str, message: str) -> None:
    """Show info dialog."""
    messagebox.showinfo(title, message, parent=parent)


def confirm(parent: Any, title: str, message: str) -> bool:
    """Show yes/no confirmation dialog. Returns True if user clicked Yes."""
    return messagebox.askyesno(title, message, parent=parent)


def safe_callback(func: Callable) -> Callable:
    """Decorator that wraps a tkinter callback to catch and display exceptions.

    Prevents the GUI from crashing silently on errors.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            try:
                # Try to show error in a dialog
                messagebox.showerror(
                    "Error",
                    f"An error occurred:\n\n{e}\n\nSee log for details.",
                )
            except Exception:
                pass  # Can't show dialog, just log
    return wrapper
