from __future__ import annotations

import logging
import tkinter as tk
from functools import wraps
from typing import Any, Callable

from flextool.gui.dialogs import message_box

logger = logging.getLogger(__name__)

SOMETHING_WENT_WRONG = "Something went wrong"
SEE_THE_LOG = "The full error is in the FlexTool log. If it happens again, please report it."


def safe_callback(func: Callable) -> Callable:
    """Decorator that wraps a tkinter callback to catch and display exceptions.

    Prevents the GUI from crashing silently on errors: the error goes to the log and to a FlexTool message box over
    the first window among the arguments (the main window for its methods).
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}")
            parent = next((arg for arg in args if isinstance(arg, tk.Misc)), None)
            try:
                message_box.tell(parent, SOMETHING_WENT_WRONG, str(e), SEE_THE_LOG, kind="error")
            except Exception:
                pass  # Can't show dialog, just log
    return wrapper
