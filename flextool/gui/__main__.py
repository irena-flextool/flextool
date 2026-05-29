from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger("flextool.gui")


def _configure_logging(log_dir: Path) -> None:
    """Set up logging for the desktop application.

    Adds a console handler only when a real stream exists -- under the
    windowed launcher on Windows (``pythonw.exe``) ``sys.stdout`` and
    ``sys.stderr`` are ``None`` -- and always adds a rotating file handler so
    diagnostics survive even when there is no console to print them to.
    """
    from logging.handlers import RotatingFileHandler

    handlers: list[logging.Handler] = []

    stream = sys.stderr or sys.stdout  # both None under pythonw.exe
    if stream is not None:
        handlers.append(logging.StreamHandler(stream))

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_dir / "flextool_gui.log",
                maxBytes=1_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )
    except OSError:
        # A missing log file must never stop the GUI from starting.
        pass

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=handlers,
    )


def _install_exception_logging() -> None:
    """Route otherwise-uncaught exceptions to the log.

    Without a console most users never see (let alone report) a traceback, so
    funnel interpreter-level exceptions into the rotating log file configured
    by :func:`_configure_logging`.
    """

    def _hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook


def main() -> None:
    # Ensure projects directory exists and load global settings for theme
    from flextool.gui.project_utils import get_projects_dir
    from flextool.gui.settings_io import load_global_settings

    projects_dir = get_projects_dir()
    projects_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging (file fallback first, so nothing is lost without a console)
    _configure_logging(projects_dir)
    _install_exception_logging()

    settings = load_global_settings(projects_dir)

    # Create and run the main window, passing the initial theme
    from flextool.gui.main_window import MainWindow

    root = MainWindow(initial_theme=settings.theme)

    # Tkinter prints callback exceptions to stderr by default, which is absent
    # under the windowed launcher; send them to the log instead.
    root.report_callback_exception = lambda exc_type, exc, tb: logger.error(
        "Unhandled Tk callback exception", exc_info=(exc_type, exc, tb)
    )

    # Set up signal handlers for clean shutdown (SIGINT = Ctrl+C, SIGTERM = kill)
    def _signal_handler(signum: int, frame: object) -> None:
        """Handle SIGINT/SIGTERM by cleaning up and exiting."""
        try:
            if hasattr(root, "execution_mgr") and root.execution_mgr:
                root.execution_mgr.cleanup()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Periodically yield control from tkinter's mainloop so Python can
    # process pending signal handlers (tkinter blocks them on some platforms).
    def _check_signals() -> None:
        root.after(500, _check_signals)

    root.after(500, _check_signals)

    root.mainloop()


if __name__ == "__main__":
    main()
