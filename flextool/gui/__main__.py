from __future__ import annotations

import logging
import signal
import sys


def main() -> None:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Ensure projects directory exists and load global settings for theme
    from flextool.gui.project_utils import get_projects_dir
    from flextool.gui.settings_io import load_global_settings

    projects_dir = get_projects_dir()
    projects_dir.mkdir(parents=True, exist_ok=True)
    settings = load_global_settings(projects_dir)

    # Create and run the main window, passing the initial theme
    from flextool.gui.main_window import MainWindow

    root = MainWindow(initial_theme=settings.theme)

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
