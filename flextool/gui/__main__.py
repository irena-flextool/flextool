from __future__ import annotations

import logging
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
    root.mainloop()


if __name__ == "__main__":
    main()
