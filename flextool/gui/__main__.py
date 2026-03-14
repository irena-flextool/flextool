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

    # Ensure projects directory exists
    from flextool.gui.project_utils import get_projects_dir

    get_projects_dir().mkdir(parents=True, exist_ok=True)

    # Create and run the main window
    from flextool.gui.main_window import MainWindow

    root = MainWindow()
    root.mainloop()


if __name__ == "__main__":
    main()
