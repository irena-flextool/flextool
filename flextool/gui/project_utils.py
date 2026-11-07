from __future__ import annotations

import importlib.util
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Subdirectories created inside every new project
PROJECT_SUBDIRS = [
    "input_sources",
    "converted",
    "intermediate",
    "work",
    "output_plots",
    "output_parquet",
    "output_csv",
    "output_excel",
    "output_plot_comparisons",
]


def get_projects_dir() -> Path:
    """Return the path to the top-level ``projects/`` directory.

    Resolved against the user's current working directory.  In an
    editable install with the GUI launched from the repo root this is
    the historical ``<repo>/projects/``; in a wheel install the user is
    expected to launch ``flextool-gui`` from their workspace, where
    ``./projects/`` is created on demand.
    """
    return Path.cwd() / "projects"


def seed_plot_settings(project_path: Path) -> Path:
    """Ensure ``<project_path>/plot_settings.yaml`` exists.

    Seeds the per-project plot color template from the bundled default so
    the project owns an editable copy.  Never clobbers an existing file
    (preserves user edits).  Returns the path to the project file.
    """
    plot_settings_path = Path(project_path) / "plot_settings.yaml"
    if not plot_settings_path.exists():
        from flextool._resources import package_data_path
        bundled = package_data_path("schemas/default_plot_settings.yaml")
        shutil.copy2(bundled, plot_settings_path)
    return plot_settings_path


def create_project(name: str) -> Path:
    """Create a new project directory with all required subdirectories.

    Returns the path to the newly created project directory.
    """
    projects_dir = get_projects_dir()
    project_path = projects_dir / name
    project_path.mkdir(parents=True, exist_ok=True)

    for subdir in PROJECT_SUBDIRS:
        (project_path / subdir).mkdir(exist_ok=True)

    # Seed the per-project plot color template from the bundled default so
    # the project owns an editable copy.
    seed_plot_settings(project_path)

    return project_path


def list_projects() -> list[str]:
    """Return a sorted list of project directory names.

    Returns an empty list if the projects directory does not exist.
    """
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in projects_dir.iterdir()
        if entry.is_dir()
    )


def rename_project(old_name: str, new_name: str) -> Path:
    """Rename a project directory.

    Returns the path to the renamed project directory.

    Raises:
        FileNotFoundError: If the source project does not exist.
        FileExistsError: If a project with the new name already exists.
    """
    projects_dir = get_projects_dir()
    old_path = projects_dir / old_name
    new_path = projects_dir / new_name

    if not old_path.exists():
        raise FileNotFoundError(f"Project '{old_name}' does not exist.")
    if new_path.exists():
        raise FileExistsError(f"Project '{new_name}' already exists.")

    old_resolved = old_path.resolve()
    old_path.rename(new_path)
    _assistant_project_renamed(old_resolved, new_path.resolve())
    return new_path


def _assistant_store():  # noqa: ANN202 — a flextool.assistant.store.Store, or None
    """The AI assistant's conversation store, ``None`` when the assistant is not installed."""
    try:
        if importlib.util.find_spec("flextool.assistant") is None:
            return None
    except (ImportError, ValueError):
        return None
    from flextool.assistant.store import open_store

    return open_store()


def _assistant_project_renamed(old: Path, new: Path) -> None:
    """Move the renamed project's saved AI assistant chats to its new name.  A failure is logged: the rename
    stands, and the chats left under the old name expire there."""
    try:
        store = _assistant_store()
        if store is not None:
            store.rename_project(old, new)
    except Exception:  # noqa: BLE001 — never undo or fail a rename for the chats
        logger.warning("The AI assistant chats of project %s were not moved to %s", old.name, new.name,
                       exc_info=True)


def notify_assistant_project_deleted(project_path: Path) -> None:
    """A project folder was deleted: delete its saved AI assistant chats (those open in another FlexTool window
    or terminal are skipped and expire).  A failure is logged; the deletion is never undone."""
    try:
        store = _assistant_store()
        if store is not None:
            store.forget_project(Path(project_path).absolute())
    except Exception:  # noqa: BLE001
        logger.warning("The AI assistant chats of deleted project %s were not deleted", Path(project_path).name,
                       exc_info=True)
