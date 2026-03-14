from __future__ import annotations

from pathlib import Path

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

    This is located at the repository root, i.e.
    ``<repo>/projects/`` which is three levels up from this file
    (``flextool/gui/project_utils.py``).
    """
    return Path(__file__).parent.parent.parent / "projects"


def create_project(name: str) -> Path:
    """Create a new project directory with all required subdirectories.

    Returns the path to the newly created project directory.
    """
    projects_dir = get_projects_dir()
    project_path = projects_dir / name
    project_path.mkdir(parents=True, exist_ok=True)

    for subdir in PROJECT_SUBDIRS:
        (project_path / subdir).mkdir(exist_ok=True)

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

    old_path.rename(new_path)
    return new_path
