from __future__ import annotations

import importlib.util
import logging
import shutil
import threading
from collections import deque
from collections.abc import Callable
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
    # This computer created the project: option files put in its solver_config/ later are trusted (item 240 B3;
    # only when nothing received is there, which a folder of that name left behind could hold).
    from flextool._solver_config_source import note_project_set_up_here
    note_project_set_up_here(project_path)

    return project_path


def list_projects() -> list[str]:
    """Return a sorted list of project directory names.

    Returns an empty list if the projects directory does not exist.  Folders whose name starts with ``.`` are
    not projects: a project import unpacks into ``.<name>.partial`` and a stopped import can leave one behind.
    """
    projects_dir = get_projects_dir()
    if not projects_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in projects_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
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
    from flextool import _local_solver_settings as local
    try:
        trusted = local.is_trusted_workspace(old_path)
    except local.LocalSolverSettingsError:
        trusted = False
    old_path.rename(new_path)
    if trusted:
        # The project keeps this computer's trust of its solver option files under its new name (item 240 B3);
        # its old path is taken off the list, so a folder made there later does not inherit it.
        from flextool._solver_config_source import forget_trusted_folder
        forget_trusted_folder(old_resolved)
        try:
            local.add_trusted_workspace(new_path)
        except local.LocalSolverSettingsError:
            pass
    _run_project_hook(_assistant_project_renamed, old_resolved, new_path.resolve())
    return new_path


# --- the AI assistant's project hooks ---------------------------------------------------------------------------
# A project rename moves its saved chats and a deletion deletes them: store work under file locks, which may wait
# for another FlexTool window or terminal (review G18, C7-n5).  The callers are on the Tk thread, so the hooks run
# on one daemon thread, one after another in the order they were asked for (a rename A→B, then B→C, moves in that
# order).  Everything that uses the store for a project waits for them first (:func:`wait_project_hooks`, on its
# own thread), so a chat window opened right after a rename sees the moved chats, and a project deleted and made
# again does not lose its new chats to the queued deletion.  The thread ends when the queue is empty.
_HOOKS: deque[tuple[Callable[..., None], tuple]] = deque()
_HOOKS_CHANGED = threading.Condition()
_hook_thread: threading.Thread | None = None
#: How long a store user waits for the queued project hooks before it goes on (logged).
PROJECT_HOOKS_WAIT_S = 10.0


def _run_project_hook(fn: Callable[..., None], *args: object) -> None:
    global _hook_thread
    with _HOOKS_CHANGED:
        _HOOKS.append((fn, args))
        if _hook_thread is None:
            _hook_thread = threading.Thread(target=_run_hooks, name="flextool-project-hooks", daemon=True)
            _hook_thread.start()


def _run_hooks() -> None:
    global _hook_thread
    while True:
        with _HOOKS_CHANGED:
            if not _HOOKS:
                _hook_thread = None
                _HOOKS_CHANGED.notify_all()
                return
            fn, args = _HOOKS[0]
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 — each hook logs its own failure; the next ones still run
            logger.exception("A project hook of the AI assistant failed")
        finally:
            with _HOOKS_CHANGED:
                _HOOKS.popleft()            # removed only once done: a waiter sees it pending while it runs
                _HOOKS_CHANGED.notify_all()


def project_hooks_idle() -> bool:
    """Whether no project hook is queued or running (no wait; the Tk thread may ask)."""
    with _HOOKS_CHANGED:
        return not _HOOKS


def wait_project_hooks(timeout: float | None = PROJECT_HOOKS_WAIT_S) -> bool:
    """Wait until the project hooks queued so far have run (never on the Tk thread); ``True`` when none is left.
    On a timeout it logs and returns ``False``: the caller goes on."""
    with _HOOKS_CHANGED:
        done = _HOOKS_CHANGED.wait_for(lambda: not _HOOKS, timeout)
    if not done:
        logger.warning("The AI assistant's project hooks (a rename or deletion of a project) are still running")
    return done


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
    """Move the renamed project's saved AI assistant chats to its new name (the hook thread).  A failure is
    logged: the rename stands, and the chats left under the old name expire there."""
    try:
        store = _assistant_store()
        if store is not None:
            store.rename_project(old, new)
    except Exception:  # noqa: BLE001 — never undo or fail a rename for the chats
        logger.warning("The AI assistant chats of project %s were not moved to %s", old.name, new.name,
                       exc_info=True)


def notify_assistant_project_deleted(project_path: Path) -> None:
    """A project folder was deleted: delete its saved AI assistant chats (those open in another FlexTool window
    or terminal are skipped and expire), on the hook thread.  A failure is logged; the deletion is never undone."""
    _run_project_hook(_assistant_project_deleted, Path(project_path).absolute())


def _assistant_project_deleted(project_path: Path) -> None:
    try:
        store = _assistant_store()
        if store is not None:
            store.forget_project(project_path)
    except Exception:  # noqa: BLE001
        logger.warning("The AI assistant chats of deleted project %s were not deleted", project_path.name,
                       exc_info=True)
