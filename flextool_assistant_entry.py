"""``flextool-assistant``: the read-only FlexTool tools for AI assistants (``flextool.assistant.cli``).

With ``--mcp`` the command speaks the protocol of AI agents on stdout, so nothing else may ever write there.
``import flextool`` loads polars, pandas, spinedb_api and FlexTool's own start-up checks, any of which may print.
So this module, which uses only the standard library, first keeps private copies of stdin and stdout for the
protocol and points the real ones away: file descriptor 1 and ``sys.stdout`` at stderr, file descriptor 0 at the
null device (a stray ``input()`` reads end of file), on Windows also the process's standard handles.  Only then is
``flextool`` imported, and the protocol runs on the private copies.  This module sets no environment variables
and starts no processes.

When ``import flextool`` stops because this processor needs the compatible polars build
(``flextool.env_check``, an ``ImportError`` named ``CompatRuntimeRequired``), the connection still starts, with
one tool that gives the user the fix (``run_degraded`` of ``flextool/assistant/mcp_server.py``, loaded from its
file because the package cannot be imported).

Without ``--mcp`` the command line goes straight to ``flextool.assistant.cli``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_PRIVATE_NAME = "flextool_assistant_entry_mcp_server"


def _rebind_windows_std_handles() -> None:
    """Point the Win32 standard handles at descriptors 0 and 1 as they are now (``os.dup2`` changes only the C
    runtime's table; child processes and native code use the handles)."""
    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetStdHandle.argtypes = (ctypes.c_ulong, ctypes.c_void_p)
    for fd, std_id in ((0, -10), (1, -11)):           # STD_INPUT_HANDLE, STD_OUTPUT_HANDLE
        kernel32.SetStdHandle(std_id, msvcrt.get_osfhandle(fd))


def redirect_stdio() -> tuple[int, int]:
    """Keep private copies of stdin and stdout for the protocol and point the real ones away.

    Returns ``(proto_in, proto_out)``, the descriptors the protocol uses.
    """
    sys.stdout.flush()
    proto_out = os.dup(1)
    proto_in = os.dup(0)
    os.dup2(2, 1)
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, 0)
    finally:
        os.close(null)
    sys.stdout = sys.stderr
    if sys.platform == "win32":
        import msvcrt

        msvcrt.setmode(proto_in, os.O_BINARY)
        msvcrt.setmode(proto_out, os.O_BINARY)
        _rebind_windows_std_handles()
    return proto_in, proto_out


def _load_mcp_server() -> ModuleType:
    """``flextool/assistant/mcp_server.py`` loaded from its file, without importing the ``flextool`` package."""
    spec = importlib.util.find_spec("flextool")
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    if not locations:
        raise ImportError("FlexTool is not installed in this Python environment (no 'flextool' package).")
    path = Path(locations[0]) / "assistant" / "mcp_server.py"
    file_spec = importlib.util.spec_from_file_location(_PRIVATE_NAME, path)
    if file_spec is None or file_spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(file_spec)
    sys.modules[_PRIVATE_NAME] = module
    try:
        file_spec.loader.exec_module(module)
    finally:
        sys.modules.pop(_PRIVATE_NAME, None)
    return module


def serve(argv: list[str], proto_in: int, proto_out: int) -> int:
    """Start the connection on the protocol descriptors (after :func:`redirect_stdio`)."""
    try:
        from flextool.assistant import cli
    except ImportError as exc:
        if type(exc).__name__ != "CompatRuntimeRequired":
            raise
        problem = str(exc)
        try:
            server = _load_mcp_server()
        except ImportError as load_error:
            print(f"{problem}\n\nThe connection for AI assistants could not start either: {load_error}",
                  file=sys.stderr)
            return 2
        try:
            server.run_degraded(problem, proto_in, proto_out)
        except ImportError as mcp_error:
            print(f"{problem}\n\nThe connection for AI assistants could not load the package mcp: {mcp_error}.",
                  file=sys.stderr)
            return 2
        return 0
    return cli.main(argv, proto_in=proto_in, proto_out=proto_out)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if "--mcp" not in args:
        from flextool.assistant import cli

        return cli.main(args)
    proto_in, proto_out = redirect_stdio()
    return serve(args, proto_in, proto_out)


if __name__ == "__main__":
    sys.exit(main())
