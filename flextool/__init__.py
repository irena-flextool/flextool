"""FlexTool energy system optimization model package."""
# On a processor without AVX2/FMA/BMI2/MOVBE, polars' default native runtime
# can crash as soon as polars is imported (below, transitively).  Read the
# CPU flags first (stdlib only) and, when the compatible runtime of the
# installed polars version is needed, say why and give the pip command: in a
# terminal FlexTool asks before running it; without one (a GUI subprocess, a
# Toolbox tool, CI) the import stops with that message (an ImportError).  The
# GUI launchers import this package first, so the GUI asks here too: in its
# terminal if it has one, otherwise in a small tkinter yes/no dialog, both
# before polars is imported; on no it exits with the message.  On a CPU that
# runs the default runtime, with the compatible runtime installed next to it
# (the Toolbox bundle ships both), POLARS_PREFER_PKG selects the default one:
# polars loads the compatible runtime first whenever it is installed.  See
# flextool.env_check.ensure_polars_runtime_before_import.  env_check.py is
# loaded from its file under a private name, so ``python -m
# flextool.env_check`` still runs a fresh module (no runpy warning).
def _ensure_polars_runtime() -> str:
    import importlib.util
    import os
    import sys

    name = "flextool._env_check_at_import"
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), "env_check.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while it executes
    try:
        spec.loader.exec_module(module)
        return module.ensure_polars_runtime_before_import()
    finally:
        sys.modules.pop(name, None)


# ``python -m flextool.env_check`` on a processor that needs the compatible
# polars build, which is not installed: the diagnostics run without polars in
# this process (the default runtime would crash it here, before they print);
# their probe imports polars in a child process.  The rest of the package is
# not imported then.
if _ensure_polars_runtime() != "diagnostics":
    # jemalloc decay config for polars, applied BEFORE the first ``import polars``
    # (which happens transitively in the imports below).  polars statically links
    # jemalloc and reads ``_RJEM_MALLOC_CONF`` once, when its runtime library
    # loads; polars 1.x's own values are ``dirty_decay_ms:500,muzzy_decay_ms:-1``,
    # and muzzy -1 leaves freed pages resident (MADV_FREE) for good (polars 2.x
    # writes ``muzzy_decay_ms:1000``, which returns them after ~2 s).  Then each
    # rolling solve's freed GB-scale coefficient frames stay resident and peak
    # memory roughly doubles on large models.  ``dirty_decay_ms:1000`` purges
    # freed pages ~1 s after they go idle without madvise-ing on every transient
    # free inside a hot operation; ``muzzy_decay_ms:0`` forces MADV_DONTNEED, so
    # Private_Dirty actually drops.  glibc knobs (MALLOC_ARENA_MAX, malloc_trim)
    # do NOT touch polars memory: this is the lever that does.  When polars was
    # imported before flextool (a Toolbox console, a notebook), the same values
    # are written into jemalloc at runtime; see ``flextool/_allocator.py``.
    # The run's thread budget (``--threads`` / ``FLEXTOOL_THREADS`` / the available cores) sizes polars' thread
    # pool, which polars also reads only at import: ``POLARS_MAX_THREADS`` is set here and put back as it was once
    # polars is loaded (below), so the environment does not carry it to child processes.  See
    # ``flextool/_thread_budget.py``.
    from flextool._thread_budget import apply_before_polars as _apply_thread_budget  # noqa: E402
    _polars_threads_before = _apply_thread_budget()
    from flextool._allocator import configure_polars_allocator as _configure_polars_allocator  # noqa: E402
    _configure_polars_allocator()
    # polars' default ``engine="auto"`` lets polars 2.x run some lazy queries on the
    # streaming engine, whose joins, group_by and unique return rows in an order
    # that changes from run to run, and whose partial sums are added in a varying
    # order (last-bit differences in the discount factors, hence in the LP).  The
    # outputs, the LP and the parity tests are built and verified on the in-memory
    # engine (what ``auto`` is in polars 1.x), so FlexTool selects it; a
    # POLARS_ENGINE_AFFINITY from the environment wins.  Calls that pass
    # ``engine="streaming"`` explicitly sort or reduce order-independently.
    import os as _os  # noqa: E402  (after the allocator config above)
    if "POLARS_ENGINE_AFFINITY" not in _os.environ:
        import polars as _pl
        _pl.Config.set_engine_affinity("in-memory")
    import warnings as _warnings  # noqa: E402  (after the allocator config above)
    # Silence requests' RequestsDependencyWarning.  The FlexTool venv ships a
    # ``requests`` whose pinned urllib3/charset_normalizer version ranges lag the
    # (working) installed versions, so ``import requests`` warns once at import
    # time.  It is harmless but leaks into Spine Toolbox's Tool console for any
    # path that pulls in requests (e.g. an ``http://`` Spine DB-server URL).
    # Match by message — set BEFORE requests is imported transitively below — so
    # we needn't import requests here just to reference its warning class, and the
    # filter stays robust across the venv's exact version numbers.
    _warnings.filterwarnings(
        "ignore",
        message=r"urllib3 .*doesn't match a supported version",
    )
    try:
        import polars as _pl  # noqa: E402,F811  (the pool size is fixed here)
        from flextool._thread_budget import absorb_polar_high_default as _absorb_polar_high_default  # noqa: E402
        _absorb_polar_high_default()
        from flextool.process_outputs import write_outputs  # noqa: E402  (after env/warning config above)
        from flextool.update_flextool import migrate_database, initialize_database, update_flextool  # noqa: E402
    finally:
        from flextool._thread_budget import restore_env as _restore_thread_env  # noqa: E402
        _restore_thread_env(_polars_threads_before)

__all__ = [
    'write_outputs',
    'migrate_database',
    'initialize_database',
    'update_flextool',
]
name = "flextool"
