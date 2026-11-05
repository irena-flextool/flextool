"""FlexTool energy system optimization model package."""
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
from flextool._allocator import configure_polars_allocator as _configure_polars_allocator
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
__all__ = [
    'write_outputs',
    'migrate_database',
    'initialize_database',
    'update_flextool',
]
from flextool.process_outputs import write_outputs  # noqa: E402  (after env/warning config above)
from flextool.update_flextool import migrate_database, initialize_database, update_flextool  # noqa: E402

name = "flextool"
