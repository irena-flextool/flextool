"""FlexTool energy system optimization model package."""
import os as _os
# jemalloc decay config — set BEFORE the first ``import polars`` (which
# happens transitively in the imports just below).  polars statically
# links jemalloc; jemalloc reads ``_RJEM_MALLOC_CONF`` exactly once at
# .so load.  Its default ``dirty_decay_ms`` (~10 s) is longer than the
# gap between rolling-solve rolls, so each roll's freed GB-scale
# coefficient frames stay resident as dirty pages when the next roll
# starts allocating — the per-roll ``priv_dirty`` floor ratchets up
# (~+2 GB/roll on the 9-roll DES run).  ``dirty_decay_ms:1000`` purges a
# roll's freed pages ~1 s after they go idle (well before the next roll,
# which takes minutes) without eagerly madvise-ing on every transient
# free inside a hot operation.  ``muzzy_decay_ms:0`` forces MADV_DONTNEED
# (so Private_Dirty actually drops) rather than leaving pages as
# reclaimable-but-resident MADV_FREE.  glibc knobs (MALLOC_ARENA_MAX,
# malloc_trim) do NOT touch polars memory — this is the lever that does.
# ``setdefault`` so a shell-provided value still wins (A/B profiling).
_os.environ.setdefault("_RJEM_MALLOC_CONF", "dirty_decay_ms:1000,muzzy_decay_ms:0")
__all__ = [
    'write_outputs',
    'migrate_database',
    'initialize_database',
    'update_flextool',
]
from flextool.process_outputs import write_outputs
from flextool.update_flextool import migrate_database, initialize_database, update_flextool

name = "flextool"
