"""The run record: ``_run_status.json`` and ``_run.log`` in each run's ``output_parquet/<subdir>/`` folder.

* :mod:`._owner` -- file names, the marker, the log writer, the pipe owner's relay, status file I/O and
  :func:`results_usable` (standard library only; the ``flextool-run`` supervisor loads it from its file);
* :mod:`._status` -- the engine's :class:`RunRecorder` and :func:`active` / :func:`set_active`;
* :mod:`._tee` -- the engine's copy of its own console output when no pipe owner writes the log.
"""
from flextool.run_record._owner import (
    LOG_HEAD_BYTES,
    MARKER,
    RUN_LOG_FILE,
    RUN_LOG_MAX_BYTES_ENV,
    RUN_LOG_OWNER_ENV,
    RUN_RECORD_FILES,
    RUN_STATUS_FILE,
    STATUS_TMP_PREFIX,
    STATUS_TMP_SUFFIX,
    RunLogOwner,
    RunLogWriter,
    classify_exit,
    finalize_unfinished,
    format_marker,
    log_tail_bytes,
    now_iso,
    parse_iso,
    parse_marker,
    read_status_file,
    results_usable,
    unusable_reason,
    worst_verdict,
    write_status_atomic,
)
from flextool.run_record._status import (
    NULL_RECORDER,
    RunRecorder,
    active,
    effective_state,
    set_active,
)
from flextool.run_record._tee import TeeStream, install_tee

__all__ = [
    "LOG_HEAD_BYTES",
    "MARKER",
    "NULL_RECORDER",
    "RUN_LOG_FILE",
    "RUN_LOG_MAX_BYTES_ENV",
    "RUN_LOG_OWNER_ENV",
    "RUN_RECORD_FILES",
    "RUN_STATUS_FILE",
    "STATUS_TMP_PREFIX",
    "STATUS_TMP_SUFFIX",
    "RunLogOwner",
    "RunLogWriter",
    "RunRecorder",
    "TeeStream",
    "active",
    "classify_exit",
    "effective_state",
    "finalize_unfinished",
    "format_marker",
    "install_tee",
    "log_tail_bytes",
    "now_iso",
    "parse_iso",
    "parse_marker",
    "read_status_file",
    "results_usable",
    "set_active",
    "unusable_reason",
    "worst_verdict",
    "write_status_atomic",
]
