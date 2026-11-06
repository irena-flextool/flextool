"""``_stream_concat_parts`` must not leave its ``.<var>__concat_*.parquet`` temp in
``output_raw/`` when the delete hits a transient Windows sharing lock.

On Windows CI a just-written-and-read temp file can stay briefly locked (antivirus /
indexer handle), so a one-shot ``unlink`` raised ``PermissionError`` and the old
best-effort cleanup swallowed it — the temp then showed up as an extra output file
(``test_csv_dump_does_not_change_outputs[multi_year]``, run 42225355914). The lock is
simulated here so the check runs on every OS."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import importlib

# The package re-exports a ``read_variables`` function that shadows the module name.
read_variables = importlib.import_module("flextool.process_outputs.read_variables")


def test_concat_temp_removed_despite_transient_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = []
    for i in range(2):
        p = tmp_path / f"v_x__solve{i}.parquet"
        pq.write_table(pa.table({"a": [float(i)], "b": [1.0 + i]}), str(p))
        parts.append(p)

    real_unlink = Path.unlink
    failures = {"left": 3}

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if "__concat_" in self.name and failures["left"] > 0:
            failures["left"] -= 1
            raise PermissionError(32, "The process cannot access the file", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    df = read_variables._stream_concat_parts(parts)

    assert failures["left"] == 0, "the simulated lock was never hit"
    assert len(df) == 2
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("."))
    assert leftovers == [], f"concat temp leaked into the output folder: {leftovers}"
