"""Every assistant test runs with the user cache folder in its own tmp folder and a fresh docs store."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def user_cache_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``platformdirs.user_cache_dir`` points into ``tmp_path``: no test writes the real user cache."""
    import platformdirs

    from flextool.assistant import docs_store

    cache = tmp_path / "user_cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *args, **kwargs: str(cache))
    docs_store.reset()
    yield cache
    docs_store.reset()
