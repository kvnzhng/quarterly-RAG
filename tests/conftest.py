from __future__ import annotations

import pytest

from rag_project.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a temp data dir, independent of any local .env."""
    return Settings(_env_file=None, data_dir=tmp_path / "data")
