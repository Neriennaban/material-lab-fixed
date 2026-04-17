"""Общие фикстуры для семейственных renderer-тестов (Phase 1+)."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def pipeline():
    """Fresh pipeline V3 instance (дорогой конструктор — scope=module)."""
    from core.metallography_v3.pipeline_v3 import MetallographyPipelineV3

    return MetallographyPipelineV3()


@pytest.fixture
def default_size():
    return (256, 256)


@pytest.fixture
def default_seed():
    return 2026
