from __future__ import annotations

import sys
from pathlib import Path

from runtime_patches import apply_runtime_patches

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

apply_runtime_patches()


def pytest_ignore_collect(collection_path: Path, path=None, config=None) -> bool:
    parts = set(collection_path.parts)
    if any(
        part in {"build", "dist", "tmp", "__pycache__", ".pytest_cache"}
        for part in parts
    ):
        return True
    return collection_path.name.startswith("pytest-cache-files-")
