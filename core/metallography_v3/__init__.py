from __future__ import annotations

from importlib import import_module

from runtime_patches import apply_runtime_patches

apply_runtime_patches()

__all__ = ["MetallographyPipelineV3", "BatchResultV3"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".pipeline_v3", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
