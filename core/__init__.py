"""Core package for synthetic microstructure generation and analysis."""

from .materials import MaterialPreset, load_preset, list_presets
from .pipeline import GenerationEngine, GenerationResult
from .pipeline_v2 import BatchResult, GenerationPipelineV2
from .generator_phase_map import generate_phase_stage_structure, supported_stages

__all__ = [
    "MaterialPreset",
    "load_preset",
    "list_presets",
    "GenerationEngine",
    "GenerationResult",
    "GenerationPipelineV2",
    "BatchResult",
    "generate_phase_stage_structure",
    "supported_stages",
]
