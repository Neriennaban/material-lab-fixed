from __future__ import annotations

from typing import Any

import numpy as np

from core.generator_phase_map import generate_phase_stage_structure
from core.metallography_v3.transformation_state import metadata_blocks_from_transformation_state

from .base import (
    SystemGenerationContext,
    SystemGenerationResult,
    build_composition_effect,
    build_phase_masks_from_intensity,
    build_phase_visibility_report,
    ensure_u8,
)


def run_phase_map_system(
    *,
    ctx: SystemGenerationContext,
    system_name: str,
    generator_name: str,
    postprocess: Any | None = None,
) -> SystemGenerationResult:
    stage = str(ctx.stage or "auto")
    raw = generate_phase_stage_structure(
        size=ctx.size,
        seed=int(ctx.seed),
        system=str(system_name),
        composition=dict(ctx.composition_wt),
        stage=stage,
        temperature_c=float(ctx.processing.temperature_c),
        cooling_mode=str(ctx.processing.cooling_mode),
        deformation_pct=float(ctx.processing.deformation_pct),
        aging_temperature_c=float(ctx.processing.aging_temperature_c),
        aging_hours=float(ctx.processing.aging_hours),
    )

    image_gray = ensure_u8(raw["image"])
    if callable(postprocess):
        image_gray = ensure_u8(postprocess(image_gray, raw.get("metadata", {})))

    raw_masks = raw.get("phase_masks")
    phase_masks: dict[str, np.ndarray] = {}
    if isinstance(raw_masks, dict):
        for name, mask in raw_masks.items():
            if isinstance(mask, np.ndarray):
                phase_masks[str(name)] = (mask > 0).astype(np.uint8)
    if not phase_masks:
        phase_masks = build_phase_masks_from_intensity(
            image_gray=image_gray,
            phase_fractions=ctx.phase_fractions,
            seed=int(ctx.seed) + 17,
        )

    visibility = build_phase_visibility_report(
        image_gray=image_gray,
        phase_masks=phase_masks,
        phase_fractions=ctx.phase_fractions,
        tolerance_pct=float(ctx.phase_fraction_tolerance_pct),
    )
    composition_effect = build_composition_effect(
        system=system_name,
        composition_wt=ctx.composition_wt,
        mode=str(ctx.composition_sensitivity_mode),
        seed=int(ctx.seed),
        single_phase_compensation=bool(len(phase_masks) <= 1),
    )

    metadata = {
        "system_generator_name": str(generator_name),
        "resolved_stage": str(raw.get("metadata", {}).get("resolved_stage", stage)),
        "phase_transition_state": dict(raw.get("metadata", {}).get("phase_transition_state", {})),
        "composition_effect": composition_effect,
        "phase_visibility_report": visibility,
        **metadata_blocks_from_transformation_state(ctx.transformation_state),
        "engineering_trace": {
            "generation_mode": str(ctx.generation_mode),
            "phase_emphasis_style": str(ctx.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(ctx.phase_fraction_tolerance_pct),
            "system_generator_name": str(generator_name),
            "physics_guided_realism": bool(ctx.transformation_state),
        },
    }
    return SystemGenerationResult(
        image_gray=image_gray,
        phase_masks=phase_masks,
        metadata=metadata,
    )
