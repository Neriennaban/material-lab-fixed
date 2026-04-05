from __future__ import annotations

import numpy as np

from .base import SystemGenerationContext, SystemGenerationResult, soft_unsharp
from .common import run_phase_map_system
from .fe_c_unified import render_fe_c_unified
from core.metallography_v3.transformation_state import metadata_blocks_from_transformation_state


def generate_fe_c(ctx: SystemGenerationContext) -> SystemGenerationResult:
    stage = str(ctx.stage).strip().lower()
    try:
        return render_fe_c_unified(ctx)
    except Exception as exc:
        # Internal fallback preserves compatibility for edge/legacy states.
        def _stretch(image_gray):
            arr = image_gray.astype(np.float32)
            lo = float(np.percentile(arr, 2.0))
            hi = float(np.percentile(arr, 98.0))
            if hi <= lo + 1e-6:
                return image_gray
            arr = (arr - lo) / (hi - lo)
            arr = arr * 210.0 + 18.0
            return np.clip(arr, 0.0, 255.0).astype(np.uint8)

        def _post(image_gray, _meta):
            arr = _stretch(image_gray)
            if stage in {
                "alpha_pearlite",
                "pearlite",
                "pearlite_cementite",
                "martensite",
                "martensite_tetragonal",
                "martensite_cubic",
                "troostite_quench",
                "troostite_temper",
                "sorbite_quench",
                "sorbite_temper",
                "bainite",
                "tempered_low",
                "tempered_medium",
                "tempered_high",
                "ledeburite",
            }:
                enhanced = arr.astype(np.float32)
                enhanced = np.where(enhanced > 128.0, np.minimum(255.0, enhanced * 1.28 + 22.0), enhanced * 0.78)
                enhanced = np.where(enhanced < 64.0, np.maximum(0.0, enhanced * 0.58), enhanced)
                return soft_unsharp(enhanced.astype(np.uint8), amount=0.66)
            return soft_unsharp(arr, amount=0.36)

        legacy = run_phase_map_system(
            ctx=ctx,
            system_name="fe-c",
            generator_name="system_fe_c",
            postprocess=_post,
        )
        legacy.metadata["system_generator_extra"] = {
            "fe_c_unified": {
                "enabled": False,
                "stage_coverage_pass": False,
                "resolved_stage": str(stage),
                "blending_mode": "fractional",
                "fallback_reason": f"internal_fallback:{type(exc).__name__}",
            }
        }
        legacy.metadata["fe_c_phase_render"] = {
            "input_phase_fractions": dict(ctx.phase_fractions),
            "normalized_phase_fractions": dict(ctx.phase_fractions),
            "rendered_phase_layers": list(dict(ctx.phase_fractions).keys()),
            "seed_split": {},
            "phase_masks_present": bool(legacy.phase_masks),
        }
        legacy.metadata.update(metadata_blocks_from_transformation_state(ctx.transformation_state))
        legacy.metadata["engineering_trace"] = {
            **dict(legacy.metadata.get("engineering_trace", {})),
            "physics_guided_realism": bool(ctx.transformation_state),
        }
        return legacy
