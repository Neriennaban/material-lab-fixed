from __future__ import annotations

import numpy as np

from core.metallography_v3.transformation_state import metadata_blocks_from_transformation_state

from .base import SystemGenerationContext, SystemGenerationResult, ensure_u8, soft_unsharp
from .common import run_phase_map_system

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def _composition_fraction(composition_wt: dict[str, float] | None, key: str) -> float:
    if not isinstance(composition_wt, dict):
        return 0.0
    total = 0.0
    cleaned: dict[str, float] = {}
    for name, value in composition_wt.items():
        try:
            vv = float(value)
        except Exception:
            continue
        if vv <= 0.0:
            continue
        cleaned[str(name).strip()] = vv
        total += vv
    if total <= 1e-12:
        return 0.0
    return float(cleaned.get(key, 0.0) / total * 100.0)


def _is_pure_iron_like(ctx: SystemGenerationContext) -> bool:
    stage = str(ctx.stage).strip().lower()
    ferritic_fraction = float(ctx.phase_fractions.get("BCC_B2", 0.0) + ctx.phase_fractions.get("FERRITE", 0.0))
    fe_pct = _composition_fraction(ctx.composition_wt, "Fe")
    c_pct = _composition_fraction(ctx.composition_wt, "C")
    si_pct = _composition_fraction(ctx.composition_wt, "Si")
    return bool(stage == "recrystallized_ferrite" and ferritic_fraction >= 0.95 and fe_pct >= 99.8 and c_pct <= 0.03 and si_pct <= 0.25)


def _lift_small_dark_defects(image_gray: np.ndarray, *, max_pixels: int = 18) -> np.ndarray:
    if ndimage is None:
        return image_gray
    arr = image_gray.astype(np.float32)
    threshold = float(np.quantile(arr, 0.08))
    mask = arr < threshold
    labels, count = ndimage.label(mask)
    if count <= 0:
        return image_gray
    local = ndimage.gaussian_filter(arr, sigma=1.1)
    out = arr.copy()
    for label in range(1, int(count) + 1):
        zone = labels == label
        if int(zone.sum()) <= int(max_pixels):
            out[zone] = 0.82 * local[zone] + 0.18 * out[zone]
    return ensure_u8(out)


def generate_fe_si(ctx: SystemGenerationContext) -> SystemGenerationResult:
    stage = str(ctx.stage).strip().lower()
    morphology_state = dict(ctx.transformation_state.get("morphology_state", {}))
    transformation_trace = dict(ctx.transformation_state.get("transformation_trace", {}))
    pure_iron_like = _is_pure_iron_like(ctx)

    def _post(image_gray, _meta):
        default_sharpness = 0.24 if pure_iron_like else (0.42 if stage == "cold_worked_ferrite" else 0.30)
        texture_sharpness = float(morphology_state.get("texture_sharpness", default_sharpness))
        img = soft_unsharp(image_gray, amount=max(0.18, min(0.72, texture_sharpness)))
        if stage == "cold_worked_ferrite":
            arr = img.astype(np.float32)
            h, w = arr.shape
            yy, xx = np.mgrid[0:h, 0:w]
            band_density = float(morphology_state.get("cold_work_band_density", transformation_trace.get("recrystallized_fraction", 0.0)))
            bands = np.sin(xx / max(1, w) * 17.0 + yy / max(1, h) * 5.0) * (2.0 + 8.0 * max(0.0, min(1.0, band_density)))
            return ensure_u8(arr + bands)
        if pure_iron_like:
            arr = img.astype(np.float32)
            if ndimage is not None:
                arr = ndimage.gaussian_filter(arr, sigma=0.35)
            lo = float(np.quantile(arr, 0.02))
            hi = float(np.quantile(arr, 0.98))
            if hi > lo + 1e-6:
                arr = (arr - lo) / (hi - lo)
                arr = arr * 88.0 + 132.0
            dark_floor = float(np.quantile(arr, 0.08))
            arr += max(0.0, 124.0 - dark_floor)
            arr = np.clip(arr, 118.0, 232.0)
            bright = ensure_u8(arr)
            bright = _lift_small_dark_defects(bright, max_pixels=22)
            if ndimage is not None:
                smooth = ndimage.gaussian_filter(bright.astype(np.float32), sigma=1.1)
                bright = ensure_u8(smooth + (bright.astype(np.float32) - smooth) * 0.12)
            return bright
        return img

    result = run_phase_map_system(
        ctx=ctx,
        system_name="fe-si",
        generator_name="system_fe_si",
        postprocess=_post,
    )
    result.metadata.update(metadata_blocks_from_transformation_state(ctx.transformation_state))
    result.metadata["engineering_trace"] = {
        **dict(result.metadata.get("engineering_trace", {})),
        "physics_guided_realism": bool(ctx.transformation_state),
    }
    if pure_iron_like:
        result.metadata["system_generator_extra"] = {
            "pure_iron_baseline": {
                "applied": True,
                "profile": "bright_clean_ferrite_v1",
                "expected_appearance": "almost_light_with_soft_boundaries",
            }
        }
        result.metadata["engineering_trace"] = {
            **dict(result.metadata.get("engineering_trace", {})),
            "pure_iron_baseline_applied": True,
            "pure_iron_target": "bright_ferritic_negative_control",
        }
    return result
