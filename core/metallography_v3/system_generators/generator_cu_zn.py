from __future__ import annotations

import numpy as np

from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    allocate_phase_masks,
    build_twins_from_labels,
    clamp,
    cooling_index,
    distance_to_mask,
    low_frequency_field,
    multiscale_noise,
    normalize01,
    rescale_to_u8,
)
from core.metallography_v3.transformation_state import metadata_blocks_from_transformation_state

from .base import (
    SystemGenerationContext,
    SystemGenerationResult,
    build_composition_effect,
    build_phase_visibility_report,
    ensure_u8,
)
from .common import run_phase_map_system


def _finalize(
    *,
    ctx: SystemGenerationContext,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray],
    stage: str,
    morphology: dict[str, object],
) -> SystemGenerationResult:
    visibility = build_phase_visibility_report(
        image_gray=ensure_u8(image_gray),
        phase_masks=phase_masks,
        phase_fractions=ctx.phase_fractions,
        tolerance_pct=float(ctx.phase_fraction_tolerance_pct),
    )
    composition_effect = build_composition_effect(
        system="cu-zn",
        composition_wt=ctx.composition_wt,
        mode=str(ctx.composition_sensitivity_mode),
        seed=int(ctx.seed),
        single_phase_compensation=bool(len(phase_masks) <= 1),
    )
    metadata = {
        "system_generator_name": "system_cu_zn",
        "resolved_stage": str(stage),
        "phase_transition_state": {"stage": str(stage), "transition_kind": "steady", "liquid_fraction": 0.0, "solid_fraction": 1.0, "thermal_direction": "steady"},
        "composition_effect": composition_effect,
        "phase_visibility_report": visibility,
        **metadata_blocks_from_transformation_state(ctx.transformation_state),
        "engineering_trace": {
            "generation_mode": str(ctx.generation_mode),
            "phase_emphasis_style": str(ctx.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(ctx.phase_fraction_tolerance_pct),
            "system_generator_name": "system_cu_zn",
            "realism_morphology": morphology,
            "physics_guided_realism": bool(ctx.transformation_state),
        },
        "system_generator_extra": {"cu_zn_morphology": morphology},
    }
    return SystemGenerationResult(image_gray=ensure_u8(image_gray), phase_masks=phase_masks, metadata=metadata)


def generate_cu_zn(ctx: SystemGenerationContext) -> SystemGenerationResult:
    stage = str(ctx.stage).strip().lower()
    if stage in {"liquid", "liquid_alpha", "liquid_beta"}:
        return run_phase_map_system(ctx=ctx, system_name="cu-zn", generator_name="system_cu_zn")

    size = ctx.size
    phases = {str(k): float(v) for k, v in dict(ctx.phase_fractions).items() if float(v) > 0.0}
    dominant = max(phases.items(), key=lambda item: float(item[1]))[0] if phases else "ALPHA"
    zn = float(ctx.composition_wt.get("Zn", 0.0))
    cool_idx = cooling_index(getattr(ctx.processing, "cooling_mode", "equilibrium"))
    def_pct = float(getattr(ctx.processing, "deformation_pct", 0.0) or 0.0)
    morphology_state = dict(ctx.transformation_state.get("morphology_state", {}))
    grain_size = clamp(float(morphology_state.get("grain_size_px", 86.0 - 0.8 * def_pct - 20.0 * cool_idx + (5.0 if stage == "beta" else 0.0))), 24.0, 120.0)
    grain = generate_grain_structure(
        size=size,
        seed=int(ctx.seed) + 101,
        mean_grain_size_px=grain_size,
        grain_size_jitter=0.22,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    labels = grain["labels"]
    boundaries = np.asarray(grain["boundaries"]).astype(bool)
    boundary_pref = normalize01(np.exp(-distance_to_mask(boundaries) / clamp(5.2 - 2.0 * cool_idx + max(0.0, zn - 36.0) * 0.04, 1.8, 7.5)).astype(np.float32))
    low = low_frequency_field(size=size, seed=int(ctx.seed) + 121, sigma=22.0)
    noise = multiscale_noise(size=size, seed=int(ctx.seed) + 143, scales=((18.0, 0.55), (6.0, 0.28), (2.0, 0.17)))

    twins = build_twins_from_labels(
        labels=labels,
        seed=int(ctx.seed) + 163,
        fraction_of_grains=clamp(float(morphology_state.get("twin_density", 0.52 if stage in {"alpha", "alpha_beta", "beta_prime"} else 0.22)), 0.10, 0.88),
        spacing_px=clamp(13.0 - 0.05 * max(0.0, zn - 32.0), 7.0, 15.0),
        width_px=1.3,
    )
    deformation_bands = build_twins_from_labels(
        labels=labels,
        seed=int(ctx.seed) + 181,
        fraction_of_grains=clamp(float(morphology_state.get("deformation_band_density", 0.20 + def_pct / 60.0)), 0.18, 0.78),
        spacing_px=clamp(18.0 - 0.08 * def_pct, 8.0, 18.0),
        width_px=2.1,
    ) if stage == "cold_worked" or "DEFORMATION_BANDS" in phases else np.zeros(size, dtype=bool)

    alpha_field = normalize01((grain["image"].astype(np.float32) / 255.0) * 0.58 + (1.0 - boundary_pref) * 0.28 + low * 0.14)
    beta_field = normalize01(boundary_pref * 0.72 + low * 0.14 + noise * 0.14)
    beta_prime_field = normalize01(boundary_pref * (0.48 + 0.24 * float(morphology_state.get("ordering_factor", 0.0))) + noise * 0.26 + twins.astype(np.float32) * 0.12)
    band_field = normalize01(deformation_bands.astype(np.float32) * 0.72 + noise * 0.28)
    ordered_fields = []
    if "BETA_PRIME" in phases:
        ordered_fields.append(("BETA_PRIME", beta_prime_field))
    if "BETA" in phases:
        ordered_fields.append(("BETA", beta_field))
    if "DEFORMATION_BANDS" in phases:
        ordered_fields.append(("DEFORMATION_BANDS", band_field))
    if "ALPHA" in phases or dominant == "ALPHA":
        ordered_fields.append(("ALPHA", alpha_field))
    phase_masks = allocate_phase_masks(size=size, phase_fractions=phases or {"ALPHA": 1.0}, ordered_fields=ordered_fields, remainder_name=dominant)

    alpha_img = 172.0 + (grain["image"].astype(np.float32) - 150.0) * 0.18 + (noise - 0.5) * 10.0
    alpha_img[boundaries] -= 12.0
    alpha_img[twins] -= 16.0
    beta_img = 108.0 + boundary_pref * 8.0 + (noise - 0.5) * 12.0
    beta_prime_img = 92.0 + (noise - 0.5) * 14.0
    band_img = 98.0 + (noise - 0.5) * 10.0
    image = np.full(size, 152.0, dtype=np.float32)
    if "ALPHA" in phase_masks:
        image[phase_masks["ALPHA"] > 0] = alpha_img[phase_masks["ALPHA"] > 0]
    if "BETA" in phase_masks:
        image[phase_masks["BETA"] > 0] = beta_img[phase_masks["BETA"] > 0]
    if "BETA_PRIME" in phase_masks:
        image[phase_masks["BETA_PRIME"] > 0] = beta_prime_img[phase_masks["BETA_PRIME"] > 0]
    if "DEFORMATION_BANDS" in phase_masks:
        image[phase_masks["DEFORMATION_BANDS"] > 0] = band_img[phase_masks["DEFORMATION_BANDS"] > 0]
    image[deformation_bands] -= 6.0
    image = rescale_to_u8(image, lo=55.0, hi=198.0)

    beta_key = "BETA_PRIME" if "BETA_PRIME" in phase_masks and np.any(phase_masks["BETA_PRIME"] > 0) else ("BETA" if "BETA" in phase_masks else "")
    beta_boundary_bias = float(boundary_pref[phase_masks[beta_key] > 0].mean()) if beta_key else 0.0
    morphology = {
        "stage_family": "brass_alpha_beta",
        "alpha_twins_density": float(twins.mean()),
        "recrystallized_fraction": float(morphology_state.get("recrystallized_fraction", 0.0)),
        "recovery_fraction": float(morphology_state.get("recovery_fraction", 0.0)),
        "ordering_factor": float(morphology_state.get("ordering_factor", 0.0)),
        "beta_boundary_bias": float(beta_boundary_bias),
        "deformation_band_density": float(deformation_bands.mean()),
        "grain_size_px": float(grain_size),
        "cooling_index": float(cool_idx),
    }
    return _finalize(ctx=ctx, image_gray=image, phase_masks=phase_masks, stage=stage, morphology=morphology)
