from __future__ import annotations

import numpy as np

from core.generator_dendritic import generate_dendritic_cast
from core.generator_eutectic import generate_aged_aluminum_structure, generate_eutectic_al_si
from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    allocate_phase_masks,
    clamp,
    cooling_index,
    draw_particle_mask,
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
        system="al-si",
        composition_wt=ctx.composition_wt,
        mode=str(ctx.composition_sensitivity_mode),
        seed=int(ctx.seed),
        single_phase_compensation=bool(len(phase_masks) <= 1),
    )
    metadata = {
        "system_generator_name": "system_al_si",
        "resolved_stage": str(stage),
        "phase_transition_state": {"stage": str(stage), "transition_kind": "steady", "liquid_fraction": 0.0, "solid_fraction": 1.0, "thermal_direction": "steady"},
        "composition_effect": composition_effect,
        "phase_visibility_report": visibility,
        **metadata_blocks_from_transformation_state(ctx.transformation_state),
        "engineering_trace": {
            "generation_mode": str(ctx.generation_mode),
            "phase_emphasis_style": str(ctx.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(ctx.phase_fraction_tolerance_pct),
            "system_generator_name": "system_al_si",
            "realism_morphology": morphology,
            "applied_realism_heuristics": {
                "dendrite_eutectic_coupling": bool(stage in {"alpha_eutectic", "eutectic", "primary_si_eutectic"}),
                "aging_precipitate_haze": bool(stage in {"supersaturated", "aged"}),
            },
            "physics_guided_realism": bool(ctx.transformation_state),
        },
        "system_generator_extra": {"al_si_morphology": morphology},
    }
    return SystemGenerationResult(image_gray=ensure_u8(image_gray), phase_masks=phase_masks, metadata=metadata)


def generate_al_si(ctx: SystemGenerationContext) -> SystemGenerationResult:
    stage = str(ctx.stage).strip().lower()
    if stage in {"liquid", "liquid_alpha", "liquid_si"}:
        return run_phase_map_system(ctx=ctx, system_name="al-si", generator_name="system_al_si")

    size = ctx.size
    phases = {str(k): float(v) for k, v in dict(ctx.phase_fractions).items() if float(v) > 0.0}
    si_wt = float(ctx.composition_wt.get("Si", 0.0))
    cool_idx = cooling_index(getattr(ctx.processing, "cooling_mode", "equilibrium"))
    transformation_trace = dict(ctx.transformation_state.get("transformation_trace", {}))
    morphology_state = dict(ctx.transformation_state.get("morphology_state", {}))
    low = low_frequency_field(size=size, seed=int(ctx.seed) + 61, sigma=24.0)
    noise = multiscale_noise(size=size, seed=int(ctx.seed) + 71, scales=((18.0, 0.58), (6.0, 0.26), (1.8, 0.16)))

    if stage in {"alpha_eutectic", "eutectic", "primary_si_eutectic"}:
        arm_spacing = clamp(float(morphology_state.get("sdas_px", 56.0 - 22.0 * cool_idx - max(0.0, si_wt - 12.6) * 0.65)), 18.0, 78.0)
        eutectic_scale = clamp(float(morphology_state.get("eutectic_scale_px", 8.8 - 3.8 * cool_idx)), 2.4, 9.5)
        modifier = str(morphology_state.get("eutectic_si_modifier", ""))
        eut_style = "fibrous" if "modified" in modifier else ("needle" if cool_idx < 0.35 else ("branched" if cool_idx < 0.8 else "network"))
        dend = generate_dendritic_cast(
            size=size,
            seed=int(ctx.seed) + 101,
            cooling_rate=18.0 + 90.0 * cool_idx,
            primary_arm_spacing=arm_spacing,
            secondary_arm_factor=0.35 + 0.12 * cool_idx,
            interdendritic_fraction=clamp(float(phases.get("EUTECTIC_ALSI", 0.36)) + float(phases.get("SI", 0.08)) * 0.55, 0.18, 0.72),
            porosity_fraction=0.0,
            gradient_angle_deg=18.0 + 28.0 * (1.0 - cool_idx),
        )
        eut = generate_eutectic_al_si(
            size=size,
            seed=int(ctx.seed) + 137,
            si_phase_fraction=clamp(float(phases.get("SI", 0.12 if stage != "alpha_eutectic" else 0.08)), 0.06, 0.55),
            eutectic_scale_px=eutectic_scale,
            morphology=eut_style,
        )
        dend_core = np.asarray(dend["phase_masks"]["dendrite_core"]) > 0
        inter = np.asarray(dend["phase_masks"]["interdendritic"]) > 0
        eut_si = np.asarray(eut["phase_mask"]) > 0
        primary_si = np.zeros(size, dtype=bool)
        if stage == "primary_si_eutectic" or si_wt >= 13.0 or "SI" in phases:
            primary_si_size_px = clamp(float(morphology_state.get("primary_si_size_px", 8.5 + max(0.0, si_wt - 12.6) * 0.6)), 4.0, 15.0)
            primary_si = draw_particle_mask(
                size=size,
                seed=int(ctx.seed) + 155,
                fraction_total=clamp(max(float(phases.get("SI", 0.12)) * 0.72, float(transformation_trace.get("primary_si_count_proxy", 0.0)) * 0.22), 0.02, 0.20),
                radius_range=(3.4, primary_si_size_px),
                angular=True,
                elongation_range=(1.0, 1.45),
                angle_spread_deg=45.0,
            )
        si_field = normalize01(primary_si.astype(np.float32) * 0.72 + (eut_si & inter).astype(np.float32) * 0.28 + noise * 0.06)
        alpha_field = normalize01(dend_core.astype(np.float32) * 0.74 + low * 0.18 + noise * 0.08)
        eut_field = normalize01(inter.astype(np.float32) * 0.72 + (eut_si.astype(np.float32) * 0.22) + low * 0.06)
        dominant = max(phases.items(), key=lambda item: float(item[1]))[0] if phases else "FCC_A1"
        ordered_fields = []
        if "SI" in phases:
            ordered_fields.append(("SI", si_field))
        if "FCC_A1" in phases:
            ordered_fields.append(("FCC_A1", alpha_field))
        if "EUTECTIC_ALSI" in phases:
            ordered_fields.append(("EUTECTIC_ALSI", eut_field))
        phase_masks = allocate_phase_masks(size=size, phase_fractions=phases or {"FCC_A1": 1.0}, ordered_fields=ordered_fields, remainder_name=dominant)

        alpha_img = 164.0 + (dend["image"].astype(np.float32) - 132.0) * 0.45 + (noise - 0.5) * 9.0
        eut_img = 132.0 + (eut["image"].astype(np.float32) - 128.0) * 0.68 + inter.astype(np.float32) * 6.0
        si_img = 66.0 + (noise - 0.5) * 13.0 - primary_si.astype(np.float32) * 18.0
        image = np.full(size, 148.0, dtype=np.float32)
        if "FCC_A1" in phase_masks:
            mask = phase_masks["FCC_A1"] > 0
            image[mask] = alpha_img[mask]
        if "EUTECTIC_ALSI" in phase_masks:
            mask = phase_masks["EUTECTIC_ALSI"] > 0
            image[mask] = eut_img[mask]
        if "SI" in phase_masks:
            mask = phase_masks["SI"] > 0
            image[mask] = si_img[mask]
        image[inter & ~eut_si] -= 4.0
        image = rescale_to_u8(image, lo=38.0, hi=205.0)
        morphology = {
            "stage_family": "cast_al_si",
            "sdas_px": float(arm_spacing),
            "dendrite_arm_spacing_px": float(arm_spacing),
            "eutectic_scale_px": float(eutectic_scale),
            "eutectic_si_style": str(eut_style),
            "eutectic_si_modifier": str(morphology_state.get("eutectic_si_modifier", "none")),
            "primary_si_size_px": float(morphology_state.get("primary_si_size_px", 0.0)),
            "primary_si_count_proxy": float(transformation_trace.get("primary_si_count_proxy", 0.0)),
            "primary_si_fraction_visual": float(primary_si.mean()),
            "interdendritic_fraction_visual": float(inter.mean()),
            "cooling_index": float(cool_idx),
        }
        return _finalize(ctx=ctx, image_gray=image, phase_masks=phase_masks, stage=stage, morphology=morphology)

    grain_scale = clamp(78.0 - 24.0 * cool_idx + 8.0 * float(ctx.effect_vector.get("grain_size_factor", 0.0)), 28.0, 112.0)
    grain = generate_grain_structure(
        size=size,
        seed=int(ctx.seed) + 201,
        mean_grain_size_px=grain_scale,
        grain_size_jitter=0.18,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    aged = generate_aged_aluminum_structure(
        size=size,
        seed=int(ctx.seed) + 221,
        precipitate_fraction=clamp(0.03 + 0.05 * (1.0 if stage == "aged" else 0.55), 0.02, 0.16),
        precipitate_scale_px=clamp(1.2 + 0.9 * (1.0 if stage == "aged" else 0.6), 0.8, 3.6),
    )
    precip_mask = np.asarray(aged["phase_mask"]) > 0
    theta_particles = draw_particle_mask(
        size=size,
        seed=int(ctx.seed) + 243,
        fraction_total=clamp(float(phases.get("PRECIPITATE", phases.get("SI", 0.05))), 0.01, 0.12),
        radius_range=(1.0, 3.2 if stage == "supersaturated" else 4.5),
        angular=False,
        elongation_range=(1.0, 1.8),
    )
    phase_key = "PRECIPITATE" if "PRECIPITATE" in phases else ("SI" if "SI" in phases else ("THETA" if "THETA" in phases else "FCC_A1"))
    dominant = max(phases.items(), key=lambda item: float(item[1]))[0] if phases else "FCC_A1"
    ordered_fields = []
    if phase_key != "FCC_A1":
        ordered_fields.append((phase_key, normalize01(theta_particles.astype(np.float32) * 0.68 + precip_mask.astype(np.float32) * 0.32 + noise * 0.08)))
    if "FCC_A1" in phases or dominant == "FCC_A1":
        ordered_fields.append(("FCC_A1", normalize01((grain["image"].astype(np.float32) / 255.0) * 0.62 + low * 0.38)))
    phase_masks = allocate_phase_masks(size=size, phase_fractions=phases or {"FCC_A1": 1.0}, ordered_fields=ordered_fields, remainder_name=dominant)

    matrix = 174.0 + (grain["image"].astype(np.float32) - 150.0) * 0.16 + (aged["image"].astype(np.float32) - 175.0) * (0.32 if stage == "aged" else 0.18)
    image = matrix.copy()
    if phase_key in phase_masks:
        image[phase_masks[phase_key] > 0] -= 28.0 if stage == "aged" else 18.0
    image = rescale_to_u8(image, lo=65.0, hi=198.0)
    morphology = {
        "stage_family": "heat_treated_al_si",
        "matrix_grain_size_px": float(grain_scale),
        "precipitate_haze_fraction": float(precip_mask.mean()),
        "constituent_particle_fraction": float(theta_particles.mean()),
        "eutectic_si_modifier": str(morphology_state.get("eutectic_si_modifier", "none")),
        "cooling_index": float(cool_idx),
    }
    return _finalize(ctx=ctx, image_gray=image, phase_masks=phase_masks, stage=stage, morphology=morphology)
