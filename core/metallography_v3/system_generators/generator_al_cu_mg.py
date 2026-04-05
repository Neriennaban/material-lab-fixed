from __future__ import annotations

import numpy as np

from core.generator_eutectic import generate_aged_aluminum_structure
from core.generator_grains import generate_grain_structure
from core.metallography_v3.realism_utils import (
    allocate_phase_masks,
    clamp,
    cooling_index,
    distance_to_mask,
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
        system="al-cu-mg",
        composition_wt=ctx.composition_wt,
        mode=str(ctx.composition_sensitivity_mode),
        seed=int(ctx.seed),
        single_phase_compensation=bool(len(phase_masks) <= 1),
    )
    metadata = {
        "system_generator_name": "system_al_cu_mg",
        "resolved_stage": str(stage),
        "phase_transition_state": {"stage": str(stage), "transition_kind": "steady", "liquid_fraction": 0.0, "solid_fraction": 1.0, "thermal_direction": "steady"},
        "composition_effect": composition_effect,
        "phase_visibility_report": visibility,
        **metadata_blocks_from_transformation_state(ctx.transformation_state),
        "engineering_trace": {
            "generation_mode": str(ctx.generation_mode),
            "phase_emphasis_style": str(ctx.phase_emphasis_style),
            "phase_fraction_tolerance_pct": float(ctx.phase_fraction_tolerance_pct),
            "system_generator_name": "system_al_cu_mg",
            "realism_morphology": morphology,
            "physics_guided_realism": bool(ctx.transformation_state),
        },
        "system_generator_extra": {"al_cu_mg_morphology": morphology},
    }
    return SystemGenerationResult(image_gray=ensure_u8(image_gray), phase_masks=phase_masks, metadata=metadata)


def generate_al_cu_mg(ctx: SystemGenerationContext) -> SystemGenerationResult:
    stage = str(ctx.stage).strip().lower()
    size = ctx.size
    phases = {str(k): float(v) for k, v in dict(ctx.phase_fractions).items() if float(v) > 0.0}
    dominant = max(phases.items(), key=lambda item: float(item[1]))[0] if phases else "FCC_A1"
    cu = float(ctx.composition_wt.get("Cu", 0.0))
    mg = float(ctx.composition_wt.get("Mg", 0.0))
    cool_idx = cooling_index(getattr(ctx.processing, "cooling_mode", "equilibrium"))
    morphology_state = dict(ctx.transformation_state.get("morphology_state", {}))
    precipitation_state = dict(ctx.transformation_state.get("precipitation_state", {}))
    age_level = {
        "solutionized": 0.12,
        "quenched": 0.22,
        "natural_aged": 0.42,
        "artificial_aged": 0.72,
        "overaged": 1.00,
    }.get(stage, 0.5)
    age_level = float(max(age_level, precipitation_state.get("peak_strength_fraction", 0.0) * 0.82))
    grain_size = clamp(float(morphology_state.get("grain_size_px", 82.0 - 16.0 * cool_idx + 10.0 * (1.0 - age_level))), 24.0, 112.0)
    grain = generate_grain_structure(
        size=size,
        seed=int(ctx.seed) + 101,
        mean_grain_size_px=grain_size,
        grain_size_jitter=0.18,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    boundaries = np.asarray(grain["boundaries"]).astype(bool)
    dist = distance_to_mask(boundaries)
    boundary_pref = normalize01(np.exp(-dist / clamp(4.8 + 2.5 * age_level, 2.0, 8.5)).astype(np.float32))
    low = low_frequency_field(size=size, seed=int(ctx.seed) + 131, sigma=26.0)
    noise = multiscale_noise(size=size, seed=int(ctx.seed) + 149, scales=((18.0, 0.55), (6.0, 0.28), (1.6, 0.17)))

    precip_scale_px = clamp(float(precipitation_state.get("precipitate_scale_px", 1.1 + 2.9 * age_level + 0.15 * cu + 0.10 * mg)), 1.0, 6.8)
    haze = generate_aged_aluminum_structure(
        size=size,
        seed=int(ctx.seed) + 171,
        precipitate_fraction=clamp(0.025 + 0.085 * age_level, 0.02, 0.18),
        precipitate_scale_px=precip_scale_px,
    )
    haze_mask = np.asarray(haze["phase_mask"]) > 0
    theta_particles = draw_particle_mask(
        size=size,
        seed=int(ctx.seed) + 191,
        fraction_total=clamp(float(phases.get("THETA", 0.05)) * (0.8 if stage != "quenched" else 0.35), 0.006, 0.12),
        radius_range=(1.0, clamp(1.8 + 1.8 * age_level, 1.2, 4.8)),
        angular=True,
        elongation_range=(1.0, 1.5),
        angle_spread_deg=70.0,
    )
    boundary_zone = dist <= clamp(2.0 + 3.2 * age_level, 2.0, 7.0)
    s_particles = draw_particle_mask(
        size=size,
        seed=int(ctx.seed) + 211,
        fraction_total=clamp(float(phases.get("S_PHASE", 0.04)) * (0.5 + 0.7 * age_level), 0.0, 0.10),
        radius_range=(1.0, clamp(2.4 + 1.6 * age_level, 1.2, 5.2)),
        angular=False,
        elongation_range=(1.2, 2.8),
        angle_spread_deg=35.0,
        restrict_to=boundary_zone | (haze_mask & (noise > 0.52)),
    )
    q_particles = draw_particle_mask(
        size=size,
        seed=int(ctx.seed) + 229,
        fraction_total=clamp(float(phases.get("QPHASE", 0.02)) * (0.3 + age_level), 0.0, 0.08),
        radius_range=(1.6, clamp(3.0 + 2.4 * age_level, 2.0, 6.5)),
        angular=True,
        elongation_range=(1.0, 1.8),
        angle_spread_deg=45.0,
        restrict_to=boundary_zone,
    )

    pfz_width_px = clamp(float(precipitation_state.get("pfz_width_px", 1.0 + 5.5 * max(0.0, age_level - 0.55))), 0.0, 7.8)
    pfz_mask = (dist <= pfz_width_px) if age_level >= 0.7 else np.zeros(size, dtype=bool)

    ordered_fields = []
    if "QPHASE" in phases:
        ordered_fields.append(("QPHASE", normalize01(q_particles.astype(np.float32) * 0.82 + boundary_pref * 0.18)))
    if "S_PHASE" in phases:
        ordered_fields.append(("S_PHASE", normalize01(s_particles.astype(np.float32) * 0.64 + boundary_pref * 0.24 + haze_mask.astype(np.float32) * 0.12)))
    if "THETA" in phases:
        ordered_fields.append(("THETA", normalize01(theta_particles.astype(np.float32) * 0.72 + haze_mask.astype(np.float32) * 0.22 + noise * 0.06)))
    if "FCC_A1" in phases or dominant == "FCC_A1":
        ordered_fields.append(("FCC_A1", normalize01((grain["image"].astype(np.float32) / 255.0) * 0.62 + low * 0.38)))
    phase_masks = allocate_phase_masks(size=size, phase_fractions=phases or {"FCC_A1": 1.0}, ordered_fields=ordered_fields, remainder_name=dominant)

    matrix = 176.0 + (grain["image"].astype(np.float32) - 150.0) * 0.15 + (noise - 0.5) * 10.0
    matrix -= haze_mask.astype(np.float32) * (6.0 + 8.0 * age_level)
    if age_level >= 0.7:
        matrix[pfz_mask] += 5.0 + 4.0 * (age_level - 0.7)
    matrix[boundaries] -= 10.0
    image = matrix.copy()
    if "THETA" in phase_masks:
        image[phase_masks["THETA"] > 0] -= 22.0
    if "S_PHASE" in phase_masks:
        image[phase_masks["S_PHASE"] > 0] -= 26.0
    if "QPHASE" in phase_masks:
        image[phase_masks["QPHASE"] > 0] -= 34.0
    image = rescale_to_u8(image, lo=60.0, hi=200.0)

    boundary_prec_fraction = 0.0
    for key in ("S_PHASE", "QPHASE"):
        if key in phase_masks and np.any(phase_masks[key] > 0):
            boundary_prec_fraction += float((phase_masks[key] > 0)[boundary_zone].mean())
    morphology = {
        "stage_family": "al_cu_mg_ageing",
        "precipitate_scale_px": float(precip_scale_px),
        "boundary_precipitation_fraction": float(boundary_prec_fraction),
        "pfz_width_px": float(pfz_width_px),
        "aging_response_level": float(age_level),
        "peak_strength_fraction": float(precipitation_state.get("peak_strength_fraction", 0.0)),
        "precipitation_sequence": dict(precipitation_state.get("precipitation_sequence", {})) if isinstance(precipitation_state.get("precipitation_sequence", {}), dict) else {},
        "grain_size_px": float(grain_size),
        "cooling_index": float(cool_idx),
    }
    return _finalize(ctx=ctx, image_gray=image, phase_masks=phase_masks, stage=stage, morphology=morphology)
