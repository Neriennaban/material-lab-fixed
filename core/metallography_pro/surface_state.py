from __future__ import annotations

from typing import Any

import numpy as np

from core.contracts_v3 import EtchProfileV3, SamplePrepRouteV3
from core.metallography_v3.etch_simulator import (
    ETCH_PRESETS,
    _phase_selectivity,
    _resolve_concentration,
)
from core.metallography_v3.prep_simulator import (
    _draw_scratch_layer,
    _phase_property_maps,
)
from core.metallography_v3.realism_utils import (
    clamp,
    low_frequency_field,
    multiscale_noise,
    normalize01,
)

from .contracts import (
    ContinuousTransformationState,
    SpatialMorphologyState,
    SurfaceState,
)

_BASE_REFLECTANCE = {
    "FERRITE": 0.74,
    "PEARLITE": 0.56,
    "CEMENTITE": 0.43,
    "AUSTENITE": 0.66,
    "MARTENSITE": 0.41,
    "MARTENSITE_TETRAGONAL": 0.39,
    "MARTENSITE_CUBIC": 0.43,
    "TROOSTITE": 0.49,
    "SORBITE": 0.54,
    "BAINITE": 0.47,
}


def _resolve_reflectance_base(
    phase_masks: dict[str, np.ndarray], size: tuple[int, int]
) -> np.ndarray:
    base = np.full(size, 0.58, dtype=np.float32)
    for phase_name, mask in phase_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        zone = mask > 0
        if np.any(zone):
            base[zone] = float(_BASE_REFLECTANCE.get(str(phase_name).upper(), 0.58))
    return base


def _rescale_u8(field: np.ndarray) -> np.ndarray:
    return np.clip(normalize01(field.astype(np.float32)) * 255.0, 0.0, 255.0).astype(
        np.uint8
    )


def _mean_orientation_deg(
    orientation_rad: np.ndarray, mask: np.ndarray | None
) -> float | None:
    if mask is None or not isinstance(mask, np.ndarray):
        return None
    zone = mask > 0
    if not np.any(zone):
        return None
    angles = orientation_rad[zone].astype(np.float64)
    if angles.size <= 0:
        return None
    # Orientation is axial, not directional.
    mean_sin = float(np.mean(np.sin(2.0 * angles)))
    mean_cos = float(np.mean(np.cos(2.0 * angles)))
    return float((0.5 * np.arctan2(mean_sin, mean_cos)) * 180.0 / np.pi)


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


def _is_pure_iron_like(
    *,
    system: str | None,
    composition_wt: dict[str, float] | None,
    phase_masks: dict[str, np.ndarray] | None,
) -> bool:
    sys_name = str(system or "").strip().lower()
    if sys_name not in {"fe-c", "fe-si", "system_fe_si"}:
        return False
    fe_pct = _composition_fraction(composition_wt, "Fe")
    c_pct = _composition_fraction(composition_wt, "C")
    si_pct = _composition_fraction(composition_wt, "Si")
    ferritic_cov = 0.0
    dark_cov = 0.0
    if isinstance(phase_masks, dict):
        ferritic_cov += float(
            (phase_masks.get("FERRITE", np.zeros((1, 1), dtype=np.uint8)) > 0).mean()
        )
        ferritic_cov += float(
            (
                phase_masks.get("DELTA_FERRITE", np.zeros((1, 1), dtype=np.uint8)) > 0
            ).mean()
        )
        for name in (
            "PEARLITE",
            "CEMENTITE",
            "MARTENSITE",
            "BAINITE",
            "TROOSTITE",
            "SORBITE",
        ):
            dark_cov += float(
                (phase_masks.get(name, np.zeros((1, 1), dtype=np.uint8)) > 0).mean()
            )
    return bool(
        fe_pct >= 99.8
        and c_pct <= 0.03
        and si_pct <= 0.25
        and ferritic_cov >= 0.92
        and dark_cov <= 0.08
    )


def build_surface_state(
    *,
    morphology_state: SpatialMorphologyState,
    transformation_state: ContinuousTransformationState,
    prep_route: SamplePrepRouteV3,
    etch_profile: EtchProfileV3,
    seed: int,
    native_um_per_px: float,
    system: str | None = None,
    composition_wt: dict[str, float] | None = None,
    artifact_level: float = 0.35,
) -> tuple[
    SurfaceState,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
]:
    size = morphology_state.phase_label_map.shape
    rng = np.random.default_rng(int(seed))
    resolved_system = str(system or "fe-c")
    artifact_level = float(clamp(float(artifact_level), 0.0, 1.0))
    artifact_scale = float(0.35 + 1.15 * artifact_level)
    hardness_map, brittleness_map, phase_coverage = _phase_property_maps(
        shape=size, phase_masks=morphology_state.phase_masks, system=resolved_system
    )
    phase_coupling_applied = bool(np.any(phase_coverage > 0))
    rough_target = float(max(0.01, prep_route.roughness_target_um))
    pure_iron_like = _is_pure_iron_like(
        system=resolved_system,
        composition_wt=composition_wt,
        phase_masks=morphology_state.phase_masks,
    )
    step_count = int(len(prep_route.steps or []))

    topo_gain = 1.6 * artifact_scale
    if pure_iron_like:
        topo_gain *= 0.42 if step_count <= 0 else 0.58
    base_topography = (
        (
            multiscale_noise(
                size=size,
                seed=seed + 11,
                scales=((34.0, 0.60), (10.0, 0.25), (3.0, 0.15)),
            )
            - 0.5
        )
        * rough_target
        * topo_gain
    )
    if morphology_state.lamella_field is not None:
        base_topography += (morphology_state.lamella_field - 0.5) * (
            0.12 + 0.08 * float(transformation_state.pearlite_fraction)
        )
    if morphology_state.packet_field is not None:
        base_topography += (morphology_state.packet_field - 0.5) * (
            0.08 + 0.04 * float(transformation_state.martensite_fraction)
        )

    damage_layer = np.zeros(size, dtype=np.float32)
    scratch_map = np.zeros(size, dtype=np.float32)
    smear_map = np.zeros(size, dtype=np.float32)
    pullout_map = np.zeros(size, dtype=np.float32)
    contamination_base = float(max(0.0, prep_route.contamination_level))
    if pure_iron_like:
        contamination_base *= 0.22 if step_count <= 0 else 0.45
    contamination_map = np.full(size, contamination_base, dtype=np.float32)
    prep_timeline: list[dict[str, Any]] = []
    running_roughness = float(max(rough_target * 3.0, rough_target))
    damage_depth_um = 0.0
    removed_depth_um = 0.0
    grinding_heat_risk = 0.0
    scratch_direction_components: list[tuple[float, float, float]] = []

    for idx, step in enumerate(prep_route.steps):
        duration = float(max(0.0, step.duration_s))
        abrasive = float(step.abrasive_um or 0.0)
        load = float(step.load_n or 0.0)
        rpm = float(step.rpm or 0.0)
        coolant = str(getattr(step, "coolant", "") or "").strip().lower()
        if str(step.method).lower().startswith("grinding"):
            direction_deg = float(getattr(step, "direction_deg", 20.0))
            density = 0.75 + 0.003 * duration + 0.03 * abrasive + 0.0018 * load
            scratches = (
                _draw_scratch_layer(
                    size=size,
                    seed=seed + idx * 31 + 101,
                    density=density,
                    angle_deg=direction_deg,
                    width_px=max(1, int(round(1 + abrasive / 8.0))),
                    oscillation_hz=float(getattr(step, "oscillation_hz", 0.0)),
                ).astype(np.float32)
                / 255.0
            )
            soft_response = 0.30 + 0.70 * (1.0 - hardness_map)
            scratch_map += scratches * (0.55 + 0.45 * soft_response)
            damage_layer += scratches * (0.05 + 0.0009 * load)
            pullout_map += scratches * 0.06 * (0.20 + 0.80 * brittleness_map)
            running_roughness *= 0.82
            damage_depth_um += max(
                0.0, 0.0025 * abrasive + 0.0004 * duration + 0.00015 * load
            )
            coolant_factor = (
                0.45
                if coolant in {"water", "alcohol"}
                else (0.75 if coolant in {"oil", "electrolyte"} else 1.0)
            )
            grinding_heat_risk = max(
                grinding_heat_risk,
                float(
                    clamp(
                        (
                            (rpm / 260.0) * 0.42
                            + (load / 30.0) * 0.34
                            + (abrasive / 18.0) * 0.24
                        )
                        * coolant_factor,
                        0.0,
                        1.0,
                    )
                ),
            )
            scratch_weight = float(max(1e-6, density))
            theta = np.deg2rad(direction_deg)
            scratch_direction_components.append(
                (np.cos(2.0 * theta), np.sin(2.0 * theta), scratch_weight)
            )
        elif str(step.method).lower().startswith("polishing"):
            smear_gain = (0.015 + 0.0008 * duration + 0.0004 * rpm) * (
                0.25 + 0.75 * (1.0 - hardness_map)
            )
            smear_map += smear_gain.astype(np.float32)
            scratch_map *= 0.86
            pit_field = multiscale_noise(
                size=size, seed=seed + idx * 31 + 151, scales=((6.0, 0.45), (1.4, 0.55))
            )
            pit_mask = (
                pit_field
                > np.quantile(
                    pit_field, clamp(0.985 - 0.01 * duration / 60.0, 0.93, 0.99)
                )
            ).astype(np.float32)
            pullout_map += (
                pit_mask * (0.03 + 0.0005 * load) * (0.10 + 0.90 * brittleness_map)
            )
            running_roughness *= 0.66
            removed_depth_um += max(
                0.0, 0.0009 * duration + 0.0035 * max(abrasive, 0.5)
            )
        elif str(step.method).lower() in {
            "electropolish",
            "electropolish_bath",
            "local_electropolish_tampon",
            "local_electropolish_flow_cell",
            "electromechanical_polish",
        }:
            scratch_map *= 0.72
            smear_map *= 0.62
            damage_layer *= 0.76
            running_roughness *= 0.60
            removed_depth_um += max(0.0, 0.0012 * duration + 0.025)
        contamination_map += (
            (0.002 + 0.00002 * duration)
            * (0.30 + 0.70 * brittleness_map)
            * artifact_scale
        )
        if bool(getattr(step, "cleaning_between_steps", False)):
            contamination_map *= 0.90
            smear_map *= 0.95
        prep_timeline.append(
            {
                "step_index": int(idx),
                "method": str(step.method),
                "duration_s": float(duration),
                "abrasive_um": step.abrasive_um,
                "load_n": step.load_n,
                "rpm": step.rpm,
                "roughness_after_um": float(max(rough_target, running_roughness)),
                "damage_depth_um_added": float(damage_depth_um),
                "removed_depth_um_total": float(removed_depth_um),
            }
        )

    scratch_map = np.clip(scratch_map, 0.0, 1.0)
    damage_layer = np.clip(damage_layer, 0.0, 1.0)
    smear_map = np.clip(smear_map, 0.0, 1.0)
    pullout_map = np.clip(pullout_map, 0.0, 1.0)
    contamination_jitter = 0.01 * artifact_scale
    if pure_iron_like:
        contamination_jitter *= 0.35
    contamination_map = np.clip(
        contamination_map
        + rng.normal(0.0, contamination_jitter, size=size).astype(np.float32),
        0.0,
        1.0,
    )
    roughness_um = (
        np.abs(base_topography).astype(np.float32)
        + scratch_map * rough_target * 0.45 * artifact_scale
    )

    concentration = _resolve_concentration(etch_profile)
    preset = ETCH_PRESETS.get(
        str(etch_profile.reagent or "custom").strip().lower(), ETCH_PRESETS["custom"]
    )
    time_factor = float(max(0.1, etch_profile.time_s / 8.0))
    temp_factor = float(
        max(0.5, min(1.8, (etch_profile.temperature_c + 273.15) / (22.0 + 273.15)))
    )
    wt_scale = clamp(float(concentration["wt_pct"]) / 2.0, 0.4, 2.2)
    base_rate = float(
        preset["base"]
        * time_factor
        * temp_factor
        * wt_scale
        * float(max(0.5, min(2.2, etch_profile.overetch_factor)))
    )
    etch_depth_um = np.full(size, base_rate * 0.06, dtype=np.float32)
    selectivity_field = np.zeros(size, dtype=np.float32)
    for phase_name, mask in morphology_state.phase_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        zone = mask > 0
        if np.any(zone):
            selectivity = float(
                _phase_selectivity(
                    phase_name=phase_name,
                    reagent=etch_profile.reagent,
                    system=resolved_system,
                )
            )
            selectivity_field[zone] = selectivity
            etch_depth_um[zone] += (selectivity - 0.95) * 0.05
    etch_depth_um += damage_layer * 0.05 + scratch_map * 0.02 + pullout_map * 0.05
    etch_depth_um -= smear_map * 0.018 + contamination_map * 0.012
    low_etch = low_frequency_field(size, seed + 211, sigma=18.0)
    low_etch_gain = 0.015 * artifact_scale
    if pure_iron_like:
        low_etch_gain *= 0.28 if step_count <= 0 else 0.45
    etch_depth_um += (low_etch - 0.5) * low_etch_gain
    etch_depth_um = np.clip(etch_depth_um, 0.001, None).astype(np.float32)

    stain_map = normalize01(
        contamination_map * 0.50
        + smear_map * 0.22
        + pullout_map * 0.12
        + low_etch * 0.16
    )
    if pure_iron_like:
        stain_map = np.clip(stain_map * (0.16 if step_count <= 0 else 0.28), 0.0, 1.0)
        etch_mean = float(etch_depth_um.mean())
        etch_depth_um = (
            etch_mean
            + (etch_depth_um - etch_mean) * (0.22 if step_count <= 0 else 0.35)
        ).astype(np.float32)
        relief_coupling = 0.14 if step_count <= 0 else 0.22
        height_um = (
            base_topography.astype(np.float32)
            + (etch_depth_um - float(etch_depth_um.mean())) * relief_coupling
            - pullout_map * 0.02
        )
    else:
        stain_map = np.clip(stain_map * (0.78 + 0.55 * artifact_level), 0.0, 1.0)
        height_um = (
            base_topography.astype(np.float32)
            + (etch_depth_um - float(etch_depth_um.mean())) * 0.45
            - pullout_map * 0.06
        )
    artifact_layer_remaining_um = float(max(0.0, damage_depth_um - removed_depth_um))
    dominant_scratch_direction_deg = None
    if scratch_direction_components:
        cos_acc = sum(x * w for x, _, w in scratch_direction_components)
        sin_acc = sum(y * w for _, y, w in scratch_direction_components)
        dominant_scratch_direction_deg = float(
            (0.5 * np.arctan2(sin_acc, cos_acc)) * 180.0 / np.pi
        )
    structural_orientation_deg = (
        _mean_orientation_deg(
            morphology_state.orientation_rad,
            morphology_state.phase_masks.get("PEARLITE"),
        )
        or _mean_orientation_deg(
            morphology_state.orientation_rad,
            morphology_state.phase_masks.get("BAINITE"),
        )
        or _mean_orientation_deg(
            morphology_state.orientation_rad,
            morphology_state.phase_masks.get("FERRITE"),
        )
    )
    directional_artifact_anisotropy_score = 0.0
    if (
        dominant_scratch_direction_deg is not None
        and structural_orientation_deg is not None
    ):
        delta = abs(
            float(dominant_scratch_direction_deg) - float(structural_orientation_deg)
        )
        delta = min(delta, abs(delta - 180.0))
        alignment = 0.5 * (1.0 + np.cos(np.deg2rad(2.0 * delta)))
        directional_artifact_anisotropy_score = float(
            clamp(
                alignment
                * (
                    0.55 * float(scratch_map.mean()) + 0.45 * float(damage_layer.mean())
                ),
                0.0,
                1.0,
            )
        )
    etch_reproducibility_risk = float(
        clamp(
            0.55 * artifact_layer_remaining_um / max(rough_target, 1e-6)
            + 0.25 * float(contamination_map.mean())
            + 0.20 * float(smear_map.mean()),
            0.0,
            1.0,
        )
    )
    scratch_trace_revelation_risk = float(
        clamp(
            float(scratch_map.mean())
            * (0.55 + 0.45 * float(etch_depth_um.mean()) / max(rough_target, 1e-6))
            * (0.70 + 0.30 * float(damage_layer.mean())),
            0.0,
            1.0,
        )
    )
    prep_directionality_banding_risk = float(
        clamp(
            directional_artifact_anisotropy_score
            * (0.60 * float(scratch_map.mean()) + 0.40 * float(damage_layer.mean()))
            * 2.2,
            0.0,
            1.0,
        )
    )
    false_porosity_pullout_risk = float(
        clamp(
            float(pullout_map.mean())
            * (0.60 + 0.40 * float(brittleness_map.mean()))
            * (0.70 + 0.30 * float(contamination_map.mean()))
            * 2.4,
            0.0,
            1.0,
        )
    )
    relief_dominance_ratio = float(
        np.ptp(height_um) / max(np.ptp(etch_depth_um) + rough_target, 1e-6)
    )
    relief_dominance_risk = float(clamp((relief_dominance_ratio - 1.0) / 3.0, 0.0, 1.0))
    stain_deposit_contrast_dominance_risk = float(
        clamp(
            float(stain_map.mean())
            * (
                0.45
                + 0.35 * float(contamination_map.mean())
                + 0.20 * float(smear_map.mean())
            )
            * (1.0 + 0.50 * etch_reproducibility_risk)
            * 2.2,
            0.0,
            1.0,
        )
    )
    pure_iron_cleanliness_score = 0.0
    pure_iron_dark_defect_suppression = 0.0
    pure_iron_boundary_visibility_score = 0.0
    if pure_iron_like:
        pure_iron_cleanliness_score = float(
            clamp(
                1.0
                - (
                    0.42 * float(stain_map.mean())
                    + 0.22 * float(contamination_map.mean())
                    + 0.18 * float(scratch_map.mean())
                    + 0.18 * float(pullout_map.mean())
                ),
                0.0,
                1.0,
            )
        )
        pure_iron_dark_defect_suppression = float(
            clamp(
                1.0
                - max(
                    stain_deposit_contrast_dominance_risk,
                    scratch_trace_revelation_risk,
                    false_porosity_pullout_risk,
                ),
                0.0,
                1.0,
            )
        )
        pure_iron_boundary_visibility_score = float(
            clamp(
                0.55 * (1.0 - abs(float(normalize01(height_um).mean()) - 0.5) * 2.0)
                + 0.45 * (1.0 - float(stain_map.mean())),
                0.0,
                1.0,
            )
        )

    reflectance_base = _resolve_reflectance_base(
        morphology_state.phase_masks, size
    ).astype(np.float32)
    if pure_iron_like:
        reflectance_base = np.maximum(reflectance_base, 0.84)
    surface = SurfaceState(
        height_um=height_um.astype(np.float32),
        etch_depth_um=etch_depth_um.astype(np.float32),
        reflectance_base=reflectance_base,
        damage_layer=damage_layer.astype(np.float32),
        smear_map=smear_map.astype(np.float32),
        pullout_map=pullout_map.astype(np.float32),
        contamination_map=contamination_map.astype(np.float32),
        stain_map=stain_map.astype(np.float32),
        roughness_um=roughness_um.astype(np.float32),
        summary={
            "backend": "pro_explicit_surface_v1",
            "native_um_per_px": float(native_um_per_px),
            "roughness_target_um": float(rough_target),
            "roughness_achieved_um": float(max(rough_target, running_roughness)),
            "phase_coupling_applied": bool(phase_coupling_applied),
            "mean_height_um": float(height_um.mean()),
            "surface_relief_range_um": float(np.ptp(height_um)),
            "etch_depth_mean_um": float(etch_depth_um.mean()),
            "etch_depth_range_um": float(np.ptp(etch_depth_um)),
            "damage_depth_um": float(damage_depth_um),
            "removed_depth_um": float(removed_depth_um),
            "artifact_layer_remaining_um": float(artifact_layer_remaining_um),
            "dominant_scratch_direction_deg": dominant_scratch_direction_deg,
            "structural_orientation_deg": structural_orientation_deg,
            "directional_artifact_anisotropy_score": float(
                directional_artifact_anisotropy_score
            ),
            "etch_reproducibility_risk": float(etch_reproducibility_risk),
            "scratch_trace_revelation_risk": float(scratch_trace_revelation_risk),
            "prep_directionality_banding_risk": float(prep_directionality_banding_risk),
            "false_porosity_pullout_risk": float(false_porosity_pullout_risk),
            "relief_dominance_risk": float(relief_dominance_risk),
            "stain_deposit_contrast_dominance_risk": float(
                stain_deposit_contrast_dominance_risk
            ),
            "grinding_heat_risk": float(grinding_heat_risk),
            "pearlite_fragmentation_risk": float(
                max(0.0, transformation_state.pearlite_fraction)
                * (0.65 * float(scratch_map.mean()) + 0.35 * float(pullout_map.mean()))
            ),
            "pearlite_lamella_integrity_mean": float(
                1.0 - np.clip(damage_layer * 0.75 + pullout_map * 0.25, 0.0, 1.0).mean()
            ),
            "pure_iron_baseline_applied": bool(pure_iron_like),
            "pure_iron_cleanliness_score": float(pure_iron_cleanliness_score),
            "pure_iron_dark_defect_suppression": float(
                pure_iron_dark_defect_suppression
            ),
            "pure_iron_boundary_visibility_score": float(
                pure_iron_boundary_visibility_score
            ),
        },
    )

    prep_maps = {
        "topography": _rescale_u8(height_um),
        "scratch": _rescale_u8(scratch_map),
        "deformation_layer": _rescale_u8(damage_layer),
        "smear": _rescale_u8(smear_map),
        "contamination": _rescale_u8(contamination_map),
        "pullout": _rescale_u8(pullout_map),
        "relief": _rescale_u8(height_um),
        "hardness_proxy": _rescale_u8(hardness_map),
        "brittleness_proxy": _rescale_u8(brittleness_map),
    }
    etch_maps = {
        "etch_rate": _rescale_u8(etch_depth_um),
        "stain": _rescale_u8(stain_map),
        "relief_shading": _rescale_u8(height_um),
        "selectivity": _rescale_u8(selectivity_field),
    }
    prep_summary = {
        "roughness_target_um": float(rough_target),
        "roughness_achieved_um": float(max(rough_target, running_roughness)),
        "relief_mode": str(prep_route.relief_mode),
        "contamination_level": float(prep_route.contamination_level),
        "contamination_mean": float(contamination_map.mean()),
        "step_count": int(len(prep_route.steps)),
        "phase_coupling_applied": bool(phase_coupling_applied),
        "relief_mean": float(normalize01(height_um).mean()),
        "surface_relief_range_um": float(np.ptp(height_um)),
        "scratch_mean": float(scratch_map.mean()),
        "smear_mean": float(smear_map.mean()),
        "pullout_mean": float(pullout_map.mean()),
        "pure_iron_baseline_applied": bool(pure_iron_like),
        "pure_iron_cleanliness_score": float(pure_iron_cleanliness_score),
        "pure_iron_dark_defect_suppression": float(pure_iron_dark_defect_suppression),
        "pure_iron_boundary_visibility_score": float(
            pure_iron_boundary_visibility_score
        ),
    }
    etch_summary = {
        "reagent": str(etch_profile.reagent),
        "time_s": float(etch_profile.time_s),
        "temperature_c": float(etch_profile.temperature_c),
        "agitation": str(etch_profile.agitation),
        "overetch_factor": float(etch_profile.overetch_factor),
        "etch_rate_mean": float(etch_depth_um.mean()),
        "etch_rate_std": float(etch_depth_um.std()),
        "concentration": concentration,
        "phase_selectivity_mode": f"{resolved_system}:{str(etch_profile.reagent).lower()}",
        "prep_coupling_applied": True,
        "stain_level_mean": float(stain_map.mean()),
        "relief_shading_mean": float(normalize01(height_um).mean()),
        "pure_iron_baseline_applied": bool(pure_iron_like),
        "pure_iron_cleanliness_score": float(pure_iron_cleanliness_score),
        "pure_iron_dark_defect_suppression": float(pure_iron_dark_defect_suppression),
        "pure_iron_boundary_visibility_score": float(
            pure_iron_boundary_visibility_score
        ),
    }
    return (
        surface,
        prep_maps,
        prep_timeline,
        prep_summary,
        etch_summary,
        concentration,
        etch_maps,
    )
