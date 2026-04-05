from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.contracts_v3 import SamplePrepRouteV3
from core.metallography_v3.realism_utils import clamp, multiscale_noise, normalize01, smooth


_PHASE_HARDNESS: dict[str, float] = {
    "FERRITE": 0.28,
    "AUSTENITE": 0.34,
    "PEARLITE": 0.48,
    "CEMENTITE": 0.94,
    "MARTENSITE": 0.86,
    "MARTENSITE_TETRAGONAL": 0.90,
    "MARTENSITE_CUBIC": 0.82,
    "TROOSTITE": 0.66,
    "SORBITE": 0.54,
    "BAINITE": 0.68,
    "SI": 0.98,
    "EUTECTIC_ALSI": 0.62,
    "FCC_A1": 0.20,
    "ALPHA": 0.34,
    "BETA": 0.56,
    "BETA_PRIME": 0.62,
    "THETA": 0.82,
    "S_PHASE": 0.88,
    "QPHASE": 0.92,
    "PRECIPITATE": 0.74,
}

_PHASE_BRITTLENESS: dict[str, float] = {
    "FERRITE": 0.08,
    "AUSTENITE": 0.12,
    "PEARLITE": 0.22,
    "CEMENTITE": 0.96,
    "MARTENSITE": 0.48,
    "MARTENSITE_TETRAGONAL": 0.54,
    "MARTENSITE_CUBIC": 0.42,
    "TROOSTITE": 0.26,
    "SORBITE": 0.16,
    "BAINITE": 0.30,
    "SI": 0.98,
    "EUTECTIC_ALSI": 0.78,
    "FCC_A1": 0.06,
    "ALPHA": 0.12,
    "BETA": 0.30,
    "BETA_PRIME": 0.44,
    "THETA": 0.86,
    "S_PHASE": 0.88,
    "QPHASE": 0.92,
    "PRECIPITATE": 0.72,
}


def _draw_scratch_layer(
    size: tuple[int, int],
    seed: int,
    density: float,
    angle_deg: float,
    width_px: int = 1,
    oscillation_hz: float = 0.0,
) -> np.ndarray:
    h, w = size
    rng = np.random.default_rng(seed)
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    count = int(max(0, density * h * w / 900.0))
    theta = np.deg2rad(float(angle_deg))
    osc = max(0.0, float(oscillation_hz))
    for _ in range(count):
        cx = float(rng.uniform(0, w - 1))
        cy = float(rng.uniform(0, h - 1))
        ln = float(rng.uniform(24, min(w, h) * 0.45))
        theta_local = theta
        if osc > 0.0:
            theta_local = theta + np.deg2rad(rng.normal(0.0, min(25.0, 3.0 * osc)))
        dx = np.cos(theta_local)
        dy = np.sin(theta_local)
        x0 = cx - dx * ln
        y0 = cy - dy * ln
        x1 = cx + dx * ln
        y1 = cy + dy * ln
        tone = int(rng.integers(110, 240))
        draw.line((x0, y0, x1, y1), fill=tone, width=max(1, int(width_px)))
    return np.asarray(canvas, dtype=np.uint8)


def _phase_property_maps(
    *,
    shape: tuple[int, int],
    phase_masks: dict[str, np.ndarray] | None,
    system: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hardness = np.full(shape, 0.45, dtype=np.float32)
    brittleness = np.full(shape, 0.18, dtype=np.float32)
    phase_coverage = np.zeros(shape, dtype=np.float32)
    if not isinstance(phase_masks, dict) or not phase_masks:
        return hardness, brittleness, phase_coverage

    for name, mask in phase_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        zone = (mask > 0).astype(np.float32)
        if zone.max() <= 0.0:
            continue
        key = str(name).upper()
        hard_val = float(_PHASE_HARDNESS.get(key, 0.52 if str(system or "").lower().startswith("fe") else 0.38))
        brittle_val = float(_PHASE_BRITTLENESS.get(key, 0.22))
        hardness = hardness * (1.0 - zone) + hard_val * zone
        brittleness = brittleness * (1.0 - zone) + brittle_val * zone
        phase_coverage = np.maximum(phase_coverage, zone)
    return hardness, brittleness, phase_coverage


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


def _phase_coverage_fraction(phase_masks: dict[str, np.ndarray] | None, candidates: set[str]) -> float:
    if not isinstance(phase_masks, dict):
        return 0.0
    total = 0.0
    for name, mask in phase_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        if str(name).upper() in candidates:
            total += float((mask > 0).mean())
    return float(total)


def _is_pure_iron_like(
    *,
    system: str | None,
    composition_wt: dict[str, float] | None,
    phase_masks: dict[str, np.ndarray] | None,
) -> bool:
    sys_name = str(system or "").strip().lower()
    if sys_name not in {"fe-si", "fe-c", "system_fe_si"}:
        return False
    fe_pct = _composition_fraction(composition_wt, "Fe")
    c_pct = _composition_fraction(composition_wt, "C")
    si_pct = _composition_fraction(composition_wt, "Si")
    if isinstance(phase_masks, dict) and set(str(k) for k in phase_masks.keys()) == {"solid"}:
        ferritic_cov = 1.0
        dark_cov = 0.0
    else:
        ferritic_cov = _phase_coverage_fraction(phase_masks, {"BCC_B2", "FERRITE", "DELTA_FERRITE"})
        dark_cov = _phase_coverage_fraction(
            phase_masks,
            {"CEMENTITE", "PEARLITE", "MARTENSITE", "BAINITE", "FESI_INTERMETALLIC", "THETA", "S_PHASE", "QPHASE"},
        )
    return bool(fe_pct >= 99.8 and c_pct <= 0.03 and si_pct <= 0.25 and ferritic_cov >= 0.92 and dark_cov <= 0.08)


def _is_aggressive_prep_route(prep_route: SamplePrepRouteV3) -> bool:
    if float(prep_route.roughness_target_um) > 0.06 or float(prep_route.contamination_level) > 0.03:
        return True
    for step in prep_route.steps:
        method = str(step.method or "").strip().lower()
        abrasive = float(step.abrasive_um or 0.0)
        load = float(step.load_n or 0.0)
        rpm = float(step.rpm or 0.0)
        duration = float(step.duration_s or 0.0)
        cloth = str(getattr(step, "cloth_type", "") or "").strip().lower()
        if method.startswith("grinding") and (abrasive >= 24.0 or load >= 28.0 or rpm >= 240.0 or duration >= 140.0):
            return True
        if method.startswith("polishing") and (cloth == "long_nap" or load >= 18.0 or duration >= 180.0):
            return True
        if method in {"section_fracture", "section_shearing", "section_abrasive_cutoff"}:
            return True
    return False


def _electropolish_area_mask(
    *,
    size: tuple[int, int],
    seed: int,
    spot_diameter_mm: float | None,
    movement_pattern: str,
) -> np.ndarray:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(int(seed) + 4041)
    center_x = float(rng.uniform(0.35, 0.65) * (w - 1))
    center_y = float(rng.uniform(0.35, 0.65) * (h - 1))
    base_radius_px = max(8.0, min(min(h, w) * 0.36, float(max(2.0, spot_diameter_mm or 4.0)) * min(h, w) / 14.0))
    mask = np.zeros((h, w), dtype=np.float32)
    pattern = str(movement_pattern or "none").strip().lower()
    if pattern in {"circular", "spiral"}:
        for shift in np.linspace(-0.35, 0.35, 5):
            rr = np.sqrt((xx - (center_x + shift * base_radius_px)) ** 2 + (yy - (center_y + shift * base_radius_px)) ** 2)
            mask = np.maximum(mask, (rr <= base_radius_px).astype(np.float32))
    elif pattern in {"back_and_forth", "linear"}:
        width_px = max(6.0, base_radius_px * 0.55)
        angle = float(rng.uniform(-20.0, 20.0))
        theta = np.deg2rad(angle)
        x_rot = (xx - center_x) * np.cos(theta) + (yy - center_y) * np.sin(theta)
        y_rot = -(xx - center_x) * np.sin(theta) + (yy - center_y) * np.cos(theta)
        mask = ((np.abs(y_rot) <= width_px) & (np.abs(x_rot) <= base_radius_px * 1.6)).astype(np.float32)
    else:
        rr = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        mask = (rr <= base_radius_px).astype(np.float32)
    return mask.astype(np.float32)


def apply_prep_route(
    *,
    image_gray: np.ndarray,
    prep_route: SamplePrepRouteV3,
    seed: int,
    phase_masks: dict[str, np.ndarray] | None = None,
    system: str | None = None,
    composition_wt: dict[str, float] | None = None,
    effect_vector: dict[str, float] | None = None,
) -> dict[str, Any]:
    arr = image_gray.astype(np.float32).copy()
    h, w = arr.shape
    rng = np.random.default_rng(seed)

    topo = multiscale_noise(size=(h, w), seed=seed + 11, scales=((30.0, 0.58), (10.0, 0.26), (3.0, 0.16)))
    hardness_map, brittleness_map, phase_coverage = _phase_property_maps(shape=(h, w), phase_masks=phase_masks, system=system)
    phase_coupling_applied = bool(np.any(phase_coverage > 0))
    pure_iron_like = _is_pure_iron_like(system=system, composition_wt=composition_wt, phase_masks=phase_masks)
    aggressive_prep = _is_aggressive_prep_route(prep_route)
    effect_vector = dict(effect_vector or {})
    dislocation = float(max(0.0, min(1.0, effect_vector.get("dislocation_proxy", 0.0))))

    deformation_layer = np.zeros((h, w), dtype=np.float32)
    scratch_map = np.zeros((h, w), dtype=np.float32)
    smear_map = np.zeros((h, w), dtype=np.float32)
    pullout_map = np.zeros((h, w), dtype=np.float32)
    contamination = np.full((h, w), float(max(0.0, prep_route.contamination_level)), dtype=np.float32)
    electropolish_local_mask = np.zeros((h, w), dtype=np.float32)

    prep_timeline: list[dict[str, Any]] = []
    running_roughness = float(max(0.02, prep_route.roughness_target_um * 3.5))
    electropolish_mode = "none"
    electropolish_profile_id = ""
    local_area_fraction = 0.0
    electropolish_voltage_v = 0.0
    electropolish_temperature_c = 0.0
    electropolish_phase_relief_risk = 0.0
    electropolish_furrowing_risk = 0.0
    electropolish_pitting_risk = 0.0
    electropolish_edge_effect_risk = 0.0
    electropolish_passivation_risk = 0.0
    post_electropolish_electroetch_used = False
    post_electropolish_chemical_etch_used = False

    phase_relief = normalize01(smooth((hardness_map - float(hardness_map.mean())).astype(np.float32), sigma=1.4)) - 0.5
    pullout_sites = normalize01(
        brittleness_map * 0.58 + hardness_map * 0.26 + multiscale_noise(size=(h, w), seed=seed + 23, scales=((7.0, 0.5), (2.2, 0.5))) * 0.16
    )
    soft_response = 0.42 + 0.58 * (1.0 - hardness_map)
    smear_response = 0.18 + 0.82 * (1.0 - hardness_map) * (1.0 - 0.25 * brittleness_map)
    pullout_response = 0.08 + 0.92 * hardness_map * brittleness_map

    for idx, step in enumerate(prep_route.steps):
        method = str(step.method).strip().lower()
        duration = float(max(0.0, step.duration_s))
        abrasive = float(step.abrasive_um or 0.0)
        load = float(step.load_n or 0.0)
        rpm = float(step.rpm or 0.0)
        direction = float(getattr(step, "direction_deg", 20.0 + idx * 12.0))
        load_profile = str(getattr(step, "load_profile", "constant") or "constant").strip().lower()
        cloth_type = str(getattr(step, "cloth_type", "standard") or "standard").strip().lower()
        slurry_type = str(getattr(step, "slurry_type", "diamond") or "diamond").strip().lower()
        lube_flow = float(getattr(step, "lubricant_flow_ml_min", 0.0) or 0.0)
        clean_between = bool(getattr(step, "cleaning_between_steps", False))
        oscill = float(getattr(step, "oscillation_hz", 0.0) or 0.0)
        path_pattern = str(getattr(step, "path_pattern", "linear") or "linear").strip().lower()

        profile_factor = {"constant": 1.0, "ramp_up": 1.08, "ramp_down": 0.94, "pulse": 1.12}.get(load_profile, 1.0)
        cloth_factor = {"standard": 1.0, "hard": 1.08, "soft": 0.92, "final": 0.85}.get(cloth_type, 1.0)
        if cloth_type == "napless":
            cloth_factor = 0.82
        elif cloth_type == "short_nap":
            cloth_factor = 0.90
        elif cloth_type == "long_nap":
            cloth_factor = 1.18
        elif cloth_type == "rigid_pad":
            cloth_factor = 0.78
        slurry_factor = {"diamond": 1.0, "alumina": 0.95, "silica": 0.9, "custom": 1.0}.get(slurry_type, 1.0)
        if slurry_type == "colloidal_silica":
            slurry_factor = 0.74
        elif slurry_type == "magnesia":
            slurry_factor = 0.80
        elif slurry_type == "diamond_poly":
            slurry_factor = 0.86
        local_load = load * profile_factor

        if method.startswith("grinding"):
            running_roughness *= 0.78
            density = 0.8 + 0.004 * duration + 0.03 * abrasive + 0.002 * local_load
            if path_pattern in {"circular", "figure8", "random"}:
                density *= 1.12
            scratches = _draw_scratch_layer(
                size=(h, w),
                seed=seed + idx * 31 + 101,
                density=density,
                angle_deg=direction,
                width_px=max(1, int(round(1 + abrasive / 9.0))),
                oscillation_hz=oscill,
            ).astype(np.float32) / 255.0
            scratch_map += scratches * (0.55 + 0.45 * soft_response)
            deformation_layer += scratches * (0.05 + 0.0007 * local_load) * (0.65 + 0.35 * soft_response)
            pullout_map += scratches * 0.08 * pullout_response
        elif method.startswith("polishing"):
            running_roughness *= 0.62
            blur_sigma = max(0.35, 1.8 - min(1.2, abrasive / 10.0))
            blur_sigma *= 1.06 if cloth_factor > 1.0 else 0.94
            arr = smooth(arr, sigma=blur_sigma)
            smear_gain = (0.02 + 0.0008 * duration + 0.0005 * max(rpm, 0.0)) * cloth_factor * slurry_factor
            smear_gain *= max(0.85, 1.0 - 0.01 * lube_flow)
            smear_map += smear_gain * smear_response
            scratch_map *= 0.88
            pit_field = multiscale_noise(size=(h, w), seed=seed + idx * 31 + 151, scales=((5.0, 0.45), (1.4, 0.55)))
            pit_mask = (pit_field > np.quantile(pit_field, clamp(0.985 - 0.010 * duration / 60.0, 0.93, 0.99))).astype(np.float32)
            pullout_map += pit_mask * pullout_response * (0.015 + 0.0004 * local_load) * max(0.8, 1.1 - 0.01 * lube_flow)
        elif method in {"electropolish", "electropolish_bath", "local_electropolish_tampon", "local_electropolish_flow_cell", "electromechanical_polish"}:
            electrolyte_code = str(getattr(step, "electrolyte_code", "") or "").strip().lower()
            voltage_v = float(getattr(step, "voltage_v", 0.0) or 0.0)
            current_density = float(getattr(step, "current_density_a_cm2", 0.0) or 0.0)
            electrolyte_temperature_c = float(getattr(step, "electrolyte_temperature_c", 20.0) or 20.0)
            movement_pattern = str(getattr(step, "movement_pattern", "none") or "none").strip().lower()
            local_mask = np.ones((h, w), dtype=np.float32)
            if method in {"local_electropolish_tampon", "local_electropolish_flow_cell"}:
                local_mask = _electropolish_area_mask(
                    size=(h, w),
                    seed=seed + idx * 31 + 909,
                    spot_diameter_mm=getattr(step, "spot_diameter_mm", None),
                    movement_pattern=movement_pattern,
                )
                electropolish_local_mask = np.maximum(electropolish_local_mask, local_mask)
            local_area_fraction = max(local_area_fraction, float(local_mask.mean()))
            running_roughness *= 0.58 if method != "electromechanical_polish" else 0.66
            arr = smooth(arr, sigma=1.1 if method != "electromechanical_polish" else 0.75)
            topo = topo * (1.0 - 0.22 * local_mask) + smooth(topo, sigma=2.4) * (0.22 * local_mask)
            scratch_map *= (1.0 - (0.38 if method != "electromechanical_polish" else 0.22) * local_mask)
            deformation_layer *= (1.0 - 0.25 * local_mask)
            smear_map *= (1.0 - 0.28 * local_mask)
            pullout_map *= (1.0 - 0.24 * local_mask)
            contamination *= (1.0 - 0.12 * local_mask)
            if method == "electromechanical_polish":
                relief_gain = (0.10 + 0.20 * brittleness_map) * local_mask
                pullout_map += relief_gain * 0.04
            if phase_coupling_applied:
                phase_relief = np.abs(hardness_map - float(hardness_map.mean()))
                electropolish_phase_relief_risk = max(
                    electropolish_phase_relief_risk,
                    float(clamp(float((phase_relief * local_mask).mean()) * (0.8 if method != "electromechanical_polish" else 1.35), 0.0, 1.0)),
                )
            voltage_factor = voltage_v / 40.0 if voltage_v > 0.0 else current_density / 0.5 if current_density > 0.0 else 0.4
            cold_bonus = 0.8 if electrolyte_temperature_c <= 10.0 else 1.0
            electropolish_furrowing_risk = max(
                electropolish_furrowing_risk,
                float(clamp((0.35 if movement_pattern in {"back_and_forth", "linear"} else 0.18) * voltage_factor * cold_bonus, 0.0, 1.0)),
            )
            electropolish_pitting_risk = max(
                electropolish_pitting_risk,
                float(clamp((0.22 + 0.30 * max(0.0, voltage_factor - 0.9)) * (1.15 if electrolyte_temperature_c > 30.0 else 1.0), 0.0, 1.0)),
            )
            electropolish_edge_effect_risk = max(
                electropolish_edge_effect_risk,
                float(clamp((1.0 - float(local_mask.mean())) * 0.55 + float(edge_rounding := np.mean(np.abs(np.gradient(local_mask.astype(np.float32), axis=0))) + np.mean(np.abs(np.gradient(local_mask.astype(np.float32), axis=1)))) * 0.15, 0.0, 1.0)),
            )
            electropolish_passivation_risk = max(
                electropolish_passivation_risk,
                float(clamp((0.18 + 0.32 * float(smear_map.mean()) + 0.24 * float(contamination.mean())) * (1.15 if electrolyte_code.startswith("sulfuric") else 1.0), 0.0, 1.0)),
            )
            electropolish_mode = "bath" if method in {"electropolish", "electropolish_bath"} else ("tampon" if method == "local_electropolish_tampon" else ("flow_cell" if method == "local_electropolish_flow_cell" else "electromechanical"))
            electropolish_profile_id = electrolyte_code
            electropolish_voltage_v = max(electropolish_voltage_v, voltage_v)
            electropolish_temperature_c = electrolyte_temperature_c
            post_mode = str(getattr(step, "post_polish_followup", "none") or "none").strip().lower()
            post_electropolish_electroetch_used = post_electropolish_electroetch_used or post_mode == "electrolytic_etch"
            post_electropolish_chemical_etch_used = post_electropolish_chemical_etch_used or post_mode == "chemical_etch"
        elif method in {"cleaning", "mounting"}:
            contamination *= 0.92
            smear_map *= 0.95
            pullout_map *= 0.98

        if clean_between:
            contamination *= 0.88
            scratch_map *= 0.97
            smear_map *= 0.94

        contam_response = 0.25 + 0.55 * pullout_response + 0.20 * scratch_map
        contamination += contam_response * (0.002 + 0.00002 * duration) * max(0.65, 1.0 - 0.02 * lube_flow)

        prep_timeline.append(
            {
                "step_index": idx,
                "method": step.method,
                "duration_s": duration,
                "abrasive_um": step.abrasive_um,
                "load_n": step.load_n,
                "rpm": step.rpm,
                "direction_deg": direction,
                "load_profile": load_profile,
                "cloth_type": cloth_type,
                "slurry_type": slurry_type,
                "lubricant_flow_ml_min": lube_flow,
                "oscillation_hz": oscill,
                "path_pattern": path_pattern,
                "roughness_after_um": float(running_roughness),
            }
        )

    scratch_map = np.clip(scratch_map, 0.0, 1.0)
    deformation_layer = np.clip(deformation_layer + dislocation * 0.04 * scratch_map, 0.0, 1.0)
    smear_map = np.clip(smear_map, 0.0, 1.0)
    pullout_map = np.clip(pullout_map, 0.0, 1.0)
    contamination = np.clip(contamination + rng.normal(0.0, 0.01, size=(h, w)), 0.0, 1.0).astype(np.float32)

    pure_iron_dark_defect_suppression = 0.0
    pure_iron_cleanliness_score = 0.0
    pure_iron_boundary_visibility_score = 0.0
    if pure_iron_like:
        if aggressive_prep:
            scratch_scale = 0.62
            deformation_scale = 0.72
            smear_scale = 0.55
            pullout_scale = 0.35
            contamination_scale = 0.42
            topo_scale = 0.86
        else:
            scratch_scale = 0.20
            deformation_scale = 0.28
            smear_scale = 0.24
            pullout_scale = 0.10
            contamination_scale = 0.16
            topo_scale = 0.72
        scratch_map *= scratch_scale
        deformation_layer *= deformation_scale
        smear_map *= smear_scale
        pullout_map *= pullout_scale
        contamination *= contamination_scale
        topo = (topo - 0.5) * topo_scale + 0.5
        running_roughness *= 0.78 if not aggressive_prep else 0.92
        pure_iron_dark_defect_suppression = float(clamp(1.0 - np.mean([scratch_scale, smear_scale, contamination_scale]), 0.0, 1.0))

    relief_mode = str(prep_route.relief_mode).strip().lower()
    relief_strength = 12.0 if relief_mode == "hardness_coupled" else 15.0
    relief_gain = 8.0 if phase_coupling_applied else 0.0
    if pure_iron_like:
        relief_strength *= 0.72 if not aggressive_prep else 0.88
        relief_gain *= 0.35
    relief_map = normalize01((topo - 0.5) * 0.72 + phase_relief * (0.55 if relief_mode in {"hardness_coupled", "phase_coupled"} else 0.25) + pullout_map * 0.18)

    arr = arr + (relief_map - 0.5) * (relief_strength + relief_gain)
    arr = arr - scratch_map * (18.0 + 4.0 * soft_response)
    arr = arr + smear_map * (4.5 + 3.0 * smear_response)
    arr = arr - pullout_map * (9.0 + 10.0 * pullout_response)
    arr = arr - contamination * (4.0 + 3.0 * pullout_response)
    if pure_iron_like:
        brightness_lift = 12.0 if not aggressive_prep else 6.0
        arr = arr + brightness_lift
        pure_iron_cleanliness_score = float(
            clamp(
                1.0
                - (
                    0.42 * float(scratch_map.mean())
                    + 0.28 * float(contamination.mean())
                    + 0.18 * float(smear_map.mean())
                    + 0.12 * float(pullout_map.mean())
                ),
                0.0,
                1.0,
            )
        )
        pure_iron_boundary_visibility_score = float(
            clamp(
                0.55 * (1.0 - abs(float(relief_map.mean()) - 0.5) * 2.0)
                + 0.45 * (1.0 - float(scratch_map.mean())),
                0.0,
                1.0,
            )
        )
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    sectioning_damage_depth_proxy = float(clamp((0.55 * float(deformation_layer.mean()) + 0.45 * float(running_roughness / max(prep_route.roughness_target_um, 1e-6) - 1.0)) / 2.0, 0.0, 1.0))
    outer_fragmented_layer_risk = float(clamp(0.70 * float(deformation_layer.mean()) + 0.30 * float(scratch_map.mean()), 0.0, 1.0))
    edge_rounding_risk = float(clamp(0.45 * float(running_roughness / max(prep_route.roughness_target_um, 1e-6) - 1.0) + 0.55 * float(smear_map.mean()), 0.0, 1.0))
    false_porosity_from_chipping_risk = float(clamp(float(pullout_map.mean()) * (0.60 + 0.40 * float(brittleness_map.mean())) * 2.1, 0.0, 1.0))
    graphite_retention_risk = float(
        clamp(
            (_phase_coverage_fraction(phase_masks, {"GRAPHITE", "GRAPHITE_FLAKE", "GRAPHITE_SPHEROIDAL", "GRAPHITE_FLAKES"}) > 0.01)
            * (0.58 * float(pullout_map.mean()) + 0.42 * float(smear_map.mean()))
            * (1.35 if any(str(getattr(step, "cloth_type", "")).strip().lower() == "long_nap" for step in prep_route.steps) else 0.75),
            0.0,
            1.0,
        )
    )
    tempering_by_grinding_risk = float(
        clamp(
            (0.25 + 0.75 * _composition_fraction(composition_wt, "C"))
            * max(0.0, min(1.0, max((float(getattr(step, "rpm", 0.0) or 0.0) / 260.0) * (float(getattr(step, "load_n", 0.0) or 0.0) / 30.0) for step in prep_route.steps if str(step.method).lower().startswith("grinding")) if any(str(step.method).lower().startswith("grinding") for step in prep_route.steps) else 0.0))
            * 0.85,
            0.0,
            1.0,
        )
    )

    prep_maps: dict[str, np.ndarray] = {
        "topography": np.clip(topo * 255.0, 0, 255).astype(np.uint8),
        "scratch": np.clip(scratch_map * 255.0, 0, 255).astype(np.uint8),
        "deformation_layer": np.clip(deformation_layer * 255.0, 0, 255).astype(np.uint8),
        "smear": np.clip(smear_map * 255.0, 0, 255).astype(np.uint8),
        "contamination": np.clip(contamination * 255.0, 0, 255).astype(np.uint8),
        "pullout": np.clip(pullout_map * 255.0, 0, 255).astype(np.uint8),
        "relief": np.clip(relief_map * 255.0, 0, 255).astype(np.uint8),
        "hardness_proxy": np.clip(hardness_map * 255.0, 0, 255).astype(np.uint8),
        "brittleness_proxy": np.clip(brittleness_map * 255.0, 0, 255).astype(np.uint8),
        "electropolish_local_mask": np.clip(electropolish_local_mask * 255.0, 0, 255).astype(np.uint8),
    }

    return {
        "image_gray": arr,
        "prep_maps": prep_maps,
        "prep_timeline": prep_timeline,
        "prep_summary": {
            "roughness_target_um": float(prep_route.roughness_target_um),
            "roughness_achieved_um": float(max(0.01, running_roughness)),
            "relief_mode": prep_route.relief_mode,
            "contamination_level": float(prep_route.contamination_level),
            "step_count": int(len(prep_route.steps)),
            "phase_coupling_applied": bool(phase_coupling_applied),
            "pure_iron_baseline_applied": bool(pure_iron_like),
            "pure_iron_cleanliness_score": float(pure_iron_cleanliness_score),
            "pure_iron_dark_defect_suppression": float(pure_iron_dark_defect_suppression),
            "pure_iron_boundary_visibility_score": float(pure_iron_boundary_visibility_score),
            "relief_mean": float(relief_map.mean()),
            "scratch_mean": float(scratch_map.mean()),
            "smear_mean": float(smear_map.mean()),
            "pullout_mean": float(pullout_map.mean()),
            "asm_realism_profile": "pure_iron_clean_ferrite" if pure_iron_like and not aggressive_prep else ("pure_iron_aggressive" if pure_iron_like else "generic_v3"),
            "sectioning_damage_depth_proxy": float(sectioning_damage_depth_proxy),
            "outer_fragmented_layer_risk": float(outer_fragmented_layer_risk),
            "edge_rounding_risk": float(edge_rounding_risk),
            "graphite_retention_risk": float(graphite_retention_risk),
            "false_porosity_from_chipping_risk": float(false_porosity_from_chipping_risk),
            "tempering_by_grinding_risk": float(tempering_by_grinding_risk),
            "electropolish_mode": electropolish_mode,
            "electropolish_profile_id": electropolish_profile_id,
            "local_area_fraction": float(local_area_fraction),
            "voltage_v": float(electropolish_voltage_v),
            "electrolyte_temperature_c": float(electropolish_temperature_c),
            "phase_relief_risk": float(electropolish_phase_relief_risk),
            "furrowing_risk": float(electropolish_furrowing_risk),
            "pitting_risk": float(electropolish_pitting_risk),
            "edge_effect_risk": float(electropolish_edge_effect_risk),
            "passivation_risk": float(electropolish_passivation_risk),
            "post_electropolish_electroetch_used": bool(post_electropolish_electroetch_used),
            "post_electropolish_chemical_etch_used": bool(post_electropolish_chemical_etch_used),
        },
    }
