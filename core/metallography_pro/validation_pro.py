from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from .contracts import (
    ContinuousTransformationState,
    SpatialMorphologyState,
    SurfaceState,
)
from core.measurements import lamellar_spacing_metrics


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _dominant_spacing_px(field: np.ndarray) -> float:
    signal = np.mean(field.astype(np.float32), axis=0)
    signal = signal - float(signal.mean())
    spec = np.abs(np.fft.rfft(signal))
    if spec.size <= 2:
        return 0.0
    peak_idx = int(np.argmax(spec[1:]) + 1)
    if peak_idx <= 0:
        return 0.0
    return float(max(1.0, signal.size / peak_idx))


def _point_count_fraction(mask: np.ndarray, step_px: int = 8) -> float:
    step = max(1, int(step_px))
    sampled = mask[::step, ::step]
    if sampled.size <= 0:
        return 0.0
    return float(np.mean(sampled > 0))


def _lineal_fraction(mask: np.ndarray, step_px: int = 8) -> float:
    step = max(1, int(step_px))
    rows = [float(np.mean(mask[idx, :] > 0)) for idx in range(0, mask.shape[0], step)]
    cols = [float(np.mean(mask[:, idx] > 0)) for idx in range(0, mask.shape[1], step)]
    samples = rows + cols
    if not samples:
        return 0.0
    return float(np.mean(samples))


def _mean_lineal_intercept_um(
    boundary_mask: np.ndarray, pixel_size_um: float
) -> tuple[float, float]:
    if boundary_mask.size <= 0:
        return 0.0, 0.0

    def _axis_intercept(axis: int) -> float:
        total_length_px = 0
        intersections = 0
        if axis == 0:
            iterable = boundary_mask
        else:
            iterable = boundary_mask.T
        for line in iterable:
            line_b = np.asarray(line > 0, dtype=np.uint8)
            total_length_px += max(0, line_b.size - 1)
            intersections += int(
                np.count_nonzero(np.abs(np.diff(line_b.astype(np.int16))))
            )
        if intersections <= 0:
            return float(total_length_px * pixel_size_um)
        return float(total_length_px * pixel_size_um / intersections)

    horizontal = _axis_intercept(0)
    vertical = _axis_intercept(1)
    return horizontal, vertical


def _boundary_length_density_mm_inv(
    boundary_mask: np.ndarray, pixel_size_um: float
) -> float:
    if boundary_mask.size <= 0:
        return 0.0
    mask = boundary_mask > 0
    edges_h = np.count_nonzero(mask[:, 1:] != mask[:, :-1])
    edges_v = np.count_nonzero(mask[1:, :] != mask[:-1, :])
    boundary_length_um = float(edges_h + edges_v) * float(pixel_size_um)
    area_mm2 = float(mask.shape[0] * pixel_size_um / 1000.0) * float(
        mask.shape[1] * pixel_size_um / 1000.0
    )
    if area_mm2 <= 1e-12:
        return 0.0
    return float((boundary_length_um / 1000.0) / area_mm2)


def _pl_from_boundary_mask(
    boundary_mask: np.ndarray, pixel_size_um: float, step_px: int = 8
) -> float:
    step = max(1, int(step_px))
    total_length_um = 0.0
    intersections = 0
    for idx in range(0, boundary_mask.shape[0], step):
        line = boundary_mask[idx, :] > 0
        total_length_um += max(0, line.size - 1) * pixel_size_um
        intersections += int(np.count_nonzero(line))
    for idx in range(0, boundary_mask.shape[1], step):
        line = boundary_mask[:, idx] > 0
        total_length_um += max(0, line.size - 1) * pixel_size_um
        intersections += int(np.count_nonzero(line))
    if total_length_um <= 1e-12:
        return 0.0
    return float(intersections / (total_length_um / 1000.0))


def _intercept_lengths_um(
    boundary_mask: np.ndarray, pixel_size_um: float, step_px: int = 8
) -> list[float]:
    step = max(1, int(step_px))
    lengths: list[float] = []

    def _collect(line: np.ndarray) -> None:
        indices = np.flatnonzero(line > 0)
        if indices.size <= 0:
            lengths.append(float(max(0, line.size - 1) * pixel_size_um))
            return
        points = np.concatenate(([0], indices, [line.size - 1]))
        diffs = np.diff(points.astype(np.float32))
        for value in diffs:
            if value > 0:
                lengths.append(float(value * pixel_size_um))

    for idx in range(0, boundary_mask.shape[0], step):
        _collect(boundary_mask[idx, :])
    for idx in range(0, boundary_mask.shape[1], step):
        _collect(boundary_mask[:, idx])
    return lengths


def _autocorrelation_curve(mask: np.ndarray, max_lag_px: int = 32) -> dict[str, Any]:
    arr = mask.astype(np.float32)
    mean = float(arr.mean())
    var = float(arr.var())
    if var <= 1e-12:
        return {
            "lags_px": [],
            "corr_x": [],
            "corr_y": [],
            "corr_len_x_px": 0.0,
            "corr_len_y_px": 0.0,
        }

    arr = arr - mean
    limit_x = max(1, min(int(max_lag_px), arr.shape[1] - 1))
    limit_y = max(1, min(int(max_lag_px), arr.shape[0] - 1))
    lags = list(range(1, max(limit_x, limit_y) + 1))
    corr_x: list[float] = []
    corr_y: list[float] = []
    for lag in lags:
        if lag <= limit_x:
            val_x = float(np.mean(arr[:, :-lag] * arr[:, lag:]) / var)
        else:
            val_x = 0.0
        if lag <= limit_y:
            val_y = float(np.mean(arr[:-lag, :] * arr[lag:, :]) / var)
        else:
            val_y = 0.0
        corr_x.append(val_x)
        corr_y.append(val_y)

    def _corr_len(values: list[float]) -> float:
        for idx, value in enumerate(values, 1):
            if value <= math.e**-1:
                return float(idx)
        return float(len(values))

    return {
        "lags_px": lags,
        "corr_x": corr_x,
        "corr_y": corr_y,
        "corr_len_x_px": _corr_len(corr_x),
        "corr_len_y_px": _corr_len(corr_y),
    }


def _lamellar_spacing_metrics(
    lamella_binary: np.ndarray, pixel_size_um: float, step_px: int = 8
) -> dict[str, float]:
    step = max(1, int(step_px))
    total_length_um = 0.0
    intersections = 0
    for idx in range(0, lamella_binary.shape[0], step):
        line = lamella_binary[idx, :] > 0
        total_length_um += max(0, line.size - 1) * pixel_size_um
        intersections += int(np.count_nonzero(np.abs(np.diff(line.astype(np.int16)))))
    for idx in range(0, lamella_binary.shape[1], step):
        line = lamella_binary[:, idx] > 0
        total_length_um += max(0, line.size - 1) * pixel_size_um
        intersections += int(np.count_nonzero(np.abs(np.diff(line.astype(np.int16)))))
    if total_length_um <= 1e-12 or intersections <= 0:
        return {
            "interlamellar_random_spacing_um": 0.0,
            "interlamellar_true_spacing_um": 0.0,
            "lamella_intersection_density_mm_inv": 0.0,
            "lamella_surface_density_mm_inv": 0.0,
        }
    return lamellar_spacing_metrics(total_length_um, intersections)


def _component_shape_metrics(
    mask: np.ndarray, pixel_size_um: float
) -> dict[str, float]:
    mask_b = mask > 0
    if not np.any(mask_b) or ndimage is None:
        return {
            "component_count": 0.0,
            "density_na_mm2": 0.0,
            "mean_length_um": 0.0,
            "mean_aspect_ratio": 0.0,
        }
    labels, num = ndimage.label(mask_b.astype(np.uint8))
    if int(num) <= 0:
        return {
            "component_count": 0.0,
            "density_na_mm2": 0.0,
            "mean_length_um": 0.0,
            "mean_aspect_ratio": 0.0,
        }
    areas = []
    lengths = []
    aspect_ratios = []
    for comp_id in range(1, int(num) + 1):
        ys, xs = np.where(labels == comp_id)
        if ys.size <= 0:
            continue
        height_px = float(ys.max() - ys.min() + 1)
        width_px = float(xs.max() - xs.min() + 1)
        areas.append(float(ys.size))
        lengths.append(float(max(height_px, width_px) * pixel_size_um))
        aspect_ratios.append(
            float(max(height_px, width_px) / max(min(height_px, width_px), 1.0))
        )
    area_mm2 = float(mask.shape[0] * pixel_size_um / 1000.0) * float(
        mask.shape[1] * pixel_size_um / 1000.0
    )
    density = float(len(areas) / area_mm2) if area_mm2 > 1e-12 else 0.0
    return {
        "component_count": float(len(areas)),
        "density_na_mm2": density,
        "mean_length_um": float(np.mean(lengths)) if lengths else 0.0,
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
    }


def run_pro_validation(
    *,
    image_gray: np.ndarray,
    phase_masks: dict[str, np.ndarray],
    morphology_state: SpatialMorphologyState,
    surface_state: SurfaceState,
    transformation_state: ContinuousTransformationState,
    native_um_per_px: float,
    reflected_light_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actual = {
        str(name): float((mask > 0).mean())
        for name, mask in phase_masks.items()
        if isinstance(mask, np.ndarray)
    }
    point_count = {
        str(name): _point_count_fraction(mask)
        for name, mask in phase_masks.items()
        if isinstance(mask, np.ndarray)
    }
    lineal_fraction = {
        str(name): _lineal_fraction(mask)
        for name, mask in phase_masks.items()
        if isinstance(mask, np.ndarray)
    }
    target = dict(transformation_state.phase_fractions)
    phase_error_pct = {}
    point_count_error_pct = {}
    lineal_fraction_error_pct = {}
    phase_fraction_consistency_error = {}
    for phase_name, target_frac in target.items():
        denom = max(1e-6, float(target_frac))
        phase_error_pct[str(phase_name)] = float(
            abs(actual.get(str(phase_name), 0.0) - float(target_frac)) / denom * 100.0
        )
        point_count_error_pct[str(phase_name)] = float(
            abs(point_count.get(str(phase_name), 0.0) - float(target_frac))
            / denom
            * 100.0
        )
        lineal_fraction_error_pct[str(phase_name)] = float(
            abs(lineal_fraction.get(str(phase_name), 0.0) - float(target_frac))
            / denom
            * 100.0
        )
        phase_fraction_consistency_error[str(phase_name)] = float(
            max(
                abs(
                    actual.get(str(phase_name), 0.0)
                    - point_count.get(str(phase_name), 0.0)
                ),
                abs(
                    actual.get(str(phase_name), 0.0)
                    - lineal_fraction.get(str(phase_name), 0.0)
                ),
                abs(
                    point_count.get(str(phase_name), 0.0)
                    - lineal_fraction.get(str(phase_name), 0.0)
                ),
            )
            * 100.0
        )

    grain_count = max(
        1, int(morphology_state.summary.get("prior_austenite_grain_count", 0))
    )
    mean_pag_area_px = float(image_gray.size / grain_count)
    grain_intercept_um_proxy = float(np.sqrt(mean_pag_area_px) * native_um_per_px)
    specimen_area_mm2 = float(image_gray.shape[0] * native_um_per_px / 1000.0) * float(
        image_gray.shape[1] * native_um_per_px / 1000.0
    )
    grains_per_mm2 = (
        float(grain_count / specimen_area_mm2) if specimen_area_mm2 > 1e-12 else 0.0
    )
    grains_per_sq_in_at_100x = float(grains_per_mm2 * 0.064516)
    astm_number_proxy = float(1.0 + np.log2(max(grains_per_sq_in_at_100x, 1e-9)))
    boundary_mask = morphology_state.feature_maps.get("phase_boundaries")
    boundary_density = (
        _boundary_length_density_mm_inv(boundary_mask, native_um_per_px)
        if isinstance(boundary_mask, np.ndarray)
        else 0.0
    )
    interphase_pl = (
        _pl_from_boundary_mask(boundary_mask, native_um_per_px)
        if isinstance(boundary_mask, np.ndarray)
        else 0.0
    )
    pag_boundary_mask = morphology_state.feature_maps.get("prior_austenite_boundaries")
    lineal_x_um, lineal_y_um = (
        _mean_lineal_intercept_um(pag_boundary_mask, native_um_per_px)
        if isinstance(pag_boundary_mask, np.ndarray)
        else (0.0, 0.0)
    )
    grain_pl = (
        _pl_from_boundary_mask(pag_boundary_mask, native_um_per_px)
        if isinstance(pag_boundary_mask, np.ndarray)
        else 0.0
    )
    intercept_lengths = (
        _intercept_lengths_um(pag_boundary_mask, native_um_per_px)
        if isinstance(pag_boundary_mask, np.ndarray)
        else []
    )
    intercept_mean_um = float(np.mean(intercept_lengths)) if intercept_lengths else 0.0
    intercept_std_um = float(np.std(intercept_lengths)) if intercept_lengths else 0.0

    spacing_px_measured = 0.0
    psd_score = 0.0
    lamellar_spacing = {
        "interlamellar_random_spacing_um": 0.0,
        "interlamellar_true_spacing_um": 0.0,
        "lamella_intersection_density_mm_inv": 0.0,
        "lamella_surface_density_mm_inv": 0.0,
    }
    if morphology_state.lamella_field is not None:
        spacing_px_measured = _dominant_spacing_px(morphology_state.lamella_field)
        expected_px = float(
            transformation_state.interlamellar_spacing_um_mean
            / max(native_um_per_px, 1e-6)
        )
        if expected_px > 1e-6 and spacing_px_measured > 0.0:
            psd_score = float(
                np.exp(-abs(spacing_px_measured - expected_px) / expected_px)
            )
    lamella_binary = morphology_state.feature_maps.get("lamellae_binary")
    if isinstance(lamella_binary, np.ndarray):
        lamellar_spacing = _lamellar_spacing_metrics(lamella_binary, native_um_per_px)

    dominant_phase_name = (
        max(actual.items(), key=lambda item: float(item[1]))[0] if actual else ""
    )
    dominant_mask = phase_masks.get(dominant_phase_name)
    corr_curve = (
        _autocorrelation_curve(
            dominant_mask > 0,
            max_lag_px=min(32, image_gray.shape[0] // 4, image_gray.shape[1] // 4),
        )
        if isinstance(dominant_mask, np.ndarray)
        else {
            "lags_px": [],
            "corr_x": [],
            "corr_y": [],
            "corr_len_x_px": 0.0,
            "corr_len_y_px": 0.0,
        }
    )
    roughness_ra_um = float(
        np.mean(np.abs(surface_state.height_um - float(surface_state.height_um.mean())))
    )
    roughness_rq_um = float(
        np.sqrt(
            np.mean(
                (surface_state.height_um - float(surface_state.height_um.mean())) ** 2
            )
        )
    )
    reflected = dict(reflected_light_model or {})
    psf_profile_family = str(reflected.get("psf_profile", "standard"))
    effective_dof_factor = float(reflected.get("effective_dof_factor", 1.0) or 1.0)
    sectioning_suppression_score = float(
        reflected.get("sectioning_suppression_score", 0.0) or 0.0
    )
    sectioning_directionality_score = float(
        reflected.get("sectioning_directionality_score", 0.0) or 0.0
    )
    extended_dof_retention_score = float(
        reflected.get("extended_dof_retention_score", 0.0) or 0.0
    )
    axial_profile_consistency_score = float(
        _clamp(
            0.40 * min(max(effective_dof_factor - 1.0, 0.0), 1.0)
            + 0.35 * sectioning_suppression_score
            + 0.25 * extended_dof_retention_score,
            0.0,
            1.0,
        )
    )
    bainite_sheaves_mask = morphology_state.feature_maps.get("bainite_sheaves_binary")
    upper_bainite_mask = morphology_state.feature_maps.get(
        "upper_bainite_sheaves_binary"
    )
    lower_bainite_mask = morphology_state.feature_maps.get(
        "lower_bainite_sheaves_binary"
    )
    widmanstatten_mask = morphology_state.feature_maps.get(
        "widmanstatten_sideplates_binary"
    )
    allotriomorphic_mask = morphology_state.feature_maps.get(
        "allotriomorphic_ferrite_binary"
    )
    martensite_laths_mask = morphology_state.feature_maps.get("martensite_laths_binary")
    ferrite_mask = phase_masks.get("FERRITE")
    bainite_mask = phase_masks.get("BAINITE")
    diagnostics: list[str] = []
    artifact_layer_remaining = float(
        surface_state.summary.get("artifact_layer_remaining_um", 0.0)
    )
    if artifact_layer_remaining > max(
        0.03, 0.6 * float(surface_state.summary.get("roughness_target_um", 0.05))
    ):
        diagnostics.append(
            "Residual damaged layer may remain after polishing; observed morphology may be prep-induced."
        )
    if float(surface_state.summary.get("pearlite_fragmentation_risk", 0.0)) > 0.12:
        diagnostics.append(
            "Pearlite lamella distortion risk is elevated; false tempered/bainitic appearance is possible."
        )
    if float(surface_state.summary.get("grinding_heat_risk", 0.0)) > 0.62:
        diagnostics.append(
            "Grinding heat risk is elevated; reheated/tempered surface artifact is plausible."
        )
    if float(surface_state.summary.get("etch_reproducibility_risk", 0.0)) > 0.35:
        diagnostics.append(
            "Etch reproducibility risk is elevated; contrast may be deposit/stain-driven."
        )
    if (
        float(surface_state.summary.get("directional_artifact_anisotropy_score", 0.0))
        > 0.18
    ):
        diagnostics.append(
            "Strong alignment between scratches and structural orientation suggests prep-induced anisotropy."
        )
    if float(surface_state.summary.get("scratch_trace_revelation_risk", 0.0)) > 0.28:
        diagnostics.append(
            "Scratch revelation risk is elevated; etched contrast may overemphasize grinding traces."
        )
    if float(surface_state.summary.get("prep_directionality_banding_risk", 0.0)) > 0.20:
        diagnostics.append(
            "Directional prep banding risk is elevated; banded contrast may not be structural."
        )
    if float(surface_state.summary.get("false_porosity_pullout_risk", 0.0)) > 0.14:
        diagnostics.append(
            "Pull-out risk is elevated; apparent porosity or particle loss may be preparation-induced."
        )
    if float(surface_state.summary.get("relief_dominance_risk", 0.0)) > 0.30:
        diagnostics.append(
            "Surface relief dominates etch contrast; observed morphology may be topography-driven."
        )
    if (
        float(surface_state.summary.get("stain_deposit_contrast_dominance_risk", 0.0))
        > 0.24
    ):
        diagnostics.append(
            "Stain/deposit contrast dominance is elevated; apparent phase contrast may be chemical residue-driven."
        )
    residual_damage_ratio = float(
        surface_state.summary.get("artifact_layer_remaining_um", 0.0)
        / max(0.03, 0.6 * float(surface_state.summary.get("roughness_target_um", 0.05)))
    )
    artifact_risk_scores = {
        "residual_damage_ratio": residual_damage_ratio,
        "pearlite_fragmentation": float(
            surface_state.summary.get("pearlite_fragmentation_risk", 0.0)
        ),
        "grinding_heat": float(surface_state.summary.get("grinding_heat_risk", 0.0)),
        "etch_reproducibility": float(
            surface_state.summary.get("etch_reproducibility_risk", 0.0)
        ),
        "directional_anisotropy": float(
            surface_state.summary.get("directional_artifact_anisotropy_score", 0.0)
        ),
    }
    trigger_ratio = {
        "residual_damage": float(residual_damage_ratio),
        "pearlite_fragmentation": float(
            artifact_risk_scores["pearlite_fragmentation"] / 0.12
        ),
        "grinding_heat": float(artifact_risk_scores["grinding_heat"] / 0.62),
        "etch_reproducibility": float(
            artifact_risk_scores["etch_reproducibility"] / 0.35
        ),
        "directional_anisotropy": float(
            artifact_risk_scores["directional_anisotropy"] / 0.18
        ),
    }
    dominant_driver = (
        max(trigger_ratio.items(), key=lambda item: float(item[1]))[0]
        if trigger_ratio
        else ""
    )
    dominant_trigger_ratio = (
        float(max(trigger_ratio.values())) if trigger_ratio else 0.0
    )
    artifact_risk_scores["trigger_ratio"] = trigger_ratio
    artifact_risk_scores["dominant_driver"] = dominant_driver
    artifact_risk_scores["dominant_trigger_ratio"] = dominant_trigger_ratio
    artifact_risk_scores["triggered_count"] = int(
        sum(1 for value in trigger_ratio.values() if float(value) >= 1.0)
    )
    bainite_components = (
        _component_shape_metrics(bainite_sheaves_mask, native_um_per_px)
        if isinstance(bainite_sheaves_mask, np.ndarray)
        else {
            "component_count": 0.0,
            "density_na_mm2": 0.0,
            "mean_length_um": 0.0,
            "mean_aspect_ratio": 0.0,
        }
    )
    bainite_family_split_label = str(transformation_state.bainite_morphology_family)
    bainite_family_split_mask = None
    if bainite_family_split_label == "upper_bainite_sheaves" and isinstance(
        upper_bainite_mask, np.ndarray
    ):
        bainite_family_split_mask = upper_bainite_mask
    elif bainite_family_split_label == "lower_bainite_sheaves" and isinstance(
        lower_bainite_mask, np.ndarray
    ):
        bainite_family_split_mask = lower_bainite_mask
    ferritic_clean_case = bool(
        float(actual.get("FERRITE", 0.0)) >= 0.90
        and float(actual.get("PEARLITE", 0.0)) <= 0.08
        and float(actual.get("CEMENTITE", 0.0)) <= 0.05
        and float(actual.get("MARTENSITE", 0.0)) <= 0.05
        and float(actual.get("BAINITE", 0.0)) <= 0.05
    )
    dark_defect_field_dominance = float(
        _clamp(
            0.45
            * float(surface_state.summary.get("scratch_trace_revelation_risk", 0.0))
            + 0.35
            * float(
                surface_state.summary.get("stain_deposit_contrast_dominance_risk", 0.0)
            )
            + 0.20
            * float(surface_state.summary.get("false_porosity_pullout_risk", 0.0)),
            0.0,
            1.0,
        )
    )
    bright_ferritic_baseline_score = float(
        _clamp(
            ((float(np.mean(image_gray.astype(np.float32))) - 110.0) / 80.0)
            * (1.0 - 0.75 * dark_defect_field_dominance),
            0.0,
            1.0,
        )
    )
    if ferritic_clean_case and bright_ferritic_baseline_score < 0.50:
        diagnostics.append(
            "Ferritic clean baseline is too dark; defect field dominates more than expected."
        )

    return {
        "grain_size_astm_proxy_um": float(grain_intercept_um_proxy),
        "grain_size_astm_number_proxy": float(astm_number_proxy),
        "grains_per_mm2_proxy": float(grains_per_mm2),
        "mean_lineal_intercept_um_x": float(lineal_x_um),
        "mean_lineal_intercept_um_y": float(lineal_y_um),
        "mean_intercept_um": float(intercept_mean_um),
        "intercept_std_um": float(intercept_std_um),
        "phase_fraction_areal_fraction_exact": actual,
        "phase_fraction_point_count_proxy": actual,
        "phase_fraction_grid_point_count_proxy": point_count,
        "phase_fraction_lineal_fraction_proxy": lineal_fraction,
        "phase_fraction_error_pct": phase_error_pct,
        "phase_fraction_point_count_error_pct": point_count_error_pct,
        "phase_fraction_lineal_fraction_error_pct": lineal_fraction_error_pct,
        "phase_fraction_consistency_error_pct": phase_fraction_consistency_error,
        "interlamellar_spacing_proxy_um": float(
            spacing_px_measured * native_um_per_px if spacing_px_measured > 0.0 else 0.0
        ),
        **lamellar_spacing,
        "two_point_corr_score": float(
            0.5
            * (
                np.exp(
                    -abs(corr_curve["corr_len_x_px"] - corr_curve["corr_len_y_px"])
                    / max(corr_curve["corr_len_x_px"] + 1e-6, 1.0)
                )
                + psd_score
            )
        ),
        "two_point_corr_curve": corr_curve,
        "psd_score": float(psd_score),
        "boundary_density": float(boundary_density),
        "pl_grain_boundaries_mm_inv": float(grain_pl),
        "pl_interphase_boundaries_mm_inv": float(interphase_pl),
        "sv_grain_boundaries_mm_inv": float(2.0 * grain_pl),
        "sv_interphase_boundaries_mm_inv": float(2.0 * interphase_pl),
        "boundary_ferrite_coverage": float((ferrite_mask > 0).mean())
        if isinstance(ferrite_mask, np.ndarray)
        else 0.0,
        "allotriomorphic_ferrite_area_fraction": float(
            (allotriomorphic_mask > 0).mean()
        )
        if isinstance(allotriomorphic_mask, np.ndarray)
        else 0.0,
        "bainite_sheaf_area_fraction": float((bainite_sheaves_mask > 0).mean())
        if isinstance(bainite_sheaves_mask, np.ndarray)
        else 0.0,
        "upper_bainite_sheaf_area_fraction": float((upper_bainite_mask > 0).mean())
        if isinstance(upper_bainite_mask, np.ndarray)
        else 0.0,
        "lower_bainite_sheaf_area_fraction": float((lower_bainite_mask > 0).mean())
        if isinstance(lower_bainite_mask, np.ndarray)
        else 0.0,
        "bainite_family_split_label": bainite_family_split_label,
        "bainite_family_split_area_fraction": float(
            (bainite_family_split_mask > 0).mean()
        )
        if isinstance(bainite_family_split_mask, np.ndarray)
        else 0.0,
        "bainite_sheaf_count": float(bainite_components["component_count"]),
        "bainite_sheaf_density_na_mm2": float(bainite_components["density_na_mm2"]),
        "mean_sheaf_length_um": float(bainite_components["mean_length_um"]),
        "sheaf_aspect_ratio_proxy": float(bainite_components["mean_aspect_ratio"]),
        "widmanstatten_sideplate_area_fraction": float((widmanstatten_mask > 0).mean())
        if isinstance(widmanstatten_mask, np.ndarray)
        else 0.0,
        "martensite_lath_density": float((martensite_laths_mask > 0).mean())
        if isinstance(martensite_laths_mask, np.ndarray)
        else 0.0,
        "bainite_fraction_exact": float((bainite_mask > 0).mean())
        if isinstance(bainite_mask, np.ndarray)
        else 0.0,
        "artifact_diagnostics": diagnostics,
        "directional_artifact_anisotropy_score": float(
            surface_state.summary.get("directional_artifact_anisotropy_score", 0.0)
        ),
        "scratch_trace_revelation_risk": float(
            surface_state.summary.get("scratch_trace_revelation_risk", 0.0)
        ),
        "prep_directionality_banding_risk": float(
            surface_state.summary.get("prep_directionality_banding_risk", 0.0)
        ),
        "false_porosity_pullout_risk": float(
            surface_state.summary.get("false_porosity_pullout_risk", 0.0)
        ),
        "relief_dominance_risk": float(
            surface_state.summary.get("relief_dominance_risk", 0.0)
        ),
        "stain_deposit_contrast_dominance_risk": float(
            surface_state.summary.get("stain_deposit_contrast_dominance_risk", 0.0)
        ),
        "artifact_risk_scores": artifact_risk_scores,
        "bright_ferritic_baseline_score": bright_ferritic_baseline_score,
        "dark_defect_field_dominance": dark_defect_field_dominance,
        "ferritic_clean_case": ferritic_clean_case,
        "dominant_scratch_direction_deg": surface_state.summary.get(
            "dominant_scratch_direction_deg", None
        ),
        "structural_orientation_deg": surface_state.summary.get(
            "structural_orientation_deg", None
        ),
        "surface_roughness_ra_um": float(roughness_ra_um),
        "surface_roughness_rq_um": float(roughness_rq_um),
        "surface_relief_range_um": float(np.ptp(surface_state.height_um)),
        "etch_depth_range_um": float(np.ptp(surface_state.etch_depth_um)),
        "psf_profile_family": psf_profile_family,
        "effective_dof_factor": effective_dof_factor,
        "sectioning_suppression_score": sectioning_suppression_score,
        "sectioning_directionality_score": sectioning_directionality_score,
        "extended_dof_retention_score": extended_dof_retention_score,
        "axial_profile_consistency_score": axial_profile_consistency_score,
    }
