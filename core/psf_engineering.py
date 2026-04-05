from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

PSF_PROFILE_STANDARD = "standard"
PSF_PROFILE_BESSEL = "bessel_extended_dof"
PSF_PROFILE_AIRY = "airy_push_pull"
PSF_PROFILE_SELF_ROTATING = "self_rotating"
PSF_PROFILE_STIR = "stir_sectioning"
PSF_PROFILE_HYBRID = "lens_axicon_hybrid"

PSF_PROFILE_KEYS: tuple[str, ...] = (
    PSF_PROFILE_STANDARD,
    PSF_PROFILE_BESSEL,
    PSF_PROFILE_AIRY,
    PSF_PROFILE_SELF_ROTATING,
    PSF_PROFILE_STIR,
    PSF_PROFILE_HYBRID,
)

DEFAULT_SECTIONING_SHEAR_DEG = 35.0
DEFAULT_HYBRID_BALANCE = 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _normalize01(field: np.ndarray) -> np.ndarray:
    arr = field.astype(np.float32, copy=False)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    arr = field.astype(np.float32, copy=False)
    sigma_v = float(max(0.0, sigma))
    if sigma_v <= 1e-6:
        return arr.copy()
    if ndimage is not None:
        return ndimage.gaussian_filter(arr, sigma=sigma_v)
    return arr.copy()


def _prepare_profile_settings(microscope_profile: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(microscope_profile or {})
    profile = str(payload.get("psf_profile", PSF_PROFILE_STANDARD)).strip().lower()
    if profile not in PSF_PROFILE_KEYS:
        profile = PSF_PROFILE_STANDARD
    strength = _clamp(float(payload.get("psf_strength", 0.0) or 0.0), 0.0, 1.0)
    shear_deg = float(payload.get("sectioning_shear_deg", DEFAULT_SECTIONING_SHEAR_DEG) or DEFAULT_SECTIONING_SHEAR_DEG)
    hybrid_balance = _clamp(float(payload.get("hybrid_balance", DEFAULT_HYBRID_BALANCE) or DEFAULT_HYBRID_BALANCE), 0.0, 1.0)
    return {
        "psf_profile": profile,
        "psf_strength": strength,
        "sectioning_shear_deg": shear_deg,
        "hybrid_balance": hybrid_balance,
    }


def apply_live_psf_profile(
    *,
    original_view: np.ndarray,
    focused_view: np.ndarray,
    microscope_profile: dict[str, Any] | None,
    focus_distance_mm: float | None,
    focus_target_mm: float | None,
    focus_quality: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = _prepare_profile_settings(microscope_profile)
    profile = str(settings["psf_profile"])
    strength = float(settings["psf_strength"])
    shear_deg = float(settings["sectioning_shear_deg"])
    hybrid_balance = float(settings["hybrid_balance"])
    focus_quality_v = float(_clamp(1.0 if focus_quality is None else focus_quality, 0.0, 1.0))
    focus_error_mm = (
        0.0
        if focus_distance_mm is None or focus_target_mm is None
        else float(abs(float(focus_distance_mm) - float(focus_target_mm)))
    )
    focus_error_norm = float(_clamp(focus_error_mm / 0.8, 0.0, 1.0))
    original_f = original_view.astype(np.float32, copy=False)
    focused_f = focused_view.astype(np.float32, copy=False)
    low = _gaussian(focused_f, 1.4)
    high = focused_f - low
    gy, gx = np.gradient(focused_f)
    edge_strength = _normalize01(np.hypot(gx, gy))

    image = focused_f.copy()
    effective_dof_factor = 1.0
    sectioning_suppression_score = 0.0
    lateral_shift_signature = 0.0
    rotation_signature_deg = None
    axial_profile_mode = "standard"
    sectioning_active = False
    extended_dof_retention_score = 0.0
    sectioning_directionality_score = 0.0

    def _bessel_variant(base: np.ndarray) -> tuple[np.ndarray, float, float]:
        local_strength = float(strength * (0.35 + 0.65 * focus_error_norm))
        restored = base + 0.62 * local_strength * (original_f - _gaussian(original_f, 1.1))
        retained = float(_clamp(np.mean(np.abs(restored - _gaussian(restored, 1.0))) / 42.0, 0.0, 1.0))
        dof_factor = float(1.0 + 1.6 * strength)
        return np.clip(restored, 0.0, 255.0), dof_factor, retained

    if profile == PSF_PROFILE_BESSEL:
        image, effective_dof_factor, extended_dof_retention_score = _bessel_variant(image)
        axial_profile_mode = "extended_dof"
    elif profile == PSF_PROFILE_AIRY:
        axial_profile_mode = "depth_coded_shift"
        signed_error = 0.0
        if focus_distance_mm is not None and focus_target_mm is not None:
            signed_error = float(focus_distance_mm) - float(focus_target_mm)
        shift_px = int(round((1.0 + 4.0 * strength) * focus_error_norm))
        shift_sign = -1 if signed_error < 0.0 else 1
        shifted = np.roll(high, shift_px * shift_sign, axis=1)
        image = np.clip(image + 0.34 * strength * shifted, 0.0, 255.0)
        lateral_shift_signature = float(shift_px * shift_sign)
    elif profile == PSF_PROFILE_SELF_ROTATING:
        axial_profile_mode = "rotational_contrast"
        theta = math.radians(15.0 + 110.0 * strength + 15.0 * focus_error_norm)
        directional = _normalize01(np.cos(theta) * gx + np.sin(theta) * gy) - 0.5
        image = np.clip(image + 34.0 * strength * directional, 0.0, 255.0)
        rotation_signature_deg = float(math.degrees(theta))
    elif profile == PSF_PROFILE_STIR:
        axial_profile_mode = "sectioning"
        theta = math.radians(shear_deg)
        directional = np.cos(theta) * gx + np.sin(theta) * gy
        low_clutter = _gaussian(focused_f, 2.1 + 1.2 * strength)
        suppression = float(strength * (0.25 + 0.75 * focus_error_norm))
        image = np.clip(
            focused_f * (1.0 - 0.18 * suppression)
            - 0.22 * suppression * (low_clutter - float(np.mean(low_clutter)))
            + 28.0 * suppression * (_normalize01(directional) - 0.5),
            0.0,
            255.0,
        )
        sectioning_suppression_score = float(_clamp(suppression, 0.0, 1.0))
        sectioning_directionality_score = float(_clamp(np.std(directional) / 18.0, 0.0, 1.0))
        lateral_shift_signature = float(np.mean(directional))
        sectioning_active = True
    elif profile == PSF_PROFILE_HYBRID:
        axial_profile_mode = "hybrid"
        bessel_img, bessel_dof, retained = _bessel_variant(image)
        mix = float(_clamp(hybrid_balance * (0.35 + 0.65 * strength), 0.0, 1.0))
        image = np.clip((1.0 - mix) * focused_f + mix * bessel_img, 0.0, 255.0)
        effective_dof_factor = float(1.0 + (bessel_dof - 1.0) * mix)
        extended_dof_retention_score = float(retained * mix)
    else:
        profile = PSF_PROFILE_STANDARD

    profile_meta = {
        **settings,
        "psf_engineering_applied": bool(profile != PSF_PROFILE_STANDARD and strength > 0.0),
        "axial_profile_mode": axial_profile_mode,
        "effective_dof_factor": float(effective_dof_factor),
        "sectioning_suppression_score": float(sectioning_suppression_score),
        "lateral_shift_signature": float(lateral_shift_signature),
        "extended_dof_retention_score": float(extended_dof_retention_score),
        "sectioning_directionality_score": float(sectioning_directionality_score),
        "sectioning_active": bool(sectioning_active),
        "focus_profile_mode": axial_profile_mode,
        "rotation_signature_deg": rotation_signature_deg,
    }
    return np.clip(image, 0.0, 255.0).astype(np.uint8), profile_meta


def apply_static_psf_profile(
    *,
    image_gray: np.ndarray,
    height_um: np.ndarray,
    edge_strength: np.ndarray,
    orientation_rad: np.ndarray,
    phase_edge_field: np.ndarray,
    microscope_profile: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = _prepare_profile_settings(microscope_profile)
    profile = str(settings["psf_profile"])
    strength = float(settings["psf_strength"])
    shear_deg = float(settings["sectioning_shear_deg"])
    hybrid_balance = float(settings["hybrid_balance"])
    image = image_gray.astype(np.float32, copy=False)
    edge = edge_strength.astype(np.float32, copy=False)
    phase_edges = _normalize01(phase_edge_field.astype(np.float32, copy=False))
    relief = height_um.astype(np.float32, copy=False)
    relief_centered = relief - float(np.mean(relief))
    relief_norm = _normalize01(relief_centered)
    gy, gx = np.gradient(relief)

    effective_dof_factor = 1.0
    sectioning_suppression_score = 0.0
    lateral_shift_signature = 0.0
    rotation_signature_deg = None
    axial_profile_mode = "standard"
    sectioning_active = False
    extended_dof_retention_score = 0.0
    sectioning_directionality_score = 0.0

    def _bessel_variant(base: np.ndarray) -> tuple[np.ndarray, float, float]:
        local = base + 28.0 * strength * (edge - 0.5) - 10.0 * strength * (relief_norm - 0.5)
        retained = float(_clamp(np.mean(np.abs(edge - 0.5)) * 2.5, 0.0, 1.0))
        return np.clip(local, 0.0, 255.0), float(1.0 + 1.4 * strength), retained

    if profile == PSF_PROFILE_BESSEL:
        image, effective_dof_factor, extended_dof_retention_score = _bessel_variant(image)
        axial_profile_mode = "extended_dof"
    elif profile == PSF_PROFILE_AIRY:
        axial_profile_mode = "depth_coded_shift"
        shift_px = max(1, int(round(1.0 + 4.0 * strength)))
        shifted = np.roll(phase_edges, shift_px, axis=1)
        image = np.clip(image + 24.0 * strength * (shifted - phase_edges), 0.0, 255.0)
        lateral_shift_signature = float(shift_px)
    elif profile == PSF_PROFILE_SELF_ROTATING:
        axial_profile_mode = "rotational_contrast"
        theta = math.radians(20.0 + 80.0 * strength)
        oriented = _normalize01(np.cos(theta + orientation_rad.astype(np.float32)) * gx + np.sin(theta + orientation_rad.astype(np.float32)) * gy) - 0.5
        image = np.clip(image + 22.0 * strength * oriented, 0.0, 255.0)
        rotation_signature_deg = float(math.degrees(theta))
    elif profile == PSF_PROFILE_STIR:
        axial_profile_mode = "sectioning"
        theta = math.radians(shear_deg)
        directional = np.cos(theta) * gx + np.sin(theta) * gy
        suppression = float(_clamp(0.35 + 0.65 * strength, 0.0, 1.0))
        low = _gaussian(image, 1.5 + 1.4 * strength)
        image = np.clip(
            image * (1.0 - 0.14 * suppression)
            - 0.24 * suppression * (low - float(np.mean(low)))
            + 20.0 * suppression * (_normalize01(directional) - 0.5)
            + 8.0 * suppression * (phase_edges - 0.5),
            0.0,
            255.0,
        )
        sectioning_suppression_score = float(suppression)
        sectioning_directionality_score = float(_clamp(np.std(directional) / 0.25, 0.0, 1.0))
        lateral_shift_signature = float(np.mean(directional))
        sectioning_active = True
    elif profile == PSF_PROFILE_HYBRID:
        axial_profile_mode = "hybrid"
        bessel_img, bessel_dof, retained = _bessel_variant(image)
        mix = float(_clamp(hybrid_balance * (0.35 + 0.65 * strength), 0.0, 1.0))
        image = np.clip((1.0 - mix) * image + mix * bessel_img, 0.0, 255.0)
        effective_dof_factor = float(1.0 + (bessel_dof - 1.0) * mix)
        extended_dof_retention_score = float(retained * mix)
    else:
        profile = PSF_PROFILE_STANDARD

    profile_meta = {
        **settings,
        "psf_engineering_applied": bool(profile != PSF_PROFILE_STANDARD and strength > 0.0),
        "axial_profile_mode": axial_profile_mode,
        "effective_dof_factor": float(effective_dof_factor),
        "sectioning_suppression_score": float(sectioning_suppression_score),
        "lateral_shift_signature": float(lateral_shift_signature),
        "extended_dof_retention_score": float(extended_dof_retention_score),
        "sectioning_directionality_score": float(sectioning_directionality_score),
        "sectioning_active": bool(sectioning_active),
        "rotation_signature_deg": rotation_signature_deg,
    }
    return np.clip(image, 0.0, 255.0).astype(np.uint8), profile_meta
