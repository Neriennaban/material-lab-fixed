from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None


def _normalize01(field: np.ndarray) -> np.ndarray:
    arr = field.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.01:
        return field.astype(np.float32, copy=False)
    if ndimage is not None:
        return ndimage.gaussian_filter(field.astype(np.float32), sigma=float(sigma))
    return field.astype(np.float32, copy=False)


def _gradient_fields(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = field.astype(np.float32, copy=False)
    gx = np.gradient(base, axis=1)
    gy = np.gradient(base, axis=0)
    mag = _normalize01(np.hypot(gx, gy))
    return gx.astype(np.float32), gy.astype(np.float32), mag


def _isotropic_fraction(phase_masks: dict[str, np.ndarray] | None, pure_iron_like: bool) -> float:
    if pure_iron_like:
        return 1.0
    if not isinstance(phase_masks, dict) or not phase_masks:
        return 0.0
    cubic = {
        "BCC_B2",
        "FERRITE",
        "AUSTENITE",
        "FCC_A1",
        "MARTENSITE_CUBIC",
        "ALPHA",
        "BETA",
    }
    total = 0.0
    cubic_cov = 0.0
    for name, mask in phase_masks.items():
        if not isinstance(mask, np.ndarray):
            continue
        frac = float((mask > 0).mean())
        total += frac
        if str(name).upper() in cubic:
            cubic_cov += frac
    if total <= 1e-9:
        return 0.0
    return float(np.clip(cubic_cov / total, 0.0, 1.0))


def build_ferromagnetic_mask(
    phase_masks: dict[str, np.ndarray] | None,
    *,
    pure_iron_like: bool = False,
    size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float]:
    if isinstance(phase_masks, dict) and phase_masks:
        inferred_size = next((mask.shape for mask in phase_masks.values() if isinstance(mask, np.ndarray)), size)
    else:
        inferred_size = size
    if inferred_size is None:
        inferred_size = (1, 1)
    ferromagnetic = np.zeros(inferred_size, dtype=np.float32)
    ferromagnetic_names = {
        "BCC_B2",
        "FERRITE",
        "DELTA_FERRITE",
        "MARTENSITE",
        "MARTENSITE_TETRAGONAL",
        "MARTENSITE_CUBIC",
        "BAINITE",
        "TROOSTITE",
        "SORBITE",
    }
    if isinstance(phase_masks, dict):
        for name, mask in phase_masks.items():
            if not isinstance(mask, np.ndarray) or mask.shape != ferromagnetic.shape:
                continue
            if str(name).upper() in ferromagnetic_names:
                ferromagnetic = np.maximum(ferromagnetic, (mask > 0).astype(np.float32))
    if pure_iron_like and float(ferromagnetic.mean()) <= 1e-6:
        ferromagnetic[:] = 1.0
    return ferromagnetic.astype(np.float32), float(np.clip(ferromagnetic.mean(), 0.0, 1.0))


def apply_optical_mode_transfer(
    *,
    image_gray: np.ndarray,
    optical_mode: str,
    optical_mode_parameters: dict[str, Any] | None = None,
    height_um: np.ndarray | None = None,
    edge_strength: np.ndarray | None = None,
    phase_edge_field: np.ndarray | None = None,
    orientation_rad: np.ndarray | None = None,
    phase_masks: dict[str, np.ndarray] | None = None,
    ferromagnetic_mask: np.ndarray | None = None,
    pure_iron_like: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = dict(optical_mode_parameters or {})
    mode = str(optical_mode or "brightfield").strip().lower()
    arr = image_gray.astype(np.float32, copy=False) / 255.0
    base_gx, base_gy, base_edges = _gradient_fields(arr)
    edge = base_edges if edge_strength is None else _normalize01(edge_strength.astype(np.float32))
    phase_edges = (
        np.zeros_like(arr, dtype=np.float32)
        if phase_edge_field is None
        else _normalize01(phase_edge_field.astype(np.float32))
    )
    if height_um is not None:
        relief_source = height_um.astype(np.float32, copy=False)
    else:
        relief_source = _smooth(arr, sigma=1.1)
    relief_gx, relief_gy, relief_edges = _gradient_fields(relief_source)
    relief_nm_proxy = float(np.quantile(relief_edges, 0.95) - np.quantile(relief_edges, 0.05))
    isotropic_fraction = _isotropic_fraction(phase_masks, pure_iron_like)
    derived_ferromagnetic_mask, ferromagnetic_fraction = build_ferromagnetic_mask(
        phase_masks,
        pure_iron_like=pure_iron_like,
        size=arr.shape,
    )
    if isinstance(ferromagnetic_mask, np.ndarray) and ferromagnetic_mask.shape == arr.shape:
        ferromagnetic_field = np.clip(ferromagnetic_mask.astype(np.float32), 0.0, 1.0)
        ferromagnetic_fraction = float(np.clip(ferromagnetic_field.mean(), 0.0, 1.0))
    else:
        ferromagnetic_field = derived_ferromagnetic_mask

    meta: dict[str, Any] = {
        "optical_mode": mode,
        "mode_parameters": {},
        "pure_iron_like": bool(pure_iron_like),
        "isotropic_fraction": float(isotropic_fraction),
    }
    out = arr.copy()

    if mode == "darkfield":
        scatter_sensitivity = float(params.get("scatter_sensitivity", 1.0) or 1.0)
        scatter = _normalize01(0.55 * edge + 0.25 * phase_edges + 0.20 * relief_edges)
        flat = 1.0 - edge
        out = np.clip(0.10 * arr + scatter * (0.90 * scatter_sensitivity), 0.0, 1.0)
        meta["mode_parameters"] = {
            "scatter_pass_fraction": float(scatter.mean()),
            "flat_field_suppression": float(np.clip(flat.mean() * 0.85, 0.0, 1.0)),
            "annulus_inner_na_fraction": float(params.get("annulus_inner_na_fraction", 0.70)),
            "annulus_outer_na_fraction": float(params.get("annulus_outer_na_fraction", 0.95)),
        }
    elif mode == "polarized":
        crossed = bool(params.get("crossed_polars", True))
        polarizer_angle_deg = float(params.get("polarizer_angle_deg", 0.0) or 0.0)
        analyzer_angle_deg = float(params.get("analyzer_angle_deg", 90.0 if crossed else 0.0) or 0.0)
        if orientation_rad is None:
            orient = np.arctan2(relief_gy, relief_gx).astype(np.float32)
        else:
            orient = orientation_rad.astype(np.float32, copy=False)
        angle_offset = np.deg2rad(analyzer_angle_deg - polarizer_angle_deg)
        anisotropy_signal = _normalize01(0.5 + 0.5 * np.cos(2.0 * (orient - angle_offset)))
        if crossed:
            extinction = 0.06 + 0.34 * (1.0 - isotropic_fraction)
            out = np.clip(arr * extinction + anisotropy_signal * (0.56 * (1.0 - isotropic_fraction)), 0.0, 1.0)
        else:
            out = np.clip(0.72 * arr + 0.28 * anisotropy_signal, 0.0, 1.0)
        meta["mode_parameters"] = {
            "crossed_polars": crossed,
            "polarizer_angle_deg": polarizer_angle_deg,
            "analyzer_angle_deg": analyzer_angle_deg,
            "anisotropy_coverage": float(max(0.0, 1.0 - isotropic_fraction)),
            "depolarization_score": float(np.clip(phase_edges.mean() * 0.25 + relief_edges.mean() * 0.30, 0.0, 1.0)),
        }
    elif mode == "phase_contrast":
        plate_type = str(params.get("phase_plate_type", "positive")).strip().lower()
        gain = float(params.get("phase_object_gain", 0.24) or 0.24)
        phase_signal = _normalize01(relief_gx + relief_gy if height_um is not None else ndimage.laplace(arr.astype(np.float32)) if ndimage is not None else base_gx + base_gy)
        signed = (phase_signal - 0.5) * gain
        if plate_type == "negative":
            signed *= -1.0
        out = np.clip(arr + signed, 0.0, 1.0)
        meta["mode_parameters"] = {
            "phase_plate_type": "negative" if plate_type == "negative" else "positive",
            "phase_object_gain": float(gain),
            "reference_beam_fraction": float(params.get("reference_beam_fraction", 0.55)),
            "nm_relief_proxy_p05_p95": float(relief_nm_proxy),
        }
    elif mode == "dic":
        shear_axis_deg = float(params.get("dic_shear_axis_deg", 35.0) or 35.0)
        theta = np.deg2rad(shear_axis_deg)
        signed_gradient = relief_gx * np.cos(theta) + relief_gy * np.sin(theta)
        g_std = float(np.std(signed_gradient)) + 1e-6
        strength = float(params.get("dic_strength", 2.4) or 2.4)
        amplitude = float(params.get("dic_amplitude", 0.24) or 0.24)
        dic_signal = np.tanh(strength * signed_gradient / g_std).astype(np.float32)
        out = np.clip(arr + amplitude * dic_signal, 0.0, 1.0)
        meta["mode_parameters"] = {
            "dic_shear_axis_deg": shear_axis_deg,
            "dic_strength": strength,
            "dic_amplitude": amplitude,
            "interference_gradient": float(np.quantile(dic_signal, 0.95) - np.quantile(dic_signal, 0.05)),
        }
    elif mode == "magnetic_etching":
        field_active = bool(params.get("magnetic_field_active", True))
        residual = float(params.get("residual_attraction_level", 0.08 if field_active else 0.04) or 0.0)
        domain_strength = float(params.get("domain_pattern_strength", 0.65 if field_active else 0.25) or 0.0)
        if orientation_rad is None:
            orient = np.arctan2(relief_gy, relief_gx + 1e-6).astype(np.float32)
        else:
            orient = orientation_rad.astype(np.float32, copy=False)
        yy, xx = np.mgrid[: arr.shape[0], : arr.shape[1]].astype(np.float32)
        axis = float(params.get("magnetic_axis_deg", 18.0) or 18.0)
        theta = np.deg2rad(axis)
        band_coord = xx * np.cos(theta) + yy * np.sin(theta)
        domain_pattern = _normalize01(
            0.52
            + 0.32 * np.cos(0.11 * band_coord + 3.5 * orient)
            + 0.16 * np.cos(0.07 * (xx - yy))
        )
        if ferromagnetic_fraction <= 1e-6:
            out = arr.copy()
            magnetic_signal = np.zeros_like(arr, dtype=np.float32)
        else:
            magnetic_signal = ferromagnetic_field * _normalize01(0.38 * phase_edges + 0.24 * edge + 0.38 * domain_pattern)
            signal_strength = residual + domain_strength * (1.0 if field_active else 0.35)
            out = np.clip(
                arr * (1.0 - 0.08 * ferromagnetic_field)
                - magnetic_signal * signal_strength * 0.58,
                0.0,
                1.0,
            )
        meta["mode_parameters"] = {
            "magnetic_field_active": field_active,
            "ferromagnetic_fraction": float(ferromagnetic_fraction),
            "magnetic_signal_fraction": float(magnetic_signal.mean()),
            "domain_pattern_strength": float(domain_strength),
            "residual_attraction_level": float(residual),
            "colloid_particle_scale_nm": 30.0,
            "magnetic_mode_limitations": "ferromagnetic_features_only",
        }
    else:
        meta["mode_parameters"] = {
            "uniformity_score": float(np.clip(1.0 - 0.35 * float(edge.mean()), 0.0, 1.0)),
            "aperture_fraction": float(params.get("aperture_fraction", 0.92)),
        }

    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8), meta
