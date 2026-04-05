from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from core.contracts_v3 import SynthesisProfileV3
from core.optical_mode_transfer import apply_optical_mode_transfer
from core.psf_engineering import apply_static_psf_profile
from core.metallography_v3.realism_utils import (
    low_frequency_field,
    normalize01,
    rescale_to_u8,
)
from core.metallography_v3.system_generators.base import soft_unsharp

from .contracts import SpatialMorphologyState, SurfaceState


def _lift_small_dark_defects(
    image_gray: np.ndarray, *, threshold: float = 38.0, max_pixels: int = 36
) -> np.ndarray:
    if ndimage is None:
        return image_gray.astype(np.uint8, copy=False)
    arr = image_gray.astype(np.float32, copy=False)
    mask = arr < float(threshold)
    labels, count = ndimage.label(mask.astype(np.uint8))
    if int(count) <= 0:
        return image_gray.astype(np.uint8, copy=False)
    local = ndimage.gaussian_filter(arr, sigma=1.0)
    out = arr.copy()
    for label in range(1, int(count) + 1):
        zone = labels == label
        if int(zone.sum()) <= int(max_pixels):
            out[zone] = 0.84 * local[zone] + 0.16 * out[zone]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _brighten_pure_ferrite_baseline(image_gray: np.ndarray) -> np.ndarray:
    arr = image_gray.astype(np.float32, copy=False)
    q01 = float(np.quantile(arr, 0.01))
    q05 = float(np.quantile(arr, 0.05))
    arr += max(0.0, 92.0 - q01)
    arr += max(0.0, 126.0 - q05) * 0.8
    bright = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return _lift_small_dark_defects(
        bright,
        threshold=44.0,
        max_pixels=max(24, int(bright.size // 32768)),
    )


def render_reflected_light(
    *,
    surface_state: SurfaceState,
    morphology_state: SpatialMorphologyState,
    synthesis_profile: SynthesisProfileV3,
    microscope_profile: dict[str, Any],
    seed: int,
    native_um_per_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height = surface_state.height_um.astype(np.float32)
    gx = np.gradient(height, axis=1)
    gy = np.gradient(height, axis=0)
    relief_shading = normalize01(0.5 + 14.0 * gx + 8.0 * gy)
    edge_strength = normalize01(np.hypot(gx, gy))
    illumination = 0.93 + 0.09 * low_frequency_field(
        height.shape, seed + 301, sigma=28.0
    )
    optical_mode = (
        str(microscope_profile.get("optical_mode", "brightfield")).strip().lower()
    )
    pure_iron_like = bool(
        dict(microscope_profile.get("pure_iron_baseline", {})).get("applied", False)
    )
    phase_edges = morphology_state.feature_maps.get("phase_boundaries")
    phase_edge_field = (
        phase_edges.astype(np.float32)
        if isinstance(phase_edges, np.ndarray)
        else np.zeros_like(height, dtype=np.float32)
    )

    img = surface_state.reflectance_base.astype(np.float32).copy()
    if pure_iron_like:
        img -= 0.10 * normalize01(surface_state.etch_depth_um)
        img -= 0.04 * surface_state.stain_map
        img -= 0.02 * surface_state.contamination_map
        img -= 0.03 * surface_state.pullout_map
        img += 0.02 * surface_state.smear_map
        img += 0.08 * relief_shading
        img = img * 0.78 + 0.22
    else:
        img -= 0.34 * normalize01(surface_state.etch_depth_um)
        img -= 0.15 * surface_state.stain_map
        img -= 0.08 * surface_state.contamination_map
        img -= 0.10 * surface_state.pullout_map
        img += 0.05 * surface_state.smear_map
        img += 0.18 * relief_shading
    mode_params = dict(microscope_profile.get("optical_mode_parameters") or {})
    if "dic_shear_axis_deg" in microscope_profile:
        mode_params["dic_shear_axis_deg"] = microscope_profile.get(
            "dic_shear_axis_deg", 35.0
        )
    if "dic_strength" in microscope_profile:
        mode_params["dic_strength"] = microscope_profile.get("dic_strength", 2.4)
    if "dic_amplitude" in microscope_profile:
        mode_params["dic_amplitude"] = microscope_profile.get("dic_amplitude", 0.24)
    if "phase_plate_type" in microscope_profile:
        mode_params["phase_plate_type"] = microscope_profile.get(
            "phase_plate_type", "positive"
        )
    optical_transfer_img, optical_meta = apply_optical_mode_transfer(
        image_gray=np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8),
        optical_mode=optical_mode,
        optical_mode_parameters=mode_params,
        height_um=height,
        edge_strength=edge_strength,
        phase_edge_field=phase_edge_field,
        orientation_rad=morphology_state.orientation_rad,
        phase_masks=morphology_state.phase_masks,
        pure_iron_like=pure_iron_like,
    )
    img = optical_transfer_img.astype(np.float32) / 255.0

    pearlite_mask = morphology_state.phase_masks.get("PEARLITE")
    lamella_applied = False
    if morphology_state.lamella_field is not None and isinstance(
        pearlite_mask, np.ndarray
    ):
        pearl_zone = pearlite_mask > 0
        if np.any(pearl_zone):
            lamella_applied = True
            lamella_integrity = 1.0 - np.clip(
                surface_state.damage_layer * 0.75 + surface_state.pullout_map * 0.25,
                0.0,
                1.0,
            )
            img[pearl_zone] += (
                (morphology_state.lamella_field[pearl_zone] - 0.5)
                * 0.24
                * lamella_integrity[pearl_zone]
            )

    bainite_modulation_applied = False
    bainite_sheaves = morphology_state.feature_maps.get("bainite_sheaves_binary")
    bainite_mask = morphology_state.phase_masks.get("BAINITE")
    if isinstance(bainite_sheaves, np.ndarray) and isinstance(bainite_mask, np.ndarray):
        bainite_zone = bainite_mask > 0
        if np.any(bainite_zone):
            bainite_modulation_applied = True
            sheaf_emphasis = normalize01(
                0.55 * bainite_sheaves.astype(np.float32) + 0.45 * edge_strength
            )
            img[bainite_zone] += (sheaf_emphasis[bainite_zone] - 0.5) * 0.16

    widmanstatten_modulation_applied = False
    widmanstatten_mask = morphology_state.feature_maps.get(
        "widmanstatten_sideplates_binary"
    )
    allotriomorphic_mask = morphology_state.feature_maps.get(
        "allotriomorphic_ferrite_binary"
    )
    ferrite_mask = morphology_state.phase_masks.get("FERRITE")
    if isinstance(widmanstatten_mask, np.ndarray) and isinstance(
        ferrite_mask, np.ndarray
    ):
        ferrite_zone = ferrite_mask > 0
        if np.any(ferrite_zone):
            widmanstatten_modulation_applied = True
            plate_contrast = normalize01(
                0.65 * widmanstatten_mask.astype(np.float32) + 0.35 * relief_shading
            )
            img[ferrite_zone] += (plate_contrast[ferrite_zone] - 0.5) * 0.12
    allotriomorphic_modulation_applied = False
    if isinstance(allotriomorphic_mask, np.ndarray) and isinstance(
        ferrite_mask, np.ndarray
    ):
        ferrite_zone = ferrite_mask > 0
        allotriomorphic_zone = allotriomorphic_mask > 0
        if np.any(ferrite_zone & allotriomorphic_zone):
            allotriomorphic_modulation_applied = True
            boundary_follow = normalize01(
                0.60 * allotriomorphic_mask.astype(np.float32)
                + 0.40 * normalize01(phase_edge_field)
            )
            active = ferrite_zone & allotriomorphic_zone
            img[active] += (boundary_follow[active] - 0.5) * 0.07

    martensite_mask = np.zeros(height.shape, dtype=bool)
    for name in (
        "MARTENSITE",
        "MARTENSITE_TETRAGONAL",
        "MARTENSITE_CUBIC",
        "TROOSTITE",
        "SORBITE",
        "BAINITE",
    ):
        mask = morphology_state.phase_masks.get(name)
        if isinstance(mask, np.ndarray):
            martensite_mask |= mask > 0
    packet_applied = False
    if morphology_state.packet_field is not None and np.any(martensite_mask):
        packet_applied = True
        img[martensite_mask] += (
            morphology_state.packet_field[martensite_mask] - 0.5
        ) * 0.14

    img_psf, psf_meta = apply_static_psf_profile(
        image_gray=np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8),
        height_um=surface_state.height_um,
        edge_strength=edge_strength,
        orientation_rad=morphology_state.orientation_rad,
        phase_edge_field=phase_edge_field,
        microscope_profile=microscope_profile,
    )
    img = img_psf.astype(np.float32) / 255.0
    img *= illumination
    contrast = float(max(0.6, min(2.0, synthesis_profile.contrast_target)))
    img = (img - 0.5) * contrast + 0.5
    img = np.clip(img, 0.0, 1.0)
    img255 = img * 255.0
    if optical_mode == "polarized" and bool(
        optical_meta.get("mode_parameters", {}).get("crossed_polars", False)
    ):
        image_gray = np.clip(img255, 0.0, 255.0).astype(np.uint8)
    else:
        if pure_iron_like and optical_mode == "brightfield":
            image_gray = np.clip(img255, 0.0, 255.0).astype(np.uint8)
        else:
            image_gray = rescale_to_u8(
                img255,
                lo=float(np.quantile(img255, 0.01)),
                hi=float(np.quantile(img255, 0.99)),
            )

    sharpness = float(max(0.5, min(2.4, synthesis_profile.boundary_sharpness)))
    if ndimage is not None and sharpness < 1.0:
        image_gray = np.clip(
            ndimage.gaussian_filter(
                image_gray.astype(np.float32), sigma=(1.0 - sharpness) * 1.1
            ),
            0,
            255,
        ).astype(np.uint8)
    image_gray = soft_unsharp(image_gray, amount=max(0.0, sharpness - 0.95) * 0.45)
    if pure_iron_like and optical_mode == "brightfield":
        image_gray = _brighten_pure_ferrite_baseline(image_gray)

    return image_gray, {
        "model": "reflected_light_explicit_surface_v1",
        "native_um_per_px": float(native_um_per_px),
        "contrast_target": float(synthesis_profile.contrast_target),
        "boundary_sharpness": float(synthesis_profile.boundary_sharpness),
        "optical_mode": str(optical_mode or "brightfield"),
        "mode_parameters": dict(optical_meta.get("mode_parameters", {})),
        "psf_profile": str(psf_meta.get("psf_profile", "standard")),
        "psf_strength": float(psf_meta.get("psf_strength", 0.0)),
        "sectioning_shear_deg": float(psf_meta.get("sectioning_shear_deg", 35.0)),
        "hybrid_balance": float(psf_meta.get("hybrid_balance", 0.5)),
        "psf_engineering_applied": bool(psf_meta.get("psf_engineering_applied", False)),
        "axial_profile_mode": str(psf_meta.get("axial_profile_mode", "standard")),
        "effective_dof_factor": float(psf_meta.get("effective_dof_factor", 1.0)),
        "sectioning_suppression_score": float(
            psf_meta.get("sectioning_suppression_score", 0.0)
        ),
        "lateral_shift_signature": float(psf_meta.get("lateral_shift_signature", 0.0)),
        "extended_dof_retention_score": float(
            psf_meta.get("extended_dof_retention_score", 0.0)
        ),
        "sectioning_directionality_score": float(
            psf_meta.get("sectioning_directionality_score", 0.0)
        ),
        "sectioning_active": bool(psf_meta.get("sectioning_active", False)),
        "lamella_modulation_applied": bool(lamella_applied),
        "bainite_modulation_applied": bool(bainite_modulation_applied),
        "widmanstatten_modulation_applied": bool(widmanstatten_modulation_applied),
        "allotriomorphic_modulation_applied": bool(allotriomorphic_modulation_applied),
        "packet_modulation_applied": bool(packet_applied),
        "contrast_mechanisms": [
            "specular_relief",
            "magnetic_particles"
            if optical_mode == "magnetic_etching"
            else ("scatter" if optical_mode == "darkfield" else "specular"),
            "phase_relief"
            if optical_mode == "phase_contrast"
            else (
                "interference_gradient"
                if optical_mode == "dic"
                else (
                    "magnetic_domains"
                    if optical_mode == "magnetic_etching"
                    else "standard"
                )
            ),
            "orientation" if optical_mode == "polarized" else "none",
        ],
        "crossed_polars": optical_meta.get("mode_parameters", {}).get(
            "crossed_polars", None
        ),
        "phase_plate_type": optical_meta.get("mode_parameters", {}).get(
            "phase_plate_type", None
        ),
        "uniformity_score": optical_meta.get("mode_parameters", {}).get(
            "uniformity_score", None
        ),
        "aperture_fraction": optical_meta.get("mode_parameters", {}).get(
            "aperture_fraction", None
        ),
        "scatter_pass_fraction": optical_meta.get("mode_parameters", {}).get(
            "scatter_pass_fraction", None
        ),
        "flat_field_suppression": optical_meta.get("mode_parameters", {}).get(
            "flat_field_suppression", None
        ),
        "anisotropy_coverage": optical_meta.get("mode_parameters", {}).get(
            "anisotropy_coverage", None
        ),
        "depolarization_score": optical_meta.get("mode_parameters", {}).get(
            "depolarization_score", None
        ),
        "reference_beam_fraction": optical_meta.get("mode_parameters", {}).get(
            "reference_beam_fraction", None
        ),
        "phase_object_gain": optical_meta.get("mode_parameters", {}).get(
            "phase_object_gain", None
        ),
        "nm_relief_proxy_p05_p95": optical_meta.get("mode_parameters", {}).get(
            "nm_relief_proxy_p05_p95", None
        ),
        "dic_shear_axis_deg": optical_meta.get("mode_parameters", {}).get(
            "dic_shear_axis_deg", None
        ),
        "dic_strength": optical_meta.get("mode_parameters", {}).get(
            "dic_strength", None
        ),
        "dic_amplitude": optical_meta.get("mode_parameters", {}).get(
            "dic_amplitude", None
        ),
        "interference_gradient": optical_meta.get("mode_parameters", {}).get(
            "interference_gradient", None
        ),
        "magnetic_field_active": optical_meta.get("mode_parameters", {}).get(
            "magnetic_field_active", None
        ),
        "ferromagnetic_fraction": optical_meta.get("mode_parameters", {}).get(
            "ferromagnetic_fraction", None
        ),
        "magnetic_signal_fraction": optical_meta.get("mode_parameters", {}).get(
            "magnetic_signal_fraction", None
        ),
        "domain_pattern_strength": optical_meta.get("mode_parameters", {}).get(
            "domain_pattern_strength", None
        ),
        "residual_attraction_level": optical_meta.get("mode_parameters", {}).get(
            "residual_attraction_level", None
        ),
        "magnetic_mode_limitations": optical_meta.get("mode_parameters", {}).get(
            "magnetic_mode_limitations", None
        ),
        "illumination_variation_mean": float(illumination.mean()),
        "relief_shading_mean": float(relief_shading.mean()),
        "rotation_signature_deg": psf_meta.get("rotation_signature_deg", None),
    }
