from __future__ import annotations

import math

import numpy as np

from core.generator_grains import generate_grain_structure
from core.metallography_v3.pure_ferrite_generator import (
    generate_pure_ferrite_micrograph,
)
from core.metallography_v3.realism_utils import (
    allocate_phase_masks,
    boundary_mask_from_labels,
    clamp,
    distance_to_mask,
    low_frequency_field,
    multiscale_noise,
    normalize01,
)
from core.metallography_v3.system_generators.base import normalize_phase_fractions

from .contracts import ContinuousTransformationState, SpatialMorphologyState

_MARTENSITIC_NAMES = {
    "MARTENSITE",
    "MARTENSITE_TETRAGONAL",
    "MARTENSITE_CUBIC",
    "TROOSTITE",
    "SORBITE",
    "BAINITE",
}


def _is_single_phase_ferrite_like(stage_l: str, phases: dict[str, float]) -> bool:
    ferrite = float(phases.get("FERRITE", 0.0) + phases.get("DELTA_FERRITE", 0.0))
    others = float(
        sum(
            float(v)
            for k, v in phases.items()
            if str(k).upper() not in {"FERRITE", "DELTA_FERRITE"}
        )
    )
    return bool(stage_l == "ferrite" and ferrite >= 0.95 and others <= 0.05)


def _field_from_angles(
    shape: tuple[int, int],
    angles: np.ndarray,
    phase_shift: np.ndarray,
    period_px: np.ndarray,
    curvature: np.ndarray,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(angles) + yy * np.sin(angles)
    wave = np.sin(
        (2.0 * math.pi / np.maximum(period_px, 1.1)) * (proj + curvature) + phase_shift
    )
    return normalize01(wave.astype(np.float32))


def _needle_field(
    *,
    shape: tuple[int, int],
    orientation: np.ndarray,
    seed: int,
    spacing_px: float,
    width_bias: float,
) -> np.ndarray:
    phase_shift = (
        np.random.default_rng(int(seed))
        .uniform(0.0, 2.0 * math.pi, size=shape)
        .astype(np.float32)
    )
    curvature = (
        (low_frequency_field(shape, seed + 5, sigma=max(3.0, spacing_px * 0.8)) - 0.5)
        * spacing_px
        * 0.4
    )
    wave = _field_from_angles(
        shape,
        orientation.astype(np.float32),
        phase_shift,
        np.full(shape, float(max(2.0, spacing_px)), dtype=np.float32),
        curvature.astype(np.float32),
    )
    return normalize01(np.clip((wave - (0.62 - 0.18 * width_bias)) * 2.8, 0.0, 1.0))


def _compress_label_count(labels: np.ndarray | None) -> int:
    if labels is None or not isinstance(labels, np.ndarray):
        return 0
    return int(np.count_nonzero(np.unique(labels)))


def build_spatial_morphology_state(
    *,
    size: tuple[int, int],
    seed: int,
    stage: str,
    phase_fractions: dict[str, float],
    transformation_state: ContinuousTransformationState,
    native_um_per_px: float,
) -> SpatialMorphologyState:
    rng = np.random.default_rng(int(seed))
    phases = normalize_phase_fractions(dict(phase_fractions))
    stage_l = str(stage or "").strip().lower()
    dominant_name = max(phases.items(), key=lambda item: float(item[1]))[0]

    if _is_single_phase_ferrite_like(stage_l, phases):
        mean_eq_d_px = clamp(
            float(transformation_state.prior_austenite_grain_size_um)
            / max(native_um_per_px, 1e-6)
            * 0.78,
            42.0,
            96.0,
        )
        ferrite_render = generate_pure_ferrite_micrograph(
            size=size,
            seed=seed + 11,
            mean_eq_d_px=float(mean_eq_d_px),
            size_sigma=0.22,
            relax_iter=1,
            boundary_width_px=2.2,
            boundary_depth=0.13,
            blur_sigma_px=0.6,
        )
        labels = np.asarray(ferrite_render["labels"], dtype=np.int32)
        boundaries = np.asarray(ferrite_render["boundary"], dtype=np.uint8) > 0
        orientation_rad = np.asarray(
            ferrite_render["orientation_rad"], dtype=np.float32
        )
        phase_masks = {"FERRITE": np.ones(size, dtype=np.uint8)}
        phase_label_map = np.ones(size, dtype=np.int32)
        boundary_class_map = np.zeros(size, dtype=np.uint8)
        boundary_class_map[boundaries] = 1
        feature_maps: dict[str, np.ndarray] = {
            "phase_boundaries": boundaries.astype(np.uint8),
            "prior_austenite_boundaries": boundaries.astype(np.uint8),
            "colony_boundaries": np.zeros(size, dtype=np.uint8),
            "allotriomorphic_ferrite_binary": boundaries.astype(np.uint8),
        }
        summary = {
            "resolved_stage": str(stage),
            "transformation_family": str(transformation_state.transformation_family),
            "ferrite_morphology_family": str(
                transformation_state.ferrite_morphology_family
            ),
            "bainite_morphology_family": str(
                transformation_state.bainite_morphology_family
            ),
            "martensite_morphology_family": str(
                transformation_state.martensite_morphology_family
            ),
            "pearlite_morphology_family": str(
                transformation_state.pearlite_morphology_family
            ),
            "phase_names": ["FERRITE"],
            "prior_austenite_grain_count": int(_compress_label_count(labels)),
            "colony_count": 0,
            "packet_count": 0,
            "prior_austenite_grain_size_um": float(
                transformation_state.prior_austenite_grain_size_um
            ),
            "colony_size_um_mean": float(transformation_state.colony_size_um_mean),
            "interlamellar_spacing_um_mean": float(
                transformation_state.interlamellar_spacing_um_mean
            ),
            "martensite_packet_size_um": float(
                transformation_state.martensite_packet_size_um
            ),
            "bainite_sheaf_length_um": float(
                transformation_state.bainite_sheaf_length_um
            ),
            "bainite_sheaf_thickness_um": float(
                transformation_state.bainite_sheaf_thickness_um
            ),
            "bainite_spacing_px": 0.0,
            "bainite_width_bias": 0.0,
            "bainite_thickness_px": 0.0,
            "bainite_density_target": 0.0,
            "native_um_per_px": float(native_um_per_px),
            "pure_ferrite_generator": dict(ferrite_render.get("metadata", {})),
        }
        return SpatialMorphologyState(
            phase_label_map=phase_label_map,
            phase_masks=phase_masks,
            pag_id_map=labels + 1,
            colony_id_map=np.zeros(size, dtype=np.int32),
            packet_id_map=None,
            orientation_rad=orientation_rad,
            lamella_field=None,
            packet_field=None,
            boundary_class_map=boundary_class_map,
            feature_maps=feature_maps,
            summary=summary,
        )

    pag_size_px = max(
        10.0,
        float(transformation_state.prior_austenite_grain_size_um)
        / max(native_um_per_px, 1e-6),
    )
    pag = generate_grain_structure(
        size=size,
        seed=seed + 11,
        mean_grain_size_px=pag_size_px,
        grain_size_jitter=0.30,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    pag_id_map = np.asarray(pag["labels"], dtype=np.int32)
    pag_boundaries = boundary_mask_from_labels(pag_id_map, width=2)
    boundary_proximity = normalize01(1.0 / (1.0 + distance_to_mask(pag_boundaries)))
    low = low_frequency_field(
        size=size, seed=seed + 17, sigma=max(8.0, pag_size_px * 0.36)
    )
    interior = normalize01((1.0 - boundary_proximity) * 0.70 + low * 0.30)
    packet_field_pref = normalize01(
        low_frequency_field(
            size=size, seed=seed + 19, sigma=max(6.0, pag_size_px * 0.24)
        )
        * 0.65
        + low * 0.35
    )
    retained_austenite_field = normalize01(
        low_frequency_field(
            size=size, seed=seed + 23, sigma=max(4.0, pag_size_px * 0.18)
        )
        * 0.72
        + boundary_proximity * 0.28
    )
    carbide_field = normalize01(
        multiscale_noise(size=size, seed=seed + 29, scales=((12.0, 0.60), (4.0, 0.40)))
        * 0.72
        + boundary_proximity * 0.28
    )
    pag_count = int(pag_id_map.max()) + 1
    pag_angles = rng.uniform(0.0, math.pi, size=pag_count).astype(np.float32)
    orientation_rad_base = pag_angles[pag_id_map]
    bainite_family = str(transformation_state.bainite_morphology_family)
    widmanstatten_factor = (
        0.30
        if str(transformation_state.ferrite_morphology_family) == "widmanstatten"
        else 0.14
    )
    bainite_orientation_noise = rng.uniform(
        -math.pi / 5.0, math.pi / 5.0, size=pag_count
    ).astype(np.float32)
    bainite_orientation_rad = (
        orientation_rad_base
        + widmanstatten_factor * bainite_orientation_noise[pag_id_map]
    ).astype(np.float32)
    widmanstatten_field = _needle_field(
        shape=size,
        orientation=orientation_rad_base,
        seed=seed + 35,
        spacing_px=max(5.0, pag_size_px * 0.42),
        width_bias=0.45,
    )
    bainite_spacing_px = max(
        4.0,
        float(transformation_state.bainite_sheaf_length_um)
        / max(native_um_per_px, 1e-6),
    )
    bainite_thickness_px = max(
        0.6,
        float(transformation_state.bainite_sheaf_thickness_um)
        / max(native_um_per_px, 1e-6),
    )
    bainite_density = float(clamp(transformation_state.bainite_sheaf_density, 0.0, 1.0))
    if bainite_family == "upper_bainite_sheaves":
        bainite_spacing_px *= 0.72
        bainite_width_bias = 0.78 + 0.08 * min(bainite_thickness_px / 3.0, 1.0)
    elif bainite_family == "lower_bainite_sheaves":
        bainite_spacing_px *= 0.46
        bainite_width_bias = 0.42 + 0.05 * min(bainite_thickness_px / 3.0, 1.0)
    else:
        bainite_spacing_px *= 0.55
        bainite_width_bias = 0.65 + 0.06 * min(bainite_thickness_px / 3.0, 1.0)
    bainite_sheaf_field_pref = _needle_field(
        shape=size,
        orientation=bainite_orientation_rad,
        seed=seed + 36,
        spacing_px=bainite_spacing_px,
        width_bias=bainite_width_bias,
    )

    proeutectoid_name = ""
    if stage_l == "alpha_pearlite" and float(phases.get("FERRITE", 0.0)) > 1e-6:
        proeutectoid_name = "FERRITE"
    elif stage_l == "pearlite_cementite" and float(phases.get("CEMENTITE", 0.0)) > 1e-6:
        proeutectoid_name = "CEMENTITE"

    ordered_fields: list[tuple[str, np.ndarray]] = []
    if proeutectoid_name:
        boundary_bias = float(transformation_state.proeutectoid_boundary_bias)
        if (
            proeutectoid_name == "FERRITE"
            and str(transformation_state.ferrite_morphology_family) == "widmanstatten"
        ):
            pro_field = normalize01(
                widmanstatten_field * 0.74 + boundary_proximity * 0.26
            )
        else:
            pro_field = normalize01(
                boundary_proximity * boundary_bias
                + low_frequency_field(
                    size=size, seed=seed + 31, sigma=max(5.0, pag_size_px * 0.20)
                )
                * (1.0 - boundary_bias)
            )
        ordered_fields.append((proeutectoid_name, pro_field))
    if float(phases.get("AUSTENITE", 0.0)) > 1e-6:
        ordered_fields.append(("AUSTENITE", retained_austenite_field))
    if float(phases.get("CEMENTITE", 0.0)) > 1e-6 and proeutectoid_name != "CEMENTITE":
        ordered_fields.append(("CEMENTITE", carbide_field))
    if float(phases.get("PEARLITE", 0.0)) > 1e-6:
        ordered_fields.append(("PEARLITE", interior))
    for name in sorted(phases):
        if name in _MARTENSITIC_NAMES:
            if name == "BAINITE":
                ordered_fields.append(
                    (
                        name,
                        normalize01(
                            bainite_sheaf_field_pref * 0.72 + packet_field_pref * 0.28
                        ),
                    )
                )
            else:
                ordered_fields.append((name, packet_field_pref))
    if float(phases.get("FERRITE", 0.0)) > 1e-6 and proeutectoid_name != "FERRITE":
        ferrite_field = interior
        if str(transformation_state.ferrite_morphology_family) == "widmanstatten":
            ferrite_field = normalize01(widmanstatten_field * 0.78 + interior * 0.22)
        ordered_fields.append(("FERRITE", ferrite_field))
    for name in sorted(phases):
        if name not in {item[0] for item in ordered_fields}:
            ordered_fields.append((name, low))

    phase_masks = allocate_phase_masks(
        size=size,
        phase_fractions=phases,
        ordered_fields=ordered_fields,
        remainder_name=str(dominant_name),
    )
    ordered_names = [
        name
        for name, _ in sorted(
            phases.items(), key=lambda item: float(item[1]), reverse=True
        )
    ]
    phase_label_map = np.zeros(size, dtype=np.int32)
    for idx, name in enumerate(ordered_names, 1):
        mask = phase_masks.get(name)
        if isinstance(mask, np.ndarray):
            phase_label_map[mask > 0] = idx

    colony_size_px = max(
        6.0,
        float(transformation_state.colony_size_um_mean) / max(native_um_per_px, 1e-6),
    )
    colony = generate_grain_structure(
        size=size,
        seed=seed + 37,
        mean_grain_size_px=colony_size_px,
        grain_size_jitter=0.32,
        elongation=1.15,
        orientation_deg=18.0,
        boundary_width_px=1,
        boundary_contrast=0.0,
    )
    colony_base = np.asarray(colony["labels"], dtype=np.int32)
    colony_id_map = pag_id_map * int(colony_base.max() + 1) + colony_base + 1
    pearlite_mask = phase_masks.get("PEARLITE", np.zeros(size, dtype=np.uint8)) > 0
    colony_id_map = np.where(pearlite_mask, colony_id_map, 0).astype(np.int32)

    colony_count = int(colony_base.max()) + 1
    colony_angles = rng.uniform(
        -math.pi / 4.0, math.pi / 4.0, size=colony_count
    ).astype(np.float32)
    orientation_rad = (
        orientation_rad_base + 0.34 * colony_angles[colony_base]
    ) % math.pi

    lamella_field: np.ndarray | None = None
    if np.any(pearlite_mask):
        spacing_px_mean = max(
            1.2,
            float(transformation_state.interlamellar_spacing_um_mean)
            / max(native_um_per_px, 1e-6),
        )
        spacing_jitter = float(
            transformation_state.interlamellar_spacing_um_std
            / max(transformation_state.interlamellar_spacing_um_mean, 1e-6)
        )
        jitter_field = (
            normalize01(
                multiscale_noise(
                    size=size, seed=seed + 41, scales=((14.0, 0.68), (5.0, 0.32))
                )
            )
            * 2.0
            - 1.0
        )
        local_spacing_px = np.clip(
            spacing_px_mean * (1.0 + 0.45 * spacing_jitter * jitter_field),
            max(1.1, spacing_px_mean * 0.55),
            spacing_px_mean * 1.85,
        )
        phase_shift = rng.uniform(0.0, 2.0 * math.pi, size=size).astype(np.float32)
        curvature = (
            (
                multiscale_noise(
                    size=size, seed=seed + 43, scales=((24.0, 0.70), (8.0, 0.30))
                )
                - 0.5
            )
            * spacing_px_mean
            * 0.85
        )
        lamella_field = _field_from_angles(
            size,
            orientation_rad.astype(np.float32),
            phase_shift,
            local_spacing_px.astype(np.float32),
            curvature.astype(np.float32),
        )
        lamella_field = np.where(pearlite_mask, lamella_field, 0.5).astype(np.float32)

    packet_id_map: np.ndarray | None = None
    packet_field: np.ndarray | None = None
    mart_mask_total = np.zeros(size, dtype=bool)
    for name in _MARTENSITIC_NAMES:
        if name in phase_masks:
            mart_mask_total |= phase_masks[name] > 0
    if np.any(mart_mask_total):
        packet_size_px = max(
            4.0,
            float(transformation_state.martensite_packet_size_um)
            / max(native_um_per_px, 1e-6),
        )
        packet = generate_grain_structure(
            size=size,
            seed=seed + 47,
            mean_grain_size_px=packet_size_px,
            grain_size_jitter=0.26,
            elongation=1.6,
            orientation_deg=32.0,
            boundary_width_px=1,
            boundary_contrast=0.0,
        )
        packet_base = np.asarray(packet["labels"], dtype=np.int32)
        packet_id_map = np.where(mart_mask_total, packet_base + 1, 0).astype(np.int32)
        packet_angles = rng.uniform(
            -math.pi / 2.0, math.pi / 2.0, size=int(packet_base.max()) + 1
        ).astype(np.float32)
        packet_orientation = packet_angles[packet_base]
        orientation_rad = np.where(
            mart_mask_total, packet_orientation, orientation_rad
        ).astype(np.float32)
        lath_factor = {
            "lath_dominant": 0.18,
            "mixed_lath_plate": 0.24,
            "plate_dominant": 0.36,
        }.get(str(transformation_state.martensite_morphology_family), 0.22)
        lath_period_px = np.clip(packet_size_px * lath_factor, 1.4, 12.0)
        phase_shift = rng.uniform(0.0, 2.0 * math.pi, size=size).astype(np.float32)
        curvature = (
            (
                low_frequency_field(
                    size=size, seed=seed + 53, sigma=max(3.0, packet_size_px * 0.50)
                )
                - 0.5
            )
            * lath_period_px
            * 0.65
        )
        packet_field = _field_from_angles(
            size,
            packet_orientation.astype(np.float32),
            phase_shift,
            np.full(size, float(lath_period_px), dtype=np.float32),
            curvature.astype(np.float32),
        )
        packet_field = np.where(mart_mask_total, packet_field, 0.5).astype(np.float32)

    colony_boundaries = (
        boundary_mask_from_labels(colony_id_map, width=1)
        if np.any(colony_id_map > 0)
        else np.zeros(size, dtype=bool)
    )
    packet_boundaries = (
        boundary_mask_from_labels(packet_id_map, width=1)
        if isinstance(packet_id_map, np.ndarray) and np.any(packet_id_map > 0)
        else np.zeros(size, dtype=bool)
    )
    phase_boundaries = boundary_mask_from_labels(phase_label_map, width=2)

    boundary_class_map = np.zeros(size, dtype=np.uint8)
    boundary_class_map[pag_boundaries] = 1
    if proeutectoid_name:
        boundary_class_map[
            phase_masks.get(proeutectoid_name, np.zeros(size, dtype=np.uint8)) > 0
        ] = 2
    boundary_class_map[colony_boundaries] = 3
    boundary_class_map[packet_boundaries] = 4

    feature_maps: dict[str, np.ndarray] = {
        "phase_boundaries": phase_boundaries.astype(np.uint8),
        "prior_austenite_boundaries": pag_boundaries.astype(np.uint8),
        "colony_boundaries": colony_boundaries.astype(np.uint8),
    }
    if lamella_field is not None:
        feature_maps["lamellae_binary"] = (
            (lamella_field > 0.58) & pearlite_mask
        ).astype(np.uint8)
    if packet_field is not None:
        feature_maps["martensite_laths_binary"] = (
            (np.abs(packet_field - 0.5) > 0.26) & mart_mask_total
        ).astype(np.uint8)
    bainite_mask = phase_masks.get("BAINITE")
    if isinstance(bainite_mask, np.ndarray) and np.any(bainite_mask > 0):
        bainite_threshold = float(
            clamp(
                0.74
                - 0.22 * bainite_density
                - 0.08 * min(bainite_thickness_px / 3.0, 1.0),
                0.42,
                0.82,
            )
        )
        feature_maps["bainite_sheaves_binary"] = (
            (bainite_sheaf_field_pref > bainite_threshold) & (bainite_mask > 0)
        ).astype(np.uint8)
        if bainite_family == "upper_bainite_sheaves":
            feature_maps["upper_bainite_sheaves_binary"] = feature_maps[
                "bainite_sheaves_binary"
            ]
        elif bainite_family == "lower_bainite_sheaves":
            feature_maps["lower_bainite_sheaves_binary"] = feature_maps[
                "bainite_sheaves_binary"
            ]
    ferrite_mask = phase_masks.get("FERRITE")
    if (
        isinstance(ferrite_mask, np.ndarray)
        and np.any(ferrite_mask > 0)
        and str(transformation_state.ferrite_morphology_family) == "widmanstatten"
    ):
        feature_maps["widmanstatten_sideplates_binary"] = (
            (widmanstatten_field > 0.55) & (ferrite_mask > 0)
        ).astype(np.uint8)
    elif isinstance(ferrite_mask, np.ndarray) and np.any(ferrite_mask > 0):
        feature_maps["allotriomorphic_ferrite_binary"] = (
            ((boundary_proximity > 0.44) | (pag_boundaries > 0)) & (ferrite_mask > 0)
        ).astype(np.uint8)
    if float(transformation_state.carbide_size_um) > 0.08:
        feature_maps["carbide_particles"] = (
            carbide_field > float(np.quantile(carbide_field, 0.88))
        ).astype(np.uint8)

    summary = {
        "resolved_stage": str(stage),
        "transformation_family": str(transformation_state.transformation_family),
        "ferrite_morphology_family": str(
            transformation_state.ferrite_morphology_family
        ),
        "bainite_morphology_family": str(
            transformation_state.bainite_morphology_family
        ),
        "martensite_morphology_family": str(
            transformation_state.martensite_morphology_family
        ),
        "pearlite_morphology_family": str(
            transformation_state.pearlite_morphology_family
        ),
        "phase_names": ordered_names,
        "prior_austenite_grain_count": int(_compress_label_count(pag_id_map)),
        "colony_count": int(_compress_label_count(colony_id_map)),
        "packet_count": int(_compress_label_count(packet_id_map))
        if isinstance(packet_id_map, np.ndarray)
        else 0,
        "prior_austenite_grain_size_um": float(
            transformation_state.prior_austenite_grain_size_um
        ),
        "colony_size_um_mean": float(transformation_state.colony_size_um_mean),
        "interlamellar_spacing_um_mean": float(
            transformation_state.interlamellar_spacing_um_mean
        ),
        "martensite_packet_size_um": float(
            transformation_state.martensite_packet_size_um
        ),
        "bainite_sheaf_length_um": float(transformation_state.bainite_sheaf_length_um),
        "bainite_sheaf_thickness_um": float(
            transformation_state.bainite_sheaf_thickness_um
        ),
        "bainite_spacing_px": float(bainite_spacing_px),
        "bainite_width_bias": float(bainite_width_bias),
        "bainite_thickness_px": float(bainite_thickness_px),
        "bainite_density_target": float(bainite_density),
        "native_um_per_px": float(native_um_per_px),
    }
    return SpatialMorphologyState(
        phase_label_map=phase_label_map,
        phase_masks=phase_masks,
        pag_id_map=pag_id_map,
        colony_id_map=colony_id_map,
        packet_id_map=packet_id_map,
        orientation_rad=orientation_rad.astype(np.float32),
        lamella_field=lamella_field,
        packet_field=packet_field,
        boundary_class_map=boundary_class_map,
        feature_maps=feature_maps,
        summary=summary,
    )
